#!/usr/bin/env python3
"""Fail-closed gate for a local Hugging Face release candidate.

An exit code of zero means the machine-checkable gates passed; it does not
replace legal, privacy, model-card, or remote-code human review.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - CLI dependency error
    raise SystemExit("release_gate.py requires PyYAML; install the project's core extra") from exc

from build_release import (
    ALLOWED_RELEASE_FILES,
    AUTO_MAP,
    BOUNDARY_MODE,
    BOUNDARY_MODE_CONFIG_KEY,
    FORBIDDEN_NAME_PARTS,
    FORBIDDEN_SUFFIXES,
    RAW_DATA_SUFFIXES,
    REQUIRED_RELEASE_FILES,
    output_artifact_fingerprint,
    validate_safetensors,
)
from scan_public_artifacts import load_canary_allowlist, scan_paths

CHECKSUM_PATTERN = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
PLACEHOLDER_PATTERN = re.compile(r"(?:\bTODO\b|\bTBD\b|<org>|<model|\{\{[^}]+\}\})", re.IGNORECASE)
UNSAFE_REMOTE_IMPORTS = frozenset(
    {
        "ctypes",
        "httpx",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
        "webbrowser",
    }
)
UNSAFE_REMOTE_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "open",
        "os.popen",
        "os.system",
        "subprocess.run",
        "subprocess.Popen",
    }
)
BLOCKING_SEVERITIES = frozenset({"critical", "high", "medium", "moderate", "unknown"})


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str


@dataclass
class GateResult:
    name: str
    passed: bool
    issues: list[GateIssue]
    details: dict[str, Any]


def _result(name: str, issues: Iterable[GateIssue], **details: Any) -> GateResult:
    materialized = list(issues)
    return GateResult(name=name, passed=not materialized, issues=materialized, details=details)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gate_file_set(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    if artifact.is_symlink() or not artifact.is_dir():
        return _result(
            "file_set",
            [GateIssue("RC_BLOCKED_INVALID_ARTIFACT", "artifact must be a real directory")],
        )
    files: set[str] = set()
    for path in sorted(artifact.rglob("*")):
        relative = path.relative_to(artifact).as_posix()
        if path.is_symlink():
            issues.append(GateIssue("RC_BLOCKED_SYMLINK", f"symlink is forbidden: {relative}"))
            continue
        if not path.is_file():
            continue
        files.add(relative)
        lowered = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(
                GateIssue("RC_BLOCKED_UNSAFE_WEIGHT", f"unsafe checkpoint format: {relative}")
            )
        if path.suffix.lower() in RAW_DATA_SUFFIXES:
            issues.append(
                GateIssue("RC_BLOCKED_RAW_DATA", f"raw/tabular data is forbidden: {relative}")
            )
        if any(fragment in lowered for fragment in FORBIDDEN_NAME_PARTS):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TRAINING_STATE", f"private/training state is forbidden: {relative}"
                )
            )

    missing = sorted(REQUIRED_RELEASE_FILES - files)
    extras = sorted(files - ALLOWED_RELEASE_FILES)
    if missing:
        issues.append(
            GateIssue("RC_BLOCKED_MISSING_FILES", f"missing required files: {', '.join(missing)}")
        )
    if extras:
        issues.append(
            GateIssue("RC_BLOCKED_UNREVIEWED_FILES", f"unreviewed extra files: {', '.join(extras)}")
        )
    weight_suffixes = FORBIDDEN_SUFFIXES | {".safetensors"}
    unsafe_weights = sorted(
        name
        for name in files
        if name != "model.safetensors" and Path(name).suffix.lower() in weight_suffixes
    )
    if unsafe_weights:
        issues.append(
            GateIssue(
                "RC_BLOCKED_WEIGHT_ALLOWLIST",
                f"model.safetensors must be the only weight artifact: {', '.join(unsafe_weights)}",
            )
        )
    if (artifact / "model.safetensors").is_file():
        try:
            validate_safetensors(artifact / "model.safetensors")
        except ValueError as exc:
            issues.append(GateIssue("RC_BLOCKED_INVALID_SAFETENSORS", str(exc)))
    return _result("file_set", issues, file_count=len(files))


def gate_checksums(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    checksum_path = artifact / "checksums.txt"
    if not checksum_path.is_file():
        return _result(
            "checksums",
            [GateIssue("RC_BLOCKED_CHECKSUMS_MISSING", "checksums.txt is missing")],
        )
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = CHECKSUM_PATTERN.fullmatch(line)
        if not match:
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_FORMAT", f"invalid checksums.txt line {line_number}")
            )
            continue
        expected, relative = match.groups()
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or relative == "checksums.txt":
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_PATH", f"unsafe checksum path on line {line_number}")
            )
            continue
        if relative in entries:
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_DUPLICATE", f"duplicate checksum entry: {relative}")
            )
            continue
        entries[relative] = expected
    actual_files = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "checksums.txt"
    }
    if set(entries) != actual_files:
        missing = sorted(actual_files - entries.keys())
        extra = sorted(entries.keys() - actual_files)
        issues.append(
            GateIssue(
                "RC_BLOCKED_CHECKSUM_COVERAGE",
                f"checksum coverage mismatch; missing={missing}, unknown={extra}",
            )
        )
    for relative in sorted(actual_files & entries.keys()):
        if _sha256_file(artifact / relative) != entries[relative]:
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_MISMATCH", f"checksum mismatch: {relative}")
            )
    return _result("checksums", issues, verified_files=len(actual_files & entries.keys()))


def gate_config_and_remote_code(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        config = _load_json(artifact / "config.json")
        training = _load_json(artifact / "training_manifest.json")
        tokenizer_config = _load_json(artifact / "tokenizer_config.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("remote_code", [GateIssue("RC_BLOCKED_CONFIG", str(exc))])
    if config.get("architectures") != ["Qwen3BiForTokenClassification"]:
        issues.append(GateIssue("RC_BLOCKED_ARCHITECTURE", "config architectures is not Qwen3Bi"))
    if config.get("auto_map") != AUTO_MAP:
        issues.append(GateIssue("RC_BLOCKED_AUTO_MAP", "config auto_map is missing or unexpected"))
    if config.get("model_type") != "qwen3_bi" or config.get("use_cache") is not False:
        issues.append(
            GateIssue(
                "RC_BLOCKED_CONFIG_INVARIANT", "qwen3_bi model_type/use_cache invariant failed"
            )
        )
    if config.get("pii_attention_mode") != "full" or config.get("pii_release_eligible") is not True:
        issues.append(
            GateIssue(
                "RC_BLOCKED_CHECKPOINT_ATTENTION_CONTRACT",
                "checkpoint does not explicitly declare release-eligible full attention",
            )
        )
    if training.get("attention_mode") != "full":
        issues.append(
            GateIssue(
                "RC_BLOCKED_ATTENTION_MODE",
                "published training_manifest attention_mode must be full",
            )
        )
    if tokenizer_config.get(BOUNDARY_MODE_CONFIG_KEY) != BOUNDARY_MODE:
        issues.append(
            GateIssue(
                "RC_BLOCKED_BOUNDARY_TOKENIZER",
                "tokenizer_config is missing the approved character-boundary mode",
            )
        )
    tokenizer_evidence = training.get("tokenizer")
    effective = (
        tokenizer_evidence.get("effective") if isinstance(tokenizer_evidence, dict) else None
    )
    if not isinstance(effective, dict) or effective.get("boundary_mode") != BOUNDARY_MODE:
        issues.append(
            GateIssue(
                "RC_BLOCKED_TRAINING_TOKENIZER_BINDING",
                "training_manifest is not bound to the packaged boundary tokenizer",
            )
        )

    for filename in ("configuration_qwen3_bi.py", "modeling_qwen3_bi.py"):
        path = artifact / filename
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            issues.append(GateIssue("RC_BLOCKED_REMOTE_CODE_PARSE", f"{filename}: {exc}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name.split(".")[0] for alias in node.names}
                if imports & UNSAFE_REMOTE_IMPORTS:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_REMOTE_CODE_IMPORT",
                            f"{filename} imports network/process module",
                        )
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in UNSAFE_REMOTE_IMPORTS:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_REMOTE_CODE_IMPORT",
                            f"{filename} imports network/process module",
                        )
                    )
            elif isinstance(node, ast.Call):
                called = _call_name(node.func)
                if called in UNSAFE_REMOTE_CALLS:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_REMOTE_CODE_CALL",
                            f"{filename} contains unsafe call {called}",
                        )
                    )
    return _result("remote_code", issues)


def gate_training_artifact_binding(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        training = _load_json(artifact / "training_manifest.json")
        expected = training.get("manifest_sha256")
        unsigned = dict(training)
        unsigned.pop("manifest_sha256", None)
        encoded = json.dumps(
            unsigned,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        actual_manifest_hash = hashlib.sha256(encoded.encode()).hexdigest()
        if (
            not isinstance(expected, str)
            or actual_manifest_hash != expected
            or training.get("status") != "completed"
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TRAINING_MANIFEST_HASH",
                    "training manifest completion/hash check failed",
                )
            )
        actual = output_artifact_fingerprint(artifact)
        if actual.get("weight_files") != ["model.safetensors"]:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_OUTPUT_WEIGHT_SET",
                    "release must be bound to exactly model.safetensors",
                )
            )
        if training.get("output_artifact") != actual:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_OUTPUT_ARTIFACT_BINDING",
                    "packaged model/config/tokenizer files do not match training manifest",
                )
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(GateIssue("RC_BLOCKED_OUTPUT_ARTIFACT_BINDING", str(exc)))
    return _result("training_artifact_binding", issues)


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _registry_sources(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry must contain a sources list")
    result: dict[str, dict[str, Any]] = {}
    for entry in sources:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("every source registry entry must have an id")
        if entry["id"] in result:
            raise ValueError(f"duplicate source registry id: {entry['id']}")
        result[entry["id"]] = entry
    return result


def _source_ids_from_manifest(manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if lowered in {"source_id", "base_source_id", "teacher_source_id"} and isinstance(
            value, str
        ):
            result.add(value)
        elif lowered in {"source_ids", "training_source_ids"} and isinstance(value, list):
            result.update(item for item in value if isinstance(item, str))
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    visit(manifest)
    return result


def _evaluation_only_references(document: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    def visit(value: Any, trail: str = "root") -> None:
        if isinstance(value, dict):
            if value.get("used_for_training") is False:
                return
            sample_count = value.get("sample_count", value.get("sampled_records"))
            if sample_count == 0:
                return
            for key, child in value.items():
                lowered = str(key).lower()
                child_trail = f"{trail}.{key}"
                if lowered in {"forbidden_training_pools", "excluded_pools"}:
                    continue
                if lowered in {"pool", "training_pool", "data_pool", "admitted_pool"}:
                    values = child if isinstance(child, list) else [child]
                    if any(str(item).lower() == "evaluation_only" for item in values):
                        failures.append(child_trail)
                if lowered.endswith(("_path", "_file", "_dir")) and isinstance(child, str):
                    if "evaluation_only" in child.lower():
                        failures.append(child_trail)
                visit(child, child_trail)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{trail}[{index}]")

    visit(document)
    return failures


def gate_provenance(artifact: Path, registry_path: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        registry = _registry_sources(_load_yaml(registry_path))
        training = _load_json(artifact / "training_manifest.json")
        data = _load_json(artifact / "data_provenance.json")
        teacher = _load_json(artifact / "teacher_provenance.json")
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return _result("provenance", [GateIssue("RC_BLOCKED_PROVENANCE_PARSE", str(exc))])

    for filename, document in (
        ("training_manifest.json", training),
        ("data_provenance.json", data),
        ("teacher_provenance.json", teacher),
    ):
        references = _evaluation_only_references(document)
        if references:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_EVALUATION_ONLY_TRAINING",
                    f"{filename} admits evaluation_only data at: {', '.join(references)}",
                )
            )

    training_source_ids = _source_ids_from_manifest(training)
    actual_provenance_ids: set[str] = set()
    data_sources = data.get("sources")
    if not isinstance(data_sources, list) or not data_sources:
        issues.append(
            GateIssue("RC_BLOCKED_DATA_PROVENANCE_EMPTY", "data provenance has no sampled sources")
        )
    else:
        for index, source in enumerate(data_sources):
            if not isinstance(source, dict):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_DATA_PROVENANCE_SCHEMA",
                        f"data sources[{index}] is not an object",
                    )
                )
                continue
            count = source.get("sample_count", source.get("sampled_records"))
            if not isinstance(count, int) or count <= 0:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_DATA_PROVENANCE_COUNT",
                        f"data sources[{index}] lacks a positive sampled count",
                    )
                )
            source_id = source.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_DATA_PROVENANCE_SOURCE_ID",
                        f"data sources[{index}] lacks source_id",
                    )
                )
            elif isinstance(count, int) and count > 0:
                actual_provenance_ids.add(source_id)

    teachers = teacher.get("teachers")
    if not isinstance(teachers, list):
        if teacher.get("teacher_used") is not False:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TEACHER_PROVENANCE_SCHEMA",
                    "teacher provenance needs a teachers list or teacher_used=false",
                )
            )
    else:
        for index, entry in enumerate(teachers):
            if not isinstance(entry, dict) or not isinstance(entry.get("used_for_training"), bool):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_TEACHER_PROVENANCE_SCHEMA",
                        f"teachers[{index}] must explicitly state used_for_training",
                    )
                )
                continue
            if entry["used_for_training"]:
                source_id = entry.get("source_id")
                if not isinstance(source_id, str) or not source_id:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_TEACHER_PROVENANCE_SOURCE_ID",
                            f"teachers[{index}] lacks source_id",
                        )
                    )
                else:
                    actual_provenance_ids.add(source_id)

    unmanifested = actual_provenance_ids - training_source_ids
    if unmanifested:
        issues.append(
            GateIssue(
                "RC_BLOCKED_PROVENANCE_INCONSISTENT",
                "sampled data/teacher source IDs are absent from training_manifest.json: "
                + ", ".join(sorted(unmanifested)),
            )
        )
    source_ids = training_source_ids | actual_provenance_ids
    if not source_ids:
        issues.append(
            GateIssue(
                "RC_BLOCKED_SOURCE_REGISTRY_EMPTY",
                "no actual training/base/teacher source IDs were recorded",
            )
        )

    for source_id in sorted(source_ids):
        source = registry.get(source_id)
        if source is None:
            issues.append(
                GateIssue("RC_BLOCKED_SOURCE_UNREGISTERED", f"unregistered source: {source_id}")
            )
            continue
        if source.get("public_weight_training_allowed") is not True:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SOURCE_NOT_LICENSED",
                    f"source is not approved for public-weight training: {source_id}",
                )
            )
        license_value = source.get("declared_license", source.get("license"))
        if not isinstance(license_value, str) or license_value.strip().lower() in {
            "",
            "contract_specific",
            "pending",
            "proprietary",
            "unknown",
        }:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SOURCE_LICENSE",
                    f"source lacks an explicit distributable license: {source_id}",
                )
            )
        revision = source.get("revision", source.get("model_revision"))
        if (
            not isinstance(revision, str)
            or not revision.strip()
            or "<" in revision
            or revision.strip().lower() in {"head", "latest", "main", "master"}
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SOURCE_REVISION", f"source lacks an immutable revision: {source_id}"
                )
            )
        review = str(source.get("review_status", "")).lower()
        if not review or any(
            word in review
            for word in (
                "incomplete",
                "no_public",
                "not_approved",
                "not_available",
                "pending",
                "quarantined",
                "requires",
                "unreviewed",
            )
        ):
            issues.append(
                GateIssue("RC_BLOCKED_SOURCE_REVIEW", f"source review is incomplete: {source_id}")
            )
        if source.get("forced_pool") == "evaluation_only":
            issues.append(
                GateIssue(
                    "RC_BLOCKED_EVALUATION_ONLY_SOURCE",
                    f"evaluation-only source was referenced by training provenance: {source_id}",
                )
            )
        if source.get("kind") == "api_teacher":
            terms_snapshot = source.get("terms_snapshot")
            if (
                not source.get("legal_review_id")
                or not isinstance(terms_snapshot, str)
                or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", terms_snapshot)
            ):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_API_TEACHER_TERMS",
                        f"API teacher lacks legal-review/terms evidence: {source_id}",
                    )
                )
    if source_ids and not any(
        registry.get(source_id, {}).get("kind") == "base_model" for source_id in source_ids
    ):
        issues.append(
            GateIssue(
                "RC_BLOCKED_BASE_PROVENANCE",
                "training provenance does not reference a registered base_model source",
            )
        )
    return _result("provenance", issues, referenced_source_ids=sorted(source_ids))


def _seed_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = report.get("seeds", report.get("runs", []))
    if not isinstance(candidates, list):
        return []
    runs: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, int):
            runs.append({"seed": candidate})
        elif isinstance(candidate, dict):
            runs.append(candidate)
    return runs


def gate_quality(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        report = _load_json(artifact / "evaluation_report.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("quality", [GateIssue("RC_BLOCKED_QUALITY_PARSE", str(exc))])
    runs = _seed_runs(report)
    unique_seeds = {run.get("seed") for run in runs if isinstance(run.get("seed"), int)}
    if len(unique_seeds) < 3:
        issues.append(
            GateIssue(
                "RC_BLOCKED_INSUFFICIENT_SEEDS",
                f"quality evidence has {len(unique_seeds)} unique seed(s); at least 3 are required",
            )
        )
    expected_metric_names: set[str] | None = None
    for run in runs:
        seed = run.get("seed")
        if not isinstance(seed, int):
            continue
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            issues.append(GateIssue("RC_BLOCKED_SEED_METRICS", f"seed {seed} has no metrics"))
        else:
            metric_names = set(metrics)
            if expected_metric_names is None:
                expected_metric_names = metric_names
            elif metric_names != expected_metric_names:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_SEED_METRIC_MISMATCH",
                        f"seed {seed} reports a different metric set",
                    )
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in metrics.values()
            ):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_SEED_METRIC_VALUE",
                        f"seed {seed} contains a non-finite/non-numeric metric",
                    )
                )
        if run.get("quality_gate_passed") is not True:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SEED_QUALITY",
                    f"seed {seed} did not explicitly pass its quality gate",
                )
            )
    quality_gate = report.get("quality_gate")
    if (
        not isinstance(quality_gate, dict)
        or str(quality_gate.get("status", "")).lower() != "passed"
    ):
        issues.append(
            GateIssue("RC_BLOCKED_QUALITY_EVIDENCE", "aggregate quality_gate.status is not passed")
        )
    else:
        criteria = quality_gate.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            issues.append(
                GateIssue("RC_BLOCKED_QUALITY_CRITERIA", "quality gate has no explicit criteria")
            )
        else:
            for index, criterion in enumerate(criteria):
                if not isinstance(criterion, dict) or criterion.get("passed") is not True:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_QUALITY_CRITERION",
                            f"quality criterion {index} did not explicitly pass",
                        )
                    )
                    continue
                name = criterion.get("name")
                value = criterion.get("value")
                threshold = criterion.get("threshold")
                operator = criterion.get("operator")
                numeric = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and not isinstance(threshold, bool)
                    and isinstance(threshold, (int, float))
                    and math.isfinite(threshold)
                )
                if not isinstance(name, str) or not name.strip() or not numeric:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_QUALITY_CRITERION_EVIDENCE",
                            f"quality criterion {index} lacks a name/value/threshold",
                        )
                    )
                    continue
                comparisons = {
                    ">": value > threshold,
                    ">=": value >= threshold,
                    "<": value < threshold,
                    "<=": value <= threshold,
                }
                if operator not in comparisons or not comparisons[operator]:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_QUALITY_CRITERION_COMPARISON",
                            f"quality criterion {index} does not satisfy its stated comparison",
                        )
                    )
    if str(report.get("release_decision", "")).lower() not in {"pass", "passed", "approved"}:
        issues.append(
            GateIssue("RC_BLOCKED_RELEASE_DECISION", "evaluation report does not approve release")
        )
    try:
        training_manifest = _load_json(artifact / "training_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(GateIssue("RC_BLOCKED_TRAINING_SEED", str(exc)))
    else:
        selected_seed = training_manifest.get("seed")
        if not isinstance(selected_seed, int) or selected_seed not in unique_seeds:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TRAINING_SEED",
                    "training manifest seed is absent from the three-seed evidence",
                )
            )

    try:
        model_index = _load_yaml(artifact / "model-index.yml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        issues.append(GateIssue("RC_BLOCKED_MODEL_INDEX", str(exc)))
    else:
        indexed_models = model_index.get("model-index")
        indexed_results = []
        if isinstance(indexed_models, list):
            for model in indexed_models:
                if isinstance(model, dict) and isinstance(model.get("results"), list):
                    indexed_results.extend(model["results"])
        indexed_metrics = [
            metric
            for result in indexed_results
            if isinstance(result, dict) and isinstance(result.get("metrics"), list)
            for metric in result["metrics"]
            if isinstance(metric, dict)
        ]
        if (
            not indexed_results
            or not indexed_metrics
            or not any(
                not isinstance(metric.get("value"), bool)
                and isinstance(metric.get("value"), (int, float))
                and math.isfinite(metric["value"])
                for metric in indexed_metrics
            )
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_MODEL_INDEX",
                    "model-index.yml has no result with a finite numeric metric",
                )
            )
    return _result("quality", issues, unique_seed_count=len(unique_seeds))


def gate_public_scan(artifact: Path, canary_allowlist: Path | None) -> GateResult:
    try:
        allowed = load_canary_allowlist(canary_allowlist)
        findings = scan_paths([artifact], allowed_canaries=allowed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("public_artifact_scan", [GateIssue("RC_BLOCKED_SCAN_ERROR", str(exc))])
    issues = [
        GateIssue(
            "RC_BLOCKED_SECRET_OR_PII",
            f"{finding.path}:{finding.line} {finding.kind} {finding.fingerprint[:19]}...",
        )
        for finding in findings
    ]
    return _result("public_artifact_scan", issues, finding_count=len(findings))


def gate_release_text(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    for filename in ("README.md", "SECURITY.md", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        path = artifact / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if PLACEHOLDER_PATTERN.search(text):
            issues.append(
                GateIssue("RC_BLOCKED_PLACEHOLDER", f"unresolved placeholder in {filename}")
            )
    security_path = artifact / "SECURITY.md"
    if security_path.is_file():
        lowered = security_path.read_text(encoding="utf-8").lower()
        if "release blocker" in lowered or "not yet configured" in lowered:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SECURITY_CONTACT",
                    "SECURITY.md says the private reporting route is not configured",
                )
            )
    third_party_path = artifact / "THIRD_PARTY_NOTICES.md"
    if third_party_path.is_file():
        lowered = third_party_path.read_text(encoding="utf-8").lower()
        generic_markers = (
            "not a legal conclusion",
            "planned sources",
            "must generate a source-specific",
        )
        if any(marker in lowered for marker in generic_markers):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_GENERIC_NOTICES",
                    "THIRD_PARTY_NOTICES.md is a repository template, "
                    "not release-specific evidence",
                )
            )
    return _result("release_text", issues)


def _normalized_vulnerabilities(scan: dict[str, Any]) -> list[dict[str, Any]]:
    findings = scan.get("findings")
    if isinstance(findings, list):
        return [item for item in findings if isinstance(item, dict)]
    dependencies = scan.get("dependencies")
    normalized: list[dict[str, Any]] = []
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                continue
            component = dependency.get("name")
            for vulnerability in dependency.get("vulns", []):
                if isinstance(vulnerability, dict):
                    normalized.append(
                        {
                            "id": vulnerability.get("id"),
                            "component": component,
                            "severity": vulnerability.get("severity", "unknown"),
                            "status": "open",
                        }
                    )
    return normalized


def _load_exceptions(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    document = _load_json(path)
    if document.get("schema_version") != 1 or not isinstance(document.get("exceptions"), list):
        raise ValueError("dependency exceptions must use schema_version 1 and contain exceptions")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(document["exceptions"]):
        if not isinstance(entry, dict):
            raise ValueError(f"dependency exception {index} must be an object")
        vulnerability_id = entry.get("vulnerability_id")
        component = entry.get("component")
        if not isinstance(vulnerability_id, str) or not isinstance(component, str):
            raise ValueError(
                f"dependency exception {index} must identify vulnerability_id and component"
            )
        if not isinstance(entry.get("approved_by"), str) or not entry["approved_by"].strip():
            raise ValueError(f"dependency exception {index} lacks approved_by")
        if (
            not isinstance(entry.get("justification"), str)
            or len(entry["justification"].strip()) < 20
        ):
            raise ValueError(f"dependency exception {index} needs a substantive justification")
        controls = entry.get("compensating_controls")
        if (
            not isinstance(controls, list)
            or not controls
            or any(not isinstance(control, str) or not control.strip() for control in controls)
        ):
            raise ValueError(f"dependency exception {index} needs compensating_controls")
        expires = entry.get("expires_on")
        try:
            expiration = date.fromisoformat(expires)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dependency exception {index} has invalid expires_on") from exc
        if expiration < datetime.now(timezone.utc).date():
            raise ValueError(f"dependency exception {index} expired on {expiration.isoformat()}")
        key = (vulnerability_id, component.lower())
        if key in result:
            raise ValueError(f"duplicate dependency exception for {vulnerability_id}/{component}")
        result[key] = entry
    return result


def gate_dependencies(
    scan_path: Path | None,
    exceptions_path: Path | None,
    sbom_path: Path,
) -> GateResult:
    if scan_path is None:
        return _result(
            "dependencies",
            [
                GateIssue(
                    "RC_BLOCKED_DEPENDENCY_SCAN_MISSING",
                    "a completed dependency vulnerability scan is required",
                )
            ],
        )
    issues: list[GateIssue] = []
    used_exceptions: list[str] = []
    try:
        scan = _load_json(scan_path)
        exceptions = _load_exceptions(exceptions_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("dependencies", [GateIssue("RC_BLOCKED_DEPENDENCY_EVIDENCE", str(exc))])
    if not isinstance(scan.get("scanner"), str) or not scan["scanner"].strip():
        issues.append(
            GateIssue("RC_BLOCKED_DEPENDENCY_SCANNER", "dependency scan does not name its scanner")
        )
    if not isinstance(scan.get("generated_at"), str):
        issues.append(
            GateIssue("RC_BLOCKED_DEPENDENCY_SCAN_TIME", "dependency scan lacks generated_at")
        )
    if scan.get("scan_complete") is not True:
        issues.append(
            GateIssue(
                "RC_BLOCKED_DEPENDENCY_SCAN_INCOMPLETE",
                "dependency scan is not explicitly complete",
            )
        )
    recorded_sbom = scan.get("sbom_sha256")
    if (
        not isinstance(recorded_sbom, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_sbom)
        or not sbom_path.is_file()
        or _sha256_file(sbom_path) != recorded_sbom
    ):
        issues.append(
            GateIssue(
                "RC_BLOCKED_DEPENDENCY_SBOM_MISMATCH",
                "dependency scan is not bound to the packaged SBOM SHA-256",
            )
        )
    for finding in _normalized_vulnerabilities(scan):
        vulnerability_id = finding.get("id")
        component = finding.get("component")
        severity = str(finding.get("severity", "unknown")).lower()
        status = str(finding.get("status", "open")).lower()
        if not isinstance(vulnerability_id, str) or not isinstance(component, str):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_DEPENDENCY_FINDING_SCHEMA", "dependency finding lacks id/component"
                )
            )
            continue
        if status in {"fixed", "not_affected", "false_positive"}:
            continue
        exception = exceptions.get((vulnerability_id, component.lower()))
        if exception is not None:
            used_exceptions.append(str(exception.get("id", f"{vulnerability_id}/{component}")))
            continue
        if severity in BLOCKING_SEVERITIES:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_DEPENDENCY_VULNERABILITY",
                    f"unresolved {severity} vulnerability {vulnerability_id} in {component}",
                )
            )
    return _result("dependencies", issues, used_exceptions=sorted(used_exceptions))


def gate_sbom(artifact: Path) -> GateResult:
    try:
        sbom = _load_json(artifact / "sbom.cdx.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("sbom", [GateIssue("RC_BLOCKED_SBOM_PARSE", str(exc))])
    issues: list[GateIssue] = []
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") not in {"1.5", "1.6"}:
        issues.append(GateIssue("RC_BLOCKED_SBOM_FORMAT", "SBOM must be CycloneDX 1.5 or 1.6"))
    if not isinstance(sbom.get("components"), list) or not sbom["components"]:
        issues.append(GateIssue("RC_BLOCKED_SBOM_EMPTY", "SBOM has no dependency components"))
    return _result("sbom", issues, component_count=len(sbom.get("components", [])))


def run_gates(args: argparse.Namespace) -> list[GateResult]:
    artifact = args.artifact.resolve()
    return [
        gate_file_set(artifact),
        gate_checksums(artifact),
        gate_config_and_remote_code(artifact),
        gate_training_artifact_binding(artifact),
        gate_provenance(artifact, args.source_registry),
        gate_quality(artifact),
        gate_public_scan(artifact, args.canary_allowlist),
        gate_release_text(artifact),
        gate_sbom(artifact),
        gate_dependencies(
            args.dependency_scan,
            args.dependency_exceptions,
            artifact / "sbom.cdx.json",
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--canary-allowlist", type=Path)
    parser.add_argument("--dependency-scan", type=Path)
    parser.add_argument("--dependency-exceptions", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        results = run_gates(args)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"release gate error: {exc}", file=sys.stderr)
        return 2
    blockers = sum(len(result.issues) for result in results)
    report = {
        "schema_version": 1,
        "status": "PASS" if blockers == 0 else "RC_BLOCKED",
        "blocker_count": blockers,
        "gates": [
            {
                "name": result.name,
                "passed": result.passed,
                "issues": [asdict(issue) for issue in result.issues],
                "details": result.details,
            }
            for result in results
        ],
        "notice": "Machine checks do not replace required legal, privacy, and human reviews.",
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"{report['status']}: {blockers} blocker(s)")
    for result in results:
        print(f"- {'PASS' if result.passed else 'BLOCKED'} {result.name}")
        for issue in result.issues:
            print(f"  {issue.code}: {issue.message}")
    return 0 if blockers == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
