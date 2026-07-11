#!/usr/bin/env python3
"""Generate a source-bound aggregate artifact for the release HTML report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

_EXPECTED_SEEDS = (13, 42, 97)
_EXPECTED_SUITES = ("synthetic_test", "pii_bench_formal", "pii_bench_chat")
_EXPECTED_GATE_NAMES = frozenset(
    {
        "file_set",
        "checksums",
        "remote_code",
        "training_artifact_binding",
        "provenance",
        "quality",
        "public_artifact_scan",
        "release_text",
        "sbom",
        "dependencies",
        "receipt_identity",
    }
)
_DIRECT_SOURCE_NAMES = frozenset(
    {
        "evaluation",
        "training",
        "data_provenance",
        "teacher_provenance",
        "dependency_scan",
        "release_gate",
        "sbom",
    }
)
_OPTIONAL_DIRECT_SOURCE_NAMES = frozenset({"dependency_exceptions"})
_SUITE_LABELS = {
    "synthetic_test": "Synthetic frozen test",
    "pii_bench_formal": "PII Bench formal",
    "pii_bench_chat": "PII Bench chat",
}
_PACKAGED_INPUT_FILES = {
    "evaluation": "evaluation_report.json",
    "training": "training_manifest.json",
    "data_provenance": "data_provenance.json",
    "teacher_provenance": "teacher_provenance.json",
    "sbom": "sbom.cdx.json",
}
_FORBIDDEN_KEYS = {"doc_id", "document_id", "entity_value", "raw_text", "text"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")


class ReleaseReportError(ValueError):
    """Raised when aggregate report evidence is incomplete, stale, or unsafe."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseReportError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        if _FORBIDDEN_KEYS & {str(key).lower() for key in value}:
            raise ReleaseReportError("aggregate evidence contains a record-level field")
        for item in value.values():
            _public_safe(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _public_safe(item)
    elif isinstance(value, str):
        if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ReleaseReportError("aggregate evidence contains an absolute path")
        if ".." in value.replace("\\", "/").split("/"):
            raise ReleaseReportError("aggregate evidence contains parent traversal")


def _load(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseReportError(f"{description} must be a regular file")
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseReportError(f"{description} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseReportError(f"{description} must be a JSON object")
    _public_safe(value)
    return value, _sha256_bytes(encoded)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseReportError(f"{field} must be an object")
    return value


def _mapping_list(value: object, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ReleaseReportError(f"{field} must be a list of objects")
    return list(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseReportError(f"{field} must be a non-empty string")
    return value


def _safe_id(value: object, *, field: str) -> str:
    result = _string(value, field=field)
    if _SAFE_ID.fullmatch(result) is None:
        raise ReleaseReportError(f"{field} must be a stable path-free identifier")
    return result


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReleaseReportError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseReportError(f"{field} must be an integer >= {minimum}")
    return value


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseReportError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ReleaseReportError(f"{field} must be between zero and one")
    return result


def _verify_self_hash(value: Mapping[str, Any], *, description: str) -> str:
    claimed = _sha256(value.get("manifest_sha256"), field=f"{description}.manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if _canonical_json_hash(unsigned) != claimed:
        raise ReleaseReportError(f"{description} self-hash verification failed")
    return claimed


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _count_text(value: int) -> str:
    return f"{value:,}"


def validate_package_dir(directory: Path) -> dict[str, Any]:
    """Verify current package checksums and return a path-free identity."""

    if directory.is_symlink() or not directory.is_dir():
        raise ReleaseReportError("artifact directory must be a real directory")
    checksums_path = directory / "checksums.txt"
    if checksums_path.is_symlink() or not checksums_path.is_file():
        raise ReleaseReportError("current artifact is missing regular checksums.txt")
    try:
        lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReleaseReportError("checksums.txt must be UTF-8") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ReleaseReportError(f"invalid checksums.txt line {line_number}")
        digest, relative = match.groups()
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative == "checksums.txt"
        ):
            raise ReleaseReportError(f"unsafe checksums.txt path on line {line_number}")
        if relative in entries:
            raise ReleaseReportError(f"duplicate checksums.txt path {relative!r}")
        entries[relative] = digest

    actual_paths: dict[str, Path] = {}
    for candidate in sorted(directory.rglob("*")):
        if candidate.is_symlink():
            raise ReleaseReportError("current artifact contains a symlink")
        if candidate.is_file() and candidate != checksums_path:
            actual_paths[candidate.relative_to(directory).as_posix()] = candidate
    if set(entries) != set(actual_paths):
        raise ReleaseReportError("checksums.txt does not cover the current artifact exactly")
    for relative, candidate in actual_paths.items():
        if _sha256_file(candidate) != entries[relative]:
            raise ReleaseReportError(f"current artifact checksum mismatch for {relative}")
    return {
        "checksums_sha256": _sha256_file(checksums_path),
        "file_count": len(actual_paths) + 1,
        "verified_file_count": len(entries),
        "files": dict(sorted(entries.items())),
    }


def _validate_source_hashes(value: Mapping[str, str]) -> dict[str, str]:
    names = set(value)
    if not set(_DIRECT_SOURCE_NAMES) <= names or names - set(_DIRECT_SOURCE_NAMES) not in (
        set(),
        set(_OPTIONAL_DIRECT_SOURCE_NAMES),
    ):
        raise ReleaseReportError("direct source SHA set is incomplete")
    return {
        name: _sha256(digest, field=f"source_file_sha256.{name}") for name, digest in value.items()
    }


def _validate_release_and_dependency_binding(
    *,
    dependency_scan: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    sbom: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    package_binding: Mapping[str, Any],
    source_registry_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dependency_scan.get("schema_version") != 1:
        raise ReleaseReportError("dependency scan schema is unsupported")
    scanner = _string(dependency_scan.get("scanner"), field="dependency_scan.scanner")
    generated_at = _string(
        dependency_scan.get("generated_at"), field="dependency_scan.generated_at"
    )
    if dependency_scan.get("scan_complete") is not True:
        raise ReleaseReportError("dependency scan is incomplete")
    dependency_count = _integer(
        dependency_scan.get("dependency_count"), field="dependency_scan.dependency_count"
    )
    findings = _mapping_list(dependency_scan.get("findings"), field="dependency_scan.findings")
    recorded_sbom = _sha256(dependency_scan.get("sbom_sha256"), field="dependency_scan.sbom_sha256")
    if recorded_sbom != source_hashes["sbom"]:
        raise ReleaseReportError("dependency scan is stale for the current packaged SBOM")
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") not in {"1.5", "1.6"}:
        raise ReleaseReportError("packaged SBOM is not supported CycloneDX")
    components = sbom.get("components")
    if not isinstance(components, list) or not components:
        raise ReleaseReportError("packaged SBOM has no components")

    files = _mapping(package_binding.get("files"), field="package_binding.files")
    for source_name, filename in _PACKAGED_INPUT_FILES.items():
        if files.get(filename) != source_hashes[source_name]:
            raise ReleaseReportError(f"current package contains stale or mismatched {filename}")
    checksums_sha256 = _sha256(
        package_binding.get("checksums_sha256"), field="package_binding.checksums_sha256"
    )
    file_count = _integer(
        package_binding.get("file_count"), field="package_binding.file_count", minimum=1
    )
    verified_file_count = _integer(
        package_binding.get("verified_file_count"),
        field="package_binding.verified_file_count",
        minimum=1,
    )
    if file_count != verified_file_count + 1:
        raise ReleaseReportError("package checksum count is internally inconsistent")

    release_gate_hash = _verify_self_hash(release_gate, description="release gate receipt")
    if (
        release_gate.get("schema_version") != 2
        or release_gate.get("status") != "PASS"
        or release_gate.get("blocker_count") != 0
    ):
        raise ReleaseReportError("machine release gate did not pass")
    identity = _mapping(
        release_gate.get("artifact_identity"), field="release_gate.artifact_identity"
    )
    if set(identity) != {
        "checksums_sha256",
        "package_file_count",
        "sbom_sha256",
        "source_registry_sha256",
        "dependency_scan_sha256",
        "dependency_exceptions_sha256",
    }:
        raise ReleaseReportError("release gate artifact identity fields are incomplete")
    if (
        identity.get("checksums_sha256") != checksums_sha256
        or identity.get("package_file_count") != file_count
        or identity.get("sbom_sha256") != recorded_sbom
        or identity.get("source_registry_sha256") != source_registry_sha256
        or identity.get("dependency_scan_sha256") != source_hashes["dependency_scan"]
    ):
        raise ReleaseReportError("release gate receipt is stale for the current package inputs")
    exceptions_sha256 = identity.get("dependency_exceptions_sha256")
    if exceptions_sha256 is None:
        if "dependency_exceptions" in source_hashes:
            raise ReleaseReportError("dependency exceptions were supplied but not gate-bound")
    elif _sha256(
        exceptions_sha256, field="artifact_identity.dependency_exceptions_sha256"
    ) != source_hashes.get("dependency_exceptions"):
        raise ReleaseReportError("release gate receipt is stale for dependency exceptions")
    raw_gates = _mapping_list(release_gate.get("gates"), field="release_gate.gates")
    gates: dict[str, Mapping[str, Any]] = {}
    gate_rows: list[dict[str, Any]] = []
    for index, gate in enumerate(raw_gates):
        name = _string(gate.get("name"), field=f"release_gate.gates[{index}].name")
        if name in gates:
            raise ReleaseReportError(f"release gate repeats {name!r}")
        issues = gate.get("issues")
        if gate.get("passed") is not True or issues != []:
            raise ReleaseReportError(f"release gate {name!r} is not a clean PASS")
        _mapping(gate.get("details"), field=f"release_gate.{name}.details")
        gates[name] = gate
        gate_rows.append({"gate": name, "status": "PASS", "issue_count": 0})
    if set(gates) != set(_EXPECTED_GATE_NAMES):
        raise ReleaseReportError("release gate set is incomplete or unexpected")

    file_set_details = _mapping(gates["file_set"].get("details"), field="file_set.details")
    checksum_details = _mapping(gates["checksums"].get("details"), field="checksums.details")
    sbom_details = _mapping(gates["sbom"].get("details"), field="sbom.details")
    if file_set_details.get("file_count") != file_count:
        raise ReleaseReportError("release gate is stale for the current package file count")
    if checksum_details.get("verified_files") != verified_file_count:
        raise ReleaseReportError("release gate is stale for current checksums coverage")
    if sbom_details.get("component_count") != len(components):
        raise ReleaseReportError("release gate is stale for the current SBOM component count")
    dependency_details = _mapping(
        gates["dependencies"].get("details"), field="dependencies.details"
    )
    used_exceptions = dependency_details.get("used_exceptions")
    if not isinstance(used_exceptions, list) or any(
        not isinstance(item, str) for item in used_exceptions
    ):
        raise ReleaseReportError("dependency gate exception detail is malformed")

    return gate_rows, {
        "checksums_sha256": checksums_sha256,
        "dependency_count": dependency_count,
        "finding_count": len(findings),
        "generated_at": generated_at,
        "scanner": scanner,
        "sbom_component_count": len(components),
        "sbom_sha256": recorded_sbom,
        "used_exception_count": len(used_exceptions),
        "release_gate_manifest_sha256": release_gate_hash,
    }


def _metric_rows(
    evaluation: Mapping[str, Any],
    *,
    training_manifest_sha256: str,
    training_file_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    frozen = _mapping(evaluation.get("frozen_system_evaluation"), field="frozen_system_evaluation")
    _verify_self_hash(frozen, description="frozen system evaluation")
    if tuple(frozen.get("suite_order", ())) != _EXPECTED_SUITES:
        raise ReleaseReportError("frozen suite order is not the release contract")
    suites = _mapping(frozen.get("suites"), field="frozen_system_evaluation.suites")
    if set(suites) != set(_EXPECTED_SUITES):
        raise ReleaseReportError("frozen suite set is incomplete")
    identity = _mapping(frozen.get("system_identity"), field="frozen.system_identity")
    bound_training = _mapping(identity.get("training"), field="system_identity.training")
    if (
        bound_training.get("manifest_sha256") != training_manifest_sha256
        or bound_training.get("manifest_file_sha256") != training_file_sha256
    ):
        raise ReleaseReportError("frozen evaluation/training binding failed")
    if identity.get("selected_seed") != evaluation.get("selected_seed"):
        raise ReleaseReportError("frozen evaluation selected seed is inconsistent")
    if identity.get("attention_mode") != "full":
        raise ReleaseReportError("frozen evaluation is not a Full-attention system")

    wide: list[dict[str, Any]] = []
    long: list[dict[str, Any]] = []
    totals = {"documents": 0, "gold": 0, "predicted": 0, "tp": 0, "fp": 0, "fn": 0}
    bootstrap_contract: tuple[int, float, str] | None = None
    for suite in _EXPECTED_SUITES:
        suite_value = _mapping(suites.get(suite), field=f"suites.{suite}")
        metrics = _mapping(suite_value.get("metrics"), field=f"suites.{suite}.metrics")
        strict = _mapping(metrics.get("strict_micro"), field=f"suites.{suite}.strict_micro")
        span_count = _mapping(metrics.get("span_count"), field=f"suites.{suite}.span_count")
        documents = _integer(
            metrics.get("document_count"), field=f"{suite}.document_count", minimum=1
        )
        gold = _integer(span_count.get("gold"), field=f"{suite}.span_count.gold")
        predicted = _integer(span_count.get("predicted"), field=f"{suite}.span_count.predicted")
        tp = _integer(strict.get("tp"), field=f"{suite}.strict_micro.tp")
        fp = _integer(strict.get("fp"), field=f"{suite}.strict_micro.fp")
        fn = _integer(strict.get("fn"), field=f"{suite}.strict_micro.fn")
        if predicted != tp + fp or gold != tp + fn:
            raise ReleaseReportError(f"{suite} strict counts disagree with span counts")
        precision = _probability(strict.get("precision"), field=f"{suite}.precision")
        recall = _probability(strict.get("recall"), field=f"{suite}.recall")
        strict_f1 = _probability(strict.get("f1"), field=f"{suite}.strict_f1")
        if not (
            _close(precision, _ratio(tp, tp + fp))
            and _close(recall, _ratio(tp, tp + fn))
            and _close(strict_f1, _f1(precision, recall))
        ):
            raise ReleaseReportError(f"{suite} strict metrics do not recompute from counts")

        bootstrap = _mapping(metrics.get("bootstrap"), field=f"suites.{suite}.bootstrap")
        samples = _integer(bootstrap.get("samples"), field=f"{suite}.bootstrap.samples", minimum=1)
        confidence = _probability(
            bootstrap.get("confidence"), field=f"{suite}.bootstrap.confidence"
        )
        method = _string(bootstrap.get("method"), field=f"{suite}.bootstrap.method")
        current_contract = (samples, confidence, method)
        if bootstrap_contract is None:
            bootstrap_contract = current_contract
        elif current_contract != bootstrap_contract:
            raise ReleaseReportError("frozen suite bootstrap contracts differ")
        intervals = _mapping(bootstrap.get("intervals"), field=f"suites.{suite}.intervals")
        strict_interval = _mapping(
            intervals.get("strict_micro_f1"), field=f"suites.{suite}.strict_micro_f1_ci"
        )
        lower = _probability(strict_interval.get("lower"), field=f"{suite}.strict_f1_ci_lower")
        point = _probability(strict_interval.get("point"), field=f"{suite}.strict_f1_ci_point")
        upper = _probability(strict_interval.get("upper"), field=f"{suite}.strict_f1_ci_upper")
        if not (lower <= point <= upper and _close(point, strict_f1)):
            raise ReleaseReportError(f"{suite} strict F1 interval is inconsistent")

        pii_free = _mapping(metrics.get("pii_free"), field=f"suites.{suite}.pii_free")
        pii_free_documents = _integer(
            pii_free.get("documents"), field=f"{suite}.pii_free.documents"
        )
        false_positive_documents = _integer(
            pii_free.get("false_positive_documents"),
            field=f"{suite}.pii_free.false_positive_documents",
        )
        raw_fpr = pii_free.get("false_positive_rate")
        if pii_free_documents == 0:
            if false_positive_documents != 0 or raw_fpr is not None:
                raise ReleaseReportError(f"{suite} PII-free metric must be unestimated")
            pii_free_fpr: float | None = None
        else:
            pii_free_fpr = _probability(raw_fpr, field=f"{suite}.pii_free.false_positive_rate")
            if false_positive_documents > pii_free_documents or not _close(
                pii_free_fpr, false_positive_documents / pii_free_documents
            ):
                raise ReleaseReportError(f"{suite} PII-free FPR does not recompute")

        row = {
            "suite_id": suite,
            "suite": _SUITE_LABELS[suite],
            "documents": documents,
            "gold_spans": gold,
            "predicted_spans": predicted,
            "precision": precision,
            "recall": recall,
            "strict_f1": strict_f1,
            "strict_f1_ci_lower": lower,
            "strict_f1_ci_upper": upper,
            "strict_macro_f1": _probability(
                metrics.get("strict_macro_f1"), field=f"{suite}.strict_macro_f1"
            ),
            "pii_free_documents": pii_free_documents,
            "false_positive_documents": false_positive_documents,
            "pii_free_fpr": pii_free_fpr,
        }
        wide.append(row)
        for metric, label in (
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("strict_f1", "Strict F1"),
        ):
            long.append(
                {
                    "suite": row["suite"],
                    "metric": label,
                    "value": row[metric],
                    "documents": documents,
                    "gold_spans": gold,
                    "predicted_spans": predicted,
                    "pii_free_documents": pii_free_documents,
                    "false_positive_documents": false_positive_documents,
                }
            )
        totals["documents"] += documents
        totals["gold"] += gold
        totals["predicted"] += predicted
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    aggregate = _mapping(frozen.get("aggregate"), field="frozen.aggregate")
    if (
        aggregate.get("document_count") != totals["documents"]
        or aggregate.get("gold_span_count") != totals["gold"]
        or aggregate.get("predicted_span_count") != totals["predicted"]
    ):
        raise ReleaseReportError("frozen pooled counts disagree with suite totals")
    pooled = _mapping(aggregate.get("strict_micro_pooled"), field="strict_micro_pooled")
    for name in ("tp", "fp", "fn"):
        if pooled.get(name) != totals[name]:
            raise ReleaseReportError("pooled strict counts disagree with suite totals")
    pooled_precision = _probability(pooled.get("precision"), field="pooled.precision")
    pooled_recall = _probability(pooled.get("recall"), field="pooled.recall")
    pooled_f1 = _probability(pooled.get("f1"), field="pooled.f1")
    if not (
        _close(pooled_precision, _ratio(totals["tp"], totals["tp"] + totals["fp"]))
        and _close(pooled_recall, _ratio(totals["tp"], totals["tp"] + totals["fn"]))
        and _close(pooled_f1, _f1(pooled_precision, pooled_recall))
    ):
        raise ReleaseReportError("pooled strict metrics do not recompute")
    pooled_row = {
        "metric": "Pooled strict span",
        "documents": totals["documents"],
        "precision": pooled_precision,
        "recall": pooled_recall,
        "f1": pooled_f1,
    }
    if bootstrap_contract is None:  # pragma: no cover - suites are required above
        raise ReleaseReportError("bootstrap evidence is absent")
    bootstrap_info = {
        "samples": bootstrap_contract[0],
        "confidence": bootstrap_contract[1],
        "method": bootstrap_contract[2],
    }
    return wide, long, pooled_row, bootstrap_info


def _seed_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = _mapping(evaluation.get("quality_gate"), field="evaluation.quality_gate")
    if quality.get("status") != "passed":
        raise ReleaseReportError("aggregate validation quality gate did not pass")
    runs = _mapping_list(evaluation.get("seeds"), field="evaluation.seeds")
    if tuple(run.get("seed") for run in runs) != _EXPECTED_SEEDS:
        raise ReleaseReportError("three-seed validation evidence is incomplete")
    rows: list[dict[str, Any]] = []
    for run in runs:
        seed = _integer(run.get("seed"), field="seed", minimum=1)
        if run.get("quality_gate_passed") is not True:
            raise ReleaseReportError(f"validation seed {seed} did not pass")
        metrics = _mapping(run.get("metrics"), field=f"seed {seed}.metrics")
        rows.append(
            {
                "seed": str(seed),
                "strict_micro_f1": _probability(
                    metrics.get("strict_micro_f1"), field=f"seed {seed}.strict_micro_f1"
                ),
                "strict_macro_f1": _probability(
                    metrics.get("strict_macro_f1"), field=f"seed {seed}.strict_macro_f1"
                ),
                "tier0_min_recall": _probability(
                    metrics.get("tier0_min_label_recall"), field=f"seed {seed}.tier0 recall"
                ),
                "tier1_min_recall": _probability(
                    metrics.get("tier1_min_label_recall"), field=f"seed {seed}.tier1 recall"
                ),
                "gate": "PASS",
            }
        )
    return rows


def _validate_training_and_provenance(
    *,
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    teacher_provenance: Mapping[str, Any],
    training_file_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    training_hash = _verify_self_hash(training, description="training manifest")
    selected_seed = _integer(evaluation.get("selected_seed"), field="evaluation.selected_seed")
    if (
        training.get("schema_version") != 4
        or training.get("status") != "completed"
        or training.get("seed") != selected_seed
        or training.get("attention_mode") != "full"
    ):
        raise ReleaseReportError("training identity is not completed selected-seed Full schema4")
    training_ids_raw = training.get("training_source_ids")
    if not isinstance(training_ids_raw, list) or any(
        not isinstance(item, str) for item in training_ids_raw
    ):
        raise ReleaseReportError("training source IDs are malformed")
    training_ids = set(training_ids_raw)
    base_source_id = _safe_id(training.get("base_source_id"), field="training.base_source_id")
    if base_source_id not in training_ids:
        raise ReleaseReportError("base source is absent from training source lineage")

    data_hash = _verify_self_hash(data_provenance, description="data provenance")
    teacher_hash = _verify_self_hash(teacher_provenance, description="teacher provenance")
    del data_hash, teacher_hash
    if data_provenance.get("artifact_type") != "release_data_provenance":
        raise ReleaseReportError("unexpected data provenance artifact type")
    if teacher_provenance.get("artifact_type") != "release_teacher_provenance":
        raise ReleaseReportError("unexpected teacher provenance artifact type")
    for name, provenance in (
        ("data", data_provenance),
        ("teacher", teacher_provenance),
    ):
        if (
            provenance.get("training_manifest_sha256") != training_hash
            or provenance.get("training_manifest_file_sha256") != training_file_sha256
        ):
            raise ReleaseReportError(f"{name} provenance/training binding failed")
    data_registry = _mapping(data_provenance.get("source_registry"), field="data.source_registry")
    teacher_registry = _mapping(
        teacher_provenance.get("source_registry"), field="teacher.source_registry"
    )
    if data_registry != teacher_registry:
        raise ReleaseReportError("data and teacher provenance use different source registries")
    _sha256(data_registry.get("sha256"), field="source_registry.sha256")

    datasets = _mapping(training.get("datasets"), field="training.datasets")
    train_summary = _mapping(
        _mapping(datasets.get("train"), field="training.datasets.train").get("summary"),
        field="training.datasets.train.summary",
    )
    validation_summary = _mapping(
        _mapping(datasets.get("validation"), field="training.datasets.validation").get("summary"),
        field="training.datasets.validation.summary",
    )
    train_documents = _integer(
        train_summary.get("document_count"), field="training.train.document_count", minimum=1
    )
    validation_documents = _integer(
        validation_summary.get("document_count"),
        field="training.validation.document_count",
        minimum=1,
    )

    data_rows: list[dict[str, Any]] = []
    gradient_total = 0
    validation_total = 0
    frozen_test_use_total = 0
    sample_total = 0
    for index, source in enumerate(
        _mapping_list(data_provenance.get("sources"), field="data_provenance.sources")
    ):
        prefix = f"data_provenance.sources[{index}]"
        source_id = _safe_id(source.get("source_id"), field=f"{prefix}.source_id")
        role = _string(source.get("source_kind"), field=f"{prefix}.source_kind")
        license_name = _string(source.get("declared_license"), field=f"{prefix}.declared_license")
        if source.get("public_weight_training_allowed") is not True:
            raise ReleaseReportError(f"{source_id} is not admitted for public-weight training")
        if (
            source.get("synthetic") is not True
            or source.get("contains_real_personal_data") is not False
        ):
            raise ReleaseReportError(f"{source_id} does not support the synthetic-only claim")
        sample_count = _integer(source.get("sample_count"), field=f"{prefix}.sample_count")
        usage = _mapping(source.get("usage_counts"), field=f"{prefix}.usage_counts")
        gradient = _integer(usage.get("gradient_training"), field=f"{prefix}.gradient_training")
        validation = _integer(
            usage.get("validation_early_stopping_and_calibration"),
            field=f"{prefix}.validation_usage",
        )
        frozen_use = _integer(
            usage.get("frozen_test_training_use"), field=f"{prefix}.frozen_test_training_use"
        )
        if sample_count != gradient + validation or frozen_use != 0:
            raise ReleaseReportError(f"{source_id} usage counts are inconsistent")
        if source_id not in training_ids:
            raise ReleaseReportError(f"{source_id} is absent from training source lineage")
        data_rows.append(
            {
                "source_id": source_id,
                "role": role,
                "sample_count": sample_count,
                "license": license_name,
                "public_weight_training_allowed": True,
            }
        )
        gradient_total += gradient
        validation_total += validation
        frozen_test_use_total += frozen_use
        sample_total += sample_count
    if not data_rows:
        raise ReleaseReportError("release data provenance is empty")
    if (
        data_provenance.get("total_sample_count") != sample_total
        or gradient_total != train_documents
        or validation_total != validation_documents
    ):
        raise ReleaseReportError("data provenance counts disagree with training summaries")

    if (
        teacher_provenance.get("teacher_used") is not True
        or teacher_provenance.get("external_api_used") is not False
        or teacher_provenance.get("external_api_teacher_used") is not False
    ):
        raise ReleaseReportError(
            "teacher provenance does not support the local reviewed-teacher claim"
        )
    teacher_rows: list[dict[str, Any]] = []
    reviewed_total = 0
    accepted_total = 0
    rejected_total = 0
    for index, teacher in enumerate(
        _mapping_list(teacher_provenance.get("teachers"), field="teacher_provenance.teachers")
    ):
        prefix = f"teacher_provenance.teachers[{index}]"
        source_id = _safe_id(teacher.get("source_id"), field=f"{prefix}.source_id")
        role = _string(teacher.get("role"), field=f"{prefix}.role")
        license_name = _string(teacher.get("declared_license"), field=f"{prefix}.declared_license")
        reviewed = _integer(
            teacher.get("reviewed_candidate_count"), field=f"{prefix}.reviewed_candidate_count"
        )
        accepted = _integer(
            teacher.get("accepted_candidate_count"), field=f"{prefix}.accepted_candidate_count"
        )
        rejected = _integer(
            teacher.get("rejected_candidate_count"), field=f"{prefix}.rejected_candidate_count"
        )
        if reviewed != accepted + rejected:
            raise ReleaseReportError(f"{source_id} review counts are inconsistent")
        required_flags = {
            "used_for_training": True,
            "accepted_placeholder_templates_only": True,
            "raw_outputs_used_for_training": False,
            "span_teacher": False,
            "pseudo_label_teacher": False,
            "external_api": False,
        }
        if any(teacher.get(field) is not expected for field, expected in required_flags.items()):
            raise ReleaseReportError(f"{source_id} teacher safety role is inconsistent")
        if source_id not in training_ids:
            raise ReleaseReportError(f"{source_id} is absent from training source lineage")
        teacher_rows.append(
            {
                "source_id": source_id,
                "role": role,
                "sample_count": accepted,
                "reviewed_count": reviewed,
                "rejected_count": rejected,
                "license": license_name,
                "used_for_training": True,
            }
        )
        reviewed_total += reviewed
        accepted_total += accepted
        rejected_total += rejected
    if not teacher_rows:
        raise ReleaseReportError("release teacher provenance is empty")

    training_rows = [{"source_id": base_source_id, "manifest_field": "base_source_id"}]
    stats = {
        "accepted_candidates": accepted_total,
        "base_source_id": base_source_id,
        "data_source_count": len(data_rows),
        "frozen_test_training_use": frozen_test_use_total,
        "gradient_documents": gradient_total,
        "rejected_candidates": rejected_total,
        "reviewed_candidates": reviewed_total,
        "teacher_source_count": len(teacher_rows),
        "validation_documents": validation_total,
        "source_registry_sha256": data_registry["sha256"],
        "selected_seed": selected_seed,
        "attention_mode": training["attention_mode"],
    }
    return training_rows, data_rows, teacher_rows, stats


def _source(
    *,
    source_id: str,
    label: str,
    path: str,
    sha256: str,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": source_id,
        "label": label,
        "path": path,
        "sha256": _sha256(sha256, field=f"sources.{source_id}.sha256"),
    }
    if manifest_sha256 is not None:
        result["manifestSha256"] = _sha256(
            manifest_sha256, field=f"sources.{source_id}.manifestSha256"
        )
    return result


def build_artifact(
    *,
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    teacher_provenance: Mapping[str, Any],
    dependency_scan: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    sbom: Mapping[str, Any],
    source_file_sha256: Mapping[str, str],
    package_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all report inputs and derive every reported claim from them."""

    source_hashes = _validate_source_hashes(source_file_sha256)
    evaluation_hash = _verify_self_hash(evaluation, description="evaluation report")
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("artifact_type") != "combined_release_evaluation_report"
        or evaluation.get("release_decision") != "passed"
        or evaluation.get("release_scope") != "community_research_release_candidate"
        or evaluation.get("production_ready") is not False
    ):
        raise ReleaseReportError("evaluation is not a passed non-production community research RC")
    training_hash = _verify_self_hash(training, description="training manifest")

    training_rows, data_rows, teacher_rows, provenance_stats = _validate_training_and_provenance(
        evaluation=evaluation,
        training=training,
        data_provenance=data_provenance,
        teacher_provenance=teacher_provenance,
        training_file_sha256=source_hashes["training"],
    )
    if training_hash != training.get("manifest_sha256"):  # pragma: no cover - helper guarantees it
        raise ReleaseReportError("training self-hash changed during validation")

    frozen_rows, frozen_long, pooled_row, bootstrap_info = _metric_rows(
        evaluation,
        training_manifest_sha256=training_hash,
        training_file_sha256=source_hashes["training"],
    )
    seed_rows = _seed_rows(evaluation)
    gate_rows, dependency_info = _validate_release_and_dependency_binding(
        dependency_scan=dependency_scan,
        release_gate=release_gate,
        sbom=sbom,
        source_hashes=source_hashes,
        package_binding=package_binding,
        source_registry_sha256=str(provenance_stats["source_registry_sha256"]),
    )

    validation_gate = _mapping(evaluation.get("validation_gate"), field="validation_gate")
    validation_file_sha = _sha256(
        validation_gate.get("file_sha256"), field="validation_gate.file_sha256"
    )
    validation_manifest_sha = _sha256(
        validation_gate.get("manifest_sha256"), field="validation_gate.manifest_sha256"
    )
    frozen = _mapping(evaluation.get("frozen_system_evaluation"), field="frozen_system_evaluation")
    frozen_file_sha = _sha256(
        evaluation.get("frozen_system_evaluation_file_sha256"),
        field="frozen_system_evaluation_file_sha256",
    )
    frozen_manifest_sha = _sha256(
        frozen.get("manifest_sha256"), field="frozen_system_evaluation.manifest_sha256"
    )

    coverage = _mapping(evaluation.get("coverage"), field="evaluation.coverage")
    measured = coverage.get("measured")
    not_measured = coverage.get("not_measured")
    if (
        not isinstance(measured, list)
        or not measured
        or any(not isinstance(item, str) or not item for item in measured)
        or not isinstance(not_measured, list)
        or not not_measured
        or any(not isinstance(item, str) or not item for item in not_measured)
    ):
        raise ReleaseReportError("evaluation coverage is incomplete")
    coverage_rows = [{"evidence": item, "status": "MEASURED"} for item in measured] + [
        {"evidence": item, "status": "NOT MEASURED"} for item in not_measured
    ]

    synthetic = next(row for row in frozen_rows if row["suite_id"] == "synthetic_test")
    synthetic_fpr = synthetic["pii_free_fpr"]
    if not isinstance(synthetic_fpr, float):
        raise ReleaseReportError("synthetic PII-free FPR is absent")
    lowest_f1 = min(frozen_rows, key=lambda row: float(row["strict_f1"]))
    lowest_recall = min(frozen_rows, key=lambda row: float(row["recall"]))
    seed_metrics_text = "、".join(
        f"seed {row['seed']} {_percent(float(row['strict_micro_f1']))}" for row in seed_rows
    )
    suite_metrics_text = "、".join(
        f"{row['suite']} {_percent(float(row['strict_f1']))}" for row in frozen_rows
    )
    missing_text = "、".join(f"`{item}`" for item in not_measured)
    selected_seed = _integer(evaluation.get("selected_seed"), field="evaluation.selected_seed")
    gate_count = len(gate_rows)
    synthetic_free_count = _count_text(int(synthetic["pii_free_documents"]))
    rejected_candidate_count = _count_text(int(provenance_stats["rejected_candidates"]))
    dependency_count_text = _count_text(int(dependency_info["dependency_count"]))
    bootstrap_confidence = _percent(float(bootstrap_info["confidence"]))

    sources = [
        _source(
            source_id="combined_evaluation",
            label="Combined release evaluation",
            path="evidence/evaluation_report.json",
            sha256=source_hashes["evaluation"],
            manifest_sha256=evaluation_hash,
        ),
        _source(
            source_id="frozen_evaluation",
            label="Frozen system evaluation summary",
            path="evidence/system_frozen_evaluation_summary.json",
            sha256=frozen_file_sha,
            manifest_sha256=frozen_manifest_sha,
        ),
        _source(
            source_id="validation_gate",
            label="Validation gate evidence",
            path="evidence/community_rc_validation_report.json",
            sha256=validation_file_sha,
            manifest_sha256=validation_manifest_sha,
        ),
        _source(
            source_id="training_manifest",
            label="Selected training manifest",
            path="evidence/training_manifest.json",
            sha256=source_hashes["training"],
            manifest_sha256=training_hash,
        ),
        _source(
            source_id="data_provenance",
            label="Release data provenance",
            path="evidence/data_provenance.json",
            sha256=source_hashes["data_provenance"],
            manifest_sha256=str(data_provenance["manifest_sha256"]),
        ),
        _source(
            source_id="teacher_provenance",
            label="Release teacher provenance",
            path="evidence/teacher_provenance.json",
            sha256=source_hashes["teacher_provenance"],
            manifest_sha256=str(teacher_provenance["manifest_sha256"]),
        ),
        _source(
            source_id="release_gate",
            label="Machine release gate receipt",
            path="audit/release-gate-report.json",
            sha256=source_hashes["release_gate"],
            manifest_sha256=str(dependency_info["release_gate_manifest_sha256"]),
        ),
        _source(
            source_id="dependency_scan",
            label="Locked dependency vulnerability scan",
            path="audit/dependency-scan.json",
            sha256=source_hashes["dependency_scan"],
        ),
        _source(
            source_id="sbom",
            label="Packaged CycloneDX SBOM",
            path="sbom.cdx.json",
            sha256=source_hashes["sbom"],
        ),
        _source(
            source_id="checksums",
            label="Current package checksums",
            path="checksums.txt",
            sha256=str(dependency_info["checksums_sha256"]),
        ),
        _source(
            source_id="source_registry",
            label="Gate-bound source registry",
            path="configs/data/source_registry.yaml",
            sha256=str(provenance_stats["source_registry_sha256"]),
        ),
    ]
    if "dependency_exceptions" in source_hashes:
        sources.append(
            _source(
                source_id="dependency_exceptions",
                label="Gate-bound dependency exceptions",
                path="audit/dependency-exceptions.json",
                sha256=source_hashes["dependency_exceptions"],
            )
        )

    summary_body = (
        "## 结论：机器检查支持研究 RC，生产边界仍关闭\n\n"
        "Strict span 要求实体标签和字符边界同时精确匹配。"
        f"{_count_text(int(pooled_row['documents']))} 篇冻结文档按 span 计数汇总的 strict F1 "
        f"为 **{_percent(float(pooled_row['f1']))}**，precision 为 "
        f"**{_percent(float(pooled_row['precision']))}**，recall 为 "
        f"**{_percent(float(pooled_row['recall']))}**。Synthetic 的 "
        f"{_count_text(int(synthetic['pii_free_documents']))} 篇无 PII 文档中有 "
        f"{_count_text(int(synthetic['false_positive_documents']))} 篇出现误报，FPR 为 "
        f"**{_percent(synthetic_fpr)}**。Evaluation decision 为 `passed`，同时明确声明 "
        "`production_ready=false`。"
    )
    gate_summary_body = (
        "## 当前包的 gate receipt 与内容地址一致\n\n"
        f"Receipt 中 {gate_count} 个唯一 gate 均为 PASS，并绑定当前 checksums、包文件数、"
        f"SBOM、source registry 和 dependency scan；使用 "
        f"{_count_text(int(dependency_info['used_exception_count']))} 个 dependency exception。"
        "这只证明机器可检查项通过，不替代人工审批。"
    )
    frozen_body = (
        "## 冻结评估显示不同 suite 的泛化差异\n\n"
        f"各 suite strict F1 为：{suite_metrics_text}。最低 strict F1 出现在 "
        f"{lowest_f1['suite']}（**{_percent(float(lowest_f1['strict_f1']))}**）；最低 recall "
        f"出现在 {lowest_recall['suite']}（**{_percent(float(lowest_recall['recall']))}**）。"
        "这些是当前证据支持的比较，不外推为生产质量。"
    )
    validation_body = (
        f"## {len(seed_rows)} 个 validation seed 均通过 gate\n\n"
        f"Validation strict micro F1 分别为 {seed_metrics_text}。选中 seed 为 "
        f"{selected_seed}；validation 结果用于 release gate，不能替代冻结测试。"
    )
    metric_definition_body = (
        "## 指标口径与不确定性来自同一冻结证据\n\n"
        "Strict precision、recall 和 F1 由 exact-label/exact-boundary 的 TP、FP、FN 重算；"
        f"各 suite 的区间使用 {_count_text(int(bootstrap_info['samples']))} 次 "
        f"`{bootstrap_info['method']}`，confidence 为 {bootstrap_confidence}。"
        "Pooled 指标按 suite 的 span 计数汇总，未把各 suite bootstrap 区间直接合并。"
    )
    training_identity_body = (
        "## 选中训练清单锁定基座、seed 与注意力模式\n\n"
        f"基座来源字段为 `{provenance_stats['base_source_id']}`，selected seed 为 "
        f"{provenance_stats['selected_seed']}，attention mode 为 "
        f"`{provenance_stats['attention_mode']}`。许可和准入结论不从该字段外推。"
    )
    data_provenance_body = (
        "## 已验证的数据来源支持 synthetic-only 训练结论\n\n"
        f"{provenance_stats['data_source_count']} 个已验证数据来源的梯度训练使用 "
        f"{_count_text(int(provenance_stats['gradient_documents']))} 篇文档，validation 使用 "
        f"{_count_text(int(provenance_stats['validation_documents']))} 篇文档；frozen test 的"
        f"训练使用计数为 {_count_text(int(provenance_stats['frozen_test_training_use']))}。"
    )
    teacher_body = (
        "## 模板候选生成来源不是 span 或伪标签 teacher\n\n"
        f"{provenance_stats['teacher_source_count']} 个经验证的 teacher 来源共生成并复核 "
        f"{_count_text(int(provenance_stats['reviewed_candidates']))} 个候选，其中接受 "
        f"{_count_text(int(provenance_stats['accepted_candidates']))} 个、拒绝 "
        f"{rejected_candidate_count} 个。输入证据声明 span teacher、"
        "pseudo-label teacher、原始输出训练使用和外部 API 均为 false。"
    )
    dependency_body = (
        "## Dependency scan 已完成，finding 计数透明保留\n\n"
        f"`{dependency_info['scanner']}` 对 {dependency_count_text} "
        f"个依赖完成扫描，记录 {_count_text(int(dependency_info['finding_count']))} 个 finding，"
        "并记录所扫描 SBOM 的内容 SHA。Finding 为零不等同于生产批准。"
    )
    sbom_body = (
        "## 当前 SBOM 保留完整 component 盘点\n\n"
        f"当前 CycloneDX SBOM 含 {_count_text(int(dependency_info['sbom_component_count']))} 个 "
        "component；其精确文件 SHA 保留在 source metadata 中。"
    )
    coverage_body = (
        "## 未测范围仍是生产声明的硬边界\n\n"
        f"Evaluation evidence 列出 {len(not_measured)} 个未测范围：{missing_text}。"
        "这些缺口不能由公开 synthetic 或 benchmark 指标外推。"
    )
    further_questions_body = "## 后续复核问题\n\n" + "\n".join(
        f"- `{item}` 需要什么受控数据、指标门槛和验收人？" for item in not_measured
    )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "中文 PII 模型社区研究 RC 发布评估",
            "description": "复现优先、证据优先的非生产发布决策报告。",
            "generatedAt": dependency_info["generated_at"],
            "cards": [
                {
                    "id": "pooled_f1",
                    "description": (
                        f"{_count_text(int(pooled_row['documents']))} 篇冻结文档按 span 计数汇总；"
                        "不是跨 suite bootstrap。"
                    ),
                    "dataset": "pooled_metrics",
                    "sourceId": "frozen_evaluation",
                    "metrics": [{"label": "Pooled strict F1", "field": "f1", "format": "percent"}],
                },
                {
                    "id": "pooled_precision",
                    "description": "冻结 suite 的 count-pooled strict precision。",
                    "dataset": "pooled_metrics",
                    "sourceId": "frozen_evaluation",
                    "metrics": [
                        {"label": "Pooled precision", "field": "precision", "format": "percent"}
                    ],
                },
                {
                    "id": "pooled_recall",
                    "description": "冻结 suite 的 count-pooled strict recall。",
                    "dataset": "pooled_metrics",
                    "sourceId": "frozen_evaluation",
                    "metrics": [{"label": "Pooled recall", "field": "recall", "format": "percent"}],
                },
                {
                    "id": "synthetic_fpr",
                    "description": (
                        f"{synthetic_free_count} 篇 synthetic 无 PII 文档中出现至少一个误报的比例。"
                    ),
                    "dataset": "synthetic_risk",
                    "sourceId": "frozen_evaluation",
                    "metrics": [
                        {"label": "PII-free document FPR", "field": "fpr", "format": "percent"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "frozen_prf",
                    "title": "冻结测试的 precision、recall 与 strict F1",
                    "subtitle": f"最低 recall suite：{lowest_recall['suite']}。",
                    "type": "bar",
                    "dataset": "frozen_metrics_long",
                    "sourceId": "frozen_evaluation",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "suite", "type": "nominal", "label": "Suite"},
                        "y": {"field": "value", "type": "quantitative", "label": "Score"},
                        "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                        "tooltip": [
                            {"field": "documents", "type": "quantitative", "label": "Documents"}
                        ],
                    },
                },
                {
                    "id": "validation_seeds",
                    "title": f"{len(seed_rows)} 个 validation seed 的 strict F1",
                    "subtitle": "Validation gate 不能替代冻结测试。",
                    "type": "bar",
                    "dataset": "validation_seeds",
                    "sourceId": "validation_gate",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "seed", "type": "nominal", "label": "Seed"},
                        "y": {
                            "field": "strict_micro_f1",
                            "type": "quantitative",
                            "label": "Strict F1",
                        },
                    },
                },
            ],
            "tables": [
                {
                    "id": "frozen_table",
                    "title": "冻结测试精确结果与置信区间",
                    "subtitle": (
                        f"CI 为各 suite 的 {_count_text(int(bootstrap_info['samples']))} 次"
                        f" {bootstrap_info['method']}，confidence={bootstrap_confidence}。"
                    ),
                    "dataset": "frozen_metrics",
                    "sourceId": "frozen_evaluation",
                    "defaultSort": {"field": "strict_f1", "direction": "desc"},
                    "columns": [
                        {"field": "suite", "label": "Suite", "type": "text"},
                        {"field": "documents", "label": "Docs", "format": "integer"},
                        {"field": "strict_f1", "label": "Strict F1", "format": "percent"},
                        {"field": "strict_f1_ci_lower", "label": "CI low", "format": "percent"},
                        {"field": "strict_f1_ci_upper", "label": "CI high", "format": "percent"},
                        {"field": "precision", "label": "Precision", "format": "percent"},
                        {"field": "recall", "label": "Recall", "format": "percent"},
                        {"field": "pii_free_fpr", "label": "PII-free FPR", "format": "percent"},
                    ],
                },
                {
                    "id": "gate_table",
                    "title": "机器发布闸门",
                    "subtitle": f"当前 gate receipt 含 {gate_count} 个唯一 PASS 项。",
                    "dataset": "gate_checks",
                    "sourceId": "release_gate",
                    "defaultSort": {"field": "gate", "direction": "asc"},
                    "columns": [
                        {"field": "gate", "label": "Gate", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {"field": "issue_count", "label": "Issues", "format": "integer"},
                    ],
                },
                {
                    "id": "coverage_table",
                    "title": "证据覆盖与缺口",
                    "subtitle": "未测项目不能由已测公开数据推断。",
                    "dataset": "coverage",
                    "sourceId": "combined_evaluation",
                    "defaultSort": {"field": "status", "direction": "asc"},
                    "columns": [
                        {"field": "evidence", "label": "Evidence", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                    ],
                },
                {
                    "id": "training_identity_table",
                    "title": "训练清单中的基座来源",
                    "subtitle": "不从训练清单之外推断许可或准入结论。",
                    "dataset": "training_identity",
                    "sourceId": "training_manifest",
                    "defaultSort": {"field": "source_id", "direction": "asc"},
                    "columns": [
                        {"field": "source_id", "label": "Source", "type": "text"},
                        {"field": "manifest_field", "label": "Manifest field", "type": "text"},
                    ],
                },
                {
                    "id": "data_sources_table",
                    "title": "训练数据来源",
                    "subtitle": "角色、许可、计数与准入值直接来自已验证 provenance。",
                    "dataset": "data_sources",
                    "sourceId": "data_provenance",
                    "defaultSort": {"field": "source_id", "direction": "asc"},
                    "columns": [
                        {"field": "source_id", "label": "Source", "type": "text"},
                        {"field": "role", "label": "Role", "type": "text"},
                        {"field": "sample_count", "label": "Count", "format": "integer"},
                        {"field": "license", "label": "License", "type": "text"},
                        {
                            "field": "public_weight_training_allowed",
                            "label": "Public weight",
                            "type": "boolean",
                        },
                    ],
                },
                {
                    "id": "teacher_sources_table",
                    "title": "模板候选生成来源",
                    "subtitle": "角色、许可和候选计数直接来自已验证 provenance。",
                    "dataset": "teacher_sources",
                    "sourceId": "teacher_provenance",
                    "defaultSort": {"field": "source_id", "direction": "asc"},
                    "columns": [
                        {"field": "source_id", "label": "Source", "type": "text"},
                        {"field": "role", "label": "Role", "type": "text"},
                        {"field": "sample_count", "label": "Accepted", "format": "integer"},
                        {"field": "reviewed_count", "label": "Reviewed", "format": "integer"},
                        {"field": "license", "label": "License", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "combined_evaluation",
                    "body": summary_body,
                },
                {
                    "id": "gate_summary",
                    "type": "markdown",
                    "sourceId": "release_gate",
                    "body": gate_summary_body,
                },
                {"id": "gates", "type": "table", "tableId": "gate_table"},
                {
                    "id": "headline",
                    "type": "metric-strip",
                    "cardIds": ["pooled_f1", "pooled_precision", "pooled_recall", "synthetic_fpr"],
                },
                {
                    "id": "metric_definitions",
                    "type": "markdown",
                    "sourceId": "frozen_evaluation",
                    "body": metric_definition_body,
                },
                {
                    "id": "frozen_title",
                    "type": "markdown",
                    "sourceId": "frozen_evaluation",
                    "body": frozen_body,
                },
                {"id": "frozen_chart", "type": "chart", "chartId": "frozen_prf"},
                {"id": "frozen_exact", "type": "table", "tableId": "frozen_table"},
                {
                    "id": "validation_title",
                    "type": "markdown",
                    "sourceId": "validation_gate",
                    "body": validation_body,
                },
                {"id": "validation_chart", "type": "chart", "chartId": "validation_seeds"},
                {
                    "id": "training_identity_title",
                    "type": "markdown",
                    "sourceId": "training_manifest",
                    "body": training_identity_body,
                },
                {"id": "training_identity", "type": "table", "tableId": "training_identity_table"},
                {
                    "id": "data_provenance_title",
                    "type": "markdown",
                    "sourceId": "data_provenance",
                    "body": data_provenance_body,
                },
                {"id": "data_sources", "type": "table", "tableId": "data_sources_table"},
                {
                    "id": "teacher_title",
                    "type": "markdown",
                    "sourceId": "teacher_provenance",
                    "body": teacher_body,
                },
                {"id": "teacher_sources", "type": "table", "tableId": "teacher_sources_table"},
                {
                    "id": "dependency_title",
                    "type": "markdown",
                    "sourceId": "dependency_scan",
                    "body": dependency_body,
                },
                {
                    "id": "sbom_title",
                    "type": "markdown",
                    "sourceId": "sbom",
                    "body": sbom_body,
                },
                {
                    "id": "coverage_title",
                    "type": "markdown",
                    "sourceId": "combined_evaluation",
                    "body": coverage_body,
                },
                {"id": "coverage", "type": "table", "tableId": "coverage_table"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "sourceId": "combined_evaluation",
                    "body": (
                        "## 下一步：补齐未测范围后再作部署决策\n\n"
                        f"当前 evidence 明确列出 {len(not_measured)} 个未测范围：{missing_text}。"
                        "任何后续改动都应使用新的、事前冻结的验证与测试协议；本报告不构成生产批准。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "sourceId": "combined_evaluation",
                    "body": further_questions_body,
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": dependency_info["generated_at"],
            "status": "ready",
            "datasets": {
                "pooled_metrics": [pooled_row],
                "synthetic_risk": [{"fpr": synthetic_fpr}],
                "frozen_metrics": frozen_rows,
                "frozen_metrics_long": frozen_long,
                "validation_seeds": seed_rows,
                "gate_checks": gate_rows,
                "coverage": coverage_rows,
                "training_identity": training_rows,
                "data_sources": data_rows,
                "teacher_sources": teacher_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://community-research-release-candidate",
            "controls": {"edit": False, "refresh": False},
        },
    }
    _public_safe(artifact)
    return artifact


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ReleaseReportError("output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, required=True)
    parser.add_argument("--teacher-provenance", type=Path, required=True)
    parser.add_argument("--dependency-scan", type=Path, required=True)
    parser.add_argument("--dependency-exceptions", type=Path)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        loaded: dict[str, tuple[dict[str, Any], str]] = {
            "evaluation": _load(args.evaluation_report, description="evaluation report"),
            "training": _load(args.training_manifest, description="training manifest"),
            "data_provenance": _load(args.data_provenance, description="data provenance"),
            "teacher_provenance": _load(args.teacher_provenance, description="teacher provenance"),
            "dependency_scan": _load(args.dependency_scan, description="dependency scan"),
            "release_gate": _load(args.release_gate, description="release gate"),
            "sbom": _load(args.artifact_dir / "sbom.cdx.json", description="packaged SBOM"),
        }
        if args.dependency_exceptions is not None:
            loaded["dependency_exceptions"] = _load(
                args.dependency_exceptions,
                description="dependency exceptions",
            )
        artifact = build_artifact(
            evaluation=loaded["evaluation"][0],
            training=loaded["training"][0],
            data_provenance=loaded["data_provenance"][0],
            teacher_provenance=loaded["teacher_provenance"][0],
            dependency_scan=loaded["dependency_scan"][0],
            release_gate=loaded["release_gate"][0],
            sbom=loaded["sbom"][0],
            source_file_sha256={name: item[1] for name, item in loaded.items()},
            package_binding=validate_package_dir(args.artifact_dir),
        )
        _write_atomic(args.output.expanduser().absolute(), artifact)
    except (OSError, ReleaseReportError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"block_count": len(artifact["manifest"]["blocks"]), "status": "written"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
