#!/usr/bin/env python3
"""Build an atomic, local-only Hugging Face model release directory.

Only an explicit allowlist of files is copied.  The command does not download,
train, upload, or invoke Hugging Face APIs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

if __package__:
    from scripts import build_model_card_evidence_v2 as model_card_v2  # noqa: E402
else:
    import build_model_card_evidence_v2 as model_card_v2  # noqa: E402

from pii_zh.data.artifact_policy import (  # noqa: E402
    ArtifactPolicyError,
    assert_document_allowed,
)
from pii_zh.evaluation import (  # noqa: E402
    release_eval_v2_prediction_provenance as model_provenance,
)
from pii_zh.taxonomy import load_taxonomy  # noqa: E402

CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
OPTIONAL_CHECKPOINT_FILES = ("added_tokens.json",)
EVIDENCE_FILES = (
    "taxonomy.yaml",
    "id2label.json",
    "calibration.json",
    "thresholds.yaml",
    "training_manifest.json",
    "data_provenance.json",
    "teacher_provenance.json",
    "evaluation_report.json",
    "model-index.yml",
    "sbom.cdx.json",
)
OPTIONAL_EVIDENCE_FILES = (
    model_card_v2.EVIDENCE_FILENAME,
    model_card_v2.CONTRACT_FILENAME,
    model_card_v2.SELECTED_RECEIPT_FILENAME,
)
PROJECT_FILES = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
    "SECURITY.md": "SECURITY.md",
}
REMOTE_CODE_FILES = ("configuration_qwen3_bi.py", "modeling_qwen3_bi.py")
COMMUNITY_PREAUTHORIZATION_FILENAME = "community_v2_preauthorization.json"
COMMUNITY_PREAUTHORIZATION_STATE = "unpublished_local_candidate_pending_community_contract"
COMMUNITY_V2_ARTIFACT_SCHEMA_PATH = (
    "configs/release/community_cascade_release_v2.artifact-evidence.schema.json"
)
COMMUNITY_V2_ARTIFACT_PRODUCER_PATH = "scripts/produce_community_cascade_release_v2_artifacts.py"
COMMUNITY_V2_ARTIFACT_SCHEMA_VERSION = "pii-zh.community-release-artifact-evidence.v2"
LEGACY_REMOTE_CODE_PROFILE = "legacy_release_eligible_true_v1"
COMMUNITY_V2_REMOTE_CODE_PROFILE = "community_v2_false_eligible_preserving_v1"
COMMUNITY_V2_PACKAGE_SUPPORT_PROFILE = "community_v2_seed97_attribution_v1"
COMMUNITY_V2_TEMPLATE_ASSET_PATH = "src/pii_zh/data/synthetic/assets/curated_templates_v1.json"
COMMUNITY_V2_PII_BENCH_DATASET_ID = "wan9yu/pii-bench-zh"
COMMUNITY_V2_PII_BENCH_DATASET_REVISION = "c350b94897af668517ff5de237d89f2ce2eaa6f0"
COMMUNITY_V2_PII_BENCH_METRICS: dict[str, dict[str, float | int]] = {
    "formal": {
        "documents": 5000,
        "strict_micro_f1": 0.59921050,
        "strict_macro_f1": 0.57994984,
    },
    "chat": {
        "documents": 3000,
        "strict_micro_f1": 0.47628738,
        "strict_macro_f1": 0.41874684,
    },
    "pooled": {
        "documents": 8000,
        "strict_micro_f1": 0.55750532,
        "strict_macro_f1": 0.52965259,
    },
}
COMMUNITY_V2_BASE_MODEL_ID = "ZJUICSR/AIguard-pii-detection-fast"
COMMUNITY_V2_BASE_MODEL_REVISION = "677a5ebc1600fef61e8973cafd3026be322b3a73"
COMMUNITY_V2_BASE_MODEL_LICENSE = "apache-2.0"
COMMUNITY_V2_MAPPED_TARGET_LABELS = (
    "PERSON_NAME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "VEHICLE_LICENSE_PLATE",
    "SECRET",
)
COMMUNITY_V2_EVIDENCE_FILES = (
    "taxonomy.yaml",
    "id2label.json",
    "training_manifest.json",
)
COMMUNITY_V2_REQUIRED_RELEASE_FILES = frozenset(
    {
        "README.md",
        *CHECKPOINT_FILES,
        *COMMUNITY_V2_EVIDENCE_FILES,
        *PROJECT_FILES.values(),
        *REMOTE_CODE_FILES,
        COMMUNITY_PREAUTHORIZATION_FILENAME,
        "checksums.txt",
    }
)
REQUIRED_RELEASE_FILES = frozenset(
    {
        "README.md",
        *CHECKPOINT_FILES,
        *EVIDENCE_FILES,
        *PROJECT_FILES.values(),
        *REMOTE_CODE_FILES,
        "checksums.txt",
    }
)
ALLOWED_RELEASE_FILES = (
    REQUIRED_RELEASE_FILES
    | frozenset(OPTIONAL_CHECKPOINT_FILES)
    | frozenset(OPTIONAL_EVIDENCE_FILES)
)
FORBIDDEN_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".joblib", ".npy", ".npz", ".pickle", ".pkl", ".pt", ".pth"}
)
RAW_DATA_SUFFIXES = frozenset({".arrow", ".csv", ".jsonl", ".parquet", ".tsv"})
FORBIDDEN_NAME_PARTS = (
    "optimizer",
    "scheduler",
    "trainer_state",
    "rng_state",
    "gradient",
    "tfevents",
    "api_prompt",
    "api_response",
    "customer_data",
)
AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
    "AutoModelForTokenClassification": ("modeling_qwen3_bi.Qwen3BiForTokenClassification"),
}
BOUNDARY_MODE_CONFIG_KEY = "pii_zh_boundary_mode"
BOUNDARY_MODE = "unicode_codepoint_v1"
BOUND_ARTIFACT_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)


class ReleaseBuildError(ValueError):
    """Raised when an input is unsafe or incomplete."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical_utf8_json_hash(value: Any, *, remove: str | None = None) -> str:
    if not isinstance(value, dict):
        raise ReleaseBuildError("canonical community metadata must be a JSON object")
    document = dict(value)
    if remove is not None:
        document.pop(remove, None)
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReleaseBuildError("community metadata is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute lexical path without resolving symlinks."""

    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ReleaseBuildError(f"community path traversal is forbidden: {path}")
    return expanded.absolute()


def _reject_symlink_path_or_ancestry(path: Path, description: str) -> Path:
    """Reject a symlink at ``path`` or any existing lexical ancestor."""

    lexical = _lexical_absolute(path)
    current = lexical
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ReleaseBuildError(f"cannot inspect {description}: {current}") from exc
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise ReleaseBuildError(f"{description} must not use symlinks: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    return lexical


def output_artifact_fingerprint(root: Path) -> dict[str, Any]:
    weights = tuple(sorted(root.glob("model*.safetensors")))
    files = {path.name: sha256_file(path) for path in weights if path.is_file()}
    for name in BOUND_ARTIFACT_METADATA_FILES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            files[name] = sha256_file(path)
    weight_hashes = {name: files[name] for name in sorted(files) if name.endswith(".safetensors")}
    return {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": _canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": _canonical_json_hash(files),
    }


def validate_training_artifact_binding(manifest_path: Path, checkpoint_dir: Path) -> None:
    training = _load_json_object(manifest_path, "training manifest")
    expected = training.get("manifest_sha256")
    unsigned = dict(training)
    unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(expected, str)
        or _canonical_json_hash(unsigned) != expected
        or training.get("status") != "completed"
    ):
        raise ReleaseBuildError("training manifest completion/hash check failed")
    actual = output_artifact_fingerprint(checkpoint_dir)
    if actual.get("weight_files") != ["model.safetensors"]:
        raise ReleaseBuildError("release checkpoint must contain exactly model.safetensors")
    if training.get("output_artifact") != actual:
        raise ReleaseBuildError("checkpoint files do not match the completed training manifest")


def _require_regular_file(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise ReleaseBuildError(f"{description} must not be a symlink: {path}")
    if not path.is_file():
        raise ReleaseBuildError(f"missing {description}: {path}")
    return path


def _reject_contaminated_tree(root: Path, description: str) -> None:
    if not root.is_dir():
        raise ReleaseBuildError(f"{description} is not a directory: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        lowered_name = path.name.lower()
        if path.is_symlink():
            raise ReleaseBuildError(f"{description} contains symlink: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ReleaseBuildError(
                f"{description} contains forbidden pickle/checkpoint file: {relative}"
            )
        if path.suffix.lower() in RAW_DATA_SUFFIXES:
            raise ReleaseBuildError(f"{description} contains raw/tabular data file: {relative}")
        if any(fragment in lowered_name for fragment in FORBIDDEN_NAME_PARTS):
            raise ReleaseBuildError(
                f"{description} contains forbidden training/private artifact: {relative}"
            )
        if any(part.lower() in {"raw", "raw_data"} for part in relative.parts[:-1]):
            raise ReleaseBuildError(f"{description} contains a raw-data directory: {relative}")


def validate_safetensors(path: Path) -> None:
    """Perform a dependency-free structural validation of a safetensors file."""

    size = path.stat().st_size
    if size < 10:
        raise ReleaseBuildError("model.safetensors is too small to contain a valid header")
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length < 2 or header_length > size - 8 or header_length > 100 * 1024 * 1024:
            raise ReleaseBuildError("model.safetensors has an invalid header length")
        raw_header = stream.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError("model.safetensors header is not valid UTF-8 JSON") from exc
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ReleaseBuildError("model.safetensors contains no tensors")
    data_length = size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in tensors.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ReleaseBuildError("model.safetensors tensor metadata is malformed")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_length
        ):
            raise ReleaseBuildError(f"model.safetensors has invalid data offsets for {name!r}")
        if not isinstance(metadata.get("shape"), list) or not isinstance(
            metadata.get("dtype"), str
        ):
            raise ReleaseBuildError(f"model.safetensors has incomplete metadata for {name!r}")
        intervals.append((offsets[0], offsets[1], name))
    expected_start = 0
    for start, end, name in sorted(intervals):
        if start != expected_start:
            raise ReleaseBuildError(
                f"model.safetensors has overlapping or unreferenced data before {name!r}"
            )
        expected_start = end
    if expected_start != data_length:
        raise ReleaseBuildError("model.safetensors has trailing or unreferenced tensor data")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"{description} must be a JSON object: {path}")
    return value


def _load_strict_json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    _require_regular_file(path, description)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReleaseBuildError(f"cannot read {description}: {path}") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseBuildError(f"{description} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ReleaseBuildError(f"{description} has a non-finite JSON number")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"{description} is not strict UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"{description} must be a JSON object: {path}")
    return value, payload


def _community_training_initialization_facts(training: dict[str, Any]) -> dict[str, Any]:
    initialization = training.get("initialization")
    if not isinstance(initialization, dict):
        raise ReleaseBuildError("community training initialization is missing")
    projections = initialization.get("head_projections")
    source_hashes = initialization.get("source_hashes")
    mapped_labels = initialization.get("mapped_target_labels")
    if (
        training.get("base_source_id") != COMMUNITY_V2_BASE_MODEL_ID
        or training.get("source_revision") != COMMUNITY_V2_BASE_MODEL_REVISION
        or not isinstance(source_hashes, dict)
        or set(source_hashes)
        != {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
        or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in source_hashes.values()
        )
        or not isinstance(projections, list)
        or len(projections) != len(COMMUNITY_V2_MAPPED_TARGET_LABELS)
        or mapped_labels != list(COMMUNITY_V2_MAPPED_TARGET_LABELS)
        or [item.get("target_label") for item in projections if isinstance(item, dict)]
        != list(COMMUNITY_V2_MAPPED_TARGET_LABELS)
        or initialization.get("source_architecture") != "Qwen3ForTokenClassification"
        or initialization.get("outside_row_copied") is not True
        or initialization.get("mapped_head_rows_verified") is not True
    ):
        raise ReleaseBuildError("community training base-model initialization facts are invalid")
    return {
        "base_model_id": COMMUNITY_V2_BASE_MODEL_ID,
        "base_model_revision": COMMUNITY_V2_BASE_MODEL_REVISION,
        "base_model_license": COMMUNITY_V2_BASE_MODEL_LICENSE,
        "head_projection_count": len(projections),
        "mapped_target_labels": list(COMMUNITY_V2_MAPPED_TARGET_LABELS),
        "source_hashes": dict(sorted(source_hashes.items())),
    }


def _community_training_data_facts(
    training: dict[str, Any], template_asset: dict[str, Any]
) -> dict[str, Any]:
    datasets = training.get("datasets")
    privacy = training.get("privacy")
    split_isolation = training.get("split_isolation")
    if not isinstance(datasets, dict):
        raise ReleaseBuildError("community training dataset summary is missing")
    summaries: dict[str, dict[str, Any]] = {}
    expected_counts = {
        "train": (87_995, 39_600),
        "validation": (2_005, 900),
    }
    for split, (document_count, pii_free_count) in expected_counts.items():
        split_document = datasets.get(split)
        summary = split_document.get("summary") if isinstance(split_document, dict) else None
        sources = summary.get("sources") if isinstance(summary, dict) else None
        label_counts = summary.get("label_counts") if isinstance(summary, dict) else None
        if (
            not isinstance(summary, dict)
            or summary.get("document_count") != document_count
            or summary.get("pii_free_document_count") != pii_free_count
            or not isinstance(label_counts, dict)
            or len(label_counts) != 24
            or not isinstance(sources, list)
            or len(sources) != 1
            or not isinstance(sources[0], dict)
            or sources[0].get("source_id") != "repo_curated_synthetic_templates"
            or sources[0].get("source_kind") != "curated_deterministic_synthetic"
            or sources[0].get("document_count") != document_count
        ):
            raise ReleaseBuildError(f"community {split} synthetic dataset facts are invalid")
        summaries[split] = summary
    if (
        privacy
        != {
            "contains_entity_values": False,
            "contains_raw_text": False,
            "synthetic_only": True,
            "trainer_reporting_integrations": [],
        }
        or not isinstance(split_isolation, dict)
        or split_isolation.get("limitation") != "validation_informed_template_family_overlap"
        or split_isolation.get("policy") != "template-overlap-development-v1"
        or split_isolation.get("release_test_eligible") is not False
        or split_isolation.get("template_group_overlap_allowed") is not True
        or split_isolation.get("collision_counts", {}).get("template_group") != 17
    ):
        raise ReleaseBuildError("community synthetic-data privacy/split limitation is invalid")

    candidate_source = template_asset.get("candidate_source")
    curation = template_asset.get("curation")
    if (
        template_asset.get("asset_id") != "pii-zh-curated-synthetic-templates"
        or template_asset.get("license") != "Apache-2.0"
        or not isinstance(candidate_source, dict)
        or candidate_source.get("model_id") != "Qwen/Qwen3-8B"
        or candidate_source.get("deployment") != "local"
        or not isinstance(curation, dict)
        or curation.get("review_method") != "manual_semantic_and_near_duplicate_review"
        or curation.get("accepted_candidate_count") != 53
        or curation.get("human_authored_positive_count") != 34
        or curation.get("human_authored_hard_negative_count") != 20
        or curation.get("raw_rejected_outputs_included") is not False
    ):
        raise ReleaseBuildError("community synthetic template provenance is invalid")
    return {
        "training_document_count": 87_995,
        "training_pii_free_document_count": 39_600,
        "validation_document_count": 2_005,
        "validation_pii_free_document_count": 900,
        "entity_type_count": 24,
        "synthetic_only": True,
        "contains_real_customer_or_production_pii": False,
        "validation_blind": False,
        "template_group_overlap_count": 17,
        "model_assisted_accepted_template_count": 53,
        "human_authored_positive_template_count": 34,
        "human_authored_hard_negative_template_count": 20,
        "template_candidate_model_id": "Qwen/Qwen3-8B",
    }


def _community_pii_bench_posthoc_summary(
    report: dict[str, Any],
    *,
    training_manifest: dict[str, Any],
    training_payload: bytes,
) -> dict[str, Any]:
    expected_self = report.get("report_sha256")
    status = report.get("status")
    benchmark = report.get("benchmark")
    candidate = report.get("candidate")
    comparison = report.get("comparison_contract")
    envelope = report.get("active_comparator_envelope")
    results = report.get("results")
    if (
        not isinstance(expected_self, str)
        or expected_self != _canonical_utf8_json_hash(report, remove="report_sha256")
        or report.get("report_type") != "bio24_model_raw_posthoc_evaluation"
        or report.get("candidate_id") != "aiguard24-v3-seed97"
        or not isinstance(status, dict)
        or status.get("evidence_classification") != "posthoc_descriptive_public_benchmark"
        or status.get("public_test_exposed") is not True
        or status.get("posthoc_lineage") is not True
        or status.get("selection_allowed") is not False
        or status.get("release_selection_eligible") is not False
        or status.get("confirmatory_claim_allowed") is not False
        or not isinstance(benchmark, dict)
        or benchmark.get("dataset_id") != COMMUNITY_V2_PII_BENCH_DATASET_ID
        or benchmark.get("dataset_revision") != COMMUNITY_V2_PII_BENCH_DATASET_REVISION
        or benchmark.get("fixed_gold_verified") is not True
        or not isinstance(candidate, dict)
        or candidate.get("training_manifest_file_sha256")
        != hashlib.sha256(training_payload).hexdigest()
        or candidate.get("training_manifest_sha256") != training_manifest.get("manifest_sha256")
        or candidate.get("attention_mode") != "full"
        or candidate.get("core_entity_count") != 24
        or not isinstance(comparison, dict)
        or comparison.get("track") != "model_raw"
        or comparison.get("threshold_tuning") is not False
        or comparison.get("checkpoint_selection") is not False
        or not isinstance(envelope, dict)
        or envelope.get("descriptive_active_envelope_passed") is not False
        or envelope.get("confirmatory_claim_allowed") is not False
        or not isinstance(results, dict)
    ):
        raise ReleaseBuildError(
            "PII Bench ZH report is not the frozen seed-97 descriptive one-shot"
        )
    for suite, expected in COMMUNITY_V2_PII_BENCH_METRICS.items():
        observed = results.get(suite)
        micro = observed.get("strict_micro") if isinstance(observed, dict) else None
        macro = observed.get("strict_macro") if isinstance(observed, dict) else None
        if {
            "documents": observed.get("documents") if isinstance(observed, dict) else None,
            "strict_micro_f1": micro.get("f1") if isinstance(micro, dict) else None,
            "strict_macro_f1": macro.get("f1") if isinstance(macro, dict) else None,
        } != expected:
            raise ReleaseBuildError(f"PII Bench ZH {suite} metrics differ from the frozen report")
    return {
        "dataset_id": COMMUNITY_V2_PII_BENCH_DATASET_ID,
        "dataset_revision": COMMUNITY_V2_PII_BENCH_DATASET_REVISION,
        "evidence_classification": "posthoc_descriptive_public_benchmark",
        "public_test_exposed": True,
        "selection_allowed": False,
        "descriptive_active_envelope_passed": False,
        "model_raw_evaluated": True,
        "full_system_evaluated": False,
        "suites": {key: dict(value) for key, value in COMMUNITY_V2_PII_BENCH_METRICS.items()},
    }


def render_community_v2_notice(training: dict[str, Any]) -> str:
    facts = _community_training_initialization_facts(training)
    return (
        "pii-zh-qwen 0.2.0rc1 community candidate\n"
        "Copyright 2026 pii-zh-qwen contributors\n\n"
        "This unpublished local release candidate contains a Simplified-Chinese PII "
        "token-classification model and its community cascade integration.\n\n"
        f"Model initialization is bound to {facts['base_model_id']} at revision "
        f"{facts['base_model_revision']}. The upstream model card declares the base model "
        f"under {facts['base_model_license']}; upstream attribution and license terms apply "
        "independently to those materials.\n\n"
        "The fixed training manifest records a strict full-attention backbone copy and "
        f"{facts['head_projection_count']} verified source-to-target classification-head "
        "projections. Exact source hashes and migration details are recorded in "
        "THIRD_PARTY_NOTICES.md and training_manifest.json. Final license approval remains "
        "a separate human release gate.\n"
    )


def render_community_v2_third_party_notices(training: dict[str, Any]) -> str:
    facts = _community_training_initialization_facts(training)
    source_hash_lines = "\n".join(
        f"- `{name}`: `sha256:{digest}`" for name, digest in facts["source_hashes"].items()
    )
    mapped_labels = ", ".join(f"`{label}`" for label in facts["mapped_target_labels"])
    return f"""# Third-Party Notices — pii-zh-qwen 0.2.0rc1 community candidate

## AIguard PII detection model

- Source: `{facts["base_model_id"]}`
- Revision: `{facts["base_model_revision"]}`
- Upstream-declared license: `{facts["base_model_license"]}`
- Release role: fixed initialization checkpoint for the community token classifier

The community training manifest binds these source files:

{source_hash_lines}

The initialization contract records a strict state-dict backbone copy from
`Qwen3ForTokenClassification`, copies the outside-label row, and verifies
{facts["head_projection_count"]} classification-head projections for: {mapped_labels}.
The target model changes attention from causal to full bidirectional attention and trains the
24-entity, 49-token-label community taxonomy. The base checkpoint itself is not redistributed
as a separate artifact; the package contains the resulting merged candidate weights.

## Runtime dependencies

Runtime dependency names, versions, licenses, wheel hashes, SBOM content and vulnerability-scan
receipts are maintained by the separate typed community release artifacts. Dependencies are not
vendored into this model package.

## Human review boundary

This notice is generated from the fixed training manifest and records mechanical attribution
facts. It does not replace the pending human license review required by the release contract.
"""


def validate_typed_model_card_companion(
    *,
    model_card_artifact_path: Path,
    markdown_path: Path,
    final_model_manifest_path: Path,
    training_manifest_path: Path,
    pii_bench_report_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate the typed model-card artifact and its exact Markdown companion."""

    artifact, _artifact_payload = _load_strict_json_object(
        model_card_artifact_path, "typed model-card artifact"
    )
    schema, _schema_payload = _load_strict_json_object(
        repository_root / COMMUNITY_V2_ARTIFACT_SCHEMA_PATH,
        "community artifact evidence schema",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(artifact)
    except (SchemaError, ValidationError) as exc:
        raise ReleaseBuildError("typed model-card artifact fails its closed schema") from exc

    if (
        artifact.get("schema_version") != COMMUNITY_V2_ARTIFACT_SCHEMA_VERSION
        or artifact.get("artifact_id") != "model_card"
        or artifact.get("status") != "PASS"
        or artifact.get("privacy")
        != {
            "contains_local_paths": False,
            "contains_raw_records": False,
            "contains_secrets": False,
            "network_used": False,
        }
        or artifact.get("artifact_sha256")
        != _canonical_utf8_json_hash(artifact, remove="artifact_sha256")
    ):
        raise ReleaseBuildError("typed model-card artifact identity does not verify")

    producer_hash = sha256_file(repository_root / COMMUNITY_V2_ARTIFACT_PRODUCER_PATH)
    schema_hash = sha256_file(repository_root / COMMUNITY_V2_ARTIFACT_SCHEMA_PATH)
    expected_producer = {
        "file_sha256": producer_hash,
        "schema_file_sha256": schema_hash,
        "implementation_identity_sha256": _canonical_utf8_json_hash(
            {
                "artifact_id": "model_card",
                "producer_file_sha256": producer_hash,
                "schema_file_sha256": schema_hash,
                "schema_version": COMMUNITY_V2_ARTIFACT_SCHEMA_VERSION,
            }
        ),
    }
    if artifact.get("producer") != expected_producer:
        raise ReleaseBuildError("typed model-card producer identity is stale")

    inputs = artifact.get("inputs")
    expected_input_ids = {
        "quality_receipt",
        "final_model_manifest",
        "service_configuration_manifest",
        "training_manifest",
        "template_asset",
        "pii_bench_posthoc_report",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_ids:
        raise ReleaseBuildError("typed model-card inputs are incomplete")
    final_payload = _require_regular_file(
        final_model_manifest_path, "final model manifest"
    ).read_bytes()
    training, training_payload = _load_strict_json_object(
        training_manifest_path, "seed-97 training manifest"
    )
    training_facts = _community_training_initialization_facts(training)
    template_asset_path = repository_root / COMMUNITY_V2_TEMPLATE_ASSET_PATH
    template_asset, template_asset_payload = _load_strict_json_object(
        template_asset_path, "community synthetic template asset"
    )
    training_data_facts = _community_training_data_facts(training, template_asset)
    pii_bench_report, pii_bench_payload = _load_strict_json_object(
        pii_bench_report_path, "PII Bench ZH posthoc report"
    )
    pii_bench_summary = _community_pii_bench_posthoc_summary(
        pii_bench_report,
        training_manifest=training,
        training_payload=training_payload,
    )
    expected_bound_inputs = {
        "final_model_manifest": {
            "file_sha256": hashlib.sha256(final_payload).hexdigest(),
            "size_bytes": len(final_payload),
        },
        "training_manifest": {
            "file_sha256": hashlib.sha256(training_payload).hexdigest(),
            "size_bytes": len(training_payload),
        },
        "template_asset": {
            "file_sha256": hashlib.sha256(template_asset_payload).hexdigest(),
            "size_bytes": len(template_asset_payload),
        },
        "pii_bench_posthoc_report": {
            "file_sha256": hashlib.sha256(pii_bench_payload).hexdigest(),
            "size_bytes": len(pii_bench_payload),
        },
    }
    for logical_id, binding in expected_bound_inputs.items():
        if inputs.get(logical_id) != binding:
            raise ReleaseBuildError(f"typed model-card {logical_id} binding is stale")

    result = artifact.get("result")
    expected_result_keys = {
        "media_type",
        "markdown",
        "markdown_sha256",
        "entity_type_count",
        "token_label_count",
        "quality_receipt_sha256",
        "model_manifest_file_sha256",
        "service_manifest_file_sha256",
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "base_model_id",
        "base_model_revision",
        "base_model_license",
        "head_projection_count",
        "ordered_entity_labels",
        "training_data_facts",
        "template_asset_file_sha256",
        "pii_bench_report_file_sha256",
        "pii_bench_report_sha256",
        "pii_bench_posthoc",
        "placeholder_count",
        "forbidden_claims_absent",
    }
    if not isinstance(result, dict) or set(result) != expected_result_keys:
        raise ReleaseBuildError("typed model-card result has an invalid closed shape")
    markdown_payload = _require_regular_file(
        markdown_path, "typed model-card Markdown companion"
    ).read_bytes()
    markdown = result.get("markdown")
    if (
        not isinstance(markdown, str)
        or markdown_payload != markdown.encode("utf-8")
        or result.get("markdown_sha256") != hashlib.sha256(markdown_payload).hexdigest()
        or result.get("media_type") != "text/markdown; charset=utf-8"
        or result.get("entity_type_count") != 24
        or result.get("token_label_count") != 49
        or result.get("placeholder_count") != 0
        or result.get("forbidden_claims_absent") is not True
        or result.get("model_manifest_file_sha256")
        != expected_bound_inputs["final_model_manifest"]["file_sha256"]
        or result.get("training_manifest_file_sha256")
        != expected_bound_inputs["training_manifest"]["file_sha256"]
        or result.get("training_manifest_sha256") != training.get("manifest_sha256")
        or result.get("template_asset_file_sha256")
        != expected_bound_inputs["template_asset"]["file_sha256"]
        or result.get("pii_bench_report_file_sha256")
        != expected_bound_inputs["pii_bench_posthoc_report"]["file_sha256"]
        or result.get("pii_bench_report_sha256") != pii_bench_report.get("report_sha256")
        or result.get("pii_bench_posthoc") != pii_bench_summary
        or result.get("training_data_facts") != training_data_facts
        or any(
            result.get(key) != training_facts[key]
            for key in (
                "base_model_id",
                "base_model_revision",
                "base_model_license",
                "head_projection_count",
            )
        )
    ):
        raise ReleaseBuildError("typed model-card Markdown/training binding does not verify")
    final_model = _load_json_object(final_model_manifest_path, "final model manifest")
    if result.get("ordered_entity_labels") != final_model.get("ordered_labels"):
        raise ReleaseBuildError("typed model-card ordered taxonomy differs from final model")
    return artifact


def _normalize_id2label_document(document: object) -> dict[str, str]:
    if not isinstance(document, dict):
        raise ReleaseBuildError("id2label must be a JSON object")
    normalized: dict[str, str] = {}
    for key, value in document.items():
        if not isinstance(value, str) or not value.strip():
            raise ReleaseBuildError("id2label values must be non-empty strings")
        try:
            numeric_key = int(key)
        except (TypeError, ValueError) as exc:
            raise ReleaseBuildError("id2label keys must be integer-like") from exc
        if numeric_key < 0 or str(numeric_key) in normalized:
            raise ReleaseBuildError("id2label keys must be unique non-negative integers")
        normalized[str(numeric_key)] = value
    expected_keys = [str(index) for index in range(len(normalized))]
    if sorted(normalized, key=int) != expected_keys:
        raise ReleaseBuildError("id2label keys must be contiguous and start at zero")
    if len(set(normalized.values())) != len(normalized):
        raise ReleaseBuildError("id2label labels must be unique")
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def _normalized_id2label(path: Path) -> dict[str, str]:
    return _normalize_id2label_document(_load_json_object(path, "id2label"))


def build_release_config(
    config_path: Path,
    id2label_path: Path,
    *,
    community_v2_preauthorization: bool = False,
) -> dict[str, Any]:
    config = _load_json_object(config_path, "checkpoint config")
    if config.get("model_type") != "qwen3_bi":
        raise ReleaseBuildError(
            "checkpoint model_type must already be qwen3_bi; causal checkpoints cannot "
            "be converted during release"
        )
    if config.get("architectures") != ["Qwen3BiForTokenClassification"]:
        raise ReleaseBuildError(
            "checkpoint architectures must already identify Qwen3BiForTokenClassification"
        )
    required_eligibility = False if community_v2_preauthorization else True
    if (
        config.get("pii_attention_mode") != "full"
        or config.get("pii_release_eligible") is not required_eligibility
    ):
        raise ReleaseBuildError(
            "checkpoint must explicitly declare full attention and the eligibility state "
            "required by the selected release mode"
        )
    id2label = _normalized_id2label(id2label_path)
    existing = config.get("id2label")
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or {str(key): value for key, value in existing.items()} != id2label
        ):
            raise ReleaseBuildError(
                "checkpoint config id2label disagrees with release id2label.json"
            )
    existing_num_labels = config.get("num_labels")
    if existing_num_labels is not None and existing_num_labels != len(id2label):
        raise ReleaseBuildError("checkpoint config num_labels disagrees with id2label.json")

    expected_values = {
        "architectures": ["Qwen3BiForTokenClassification"],
        "auto_map": dict(AUTO_MAP),
        "id2label": id2label,
        "label2id": {label: int(index) for index, label in id2label.items()},
        "model_type": "qwen3_bi",
        "use_cache": False,
    }
    mismatched = sorted(key for key, value in expected_values.items() if config.get(key) != value)
    if mismatched:
        raise ReleaseBuildError(
            "checkpoint config is not already release-ready; mismatched fields: "
            + ", ".join(mismatched)
        )
    return config


def validate_bidirectional_release_contract(
    *,
    training_manifest_path: Path,
    tokenizer_config_path: Path,
    checkpoint_dir: Path,
    community_v2_preauthorization: bool = False,
) -> None:
    """Bind release inputs to the trained full-attention tokenizer contract."""

    training = _load_json_object(training_manifest_path, "training manifest")
    try:
        assert_document_allowed(
            training,
            purpose="ancestry" if community_v2_preauthorization else "release",
        )
    except ArtifactPolicyError as exc:
        raise ReleaseBuildError("training manifest is denied by artifact lineage policy") from exc
    if training.get("attention_mode") != "full":
        raise ReleaseBuildError("training_manifest attention_mode must be full for publication")
    tokenizer_config = _load_json_object(tokenizer_config_path, "tokenizer config")
    if tokenizer_config.get(BOUNDARY_MODE_CONFIG_KEY) != BOUNDARY_MODE:
        raise ReleaseBuildError("tokenizer config is missing the approved character-boundary mode")
    tokenizer_evidence = training.get("tokenizer")
    effective = (
        tokenizer_evidence.get("effective") if isinstance(tokenizer_evidence, dict) else None
    )
    if not isinstance(effective, dict) or effective.get("boundary_mode") != BOUNDARY_MODE:
        raise ReleaseBuildError(
            "training manifest is not bound to the approved character-boundary tokenizer"
        )
    validate_training_artifact_binding(training_manifest_path, checkpoint_dir)


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ReleaseBuildError("could not extract reviewed remote-code definition")
    return segment


def _replace_remote_code_contract_once(
    rendered: str,
    *,
    old: str,
    new: str,
    field: str,
) -> str:
    if rendered.count(old) != 1:
        raise ReleaseBuildError(f"community remote-code {field} source pattern drifted")
    return rendered.replace(old, new, 1)


def render_remote_code(
    source_path: Path,
    *,
    community_v2_preauthorization: bool = False,
) -> tuple[str, str]:
    """Split the repository implementation into HF configuration/model modules."""

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = {
        getattr(node, "name", ""): node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef))
    }
    constants: dict[str, ast.AST] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "ARCHITECTURE_VERSION",
                "SUPPORTED_ATTENTION_BACKENDS",
            }:
                constants[target.id] = node
    required_definitions = {
        "Qwen3BiConfig",
        "build_full_attention_mask",
        "convert_qwen3_token_classifier_state_dict",
        "Qwen3BiForTokenClassification",
    }
    missing = sorted(required_definitions - definitions.keys())
    missing_constants = sorted(
        {"ARCHITECTURE_VERSION", "SUPPORTED_ATTENTION_BACKENDS"} - constants.keys()
    )
    if missing or missing_constants:
        raise ReleaseBuildError(
            "remote-code source is missing reviewed definitions: "
            + ", ".join([*missing, *missing_constants])
        )

    banner = (
        '"""Generated from src/pii_zh/models/qwen3_bi.py; do not edit in the model repo."""\n\n'
        "from __future__ import annotations\n\n"
    )
    configuration = banner + "from typing import Any\n\nfrom transformers import Qwen3Config\n\n"
    configuration += _source_segment(source, constants["SUPPORTED_ATTENTION_BACKENDS"]) + "\n"
    configuration += _source_segment(source, constants["ARCHITECTURE_VERSION"]) + "\n\n"
    configuration += _source_segment(source, definitions["Qwen3BiConfig"]) + "\n"

    modeling = banner
    modeling += "from collections.abc import Mapping\nfrom typing import Any\n\n"
    modeling += "import torch\nfrom torch import nn\n"
    modeling += "from transformers import Qwen3Config, Qwen3Model\n"
    modeling += "from transformers.modeling_outputs import TokenClassifierOutput\n"
    modeling += "from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel\n\n"
    modeling += (
        "from .configuration_qwen3_bi import (\n"
        "    ARCHITECTURE_VERSION,\n"
        "    Qwen3BiConfig,\n"
        "    SUPPORTED_ATTENTION_BACKENDS,\n"
        ")\n\n"
    )
    for name in (
        "build_full_attention_mask",
        "convert_qwen3_token_classifier_state_dict",
        "Qwen3BiForTokenClassification",
    ):
        modeling += _source_segment(source, definitions[name]) + "\n\n"

    if community_v2_preauthorization:
        configuration = _replace_remote_code_contract_once(
            configuration,
            old='    model_type = "qwen3_bi"',
            new=(
                "    # Transformers must not instantiate this false-only config without "
                "checkpoint values.\n"
                "    has_no_defaults_at_init = True\n"
                '    model_type = "qwen3_bi"'
            ),
            field="configuration no-default construction",
        )
        configuration = _replace_remote_code_contract_once(
            configuration,
            old="        self.pii_release_eligible = True",
            new=(
                '        community_release_eligible = kwargs.get("pii_release_eligible")\n'
                "        if community_release_eligible is not False:\n"
                "            raise ValueError(\n"
                '                "Community-v2 Qwen3BiConfig requires an explicit "\n'
                '                "pii_release_eligible=false contract."\n'
                "            )\n"
                "        self.pii_release_eligible = community_release_eligible"
            ),
            field="configuration eligibility",
        )
        modeling = _replace_remote_code_contract_once(
            modeling,
            old='getattr(config, "pii_release_eligible", None) is True',
            new='getattr(config, "pii_release_eligible", None) is False',
            field="model compatibility eligibility",
        )

    for name, rendered in (
        ("configuration_qwen3_bi.py", configuration),
        ("modeling_qwen3_bi.py", modeling),
    ):
        try:
            compile(rendered, name, "exec")
        except SyntaxError as exc:
            raise ReleaseBuildError(f"generated {name} is invalid Python") from exc
    return configuration, modeling


def write_checksums(directory: Path) -> Path:
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "checksums.txt"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}" for path in files]
    destination = directory / "checksums.txt"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(destination, 0o644)
    return destination


def _copy_file(source: Path, destination: Path, description: str) -> None:
    _require_regular_file(source, description)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)


def validate_release_evidence_lineage(
    evidence_paths: dict[str, Path], *, community_v2_preauthorization: bool = False
) -> None:
    """Reject denied identities and unsafe semantic state in every evidence mapping."""

    for name, path in evidence_paths.items():
        try:
            if path.suffix == ".json":
                document = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix in {".yaml", ".yml"}:
                document = yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                continue
        except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ReleaseBuildError(f"release evidence {name} is not parseable") from exc
        if not isinstance(document, dict):
            raise ReleaseBuildError(f"release evidence {name} must be a mapping")
        try:
            assert_document_allowed(
                document,
                purpose="ancestry" if community_v2_preauthorization else "release",
            )
        except ArtifactPolicyError as exc:
            raise ReleaseBuildError(
                f"release evidence {name} is denied by artifact lineage policy"
            ) from exc


def build_community_v2_preauthorization(
    *,
    checkpoint_dir: Path,
    training_manifest_path: Path,
    id2label_path: Path,
    final_model_manifest_path: Path,
) -> dict[str, Any]:
    """Bind the frozen seed-97 candidate without changing its false release flags."""

    final_model = _load_json_object(final_model_manifest_path, "final model manifest")
    final_self = final_model.get("manifest_sha256")
    unsigned_final = dict(final_model)
    unsigned_final.pop("manifest_sha256", None)
    if not isinstance(final_self, str) or final_self != _canonical_json_hash(unsigned_final):
        raise ReleaseBuildError("final model manifest self hash does not verify")
    training = _load_json_object(training_manifest_path, "training manifest")
    config = _load_json_object(checkpoint_dir / "config.json", "checkpoint config")
    id2label = _normalized_id2label(id2label_path)
    values = [id2label[str(index)] for index in range(len(id2label))]
    ordered_entities = final_model.get("ordered_labels")
    expected_bio = ["O"]
    if isinstance(ordered_entities, list):
        for label in ordered_entities:
            expected_bio.extend((f"B-{label}", f"I-{label}"))
    try:
        model = model_provenance._model_identity(checkpoint_dir)
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseBuildError("community preauthorization model identity is invalid") from exc
    initialization = training.get("initialization")
    if not isinstance(initialization, dict):
        raise ReleaseBuildError("community preauthorization initialization is missing")
    if (
        final_model.get("model_id") != "aiguard24-full-seed97"
        or final_model.get("label_count") != 24
        or final_model.get("attention_mode") != "full"
        or len(ordered_entities or []) != 24
        or values != expected_bio
        or len(values) != 49
        or model.get("seed") != 97
        or model.get("label_count") != 49
        or model.get("attention_mode") != "full"
        or training.get("release_eligible") is not False
        or initialization.get("release_eligible") is not False
        or config.get("pii_release_eligible") is not False
        or model.get("training_manifest_file_sha256")
        != final_model.get("training_manifest_file_sha256")
        or model.get("training_manifest_sha256") != final_model.get("training_manifest_sha256")
        or model.get("identity_sha256") != final_model.get("model_identity_sha256")
        or model.get("output_artifact_sha256") != final_model.get("artifact_sha256")
    ):
        raise ReleaseBuildError(
            "community preauthorization differs from the exact frozen seed-97 final model"
        )
    document: dict[str, Any] = {
        "schema_version": "pii-zh.community-v2-preauthorization.v1",
        "status": COMMUNITY_PREAUTHORIZATION_STATE,
        "model_id": final_model["model_id"],
        "seed": 97,
        "entity_type_count": 24,
        "token_label_count": 49,
        "training_release_eligible": False,
        "config_release_eligible": False,
        "final_model_manifest_file_sha256": sha256_file(final_model_manifest_path),
        "final_model_manifest_sha256": final_self,
        "training_manifest_file_sha256": model["training_manifest_file_sha256"],
        "training_manifest_sha256": model["training_manifest_sha256"],
        "model_identity_sha256": model["identity_sha256"],
        "output_artifact_sha256": model["output_artifact_sha256"],
        "preauthorization_sha256": "",
    }
    document["preauthorization_sha256"] = _canonical_json_hash(
        {key: value for key, value in document.items() if key != "preauthorization_sha256"}
    )
    return document


def build_release(
    *,
    checkpoint_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    model_card: Path,
    remote_code_source: Path,
    repository_root: Path = REPOSITORY_ROOT,
    project_file_overrides: dict[str, Path] | None = None,
    model_card_v2_dir: Path | None = None,
    v2_contract_sha256: str | None = None,
    v2_selected_receipt_sha256: str | None = None,
    community_v2_preauthorization: bool = False,
    final_model_manifest: Path | None = None,
    force: bool = False,
) -> Path:
    if community_v2_preauthorization:
        raise ReleaseBuildError(
            "community-v2 packaging must use build_community_v2_release without legacy evidence"
        )
    checkpoint_dir = checkpoint_dir.resolve()
    evidence_dir = evidence_dir.resolve()
    output_dir = output_dir.resolve()
    repository_root = repository_root.resolve()
    if model_card_v2_dir is not None:
        model_card_v2_dir = model_card_v2_dir.resolve()
    if final_model_manifest is not None:
        final_model_manifest = final_model_manifest.resolve()
    protected_paths = {
        Path(output_dir.anchor),
        Path.home().resolve(),
        checkpoint_dir,
        evidence_dir,
        repository_root,
    }
    if output_dir in protected_paths or output_dir in checkpoint_dir.parents:
        raise ReleaseBuildError(f"refusing unsafe output directory: {output_dir}")
    if output_dir in evidence_dir.parents or output_dir in repository_root.parents:
        raise ReleaseBuildError(f"refusing unsafe output directory: {output_dir}")
    if checkpoint_dir in output_dir.parents or evidence_dir in output_dir.parents:
        raise ReleaseBuildError("output directory must not be nested inside checkpoint/evidence")
    if model_card_v2_dir is not None and (
        output_dir == model_card_v2_dir
        or output_dir in model_card_v2_dir.parents
        or model_card_v2_dir in output_dir.parents
    ):
        raise ReleaseBuildError("output directory must be separate from the v2 Model Card bundle")
    if repository_root / ".git" in {output_dir, *output_dir.parents}:
        raise ReleaseBuildError("output directory must not be inside repository metadata")
    if force and repository_root / "release" not in output_dir.parents:
        raise ReleaseBuildError(
            "--force is restricted to a child of the repository release directory"
        )
    _reject_contaminated_tree(checkpoint_dir, "checkpoint directory")
    _reject_contaminated_tree(evidence_dir, "evidence directory")
    v2_names_in_legacy_evidence = {
        name for name in OPTIONAL_EVIDENCE_FILES if (evidence_dir / name).exists()
    }
    if v2_names_in_legacy_evidence:
        raise ReleaseBuildError(
            "v2 evidence must come from one explicit --model-card-v2-dir bundle"
        )

    v2_bundle_paths: dict[str, Path] = {}
    if model_card_v2_dir is not None:
        _reject_contaminated_tree(model_card_v2_dir, "v2 Model Card bundle")
        expected_bundle_files = {
            *OPTIONAL_EVIDENCE_FILES,
            model_card_v2.MODEL_INDEX_FILENAME,
            model_card_v2.README_FILENAME,
        }
        actual_bundle_entries = {path.name for path in model_card_v2_dir.iterdir()}
        if actual_bundle_entries != expected_bundle_files:
            raise ReleaseBuildError(
                "v2 Model Card bundle file set differs from the generator contract: "
                f"missing={sorted(expected_bundle_files - actual_bundle_entries)}, "
                f"extra={sorted(actual_bundle_entries - expected_bundle_files)}"
            )
        v2_bundle_paths = {
            name: _require_regular_file(
                model_card_v2_dir / name, f"v2 Model Card bundle file {name}"
            )
            for name in sorted(expected_bundle_files)
        }

    checkpoint_paths = {
        name: _require_regular_file(checkpoint_dir / name, f"checkpoint file {name}")
        for name in CHECKPOINT_FILES
    }
    optional_checkpoint_paths = {
        name: _require_regular_file(checkpoint_dir / name, f"checkpoint file {name}")
        for name in OPTIONAL_CHECKPOINT_FILES
        if (checkpoint_dir / name).exists()
    }
    evidence_paths: dict[str, Path] = {}
    for name in EVIDENCE_FILES:
        source = (
            v2_bundle_paths[name]
            if v2_bundle_paths and name == model_card_v2.MODEL_INDEX_FILENAME
            else evidence_dir / name
        )
        evidence_paths[name] = _require_regular_file(source, f"release evidence {name}")
    optional_evidence_paths = {
        name: v2_bundle_paths[name] for name in OPTIONAL_EVIDENCE_FILES if v2_bundle_paths
    }
    validate_release_evidence_lineage(
        {**evidence_paths, **optional_evidence_paths},
        community_v2_preauthorization=community_v2_preauthorization,
    )
    effective_model_card = (
        v2_bundle_paths[model_card_v2.README_FILENAME] if v2_bundle_paths else model_card
    )
    _require_regular_file(effective_model_card, "model card")
    _require_regular_file(remote_code_source, "Qwen3Bi remote-code source")
    overrides = project_file_overrides or {}
    project_paths = {
        destination_name: _require_regular_file(
            overrides.get(destination_name, repository_root / source_name),
            f"project file {source_name}",
        )
        for source_name, destination_name in PROJECT_FILES.items()
    }
    validate_safetensors(checkpoint_paths["model.safetensors"])
    validate_bidirectional_release_contract(
        training_manifest_path=evidence_paths["training_manifest.json"],
        tokenizer_config_path=checkpoint_paths["tokenizer_config.json"],
        checkpoint_dir=checkpoint_dir,
        community_v2_preauthorization=community_v2_preauthorization,
    )
    build_release_config(
        checkpoint_paths["config.json"],
        evidence_paths["id2label.json"],
        community_v2_preauthorization=community_v2_preauthorization,
    )
    community_preauthorization: dict[str, Any] | None = None
    if community_v2_preauthorization:
        if final_model_manifest is None:
            raise ReleaseBuildError(
                "--community-v2-preauthorization requires --final-model-manifest"
            )
        community_preauthorization = build_community_v2_preauthorization(
            checkpoint_dir=checkpoint_dir,
            training_manifest_path=evidence_paths["training_manifest.json"],
            id2label_path=evidence_paths["id2label.json"],
            final_model_manifest_path=final_model_manifest,
        )
    elif final_model_manifest is not None:
        raise ReleaseBuildError(
            "--final-model-manifest is only accepted with --community-v2-preauthorization"
        )
    if v2_bundle_paths:
        if (
            not isinstance(v2_contract_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", v2_contract_sha256) is None
            or not isinstance(v2_selected_receipt_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", v2_selected_receipt_sha256) is None
        ):
            raise ReleaseBuildError(
                "explicit v2 build requires frozen contract and selected-receipt SHA-256 anchors"
            )
        try:
            model_card_v2.validate_release_components(
                evidence_path=optional_evidence_paths[model_card_v2.EVIDENCE_FILENAME],
                contract_path=optional_evidence_paths[model_card_v2.CONTRACT_FILENAME],
                selected_receipt_path=optional_evidence_paths[
                    model_card_v2.SELECTED_RECEIPT_FILENAME
                ],
                training_path=evidence_paths["training_manifest.json"],
                evaluation_path=evidence_paths["evaluation_report.json"],
                checkpoint_dir=checkpoint_dir,
                model_index_path=evidence_paths["model-index.yml"],
                readme_path=effective_model_card,
                bound_release_text_paths=project_paths,
                expected_contract_sha256=v2_contract_sha256,
                expected_selected_receipt_sha256=v2_selected_receipt_sha256,
            )
        except model_card_v2.ModelCardEvidenceError as exc:
            raise ReleaseBuildError(f"explicit v2 release evidence is invalid: {exc}") from exc
    configuration_code, modeling_code = render_remote_code(remote_code_source)

    if output_dir.exists() and not force:
        raise ReleaseBuildError(f"output already exists (use --force to replace it): {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _copy_file(effective_model_card, staging / "README.md", "model card")
        for name, source in checkpoint_paths.items():
            _copy_file(source, staging / name, f"checkpoint file {name}")
        for name, source in optional_checkpoint_paths.items():
            _copy_file(source, staging / name, f"checkpoint file {name}")
        for name, source in evidence_paths.items():
            _copy_file(source, staging / name, f"release evidence {name}")
        for name, source in optional_evidence_paths.items():
            _copy_file(source, staging / name, f"release evidence {name}")
        for destination_name, source in project_paths.items():
            _copy_file(
                source,
                staging / destination_name,
                f"project file {destination_name}",
            )
        if v2_bundle_paths:
            packaged_contract = _load_json_object(
                v2_bundle_paths[model_card_v2.CONTRACT_FILENAME],
                "v2 release contract",
            )
            model_card_v2.validate_bound_release_text_files(
                packaged_contract,
                {name: staging / name for name in model_card_v2.BOUND_RELEASE_TEXT_FILES},
            )
        (staging / "configuration_qwen3_bi.py").write_text(configuration_code, encoding="utf-8")
        (staging / "modeling_qwen3_bi.py").write_text(modeling_code, encoding="utf-8")
        if community_preauthorization is not None:
            (staging / COMMUNITY_PREAUTHORIZATION_FILENAME).write_text(
                json.dumps(
                    community_preauthorization,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        write_checksums(staging)
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        expected = (
            REQUIRED_RELEASE_FILES
            | optional_checkpoint_paths.keys()
            | optional_evidence_paths.keys()
            | (
                {COMMUNITY_PREAUTHORIZATION_FILENAME}
                if community_preauthorization is not None
                else set()
            )
        )
        if actual != expected:
            raise ReleaseBuildError(
                "internal error: release file set differs from the required allowlist: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        if force:
            if output_dir.exists():
                if not output_dir.is_dir() or output_dir.is_symlink():
                    raise ReleaseBuildError(f"refusing to replace unsafe output path: {output_dir}")
                shutil.rmtree(output_dir)
            os.replace(staging, output_dir)
        else:
            try:
                model_card_v2._rename_directory_no_replace(staging, output_dir)
            except model_card_v2.ModelCardEvidenceError as exc:
                raise ReleaseBuildError(str(exc)) from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def build_community_v2_release(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    model_card: Path,
    model_card_artifact: Path,
    pii_bench_report: Path,
    final_model_manifest: Path,
    remote_code_source: Path,
    repository_root: Path = REPOSITORY_ROOT,
    project_file_overrides: dict[str, Path] | None = None,
) -> Path:
    """Build the minimal seed-97 community package without legacy seed-42 evidence."""

    checkpoint_dir = _reject_symlink_path_or_ancestry(
        checkpoint_dir, "community checkpoint directory"
    )
    output_dir = _reject_symlink_path_or_ancestry(output_dir, "community output path")
    repository_root = _reject_symlink_path_or_ancestry(repository_root, "repository root")
    final_model_manifest = _reject_symlink_path_or_ancestry(
        final_model_manifest, "final model manifest"
    )
    model_card = _reject_symlink_path_or_ancestry(model_card, "typed model-card Markdown companion")
    model_card_artifact = _reject_symlink_path_or_ancestry(
        model_card_artifact, "typed model-card artifact"
    )
    pii_bench_report = _reject_symlink_path_or_ancestry(
        pii_bench_report, "PII Bench ZH posthoc report"
    )
    remote_code_source = _reject_symlink_path_or_ancestry(
        remote_code_source, "Qwen3Bi remote-code source"
    )
    current_repository_root = REPOSITORY_ROOT.resolve()
    if repository_root != current_repository_root:
        raise ReleaseBuildError("community-v2 packaging is locked to the current repository")
    expected_remote_source = repository_root / "src/pii_zh/models/qwen3_bi.py"
    if remote_code_source != expected_remote_source:
        raise ReleaseBuildError("community-v2 packaging forbids an alternate remote-code source")
    if project_file_overrides:
        raise ReleaseBuildError("community-v2 packaging forbids legal/security file overrides")
    protected_paths = {
        Path(output_dir.anchor),
        Path.home().resolve(),
        checkpoint_dir,
        repository_root,
    }
    if (
        output_dir in protected_paths
        or output_dir in checkpoint_dir.parents
        or output_dir in repository_root.parents
        or checkpoint_dir in output_dir.parents
    ):
        raise ReleaseBuildError(f"refusing unsafe community output directory: {output_dir}")
    if repository_root / ".git" in {output_dir, *output_dir.parents}:
        raise ReleaseBuildError("community output directory must not be inside repository metadata")
    if output_dir.exists() or output_dir.is_symlink():
        raise ReleaseBuildError("community output already exists; no-clobber is mandatory")

    _reject_contaminated_tree(checkpoint_dir, "community checkpoint directory")
    checkpoint_paths = {
        name: _require_regular_file(checkpoint_dir / name, f"checkpoint file {name}")
        for name in CHECKPOINT_FILES
    }
    optional_checkpoint_paths = {
        name: _require_regular_file(checkpoint_dir / name, f"checkpoint file {name}")
        for name in OPTIONAL_CHECKPOINT_FILES
        if (checkpoint_dir / name).exists()
    }
    if optional_checkpoint_paths:
        raise ReleaseBuildError(
            "community seed-97 checkpoint unexpectedly contains optional tokenizer files"
        )
    training_manifest = _require_regular_file(
        checkpoint_dir / "training_manifest.json", "seed-97 training manifest"
    )
    taxonomy_path = _require_regular_file(
        repository_root / "src/pii_zh/taxonomy/taxonomy.yaml", "repository taxonomy"
    )
    _require_regular_file(model_card, "typed model-card Markdown companion")
    _require_regular_file(model_card_artifact, "typed model-card artifact")
    _require_regular_file(pii_bench_report, "PII Bench ZH posthoc report")
    _require_regular_file(final_model_manifest, "final model manifest")
    _require_regular_file(remote_code_source, "Qwen3Bi remote-code source")
    project_paths = {
        destination_name: _require_regular_file(
            repository_root / source_name,
            f"project file {source_name}",
        )
        for source_name, destination_name in PROJECT_FILES.items()
        if destination_name in {"LICENSE", "SECURITY.md"}
    }

    config = _load_json_object(checkpoint_paths["config.json"], "checkpoint config")
    id2label = _normalize_id2label_document(config.get("id2label"))
    try:
        taxonomy = load_taxonomy(taxonomy_path)
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseBuildError("repository taxonomy is invalid") from exc
    expected_labels = [taxonomy.outside_label]
    for entity in taxonomy.label_sets["core"]:
        expected_labels.extend((f"B-{entity.name}", f"I-{entity.name}"))
    if (
        len(expected_labels) != 49
        or [id2label[str(index)] for index in range(len(id2label))] != expected_labels
    ):
        raise ReleaseBuildError("community config is not exact repository core-24 BIO")

    validate_safetensors(checkpoint_paths["model.safetensors"])
    validate_bidirectional_release_contract(
        training_manifest_path=training_manifest,
        tokenizer_config_path=checkpoint_paths["tokenizer_config.json"],
        checkpoint_dir=checkpoint_dir,
        community_v2_preauthorization=True,
    )
    validate_release_evidence_lineage(
        {"training_manifest.json": training_manifest, "taxonomy.yaml": taxonomy_path},
        community_v2_preauthorization=True,
    )
    training = _load_json_object(training_manifest, "seed-97 training manifest")
    validate_typed_model_card_companion(
        model_card_artifact_path=model_card_artifact,
        markdown_path=model_card,
        final_model_manifest_path=final_model_manifest,
        training_manifest_path=training_manifest,
        pii_bench_report_path=pii_bench_report,
        repository_root=repository_root,
    )
    configuration_code, modeling_code = render_remote_code(
        remote_code_source,
        community_v2_preauthorization=True,
    )
    notice = render_community_v2_notice(training)
    third_party_notices = render_community_v2_third_party_notices(training)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        id2label_path = staging / "id2label.json"
        id2label_path.write_text(
            json.dumps(id2label, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(id2label_path, 0o644)
        build_release_config(
            checkpoint_paths["config.json"],
            id2label_path,
            community_v2_preauthorization=True,
        )
        preauthorization = build_community_v2_preauthorization(
            checkpoint_dir=checkpoint_dir,
            training_manifest_path=training_manifest,
            id2label_path=id2label_path,
            final_model_manifest_path=final_model_manifest,
        )

        _copy_file(model_card, staging / "README.md", "typed model-card Markdown companion")
        for name, source in checkpoint_paths.items():
            _copy_file(source, staging / name, f"checkpoint file {name}")
        for name, source in optional_checkpoint_paths.items():
            _copy_file(source, staging / name, f"checkpoint file {name}")
        _copy_file(training_manifest, staging / "training_manifest.json", "training manifest")
        _copy_file(taxonomy_path, staging / "taxonomy.yaml", "repository taxonomy")
        for destination_name, source in project_paths.items():
            _copy_file(source, staging / destination_name, f"project file {destination_name}")
        generated_payloads = {
            "NOTICE": notice,
            "THIRD_PARTY_NOTICES.md": third_party_notices,
            "configuration_qwen3_bi.py": configuration_code,
            "modeling_qwen3_bi.py": modeling_code,
        }
        for name, payload in generated_payloads.items():
            destination = staging / name
            destination.write_text(payload, encoding="utf-8")
            os.chmod(destination, 0o644)
        preauthorization_path = staging / COMMUNITY_PREAUTHORIZATION_FILENAME
        preauthorization_path.write_text(
            json.dumps(preauthorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(preauthorization_path, 0o644)
        write_checksums(staging)
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        expected = COMMUNITY_V2_REQUIRED_RELEASE_FILES
        if actual != expected:
            raise ReleaseBuildError(
                "internal error: community package differs from the minimal allowlist: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        try:
            model_card_v2._rename_directory_no_replace(staging, output_dir)
        except model_card_v2.ModelCardEvidenceError as exc:
            raise ReleaseBuildError(str(exc)) from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="legacy release evidence directory; forbidden in community-v2 preauthorization mode",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument(
        "--model-card-artifact",
        type=Path,
        help="typed model-card JSON emitted together with the Markdown companion",
    )
    parser.add_argument(
        "--pii-bench-report",
        type=Path,
        help="frozen seed-97 PII Bench ZH posthoc descriptive report",
    )
    parser.add_argument(
        "--model-card-v2-dir",
        type=Path,
        help="explicit five-file bundle emitted by build_model_card_evidence_v2.py",
    )
    parser.add_argument(
        "--remote-code-source",
        type=Path,
        default=REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--license-file", type=Path)
    parser.add_argument("--notice-file", type=Path)
    parser.add_argument("--third-party-notices", type=Path)
    parser.add_argument("--security-policy", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--v2-contract-sha256")
    parser.add_argument("--v2-selected-receipt-sha256")
    parser.add_argument(
        "--community-v2-preauthorization",
        action="store_true",
        help=(
            "package the exact frozen seed-97 false-eligible candidate for local "
            "community-contract review"
        ),
    )
    parser.add_argument("--final-model-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        project_overrides = {
            name: path
            for name, path in {
                "LICENSE": args.license_file,
                "NOTICE": args.notice_file,
                "THIRD_PARTY_NOTICES.md": args.third_party_notices,
                "SECURITY.md": args.security_policy,
            }.items()
            if path is not None
        }
        if args.community_v2_preauthorization:
            if (
                args.evidence_dir is not None
                or args.model_card_v2_dir is not None
                or args.v2_contract_sha256 is not None
                or args.v2_selected_receipt_sha256 is not None
                or args.force
            ):
                raise ReleaseBuildError(
                    "community-v2 packaging forbids legacy evidence/bundle anchors and --force"
                )
            if args.final_model_manifest is None:
                raise ReleaseBuildError(
                    "--community-v2-preauthorization requires --final-model-manifest"
                )
            if (
                args.model_card is None
                or args.model_card_artifact is None
                or args.pii_bench_report is None
            ):
                raise ReleaseBuildError(
                    "community-v2 packaging requires --model-card, --model-card-artifact and "
                    "--pii-bench-report"
                )
            output = build_community_v2_release(
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir,
                model_card=args.model_card,
                model_card_artifact=args.model_card_artifact,
                pii_bench_report=args.pii_bench_report,
                final_model_manifest=args.final_model_manifest,
                remote_code_source=args.remote_code_source,
                repository_root=args.repository_root,
                project_file_overrides=project_overrides,
            )
        else:
            if args.model_card_artifact is not None or args.pii_bench_report is not None:
                raise ReleaseBuildError(
                    "legacy release mode forbids community model-card/PII-Bench artifacts"
                )
            if args.evidence_dir is None:
                raise ReleaseBuildError("legacy release mode requires --evidence-dir")
            model_card = args.model_card or REPOSITORY_ROOT / "model_cards/README.md"
            output = build_release(
                checkpoint_dir=args.checkpoint_dir,
                evidence_dir=args.evidence_dir,
                output_dir=args.output_dir,
                model_card=model_card,
                remote_code_source=args.remote_code_source,
                repository_root=args.repository_root,
                project_file_overrides=project_overrides,
                model_card_v2_dir=args.model_card_v2_dir,
                v2_contract_sha256=args.v2_contract_sha256,
                v2_selected_receipt_sha256=args.v2_selected_receipt_sha256,
                community_v2_preauthorization=False,
                final_model_manifest=args.final_model_manifest,
                force=args.force,
            )
    except (OSError, ReleaseBuildError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"release build blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Built local release package: {output}")
    print("No upload was performed. Run scripts/release_gate.py before any publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
