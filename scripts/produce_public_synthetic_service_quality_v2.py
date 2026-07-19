#!/usr/bin/env python3
"""Produce or strictly replay an aggregate-only Open-24 community quality receipt.

The producer is intentionally record-reading while its output is not: gold and
prediction JSONL are reduced to aggregate counts, hashes and paired-bootstrap
statistics.  Candidate and comparator prediction-provenance manifests are
metadata-validated and content-addressed; their origin replays remain an
explicitly separate release gate.  ``replay`` requires the same frozen inputs,
recomputes every field, and compares the complete receipt.  It never reads model
weights, queries a GPU, uses the network, or reads PII Bench.

This is a successor execution surface for the thresholds preregistered in
``public_synthetic_service_quality_v1.json``.  The v1 comparator registry is a
closed-eight PII Bench registry and is therefore deliberately not reused as an
Open-24 comparator claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

from pii_zh.evaluation.io import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    Span,
    load_gold_jsonl,
    load_prediction_jsonl,
)
from pii_zh.evaluation.metrics import evaluate_documents
from pii_zh.evaluation.paired_bootstrap import StrictCounts
from pii_zh.evaluation.paired_bootstrap_three_metric import (
    ThreeMetricPairedSuite,
    build_three_metric_paired_suite,
    three_metric_intersection_union_bootstrap,
)
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS

try:
    from scripts.validate_public_synthetic_service_quality_v1 import (
        CONTRACT_ID as THRESHOLD_CONTRACT_ID,
    )
    from scripts.validate_public_synthetic_service_quality_v1 import (
        CONTRACT_PATH as THRESHOLD_CONTRACT_PATH,
    )
    from scripts.validate_public_synthetic_service_quality_v1 import (
        validate_preregistration,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    from validate_public_synthetic_service_quality_v1 import (  # type: ignore[no-redef]
        CONTRACT_ID as THRESHOLD_CONTRACT_ID,
    )
    from validate_public_synthetic_service_quality_v1 import (
        CONTRACT_PATH as THRESHOLD_CONTRACT_PATH,
    )
    from validate_public_synthetic_service_quality_v1 import (
        validate_preregistration,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = "configs/evaluation/public_synthetic_service_quality_result_v2.schema.json"
REGISTRY_SCHEMA_PATH = (
    "configs/evaluation/public_synthetic_quality_comparator_registry_v2.schema.json"
)
COMPARATOR_PREDICTION_SCHEMA_PATH = (
    "configs/evaluation/open24_comparator_prediction_provenance_v1.schema.json"
)
CANDIDATE_PREDICTION_SCHEMA_PATH = (
    "configs/evaluation/release_eval_v2_prediction_provenance_v1.schema.json"
)
CANDIDATE_PREDICTION_MODULE_PATH = "src/pii_zh/evaluation/release_eval_v2_prediction_provenance.py"
CANDIDATE_PREDICTION_CLI_PATH = "scripts/release_eval_v2_prediction_provenance.py"
PRODUCER_PATH = "scripts/produce_public_synthetic_service_quality_v2.py"
RESULT_SCHEMA_VERSION = "pii-zh.public-synthetic-service-quality-result.v2"
REGISTRY_SCHEMA_VERSION = "pii-zh.public-synthetic-quality-comparator-registry.v2"
MODEL_MANIFEST_SCHEMA_VERSION = "pii-zh.community-final-model-binding.v2"
SERVICE_MANIFEST_SCHEMA_VERSION = "pii-zh.community-service-configuration-binding.v2"
PERFORMANCE_SCHEMA_VERSION = "pii-zh.community-quality-performance-binding.v2"
TRACKS = ("model_raw", "full_system")
MIN_BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
DEFAULT_SEED = 42
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 1024 * 1024 * 1024


class CommunityQualityProducerError(RuntimeError):
    """A fail-closed producer/replay error without record values."""


@dataclass(frozen=True, slots=True)
class TrackInput:
    """Files supplied for one canonical candidate/comparator track."""

    track: str
    candidate_predictions: Path
    candidate_prediction_manifest: Path
    comparator_predictions: Path
    comparator_prediction_manifest: Path
    performance_manifest: Path


@dataclass(frozen=True, slots=True)
class ProducerInputs:
    """Complete frozen input inventory for production or replay."""

    gold: Path
    gold_dataset_manifest: Path
    gold_materialization_receipt: Path
    gold_freeze_receipt: Path
    gold_split: str
    final_model_manifest: Path
    service_configuration: Path
    comparator_registry: Path
    tracks: tuple[TrackInput, ...]
    internal_unlock: Path | None = None
    final_model_binding: Path | None = None
    service_configuration_binding: Path | None = None


def _formal_internal_preopen_guard(
    inputs: ProducerInputs,
) -> tuple[dict[str, Any], bytes]:
    """Validate the unlock before the formal internal gold is opened."""

    manifest, payload = _load_json_file(
        inputs.gold_dataset_manifest, field="gold dataset manifest"
    )
    from pii_zh.evaluation.release_eval_v2_prediction_provenance import FROZEN_V2

    if _sha256(payload) == FROZEN_V2["dataset_manifest_file_sha256"]:
        if (
            inputs.internal_unlock is None
            or inputs.final_model_binding is None
            or inputs.service_configuration_binding is None
        ):
            raise CommunityQualityProducerError(
                "formal internal quality requires unlock and final binding receipts"
            )
        from pii_zh.evaluation.release_eval_v2_stage_gate import (
            ReleaseEvalV2StageGateError,
            guard_internal_preopen,
        )

        try:
            guard_internal_preopen(
                inputs.internal_unlock,
                inputs.final_model_binding,
                inputs.service_configuration_binding,
                input_path=inputs.gold,
                mode="quality",
            )
        except ReleaseEvalV2StageGateError as exc:
            raise CommunityQualityProducerError(
                "formal internal pre-open gate failed"
            ) from exc
    return manifest, payload


def _canonical_json_bytes(value: Mapping[str, Any], *, remove: str | None = None) -> bytes:
    document = dict(value)
    if remove is not None:
        document.pop(remove, None)
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommunityQualityProducerError("JSON value is not canonicalizable") from exc


def canonical_json_hash(value: Mapping[str, Any], *, remove: str | None = None) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommunityQualityProducerError("receipt is not strict JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, field: str, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CommunityQualityProducerError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise CommunityQualityProducerError(f"{field} must be a bounded regular file")
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise CommunityQualityProducerError(f"{field} exceeds the size limit")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(payload) != after.st_size:
            raise CommunityQualityProducerError(f"{field} changed while it was read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommunityQualityProducerError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _load_json_payload(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                CommunityQualityProducerError("JSON contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommunityQualityProducerError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CommunityQualityProducerError(f"{field} must be a JSON object")
    return value


def _load_json_file(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, field=field, maximum_bytes=_MAX_JSON_BYTES)
    return _load_json_payload(payload, field=field), payload


def _exact_keys(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CommunityQualityProducerError(f"{field} has an invalid closed shape")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CommunityQualityProducerError(f"{field} is not a path-free stable identifier")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CommunityQualityProducerError(f"{field} must be a lowercase SHA-256")
    return value


def _nonnegative_number(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommunityQualityProducerError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise CommunityQualityProducerError(f"{field} is outside its allowed range")
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CommunityQualityProducerError(f"{field} must be a positive integer")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommunityQualityProducerError(f"{field} must be an object")
    return value


def _required_keys(
    value: object,
    required: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    result = _mapping(value, field=field)
    if not required <= set(result):
        raise CommunityQualityProducerError(f"{field} is missing required provenance fields")
    return result


def _validate_gold_provenance(
    *,
    dataset_manifest: Mapping[str, Any],
    dataset_manifest_payload: bytes,
    materialization_receipt: Mapping[str, Any],
    materialization_receipt_payload: bytes,
    freeze_receipt: Mapping[str, Any],
    freeze_receipt_payload: bytes,
    freeze_receipt_path: Path,
    gold_path: Path,
    gold_file_sha256: str,
    gold_size_bytes: int,
    gold_split: str,
    gold_document_count: int,
) -> dict[str, int | str]:
    """Validate a generic evaluation-only manifest/materialization/freeze chain."""

    if gold_split != "internal_evaluation":
        raise CommunityQualityProducerError(
            "Open-24 final quality accepts only the one-shot internal_evaluation split"
        )
    manifest = _required_keys(
        dataset_manifest,
        {
            "manifest_schema_version",
            "dataset_id",
            "dataset_version",
            "license",
            "language",
            "publication_pool",
            "record_data_pool",
            "public_weight_training_allowed",
            "public_artifact_release_allowed",
            "config",
            "freeze",
            "generation",
            "input_policy",
            "source",
            "usage_policy",
            "counts",
            "audits",
            "files",
            "manifest_sha256",
        },
        field="gold dataset manifest",
    )
    schema_version = manifest["manifest_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 2
    ):
        raise CommunityQualityProducerError(
            "gold dataset manifest must use provenance schema version 2 or newer"
        )
    dataset_id = _safe_id(manifest["dataset_id"], field="gold dataset_id")
    dataset_version = _safe_id(manifest["dataset_version"], field="gold dataset_version")
    manifest_self_hash = _digest(
        manifest["manifest_sha256"], field="gold dataset manifest self hash"
    )
    if manifest_self_hash != canonical_json_hash(manifest, remove="manifest_sha256"):
        raise CommunityQualityProducerError("gold dataset manifest self hash does not verify")
    if (
        manifest["language"] != "zh-Hans-CN"
        or manifest["publication_pool"] != "public_release_pool"
        or manifest["record_data_pool"] != "evaluation_only"
        or manifest["public_weight_training_allowed"] is not False
        or manifest["public_artifact_release_allowed"] is not True
        or not isinstance(manifest["license"], str)
        or not manifest["license"].strip()
    ):
        raise CommunityQualityProducerError(
            "gold dataset manifest violates the public evaluation-only policy"
        )

    generation = _required_keys(
        manifest["generation"],
        {
            "contains_real_personal_data",
            "protected_evaluation_data_read",
            "jsonl_rows_printed",
            "generation_salt_sha256",
            "external_dataset_inputs",
        },
        field="gold generation provenance",
    )
    generation_salt = _digest(generation["generation_salt_sha256"], field="gold generation salt")
    if (
        generation["contains_real_personal_data"] is not False
        or generation["protected_evaluation_data_read"] is not False
        or generation["jsonl_rows_printed"] is not False
        or generation["external_dataset_inputs"] != []
    ):
        raise CommunityQualityProducerError("gold generation provenance is not privacy-safe")
    input_policy = _required_keys(
        manifest["input_policy"],
        {"external_dataset_inputs", "forbidden_sources_read"},
        field="gold input policy",
    )
    if (
        input_policy["external_dataset_inputs"] != []
        or input_policy["forbidden_sources_read"] != []
    ):
        raise CommunityQualityProducerError(
            "gold input policy admits an unverified external or forbidden source"
        )
    source = _required_keys(
        manifest["source"], {"source_id", "declared_license"}, field="gold source"
    )
    _safe_id(source["source_id"], field="gold source_id")
    if source["declared_license"] != manifest["license"]:
        raise CommunityQualityProducerError("gold source license does not match the manifest")

    usage = _required_keys(manifest["usage_policy"], {gold_split}, field="gold usage policy")
    split_usage = _required_keys(
        usage[gold_split],
        {
            "gradient_training_allowed",
            "checkpoint_or_model_selection_allowed",
            "threshold_tuning_allowed",
            "temperature_scaling_allowed",
            "fusion_calibration_allowed",
            "final_metric_claim_allowed",
            "result_informed_model_or_threshold_change_allowed",
            "content_read_before_final_freeze_allowed",
        },
        field=f"gold usage policy {gold_split}",
    )
    if (
        split_usage["gradient_training_allowed"] is not False
        or split_usage["checkpoint_or_model_selection_allowed"] is not False
        or split_usage["threshold_tuning_allowed"] is not False
        or split_usage["temperature_scaling_allowed"] is not False
        or split_usage["fusion_calibration_allowed"] is not False
        or split_usage["final_metric_claim_allowed"] is not True
        or split_usage["result_informed_model_or_threshold_change_allowed"] is not False
        or split_usage["content_read_before_final_freeze_allowed"] is not False
    ):
        raise CommunityQualityProducerError("gold split is not a frozen one-shot final split")

    counts = _required_keys(manifest["counts"], {"by_split", "core_labels"}, field="gold counts")
    split_counts = _mapping(counts["by_split"], field="gold split counts")
    if counts["core_labels"] != 24 or split_counts.get(gold_split) != gold_document_count:
        raise CommunityQualityProducerError("gold manifest counts do not match Open-24 input")
    audits = _required_keys(
        manifest["audits"],
        {
            "core_label_count",
            "pii_free_counts_by_split",
            "label_counts_by_split",
            "cross_corpus_isolation",
        },
        field="gold audits",
    )
    pii_free_counts = _mapping(audits["pii_free_counts_by_split"], field="gold PII-free counts")
    labels_by_split = _mapping(audits["label_counts_by_split"], field="gold label counts")
    split_label_counts = _mapping(labels_by_split.get(gold_split), field="gold split label counts")
    isolation = _required_keys(
        audits["cross_corpus_isolation"],
        {"all_collision_counts_zero", "comparisons"},
        field="gold cross-corpus isolation",
    )
    comparisons = _mapping(isolation["comparisons"], field="gold isolation comparisons")
    if (
        audits["core_label_count"] != 24
        or set(split_label_counts) != set(PII_CORE_LABELS)
        or any(
            _positive_integer(value, field="gold label support") < 1
            for value in split_label_counts.values()
        )
        or _positive_integer(pii_free_counts.get(gold_split), field="gold PII-free support") < 1
        or isolation["all_collision_counts_zero"] is not True
        or not comparisons
        or any(
            not isinstance(values, Mapping)
            or not values
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count != 0
                for count in values.values()
            )
            for values in comparisons.values()
        )
    ):
        raise CommunityQualityProducerError("gold coverage/isolation audit is invalid")

    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise CommunityQualityProducerError("gold dataset manifest files must be a list")
    files_by_split: dict[str, Mapping[str, Any]] = {}
    for index, raw_file in enumerate(raw_files):
        item = _exact_keys(
            raw_file,
            {"name", "split", "records", "bytes", "sha256", "mode"},
            field=f"gold dataset file {index}",
        )
        split = _safe_id(item["split"], field=f"gold dataset file {index} split")
        if split in files_by_split:
            raise CommunityQualityProducerError("gold dataset manifest has duplicate splits")
        if item["name"] != f"{split}.jsonl" or item["mode"] != "0444":
            raise CommunityQualityProducerError("gold dataset file identity/mode is invalid")
        _positive_integer(item["records"], field=f"gold dataset file {index} records")
        _positive_integer(item["bytes"], field=f"gold dataset file {index} bytes")
        _digest(item["sha256"], field=f"gold dataset file {index} hash")
        files_by_split[split] = item
    selected_file = files_by_split.get(gold_split)
    if selected_file is None:
        raise CommunityQualityProducerError("gold split is absent from the dataset manifest")
    if (
        selected_file["name"] != gold_path.name
        or selected_file["records"] != gold_document_count
        or selected_file["bytes"] != gold_size_bytes
        or selected_file["sha256"] != gold_file_sha256
    ):
        raise CommunityQualityProducerError("gold JSONL does not match its dataset manifest")

    freeze = _required_keys(
        manifest["freeze"],
        {
            "frozen_at_utc",
            "frozen_before_any_v3_seed42_validation_result",
            "supersession_freeze_receipt",
        },
        field="gold freeze",
    )
    freeze_binding = _exact_keys(
        freeze["supersession_freeze_receipt"],
        {"path", "file_sha256", "receipt_id", "frozen_at_utc", "chronology"},
        field="gold freeze receipt binding",
    )
    freeze_file_hash = _digest(freeze_binding["file_sha256"], field="gold freeze receipt file hash")
    raw_freeze_path = freeze_binding["path"]
    if not isinstance(raw_freeze_path, str):
        raise CommunityQualityProducerError("gold freeze receipt path must be relative")
    normalized_freeze_path = PurePosixPath(raw_freeze_path)
    if (
        normalized_freeze_path.is_absolute()
        or not normalized_freeze_path.parts
        or any(part in {"", ".", ".."} for part in normalized_freeze_path.parts)
        or normalized_freeze_path.suffix != ".json"
        or normalized_freeze_path.name != freeze_receipt_path.name
    ):
        raise CommunityQualityProducerError(
            "gold freeze receipt path is not a normalized repository-relative JSON path"
        )
    if freeze_file_hash != _sha256(freeze_receipt_payload):
        raise CommunityQualityProducerError("gold freeze receipt bytes do not match the manifest")
    if (
        freeze["frozen_before_any_v3_seed42_validation_result"] is not True
        or freeze_binding["receipt_id"] != freeze_receipt.get("receipt_id")
        or freeze_binding["frozen_at_utc"] != freeze_receipt.get("frozen_at_utc")
        or freeze_binding["chronology"] != freeze_receipt.get("chronology")
        or freeze["frozen_at_utc"] != freeze_receipt.get("frozen_at_utc")
    ):
        raise CommunityQualityProducerError("gold freeze receipt semantic binding failed")
    frozen_design = _required_keys(
        freeze_receipt.get("frozen_design"),
        {
            "config_file_sha256",
            "generation_salt_sha256",
            "internal_evaluation_records",
            "core_label_count",
            "record_data_pool",
            "materialization_pending_at_freeze",
        },
        field="gold freeze design",
    )
    frozen_supersession = _required_keys(
        freeze_receipt.get("supersession"),
        {"replacement_dataset_id", "replacement_dataset_version"},
        field="gold freeze supersession",
    )
    content_boundary = _required_keys(
        freeze_receipt.get("content_access_boundary"),
        {
            "jsonl_content_read_or_printed_by_freezing_agent",
            "v1_or_v2_model_predictions_read",
            "v1_or_v2_model_metrics_read",
        },
        field="gold freeze content boundary",
    )
    freeze_protocol = _required_keys(
        freeze_receipt.get("protocol_effect"),
        {
            "v2_internal_evaluation_content_must_remain_unread_until_model_service_and_calibration_are_frozen",
            "does_not_modify_frozen_v3_training_protocol_or_selector",
        },
        field="gold freeze protocol effect",
    )
    config = _required_keys(manifest["config"], {"sha256"}, field="gold config")
    config_hash = _digest(config["sha256"], field="gold config hash")
    if (
        frozen_design["config_file_sha256"] != config_hash
        or frozen_design["generation_salt_sha256"] != generation_salt
        or frozen_design["internal_evaluation_records"] != gold_document_count
        or frozen_design["core_label_count"] != 24
        or frozen_design["record_data_pool"] != "evaluation_only"
        or frozen_design["materialization_pending_at_freeze"] is not True
        or frozen_supersession["replacement_dataset_id"] != dataset_id
        or frozen_supersession["replacement_dataset_version"] != dataset_version
        or content_boundary["jsonl_content_read_or_printed_by_freezing_agent"] is not False
        or content_boundary["v1_or_v2_model_predictions_read"] is not False
        or content_boundary["v1_or_v2_model_metrics_read"] is not False
        or freeze_protocol[
            "v2_internal_evaluation_content_must_remain_unread_until_model_service_and_calibration_are_frozen"
        ]
        is not True
        or freeze_protocol["does_not_modify_frozen_v3_training_protocol_or_selector"] is not True
    ):
        raise CommunityQualityProducerError("gold freeze design does not match the manifest")

    receipt = _exact_keys(
        materialization_receipt,
        {
            "receipt_schema_version",
            "receipt_id",
            "frozen_at_utc",
            "freeze_receipt",
            "chronology",
            "replacement",
            "verification",
            "protocol_effect",
            "receipt_sha256",
        },
        field="gold materialization receipt",
    )
    receipt_self_hash = _digest(
        receipt["receipt_sha256"], field="gold materialization receipt self hash"
    )
    if receipt_self_hash != canonical_json_hash(receipt, remove="receipt_sha256"):
        raise CommunityQualityProducerError("gold materialization receipt self hash failed")
    _safe_id(receipt["receipt_id"], field="gold materialization receipt_id")
    replacement = _required_keys(
        receipt["replacement"],
        {
            "replacement_dataset_id",
            "replacement_dataset_version",
            "config_file_sha256",
            "generation_salt_sha256",
            "dataset_manifest_file_sha256",
            "dataset_manifest_sha256",
            "files",
        },
        field="gold materialization replacement",
    )
    verification = _required_keys(
        receipt["verification"],
        {
            "record_data_pool",
            "public_weight_training_allowed",
            "internal_evaluation_records",
            "pii_free_counts_by_split",
            "core_label_count_each_split",
            "all_required_collision_counts_zero",
            "jsonl_rows_printed_by_materializer",
            "protected_public_benchmark_read",
        },
        field="gold materialization verification",
    )
    protocol_effect = _required_keys(
        receipt["protocol_effect"],
        {
            "v2_internal_evaluation_is_one_shot_after_full_freeze",
            "frozen_v3_training_protocol_or_selector_modified",
        },
        field="gold materialization protocol effect",
    )
    receipt_pii_free = _mapping(
        verification["pii_free_counts_by_split"],
        field="gold materialization PII-free counts",
    )
    if (
        receipt["frozen_at_utc"] != freeze["frozen_at_utc"]
        or receipt["freeze_receipt"] != freeze_binding
        or receipt["chronology"] != freeze_binding["chronology"]
        or replacement["replacement_dataset_id"] != dataset_id
        or replacement["replacement_dataset_version"] != dataset_version
        or replacement["config_file_sha256"] != config_hash
        or replacement["generation_salt_sha256"] != generation_salt
        or replacement["dataset_manifest_file_sha256"] != _sha256(dataset_manifest_payload)
        or replacement["dataset_manifest_sha256"] != manifest_self_hash
        or replacement["files"] != raw_files
        or verification["record_data_pool"] != "evaluation_only"
        or verification["public_weight_training_allowed"] is not False
        or verification["internal_evaluation_records"] != gold_document_count
        or verification["core_label_count_each_split"] != 24
        or verification["all_required_collision_counts_zero"] is not True
        or verification["jsonl_rows_printed_by_materializer"] is not False
        or verification["protected_public_benchmark_read"] is not False
        or receipt_pii_free.get(gold_split) != pii_free_counts.get(gold_split)
        or protocol_effect["v2_internal_evaluation_is_one_shot_after_full_freeze"] is not True
        or protocol_effect["frozen_v3_training_protocol_or_selector_modified"] is not False
    ):
        raise CommunityQualityProducerError(
            "gold materialization receipt does not bind the frozen dataset"
        )

    return {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "gold_split": gold_split,
        "document_count": gold_document_count,
        "dataset_manifest_file_sha256": _sha256(dataset_manifest_payload),
        "dataset_manifest_sha256": manifest_self_hash,
        "materialization_receipt_file_sha256": _sha256(materialization_receipt_payload),
        "materialization_receipt_sha256": receipt_self_hash,
        "freeze_receipt_file_sha256": freeze_file_hash,
    }


def _verify_self_hash(document: Mapping[str, Any], *, hash_field: str, field: str) -> None:
    claimed = _digest(document.get(hash_field), field=f"{field}.{hash_field}")
    if claimed != canonical_json_hash(document, remove=hash_field):
        raise CommunityQualityProducerError(f"{field} canonical self hash does not verify")


def _validate_model_manifest(document: Mapping[str, Any]) -> str:
    value = _exact_keys(
        document,
        {
            "schema_version",
            "model_id",
            "artifact_class",
            "label_count",
            "ordered_labels",
            "taxonomy_version",
            "attention_mode",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "model_identity_sha256",
            "artifact_sha256",
            "manifest_sha256",
        },
        field="final model manifest",
    )
    if (
        value["schema_version"] != MODEL_MANIFEST_SCHEMA_VERSION
        or value["artifact_class"] != "24_label_zh_hans_token_classifier"
        or value["label_count"] != 24
        or value["ordered_labels"] != list(PII_CORE_LABELS)
        or value["taxonomy_version"] != "1.0.0"
        or value["attention_mode"] not in {"causal", "full"}
    ):
        raise CommunityQualityProducerError("final model manifest is not canonical Open-24")
    model_id = _safe_id(value["model_id"], field="model_id")
    _digest(value["training_manifest_file_sha256"], field="training manifest file hash")
    _digest(value["training_manifest_sha256"], field="training manifest self hash")
    _digest(value["model_identity_sha256"], field="model identity hash")
    _digest(value["artifact_sha256"], field="model artifact hash")
    _verify_self_hash(value, hash_field="manifest_sha256", field="final model manifest")
    return model_id


def _validate_service_manifest(
    document: Mapping[str, Any],
    *,
    model_id: str,
    model_manifest_sha256: str,
    model_identity_sha256: str,
) -> str:
    value = _exact_keys(
        document,
        {
            "schema_version",
            "service_id",
            "profile_id",
            "canonical_track",
            "final_model_id",
            "final_model_manifest_sha256",
            "model_identity_sha256",
            "calibration_bundle_file_sha256",
            "implementation_sha256",
            "configuration_sha256",
            "manifest_sha256",
        },
        field="service configuration manifest",
    )
    if (
        value["schema_version"] != SERVICE_MANIFEST_SCHEMA_VERSION
        or value["canonical_track"] != "full_system"
        or value["final_model_id"] != model_id
        or value["final_model_manifest_sha256"] != model_manifest_sha256
        or value["model_identity_sha256"] != model_identity_sha256
    ):
        raise CommunityQualityProducerError(
            "service configuration does not bind the final Open-24 model"
        )
    service_id = _safe_id(value["service_id"], field="service_id")
    _safe_id(value["profile_id"], field="profile_id")
    _digest(value["calibration_bundle_file_sha256"], field="calibration bundle hash")
    _digest(value["implementation_sha256"], field="service implementation hash")
    _digest(value["configuration_sha256"], field="service configuration hash")
    _verify_self_hash(value, hash_field="manifest_sha256", field="service manifest")
    return service_id


def _validate_performance_manifest(
    document: Mapping[str, Any],
    *,
    track: str,
    candidate_id: str,
    document_count: int,
    gold_sha256: str,
    predictions_sha256: str,
    model_manifest_sha256: str,
    service_manifest_sha256: str,
) -> dict[str, int | float]:
    value = _exact_keys(
        document,
        {
            "schema_version",
            "track",
            "system_id",
            "gold_sha256",
            "predictions_sha256",
            "final_model_manifest_sha256",
            "service_configuration_manifest_sha256",
            "measurement",
            "manifest_sha256",
        },
        field=f"{track} performance manifest",
    )
    expected = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "track": track,
        "system_id": candidate_id,
        "gold_sha256": gold_sha256,
        "predictions_sha256": predictions_sha256,
        "final_model_manifest_sha256": model_manifest_sha256,
        "service_configuration_manifest_sha256": service_manifest_sha256,
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise CommunityQualityProducerError(
            f"{track} performance manifest is bound to different inputs or track"
        )
    _verify_self_hash(value, hash_field="manifest_sha256", field=f"{track} performance")
    measurement = _exact_keys(
        value["measurement"],
        {
            "sample_count",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "throughput_documents_per_second",
            "throughput_characters_per_second",
            "peak_cpu_rss_mb",
            "peak_gpu_memory_mb",
        },
        field=f"{track} performance measurement",
    )
    result: dict[str, int | float] = {
        "sample_count": _positive_integer(
            measurement["sample_count"], field=f"{track}.sample_count"
        )
    }
    if result["sample_count"] != document_count:
        raise CommunityQualityProducerError(
            f"{track} performance sample_count must cover every Open-24 document"
        )
    for key in set(measurement) - {"sample_count"}:
        result[key] = _nonnegative_number(
            measurement[key],
            field=f"{track}.{key}",
            positive=key.startswith("throughput_"),
        )
    if not (
        float(result["latency_p50_ms"])
        <= float(result["latency_p95_ms"])
        <= float(result["latency_p99_ms"])
    ):
        raise CommunityQualityProducerError(f"{track} latency percentiles are inconsistent")
    return result


def _validate_registry_schema(document: Mapping[str, Any], *, repository_root: Path) -> None:
    schema_path = repository_root / REGISTRY_SCHEMA_PATH
    schema, _payload = _load_json_file(schema_path, field="Open-24 registry schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise CommunityQualityProducerError(
            "Open-24 comparator registry does not satisfy its strict schema"
        ) from exc


def _release_candidate_prediction_metadata(path: Path) -> dict[str, Any]:
    """Delegate candidate provenance parsing to its frozen standalone validator."""

    try:
        from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
            ReleaseEvalV2PredictionProvenanceError,
            validate_release_eval_v2_prediction_manifest,
        )

        return validate_release_eval_v2_prediction_manifest(path)
    except ReleaseEvalV2PredictionProvenanceError as exc:
        raise CommunityQualityProducerError(
            "candidate prediction provenance metadata validation failed"
        ) from exc


def _validate_candidate_prediction_manifest(
    path: Path,
    payload: bytes,
    *,
    track: str,
    gold_sha256: str,
    gold_size_bytes: int,
    gold_document_count: int,
    gold_provenance: Mapping[str, int | str],
    predictions_sha256: str,
    predictions_size_bytes: int,
    prediction_document_count: int,
    model_manifest: Mapping[str, Any],
    service_manifest: Mapping[str, Any],
) -> str:
    metadata = _exact_keys(
        _release_candidate_prediction_metadata(path),
        {
            "status",
            "full_replay_required",
            "manifest_file_sha256",
            "manifest_sha256",
            "track",
            "target_split",
            "dataset_id",
            "dataset_version",
            "dataset_manifest_file_sha256",
            "dataset_manifest_sha256",
            "materialization_receipt_file_sha256",
            "materialization_receipt_sha256",
            "freeze_receipt_file_sha256",
            "gold_file_sha256",
            "gold_size_bytes",
            "gold_document_count",
            "predictions_file_sha256",
            "predictions_size_bytes",
            "prediction_document_count",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "model_identity_sha256",
            "output_artifact_sha256",
            "service",
        },
        field="candidate prediction provenance metadata",
    )
    manifest_file_sha256 = _digest(
        metadata["manifest_file_sha256"], field="candidate provenance file hash"
    )
    manifest_sha256 = _digest(metadata["manifest_sha256"], field="candidate provenance self hash")
    expected_common = {
        "status": "PASS_METADATA_ONLY",
        "full_replay_required": True,
        "track": track,
        "target_split": gold_provenance["gold_split"],
        "dataset_id": gold_provenance["dataset_id"],
        "dataset_version": gold_provenance["dataset_version"],
        "dataset_manifest_file_sha256": gold_provenance["dataset_manifest_file_sha256"],
        "dataset_manifest_sha256": gold_provenance["dataset_manifest_sha256"],
        "materialization_receipt_file_sha256": gold_provenance[
            "materialization_receipt_file_sha256"
        ],
        "materialization_receipt_sha256": gold_provenance["materialization_receipt_sha256"],
        "freeze_receipt_file_sha256": gold_provenance["freeze_receipt_file_sha256"],
        "gold_file_sha256": gold_sha256,
        "gold_size_bytes": gold_size_bytes,
        "gold_document_count": gold_document_count,
        "predictions_file_sha256": predictions_sha256,
        "predictions_size_bytes": predictions_size_bytes,
        "prediction_document_count": prediction_document_count,
        "training_manifest_file_sha256": model_manifest["training_manifest_file_sha256"],
        "training_manifest_sha256": model_manifest["training_manifest_sha256"],
        "model_identity_sha256": model_manifest["model_identity_sha256"],
        "output_artifact_sha256": model_manifest["artifact_sha256"],
    }
    if (
        any(metadata[key] != value for key, value in expected_common.items())
        or manifest_file_sha256 != _sha256(payload)
        or prediction_document_count != gold_document_count
    ):
        raise CommunityQualityProducerError(
            "candidate prediction provenance is bound to different inputs"
        )
    expected_service: Mapping[str, Any] | None
    if track == "model_raw":
        expected_service = None
    else:
        expected_service = {
            "profile_id": service_manifest["profile_id"],
            "configuration_sha256": service_manifest["configuration_sha256"],
            "implementation_sha256": service_manifest["implementation_sha256"],
            "model_identity_sha256": model_manifest["model_identity_sha256"],
            "calibration_bundle_file_sha256": service_manifest["calibration_bundle_file_sha256"],
        }
    if metadata["service"] != expected_service:
        raise CommunityQualityProducerError(
            "candidate prediction provenance does not bind the final service"
        )
    # The standalone validator re-reads the file.  Require the exact bytes read
    # by this producer to remain unchanged across that independent validation.
    current_payload = _read_regular(
        path,
        field="candidate prediction provenance manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if current_payload != payload:
        raise CommunityQualityProducerError(
            "candidate prediction provenance changed during validation"
        )
    return manifest_sha256


def _validate_comparator_prediction_manifest(
    document: Mapping[str, Any],
    payload: bytes,
    *,
    repository_root: Path,
    track: str,
    comparator_id: str,
    gold_sha256: str,
    gold_size_bytes: int,
    gold_document_count: int,
    gold_provenance: Mapping[str, int | str],
    predictions_sha256: str,
    predictions_size_bytes: int,
    prediction_document_count: int,
) -> str:
    schema, _schema_payload = _load_json_file(
        repository_root / COMPARATOR_PREDICTION_SCHEMA_PATH,
        field="Open-24 comparator prediction provenance schema",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise CommunityQualityProducerError(
            "comparator prediction provenance violates its strict schema"
        ) from exc
    if document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256"):
        raise CommunityQualityProducerError(
            "comparator prediction provenance self hash does not verify"
        )
    if document["execution"]["fixture"] is not False:
        raise CommunityQualityProducerError(
            "fixture comparator provenance is not admissible for formal quality production"
        )
    dataset = document["dataset"]
    prediction = document["prediction"]
    if (
        document["track"] != track
        or document["system_id"] != comparator_id
        or dataset["dataset_id"] != gold_provenance["dataset_id"]
        or dataset["dataset_version"] != gold_provenance["dataset_version"]
        or dataset["target_split"] != gold_provenance["gold_split"]
        or dataset["gold_file_sha256"] != gold_sha256
        or dataset["gold_size_bytes"] != gold_size_bytes
        or dataset["gold_document_count"] != gold_document_count
        or dataset["dataset_manifest_file_sha256"]
        != gold_provenance["dataset_manifest_file_sha256"]
        or dataset["dataset_manifest_sha256"] != gold_provenance["dataset_manifest_sha256"]
        or dataset["materialization_receipt_file_sha256"]
        != gold_provenance["materialization_receipt_file_sha256"]
        or dataset["materialization_receipt_sha256"]
        != gold_provenance["materialization_receipt_sha256"]
        or dataset["freeze_receipt_file_sha256"] != gold_provenance["freeze_receipt_file_sha256"]
        or prediction["file_sha256"] != predictions_sha256
        or prediction["size_bytes"] != predictions_size_bytes
        or prediction["document_count"] != prediction_document_count
        or prediction_document_count != gold_document_count
    ):
        raise CommunityQualityProducerError(
            "comparator prediction provenance is bound to different inputs"
        )
    # Do not permit callers to validate a parsed object from one file while the
    # registry hashes another file's bytes.
    if _load_json_payload(payload, field="comparator prediction provenance") != document:
        raise CommunityQualityProducerError("comparator provenance object/bytes differ")
    return str(document["manifest_sha256"])


def _validate_registry(
    document: Mapping[str, Any],
    *,
    repository_root: Path,
    gold_sha256: str,
    gold_provenance: Mapping[str, int | str],
    model_id: str,
    service_id: str,
    model_manifest_sha256: str,
    service_manifest_sha256: str,
    track_payload_sha256: Mapping[str, Mapping[str, str]],
) -> dict[str, tuple[str, str]]:
    _validate_registry_schema(document, repository_root=repository_root)
    _verify_self_hash(document, hash_field="registry_sha256", field="Open-24 registry")
    if (
        document["schema_version"] != REGISTRY_SCHEMA_VERSION
        or document["benchmark"]["protocol_id"] != "public_synthetic_chinese_pii_open24_v2"
        or document["benchmark"]["ordered_labels"] != list(PII_CORE_LABELS)
        or document["benchmark"]["gold_sha256"] != gold_sha256
        or document["benchmark"]["gold_split"] != gold_provenance["gold_split"]
        or document["benchmark"]["dataset_id"] != gold_provenance["dataset_id"]
        or document["benchmark"]["dataset_version"] != gold_provenance["dataset_version"]
        or document["benchmark"]["dataset_manifest_file_sha256"]
        != gold_provenance["dataset_manifest_file_sha256"]
        or document["benchmark"]["dataset_manifest_sha256"]
        != gold_provenance["dataset_manifest_sha256"]
        or document["benchmark"]["materialization_receipt_file_sha256"]
        != gold_provenance["materialization_receipt_file_sha256"]
        or document["benchmark"]["materialization_receipt_sha256"]
        != gold_provenance["materialization_receipt_sha256"]
        or document["benchmark"]["freeze_receipt_file_sha256"]
        != gold_provenance["freeze_receipt_file_sha256"]
        or document["final_model_manifest_sha256"] != model_manifest_sha256
        or document["service_configuration_manifest_sha256"] != service_manifest_sha256
        or set(document["tracks"]) != set(TRACKS)
    ):
        raise CommunityQualityProducerError("Open-24 comparator registry binding drifted")
    result: dict[str, tuple[str, str]] = {}
    for track in TRACKS:
        item = document["tracks"][track]
        hashes = track_payload_sha256[track]
        if (
            item["canonical_track"] != track
            or item["candidate_predictions_sha256"] != hashes["candidate"]
            or item["candidate_prediction_manifest_file_sha256"] != hashes["candidate_provenance"]
            or item["comparator_predictions_sha256"] != hashes["comparator"]
            or item["comparator_prediction_manifest_file_sha256"] != hashes["comparator_provenance"]
            or item["performance_manifest_sha256"] != hashes["performance"]
        ):
            raise CommunityQualityProducerError(
                f"{track} files do not match the frozen Open-24 registry"
            )
        candidate_id = _safe_id(item["candidate_id"], field=f"{track} candidate_id")
        comparator_id = _safe_id(item["comparator_id"], field=f"{track} comparator_id")
        expected_candidate_id = model_id if track == "model_raw" else service_id
        if candidate_id != expected_candidate_id:
            raise CommunityQualityProducerError(
                f"{track} candidate ID does not match its final model/service binding"
            )
        if candidate_id == comparator_id:
            raise CommunityQualityProducerError(f"{track} candidate equals comparator")
        result[track] = (candidate_id, comparator_id)
    return result


def _load_gold(payload: bytes) -> tuple[GoldRecord, ...]:
    try:
        records = tuple(load_gold_jsonl(StringIO(payload.decode("utf-8"))))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise CommunityQualityProducerError("Open-24 gold JSONL validation failed") from exc
    if not records:
        raise CommunityQualityProducerError("Open-24 gold must contain documents")
    labels = frozenset(PII_CORE_LABELS)
    observed: set[str] = set()
    seen: set[str] = set()
    pii_free = 0
    total_characters = 0
    for record in records:
        record.validate()
        if record.doc_id in seen:
            raise CommunityQualityProducerError("Open-24 gold contains duplicate document IDs")
        seen.add(record.doc_id)
        if record.text_length is None:
            raise CommunityQualityProducerError("Open-24 gold requires text_length for every row")
        total_characters += record.text_length
        record_labels = {span.label for span in record.spans}
        if record_labels - labels:
            raise CommunityQualityProducerError("Open-24 gold contains an out-of-taxonomy label")
        observed.update(record_labels)
        pii_free += int(not record.spans)
    if observed != labels:
        raise CommunityQualityProducerError("Open-24 gold lacks support for one or more labels")
    if pii_free < 1 or total_characters < 1:
        raise CommunityQualityProducerError(
            "Open-24 gold requires PII-free documents and non-empty character support"
        )
    return records


def _load_predictions(payload: bytes, *, field: str) -> tuple[PredictionRecord, ...]:
    try:
        records = tuple(load_prediction_jsonl(StringIO(payload.decode("utf-8"))))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise CommunityQualityProducerError(f"{field} validation failed") from exc
    if not records:
        raise CommunityQualityProducerError(f"{field} must contain documents")
    allowed = frozenset(PII_CORE_LABELS)
    if any(span.label not in allowed for record in records for span in record.spans):
        raise CommunityQualityProducerError(f"{field} contains an out-of-taxonomy label")
    return records


def _sum_counts(values: Sequence[StrictCounts]) -> StrictCounts:
    return StrictCounts(
        tp=sum(value.tp for value in values),
        fp=sum(value.fp for value in values),
        fn=sum(value.fn for value in values),
    )


def _score(counts: StrictCounts) -> dict[str, int | float]:
    precision = Fraction(counts.tp, counts.tp + counts.fp) if counts.tp + counts.fp else Fraction(0)
    recall = Fraction(counts.tp, counts.tp + counts.fn) if counts.tp + counts.fn else Fraction(0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _strict_span(suite: ThreeMetricPairedSuite, *, side: str) -> dict[str, Any]:
    systems = [getattr(document, side) for document in suite.documents]
    per_label: dict[str, dict[str, int | float]] = {}
    label_f1: list[float] = []
    for index, label in enumerate(PII_CORE_LABELS):
        scored = _score(_sum_counts([system.strict_per_label[index] for system in systems]))
        per_label[label] = scored
        label_f1.append(float(scored["f1"]))
    micro = _score(_sum_counts([system.strict_micro for system in systems]))
    return {
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": sum(label_f1) / len(label_f1),
        "per_label": per_label,
    }


def _verify_strict_metric_reuse(
    gold: Sequence[GoldRecord],
    predictions: Sequence[PredictionRecord],
    strict: Mapping[str, Any],
    *,
    track: str,
) -> None:
    """Cross-check point counts against the project's canonical span evaluator."""

    canonical = evaluate_documents(gold, predictions, bootstrap_samples=0)["strict"]
    micro = canonical["micro"]
    expected = {
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": canonical["macro"]["f1"],
    }
    counts_match = all(strict[name] == expected[name] for name in ("tp", "fp", "fn"))
    scores_match = all(
        math.isclose(
            float(strict[name]),
            float(expected[name]),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        for name in ("micro_precision", "micro_recall", "micro_f1", "macro_f1")
    )
    if not counts_match or not scores_match:
        raise CommunityQualityProducerError(
            f"{track} strict metrics disagree with the canonical evaluator"
        )


def _merged_intervals(spans: Sequence[Span]) -> tuple[tuple[int, int], ...]:
    values = sorted((span.start, span.end) for span in spans)
    merged: list[list[int]] = []
    for start, end in values:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _interval_length(values: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in values)


def _intersection_length(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> int:
    total = left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _character_error(
    gold: Sequence[GoldRecord], predictions: Sequence[PredictionRecord]
) -> dict[str, int | float | str]:
    prediction_by_id = {record.doc_id: record for record in predictions}
    gold_pii = missed = non_pii = falsely_flagged = 0
    for record in gold:
        predicted = prediction_by_id[record.doc_id]
        assert record.text_length is not None
        gold_intervals = _merged_intervals(record.spans)
        prediction_intervals = _merged_intervals(predicted.spans)
        gold_length = _interval_length(gold_intervals)
        predicted_length = _interval_length(prediction_intervals)
        overlap = _intersection_length(gold_intervals, prediction_intervals)
        gold_pii += gold_length
        missed += gold_length - overlap
        non_pii += record.text_length - gold_length
        falsely_flagged += predicted_length - overlap
    if gold_pii < 1 or non_pii < 1:
        raise CommunityQualityProducerError("character-error denominators are empty")
    return {
        "definition": "label_agnostic_union_of_left_closed_right_open_pii_intervals",
        "gold_pii_character_count": gold_pii,
        "missed_pii_character_count": missed,
        "pii_character_miss_rate": missed / gold_pii,
        "non_pii_character_count": non_pii,
        "falsely_flagged_non_pii_character_count": falsely_flagged,
        "non_pii_character_false_positive_rate": falsely_flagged / non_pii,
    }


def _pii_free_metrics(
    gold: Sequence[GoldRecord], predictions: Sequence[PredictionRecord]
) -> dict[str, int | float | str]:
    prediction_by_id = {record.doc_id: record for record in predictions}
    eligible = [record for record in gold if not record.spans]
    characters = sum(record.text_length or 0 for record in eligible)
    if not eligible or characters < 1:
        raise CommunityQualityProducerError("PII-free metric denominator is empty")
    fp_documents = sum(bool(prediction_by_id[record.doc_id].spans) for record in eligible)
    fp_spans = sum(len(prediction_by_id[record.doc_id].spans) for record in eligible)
    return {
        "eligibility": "explicit_synthetic_or_open24_empty_gold_partition",
        "document_count": len(eligible),
        "false_positive_document_count": fp_documents,
        "document_fpr": fp_documents / len(eligible),
        "text_character_count": characters,
        "false_positive_span_count": fp_spans,
        "false_positive_spans_per_1000_characters": fp_spans * 1000.0 / characters,
    }


def _paired_bootstrap(
    *,
    track: str,
    candidate_id: str,
    comparator_id: str,
    suite: ThreeMetricPairedSuite,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    result = three_metric_intersection_union_bootstrap(
        {comparator_id: (suite,)},
        candidate_id=candidate_id,
        admitted_comparator_ids=(comparator_id,),
        samples=samples,
        confidence=CONFIDENCE,
        seed=seed,
        fpr_noninferiority_margin=0.0,
        fpr_absolute_maximum=1.0,
    )
    comparison = result["comparisons"][comparator_id]

    def higher(metric: str) -> dict[str, Any]:
        value = comparison["metrics"][metric]
        interval = value["gate_oriented_confidence_interval"]
        return {
            "point_delta_candidate_minus_comparator": value[
                "raw_point_delta_candidate_minus_comparator"
            ],
            "ci_low": interval["lower"],
            "ci_high": interval["upper"],
        }

    fpr = comparison["metrics"]["pii_free_document_fpr"]
    gate_interval = fpr["gate_oriented_confidence_interval"]
    fpr_delta = {
        "point_delta_candidate_minus_comparator": fpr["raw_point_delta_candidate_minus_comparator"],
        # gate-oriented delta is comparator-candidate because margin is zero.
        "ci_low": -gate_interval["upper"],
        "ci_high": -gate_interval["lower"],
    }
    return {
        "method": "paired_atomic_stratified_document_bootstrap_percentile",
        "comparison_unit": "document",
        "track": track,
        "candidate_id": candidate_id,
        "comparator_id": comparator_id,
        "comparator_track": track,
        "seed": seed,
        "resamples": samples,
        "confidence_level": CONFIDENCE,
        "prng": result["prng"],
        "resample_plan_sha256": result["shared_resample_plan_sha256"],
        "micro_f1_delta": higher("strict_micro_f1"),
        "macro_f1_delta": higher("strict_macro_f1"),
        "document_fpr_delta": fpr_delta,
    }


def _binding(payload: bytes) -> dict[str, int | str]:
    return {"file_sha256": _sha256(payload), "size_bytes": len(payload)}


def _gate_track(
    *,
    track: str,
    strict: Mapping[str, Any],
    pii_free: Mapping[str, Any],
    character: Mapping[str, Any],
    performance: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, list[str]]:
    thresholds = policy["thresholds"]
    bootstrap_policy = policy["bootstrap"]
    low_labels = [
        label
        for label in PII_CORE_LABELS
        if strict["per_label"][label]["recall"] < thresholds["minimum_per_label_recall_min"]
    ]
    checks = {
        "strict_micro_f1": strict["micro_f1"] >= thresholds["strict_micro_f1_min"],
        "strict_macro_f1": strict["macro_f1"] >= thresholds["strict_macro_f1_min"],
        "minimum_per_label_recall": not low_labels,
        "pii_free_document_fpr": (
            pii_free["document_fpr"] <= thresholds["pii_free_document_fpr_max"]
        ),
        "fp_spans_per_1000_characters": (
            pii_free["false_positive_spans_per_1000_characters"]
            <= thresholds["fp_spans_per_1000_characters_max"]
        ),
        "pii_character_miss_rate": (
            character["pii_character_miss_rate"] <= thresholds["pii_character_miss_rate_max"]
        ),
        "non_pii_character_false_positive_rate": (
            character["non_pii_character_false_positive_rate"]
            <= thresholds["non_pii_character_false_positive_rate_max"]
        ),
        "latency_p95_ms": performance["latency_p95_ms"] <= thresholds["latency_p95_ms_max"],
        "throughput_documents_per_second": (
            performance["throughput_documents_per_second"]
            >= thresholds["throughput_documents_per_second_min"]
        ),
        "peak_cpu_rss_mb": performance["peak_cpu_rss_mb"] <= thresholds["peak_cpu_rss_mb_max"],
        "peak_gpu_memory_mb": (
            performance["peak_gpu_memory_mb"] <= thresholds["peak_gpu_memory_mb_max"]
        ),
        "bootstrap_resamples": bootstrap["resamples"] >= bootstrap_policy["resamples_min"],
        "bootstrap_micro_f1": (
            bootstrap["micro_f1_delta"]["ci_low"] >= bootstrap_policy["micro_f1_delta_ci_low_min"]
        ),
        "bootstrap_macro_f1": (
            bootstrap["macro_f1_delta"]["ci_low"] >= bootstrap_policy["macro_f1_delta_ci_low_min"]
        ),
        "bootstrap_document_fpr": (
            bootstrap["document_fpr_delta"]["ci_high"]
            <= bootstrap_policy["document_fpr_delta_ci_high_max"]
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    failures.extend(f"per_label_recall:{label}" for label in low_labels)
    failures = sorted(set(failures))
    return ("PASS" if not failures else "FAIL"), failures


def _validate_track_inputs(values: Sequence[TrackInput]) -> dict[str, TrackInput]:
    tracks: dict[str, TrackInput] = {}
    for value in values:
        if value.track not in TRACKS or value.track in tracks:
            raise CommunityQualityProducerError(
                "track inputs must contain model_raw and full_system exactly once"
            )
        tracks[value.track] = value
    if tuple(sorted(tracks, key=TRACKS.index)) != TRACKS:
        raise CommunityQualityProducerError(
            "track inputs must contain model_raw and full_system exactly once"
        )
    return tracks


def build_receipt(
    inputs: ProducerInputs,
    *,
    samples: int = MIN_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Read frozen inputs and return a deterministic aggregate-only receipt."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < MIN_BOOTSTRAP_SAMPLES:
        raise CommunityQualityProducerError("paired bootstrap requires at least 10000 resamples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CommunityQualityProducerError("bootstrap seed must be an integer")
    track_inputs = _validate_track_inputs(inputs.tracks)

    dataset_manifest_document, dataset_manifest_payload = _formal_internal_preopen_guard(inputs)
    gold_payload = _read_regular(inputs.gold, field="Open-24 gold", maximum_bytes=_MAX_JSONL_BYTES)
    materialization_document, materialization_payload = _load_json_file(
        inputs.gold_materialization_receipt, field="gold materialization receipt"
    )
    freeze_document, freeze_payload = _load_json_file(
        inputs.gold_freeze_receipt, field="gold freeze receipt"
    )
    model_document, model_payload = _load_json_file(
        inputs.final_model_manifest, field="final model manifest"
    )
    service_document, service_payload = _load_json_file(
        inputs.service_configuration, field="service configuration manifest"
    )
    registry_document, registry_payload = _load_json_file(
        inputs.comparator_registry, field="Open-24 comparator registry"
    )
    gold_sha256 = _sha256(gold_payload)
    model_sha256 = _sha256(model_payload)
    service_sha256 = _sha256(service_payload)
    model_id = _validate_model_manifest(model_document)
    service_id = _validate_service_manifest(
        service_document,
        model_id=model_id,
        model_manifest_sha256=model_sha256,
        model_identity_sha256=str(model_document["model_identity_sha256"]),
    )
    gold = _load_gold(gold_payload)
    gold_provenance = _validate_gold_provenance(
        dataset_manifest=dataset_manifest_document,
        dataset_manifest_payload=dataset_manifest_payload,
        materialization_receipt=materialization_document,
        materialization_receipt_payload=materialization_payload,
        freeze_receipt=freeze_document,
        freeze_receipt_payload=freeze_payload,
        freeze_receipt_path=inputs.gold_freeze_receipt,
        gold_path=inputs.gold,
        gold_file_sha256=gold_sha256,
        gold_size_bytes=len(gold_payload),
        gold_split=inputs.gold_split,
        gold_document_count=len(gold),
    )

    payloads: dict[str, dict[str, bytes]] = {}
    track_hashes: dict[str, dict[str, str]] = {}
    for track in TRACKS:
        item = track_inputs[track]
        candidate_payload = _read_regular(
            item.candidate_predictions,
            field=f"{track} candidate predictions",
            maximum_bytes=_MAX_JSONL_BYTES,
        )
        comparator_payload = _read_regular(
            item.comparator_predictions,
            field=f"{track} comparator predictions",
            maximum_bytes=_MAX_JSONL_BYTES,
        )
        candidate_provenance_payload = _read_regular(
            item.candidate_prediction_manifest,
            field=f"{track} candidate prediction provenance",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        comparator_provenance_payload = _read_regular(
            item.comparator_prediction_manifest,
            field=f"{track} comparator prediction provenance",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        performance_payload = _read_regular(
            item.performance_manifest,
            field=f"{track} performance manifest",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        payloads[track] = {
            "candidate": candidate_payload,
            "comparator": comparator_payload,
            "candidate_provenance": candidate_provenance_payload,
            "comparator_provenance": comparator_provenance_payload,
            "performance": performance_payload,
        }
        track_hashes[track] = {name: _sha256(payload) for name, payload in payloads[track].items()}

    identities = _validate_registry(
        registry_document,
        repository_root=repository_root,
        gold_sha256=gold_sha256,
        gold_provenance=gold_provenance,
        model_id=model_id,
        service_id=service_id,
        model_manifest_sha256=model_sha256,
        service_manifest_sha256=service_sha256,
        track_payload_sha256=track_hashes,
    )
    gold_ids = {record.doc_id for record in gold}
    pii_free_ids = tuple(record.doc_id for record in gold if not record.spans)

    threshold_contract = validate_preregistration(repository_root=repository_root)
    threshold_payload = _read_regular(
        repository_root / THRESHOLD_CONTRACT_PATH,
        field="v1 threshold contract",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    result_tracks: dict[str, Any] = {}
    input_track_bindings: dict[str, Any] = {}
    for track in TRACKS:
        candidate_id, comparator_id = identities[track]
        candidate = _load_predictions(
            payloads[track]["candidate"], field=f"{track} candidate predictions"
        )
        comparator = _load_predictions(
            payloads[track]["comparator"], field=f"{track} comparator predictions"
        )
        if {record.doc_id for record in candidate} != gold_ids:
            raise CommunityQualityProducerError(
                f"{track} candidate predictions do not pair exactly with gold"
            )
        if {record.doc_id for record in comparator} != gold_ids:
            raise CommunityQualityProducerError(
                f"{track} comparator predictions do not pair exactly with gold"
            )
        candidate_manifest_sha256 = _validate_candidate_prediction_manifest(
            track_inputs[track].candidate_prediction_manifest,
            payloads[track]["candidate_provenance"],
            track=track,
            gold_sha256=gold_sha256,
            gold_size_bytes=len(gold_payload),
            gold_document_count=len(gold),
            gold_provenance=gold_provenance,
            predictions_sha256=track_hashes[track]["candidate"],
            predictions_size_bytes=len(payloads[track]["candidate"]),
            prediction_document_count=len(candidate),
            model_manifest=model_document,
            service_manifest=service_document,
        )
        comparator_manifest_document = _load_json_payload(
            payloads[track]["comparator_provenance"],
            field=f"{track} comparator prediction provenance",
        )
        comparator_manifest_sha256 = _validate_comparator_prediction_manifest(
            comparator_manifest_document,
            payloads[track]["comparator_provenance"],
            repository_root=repository_root,
            track=track,
            comparator_id=comparator_id,
            gold_sha256=gold_sha256,
            gold_size_bytes=len(gold_payload),
            gold_document_count=len(gold),
            gold_provenance=gold_provenance,
            predictions_sha256=track_hashes[track]["comparator"],
            predictions_size_bytes=len(payloads[track]["comparator"]),
            prediction_document_count=len(comparator),
        )
        registry_track = registry_document["tracks"][track]
        if (
            registry_track["candidate_prediction_manifest_sha256"] != candidate_manifest_sha256
            or registry_track["comparator_prediction_manifest_sha256"] != comparator_manifest_sha256
        ):
            raise CommunityQualityProducerError(
                f"{track} prediction provenance self hash drifted from registry"
            )
        performance_document = _load_json_payload(
            payloads[track]["performance"], field=f"{track} performance manifest"
        )
        performance = _validate_performance_manifest(
            performance_document,
            track=track,
            candidate_id=candidate_id,
            document_count=len(gold),
            gold_sha256=gold_sha256,
            predictions_sha256=track_hashes[track]["candidate"],
            model_manifest_sha256=model_sha256,
            service_manifest_sha256=service_sha256,
        )
        suite = build_three_metric_paired_suite(
            name=f"open24-{track}",
            gold_records=gold,
            candidate_records=candidate,
            comparator_records=comparator,
            labels=PII_CORE_LABELS,
            pii_free_eligible_doc_ids=pii_free_ids,
        )
        strict = _strict_span(suite, side="candidate")
        comparator_strict = _strict_span(suite, side="comparator")
        _verify_strict_metric_reuse(gold, candidate, strict, track=track)
        _verify_strict_metric_reuse(
            gold,
            comparator,
            comparator_strict,
            track=f"{track} comparator",
        )
        pii_free = _pii_free_metrics(gold, candidate)
        comparator_pii_free = _pii_free_metrics(gold, comparator)
        character = _character_error(gold, candidate)
        comparator_character = _character_error(gold, comparator)
        bootstrap = _paired_bootstrap(
            track=track,
            candidate_id=candidate_id,
            comparator_id=comparator_id,
            suite=suite,
            samples=samples,
            seed=seed,
        )
        status, failures = _gate_track(
            track=track,
            strict=strict,
            pii_free=pii_free,
            character=character,
            performance=performance,
            bootstrap=bootstrap,
            policy=threshold_contract["track_policies"][track],
        )
        benchmark_leading = (
            status == "PASS"
            and bootstrap["micro_f1_delta"]["ci_low"] > 0.0
            and bootstrap["macro_f1_delta"]["ci_low"] > 0.0
            and bootstrap["document_fpr_delta"]["ci_high"] <= 0.0
        )
        result_tracks[track] = {
            "declared_status": status,
            "failure_ids": failures,
            "canonical_track": track,
            "candidate_id": candidate_id,
            "comparator_id": comparator_id,
            "strict_span": strict,
            "pii_free": pii_free,
            "character_error": character,
            "performance": performance,
            "comparator_aggregate": {
                "strict_span": comparator_strict,
                "pii_free": comparator_pii_free,
                "character_error": comparator_character,
            },
            "paired_bootstrap": bootstrap,
            "benchmark_leading_on_named_open24_suite": benchmark_leading,
        }
        input_track_bindings[track] = {
            "candidate_predictions": _binding(payloads[track]["candidate"]),
            "candidate_prediction_manifest": _binding(payloads[track]["candidate_provenance"]),
            "candidate_prediction_manifest_sha256": candidate_manifest_sha256,
            "comparator_predictions": _binding(payloads[track]["comparator"]),
            "comparator_prediction_manifest": _binding(payloads[track]["comparator_provenance"]),
            "comparator_prediction_manifest_sha256": comparator_manifest_sha256,
            "candidate_prediction_full_replay_required": True,
            "comparator_generation_replay_required": True,
            "performance_harness_replay_required": True,
            "performance_manifest": _binding(payloads[track]["performance"]),
        }

    overall_status = result_tracks["full_system"]["declared_status"]
    producer_payload = _read_regular(
        repository_root / PRODUCER_PATH,
        field="quality producer source",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    schema_payload = _read_regular(
        repository_root / SCHEMA_PATH,
        field="quality result schema",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    registry_schema_payload = _read_regular(
        repository_root / REGISTRY_SCHEMA_PATH,
        field="quality registry schema",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    metric_bindings = {}
    for relative in (
        "src/pii_zh/evaluation/io.py",
        "src/pii_zh/evaluation/metrics.py",
        "src/pii_zh/evaluation/paired_bootstrap.py",
        "src/pii_zh/evaluation/paired_bootstrap_three_metric.py",
        "src/pii_zh/evaluation/service_quality_suite.py",
    ):
        payload = _read_regular(
            repository_root / relative,
            field="bound metric source",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        metric_bindings[relative] = _sha256(payload)
    prediction_provenance_bindings = {}
    for relative in (
        CANDIDATE_PREDICTION_SCHEMA_PATH,
        CANDIDATE_PREDICTION_MODULE_PATH,
        CANDIDATE_PREDICTION_CLI_PATH,
        COMPARATOR_PREDICTION_SCHEMA_PATH,
    ):
        payload = _read_regular(
            repository_root / relative,
            field="bound prediction provenance implementation",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        prediction_provenance_bindings[relative] = _sha256(payload)

    receipt: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "receipt_id": canonical_json_hash(
            {
                "domain": "pii-zh-public-synthetic-quality-result-v2",
                "gold_sha256": gold_sha256,
                "registry_sha256": _sha256(registry_payload),
                "samples": samples,
                "seed": seed,
            }
        ),
        "reported_status": overall_status,
        "scope": {
            "benchmark_id": "public_synthetic_chinese_pii_community_v2",
            "data": "public_open_source_and_synthetic_only",
            "input_provenance_verified_by_this_producer": False,
            "input_manifest_chain_verified": True,
            "prediction_origin_replay_verified_by_this_producer": False,
            "performance_measurement_replay_verified_by_this_producer": False,
            "public_test_exposed": True,
            "production_evidence": False,
            "real_world_generalization_established": False,
        },
        "protocol": {
            "taxonomy_version": "1.0.0",
            "label_count": 24,
            "ordered_labels": list(PII_CORE_LABELS),
            "strict_span_matching": "document_label_start_end_left_closed_right_open",
            "document_count": len(gold),
            "pii_free_document_count": len(pii_free_ids),
            "gold_label_support_complete": True,
        },
        "candidate": {"model_id": model_id, "service_id": service_id},
        "threshold_contract": {
            "contract_id": THRESHOLD_CONTRACT_ID,
            "file_sha256": _sha256(threshold_payload),
            "canonical_sha256": threshold_contract["contract_sha256"],
            "thresholds_reused_without_change": True,
            "v1_closed8_comparator_registry_reused": False,
        },
        "input_bindings": {
            "gold": _binding(gold_payload),
            "gold_provenance": {
                "dataset_manifest": _binding(dataset_manifest_payload),
                "materialization_receipt": _binding(materialization_payload),
                "freeze_receipt": _binding(freeze_payload),
                **gold_provenance,
            },
            "final_model_manifest": _binding(model_payload),
            "service_configuration_manifest": _binding(service_payload),
            "open24_comparator_registry": _binding(registry_payload),
            "tracks": input_track_bindings,
        },
        "bootstrap": {
            "minimum_resamples": MIN_BOOTSTRAP_SAMPLES,
            "actual_resamples": samples,
            "confidence_level": CONFIDENCE,
            "seed": seed,
            "paired_by_document": True,
        },
        "tracks": result_tracks,
        "claims": {
            "first_chinese_pii_model": False,
            "global_sota": False,
            "production_ready": False,
            "real_world_sota": False,
            "claim_activation_allowed": False,
            "named_open24_descriptive_leader_by_track": {
                track: result_tracks[track]["benchmark_leading_on_named_open24_suite"]
                for track in TRACKS
            },
        },
        "implementation_bindings": {
            "producer_sha256": _sha256(producer_payload),
            "result_schema_sha256": _sha256(schema_payload),
            "registry_schema_sha256": _sha256(registry_schema_payload),
            "metric_source_sha256": metric_bindings,
            "prediction_provenance_source_sha256": prediction_provenance_bindings,
        },
        "privacy": {
            "aggregate_only": True,
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "contains_per_document_statistics": False,
            "model_weights_read": False,
            "gpu_queried_or_used": False,
            "network_used": False,
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, remove="receipt_sha256")
    validate_receipt_schema(receipt, repository_root=repository_root)
    return receipt


def validate_receipt_schema(
    receipt: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT
) -> None:
    schema, _payload = _load_json_file(repository_root / SCHEMA_PATH, field="quality result schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(receipt)
    except Exception as exc:
        raise CommunityQualityProducerError("quality receipt violates its strict schema") from exc
    _verify_self_hash(receipt, hash_field="receipt_sha256", field="quality receipt")


def replay_receipt(
    receipt_path: Path,
    inputs: ProducerInputs,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Recompute a published receipt and require exact semantic equality."""

    receipt, _payload = _load_json_file(receipt_path, field="quality receipt")
    validate_receipt_schema(receipt, repository_root=repository_root)
    bootstrap = receipt["bootstrap"]
    rebuilt = build_receipt(
        inputs,
        samples=bootstrap["actual_resamples"],
        seed=bootstrap["seed"],
        repository_root=repository_root,
    )
    if rebuilt != receipt:
        raise CommunityQualityProducerError(
            "strict replay mismatch: receipt is not reproduced by the frozen inputs"
        )
    return rebuilt


def publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically publish a read-only receipt without clobbering an existing file."""

    if path.exists() or path.is_symlink():
        raise CommunityQualityProducerError("refusing to overwrite an existing receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _pretty_json_bytes(receipt)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o444)
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise CommunityQualityProducerError(
                "refusing to overwrite an existing receipt"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("produce", "replay"))
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--gold-dataset-manifest", required=True, type=Path)
    parser.add_argument("--gold-materialization-receipt", required=True, type=Path)
    parser.add_argument("--gold-freeze-receipt", required=True, type=Path)
    parser.add_argument("--gold-split", required=True, choices=("internal_evaluation",))
    parser.add_argument("--final-model-manifest", required=True, type=Path)
    parser.add_argument("--service-configuration", required=True, type=Path)
    parser.add_argument("--internal-unlock", type=Path)
    parser.add_argument("--final-model-binding", type=Path)
    parser.add_argument("--service-configuration-binding", type=Path)
    parser.add_argument("--comparator-registry", required=True, type=Path)
    parser.add_argument(
        "--track",
        action="append",
        nargs=6,
        metavar=(
            "TRACK",
            "CANDIDATE_PREDICTIONS",
            "CANDIDATE_PREDICTION_MANIFEST",
            "COMPARATOR_PREDICTIONS",
            "COMPARATOR_PREDICTION_MANIFEST",
            "PERFORMANCE",
        ),
        required=True,
    )
    parser.add_argument("--samples", type=int, default=MIN_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser


def _inputs(args: argparse.Namespace) -> ProducerInputs:
    tracks = tuple(
        TrackInput(
            track=raw[0],
            candidate_predictions=Path(raw[1]),
            candidate_prediction_manifest=Path(raw[2]),
            comparator_predictions=Path(raw[3]),
            comparator_prediction_manifest=Path(raw[4]),
            performance_manifest=Path(raw[5]),
        )
        for raw in args.track
    )
    return ProducerInputs(
        gold=args.gold,
        gold_dataset_manifest=args.gold_dataset_manifest,
        gold_materialization_receipt=args.gold_materialization_receipt,
        gold_freeze_receipt=args.gold_freeze_receipt,
        gold_split=args.gold_split,
        final_model_manifest=args.final_model_manifest,
        service_configuration=args.service_configuration,
        comparator_registry=args.comparator_registry,
        tracks=tracks,
        internal_unlock=args.internal_unlock,
        final_model_binding=args.final_model_binding,
        service_configuration_binding=args.service_configuration_binding,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = _inputs(args)
        if args.command == "produce":
            if args.output is None or args.receipt is not None:
                raise CommunityQualityProducerError(
                    "produce requires --output and does not accept --receipt"
                )
            receipt = build_receipt(inputs, samples=args.samples, seed=args.seed)
            publish_receipt(args.output, receipt)
            print(
                json.dumps(
                    {
                        "receipt_sha256": receipt["receipt_sha256"],
                        "reported_status": receipt["reported_status"],
                    },
                    sort_keys=True,
                )
            )
        else:
            if args.receipt is None or args.output is not None:
                raise CommunityQualityProducerError(
                    "replay requires --receipt and does not accept --output"
                )
            receipt = replay_receipt(args.receipt, inputs)
            print(
                json.dumps(
                    {"receipt_sha256": receipt["receipt_sha256"], "replay": "PASS"},
                    sort_keys=True,
                )
            )
    except (CommunityQualityProducerError, EvaluationDataError, ValueError) as exc:
        print(f"public-synthetic-quality-v2: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
