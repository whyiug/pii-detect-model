#!/usr/bin/env python3
"""Assemble publication-safe, release-bound model evidence.

The command consumes separate aggregate/self-hashed model-track and system-track
evaluation artifacts plus the selected checkpoint.  It never reads dataset rows
or prediction JSONL files.  Output is deterministic, path-free, and limited to
the evidence allowlist consumed by ``scripts/build_release.py`` (except for the
SBOM, which is generated separately).
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
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import canonical_json_hash  # noqa: E402
from pii_zh.taxonomy import validate_taxonomy_document  # noqa: E402

_EXPECTED_SEEDS = (13, 42, 97)
_EXPECTED_SUITES = ("synthetic_test", "pii_bench_formal", "pii_bench_chat")
_SELECTED_SEED = 42
_MODEL_ATTRIBUTABLE_TRACKS = frozenset({"model_raw", "model_calibrated"})
_MODEL_INDEX_PREFIX = {
    "model_raw": "Model Raw",
    "model_calibrated": "Model Calibrated",
}
_MAXIMUM_INPUT_BYTES = 64 * 1024 * 1024
_ASSEMBLED_FILES = (
    "calibration.json",
    "data_provenance.json",
    "evaluation_report.json",
    "id2label.json",
    "model-index.yml",
    "taxonomy.yaml",
    "teacher_provenance.json",
    "thresholds.yaml",
    "training_manifest.json",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_FORBIDDEN_RECORD_KEYS = {
    "doc_id",
    "document_id",
    "entity_value",
    "raw_text",
    "text",
}

# Release evidence is copied byte-for-byte so upstream digests remain meaningful.
# A self-hash proves integrity, not publication safety.  The schemas below are
# therefore keyed by the complete object path: a field approved inside one
# component cannot be reused as a top-level or unrelated nested extension.
_Path = tuple[str, ...]


def _fields(value: str) -> frozenset[str]:
    return frozenset(value.split())


_CALIBRATION_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "schema_version calibration_version global_temperature default_threshold "
        "entity_temperatures entity_thresholds model_version"
    ),
    ("entity_temperatures",): frozenset(),
    ("entity_thresholds",): frozenset(),
}

_LOADING_AUDIT_FIELDS = _fields(
    "attention_mode config_sha256 error_messages lm_head_discarded "
    "lm_head_present_in_checkpoint mismatched_keys missing_keys newly_initialized_score_keys "
    "safetensor_files unexpected_keys weights_sha256"
)
_SUMMARY_FIELDS = _fields(
    "data_pool_counts document_count entity_count label_counts pii_free_document_count "
    "public_weight_training_allowed_document_count quality_gate_passed_document_count "
    "quality_tier_counts sources split_counts token_count unalignable_boundary_count "
    "validators_passed_document_count"
)
_SUMMARY_SOURCE_FIELDS = _fields(
    "data_pool document_count entity_count license public_weight_training_allowed_document_count "
    "quality_gate_passed_document_count quality_tier_counts revision source_id source_kind split "
    "validators_passed_document_count"
)
_QUALITY_TIERS = frozenset({"G0", "S0", "S1", "S2", "U", "N"})
_DATA_POOLS = frozenset(
    {"public_release_pool", "private_enterprise_pool", "evaluation_only", "quarantined"}
)
_SPLITS = frozenset({"train", "validation", "test"})
_TRAINING_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "attention_mode base_checkpoint base_source_id code_revision completed_at created_at "
        "datasets fine_tuning initialization label2id label_schema_sha256 manifest_sha256 "
        "output_artifact privacy provenance_migration recipe recipe_sha256 runtime_environment "
        "schema_version seed status taxonomy_version tokenizer training_source_ids versions"
    ),
    ("base_checkpoint",): _fields("config_sha256 loading_audit safetensor_files weights_sha256"),
    ("base_checkpoint", "loading_audit"): _LOADING_AUDIT_FIELDS,
    ("datasets",): frozenset({"train", "validation"}),
    ("initialization",): _fields(
        "base_config_sha256 base_source_id base_weights_sha256 label_schema_sha256 "
        "mismatched_keys missing_keys score_keys source_architecture_sha256 "
        "source_attention_mode source_code_revision source_config_sha256 source_fine_tuning "
        "source_manifest_file_sha256 source_manifest_schema_version source_manifest_sha256 "
        "source_output_artifact_sha256 source_safetensor_files source_weights_sha256 strategy "
        "taxonomy_version tensor_count tensor_dtypes tokenizer_effective_contract_sha256 "
        "train_sha256 unexpected_keys validation_sha256"
    ),
    ("label2id",): frozenset(),
    ("output_artifact",): _fields(
        "artifact_files_combined_sha256 files format schema_version weight_files "
        "weights_combined_sha256"
    ),
    ("output_artifact", "files"): frozenset(
        {
            "added_tokens.json",
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    ),
    ("privacy",): _fields(
        "contains_entity_values contains_raw_text trainer_reporting_integrations"
    ),
    ("provenance_migration",): _fields(
        "changed_fields migration_code_revision migration_id model_weights_changed "
        "previous_manifest_file_sha256 previous_manifest_sha256 previous_schema_version reason "
        "tokenizer_changed training_data_changed training_recipe_changed"
    ),
    ("recipe",): _fields(
        "attention_mode base_source_id bf16 class_weight_cap class_weighting "
        "classifier_dropout classifier_learning_rate dataloader_num_workers document_sampling "
        "document_sampling_cap document_sampling_strategy early_stopping_patience "
        "effective_classifier_learning_rate epochs fine_tuning gradient_accumulation_steps "
        "gradient_checkpointing include_auxiliary_labels include_extended_labels "
        "initialization_strategy jpt_sep_token_id learning_rate lora max_length "
        "per_device_eval_batch_size per_device_train_batch_size resume save_total_limit seed "
        "use_cpu warmup_ratio weight_decay"
    ),
    ("recipe", "lora"): _fields("alpha dropout rank target_modules"),
    ("runtime_environment",): _fields(
        "bf16_enabled compute_capability cuda_available cuda_driver cuda_runtime "
        "cuda_visible_devices device_name execution_device selected_device_index "
        "visible_device_count"
    ),
    ("tokenizer",): _fields(
        "effective effective_contract_sha256 files source_files_combined_sha256"
    ),
    ("tokenizer", "effective"): _fields(
        "backend_sha256 boundary_mode special_token_ids vocab_size"
    ),
    ("tokenizer", "effective", "special_token_ids"): _fields(
        "bos_token_id eos_token_id pad_token_id unk_token_id"
    ),
    ("tokenizer", "files"): frozenset(
        {"merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    ),
    ("versions",): _fields("accelerate datasets peft pii_zh_qwen python torch transformers"),
}
for _split in ("train", "validation"):
    _summary_path = ("datasets", _split, "summary")
    _TRAINING_SCHEMA[("datasets", _split)] = _fields("sha256 summary")
    _TRAINING_SCHEMA[_summary_path] = _SUMMARY_FIELDS
    _TRAINING_SCHEMA[_summary_path + ("data_pool_counts",)] = _DATA_POOLS
    _TRAINING_SCHEMA[_summary_path + ("label_counts",)] = frozenset()
    _TRAINING_SCHEMA[_summary_path + ("quality_tier_counts",)] = _QUALITY_TIERS
    _TRAINING_SCHEMA[_summary_path + ("sources", "[]")] = _SUMMARY_SOURCE_FIELDS
    _TRAINING_SCHEMA[_summary_path + ("sources", "[]", "quality_tier_counts")] = _QUALITY_TIERS
    _TRAINING_SCHEMA[_summary_path + ("split_counts",)] = _SPLITS

_PROVENANCE_PRIVACY_FIELDS = _fields(
    "contains_document_ids contains_entity_values contains_paths "
    "contains_prompts_or_responses contains_raw_text"
)
_SOURCE_REGISTRY_FIELDS = _fields("registry_version schema_version sha256")
_DATA_PROVENANCE_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "artifact_type manifest_sha256 privacy schema_version source_registry sources "
        "total_sample_count training_manifest_file_sha256 training_manifest_sha256"
    ),
    ("privacy",): _PROVENANCE_PRIVACY_FIELDS,
    ("source_registry",): _SOURCE_REGISTRY_FIELDS,
    ("sources", "[]"): _fields(
        "contains_real_personal_data declared_license frozen_validation_lineage pool "
        "public_weight_training_allowed revision sample_count source_id source_kind split_sha256 "
        "split_source_revisions synthetic template_asset upstream_generation usage_counts"
    ),
    ("sources", "[]", "frozen_validation_lineage"): _fields(
        "copy_mode parent_dataset_version parent_manifest_file_sha256 parent_manifest_sha256"
    ),
    ("sources", "[]", "split_sha256"): frozenset({"train", "validation"}),
    ("sources", "[]", "split_source_revisions"): frozenset({"train", "validation"}),
    ("sources", "[]", "template_asset"): _fields(
        "asset_id asset_version curation_audit_sha256 revision"
    ),
    ("sources", "[]", "upstream_generation"): _fields(
        "accepted_candidate_count direct_pseudo_labeling direct_span_annotation role source_id"
    ),
    ("sources", "[]", "usage_counts"): _fields(
        "frozen_test_training_use gradient_training validation_early_stopping_and_calibration"
    ),
}
_TEACHER_PROVENANCE_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "artifact_type external_api_teacher_used external_api_used manifest_sha256 privacy "
        "schema_version source_registry teacher_used teachers training_manifest_file_sha256 "
        "training_manifest_sha256"
    ),
    ("privacy",): _PROVENANCE_PRIVACY_FIELDS,
    ("source_registry",): _SOURCE_REGISTRY_FIELDS,
    ("teachers", "[]"): _fields(
        "accepted_candidate_count accepted_placeholder_templates_only candidate_payload_sha256 "
        "curation_audit_sha256 declared_license external_api input_class pseudo_label_teacher "
        "raw_outputs_used_for_training rejected_candidate_count review_method "
        "reviewed_candidate_count revision role source_id span_teacher template_asset_revision "
        "used_for_training"
    ),
}

_REPORT_PRIVACY_FIELDS = _fields(
    "contains_document_ids contains_entity_values contains_paths contains_raw_text "
    "contains_record_level_data"
)
_CRITERION_FIELDS = _fields("name operator passed threshold value")
_COMMUNITY_REPORT_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "artifact_type dataset decision_config decision_id manifest_sha256 metric_set privacy "
        "production_ready quality_gate release_decision release_scope schema_version seeds "
        "system_contract"
    ),
    ("dataset",): _fields("frozen_test_access sha256 split"),
    ("decision_config",): _fields(
        "artifact_id file_sha256 parent_config_sha256 parent_decision_id"
    ),
    ("privacy",): _REPORT_PRIVACY_FIELDS,
    ("quality_gate",): _fields("criteria status"),
    ("quality_gate", "criteria", "[]"): _CRITERION_FIELDS,
    ("seeds", "[]"): _fields("criteria metrics provenance quality_gate_passed seed"),
    ("seeds", "[]", "criteria", "[]"): _CRITERION_FIELDS,
    ("seeds", "[]", "metrics"): _fields(
        "calibration_completed pii_free_false_positive_rate strict_macro_f1 strict_micro_f1 "
        "tier0_min_label_recall tier1_min_label_recall"
    ),
    ("seeds", "[]", "provenance"): _fields("calibration evaluation training"),
    ("seeds", "[]", "provenance", "calibration"): _fields(
        "artifact_id calibration_bundle_sha256 calibration_version file_sha256 fit_gold_sha256 "
        "fit_predictions_sha256 manifest_sha256"
    ),
    ("seeds", "[]", "provenance", "evaluation"): _fields(
        "artifact_id evaluation_sha256 file_sha256 predictions_sha256"
    ),
    ("seeds", "[]", "provenance", "training"): _fields(
        "artifact_id file_sha256 manifest_sha256 output_artifact_sha256"
    ),
    ("system_contract",): _fields(
        "calibration_holdout_policy fusion_id fusion_implementation_sha256 refinement "
        "refinement_id refinement_implementation_sha256 rules_implementation_sha256 "
        "ruleset_id runtime training_lineage"
    ),
    ("system_contract", "runtime"): _fields(
        "dependency_lock_sha256 tokenizer_backend_sha256 tokenizer_implementation_sha256 "
        "tokenizer_loader transformers_version"
    ),
    ("system_contract", "training_lineage"): _fields(
        "manifest_schema_version migration_id migration_implementation_sha256 "
        "required_training_source_ids source_registry_sha256"
    ),
}

_COUNT_METRIC_FIELDS = _fields("f1 f2 fn fp precision recall tp")
_SYSTEM_SUMMARY_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "aggregate artifact_type manifest_sha256 privacy schema_version suite_order suites "
        "system_identity"
    ),
    ("aggregate",): _fields(
        "bootstrap_intervals character_micro_pooled document_count gold_span_count "
        "predicted_span_count relaxed_micro_pooled strict_micro_pooled"
    ),
    ("aggregate", "character_micro_pooled"): _COUNT_METRIC_FIELDS,
    ("aggregate", "relaxed_micro_pooled"): _COUNT_METRIC_FIELDS,
    ("aggregate", "strict_micro_pooled"): _COUNT_METRIC_FIELDS,
    ("privacy",): _REPORT_PRIVACY_FIELDS,
    ("suites",): frozenset(_EXPECTED_SUITES),
    ("system_identity",): _fields(
        "attention_mode calibration_fit fusion_identity model_identity refinement_identity "
        "rules_identity selected_seed training"
    ),
    ("system_identity", "calibration_fit"): _fields(
        "bundle_sha256 calibration_version diagnostics_manifest_sha256 document_count "
        "fused_predictions_sha256 gold_sha256"
    ),
    ("system_identity", "fusion_identity"): _fields(
        "cli_sha256 configuration_sha256 implementation_sha256 module_sha256"
    ),
    ("system_identity", "model_identity"): _fields(
        "architecture_version config_sha256 identity_sha256 label_schema_sha256 model_type "
        "weights_sha256"
    ),
    ("system_identity", "refinement_identity"): _fields("implementation_sha256 refinement_id"),
    ("system_identity", "rules_identity"): _fields(
        "configuration_sha256 implementation_sha256 ruleset_id"
    ),
    ("system_identity", "training"): _fields("manifest_file_sha256 manifest_sha256"),
}
_SUITE_FIELDS = _fields("dataset evaluation_report_file_sha256 metrics system")
_SUITE_DATASET_FIELDS = _fields(
    "evaluation_only gold_sha256 license manifest_file_sha256 manifest_sha256 record_count "
    "source_id span_count subset upstream_revision upstream_source"
)
_SUITE_METRIC_FIELDS = _fields(
    "bootstrap character_micro document_count pii_free relaxed_micro span_count strict_macro_f1 "
    "strict_micro"
)
_SPAN_METRIC_FIELDS = _fields("f1 f2 fn fp precision predicted recall support tp")
_INTERVAL_FIELDS = _fields("effective_samples lower point upper")
_INTERVAL_NAMES = _fields(
    "boundary_exact_f1 character_micro_f1 pii_free_false_positive_rate relaxed_micro_f1 "
    "strict_macro_f1 strict_micro_f1"
)
for _suite in _EXPECTED_SUITES:
    _suite_path = ("suites", _suite)
    _SYSTEM_SUMMARY_SCHEMA[_suite_path] = _SUITE_FIELDS
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("dataset",)] = _SUITE_DATASET_FIELDS
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics",)] = _SUITE_METRIC_FIELDS
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics", "bootstrap")] = _fields(
        "confidence intervals method samples seed"
    )
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics", "bootstrap", "intervals")] = _INTERVAL_NAMES
    for _interval in _INTERVAL_NAMES:
        _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics", "bootstrap", "intervals", _interval)] = (
            _INTERVAL_FIELDS
        )
    for _metric in ("character_micro", "relaxed_micro", "strict_micro"):
        _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics", _metric)] = _SPAN_METRIC_FIELDS
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics", "pii_free")] = _fields(
        "documents false_positive_documents false_positive_rate"
    )
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("metrics", "span_count")] = _fields("gold predicted")
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("system",)] = _fields(
        "evaluation_sha256 predictions_sha256 refinement_audit_manifest_sha256 "
        "system_prediction_manifest_file_sha256 system_prediction_manifest_sha256 "
        "target_application"
    )
    _SYSTEM_SUMMARY_SCHEMA[_suite_path + ("system", "target_application")] = _fields(
        "input_predictions_sha256 output_predictions_sha256"
    )

# ``model-index.yml`` describes the released checkpoint, so its source evidence
# has a deliberately smaller component surface than the full-system summary.
# Rules, framework orchestration, fusion, refinement, and cascades are prohibited
# by the attribution contract below instead of being silently stripped later.
_MODEL_SUMMARY_SCHEMA: dict[_Path, frozenset[str]] = {
    (): _fields(
        "artifact_type attribution canonical_track manifest_sha256 model_identity privacy "
        "schema_version suite_order suites"
    ),
    ("attribution",): _fields(
        "calibration_applied cascade_applied checkpoint_output decoder_applied framework_applied "
        "fusion_applied output_stage refinement_applied rules_applied"
    ),
    ("model_identity",): _fields(
        "attention_mode calibration decoder selected_seed training"
    ),
    ("model_identity", "calibration"): _fields(
        "bundle_sha256 calibration_version diagnostics_manifest_sha256"
    ),
    ("model_identity", "decoder"): _fields("decoder_id implementation_sha256"),
    ("model_identity", "training"): _fields("manifest_file_sha256 manifest_sha256"),
    ("privacy",): _REPORT_PRIVACY_FIELDS,
    ("suites",): frozenset(_EXPECTED_SUITES),
}
for _suite in _EXPECTED_SUITES:
    _suite_path = ("suites", _suite)
    _MODEL_SUMMARY_SCHEMA[_suite_path] = _fields("dataset metrics model_output")
    _MODEL_SUMMARY_SCHEMA[_suite_path + ("dataset",)] = _SUITE_DATASET_FIELDS
    _MODEL_SUMMARY_SCHEMA[_suite_path + ("metrics",)] = _fields(
        "document_count pii_free strict_macro_f1 strict_micro"
    )
    _MODEL_SUMMARY_SCHEMA[_suite_path + ("metrics", "pii_free")] = _fields(
        "documents false_positive_documents false_positive_rate"
    )
    _MODEL_SUMMARY_SCHEMA[_suite_path + ("metrics", "strict_micro")] = _SPAN_METRIC_FIELDS
    _MODEL_SUMMARY_SCHEMA[_suite_path + ("model_output",)] = _fields(
        "evaluation_sha256 predictions_sha256"
    )

_TRAINING_PRIVACY = {
    "contains_entity_values": False,
    "contains_raw_text": False,
    "trainer_reporting_integrations": [],
}
_PROVENANCE_PRIVACY = {key: False for key in _PROVENANCE_PRIVACY_FIELDS}
_REPORT_PRIVACY = {key: False for key in _REPORT_PRIVACY_FIELDS}

_TRAINING_LIST_PATHS = frozenset(
    {
        ("base_checkpoint", "loading_audit", "error_messages"),
        ("base_checkpoint", "loading_audit", "mismatched_keys"),
        ("base_checkpoint", "loading_audit", "missing_keys"),
        ("base_checkpoint", "loading_audit", "newly_initialized_score_keys"),
        ("base_checkpoint", "loading_audit", "safetensor_files"),
        ("base_checkpoint", "loading_audit", "unexpected_keys"),
        ("base_checkpoint", "safetensor_files"),
        ("datasets", "train", "summary", "sources"),
        ("datasets", "validation", "summary", "sources"),
        ("initialization", "mismatched_keys"),
        ("initialization", "missing_keys"),
        ("initialization", "score_keys"),
        ("initialization", "source_safetensor_files"),
        ("initialization", "tensor_dtypes"),
        ("initialization", "unexpected_keys"),
        ("output_artifact", "weight_files"),
        ("privacy", "trainer_reporting_integrations"),
        ("provenance_migration", "changed_fields"),
        ("recipe", "lora", "target_modules"),
        ("training_source_ids",),
    }
)
_COMMUNITY_LIST_PATHS = frozenset(
    {
        ("metric_set",),
        ("quality_gate", "criteria"),
        ("seeds",),
        ("seeds", "[]", "criteria"),
        ("system_contract", "training_lineage", "required_training_source_ids"),
    }
)
_SYSTEM_LIST_PATHS = frozenset({("suite_order",)})
_DATA_PROVENANCE_LIST_PATHS = frozenset({("sources",)})
_TEACHER_PROVENANCE_LIST_PATHS = frozenset({("teachers",)})


class ReleaseEvidenceError(ValueError):
    """Raised when release evidence is incomplete, unsafe, or inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseEvidenceError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_regular(path: Path, *, description: str) -> tuple[bytes, str]:
    """Read one regular file descriptor and detect concurrent mutation."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot open regular {description}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseEvidenceError(f"{description} must be a regular file")
        if before.st_size > _MAXIMUM_INPUT_BYTES:
            raise ReleaseEvidenceError(f"{description} exceeds the evidence size limit")
        digest = hashlib.sha256()
        encoded = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            if len(encoded) + len(chunk) > _MAXIMUM_INPUT_BYTES:
                raise ReleaseEvidenceError(f"{description} exceeds the evidence size limit")
            encoded.extend(chunk)
            digest.update(chunk)
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
        if identity_before != identity_after:
            raise ReleaseEvidenceError(f"{description} changed while being read")
        return bytes(encoded), digest.hexdigest()
    finally:
        os.close(descriptor)


def sha256_file(path: Path, *, description: str = "file") -> str:
    """Hash a regular, non-symlink evidence file."""

    _, digest = _read_regular(path, description=description)
    return digest


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], str, bytes]:
    encoded, digest = _read_regular(path, description=description)
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceError(f"{description} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"{description} must be a JSON object")
    _assert_public_safe(value)
    return value, digest, encoded


def _verify_self_hash(value: Mapping[str, Any], *, description: str) -> str:
    claimed = value.get("manifest_sha256")
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise ReleaseEvidenceError(f"{description} lacks a valid manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if canonical_json_hash(unsigned) != claimed:
        raise ReleaseEvidenceError(f"{description} self-hash verification failed")
    return claimed


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError(f"{field} must be an object")
    return value


def _finite_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseEvidenceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ReleaseEvidenceError(f"{field} must be finite and between zero and one")
    return result


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReleaseEvidenceError(f"{field} must be a SHA-256 digest")
    return value


def _assert_public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        if keys & _FORBIDDEN_RECORD_KEYS:
            raise ReleaseEvidenceError("evidence contains a forbidden record-level field")
        for item in value.values():
            _assert_public_safe(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_public_safe(item)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ReleaseEvidenceError("evidence contains an absolute path")
        if ".." in normalized.split("/"):
            raise ReleaseEvidenceError("evidence contains an unsafe relative path")
        if len(value) > 2_048 or any(ord(character) < 32 for character in value):
            raise ReleaseEvidenceError("evidence contains an unsafe free-form string")


def _assert_structured_fields(
    value: object,
    *,
    schema: Mapping[_Path, frozenset[str]],
    list_paths: frozenset[_Path],
    dynamic_fields: Mapping[_Path, frozenset[str]] | None = None,
    description: str,
    path: _Path = (),
) -> None:
    """Validate every object, list, and field against its exact artifact path."""

    if isinstance(value, Mapping):
        if path not in schema:
            location = ".".join(path) or "<root>"
            raise ReleaseEvidenceError(f"{description} contains an unapproved object at {location}")
        allowed_fields = schema[path]
        allowed_dynamic = (dynamic_fields or {}).get(path, frozenset())
        for key, item in value.items():
            if not isinstance(key, str) or (
                key not in allowed_fields and key not in allowed_dynamic
            ):
                location = ".".join(path) or "<root>"
                raise ReleaseEvidenceError(
                    f"{description} contains an unapproved field {key!r} at {location}"
                )
            if key.startswith("contains_") and item is not False:
                location = ".".join((*path, key))
                raise ReleaseEvidenceError(f"{description}.{location} must explicitly remain false")
            if key == "trainer_reporting_integrations" and item != []:
                raise ReleaseEvidenceError(
                    "training evidence must not contain reporting integration metadata"
                )
            _assert_structured_fields(
                item,
                schema=schema,
                list_paths=list_paths,
                dynamic_fields=dynamic_fields,
                description=description,
                path=(*path, key),
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if path not in list_paths:
            location = ".".join(path) or "<root>"
            raise ReleaseEvidenceError(f"{description} contains an unapproved list at {location}")
        for item in value:
            _assert_structured_fields(
                item,
                schema=schema,
                list_paths=list_paths,
                dynamic_fields=dynamic_fields,
                description=description,
                path=(*path, "[]"),
            )
    elif path in schema:
        location = ".".join(path) or "<root>"
        raise ReleaseEvidenceError(f"{description}.{location} must be an object")
    elif path in list_paths:
        location = ".".join(path) or "<root>"
        raise ReleaseEvidenceError(f"{description}.{location} must be a list")


def _assert_privacy_declaration(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, object],
    description: str,
) -> None:
    declaration = _mapping(value.get("privacy"), field=f"{description}.privacy")
    missing = set(expected) - set(declaration)
    unknown = set(declaration) - set(expected)
    if missing or unknown:
        raise ReleaseEvidenceError(
            f"{description}.privacy does not match its required fields; "
            f"missing={sorted(missing)}, unapproved={sorted(unknown)}"
        )
    for key, expected_value in expected.items():
        if declaration[key] != expected_value:
            raise ReleaseEvidenceError(f"{description}.privacy.{key} is not publication-safe")


def _atomic_write_text(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ReleaseEvidenceError("output must not be a symlink")
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
            handle.write(content)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ReleaseEvidenceError("output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _assert_public_safe(value)
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _validate_seed_report(community: Mapping[str, Any]) -> list[dict[str, Any]]:
    if community.get("release_decision") != "passed":
        raise ReleaseEvidenceError("community validation release decision did not pass")
    if community.get("release_scope") != "community_research_release_candidate":
        raise ReleaseEvidenceError("community validation release scope is unexpected")
    if community.get("production_ready") is not False:
        raise ReleaseEvidenceError("community validation must not claim production readiness")
    quality = _mapping(community.get("quality_gate"), field="community.quality_gate")
    if quality.get("status") != "passed":
        raise ReleaseEvidenceError("community aggregate quality gate did not pass")
    raw_runs = community.get("seeds")
    if not isinstance(raw_runs, list) or any(not isinstance(item, dict) for item in raw_runs):
        raise ReleaseEvidenceError("community seed evidence is malformed")
    runs = [dict(item) for item in raw_runs]
    if tuple(item.get("seed") for item in runs) != _EXPECTED_SEEDS:
        raise ReleaseEvidenceError("community evidence does not contain the frozen three seeds")
    expected_metric_names: set[str] | None = None
    for run in runs:
        seed = run["seed"]
        if run.get("quality_gate_passed") is not True:
            raise ReleaseEvidenceError(f"seed {seed} did not pass its preregistered gate")
        metrics = _mapping(run.get("metrics"), field=f"seed {seed}.metrics")
        names = set(metrics)
        if not names or (expected_metric_names is not None and names != expected_metric_names):
            raise ReleaseEvidenceError("three-seed metric sets differ")
        expected_metric_names = names
        for name, value in metrics.items():
            _finite_probability(value, field=f"seed {seed}.{name}")
    return runs


def _validate_system_summary(
    summary: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    training_file_sha256: str,
    calibration: Mapping[str, Any],
    calibration_file_sha256: str,
) -> None:
    if tuple(summary.get("suite_order", ())) != _EXPECTED_SUITES:
        raise ReleaseEvidenceError("system summary suite order is not frozen")
    suites = _mapping(summary.get("suites"), field="system_summary.suites")
    if set(suites) != set(_EXPECTED_SUITES):
        raise ReleaseEvidenceError("system summary suite set is incomplete")
    identity = _mapping(summary.get("system_identity"), field="system_summary.system_identity")
    if identity.get("selected_seed") != _SELECTED_SEED or identity.get("attention_mode") != "full":
        raise ReleaseEvidenceError("system summary is not bound to selected seed42 Full model")
    bound_training = _mapping(identity.get("training"), field="system_identity.training")
    if (
        bound_training.get("manifest_sha256") != training.get("manifest_sha256")
        or bound_training.get("manifest_file_sha256") != training_file_sha256
    ):
        raise ReleaseEvidenceError("system summary/training manifest binding failed")
    fit = _mapping(identity.get("calibration_fit"), field="system_identity.calibration_fit")
    if fit.get("bundle_sha256") != calibration_file_sha256 or fit.get(
        "calibration_version"
    ) != calibration.get("calibration_version"):
        raise ReleaseEvidenceError("system summary/calibration binding failed")
    for suite in _EXPECTED_SUITES:
        metrics = _mapping(
            _mapping(suites[suite], field=f"suites.{suite}").get("metrics"),
            field=f"suites.{suite}.metrics",
        )
        strict = _mapping(metrics.get("strict_micro"), field=f"suites.{suite}.strict_micro")
        for metric in ("precision", "recall", "f1"):
            _finite_probability(strict.get(metric), field=f"suites.{suite}.strict_micro.{metric}")
    aggregate = _mapping(summary.get("aggregate"), field="system_summary.aggregate")
    pooled = _mapping(aggregate.get("strict_micro_pooled"), field="aggregate.strict_micro_pooled")
    for metric in ("precision", "recall", "f1"):
        _finite_probability(pooled.get(metric), field=f"aggregate.strict_micro_pooled.{metric}")


def _validate_model_summary_contract(summary: Mapping[str, Any]) -> str:
    """Validate that an aggregate can be attributed to checkpoint-only output."""

    _assert_structured_fields(
        summary,
        schema=_MODEL_SUMMARY_SCHEMA,
        list_paths=frozenset({("suite_order",)}),
        description="frozen model evaluation summary",
    )
    _assert_privacy_declaration(
        summary,
        expected=_REPORT_PRIVACY,
        description="frozen model evaluation summary",
    )
    _verify_self_hash(summary, description="frozen model evaluation summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("artifact_type") != "frozen_model_evaluation_summary"
    ):
        raise ReleaseEvidenceError("model-index source is not a frozen model evaluation summary")

    track = summary.get("canonical_track")
    if not isinstance(track, str) or track not in _MODEL_ATTRIBUTABLE_TRACKS:
        raise ReleaseEvidenceError(
            "model-index source track must be model_raw or model_calibrated"
        )
    if tuple(summary.get("suite_order", ())) != _EXPECTED_SUITES:
        raise ReleaseEvidenceError("model evaluation suite order is not frozen")
    suites = _mapping(summary.get("suites"), field="model_summary.suites")
    if set(suites) != set(_EXPECTED_SUITES):
        raise ReleaseEvidenceError("model evaluation suite set is incomplete")

    identity = _mapping(summary.get("model_identity"), field="model_summary.model_identity")
    if identity.get("selected_seed") != _SELECTED_SEED or identity.get("attention_mode") != "full":
        raise ReleaseEvidenceError("model evaluation is not bound to selected seed42 Full model")
    decoder = _mapping(identity.get("decoder"), field="model_identity.decoder")
    decoder_id = decoder.get("decoder_id")
    if not isinstance(decoder_id, str) or _SAFE_ID.fullmatch(decoder_id) is None:
        raise ReleaseEvidenceError("model evaluation decoder_id is not a stable identifier")
    _require_sha256(
        decoder.get("implementation_sha256"),
        field="model_identity.decoder.implementation_sha256",
    )

    expected_attribution = {
        "output_stage": track,
        "checkpoint_output": True,
        "decoder_applied": True,
        "calibration_applied": track == "model_calibrated",
        "rules_applied": False,
        "framework_applied": False,
        "cascade_applied": False,
        "fusion_applied": False,
        "refinement_applied": False,
    }
    attribution = _mapping(summary.get("attribution"), field="model_summary.attribution")
    if dict(attribution) != expected_attribution:
        raise ReleaseEvidenceError(
            "model-index source attribution includes a non-model component or mismatched stage"
        )

    for suite in _EXPECTED_SUITES:
        suite_summary = _mapping(suites[suite], field=f"model_summary.suites.{suite}")
        dataset = _mapping(suite_summary.get("dataset"), field=f"model_summary.{suite}.dataset")
        if dataset.get("evaluation_only") is not True:
            raise ReleaseEvidenceError(f"model_summary.{suite}.dataset is not evaluation-only")
        _require_sha256(dataset.get("gold_sha256"), field=f"model_summary.{suite}.gold_sha256")
        _require_sha256(
            dataset.get("manifest_sha256"), field=f"model_summary.{suite}.manifest_sha256"
        )
        _require_sha256(
            dataset.get("manifest_file_sha256"),
            field=f"model_summary.{suite}.manifest_file_sha256",
        )
        record_count = dataset.get("record_count")
        if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
            raise ReleaseEvidenceError(f"model_summary.{suite}.record_count must be positive")
        metrics = _mapping(suite_summary.get("metrics"), field=f"model_summary.{suite}.metrics")
        if metrics.get("document_count") != record_count:
            raise ReleaseEvidenceError(
                f"model_summary.{suite} metric and dataset document counts differ"
            )
        strict = _mapping(
            metrics.get("strict_micro"), field=f"model_summary.{suite}.strict_micro"
        )
        for metric in ("precision", "recall", "f1"):
            _finite_probability(
                strict.get(metric), field=f"model_summary.{suite}.strict_micro.{metric}"
            )
        model_output = _mapping(
            suite_summary.get("model_output"), field=f"model_summary.{suite}.model_output"
        )
        _require_sha256(
            model_output.get("evaluation_sha256"),
            field=f"model_summary.{suite}.model_output.evaluation_sha256",
        )
        _require_sha256(
            model_output.get("predictions_sha256"),
            field=f"model_summary.{suite}.model_output.predictions_sha256",
        )
    return track


def _validate_model_summary_binding(
    summary: Mapping[str, Any],
    *,
    system_summary: Mapping[str, Any],
    training: Mapping[str, Any],
    training_file_sha256: str,
    calibration: Mapping[str, Any],
    calibration_file_sha256: str,
    diagnostics_manifest_sha256: str,
) -> str:
    track = _validate_model_summary_contract(summary)
    identity = _mapping(summary.get("model_identity"), field="model_summary.model_identity")
    bound_training = _mapping(identity.get("training"), field="model_identity.training")
    if (
        bound_training.get("manifest_sha256") != training.get("manifest_sha256")
        or bound_training.get("manifest_file_sha256") != training_file_sha256
    ):
        raise ReleaseEvidenceError("model evaluation/training manifest binding failed")

    model_suites = _mapping(summary.get("suites"), field="model_summary.suites")
    system_suites = _mapping(system_summary.get("suites"), field="system_summary.suites")
    for suite in _EXPECTED_SUITES:
        model_dataset = _mapping(
            _mapping(model_suites[suite], field=f"model_summary.suites.{suite}").get("dataset"),
            field=f"model_summary.suites.{suite}.dataset",
        )
        system_dataset = _mapping(
            _mapping(system_suites[suite], field=f"system_summary.suites.{suite}").get("dataset"),
            field=f"system_summary.suites.{suite}.dataset",
        )
        if dict(model_dataset) != dict(system_dataset):
            raise ReleaseEvidenceError(
                f"model and system evaluations use different {suite} dataset identities"
            )

    calibration_identity = identity.get("calibration")
    if track == "model_raw":
        if calibration_identity is not None:
            raise ReleaseEvidenceError("model_raw evidence must not include calibration")
    else:
        bound_calibration = _mapping(
            calibration_identity, field="model_identity.calibration"
        )
        if (
            bound_calibration.get("bundle_sha256") != calibration_file_sha256
            or bound_calibration.get("calibration_version")
            != calibration.get("calibration_version")
            or bound_calibration.get("diagnostics_manifest_sha256")
            != diagnostics_manifest_sha256
        ):
            raise ReleaseEvidenceError("model_calibrated evidence/calibration binding failed")
    return track


def build_evaluation_report(
    community: Mapping[str, Any],
    system_summary: Mapping[str, Any],
    model_summary: Mapping[str, Any],
    *,
    community_file_sha256: str,
    system_summary_file_sha256: str,
    model_summary_file_sha256: str,
) -> dict[str, Any]:
    """Build the gate-compatible combined aggregate evaluation report."""

    runs = _validate_seed_report(community)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "combined_release_evaluation_report",
        "release_scope": "community_research_release_candidate",
        "production_ready": False,
        "release_decision": "passed",
        "selected_seed": _SELECTED_SEED,
        "seeds": runs,
        "quality_gate": dict(
            _mapping(community.get("quality_gate"), field="community.quality_gate")
        ),
        "validation_gate": {
            "decision_id": community.get("decision_id"),
            "manifest_sha256": community.get("manifest_sha256"),
            "file_sha256": community_file_sha256,
            "split": "validation",
            "purpose": "preregistered_three_seed_model_and_system_selection",
        },
        "frozen_model_evaluation": dict(model_summary),
        "frozen_model_evaluation_file_sha256": model_summary_file_sha256,
        "frozen_system_evaluation": dict(system_summary),
        "frozen_system_evaluation_file_sha256": system_summary_file_sha256,
        "coverage": {
            "measured": [
                "synthetic_v1_3_validation_three_seeds",
                "synthetic_v1_3_frozen_test_selected_seed42",
                "pii_bench_zh_formal_frozen_selected_seed42",
                "pii_bench_zh_chat_frozen_selected_seed42",
                "document_level_95_percent_bootstrap_intervals_per_frozen_suite",
            ],
            "not_measured": [
                "private_enterprise_gold",
                "tenant_holdout",
                "time_holdout",
                "dedicated_long_document_quality_benchmark",
                "production_presidio_end_to_end_quality_and_latency",
                "demographic_fairness",
            ],
            "claim_boundary": (
                "Research release candidate; public synthetic and public evaluation-only "
                "evidence must not be treated as a production deployment guarantee."
            ),
        },
        "privacy": {
            "contains_paths": False,
            "contains_document_ids": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_record_level_data": False,
        },
    }
    _assert_public_safe(report)
    report["manifest_sha256"] = canonical_json_hash(report)
    return report


def build_threshold_document(
    calibration: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    thresholds = _mapping(
        calibration.get("entity_thresholds"), field="calibration.entity_thresholds"
    )
    result = {
        "schema_version": 1,
        "calibration_version": calibration.get("calibration_version"),
        "score_space": "temperature_scaled_probability",
        "selection_split": "synthetic_v1_3_validation",
        "selection_gold_sha256": _mapping(
            diagnostics.get("inputs"), field="diagnostics.inputs"
        ).get("gold_sha256"),
        "holdout_policy": "apply_only_no_refit",
        "default_threshold": _finite_probability(
            calibration.get("default_threshold"), field="calibration.default_threshold"
        ),
        "entity_thresholds": {
            str(label): _finite_probability(value, field=f"thresholds.{label}")
            for label, value in sorted(thresholds.items())
        },
    }
    return result


def build_model_index(model_summary: Mapping[str, Any], *, release_name: str) -> dict[str, Any]:
    """Build a model index from an explicitly model-attributable track."""

    track = _validate_model_summary_contract(model_summary)
    metric_prefix = _MODEL_INDEX_PREFIX[track]
    suites = _mapping(model_summary.get("suites"), field="model_summary.suites")
    dataset_names = {
        "synthetic_test": "Synthetic v1.3 frozen test",
        "pii_bench_formal": "PII Bench zh formal (frozen)",
        "pii_bench_chat": "PII Bench zh chat (frozen)",
    }
    results: list[dict[str, Any]] = []
    for suite in _EXPECTED_SUITES:
        metrics = _mapping(
            _mapping(suites[suite], field=f"suites.{suite}").get("metrics"),
            field=f"suites.{suite}.metrics",
        )
        strict = _mapping(metrics.get("strict_micro"), field=f"suites.{suite}.strict_micro")
        results.append(
            {
                "task": {"name": "Token Classification", "type": "token-classification"},
                "dataset": {
                    "name": dataset_names[suite],
                    "type": suite,
                    "split": "test",
                },
                "metrics": [
                    {
                        "name": f"{metric_prefix} Strict Span Micro F1",
                        "type": "strict_span_f1",
                        "value": float(strict["f1"]),
                    },
                    {
                        "name": f"{metric_prefix} Strict Span Precision",
                        "type": "strict_span_precision",
                        "value": float(strict["precision"]),
                    },
                    {
                        "name": f"{metric_prefix} Strict Span Recall",
                        "type": "strict_span_recall",
                        "value": float(strict["recall"]),
                    },
                ],
            }
        )
    return {"model-index": [{"name": release_name, "results": results}]}


def assemble(
    *,
    checkpoint_dir: Path,
    taxonomy_path: Path,
    training_manifest_path: Path,
    calibration_path: Path,
    calibration_diagnostics_path: Path,
    community_report_path: Path,
    system_summary_path: Path,
    model_summary_path: Path | None = None,
    data_provenance_path: Path,
    teacher_provenance_path: Path,
    output_dir: Path,
    release_name: str,
) -> dict[str, str]:
    if model_summary_path is None:
        raise ReleaseEvidenceError("frozen model-track evaluation evidence is required")
    output_dir = output_dir.expanduser().absolute()
    cursor = output_dir
    while True:
        if cursor.is_symlink():
            raise ReleaseEvidenceError("output directory and its ancestors must not be symlinks")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    training, training_file_sha256, training_bytes = _load_json(
        training_manifest_path, description="training manifest"
    )
    training_sha256 = _verify_self_hash(training, description="training manifest")
    if (
        training.get("schema_version") != 4
        or training.get("status") != "completed"
        or training.get("seed") != _SELECTED_SEED
        or training.get("attention_mode") != "full"
    ):
        raise ReleaseEvidenceError(
            "selected training manifest is not completed seed42 Full schema4"
        )

    config, _, _ = _load_json(checkpoint_dir / "config.json", description="checkpoint config")
    id2label_raw = _mapping(config.get("id2label"), field="checkpoint.config.id2label")
    id2label = {str(key): value for key, value in id2label_raw.items()}
    expected_keys = [str(index) for index in range(len(id2label))]
    if sorted(id2label, key=int) != expected_keys or len(set(id2label.values())) != len(id2label):
        raise ReleaseEvidenceError("checkpoint id2label is not contiguous and unique")
    id2label = dict(sorted(id2label.items(), key=lambda item: int(item[0])))

    taxonomy_bytes, _ = _read_regular(taxonomy_path, description="taxonomy")
    try:
        taxonomy_document = yaml.safe_load(taxonomy_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReleaseEvidenceError("taxonomy must be UTF-8 YAML") from exc
    if not isinstance(taxonomy_document, Mapping):
        raise ReleaseEvidenceError("taxonomy must be a YAML object")
    _assert_public_safe(taxonomy_document)
    taxonomy = validate_taxonomy_document(taxonomy_document)
    expected_labels = ["O"]
    for entity in taxonomy.label_sets["core"]:
        expected_labels.extend((f"B-{entity.name}", f"I-{entity.name}"))
    if list(id2label.values()) != expected_labels:
        raise ReleaseEvidenceError(
            "checkpoint id2label does not equal the complete ordered core BIO contract"
        )
    entity_names = frozenset(entity.name for entity in taxonomy.label_sets["core"])
    bio_fields = frozenset(expected_labels) | entity_names
    training_dynamic_fields: dict[_Path, frozenset[str]] = {
        ("label2id",): bio_fields,
        ("datasets", "train", "summary", "label_counts"): entity_names,
        ("datasets", "validation", "summary", "label_counts"): entity_names,
    }
    _assert_structured_fields(
        training,
        schema=_TRAINING_SCHEMA,
        list_paths=_TRAINING_LIST_PATHS,
        dynamic_fields=training_dynamic_fields,
        description="training manifest",
    )
    _assert_privacy_declaration(
        training,
        expected=_TRAINING_PRIVACY,
        description="training manifest",
    )

    calibration, calibration_file_sha256, calibration_bytes = _load_json(
        calibration_path, description="calibration bundle"
    )
    calibration_dynamic_fields: dict[_Path, frozenset[str]] = {
        ("entity_temperatures",): entity_names,
        ("entity_thresholds",): entity_names,
    }
    _assert_structured_fields(
        calibration,
        schema=_CALIBRATION_SCHEMA,
        list_paths=frozenset(),
        dynamic_fields=calibration_dynamic_fields,
        description="calibration bundle",
    )
    thresholds = _mapping(
        calibration.get("entity_thresholds"), field="calibration.entity_thresholds"
    )
    calibration_labels = set(thresholds)
    if calibration_labels != taxonomy.core_label_names:
        raise ReleaseEvidenceError("calibration thresholds do not cover exactly the core labels")
    for label, threshold in thresholds.items():
        _finite_probability(threshold, field=f"calibration.entity_thresholds.{label}")
    entity_temperatures = _mapping(
        calibration.get("entity_temperatures", {}), field="calibration.entity_temperatures"
    )
    for label, temperature in entity_temperatures.items():
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0.0
        ):
            raise ReleaseEvidenceError(
                f"calibration.entity_temperatures.{label} must be finite and positive"
            )
    _finite_probability(calibration.get("default_threshold"), field="calibration.default_threshold")
    calibration_version = calibration.get("calibration_version")
    if not isinstance(calibration_version, str) or _SAFE_ID.fullmatch(calibration_version) is None:
        raise ReleaseEvidenceError("calibration version must be a path-free stable identifier")
    global_temperature = calibration.get("global_temperature")
    if (
        isinstance(global_temperature, bool)
        or not isinstance(global_temperature, (int, float))
        or not math.isfinite(global_temperature)
        or global_temperature <= 0.0
    ):
        raise ReleaseEvidenceError("global calibration temperature must be finite and positive")
    diagnostics, _, _ = _load_json(
        calibration_diagnostics_path, description="calibration diagnostics"
    )
    diagnostics_sha256 = _verify_self_hash(diagnostics, description="calibration diagnostics")
    if diagnostics.get("calibration_bundle_sha256") != calibration_file_sha256 or diagnostics.get(
        "calibration_version"
    ) != calibration.get("calibration_version"):
        raise ReleaseEvidenceError("calibration bundle/diagnostics binding failed")

    community, community_file_sha256, _ = _load_json(
        community_report_path, description="community validation report"
    )
    _assert_structured_fields(
        community,
        schema=_COMMUNITY_REPORT_SCHEMA,
        list_paths=_COMMUNITY_LIST_PATHS,
        description="community validation report",
    )
    _assert_privacy_declaration(
        community,
        expected=_REPORT_PRIVACY,
        description="community validation report",
    )
    _verify_self_hash(community, description="community validation report")
    system_summary, system_summary_file_sha256, _ = _load_json(
        system_summary_path, description="frozen system evaluation summary"
    )
    _assert_structured_fields(
        system_summary,
        schema=_SYSTEM_SUMMARY_SCHEMA,
        list_paths=_SYSTEM_LIST_PATHS,
        description="frozen system evaluation summary",
    )
    _assert_privacy_declaration(
        system_summary,
        expected=_REPORT_PRIVACY,
        description="frozen system evaluation summary",
    )
    _verify_self_hash(system_summary, description="frozen system evaluation summary")
    _validate_seed_report(community)
    _validate_system_summary(
        system_summary,
        training=training,
        training_file_sha256=training_file_sha256,
        calibration=calibration,
        calibration_file_sha256=calibration_file_sha256,
    )
    model_summary, model_summary_file_sha256, _ = _load_json(
        model_summary_path, description="frozen model evaluation summary"
    )
    _validate_model_summary_binding(
        model_summary,
        system_summary=system_summary,
        training=training,
        training_file_sha256=training_file_sha256,
        calibration=calibration,
        calibration_file_sha256=calibration_file_sha256,
        diagnostics_manifest_sha256=diagnostics_sha256,
    )

    inputs = _mapping(diagnostics.get("inputs"), field="calibration diagnostics.inputs")
    dataset = _mapping(community.get("dataset"), field="community.dataset")
    fit_gold_sha256 = inputs.get("gold_sha256")
    if (
        not isinstance(fit_gold_sha256, str)
        or _SHA256.fullmatch(fit_gold_sha256) is None
        or fit_gold_sha256 != dataset.get("sha256")
        or inputs.get("document_count") != 2_000
    ):
        raise ReleaseEvidenceError("calibration was not fit on the gated validation identity")
    system_fit = _mapping(
        _mapping(system_summary.get("system_identity"), field="system_identity").get(
            "calibration_fit"
        ),
        field="system_identity.calibration_fit",
    )
    if system_fit.get("diagnostics_manifest_sha256") != diagnostics_sha256:
        raise ReleaseEvidenceError("system summary/calibration diagnostics binding failed")

    data_provenance, _, data_provenance_bytes = _load_json(
        data_provenance_path, description="data provenance"
    )
    teacher_provenance, _, teacher_provenance_bytes = _load_json(
        teacher_provenance_path, description="teacher provenance"
    )
    _assert_structured_fields(
        data_provenance,
        schema=_DATA_PROVENANCE_SCHEMA,
        list_paths=_DATA_PROVENANCE_LIST_PATHS,
        description="data provenance",
    )
    _assert_privacy_declaration(
        data_provenance,
        expected=_PROVENANCE_PRIVACY,
        description="data provenance",
    )
    _assert_structured_fields(
        teacher_provenance,
        schema=_TEACHER_PROVENANCE_SCHEMA,
        list_paths=_TEACHER_PROVENANCE_LIST_PATHS,
        description="teacher provenance",
    )
    _assert_privacy_declaration(
        teacher_provenance,
        expected=_PROVENANCE_PRIVACY,
        description="teacher provenance",
    )
    _verify_self_hash(data_provenance, description="data provenance")
    _verify_self_hash(teacher_provenance, description="teacher provenance")
    for name, provenance in (
        ("data", data_provenance),
        ("teacher", teacher_provenance),
    ):
        if (
            provenance.get("training_manifest_sha256") != training_sha256
            or provenance.get("training_manifest_file_sha256") != training_file_sha256
        ):
            raise ReleaseEvidenceError(f"{name} provenance/training binding failed")

    evaluation_report = build_evaluation_report(
        community,
        system_summary,
        model_summary,
        community_file_sha256=community_file_sha256,
        system_summary_file_sha256=system_summary_file_sha256,
        model_summary_file_sha256=model_summary_file_sha256,
    )
    thresholds = build_threshold_document(calibration, diagnostics)
    model_index = build_model_index(model_summary, release_name=release_name)
    _assert_public_safe(evaluation_report)
    _assert_public_safe(thresholds)
    _assert_public_safe(model_index)

    _atomic_write_bytes(output_dir / "taxonomy.yaml", taxonomy_bytes)
    _write_json(output_dir / "id2label.json", id2label)
    _atomic_write_bytes(output_dir / "calibration.json", calibration_bytes)
    _atomic_write_text(
        output_dir / "thresholds.yaml",
        yaml.safe_dump(thresholds, allow_unicode=True, sort_keys=True),
    )
    _atomic_write_bytes(output_dir / "training_manifest.json", training_bytes)
    _atomic_write_bytes(output_dir / "data_provenance.json", data_provenance_bytes)
    _atomic_write_bytes(output_dir / "teacher_provenance.json", teacher_provenance_bytes)
    _write_json(output_dir / "evaluation_report.json", evaluation_report)
    _atomic_write_text(
        output_dir / "model-index.yml",
        yaml.safe_dump(model_index, allow_unicode=True, sort_keys=False),
    )
    return {
        name: sha256_file(output_dir / name, description=f"assembled {name}")
        for name in _ASSEMBLED_FILES
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-diagnostics", type=Path, required=True)
    parser.add_argument("--community-report", type=Path, required=True)
    parser.add_argument("--system-summary", type=Path, required=True)
    parser.add_argument("--model-summary", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, required=True)
    parser.add_argument("--teacher-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--release-name",
        default="zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        hashes = assemble(
            checkpoint_dir=args.checkpoint_dir,
            taxonomy_path=args.taxonomy,
            training_manifest_path=args.training_manifest,
            calibration_path=args.calibration,
            calibration_diagnostics_path=args.calibration_diagnostics,
            community_report_path=args.community_report,
            system_summary_path=args.system_summary,
            model_summary_path=args.model_summary,
            data_provenance_path=args.data_provenance,
            teacher_provenance_path=args.teacher_provenance,
            output_dir=args.output_dir,
            release_name=args.release_name,
        )
    except (OSError, ReleaseEvidenceError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps({"files": hashes, "status": "assembled"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
