#!/usr/bin/env python3
"""Select the frozen v3 cross-seed candidate for calibration, never release.

The selector has no public-benchmark or prediction input. It verifies the
byte-pinned v3 protocol, the validation-informed development corpus, the
independent post-selection evaluation manifest, three completed training
candidates, merged model bytes, and Trainer-selected checkpoints. The output
only authorizes reading the frozen calibration split; it does not make any
quality or release-readiness claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from scripts import select_aiguard24_sota_candidate as common  # noqa: E402

from pii_zh.training.manifest import (  # noqa: E402
    canonical_json_hash,
    output_artifact_fingerprint,
    verify_training_manifest,
)

FROZEN_PROTOCOL_FILE_SHA256 = (
    "9f330eca4dbc6e522fc52b06f6bfb7c4047e455a0b3be555a94a03e6dd4ca51e"
)
FROZEN_PROTOCOL_ID = "aiguard24_synthetic_sota_v3_resplit_dev_selection"
FROZEN_PROTOCOL_STATUS = (
    "frozen_after_full_v2_seed42_validation_before_resplit_v3_training"
)
SELECTION_STATUS = "SELECTED_FOR_CALIBRATION_NOT_RELEASE"
_EXPECTED_MANIFEST_TYPE = "aiguard24_community_training_v1"
_EXPECTED_PUBLIC_BENCHMARK = "wan9yu/pii-bench-zh"
_EXPECTED_RELEASE_EVAL_DATASET = "pii_zh_synthetic_sota_release_eval_v1"
_PUBLIC_FALSE_KEYS = (
    "read_for_training",
    "read_for_validation",
    "read_for_checkpoint_selection",
    "read_for_threshold_tuning",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

CandidateSelectionError = common.CandidateSelectionError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--release-evaluation-manifest", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _object(value: Any, *, field: str) -> dict[str, Any]:
    return common._object(value, field=field)


def _array(value: Any, *, field: str) -> list[Any]:
    return common._array(value, field=field)


def _sha256(value: Any, *, field: str) -> str:
    return common._sha256(value, field=field)


def _finite(value: Any, *, field: str) -> float:
    return common._finite_number(value, field=field)


def _load_json(path: Path, *, field: str) -> tuple[dict[str, Any], str]:
    return common._load_regular_json(path, field=field)


def _verify_self_hash(value: Mapping[str, Any], *, field: str) -> str:
    return common._verify_self_hash(
        value, field=field, hash_field="manifest_sha256"
    )


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise CandidateSelectionError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateSelectionError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CandidateSelectionError(f"{field} must use UTC")
    return parsed


def _expected_recipe() -> dict[str, Any]:
    return {
        "attention_mode": "full",
        "fine_tuning": "lora_merged",
        "epochs": 2.0,
        "max_steps": -1,
        "max_length": 512,
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.00002,
        "classifier_learning_rate": 0.0002,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "class_weighting": False,
        "document_sampling": False,
        "gradient_checkpointing": True,
        "bf16": True,
        "initialization_strategy": (
            "aiguard24_projection_then_attention_conversion_v1"
        ),
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_metric": "risk_weighted_score",
        "min_pii_free_ratio": 0.4,
        "split_isolation_policy": "template-overlap-development-v1",
    }


def _expected_order() -> list[dict[str, str]]:
    return [
        {"metric": "eval_risk_weighted_score", "direction": "maximize"},
        {"metric": "eval_strict_micro_f1", "direction": "maximize"},
        {"metric": "eval_strict_macro_f1", "direction": "maximize"},
        {"metric": "eval_pii_free_precision", "direction": "maximize"},
        {"metric": "seed", "direction": "minimize"},
    ]


def _load_protocol(
    path: Path, *, expected_file_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    protocol, file_hash = _load_json(path, field="v3 selection protocol")
    frozen_hash = expected_file_sha256 or FROZEN_PROTOCOL_FILE_SHA256
    if file_hash != frozen_hash:
        raise CandidateSelectionError("v3 selection protocol file hash is not frozen")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != FROZEN_PROTOCOL_ID
        or protocol.get("status") != FROZEN_PROTOCOL_STATUS
    ):
        raise CandidateSelectionError("v3 selection protocol identity/status is invalid")
    _parse_utc_timestamp(protocol.get("frozen_at"), field="protocol frozen_at")
    if protocol.get("freeze_assertions") != {
        "v3_training_started_before_freeze": False,
        "v3_candidate_result_observed_before_freeze": False,
        "protocol_changes_based_on_v3_results": False,
        "public_benchmark_artifacts_read_before_freeze": False,
    }:
        raise CandidateSelectionError("protocol freeze assertions are invalid")

    history = _object(protocol.get("design_history"), field="protocol design history")
    if (
        history.get("validation_informed_data_change") is not True
        or history.get("optimizer_or_lora_change_from_v2") is not False
        or history.get("public_benchmark_read_for_change") is not False
        or history.get("release_test_evidence_used_for_change") is not False
    ):
        raise CandidateSelectionError("protocol design-history boundary is invalid")
    seeds = _array(protocol.get("candidate_seeds"), field="protocol candidate seeds")
    if seeds != [13, 42, 97] or any(isinstance(seed, bool) for seed in seeds):
        raise CandidateSelectionError("protocol candidate seeds are invalid")
    if _object(protocol.get("fixed_recipe"), field="protocol fixed recipe") != _expected_recipe():
        raise CandidateSelectionError("protocol fixed recipe changed")
    if protocol.get("within_seed_checkpoint_selection") != {
        "primary_metric": "eval_risk_weighted_score",
        "direction": "maximize",
        "performed_by_transformers_load_best_model_at_end": True,
    }:
        raise CandidateSelectionError("within-seed selection policy changed")

    cross_seed = _object(
        protocol.get("cross_seed_selection"), field="protocol cross-seed selection"
    )
    if (
        cross_seed.get("candidate_quality_guardrails") != []
        or cross_seed.get("order") != _expected_order()
        or cross_seed.get("output_status") != SELECTION_STATUS
        or cross_seed.get("release_eligibility_after_selection") is not False
        or cross_seed.get("selection_must_not_be_revisited_after_calibration_or_evaluation")
        is not True
    ):
        raise CandidateSelectionError("cross-seed calibration selection policy changed")

    public = _object(
        protocol.get("public_benchmark_policy"), field="public benchmark policy"
    )
    if (
        public.get("dataset_id") != _EXPECTED_PUBLIC_BENCHMARK
        or any(public.get(key) is not False for key in _PUBLIC_FALSE_KEYS)
        or public.get(
            "run_at_most_once_after_cross_seed_selection_and_calibration_freeze"
        )
        is not True
        or public.get("descriptive_only") is not True
        or public.get("selection_or_tuning_allowed_from_result") is not False
    ):
        raise CandidateSelectionError("public benchmark isolation policy changed")
    claims = _object(protocol.get("claims_policy"), field="protocol claims policy")
    if (
        claims.get("selected_candidate_is_release_ready") is not False
        or claims.get("production_ready") is not False
        or claims.get("real_world_sota") is not False
        or claims.get("global_first_claim_allowed") is not False
    ):
        raise CandidateSelectionError("protocol claims policy is unsafe")
    _verify_post_selection_policy(protocol)
    return protocol, {
        "file_sha256": file_hash,
        "canonical_sha256": canonical_json_hash(protocol),
    }


def _verify_post_selection_policy(protocol: Mapping[str, Any]) -> None:
    evaluation = _object(
        protocol.get("post_selection_evaluation"),
        field="post-selection evaluation policy",
    )
    if (
        evaluation.get("dataset_id") != _EXPECTED_RELEASE_EVAL_DATASET
        or evaluation.get("dataset_version") != "1.0.0"
    ):
        raise CandidateSelectionError("post-selection evaluation identity changed")
    for name in ("manifest_file_sha256", "manifest_sha256"):
        _sha256(evaluation.get(name), field=f"post-selection evaluation {name}")
    calibration = _object(evaluation.get("calibration"), field="calibration policy")
    internal = _object(
        evaluation.get("internal_evaluation"), field="internal evaluation policy"
    )
    _sha256(calibration.get("sha256"), field="calibration sha256")
    _sha256(internal.get("sha256"), field="internal evaluation sha256")
    expected_calibration = {
        "records": 10000,
        "may_read_only_after_cross_seed_selection_receipt": True,
        "gradient_training_allowed": False,
        "checkpoint_or_model_selection_allowed": False,
        "threshold_tuning_allowed": True,
        "temperature_scaling_allowed": True,
        "fusion_calibration_allowed": True,
        "final_metric_claim_allowed": False,
    }
    expected_internal = {
        "records": 10000,
        "may_read_only_after_model_and_calibration_artifacts_are_frozen": True,
        "one_shot_final_evaluation": True,
        "gradient_training_allowed": False,
        "checkpoint_or_model_selection_allowed": False,
        "threshold_tuning_allowed": False,
        "temperature_scaling_allowed": False,
        "fusion_calibration_allowed": False,
        "final_metric_claim_allowed": True,
        "result_informed_model_or_threshold_change_allowed": False,
    }
    if {key: calibration.get(key) for key in expected_calibration} != expected_calibration:
        raise CandidateSelectionError("calibration isolation policy changed")
    if {key: internal.get(key) for key in expected_internal} != expected_internal:
        raise CandidateSelectionError("internal evaluation isolation policy changed")


def _file_inventory(value: Any, *, field: str) -> dict[str, dict[str, Any]]:
    files = _array(value, field=field)
    result: dict[str, dict[str, Any]] = {}
    for item in files:
        entry = _object(item, field=f"{field} entry")
        split = entry.get("split")
        if not isinstance(split, str) or split in result:
            raise CandidateSelectionError(f"{field} split inventory is invalid")
        result[split] = entry
    return result


def _verify_training_dataset_manifest(
    path: Path, *, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest, file_hash = _load_json(path, field="v3 training dataset manifest")
    manifest_hash = _verify_self_hash(manifest, field="v3 training dataset manifest")
    expected = _object(protocol.get("dataset"), field="protocol dataset")
    if (
        file_hash != expected.get("manifest_file_sha256")
        or manifest_hash != expected.get("manifest_sha256")
        or manifest.get("manifest_schema_version") != 1
        or manifest.get("manifest_type") != expected.get("manifest_type")
        or manifest.get("dataset_id") != expected.get("id")
        or manifest.get("dataset_version") != expected.get("version")
        or manifest.get("license") != "Apache-2.0"
        or manifest.get("language") != "zh-Hans-CN"
        or manifest.get("data_pool") != "public_release_pool"
        or manifest.get("public_weight_training_allowed") is not True
        or manifest.get("release_test_eligible") is not False
    ):
        raise CandidateSelectionError("training dataset identity or content hash changed")
    if manifest.get("roles") != {
        "train": "weight_training",
        "validation": "development_checkpoint_selection_and_calibration_only",
        "release_test_eligible": False,
        "public_benchmark_read": False,
    }:
        raise CandidateSelectionError("training dataset roles changed")
    if manifest.get("benchmark_isolation") != {
        "pii_bench_zh_read": False,
        "public_benchmark_prediction_read": False,
    }:
        raise CandidateSelectionError("training dataset benchmark isolation changed")
    if manifest.get("resplit") != {
        "strategy": "template-and-pii-state-stratified-hash-v1",
        "salt": "pii-zh-synthetic-sota-v2-resplit-20260718",
        "validation_to_train_ratio": 0.8,
        "allow_template_group_overlap": True,
        "require_document_id_disjoint": True,
        "require_entity_value_group_disjoint": True,
        "require_source_group_disjoint": True,
    }:
        raise CandidateSelectionError("training dataset resplit policy changed")
    counts = _object(manifest.get("counts"), field="training dataset counts")
    if (
        counts.get("by_split")
        != {
            "train": expected.get("train_records"),
            "validation": expected.get("validation_records"),
        }
        or counts.get("total")
        != int(expected.get("train_records", -1))
        + int(expected.get("validation_records", -1))
    ):
        raise CandidateSelectionError("training dataset counts changed")
    files = _file_inventory(manifest.get("files"), field="training dataset files")
    if set(files) != {"train", "validation"}:
        raise CandidateSelectionError("training dataset split inventory changed")
    for split in ("train", "validation"):
        entry = files[split]
        if (
            entry.get("name") != f"{split}.jsonl"
            or entry.get("records") != expected.get(f"{split}_records")
            or entry.get("sha256") != expected.get(f"{split}_sha256")
            or entry.get("mode") != "0444"
        ):
            raise CandidateSelectionError("training dataset file binding changed")
    audit = _object(manifest.get("audit"), field="training dataset audit")
    if (
        audit.get("cross_split_collisions")
        != expected.get("required_cross_split_collisions")
        or audit.get("template_group_overlap_allowed") is not True
        or audit.get("template_group_overlap_count")
        != expected.get("template_group_overlap_count")
        or audit.get("pii_free")
        != {
            "train": expected.get("train_pii_free_records"),
            "validation": expected.get("validation_pii_free_records"),
        }
    ):
        raise CandidateSelectionError("training dataset isolation audit changed")
    label_counts = _object(audit.get("label_counts"), field="training label counts")
    if set(label_counts) != {"train", "validation"} or any(
        not isinstance(label_counts[split], dict)
        or len(label_counts[split]) != 24
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in label_counts[split].values()
        )
        for split in ("train", "validation")
    ):
        raise CandidateSelectionError("training dataset lost 24-label coverage")
    return manifest, {
        "file_sha256": file_hash,
        "manifest_sha256": manifest_hash,
    }


def _verify_release_evaluation_manifest(
    path: Path, *, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest, file_hash = _load_json(path, field="release evaluation manifest")
    manifest_hash = _verify_self_hash(manifest, field="release evaluation manifest")
    expected = _object(
        protocol.get("post_selection_evaluation"), field="post-selection evaluation"
    )
    if (
        file_hash != expected.get("manifest_file_sha256")
        or manifest_hash != expected.get("manifest_sha256")
        or manifest.get("manifest_schema_version") != 1
        or manifest.get("dataset_id") != expected.get("dataset_id")
        or manifest.get("dataset_version") != expected.get("dataset_version")
        or manifest.get("record_data_pool") != "evaluation_only"
        or manifest.get("public_weight_training_allowed") is not False
        or manifest.get("public_artifact_release_allowed") is not True
    ):
        raise CandidateSelectionError("release evaluation manifest binding changed")
    generation = _object(manifest.get("generation"), field="release evaluation generation")
    if (
        generation.get("contains_real_personal_data") is not False
        or generation.get("external_dataset_inputs") != []
        or generation.get("protected_evaluation_data_read") is not False
        or generation.get("seeds_independent") is not True
    ):
        raise CandidateSelectionError("release evaluation generation boundary changed")
    files = _file_inventory(manifest.get("files"), field="release evaluation files")
    if set(files) != {"calibration", "internal_evaluation"}:
        raise CandidateSelectionError("release evaluation split inventory changed")
    for split in ("calibration", "internal_evaluation"):
        policy = _object(expected.get(split), field=f"protocol {split}")
        entry = files[split]
        if (
            entry.get("name") != f"{split}.jsonl"
            or entry.get("records") != policy.get("records")
            or entry.get("sha256") != policy.get("sha256")
            or entry.get("mode") != "0444"
        ):
            raise CandidateSelectionError("release evaluation file binding changed")
    usage = _object(manifest.get("usage_policy"), field="release evaluation usage policy")
    for split in ("calibration", "internal_evaluation"):
        protocol_policy = _object(expected.get(split), field=f"protocol {split}")
        manifest_policy = _object(usage.get(split), field=f"manifest {split} policy")
        common_keys = set(manifest_policy)
        if {key: protocol_policy.get(key) for key in common_keys} != manifest_policy:
            raise CandidateSelectionError("release evaluation usage policy changed")
    isolation = _object(
        _object(manifest.get("audits"), field="release evaluation audits").get(
            "cross_corpus_isolation"
        ),
        field="release evaluation cross-corpus isolation",
    )
    comparisons = _object(isolation.get("comparisons"), field="isolation comparisons")
    if isolation.get("all_collision_counts_zero") is not True or any(
        any(value != 0 for value in _object(counts, field="collision counts").values())
        for counts in comparisons.values()
    ):
        raise CandidateSelectionError("release evaluation corpus is not isolated")
    return manifest, {
        "file_sha256": file_hash,
        "manifest_sha256": manifest_hash,
    }


def _candidate_recipe(manifest: Mapping[str, Any], *, protocol: Mapping[str, Any]) -> str:
    recipe = _object(manifest.get("recipe"), field="candidate recipe")
    recipe_hash = _sha256(manifest.get("recipe_sha256"), field="candidate recipe hash")
    if canonical_json_hash(recipe) != recipe_hash:
        raise CandidateSelectionError("candidate recipe self-hash is invalid")
    fixed = _object(protocol.get("fixed_recipe"), field="protocol fixed recipe")
    expected = {
        key: fixed[key]
        for key in (
            "epochs",
            "max_steps",
            "max_length",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "classifier_learning_rate",
            "weight_decay",
            "warmup_ratio",
            "class_weighting",
            "document_sampling",
            "gradient_checkpointing",
            "bf16",
            "attention_mode",
            "initialization_strategy",
            "fine_tuning",
            "checkpoint_selection_split",
            "checkpoint_selection_metric",
            "min_pii_free_ratio",
            "split_isolation_policy",
        )
    }
    expected["lora"] = {
        "rank": fixed["lora_rank"],
        "alpha": fixed["lora_alpha"],
        "dropout": fixed["lora_dropout"],
        "target_modules": fixed["lora_target_modules"],
    }
    if recipe != expected:
        raise CandidateSelectionError("candidate recipe differs from frozen v3 recipe")
    return recipe_hash


def _verify_dataset_binding(
    manifest: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> None:
    expected = _object(protocol.get("dataset"), field="protocol dataset")
    datasets = _object(manifest.get("datasets"), field="candidate datasets")
    audit = _object(dataset_manifest.get("audit"), field="training dataset audit")
    counts = _object(audit.get("label_counts"), field="training label counts")
    for split in ("train", "validation"):
        bound = _object(datasets.get(split), field=f"candidate {split} dataset")
        summary = _object(bound.get("summary"), field=f"candidate {split} summary")
        records = expected[f"{split}_records"]
        pii_free = expected[f"{split}_pii_free_records"]
        if (
            bound.get("sha256") != expected[f"{split}_sha256"]
            or summary.get("document_count") != records
            or summary.get("pii_free_document_count") != pii_free
            or summary.get("label_counts") != counts[split]
            or summary.get("entity_count") != sum(counts[split].values())
            or summary.get("unalignable_boundary_count") != 0
            or summary.get("split_counts") != {split: records}
            or summary.get("data_pool_counts") != {"public_release_pool": records}
            or summary.get("public_weight_training_allowed_document_count") != records
            or summary.get("quality_gate_passed_document_count") != records
            or summary.get("validators_passed_document_count") != records
        ):
            raise CandidateSelectionError("candidate dataset binding changed")
    isolation = _object(manifest.get("split_isolation"), field="split isolation receipt")
    expected_collisions = {
        **expected["required_cross_split_collisions"],
        "template_group": expected["template_group_overlap_count"],
    }
    if (
        isolation.get("policy") != expected.get("split_isolation_policy")
        or isolation.get("template_group_overlap_allowed") is not True
        or isolation.get("collision_counts") != expected_collisions
        or isolation.get("release_test_eligible") is not False
        or isolation.get("limitation")
        != "validation_informed_template_family_overlap"
    ):
        raise CandidateSelectionError("candidate split-isolation receipt changed")


def _verify_candidate(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    label2id: Mapping[str, int],
    taxonomy_version: str,
    label_schema_sha256: str,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise CandidateSelectionError("candidate root must be a regular directory")
    manifest, manifest_file_hash = _load_json(
        root / "training_manifest.json", field="candidate training manifest"
    )
    if not verify_training_manifest(manifest):
        raise CandidateSelectionError("candidate training manifest self-hash is invalid")
    manifest_hash = _sha256(
        manifest.get("manifest_sha256"), field="candidate manifest hash"
    )
    seed = manifest.get("seed")
    if (
        manifest.get("schema_version") != 4
        or manifest.get("manifest_type") != _EXPECTED_MANIFEST_TYPE
        or manifest.get("status") != "completed"
        or manifest.get("release_eligible") is not False
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise CandidateSelectionError("candidate identity/status is invalid")
    frozen_at = _parse_utc_timestamp(protocol.get("frozen_at"), field="protocol frozen_at")
    created_at = _parse_utc_timestamp(manifest.get("created_at"), field="candidate created_at")
    completed_at = _parse_utc_timestamp(
        manifest.get("completed_at"), field="candidate completed_at"
    )
    if created_at <= frozen_at or completed_at < created_at:
        raise CandidateSelectionError("candidate was not trained after protocol freeze")

    source = _object(protocol.get("source_model"), field="protocol source model")
    initialization = _object(manifest.get("initialization"), field="initialization audit")
    training_sources = _array(
        manifest.get("training_source_ids"), field="candidate training sources"
    )
    if (
        manifest.get("attention_mode") != "full"
        or manifest.get("fine_tuning") != "lora_merged"
        or manifest.get("base_source_id") != source.get("id")
        or manifest.get("source_revision") != source.get("revision")
        or training_sources
        != [source.get("id"), "repo_curated_synthetic_templates"]
        or initialization.get("source_model_id") != source.get("id")
        or initialization.get("source_revision") != source.get("revision")
        or initialization.get("initialization_seed") != seed
        or initialization.get("target_label2id") != label2id
        or initialization.get("mapped_head_rows_verified") is not True
    ):
        raise CandidateSelectionError("candidate source/initialization binding changed")
    conversion = _object(
        initialization.get("attention_conversion"), field="attention conversion audit"
    )
    if (
        conversion.get("source_attention_mode") != "causal"
        or conversion.get("target_attention_mode") != "full"
        or conversion.get("conversion")
        != "qwen3_to_qwen3_bi_shared_state_dict_v1"
        or conversion.get("strict_state_dict_load") is not True
        or conversion.get("newly_initialized_parameter_keys") != []
        or conversion.get("discarded_parameter_keys") != []
    ):
        raise CandidateSelectionError("candidate full-attention conversion audit changed")
    source_hashes = _object(
        initialization.get("source_hashes"), field="source artifact hashes"
    )
    if not source_hashes or any(
        _SHA256_RE.fullmatch(value) is None
        for value in source_hashes.values()
        if isinstance(value, str)
    ) or any(not isinstance(value, str) for value in source_hashes.values()):
        raise CandidateSelectionError("candidate source artifact hashes are invalid")
    source_artifact_sha256 = canonical_json_hash(source_hashes)
    if (
        manifest.get("taxonomy_version") != taxonomy_version
        or manifest.get("label2id") != label2id
        or manifest.get("label_schema_sha256") != label_schema_sha256
    ):
        raise CandidateSelectionError("candidate label schema changed")

    recipe_sha256 = _candidate_recipe(manifest, protocol=protocol)
    _verify_dataset_binding(
        manifest, protocol=protocol, dataset_manifest=dataset_manifest
    )
    benchmark = _object(manifest.get("benchmark_isolation"), field="benchmark isolation")
    if (
        benchmark.get("dataset_id") != _EXPECTED_PUBLIC_BENCHMARK
        or set(benchmark) != {"dataset_id", *_PUBLIC_FALSE_KEYS}
        or any(benchmark.get(key) is not False for key in _PUBLIC_FALSE_KEYS)
    ):
        raise CandidateSelectionError("candidate public benchmark isolation changed")

    recorded_output = _object(manifest.get("output_artifact"), field="output artifact")
    try:
        actual_output = output_artifact_fingerprint(root)
    except (OSError, ValueError) as exc:
        raise CandidateSelectionError("candidate model artifact cannot be verified") from exc
    if recorded_output != actual_output or actual_output.get("weight_files") != [
        "model.safetensors"
    ]:
        raise CandidateSelectionError("candidate output artifact binding changed")
    validation = _object(manifest.get("validation"), field="validation metrics")
    final_metrics, final_metrics_file_hash = _load_json(
        root / "final_eval_metrics.json", field="candidate final evaluation metrics"
    )
    if final_metrics != validation:
        raise CandidateSelectionError("candidate validation result file is not manifest-bound")
    checkpoint = common._verify_checkpoint_selection(
        root, manifest=manifest, protocol=protocol, validation=validation
    )
    ranking_metrics = {
        item["metric"]: (
            seed
            if item["metric"] == "seed"
            else _finite(validation.get(item["metric"]), field=item["metric"])
        )
        for item in protocol["cross_seed_selection"]["order"]
    }
    return {
        "candidate_id": f"seed-{seed}",
        "seed": seed,
        "training_manifest_sha256": manifest_hash,
        "training_manifest_file_sha256": manifest_file_hash,
        "recipe_sha256": recipe_sha256,
        "label_schema_sha256": label_schema_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "output_artifact_sha256": canonical_json_hash(actual_output),
        "weights_combined_sha256": actual_output["weights_combined_sha256"],
        "final_eval_metrics_file_sha256": final_metrics_file_hash,
        "checkpoint_selection": checkpoint,
        "ranking_metrics": ranking_metrics,
        "release_eligible": False,
    }


def _rank_key(
    candidate: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> tuple[float, ...]:
    metrics = _object(candidate.get("ranking_metrics"), field="ranking metrics")
    values: list[float] = []
    for item in protocol["cross_seed_selection"]["order"]:
        value = _finite(metrics.get(item["metric"]), field=item["metric"])
        values.append(value if item["direction"] == "maximize" else -value)
    return tuple(values)


def build_selection_receipt(
    *,
    protocol_path: Path,
    dataset_manifest_path: Path,
    release_evaluation_manifest_path: Path,
    candidate_roots: Sequence[Path],
    expected_protocol_file_sha256: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Verify all frozen inputs and return a path-free calibration selection."""

    protocol, protocol_hashes = _load_protocol(
        protocol_path, expected_file_sha256=expected_protocol_file_sha256
    )
    dataset_manifest, dataset_hashes = _verify_training_dataset_manifest(
        dataset_manifest_path, protocol=protocol
    )
    _, evaluation_hashes = _verify_release_evaluation_manifest(
        release_evaluation_manifest_path, protocol=protocol
    )
    expected_seeds = protocol["candidate_seeds"]
    if len(candidate_roots) != len(expected_seeds):
        raise CandidateSelectionError("exactly one candidate per frozen v3 seed is required")
    label2id, taxonomy_version, label_schema_sha256 = common._expected_label_schema()
    candidates = [
        _verify_candidate(
            root.expanduser(),
            protocol=protocol,
            dataset_manifest=dataset_manifest,
            label2id=label2id,
            taxonomy_version=taxonomy_version,
            label_schema_sha256=label_schema_sha256,
        )
        for root in candidate_roots
    ]
    seeds = [candidate["seed"] for candidate in candidates]
    if sorted(seeds) != expected_seeds or len(set(seeds)) != len(seeds):
        raise CandidateSelectionError("candidate seeds differ from frozen v3 seeds")
    if (
        len({candidate["recipe_sha256"] for candidate in candidates}) != 1
        or len({candidate["source_artifact_sha256"] for candidate in candidates}) != 1
        or {candidate["label_schema_sha256"] for candidate in candidates}
        != {label_schema_sha256}
    ):
        raise CandidateSelectionError(
            "candidate recipes/source artifacts/label schemas are not identical"
        )
    candidates.sort(key=lambda item: int(item["seed"]))
    selected = max(candidates, key=lambda item: _rank_key(item, protocol=protocol))
    selected_fields = (
        "candidate_id",
        "seed",
        "training_manifest_sha256",
        "source_artifact_sha256",
        "output_artifact_sha256",
        "weights_combined_sha256",
        "final_eval_metrics_file_sha256",
    )
    evaluation = protocol["post_selection_evaluation"]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "aiguard24_v3_cross_seed_calibration_selection_v1",
        "status": SELECTION_STATUS,
        "release_eligible": False,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "protocol": {"protocol_id": protocol["protocol_id"], **protocol_hashes},
        "development_dataset": {
            "dataset_id": protocol["dataset"]["id"],
            "dataset_version": protocol["dataset"]["version"],
            **dataset_hashes,
            "train_sha256": protocol["dataset"]["train_sha256"],
            "validation_sha256": protocol["dataset"]["validation_sha256"],
            "release_test_eligible": False,
        },
        "post_selection_evaluation": {
            "dataset_id": evaluation["dataset_id"],
            "dataset_version": evaluation["dataset_version"],
            **evaluation_hashes,
            "calibration_sha256": evaluation["calibration"]["sha256"],
            "internal_evaluation_sha256": evaluation["internal_evaluation"]["sha256"],
            "calibration_artifacts_read_by_selector": False,
            "internal_evaluation_artifacts_read_by_selector": False,
            "next_allowed_stage": "calibration",
        },
        "selection_policy": {
            "raw_quality_guardrails": [],
            "cross_seed_order": protocol["cross_seed_selection"]["order"],
            "selected_for_calibration_only": True,
            "selection_may_not_be_revisited": True,
        },
        "public_benchmark_isolation": {
            "dataset_id": _EXPECTED_PUBLIC_BENCHMARK,
            "artifacts_read_by_selector": False,
            "descriptive_run_allowed_only_after_calibration_freeze": True,
            "result_may_not_change_model_selection_or_calibration": True,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": {field: selected[field] for field in selected_fields},
    }
    common._assert_path_free(receipt)
    receipt["selection_receipt_sha256"] = canonical_json_hash(receipt)
    return receipt


def _write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise CandidateSelectionError("selection output overwrite is forbidden") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise CandidateSelectionError("selection output overwrite is forbidden")
    receipt = build_selection_receipt(
        protocol_path=args.protocol.expanduser(),
        dataset_manifest_path=args.dataset_manifest.expanduser(),
        release_evaluation_manifest_path=args.release_evaluation_manifest.expanduser(),
        candidate_roots=args.candidate,
    )
    _write_fresh_json(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "release_eligible": False,
                "output": output.name,
                "selected_candidate_id": receipt["selected"]["candidate_id"],
                "selection_receipt_sha256": receipt["selection_receipt_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CandidateSelectionError, OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"v3 candidate selection failed: {exc}") from None
