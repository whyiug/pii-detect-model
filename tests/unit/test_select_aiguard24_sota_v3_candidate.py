from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from scripts import select_aiguard24_sota_v3_candidate as selector

from pii_zh.training.manifest import (
    canonical_json_hash,
    output_artifact_fingerprint,
    sha256_file,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seal(value: dict[str, Any], field: str = "manifest_sha256") -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = canonical_json_hash(result)
    return result


def _core_label_counts() -> dict[str, int]:
    label2id, _, _ = selector.common._expected_label_schema()
    return {
        label.removeprefix("B-"): 1
        for label in label2id
        if label.startswith("B-")
    }


def _training_dataset_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    dataset = protocol["dataset"]
    counts = _core_label_counts()
    return _seal(
        {
            "manifest_schema_version": 1,
            "manifest_type": "development_training_dataset",
            "dataset_id": dataset["id"],
            "dataset_version": dataset["version"],
            "license": "Apache-2.0",
            "language": "zh-Hans-CN",
            "data_pool": "public_release_pool",
            "public_weight_training_allowed": True,
            "release_test_eligible": False,
            "roles": {
                "train": "weight_training",
                "validation": (
                    "development_checkpoint_selection_and_calibration_only"
                ),
                "release_test_eligible": False,
                "public_benchmark_read": False,
            },
            "benchmark_isolation": {
                "pii_bench_zh_read": False,
                "public_benchmark_prediction_read": False,
            },
            "resplit": {
                "strategy": "template-and-pii-state-stratified-hash-v1",
                "salt": "pii-zh-synthetic-sota-v2-resplit-20260718",
                "validation_to_train_ratio": 0.8,
                "allow_template_group_overlap": True,
                "require_document_id_disjoint": True,
                "require_entity_value_group_disjoint": True,
                "require_source_group_disjoint": True,
            },
            "counts": {
                "total": dataset["train_records"] + dataset["validation_records"],
                "by_split": {
                    "train": dataset["train_records"],
                    "validation": dataset["validation_records"],
                },
            },
            "audit": {
                "cross_split_collisions": {
                    "document_id": 0,
                    "entity_value_group": 0,
                    "source_group": 0,
                },
                "template_group_overlap_allowed": True,
                "template_group_overlap_count": 17,
                "pii_free": {
                    "train": dataset["train_pii_free_records"],
                    "validation": dataset["validation_pii_free_records"],
                },
                "label_counts": {"train": counts, "validation": counts},
            },
            "files": [
                {
                    "name": "train.jsonl",
                    "split": "train",
                    "records": dataset["train_records"],
                    "sha256": dataset["train_sha256"],
                    "mode": "0444",
                },
                {
                    "name": "validation.jsonl",
                    "split": "validation",
                    "records": dataset["validation_records"],
                    "sha256": dataset["validation_sha256"],
                    "mode": "0444",
                },
            ],
        }
    )


def _release_evaluation_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    evaluation = protocol["post_selection_evaluation"]
    calibration = evaluation["calibration"]
    internal = evaluation["internal_evaluation"]
    return _seal(
        {
            "manifest_schema_version": 1,
            "dataset_id": evaluation["dataset_id"],
            "dataset_version": evaluation["dataset_version"],
            "record_data_pool": "evaluation_only",
            "public_weight_training_allowed": False,
            "public_artifact_release_allowed": True,
            "generation": {
                "contains_real_personal_data": False,
                "external_dataset_inputs": [],
                "protected_evaluation_data_read": False,
                "seeds_independent": True,
            },
            "files": [
                {
                    "name": "calibration.jsonl",
                    "split": "calibration",
                    "records": calibration["records"],
                    "sha256": calibration["sha256"],
                    "mode": "0444",
                },
                {
                    "name": "internal_evaluation.jsonl",
                    "split": "internal_evaluation",
                    "records": internal["records"],
                    "sha256": internal["sha256"],
                    "mode": "0444",
                },
            ],
            "usage_policy": {
                "calibration": {
                    "checkpoint_or_model_selection_allowed": False,
                    "final_metric_claim_allowed": False,
                    "fusion_calibration_allowed": True,
                    "gradient_training_allowed": False,
                    "temperature_scaling_allowed": True,
                    "threshold_tuning_allowed": True,
                },
                "internal_evaluation": {
                    "checkpoint_or_model_selection_allowed": False,
                    "final_metric_claim_allowed": True,
                    "fusion_calibration_allowed": False,
                    "gradient_training_allowed": False,
                    "result_informed_model_or_threshold_change_allowed": False,
                    "temperature_scaling_allowed": False,
                    "threshold_tuning_allowed": False,
                },
            },
            "audits": {
                "cross_corpus_isolation": {
                    "all_collision_counts_zero": True,
                    "comparisons": {
                        "calibration_vs_internal_evaluation": {
                            "doc_id": 0,
                            "exact_entity_value_sha256": 0,
                            "exact_text_sha256": 0,
                        }
                    },
                }
            },
        }
    )


def _recipe(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_recipe"]
    fields = (
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
    result = {field: fixed[field] for field in fields}
    result["lora"] = {
        "rank": fixed["lora_rank"],
        "alpha": fixed["lora_alpha"],
        "dropout": fixed["lora_dropout"],
        "target_modules": fixed["lora_target_modules"],
    }
    return result


def _summary(
    *, split: str, records: int, pii_free: int, label_counts: dict[str, int]
) -> dict[str, Any]:
    return {
        "document_count": records,
        "entity_count": sum(label_counts.values()),
        "token_count": records * 8,
        "pii_free_document_count": pii_free,
        "unalignable_boundary_count": 0,
        "label_counts": label_counts,
        "split_counts": {split: records},
        "data_pool_counts": {"public_release_pool": records},
        "quality_tier_counts": {"N": pii_free, "S0": records - pii_free},
        "quality_gate_passed_document_count": records,
        "validators_passed_document_count": records,
        "public_weight_training_allowed_document_count": records,
        "sources": [],
    }


def _candidate(
    tmp_path: Path,
    *,
    protocol: dict[str, Any],
    dataset_manifest: dict[str, Any],
    seed: int,
    scores: tuple[float, float, float, float],
    created_at: str = "2026-07-18T01:00:00+00:00",
) -> Path:
    root = tmp_path / f"candidate-seed-{seed}"
    root.mkdir()
    (root / "model.safetensors").write_bytes(f"weight-{seed}".encode())
    _write_json(root / "config.json", {"model_type": "qwen3_bi"})
    _write_json(root / "tokenizer.json", {"version": "1.0", "model": {}})
    _write_json(root / "tokenizer_config.json", {"tokenizer_class": "fast"})

    risk, micro, macro, pii_free_precision = scores
    validation = {
        "eval_risk_weighted_score": risk,
        "eval_strict_micro_f1": micro,
        "eval_strict_macro_f1": macro,
        "eval_pii_free_precision": pii_free_precision,
    }
    _write_json(root / "final_eval_metrics.json", validation)
    trainer_root = root.parent / f"{root.name}.trainer-state"
    checkpoint = trainer_root / "checkpoint-30"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(f"adapter-{seed}".encode())
    _write_json(
        trainer_root / "trainer_state.json",
        {
            "best_global_step": 30,
            "best_metric": risk,
            "best_model_checkpoint": str(checkpoint.resolve()),
            "log_history": [{"step": 30, "eval_risk_weighted_score": risk}],
        },
    )

    label2id, taxonomy_version, label_schema_sha256 = (
        selector.common._expected_label_schema()
    )
    recipe = _recipe(protocol)
    source = protocol["source_model"]
    dataset = protocol["dataset"]
    label_counts = dataset_manifest["audit"]["label_counts"]
    manifest = {
        "schema_version": 4,
        "manifest_type": "aiguard24_community_training_v1",
        "status": "completed",
        "release_eligible": False,
        "created_at": created_at,
        "completed_at": "2026-07-18T01:30:00+00:00",
        "seed": seed,
        "attention_mode": "full",
        "fine_tuning": "lora_merged",
        "base_source_id": source["id"],
        "source_revision": source["revision"],
        "training_source_ids": [source["id"], "repo_curated_synthetic_templates"],
        "recipe": recipe,
        "recipe_sha256": canonical_json_hash(recipe),
        "taxonomy_version": taxonomy_version,
        "label2id": label2id,
        "label_schema_sha256": label_schema_sha256,
        "initialization": {
            "source_model_id": source["id"],
            "source_revision": source["revision"],
            "initialization_seed": seed,
            "target_label2id": label2id,
            "mapped_head_rows_verified": True,
            "source_hashes": {
                "config.json": "5" * 64,
                "model.safetensors": "6" * 64,
                "tokenizer.json": "7" * 64,
                "tokenizer_config.json": "8" * 64,
            },
            "attention_conversion": {
                "source_attention_mode": "causal",
                "target_attention_mode": "full",
                "conversion": "qwen3_to_qwen3_bi_shared_state_dict_v1",
                "strict_state_dict_load": True,
                "newly_initialized_parameter_keys": [],
                "discarded_parameter_keys": [],
            },
        },
        "datasets": {
            "train": {
                "sha256": dataset["train_sha256"],
                "summary": _summary(
                    split="train",
                    records=dataset["train_records"],
                    pii_free=dataset["train_pii_free_records"],
                    label_counts=label_counts["train"],
                ),
            },
            "validation": {
                "sha256": dataset["validation_sha256"],
                "summary": _summary(
                    split="validation",
                    records=dataset["validation_records"],
                    pii_free=dataset["validation_pii_free_records"],
                    label_counts=label_counts["validation"],
                ),
            },
        },
        "split_isolation": {
            "policy": "template-overlap-development-v1",
            "template_group_overlap_allowed": True,
            "collision_counts": {
                "document_id": 0,
                "template_group": 17,
                "entity_value_group": 0,
                "source_group": 0,
            },
            "release_test_eligible": False,
            "limitation": "validation_informed_template_family_overlap",
        },
        "benchmark_isolation": {
            "dataset_id": "wan9yu/pii-bench-zh",
            "read_for_training": False,
            "read_for_validation": False,
            "read_for_checkpoint_selection": False,
            "read_for_threshold_tuning": False,
        },
        "checkpoint_selection": {
            "selection_split": "validation",
            "selection_metric": "risk_weighted_score",
            "selected_checkpoint_id": "checkpoint-30",
            "selected_global_step": 30,
            "best_metric": risk,
            "final_validation_replay_metric": risk,
            "adapter_model_sha256": sha256_file(adapter),
        },
        "validation": validation,
        "output_artifact": output_artifact_fingerprint(root),
    }
    _write_json(root / "training_manifest.json", _seal(manifest))
    return root


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[Path], str]:
    source = (
        selector._REPOSITORY_ROOT
        / "configs/train/aiguard24_sota_v3_resplit_selection.json"
    )
    protocol = json.loads(source.read_text(encoding="utf-8"))
    protocol["dataset"].update(
        {
            "train_sha256": "1" * 64,
            "validation_sha256": "2" * 64,
            "train_records": 80,
            "validation_records": 20,
            "train_pii_free_records": 36,
            "validation_pii_free_records": 9,
        }
    )
    training_manifest = _training_dataset_manifest(protocol)
    training_path = tmp_path / "training_dataset_manifest.json"
    _write_json(training_path, training_manifest)
    protocol["dataset"].update(
        {
            "manifest_sha256": training_manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(training_path),
        }
    )

    protocol["post_selection_evaluation"]["calibration"].update(
        {"sha256": "3" * 64}
    )
    protocol["post_selection_evaluation"]["internal_evaluation"].update(
        {"sha256": "4" * 64}
    )
    release_manifest = _release_evaluation_manifest(protocol)
    release_path = tmp_path / "release_evaluation_manifest.json"
    _write_json(release_path, release_manifest)
    protocol["post_selection_evaluation"].update(
        {
            "manifest_sha256": release_manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(release_path),
        }
    )
    protocol_path = tmp_path / "selection_protocol.json"
    _write_json(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)
    candidates = [
        _candidate(
            tmp_path,
            protocol=protocol,
            dataset_manifest=training_manifest,
            seed=seed,
            scores=scores,
        )
        for seed, scores in (
            (13, (0.80, 0.90, 0.70, 0.99)),
            (42, (0.90, 0.80, 0.75, 0.50)),
            (97, (0.90, 0.80, 0.75, 0.80)),
        )
    ]
    return protocol_path, training_path, release_path, candidates, protocol_hash


def _build(
    protocol: Path,
    dataset: Path,
    release_evaluation: Path,
    candidates: list[Path],
    protocol_hash: str,
) -> dict[str, Any]:
    return selector.build_selection_receipt(
        protocol_path=protocol,
        dataset_manifest_path=dataset,
        release_evaluation_manifest_path=release_evaluation,
        candidate_roots=candidates,
        expected_protocol_file_sha256=protocol_hash,
        created_at="2026-07-18T02:00:00+00:00",
    )


def test_frozen_protocol_and_real_manifests_are_byte_pinned() -> None:
    frozen_data_root_raw = os.environ.get("PII_ZH_FROZEN_DATA_ROOT")
    if not frozen_data_root_raw:
        pytest.skip("set PII_ZH_FROZEN_DATA_ROOT to verify local frozen manifests")
    frozen_data_root = Path(frozen_data_root_raw).resolve(strict=True)
    protocol_path = (
        selector._REPOSITORY_ROOT
        / "configs/train/aiguard24_sota_v3_resplit_selection.json"
    )
    protocol, protocol_hashes = selector._load_protocol(protocol_path)
    _, dataset_hashes = selector._verify_training_dataset_manifest(
        frozen_data_root / "synthetic_sota_v2_resplit/dataset_manifest.json",
        protocol=protocol,
    )
    _, release_hashes = selector._verify_release_evaluation_manifest(
        frozen_data_root / "synthetic_sota_release_eval_v1/dataset_manifest.json",
        protocol=protocol,
    )

    assert protocol_hashes["file_sha256"] == selector.FROZEN_PROTOCOL_FILE_SHA256
    assert dataset_hashes["manifest_sha256"] == protocol["dataset"]["manifest_sha256"]
    assert (
        release_hashes["manifest_sha256"]
        == protocol["post_selection_evaluation"]["manifest_sha256"]
    )


def test_selects_for_calibration_only_with_pii_free_as_fourth_tie_breaker(
    tmp_path: Path,
) -> None:
    protocol, dataset, release, candidates, protocol_hash = _fixture(tmp_path)
    receipt = _build(protocol, dataset, release, candidates, protocol_hash)

    assert receipt["status"] == selector.SELECTION_STATUS
    assert receipt["release_eligible"] is False
    assert receipt["selected"]["candidate_id"] == "seed-97"
    assert receipt["post_selection_evaluation"]["next_allowed_stage"] == "calibration"
    assert receipt["post_selection_evaluation"]["calibration_artifacts_read_by_selector"] is False
    assert (
        receipt["post_selection_evaluation"][
            "internal_evaluation_artifacts_read_by_selector"
        ]
        is False
    )
    assert receipt["public_benchmark_isolation"]["artifacts_read_by_selector"] is False
    assert str(tmp_path) not in json.dumps(receipt, ensure_ascii=True)
    assert receipt["selection_receipt_sha256"] == canonical_json_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "selection_receipt_sha256"
        }
    )


def test_candidate_must_be_created_after_protocol_freeze(tmp_path: Path) -> None:
    protocol, dataset, release, candidates, protocol_hash = _fixture(tmp_path)
    manifest_path = candidates[0] / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = "2026-07-18T00:30:00+00:00"
    _write_json(manifest_path, _seal(manifest))

    with pytest.raises(selector.CandidateSelectionError, match="after protocol freeze"):
        _build(protocol, dataset, release, candidates, protocol_hash)


def test_candidate_public_benchmark_evidence_fails_closed(tmp_path: Path) -> None:
    protocol, dataset, release, candidates, protocol_hash = _fixture(tmp_path)
    manifest_path = candidates[1] / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_isolation"]["read_for_validation"] = True
    _write_json(manifest_path, _seal(manifest))

    with pytest.raises(selector.CandidateSelectionError, match="benchmark isolation"):
        _build(protocol, dataset, release, candidates, protocol_hash)


def test_frozen_protocol_rejects_post_result_edits(tmp_path: Path) -> None:
    source = (
        selector._REPOSITORY_ROOT
        / "configs/train/aiguard24_sota_v3_resplit_selection.json"
    )
    protocol = json.loads(source.read_text(encoding="utf-8"))
    protocol["fixed_recipe"]["epochs"] = 3.0
    path = tmp_path / "edited_protocol.json"
    _write_json(path, protocol)

    with pytest.raises(selector.CandidateSelectionError, match="file hash is not frozen"):
        selector._load_protocol(path)


def test_output_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    selector._write_fresh_json(output, {"safe": True})
    with pytest.raises(selector.CandidateSelectionError, match="overwrite is forbidden"):
        selector._write_fresh_json(output, {"safe": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {"safe": True}
