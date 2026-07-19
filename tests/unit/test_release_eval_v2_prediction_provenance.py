from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pii_zh.cascade import expected_core24_label2id
from pii_zh.evaluation import canonical_json_hash
from pii_zh.evaluation import release_eval_v2_prediction_provenance as provenance
from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
    AmendmentReplayInputs,
    CalibrationInputs,
    GenerationRuntime,
    ModelRawInputs,
    PredictionProvenanceInputs,
    ReleaseEvalV2PredictionProvenanceError,
    build_release_eval_v2_generation_receipt,
    build_release_eval_v2_prediction_manifest,
    replay_release_eval_v2_prediction_manifest,
    validate_release_eval_v2_prediction_manifest,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object, *, ensure_ascii: bool = True) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if readonly:
        path.chmod(0o444)


def _seal(value: dict[str, Any], field: str, *, ensure_ascii: bool = True) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = _canonical(result, ensure_ascii=ensure_ascii)
    return result


def _community_artifact(root: Path) -> Path:
    root.mkdir()
    labels = expected_core24_label2id()
    config = {
        "model_type": "qwen3_bi",
        "pii_attention_mode": "full",
        "architectures": ["Qwen3BiForTokenClassification"],
        "architecture_version": "qwen3_bi_token_cls_v1",
        "bi_attention_backend": "sdpa",
        "use_cache": False,
        "pii_training_status": "completed_candidate_not_benchmark_evaluated",
        "pii_taxonomy_version": "1.0.0",
        "pii_release_eligible": False,
        "num_labels": 49,
        "label2id": labels,
        "id2label": {str(index): label for label, index in labels.items()},
    }
    _write_json(root / "config.json", config)
    _write_json(root / "tokenizer.json", {})
    _write_json(root / "tokenizer_config.json", {})
    (root / "model.safetensors").write_bytes(b"non-executable-safe-test-weight")
    files = {
        name: _hash(root / name)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    weight_hashes = {"model.safetensors": files["model.safetensors"]}
    output = {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": ["model.safetensors"],
        "weights_combined_sha256": canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": canonical_json_hash(files),
    }
    manifest = {
        "schema_version": 4,
        "manifest_type": "aiguard24_community_training_v1",
        "status": "completed",
        "release_eligible": False,
        "seed": 97,
        "attention_mode": "full",
        "fine_tuning": "lora_merged",
        "base_source_id": "ZJUICSR/AIguard-pii-detection-fast",
        "source_revision": "677a5ebc1600fef61e8973cafd3026be322b3a73",
        "taxonomy_version": "1.0.0",
        "label2id": labels,
        "label_schema_sha256": canonical_json_hash(labels),
        "initialization": {
            "strategy": "aiguard_bie_to_pii_zh_core24_bio_v1",
            "target_label_count": 49,
            "target_label2id": labels,
            "release_eligible": False,
            "attention_conversion": {
                "source_attention_mode": "causal",
                "target_attention_mode": "full",
                "conversion": "qwen3_to_qwen3_bi_shared_state_dict_v1",
                "attention_backend": "sdpa",
                "strict_state_dict_load": True,
                "newly_initialized_parameter_keys": [],
                "discarded_parameter_keys": [],
            },
        },
        "checkpoint_selection": {
            "selection_split": "validation",
            "selection_metric": "risk_weighted_score",
            "selected_checkpoint_id": "checkpoint-10",
            "selected_global_step": 10,
            "best_metric": 0.5,
            "final_validation_replay_metric": 0.5,
            "adapter_model_sha256": "a" * 64,
        },
        "output_artifact": output,
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    _write_json(root / "training_manifest.json", manifest)
    return root


def _prediction(path: Path, doc_id: str, *, start: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "spans": [{"start": start, "end": start + 1, "label": "PERSON_NAME", "score": 0.8}],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _dataset_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    calibration = tmp_path / "calibration.jsonl"
    internal = tmp_path / "internal_evaluation.jsonl"
    calibration.write_text('{"fixture":"calibration"}\n', encoding="utf-8")
    internal.write_text('{"fixture":"internal"}\n', encoding="utf-8")
    split_entries = []
    for split, path in (("calibration", calibration), ("internal_evaluation", internal)):
        split_entries.append(
            {
                "name": f"{split}.jsonl",
                "split": split,
                "records": 1,
                "bytes": path.stat().st_size,
                "sha256": _hash(path),
                "mode": "0444",
            }
        )
    chronology = {
        "frozen_before_any_v3_seed42_validation_result": True,
        "v1_confirmatory_calibration_withdrawn": True,
        "v1_structural_pre_read_one_record_no_model_result": True,
        "v2_zero_content_read_before_cross_seed_selection": True,
    }
    freeze = {
        "receipt_schema_version": 1,
        "receipt_id": "synthetic_sota_release_eval_v2_supersession_freeze",
        "chronology": chronology,
        "supersession": {
            "withdrawn_dataset_id": "pii_zh_synthetic_sota_release_eval_v1",
            "withdrawn_dataset_version": "1.0.0",
            "replacement_dataset_id": "pii_zh_synthetic_sota_release_eval_v2",
            "replacement_dataset_version": "2.0.0",
        },
        "protocol_effect": {
            "does_not_modify_frozen_v3_training_protocol_or_selector": True,
            "v1_may_not_be_used_for_zero-read_confirmatory_calibration": True,
        },
    }
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, freeze)
    freeze_hash = _hash(freeze_path)
    freeze_binding = {
        "file_sha256": freeze_hash,
        "receipt_id": freeze["receipt_id"],
        "chronology": chronology,
    }
    manifest = _seal(
        {
            "manifest_schema_version": 2,
            "dataset_id": "pii_zh_synthetic_sota_release_eval_v2",
            "dataset_version": "2.0.0",
            "record_data_pool": "evaluation_only",
            "public_weight_training_allowed": False,
            "public_artifact_release_allowed": True,
            "generation": {
                "generation_salt_sha256": "6" * 64,
                "contains_real_personal_data": False,
                "protected_evaluation_data_read": False,
                "jsonl_rows_printed": False,
            },
            "audits": {
                "core_label_count": 24,
                "cross_corpus_isolation": {"all_collision_counts_zero": True},
            },
            "freeze": {"supersession_freeze_receipt": freeze_binding},
            "usage_policy": {
                "calibration": {
                    "gradient_training_allowed": False,
                    "checkpoint_or_model_selection_allowed": False,
                    "final_metric_claim_allowed": False,
                    "threshold_tuning_allowed": True,
                    "content_read_before_cross_seed_selection_allowed": False,
                },
                "internal_evaluation": {
                    "gradient_training_allowed": False,
                    "checkpoint_or_model_selection_allowed": False,
                    "final_metric_claim_allowed": True,
                    "threshold_tuning_allowed": False,
                    "temperature_scaling_allowed": False,
                    "fusion_calibration_allowed": False,
                    "result_informed_model_or_threshold_change_allowed": False,
                    "content_read_before_final_freeze_allowed": False,
                },
            },
            "files": split_entries,
        },
        "manifest_sha256",
        ensure_ascii=False,
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_file_hash = _hash(manifest_path)
    materialization = _seal(
        {
            "freeze_receipt": freeze_binding,
            "replacement": {
                "dataset_manifest_file_sha256": manifest_file_hash,
                "dataset_manifest_sha256": manifest["manifest_sha256"],
                "replacement_dataset_id": manifest["dataset_id"],
                "replacement_dataset_version": manifest["dataset_version"],
                "generation_salt_sha256": "6" * 64,
                "files": split_entries,
            },
        },
        "receipt_sha256",
        ensure_ascii=False,
    )
    materialization_path = tmp_path / "materialization.json"
    _write_json(materialization_path, materialization)
    frozen = {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_file_sha256": manifest_file_hash,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "materialization_receipt_file_sha256": _hash(materialization_path),
        "materialization_receipt_sha256": materialization["receipt_sha256"],
        "freeze_receipt_file_sha256": freeze_hash,
        "generation_salt_sha256": "6" * 64,
        "splits": {
            item["split"]: {
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "records": item["records"],
            }
            for item in split_entries
        },
    }
    monkeypatch.setattr(provenance, "FROZEN_V2", frozen)
    return {
        "calibration": calibration,
        "internal_evaluation": internal,
        "dataset_manifest": manifest_path,
        "materialization": materialization_path,
        "freeze": freeze_path,
    }


def _selection(path: Path, model: dict[str, Any]) -> Path:
    selected = {
        "candidate_id": "seed-97",
        "seed": model["seed"],
        "training_manifest_sha256": model["training_manifest_sha256"],
        "source_artifact_sha256": "b" * 64,
        "output_artifact_sha256": model["output_artifact_sha256"],
        "weights_combined_sha256": model["weights_combined_sha256"],
        "final_eval_metrics_file_sha256": "c" * 64,
    }
    candidate = {
        **selected,
        "training_manifest_file_sha256": model["training_manifest_file_sha256"],
        "label_schema_sha256": model["label_schema_sha256"],
    }
    receipt = _seal(
        {
            "schema_version": 1,
            "receipt_type": "aiguard24_v3_cross_seed_calibration_selection_v1",
            "status": "SELECTED_FOR_CALIBRATION_NOT_RELEASE",
            "release_eligible": False,
            "protocol": {
                "protocol_id": "aiguard24_synthetic_sota_v3_resplit_dev_selection",
                "file_sha256": "9f330eca4dbc6e522fc52b06f6bfb7c4047e455a0b3be555a94a03e6dd4ca51e",
            },
            "post_selection_evaluation": {
                "dataset_id": "pii_zh_synthetic_sota_release_eval_v1",
                "dataset_version": "1.0.0",
                "file_sha256": "aa4817080097681688948104e867ebb510be463fb87d23f9f9e3f786f58dfa7b",
                "manifest_sha256": (
                    "f0500a5c0672b5285c8d95782d9491b6b5dd3df6a14977e6ade3f53bbd7130ca"
                ),
            },
            "candidate_count": 3,
            "candidates": [
                {**candidate, "seed": 13, "training_manifest_sha256": "1" * 64},
                {**candidate, "seed": 42, "training_manifest_sha256": "2" * 64},
                candidate,
            ],
            "selected": selected,
        },
        "selection_receipt_sha256",
    )
    _write_json(path, receipt)
    return path


def _runtime(target_split: str) -> GenerationRuntime:
    return GenerationRuntime(
        document_batch_size=1,
        micro_batch_size=1,
        max_tokens=512,
        stride_fraction=0.25,
        device_class="cpu",
        dtype="float32",
        visible_cuda_device_count=0,
        scope="open24",
        target_split=target_split,  # type: ignore[arg-type]
    )


def _stage_gate(path: Path, target_split: str) -> dict[str, str]:
    if target_split == "calibration":
        return {
            "status": "CALIBRATION_PREOPEN_PASS",
            "authorization_file_sha256": "7" * 64,
            "authorization_sha256": "8" * 64,
            "expected_input_sha256": _hash(path),
        }
    return {
        "status": "INTERNAL_PREOPEN_PASS",
        "unlock_file_sha256": "9" * 64,
        "unlock_sha256": "a" * 64,
        "expected_input_sha256": _hash(path),
    }


def _write_receipt(path: Path, value: dict[str, Any]) -> Path:
    _write_json(path, value)
    return path


def _base_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data = _dataset_chain(tmp_path, monkeypatch)
    model_root = _community_artifact(tmp_path / "model")
    model = provenance._model_identity(model_root)
    selection = _selection(tmp_path / "selection.json", model)
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}", encoding="utf-8")

    def replay_fixture_amendment(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        del args
        selection_value = json.loads(kwargs["selection_receipt_path"].read_text())
        selected = selection_value["selected"]
        amendment = _seal(
            {
                "status": "V2_CALIBRATION_NEXT_NOT_RELEASE",
                "selection_replay": {
                    "receipt_file_sha256": _hash(kwargs["selection_receipt_path"]),
                    "receipt_sha256": selection_value["selection_receipt_sha256"],
                    "selected": selected,
                },
                "replacement_v2": {
                    "dataset_id": provenance.FROZEN_V2["dataset_id"],
                    "dataset_version": provenance.FROZEN_V2["dataset_version"],
                    "manifest_file_sha256": provenance.FROZEN_V2[
                        "dataset_manifest_file_sha256"
                    ],
                    "manifest_sha256": provenance.FROZEN_V2["dataset_manifest_sha256"],
                    "calibration_sha256": provenance.FROZEN_V2["splits"]["calibration"][
                        "sha256"
                    ],
                },
            },
            "amendment_sha256",
            ensure_ascii=False,
        )
        return amendment, "b" * 64

    from scripts import build_aiguard24_v3_release_eval_v2_amendment as amendment_builder

    monkeypatch.setattr(
        amendment_builder,
        "replay_amendment_from_selection_evidence",
        replay_fixture_amendment,
    )
    amendment = AmendmentReplayInputs(
        amendment=amendment_path,
        protocol=tmp_path / "protocol.json",
        development_manifest=tmp_path / "development.json",
        v1_release_manifest=tmp_path / "v1.json",
        candidate_roots=(tmp_path / "seed13", tmp_path / "seed42", tmp_path / "seed97"),
    )
    fit_predictions = tmp_path / "fit_predictions.jsonl"
    _prediction(fit_predictions, "calibration-doc")
    fit_receipt = _write_receipt(
        tmp_path / "fit_generation.json",
        build_release_eval_v2_generation_receipt(
            track="model_raw",
            input_path=data["calibration"],
            predictions_path=fit_predictions,
            model_artifact=model_root,
            calibration_bundle=None,
            runtime=_runtime("calibration"),
            stage_gate=_stage_gate(data["calibration"], "calibration"),
        ),
    )
    fit_inputs = PredictionProvenanceInputs(
        track="model_raw",
        target_split="calibration",
        predictions=fit_predictions,
        generation_receipt=fit_receipt,
        gold=data["calibration"],
        dataset_manifest=data["dataset_manifest"],
        materialization_receipt=data["materialization"],
        freeze_receipt=data["freeze"],
        model_artifact=model_root,
        selection_receipt=selection,
        amendment=amendment,
    )
    fit_manifest_value = build_release_eval_v2_prediction_manifest(fit_inputs)
    fit_manifest = tmp_path / "fit_prediction_manifest.json"
    _write_json(fit_manifest, fit_manifest_value)
    bundle = tmp_path / "calibration_bundle.json"
    bundle_value = {
        "model_version": model["training_manifest_sha256"],
        "calibration_version": "cal-fixture-v1",
        "global_temperature": 1.0,
        "entity_temperatures": {},
        "entity_thresholds": {"PERSON_NAME": 0.5},
        "default_threshold": 0.5,
    }
    _write_json(bundle, bundle_value)
    diagnostics = _seal(
        {
            "schema_version": 1,
            "manifest_type": "calibration_diagnostics",
            "calibration_version": "cal-fixture-v1",
            "calibration_bundle_sha256": _hash(bundle),
            "taxonomy_version": "1.0.0",
            "inputs": {
                "gold_sha256": _hash(data["calibration"]),
                "predictions_sha256": _hash(fit_predictions),
                "document_count": 1,
            },
            "dataset_contract": {
                "dataset_id": "pii_zh_synthetic_sota_release_eval_v2",
                "dataset_version": "2.0.0",
                "gold_sha256": _hash(data["calibration"]),
                "manifest_sha256": provenance.FROZEN_V2["dataset_manifest_sha256"],
                "manifest_file_sha256": provenance.FROZEN_V2["dataset_manifest_file_sha256"],
                "supersession_freeze_file_sha256": provenance.FROZEN_V2[
                    "freeze_receipt_file_sha256"
                ],
                "records": 1,
                "role": "post_cross_seed_selection_calibration_only",
            },
            "parameters": {},
            "temperature": {},
            "confidence_calibration": {},
            "per_label": {},
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
            },
        },
        "manifest_sha256",
    )
    diagnostics_path = tmp_path / "calibration_diagnostics.json"
    _write_json(diagnostics_path, diagnostics)
    calibration = CalibrationInputs(
        bundle=bundle,
        diagnostics=diagnostics_path,
        fit_gold=data["calibration"],
        fit_predictions=fit_predictions,
        fit_generation_receipt=fit_receipt,
        fit_prediction_manifest=fit_manifest,
    )
    return {
        **data,
        "model_root": model_root,
        "model": model,
        "selection": selection,
        "fit_inputs": fit_inputs,
        "fit_manifest": fit_manifest,
        "calibration": calibration,
        "bundle": bundle,
    }


def _internal_raw(
    fixture: dict[str, Any], tmp_path: Path
) -> tuple[PredictionProvenanceInputs, Path]:
    predictions = tmp_path / "internal_raw.jsonl"
    _prediction(predictions, "internal-doc")
    generation = _write_receipt(
        tmp_path / "internal_raw_generation.json",
        build_release_eval_v2_generation_receipt(
            track="model_raw",
            input_path=fixture["internal_evaluation"],
            predictions_path=predictions,
            model_artifact=fixture["model_root"],
            calibration_bundle=None,
            runtime=_runtime("internal_evaluation"),
            stage_gate=_stage_gate(fixture["internal_evaluation"], "internal_evaluation"),
        ),
    )
    inputs = replace(
        fixture["fit_inputs"],
        target_split="internal_evaluation",
        predictions=predictions,
        generation_receipt=generation,
        gold=fixture["internal_evaluation"],
        calibration=fixture["calibration"],
    )
    manifest = tmp_path / "internal_raw_manifest.json"
    _write_json(manifest, build_release_eval_v2_prediction_manifest(inputs))
    return inputs, manifest


def test_build_replay_and_metadata_validation_are_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_fixture(tmp_path, monkeypatch)
    inputs, manifest = _internal_raw(fixture, tmp_path)

    replay = replay_release_eval_v2_prediction_manifest(manifest, inputs)
    metadata = validate_release_eval_v2_prediction_manifest(manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    rendered = json.dumps(value, ensure_ascii=True)

    assert replay["status"] == "PASS"
    assert metadata["status"] == "PASS_METADATA_ONLY"
    assert metadata["target_split"] == "internal_evaluation"
    assert metadata["output_artifact_sha256"] == fixture["model"]["output_artifact_sha256"]
    assert str(tmp_path) not in rendered
    assert "internal-doc" not in rendered
    assert value["prediction_semantics"] == "raw_model_scores_pre_service"
    assert value["calibration"] is not None
    assert value["generation"]["mode"] == "model_raw"


def test_full_system_binds_same_model_calibration_and_actual_community_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_fixture(tmp_path, monkeypatch)
    raw_inputs, raw_manifest = _internal_raw(fixture, tmp_path)
    predictions = tmp_path / "full_system.jsonl"
    _prediction(predictions, "internal-doc")
    generation = _write_receipt(
        tmp_path / "full_generation.json",
        build_release_eval_v2_generation_receipt(
            track="full_system",
            input_path=fixture["internal_evaluation"],
            predictions_path=predictions,
            model_artifact=fixture["model_root"],
            calibration_bundle=fixture["bundle"],
            runtime=_runtime("internal_evaluation"),
            stage_gate=_stage_gate(fixture["internal_evaluation"], "internal_evaluation"),
        ),
    )
    inputs = replace(
        raw_inputs,
        track="full_system",
        predictions=predictions,
        generation_receipt=generation,
        model_raw=ModelRawInputs(
            predictions=raw_inputs.predictions,
            generation_receipt=raw_inputs.generation_receipt,
            prediction_manifest=raw_manifest,
        ),
    )

    value = build_release_eval_v2_prediction_manifest(inputs)

    assert value["service"]["profile_id"] == "community-model-cascade-v1"
    assert value["service"]["mode"] == "cascade"
    assert value["service"]["route_count"] == 24
    assert value["service"]["validators"]["validator_count"] >= 20
    assert value["service"]["rules"]["ruleset_id"] == "cn_common_v6"
    assert (
        value["service"]["calibration_bundle_file_sha256"]
        == value["calibration"]["bundle_file_sha256"]
    )
    assert (
        value["upstream_model_raw"]["manifest_sha256"]
        == json.loads(raw_manifest.read_text(encoding="utf-8"))["manifest_sha256"]
    )


def test_model_calibrated_is_a_distinct_no_rules_community_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_fixture(tmp_path, monkeypatch)
    raw_inputs, raw_manifest = _internal_raw(fixture, tmp_path)
    predictions = tmp_path / "model_calibrated.jsonl"
    _prediction(predictions, "internal-doc")
    generation = _write_receipt(
        tmp_path / "model_calibrated_generation.json",
        build_release_eval_v2_generation_receipt(
            track="model_calibrated",
            input_path=fixture["internal_evaluation"],
            predictions_path=predictions,
            model_artifact=fixture["model_root"],
            calibration_bundle=fixture["bundle"],
            runtime=_runtime("internal_evaluation"),
            stage_gate=_stage_gate(
                fixture["internal_evaluation"], "internal_evaluation"
            ),
        ),
    )
    inputs = replace(
        raw_inputs,
        track="model_calibrated",
        predictions=predictions,
        generation_receipt=generation,
        model_raw=ModelRawInputs(
            predictions=raw_inputs.predictions,
            generation_receipt=raw_inputs.generation_receipt,
            prediction_manifest=raw_manifest,
        ),
    )

    value = build_release_eval_v2_prediction_manifest(inputs)

    assert value["prediction_semantics"] == (
        "community_model_only_calibrated_output_no_rules"
    )
    assert value["generation"]["mode"] == "model-only"
    assert value["service"]["mode"] == "model-only"
    assert value["service"]["rules_enabled"] is False
    assert value["calibration"]["bundle_file_sha256"] == _hash(fixture["bundle"])


def test_resigned_unselected_seed_and_prediction_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_fixture(tmp_path, monkeypatch)
    inputs, manifest = _internal_raw(fixture, tmp_path)
    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    selection["selected"]["seed"] = 13
    selection = _seal(selection, "selection_receipt_sha256")
    fixture["selection"].write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(
        ReleaseEvalV2PredictionProvenanceError,
        match="not the exact frozen cross-seed selection",
    ):
        build_release_eval_v2_prediction_manifest(inputs)

    _selection(tmp_path / "selection2.json", fixture["model"])
    # Restore through a fresh fixture path is simpler than allowing a stale
    # generation receipt to mask the intended prediction-byte replay check.
    inputs = replace(inputs, selection_receipt=tmp_path / "selection2.json")
    inputs.predictions.write_text('{"doc_id":"internal-doc","spans":[]}\n', encoding="utf-8")
    with pytest.raises(ReleaseEvalV2PredictionProvenanceError):
        replay_release_eval_v2_prediction_manifest(manifest, inputs)
