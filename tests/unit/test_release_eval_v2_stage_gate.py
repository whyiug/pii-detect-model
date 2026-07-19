from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pii_zh.evaluation import release_eval_v2_stage_gate as gate


def _write(path: Path, value: dict[str, Any]) -> str:
    payload = gate._pretty_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    result[key] = gate._canonical_hash(result)
    return result


def _frozen() -> dict[str, Any]:
    return {
        "dataset_id": "pii_zh_synthetic_sota_release_eval_v2",
        "dataset_version": "2.0.0",
        "dataset_manifest_file_sha256": "1" * 64,
        "dataset_manifest_sha256": "2" * 64,
        "materialization_receipt_file_sha256": "3" * 64,
        "materialization_receipt_sha256": "4" * 64,
        "freeze_receipt_file_sha256": "5" * 64,
        "splits": {
            "calibration": {"sha256": "6" * 64, "bytes": 17, "records": 2},
            "internal_evaluation": {"sha256": "7" * 64, "bytes": 23, "records": 3},
        },
    }


def _authorization() -> dict[str, Any]:
    frozen = gate.FROZEN_V2
    split = frozen["splits"]["calibration"]
    return _seal(
        {
            "schema_version": gate.CALIBRATION_AUTHORIZATION_SCHEMA_VERSION,
            "receipt_type": "release_eval_v2_calibration_authorization",
            "status": "CALIBRATION_MODEL_RAW_AUTHORIZED_NOT_RELEASE",
            "amendment": {
                "file_sha256": "8" * 64,
                "amendment_sha256": "9" * 64,
                "selection_receipt_file_sha256": "a" * 64,
                "selection_receipt_sha256": "b" * 64,
            },
            "dataset": {
                "dataset_id": frozen["dataset_id"],
                "dataset_version": frozen["dataset_version"],
                "dataset_manifest_file_sha256": frozen[
                    "dataset_manifest_file_sha256"
                ],
                "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
                "materialization_receipt_file_sha256": frozen[
                    "materialization_receipt_file_sha256"
                ],
                "materialization_receipt_sha256": frozen[
                    "materialization_receipt_sha256"
                ],
                "freeze_receipt_file_sha256": frozen["freeze_receipt_file_sha256"],
                "target_split": "calibration",
                "gold_file_sha256": split["sha256"],
                "gold_size_bytes": split["bytes"],
                "gold_document_count": split["records"],
            },
            "model": {
                "training_manifest_file_sha256": "c" * 64,
                "training_manifest_sha256": "d" * 64,
                "model_identity_sha256": "e" * 64,
                "weights_combined_sha256": "f" * 64,
                "output_artifact_sha256": "0" * 64,
            },
            "authorization": {
                "allowed_track": "model_raw",
                "allowed_generator_mode": "model-raw",
                "calibration_bundle_already_applied": False,
                "checkpoint_or_model_reselection_allowed": False,
                "final_metric_claim_allowed": False,
                "internal_evaluation_content_read_allowed": False,
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
                "internal_evaluation_opened_hashed_or_decoded": False,
            },
        },
        "receipt_sha256",
    )


def _final_documents(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = _seal(
        {
            "schema_version": gate.FINAL_MODEL_BINDING_SCHEMA_VERSION,
            "model_id": "aiguard24-full-seed97",
            "artifact_class": "24_label_zh_hans_token_classifier",
            "label_count": 24,
            "ordered_labels": [f"LABEL_{index}" for index in range(24)],
            "taxonomy_version": "1.0.0",
            "attention_mode": "full",
            "training_manifest_file_sha256": "a" * 64,
            "training_manifest_sha256": "b" * 64,
            "model_identity_sha256": "c" * 64,
            "artifact_sha256": "d" * 64,
        },
        "manifest_sha256",
    )
    model_path = tmp_path / "model-binding.json"
    model_file = _write(model_path, model)
    service = _seal(
        {
            "schema_version": gate.SERVICE_CONFIGURATION_BINDING_SCHEMA_VERSION,
            "service_id": "community-model-cascade-v1",
            "profile_id": "community-model-cascade-v1",
            "canonical_track": "full_system",
            "final_model_id": model["model_id"],
            "final_model_manifest_sha256": model_file,
            "model_identity_sha256": model["model_identity_sha256"],
            "calibration_bundle_file_sha256": "e" * 64,
            "implementation_sha256": "f" * 64,
            "configuration_sha256": "0" * 64,
        },
        "manifest_sha256",
    )
    service_path = tmp_path / "service-binding.json"
    service_file = _write(service_path, service)
    frozen = gate.FROZEN_V2
    internal = frozen["splits"]["internal_evaluation"]
    unlock = _seal(
        {
            "schema_version": gate.INTERNAL_UNLOCK_SCHEMA_VERSION,
            "receipt_type": "release_eval_v2_internal_preopen_unlock",
            "status": "INTERNAL_EVALUATION_AUTHORIZED_ONE_SHOT",
            "dataset": {
                "dataset_id": frozen["dataset_id"],
                "dataset_version": frozen["dataset_version"],
                "dataset_manifest_file_sha256": frozen[
                    "dataset_manifest_file_sha256"
                ],
                "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
                "materialization_receipt_file_sha256": frozen[
                    "materialization_receipt_file_sha256"
                ],
                "materialization_receipt_sha256": frozen[
                    "materialization_receipt_sha256"
                ],
                "freeze_receipt_file_sha256": frozen["freeze_receipt_file_sha256"],
                "target_split": "internal_evaluation",
                "gold_file_sha256": internal["sha256"],
                "gold_size_bytes": internal["bytes"],
                "gold_document_count": internal["records"],
            },
            "upstream": {
                "calibration_authorization_file_sha256": "1" * 64,
                "calibration_authorization_sha256": "2" * 64,
                "amendment_file_sha256": "3" * 64,
                "amendment_sha256": "4" * 64,
            },
            "bindings": {
                "final_model_binding_file_sha256": model_file,
                "final_model_binding_sha256": model["manifest_sha256"],
                "service_configuration_binding_file_sha256": service_file,
                "service_configuration_binding_sha256": service["manifest_sha256"],
                "calibration_bundle_file_sha256": service[
                    "calibration_bundle_file_sha256"
                ],
                "calibration_diagnostics_manifest_sha256": "5" * 64,
            },
            "authorization": {
                "allowed_tracks": ["model_raw", "model_calibrated", "full_system"],
                "allowed_consumers": [
                    "candidate_prediction_generation",
                    "comparator_generation",
                    "quality_production",
                ],
                "one_shot": True,
                "model_checkpoint_or_calibration_change_after_read_allowed": False,
                "service_configuration_change_after_read_allowed": False,
                "quality_production_allowed": True,
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
                "internal_evaluation_opened_hashed_or_decoded_during_unlock": False,
            },
        },
        "receipt_sha256",
    )
    unlock_path = tmp_path / "unlock.json"
    _write(unlock_path, unlock)
    return unlock_path, model_path, service_path


def test_calibration_authorization_is_closed_and_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "FROZEN_V2", _frozen())
    value = _authorization()
    assert gate.validate_calibration_authorization(value) == value["receipt_sha256"]

    value["dataset"]["target_split"] = "internal_evaluation"
    value = _seal(value, "receipt_sha256")
    with pytest.raises(gate.ReleaseEvalV2StageGateError):
        gate.validate_calibration_authorization(value)


def test_internal_guard_checks_unlock_before_target_content_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "FROZEN_V2", _frozen())
    unlock, model, service = _final_documents(tmp_path)
    target = tmp_path / "internal_evaluation.jsonl"
    with target.open("wb") as handle:
        handle.truncate(gate.FROZEN_V2["splits"]["internal_evaluation"]["bytes"])

    result = gate.guard_internal_preopen(
        unlock,
        model,
        service,
        input_path=target,
        mode="model-only",
    )
    assert result["status"] == "INTERNAL_PREOPEN_PASS"

    document = json.loads(unlock.read_text(encoding="utf-8"))
    document["authorization"]["one_shot"] = False
    _write(unlock, _seal(document, "receipt_sha256"))
    with pytest.raises(gate.ReleaseEvalV2StageGateError):
        gate.guard_internal_preopen(
            unlock,
            model,
            service,
            input_path=target,
            mode="quality",
        )
