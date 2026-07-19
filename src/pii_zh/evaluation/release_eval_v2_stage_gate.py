"""Pre-open stage gates for the frozen release-evaluation-v2 corpus.

The two receipts in this module are deliberately not generic signatures:

* ``calibration_authorization`` is rebuilt from the frozen three-seed selector
  and the published v1-to-v2 amendment before the calibration JSONL may be
  opened;
* ``internal_unlock`` is built only after the selected model, calibration and
  complete community service identity have been frozen.

The pre-open guards inspect only receipt/config/model metadata and ``lstat``
the target JSONL.  They never open, hash or decode the target split.  The
caller may open it only after a guard returns successfully.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .release_eval_v2_prediction_provenance import (
    FROZEN_V2,
    AmendmentReplayInputs,
    CalibrationInputs,
    PredictionProvenanceInputs,
    _calibration_identity,
    _dataset_identity,
    _dataset_with_calibration_split,
    _model_identity,
    _service_identity,
)

CALIBRATION_AUTHORIZATION_SCHEMA_VERSION = (
    "pii-zh.release-eval-v2-calibration-authorization.v1"
)
INTERNAL_UNLOCK_SCHEMA_VERSION = "pii-zh.release-eval-v2-internal-unlock.v1"
FINAL_MODEL_BINDING_SCHEMA_VERSION = "pii-zh.community-final-model-binding.v2"
SERVICE_CONFIGURATION_BINDING_SCHEMA_VERSION = (
    "pii-zh.community-service-configuration-binding.v2"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024


class ReleaseEvalV2StageGateError(RuntimeError):
    """Fail-closed stage transition error without record contents or paths."""


@dataclass(frozen=True, slots=True)
class SelectionReplayEvidence:
    """Local evidence required to replay the immutable v3 selector."""

    protocol: Path
    development_manifest: Path
    v1_release_manifest: Path
    candidates: tuple[Path, Path, Path]
    selection_receipt: Path
    amendment: Path


@dataclass(frozen=True, slots=True)
class CalibrationAuthorizationInputs:
    """Inputs which authorize the first read of the v2 calibration split."""

    evidence: SelectionReplayEvidence
    dataset_manifest: Path
    materialization_receipt: Path
    freeze_receipt: Path
    calibration_gold: Path
    model_artifact: Path


@dataclass(frozen=True, slots=True)
class InternalUnlockInputs:
    """Post-calibration inputs frozen before any internal JSONL read."""

    authorization_inputs: CalibrationAuthorizationInputs
    authorization_receipt: Path
    calibration: CalibrationInputs


def _canonical_hash(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReleaseEvalV2StageGateError("stage receipt is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _pretty_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReleaseEvalV2StageGateError("stage receipt is not strict JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseEvalV2StageGateError("stage receipt contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ReleaseEvalV2StageGateError("stage receipt contains a non-finite number")


def _read_json(path: Path, *, field: str) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseEvalV2StageGateError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
            raise ReleaseEvalV2StageGateError(f"{field} must be a bounded regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ReleaseEvalV2StageGateError(f"{field} changed while being verified")
    except OSError as exc:
        raise ReleaseEvalV2StageGateError(f"{field} cannot be read") from exc
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvalV2StageGateError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseEvalV2StageGateError(f"{field} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseEvalV2StageGateError(f"{field} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], *, field: str) -> None:
    if set(value) != keys:
        raise ReleaseEvalV2StageGateError(f"{field} has an unexpected shape")


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReleaseEvalV2StageGateError(f"{field} is not a lowercase SHA-256")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, field: str, key: str) -> str:
    claimed = _digest(value.get(key), field=f"{field} self hash")
    unsigned = dict(value)
    unsigned.pop(key, None)
    if _canonical_hash(unsigned) != claimed:
        raise ReleaseEvalV2StageGateError(f"{field} self hash does not verify")
    return claimed


def _assert_path_free(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_path_free(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_path_free(nested)
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "~\\", "file:"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ReleaseEvalV2StageGateError("stage receipt contains a local path")


def _strict_amendment_replay(
    evidence: SelectionReplayEvidence,
) -> tuple[dict[str, Any], str]:
    try:
        from scripts.build_aiguard24_v3_release_eval_v2_amendment import (
            replay_amendment_from_selection_evidence,
        )

        return replay_amendment_from_selection_evidence(
            evidence.amendment,
            protocol_path=evidence.protocol,
            development_manifest_path=evidence.development_manifest,
            v1_release_manifest_path=evidence.v1_release_manifest,
            candidate_roots=evidence.candidates,
            selection_receipt_path=evidence.selection_receipt,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2StageGateError("v3 selection/v2 amendment replay failed") from exc


def _calibration_dataset(inputs: CalibrationAuthorizationInputs) -> dict[str, Any]:
    placeholder = PredictionProvenanceInputs(
        track="model_raw",
        target_split="calibration",
        predictions=inputs.calibration_gold,
        gold=inputs.calibration_gold,
        dataset_manifest=inputs.dataset_manifest,
        materialization_receipt=inputs.materialization_receipt,
        freeze_receipt=inputs.freeze_receipt,
        model_artifact=inputs.model_artifact,
        selection_receipt=inputs.evidence.selection_receipt,
        generation_receipt=inputs.calibration_gold,
    )
    dataset = _dataset_identity(placeholder)
    return _dataset_with_calibration_split(dataset, placeholder)


def build_calibration_authorization(
    inputs: CalibrationAuthorizationInputs,
) -> dict[str, Any]:
    """Strictly authorize calibration without opening internal evaluation."""

    amendment, amendment_file_sha256 = _strict_amendment_replay(inputs.evidence)
    amendment_sha256 = _verify_self_hash(
        amendment,
        field="v2 amendment",
        key="amendment_sha256",
    )
    dataset = _calibration_dataset(inputs)
    model = _model_identity(inputs.model_artifact)
    replay = _mapping(amendment.get("selection_replay"), field="amendment selection replay")
    selected = _mapping(replay.get("selected"), field="amendment selected model")
    replacement = _mapping(amendment.get("replacement_v2"), field="amendment v2 replacement")
    authorization = _mapping(
        amendment.get("stage_authorization"), field="amendment stage authorization"
    )
    _selection_document, selection_file_sha256 = _read_json(
        inputs.evidence.selection_receipt,
        field="v3 selection receipt",
    )
    if (
        amendment.get("status") != "V2_CALIBRATION_NEXT_NOT_RELEASE"
        or replay.get("receipt_file_sha256")
        != selection_file_sha256
        or selected.get("training_manifest_sha256") != model["training_manifest_sha256"]
        or selected.get("weights_combined_sha256") != model["weights_combined_sha256"]
        or selected.get("output_artifact_sha256") != model["output_artifact_sha256"]
        or replacement.get("dataset_id") != dataset["dataset_id"]
        or replacement.get("dataset_version") != dataset["dataset_version"]
        or replacement.get("manifest_file_sha256")
        != dataset["dataset_manifest_file_sha256"]
        or replacement.get("manifest_sha256") != dataset["dataset_manifest_sha256"]
        or replacement.get("calibration_sha256") != dataset["gold_file_sha256"]
        or replacement.get("supersession_receipt_file_sha256")
        != dataset["materialization_receipt_file_sha256"]
        or replacement.get("supersession_receipt_sha256")
        != dataset["materialization_receipt_sha256"]
        or replacement.get("supersession_freeze_file_sha256")
        != dataset["freeze_receipt_file_sha256"]
        or authorization.get("next_allowed_stage") != "v2_calibration"
        or authorization.get("v2_calibration_content_read_allowed") is not True
        or authorization.get("v2_internal_evaluation_content_read_allowed") is not False
    ):
        raise ReleaseEvalV2StageGateError(
            "selection, amendment, dataset and selected model do not form one closed chain"
        )
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_AUTHORIZATION_SCHEMA_VERSION,
        "receipt_type": "release_eval_v2_calibration_authorization",
        "status": "CALIBRATION_MODEL_RAW_AUTHORIZED_NOT_RELEASE",
        "amendment": {
            "file_sha256": amendment_file_sha256,
            "amendment_sha256": amendment_sha256,
            "selection_receipt_file_sha256": replay["receipt_file_sha256"],
            "selection_receipt_sha256": replay["receipt_sha256"],
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "dataset_version": dataset["dataset_version"],
            "dataset_manifest_file_sha256": dataset["dataset_manifest_file_sha256"],
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "materialization_receipt_file_sha256": dataset[
                "materialization_receipt_file_sha256"
            ],
            "materialization_receipt_sha256": dataset["materialization_receipt_sha256"],
            "freeze_receipt_file_sha256": dataset["freeze_receipt_file_sha256"],
            "target_split": "calibration",
            "gold_file_sha256": dataset["gold_file_sha256"],
            "gold_size_bytes": dataset["gold_size_bytes"],
            "gold_document_count": dataset["gold_document_count"],
        },
        "model": {
            "training_manifest_file_sha256": model["training_manifest_file_sha256"],
            "training_manifest_sha256": model["training_manifest_sha256"],
            "model_identity_sha256": model["identity_sha256"],
            "weights_combined_sha256": model["weights_combined_sha256"],
            "output_artifact_sha256": model["output_artifact_sha256"],
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
    }
    payload["receipt_sha256"] = _canonical_hash(payload)
    validate_calibration_authorization(payload)
    return payload


def validate_calibration_authorization(value: Mapping[str, Any]) -> str:
    """Validate closed, path-free authorization metadata and frozen v2 IDs."""

    _exact(
        value,
        {
            "schema_version",
            "receipt_type",
            "status",
            "amendment",
            "dataset",
            "model",
            "authorization",
            "privacy",
            "receipt_sha256",
        },
        field="calibration authorization",
    )
    amendment = _mapping(value.get("amendment"), field="authorization amendment")
    dataset = _mapping(value.get("dataset"), field="authorization dataset")
    model = _mapping(value.get("model"), field="authorization model")
    authorization = _mapping(value.get("authorization"), field="authorization policy")
    privacy = _mapping(value.get("privacy"), field="authorization privacy")
    _exact(
        amendment,
        {
            "file_sha256",
            "amendment_sha256",
            "selection_receipt_file_sha256",
            "selection_receipt_sha256",
        },
        field="authorization amendment",
    )
    _exact(
        dataset,
        {
            "dataset_id",
            "dataset_version",
            "dataset_manifest_file_sha256",
            "dataset_manifest_sha256",
            "materialization_receipt_file_sha256",
            "materialization_receipt_sha256",
            "freeze_receipt_file_sha256",
            "target_split",
            "gold_file_sha256",
            "gold_size_bytes",
            "gold_document_count",
        },
        field="authorization dataset",
    )
    _exact(
        model,
        {
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "model_identity_sha256",
            "weights_combined_sha256",
            "output_artifact_sha256",
        },
        field="authorization model",
    )
    _exact(
        authorization,
        {
            "allowed_track",
            "allowed_generator_mode",
            "calibration_bundle_already_applied",
            "checkpoint_or_model_reselection_allowed",
            "final_metric_claim_allowed",
            "internal_evaluation_content_read_allowed",
        },
        field="authorization policy",
    )
    _exact(
        privacy,
        {
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
            "internal_evaluation_opened_hashed_or_decoded",
        },
        field="authorization privacy",
    )
    frozen_split = _mapping(FROZEN_V2["splits"], field="frozen v2 splits")["calibration"]
    if (
        value.get("schema_version") != CALIBRATION_AUTHORIZATION_SCHEMA_VERSION
        or value.get("receipt_type") != "release_eval_v2_calibration_authorization"
        or value.get("status") != "CALIBRATION_MODEL_RAW_AUTHORIZED_NOT_RELEASE"
        or dataset.get("dataset_id") != FROZEN_V2["dataset_id"]
        or dataset.get("dataset_version") != FROZEN_V2["dataset_version"]
        or dataset.get("dataset_manifest_file_sha256")
        != FROZEN_V2["dataset_manifest_file_sha256"]
        or dataset.get("dataset_manifest_sha256") != FROZEN_V2["dataset_manifest_sha256"]
        or dataset.get("materialization_receipt_file_sha256")
        != FROZEN_V2["materialization_receipt_file_sha256"]
        or dataset.get("materialization_receipt_sha256")
        != FROZEN_V2["materialization_receipt_sha256"]
        or dataset.get("freeze_receipt_file_sha256")
        != FROZEN_V2["freeze_receipt_file_sha256"]
        or dataset.get("target_split") != "calibration"
        or dataset.get("gold_file_sha256") != frozen_split["sha256"]
        or dataset.get("gold_size_bytes") != frozen_split["bytes"]
        or dataset.get("gold_document_count") != frozen_split["records"]
        or authorization
        != {
            "allowed_track": "model_raw",
            "allowed_generator_mode": "model-raw",
            "calibration_bundle_already_applied": False,
            "checkpoint_or_model_reselection_allowed": False,
            "final_metric_claim_allowed": False,
            "internal_evaluation_content_read_allowed": False,
        }
        or any(privacy.values())
    ):
        raise ReleaseEvalV2StageGateError("calibration authorization is not the frozen v2 gate")
    for field, digest in {**amendment, **model}.items():
        _digest(digest, field=f"authorization {field}")
    _assert_path_free(value)
    return _verify_self_hash(value, field="calibration authorization", key="receipt_sha256")


def _load_calibration_authorization(path: Path) -> tuple[dict[str, Any], str, str]:
    value, file_sha256 = _read_json(path, field="calibration authorization")
    return value, file_sha256, validate_calibration_authorization(value)


def replay_calibration_authorization(
    path: Path, inputs: CalibrationAuthorizationInputs
) -> tuple[dict[str, Any], str]:
    """Recompute the authorization from source evidence and require equality."""

    observed, file_sha256, _self = _load_calibration_authorization(path)
    expected = build_calibration_authorization(inputs)
    if observed != expected:
        raise ReleaseEvalV2StageGateError("calibration authorization strict replay differs")
    return observed, file_sha256


def _binding_payloads(
    *,
    model: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS

    seed = model.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReleaseEvalV2StageGateError("selected model seed is invalid")
    model_id = f"aiguard24-full-seed{seed}"
    model_binding: dict[str, Any] = {
        "schema_version": FINAL_MODEL_BINDING_SCHEMA_VERSION,
        "model_id": model_id,
        "artifact_class": "24_label_zh_hans_token_classifier",
        "label_count": 24,
        "ordered_labels": list(PII_CORE_LABELS),
        "taxonomy_version": model["taxonomy_version"],
        "attention_mode": model["attention_mode"],
        "training_manifest_file_sha256": model["training_manifest_file_sha256"],
        "training_manifest_sha256": model["training_manifest_sha256"],
        "model_identity_sha256": model["identity_sha256"],
        "artifact_sha256": model["output_artifact_sha256"],
    }
    model_binding["manifest_sha256"] = _canonical_hash(model_binding)
    service = _service_identity(model=model, calibration=calibration)
    service_binding: dict[str, Any] = {
        "schema_version": SERVICE_CONFIGURATION_BINDING_SCHEMA_VERSION,
        "service_id": "community-model-cascade-v1",
        "profile_id": service["profile_id"],
        "canonical_track": "full_system",
        "final_model_id": model_id,
        "final_model_manifest_sha256": hashlib.sha256(
            _pretty_bytes(model_binding)
        ).hexdigest(),
        "model_identity_sha256": model["identity_sha256"],
        "calibration_bundle_file_sha256": calibration["bundle_file_sha256"],
        "implementation_sha256": service["implementation_sha256"],
        "configuration_sha256": service["configuration_sha256"],
    }
    service_binding["manifest_sha256"] = _canonical_hash(service_binding)
    _assert_path_free(model_binding)
    _assert_path_free(service_binding)
    return model_binding, service_binding


def build_internal_unlock(
    inputs: InternalUnlockInputs,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Freeze model/calibration/service and authorize a one-shot internal read."""

    authorization, authorization_file_sha256 = replay_calibration_authorization(
        inputs.authorization_receipt,
        inputs.authorization_inputs,
    )
    dataset = dict(_mapping(authorization["dataset"], field="authorization dataset"))
    dataset["calibration_gold_file_sha256"] = dataset["gold_file_sha256"]
    dataset["calibration_document_count"] = dataset["gold_document_count"]
    model = _model_identity(inputs.authorization_inputs.model_artifact)
    placeholder = PredictionProvenanceInputs(
        track="model_raw",
        target_split="internal_evaluation",
        predictions=inputs.calibration.fit_predictions,
        gold=inputs.calibration.fit_gold,
        dataset_manifest=inputs.authorization_inputs.dataset_manifest,
        materialization_receipt=inputs.authorization_inputs.materialization_receipt,
        freeze_receipt=inputs.authorization_inputs.freeze_receipt,
        model_artifact=inputs.authorization_inputs.model_artifact,
        selection_receipt=inputs.authorization_inputs.evidence.selection_receipt,
        generation_receipt=inputs.calibration.fit_generation_receipt,
        amendment=AmendmentReplayInputs(
            amendment=inputs.authorization_inputs.evidence.amendment,
            protocol=inputs.authorization_inputs.evidence.protocol,
            development_manifest=inputs.authorization_inputs.evidence.development_manifest,
            v1_release_manifest=inputs.authorization_inputs.evidence.v1_release_manifest,
            candidate_roots=inputs.authorization_inputs.evidence.candidates,
        ),
        calibration=inputs.calibration,
    )
    calibration = _calibration_identity(
        inputs.calibration,
        dataset=dataset,
        model=model,
        base_inputs=placeholder,
    )
    model_binding, service_binding = _binding_payloads(
        model=model,
        calibration=calibration,
    )
    internal = _mapping(FROZEN_V2["splits"], field="frozen v2 splits")[
        "internal_evaluation"
    ]
    unlock: dict[str, Any] = {
        "schema_version": INTERNAL_UNLOCK_SCHEMA_VERSION,
        "receipt_type": "release_eval_v2_internal_preopen_unlock",
        "status": "INTERNAL_EVALUATION_AUTHORIZED_ONE_SHOT",
        "dataset": {
            "dataset_id": FROZEN_V2["dataset_id"],
            "dataset_version": FROZEN_V2["dataset_version"],
            "dataset_manifest_file_sha256": FROZEN_V2["dataset_manifest_file_sha256"],
            "dataset_manifest_sha256": FROZEN_V2["dataset_manifest_sha256"],
            "materialization_receipt_file_sha256": FROZEN_V2[
                "materialization_receipt_file_sha256"
            ],
            "materialization_receipt_sha256": FROZEN_V2[
                "materialization_receipt_sha256"
            ],
            "freeze_receipt_file_sha256": FROZEN_V2["freeze_receipt_file_sha256"],
            "target_split": "internal_evaluation",
            "gold_file_sha256": internal["sha256"],
            "gold_size_bytes": internal["bytes"],
            "gold_document_count": internal["records"],
        },
        "upstream": {
            "calibration_authorization_file_sha256": authorization_file_sha256,
            "calibration_authorization_sha256": authorization["receipt_sha256"],
            "amendment_file_sha256": authorization["amendment"]["file_sha256"],
            "amendment_sha256": authorization["amendment"]["amendment_sha256"],
        },
        "bindings": {
            "final_model_binding_file_sha256": hashlib.sha256(
                _pretty_bytes(model_binding)
            ).hexdigest(),
            "final_model_binding_sha256": model_binding["manifest_sha256"],
            "service_configuration_binding_file_sha256": hashlib.sha256(
                _pretty_bytes(service_binding)
            ).hexdigest(),
            "service_configuration_binding_sha256": service_binding["manifest_sha256"],
            "calibration_bundle_file_sha256": calibration["bundle_file_sha256"],
            "calibration_diagnostics_manifest_sha256": calibration[
                "diagnostics_manifest_sha256"
            ],
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
    }
    unlock["receipt_sha256"] = _canonical_hash(unlock)
    validate_internal_unlock_documents(unlock, model_binding, service_binding)
    return model_binding, service_binding, unlock


def validate_internal_unlock_documents(
    unlock: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    service_binding: Mapping[str, Any],
) -> str:
    """Validate the three closed path-free final-freeze documents."""

    _exact(
        unlock,
        {
            "schema_version",
            "receipt_type",
            "status",
            "dataset",
            "upstream",
            "bindings",
            "authorization",
            "privacy",
            "receipt_sha256",
        },
        field="internal unlock",
    )
    _exact(
        model_binding,
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
        field="final model binding",
    )
    _exact(
        service_binding,
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
        field="service configuration binding",
    )
    dataset = _mapping(unlock.get("dataset"), field="unlock dataset")
    upstream = _mapping(unlock.get("upstream"), field="unlock upstream")
    bindings = _mapping(unlock.get("bindings"), field="unlock bindings")
    authorization = _mapping(unlock.get("authorization"), field="unlock authorization")
    privacy = _mapping(unlock.get("privacy"), field="unlock privacy")
    _exact(
        dataset,
        {
            "dataset_id",
            "dataset_version",
            "dataset_manifest_file_sha256",
            "dataset_manifest_sha256",
            "materialization_receipt_file_sha256",
            "materialization_receipt_sha256",
            "freeze_receipt_file_sha256",
            "target_split",
            "gold_file_sha256",
            "gold_size_bytes",
            "gold_document_count",
        },
        field="unlock dataset",
    )
    _exact(
        upstream,
        {
            "calibration_authorization_file_sha256",
            "calibration_authorization_sha256",
            "amendment_file_sha256",
            "amendment_sha256",
        },
        field="unlock upstream",
    )
    _exact(
        bindings,
        {
            "final_model_binding_file_sha256",
            "final_model_binding_sha256",
            "service_configuration_binding_file_sha256",
            "service_configuration_binding_sha256",
            "calibration_bundle_file_sha256",
            "calibration_diagnostics_manifest_sha256",
        },
        field="unlock bindings",
    )
    _exact(
        authorization,
        {
            "allowed_tracks",
            "allowed_consumers",
            "one_shot",
            "model_checkpoint_or_calibration_change_after_read_allowed",
            "service_configuration_change_after_read_allowed",
            "quality_production_allowed",
        },
        field="unlock authorization",
    )
    _exact(
        privacy,
        {
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
            "internal_evaluation_opened_hashed_or_decoded_during_unlock",
        },
        field="unlock privacy",
    )
    for name, digest in {**upstream, **bindings}.items():
        _digest(digest, field=f"unlock {name}")
    internal = _mapping(FROZEN_V2["splits"], field="frozen v2 splits")[
        "internal_evaluation"
    ]
    if (
        unlock.get("schema_version") != INTERNAL_UNLOCK_SCHEMA_VERSION
        or unlock.get("receipt_type") != "release_eval_v2_internal_preopen_unlock"
        or unlock.get("status") != "INTERNAL_EVALUATION_AUTHORIZED_ONE_SHOT"
        or model_binding.get("schema_version") != FINAL_MODEL_BINDING_SCHEMA_VERSION
        or model_binding.get("artifact_class") != "24_label_zh_hans_token_classifier"
        or model_binding.get("label_count") != 24
        or service_binding.get("schema_version")
        != SERVICE_CONFIGURATION_BINDING_SCHEMA_VERSION
        or service_binding.get("canonical_track") != "full_system"
        or service_binding.get("final_model_id") != model_binding.get("model_id")
        or service_binding.get("model_identity_sha256")
        != model_binding.get("model_identity_sha256")
        or dataset.get("dataset_id") != FROZEN_V2["dataset_id"]
        or dataset.get("dataset_version") != FROZEN_V2["dataset_version"]
        or dataset.get("dataset_manifest_file_sha256")
        != FROZEN_V2["dataset_manifest_file_sha256"]
        or dataset.get("dataset_manifest_sha256") != FROZEN_V2["dataset_manifest_sha256"]
        or dataset.get("materialization_receipt_file_sha256")
        != FROZEN_V2["materialization_receipt_file_sha256"]
        or dataset.get("materialization_receipt_sha256")
        != FROZEN_V2["materialization_receipt_sha256"]
        or dataset.get("freeze_receipt_file_sha256")
        != FROZEN_V2["freeze_receipt_file_sha256"]
        or dataset.get("target_split") != "internal_evaluation"
        or dataset.get("gold_file_sha256") != internal["sha256"]
        or dataset.get("gold_size_bytes") != internal["bytes"]
        or dataset.get("gold_document_count") != internal["records"]
        or authorization
        != {
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
        }
        or any(privacy.values())
        or bindings.get("final_model_binding_file_sha256")
        != hashlib.sha256(_pretty_bytes(model_binding)).hexdigest()
        or bindings.get("final_model_binding_sha256")
        != model_binding.get("manifest_sha256")
        or bindings.get("service_configuration_binding_file_sha256")
        != hashlib.sha256(_pretty_bytes(service_binding)).hexdigest()
        or bindings.get("service_configuration_binding_sha256")
        != service_binding.get("manifest_sha256")
        or service_binding.get("final_model_manifest_sha256")
        != hashlib.sha256(_pretty_bytes(model_binding)).hexdigest()
        or bindings.get("calibration_bundle_file_sha256")
        != service_binding.get("calibration_bundle_file_sha256")
    ):
        raise ReleaseEvalV2StageGateError("internal unlock/final bindings are inconsistent")
    _verify_self_hash(model_binding, field="final model binding", key="manifest_sha256")
    _verify_self_hash(
        service_binding,
        field="service configuration binding",
        key="manifest_sha256",
    )
    _assert_path_free(unlock)
    _assert_path_free(model_binding)
    _assert_path_free(service_binding)
    return _verify_self_hash(unlock, field="internal unlock", key="receipt_sha256")


def replay_internal_unlock(
    unlock_path: Path,
    model_binding_path: Path,
    service_binding_path: Path,
    inputs: InternalUnlockInputs,
) -> dict[str, Any]:
    """Rebuild all final-freeze documents without opening internal JSONL."""

    observed_unlock, unlock_file_sha256 = _read_json(
        unlock_path, field="internal unlock"
    )
    observed_model, model_file_sha256 = _read_json(
        model_binding_path, field="final model binding"
    )
    observed_service, service_file_sha256 = _read_json(
        service_binding_path, field="service configuration binding"
    )
    expected_model, expected_service, expected_unlock = build_internal_unlock(inputs)
    if (
        observed_model != expected_model
        or observed_service != expected_service
        or observed_unlock != expected_unlock
    ):
        raise ReleaseEvalV2StageGateError("internal unlock strict replay differs")
    validate_internal_unlock_documents(
        observed_unlock,
        observed_model,
        observed_service,
    )
    return {
        "status": "INTERNAL_UNLOCK_STRICT_REPLAY_PASS",
        "unlock_file_sha256": unlock_file_sha256,
        "unlock_sha256": observed_unlock["receipt_sha256"],
        "final_model_binding_file_sha256": model_file_sha256,
        "final_model_binding_sha256": observed_model["manifest_sha256"],
        "service_configuration_binding_file_sha256": service_file_sha256,
        "service_configuration_binding_sha256": observed_service["manifest_sha256"],
        "internal_evaluation_opened_hashed_or_decoded": False,
    }


def _preopen_target_stat(
    path: Path, *, split: Literal["calibration", "internal_evaluation"]
) -> None:
    """Check target identity hints without opening or hashing the JSONL."""

    expected = _mapping(FROZEN_V2["splits"], field="frozen v2 splits")[split]
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ReleaseEvalV2StageGateError("target split is unavailable before open") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or path.is_symlink()
        or observed.st_size != expected["bytes"]
        or path.name != f"{split}.jsonl"
    ):
        raise ReleaseEvalV2StageGateError(
            "target split pre-open stat does not match the frozen identity"
        )


def guard_calibration_preopen(
    authorization_path: Path,
    *,
    input_path: Path,
    mode: str,
    model_artifact: Path,
) -> dict[str, Any]:
    """Fail before opening calibration unless exact authorization/model match."""

    value, file_sha256, self_hash = _load_calibration_authorization(authorization_path)
    model = _model_identity(model_artifact)
    bound = _mapping(value["model"], field="authorization model")
    if (
        mode != "model-raw"
        or bound.get("training_manifest_sha256") != model["training_manifest_sha256"]
        or bound.get("model_identity_sha256") != model["identity_sha256"]
    ):
        raise ReleaseEvalV2StageGateError(
            "calibration authorizes only threshold-zero selected-model output"
        )
    _preopen_target_stat(input_path, split="calibration")
    return {
        "status": "CALIBRATION_PREOPEN_PASS",
        "authorization_file_sha256": file_sha256,
        "authorization_sha256": self_hash,
        "expected_input_sha256": value["dataset"]["gold_file_sha256"],
    }


def guard_internal_preopen(
    unlock_path: Path,
    model_binding_path: Path,
    service_binding_path: Path,
    *,
    input_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Validate final freeze before any open/hash/decode of internal JSONL."""

    unlock, unlock_file_sha256 = _read_json(unlock_path, field="internal unlock")
    model_binding, model_file_sha256 = _read_json(
        model_binding_path, field="final model binding"
    )
    service_binding, service_file_sha256 = _read_json(
        service_binding_path, field="service configuration binding"
    )
    unlock_self = validate_internal_unlock_documents(
        unlock,
        model_binding,
        service_binding,
    )
    bindings = _mapping(unlock["bindings"], field="unlock bindings")
    if (
        mode
        not in {
            "model-raw",
            "model-only",
            "cascade",
            "comparator-model-raw",
            "comparator-full-system",
            "quality",
        }
        or bindings.get("final_model_binding_file_sha256") != model_file_sha256
        or bindings.get("service_configuration_binding_file_sha256") != service_file_sha256
    ):
        raise ReleaseEvalV2StageGateError("internal pre-open mode/binding is not authorized")
    _preopen_target_stat(input_path, split="internal_evaluation")
    return {
        "status": "INTERNAL_PREOPEN_PASS",
        "unlock_file_sha256": unlock_file_sha256,
        "unlock_sha256": unlock_self,
        "expected_input_sha256": unlock["dataset"]["gold_file_sha256"],
    }


def write_read_only_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically publish one new deterministic mode-0444 JSON document."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ReleaseEvalV2StageGateError("refusing to overwrite stage receipt")
    content = _pretty_bytes(value)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written < 1:
                raise OSError("short write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, destination, follow_symlinks=False)
    except OSError as exc:
        raise ReleaseEvalV2StageGateError("cannot publish stage receipt") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


__all__ = [
    "CALIBRATION_AUTHORIZATION_SCHEMA_VERSION",
    "FINAL_MODEL_BINDING_SCHEMA_VERSION",
    "INTERNAL_UNLOCK_SCHEMA_VERSION",
    "SERVICE_CONFIGURATION_BINDING_SCHEMA_VERSION",
    "CalibrationAuthorizationInputs",
    "InternalUnlockInputs",
    "ReleaseEvalV2StageGateError",
    "SelectionReplayEvidence",
    "build_calibration_authorization",
    "build_internal_unlock",
    "guard_calibration_preopen",
    "guard_internal_preopen",
    "replay_calibration_authorization",
    "replay_internal_unlock",
    "validate_calibration_authorization",
    "validate_internal_unlock_documents",
    "write_read_only_json",
]
