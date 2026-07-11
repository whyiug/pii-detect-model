"""Content-addressed, privacy-safe provenance for evaluation artifacts.

The helpers in this module deliberately return only hashes, counters, and
allowlisted identity fields.  Local paths and manifest bodies are never copied
into an evaluation report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from .io import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    load_prediction_jsonl,
    write_prediction_jsonl,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_SAFE_LOCATOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+/-]{0,127}$")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_line_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def canonical_json_hash(value: Any) -> str:
    """Hash a JSON value using the repository's canonical manifest encoding."""

    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def add_manifest_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a manifest payload and attach its logical content address."""

    manifest = dict(payload)
    if "manifest_sha256" in manifest:
        raise ValueError("manifest payload already contains manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    return manifest


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise EvaluationDataError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_PATTERN.fullmatch(value) is None:
        raise EvaluationDataError(f"{field} must be a path-free stable identifier")
    return value


def _require_safe_locator(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_LOCATOR_PATTERN.fullmatch(value) is None
        or ".." in value
    ):
        raise EvaluationDataError(f"{field} must be a stable public source locator")
    return value


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationDataError(f"{field} must be a non-negative integer")
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, field: str, allowed: set[str], required: set[str] | None = None
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise EvaluationDataError(f"{field} contains unsupported fields")
    missing = (required or allowed) - set(value)
    if missing:
        raise EvaluationDataError(f"{field} is missing required fields")


def _require_privacy_flags(
    value: object,
    *,
    field: str,
    required_flags: Sequence[str],
) -> None:
    privacy = _require_mapping(value, field=field)
    for name in required_flags:
        if privacy.get(name) is not False:
            raise EvaluationDataError(f"{field}.{name} must be false")


def _load_manifest(path: str | Path, *, kind: str) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"{kind} manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationDataError(f"{kind} manifest must be a JSON object")
    expected = _require_sha256(value.get("manifest_sha256"), field=f"{kind}.manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if canonical_json_hash(unsigned) != expected:
        raise EvaluationDataError(f"{kind} manifest content hash does not verify")
    return value, sha256_file(manifest_path)


def verified_manifest_hash(path: str | Path, *, kind: str = "manifest") -> str:
    """Verify a self-hashed JSON manifest and return its logical content address."""

    manifest, _ = _load_manifest(path, kind=kind)
    return _require_sha256(manifest.get("manifest_sha256"), field=f"{kind}.manifest_sha256")


def _prediction_base(
    *,
    predictions_path: str | Path,
    dataset_manifest_path: str | Path,
    prediction_document_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        isinstance(prediction_document_count, bool)
        or not isinstance(prediction_document_count, int)
        or prediction_document_count < 0
    ):
        raise EvaluationDataError("prediction_document_count must be a non-negative integer")
    dataset, _ = _load_manifest(dataset_manifest_path, kind="dataset")
    if dataset.get("manifest_type") != "evaluation_dataset":
        raise EvaluationDataError("dataset manifest_type must be evaluation_dataset")
    base = {
        "schema_version": 1,
        "manifest_type": "prediction",
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_document_count": prediction_document_count,
        "dataset_manifest_sha256": _require_sha256(
            dataset.get("manifest_sha256"), field="dataset.manifest_sha256"
        ),
    }
    return dataset, base


def build_model_prediction_manifest(
    *,
    predictions_path: str | Path,
    dataset_manifest_path: str | Path,
    model_training_manifest_path: str | Path,
    prediction_document_count: int,
    attention_mode: str,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a path-free manifest for predictions from a trained model.

    The seed and training content address are taken from the verified training
    manifest.  The caller supplies only allowlisted identity of the model that
    was actually loaded; local directory names and config bodies are excluded.
    """

    _, manifest = _prediction_base(
        predictions_path=predictions_path,
        dataset_manifest_path=dataset_manifest_path,
        prediction_document_count=prediction_document_count,
    )
    training, _ = _load_manifest(model_training_manifest_path, kind="model training")
    if training.get("status") != "completed":
        raise EvaluationDataError("model training manifest must have completed status")
    output_artifact = training.get("output_artifact")
    if not isinstance(output_artifact, Mapping):
        raise EvaluationDataError("model training manifest lacks output artifact binding")
    output_files = output_artifact.get("files")
    if not isinstance(output_files, Mapping):
        raise EvaluationDataError("model training output artifact files must be an object")
    recorded_config_sha256 = _require_sha256(
        output_files.get("config.json"),
        field="model_training.output_artifact.files.config.json",
    )
    recorded_weights_sha256 = _require_sha256(
        output_files.get("model.safetensors"),
        field="model_training.output_artifact.files.model.safetensors",
    )
    seed = training.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationDataError("model training manifest must contain an integer seed")
    normalized_attention = _require_safe_id(attention_mode, field="attention_mode")
    if training.get("attention_mode") != normalized_attention:
        raise EvaluationDataError("loaded model attention mode does not match training manifest")

    allowed_model_fields = {
        "architecture_version",
        "config_sha256",
        "label_schema_sha256",
        "model_type",
        "weights_sha256",
    }
    if set(model_identity) - allowed_model_fields:
        raise EvaluationDataError("model identity contains unsupported fields")
    normalized_model: dict[str, str] = {}
    for field in ("config_sha256", "label_schema_sha256", "weights_sha256"):
        value = model_identity.get(field)
        if value is not None:
            normalized_model[field] = _require_sha256(value, field=f"model_identity.{field}")
    for field in ("architecture_version", "model_type"):
        value = model_identity.get(field)
        if value is not None:
            normalized_model[field] = _require_safe_id(value, field=f"model_identity.{field}")
    if not {"config_sha256", "model_type", "weights_sha256"} <= set(normalized_model):
        raise EvaluationDataError("model identity is incomplete")
    if normalized_model["config_sha256"] != recorded_config_sha256:
        raise EvaluationDataError("loaded model config does not match training output artifact")
    if normalized_model["weights_sha256"] != recorded_weights_sha256:
        raise EvaluationDataError("loaded model weights do not match training output artifact")
    training_label_schema = training.get("label_schema_sha256")
    model_label_schema = normalized_model.get("label_schema_sha256")
    if training_label_schema is not None:
        expected_label_schema = _require_sha256(
            training_label_schema, field="model_training.label_schema_sha256"
        )
        if model_label_schema != expected_label_schema:
            raise EvaluationDataError("loaded model label schema does not match training manifest")
    normalized_model["identity_sha256"] = canonical_json_hash(normalized_model)

    manifest.update(
        {
            "predictor_type": "model",
            "model_training_manifest_sha256": _require_sha256(
                training.get("manifest_sha256"), field="model_training.manifest_sha256"
            ),
            "seed": seed,
            "attention_mode": normalized_attention,
            "model_identity": normalized_model,
        }
    )
    manifest["prediction_id"] = "model-" + canonical_json_hash(manifest)[:24]
    manifest["privacy"] = {
        "contains_paths": False,
        "contains_raw_text": False,
        "contains_entity_values": False,
    }
    return add_manifest_hash(manifest)


def build_rule_prediction_manifest(
    *,
    predictions_path: str | Path,
    dataset_manifest_path: str | Path,
    prediction_document_count: int,
    ruleset_id: str,
    implementation_sha256: str,
    configuration_sha256: str,
) -> dict[str, Any]:
    """Build a rule-only prediction manifest without a fabricated model identity."""

    _, manifest = _prediction_base(
        predictions_path=predictions_path,
        dataset_manifest_path=dataset_manifest_path,
        prediction_document_count=prediction_document_count,
    )
    manifest.update(
        {
            "predictor_type": "rules",
            "rules_identity": {
                "ruleset_id": _require_safe_id(ruleset_id, field="ruleset_id"),
                "implementation_sha256": _require_sha256(
                    implementation_sha256, field="implementation_sha256"
                ),
                "configuration_sha256": _require_sha256(
                    configuration_sha256, field="configuration_sha256"
                ),
            },
        }
    )
    manifest["prediction_id"] = "rules-" + canonical_json_hash(manifest)[:24]
    manifest["privacy"] = {
        "contains_paths": False,
        "contains_raw_text": False,
        "contains_entity_values": False,
    }
    return add_manifest_hash(manifest)


def _release_component_prediction(
    *,
    manifest_path: str | Path,
    predictions_path: str | Path,
    predictor_type: str,
    dataset_manifest_sha256: str,
    prediction_document_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, identity = _prediction_identity(manifest_path, predictions_path)
    if identity["predictor_type"] != predictor_type:
        raise EvaluationDataError(f"system {predictor_type} component has the wrong predictor_type")
    allowed = {
        "schema_version",
        "manifest_type",
        "prediction_id",
        "predictor_type",
        "predictions_sha256",
        "prediction_document_count",
        "dataset_manifest_sha256",
        "privacy",
        "manifest_sha256",
    }
    if predictor_type == "model":
        allowed |= {
            "attention_mode",
            "model_identity",
            "model_training_manifest_sha256",
            "seed",
        }
    else:
        allowed.add("rules_identity")
    _require_exact_keys(manifest, field=f"system.{predictor_type}_prediction", allowed=allowed)
    if manifest.get("schema_version") != 1:
        raise EvaluationDataError(f"system {predictor_type} prediction schema_version must be 1")
    if identity["dataset_manifest_sha256"] != dataset_manifest_sha256:
        raise EvaluationDataError(
            f"system {predictor_type} prediction references a different dataset manifest"
        )
    if identity.get("prediction_document_count") != prediction_document_count:
        raise EvaluationDataError(
            f"system {predictor_type} prediction document count does not match the system"
        )
    _require_privacy_flags(
        manifest.get("privacy"),
        field=f"system.{predictor_type}_prediction.privacy",
        required_flags=("contains_paths", "contains_raw_text", "contains_entity_values"),
    )
    return manifest, identity


def _load_calibration_bundle_identity(path: str | Path) -> dict[str, str]:
    # Import lazily so the low-level evaluation package does not impose a
    # calibration dependency on callers that never build system manifests.
    from pii_zh.calibration import CalibrationBundle

    try:
        bundle = CalibrationBundle.from_json(path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        raise EvaluationDataError("calibration bundle is invalid") from exc
    calibration_version = _require_safe_id(
        bundle.calibration_version, field="calibration_bundle.calibration_version"
    )
    return {
        "calibration_version": calibration_version,
        "bundle_sha256": sha256_file(path),
    }


def _verify_target_calibration_application(
    *,
    fused_predictions_path: str | Path,
    calibration_bundle_path: str | Path,
    calibrated_predictions_path: str | Path,
) -> None:
    from pii_zh.calibration import CalibrationBundle, apply_calibration

    try:
        bundle = CalibrationBundle.from_json(calibration_bundle_path)
        fused = load_prediction_jsonl(fused_predictions_path)
        observed = load_prediction_jsonl(calibrated_predictions_path)
        expected = apply_calibration(fused, bundle)
    except EvaluationDataError:
        raise
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        raise EvaluationDataError("target calibration application is invalid") from exc
    _require_prediction_output_matches(
        expected,
        output_path=calibrated_predictions_path,
        field="target calibrated predictions",
        observed=observed,
    )


def _require_prediction_output_matches(
    expected: Sequence[PredictionRecord],
    *,
    output_path: str | Path,
    field: str,
    observed: Sequence[PredictionRecord] | None = None,
) -> None:
    actual = list(observed) if observed is not None else load_prediction_jsonl(output_path)
    if actual != list(expected):
        raise EvaluationDataError(f"{field} do not semantically match deterministic recomputation")
    buffer = StringIO()
    write_prediction_jsonl(expected, buffer)
    if Path(output_path).read_bytes() != buffer.getvalue().encode("utf-8"):
        raise EvaluationDataError(f"{field} do not byte-match deterministic recomputation")


def _verify_system_fusion(
    *,
    model_predictions_path: str | Path,
    rules_predictions_path: str | Path,
    fused_predictions_path: str | Path,
) -> dict[str, str]:
    from pii_zh.fusion import fuse_prediction_records, system_fusion_configuration

    expected = fuse_prediction_records(
        load_prediction_jsonl(rules_predictions_path),
        load_prediction_jsonl(model_predictions_path),
        model_threshold=0.0,
        keep_nested_different_types=False,
    )
    _require_prediction_output_matches(
        expected,
        output_path=fused_predictions_path,
        field="target fused predictions",
    )
    repository_root = Path(__file__).resolve().parents[3]
    module_components = {
        name: sha256_file(repository_root / "src/pii_zh/fusion" / name)
        for name in ("__init__.py", "deterministic.py", "predictions.py")
    }
    module_sha256 = canonical_json_hash(module_components)
    cli_sha256 = sha256_file(repository_root / "scripts/fuse_predictions.py")
    return {
        "module_sha256": module_sha256,
        "cli_sha256": cli_sha256,
        "implementation_sha256": canonical_json_hash(
            {"module_sha256": module_sha256, "cli_sha256": cli_sha256}
        ),
        "configuration_sha256": canonical_json_hash(system_fusion_configuration()),
    }


def _dataset_subset_for_canonical_sha256(
    dataset: Mapping[str, Any], *, canonical_sha256: str
) -> tuple[str, Mapping[str, Any]]:
    subsets = _require_mapping(dataset.get("subsets"), field="dataset.subsets")
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in subsets.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise EvaluationDataError("dataset subset entries must be named objects")
        canonical = _require_mapping(
            value.get("canonical"), field=f"dataset.subsets.{name}.canonical"
        )
        digest = _require_sha256(
            canonical.get("sha256"), field=f"dataset.subsets.{name}.canonical.sha256"
        )
        if digest == canonical_sha256:
            matches.append((name, value))
    if len(matches) != 1:
        raise EvaluationDataError(
            "canonical SHA-256 does not identify exactly one dataset manifest subset"
        )
    return matches[0]


def _system_prediction_components(
    *,
    dataset: Mapping[str, Any],
    dataset_manifest_sha256: str,
    model_training_manifest_path: str | Path,
    model_predictions_path: str | Path,
    model_prediction_manifest_path: str | Path,
    rules_predictions_path: str | Path,
    rules_prediction_manifest_path: str | Path,
    fused_predictions_path: str | Path,
    calibration_bundle_path: str | Path,
    calibration_diagnostics_path: str | Path,
    calibration_fit_gold_path: str | Path,
    calibration_fit_predictions_path: str | Path,
    calibrated_predictions_path: str | Path,
    refinement_audit_path: str | Path,
    refinement_implementation_sha256: str,
    final_predictions_path: str | Path,
    prediction_document_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_manifest, model_identity = _release_component_prediction(
        manifest_path=model_prediction_manifest_path,
        predictions_path=model_predictions_path,
        predictor_type="model",
        dataset_manifest_sha256=dataset_manifest_sha256,
        prediction_document_count=prediction_document_count,
    )
    rules_manifest, rules_identity = _release_component_prediction(
        manifest_path=rules_prediction_manifest_path,
        predictions_path=rules_predictions_path,
        predictor_type="rules",
        dataset_manifest_sha256=dataset_manifest_sha256,
        prediction_document_count=prediction_document_count,
    )

    training, _ = _load_manifest(model_training_manifest_path, kind="model training")
    if training.get("status") != "completed":
        raise EvaluationDataError("model training manifest must have completed status")
    training_sha256 = _require_sha256(
        training.get("manifest_sha256"), field="model_training.manifest_sha256"
    )
    if model_identity["model_training_manifest_sha256"] != training_sha256:
        raise EvaluationDataError(
            "system model prediction references a different model training manifest"
        )
    for field in ("seed", "attention_mode"):
        if model_manifest.get(field) != training.get(field):
            raise EvaluationDataError(
                f"system model prediction {field} does not match model training manifest"
            )

    count = _require_non_negative_int(prediction_document_count, field="prediction_document_count")
    file_counts = {
        "fused": _nonempty_line_count(fused_predictions_path),
        "calibrated": _nonempty_line_count(calibrated_predictions_path),
        "final": _nonempty_line_count(final_predictions_path),
    }
    for name, observed in file_counts.items():
        if observed != count:
            raise EvaluationDataError(
                f"system {name} prediction document count does not match the system"
            )

    model_predictions_sha256 = sha256_file(model_predictions_path)
    rules_predictions_sha256 = sha256_file(rules_predictions_path)
    fused_predictions_sha256 = sha256_file(fused_predictions_path)
    fusion_identity = _verify_system_fusion(
        model_predictions_path=model_predictions_path,
        rules_predictions_path=rules_predictions_path,
        fused_predictions_path=fused_predictions_path,
    )
    calibrated_predictions_sha256 = sha256_file(calibrated_predictions_path)
    final_predictions_sha256 = sha256_file(final_predictions_path)

    calibration = _load_calibration_bundle_identity(calibration_bundle_path)
    diagnostics, _ = _load_manifest(calibration_diagnostics_path, kind="calibration diagnostics")
    if diagnostics.get("manifest_type") != "calibration_diagnostics":
        raise EvaluationDataError(
            "calibration diagnostics manifest_type must be calibration_diagnostics"
        )
    _require_exact_keys(
        diagnostics,
        field="calibration_diagnostics",
        allowed={
            "schema_version",
            "manifest_type",
            "calibration_version",
            "calibration_bundle_sha256",
            "taxonomy_version",
            "inputs",
            "parameters",
            "temperature",
            "confidence_calibration",
            "per_label",
            "privacy",
            "manifest_sha256",
        },
    )
    if diagnostics.get("schema_version") != 1:
        raise EvaluationDataError("calibration diagnostics schema_version must be 1")
    if diagnostics.get("calibration_version") != calibration["calibration_version"]:
        raise EvaluationDataError("calibration diagnostics and bundle versions differ")
    if (
        _require_sha256(
            diagnostics.get("calibration_bundle_sha256"),
            field="calibration_diagnostics.calibration_bundle_sha256",
        )
        != calibration["bundle_sha256"]
    ):
        raise EvaluationDataError("calibration diagnostics do not bind the calibration bundle")
    diagnostics_inputs = _require_mapping(
        diagnostics.get("inputs"), field="calibration_diagnostics.inputs"
    )
    calibration_fit_gold_sha256 = sha256_file(calibration_fit_gold_path)
    calibration_fit_predictions_sha256 = sha256_file(calibration_fit_predictions_path)
    calibration_fit_count = _nonempty_line_count(calibration_fit_gold_path)
    calibration_fit_records = load_prediction_jsonl(calibration_fit_predictions_path)
    if len(calibration_fit_records) != calibration_fit_count:
        raise EvaluationDataError("calibration fit gold and prediction counts differ")
    if (
        _require_sha256(
            diagnostics_inputs.get("predictions_sha256"),
            field="calibration_diagnostics.inputs.predictions_sha256",
        )
        != calibration_fit_predictions_sha256
    ):
        raise EvaluationDataError("calibration diagnostics do not bind fit predictions")
    if (
        _require_non_negative_int(
            diagnostics_inputs.get("document_count"),
            field="calibration_diagnostics.inputs.document_count",
        )
        != calibration_fit_count
    ):
        raise EvaluationDataError("calibration diagnostics fit document count does not match")
    if (
        _require_sha256(
            diagnostics_inputs.get("gold_sha256"),
            field="calibration_diagnostics.inputs.gold_sha256",
        )
        != calibration_fit_gold_sha256
    ):
        raise EvaluationDataError("calibration diagnostics do not bind fit gold")
    _require_privacy_flags(
        diagnostics.get("privacy"),
        field="calibration_diagnostics.privacy",
        required_flags=(
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
        ),
    )
    _verify_target_calibration_application(
        fused_predictions_path=fused_predictions_path,
        calibration_bundle_path=calibration_bundle_path,
        calibrated_predictions_path=calibrated_predictions_path,
    )

    refinement, _ = _load_manifest(refinement_audit_path, kind="refinement audit")
    if refinement.get("manifest_type") != "structured_prediction_refinement":
        raise EvaluationDataError(
            "refinement audit manifest_type must be structured_prediction_refinement"
        )
    _require_exact_keys(
        refinement,
        field="refinement_audit",
        allowed={
            "schema_version",
            "manifest_type",
            "refinement_id",
            "inputs",
            "output",
            "suppressed_by_label",
            "privacy",
            "manifest_sha256",
        },
    )
    if refinement.get("schema_version") != 1:
        raise EvaluationDataError("refinement audit schema_version must be 1")
    refinement_id = _require_safe_id(
        refinement.get("refinement_id"), field="refinement_audit.refinement_id"
    )
    refinement_inputs = _require_mapping(refinement.get("inputs"), field="refinement_audit.inputs")
    refinement_output = _require_mapping(refinement.get("output"), field="refinement_audit.output")
    if (
        _require_sha256(
            refinement_inputs.get("predictions_sha256"),
            field="refinement_audit.inputs.predictions_sha256",
        )
        != calibrated_predictions_sha256
    ):
        raise EvaluationDataError("refinement audit does not bind calibrated predictions")
    if (
        _require_sha256(
            refinement_output.get("predictions_sha256"),
            field="refinement_audit.output.predictions_sha256",
        )
        != final_predictions_sha256
    ):
        raise EvaluationDataError("refinement audit does not bind final predictions")
    refinement_documents_sha256 = _require_sha256(
        refinement_inputs.get("documents_sha256"),
        field="refinement_audit.inputs.documents_sha256",
    )
    _, target_subset = _dataset_subset_for_canonical_sha256(
        dataset, canonical_sha256=refinement_documents_sha256
    )
    if target_subset.get("record_count") != count:
        raise EvaluationDataError("target dataset subset record count does not match the system")
    if (
        _require_non_negative_int(
            refinement_inputs.get("document_count"),
            field="refinement_audit.inputs.document_count",
        )
        != count
    ):
        raise EvaluationDataError("refinement audit document count does not match")
    _require_privacy_flags(
        refinement.get("privacy"),
        field="refinement_audit.privacy",
        required_flags=(
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
        ),
    )

    components = {
        "model_prediction": {
            "manifest_sha256": model_identity["manifest_sha256"],
            "predictions_sha256": model_predictions_sha256,
            "prediction_document_count": count,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "model_training_manifest_sha256": training_sha256,
        },
        "rules_prediction": {
            "manifest_sha256": rules_identity["manifest_sha256"],
            "predictions_sha256": rules_predictions_sha256,
            "prediction_document_count": count,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "rules_identity": rules_identity["rules_identity"],
        },
        "fusion": {
            **fusion_identity,
            "model_prediction_manifest_sha256": model_identity["manifest_sha256"],
            "model_predictions_sha256": model_predictions_sha256,
            "rules_prediction_manifest_sha256": rules_identity["manifest_sha256"],
            "rules_predictions_sha256": rules_predictions_sha256,
            "output_predictions_sha256": fused_predictions_sha256,
            "prediction_document_count": count,
        },
        "calibration_fit": {
            "bundle_sha256": calibration["bundle_sha256"],
            "diagnostics_manifest_sha256": diagnostics["manifest_sha256"],
            "calibration_version": calibration["calibration_version"],
            "gold_sha256": calibration_fit_gold_sha256,
            "fused_predictions_sha256": calibration_fit_predictions_sha256,
            "document_count": calibration_fit_count,
        },
        "target_application": {
            "bundle_sha256": calibration["bundle_sha256"],
            "calibration_version": calibration["calibration_version"],
            "input_predictions_sha256": fused_predictions_sha256,
            "output_predictions_sha256": calibrated_predictions_sha256,
            "prediction_document_count": count,
        },
        "refinement": {
            "audit_manifest_sha256": refinement["manifest_sha256"],
            "refinement_id": refinement_id,
            "implementation_sha256": _require_sha256(
                refinement_implementation_sha256, field="refinement_implementation_sha256"
            ),
            "input_predictions_sha256": calibrated_predictions_sha256,
            "output_predictions_sha256": final_predictions_sha256,
            "prediction_document_count": count,
        },
    }
    model_release_identity = {
        "model_training_manifest_sha256": training_sha256,
        "seed": model_manifest["seed"],
        "attention_mode": model_manifest["attention_mode"],
        "model_identity": model_manifest["model_identity"],
    }
    return components, model_release_identity


def build_system_prediction_manifest(
    *,
    predictions_path: str | Path,
    dataset_manifest_path: str | Path,
    model_training_manifest_path: str | Path,
    model_predictions_path: str | Path,
    model_prediction_manifest_path: str | Path,
    rules_predictions_path: str | Path,
    rules_prediction_manifest_path: str | Path,
    fused_predictions_path: str | Path,
    calibration_bundle_path: str | Path,
    calibration_diagnostics_path: str | Path,
    calibration_fit_gold_path: str | Path,
    calibration_fit_predictions_path: str | Path,
    calibrated_predictions_path: str | Path,
    refinement_audit_path: str | Path,
    refinement_implementation_sha256: str,
    prediction_document_count: int,
) -> dict[str, Any]:
    """Build a release-grade manifest for the complete deterministic system.

    Every self-hashed component manifest is verified and every intermediate
    output is cross-bound by SHA-256 and document count.  The returned manifest
    contains only aggregate counters, stable identifiers, and hashes.
    """

    dataset, manifest = _prediction_base(
        predictions_path=predictions_path,
        dataset_manifest_path=dataset_manifest_path,
        prediction_document_count=prediction_document_count,
    )
    if _nonempty_line_count(predictions_path) != prediction_document_count:
        raise EvaluationDataError("prediction_document_count does not match final predictions")
    dataset_manifest_sha256 = manifest["dataset_manifest_sha256"]
    components, model_release_identity = _system_prediction_components(
        dataset=dataset,
        dataset_manifest_sha256=dataset_manifest_sha256,
        model_training_manifest_path=model_training_manifest_path,
        model_predictions_path=model_predictions_path,
        model_prediction_manifest_path=model_prediction_manifest_path,
        rules_predictions_path=rules_predictions_path,
        rules_prediction_manifest_path=rules_prediction_manifest_path,
        fused_predictions_path=fused_predictions_path,
        calibration_bundle_path=calibration_bundle_path,
        calibration_diagnostics_path=calibration_diagnostics_path,
        calibration_fit_gold_path=calibration_fit_gold_path,
        calibration_fit_predictions_path=calibration_fit_predictions_path,
        calibrated_predictions_path=calibrated_predictions_path,
        refinement_audit_path=refinement_audit_path,
        refinement_implementation_sha256=refinement_implementation_sha256,
        final_predictions_path=predictions_path,
        prediction_document_count=prediction_document_count,
    )
    manifest.update(
        {
            "predictor_type": "system",
            **model_release_identity,
            "components": components,
        }
    )
    manifest["prediction_id"] = "system-" + canonical_json_hash(manifest)[:24]
    manifest["privacy"] = {
        "contains_paths": False,
        "contains_raw_text": False,
        "contains_entity_values": False,
        "contains_document_ids": False,
    }
    return add_manifest_hash(manifest)


def _dataset_identity(
    *,
    manifest_path: str | Path,
    gold_path: str | Path,
    gold_records: Sequence[GoldRecord],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, file_sha256 = _load_manifest(manifest_path, kind="dataset")
    if manifest.get("manifest_type") != "evaluation_dataset":
        raise EvaluationDataError("dataset manifest_type must be evaluation_dataset")

    source_id = _require_safe_id(manifest.get("source_id"), field="dataset.source_id")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, Mapping):
        raise EvaluationDataError("dataset.upstream must be an object")
    locator = _require_safe_locator(upstream.get("source"), field="dataset.upstream.source")
    revision = _require_safe_id(upstream.get("revision"), field="dataset.upstream.revision")
    license_id = _require_safe_id(upstream.get("license"), field="dataset.upstream.license")

    isolation = manifest.get("isolation")
    if not isinstance(isolation, Mapping):
        raise EvaluationDataError("dataset.isolation must be an object")
    if isolation.get("pool") != "evaluation_only" or isolation.get("training_allowed") is not False:
        raise EvaluationDataError("dataset manifest must be frozen evaluation_only data")

    subsets = manifest.get("subsets")
    if not isinstance(subsets, Mapping) or not subsets:
        raise EvaluationDataError("dataset.subsets must be a non-empty object")
    gold_sha256 = sha256_file(gold_path)
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for subset_name, subset_value in subsets.items():
        if not isinstance(subset_name, str) or not isinstance(subset_value, Mapping):
            raise EvaluationDataError("dataset subset entries must be named objects")
        canonical = subset_value.get("canonical")
        if not isinstance(canonical, Mapping):
            raise EvaluationDataError("dataset subset canonical identity must be an object")
        canonical_sha256 = _require_sha256(
            canonical.get("sha256"), field=f"dataset.subsets.{subset_name}.canonical.sha256"
        )
        if canonical_sha256 == gold_sha256:
            matches.append((subset_name, subset_value))
    if len(matches) != 1:
        raise EvaluationDataError(
            "gold SHA-256 does not identify exactly one dataset manifest subset"
        )

    subset_name, subset = matches[0]
    expected_records = subset.get("record_count")
    expected_spans = subset.get("span_count")
    expected_labels = subset.get("label_counts")
    observed_labels = Counter(span.label for record in gold_records for span in record.spans)
    if expected_records != len(gold_records):
        raise EvaluationDataError("dataset manifest record_count does not match gold")
    if expected_spans != sum(len(record.spans) for record in gold_records):
        raise EvaluationDataError("dataset manifest span_count does not match gold")
    if expected_labels != dict(sorted(observed_labels.items())):
        raise EvaluationDataError("dataset manifest label_counts do not match gold")

    logical_sha256 = _require_sha256(
        manifest.get("manifest_sha256"), field="dataset.manifest_sha256"
    )
    return manifest, {
        "source_id": source_id,
        "upstream_source": locator,
        "upstream_revision": revision,
        "license": license_id,
        "subset": subset_name,
        "gold_sha256": gold_sha256,
        "record_count": len(gold_records),
        "span_count": sum(len(record.spans) for record in gold_records),
        "manifest_sha256": logical_sha256,
        "manifest_file_sha256": file_sha256,
        "evaluation_only": True,
    }


def _training_identity(manifest_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, file_sha256 = _load_manifest(manifest_path, kind="model training")
    logical_sha256 = _require_sha256(
        manifest.get("manifest_sha256"), field="model_training.manifest_sha256"
    )
    identity: dict[str, Any] = {
        "manifest_sha256": logical_sha256,
        "manifest_file_sha256": file_sha256,
    }
    for field in ("recipe_sha256", "label_schema_sha256"):
        value = manifest.get(field)
        if value is not None:
            identity[field] = _require_sha256(value, field=f"model_training.{field}")
    for field in ("attention_mode", "code_revision", "fine_tuning", "taxonomy_version"):
        value = manifest.get(field)
        if value is not None:
            identity[field] = _require_safe_id(value, field=f"model_training.{field}")
    seed = manifest.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluationDataError("model_training.seed must be an integer")
        identity["seed"] = seed
    base = manifest.get("base_checkpoint")
    if isinstance(base, Mapping) and base.get("weights_sha256") is not None:
        identity["base_weights_sha256"] = _require_sha256(
            base["weights_sha256"], field="model_training.base_checkpoint.weights_sha256"
        )
    return manifest, identity


def _normalized_model_identity(value: object, *, field: str) -> dict[str, str]:
    model_identity = _require_mapping(value, field=field)
    allowed_model_fields = {
        "architecture_version",
        "config_sha256",
        "identity_sha256",
        "label_schema_sha256",
        "model_type",
        "weights_sha256",
    }
    if set(model_identity) - allowed_model_fields:
        raise EvaluationDataError(f"{field} has unsupported fields")
    normalized_model: dict[str, str] = {}
    for name in ("config_sha256", "identity_sha256", "label_schema_sha256", "weights_sha256"):
        item = model_identity.get(name)
        if item is not None:
            normalized_model[name] = _require_sha256(item, field=f"{field}.{name}")
    for name in ("architecture_version", "model_type"):
        item = model_identity.get(name)
        if item is not None:
            normalized_model[name] = _require_safe_id(item, field=f"{field}.{name}")
    if not {"config_sha256", "identity_sha256", "model_type", "weights_sha256"} <= set(
        normalized_model
    ):
        raise EvaluationDataError(f"{field} is incomplete")
    unsigned_model = dict(normalized_model)
    claimed_model_hash = unsigned_model.pop("identity_sha256")
    if canonical_json_hash(unsigned_model) != claimed_model_hash:
        raise EvaluationDataError(f"{field} content hash does not verify")
    return normalized_model


def _normalized_system_components(
    manifest: Mapping[str, Any], *, prediction_count: int, predictions_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_exact_keys(
        manifest,
        field="system prediction manifest",
        allowed={
            "schema_version",
            "manifest_type",
            "prediction_id",
            "predictor_type",
            "predictions_sha256",
            "prediction_document_count",
            "dataset_manifest_sha256",
            "model_training_manifest_sha256",
            "seed",
            "attention_mode",
            "model_identity",
            "components",
            "privacy",
            "manifest_sha256",
        },
    )
    if manifest.get("schema_version") != 1:
        raise EvaluationDataError("system prediction schema_version must be 1")
    _require_privacy_flags(
        manifest.get("privacy"),
        field="system prediction privacy",
        required_flags=(
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
        ),
    )
    dataset_sha256 = _require_sha256(
        manifest.get("dataset_manifest_sha256"),
        field="prediction.dataset_manifest_sha256",
    )
    training_sha256 = _require_sha256(
        manifest.get("model_training_manifest_sha256"),
        field="prediction.model_training_manifest_sha256",
    )
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationDataError("prediction.seed must be an integer")
    attention_mode = _require_safe_id(
        manifest.get("attention_mode"), field="prediction.attention_mode"
    )
    model_identity = _normalized_model_identity(
        manifest.get("model_identity"), field="prediction.model_identity"
    )

    components = _require_mapping(manifest.get("components"), field="prediction.components")
    _require_exact_keys(
        components,
        field="prediction.components",
        allowed={
            "model_prediction",
            "rules_prediction",
            "fusion",
            "calibration_fit",
            "target_application",
            "refinement",
        },
    )
    model = _require_mapping(
        components.get("model_prediction"), field="prediction.components.model_prediction"
    )
    _require_exact_keys(
        model,
        field="prediction.components.model_prediction",
        allowed={
            "manifest_sha256",
            "predictions_sha256",
            "prediction_document_count",
            "dataset_manifest_sha256",
            "model_training_manifest_sha256",
        },
    )
    normalized_model = {
        "manifest_sha256": _require_sha256(
            model.get("manifest_sha256"),
            field="prediction.components.model_prediction.manifest_sha256",
        ),
        "predictions_sha256": _require_sha256(
            model.get("predictions_sha256"),
            field="prediction.components.model_prediction.predictions_sha256",
        ),
        "prediction_document_count": _require_non_negative_int(
            model.get("prediction_document_count"),
            field="prediction.components.model_prediction.prediction_document_count",
        ),
        "dataset_manifest_sha256": _require_sha256(
            model.get("dataset_manifest_sha256"),
            field="prediction.components.model_prediction.dataset_manifest_sha256",
        ),
        "model_training_manifest_sha256": _require_sha256(
            model.get("model_training_manifest_sha256"),
            field="prediction.components.model_prediction.model_training_manifest_sha256",
        ),
    }

    rules = _require_mapping(
        components.get("rules_prediction"), field="prediction.components.rules_prediction"
    )
    _require_exact_keys(
        rules,
        field="prediction.components.rules_prediction",
        allowed={
            "manifest_sha256",
            "predictions_sha256",
            "prediction_document_count",
            "dataset_manifest_sha256",
            "rules_identity",
        },
    )
    normalized_rules = {
        "manifest_sha256": _require_sha256(
            rules.get("manifest_sha256"),
            field="prediction.components.rules_prediction.manifest_sha256",
        ),
        "predictions_sha256": _require_sha256(
            rules.get("predictions_sha256"),
            field="prediction.components.rules_prediction.predictions_sha256",
        ),
        "prediction_document_count": _require_non_negative_int(
            rules.get("prediction_document_count"),
            field="prediction.components.rules_prediction.prediction_document_count",
        ),
        "dataset_manifest_sha256": _require_sha256(
            rules.get("dataset_manifest_sha256"),
            field="prediction.components.rules_prediction.dataset_manifest_sha256",
        ),
    }
    rules_component_identity = _require_mapping(
        rules.get("rules_identity"),
        field="prediction.components.rules_prediction.rules_identity",
    )
    _require_exact_keys(
        rules_component_identity,
        field="prediction.components.rules_prediction.rules_identity",
        allowed={"ruleset_id", "implementation_sha256", "configuration_sha256"},
    )
    normalized_rules["rules_identity"] = {
        "ruleset_id": _require_safe_id(
            rules_component_identity.get("ruleset_id"),
            field="prediction.components.rules_prediction.rules_identity.ruleset_id",
        ),
        "implementation_sha256": _require_sha256(
            rules_component_identity.get("implementation_sha256"),
            field=("prediction.components.rules_prediction.rules_identity.implementation_sha256"),
        ),
        "configuration_sha256": _require_sha256(
            rules_component_identity.get("configuration_sha256"),
            field=("prediction.components.rules_prediction.rules_identity.configuration_sha256"),
        ),
    }

    fusion = _require_mapping(components.get("fusion"), field="prediction.components.fusion")
    fusion_fields = {
        "implementation_sha256",
        "module_sha256",
        "cli_sha256",
        "configuration_sha256",
        "model_prediction_manifest_sha256",
        "model_predictions_sha256",
        "rules_prediction_manifest_sha256",
        "rules_predictions_sha256",
        "output_predictions_sha256",
        "prediction_document_count",
    }
    _require_exact_keys(fusion, field="prediction.components.fusion", allowed=fusion_fields)
    normalized_fusion: dict[str, str | int] = {
        name: _require_sha256(fusion.get(name), field=f"prediction.components.fusion.{name}")
        for name in fusion_fields - {"prediction_document_count"}
    }
    normalized_fusion["prediction_document_count"] = _require_non_negative_int(
        fusion.get("prediction_document_count"),
        field="prediction.components.fusion.prediction_document_count",
    )
    if normalized_fusion["implementation_sha256"] != canonical_json_hash(
        {
            "module_sha256": normalized_fusion["module_sha256"],
            "cli_sha256": normalized_fusion["cli_sha256"],
        }
    ):
        raise EvaluationDataError("system fusion implementation identity does not verify")
    from pii_zh.fusion import system_fusion_configuration

    if normalized_fusion["configuration_sha256"] != canonical_json_hash(
        system_fusion_configuration()
    ):
        raise EvaluationDataError("system fusion configuration is not the fixed release config")

    calibration_fit = _require_mapping(
        components.get("calibration_fit"), field="prediction.components.calibration_fit"
    )
    _require_exact_keys(
        calibration_fit,
        field="prediction.components.calibration_fit",
        allowed={
            "bundle_sha256",
            "diagnostics_manifest_sha256",
            "calibration_version",
            "gold_sha256",
            "fused_predictions_sha256",
            "document_count",
        },
    )
    normalized_calibration_fit = {
        "bundle_sha256": _require_sha256(
            calibration_fit.get("bundle_sha256"),
            field="prediction.components.calibration_fit.bundle_sha256",
        ),
        "diagnostics_manifest_sha256": _require_sha256(
            calibration_fit.get("diagnostics_manifest_sha256"),
            field="prediction.components.calibration_fit.diagnostics_manifest_sha256",
        ),
        "calibration_version": _require_safe_id(
            calibration_fit.get("calibration_version"),
            field="prediction.components.calibration_fit.calibration_version",
        ),
        "gold_sha256": _require_sha256(
            calibration_fit.get("gold_sha256"),
            field="prediction.components.calibration_fit.gold_sha256",
        ),
        "fused_predictions_sha256": _require_sha256(
            calibration_fit.get("fused_predictions_sha256"),
            field="prediction.components.calibration_fit.fused_predictions_sha256",
        ),
        "document_count": _require_non_negative_int(
            calibration_fit.get("document_count"),
            field="prediction.components.calibration_fit.document_count",
        ),
    }

    target_application = _require_mapping(
        components.get("target_application"),
        field="prediction.components.target_application",
    )
    _require_exact_keys(
        target_application,
        field="prediction.components.target_application",
        allowed={
            "bundle_sha256",
            "calibration_version",
            "input_predictions_sha256",
            "output_predictions_sha256",
            "prediction_document_count",
        },
    )
    normalized_target_application = {
        "bundle_sha256": _require_sha256(
            target_application.get("bundle_sha256"),
            field="prediction.components.target_application.bundle_sha256",
        ),
        "calibration_version": _require_safe_id(
            target_application.get("calibration_version"),
            field="prediction.components.target_application.calibration_version",
        ),
        "input_predictions_sha256": _require_sha256(
            target_application.get("input_predictions_sha256"),
            field="prediction.components.target_application.input_predictions_sha256",
        ),
        "output_predictions_sha256": _require_sha256(
            target_application.get("output_predictions_sha256"),
            field="prediction.components.target_application.output_predictions_sha256",
        ),
        "prediction_document_count": _require_non_negative_int(
            target_application.get("prediction_document_count"),
            field="prediction.components.target_application.prediction_document_count",
        ),
    }

    refinement = _require_mapping(
        components.get("refinement"), field="prediction.components.refinement"
    )
    _require_exact_keys(
        refinement,
        field="prediction.components.refinement",
        allowed={
            "audit_manifest_sha256",
            "refinement_id",
            "implementation_sha256",
            "input_predictions_sha256",
            "output_predictions_sha256",
            "prediction_document_count",
        },
    )
    normalized_refinement = {
        "audit_manifest_sha256": _require_sha256(
            refinement.get("audit_manifest_sha256"),
            field="prediction.components.refinement.audit_manifest_sha256",
        ),
        "refinement_id": _require_safe_id(
            refinement.get("refinement_id"),
            field="prediction.components.refinement.refinement_id",
        ),
        "implementation_sha256": _require_sha256(
            refinement.get("implementation_sha256"),
            field="prediction.components.refinement.implementation_sha256",
        ),
        "input_predictions_sha256": _require_sha256(
            refinement.get("input_predictions_sha256"),
            field="prediction.components.refinement.input_predictions_sha256",
        ),
        "output_predictions_sha256": _require_sha256(
            refinement.get("output_predictions_sha256"),
            field="prediction.components.refinement.output_predictions_sha256",
        ),
        "prediction_document_count": _require_non_negative_int(
            refinement.get("prediction_document_count"),
            field="prediction.components.refinement.prediction_document_count",
        ),
    }

    counts = {
        normalized_model["prediction_document_count"],
        normalized_rules["prediction_document_count"],
        normalized_fusion["prediction_document_count"],
        normalized_target_application["prediction_document_count"],
        normalized_refinement["prediction_document_count"],
        prediction_count,
    }
    if len(counts) != 1:
        raise EvaluationDataError("system prediction component document counts differ")
    if {
        normalized_model["dataset_manifest_sha256"],
        normalized_rules["dataset_manifest_sha256"],
        dataset_sha256,
    } != {dataset_sha256}:
        raise EvaluationDataError("system prediction component dataset manifests differ")
    if normalized_model["model_training_manifest_sha256"] != training_sha256:
        raise EvaluationDataError("system prediction model training manifests differ")
    if (
        normalized_fusion["model_prediction_manifest_sha256"] != normalized_model["manifest_sha256"]
        or normalized_fusion["model_predictions_sha256"] != normalized_model["predictions_sha256"]
        or normalized_fusion["rules_prediction_manifest_sha256"]
        != normalized_rules["manifest_sha256"]
        or normalized_fusion["rules_predictions_sha256"] != normalized_rules["predictions_sha256"]
    ):
        raise EvaluationDataError("system fusion inputs do not match prediction components")
    if (
        normalized_target_application["input_predictions_sha256"]
        != normalized_fusion["output_predictions_sha256"]
    ):
        raise EvaluationDataError("system target application input does not match fusion output")
    if (
        normalized_calibration_fit["bundle_sha256"]
        != normalized_target_application["bundle_sha256"]
        or normalized_calibration_fit["calibration_version"]
        != normalized_target_application["calibration_version"]
    ):
        raise EvaluationDataError("system calibration fit and target application bundle differ")
    if (
        normalized_refinement["input_predictions_sha256"]
        != normalized_target_application["output_predictions_sha256"]
    ):
        raise EvaluationDataError(
            "system refinement input does not match target calibration output"
        )
    if normalized_refinement["output_predictions_sha256"] != predictions_sha256:
        raise EvaluationDataError("system refinement output does not match final predictions")

    normalized_components = {
        "model_prediction": normalized_model,
        "rules_prediction": normalized_rules,
        "fusion": normalized_fusion,
        "calibration_fit": normalized_calibration_fit,
        "target_application": normalized_target_application,
        "refinement": normalized_refinement,
    }
    release_identity = {
        "model_training_manifest_sha256": training_sha256,
        "seed": seed,
        "attention_mode": attention_mode,
        "model_identity": model_identity,
    }
    return normalized_components, release_identity


def _prediction_identity(
    manifest_path: str | Path,
    predictions_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, file_sha256 = _load_manifest(manifest_path, kind="prediction")
    if manifest.get("manifest_type") != "prediction":
        raise EvaluationDataError("prediction manifest_type must be prediction")
    expected_predictions_sha256 = _require_sha256(
        manifest.get("predictions_sha256"), field="prediction.predictions_sha256"
    )
    observed_predictions_sha256 = sha256_file(predictions_path)
    if expected_predictions_sha256 != observed_predictions_sha256:
        raise EvaluationDataError("prediction manifest does not match predictions SHA-256")
    prediction_id = _require_safe_id(
        manifest.get("prediction_id"), field="prediction.prediction_id"
    )
    dataset_sha256 = _require_sha256(
        manifest.get("dataset_manifest_sha256"), field="prediction.dataset_manifest_sha256"
    )
    predictor_type = manifest.get("predictor_type", "model")
    if predictor_type not in {"model", "rules", "system"}:
        raise EvaluationDataError("prediction.predictor_type must be model, rules, or system")
    identity: dict[str, Any] = {
        "prediction_id": prediction_id,
        "predictor_type": predictor_type,
        "predictions_sha256": observed_predictions_sha256,
        "manifest_sha256": _require_sha256(
            manifest.get("manifest_sha256"), field="prediction.manifest_sha256"
        ),
        "manifest_file_sha256": file_sha256,
        "dataset_manifest_sha256": dataset_sha256,
    }
    prediction_count = manifest.get("prediction_document_count")
    if prediction_count is not None:
        prediction_count = _require_non_negative_int(
            prediction_count, field="prediction.prediction_document_count"
        )
        if prediction_count != _nonempty_line_count(predictions_path):
            raise EvaluationDataError(
                "prediction.prediction_document_count does not match predictions"
            )
        identity["prediction_document_count"] = prediction_count

    if predictor_type == "rules":
        forbidden = {
            "attention_mode",
            "model_identity",
            "model_training_manifest_sha256",
            "seed",
        } & set(manifest)
        if forbidden:
            raise EvaluationDataError(
                "rule prediction manifest contains forbidden model identity fields"
            )
        rules_identity = manifest.get("rules_identity")
        if not isinstance(rules_identity, Mapping):
            raise EvaluationDataError("prediction.rules_identity must be an object")
        ruleset_id = _require_safe_id(
            rules_identity.get("ruleset_id"), field="prediction.rules_identity.ruleset_id"
        )
        implementation_sha256 = _require_sha256(
            rules_identity.get("implementation_sha256"),
            field="prediction.rules_identity.implementation_sha256",
        )
        configuration_sha256 = _require_sha256(
            rules_identity.get("configuration_sha256"),
            field="prediction.rules_identity.configuration_sha256",
        )
        identity["rules_identity"] = {
            "ruleset_id": ruleset_id,
            "implementation_sha256": implementation_sha256,
            "configuration_sha256": configuration_sha256,
        }
        return manifest, identity

    if predictor_type == "system":
        if prediction_count is None:
            raise EvaluationDataError(
                "system prediction manifest requires prediction_document_count"
            )
        components, release_identity = _normalized_system_components(
            manifest,
            prediction_count=prediction_count,
            predictions_sha256=observed_predictions_sha256,
        )
        identity.update(release_identity)
        identity["components"] = components
        return manifest, identity

    training_sha256 = _require_sha256(
        manifest.get("model_training_manifest_sha256"),
        field="prediction.model_training_manifest_sha256",
    )
    identity["model_training_manifest_sha256"] = training_sha256
    seed = manifest.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluationDataError("prediction.seed must be an integer")
        identity["seed"] = seed
    attention_mode = manifest.get("attention_mode")
    if attention_mode is not None:
        identity["attention_mode"] = _require_safe_id(
            attention_mode, field="prediction.attention_mode"
        )
    model_identity = manifest.get("model_identity")
    if model_identity is not None:
        identity["model_identity"] = _normalized_model_identity(
            model_identity, field="prediction.model_identity"
        )
    return manifest, identity


def _require_system_training_consistency(
    prediction_manifest: Mapping[str, Any], training_manifest: Mapping[str, Any]
) -> None:
    if prediction_manifest.get("model_training_manifest_sha256") != training_manifest.get(
        "manifest_sha256"
    ):
        raise EvaluationDataError(
            "prediction manifest references a different model training manifest"
        )
    if training_manifest.get("status") != "completed":
        raise EvaluationDataError("system model training manifest must have completed status")
    for field in ("seed", "attention_mode"):
        if prediction_manifest.get(field) != training_manifest.get(field):
            raise EvaluationDataError(
                f"prediction manifest {field} does not match model training manifest"
            )
    model_identity = _normalized_model_identity(
        prediction_manifest.get("model_identity"), field="prediction.model_identity"
    )
    output_artifact = _require_mapping(
        training_manifest.get("output_artifact"), field="model_training.output_artifact"
    )
    output_files = _require_mapping(
        output_artifact.get("files"), field="model_training.output_artifact.files"
    )
    for model_field, artifact_name in (
        ("config_sha256", "config.json"),
        ("weights_sha256", "model.safetensors"),
    ):
        expected = _require_sha256(
            output_files.get(artifact_name),
            field=f"model_training.output_artifact.files.{artifact_name}",
        )
        if model_identity[model_field] != expected:
            raise EvaluationDataError(
                f"system model {model_field} does not match model training output artifact"
            )
    training_label_schema = training_manifest.get("label_schema_sha256")
    if training_label_schema is not None:
        expected_label_schema = _require_sha256(
            training_label_schema, field="model_training.label_schema_sha256"
        )
        if model_identity.get("label_schema_sha256") != expected_label_schema:
            raise EvaluationDataError(
                "system model label schema does not match model training manifest"
            )


def validate_system_prediction_manifest(
    *,
    manifest_path: str | Path,
    predictions_path: str | Path,
    dataset_manifest_path: str | Path,
    model_training_manifest_path: str | Path,
    model_predictions_path: str | Path,
    model_prediction_manifest_path: str | Path,
    rules_predictions_path: str | Path,
    rules_prediction_manifest_path: str | Path,
    fused_predictions_path: str | Path,
    calibration_bundle_path: str | Path,
    calibration_diagnostics_path: str | Path,
    calibration_fit_gold_path: str | Path,
    calibration_fit_predictions_path: str | Path,
    calibrated_predictions_path: str | Path,
    refinement_audit_path: str | Path,
    refinement_implementation_sha256: str,
) -> dict[str, Any]:
    """Rebuild and validate the complete system chain against a stored manifest."""

    manifest, identity = _prediction_identity(manifest_path, predictions_path)
    if manifest.get("predictor_type") != "system":
        raise EvaluationDataError("prediction manifest predictor_type must be system")
    dataset, _ = _load_manifest(dataset_manifest_path, kind="dataset")
    if dataset.get("manifest_type") != "evaluation_dataset":
        raise EvaluationDataError("dataset manifest_type must be evaluation_dataset")
    if manifest["dataset_manifest_sha256"] != dataset["manifest_sha256"]:
        raise EvaluationDataError("system prediction references a different dataset manifest")
    training, _ = _load_manifest(model_training_manifest_path, kind="model training")
    _require_system_training_consistency(manifest, training)
    expected = build_system_prediction_manifest(
        predictions_path=predictions_path,
        prediction_document_count=identity["prediction_document_count"],
        dataset_manifest_path=dataset_manifest_path,
        model_training_manifest_path=model_training_manifest_path,
        model_predictions_path=model_predictions_path,
        model_prediction_manifest_path=model_prediction_manifest_path,
        rules_predictions_path=rules_predictions_path,
        rules_prediction_manifest_path=rules_prediction_manifest_path,
        fused_predictions_path=fused_predictions_path,
        calibration_bundle_path=calibration_bundle_path,
        calibration_diagnostics_path=calibration_diagnostics_path,
        calibration_fit_gold_path=calibration_fit_gold_path,
        calibration_fit_predictions_path=calibration_fit_predictions_path,
        calibrated_predictions_path=calibrated_predictions_path,
        refinement_audit_path=refinement_audit_path,
        refinement_implementation_sha256=refinement_implementation_sha256,
    )
    if manifest != expected:
        raise EvaluationDataError(
            "system prediction manifest does not match the verified component chain"
        )
    return identity


def _evaluation_parameters(value: Mapping[str, Any] | None) -> dict[str, int | float]:
    if value is None:
        return {}
    allowed = {"bootstrap_samples", "confidence", "iou_threshold", "seed"}
    unknown = set(value) - allowed
    if unknown:
        raise EvaluationDataError("evaluation parameters contain unsupported fields")
    result: dict[str, int | float] = {}
    for name, item in value.items():
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise EvaluationDataError(f"evaluation parameter {name} must be numeric")
        result[name] = item
    return result


def _implementation_identity() -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    component_sha256 = {
        name: sha256_file(module_root / name) for name in ("io.py", "metrics.py", "provenance.py")
    }
    return {
        "schema_version": 1,
        "component_sha256": component_sha256,
        "implementation_sha256": canonical_json_hash(component_sha256),
    }


def build_evaluation_provenance(
    *,
    gold_path: str | Path,
    predictions_path: str | Path,
    gold_records: Sequence[GoldRecord],
    dataset_manifest_path: str | Path | None = None,
    prediction_manifest_path: str | Path | None = None,
    model_training_manifest_path: str | Path | None = None,
    release_mode: bool = False,
    evaluation_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and cross-check provenance for one evaluation invocation.

    In release mode, all three manifests are required.  Outside release mode,
    omitted identities remain explicit ``null`` values while the two actual
    input files are still content-addressed.
    """

    missing = [
        name
        for name, value in (
            ("dataset manifest", dataset_manifest_path),
            ("prediction manifest", prediction_manifest_path),
            ("model training manifest", model_training_manifest_path),
        )
        if value is None
    ]
    if release_mode and missing:
        raise EvaluationDataError("release mode requires " + ", ".join(missing))

    dataset_manifest: dict[str, Any] | None = None
    dataset_identity: dict[str, Any] | None = None
    if dataset_manifest_path is not None:
        dataset_manifest, dataset_identity = _dataset_identity(
            manifest_path=dataset_manifest_path,
            gold_path=gold_path,
            gold_records=gold_records,
        )

    training_manifest: dict[str, Any] | None = None
    training_identity: dict[str, Any] | None = None
    if model_training_manifest_path is not None:
        training_manifest, training_identity = _training_identity(model_training_manifest_path)

    prediction_manifest: dict[str, Any] | None = None
    prediction_identity: dict[str, Any] | None = None
    if prediction_manifest_path is not None:
        prediction_manifest, prediction_identity = _prediction_identity(
            prediction_manifest_path, predictions_path
        )

    if prediction_manifest is not None and dataset_manifest is not None:
        if prediction_manifest["dataset_manifest_sha256"] != dataset_manifest["manifest_sha256"]:
            raise EvaluationDataError("prediction manifest references a different dataset manifest")
    if prediction_manifest is not None and training_manifest is not None:
        predictor_type = prediction_manifest.get("predictor_type", "model")
        if predictor_type == "rules":
            raise EvaluationDataError(
                "rule predictions must not reference a model training manifest"
            )
        if (
            prediction_manifest["model_training_manifest_sha256"]
            != training_manifest["manifest_sha256"]
        ):
            raise EvaluationDataError(
                "prediction manifest references a different model training manifest"
            )
        for field in ("seed", "attention_mode"):
            if field in prediction_manifest and prediction_manifest[field] != training_manifest.get(
                field
            ):
                raise EvaluationDataError(
                    f"prediction manifest {field} does not match model training manifest"
                )
        if predictor_type == "system":
            _require_system_training_consistency(prediction_manifest, training_manifest)

    predictions_sha256 = sha256_file(predictions_path)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "mode": "release" if release_mode else "development",
        "gold": {
            "sha256": sha256_file(gold_path),
            "dataset": dataset_identity,
        },
        "predictions": {
            "sha256": predictions_sha256,
            "manifest": prediction_identity,
        },
        "model_training": training_identity,
        "parameters": _evaluation_parameters(evaluation_parameters),
        "implementation": _implementation_identity(),
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
        },
    }
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    return provenance


__all__ = [
    "add_manifest_hash",
    "build_evaluation_provenance",
    "build_model_prediction_manifest",
    "build_rule_prediction_manifest",
    "build_system_prediction_manifest",
    "canonical_json_hash",
    "sha256_file",
    "validate_system_prediction_manifest",
    "verified_manifest_hash",
]
