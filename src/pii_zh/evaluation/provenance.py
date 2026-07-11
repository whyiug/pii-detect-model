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
from pathlib import Path
from typing import Any

from .io import EvaluationDataError, GoldRecord

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
    if predictor_type not in {"model", "rules"}:
        raise EvaluationDataError("prediction.predictor_type must be model or rules")
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
        if (
            isinstance(prediction_count, bool)
            or not isinstance(prediction_count, int)
            or prediction_count < 0
        ):
            raise EvaluationDataError(
                "prediction.prediction_document_count must be a non-negative integer"
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
        if not isinstance(model_identity, Mapping):
            raise EvaluationDataError("prediction.model_identity must be an object")
        allowed_model_fields = {
            "architecture_version",
            "config_sha256",
            "identity_sha256",
            "label_schema_sha256",
            "model_type",
            "weights_sha256",
        }
        if set(model_identity) - allowed_model_fields:
            raise EvaluationDataError("prediction.model_identity has unsupported fields")
        normalized_model: dict[str, str] = {}
        for field in ("config_sha256", "identity_sha256", "label_schema_sha256", "weights_sha256"):
            value = model_identity.get(field)
            if value is not None:
                normalized_model[field] = _require_sha256(
                    value, field=f"prediction.model_identity.{field}"
                )
        for field in ("architecture_version", "model_type"):
            value = model_identity.get(field)
            if value is not None:
                normalized_model[field] = _require_safe_id(
                    value, field=f"prediction.model_identity.{field}"
                )
        if not {"config_sha256", "identity_sha256", "model_type", "weights_sha256"} <= set(
            normalized_model
        ):
            raise EvaluationDataError("prediction.model_identity is incomplete")
        unsigned_model = dict(normalized_model)
        claimed_model_hash = unsigned_model.pop("identity_sha256")
        if canonical_json_hash(unsigned_model) != claimed_model_hash:
            raise EvaluationDataError("prediction.model_identity content hash does not verify")
        identity["model_identity"] = normalized_model
    return manifest, identity


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
        if prediction_manifest.get("predictor_type", "model") != "model":
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
    "canonical_json_hash",
    "sha256_file",
    "verified_manifest_hash",
]
