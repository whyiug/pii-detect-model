"""Versioned service profiles kept separate from historical evaluation profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pii_zh.calibration import CalibrationBundle
from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
from pii_zh.data.validators.cn_vehicle_plate import validate_cn_vehicle_license_plate
from pii_zh.rules.cn_common_v6 import CN_COMMON_V6_SOURCE, CnCommonRulePackV6
from pii_zh.taxonomy import load_presidio_mapping, load_taxonomy

from .community_model import canonical_json_hash, verify_community_model_artifact
from .config import CascadeConfig, CascadeMode
from .pipeline import CascadePipeline
from .routing import community_full24_routes, conservative_v2_routes, default_routes

LEGACY_SERVICE_PROFILE_VERSION = "c1-conservative-v1"
SUCCESSOR_SERVICE_PROFILE_VERSION = "c1-conservative-v2"
COMMUNITY_MODEL_SERVICE_PROFILE_VERSION = "community-model-cascade-v1"
DEFAULT_SERVICE_PROFILE_VERSION = LEGACY_SERVICE_PROFILE_VERSION
SERVICE_PROFILE_VERSIONS = (
    LEGACY_SERVICE_PROFILE_VERSION,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
)
SERVICE_PROFILE_MODE_MATRIX: Mapping[str, tuple[CascadeMode, ...]] = MappingProxyType(
    {
        LEGACY_SERVICE_PROFILE_VERSION: ("rules-only",),
        SUCCESSOR_SERVICE_PROFILE_VERSION: ("rules-only",),
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION: ("model-only", "cascade"),
    }
)


def validate_service_profile_mode(profile_version: str, mode: CascadeMode) -> None:
    """Freeze the public service profile/mode matrix without implicit fallback."""

    try:
        allowed = SERVICE_PROFILE_MODE_MATRIX[profile_version]
    except KeyError as exc:
        raise ValueError(f"unknown service profile: {profile_version}") from exc
    if mode not in allowed:
        rendered = " or ".join(allowed)
        raise ValueError(f"service profile {profile_version} requires {rendered} mode")


def load_service_config(
    profile_version: str = DEFAULT_SERVICE_PROFILE_VERSION,
    *,
    mode: CascadeMode = "rules-only",
) -> CascadeConfig:
    """Build a known service profile and reject unknown or implicit upgrades."""

    validate_service_profile_mode(profile_version, mode)
    if profile_version == LEGACY_SERVICE_PROFILE_VERSION:
        return CascadeConfig(
            profile_version=profile_version,
            mode=mode,
            routes=default_routes(),
            rule_source="rule:cn_common",
        )
    if profile_version == SUCCESSOR_SERVICE_PROFILE_VERSION:
        return CascadeConfig(
            profile_version=profile_version,
            mode=mode,
            routes=conservative_v2_routes(),
            rule_source=CN_COMMON_V6_SOURCE,
        )
    if profile_version == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION:
        return CascadeConfig(
            profile_version=profile_version,
            mode=mode,
            routes=community_full24_routes(),
            rule_source=CN_COMMON_V6_SOURCE,
            model_source="model:pii-zh-24",
        )
    raise ValueError(f"unknown service profile: {profile_version}")


def build_rules_only_service_pipeline(
    profile_version: str = DEFAULT_SERVICE_PROFILE_VERSION,
) -> CascadePipeline:
    """Build an atomically bound rules-only profile without touching v5 globals."""

    config = load_service_config(profile_version, mode="rules-only")
    if profile_version == LEGACY_SERVICE_PROFILE_VERSION:
        return CascadePipeline(config=config)
    return CascadePipeline(
        config=config,
        rule_recognizer=CnCommonRulePackV6(),
        validators={
            "CN_VEHICLE_LICENSE_PLATE": validate_cn_vehicle_license_plate,
        },
    )


def _community_policy_labels() -> tuple[dict[str, str], frozenset[str]]:
    taxonomy = load_taxonomy()
    mapping = load_presidio_mapping()
    raw_to_output = {
        entity.name: mapping.model_to_presidio[entity.name]
        for entity in taxonomy.label_sets["core"]
    }
    return raw_to_output, frozenset(raw_to_output.values())


def _normalize_policy_values(values: Mapping[str, float], *, kind: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{kind} must be a mapping")
    raw_to_output, output_labels = _community_policy_labels()
    normalized: dict[str, float] = {}
    for raw_label, raw_value in values.items():
        if not isinstance(raw_label, str) or not raw_label:
            raise ValueError(f"{kind} keys must be non-empty entity labels")
        if raw_label in raw_to_output:
            label = raw_to_output[raw_label]
        elif raw_label in output_labels:
            label = raw_label
        else:
            raise ValueError(f"{kind} contains a non-core24 label")
        if label in normalized:
            raise ValueError(f"{kind} contains duplicate raw/output aliases for one core24 label")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{kind} values must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{kind} values must be finite")
        normalized[label] = value
    return dict(sorted(normalized.items()))


def _normalize_calibration(
    calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None,
) -> CalibrationBundle | None:
    if calibration is None:
        return None
    if isinstance(calibration, CalibrationBundle):
        bundle = calibration
    elif isinstance(calibration, Mapping):
        bundle = CalibrationBundle.from_dict(calibration)
    elif isinstance(calibration, (str, Path)):
        candidate = Path(calibration).expanduser()
        if candidate.is_symlink():
            raise ValueError("calibration must be a non-symlink local JSON file")
        try:
            path = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("calibration must be an existing local JSON file") from exc
        if not path.is_file():
            raise ValueError("calibration must be an existing local JSON file")
        bundle = CalibrationBundle.from_json(path)
    else:
        raise TypeError("calibration must be a bundle, mapping, local JSON path, or null")
    temperatures = _normalize_policy_values(
        bundle.entity_temperatures,
        kind="calibration entity_temperatures",
    )
    thresholds = _normalize_policy_values(
        bundle.entity_thresholds,
        kind="calibration entity_thresholds",
    )
    return CalibrationBundle(
        global_temperature=bundle.global_temperature,
        entity_temperatures=temperatures,
        entity_thresholds=thresholds,
        default_threshold=bundle.default_threshold,
        model_version=bundle.model_version,
        calibration_version=bundle.calibration_version,
    )


def _normalize_thresholds(
    thresholds: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if thresholds is None:
        return None
    normalized = _normalize_policy_values(thresholds, kind="thresholds")
    if any(not 0.0 <= value <= 1.0 for value in normalized.values()):
        raise ValueError("thresholds must be between zero and one")
    return normalized


def _normalize_allowed_model_labels(
    labels: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if labels is None:
        return None
    if isinstance(labels, (str, bytes)):
        raise TypeError("allowed_model_labels must be a sequence of model labels")
    values = tuple(labels)
    if not values or any(not isinstance(label, str) or not label for label in values):
        raise ValueError("allowed_model_labels must contain model labels")
    if len(set(values)) != len(values):
        raise ValueError("allowed_model_labels must not contain duplicates")
    core_labels = tuple(entity.name for entity in load_taxonomy().label_sets["core"])
    requested = set(values)
    if not requested <= set(core_labels):
        raise ValueError("allowed_model_labels must be a subset of the core-24 model labels")
    return tuple(label for label in core_labels if label in requested)


def build_community_model_service_pipeline(
    model_path: str | Path,
    *,
    mode: CascadeMode = "cascade",
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
    calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    allowed_model_labels: Sequence[str] | None = None,
) -> CascadePipeline:
    """Build the opt-in v6-rules + local 24-class model community profile.

    The checkpoint remains an explicit caller-owned local directory.  The
    shared loader enforces an artifact-bound completed training manifest,
    safetensors-only weights, ``local_files_only=True`` and disabled remote
    code.  A load or inference error is intentionally propagated; this factory
    never silently falls back to the rules-only profile.
    """

    validate_service_profile_mode(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode)
    verified = verify_community_model_artifact(model_path)
    normalized_calibration = _normalize_calibration(calibration)
    normalized_thresholds = _normalize_thresholds(thresholds)
    normalized_allowed_labels = _normalize_allowed_model_labels(allowed_model_labels)
    explicit_policy = calibration is not None or thresholds is not None
    config = load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode=mode)
    if explicit_policy:
        config = replace(
            config,
            routes=community_full24_routes(recognizer_thresholds_authoritative=True),
        )
    model_identity: dict[str, str | int | bool | None] = {
        **verified.identity.to_dict(),
        "calibration_sha256": (
            canonical_json_hash(normalized_calibration.to_dict())
            if normalized_calibration is not None
            else None
        ),
        "thresholds_sha256": (
            canonical_json_hash(normalized_thresholds)
            if normalized_thresholds is not None
            else None
        ),
    }
    if normalized_allowed_labels is not None:
        model_identity["allowed_model_labels_sha256"] = canonical_json_hash(
            normalized_allowed_labels
        )
    loader_options: dict[str, Any] = {
        "config": config,
        "rule_recognizer": CnCommonRulePackV6() if config.uses_rules else None,
        "validators": COMMUNITY_STRUCTURED_VALIDATORS,
        "device": device,
        "micro_batch_size": micro_batch_size,
        "local_files_only": True,
        "model_identity": model_identity,
    }
    if dtype is not None:
        loader_options["dtype"] = dtype
    if normalized_calibration is not None:
        loader_options["calibration"] = normalized_calibration
    if normalized_thresholds is not None:
        loader_options["thresholds"] = normalized_thresholds
    if normalized_allowed_labels is not None:
        loader_options["allowed_model_labels"] = normalized_allowed_labels
    return CascadePipeline.from_pretrained(
        verified.root,
        **loader_options,
    )


__all__ = [
    "COMMUNITY_MODEL_SERVICE_PROFILE_VERSION",
    "DEFAULT_SERVICE_PROFILE_VERSION",
    "LEGACY_SERVICE_PROFILE_VERSION",
    "SERVICE_PROFILE_VERSIONS",
    "SERVICE_PROFILE_MODE_MATRIX",
    "SUCCESSOR_SERVICE_PROFILE_VERSION",
    "build_community_model_service_pipeline",
    "build_rules_only_service_pipeline",
    "load_service_config",
    "validate_service_profile_mode",
]
