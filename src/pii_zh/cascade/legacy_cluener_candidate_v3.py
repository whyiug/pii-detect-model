"""Post-disclosure, service-priority legacy CLUENER candidate v3.

This immutable successor leaves the v2 factory and compatibility profile
untouched.  It reuses the exact v2 ERNIE/native-Presidio loader, then builds a
new shared :class:`CascadePipeline` profile whose ten model routes all enforce
the conservative semantic production threshold of ``0.50``.  Thresholding is
owned by ``EntityRoute.evaluate`` inside the pipeline; evaluators receive only
the already-routed detections.
"""

from __future__ import annotations

from pathlib import Path

from .config import CascadeConfig
from .legacy_cluener import (
    LEGACY_CLUENER_SOURCE_LABELS,
    LEGACY_CLUENER_SOURCE_TO_RUNTIME,
)
from .legacy_cluener_candidate import (
    ERNIE_AUXILIARY_MICRO_BATCH_SIZE,
    build_ernie_auxiliary_native_framework,
)
from .pipeline import CascadePipeline
from .routing import EntityRoute
from .stages import PipelineStagePolicy

LEGACY_CLUENER_V3_PROFILE_ID = "legacy_cluener_10_threshold_050_v3"
LEGACY_CLUENER_V3_MODEL_THRESHOLD = 0.50
LEGACY_CLUENER_V3_STAGE_POLICY = PipelineStagePolicy(
    policy_id="legacy-cluener-10-native-presidio-threshold-050-v3",
    purpose="evaluation_ablation",
    context_enabled=False,
    context_required=False,
    validators_enabled=False,
    thresholds_enabled=True,
    conflict_policy="exact_duplicates_only",
    rule_branch_entities=(),
    model_branch_entities=tuple(LEGACY_CLUENER_SOURCE_TO_RUNTIME.values()),
)


def legacy_cluener_v3_routes() -> tuple[EntityRoute, ...]:
    """Return the ten isolated model routes with one global threshold."""

    return tuple(
        EntityRoute(
            entity_type=LEGACY_CLUENER_SOURCE_TO_RUNTIME[label],
            category="semantic",
            rule_enabled=False,
            model_enabled=True,
            model_threshold=LEGACY_CLUENER_V3_MODEL_THRESHOLD,
        )
        for label in LEGACY_CLUENER_SOURCE_LABELS
    )


def _legacy_cluener_v3_config() -> CascadeConfig:
    return CascadeConfig(
        profile_version=LEGACY_CLUENER_V3_PROFILE_ID,
        mode="model-only",
        routes=legacy_cluener_v3_routes(),
        rule_source="rule:disabled-in-native-framework-profile",
        model_source="presidio:native-framework",
        keep_nested_different_types=False,
    )


def build_thresholded_legacy_cluener_pipeline(
    *,
    native_presidio_recognizer: object,
    allow_evaluation_profile: bool = False,
) -> CascadePipeline:
    """Build v3 only after a real AnalyzerEngine and explicit authorization."""

    if allow_evaluation_profile is not True:
        raise ValueError("legacy CLUENER v3 requires explicit evaluation authorization")
    from pii_zh.presidio.engine import _presidio_api
    from pii_zh.presidio.native_framework import NativePresidioFrameworkRecognizer

    if type(native_presidio_recognizer) is not NativePresidioFrameworkRecognizer:
        raise TypeError("legacy CLUENER v3 requires the native Presidio framework")
    analyzer_type, _, _ = _presidio_api()
    if type(getattr(native_presidio_recognizer, "engine", None)) is not analyzer_type:
        raise TypeError("legacy CLUENER v3 requires a real Presidio AnalyzerEngine")
    return CascadePipeline(
        config=_legacy_cluener_v3_config(),
        rule_recognizer=None,
        model_recognizer=native_presidio_recognizer,
        stage_policy=LEGACY_CLUENER_V3_STAGE_POLICY,
        allow_evaluation_profile=True,
    )


def build_ernie_auxiliary_legacy_pipeline_v3(
    model_path: str | Path,
    *,
    device: str = "cpu",
    micro_batch_size: int = ERNIE_AUXILIARY_MICRO_BATCH_SIZE,
) -> CascadePipeline:
    """Load the v2-pinned framework and apply the v3 routing threshold."""

    framework = build_ernie_auxiliary_native_framework(
        model_path,
        profile="legacy_cluener_10",
        device=device,
        micro_batch_size=micro_batch_size,
    )
    return build_thresholded_legacy_cluener_pipeline(
        native_presidio_recognizer=framework,
        allow_evaluation_profile=True,
    )


__all__ = [
    "LEGACY_CLUENER_V3_MODEL_THRESHOLD",
    "LEGACY_CLUENER_V3_PROFILE_ID",
    "LEGACY_CLUENER_V3_STAGE_POLICY",
    "build_ernie_auxiliary_legacy_pipeline_v3",
    "build_thresholded_legacy_cluener_pipeline",
    "legacy_cluener_v3_routes",
]
