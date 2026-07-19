"""Evaluation-only CLUENER compatibility profile for legacy service replay.

The legacy benchmark contains ten semantic NER labels, including labels such
as books, games, movies, and scenic locations which are not PII.  This module
keeps those labels in a deliberately non-default namespace and runs them
through :class:`CascadePipeline`.  Importing the module does not change the
production route table and cannot make the profile selectable by the service.

The recognizer injected here is expected to be the same Presidio-backed
recognizer used by the candidate service.  The profile changes only the
evaluation label allow-list; it must not introduce an alternate prediction
union, evaluator-only overlap resolver, or production output type.
"""

from __future__ import annotations

from .config import CascadeConfig
from .pipeline import CascadePipeline
from .routing import EntityRoute, default_routes
from .stages import PipelineStagePolicy

LEGACY_CLUENER_PROFILE_ID = "legacy_cluener_10"
LEGACY_CLUENER_PROFILE_PURPOSE = "evaluation_compatibility"

LEGACY_CLUENER_SOURCE_LABELS = (
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
)

# A separate namespace prevents a compatibility-only label from accidentally
# passing a production PII allow list.  The mapping is one-to-one so company
# and organization, for example, remain distinguishable in strict metrics.
LEGACY_CLUENER_SOURCE_TO_RUNTIME = {
    label: f"LEGACY_CLUENER_{label.upper()}" for label in LEGACY_CLUENER_SOURCE_LABELS
}
LEGACY_CLUENER_RUNTIME_TO_SOURCE = {
    runtime: source for source, runtime in LEGACY_CLUENER_SOURCE_TO_RUNTIME.items()
}

LEGACY_CLUENER_NON_PII_SOURCE_LABELS = frozenset({"book", "game", "movie", "scene"})
LEGACY_CLUENER_NON_PII_RUNTIME_LABELS = frozenset(
    LEGACY_CLUENER_SOURCE_TO_RUNTIME[label]
    for label in LEGACY_CLUENER_NON_PII_SOURCE_LABELS
)

LEGACY_CLUENER_STAGE_POLICY = PipelineStagePolicy(
    policy_id="legacy-cluener-10-native-presidio-pass-through-v1",
    purpose="evaluation_ablation",
    context_enabled=False,
    context_required=False,
    validators_enabled=False,
    thresholds_enabled=False,
    conflict_policy="exact_duplicates_only",
    rule_branch_entities=(),
    model_branch_entities=tuple(LEGACY_CLUENER_SOURCE_TO_RUNTIME.values()),
)


def _legacy_cluener_routes() -> tuple[EntityRoute, ...]:
    """Return the isolated ten-label semantic allow list in source order."""

    return tuple(
        EntityRoute(
            entity_type=LEGACY_CLUENER_SOURCE_TO_RUNTIME[label],
            category="semantic",
            rule_enabled=False,
            model_enabled=True,
        )
        for label in LEGACY_CLUENER_SOURCE_LABELS
    )


def _legacy_cluener_config() -> CascadeConfig:
    """Build the non-default config consumed by the shared cascade runtime."""

    return CascadeConfig(
        profile_version="legacy-cluener-10-compatibility-v1",
        mode="model-only",
        routes=_legacy_cluener_routes(),
        rule_source="rule:disabled-in-native-framework-profile",
        model_source="presidio:native-framework",
        keep_nested_different_types=False,
    )


def build_legacy_cluener_pipeline(
    *,
    native_presidio_recognizer: object,
    allow_evaluation_profile: bool = False,
) -> CascadePipeline:
    """Build the shared runtime only after an explicit evaluation opt-in.

    ``native_presidio_recognizer`` should own the service's rule/NER Presidio
    orchestration.  It is registered as the single model-side recognizer so an
    outer rule branch cannot silently duplicate or replace Presidio conflict
    behavior.
    """

    if allow_evaluation_profile is not True:
        raise ValueError("legacy_cluener_10 requires explicit evaluation authorization")
    # Keep the heavyweight/optional Presidio import out of module import time,
    # while refusing a generic recognizer that could bypass AnalyzerEngine.
    from pii_zh.presidio.engine import _presidio_api
    from pii_zh.presidio.native_framework import NativePresidioFrameworkRecognizer

    if type(native_presidio_recognizer) is not NativePresidioFrameworkRecognizer:
        raise TypeError(
            "legacy_cluener_10 requires the native Presidio framework recognizer"
        )
    engine = getattr(native_presidio_recognizer, "engine", None)
    analyzer_type, _, _ = _presidio_api()
    if type(engine) is not analyzer_type:
        raise TypeError("legacy_cluener_10 requires a real Presidio AnalyzerEngine")
    return CascadePipeline(
        config=_legacy_cluener_config(),
        rule_recognizer=None,
        model_recognizer=native_presidio_recognizer,
        stage_policy=LEGACY_CLUENER_STAGE_POLICY,
        allow_evaluation_profile=True,
    )


def production_pii_core_entities() -> frozenset[str]:
    """Return the unchanged production default allow list for audit/tests."""

    return frozenset(route.entity_type for route in default_routes())


__all__ = [
    "LEGACY_CLUENER_NON_PII_RUNTIME_LABELS",
    "LEGACY_CLUENER_NON_PII_SOURCE_LABELS",
    "LEGACY_CLUENER_PROFILE_ID",
    "LEGACY_CLUENER_PROFILE_PURPOSE",
    "LEGACY_CLUENER_RUNTIME_TO_SOURCE",
    "LEGACY_CLUENER_SOURCE_LABELS",
    "LEGACY_CLUENER_SOURCE_TO_RUNTIME",
    "LEGACY_CLUENER_STAGE_POLICY",
    "build_legacy_cluener_pipeline",
    "production_pii_core_entities",
]
