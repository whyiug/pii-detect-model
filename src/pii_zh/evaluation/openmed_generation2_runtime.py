"""Shared, result-blind runtime for OpenMed Generation-2 closed-eight tracks.

The runtime deliberately separates model execution from track orchestration:

* the bound model ``LocalRecognizer`` is called once for one non-empty batch;
* its raw-text-free candidates are bound to the batch by domain-separated
  SHA-256 digests and exposed through two single-use recognizer views;
* ``model_raw`` passes the precomputed model candidates through the shared
  ``CascadePipeline`` with every score-changing stage disabled;
* ``framework_hybrid`` first lets a real Presidio ``AnalyzerEngine`` run the
  precomputed model recognizer and ``CnCommonRecognizer`` as two separately
  registered recognizers, then uses the same Pipeline pass-through profile.

The exact-only claim applies to the pass-through Pipeline and the bound BIO
window recognizer.  Presidio's own result/overlap behavior remains an upstream,
version-bound part of ``framework_hybrid`` and is not relabeled as exact-only.

No benchmark-specific row, label frequency, result, text, document identifier,
or filesystem path is part of this module's policy identity.  The fixed label
mapping and both Pipeline identities are content hashed so a preregistration
can bind them before any benchmark row is read.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from pii_zh.cascade import (
    CascadeConfig,
    CascadeDetection,
    CascadePipeline,
    EntityRoute,
    PipelineStagePolicy,
)
from pii_zh.presidio import native_framework as native_framework_module
from pii_zh.presidio._compat import (
    PRESIDIO_AVAILABLE,
    LocalRecognizer,
    RecognizerResult,
    make_recognizer_result,
)
from pii_zh.presidio.bio_token_recognizer import BioTokenClassifierRecognizer
from pii_zh.presidio.cn_common_recognizer import CnCommonRecognizer
from pii_zh.presidio.engine import _MISSING_PRESIDIO_MESSAGE
from pii_zh.presidio.native_framework import (
    NATIVE_FRAMEWORK_SCHEMA_VERSION,
    NativePresidioFrameworkRecognizer,
    presidio_analyzer_version,
)

OPENMED_GENERATION2_RUNTIME_SCHEMA_VERSION = "pii-zh.openmed-generation2-runtime.v1"
OPENMED_GENERATION2_MAPPING_SCHEMA_VERSION = "pii-zh.openmed-generation2-closed8-map.v1"
OPENMED_GENERATION2_BINDING_SCHEMA_VERSION = "pii-zh.precomputed-batch-binding.v1"
OPENMED_GENERATION2_PASS_THROUGH_PROFILE_ID = "openmed-generation2-exact-pass-through-v1"
REQUIRED_BIO_DEDUPLICATION_POLICY = "exact_only_v1"

MODEL_RAW_TRACK = "model_raw"
FRAMEWORK_HYBRID_TRACK = "framework_hybrid"
OPENMED_GENERATION2_TRACKS = (MODEL_RAW_TRACK, FRAMEWORK_HYBRID_TRACK)

_TEXT_SHA256_DOMAIN = b"pii-zh.openmed-generation2.text.v1\x00"
_BATCH_SHA256_DOMAIN = b"pii-zh.openmed-generation2.batch.v1\x00"
_MODEL_SOURCE = "model:openmed_generation2_precomputed"
_UNUSED_RULE_SOURCE = "rule:disabled_openmed_generation2_pass_through"


OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO: Mapping[str, str] = MappingProxyType(
    {
        "ADDRESS": "CN_ADDRESS",
        "BANK_CARD_NUMBER": "BANK_CARD_NUMBER",
        "CN_RESIDENT_ID": "CN_ID_CARD",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PASSPORT_NUMBER": "PASSPORT_NUMBER",
        "PERSON_NAME": "PERSON",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "VEHICLE_LICENSE_PLATE": "CN_VEHICLE_LICENSE_PLATE",
    }
)
OPENMED_GENERATION2_CLOSED8_PRESIDIO_TO_BENCHMARK: Mapping[str, str] = MappingProxyType(
    {value: key for key, value in OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO.items()}
)
OPENMED_GENERATION2_CLOSED8_BENCHMARK_LABELS = tuple(
    OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO
)
OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS = tuple(
    OPENMED_GENERATION2_CLOSED8_PRESIDIO_TO_BENCHMARK
)


class OpenMedGeneration2RuntimeError(RuntimeError):
    """Raised when a frozen Generation-2 runtime invariant is violated."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def openmed_generation2_mapping_document() -> dict[str, object]:
    """Return a mutable copy of the complete fixed bidirectional mapping."""

    return {
        "schema_version": OPENMED_GENERATION2_MAPPING_SCHEMA_VERSION,
        "benchmark_to_presidio": dict(
            sorted(OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO.items())
        ),
        "presidio_to_benchmark": dict(
            sorted(OPENMED_GENERATION2_CLOSED8_PRESIDIO_TO_BENCHMARK.items())
        ),
    }


OPENMED_GENERATION2_CLOSED8_MAPPING_SHA256 = _canonical_sha256(
    openmed_generation2_mapping_document()
)


def openmed_generation2_mapping_identity() -> dict[str, object]:
    """Return report-safe mapping content plus its canonical content hash."""

    return {
        **openmed_generation2_mapping_document(),
        "canonical_sha256": OPENMED_GENERATION2_CLOSED8_MAPPING_SHA256,
        "benchmark_label_count": len(OPENMED_GENERATION2_CLOSED8_BENCHMARK_LABELS),
        "presidio_label_count": len(OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS),
        "bijection": True,
    }


def benchmark_label_to_presidio(label: str) -> str:
    """Map exactly one benchmark closed-eight label, rejecting all others."""

    try:
        return OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO[label]
    except (KeyError, TypeError) as exc:
        raise OpenMedGeneration2RuntimeError(
            "label is outside the fixed benchmark closed-eight taxonomy"
        ) from exc


def presidio_label_to_benchmark(label: str) -> str:
    """Map exactly one canonical Presidio closed-eight label, rejecting others."""

    try:
        return OPENMED_GENERATION2_CLOSED8_PRESIDIO_TO_BENCHMARK[label]
    except (KeyError, TypeError) as exc:
        raise OpenMedGeneration2RuntimeError(
            "label is outside the fixed Presidio closed-eight taxonomy"
        ) from exc


def _source_label_to_presidio(label: str) -> str:
    if label in OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO:
        return OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO[label]
    if label in OPENMED_GENERATION2_CLOSED8_PRESIDIO_TO_BENCHMARK:
        return label
    raise OpenMedGeneration2RuntimeError(
        "model recognizer exposed a label outside the fixed closed-eight taxonomy"
    )


def _pass_through_routes() -> tuple[EntityRoute, ...]:
    # This is an evaluation-only pass-through graph.  Every route is marked
    # semantic so construction does not silently re-enable structured
    # validators which the frozen profile explicitly disables.
    return tuple(
        EntityRoute(
            entity_type=entity_type,
            category="semantic",
            rule_enabled=False,
            model_enabled=True,
            rule_threshold=0.0,
            model_threshold=0.0,
        )
        for entity_type in OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS
    )


OPENMED_GENERATION2_PASS_THROUGH_CONFIG = CascadeConfig(
    profile_version=OPENMED_GENERATION2_PASS_THROUGH_PROFILE_ID,
    mode="model-only",
    routes=_pass_through_routes(),
    rule_source=_UNUSED_RULE_SOURCE,
    model_source=_MODEL_SOURCE,
    keep_nested_different_types=True,
)
OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY = PipelineStagePolicy(
    policy_id=OPENMED_GENERATION2_PASS_THROUGH_PROFILE_ID,
    purpose="evaluation_ablation",
    context_enabled=False,
    context_required=False,
    validators_enabled=False,
    thresholds_enabled=False,
    conflict_policy="exact_duplicates_only",
    rule_branch_entities=None,
    model_branch_entities=OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS,
)


def _route_document(route: EntityRoute) -> dict[str, object]:
    return {
        "entity_type": route.entity_type,
        "category": route.category,
        "rule_enabled": route.rule_enabled,
        "model_enabled": route.model_enabled,
        "validator_label": route.validator_label,
        "model_requires_validator": route.model_requires_validator,
        "rule_threshold": route.rule_threshold,
        "model_threshold": route.model_threshold,
        "short_span_max_chars": route.short_span_max_chars,
        "short_model_threshold": route.short_model_threshold,
    }


def openmed_generation2_pass_through_config_document() -> dict[str, object]:
    config = OPENMED_GENERATION2_PASS_THROUGH_CONFIG
    return {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "routes": [_route_document(route) for route in config.routes],
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
    }


def openmed_generation2_pass_through_stage_policy_document() -> dict[str, object]:
    return OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY.to_dict()


OPENMED_GENERATION2_PASS_THROUGH_CONFIG_SHA256 = _canonical_sha256(
    openmed_generation2_pass_through_config_document()
)
OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY_SHA256 = _canonical_sha256(
    openmed_generation2_pass_through_stage_policy_document()
)


def build_openmed_generation2_pass_through_pipeline(
    recognizer: object,
) -> CascadePipeline:
    """Build the shared model-only, score-neutral, exact-only Pipeline pass."""

    if not callable(getattr(recognizer, "analyze", None)) and not callable(
        getattr(recognizer, "analyze_batch", None)
    ):
        raise TypeError("pass-through recognizer must expose analyze or analyze_batch")
    return CascadePipeline(
        config=OPENMED_GENERATION2_PASS_THROUGH_CONFIG,
        model_recognizer=cast(Any, recognizer),
        context_enhancer=None,
        stage_policy=OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY,
        allow_evaluation_profile=True,
    )


def _module_sha256(module: object) -> str:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise OpenMedGeneration2RuntimeError("runtime module has no hashable source file")
    return hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()


def openmed_generation2_runtime_identity() -> dict[str, object]:
    """Return a preregistration-ready, raw-text/path-free runtime identity."""

    if not PRESIDIO_AVAILABLE:
        raise ImportError(_MISSING_PRESIDIO_MESSAGE)
    framework_identity = {
        "framework": "presidio-analyzer",
        "framework_version": presidio_analyzer_version(),
        "framework_factory_schema_version": NATIVE_FRAMEWORK_SCHEMA_VERSION,
        "framework_factory_file_sha256": _module_sha256(native_framework_module),
        "registered_rule_recognizer": "CnCommonRecognizer",
        "registered_ner_recognizer": "BatchScopedPrecomputedRecognizer",
        "separate_native_registrations": True,
        "cascade_recognizer_factory_used": False,
    }
    shared_pipeline = {
        "config": openmed_generation2_pass_through_config_document(),
        "config_canonical_sha256": OPENMED_GENERATION2_PASS_THROUGH_CONFIG_SHA256,
        "stage_policy": openmed_generation2_pass_through_stage_policy_document(),
        "stage_policy_canonical_sha256": (OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY_SHA256),
    }
    return {
        "schema_version": OPENMED_GENERATION2_RUNTIME_SCHEMA_VERSION,
        "runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mapping": openmed_generation2_mapping_identity(),
        "precomputation": {
            "base_recognizer_batch_calls_per_non_empty_batch": 1,
            "binding_schema_version": OPENMED_GENERATION2_BINDING_SCHEMA_VERSION,
            "text_digest": "sha256_domain_separated_length_prefixed_utf8_v1",
            "batch_digest": "sha256_domain_separated_count_and_ordered_text_digests_v1",
            "raw_text_retained": False,
            "candidate_metadata_retained": False,
            "cleared_after_both_tracks": True,
        },
        "bio_window_deduplication_policy": REQUIRED_BIO_DEDUPLICATION_POLICY,
        "shared_pass_through_pipeline": shared_pipeline,
        "tracks": {
            MODEL_RAW_TRACK: {
                "input": "single_use_precomputed_model_batch",
                "pipeline_config_canonical_sha256": (
                    OPENMED_GENERATION2_PASS_THROUGH_CONFIG_SHA256
                ),
                "pipeline_stage_policy_canonical_sha256": (
                    OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY_SHA256
                ),
                "rules": False,
                "native_presidio_analyzer": False,
                "pre_pipeline_overlap_semantics": REQUIRED_BIO_DEDUPLICATION_POLICY,
                "partial_overlaps_entering_pipeline_preserved": True,
                "exact_only_claim_scope": "model_recognizer_and_pass_through_pipeline",
            },
            FRAMEWORK_HYBRID_TRACK: {
                "input": "native_presidio_two_recognizer_output",
                "pipeline_config_canonical_sha256": (
                    OPENMED_GENERATION2_PASS_THROUGH_CONFIG_SHA256
                ),
                "pipeline_stage_policy_canonical_sha256": (
                    OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY_SHA256
                ),
                "rules": True,
                "native_presidio_analyzer": True,
                "pre_pipeline_overlap_semantics": (
                    "presidio_analyzer_native_version_bound_result_semantics"
                ),
                "native_presidio_overlap_semantics_result_determining": True,
                "exact_only_claim_scope": "post_presidio_pass_through_pipeline_only",
                "native_framework": framework_identity,
            },
        },
    }


def _text_sha256(text: str) -> str:
    encoded = text.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_TEXT_SHA256_DOMAIN)
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)
    return digest.hexdigest()


def _batch_sha256(text_digests: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_BATCH_SHA256_DOMAIN)
    digest.update(len(text_digests).to_bytes(8, byteorder="big", signed=False))
    for index, value in enumerate(text_digests):
        if len(value) != 64:
            raise OpenMedGeneration2RuntimeError("invalid bound text digest")
        digest.update(index.to_bytes(8, byteorder="big", signed=False))
        try:
            digest.update(bytes.fromhex(value))
        except ValueError as exc:
            raise OpenMedGeneration2RuntimeError("invalid bound text digest") from exc
    return digest.hexdigest()


def _texts(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("texts must be a sequence of strings")
    texts = tuple(value)
    if not texts:
        raise ValueError("Generation-2 batches must be non-empty")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain strings")
    return texts


def _sequence(value: object, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise OpenMedGeneration2RuntimeError(f"{field} must be a sequence")
    try:
        return list(cast(Sequence[Any], value))
    except TypeError as exc:
        raise OpenMedGeneration2RuntimeError(f"{field} must be a sequence") from exc


def _value(candidate: object, *names: str, default: object = None) -> object:
    if isinstance(candidate, Mapping):
        for name in names:
            if name in candidate:
                return candidate[name]
        return default
    for name in names:
        if hasattr(candidate, name):
            return getattr(candidate, name)
    return default


@dataclass(frozen=True, slots=True)
class _CanonicalCandidate:
    entity_type: str
    start: int
    end: int
    score: float


def _canonical_candidate(candidate: object, *, text_length: int) -> _CanonicalCandidate:
    raw_label = _value(candidate, "entity_type", "label")
    if not isinstance(raw_label, str):
        raise OpenMedGeneration2RuntimeError("model candidate label must be a string")
    entity_type = _source_label_to_presidio(raw_label)
    start = _value(candidate, "start")
    end = _value(candidate, "end")
    score = _value(candidate, "score", default=0.0)
    if isinstance(start, bool) or not isinstance(start, Integral):
        raise OpenMedGeneration2RuntimeError("model candidate start must be an integer")
    if isinstance(end, bool) or not isinstance(end, Integral):
        raise OpenMedGeneration2RuntimeError("model candidate end must be an integer")
    normalized_start = int(start)
    normalized_end = int(end)
    if not 0 <= normalized_start < normalized_end <= text_length:
        raise OpenMedGeneration2RuntimeError("model candidate lies outside its bound text")
    if isinstance(score, bool) or not isinstance(score, Real):
        raise OpenMedGeneration2RuntimeError("model candidate score must be numeric")
    normalized_score = float(score)
    if not math.isfinite(normalized_score) or not 0.0 <= normalized_score <= 1.0:
        raise OpenMedGeneration2RuntimeError(
            "model candidate score must be finite and between zero and one"
        )
    return _CanonicalCandidate(
        entity_type=entity_type,
        start=normalized_start,
        end=normalized_end,
        score=normalized_score,
    )


class _PrecomputedBatchStore:
    """Raw-text-free candidate storage owned by one synchronous batch."""

    def __init__(
        self,
        *,
        texts: Sequence[str],
        candidates: Sequence[Sequence[_CanonicalCandidate]],
    ) -> None:
        text_digests = tuple(_text_sha256(text) for text in texts)
        normalized_candidates = tuple(tuple(group) for group in candidates)
        if len(text_digests) != len(normalized_candidates):
            raise OpenMedGeneration2RuntimeError(
                "precomputed candidate count does not match the bound text count"
            )
        self._text_digests = text_digests
        self._batch_digest = _batch_sha256(text_digests)
        self._candidates = normalized_candidates
        self._closed = False

    @property
    def count(self) -> int:
        return len(self._text_digests)

    @property
    def candidate_count(self) -> int:
        return sum(len(group) for group in self._candidates)

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise OpenMedGeneration2RuntimeError("precomputed batch scope is closed")

    def batch(self, texts: Sequence[str]) -> tuple[tuple[_CanonicalCandidate, ...], ...]:
        self._require_open()
        values = tuple(texts)
        if len(values) != len(self._text_digests):
            raise OpenMedGeneration2RuntimeError("precomputed batch text count mismatch")
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        observed = tuple(_text_sha256(text) for text in values)
        if _batch_sha256(observed) != self._batch_digest:
            raise OpenMedGeneration2RuntimeError("precomputed batch text digest mismatch")
        return self._candidates

    def item(self, index: int, text: str) -> tuple[_CanonicalCandidate, ...]:
        self._require_open()
        if index >= len(self._text_digests):
            raise OpenMedGeneration2RuntimeError("precomputed batch was over-consumed")
        if _text_sha256(text) != self._text_digests[index]:
            raise OpenMedGeneration2RuntimeError("precomputed item text digest mismatch")
        return self._candidates[index]

    def clear(self) -> None:
        self._text_digests = ()
        self._batch_digest = ""
        self._candidates = ()
        self._closed = True


_ConsumptionMode = Literal["batch_once", "ordered_items_once"]


class BatchScopedPrecomputedRecognizer(LocalRecognizer):
    """Single-use, hash-bound native Presidio view over one candidate batch."""

    def __init__(
        self,
        *,
        store: _PrecomputedBatchStore,
        consumption_mode: _ConsumptionMode,
        name: str,
    ) -> None:
        if consumption_mode not in {"batch_once", "ordered_items_once"}:
            raise ValueError("invalid precomputed recognizer consumption mode")
        self._store: _PrecomputedBatchStore | None = store
        self._consumption_mode = consumption_mode
        self._batch_consumed = False
        self._item_cursor = 0
        super().__init__(
            supported_entities=list(OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS),
            name=name,
            supported_language="zh",
            version="1.0.0",
            context=[],
        )

    def load(self) -> None:
        return None

    def _active_store(self) -> _PrecomputedBatchStore:
        if self._store is None:
            raise OpenMedGeneration2RuntimeError("precomputed recognizer view is closed")
        return self._store

    @staticmethod
    def _filter(
        candidates: Sequence[_CanonicalCandidate], entities: Sequence[str] | None
    ) -> list[RecognizerResult]:
        requested = (
            frozenset(OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS)
            if entities is None
            else frozenset(entities)
        )
        if not requested <= frozenset(OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS):
            raise OpenMedGeneration2RuntimeError(
                "precomputed recognizer received a non-closed-eight entity filter"
            )
        return [
            make_recognizer_result(
                entity_type=candidate.entity_type,
                start=candidate.start,
                end=candidate.end,
                score=candidate.score,
                metadata={
                    "source": _MODEL_SOURCE,
                    "binding_schema_version": OPENMED_GENERATION2_BINDING_SCHEMA_VERSION,
                    "raw_text_retained": False,
                },
            )
            for candidate in candidates
            if candidate.entity_type in requested
        ]

    def analyze(
        self,
        text: str,
        entities: Sequence[str] | None,
        nlp_artifacts: Any | None = None,
    ) -> list[RecognizerResult]:
        del nlp_artifacts
        if self._consumption_mode != "ordered_items_once":
            raise OpenMedGeneration2RuntimeError(
                "batch-only precomputed recognizer cannot consume individual texts"
            )
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        store = self._active_store()
        candidates = store.item(self._item_cursor, text)
        self._item_cursor += 1
        return self._filter(candidates, entities)

    def analyze_batch(
        self,
        texts: Sequence[str],
        entities: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> list[list[RecognizerResult]]:
        if self._consumption_mode != "batch_once":
            raise OpenMedGeneration2RuntimeError(
                "item-only precomputed recognizer cannot consume a whole batch"
            )
        if self._batch_consumed:
            raise OpenMedGeneration2RuntimeError("precomputed batch was consumed more than once")
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = tuple(texts)
        if entities is None or not entities or all(isinstance(item, str) for item in entities):
            filters: list[Sequence[str] | None] = [cast(Sequence[str] | None, entities)] * len(
                values
            )
        else:
            if len(entities) != len(values):
                raise OpenMedGeneration2RuntimeError(
                    "per-text entity filter count does not match the batch"
                )
            filters = [cast(Sequence[str], item) for item in entities]
        candidates = self._active_store().batch(values)
        self._batch_consumed = True
        return [
            self._filter(group, entity_filter)
            for group, entity_filter in zip(candidates, filters, strict=True)
        ]

    def assert_fully_consumed(self) -> None:
        store = self._active_store()
        if self._consumption_mode == "batch_once" and not self._batch_consumed:
            raise OpenMedGeneration2RuntimeError("precomputed batch was not consumed")
        if self._consumption_mode == "ordered_items_once" and self._item_cursor != store.count:
            raise OpenMedGeneration2RuntimeError(
                "precomputed item count does not match the bound batch"
            )

    def close(self) -> None:
        self._store = None
        self._batch_consumed = False
        self._item_cursor = 0


@dataclass(frozen=True, slots=True)
class OpenMedGeneration2Detection:
    """Benchmark-facing, raw-text-free closed-eight detection."""

    entity_type: str
    start: int
    end: int
    score: float
    source: str
    sources: tuple[str, ...]
    decision_process: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.entity_type not in OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO:
            raise OpenMedGeneration2RuntimeError(
                "output detection is outside benchmark closed-eight"
            )
        if not 0 <= self.start < self.end:
            raise OpenMedGeneration2RuntimeError("output detection has invalid offsets")

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
            "sources": list(self.sources),
            "decision_process": list(self.decision_process),
        }


OpenMedGeneration2BatchPredictions = dict[str, tuple[tuple[OpenMedGeneration2Detection, ...], ...]]


def _benchmark_detections(
    batches: Sequence[Sequence[CascadeDetection]],
) -> tuple[tuple[OpenMedGeneration2Detection, ...], ...]:
    return tuple(
        tuple(
            OpenMedGeneration2Detection(
                entity_type=presidio_label_to_benchmark(item.entity_type),
                start=item.start,
                end=item.end,
                score=item.score,
                source=item.source,
                sources=item.sources,
                decision_process=item.decision_process,
            )
            for item in batch
        )
        for batch in batches
    )


def _recognizer_supported_entities(recognizer: LocalRecognizer) -> tuple[str, ...]:
    getter = getattr(recognizer, "get_supported_entities", None)
    raw_entities = getter() if callable(getter) else getattr(recognizer, "supported_entities", None)
    if isinstance(raw_entities, (str, bytes)) or not isinstance(raw_entities, Sequence):
        raise OpenMedGeneration2RuntimeError("model recognizer must declare supported_entities")
    entities = tuple(raw_entities)
    if not entities or any(not isinstance(entity, str) for entity in entities):
        raise OpenMedGeneration2RuntimeError(
            "model recognizer supported_entities must contain strings"
        )
    canonical: set[str] = set()
    for entity in entities:
        mapped = _source_label_to_presidio(entity)
        if mapped in canonical:
            raise OpenMedGeneration2RuntimeError(
                "model recognizer declares duplicate closed-eight label aliases"
            )
        canonical.add(mapped)
    return tuple(sorted(entities))


class OpenMedGeneration2Runtime:
    """Run the two frozen Generation-2 tracks from one model batch call."""

    def __init__(self, *, model_recognizer: LocalRecognizer) -> None:
        if not PRESIDIO_AVAILABLE:
            raise ImportError(_MISSING_PRESIDIO_MESSAGE)
        if not isinstance(model_recognizer, LocalRecognizer):
            raise TypeError("model_recognizer must be a native Presidio LocalRecognizer")
        if getattr(model_recognizer, "supported_language", None) != "zh":
            raise ValueError("model_recognizer must support language='zh'")
        if not callable(getattr(model_recognizer, "analyze_batch", None)):
            raise TypeError("model_recognizer must expose one batch inference method")
        if (
            isinstance(model_recognizer, BioTokenClassifierRecognizer)
            and getattr(model_recognizer, "deduplication_policy", None)
            != REQUIRED_BIO_DEDUPLICATION_POLICY
        ):
            raise ValueError(
                "Generation-2 BIO recognizers must use deduplication_policy='exact_only_v1'"
            )
        declared_policy = getattr(model_recognizer, "deduplication_policy", None)
        if declared_policy is not None and declared_policy != REQUIRED_BIO_DEDUPLICATION_POLICY:
            raise ValueError(
                "Generation-2 model recognizer declared a non-exact deduplication policy"
            )
        self._model_recognizer = model_recognizer
        self._source_entities = _recognizer_supported_entities(model_recognizer)
        self._source_presidio_entities = frozenset(
            _source_label_to_presidio(entity) for entity in self._source_entities
        )
        self._rule_recognizer = CnCommonRecognizer()
        self._active_store: _PrecomputedBatchStore | None = None

    @property
    def active_precomputed_text_count(self) -> int:
        """Return zero outside a synchronous call; raw text is never retained."""

        return 0

    @property
    def active_precomputed_candidate_count(self) -> int:
        return 0 if self._active_store is None else self._active_store.candidate_count

    def runtime_identity(self) -> dict[str, object]:
        return openmed_generation2_runtime_identity()

    def _precompute(self, texts: tuple[str, ...]) -> tuple[tuple[_CanonicalCandidate, ...], ...]:
        method = cast(Any, self._model_recognizer).analyze_batch
        try:
            raw_batches = method(texts=texts, entities=self._source_entities)
        except Exception:
            raise OpenMedGeneration2RuntimeError("base model batch inference failed") from None
        batches = _sequence(raw_batches, field="base model batch output")
        if len(batches) != len(texts):
            raise OpenMedGeneration2RuntimeError(
                "base model batch output count does not match input count"
            )
        normalized: list[tuple[_CanonicalCandidate, ...]] = []
        for text, raw_batch in zip(texts, batches, strict=True):
            candidates = _sequence(raw_batch, field="base model batch item")
            document_candidates: list[_CanonicalCandidate] = []
            for candidate in candidates:
                normalized_candidate = _canonical_candidate(
                    candidate,
                    text_length=len(text),
                )
                if normalized_candidate.entity_type not in self._source_presidio_entities:
                    raise OpenMedGeneration2RuntimeError(
                        "model candidate label was not declared by the recognizer"
                    )
                document_candidates.append(normalized_candidate)
            normalized.append(tuple(document_candidates))
        return tuple(normalized)

    def predict_batch(self, texts: Sequence[str]) -> OpenMedGeneration2BatchPredictions:
        """Return both tracks, calling the bound model recognizer exactly once."""

        if self._active_store is not None:
            raise OpenMedGeneration2RuntimeError("Generation-2 runtime is not re-entrant")
        values = _texts(texts)
        candidates = self._precompute(values)
        store = _PrecomputedBatchStore(texts=values, candidates=candidates)
        raw_view = BatchScopedPrecomputedRecognizer(
            store=store,
            consumption_mode="batch_once",
            name="OpenMedGeneration2ModelRawPrecomputedRecognizer",
        )
        hybrid_model_view = BatchScopedPrecomputedRecognizer(
            store=store,
            consumption_mode="ordered_items_once",
            name="OpenMedGeneration2HybridPrecomputedRecognizer",
        )
        self._active_store = store
        try:
            raw_pipeline = build_openmed_generation2_pass_through_pipeline(raw_view)
            raw = raw_pipeline.detect_batch(
                values,
                entities=OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS,
            )
            raw_view.assert_fully_consumed()

            native_framework = NativePresidioFrameworkRecognizer(
                ner_recognizer=hybrid_model_view,
                rule_recognizer=self._rule_recognizer,
            )
            actual_native_identity = native_framework.runtime_identity()
            expected_native_identity = cast(
                Mapping[str, object],
                cast(Mapping[str, object], self.runtime_identity()["tracks"])[
                    FRAMEWORK_HYBRID_TRACK
                ],
            )["native_framework"]
            for key in (
                "framework",
                "framework_version",
                "framework_factory_schema_version",
                "framework_factory_file_sha256",
                "registered_rule_recognizer",
                "registered_ner_recognizer",
            ):
                expected_value = cast(Mapping[str, object], expected_native_identity).get(key)
                if actual_native_identity.get(key) != expected_value:
                    raise OpenMedGeneration2RuntimeError(
                        "native Presidio runtime identity changed during batch execution"
                    )
            hybrid_pipeline = build_openmed_generation2_pass_through_pipeline(native_framework)
            hybrid = hybrid_pipeline.detect_batch(
                values,
                entities=OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS,
            )
            hybrid_model_view.assert_fully_consumed()
            return {
                MODEL_RAW_TRACK: _benchmark_detections(raw),
                FRAMEWORK_HYBRID_TRACK: _benchmark_detections(hybrid),
            }
        finally:
            raw_view.close()
            hybrid_model_view.close()
            store.clear()
            self._active_store = None


__all__ = [
    "BatchScopedPrecomputedRecognizer",
    "FRAMEWORK_HYBRID_TRACK",
    "MODEL_RAW_TRACK",
    "OPENMED_GENERATION2_BINDING_SCHEMA_VERSION",
    "OPENMED_GENERATION2_CLOSED8_BENCHMARK_LABELS",
    "OPENMED_GENERATION2_CLOSED8_BENCHMARK_TO_PRESIDIO",
    "OPENMED_GENERATION2_CLOSED8_MAPPING_SHA256",
    "OPENMED_GENERATION2_CLOSED8_PRESIDIO_LABELS",
    "OPENMED_GENERATION2_CLOSED8_PRESIDIO_TO_BENCHMARK",
    "OPENMED_GENERATION2_MAPPING_SCHEMA_VERSION",
    "OPENMED_GENERATION2_PASS_THROUGH_CONFIG",
    "OPENMED_GENERATION2_PASS_THROUGH_CONFIG_SHA256",
    "OPENMED_GENERATION2_PASS_THROUGH_PROFILE_ID",
    "OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY",
    "OPENMED_GENERATION2_PASS_THROUGH_STAGE_POLICY_SHA256",
    "OPENMED_GENERATION2_RUNTIME_SCHEMA_VERSION",
    "OPENMED_GENERATION2_TRACKS",
    "OpenMedGeneration2BatchPredictions",
    "OpenMedGeneration2Detection",
    "OpenMedGeneration2Runtime",
    "OpenMedGeneration2RuntimeError",
    "REQUIRED_BIO_DEDUPLICATION_POLICY",
    "benchmark_label_to_presidio",
    "build_openmed_generation2_pass_through_pipeline",
    "openmed_generation2_mapping_document",
    "openmed_generation2_mapping_identity",
    "openmed_generation2_pass_through_config_document",
    "openmed_generation2_pass_through_stage_policy_document",
    "openmed_generation2_runtime_identity",
    "presidio_label_to_benchmark",
]
