"""Presidio ``LocalRecognizer`` adapter for generic BIO span predictors.

The recognizer is model-architecture independent.  It creates tokenizer-aware
overlapping windows, translates relative offsets back to document offsets,
uses center ownership to prevent double emission, and deterministically removes
high-overlap same-label duplicates across windows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Protocol

from ._compat import LocalRecognizer, RecognizerResult, make_recognizer_result
from .token_chunker import TokenizerAwareChunker, TokenWindow


class BioWindowPredictor(Protocol):
    """Raw-text-free batch predictor accepted by the recognizer."""

    def predict_batch(self, texts: Sequence[str]) -> Sequence[Sequence[Any]]: ...


BioRecognizerDeduplicationPolicy = Literal["high_overlap_v1", "exact_only_v1"]
_DEDUPLICATION_POLICIES = frozenset({"high_overlap_v1", "exact_only_v1"})


@dataclass(frozen=True, slots=True)
class _Candidate:
    entity_type: str
    start: int
    end: int
    score: float
    window_id: int
    metadata: Mapping[str, Any]


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("prediction score must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("prediction score must be finite and between zero and one")
    return result


def _offset(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"prediction {name} must be an integer")
    return int(value)


class BioTokenClassifierRecognizer(LocalRecognizer):
    """Expose a normalized BIO predictor through Presidio's recognizer API."""

    def __init__(
        self,
        *,
        predictor: BioWindowPredictor,
        tokenizer: Any | None = None,
        chunker: TokenizerAwareChunker | None = None,
        supported_entities: Sequence[str],
        default_threshold: float = 0.0,
        max_tokens: int = 512,
        stride_fraction: float = 0.25,
        model_version: str = "unknown-local-model",
        source_name: str = "transformers_bio_token_classifier",
        deduplication_policy: BioRecognizerDeduplicationPolicy = "high_overlap_v1",
        name: str = "BioTokenClassifierRecognizer",
        supported_language: str = "zh",
        version: str = "1.0.0",
    ) -> None:
        entities = list(supported_entities)
        if not entities or any(
            not isinstance(entity, str) or not entity.strip() for entity in entities
        ):
            raise ValueError("supported_entities must contain non-empty strings")
        if len(set(entities)) != len(entities):
            raise ValueError("supported_entities must be unique")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("source_name must be a non-empty string")
        self.predictor = predictor
        self.tokenizer = tokenizer or getattr(predictor, "tokenizer", None)
        self.chunker = chunker
        self.default_threshold = _score(default_threshold)
        self.max_tokens = max_tokens
        self.stride_fraction = stride_fraction
        self.model_version = model_version
        self.source_name = source_name
        if deduplication_policy not in _DEDUPLICATION_POLICIES:
            raise ValueError("deduplication_policy must be high_overlap_v1 or exact_only_v1")
        self.deduplication_policy = deduplication_policy
        self.recognizer_name = name
        self._loaded = chunker is not None
        super().__init__(
            supported_entities=sorted(entities),
            name=name,
            supported_language=supported_language,
            version=version,
            context=[],
        )

    def load(self) -> None:
        if self._loaded:
            return
        if self.tokenizer is None:
            raise RuntimeError("predictor or caller must provide a tokenizer")
        self.chunker = TokenizerAwareChunker(
            self.tokenizer,
            max_tokens=self.max_tokens,
            stride_fraction=self.stride_fraction,
        )
        self._loaded = True

    @staticmethod
    def _raw_value(prediction: Any, *names: str, default: object = None) -> object:
        if isinstance(prediction, Mapping):
            for name in names:
                if name in prediction:
                    return prediction[name]
            return default
        for name in names:
            if hasattr(prediction, name):
                return getattr(prediction, name)
        return default

    def _predict_windows(self, windows: Sequence[TokenWindow]) -> list[Sequence[Any]]:
        batches = list(self.predictor.predict_batch([window.text for window in windows]))
        if len(batches) != len(windows):
            raise ValueError("predictor must return one result sequence per input window")
        normalized: list[Sequence[Any]] = []
        for raw_batch in batches:
            batch: object = raw_batch
            if isinstance(batch, Mapping) and "entities" in batch:
                batch = batch["entities"]
            if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
                raise TypeError("each predictor batch result must be a sequence of spans")
            normalized.append(batch)
        return normalized

    def _candidate(self, prediction: Any, window: TokenWindow) -> _Candidate | None:
        raw_label = self._raw_value(prediction, "entity_type", "entity_group", "entity", "label")
        if not isinstance(raw_label, str) or not raw_label:
            raise ValueError("prediction label must be a non-empty string")
        if raw_label not in self.supported_entities:
            raise ValueError("prediction label is absent from supported_entities")
        score = _score(self._raw_value(prediction, "score", "confidence", default=1.0))
        if score < self.default_threshold:
            return None
        start = _offset("start", self._raw_value(prediction, "start"))
        end = _offset("end", self._raw_value(prediction, "end"))
        raw_metadata = self._raw_value(prediction, "metadata", default={})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("prediction metadata must be a mapping")
        offset_mode = self._raw_value(prediction, "offset_mode", default=None)
        if offset_mode is None:
            offset_mode = raw_metadata.get("offset_mode", "relative")
        if offset_mode == "relative":
            start, end = window.to_global(start, end)
        elif offset_mode == "global":
            if not window.start <= start < end <= window.end:
                raise ValueError("global prediction span lies outside its source window")
        else:
            raise ValueError("prediction offset_mode must be relative or global")
        if not window.owns(start, end):
            return None
        return _Candidate(
            entity_type=raw_label,
            start=start,
            end=end,
            score=score,
            window_id=window.window_id,
            metadata=dict(raw_metadata),
        )

    def _deduplicate(self, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        kept: list[_Candidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.entity_type,
                item.start,
                item.end,
                -item.score,
                item.window_id,
            ),
        ):
            duplicate_indices: list[int] = []
            for index, existing in enumerate(kept):
                if existing.entity_type != candidate.entity_type:
                    continue
                intersection = max(
                    0,
                    min(existing.end, candidate.end) - max(existing.start, candidate.start),
                )
                shorter = min(existing.end - existing.start, candidate.end - candidate.start)
                exact = existing.start == candidate.start and existing.end == candidate.end
                high_overlap = bool(intersection and intersection / shorter >= 0.5)
                if exact or (
                    self.deduplication_policy == "high_overlap_v1" and high_overlap
                ):
                    duplicate_indices.append(index)
            if not duplicate_indices:
                kept.append(candidate)
                continue
            cluster = [candidate, *(kept[index] for index in duplicate_indices)]
            winner = max(
                cluster,
                key=lambda item: (
                    item.score,
                    item.end - item.start,
                    -item.window_id,
                    -item.start,
                ),
            )
            for index in reversed(duplicate_indices):
                kept.pop(index)
            kept.append(winner)
        return sorted(
            kept,
            key=lambda item: (item.start, item.end, item.entity_type, -item.score),
        )

    def _analyze_many(
        self, texts: Sequence[str], entity_filters: Sequence[set[str] | None]
    ) -> list[list[RecognizerResult]]:
        self.load()
        assert self.chunker is not None
        all_windows: list[TokenWindow] = []
        document_indices: list[int] = []
        for document_index, text in enumerate(texts):
            windows = self.chunker.chunk(text)
            all_windows.extend(windows)
            document_indices.extend([document_index] * len(windows))
        raw_batches = self._predict_windows(all_windows) if all_windows else []
        candidates: list[list[_Candidate]] = [[] for _ in texts]
        for window, document_index, predictions in zip(
            all_windows, document_indices, raw_batches, strict=True
        ):
            for prediction in predictions:
                candidate = self._candidate(prediction, window)
                if candidate is None:
                    continue
                requested = entity_filters[document_index]
                if requested is not None and candidate.entity_type not in requested:
                    continue
                candidates[document_index].append(candidate)

        recognizer_identifier = getattr(self, "id", self.recognizer_name)
        results: list[list[RecognizerResult]] = []
        for document_candidates in candidates:
            document_results: list[RecognizerResult] = []
            for candidate in self._deduplicate(document_candidates):
                metadata = {
                    **dict(candidate.metadata),
                    "recognizer_name": self.recognizer_name,
                    "recognizer_identifier": recognizer_identifier,
                    "source": self.source_name,
                    "model_version": self.model_version,
                    "entity_threshold": self.default_threshold,
                    "window_id": candidate.window_id,
                    "deduplication_policy": self.deduplication_policy,
                }
                document_results.append(
                    make_recognizer_result(
                        entity_type=candidate.entity_type,
                        start=candidate.start,
                        end=candidate.end,
                        score=candidate.score,
                        metadata=metadata,
                    )
                )
            results.append(document_results)
        return results

    @staticmethod
    def _entity_filters(
        entities: Sequence[str] | Sequence[Sequence[str]] | None, count: int
    ) -> list[set[str] | None]:
        if entities is None:
            return [None] * count
        entity_values = list(entities)
        if not entity_values or all(isinstance(entity, str) for entity in entity_values):
            shared = {str(entity) for entity in entity_values}
            return [shared] * count
        if len(entity_values) != count:
            raise ValueError("per-text entities must have the same length as texts")
        filters: list[set[str] | None] = []
        for value in entity_values:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError("each per-text entity filter must be a sequence of strings")
            if any(not isinstance(entity, str) for entity in value):
                raise TypeError("entity filters must contain strings")
            filters.append(set(value))
        return filters

    def analyze(
        self,
        text: str,
        entities: Sequence[str] | None,
        nlp_artifacts: Any | None = None,
    ) -> list[RecognizerResult]:
        del nlp_artifacts
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._analyze_many([text], self._entity_filters(entities, 1))[0]

    def analyze_batch(
        self,
        texts: Sequence[str],
        entities: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> list[list[RecognizerResult]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return self._analyze_many(values, self._entity_filters(entities, len(values)))


__all__ = [
    "BioRecognizerDeduplicationPolicy",
    "BioTokenClassifierRecognizer",
    "BioWindowPredictor",
]
