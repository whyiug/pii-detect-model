"""A spaCy-independent Presidio recognizer for local Qwen PII predictors."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

from pii_zh.calibration import CalibrationBundle
from pii_zh.taxonomy import load_presidio_mapping

from ._compat import LocalRecognizer, RecognizerResult, make_recognizer_result
from .token_chunker import TokenizerAwareChunker, TokenWindow


class WindowPredictor(Protocol):
    """Minimal custom-model adapter accepted by :class:`QwenPiiRecognizer`."""

    def predict_batch(self, texts: Sequence[str]) -> Sequence[Sequence[Any]]: ...


ModelLoader = Callable[[Path], Any]


@dataclass(frozen=True, slots=True)
class LocalModelBundle:
    """Return type for custom local model loaders."""

    predictor: Any
    tokenizer: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Candidate:
    entity_type: str
    model_label: str
    start: int
    end: int
    raw_score: float
    score: float
    window_id: int
    boundary_refined: bool
    metadata: Mapping[str, Any]


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("prediction score must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("prediction score must be finite and between zero and one")
    return result


def _integer_offset(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"prediction {name} must be an integer")
    return int(value)


class QwenPiiRecognizer(LocalRecognizer):
    """Run a caller-supplied local predictor as an additional Presidio recognizer.

    No model class is imported here.  A Full-Attention or other custom Qwen
    checkpoint can be supplied through ``model_loader`` or as already-loaded
    ``predictor``/``tokenizer`` objects.  This keeps loading local-only and
    makes the integration testable without downloading weights.
    """

    def __init__(
        self,
        *,
        predictor: Any | None = None,
        tokenizer: Any | None = None,
        model_path: str | Path | None = None,
        model_loader: ModelLoader | None = None,
        chunker: TokenizerAwareChunker | None = None,
        label_mapping: Mapping[str, str] | None = None,
        ignored_labels: Sequence[str] | None = None,
        thresholds: Mapping[str, float] | None = None,
        calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None = None,
        default_threshold: float | None = None,
        max_tokens: int = 512,
        stride_fraction: float = 0.25,
        model_version: str = "unknown-local-model",
        attention_mode: str = "unknown",
        name: str = "QwenPiiRecognizer",
        supported_language: str = "zh",
        version: str = "1.0.0",
    ) -> None:
        packaged_mapping = load_presidio_mapping()
        if label_mapping is None:
            resolved_mapping = dict(packaged_mapping.model_to_presidio)
            resolved_ignored = set(packaged_mapping.ignored_labels)
        else:
            resolved_mapping = dict(label_mapping)
            resolved_ignored = set(ignored_labels or ())
        if not resolved_mapping or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in resolved_mapping.items()
        ):
            raise ValueError("label_mapping must contain non-empty string labels")
        if ignored_labels is not None:
            resolved_ignored.update(ignored_labels)
        supported_entities = sorted(set(resolved_mapping.values()))
        if calibration is None:
            resolved_calibration = CalibrationBundle(
                default_threshold=0.0 if default_threshold is None else default_threshold
            )
        elif isinstance(calibration, CalibrationBundle):
            resolved_calibration = calibration
        elif isinstance(calibration, (str, Path)):
            resolved_calibration = CalibrationBundle.from_json(calibration)
        elif isinstance(calibration, Mapping):
            resolved_calibration = CalibrationBundle.from_dict(calibration)
        else:
            raise TypeError("calibration must be a bundle, mapping, JSON path, or null")
        normalized_thresholds = dict(resolved_calibration.entity_thresholds)
        for entity, threshold in (thresholds or {}).items():
            normalized_thresholds[str(entity)] = _score(threshold)

        self.predictor = predictor
        self.tokenizer = tokenizer
        self.model_path = Path(model_path).expanduser() if model_path is not None else None
        self.model_loader = model_loader
        self.chunker = chunker
        self.label_mapping = resolved_mapping
        self.ignored_labels = frozenset(resolved_ignored)
        self.calibration = resolved_calibration
        self.thresholds = normalized_thresholds
        self.default_threshold = (
            resolved_calibration.default_threshold
            if default_threshold is None
            else _score(default_threshold)
        )
        self.max_tokens = max_tokens
        self.stride_fraction = stride_fraction
        self.model_version = model_version
        self.attention_mode = attention_mode
        self.loader_metadata: dict[str, Any] = {}
        self._loaded = predictor is not None and chunker is not None
        # Presidio's EntityRecognizer constructor eagerly calls ``load``.
        # Initialize every field consumed by our loader before delegating.
        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            version=version,
            context=[],
        )

    def load(self) -> None:
        if self._loaded:
            return
        if self.predictor is None:
            if self.model_loader is None:
                raise RuntimeError(
                    "no predictor is loaded; supply predictor/tokenizer or a local model_loader"
                )
            if self.model_path is None:
                raise ValueError("model_path is required when model_loader is used")
            if not self.model_path.exists():
                raise FileNotFoundError(f"local model path does not exist: {self.model_path}")
            loaded = self.model_loader(self.model_path)
            self._consume_loaded_bundle(loaded)
        if self.tokenizer is None and self.chunker is None:
            self.tokenizer = getattr(self.predictor, "tokenizer", None)
        if self.predictor is None or (self.tokenizer is None and self.chunker is None):
            raise RuntimeError("local model loader must provide both predictor and tokenizer")
        if self.chunker is None:
            self.chunker = TokenizerAwareChunker(
                self.tokenizer,
                max_tokens=self.max_tokens,
                stride_fraction=self.stride_fraction,
            )
        self._loaded = True

    def _consume_loaded_bundle(self, loaded: Any) -> None:
        if isinstance(loaded, LocalModelBundle):
            self.predictor = loaded.predictor
            self.tokenizer = loaded.tokenizer
            self.loader_metadata = dict(loaded.metadata)
            return
        if isinstance(loaded, Mapping):
            self.predictor = loaded.get("predictor")
            self.tokenizer = loaded.get("tokenizer")
            metadata = loaded.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise TypeError("local loader metadata must be a mapping")
            self.loader_metadata = dict(metadata)
            return
        if isinstance(loaded, tuple) and len(loaded) == 2:
            self.predictor, self.tokenizer = loaded
            return
        self.predictor = loaded
        self.tokenizer = getattr(loaded, "tokenizer", None)

    def _mapped_label(self, raw_label: object) -> tuple[str, str] | None:
        if not isinstance(raw_label, str) or not raw_label:
            raise ValueError("prediction label must be a non-empty string")
        stripped = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
        if stripped in self.ignored_labels or raw_label in self.ignored_labels:
            return None
        mapped = self.label_mapping.get(raw_label, self.label_mapping.get(stripped))
        if mapped is None and stripped in self.label_mapping.values():
            mapped = stripped
        if mapped is None:
            raise ValueError(f"prediction label {raw_label!r} is absent from label_mapping")
        return stripped, mapped

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
        assert self.predictor is not None
        texts = [window.text for window in windows]
        batch_method = getattr(self.predictor, "predict_batch", None)
        if callable(batch_method):
            raw_batches = batch_method(texts)
        else:
            single_method = getattr(self.predictor, "predict", None)
            if callable(single_method):
                raw_batches = [single_method(text) for text in texts]
            elif callable(self.predictor):
                raw_batches = [self.predictor(text) for text in texts]
            else:
                raise TypeError("predictor must be callable or expose predict/predict_batch")
        batches = list(raw_batches)
        if len(batches) != len(windows):
            raise ValueError("predictor must return one result sequence per input window")
        normalized: list[Sequence[Any]] = []
        for batch in batches:
            if isinstance(batch, Mapping) and "entities" in batch:
                batch = batch["entities"]
            if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
                raise TypeError("each predictor batch result must be a sequence of spans")
            normalized.append(batch)
        return normalized

    def _threshold_for(self, model_label: str, entity_type: str) -> float:
        if entity_type in self.thresholds:
            return self.thresholds[entity_type]
        if model_label in self.thresholds:
            return self.thresholds[model_label]
        if entity_type in self.calibration.entity_thresholds:
            return self.calibration.threshold_for(entity_type)
        if model_label in self.calibration.entity_thresholds:
            return self.calibration.threshold_for(model_label)
        return self.default_threshold

    def _candidate(self, prediction: Any, window: TokenWindow) -> _Candidate | None:
        raw_label = self._raw_value(prediction, "entity_type", "entity_group", "entity", "label")
        mapped = self._mapped_label(raw_label)
        if mapped is None:
            return None
        model_label, entity_type = mapped
        raw_score = _score(self._raw_value(prediction, "score", "confidence", default=1.0))
        start = _integer_offset("start", self._raw_value(prediction, "start"))
        end = _integer_offset("end", self._raw_value(prediction, "end"))
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
            raise ValueError("prediction offset_mode must be 'relative' or 'global'")
        if not window.owns(start, end):
            return None
        calibrated_score = self.calibration.calibrate_score(entity_type, raw_score)
        if calibrated_score < self._threshold_for(model_label, entity_type):
            return None
        boundary_refined = bool(self._raw_value(prediction, "boundary_refined", default=False))
        return _Candidate(
            entity_type=entity_type,
            model_label=model_label,
            start=start,
            end=end,
            raw_score=raw_score,
            score=calibrated_score,
            window_id=window.window_id,
            boundary_refined=boundary_refined,
            metadata=dict(raw_metadata),
        )

    @staticmethod
    def _deduplicate(candidates: Sequence[_Candidate]) -> list[_Candidate]:
        """Remove exact and high-overlap same-label cross-window duplicates."""

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
                    0, min(existing.end, candidate.end) - max(existing.start, candidate.start)
                )
                shorter = min(existing.end - existing.start, candidate.end - candidate.start)
                if intersection and intersection / shorter >= 0.5:
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
        return sorted(kept, key=lambda item: (item.start, item.end, item.entity_type, -item.score))

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

        results: list[list[RecognizerResult]] = []
        recognizer_identifier = getattr(self, "id", self.name)
        for document_candidates in candidates:
            document_results: list[RecognizerResult] = []
            for candidate in self._deduplicate(document_candidates):
                metadata = {
                    **self.loader_metadata,
                    **dict(candidate.metadata),
                    "recognizer_name": self.name,
                    "recognizer_identifier": recognizer_identifier,
                    "source": "qwen",
                    "raw_score": candidate.raw_score,
                    "calibrated_score": candidate.score,
                    "raw_model_label": candidate.model_label,
                    "model_version": self.model_version,
                    "calibration_version": self.calibration.calibration_version,
                    "entity_threshold": self._threshold_for(
                        candidate.model_label, candidate.entity_type
                    ),
                    "attention_mode": self.attention_mode,
                    "window_id": candidate.window_id,
                    "boundary_refined": candidate.boundary_refined,
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
        del nlp_artifacts  # Character offsets come directly from the tokenizer/predictor.
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._analyze_many([text], self._entity_filters(entities, 1))[0]

    def analyze_batch(
        self,
        texts: Sequence[str],
        entities: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> list[list[RecognizerResult]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return self._analyze_many(values, self._entity_filters(entities, len(values)))


__all__ = ["LocalModelBundle", "ModelLoader", "QwenPiiRecognizer", "WindowPredictor"]
