"""Deterministic, calibration-bound span post-processing primitives.

These functions are model-agnostic and never persist source text.  Thresholds
must be selected on a declared development/calibration split; callers are
responsible for recording that lineage instead of tuning on a test benchmark.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pii_zh.evaluation import PredictionRecord, Span

_SCHEME_URL = re.compile(
    r"(?i)(?<![A-Za-z0-9_@])(?:"
    r"(?:https?|ftp)://[^\s<>\"'，。！？；：、（）【】《》]+"
    r"|www\.[^\s<>\"'，。！？；：、（）【】《》]+"
    r")"
)
_TRAILING_URL_PUNCTUATION = frozenset(".,;:!?，。！？；：、)]}）》】")


class SpanPostprocessingError(ValueError):
    """Raised without including raw source text or span surface values."""


@dataclass(frozen=True, slots=True)
class LengthScorePolicy:
    """Two-bin threshold policy selected on a non-test calibration split."""

    short_max_characters: int
    short_minimum_score: float
    long_minimum_score: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.short_max_characters, bool)
            or not isinstance(self.short_max_characters, int)
            or self.short_max_characters < 1
        ):
            raise SpanPostprocessingError("short_max_characters must be a positive integer")
        for value in (self.short_minimum_score, self.long_minimum_score):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise SpanPostprocessingError("score thresholds must be finite within [0, 1]")

    def threshold_for(self, span: Span) -> float:
        return (
            float(self.short_minimum_score)
            if span.end - span.start <= self.short_max_characters
            else float(self.long_minimum_score)
        )


def filter_prediction_records(
    records: Iterable[PredictionRecord], *, policy: LengthScorePolicy
) -> tuple[PredictionRecord, ...]:
    """Apply a frozen length/score policy while preserving document order."""

    result: list[PredictionRecord] = []
    seen_ids: set[str] = set()
    for record in records:
        record.validate()
        if record.doc_id in seen_ids:
            raise SpanPostprocessingError("prediction records contain a duplicate document id")
        seen_ids.add(record.doc_id)
        kept: list[Span] = []
        for span in record.spans:
            if span.score is None or not math.isfinite(float(span.score)):
                raise SpanPostprocessingError("length/score filtering requires finite span scores")
            if float(span.score) >= policy.threshold_for(span):
                kept.append(span)
        result.append(PredictionRecord(doc_id=record.doc_id, spans=tuple(kept)))
    return tuple(result)


def detect_scheme_urls(text: str, *, label: str = "URL", score: float = 1.0) -> tuple[Span, ...]:
    """Detect explicit scheme/www URLs; deliberately exclude bare domains."""

    if not isinstance(text, str):
        raise SpanPostprocessingError("URL detection input must be text")
    if not isinstance(label, str) or not label:
        raise SpanPostprocessingError("URL detection label must be non-empty")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise SpanPostprocessingError("URL detection score must be finite within [0, 1]")
    spans: list[Span] = []
    for match in _SCHEME_URL.finditer(text):
        end = match.end()
        while end > match.start() and text[end - 1] in _TRAILING_URL_PUNCTUATION:
            end -= 1
        if end > match.start():
            spans.append(Span(match.start(), end, label, float(score)))
    return tuple(spans)


def augment_with_scheme_urls(
    records: Iterable[PredictionRecord],
    texts_by_id: Mapping[str, str],
    *,
    label: str = "URL",
) -> tuple[PredictionRecord, ...]:
    """Add fixed URL spans and collapse exact duplicate model/rule spans."""

    result: list[PredictionRecord] = []
    seen: set[str] = set()
    for record in records:
        record.validate()
        if record.doc_id in seen:
            raise SpanPostprocessingError("prediction records contain a duplicate document id")
        seen.add(record.doc_id)
        text = texts_by_id.get(record.doc_id)
        if not isinstance(text, str):
            raise SpanPostprocessingError("URL augmentation lacks a source document")
        by_key = {span.metric_key(): span for span in record.spans}
        for span in detect_scheme_urls(text, label=label):
            previous = by_key.get(span.metric_key())
            if previous is None or previous.score is None or float(previous.score) < float(span.score):
                by_key[span.metric_key()] = span
        result.append(
            PredictionRecord(
                doc_id=record.doc_id,
                spans=tuple(sorted(by_key.values(), key=lambda item: item.metric_key())),
            )
        )
    if set(texts_by_id) != seen:
        raise SpanPostprocessingError("URL source and prediction document sets differ")
    return tuple(result)


__all__ = [
    "LengthScorePolicy",
    "SpanPostprocessingError",
    "augment_with_scheme_urls",
    "detect_scheme_urls",
    "filter_prediction_records",
]
