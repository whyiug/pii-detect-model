"""Privacy-preserving JSONL contracts for character-span evaluation.

Gold rows may be canonical :class:`~pii_zh.data.schema.DocumentRecord` JSON or
the smaller ``{"doc_id", "text_length", "spans"}`` evaluation form.  Raw gold
text is used only long enough to validate offsets and is never retained.

Prediction rows are intentionally stricter: their only top-level fields are
``doc_id`` and ``spans``.  This makes it difficult for an evaluation or export
step to accidentally persist source text.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


class EvaluationDataError(ValueError):
    """Raised when evaluation JSONL violates the safe span contract."""


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """A left-closed, right-open character span.

    ``score`` is allowed for prediction serialization but is deliberately not
    used by the uncalibrated span metrics in this module.
    """

    start: int
    end: int
    label: str
    score: float | None = None

    def validate(self, *, text_length: int | None = None) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise EvaluationDataError("span.start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise EvaluationDataError("span.end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise EvaluationDataError("span offsets must satisfy 0 <= start < end")
        if text_length is not None and self.end > text_length:
            raise EvaluationDataError("span end exceeds the gold document length")
        if not isinstance(self.label, str) or not self.label.strip():
            raise EvaluationDataError("span.label must be a non-empty string")
        if self.score is not None:
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (int, float))
                or not math.isfinite(float(self.score))
                or not 0.0 <= float(self.score) <= 1.0
            ):
                raise EvaluationDataError("span.score must be finite and between 0 and 1")

    def metric_key(self) -> tuple[int, int, str]:
        """Return the score-independent identity used by evaluation."""

        return self.start, self.end, self.label

    def to_prediction_dict(self) -> dict[str, int | str | float]:
        """Return a prediction-safe mapping that can never contain raw text."""

        self.validate()
        result: dict[str, int | str | float] = {
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }
        if self.score is not None:
            result["score"] = float(self.score)
        return result


@dataclass(frozen=True, slots=True)
class GoldRecord:
    """The non-sensitive portion of one gold document.

    ``text_length`` can be omitted for already-sanitized span-only gold data.
    Canonical gold JSON always supplies it through its ``text`` field.
    """

    doc_id: str
    spans: tuple[Span, ...]
    text_length: int | None = None

    def validate(self) -> None:
        _validate_doc_id(self.doc_id)
        if self.text_length is not None and (
            isinstance(self.text_length, bool)
            or not isinstance(self.text_length, int)
            or self.text_length < 0
        ):
            raise EvaluationDataError("gold text_length must be a non-negative integer or null")
        _validate_spans(self.spans, text_length=self.text_length)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """A prediction document containing only an identifier and spans."""

    doc_id: str
    spans: tuple[Span, ...]

    def validate(self, *, text_length: int | None = None) -> None:
        _validate_doc_id(self.doc_id)
        _validate_spans(self.spans, text_length=text_length)

    def to_dict(self) -> dict[str, str | list[dict[str, int | str | float]]]:
        self.validate()
        return {
            "doc_id": self.doc_id,
            "spans": [span.to_prediction_dict() for span in _sorted_spans(self.spans)],
        }


def _validate_doc_id(doc_id: object) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise EvaluationDataError("doc_id must be a non-empty string")
    return doc_id


def _validate_spans(spans: tuple[Span, ...], *, text_length: int | None) -> None:
    if not isinstance(spans, tuple):
        raise EvaluationDataError("spans must be a tuple in the in-memory evaluation form")
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        if not isinstance(span, Span):
            raise EvaluationDataError("spans must contain Span values")
        span.validate(text_length=text_length)
        key = span.metric_key()
        if key in seen:
            raise EvaluationDataError("duplicate spans are not permitted")
        seen.add(key)


def _sorted_spans(spans: Iterable[Span]) -> list[Span]:
    return sorted(spans, key=lambda span: (span.start, span.end, span.label, -(span.score or 0.0)))


def _load_json_object(line: str, *, kind: str, line_number: int) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationDataError(
                    f"{kind} JSONL line {line_number} contains a duplicate object key"
                )
            result[key] = value
        return result

    def reject_non_finite(_token: str) -> None:
        raise EvaluationDataError(
            f"{kind} JSONL line {line_number} contains a non-finite number"
        )

    try:
        value = json.loads(
            line,
            object_pairs_hook=unique_object,
            parse_constant=reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"{kind} JSONL line {line_number} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{kind} JSONL line {line_number} must be an object")
    return value


def _parse_span(
    value: object,
    *,
    kind: str,
    line_number: int,
    span_number: int,
    text: str | None = None,
    prediction: bool = False,
) -> Span:
    location = f"{kind} JSONL line {line_number} span {span_number}"
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{location} must be an object")
    if prediction:
        unknown = set(value) - {"start", "end", "label", "score"}
        if unknown:
            raise EvaluationDataError(
                f"{location} contains forbidden or unknown fields: {sorted(unknown)}"
            )
    required = {"start", "end", "label"}
    missing = required - set(value)
    if missing:
        raise EvaluationDataError(f"{location} is missing fields: {sorted(missing)}")
    span = Span(
        start=value["start"],
        end=value["end"],
        label=value["label"],
        score=value.get("score"),
    )
    try:
        span.validate(text_length=len(text) if text is not None else None)
    except EvaluationDataError as exc:
        raise EvaluationDataError(f"{location}: {exc}") from exc
    if text is not None and "text" in value:
        quote = value["text"]
        if not isinstance(quote, str) or text[span.start : span.end] != quote:
            # Do not echo either value: error output is part of the privacy boundary.
            raise EvaluationDataError(f"{location} does not match the gold text slice")
    return span


def _parse_gold(value: Mapping[str, Any], *, line_number: int) -> GoldRecord:
    doc_id = _validate_doc_id(value.get("doc_id"))
    if "spans" in value and "entities" in value:
        raise EvaluationDataError(
            f"gold JSONL line {line_number} must not contain both spans and entities"
        )
    raw_spans = value.get("spans", value.get("entities"))
    if not isinstance(raw_spans, list):
        raise EvaluationDataError(f"gold JSONL line {line_number} spans must be an array")

    raw_text = value.get("text")
    if raw_text is not None and not isinstance(raw_text, str):
        raise EvaluationDataError(f"gold JSONL line {line_number} text must be a string")
    raw_length = value.get("text_length")
    if raw_length is not None and (
        isinstance(raw_length, bool) or not isinstance(raw_length, int) or raw_length < 0
    ):
        raise EvaluationDataError(
            f"gold JSONL line {line_number} text_length must be a non-negative integer"
        )
    if raw_text is not None and raw_length is not None and len(raw_text) != raw_length:
        raise EvaluationDataError(f"gold JSONL line {line_number} text_length does not match text")
    text_length = len(raw_text) if raw_text is not None else raw_length
    spans = tuple(
        _parse_span(
            item,
            kind="gold",
            line_number=line_number,
            span_number=index,
            text=raw_text,
        )
        for index, item in enumerate(raw_spans)
    )
    record = GoldRecord(doc_id=doc_id, spans=spans, text_length=text_length)
    try:
        record.validate()
    except EvaluationDataError as exc:
        raise EvaluationDataError(f"gold JSONL line {line_number}: {exc}") from exc
    return record


def _parse_prediction(value: Mapping[str, Any], *, line_number: int) -> PredictionRecord:
    unknown = set(value) - {"doc_id", "spans"}
    if unknown:
        raise EvaluationDataError(
            "prediction JSONL may contain only doc_id and spans; "
            f"line {line_number} has forbidden or unknown fields: {sorted(unknown)}"
        )
    missing = {"doc_id", "spans"} - set(value)
    if missing:
        raise EvaluationDataError(
            f"prediction JSONL line {line_number} is missing fields: {sorted(missing)}"
        )
    raw_spans = value["spans"]
    if not isinstance(raw_spans, list):
        raise EvaluationDataError(f"prediction JSONL line {line_number} spans must be an array")
    spans = tuple(
        _parse_span(
            item,
            kind="prediction",
            line_number=line_number,
            span_number=index,
            prediction=True,
        )
        for index, item in enumerate(raw_spans)
    )
    record = PredictionRecord(doc_id=_validate_doc_id(value["doc_id"]), spans=spans)
    try:
        record.validate()
    except EvaluationDataError as exc:
        raise EvaluationDataError(f"prediction JSONL line {line_number}: {exc}") from exc
    return record


def _iter_lines(source: str | Path | TextIO) -> Iterator[tuple[int, str]]:
    handle: TextIO
    should_close = False
    if isinstance(source, (str, Path)):
        handle = Path(source).open("r", encoding="utf-8")
        should_close = True
    else:
        handle = source
    try:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, line
    finally:
        if should_close:
            handle.close()


def load_gold_jsonl(source: str | Path | TextIO) -> list[GoldRecord]:
    """Load gold JSONL without retaining the raw ``text`` field."""

    records = [
        _parse_gold(
            _load_json_object(line, kind="gold", line_number=line_number),
            line_number=line_number,
        )
        for line_number, line in _iter_lines(source)
    ]
    _validate_unique_doc_ids(records, kind="gold")
    return records


def load_prediction_jsonl(source: str | Path | TextIO) -> list[PredictionRecord]:
    """Load strict, raw-text-free prediction JSONL."""

    records = [
        _parse_prediction(
            _load_json_object(line, kind="prediction", line_number=line_number),
            line_number=line_number,
        )
        for line_number, line in _iter_lines(source)
    ]
    _validate_unique_doc_ids(records, kind="prediction")
    return records


def _validate_unique_doc_ids(
    records: Iterable[GoldRecord | PredictionRecord], *, kind: str
) -> None:
    seen: set[str] = set()
    for record in records:
        if record.doc_id in seen:
            raise EvaluationDataError(f"duplicate {kind} doc_id: {record.doc_id}")
        seen.add(record.doc_id)


def dumps_prediction(record: PredictionRecord) -> str:
    """Serialize one prediction without any raw document or entity text."""

    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)


def write_prediction_jsonl(records: Iterable[PredictionRecord], target: str | Path | TextIO) -> int:
    """Write normalized prediction JSONL containing only ``doc_id`` and spans."""

    handle: TextIO
    should_close = False
    if isinstance(target, (str, Path)):
        handle = Path(target).open("w", encoding="utf-8")
        should_close = True
    else:
        handle = target
    count = 0
    seen: set[str] = set()
    try:
        for record in records:
            record.validate()
            if record.doc_id in seen:
                raise EvaluationDataError(f"duplicate prediction doc_id: {record.doc_id}")
            seen.add(record.doc_id)
            handle.write(dumps_prediction(record) + "\n")
            count += 1
    finally:
        if should_close:
            handle.close()
    return count
