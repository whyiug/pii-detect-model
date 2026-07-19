"""Frozen-metric primitives for the external ``tw-PII-bench`` benchmark.

The benchmark's public rows contain raw synthetic text.  This module validates
that text only while constructing span-only records; returned reports and
prediction artifacts never contain source text or quoted entity values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .io import EvaluationDataError, PredictionRecord, Span

EFFECTIVE_8_LABELS: tuple[str, ...] = (
    "private_person",
    "private_phone",
    "private_email",
    "private_address",
    "private_date",
    "private_url",
    "account_number",
    "secret",
)

ACCOUNT_NUMBER_PROJECT_LABELS: tuple[str, ...] = (
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER",
)

PROJECT_TO_EFFECTIVE_8: dict[str, str] = {
    "PERSON_NAME": "private_person",
    "PHONE_NUMBER": "private_phone",
    "EMAIL_ADDRESS": "private_email",
    "ADDRESS": "private_address",
    "DATE_OF_BIRTH": "private_date",
    **{label: "account_number" for label in ACCOUNT_NUMBER_PROJECT_LABELS},
    "SECRET": "secret",
}

OOD_EXPECTED_FALLBACK: dict[str, str | None] = {
    "tw_national_id": "account_number",
    "tw_nhi_card": "account_number",
    "tw_company_id": "account_number",
    "tw_license_plate": None,
    "tw_passport": "account_number",
    "tw_driver_license": "account_number",
    "tw_line_id": "private_url",
    "tw_ptt_id": None,
    "tw_household_no": "account_number",
    "tw_medical_license": "account_number",
    "tw_military_id": "account_number",
}

BENCHMARK_SPLITS: tuple[str, ...] = ("short", "mid", "long")


class TwPiiProtocolError(EvaluationDataError):
    """Raised when benchmark rows or protocol predictions violate the contract."""


@dataclass(frozen=True, slots=True)
class _GoldSpan:
    start: int
    end: int
    source_label: str
    expected_model_label: str | None


@dataclass(frozen=True, slots=True)
class _ProtocolDocument:
    doc_id: str
    split: str
    text_length: int
    in_schema_gold: tuple[Span, ...]
    effective_gold: tuple[Span, ...]
    ood_gold: tuple[_GoldSpan, ...]
    is_negative: bool


def _score(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _normalize_benchmark_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[_ProtocolDocument, ...]:
    documents: list[_ProtocolDocument] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise TwPiiProtocolError(f"benchmark row {row_number} must be an object")
        doc_id = row.get("id")
        split = row.get("split")
        text = row.get("text")
        is_negative = row.get("is_negative")
        if not isinstance(doc_id, str) or not doc_id:
            raise TwPiiProtocolError(f"benchmark row {row_number} has an invalid id")
        if doc_id in seen:
            raise TwPiiProtocolError(f"duplicate benchmark id: {doc_id}")
        seen.add(doc_id)
        if split not in BENCHMARK_SPLITS:
            raise TwPiiProtocolError(f"benchmark row {doc_id} has an invalid split")
        if not isinstance(text, str):
            raise TwPiiProtocolError(f"benchmark row {doc_id} text must be a string")
        if row.get("char_length") != len(text):
            raise TwPiiProtocolError(f"benchmark row {doc_id} char_length mismatch")
        if not isinstance(is_negative, bool):
            raise TwPiiProtocolError(f"benchmark row {doc_id} is_negative must be boolean")
        raw_spans = row.get("spans")
        if not isinstance(raw_spans, list) or row.get("n_spans") != len(raw_spans):
            raise TwPiiProtocolError(f"benchmark row {doc_id} span count mismatch")
        if is_negative and raw_spans:
            raise TwPiiProtocolError(f"negative benchmark row {doc_id} contains spans")
        in_schema: list[Span] = []
        effective: list[Span] = []
        ood: list[_GoldSpan] = []
        seen_gold: set[tuple[int, int, str]] = set()
        for span_number, raw_span in enumerate(raw_spans, start=1):
            if not isinstance(raw_span, Mapping):
                raise TwPiiProtocolError(
                    f"benchmark row {doc_id} span {span_number} must be an object"
                )
            start = raw_span.get("start")
            end = raw_span.get("end")
            label = raw_span.get("label")
            raw_expected = raw_span.get("expected_model_label")
            expected = None if raw_expected == "" else raw_expected
            quote = raw_span.get("text")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(text)
            ):
                raise TwPiiProtocolError(
                    f"benchmark row {doc_id} span {span_number} has invalid offsets"
                )
            if not isinstance(label, str) or not label:
                raise TwPiiProtocolError(
                    f"benchmark row {doc_id} span {span_number} has an invalid label"
                )
            if not isinstance(quote, str) or text[start:end] != quote:
                raise TwPiiProtocolError(f"benchmark row {doc_id} span {span_number} text mismatch")
            if label in EFFECTIVE_8_LABELS:
                if expected != label:
                    raise TwPiiProtocolError(f"benchmark row {doc_id} in-schema fallback mismatch")
                effective_label: str | None = label
                in_schema.append(Span(start=start, end=end, label=label))
            elif label in OOD_EXPECTED_FALLBACK:
                if expected != OOD_EXPECTED_FALLBACK[label]:
                    raise TwPiiProtocolError(f"benchmark row {doc_id} OOD fallback mismatch")
                effective_label = expected
                ood.append(_GoldSpan(start, end, label, expected))
            else:
                raise TwPiiProtocolError(f"benchmark row {doc_id} has an unknown label")
            if effective_label is not None:
                key = (start, end, effective_label)
                if key in seen_gold:
                    raise TwPiiProtocolError(f"benchmark row {doc_id} has duplicate gold spans")
                seen_gold.add(key)
                effective.append(Span(start=start, end=end, label=effective_label))
        documents.append(
            _ProtocolDocument(
                doc_id=doc_id,
                split=split,
                text_length=len(text),
                in_schema_gold=tuple(in_schema),
                effective_gold=tuple(effective),
                ood_gold=tuple(ood),
                is_negative=is_negative,
            )
        )
    if not documents:
        raise TwPiiProtocolError("benchmark must contain at least one row")
    return tuple(documents)


def project_project_predictions(
    records: Iterable[PredictionRecord],
) -> tuple[PredictionRecord, ...]:
    """Project native project labels to effective-8 and collapse mapped duplicates."""

    projected_records: list[PredictionRecord] = []
    for record in records:
        record.validate()
        by_key: dict[tuple[int, int, str], Span] = {}
        for span in record.spans:
            target = PROJECT_TO_EFFECTIVE_8.get(span.label)
            if target is None:
                continue
            key = (span.start, span.end, target)
            previous = by_key.get(key)
            previous_score = -1.0 if previous is None or previous.score is None else previous.score
            score = -1.0 if span.score is None else span.score
            if previous is None or score > previous_score:
                by_key[key] = Span(span.start, span.end, target, span.score)
        projected_records.append(
            PredictionRecord(
                doc_id=record.doc_id,
                spans=tuple(sorted(by_key.values(), key=lambda value: value.metric_key())),
            )
        )
    return tuple(projected_records)


def _prediction_map(
    records: Iterable[PredictionRecord],
    *,
    expected_ids: set[str],
    effective_only: bool,
) -> dict[str, PredictionRecord]:
    result: dict[str, PredictionRecord] = {}
    for record in records:
        record.validate()
        if record.doc_id in result:
            raise TwPiiProtocolError(f"duplicate prediction id: {record.doc_id}")
        if effective_only and any(span.label not in EFFECTIVE_8_LABELS for span in record.spans):
            raise TwPiiProtocolError("effective predictions contain a non-protocol label")
        result[record.doc_id] = record
    missing = sorted(expected_ids - set(result))
    unknown = sorted(set(result) - expected_ids)
    if missing or unknown:
        raise TwPiiProtocolError(
            f"prediction ids must exactly match benchmark ids; missing={missing[:3]}, "
            f"unknown={unknown[:3]}"
        )
    return result


def _strict_result(
    documents: Sequence[_ProtocolDocument],
    predictions: Mapping[str, PredictionRecord],
    *,
    gold_track: str,
    ignore_ood_prediction_regions: bool,
) -> dict[str, Any]:
    counts = {label: [0, 0, 0] for label in EFFECTIVE_8_LABELS}
    for document in documents:
        gold_spans = (
            document.in_schema_gold
            if gold_track == "original_in_schema"
            else document.effective_gold
        )
        gold = {span.metric_key() for span in gold_spans}
        predicted_spans = predictions[document.doc_id].spans
        if ignore_ood_prediction_regions:
            predicted_spans = tuple(
                prediction
                for prediction in predicted_spans
                if not any(_overlaps(ood.start, ood.end, prediction) for ood in document.ood_gold)
            )
        predicted = {span.metric_key() for span in predicted_spans}
        matched = gold & predicted
        for _, _, label in matched:
            counts[label][0] += 1
        for _, _, label in predicted - matched:
            counts[label][1] += 1
        for _, _, label in gold - matched:
            counts[label][2] += 1
    per_label = {label: _score(*counts[label]) for label in EFFECTIVE_8_LABELS}
    micro = _score(
        sum(value[0] for value in counts.values()),
        sum(value[1] for value in counts.values()),
        sum(value[2] for value in counts.values()),
    )
    macro = {
        "label_count": len(EFFECTIVE_8_LABELS),
        **{
            field: sum(float(per_label[label][field]) for label in EFFECTIVE_8_LABELS)
            / len(EFFECTIVE_8_LABELS)
            for field in ("precision", "recall", "f1")
        },
    }
    gold_attribute = "in_schema_gold" if gold_track == "original_in_schema" else "effective_gold"
    return {
        "documents": len(documents),
        "gold_spans": sum(len(getattr(document, gold_attribute)) for document in documents),
        "predicted_spans": sum(value[0] + value[1] for value in counts.values()),
        "strict_micro": micro,
        "strict_macro": macro,
        "strict_per_label": per_label,
    }


def _negative_result(
    documents: Sequence[_ProtocolDocument], predictions: Mapping[str, PredictionRecord]
) -> dict[str, int | float | None]:
    negatives = [document for document in documents if document.is_negative]
    span_count = sum(len(predictions[document.doc_id].spans) for document in negatives)
    false_positive_documents = sum(
        bool(predictions[document.doc_id].spans) for document in negatives
    )
    return {
        "documents": len(negatives),
        "false_positive_documents": false_positive_documents,
        "false_positive_document_rate": (
            false_positive_documents / len(negatives) if negatives else None
        ),
        "false_positive_spans": span_count,
        "false_positive_spans_per_item": span_count / len(negatives) if negatives else None,
    }


def _overlaps(left_start: int, left_end: int, right: Span) -> bool:
    return max(left_start, right.start) < min(left_end, right.end)


def _effective_prediction_label(label: str) -> str | None:
    if label in EFFECTIVE_8_LABELS:
        return label
    return PROJECT_TO_EFFECTIVE_8.get(label)


def _ood_result(
    documents: Sequence[_ProtocolDocument], native: Mapping[str, PredictionRecord]
) -> dict[str, Any]:
    counts = {
        label: {
            "total": 0,
            "expected_fallback_support": 0,
            "detected_any_overlap": 0,
            "correct_fallback_overlap": 0,
            "wrong_label_overlap": 0,
            "missed": 0,
        }
        for label in OOD_EXPECTED_FALLBACK
    }
    for document in documents:
        predictions = native[document.doc_id].spans
        for gold in document.ood_gold:
            current = counts[gold.source_label]
            current["total"] += 1
            if gold.expected_model_label is not None:
                current["expected_fallback_support"] += 1
            overlapping = [
                prediction
                for prediction in predictions
                if _overlaps(gold.start, gold.end, prediction)
            ]
            if not overlapping:
                current["missed"] += 1
                continue
            current["detected_any_overlap"] += 1
            if gold.expected_model_label is not None and any(
                _effective_prediction_label(prediction.label) == gold.expected_model_label
                for prediction in overlapping
            ):
                current["correct_fallback_overlap"] += 1
            else:
                current["wrong_label_overlap"] += 1

    def rates(values: Mapping[str, int]) -> dict[str, int | float | None]:
        total = values["total"]
        fallback = values["expected_fallback_support"]
        return {
            **values,
            "detection_rate": values["detected_any_overlap"] / total if total else None,
            # This matches the benchmark card: null-fallback OOD spans remain
            # in the denominator and therefore contribute zero generalization.
            "generalization_rate": (values["correct_fallback_overlap"] / total if total else None),
            "generalization_rate_on_expected_fallback": (
                values["correct_fallback_overlap"] / fallback if fallback else None
            ),
        }

    pooled_counts = {
        field: sum(values[field] for values in counts.values())
        for field in next(iter(counts.values()))
    }
    return {
        "matching": "any_positive_character_overlap_per_gold_ood_span",
        "per_label": {label: rates(counts[label]) for label in OOD_EXPECTED_FALLBACK},
        "pooled": rates(pooled_counts),
    }


def evaluate_tw_pii_benchmark(
    rows: Iterable[Mapping[str, Any]],
    effective_predictions: Iterable[PredictionRecord],
    *,
    native_predictions: Iterable[PredictionRecord] | None = None,
) -> dict[str, Any]:
    """Evaluate one complete prediction set without thresholds or text output."""

    documents = _normalize_benchmark_rows(rows)
    expected_ids = {document.doc_id for document in documents}
    effective = _prediction_map(
        effective_predictions, expected_ids=expected_ids, effective_only=True
    )
    native = (
        effective
        if native_predictions is None
        else _prediction_map(native_predictions, expected_ids=expected_ids, effective_only=False)
    )
    by_split = {
        split: tuple(document for document in documents if document.split == split)
        for split in BENCHMARK_SPLITS
    }
    return {
        "schema_version": 1,
        "benchmark": "lianghsun/tw-PII-bench",
        "protocol": {
            "effective_labels": list(EFFECTIVE_8_LABELS),
            "strict_match": "exact_left_closed_right_open_start_end_and_label",
            "headline_gold": "original_in_schema_labels_only",
            "headline_ood_regions": "ignore_predictions_with_any_positive_character_overlap",
            "missing_project_output_label": "private_url",
            "threshold_selection": "none",
            "ood_matching": "any_positive_character_overlap",
            "pii_free_scope": "explicit_is_negative_rows_only",
        },
        "original_in_schema_8": {
            "pooled": _strict_result(
                documents,
                effective,
                gold_track="original_in_schema",
                ignore_ood_prediction_regions=True,
            ),
            "by_split": {
                split: _strict_result(
                    split_documents,
                    effective,
                    gold_track="original_in_schema",
                    ignore_ood_prediction_regions=True,
                )
                for split, split_documents in by_split.items()
            },
        },
        "effective_8": {
            "pooled": _strict_result(
                documents,
                effective,
                gold_track="effective",
                ignore_ood_prediction_regions=False,
            ),
            "by_split": {
                split: _strict_result(
                    split_documents,
                    effective,
                    gold_track="effective",
                    ignore_ood_prediction_regions=False,
                )
                for split, split_documents in by_split.items()
            },
        },
        "hard_negatives": {
            "effective_8": {
                "pooled": _negative_result(documents, effective),
                "by_split": {
                    split: _negative_result(split_documents, effective)
                    for split, split_documents in by_split.items()
                },
            },
            "native_any_label": {
                "pooled": _negative_result(documents, native),
                "by_split": {
                    split: _negative_result(split_documents, native)
                    for split, split_documents in by_split.items()
                },
            },
        },
        "ood_diagnostics": _ood_result(documents, native),
    }


__all__ = [
    "ACCOUNT_NUMBER_PROJECT_LABELS",
    "BENCHMARK_SPLITS",
    "EFFECTIVE_8_LABELS",
    "OOD_EXPECTED_FALLBACK",
    "PROJECT_TO_EFFECTIVE_8",
    "TwPiiProtocolError",
    "evaluate_tw_pii_benchmark",
    "project_project_predictions",
]
