"""Additive release diagnostics for the cascade evaluator.

This module is deliberately separate from the frozen core evaluator.  It uses
only privacy-safe ``GoldRecord``/``PredictionRecord`` values and never retains
or emits raw text, entity values, or document identifiers.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pii_zh.taxonomy import TaxonomyValidationError, load_taxonomy

from .io import EvaluationDataError, GoldRecord, PredictionRecord, Span

REQUIRED_METRICS_SCHEMA_VERSION = "pii-zh.required-metrics.v1"


@dataclass(frozen=True, slots=True)
class _DocumentPair:
    gold: GoldRecord
    prediction: PredictionRecord


def _align_documents(
    gold_records: Iterable[GoldRecord],
    prediction_records: Iterable[PredictionRecord],
) -> list[_DocumentPair]:
    gold_by_id: dict[str, GoldRecord] = {}
    for gold in gold_records:
        gold.validate()
        if gold.doc_id in gold_by_id:
            raise EvaluationDataError("duplicate gold document in required metrics")
        gold_by_id[gold.doc_id] = gold
    if not gold_by_id:
        raise EvaluationDataError("required metrics need at least one gold document")

    prediction_by_id: dict[str, PredictionRecord] = {}
    for prediction in prediction_records:
        if prediction.doc_id in prediction_by_id:
            raise EvaluationDataError("duplicate prediction document in required metrics")
        prediction_by_id[prediction.doc_id] = prediction
    if set(prediction_by_id) - set(gold_by_id):
        raise EvaluationDataError("required metrics predictions contain an unknown document")

    pairs: list[_DocumentPair] = []
    for doc_id in sorted(gold_by_id):
        gold = gold_by_id[doc_id]
        prediction = prediction_by_id.get(doc_id, PredictionRecord(doc_id, ()))
        prediction.validate(text_length=gold.text_length)
        pairs.append(_DocumentPair(gold, prediction))
    return pairs


def _merged_intervals(spans: Iterable[Span]) -> list[tuple[int, int]]:
    intervals = sorted((span.start, span.end) for span in spans)
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_length(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_length(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> int:
    total = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _positive_overlap(left: Span, right: Span) -> bool:
    return min(left.end, right.end) > max(left.start, right.start)


def _strict_label_recall(pairs: Sequence[_DocumentPair]) -> dict[str, tuple[int, float]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pair in pairs:
        gold_keys = {span.metric_key() for span in pair.gold.spans}
        prediction_keys = {span.metric_key() for span in pair.prediction.spans}
        for key in gold_keys:
            label = key[2]
            counts[label][1] += 1
            counts[label][0] += key in prediction_keys
    return {
        label: (support, true_positive / support)
        for label, (true_positive, support) in counts.items()
    }


@lru_cache(maxsize=1)
def _risk_tier_identity() -> tuple[str, dict[str, tuple[str, ...]], str]:
    taxonomy = load_taxonomy()
    t0_labels = tuple(
        sorted(
            entity.name
            for entity in taxonomy.entities
            if entity.output and "T0" in entity.risk_tiers
        )
    )
    # Match the training/calibration gate: T0 takes precedence when one label
    # is declared in both T0 and T1.
    t1_labels = tuple(
        sorted(
            entity.name
            for entity in taxonomy.entities
            if entity.output and "T0" not in entity.risk_tiers and "T1" in entity.risk_tiers
        )
    )
    tiers = {"T0": t0_labels, "T1": t1_labels}
    encoded = json.dumps(
        {tier: list(labels) for tier, labels in tiers.items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return taxonomy.taxonomy_version, tiers, hashlib.sha256(encoded).hexdigest()


def _minimum_tier_recall(
    recall_by_label: dict[str, tuple[int, float]], labels: Sequence[str]
) -> dict[str, Any]:
    supported = [(label, *recall_by_label[label]) for label in labels if label in recall_by_label]
    if not supported:
        return {
            "status": "unavailable",
            "reason": "no_gold_supported_labels_for_tier",
            "evaluated_label_count": 0,
            "gold_span_support": 0,
            "minimum_recall": None,
            "minimum_recall_labels": [],
        }
    minimum = min(recall for _, _, recall in supported)
    return {
        "status": "available",
        "reason": None,
        "evaluated_label_count": len(supported),
        "gold_span_support": sum(support for _, support, _ in supported),
        "minimum_recall": minimum,
        "minimum_recall_labels": sorted(
            label for label, _, recall in supported if recall == minimum
        ),
    }


def _risk_tier_minimum_recall(pairs: Sequence[_DocumentPair]) -> dict[str, Any]:
    try:
        taxonomy_version, tier_labels, assignment_sha256 = _risk_tier_identity()
    except (OSError, TaxonomyValidationError, TypeError, ValueError):
        unavailable = {
            "status": "unavailable",
            "reason": "taxonomy_unavailable",
            "evaluated_label_count": 0,
            "gold_span_support": 0,
            "minimum_recall": None,
            "minimum_recall_labels": [],
        }
        return {
            "status": "unavailable",
            "reason": "taxonomy_unavailable",
            "taxonomy_version": None,
            "tier_assignment_sha256": None,
            "assignment": {"T0": [], "T1": []},
            "tiers": {"T0": dict(unavailable), "T1": dict(unavailable)},
        }

    recall_by_label = _strict_label_recall(pairs)
    tiers = {
        tier: _minimum_tier_recall(recall_by_label, labels) for tier, labels in tier_labels.items()
    }
    available_count = sum(item["status"] == "available" for item in tiers.values())
    if available_count == len(tiers):
        status = "available"
        reason = None
    elif available_count:
        status = "partially_available"
        reason = "one_or_more_tiers_lack_gold_support"
    else:
        status = "unavailable"
        reason = "no_tier_has_gold_support"
    return {
        "status": status,
        "reason": reason,
        "scope": "gold_supported_taxonomy_output_labels_with_T0_precedence",
        "taxonomy_version": taxonomy_version,
        "tier_assignment_sha256": assignment_sha256,
        "assignment": {tier: list(labels) for tier, labels in tier_labels.items()},
        "tiers": tiers,
    }


def _pii_free_false_positive_span_density(pairs: Sequence[_DocumentPair]) -> dict[str, Any]:
    pii_free = [pair for pair in pairs if not pair.gold.spans]
    false_positive_spans = sum(len(pair.prediction.spans) for pair in pii_free)
    missing_lengths = sum(pair.gold.text_length is None for pair in pii_free)
    result: dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "pii_free_document_count": len(pii_free),
        "false_positive_span_count": false_positive_spans,
        "character_count": None,
        "missing_text_length_document_count": missing_lengths,
        "false_positive_spans_per_1000_characters": None,
    }
    if not pii_free:
        result["reason"] = "no_pii_free_documents"
        return result
    if missing_lengths:
        result["reason"] = "pii_free_text_length_incomplete"
        return result
    character_count = sum(int(pair.gold.text_length or 0) for pair in pii_free)
    result["character_count"] = character_count
    if character_count == 0:
        result["reason"] = "zero_pii_free_character_count"
        return result
    result.update(
        {
            "status": "available",
            "false_positive_spans_per_1000_characters": (
                false_positive_spans * 1000.0 / character_count
            ),
        }
    )
    return result


def _redaction_character_error_rates(pairs: Sequence[_DocumentPair]) -> dict[str, Any]:
    gold_characters = 0
    predicted_characters = 0
    covered_characters = 0
    for pair in pairs:
        gold = _merged_intervals(pair.gold.spans)
        predicted = _merged_intervals(pair.prediction.spans)
        gold_characters += _interval_length(gold)
        predicted_characters += _interval_length(predicted)
        covered_characters += _intersection_length(gold, predicted)

    missed_characters = gold_characters - covered_characters
    over_redacted_characters = predicted_characters - covered_characters
    return {
        "matching": "label_agnostic_per_document_character_union",
        "gold_sensitive_character_count": gold_characters,
        "predicted_redacted_character_count": predicted_characters,
        "covered_sensitive_character_count": covered_characters,
        "missed_sensitive_character_rate": {
            "status": "available" if gold_characters else "unavailable",
            "reason": None if gold_characters else "no_gold_sensitive_characters",
            "numerator": missed_characters,
            "denominator": gold_characters,
            "value": missed_characters / gold_characters if gold_characters else None,
        },
        "over_redaction_character_rate": {
            "status": "available" if predicted_characters else "unavailable",
            "reason": None if predicted_characters else "no_predicted_redacted_characters",
            "numerator": over_redacted_characters,
            "denominator": predicted_characters,
            "value": (
                over_redacted_characters / predicted_characters if predicted_characters else None
            ),
        },
    }


def _fragmentation_metrics(pairs: Sequence[_DocumentPair]) -> dict[str, Any]:
    gold_span_count = sum(len(pair.gold.spans) for pair in pairs)
    fragmented = 0
    for pair in pairs:
        for gold in pair.gold.spans:
            intersecting = sum(
                predicted.label == gold.label and _positive_overlap(gold, predicted)
                for predicted in pair.prediction.spans
            )
            fragmented += intersecting >= 2
    return {
        "status": "available" if gold_span_count else "unavailable",
        "reason": None if gold_span_count else "no_gold_spans",
        "matching": "same_label_positive_character_overlap",
        "fragmented_gold_span_count": fragmented,
        "gold_span_count": gold_span_count,
        "fragmentation_rate": fragmented / gold_span_count if gold_span_count else None,
    }


def build_required_metrics(
    gold_records: Iterable[GoldRecord],
    prediction_records: Iterable[PredictionRecord],
) -> dict[str, Any]:
    """Return additive aggregate diagnostics without changing core metrics."""

    pairs = _align_documents(gold_records, prediction_records)
    return {
        "schema_version": REQUIRED_METRICS_SCHEMA_VERSION,
        "risk_tier_minimum_recall": _risk_tier_minimum_recall(pairs),
        "pii_free_false_positive_span_density": _pii_free_false_positive_span_density(pairs),
        "redaction_character_errors": _redaction_character_error_rates(pairs),
        "fragmentation": _fragmentation_metrics(pairs),
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
        },
    }


__all__ = ["REQUIRED_METRICS_SCHEMA_VERSION", "build_required_metrics"]
