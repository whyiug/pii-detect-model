"""Deterministic character-span metrics for Chinese PII detection.

Strict and relaxed span matches are one-to-one.  Strict matching requires the
same label and exact left-closed/right-open offsets.  Relaxed matching requires
the same label and an interval IoU at or above the configured threshold.

Character metrics treat each ``(document, character, label)`` as a decision.
Consequently, a wrong label contributes both a false positive for the predicted
label and a false negative for the gold label.  Repeated or overlapping spans
of the same label do not double-count characters.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .io import EvaluationDataError, GoldRecord, PredictionRecord, Span

_MISSED = "__MISSED__"
_SPURIOUS = "__SPURIOUS__"


@dataclass(frozen=True, slots=True)
class _DocumentPair:
    gold: GoldRecord
    prediction: PredictionRecord


def span_iou(left: Span, right: Span) -> float:
    """Return character-interval intersection over union, ignoring labels."""

    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    if intersection == 0:
        return 0.0
    union = (left.end - left.start) + (right.end - right.start) - intersection
    return intersection / union


def _score(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = 4.0
    f2 = (
        (1.0 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def _macro(
    per_class: Mapping[str, Mapping[str, int | float]], labels: Sequence[str]
) -> dict[str, int | float]:
    fields = ("precision", "recall", "f1", "f2")
    if not labels:
        return {"label_count": 0, **{field: 0.0 for field in fields}}
    return {
        "label_count": len(labels),
        **{
            field: sum(float(per_class[label][field]) for label in labels) / len(labels)
            for field in fields
        },
    }


def _span_counts_strict(
    pairs: Sequence[_DocumentPair], labels: Sequence[str]
) -> tuple[dict[str, dict[str, int | float]], dict[str, int | float]]:
    counts = {label: [0, 0, 0] for label in labels}
    for pair in pairs:
        gold_keys = {span.metric_key() for span in pair.gold.spans}
        prediction_keys = {span.metric_key() for span in pair.prediction.spans}
        matched = gold_keys & prediction_keys
        for _, _, label in matched:
            counts[label][0] += 1
        for _, _, label in prediction_keys - matched:
            counts[label][1] += 1
        for _, _, label in gold_keys - matched:
            counts[label][2] += 1
    per_class = {
        label: _score(tp=values[0], fp=values[1], fn=values[2]) for label, values in counts.items()
    }
    micro = _score(
        tp=sum(values[0] for values in counts.values()),
        fp=sum(values[1] for values in counts.values()),
        fn=sum(values[2] for values in counts.values()),
    )
    return per_class, micro


def _maximum_cardinality_matching(
    gold: Sequence[Span],
    predicted: Sequence[Span],
    *,
    candidate: Callable[[Span, Span], bool],
) -> list[tuple[int, int]]:
    """Find a deterministic maximum-cardinality bipartite span matching.

    Candidate order prefers exact boundaries and then larger IoU.  Cardinality,
    rather than summed IoU, controls P/R/F1 and avoids greedy under-counting.
    """

    adjacency: dict[int, list[int]] = {}
    for gold_index, gold_span in enumerate(gold):
        candidates = [
            prediction_index
            for prediction_index, prediction_span in enumerate(predicted)
            if candidate(gold_span, prediction_span)
        ]
        candidates.sort(
            key=lambda index: (
                gold_span.start != predicted[index].start or gold_span.end != predicted[index].end,
                -span_iou(gold_span, predicted[index]),
                predicted[index].start,
                predicted[index].end,
                predicted[index].label,
                index,
            )
        )
        adjacency[gold_index] = candidates

    prediction_to_gold: dict[int, int] = {}

    def augment(gold_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[gold_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous_gold = prediction_to_gold.get(prediction_index)
            if previous_gold is None or augment(previous_gold, visited):
                prediction_to_gold[prediction_index] = gold_index
                return True
        return False

    gold_order = sorted(
        range(len(gold)),
        key=lambda index: (
            len(adjacency[index]),
            gold[index].start,
            gold[index].end,
            gold[index].label,
            index,
        ),
    )
    for gold_index in gold_order:
        augment(gold_index, set())
    return sorted(
        (
            (gold_index, prediction_index)
            for prediction_index, gold_index in prediction_to_gold.items()
        ),
        key=lambda pair: pair[0],
    )


def _span_counts_relaxed(
    pairs: Sequence[_DocumentPair], labels: Sequence[str], iou_threshold: float
) -> tuple[dict[str, dict[str, int | float]], dict[str, int | float]]:
    counts = {label: [0, 0, 0] for label in labels}
    for pair in pairs:
        for label in labels:
            gold = [span for span in pair.gold.spans if span.label == label]
            predicted = [span for span in pair.prediction.spans if span.label == label]
            matched = _maximum_cardinality_matching(
                gold,
                predicted,
                candidate=lambda left, right: span_iou(left, right) >= iou_threshold,
            )
            true_positives = len(matched)
            counts[label][0] += true_positives
            counts[label][1] += len(predicted) - true_positives
            counts[label][2] += len(gold) - true_positives
    per_class = {
        label: _score(tp=values[0], fp=values[1], fn=values[2]) for label, values in counts.items()
    }
    micro = _score(
        tp=sum(values[0] for values in counts.values()),
        fp=sum(values[1] for values in counts.values()),
        fn=sum(values[2] for values in counts.values()),
    )
    return per_class, micro


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


def _character_counts(
    pairs: Sequence[_DocumentPair], labels: Sequence[str]
) -> tuple[dict[str, dict[str, int | float]], dict[str, int | float]]:
    counts = {label: [0, 0, 0] for label in labels}
    for pair in pairs:
        for label in labels:
            gold = _merged_intervals(span for span in pair.gold.spans if span.label == label)
            predicted = _merged_intervals(
                span for span in pair.prediction.spans if span.label == label
            )
            true_positives = _intersection_length(gold, predicted)
            counts[label][0] += true_positives
            counts[label][1] += _interval_length(predicted) - true_positives
            counts[label][2] += _interval_length(gold) - true_positives
    per_class = {
        label: _score(tp=values[0], fp=values[1], fn=values[2]) for label, values in counts.items()
    }
    micro = _score(
        tp=sum(values[0] for values in counts.values()),
        fp=sum(values[1] for values in counts.values()),
        fn=sum(values[2] for values in counts.values()),
    )
    return per_class, micro


def _boundary_metrics(pairs: Sequence[_DocumentPair]) -> dict[str, Any]:
    gold_total = sum(len(pair.gold.spans) for pair in pairs)
    prediction_total = sum(len(pair.prediction.spans) for pair in pairs)
    exact_matches = 0
    overlap_pairs: list[tuple[Span, Span]] = []
    for pair in pairs:
        gold_boundaries = Counter((span.start, span.end) for span in pair.gold.spans)
        prediction_boundaries = Counter((span.start, span.end) for span in pair.prediction.spans)
        exact_matches += sum((gold_boundaries & prediction_boundaries).values())
        matching = _maximum_cardinality_matching(
            pair.gold.spans,
            pair.prediction.spans,
            candidate=lambda left, right: span_iou(left, right) > 0.0,
        )
        overlap_pairs.extend(
            (pair.gold.spans[gold_index], pair.prediction.spans[prediction_index])
            for gold_index, prediction_index in matching
        )
    exact = _score(
        tp=exact_matches,
        fp=prediction_total - exact_matches,
        fn=gold_total - exact_matches,
    )
    matched_count = len(overlap_pairs)
    start_matches = sum(gold.start == predicted.start for gold, predicted in overlap_pairs)
    end_matches = sum(gold.end == predicted.end for gold, predicted in overlap_pairs)
    return {
        "definition": "label_agnostic_character_boundaries",
        "exact": exact,
        # Plan-compatible recall-style boundary accuracy: fraction of gold
        # spans for which any predicted span has exactly the same boundaries.
        "exact_accuracy": exact_matches / gold_total if gold_total else 0.0,
        "overlap_matched_pairs": matched_count,
        "start_accuracy": start_matches / matched_count if matched_count else 0.0,
        "end_accuracy": end_matches / matched_count if matched_count else 0.0,
    }


def _confusion_matrix(pairs: Sequence[_DocumentPair], labels: Sequence[str]) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        matching = _maximum_cardinality_matching(
            pair.gold.spans,
            pair.prediction.spans,
            candidate=lambda left, right: span_iou(left, right) > 0.0,
        )
        matched_gold = {gold_index for gold_index, _ in matching}
        matched_predictions = {prediction_index for _, prediction_index in matching}
        for gold_index, prediction_index in matching:
            matrix[pair.gold.spans[gold_index].label][
                pair.prediction.spans[prediction_index].label
            ] += 1
        for index, span in enumerate(pair.gold.spans):
            if index not in matched_gold:
                matrix[span.label][_MISSED] += 1
        for index, span in enumerate(pair.prediction.spans):
            if index not in matched_predictions:
                matrix[_SPURIOUS][span.label] += 1

    row_labels = [*labels, _SPURIOUS]
    column_labels = [*labels, _MISSED]
    return {
        "matching": "maximum_cardinality_positive_character_overlap",
        "gold_labels": row_labels,
        "predicted_labels": column_labels,
        "matrix": {
            row: {column: matrix[row][column] for column in column_labels} for row in row_labels
        },
    }


def _pii_free_metrics(pairs: Sequence[_DocumentPair]) -> dict[str, int | float | None]:
    pii_free = [pair for pair in pairs if not pair.gold.spans]
    false_positive_documents = sum(bool(pair.prediction.spans) for pair in pii_free)
    return {
        "documents": len(pii_free),
        "false_positive_documents": false_positive_documents,
        "false_positive_rate": (false_positive_documents / len(pii_free) if pii_free else None),
    }


def _confidence_calibration(
    pairs: Sequence[_DocumentPair], *, bin_count: int = 10
) -> dict[str, Any]:
    """Measure confidence calibration over emitted prediction spans.

    A prediction is positive exactly when ``(start, end, label)`` strictly
    matches a gold span in the same document.  False negatives have no emitted
    confidence and are therefore outside this precision-calibration scope.
    Metrics are reported only when every emitted span has a score; calculating
    them on a selectively scored subset would be misleading.
    """

    predictions: list[tuple[float | None, bool]] = []
    for pair in pairs:
        gold_keys = {span.metric_key() for span in pair.gold.spans}
        predictions.extend(
            (span.score, span.metric_key() in gold_keys) for span in pair.prediction.spans
        )
    scored_count = sum(score is not None for score, _ in predictions)
    base: dict[str, Any] = {
        "scope": "emitted_prediction_spans",
        "target": "strict_start_end_label_match",
        "predicted_span_count": len(predictions),
        "scored_span_count": scored_count,
        "binning": {
            "method": "equal_width",
            "bin_count": bin_count,
            "intervals": "left_closed_right_open_except_last_closed",
        },
        "ece": None,
        "brier": None,
        "bins": None,
    }
    if not predictions:
        return {**base, "status": "not_applicable", "reason": "no_prediction_spans"}
    if scored_count == 0:
        return {**base, "status": "not_applicable", "reason": "scores_absent"}
    if scored_count != len(predictions):
        return {**base, "status": "not_applicable", "reason": "scores_incomplete"}

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bin_count)]
    observations: list[tuple[float, bool]] = []
    for score, positive in predictions:
        # Span.validate has already guaranteed a finite score in [0, 1].
        assert score is not None
        normalized_score = float(score)
        index = min(int(normalized_score * bin_count), bin_count - 1)
        buckets[index].append((normalized_score, positive))
        observations.append((normalized_score, positive))

    bins: list[dict[str, int | float | None]] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        count = len(bucket)
        mean_confidence = sum(score for score, _ in bucket) / count if count else None
        accuracy = sum(positive for _, positive in bucket) / count if count else None
        gap = (
            abs(float(accuracy) - float(mean_confidence))
            if accuracy is not None and mean_confidence is not None
            else None
        )
        if gap is not None:
            ece += count / len(observations) * gap
        bins.append(
            {
                "index": index,
                "lower": index / bin_count,
                "upper": (index + 1) / bin_count,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    brier = sum((score - float(positive)) ** 2 for score, positive in observations) / len(
        observations
    )
    return {
        **base,
        "status": "available",
        "reason": None,
        "ece": ece,
        "brier": brier,
        "bins": bins,
    }


def _evaluate_core(
    pairs: Sequence[_DocumentPair], *, labels: Sequence[str], iou_threshold: float
) -> dict[str, Any]:
    strict_per_class, strict_micro = _span_counts_strict(pairs, labels)
    relaxed_per_class, relaxed_micro = _span_counts_relaxed(pairs, labels, iou_threshold)
    character_per_class, character_micro = _character_counts(pairs, labels)
    gold_span_count = sum(len(pair.gold.spans) for pair in pairs)
    prediction_span_count = sum(len(pair.prediction.spans) for pair in pairs)
    return {
        "document_count": len(pairs),
        "span_count": {"gold": gold_span_count, "predicted": prediction_span_count},
        "strict": {
            "micro": strict_micro,
            "macro": _macro(strict_per_class, labels),
            "per_class": strict_per_class,
        },
        "boundary": _boundary_metrics(pairs),
        "relaxed": {
            "iou_threshold": iou_threshold,
            "micro": relaxed_micro,
            "macro": _macro(relaxed_per_class, labels),
            "per_class": relaxed_per_class,
        },
        "character": {
            "micro": character_micro,
            "macro": _macro(character_per_class, labels),
            "per_class": character_per_class,
        },
        "pii_free": _pii_free_metrics(pairs),
        "confidence_calibration": _confidence_calibration(pairs),
        "confusion": _confusion_matrix(pairs, labels),
    }


def _validate_threshold(iou_threshold: float) -> float:
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not math.isfinite(float(iou_threshold))
        or not 0.0 < float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    return float(iou_threshold)


def _align_documents(
    gold_records: Iterable[GoldRecord], prediction_records: Iterable[PredictionRecord]
) -> tuple[list[_DocumentPair], list[str]]:
    gold_by_id: dict[str, GoldRecord] = {}
    for record in gold_records:
        record.validate()
        if record.doc_id in gold_by_id:
            raise EvaluationDataError(f"duplicate gold doc_id: {record.doc_id}")
        gold_by_id[record.doc_id] = record
    if not gold_by_id:
        raise EvaluationDataError("at least one gold document is required")

    prediction_by_id: dict[str, PredictionRecord] = {}
    for record in prediction_records:
        record.validate()
        if record.doc_id in prediction_by_id:
            raise EvaluationDataError(f"duplicate prediction doc_id: {record.doc_id}")
        prediction_by_id[record.doc_id] = record
    unknown = sorted(set(prediction_by_id) - set(gold_by_id))
    if unknown:
        raise EvaluationDataError(
            "predictions contain doc_id values absent from gold: " + ", ".join(unknown)
        )

    pairs: list[_DocumentPair] = []
    labels: set[str] = set()
    for doc_id in sorted(gold_by_id):
        gold = gold_by_id[doc_id]
        prediction = prediction_by_id.get(doc_id, PredictionRecord(doc_id, ()))
        prediction.validate(text_length=gold.text_length)
        labels.update(span.label for span in gold.spans)
        labels.update(span.label for span in prediction.spans)
        pairs.append(_DocumentPair(gold=gold, prediction=prediction))
    return pairs, sorted(labels)


def _nested_value(result: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = result
    for key in path:
        value = value[key]
    return None if value is None else float(value)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _bootstrap(
    pairs: Sequence[_DocumentPair],
    *,
    labels: Sequence[str],
    iou_threshold: float,
    samples: int,
    confidence: float,
    seed: int,
    point_result: Mapping[str, Any],
) -> dict[str, Any]:
    metric_paths = {
        "strict_micro_f1": ("strict", "micro", "f1"),
        "strict_macro_f1": ("strict", "macro", "f1"),
        "relaxed_micro_f1": ("relaxed", "micro", "f1"),
        "character_micro_f1": ("character", "micro", "f1"),
        "boundary_exact_f1": ("boundary", "exact", "f1"),
        "pii_free_false_positive_rate": ("pii_free", "false_positive_rate"),
    }
    values: dict[str, list[float]] = {metric: [] for metric in metric_paths}
    random_generator = random.Random(seed)
    for _ in range(samples):
        resampled = [pairs[random_generator.randrange(len(pairs))] for _ in pairs]
        result = _evaluate_core(resampled, labels=labels, iou_threshold=iou_threshold)
        for metric, path in metric_paths.items():
            value = _nested_value(result, path)
            if value is not None:
                values[metric].append(value)

    tail = (1.0 - confidence) / 2.0
    intervals: dict[str, Any] = {}
    for metric, path in metric_paths.items():
        observed = values[metric]
        intervals[metric] = {
            "point": _nested_value(point_result, path),
            "lower": _percentile(observed, tail) if observed else None,
            "upper": _percentile(observed, 1.0 - tail) if observed else None,
            "effective_samples": len(observed),
        }
    return {
        "method": "document_level_percentile_bootstrap",
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "intervals": intervals,
    }


def evaluate_documents(
    gold_records: Iterable[GoldRecord],
    prediction_records: Iterable[PredictionRecord],
    *,
    iou_threshold: float = 0.5,
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate prediction spans against gold and return a JSON-safe report.

    Bootstrap sampling is paired at document level.  Set ``bootstrap_samples``
    to zero for fast point-estimate-only checks.
    """

    threshold = _validate_threshold(iou_threshold)
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise ValueError("bootstrap_samples must be a non-negative integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be finite and in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    pairs, labels = _align_documents(gold_records, prediction_records)
    result = _evaluate_core(pairs, labels=labels, iou_threshold=threshold)
    result["bootstrap"] = (
        _bootstrap(
            pairs,
            labels=labels,
            iou_threshold=threshold,
            samples=bootstrap_samples,
            confidence=float(confidence),
            seed=seed,
            point_result=result,
        )
        if bootstrap_samples
        else None
    )
    return result


# A concise public alias for callers that do not need to distinguish future
# model-level and end-to-end evaluators.
evaluate = evaluate_documents
