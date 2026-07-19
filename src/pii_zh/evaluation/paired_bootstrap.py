"""Deterministic paired document bootstrap for strict span-F1 differences.

The resampling unit is a document.  Candidate and baseline counts from the
same document are always selected together.  When multiple suites are passed,
each suite is resampled at its original size so a pooled interval preserves
the frozen suite mixture.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .io import EvaluationDataError, GoldRecord, PredictionRecord


@dataclass(frozen=True, slots=True)
class StrictCounts:
    """Strict exact-match counts for one system."""

    tp: int
    fp: int
    fn: int

    def __add__(self, other: StrictCounts) -> StrictCounts:
        return StrictCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


@dataclass(frozen=True, slots=True)
class PairedDocumentCounts:
    """Candidate and baseline counts for the same document."""

    candidate: StrictCounts
    baseline: StrictCounts


@dataclass(frozen=True, slots=True)
class PairedSuite:
    """Privacy-safe document-level sufficient statistics for one suite."""

    name: str
    documents: tuple[PairedDocumentCounts, ...]


def strict_f1(counts: StrictCounts) -> float:
    """Compute strict micro F1 directly from integer counts."""

    denominator = 2 * counts.tp + counts.fp + counts.fn
    return 2 * counts.tp / denominator if denominator else 0.0


def _strict_counts(
    gold: GoldRecord,
    prediction: PredictionRecord,
    *,
    labels: frozenset[str],
) -> StrictCounts:
    prediction.validate(text_length=gold.text_length)
    gold_keys = {span.metric_key() for span in gold.spans}
    prediction_keys = {span.metric_key() for span in prediction.spans}
    unexpected_gold = {label for _, _, label in gold_keys} - labels
    unexpected_predictions = {label for _, _, label in prediction_keys} - labels
    if unexpected_gold:
        raise EvaluationDataError("gold contains labels outside the paired comparison protocol")
    if unexpected_predictions:
        raise EvaluationDataError(
            "predictions contain labels outside the paired comparison protocol"
        )
    matched = gold_keys & prediction_keys
    return StrictCounts(
        tp=len(matched),
        fp=len(prediction_keys - matched),
        fn=len(gold_keys - matched),
    )


def build_paired_suite(
    *,
    name: str,
    gold_records: Sequence[GoldRecord],
    candidate_records: Sequence[PredictionRecord],
    baseline_records: Sequence[PredictionRecord],
    labels: Sequence[str],
) -> PairedSuite:
    """Align two systems to identical gold documents and retain counts only."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("suite name must be a non-empty string")
    protocol_labels = tuple(labels)
    if not protocol_labels or len(set(protocol_labels)) != len(protocol_labels):
        raise ValueError("labels must be unique and non-empty")
    label_set = frozenset(protocol_labels)

    gold_by_id: dict[str, GoldRecord] = {}
    for record in gold_records:
        record.validate()
        if record.doc_id in gold_by_id:
            raise EvaluationDataError("duplicate gold doc_id in paired comparison")
        gold_by_id[record.doc_id] = record
    if not gold_by_id:
        raise EvaluationDataError("paired comparison requires at least one gold document")

    def prediction_map(
        records: Sequence[PredictionRecord], *, kind: str
    ) -> dict[str, PredictionRecord]:
        result: dict[str, PredictionRecord] = {}
        for record in records:
            record.validate()
            if record.doc_id in result:
                raise EvaluationDataError(f"duplicate {kind} doc_id in paired comparison")
            result[record.doc_id] = record
        if set(result) != set(gold_by_id):
            raise EvaluationDataError(
                f"{kind} predictions must contain exactly the paired gold documents"
            )
        return result

    candidate_by_id = prediction_map(candidate_records, kind="candidate")
    baseline_by_id = prediction_map(baseline_records, kind="baseline")
    documents: list[PairedDocumentCounts] = []
    for doc_id in sorted(gold_by_id):
        gold = gold_by_id[doc_id]
        documents.append(
            PairedDocumentCounts(
                candidate=_strict_counts(
                    gold,
                    candidate_by_id[doc_id],
                    labels=label_set,
                ),
                baseline=_strict_counts(
                    gold,
                    baseline_by_id[doc_id],
                    labels=label_set,
                ),
            )
        )
    return PairedSuite(name=name, documents=tuple(documents))


def _sum_counts(documents: Sequence[PairedDocumentCounts]) -> PairedDocumentCounts:
    candidate = StrictCounts(0, 0, 0)
    baseline = StrictCounts(0, 0, 0)
    for document in documents:
        candidate += document.candidate
        baseline += document.baseline
    return PairedDocumentCounts(candidate=candidate, baseline=baseline)


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


def _resampled_delta(suites: Sequence[PairedSuite], rng: random.Random) -> float:
    candidate_tp = candidate_fp = candidate_fn = 0
    baseline_tp = baseline_fp = baseline_fn = 0
    for suite in suites:
        documents = suite.documents
        for _ in documents:
            sampled = documents[rng.randrange(len(documents))]
            candidate_tp += sampled.candidate.tp
            candidate_fp += sampled.candidate.fp
            candidate_fn += sampled.candidate.fn
            baseline_tp += sampled.baseline.tp
            baseline_fp += sampled.baseline.fp
            baseline_fn += sampled.baseline.fn
    candidate = strict_f1(StrictCounts(candidate_tp, candidate_fp, candidate_fn))
    baseline = strict_f1(StrictCounts(baseline_tp, baseline_fp, baseline_fn))
    return candidate - baseline


def paired_bootstrap_delta(
    suites: Sequence[PairedSuite],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Return a paired percentile interval and descriptive tail probabilities."""

    values = tuple(suites)
    if not values or any(not suite.documents for suite in values):
        raise ValueError("paired bootstrap requires non-empty suites")
    names = [suite.name for suite in values]
    if len(set(names)) != len(names):
        raise ValueError("paired bootstrap suite names must be unique")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be finite and in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    point_candidate = StrictCounts(0, 0, 0)
    point_baseline = StrictCounts(0, 0, 0)
    for suite in values:
        point = _sum_counts(suite.documents)
        point_candidate += point.candidate
        point_baseline += point.baseline
    candidate_f1 = strict_f1(point_candidate)
    baseline_f1 = strict_f1(point_baseline)
    point_delta = candidate_f1 - baseline_f1

    rng = random.Random(seed)
    deltas = [_resampled_delta(values, rng) for _ in range(samples)]
    tail = (1.0 - float(confidence)) / 2.0
    positive = sum(delta > 0.0 for delta in deltas)
    negative = sum(delta < 0.0 for delta in deltas)
    ties = samples - positive - negative
    win_probability = (positive + 0.5 * ties) / samples
    lower_tail = (negative + 0.5 * ties + 1.0) / (samples + 1.0)
    upper_tail = (positive + 0.5 * ties + 1.0) / (samples + 1.0)
    two_sided_approximation = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "documents": sum(len(suite.documents) for suite in values),
        "suite_document_counts": {suite.name: len(suite.documents) for suite in values},
        "candidate_strict_micro_f1": candidate_f1,
        "baseline_strict_micro_f1": baseline_f1,
        "delta_candidate_minus_baseline": point_delta,
        "confidence_interval": {
            "lower": _percentile(deltas, tail),
            "upper": _percentile(deltas, 1.0 - tail),
        },
        "bootstrap_outcomes": {
            "candidate_wins": positive,
            "ties": ties,
            "baseline_wins": negative,
        },
        "candidate_win_probability": win_probability,
        "two_sided_tail_probability_approximation": two_sided_approximation,
        "effective_samples": samples,
        "seed": seed,
    }


__all__ = [
    "PairedDocumentCounts",
    "PairedSuite",
    "StrictCounts",
    "build_paired_suite",
    "paired_bootstrap_delta",
    "strict_f1",
]
