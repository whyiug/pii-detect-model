"""Validation-only threshold selection with explicit un-emitted false negatives."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .thresholds import CalibrationExample


class RecallFloorError(ValueError):
    """Raised when emitted candidates cannot meet a configured recall floor."""


def _positive_number(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _probability(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class SpanThresholdSelection:
    """Counts and objective value at one selected per-entity threshold."""

    entity_type: str
    threshold: float
    beta: float
    recall_floor: float
    gold_support: int
    emitted_count: int
    emitted_match_count: int
    un_emitted_gold_count: int
    candidate_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f_beta: float

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "entity_type": self.entity_type,
            "threshold": self.threshold,
            "beta": self.beta,
            "recall_floor": self.recall_floor,
            "recall_floor_met": self.recall >= self.recall_floor,
            "gold_support": self.gold_support,
            "emitted_count": self.emitted_count,
            "emitted_match_count": self.emitted_match_count,
            "un_emitted_gold_fixed_fn": self.un_emitted_gold_count,
            "candidate_count": self.candidate_count,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f_beta": self.f_beta,
        }


def _metrics(
    rows: list[CalibrationExample], *, threshold: float, gold_support: int, beta: float
) -> tuple[int, int, int, float, float, float]:
    true_positive = sum(row.is_positive and row.score >= threshold for row in rows)
    false_positive = sum(not row.is_positive and row.score >= threshold for row in rows)
    # This is intentionally based on total gold support, not merely emitted
    # matches. Gold spans that the model never emitted remain fixed FNs at
    # every threshold and therefore cannot be "recovered" by threshold zero.
    false_negative = gold_support - true_positive
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = true_positive / gold_support if gold_support else 0.0
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    f_beta = (1.0 + beta_squared) * precision * recall / denominator if denominator else 0.0
    return true_positive, false_positive, false_negative, precision, recall, f_beta


def select_span_thresholds(
    examples: Iterable[CalibrationExample],
    *,
    gold_support: Mapping[str, int],
    beta: float = 1.0,
    beta_by_entity: Mapping[str, float] | None = None,
    minimum_recall: Mapping[str, float] | None = None,
) -> dict[str, SpanThresholdSelection]:
    """Select thresholds while treating un-emitted gold spans as fixed FNs.

    ``examples`` contains emitted spans only. An example is positive when its
    ``(document, start, end, label)`` exactly matches gold. ``gold_support`` is
    the independent count of all gold spans and is never converted into
    synthetic score-zero examples.
    """

    default_beta = _positive_number("beta", beta)
    normalized_support: dict[str, int] = {}
    for entity, count in gold_support.items():
        if not isinstance(entity, str) or not entity:
            raise ValueError("gold_support keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("gold_support values must be non-negative integers")
        normalized_support[entity] = count

    groups: dict[str, list[CalibrationExample]] = defaultdict(list)
    for example in examples:
        if not isinstance(example, CalibrationExample):
            raise TypeError("examples must contain CalibrationExample values")
        groups[example.entity_type].append(example)
    labels = sorted(set(normalized_support) | set(groups))
    if not labels:
        raise ValueError("gold_support and examples must not both be empty")

    selected: dict[str, SpanThresholdSelection] = {}
    for entity in labels:
        rows = groups.get(entity, [])
        support = normalized_support.get(entity, 0)
        emitted_matches = sum(row.is_positive for row in rows)
        if emitted_matches > support:
            raise ValueError(f"emitted exact matches exceed gold support for {entity}")
        entity_beta = _positive_number(
            f"beta for {entity}", (beta_by_entity or {}).get(entity, default_beta)
        )
        recall_floor = _probability(
            f"minimum recall for {entity}", (minimum_recall or {}).get(entity, 0.0)
        )
        candidates = sorted({0.0, 1.0, *(row.score for row in rows)})
        scored: list[
            tuple[
                tuple[float, float, float, float],
                tuple[int, int, int, float, float, float],
                float,
            ]
        ] = []
        for threshold in candidates:
            metrics = _metrics(
                rows,
                threshold=threshold,
                gold_support=support,
                beta=entity_beta,
            )
            precision, recall, f_beta = metrics[3], metrics[4], metrics[5]
            if recall >= recall_floor:
                scored.append(((f_beta, recall, precision, threshold), metrics, threshold))
        if not scored:
            reachable_recall = emitted_matches / support if support else 0.0
            raise RecallFloorError(
                f"no threshold for {entity} can satisfy recall floor; "
                f"maximum emitted-candidate recall is {reachable_recall:.6f}"
            )
        _, metrics, threshold = max(scored, key=lambda item: item[0])
        selected[entity] = SpanThresholdSelection(
            entity_type=entity,
            threshold=threshold,
            beta=entity_beta,
            recall_floor=recall_floor,
            gold_support=support,
            emitted_count=len(rows),
            emitted_match_count=emitted_matches,
            un_emitted_gold_count=support - emitted_matches,
            candidate_count=len(candidates),
            true_positive=metrics[0],
            false_positive=metrics[1],
            false_negative=metrics[2],
            precision=metrics[3],
            recall=metrics[4],
            f_beta=metrics[5],
        )
    return selected


__all__ = ["RecallFloorError", "SpanThresholdSelection", "select_span_thresholds"]
