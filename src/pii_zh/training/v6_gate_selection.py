"""Fail-closed raw v6 slice gates and feasibility-first checkpoint selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast


class V6GateSelectionError(ValueError):
    """Raised when checkpoint evidence does not satisfy the frozen schema."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise V6GateSelectionError(f"{field} must be a string-keyed object")
    return cast(Mapping[str, Any], value)


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise V6GateSelectionError(f"{field} must be a finite number >= {minimum}")
    return float(value)


def _rate(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if result > 1.0:
        raise V6GateSelectionError(f"{field} must be at most one")
    return result


def _count(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise V6GateSelectionError(f"{field} must be a {qualifier} integer")
    return value


def _strict_metric(slice_result: Mapping[str, Any], name: str, *, field: str) -> float:
    strict = _mapping(slice_result.get("strict_micro"), field=f"{field}.strict_micro")
    return _rate(strict.get(name), field=f"{field}.strict_micro.{name}")


def _pii_free_fpr(slice_result: Mapping[str, Any], *, field: str) -> float:
    pii_free = _mapping(slice_result.get("pii_free"), field=f"{field}.pii_free")
    return _rate(
        pii_free.get("false_positive_rate"),
        field=f"{field}.pii_free.false_positive_rate",
    )


def evaluate_v6_feasibility(
    *,
    slices: Mapping[str, Any],
    trainer_metrics: Mapping[str, Any],
    stage_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen raw gates without ranking or postprocessing."""

    required_slices = {
        "combined",
        "parent_short",
        "long_mixed",
        "long_negative",
        "long_low_density",
        "traditional",
    }
    if set(slices) != required_slices:
        raise V6GateSelectionError("raw slice evidence must contain exactly six slices")

    checks: dict[str, bool] = {}
    for name in ("parent_short", "long_mixed", "long_low_density", "traditional"):
        metrics = _mapping(slices.get(name), field=name)
        gate = _mapping(stage_gate.get(name), field=f"{name} gate")
        checks[f"{name}.minimum_f1"] = _strict_metric(
            metrics, "f1", field=name
        ) >= _rate(gate.get("minimum_f1"), field=f"{name}.minimum_f1")
        checks[f"{name}.minimum_recall"] = _strict_metric(
            metrics, "recall", field=name
        ) >= _rate(gate.get("minimum_recall"), field=f"{name}.minimum_recall")
        checks[f"{name}.maximum_predicted_gold_ratio"] = _number(
            metrics.get("predicted_gold_ratio"),
            field=f"{name}.predicted_gold_ratio",
        ) <= _number(
            gate.get("maximum_predicted_gold_ratio"),
            field=f"{name}.maximum_predicted_gold_ratio",
        )

    parent = _mapping(slices.get("parent_short"), field="parent_short")
    parent_gate = _mapping(stage_gate.get("parent_short"), field="parent_short gate")
    checks["parent_short.maximum_pii_free_fpr"] = _pii_free_fpr(
        parent, field="parent_short"
    ) <= _rate(
        parent_gate.get("maximum_pii_free_fpr"),
        field="parent_short.maximum_pii_free_fpr",
    )

    negative = _mapping(slices.get("long_negative"), field="long_negative")
    negative_gate = _mapping(stage_gate.get("long_negative"), field="long_negative gate")
    checks["long_negative.maximum_pii_free_fpr"] = _pii_free_fpr(
        negative, field="long_negative"
    ) <= _rate(
        negative_gate.get("maximum_pii_free_fpr"),
        field="long_negative.maximum_pii_free_fpr",
    )
    fragments = _mapping(
        negative.get("short_fragment_false_positives"),
        field="long_negative.short_fragment_false_positives",
    )
    checks["long_negative.maximum_short_fragment_false_positives"] = _count(
        fragments.get("length_le_2"),
        field="long_negative.short_fragment_false_positives.length_le_2",
    ) <= _count(
        negative_gate.get("maximum_short_fragment_false_positives"),
        field="long_negative.maximum_short_fragment_false_positives",
    )

    validation_gate = _mapping(
        stage_gate.get("training_validation"), field="training_validation gate"
    )
    flag_fields = {
        "require_tier0_recall_floor_met": "tier0_recall_floor_met",
        "require_tier1_recall_floor_met": "tier1_recall_floor_met",
        "require_calibration_recall_eligible": "calibration_recall_eligible",
    }
    for requirement, metric in flag_fields.items():
        if validation_gate.get(requirement) is not True:
            raise V6GateSelectionError(f"training_validation.{requirement} must stay true")
        value = _rate(trainer_metrics.get(metric), field=f"trainer_metrics.{metric}")
        if value not in {0.0, 1.0}:
            raise V6GateSelectionError(f"trainer_metrics.{metric} must be zero or one")
        checks[f"training_validation.{metric}"] = value == 1.0

    risk = _rate(
        trainer_metrics.get("risk_weighted_score"),
        field="trainer_metrics.risk_weighted_score",
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "risk_weighted_score": risk,
    }


def select_feasible_checkpoint(records: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the feasible max-risk checkpoint step; exact ties keep the earliest."""

    if not records:
        raise V6GateSelectionError("checkpoint evidence must not be empty")
    seen: set[int] = set()
    ranked: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        step = _count(
            record.get("global_step"), field=f"checkpoint[{index}].global_step", positive=True
        )
        if step in seen:
            raise V6GateSelectionError("checkpoint steps must be unique")
        seen.add(step)
        feasibility = _mapping(
            record.get("feasibility"), field=f"checkpoint[{index}].feasibility"
        )
        passed = feasibility.get("passed")
        if not isinstance(passed, bool):
            raise V6GateSelectionError("checkpoint feasibility.passed must be boolean")
        risk = _rate(
            feasibility.get("risk_weighted_score"),
            field=f"checkpoint[{index}].feasibility.risk_weighted_score",
        )
        if passed:
            ranked.append((risk, step))
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], -item[1]))[1]


__all__ = [
    "V6GateSelectionError",
    "evaluate_v6_feasibility",
    "select_feasible_checkpoint",
]
