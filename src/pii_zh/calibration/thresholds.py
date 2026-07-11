"""Per-entity score calibration and validation-set threshold selection."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .temperature import TemperatureScaler


def _probability(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    entity_type: str
    score: float
    is_positive: bool

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, str) or not self.entity_type:
            raise ValueError("entity_type must be a non-empty string")
        object.__setattr__(self, "score", _probability("score", self.score))
        if not isinstance(self.is_positive, bool):
            raise TypeError("is_positive must be a bool")


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    """Serving contract for global/per-entity temperature and thresholds."""

    global_temperature: float = 1.0
    entity_temperatures: Mapping[str, float] = field(default_factory=dict)
    entity_thresholds: Mapping[str, float] = field(default_factory=dict)
    default_threshold: float = 0.5
    model_version: str | None = None
    calibration_version: str | None = None

    def __post_init__(self) -> None:
        TemperatureScaler(self.global_temperature)
        normalized_temperatures: dict[str, float] = {}
        for entity, temperature in self.entity_temperatures.items():
            if not isinstance(entity, str) or not entity:
                raise ValueError("entity temperature keys must be non-empty strings")
            normalized_temperatures[entity] = TemperatureScaler(temperature).temperature
        normalized_thresholds: dict[str, float] = {}
        for entity, threshold in self.entity_thresholds.items():
            if not isinstance(entity, str) or not entity:
                raise ValueError("entity threshold keys must be non-empty strings")
            normalized_thresholds[entity] = _probability("entity threshold", threshold)
        object.__setattr__(self, "entity_temperatures", normalized_temperatures)
        object.__setattr__(self, "entity_thresholds", normalized_thresholds)
        object.__setattr__(
            self, "default_threshold", _probability("default_threshold", self.default_threshold)
        )

    def calibrate_score(self, entity_type: str, raw_score: float) -> float:
        temperature = self.entity_temperatures.get(entity_type, self.global_temperature)
        return TemperatureScaler(temperature).transform_probability(raw_score)

    # ``calibrate`` is deliberately provided for small predictor adapters.
    calibrate = calibrate_score

    def threshold_for(self, entity_type: str) -> float:
        return self.entity_thresholds.get(entity_type, self.default_threshold)

    def accepts(self, entity_type: str, raw_score: float, *, calibrated: bool = False) -> bool:
        score = float(raw_score) if calibrated else self.calibrate_score(entity_type, raw_score)
        return score >= self.threshold_for(entity_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "global_temperature": self.global_temperature,
            "entity_temperatures": dict(sorted(self.entity_temperatures.items())),
            "entity_thresholds": dict(sorted(self.entity_thresholds.items())),
            "default_threshold": self.default_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationBundle:
        allowed = {
            "model_version",
            "calibration_version",
            "global_temperature",
            "entity_temperatures",
            "entity_thresholds",
            "default_threshold",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown calibration fields: {sorted(unknown)}")
        temperatures = value.get("entity_temperatures", {})
        thresholds = value.get("entity_thresholds", {})
        if not isinstance(temperatures, Mapping) or not isinstance(thresholds, Mapping):
            raise TypeError("entity_temperatures and entity_thresholds must be mappings")
        return cls(
            global_temperature=float(value.get("global_temperature", 1.0)),
            entity_temperatures={str(key): float(item) for key, item in temperatures.items()},
            entity_thresholds={str(key): float(item) for key, item in thresholds.items()},
            default_threshold=float(value.get("default_threshold", 0.5)),
            model_version=_optional_string(value.get("model_version"), "model_version"),
            calibration_version=_optional_string(
                value.get("calibration_version"), "calibration_version"
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CalibrationBundle:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise TypeError("calibration JSON root must be an object")
        return cls.from_dict(value)


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be null or a non-empty string")
    return value


def select_entity_thresholds(
    examples: Iterable[CalibrationExample],
    *,
    beta: float = 1.0,
    beta_by_entity: Mapping[str, float] | None = None,
    minimum_recall: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Select deterministic per-entity thresholds by validation-set F-beta.

    Candidate thresholds are observed scores plus the endpoints.  When scores
    tie, recall wins first, then precision, then the stricter threshold.  A
    per-entity minimum recall can be supplied for leakage-sensitive classes.
    """

    default_beta = float(beta)
    if not math.isfinite(default_beta) or default_beta <= 0.0:
        raise ValueError("beta must be finite and greater than zero")
    groups: dict[str, list[CalibrationExample]] = defaultdict(list)
    for example in examples:
        if not isinstance(example, CalibrationExample):
            raise TypeError("examples must contain CalibrationExample values")
        groups[example.entity_type].append(example)
    if not groups:
        raise ValueError("examples must not be empty")

    selected: dict[str, float] = {}
    for entity, rows in sorted(groups.items()):
        entity_beta = float((beta_by_entity or {}).get(entity, default_beta))
        if not math.isfinite(entity_beta) or entity_beta <= 0.0:
            raise ValueError(f"beta for {entity} must be finite and greater than zero")
        recall_floor = _probability(
            f"minimum recall for {entity}", (minimum_recall or {}).get(entity, 0.0)
        )
        positives = sum(row.is_positive for row in rows)
        if positives == 0:
            selected[entity] = 1.0
            continue
        candidates = sorted({0.0, 1.0, *(row.score for row in rows)})
        scored: list[tuple[tuple[float, float, float, float], float]] = []
        beta_squared = entity_beta * entity_beta
        for threshold in candidates:
            true_positive = sum(row.is_positive and row.score >= threshold for row in rows)
            false_positive = sum(not row.is_positive and row.score >= threshold for row in rows)
            false_negative = positives - true_positive
            recall = true_positive / (true_positive + false_negative)
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            if recall < recall_floor:
                continue
            denominator = beta_squared * precision + recall
            f_beta = (1.0 + beta_squared) * precision * recall / denominator if denominator else 0.0
            scored.append(((f_beta, recall, precision, threshold), threshold))
        if not scored:
            # Threshold zero always reaches all positives, so this only protects
            # against malformed future candidate generation.
            raise ValueError(f"no threshold for {entity} satisfies the recall constraint")
        selected[entity] = max(scored, key=lambda item: item[0])[1]
    return selected


__all__ = ["CalibrationBundle", "CalibrationExample", "select_entity_thresholds"]
