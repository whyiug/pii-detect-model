"""Dependency-light temperature scaling for token and span probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be a finite value greater than zero")
    return value


def _validate_logits(logits: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in logits)
    if not values:
        raise ValueError("logits must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("logits must contain only finite values")
    return values


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    values = _validate_logits(logits)
    temperature = _validate_temperature(temperature)
    scaled = tuple(value / temperature for value in values)
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Apply one positive temperature to logits or a binary probability.

    ``transform_probability`` treats a span confidence as the positive-class
    probability of a binary logit.  This lets the same persisted temperature
    be consumed by the Presidio adapter without requiring NumPy or PyTorch at
    serving import time.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", _validate_temperature(self.temperature))

    def transform_logits(self, logits: Sequence[float]) -> list[float]:
        return _softmax(logits, self.temperature)

    def transform_batch(self, logits: Sequence[Sequence[float]]) -> list[list[float]]:
        return [self.transform_logits(row) for row in logits]

    def transform_probability(self, probability: float) -> float:
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("probability must be finite and between zero and one")
        if value in {0.0, 1.0}:
            return value
        if self.temperature == 1.0:
            return value
        logit = math.log(value) - math.log1p(-value)
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    @classmethod
    def fit(
        cls,
        logits: Sequence[Sequence[float]],
        labels: Sequence[int],
        **kwargs: float | int,
    ) -> TemperatureScaler:
        return fit_temperature(logits, labels, **kwargs)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    iterations: int = 96,
) -> TemperatureScaler:
    """Fit temperature by minimizing multiclass negative log likelihood.

    Optimization is performed in log-temperature space with a deterministic
    golden-section search.  The function intentionally consumes validation
    logits only; callers remain responsible for split isolation.
    """

    rows = tuple(_validate_logits(row) for row in logits)
    targets = tuple(labels)
    if not rows or len(rows) != len(targets):
        raise ValueError("logits and labels must be non-empty and have equal length")
    class_count = len(rows[0])
    if class_count < 2 or any(len(row) != class_count for row in rows):
        raise ValueError("all logit rows must have the same size of at least two")
    if any(isinstance(label, bool) or not isinstance(label, int) for label in targets):
        raise TypeError("labels must contain integer class indices")
    if any(label < 0 or label >= class_count for label in targets):
        raise ValueError("label index is outside the logit class range")
    minimum = _validate_temperature(minimum)
    maximum = _validate_temperature(maximum)
    if minimum >= maximum:
        raise ValueError("minimum temperature must be lower than maximum")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        total = 0.0
        for row, target in zip(rows, targets, strict=True):
            scaled = tuple(value / temperature for value in row)
            maximum_logit = max(scaled)
            log_denominator = maximum_logit + math.log(
                sum(math.exp(value - maximum_logit) for value in scaled)
            )
            total += log_denominator - scaled[target]
        return total / len(rows)

    left = math.log(minimum)
    right = math.log(maximum)
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - inverse_phi * (right - left)
    second = left + inverse_phi * (right - left)
    first_loss = nll(first)
    second_loss = nll(second)
    for _ in range(iterations):
        if first_loss <= second_loss:
            right = second
            second = first
            second_loss = first_loss
            first = right - inverse_phi * (right - left)
            first_loss = nll(first)
        else:
            left = first
            first = second
            first_loss = second_loss
            second = left + inverse_phi * (right - left)
            second_loss = nll(second)
    return TemperatureScaler(math.exp((left + right) / 2.0))


__all__ = ["TemperatureScaler", "fit_temperature"]
