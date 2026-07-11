from __future__ import annotations

import math

import pytest

from pii_zh.calibration import (
    CalibrationBundle,
    CalibrationExample,
    TemperatureScaler,
    fit_temperature,
    select_entity_thresholds,
)


def _nll(logits: list[list[float]], labels: list[int], temperature: float) -> float:
    losses = []
    for row, label in zip(logits, labels, strict=True):
        probabilities = TemperatureScaler(temperature).transform_logits(row)
        losses.append(-math.log(probabilities[label]))
    return sum(losses) / len(losses)


def test_temperature_fit_reduces_validation_nll() -> None:
    logits = [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 0.0]]
    labels = [0, 1, 1, 0]

    fitted = fit_temperature(logits, labels)

    assert fitted.temperature > 1.0
    assert _nll(logits, labels, fitted.temperature) < _nll(logits, labels, 1.0)


def test_entity_temperature_and_threshold_selection() -> None:
    bundle = CalibrationBundle(
        global_temperature=1.0,
        entity_temperatures={"PERSON": 2.0},
        entity_thresholds={"PERSON": 0.70},
    )
    assert bundle.calibrate_score("PERSON", 0.90) == pytest.approx(0.75)
    assert bundle.accepts("PERSON", 0.90)

    examples = [
        CalibrationExample("PERSON", 0.90, True),
        CalibrationExample("PERSON", 0.70, True),
        CalibrationExample("PERSON", 0.60, False),
        CalibrationExample("PERSON", 0.20, False),
        CalibrationExample("CN_ID_CARD", 0.80, True),
        CalibrationExample("CN_ID_CARD", 0.75, False),
    ]
    thresholds = select_entity_thresholds(
        examples, beta_by_entity={"CN_ID_CARD": 2.0}, minimum_recall={"CN_ID_CARD": 1.0}
    )

    assert thresholds == {"CN_ID_CARD": 0.8, "PERSON": 0.7}
