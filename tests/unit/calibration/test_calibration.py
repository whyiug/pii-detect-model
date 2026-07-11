from __future__ import annotations

import math

import pytest

from pii_zh.calibration import (
    CalibrationBundle,
    CalibrationExample,
    TemperatureScaler,
    apply_calibration,
    fit_temperature,
    select_entity_thresholds,
)
from pii_zh.evaluation import PredictionRecord, Span


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


def test_apply_calibration_filters_and_persists_calibrated_scores() -> None:
    bundle = CalibrationBundle(
        global_temperature=2.0,
        entity_thresholds={"PERSON_NAME": 0.70},
        default_threshold=0.60,
    )
    records = [
        PredictionRecord(
            doc_id="one",
            spans=(
                Span(0, 2, "PERSON_NAME", 0.90),
                Span(3, 8, "PHONE_NUMBER", 0.60),
            ),
        )
    ]

    calibrated = apply_calibration(records, bundle)

    assert len(calibrated[0].spans) == 1
    assert calibrated[0].spans[0].label == "PERSON_NAME"
    assert calibrated[0].spans[0].score == pytest.approx(0.75)


def test_apply_calibration_rejects_unscored_spans() -> None:
    records = [PredictionRecord(doc_id="one", spans=(Span(0, 2, "PERSON_NAME"),))]

    with pytest.raises(ValueError, match="score on every prediction span"):
        apply_calibration(records, CalibrationBundle())
