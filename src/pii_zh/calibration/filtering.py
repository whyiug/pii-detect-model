"""Apply a frozen calibration bundle to raw, text-free span predictions."""

from __future__ import annotations

from collections.abc import Iterable

from pii_zh.evaluation.io import PredictionRecord, Span

from .thresholds import CalibrationBundle


def apply_calibration(
    records: Iterable[PredictionRecord],
    bundle: CalibrationBundle,
) -> list[PredictionRecord]:
    """Calibrate scores and retain spans meeting their entity threshold.

    Raw prediction scores are mandatory.  The returned scores are calibrated
    probabilities, matching the serving-time ``QwenPiiRecognizer`` contract.
    """

    output: list[PredictionRecord] = []
    for record in records:
        calibrated: list[Span] = []
        for span in record.spans:
            if span.score is None:
                raise ValueError("calibration requires a score on every prediction span")
            score = bundle.calibrate_score(span.label, span.score)
            if score >= bundle.threshold_for(span.label):
                calibrated.append(
                    Span(
                        start=span.start,
                        end=span.end,
                        label=span.label,
                        score=score,
                    )
                )
        output.append(PredictionRecord(doc_id=record.doc_id, spans=tuple(calibrated)))
    return output


__all__ = ["apply_calibration"]
