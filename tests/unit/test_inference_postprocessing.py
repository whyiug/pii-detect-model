from __future__ import annotations

import pytest

from pii_zh.evaluation import PredictionRecord, Span
from pii_zh.inference.postprocessing import (
    LengthScorePolicy,
    SpanPostprocessingError,
    augment_with_scheme_urls,
    detect_scheme_urls,
    filter_prediction_records,
)


def test_scheme_url_detector_excludes_bare_domains_and_trims_punctuation() -> None:
    text = "入口 https://example.invalid/a?q=1，镜像www.example.invalid/x。裸域 example.invalid"
    spans = detect_scheme_urls(text)
    assert [text[span.start : span.end] for span in spans] == [
        "https://example.invalid/a?q=1",
        "www.example.invalid/x",
    ]


def test_length_score_policy_uses_stricter_short_threshold() -> None:
    records = (
        PredictionRecord(
            doc_id="a",
            spans=(
                Span(0, 2, "PERSON_NAME", 0.97),
                Span(3, 6, "PERSON_NAME", 0.70),
                Span(7, 10, "ADDRESS", 0.60),
            ),
        ),
    )
    filtered = filter_prediction_records(
        records,
        policy=LengthScorePolicy(
            short_max_characters=2,
            short_minimum_score=0.98,
            long_minimum_score=0.68,
        ),
    )
    assert [(span.start, span.end) for span in filtered[0].spans] == [(3, 6)]


def test_filter_requires_scores_and_augmentation_requires_exact_doc_set() -> None:
    with pytest.raises(SpanPostprocessingError, match="requires finite"):
        filter_prediction_records(
            (PredictionRecord(doc_id="a", spans=(Span(0, 3, "PERSON_NAME"),)),),
            policy=LengthScorePolicy(2, 0.98, 0.68),
        )
    with pytest.raises(SpanPostprocessingError, match="sets differ"):
        augment_with_scheme_urls(
            (PredictionRecord(doc_id="a", spans=()),),
            {"a": "无链接", "b": "https://example.invalid"},
        )
