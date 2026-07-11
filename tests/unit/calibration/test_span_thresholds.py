from __future__ import annotations

import pytest

from pii_zh.calibration import (
    CalibrationExample,
    RecallFloorError,
    select_span_thresholds,
)


def test_unemitted_gold_is_a_fixed_false_negative_at_every_threshold() -> None:
    examples = [
        CalibrationExample("PERSON_NAME", 0.9, True),
        CalibrationExample("PERSON_NAME", 0.8, False),
    ]

    selection = select_span_thresholds(
        examples,
        gold_support={"PERSON_NAME": 2},
        beta_by_entity={"PERSON_NAME": 2.0},
        minimum_recall={"PERSON_NAME": 0.5},
    )["PERSON_NAME"]

    assert selection.threshold == 0.9
    assert selection.un_emitted_gold_count == 1
    assert selection.true_positive == 1
    assert selection.false_positive == 0
    assert selection.false_negative == 1
    assert selection.recall == 0.5

    with pytest.raises(RecallFloorError, match="maximum emitted-candidate recall is 0.500000"):
        select_span_thresholds(
            examples,
            gold_support={"PERSON_NAME": 2},
            beta_by_entity={"PERSON_NAME": 2.0},
            minimum_recall={"PERSON_NAME": 0.75},
        )


def test_label_with_no_emitted_candidate_cannot_satisfy_positive_recall_floor() -> None:
    with pytest.raises(RecallFloorError, match="CN_RESIDENT_ID"):
        select_span_thresholds(
            [],
            gold_support={"CN_RESIDENT_ID": 1},
            beta_by_entity={"CN_RESIDENT_ID": 2.0},
            minimum_recall={"CN_RESIDENT_ID": 0.1},
        )
