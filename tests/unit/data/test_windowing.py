from __future__ import annotations

import pytest

from pii_zh.data.windowing import (
    WindowGoldSpan,
    WindowPlanningError,
    plan_entity_safe_windows,
)


def _character_offsets(length: int, *, specials: bool = False):
    offsets = [(index, index + 1) for index in range(length)]
    if specials:
        return [(0, 0), *offsets[:2], (2, 2), *offsets[2:], (0, 0)]
    return offsets


def test_plain_windows_have_deterministic_25_percent_overlap() -> None:
    first = plan_entity_safe_windows(
        text_length=10,
        offset_mapping=_character_offsets(10),
        gold_spans=[],
        max_tokens=4,
        overlap_fraction=0.25,
    )
    repeated = plan_entity_safe_windows(
        text_length=10,
        offset_mapping=_character_offsets(10),
        gold_spans=[],
        max_tokens=4,
        overlap_fraction=0.25,
    )

    assert repeated == first
    assert [(window.token_start, window.token_end) for window in first] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]
    assert [(window.start, window.end) for window in first] == [(0, 4), (3, 7), (6, 10)]
    assert all(window.token_count <= 4 for window in first)
    assert first[0].owner_start == 0
    assert first[-1].owner_end == 10


def test_overlap_rounding_matches_the_shared_inference_contract() -> None:
    windows = plan_entity_safe_windows(
        text_length=12,
        offset_mapping=_character_offsets(12),
        gold_spans=[],
        max_tokens=5,
        overlap_fraction=0.25,
    )

    assert [(window.token_start, window.token_end) for window in windows] == [
        (0, 5),
        (4, 9),
        (8, 12),
    ]


def test_zero_width_special_tokens_are_excluded_but_source_indices_are_preserved() -> None:
    windows = plan_entity_safe_windows(
        text_length=6,
        offset_mapping=_character_offsets(6, specials=True),
        gold_spans=[],
        max_tokens=4,
        overlap_fraction=0.25,
    )

    assert windows[0].source_token_indices == (1, 2, 4, 5)
    assert windows[1].source_token_indices == (5, 6, 7)
    assert all(0 not in window.source_token_indices for window in windows)
    assert all(3 not in window.source_token_indices for window in windows)
    assert all(8 not in window.source_token_indices for window in windows)


def test_tokens_sharing_one_character_are_kept_in_the_same_window() -> None:
    windows = plan_entity_safe_windows(
        text_length=3,
        offset_mapping=[(0, 1), (0, 1), (1, 2), (2, 3)],
        gold_spans=[],
        max_tokens=2,
        overlap_fraction=0.25,
    )

    assert [(window.token_start, window.token_end) for window in windows] == [
        (0, 2),
        (2, 4),
    ]
    assert [(window.start, window.end) for window in windows] == [(0, 1), (1, 3)]


def test_one_character_requiring_more_than_a_window_fails_closed() -> None:
    with pytest.raises(WindowPlanningError, match="indivisible token range"):
        plan_entity_safe_windows(
            text_length=2,
            offset_mapping=[(0, 1), (0, 1), (0, 1), (1, 2)],
            gold_spans=[],
            max_tokens=2,
            overlap_fraction=0.25,
        )


def test_entity_crossing_nominal_boundary_moves_windows_without_cutting_it() -> None:
    gold = WindowGoldSpan(3, 5, "PERSON_NAME")
    windows = plan_entity_safe_windows(
        text_length=10,
        offset_mapping=_character_offsets(10),
        gold_spans=[gold],
        max_tokens=4,
        overlap_fraction=0.25,
    )

    assert [(window.token_start, window.token_end) for window in windows[:2]] == [(0, 3), (2, 6)]
    owners = [window for window in windows if window.spans]
    assert len(owners) == 1
    owner = owners[0]
    span = owner.spans[0]
    assert owner.start <= gold.start < gold.end <= owner.end
    assert owner.owner_start <= gold.start < gold.end <= owner.owner_end
    assert (span.global_start, span.global_end) == (3, 5)
    assert (span.local_start, span.local_end) == (3 - owner.start, 5 - owner.start)

    for window in windows:
        assert not (window.start < gold.start < window.end < gold.end)
        assert not (gold.start < window.start < gold.end < window.end)


def test_entity_near_document_end_does_not_create_a_redundant_window() -> None:
    windows = plan_entity_safe_windows(
        text_length=9,
        offset_mapping=_character_offsets(9),
        gold_spans=[WindowGoldSpan(5, 9, "ADDRESS")],
        max_tokens=5,
        overlap_fraction=0.25,
    )

    assert [(window.token_start, window.token_end) for window in windows] == [
        (0, 5),
        (4, 9),
    ]
    assert all(
        current.token_end < following.token_end
        for current, following in zip(windows, windows[1:], strict=False)
    )
    assert sum(len(window.spans) for window in windows) == 1


def test_multiple_spans_have_exactly_one_owner_and_local_offsets() -> None:
    spans = [
        WindowGoldSpan(1, 3, "PERSON_NAME"),
        WindowGoldSpan(7, 10, "ADDRESS"),
        WindowGoldSpan(13, 15, "DATE_OF_BIRTH"),
    ]
    windows = plan_entity_safe_windows(
        text_length=16,
        offset_mapping=_character_offsets(16),
        gold_spans=reversed(spans),
        max_tokens=6,
        overlap_fraction=0.25,
    )

    owned = [span for window in windows for span in window.spans]
    assert len(owned) == len(spans)
    assert [span.span_index for span in owned] == [0, 1, 2]
    assert {(span.global_start, span.global_end, span.label) for span in owned} == {
        (span.start, span.end, span.label) for span in spans
    }
    for window in windows:
        for span in window.spans:
            assert span.local_start == span.global_start - window.start
            assert span.local_end == span.global_end - window.start
            assert window.owner_start <= span.global_start
            assert span.global_end <= window.owner_end


def test_gold_input_order_does_not_change_the_plan() -> None:
    left = WindowGoldSpan(2, 4, "PERSON_NAME")
    right = WindowGoldSpan(8, 10, "ADDRESS")
    forward = plan_entity_safe_windows(
        text_length=12,
        offset_mapping=_character_offsets(12),
        gold_spans=[left, right],
        max_tokens=5,
        overlap_fraction=0.25,
    )
    reverse = plan_entity_safe_windows(
        text_length=12,
        offset_mapping=_character_offsets(12),
        gold_spans=[right, left],
        max_tokens=5,
        overlap_fraction=0.25,
    )
    assert reverse == forward


def test_entity_longer_than_one_window_fails_closed() -> None:
    with pytest.raises(WindowPlanningError, match="exceeds one token window"):
        plan_entity_safe_windows(
            text_length=10,
            offset_mapping=_character_offsets(10),
            gold_spans=[WindowGoldSpan(1, 8, "PERSON_NAME")],
            max_tokens=4,
            overlap_fraction=0.25,
        )


def test_token_coverage_must_be_complete_and_monotonic() -> None:
    with pytest.raises(WindowPlanningError, match="character gap"):
        plan_entity_safe_windows(
            text_length=4,
            offset_mapping=[(0, 1), (2, 3), (3, 4)],
            gold_spans=[],
            max_tokens=4,
            overlap_fraction=0.25,
        )
    with pytest.raises(WindowPlanningError, match="monotonic"):
        plan_entity_safe_windows(
            text_length=4,
            offset_mapping=[(0, 2), (1, 1), (1, 3), (0, 4)],
            gold_spans=[],
            max_tokens=4,
            overlap_fraction=0.25,
        )


def test_gold_with_incomplete_token_coverage_fails_without_sensitive_reflection() -> None:
    sensitive_label = "PRIVATE-LABEL-MUST-NOT-APPEAR"
    with pytest.raises(WindowPlanningError) as error:
        plan_entity_safe_windows(
            text_length=4,
            offset_mapping=[(0, 2), (2, 4)],
            gold_spans=[WindowGoldSpan(0, 5, sensitive_label)],
            max_tokens=2,
            overlap_fraction=0.25,
        )
    assert sensitive_label not in str(error.value)


def test_duplicate_and_invalid_gold_spans_are_rejected() -> None:
    duplicate = WindowGoldSpan(1, 2, "PERSON_NAME")
    with pytest.raises(WindowPlanningError, match="duplicate"):
        plan_entity_safe_windows(
            text_length=3,
            offset_mapping=_character_offsets(3),
            gold_spans=[duplicate, duplicate],
            max_tokens=3,
            overlap_fraction=0.25,
        )
    with pytest.raises(WindowPlanningError, match="out of bounds"):
        plan_entity_safe_windows(
            text_length=3,
            offset_mapping=_character_offsets(3),
            gold_spans=[WindowGoldSpan(2, 4, "PERSON_NAME")],
            max_tokens=3,
            overlap_fraction=0.25,
        )


@pytest.mark.parametrize("fraction", [0.0, 0.19, 0.26, 1.0, float("nan")])
def test_overlap_fraction_outside_shared_contract_is_rejected(fraction: float) -> None:
    with pytest.raises(WindowPlanningError, match="between 0.20 and 0.25"):
        plan_entity_safe_windows(
            text_length=2,
            offset_mapping=_character_offsets(2),
            gold_spans=[],
            max_tokens=2,
            overlap_fraction=fraction,
        )


def test_empty_document_with_only_special_offsets_has_no_windows() -> None:
    assert (
        plan_entity_safe_windows(
            text_length=0,
            offset_mapping=[(0, 0), (0, 0)],
            gold_spans=[],
            max_tokens=4,
            overlap_fraction=0.25,
        )
        == ()
    )


def test_gold_iterable_failure_is_sanitized_for_empty_documents() -> None:
    def failing_spans():
        raise RuntimeError("PRIVATE-ENTITY-VALUE")
        yield WindowGoldSpan(0, 1, "PERSON_NAME")

    with pytest.raises(TypeError, match="gold spans must be iterable") as error:
        plan_entity_safe_windows(
            text_length=0,
            offset_mapping=[],
            gold_spans=failing_spans(),
            max_tokens=2,
            overlap_fraction=0.25,
        )
    assert "PRIVATE-ENTITY-VALUE" not in str(error.value)
