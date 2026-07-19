from __future__ import annotations

import pytest

from pii_zh.presidio import ChinesePhraseContextEnhancer, RecognizerResult


def _result(
    text: str, value: str, *, metadata: dict[str, object] | None = None
) -> RecognizerResult:
    start = text.index(value)
    return RecognizerResult(
        entity_type="CN_ID_CARD",
        start=start,
        end=start + len(value),
        score=0.50,
        recognition_metadata=metadata or {},
    )


def test_positive_and_negative_chinese_field_context() -> None:
    enhancer = ChinesePhraseContextEnhancer(
        positive_context={"CN_ID_CARD": ["身份证号"]},
        negative_context={"CN_ID_CARD": ["订单号"]},
    )
    value = "990000\x3199001011239"
    positive_text = f"身份证号：{value}，姓名：测试"
    negative_text = f"订单号：{value}，状态：测试"

    positive = enhancer.enhance(positive_text, [_result(positive_text, value)])[0]
    negative = enhancer.enhance(negative_text, [_result(negative_text, value)])[0]

    assert positive.score == pytest.approx(0.65)
    assert negative.score == pytest.approx(0.30)
    assert positive.recognition_metadata["context_enhancement"]["positive_match_count"] == 1
    assert negative.recognition_metadata["context_enhancement"]["negative_match_count"] == 1
    assert "身份证号" not in repr(positive.recognition_metadata)
    assert "订单号" not in repr(negative.recognition_metadata)


def test_context_does_not_resurrect_validator_failure() -> None:
    enhancer = ChinesePhraseContextEnhancer(positive_context={"CN_ID_CARD": ["身份证号"]})
    value = "990000\x3199001011230"
    text = f"身份证号：{value}"
    invalid = _result(text, value, metadata={"validator_valid": False})

    enhanced = enhancer.enhance(text, [invalid])[0]

    assert enhanced.score == 0.50
    assert "context_enhancement" not in enhanced.recognition_metadata


def test_tabular_header_is_entity_context() -> None:
    enhancer = ChinesePhraseContextEnhancer(positive_context={"CN_ID_CARD": ["身份证号"]})
    value = "990000\x3199001011239"
    text = f"姓名\t身份证号\n测试\t{value}"

    enhanced = enhancer.enhance(text, [_result(text, value)])[0]

    assert enhanced.score == pytest.approx(0.65)
