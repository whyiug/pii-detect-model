from __future__ import annotations

from pii_zh.fusion import DeterministicFusion, FusedDetection, apply_replacements


def _detection(
    entity: str,
    start: int,
    end: int,
    score: float,
    source: str,
    **metadata: object,
) -> FusedDetection:
    return FusedDetection(entity, start, end, score, source, metadata)


def test_exact_agreement_keeps_highest_score_and_records_sources() -> None:
    qwen = _detection("PERSON", 0, 2, 0.82, "qwen")
    rule = _detection("PERSON", 0, 2, 0.91, "customer_rule")

    fused = DeterministicFusion().fuse([qwen], [rule])

    assert len(fused) == 1
    assert fused[0].score == 0.91
    assert fused[0].metadata["sources"] == ["customer_rule", "qwen"]
    assert fused[0].metadata["agreement_count"] == 2


def test_specific_type_wins_same_span_and_containing_generic_span() -> None:
    specific = _detection("CN_ID_CARD", 4, 22, 0.70, "rule:cn_common", validator_valid=True)
    same_span_generic = _detection("GENERIC_ID", 4, 22, 0.99, "qwen")
    wider_generic = _detection("NUMBER", 2, 24, 1.0, "qwen")

    fused = DeterministicFusion().fuse([wider_generic, same_span_generic, specific])

    assert [(item.entity_type, item.start, item.end) for item in fused] == [("CN_ID_CARD", 4, 22)]


def test_validator_failure_is_suppressed_before_fusion() -> None:
    invalid = _detection("BANK_CARD_NUMBER", 0, 16, 1.0, "rule:cn_common", validator_valid=False)
    assert DeterministicFusion().fuse([invalid]) == []


def test_replacements_run_from_right_to_left_with_original_offsets() -> None:
    text = "张三电话19912345678"
    spans = [
        _detection("PERSON", 0, 2, 0.9, "qwen"),
        _detection("PHONE_NUMBER", 4, 15, 0.9, "rule:cn_common"),
    ]

    replaced = apply_replacements(
        text, spans, {"PERSON": "<姓名较长占位>", "PHONE_NUMBER": "<电话>"}
    )

    assert replaced == "<姓名较长占位>电话<电话>"
