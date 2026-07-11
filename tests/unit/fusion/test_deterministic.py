from __future__ import annotations

from pii_zh.data.validators import luhn_check_digit
from pii_zh.evaluation import PredictionRecord, Span
from pii_zh.fusion import (
    DeterministicFusion,
    FusedDetection,
    apply_replacements,
    suppress_invalid_structured_spans,
)
from pii_zh.rules import CnCommonRulePack


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


def test_vetted_rule_boundary_wins_over_a_higher_scoring_truncated_model_span() -> None:
    rule = _detection("SECRET", 4, 32, 0.94, "rule:cn_common", validator_valid=True)
    truncated_model = _detection("SECRET", 7, 32, 0.999, "qwen")

    fused = DeterministicFusion().fuse([truncated_model], [rule])

    assert [(item.entity_type, item.start, item.end) for item in fused] == [("SECRET", 4, 32)]


def test_overlong_secret_rule_cannot_replace_a_complete_model_span_with_a_prefix() -> None:
    token = "key_" + "A" * 64 + "-" + "B" * 8
    text = f"凭据为{token}，请轮换。"
    start = text.index(token)
    rules = CnCommonRulePack().analyze(text, entities=["SECRET"])
    model = _detection("SECRET", start, start + len(token), 0.999, "qwen")

    fused = DeterministicFusion().fuse(rules, [model])

    assert [(item.start, item.end) for item in fused] == [(start, start + len(token))]


def test_canonical_vehicle_and_contextual_account_beat_generic_model_types() -> None:
    vehicle = _detection(
        "VEHICLE_LICENSE_PLATE", 4, 14, 0.90, "rule:cn_common", validator_valid=True
    )
    person = _detection("PERSON_NAME", 4, 14, 0.999, "qwen")
    account = _detection("ALIPAY_ACCOUNT", 20, 39, 0.85, "qwen")
    email = _detection("EMAIL_ADDRESS", 20, 39, 0.99, "rule:cn_common")

    fused = DeterministicFusion().fuse([person, account], [vehicle, email])

    assert [(item.entity_type, item.start, item.end) for item in fused] == [
        ("VEHICLE_LICENSE_PLATE", 4, 14),
        ("ALIPAY_ACCOUNT", 20, 39),
    ]


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


def test_structured_refinement_suppresses_invalid_values_but_keeps_semantic_spans() -> None:
    text = "张三的卡号123456789012"
    record = PredictionRecord(
        doc_id="one",
        spans=(
            Span(0, 2, "PERSON_NAME", 0.9),
            Span(5, len(text), "BANK_CARD_NUMBER", 0.9),
        ),
    )

    refined, suppressed = suppress_invalid_structured_spans(record, text)

    assert [(span.start, span.end, span.label) for span in refined.spans] == [(0, 2, "PERSON_NAME")]
    assert suppressed == {"BANK_CARD_NUMBER": 1}


def test_identity_context_suppresses_card_surface_but_nearest_card_context_wins() -> None:
    card_prefix = "12345678901234567"
    card = card_prefix + luhn_check_digit(card_prefix)
    identity_text = f"身份证号：{card}"
    identity_start = identity_text.index(card)
    identity_record = PredictionRecord(
        doc_id="identity",
        spans=(Span(identity_start, identity_start + len(card), "BANK_CARD_NUMBER", 0.9),),
    )

    identity_refined, identity_suppressed = suppress_invalid_structured_spans(
        identity_record, identity_text
    )

    assert identity_refined.spans == ()
    assert identity_suppressed == {"BANK_CARD_NUMBER": 1}

    trailing_identity_text = f"校验未通过的号码{card}应作为无效证件串处理。"
    trailing_start = trailing_identity_text.index(card)
    trailing_record = PredictionRecord(
        doc_id="trailing-identity",
        spans=(Span(trailing_start, trailing_start + len(card), "BANK_CARD_NUMBER", 0.9),),
    )

    trailing_refined, trailing_suppressed = suppress_invalid_structured_spans(
        trailing_record, trailing_identity_text
    )

    assert trailing_refined.spans == ()
    assert trailing_suppressed == {"BANK_CARD_NUMBER": 1}

    card_text = f"身份证已核验，银行卡号：{card}"
    card_start = card_text.index(card)
    card_record = PredictionRecord(
        doc_id="card",
        spans=(Span(card_start, card_start + len(card), "BANK_CARD_NUMBER", 0.9),),
    )

    card_refined, card_suppressed = suppress_invalid_structured_spans(card_record, card_text)

    assert len(card_refined.spans) == 1
    assert card_suppressed == {}
