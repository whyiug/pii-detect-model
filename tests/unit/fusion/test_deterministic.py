from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from pii_zh.data.validators import luhn_check_digit
from pii_zh.evaluation import PredictionRecord, Span
from pii_zh.fusion import (
    DeterministicFusion,
    FusedDetection,
    ReplacementSpan,
    apply_replacements,
    merge_replacement_coverage,
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


@pytest.mark.parametrize("trailing_digits_omitted", [0, 1, 2])
def test_contextual_bank_account_rule_beats_same_or_truncated_model_card_surface(
    trailing_digits_omitted: int,
) -> None:
    prefix = "62220212345678901"
    account = prefix + luhn_check_digit(prefix)
    text = f"收款账户：{account}"
    start = text.index(account)
    rules = CnCommonRulePack().analyze(text, entities=["BANK_ACCOUNT_NUMBER"])
    model_end = start + len(account) - trailing_digits_omitted
    model = _detection("BANK_CARD_NUMBER", start, model_end, 0.999, "qwen")

    fused = DeterministicFusion().fuse(rules, [model])

    assert [(item.entity_type, item.start, item.end) for item in fused] == [
        ("BANK_ACCOUNT_NUMBER", start, start + len(account))
    ]


def test_explicit_bank_card_rule_beats_same_boundary_model_account() -> None:
    prefix = "622202123456789"
    card = prefix + luhn_check_digit(prefix)
    text = f"工资账户绑定的银行卡号：{card}"
    start = text.index(card)
    rules = CnCommonRulePack().analyze(text, entities=["BANK_CARD_NUMBER"])
    model = _detection("BANK_ACCOUNT_NUMBER", start, start + len(card), 0.999, "qwen")

    fused = DeterministicFusion().fuse(rules, [model])

    assert [(item.entity_type, item.start, item.end) for item in fused] == [
        ("BANK_CARD_NUMBER", start, start + len(card))
    ]


def test_replacements_run_from_right_to_left_with_original_offsets() -> None:
    text = "张三电话19912\x3345678"
    spans = [
        _detection("PERSON", 0, 2, 0.9, "qwen"),
        _detection("PHONE_NUMBER", 4, 15, 0.9, "rule:cn_common"),
    ]

    replaced = apply_replacements(
        text, spans, {"PERSON": "<姓名较长占位>", "PHONE_NUMBER": "<电话>"}
    )

    assert replaced == "<姓名较长占位>电话<电话>"


def test_replacement_mapping_uses_one_get_lookup_without_membership_probe() -> None:
    class GetOnlyMapping(Mapping[str, str]):
        def __init__(self) -> None:
            self.lookup_count = 0

        def __getitem__(self, key: str) -> str:
            self.lookup_count += 1
            if key == "PERSON":
                return "<姓名>"
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(("PERSON",))

        def __len__(self) -> int:
            return 1

        def __contains__(self, key: object) -> bool:
            raise AssertionError(f"unexpected membership probe for {key!r}")

    replacements = GetOnlyMapping()
    replaced = apply_replacements("张三", [_detection("PERSON", 0, 2, 0.9, "qwen")], replacements)

    assert replaced == "<姓名>"
    assert replacements.lookup_count == 1


def test_replacement_coverage_merges_transitive_overlap_but_not_adjacency() -> None:
    spans = [
        _detection("PERSON", 0, 2, 0.9, "qwen"),
        _detection("PERSON", 1, 4, 0.8, "rule:cn_common"),
        _detection("PHONE_NUMBER", 3, 6, 0.9, "rule:cn_common"),
        _detection("SECRET", 6, 8, 0.9, "rule:cn_common"),
    ]

    assert merge_replacement_coverage(list(reversed(spans))) == [
        ReplacementSpan("PII", 0, 6),
        ReplacementSpan("SECRET", 6, 8),
    ]


def test_replacement_coverage_preserves_one_type_across_overlap() -> None:
    spans = [
        _detection("CN_ADDRESS", 1, 5, 0.8, "qwen"),
        _detection("CN_ADDRESS", 3, 8, 0.9, "rule:cn_common"),
    ]

    assert merge_replacement_coverage(spans) == [ReplacementSpan("CN_ADDRESS", 1, 8)]


@pytest.mark.parametrize(
    ("entity_type", "start", "end", "error"),
    [
        ("", 0, 1, TypeError),
        ("PERSON", True, 2, TypeError),
        ("PERSON", 0, False, TypeError),
        ("PERSON", 0.0, 2, TypeError),
        ("PERSON", -1, 2, ValueError),
        ("PERSON", 1, 1, ValueError),
        ("PERSON", 2, 1, ValueError),
    ],
)
def test_replacement_span_rejects_invalid_identity_or_offsets(
    entity_type: str,
    start: object,
    end: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ReplacementSpan(entity_type, start, end)  # type: ignore[arg-type]


def test_apply_replacements_rejects_span_beyond_the_input() -> None:
    with pytest.raises(ValueError, match="outside text"):
        apply_replacements("张三", [ReplacementSpan("PERSON", 0, 3)])


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


def test_identity_context_suppresses_student_id_fragments_inside_identity_numbers() -> None:
    identity_number = "123456\x3789012345678"
    identity_text = f"无效证件串：{identity_number}，需要人工复核。"
    run_start = identity_text.index(identity_number)
    fragment = PredictionRecord(
        doc_id="identity-fragment",
        spans=(Span(run_start + 8, run_start + 10, "STUDENT_ID", 0.9),),
    )

    refined, suppressed = suppress_invalid_structured_spans(fragment, identity_text)

    assert refined.spans == ()
    assert suppressed == {"STUDENT_ID": 1}

    student_text = f"学生学号：{identity_number}，已完成学籍登记。"
    student_start = student_text.index(identity_number)
    student = PredictionRecord(
        doc_id="student",
        spans=(Span(student_start, student_start + len(identity_number), "STUDENT_ID", 0.9),),
    )

    retained, retained_suppressed = suppress_invalid_structured_spans(student, student_text)

    assert len(retained.spans) == 1
    assert retained_suppressed == {}


def test_account_context_suppresses_card_label_but_nearer_explicit_card_context_wins() -> None:
    prefix = "622202123456789"
    number = prefix + luhn_check_digit(prefix)
    account_text = f"收款账户：{number}"
    account_start = account_text.index(number)
    account_record = PredictionRecord(
        doc_id="account",
        spans=(Span(account_start, account_start + len(number), "BANK_CARD_NUMBER", 0.9),),
    )

    account_refined, account_suppressed = suppress_invalid_structured_spans(
        account_record, account_text
    )

    assert account_refined.spans == ()
    assert account_suppressed == {"BANK_CARD_NUMBER": 1}

    card_text = f"工资账户绑定的银行卡号：{number}"
    card_start = card_text.index(number)
    card_record = PredictionRecord(
        doc_id="card-account-conflict",
        spans=(Span(card_start, card_start + len(number), "BANK_CARD_NUMBER", 0.9),),
    )

    card_refined, card_suppressed = suppress_invalid_structured_spans(card_record, card_text)

    assert len(card_refined.spans) == 1
    assert card_suppressed == {}


@pytest.mark.parametrize(
    ("text", "date_value"),
    [
        ("出生日期：1990-01-02", "1990-01-02"),
        ("会员生日为1990.01.02", "1990.01.02"),
        ("DOB=19900102", "19900102"),
        ("年龄核验所用日期：1990年1月2日", "1990年1月2日"),
    ],
)
def test_birth_date_refinement_requires_and_keeps_affirmative_semantics(
    text: str, date_value: str
) -> None:
    start = text.index(date_value)
    record = PredictionRecord(
        doc_id="affirmative-birth",
        spans=(Span(start, start + len(date_value), "DATE_OF_BIRTH", 0.9),),
    )

    refined, suppressed = suppress_invalid_structured_spans(record, text)

    assert len(refined.spans) == 1
    assert suppressed == {}


@pytest.mark.parametrize(
    ("text", "date_value"),
    [
        ("版本计划发布日期为2026-07-11", "2026-07-11"),
        ("业务日期：2026/07/11", "2026/07/11"),
        ("这不是出生信息，发布日期为2026-07-11", "2026-07-11"),
        ("非出生日期：2026-07-11", "2026-07-11"),
        ("出生日期不适用：2026-07-11", "2026-07-11"),
    ],
)
def test_birth_date_refinement_suppresses_business_dates_and_negated_birth_context(
    text: str, date_value: str
) -> None:
    start = text.index(date_value)
    record = PredictionRecord(
        doc_id="non-birth-date",
        spans=(Span(start, start + len(date_value), "DATE_OF_BIRTH", 0.9),),
    )

    refined, suppressed = suppress_invalid_structured_spans(record, text)

    assert refined.spans == ()
    assert suppressed == {"DATE_OF_BIRTH": 1}


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("QQ号码：123456789", "123456789"),
        ("备用QQ号为876543210", "876543210"),
        ("企鹅号 55667788", "55667788"),
        ("contact_qq_id=11223344", "11223344"),
    ],
)
def test_qq_refinement_keeps_affirmative_qq_semantics(text: str, number: str) -> None:
    start = text.index(number)
    record = PredictionRecord(
        doc_id="affirmative-qq",
        spans=(Span(start, start + len(number), "QQ_NUMBER", 0.9),),
    )

    refined, suppressed = suppress_invalid_structured_spans(record, text)

    assert len(refined.spans) == 1
    assert suppressed == {}


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("无效卡号校验值：6222021234567890", "6222021234567890"),
        ("普通业务编号：123456789", "123456789"),
        ("这不是QQ号码，校验值为123456789", "123456789"),
        ("QQ号码不适用：123456789", "123456789"),
    ],
)
def test_qq_refinement_suppresses_unrelated_and_negated_numeric_surfaces(
    text: str, number: str
) -> None:
    start = text.index(number)
    record = PredictionRecord(
        doc_id="non-qq-number",
        spans=(Span(start, start + len(number), "QQ_NUMBER", 0.9),),
    )

    refined, suppressed = suppress_invalid_structured_spans(record, text)

    assert refined.spans == ()
    assert suppressed == {"QQ_NUMBER": 1}
