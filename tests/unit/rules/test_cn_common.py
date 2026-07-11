from __future__ import annotations

from pii_zh.data.validators import cn_resident_id_check_code, luhn_check_digit
from pii_zh.rules import CnCommonRulePack


def _invalid_last(value: str) -> str:
    return value[:-1] + ("0" if value[-1] != "0" else "1")


def test_cn_id_and_bank_card_require_shared_checksum_validators() -> None:
    first_seventeen = "99000019900101123"
    valid_id = first_seventeen + cn_resident_id_check_code(first_seventeen)
    valid_card = "999999123456789" + luhn_check_digit("999999123456789")
    text = f"身份证号：{valid_id}，银行卡号：{valid_card}"

    matches = CnCommonRulePack().analyze(text)

    assert [(match.entity_type, text[match.start : match.end]) for match in matches] == [
        ("CN_ID_CARD", valid_id),
        ("BANK_CARD_NUMBER", valid_card),
    ]
    assert all(match.metadata["validator_valid"] is True for match in matches)

    invalid_text = f"身份证号：{_invalid_last(valid_id)}，银行卡号：{_invalid_last(valid_card)}"
    assert CnCommonRulePack().analyze(invalid_text) == []


def test_common_contact_rule_offsets_are_character_exact() -> None:
    text = "邮箱a@example.invalid，手机199-1234-5678"
    matches = CnCommonRulePack().analyze(text)

    assert [(match.entity_type, text[match.start : match.end]) for match in matches] == [
        ("EMAIL_ADDRESS", "a@example.invalid"),
        ("PHONE_NUMBER", "199-1234-5678"),
    ]


def test_contextual_identifier_fallbacks_require_semantic_context() -> None:
    employee_id = "E1234567890"
    vehicle_plate = "岚A·AB12CD3"
    text = f"员工{employee_id}已登记，所属车辆{vehicle_plate}已核验。"

    matches = CnCommonRulePack().analyze(text)

    assert ("EMPLOYEE_ID", employee_id) in {
        (match.entity_type, text[match.start : match.end]) for match in matches
    }
    assert ("CN_VEHICLE_LICENSE_PLATE", vehicle_plate) in {
        (match.entity_type, text[match.start : match.end]) for match in matches
    }
    assert CnCommonRulePack().analyze(f"构建产物{employee_id}，版本别名{vehicle_plate}") == []


def test_opaque_secret_rule_includes_the_full_self_identifying_prefix() -> None:
    secret = "key_abcdefghijklmnopqrstuvwxyz"
    text = f"会话令牌是{secret}，请立即轮换。"

    matches = CnCommonRulePack().analyze(text)

    assert ("SECRET", secret) in {
        (match.entity_type, text[match.start : match.end]) for match in matches
    }


def test_opaque_secret_rule_never_truncates_an_overlong_token() -> None:
    token = "key_" + "A" * 64 + "-" + "B" * 8
    text = f"凭据为{token}，请轮换。"

    matches = CnCommonRulePack().analyze(text, entities=["SECRET"])

    assert matches == []


def test_assignment_rule_returns_only_the_secret_value_capture() -> None:
    secret = "A1b2C3d4E5f6G7h8"
    text = f"api_key='{secret}'，由密钥平台托管。"

    matches = CnCommonRulePack().analyze(text)

    assert ("SECRET", secret) in {
        (match.entity_type, text[match.start : match.end]) for match in matches
    }
    assert all(text[match.start : match.end] != f"api_key='{secret}'" for match in matches)
