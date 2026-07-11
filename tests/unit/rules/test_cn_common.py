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
