from __future__ import annotations

from pii_zh.data.validators import (
    cn_resident_id_check_code,
    luhn_check_digit,
    validate_cn_resident_id,
    validate_coordinate,
    validate_email,
    validate_ip,
    validate_luhn,
    validate_mac,
    validate_phone,
)


def test_cn_resident_id_mod_11_2_and_invalid_date() -> None:
    first = "99000019900101123"
    value = first + cn_resident_id_check_code(first)
    assert validate_cn_resident_id(value).valid
    wrong = value[:-1] + ("0" if value[-1] != "0" else "1")
    assert not validate_cn_resident_id(wrong).valid
    invalid_date_first = "99000019901340123"
    invalid_date = invalid_date_first + cn_resident_id_check_code(invalid_date_first)
    assert not validate_cn_resident_id(invalid_date).valid


def test_luhn_generation_and_validation() -> None:
    without_check = "999999123456789"
    value = without_check + luhn_check_digit(without_check)
    assert validate_luhn(value).valid
    assert not validate_luhn(value[:-1] + str((int(value[-1]) + 1) % 10)).valid


def test_basic_contact_and_network_validators() -> None:
    assert validate_email("pii-test@example.invalid").valid
    assert not validate_email("missing-at.invalid").valid
    assert validate_phone("+86 199-1234-5678").valid
    assert validate_phone("010-12345678").valid
    assert not validate_phone("12345").valid
    assert validate_ip("192.0.2.10").valid
    assert validate_ip("2001:db8::1").valid
    assert not validate_ip("999.1.1.1").valid
    assert validate_mac("02:00:00:AA:BB:CC").valid
    assert not validate_mac("02:00:00:AA:BB").valid
    assert validate_coordinate("31.2,121.5").valid
    assert not validate_coordinate("91,121.5").valid
