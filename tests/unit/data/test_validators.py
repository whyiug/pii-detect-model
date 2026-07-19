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
from pii_zh.data.validators.cn_vehicle_plate import validate_cn_vehicle_license_plate


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


def test_conservative_mainland_vehicle_plate_validator_accepts_bound_v6_scope() -> None:
    cases = {
        "京A12345": "京A12345",
        "粤B12C34": "粤B12C34",
        "沪C1D2E3": "沪C1D2E3",
        "浙AD12345": "浙AD12345",
        "苏A12345F": "苏A12345F",
        "京A·12345": "京A12345",
        "粤b•12c34": "粤B12C34",
    }

    for candidate, normalized in cases.items():
        result = validate_cn_vehicle_license_plate(candidate)
        assert result.valid, (candidate, result.reason)
        assert result.normalized == normalized


def test_conservative_mainland_vehicle_plate_validator_rejects_unsupported_shapes() -> None:
    cases = (
        "岚A12345",
        "京I12345",
        "京A12I45",
        "京AABCD1",
        "京A-12345",
        "京A1234",
        "京A123456",
        "京AD1234A",
        "京A1234警",
        "京A12",
        "京Ａ１２３４５",
    )

    for candidate in cases:
        assert not validate_cn_vehicle_license_plate(candidate).valid, candidate
