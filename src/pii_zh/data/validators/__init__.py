"""Deterministic validators for structured Chinese PII candidates."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    normalized: str | None
    reason: str

    def __bool__(self) -> bool:
        return self.valid


Validator = Callable[[str], ValidationResult]


_CN_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_ID_CHECK_CODES = "10X98765432"


def cn_resident_id_check_code(first_seventeen_digits: str) -> str:
    """Compute the GB 11643 MOD 11-2 checksum character."""

    if not re.fullmatch(r"\d{17}", first_seventeen_digits):
        raise ValueError("CN resident ID checksum input must contain exactly 17 digits")
    weighted_sum = sum(
        int(digit) * weight
        for digit, weight in zip(first_seventeen_digits, _CN_ID_WEIGHTS, strict=True)
    )
    return _CN_ID_CHECK_CODES[weighted_sum % 11]


def validate_cn_resident_id(value: str) -> ValidationResult:
    normalized = "".join(value.split()).upper() if isinstance(value, str) else None
    if normalized is None or not re.fullmatch(r"\d{17}[0-9X]", normalized):
        return ValidationResult(False, normalized, "expected 17 digits plus a digit/X checksum")
    if normalized[:6] == "000000":
        return ValidationResult(False, normalized, "region code cannot be all zero")
    try:
        birth = normalized[6:14]
        date(int(birth[:4]), int(birth[4:6]), int(birth[6:8]))
    except ValueError:
        return ValidationResult(False, normalized, "invalid birth date")
    if normalized[14:17] == "000":
        return ValidationResult(False, normalized, "sequence code cannot be 000")
    expected = cn_resident_id_check_code(normalized[:17])
    if normalized[-1] != expected:
        return ValidationResult(False, normalized, "MOD 11-2 checksum mismatch")
    return ValidationResult(True, normalized, "valid MOD 11-2 checksum")


def luhn_check_digit(number_without_check: str) -> str:
    if not isinstance(number_without_check, str) or not re.fullmatch(r"\d+", number_without_check):
        raise ValueError("Luhn checksum input must contain digits only")
    digits = [int(digit) for digit in number_without_check] + [0]
    parity = len(digits) % 2
    total = 0
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((-total) % 10)


def validate_luhn(value: str) -> ValidationResult:
    normalized = re.sub(r"[ -]", "", value) if isinstance(value, str) else None
    if normalized is None or not re.fullmatch(r"\d{12,19}", normalized):
        return ValidationResult(False, normalized, "expected 12-19 digits")
    if len(set(normalized)) == 1:
        return ValidationResult(False, normalized, "repeated digits are rejected")
    if luhn_check_digit(normalized[:-1]) != normalized[-1]:
        return ValidationResult(False, normalized, "Luhn checksum mismatch")
    return ValidationResult(True, normalized, "valid Luhn checksum")


_EMAIL_RE = re.compile(
    r"(?=.{3,254}\Z)[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}\Z"
)


def validate_email(value: str) -> ValidationResult:
    normalized = value.strip() if isinstance(value, str) else None
    if normalized is None or not _EMAIL_RE.fullmatch(normalized):
        return ValidationResult(False, normalized, "invalid basic email syntax")
    return ValidationResult(True, normalized, "valid basic email syntax")


_MOBILE_RE = re.compile(r"(?:\+?86)?1[3-9]\d{9}\Z")
_LANDLINE_RE = re.compile(r"0\d{2,3}\d{7,8}(?:转\d{1,6})?\Z")


def validate_phone(value: str) -> ValidationResult:
    if not isinstance(value, str):
        return ValidationResult(False, None, "phone must be a string")
    normalized = re.sub(r"[\s()-]", "", value)
    normalized = re.sub(r"(?:ext\.?|分机)", "转", normalized, flags=re.IGNORECASE)
    if _MOBILE_RE.fullmatch(normalized) or _LANDLINE_RE.fullmatch(normalized):
        return ValidationResult(True, normalized, "valid basic mainland phone syntax")
    return ValidationResult(False, normalized, "invalid basic mainland phone syntax")


def validate_ip(value: str) -> ValidationResult:
    normalized = value.strip() if isinstance(value, str) else None
    if normalized is None:
        return ValidationResult(False, None, "IP address must be a string")
    try:
        parsed = ipaddress.ip_address(normalized)
    except ValueError:
        return ValidationResult(False, normalized, "invalid IPv4/IPv6 address")
    return ValidationResult(True, parsed.compressed, f"valid IPv{parsed.version} address")


_MAC_RE = re.compile(r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\Z")


def validate_mac(value: str) -> ValidationResult:
    normalized = value.strip() if isinstance(value, str) else None
    if normalized is None or not _MAC_RE.fullmatch(normalized):
        return ValidationResult(False, normalized, "invalid six-octet MAC address")
    octets = re.split(r"[:-]", normalized)
    canonical = ":".join(octet.upper() for octet in octets)
    return ValidationResult(True, canonical, "valid six-octet MAC address")


_COORDINATE_RE = re.compile(
    r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[,，]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\Z"
)


def validate_coordinate(value: str) -> ValidationResult:
    if not isinstance(value, str):
        return ValidationResult(False, None, "coordinate must be a string")
    match = _COORDINATE_RE.fullmatch(value)
    if not match:
        return ValidationResult(False, value.strip(), "expected 'latitude,longitude'")
    latitude, longitude = (float(component) for component in match.groups())
    normalized = f"{latitude:.6f},{longitude:.6f}"
    if not -90 <= latitude <= 90:
        return ValidationResult(False, normalized, "latitude outside [-90, 90]")
    if not -180 <= longitude <= 180:
        return ValidationResult(False, normalized, "longitude outside [-180, 180]")
    return ValidationResult(True, normalized, "valid WGS84 coordinate range")


VALIDATORS: dict[str, Validator] = {
    "CN_RESIDENT_ID": validate_cn_resident_id,
    "BANK_CARD_NUMBER": validate_luhn,
    "EMAIL_ADDRESS": validate_email,
    "PHONE_NUMBER": validate_phone,
    "IP_ADDRESS": validate_ip,
    "MAC_ADDRESS": validate_mac,
    "GEO_COORDINATE": validate_coordinate,
}


def validate_entity_value(label: str, value: str) -> ValidationResult | None:
    """Validate supported structured labels; return ``None`` for semantic labels."""

    validator = VALIDATORS.get(label)
    return validator(value) if validator is not None else None


# Imported after ``ValidationResult`` and the base validators are defined: the
# community module intentionally reuses this package's stable validation type.
from .community import (  # noqa: E402
    COMMUNITY_STRUCTURED_VALIDATORS,
    validate_alipay_account,
    validate_bank_account_number,
    validate_community_vehicle_plate,
    validate_device_id,
    validate_driver_license_number,
    validate_employee_id,
    validate_medical_record_number,
    validate_passport_number,
    validate_qq_number,
    validate_secret,
    validate_social_security_number,
    validate_student_id,
    validate_wechat_id,
)

__all__ = [
    "VALIDATORS",
    "COMMUNITY_STRUCTURED_VALIDATORS",
    "ValidationResult",
    "cn_resident_id_check_code",
    "luhn_check_digit",
    "validate_cn_resident_id",
    "validate_community_vehicle_plate",
    "validate_coordinate",
    "validate_email",
    "validate_employee_id",
    "validate_entity_value",
    "validate_ip",
    "validate_luhn",
    "validate_mac",
    "validate_medical_record_number",
    "validate_passport_number",
    "validate_phone",
    "validate_qq_number",
    "validate_secret",
    "validate_social_security_number",
    "validate_student_id",
    "validate_wechat_id",
    "validate_alipay_account",
    "validate_bank_account_number",
    "validate_device_id",
    "validate_driver_license_number",
]
