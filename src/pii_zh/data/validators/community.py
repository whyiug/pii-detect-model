"""Format-only validators for the opt-in community 24-class model profile.

These validators answer one deliberately narrow question: is a model span a
plausible value for the claimed structured entity?  They do not prove that an
identifier was issued, belongs to a person, or should be redacted.  Tenant
policy and semantic context remain separate service stages.

The accepted grammars include both the repository's visibly synthetic
sentinels and common deployment formats.  Keeping the sentinel branches
explicit makes the end-to-end synthetic release suite auditable without
weakening the historical ``cn_common_v6`` vehicle-plate validator.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from types import MappingProxyType

from . import (
    ValidationResult,
    validate_cn_resident_id,
    validate_coordinate,
    validate_email,
    validate_ip,
    validate_luhn,
    validate_mac,
    validate_phone,
)
from .cn_vehicle_plate import validate_cn_vehicle_license_plate

CommunityValidator = Callable[[str], ValidationResult]

_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/@:+-]{2,62}[A-Za-z0-9])?\Z")
_PASSPORT_RE = re.compile(r"(?:[EGPDS]\d{7,8}|[A-Z]{1,2}[A-Z0-9]{6,9})\Z")
_SYNTHETIC_PASSPORT_RE = re.compile(r"PZ-?[A-Z0-9]{8}\Z")
_SYNTHETIC_DRIVER_RE = re.compile(r"(?:DL[A-Z0-9]{10}|DL-ZZ-[A-Z0-9]{8})\Z")
_SOCIAL_SECURITY_RE = re.compile(r"[0-9A-Z]{8,24}\Z")
_BANK_ACCOUNT_RE = re.compile(r"\d{12,30}\Z")
_EMPLOYEE_RE = re.compile(
    r"(?:E\d{4,16}|DPT-[A-Z0-9]{2,12}-[A-Z0-9]{4,16}|STAFF/[A-Z0-9]{2,12}/[A-Z0-9]{4,16})\Z"
)
_STUDENT_RE = re.compile(r"(?:S\d{6,20}|[A-Z]{0,4}\d{6,20})\Z")
_MEDICAL_RECORD_RE = re.compile(r"(?:M(?:R)?[-/]?[A-Z0-9]{6,24}|[A-Z]{1,4}\d{6,20})\Z")
_WECHAT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{5,19}\Z")
_QQ_RE = re.compile(r"[1-9]\d{4,11}\Z")
_DEVICE_SENTINEL_RE = re.compile(r"DEV-[A-Z0-9]{8,32}\Z")
_MEID_RE = re.compile(r"[0-9A-F]{14}\Z")
_UUID_RE = re.compile(
    r"[0-9A-F]{8}-[0-9A-F]{4}-[1-5][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}\Z"
)
_SECRET_PREFIX_RE = re.compile(
    r"(?:key_|sk[-_]|gh[pousr]_|AKIA|ASIA|xox[baprs]-)[A-Za-z0-9_./+=-]{12,252}\Z"
)
_SYNTHETIC_PLATE_RE = re.compile(r"岚[A-Z](?:[·-]?[A-Z0-9]{5,8})\Z")


def _string(
    value: object, *, kind: str, upper: bool = False
) -> tuple[str | None, ValidationResult | None]:
    if not isinstance(value, str):
        return None, ValidationResult(False, None, f"{kind} must be a string")
    normalized = value.strip()
    if upper:
        normalized = normalized.upper()
    if not normalized:
        return normalized, ValidationResult(False, normalized, f"{kind} must not be empty")
    return normalized, None


def _format_result(valid: bool, normalized: str, *, kind: str) -> ValidationResult:
    return ValidationResult(
        valid,
        normalized,
        f"plausible {kind} format" if valid else f"invalid {kind} format",
    )


def validate_passport_number(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="passport number", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    valid = bool(
        _SYNTHETIC_PASSPORT_RE.fullmatch(normalized)
        or _PASSPORT_RE.fullmatch(normalized)
    )
    return _format_result(valid, normalized, kind="passport number")


def validate_driver_license_number(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="driver-license number", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    if _SYNTHETIC_DRIVER_RE.fullmatch(normalized):
        return _format_result(True, normalized, kind="synthetic driver-license number")
    resident = validate_cn_resident_id(normalized)
    if resident.valid:
        return ValidationResult(
            True,
            resident.normalized,
            "valid resident-ID-shaped driver license",
        )
    # Some deployments use a locally issued alphanumeric license namespace.
    valid = bool(re.fullmatch(r"[A-Z]{1,4}[-/]?[A-Z0-9]{6,20}", normalized))
    return _format_result(valid, normalized, kind="driver-license number")


def validate_social_security_number(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="social-security number", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    normalized = re.sub(r"[ -]", "", normalized)
    valid = bool(_SOCIAL_SECURITY_RE.fullmatch(normalized)) and not (
        len(set(normalized)) == 1
    )
    return _format_result(valid, normalized, kind="social-security number")


def validate_bank_account_number(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="bank-account number")
    if error is not None:
        return error
    assert normalized is not None
    normalized = re.sub(r"[ -]", "", normalized)
    valid = bool(_BANK_ACCOUNT_RE.fullmatch(normalized)) and len(set(normalized)) > 1
    return _format_result(valid, normalized, kind="bank-account number")


def validate_employee_id(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="employee ID", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    valid = bool(_EMPLOYEE_RE.fullmatch(normalized))
    return _format_result(valid, normalized, kind="employee ID")


def validate_student_id(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="student ID", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    valid = bool(_STUDENT_RE.fullmatch(normalized)) and any(
        character.isdigit() for character in normalized
    )
    return _format_result(valid, normalized, kind="student ID")


def validate_medical_record_number(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="medical-record number", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    valid = bool(_MEDICAL_RECORD_RE.fullmatch(normalized))
    return _format_result(valid, normalized, kind="medical-record number")


def validate_wechat_id(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="WeChat ID")
    if error is not None:
        return error
    assert normalized is not None
    return _format_result(bool(_WECHAT_RE.fullmatch(normalized)), normalized, kind="WeChat ID")


def validate_qq_number(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="QQ number")
    if error is not None:
        return error
    assert normalized is not None
    return _format_result(bool(_QQ_RE.fullmatch(normalized)), normalized, kind="QQ number")


def validate_alipay_account(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="Alipay account")
    if error is not None:
        return error
    assert normalized is not None
    email = validate_email(normalized)
    if email.valid:
        return ValidationResult(True, email.normalized, "valid email-shaped Alipay account")
    phone = validate_phone(normalized)
    if phone.valid:
        return ValidationResult(True, phone.normalized, "valid phone-shaped Alipay account")
    return _format_result(False, normalized, kind="Alipay account")


def validate_device_id(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="device ID", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    if _DEVICE_SENTINEL_RE.fullmatch(normalized):
        return _format_result(True, normalized, kind="synthetic device ID")
    if _UUID_RE.fullmatch(normalized) or _MEID_RE.fullmatch(normalized):
        return _format_result(True, normalized, kind="device ID")
    if re.fullmatch(r"\d{15}", normalized):
        imei = validate_luhn(normalized)
        return ValidationResult(imei.valid, imei.normalized, "IMEI " + imei.reason)
    valid = bool(_TOKEN_RE.fullmatch(normalized)) and any(
        character.isalpha() for character in normalized
    ) and any(character.isdigit() for character in normalized)
    return _format_result(valid, normalized, kind="device serial")


def validate_secret(value: str) -> ValidationResult:
    normalized, error = _string(value, kind="secret")
    if error is not None:
        return error
    assert normalized is not None
    if not 16 <= len(normalized) <= 256:
        return _format_result(False, normalized, kind="secret")
    prefixed = bool(_SECRET_PREFIX_RE.fullmatch(normalized))
    character_classes = sum(
        bool(re.search(pattern, normalized))
        for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[_./+=-]")
    )
    valid = prefixed or (bool(_TOKEN_RE.fullmatch(normalized)) and character_classes >= 3)
    return _format_result(valid, normalized, kind="secret")


def validate_community_vehicle_plate(value: str) -> ValidationResult:
    strict = validate_cn_vehicle_license_plate(value)
    if strict.valid:
        return strict
    normalized, error = _string(value, kind="vehicle plate", upper=True)
    if error is not None:
        return error
    assert normalized is not None
    if _SYNTHETIC_PLATE_RE.fullmatch(normalized):
        return ValidationResult(True, normalized, "valid visibly synthetic vehicle-plate sentinel")
    return strict


COMMUNITY_STRUCTURED_VALIDATORS = MappingProxyType(
    {
        "PHONE_NUMBER": validate_phone,
        "EMAIL_ADDRESS": validate_email,
        "CN_RESIDENT_ID": validate_cn_resident_id,
        "PASSPORT_NUMBER": validate_passport_number,
        "DRIVER_LICENSE_NUMBER": validate_driver_license_number,
        "CN_SOCIAL_SECURITY_NUMBER": validate_social_security_number,
        "BANK_CARD_NUMBER": validate_luhn,
        "BANK_ACCOUNT_NUMBER": validate_bank_account_number,
        "CN_VEHICLE_LICENSE_PLATE": validate_community_vehicle_plate,
        "EMPLOYEE_ID": validate_employee_id,
        "STUDENT_ID": validate_student_id,
        "MEDICAL_RECORD_NUMBER": validate_medical_record_number,
        "WECHAT_ID": validate_wechat_id,
        "QQ_NUMBER": validate_qq_number,
        "ALIPAY_ACCOUNT": validate_alipay_account,
        "IP_ADDRESS": validate_ip,
        "MAC_ADDRESS": validate_mac,
        "DEVICE_ID": validate_device_id,
        "GEO_COORDINATE": validate_coordinate,
        "SECRET": validate_secret,
    }
)


__all__ = [
    "COMMUNITY_STRUCTURED_VALIDATORS",
    "CommunityValidator",
    "validate_alipay_account",
    "validate_bank_account_number",
    "validate_community_vehicle_plate",
    "validate_device_id",
    "validate_driver_license_number",
    "validate_employee_id",
    "validate_medical_record_number",
    "validate_passport_number",
    "validate_qq_number",
    "validate_secret",
    "validate_social_security_number",
    "validate_student_id",
    "validate_wechat_id",
]
