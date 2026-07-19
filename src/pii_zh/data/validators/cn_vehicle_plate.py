"""Successor-only conservative mainland civilian vehicle-plate validator."""

from __future__ import annotations

import re

from . import ValidationResult

_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
_LETTERS = "A-HJ-NP-Z"
_PLATE_RE = re.compile(
    rf"(?P<province>[{_PROVINCES}])"
    rf"(?P<office>[{_LETTERS}])"
    rf"(?P<separator>[·•]?)"
    rf"(?P<serial>[{_LETTERS}0-9]{{5}}|[DF][0-9]{{5}}|[0-9]{{5}}[DF])\Z"
)


def validate_cn_vehicle_license_plate(value: str) -> ValidationResult:
    """Validate the explicitly documented ``cn_common_v6`` plate subset."""

    if not isinstance(value, str):
        return ValidationResult(False, None, "vehicle plate must be a string")
    candidate = value.strip().upper()
    match = _PLATE_RE.fullmatch(candidate)
    if match is None:
        return ValidationResult(
            False,
            candidate,
            "unsupported or invalid civilian mainland vehicle plate syntax",
        )
    serial = match.group("serial")
    normalized = f"{match.group('province')}{match.group('office')}{serial}"
    if len(serial) == 5:
        digit_count = sum(character.isdigit() for character in serial)
        if digit_count < 3:
            return ValidationResult(
                False,
                normalized,
                "ordinary civilian serial must contain at least three digits",
            )
        return ValidationResult(True, normalized, "valid ordinary civilian mainland plate")
    if serial[0] in "DF":
        return ValidationResult(True, normalized, "valid small new-energy mainland plate")
    return ValidationResult(True, normalized, "valid large new-energy mainland plate")


__all__ = ["validate_cn_vehicle_license_plate"]
