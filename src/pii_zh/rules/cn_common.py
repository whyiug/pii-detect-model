"""Deterministic common-format rules for mainland Chinese PII."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from re import Pattern
from typing import Any

from pii_zh.data.validators import validate_cn_resident_id, validate_entity_value


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    entity_type: str
    pattern_id: str
    pattern: Pattern[str]
    score: float
    validator_label: str | None = None
    required_context: tuple[str, ...] = ()
    already_masked: bool = False
    value_group: str | int | None = None


@dataclass(frozen=True, slots=True)
class RuleMatch:
    entity_type: str
    start: int
    end: int
    score: float
    pattern_id: str
    source: str = "rule:cn_common"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError("rule match must use a non-empty left-closed/right-open span")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("rule match score must be between zero and one")


def _rule(
    entity_type: str,
    pattern_id: str,
    regex: str,
    score: float,
    *,
    validator_label: str | None = None,
    required_context: Sequence[str] = (),
    already_masked: bool = False,
    value_group: str | int | None = None,
    flags: int = 0,
) -> RuleDefinition:
    return RuleDefinition(
        entity_type=entity_type,
        pattern_id=pattern_id,
        pattern=re.compile(regex, flags),
        score=score,
        validator_label=validator_label,
        required_context=tuple(required_context),
        already_masked=already_masked,
        value_group=value_group,
    )


CN_COMMON_RULES: tuple[RuleDefinition, ...] = (
    _rule(
        "CN_ID_CARD",
        "cn_resident_id_mod_11_2",
        r"(?<!\d)(?:\d\s*){17}[0-9Xx](?![0-9A-Za-z])",
        0.99,
        validator_label="CN_RESIDENT_ID",
    ),
    _rule(
        "PHONE_NUMBER",
        "cn_mobile",
        r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d(?:[\s-]?\d){8}(?!\d)",
        0.90,
        validator_label="PHONE_NUMBER",
    ),
    _rule(
        "PHONE_NUMBER",
        "cn_landline_extension",
        r"(?<!\d)0\d{2,3}[\s-]?\d{7,8}(?:(?:转|分机|\s*ext\.?\s*)\d{1,6})?(?!\d)",
        0.88,
        validator_label="PHONE_NUMBER",
        flags=re.IGNORECASE,
    ),
    _rule(
        "EMAIL_ADDRESS",
        "email_basic",
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}",
        0.92,
        validator_label="EMAIL_ADDRESS",
    ),
    _rule(
        "BANK_CARD_NUMBER",
        "bank_card_luhn",
        r"(?<!\d)(?:\d[ -]?){11,18}\d(?!\d)",
        0.99,
        validator_label="BANK_CARD_NUMBER",
    ),
    _rule(
        "IP_ADDRESS",
        "ipv4",
        r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
        0.94,
        validator_label="IP_ADDRESS",
    ),
    _rule(
        "IP_ADDRESS",
        "ipv6",
        r"(?<![0-9A-Fa-f:])[0-9A-Fa-f]*:[0-9A-Fa-f:]+(?![0-9A-Fa-f:])",
        0.94,
        validator_label="IP_ADDRESS",
    ),
    _rule(
        "MAC_ADDRESS",
        "mac_48bit",
        r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])",
        0.98,
        validator_label="MAC_ADDRESS",
    ),
    _rule(
        "GEO_COORDINATE",
        "wgs84_coordinate",
        r"(?<![\d.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*[,，]\s*"
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\d.])",
        0.88,
        validator_label="GEO_COORDINATE",
    ),
    _rule(
        "PASSPORT_NUMBER",
        "cn_passport_common",
        r"(?<![A-Z0-9])(?:E\d{8}|[GDESP]\d{7,8})(?![A-Z0-9])",
        0.78,
        required_context=("护照", "passport"),
        flags=re.IGNORECASE,
    ),
    _rule(
        "DRIVER_LICENSE_NUMBER",
        "cn_driver_license_context",
        r"(?<!\d)\d{17}[0-9Xx](?![0-9A-Za-z])",
        0.94,
        validator_label="CN_RESIDENT_ID",
        required_context=("驾驶证", "驾照"),
    ),
    _rule(
        "EMPLOYEE_ID",
        "employee_id_context",
        r"(?<![A-Za-z0-9])(?:E\d{6,12}|EMP[-_]?[A-Z0-9]{4,12})(?![A-Za-z0-9])",
        0.90,
        required_context=("员工", "工号", "雇员", "人事"),
        flags=re.IGNORECASE,
    ),
    _rule(
        "CN_VEHICLE_LICENSE_PLATE",
        "cn_vehicle_plate_common",
        r"(?<![\u4e00-\u9fffA-Z0-9])"
        r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
        r"[A-Z][A-Z0-9]{5,6}(?![A-Z0-9])",
        0.88,
    ),
    _rule(
        "CN_VEHICLE_LICENSE_PLATE",
        "vehicle_plate_context_fallback",
        r"(?<![A-Z0-9])[\u4e00-\u9fff][A-Z][·•.-]?[A-Z0-9]{5,8}(?![A-Z0-9])",
        0.90,
        required_context=("车牌", "车辆", "号牌"),
    ),
    _rule(
        "POSTAL_CODE",
        "cn_postal_code_context",
        r"(?<!\d)[1-9]\d{5}(?!\d)",
        0.72,
        required_context=("邮编", "邮政编码"),
    ),
    _rule(
        "SECRET",
        "jwt_compact",
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\."
        r"[A-Za-z0-9_-]{6,}(?![A-Za-z0-9_-])",
        0.98,
    ),
    _rule(
        "SECRET",
        "opaque_secret_token",
        r"(?i)(?<![A-Za-z0-9_])(?:key_|sk_|tok_|secret_)[A-Za-z0-9_./+=-]{12,64}"
        r"(?![A-Za-z0-9_./+=-])",
        0.94,
    ),
    _rule(
        "SECRET",
        "key_assignment",
        r"(?i)(?:api[_-]?key|access[_-]?token|secret|session[_-]?token)\s*[:=]\s*"
        r"[\"']?(?P<value>[A-Za-z0-9_./+=-]{12,})[\"']?",
        0.90,
        value_group="value",
    ),
    _rule(
        "SECRET",
        "private_key_header",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        1.0,
    ),
    _rule(
        "MASKED_VALUE",
        "already_masked_phone_or_id",
        r"(?<!\d)\d{3,6}[*＊xX]{3,12}\d{2,4}(?!\d)",
        1.0,
        already_masked=True,
    ),
)


class CnCommonRulePack:
    """Apply common rules and suppress candidates rejected by shared validators."""

    def __init__(
        self,
        rules: Iterable[RuleDefinition] = CN_COMMON_RULES,
        *,
        context_window: int = 12,
    ) -> None:
        self.rules = tuple(rules)
        if not self.rules:
            raise ValueError("rules must not be empty")
        if isinstance(context_window, bool) or not isinstance(context_window, int):
            raise TypeError("context_window must be an integer")
        if context_window < 1:
            raise ValueError("context_window must be positive")
        self.context_window = context_window

    def _has_required_context(self, text: str, start: int, end: int, rule: RuleDefinition) -> bool:
        if not rule.required_context:
            return True
        context = text[
            max(0, start - self.context_window) : min(len(text), end + self.context_window)
        ].lower()
        return any(phrase.lower() in context for phrase in rule.required_context)

    def analyze(
        self,
        text: str,
        entities: Sequence[str] | None = None,
        *,
        include_invalid: bool = False,
    ) -> list[RuleMatch]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        requested = None if entities is None else set(entities)
        matches: list[RuleMatch] = []
        for rule in self.rules:
            if requested is not None and rule.entity_type not in requested:
                continue
            for candidate in rule.pattern.finditer(text):
                start, end = candidate.span(rule.value_group or 0)
                if not self._has_required_context(text, start, end, rule):
                    continue
                value = candidate.group(rule.value_group or 0)
                validation = (
                    validate_entity_value(rule.validator_label, value)
                    if rule.validator_label is not None
                    else None
                )
                # An 18-digit resident ID can occasionally also satisfy Luhn;
                # keep the semantically stronger checksum rule only.
                if rule.entity_type == "BANK_CARD_NUMBER":
                    nearby = text[max(0, start - self.context_window) : start]
                    looks_like_id_field = any(
                        phrase in nearby for phrase in ("身份证", "证件号", "公民身份")
                    )
                    if validate_cn_resident_id(value).valid or looks_like_id_field:
                        continue
                valid = validation is None or validation.valid
                if not valid and not include_invalid:
                    continue
                metadata: dict[str, Any] = {
                    "pattern_id": rule.pattern_id,
                    "source": "rule:cn_common",
                    "validator_label": rule.validator_label,
                    "validator_valid": validation.valid if validation is not None else None,
                    "validation_reason": validation.reason if validation is not None else None,
                    "already_masked": rule.already_masked,
                    "value_group": rule.value_group,
                }
                matches.append(
                    RuleMatch(
                        entity_type=rule.entity_type,
                        start=start,
                        end=end,
                        score=rule.score if valid else 0.0,
                        pattern_id=rule.pattern_id,
                        metadata=metadata,
                    )
                )
        deduplicated: dict[tuple[str, int, int], RuleMatch] = {}
        for match in matches:
            key = (match.entity_type, match.start, match.end)
            existing = deduplicated.get(key)
            if existing is None or (match.score, match.pattern_id) > (
                existing.score,
                existing.pattern_id,
            ):
                deduplicated[key] = match
        return sorted(
            deduplicated.values(),
            key=lambda match: (match.start, match.end, match.entity_type, -match.score),
        )

    __call__ = analyze


__all__ = ["CN_COMMON_RULES", "CnCommonRulePack", "RuleDefinition", "RuleMatch"]
