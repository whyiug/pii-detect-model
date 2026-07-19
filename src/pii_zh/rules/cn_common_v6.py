"""Successor ``cn_common_v6`` rules without rewriting the historical v5 pack."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pii_zh.data.validators.cn_vehicle_plate import validate_cn_vehicle_license_plate

from .cn_common import CN_COMMON_RULES, CnCommonRulePack, RuleDefinition, RuleMatch

CN_COMMON_V6_RULESET_ID = "cn_common_v6"
CN_COMMON_V6_SOURCE = "rule:cn_common_v6"

_PLATE_ENTITY = "CN_VEHICLE_LICENSE_PLATE"
_PLATE_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
_PLATE_LETTERS = "A-HJ-NP-Z"

_STRICT_VEHICLE_PLATE_RULE = RuleDefinition(
    entity_type=_PLATE_ENTITY,
    pattern_id="cn_vehicle_plate_strict_v6",
    pattern=re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"[{_PLATE_PROVINCES}]"
        rf"[{_PLATE_LETTERS}]"
        rf"[·•]?"
        rf"(?:[{_PLATE_LETTERS}0-9]{{5}}|[DF][0-9]{{5}}|[0-9]{{5}}[DF])"
        rf"(?![A-Za-z0-9挂学警港澳使领应临])",
    ),
    score=0.96,
)


def _build_v6_rules() -> tuple[RuleDefinition, ...]:
    rules: list[RuleDefinition] = []
    inserted = False
    for rule in CN_COMMON_RULES:
        if rule.entity_type != _PLATE_ENTITY:
            rules.append(rule)
            continue
        if not inserted:
            rules.append(_STRICT_VEHICLE_PLATE_RULE)
            inserted = True
    if not inserted:  # pragma: no cover - guards accidental historical rule deletion
        raise RuntimeError("historical cn_common rules contain no vehicle-plate slot")
    return tuple(rules)


CN_COMMON_RULES_V6 = _build_v6_rules()


class CnCommonRulePackV6(CnCommonRulePack):
    """Use the strict plate rule while preserving all non-plate v5 behavior."""

    def __init__(self, *, context_window: int = 12) -> None:
        super().__init__(CN_COMMON_RULES_V6, context_window=context_window)

    def analyze(
        self,
        text: str,
        entities: Sequence[str] | None = None,
        *,
        include_invalid: bool = False,
    ) -> list[RuleMatch]:
        """Apply successor validation without mutating the frozen v5 registry."""

        matches = super().analyze(text, entities, include_invalid=include_invalid)
        successor: list[RuleMatch] = []
        for match in matches:
            metadata = dict(match.metadata)
            score = match.score
            if match.entity_type == _PLATE_ENTITY:
                validation = validate_cn_vehicle_license_plate(text[match.start : match.end])
                if not validation.valid and not include_invalid:
                    continue
                score = match.score if validation.valid else 0.0
                metadata.update(
                    {
                        "validator_label": _PLATE_ENTITY,
                        "validator_valid": validation.valid,
                        "validation_reason": validation.reason,
                    }
                )
            metadata["source"] = CN_COMMON_V6_SOURCE
            successor.append(
                RuleMatch(
                    entity_type=match.entity_type,
                    start=match.start,
                    end=match.end,
                    score=score,
                    pattern_id=match.pattern_id,
                    source=CN_COMMON_V6_SOURCE,
                    metadata=metadata,
                )
            )
        return successor

    __call__ = analyze


__all__ = [
    "CN_COMMON_RULES_V6",
    "CN_COMMON_V6_RULESET_ID",
    "CN_COMMON_V6_SOURCE",
    "CnCommonRulePackV6",
]
