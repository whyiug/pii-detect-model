"""Order-independent deterministic fusion for model and rule spans."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_SPECIFICITY: dict[str, int] = {
    "CN_ID_CARD": 120,
    "CN_RESIDENT_ID": 120,
    "BANK_CARD_NUMBER": 120,
    "BANK_ACCOUNT_NUMBER": 115,
    "PASSPORT_NUMBER": 115,
    "DRIVER_LICENSE_NUMBER": 115,
    "CN_SOCIAL_SECURITY_NUMBER": 115,
    "MEDICAL_RECORD_NUMBER": 115,
    "PHONE_NUMBER": 110,
    "EMAIL_ADDRESS": 110,
    "MAC_ADDRESS": 110,
    "IP_ADDRESS": 105,
    "CN_VEHICLE_LICENSE_PLATE": 105,
    "VEHICLE_LICENSE_PLATE": 105,
    "GEO_COORDINATE": 105,
    "SECRET": 105,
    "WECHAT_ID": 100,
    "QQ_NUMBER": 100,
    # An Alipay account may deliberately use an email-shaped identifier.  A
    # contextual account type is more specific than its surface EMAIL form.
    "ALIPAY_ACCOUNT": 115,
    "EMPLOYEE_ID": 100,
    "STUDENT_ID": 100,
    "DEVICE_ID": 95,
    "DATE_OF_BIRTH": 90,
    "CN_ADDRESS": 80,
    "ADDRESS": 80,
    "PERSON": 70,
    "PERSON_NAME": 70,
    "USERNAME": 65,
    "LOCATION": 30,
    "ORGANIZATION": 20,
    "GENERIC_ID": 10,
    "ID": 10,
    "NUMBER": 0,
}


@dataclass(frozen=True, slots=True)
class FusedDetection:
    entity_type: str
    start: int
    end: int
    score: float
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, str) or not self.entity_type:
            raise ValueError("entity_type must be a non-empty string")
        if not 0 <= self.start < self.end:
            raise ValueError("detection must use a non-empty left-closed/right-open span")
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("detection score must be finite and between zero and one")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")


def _value(result: Any, *names: str, default: object = None) -> object:
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
        return default
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return default


def _metadata(result: Any) -> dict[str, Any]:
    value = _value(result, "metadata", "recognition_metadata", default={})
    return dict(value) if isinstance(value, Mapping) else {}


def as_detection(result: Any, *, default_source: str = "unknown") -> FusedDetection:
    if isinstance(result, FusedDetection):
        return result
    metadata = _metadata(result)
    source = _value(result, "source", default=metadata.get("source", default_source))
    entity_type = _value(result, "entity_type", "label")
    start = _value(result, "start")
    end = _value(result, "end")
    score = _value(result, "score", default=0.0)
    if not isinstance(entity_type, str):
        raise TypeError("fusion input entity_type must be a string")
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("fusion input start must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise TypeError("fusion input end must be an integer")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("fusion input score must be numeric")
    if not isinstance(source, str):
        raise TypeError("fusion input source must be a string")
    return FusedDetection(entity_type, start, end, float(score), source, metadata)


def _overlap(left: FusedDetection, right: FusedDetection) -> bool:
    return left.start < right.end and right.start < left.end


class DeterministicFusion:
    """Fuse detections with validator and specific-type precedence."""

    def __init__(
        self,
        *,
        thresholds: Mapping[str, float] | None = None,
        default_threshold: float = 0.0,
        specificity: Mapping[str, int] | None = None,
        source_priority: Mapping[str, int] | None = None,
        keep_nested_different_types: bool = False,
    ) -> None:
        self.thresholds = {str(key): float(value) for key, value in (thresholds or {}).items()}
        self.default_threshold = float(default_threshold)
        for value in [self.default_threshold, *self.thresholds.values()]:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("fusion thresholds must be finite and between zero and one")
        self.specificity = {**DEFAULT_SPECIFICITY, **dict(specificity or {})}
        self.source_priority = {
            "customer_rule": 50,
            "rule:cn_common": 40,
            "qwen": 30,
            "ernie": 20,
            **dict(source_priority or {}),
        }
        self.keep_nested_different_types = keep_nested_different_types

    def _specificity(self, detection: FusedDetection) -> int:
        if detection.entity_type in self.specificity:
            return self.specificity[detection.entity_type]
        # Unknown names ending in a concrete identifier suffix are safer than
        # generic ID/NUMBER fallbacks, while remaining below configured types.
        if detection.entity_type.endswith(("_NUMBER", "_ID", "_ACCOUNT")):
            return 60
        return 50

    def _priority(self, detection: FusedDetection) -> tuple[Any, ...]:
        validator_valid = detection.metadata.get("validator_valid") is True
        source_priority = self.source_priority.get(detection.source, 0)
        # Lower start/end/source values win final ties by negating numeric
        # offsets and using the reversed sort below.
        return (
            self._specificity(detection),
            int(validator_valid),
            source_priority,
            detection.score,
            -(detection.end - detection.start),
            -detection.start,
            detection.entity_type,
            detection.source,
        )

    def _collapse_exact(self, detections: Sequence[FusedDetection]) -> list[FusedDetection]:
        groups: dict[tuple[str, int, int], list[FusedDetection]] = {}
        for detection in detections:
            groups.setdefault((detection.entity_type, detection.start, detection.end), []).append(
                detection
            )
        collapsed: list[FusedDetection] = []
        for group in groups.values():
            winner = max(group, key=self._priority)
            sources = sorted({item.source for item in group})
            metadata = dict(winner.metadata)
            metadata.update(
                {
                    "sources": sources,
                    "agreement_count": len(sources),
                    "fusion_decision": "exact_same_type_collapsed"
                    if len(group) > 1
                    else "single_source",
                }
            )
            collapsed.append(replace(winner, metadata=metadata))
        return collapsed

    def fuse(self, *result_groups: Iterable[Any]) -> list[FusedDetection]:
        detections: list[FusedDetection] = []
        for group in result_groups:
            for result in group:
                detection = as_detection(result)
                if detection.metadata.get("validator_valid") is False:
                    continue
                threshold = self.thresholds.get(detection.entity_type, self.default_threshold)
                if detection.score >= threshold:
                    detections.append(detection)
        collapsed = self._collapse_exact(detections)
        ranked = sorted(collapsed, key=self._priority, reverse=True)
        kept: list[FusedDetection] = []
        for candidate in ranked:
            conflicts = [existing for existing in kept if _overlap(candidate, existing)]
            if not conflicts:
                kept.append(candidate)
                continue
            if self.keep_nested_different_types and all(
                candidate.entity_type != existing.entity_type
                and (
                    candidate.start <= existing.start < existing.end <= candidate.end
                    or existing.start <= candidate.start < candidate.end <= existing.end
                )
                for existing in conflicts
            ):
                kept.append(candidate)
        return sorted(kept, key=lambda item: (item.start, item.end, item.entity_type))

    __call__ = fuse


__all__ = [
    "DEFAULT_SPECIFICITY",
    "DeterministicFusion",
    "FusedDetection",
    "as_detection",
]
