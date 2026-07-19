"""Fail-closed entity routing for the in-memory cascade runtime.

The runtime operates on Presidio-facing entity names.  A small alias table
normalizes the project's model-taxonomy names before routing so callers may
inject either :class:`QwenPiiRecognizer` or a generic BIO recognizer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

CandidateBranch = Literal["rule", "model"]
EntityCategory = Literal["structured", "semantic"]

_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_BRANCHES = frozenset({"rule", "model"})
_CATEGORIES = frozenset({"structured", "semantic"})

# Model-taxonomy -> policy-facing Presidio name.  Unknown labels are not
# rewritten and are subsequently rejected unless explicitly configured.
ENTITY_ALIASES: dict[str, str] = {
    "ADDRESS": "CN_ADDRESS",
    "CN_RESIDENT_ID": "CN_ID_CARD",
    "PERSON_NAME": "PERSON",
    "SOCIAL_SECURITY_NUMBER": "CN_SOCIAL_SECURITY_NUMBER",
    "VEHICLE_LICENSE_PLATE": "CN_VEHICLE_LICENSE_PLATE",
}


def canonicalize_entity_type(entity_type: str) -> str:
    if not isinstance(entity_type, str) or not _LABEL_RE.fullmatch(entity_type):
        raise ValueError("entity_type must be an uppercase taxonomy label")
    return ENTITY_ALIASES.get(entity_type, entity_type)


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """One raw-text-free threshold/routing decision."""

    accepted: bool
    threshold: float
    decision_process: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EntityRoute:
    """Explicit allow-list entry for one canonical output entity.

    Structured model branches cannot be enabled without a named validator.
    Whether that validator is installed is checked by ``CascadePipeline``.
    """

    entity_type: str
    category: EntityCategory
    rule_enabled: bool = True
    model_enabled: bool = False
    validator_label: str | None = None
    model_requires_validator: bool = False
    rule_threshold: float = 0.0
    model_threshold: float = 0.0
    short_span_max_chars: int = 0
    short_model_threshold: float | None = None

    def __post_init__(self) -> None:
        canonical = canonicalize_entity_type(self.entity_type)
        if canonical != self.entity_type:
            raise ValueError("route entity_type must use the canonical policy-facing label")
        if self.category not in _CATEGORIES:
            raise ValueError("route category must be 'structured' or 'semantic'")
        if not isinstance(self.rule_enabled, bool) or not isinstance(self.model_enabled, bool):
            raise TypeError("route enabled flags must be booleans")
        if not self.rule_enabled and not self.model_enabled:
            raise ValueError("a route must enable at least one recognizer branch")
        if self.validator_label is not None:
            canonicalize_entity_type(self.validator_label)
        if not isinstance(self.model_requires_validator, bool):
            raise TypeError("model_requires_validator must be a boolean")
        if self.model_requires_validator and not self.validator_label:
            raise ValueError("model_requires_validator needs validator_label")
        if self.category == "structured" and self.model_enabled:
            if not self.model_requires_validator or not self.validator_label:
                raise ValueError(
                    "structured model routes must fail closed through a named validator"
                )
        rule_threshold = _probability("rule_threshold", self.rule_threshold)
        model_threshold = _probability("model_threshold", self.model_threshold)
        object.__setattr__(self, "rule_threshold", rule_threshold)
        object.__setattr__(self, "model_threshold", model_threshold)
        if isinstance(self.short_span_max_chars, bool) or not isinstance(
            self.short_span_max_chars, int
        ):
            raise TypeError("short_span_max_chars must be an integer")
        if self.short_span_max_chars < 0:
            raise ValueError("short_span_max_chars must not be negative")
        if self.short_span_max_chars == 0 and self.short_model_threshold is not None:
            raise ValueError("short_model_threshold requires a positive short_span_max_chars")
        if self.short_span_max_chars > 0 and self.short_model_threshold is None:
            raise ValueError("short spans require an explicit fail-closed model threshold")
        if self.short_model_threshold is not None:
            short_threshold = _probability("short_model_threshold", self.short_model_threshold)
            if short_threshold < model_threshold:
                raise ValueError("short_model_threshold must not be below model_threshold")
            object.__setattr__(self, "short_model_threshold", short_threshold)

    def evaluate(
        self, *, branch: CandidateBranch, score: float, span_length: int
    ) -> RoutingDecision:
        """Apply branch allow-list and threshold policy without inspecting text."""

        if branch not in _BRANCHES:
            raise ValueError("branch must be 'rule' or 'model'")
        if isinstance(span_length, bool) or not isinstance(span_length, int):
            raise TypeError("span_length must be an integer")
        if span_length < 1:
            raise ValueError("span_length must be positive")
        normalized_score = _probability("candidate score", score)
        enabled = self.rule_enabled if branch == "rule" else self.model_enabled
        base_threshold = self.rule_threshold if branch == "rule" else self.model_threshold
        steps = [f"candidate:{branch}", f"route:{self.category}"]
        if not enabled:
            return RoutingDecision(False, base_threshold, (*steps, "route:branch_disabled"))
        threshold = base_threshold
        if (
            branch == "model"
            and self.short_span_max_chars > 0
            and span_length <= self.short_span_max_chars
        ):
            assert self.short_model_threshold is not None
            threshold = self.short_model_threshold
            steps.append("route:short_span_guard")
        if normalized_score < threshold:
            return RoutingDecision(False, threshold, (*steps, "threshold:rejected"))
        return RoutingDecision(True, threshold, (*steps, "threshold:passed"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EntityRoute:
        allowed = {
            "entity_type",
            "category",
            "rule_enabled",
            "model_enabled",
            "validator_label",
            "model_requires_validator",
            "rule_threshold",
            "model_threshold",
            "short_span_max_chars",
            "short_model_threshold",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"route contains unknown fields: {sorted(unknown)}")
        missing = {"entity_type", "category"} - set(value)
        if missing:
            raise ValueError(f"route is missing fields: {sorted(missing)}")
        return cls(**cast(dict[str, Any], dict(value)))


def _structured(
    entity_type: str,
    *,
    validator_label: str | None = None,
) -> EntityRoute:
    has_validator = validator_label is not None
    return EntityRoute(
        entity_type=entity_type,
        category="structured",
        rule_enabled=True,
        model_enabled=has_validator,
        validator_label=validator_label,
        model_requires_validator=has_validator,
    )


def default_routes() -> tuple[EntityRoute, ...]:
    """Return the conservative C1 route table in stable taxonomy order.

    A model may add semantic entities and structured values backed by an
    installed deterministic validator.  Other structured model branches stay
    closed until a project/customer validator and calibration evidence exist.
    """

    return (
        EntityRoute(
            "PERSON",
            "semantic",
            model_enabled=True,
            model_threshold=0.50,
            short_span_max_chars=2,
            short_model_threshold=0.98,
        ),
        _structured("PHONE_NUMBER", validator_label="PHONE_NUMBER"),
        _structured("EMAIL_ADDRESS", validator_label="EMAIL_ADDRESS"),
        EntityRoute(
            "CN_ADDRESS",
            "semantic",
            model_enabled=True,
            model_threshold=0.50,
            short_span_max_chars=6,
            short_model_threshold=0.98,
        ),
        EntityRoute("DATE_OF_BIRTH", "semantic", model_enabled=True, model_threshold=0.85),
        _structured("CN_ID_CARD", validator_label="CN_RESIDENT_ID"),
        _structured("PASSPORT_NUMBER"),
        _structured("DRIVER_LICENSE_NUMBER"),
        _structured("CN_SOCIAL_SECURITY_NUMBER"),
        _structured("BANK_CARD_NUMBER", validator_label="BANK_CARD_NUMBER"),
        _structured("BANK_ACCOUNT_NUMBER"),
        _structured("CN_VEHICLE_LICENSE_PLATE"),
        _structured("EMPLOYEE_ID"),
        _structured("STUDENT_ID"),
        _structured("MEDICAL_RECORD_NUMBER"),
        _structured("WECHAT_ID"),
        _structured("QQ_NUMBER"),
        _structured("ALIPAY_ACCOUNT"),
        EntityRoute(
            "USERNAME",
            "semantic",
            model_enabled=True,
            model_threshold=0.70,
            short_span_max_chars=4,
            short_model_threshold=0.99,
        ),
        _structured("IP_ADDRESS", validator_label="IP_ADDRESS"),
        _structured("MAC_ADDRESS", validator_label="MAC_ADDRESS"),
        _structured("DEVICE_ID"),
        _structured("GEO_COORDINATE", validator_label="GEO_COORDINATE"),
        _structured("SECRET"),
    )


def conservative_v2_routes() -> tuple[EntityRoute, ...]:
    """Return successor routes with strict plate validation and model route closed."""

    routes: list[EntityRoute] = []
    for route in default_routes():
        if route.entity_type != "CN_VEHICLE_LICENSE_PLATE":
            routes.append(route)
            continue
        routes.append(
            EntityRoute(
                entity_type=route.entity_type,
                category=route.category,
                rule_enabled=True,
                model_enabled=False,
                validator_label="CN_VEHICLE_LICENSE_PLATE",
                model_requires_validator=False,
                rule_threshold=route.rule_threshold,
                model_threshold=route.model_threshold,
            )
        )
    return tuple(routes)


_COMMUNITY_STRUCTURED_VALIDATOR_LABELS: dict[str, str] = {
    "PHONE_NUMBER": "PHONE_NUMBER",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "CN_ID_CARD": "CN_RESIDENT_ID",
    "PASSPORT_NUMBER": "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER": "DRIVER_LICENSE_NUMBER",
    "CN_SOCIAL_SECURITY_NUMBER": "CN_SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER": "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER": "BANK_ACCOUNT_NUMBER",
    "CN_VEHICLE_LICENSE_PLATE": "CN_VEHICLE_LICENSE_PLATE",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
    "STUDENT_ID": "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER": "MEDICAL_RECORD_NUMBER",
    "WECHAT_ID": "WECHAT_ID",
    "QQ_NUMBER": "QQ_NUMBER",
    "ALIPAY_ACCOUNT": "ALIPAY_ACCOUNT",
    "IP_ADDRESS": "IP_ADDRESS",
    "MAC_ADDRESS": "MAC_ADDRESS",
    "DEVICE_ID": "DEVICE_ID",
    "GEO_COORDINATE": "GEO_COORDINATE",
    "SECRET": "SECRET",
}

# The v2 calibration split permits fusion-policy tuning.  These rule branches
# were the only ones which either introduced PII-free false positives or
# displaced correct calibrated-model spans under deterministic fusion.  Keep
# the uncalibrated community profile unchanged, but fail closed for these
# branches whenever the frozen calibration policy is authoritative.
COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES = frozenset(
    {
        "BANK_ACCOUNT_NUMBER",
        "BANK_CARD_NUMBER",
        "CN_SOCIAL_SECURITY_NUMBER",
        "IP_ADDRESS",
        "QQ_NUMBER",
    }
)


def community_full24_routes(
    *, recognizer_thresholds_authoritative: bool = False
) -> tuple[EntityRoute, ...]:
    """Open every core-24 model route behind a semantic/validator boundary.

    This is a new community-only table.  ``default_routes`` and
    ``conservative_v2_routes`` remain byte-for-byte behaviorally unchanged.
    With no explicit calibration, the historical semantic guards are retained
    and newly opened structured routes use a conservative 0.50 model floor.
    When a caller supplies a frozen calibration/threshold policy, recognizer
    filtering becomes authoritative and route thresholds are reduced to zero
    so the same builder can be used for formal evaluation and serving.
    """

    if not isinstance(recognizer_thresholds_authoritative, bool):
        raise TypeError("recognizer_thresholds_authoritative must be a boolean")
    routes: list[EntityRoute] = []
    for route in default_routes():
        validator_label = _COMMUNITY_STRUCTURED_VALIDATOR_LABELS.get(route.entity_type)
        rule_enabled = route.rule_enabled and not (
            recognizer_thresholds_authoritative
            and route.entity_type in COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES
        )
        if route.category == "semantic":
            routes.append(
                EntityRoute(
                    entity_type=route.entity_type,
                    category="semantic",
                    rule_enabled=rule_enabled,
                    model_enabled=True,
                    rule_threshold=route.rule_threshold,
                    model_threshold=(
                        0.0 if recognizer_thresholds_authoritative else route.model_threshold
                    ),
                    short_span_max_chars=(
                        0
                        if recognizer_thresholds_authoritative
                        else route.short_span_max_chars
                    ),
                    short_model_threshold=(
                        None
                        if recognizer_thresholds_authoritative
                        else route.short_model_threshold
                    ),
                )
            )
            continue
        if validator_label is None:  # pragma: no cover - guarded by the exact core-24 test
            raise RuntimeError(
                f"community structured route lacks a validator: {route.entity_type}"
            )
        routes.append(
            EntityRoute(
                entity_type=route.entity_type,
                category="structured",
                rule_enabled=rule_enabled,
                model_enabled=True,
                validator_label=validator_label,
                model_requires_validator=True,
                rule_threshold=route.rule_threshold,
                model_threshold=(
                    0.0 if recognizer_thresholds_authoritative else 0.50
                ),
            )
        )
    if len(routes) != 24 or any(not route.model_enabled for route in routes):
        raise RuntimeError("community route table must expose exactly 24 model routes")
    return tuple(routes)


__all__ = [
    "CandidateBranch",
    "COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES",
    "ENTITY_ALIASES",
    "EntityCategory",
    "EntityRoute",
    "RoutingDecision",
    "canonicalize_entity_type",
    "community_full24_routes",
    "conservative_v2_routes",
    "default_routes",
]
