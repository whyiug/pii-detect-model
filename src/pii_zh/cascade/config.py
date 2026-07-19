"""Strict, fail-closed configuration contract for the cascade runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast

from .routing import EntityRoute, default_routes

CascadeMode = Literal["rules-only", "model-only", "cascade"]
CASCADE_CONFIG_SCHEMA_VERSION = 1
_MODES = frozenset({"rules-only", "model-only", "cascade"})


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """Versioned runtime profile with an explicit per-entity allow list."""

    schema_version: int = CASCADE_CONFIG_SCHEMA_VERSION
    profile_version: str = "c1-conservative-v1"
    mode: CascadeMode = "rules-only"
    routes: tuple[EntityRoute, ...] = field(default_factory=default_routes)
    rule_source: str = "rule:cn_common"
    model_source: str = "qwen"
    keep_nested_different_types: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CASCADE_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CASCADE_CONFIG_SCHEMA_VERSION}")
        if not isinstance(self.profile_version, str) or not self.profile_version.strip():
            raise ValueError("profile_version must be a non-empty string")
        if self.mode not in _MODES:
            raise ValueError("mode must be 'rules-only', 'model-only', or 'cascade'")
        if not isinstance(self.routes, tuple) or not self.routes:
            raise ValueError("routes must be a non-empty tuple")
        if any(not isinstance(route, EntityRoute) for route in self.routes):
            raise TypeError("routes must contain EntityRoute values")
        labels = [route.entity_type for route in self.routes]
        if len(labels) != len(set(labels)):
            raise ValueError("routes must not contain duplicate entity types")
        for name, value in (("rule_source", self.rule_source), ("model_source", self.model_source)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.rule_source == self.model_source:
            raise ValueError("rule_source and model_source must be distinct")
        if not isinstance(self.keep_nested_different_types, bool):
            raise TypeError("keep_nested_different_types must be a boolean")

    @property
    def route_map(self) -> Mapping[str, EntityRoute]:
        return MappingProxyType({route.entity_type: route for route in self.routes})

    @property
    def uses_rules(self) -> bool:
        return self.mode in {"rules-only", "cascade"}

    @property
    def uses_model(self) -> bool:
        return self.mode in {"model-only", "cascade"}

    def with_mode(self, mode: CascadeMode) -> CascadeConfig:
        return replace(self, mode=mode)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CascadeConfig:
        """Parse mappings without accepting misspelled or future fields."""

        allowed = {
            "schema_version",
            "profile_version",
            "mode",
            "routes",
            "rule_source",
            "model_source",
            "keep_nested_different_types",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"cascade config contains unknown fields: {sorted(unknown)}")
        document = dict(value)
        raw_routes = document.get("routes")
        if raw_routes is not None:
            if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
                raise TypeError("routes must be a sequence of mappings")
            routes: list[EntityRoute] = []
            for index, raw_route in enumerate(raw_routes):
                if not isinstance(raw_route, Mapping):
                    raise TypeError(f"routes[{index}] must be a mapping")
                routes.append(EntityRoute.from_mapping(cast(Mapping[str, Any], raw_route)))
            document["routes"] = tuple(routes)
        return cls(**document)


__all__ = ["CASCADE_CONFIG_SCHEMA_VERSION", "CascadeConfig", "CascadeMode"]
