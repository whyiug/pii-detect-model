"""Load and validate the packaged taxonomy contracts without model dependencies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

_LABEL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SUPPORTED_SCHEMA_VERSION = 1
_EXPECTED_LABEL_SETS = ("core", "extended", "auxiliary")


class TaxonomyValidationError(ValueError):
    """Raised when taxonomy or mapping data violates the public contract."""


@dataclass(frozen=True, slots=True)
class RiskTier:
    """A risk tier used by policy and evaluation, not a redaction instruction."""

    code: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class EntityDefinition:
    """One model label definition."""

    name: str
    description: str
    risk_tiers: tuple[str, ...]
    detector_roles: tuple[str, ...]
    enabled_by_default: bool
    output: bool
    label_set: str


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """Validated, immutable view of ``taxonomy.yaml``."""

    schema_version: int
    taxonomy_version: str
    language: str
    task: str
    tagging_scheme: str
    outside_label: str
    risk_tiers: Mapping[str, RiskTier]
    label_sets: Mapping[str, tuple[EntityDefinition, ...]]
    flattening_priority: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def entities(self) -> tuple[EntityDefinition, ...]:
        """Return all labels in stable set/file order."""

        return tuple(
            entity for set_name in _EXPECTED_LABEL_SETS for entity in self.label_sets[set_name]
        )

    @property
    def label_names(self) -> frozenset[str]:
        return frozenset(entity.name for entity in self.entities)

    @property
    def core_label_names(self) -> frozenset[str]:
        return frozenset(entity.name for entity in self.label_sets["core"])

    @property
    def extended_label_names(self) -> frozenset[str]:
        return frozenset(entity.name for entity in self.label_sets["extended"])

    @property
    def auxiliary_label_names(self) -> frozenset[str]:
        return frozenset(entity.name for entity in self.label_sets["auxiliary"])

    @property
    def output_label_names(self) -> frozenset[str]:
        return frozenset(entity.name for entity in self.entities if entity.output)


@dataclass(frozen=True, slots=True)
class PresidioMapping:
    """Validated mapping from model labels to policy-facing Presidio entities."""

    schema_version: int
    mapping_version: str
    taxonomy_version: str
    language: str
    bio_prefixes: tuple[str, ...]
    model_to_presidio: Mapping[str, str]
    ignored_labels: frozenset[str]


def _as_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaxonomyValidationError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TaxonomyValidationError(f"{location} keys must be strings")
    return cast(Mapping[str, Any], value)


def _as_sequence(value: object, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaxonomyValidationError(f"{location} must be a sequence")
    return value


def _required_string(document: Mapping[str, Any], key: str, location: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyValidationError(f"{location}.{key} must be a non-empty string")
    return value


def _required_bool(document: Mapping[str, Any], key: str, location: str) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise TaxonomyValidationError(f"{location}.{key} must be a boolean")
    return value


def _schema_version(document: Mapping[str, Any], location: str) -> int:
    version = document.get("schema_version")
    if version != _SUPPORTED_SCHEMA_VERSION:
        raise TaxonomyValidationError(
            f"{location}.schema_version must be {_SUPPORTED_SCHEMA_VERSION}, got {version!r}"
        )
    return cast(int, version)


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    result = tuple(_as_sequence(value, location))
    if not all(isinstance(item, str) and item for item in result):
        raise TaxonomyValidationError(f"{location} must contain non-empty strings")
    return cast(tuple[str, ...], result)


def _read_yaml(path: str | Path | None, packaged_name: str) -> Mapping[str, Any]:
    if path is None:
        resource = resources.files("pii_zh.taxonomy").joinpath(packaged_name)
        with resource.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    else:
        with Path(path).open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    return _as_mapping(loaded, packaged_name)


def validate_taxonomy_document(document: Mapping[str, Any]) -> Taxonomy:
    """Validate a parsed taxonomy document and return its immutable view."""

    schema_version = _schema_version(document, "taxonomy")
    taxonomy_version = _required_string(document, "taxonomy_version", "taxonomy")
    language = _required_string(document, "language", "taxonomy")
    task = _required_string(document, "task", "taxonomy")
    tagging_scheme = _required_string(document, "tagging_scheme", "taxonomy")
    outside_label = _required_string(document, "outside_label", "taxonomy")
    if tagging_scheme != "BIO" or outside_label != "O":
        raise TaxonomyValidationError("phase-A taxonomy requires BIO tagging with outside label O")

    risk_documents = _as_mapping(document.get("risk_tiers"), "taxonomy.risk_tiers")
    if not risk_documents:
        raise TaxonomyValidationError("taxonomy.risk_tiers must not be empty")
    risks: dict[str, RiskTier] = {}
    for code, raw_risk in risk_documents.items():
        risk = _as_mapping(raw_risk, f"taxonomy.risk_tiers.{code}")
        if not re.fullmatch(r"T[0-9]+", code):
            raise TaxonomyValidationError(f"invalid risk tier code: {code}")
        risks[code] = RiskTier(
            code=code,
            name=_required_string(risk, "name", f"taxonomy.risk_tiers.{code}"),
            description=_required_string(risk, "description", f"taxonomy.risk_tiers.{code}"),
        )

    raw_label_sets = _as_mapping(document.get("label_sets"), "taxonomy.label_sets")
    if set(raw_label_sets) != set(_EXPECTED_LABEL_SETS):
        raise TaxonomyValidationError(
            "taxonomy.label_sets must contain exactly core, extended, and auxiliary"
        )

    seen_names: set[str] = set()
    parsed_sets: dict[str, tuple[EntityDefinition, ...]] = {}
    for set_name in _EXPECTED_LABEL_SETS:
        raw_entities = _as_sequence(raw_label_sets[set_name], f"taxonomy.label_sets.{set_name}")
        if not raw_entities:
            raise TaxonomyValidationError(f"taxonomy.label_sets.{set_name} must not be empty")
        entities: list[EntityDefinition] = []
        for index, raw_entity in enumerate(raw_entities):
            location = f"taxonomy.label_sets.{set_name}[{index}]"
            entity = _as_mapping(raw_entity, location)
            name = _required_string(entity, "name", location)
            if not _LABEL_PATTERN.fullmatch(name):
                raise TaxonomyValidationError(f"{location}.name is not a valid label: {name}")
            if name in seen_names:
                raise TaxonomyValidationError(f"duplicate label: {name}")
            seen_names.add(name)
            entity_risks = _string_tuple(entity.get("risk_tiers"), f"{location}.risk_tiers")
            unknown_risks = set(entity_risks) - set(risks)
            if unknown_risks:
                raise TaxonomyValidationError(
                    f"{location}.risk_tiers contains unknown values: {sorted(unknown_risks)}"
                )
            detector_roles = _string_tuple(
                entity.get("detector_roles"), f"{location}.detector_roles"
            )
            enabled = _required_bool(entity, "enabled_by_default", location)
            output = _required_bool(entity, "output", location)
            if set_name == "core" and (not enabled or not output or not entity_risks):
                raise TaxonomyValidationError(
                    f"core label {name} must be enabled, output-capable, and risk-tiered"
                )
            if set_name == "extended" and (enabled or not output or not entity_risks):
                raise TaxonomyValidationError(
                    f"extended label {name} must be disabled by default, "
                    "output-capable, and risk-tiered"
                )
            if set_name == "auxiliary" and (enabled or output or entity_risks):
                raise TaxonomyValidationError(
                    f"auxiliary label {name} must be disabled, ignored at output, "
                    "and have no risk tier"
                )
            entities.append(
                EntityDefinition(
                    name=name,
                    description=_required_string(entity, "description", location),
                    risk_tiers=entity_risks,
                    detector_roles=detector_roles,
                    enabled_by_default=enabled,
                    output=output,
                    label_set=set_name,
                )
            )
        parsed_sets[set_name] = tuple(entities)

    flattening = _as_mapping(document.get("flattening"), "taxonomy.flattening")
    raw_priorities = _as_sequence(flattening.get("priority"), "taxonomy.flattening.priority")
    priorities: list[tuple[str, tuple[str, ...]]] = []
    prioritized_labels: list[str] = []
    for index, raw_priority in enumerate(raw_priorities):
        location = f"taxonomy.flattening.priority[{index}]"
        priority = _as_mapping(raw_priority, location)
        priority_name = _required_string(priority, "name", location)
        labels = _string_tuple(priority.get("labels"), f"{location}.labels")
        priorities.append((priority_name, labels))
        prioritized_labels.extend(labels)
    if len(prioritized_labels) != len(set(prioritized_labels)):
        raise TaxonomyValidationError("flattening priority contains duplicate labels")
    if set(prioritized_labels) != seen_names:
        missing = sorted(seen_names - set(prioritized_labels))
        unknown = sorted(set(prioritized_labels) - seen_names)
        raise TaxonomyValidationError(
            "flattening priority must cover every label exactly once; "
            f"missing={missing}, unknown={unknown}"
        )

    return Taxonomy(
        schema_version=schema_version,
        taxonomy_version=taxonomy_version,
        language=language,
        task=task,
        tagging_scheme=tagging_scheme,
        outside_label=outside_label,
        risk_tiers=MappingProxyType(risks),
        label_sets=MappingProxyType(parsed_sets),
        flattening_priority=tuple(priorities),
    )


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load the packaged taxonomy, or a caller-supplied taxonomy file."""

    return validate_taxonomy_document(_read_yaml(path, "taxonomy.yaml"))


def validate_presidio_mapping_document(
    document: Mapping[str, Any], taxonomy: Taxonomy
) -> PresidioMapping:
    """Validate a parsed Presidio mapping against an already validated taxonomy."""

    schema_version = _schema_version(document, "presidio_mapping")
    mapping_version = _required_string(document, "mapping_version", "presidio_mapping")
    taxonomy_version = _required_string(document, "taxonomy_version", "presidio_mapping")
    if taxonomy_version != taxonomy.taxonomy_version:
        raise TaxonomyValidationError(
            "presidio mapping taxonomy_version does not match the loaded taxonomy"
        )
    language = _required_string(document, "language", "presidio_mapping")
    bio_prefixes = _string_tuple(document.get("bio_prefixes"), "presidio_mapping.bio_prefixes")
    if bio_prefixes != ("B-", "I-"):
        raise TaxonomyValidationError("presidio mapping must declare BIO prefixes B- and I-")

    raw_mapping = _as_mapping(
        document.get("model_to_presidio"), "presidio_mapping.model_to_presidio"
    )
    model_to_presidio: dict[str, str] = {}
    for model_label, presidio_label in raw_mapping.items():
        if not isinstance(presidio_label, str) or not _LABEL_PATTERN.fullmatch(presidio_label):
            raise TaxonomyValidationError(
                f"invalid Presidio entity for {model_label}: {presidio_label!r}"
            )
        model_to_presidio[model_label] = presidio_label

    mapped_labels = set(model_to_presidio)
    expected_output = set(taxonomy.output_label_names)
    if mapped_labels != expected_output:
        missing = sorted(expected_output - mapped_labels)
        unknown = sorted(mapped_labels - expected_output)
        raise TaxonomyValidationError(
            "Presidio mapping must cover output labels exactly; "
            f"missing={missing}, unknown={unknown}"
        )

    ignored_labels = frozenset(
        _string_tuple(document.get("ignored_labels"), "presidio_mapping.ignored_labels")
    )
    if ignored_labels != taxonomy.auxiliary_label_names:
        missing = sorted(taxonomy.auxiliary_label_names - ignored_labels)
        unknown = sorted(ignored_labels - taxonomy.auxiliary_label_names)
        raise TaxonomyValidationError(
            f"ignored labels must equal auxiliary labels; missing={missing}, unknown={unknown}"
        )

    return PresidioMapping(
        schema_version=schema_version,
        mapping_version=mapping_version,
        taxonomy_version=taxonomy_version,
        language=language,
        bio_prefixes=bio_prefixes,
        model_to_presidio=MappingProxyType(model_to_presidio),
        ignored_labels=ignored_labels,
    )


def load_presidio_mapping(
    path: str | Path | None = None, *, taxonomy: Taxonomy | None = None
) -> PresidioMapping:
    """Load the packaged mapping, validating it against ``taxonomy``."""

    resolved_taxonomy = taxonomy if taxonomy is not None else load_taxonomy()
    return validate_presidio_mapping_document(
        _read_yaml(path, "presidio_mapping.yaml"), resolved_taxonomy
    )
