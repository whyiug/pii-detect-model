"""Strict, character-span-first JSONL schema used by every data source.

The schema deliberately uses only the Python standard library.  Training labels
are a derived view: the authoritative annotation is always a Python character
offset with a left-closed, right-open interval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

SCHEMA_VERSION = "pii-zh-v1"


class SchemaError(ValueError):
    """Raised when a record violates the canonical data contract."""


class DataPool(str, Enum):
    PUBLIC_RELEASE = "public_release_pool"
    PRIVATE_ENTERPRISE = "private_enterprise_pool"
    EVALUATION_ONLY = "evaluation_only"
    QUARANTINED = "quarantined"


class QualityTier(str, Enum):
    GOLD = "G0"
    SYNTHETIC_VALIDATED = "S0"
    MULTI_TEACHER = "S1"
    SINGLE_TEACHER = "S2"
    UNCERTAIN = "U"
    HARD_NEGATIVE = "N"


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{name} must be a bool")
    return value


def _require_non_empty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def stable_text_hash(value: str) -> str:
    """Return a non-reversible grouping hash without retaining the raw value."""

    normalized = "".join(value.split()).casefold()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EntitySpan:
    start: int
    end: int
    label: str
    text: str
    source: str
    risk_tier: str | None = None
    validator_state: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, document_text: str) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise SchemaError("entity.start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise SchemaError("entity.end must be an integer")
        if not (0 <= self.start < self.end <= len(document_text)):
            raise SchemaError(
                f"invalid left-closed/right-open span [{self.start}, {self.end}) "
                f"for text length {len(document_text)}"
            )
        _require_non_empty("entity.label", self.label)
        _require_non_empty("entity.source", self.source)
        if document_text[self.start : self.end] != self.text:
            raise SchemaError(
                "entity text mismatch: "
                f"text[{self.start}:{self.end}]={document_text[self.start : self.end]!r} "
                f"!= entity.text={self.text!r}"
            )
        if self.risk_tier is not None and self.risk_tier not in {"T0", "T1", "T2"}:
            raise SchemaError("entity.risk_tier must be T0, T1, T2, or null")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("entity.metadata must be a mapping")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EntitySpan:
        allowed = {
            "start",
            "end",
            "label",
            "text",
            "source",
            "risk_tier",
            "validator_state",
            "metadata",
        }
        _reject_unknown("entity", value, allowed)
        required = {"start", "end", "label", "text", "source"}
        _require_keys("entity", value, required)
        return cls(
            start=value["start"],
            end=value["end"],
            label=value["label"],
            text=value["text"],
            source=value["source"],
            risk_tier=value.get("risk_tier"),
            validator_state=value.get("validator_state"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    source_id: str
    source_kind: str
    source_revision: str
    license: str
    synthetic: bool
    evaluation_only: bool = False
    private_enterprise: bool = False
    generator_name: str | None = None
    generator_version: str | None = None
    generator_seed: int | None = None
    template_id: str | None = None
    terms_snapshot: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty("provenance.source_id", self.source_id)
        _require_non_empty("provenance.source_kind", self.source_kind)
        _require_non_empty("provenance.source_revision", self.source_revision)
        _require_non_empty("provenance.license", self.license)
        _require_bool("provenance.synthetic", self.synthetic)
        _require_bool("provenance.evaluation_only", self.evaluation_only)
        _require_bool("provenance.private_enterprise", self.private_enterprise)
        if self.evaluation_only and self.private_enterprise:
            raise SchemaError("a source cannot be both evaluation-only and private-enterprise")
        if self.synthetic:
            _require_non_empty("provenance.generator_name", self.generator_name)
            _require_non_empty("provenance.generator_version", self.generator_version)
        if self.generator_seed is not None and (
            isinstance(self.generator_seed, bool) or not isinstance(self.generator_seed, int)
        ):
            raise SchemaError("provenance.generator_seed must be an integer or null")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("provenance.metadata must be a mapping")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Provenance:
        allowed = {
            "source_id",
            "source_kind",
            "source_revision",
            "license",
            "synthetic",
            "evaluation_only",
            "private_enterprise",
            "generator_name",
            "generator_version",
            "generator_seed",
            "template_id",
            "terms_snapshot",
            "metadata",
        }
        _reject_unknown("provenance", value, allowed)
        required = {"source_id", "source_kind", "source_revision", "license", "synthetic"}
        _require_keys("provenance", value, required)
        return cls(
            source_id=value["source_id"],
            source_kind=value["source_kind"],
            source_revision=value["source_revision"],
            license=value["license"],
            synthetic=value["synthetic"],
            evaluation_only=value.get("evaluation_only", False),
            private_enterprise=value.get("private_enterprise", False),
            generator_name=value.get("generator_name"),
            generator_version=value.get("generator_version"),
            generator_seed=value.get("generator_seed"),
            template_id=value.get("template_id"),
            terms_snapshot=value.get("terms_snapshot"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class QualityMetadata:
    tier: QualityTier
    quality_gate: bool
    validators_passed: bool
    review_status: str
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.tier, QualityTier):
            raise SchemaError("quality.tier must be a QualityTier")
        _require_bool("quality.quality_gate", self.quality_gate)
        _require_bool("quality.validators_passed", self.validators_passed)
        _require_non_empty("quality.review_status", self.review_status)
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise SchemaError("quality.confidence must be numeric or null")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise SchemaError("quality.confidence must be between 0 and 1")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("quality.metadata must be a mapping")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualityMetadata:
        allowed = {
            "tier",
            "quality_gate",
            "validators_passed",
            "review_status",
            "confidence",
            "metadata",
        }
        _reject_unknown("quality", value, allowed)
        required = {"tier", "quality_gate", "validators_passed", "review_status"}
        _require_keys("quality", value, required)
        try:
            tier = QualityTier(value["tier"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"unknown quality tier: {value.get('tier')!r}") from exc
        return cls(
            tier=tier,
            quality_gate=value["quality_gate"],
            validators_passed=value["validators_passed"],
            review_status=value["review_status"],
            confidence=value.get("confidence"),
            metadata=dict(value.get("metadata", {})),
        )


def route_data_pool(
    *,
    quality_gate: bool,
    public_weight_training_allowed: bool | None,
    evaluation_only: bool = False,
    private_enterprise: bool = False,
) -> DataPool:
    """Route a record into exactly one of the four physically isolated pools.

    Unknown legal status is quarantined.  An explicitly private source can be
    quality-approved for private training without ever becoming public-release
    eligible.
    """

    _require_bool("quality_gate", quality_gate)
    if public_weight_training_allowed is not None:
        _require_bool("public_weight_training_allowed", public_weight_training_allowed)
    _require_bool("evaluation_only", evaluation_only)
    _require_bool("private_enterprise", private_enterprise)
    if evaluation_only:
        return DataPool.EVALUATION_ONLY
    if not quality_gate or public_weight_training_allowed is None:
        return DataPool.QUARANTINED
    if private_enterprise:
        return DataPool.PRIVATE_ENTERPRISE
    if public_weight_training_allowed:
        return DataPool.PUBLIC_RELEASE
    return DataPool.QUARANTINED


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    doc_id: str
    text: str
    entities: tuple[EntitySpan, ...]
    language: str
    domain: str
    scene: str
    license: str
    provenance: Provenance
    quality: QualityMetadata
    public_weight_training_allowed: bool | None
    data_pool: DataPool
    schema_version: str = SCHEMA_VERSION
    tenant_group: str | None = None
    conversation_group: str | None = None
    template_group: str | None = None
    entity_value_groups: tuple[str, ...] = ()
    created_at_bucket: str | None = None
    generator_version: str | None = None
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        _require_non_empty("doc_id", self.doc_id)
        if not isinstance(self.text, str):
            raise SchemaError("text must be a string")
        _require_non_empty("language", self.language)
        _require_non_empty("domain", self.domain)
        _require_non_empty("scene", self.scene)
        _require_non_empty("license", self.license)
        if self.license != self.provenance.license:
            raise SchemaError("top-level license must match provenance.license")
        if not isinstance(self.entities, tuple):
            raise SchemaError("entities must be a tuple in the in-memory canonical form")
        for entity in self.entities:
            if not isinstance(entity, EntitySpan):
                raise SchemaError("entities must contain EntitySpan values")
            entity.validate(self.text)
        self.provenance.validate()
        self.quality.validate()
        expected_pool = route_data_pool(
            quality_gate=self.quality.quality_gate,
            public_weight_training_allowed=self.public_weight_training_allowed,
            evaluation_only=self.provenance.evaluation_only,
            private_enterprise=self.provenance.private_enterprise,
        )
        if not isinstance(self.data_pool, DataPool) or self.data_pool is not expected_pool:
            raise SchemaError(
                f"data_pool must be routed to {expected_pool.value!r}, got "
                f"{getattr(self.data_pool, 'value', self.data_pool)!r}"
            )
        if self.provenance.evaluation_only and self.public_weight_training_allowed is not False:
            raise SchemaError("evaluation-only records must explicitly forbid weight training")
        if self.data_pool is DataPool.PUBLIC_RELEASE and not (
            self.quality.quality_gate and self.public_weight_training_allowed is True
        ):
            raise SchemaError("public-release records require both quality and license gates")
        if (
            self.provenance.synthetic
            and self.generator_version != self.provenance.generator_version
        ):
            raise SchemaError("synthetic generator_version must match provenance.generator_version")
        if not isinstance(self.entity_value_groups, tuple) or any(
            not isinstance(group, str) or not group.startswith("sha256:")
            for group in self.entity_value_groups
        ):
            raise SchemaError("entity_value_groups must contain only sha256: hashes")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("metadata must be a mapping")

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        text: str,
        entities: Iterable[EntitySpan],
        language: str,
        domain: str,
        scene: str,
        provenance: Provenance,
        quality: QualityMetadata,
        public_weight_training_allowed: bool | None,
        **kwargs: Any,
    ) -> DocumentRecord:
        data_pool = route_data_pool(
            quality_gate=quality.quality_gate,
            public_weight_training_allowed=public_weight_training_allowed,
            evaluation_only=provenance.evaluation_only,
            private_enterprise=provenance.private_enterprise,
        )
        record = cls(
            doc_id=doc_id,
            text=text,
            entities=tuple(entities),
            language=language,
            domain=domain,
            scene=scene,
            license=provenance.license,
            provenance=provenance,
            quality=quality,
            public_weight_training_allowed=public_weight_training_allowed,
            data_pool=data_pool,
            **kwargs,
        )
        record.validate()
        return record

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_pool"] = self.data_pool.value
        result["quality"]["tier"] = self.quality.tier.value
        result["entities"] = [dict(entity) for entity in result["entities"]]
        result["entity_value_groups"] = list(self.entity_value_groups)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DocumentRecord:
        allowed = {
            "schema_version",
            "doc_id",
            "text",
            "entities",
            "language",
            "domain",
            "scene",
            "license",
            "provenance",
            "quality",
            "public_weight_training_allowed",
            "data_pool",
            "tenant_group",
            "conversation_group",
            "template_group",
            "entity_value_groups",
            "created_at_bucket",
            "generator_version",
            "split",
            "metadata",
        }
        _reject_unknown("document", value, allowed)
        required = {
            "doc_id",
            "text",
            "entities",
            "language",
            "domain",
            "scene",
            "license",
            "provenance",
            "quality",
            "public_weight_training_allowed",
            "data_pool",
        }
        _require_keys("document", value, required)
        raw_entities = value["entities"]
        if not isinstance(raw_entities, list):
            raise SchemaError("document.entities must be a JSON array")
        if not isinstance(value["provenance"], Mapping):
            raise SchemaError("document.provenance must be an object")
        if not isinstance(value["quality"], Mapping):
            raise SchemaError("document.quality must be an object")
        try:
            pool = DataPool(value["data_pool"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"unknown data pool: {value.get('data_pool')!r}") from exc
        record = cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            doc_id=value["doc_id"],
            text=value["text"],
            entities=tuple(EntitySpan.from_dict(item) for item in raw_entities),
            language=value["language"],
            domain=value["domain"],
            scene=value["scene"],
            license=value["license"],
            provenance=Provenance.from_dict(value["provenance"]),
            quality=QualityMetadata.from_dict(value["quality"]),
            public_weight_training_allowed=value["public_weight_training_allowed"],
            data_pool=pool,
            tenant_group=value.get("tenant_group"),
            conversation_group=value.get("conversation_group"),
            template_group=value.get("template_group"),
            entity_value_groups=tuple(value.get("entity_value_groups", [])),
            created_at_bucket=value.get("created_at_bucket"),
            generator_version=value.get("generator_version"),
            split=value.get("split"),
            metadata=dict(value.get("metadata", {})),
        )
        record.validate()
        return record


def _reject_unknown(context: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise SchemaError(f"unknown {context} fields: {sorted(unknown)}")


def _require_keys(context: str, value: Mapping[str, Any], required: set[str]) -> None:
    missing = required - set(value)
    if missing:
        raise SchemaError(f"missing {context} fields: {sorted(missing)}")


def dumps_document(record: DocumentRecord) -> str:
    record.validate()
    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)


def loads_document(line: str) -> DocumentRecord:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise SchemaError("each JSONL row must be an object")
    return DocumentRecord.from_dict(value)


def iter_jsonl(source: str | Path | TextIO) -> Iterator[DocumentRecord]:
    handle: TextIO
    should_close = False
    if isinstance(source, (str, Path)):
        handle = Path(source).open("r", encoding="utf-8")
        should_close = True
    else:
        handle = source
    try:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield loads_document(line)
            except SchemaError as exc:
                raise SchemaError(f"line {line_number}: {exc}") from exc
    finally:
        if should_close:
            handle.close()


def write_jsonl(records: Iterable[DocumentRecord], target: str | Path | TextIO) -> int:
    handle: TextIO
    should_close = False
    if isinstance(target, (str, Path)):
        handle = Path(target).open("w", encoding="utf-8")
        should_close = True
    else:
        handle = target
    count = 0
    try:
        for record in records:
            handle.write(dumps_document(record) + "\n")
            count += 1
    finally:
        if should_close:
            handle.close()
    return count
