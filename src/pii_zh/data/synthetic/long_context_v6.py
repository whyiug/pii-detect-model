"""Deterministic long-context augmentation used by the v6 development run.

The builder operates only on already-admitted synthetic train/validation
records.  It never imports or reads an evaluation corpus.  The design is
test-informed (long-document over-tagging was observed externally), but no
benchmark text, span, value, or template is copied into the generated rows.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
)

GENERATOR_NAME = "pii_zh_long_context_v6"
GENERATOR_VERSION = "0.6.0-dev1"

_CONNECTORS = (
    "\n\n补充记录：",
    "\n随后又记录：",
    "。另有一项说明：",
    "\n---\n后续信息：",
    "；同时，",
    "\n附注：",
)


class LongContextV6Error(ValueError):
    """Raised without echoing source text, identifiers, or entity values."""


def _digest(*parts: str, length: int = 24) -> str:
    payload = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _source_token(parent: DocumentRecord) -> str:
    return _digest(parent.provenance.source_id, parent.doc_id, length=20)


def _sorted_flat_entities(record: DocumentRecord) -> tuple[EntitySpan, ...]:
    entities = tuple(sorted(record.entities, key=lambda item: (item.start, item.end, item.label)))
    previous_end = 0
    for entity in entities:
        if entity.start < previous_end:
            raise LongContextV6Error("source entities must be a non-overlapping flattened view")
        previous_end = entity.end
    return entities


def convert_record_to_traditional(record: DocumentRecord, converter: Any) -> DocumentRecord:
    """Convert one packed record while rebuilding exact entity offsets.

    Segments are converted independently so entity boundaries remain explicit
    even when OpenCC performs phrase-level substitutions of different length.
    """

    pieces: list[str] = []
    entities: list[EntitySpan] = []
    cursor = 0
    output_cursor = 0
    for entity in _sorted_flat_entities(record):
        prefix = str(converter.convert(record.text[cursor : entity.start]))
        value = str(converter.convert(record.text[entity.start : entity.end]))
        if not value:
            raise LongContextV6Error("traditional conversion removed an entity surface")
        pieces.extend((prefix, value))
        start = output_cursor + len(prefix)
        end = start + len(value)
        entities.append(
            replace(
                entity,
                start=start,
                end=end,
                text=value,
                metadata={**dict(entity.metadata), "script_variant": "zh-Hant-TW"},
            )
        )
        output_cursor = end
        cursor = entity.end
    suffix = str(converter.convert(record.text[cursor:]))
    pieces.append(suffix)
    text = "".join(pieces)
    metadata = {
        **dict(record.metadata),
        "script_variant": "zh-Hant-TW",
        "opencc_config": "s2twp",
    }
    converted = replace(
        record,
        doc_id=f"{record.doc_id}-hant",
        text=text,
        entities=tuple(entities),
        language="zh-Hant-TW",
        template_group=(
            f"sha256:{_digest('hant', record.template_group or record.doc_id, length=64)}"
        ),
        conversation_group=(
            f"v6-hant-{_digest(record.conversation_group or record.doc_id, length=32)}"
        ),
        metadata=metadata,
    )
    converted.validate()
    return converted


def combine_records(
    parents: Sequence[DocumentRecord],
    *,
    split: str,
    variant: str,
    index: int,
    seed: int,
) -> DocumentRecord:
    """Combine complete parent records and shift every character span exactly."""

    if not parents:
        raise LongContextV6Error("a packed document needs at least one parent")
    if split not in {"train", "validation"} or any(parent.split != split for parent in parents):
        raise LongContextV6Error("packed parents must belong to the requested split")
    parent_tokens = tuple(_source_token(parent) for parent in parents)
    identity = _digest(str(seed), split, variant, str(index), *parent_tokens, length=32)
    pieces: list[str] = []
    entities: list[EntitySpan] = []
    cursor = 0
    for parent_index, parent in enumerate(parents):
        connector = "记录开始：" if parent_index == 0 else _CONNECTORS[
            int(_digest(identity, str(parent_index), length=8), 16) % len(_CONNECTORS)
        ]
        pieces.append(connector)
        cursor += len(connector)
        pieces.append(parent.text)
        for entity in _sorted_flat_entities(parent):
            entities.append(
                replace(
                    entity,
                    start=cursor + entity.start,
                    end=cursor + entity.end,
                    metadata={
                        **dict(entity.metadata),
                        "v6_parent_token": parent_tokens[parent_index],
                    },
                )
            )
        cursor += len(parent.text)

    non_entity_groups: set[str] = set()
    entity_value_groups: set[str] = set()
    parent_template_tokens: list[str] = []
    for parent in parents:
        entity_value_groups.update(parent.entity_value_groups)
        raw_non_entity = parent.metadata.get("non_entity_value_groups", [])
        if not isinstance(raw_non_entity, list) or any(
            not isinstance(value, str) for value in raw_non_entity
        ):
            raise LongContextV6Error("parent non-entity groups are malformed")
        non_entity_groups.update(raw_non_entity)
        parent_template_tokens.append(parent.template_group or _source_token(parent))

    text = "".join(pieces)
    source_ids = tuple(sorted({parent.provenance.source_id for parent in parents}))
    provenance = Provenance(
        source_id="pii_zh_v6_long_context_augmentation",
        source_kind="synthetic_derived_long_context",
        source_revision=GENERATOR_VERSION,
        license="Apache-2.0",
        synthetic=True,
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=seed,
        template_id=variant,
        metadata={
            "parent_source_ids": list(source_ids),
            "parent_tokens": list(parent_tokens),
            "test_informed_design": True,
            "benchmark_content_copied": False,
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.SYNTHETIC_VALIDATED,
        quality_gate=True,
        validators_passed=True,
        review_status="deterministic_span_rebuild_validated",
        confidence=1.0,
        metadata={"parent_count": len(parents)},
    )
    record = DocumentRecord.create(
        doc_id=f"v6-{split}-{variant}-{identity}",
        text=text,
        entities=entities,
        language="zh-Hans-CN",
        domain="multi_scene_long_context",
        scene=variant,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        split=split,
        conversation_group=f"v6-pack-{identity}",
        template_group=(
            f"sha256:{_digest(split, variant, *parent_template_tokens, length=64)}"
        ),
        entity_value_groups=tuple(sorted(entity_value_groups)),
        generator_version=GENERATOR_VERSION,
        metadata={
            "v6_slice": variant,
            "parent_count": len(parents),
            "parent_tokens": list(parent_tokens),
            "non_entity_value_groups": sorted(non_entity_groups),
            "test_informed_design": True,
            "tw_pii_bench_text_or_span_copied": False,
        },
    )
    return record


def _token_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def greedy_long_packs(
    records: Sequence[DocumentRecord],
    *,
    tokenizer: Any,
    split: str,
    variant: str,
    seed: int,
    minimum_tokens: int = 128,
    maximum_tokens: int = 192,
    maximum_packs: int | None = None,
) -> tuple[DocumentRecord, ...]:
    """Shuffle then greedily build long records without truncating a parent."""

    if minimum_tokens < 2 or maximum_tokens < minimum_tokens:
        raise LongContextV6Error("invalid token length bounds")
    order = list(records)
    random.Random(seed).shuffle(order)
    packs: list[DocumentRecord] = []
    current: list[DocumentRecord] = []

    def emit() -> None:
        nonlocal current
        if not current or (maximum_packs is not None and len(packs) >= maximum_packs):
            current = []
            return
        candidate = combine_records(
            current,
            split=split,
            variant=variant,
            index=len(packs),
            seed=seed,
        )
        length = _token_length(tokenizer, candidate.text)
        if minimum_tokens <= length <= maximum_tokens:
            packs.append(candidate)
        current = []

    for record in order:
        trial = combine_records(
            (*current, record),
            split=split,
            variant=variant,
            index=len(packs),
            seed=seed,
        )
        if _token_length(tokenizer, trial.text) > maximum_tokens and current:
            emit()
            current = [record]
        else:
            current.append(record)
        if current:
            candidate = combine_records(
                current,
                split=split,
                variant=variant,
                index=len(packs),
                seed=seed,
            )
            if _token_length(tokenizer, candidate.text) >= minimum_tokens:
                emit()
        if maximum_packs is not None and len(packs) >= maximum_packs:
            break
    emit()
    return tuple(packs)


def low_density_packs(
    positives: Sequence[DocumentRecord],
    negatives: Sequence[DocumentRecord],
    *,
    tokenizer: Any,
    split: str,
    seed: int,
    maximum_packs: int,
    minimum_tokens: int = 128,
    maximum_tokens: int = 192,
) -> tuple[DocumentRecord, ...]:
    """Create one-positive-plus-negative long records to reduce entity density."""

    if not positives or not negatives or maximum_packs < 1:
        raise LongContextV6Error("low-density packing needs positive and negative parents")
    positive_order = list(positives)
    negative_order = list(negatives)
    random.Random(seed).shuffle(positive_order)
    random.Random(seed + 1).shuffle(negative_order)
    packs: list[DocumentRecord] = []
    negative_index = 0
    for positive in positive_order:
        parents = [positive]
        attempts = 0
        while attempts < len(negative_order):
            negative = negative_order[negative_index % len(negative_order)]
            negative_index += 1
            attempts += 1
            trial = combine_records(
                (*parents, negative),
                split=split,
                variant="long_low_density",
                index=len(packs),
                seed=seed,
            )
            length = _token_length(tokenizer, trial.text)
            if length > maximum_tokens:
                break
            parents.append(negative)
            if length >= minimum_tokens:
                break
        candidate = combine_records(
            parents,
            split=split,
            variant="long_low_density",
            index=len(packs),
            seed=seed,
        )
        length = _token_length(tokenizer, candidate.text)
        if minimum_tokens <= length <= maximum_tokens:
            packs.append(candidate)
        if len(packs) >= maximum_packs:
            break
    return tuple(packs)


def select_originals(
    records: Sequence[DocumentRecord], *, seed: int, fraction: float
) -> tuple[DocumentRecord, ...]:
    if not 0.0 <= fraction <= 1.0:
        raise LongContextV6Error("original retention fraction must be in [0, 1]")
    order = list(records)
    random.Random(seed).shuffle(order)
    count = round(len(order) * fraction)
    return tuple(order[:count])


def unique_records(records: Iterable[DocumentRecord]) -> tuple[DocumentRecord, ...]:
    """Fail closed on duplicate IDs or exact texts within one derived split."""

    result: list[DocumentRecord] = []
    ids: set[str] = set()
    texts: set[str] = set()
    for record in records:
        text_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        if record.doc_id in ids or text_hash in texts:
            continue
        ids.add(record.doc_id)
        texts.add(text_hash)
        result.append(record)
    if not result:
        raise LongContextV6Error("derived split is empty")
    return tuple(result)


__all__ = [
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "LongContextV6Error",
    "combine_records",
    "convert_record_to_traditional",
    "greedy_long_packs",
    "low_density_packs",
    "select_originals",
    "unique_records",
]
