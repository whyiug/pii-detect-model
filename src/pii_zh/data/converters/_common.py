"""Shared strict parsing helpers for already-downloaded dataset rows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..schema import EntitySpan, stable_text_hash


class ConversionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int
    label: str
    text: str
    metadata: Mapping[str, Any]


def get_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "source_text", "content", "sentence"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ConversionError("row is missing a string text/source_text/content/sentence field")


def get_spans(row: Mapping[str, Any], text: str) -> list[SourceSpan]:
    raw_spans = None
    for key in ("entities", "spans", "annotations", "privacy_mask", "pii"):
        if key in row:
            raw_spans = row[key]
            break
    if raw_spans is None:
        return []
    if not isinstance(raw_spans, list):
        raise ConversionError("entity/span field must be an array")
    spans: list[SourceSpan] = []
    for index, item in enumerate(raw_spans):
        if not isinstance(item, Mapping):
            raise ConversionError(f"span {index} must be an object")
        start = _first(item, "start", "start_offset", "begin")
        end = _first(item, "end", "end_offset", "stop")
        label = _first(item, "label", "entity_type", "type", "category")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ConversionError(f"span {index} start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ConversionError(f"span {index} end must be an integer")
        if not isinstance(label, str) or not label.strip():
            raise ConversionError(f"span {index} label must be a non-empty string")
        if not 0 <= start < end <= len(text):
            raise ConversionError(f"span {index} [{start}, {end}) is outside text")
        quote = item.get("text", item.get("value", item.get("quote", text[start:end])))
        if quote != text[start:end]:
            raise ConversionError(f"span {index} quote mismatch: {quote!r} != text[{start}:{end}]")
        spans.append(
            SourceSpan(
                start=start,
                end=end,
                label=label.strip(),
                text=quote,
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"text", "value", "quote"}
                },
            )
        )
    return sorted(spans, key=lambda span: (span.start, span.end, span.label))


def _first(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def stable_doc_id(source_id: str, row: Mapping[str, Any], text: str) -> str:
    raw_id = row.get("doc_id", row.get("id", row.get("uid")))
    material = f"{source_id}:{raw_id if raw_id is not None else text}"
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def ensure_non_overlapping(entities: Iterable[EntitySpan]) -> tuple[EntitySpan, ...]:
    result = sorted(entities, key=lambda entity: (entity.start, entity.end, entity.label))
    deduplicated: list[EntitySpan] = []
    for entity in result:
        if deduplicated and (
            entity.start,
            entity.end,
            entity.label,
        ) == (
            deduplicated[-1].start,
            deduplicated[-1].end,
            deduplicated[-1].label,
        ):
            continue
        if deduplicated and entity.start < deduplicated[-1].end:
            raise ConversionError(
                "overlapping source spans require an explicit taxonomy flattening decision: "
                f"{deduplicated[-1]} vs {entity}"
            )
        deduplicated.append(entity)
    return tuple(deduplicated)


def entity_value_groups(entities: Iterable[EntitySpan]) -> tuple[str, ...]:
    return tuple(sorted({stable_text_hash(entity.text) for entity in entities}))
