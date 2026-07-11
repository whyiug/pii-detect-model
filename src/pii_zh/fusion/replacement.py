"""Offset-safe replacement utilities for already fused non-overlapping spans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplacementSpan:
    entity_type: str
    start: int
    end: int


ReplacementFactory = Callable[[ReplacementSpan, str], str]
ReplacementSpec = str | Mapping[str, str] | ReplacementFactory


def _span(value: Any) -> ReplacementSpan:
    if isinstance(value, ReplacementSpan):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        if isinstance(value[0], str):
            entity_type, start, end = value
        else:
            start, end, entity_type = value
    elif isinstance(value, Mapping):
        start, end = value.get("start"), value.get("end")
        entity_type = value.get("entity_type", value.get("label"))
    else:
        start = getattr(value, "start", None)
        end = getattr(value, "end", None)
        entity_type = getattr(value, "entity_type", getattr(value, "label", None))
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("replacement span start must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise TypeError("replacement span end must be an integer")
    if not isinstance(entity_type, str) or not entity_type:
        raise TypeError("replacement span entity_type must be a non-empty string")
    return ReplacementSpan(entity_type=entity_type, start=start, end=end)


def apply_replacements(
    text: str,
    spans: Sequence[Any],
    replacement: ReplacementSpec = "<PII>",
    *,
    default_replacement: str | None = None,
) -> str:
    """Replace spans from right to left so original offsets remain valid."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = sorted((_span(value) for value in spans), key=lambda item: (item.start, item.end))
    previous_end = 0
    for span in normalized:
        if not 0 <= span.start < span.end <= len(text):
            raise ValueError("replacement span lies outside text")
        if span.start < previous_end:
            raise ValueError("replacement spans must not overlap; fuse them first")
        previous_end = span.end

    output = text
    for span in reversed(normalized):
        original = text[span.start : span.end]
        if callable(replacement):
            value = replacement(span, original)
        elif isinstance(replacement, Mapping):
            value = replacement.get(span.entity_type, default_replacement)
            if value is None:
                raise KeyError(f"no replacement configured for {span.entity_type}")
        else:
            value = replacement
        if not isinstance(value, str):
            raise TypeError("replacement values must be strings")
        output = output[: span.start] + value + output[span.end :]
    return output


# Readable synonym used by policy/anonymizer adapters.
replace_spans = apply_replacements


__all__ = [
    "ReplacementFactory",
    "ReplacementSpan",
    "ReplacementSpec",
    "apply_replacements",
    "replace_spans",
]
