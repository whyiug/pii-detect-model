"""Offset-safe replacement utilities and conservative coverage merging."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplacementSpan:
    entity_type: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, str) or not self.entity_type:
            raise TypeError("replacement span entity_type must be a non-empty string")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("replacement span start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("replacement span end must be an integer")
        if not 0 <= self.start < self.end:
            raise ValueError("replacement span must be a non-empty half-open range")


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


def merge_replacement_coverage(
    spans: Sequence[Any],
    *,
    mixed_entity_type: str = "PII",
) -> list[ReplacementSpan]:
    """Merge overlapping accepted spans without losing sensitive coverage.

    Extraction may select one label from a group of conflicting candidates,
    but redaction has a stricter safety objective: every character covered by
    any accepted candidate must remain covered.  Connected overlapping spans
    therefore become one replacement interval.  A component containing more
    than one entity type receives the explicit ``mixed_entity_type`` label so
    mapping-based replacement policies cannot silently pick one loser's type.

    Adjacent spans are intentionally kept separate because neither leaves an
    uncovered character.  Invalid or empty half-open spans fail closed before
    any replacement is attempted.
    """

    if not isinstance(mixed_entity_type, str) or not mixed_entity_type:
        raise TypeError("mixed_entity_type must be a non-empty string")
    normalized = sorted(
        (_span(value) for value in spans),
        key=lambda item: (item.start, item.end, item.entity_type),
    )
    for span in normalized:
        if not 0 <= span.start < span.end:
            raise ValueError("replacement span must be a non-empty half-open range")
    if not normalized:
        return []

    merged: list[ReplacementSpan] = []
    current_start = normalized[0].start
    current_end = normalized[0].end
    current_types = {normalized[0].entity_type}

    def append_current() -> None:
        entity_type = next(iter(current_types)) if len(current_types) == 1 else mixed_entity_type
        merged.append(
            ReplacementSpan(
                entity_type=entity_type,
                start=current_start,
                end=current_end,
            )
        )

    for span in normalized[1:]:
        if span.start < current_end:
            current_end = max(current_end, span.end)
            current_types.add(span.entity_type)
            continue
        append_current()
        current_start = span.start
        current_end = span.end
        current_types = {span.entity_type}
    append_current()
    return merged


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
        replacement_value: object
        if callable(replacement):
            replacement_value = replacement(span, original)
        elif isinstance(replacement, Mapping):
            # Mapping is covariant in its value type.  Widening it to object
            # lets the optional default retain the original single ``get``
            # lookup semantics without weakening the public ReplacementSpec.
            replacement_values: Mapping[str, object] = replacement
            replacement_value = replacement_values.get(span.entity_type, default_replacement)
            if replacement_value is None:
                raise KeyError(f"no replacement configured for {span.entity_type}")
        else:
            replacement_value = replacement
        if not isinstance(replacement_value, str):
            raise TypeError("replacement values must be strings")
        output = output[: span.start] + replacement_value + output[span.end :]
    return output


# Readable synonym used by policy/anonymizer adapters.
replace_spans = apply_replacements


__all__ = [
    "ReplacementFactory",
    "ReplacementSpan",
    "ReplacementSpec",
    "apply_replacements",
    "merge_replacement_coverage",
    "replace_spans",
]
