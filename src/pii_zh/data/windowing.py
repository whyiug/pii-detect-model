"""Deterministic entity-safe character/token window planning.

The planner operates only on a document length, tokenizer offsets, and
value-free gold spans.  It never needs raw text or entity values, which keeps
validation failures safe to log.  Zero-width tokenizer entries (normally
special tokens) are ignored for the content-token budget but their source
indices remain available on every planned window.

``max_tokens`` is the non-zero-width content-token budget.  Callers must
reserve any model-added special tokens before invoking the planner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_MINIMUM_OVERLAP_FRACTION = 0.20
_MAXIMUM_OVERLAP_FRACTION = 0.25


class WindowPlanningError(ValueError):
    """Raised when safe, complete entity windows cannot be constructed."""


@dataclass(frozen=True, slots=True, order=True)
class WindowGoldSpan:
    """A value-free, left-closed/right-open gold character span."""

    start: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class WindowedSpan:
    """One uniquely owned gold span with global and local character offsets."""

    span_index: int
    label: str
    global_start: int
    global_end: int
    local_start: int
    local_end: int


@dataclass(frozen=True, slots=True)
class CharacterWindow:
    """A content-token window and the global character interval it owns.

    ``token_start`` and ``token_end`` index the filtered sequence of non-zero
    offsets.  ``source_token_indices`` maps that range back to indices in the
    caller's original offset mapping, including any skipped special tokens.
    """

    window_id: int
    token_start: int
    token_end: int
    source_token_indices: tuple[int, ...]
    start: int
    end: int
    owner_start: int
    owner_end: int
    spans: tuple[WindowedSpan, ...]

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True, slots=True)
class _ContentToken:
    source_index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _CanonicalSpan:
    index: int
    start: int
    end: int
    label: str
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class _WindowBounds:
    token_start: int
    token_end: int
    start: int
    end: int


def _validate_text_length(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WindowPlanningError("text_length must be a non-negative integer")
    return value


def _validate_max_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WindowPlanningError("max_tokens must be a positive integer")
    return value


def _validate_overlap_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WindowPlanningError("overlap_fraction must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not (
        _MINIMUM_OVERLAP_FRACTION <= normalized <= _MAXIMUM_OVERLAP_FRACTION
    ):
        raise WindowPlanningError("overlap_fraction must be between 0.20 and 0.25")
    return normalized


def _normalize_offsets(
    offsets: Sequence[Sequence[int]],
    *,
    text_length: int,
) -> tuple[_ContentToken, ...]:
    if isinstance(offsets, (str, bytes)) or not isinstance(offsets, Sequence):
        raise TypeError("offset_mapping must be a sequence")
    content: list[_ContentToken] = []
    previous_start = 0
    previous_end = 0
    for source_index, pair in enumerate(offsets):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise WindowPlanningError(f"offset_mapping entry {source_index} is invalid")
        start, end = pair
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise WindowPlanningError(f"offset_mapping entry {source_index} must contain integers")
        if not 0 <= start <= end <= text_length:
            raise WindowPlanningError(f"offset_mapping entry {source_index} is out of bounds")
        if start == end:
            continue
        if content and (start < previous_start or end < previous_end):
            raise WindowPlanningError("content token offsets must be monotonic")
        if not content and start != 0:
            raise WindowPlanningError("content token offsets do not cover the document start")
        if content and start > previous_end:
            raise WindowPlanningError("content token offsets contain a character gap")
        content.append(_ContentToken(source_index=source_index, start=start, end=end))
        previous_start = start
        previous_end = end

    if text_length == 0:
        if content:
            raise WindowPlanningError("empty documents cannot contain content token offsets")
        return ()
    if not content:
        raise WindowPlanningError("non-empty documents require content token offsets")
    if content[-1].end != text_length:
        raise WindowPlanningError("content token offsets do not cover the document end")
    return tuple(content)


def _normalize_gold_spans(
    spans: tuple[WindowGoldSpan, ...],
    *,
    text_length: int,
    content: tuple[_ContentToken, ...],
    max_tokens: int,
) -> tuple[_CanonicalSpan, ...]:
    raw_keys: list[tuple[int, int, str]] = []
    for source_index, span in enumerate(spans):
        if isinstance(span.start, bool) or not isinstance(span.start, int):
            raise WindowPlanningError(f"gold span {source_index} has an invalid start offset")
        if isinstance(span.end, bool) or not isinstance(span.end, int):
            raise WindowPlanningError(f"gold span {source_index} has an invalid end offset")
        if not 0 <= span.start < span.end <= text_length:
            raise WindowPlanningError(f"gold span {source_index} is out of bounds")
        if not isinstance(span.label, str) or not span.label.strip():
            raise WindowPlanningError(f"gold span {source_index} has an invalid label")
        raw_keys.append((span.start, span.end, span.label))
    if len(set(raw_keys)) != len(raw_keys):
        raise WindowPlanningError("duplicate gold spans are not permitted")

    canonical: list[_CanonicalSpan] = []
    for index, span in enumerate(sorted(spans)):
        covered = tuple(
            token_index
            for token_index, token in enumerate(content)
            if token.start < span.end and token.end > span.start
        )
        if not covered:
            raise WindowPlanningError(f"gold span {index} has no content token coverage")
        token_start = covered[0]
        token_end = covered[-1] + 1
        if content[token_start].start > span.start or content[token_end - 1].end < span.end:
            raise WindowPlanningError(f"gold span {index} has incomplete token coverage")
        if token_end - token_start > max_tokens:
            raise WindowPlanningError(f"gold span {index} exceeds one token window")
        canonical.append(
            _CanonicalSpan(
                index=index,
                start=span.start,
                end=span.end,
                label=span.label,
                token_start=token_start,
                token_end=token_end,
            )
        )
    return tuple(canonical)


def _materialize_gold_spans(spans: Iterable[WindowGoldSpan]) -> tuple[WindowGoldSpan, ...]:
    try:
        values = tuple(spans)
    except Exception:
        raise TypeError("gold spans must be iterable") from None
    if any(not isinstance(span, WindowGoldSpan) for span in values):
        raise TypeError("gold spans must contain only WindowGoldSpan values")
    return values


def _protected_token_ranges(
    spans: tuple[_CanonicalSpan, ...],
    *,
    content: tuple[_ContentToken, ...],
    max_tokens: int,
) -> tuple[tuple[int, int], ...]:
    ranges = [(span.token_start, span.token_end) for span in spans]
    group_start = 0
    for boundary in range(1, len(content) + 1):
        if boundary < len(content) and content[boundary].start < content[boundary - 1].end:
            continue
        if boundary - group_start > 1:
            ranges.append((group_start, boundary))
        group_start = boundary

    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start >= merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    if any(end - start > max_tokens for start, end in merged):
        raise WindowPlanningError("an indivisible token range cannot fit in one window")
    return tuple((start, end) for start, end in merged)


def _containing_range(
    boundary: int,
    protected: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    return next(
        ((start, end) for start, end in protected if start < boundary < end),
        None,
    )


def _plan_token_ranges(
    *,
    content_token_count: int,
    max_tokens: int,
    overlap_fraction: float,
    protected: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    def safe_end(token_start: int) -> int:
        token_end = min(token_start + max_tokens, content_token_count)
        cut = _containing_range(token_end, protected)
        return cut[0] if cut is not None else token_end

    ranges: list[tuple[int, int]] = []
    token_start = 0
    while token_start < content_token_count:
        token_end = safe_end(token_start)
        if token_end <= token_start:
            raise WindowPlanningError("entity-safe token boundaries cannot make progress")
        if ranges and token_end <= ranges[-1][1]:
            raise WindowPlanningError("entity-safe token windows cannot extend coverage")
        ranges.append((token_start, token_end))
        if token_end == content_token_count:
            break

        width = token_end - token_start
        requested_overlap = min(
            max(1, round(width * overlap_fraction)),
            width - 1,
        )
        desired_start = token_end - requested_overlap
        candidates = (
            candidate
            for candidate in range(token_start + 1, token_end + 1)
            if _containing_range(candidate, protected) is None and safe_end(candidate) > token_end
        )
        try:
            next_start = min(
                candidates,
                key=lambda candidate: (abs(candidate - desired_start), candidate),
            )
        except ValueError:
            raise WindowPlanningError("entity-safe overlap cannot make progress") from None
        token_start = next_start
    return tuple(ranges)


def _char_window_bounds(
    token_ranges: tuple[tuple[int, int], ...],
    content: tuple[_ContentToken, ...],
) -> tuple[_WindowBounds, ...]:
    return tuple(
        _WindowBounds(
            token_start=token_start,
            token_end=token_end,
            start=content[token_start].start,
            end=content[token_end - 1].end,
        )
        for token_start, token_end in token_ranges
    )


def _safe_owner_boundary(
    *,
    left: _WindowBounds,
    right: _WindowBounds,
    spans: tuple[_CanonicalSpan, ...],
) -> int:
    overlap_start = right.start
    overlap_end = left.end
    if overlap_start > overlap_end:
        raise WindowPlanningError("adjacent token windows leave an ownership gap")
    desired = (overlap_start + overlap_end) // 2
    candidates = {overlap_start, overlap_end, desired}
    for span in spans:
        if overlap_start <= span.start <= overlap_end:
            candidates.add(span.start)
        if overlap_start <= span.end <= overlap_end:
            candidates.add(span.end)
    safe = [
        candidate
        for candidate in candidates
        if not any(span.start < candidate < span.end for span in spans)
    ]
    if not safe:
        raise WindowPlanningError("no entity-safe ownership boundary is available")
    return min(safe, key=lambda candidate: (abs(candidate - desired), candidate))


def _ownership_boundaries(
    bounds: tuple[_WindowBounds, ...],
    spans: tuple[_CanonicalSpan, ...],
) -> tuple[int, ...]:
    boundaries = tuple(
        _safe_owner_boundary(left=left, right=right, spans=spans)
        for left, right in zip(bounds, bounds[1:], strict=False)
    )
    if any(
        current > following for current, following in zip(boundaries, boundaries[1:], strict=False)
    ):
        raise WindowPlanningError("ownership boundaries are not monotonic")
    return boundaries


def plan_entity_safe_windows(
    *,
    text_length: int,
    offset_mapping: Sequence[Sequence[int]],
    gold_spans: Iterable[WindowGoldSpan],
    max_tokens: int,
    overlap_fraction: float = 0.25,
) -> tuple[CharacterWindow, ...]:
    """Plan deterministic windows and assign every gold span exactly once."""

    normalized_length = _validate_text_length(text_length)
    normalized_max_tokens = _validate_max_tokens(max_tokens)
    normalized_overlap = _validate_overlap_fraction(overlap_fraction)
    content = _normalize_offsets(offset_mapping, text_length=normalized_length)
    raw_spans = _materialize_gold_spans(gold_spans)
    if not content:
        if raw_spans:
            raise WindowPlanningError("empty documents cannot contain gold spans")
        return ()

    spans = _normalize_gold_spans(
        raw_spans,
        text_length=normalized_length,
        content=content,
        max_tokens=normalized_max_tokens,
    )
    protected = _protected_token_ranges(
        spans,
        content=content,
        max_tokens=normalized_max_tokens,
    )
    token_ranges = _plan_token_ranges(
        content_token_count=len(content),
        max_tokens=normalized_max_tokens,
        overlap_fraction=normalized_overlap,
        protected=protected,
    )
    bounds = _char_window_bounds(token_ranges, content)
    owner_boundaries = _ownership_boundaries(bounds, spans)

    owners: dict[int, int] = {}
    owner_intervals: list[tuple[int, int]] = []
    for window_id, bound in enumerate(bounds):
        owner_start = 0 if window_id == 0 else owner_boundaries[window_id - 1]
        owner_end = (
            normalized_length if window_id == len(bounds) - 1 else owner_boundaries[window_id]
        )
        if not bound.start <= owner_start < owner_end <= bound.end:
            raise WindowPlanningError("window ownership interval is invalid")
        owner_intervals.append((owner_start, owner_end))

    for span in spans:
        matching = [
            window_id
            for window_id, (owner_start, owner_end) in enumerate(owner_intervals)
            if owner_start <= span.start and span.end <= owner_end
        ]
        if len(matching) != 1:
            raise WindowPlanningError(f"gold span {span.index} lacks one unique owner")
        window_id = matching[0]
        bound = bounds[window_id]
        if not bound.start <= span.start < span.end <= bound.end:
            raise WindowPlanningError(f"gold span {span.index} is incomplete in its owner")
        owners[span.index] = window_id

    planned: list[CharacterWindow] = []
    for window_id, bound in enumerate(bounds):
        owner_start, owner_end = owner_intervals[window_id]
        owned_spans = tuple(
            WindowedSpan(
                span_index=span.index,
                label=span.label,
                global_start=span.start,
                global_end=span.end,
                local_start=span.start - bound.start,
                local_end=span.end - bound.start,
            )
            for span in spans
            if owners[span.index] == window_id
        )
        planned.append(
            CharacterWindow(
                window_id=window_id,
                token_start=bound.token_start,
                token_end=bound.token_end,
                source_token_indices=tuple(
                    token.source_index for token in content[bound.token_start : bound.token_end]
                ),
                start=bound.start,
                end=bound.end,
                owner_start=owner_start,
                owner_end=owner_end,
                spans=owned_spans,
            )
        )
    if sum(len(window.spans) for window in planned) != len(spans):
        raise RuntimeError("window planner ownership invariant failed")
    return tuple(planned)


__all__ = [
    "CharacterWindow",
    "WindowGoldSpan",
    "WindowPlanningError",
    "WindowedSpan",
    "plan_entity_safe_windows",
]
