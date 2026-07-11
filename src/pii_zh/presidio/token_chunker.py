"""Tokenizer-aware long-text windows with deterministic center ownership."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

_STRUCTURAL_BOUNDARIES = re.compile(
    r"(?:\r?\n)+"  # paragraph/table row
    r"|[。！？!?；;]+"  # sentence/clause
    r"|\t+"  # table cell
    r"|[}\]]\s*[,，]?"  # JSON object/array end
    r"|[,，]\s*(?=(?:\"[^\"\n]+\"|'[^'\n]+'|[\w\u4e00-\u9fff]+)\s*[:：])"
)


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """A model window and the global character interval it owns."""

    window_id: int
    text: str
    start: int
    end: int
    owner_start: int
    owner_end: int
    token_start: int
    token_end: int
    token_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.owner_start <= self.owner_end <= self.end:
            raise ValueError("window and ownership character offsets are inconsistent")
        if self.token_count != self.token_end - self.token_start or self.token_count < 1:
            raise ValueError("token_count must match the non-empty token interval")
        if len(self.text) != self.end - self.start:
            raise ValueError("window text length must match its global character interval")

    def owns(self, start: int, end: int) -> bool:
        """Return whether this window owns the center of a global span."""

        if not self.start <= start < end <= self.end:
            return False
        doubled_center = start + end
        left_ok = doubled_center >= 2 * self.owner_start
        if self.owner_end == self.end:
            right_ok = doubled_center <= 2 * self.owner_end
        else:
            right_ok = doubled_center < 2 * self.owner_end
        return left_ok and right_ok

    def to_global(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.text):
            raise ValueError("relative span lies outside its token window")
        return self.start + start, self.start + end


class TokenizerAwareChunker:
    """Create overlapping token windows, preferring Chinese structure boundaries.

    ``max_tokens`` includes any special tokens the tokenizer reports through
    ``num_special_tokens_to_add``.  Overlap defaults to 25% and is constrained
    to the required 20%--25% range.  Result ownership partitions the document
    at overlap centers, so batch and single-item prediction use one canonical
    window for each span.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_tokens: int = 512,
        stride_fraction: float = 0.25,
        minimum_structural_fraction: float = 0.60,
    ) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 4:
            raise ValueError("max_tokens must be an integer of at least four")
        stride_fraction = float(stride_fraction)
        if not 0.20 <= stride_fraction <= 0.25:
            raise ValueError("stride_fraction must be between 0.20 and 0.25")
        minimum_structural_fraction = float(minimum_structural_fraction)
        if not 0.5 <= minimum_structural_fraction <= 1.0:
            raise ValueError("minimum_structural_fraction must be between 0.5 and 1.0")
        special_tokens = self._special_token_count(tokenizer)
        if special_tokens >= max_tokens:
            raise ValueError("max_tokens must leave room for at least one content token")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.content_tokens = max_tokens - special_tokens
        self.stride_fraction = stride_fraction
        self.minimum_structural_fraction = minimum_structural_fraction

    @staticmethod
    def _special_token_count(tokenizer: Any) -> int:
        counter = getattr(tokenizer, "num_special_tokens_to_add", None)
        if not callable(counter):
            return 0
        try:
            count = counter(pair=False)
        except TypeError:
            count = counter(False)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TypeError("tokenizer.num_special_tokens_to_add must return a non-negative int")
        return count

    def _token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
        except TypeError as exc:
            raise TypeError(
                "tokenizer must support return_offsets_mapping for exact character windows"
            ) from exc
        if not isinstance(encoded, Mapping) and not hasattr(encoded, "get"):
            raise TypeError("tokenizer output must expose an offset_mapping")
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise TypeError("tokenizer did not return offset_mapping")
        # Some simple adapters retain a batch axis even for one string.
        if (
            isinstance(offsets, Sequence)
            and len(offsets) == 1
            and isinstance(offsets[0], Sequence)
            and offsets[0]
            and isinstance(offsets[0][0], Sequence)
        ):
            offsets = offsets[0]
        normalized: list[tuple[int, int]] = []
        last_start = 0
        for raw_offset in offsets:
            if not isinstance(raw_offset, Sequence) or len(raw_offset) != 2:
                raise TypeError("each tokenizer offset must be a (start, end) pair")
            start, end = raw_offset
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
            ):
                raise TypeError("tokenizer offsets must contain integers")
            if start == end:
                # Special tokens are not content and cannot define a precise
                # source window even if a tokenizer emits them unexpectedly.
                continue
            if start < last_start or start < 0 or end <= start or end > len(text):
                raise ValueError("tokenizer returned invalid or non-monotonic character offsets")
            normalized.append((start, end))
            last_start = start
        if text and not normalized:
            raise ValueError("tokenizer produced no content offsets for non-empty text")
        return tuple(normalized)

    @staticmethod
    def _structural_token_boundaries(text: str, offsets: Sequence[tuple[int, int]]) -> set[int]:
        char_boundaries = {match.end() for match in _STRUCTURAL_BOUNDARIES.finditer(text)}
        token_boundaries: set[int] = set()
        for index, (_, end) in enumerate(offsets, start=1):
            next_start = offsets[index][0] if index < len(offsets) else end
            if any(end <= boundary <= next_start for boundary in char_boundaries):
                token_boundaries.add(index)
        return token_boundaries

    def chunk(self, text: str) -> list[TokenWindow]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text or not text.strip():
            return []
        offsets = self._token_offsets(text)
        structural_boundaries = self._structural_token_boundaries(text, offsets)
        token_ranges: list[tuple[int, int]] = []
        token_start = 0
        while token_start < len(offsets):
            hard_end = min(token_start + self.content_tokens, len(offsets))
            token_end = hard_end
            if hard_end < len(offsets):
                minimum_end = token_start + max(
                    1, int(self.content_tokens * self.minimum_structural_fraction)
                )
                preferred = [
                    boundary
                    for boundary in structural_boundaries
                    if minimum_end <= boundary <= hard_end
                ]
                if preferred:
                    token_end = max(preferred)
            token_ranges.append((token_start, token_end))
            if token_end == len(offsets):
                break
            window_size = token_end - token_start
            overlap = max(1, round(window_size * self.stride_fraction))
            overlap = min(overlap, window_size - 1)
            token_start = token_end - overlap

        windows: list[TokenWindow] = []
        for window_id, (start_token, end_token) in enumerate(token_ranges):
            start = 0 if window_id == 0 else offsets[start_token][0]
            end = len(text) if window_id == len(token_ranges) - 1 else offsets[end_token - 1][1]
            windows.append(
                TokenWindow(
                    window_id=window_id,
                    text=text[start:end],
                    start=start,
                    end=end,
                    owner_start=start,
                    owner_end=end,
                    token_start=start_token,
                    token_end=end_token,
                    token_count=end_token - start_token,
                )
            )

        ownership_boundaries: list[int] = []
        for current, following in zip(windows, windows[1:], strict=False):
            overlap_start = following.start
            overlap_end = current.end
            if overlap_start >= overlap_end:
                boundary = overlap_start
            else:
                boundary = (overlap_start + overlap_end) // 2
            ownership_boundaries.append(boundary)
        for index, window in enumerate(windows):
            owner_start = window.start if index == 0 else ownership_boundaries[index - 1]
            owner_end = window.end if index == len(windows) - 1 else ownership_boundaries[index]
            windows[index] = replace(window, owner_start=owner_start, owner_end=owner_end)
        return windows

    __call__ = chunk


# A short alias keeps configuration and downstream imports readable.
TokenChunker = TokenizerAwareChunker


__all__ = ["TokenChunker", "TokenWindow", "TokenizerAwareChunker"]
