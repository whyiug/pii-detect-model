"""Strict character-span to fast-tokenizer BIO alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .schema import DocumentRecord, EntitySpan, SchemaError

IGNORE_LABEL = -100


class AlignmentError(ValueError):
    """Raised when a character annotation cannot form a safe token view."""


class FastTokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class UnalignableBoundary:
    entity_index: int
    label: str
    start: int
    end: int
    start_aligned: bool
    end_aligned: bool
    token_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AlignmentStatistics:
    entity_count: int
    unalignable_count: int
    unalignable_boundary_rate: float
    unalignable_start_count: int
    unalignable_end_count: int


@dataclass(frozen=True, slots=True)
class TokenAlignment:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    bio_labels: tuple[str | int, ...]
    unalignable_boundaries: tuple[UnalignableBoundary, ...]
    statistics: AlignmentStatistics

    def label_ids(self, label2id: Mapping[str, int]) -> tuple[int, ...]:
        encoded: list[int] = []
        for label in self.bio_labels:
            if label == IGNORE_LABEL:
                encoded.append(IGNORE_LABEL)
                continue
            try:
                encoded.append(label2id[str(label)])
            except KeyError as exc:
                raise AlignmentError(f"label {label!r} is missing from label2id") from exc
        return tuple(encoded)


def _as_flat_sequence(name: str, value: Any) -> list[Any]:
    if value is None:
        raise AlignmentError(f"tokenizer output is missing {name}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise AlignmentError(f"tokenizer output {name} must be a sequence")
    # Reject batched output: this function intentionally aligns one document.
    if value and isinstance(value[0], list) and name != "offset_mapping":
        if len(value) != 1:
            raise AlignmentError("batched tokenizer output is not supported")
        value = value[0]
    if name == "offset_mapping" and value and len(value) == 1 and isinstance(value[0], list):
        nested = value[0]
        if nested and isinstance(nested[0], (list, tuple)):
            value = nested
    return list(value)


def _validate_offsets(text: str, offsets: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    previous_start = 0
    for index, pair in enumerate(offsets):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AlignmentError(f"offset_mapping[{index}] must contain two integers")
        start, end = pair
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
            raise AlignmentError(f"offset_mapping[{index}] must contain integers")
        if start == end == 0:
            normalized.append((0, 0))
            continue
        if not 0 <= start < end <= len(text):
            raise AlignmentError(
                f"offset_mapping[{index}]={pair!r} is outside text length {len(text)}"
            )
        if start < previous_start:
            raise AlignmentError("token offsets must be monotonically ordered")
        previous_start = start
        normalized.append((start, end))
    return tuple(normalized)


def _intersecting_tokens(
    entity: EntitySpan,
    offsets: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < entity.end and token_end > entity.start
    )


def align_document_to_bio(
    record: DocumentRecord,
    tokenizer: FastTokenizer,
    *,
    max_length: int | None = None,
    truncation: bool = False,
    add_special_tokens: bool = True,
) -> TokenAlignment:
    """Create a BIO training view while preserving character truth.

    Entity boundaries that fall within a BPE token are reported, not silently
    rewritten.  Intersecting tokens receive BIO labels, matching the documented
    baseline expansion policy.  Token collisions and truncated entities fail
    closed because they require an explicit flattening/windowing decision.
    """

    try:
        record.validate()
    except SchemaError as exc:
        raise AlignmentError(f"invalid source record: {exc}") from exc
    if not getattr(tokenizer, "is_fast", False):
        raise AlignmentError("a fast tokenizer with offset_mapping is required")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
    ):
        raise AlignmentError("max_length must be a positive integer or null")
    kwargs: dict[str, Any] = {
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
        "add_special_tokens": add_special_tokens,
        "truncation": truncation,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(record.text, **kwargs)
    input_ids_raw = _as_flat_sequence("input_ids", encoded.get("input_ids"))
    offsets_raw = _as_flat_sequence("offset_mapping", encoded.get("offset_mapping"))
    attention_raw = encoded.get("attention_mask", [1] * len(input_ids_raw))
    attention_mask_raw = _as_flat_sequence("attention_mask", attention_raw)
    special_raw = encoded.get("special_tokens_mask", [0] * len(input_ids_raw))
    special_tokens_mask = _as_flat_sequence("special_tokens_mask", special_raw)
    offsets = _validate_offsets(record.text, offsets_raw)
    if not (
        len(input_ids_raw) == len(attention_mask_raw) == len(special_tokens_mask) == len(offsets)
    ):
        raise AlignmentError(
            "input_ids, attention_mask, special_tokens_mask, and offsets must have equal lengths"
        )
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in input_ids_raw
    ):
        raise AlignmentError("input_ids must contain integers")
    if any(mask not in (0, 1) for mask in attention_mask_raw):
        raise AlignmentError("attention_mask must contain only 0/1")
    if any(mask not in (0, 1) for mask in special_tokens_mask):
        raise AlignmentError("special_tokens_mask must contain only 0/1")

    labels: list[str | int] = [
        (
            IGNORE_LABEL
            if start == end or attention_mask_raw[index] == 0 or special_tokens_mask[index] == 1
            else "O"
        )
        for index, (start, end) in enumerate(offsets)
    ]
    occupied: dict[int, int] = {}
    unalignable: list[UnalignableBoundary] = []
    for entity_index, entity in enumerate(record.entities):
        token_indices = _intersecting_tokens(entity, offsets)
        if not token_indices:
            raise AlignmentError(
                f"entity {entity_index} {entity.label} [{entity.start}, {entity.end}) "
                "has no token coverage (possibly truncated or whitespace-only)"
            )
        if any(
            attention_mask_raw[token_index] == 0 or special_tokens_mask[token_index] == 1
            for token_index in token_indices
        ):
            raise AlignmentError(f"entity {entity_index} intersects a padding or special token")
        first_start = offsets[token_indices[0]][0]
        last_end = offsets[token_indices[-1]][1]
        if first_start > entity.start or last_end < entity.end:
            raise AlignmentError(
                f"entity {entity_index} [{entity.start}, {entity.end}) is only "
                f"partially covered by tokenizer offsets [{first_start}, {last_end}); "
                "do not silently train on truncated entities"
            )
        start_aligned = offsets[token_indices[0]][0] == entity.start
        end_aligned = offsets[token_indices[-1]][1] == entity.end
        if not start_aligned or not end_aligned:
            unalignable.append(
                UnalignableBoundary(
                    entity_index=entity_index,
                    label=entity.label,
                    start=entity.start,
                    end=entity.end,
                    start_aligned=start_aligned,
                    end_aligned=end_aligned,
                    token_indices=token_indices,
                )
            )
        for position, token_index in enumerate(token_indices):
            if token_index in occupied:
                other = occupied[token_index]
                raise AlignmentError(
                    f"entities {other} and {entity_index} both claim token {token_index}; "
                    "flatten overlapping spans before BIO alignment"
                )
            occupied[token_index] = entity_index
            prefix = "B" if position == 0 else "I"
            labels[token_index] = f"{prefix}-{entity.label}"

    count = len(record.entities)
    stats = AlignmentStatistics(
        entity_count=count,
        unalignable_count=len(unalignable),
        unalignable_boundary_rate=(len(unalignable) / count if count else 0.0),
        unalignable_start_count=sum(not item.start_aligned for item in unalignable),
        unalignable_end_count=sum(not item.end_aligned for item in unalignable),
    )
    return TokenAlignment(
        input_ids=tuple(input_ids_raw),
        attention_mask=tuple(attention_mask_raw),
        offset_mapping=offsets,
        bio_labels=tuple(labels),
        unalignable_boundaries=tuple(unalignable),
        statistics=stats,
    )
