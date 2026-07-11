"""BIO repair and tokenizer-offset decoding utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DecodedSpan:
    """A character-level entity span using left-closed, right-open offsets."""

    entity_type: str
    start: int
    end: int
    score: float | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, str | int | float | None]:
        return asdict(self)


def _parse_bio_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    prefix, separator, entity_type = tag.partition("-")
    if separator != "-" or prefix not in {"B", "I"} or not entity_type:
        raise ValueError(f"Invalid BIO tag {tag!r}; expected 'O', 'B-<TYPE>', or 'I-<TYPE>'.")
    return prefix, entity_type


def repair_bio_tags(tags: Sequence[str]) -> list[str]:
    """Repair illegal ``I`` transitions deterministically.

    An ``I-X`` at sequence start, after ``O``, or after another entity type is
    converted to ``B-X``.  Structurally malformed labels still raise because
    guessing their intended taxonomy would hide data/model-contract errors.
    """

    repaired: list[str] = []
    active_type: str | None = None
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError(f"BIO tags must be strings, received {type(tag).__name__}.")
        prefix, entity_type = _parse_bio_tag(tag)
        if prefix == "O":
            repaired.append("O")
            active_type = None
        elif prefix == "B":
            repaired.append(tag)
            active_type = entity_type
        elif active_type != entity_type:
            repaired.append(f"B-{entity_type}")
            active_type = entity_type
        else:
            repaired.append(tag)
    return repaired


def decode_bio_spans(
    tags: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    *,
    text: str | None = None,
    token_scores: Sequence[float] | None = None,
    repair: bool = True,
) -> list[DecodedSpan]:
    """Decode BIO labels into character spans using tokenizer offsets.

    Zero-width offsets (normally special or padding tokens) are treated as
    ``O`` before BIO repair.  A span score is the minimum of its token scores,
    a conservative confidence aggregation suitable for PII boundaries.
    """

    if len(tags) != len(offsets):
        raise ValueError(f"tags and offsets differ in length: {len(tags)} != {len(offsets)}.")
    if token_scores is not None and len(token_scores) != len(tags):
        raise ValueError("token_scores must have the same length as tags.")

    normalized_tags: list[str] = []
    normalized_offsets: list[tuple[int, int]] = []
    for tag, offset in zip(tags, offsets, strict=True):
        if len(offset) != 2:
            raise ValueError(f"Each offset must be a (start, end) pair, got {offset!r}.")
        start, end = offset
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"Offsets must contain integers, got {offset!r}.")
        if start < 0 or end < start:
            raise ValueError(f"Invalid left-closed/right-open offset {offset!r}.")
        if text is not None and end > len(text):
            raise ValueError(f"Offset {offset!r} exceeds text length {len(text)}.")
        normalized_offsets.append((start, end))
        normalized_tags.append("O" if start == end else tag)

    decoded_tags = repair_bio_tags(normalized_tags) if repair else normalized_tags
    if not repair:
        for tag in decoded_tags:
            _parse_bio_tag(tag)

    spans: list[DecodedSpan] = []
    active_type: str | None = None
    active_start = 0
    active_end = 0
    active_scores: list[float] = []

    def close_active() -> None:
        nonlocal active_type, active_scores
        if active_type is None:
            return
        score = min(active_scores) if active_scores else None
        spans.append(
            DecodedSpan(
                entity_type=active_type,
                start=active_start,
                end=active_end,
                score=score,
                text=text[active_start:active_end] if text is not None else None,
            )
        )
        active_type = None
        active_scores = []

    for index, (tag, (start, end)) in enumerate(zip(decoded_tags, normalized_offsets, strict=True)):
        prefix, entity_type = _parse_bio_tag(tag)
        if prefix == "O":
            close_active()
            continue
        if prefix == "B":
            close_active()
            active_type = entity_type
            active_start = start
            active_end = end
            active_scores = []
        else:
            # With repair=True this is guaranteed to extend the active type.
            if active_type != entity_type:
                raise ValueError(
                    f"Illegal BIO transition at token {index}: {tag!r}; set repair=True "
                    "to convert it into a B tag."
                )
            active_end = max(active_end, end)
        if token_scores is not None:
            active_scores.append(float(token_scores[index]))
    close_active()
    return spans


def decode_bio_ids(
    label_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    id2label: Mapping[int, str],
    *,
    text: str | None = None,
    token_scores: Sequence[float] | None = None,
    repair: bool = True,
) -> list[DecodedSpan]:
    """Map integer predictions through ``id2label`` and decode their spans."""

    try:
        tags = [id2label[int(label_id)] for label_id in label_ids]
    except KeyError as exc:
        raise ValueError(f"Label id {exc.args[0]!r} is absent from id2label.") from exc
    return decode_bio_spans(
        tags,
        offsets,
        text=text,
        token_scores=token_scores,
        repair=repair,
    )
