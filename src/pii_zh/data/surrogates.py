"""Keyed surrogate identities and offset-safe forward text reconstruction.

This module intentionally does not generate PII-shaped surface values.  A
caller supplies an already policy-approved, non-contactable replacement.  The
module provides two narrower security boundaries:

* HMAC-SHA256/HKDF-style domain separation for deterministic, split-isolated
  surrogate seeds and opaque public relation identifiers; and
* one-pass, left-to-right reconstruction of text and authoritative character
  offsets.

Raw relation material is accepted only as bytes and is never returned, hashed
with bare SHA-256, logged, or included in an exception.  Object ``repr`` values
also hide keys, derived seeds, source text, and replacements.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_HASH_LENGTH = hashlib.sha256().digest_size
_MINIMUM_KEY_LENGTH = 32
_MAX_FIELD_BYTES = 4096
_MAX_RELATION_BYTES = 1024 * 1024
_RELATION_ID_BYTES = 18
_RELATION_ID_PATTERN = re.compile(r"rel_v2_[A-Za-z0-9_-]{24}")

_PROTOCOL = b"pii-zh/surrogate/v2"
_EXTRACT_SALT = _PROTOCOL + b"/hkdf-extract"
_RELATION_KEY_DOMAIN = _PROTOCOL + b"/relation-key"
_SPLIT_KEY_DOMAIN = _PROTOCOL + b"/split-key"
_LABEL_KEY_DOMAIN = _PROTOCOL + b"/label-key"
_RELATION_MAC_DOMAIN = _PROTOCOL + b"/relation-mac"
_SEED_DOMAIN = _PROTOCOL + b"/seed"
_RELATION_ID_KEY_DOMAIN = _PROTOCOL + b"/relation-id-key"
_RELATION_ID_DOMAIN = _PROTOCOL + b"/relation-id"


class SurrogateError(ValueError):
    """Raised when surrogate inputs violate a fail-closed contract."""


def _frame(*parts: bytes) -> bytes:
    """Length-frame fields so concatenated values cannot share an encoding."""

    framed = bytearray()
    for part in parts:
        framed.extend(len(part).to_bytes(8, "big"))
        framed.extend(part)
    return bytes(framed)


def _hkdf_extract(salt: bytes, input_key_material: bytes) -> bytes:
    return hmac.new(salt, input_key_material, hashlib.sha256).digest()


def _hkdf_expand(pseudorandom_key: bytes, info: bytes, length: int = _HASH_LENGTH) -> bytes:
    if length < 1 or length > 255 * _HASH_LENGTH:
        raise ValueError("HKDF output length is outside the supported range")
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            pseudorandom_key,
            previous + info + bytes((counter,)),
            hashlib.sha256,
        ).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def _public_field(name: str, value: object) -> bytes:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SurrogateError(f"{name} must be a canonical non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise SurrogateError(f"{name} is not valid Unicode") from None
    if len(encoded) > _MAX_FIELD_BYTES:
        raise SurrogateError(f"{name} exceeds the supported length")
    return encoded


def _relation_material(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("relation material must be bytes")
    if not value:
        raise SurrogateError("relation material must not be empty")
    if len(value) > _MAX_RELATION_BYTES:
        raise SurrogateError("relation material exceeds the supported length")
    return value


def _opaque_relation_id(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    return f"rel_v2_{encoded}"


def _valid_relation_id(value: object) -> bool:
    return isinstance(value, str) and _RELATION_ID_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class SurrogateDerivation:
    """A public opaque relation ID and a private deterministic generator seed."""

    relation_id: str
    seed: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not _valid_relation_id(self.relation_id):
            raise SurrogateError("derived relation identifier is invalid")
        if not isinstance(self.seed, bytes) or len(self.seed) != _HASH_LENGTH:
            raise SurrogateError("derived surrogate seed is invalid")


class SurrogateKeyDeriver:
    """Derive split-isolated surrogate material from caller-owned key bytes.

    The public relation identifier deliberately excludes ``label`` so related
    fields can be linked inside one split.  Both it and the seed include the
    split and scope.  The private seed additionally includes the label, so two
    labels cannot accidentally reuse a generator stream.
    """

    __slots__ = ("_root_key",)

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes):
            raise TypeError("surrogate key must be bytes")
        if len(key) < _MINIMUM_KEY_LENGTH:
            raise SurrogateError("surrogate key does not meet the minimum length")
        self._root_key = _hkdf_extract(_EXTRACT_SALT, key)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    def derive(
        self,
        *,
        split: str,
        label: str,
        scope: str,
        relation: bytes,
    ) -> SurrogateDerivation:
        """Return deterministic seed material without exposing ``relation``.

        ``relation`` is an internal canonical relation value.  It may be based
        on normalized source data in a controlled build environment, but the
        caller must not persist it beside the returned public identifier.
        """

        split_bytes = _public_field("split", split)
        label_bytes = _public_field("label", label)
        scope_bytes = _public_field("scope", scope)
        relation_bytes = _relation_material(relation)

        relation_key = _hkdf_expand(self._root_key, _frame(_RELATION_KEY_DOMAIN))
        relation_mac = hmac.new(
            relation_key,
            _frame(_RELATION_MAC_DOMAIN, scope_bytes, relation_bytes),
            hashlib.sha256,
        ).digest()

        split_key = _hkdf_expand(
            self._root_key,
            _frame(_SPLIT_KEY_DOMAIN, split_bytes),
        )
        label_key = _hkdf_expand(
            split_key,
            _frame(_LABEL_KEY_DOMAIN, label_bytes),
        )
        seed = hmac.new(
            label_key,
            _frame(_SEED_DOMAIN, scope_bytes, relation_mac),
            hashlib.sha256,
        ).digest()

        relation_id_key = _hkdf_expand(
            split_key,
            _frame(_RELATION_ID_KEY_DOMAIN),
        )
        relation_id_bytes = hmac.new(
            relation_id_key,
            _frame(_RELATION_ID_DOMAIN, scope_bytes, relation_mac),
            hashlib.sha256,
        ).digest()[:_RELATION_ID_BYTES]
        return SurrogateDerivation(
            relation_id=_opaque_relation_id(relation_id_bytes),
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class OriginalSpan:
    """An authoritative source span whose value is hidden from ``repr``."""

    start: int
    end: int
    label: str
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SurrogateEdit:
    """One source span, its approved replacement, and opaque relation ID."""

    span: OriginalSpan
    replacement: str = field(repr=False)
    relation_id: str


@dataclass(frozen=True, slots=True)
class SurrogateSpan:
    """A rebuilt character span over public surrogate text."""

    start: int
    end: int
    label: str
    text: str = field(repr=False)
    relation_id: str


@dataclass(frozen=True, slots=True)
class RebuiltSurrogateText:
    """Rebuilt text and spans; text is deliberately absent from ``repr``."""

    text: str = field(repr=False)
    spans: tuple[SurrogateSpan, ...]


def _validate_edit(edit: SurrogateEdit, *, index: int, document: str) -> None:
    if not isinstance(edit.span, OriginalSpan):
        raise TypeError(f"surrogate edit {index} must contain an OriginalSpan")
    span = edit.span
    if isinstance(span.start, bool) or not isinstance(span.start, int):
        raise SurrogateError(f"surrogate edit {index} has an invalid start offset")
    if isinstance(span.end, bool) or not isinstance(span.end, int):
        raise SurrogateError(f"surrogate edit {index} has an invalid end offset")
    if not 0 <= span.start < span.end <= len(document):
        raise SurrogateError(f"surrogate edit {index} lies outside the document")
    if not isinstance(span.label, str) or not span.label.strip():
        raise SurrogateError(f"surrogate edit {index} has an invalid label")
    if not isinstance(span.text, str) or not span.text.strip():
        raise SurrogateError(f"surrogate edit {index} has an empty source value")
    if document[span.start : span.end] != span.text:
        raise SurrogateError(f"surrogate edit {index} does not match the source span")
    if not isinstance(edit.replacement, str) or not edit.replacement.strip():
        raise SurrogateError(f"surrogate edit {index} has an empty replacement")
    if not _valid_relation_id(edit.relation_id):
        raise SurrogateError(f"surrogate edit {index} has an invalid opaque relation ID")


def rebuild_surrogate_text(
    text: str,
    edits: Iterable[SurrogateEdit],
) -> RebuiltSurrogateText:
    """Rebuild ``text`` and replacement spans in one left-to-right pass.

    Original offsets are left-closed/right-open Python character offsets.  The
    function validates every source slice before emitting output.  It never
    mutates an intermediate text or attempts length-delta offset repair.
    """

    if not isinstance(text, str):
        raise TypeError("source text must be a string")
    try:
        normalized = tuple(edits)
    except Exception:
        # Custom iterables can put source text in their exception message.
        # Suppress the cause so logging this failure cannot reflect it.
        raise TypeError("surrogate edits must be iterable") from None
    if any(not isinstance(edit, SurrogateEdit) for edit in normalized):
        raise TypeError("surrogate edits must contain only SurrogateEdit values")

    for index, edit in enumerate(normalized):
        _validate_edit(edit, index=index, document=text)

    # Validate offset types before sorting.  Otherwise a malformed object's
    # comparison exception could reflect caller-controlled content.
    ordered = sorted(normalized, key=lambda edit: (edit.span.start, edit.span.end))
    previous_end = 0
    for edit in ordered:
        if edit.span.start < previous_end:
            raise SurrogateError("surrogate edits must not overlap")
        previous_end = edit.span.end

    output_parts: list[str] = []
    output_spans: list[SurrogateSpan] = []
    source_cursor = 0
    output_length = 0
    for edit in ordered:
        prefix = text[source_cursor : edit.span.start]
        output_parts.append(prefix)
        output_length += len(prefix)

        start = output_length
        output_parts.append(edit.replacement)
        output_length += len(edit.replacement)
        end = output_length
        output_spans.append(
            SurrogateSpan(
                start=start,
                end=end,
                label=edit.span.label,
                text=edit.replacement,
                relation_id=edit.relation_id,
            )
        )
        source_cursor = edit.span.end

    output_parts.append(text[source_cursor:])
    rebuilt_text = "".join(output_parts)
    result = RebuiltSurrogateText(text=rebuilt_text, spans=tuple(output_spans))
    if any(result.text[span.start : span.end] != span.text for span in result.spans):
        raise RuntimeError("surrogate reconstruction invariant failed")
    return result


__all__ = [
    "OriginalSpan",
    "RebuiltSurrogateText",
    "SurrogateDerivation",
    "SurrogateEdit",
    "SurrogateError",
    "SurrogateKeyDeriver",
    "SurrogateSpan",
    "rebuild_surrogate_text",
]
