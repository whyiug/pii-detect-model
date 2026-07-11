"""Boundary-safe tokenizer configuration for exact character-span PII tagging.

Qwen's default byte-level BPE pre-tokenizer may merge an entity with adjacent
literal text. A BIO classifier cannot recover a boundary inside such a token.
Training and release tokenizers therefore isolate every Unicode code point
before applying the unchanged Qwen byte-level BPE.
"""

from __future__ import annotations

from typing import Any

BOUNDARY_MODE_CONFIG_KEY = "pii_zh_boundary_mode"
BOUNDARY_MODE = "unicode_codepoint_v1"
_AUDIT_TEXT = "甲A 1\n🙂é"


class TokenizerBoundaryError(RuntimeError):
    """Raised when a tokenizer cannot guarantee exact character boundaries."""


def _marker(tokenizer: Any) -> object:
    direct = getattr(tokenizer, BOUNDARY_MODE_CONFIG_KEY, None)
    if direct is not None:
        return direct
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    return init_kwargs.get(BOUNDARY_MODE_CONFIG_KEY) if isinstance(init_kwargs, dict) else None


def assert_character_boundary_tokenizer(
    tokenizer: Any,
    *,
    require_marker: bool = True,
) -> None:
    """Fail closed unless tokenizer offsets are code-point boundary safe."""

    if not getattr(tokenizer, "is_fast", False):
        raise TokenizerBoundaryError("exact character spans require a fast tokenizer")
    if require_marker and _marker(tokenizer) != BOUNDARY_MODE:
        raise TokenizerBoundaryError("tokenizer is missing the versioned PII boundary mode")
    try:
        encoded = tokenizer(
            _AUDIT_TEXT,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        raw_offsets = encoded["offset_mapping"]
    except Exception as exc:  # pragma: no cover - tokenizer implementation boundary
        raise TokenizerBoundaryError("tokenizer offset audit failed") from exc

    offsets = [tuple(map(int, pair)) for pair in raw_offsets]
    non_empty = [(start, end) for start, end in offsets if end > start]
    if not non_empty or any(end - start != 1 for start, end in non_empty):
        raise TokenizerBoundaryError("a tokenizer token crosses a Unicode code-point boundary")
    covered = {position for start, end in non_empty for position in range(start, end)}
    if covered != set(range(len(_AUDIT_TEXT))):
        raise TokenizerBoundaryError("tokenizer offsets do not cover every input code point")


def configure_character_boundary_tokenizer(tokenizer: Any) -> Any:
    """Mutate a Qwen-compatible fast tokenizer into the versioned safe view.

    Only the pre-tokenizer changes. Normalization, vocabulary, merge table,
    decoder, and special-token IDs remain those of the pinned checkpoint.
    ``save_pretrained`` persists the backend graph and boundary-mode marker.
    """

    if not getattr(tokenizer, "is_fast", False) or not hasattr(tokenizer, "backend_tokenizer"):
        raise TokenizerBoundaryError("boundary configuration requires a fast backend tokenizer")
    try:
        from tokenizers import Regex
        from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    except ImportError as exc:  # pragma: no cover - Transformers depends on tokenizers
        raise TokenizerBoundaryError("the tokenizers package is required") from exc

    tokenizer.backend_tokenizer.pre_tokenizer = Sequence(
        [
            Split(Regex(r"[\s\S]"), behavior="isolated"),
            ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        raise TokenizerBoundaryError("tokenizer init_kwargs are unavailable")
    init_kwargs[BOUNDARY_MODE_CONFIG_KEY] = BOUNDARY_MODE
    setattr(tokenizer, BOUNDARY_MODE_CONFIG_KEY, BOUNDARY_MODE)
    assert_character_boundary_tokenizer(tokenizer)
    return tokenizer


__all__ = [
    "BOUNDARY_MODE",
    "BOUNDARY_MODE_CONFIG_KEY",
    "TokenizerBoundaryError",
    "assert_character_boundary_tokenizer",
    "configure_character_boundary_tokenizer",
]
