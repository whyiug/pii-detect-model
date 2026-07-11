"""Boundary-safe tokenizer configuration for exact character-span PII tagging.

Qwen's default byte-level BPE pre-tokenizer may merge an entity with adjacent
literal text. A BIO classifier cannot recover a boundary inside such a token.
Training and release tokenizers therefore isolate every Unicode code point
before applying the unchanged Qwen byte-level BPE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BOUNDARY_MODE_CONFIG_KEY = "pii_zh_boundary_mode"
BOUNDARY_MODE = "unicode_codepoint_v1"
_AUDIT_TEXT = "甲乙AB 12\n🙂é"
_BOUNDARY_PRETOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {"Regex": r"[\s\S]"},
            "behavior": "Isolated",
            "invert": False,
        },
        {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
    ],
}


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
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
    except (AttributeError, TypeError, ValueError) as exc:
        raise TokenizerBoundaryError("tokenizer backend graph is unavailable") from exc
    if backend.get("pre_tokenizer") != _BOUNDARY_PRETOKENIZER:
        raise TokenizerBoundaryError(
            "tokenizer pre-tokenizer graph violates the Unicode code-point boundary contract"
        )
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


def load_serialized_fast_tokenizer(
    checkpoint: str | Path,
    *,
    require_boundary: bool = False,
) -> Any:
    """Load a local ``tokenizer.json`` without model-specific reconstruction.

    Transformers 5's unified model tokenizers may rebuild parts of a backend
    graph from vocabulary files.  The serialized pre-tokenizer is a trained
    and attested character-boundary contract, so this project deliberately
    uses the generic fast-tokenizer loader to preserve the complete graph.
    """

    root = Path(checkpoint).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise TokenizerBoundaryError("tokenizer checkpoint must be a local directory")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise TokenizerBoundaryError(f"tokenizer checkpoint is missing {name}")
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise TokenizerBoundaryError("Transformers fast-tokenizer support is required") from exc

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        root,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise TokenizerBoundaryError("serialized tokenizer did not load as a fast tokenizer")
    if require_boundary:
        assert_character_boundary_tokenizer(tokenizer)
    return tokenizer


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
    "load_serialized_fast_tokenizer",
]
