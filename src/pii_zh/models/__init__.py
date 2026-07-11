"""Model building blocks for Chinese PII token classification.

The Qwen dependency is imported lazily so that lightweight utilities such as
BIO decoding remain usable when the optional Transformers dependency is not
installed.
"""

from __future__ import annotations

from typing import Any

from .decoding import DecodedSpan, decode_bio_ids, decode_bio_spans, repair_bio_tags
from .qwen3_jpt import apply_second_copy_label_mask, build_jpt_inputs

__all__ = [
    "DecodedSpan",
    "Qwen3BiConfig",
    "Qwen3BiForTokenClassification",
    "apply_second_copy_label_mask",
    "build_full_attention_mask",
    "build_jpt_inputs",
    "convert_qwen3_token_classifier_state_dict",
    "decode_bio_ids",
    "decode_bio_spans",
    "repair_bio_tags",
]

_QWEN_EXPORTS = {
    "Qwen3BiConfig",
    "Qwen3BiForTokenClassification",
    "build_full_attention_mask",
    "convert_qwen3_token_classifier_state_dict",
}


def __getattr__(name: str) -> Any:
    if name not in _QWEN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from . import qwen3_bi
    except ImportError as exc:  # pragma: no cover - exercised without optional deps
        raise ImportError(
            "Qwen3 bidirectional models require PyTorch and a Transformers "
            "release that includes Qwen3 support (tested with >=5.13,<6)."
        ) from exc
    return getattr(qwen3_bi, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _QWEN_EXPORTS)
