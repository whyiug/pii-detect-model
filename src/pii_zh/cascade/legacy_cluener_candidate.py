"""Shared ERNIE auxiliary used by PII-core and legacy CLUENER profiles.

This module deliberately keeps model loading outside evaluation scripts.  The
same checkpoint, ten-source-label logit mask, threshold, windowing, and
deduplication policy are used for both profiles.  Only the final namespace for
``address`` and ``name`` changes: the production-facing profile exposes them
as ``CN_ADDRESS``/``PERSON`` while the compatibility profile exposes all ten
labels under the isolated ``LEGACY_CLUENER_*`` namespace.

The remaining compatibility labels are also decoded in the PII-core variant,
but the ordinary cascade route table never requests or emits them.  This is
important: it prevents a smaller production logit mask from changing the
address/name probability distribution used by the compatibility run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from .legacy_cluener import (
    LEGACY_CLUENER_SOURCE_LABELS,
    LEGACY_CLUENER_SOURCE_TO_RUNTIME,
    build_legacy_cluener_pipeline,
)

ERNIE_AUXILIARY_MODEL_ID = "gyr66/Ernie-3.0-base-chinese-finetuned-ner"
ERNIE_AUXILIARY_REVISION = "3f3c3fa40cf87298586dbbed55fd7ca1988719a8"
ERNIE_AUXILIARY_PROTOCOL_ID = "pii_zh_ernie_legacy_cluener_10_v1"
ERNIE_AUXILIARY_MAX_TOKENS = 512
ERNIE_AUXILIARY_STRIDE_FRACTION = 0.25
ERNIE_AUXILIARY_DEFAULT_THRESHOLD = 0.0
ERNIE_AUXILIARY_DEDUPLICATION_POLICY = "exact_only_v1"
ERNIE_AUXILIARY_MICRO_BATCH_SIZE = 16

ErnieAuxiliaryProfile = Literal["pii_core", "legacy_cluener_10"]


def ernie_auxiliary_source_mapping(
    profile: ErnieAuxiliaryProfile,
) -> dict[str, str]:
    """Return a full-ten mapping while isolating non-PII output by namespace."""

    if profile not in {"pii_core", "legacy_cluener_10"}:
        raise ValueError("unknown ERNIE auxiliary profile")
    mapping = dict(LEGACY_CLUENER_SOURCE_TO_RUNTIME)
    if profile == "pii_core":
        mapping["address"] = "CN_ADDRESS"
        mapping["name"] = "PERSON"
    return mapping


def _validate_mapping_pair() -> None:
    production = ernie_auxiliary_source_mapping("pii_core")
    compatibility = ernie_auxiliary_source_mapping("legacy_cluener_10")
    expected = set(LEGACY_CLUENER_SOURCE_LABELS)
    if set(production) != expected or set(compatibility) != expected:
        raise RuntimeError("ERNIE auxiliary profiles must use the same ten-source-label mask")
    if any(
        production[label] != compatibility[label]
        for label in expected - {"address", "name"}
    ):
        raise RuntimeError("ERNIE auxiliary profiles differ beyond the two PII aliases")


def build_ernie_auxiliary_native_framework(
    model_path: str | Path,
    *,
    profile: ErnieAuxiliaryProfile,
    device: str = "cpu",
    micro_batch_size: int = ERNIE_AUXILIARY_MICRO_BATCH_SIZE,
) -> object:
    """Load the pinned-safe BIO path and return a native Presidio framework.

    Exact file/revision verification is intentionally the responsibility of a
    preregistration or release receipt before this loader is called.  The
    underlying Transformers loader remains local-only, safetensors-only, and
    ``trust_remote_code=False``.
    """

    _validate_mapping_pair()
    if micro_batch_size != ERNIE_AUXILIARY_MICRO_BATCH_SIZE:
        raise ValueError(
            "ERNIE auxiliary micro_batch_size must match the frozen value 16"
        )
    mapping: Mapping[str, str] = ernie_auxiliary_source_mapping(profile)

    # Keep torch/transformers imports behind the explicit loader call.
    from pii_zh.inference.bio_token_classifier import (
        load_closed_bio_token_classifier,
    )
    from pii_zh.presidio.bio_token_recognizer import BioTokenClassifierRecognizer
    from pii_zh.presidio.cn_common_recognizer import CnCommonRecognizer
    from pii_zh.presidio.native_framework import NativePresidioFrameworkRecognizer

    predictor = load_closed_bio_token_classifier(
        model_path,
        source_to_target=mapping,
        protocol_id=ERNIE_AUXILIARY_PROTOCOL_ID,
        device=device,
        dtype=None,
        micro_batch_size=micro_batch_size,
    )
    recognizer = BioTokenClassifierRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        supported_entities=sorted(set(mapping.values())),
        default_threshold=ERNIE_AUXILIARY_DEFAULT_THRESHOLD,
        max_tokens=ERNIE_AUXILIARY_MAX_TOKENS,
        stride_fraction=ERNIE_AUXILIARY_STRIDE_FRACTION,
        model_version=f"{ERNIE_AUXILIARY_MODEL_ID}@{ERNIE_AUXILIARY_REVISION}",
        source_name="native_presidio:ernie_auxiliary",
        deduplication_policy=ERNIE_AUXILIARY_DEDUPLICATION_POLICY,
        name="ErnieAuxiliaryNativeFrameworkNerRecognizer",
    )
    return NativePresidioFrameworkRecognizer(
        ner_recognizer=recognizer,
        rule_recognizer=CnCommonRecognizer(),
    )


def build_ernie_auxiliary_legacy_pipeline(
    model_path: str | Path,
    *,
    device: str = "cpu",
    micro_batch_size: int = ERNIE_AUXILIARY_MICRO_BATCH_SIZE,
) -> object:
    """Build the explicitly authorized compatibility pipeline for G2."""

    framework = build_ernie_auxiliary_native_framework(
        model_path,
        profile="legacy_cluener_10",
        device=device,
        micro_batch_size=micro_batch_size,
    )
    return build_legacy_cluener_pipeline(
        native_presidio_recognizer=framework,
        allow_evaluation_profile=True,
    )


__all__ = [
    "ERNIE_AUXILIARY_DEDUPLICATION_POLICY",
    "ERNIE_AUXILIARY_DEFAULT_THRESHOLD",
    "ERNIE_AUXILIARY_MAX_TOKENS",
    "ERNIE_AUXILIARY_MICRO_BATCH_SIZE",
    "ERNIE_AUXILIARY_MODEL_ID",
    "ERNIE_AUXILIARY_PROTOCOL_ID",
    "ERNIE_AUXILIARY_REVISION",
    "ERNIE_AUXILIARY_STRIDE_FRACTION",
    "ErnieAuxiliaryProfile",
    "build_ernie_auxiliary_legacy_pipeline",
    "build_ernie_auxiliary_native_framework",
    "ernie_auxiliary_source_mapping",
]
