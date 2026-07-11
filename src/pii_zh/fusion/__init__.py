"""Deterministic multi-recognizer fusion and offset-safe replacement."""

from .deterministic import (
    DEFAULT_SPECIFICITY,
    DeterministicFusion,
    FusedDetection,
    as_detection,
)
from .refinement import suppress_invalid_structured_spans
from .replacement import ReplacementSpan, apply_replacements, replace_spans

__all__ = [
    "DEFAULT_SPECIFICITY",
    "DeterministicFusion",
    "FusedDetection",
    "ReplacementSpan",
    "apply_replacements",
    "as_detection",
    "replace_spans",
    "suppress_invalid_structured_spans",
]
