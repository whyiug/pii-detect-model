"""Deterministic multi-recognizer fusion and offset-safe replacement."""

from .deterministic import (
    DEFAULT_SOURCE_PRIORITY,
    DEFAULT_SPECIFICITY,
    DeterministicFusion,
    FusedDetection,
    as_detection,
)
from .predictions import fuse_prediction_records, system_fusion_configuration
from .refinement import suppress_invalid_structured_spans
from .replacement import (
    ReplacementSpan,
    apply_replacements,
    merge_replacement_coverage,
    replace_spans,
)

__all__ = [
    "DEFAULT_SOURCE_PRIORITY",
    "DEFAULT_SPECIFICITY",
    "DeterministicFusion",
    "FusedDetection",
    "ReplacementSpan",
    "apply_replacements",
    "as_detection",
    "fuse_prediction_records",
    "merge_replacement_coverage",
    "replace_spans",
    "suppress_invalid_structured_spans",
    "system_fusion_configuration",
]
