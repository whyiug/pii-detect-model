"""Local, character-span inference adapters."""

from .aiguard import (
    AIGUARD_CLOSED_8_LABEL_MAPPING,
    CLOSED_8_TARGET_LABELS,
    AiguardClosed8SpanPredictor,
    AiguardProtocolError,
    decode_bie_tags,
    load_aiguard_closed8_predictor,
)
from .transformers_predictor import (
    InferenceSafetyError,
    TransformersSpanPredictor,
    load_local_predictor,
)

__all__ = [
    "AIGUARD_CLOSED_8_LABEL_MAPPING",
    "CLOSED_8_TARGET_LABELS",
    "AiguardClosed8SpanPredictor",
    "AiguardProtocolError",
    "InferenceSafetyError",
    "TransformersSpanPredictor",
    "decode_bie_tags",
    "load_aiguard_closed8_predictor",
    "load_local_predictor",
]
