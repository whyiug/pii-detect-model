"""Optional Presidio adapters for Chinese Qwen PII detection."""

from ._compat import PRESIDIO_AVAILABLE, RecognizerResult
from .cascade_recognizer import CascadeRecognizer
from .cn_common_recognizer import CnCommonRecognizer
from .context_enhancer import ChinesePhraseContextEnhancer
from .engine import create_analyzer_engine
from .qwen_recognizer import (
    LocalModelBundle,
    QwenPiiRecognizer,
    RecognizerDeduplicationPolicy,
)
from .token_chunker import TokenChunker, TokenizerAwareChunker, TokenWindow

__all__ = [
    "CascadeRecognizer",
    "ChinesePhraseContextEnhancer",
    "CnCommonRecognizer",
    "LocalModelBundle",
    "PRESIDIO_AVAILABLE",
    "QwenPiiRecognizer",
    "RecognizerDeduplicationPolicy",
    "RecognizerResult",
    "TokenChunker",
    "TokenWindow",
    "TokenizerAwareChunker",
    "create_analyzer_engine",
]
