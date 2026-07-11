"""Optional Presidio adapters for Chinese Qwen PII detection."""

from ._compat import PRESIDIO_AVAILABLE, RecognizerResult
from .context_enhancer import ChinesePhraseContextEnhancer
from .qwen_recognizer import LocalModelBundle, QwenPiiRecognizer
from .token_chunker import TokenChunker, TokenizerAwareChunker, TokenWindow

__all__ = [
    "ChinesePhraseContextEnhancer",
    "LocalModelBundle",
    "PRESIDIO_AVAILABLE",
    "QwenPiiRecognizer",
    "RecognizerResult",
    "TokenChunker",
    "TokenWindow",
    "TokenizerAwareChunker",
]
