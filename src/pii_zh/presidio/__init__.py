"""Optional Presidio adapters for Chinese Qwen PII detection."""

from ._compat import PRESIDIO_AVAILABLE, RecognizerResult
from .cascade_recognizer import CascadeRecognizer
from .cluener_prepare import (
    CLUENER_PRIMARY_PREPARATION_SCHEMA_VERSION,
    CluenerPrimaryPreparationError,
    prepare_cluener_primary_artifact,
)
from .cluener_primary import (
    CLUENER_PRIMARY_CLOSED8_ENTITIES,
    CLUENER_PRIMARY_MODEL_ID,
    CLUENER_PRIMARY_MODEL_REVISION,
    CLUENER_PRIMARY_PRESIDIO_VERSION,
    CLUENER_PRIMARY_PROFILE_VERSION,
    CluenerPrimaryContractError,
    CluenerPrimaryPipeline,
    VerifiedCluenerPrimaryArtifact,
    build_cluener_primary_native_framework,
    build_cluener_primary_pipeline,
    verify_cluener_primary_artifact,
)
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
    "CLUENER_PRIMARY_CLOSED8_ENTITIES",
    "CLUENER_PRIMARY_MODEL_ID",
    "CLUENER_PRIMARY_MODEL_REVISION",
    "CLUENER_PRIMARY_PRESIDIO_VERSION",
    "CLUENER_PRIMARY_PROFILE_VERSION",
    "CLUENER_PRIMARY_PREPARATION_SCHEMA_VERSION",
    "CluenerPrimaryContractError",
    "CluenerPrimaryPreparationError",
    "CluenerPrimaryPipeline",
    "CnCommonRecognizer",
    "LocalModelBundle",
    "PRESIDIO_AVAILABLE",
    "QwenPiiRecognizer",
    "RecognizerDeduplicationPolicy",
    "RecognizerResult",
    "TokenChunker",
    "TokenWindow",
    "TokenizerAwareChunker",
    "VerifiedCluenerPrimaryArtifact",
    "build_cluener_primary_native_framework",
    "build_cluener_primary_pipeline",
    "create_analyzer_engine",
    "prepare_cluener_primary_artifact",
    "verify_cluener_primary_artifact",
]
