"""Native Presidio framework comparator for the six-way ablation.

Unlike the production Presidio factory, this evaluation-only adapter registers
``CnCommonRecognizer`` and the bound BIO NER recognizer separately. Presidio's
native ``AnalyzerEngine`` therefore owns framework orchestration and its own
duplicate semantics. The resulting spans are then normalized by the shared
``CascadePipeline`` under an exact-only pass-through profile.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ._compat import PRESIDIO_AVAILABLE, LocalRecognizer
from .cn_common_recognizer import CnCommonRecognizer
from .engine import (
    _MISSING_PRESIDIO_MESSAGE,
    _NoOpChineseNlpEngine,
    _NoOpPresidioContextEnhancer,
    _presidio_api,
    _validate_chinese_recognizer,
)

NATIVE_FRAMEWORK_SCHEMA_VERSION = "pii-zh.native-presidio-framework.v1"


def presidio_analyzer_version() -> str:
    if not PRESIDIO_AVAILABLE:
        raise ImportError(_MISSING_PRESIDIO_MESSAGE)
    try:
        value = version("presidio-analyzer")
    except PackageNotFoundError as exc:  # pragma: no cover - broken optional install
        raise ImportError(_MISSING_PRESIDIO_MESSAGE) from exc
    if not value or any(character.isspace() for character in value) or len(value) > 64:
        raise RuntimeError("Presidio distribution version is not a safe report token")
    return value


class NativePresidioFrameworkRecognizer:
    """Expose a real two-recognizer AnalyzerEngine as one Pipeline branch."""

    def __init__(
        self,
        *,
        ner_recognizer: LocalRecognizer,
        rule_recognizer: CnCommonRecognizer | None = None,
    ) -> None:
        if not PRESIDIO_AVAILABLE:
            raise ImportError(_MISSING_PRESIDIO_MESSAGE)
        if not isinstance(ner_recognizer, LocalRecognizer):
            raise TypeError("NER comparator must be a native Presidio LocalRecognizer")
        rules = rule_recognizer or CnCommonRecognizer()
        if not isinstance(rules, LocalRecognizer):
            raise TypeError("cn_common comparator must be a native Presidio LocalRecognizer")
        _validate_chinese_recognizer(rules, argument="rule_recognizer")
        _validate_chinese_recognizer(ner_recognizer, argument="ner_recognizer")

        analyzer_type, registry_type, nlp_artifacts_type = _presidio_api()
        registry = registry_type(
            recognizers=[rules, ner_recognizer],
            supported_languages=["zh"],
        )
        self.engine = analyzer_type(
            registry=registry,
            nlp_engine=_NoOpChineseNlpEngine(nlp_artifacts_type),
            supported_languages=["zh"],
            default_score_threshold=0.0,
            context_aware_enhancer=_NoOpPresidioContextEnhancer(),
        )
        self.rule_recognizer = rules
        self.ner_recognizer = ner_recognizer
        self.presidio_version = presidio_analyzer_version()

    def analyze(self, text: str, entities: Sequence[str] | None) -> list[Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        requested = list(entities) if entities is not None else None
        return list(
            self.engine.analyze(
                text=text,
                language="zh",
                entities=requested,
                return_decision_process=False,
            )
        )

    def runtime_identity(self) -> dict[str, str]:
        return {
            "framework": "presidio-analyzer",
            "framework_version": self.presidio_version,
            "framework_factory_schema_version": NATIVE_FRAMEWORK_SCHEMA_VERSION,
            "framework_factory_file_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "registered_rule_recognizer": type(self.rule_recognizer).__name__,
            "registered_ner_recognizer": type(self.ner_recognizer).__name__,
        }


__all__ = [
    "NATIVE_FRAMEWORK_SCHEMA_VERSION",
    "NativePresidioFrameworkRecognizer",
    "presidio_analyzer_version",
]
