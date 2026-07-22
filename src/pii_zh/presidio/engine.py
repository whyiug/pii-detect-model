"""Offline-first construction of a Chinese Presidio ``AnalyzerEngine``."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, cast

from pii_zh.cascade import CascadeConfig, CascadePipeline

from ._compat import PRESIDIO_AVAILABLE, LocalRecognizer
from .cascade_recognizer import CascadeRecognizer, DetectionPipeline
from .cn_common_recognizer import CnCommonRecognizer
from .context_enhancer import ChinesePhraseContextEnhancer

if TYPE_CHECKING:  # pragma: no cover - imported for static analysis only
    from presidio_analyzer import AnalyzerEngine


_MISSING_PRESIDIO_MESSAGE = (
    "The Presidio integration requires the optional 'presidio' dependencies. "
    "Install them with: pip install 'pii-zh-qwen[presidio]'"
)


def _presidio_api() -> tuple[type[Any], type[Any], type[Any]]:
    if not PRESIDIO_AVAILABLE:
        raise ImportError(_MISSING_PRESIDIO_MESSAGE)
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpArtifacts
    except ImportError as exc:  # pragma: no cover - broken optional installation
        raise ImportError(_MISSING_PRESIDIO_MESSAGE) from exc
    return AnalyzerEngine, RecognizerRegistry, NlpArtifacts


class _NoOpChineseNlpEngine:
    """Minimal Presidio NLP contract for offset-only local recognizers.

    ``cn_common`` and the project's model recognizers operate directly on raw
    text, so loading a spaCy pipeline would add an implicit model dependency
    without contributing to their output.  Empty artifacts are sufficient for
    Presidio's recognizer orchestration and the raw-text context enhancer.
    """

    def __init__(self, nlp_artifacts_type: type[Any]) -> None:
        self._nlp_artifacts_type = nlp_artifacts_type
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def process_text(self, text: str, language: str) -> Any:
        if language != "zh":
            raise ValueError("the Chinese Presidio engine only supports language='zh'")
        return self._nlp_artifacts_type(
            entities=[],
            tokens=[],
            tokens_indices=[],
            lemmas=[],
            nlp_engine=self,
            language=language,
        )

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs: Any,
    ) -> Iterator[tuple[str, Any]]:
        del batch_size, n_process, kwargs
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        del word
        if language != "zh":
            raise ValueError("the Chinese Presidio engine only supports language='zh'")
        return False

    def is_punct(self, word: str, language: str) -> bool:
        del word
        if language != "zh":
            raise ValueError("the Chinese Presidio engine only supports language='zh'")
        return False

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return ["zh"]


class _NoOpPresidioContextEnhancer:
    """Prevent Presidio from applying a second score-changing stage."""

    def enhance_using_context(
        self,
        text: str,
        raw_results: list[Any],
        nlp_artifacts: Any,
        recognizers: list[Any],
        context: list[str] | None = None,
    ) -> list[Any]:
        del text, nlp_artifacts, recognizers, context
        return list(raw_results)


def _validate_chinese_recognizer(recognizer: LocalRecognizer, *, argument: str) -> None:
    language = getattr(recognizer, "supported_language", None)
    if language != "zh":
        raise ValueError(f"{argument} must support language='zh'; received {language!r}")


def create_analyzer_engine(
    *,
    model_recognizer: LocalRecognizer | None = None,
    rule_recognizer: CnCommonRecognizer | None = None,
    cascade_pipeline: DetectionPipeline | None = None,
    context_enhancer: ChinesePhraseContextEnhancer | None = None,
    default_score_threshold: float = 0.0,
) -> AnalyzerEngine:
    """Create an offline Chinese Presidio analyzer.

    By default the factory builds one :class:`CascadePipeline` from the
    validator-backed ``cn_common`` recognizer and an optional already-created
    model recognizer, then registers that pipeline as a single Presidio
    recognizer.  This prevents Presidio and the Python/CLI path from applying
    different overlap or validator policies.  It neither downloads models nor
    initializes a spaCy pipeline.
    """

    analyzer_type, registry_type, nlp_artifacts_type = _presidio_api()
    if cascade_pipeline is not None and (
        model_recognizer is not None or rule_recognizer is not None
    ):
        raise ValueError("cascade_pipeline cannot be combined with separate model/rule recognizers")
    if cascade_pipeline is None:
        rules = rule_recognizer or CnCommonRecognizer()
        _validate_chinese_recognizer(rules, argument="rule_recognizer")
        if model_recognizer is not None:
            _validate_chinese_recognizer(model_recognizer, argument="model_recognizer")
        config = (
            CascadeConfig(mode="cascade")
            if model_recognizer is not None
            else CascadeConfig(mode="rules-only")
        )
        cascade_pipeline = CascadePipeline(
            config=config,
            rule_recognizer=rules,
            model_recognizer=cast(Any, model_recognizer),
            context_enhancer=context_enhancer,
        )
    elif context_enhancer is not None:
        # Preserve the historical factory argument while moving its semantics
        # into the shared runtime instead of applying it after fusion.
        replace_context = getattr(cascade_pipeline, "with_context_enhancer", None)
        if not callable(replace_context):
            raise ValueError(
                "context_enhancer is unsupported by the injected composite pipeline"
            )
        cascade_pipeline = cast(DetectionPipeline, replace_context(context_enhancer))

    recognizers: list[LocalRecognizer] = [CascadeRecognizer(cascade_pipeline)]

    registry = registry_type(recognizers=recognizers, supported_languages=["zh"])
    nlp_engine = _NoOpChineseNlpEngine(nlp_artifacts_type)
    analyzer = analyzer_type(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["zh"],
        default_score_threshold=default_score_threshold,
        context_aware_enhancer=_NoOpPresidioContextEnhancer(),
    )
    return cast("AnalyzerEngine", analyzer)


__all__ = ["create_analyzer_engine"]
