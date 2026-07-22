"""Expose the release cascade as one Presidio recognizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pii_zh.cascade import CascadeConfig, CascadeDetection

from ._compat import LocalRecognizer, RecognizerResult, make_recognizer_result


class DetectionPipeline(Protocol):
    """Structural contract shared by cascade and primary/augmentation services."""

    config: CascadeConfig

    def detect(
        self,
        text: str,
        *,
        entities: Sequence[str] | None = None,
    ) -> list[CascadeDetection]: ...


class CascadeRecognizer(LocalRecognizer):
    """Route Presidio through the same validator/fusion runtime as Python/CLI."""

    def __init__(
        self,
        pipeline: DetectionPipeline,
        *,
        name: str = "PiiZhCascadeRecognizer",
        supported_language: str = "zh",
        version: str = "1.0.0",
    ) -> None:
        if not isinstance(getattr(pipeline, "config", None), CascadeConfig) or not callable(
            getattr(pipeline, "detect", None)
        ):
            raise TypeError("pipeline must expose CascadeConfig and detect")
        self.pipeline = pipeline
        super().__init__(
            supported_entities=sorted(pipeline.config.route_map),
            name=name,
            supported_language=supported_language,
            version=version,
            context=[],
        )

    def load(self) -> None:
        """The injected pipeline owns all already-created local resources."""

    def analyze(
        self,
        text: str,
        entities: Sequence[str] | None,
        nlp_artifacts: Any | None = None,
    ) -> list[RecognizerResult]:
        del nlp_artifacts
        return [
            make_recognizer_result(
                entity_type=item.entity_type,
                start=item.start,
                end=item.end,
                score=item.score,
                metadata={
                    "source": item.source,
                    "sources": list(item.sources),
                    "decision_process": list(item.decision_process),
                    "cascade_profile_version": self.pipeline.config.profile_version,
                    "cascade_mode": self.pipeline.config.mode,
                },
            )
            for item in self.pipeline.detect(text, entities=entities)
        ]


__all__ = ["CascadeRecognizer", "DetectionPipeline"]
