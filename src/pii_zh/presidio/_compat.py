"""Small Presidio compatibility boundary used when the extra is not installed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised only in environments with the optional extra
    from presidio_analyzer import LocalRecognizer, RecognizerResult

    PRESIDIO_AVAILABLE = True
except ImportError:  # pragma: no cover - the fallback itself is exercised instead
    PRESIDIO_AVAILABLE = False

    class LocalRecognizer:  # type: ignore[no-redef]
        """Import-safe subset of Presidio's LocalRecognizer contract."""

        def __init__(
            self,
            supported_entities: list[str],
            name: str | None = None,
            supported_language: str = "en",
            version: str = "0.0.1",
            context: list[str] | None = None,
        ) -> None:
            self.supported_entities = supported_entities
            self.name = name or type(self).__name__
            self.supported_language = supported_language
            self.version = version
            self.context = context or []
            self.id = f"{self.name}_{self.supported_language}_{self.version}"

        def get_supported_entities(self) -> list[str]:
            return list(self.supported_entities)

        def get_supported_language(self) -> str:
            return self.supported_language

        def load(self) -> None:
            return None

    @dataclass(slots=True)
    class RecognizerResult:  # type: ignore[no-redef]
        entity_type: str
        start: int
        end: int
        score: float
        analysis_explanation: Any | None = None
        recognition_metadata: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return {
                "entity_type": self.entity_type,
                "start": self.start,
                "end": self.end,
                "score": self.score,
                "analysis_explanation": self.analysis_explanation,
                "recognition_metadata": dict(self.recognition_metadata),
            }


def make_recognizer_result(
    *,
    entity_type: str,
    start: int,
    end: int,
    score: float,
    metadata: dict[str, Any],
) -> RecognizerResult:
    """Construct a result across supported Presidio 2.2 patch versions."""

    try:
        return RecognizerResult(
            entity_type=entity_type,
            start=start,
            end=end,
            score=score,
            recognition_metadata=metadata,
        )
    except TypeError:  # pragma: no cover - compatibility with older 2.2 patches
        result = RecognizerResult(
            entity_type=entity_type,
            start=start,
            end=end,
            score=score,
        )
        result.recognition_metadata = metadata
        return result


def result_metadata(result: Any) -> dict[str, Any]:
    value = getattr(result, "recognition_metadata", None)
    return dict(value) if isinstance(value, dict) else {}


__all__ = [
    "LocalRecognizer",
    "PRESIDIO_AVAILABLE",
    "RecognizerResult",
    "make_recognizer_result",
    "result_metadata",
]
