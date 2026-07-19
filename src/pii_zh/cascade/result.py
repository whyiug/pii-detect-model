"""Stable, raw-text-free public result schema for cascade detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

CASCADE_DETECTION_SCHEMA_VERSION = "pii-zh.cascade.detection.v1"


@dataclass(frozen=True, slots=True, order=True)
class CascadeDetection:
    """A left-closed/right-open detection with a privacy-safe decision trace."""

    start: int
    end: int
    entity_type: str
    score: float
    source: str
    sources: tuple[str, ...]
    decision_process: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("end must be an integer")
        if not 0 <= self.start < self.end:
            raise ValueError("detection offsets must satisfy 0 <= start < end")
        if not isinstance(self.entity_type, str) or not self.entity_type:
            raise ValueError("entity_type must be a non-empty string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and between zero and one")
        object.__setattr__(self, "score", score)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("sources must be a non-empty tuple")
        if any(not isinstance(source, str) or not source for source in self.sources):
            raise ValueError("sources must contain non-empty strings")
        if tuple(sorted(set(self.sources))) != self.sources:
            raise ValueError("sources must be sorted and unique")
        if self.source not in self.sources:
            raise ValueError("winning source must be present in sources")
        if not isinstance(self.decision_process, tuple) or not self.decision_process:
            raise ValueError("decision_process must be a non-empty tuple")
        if any(not isinstance(step, str) or not step for step in self.decision_process):
            raise ValueError("decision_process must contain non-empty strings")

    def to_dict(self) -> dict[str, object]:
        """Serialize only offsets, labels, scores, sources, and decisions."""

        return {
            "schema_version": CASCADE_DETECTION_SCHEMA_VERSION,
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "score": self.score,
            "source": self.source,
            "sources": list(self.sources),
            "decision_process": list(self.decision_process),
        }


__all__ = ["CASCADE_DETECTION_SCHEMA_VERSION", "CascadeDetection"]
