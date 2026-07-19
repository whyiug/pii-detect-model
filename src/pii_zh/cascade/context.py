"""Privacy-safe contract for the cascade context-scoring stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

ContextOutcome = Literal["no_match", "positive", "negative", "mixed"]
_CONTEXT_OUTCOMES = frozenset({"no_match", "positive", "negative", "mixed"})


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """Minimal candidate view exposed to a context policy.

    The candidate intentionally contains no matched value or recognizer
    metadata.  A context policy receives the request text only for the duration
    of the call and must not retain it.
    """

    entity_type: str
    start: int
    end: int
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, str) or not self.entity_type:
            raise ValueError("context candidate entity_type must be non-empty")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("context candidate start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("context candidate end must be an integer")
        if not 0 <= self.start < self.end:
            raise ValueError("context candidate must use a non-empty span")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("context candidate score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("context candidate score must be between zero and one")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class ContextDecision:
    """Score-only context result with a raw-text-free audit outcome."""

    score: float
    outcome: ContextOutcome
    positive_match_count: int = 0
    negative_match_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("context decision score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("context decision score must be between zero and one")
        object.__setattr__(self, "score", score)
        if self.outcome not in _CONTEXT_OUTCOMES:
            raise ValueError("context decision outcome is invalid")
        for name, value in (
            ("positive_match_count", self.positive_match_count),
            ("negative_match_count", self.negative_match_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        has_positive = self.positive_match_count > 0
        has_negative = self.negative_match_count > 0
        expected: ContextOutcome
        if has_positive and has_negative:
            expected = "mixed"
        elif has_positive:
            expected = "positive"
        elif has_negative:
            expected = "negative"
        else:
            expected = "no_match"
        if self.outcome != expected:
            raise ValueError("context outcome must agree with match counts")

    @property
    def decision_step(self) -> str:
        """Return the stable audit token; never include a matched phrase."""

        return f"context:{self.outcome}"


class ContextEnhancer(Protocol):
    """Score-only context policy consumed by :class:`CascadePipeline`."""

    def enhance_candidate(
        self,
        text: str,
        candidate: ContextCandidate,
    ) -> ContextDecision: ...


__all__ = [
    "ContextCandidate",
    "ContextDecision",
    "ContextEnhancer",
    "ContextOutcome",
]
