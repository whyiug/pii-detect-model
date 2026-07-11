"""Chinese phrase context scoring without spaCy token or lemma alignment."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._compat import RecognizerResult, result_metadata


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, _TrieNode] = field(default_factory=dict)
    phrase: str | None = None


class _PhraseTrie:
    def __init__(self, phrases: Sequence[str]) -> None:
        self.root = _TrieNode()
        for phrase in phrases:
            node = self.root
            for character in phrase:
                node = node.children.setdefault(character, _TrieNode())
            node.phrase = phrase

    def find(self, text: str) -> tuple[str, ...]:
        matches: set[str] = set()
        for start in range(len(text)):
            node = self.root
            for character in text[start:]:
                node = node.children.get(character)  # type: ignore[assignment]
                if node is None:
                    break
                if node.phrase is not None:
                    matches.add(node.phrase)
        return tuple(sorted(matches))


def _phrase_mapping(
    name: str, value: Mapping[str, Sequence[str]] | None
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for entity, phrases in (value or {}).items():
        if not isinstance(entity, str) or not entity:
            raise ValueError(f"{name} entity keys must be non-empty strings")
        unique: set[str] = set()
        for phrase in phrases:
            if not isinstance(phrase, str) or not 1 <= len(phrase) <= 20:
                raise ValueError(f"{name} phrases must contain 1-20 characters")
            unique.add(phrase)
        normalized[entity] = tuple(sorted(unique))
    return normalized


class ChinesePhraseContextEnhancer:
    """Adjust scores from entity-specific positive and negative Chinese phrases."""

    _FIELD_BOUNDARIES = frozenset("\r\n,，;；{}[]\t")

    def __init__(
        self,
        *,
        positive_context: Mapping[str, Sequence[str]] | None = None,
        negative_context: Mapping[str, Sequence[str]] | None = None,
        context_window: int = 20,
        positive_delta: float = 0.15,
        negative_delta: float = 0.20,
    ) -> None:
        if isinstance(context_window, bool) or not isinstance(context_window, int):
            raise TypeError("context_window must be an integer")
        if context_window < 1:
            raise ValueError("context_window must be positive")
        for name, delta in {
            "positive_delta": positive_delta,
            "negative_delta": negative_delta,
        }.items():
            if not math.isfinite(float(delta)) or not 0.0 <= float(delta) <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        self.positive_context = _phrase_mapping("positive_context", positive_context)
        self.negative_context = _phrase_mapping("negative_context", negative_context)
        self.context_window = context_window
        self.positive_delta = float(positive_delta)
        self.negative_delta = float(negative_delta)
        self._positive_tries = {
            entity: _PhraseTrie(phrases) for entity, phrases in self.positive_context.items()
        }
        self._negative_tries = {
            entity: _PhraseTrie(phrases) for entity, phrases in self.negative_context.items()
        }

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> ChinesePhraseContextEnhancer:
        positive = value.get("positive_context", value.get("positive", {}))
        negative = value.get("negative_context", value.get("negative", {}))
        if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
            raise TypeError("positive and negative context config must be mappings")
        return cls(
            positive_context=positive,
            negative_context=negative,
            context_window=int(value.get("context_window", 20)),
            positive_delta=float(value.get("positive_delta", 0.15)),
            negative_delta=float(value.get("negative_delta", 0.20)),
        )

    def _field_window(self, text: str, start: int, end: int) -> tuple[int, int]:
        left_limit = max(0, start - self.context_window)
        right_limit = min(len(text), end + self.context_window)
        left = left_limit
        for index in range(start - 1, left_limit - 1, -1):
            if text[index] in self._FIELD_BOUNDARIES:
                left = index + 1
                break
        right = right_limit
        for index in range(end, right_limit):
            if text[index] in self._FIELD_BOUNDARIES:
                right = index
                break
        return left, right

    @staticmethod
    def _table_header(text: str, start: int) -> str:
        """Return the matching header cell for a tab-separated data row."""

        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        if line_end < 0:
            line_end = len(text)
        current_line = text[line_start:line_end]
        if "\t" not in current_line:
            return ""
        column = text[line_start:start].count("\t")
        previous_end = line_start - 1
        while previous_end >= 0:
            previous_start = text.rfind("\n", 0, previous_end) + 1
            previous_line = text[previous_start:previous_end]
            if previous_line.strip():
                cells = previous_line.split("\t")
                return cells[column] if column < len(cells) else ""
            previous_end = previous_start - 1
        return ""

    def enhance(self, text: str, results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        enhanced: list[RecognizerResult] = []
        for result in results:
            start = int(result.start)
            end = int(result.end)
            if not 0 <= start < end <= len(text):
                raise ValueError("recognizer result lies outside text")
            clone = copy.copy(result)
            metadata = result_metadata(result)
            if metadata.get("validator_valid") is False:
                clone.recognition_metadata = metadata
                enhanced.append(clone)
                continue
            left, right = self._field_window(text, start, end)
            context = text[left:right] + self._table_header(text, start)
            entity_type = str(result.entity_type)
            positive = self._positive_tries.get(entity_type)
            negative = self._negative_tries.get(entity_type)
            positive_matches = positive.find(context) if positive is not None else ()
            negative_matches = negative.find(context) if negative is not None else ()
            delta = (self.positive_delta if positive_matches else 0.0) - (
                self.negative_delta if negative_matches else 0.0
            )
            original_score = float(result.score)
            updated_score = min(1.0, max(0.0, original_score + delta))
            clone.score = updated_score
            decision = {
                "stage": "chinese_phrase_context",
                "positive_matches": list(positive_matches),
                "negative_matches": list(negative_matches),
                "score_delta": delta,
                "score_before": original_score,
                "score_after": updated_score,
            }
            process = metadata.get("decision_process", [])
            if not isinstance(process, list):
                process = [process]
            metadata["decision_process"] = [*process, decision]
            metadata["context_enhancement"] = decision
            metadata["is_score_enhanced_by_context"] = delta != 0.0
            clone.recognition_metadata = metadata
            enhanced.append(clone)
        return enhanced

    def enhance_using_context(
        self,
        text: str,
        raw_results: Sequence[RecognizerResult] | None = None,
        nlp_artifacts: Any | None = None,
        recognizers: Any | None = None,
        context: Any | None = None,
        **kwargs: Any,
    ) -> list[RecognizerResult]:
        """Presidio ContextAwareEnhancer-compatible entry point."""

        del nlp_artifacts, recognizers, context
        if raw_results is None:
            raw_results = kwargs.pop("raw_recognizer_results", None)
        if kwargs:
            raise TypeError(f"unexpected context enhancer arguments: {sorted(kwargs)}")
        if raw_results is None:
            raise TypeError("raw_results is required")
        return self.enhance(text, raw_results)


__all__ = ["ChinesePhraseContextEnhancer"]
