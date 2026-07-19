"""Presidio adapter for the validator-backed ``cn_common`` rule pack."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pii_zh.rules import CnCommonRulePack
from pii_zh.taxonomy import load_presidio_mapping

from ._compat import LocalRecognizer, RecognizerResult, make_recognizer_result


class CnCommonRecognizer(LocalRecognizer):
    """Expose :class:`CnCommonRulePack` through Presidio's recognizer contract.

    The underlying rule pack owns candidate validation and exact character
    offsets.  This adapter does not compile a second set of Presidio patterns,
    so direct rule use and ``AnalyzerEngine`` use share one implementation.
    """

    def __init__(
        self,
        *,
        rule_pack: CnCommonRulePack | None = None,
        name: str = "CnCommonRecognizer",
        supported_language: str = "zh",
        version: str = "1.0.0",
    ) -> None:
        self.rule_pack = rule_pack or CnCommonRulePack()
        public_entities = set(load_presidio_mapping().model_to_presidio.values())
        supported_entities = sorted(
            {rule.entity_type for rule in self.rule_pack.rules} & public_entities
        )
        self._public_entities = frozenset(supported_entities)
        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            version=version,
            context=[],
        )

    def load(self) -> None:
        """The local rule pack is ready at construction time."""

    def analyze(
        self,
        text: str,
        entities: Sequence[str] | None,
        nlp_artifacts: Any | None = None,
    ) -> list[RecognizerResult]:
        """Return validator-approved rule matches with Presidio offsets."""

        del nlp_artifacts
        requested = (
            sorted(self._public_entities)
            if entities is None
            else [entity for entity in entities if entity in self._public_entities]
        )
        matches = self.rule_pack.analyze(text, entities=requested)
        return [
            make_recognizer_result(
                entity_type=match.entity_type,
                start=match.start,
                end=match.end,
                score=match.score,
                metadata={
                    **match.metadata,
                    "pattern_id": match.pattern_id,
                    "source": match.source,
                },
            )
            for match in matches
        ]


__all__ = ["CnCommonRecognizer"]
