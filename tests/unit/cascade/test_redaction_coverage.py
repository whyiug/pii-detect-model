from __future__ import annotations

from collections.abc import Sequence

import pytest

from pii_zh.cascade import CascadeConfig, CascadePipeline


def _candidate(
    entity_type: str,
    start: int,
    end: int,
    *,
    score: float = 0.99,
) -> dict[str, object]:
    return {
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "score": score,
    }


class _StaticRuleRecognizer:
    def __init__(self, candidates: Sequence[dict[str, object]]) -> None:
        self._candidates = tuple(candidates)
        self.calls = 0

    def analyze(
        self,
        text: str,
        entities: Sequence[str],
    ) -> list[dict[str, object]]:
        del text
        self.calls += 1
        allowed = frozenset(entities)
        return [
            dict(candidate) for candidate in self._candidates if candidate["entity_type"] in allowed
        ]


def _pipeline(*candidates: dict[str, object]) -> CascadePipeline:
    return CascadePipeline(
        config=CascadeConfig(mode="rules-only"),
        rule_recognizer=_StaticRuleRecognizer(candidates),
    )


def _detection_core(pipeline: CascadePipeline, text: str) -> list[tuple[str, int, int]]:
    return [(item.entity_type, item.start, item.end) for item in pipeline.detect(text)]


def test_nested_different_labels_keep_specific_extraction_but_redact_outer_union() -> None:
    text = "甲乙丙丁戊己庚"
    pipeline = _pipeline(
        _candidate("PERSON", 1, 6, score=1.0),
        _candidate("CN_ADDRESS", 2, 5, score=0.80),
    )

    assert _detection_core(pipeline, text) == [("CN_ADDRESS", 2, 5)]
    assert pipeline.redact(text, "<隐藏>") == "甲<隐藏>庚"
    assert _detection_core(pipeline, text) == [("CN_ADDRESS", 2, 5)]


def test_partially_overlapping_same_label_uses_union_and_typed_replacement() -> None:
    text = "甲乙丙丁戊己庚"
    pipeline = _pipeline(
        _candidate("PERSON", 1, 4, score=0.99),
        _candidate("PERSON", 3, 6, score=0.90),
    )

    assert _detection_core(pipeline, text) == [("PERSON", 1, 4)]
    assert pipeline.redact(text, {"PERSON": "<姓名>"}) == "甲<姓名>庚"


def test_exact_boundary_different_labels_use_mixed_label_replacement() -> None:
    text = "甲乙丙丁戊己"
    pipeline = _pipeline(
        _candidate("PERSON", 1, 5, score=1.0),
        _candidate("CN_ADDRESS", 1, 5, score=0.80),
    )

    assert _detection_core(pipeline, text) == [("CN_ADDRESS", 1, 5)]
    assert pipeline.redact(text, {"PII": "<异类重叠>"}) == "甲<异类重叠>己"


def test_disjoint_spans_remain_separate_and_retain_their_entity_types() -> None:
    text = "甲乙，丙丁戊，己庚"
    pipeline = _pipeline(
        _candidate("PERSON", 0, 2),
        _candidate("CN_ADDRESS", 3, 6),
    )

    assert _detection_core(pipeline, text) == [
        ("PERSON", 0, 2),
        ("CN_ADDRESS", 3, 6),
    ]
    assert (
        pipeline.redact(
            text,
            {"PERSON": "<姓名>", "CN_ADDRESS": "<地址>"},
        )
        == "<姓名>，<地址>，己庚"
    )


def test_unicode_offsets_are_python_character_offsets_across_coverage_union() -> None:
    text = "前🙂甲乙丙丁后"
    union_start = text.index("🙂")
    union_end = text.index("后")
    pipeline = _pipeline(
        _candidate("PERSON", union_start, union_start + 3, score=0.99),
        _candidate("CN_ADDRESS", union_start + 2, union_end, score=0.80),
    )

    assert text[union_start:union_end] == "🙂甲乙丙丁"
    assert _detection_core(pipeline, text) == [("CN_ADDRESS", union_start + 2, union_end)]
    assert pipeline.redact(text, "<PII>") == "前<PII>后"


def test_standard_redaction_runs_the_recognizer_once() -> None:
    text = "前甲乙丙丁戊己后"
    recognizer = _StaticRuleRecognizer(
        (
            _candidate("PERSON", 1, 5, score=0.99),
            _candidate("CN_ADDRESS", 3, 7, score=0.80),
        )
    )
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="rules-only"),
        rule_recognizer=recognizer,
    )

    assert pipeline.redact(text) == "前<PII>后"
    assert recognizer.calls == 1


def test_mixed_label_mapping_requires_pii_or_default_and_fails_without_raw_value() -> None:
    text = "前甲乙丙丁戊己后"
    pipeline = _pipeline(
        _candidate("PERSON", 1, 5, score=0.99),
        _candidate("CN_ADDRESS", 3, 7, score=0.80),
    )

    with pytest.raises(KeyError, match="PII") as error:
        pipeline.redact(
            text,
            {"PERSON": "<姓名>", "CN_ADDRESS": "<地址>"},
        )
    assert text not in str(error.value)

    assert pipeline.redact(text, {"PII": "<混合>"}) == "前<混合>后"
    assert (
        pipeline.redact(
            text,
            {"PERSON": "<姓名>", "CN_ADDRESS": "<地址>"},
            default_replacement="<默认>",
        )
        == "前<默认>后"
    )


@pytest.mark.parametrize("override", ["detect", "_run_many"])
def test_custom_detection_path_cannot_silently_downgrade_redaction_coverage(
    override: str,
) -> None:
    text = "前甲乙丙丁戊己后"
    candidates = (
        _candidate("PERSON", 1, 5, score=0.99),
        _candidate("CN_ADDRESS", 3, 7, score=0.80),
    )

    class _WrappedDetect(CascadePipeline):
        def detect(
            self,
            text: str,
            *,
            entities: Sequence[str] | None = None,
        ) -> list[object]:  # type: ignore[override]
            return list(super().detect(text, entities=entities))

    class _WrappedRunMany(CascadePipeline):
        def _run_many(
            self,
            texts: Sequence[str],
            *,
            entities: Sequence[str] | None,
        ) -> list[list[object]]:  # type: ignore[override]
            return [list(items) for items in super()._run_many(texts, entities=entities)]

    pipeline_type = _WrappedDetect if override == "detect" else _WrappedRunMany
    pipeline = pipeline_type(
        config=CascadeConfig(mode="rules-only"),
        rule_recognizer=_StaticRuleRecognizer(candidates),
    )

    with pytest.raises(RuntimeError, match="explicit safe redact") as error:
        pipeline.redact(text)
    assert text not in str(error.value)
