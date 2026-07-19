from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.cascade import CascadePipeline, ContextDecision, load_ablation_profile
from pii_zh.cascade.ablation_profiles import REQUIRED_ABLATION_IDS, load_ablation_profiles
from pii_zh.cascade.ablation_runtime import ablation_config, build_ablation_pipeline


class StaticRecognizer:
    def __init__(self, results: Sequence[dict[str, object]]) -> None:
        self.results = list(results)
        self.requested: list[tuple[str, ...]] = []

    def analyze(self, text: str, entities: Sequence[str] | None) -> list[dict[str, object]]:
        del text
        self.requested.append(tuple(entities or ()))
        return list(self.results)


class NoMatchContext:
    def enhance_candidate(self, text: str, candidate: object) -> ContextDecision:
        del text
        return ContextDecision(score=candidate.score, outcome="no_match")  # type: ignore[attr-defined]


def _pipeline(
    adapter_id: str,
    *,
    rule_results: Sequence[dict[str, object]] = (),
    model_results: Sequence[dict[str, object]] = (),
) -> tuple[CascadePipeline, StaticRecognizer, StaticRecognizer]:
    profile = load_ablation_profile(adapter_id)
    rule = StaticRecognizer(rule_results)
    model = StaticRecognizer(model_results)
    pipeline = CascadePipeline(
        config=ablation_config(profile),
        rule_recognizer=rule,
        model_recognizer=model,
        stage_policy=profile.stage_policy,
        allow_evaluation_profile=True,
    )
    return pipeline, rule, model


def test_packaged_profile_set_is_exact_and_raw_profiles_cover_all_24_routes() -> None:
    profiles = load_ablation_profiles()

    assert tuple(profiles) == REQUIRED_ABLATION_IDS
    raw = profiles["project_model_raw"]
    union = profiles["simple_union"]
    configured = frozenset(ablation_config(raw).route_map)
    assert len(configured) == 24
    assert raw.stage_policy.branch_override("model") == configured
    assert union.stage_policy.branch_override("model") == configured
    assert union.stage_policy.branch_override("rule") == configured
    native_presidio = profiles["presidio_zh_ner_cn_common"]
    assert native_presidio.stage_policy.branch_override("model") == configured
    assert native_presidio.stage_policy.branch_override("rule") == frozenset()
    assert native_presidio.recognizer_kind == "native_presidio_cn_common_closed_bio_ner"
    assert native_presidio.mode == "model-only"


def test_evaluation_policy_requires_explicit_opt_in_and_final_requires_context() -> None:
    raw = load_ablation_profile("project_model_raw")
    recognizer = StaticRecognizer([])
    with pytest.raises(ValueError, match="explicit authorization"):
        CascadePipeline(
            config=ablation_config(raw),
            model_recognizer=recognizer,
            stage_policy=raw.stage_policy,
        )

    final = load_ablation_profile("final_cascade")
    with pytest.raises(ValueError, match="requires an explicit context"):
        build_ablation_pipeline(final, recognizer=recognizer)
    assert build_ablation_pipeline(
        final,
        recognizer=recognizer,
        context_enhancer=NoMatchContext(),
    )


def test_presidio_ner_profile_is_model_branch_pass_through_for_native_framework() -> None:
    profile = load_ablation_profile("presidio_zh_ner_cn_common")
    framework = StaticRecognizer([])
    pipeline = build_ablation_pipeline(profile, recognizer=framework)

    assert pipeline.rule_recognizer is None
    assert pipeline.model_recognizer is framework
    assert pipeline.config.model_source == "presidio:native-framework"
    assert profile.stage_policy.validators_enabled is False
    assert profile.stage_policy.thresholds_enabled is False
    assert profile.stage_policy.conflict_policy == "exact_duplicates_only"


def test_project_model_raw_requests_and_keeps_structured_route_disabled_in_release() -> None:
    structured = {
        "entity_type": "PASSPORT_NUMBER",
        "start": 0,
        "end": 3,
        "score": 0.01,
    }
    pipeline, _, model = _pipeline("project_model_raw", model_results=[structured])

    detections = pipeline.detect("ABC")

    assert [item.entity_type for item in detections] == ["PASSPORT_NUMBER"]
    assert detections[0].score == 0.01
    assert "validator:disabled_by_profile" in detections[0].decision_process
    assert "threshold:disabled_by_profile" in detections[0].decision_process
    assert len(model.requested[0]) == 24
    assert "PASSPORT_NUMBER" in model.requested[0]


@pytest.mark.parametrize(
    "result, error",
    [
        ({"entity_type": "PERSON", "start": -1, "end": 1, "score": 0.5}, ValueError),
        ({"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.5}, ValueError),
        ({"entity_type": "PERSON", "start": 0, "end": 1, "score": 1.1}, ValueError),
        ({"entity_type": "PERSON", "start": 0, "end": 1, "score": float("nan")}, ValueError),
    ],
)
def test_raw_profile_still_rejects_invalid_offsets_and_scores(
    result: dict[str, object], error: type[Exception]
) -> None:
    pipeline, _, _ = _pipeline("project_model_raw", model_results=[result])
    with pytest.raises(error):
        pipeline.detect("abc")


def test_raw_profile_drops_unknown_canonical_label() -> None:
    pipeline, _, _ = _pipeline(
        "project_model_raw",
        model_results=[{"entity_type": "UNCONFIGURED_LABEL", "start": 0, "end": 1, "score": 1.0}],
    )
    assert pipeline.detect("abc") == []


def test_simple_union_collapses_only_exact_same_type_duplicates() -> None:
    pipeline, _, _ = _pipeline(
        "simple_union",
        rule_results=[
            {"entity_type": "PERSON", "start": 0, "end": 2, "score": 0.8},
            {"entity_type": "PHONE_NUMBER", "start": 3, "end": 6, "score": 0.7},
        ],
        model_results=[
            # Exact same label/offset: collapse and retain both source IDs.
            {"entity_type": "PERSON", "start": 0, "end": 2, "score": 0.9},
            # Exact offset, different label: do not suppress.
            {"entity_type": "USERNAME", "start": 3, "end": 6, "score": 0.6},
            # Partial overlap, same label as the first span: do not suppress.
            {"entity_type": "PERSON", "start": 1, "end": 4, "score": 0.95},
        ],
    )

    detections = pipeline.detect("abcdef")

    assert [(item.start, item.end, item.entity_type) for item in detections] == [
        (0, 2, "PERSON"),
        (1, 4, "PERSON"),
        (3, 6, "PHONE_NUMBER"),
        (3, 6, "USERNAME"),
    ]
    exact = detections[0]
    assert exact.sources == ("qwen", "rule:cn_common")
    assert "fusion:exact_agreement" in exact.decision_process


def test_validator_threshold_profile_and_final_differ_only_on_required_context_stage() -> None:
    profiles = load_ablation_profiles()
    validator = profiles["validator_threshold_fusion"].stage_policy.to_dict()
    final = profiles["final_cascade"].stage_policy.to_dict()
    differing = {key for key in validator if validator[key] != final[key]}
    assert differing == {"policy_id", "context_enabled", "context_required"}


def test_from_pretrained_final_profile_passes_required_context_into_shared_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = load_ablation_profile("final_cascade")
    context = NoMatchContext()
    fake_predictor = SimpleNamespace(tokenizer=object(), attention_mode="full")

    def fake_loader(model_path: object, **kwargs: object) -> object:
        del model_path, kwargs
        return fake_predictor

    class FakeQwenRecognizer(StaticRecognizer):
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["predictor"] is fake_predictor
            assert kwargs["deduplication_policy"] == "high_overlap_v1"
            super().__init__([])

    import pii_zh.inference.transformers_predictor as predictor_module
    import pii_zh.presidio as presidio_module

    monkeypatch.setattr(predictor_module, "load_local_predictor", fake_loader)
    monkeypatch.setattr(presidio_module, "QwenPiiRecognizer", FakeQwenRecognizer)

    pipeline = CascadePipeline.from_pretrained(
        tmp_path,
        config=ablation_config(profile),
        stage_policy=profile.stage_policy,
        allow_evaluation_profile=True,
        context_enhancer=context,
    )

    assert pipeline.context_enhancer is context
    assert pipeline.stage_policy.context_required is True
