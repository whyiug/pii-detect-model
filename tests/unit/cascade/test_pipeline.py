from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.cascade import (
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    CascadeConfig,
    CascadePipeline,
    ContextDecision,
    EntityRoute,
    build_rules_only_service_pipeline,
    load_service_config,
)
from pii_zh.presidio import ChinesePhraseContextEnhancer, QwenPiiRecognizer


def _span(
    entity_type: str,
    start: int,
    end: int,
    score: float = 0.99,
) -> dict[str, object]:
    return {
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "score": score,
    }


class StaticRecognizer:
    def __init__(self, results: Sequence[dict[str, object]]) -> None:
        self.results = list(results)
        self.calls = 0
        self.entities_seen: list[tuple[str, ...]] = []

    def analyze(self, text: str, entities: Sequence[str]) -> list[dict[str, object]]:
        del text
        self.calls += 1
        self.entities_seen.append(tuple(entities))
        return list(self.results)


class BatchPersonRecognizer:
    def __init__(self) -> None:
        self.batch_calls = 0

    def analyze_batch(
        self, texts: Sequence[str], entities: Sequence[str]
    ) -> list[list[dict[str, object]]]:
        assert "PERSON" in entities
        self.batch_calls += 1
        return [[_span("PERSON_NAME", 0, len(text), 0.99)] for text in texts]


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


class WindowNamePredictor:
    def __init__(self) -> None:
        self.window_count = 0
        self.entity_window_count = 0

    def predict_batch(self, texts: Sequence[str]) -> list[list[dict[str, object]]]:
        self.window_count += len(texts)
        output: list[list[dict[str, object]]] = []
        for text in texts:
            start = text.find("张三")
            if start < 0:
                output.append([])
                continue
            self.entity_window_count += 1
            output.append([_span("PERSON_NAME", start, start + 2, 0.99)])
        return output


def test_rules_only_never_calls_or_loads_injected_model_and_redacts() -> None:
    model = StaticRecognizer([_span("PERSON", 0, 2)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="rules-only"),
        model_recognizer=model,
    )
    text = "请联系19912345678。"

    detections = pipeline.detect(text)

    assert model.calls == 0
    assert [(item.entity_type, text[item.start : item.end]) for item in detections] == [
        ("PHONE_NUMBER", "19912345678")
    ]
    assert detections[0].decision_process == (
        "candidate:rule",
        "route:structured",
        "validator:passed",
        "threshold:passed",
        "fusion:single_source",
    )
    assert pipeline.redact(text, {"PHONE_NUMBER": "<电话>"}) == "请联系<电话>。"


def test_validator_runs_before_fusion_so_invalid_specific_model_cannot_hide_rule() -> None:
    text = "手机19912345678"
    start = text.index("19912345678")
    # If this invalid, more-specific ID candidate reached fusion first, it
    # would displace the valid phone candidate on the same boundary.
    model = StaticRecognizer([_span("CN_ID_CARD", start, len(text), 1.0)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="cascade"),
        model_recognizer=model,
    )

    detections = pipeline.detect(text)

    assert [(item.entity_type, item.start, item.end) for item in detections] == [
        ("PHONE_NUMBER", start, len(text))
    ]


def test_exact_rule_model_agreement_records_both_sources_without_raw_value() -> None:
    text = "张三"
    rule = StaticRecognizer([_span("PERSON", 0, 2, 0.80)])
    model = StaticRecognizer([_span("PERSON_NAME", 0, 2, 0.99)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="cascade"),
        rule_recognizer=rule,
        model_recognizer=model,
    )

    detections = pipeline.detect(text)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.entity_type == "PERSON"
    assert detection.source == "rule:cn_common"
    assert detection.sources == ("qwen", "rule:cn_common")
    assert detection.decision_process[-2:] == (
        "fusion:exact_same_type_collapsed",
        "fusion:exact_agreement",
    )
    assert "张三" not in repr(detection.to_dict())


def test_default_route_closes_unvalidated_structured_model_branch() -> None:
    model = StaticRecognizer([_span("PASSPORT_NUMBER", 0, 9)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=model,
    )

    assert pipeline.detect("E12345678") == []
    assert "PASSPORT_NUMBER" not in model.entities_seen[0]


def test_successor_plate_rule_is_validator_backed_and_auditable() -> None:
    text = "登记车牌京A12345已核验。"
    start = text.index("京A12345")
    pipeline = build_rules_only_service_pipeline(SUCCESSOR_SERVICE_PROFILE_VERSION)

    detections = pipeline.detect(text, entities=["CN_VEHICLE_LICENSE_PLATE"])

    assert len(detections) == 1
    detection = detections[0]
    assert (detection.entity_type, detection.start, detection.end) == (
        "CN_VEHICLE_LICENSE_PLATE",
        start,
        start + len("京A12345"),
    )
    assert detection.source == "rule:cn_common_v6"
    assert detection.sources == ("rule:cn_common_v6",)
    assert detection.decision_process == (
        "candidate:rule",
        "route:structured",
        "validator:passed",
        "threshold:passed",
        "fusion:single_source",
    )


def test_successor_profile_rejects_model_mode_instead_of_constructing_a_partial_model() -> None:
    with pytest.raises(ValueError, match="requires rules-only mode"):
        load_service_config(
            SUCCESSOR_SERVICE_PROFILE_VERSION,
            mode="model-only",
        )


def test_custom_structured_model_route_requires_an_installed_validator() -> None:
    route = EntityRoute(
        "PASSPORT_NUMBER",
        "structured",
        rule_enabled=False,
        model_enabled=True,
        validator_label="PASSPORT_NUMBER",
        model_requires_validator=True,
    )
    config = CascadeConfig(mode="model-only", routes=(route,))
    model = StaticRecognizer([_span("PASSPORT_NUMBER", 0, 9)])

    with pytest.raises(ValueError, match="unavailable validator"):
        CascadePipeline(config=config, model_recognizer=model)

    pipeline = CascadePipeline(
        config=config,
        model_recognizer=model,
        validators={"PASSPORT_NUMBER": lambda value: value.startswith("E")},
    )
    assert len(pipeline.detect("E12345678")) == 1
    assert pipeline.detect("X12345678") == []


def test_short_semantic_model_span_uses_stricter_threshold() -> None:
    weak = StaticRecognizer([_span("PERSON_NAME", 0, 2, 0.97)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=weak,
    )

    assert pipeline.detect("张三") == []

    strong = StaticRecognizer([_span("PERSON_NAME", 0, 2, 0.99)])
    strong_pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=strong,
    )
    assert [item.entity_type for item in strong_pipeline.detect("张三")] == ["PERSON"]


def test_context_stage_runs_before_threshold_with_raw_text_free_audit_step() -> None:
    route = EntityRoute(
        "PERSON",
        "semantic",
        rule_enabled=False,
        model_enabled=True,
        model_threshold=0.60,
    )
    model = StaticRecognizer([_span("PERSON", 3, 5, 0.55)])
    config = CascadeConfig(mode="model-only", routes=(route,))
    text = "联系人张三"

    assert CascadePipeline(config=config, model_recognizer=model).detect(text) == []
    pipeline = CascadePipeline(
        config=config,
        model_recognizer=model,
        context_enhancer=ChinesePhraseContextEnhancer(
            positive_context={"PERSON": ["联系人"]},
        ),
    )

    detection = pipeline.detect(text)[0]

    assert detection.score == pytest.approx(0.70)
    assert detection.decision_process == (
        "candidate:model",
        "route:semantic",
        "context:positive",
        "validator:not_required",
        "threshold:passed",
        "fusion:single_source",
    )
    assert "联系人" not in repr(detection.to_dict())


def test_positive_context_cannot_resurrect_a_validator_rejection() -> None:
    route = EntityRoute(
        "PASSPORT_NUMBER",
        "structured",
        rule_enabled=False,
        model_enabled=True,
        validator_label="PASSPORT_NUMBER",
        model_requires_validator=True,
        model_threshold=0.80,
    )
    text = "护照号E12345678"
    model = StaticRecognizer([_span("PASSPORT_NUMBER", 3, len(text), 0.70)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only", routes=(route,)),
        model_recognizer=model,
        validators={"PASSPORT_NUMBER": lambda value: False},
        context_enhancer=ChinesePhraseContextEnhancer(
            positive_context={"PASSPORT_NUMBER": ["护照号"]},
            positive_delta=0.20,
        ),
    )

    assert pipeline.detect(text) == []


def test_low_score_invalid_candidate_is_validated_before_threshold_rejection() -> None:
    route = EntityRoute(
        "PASSPORT_NUMBER",
        "structured",
        rule_enabled=False,
        model_enabled=True,
        validator_label="PASSPORT_NUMBER",
        model_requires_validator=True,
        model_threshold=0.95,
    )
    text = "护照号E12345678"
    validated_values: list[str] = []

    def reject(value: str) -> bool:
        validated_values.append(value)
        return False

    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only", routes=(route,)),
        model_recognizer=StaticRecognizer([_span("PASSPORT_NUMBER", 3, len(text), 0.10)]),
        validators={"PASSPORT_NUMBER": reject},
        context_enhancer=ChinesePhraseContextEnhancer(
            positive_context={"PASSPORT_NUMBER": ["护照号"]},
            positive_delta=0.20,
        ),
    )

    assert pipeline.detect(text) == []
    assert validated_values == ["E12345678"]


@pytest.mark.parametrize("mode", ["rules-only", "model-only", "cascade"])
def test_context_failure_is_fail_closed_in_every_mode(mode: str) -> None:
    sensitive = "联系人张三"

    class FailingContextEnhancer:
        def enhance_candidate(self, text: str, candidate: object) -> ContextDecision:
            del candidate
            raise ValueError(text)

    recognizer = StaticRecognizer([_span("PERSON", 3, 5, 0.99)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode=mode),  # type: ignore[arg-type]
        rule_recognizer=recognizer,
        model_recognizer=recognizer,
        context_enhancer=FailingContextEnhancer(),
    )

    with pytest.raises(RuntimeError, match="context enhancer failed") as error:
        pipeline.detect(sensitive)
    assert sensitive not in str(error.value)


def test_detect_batch_uses_batch_adapter_and_matches_single_schema() -> None:
    model = BatchPersonRecognizer()
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=model,
    )

    batch = pipeline.detect_batch(["张三", "李四"])

    assert model.batch_calls == 1
    assert [[item.to_dict() for item in row] for row in batch] == [
        [pipeline.detect("张三")[0].to_dict()],
        [pipeline.detect("李四")[0].to_dict()],
    ]


def test_long_unicode_windows_keep_batch_single_offsets_and_replacement_in_parity() -> None:
    predictor = WindowNamePredictor()
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=CharacterTokenizer(),
        label_mapping={"PERSON_NAME": "PERSON"},
        max_tokens=12,
        stride_fraction=0.25,
    )
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=recognizer,
    )
    texts = [
        "甲🙂Ａ乙" * 8 + "张三" + "后缀\n" + "戊" * 24,
        "丙🙂Ｂ丁" * 4 + "张三" + "尾" * 20,
    ]

    batched = pipeline.detect_batch(texts)
    batch_entity_window_count = predictor.entity_window_count
    singles = [pipeline.detect(text) for text in texts]

    assert predictor.window_count > len(texts)
    assert batch_entity_window_count > len(texts)
    assert [[item.to_dict() for item in row] for row in batched] == [
        [item.to_dict() for item in row] for row in singles
    ]
    for text, detections in zip(texts, batched, strict=True):
        assert len(detections) == 1
        detection = detections[0]
        expected_start = text.index("张三")
        assert (detection.start, detection.end) == (expected_start, expected_start + 2)
        assert text[detection.start : detection.end] == "张三"
        assert "张三" not in repr(detection.to_dict())

    first = texts[0]
    start = first.index("张三")
    assert pipeline.redact(first, {"PERSON": "<姓名>"}) == (
        first[:start] + "<姓名>" + first[start + 2 :]
    )


def test_empty_entity_filter_short_circuits_both_recognizers() -> None:
    rule = StaticRecognizer([_span("PERSON", 0, 2)])
    model = StaticRecognizer([_span("PERSON", 0, 2)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="cascade"),
        rule_recognizer=rule,
        model_recognizer=model,
    )

    assert pipeline.detect("张三", entities=[]) == []
    assert rule.calls == 0
    assert model.calls == 0


def test_out_of_bounds_recognizer_result_raises_instead_of_claiming_safe_output() -> None:
    model = StaticRecognizer([_span("PERSON", 0, 20)])
    pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=model,
    )

    with pytest.raises(ValueError, match="outside the input length"):
        pipeline.detect("短文本")


def test_modes_require_their_recognizers_and_entity_filters_are_strict() -> None:
    with pytest.raises(ValueError, match="model recognizer"):
        CascadePipeline(config=CascadeConfig(mode="model-only"))

    pipeline = CascadePipeline(config=CascadeConfig(mode="rules-only"))
    with pytest.raises(ValueError, match="not configured"):
        pipeline.detect("文本", entities=["NOT_A_PII_LABEL"])
    with pytest.raises(TypeError, match="sequence"):
        pipeline.detect("文本", entities="PERSON")  # type: ignore[arg-type]


def test_recognizer_and_validator_failures_do_not_echo_sensitive_values() -> None:
    sensitive = "synthetic-secret-value"

    class FailingRecognizer:
        def analyze(self, text: str, entities: Sequence[str]) -> list[object]:
            del text, entities
            raise ValueError(sensitive)

    recognizer_pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only"),
        model_recognizer=FailingRecognizer(),
    )
    with pytest.raises(RuntimeError, match="model recognizer failed") as recognizer_error:
        recognizer_pipeline.detect(sensitive)
    assert sensitive not in str(recognizer_error.value)

    route = EntityRoute(
        "PASSPORT_NUMBER",
        "structured",
        rule_enabled=False,
        model_enabled=True,
        validator_label="PASSPORT_NUMBER",
        model_requires_validator=True,
    )

    def failing_validator(value: str) -> bool:
        raise ValueError(value)

    validator_pipeline = CascadePipeline(
        config=CascadeConfig(mode="model-only", routes=(route,)),
        model_recognizer=StaticRecognizer([_span("PASSPORT_NUMBER", 0, len(sensitive))]),
        validators={"PASSPORT_NUMBER": failing_validator},
    )
    with pytest.raises(RuntimeError, match="structured validator failed") as validator_error:
        validator_pipeline.detect(sensitive)
    assert sensitive not in str(validator_error.value)


def test_model_failure_never_silently_falls_back_but_rules_only_is_explicitly_available() -> None:
    text = "邮箱 demo@example.com"
    start = text.index("demo@example.com")
    rule = StaticRecognizer([_span("EMAIL_ADDRESS", start, len(text), 0.99)])

    class FailingModel:
        def analyze(self, text: str, entities: Sequence[str]) -> list[object]:
            del entities
            raise ValueError(text)

    cascade = CascadePipeline(
        config=CascadeConfig(mode="cascade"),
        rule_recognizer=rule,
        model_recognizer=FailingModel(),
    )
    with pytest.raises(RuntimeError, match="model recognizer failed") as error:
        cascade.detect(text)
    assert text not in str(error.value)

    explicit_fallback = CascadePipeline(
        config=CascadeConfig(mode="rules-only"),
        rule_recognizer=rule,
    )
    detections = explicit_fallback.detect(text)
    assert [(item.entity_type, item.start, item.end) for item in detections] == [
        ("EMAIL_ADDRESS", start, len(text))
    ]


def test_from_pretrained_is_local_only_and_builds_a_model_enabled_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="local_files_only=True"):
        CascadePipeline.from_pretrained(tmp_path, local_files_only=False)
    with pytest.raises(ValueError, match="model-enabled"):
        CascadePipeline.from_pretrained(
            tmp_path,
            config=CascadeConfig(mode="rules-only"),
        )

    loader_calls: list[tuple[object, object, int]] = []
    fake_predictor = SimpleNamespace(tokenizer=object(), attention_mode="full")

    def fake_loader(
        model_path: object,
        *,
        device: object,
        dtype: object,
        micro_batch_size: int,
    ) -> object:
        loader_calls.append((model_path, device, micro_batch_size))
        assert dtype is None
        return fake_predictor

    class FakeQwenRecognizer(StaticRecognizer):
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["predictor"] is fake_predictor
            assert kwargs["tokenizer"] is fake_predictor.tokenizer
            assert kwargs["attention_mode"] == "full"
            super().__init__([])

    import pii_zh.inference.transformers_predictor as predictor_module
    import pii_zh.presidio as presidio_module

    monkeypatch.setattr(predictor_module, "load_local_predictor", fake_loader)
    monkeypatch.setattr(presidio_module, "QwenPiiRecognizer", FakeQwenRecognizer)

    pipeline = CascadePipeline.from_pretrained(
        tmp_path,
        device="cpu",
        micro_batch_size=4,
    )

    assert pipeline.config.mode == "cascade"
    assert pipeline.model_recognizer is not None
    assert loader_calls == [(tmp_path, "cpu", 4)]
