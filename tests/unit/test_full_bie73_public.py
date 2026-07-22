from __future__ import annotations

import builtins
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

OPEN24 = tuple(f"LABEL_{index:02d}" for index in range(24))
CLOSED8 = OPEN24[:8]


@dataclass(frozen=True)
class _FakeConfig:
    profile_version: str = "research-full-bie73"
    mode: str = "cascade"


def _module() -> Any:
    return importlib.import_module("pii_zh.full_bie73")


def _predictor_runtime(calls: list[dict[str, Any]]) -> dict[str, Any]:
    id2label = {0: "O", **{index + 1: label for index, label in enumerate(OPEN24)}}

    def allowed_ids(_id2label: object, labels: object) -> tuple[int, ...]:
        selected = set(labels)  # type: ignore[arg-type]
        return tuple(index + 1 for index, label in enumerate(OPEN24) if label in selected)

    def loader(path: object, **kwargs: Any) -> object:
        calls.append({"path": path, **kwargs})
        labels = tuple(kwargs["allowed_project_labels"])
        return SimpleNamespace(
            id2label=id2label,
            allowed_label_ids=allowed_ids(id2label, labels),
            allowed_project_labels=frozenset(labels),
            protocol_id="unit-full-bie73",
            attention_mode="full",
            decoder_id=kwargs["decoder_id"],
        )

    return {
        "closed8_labels": CLOSED8,
        "loader": loader,
        "model_protocol": "FULL_BIE73",
        "open24_labels": OPEN24,
        "project_allowed_label_ids": allowed_ids,
        "protocol_id": "unit-full-bie73",
    }


def _identity(
    module: Any,
    *,
    scope: str,
    labels: tuple[str, ...],
    policy: str,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    resolved_thresholds = thresholds or {label: 0.5 for label in OPEN24}
    return {
        "schema_version": "pii-zh.full-bie73-model-identity.v1",
        "manifest_type": "unit-full-manifest",
        "manifest_sha256": "a" * 64,
        "model_protocol": "FULL_BIE73",
        "protocol_id": "unit-full-bie73",
        "model_type": "qwen3_bi",
        "attention_mode": "full",
        "attention_backend": "sdpa",
        "tag_scheme": "BIE",
        "taxonomy_version": "unit",
        "label_count": 73,
        "label_schema_sha256": "b" * 64,
        "weights_combined_sha256": "c" * 64,
        "calibration_sha256": None,
        "thresholds_sha256": module._canonical_json_hash(resolved_thresholds),
        "allowed_model_labels_sha256": module._canonical_json_hash(labels),
        "decoder_id": "constrained_viterbi",
        "model_scope": scope,
        "rule_policy": policy,
    }


def _service_runtime(module: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    def builder_for(scope: str) -> Any:
        labels = OPEN24 if scope == "open24" else CLOSED8

        def builder(path: object, **kwargs: Any) -> object:
            pipeline = SimpleNamespace(
                config=_FakeConfig(),
                rule_recognizer=object(),
                model_recognizer=object(),
                validators={"unit": object()},
                stage_policy=object(),
                _fusion=object(),
                model_identity=_identity(
                    module,
                    scope=scope,
                    labels=labels,
                    policy=kwargs["rule_policy"],
                    thresholds=kwargs["thresholds"],
                ),
            )
            calls.append(
                {
                    "path": path,
                    "scope": scope,
                    "pipeline": pipeline,
                    "rule_recognizer": pipeline.rule_recognizer,
                    "model_recognizer": pipeline.model_recognizer,
                    "validators": pipeline.validators,
                    "stage_policy": pipeline.stage_policy,
                    "fusion": pipeline._fusion,
                    **kwargs,
                }
            )
            return pipeline

        return builder

    return {
        "adaptive_rule_policies": ("fpr_guarded_weak5", "fpr_guarded_all6"),
        "closed8_builder": builder_for("closed8"),
        "closed8_labels": CLOSED8,
        "decoder_id": "constrained_viterbi",
        "manifest_type": "unit-full-manifest",
        "model_protocol": "FULL_BIE73",
        "model_only_rule_policy": "model_only_rules_disabled",
        "open24_builder": builder_for("open24"),
        "open24_labels": OPEN24,
        "protocol_id": "unit-full-bie73",
        "threshold_normalizer": lambda values: dict(values),
    }


def test_import_is_lazy_and_uses_only_standard_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "pii_zh.full_bie73", raising=False)
    original_import = builtins.__import__
    blocked = (
        "torch",
        "transformers",
        "presidio_analyzer",
        "pii_zh.cascade",
        "pii_zh.evaluation",
        "pii_zh.inference",
    )

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked or name.startswith(tuple(f"{item}." for item in blocked)):
            raise AssertionError(f"eager runtime import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("pii_zh.full_bie73")
    assert module.FULL_BIE73_DECODER_ID == "constrained_viterbi"


@pytest.mark.parametrize(
    ("scope", "expected_labels"),
    (("open24", OPEN24), ("closed8", CLOSED8)),
)
def test_predictor_routes_scope_before_fixed_decoding(
    scope: str,
    expected_labels: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "_load_predictor_runtime", lambda: _predictor_runtime(calls))
    dtype = object()

    predictor = module.load_full_bie73_predictor(
        "/model",
        scope=scope,
        device="cuda:4",
        dtype=dtype,
        micro_batch_size=3,
    )

    assert predictor is not None
    assert calls == [
        {
            "path": "/model",
            "decoder_id": "constrained_viterbi",
            "device": "cuda:4",
            "micro_batch_size": 3,
            "allowed_project_labels": expected_labels,
            "dtype": dtype,
        }
    ]


def test_predictor_rejects_parameters_before_runtime_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    def forbidden() -> object:
        raise AssertionError("runtime must remain lazy")

    monkeypatch.setattr(module, "_load_predictor_runtime", forbidden)
    with pytest.raises(ValueError, match="scope"):
        module.load_full_bie73_predictor("/model", scope="all")
    with pytest.raises(TypeError, match="micro_batch_size"):
        module.load_full_bie73_predictor("/model", scope="open24", micro_batch_size=True)
    with pytest.raises(ValueError, match="positive"):
        module.load_full_bie73_predictor("/model", scope="open24", micro_batch_size=0)


def test_service_defaults_to_presidio_primary_open24_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, Any]] = []

    pipeline = SimpleNamespace(config=_FakeConfig(), model_identity={"profile": "unit"})

    def build(path: object, **kwargs: Any) -> object:
        calls.append({"path": path, **kwargs})
        return pipeline

    monkeypatch.setattr(module, "_build_presidio_primary_service", build)

    result = module.build_full_bie73_service_pipeline(
        "/model",
        primary_model_path="/primary",
    )

    assert result is pipeline
    assert calls == [
        {
            "path": "/model",
            "primary_model_path": "/primary",
            "device": "cpu",
            "dtype": None,
            "micro_batch_size": 16,
        }
    ]


def test_presidio_primary_service_rejects_missing_or_research_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("runtime must not load for an invalid public contract")

    monkeypatch.setattr(module, "_build_presidio_primary_service", forbidden)
    thresholds = {label: 0.5 for label in OPEN24}
    with pytest.raises(ValueError, match="primary_model_path"):
        module.build_full_bie73_service_pipeline("/model")
    with pytest.raises(ValueError, match="fixed open24"):
        module.build_full_bie73_service_pipeline(
            "/model",
            primary_model_path="/primary",
            scope="closed8",
        )
    with pytest.raises(ValueError, match="research rule_policy"):
        module.build_full_bie73_service_pipeline(
            "/model",
            primary_model_path="/primary",
            rule_policy="preserve_all_validated_rules",
        )
    with pytest.raises(ValueError, match="frozen thresholds"):
        module.build_full_bie73_service_pipeline(
            "/model",
            primary_model_path="/primary",
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="internal evaluated"):
        module.build_full_bie73_service_pipeline(
            "/model",
            rule_policy="fpr_guarded_weak5",
            thresholds=thresholds,
            scope="open24",
            mode="model-only",
        )
    with pytest.raises(ValueError, match="only valid"):
        module.build_full_bie73_service_pipeline(
            "/model",
            primary_model_path="/primary",
            mode="model-only",
        )
    with pytest.raises(ValueError, match="mode"):
        module.build_full_bie73_service_pipeline(
            "/model",
            thresholds=thresholds,
            scope="open24",
            mode="rules-only",
        )



def test_service_accepts_an_exact_threshold_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "_load_service_runtime", lambda: _service_runtime(module, calls))
    threshold_file = tmp_path / "thresholds.json"
    threshold_file.write_text(
        json.dumps({label: 0.5 for label in reversed(OPEN24)}),
        encoding="utf-8",
    )

    module.build_full_bie73_service_pipeline(
        "/model",
        thresholds=threshold_file,
        scope="open24",
        mode="model-only",
    )

    assert calls[0]["thresholds"] == {label: 0.5 for label in OPEN24}


def test_service_rejects_duplicate_threshold_file_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "_load_service_runtime", lambda: _service_runtime(module, calls))
    items = [f'"{label}": 0.5' for label in OPEN24]
    items.append(f'"{OPEN24[0]}": 0.5')
    threshold_file = tmp_path / "duplicate.json"
    threshold_file.write_text("{" + ",".join(items) + "}", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        module.build_full_bie73_service_pipeline(
            "/model",
            thresholds=threshold_file,
            scope="open24",
            mode="model-only",
        )
    assert calls == []


def test_model_only_selects_internal_policy_and_marks_public_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "_load_service_runtime", lambda: _service_runtime(module, calls))
    thresholds = {label: 0.5 for label in OPEN24}

    pipeline = module.build_full_bie73_service_pipeline(
        "/model",
        thresholds=thresholds,
        scope="open24",
        mode="model-only",
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["mode"] == "cascade"
    assert call["rule_policy"] == "model_only_rules_disabled"
    assert pipeline.config.mode == "model-only"
    assert pipeline.config.profile_version == "community-presidio-bie73-cascade-v1"
    assert pipeline.rule_recognizer is None
    identity = dict(pipeline.model_identity)
    assert identity["service_mode"] == "model-only"
    assert identity["rule_policy"] == "model_only_rules_disabled"
    assert identity["profile_purpose"] == "model_only_service_ablation_with_validators"
    assert identity["validators_enabled"] is True
    assert identity["raw_model_benchmark_equivalent"] is False
    assert identity["matches_selected_cascade"] is False


def test_model_only_real_pipeline_skips_rules_but_keeps_validators_and_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from pii_zh.cascade.pipeline import CascadePipeline
    from pii_zh.cascade.research_bie_profile import (
        ADAPTIVE_RESEARCH_BIE_RULE_POLICIES,
        MODEL_ONLY_RULES_DISABLED,
        _research_routes,
    )
    from pii_zh.cascade.service_profiles import (
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        _normalize_thresholds,
        load_service_config,
    )
    from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
    from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
    from pii_zh.inference.project_bie import PROJECT_CLOSED8_LABELS

    module = _module()
    builder_calls: list[dict[str, Any]] = []
    rule_calls: list[object] = []
    model_calls: list[tuple[str, tuple[str, ...]]] = []

    class ForbiddenRuleRecognizer:
        def analyze(self, text: str, entities: object) -> list[object]:
            rule_calls.append((text, entities))
            raise AssertionError("model-only must not call the rule recognizer")

    class ModelRecognizer:
        def analyze(self, text: str, entities: object) -> list[dict[str, object]]:
            requested = tuple(entities)  # type: ignore[arg-type]
            model_calls.append((text, requested))
            return [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "start": 0,
                    "end": len(text),
                    "score": 0.99,
                    "offset_mode": "relative",
                }
            ]

    model_recognizer = ModelRecognizer()
    captured: dict[str, Any] = {}

    def builder(path: object, **kwargs: Any) -> CascadePipeline:
        del path
        builder_calls.append(dict(kwargs))
        config = load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode="cascade")
        config = replace(
            config,
            routes=_research_routes(
                recognizer_thresholds_authoritative=True,
                rule_policy=kwargs["rule_policy"],
            ),
        )
        pipeline = CascadePipeline(
            config=config,
            rule_recognizer=ForbiddenRuleRecognizer(),
            model_recognizer=model_recognizer,
            validators=COMMUNITY_STRUCTURED_VALIDATORS,
            model_identity=_identity(
                module,
                scope="open24",
                labels=PII_CORE_LABELS,
                policy=kwargs["rule_policy"],
                thresholds=kwargs["thresholds"],
            ),
        )
        captured.update(
            {
                "fusion": pipeline._fusion,
                "model_recognizer": pipeline.model_recognizer,
                "validators": pipeline.validators,
            }
        )
        return pipeline

    runtime = {
        "adaptive_rule_policies": ADAPTIVE_RESEARCH_BIE_RULE_POLICIES,
        "closed8_builder": builder,
        "closed8_labels": PROJECT_CLOSED8_LABELS,
        "decoder_id": "constrained_viterbi",
        "manifest_type": "unit-full-manifest",
        "model_protocol": "FULL_BIE73",
        "model_only_rule_policy": MODEL_ONLY_RULES_DISABLED,
        "open24_builder": builder,
        "open24_labels": PII_CORE_LABELS,
        "protocol_id": "unit-full-bie73",
        "threshold_normalizer": _normalize_thresholds,
    }
    monkeypatch.setattr(module, "_load_service_runtime", lambda: runtime)

    pipeline = module.build_full_bie73_service_pipeline(
        "/model",
        mode="model-only",
    )

    assert builder_calls[0]["mode"] == "cascade"
    assert builder_calls[0]["rule_policy"] == MODEL_ONLY_RULES_DISABLED
    assert pipeline.config.mode == "model-only"
    assert pipeline.rule_recognizer is None
    assert pipeline.model_recognizer is captured["model_recognizer"] is model_recognizer
    assert pipeline.validators is captured["validators"]
    assert pipeline._fusion is captured["fusion"]
    assert pipeline.detect("not-an-email", entities=["EMAIL_ADDRESS"]) == []
    detections = pipeline.detect("demo@example.com", entities=["EMAIL_ADDRESS"])
    assert len(detections) == 1
    assert detections[0].entity_type == "EMAIL_ADDRESS"
    assert "validator:passed" in detections[0].decision_process
    assert rule_calls == []
    assert [text for text, _entities in model_calls] == ["not-an-email", "demo@example.com"]
