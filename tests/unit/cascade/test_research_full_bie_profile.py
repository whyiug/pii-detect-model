from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.calibration import CalibrationBundle
from pii_zh.cascade.research_bie_profile import (
    ADAPTIVE_RESEARCH_BIE_RULE_POLICIES,
    FPR_GUARDED_ALL6,
    FPR_GUARDED_WEAK5,
    MODEL_ONLY_RULES_DISABLED,
    RESEARCH_FULL_BIE_DECODER_ID,
    _research_routes,
    build_research_full_bie_closed8_pipeline,
    build_research_full_bie_open24_pipeline,
    research_bie_disabled_rule_entities,
)
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.inference.project_bie import (
    PROJECT_CLOSED8_LABELS,
    project_bie_allowed_label_ids,
)
from pii_zh.inference.project_full_bie import (
    FULL_BIE73_MANIFEST_TYPE,
    FULL_BIE73_MODEL_PROTOCOL,
    FULL_BIE73_PROTOCOL_ID,
)
from pii_zh.models.aiguard24_bie import build_core_bie_label_maps


class _Recognizer:
    def __init__(self, **kwargs: object) -> None:
        self.predictor = kwargs["predictor"]

    def analyze(self, text: str, entities: object) -> list[object]:
        del text, entities
        return []


def test_adaptive_rule_policies_disable_only_rules_and_keep_weak5_as_warmup_superset() -> None:
    assert ADAPTIVE_RESEARCH_BIE_RULE_POLICIES == (
        FPR_GUARDED_WEAK5,
        FPR_GUARDED_ALL6,
    )
    weak5 = research_bie_disabled_rule_entities(FPR_GUARDED_WEAK5)
    all6 = research_bie_disabled_rule_entities(FPR_GUARDED_ALL6)
    assert all6 == weak5 | {"CN_ID_CARD"}
    assert all6 - weak5 == {"CN_ID_CARD"}
    assert {
        "EMAIL_ADDRESS",
        "GEO_COORDINATE",
        "MAC_ADDRESS",
        "PHONE_NUMBER",
        "SECRET",
    } <= weak5

    for policy, disabled in ((FPR_GUARDED_WEAK5, weak5), (FPR_GUARDED_ALL6, all6)):
        routes = _research_routes(
            recognizer_thresholds_authoritative=True,
            rule_policy=policy,
        )
        assert {route.entity_type for route in routes if not route.rule_enabled} == disabled
        assert all(route.model_enabled for route in routes)
        assert all(route.model_threshold == 0.0 for route in routes)

    model_only = _research_routes(
        recognizer_thresholds_authoritative=True,
        rule_policy=MODEL_ONLY_RULES_DISABLED,
    )
    assert all(not route.rule_enabled and route.model_enabled for route in model_only)


def test_full_research_builders_bind_explicit_decoder_and_predecode_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "full-model"
    model.mkdir()
    identity = SimpleNamespace(
        to_dict=lambda: {
            "schema_version": "pii-zh.full-bie73-model-identity.v1",
            "manifest_type": FULL_BIE73_MANIFEST_TYPE,
            "manifest_sha256": "a" * 64,
            "model_protocol": FULL_BIE73_MODEL_PROTOCOL,
            "protocol_id": FULL_BIE73_PROTOCOL_ID,
            "model_type": "qwen3_bi",
            "attention_mode": "full",
            "attention_backend": "sdpa",
            "tag_scheme": "BIE",
            "taxonomy_version": "unit",
            "label_count": 73,
            "label_schema_sha256": "b" * 64,
            "weights_combined_sha256": "c" * 64,
        }
    )
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.verify_full_bie73_artifact",
        lambda path: SimpleNamespace(root=Path(path).resolve(), identity=identity),
    )
    _, id2label = build_core_bie_label_maps()
    calls: list[dict[str, object]] = []

    def fake_loader(path: Path, **kwargs: object) -> object:
        calls.append({"path": path, **kwargs})
        allowed = tuple(kwargs["allowed_project_labels"])  # type: ignore[arg-type]
        return SimpleNamespace(
            tokenizer=object(),
            attention_mode="full",
            protocol_id=FULL_BIE73_PROTOCOL_ID,
            decoder_id=kwargs["decoder_id"],
            id2label=id2label,
            allowed_project_labels=frozenset(allowed),
            allowed_label_ids=project_bie_allowed_label_ids(id2label, allowed),
        )

    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.load_local_full_bie73_predictor",
        fake_loader,
    )
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.QwenPiiRecognizer",
        _Recognizer,
    )
    open24 = build_research_full_bie_open24_pipeline(
        model,
        decoder_id=RESEARCH_FULL_BIE_DECODER_ID,
        calibration=CalibrationBundle(default_threshold=0.2),
    )
    closed8 = build_research_full_bie_closed8_pipeline(
        model,
        decoder_id=RESEARCH_FULL_BIE_DECODER_ID,
        calibration=CalibrationBundle(default_threshold=0.2),
    )

    assert [call["allowed_project_labels"] for call in calls] == [
        PII_CORE_LABELS,
        PROJECT_CLOSED8_LABELS,
    ]
    assert all(call["decoder_id"] == "constrained_viterbi" for call in calls)
    assert open24.model_identity is not None
    assert open24.model_identity["model_scope"] == "open24"
    assert open24.model_identity["decoder_id"] == "constrained_viterbi"
    assert closed8.model_identity is not None
    assert closed8.model_identity["model_scope"] == "closed8"
    assert len(calls[0]["allowed_project_labels"]) == 24  # type: ignore[arg-type]
    assert len(calls[1]["allowed_project_labels"]) == 8  # type: ignore[arg-type]


def test_full_research_builder_rejects_decoder_before_artifact_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = False

    def forbidden(_: object) -> object:
        nonlocal reached
        reached = True
        raise AssertionError("artifact verifier must not be reached")

    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.verify_full_bie73_artifact",
        forbidden,
    )
    with pytest.raises(ValueError, match="requires constrained_viterbi"):
        build_research_full_bie_open24_pipeline(
            tmp_path / "missing",
            decoder_id="greedy",
        )
    assert reached is False

    with pytest.raises(TypeError):
        build_research_full_bie_open24_pipeline(tmp_path / "missing")  # type: ignore[call-arg]
    assert reached is False
