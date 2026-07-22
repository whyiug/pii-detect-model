from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.calibration import CalibrationBundle
from pii_zh.cascade.community_model import (
    CommunityModelContractError,
    verify_community_model_artifact,
)
from pii_zh.cascade.research_bie_profile import (
    COMMUNITY_CALIBRATED_RULE_SUPPRESSION,
    PRESERVE_ALL_VALIDATED_RULES,
    RESEARCH_BIE_MANIFEST_TYPE,
    RESEARCH_BIE_MODEL_PROTOCOL,
    ResearchBieArtifactError,
    build_research_bie_closed8_pipeline,
    verify_research_bie_artifact,
)
from pii_zh.inference.project_bie import PROJECT_CLOSED8_LABELS
from pii_zh.models.aiguard24 import AIGUARD_SOURCE_MODEL_ID, AIGUARD_SOURCE_REVISION
from pii_zh.models.aiguard24_bie import (
    AIGUARD24_BIE_INITIALIZATION_STRATEGY,
    build_core_bie_label_maps,
)
from pii_zh.taxonomy import load_taxonomy
from pii_zh.training.manifest import (
    canonical_json_hash,
    output_artifact_fingerprint,
)


def _write_complete_bie_artifact(root: Path) -> dict[str, object]:
    root.mkdir()
    label2id, id2label = build_core_bie_label_maps()
    taxonomy_version = load_taxonomy().taxonomy_version
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForTokenClassification"],
        "pii_attention_mode": "causal",
        "pii_tagging_scheme": "BIE",
        "pii_training_status": "completed_independent_dev_feasible_candidate",
        "pii_taxonomy_version": taxonomy_version,
        "pii_release_eligible": False,
        "use_cache": False,
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": 73,
    }
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"unit-test-safetensors")
    (root / "tokenizer.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"fixture":true}\n', encoding="utf-8")
    recipe = {
        "attention_mode": "causal",
        "tag_scheme": "BIE",
        "label_count": 73,
        "initialization_strategy": AIGUARD24_BIE_INITIALIZATION_STRATEGY,
        "checkpoint_selection_split": "independent_validation",
        "checkpoint_selection_metric": "independent_dev_selection_score",
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": RESEARCH_BIE_MANIFEST_TYPE,
        "status": "completed",
        "release_eligible": False,
        "created_at": "2026-07-21T00:00:00+00:00",
        "completed_at": "2026-07-21T00:01:00+00:00",
        "seed": 42,
        "attention_mode": "causal",
        "tag_scheme": "BIE",
        "update_mode": "head_only",
        "base_source_id": AIGUARD_SOURCE_MODEL_ID,
        "source_revision": AIGUARD_SOURCE_REVISION,
        "training_source_ids": [AIGUARD_SOURCE_MODEL_ID, "synthetic-unit-test"],
        "taxonomy_version": taxonomy_version,
        "label2id": label2id,
        "label_schema_sha256": canonical_json_hash(label2id),
        "recipe": recipe,
        "recipe_sha256": canonical_json_hash(recipe),
        "initialization": {
            "strategy": AIGUARD24_BIE_INITIALIZATION_STRATEGY,
            "source_model_id": AIGUARD_SOURCE_MODEL_ID,
            "source_revision": AIGUARD_SOURCE_REVISION,
            "target_tag_scheme": "BIE",
            "target_label_count": 73,
            "target_label2id": label2id,
            "attention_mode": "causal",
            "release_eligible": False,
        },
        "parameter_update_audit": {
            "trainable_parameter_count": 1,
            "frozen_parameter_count": 1,
        },
        "tokenizer": {"effective_contract_sha256": "1" * 64},
        "datasets": {
            "train": {
                "sha256": "2" * 64,
                "summary": {"document_count": 1},
                "admission": {"documents": 1},
            },
            "independent_validation": {
                "sha256": "3" * 64,
                "summary": {"document_count": 1},
                "admission": {"documents": 1},
            },
        },
        "split_isolation": {
            "policy": "strict_all_groups",
            "collision_counts": {
                "document_id": 0,
                "template_group": 0,
                "entity_value_group": 0,
                "source_group": 0,
            },
            "template_group_overlap_allowed": False,
        },
        "dev_selection_gates": {"feasibility_first": True},
        "benchmark_isolation": {
            "public_test_read_for_training": False,
            "public_test_read_for_validation": False,
            "public_test_read_for_checkpoint_selection": False,
            "pii_bench_zh_read": False,
        },
        "versions": {"python": "fixture"},
        "code_revision": "4" * 40,
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "trainer_reporting_integrations": [],
        },
        "runtime_environment": {"execution_device": "cuda"},
        "checkpoint_selection": {
            "selection_split": "independent_validation",
            "selection_metric": "independent_dev_selection_score",
            "selected_checkpoint_id": "checkpoint-1",
            "selected_global_step": 1,
            "best_metric": 1.5,
            "final_validation_replay_metric": 1.5,
            "safetensor_sha256": {"model.safetensors": "5" * 64},
        },
        "independent_validation": {"eval_dev_gate_passed": 1.0},
        "output_artifact": output_artifact_fingerprint(root),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    (root / "training_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _rewrite_manifest(root: Path, manifest: dict[str, object]) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    (root / "training_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_research_bie_verifier_accepts_only_its_complete_bound_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bie"
    manifest = _write_complete_bie_artifact(root)

    verified = verify_research_bie_artifact(root)

    assert verified.root == root.resolve()
    assert verified.identity.manifest_type == RESEARCH_BIE_MANIFEST_TYPE
    assert verified.identity.model_protocol == RESEARCH_BIE_MODEL_PROTOCOL
    assert verified.identity.label_count == 73
    with pytest.raises(CommunityModelContractError):
        verify_community_model_artifact(root)

    manifest["manifest_type"] = "aiguard24_community_training_v1"
    _rewrite_manifest(root, manifest)
    with pytest.raises(ResearchBieArtifactError, match="manifest protocol"):
        verify_research_bie_artifact(root)


def test_research_bie_verifier_rejects_incomplete_manifest(tmp_path: Path) -> None:
    root = tmp_path / "bie"
    manifest = _write_complete_bie_artifact(root)
    manifest.pop("checkpoint_selection")
    _rewrite_manifest(root, manifest)

    with pytest.raises(ResearchBieArtifactError, match="manifest is incomplete"):
        verify_research_bie_artifact(root)


def test_research_builder_installs_fixed_project_mask_before_recognizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    identity = SimpleNamespace(
        to_dict=lambda: {
            "manifest_sha256": "a" * 64,
            "model_protocol": RESEARCH_BIE_MODEL_PROTOCOL,
        }
    )
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.verify_research_bie_artifact",
        lambda path: SimpleNamespace(root=Path(path).resolve(), identity=identity),
    )
    loader_calls: list[dict[str, object]] = []
    predictor = SimpleNamespace(
        tokenizer=object(),
        attention_mode="causal",
        allowed_project_labels=frozenset(PROJECT_CLOSED8_LABELS),
    )

    def fake_loader(path: Path, **kwargs: object) -> object:
        loader_calls.append({"path": path, **kwargs})
        return predictor

    class FakeRecognizer:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["predictor"] is predictor

        def analyze(self, text: str, entities: object) -> list[object]:
            del text, entities
            return []

    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.load_local_project_bie_predictor",
        fake_loader,
    )
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.QwenPiiRecognizer",
        FakeRecognizer,
    )

    pipeline = build_research_bie_closed8_pipeline(
        model,
        calibration=CalibrationBundle(default_threshold=0.2),
        micro_batch_size=3,
    )

    assert loader_calls == [
        {
            "path": model.resolve(),
            "device": "cpu",
            "micro_batch_size": 3,
            "allowed_project_labels": PROJECT_CLOSED8_LABELS,
        }
    ]
    assert pipeline.config.profile_version == "community-model-cascade-v1"
    assert pipeline.config.mode == "cascade"
    assert pipeline.model_recognizer is not None
    assert pipeline.model_identity is not None
    assert pipeline.model_identity["model_protocol"] == "BIE73"
    assert pipeline.model_identity["rule_policy"] == COMMUNITY_CALIBRATED_RULE_SUPPRESSION
    assert isinstance(pipeline.model_identity["allowed_model_labels_sha256"], str)
    routes = pipeline.config.route_map
    assert routes["BANK_CARD_NUMBER"].rule_enabled is False
    assert all(route.model_threshold == 0.0 for route in routes.values())


def test_research_builder_refuses_a_loader_without_the_predecode_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    identity = SimpleNamespace(to_dict=lambda: {"manifest_sha256": "a" * 64})
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.verify_research_bie_artifact",
        lambda path: SimpleNamespace(root=Path(path).resolve(), identity=identity),
    )
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.load_local_project_bie_predictor",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer=object(),
            attention_mode="causal",
            allowed_project_labels=frozenset({"PERSON_NAME"}),
        ),
    )

    with pytest.raises(ResearchBieArtifactError, match="closed-8 mask"):
        build_research_bie_closed8_pipeline(model)


def test_research_builder_can_preserve_all_validated_rules_with_zero_model_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    identity = SimpleNamespace(to_dict=lambda: {"manifest_sha256": "a" * 64})
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.verify_research_bie_artifact",
        lambda path: SimpleNamespace(root=Path(path).resolve(), identity=identity),
    )
    predictor = SimpleNamespace(
        tokenizer=object(),
        attention_mode="causal",
        allowed_project_labels=frozenset(PROJECT_CLOSED8_LABELS),
    )
    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.load_local_project_bie_predictor",
        lambda *args, **kwargs: predictor,
    )

    class FakeRecognizer:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["predictor"] is predictor

        def analyze(self, text: str, entities: object) -> list[object]:
            del text, entities
            return []

    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.QwenPiiRecognizer",
        FakeRecognizer,
    )

    pipeline = build_research_bie_closed8_pipeline(
        model,
        calibration=CalibrationBundle(default_threshold=0.2),
        rule_policy=PRESERVE_ALL_VALIDATED_RULES,
    )

    assert pipeline.model_identity is not None
    assert pipeline.model_identity["rule_policy"] == PRESERVE_ALL_VALIDATED_RULES
    assert all(route.rule_enabled for route in pipeline.config.routes)
    assert all(route.model_threshold == 0.0 for route in pipeline.config.routes)
    assert all(
        route.validator_label is not None
        for route in pipeline.config.routes
        if route.category == "structured"
    )


def test_research_builder_rejects_unknown_rule_policy_before_artifact_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_verifier(path: object) -> object:
        nonlocal called
        del path
        called = True
        raise AssertionError("invalid policy must fail before model artifact access")

    monkeypatch.setattr(
        "pii_zh.cascade.research_bie_profile.verify_research_bie_artifact",
        forbidden_verifier,
    )

    with pytest.raises(ValueError, match="rule_policy is unsupported"):
        build_research_bie_closed8_pipeline(
            tmp_path / "does-not-exist",
            rule_policy="unknown",  # type: ignore[arg-type]
        )
    assert called is False
