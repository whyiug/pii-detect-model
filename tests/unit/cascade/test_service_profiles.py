from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pii_zh.calibration import CalibrationBundle
from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    DEFAULT_SERVICE_PROFILE_VERSION,
    LEGACY_SERVICE_PROFILE_VERSION,
    SERVICE_PROFILE_MODE_MATRIX,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    CascadeConfig,
    CascadePipeline,
    CommunityModelContractError,
    EntityRoute,
    build_community_model_service_pipeline,
    build_rules_only_service_pipeline,
    expected_core24_label2id,
    load_service_config,
    verify_community_model_artifact,
)
from pii_zh.rules import CnCommonRulePack
from pii_zh.rules.cn_common_v6 import CnCommonRulePackV6


def _plate_route(config: CascadeConfig) -> EntityRoute:
    return config.route_map["CN_VEHICLE_LICENSE_PLATE"]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _community_artifact(root: Path) -> Path:
    root.mkdir()
    labels = expected_core24_label2id()
    config = {
        "model_type": "qwen3_bi",
        "pii_attention_mode": "full",
        "architectures": ["Qwen3BiForTokenClassification"],
        "architecture_version": "qwen3_bi_token_cls_v1",
        "bi_attention_backend": "sdpa",
        "use_cache": False,
        "pii_training_status": "completed_candidate_not_benchmark_evaluated",
        "pii_taxonomy_version": "1.0.0",
        "pii_release_eligible": False,
        "num_labels": 49,
        "label2id": labels,
        "id2label": {str(index): label for label, index in labels.items()},
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe-test-weight")
    files = {
        name: _file_hash(root / name)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    weight_hashes = {"model.safetensors": files["model.safetensors"]}
    output_artifact = {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": ["model.safetensors"],
        "weights_combined_sha256": _canonical_hash(weight_hashes),
        "artifact_files_combined_sha256": _canonical_hash(files),
    }
    manifest = {
        "schema_version": 4,
        "manifest_type": "aiguard24_community_training_v1",
        "status": "completed",
        "release_eligible": False,
        "attention_mode": "full",
        "fine_tuning": "lora_merged",
        "base_source_id": "ZJUICSR/AIguard-pii-detection-fast",
        "source_revision": "677a5ebc1600fef61e8973cafd3026be322b3a73",
        "taxonomy_version": "1.0.0",
        "label2id": labels,
        "label_schema_sha256": _canonical_hash(labels),
        "initialization": {
            "strategy": "aiguard_bie_to_pii_zh_core24_bio_v1",
            "target_label_count": 49,
            "target_label2id": labels,
            "release_eligible": False,
            "attention_conversion": {
                "source_attention_mode": "causal",
                "target_attention_mode": "full",
                "conversion": "qwen3_to_qwen3_bi_shared_state_dict_v1",
                "attention_backend": "sdpa",
                "strict_state_dict_load": True,
                "newly_initialized_parameter_keys": [],
                "discarded_parameter_keys": [],
            },
        },
        "checkpoint_selection": {
            "selection_split": "validation",
            "selection_metric": "risk_weighted_score",
            "selected_checkpoint_id": "checkpoint-10",
            "selected_global_step": 10,
            "best_metric": 0.5,
            "final_validation_replay_metric": 0.5,
            "adapter_model_sha256": "a" * 64,
        },
        "output_artifact": output_artifact,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    (root / "training_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _rewrite_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _reseal_manifest(path: Path, mutation) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    mutation(manifest)
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _rewrite_json(path, manifest)


def test_default_service_profile_preserves_legacy_v1_binding() -> None:
    config = load_service_config()
    route = _plate_route(config)

    assert DEFAULT_SERVICE_PROFILE_VERSION == LEGACY_SERVICE_PROFILE_VERSION
    assert config.profile_version == LEGACY_SERVICE_PROFILE_VERSION
    assert config.rule_source == "rule:cn_common"
    assert route.rule_enabled is True
    assert route.model_enabled is False
    assert route.validator_label is None
    assert type(CascadePipeline(config=config).rule_recognizer) is CnCommonRulePack


def test_successor_profile_atomically_binds_v6_rules_validator_and_closed_model_route() -> None:
    config = load_service_config(SUCCESSOR_SERVICE_PROFILE_VERSION)
    route = _plate_route(config)

    assert config.profile_version == SUCCESSOR_SERVICE_PROFILE_VERSION
    assert config.mode == "rules-only"
    assert config.rule_source == "rule:cn_common_v6"
    assert route.rule_enabled is True
    assert route.model_enabled is False
    assert route.validator_label == "CN_VEHICLE_LICENSE_PLATE"
    assert route.model_requires_validator is False

    pipeline = build_rules_only_service_pipeline(SUCCESSOR_SERVICE_PROFILE_VERSION)
    assert isinstance(pipeline.rule_recognizer, CnCommonRulePackV6)
    assert "CN_VEHICLE_LICENSE_PLATE" in pipeline.validators


def test_unknown_service_profile_and_uninjected_successor_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown service profile"):
        load_service_config("c1-conservative-future")

    config = load_service_config(SUCCESSOR_SERVICE_PROFILE_VERSION)
    with pytest.raises(ValueError, match="requires unavailable validator"):
        CascadePipeline(config=config)


def test_community_profile_is_non_default_and_routes_all_24_labels() -> None:
    config = load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode="cascade")

    assert DEFAULT_SERVICE_PROFILE_VERSION == LEGACY_SERVICE_PROFILE_VERSION
    assert config.profile_version == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
    assert config.rule_source == "rule:cn_common_v6"
    assert config.model_source == "model:pii-zh-24"
    assert len(config.routes) == 24
    assert config.route_map["PERSON"].model_enabled is True
    assert config.route_map["CN_ID_CARD"].model_requires_validator is True
    assert all(route.model_enabled for route in config.routes)
    assert config.route_map["PASSPORT_NUMBER"].model_requires_validator is True
    assert config.route_map["CN_VEHICLE_LICENSE_PLATE"].model_requires_validator is True


def test_community_artifact_identity_is_exact_and_path_free(tmp_path: Path) -> None:
    model = _community_artifact(tmp_path / "model")

    verified = verify_community_model_artifact(model)

    identity = verified.identity.to_dict()
    assert verified.root == model.resolve()
    assert identity["model_type"] == "qwen3_bi"
    assert identity["attention_mode"] == "full"
    assert identity["label_count"] == 49
    assert str(model) not in json.dumps(identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3"),
        ("pii_attention_mode", "causal"),
        ("pii_attention_mode", "jpt"),
        ("model_type", "bert"),
        ("architectures", ["BertForTokenClassification"]),
    ],
)
def test_community_artifact_rejects_causal_jpt_macbert_and_non_bi_configs(
    tmp_path: Path, field: str, value: object
) -> None:
    model = _community_artifact(tmp_path / "model")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    _rewrite_json(config_path, config)

    with pytest.raises(CommunityModelContractError, match="qwen3_bi/full"):
        verify_community_model_artifact(model)


def test_community_artifact_rejects_non_exact_49_bio_and_wrong_manifest(
    tmp_path: Path,
) -> None:
    wrong_labels = _community_artifact(tmp_path / "wrong-labels")
    config_path = wrong_labels / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["label2id"].pop("I-SECRET")
    config["id2label"].pop("48")
    _rewrite_json(config_path, config)
    with pytest.raises(CommunityModelContractError, match="exact core-24 BIO"):
        verify_community_model_artifact(wrong_labels)

    wrong_manifest = _community_artifact(tmp_path / "wrong-manifest")
    _reseal_manifest(
        wrong_manifest / "training_manifest.json",
        lambda value: value.update({"manifest_type": "experimental_macbert_training"}),
    )
    with pytest.raises(CommunityModelContractError, match="completed full-attention"):
        verify_community_model_artifact(wrong_manifest)

    corrupt_hash = _community_artifact(tmp_path / "corrupt-hash")
    manifest_path = corrupt_hash / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = "0" * 64
    _rewrite_json(manifest_path, manifest)
    with pytest.raises(CommunityModelContractError, match="manifest hash"):
        verify_community_model_artifact(corrupt_hash)


def test_community_artifact_rejects_weight_tampering(tmp_path: Path) -> None:
    model = _community_artifact(tmp_path / "model")
    (model / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(CommunityModelContractError, match="do not match"):
        verify_community_model_artifact(model)


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        (LEGACY_SERVICE_PROFILE_VERSION, "model-only"),
        (LEGACY_SERVICE_PROFILE_VERSION, "cascade"),
        (SUCCESSOR_SERVICE_PROFILE_VERSION, "model-only"),
        (SUCCESSOR_SERVICE_PROFILE_VERSION, "cascade"),
        (COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, "rules-only"),
    ],
)
def test_service_profile_mode_matrix_rejects_every_illegal_pair(profile: str, mode: str) -> None:
    with pytest.raises(ValueError, match="requires"):
        load_service_config(profile, mode=mode)  # type: ignore[arg-type]


def test_service_profile_mode_matrix_is_immutable() -> None:
    with pytest.raises(TypeError):
        SERVICE_PROFILE_MODE_MATRIX[LEGACY_SERVICE_PROFILE_VERSION] = (  # type: ignore[index]
            "cascade",
        )


def test_community_factory_atomically_binds_v6_rules_and_local_model_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = _community_artifact(tmp_path / "local-model")
    calls: list[dict[str, object]] = []

    class _EmptyModelRecognizer:
        def analyze(self, text: str, entities: object) -> list[object]:
            if "PERSON" not in entities or "张三" not in text:  # type: ignore[operator]
                return []
            start = text.index("张三")
            return [
                {
                    "entity_type": "PERSON",
                    "start": start,
                    "end": start + 2,
                    "score": 1.0,
                }
            ]

    def fake_from_pretrained(
        cls: type[CascadePipeline],
        model_path,
        *,
        config: CascadeConfig,
        rule_recognizer,
        validators,
        device: str,
        micro_batch_size: int,
        local_files_only: bool,
        model_identity,
    ) -> CascadePipeline:
        del cls
        calls.append(
            {
                "model_path": model_path,
                "profile": config.profile_version,
                "device": device,
                "micro_batch_size": micro_batch_size,
                "local_files_only": local_files_only,
                "model_identity": model_identity,
            }
        )
        return CascadePipeline(
            config=config,
            rule_recognizer=rule_recognizer,
            model_recognizer=_EmptyModelRecognizer(),
            validators=validators,
            model_identity=model_identity,
        )

    monkeypatch.setattr(
        CascadePipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    pipeline = build_community_model_service_pipeline(
        model,
        mode="cascade",
        device="cpu",
        micro_batch_size=2,
    )

    assert isinstance(pipeline.rule_recognizer, CnCommonRulePackV6)
    assert pipeline.config.profile_version == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
    assert "CN_VEHICLE_LICENSE_PLATE" in pipeline.validators
    detections = pipeline.detect("联系人张三，邮箱 demo@example.com")
    assert {item.entity_type for item in detections} >= {"PERSON", "EMAIL_ADDRESS"}
    assert next(item for item in detections if item.entity_type == "PERSON").source == (
        "model:pii-zh-24"
    )
    assert calls == [
        {
            "model_path": model,
            "profile": COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
            "device": "cpu",
            "micro_batch_size": 2,
            "local_files_only": True,
            "model_identity": pipeline.model_identity,
        }
    ]

    with pytest.raises(ValueError, match="model-only or cascade"):
        build_community_model_service_pipeline(model, mode="rules-only")


def test_community_builder_normalizes_explicit_calibration_and_threshold_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _community_artifact(tmp_path / "local-model")
    observed: dict[str, object] = {}

    def fake_from_pretrained(cls, model_path, **kwargs):  # type: ignore[no-untyped-def]
        del cls
        observed["model_path"] = model_path
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(
        CascadePipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    dtype_marker = object()

    build_community_model_service_pipeline(
        model,
        mode="model-only",
        dtype=dtype_marker,
        calibration={
            "global_temperature": 1.25,
            "entity_temperatures": {"PERSON_NAME": 1.5},
            "entity_thresholds": {"PERSON_NAME": 0.70},
            "default_threshold": 0.55,
        },
        thresholds={"ADDRESS": 0.65},
    )

    assert observed["model_path"] == model.resolve()
    assert observed["dtype"] is dtype_marker
    config = observed["config"]
    assert isinstance(config, CascadeConfig)
    assert config.mode == "model-only"
    assert all(route.model_threshold == 0.0 for route in config.routes)
    assert all(route.short_span_max_chars == 0 for route in config.routes)
    calibration = observed["calibration"]
    assert isinstance(calibration, CalibrationBundle)
    assert calibration.entity_temperatures == {"PERSON": 1.5}
    assert calibration.entity_thresholds == {"PERSON": 0.70}
    assert observed["thresholds"] == {"CN_ADDRESS": 0.65}
    identity = observed["model_identity"]
    assert isinstance(identity, dict)
    assert isinstance(identity["calibration_sha256"], str)
    assert isinstance(identity["thresholds_sha256"], str)
    assert str(model) not in json.dumps(identity)


@pytest.mark.parametrize(
    "thresholds",
    [
        {"NOT_CORE24": 0.5},
        {"PERSON": float("nan")},
        {"PERSON": 1.1},
        {"PERSON": True},
        {"PERSON": 0.5, "PERSON_NAME": 0.5},
    ],
)
def test_community_builder_rejects_ambiguous_or_invalid_thresholds(
    tmp_path: Path,
    thresholds: dict[str, object],
) -> None:
    model = _community_artifact(tmp_path / "local-model")

    with pytest.raises((TypeError, ValueError)):
        build_community_model_service_pipeline(
            model,
            mode="model-only",
            thresholds=thresholds,  # type: ignore[arg-type]
        )
