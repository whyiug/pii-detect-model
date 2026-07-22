from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from pii_zh.presidio import cluener_primary as primary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cluener"
    root.mkdir()
    receipt = dict(primary.CLUENER_PRIMARY_CONVERSION_RECEIPT_EXPECTED)
    for name in primary.CLUENER_PRIMARY_REQUIRED_FILES:
        path = root / name
        if name == "conversion_receipt.json":
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        else:
            path.write_bytes(f"fixture::{name}".encode())
    required = {name: _sha256(root / name) for name in primary.CLUENER_PRIMARY_REQUIRED_FILES}
    monkeypatch.setattr(primary, "CLUENER_PRIMARY_REQUIRED_FILES", MappingProxyType(required))
    return root


def test_verifier_returns_flat_path_free_identity_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _artifact_fixture(tmp_path, monkeypatch)
    (root / ".cache").mkdir()

    verified = primary.verify_cluener_primary_artifact(root)

    assert verified.root == root.resolve()
    identity = verified.model_identity
    assert identity["source_model_id"] == primary.CLUENER_PRIMARY_MODEL_ID
    assert identity["source_revision"] == primary.CLUENER_PRIMARY_MODEL_REVISION
    assert identity["presidio_analyzer_version"] == "2.2.363"
    assert identity["profile_version"] == "native-presidio-cluener-cn-common-v5-v1"
    assert identity["rule_pack_id"] == "cn_common_v5"
    assert identity["model_safetensors_sha256"] == _sha256(root / "model.safetensors")
    assert identity["local_files_only"] is True
    assert identity["safetensors_only"] is True
    assert identity["trust_remote_code"] is False
    assert identity["weights_bundled"] is False
    assert all(isinstance(value, (str, int, bool, type(None))) for value in identity.values())
    assert str(root) not in json.dumps(dict(identity), sort_keys=True)
    with pytest.raises(TypeError):
        identity["changed"] = True  # type: ignore[index]

    (root / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(primary.CluenerPrimaryContractError, match="pinned revision"):
        primary.verify_cluener_primary_artifact(root)


def test_verifier_rejects_extra_top_level_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _artifact_fixture(tmp_path, monkeypatch)
    (root / "pytorch_model.bin").write_bytes(b"unsafe")

    with pytest.raises(primary.CluenerPrimaryContractError, match="inventory"):
        primary.verify_cluener_primary_artifact(root)


def test_verifier_accepts_public_preparation_receipt_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cluener-public-prepare"
    root.mkdir()
    receipt = dict(primary.CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED)
    for name in primary.CLUENER_PRIMARY_REQUIRED_FILES:
        path = root / name
        if name == "conversion_receipt.json":
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        else:
            path.write_bytes(f"fixture::{name}".encode())
    required = {name: _sha256(root / name) for name in primary.CLUENER_PRIMARY_REQUIRED_FILES}
    monkeypatch.setattr(primary, "CLUENER_PRIMARY_REQUIRED_FILES", MappingProxyType(required))

    verified = primary.verify_cluener_primary_artifact(root)

    assert verified.model_identity["conversion_receipt_sha256"] == required[
        "conversion_receipt.json"
    ]


def test_native_framework_loader_reuses_closed_bios_and_cn_common_v5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pii_zh.inference.bios_token_classifier as bios_module
    import pii_zh.presidio.bio_token_recognizer as bio_recognizer_module
    import pii_zh.presidio.cn_common_recognizer as common_recognizer_module
    import pii_zh.presidio.native_framework as native_module
    import pii_zh.rules as rules_module

    observed: dict[str, Any] = {}
    predictor = SimpleNamespace(tokenizer=object())

    def fake_loader(path: Path, **kwargs: object) -> object:
        observed["loader_path"] = path
        observed["loader_kwargs"] = kwargs
        return predictor

    class FakeBioRecognizer:
        def __init__(self, **kwargs: object) -> None:
            observed["ner_kwargs"] = kwargs

    class FakeRulePackV5:
        pass

    class FakeCnCommonRecognizer:
        def __init__(self, **kwargs: object) -> None:
            observed["rule_kwargs"] = kwargs

    class FakeNativeFramework:
        def __init__(self, **kwargs: object) -> None:
            observed["framework_kwargs"] = kwargs

    monkeypatch.setattr(bios_module, "load_closed_bios_token_classifier", fake_loader)
    monkeypatch.setattr(
        bio_recognizer_module,
        "BioTokenClassifierRecognizer",
        FakeBioRecognizer,
    )
    monkeypatch.setattr(common_recognizer_module, "CnCommonRecognizer", FakeCnCommonRecognizer)
    monkeypatch.setattr(native_module, "NativePresidioFrameworkRecognizer", FakeNativeFramework)
    monkeypatch.setattr(rules_module, "CnCommonRulePack", FakeRulePackV5)
    artifact = primary.VerifiedCluenerPrimaryArtifact(
        root=tmp_path,
        model_identity=MappingProxyType({"fixture": True}),
        _snapshot=(),
    )

    framework = primary._load_native_framework(artifact, device="cpu", micro_batch_size=16)

    assert isinstance(framework, FakeNativeFramework)
    assert observed["loader_path"] == tmp_path
    loader_kwargs = observed["loader_kwargs"]
    assert loader_kwargs == {
        "source_to_target": {"address": "CN_ADDRESS", "name": "PERSON"},
        "protocol_id": primary.CLUENER_PRIMARY_PROTOCOL_ID,
        "device": "cpu",
        "dtype": None,
        "micro_batch_size": 16,
    }
    ner_kwargs = observed["ner_kwargs"]
    assert ner_kwargs["predictor"] is predictor
    assert ner_kwargs["supported_entities"] == ["CN_ADDRESS", "PERSON"]
    assert ner_kwargs["default_threshold"] == 0.0
    assert ner_kwargs["max_tokens"] == 512
    assert ner_kwargs["stride_fraction"] == 0.25
    assert ner_kwargs["deduplication_policy"] == "exact_only_v1"
    rule_kwargs = observed["rule_kwargs"]
    assert isinstance(rule_kwargs["rule_pack"], FakeRulePackV5)
    assert rule_kwargs["name"] == "CnCommonV5Open24ComparatorRecognizer"
    assert rule_kwargs["version"] == "5.0.0"
    assert observed["framework_kwargs"] == {
        "ner_recognizer": observed["framework_kwargs"]["ner_recognizer"],
        "rule_recognizer": observed["framework_kwargs"]["rule_recognizer"],
    }


def test_public_native_builder_attaches_verified_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _artifact_fixture(tmp_path, monkeypatch)
    analyzer_type = type("AnalyzerEngine", (), {"__module__": "presidio_analyzer"})

    class FakeFramework:
        def __init__(self) -> None:
            self.engine = analyzer_type()

        @staticmethod
        def runtime_identity() -> dict[str, str]:
            return {
                "framework": "presidio-analyzer",
                "framework_version": "2.2.363",
            }

        @staticmethod
        def analyze(text: str, entities: object) -> list[object]:
            return []

    seen: dict[str, object] = {}

    def fake_load(
        artifact: primary.VerifiedCluenerPrimaryArtifact,
        *,
        device: str,
        micro_batch_size: int,
    ) -> object:
        seen.update(artifact=artifact, device=device, micro_batch_size=micro_batch_size)
        return FakeFramework()

    monkeypatch.setattr(primary, "_require_exact_presidio_version", lambda: "2.2.363")
    monkeypatch.setattr(primary, "_load_native_framework", fake_load)

    framework = primary.build_cluener_primary_native_framework(
        root,
        device="cpu",
        micro_batch_size=7,
    )

    assert seen["device"] == "cpu"
    assert seen["micro_batch_size"] == 7
    assert framework.model_identity["source_model_id"] == primary.CLUENER_PRIMARY_MODEL_ID


class _FakeFramework:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def analyze(self, text: str, entities: tuple[str, ...]) -> list[object]:
        self.calls.append((text, entities))
        results: list[object] = []
        if "PERSON" in entities and "张三" in text:
            start = text.index("张三")
            results.append(
                SimpleNamespace(entity_type="PERSON", start=start, end=start + 2, score=0.8)
            )
        if "CN_ADDRESS" in entities and "海淀" in text:
            start = text.index("海淀")
            results.append(
                SimpleNamespace(entity_type="CN_ADDRESS", start=start, end=start + 2, score=0.7)
            )
        return list(reversed(results))


def test_primary_pipeline_normalizes_closed8_requests_and_preserves_native_fields() -> None:
    framework = _FakeFramework()
    pipeline = primary.CluenerPrimaryPipeline(
        native_framework=framework,
        model_identity={"source_model_id": primary.CLUENER_PRIMARY_MODEL_ID},
    )

    detections = pipeline.detect("张三住海淀", entities=["ADDRESS", "PERSON_NAME"])

    assert framework.calls == [("张三住海淀", ("CN_ADDRESS", "PERSON"))]
    assert [(item.start, item.end, item.entity_type, item.score) for item in detections] == [
        (0, 2, "PERSON", 0.8),
        (3, 5, "CN_ADDRESS", 0.7),
    ]
    assert all(item.source == primary.CLUENER_PRIMARY_SOURCE for item in detections)
    assert all(item.sources == (primary.CLUENER_PRIMARY_SOURCE,) for item in detections)
    assert all(item.decision_process == ("primary:native_presidio",) for item in detections)
    assert pipeline.model_identity == {"source_model_id": primary.CLUENER_PRIMARY_MODEL_ID}


def test_primary_pipeline_config_exposes_exact_primary_only_closed8_routes() -> None:
    pipeline = primary.CluenerPrimaryPipeline(
        native_framework=_FakeFramework(),
        model_identity={"fixture": True},
    )

    assert pipeline.config.profile_version == primary.CLUENER_PRIMARY_PROFILE_VERSION
    assert pipeline.config.profile_version == "native-presidio-cluener-cn-common-v5-v1"
    assert pipeline.config.mode == "rules-only"
    assert pipeline.config.rule_source == "presidio:cluener-cn-common-v5-primary"
    assert tuple(route.entity_type for route in pipeline.config.routes) == (
        primary.CLUENER_PRIMARY_CLOSED8_ENTITIES
    )
    assert all(route.rule_enabled for route in pipeline.config.routes)
    assert all(not route.model_enabled for route in pipeline.config.routes)
    assert all(not route.model_requires_validator for route in pipeline.config.routes)


def test_primary_pipeline_batch_default_and_empty_entity_request() -> None:
    framework = _FakeFramework()
    pipeline = primary.CluenerPrimaryPipeline(
        native_framework=framework,
        model_identity={"fixture": True},
    )

    batched = pipeline.detect_batch(["张三", "海淀"])
    empty = pipeline.detect("张三", entities=[])

    assert [[item.entity_type for item in row] for row in batched] == [
        ["PERSON"],
        ["CN_ADDRESS"],
    ]
    assert all(call[1] == primary.CLUENER_PRIMARY_CLOSED8_ENTITIES for call in framework.calls)
    assert empty == []
    assert len(framework.calls) == 2


@pytest.mark.parametrize("entity", ["SECRET", "ORGANIZATION", "unknown"])
def test_primary_pipeline_rejects_non_closed8_request(entity: str) -> None:
    pipeline = primary.CluenerPrimaryPipeline(
        native_framework=_FakeFramework(),
        model_identity={"fixture": True},
    )

    with pytest.raises(ValueError, match="canonical closed8"):
        pipeline.detect("fixture", entities=[entity])
