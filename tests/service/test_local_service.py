from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from importlib.metadata import version as distribution_version

import pytest
from fastapi.testclient import TestClient

import pii_zh.service.app as service_app
from pii_zh import __version__
from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    LEGACY_SERVICE_PROFILE_VERSION,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    CascadeConfig,
    CascadeDetection,
    CascadePipeline,
    load_service_config,
)
from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
from pii_zh.full_bie73 import COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION
from pii_zh.rules.cn_common_v6 import CnCommonRulePackV6
from pii_zh.service.app import SERVICE_OUTPUT_SCHEMA_VERSION, create_app


def test_runtime_and_openapi_versions_match_distribution_metadata() -> None:
    assert __version__ == distribution_version("pii-zh-qwen")

    app = create_app()
    with TestClient(app):
        assert app.version == __version__


def test_default_factory_is_rules_only_and_analyze_never_returns_raw_value() -> None:
    synthetic_value = "demo@example.com"
    app = create_app()

    with TestClient(app) as client:
        health = client.get("/healthz")
        response = client.post("/v1/analyze", json={"text": synthetic_value})

    assert health.status_code == 200
    assert health.json()["mode"] == "rules-only"
    assert health.json()["profile_version"] == LEGACY_SERVICE_PROFILE_VERSION
    assert health.json()["model_enabled"] is False
    assert health.headers["cache-control"] == "no-store"
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == SERVICE_OUTPUT_SCHEMA_VERSION
    assert payload["mode"] == "rules-only"
    assert payload["detections"][0]["entity_type"] == "EMAIL_ADDRESS"
    assert synthetic_value not in json.dumps(payload, ensure_ascii=False)


def test_rules_only_http_factory_does_not_construct_the_full_bie73_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("rules-only must remain independent of BIE73 construction")

    monkeypatch.setattr(service_app, "build_full_bie73_service_pipeline", forbidden)

    with TestClient(create_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["mode"] == "rules-only"


def test_default_factory_preserves_the_frozen_legacy_profile() -> None:
    text = "车辆模型A12345已加载"
    app = create_app()

    assert app.state.pipeline.config.profile_version == LEGACY_SERVICE_PROFILE_VERSION
    assert app.state.pipeline.config.rule_source == "rule:cn_common"
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze",
            json={"text": text, "entities": ["CN_VEHICLE_LICENSE_PLATE"]},
        )

    assert response.status_code == 200
    detections = response.json()["detections"]
    assert len(detections) == 1
    assert (detections[0]["start"], detections[0]["end"]) == (3, 10)
    assert detections[0]["source"] == "rule:cn_common"
    assert text[3:10] == "型A12345"
    assert "型A12345" not in response.text


def test_service_profile_selection_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown service profile"):
        create_app(profile_version="c1-conservative-typo")

    pipeline = CascadePipeline(config=CascadeConfig(mode="rules-only"))
    with pytest.raises(ValueError, match="cannot be combined"):
        create_app(
            pipeline=pipeline,
            profile_version=SUCCESSOR_SERVICE_PROFILE_VERSION,
        )


def test_redact_replaces_detected_value_and_uses_no_store() -> None:
    synthetic_value = "demo@example.com"

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/redact",
            json={"text": f"测试邮箱 {synthetic_value}", "replacement": "<EMAIL>"},
        )

    assert response.status_code == 200
    assert response.json()["redacted_text"] == "测试邮箱 <EMAIL>"
    assert synthetic_value not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_long_unicode_text_preserves_http_offsets_and_redacts_from_original_span() -> None:
    synthetic_value = "demo@example.com"
    prefix = "段🙂Ａ\n" * 1024 + "邮箱 "
    text = prefix + synthetic_value + " 结束"
    start = len(prefix)

    with TestClient(create_app()) as client:
        analyze = client.post("/v1/analyze", json={"text": text})
        redact = client.post(
            "/v1/redact",
            json={"text": text, "replacement": "<EMAIL>"},
        )

    assert analyze.status_code == 200
    email = next(
        item for item in analyze.json()["detections"] if item["entity_type"] == "EMAIL_ADDRESS"
    )
    assert (email["start"], email["end"]) == (start, start + len(synthetic_value))
    assert text[email["start"] : email["end"]] == synthetic_value
    assert synthetic_value not in analyze.text
    assert redact.status_code == 200
    assert redact.json()["redacted_text"] == text[:start] + "<EMAIL>" + text[email["end"] :]
    assert synthetic_value not in redact.text


def test_body_text_and_schema_limits_fail_without_echoing_input() -> None:
    sensitive_value = "never-echo-this-sensitive-request"

    with TestClient(create_app(max_request_body_bytes=24)) as client:
        body_response = client.post("/v1/analyze", json={"text": sensitive_value})
    with TestClient(create_app(max_request_body_bytes=1024, max_text_chars=8)) as client:
        text_response = client.post("/v1/analyze", json={"text": sensitive_value})
        schema_response = client.post(
            "/v1/analyze",
            json={"text": "short", "unknown": sensitive_value},
        )

    assert body_response.status_code == 413
    assert body_response.json() == {"detail": "request body too large"}
    assert sensitive_value not in body_response.text
    assert text_response.status_code == 413
    assert text_response.json() == {"detail": "text too long"}
    assert sensitive_value not in text_response.text
    assert schema_response.status_code == 422
    assert sensitive_value not in schema_response.text


def test_pipeline_failure_and_timeout_are_generic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_value = "synthetic-sensitive-value"
    caplog.set_level(logging.DEBUG)

    class FailingPipeline(CascadePipeline):
        def detect(self, text: str, *, entities: object = None) -> list[object]:  # type: ignore[override]
            del entities
            raise RuntimeError(text)

    failing = FailingPipeline(config=CascadeConfig(mode="rules-only"))
    with TestClient(create_app(pipeline=failing)) as client:
        failure = client.post("/v1/analyze", json={"text": sensitive_value})
        redaction_failure = client.post("/v1/redact", json={"text": sensitive_value})

    assert failure.status_code == 500
    assert failure.json() == {"detail": "analysis failed"}
    assert sensitive_value not in failure.text
    assert sensitive_value not in caplog.text
    assert redaction_failure.status_code == 500
    assert redaction_failure.json() == {"detail": "redaction failed"}
    assert sensitive_value not in redaction_failure.text

    class SlowPipeline(CascadePipeline):
        def detect(self, text: str, *, entities: object = None) -> list[object]:  # type: ignore[override]
            del text, entities
            time.sleep(0.05)
            return []

    slow = SlowPipeline(config=CascadeConfig(mode="rules-only"))
    with TestClient(create_app(pipeline=slow, request_timeout_seconds=0.01)) as client:
        timeout = client.post("/v1/analyze", json={"text": sensitive_value})

    assert timeout.status_code == 504
    assert timeout.json() == {"detail": "analysis timed out"}
    assert sensitive_value not in timeout.text


def test_full_bie73_http_factory_uses_selected_defaults_and_explicit_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "full-bie73"
    model_dir.mkdir()
    calls: list[dict[str, object]] = []

    class EmptyModelRecognizer:
        def analyze(self, text: str, entities: object) -> list[object]:
            del text, entities
            return []

    def fake_factory(model_path, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"model_path": model_path, **kwargs})
        mode = kwargs["mode"]
        config = replace(
            load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode=mode),
            profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        )
        return CascadePipeline(
            config=config,
            rule_recognizer=CnCommonRulePackV6() if mode == "cascade" else None,
            model_recognizer=EmptyModelRecognizer(),
            validators=COMMUNITY_STRUCTURED_VALIDATORS,
            model_identity={
                "public_profile_version": COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
                "service_mode": mode,
                "model_scope": kwargs["scope"],
            },
        )

    monkeypatch.setattr(service_app, "build_full_bie73_service_pipeline", fake_factory)

    selected_app = create_app(
        profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        model_path=model_dir,
    )
    with TestClient(selected_app) as client:
        selected_health = client.get("/healthz")
        selected_analyze = client.post("/v1/analyze", json={"text": "合成文本"})

    assert selected_health.status_code == 200
    assert selected_health.json()["mode"] == "cascade"
    assert selected_health.json()["profile_version"] == (
        COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION
    )
    assert selected_health.json()["model_identity"]["service_mode"] == "cascade"
    assert selected_analyze.json()["model_identity"] == selected_health.json()["model_identity"]
    assert calls[-1] == {
        "model_path": model_dir,
        "scope": "open24",
        "mode": "cascade",
        "device": "cpu",
        "micro_batch_size": 16,
    }

    ablation_app = create_app(
        profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        mode="model-only",
        model_path=model_dir,
        scope="closed8",
    )
    with TestClient(ablation_app) as client:
        ablation_health = client.get("/healthz")
    assert ablation_health.json()["mode"] == "model-only"
    assert ablation_health.json()["model_identity"]["model_scope"] == "closed8"
    assert calls[-1]["mode"] == "model-only"
    assert calls[-1]["scope"] == "closed8"

    with pytest.raises(ValueError, match="does not accept calibration or threshold overrides"):
        create_app(
            profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
            model_path=model_dir,
            thresholds={"PERSON_NAME": 0.5},
        )


def test_full_bie73_cli_and_http_emit_the_same_detection_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pii_zh.cli as cli

    model_dir = tmp_path / "full-bie73"
    model_dir.mkdir()
    identity = {
        "public_profile_version": COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        "service_mode": "cascade",
        "model_scope": "open24",
        "public_profile_identity_sha256": "d" * 64,
    }
    config = replace(
        load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode="cascade"),
        profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
    )
    detection = CascadeDetection(
        start=0,
        end=2,
        entity_type="PERSON",
        score=0.9,
        source="model:pii-zh-24",
        sources=("model:pii-zh-24",),
        decision_process=("model:accepted",),
    )

    class SharedPipeline:
        model_identity = identity

        def __init__(self) -> None:
            self.config = config

        def detect(self, text: str, *, entities: object = None) -> list[CascadeDetection]:
            del text, entities
            return [detection]

        def redact(
            self,
            text: str,
            replacement: str = "<PII>",
            *,
            entities: object = None,
        ) -> str:
            del entities
            return replacement + text[2:]

    def fake_factory(*args: object, **kwargs: object) -> SharedPipeline:
        del args, kwargs
        return SharedPipeline()

    monkeypatch.setattr(cli, "build_full_bie73_service_pipeline", fake_factory)
    monkeypatch.setattr(service_app, "build_full_bie73_service_pipeline", fake_factory)

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
                "--model-path",
                str(model_dir),
                "--text",
                "测试文本",
            ]
        )
        == 0
    )
    cli_payload = json.loads(capsys.readouterr().out)

    app = create_app(
        profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        model_path=model_dir,
    )
    with TestClient(app) as client:
        api_payload = client.post("/v1/analyze", json={"text": "测试文本"}).json()

    for key in ("mode", "profile_version", "model_identity", "detections"):
        assert cli_payload[key] == api_payload[key]


def test_model_http_factory_requires_explicit_community_profile_and_local_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "community-model"
    model_dir.mkdir()
    calls: list[dict[str, object]] = []

    class EmptyModelRecognizer:
        def analyze(self, text: str, entities: object) -> list[object]:
            del text, entities
            return []

    def fake_factory(
        model_path,
        *,
        mode: str,
        device: str,
        micro_batch_size: int,
    ) -> CascadePipeline:
        calls.append(
            {
                "model_path": model_path,
                "mode": mode,
                "device": device,
                "micro_batch_size": micro_batch_size,
            }
        )
        config = load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode=mode)
        return CascadePipeline(
            config=config,
            rule_recognizer=CnCommonRulePackV6(),
            model_recognizer=EmptyModelRecognizer(),
            validators=COMMUNITY_STRUCTURED_VALIDATORS,
            model_identity={
                "schema_version": "pii-zh.community-model-identity.v1",
                "manifest_sha256": "a" * 64,
                "model_type": "qwen3_bi",
                "attention_mode": "full",
            },
        )

    monkeypatch.setattr(service_app, "build_community_model_service_pipeline", fake_factory)
    app = create_app(
        profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="cascade",
        model_path=model_dir,
        device="cpu",
        micro_batch_size=2,
    )

    with TestClient(app) as client:
        health = client.get("/healthz")
        analyze = client.post("/v1/analyze", json={"text": "无敏感信息"})

    assert health.status_code == 200
    assert health.json()["profile_version"] == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
    assert health.json()["mode"] == "cascade"
    assert health.json()["model_enabled"] is True
    assert health.json()["model_identity"]["model_type"] == "qwen3_bi"
    assert analyze.json()["profile_version"] == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
    assert analyze.json()["model_identity"] == health.json()["model_identity"]
    assert str(model_dir) not in health.text
    assert str(model_dir) not in analyze.text
    assert analyze.status_code == 200
    assert calls == [
        {
            "model_path": model_dir,
            "mode": "cascade",
            "device": "cpu",
            "micro_batch_size": 2,
        }
    ]

    cuda_app = create_app(
        profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="model-only",
        model_path=model_dir,
        device="cuda:0",
    )
    with TestClient(cuda_app) as client:
        cuda_health = client.get("/healthz")
    assert cuda_health.json()["max_concurrency"] == 1

    bounded_cuda_app = create_app(
        profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="model-only",
        model_path=model_dir,
        device="cuda:0",
        max_concurrency=3,
    )
    with TestClient(bounded_cuda_app) as client:
        bounded_health = client.get("/healthz")
    assert bounded_health.json()["max_concurrency"] == 3

    with pytest.raises(ValueError, match="requires the community model profile"):
        create_app(mode="cascade", model_path=model_dir)
    with pytest.raises(ValueError, match="model_path is required"):
        create_app(
            profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
            mode="cascade",
        )
    with pytest.raises(ValueError, match="only valid"):
        create_app(model_path=model_dir)


def test_http_factory_forwards_frozen_calibration_and_threshold_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "community-model"
    model_dir.mkdir()
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"default_threshold":0.5}', encoding="utf-8")
    observed: dict[str, object] = {}

    class EmptyModelRecognizer:
        def analyze(self, text: str, entities: object) -> list[object]:
            del text, entities
            return []

    def fake_factory(model_path, **kwargs):  # type: ignore[no-untyped-def]
        observed["model_path"] = model_path
        observed.update(kwargs)
        config = load_service_config(
            COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
            mode=kwargs["mode"],
        )
        return CascadePipeline(
            config=config,
            model_recognizer=EmptyModelRecognizer(),
            validators=COMMUNITY_STRUCTURED_VALIDATORS,
            model_identity={"manifest_sha256": "a" * 64},
        )

    monkeypatch.setattr(service_app, "build_community_model_service_pipeline", fake_factory)

    app = create_app(
        profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="model-only",
        model_path=model_dir,
        calibration=calibration,
        thresholds={"PERSON_NAME": 0.7},
    )
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert observed == {
        "model_path": model_dir,
        "mode": "model-only",
        "device": "cpu",
        "micro_batch_size": 16,
        "calibration": calibration,
        "thresholds": {"PERSON_NAME": 0.7},
    }
