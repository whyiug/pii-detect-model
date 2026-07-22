from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import pii_zh.cli as cli
from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    CascadeConfig,
)
from pii_zh.full_bie73 import COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    assert isinstance(value, dict)
    return value


def test_detect_text_emits_raw_text_free_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    synthetic_value = "demo@example.com"

    assert cli.main(["detect", "--text", f"测试邮箱 {synthetic_value}"]) == 0

    payload = _stdout_json(capsys)
    assert payload["schema_version"] == cli.CLI_OUTPUT_SCHEMA_VERSION
    assert payload["mode"] == "rules-only"
    assert payload["profile_version"] == cli.DEFAULT_SERVICE_PROFILE_VERSION
    assert payload["model_identity"] is None
    assert payload["detections"][0]["entity_type"] == "EMAIL_ADDRESS"
    assert "text" not in payload
    assert "value" not in payload["detections"][0]
    assert synthetic_value not in json.dumps(payload, ensure_ascii=False)


def test_rules_only_cli_does_not_construct_the_full_bie73_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("rules-only must remain independent of BIE73 construction")

    monkeypatch.setattr(cli, "build_full_bie73_service_pipeline", forbidden)

    assert cli.main(["detect", "--text", "demo@example.com"]) == 0
    assert _stdout_json(capsys)["mode"] == "rules-only"


def test_successor_profile_detects_strict_plates_and_rejects_legacy_fallbacks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    entity = "CN_VEHICLE_LICENSE_PLATE"
    valid = "沪AD12345"
    valid_text = f"新能源车牌{valid}已登记"

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                SUCCESSOR_SERVICE_PROFILE_VERSION,
                "--entity",
                entity,
                "--text",
                valid_text,
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert len(payload["detections"]) == 1
    detection = payload["detections"][0]
    assert (detection["start"], detection["end"]) == (
        valid_text.index(valid),
        valid_text.index(valid) + len(valid),
    )
    assert detection["source"] == "rule:cn_common_v6"
    assert "validator:passed" in detection["decision_process"]
    assert valid not in captured.out

    broad_false_positive = "车辆模型A12345已加载"
    assert (
        cli.main(
            [
                "detect",
                "--profile",
                SUCCESSOR_SERVICE_PROFILE_VERSION,
                "--entity",
                entity,
                "--text",
                broad_false_positive,
            ]
        )
        == 0
    )
    rejected = _stdout_json(capsys)
    assert rejected["detections"] == []


def test_default_cli_profile_preserves_frozen_v1_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = "车辆模型A12345已加载"

    assert (
        cli.main(
            [
                "detect",
                "--entity",
                "CN_VEHICLE_LICENSE_PLATE",
                "--text",
                text,
            ]
        )
        == 0
    )

    payload = _stdout_json(capsys)
    assert len(payload["detections"]) == 1
    detection = payload["detections"][0]
    assert (detection["start"], detection["end"]) == (3, 10)
    assert detection["source"] == "rule:cn_common"
    assert text[3:10] == "型A12345"


def test_unknown_cli_profile_choice_fails_before_processing_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive_input = "never-process-this-profile-input"

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "detect",
                "--profile",
                "c1-conservative-typo",
                "--text",
                sensitive_input,
            ]
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid choice" in captured.err
    assert sensitive_input not in captured.err


def test_detect_reads_stdin_when_text_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", StringIO("测试地址 203.0.113.7"))

    assert cli.main(["detect", "--entity", "IP_ADDRESS"]) == 0

    payload = _stdout_json(capsys)
    assert [item["entity_type"] for item in payload["detections"]] == ["IP_ADDRESS"]


def test_detect_reads_one_regular_utf8_text_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    synthetic_value = "demo@example.com"
    input_file = tmp_path / "input.txt"
    input_file.write_text(f"测试邮箱 {synthetic_value}", encoding="utf-8")

    assert cli.main(["detect", "--input-file", str(input_file)]) == 0

    payload = _stdout_json(capsys)
    assert payload["detections"][0]["entity_type"] == "EMAIL_ADDRESS"
    assert synthetic_value not in json.dumps(payload, ensure_ascii=False)


def test_input_sources_are_mutually_exclusive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "input.txt"
    input_file.write_text("safe fixture", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "detect",
                "--text",
                "inline",
                "--input-file",
                str(input_file),
            ]
        )

    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_detect_jsonl_file_emits_ordered_raw_text_free_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = ["demo@example.com", "203.0.113.7"]
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        "".join(json.dumps({"text": value}, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )

    assert cli.main(["detect", "--jsonl", "--input-file", str(input_file)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    payloads = [json.loads(line) for line in captured.out.splitlines()]
    assert [payload["record_index"] for payload in payloads] == [0, 1]
    assert [payload["detections"][0]["entity_type"] for payload in payloads] == [
        "EMAIL_ADDRESS",
        "IP_ADDRESS",
    ]
    assert all(value not in captured.out for value in values)


def test_redact_jsonl_stdin_emits_redacted_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = ["demo@example.com", "203.0.113.7"]
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO("".join(json.dumps({"text": value}) + "\n" for value in values)),
    )

    assert cli.main(["redact", "--jsonl", "--replacement", "<PII>"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    payloads = [json.loads(line) for line in captured.out.splitlines()]
    assert [payload["redacted_text"] for payload in payloads] == ["<PII>", "<PII>"]
    assert all(value not in captured.out for value in values)


@pytest.mark.parametrize(
    "content, expected_error",
    [
        ('{"text":"safe"}\nnot-json-sensitive-value\n', "record 2 is invalid JSON"),
        ('{"text":"safe","secret":"never-echo-this"}\n', "containing only text"),
        ('{"text":"safe"}\n\n', "record 2 must not be blank"),
    ],
)
def test_invalid_jsonl_fails_atomically_without_echoing_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
    expected_error: str,
) -> None:
    monkeypatch.setattr(sys, "stdin", StringIO(content))

    assert cli.main(["detect", "--jsonl"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert expected_error in captured.err
    assert "not-json-sensitive-value" not in captured.err
    assert "never-echo-this" not in captured.err


def test_invalid_utf8_file_and_pretty_jsonl_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "invalid.txt"
    input_file.write_bytes(b"\xffsecret")

    assert cli.main(["detect", "--input-file", str(input_file)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "valid UTF-8" in captured.err
    assert "secret" not in captured.err

    monkeypatch.setattr(sys, "stdin", StringIO('{"text":"safe"}\n'))
    assert cli.main(["detect", "--jsonl", "--pretty"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--pretty is not supported" in captured.err


def test_redact_replaces_detected_value_and_returns_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    synthetic_value = "demo@example.com"

    assert (
        cli.main(
            [
                "redact",
                "--text",
                f"测试邮箱 {synthetic_value}",
                "--replacement",
                "<EMAIL>",
            ]
        )
        == 0
    )

    payload = _stdout_json(capsys)
    assert payload["redacted_text"] == "测试邮箱 <EMAIL>"
    assert synthetic_value not in json.dumps(payload, ensure_ascii=False)


def test_model_modes_require_an_existing_local_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "organization" / "hub-model"

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "detect",
                "--mode",
                "cascade",
                "--model-path",
                str(missing),
                "--text",
                "测试文本",
            ]
        )

    assert error.value.code == 2
    assert "existing local directory" in capsys.readouterr().err


def test_legacy_model_mode_is_rejected_before_the_generic_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: list[dict[str, object]] = []

    class EmptyPipeline:
        def detect(self, text: str, *, entities: object = None) -> list[object]:
            del text, entities
            return []

    def fake_from_pretrained(
        cls: type[object],
        model_path: Path,
        *,
        config: CascadeConfig,
        device: str,
        local_files_only: bool,
    ) -> EmptyPipeline:
        del cls
        calls.append(
            {
                "model_path": model_path,
                "mode": config.mode,
                "device": device,
                "local_files_only": local_files_only,
            }
        )
        return EmptyPipeline()

    monkeypatch.setattr(
        cli.CascadePipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    assert (
        cli.main(
            [
                "detect",
                "--mode",
                "model-only",
                "--model-path",
                str(model_dir),
                "--device",
                "cpu",
                "--text",
                "测试文本",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "only support rules-only mode" in captured.err
    assert calls == []


def test_missing_model_path_fails_without_echoing_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    synthetic_input = "never-echo-this-input"

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
                "--mode",
                "cascade",
                "--text",
                synthetic_input,
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "--model-path is required" in captured.err
    assert synthetic_input not in captured.err
    assert captured.out == ""


def test_full_bie73_cli_uses_selected_defaults_and_explicit_service_ablation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "full-bie73"
    model_dir.mkdir()
    calls: list[dict[str, object]] = []

    class EmptyPipeline:
        def __init__(self, *, mode: str, scope: str) -> None:
            self.model_identity = {
                "public_profile_version": COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
                "service_mode": mode,
                "model_scope": scope,
            }

        def detect(self, text: str, *, entities: object = None) -> list[object]:
            del text, entities
            return []

    def fake_factory(model_path: Path, **kwargs: object) -> EmptyPipeline:
        calls.append({"model_path": model_path, **kwargs})
        return EmptyPipeline(mode=str(kwargs["mode"]), scope=str(kwargs["scope"]))

    monkeypatch.setattr(cli, "build_full_bie73_service_pipeline", fake_factory)

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
                "--model-path",
                str(model_dir),
                "--text",
                "合成文本",
            ]
        )
        == 0
    )
    selected = _stdout_json(capsys)
    assert selected["mode"] == "cascade"
    assert selected["profile_version"] == COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION
    assert selected["model_identity"]["service_mode"] == "cascade"
    assert calls[-1] == {
        "model_path": model_dir.resolve(),
        "scope": "open24",
        "mode": "cascade",
        "device": "cpu",
        "micro_batch_size": 16,
    }

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
                "--mode",
                "model-only",
                "--scope",
                "closed8",
                "--model-path",
                str(model_dir),
                "--text",
                "合成文本",
            ]
        )
        == 0
    )
    ablation = _stdout_json(capsys)
    assert ablation["mode"] == "model-only"
    assert ablation["model_identity"] == {
        "public_profile_version": COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        "service_mode": "model-only",
        "model_scope": "closed8",
    }
    assert calls[-1]["scope"] == "closed8"
    assert calls[-1]["mode"] == "model-only"


def test_full_bie73_cli_rejects_internal_threshold_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "full-bie73"
    model_dir.mkdir()
    calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "build_full_bie73_service_pipeline",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
                "--model-path",
                str(model_dir),
                "--threshold",
                "PERSON_NAME=0.5",
                "--text",
                "合成文本",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "does not accept calibration or threshold overrides" in captured.err
    assert captured.out == ""
    assert calls == []


def test_community_profile_uses_explicit_local_model_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "community-model"
    model_dir.mkdir()
    calls: list[dict[str, object]] = []

    class EmptyPipeline:
        model_identity = {
            "schema_version": "pii-zh.community-model-identity.v1",
            "manifest_sha256": "a" * 64,
            "model_type": "qwen3_bi",
            "attention_mode": "full",
        }

        def detect(self, text: str, *, entities: object = None) -> list[object]:
            del text, entities
            return []

    def fake_factory(
        model_path: Path,
        *,
        mode: str,
        device: str,
        micro_batch_size: int,
    ) -> EmptyPipeline:
        calls.append(
            {
                "model_path": model_path,
                "mode": mode,
                "device": device,
                "micro_batch_size": micro_batch_size,
            }
        )
        return EmptyPipeline()

    monkeypatch.setattr(cli, "build_community_model_service_pipeline", fake_factory)

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
                "--mode",
                "cascade",
                "--model-path",
                str(model_dir),
                "--text",
                "本地合成文本",
            ]
        )
        == 0
    )

    payload = _stdout_json(capsys)
    assert payload["detections"] == []
    assert payload["profile_version"] == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
    assert payload["model_identity"] == EmptyPipeline.model_identity
    assert str(model_dir) not in json.dumps(payload)
    assert calls == [
        {
            "model_path": model_dir.resolve(),
            "mode": "cascade",
            "device": "cpu",
            "micro_batch_size": 16,
        }
    ]


def test_community_cli_forwards_explicit_calibration_and_normalized_threshold_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "community-model"
    model_dir.mkdir()
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"default_threshold":0.5}', encoding="utf-8")
    calls: list[dict[str, object]] = []

    class EmptyPipeline:
        model_identity = {"manifest_sha256": "a" * 64}

        def detect(self, text: str, *, entities: object = None) -> list[object]:
            del text, entities
            return []

    def fake_factory(model_path: Path, **kwargs: object) -> EmptyPipeline:
        calls.append({"model_path": model_path, **kwargs})
        return EmptyPipeline()

    monkeypatch.setattr(cli, "build_community_model_service_pipeline", fake_factory)

    assert (
        cli.main(
            [
                "detect",
                "--profile",
                COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
                "--mode",
                "model-only",
                "--model-path",
                str(model_dir),
                "--calibration",
                str(calibration),
                "--threshold",
                "PERSON_NAME=0.70",
                "--threshold",
                "ADDRESS=0.65",
                "--text",
                "合成文本",
            ]
        )
        == 0
    )

    payload = _stdout_json(capsys)
    assert payload["model_identity"] == EmptyPipeline.model_identity
    assert calls == [
        {
            "model_path": model_dir.resolve(),
            "mode": "model-only",
            "device": "cpu",
            "micro_batch_size": 16,
            "calibration": calibration.resolve(),
            "thresholds": {"PERSON_NAME": 0.70, "ADDRESS": 0.65},
        }
    ]


def test_serve_is_loopback_only_and_passes_prebuilt_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pii_zh.service as service

    model_dir = tmp_path / "community-model"
    model_dir.mkdir()
    pipeline = object()
    run_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "build_community_model_service_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    monkeypatch.setattr(
        service,
        "run",
        lambda **kwargs: run_calls.append(kwargs),
    )

    assert (
        cli.main(
            [
                "serve",
                "--profile",
                COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
                "--mode",
                "cascade",
                "--model-path",
                str(model_dir),
                "--host",
                "127.0.0.1",
                "--port",
                "18080",
                "--micro-batch-size",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert run_calls == [
        {
            "pipeline": pipeline,
            "host": "127.0.0.1",
            "port": 18080,
            "max_concurrency": 4,
        }
    ]

    run_calls.clear()
    assert (
        cli.main(
            [
                "serve",
                "--profile",
                COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
                "--mode",
                "model-only",
                "--model-path",
                str(model_dir),
                "--device",
                "cuda:0",
                "--max-concurrency",
                "3",
            ]
        )
        == 0
    )
    assert run_calls[0]["max_concurrency"] == 3

    run_calls.clear()
    assert (
        cli.main(
            [
                "serve",
                "--profile",
                COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
                "--mode",
                "model-only",
                "--model-path",
                str(model_dir),
                "--device",
                "cuda:0",
            ]
        )
        == 0
    )
    assert run_calls[0]["max_concurrency"] == 1

    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--host", "0.0.0.0"])
    assert error.value.code == 2
    assert "only binds to a loopback host" in capsys.readouterr().err
