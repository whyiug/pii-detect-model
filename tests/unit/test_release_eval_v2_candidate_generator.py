from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import generate_release_eval_v2_candidate_predictions as generator


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "input.jsonl"
    source.write_text('{"fixture":"synthetic-only"}\n', encoding="utf-8")
    return model, source


def _fake_generate(_args: Any, *, labels: Any, temporary: Path) -> int:
    assert labels
    temporary.write_text('{"doc_id":"synthetic-doc","spans":[]}\n', encoding="utf-8")
    return 1


def _fake_receipt(**kwargs: Any) -> dict[str, Any]:
    assert kwargs["runtime"].scope in {"open24", "closed8"}
    return {
        "output": {"file_sha256": "a" * 64},
        "receipt_sha256": "b" * 64,
    }


def test_model_raw_generator_is_no_clobber_and_emits_aggregate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, source = _fixture(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(generator, "_generate_model_raw", _fake_generate)
    monkeypatch.setattr(generator, "build_release_eval_v2_generation_receipt", _fake_receipt)

    assert (
        generator.main(
            [
                "--mode",
                "model-raw",
                "--target-split",
                "fixture",
                "--model",
                str(model),
                "--input",
                str(source),
                "--predictions",
                str(predictions),
                "--receipt",
                str(receipt),
            ]
        )
        == 0
    )
    assert predictions.stat().st_mode & 0o777 == 0o444
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert "synthetic-doc" not in receipt.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        generator.main(
            [
                "--mode",
                "model-raw",
                "--target-split",
                "fixture",
                "--model",
                str(model),
                "--input",
                str(source),
                "--predictions",
                str(predictions),
                "--receipt",
                str(receipt),
            ]
        )


def test_generator_rejects_leaf_input_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, source = _fixture(tmp_path)
    linked_source = tmp_path / "linked-input.jsonl"
    linked_source.symlink_to(source)
    monkeypatch.setattr(generator, "_generate_model_raw", _fake_generate)
    monkeypatch.setattr(generator, "build_release_eval_v2_generation_receipt", _fake_receipt)

    with pytest.raises(SystemExit):
        generator.main(
            [
                "--mode",
                "model-raw",
                "--target-split",
                "fixture",
                "--model",
                str(model),
                "--input",
                str(linked_source),
                "--predictions",
                str(tmp_path / "predictions.jsonl"),
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )


def test_cascade_generator_requires_calibration_and_uses_community_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, source = _fixture(tmp_path)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    predictions = tmp_path / "cascade.jsonl"
    receipt = tmp_path / "cascade-receipt.json"
    calls: list[str] = []

    def fake_service(args: Any, *, labels: Any, temporary: Path) -> int:
        calls.append(args.mode)
        return _fake_generate(args, labels=labels, temporary=temporary)

    monkeypatch.setattr(generator, "_generate_service", fake_service)
    monkeypatch.setattr(generator, "build_release_eval_v2_generation_receipt", _fake_receipt)

    assert (
        generator.main(
            [
                "--mode",
                "cascade",
                "--target-split",
                "fixture",
                "--model",
                str(model),
                "--input",
                str(source),
                "--predictions",
                str(predictions),
                "--receipt",
                str(receipt),
                "--calibration",
                str(calibration),
            ]
        )
        == 0
    )
    assert calls == ["cascade"]


def test_cuda_service_generation_passes_bfloat16_to_community_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    model, _source = _fixture(tmp_path)
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    captured: dict[str, Any] = {}

    class FakePipeline:
        config = SimpleNamespace(
            profile_version=generator.COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
            mode="cascade",
        )

        def detect_batch(self, texts: Any, entities: Any) -> list[list[Any]]:
            del entities
            return [[] for _text in texts]

    def fake_builder(model_path: Path, **kwargs: Any) -> FakePipeline:
        captured["model_path"] = model_path
        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generator, "build_community_model_service_pipeline", fake_builder)
    monkeypatch.setattr(generator, "_write_batches", lambda *args, **kwargs: 1)
    args = SimpleNamespace(
        mode="cascade",
        model=model,
        device="cuda:0",
        micro_batch_size=2,
        calibration=calibration,
    )

    assert (
        generator._generate_service(
            args,
            labels=("PERSON_NAME",),
            temporary=tmp_path / "temporary.jsonl",
        )
        == 1
    )
    assert captured["dtype"] is torch.bfloat16
