from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.cascade import CascadePipeline, build_community_model_service_pipeline


def test_pipeline_passes_allowed_model_labels_to_predictor_before_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    predictor = SimpleNamespace(tokenizer=object(), attention_mode="full")

    def fake_loader(model_path: Path, **kwargs: object) -> object:
        calls.append({"model_path": model_path, **kwargs})
        return predictor

    class FakeRecognizer:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["predictor"] is predictor

        def analyze(self, text: str, entities: object) -> list[object]:
            del text, entities
            return []

    import pii_zh.inference.transformers_predictor as predictor_module
    import pii_zh.presidio as presidio_module

    monkeypatch.setattr(predictor_module, "load_local_predictor", fake_loader)
    monkeypatch.setattr(presidio_module, "QwenPiiRecognizer", FakeRecognizer)

    CascadePipeline.from_pretrained(
        tmp_path,
        allowed_model_labels=("PERSON_NAME", "ADDRESS"),
    )
    CascadePipeline.from_pretrained(tmp_path)

    assert calls[0]["allowed_labels"] == ("PERSON_NAME", "ADDRESS")
    assert "allowed_labels" not in calls[1]


def test_community_builder_normalizes_and_binds_allowed_model_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    identity = SimpleNamespace(to_dict=lambda: {"manifest_sha256": "a" * 64})
    monkeypatch.setattr(
        "pii_zh.cascade.service_profiles.verify_community_model_artifact",
        lambda path: SimpleNamespace(root=Path(path).resolve(), identity=identity),
    )
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(cls: type[CascadePipeline], path: Path, **kwargs: object) -> object:
        del cls
        calls.append({"path": path, **kwargs})
        return object()

    monkeypatch.setattr(
        CascadePipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    build_community_model_service_pipeline(
        model,
        mode="cascade",
        allowed_model_labels=("ADDRESS", "PERSON_NAME"),
    )
    build_community_model_service_pipeline(model, mode="cascade")

    assert calls[0]["allowed_model_labels"] == ("PERSON_NAME", "ADDRESS")
    bound_identity = calls[0]["model_identity"]
    assert isinstance(bound_identity, dict)
    assert isinstance(bound_identity["allowed_model_labels_sha256"], str)
    assert len(bound_identity["allowed_model_labels_sha256"]) == 64
    assert "allowed_model_labels" not in calls[1]
    assert "allowed_model_labels_sha256" not in calls[1]["model_identity"]  # type: ignore[operator]


@pytest.mark.parametrize(
    "labels",
    [
        "PERSON_NAME",
        (),
        ("PERSON_NAME", "PERSON_NAME"),
        ("NOT_CORE24",),
    ],
)
def test_community_builder_rejects_invalid_allowed_model_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    labels: object,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    identity = SimpleNamespace(to_dict=lambda: {"manifest_sha256": "a" * 64})
    monkeypatch.setattr(
        "pii_zh.cascade.service_profiles.verify_community_model_artifact",
        lambda path: SimpleNamespace(root=Path(path).resolve(), identity=identity),
    )

    with pytest.raises((TypeError, ValueError)):
        build_community_model_service_pipeline(
            model,
            mode="cascade",
            allowed_model_labels=labels,  # type: ignore[arg-type]
        )
