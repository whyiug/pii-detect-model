from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pii_zh.data.schema import DocumentRecord as CanonicalDocument
from pii_zh.data.schema import (
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    dumps_document,
)
from pii_zh.evaluation import release_eval_v2_calibration_service as diagnostic
from pii_zh.evaluation.io import PredictionRecord, Span, dumps_prediction, load_prediction_jsonl


def test_cli_help_runs_without_pythonpath() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/evaluate_release_eval_v2_calibration_service.py"),
            "--help",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Development-only calibration evaluation" in completed.stdout


def _document(
    *,
    doc_id: str,
    text: str,
    entities: tuple[tuple[int, int, str], ...],
) -> CanonicalDocument:
    return CanonicalDocument.create(
        doc_id=doc_id,
        text=text,
        entities=tuple(
            EntitySpan(
                start=start,
                end=end,
                label=label,
                text=text[start:end],
                source="unit-fixture",
            )
            for start, end, label in entities
        ),
        language="zh-Hans",
        domain="unit",
        scene="unit",
        provenance=Provenance(
            source_id="unit",
            source_kind="fixture",
            source_revision="v1",
            license="CC0-1.0",
            synthetic=True,
            generator_name="unit",
            generator_version="v1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="fixture",
        ),
        public_weight_training_allowed=True,
        generator_version="v1",
        split="calibration",
    )


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[
    diagnostic.CalibrationServiceRequest,
    list[str],
]:
    events: list[str] = []
    text = "联系人张三，邮箱foo@example.com。"
    person_start = text.index("张三")
    email_start = text.index("foo@example.com")
    documents = [
        _document(
            doc_id="doc-1",
            text=text,
            entities=(
                (person_start, person_start + 2, "PERSON_NAME"),
                (email_start, email_start + len("foo@example.com"), "EMAIL_ADDRESS"),
            ),
        ),
        _document(doc_id="doc-2", text="今天天气很好。", entities=()),
    ]
    gold = tmp_path / "calibration.jsonl"
    gold_payload = ("\n".join(dumps_document(record) for record in documents) + "\n").encode()
    gold_sha = _write(gold, gold_payload)

    raw = tmp_path / "raw.jsonl"
    raw_payload = (
        dumps_prediction(
            PredictionRecord(
                doc_id="doc-1",
                spans=(
                    Span(
                        start=person_start,
                        end=person_start + 2,
                        label="PERSON_NAME",
                        score=0.9,
                    ),
                ),
            )
        )
        + "\n"
        + dumps_prediction(PredictionRecord(doc_id="doc-2", spans=()))
        + "\n"
    ).encode()
    raw_sha = _write(raw, raw_payload)

    manifest = tmp_path / "dataset.yaml"
    manifest_payload = b"""usage_policy:
  calibration:
    gradient_training_allowed: false
    checkpoint_or_model_selection_allowed: false
    threshold_tuning_allowed: true
    temperature_scaling_allowed: true
    fusion_calibration_allowed: true
    final_metric_claim_allowed: false
    content_read_before_cross_seed_selection_allowed: false
"""
    manifest_sha = _write(manifest, manifest_payload)

    authorization = tmp_path / "authorization.json"
    authorization_value: dict[str, Any] = {
        "dataset": {
            "dataset_id": "unit-v2",
            "dataset_version": "2.0.0",
            "dataset_manifest_file_sha256": manifest_sha,
            "dataset_manifest_sha256": "a" * 64,
            "gold_file_sha256": gold_sha,
            "gold_size_bytes": len(gold_payload),
            "gold_document_count": 2,
        },
        "model": {
            "training_manifest_file_sha256": "b" * 64,
            "training_manifest_sha256": "c" * 64,
            "model_identity_sha256": "d" * 64,
        },
        "receipt_sha256": "e" * 64,
    }
    authorization_payload = (json.dumps(authorization_value, sort_keys=True) + "\n").encode()
    authorization_sha = _write(authorization, authorization_payload)

    bundle = tmp_path / "calibration.json"
    _write(
        bundle,
        (
            json.dumps(
                {
                    "model_version": "c" * 64,
                    "calibration_version": "unit-v1",
                    "global_temperature": 1.0,
                    "entity_temperatures": {},
                    "entity_thresholds": {"PERSON_NAME": 0.5},
                    "default_threshold": 0.5,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    candidates = tuple(tmp_path / f"candidate-{index}" for index in range(3))
    request = diagnostic.CalibrationServiceRequest(
        authorization=authorization,
        protocol=tmp_path / "protocol.json",
        development_manifest=tmp_path / "development.json",
        v1_release_manifest=tmp_path / "v1.json",
        candidate_roots=(candidates[0], candidates[1], candidates[2]),
        selection_receipt=tmp_path / "selection.json",
        amendment=tmp_path / "amendment.json",
        dataset_manifest=manifest,
        materialization_receipt=tmp_path / "materialization.json",
        freeze_receipt=tmp_path / "freeze.json",
        calibration_gold=gold,
        model_artifact=tmp_path / "model",
        model_raw_predictions=raw,
        model_raw_generation_receipt=tmp_path / "generation.json",
        model_raw_prediction_manifest=tmp_path / "raw-manifest.json",
        calibration_bundle=bundle,
        model_calibrated_predictions=tmp_path / "model-calibrated.jsonl",
        full_system_predictions=tmp_path / "full-system.jsonl",
        receipt=tmp_path / "receipt.json",
    )

    def guard(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("guard")
        return {
            "status": "CALIBRATION_PREOPEN_PASS",
            "authorization_file_sha256": authorization_sha,
            "authorization_sha256": authorization_value["receipt_sha256"],
            "expected_input_sha256": gold_sha,
        }

    def replay_auth(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], str]:
        assert events == ["guard"]
        events.append("authorization_replay")
        return authorization_value, authorization_sha

    def replay_raw(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        assert events == ["guard", "authorization_replay"]
        events.append("raw_replay")
        return {
            "manifest_file_sha256": "f" * 64,
            "manifest_sha256": "1" * 64,
            "predictions_file_sha256": raw_sha,
        }

    monkeypatch.setattr(diagnostic, "guard_calibration_preopen", guard)
    monkeypatch.setattr(diagnostic, "replay_calibration_authorization", replay_auth)
    monkeypatch.setattr(diagnostic, "replay_release_eval_v2_prediction_manifest", replay_raw)
    return request, events


def test_produce_and_replay_are_text_free_read_only_and_use_real_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, events = _fixture(tmp_path, monkeypatch)
    result = diagnostic.produce_calibration_service_diagnostic(request)

    assert result["formal_claim_allowed"] is False
    assert events == ["guard", "authorization_replay", "raw_replay"]
    assert stat.S_IMODE(request.receipt.stat().st_mode) == 0o444
    assert stat.S_IMODE(request.model_calibrated_predictions.stat().st_mode) == 0o444
    assert b"foo@example.com" not in request.full_system_predictions.read_bytes()

    model = load_prediction_jsonl(request.model_calibrated_predictions)
    full = load_prediction_jsonl(request.full_system_predictions)
    assert [span.label for span in model[0].spans] == ["PERSON_NAME"]
    assert {span.label for span in full[0].spans} == {"PERSON_NAME", "EMAIL_ADDRESS"}

    receipt = json.loads(request.receipt.read_text(encoding="utf-8"))
    assert receipt["scope"]["gpu_used"] is False
    assert receipt["scope"]["bootstrap_samples"] == 0
    assert receipt["claim_policy"]["release_gate_eligible"] is False
    assert (
        receipt["metrics"]["full_system"]["strict"]["micro"]["f1"]
        > receipt["metrics"]["model_calibrated"]["strict"]["micro"]["f1"]
    )

    events.clear()
    replay = diagnostic.replay_calibration_service_diagnostic(request)
    assert replay["status"] == "DEVELOPMENT_ONLY_REPLAY_PASS_NOT_RELEASE"
    assert replay["prediction_byte_identical"] is True


def test_no_clobber_fails_before_calibration_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, events = _fixture(tmp_path, monkeypatch)
    request.receipt.write_text("occupied", encoding="utf-8")
    with pytest.raises(diagnostic.CalibrationServiceDiagnosticError, match="overwrite"):
        diagnostic.produce_calibration_service_diagnostic(request)
    assert events == []


def test_usage_policy_rejects_formal_claim_or_disabled_fusion(tmp_path: Path) -> None:
    payload = b"""usage_policy:
  calibration:
    gradient_training_allowed: false
    checkpoint_or_model_selection_allowed: false
    threshold_tuning_allowed: true
    temperature_scaling_allowed: true
    fusion_calibration_allowed: false
    final_metric_claim_allowed: false
    content_read_before_cross_seed_selection_allowed: false
"""
    path = tmp_path / "manifest.yaml"
    digest = _write(path, payload)
    snapshot = diagnostic._snapshot(
        path,
        field="fixture manifest",
        maximum_bytes=1024,
    )
    with pytest.raises(diagnostic.CalibrationServiceDiagnosticError, match="usage policy"):
        diagnostic._usage_policy(snapshot, expected_sha256=digest)
