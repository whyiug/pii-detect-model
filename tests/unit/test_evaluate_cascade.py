from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import evaluate_cascade

from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)


def _canonical_record(
    *, doc_id: str, text: str, start: int, end: int, label: str
) -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(EntitySpan(start, end, label, text[start:end], "unit-test"),),
        language="zh-Hans-CN",
        domain="synthetic-test",
        scene="unit-test",
        provenance=Provenance(
            source_id="unit-test",
            source_kind="synthetic",
            source_revision="v1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit-test",
            generator_version="v1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="synthetic_fixture",
        ),
        public_weight_training_allowed=True,
        generator_version="v1",
    )


def test_rules_only_generates_raw_text_free_predictions_and_shared_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "测试邮箱 demo@example.com"
    input_path = tmp_path / "input.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    start = sentinel.index("demo@example.com")
    write_jsonl(
        [
            _canonical_record(
                doc_id="synthetic-doc",
                text=sentinel,
                start=start,
                end=len(sentinel),
                label="EMAIL_ADDRESS",
            )
        ],
        input_path,
    )

    status = evaluate_cascade.main(
        [
            "--input",
            str(input_path),
            "--predictions",
            str(predictions_path),
            "--report",
            str(report_path),
            "--mode",
            "rules-only",
            "--bootstrap-samples",
            "0",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    serialized_predictions = predictions_path.read_text(encoding="utf-8")
    serialized_report = report_path.read_text(encoding="utf-8")
    assert sentinel not in serialized_predictions
    assert sentinel not in serialized_report
    assert sentinel not in captured.out
    prediction = json.loads(serialized_predictions)
    assert set(prediction) == {"doc_id", "spans"}
    assert prediction["spans"][0]["label"] == "EMAIL_ADDRESS"
    report = json.loads(serialized_report)
    assert report["strict"]["micro"]["f1"] == 1.0
    required = report["required_metrics"]
    assert required["schema_version"] == "pii-zh.required-metrics.v1"
    assert required["risk_tier_minimum_recall"]["tiers"]["T1"]["minimum_recall"] == 1.0
    assert required["redaction_character_errors"]["missed_sensitive_character_rate"]["value"] == 0.0
    assert required["fragmentation"]["fragmentation_rate"] == 0.0
    assert required["privacy"] == {
        "contains_raw_text": False,
        "contains_entity_values": False,
        "contains_document_ids": False,
    }
    runtime = report["cascade_runtime"]
    assert runtime["mode"] == "rules-only"
    assert runtime["configuration"]["mode"] == "rules-only"
    assert len(runtime["configuration_sha256"]) == 64
    assert runtime["predictions_sha256"] == report["provenance"]["predictions"]["sha256"]
    assert predictions_path.stat().st_mode & 0o222 == 0
    assert report_path.stat().st_mode & 0o222 == 0


def test_invalid_canonical_input_error_does_not_echo_text_or_entity_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "never-echo-this-entity-value"
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "doc_id": "synthetic-doc",
                "text": sentinel,
                "entities": [
                    {
                        "start": 0,
                        "end": 5,
                        "label": "PERSON_NAME",
                        "text": "wrong",
                        "source": "unit-test",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = evaluate_cascade.main(
        [
            "--input",
            str(input_path),
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--bootstrap-samples",
            "0",
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert sentinel not in captured.err
    assert "wrong" not in captured.err
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("mode", ["model-only", "cascade"])
def test_model_modes_use_local_from_pretrained_pipeline(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "测试甲"
    input_path = tmp_path / f"{mode}.jsonl"
    write_jsonl(
        [
            _canonical_record(
                doc_id=f"{mode}-doc",
                text=text,
                start=0,
                end=len(text),
                label="PERSON_NAME",
            )
        ],
        input_path,
    )
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    (model_path / "training_manifest.json").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[Path, str, bool]] = []

    class Recognizer:
        def analyze_batch(
            self, texts: list[str], entities: list[str]
        ) -> list[list[dict[str, object]]]:
            assert "PERSON" in entities
            return [
                [{"entity_type": "PERSON_NAME", "start": 0, "end": len(value), "score": 0.99}]
                for value in texts
            ]

    def fake_from_pretrained(
        path: Path, *, config: object, device: str, local_files_only: bool, **kwargs: object
    ) -> object:
        del kwargs
        calls.append((path, device, local_files_only))
        return evaluate_cascade.CascadePipeline(
            config=config,  # type: ignore[arg-type]
            model_recognizer=Recognizer(),
        )

    monkeypatch.setattr(
        evaluate_cascade.CascadePipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    predictions_path = tmp_path / f"{mode}-predictions.jsonl"
    report_path = tmp_path / f"{mode}-report.json"

    status = evaluate_cascade.main(
        [
            "--input",
            str(input_path),
            "--predictions",
            str(predictions_path),
            "--report",
            str(report_path),
            "--mode",
            mode,
            "--model-path",
            str(model_path),
            "--bootstrap-samples",
            "0",
        ]
    )

    assert status == 0
    assert calls == [(model_path.resolve(), "cpu", True)]
    prediction = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert prediction["spans"][0]["label"] == "PERSON_NAME"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cascade_runtime"]["mode"] == mode
    assert report["strict"]["micro"]["f1"] == 1.0
