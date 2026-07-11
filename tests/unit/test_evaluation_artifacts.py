from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pii_zh.evaluation import (
    EvaluationDataError,
    add_manifest_hash,
    build_model_prediction_manifest,
    canonical_json_hash,
    sha256_file,
)


def _write_manifest(path: Path, payload: dict[str, object]) -> dict[str, object]:
    manifest = add_manifest_hash(payload)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def test_create_prediction_manifest_is_deterministic_and_path_free(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"doc_id":"doc","spans":[]}\n', encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset = _write_manifest(dataset_path, {"manifest_type": "evaluation_dataset"})
    training_path = tmp_path / "training.json"
    training = _write_manifest(training_path, {"seed": 13})
    output = tmp_path / "prediction_manifest.json"
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/create_prediction_manifest.py"),
            "--predictions",
            str(predictions),
            "--dataset-manifest",
            str(dataset_path),
            "--model-training-manifest",
            str(training_path),
            "--prediction-id",
            "fixture-seed13",
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    logical_hash = unsigned.pop("manifest_sha256")
    assert logical_hash == canonical_json_hash(unsigned)
    assert manifest["predictions_sha256"] == sha256_file(predictions)
    assert manifest["dataset_manifest_sha256"] == dataset["manifest_sha256"]
    assert manifest["model_training_manifest_sha256"] == training["manifest_sha256"]
    assert output.stat().st_mode & 0o777 == 0o444
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_model_prediction_manifest_binds_loaded_model_seed_and_attention(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"doc_id":"doc","spans":[]}\n', encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset = _write_manifest(
        dataset_path,
        {"schema_version": 1, "manifest_type": "evaluation_dataset"},
    )
    label_schema_sha256 = canonical_json_hash({"O": 0, "B-PERSON_NAME": 1})
    training_path = tmp_path / "training.json"
    training = _write_manifest(
        training_path,
        {
            "schema_version": 2,
            "status": "completed",
            "seed": 13,
            "attention_mode": "full",
            "label_schema_sha256": label_schema_sha256,
            "output_artifact": {
                "files": {
                    "config.json": "1" * 64,
                    "model.safetensors": "2" * 64,
                }
            },
        },
    )

    manifest = build_model_prediction_manifest(
        predictions_path=predictions,
        dataset_manifest_path=dataset_path,
        model_training_manifest_path=training_path,
        prediction_document_count=1,
        attention_mode="full",
        model_identity={
            "model_type": "qwen3_bi",
            "architecture_version": "qwen3_bi_token_cls_v1",
            "config_sha256": "1" * 64,
            "weights_sha256": "2" * 64,
            "label_schema_sha256": label_schema_sha256,
        },
    )

    assert manifest["dataset_manifest_sha256"] == dataset["manifest_sha256"]
    assert manifest["model_training_manifest_sha256"] == training["manifest_sha256"]
    assert manifest["seed"] == 13
    assert manifest["attention_mode"] == "full"
    assert manifest["model_identity"]["model_type"] == "qwen3_bi"
    assert manifest["model_identity"]["identity_sha256"]
    assert str(tmp_path) not in json.dumps(manifest)


@pytest.mark.parametrize(
    ("field", "digest", "message"),
    [
        ("config_sha256", "3" * 64, "config does not match"),
        ("weights_sha256", "4" * 64, "weights do not match"),
    ],
)
def test_model_prediction_manifest_rejects_checkpoint_tampering(
    tmp_path: Path, field: str, digest: str, message: str
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"doc_id":"doc","spans":[]}\n', encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    _write_manifest(dataset_path, {"manifest_type": "evaluation_dataset"})
    label_schema_sha256 = canonical_json_hash({"O": 0})
    training_path = tmp_path / "training.json"
    _write_manifest(
        training_path,
        {
            "status": "completed",
            "seed": 13,
            "attention_mode": "full",
            "label_schema_sha256": label_schema_sha256,
            "output_artifact": {
                "files": {
                    "config.json": "1" * 64,
                    "model.safetensors": "2" * 64,
                }
            },
        },
    )
    model_identity = {
        "model_type": "qwen3_bi",
        "config_sha256": "1" * 64,
        "weights_sha256": "2" * 64,
        "label_schema_sha256": label_schema_sha256,
    }
    model_identity[field] = digest

    with pytest.raises(EvaluationDataError, match=message):
        build_model_prediction_manifest(
            predictions_path=predictions,
            dataset_manifest_path=dataset_path,
            model_training_manifest_path=training_path,
            prediction_document_count=1,
            attention_mode="full",
            model_identity=model_identity,
        )


def test_model_predict_cli_requires_all_three_provenance_arguments(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/predict.py"),
            "--model",
            str(tmp_path / "model"),
            "--input",
            str(tmp_path / "input.jsonl"),
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--dataset-manifest",
            str(tmp_path / "dataset.json"),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "must be supplied together" in completed.stderr


def _evaluation_report(subset: str, dataset_manifest_sha256: str) -> dict[str, object]:
    micro = {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "support": 1,
        "predicted": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "f2": 1.0,
    }
    interval = {"point": 1.0, "lower": 1.0, "upper": 1.0, "effective_samples": 100}
    return {
        "document_count": 1,
        "span_count": {"gold": 1, "predicted": 1},
        "strict": {"micro": micro},
        "relaxed": {"micro": micro},
        "character": {"micro": micro},
        "pii_free": {
            "documents": 0,
            "false_positive_documents": 0,
            "false_positive_rate": None,
        },
        "confidence_calibration": {
            "status": "available",
            "reason": None,
            "scope": "emitted_prediction_spans",
            "target": "strict_start_end_label_match",
            "predicted_span_count": 1,
            "scored_span_count": 1,
            "binning": {"method": "equal_width", "bin_count": 10},
            "ece": 0.1,
            "brier": 0.01,
        },
        "bootstrap": {
            "method": "document_level_percentile_bootstrap",
            "confidence": 0.95,
            "samples": 100,
            "seed": 42,
            "intervals": {
                name: interval
                for name in (
                    "strict_micro_f1",
                    "strict_macro_f1",
                    "relaxed_micro_f1",
                    "character_micro_f1",
                    "boundary_exact_f1",
                    "pii_free_false_positive_rate",
                )
            },
        },
        "provenance": {
            "evaluation_sha256": "e" * 64,
            "gold": {
                "dataset": {
                    "source_id": "pii_bench_zh",
                    "upstream_source": "wan9yu/pii-bench-zh",
                    "upstream_revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
                    "license": "Apache-2.0",
                    "subset": subset,
                    "gold_sha256": "a" * 64,
                    "record_count": 1,
                    "span_count": 1,
                    "manifest_sha256": dataset_manifest_sha256,
                    "manifest_file_sha256": "b" * 64,
                    "evaluation_only": True,
                }
            },
            "predictions": {"sha256": "f" * 64},
        },
    }


def test_rule_baseline_summary_is_self_hashed_read_only_and_path_free(tmp_path: Path) -> None:
    dataset_manifest_sha256 = "d" * 64
    formal = tmp_path / "formal.json"
    chat = tmp_path / "chat.json"
    formal.write_text(
        json.dumps(_evaluation_report("formal", dataset_manifest_sha256)), encoding="utf-8"
    )
    chat.write_text(
        json.dumps(_evaluation_report("chat", dataset_manifest_sha256)), encoding="utf-8"
    )
    output = tmp_path / "summary.json"
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/summarize_rule_baseline.py"),
            "--formal-report",
            str(formal),
            "--chat-report",
            str(chat),
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(summary)
    logical_hash = unsigned.pop("manifest_sha256")
    assert logical_hash == canonical_json_hash(unsigned)
    assert summary["dataset_manifest_sha256"] == dataset_manifest_sha256
    assert output.stat().st_mode & 0o777 == 0o444
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
