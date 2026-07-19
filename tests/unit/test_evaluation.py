from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from pii_zh.evaluation import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    Span,
    add_manifest_hash,
    build_evaluation_provenance,
    build_rule_prediction_manifest,
    canonical_json_hash,
    evaluate_documents,
    load_gold_jsonl,
    load_prediction_jsonl,
    write_prediction_jsonl,
)


def test_exact_relaxed_character_boundary_and_confusion_metrics() -> None:
    gold = [
        GoldRecord(
            doc_id="doc-1",
            text_length=10,
            spans=(Span(0, 2, "PERSON"), Span(4, 8, "PHONE")),
        )
    ]
    predictions = [
        PredictionRecord(
            doc_id="doc-1",
            spans=(
                Span(0, 2, "PERSON"),
                Span(5, 8, "PHONE"),
                Span(8, 9, "EMAIL"),
            ),
        )
    ]

    result = evaluate_documents(gold, predictions, bootstrap_samples=0)

    assert result["strict"]["micro"] == pytest.approx(
        {
            "tp": 1,
            "fp": 2,
            "fn": 1,
            "support": 2,
            "predicted": 3,
            "precision": 1 / 3,
            "recall": 1 / 2,
            "f1": 0.4,
            "f2": 5 / 11,
        }
    )
    assert result["strict"]["macro"]["f1"] == pytest.approx(1 / 3)
    assert result["strict"]["per_class"]["PERSON"]["f2"] == 1.0
    assert result["strict"]["per_class"]["PHONE"]["recall"] == 0.0

    # PHONE has IoU 3/4 and therefore becomes a relaxed match at the 0.5 default.
    assert result["relaxed"]["micro"]["tp"] == 2
    assert result["relaxed"]["micro"]["f1"] == pytest.approx(0.8)
    assert result["boundary"]["exact_accuracy"] == pytest.approx(0.5)
    assert result["boundary"]["start_accuracy"] == pytest.approx(0.5)
    assert result["boundary"]["end_accuracy"] == 1.0

    assert result["character"]["micro"]["tp"] == 5
    assert result["character"]["micro"]["fp"] == 1
    assert result["character"]["micro"]["fn"] == 1
    assert result["character"]["micro"]["f1"] == pytest.approx(5 / 6)
    matrix = result["confusion"]["matrix"]
    assert matrix["PERSON"]["PERSON"] == 1
    assert matrix["PHONE"]["PHONE"] == 1
    assert matrix["__SPURIOUS__"]["EMAIL"] == 1


def test_pii_free_false_positive_document_rate() -> None:
    gold = [
        GoldRecord("negative-with-fp", (), text_length=8),
        GoldRecord("negative-clean", (), text_length=8),
        GoldRecord("positive", (Span(0, 2, "PERSON"),), text_length=8),
    ]
    predictions = [
        PredictionRecord("negative-with-fp", (Span(3, 4, "PERSON"),)),
        PredictionRecord("negative-clean", ()),
        PredictionRecord("positive", (Span(0, 2, "PERSON"),)),
    ]

    result = evaluate_documents(gold, predictions, bootstrap_samples=0)

    assert result["pii_free"] == {
        "documents": 2,
        "false_positive_documents": 1,
        "false_positive_rate": 0.5,
    }


def test_span_confidence_ece_and_brier_use_strict_match_target() -> None:
    gold = [GoldRecord("doc", (Span(0, 2, "PERSON"),), text_length=8)]
    predictions = [
        PredictionRecord(
            "doc",
            (
                Span(0, 2, "PERSON", score=0.8),
                Span(4, 6, "PHONE", score=0.4),
            ),
        )
    ]

    result = evaluate_documents(gold, predictions, bootstrap_samples=0)
    calibration = result["confidence_calibration"]

    assert calibration["status"] == "available"
    assert calibration["target"] == "strict_start_end_label_match"
    assert calibration["predicted_span_count"] == 2
    assert calibration["scored_span_count"] == 2
    assert calibration["brier"] == pytest.approx(0.1)
    assert calibration["ece"] == pytest.approx(0.3)
    assert calibration["bins"][4]["accuracy"] == 0.0
    assert calibration["bins"][8]["accuracy"] == 1.0


@pytest.mark.parametrize(
    ("spans", "reason"),
    [
        ((), "no_prediction_spans"),
        ((Span(0, 2, "PERSON"),), "scores_absent"),
        (
            (Span(0, 2, "PERSON", score=0.8), Span(3, 4, "PHONE")),
            "scores_incomplete",
        ),
    ],
)
def test_span_confidence_is_not_applicable_without_complete_scores(
    spans: tuple[Span, ...], reason: str
) -> None:
    gold = [GoldRecord("doc", (Span(0, 2, "PERSON"),), text_length=8)]
    result = evaluate_documents(
        gold,
        [PredictionRecord("doc", spans)],
        bootstrap_samples=0,
    )

    calibration = result["confidence_calibration"]
    assert calibration["status"] == "not_applicable"
    assert calibration["reason"] == reason
    assert calibration["ece"] is None
    assert calibration["brier"] is None


def test_document_bootstrap_is_seed_deterministic() -> None:
    gold = [
        GoldRecord("a", (Span(0, 1, "PERSON"),), text_length=3),
        GoldRecord("b", (Span(0, 1, "PERSON"),), text_length=3),
        GoldRecord("c", (), text_length=3),
        GoldRecord("d", (), text_length=3),
    ]
    predictions = [
        PredictionRecord("a", (Span(0, 1, "PERSON"),)),
        PredictionRecord("b", ()),
        PredictionRecord("c", (Span(1, 2, "PERSON"),)),
        PredictionRecord("d", ()),
    ]

    first = evaluate_documents(gold, predictions, bootstrap_samples=80, seed=20260711)
    second = evaluate_documents(gold, predictions, bootstrap_samples=80, seed=20260711)

    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["confidence"] == 0.95
    strict_interval = first["bootstrap"]["intervals"]["strict_micro_f1"]
    assert strict_interval["effective_samples"] == 80
    assert strict_interval["lower"] <= strict_interval["point"] <= strict_interval["upper"]


def test_prediction_jsonl_never_contains_raw_text() -> None:
    sentinel = "绝不写入预测输出的原文"
    canonical_gold = StringIO(
        json.dumps(
            {
                "doc_id": "safe-doc",
                "text": sentinel,
                "entities": [
                    {
                        "start": 0,
                        "end": 2,
                        "label": "PERSON",
                        "text": sentinel[:2],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    gold = load_gold_jsonl(canonical_gold)
    assert gold == [GoldRecord("safe-doc", (Span(0, 2, "PERSON"),), len(sentinel))]

    output = StringIO()
    write_prediction_jsonl(
        [PredictionRecord("safe-doc", (Span(0, 2, "PERSON", score=0.9),))], output
    )
    serialized = output.getvalue()
    payload = json.loads(serialized)
    assert set(payload) == {"doc_id", "spans"}
    assert set(payload["spans"][0]) == {"start", "end", "label", "score"}
    assert sentinel not in serialized
    assert "text" not in serialized

    with pytest.raises(EvaluationDataError, match="only doc_id and spans"):
        load_prediction_jsonl(
            StringIO(
                json.dumps(
                    {"doc_id": "unsafe", "text": sentinel, "spans": []},
                    ensure_ascii=False,
                )
            )
        )


@pytest.mark.parametrize(
    ("loader", "payload"),
    [
        (
            load_gold_jsonl,
            '{"doc_id":"safe","doc_id":"shadow","text_length":1,"spans":[]}\n',
        ),
        (
            load_prediction_jsonl,
            '{"doc_id":"safe","spans":[],"spans":[]}\n',
        ),
    ],
)
def test_evaluation_jsonl_rejects_duplicate_object_keys(loader: object, payload: str) -> None:
    with pytest.raises(EvaluationDataError, match="duplicate object key"):
        loader(StringIO(payload))  # type: ignore[operator]


@pytest.mark.parametrize("loader", [load_gold_jsonl, load_prediction_jsonl])
def test_evaluation_jsonl_rejects_non_finite_numbers(loader: object) -> None:
    payload = '{"doc_id":"safe","spans":[],"unused":NaN}\n'
    with pytest.raises(EvaluationDataError, match="non-finite number"):
        loader(StringIO(payload))  # type: ignore[operator]


def test_jsonl_cli_writes_aggregate_report_without_gold_text(tmp_path: Path) -> None:
    sentinel = "命令行报告不得包含这段原文"
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    gold_path.write_text(
        json.dumps(
            {
                "doc_id": "cli-doc",
                "text": sentinel,
                "spans": [{"start": 0, "end": 2, "label": "PERSON"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(
            {
                "doc_id": "cli-doc",
                "spans": [{"start": 0, "end": 2, "label": "PERSON"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "evaluate.py"),
            "--gold",
            str(gold_path),
            "--predictions",
            str(prediction_path),
            "--output",
            str(report_path),
            "--bootstrap-samples",
            "0",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    serialized = report_path.read_text(encoding="utf-8")
    assert sentinel not in serialized
    report = json.loads(serialized)
    assert report["strict"]["micro"]["f1"] == 1.0
    assert report["provenance"]["gold"]["sha256"]
    assert report["provenance"]["predictions"]["sha256"]
    assert str(gold_path) not in serialized


def test_evaluate_cli_refuses_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "report.json"
    output.write_text("sentinel", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/evaluate.py"),
            "--gold",
            str(tmp_path / "missing-gold.jsonl"),
            "--predictions",
            str(tmp_path / "missing-predictions.jsonl"),
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "refusing to overwrite" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel"


def _write_self_hashed_manifest(path: Path, payload: dict[str, object]) -> dict[str, object]:
    manifest = add_manifest_hash(payload)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def test_release_provenance_cross_binds_gold_predictions_and_model(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "text": "甲乙丙",
                "spans": [{"start": 0, "end": 1, "label": "PERSON"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "spans": [{"start": 0, "end": 1, "label": "PERSON", "score": 0.9}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from pii_zh.evaluation import sha256_file

    dataset_path = tmp_path / "dataset.json"
    dataset = _write_self_hashed_manifest(
        dataset_path,
        {
            "schema_version": 1,
            "manifest_type": "evaluation_dataset",
            "source_id": "fixture",
            "upstream": {
                "source": "fixture/evaluation-data",
                "revision": "a" * 40,
                "license": "Apache-2.0",
            },
            "isolation": {"pool": "evaluation_only", "training_allowed": False},
            "subsets": {
                "formal": {
                    "record_count": 1,
                    "span_count": 1,
                    "label_counts": {"PERSON": 1},
                    "raw": {"sha256": "b" * 64},
                    "canonical": {"sha256": sha256_file(gold_path)},
                }
            },
        },
    )
    training_path = tmp_path / "training.json"
    training = _write_self_hashed_manifest(
        training_path,
        {
            "schema_version": 1,
            "seed": 13,
            "attention_mode": "full",
            "taxonomy_version": "1.0.0",
            "recipe_sha256": "c" * 64,
            "label_schema_sha256": "d" * 64,
        },
    )
    prediction_manifest_path = tmp_path / "prediction.json"
    prediction_manifest = _write_self_hashed_manifest(
        prediction_manifest_path,
        {
            "schema_version": 1,
            "manifest_type": "prediction",
            "prediction_id": "fixture-formal-seed13",
            "predictor_type": "model",
            "predictions_sha256": sha256_file(prediction_path),
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "model_training_manifest_sha256": training["manifest_sha256"],
            "seed": 13,
            "attention_mode": "full",
            "model_identity": {
                "model_type": "qwen3_bi",
                "config_sha256": "1" * 64,
                "weights_sha256": "2" * 64,
                "label_schema_sha256": "d" * 64,
                "identity_sha256": canonical_json_hash(
                    {
                        "model_type": "qwen3_bi",
                        "config_sha256": "1" * 64,
                        "weights_sha256": "2" * 64,
                        "label_schema_sha256": "d" * 64,
                    }
                ),
            },
        },
    )

    provenance = build_evaluation_provenance(
        gold_path=gold_path,
        predictions_path=prediction_path,
        gold_records=load_gold_jsonl(gold_path),
        dataset_manifest_path=dataset_path,
        prediction_manifest_path=prediction_manifest_path,
        model_training_manifest_path=training_path,
        release_mode=True,
        evaluation_parameters={"seed": 42},
    )

    assert provenance["mode"] == "release"
    assert provenance["gold"]["dataset"]["subset"] == "formal"
    assert (
        provenance["predictions"]["manifest"]["manifest_sha256"]
        == prediction_manifest["manifest_sha256"]
    )
    assert provenance["model_training"]["seed"] == 13
    assert provenance["predictions"]["manifest"]["attention_mode"] == "full"
    assert provenance["predictions"]["manifest"]["model_identity"]["model_type"] == "qwen3_bi"
    serialized = json.dumps(provenance)
    assert str(tmp_path) not in serialized
    assert "甲乙丙" not in serialized


def test_release_provenance_rejects_dataset_manifest_mismatch(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    gold_path.write_text('{"doc_id":"doc","text":"x","spans":[]}\n', encoding="utf-8")
    predictions_path.write_text('{"doc_id":"doc","spans":[]}\n', encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    _write_self_hashed_manifest(
        dataset_path,
        {
            "schema_version": 1,
            "manifest_type": "evaluation_dataset",
            "source_id": "fixture",
            "upstream": {
                "source": "fixture/data",
                "revision": "a" * 40,
                "license": "Apache-2.0",
            },
            "isolation": {"pool": "evaluation_only", "training_allowed": False},
            "subsets": {
                "formal": {
                    "record_count": 1,
                    "span_count": 0,
                    "label_counts": {},
                    "canonical": {"sha256": "0" * 64},
                }
            },
        },
    )

    with pytest.raises(EvaluationDataError, match="does not identify exactly one"):
        build_evaluation_provenance(
            gold_path=gold_path,
            predictions_path=predictions_path,
            gold_records=load_gold_jsonl(gold_path),
            dataset_manifest_path=dataset_path,
        )


def test_release_provenance_requires_all_three_manifests(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    gold_path.write_text('{"doc_id":"doc","text":"x","spans":[]}\n', encoding="utf-8")
    predictions_path.write_text('{"doc_id":"doc","spans":[]}\n', encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="release mode requires"):
        build_evaluation_provenance(
            gold_path=gold_path,
            predictions_path=predictions_path,
            gold_records=load_gold_jsonl(gold_path),
            release_mode=True,
        )


def test_rule_prediction_manifest_has_rule_identity_and_no_training_hash(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    gold_path.write_text('{"doc_id":"doc","text":"x","spans":[]}\n', encoding="utf-8")
    predictions_path.write_text('{"doc_id":"doc","spans":[]}\n', encoding="utf-8")
    from pii_zh.evaluation import sha256_file

    dataset_path = tmp_path / "dataset.json"
    _write_self_hashed_manifest(
        dataset_path,
        {
            "schema_version": 1,
            "manifest_type": "evaluation_dataset",
            "source_id": "fixture",
            "upstream": {
                "source": "fixture/data",
                "revision": "a" * 40,
                "license": "Apache-2.0",
            },
            "isolation": {"pool": "evaluation_only", "training_allowed": False},
            "subsets": {
                "formal": {
                    "record_count": 1,
                    "span_count": 0,
                    "label_counts": {},
                    "canonical": {"sha256": sha256_file(gold_path)},
                }
            },
        },
    )
    manifest = build_rule_prediction_manifest(
        predictions_path=predictions_path,
        dataset_manifest_path=dataset_path,
        prediction_document_count=1,
        ruleset_id="fixture_rules_v1",
        implementation_sha256="1" * 64,
        configuration_sha256="2" * 64,
    )
    prediction_manifest_path = tmp_path / "rule-prediction.json"
    prediction_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    provenance = build_evaluation_provenance(
        gold_path=gold_path,
        predictions_path=predictions_path,
        gold_records=load_gold_jsonl(gold_path),
        dataset_manifest_path=dataset_path,
        prediction_manifest_path=prediction_manifest_path,
    )

    identity = provenance["predictions"]["manifest"]
    assert identity["predictor_type"] == "rules"
    assert identity["rules_identity"]["ruleset_id"] == "fixture_rules_v1"
    assert "model_training_manifest_sha256" not in identity
