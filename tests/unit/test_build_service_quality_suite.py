from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest
from scripts import build_service_quality_suite as builder

from pii_zh.evaluation import (
    PredictionRecord,
    Span,
    canonical_json_hash,
    sha256_file,
    write_prediction_jsonl,
)
from pii_zh.evaluation.service_eval_cli import (
    SERVICE_EVALUATION_DOCUMENT_ID_VERSION,
    SERVICE_EVALUATION_SCHEMA_VERSION,
    service_evaluation_doc_id,
)
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.taxonomy import load_presidio_mapping

_RAW_ID_SENTINEL = "RAW_DOCUMENT_ID_MUST_NOT_ESCAPE"
_RAW_TEXT_SENTINEL = "RAW_TEXT_AND_ENTITY_VALUE_MUST_NOT_ESCAPE"


@dataclass(frozen=True, slots=True)
class _Fixture:
    gold: Path
    candidate_predictions: Path
    candidate_report: Path
    comparator_predictions: Path
    comparator_report: Path
    raw_doc_ids: tuple[str, ...]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_gold(path: Path) -> tuple[str, ...]:
    rows: list[dict[str, object]] = []
    raw_doc_ids: list[str] = []
    for index, label in enumerate(PII_CORE_LABELS):
        doc_id = f"{_RAW_ID_SENTINEL}-positive-{index:02d}"
        text = f"X-{_RAW_TEXT_SENTINEL}-{index:02d}"
        raw_doc_ids.append(doc_id)
        rows.append(
            {
                "doc_id": doc_id,
                "text": text,
                "spans": [{"start": 0, "end": 1, "label": label, "text": "X"}],
            }
        )
    for index in range(2):
        doc_id = f"{_RAW_ID_SENTINEL}-negative-{index:02d}"
        raw_doc_ids.append(doc_id)
        rows.append(
            {
                "doc_id": doc_id,
                "text": f"ordinary-{_RAW_TEXT_SENTINEL}-{index:02d}",
                "spans": [],
            }
        )
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return tuple(raw_doc_ids)


def _predictions(
    raw_doc_ids: tuple[str, ...], *, miss_first_positive: bool
) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    for index, doc_id in enumerate(raw_doc_ids):
        spans: tuple[Span, ...] = ()
        if index < len(PII_CORE_LABELS) and not (miss_first_positive and index == 0):
            spans = (Span(0, 1, PII_CORE_LABELS[index], score=0.9),)
        records.append(PredictionRecord(service_evaluation_doc_id(doc_id), spans))
    return tuple(records)


def _package_closure(*, variant: str = "shared") -> dict[str, object]:
    components = {
        "cascade/service_profiles.py": ("1" if variant == "shared" else "3") * 64,
        "evaluation/service_eval_cli.py": "2" * 64,
    }
    return {
        "closure_mode": "full_pii_zh_python_and_packaged_resource_superset_v1",
        "source_file_count": len(components),
        "component_sha256": components,
        "implementation_sha256": canonical_json_hash(components),
    }


def _service_report(
    *,
    gold: Path,
    predictions: Path,
    document_count: int,
    package_variant: str = "shared",
) -> dict[str, object]:
    gold_sha256 = sha256_file(gold)
    predictions_sha256 = sha256_file(predictions)
    taxonomy = load_presidio_mapping()
    policy_labels = [taxonomy.model_to_presidio[label] for label in PII_CORE_LABELS]
    return {
        "provenance": {
            "schema_version": 1,
            "mode": "development",
            "gold": {"sha256": gold_sha256},
            "predictions": {"sha256": predictions_sha256},
        },
        "service_evaluation_runtime": {
            "schema_version": SERVICE_EVALUATION_SCHEMA_VERSION,
            "run_intent": "development-non-claim",
            "claim_eligible": False,
            "claim_status": "DEVELOPMENT_NON_CLAIM",
            "human_gold_admission_contract_status": "UNAVAILABLE",
            "canonical_input_sha256": gold_sha256,
            "predictions_sha256": predictions_sha256,
            "prediction_document_count": document_count,
            "canonical_evaluation_scope": {
                "scope_mode": "fixed_canonical_24_input_and_service_routing",
                "entity_count": 24,
                "metric_model_labels": list(PII_CORE_LABELS),
                "pipeline_policy_entities": policy_labels,
                "formal_fixed_24_macro_status": (
                    "NOT_CLAIMED_IN_THIS_DEVELOPMENT_RUN"
                ),
            },
            "document_id_policy": {
                "schema_version": SERVICE_EVALUATION_DOCUMENT_ID_VERSION,
                "algorithm": "domain_separated_sha256",
                "original_document_ids_persisted": False,
            },
            "privacy": {
                "contains_absolute_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_original_document_ids": False,
            },
            "package_source_closure": _package_closure(variant=package_variant),
        },
    }


def _fixture(tmp_path: Path) -> _Fixture:
    gold = tmp_path / "canonical-gold.jsonl"
    raw_doc_ids = _write_gold(gold)
    candidate_predictions = tmp_path / "candidate.predictions.jsonl"
    comparator_predictions = tmp_path / "comparator.predictions.jsonl"
    write_prediction_jsonl(
        _predictions(raw_doc_ids, miss_first_positive=False), candidate_predictions
    )
    write_prediction_jsonl(
        _predictions(raw_doc_ids, miss_first_positive=True), comparator_predictions
    )
    candidate_report = tmp_path / "candidate.report.json"
    comparator_report = tmp_path / "comparator.report.json"
    _write_json(
        candidate_report,
        _service_report(
            gold=gold,
            predictions=candidate_predictions,
            document_count=len(raw_doc_ids),
        ),
    )
    _write_json(
        comparator_report,
        _service_report(
            gold=gold,
            predictions=comparator_predictions,
            document_count=len(raw_doc_ids),
        ),
    )
    return _Fixture(
        gold=gold,
        candidate_predictions=candidate_predictions,
        candidate_report=candidate_report,
        comparator_predictions=comparator_predictions,
        comparator_report=comparator_report,
        raw_doc_ids=raw_doc_ids,
    )


def _argv(
    fixture: _Fixture,
    output: Path,
    *,
    run_intent: str = "development-non-claim",
) -> list[str]:
    return [
        "--run-intent",
        run_intent,
        "--suite-name",
        "production-pii-core-dev-test",
        "--gold",
        str(fixture.gold),
        "--candidate-id",
        "candidate-cascade",
        "--candidate-predictions",
        str(fixture.candidate_predictions),
        "--candidate-report",
        str(fixture.candidate_report),
        "--comparator",
        "rules-only",
        str(fixture.comparator_predictions),
        str(fixture.comparator_report),
        "--output",
        str(output),
    ]


def test_builds_private_aggregate_only_development_suite_without_sensitive_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "quality-suite.json"

    assert builder.main(_argv(fixture, output)) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(report)
    claimed_hash = unsigned.pop("artifact_sha256")
    assert claimed_hash == canonical_json_hash(unsigned)
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert report["run_intent"] == "development-non-claim"
    assert report["claims"] == {
        "first": False,
        "best": False,
        "global_best": False,
        "sota": False,
        "claim_eligible": False,
    }
    assert report["pii_free_eligibility"] == {
        "status": "DEVELOPMENT_INFERRED_NON_ADJUDICATED_NONFORMAL",
        "basis": "empty_gold_within_fixed_canonical_24_scope",
        "document_count": 2,
        "human_adjudicated": False,
        "formal_eligibility": False,
        "claim_eligible": False,
    }
    comparison = report["aggregate_suite"]["comparisons"]["rules-only"]
    assert comparison["recovered_comparator_false_negative_count"] == 1
    assert comparison["candidate_added_unique_true_positive_count"] == 1
    assert report["aggregate_suite"]["claim_activation"]["allowed"] is False

    serialized = output.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    for sensitive in (
        _RAW_ID_SENTINEL,
        _RAW_TEXT_SENTINEL,
        *fixture.raw_doc_ids,
        str(tmp_path),
    ):
        assert sensitive not in serialized
        assert sensitive not in captured.out
        assert sensitive not in captured.err
    assert report["privacy"] == {
        "aggregate_only": True,
        "contains_paths": False,
        "contains_raw_text": False,
        "contains_document_ids": False,
        "contains_entity_values": False,
        "per_document_statistics_output": False,
    }


def test_missing_comparator_is_rejected_before_gold_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "output.json"
    status = builder.main(
        [
            "--run-intent",
            "development-non-claim",
            "--gold",
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--candidate-id",
            "candidate",
            "--candidate-predictions",
            str(tmp_path / "missing.predictions.jsonl"),
            "--candidate-report",
            str(tmp_path / "missing.report.json"),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert capsys.readouterr().err.strip() == (
        "service-quality-suite: error: at least one comparator is required"
    )
    assert not output.exists()


def test_confirmatory_intent_fails_before_gold_or_comparator_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "output.json"
    status = builder.main(
        [
            "--run-intent",
            "frozen-confirmatory",
            "--gold",
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--candidate-id",
            "candidate",
            "--candidate-predictions",
            str(tmp_path / "must-not-be-opened.predictions.jsonl"),
            "--candidate-report",
            str(tmp_path / "must-not-be-opened.report.json"),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert capsys.readouterr().err.strip() == (
        "service-quality-suite: error: frozen-confirmatory service-quality admission "
        "contract unavailable"
    )
    assert not output.exists()


def test_wrong_prediction_hash_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture.candidate_report.read_text(encoding="utf-8"))
    report["service_evaluation_runtime"]["predictions_sha256"] = "0" * 64
    _write_json(fixture.candidate_report, report)
    output = tmp_path / "output.json"

    assert builder.main(_argv(fixture, output)) == 2

    assert "prediction hash mismatch" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["gold_hash", "document_count", "scope", "package_hash"])
def test_wrong_service_report_data_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture.candidate_report.read_text(encoding="utf-8"))
    runtime = report["service_evaluation_runtime"]
    if mutation == "gold_hash":
        runtime["canonical_input_sha256"] = "0" * 64
    elif mutation == "document_count":
        runtime["prediction_document_count"] -= 1
    elif mutation == "scope":
        runtime["canonical_evaluation_scope"]["entity_count"] = 23
    else:
        runtime["package_source_closure"]["implementation_sha256"] = "0" * 64
    _write_json(fixture.candidate_report, report)
    output = tmp_path / "output.json"

    assert builder.main(_argv(fixture, output)) == 2

    assert "service-quality-suite: error:" in capsys.readouterr().err
    assert not output.exists()


def test_different_package_implementations_are_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture.comparator_report,
        _service_report(
            gold=fixture.gold,
            predictions=fixture.comparator_predictions,
            document_count=len(fixture.raw_doc_ids),
            package_variant="different",
        ),
    )
    output = tmp_path / "output.json"

    assert builder.main(_argv(fixture, output)) == 2

    assert "different package implementations" in capsys.readouterr().err
    assert not output.exists()


def test_existing_output_is_never_overwritten(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output.json"
    output.write_text("owner-data\n", encoding="utf-8")

    assert builder.main(_argv(fixture, output)) == 2

    assert output.read_text(encoding="utf-8") == "owner-data\n"
    assert capsys.readouterr().err.strip() == (
        "service-quality-suite: error: refusing to overwrite an existing output"
    )
