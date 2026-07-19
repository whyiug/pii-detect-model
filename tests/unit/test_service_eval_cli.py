from __future__ import annotations

import json
import re
import stat
from pathlib import Path

import pytest

from pii_zh.cascade import (
    LEGACY_SERVICE_PROFILE_VERSION,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    CascadePipeline,
)
from pii_zh.data.schema import (
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)
from pii_zh.evaluation import service_eval_cli


def _pii_free_record(*, text: str, doc_id: str) -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(),
        language="zh-Hans-CN",
        domain="synthetic-test",
        scene="service-evaluation-unit-test",
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


def _run(
    *,
    profile: str,
    input_path: Path,
    predictions_path: Path,
    report_path: Path,
) -> int:
    return service_eval_cli.main(
        [
            "--service-profile",
            profile,
            "--run-intent",
            "development-non-claim",
            "--input",
            str(input_path),
            "--predictions",
            str(predictions_path),
            "--report",
            str(report_path),
            "--bootstrap-samples",
            "0",
        ]
    )


def _absolute_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _absolute_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _absolute_strings(nested)]
    if isinstance(value, str) and (
        value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        return [value]
    return []


def test_service_profiles_use_factory_and_v2_rejects_legacy_broad_plate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = "车辆岚A·AB12CD3已核验。"
    original_doc_id = "/private/human-hidden/病例-张三.json"
    input_path = tmp_path / "input.jsonl"
    write_jsonl([_pii_free_record(text=text, doc_id=original_doc_id)], input_path)

    factory_calls: list[str] = []
    real_factory = service_eval_cli.build_rules_only_service_pipeline

    def factory(profile: str) -> CascadePipeline:
        factory_calls.append(profile)
        return real_factory(profile)

    monkeypatch.setattr(service_eval_cli, "build_rules_only_service_pipeline", factory)
    outputs: dict[str, tuple[Path, Path]] = {}
    for profile in (LEGACY_SERVICE_PROFILE_VERSION, SUCCESSOR_SERVICE_PROFILE_VERSION):
        predictions = tmp_path / f"{profile}.predictions.jsonl"
        report = tmp_path / f"{profile}.report.json"
        assert (
            _run(
                profile=profile,
                input_path=input_path,
                predictions_path=predictions,
                report_path=report,
            )
            == 0
        )
        outputs[profile] = (predictions, report)

    assert factory_calls == [LEGACY_SERVICE_PROFILE_VERSION, SUCCESSOR_SERVICE_PROFILE_VERSION]
    legacy_prediction = json.loads(
        outputs[LEGACY_SERVICE_PROFILE_VERSION][0].read_text(encoding="utf-8")
    )
    successor_prediction = json.loads(
        outputs[SUCCESSOR_SERVICE_PROFILE_VERSION][0].read_text(encoding="utf-8")
    )
    assert [span["label"] for span in legacy_prediction["spans"]] == ["VEHICLE_LICENSE_PLATE"]
    assert successor_prediction["spans"] == []

    legacy_report = json.loads(
        outputs[LEGACY_SERVICE_PROFILE_VERSION][1].read_text(encoding="utf-8")
    )
    successor_report = json.loads(
        outputs[SUCCESSOR_SERVICE_PROFILE_VERSION][1].read_text(encoding="utf-8")
    )
    assert legacy_report["pii_free"]["false_positive_rate"] == 1.0
    assert successor_report["pii_free"]["false_positive_rate"] == 0.0
    assert legacy_report["service_evaluation_runtime"]["configuration"]["rule_source"] == (
        "rule:cn_common"
    )
    successor_runtime = successor_report["service_evaluation_runtime"]
    assert successor_runtime["configuration"]["rule_source"] == "rule:cn_common_v6"
    plate_route = next(
        route
        for route in successor_runtime["configuration"]["routes"]
        if route["entity_type"] == "CN_VEHICLE_LICENSE_PLATE"
    )
    assert plate_route["validator_label"] == "CN_VEHICLE_LICENSE_PLATE"
    assert successor_runtime["canonical_evaluation_scope"]["entity_count"] == 24
    assert (
        successor_runtime["canonical_evaluation_scope"]["core_evaluator_macro_denominator"]
        == "implementation_defined_observed_label_scope_may_apply"
    )
    assert (
        successor_runtime["canonical_evaluation_scope"]["formal_fixed_24_macro_status"]
        == "NOT_CLAIMED_IN_THIS_DEVELOPMENT_RUN"
    )
    assert successor_runtime["claim_eligible"] is False
    assert successor_runtime["human_gold_admission_contract_status"] == "UNAVAILABLE"

    captured = capsys.readouterr()
    for predictions_path, report_path in outputs.values():
        predictions_text = predictions_path.read_text(encoding="utf-8")
        report_text = report_path.read_text(encoding="utf-8")
        for sensitive in (text, "岚A·AB12CD3", original_doc_id, str(tmp_path)):
            assert sensitive not in predictions_text
            assert sensitive not in report_text
            assert sensitive not in captured.out
            assert sensitive not in captured.err
        row = json.loads(predictions_text)
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", row["doc_id"])
        assert stat.S_IMODE(predictions_path.stat().st_mode) == 0o400
        assert stat.S_IMODE(report_path.stat().st_mode) == 0o400
        assert _absolute_strings(json.loads(report_text)) == []

    closure = successor_runtime["package_source_closure"]
    assert closure["closure_mode"].startswith("full_pii_zh_")
    assert closure["source_file_count"] == len(closure["component_sha256"])
    for required in (
        "cascade/service_profiles.py",
        "data/validators/cn_vehicle_plate.py",
        "evaluation/metrics.py",
        "evaluation/required_metrics.py",
        "evaluation/service_eval_cli.py",
        "rules/cn_common.py",
        "rules/cn_common_v6.py",
    ):
        assert required in closure["component_sha256"]
    assert (
        successor_runtime["canonical_input_sha256"]
        == successor_report["provenance"]["gold"]["sha256"]
    )
    assert (
        successor_runtime["predictions_sha256"]
        == successor_report["provenance"]["predictions"]["sha256"]
    )


def test_existing_output_is_not_overwritten(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl([_pii_free_record(text="普通文本", doc_id="safe-doc")], input_path)
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    predictions_path.write_text("owner-data\n", encoding="utf-8")

    assert (
        _run(
            profile=SUCCESSOR_SERVICE_PROFILE_VERSION,
            input_path=input_path,
            predictions_path=predictions_path,
            report_path=report_path,
        )
        == 2
    )

    assert predictions_path.read_text(encoding="utf-8") == "owner-data\n"
    assert not report_path.exists()
    assert capsys.readouterr().err.strip() == "error: refusing to overwrite an existing output"


def test_confirmatory_fails_before_input_access_because_admission_is_unavailable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"

    status = service_eval_cli.main(
        [
            "--service-profile",
            SUCCESSOR_SERVICE_PROFILE_VERSION,
            "--run-intent",
            "frozen-confirmatory",
            "--input",
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--predictions",
            str(predictions_path),
            "--report",
            str(report_path),
        ]
    )

    assert status == 2
    assert capsys.readouterr().err.strip() == (
        "error: frozen-confirmatory admission contract unavailable"
    )
    assert not predictions_path.exists()
    assert not report_path.exists()


def test_formal_surface_has_no_entity_filter() -> None:
    with pytest.raises(SystemExit):
        service_eval_cli.main(
            [
                "--service-profile",
                SUCCESSOR_SERVICE_PROFILE_VERSION,
                "--run-intent",
                "development-non-claim",
                "--input",
                "unused.jsonl",
                "--predictions",
                "unused.predictions.jsonl",
                "--report",
                "unused.report.json",
                "--entity",
                "PERSON",
            ]
        )
