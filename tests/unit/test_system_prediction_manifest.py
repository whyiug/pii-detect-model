from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pii_zh.calibration import CalibrationBundle, apply_calibration
from pii_zh.evaluation import (
    EvaluationDataError,
    PredictionRecord,
    Span,
    add_manifest_hash,
    build_evaluation_provenance,
    build_model_prediction_manifest,
    build_rule_prediction_manifest,
    build_system_prediction_manifest,
    canonical_json_hash,
    load_gold_jsonl,
    load_prediction_jsonl,
    sha256_file,
    validate_system_prediction_manifest,
    write_prediction_jsonl,
)
from pii_zh.fusion import fuse_prediction_records


def _write_manifest(path: Path, payload: dict[str, object]) -> dict[str, object]:
    manifest = add_manifest_hash(payload)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def _prediction(path: Path, *, score: float, doc_id: str = "fixture-doc") -> None:
    path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "spans": [{"start": 0, "end": 1, "label": "PERSON", "score": score}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _system_fixture(
    tmp_path: Path,
    *,
    cross_dataset_calibration_fit: bool = False,
    target_subset: str = "validation",
) -> dict[str, Path | str | int]:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "doc_id": "fixture-doc",
                "text": "甲乙丙",
                "spans": [{"start": 0, "end": 1, "label": "PERSON"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.json"
    _write_manifest(
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
                target_subset: {
                    "record_count": 1,
                    "span_count": 1,
                    "label_counts": {"PERSON": 1},
                    "canonical": {"sha256": sha256_file(gold)},
                }
            },
        },
    )

    label_schema_sha256 = canonical_json_hash({"O": 0, "B-PERSON": 1})
    training_path = tmp_path / "training.json"
    _write_manifest(
        training_path,
        {
            "schema_version": 2,
            "status": "completed",
            "seed": 42,
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

    model_predictions = tmp_path / "model.jsonl"
    rules_predictions = tmp_path / "rules.jsonl"
    fused_predictions = tmp_path / "fused.jsonl"
    calibrated_predictions = tmp_path / "calibrated.jsonl"
    final_predictions = tmp_path / "final.jsonl"
    _prediction(model_predictions, score=0.51)
    _prediction(rules_predictions, score=0.61)

    model_manifest_path = tmp_path / "model-prediction.json"
    model_manifest = build_model_prediction_manifest(
        predictions_path=model_predictions,
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
    model_manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")

    rules_manifest_path = tmp_path / "rules-prediction.json"
    rules_manifest = build_rule_prediction_manifest(
        predictions_path=rules_predictions,
        dataset_manifest_path=dataset_path,
        prediction_document_count=1,
        ruleset_id="fixture_rules_v1",
        implementation_sha256="3" * 64,
        configuration_sha256="4" * 64,
    )
    rules_manifest_path.write_text(json.dumps(rules_manifest), encoding="utf-8")

    calibration_bundle = tmp_path / "calibration.json"
    calibration_bundle.write_text(
        json.dumps(
            {
                "model_version": "fixture-model",
                "calibration_version": "cal-fixture-v1",
                "global_temperature": 1.0,
                "entity_temperatures": {},
                "entity_thresholds": {"PERSON": 0.5},
                "default_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    write_prediction_jsonl(
        fuse_prediction_records(
            load_prediction_jsonl(rules_predictions),
            load_prediction_jsonl(model_predictions),
        ),
        fused_predictions,
    )
    bundle = CalibrationBundle.from_json(calibration_bundle)
    write_prediction_jsonl(
        apply_calibration(load_prediction_jsonl(fused_predictions), bundle),
        calibrated_predictions,
    )
    write_prediction_jsonl(load_prediction_jsonl(calibrated_predictions), final_predictions)

    calibration_fit_gold = gold
    calibration_fit_predictions = fused_predictions
    calibration_fit_count = 1
    if cross_dataset_calibration_fit:
        calibration_fit_gold = tmp_path / "fit-gold.jsonl"
        calibration_fit_gold.write_text(
            "\n".join(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "text": text,
                        "spans": [{"start": 0, "end": 1, "label": "PERSON"}],
                    },
                    ensure_ascii=False,
                )
                for doc_id, text in (("fit-one", "丁戊己"), ("fit-two", "庚辛壬"))
            )
            + "\n",
            encoding="utf-8",
        )
        calibration_fit_predictions = tmp_path / "fit-fused.jsonl"
        write_prediction_jsonl(
            [
                PredictionRecord(
                    doc_id,
                    (Span(0, 1, "PERSON", score),),
                )
                for doc_id, score in (("fit-one", 0.73), ("fit-two", 0.83))
            ],
            calibration_fit_predictions,
        )
        calibration_fit_count = 2
    diagnostics_path = tmp_path / "calibration-diagnostics.json"
    _write_manifest(
        diagnostics_path,
        {
            "schema_version": 1,
            "manifest_type": "calibration_diagnostics",
            "calibration_version": "cal-fixture-v1",
            "calibration_bundle_sha256": sha256_file(calibration_bundle),
            "taxonomy_version": "1.0.0",
            "inputs": {
                "gold_sha256": sha256_file(calibration_fit_gold),
                "predictions_sha256": sha256_file(calibration_fit_predictions),
                "document_count": calibration_fit_count,
            },
            "parameters": {},
            "temperature": {},
            "confidence_calibration": {},
            "per_label": {},
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
            },
        },
    )
    refinement_path = tmp_path / "refinement.json"
    _write_manifest(
        refinement_path,
        {
            "schema_version": 1,
            "manifest_type": "structured_prediction_refinement",
            "refinement_id": "structured_refinement_v3",
            "inputs": {
                "documents_sha256": sha256_file(gold),
                "predictions_sha256": sha256_file(calibrated_predictions),
                "document_count": 1,
                "input_span_count": 1,
            },
            "output": {"predictions_sha256": sha256_file(final_predictions), "span_count": 1},
            "suppressed_by_label": {},
            "privacy": {
                "contains_document_ids": False,
                "contains_entity_values": False,
                "contains_paths": False,
                "contains_raw_text": False,
            },
        },
    )
    refinement_implementation = tmp_path / "refinement.py"
    refinement_implementation.write_text("# deterministic refinement\n", encoding="utf-8")
    return {
        "gold": gold,
        "dataset": dataset_path,
        "training": training_path,
        "model_predictions": model_predictions,
        "model_manifest": model_manifest_path,
        "rules_predictions": rules_predictions,
        "rules_manifest": rules_manifest_path,
        "fused": fused_predictions,
        "calibration_bundle": calibration_bundle,
        "calibration_diagnostics": diagnostics_path,
        "calibration_fit_gold": calibration_fit_gold,
        "calibration_fit_predictions": calibration_fit_predictions,
        "calibrated": calibrated_predictions,
        "refinement_audit": refinement_path,
        "refinement_implementation": refinement_implementation,
        "final": final_predictions,
        "count": 1,
    }


def _build(paths: dict[str, Path | str | int]) -> dict[str, object]:
    return build_system_prediction_manifest(
        predictions_path=paths["final"],
        prediction_document_count=paths["count"],
        dataset_manifest_path=paths["dataset"],
        model_training_manifest_path=paths["training"],
        model_predictions_path=paths["model_predictions"],
        model_prediction_manifest_path=paths["model_manifest"],
        rules_predictions_path=paths["rules_predictions"],
        rules_prediction_manifest_path=paths["rules_manifest"],
        fused_predictions_path=paths["fused"],
        calibration_bundle_path=paths["calibration_bundle"],
        calibration_diagnostics_path=paths["calibration_diagnostics"],
        calibration_fit_gold_path=paths["calibration_fit_gold"],
        calibration_fit_predictions_path=paths["calibration_fit_predictions"],
        calibrated_predictions_path=paths["calibrated"],
        refinement_audit_path=paths["refinement_audit"],
        refinement_implementation_sha256=sha256_file(paths["refinement_implementation"]),
    )


def test_system_manifest_binds_complete_chain_and_release_evaluation(tmp_path: Path) -> None:
    paths = _system_fixture(tmp_path)
    manifest = _build(paths)
    manifest_path = tmp_path / "system-prediction.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    unsigned = dict(manifest)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert manifest["predictor_type"] == "system"
    assert manifest["components"]["refinement"]["output_predictions_sha256"] == sha256_file(
        paths["final"]
    )
    assert manifest["components"]["calibration_fit"]["gold_sha256"] == sha256_file(
        paths["calibration_fit_gold"]
    )
    assert manifest["components"]["target_application"]["input_predictions_sha256"] == sha256_file(
        paths["fused"]
    )
    assert str(tmp_path) not in json.dumps(manifest)
    assert "fixture-doc" not in json.dumps(manifest)

    identity = validate_system_prediction_manifest(
        manifest_path=manifest_path,
        predictions_path=paths["final"],
        dataset_manifest_path=paths["dataset"],
        model_training_manifest_path=paths["training"],
        model_predictions_path=paths["model_predictions"],
        model_prediction_manifest_path=paths["model_manifest"],
        rules_predictions_path=paths["rules_predictions"],
        rules_prediction_manifest_path=paths["rules_manifest"],
        fused_predictions_path=paths["fused"],
        calibration_bundle_path=paths["calibration_bundle"],
        calibration_diagnostics_path=paths["calibration_diagnostics"],
        calibration_fit_gold_path=paths["calibration_fit_gold"],
        calibration_fit_predictions_path=paths["calibration_fit_predictions"],
        calibrated_predictions_path=paths["calibrated"],
        refinement_audit_path=paths["refinement_audit"],
        refinement_implementation_sha256=sha256_file(paths["refinement_implementation"]),
    )
    assert identity["predictor_type"] == "system"
    assert identity["components"]["rules_prediction"]["rules_identity"]["ruleset_id"] == (
        "fixture_rules_v1"
    )
    assert identity["components"]["calibration_fit"]["calibration_version"] == "cal-fixture-v1"
    assert identity["components"]["target_application"]["calibration_version"] == ("cal-fixture-v1")
    provenance = build_evaluation_provenance(
        gold_path=paths["gold"],
        predictions_path=paths["final"],
        gold_records=load_gold_jsonl(paths["gold"]),
        dataset_manifest_path=paths["dataset"],
        prediction_manifest_path=manifest_path,
        model_training_manifest_path=paths["training"],
        release_mode=True,
    )
    assert provenance["mode"] == "release"
    assert provenance["predictions"]["manifest"]["predictor_type"] == "system"
    assert provenance["predictions"]["manifest"]["components"]["fusion"]


@pytest.mark.parametrize("target_subset", ["formal", "chat"])
def test_system_manifest_reuses_validation_calibration_on_a_different_target_dataset(
    tmp_path: Path, target_subset: str
) -> None:
    paths = _system_fixture(
        tmp_path,
        cross_dataset_calibration_fit=True,
        target_subset=target_subset,
    )
    manifest = _build(paths)
    manifest_path = tmp_path / "cross-dataset-system.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert manifest["components"]["calibration_fit"]["gold_sha256"] == sha256_file(
        paths["calibration_fit_gold"]
    )
    assert manifest["components"]["calibration_fit"]["gold_sha256"] != sha256_file(paths["gold"])
    assert manifest["components"]["calibration_fit"]["document_count"] == 2
    assert manifest["components"]["target_application"]["prediction_document_count"] == 1
    provenance = build_evaluation_provenance(
        gold_path=paths["gold"],
        predictions_path=paths["final"],
        gold_records=load_gold_jsonl(paths["gold"]),
        dataset_manifest_path=paths["dataset"],
        prediction_manifest_path=manifest_path,
        model_training_manifest_path=paths["training"],
        release_mode=True,
    )

    assert provenance["gold"]["dataset"]["subset"] == target_subset
    assert (
        provenance["predictions"]["manifest"]["components"]["target_application"]["bundle_sha256"]
        == manifest["components"]["calibration_fit"]["bundle_sha256"]
    )


@pytest.mark.parametrize("tamper", ["component_output", "diagnostics", "refinement"])
def test_system_manifest_builder_rejects_broken_component_bindings(
    tmp_path: Path, tamper: str
) -> None:
    paths = _system_fixture(tmp_path)
    if tamper == "component_output":
        _prediction(paths["model_predictions"], score=0.99)
        expected = "does not match predictions SHA-256"
    elif tamper == "diagnostics":
        diagnostics = json.loads(paths["calibration_diagnostics"].read_text(encoding="utf-8"))
        diagnostics.pop("manifest_sha256")
        diagnostics["inputs"]["predictions_sha256"] = "f" * 64
        _write_manifest(paths["calibration_diagnostics"], diagnostics)
        expected = "do not bind fit predictions"
    else:
        refinement = json.loads(paths["refinement_audit"].read_text(encoding="utf-8"))
        refinement.pop("manifest_sha256")
        refinement["output"]["predictions_sha256"] = "e" * 64
        _write_manifest(paths["refinement_audit"], refinement)
        expected = "does not bind final predictions"

    with pytest.raises(EvaluationDataError, match=expected):
        _build(paths)


def test_system_manifest_builder_rejects_tampered_self_hash(tmp_path: Path) -> None:
    paths = _system_fixture(tmp_path)
    diagnostics = json.loads(paths["calibration_diagnostics"].read_text(encoding="utf-8"))
    diagnostics["taxonomy_version"] = "tampered"
    paths["calibration_diagnostics"].write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="content hash does not verify"):
        _build(paths)


def test_system_manifest_builder_rejects_bundle_not_bound_by_fit_diagnostics(
    tmp_path: Path,
) -> None:
    paths = _system_fixture(tmp_path)
    bundle = json.loads(paths["calibration_bundle"].read_text(encoding="utf-8"))
    bundle["default_threshold"] = 0.9
    paths["calibration_bundle"].write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="do not bind the calibration bundle"):
        _build(paths)


def test_system_manifest_builder_rejects_noncanonical_target_fusion(tmp_path: Path) -> None:
    paths = _system_fixture(tmp_path)
    _prediction(paths["fused"], score=0.2)

    with pytest.raises(EvaluationDataError, match="target fused predictions do not semantically"):
        _build(paths)


@pytest.mark.parametrize("tamper", ["semantic", "bytes"])
def test_system_manifest_builder_rejects_fabricated_target_calibration(
    tmp_path: Path, tamper: str
) -> None:
    paths = _system_fixture(tmp_path)
    if tamper == "semantic":
        _prediction(paths["calibrated"], score=0.99)
        expected = "do not semantically match"
    else:
        rows = [
            json.loads(line)
            for line in paths["calibrated"].read_text(encoding="utf-8").splitlines()
        ]
        paths["calibrated"].write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) + "  " for row in rows) + "\n",
            encoding="utf-8",
        )
        expected = "do not byte-match"

    with pytest.raises(EvaluationDataError, match=expected):
        _build(paths)


def test_system_manifest_validator_rejects_fabricated_fusion_configuration(
    tmp_path: Path,
) -> None:
    paths = _system_fixture(tmp_path)
    manifest = _build(paths)
    manifest["components"]["fusion"]["configuration_sha256"] = "f" * 64
    manifest.pop("manifest_sha256")
    manifest = add_manifest_hash(manifest)
    manifest_path = tmp_path / "fake-config-system.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="fixed release config"):
        validate_system_prediction_manifest(
            manifest_path=manifest_path,
            predictions_path=paths["final"],
            dataset_manifest_path=paths["dataset"],
            model_training_manifest_path=paths["training"],
            model_predictions_path=paths["model_predictions"],
            model_prediction_manifest_path=paths["model_manifest"],
            rules_predictions_path=paths["rules_predictions"],
            rules_prediction_manifest_path=paths["rules_manifest"],
            fused_predictions_path=paths["fused"],
            calibration_bundle_path=paths["calibration_bundle"],
            calibration_diagnostics_path=paths["calibration_diagnostics"],
            calibration_fit_gold_path=paths["calibration_fit_gold"],
            calibration_fit_predictions_path=paths["calibration_fit_predictions"],
            calibrated_predictions_path=paths["calibrated"],
            refinement_audit_path=paths["refinement_audit"],
            refinement_implementation_sha256=sha256_file(paths["refinement_implementation"]),
        )


def test_release_evaluation_rejects_system_model_training_mismatch(tmp_path: Path) -> None:
    paths = _system_fixture(tmp_path)
    manifest_path = tmp_path / "system-prediction.json"
    manifest_path.write_text(json.dumps(_build(paths)), encoding="utf-8")
    alternate_training_path = tmp_path / "alternate-training.json"
    alternate_training = json.loads(paths["training"].read_text(encoding="utf-8"))
    alternate_training.pop("manifest_sha256")
    alternate_training["seed"] = 13
    _write_manifest(alternate_training_path, alternate_training)

    with pytest.raises(EvaluationDataError, match="different model training manifest"):
        build_evaluation_provenance(
            gold_path=paths["gold"],
            predictions_path=paths["final"],
            gold_records=load_gold_jsonl(paths["gold"]),
            dataset_manifest_path=paths["dataset"],
            prediction_manifest_path=manifest_path,
            model_training_manifest_path=alternate_training_path,
            release_mode=True,
        )


def test_create_system_prediction_manifest_cli_is_read_only_and_path_free(
    tmp_path: Path,
) -> None:
    paths = _system_fixture(tmp_path)
    output = tmp_path / "system.json"
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/create_system_prediction_manifest.py"),
            "--predictions",
            str(paths["final"]),
            "--prediction-document-count",
            "1",
            "--dataset-manifest",
            str(paths["dataset"]),
            "--model-training-manifest",
            str(paths["training"]),
            "--model-predictions",
            str(paths["model_predictions"]),
            "--model-prediction-manifest",
            str(paths["model_manifest"]),
            "--rules-predictions",
            str(paths["rules_predictions"]),
            "--rules-prediction-manifest",
            str(paths["rules_manifest"]),
            "--fused-predictions",
            str(paths["fused"]),
            "--calibration-bundle",
            str(paths["calibration_bundle"]),
            "--calibration-diagnostics",
            str(paths["calibration_diagnostics"]),
            "--calibration-fit-gold",
            str(paths["calibration_fit_gold"]),
            "--calibration-fit-predictions",
            str(paths["calibration_fit_predictions"]),
            "--calibrated-predictions",
            str(paths["calibrated"]),
            "--refinement-audit",
            str(paths["refinement_audit"]),
            "--refinement-implementation",
            str(paths["refinement_implementation"]),
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
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "fixture-doc" not in serialized
    assert output.stat().st_mode & 0o777 == 0o444
