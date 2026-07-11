from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pii_zh.evaluation import canonical_json_hash

_SUITES = {
    "synthetic_test": {
        "source_id": "pii_zh_synthetic_v1",
        "upstream_source": "repo_curated_synthetic_templates",
        "upstream_revision": "1dc82af3ac68e8c9c1f20e985304dd9121a01b9db5fe112ef82e3cdc59722d1e",
        "license": "Apache-2.0",
        "subset": "test",
        "gold_sha256": "26e49949d0bd876b51c693b7b31135b5ce503ca6a5ab7cfca37177e0ccbd4384",
        "record_count": 2_000,
        "span_count": 3_267,
        "manifest_sha256": "4" * 64,
        "manifest_file_sha256": "5" * 64,
        "evaluation_only": True,
    },
    "pii_bench_formal": {
        "source_id": "pii_bench_zh",
        "upstream_source": "wan9yu/pii-bench-zh",
        "upstream_revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
        "license": "Apache-2.0",
        "subset": "formal",
        "gold_sha256": "bb978137711cc90ae0c16df0645983413f3b34943bf29469e58b795b183a38af",
        "record_count": 5_000,
        "span_count": 15_662,
        "manifest_sha256": "545c990c96f23118ddc097ec0bed1d7f180785ac72e9f2a6ed56eb1ba3ee5e9b",
        "manifest_file_sha256": "6" * 64,
        "evaluation_only": True,
    },
    "pii_bench_chat": {
        "source_id": "pii_bench_zh",
        "upstream_source": "wan9yu/pii-bench-zh",
        "upstream_revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
        "license": "Apache-2.0",
        "subset": "chat",
        "gold_sha256": "2d73326a7ea2da9966ba13336db1a791f03d5f18b7982fdc4495cb726c1180a5",
        "record_count": 3_000,
        "span_count": 7_544,
        "manifest_sha256": "545c990c96f23118ddc097ec0bed1d7f180785ac72e9f2a6ed56eb1ba3ee5e9b",
        "manifest_file_sha256": "6" * 64,
        "evaluation_only": True,
    },
}


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _score(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def _digest(suite: str, suffix: str) -> str:
    return canonical_json_hash({"suite": suite, "suffix": suffix})


def _system_identity(suite: str, dataset: dict[str, Any], predicted: int) -> dict[str, Any]:
    final_sha = _digest(suite, "final")
    fused_sha = _digest(suite, "fused")
    calibrated_sha = _digest(suite, "calibrated")
    training_sha = "a" * 64
    model_identity = {
        "model_type": "qwen3_bi",
        "architecture_version": "qwen3_bi_token_cls_v1",
        "config_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "label_schema_sha256": "d" * 64,
    }
    model_identity["identity_sha256"] = canonical_json_hash(model_identity)
    fusion_module_sha256 = "3" * 64
    fusion_cli_sha256 = "4" * 64
    components = {
        "model_prediction": {
            "manifest_sha256": _digest(suite, "model-manifest"),
            "predictions_sha256": _digest(suite, "model-predictions"),
            "prediction_document_count": dataset["record_count"],
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "model_training_manifest_sha256": training_sha,
        },
        "rules_prediction": {
            "manifest_sha256": _digest(suite, "rules-manifest"),
            "predictions_sha256": _digest(suite, "rules-predictions"),
            "prediction_document_count": dataset["record_count"],
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "rules_identity": {
                "ruleset_id": "cn_common_v5",
                "implementation_sha256": "f" * 64,
                "configuration_sha256": "1" * 64,
            },
        },
        "fusion": {
            "implementation_sha256": canonical_json_hash(
                {
                    "module_sha256": fusion_module_sha256,
                    "cli_sha256": fusion_cli_sha256,
                }
            ),
            "module_sha256": fusion_module_sha256,
            "cli_sha256": fusion_cli_sha256,
            "configuration_sha256": "5" * 64,
            "model_prediction_manifest_sha256": _digest(suite, "model-manifest"),
            "model_predictions_sha256": _digest(suite, "model-predictions"),
            "rules_prediction_manifest_sha256": _digest(suite, "rules-manifest"),
            "rules_predictions_sha256": _digest(suite, "rules-predictions"),
            "output_predictions_sha256": fused_sha,
            "prediction_document_count": dataset["record_count"],
        },
        "calibration_fit": {
            "bundle_sha256": "6" * 64,
            "diagnostics_manifest_sha256": "7" * 64,
            "calibration_version": "cal-seed42-frozen-v1",
            "gold_sha256": ("9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d"),
            "fused_predictions_sha256": "9" * 64,
            "document_count": 2_000,
        },
        "target_application": {
            "bundle_sha256": "6" * 64,
            "calibration_version": "cal-seed42-frozen-v1",
            "input_predictions_sha256": fused_sha,
            "output_predictions_sha256": calibrated_sha,
            "prediction_document_count": dataset["record_count"],
        },
        "refinement": {
            "audit_manifest_sha256": _digest(suite, "refinement-audit"),
            "refinement_id": "structured_refinement_v4",
            "implementation_sha256": "0" * 64,
            "input_predictions_sha256": calibrated_sha,
            "output_predictions_sha256": final_sha,
            "prediction_document_count": dataset["record_count"],
        },
    }
    return {
        "prediction_id": f"system-{suite}",
        "predictor_type": "system",
        "predictions_sha256": final_sha,
        "manifest_sha256": _digest(suite, "system-manifest"),
        "manifest_file_sha256": _digest(suite, "system-manifest-file"),
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "prediction_document_count": dataset["record_count"],
        "model_training_manifest_sha256": training_sha,
        "seed": 42,
        "attention_mode": "full",
        "model_identity": model_identity,
        "components": components,
        "_predicted": predicted,
    }


def _report(path: Path, *, suite: str) -> Path:
    dataset = dict(_SUITES[suite])
    support = dataset["span_count"]
    strict = _score(support - 10, 5, 10)
    system = _system_identity(suite, dataset, int(strict["predicted"]))
    predicted = system.pop("_predicted")
    intervals = {
        name: {
            "point": 0.95,
            "lower": 0.94,
            "upper": 0.96,
            "effective_samples": 1_000,
        }
        for name in (
            "boundary_exact_f1",
            "character_micro_f1",
            "relaxed_micro_f1",
            "strict_macro_f1",
            "strict_micro_f1",
        )
    }
    intervals["pii_free_false_positive_rate"] = {
        "point": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "effective_samples": 1_000 if suite == "synthetic_test" else 0,
    }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "mode": "release",
        "gold": {"sha256": dataset["gold_sha256"], "dataset": dataset},
        "predictions": {
            "sha256": system["predictions_sha256"],
            "manifest": system,
        },
        "model_training": {
            "manifest_sha256": "a" * 64,
            "manifest_file_sha256": "b" * 64,
            "seed": 42,
            "attention_mode": "full",
        },
        "parameters": {
            "bootstrap_samples": 1_000,
            "confidence": 0.95,
            "iou_threshold": 0.5,
            "seed": 42,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
        },
    }
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    report = {
        "document_count": dataset["record_count"],
        "span_count": {"gold": support, "predicted": predicted},
        "strict": {
            "micro": strict,
            "macro": {"f1": strict["f1"]},
            "per_class": {"FIXTURE_LABEL": strict},
        },
        "relaxed": {"micro": strict},
        "character": {"micro": strict},
        "pii_free": {
            "documents": 600 if suite == "synthetic_test" else 0,
            "false_positive_documents": 0,
            "false_positive_rate": 0.0 if suite == "synthetic_test" else None,
        },
        "bootstrap": {
            "method": "document_level_percentile_bootstrap",
            "samples": 1_000,
            "seed": 42,
            "confidence": 0.95,
            "intervals": intervals,
        },
        "provenance": provenance,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    return {suite: _report(tmp_path / f"{suite}.json", suite=suite) for suite in _SUITES}


def _run(
    repository_root: Path, reports: dict[str, Path], output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/summarize_system_evaluations.py"),
            "--synthetic-test",
            str(reports["synthetic_test"]),
            "--pii-bench-formal",
            str(reports["pii_bench_formal"]),
            "--pii-bench-chat",
            str(reports["pii_bench_chat"]),
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _rewrite(path: Path, mutation: Any) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    mutation(report)
    provenance = report["provenance"]
    provenance.pop("evaluation_sha256", None)
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def test_summary_is_self_hashed_path_free_and_pools_counts(
    tmp_path: Path, repository_root: Path
) -> None:
    reports = _fixture(tmp_path)
    output = tmp_path / "summary.json"

    completed = _run(repository_root, reports, output)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(summary)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert summary["suite_order"] == list(_SUITES)
    assert summary["system_identity"]["selected_seed"] == 42
    assert summary["aggregate"]["document_count"] == 10_000
    assert summary["aggregate"]["gold_span_count"] == 26_473
    assert output.stat().st_mode & 0o777 == 0o444
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "doc_id" not in serialized
    assert '"raw_text":' not in serialized


@pytest.mark.parametrize(
    "identity",
    ["training", "model", "calibration", "rules", "fusion", "refinement"],
)
def test_fails_closed_when_common_system_identity_differs(
    tmp_path: Path, repository_root: Path, identity: str
) -> None:
    reports = _fixture(tmp_path)

    def mutate(report: dict[str, Any]) -> None:
        provenance = report["provenance"]
        system = provenance["predictions"]["manifest"]
        components = system["components"]
        if identity == "training":
            provenance["model_training"]["manifest_sha256"] = "f" * 64
            system["model_training_manifest_sha256"] = "f" * 64
            components["model_prediction"]["model_training_manifest_sha256"] = "f" * 64
        elif identity == "model":
            system["model_identity"]["weights_sha256"] = "f" * 64
            model_identity = dict(system["model_identity"])
            model_identity.pop("identity_sha256")
            system["model_identity"]["identity_sha256"] = canonical_json_hash(model_identity)
        elif identity == "calibration":
            components["calibration_fit"]["bundle_sha256"] = "f" * 64
            components["target_application"]["bundle_sha256"] = "f" * 64
        elif identity == "rules":
            components["rules_prediction"]["rules_identity"]["implementation_sha256"] = "e" * 64
        elif identity == "fusion":
            components["fusion"]["configuration_sha256"] = "e" * 64
        else:
            components["refinement"]["implementation_sha256"] = "e" * 64

    _rewrite(reports["pii_bench_chat"], mutate)
    output = tmp_path / "summary.json"
    output.write_text('{"stale":"pass"}\n', encoding="utf-8")

    completed = _run(repository_root, reports, output)

    assert completed.returncode != 0
    assert "different model/system/calibration identity" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "dataset",
        "bootstrap",
        "predictor",
        "mode",
        "ruleset",
        "refinement",
        "calibration_fit",
        "fusion_implementation",
    ],
)
def test_rejects_non_frozen_or_non_system_report(
    tmp_path: Path, repository_root: Path, failure: str
) -> None:
    reports = _fixture(tmp_path)

    def mutate(report: dict[str, Any]) -> None:
        if failure == "dataset":
            report["provenance"]["gold"]["dataset"]["subset"] = "formal"
        elif failure == "bootstrap":
            report["bootstrap"]["samples"] = 999
        elif failure == "predictor":
            report["provenance"]["predictions"]["manifest"]["predictor_type"] = "model"
        elif failure == "mode":
            report["provenance"]["mode"] = "development"
        else:
            components = report["provenance"]["predictions"]["manifest"]["components"]
            if failure == "ruleset":
                components["rules_prediction"]["rules_identity"]["ruleset_id"] = "cn_common_v4"
            elif failure == "refinement":
                components["refinement"]["refinement_id"] = "structured_refinement_v3"
            elif failure == "calibration_fit":
                components["calibration_fit"]["gold_sha256"] = "8" * 64
            else:
                components["fusion"]["implementation_sha256"] = "2" * 64

    _rewrite(reports["pii_bench_chat"], mutate)
    output = tmp_path / "summary.json"

    completed = _run(repository_root, reports, output)

    assert completed.returncode != 0
    assert not output.exists()
