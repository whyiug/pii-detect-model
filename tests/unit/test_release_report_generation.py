from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.generate_release_report import (
    ReleaseReportError,
    build_artifact,
    main,
    validate_package_dir,
)

_GATES = (
    "file_set",
    "checksums",
    "remote_code",
    "training_artifact_binding",
    "provenance",
    "quality",
    "public_artifact_scan",
    "release_text",
    "sbom",
    "dependencies",
    "receipt_identity",
)


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = _canonical_hash(result)
    return result


def _file_hash(value: dict[str, Any]) -> str:
    encoded = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "f1": f1,
        "fn": fn,
        "fp": fp,
        "precision": precision,
        "predicted": tp + fp,
        "recall": recall,
        "support": tp + fn,
        "tp": tp,
    }


def _suite(
    *,
    documents: int,
    tp: int,
    fp: int,
    fn: int,
    pii_free_documents: int,
    false_positive_documents: int,
) -> dict[str, Any]:
    strict = _strict_counts(tp, fp, fn)
    f1 = strict["f1"]
    fpr = false_positive_documents / pii_free_documents if pii_free_documents else None
    return {
        "metrics": {
            "document_count": documents,
            "span_count": {"gold": tp + fn, "predicted": tp + fp},
            "strict_micro": strict,
            "strict_macro_f1": max(0.0, f1 - 0.04),
            "pii_free": {
                "documents": pii_free_documents,
                "false_positive_documents": false_positive_documents,
                "false_positive_rate": fpr,
            },
            "bootstrap": {
                "confidence": 0.95,
                "method": "document_level_percentile_bootstrap",
                "samples": 321,
                "seed": 19,
                "intervals": {
                    "strict_micro_f1": {
                        "effective_samples": 321,
                        "lower": max(0.0, f1 - 0.01),
                        "point": f1,
                        "upper": min(1.0, f1 + 0.01),
                    }
                },
            },
        }
    }


def _inputs() -> dict[str, Any]:
    training = _seal(
        {
            "schema_version": 4,
            "status": "completed",
            "seed": 42,
            "attention_mode": "full",
            "base_source_id": "fixture_base_model",
            "training_source_ids": [
                "fixture_base_model",
                "fixture_synthetic_data",
                "fixture_template_generator",
            ],
            "datasets": {
                "train": {"summary": {"document_count": 12}},
                "validation": {"summary": {"document_count": 5}},
            },
        }
    )
    training_file_sha256 = _file_hash(training)
    registry = {
        "schema_version": 1,
        "registry_version": "fixture-v7",
        "sha256": hashlib.sha256(b"fixture-registry").hexdigest(),
    }
    data_provenance = _seal(
        {
            "schema_version": 1,
            "artifact_type": "release_data_provenance",
            "training_manifest_sha256": training["manifest_sha256"],
            "training_manifest_file_sha256": training_file_sha256,
            "source_registry": registry,
            "sources": [
                {
                    "source_id": "fixture_synthetic_data",
                    "source_kind": "fixture_deterministic_generator",
                    "sample_count": 17,
                    "declared_license": "CC-BY-4.0",
                    "public_weight_training_allowed": True,
                    "synthetic": True,
                    "contains_real_personal_data": False,
                    "usage_counts": {
                        "gradient_training": 12,
                        "validation_early_stopping_and_calibration": 5,
                        "frozen_test_training_use": 0,
                    },
                }
            ],
            "total_sample_count": 17,
        }
    )
    teacher_provenance = _seal(
        {
            "schema_version": 1,
            "artifact_type": "release_teacher_provenance",
            "training_manifest_sha256": training["manifest_sha256"],
            "training_manifest_file_sha256": training_file_sha256,
            "source_registry": registry,
            "teacher_used": True,
            "external_api_used": False,
            "external_api_teacher_used": False,
            "teachers": [
                {
                    "source_id": "fixture_template_generator",
                    "role": "fixture_reviewed_candidate_role",
                    "declared_license": "Fixture-Research-License",
                    "reviewed_candidate_count": 11,
                    "accepted_candidate_count": 7,
                    "rejected_candidate_count": 4,
                    "used_for_training": True,
                    "accepted_placeholder_templates_only": True,
                    "raw_outputs_used_for_training": False,
                    "span_teacher": False,
                    "pseudo_label_teacher": False,
                    "external_api": False,
                }
            ],
        }
    )

    suites = {
        "synthetic_test": _suite(
            documents=20,
            tp=90,
            fp=20,
            fn=30,
            pii_free_documents=6,
            false_positive_documents=2,
        ),
        "pii_bench_formal": _suite(
            documents=50,
            tp=180,
            fp=20,
            fn=40,
            pii_free_documents=0,
            false_positive_documents=0,
        ),
        "pii_bench_chat": _suite(
            documents=30,
            tp=100,
            fp=20,
            fn=60,
            pii_free_documents=0,
            false_positive_documents=0,
        ),
    }
    totals = {
        "documents": sum(item["metrics"]["document_count"] for item in suites.values()),
        "gold": sum(item["metrics"]["span_count"]["gold"] for item in suites.values()),
        "predicted": sum(item["metrics"]["span_count"]["predicted"] for item in suites.values()),
        "tp": sum(item["metrics"]["strict_micro"]["tp"] for item in suites.values()),
        "fp": sum(item["metrics"]["strict_micro"]["fp"] for item in suites.values()),
        "fn": sum(item["metrics"]["strict_micro"]["fn"] for item in suites.values()),
    }
    frozen = _seal(
        {
            "schema_version": 1,
            "artifact_type": "system_frozen_evaluation_summary",
            "suite_order": ["synthetic_test", "pii_bench_formal", "pii_bench_chat"],
            "suites": suites,
            "aggregate": {
                "document_count": totals["documents"],
                "gold_span_count": totals["gold"],
                "predicted_span_count": totals["predicted"],
                "strict_micro_pooled": _strict_counts(totals["tp"], totals["fp"], totals["fn"]),
            },
            "system_identity": {
                "selected_seed": 42,
                "attention_mode": "full",
                "training": {
                    "manifest_sha256": training["manifest_sha256"],
                    "manifest_file_sha256": training_file_sha256,
                },
            },
        }
    )
    evaluation = _seal(
        {
            "schema_version": 1,
            "artifact_type": "combined_release_evaluation_report",
            "release_decision": "passed",
            "release_scope": "community_research_release_candidate",
            "production_ready": False,
            "selected_seed": 42,
            "quality_gate": {"status": "passed"},
            "seeds": [
                {
                    "seed": seed,
                    "quality_gate_passed": True,
                    "metrics": {
                        "strict_micro_f1": score,
                        "strict_macro_f1": score - 0.01,
                        "tier0_min_label_recall": score - 0.02,
                        "tier1_min_label_recall": score - 0.03,
                    },
                }
                for seed, score in ((13, 0.91), (42, 0.89), (97, 0.90))
            ],
            "validation_gate": {
                "file_sha256": hashlib.sha256(b"validation-file").hexdigest(),
                "manifest_sha256": hashlib.sha256(b"validation-manifest").hexdigest(),
            },
            "frozen_system_evaluation": frozen,
            "frozen_system_evaluation_file_sha256": hashlib.sha256(b"frozen-file").hexdigest(),
            "coverage": {
                "measured": ["fixture_public_evaluation"],
                "not_measured": ["fixture_private_evaluation", "fixture_long_documents"],
            },
        }
    )

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "components": [{"name": "dep-a"}, {"name": "dep-b"}],
    }
    sbom_sha256 = _file_hash(sbom)
    dependency_scan = {
        "schema_version": 1,
        "scanner": "fixture-auditor 9",
        "generated_at": "2026-07-11T14:43:10Z",
        "scan_complete": True,
        "sbom_sha256": sbom_sha256,
        "dependency_count": 7,
        "findings": [],
    }
    package_files = {
        "evaluation_report.json": _file_hash(evaluation),
        "training_manifest.json": training_file_sha256,
        "data_provenance.json": _file_hash(data_provenance),
        "teacher_provenance.json": _file_hash(teacher_provenance),
        "sbom.cdx.json": sbom_sha256,
    }
    package_binding = {
        "checksums_sha256": hashlib.sha256(b"fixture-checksums").hexdigest(),
        "file_count": len(package_files) + 1,
        "verified_file_count": len(package_files),
        "files": package_files,
    }
    dependency_scan_sha256 = _file_hash(dependency_scan)
    release_gate = _seal(
        {
            "schema_version": 2,
            "status": "PASS",
            "blocker_count": 0,
            "artifact_identity": {
                "checksums_sha256": package_binding["checksums_sha256"],
                "package_file_count": package_binding["file_count"],
                "sbom_sha256": sbom_sha256,
                "source_registry_sha256": registry["sha256"],
                "dependency_scan_sha256": dependency_scan_sha256,
                "dependency_exceptions_sha256": None,
            },
            "gates": [
                {
                    "name": name,
                    "passed": True,
                    "issues": [],
                    "details": (
                        {"file_count": package_binding["file_count"]}
                        if name == "file_set"
                        else {"verified_files": package_binding["verified_file_count"]}
                        if name == "checksums"
                        else {"component_count": len(sbom["components"])}
                        if name == "sbom"
                        else {"used_exceptions": []}
                        if name == "dependencies"
                        else {}
                    ),
                }
                for name in _GATES
            ],
        }
    )
    return {
        "evaluation": evaluation,
        "training": training,
        "data_provenance": data_provenance,
        "teacher_provenance": teacher_provenance,
        "dependency_scan": dependency_scan,
        "release_gate": release_gate,
        "sbom": sbom,
        "source_file_sha256": {
            "evaluation": _file_hash(evaluation),
            "training": training_file_sha256,
            "data_provenance": _file_hash(data_provenance),
            "teacher_provenance": _file_hash(teacher_provenance),
            "dependency_scan": dependency_scan_sha256,
            "release_gate": _file_hash(release_gate),
            "sbom": sbom_sha256,
        },
        "package_binding": package_binding,
    }


def test_report_is_dynamic_source_bound_and_public_safe() -> None:
    artifact = build_artifact(**_inputs())

    assert artifact["surface"] == "report"
    assert len(artifact["manifest"]["blocks"]) >= 10
    assert len(artifact["manifest"]["charts"]) == 2
    assert len(artifact["manifest"]["tables"]) >= 4
    assert artifact["snapshot"]["datasets"]["synthetic_risk"] == [{"fpr": 1 / 3}]
    assert artifact["snapshot"]["datasets"]["training_identity"] == [
        {"source_id": "fixture_base_model", "manifest_field": "base_source_id"}
    ]
    data_row = artifact["snapshot"]["datasets"]["data_sources"][0]
    teacher_row = artifact["snapshot"]["datasets"]["teacher_sources"][0]
    assert data_row["role"] == "fixture_deterministic_generator"
    assert data_row["license"] == "CC-BY-4.0"
    assert teacher_row["role"] == "fixture_reviewed_candidate_role"
    assert teacher_row["license"] == "Fixture-Research-License"

    source_rows = artifact["manifest"]["sources"]
    assert source_rows == artifact["sources"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) for source in source_rows)
    assert {source["id"] for source in source_rows} >= {
        "combined_evaluation",
        "frozen_evaluation",
        "validation_gate",
        "training_manifest",
        "data_provenance",
        "teacher_provenance",
        "release_gate",
        "dependency_scan",
        "sbom",
        "checksums",
        "source_registry",
    }
    sources_by_id = {source["id"]: source for source in source_rows}
    for item in (
        *artifact["manifest"]["cards"],
        *artifact["manifest"]["charts"],
        *artifact["manifest"]["tables"],
    ):
        source = sources_by_id[item["sourceId"]]
        assert source["query"]["engine"] == "sqlite"
        assert source["query"]["sql"].startswith("WITH ")
        assert source["query"]["metric_definitions"]
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert "81.48%" not in serialized
    assert "23.83%" not in serialized
    assert "33.33%" in serialized
    assert "12 篇文档" in serialized
    assert "复核 11 个候选" in serialized
    assert '"raw_text"' not in serialized
    assert '"doc_id"' not in serialized
    assert "/data1/" not in serialized


def test_report_refuses_tampered_or_failed_evaluation() -> None:
    inputs = _inputs()
    inputs["evaluation"]["release_decision"] = "blocked"
    with pytest.raises(ReleaseReportError, match="self-hash"):
        build_artifact(**inputs)

    inputs = _inputs()
    inputs["evaluation"] = _seal(inputs["evaluation"] | {"release_decision": "blocked"})
    with pytest.raises(ReleaseReportError, match="non-production community research RC"):
        build_artifact(**inputs)


@pytest.mark.parametrize("name", ["training", "data_provenance", "teacher_provenance"])
def test_report_refuses_tampered_self_hashed_lineage(name: str) -> None:
    inputs = _inputs()
    inputs[name]["tampered"] = True
    with pytest.raises(ReleaseReportError, match="self-hash"):
        build_artifact(**inputs)


def test_report_refuses_data_or_teacher_training_binding_mismatch() -> None:
    inputs = _inputs()
    inputs["data_provenance"]["training_manifest_sha256"] = "0" * 64
    inputs["data_provenance"] = _seal(inputs["data_provenance"])
    with pytest.raises(ReleaseReportError, match="data provenance/training binding"):
        build_artifact(**inputs)

    inputs = _inputs()
    inputs["teacher_provenance"]["training_manifest_file_sha256"] = "1" * 64
    inputs["teacher_provenance"] = _seal(inputs["teacher_provenance"])
    with pytest.raises(ReleaseReportError, match="teacher provenance/training binding"):
        build_artifact(**inputs)


def test_report_never_overrides_provenance_permission_or_teacher_role() -> None:
    inputs = _inputs()
    inputs["data_provenance"]["sources"][0]["public_weight_training_allowed"] = False
    inputs["data_provenance"] = _seal(inputs["data_provenance"])
    with pytest.raises(ReleaseReportError, match="not admitted"):
        build_artifact(**inputs)

    inputs = _inputs()
    inputs["teacher_provenance"]["teachers"][0]["span_teacher"] = True
    inputs["teacher_provenance"] = _seal(inputs["teacher_provenance"])
    with pytest.raises(ReleaseReportError, match="teacher safety role"):
        build_artifact(**inputs)


def test_report_refuses_stale_dependency_or_package_binding() -> None:
    inputs = _inputs()
    inputs["dependency_scan"]["sbom_sha256"] = "2" * 64
    with pytest.raises(ReleaseReportError, match="stale for the current packaged SBOM"):
        build_artifact(**inputs)

    inputs = _inputs()
    inputs["package_binding"]["files"]["evaluation_report.json"] = "3" * 64
    with pytest.raises(ReleaseReportError, match="stale or mismatched evaluation_report"):
        build_artifact(**inputs)


def test_report_refuses_stale_or_incomplete_gate_receipt() -> None:
    inputs = _inputs()
    file_set = next(gate for gate in inputs["release_gate"]["gates"] if gate["name"] == "file_set")
    file_set["details"]["file_count"] += 1
    inputs["release_gate"] = _seal(inputs["release_gate"])
    with pytest.raises(ReleaseReportError, match="stale for the current package file count"):
        build_artifact(**inputs)

    inputs = _inputs()
    inputs["release_gate"]["gates"].pop()
    inputs["release_gate"] = _seal(inputs["release_gate"])
    with pytest.raises(ReleaseReportError, match="gate set is incomplete"):
        build_artifact(**inputs)


def test_report_refuses_tampered_or_mismatched_gate_identity() -> None:
    inputs = _inputs()
    inputs["release_gate"]["artifact_identity"]["sbom_sha256"] = "4" * 64
    with pytest.raises(ReleaseReportError, match="self-hash"):
        build_artifact(**inputs)

    inputs = _inputs()
    inputs["release_gate"]["artifact_identity"]["dependency_scan_sha256"] = "5" * 64
    inputs["release_gate"] = _seal(inputs["release_gate"])
    with pytest.raises(ReleaseReportError, match="stale for the current package inputs"):
        build_artifact(**inputs)


def test_report_binds_optional_dependency_exceptions() -> None:
    inputs = _inputs()
    exceptions_sha256 = hashlib.sha256(b"reviewed-empty-exceptions").hexdigest()
    inputs["source_file_sha256"]["dependency_exceptions"] = exceptions_sha256
    inputs["release_gate"]["artifact_identity"]["dependency_exceptions_sha256"] = exceptions_sha256
    inputs["release_gate"] = _seal(inputs["release_gate"])
    inputs["source_file_sha256"]["release_gate"] = _file_hash(inputs["release_gate"])

    artifact = build_artifact(**inputs)
    exception_source = next(
        source for source in artifact["sources"] if source["id"] == "dependency_exceptions"
    )
    assert exception_source["sha256"] == exceptions_sha256

    inputs["source_file_sha256"]["dependency_exceptions"] = "6" * 64
    with pytest.raises(ReleaseReportError, match="stale for dependency exceptions"):
        build_artifact(**inputs)


def test_report_recomputes_metrics_instead_of_trusting_point_values() -> None:
    inputs = _inputs()
    frozen = inputs["evaluation"]["frozen_system_evaluation"]
    frozen["suites"]["synthetic_test"]["metrics"]["strict_micro"]["f1"] = 0.99
    frozen = _seal(frozen)
    inputs["evaluation"]["frozen_system_evaluation"] = frozen
    inputs["evaluation"] = _seal(inputs["evaluation"])
    with pytest.raises(ReleaseReportError, match="do not recompute"):
        build_artifact(**inputs)


def test_validate_package_dir_detects_current_file_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    first = artifact / "one.json"
    second = artifact / "two.txt"
    first.write_text('{"ok":true}\n', encoding="utf-8")
    second.write_text("stable\n", encoding="utf-8")
    checksums = artifact / "checksums.txt"
    checksums.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in (first, second)
        )
        + "\n",
        encoding="utf-8",
    )

    binding = validate_package_dir(artifact)
    assert binding["file_count"] == 3
    assert binding["verified_file_count"] == 2

    second.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ReleaseReportError, match="checksum mismatch"):
        validate_package_dir(artifact)


def test_cli_reads_and_binds_the_current_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs()
    artifact_dir = tmp_path / "hf-model"
    artifact_dir.mkdir()
    packaged_documents = {
        "evaluation_report.json": inputs["evaluation"],
        "training_manifest.json": inputs["training"],
        "data_provenance.json": inputs["data_provenance"],
        "teacher_provenance.json": inputs["teacher_provenance"],
        "sbom.cdx.json": inputs["sbom"],
    }
    for name, document in packaged_documents.items():
        (artifact_dir / name).write_text(
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (artifact_dir / "checksums.txt").write_text(
        "\n".join(
            f"{digest}  {name}"
            for name, digest in sorted(inputs["package_binding"]["files"].items())
        )
        + "\n",
        encoding="utf-8",
    )
    inputs["release_gate"]["artifact_identity"]["checksums_sha256"] = hashlib.sha256(
        (artifact_dir / "checksums.txt").read_bytes()
    ).hexdigest()
    inputs["release_gate"] = _seal(inputs["release_gate"])

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    dependency_path = audit_dir / "dependency-scan.json"
    gate_path = audit_dir / "release-gate-report.json"
    dependency_path.write_text(
        json.dumps(inputs["dependency_scan"], ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(inputs["release_gate"], ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "artifact.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_release_report.py",
            "--evaluation-report",
            str(artifact_dir / "evaluation_report.json"),
            "--training-manifest",
            str(artifact_dir / "training_manifest.json"),
            "--data-provenance",
            str(artifact_dir / "data_provenance.json"),
            "--teacher-provenance",
            str(artifact_dir / "teacher_provenance.json"),
            "--dependency-scan",
            str(dependency_path),
            "--release-gate",
            str(gate_path),
            "--artifact-dir",
            str(artifact_dir),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["snapshot"]["status"] == "ready"
    assert (
        next(source for source in generated["sources"] if source["id"] == "checksums")["sha256"]
        == hashlib.sha256((artifact_dir / "checksums.txt").read_bytes()).hexdigest()
    )
