from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

from scripts import release_gate
from tests.release.conftest import ReleaseFixture, rewrite_checksums, run_script, write_json


def _gate(
    repository_root: Path,
    fixture: ReleaseFixture,
    report: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return cast(
        subprocess.CompletedProcess[str],
        run_script(
            repository_root,
            "release_gate.py",
            "--artifact",
            str(fixture.artifact),
            "--source-registry",
            str(fixture.registry),
            "--dependency-scan",
            str(fixture.dependency_scan),
            "--dependency-exceptions",
            str(fixture.dependency_exceptions),
            "--json-output",
            str(report),
            *extra,
        ),
    )


def _issue_codes(report: Path) -> set[str]:
    document = json.loads(report.read_text(encoding="utf-8"))
    return {issue["code"] for gate in document["gates"] for issue in gate["issues"]}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _artifact_snapshot_hash(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return _canonical_hash(entries)


def _assert_receipt_self_hash(document: dict[str, object]) -> None:
    unsigned = dict(document)
    claimed = unsigned.pop("manifest_sha256")
    assert claimed == _canonical_hash(unsigned)


def _upgrade_artifact_to_schema4_lineage(
    fixture: ReleaseFixture, training_source_ids: list[str]
) -> None:
    training_path = fixture.artifact / "training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    strategy = "base_causal_lm_v1"
    recipe = {"initialization_strategy": strategy}
    training.update(
        {
            "schema_version": 4,
            "training_source_ids": training_source_ids,
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(
                json.dumps(
                    recipe,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "initialization": {"strategy": strategy},
            "datasets": {
                "train": {
                    "summary": {
                        "sources": [{"source_id": "synthetic-data"}],
                    }
                },
                "validation": {
                    "summary": {
                        "sources": [{"source_id": "synthetic-data"}],
                    }
                },
            },
        }
    )
    training.pop("source_ids", None)
    training.pop("manifest_sha256", None)
    encoded = json.dumps(training, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    training["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    write_json(training_path, training)
    rewrite_checksums(fixture.artifact)


def test_gate_detects_checksum_tampering(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    with (built_release.artifact / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_CHECKSUM_MISMATCH" in _issue_codes(report)
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["status"] == "RC_BLOCKED"
    assert (
        document["artifact_identity"]["checksums_sha256"]
        == hashlib.sha256((built_release.artifact / "checksums.txt").read_bytes()).hexdigest()
    )
    _assert_receipt_self_hash(document)


def test_gate_receipt_v2_binds_package_and_evidence_inputs(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)

    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(report.read_text(encoding="utf-8"))
    identity = document["artifact_identity"]
    assert document["schema_version"] == 2
    assert identity == {
        "artifact_snapshot_sha256": _artifact_snapshot_hash(built_release.artifact),
        "checksums_sha256": hashlib.sha256(
            (built_release.artifact / "checksums.txt").read_bytes()
        ).hexdigest(),
        "dependency_exceptions_sha256": hashlib.sha256(
            built_release.dependency_exceptions.read_bytes()
        ).hexdigest(),
        "dependency_scan_sha256": hashlib.sha256(
            built_release.dependency_scan.read_bytes()
        ).hexdigest(),
        "package_file_count": sum(path.is_file() for path in built_release.artifact.rglob("*")),
        "sbom_sha256": hashlib.sha256(
            (built_release.artifact / "sbom.cdx.json").read_bytes()
        ).hexdigest(),
        "source_registry_sha256": hashlib.sha256(built_release.registry.read_bytes()).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(identity)
    receipt_gate = next(gate for gate in document["gates"] if gate["name"] == "receipt_identity")
    assert receipt_gate["passed"] is True
    _assert_receipt_self_hash(document)


def test_gate_rejects_system_metrics_attributed_to_model_index(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    model_index_path = built_release.artifact / "model-index.yml"
    model_index = model_index_path.read_text(encoding="utf-8")
    assert "Model Raw " in model_index
    model_index_path.write_text(model_index.replace("Model Raw ", "System "), encoding="utf-8")
    rewrite_checksums(built_release.artifact)

    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)

    assert result.returncode == 1
    assert "RC_BLOCKED_MODEL_INDEX_ATTRIBUTION" in _issue_codes(report)


def test_gate_rejects_unknown_generator_defect_semantics(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    provenance_path = built_release.artifact / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["lineage"] = {
        "dataset_id": "unknown-dataset-id",
        "generator_defect_retired": True,
    }
    write_json(provenance_path, provenance)
    rewrite_checksums(built_release.artifact)

    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)

    assert result.returncode == 1
    assert "RC_BLOCKED_ARTIFACT_LINEAGE" in _issue_codes(report)


def test_gate_rejects_non_boolean_release_eligibility(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    provenance_path = built_release.artifact / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["release_eligible"] = None
    write_json(provenance_path, provenance)
    rewrite_checksums(built_release.artifact)

    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)

    assert result.returncode == 1
    assert "RC_BLOCKED_ARTIFACT_LINEAGE" in _issue_codes(report)


def test_gate_receipt_identity_changes_with_package_and_scan_content(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    baseline_report = tmp_path / "baseline.json"
    baseline_result = _gate(repository_root, built_release, baseline_report)
    assert baseline_result.returncode == 0, baseline_result.stdout + baseline_result.stderr
    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    _assert_receipt_self_hash(baseline)

    with (built_release.artifact / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\nAdditional reviewed release note.\n")
    rewrite_checksums(built_release.artifact)
    package_report = tmp_path / "package-change.json"
    package_result = _gate(repository_root, built_release, package_report)
    assert package_result.returncode == 0, package_result.stdout + package_result.stderr
    package_changed = json.loads(package_report.read_text(encoding="utf-8"))
    _assert_receipt_self_hash(package_changed)
    assert (
        package_changed["artifact_identity"]["checksums_sha256"]
        != baseline["artifact_identity"]["checksums_sha256"]
    )
    assert (
        package_changed["artifact_identity"]["dependency_scan_sha256"]
        == baseline["artifact_identity"]["dependency_scan_sha256"]
    )

    dependency_scan = json.loads(built_release.dependency_scan.read_text(encoding="utf-8"))
    dependency_scan["generated_at"] = "2026-07-11T01:00:00Z"
    write_json(built_release.dependency_scan, dependency_scan)
    scan_report = tmp_path / "scan-change.json"
    scan_result = _gate(repository_root, built_release, scan_report)
    assert scan_result.returncode == 0, scan_result.stdout + scan_result.stderr
    scan_changed = json.loads(scan_report.read_text(encoding="utf-8"))
    _assert_receipt_self_hash(scan_changed)
    assert (
        scan_changed["artifact_identity"]["checksums_sha256"]
        == package_changed["artifact_identity"]["checksums_sha256"]
    )
    assert (
        scan_changed["artifact_identity"]["dependency_scan_sha256"]
        != package_changed["artifact_identity"]["dependency_scan_sha256"]
    )


def test_gate_blocks_a_b_artifact_root_replacement_and_receipt_stays_on_snapshot(
    built_release: ReleaseFixture,
    monkeypatch,
    tmp_path: Path,
) -> None:
    original = built_release.artifact
    replacement = tmp_path / "replacement-b"
    displaced = tmp_path / "displaced-a"
    shutil.copytree(original, replacement)
    with (replacement / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\nA different, independently checksummed B directory.\n")
    rewrite_checksums(replacement)
    original_checksums_sha256 = hashlib.sha256(
        (original / "checksums.txt").read_bytes()
    ).hexdigest()

    real_run_gates = release_gate.run_gates

    def swap_root_after_gate_reads(
        snapshot_args: argparse.Namespace,
    ) -> list[release_gate.GateResult]:
        results = real_run_gates(snapshot_args)
        original.rename(displaced)
        replacement.rename(original)
        return results

    monkeypatch.setattr(release_gate, "run_gates", swap_root_after_gate_reads)
    args = argparse.Namespace(
        artifact=original,
        source_registry=built_release.registry,
        canary_allowlist=None,
        dependency_scan=built_release.dependency_scan,
        dependency_exceptions=built_release.dependency_exceptions,
        v2_contract_sha256=None,
        v2_selected_receipt_sha256=None,
        json_output=None,
    )

    results, identity, execution_error = release_gate.execute_gates(args)

    assert execution_error is False
    assert identity["checksums_sha256"] == original_checksums_sha256
    assert identity["artifact_snapshot_sha256"] == _artifact_snapshot_hash(displaced)
    snapshot_gate = next(
        result for result in results if result.name == "artifact_snapshot_integrity"
    )
    assert snapshot_gate.passed is False
    assert {issue.code for issue in snapshot_gate.issues} == {
        "RC_BLOCKED_ARTIFACT_ROOT_REPLACED"
    }


def test_gate_blocks_in_place_source_mutation_without_changing_private_snapshot(
    built_release: ReleaseFixture,
    monkeypatch,
) -> None:
    original = built_release.artifact
    original_snapshot_sha256 = _artifact_snapshot_hash(original)
    original_checksums_sha256 = hashlib.sha256(
        (original / "checksums.txt").read_bytes()
    ).hexdigest()
    real_run_gates = release_gate.run_gates

    def mutate_source_after_gate_reads(
        snapshot_args: argparse.Namespace,
    ) -> list[release_gate.GateResult]:
        results = real_run_gates(snapshot_args)
        with (original / "README.md").open("a", encoding="utf-8") as stream:
            stream.write("\nConcurrent in-place source mutation.\n")
        rewrite_checksums(original)
        return results

    monkeypatch.setattr(release_gate, "run_gates", mutate_source_after_gate_reads)
    args = argparse.Namespace(
        artifact=original,
        source_registry=built_release.registry,
        canary_allowlist=None,
        dependency_scan=built_release.dependency_scan,
        dependency_exceptions=built_release.dependency_exceptions,
        v2_contract_sha256=None,
        v2_selected_receipt_sha256=None,
        json_output=None,
    )

    results, identity, execution_error = release_gate.execute_gates(args)

    assert execution_error is False
    assert identity["checksums_sha256"] == original_checksums_sha256
    assert identity["artifact_snapshot_sha256"] == original_snapshot_sha256
    snapshot_gate = next(
        result for result in results if result.name == "artifact_snapshot_integrity"
    )
    assert snapshot_gate.passed is False
    assert {issue.code for issue in snapshot_gate.issues} == {
        "RC_BLOCKED_ARTIFACT_SOURCE_MUTATED"
    }


def test_gate_fails_closed_when_source_changes_during_snapshot_construction(
    built_release: ReleaseFixture,
    monkeypatch,
) -> None:
    original = built_release.artifact
    real_copy_tree = release_gate._copy_fd_tree
    mutated = False

    def mutate_after_copy(source_fd: int, destination: Path) -> None:
        nonlocal mutated
        real_copy_tree(source_fd, destination)
        if not mutated:
            mutated = True
            with (original / "README.md").open("a", encoding="utf-8") as stream:
                stream.write("\nMutation during snapshot construction.\n")
            rewrite_checksums(original)

    monkeypatch.setattr(release_gate, "_copy_fd_tree", mutate_after_copy)
    args = argparse.Namespace(
        artifact=original,
        source_registry=built_release.registry,
        canary_allowlist=None,
        dependency_scan=built_release.dependency_scan,
        dependency_exceptions=built_release.dependency_exceptions,
        v2_contract_sha256=None,
        v2_selected_receipt_sha256=None,
        json_output=None,
    )

    results, _, execution_error = release_gate.execute_gates(args)

    assert execution_error is True
    snapshot_gate = next(
        result for result in results if result.name == "artifact_snapshot_integrity"
    )
    assert snapshot_gate.passed is False
    assert {issue.code for issue in snapshot_gate.issues} == {
        "RC_BLOCKED_ARTIFACT_SNAPSHOT"
    }


def test_gate_refuses_to_write_receipt_inside_artifact_tree(
    built_release: ReleaseFixture,
    repository_root: Path,
) -> None:
    unsafe_report = built_release.artifact / "gate-receipt.json"

    result = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(built_release.artifact),
        "--source-registry",
        str(built_release.registry),
        "--dependency-scan",
        str(built_release.dependency_scan),
        "--dependency-exceptions",
        str(built_release.dependency_exceptions),
        "--json-output",
        str(unsafe_report),
    )

    assert result.returncode == 2
    assert "receipt output must be outside" in result.stderr
    assert not unsafe_report.exists()


def test_gate_writes_self_hashed_receipt_for_missing_artifact(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    report = tmp_path / "missing-artifact.json"
    missing_artifact = tmp_path / "does-not-exist"
    result = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(missing_artifact),
        "--source-registry",
        str(built_release.registry),
        "--dependency-scan",
        str(built_release.dependency_scan),
        "--dependency-exceptions",
        str(built_release.dependency_exceptions),
        "--json-output",
        str(report),
    )

    assert result.returncode == 1
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["status"] == "RC_BLOCKED"
    assert document["artifact_identity"]["checksums_sha256"] is None
    assert document["artifact_identity"]["sbom_sha256"] is None
    assert document["artifact_identity"]["package_file_count"] == 0
    _assert_receipt_self_hash(document)


def test_gate_allows_unconfigured_dependency_exceptions_with_null_identity(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    report = tmp_path / "no-exceptions.json"
    result = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(built_release.artifact),
        "--source-registry",
        str(built_release.registry),
        "--dependency-scan",
        str(built_release.dependency_scan),
        "--json-output",
        str(report),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["artifact_identity"]["dependency_exceptions_sha256"] is None
    _assert_receipt_self_hash(document)


def test_gate_detects_model_manifest_mismatch(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    tokenizer_path = built_release.artifact / "tokenizer_config.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model_max_length"] = 64
    write_json(tokenizer_path, tokenizer)
    rewrite_checksums(built_release.artifact)
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_OUTPUT_ARTIFACT_BINDING" in _issue_codes(report)


def test_gate_blocks_staged_recipe_without_initialization_audit(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    training_path = built_release.artifact / "training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["schema_version"] = 3
    training["recipe"] = {"initialization_strategy": "verified_token_classifier_to_full_v1"}
    training.pop("initialization", None)
    training.pop("manifest_sha256", None)
    encoded = json.dumps(training, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    training["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    write_json(training_path, training)
    rewrite_checksums(built_release.artifact)

    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_INITIALIZATION_AUDIT" in _issue_codes(report)


def test_gate_accepts_schema3_jpt_to_full_staged_provenance(
    staged_jpt_built_release: ReleaseFixture,
    repository_root: Path,
    tmp_path: Path,
) -> None:
    report = tmp_path / "gate.json"
    result = _gate(repository_root, staged_jpt_built_release, report)
    document = json.loads(report.read_text(encoding="utf-8"))
    binding = next(
        gate for gate in document["gates"] if gate["name"] == "training_artifact_binding"
    )
    manifest = json.loads(
        (staged_jpt_built_release.artifact / "training_manifest.json").read_text(encoding="utf-8")
    )
    initialization = manifest["initialization"]

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 0 blocker(s)" in result.stdout
    assert document["status"] == "PASS"
    assert document["blocker_count"] == 0
    assert _issue_codes(report) == set()
    assert binding["passed"] is True
    assert binding["issues"] == []

    assert manifest["schema_version"] == 3
    assert manifest["attention_mode"] == "full"
    assert manifest["recipe"]["resume"] is False
    assert (
        manifest["recipe"]["initialization_strategy"]
        == initialization["strategy"]
        == "verified_token_classifier_to_full_v1"
    )
    assert initialization["source_attention_mode"] == "jpt"
    assert initialization["source_manifest_schema_version"] == 3
    assert initialization["base_source_id"] == manifest["base_source_id"]
    assert initialization["base_config_sha256"] == manifest["base_checkpoint"]["config_sha256"]
    assert initialization["base_weights_sha256"] == manifest["base_checkpoint"]["weights_sha256"]
    assert initialization["taxonomy_version"] == manifest["taxonomy_version"]
    assert initialization["label_schema_sha256"] == manifest["label_schema_sha256"]
    assert (
        initialization["tokenizer_effective_contract_sha256"]
        == manifest["tokenizer"]["effective_contract_sha256"]
    )
    assert initialization["train_sha256"] == manifest["datasets"]["train"]["sha256"]
    assert initialization["validation_sha256"] == manifest["datasets"]["validation"]["sha256"]
    assert str(staged_jpt_built_release.checkpoint) not in json.dumps(manifest)


def test_gate_accepts_schema4_canonical_training_source_lineage(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    _upgrade_artifact_to_schema4_lineage(
        built_release,
        ["base", "synthetic-data", "teacher"],
    )
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _issue_codes(report) == set()


def test_gate_blocks_schema4_lineage_missing_direct_dataset_source(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    _upgrade_artifact_to_schema4_lineage(built_release, ["base", "teacher"])
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)

    assert result.returncode == 1
    assert "RC_BLOCKED_TRAINING_SOURCE_LINEAGE" in _issue_codes(report)


def test_gate_blocks_incomplete_three_seed_evidence(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    evaluation_path = built_release.artifact / "evaluation_report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["seeds"] = evaluation["seeds"][:1]
    write_json(evaluation_path, evaluation)
    rewrite_checksums(built_release.artifact)
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_INSUFFICIENT_SEEDS" in _issue_codes(report)
    assert "RC_BLOCKED" in result.stdout


def test_gate_blocks_evaluation_only_training_source(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    training_path = built_release.artifact / "training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["training_pool"] = "evaluation_only"
    write_json(training_path, training)
    rewrite_checksums(built_release.artifact)
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_EVALUATION_ONLY_TRAINING" in _issue_codes(report)


def test_gate_blocks_non_full_or_unbound_tokenizer_contract(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    training_path = built_release.artifact / "training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["attention_mode"] = "causal"
    training["tokenizer"]["effective"]["boundary_mode"] = "missing"
    write_json(training_path, training)
    tokenizer_path = built_release.artifact / "tokenizer_config.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer.pop("pii_zh_boundary_mode")
    write_json(tokenizer_path, tokenizer)
    rewrite_checksums(built_release.artifact)
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert {
        "RC_BLOCKED_ATTENTION_MODE",
        "RC_BLOCKED_BOUNDARY_TOKENIZER",
        "RC_BLOCKED_TRAINING_TOKENIZER_BINDING",
    } <= _issue_codes(report)


def test_gate_allows_explicitly_unused_evaluation_reference(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    teacher_path = built_release.artifact / "teacher_provenance.json"
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    teacher["teachers"].append(
        {
            "pool": "evaluation_only",
            "source_id": "not-sampled-evaluation-source",
            "used_for_training": False,
        }
    )
    write_json(teacher_path, teacher)
    rewrite_checksums(built_release.artifact)
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_blocks_sampled_source_without_public_weight_license(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    registry = built_release.registry.read_text(encoding="utf-8")
    registry = registry.replace(
        "  - id: synthetic-data\n"
        "    kind: synthetic_dataset\n"
        "    declared_license: Apache-2.0\n"
        "    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        "    review_status: approved\n"
        "    public_weight_training_allowed: true\n",
        "  - id: synthetic-data\n"
        "    kind: synthetic_dataset\n"
        "    declared_license: Apache-2.0\n"
        "    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        "    review_status: pending_legal_review\n"
        "    public_weight_training_allowed: false\n",
    )
    built_release.registry.write_text(registry, encoding="utf-8")
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert {
        "RC_BLOCKED_SOURCE_NOT_LICENSED",
        "RC_BLOCKED_SOURCE_REVIEW",
    } <= _issue_codes(report)


def test_gate_requires_completed_dependency_scan(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    write_json(
        built_release.dependency_scan,
        {
            "findings": [],
            "generated_at": "2026-07-11T00:00:00Z",
            "scan_complete": False,
            "scanner": "synthetic-test-scanner 1",
            "sbom_sha256": hashlib.sha256(
                (built_release.artifact / "sbom.cdx.json").read_bytes()
            ).hexdigest(),
        },
    )
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_DEPENDENCY_SCAN_INCOMPLETE" in _issue_codes(report)


def test_narrow_unexpired_dependency_exception_is_recorded(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    write_json(
        built_release.dependency_scan,
        {
            "findings": [
                {"component": "tiny-dependency", "id": "CVE-SYNTHETIC-1", "severity": "high"}
            ],
            "generated_at": "2026-07-11T00:00:00Z",
            "scan_complete": True,
            "scanner": "synthetic-test-scanner 1",
            "sbom_sha256": hashlib.sha256(
                (built_release.artifact / "sbom.cdx.json").read_bytes()
            ).hexdigest(),
        },
    )
    write_json(
        built_release.dependency_exceptions,
        {
            "exceptions": [
                {
                    "approved_by": "release-security-reviewer",
                    "compensating_controls": ["The affected optional code path is disabled."],
                    "component": "tiny-dependency",
                    "expires_on": "2099-01-01",
                    "id": "EX-SYNTHETIC-1",
                    "justification": "Synthetic fixture exercises a narrow reviewed exception.",
                    "vulnerability_id": "CVE-SYNTHETIC-1",
                }
            ],
            "schema_version": 1,
        },
    )
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(report.read_text(encoding="utf-8"))
    dependencies = next(gate for gate in document["gates"] if gate["name"] == "dependencies")
    assert dependencies["details"]["used_exceptions"] == ["EX-SYNTHETIC-1"]
