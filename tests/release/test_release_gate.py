from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conftest import ReleaseFixture, rewrite_checksums, run_script, write_json


def _gate(
    repository_root: Path,
    fixture: ReleaseFixture,
    report: Path,
    *extra: str,
):
    return run_script(
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
    )


def _issue_codes(report: Path) -> set[str]:
    document = json.loads(report.read_text(encoding="utf-8"))
    return {issue["code"] for gate in document["gates"] for issue in gate["issues"]}


def test_gate_detects_checksum_tampering(
    built_release: ReleaseFixture, repository_root: Path, tmp_path: Path
) -> None:
    with (built_release.artifact / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    report = tmp_path / "gate.json"
    result = _gate(repository_root, built_release, report)
    assert result.returncode == 1
    assert "RC_BLOCKED_CHECKSUM_MISMATCH" in _issue_codes(report)


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
