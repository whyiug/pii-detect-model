from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class ReleaseFixture:
    repository_root: Path
    checkpoint: Path
    evidence: Path
    model_card: Path
    registry: Path
    dependency_scan: Path
    dependency_exceptions: Path
    artifact: Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tiny_safetensors(path: Path) -> None:
    header = json.dumps(
        {"weight": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))


def rewrite_checksums(artifact: Path) -> None:
    lines: list[str] = []
    for path in sorted(artifact.iterdir()):
        if path.is_file() and path.name != "checksums.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}")
    (artifact / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def rewrite_training_manifest_binding(checkpoint: Path, evidence: Path) -> None:
    names = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    files = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in names
        if (checkpoint / name).is_file()
    }
    weight_hashes = {
        name: digest for name, digest in files.items() if name.endswith(".safetensors")
    }
    manifest_path = evidence / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256", None)
    manifest["status"] = "completed"
    manifest["output_artifact"] = {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": _canonical_hash(weight_hashes),
        "artifact_files_combined_sha256": _canonical_hash(files),
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    write_json(manifest_path, manifest)


def run_script(
    repository_root: Path, name: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repository_root / "scripts" / name), *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def release_fixture(tmp_path: Path, repository_root: Path) -> ReleaseFixture:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(
        checkpoint / "config.json",
        {
            "architectures": ["Qwen3BiForTokenClassification"],
            "architecture_version": "qwen3_bi_token_cls_v1",
            "auto_map": {
                "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
                "AutoModelForTokenClassification": (
                    "modeling_qwen3_bi.Qwen3BiForTokenClassification"
                ),
            },
            "hidden_size": 4,
            "id2label": {"0": "O", "1": "B-PERSON_NAME"},
            "label2id": {"B-PERSON_NAME": 1, "O": 0},
            "model_type": "qwen3_bi",
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "pii_attention_mode": "full",
            "pii_release_eligible": True,
            "use_cache": False,
            "vocab_size": 8,
        },
    )
    write_tiny_safetensors(checkpoint / "model.safetensors")
    write_json(checkpoint / "tokenizer.json", {"model": {"type": "WordLevel", "vocab": {}}})
    write_json(
        checkpoint / "tokenizer_config.json",
        {"model_max_length": 32, "pii_zh_boundary_mode": "unicode_codepoint_v1"},
    )
    write_json(checkpoint / "special_tokens_map.json", {"unk_token": "[UNK]"})

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "taxonomy.yaml").write_text(
        "schema_version: 1\nversion: tiny-v1\n", encoding="utf-8"
    )
    write_json(evidence / "id2label.json", {"0": "O", "1": "B-PERSON_NAME"})
    write_json(evidence / "calibration.json", {"method": "temperature", "temperature": 1.0})
    (evidence / "thresholds.yaml").write_text("schema_version: 1\ndefault: 0.5\n", encoding="utf-8")
    write_json(
        evidence / "training_manifest.json",
        {
            "base_source_id": "base",
            "attention_mode": "full",
            "data_manifest_sha256": "1" * 64,
            "seed": 11,
            "source_ids": ["synthetic-data", "teacher"],
            "training_pool": "public_release_pool",
            "tokenizer": {"effective": {"boundary_mode": "unicode_codepoint_v1"}},
        },
    )
    write_json(
        evidence / "data_provenance.json",
        {
            "sources": [
                {
                    "pool": "public_release_pool",
                    "sample_count": 12,
                    "source_id": "synthetic-data",
                }
            ]
        },
    )
    write_json(
        evidence / "teacher_provenance.json",
        {"teachers": [{"source_id": "teacher", "used_for_training": True}]},
    )
    write_json(
        evidence / "evaluation_report.json",
        {
            "quality_gate": {
                "criteria": [
                    {
                        "name": "tiny synthetic release-test criterion",
                        "operator": ">=",
                        "passed": True,
                        "threshold": 0.5,
                        "value": 0.8,
                    }
                ],
                "status": "passed",
            },
            "release_decision": "passed",
            "seeds": [
                {"metrics": {"strict_span_f1": 0.80}, "quality_gate_passed": True, "seed": 11},
                {"metrics": {"strict_span_f1": 0.81}, "quality_gate_passed": True, "seed": 22},
                {"metrics": {"strict_span_f1": 0.82}, "quality_gate_passed": True, "seed": 33},
            ],
        },
    )
    (evidence / "model-index.yml").write_text(
        "model-index:\n"
        "  - name: tiny-synthetic-release-test\n"
        "    results:\n"
        "      - task: {type: token-classification}\n"
        "        dataset: {name: synthetic-release-test, type: synthetic}\n"
        "        metrics:\n"
        "          - {name: Strict Span F1, type: strict_span_f1, value: 0.8}\n",
        encoding="utf-8",
    )
    write_json(
        evidence / "sbom.cdx.json",
        {
            "bomFormat": "CycloneDX",
            "components": [{"name": "tiny-dependency", "type": "library", "version": "1"}],
            "specVersion": "1.6",
            "version": 1,
        },
    )
    rewrite_training_manifest_binding(checkpoint, evidence)

    model_card = tmp_path / "README.md"
    model_card.write_text(
        "---\nlanguage: [zh]\npipeline_tag: token-classification\nlicense: apache-2.0\n---\n"
        "# Tiny synthetic release test\n\n"
        "No real-person data is included. Contact security@example.invalid.\n",
        encoding="utf-8",
    )
    project_files = tmp_path / "project-files"
    project_files.mkdir()
    (project_files / "LICENSE").write_text(
        "Apache License 2.0 synthetic fixture\n", encoding="utf-8"
    )
    (project_files / "NOTICE").write_text("Synthetic release-test notice\n", encoding="utf-8")
    (project_files / "THIRD_PARTY_NOTICES.md").write_text(
        "# Third party notices\n\nTiny dependency test fixture.\n", encoding="utf-8"
    )
    (project_files / "SECURITY.md").write_text(
        "# Security\n\nUse the repository private security advisory form.\n", encoding="utf-8"
    )

    registry = tmp_path / "source_registry.yaml"
    registry.write_text(
        "schema_version: 1\n"
        "sources:\n"
        "  - id: base\n"
        "    kind: base_model\n"
        "    declared_license: Apache-2.0\n"
        "    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "    review_status: approved\n"
        "    public_weight_training_allowed: true\n"
        "  - id: synthetic-data\n"
        "    kind: synthetic_dataset\n"
        "    declared_license: Apache-2.0\n"
        "    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        "    review_status: approved\n"
        "    public_weight_training_allowed: true\n"
        "  - id: teacher\n"
        "    kind: self_hosted_teacher\n"
        "    declared_license: Apache-2.0\n"
        "    revision: cccccccccccccccccccccccccccccccccccccccc\n"
        "    review_status: approved\n"
        "    public_weight_training_allowed: true\n",
        encoding="utf-8",
    )
    dependency_scan = tmp_path / "dependency-scan.json"
    sbom_sha256 = hashlib.sha256((evidence / "sbom.cdx.json").read_bytes()).hexdigest()
    write_json(
        dependency_scan,
        {
            "findings": [],
            "generated_at": "2026-07-11T00:00:00Z",
            "scan_complete": True,
            "scanner": "synthetic-test-scanner 1",
            "sbom_sha256": sbom_sha256,
        },
    )
    dependency_exceptions = tmp_path / "dependency-exceptions.json"
    write_json(dependency_exceptions, {"exceptions": [], "schema_version": 1})
    artifact = tmp_path / "hf-model"
    return ReleaseFixture(
        repository_root=project_files,
        checkpoint=checkpoint,
        evidence=evidence,
        model_card=model_card,
        registry=registry,
        dependency_scan=dependency_scan,
        dependency_exceptions=dependency_exceptions,
        artifact=artifact,
    )


@pytest.fixture
def built_release(release_fixture: ReleaseFixture, repository_root: Path) -> ReleaseFixture:
    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )
    assert result.returncode == 0, result.stderr
    return release_fixture
