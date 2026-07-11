from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pii_zh.training.manifest import (
    canonical_json_hash,
    output_artifact_fingerprint,
    sha256_file,
    verify_training_manifest,
)


def _write_completed_manifest(checkpoint: Path) -> dict[str, object]:
    for name, content in {
        "config.json": "{}\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "special_tokens_map.json": "{}\n",
    }.items():
        (checkpoint / name).write_text(content, encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"safe-test-weights")
    manifest: dict[str, object] = {
        "schema_version": 3,
        "status": "completed",
        "seed": 42,
        "attention_mode": "full",
        "base_source_id": "qwen3_0_6b_base",
        "datasets": {
            "train": {"summary": {"sources": [{"source_id": "repo_curated_synthetic_templates"}]}}
        },
        "output_artifact": output_artifact_fingerprint(checkpoint),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_training_source_lineage_migration_is_bound_atomic_and_path_free(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original = _write_completed_manifest(checkpoint)
    manifest_path = checkpoint / "training_manifest.json"
    original_file_sha256 = sha256_file(manifest_path)
    backup = tmp_path / "evidence" / "before.json"
    receipt = tmp_path / "evidence" / "receipt.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "migrate_training_source_lineage.py"),
            "--checkpoint-dir",
            str(checkpoint),
            "--expected-manifest-sha256",
            str(original["manifest_sha256"]),
            "--expected-manifest-file-sha256",
            original_file_sha256,
            "--training-source-id",
            "qwen3_0_6b_base",
            "--training-source-id",
            "repo_curated_synthetic_templates",
            "--training-source-id",
            "qwen3_8b_placeholder_template_generator",
            "--migration-code-revision",
            "a" * 40,
            "--backup",
            str(backup),
            "--receipt",
            str(receipt),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    migration_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 4
    assert migrated["training_source_ids"] == [
        "qwen3_0_6b_base",
        "qwen3_8b_placeholder_template_generator",
        "repo_curated_synthetic_templates",
    ]
    assert migrated["output_artifact"] == original["output_artifact"]
    assert (
        migrated["provenance_migration"]["previous_manifest_sha256"] == original["manifest_sha256"]
    )
    assert verify_training_manifest(migrated)
    assert sha256_file(backup) == original_file_sha256
    assert migration_receipt["current"]["manifest_sha256"] == migrated["manifest_sha256"]
    assert migration_receipt["model_weights_changed"] is False
    serialized_receipt = json.dumps(migration_receipt)
    assert str(tmp_path) not in serialized_receipt
    assert verify_training_manifest(migration_receipt)


def test_training_source_lineage_migration_rejects_unapproved_input_hash(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original = _write_completed_manifest(checkpoint)

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "migrate_training_source_lineage.py"),
            "--checkpoint-dir",
            str(checkpoint),
            "--expected-manifest-sha256",
            str(original["manifest_sha256"]),
            "--expected-manifest-file-sha256",
            "0" * 64,
            "--training-source-id",
            "qwen3_0_6b_base",
            "--training-source-id",
            "repo_curated_synthetic_templates",
            "--migration-code-revision",
            "a" * 40,
            "--backup",
            str(tmp_path / "before.json"),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "file hash differs" in completed.stderr
    assert not (tmp_path / "before.json").exists()
