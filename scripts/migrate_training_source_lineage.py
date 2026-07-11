#!/usr/bin/env python3
"""Atomically add reviewed upstream source lineage to a completed checkpoint.

This migration changes only the self-hashed training manifest.  It verifies
the existing manifest and model-artifact binding, records the previous content
address, keeps an explicit backup outside the checkpoint, and emits a path-free
receipt.  Model weights, tokenizer files, datasets, and recipes are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.training.manifest import (  # noqa: E402
    canonical_json_hash,
    sha256_file,
    verify_output_artifact_binding,
    verify_training_manifest,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}")
_MIGRATION_ID = "training_source_lineage_v1"
_TARGET_SCHEMA_VERSION = 4


class LineageMigrationError(ValueError):
    """Raised when a completed training manifest cannot be migrated safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    parser.add_argument("--training-source-id", action="append", required=True)
    parser.add_argument("--migration-code-revision", required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _load_object(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LineageMigrationError(f"{description} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LineageMigrationError(f"{description} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LineageMigrationError(f"{description} must be a JSON object")
    return value


def _write_json_atomic(value: dict[str, Any], destination: Path, *, mode: int = 0o444) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.chmod(temporary, mode)
    os.replace(temporary, destination)


def _validated_sources(values: list[str]) -> list[str]:
    if any(_SAFE_ID.fullmatch(value) is None for value in values):
        raise LineageMigrationError("training source IDs must be non-empty path-free identifiers")
    normalized = sorted(set(values))
    if len(normalized) != len(values):
        raise LineageMigrationError("training source IDs must be unique")
    if "qwen3_0_6b_base" not in normalized:
        raise LineageMigrationError("training sources must retain the registered base model")
    if "repo_curated_synthetic_templates" not in normalized:
        raise LineageMigrationError("training sources must retain the sampled synthetic dataset")
    return normalized


def migrate_manifest(
    *,
    checkpoint_dir: Path,
    expected_manifest_sha256: str,
    expected_manifest_file_sha256: str,
    training_source_ids: list[str],
    migration_code_revision: str,
    backup: Path,
    receipt: Path,
) -> dict[str, Any]:
    """Perform the verified migration and return its path-free receipt."""

    if (
        _SHA256.fullmatch(expected_manifest_sha256) is None
        or _SHA256.fullmatch(expected_manifest_file_sha256) is None
    ):
        raise LineageMigrationError("expected manifest hashes must be lowercase SHA-256")
    if _REVISION.fullmatch(migration_code_revision) is None:
        raise LineageMigrationError("migration code revision must be a full Git commit")
    sources = _validated_sources(training_source_ids)

    root = checkpoint_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise LineageMigrationError("checkpoint-dir must be a local directory")
    manifest_path = root / "training_manifest.json"
    backup = backup.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    if backup == receipt or manifest_path in {backup, receipt}:
        raise LineageMigrationError("backup, receipt, and live manifest must be distinct")

    manifest = _load_object(manifest_path, description="training manifest")
    observed_file_sha256 = sha256_file(manifest_path)
    if observed_file_sha256 != expected_manifest_file_sha256:
        raise LineageMigrationError("training manifest file hash differs from the approved input")
    if (
        manifest.get("manifest_sha256") != expected_manifest_sha256
        or not verify_training_manifest(manifest)
        or manifest.get("status") != "completed"
    ):
        raise LineageMigrationError("training manifest content hash/status is invalid")
    if manifest.get("schema_version") != 3:
        raise LineageMigrationError("only an unmigrated schema-3 training manifest is accepted")
    if "training_source_ids" in manifest or "provenance_migration" in manifest:
        raise LineageMigrationError("training manifest already contains migrated lineage")
    if not verify_output_artifact_binding(manifest, root, require_single_weight=True):
        raise LineageMigrationError("training manifest no longer binds the checkpoint artifact")

    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if (
            backup.is_symlink()
            or not backup.is_file()
            or sha256_file(backup) != observed_file_sha256
        ):
            raise LineageMigrationError("existing backup does not match the approved input")
    else:
        shutil.copyfile(manifest_path, backup)
        os.chmod(backup, 0o444)

    output_artifact_sha256 = canonical_json_hash(manifest["output_artifact"])
    migrated = dict(manifest)
    migrated.pop("manifest_sha256", None)
    migrated["schema_version"] = _TARGET_SCHEMA_VERSION
    migrated["training_source_ids"] = sources
    migrated["provenance_migration"] = {
        "migration_id": _MIGRATION_ID,
        "migration_code_revision": migration_code_revision,
        "previous_schema_version": 3,
        "previous_manifest_sha256": expected_manifest_sha256,
        "previous_manifest_file_sha256": expected_manifest_file_sha256,
        "reason": "record_reviewed_upstream_placeholder_template_generator",
        "changed_fields": [
            "schema_version",
            "training_source_ids",
            "provenance_migration",
        ],
        "model_weights_changed": False,
        "tokenizer_changed": False,
        "training_data_changed": False,
        "training_recipe_changed": False,
    }
    migrated["manifest_sha256"] = canonical_json_hash(migrated)
    if not verify_training_manifest(migrated):
        raise AssertionError("internal error: migrated manifest self-hash failed")

    _write_json_atomic(migrated, manifest_path)
    if not verify_output_artifact_binding(migrated, root, require_single_weight=True):
        raise LineageMigrationError("migrated manifest lost the checkpoint artifact binding")
    new_file_sha256 = sha256_file(manifest_path)
    migration_receipt = {
        "schema_version": 1,
        "artifact_type": "training_source_lineage_migration_receipt",
        "migration_id": _MIGRATION_ID,
        "migration_code_revision": migration_code_revision,
        "previous": {
            "schema_version": 3,
            "manifest_sha256": expected_manifest_sha256,
            "manifest_file_sha256": expected_manifest_file_sha256,
        },
        "current": {
            "schema_version": _TARGET_SCHEMA_VERSION,
            "manifest_sha256": migrated["manifest_sha256"],
            "manifest_file_sha256": new_file_sha256,
        },
        "training_source_ids": sources,
        "output_artifact_sha256": output_artifact_sha256,
        "model_weights_changed": False,
        "tokenizer_changed": False,
        "training_data_changed": False,
        "training_recipe_changed": False,
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
        },
    }
    migration_receipt["manifest_sha256"] = canonical_json_hash(migration_receipt)
    _write_json_atomic(migration_receipt, receipt)
    return migration_receipt


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = migrate_manifest(
            checkpoint_dir=args.checkpoint_dir,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_manifest_file_sha256=args.expected_manifest_file_sha256,
            training_source_ids=args.training_source_id,
            migration_code_revision=args.migration_code_revision,
            backup=args.backup,
            receipt=args.receipt,
        )
    except (LineageMigrationError, OSError, TypeError, ValueError) as exc:
        print(f"training source lineage migration blocked: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "manifest_sha256": receipt["current"]["manifest_sha256"],
                "receipt_sha256": receipt["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
