from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _registry_source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    return next(source for source in registry["sources"] if source["id"] == source_id)


def _source_summary(*, split: str, count: int, revision: str) -> dict[str, Any]:
    return {
        "source_id": "repo_curated_synthetic_templates",
        "source_kind": "curated_deterministic_synthetic",
        "revision": revision,
        "license": "Apache-2.0",
        "data_pool": "public_release_pool",
        "split": split,
        "document_count": count,
        "entity_count": count,
        "quality_tier_counts": {"synthetic_validated": count},
        "quality_gate_passed_document_count": count,
        "validators_passed_document_count": count,
        "public_weight_training_allowed_document_count": count,
    }


def _split_summary(
    *, split: str, count: int, revision: str, split_sha256: str | None = None
) -> dict[str, Any]:
    return {
        "sha256": split_sha256 or hashlib.sha256(split.encode()).hexdigest(),
        "summary": {
            "document_count": count,
            "entity_count": count,
            "token_count": count,
            "pii_free_document_count": 0,
            "unalignable_boundary_count": 0,
            "label_counts": {"PERSON_NAME": count},
            "split_counts": {split: count},
            "data_pool_counts": {"public_release_pool": count},
            "quality_tier_counts": {"synthetic_validated": count},
            "quality_gate_passed_document_count": count,
            "validators_passed_document_count": count,
            "public_weight_training_allowed_document_count": count,
            "sources": [_source_summary(split=split, count=count, revision=revision)],
        },
    }


def _training_manifest(path: Path, *, revision: str, validation_sha256: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 4,
        "status": "completed",
        "seed": 42,
        "attention_mode": "full",
        "fine_tuning": "lora",
        "base_source_id": "qwen3_0_6b_base",
        "training_source_ids": [
            "qwen3_0_6b_base",
            "qwen3_8b_placeholder_template_generator",
            "repo_curated_synthetic_templates",
        ],
        "datasets": {
            "train": _split_summary(split="train", count=16_000, revision=revision),
            "validation": _split_summary(
                split="validation",
                count=2_000,
                revision="sha256:" + "9" * 64,
                split_sha256=validation_sha256,
            ),
        },
        "output_artifact": {"weights_combined_sha256": "a" * 64},
    }
    value["manifest_sha256"] = _canonical_json_hash(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _run_export(
    repository_root: Path,
    *,
    training_manifest: Path,
    source_registry: Path,
    template_asset: Path,
    data_output: Path,
    teacher_output: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/export_release_provenance.py"),
            "--training-manifest",
            str(training_manifest),
            "--source-registry",
            str(source_registry),
            "--template-asset",
            str(template_asset),
            "--data-output",
            str(data_output),
            "--teacher-output",
            str(teacher_output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_self_hash(value: dict[str, Any]) -> None:
    claimed = value.pop("manifest_sha256")
    assert claimed == _canonical_json_hash(value)
    value["manifest_sha256"] = claimed


def test_exporter_is_deterministic_self_hashed_and_path_free(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    registry_path = repository_root / "configs/data/source_registry.yaml"
    asset_path = repository_root / "src/pii_zh/data/synthetic/assets/curated_templates_v1.json"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    data_registry = _registry_source(registry, "repo_curated_synthetic_templates")
    revision = data_registry["revision"]
    validation_sha256 = data_registry["frozen_holdout"]["validation_sha256"]
    training_path = tmp_path / "selected-full-training-manifest.json"
    _training_manifest(
        training_path,
        revision=revision,
        validation_sha256=validation_sha256,
    )

    first_data = tmp_path / "first/data_provenance.json"
    first_teacher = tmp_path / "first/teacher_provenance.json"
    second_data = tmp_path / "second/data_provenance.json"
    second_teacher = tmp_path / "second/teacher_provenance.json"
    first = _run_export(
        repository_root,
        training_manifest=training_path,
        source_registry=registry_path,
        template_asset=asset_path,
        data_output=first_data,
        teacher_output=first_teacher,
    )
    second = _run_export(
        repository_root,
        training_manifest=training_path,
        source_registry=registry_path,
        template_asset=asset_path,
        data_output=second_data,
        teacher_output=second_teacher,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_data.read_bytes() == second_data.read_bytes()
    assert first_teacher.read_bytes() == second_teacher.read_bytes()
    assert stat.S_IMODE(first_data.stat().st_mode) == 0o444
    assert stat.S_IMODE(first_teacher.stat().st_mode) == 0o444

    data = json.loads(first_data.read_text(encoding="utf-8"))
    teacher = json.loads(first_teacher.read_text(encoding="utf-8"))
    _assert_self_hash(data)
    _assert_self_hash(teacher)
    source = data["sources"][0]
    assert source["sample_count"] == 18_000
    assert source["usage_counts"] == {
        "gradient_training": 16_000,
        "validation_early_stopping_and_calibration": 2_000,
        "frozen_test_training_use": 0,
    }
    assert source["upstream_generation"] == {
        "source_id": "qwen3_8b_placeholder_template_generator",
        "role": "upstream_template_candidate_generator",
        "accepted_candidate_count": 53,
        "direct_span_annotation": False,
        "direct_pseudo_labeling": False,
    }
    assert source["split_source_revisions"] == {
        "train": revision,
        "validation": "sha256:" + "9" * 64,
    }
    assert source["frozen_validation_lineage"]["copy_mode"] == "byte_for_byte"
    assert data["total_sample_count"] == 18_000
    entry = teacher["teachers"][0]
    assert entry["used_for_training"] is True
    assert entry["role"] == "upstream_template_candidate_generator"
    assert entry["span_teacher"] is False
    assert entry["pseudo_label_teacher"] is False
    assert entry["raw_outputs_used_for_training"] is False
    assert entry["external_api"] is False
    assert teacher["external_api_used"] is False
    assert teacher["external_api_teacher_used"] is False
    serialized = first_data.read_text(encoding="utf-8") + first_teacher.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "不可泄漏" not in serialized
    assert all(value is False for value in data["privacy"].values())
    assert all(value is False for value in teacher["privacy"].values())


def _remove_teacher_lineage(manifest: dict[str, Any], registry: dict[str, Any]) -> None:
    del registry
    manifest["training_source_ids"].remove("qwen3_8b_placeholder_template_generator")


def _change_validation_count(manifest: dict[str, Any], registry: dict[str, Any]) -> None:
    del registry
    manifest["datasets"]["validation"]["summary"]["document_count"] = 1_999


def _change_candidate_hash(manifest: dict[str, Any], registry: dict[str, Any]) -> None:
    del manifest
    teacher = _registry_source(registry, "qwen3_8b_placeholder_template_generator")
    teacher["review_evidence"]["candidate_payload_sha256"] = "0" * 64


def _change_curation_hash(manifest: dict[str, Any], registry: dict[str, Any]) -> None:
    del manifest
    source = _registry_source(registry, "repo_curated_synthetic_templates")
    source["curation_audit_sha256"] = "0" * 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_remove_teacher_lineage, "training_source_ids"),
        (_change_validation_count, "validation.document_count"),
        (_change_candidate_hash, "candidate payload hash differs"),
        (_change_curation_hash, "curation audit hash differs"),
    ],
)
def test_exporter_fails_closed_on_lineage_count_and_registry_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
    message: str,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    original_registry = repository_root / "configs/data/source_registry.yaml"
    asset_path = repository_root / "src/pii_zh/data/synthetic/assets/curated_templates_v1.json"
    registry = yaml.safe_load(original_registry.read_text(encoding="utf-8"))
    data_registry = _registry_source(registry, "repo_curated_synthetic_templates")
    revision = data_registry["revision"]
    validation_sha256 = data_registry["frozen_holdout"]["validation_sha256"]
    training_path = tmp_path / "training_manifest.json"
    manifest = _training_manifest(
        training_path,
        revision=revision,
        validation_sha256=validation_sha256,
    )
    mutate(manifest, registry)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_json_hash(manifest)
    training_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    registry_path = tmp_path / "source_registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    result = _run_export(
        repository_root,
        training_manifest=training_path,
        source_registry=registry_path,
        template_asset=asset_path,
        data_output=tmp_path / "data_provenance.json",
        teacher_output=tmp_path / "teacher_provenance.json",
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert not (tmp_path / "data_provenance.json").exists()
    assert not (tmp_path / "teacher_provenance.json").exists()
