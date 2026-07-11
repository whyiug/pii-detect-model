from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pii_zh.evaluation import canonical_json_hash


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(path: Path, *, records: int) -> tuple[str, int]:
    rows = [
        {
            "doc_id": f"private-fixture-{index}",
            "text": f"不可泄漏原文-{index}",
            "entities": [
                {
                    "start": 0,
                    "end": 1,
                    "label": "PERSON_NAME",
                    "value": f"不可泄漏实体-{index}",
                }
            ],
        }
        for index in range(records)
    ]
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _source_manifest(
    path: Path,
    *,
    validation: tuple[str, int, int],
    test: tuple[str, int, int],
) -> Path:
    def file_entry(name: str, identity: tuple[str, int, int]) -> dict[str, Any]:
        digest, records, byte_count = identity
        return {
            "bytes": byte_count,
            "mode": "0444",
            "name": f"{name}.jsonl",
            "records": records,
            "sha256": digest,
            "split": name,
        }

    value = {
        "manifest_schema_version": 2,
        "dataset_id": "pii_zh_synthetic_fixture",
        "dataset_version": "1.3.0",
        "review_note": "合成评测夹具",
        "files": [
            {
                "bytes": 1,
                "mode": "0444",
                "name": "train.jsonl",
                "records": 1,
                "sha256": "0" * 64,
                "split": "train",
            },
            file_entry("validation", validation),
            file_entry("test", test),
        ],
        "coverage": {
            "labels_by_split": {
                "validation": {"PERSON_NAME": validation[1]},
                "test": {"PERSON_NAME": test[1]},
            }
        },
        "frozen_splits": {
            "policy_version": "byte-exact-parent-holdout-v1",
            "copy_mode": "byte_for_byte",
            "raw_jsonl_records_parsed": False,
            "files": {
                subset: {
                    "bytes": identity[2],
                    "name": f"{subset}.jsonl",
                    "output_sha256": identity[0],
                    "records": identity[1],
                    "source_sha256": identity[0],
                }
                for subset, identity in (("validation", validation), ("test", test))
            },
        },
        "provenance": {
            "validation_test": {
                "lineage": "byte_exact_parent_dataset_copy",
                "parent_dataset_version": "1.2.0",
                "parent_provenance_snapshot_sha256": "1" * 64,
            }
        },
    }
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _run(
    repository_root: Path,
    *,
    source: Path,
    canonical: Path,
    subset: str,
    output: Path,
    verify_existing: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(repository_root / "scripts/prepare_synthetic_evaluation_data.py"),
        "--source-manifest",
        str(source),
        "--subset",
        subset,
        "--canonical",
        str(canonical),
        "--output",
        str(output),
    ]
    if verify_existing:
        command.append("--verify-existing")
    return subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("subset", "role", "threshold_selection", "test_access"),
    [
        ("validation", "calibration_validation", True, "not_applicable"),
        ("test", "frozen_test", False, "forbidden"),
    ],
)
def test_attests_existing_split_without_leaking_records(
    tmp_path: Path,
    repository_root: Path,
    subset: str,
    role: str,
    threshold_selection: bool,
    test_access: str,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    validation_sha, validation_bytes = _canonical(validation_path, records=2)
    test_sha, test_bytes = _canonical(test_path, records=3)
    source = _source_manifest(
        tmp_path / "dataset_manifest.json",
        validation=(validation_sha, 2, validation_bytes),
        test=(test_sha, 3, test_bytes),
    )
    output = tmp_path / f"{subset}.evaluation_dataset.json"

    completed = _run(
        repository_root,
        source=source,
        canonical=validation_path if subset == "validation" else test_path,
        subset=subset,
        output=output,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert manifest["manifest_type"] == "evaluation_dataset"
    selected = manifest["subsets"][subset]
    assert selected["subset_role"] == role
    assert selected["used_for_threshold_selection"] is threshold_selection
    assert selected["frozen_holdout"] is True
    assert selected["test_access"]["access_before_validation_gate"] == test_access
    assert selected["canonical"]["record_count"] == (2 if subset == "validation" else 3)
    assert output.stat().st_mode & 0o777 == 0o444
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "private-fixture" not in serialized
    assert "不可泄漏原文" not in serialized
    assert "不可泄漏实体" not in serialized


def test_rejects_train_and_requires_verify_existing(tmp_path: Path, repository_root: Path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    validation_sha, validation_bytes = _canonical(validation_path, records=1)
    test_sha, test_bytes = _canonical(test_path, records=1)
    source = _source_manifest(
        tmp_path / "dataset_manifest.json",
        validation=(validation_sha, 1, validation_bytes),
        test=(test_sha, 1, test_bytes),
    )

    train_output = tmp_path / "train.json"
    train = _run(
        repository_root,
        source=source,
        canonical=validation_path,
        subset="train",
        output=train_output,
    )
    assert train.returncode != 0
    assert not train_output.exists()

    unsafe_output = tmp_path / "unsafe.json"
    unsafe = _run(
        repository_root,
        source=source,
        canonical=validation_path,
        subset="validation",
        output=unsafe_output,
        verify_existing=False,
    )
    assert unsafe.returncode != 0
    assert "--verify-existing is required" in unsafe.stderr
    assert not unsafe_output.exists()


@pytest.mark.parametrize("mutation", ["content", "count", "source_manifest"])
def test_fails_closed_on_identity_mismatch(
    tmp_path: Path, repository_root: Path, mutation: str
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    validation_sha, validation_bytes = _canonical(validation_path, records=2)
    test_sha, test_bytes = _canonical(test_path, records=2)
    source = _source_manifest(
        tmp_path / "dataset_manifest.json",
        validation=(validation_sha, 2, validation_bytes),
        test=(test_sha, 2, test_bytes),
    )
    if mutation == "content":
        validation_path.write_text("tampered\n", encoding="utf-8")
    elif mutation == "count":
        source_value = json.loads(source.read_text(encoding="utf-8"))
        source_value.pop("manifest_sha256")
        next(item for item in source_value["files"] if item["split"] == "validation")["records"] = 3
        source_value["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                source_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        source.write_text(json.dumps(source_value), encoding="utf-8")
    else:
        source_value = json.loads(source.read_text(encoding="utf-8"))
        source_value["dataset_version"] = "tampered"
        source.write_text(json.dumps(source_value), encoding="utf-8")
    output = tmp_path / "output.json"
    output.write_text('{"stale":true}\n', encoding="utf-8")

    completed = _run(
        repository_root,
        source=source,
        canonical=validation_path,
        subset="validation",
        output=output,
    )

    assert completed.returncode != 0
    assert not output.exists()
