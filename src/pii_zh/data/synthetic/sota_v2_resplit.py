"""Derive a validation-informed v2 training split from ``synthetic_sota_v1``.

The first full-attention diagnostic exposed an intentional distribution gap:
the v1 train and validation partitions own disjoint template families. This
derivation moves a deterministic, stratified 80% of the old validation rows
into training and keeps the remainder for development-only checkpoint
selection/calibration. Exact documents, entity values, and source groups stay
disjoint. Template-family overlap is explicit and means this split can never
serve as release-test evidence; an independently materialized evaluation-only
corpus is required for that role.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from pii_zh.data.schema import DataPool, DocumentRecord, iter_jsonl, write_jsonl

DATASET_ID = "pii_zh_synthetic_sota_v2_resplit"
DATASET_VERSION = "2.0.0"
MATERIALIZER_VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = Path("configs/data/synthetic_sota_v2_resplit.yaml")
DEFAULT_DATA_ROOT = Path(os.environ.get("PII_ZH_DATA_ROOT", "data"))
DEFAULT_SOURCE_DIR = DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_v2_resplit"
CORE_LABEL_COUNT = 24


class SyntheticSotaV2ResplitError(ValueError):
    """Raised when the derived split cannot satisfy its frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any, *, remove: str | None = None) -> str:
    document = dict(value) if isinstance(value, Mapping) else value
    if remove is not None and isinstance(document, dict):
        document.pop(remove, None)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SyntheticSotaV2ResplitError("invalid resplit config") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "dataset",
        "source",
        "resplit",
        "roles",
    }:
        raise SyntheticSotaV2ResplitError("resplit config has an invalid closed shape")
    if value["schema_version"] != 1:
        raise SyntheticSotaV2ResplitError("unsupported resplit config version")
    if value["dataset"] != {
        "id": DATASET_ID,
        "version": DATASET_VERSION,
        "license": "Apache-2.0",
        "language": "zh-Hans-CN",
        "data_pool": "public_release_pool",
    }:
        raise SyntheticSotaV2ResplitError("derived dataset identity changed")
    expected_roles = {
        "train": "weight_training",
        "validation": "development_checkpoint_selection_and_calibration_only",
        "release_test_eligible": False,
        "public_benchmark_read": False,
    }
    if value["roles"] != expected_roles:
        raise SyntheticSotaV2ResplitError("derived split roles changed")
    policy = value["resplit"]
    if not isinstance(policy, dict) or set(policy) != {
        "strategy",
        "salt",
        "validation_to_train_ratio",
        "allow_template_group_overlap",
        "require_document_id_disjoint",
        "require_entity_value_group_disjoint",
        "require_source_group_disjoint",
    }:
        raise SyntheticSotaV2ResplitError("resplit policy has an invalid closed shape")
    if (
        policy["strategy"] != "template-and-pii-state-stratified-hash-v1"
        or not isinstance(policy["salt"], str)
        or not policy["salt"]
        or isinstance(policy["validation_to_train_ratio"], bool)
        or not isinstance(policy["validation_to_train_ratio"], (int, float))
        or not 0.5 <= float(policy["validation_to_train_ratio"]) < 1.0
        or policy["allow_template_group_overlap"] is not True
        or any(
            policy[name] is not True
            for name in (
                "require_document_id_disjoint",
                "require_entity_value_group_disjoint",
                "require_source_group_disjoint",
            )
        )
    ):
        raise SyntheticSotaV2ResplitError("resplit policy boundary changed")
    return value


def _rank(record: DocumentRecord, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{record.doc_id}".encode()).hexdigest()


def partition_validation_records(
    records: Sequence[DocumentRecord],
    *,
    train_ratio: float,
    salt: str,
) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    """Split old validation rows deterministically within template/PII strata."""

    if not records:
        raise SyntheticSotaV2ResplitError("source validation split is empty")
    if not 0.5 <= train_ratio < 1.0 or not salt:
        raise SyntheticSotaV2ResplitError("invalid partition parameters")
    strata: dict[tuple[str, bool], list[DocumentRecord]] = defaultdict(list)
    for record in records:
        record.validate()
        if record.split != "validation" or not record.template_group:
            raise SyntheticSotaV2ResplitError(
                "source validation rows require a template group and validation split"
            )
        strata[(record.template_group, not record.entities)].append(record)

    promoted: list[DocumentRecord] = []
    retained: list[DocumentRecord] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda item: _rank(item, salt=salt))
        promote_count = (
            0 if len(rows) == 1 else min(len(rows) - 1, math.floor(len(rows) * train_ratio))
        )
        promoted.extend(rows[:promote_count])
        retained.extend(rows[promote_count:])
    if not promoted or not retained:
        raise SyntheticSotaV2ResplitError("partition produced an empty role")
    return promoted, retained


def _source_group_values(record: DocumentRecord) -> tuple[str, ...]:
    groups = []
    if record.tenant_group:
        groups.append(f"tenant:{record.tenant_group}")
    if record.conversation_group:
        groups.append(f"conversation:{record.conversation_group}")
    source_group = record.metadata.get("source_group")
    if source_group is not None:
        if not isinstance(source_group, str) or not source_group:
            raise SyntheticSotaV2ResplitError("invalid source_group metadata")
        groups.append(f"metadata:{source_group}")
    return tuple(groups)


def _assert_output_contract(
    train: Sequence[DocumentRecord], validation: Sequence[DocumentRecord]
) -> dict[str, Any]:
    train_doc_ids = {record.doc_id for record in train}
    validation_doc_ids = {record.doc_id for record in validation}
    train_values = {value for record in train for value in record.entity_value_groups}
    validation_values = {value for record in validation for value in record.entity_value_groups}
    train_sources = {value for record in train for value in _source_group_values(record)}
    validation_sources = {value for record in validation for value in _source_group_values(record)}
    collisions = {
        "document_id": len(train_doc_ids & validation_doc_ids),
        "entity_value_group": len(train_values & validation_values),
        "source_group": len(train_sources & validation_sources),
    }
    if any(collisions.values()):
        raise SyntheticSotaV2ResplitError("identity/value/source split isolation failed")
    template_overlap = {record.template_group for record in train if record.template_group} & {
        record.template_group for record in validation if record.template_group
    }
    if not template_overlap:
        raise SyntheticSotaV2ResplitError("the disclosed template-overlap policy was not exercised")

    label_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    pii_free: Counter[str] = Counter()
    for split, records in (("train", train), ("validation", validation)):
        for record in records:
            record.validate()
            if record.split != split or record.data_pool is not DataPool.PUBLIC_RELEASE:
                raise SyntheticSotaV2ResplitError("derived record role/pool mismatch")
            if not record.provenance.synthetic or record.provenance.evaluation_only:
                raise SyntheticSotaV2ResplitError("derived records must remain trainable synthetic")
            label_counts[split].update(entity.label for entity in record.entities)
            pii_free[split] += int(not record.entities)
        if len(label_counts[split]) != CORE_LABEL_COUNT:
            raise SyntheticSotaV2ResplitError(f"{split} lost 24-label coverage")
        ratio = pii_free[split] / len(records)
        if not 0.40 <= ratio <= 0.50:
            raise SyntheticSotaV2ResplitError(f"{split} PII-free ratio left [0.40, 0.50]")
    return {
        "cross_split_collisions": collisions,
        "template_group_overlap_count": len(template_overlap),
        "template_group_overlap_allowed": True,
        "label_counts": {
            split: dict(sorted(counts.items())) for split, counts in label_counts.items()
        },
        "pii_free": dict(sorted(pii_free.items())),
    }


def _write(records: Sequence[DocumentRecord], path: Path, *, split: str) -> dict[str, Any]:
    count = write_jsonl(records, path)
    path.chmod(0o444)
    replay = sum(1 for record in iter_jsonl(path) if record.split == split)
    if count != replay or count != len(records):
        raise SyntheticSotaV2ResplitError("derived split readback failed")
    return {
        "name": path.name,
        "split": split,
        "records": count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "mode": "0444",
    }


def materialize_synthetic_sota_v2_resplit(
    output_dir: Path,
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Create one immutable v2 resplit without modifying its source corpus."""

    root = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    config_file = (
        (config_path if config_path.is_absolute() else root / config_path)
        .expanduser()
        .resolve(strict=True)
    )
    config = _load_config(config_file)
    source = source_dir.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite a derived dataset")

    source_files = {
        "manifest": source / "dataset_manifest.json",
        "train": source / "train.jsonl",
        "validation": source / "validation.jsonl",
    }
    expected = config["source"]
    observed_hashes = {name: _sha256_file(path) for name, path in source_files.items()}
    if observed_hashes != {
        "manifest": expected["manifest_file_sha256"],
        "train": expected["train_sha256"],
        "validation": expected["validation_sha256"],
    }:
        raise SyntheticSotaV2ResplitError("source corpus file hash changed")
    source_manifest = json.loads(source_files["manifest"].read_text(encoding="utf-8"))
    if (
        source_manifest.get("dataset_id") != expected["dataset_id"]
        or source_manifest.get("dataset_version") != expected["dataset_version"]
        or source_manifest.get("manifest_sha256") != expected["manifest_canonical_sha256"]
        or _canonical_hash(source_manifest, remove="manifest_sha256")
        != expected["manifest_canonical_sha256"]
    ):
        raise SyntheticSotaV2ResplitError("source dataset manifest changed")

    original_train = list(iter_jsonl(source_files["train"]))
    original_validation = list(iter_jsonl(source_files["validation"]))
    if (
        len(original_train) != expected["train_records"]
        or len(original_validation) != expected["validation_records"]
    ):
        raise SyntheticSotaV2ResplitError("source corpus record count changed")
    policy = config["resplit"]
    promoted, retained = partition_validation_records(
        original_validation,
        train_ratio=float(policy["validation_to_train_ratio"]),
        salt=str(policy["salt"]),
    )

    def with_role(record: DocumentRecord, split: str, source_split: str) -> DocumentRecord:
        return replace(
            record,
            split=split,
            metadata={
                **record.metadata,
                "synthetic_sota_v2_resplit": {
                    "dataset_version": DATASET_VERSION,
                    "source_split": source_split,
                    "derived_role": split,
                    "release_test_eligible": False,
                },
            },
        )

    train = [with_role(record, "train", "train") for record in original_train]
    train.extend(with_role(record, "train", "validation") for record in promoted)
    validation = [with_role(record, "validation", "validation") for record in retained]
    audit = _assert_output_contract(train, validation)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError("derived dataset staging directory exists")
    staging.mkdir(mode=0o755)
    try:
        files = [
            _write(train, staging / "train.jsonl", split="train"),
            _write(validation, staging / "validation.jsonl", split="validation"),
        ]
        manifest: dict[str, Any] = {
            "manifest_schema_version": 1,
            "manifest_type": "development_training_dataset",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "license": "Apache-2.0",
            "language": "zh-Hans-CN",
            "data_pool": "public_release_pool",
            "public_weight_training_allowed": True,
            "release_test_eligible": False,
            "materializer_version": MATERIALIZER_VERSION,
            "config": {
                "path": str(config_file.relative_to(root)),
                "sha256": _sha256_file(config_file),
            },
            "source": {
                "dataset_id": expected["dataset_id"],
                "dataset_version": expected["dataset_version"],
                "manifest_canonical_sha256": expected["manifest_canonical_sha256"],
                "files": observed_hashes,
            },
            "resplit": dict(policy),
            "roles": dict(config["roles"]),
            "counts": {
                "total": len(train) + len(validation),
                "by_split": {"train": len(train), "validation": len(validation)},
                "promoted_from_old_validation": len(promoted),
                "retained_development_validation": len(retained),
            },
            "audit": audit,
            "files": files,
            "benchmark_isolation": {
                "pii_bench_zh_read": False,
                "public_benchmark_prediction_read": False,
            },
        }
        manifest["manifest_sha256"] = _canonical_hash(manifest)
        manifest_path = staging / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o444)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_DIR",
    "SyntheticSotaV2ResplitError",
    "materialize_synthetic_sota_v2_resplit",
    "partition_validation_records",
]
