"""Reproducible, leakage-safe materialization of the public synthetic corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..schema import DataPool, DocumentRecord, iter_jsonl, write_jsonl
from ..splitting import assert_no_group_leakage
from .allocation import (
    FAMILY_ALLOCATION_ALGORITHM_VERSION,
    FAMILY_ALLOCATION_ASSET_ID,
    FAMILY_ALLOCATION_ASSET_SHA256,
    FAMILY_ALLOCATION_ASSET_VERSION,
    LEGACY_FAMILY_ALLOCATIONS,
    LEGACY_TEMPLATE_SPLITS,
    TEMPLATE_SPLITS,
    TRAIN_ONLY_FAMILY_ADDITIONS,
)
from .generator import GENERATOR_NAME, GENERATOR_VERSION, SyntheticGenerator
from .templates import (
    CORE_LABELS,
    CURATED_ASSET_ID,
    CURATED_ASSET_SHA256,
    CURATED_ASSET_VERSION,
    CURATION_AUDIT_SHA256,
    HARD_NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    SyntheticTemplate,
)

DATASET_ID = "pii_zh_synthetic_v1"
DATASET_VERSION = "1.3.0"
MATERIALIZER_VERSION = "1.3.0"
SPLIT_ALGORITHM_VERSION = FAMILY_ALLOCATION_ALGORITHM_VERSION
DEFAULT_COUNT = 20_000
DEFAULT_SEED = 20_260_711
DEFAULT_HARD_NEGATIVE_RATIO = 0.30
DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}
SURFACE_SHORTCUTS: tuple[str, ...] = (
    "test",
    "syn",
    "测试",
    "示例",
    "合成",
    "虚构",
    "演练",
)
DEFAULT_FROZEN_HOLDOUT_SOURCE_DIR = Path(
    "/data1/datasets/pii-detect-model/processed/public_release_pool/synthetic_v1_2"
)


@dataclass(frozen=True, slots=True)
class FrozenHoldoutFileContract:
    split: str
    name: str
    records: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenHoldoutContract:
    parent_dataset_version: str
    parent_manifest_sha256: str
    parent_manifest_file_sha256: str
    entity_value_group_count: int
    entity_value_groups_sha256: str
    files: tuple[FrozenHoldoutFileContract, ...]


OFFICIAL_FROZEN_HOLDOUT_CONTRACT = FrozenHoldoutContract(
    parent_dataset_version="1.2.0",
    parent_manifest_sha256="58a6067b93c04ba4a4d3a245145a0a501af07d6eb4f3c740b8e82e5681b07c2e",
    parent_manifest_file_sha256=(
        "e1bd5a6b094bdcd6566c94f5774313505924051d551d74c74617a21e309d023b"
    ),
    # Built from a metadata-only in-memory replay of the pinned pre-v1.3
    # templates/generator.  Frozen JSONL records were not parsed.  The
    # commitment makes future replay drift fail closed before collision checks.
    entity_value_group_count=6_832,
    entity_value_groups_sha256=("28d7eda8260a8f908defef2a36d4434500b0c3244aa5d62e9d7daf3988306ae3"),
    files=(
        FrozenHoldoutFileContract(
            split="validation",
            name="validation.jsonl",
            records=2_000,
            bytes=5_250_600,
            sha256="9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d",
        ),
        FrozenHoldoutFileContract(
            split="test",
            name="test.jsonl",
            records=2_000,
            bytes=5_096_362,
            sha256="26e49949d0bd876b51c693b7b31135b5ce503ca6a5ab7cfca37177e0ccbd4384",
        ),
    ),
)


class MaterializationError(ValueError):
    """Raised when a corpus cannot satisfy its immutable data contract."""


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_file_once(
    path: Path,
    *,
    description: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    """Read and hash one regular file through a single O_NOFOLLOW descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as exc:
        raise MaterializationError(f"{description} could not be opened safely") from exc
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise MaterializationError(f"{description} is not a bounded regular file")
        with os.fdopen(file_descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                chunks.append(block)
                digest.update(block)
                observed_bytes += len(block)
        after = os.fstat(file_descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or observed_bytes != before.st_size:
            raise MaterializationError(f"{description} changed while it was being read")
    finally:
        os.close(file_descriptor)
    return b"".join(chunks), digest.hexdigest()


def _integer_allocation(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {name: total * ratio for name, ratio in ratios.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(
        ratios,
        key=lambda name: (-(exact[name] - result[name]), name),
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    result = dict(ratios)
    if tuple(result) != ("train", "validation", "test"):
        raise MaterializationError("split ratios must be ordered as train/validation/test")
    if any(
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not 0.0 < float(ratio) < 1.0
        for ratio in result.values()
    ):
        raise MaterializationError("split ratios must be numeric values strictly between 0 and 1")
    if abs(sum(result.values()) - 1.0) > 1e-9:
        raise MaterializationError("split ratios must sum to one")
    return {name: float(ratio) for name, ratio in result.items()}


def _load_frozen_holdout(
    source_dir: Path,
    *,
    contract: FrozenHoldoutContract,
    count: int,
    seed: int,
    hard_negative_ratio: float,
    split_counts: Mapping[str, int],
    negative_counts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify a raw-text-free parent manifest and holdout bytes without parsing JSONL."""

    expected_splits = {"validation", "test"}
    contracts = {item.split: item for item in contract.files}
    if set(contracts) != expected_splits or len(contracts) != len(contract.files):
        raise MaterializationError("frozen holdout contract must define validation and test once")
    if any(
        item.name != f"{item.split}.jsonl"
        or item.records < 1
        or item.bytes < 1
        or len(item.sha256) != 64
        for item in contract.files
    ):
        raise MaterializationError("frozen holdout file contract is malformed")
    if (
        not contract.parent_dataset_version
        or len(contract.parent_manifest_sha256) != 64
        or len(contract.parent_manifest_file_sha256) != 64
        or contract.entity_value_group_count < 1
        or len(contract.entity_value_groups_sha256) != 64
    ):
        raise MaterializationError("frozen parent manifest contract is malformed")

    source_dir = source_dir.expanduser().resolve(strict=True)
    manifest_path = source_dir / "dataset_manifest.json"
    manifest_raw, manifest_file_hash = _read_regular_file_once(
        manifest_path,
        description="frozen parent manifest",
        max_bytes=16 * 1024 * 1024,
    )
    if manifest_file_hash != contract.parent_manifest_file_sha256:
        raise MaterializationError("frozen parent manifest file hash changed")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("frozen parent manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise MaterializationError("frozen parent manifest must contain an object")
    claimed_manifest_hash = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if (
        claimed_manifest_hash != contract.parent_manifest_sha256
        or _canonical_json_hash(unsigned_manifest) != contract.parent_manifest_sha256
    ):
        raise MaterializationError("frozen parent manifest logical hash changed")
    if (
        manifest.get("manifest_schema_version") != 1
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_version") != contract.parent_dataset_version
        or manifest.get("data_pool") != DataPool.PUBLIC_RELEASE.value
        or manifest.get("public_weight_training_allowed") is not True
    ):
        raise MaterializationError("frozen parent dataset identity or routing changed")
    parent_assets = manifest.get("assets")
    parent_provenance = manifest.get("provenance")
    if (
        not isinstance(parent_assets, dict)
        or not isinstance(parent_provenance, dict)
        or manifest.get("provenance_snapshot_sha256") != _canonical_json_hash(parent_provenance)
    ):
        raise MaterializationError("frozen parent asset or provenance snapshot changed")

    generation = manifest.get("generation")
    if not isinstance(generation, dict) or (
        generation.get("generator_name") != GENERATOR_NAME
        or generation.get("generator_version") != "2.3.0"
        or generation.get("materializer_version") != "1.2.0"
        or generation.get("seed") != seed
        or generation.get("requested_count") != count
        or generation.get("requested_hard_negative_ratio") != float(hard_negative_ratio)
    ):
        raise MaterializationError("frozen parent generation contract changed")

    splitting = manifest.get("splitting")
    if not isinstance(splitting, dict) or (
        splitting.get("algorithm_version") != "stable-pinned-v2"
        or splitting.get("family_assignment_uses_seed") is not False
        or splitting.get("ratios") != dict(DEFAULT_SPLIT_RATIOS)
        or splitting.get("cross_split_group_collisions") != 0
        or splitting.get("entity_value_repetitions") != 0
        or splitting.get("template_families_by_split")
        != {"train": 65, "validation": 15, "test": 16}
    ):
        raise MaterializationError("frozen parent split isolation contract changed")

    counts = manifest.get("counts")
    expected_negative = dict(negative_counts)
    if not isinstance(counts, dict) or (
        counts.get("total") != count
        or counts.get("by_split") != dict(split_counts)
        or counts.get("hard_negative") != sum(negative_counts.values())
        or counts.get("hard_negative_by_split") != expected_negative
    ):
        raise MaterializationError("frozen parent split counts changed")

    raw_files = manifest.get("files")
    if (
        not isinstance(raw_files, list)
        or len(raw_files) != 3
        or not all(isinstance(item, dict) for item in raw_files)
    ):
        raise MaterializationError("frozen parent files metadata is malformed")
    files_by_split = {item.get("split"): item for item in raw_files}
    if len(files_by_split) != 3 or set(files_by_split) != {"train", "validation", "test"}:
        raise MaterializationError("frozen parent must describe exactly three split files")
    verified_paths: dict[str, Path] = {}
    for split, expected in contracts.items():
        metadata = files_by_split[split]
        expected_metadata = {
            "name": expected.name,
            "split": split,
            "records": expected.records,
            "bytes": expected.bytes,
            "sha256": expected.sha256,
            "mode": "0444",
        }
        if metadata != expected_metadata or expected.records != split_counts[split]:
            raise MaterializationError(f"frozen {split} file metadata changed")
        source = source_dir / expected.name
        if (
            source.is_symlink()
            or not source.is_file()
            or not stat.S_ISREG(source.stat(follow_symlinks=False).st_mode)
            or source.stat().st_size != expected.bytes
            or sha256_file(source) != expected.sha256
        ):
            raise MaterializationError(f"frozen {split} bytes changed")
        verified_paths[split] = source

    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise MaterializationError("frozen parent coverage metadata is missing")
    for section in ("labels_by_split", "domains_by_split", "quality_tiers_by_split"):
        values = coverage.get(section)
        if not isinstance(values, dict):
            raise MaterializationError(f"frozen parent {section} is malformed")
        for split in expected_splits:
            if not isinstance(values.get(split), dict) or not values[split]:
                raise MaterializationError(f"frozen parent {section}.{split} is empty")
    for split in expected_splits:
        if set(coverage["labels_by_split"][split]) != set(CORE_LABELS):
            raise MaterializationError(f"frozen parent {split} no longer covers all core labels")
    return manifest, verified_paths


def _generation_plan(
    *,
    count: int,
    seed: int,
    hard_negative_ratio: float,
    split_ratios: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 300:
        raise MaterializationError("count must be an integer of at least 300")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MaterializationError("seed must be an integer")
    if (
        isinstance(hard_negative_ratio, bool)
        or not isinstance(hard_negative_ratio, (int, float))
        or not 0.30 <= float(hard_negative_ratio) < 1.0
    ):
        raise MaterializationError("hard_negative_ratio must be at least 0.30 and below 1.0")
    ratios = _validate_ratios(split_ratios)
    split_counts = _integer_allocation(count, ratios)
    negative_counts = _integer_allocation(
        math.ceil(count * float(hard_negative_ratio)),
        ratios,
    )
    return ratios, split_counts, negative_counts


def _allocate_template_families(
    *,
    ratios: Mapping[str, float],
) -> tuple[
    dict[str, tuple[SyntheticTemplate, ...]],
    dict[str, tuple[SyntheticTemplate, ...]],
]:
    """Apply the audited, fail-closed template-family allocation asset."""

    split_names = tuple(ratios)
    if split_names != ("train", "validation", "test"):
        raise MaterializationError("stable family allocation requires train/validation/test")

    positive = {
        split: tuple(
            template
            for template in POSITIVE_TEMPLATES
            if TEMPLATE_SPLITS[template.template_id] == split
        )
        for split in split_names
    }
    negative = {
        split: tuple(
            template
            for template in HARD_NEGATIVE_TEMPLATES
            if TEMPLATE_SPLITS[template.template_id] == split
        )
        for split in split_names
    }
    if any(not positive[split] or not negative[split] for split in split_names):
        raise MaterializationError("every split must receive positive and hard-negative families")
    return positive, negative


def assert_surface_contract(records: Sequence[DocumentRecord]) -> None:
    """Reject training shortcuts without exposing the offending raw record."""

    seen_values: set[str] = set()
    for record in records:
        record.validate()
        lowered = record.text.casefold()
        marker = next((item for item in SURFACE_SHORTCUTS if item in lowered), None)
        if marker is not None:
            raise MaterializationError(
                f"surface shortcut {marker!r} detected in document {record.doc_id}"
            )
        for entity in record.entities:
            entity_lowered = entity.text.casefold()
            marker = next((item for item in SURFACE_SHORTCUTS if item in entity_lowered), None)
            if marker is not None:
                raise MaterializationError(
                    f"surface shortcut {marker!r} detected in an entity of {record.doc_id}"
                )
        for value_group in record.entity_value_groups:
            if value_group in seen_values:
                raise MaterializationError("generated entity values must be globally unique")
            seen_values.add(value_group)


def _build_regenerated_test_corpus(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> list[DocumentRecord]:
    """Test-only regenerated corpus; never use this path for release holdouts."""

    ratios, split_counts, negative_counts = _generation_plan(
        count=count,
        seed=seed,
        hard_negative_ratio=hard_negative_ratio,
        split_ratios=split_ratios,
    )
    positive_templates, negative_templates = _allocate_template_families(
        ratios=ratios,
    )

    generator = SyntheticGenerator(seed)
    result: list[DocumentRecord] = []
    for split in ratios:
        positive_count = split_counts[split] - negative_counts[split]
        if positive_count < len(positive_templates[split]):
            raise MaterializationError(
                f"split {split!r} is too small to exercise every assigned positive family"
            )
        if negative_counts[split] < len(negative_templates[split]):
            raise MaterializationError(
                f"split {split!r} is too small to exercise every assigned negative family"
            )
        records = generator.generate_partition(
            positive_count=positive_count,
            hard_negative_count=negative_counts[split],
            positive_templates=positive_templates[split],
            hard_negative_templates=negative_templates[split],
        )
        result.extend(replace(record, split=split) for record in records)

    assert_surface_contract(result)
    assert_no_group_leakage(result)
    counts = Counter(record.split for record in result)
    if counts != Counter(split_counts):
        raise MaterializationError("materialized split counts differ from the requested ratios")
    for split in ratios:
        labels = {
            entity.label for record in result if record.split == split for entity in record.entities
        }
        if labels != set(CORE_LABELS):
            raise MaterializationError(f"split {split!r} does not cover all core labels")
    return result


def _build_train_partition(
    *,
    generator: SyntheticGenerator,
    split_counts: Mapping[str, int],
    negative_counts: Mapping[str, int],
    ratios: Mapping[str, float],
) -> list[DocumentRecord]:
    positive_templates, negative_templates = _allocate_template_families(ratios=ratios)
    positive_count = split_counts["train"] - negative_counts["train"]
    if positive_count < len(positive_templates["train"]):
        raise MaterializationError("train split is too small for every assigned positive family")
    if negative_counts["train"] < len(negative_templates["train"]):
        raise MaterializationError("train split is too small for every assigned negative family")
    records = generator.generate_partition(
        positive_count=positive_count,
        hard_negative_count=negative_counts["train"],
        positive_templates=positive_templates["train"],
        hard_negative_templates=negative_templates["train"],
    )
    result = [replace(record, split="train") for record in records]
    assert_surface_contract(result)
    assert_no_group_leakage(result)
    labels = {entity.label for record in result for entity in record.entities}
    if labels != set(CORE_LABELS):
        raise MaterializationError("generated train split does not cover all core labels")
    return result


def _replay_parent_sequence(
    *,
    seed: int,
    split_counts: Mapping[str, int],
    negative_counts: Mapping[str, int],
    ratios: Mapping[str, float],
) -> tuple[SyntheticGenerator, set[str]]:
    """Replay all v1.2 splits and return its consumed generator state plus holdout hashes."""

    legacy_positive = tuple(
        template
        for template in POSITIVE_TEMPLATES
        if template.template_id in LEGACY_TEMPLATE_SPLITS
    )
    legacy_negative = tuple(
        template
        for template in HARD_NEGATIVE_TEMPLATES
        if template.template_id in LEGACY_TEMPLATE_SPLITS
    )
    if any(
        spec.value_variant != "legacy"
        for template in (*legacy_positive, *legacy_negative)
        for spec in template.placeholders
    ):
        raise MaterializationError("a pre-v1.3 template changed its legacy value format")

    generator = SyntheticGenerator(seed)
    holdout_values: set[str] = set()
    observed_values = 0
    for split in ratios:
        positive_templates = tuple(
            template
            for template in legacy_positive
            if LEGACY_TEMPLATE_SPLITS[template.template_id] == split
        )
        negative_templates = tuple(
            template
            for template in legacy_negative
            if LEGACY_TEMPLATE_SPLITS[template.template_id] == split
        )
        records = generator.generate_partition(
            positive_count=split_counts[split] - negative_counts[split],
            hard_negative_count=negative_counts[split],
            positive_templates=positive_templates,
            hard_negative_templates=negative_templates,
        )
        if split == "train":
            continue
        for record in records:
            for value_group in record.entity_value_groups:
                observed_values += 1
                holdout_values.add(value_group)
    if observed_values != len(holdout_values):
        raise MaterializationError("replayed v1.2 holdout contains repeated entity values")
    return generator, holdout_values


def _replay_parent_holdout_value_groups(
    *,
    seed: int,
    split_counts: Mapping[str, int],
    negative_counts: Mapping[str, int],
    ratios: Mapping[str, float],
) -> set[str]:
    """Return the committed v1.2 holdout hashes without opening frozen JSONL text."""

    _, holdout_values = _replay_parent_sequence(
        seed=seed,
        split_counts=split_counts,
        negative_counts=negative_counts,
        ratios=ratios,
    )
    return holdout_values


def _verify_parent_replay_commitment(
    replayed_holdout: set[str],
    *,
    contract: FrozenHoldoutContract,
) -> str:
    replay_commitment = _canonical_json_hash(sorted(replayed_holdout))
    if (
        len(replayed_holdout) != contract.entity_value_group_count
        or replay_commitment != contract.entity_value_groups_sha256
    ):
        raise MaterializationError("v1.2 holdout value-group replay differs from its commitment")
    return replay_commitment


def _verify_train_value_isolation_from_frozen_holdout(
    train_records: Sequence[DocumentRecord],
    *,
    replayed_holdout: set[str],
    contract: FrozenHoldoutContract,
    parent_records_preconsumed: int,
) -> dict[str, Any]:
    replay_commitment = _verify_parent_replay_commitment(
        replayed_holdout,
        contract=contract,
    )
    train_values = {
        value_group for record in train_records for value_group in record.entity_value_groups
    }
    collisions = train_values & replayed_holdout
    if collisions:
        raise MaterializationError("generated train values collide with frozen holdout replay")
    return {
        "algorithm_version": "v1.2-parent-sequence-continuation-v1",
        "namespace": "after-parent-train-validation-test",
        "parent_records_preconsumed": parent_records_preconsumed,
        "frozen_jsonl_text_read": False,
        "train_entity_value_groups": len(train_values),
        "replayed_holdout_entity_value_groups": len(replayed_holdout),
        "replayed_holdout_entity_value_groups_sha256": replay_commitment,
        "cross_split_collisions": 0,
    }


def _build_continuation_train_partition(
    *,
    seed: int,
    split_counts: Mapping[str, int],
    negative_counts: Mapping[str, int],
    ratios: Mapping[str, float],
    contract: FrozenHoldoutContract,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    generator, replayed_holdout = _replay_parent_sequence(
        seed=seed,
        split_counts=split_counts,
        negative_counts=negative_counts,
        ratios=ratios,
    )
    _verify_parent_replay_commitment(replayed_holdout, contract=contract)
    train_records = _build_train_partition(
        generator=generator,
        split_counts=split_counts,
        negative_counts=negative_counts,
        ratios=ratios,
    )
    audit = _verify_train_value_isolation_from_frozen_holdout(
        train_records,
        replayed_holdout=replayed_holdout,
        contract=contract,
        parent_records_preconsumed=sum(split_counts.values()),
    )
    return train_records, audit


def _coverage_summary(records: Sequence[DocumentRecord]) -> dict[str, Any]:
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in DEFAULT_SPLIT_RATIOS}
    domain_counts: dict[str, Counter[str]] = {split: Counter() for split in DEFAULT_SPLIT_RATIOS}
    quality_counts: dict[str, Counter[str]] = {split: Counter() for split in DEFAULT_SPLIT_RATIOS}
    for record in records:
        if record.split is None:
            raise MaterializationError("all materialized records must have a split")
        label_counts[record.split].update(entity.label for entity in record.entities)
        domain_counts[record.split][record.domain] += 1
        quality_counts[record.split][record.quality.tier.value] += 1
    return {
        "labels_by_split": {
            split: dict(sorted(counts.items())) for split, counts in label_counts.items()
        },
        "domains_by_split": {
            split: dict(sorted(counts.items())) for split, counts in domain_counts.items()
        },
        "quality_tiers_by_split": {
            split: dict(sorted(counts.items())) for split, counts in quality_counts.items()
        },
    }


def _catalog_template_allocation_hash() -> str:
    templates = {
        template.template_id: template
        for template in (*POSITIVE_TEMPLATES, *HARD_NEGATIVE_TEMPLATES)
    }
    allocation = sorted(
        (f"synthetic-family:{templates[template_id].family}", split)
        for template_id, split in TEMPLATE_SPLITS.items()
    )
    return _canonical_json_hash(allocation)


def _write_and_verify_split(
    records: Sequence[DocumentRecord],
    *,
    split: str,
    destination: Path,
) -> dict[str, Any]:
    selected = [record for record in records if record.split == split]
    written = write_jsonl(selected, destination)
    destination.chmod(0o444)
    verified = 0
    for record in iter_jsonl(destination):
        record.validate()
        if record.split != split or record.data_pool is not DataPool.PUBLIC_RELEASE:
            raise MaterializationError(f"readback isolation check failed for split {split!r}")
        verified += 1
    if written != verified or written != len(selected):
        raise MaterializationError(f"readback count check failed for split {split!r}")
    return {
        "name": destination.name,
        "split": split,
        "records": verified,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "mode": "0444",
    }


def _copy_frozen_split(
    source: Path,
    *,
    contract: FrozenHoldoutFileContract,
    destination: Path,
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    digest = hashlib.sha256()
    copied = 0
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise MaterializationError(f"frozen {contract.split} source is not a regular file")
        with (
            os.fdopen(source_fd, "rb", closefd=False) as source_handle,
            destination.open("xb") as destination_handle,
        ):
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                destination_handle.write(block)
                digest.update(block)
                copied += len(block)
    finally:
        os.close(source_fd)
    destination.chmod(0o444)
    if (
        copied != contract.bytes
        or digest.hexdigest() != contract.sha256
        or destination.stat().st_size != contract.bytes
        or sha256_file(destination) != contract.sha256
    ):
        raise MaterializationError(f"copied frozen {contract.split} bytes changed")
    return {
        "name": contract.name,
        "split": contract.split,
        "records": contract.records,
        "bytes": contract.bytes,
        "sha256": contract.sha256,
        "mode": "0444",
    }


def _merge_coverage(
    train_records: Sequence[DocumentRecord],
    *,
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    train_coverage = _coverage_summary(train_records)
    parent_coverage = parent_manifest["coverage"]
    result: dict[str, Any] = {}
    for section in ("labels_by_split", "domains_by_split", "quality_tiers_by_split"):
        result[section] = {
            "train": train_coverage[section]["train"],
            "validation": dict(parent_coverage[section]["validation"]),
            "test": dict(parent_coverage[section]["test"]),
        }
    return result


def materialize_synthetic_corpus(
    output_dir: Path,
    *,
    plan_path: Path,
    holdout_source_dir: Path,
    holdout_contract: FrozenHoldoutContract = OFFICIAL_FROZEN_HOLDOUT_CONTRACT,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
) -> dict[str, Any]:
    """Generate train and byte-copy cryptographically frozen validation/test splits."""

    output_dir = output_dir.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(mode=0o755)

    try:
        ratios, split_counts, negative_counts = _generation_plan(
            count=count,
            seed=seed,
            hard_negative_ratio=hard_negative_ratio,
            split_ratios=DEFAULT_SPLIT_RATIOS,
        )
        parent_manifest, frozen_paths = _load_frozen_holdout(
            holdout_source_dir,
            contract=holdout_contract,
            count=count,
            seed=seed,
            hard_negative_ratio=hard_negative_ratio,
            split_counts=split_counts,
            negative_counts=negative_counts,
        )
        train_records, value_isolation = _build_continuation_train_partition(
            seed=seed,
            split_counts=split_counts,
            negative_counts=negative_counts,
            ratios=ratios,
            contract=holdout_contract,
        )
        contracts = {item.split: item for item in holdout_contract.files}
        files = [
            _write_and_verify_split(
                train_records,
                split="train",
                destination=staging / "train.jsonl",
            ),
            *[
                _copy_frozen_split(
                    frozen_paths[split],
                    contract=contracts[split],
                    destination=staging / contracts[split].name,
                )
                for split in ("validation", "test")
            ],
        ]
        template_families_by_split = dict(sorted(Counter(TEMPLATE_SPLITS.values()).items()))
        provenance_snapshot = {
            "train": {
                "source_id": "repo_curated_synthetic_templates",
                "source_kind": "curated_deterministic_synthetic",
                "source_revision": f"sha256:{CURATED_ASSET_SHA256}",
                "license": "Apache-2.0",
                "synthetic": True,
                "contains_real_personal_data": False,
            },
            "validation_test": {
                "lineage": "byte_exact_parent_dataset_copy",
                "parent_dataset_version": holdout_contract.parent_dataset_version,
                "parent_provenance_snapshot_sha256": parent_manifest.get(
                    "provenance_snapshot_sha256"
                ),
            },
        }
        manifest: dict[str, Any] = {
            "manifest_schema_version": 2,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "data_pool": DataPool.PUBLIC_RELEASE.value,
            "public_weight_training_allowed": True,
            "plan": {
                "name": plan_path.name,
                "sha256": sha256_file(plan_path),
            },
            "generation": {
                "generator_name": GENERATOR_NAME,
                "generator_version": GENERATOR_VERSION,
                "materializer_version": MATERIALIZER_VERSION,
                "applies_to_splits": ["train"],
                "seed": seed,
                "requested_count": count,
                "requested_hard_negative_ratio": float(hard_negative_ratio),
                "generated_splits": ["train"],
                "byte_copied_splits": ["validation", "test"],
                "value_namespace": {
                    "algorithm_version": "v1.2-parent-sequence-continuation-v1",
                    "parent_dataset_version": holdout_contract.parent_dataset_version,
                    "parent_generator_version": "2.3.0",
                    "replay_runtime_generator_version": GENERATOR_VERSION,
                    "parent_splits_preconsumed": ["train", "validation", "test"],
                    "parent_records_preconsumed": count,
                    "new_train_document_index_start": count,
                    "legacy_value_counters_continue_after_parent_sequence": True,
                    "parent_holdout_entity_value_groups_sha256": (
                        holdout_contract.entity_value_groups_sha256
                    ),
                },
                "surface_shortcuts_rejected_casefolded": list(SURFACE_SHORTCUTS),
            },
            "assets": {
                "template_asset_id": CURATED_ASSET_ID,
                "template_asset_version": CURATED_ASSET_VERSION,
                "template_asset_sha256": CURATED_ASSET_SHA256,
                "curation_audit_sha256": CURATION_AUDIT_SHA256,
                "template_asset_applies_to_splits": ["train"],
                "family_allocation_asset_id": FAMILY_ALLOCATION_ASSET_ID,
                "family_allocation_asset_version": FAMILY_ALLOCATION_ASSET_VERSION,
                "family_allocation_asset_sha256": FAMILY_ALLOCATION_ASSET_SHA256,
                "family_allocation_applies_to_splits": ["train", "validation", "test"],
            },
            "provenance": provenance_snapshot,
            "provenance_snapshot_sha256": _canonical_json_hash(provenance_snapshot),
            "frozen_splits": {
                "policy_version": "byte-exact-parent-holdout-v1",
                "parent_dataset_version": holdout_contract.parent_dataset_version,
                "parent_manifest_sha256": holdout_contract.parent_manifest_sha256,
                "parent_manifest_file_sha256": holdout_contract.parent_manifest_file_sha256,
                "copy_mode": "byte_for_byte",
                "raw_jsonl_records_parsed": False,
                "entity_value_group_count": holdout_contract.entity_value_group_count,
                "entity_value_groups_sha256": holdout_contract.entity_value_groups_sha256,
                "parent_generation": parent_manifest["generation"],
                "parent_assets": parent_manifest["assets"],
                "parent_provenance": parent_manifest["provenance"],
                "parent_provenance_snapshot_sha256": parent_manifest["provenance_snapshot_sha256"],
                "files": {
                    split: {
                        "name": contracts[split].name,
                        "records": contracts[split].records,
                        "bytes": contracts[split].bytes,
                        "source_sha256": contracts[split].sha256,
                        "output_sha256": contracts[split].sha256,
                    }
                    for split in ("validation", "test")
                },
            },
            "splitting": {
                "algorithm_version": SPLIT_ALGORITHM_VERSION,
                "seed": seed,
                "ratios": dict(DEFAULT_SPLIT_RATIOS),
                "group_keys": ["template_group", "entity_value_groups"],
                "family_assignment_uses_seed": False,
                "allocation_policy": "legacy-pinned-plus-explicit-train-only",
                "legacy_template_families_pinned": len(LEGACY_FAMILY_ALLOCATIONS),
                "train_only_addition_families": len(TRAIN_ONLY_FAMILY_ADDITIONS),
                "legacy_template_assignment_sha256": _canonical_json_hash(
                    [
                        (entry.template_id, entry.template_group, entry.split)
                        for entry in LEGACY_FAMILY_ALLOCATIONS
                    ]
                ),
                "train_only_addition_assignment_sha256": _canonical_json_hash(
                    [
                        (
                            entry.template_id,
                            entry.template_group,
                            entry.split,
                            entry.coverage_role,
                        )
                        for entry in TRAIN_ONLY_FAMILY_ADDITIONS
                    ]
                ),
                "template_family_allocation_sha256": _catalog_template_allocation_hash(),
                "template_families_by_split": template_families_by_split,
                "cross_split_group_collisions": 0,
                "entity_value_repetitions": 0,
                "entity_value_isolation": value_isolation,
            },
            "counts": {
                "total": count,
                "positive": count - sum(negative_counts.values()),
                "hard_negative": sum(negative_counts.values()),
                "by_split": dict(sorted(split_counts.items())),
                "hard_negative_by_split": dict(negative_counts),
            },
            "coverage": _merge_coverage(train_records, parent_manifest=parent_manifest),
            "files": files,
        }
        manifest["manifest_sha256"] = _canonical_json_hash(manifest)
        manifest_path = staging / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o444)
        os.replace(staging, output_dir)
        output_dir.chmod(0o555)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_COUNT",
    "DEFAULT_FROZEN_HOLDOUT_SOURCE_DIR",
    "DEFAULT_HARD_NEGATIVE_RATIO",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT_RATIOS",
    "MATERIALIZER_VERSION",
    "FrozenHoldoutContract",
    "FrozenHoldoutFileContract",
    "MaterializationError",
    "OFFICIAL_FROZEN_HOLDOUT_CONTRACT",
    "SPLIT_ALGORITHM_VERSION",
    "SURFACE_SHORTCUTS",
    "assert_surface_contract",
    "materialize_synthetic_corpus",
    "sha256_file",
]
