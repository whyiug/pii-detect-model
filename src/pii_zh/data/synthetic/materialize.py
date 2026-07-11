"""Reproducible, leakage-safe materialization of the public synthetic corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..schema import DataPool, DocumentRecord, iter_jsonl, write_jsonl
from ..splitting import assert_no_group_leakage, group_split
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
DATASET_VERSION = "1.0.0"
MATERIALIZER_VERSION = "1.0.0"
SPLIT_ALGORITHM_VERSION = "template-family-pilot-v1"
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


def _derived_seed(seed: int, purpose: str) -> int:
    material = f"{seed}:{purpose}".encode()
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


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


def _allocate_template_families(
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> tuple[
    dict[str, tuple[SyntheticTemplate, ...]],
    dict[str, tuple[SyntheticTemplate, ...]],
]:
    """Assign each positive/negative template family to exactly one split.

    Positive families use the generic group splitter with three-way core-label
    coverage.  Negative families use a 50/25/25 family holdout so validation
    and test retain several unseen negative families; record counts are still
    generated at the requested 80/10/10 and hard-negative ratios.
    """

    split_names = tuple(ratios)
    positive_pilot = SyntheticGenerator(_derived_seed(seed, "positive-template-allocation"))
    positive_records = [
        positive_pilot.generate_document(template) for template in POSITIVE_TEMPLATES
    ]
    positive_split = group_split(
        positive_records,
        ratios=ratios,
        seed=_derived_seed(seed, "positive-template-split"),
        required_label_splits=split_names,
        required_labels=CORE_LABELS,
    )
    positive_assignment = {record.provenance.template_id: record.split for record in positive_split}

    negative_pilot = SyntheticGenerator(_derived_seed(seed, "negative-template-allocation"))
    negative_records = [
        negative_pilot.generate_document(template) for template in HARD_NEGATIVE_TEMPLATES
    ]
    negative_split = group_split(
        negative_records,
        ratios={"train": 0.50, "validation": 0.25, "test": 0.25},
        seed=_derived_seed(seed, "negative-template-split"),
    )
    negative_assignment = {record.provenance.template_id: record.split for record in negative_split}

    positive = {
        split: tuple(
            template
            for template in POSITIVE_TEMPLATES
            if positive_assignment[template.template_id] == split
        )
        for split in split_names
    }
    negative = {
        split: tuple(
            template
            for template in HARD_NEGATIVE_TEMPLATES
            if negative_assignment[template.template_id] == split
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


def build_synthetic_corpus(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> list[DocumentRecord]:
    """Build an exact-ratio corpus with disjoint template and entity groups."""

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
    negative_total = math.ceil(count * float(hard_negative_ratio))
    negative_counts = _integer_allocation(negative_total, ratios)
    positive_templates, negative_templates = _allocate_template_families(
        seed=seed,
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


def _template_allocation_hash(records: Sequence[DocumentRecord]) -> str:
    allocation = sorted(
        {
            (record.template_group, record.split)
            for record in records
            if record.template_group is not None and record.split is not None
        }
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


def materialize_synthetic_corpus(
    output_dir: Path,
    *,
    plan_path: Path,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
) -> dict[str, Any]:
    """Atomically write immutable JSONL splits and a raw-text-free manifest."""

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
        records = build_synthetic_corpus(
            count,
            seed=seed,
            hard_negative_ratio=hard_negative_ratio,
        )
        files = [
            _write_and_verify_split(
                records,
                split=split,
                destination=staging / f"{split}.jsonl",
            )
            for split in DEFAULT_SPLIT_RATIOS
        ]
        by_split = Counter(record.split for record in records)
        hard_negative_by_split = {
            split: sum(
                bool(record.metadata.get("hard_negative"))
                for record in records
                if record.split == split
            )
            for split in DEFAULT_SPLIT_RATIOS
        }
        template_families_by_split = {
            split: len(
                {
                    record.template_group
                    for record in records
                    if record.split == split and record.template_group is not None
                }
            )
            for split in DEFAULT_SPLIT_RATIOS
        }
        provenance_snapshot = {
            "source_id": "repo_curated_synthetic_templates",
            "source_kind": "curated_deterministic_synthetic",
            "source_revision": f"sha256:{CURATED_ASSET_SHA256}",
            "license": "Apache-2.0",
            "synthetic": True,
            "contains_real_personal_data": False,
        }
        manifest: dict[str, Any] = {
            "manifest_schema_version": 1,
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
                "seed": seed,
                "requested_count": count,
                "requested_hard_negative_ratio": float(hard_negative_ratio),
                "surface_shortcuts_rejected_casefolded": list(SURFACE_SHORTCUTS),
            },
            "assets": {
                "template_asset_id": CURATED_ASSET_ID,
                "template_asset_version": CURATED_ASSET_VERSION,
                "template_asset_sha256": CURATED_ASSET_SHA256,
                "curation_audit_sha256": CURATION_AUDIT_SHA256,
            },
            "provenance": provenance_snapshot,
            "provenance_snapshot_sha256": _canonical_json_hash(provenance_snapshot),
            "splitting": {
                "algorithm_version": SPLIT_ALGORITHM_VERSION,
                "seed": seed,
                "ratios": dict(DEFAULT_SPLIT_RATIOS),
                "group_keys": ["template_group", "entity_value_groups"],
                "template_family_allocation_sha256": _template_allocation_hash(records),
                "template_families_by_split": template_families_by_split,
                "cross_split_group_collisions": 0,
                "entity_value_repetitions": 0,
            },
            "counts": {
                "total": len(records),
                "positive": sum(bool(record.entities) for record in records),
                "hard_negative": sum(not record.entities for record in records),
                "by_split": dict(sorted(by_split.items())),
                "hard_negative_by_split": hard_negative_by_split,
            },
            "coverage": _coverage_summary(records),
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
    "DEFAULT_HARD_NEGATIVE_RATIO",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT_RATIOS",
    "MATERIALIZER_VERSION",
    "MaterializationError",
    "SPLIT_ALGORITHM_VERSION",
    "SURFACE_SHORTCUTS",
    "assert_surface_contract",
    "build_synthetic_corpus",
    "materialize_synthetic_corpus",
    "sha256_file",
]
