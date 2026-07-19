from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from pii_zh.data.synthetic.sota_v1 import (
    DATASET_ID,
    GROUP_KEYS,
    SyntheticSotaMaterializationError,
    SyntheticSotaValueFactory,
    assert_synthetic_sota_isolation,
    build_near_duplicate_groups,
    build_synthetic_sota_records,
    load_synthetic_sota_config,
    materialize_synthetic_sota_corpus,
)

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs/data/synthetic_sota_v1.yaml"


def test_release_config_pins_scale_sources_and_protected_input_denials() -> None:
    config = load_synthetic_sota_config(_CONFIG, repository_root=_ROOT)
    assert config.split_counts == {"train": 80_000, "validation": 10_000}
    assert config.pii_free_ratio == 0.45
    assert config.hard_negative_ratio == 0.45
    assert config.source_registry_entry["declared_license"] == "Apache-2.0"
    assert config.source_registry_entry["public_weight_training_allowed"] is True


def test_small_generation_is_deterministic_24_class_and_strictly_isolated() -> None:
    config = load_synthetic_sota_config(_CONFIG, repository_root=_ROOT)
    small = replace(config, split_counts={"train": 300, "validation": 150})
    left, left_audit = build_synthetic_sota_records(small)
    right, right_audit = build_synthetic_sota_records(small)
    assert [record.to_dict() for record in left] == [record.to_dict() for record in right]
    assert left_audit == right_audit
    assert len(left) == 450
    assert left_audit["isolation"]["cross_split_collisions"] == 0
    assert left_audit["isolation"]["group_keys"] == list(GROUP_KEYS)
    assert all(
        set(left_audit["label_counts_by_split"][split])
        == set(left_audit["label_counts_by_split"]["train"])
        for split in ("train", "validation")
    )
    assert len(left_audit["label_counts_by_split"]["train"]) == 24
    for split, expected in (("train", 135), ("validation", 68)):
        selected = [record for record in left if record.split == split]
        assert sum(not record.entities for record in selected) == expected
        assert sum(record.metadata["hard_negative"] is True for record in selected) == expected
    assert assert_synthetic_sota_isolation(left)["cross_split_collisions"] == 0
    assert left_audit["exact_texts_globally_unique"] is True
    assert sum(
        record.provenance.template_id == "hard_negative_public_hotline" for record in left
    ) == 1


def test_near_duplicate_components_are_stable_and_do_not_cross_pinned_splits() -> None:
    groups, audit = build_near_duplicate_groups()
    assert groups["government_documents_005"] == groups["government_documents_006"]
    assert audit["method"] == "normalized-placeholder-char-3gram-jaccard-components-v1"
    assert audit["threshold"] == 0.75
    assert audit["similar_pair_count"] >= 1


def test_scale_safe_name_factory_does_not_repeat_at_release_scale() -> None:
    factory = SyntheticSotaValueFactory(random.Random(20260717))
    names = [factory.person_name() for _ in range(30_000)]
    assert len(names) == len(set(names))
    masked_phones = [factory.masked_phone() for _ in range(5_000)]
    quad_versions = [factory.quad_version() for _ in range(5_000)]
    assert len(masked_phones) == len(set(masked_phones))
    assert len(quad_versions) == len(set(quad_versions))


def test_config_rejects_neutral_negative_gap_until_an_asset_is_versioned(tmp_path: Path) -> None:
    raw = _CONFIG.read_text(encoding="utf-8").replace(
        "hard_negative_ratio: 0.45", "hard_negative_ratio: 0.35"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(raw, encoding="utf-8")
    with pytest.raises(SyntheticSotaMaterializationError, match="every PII-free"):
        load_synthetic_sota_config(config_path, repository_root=_ROOT)


def test_materializer_writes_read_only_manifest_without_evaluation_inputs(
    tmp_path: Path,
) -> None:
    raw = _CONFIG.read_text(encoding="utf-8")
    raw = raw.replace("train: 80000", "train: 300").replace(
        "validation: 10000", "validation: 150"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(raw, encoding="utf-8")
    output = tmp_path / "dataset"
    manifest = materialize_synthetic_sota_corpus(
        output,
        config_path=config_path,
        repository_root=_ROOT,
    )
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["counts"]["by_split"] == {"train": 300, "validation": 150}
    assert manifest["generation"]["external_dataset_inputs"] == []
    assert manifest["generation"]["protected_evaluation_data_read"] is False
    assert manifest["audits"]["isolation"]["cross_split_collisions"] == 0
    disk_manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["manifest_sha256"] == manifest["manifest_sha256"]
    assert (output.stat().st_mode & 0o777) == 0o555
    assert all((output / name).stat().st_mode & 0o777 == 0o444 for name in (
        "train.jsonl",
        "validation.jsonl",
        "dataset_manifest.json",
    ))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_synthetic_sota_corpus(
            output,
            config_path=config_path,
            repository_root=_ROOT,
        )
