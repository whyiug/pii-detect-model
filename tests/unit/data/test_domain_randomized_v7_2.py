from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pii_zh.data.synthetic.domain_randomized_v7_2 as v72
from pii_zh.data.schema import DataPool, iter_jsonl
from pii_zh.data.synthetic.domain_randomized_v7_2 import (
    CUE_PLACEMENTS,
    DATASET_ID,
    DATASET_VERSION,
    GENERATION_ORDER,
    GROUP_KEYS,
    POSITIVE_COMPOSITIONS,
    DomainRandomizedV72Error,
    build_domain_randomized_v7_2_records,
    load_domain_randomized_v7_2_config,
    materialize_domain_randomized_v7_2,
)
from pii_zh.data.synthetic.templates import CORE_LABELS

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs/data/domain_randomized_v7_2.yaml"
_SMALL_COUNT = 2_560


def _small_config_text() -> str:
    return (
        _CONFIG.read_text(encoding="utf-8")
        .replace("train: 72000", f"train: {_SMALL_COUNT}")
        .replace("dev: 8000", f"dev: {_SMALL_COUNT}")
        .replace("independent_test: 8000", f"independent_test: {_SMALL_COUNT}")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_default_config_is_three_way_clean_room_and_test_is_one_shot() -> None:
    config = load_domain_randomized_v7_2_config(_CONFIG, repository_root=_ROOT)

    assert config.split_counts == {
        "train": 72_000,
        "dev": 8_000,
        "independent_test": 8_000,
    }
    assert config.pii_free_hard_negative_ratio == 0.40
    assert config.natural_positive_ratio == 0.75
    assert config.cue_placements == CUE_PLACEMENTS
    assert DATASET_ID == "pii_zh_domain_randomized_v7"
    assert DATASET_VERSION == "7.2.0"
    assert GENERATION_ORDER == ("independent_test", "train", "dev")
    assert config.clean_room == {
        "benchmark_rows_read": False,
        "benchmark_predictions_read": False,
        "benchmark_templates_read": False,
        "prior_dataset_rows_read": False,
        "prior_training_rows_read": False,
        "external_dataset_inputs": [],
        "repository_authored_grammars_only": True,
    }
    test_policy = config.usage_policy["independent_test"]
    assert test_policy["evaluation_only"] is True
    assert test_policy["public_weight_training_allowed"] is False
    assert test_policy["checkpoint_or_model_selection_allowed"] is False
    assert test_policy["threshold_or_fusion_selection_allowed"] is False
    assert test_policy["candidate_must_be_frozen_before_content_access"] is True
    assert test_policy["one_shot_only"] is True


def test_every_split_and_cue_placement_has_four_unique_surface_patterns() -> None:
    for split in ("train", "dev", "independent_test"):
        surfaces = v72._SPLIT_SURFACES[split]
        for placement in ("left", "right", "bilateral"):
            patterns = surfaces[placement]
            assert len(patterns) == 4
            assert len(set(patterns)) == 4


def test_semantic_clause_places_label_cue_on_declared_side() -> None:
    for split in ("train", "dev", "independent_test"):
        for placement in CUE_PLACEMENTS:
            for variant in range(4):
                spec = {
                    "placeholder": "VALUE_1",
                    "label": "ADDRESS",
                    "cue_placement": placement,
                    "alias_index": variant % 3,
                    "surface_variant": variant,
                    "semantic_family": "unused-by-renderer",
                }
                clause = v72._semantic_clause(split, spec)
                alias = v72._LABEL_ALIASES["ADDRESS"][variant % 3]
                before, after = clause.split("<<VALUE_1>>", 1)
                expected = {
                    "left_only": (True, False),
                    "right_only": (False, True),
                    "bilateral": (True, True),
                }[placement]
                assert (alias in before, alias in after) == expected


def test_small_generation_is_deterministic_balanced_and_pairwise_isolated() -> None:
    config = load_domain_randomized_v7_2_config(_CONFIG, repository_root=_ROOT)
    small = replace(
        config,
        split_counts={split: _SMALL_COUNT for split in ("train", "dev", "independent_test")},
    )
    left, left_audit = build_domain_randomized_v7_2_records(small)
    right, right_audit = build_domain_randomized_v7_2_records(small)

    assert [record.to_dict() for record in left] == [record.to_dict() for record in right]
    assert left_audit == right_audit
    assert [record.split for record in left[:_SMALL_COUNT]] == ["independent_test"] * _SMALL_COUNT
    assert left_audit["counts"]["by_split"] == {
        "train": _SMALL_COUNT,
        "dev": _SMALL_COUNT,
        "independent_test": _SMALL_COUNT,
    }
    assert left_audit["counts"]["hard_negative_by_split"] == {
        "train": 1_024,
        "dev": 1_024,
        "independent_test": 1_024,
    }
    assert left_audit["isolation"]["group_keys"] == list(GROUP_KEYS)
    assert left_audit["isolation"]["pairwise_collision_total"] == 0
    assert all(
        count == 0
        for pair in left_audit["isolation"]["pairwise_collisions_by_group"].values()
        for count in pair.values()
    )

    for split in ("train", "dev", "independent_test"):
        label_counts = left_audit["composition"]["label_counts_by_split"][split]
        assert set(label_counts) == set(CORE_LABELS)
        assert set(label_counts.values()) == {72}
        expected_pool = DataPool.PUBLIC_RELEASE if split == "train" else DataPool.EVALUATION_ONLY
        selected = [record for record in left if record.split == split]
        assert all(record.data_pool is expected_pool for record in selected)
        assert all(
            record.public_weight_training_allowed is (split == "train") for record in selected
        )
        composition = left_audit["composition"]["primary_label_counts_by_split_and_composition"][
            split
        ]
        assert set(composition) == set(POSITIVE_COMPOSITIONS)
        for per_label in composition.values():
            assert set(per_label) == set(CORE_LABELS)
            assert len(set(per_label.values())) == 1

        natural = left_audit["coverage"]["natural_context"]["by_split"][split]
        assert natural["documents"] == 1_152
        assert natural["document_ratio"] == 0.75
        for label_coverage in natural["by_label"].values():
            placements = label_coverage["cue_placements"]
            assert set(placements) == set(CUE_PLACEMENTS)
            assert max(placements.values()) - min(placements.values()) <= 1
            assert label_coverage["right_visible_ratio"] >= 2 / 3
            assert label_coverage["alias_count"] == 3
            assert label_coverage["surface_variant_count"] == 4
            assert label_coverage["semantic_skeleton_count"] >= 24
            assert label_coverage["minimum_right_context_chars"] >= 4


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("benchmark_rows_read: false", "benchmark_rows_read: true"),
        ("prior_dataset_rows_read: false", "prior_dataset_rows_read: true"),
        ("prior_training_rows_read: false", "prior_training_rows_read: true"),
        ("external_dataset_inputs: []", "external_dataset_inputs: [forbidden.jsonl]"),
        (
            "candidate_must_be_frozen_before_content_access: true",
            "candidate_must_be_frozen_before_content_access: false",
        ),
    ),
)
def test_config_rejects_external_inputs_or_test_policy_drift(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _CONFIG.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    with pytest.raises(DomainRandomizedV72Error):
        load_domain_randomized_v7_2_config(config_path, repository_root=_ROOT)


def test_materializer_seals_test_first_binds_source_taxonomy_and_never_clobbers(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_small_config_text(), encoding="utf-8")
    output = tmp_path / "dataset"

    manifest = materialize_domain_randomized_v7_2(
        output,
        config_path=config_path,
        repository_root=_ROOT,
    )

    assert manifest["counts"]["by_split"] == {
        "train": _SMALL_COUNT,
        "dev": _SMALL_COUNT,
        "independent_test": _SMALL_COUNT,
    }
    assert manifest["write_order"][:2] == [
        "independent_test.jsonl",
        "independent_test.seal.json",
    ]
    assert manifest["generation"]["test_was_written_and_sealed_before_train_dev"] is True
    assert manifest["audits"]["isolation"]["pairwise_collision_total"] == 0
    assert manifest["clean_room"]["external_dataset_inputs"] == []
    seal_path = output / "independent_test.seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["seal_created_before_train_and_dev_files"] is True
    assert seal["content_access_boundary"] == {
        "candidate_must_be_frozen": True,
        "training_forbidden": True,
        "checkpoint_model_threshold_and_fusion_selection_forbidden": True,
        "one_shot_final_evaluation_only": True,
    }
    assert seal["bindings"]["generator_source"]["sha256"] == _sha256_file(
        Path(v72.__file__).resolve()
    )
    assert seal["bindings"]["taxonomy"]["sha256"] == _sha256_file(
        _ROOT / "src/pii_zh/taxonomy/taxonomy.yaml"
    )
    assert seal["bindings"]["independent_test_file"]["sha256"] == _sha256_file(
        output / "independent_test.jsonl"
    )
    assert manifest["independent_test_seal"]["file_sha256"] == _sha256_file(seal_path)
    assert (output.stat().st_mode & 0o777) == 0o555
    assert all(
        (output / name).stat().st_mode & 0o777 == 0o444
        for name in (
            "train.jsonl",
            "dev.jsonl",
            "independent_test.jsonl",
            "independent_test.seal.json",
            "dataset_manifest.json",
        )
    )
    first_test_record = next(iter_jsonl(output / "independent_test.jsonl"))
    assert first_test_record.data_pool is DataPool.EVALUATION_ONLY
    assert first_test_record.public_weight_training_allowed is False
    assert first_test_record.provenance.evaluation_only is True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_domain_randomized_v7_2(
            output,
            config_path=config_path,
            repository_root=_ROOT,
        )
