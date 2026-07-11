from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from pii_zh.data.schema import DataPool, dumps_document
from pii_zh.data.splitting import detect_group_leakage, group_split
from pii_zh.data.synthetic import (
    ALL_TEMPLATES,
    CORE_LABELS,
    CURATED_ASSET_ID,
    CURATED_ASSET_SHA256,
    CURATED_ASSET_VERSION,
    CURATION_AUDIT_SHA256,
    GENERATOR_VERSION,
    HARD_NEGATIVE_TEMPLATES,
    SURFACE_SHORTCUTS,
    SYNTHETIC_CARD_IIN,
    SYNTHETIC_ID_REGION_PREFIX,
    SyntheticGenerator,
    assert_surface_contract,
    build_synthetic_corpus,
    core_label_family_coverage,
    materialize_synthetic_corpus,
)
from pii_zh.data.synthetic.templates import CURATED_ASSET_METADATA, CURATION_AUDIT_METADATA
from pii_zh.data.validators import validate_cn_resident_id, validate_luhn
from pii_zh.taxonomy import load_taxonomy


def test_generation_is_seed_deterministic_and_offsets_are_exact() -> None:
    left = SyntheticGenerator(20260711).generate_dataset(40)
    right = SyntheticGenerator(20260711).generate_dataset(40)
    assert [dumps_document(item) for item in left] == [dumps_document(item) for item in right]
    for record in left:
        record.validate()
        assert record.data_pool is DataPool.PUBLIC_RELEASE
        assert record.provenance.synthetic is True
        assert record.generator_version == GENERATOR_VERSION
        assert record.metadata["contains_real_personal_data"] is False
        assert record.metadata["template_asset_id"] == CURATED_ASSET_ID
        assert record.provenance.source_revision == f"sha256:{CURATED_ASSET_SHA256}"
        assert "synthetic_notice" not in record.metadata
        for entity in record.entities:
            assert record.text[entity.start : entity.end] == entity.text
            assert entity.metadata["synthetic"] is True


def test_at_least_twelve_domains_and_thirty_percent_hard_negatives() -> None:
    records = SyntheticGenerator(7).generate_dataset(50, hard_negative_ratio=0.30)
    negatives = [record for record in records if record.metadata["hard_negative"]]
    assert len(negatives) == math.ceil(len(records) * 0.30)
    assert all(not record.entities for record in negatives)
    positive_domains = {record.domain for record in records if record.entities}
    assert len(positive_domains) >= 12


def test_checksum_values_use_nonproduction_sentinels_and_explicit_provenance() -> None:
    records = SyntheticGenerator(99).generate_dataset(100)
    ids = [
        entity
        for record in records
        for entity in record.entities
        if entity.label == "CN_RESIDENT_ID"
    ]
    cards = [
        entity
        for record in records
        for entity in record.entities
        if entity.label == "BANK_CARD_NUMBER"
    ]
    assert ids and cards
    assert all(entity.text.startswith(SYNTHETIC_ID_REGION_PREFIX) for entity in ids)
    assert all(validate_cn_resident_id(entity.text).valid for entity in ids)
    assert all(entity.text.startswith(SYNTHETIC_CARD_IIN) for entity in cards)
    assert all(validate_luhn(entity.text).valid for entity in cards)


def test_curated_asset_provenance_and_candidate_audit_are_complete() -> None:
    assert CURATED_ASSET_VERSION == "1.0.0"
    assert len(CURATED_ASSET_SHA256) == 64
    assert len(CURATION_AUDIT_SHA256) == 64
    assert CURATED_ASSET_METADATA["license"] == "Apache-2.0"
    source = CURATED_ASSET_METADATA["candidate_source"]
    assert source["model_id"] == "Qwen/Qwen3-8B"
    assert set(source["model_file_hashes"]) == {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
    assert len(source["prompt_hashes"]) == 7
    assert CURATED_ASSET_METADATA["curation"]["raw_rejected_outputs_included"] is False
    assert CURATION_AUDIT_METADATA["reviewed_candidate_count"] == 70
    assert len(CURATION_AUDIT_METADATA["accepted_candidate_ids"]) == 53
    assert len(CURATION_AUDIT_METADATA["rejected_candidates"]) == 17


def test_every_core_label_has_three_distinct_semantic_families() -> None:
    coverage = core_label_family_coverage()
    assert set(CORE_LABELS) == load_taxonomy().core_label_names
    assert set(coverage) == set(CORE_LABELS)
    assert all(len(families) >= 3 for families in coverage.values())
    for label in ("DATE_OF_BIRTH", "SECRET", "MAC_ADDRESS", "QQ_NUMBER", "WECHAT_ID"):
        assert len(coverage[label]) >= 3


def test_training_template_text_has_no_synthetic_shortcut_words() -> None:
    assert HARD_NEGATIVE_TEMPLATES
    assert len({template.domain for template in ALL_TEMPLATES}) >= 12
    for template in ALL_TEMPLATES:
        assert not any(word in template.body for word in ("测试", "示例", "虚构", "演练"))


def test_group_split_is_balanced_leak_free_and_covers_all_core_labels() -> None:
    records = SyntheticGenerator(20260711).generate_dataset(180)
    person_values = [
        entity.text
        for record in records
        for entity in record.entities
        if entity.label == "PERSON_NAME"
    ]
    assert len(person_values) == len(set(person_values))
    split_records = group_split(
        records,
        seed=23,
        required_label_splits=("train", "validation", "test"),
        required_labels=CORE_LABELS,
    )
    assert not detect_group_leakage(split_records).has_leakage

    counts = Counter(record.split for record in split_records)
    expected = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert set(counts) == set(expected)
    for split, ratio in expected.items():
        assert abs(counts[split] / len(split_records) - ratio) <= 0.02
        labels = {
            entity.label
            for record in split_records
            if record.split == split
            for entity in record.entities
        }
        assert set(CORE_LABELS) <= labels

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_templates = {record.template_group for record in split_records if record.split == left}
        right_templates = {
            record.template_group for record in split_records if record.split == right
        }
        left_values = {
            value
            for record in split_records
            if record.split == left
            for value in record.entity_value_groups
        }
        right_values = {
            value
            for record in split_records
            if record.split == right
            for value in record.entity_value_groups
        }
        assert left_templates.isdisjoint(right_templates)
        assert left_values.isdisjoint(right_values)


def test_large_known_seed_has_valid_unique_values_and_no_surface_shortcuts() -> None:
    records = SyntheticGenerator(20260711).generate_dataset(2_000, hard_negative_ratio=0.30)
    assert len(records) == 2_000
    assert_surface_contract(records)
    value_groups = [group for record in records for group in record.entity_value_groups]
    assert len(value_groups) == len(set(value_groups))
    assert set(CORE_LABELS) == {entity.label for record in records for entity in record.entities}
    for record in records:
        record.validate()
        lowered = record.text.casefold()
        assert not any(marker in lowered for marker in SURFACE_SHORTCUTS)
        assert all(
            not any(marker in entity.text.casefold() for marker in SURFACE_SHORTCUTS)
            for entity in record.entities
        )

    names = [
        entity.text
        for record in records
        for entity in record.entities
        if entity.label == "PERSON_NAME"
    ]
    chinese_names = [
        value for value in names if all("\u4e00" <= character <= "\u9fff" for character in value)
    ]
    mixed_names = [value for value in names if value not in chinese_names]
    assert {2, 3, 4} <= {len(value) for value in chinese_names}
    assert 0.08 <= len(mixed_names) / len(names) <= 0.12
    assert len({value[0] for value in names}) >= 40

    addresses = [
        entity.text for record in records for entity in record.entities if entity.label == "ADDRESS"
    ]
    assert len(addresses) == len(set(addresses))
    assert len({value[:8] for value in addresses}) >= 50
    assert all(any(marker in value for value in addresses) for marker in ("栋", "弄", "园", "村"))


def test_materialization_build_is_exact_balanced_and_group_safe() -> None:
    records = build_synthetic_corpus(2_000, seed=20260711, hard_negative_ratio=0.30)
    assert Counter(record.split for record in records) == Counter(
        {"train": 1_600, "validation": 200, "test": 200}
    )
    assert not detect_group_leakage(records).has_leakage
    for split, expected_count in (("train", 1_600), ("validation", 200), ("test", 200)):
        selected = [record for record in records if record.split == split]
        assert len(selected) == expected_count
        assert sum(bool(record.metadata["hard_negative"]) for record in selected) == round(
            expected_count * 0.30
        )
        assert set(CORE_LABELS) == {
            entity.label for record in selected for entity in record.entities
        }


def test_materializer_writes_read_only_files_and_raw_text_free_manifest(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("immutable test plan\n", encoding="utf-8")
    output = tmp_path / "synthetic_v1"
    manifest = materialize_synthetic_corpus(
        output,
        plan_path=plan,
        count=600,
        seed=20260711,
    )
    try:
        assert manifest["counts"]["by_split"] == {
            "test": 60,
            "train": 480,
            "validation": 60,
        }
        assert manifest["counts"]["hard_negative"] == 180
        assert manifest["splitting"]["cross_split_group_collisions"] == 0
        assert set(manifest["coverage"]["labels_by_split"]) == {
            "train",
            "validation",
            "test",
        }
        assert "text" not in manifest
        assert "entities" not in manifest
        assert output.stat().st_mode & 0o777 == 0o555
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json"):
            assert (output / name).stat().st_mode & 0o777 == 0o444
    finally:
        output.chmod(0o755)
        for path in output.iterdir():
            path.chmod(0o644)
