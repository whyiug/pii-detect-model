from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from pii_zh.data.schema import DataPool, dumps_document
from pii_zh.data.splitting import detect_group_leakage, group_split
from pii_zh.data.synthetic import (
    ALL_TEMPLATES,
    CORE_LABELS,
    CURATED_ASSET_ID,
    CURATED_ASSET_SHA256,
    CURATED_ASSET_VERSION,
    CURATION_AUDIT_SHA256,
    DATASET_VERSION,
    FAMILY_ALLOCATION_ALGORITHM_VERSION,
    FAMILY_ALLOCATION_ASSET_ID,
    FAMILY_ALLOCATION_ASSET_SHA256,
    FAMILY_ALLOCATION_ASSET_VERSION,
    GENERATOR_VERSION,
    HARD_NEGATIVE_TEMPLATES,
    MATERIALIZER_VERSION,
    OFFICIAL_FROZEN_HOLDOUT_CONTRACT,
    POSITIVE_TEMPLATES,
    SURFACE_SHORTCUTS,
    SYNTHETIC_CARD_IIN,
    SYNTHETIC_ID_REGION_PREFIX,
    TEMPLATE_SPLITS,
    TRAIN_ONLY_ADDITION_TEMPLATE_IDS,
    TRAIN_ONLY_FAMILY_ADDITIONS,
    FamilyAllocationAssetError,
    FrozenHoldoutContract,
    FrozenHoldoutFileContract,
    MaterializationError,
    SyntheticGenerator,
    assert_surface_contract,
    core_label_family_coverage,
    materialize_synthetic_corpus,
    validate_template_family_catalog,
)
from pii_zh.data.synthetic.allocation import (
    ALL_FAMILY_ALLOCATIONS,
    FAMILY_ALLOCATION_METADATA,
    LEGACY_TEMPLATE_SPLITS,
)
from pii_zh.data.synthetic.materialize import (
    _build_continuation_train_partition,
    _build_regenerated_test_corpus,
    _canonical_json_hash,
    _generation_plan,
    _replay_parent_holdout_value_groups,
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
    assert CURATED_ASSET_VERSION == "1.2.0"
    assert GENERATOR_VERSION == "2.4.0"
    assert DATASET_VERSION == "1.3.0"
    assert MATERIALIZER_VERSION == "1.3.0"
    assert len(CURATED_ASSET_SHA256) == 64
    assert len(CURATION_AUDIT_SHA256) == 64
    assert CURATED_ASSET_METADATA["license"] == "Apache-2.0"
    assert CURATED_ASSET_METADATA["schema_version"] == 2
    source = CURATED_ASSET_METADATA["candidate_source"]
    assert source["model_id"] == "Qwen/Qwen3-8B"
    assert set(source["model_file_hashes"]) == {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
    assert len(source["prompt_hashes"]) == 7
    assert CURATED_ASSET_METADATA["curation"]["raw_rejected_outputs_included"] is False
    assert CURATED_ASSET_METADATA["curation"]["human_authored_positive_count"] == 34
    assert CURATED_ASSET_METADATA["curation"]["human_authored_hard_negative_count"] == 20
    assert CURATION_AUDIT_METADATA["reviewed_candidate_count"] == 70
    assert len(CURATION_AUDIT_METADATA["accepted_candidate_ids"]) == 53
    assert len(CURATION_AUDIT_METADATA["rejected_candidates"]) == 17


def test_stable_family_allocation_exactly_preserves_the_legacy_snapshot() -> None:
    assert FAMILY_ALLOCATION_ASSET_ID == "pii-zh-synthetic-stable-family-allocation"
    assert FAMILY_ALLOCATION_ASSET_VERSION == "3.0.0"
    assert FAMILY_ALLOCATION_ALGORITHM_VERSION == "stable-pinned-v3"
    assert (
        FAMILY_ALLOCATION_ASSET_SHA256
        == "49b3c265991f2659ae3796f278da48361c440f57acbd91665df06616bf071647"
    )
    source_snapshot = FAMILY_ALLOCATION_METADATA["source_snapshot"]
    assert source_snapshot["fields_read"] == [
        "provenance.template_id",
        "template_group",
        "split",
    ]
    assert source_snapshot["raw_text_or_entity_values_read"] is False
    assert source_snapshot["source_asset_filename"] == "stable_family_allocation_v2.json"
    assert (
        source_snapshot["source_asset_sha256"]
        == "755e4d77e18003afd82bbb30a91942022fe68a19acb9e5e310592fc39552984f"
    )

    expected_validation_ids = {
        "authored_birth_hospital_registration",
        "authored_mac_asset_inventory",
        "authored_secret_mobile_session",
        "authored_wechat_class_group",
        "contact_and_identity_002",
        "digital_accounts_004",
        "financial_002",
        "government_documents_001",
        "hard_negative_generic_date",
        "hard_negative_invalid_card",
        "hard_negative_invalid_resident_id",
        "hard_negative_organization",
        "html_bank_account_driver_license",
        "security_location_vehicle_001",
        "table_passport_number_qq_number",
    }
    expected_test_ids = {
        "asr_address_email",
        "asr_ip_address_mac_address",
        "authored_birth_student_profile",
        "authored_employee_payroll_account",
        "authored_qq_recruiting_contact",
        "authored_secret_ci_deployment",
        "financial_005",
        "financial_008",
        "government_documents_002",
        "government_documents_009",
        "hard_negative_invoice_code",
        "hard_negative_metrics",
        "hard_negative_region_code",
        "hard_negative_trace_version",
        "security_location_vehicle_007",
        "table_medical_record_wechat_id",
    }
    all_template_ids = {template.template_id for template in ALL_TEMPLATES}
    expected_legacy_ids = all_template_ids - TRAIN_ONLY_ADDITION_TEMPLATE_IDS
    assert len(expected_legacy_ids) == len(LEGACY_TEMPLATE_SPLITS) == 96
    assert set(LEGACY_TEMPLATE_SPLITS) == expected_legacy_ids
    assert {
        template_id
        for template_id, split in LEGACY_TEMPLATE_SPLITS.items()
        if split == "validation"
    } == expected_validation_ids
    assert {
        template_id for template_id, split in LEGACY_TEMPLATE_SPLITS.items() if split == "test"
    } == expected_test_ids
    assert {
        template_id for template_id, split in LEGACY_TEMPLATE_SPLITS.items() if split == "train"
    } == expected_legacy_ids - expected_validation_ids - expected_test_ids

    repository_root = Path(__file__).parents[3]
    prior = json.loads(
        (
            repository_root / "src/pii_zh/data/synthetic/assets/stable_family_allocation_v2.json"
        ).read_text(encoding="utf-8")
    )
    prior_splits = {
        item["template_id"]: item["split"]
        for item in prior["legacy_assignments"] + prior["train_only_additions"]
    }
    assert len(prior_splits) == 96
    assert LEGACY_TEMPLATE_SPLITS == prior_splits

    templates_by_id = {template.template_id: template for template in ALL_TEMPLATES}
    assert {entry.template_id: entry.template_group for entry in ALL_FAMILY_ALLOCATIONS} == {
        template_id: f"synthetic-family:{template.family}"
        for template_id, template in templates_by_id.items()
    }


def test_new_family_additions_are_explicitly_train_only_and_fail_closed() -> None:
    expected = {
        "authored_vehicle_plate_parking_subscription",
        "authored_vehicle_plate_inspection_queue",
        "authored_vehicle_plate_toll_dispute",
        "authored_employee_shift_roster",
        "authored_employee_facility_access",
        "authored_birthdate_age_verification",
        "authored_birthdate_benefit_enrollment",
        "authored_driver_license_ride_hailing",
        "authored_coordinate_rescue_dispatch",
        "authored_mac_zero_trust_admission",
        "authored_passport_cross_border_lodging",
    }
    assert TRAIN_ONLY_ADDITION_TEMPLATE_IDS == expected
    assert Counter(entry.coverage_role for entry in TRAIN_ONLY_FAMILY_ADDITIONS) == Counter(
        {
            "VEHICLE_LICENSE_PLATE": 3,
            "EMPLOYEE_ID": 2,
            "DATE_OF_BIRTH": 2,
            "DRIVER_LICENSE_NUMBER": 1,
            "GEO_COORDINATE": 1,
            "MAC_ADDRESS": 1,
            "PASSPORT_NUMBER": 1,
        }
    )
    assert "SECRET" not in {entry.coverage_role for entry in TRAIN_ONLY_FAMILY_ADDITIONS}
    assert all(TEMPLATE_SPLITS[template_id] == "train" for template_id in expected)

    unregistered = replace(ALL_TEMPLATES[0], template_id="unregistered_future_template")
    with pytest.raises(FamilyAllocationAssetError, match="explicit family allocation"):
        validate_template_family_catalog((*ALL_TEMPLATES, unregistered))

    drifted = list(ALL_TEMPLATES)
    drifted[0] = replace(drifted[0], family="changed_family")
    with pytest.raises(FamilyAllocationAssetError, match="template family changed"):
        validate_template_family_catalog(drifted)

    role_drifted = list(ALL_TEMPLATES)
    index = next(
        index
        for index, template in enumerate(role_drifted)
        if template.template_id == "authored_coordinate_rescue_dispatch"
    )
    target = role_drifted[index]
    role_drifted[index] = replace(
        target,
        placeholders=(replace(target.placeholders[0], label="SECRET"),),
    )
    with pytest.raises(FamilyAllocationAssetError, match="positive coverage role changed"):
        validate_template_family_catalog(role_drifted)

    variant_drifted = list(ALL_TEMPLATES)
    index = next(
        index
        for index, template in enumerate(variant_drifted)
        if template.template_id == "markdown_social_security_vehicle_plate"
    )
    target = variant_drifted[index]
    vehicle_index = next(
        index
        for index, spec in enumerate(target.placeholders)
        if spec.label == "VEHICLE_LICENSE_PLATE"
    )
    placeholders = list(target.placeholders)
    placeholders[vehicle_index] = replace(
        placeholders[vehicle_index],
        value_variant="fictional_compact",
    )
    variant_drifted[index] = replace(target, placeholders=tuple(placeholders))
    with pytest.raises(FamilyAllocationAssetError, match="must keep legacy value formats"):
        validate_template_family_catalog(variant_drifted)


def test_source_registry_binds_the_stable_family_allocation_asset() -> None:
    repository_root = Path(__file__).parents[3]
    registry = yaml.safe_load(
        (repository_root / "configs/data/source_registry.yaml").read_text(encoding="utf-8")
    )
    source = next(
        item for item in registry["sources"] if item["id"] == "repo_curated_synthetic_templates"
    )
    allocation = source["family_allocation_asset"]
    assert source["template_schema_version"] == 2
    assert source["template_asset_version"] == "1.2.0"
    assert source["revision"] == f"sha256:{CURATED_ASSET_SHA256}"
    assert source["curation_audit_sha256"] == CURATION_AUDIT_SHA256
    assert source["generator_version"] == "2.4.0"
    assert source["dataset_version"] == "1.3.0"
    assert source["materializer_version"] == "1.3.0"
    assert source["split_algorithm_version"] == FAMILY_ALLOCATION_ALGORITHM_VERSION
    assert allocation == {
        "locator": "src/pii_zh/data/synthetic/assets/stable_family_allocation_v3.json",
        "revision": f"sha256:{FAMILY_ALLOCATION_ASSET_SHA256}",
        "prior_revision": (
            "sha256:755e4d77e18003afd82bbb30a91942022fe68a19acb9e5e310592fc39552984f"
        ),
        "pre_v1_3_render_projection_sha256": (
            "df4c63239ae319a287ed7ecaa966aa3fdbbdaeff3987f3ecdd6916b60de53b76"
        ),
        "legacy_template_families_pinned": 96,
        "train_only_addition_families": 11,
        "unregistered_template_policy": "reject",
    }
    assert source["frozen_holdout"] == {
        "parent_dataset_version": "1.2.0",
        "parent_manifest_sha256": (
            "58a6067b93c04ba4a4d3a245145a0a501af07d6eb4f3c740b8e82e5681b07c2e"
        ),
        "parent_manifest_file_sha256": (
            "e1bd5a6b094bdcd6566c94f5774313505924051d551d74c74617a21e309d023b"
        ),
        "entity_value_group_count": 6832,
        "entity_value_groups_sha256": (
            "28d7eda8260a8f908defef2a36d4434500b0c3244aa5d62e9d7daf3988306ae3"
        ),
        "train_value_namespace_algorithm": "v1.2-parent-sequence-continuation-v1",
        "parent_records_preconsumed": 20000,
        "validation_sha256": ("9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d"),
        "test_sha256": ("26e49949d0bd876b51c693b7b31135b5ce503ca6a5ab7cfca37177e0ccbd4384"),
    }


def test_release_api_has_no_regenerated_holdout_bypass_and_cli_is_frozen() -> None:
    from scripts.materialize_synthetic_data import parse_args

    import pii_zh.data.synthetic as synthetic_package

    assert not hasattr(synthetic_package, "build_synthetic_corpus")
    args = parse_args([])
    assert args.output_dir.name == "synthetic_v1_3"
    for argv in (
        ["--count", "600"],
        ["--seed", "7"],
        ["--hard-negative-ratio", "0.5"],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_every_core_label_has_three_distinct_semantic_families() -> None:
    coverage = core_label_family_coverage()
    assert set(CORE_LABELS) == load_taxonomy().core_label_names
    assert set(coverage) == set(CORE_LABELS)
    assert all(len(families) >= 3 for families in coverage.values())
    for label in ("DATE_OF_BIRTH", "SECRET", "MAC_ADDRESS", "QQ_NUMBER", "WECHAT_ID"):
        assert len(coverage[label]) >= 3

    train_templates = tuple(
        template
        for template in POSITIVE_TEMPLATES
        if TEMPLATE_SPLITS[template.template_id] == "train"
    )
    train_coverage = core_label_family_coverage(train_templates)
    assert all(len(families) >= 3 for families in train_coverage.values())
    assert {
        label: len(train_coverage[label])
        for label in (
            "VEHICLE_LICENSE_PLATE",
            "EMPLOYEE_ID",
            "DATE_OF_BIRTH",
            "DRIVER_LICENSE_NUMBER",
            "GEO_COORDINATE",
            "MAC_ADDRESS",
            "PASSPORT_NUMBER",
        )
    } == {
        "VEHICLE_LICENSE_PLATE": 5,
        "EMPLOYEE_ID": 5,
        "DATE_OF_BIRTH": 3,
        "DRIVER_LICENSE_NUMBER": 3,
        "GEO_COORDINATE": 3,
        "MAC_ADDRESS": 3,
        "PASSPORT_NUMBER": 3,
    }


def test_v13_train_only_families_have_diverse_boundaries_and_value_variants() -> None:
    expected = {
        "authored_vehicle_plate_parking_subscription": (
            "VEHICLE_LICENSE_PLATE_1",
            "VEHICLE_LICENSE_PLATE",
            "《",
            "》",
            "fictional_compact",
            r"岚B[A-Z0-9]{5}",
        ),
        "authored_vehicle_plate_inspection_queue": (
            "VEHICLE_LICENSE_PLATE_1",
            "VEHICLE_LICENSE_PLATE",
            "plate=",
            ";",
            "fictional_hyphenated",
            r"岚C-[A-Z0-9]{6}",
        ),
        "authored_vehicle_plate_toll_dispute": (
            "VEHICLE_LICENSE_PLATE_1",
            "VEHICLE_LICENSE_PLATE",
            "（",
            "/",
            "fictional_extended",
            r"岚D·[A-Z0-9]{8}",
        ),
        "authored_employee_shift_roster": (
            "EMPLOYEE_ID_1",
            "EMPLOYEE_ID",
            "工号:",
            "}",
            "department_hyphenated",
            r"DPT-ZZ-[0-9]{8}",
        ),
        "authored_employee_facility_access": (
            "EMPLOYEE_ID_1",
            "EMPLOYEE_ID",
            'employee_no="',
            '"',
            "staff_slash",
            r"STAFF/ZZ/[0-9]{8}",
        ),
        "authored_birthdate_age_verification": (
            "DATE_OF_BIRTH_1",
            "DATE_OF_BIRTH",
            "birth_date/",
            "/",
            "compact_yyyymmdd",
            r"[0-9]{8}",
        ),
        "authored_birthdate_benefit_enrollment": (
            "DATE_OF_BIRTH_1",
            "DATE_OF_BIRTH",
            "（",
            "）",
            "dot_separated",
            r"[0-9]{4}\.[0-9]{2}\.[0-9]{2}",
        ),
        "authored_driver_license_ride_hailing": (
            "DRIVER_LICENSE_NUMBER_1",
            "DRIVER_LICENSE_NUMBER",
            "license_no=",
            "]",
            "fictional_hyphenated",
            r"DL-ZZ-[A-Z0-9]{8}",
        ),
        "authored_coordinate_rescue_dispatch": (
            "GEO_COORDINATE_1",
            "GEO_COORDINATE",
            "(",
            ")",
            "signed_spaced",
            r"[+-][0-9]+\.[0-9]{6}, [+-][0-9]+\.[0-9]{6}",
        ),
        "authored_mac_zero_trust_admission": (
            "MAC_ADDRESS_1",
            "MAC_ADDRESS",
            '"client_mac":"',
            '"',
            "hyphen_lowercase",
            r"02-00-00-[0-9a-f]{2}-[0-9a-f]{2}-[0-9a-f]{2}",
        ),
        "authored_passport_cross_border_lodging": (
            "PASSPORT_NUMBER_1",
            "PASSPORT_NUMBER",
            "passport#",
            "/",
            "fictional_hyphenated",
            r"PZ-[A-Z0-9]{8}",
        ),
    }
    templates = {template.template_id: template for template in POSITIVE_TEMPLATES}
    observed_boundaries: set[tuple[str, str]] = set()
    observed_domains: set[str] = set()
    generator = SyntheticGenerator(20260711)
    for template_id, contract in expected.items():
        placeholder, label, left_boundary, right_boundary, variant, pattern = contract
        template = templates[template_id]
        left, right = template.body.split(f"<<{placeholder}>>")
        assert template.origin_kind == "human_authored"
        assert template.source_candidate_id is None
        assert left.endswith(left_boundary)
        assert right.startswith(right_boundary)
        assert TEMPLATE_SPLITS[template_id] == "train"
        observed_boundaries.add((left_boundary, right_boundary))
        observed_domains.add(template.domain)

        record = generator.generate_document(template)
        record.validate()
        assert len(record.entities) == 1
        entity = record.entities[0]
        assert entity.label == label
        assert entity.metadata["value_variant"] == variant
        assert re.fullmatch(pattern, entity.text)
        assert record.provenance.license == "Apache-2.0"
        assert record.quality.validators_passed is True
        assert record.metadata["contains_real_personal_data"] is False
    assert len(observed_boundaries) == 11
    assert len(observed_domains) == 11


def test_pre_v13_templates_keep_legacy_render_and_stable_metadata_projection() -> None:
    templates = {template.template_id: template for template in ALL_TEMPLATES}
    projection = []
    for template_id in sorted(LEGACY_TEMPLATE_SPLITS):
        template = templates[template_id]
        assert all(spec.value_variant == "legacy" for spec in template.placeholders)
        record = SyntheticGenerator(20260711).generate_document(template)
        item = record.to_dict()
        item.pop("generator_version")
        item["provenance"].pop("source_revision")
        item["provenance"].pop("generator_version")
        for key in ("template_asset_version", "template_asset_sha256", "curation_audit_sha256"):
            item["provenance"]["metadata"].pop(key)
        item["metadata"].pop("template_asset_version")
        projection.append(item)
    assert _canonical_json_hash(projection) == (
        "df4c63239ae319a287ed7ecaa966aa3fdbbdaeff3987f3ecdd6916b60de53b76"
    )


def test_authored_secret_and_medical_families_have_diverse_context_boundaries() -> None:
    templates = {template.template_id: template for template in POSITIVE_TEMPLATES}
    generator = SyntheticGenerator(20260711)
    secret_boundaries = {
        "authored_secret_database_password": ("password='", "'，"),
        "authored_secret_bearer_header": ("Bearer ", "；"),
        "authored_secret_private_key_fragment": ("[", "]"),
    }
    for template_id, (left_boundary, right_boundary) in secret_boundaries.items():
        template = templates[template_id]
        left, right = template.body.split("<<SECRET_1>>")
        assert template.origin_kind == "human_authored"
        assert template.source_candidate_id is None
        assert left.endswith(left_boundary)
        assert right.startswith(right_boundary)
        assert {spec.label for spec in template.placeholders} >= {"SECRET"}
        record = generator.generate_document(template)
        record.validate()
        assert record.provenance.license == "Apache-2.0"
        assert record.quality.validators_passed is True
        assert record.metadata["contains_real_personal_data"] is False
    assert (
        len(
            {
                template.family
                for template in POSITIVE_TEMPLATES
                if any(spec.label == "SECRET" for spec in template.placeholders)
            }
        )
        >= 6
    )

    medical_boundaries = {
        "authored_medical_laboratory_result": ("就诊号:", "]"),
        "authored_medical_discharge_summary": ("住院编号（", "）"),
    }
    for template_id, (left_boundary, right_boundary) in medical_boundaries.items():
        template = templates[template_id]
        left, right = template.body.split("<<MEDICAL_RECORD_NUMBER_1>>")
        assert template.origin_kind == "human_authored"
        assert template.source_candidate_id is None
        assert left.endswith(left_boundary)
        assert right.startswith(right_boundary)
        assert {spec.label for spec in template.placeholders} >= {"MEDICAL_RECORD_NUMBER"}
        record = generator.generate_document(template)
        record.validate()
        assert record.provenance.license == "Apache-2.0"
        assert record.quality.validators_passed is True
        assert record.metadata["contains_real_personal_data"] is False
    assert (
        len(
            {
                template.family
                for template in POSITIVE_TEMPLATES
                if any(spec.label == "MEDICAL_RECORD_NUMBER" for spec in template.placeholders)
            }
        )
        >= 8
    )


def test_new_hard_negative_families_are_non_entity_semantic_contrasts() -> None:
    expected = {
        "hard_negative_build_id": ("BUILD_ID", "软件构建"),
        "hard_negative_config_key_name": ("CONFIG_KEY_NAME", "未保存对应凭据值"),
        "hard_negative_release_date": ("RELEASE_DATE", "系统发布计划"),
        "hard_negative_device_model": ("DEVICE_MODEL", "产品系列"),
    }
    templates = {template.template_id: template for template in HARD_NEGATIVE_TEMPLATES}
    generator = SyntheticGenerator(20260711)
    for template_id, (generator_key, semantic_guard) in expected.items():
        template = templates[template_id]
        assert template.hard_negative is True
        assert template.origin_kind == "human_authored"
        assert {spec.generator_key for spec in template.placeholders} == {generator_key}
        assert all(spec.label is None and spec.risk_tier is None for spec in template.placeholders)
        record = generator.generate_document(template)
        assert record.entities == ()
        assert record.provenance.license == "Apache-2.0"
        assert record.quality.validators_passed is True
        assert record.quality.metadata["validator_results"] == {}
        assert semantic_guard in record.text


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
    records = _build_regenerated_test_corpus(2_000, seed=20260711, hard_negative_ratio=0.30)
    assert Counter(record.split for record in records) == Counter(
        {"train": 1_600, "validation": 200, "test": 200}
    )
    assert not detect_group_leakage(records).has_leakage
    newly_authored_template_ids = {
        "authored_secret_database_password",
        "authored_secret_bearer_header",
        "authored_secret_private_key_fragment",
        "authored_medical_laboratory_result",
        "authored_medical_discharge_summary",
        "hard_negative_build_id",
        "hard_negative_config_key_name",
        "hard_negative_release_date",
        "hard_negative_device_model",
    } | set(TRAIN_ONLY_ADDITION_TEMPLATE_IDS)
    assert newly_authored_template_ids <= {record.provenance.template_id for record in records}
    for template_id in newly_authored_template_ids:
        assert {
            record.split for record in records if record.provenance.template_id == template_id
        } == {"train"}
    observed_template_splits = {record.provenance.template_id: record.split for record in records}
    assert observed_template_splits == TEMPLATE_SPLITS
    for split, expected_count in (("train", 1_600), ("validation", 200), ("test", 200)):
        selected = [record for record in records if record.split == split]
        assert len(selected) == expected_count
        assert sum(bool(record.metadata["hard_negative"]) for record in selected) == round(
            expected_count * 0.30
        )
        assert set(CORE_LABELS) == {
            entity.label for record in selected for entity in record.entities
        }


def test_materialized_corpus_is_deterministic_without_seeded_family_reallocation() -> None:
    left = _build_regenerated_test_corpus(600, seed=20260711, hard_negative_ratio=0.30)
    right = _build_regenerated_test_corpus(600, seed=20260711, hard_negative_ratio=0.30)
    other_seed = _build_regenerated_test_corpus(600, seed=20260712, hard_negative_ratio=0.30)
    assert [dumps_document(record) for record in left] == [
        dumps_document(record) for record in right
    ]
    for records in (left, other_seed):
        assert {
            record.provenance.template_id: record.split for record in records
        } == TEMPLATE_SPLITS
        assert not detect_group_leakage(records).has_leakage
        for split in ("train", "validation", "test"):
            assert set(CORE_LABELS) == {
                entity.label
                for record in records
                if record.split == split
                for entity in record.entities
            }


def test_default_20k_continuation_namespace_has_no_pinned_holdout_value_collision() -> None:
    ratios, split_counts, negative_counts = _generation_plan(
        count=20_000,
        seed=20260711,
        hard_negative_ratio=0.30,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    train_records, audit = _build_continuation_train_partition(
        seed=20260711,
        split_counts=split_counts,
        negative_counts=negative_counts,
        ratios=ratios,
        contract=OFFICIAL_FROZEN_HOLDOUT_CONTRACT,
    )
    assert len(train_records) == 16_000
    assert all(record.split == "train" for record in train_records)
    assert audit == {
        "algorithm_version": "v1.2-parent-sequence-continuation-v1",
        "namespace": "after-parent-train-validation-test",
        "parent_records_preconsumed": 20_000,
        "frozen_jsonl_text_read": False,
        "train_entity_value_groups": 23_975,
        "replayed_holdout_entity_value_groups": 6_832,
        "replayed_holdout_entity_value_groups_sha256": (
            "28d7eda8260a8f908defef2a36d4434500b0c3244aa5d62e9d7daf3988306ae3"
        ),
        "cross_split_collisions": 0,
    }
    train_value_groups = [
        value_group for record in train_records for value_group in record.entity_value_groups
    ]
    assert len(train_value_groups) == len(set(train_value_groups)) == 23_975


def _frozen_parent_fixture(
    tmp_path: Path,
    *,
    count: int = 600,
    seed: int = 20260711,
) -> tuple[Path, FrozenHoldoutContract, dict[str, object], dict[str, bytes]]:
    source = tmp_path / "frozen-v1.2"
    source.mkdir()
    frozen_bytes = {
        "validation": b"opaque frozen validation fixture\n",
        "test": b"opaque frozen test fixture\n",
    }
    file_contracts = []
    ratios, split_counts, negative_counts = _generation_plan(
        count=count,
        seed=seed,
        hard_negative_ratio=0.30,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    for split in ("validation", "test"):
        path = source / f"{split}.jsonl"
        path.write_bytes(frozen_bytes[split])
        path.chmod(0o444)
        file_contracts.append(
            FrozenHoldoutFileContract(
                split=split,
                name=path.name,
                records=split_counts[split],
                bytes=len(frozen_bytes[split]),
                sha256=hashlib.sha256(frozen_bytes[split]).hexdigest(),
            )
        )

    replayed_groups = _replay_parent_holdout_value_groups(
        seed=seed,
        split_counts=split_counts,
        negative_counts=negative_counts,
        ratios=ratios,
    )
    provenance = {
        "source_id": "frozen-parent-fixture",
        "source_kind": "curated_deterministic_synthetic",
        "license": "Apache-2.0",
        "synthetic": True,
    }
    coverage = {
        "labels_by_split": {
            split: {label: 1 for label in CORE_LABELS} for split in ("train", "validation", "test")
        },
        "domains_by_split": {
            split: {"fixture-domain": split_counts[split]}
            for split in ("train", "validation", "test")
        },
        "quality_tiers_by_split": {
            split: {"synthetic_validated": split_counts[split]}
            for split in ("train", "validation", "test")
        },
    }
    manifest: dict[str, object] = {
        "manifest_schema_version": 1,
        "dataset_id": "pii_zh_synthetic_v1",
        "dataset_version": "1.2.0",
        "data_pool": "public_release_pool",
        "public_weight_training_allowed": True,
        "plan": {"name": "fixture", "sha256": "0" * 64},
        "generation": {
            "generator_name": "pii_zh_deterministic_synthetic",
            "generator_version": "2.3.0",
            "materializer_version": "1.2.0",
            "seed": seed,
            "requested_count": count,
            "requested_hard_negative_ratio": 0.30,
            "surface_shortcuts_rejected_casefolded": list(SURFACE_SHORTCUTS),
        },
        "assets": {"fixture": "frozen-v1.2"},
        "provenance": provenance,
        "provenance_snapshot_sha256": _canonical_json_hash(provenance),
        "splitting": {
            "algorithm_version": "stable-pinned-v2",
            "family_assignment_uses_seed": False,
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "template_families_by_split": {"train": 65, "validation": 15, "test": 16},
            "cross_split_group_collisions": 0,
            "entity_value_repetitions": 0,
        },
        "counts": {
            "total": count,
            "positive": count - sum(negative_counts.values()),
            "hard_negative": sum(negative_counts.values()),
            "by_split": split_counts,
            "hard_negative_by_split": negative_counts,
        },
        "coverage": coverage,
        "files": [
            {
                "name": "train.jsonl",
                "split": "train",
                "records": split_counts["train"],
                "bytes": 1,
                "sha256": "0" * 64,
                "mode": "0444",
            },
            *[
                {
                    "name": item.name,
                    "split": item.split,
                    "records": item.records,
                    "bytes": item.bytes,
                    "sha256": item.sha256,
                    "mode": "0444",
                }
                for item in file_contracts
            ],
        ],
    }
    manifest["manifest_sha256"] = _canonical_json_hash(manifest)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    (source / "dataset_manifest.json").write_bytes(manifest_bytes)
    contract = FrozenHoldoutContract(
        parent_dataset_version="1.2.0",
        parent_manifest_sha256=str(manifest["manifest_sha256"]),
        parent_manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        entity_value_group_count=len(replayed_groups),
        entity_value_groups_sha256=_canonical_json_hash(sorted(replayed_groups)),
        files=tuple(file_contracts),
    )
    return source, contract, manifest, frozen_bytes


def test_materializer_writes_read_only_files_and_raw_text_free_manifest(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("immutable test plan\n", encoding="utf-8")
    source, contract, parent_manifest, frozen_bytes = _frozen_parent_fixture(tmp_path)
    output = tmp_path / "synthetic_v1_3"
    manifest = materialize_synthetic_corpus(
        output,
        plan_path=plan,
        holdout_source_dir=source,
        holdout_contract=contract,
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
        assert manifest["manifest_schema_version"] == 2
        assert manifest["dataset_version"] == "1.3.0"
        assert manifest["generation"]["materializer_version"] == "1.3.0"
        assert manifest["generation"]["applies_to_splits"] == ["train"]
        assert manifest["generation"]["value_namespace"] == {
            "algorithm_version": "v1.2-parent-sequence-continuation-v1",
            "parent_dataset_version": "1.2.0",
            "parent_generator_version": "2.3.0",
            "replay_runtime_generator_version": "2.4.0",
            "parent_splits_preconsumed": ["train", "validation", "test"],
            "parent_records_preconsumed": 600,
            "new_train_document_index_start": 600,
            "legacy_value_counters_continue_after_parent_sequence": True,
            "parent_holdout_entity_value_groups_sha256": (contract.entity_value_groups_sha256),
        }
        assert manifest["splitting"]["algorithm_version"] == "stable-pinned-v3"
        assert manifest["splitting"]["family_assignment_uses_seed"] is False
        assert manifest["splitting"]["legacy_template_families_pinned"] == 96
        assert manifest["splitting"]["train_only_addition_families"] == 11
        assert manifest["splitting"]["template_families_by_split"] == {
            "train": 76,
            "validation": 15,
            "test": 16,
        }
        assert manifest["splitting"]["legacy_template_assignment_sha256"] == (
            "ef7f251e3a243d700ac35ea26543b1a3446522936256b3c5ac504be624d2452a"
        )
        assert manifest["splitting"]["train_only_addition_assignment_sha256"] == (
            "7ebcf0e2e363b9a8261abed39cd262ff9e4a8c24929e4a4ce1f3b103c2b65e27"
        )
        assert manifest["splitting"]["template_family_allocation_sha256"] == (
            "ff587ab3cdba3391adf2e35d91845dc15423d99a0f6c6fdfd7713c3893a06159"
        )
        assert manifest["splitting"]["entity_value_isolation"]["cross_split_collisions"] == 0
        assert manifest["frozen_splits"]["copy_mode"] == "byte_for_byte"
        assert manifest["frozen_splits"]["raw_jsonl_records_parsed"] is False
        assert manifest["frozen_splits"]["parent_generation"]["generator_version"] == "2.3.0"
        assert (
            manifest["coverage"]["labels_by_split"]["validation"]
            == parent_manifest["coverage"]["labels_by_split"]["validation"]
        )
        assert (
            manifest["coverage"]["labels_by_split"]["test"]
            == parent_manifest["coverage"]["labels_by_split"]["test"]
        )
        assert (output / "validation.jsonl").read_bytes() == frozen_bytes["validation"]
        assert (output / "test.jsonl").read_bytes() == frozen_bytes["test"]
        assert (
            _canonical_json_hash(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            == manifest["manifest_sha256"]
        )
        assert (
            manifest["assets"]["family_allocation_asset_sha256"] == FAMILY_ALLOCATION_ASSET_SHA256
        )
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


def test_materializer_rejects_mutated_frozen_holdout_and_cleans_staging(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("immutable test plan\n", encoding="utf-8")
    source, contract, _, _ = _frozen_parent_fixture(tmp_path)
    validation = source / "validation.jsonl"
    validation.chmod(0o644)
    validation.write_bytes(validation.read_bytes() + b"tampered")
    output = tmp_path / "synthetic_v1_3"
    with pytest.raises(MaterializationError, match="frozen validation bytes changed"):
        materialize_synthetic_corpus(
            output,
            plan_path=plan,
            holdout_source_dir=source,
            holdout_contract=contract,
            count=600,
            seed=20260711,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".synthetic_v1_3.staging-*"))


def test_materializer_rejects_symlinked_parent_manifest(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("immutable test plan\n", encoding="utf-8")
    source, contract, _, _ = _frozen_parent_fixture(tmp_path)
    manifest = source / "dataset_manifest.json"
    target = source / "manifest-target.json"
    manifest.rename(target)
    manifest.symlink_to(target.name)
    output = tmp_path / "synthetic_v1_3"
    with pytest.raises(MaterializationError, match="could not be opened safely"):
        materialize_synthetic_corpus(
            output,
            plan_path=plan,
            holdout_source_dir=source,
            holdout_contract=contract,
            count=600,
            seed=20260711,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".synthetic_v1_3.staging-*"))
