from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from pii_zh.taxonomy import (
    TaxonomyValidationError,
    load_presidio_mapping,
    load_taxonomy,
    validate_presidio_mapping_document,
    validate_taxonomy_document,
)

EXPECTED_CORE_LABELS = {
    "PERSON_NAME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "VEHICLE_LICENSE_PLATE",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER",
    "WECHAT_ID",
    "QQ_NUMBER",
    "ALIPAY_ACCOUNT",
    "USERNAME",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "DEVICE_ID",
    "GEO_COORDINATE",
    "SECRET",
}


def _yaml_document(filename: str) -> dict[str, Any]:
    path = Path(__file__).parents[2] / "src" / "pii_zh" / "taxonomy" / filename
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_packaged_taxonomy_has_exact_v1_core_contract() -> None:
    taxonomy = load_taxonomy()

    assert taxonomy.schema_version == 1
    assert taxonomy.taxonomy_version == "1.0.0"
    assert taxonomy.language == "zh-Hans-CN"
    assert taxonomy.tagging_scheme == "BIO"
    assert taxonomy.outside_label == "O"
    assert taxonomy.core_label_names == EXPECTED_CORE_LABELS
    assert len(taxonomy.core_label_names) == 24
    assert set(taxonomy.risk_tiers) == {"T0", "T1", "T2"}


def test_label_sets_have_safe_default_behavior() -> None:
    taxonomy = load_taxonomy()

    assert all(
        entity.enabled_by_default and entity.output for entity in taxonomy.label_sets["core"]
    )
    assert all(
        not entity.enabled_by_default and entity.output
        for entity in taxonomy.label_sets["extended"]
    )
    assert all(
        not entity.enabled_by_default and not entity.output and not entity.risk_tiers
        for entity in taxonomy.label_sets["auxiliary"]
    )


def test_flattening_priority_covers_every_label_once() -> None:
    taxonomy = load_taxonomy()
    prioritized = [
        label
        for _priority_name, priority_labels in taxonomy.flattening_priority
        for label in priority_labels
    ]

    assert len(prioritized) == len(set(prioritized))
    assert set(prioritized) == taxonomy.label_names


def test_packaged_presidio_mapping_covers_outputs_and_ignores_auxiliary() -> None:
    taxonomy = load_taxonomy()
    mapping = load_presidio_mapping(taxonomy=taxonomy)

    assert mapping.taxonomy_version == taxonomy.taxonomy_version
    assert set(mapping.model_to_presidio) == taxonomy.output_label_names
    assert mapping.ignored_labels == taxonomy.auxiliary_label_names
    assert mapping.model_to_presidio["PERSON_NAME"] == "PERSON"
    assert mapping.model_to_presidio["ADDRESS"] == "CN_ADDRESS"
    assert mapping.model_to_presidio["CN_RESIDENT_ID"] == "CN_ID_CARD"


def test_duplicate_label_is_rejected() -> None:
    document = _yaml_document("taxonomy.yaml")
    duplicate = deepcopy(document["label_sets"]["core"][0])
    document["label_sets"]["extended"].append(duplicate)

    with pytest.raises(TaxonomyValidationError, match="duplicate label"):
        validate_taxonomy_document(document)


def test_extended_label_cannot_be_enabled_without_contract_change() -> None:
    document = _yaml_document("taxonomy.yaml")
    document["label_sets"]["extended"][0]["enabled_by_default"] = True

    with pytest.raises(TaxonomyValidationError, match="extended label"):
        validate_taxonomy_document(document)


def test_unknown_risk_tier_is_rejected() -> None:
    document = _yaml_document("taxonomy.yaml")
    document["label_sets"]["core"][0]["risk_tiers"] = ["T_UNKNOWN"]

    with pytest.raises(TaxonomyValidationError, match="unknown values"):
        validate_taxonomy_document(document)


def test_incomplete_presidio_mapping_is_rejected() -> None:
    taxonomy = load_taxonomy()
    document = _yaml_document("presidio_mapping.yaml")
    del document["model_to_presidio"]["PHONE_NUMBER"]

    with pytest.raises(TaxonomyValidationError, match="cover output labels exactly"):
        validate_presidio_mapping_document(document, taxonomy)


def test_auxiliary_label_cannot_leak_into_presidio_outputs() -> None:
    taxonomy = load_taxonomy()
    document = _yaml_document("presidio_mapping.yaml")
    document["model_to_presidio"]["ORDER_ID"] = "ORDER_ID"

    with pytest.raises(TaxonomyValidationError, match="cover output labels exactly"):
        validate_presidio_mapping_document(document, taxonomy)


def test_mapping_version_must_match_taxonomy() -> None:
    taxonomy = load_taxonomy()
    document = _yaml_document("presidio_mapping.yaml")
    document["taxonomy_version"] = "99.0.0"

    with pytest.raises(TaxonomyValidationError, match="does not match"):
        validate_presidio_mapping_document(document, taxonomy)


def test_loader_accepts_an_explicit_taxonomy_path(tmp_path: Path) -> None:
    source_document = _yaml_document("taxonomy.yaml")
    copied_path = tmp_path / "taxonomy-copy.yaml"
    copied_path.write_text(
        yaml.safe_dump(source_document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    taxonomy = load_taxonomy(copied_path)

    assert taxonomy.core_label_names == EXPECTED_CORE_LABELS
