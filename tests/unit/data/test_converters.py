from __future__ import annotations

import pytest

from pii_zh.data.converters._common import ConversionError
from pii_zh.data.converters.openpii import convert_openpii_record, iter_openpii_records
from pii_zh.data.converters.pii_bench_zh import convert_pii_bench_zh_record
from pii_zh.data.schema import DataPool


def span(text: str, value: str, label: str) -> dict[str, object]:
    start = text.index(value)
    return {"start": start, "end": start + len(value), "text": value, "label": label}


def test_openpii_conversion_defaults_to_quarantine_pending_audit() -> None:
    text = "张三的出生日期是2000-01-02，电话19912\x3345678。"
    row = {
        "id": "open-1",
        "language": "zh",
        "region": "CN",
        "text": text,
        "entities": [
            span(text, "张三", "NAME"),
            span(text, "2000-01-02", "DATE"),
            span(text, "19912\x3345678", "TELEPHONENUM"),
        ],
    }
    record = convert_openpii_record(row, source_revision="fixture-rev")
    assert record.data_pool is DataPool.QUARANTINED
    assert record.public_weight_training_allowed is None
    assert record.quality.review_status == "pending_manual_audit"
    assert [entity.label for entity in record.entities] == [
        "PERSON_NAME",
        "DATE_OF_BIRTH",
        "PHONE_NUMBER",
    ]
    record.validate()


def test_openpii_iterator_filters_non_zh_cn_and_can_open_public_gate() -> None:
    text = "邮箱pii-test@example.invalid"
    zh = {
        "language": "zh-CN",
        "text": text,
        "privacy_mask": [span(text, "pii-test@example.invalid", "EMAIL")],
    }
    en = {"language": "en", "region": "US", "text": "nothing", "entities": []}
    records = list(
        iter_openpii_records(
            [zh, en],
            source_revision="fixture-rev",
            quality_gate=True,
            public_weight_training_allowed=True,
        )
    )
    assert len(records) == 1
    assert records[0].data_pool is DataPool.PUBLIC_RELEASE


def test_openpii_maps_released_source_labels_and_drops_invalid_generic_id() -> None:
    text = "驾驶证TEST-DL-AB12，社保号SYN-SS-1234567890，无效证件123456。"
    row = {
        "language": "zh",
        "region": "CN",
        "text": text,
        "privacy_mask": [
            span(text, "TEST-DL-AB12", "DRIVERLICENSENUM"),
            span(text, "SYN-SS-1234567890", "SOCIALNUM"),
            span(text, "123456", "IDCARDNUM"),
        ],
    }
    record = convert_openpii_record(row, source_revision="fixture-rev")
    assert [entity.label for entity in record.entities] == [
        "DRIVER_LICENSE_NUMBER",
        "SOCIAL_SECURITY_NUMBER",
    ]
    assert record.quality.metadata["dropped_unknown_labels"] == 1


def test_pii_bench_is_irreversibly_evaluation_only() -> None:
    text = "测试用户甲的电话是19912\x3345678"
    row = {
        "id": "bench-1",
        "text": text,
        "subset": "chat",
        "entities": [
            span(text, "测试用户甲", "PERSON"),
            span(text, "19912\x3345678", "PHONE"),
        ],
    }
    record = convert_pii_bench_zh_record(row, source_revision="fixture-rev")
    assert record.data_pool is DataPool.EVALUATION_ONLY
    assert record.provenance.evaluation_only is True
    assert record.public_weight_training_allowed is False
    assert record.metadata["never_train"] is True
    assert record.split == "test"


def test_pii_bench_unknown_labels_fail_closed() -> None:
    text = "未知内容"
    row = {"text": text, "entities": [span(text, "未知", "NOT_A_LABEL")]}
    with pytest.raises(ConversionError, match="unknown pii-bench-zh label"):
        convert_pii_bench_zh_record(row, source_revision="fixture-rev")


def test_pii_bench_maps_all_released_benchmark_label_spellings() -> None:
    text = "证件123，银行卡456，护照E00000000，车牌京A00000"
    row = {
        "text": text,
        "subset": "formal",
        "entities": [
            span(text, "123", "id_number"),
            span(text, "456", "bank_card"),
            span(text, "E00000000", "passport"),
            span(text, "京A00000", "license_plate"),
        ],
    }
    record = convert_pii_bench_zh_record(row, source_revision="fixture-rev")
    assert [entity.label for entity in record.entities] == [
        "CN_RESIDENT_ID",
        "BANK_CARD_NUMBER",
        "PASSPORT_NUMBER",
        "VEHICLE_LICENSE_PLATE",
    ]
