from __future__ import annotations

from dataclasses import replace

from pii_zh.data.synthetic.sota_v1 import SyntheticSotaGenerator
from pii_zh.data.synthetic.sota_v2_resplit import partition_validation_records
from pii_zh.data.synthetic.templates import HARD_NEGATIVE_TEMPLATES, POSITIVE_TEMPLATES


def _records():  # type: ignore[no-untyped-def]
    generator = SyntheticSotaGenerator(8181)
    records = generator.generate_partition(
        positive_count=12,
        hard_negative_count=8,
        positive_templates=POSITIVE_TEMPLATES[:3],
        hard_negative_templates=HARD_NEGATIVE_TEMPLATES[:3],
    )
    return [
        replace(record, split="validation", template_group=f"group-{index % 3}")
        for index, record in enumerate(records)
    ]


def test_partition_is_deterministic_disjoint_and_retains_each_stratum() -> None:
    records = _records()
    first_train, first_validation = partition_validation_records(
        records,
        train_ratio=0.8,
        salt="fixed-test-salt",
    )
    second_train, second_validation = partition_validation_records(
        list(reversed(records)),
        train_ratio=0.8,
        salt="fixed-test-salt",
    )

    assert [record.doc_id for record in first_train] == [
        record.doc_id for record in second_train
    ]
    assert [record.doc_id for record in first_validation] == [
        record.doc_id for record in second_validation
    ]
    assert {record.doc_id for record in first_train}.isdisjoint(
        record.doc_id for record in first_validation
    )
    assert {record.template_group for record in first_validation} == {
        record.template_group for record in records
    }
