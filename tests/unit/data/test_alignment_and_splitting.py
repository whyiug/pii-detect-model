from __future__ import annotations

from dataclasses import replace

import pytest

from pii_zh.data.alignment import IGNORE_LABEL, AlignmentError, align_document_to_bio
from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    stable_text_hash,
)
from pii_zh.data.splitting import detect_group_leakage, group_split


def make_record(
    doc_id: str,
    text: str,
    entity: EntitySpan | None = None,
    *,
    template_group: str | None = None,
    value_group: str | None = None,
) -> DocumentRecord:
    provenance = Provenance("fixture", "unit", "1", "Apache-2.0", False)
    quality = QualityMetadata(QualityTier.GOLD, True, True, "unit_test", 1.0)
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=() if entity is None else (entity,),
        language="zh-Hans-CN",
        domain="test",
        scene="unit",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        template_group=template_group,
        entity_value_groups=() if value_group is None else (value_group,),
    )


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs):
        offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))] + [(0, 0)]
        return {
            "input_ids": list(range(len(offsets))),
            "attention_mask": [1] * len(offsets),
            "offset_mapping": offsets,
            "special_tokens_mask": [1] + [0] * len(text) + [1],
        }


class PairTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs):
        assert len(text) == 2
        return {"input_ids": [1], "attention_mask": [1], "offset_mapping": [(0, 2)]}


class TruncatingTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs):
        return {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "offset_mapping": [(0, 1), (1, 2)],
            "special_tokens_mask": [0, 0],
        }


def test_character_span_to_bio_and_label_ids() -> None:
    text = "联系测试用户甲"
    start = text.index("测试")
    entity = EntitySpan(start, len(text), "PERSON_NAME", text[start:], "gold", "T1")
    aligned = align_document_to_bio(make_record("a", text, entity), CharacterTokenizer())
    assert aligned.bio_labels[0] == IGNORE_LABEL
    assert aligned.bio_labels[start + 1] == "B-PERSON_NAME"
    assert aligned.bio_labels[start + 2] == "I-PERSON_NAME"
    assert aligned.statistics.unalignable_boundary_rate == 0.0
    ids = aligned.label_ids({"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2})
    assert ids[start + 1] == 1


def test_unalignable_boundary_is_reported_without_rewriting_gold_span() -> None:
    entity = EntitySpan(1, 2, "PERSON_NAME", "试", "gold", "T1")
    aligned = align_document_to_bio(make_record("b", "测试", entity), PairTokenizer())
    assert aligned.statistics.unalignable_boundary_rate == 1.0
    issue = aligned.unalignable_boundaries[0]
    assert issue.start_aligned is False
    assert issue.end_aligned is True
    assert aligned.bio_labels == ("B-PERSON_NAME",)


def test_slow_tokenizer_is_rejected() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.is_fast = False
    with pytest.raises(AlignmentError, match="fast tokenizer"):
        align_document_to_bio(make_record("c", "无实体"), tokenizer)


def test_partially_truncated_entity_fails_closed() -> None:
    entity = EntitySpan(1, 3, "PERSON_NAME", "试值", "gold", "T1")
    with pytest.raises(AlignmentError, match="partially covered"):
        align_document_to_bio(make_record("truncated", "测试值", entity), TruncatingTokenizer())


def test_group_split_keeps_template_and_value_components_together() -> None:
    shared_value = stable_text_hash("synthetic-shared")
    records = [
        make_record("d1", "一", template_group="template-a"),
        make_record("d2", "二", template_group="template-a"),
        make_record("d3", "三", value_group=shared_value),
        make_record("d4", "四", value_group=shared_value),
        make_record("d5", "五", template_group="template-b"),
        make_record("d6", "六", template_group="template-c"),
    ]
    split_records = group_split(records, seed=17)
    by_id = {record.doc_id: record.split for record in split_records}
    assert by_id["d1"] == by_id["d2"]
    assert by_id["d3"] == by_id["d4"]
    assert not detect_group_leakage(split_records).has_leakage

    leaked = [replace(split_records[0], split="train"), replace(split_records[1], split="test")]
    report = detect_group_leakage(leaked)
    assert report.has_leakage
    assert report.collisions[0].group_type == "template"
