from __future__ import annotations

import json

import pytest

from pii_zh.data.schema import (
    DataPool,
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    SchemaError,
    dumps_document,
    loads_document,
    route_data_pool,
    stable_text_hash,
)


def make_record() -> DocumentRecord:
    text = "测试用户甲"
    entity = EntitySpan(0, len(text), "PERSON_NAME", text, "gold", "T1")
    provenance = Provenance(
        source_id="unit-test",
        source_kind="fixture",
        source_revision="1",
        license="Apache-2.0",
        synthetic=False,
    )
    quality = QualityMetadata(QualityTier.GOLD, True, True, "unit_test", 1.0)
    return DocumentRecord.create(
        doc_id="fixture-1",
        text=text,
        entities=(entity,),
        language="zh-Hans-CN",
        domain="test",
        scene="unit",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        entity_value_groups=(stable_text_hash(text),),
    )


def test_schema_round_trip_is_strict_and_lossless() -> None:
    original = make_record()
    restored = loads_document(dumps_document(original))
    assert restored == original
    payload = json.loads(dumps_document(original))
    payload["unexpected"] = 1
    with pytest.raises(SchemaError, match="unknown document fields"):
        DocumentRecord.from_dict(payload)


def test_entity_text_must_match_left_closed_right_open_slice() -> None:
    record = make_record()
    bad = EntitySpan(0, 2, "PERSON_NAME", "错误", "gold", "T1")
    with pytest.raises(SchemaError, match="entity text mismatch"):
        DocumentRecord.create(
            doc_id="bad",
            text=record.text,
            entities=(bad,),
            language=record.language,
            domain=record.domain,
            scene=record.scene,
            provenance=record.provenance,
            quality=record.quality,
            public_weight_training_allowed=True,
        )


def test_four_pool_routing_fails_closed_on_unknown_permission() -> None:
    assert (
        route_data_pool(quality_gate=True, public_weight_training_allowed=True)
        is DataPool.PUBLIC_RELEASE
    )
    assert (
        route_data_pool(
            quality_gate=True,
            public_weight_training_allowed=False,
            private_enterprise=True,
        )
        is DataPool.PRIVATE_ENTERPRISE
    )
    assert (
        route_data_pool(
            quality_gate=True,
            public_weight_training_allowed=False,
            evaluation_only=True,
        )
        is DataPool.EVALUATION_ONLY
    )
    assert (
        route_data_pool(quality_gate=True, public_weight_training_allowed=None)
        is DataPool.QUARANTINED
    )
    assert (
        route_data_pool(quality_gate=False, public_weight_training_allowed=True)
        is DataPool.QUARANTINED
    )
