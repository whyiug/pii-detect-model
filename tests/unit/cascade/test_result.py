from __future__ import annotations

import json

import pytest

from pii_zh.cascade import CASCADE_DETECTION_SCHEMA_VERSION, CascadeDetection


def test_detection_schema_is_stable_and_raw_text_free() -> None:
    detection = CascadeDetection(
        start=4,
        end=15,
        entity_type="PHONE_NUMBER",
        score=0.99,
        source="rule:cn_common",
        sources=("qwen", "rule:cn_common"),
        decision_process=("candidate:rule", "fusion:exact_same_type_collapsed"),
    )

    payload = detection.to_dict()

    assert payload == {
        "schema_version": CASCADE_DETECTION_SCHEMA_VERSION,
        "start": 4,
        "end": 15,
        "entity_type": "PHONE_NUMBER",
        "score": 0.99,
        "source": "rule:cn_common",
        "sources": ["qwen", "rule:cn_common"],
        "decision_process": ["candidate:rule", "fusion:exact_same_type_collapsed"],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "text" not in payload
    assert "value" not in payload
    assert "19912345678" not in serialized


def test_detection_rejects_ambiguous_sources_and_offsets() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        CascadeDetection(0, 1, "PERSON", 0.9, "qwen", ("qwen", "qwen"), ("selected",))
    with pytest.raises(ValueError, match="winning source"):
        CascadeDetection(0, 1, "PERSON", 0.9, "qwen", ("other",), ("selected",))
    with pytest.raises(ValueError, match="offsets"):
        CascadeDetection(1, 1, "PERSON", 0.9, "qwen", ("qwen",), ("selected",))
