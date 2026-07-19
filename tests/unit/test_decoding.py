from __future__ import annotations

import pytest
import torch

from pii_zh.models.decoding import decode_bio_ids, decode_bio_spans, repair_bio_tags
from pii_zh.models.qwen3_jpt import build_jpt_inputs


def test_repair_bio_tags_fixes_start_and_type_transitions() -> None:
    tags = ["I-PERSON", "I-PERSON", "I-PHONE", "O", "I-ADDRESS"]
    assert repair_bio_tags(tags) == [
        "B-PERSON",
        "I-PERSON",
        "B-PHONE",
        "O",
        "B-ADDRESS",
    ]


def test_decode_bio_spans_repairs_and_uses_character_offsets() -> None:
    text = "张三电话13800\x3138000"
    tags = ["O", "I-PERSON", "I-PERSON", "O", "B-PHONE", "I-PHONE"]
    offsets = [(0, 0), (0, 1), (1, 2), (2, 4), (4, 9), (9, 15)]
    spans = decode_bio_spans(
        tags,
        offsets,
        text=text,
        token_scores=[1.0, 0.98, 0.95, 1.0, 0.92, 0.88],
    )
    assert [(span.entity_type, span.start, span.end) for span in spans] == [
        ("PERSON", 0, 2),
        ("PHONE", 4, 15),
    ]
    assert spans[0].text == "张三"
    assert spans[0].score == pytest.approx(0.95)
    assert spans[1].text == "13800\x3138000"
    assert spans[1].score == pytest.approx(0.88)


def test_decode_ids_rejects_unknown_label_id() -> None:
    with pytest.raises(ValueError, match="absent from id2label"):
        decode_bio_ids([3], [(0, 1)], {0: "O"})


def test_malformed_bio_label_is_not_silently_guessed() -> None:
    with pytest.raises(ValueError, match="Invalid BIO tag"):
        repair_bio_tags(["PERSON"])


def test_jpt_builds_repeated_input_and_masks_first_copy_labels() -> None:
    input_ids = torch.tensor([[0, 10, 11, 12], [20, 21, 0, 0]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    labels = torch.tensor([[-100, 1, 2, 0], [3, 4, -100, -100]])
    batch = build_jpt_inputs(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sep_token_id=63,
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 0, 63, 10, 11, 12, 0],
        [20, 21, 0, 0, 63, 20, 21, 0, 0],
    ]
    assert batch["attention_mask"].tolist() == [
        [True, True, True, False, True, True, True, True, False],
        [True, True, False, False, True, True, True, False, False],
    ]
    assert batch["second_copy_mask"].tolist() == [
        [False, False, False, False, False, True, True, True, False],
        [False, False, False, False, False, True, True, False, False],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, -100, -100, -100, 1, 2, 0, -100],
        [-100, -100, -100, -100, -100, 3, 4, -100, -100],
    ]


def test_jpt_refuses_entity_unsafe_truncation() -> None:
    with pytest.raises(ValueError, match="pre-chunk"):
        build_jpt_inputs(
            torch.tensor([[1, 2, 3]]),
            sep_token_id=9,
            pad_token_id=0,
            max_length=6,
        )
