from __future__ import annotations

import pytest
import torch

from pii_zh.inference.bie_decoding import (
    BieDecodingError,
    constrained_bie_viterbi,
    decode_bie_logits,
    validate_bie_sequence,
)
from pii_zh.inference.project_bie import closed8_project_bie_label_ids
from pii_zh.models.aiguard24_bie import build_core_bie_label_maps


def _person_scope() -> tuple[list[int], dict[int, str]]:
    label2id, id2label = build_core_bie_label_maps()
    return [
        label2id["O"],
        label2id["B-PERSON_NAME"],
        label2id["I-PERSON_NAME"],
        label2id["E-PERSON_NAME"],
    ], id2label


def test_constrained_viterbi_allows_singleton_b_and_complete_bie() -> None:
    allowed, id2label = _person_scope()
    singleton = constrained_bie_viterbi(
        torch.tensor([[0.0, 8.0, 9.0, -4.0], [7.0, -3.0, 1.0, 0.0]]),
        allowed_label_ids=allowed,
        id2label=id2label,
    )
    complete = constrained_bie_viterbi(
        torch.tensor(
            [
                [0.0, 10.0, 11.0, -4.0],
                [0.0, -4.0, 10.0, 9.0],
                [0.0, -4.0, 9.0, 10.0],
            ]
        ),
        allowed_label_ids=allowed,
        id2label=id2label,
    )

    assert [id2label[int(value)] for value in singleton] == ["B-PERSON_NAME", "O"]
    assert [id2label[int(value)] for value in complete] == [
        "B-PERSON_NAME",
        "I-PERSON_NAME",
        "E-PERSON_NAME",
    ]
    validate_bie_sequence(singleton.tolist(), id2label=id2label)
    validate_bie_sequence(complete.tolist(), id2label=id2label)


def test_constrained_viterbi_prevents_orphan_and_mismatched_states() -> None:
    label2id, id2label = build_core_bie_label_maps()
    allowed = [
        label2id["O"],
        label2id["B-PERSON_NAME"],
        label2id["I-PERSON_NAME"],
        label2id["E-PERSON_NAME"],
        label2id["B-PHONE_NUMBER"],
        label2id["I-PHONE_NUMBER"],
        label2id["E-PHONE_NUMBER"],
    ]
    logits = torch.tensor(
        [
            [0.0, 8.0, 10.0, -4.0, 0.0, -4.0, -4.0],
            [0.0, -4.0, -4.0, -4.0, -4.0, -4.0, 10.0],
        ]
    )
    decoded = constrained_bie_viterbi(
        logits,
        allowed_label_ids=allowed,
        id2label=id2label,
    )

    validate_bie_sequence(decoded.tolist(), id2label=id2label)

    invalid_sequences = (
        ([label2id["I-PERSON_NAME"]], "orphan or mismatched I"),
        ([label2id["E-PERSON_NAME"]], "orphan or mismatched E"),
        (
            [label2id["B-PERSON_NAME"], label2id["I-PHONE_NUMBER"]],
            "orphan or mismatched I",
        ),
        (
            [label2id["B-PERSON_NAME"], label2id["E-PHONE_NUMBER"]],
            "orphan or mismatched E",
        ),
        (
            [
                label2id["B-PERSON_NAME"],
                label2id["I-PERSON_NAME"],
                label2id["O"],
            ],
            "ended without E",
        ),
    )
    for invalid, message in invalid_sequences:
        with pytest.raises(BieDecodingError, match=message):
            validate_bie_sequence(invalid, id2label=id2label)


@pytest.mark.parametrize("decoder_id", ["greedy", "constrained_viterbi"])
def test_decoder_masks_to_exact_closed8_allowed_ids(decoder_id: str) -> None:
    _, id2label = build_core_bie_label_maps()
    allowed = closed8_project_bie_label_ids(id2label)
    torch.manual_seed(19)
    logits = torch.randn(2, 7, 73)
    logits[..., 70] = 100.0
    valid = torch.tensor([[0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    label_ids, scores = decode_bie_logits(
        logits,
        allowed_label_ids=allowed,
        id2label=id2label,
        decoder_id=decoder_id,
        valid_token_mask=valid,
    )

    assert set(label_ids.flatten().tolist()) <= set(allowed)
    assert torch.equal(label_ids[~valid], torch.zeros_like(label_ids[~valid]))
    assert scores.shape == label_ids.shape
    assert torch.isfinite(scores).all()
    if decoder_id == "constrained_viterbi":
        for row in label_ids.tolist():
            validate_bie_sequence(row, id2label=id2label)


def test_decoder_id_is_required_and_has_no_fallback() -> None:
    allowed, id2label = _person_scope()
    with pytest.raises(BieDecodingError, match="decoder_id"):
        decode_bie_logits(
            torch.zeros(2, 73),
            allowed_label_ids=allowed,
            id2label=id2label,
            decoder_id="unknown",
        )
