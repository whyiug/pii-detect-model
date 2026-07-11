from __future__ import annotations

from types import SimpleNamespace

import torch

from pii_zh.inference import TransformersSpanPredictor


class _CharacterBatchTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        width = max(len(text) for text in texts) + 2
        input_ids = []
        attention = []
        offsets = []
        for text in texts:
            ids = [1, *([7] * len(text)), 2]
            mask = [1] * len(ids)
            row_offsets = [(0, 0), *[(i, i + 1) for i in range(len(text))], (0, 0)]
            padding = width - len(ids)
            input_ids.append(ids + [0] * padding)
            attention.append(mask + [0] * padding)
            offsets.append(row_offsets + [(0, 0)] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "offset_mapping": torch.tensor(offsets),
        }


class _FixedBioModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(id2label={0: "O", 1: "B-PERSON_NAME", 2: "I-PERSON_NAME"})

    def forward(self, input_ids, attention_mask):
        del attention_mask
        logits = torch.full((*input_ids.shape, 3), -4.0, device=input_ids.device)
        logits[..., 0] = 4.0
        logits[:, 1, 0] = -4.0
        logits[:, 1, 1] = 4.0
        logits[:, 2, 0] = -4.0
        logits[:, 2, 2] = 4.0
        return SimpleNamespace(logits=logits + self.anchor)


def test_predictor_emits_character_spans_without_raw_text_and_has_batch_parity() -> None:
    predictor = TransformersSpanPredictor(
        _FixedBioModel(), _CharacterBatchTokenizer(), micro_batch_size=2
    )
    batch = predictor.predict_batch(["甲乙到访", "丙丁登记"])
    assert batch[0] == predictor.predict("甲乙到访")
    assert batch[1] == predictor.predict("丙丁登记")
    for result in batch:
        assert len(result) == 1
        assert result[0]["entity_type"] == "PERSON_NAME"
        assert (result[0]["start"], result[0]["end"]) == (0, 2)
        assert "text" not in result[0]
        assert 0.0 <= result[0]["score"] <= 1.0
    assert predictor.predict_batch(["", "甲乙到访"])[0] == []
