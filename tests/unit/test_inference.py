from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from pii_zh.inference import (
    InferenceSafetyError,
    TransformersSpanPredictor,
    load_local_predictor,
)


def _sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _tiny_macbert_v5b_artifact(path, *, selected_evaluation_point=1.5) -> None:
    path.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "甲", "乙"]
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(path)
    # Transformers 5 stores these values in tokenizer_config.json and no
    # longer emits the legacy file that this historical artifact contract binds.
    (path / "special_tokens_map.json").write_text(
        json.dumps(tokenizer.special_tokens_map, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
        id2label={0: "O", 1: "B-PERSON_NAME", 2: "I-PERSON_NAME"},
        label2id={"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2},
    )
    config.pii_attention_mode = "checkpoint_native_bidirectional"
    BertForTokenClassification(config).save_pretrained(path, safe_serialization=True)
    output_names = {
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
    manifest = {
        "schema_version": 1,
        "manifest_type": "experimental_macbert_numeric_o_v5b_training",
        "status": "completed",
        "output_files": {name: _sha256_file(path / name) for name in sorted(output_names)},
        "selection": {
            "gate_passed": True,
            "selected_evaluation_point": selected_evaluation_point,
        },
        "pilot_preregistration": {
            "status": "verified_development_only_loss_causal_pilot",
            "numeric_O_token_weight": 4.0,
        },
        "release_eligible": False,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    (path / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


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


class _CompetingBioModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            id2label={
                0: "O",
                1: "B-PERSON_NAME",
                2: "I-PERSON_NAME",
                3: "B-SECRET",
                4: "I-SECRET",
            }
        )

    def forward(self, input_ids, attention_mask):
        del attention_mask
        logits = torch.full((*input_ids.shape, 5), -5.0, device=input_ids.device)
        logits[..., 0] = 5.0
        logits[:, 1, 0] = -5.0
        logits[:, 1, 1] = 4.0
        logits[:, 1, 3] = 5.0
        logits[:, 2, 0] = -5.0
        logits[:, 2, 2] = 4.0
        logits[:, 2, 4] = 5.0
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


def test_predictor_masks_non_protocol_logits_before_argmax() -> None:
    unrestricted = TransformersSpanPredictor(
        _CompetingBioModel(), _CharacterBatchTokenizer(), micro_batch_size=1
    )
    restricted = TransformersSpanPredictor(
        _CompetingBioModel(),
        _CharacterBatchTokenizer(),
        micro_batch_size=1,
        allowed_labels=("PERSON_NAME",),
    )

    assert unrestricted.predict("甲乙")[0]["entity_type"] == "SECRET"
    assert restricted.predict("甲乙") == [
        {
            "entity_type": "PERSON_NAME",
            "start": 0,
            "end": 2,
            "score": restricted.predict("甲乙")[0]["score"],
            "offset_mode": "relative",
            "metadata": {"boundary_refined": False},
        }
    ]


def test_load_local_predictor_accepts_bound_macbert_v5b_and_rejects_tamper(
    tmp_path,
) -> None:
    artifact = tmp_path / "macbert-v5b"
    _tiny_macbert_v5b_artifact(artifact)

    predictor = load_local_predictor(artifact, device="cpu", micro_batch_size=1)

    assert predictor.model.config.model_type == "bert"
    assert predictor.attention_mode == "full"
    assert len(predictor.predict_batch(["甲乙"])) == 1
    with (artifact / "vocab.txt").open("a", encoding="utf-8") as stream:
        stream.write("丙\n")
    with pytest.raises(InferenceSafetyError, match="files do not match"):
        load_local_predictor(artifact, device="cpu")


def test_load_local_predictor_rejects_boolean_macbert_selection_point(tmp_path) -> None:
    artifact = tmp_path / "macbert-v5b"
    _tiny_macbert_v5b_artifact(artifact, selected_evaluation_point=True)

    with pytest.raises(InferenceSafetyError, match="safety contract"):
        load_local_predictor(artifact, device="cpu")
