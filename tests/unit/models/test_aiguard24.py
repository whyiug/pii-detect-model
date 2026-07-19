from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForTokenClassification

from pii_zh.models import aiguard24
from pii_zh.models.aiguard24 import (
    AIGUARD24_TARGET_TO_SOURCE,
    AIGUARD_SOURCE_ENTITY_TYPES,
    Aiguard24InitializationError,
    initialize_aiguard24,
)
from pii_zh.taxonomy import load_taxonomy
from pii_zh.training.data import build_bio_label_maps


def _source_labels() -> list[str]:
    return [
        "O",
        *(f"B-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
        *(f"I-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
        *(f"E-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(path: Path) -> dict[str, str]:
    return {
        name: _sha256(path / name)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }


def _make_source(path: Path) -> Qwen3ForTokenClassification:
    labels = _source_labels()
    id2label = dict(enumerate(labels))
    label2id = {label: label_id for label_id, label in id2label.items()}
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        architectures=["Qwen3ForTokenClassification"],
        classifier_dropout=0.0,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    torch.manual_seed(17)
    model = Qwen3ForTokenClassification(config)
    with torch.no_grad():
        values = torch.arange(len(labels) * config.hidden_size, dtype=torch.float32)
        model.score.weight.copy_(values.reshape(len(labels), config.hidden_size))
        model.score.bias.copy_(torch.arange(len(labels), dtype=torch.float32) + 0.25)
    model.save_pretrained(path, safe_serialization=True)
    # The initializer pins the whole upstream model package, even though it
    # only consumes config and weights.  Minimal fixtures keep that contract.
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return model


def _patch_fixture_hashes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aiguard24, "_EXPECTED_SOURCE_HASHES", _source_hashes(path))


def test_initializes_standard_core24_model_and_projects_head_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    source = _make_source(source_path)
    _patch_fixture_hashes(source_path, monkeypatch)
    result = initialize_aiguard24(source_path, seed=73)
    target = result.model

    assert type(target) is Qwen3ForTokenClassification
    label2id, id2label = build_bio_label_maps(load_taxonomy())
    assert target.config.label2id == label2id
    assert target.config.id2label == id2label
    assert target.config.num_labels == 49
    assert target.config.architectures == ["Qwen3ForTokenClassification"]
    assert target.config.pii_attention_mode == "causal"
    assert target.config.pii_release_eligible is False
    assert target.config.pii_training_status == "initialized_untrained"
    assert target.config._name_or_path == ""

    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[key], value, rtol=0.0, atol=0.0)

    source_label2id = source.config.label2id
    torch.testing.assert_close(
        target.score.weight[label2id["O"]],
        source.score.weight[source_label2id["O"]],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        target.score.weight[label2id["B-PERSON_NAME"]],
        source.score.weight[source_label2id["B-name"]],
        rtol=0.0,
        atol=0.0,
    )
    expected_person_i = torch.stack(
        [
            source.score.weight[source_label2id["I-name"]],
            source.score.weight[source_label2id["E-name"]],
        ]
    ).mean(dim=0)
    torch.testing.assert_close(
        target.score.weight[label2id["I-PERSON_NAME"]],
        expected_person_i,
        rtol=0.0,
        atol=0.0,
    )
    expected_bank_b = torch.stack(
        [
            source.score.weight[source_label2id["B-bank_card"]],
            source.score.weight[source_label2id["B-credit_card"]],
        ]
    ).mean(dim=0)
    expected_bank_i = torch.stack(
        [
            source.score.weight[source_label2id[tag]]
            for tag in (
                "I-bank_card",
                "E-bank_card",
                "I-credit_card",
                "E-credit_card",
            )
        ]
    ).mean(dim=0)
    torch.testing.assert_close(
        target.score.weight[label2id["B-BANK_CARD_NUMBER"]],
        expected_bank_b,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        target.score.weight[label2id["I-BANK_CARD_NUMBER"]],
        expected_bank_i,
        rtol=0.0,
        atol=0.0,
    )

    audit = result.audit
    assert audit.mapped_target_labels == tuple(
        entity.name
        for entity in load_taxonomy().label_sets["core"]
        if entity.name in AIGUARD24_TARGET_TO_SOURCE
    )
    assert len(audit.mapped_target_labels) == 12
    assert len(audit.unmapped_target_labels) == 12
    assert audit.backbone_missing_keys == ()
    assert audit.backbone_unexpected_keys == ()
    assert audit.outside_row_copied is True
    assert audit.mapped_head_rows_verified is True
    assert audit.unmapped_head_rows_preserved is True
    assert audit.release_eligible is False
    assert "source_path" not in json.dumps(audit.to_dict(), sort_keys=True)
    assert target.config.pii_lineage == audit.to_dict()


def test_unmapped_rows_are_deterministic_and_save_as_a_standard_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    _make_source(source_path)
    _patch_fixture_hashes(source_path, monkeypatch)

    first = initialize_aiguard24(source_path, seed=101).model
    second = initialize_aiguard24(source_path, seed=101).model
    different = initialize_aiguard24(source_path, seed=102).model
    label2id = first.config.label2id
    unmapped_ids = [
        label2id[f"{prefix}-{label}"]
        for label in first.config.pii_lineage["unmapped_target_labels"]
        for prefix in ("B", "I")
    ]
    torch.testing.assert_close(
        first.score.weight[unmapped_ids],
        second.score.weight[unmapped_ids],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(first.score.weight[unmapped_ids], different.score.weight[unmapped_ids])

    output = tmp_path / "output"
    first.save_pretrained(output, safe_serialization=True)
    reloaded = Qwen3ForTokenClassification.from_pretrained(
        output,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        weights_only=True,
    )
    assert type(reloaded) is Qwen3ForTokenClassification
    assert reloaded.config.pii_lineage == first.config.pii_lineage
    assert reloaded.config.pii_release_eligible is False


def test_rejects_hash_mismatch_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    _make_source(source_path)
    hashes = _source_hashes(source_path)
    hashes["config.json"] = "0" * 64
    monkeypatch.setattr(aiguard24, "_EXPECTED_SOURCE_HASHES", hashes)

    with pytest.raises(Aiguard24InitializationError, match="pinned AIguard revision"):
        initialize_aiguard24(source_path)


def test_rejects_changed_source_label_inventory_even_when_fixture_hash_is_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    _make_source(source_path)
    config_path = source_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    first = config["id2label"]["1"]
    second = config["id2label"]["14"]
    config["id2label"]["1"] = second
    config["id2label"]["14"] = first
    config["label2id"] = {label: int(label_id) for label_id, label in config["id2label"].items()}
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    _patch_fixture_hashes(source_path, monkeypatch)

    with pytest.raises(Aiguard24InitializationError, match="label inventory"):
        initialize_aiguard24(source_path)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_rejects_invalid_seed(seed: object, tmp_path: Path) -> None:
    with pytest.raises(Aiguard24InitializationError, match="seed"):
        initialize_aiguard24(tmp_path, seed=seed)  # type: ignore[arg-type]
