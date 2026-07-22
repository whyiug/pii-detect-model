from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForTokenClassification

from pii_zh.inference import project_full_bie
from pii_zh.inference.bie_decoding import BIE_DECODER_IDS
from pii_zh.inference.project_bie import PROJECT_CLOSED8_LABELS
from pii_zh.inference.project_full_bie import (
    FULL_BIE73_INITIALIZATION_STRATEGY,
    FULL_BIE73_MANIFEST_TYPE,
    FULL_BIE73_PROTOCOL_ID,
    FullBie73Error,
    FullBie73SpanPredictor,
    load_local_full_bie73_predictor,
    verify_full_bie73_artifact,
)
from pii_zh.models.aiguard24 import AIGUARD_SOURCE_MODEL_ID, AIGUARD_SOURCE_REVISION
from pii_zh.models.aiguard24_bie import (
    AIGUARD24_BIE_INITIALIZATION_STRATEGY,
    build_core_bie_label_maps,
)
from pii_zh.models.bie_attention_conversion import convert_causal_bie73_to_full_attention
from pii_zh.taxonomy import load_taxonomy
from pii_zh.tokenization import configure_character_boundary_tokenizer
from pii_zh.training.config import LoraSettings
from pii_zh.training.manifest import (
    canonical_json_hash,
    output_artifact_fingerprint,
    write_training_manifest,
)


def _make_full_bie73_model() -> tuple[torch.nn.Module, dict[str, object]]:
    label2id, id2label = build_core_bie_label_maps()
    config = Qwen3Config(
        vocab_size=96,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=73,
        label2id=label2id,
        id2label=id2label,
        architectures=["Qwen3ForTokenClassification"],
        classifier_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    config.pii_attention_mode = "causal"
    config.pii_tagging_scheme = "BIE"
    config.pii_release_eligible = False
    config.pii_training_status = "initialized_untrained"
    config.pii_taxonomy_version = load_taxonomy().taxonomy_version
    torch.manual_seed(73)
    source = Qwen3ForTokenClassification(config)
    targets = LoraSettings().target_modules
    model, audit = convert_causal_bie73_to_full_attention(
        source,
        lora_target_modules=targets,
    )
    model.config.pii_training_status = "completed_independent_dev_feasible_candidate"
    model.config.pii_taxonomy_version = load_taxonomy().taxonomy_version
    model.config.pii_release_eligible = False
    return model, {"attention_conversion": audit.to_dict(), "lora_targets": list(targets)}


def _write_tokenizer(root: Path) -> None:
    backend = Tokenizer(BPE(vocab={"<unk>": 0}, merges=[], unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.eos_token = tokenizer.unk_token
    configure_character_boundary_tokenizer(tokenizer).save_pretrained(root)


def _completed_manifest(
    root: Path,
    *,
    conversion: dict[str, object],
) -> dict[str, object]:
    label2id, _ = build_core_bie_label_maps()
    targets = conversion["lora_targets"]
    recipe = {
        "attention_mode": "full",
        "attention_backend": "sdpa",
        "tag_scheme": "BIE",
        "label_count": 73,
        "update_mode": "lora",
        "initialization_strategy": FULL_BIE73_INITIALIZATION_STRATEGY,
        "checkpoint_selection_split": "independent_validation",
        "checkpoint_selection_metric": "independent_dev_selection_score",
        "lora": {"target_modules": targets},
    }
    initialization = {
        "strategy": AIGUARD24_BIE_INITIALIZATION_STRATEGY,
        "source_model_id": AIGUARD_SOURCE_MODEL_ID,
        "source_revision": AIGUARD_SOURCE_REVISION,
        "target_tag_scheme": "BIE",
        "target_label_count": 73,
        "target_label2id": label2id,
        "attention_mode": "causal",
        "release_eligible": False,
        "attention_conversion": conversion["attention_conversion"],
    }
    dataset_receipt = {
        "sha256": "b" * 64,
        "summary": {"document_count": 24},
        "admission": {"status": "passed"},
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": FULL_BIE73_MANIFEST_TYPE,
        "status": "completed",
        "release_eligible": False,
        "created_at": "2026-07-21T00:00:00Z",
        "completed_at": "2026-07-21T00:01:00Z",
        "seed": 73,
        "attention_mode": "full",
        "tag_scheme": "BIE",
        "update_mode": "lora",
        "validation_split_name": "dev",
        "base_source_id": AIGUARD_SOURCE_MODEL_ID,
        "source_revision": AIGUARD_SOURCE_REVISION,
        "training_source_ids": [AIGUARD_SOURCE_MODEL_ID, "unit-clean-room-v7"],
        "taxonomy_version": load_taxonomy().taxonomy_version,
        "label2id": label2id,
        "label_schema_sha256": canonical_json_hash(label2id),
        "recipe": recipe,
        "recipe_sha256": canonical_json_hash(recipe),
        "initialization": initialization,
        "parameter_update_audit": {"status": "passed"},
        "tokenizer": {"boundary_mode": "unicode_codepoint_v1"},
        "datasets": {
            "train": deepcopy(dataset_receipt),
            "independent_validation": deepcopy(dataset_receipt),
        },
        "split_isolation": {
            "policy": "strict_all_groups",
            "collision_counts": {
                "document_id": 0,
                "template_group": 0,
                "entity_value_group": 0,
                "source_group": 0,
            },
            "template_group_overlap_allowed": False,
        },
        "dev_selection_gates": {"status": "passed"},
        "benchmark_isolation": {
            "public_test_read_for_training": False,
            "public_test_read_for_validation": False,
            "public_test_read_for_checkpoint_selection": False,
            "pii_bench_zh_read": False,
        },
        "versions": {"torch": torch.__version__},
        "code_revision": "0" * 40,
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "trainer_reporting_integrations": [],
        },
        "runtime_environment": {"device": "cuda:0"},
        "checkpoint_selection": {
            "selection_split": "independent_validation",
            "selection_metric": "independent_dev_selection_score",
            "update_mode": "lora",
            "selected_checkpoint_id": "checkpoint-7",
            "selected_global_step": 7,
            "best_metric": 3.25,
            "final_validation_replay_metric": 3.25,
            "safetensor_sha256": {"adapter_model.safetensors": "a" * 64},
        },
        "independent_validation": {"eval_dev_gate_passed": 1.0},
        "output_artifact": output_artifact_fingerprint(root),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    return manifest


def _write_valid_artifact(root: Path) -> dict[str, object]:
    model, conversion = _make_full_bie73_model()
    model.save_pretrained(root, safe_serialization=True)
    _write_tokenizer(root)
    manifest = _completed_manifest(root, conversion=conversion)
    write_training_manifest(manifest, root)
    return manifest


def test_valid_full_bie73_artifact_loads_locally_and_restores_release_flag(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_artifact(tmp_path)
    raw_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert raw_config["pii_release_eligible"] is False

    verified = verify_full_bie73_artifact(tmp_path)
    assert verified.identity.manifest_type == FULL_BIE73_MANIFEST_TYPE
    assert verified.identity.protocol_id == FULL_BIE73_PROTOCOL_ID
    assert verified.identity.model_type == "qwen3_bi"
    assert verified.identity.attention_mode == "full"
    assert verified.identity.tag_scheme == "BIE"
    assert verified.identity.label_count == 73
    assert verified.identity.manifest_sha256 == manifest["manifest_sha256"]

    predictor = load_local_full_bie73_predictor(
        tmp_path,
        decoder_id="constrained_viterbi",
        device="cpu",
        micro_batch_size=1,
    )
    assert type(predictor.model) is project_full_bie.Qwen3BiForTokenClassification
    assert predictor.model.config.pii_release_eligible is False
    assert predictor.model.config.model_type == "qwen3_bi"
    assert predictor.decoder_id == "constrained_viterbi"
    assert predictor.predict("") == []


def test_tampered_release_flag_is_rejected_before_model_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_artifact(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pii_release_eligible"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    reached = False

    def forbidden(*_: object, **__: object) -> None:
        nonlocal reached
        reached = True
        pytest.fail("model deserialization was reached")

    monkeypatch.setattr(
        project_full_bie.Qwen3BiForTokenClassification,
        "from_pretrained",
        forbidden,
    )
    with pytest.raises(FullBie73Error, match="config protocol"):
        load_local_full_bie73_predictor(tmp_path, decoder_id="greedy")
    assert reached is False


def test_artifact_rejects_any_symlink_and_causal_manifest_type(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "outside").write_text("unit", encoding="utf-8")
    (nested / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(FullBie73Error, match="no symlinks"):
        verify_full_bie73_artifact(tmp_path)

    clean = tmp_path / "clean"
    manifest = _write_valid_artifact(clean)
    manifest["manifest_type"] = "aiguard24_native_causal_bie_training_v1"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_training_manifest(manifest, clean)
    with pytest.raises(FullBie73Error, match="manifest protocol"):
        verify_full_bie73_artifact(clean)

    (clean / "nested").mkdir()
    (clean / "nested" / "extra.safetensors").write_bytes(b"fixture")
    with pytest.raises(FullBie73Error, match="exactly model.safetensors"):
        verify_full_bie73_artifact(clean)


class _FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        assert all(len(text) == 2 for text in texts)
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
            "offset_mapping": torch.tensor([[[0, 1], [1, 2]]] * batch, dtype=torch.long),
        }


class _FakeFullBieModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        label2id, id2label = build_core_bie_label_maps()
        self.label2id = label2id
        self.config = SimpleNamespace(
            model_type="qwen3_bi",
            pii_attention_mode="full",
            pii_tagging_scheme="BIE",
            pii_release_eligible=False,
            use_cache=False,
            id2label=id2label,
        )

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        logits = torch.zeros((*input_ids.shape, 73), device=input_ids.device)
        logits[..., self.label2id["B-WECHAT_ID"]] = 100
        logits[:, 0, self.label2id["B-PERSON_NAME"]] = 10
        logits[:, 1, self.label2id["E-PERSON_NAME"]] = 10
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("decoder_id", sorted(BIE_DECODER_IDS))
def test_predictor_binds_decoder_and_exact_closed8_predecode_mask(decoder_id: str) -> None:
    predictor = FullBie73SpanPredictor(
        _FakeFullBieModel(),
        _FakeTokenizer(),
        decoder_id=decoder_id,
        allowed_project_labels=PROJECT_CLOSED8_LABELS,
    )
    assert predictor.decoder_id == decoder_id
    assert len(predictor.allowed_label_ids) == 25
    assert {predictor.id2label[label_id] for label_id in predictor.allowed_label_ids} == {
        "O",
        *(f"{prefix}-{label}" for label in PROJECT_CLOSED8_LABELS for prefix in ("B", "I", "E")),
    }
    prediction = predictor.predict("张三")
    assert prediction[0].pop("score") > 0.99
    assert prediction == [
        {
            "entity_type": "PERSON_NAME",
            "start": 0,
            "end": 2,
            "offset_mode": "relative",
            "metadata": {
                "boundary_refined": False,
                "protocol_id": FULL_BIE73_PROTOCOL_ID,
                "decoder_id": decoder_id,
            },
        }
    ]


def test_decoder_id_is_mandatory_and_unknown_decoder_fails_closed() -> None:
    with pytest.raises(TypeError, match="decoder_id"):
        FullBie73SpanPredictor(_FakeFullBieModel(), _FakeTokenizer())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="decoder"):
        FullBie73SpanPredictor(
            _FakeFullBieModel(),
            _FakeTokenizer(),
            decoder_id="automatic",
        )
