from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    EvalPrediction,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    TrainingArguments,
)

from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    iter_jsonl,
    stable_text_hash,
    write_jsonl,
)
from pii_zh.inference import InferenceSafetyError, load_local_predictor
from pii_zh.models.qwen3_bi import Qwen3BiConfig, Qwen3BiForTokenClassification
from pii_zh.tokenization import configure_character_boundary_tokenizer
from pii_zh.training.config import LoraSettings, TrainingConfig, load_training_config
from pii_zh.training.data import (
    DocumentTokenDataset,
    DynamicTokenCollator,
    EncodedDocument,
    GoldSpan,
    TrainingDataError,
    TrainingDataSummary,
    TrainingSourceSummary,
    assert_disjoint_training_splits,
    compute_class_weights,
    load_aligned_jsonl,
)
from pii_zh.training.loading import load_token_classifier_from_local_causal_lm
from pii_zh.training.manifest import (
    build_training_manifest,
    canonical_json_hash,
    finalize_training_manifest,
    verify_output_artifact_binding,
    verify_training_manifest,
    write_training_manifest,
)
from pii_zh.training.pipeline import run_training
from pii_zh.training.trainer import (
    CharacterValidationComputer,
    SecureTokenTrainer,
    merge_parameter_efficient_model_for_release,
    prepare_parameter_efficient_model,
)


def _tiny_causal_checkpoint(path) -> None:
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
    )
    Qwen3ForCausalLM(config).save_pretrained(path, safe_serialization=True)


@pytest.mark.parametrize("attention_mode", ["causal", "full"])
def test_causal_lm_backbone_mapping_is_explicitly_audited(tmp_path, attention_mode: str) -> None:
    checkpoint = tmp_path / "base"
    _tiny_causal_checkpoint(checkpoint)
    label2id = {"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2}
    id2label = {value: key for key, value in label2id.items()}

    model, audit = load_token_classifier_from_local_causal_lm(
        checkpoint,
        attention_mode=attention_mode,
        label2id=label2id,
        id2label=id2label,
    )

    assert audit.lm_head_present_in_checkpoint is True
    assert audit.lm_head_discarded is True
    assert audit.unexpected_keys == ("lm_head.weight",)
    assert audit.missing_keys == ("score.bias", "score.weight")
    assert audit.newly_initialized_score_keys == ("score.bias", "score.weight")
    assert not audit.mismatched_keys
    assert not audit.error_messages
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )
    assert output.logits.shape == (1, 3, 3)


def test_class_weights_use_sqrt_frequency_with_cap() -> None:
    weights = compute_class_weights([(0, 0, 0, 1, -100), (0, 2, -100)], num_labels=4, cap=2.0)
    assert weights[0].item() == pytest.approx((6 / 4) ** 0.5)
    assert weights[1].item() == pytest.approx(2.0)
    assert weights[2].item() == pytest.approx(2.0)
    assert weights[3].item() == pytest.approx(1.0)


class _CharacterTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs):
        del kwargs
        return {
            "input_ids": [1, *range(10, 10 + len(text)), 2],
            "attention_mask": [1] * (len(text) + 2),
            "offset_mapping": [(0, 0), *[(i, i + 1) for i in range(len(text))], (0, 0)],
            "special_tokens_mask": [1, *([0] * len(text)), 1],
        }


def test_jsonl_alignment_discards_raw_text_and_dynamic_padding(tmp_path) -> None:
    record = DocumentRecord.create(
        doc_id="synthetic-training-1",
        text="张三到访",
        entities=[EntitySpan(start=0, end=2, label="PERSON_NAME", text="张三", source="gold")],
        language="zh-Hans-CN",
        domain="synthetic",
        scene="unit-test",
        provenance=Provenance(
            source_id="unit-test",
            source_kind="synthetic",
            source_revision="1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit-test",
            generator_version="1",
            generator_seed=1,
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="validated",
        ),
        public_weight_training_allowed=True,
        generator_version="1",
        split="train",
    )
    path = tmp_path / "train.jsonl"
    write_jsonl([record], path)
    dataset, summary = load_aligned_jsonl(
        path,
        tokenizer=_CharacterTokenizer(),
        label2id={"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2},
        max_length=16,
        expected_split="train",
    )

    assert summary.document_count == 1
    assert not hasattr(dataset.documents[0], "text")
    assert dataset.documents[0].labels == (-100, 1, 2, 0, 0, -100)
    collated = DynamicTokenCollator(pad_token_id=0, pad_to_multiple_of=8)([dataset[0]])
    assert collated["input_ids"].shape == (1, 8)
    assert collated["labels"][0, -2:].tolist() == [-100, -100]
    assert summary.split_counts == {"train": 1}
    assert summary.data_pool_counts == {"public_release_pool": 1}
    assert summary.quality_tier_counts == {"S0": 1}
    assert summary.sources[0].source_id == "unit-test"


@pytest.mark.parametrize(
    ("kind", "train_updates", "validation_updates"),
    [
        ("document_id", {"doc_id": "shared"}, {"doc_id": "shared"}),
        (
            "template_group",
            {"template_group": "private-template"},
            {"template_group": "private-template"},
        ),
        (
            "entity_value_group",
            {"entity_value_groups": (stable_text_hash("synthetic-value"),)},
            {"entity_value_groups": (stable_text_hash("synthetic-value"),)},
        ),
        (
            "source_group",
            {"tenant_group": "private-tenant"},
            {"tenant_group": "private-tenant"},
        ),
    ],
)
def test_training_files_require_exact_splits_and_zero_group_overlap(
    tmp_path, kind: str, train_updates: dict, validation_updates: dict
) -> None:
    base = next(iter_jsonl(_write_training_record(tmp_path, split="train")))
    train_record = replace(base, **train_updates, split="train")
    validation_record = replace(
        base,
        **({"doc_id": "validation-doc", "split": "validation"} | validation_updates),
    )
    train_path = tmp_path / "strict-train.jsonl"
    validation_path = tmp_path / "strict-validation.jsonl"
    write_jsonl([train_record], train_path)
    write_jsonl([validation_record], validation_path)
    kwargs = {
        "tokenizer": _CharacterTokenizer(),
        "label2id": {"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2},
        "max_length": 16,
    }
    train_dataset, _ = load_aligned_jsonl(train_path, expected_split="train", **kwargs)
    validation_dataset, _ = load_aligned_jsonl(
        validation_path, expected_split="validation", **kwargs
    )
    with pytest.raises(TrainingDataError, match=kind) as captured:
        assert_disjoint_training_splits(train_dataset, validation_dataset)
    assert "private-" not in str(captured.value)

    with pytest.raises(TrainingDataError, match="required 'validation' split"):
        load_aligned_jsonl(train_path, expected_split="validation", **kwargs)


def _write_training_record(tmp_path, *, split: str):
    record = DocumentRecord.create(
        doc_id="training-doc",
        text="张三到访",
        entities=[EntitySpan(start=0, end=2, label="PERSON_NAME", text="张三", source="gold")],
        language="zh-Hans-CN",
        domain="synthetic",
        scene="unit-test",
        provenance=Provenance(
            source_id="unit-test",
            source_kind="synthetic",
            source_revision="1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit-test",
            generator_version="1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="validated",
        ),
        public_weight_training_allowed=True,
        generator_version="1",
        split=split,
    )
    path = tmp_path / f"source-{split}.jsonl"
    write_jsonl([record], path)
    return path


def test_training_manifest_is_hashed_and_contains_no_paths_or_text(tmp_path) -> None:
    checkpoint = tmp_path / "base"
    _tiny_causal_checkpoint(checkpoint)
    data_path = tmp_path / "records.jsonl"
    data_path.write_text('{"synthetic":"metadata-only"}\n', encoding="utf-8")
    label2id = {"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2}
    _, audit = load_token_classifier_from_local_causal_lm(
        checkpoint,
        attention_mode="causal",
        label2id=label2id,
        id2label={value: key for key, value in label2id.items()},
    )
    config = TrainingConfig(
        base_model=str(checkpoint),
        train_file=str(data_path),
        validation_file=str(data_path),
        output_dir=str(tmp_path / "output"),
        attention_mode="causal",
        fine_tuning="full",
        bf16=False,
        use_cpu=True,
    )
    summary = TrainingDataSummary(
        document_count=1,
        entity_count=0,
        token_count=3,
        pii_free_document_count=1,
        unalignable_boundary_count=0,
        label_counts={},
        split_counts={"train": 1},
        data_pool_counts={"public_release_pool": 1},
        quality_tier_counts={"S0": 1},
        quality_gate_passed_document_count=1,
        validators_passed_document_count=1,
        public_weight_training_allowed_document_count=1,
        sources=(
            TrainingSourceSummary(
                source_id="synthetic-source",
                source_kind="synthetic",
                source_revision="immutable-v1",
                license="Apache-2.0",
                data_pool="public_release_pool",
                split="train",
                document_count=1,
                entity_count=0,
                quality_tier_counts={"S0": 1},
                quality_gate_passed_document_count=1,
                validators_passed_document_count=1,
                public_weight_training_allowed_document_count=1,
            ),
        ),
    )
    manifest = build_training_manifest(
        config=config,
        loading_audit=audit,
        train_summary=summary,
        validation_summary=summary,
        taxonomy_version="1.0.0",
        label2id=label2id,
        created_at="2026-07-11T00:00:00+00:00",
        worktree=tmp_path,
    )

    assert verify_training_manifest(manifest)
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "synthetic-training" not in serialized
    destination = write_training_manifest(manifest, config.output_dir)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["seed"] == 42
    source = loaded["datasets"]["train"]["summary"]["sources"][0]
    assert source == {
        "data_pool": "public_release_pool",
        "document_count": 1,
        "entity_count": 0,
        "license": "Apache-2.0",
        "public_weight_training_allowed_document_count": 1,
        "quality_gate_passed_document_count": 1,
        "quality_tier_counts": {"S0": 1},
        "revision": "immutable-v1",
        "source_id": "synthetic-source",
        "source_kind": "synthetic",
        "split": "train",
        "validators_passed_document_count": 1,
    }
    assert loaded["versions"]["torch"] == torch.__version__.split("+")[0]
    assert verify_training_manifest(loaded)


def test_classifier_learning_rate_defaults_and_manifest_recipe(tmp_path) -> None:
    common = {
        "base_model": str(tmp_path / "base"),
        "train_file": str(tmp_path / "train"),
        "validation_file": str(tmp_path / "validation"),
        "output_dir": str(tmp_path / "output"),
        "learning_rate": 2e-5,
    }
    full = TrainingConfig(**common, fine_tuning="full")
    lora = TrainingConfig(**common, fine_tuning="lora")
    explicit = TrainingConfig(
        **common,
        fine_tuning="full",
        classifier_learning_rate=3e-4,
    )
    assert full.effective_classifier_learning_rate == pytest.approx(2e-4)
    assert lora.effective_classifier_learning_rate == pytest.approx(5e-4)
    assert explicit.manifest_recipe()["effective_classifier_learning_rate"] == pytest.approx(3e-4)


def test_public_runtime_recipe_is_strict_full_attention() -> None:
    config = load_training_config("configs/train/qwen3_0_6b_full_attention.yaml")
    assert config.attention_mode == "full"
    assert config.fine_tuning == "full"
    assert config.learning_rate == pytest.approx(2e-5)
    assert config.effective_classifier_learning_rate == pytest.approx(2e-4)


class _OptimizerGroupingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.score = torch.nn.Module()
        self.base_model.model.score.modules_to_save = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(4, 3)}
        )

    def forward(self, input_ids=None, **kwargs):
        del input_ids, kwargs
        return SimpleNamespace(logits=torch.zeros((1, 1, 3)))


def test_optimizer_groups_classifier_without_omission_or_duplication(tmp_path) -> None:
    model = _OptimizerGroupingModel()
    trainer = SecureTokenTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            learning_rate=2e-5,
            weight_decay=0.1,
            report_to=[],
            use_cpu=True,
        ),
        classifier_learning_rate=2e-4,
    )
    optimizer = trainer.create_optimizer()
    grouped_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    head_ids = {id(parameter) for parameter in model.base_model.model.score.parameters()}
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == trainable_ids
    for group in optimizer.param_groups:
        identities = {id(parameter) for parameter in group["params"]}
        expected_lr = 2e-4 if identities & head_ids else 2e-5
        assert group["lr"] == pytest.approx(expected_lr)
        assert identities <= head_ids or identities.isdisjoint(head_ids)


def test_jpt_validation_uses_collated_width_for_different_length_rows() -> None:
    documents = [
        EncodedDocument(
            doc_id="short",
            input_ids=(1, 2, 3),
            attention_mask=(1, 1, 1),
            labels=(1, 0, 0),
            offset_mapping=((0, 1), (1, 2), (2, 3)),
            gold_spans=(GoldSpan(0, 1, "PERSON_NAME"),),
        ),
        EncodedDocument(
            doc_id="long",
            input_ids=(1, 2, 3, 4, 5),
            attention_mask=(1, 1, 1, 1, 1),
            labels=(1, 0, 0, 0, 0),
            offset_mapping=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
            gold_spans=(GoldSpan(0, 1, "PERSON_NAME"),),
        ),
    ]
    dataset = DocumentTokenDataset(documents)
    # Collated source width is five, so the second copy begins at index six
    # for both rows.  The old per-document start would read index four for row 0.
    predicted = np.zeros((2, 11), dtype=np.int64)
    predicted[:, 6] = 1
    logits = np.full((2, 11, 3), -10.0, dtype=np.float32)
    for row in range(2):
        logits[row, np.arange(11), predicted[row]] = 10.0
    labels = np.full((2, 11), -100, dtype=np.int64)
    labels[0, 6:9] = (1, 0, 0)
    labels[1, 6:11] = (1, 0, 0, 0, 0)
    computer = CharacterValidationComputer(
        dataset,
        id2label={0: "O", 1: "B-PERSON_NAME", 2: "I-PERSON_NAME"},
        label_to_risk_tiers={"PERSON_NAME": ("T1",)},
        attention_mode="jpt",
    )
    metrics = computer(EvalPrediction(predictions=logits, label_ids=labels))
    assert metrics["strict_macro_f1"] == pytest.approx(1.0)
    assert metrics["tier1_f2"] == pytest.approx(1.0)


def _boundary_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(BPE(vocab={"<unk>": 0}, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.pad_token = tokenizer.unk_token
    return configure_character_boundary_tokenizer(tokenizer)


def _bind_manual_artifact(path) -> None:
    manifest = {"schema_version": 2, "status": "in_progress"}
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    finalized = finalize_training_manifest(
        manifest,
        path,
        completed_at="2026-07-11T00:00:00+00:00",
        runtime_environment={"execution_device": "cpu"},
    )
    write_training_manifest(finalized, path)


@pytest.mark.parametrize("fine_tuning", ["full", "lora"])
def test_final_model_is_standalone_qwen3bi_and_loadable_on_cpu(tmp_path, fine_tuning: str) -> None:
    config = Qwen3BiConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=3,
        id2label={0: "O", 1: "B-PERSON_NAME", 2: "I-PERSON_NAME"},
        label2id={"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2},
        bi_attention_backend="eager",
    )
    config._name_or_path = "/private/base/checkpoint"
    model = Qwen3BiForTokenClassification(config)
    if fine_tuning == "lora":
        model = prepare_parameter_efficient_model(
            model,
            fine_tuning="lora",
            lora=LoraSettings(rank=2, alpha=4, target_modules=("q_proj",)),
        )
    final_model = merge_parameter_efficient_model_for_release(
        model,
        fine_tuning=fine_tuning,
    )
    final_model.save_pretrained(tmp_path, safe_serialization=True)
    _boundary_tokenizer().save_pretrained(tmp_path)
    _bind_manual_artifact(tmp_path)

    serialized_config = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "/private/base/checkpoint" not in serialized_config
    assert not (tmp_path / "adapter_config.json").exists()
    assert not (tmp_path / "adapter_model.safetensors").exists()
    predictor = load_local_predictor(tmp_path, device="cpu", micro_batch_size=1)
    assert predictor.model.config.model_type == "qwen3_bi"
    with (tmp_path / "model.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(InferenceSafetyError, match="training manifest"):
        load_local_predictor(tmp_path, device="cpu")


@pytest.mark.parametrize("attention_mode", ["causal", "jpt"])
def test_research_ablation_saves_complete_native_qwen3_artifact(
    tmp_path, attention_mode: str
) -> None:
    checkpoint = tmp_path / "base"
    _tiny_causal_checkpoint(checkpoint)
    label2id = {"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2}
    model, _ = load_token_classifier_from_local_causal_lm(
        checkpoint,
        attention_mode=attention_mode,
        label2id=label2id,
        id2label={value: key for key, value in label2id.items()},
    )
    final_model = merge_parameter_efficient_model_for_release(
        model,
        fine_tuning="full",
        attention_mode=attention_mode,
        jpt_sep_token_id=2 if attention_mode == "jpt" else None,
    )
    destination = tmp_path / attention_mode
    final_model.save_pretrained(destination, safe_serialization=True)
    _boundary_tokenizer().save_pretrained(destination)
    _bind_manual_artifact(destination)
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen3"
    assert config["architectures"] == ["Qwen3ForTokenClassification"]
    assert config["pii_attention_mode"] == attention_mode
    assert config["pii_release_eligible"] is False
    assert (destination / "model.safetensors").is_file()
    assert not (destination / "adapter_config.json").exists()
    predictor = load_local_predictor(destination, device="cpu", micro_batch_size=2)
    assert predictor.attention_mode == attention_mode
    assert len(predictor.predict_batch(["甲", "甲乙"])) == 2


@pytest.mark.parametrize("fine_tuning", ["full", "lora"])
def test_tiny_cpu_training_pipeline_saves_loadable_complete_model(
    tmp_path, fine_tuning: str
) -> None:
    checkpoint = tmp_path / "base"
    _tiny_causal_checkpoint(checkpoint)
    _boundary_tokenizer().save_pretrained(checkpoint)
    train_record = next(iter_jsonl(_write_training_record(tmp_path, split="train")))
    validation_record = replace(
        train_record,
        doc_id="validation-doc",
        split="validation",
    )
    train_path = tmp_path / "pipeline-train.jsonl"
    validation_path = tmp_path / "pipeline-validation.jsonl"
    write_jsonl([train_record], train_path)
    write_jsonl([validation_record], validation_path)
    output_dir = tmp_path / f"final-{fine_tuning}"
    result = run_training(
        TrainingConfig(
            base_model=str(checkpoint),
            train_file=str(train_path),
            validation_file=str(validation_path),
            output_dir=str(output_dir),
            attention_mode="full",
            fine_tuning=fine_tuning,
            max_length=32,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            classifier_learning_rate=2e-3,
            epochs=1.0,
            bf16=False,
            use_cpu=True,
            gradient_checkpointing=False,
            class_weighting=False,
            document_sampling=False,
            early_stopping_patience=1,
            save_total_limit=1,
        )
    )
    assert result.eval_metrics
    assert (output_dir / "model.safetensors").is_file()
    assert (output_dir / "final_eval_metrics.json").is_file()
    assert not (output_dir / "trainer_state.json").exists()
    assert (output_dir.parent / f"{output_dir.name}.trainer-state" / "trainer_state.json").is_file()
    assert not any(output_dir.glob("adapter_*"))
    final_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert final_manifest["status"] == "completed"
    assert verify_output_artifact_binding(final_manifest, output_dir, require_single_weight=True)
    assert final_manifest["runtime_environment"]["execution_device"] == "cpu"
    assert final_manifest["runtime_environment"]["bf16_enabled"] is False
    assert isinstance(final_manifest["runtime_environment"]["cuda_available"], bool)
    assert str(tmp_path) not in (output_dir / "config.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    loaded = load_local_predictor(output_dir, device="cpu", micro_batch_size=1)
    assert loaded.attention_mode == "full"
