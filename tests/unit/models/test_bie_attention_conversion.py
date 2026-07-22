from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForTokenClassification

from pii_zh.models.aiguard24_bie import build_core_bie_label_maps
from pii_zh.models.bie_attention_conversion import (
    BIE73_FULL_ATTENTION_CONVERSION_STRATEGY,
    BieAttentionConversionError,
    convert_causal_bie73_to_full_attention,
)
from pii_zh.models.qwen3_bi import Qwen3BiForTokenClassification
from pii_zh.training.config import LoraSettings
from pii_zh.training.trainer import prepare_parameter_efficient_model


def _make_causal_bie73(*, layers: int = 2) -> Qwen3ForTokenClassification:
    label2id, id2label = build_core_bie_label_maps()
    config = Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
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
    config.pii_lineage = {
        "strategy": "unit_aiguard24_bie_projection",
        "target_label_count": 73,
    }
    torch.manual_seed(73)
    return Qwen3ForTokenClassification(config)


@torch.no_grad()
def test_bie73_conversion_preserves_state_head_config_and_runs_cpu_forward() -> None:
    source = _make_causal_bie73().eval()
    for parameter in source.model.parameters():
        parameter.requires_grad_(False)
    source_state = source.state_dict()
    source_head_weight = source.score.weight.detach().clone()
    source_head_bias = source.score.bias.detach().clone()
    lora_targets = LoraSettings().target_modules

    converted, audit = convert_causal_bie73_to_full_attention(
        source,
        lora_target_modules=lora_targets,
    )

    assert type(converted) is Qwen3BiForTokenClassification
    assert converted.training is False
    assert all(parameter.device.type == "cpu" for parameter in converted.parameters())
    assert converted.config.model_type == "qwen3_bi"
    assert converted.config.architectures == ["Qwen3BiForTokenClassification"]
    assert converted.config.num_labels == 73
    assert converted.config.label2id == source.config.label2id
    assert converted.config.id2label == source.config.id2label
    assert converted.config.pii_attention_mode == "full"
    assert converted.config.pii_tagging_scheme == "BIE"
    assert converted.config.pii_training_status == "initialized_untrained"
    assert converted.config.pii_lineage == source.config.pii_lineage
    assert converted.config.pii_release_eligible is False
    assert converted.config.use_cache is False

    assert tuple(converted.state_dict()) == tuple(source_state)
    for key, tensor in converted.state_dict().items():
        torch.testing.assert_close(tensor, source_state[key], rtol=0.0, atol=0.0)
    torch.testing.assert_close(converted.score.weight, source_head_weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(converted.score.bias, source_head_bias, rtol=0.0, atol=0.0)
    assert not any(parameter.requires_grad for parameter in converted.model.parameters())
    assert all(parameter.requires_grad for parameter in converted.score.parameters())

    input_ids = torch.tensor([[3, 5, 7, 9, 0], [11, 13, 17, 19, 23]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    output = converted(input_ids, attention_mask=attention_mask).logits
    assert output.shape == (2, 5, 73)
    assert torch.isfinite(output).all()

    assert audit.strategy == BIE73_FULL_ATTENTION_CONVERSION_STRATEGY
    assert audit.attention_backend == "sdpa"
    assert audit.label_count == 73
    assert audit.state_dict_keys_identical
    assert audit.state_dict_values_identical
    assert audit.classifier_keys == ("score.weight", "score.bias")
    assert audit.classifier_values_identical
    assert audit.config_payload_identical_except_attention_contract
    assert audit.trainability_preserved
    assert audit.source_training_mode_preserved
    assert dict(audit.lora_target_module_counts) == {
        target: 2 for target in lora_targets
    }
    assert audit.newly_initialized_parameter_keys == ()
    assert audit.discarded_parameter_keys == ()
    assert audit.release_eligible is False


@torch.no_grad()
def test_same_bie73_weights_change_future_context_only_under_full_attention() -> None:
    source = _make_causal_bie73(layers=1).eval()
    converted, _ = convert_causal_bie73_to_full_attention(
        source,
        bi_attention_backend="eager",
    )
    converted.eval()
    original = torch.tensor([[3, 5, 7, 9]])
    changed = torch.tensor([[3, 5, 7, 31]])
    attention_mask = torch.ones_like(original)

    causal_original = source(original, attention_mask=attention_mask).logits[:, 0]
    causal_changed = source(changed, attention_mask=attention_mask).logits[:, 0]
    full_original = converted(original, attention_mask=attention_mask).logits[:, 0]
    full_changed = converted(changed, attention_mask=attention_mask).logits[:, 0]

    torch.testing.assert_close(causal_original, causal_changed, rtol=0.0, atol=1e-7)
    assert (full_original - full_changed).abs().max().item() > 1e-7


def test_default_qwen_lora_targets_attach_to_full_bie73_and_forward_on_cpu() -> None:
    source = _make_causal_bie73(layers=1)
    settings = LoraSettings(rank=2, alpha=4, dropout=0.0)
    converted, audit = convert_causal_bie73_to_full_attention(
        source,
        bi_attention_backend="eager",
        lora_target_modules=settings.target_modules,
    )
    peft_model = prepare_parameter_efficient_model(
        converted,
        fine_tuning="lora",
        lora=settings,
    )

    trainable_names = {
        name for name, parameter in peft_model.named_parameters() if parameter.requires_grad
    }
    for target in settings.target_modules:
        assert any(f".{target}.lora_A." in name for name in trainable_names)
    assert any("score.modules_to_save.default" in name for name in trainable_names)
    assert dict(audit.lora_target_module_counts) == {
        target: 1 for target in settings.target_modules
    }

    output = peft_model(
        torch.tensor([[2, 4, 6, 8]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
    ).logits
    assert output.shape == (1, 4, 73)
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pii_attention_mode", "full"),
        ("pii_tagging_scheme", "BIO"),
        ("pii_release_eligible", True),
        ("pii_training_status", "completed_independent_dev_feasible_candidate"),
    ],
)
def test_conversion_rejects_non_causal_or_non_bie_research_contract(
    field: str,
    value: object,
) -> None:
    source = _make_causal_bie73(layers=1)
    setattr(source.config, field, value)
    with pytest.raises(BieAttentionConversionError, match="causal BIE research contract"):
        convert_causal_bie73_to_full_attention(source, bi_attention_backend="eager")


def test_conversion_rejects_missing_lora_target() -> None:
    source = _make_causal_bie73(layers=1)
    with pytest.raises(BieAttentionConversionError, match="requested LoRA target is absent"):
        convert_causal_bie73_to_full_attention(
            source,
            bi_attention_backend="eager",
            lora_target_modules=("q_proj", "not_a_projection"),
        )
