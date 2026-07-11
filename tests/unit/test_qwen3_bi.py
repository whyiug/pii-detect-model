from __future__ import annotations

from fractions import Fraction

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForTokenClassification

from pii_zh.models.qwen3_bi import (
    Qwen3BiConfig,
    Qwen3BiForTokenClassification,
    build_full_attention_mask,
)


def tiny_config(*, backend: str = "eager") -> Qwen3BiConfig:
    return Qwen3BiConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=5,
        classifier_dropout=0.0,
        attention_dropout=0.0,
        bi_attention_backend=backend,
    )


def make_model(*, backend: str = "eager") -> Qwen3BiForTokenClassification:
    torch.manual_seed(7)
    model = Qwen3BiForTokenClassification(tiny_config(backend=backend))
    model.eval()
    return model


def test_classifier_dropout_accepts_real_values_but_rejects_bool() -> None:
    config = tiny_config()
    config.classifier_dropout = Fraction(1, 4)
    model = Qwen3BiForTokenClassification(config)
    assert model.dropout.p == pytest.approx(0.25)

    config.classifier_dropout = True
    with pytest.raises(TypeError, match="real number"):
        Qwen3BiForTokenClassification(config)


@torch.no_grad()
def test_future_token_changes_bi_logits_but_not_causal_logits() -> None:
    torch.manual_seed(11)
    causal_config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=5,
        classifier_dropout=0.0,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    causal = Qwen3ForTokenClassification(causal_config).eval()
    bi = Qwen3BiForTokenClassification.from_qwen3_token_classifier_state_dict(
        causal_config, causal.state_dict(), bi_attention_backend="eager"
    ).eval()

    original = torch.tensor([[3, 5, 7, 9]])
    changed = torch.tensor([[3, 5, 7, 31]])
    mask = torch.ones_like(original)

    causal_original = causal(original, attention_mask=mask).logits[:, 0]
    causal_changed = causal(changed, attention_mask=mask).logits[:, 0]
    torch.testing.assert_close(causal_original, causal_changed, rtol=0.0, atol=1e-7)

    bi_original = bi(original, attention_mask=mask).logits[:, 0]
    bi_changed = bi(changed, attention_mask=mask).logits[:, 0]
    assert (bi_original - bi_changed).abs().max().item() > 1e-7


@torch.no_grad()
def test_padding_token_content_cannot_change_valid_logits() -> None:
    model = make_model()
    first = torch.tensor([[3, 5, 7, 11, 13]])
    second = torch.tensor([[3, 5, 7, 41, 43]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    first_logits = model(first, attention_mask=mask).logits[:, :3]
    second_logits = model(second, attention_mask=mask).logits[:, :3]
    torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=1e-6)


@torch.no_grad()
def test_save_load_parity(tmp_path) -> None:
    model = make_model()
    input_ids = torch.tensor([[2, 4, 6, 8]])
    attention_mask = torch.ones_like(input_ids)
    expected = model(input_ids, attention_mask=attention_mask).logits

    model.save_pretrained(tmp_path)
    loaded = Qwen3BiForTokenClassification.from_pretrained(tmp_path).eval()
    actual = loaded(input_ids, attention_mask=attention_mask).logits

    assert loaded.config.model_type == "qwen3_bi"
    assert loaded.config.use_cache is False
    assert loaded.config.bi_attention_backend == "eager"
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_batch_and_individual_logits_match_on_valid_tokens() -> None:
    model = make_model()
    batch_ids = torch.tensor([[3, 4, 5, 0], [7, 8, 9, 10]])
    batch_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    batch_logits = model(batch_ids, attention_mask=batch_mask).logits

    for row, length in enumerate((3, 4)):
        single_logits = model(
            batch_ids[row : row + 1, :length],
            attention_mask=torch.ones((1, length), dtype=torch.long),
        ).logits
        torch.testing.assert_close(
            batch_logits[row : row + 1, :length], single_logits, rtol=0.0, atol=1e-6
        )


def test_early_loss_has_gradient_to_future_embedding() -> None:
    model = make_model()
    input_ids = torch.tensor([[3, 5, 7, 9]])
    inputs_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=inputs_embeds, attention_mask=torch.ones_like(input_ids)).logits
    logits[0, 0].square().sum().backward()
    assert inputs_embeds.grad is not None
    assert inputs_embeds.grad[0, -1].abs().sum().item() > 0.0


def test_full_mask_has_no_all_infinite_query_row() -> None:
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    mask = build_full_attention_mask(valid, dtype=torch.float32)
    assert mask.shape == (2, 1, 4, 4)
    assert not torch.isinf(mask).any()
    assert not torch.isneginf(mask).all(dim=-1).any()
    assert torch.equal(mask[0, 0, 3, :2], torch.zeros(2))
    assert (mask[0, 0, :, 2:] == torch.finfo(torch.float32).min).all()


@torch.no_grad()
def test_sdpa_backend_smoke() -> None:
    model = make_model(backend="sdpa")
    input_ids = torch.tensor([[3, 5, 7, 0]])
    output = model(input_ids, attention_mask=torch.tensor([[1, 1, 1, 0]])).logits
    assert output.shape == (1, 4, model.config.num_labels)
    assert torch.isfinite(output).all()
