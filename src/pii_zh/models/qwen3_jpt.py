"""Sequence-repetition (JPT-style) input construction for causal baselines."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise ImportError("qwen3_jpt requires the optional PyTorch dependency.") from exc


def apply_second_copy_label_mask(
    labels: torch.Tensor,
    second_copy_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Keep labels on the repeated second copy and ignore every other token."""

    if labels.shape != second_copy_mask.shape:
        raise ValueError(
            f"labels and second_copy_mask must have the same shape, got "
            f"{tuple(labels.shape)!r} and {tuple(second_copy_mask.shape)!r}."
        )
    if second_copy_mask.dtype is not torch.bool:
        second_copy_mask = second_copy_mask.to(dtype=torch.bool)
    return labels.masked_fill(~second_copy_mask, ignore_index)


def build_jpt_inputs(
    input_ids: torch.LongTensor,
    *,
    sep_token_id: int,
    pad_token_id: int,
    attention_mask: torch.Tensor | None = None,
    labels: torch.LongTensor | None = None,
    ignore_index: int = -100,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build ``tokens + separator + tokens`` inputs for JPT training.

    Valid tokens are selected with ``attention_mask`` and compacted into a
    common padded source width.  The layout is always
    ``padded_source + separator + padded_source``; consequently every row in
    a collated batch has the same second-copy start.  Labels are copied only
    onto valid positions in the second sequence.  The function raises instead
    of truncating when ``max_length`` would cut a sequence.
    """

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank 2, got shape {tuple(input_ids.shape)!r}.")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids.")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids.")

    valid = attention_mask.to(dtype=torch.bool)
    lengths = valid.sum(dim=-1)
    if not bool((lengths > 0).all()):
        raise ValueError("Each example must contain at least one valid source token.")
    source_width = input_ids.shape[1]
    output_length = source_width * 2 + 1
    if max_length is not None and output_length > max_length:
        raise ValueError(
            f"JPT input requires {output_length} tokens but max_length={max_length}; "
            "pre-chunk the source rather than truncating an entity."
        )

    batch_size = input_ids.shape[0]
    output_ids = input_ids.new_full((batch_size, output_length), pad_token_id)
    output_attention = torch.zeros(
        (batch_size, output_length), dtype=torch.bool, device=input_ids.device
    )
    second_copy_mask = torch.zeros_like(output_attention)
    output_labels = None
    if labels is not None:
        output_labels = labels.new_full((batch_size, output_length), ignore_index)

    for row in range(batch_size):
        source_ids = input_ids[row][valid[row]]
        source_length = source_ids.numel()
        separator_index = source_width
        second_start = separator_index + 1
        second_end = second_start + source_length

        output_ids[row, :source_length] = source_ids
        output_ids[row, separator_index] = sep_token_id
        output_ids[row, second_start:second_end] = source_ids
        output_attention[row, :source_length] = True
        output_attention[row, separator_index] = True
        output_attention[row, second_start:second_end] = True
        second_copy_mask[row, second_start:second_end] = True

        if output_labels is not None and labels is not None:
            output_labels[row, second_start:second_end] = labels[row][valid[row]]

    result: dict[str, torch.Tensor] = {
        "input_ids": output_ids,
        "attention_mask": output_attention,
        "second_copy_mask": second_copy_mask,
    }
    if output_labels is not None:
        result["labels"] = apply_second_copy_label_mask(
            output_labels, second_copy_mask, ignore_index=ignore_index
        )
    return result
