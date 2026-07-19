"""Arbitrary explicit-label projection for the pinned AIguard BIE checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Qwen3ForTokenClassification

from .aiguard import (
    AiguardProtocolError,
    decode_bie_ids,
    normalize_id2label,
)


class AiguardMappedSpanPredictor:
    """Mask AIguard logits to an explicit source mapping before BIE decoding."""

    attention_mode = "checkpoint_native_causal"
    tag_scheme = "BIE"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        source_to_target: Mapping[str, str],
        protocol_id: str,
        device: str | torch.device,
        micro_batch_size: int,
    ) -> None:
        if not source_to_target or any(
            not isinstance(source, str) or not source or not isinstance(target, str) or not target
            for source, target in source_to_target.items()
        ):
            raise AiguardProtocolError("source_to_target must contain non-empty strings")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise AiguardProtocolError("protocol_id must be a non-empty string")
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int):
            raise TypeError("micro_batch_size must be an integer")
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if not getattr(tokenizer, "is_fast", False):
            raise AiguardProtocolError("exact character inference requires a fast tokenizer")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise AiguardProtocolError("tokenizer has no padding token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        raw_id2label = getattr(getattr(model, "config", None), "id2label", None)
        if not isinstance(raw_id2label, Mapping):
            raise AiguardProtocolError("model config must provide id2label")
        self.id2label = normalize_id2label(raw_id2label)
        source_types = {label.split("-", 1)[1] for label in self.id2label.values() if label != "O"}
        if not set(source_to_target) <= source_types:
            raise AiguardProtocolError("source mapping contains labels absent from checkpoint")
        allowed_tags = {
            "O",
            *(f"{prefix}-{source}" for source in source_to_target for prefix in ("B", "I", "E")),
        }
        if not allowed_tags <= set(self.id2label.values()):
            raise AiguardProtocolError("checkpoint lacks complete BIE tags for source mapping")
        self.allowed_label_ids = tuple(
            label_id for label_id, label in self.id2label.items() if label in allowed_tags
        )
        self.source_to_target = dict(source_to_target)
        self.protocol_id = protocol_id
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.device = torch.device(device)
        self.model.to(self.device)

    def _predict_micro_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.inference_mode():
            logits = self.model(**inputs).logits.float()
        allowed = torch.tensor(self.allowed_label_ids, device=logits.device, dtype=torch.long)
        probabilities = torch.softmax(logits.index_select(-1, allowed), dim=-1)
        scores, restricted_ids = probabilities.max(dim=-1)
        label_ids = allowed[restricted_ids]
        results: list[list[dict[str, Any]]] = []
        for row in range(len(texts)):
            row_offsets = [(int(pair[0]), int(pair[1])) for pair in offsets[row].tolist()]
            decoded = decode_bie_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                token_scores=scores[row].tolist(),
            )
            results.append(
                [
                    {
                        "entity_type": self.source_to_target[span.entity_type],
                        "start": span.start,
                        "end": span.end,
                        "score": span.score,
                        "offset_mode": "relative",
                        "metadata": {
                            "boundary_refined": False,
                            "protocol_id": self.protocol_id,
                        },
                    }
                    for span in decoded
                ]
            )
        return results

    def predict_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        results: list[list[dict[str, Any]]] = [[] for _ in values]
        non_empty = [(index, text) for index, text in enumerate(values) if text]
        for start in range(0, len(non_empty), self.micro_batch_size):
            batch = non_empty[start : start + self.micro_batch_size]
            predictions = self._predict_micro_batch([text for _, text in batch])
            for (index, _), prediction in zip(batch, predictions, strict=True):
                results[index] = prediction
        return results


def load_aiguard_mapped_predictor(
    model_path: str | Path,
    *,
    source_to_target: Mapping[str, str],
    protocol_id: str,
    device: str | torch.device,
    dtype: torch.dtype | None,
    micro_batch_size: int,
) -> AiguardMappedSpanPredictor:
    """Load the pinned local checkpoint without pickle or remote code."""

    path = Path(model_path).expanduser().resolve(strict=True)
    required = ("config.json", "model.safetensors", "tokenizer.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise AiguardProtocolError("model_path lacks required files")
    if any(
        any(path.glob(pattern))
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pth", "*.pt", "*.ckpt")
    ):
        raise AiguardProtocolError("unsafe serialized model files are forbidden")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "weights_only": True,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = Qwen3ForTokenClassification.from_pretrained(path, **kwargs)
    return AiguardMappedSpanPredictor(
        model,
        tokenizer,
        source_to_target=source_to_target,
        protocol_id=protocol_id,
        device=device,
        micro_batch_size=micro_batch_size,
    )


__all__ = ["AiguardMappedSpanPredictor", "load_aiguard_mapped_predictor"]
