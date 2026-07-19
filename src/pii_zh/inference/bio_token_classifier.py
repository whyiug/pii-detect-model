"""Protocol-restricted inference for generic BIO token classifiers.

The adapter deliberately restricts logits to ``O`` and explicitly mapped
``B-``/``I-`` labels *before* softmax, argmax, and span decoding.  This makes a
closed-label comparison auditable: unsupported checkpoint labels cannot be
decoded and discarded after the fact, while unsupported benchmark labels stay
absent and therefore count as false negatives in the evaluator.

Predictions contain only normalized labels, character offsets, scores, and
non-sensitive protocol metadata.  Input text and token strings are never
returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError("BIO token-classifier inference requires torch and transformers.") from exc

from pii_zh.models.decoding import decode_bio_ids


class BioTokenClassifierError(ValueError):
    """Raised when a checkpoint or prediction violates the BIO protocol."""


def normalize_bio_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    """Validate a contiguous ``id2label`` mapping containing BIO tags."""

    normalized: dict[int, str] = {}
    for raw_id, raw_label in value.items():
        if isinstance(raw_id, bool):
            raise BioTokenClassifierError("id2label keys must be non-negative integers")
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise BioTokenClassifierError("id2label contains a non-integer key") from exc
        if label_id < 0 or label_id in normalized:
            raise BioTokenClassifierError("id2label keys must be unique non-negative integers")
        if not isinstance(raw_label, str) or not raw_label:
            raise BioTokenClassifierError("id2label contains an invalid label")
        if raw_label != "O":
            prefix, separator, entity_type = raw_label.partition("-")
            if separator != "-" or prefix not in {"B", "I"} or not entity_type:
                raise BioTokenClassifierError("id2label must contain only O or B/I entity tags")
        normalized[label_id] = raw_label
    if not normalized or sorted(normalized) != list(range(len(normalized))):
        raise BioTokenClassifierError("id2label must be contiguous from zero")
    if sum(label == "O" for label in normalized.values()) != 1:
        raise BioTokenClassifierError("id2label must contain exactly one O label")
    return normalized


def _source_types(id2label: Mapping[int, str]) -> set[str]:
    return {label.split("-", 1)[1] for label in id2label.values() if label != "O"}


def closed_bio_label_ids(
    id2label: Mapping[int, str], source_to_target: Mapping[str, str]
) -> tuple[int, ...]:
    """Return label IDs admitted by an explicit source-to-target protocol."""

    normalized = normalize_bio_id2label(id2label)
    if not source_to_target or any(
        not isinstance(source, str) or not source or not isinstance(target, str) or not target
        for source, target in source_to_target.items()
    ):
        raise BioTokenClassifierError("source_to_target must contain non-empty string labels")
    present = set(normalized.values())
    required = {
        "O",
        *(f"{prefix}-{source}" for source in source_to_target for prefix in ("B", "I")),
    }
    missing = required - present
    if missing:
        raise BioTokenClassifierError(
            "checkpoint lacks required closed-protocol tags: " + ", ".join(sorted(missing))
        )
    return tuple(label_id for label_id, label in normalized.items() if label in required)


def excluded_bio_source_types(
    id2label: Mapping[int, str], source_to_target: Mapping[str, str]
) -> tuple[str, ...]:
    """List checkpoint entity types masked out of the closed protocol."""

    normalized = normalize_bio_id2label(id2label)
    return tuple(sorted(_source_types(normalized) - set(source_to_target)))


def _offset_row(value: object, *, text_length: int) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence):
        raise BioTokenClassifierError("tokenizer offset row must be a sequence")
    result: list[tuple[int, int]] = []
    for pair in value:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise BioTokenClassifierError("each tokenizer offset must be a pair")
        start, end = pair
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end > text_length
        ):
            raise BioTokenClassifierError("tokenizer returned an invalid character offset")
        result.append((start, end))
    return result


class ClosedBioTokenClassifierPredictor:
    """Decode a local BIO classifier under an explicit closed-label mapping."""

    attention_mode = "checkpoint_native_bidirectional"
    tag_scheme = "BIO"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        source_to_target: Mapping[str, str],
        protocol_id: str,
        device: str | torch.device | None = None,
        micro_batch_size: int = 16,
    ) -> None:
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int):
            raise TypeError("micro_batch_size must be an integer")
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise ValueError("protocol_id must be a non-empty string")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("exact character inference requires a fast tokenizer")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        config = getattr(model, "config", None)
        raw_id2label = getattr(config, "id2label", None)
        if not isinstance(raw_id2label, Mapping):
            raise ValueError("model config must provide id2label")
        self.id2label = normalize_bio_id2label(raw_id2label)
        self.allowed_label_ids = closed_bio_label_ids(self.id2label, source_to_target)
        self.excluded_source_types = excluded_bio_source_types(self.id2label, source_to_target)
        self.source_to_target = dict(source_to_target)
        self.protocol_id = protocol_id
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
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
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
        }
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits.float()
        if logits.ndim != 3 or logits.shape[-1] != len(self.id2label):
            raise BioTokenClassifierError("model logits do not match id2label")
        allowed = torch.tensor(self.allowed_label_ids, device=logits.device, dtype=torch.long)
        # The order is a protocol invariant: masking happens before softmax,
        # argmax, BIO repair, and span decoding.
        protocol_logits = logits.index_select(dim=-1, index=allowed)
        probabilities = torch.softmax(protocol_logits, dim=-1)
        token_scores, restricted_ids = probabilities.max(dim=-1)
        label_ids = allowed[restricted_ids]

        results: list[list[dict[str, Any]]] = []
        for row, text in enumerate(texts):
            raw_offsets = offsets[row].tolist()
            row_offsets = _offset_row(raw_offsets, text_length=len(text))
            decoded = decode_bio_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                token_scores=token_scores[row].tolist(),
                repair=True,
            )
            results.append(
                [
                    {
                        "entity_type": self.source_to_target[span.entity_type],
                        "start": span.start,
                        "end": span.end,
                        "score": float(span.score if span.score is not None else 1.0),
                        "offset_mode": "relative",
                        "metadata": {
                            "boundary_refined": False,
                            "protocol_id": self.protocol_id,
                            "tag_scheme": self.tag_scheme,
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

    def predict(self, text: str) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.predict_batch([text])[0]


def load_closed_bio_token_classifier(
    model_path: str | Path,
    *,
    source_to_target: Mapping[str, str],
    protocol_id: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    micro_batch_size: int = 16,
) -> ClosedBioTokenClassifierPredictor:
    """Safely load a local, safetensors-only Transformers BIO checkpoint."""

    path = Path(model_path).expanduser().resolve(strict=True)
    required = ("config.json", "model.safetensors", "tokenizer.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise BioTokenClassifierError("model_path lacks the required local checkpoint files")
    forbidden = sorted(
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pth", "*.pt", "*.ckpt")
        for item in path.glob(pattern)
    )
    if forbidden:
        raise BioTokenClassifierError(
            "unsafe serialized model files are forbidden: " + ", ".join(forbidden)
        )
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
    model = AutoModelForTokenClassification.from_pretrained(path, **kwargs)
    return ClosedBioTokenClassifierPredictor(
        model,
        tokenizer,
        source_to_target=source_to_target,
        protocol_id=protocol_id,
        device=device,
        micro_batch_size=micro_batch_size,
    )


__all__ = [
    "BioTokenClassifierError",
    "ClosedBioTokenClassifierPredictor",
    "closed_bio_label_ids",
    "excluded_bio_source_types",
    "load_closed_bio_token_classifier",
    "normalize_bio_id2label",
]
