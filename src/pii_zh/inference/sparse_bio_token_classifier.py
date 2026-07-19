"""Closed-protocol inference for sparse BIO token classifiers.

Some token-classification checkpoints expose ``B`` for every source type but
``I`` for only a subset.  Requiring artificial B/I pairs changes the artifact
contract and can make such a checkpoint impossible to evaluate.  This adapter
instead masks logits to ``O`` plus explicitly mapped source tags before
softmax/argmax, maps each predicted token to the target ontology, and only
then performs a frozen character-offset merge.

Ordinary entities merge across adjacent or overlapping predicted token offsets
when they map to the same target.  Overlapping ranges are unioned; overlapping
different targets remain independent spans.  A preregistered subset of
structured targets may additionally bridge a gap that fully consists of
whitespace or ASCII hyphens.  Output starts and ends remain predicted token
boundaries; arbitrary unpredicted text is never added.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError(
        "sparse BIO token-classifier inference requires torch and transformers"
    ) from exc


class SparseBioTokenClassifierError(ValueError):
    """Raised when a checkpoint or prediction violates the frozen protocol."""


_CONNECTOR_GAP = re.compile(r"[\s-]+\Z")


def normalize_sparse_bio_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    """Validate contiguous ``O`` and sparse ``B/I-<TYPE>`` tags."""

    normalized: dict[int, str] = {}
    for raw_id, raw_label in value.items():
        if isinstance(raw_id, bool):
            raise SparseBioTokenClassifierError("id2label keys must be non-negative integers")
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise SparseBioTokenClassifierError("id2label contains a non-integer key") from exc
        if label_id < 0 or label_id in normalized:
            raise SparseBioTokenClassifierError(
                "id2label keys must be unique non-negative integers"
            )
        if not isinstance(raw_label, str) or not raw_label:
            raise SparseBioTokenClassifierError("id2label contains an invalid label")
        if raw_label != "O":
            prefix, separator, source_type = raw_label.partition("-")
            if separator != "-" or prefix not in {"B", "I"} or not source_type:
                raise SparseBioTokenClassifierError(
                    "id2label must contain only O or B/I source tags"
                )
        normalized[label_id] = raw_label
    if not normalized or sorted(normalized) != list(range(len(normalized))):
        raise SparseBioTokenClassifierError("id2label must be contiguous from zero")
    if sum(label == "O" for label in normalized.values()) != 1:
        raise SparseBioTokenClassifierError("id2label must contain exactly one O label")
    return normalized


def _source_type(tag: str) -> str | None:
    return None if tag == "O" else tag.split("-", 1)[1]


def _validate_mapping(source_to_target: Mapping[str, str]) -> dict[str, str]:
    if not source_to_target or any(
        not isinstance(source, str) or not source or not isinstance(target, str) or not target
        for source, target in source_to_target.items()
    ):
        raise SparseBioTokenClassifierError("source_to_target must contain non-empty string labels")
    return dict(source_to_target)


def closed_sparse_bio_label_ids(
    id2label: Mapping[int, str], source_to_target: Mapping[str, str]
) -> tuple[int, ...]:
    """Return ``O`` and every present B/I tag for mapped source types.

    Each mapped source type must exist in at least one prefix form, but no
    artificial B/I pair is required.
    """

    normalized = normalize_sparse_bio_id2label(id2label)
    mapping = _validate_mapping(source_to_target)
    present_sources = {
        source for tag in normalized.values() if (source := _source_type(tag)) is not None
    }
    missing = set(mapping) - present_sources
    if missing:
        raise SparseBioTokenClassifierError(
            "checkpoint lacks a tag for one or more mapped source types"
        )
    return tuple(
        label_id
        for label_id, tag in normalized.items()
        if tag == "O" or _source_type(tag) in mapping
    )


def excluded_sparse_bio_source_types(
    id2label: Mapping[int, str], source_to_target: Mapping[str, str]
) -> tuple[str, ...]:
    """List checkpoint source types removed before softmax and argmax."""

    normalized = normalize_sparse_bio_id2label(id2label)
    mapping = _validate_mapping(source_to_target)
    present = {source for tag in normalized.values() if (source := _source_type(tag)) is not None}
    return tuple(sorted(present - set(mapping)))


def _offset_row(value: object, *, text_length: int) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence):
        raise SparseBioTokenClassifierError("tokenizer offset row must be a sequence")
    result: list[tuple[int, int]] = []
    prior_start = 0
    for pair in value:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise SparseBioTokenClassifierError("each tokenizer offset must be a pair")
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
            raise SparseBioTokenClassifierError("tokenizer returned an invalid character offset")
        if start != end and start < prior_start:
            raise SparseBioTokenClassifierError(
                "tokenizer returned non-monotonic character offsets"
            )
        if start != end:
            prior_start = start
        result.append((start, end))
    return result


@dataclass(frozen=True, slots=True)
class SparseDecodedSpan:
    entity_type: str
    start: int
    end: int
    score: float
    token_count: int
    bridged_connector_count: int


@dataclass(frozen=True, slots=True)
class _MappedToken:
    target: str
    start: int
    end: int
    score: float


def decode_sparse_bio_ids(
    label_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    id2label: Mapping[int, str],
    *,
    source_to_target: Mapping[str, str],
    structured_targets: Sequence[str],
    text: str,
    token_scores: Sequence[float],
) -> list[SparseDecodedSpan]:
    """Map predicted tokens to targets and apply the frozen merge policy."""

    if len(label_ids) != len(offsets) or len(token_scores) != len(label_ids):
        raise SparseBioTokenClassifierError("label IDs, offsets, and scores must have equal length")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_offsets = _offset_row(offsets, text_length=len(text))
    normalized = normalize_sparse_bio_id2label(id2label)
    mapping = _validate_mapping(source_to_target)
    structured = frozenset(structured_targets)
    if any(not isinstance(target, str) or not target for target in structured):
        raise SparseBioTokenClassifierError("structured_targets must contain non-empty strings")
    if not structured <= set(mapping.values()):
        raise SparseBioTokenClassifierError(
            "structured_targets must be a subset of mapped target labels"
        )

    tokens: list[_MappedToken] = []
    for raw_id, (start, end), raw_score in zip(
        label_ids, normalized_offsets, token_scores, strict=True
    ):
        try:
            tag = normalized[int(raw_id)]
        except (KeyError, ValueError, TypeError) as exc:
            raise SparseBioTokenClassifierError(
                "predicted label ID is absent from id2label"
            ) from exc
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise SparseBioTokenClassifierError(
                "token score must be finite and between zero and one"
            )
        if start == end or tag == "O":
            continue
        source = _source_type(tag)
        if source not in mapping:
            raise SparseBioTokenClassifierError(
                "a non-protocol source type survived the logit mask"
            )
        tokens.append(_MappedToken(mapping[source], start, end, score))

    spans: list[SparseDecodedSpan] = []
    active_target: str | None = None
    active_start = 0
    active_end = 0
    active_scores: list[float] = []
    active_token_count = 0
    active_bridge_count = 0

    def close_active() -> None:
        nonlocal active_target, active_scores, active_token_count, active_bridge_count
        if active_target is None:
            return
        spans.append(
            SparseDecodedSpan(
                entity_type=active_target,
                start=active_start,
                end=active_end,
                score=min(active_scores),
                token_count=active_token_count,
                bridged_connector_count=active_bridge_count,
            )
        )
        active_target = None
        active_scores = []
        active_token_count = 0
        active_bridge_count = 0

    for token in tokens:
        adjacent = active_target == token.target and token.start == active_end
        overlap = active_target == token.target and token.start < active_end
        connector_bridge = False
        if (
            active_target == token.target
            and token.target in structured
            and token.start > active_end
        ):
            gap = text[active_end : token.start]
            connector_bridge = _CONNECTOR_GAP.fullmatch(gap) is not None
        if active_target is None or (not adjacent and not overlap and not connector_bridge):
            close_active()
            active_target = token.target
            active_start = token.start
            active_end = token.end
            active_scores = [token.score]
            active_token_count = 1
            continue
        active_end = max(active_end, token.end)
        active_scores.append(token.score)
        active_token_count += 1
        active_bridge_count += int(connector_bridge)
    close_active()
    return spans


class ClosedSparseBioTokenClassifierPredictor:
    """Decode a local sparse-BIO classifier under a closed target protocol."""

    attention_mode = "checkpoint_native_bidirectional"
    tag_scheme = "SPARSE_BIO_TOKEN_MAPPED"
    merge_policy = (
        "adjacent_or_overlapping_same_target_union_with_preregistered_structured_connectors"
    )

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        source_to_target: Mapping[str, str],
        structured_targets: Sequence[str],
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
        self.id2label = normalize_sparse_bio_id2label(raw_id2label)
        self.source_to_target = _validate_mapping(source_to_target)
        self.allowed_label_ids = closed_sparse_bio_label_ids(self.id2label, self.source_to_target)
        self.excluded_source_types = excluded_sparse_bio_source_types(
            self.id2label, self.source_to_target
        )
        self.structured_targets = tuple(sorted(set(structured_targets)))
        if not set(self.structured_targets) <= set(self.source_to_target.values()):
            raise SparseBioTokenClassifierError(
                "structured_targets must be mapped protocol targets"
            )
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

    @staticmethod
    def protocol_argmax(
        logits: torch.Tensor,
        *,
        allowed_label_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask before softmax/argmax and return original IDs and probabilities."""

        if logits.ndim != 3:
            raise SparseBioTokenClassifierError("model logits must have rank three")
        if not allowed_label_ids:
            raise SparseBioTokenClassifierError("allowed label IDs cannot be empty")
        allowed = torch.tensor(tuple(allowed_label_ids), device=logits.device, dtype=torch.long)
        if int(allowed.min()) < 0 or int(allowed.max()) >= logits.shape[-1]:
            raise SparseBioTokenClassifierError("allowed label ID lies outside logits")
        protocol_logits = logits.index_select(dim=-1, index=allowed).float()
        probabilities = torch.softmax(protocol_logits, dim=-1)
        token_scores, restricted_ids = probabilities.max(dim=-1)
        return allowed[restricted_ids], token_scores

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
            logits = self.model(**model_inputs).logits
        if logits.ndim != 3 or logits.shape[-1] != len(self.id2label):
            raise SparseBioTokenClassifierError("model logits do not match id2label")
        label_ids, token_scores = self.protocol_argmax(
            logits,
            allowed_label_ids=self.allowed_label_ids,
        )

        results: list[list[dict[str, Any]]] = []
        for row, text in enumerate(texts):
            row_offsets = _offset_row(offsets[row].tolist(), text_length=len(text))
            decoded = decode_sparse_bio_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                source_to_target=self.source_to_target,
                structured_targets=self.structured_targets,
                text=text,
                token_scores=token_scores[row].tolist(),
            )
            results.append(
                [
                    {
                        "entity_type": span.entity_type,
                        "start": span.start,
                        "end": span.end,
                        "score": span.score,
                        "offset_mode": "relative",
                        "metadata": {
                            "boundary_refined": False,
                            "protocol_id": self.protocol_id,
                            "tag_scheme": self.tag_scheme,
                            "merge_policy": self.merge_policy,
                            "token_count": span.token_count,
                            "bridged_connector_count": span.bridged_connector_count,
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


def load_closed_sparse_bio_token_classifier(
    model_path: str | Path,
    *,
    source_to_target: Mapping[str, str],
    structured_targets: Sequence[str],
    protocol_id: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    micro_batch_size: int = 16,
) -> ClosedSparseBioTokenClassifierPredictor:
    """Load a local safetensors-only sparse-BIO Transformers checkpoint."""

    path = Path(model_path).expanduser().resolve(strict=True)
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise SparseBioTokenClassifierError("model_path lacks required local checkpoint files")
    forbidden = sorted(
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pth", "*.pt", "*.ckpt")
        for item in path.glob(pattern)
    )
    if forbidden:
        raise SparseBioTokenClassifierError("unsafe serialized model files are forbidden")
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
    return ClosedSparseBioTokenClassifierPredictor(
        model,
        tokenizer,
        source_to_target=source_to_target,
        structured_targets=structured_targets,
        protocol_id=protocol_id,
        device=device,
        micro_batch_size=micro_batch_size,
    )


__all__ = [
    "ClosedSparseBioTokenClassifierPredictor",
    "SparseBioTokenClassifierError",
    "SparseDecodedSpan",
    "closed_sparse_bio_label_ids",
    "decode_sparse_bio_ids",
    "excluded_sparse_bio_source_types",
    "load_closed_sparse_bio_token_classifier",
    "normalize_sparse_bio_id2label",
]
