"""Protocol-restricted inference for the public AIguard Chinese PII baseline.

The upstream checkpoint uses ``O`` plus ``B/I/E`` tags for 21 source entity
types.  PII Bench ZH contains a closed set of eight target labels.  This
adapter restricts the argmax *before decoding* to ``O`` and the source tags
that have an explicit mapping into that closed protocol.  Predictions from
unrelated upstream labels therefore cannot leak into the comparison as
unmapped false positives.

Returned spans contain offsets, a normalized label, and a score only.  Raw
text and token strings are intentionally absent from the public prediction
interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoTokenizer, Qwen3ForTokenClassification
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError("AIguard inference requires torch and transformers.") from exc


AIGUARD_CLOSED_8_LABEL_MAPPING: dict[str, str] = {
    "address": "ADDRESS",
    "bank_card": "BANK_CARD_NUMBER",
    "credit_card": "BANK_CARD_NUMBER",
    "email": "EMAIL_ADDRESS",
    "id_card": "CN_RESIDENT_ID",
    "mobile": "PHONE_NUMBER",
    "name": "PERSON_NAME",
    "passport": "PASSPORT_NUMBER",
    "plate_number": "VEHICLE_LICENSE_PLATE",
}

CLOSED_8_TARGET_LABELS: tuple[str, ...] = (
    "ADDRESS",
    "BANK_CARD_NUMBER",
    "CN_RESIDENT_ID",
    "EMAIL_ADDRESS",
    "PASSPORT_NUMBER",
    "PERSON_NAME",
    "PHONE_NUMBER",
    "VEHICLE_LICENSE_PLATE",
)

_BIE_PREFIXES = ("B", "I", "E")


class AiguardProtocolError(ValueError):
    """Raised when a checkpoint cannot satisfy the frozen comparison protocol."""


@dataclass(frozen=True, slots=True)
class AiguardDecodedSpan:
    """A raw-text-free left-closed/right-open character span."""

    entity_type: str
    start: int
    end: int
    score: float


def normalize_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    """Validate and normalize a Transformers ``id2label`` mapping."""

    normalized: dict[int, str] = {}
    for raw_id, raw_label in value.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise AiguardProtocolError("id2label contains a non-integer key") from exc
        if isinstance(raw_id, bool) or label_id < 0:
            raise AiguardProtocolError("id2label keys must be non-negative integers")
        if not isinstance(raw_label, str) or not raw_label:
            raise AiguardProtocolError("id2label contains an invalid label")
        if label_id in normalized:
            raise AiguardProtocolError("id2label contains duplicate integer keys")
        normalized[label_id] = raw_label
    if not normalized or sorted(normalized) != list(range(len(normalized))):
        raise AiguardProtocolError("id2label must be contiguous from zero")
    if sum(label == "O" for label in normalized.values()) != 1:
        raise AiguardProtocolError("id2label must contain exactly one O label")
    return normalized


def closed_protocol_label_ids(
    id2label: Mapping[int, str],
    source_to_target: Mapping[str, str] = AIGUARD_CLOSED_8_LABEL_MAPPING,
) -> tuple[int, ...]:
    """Return checkpoint label IDs admitted by the closed-eight protocol."""

    normalized = normalize_id2label(id2label)
    if not source_to_target:
        raise AiguardProtocolError("source_to_target must not be empty")
    if set(source_to_target.values()) != set(CLOSED_8_TARGET_LABELS):
        raise AiguardProtocolError("source_to_target must cover exactly the closed-eight labels")
    allowed_tags = {
        "O",
        *(
            f"{prefix}-{source}"
            for source in source_to_target
            for prefix in _BIE_PREFIXES
        ),
    }
    present = set(normalized.values())
    missing = allowed_tags - present
    if missing:
        raise AiguardProtocolError(
            "checkpoint lacks required closed-protocol tags: " + ", ".join(sorted(missing))
        )
    return tuple(label_id for label_id, label in normalized.items() if label in allowed_tags)


def excluded_source_entity_types(
    id2label: Mapping[int, str],
    source_to_target: Mapping[str, str] = AIGUARD_CLOSED_8_LABEL_MAPPING,
) -> tuple[str, ...]:
    """List upstream entity types deliberately masked from closed-eight argmax."""

    normalized = normalize_id2label(id2label)
    source_types = {
        label.split("-", 1)[1]
        for label in normalized.values()
        if label != "O" and label[:2] in {"B-", "I-", "E-"}
    }
    return tuple(sorted(source_types - set(source_to_target)))


def _parse_bie_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    prefix, separator, entity_type = tag.partition("-")
    if separator != "-" or prefix not in _BIE_PREFIXES or not entity_type:
        raise AiguardProtocolError(
            f"invalid BIE tag {tag!r}; expected O or B/I/E-<source_type>"
        )
    return prefix, entity_type


def decode_bie_tags(
    tags: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    *,
    token_scores: Sequence[float] | None = None,
) -> list[AiguardDecodedSpan]:
    """Decode AIguard B/I/E tags using a conservative malformed-tag policy.

    ``B`` opens a span and ``E`` closes it.  A span that has not received an
    ``E`` is closed by the next ``B``/``O`` or end of sequence, matching the
    checkpoint's published reference decoder.  Orphan or type-mismatched
    ``I``/``E`` tags are dropped; they never bridge a gap or invent a span.
    Zero-width special/padding offsets behave as ``O``.
    """

    if len(tags) != len(offsets):
        raise AiguardProtocolError("tags and offsets must have equal length")
    if token_scores is not None and len(token_scores) != len(tags):
        raise AiguardProtocolError("token_scores and tags must have equal length")

    spans: list[AiguardDecodedSpan] = []
    active_type: str | None = None
    active_start = 0
    active_end = 0
    active_scores: list[float] = []

    def close_active() -> None:
        nonlocal active_type, active_scores
        if active_type is None:
            return
        score = min(active_scores) if active_scores else 1.0
        spans.append(
            AiguardDecodedSpan(
                entity_type=active_type,
                start=active_start,
                end=active_end,
                score=score,
            )
        )
        active_type = None
        active_scores = []

    for index, (tag, offset) in enumerate(zip(tags, offsets, strict=True)):
        if not isinstance(tag, str):
            raise AiguardProtocolError("BIE tags must be strings")
        if not isinstance(offset, Sequence) or len(offset) != 2:
            raise AiguardProtocolError("each tokenizer offset must be a pair")
        start, end = offset
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise AiguardProtocolError("tokenizer offsets must be valid non-negative integers")
        score = 1.0
        if token_scores is not None:
            raw_score = token_scores[index]
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise AiguardProtocolError("token scores must be numeric")
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise AiguardProtocolError("token scores must be finite probabilities")

        prefix, entity_type = _parse_bie_tag("O" if start == end else tag)
        if prefix == "O":
            close_active()
            continue
        if prefix == "B":
            close_active()
            assert entity_type is not None
            active_type = entity_type
            active_start = start
            active_end = end
            active_scores = [score]
            continue
        if active_type != entity_type:
            # Conservative repair: do not let an invalid continuation make the
            # preceding span jump across an unrelated token.
            close_active()
            continue
        active_end = end
        active_scores.append(score)
        if prefix == "E":
            close_active()
    close_active()
    return spans


def decode_bie_ids(
    label_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    id2label: Mapping[int, str],
    *,
    token_scores: Sequence[float] | None = None,
) -> list[AiguardDecodedSpan]:
    """Map integer predictions through ``id2label`` and decode B/I/E spans."""

    normalized = normalize_id2label(id2label)
    try:
        tags = [normalized[int(label_id)] for label_id in label_ids]
    except (KeyError, TypeError, ValueError) as exc:
        raise AiguardProtocolError("predicted label id is absent from id2label") from exc
    return decode_bie_tags(tags, offsets, token_scores=token_scores)


class AiguardClosed8SpanPredictor:
    """Transformers window predictor for the frozen PII Bench closed-eight track."""

    protocol_id = "pii_bench_zh_closed_8_v1"
    attention_mode = "checkpoint_native_causal"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        device: str | torch.device | None = None,
        micro_batch_size: int = 16,
        source_to_target: Mapping[str, str] = AIGUARD_CLOSED_8_LABEL_MAPPING,
    ) -> None:
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int):
            raise TypeError("micro_batch_size must be an integer")
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
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
        self.id2label = normalize_id2label(raw_id2label)
        self.allowed_label_ids = closed_protocol_label_ids(self.id2label, source_to_target)
        self.excluded_source_types = excluded_source_entity_types(
            self.id2label, source_to_target
        )
        self.source_to_target = dict(source_to_target)
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
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits.float()
        allowed = torch.tensor(self.allowed_label_ids, device=logits.device, dtype=torch.long)
        # Restrict before softmax/argmax.  This is materially different from
        # predicting all 21 source types and discarding them after decoding.
        protocol_logits = logits.index_select(dim=-1, index=allowed)
        probabilities = torch.softmax(protocol_logits, dim=-1)
        token_scores, restricted_ids = probabilities.max(dim=-1)
        label_ids = allowed[restricted_ids]

        results: list[list[dict[str, Any]]] = []
        for row in range(len(texts)):
            row_offsets: list[tuple[int, int]] = []
            for pair in offsets[row].tolist():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise AiguardProtocolError("tokenizer returned an invalid offset mapping")
                row_offsets.append((int(pair[0]), int(pair[1])))
            decoded = decode_bie_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                token_scores=token_scores[row].tolist(),
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

    def predict(self, text: str) -> list[dict[str, Any]]:
        return self.predict_batch([text])[0]


def load_aiguard_closed8_predictor(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    micro_batch_size: int = 16,
) -> AiguardClosed8SpanPredictor:
    """Safely load a local, safetensors-only AIguard checkpoint."""

    path = Path(model_path).expanduser().resolve(strict=True)
    required = ("config.json", "model.safetensors", "tokenizer.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise AiguardProtocolError("model_path lacks the required local checkpoint files")
    forbidden = [
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pth", "*.pt")
        for item in path.glob(pattern)
    ]
    if forbidden:
        raise AiguardProtocolError(
            "unsafe serialized model files are forbidden: " + ", ".join(sorted(forbidden))
        )
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
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
    return AiguardClosed8SpanPredictor(
        model,
        tokenizer,
        device=device,
        micro_batch_size=micro_batch_size,
    )


__all__ = [
    "AIGUARD_CLOSED_8_LABEL_MAPPING",
    "CLOSED_8_TARGET_LABELS",
    "AiguardClosed8SpanPredictor",
    "AiguardDecodedSpan",
    "AiguardProtocolError",
    "closed_protocol_label_ids",
    "decode_bie_ids",
    "decode_bie_tags",
    "excluded_source_entity_types",
    "load_aiguard_closed8_predictor",
    "normalize_id2label",
]
