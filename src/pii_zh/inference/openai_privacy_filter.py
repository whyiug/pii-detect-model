"""Frozen closed-eight adapter for the OpenAI Privacy Filter baseline.

The upstream checkpoint emits a 33-class BIOES taxonomy.  PII Bench ZH uses
eight different span labels.  This adapter freezes the semantic intersection,
masks out-of-protocol native logits before sequence decoding, and delegates the
actual path search and BIOES span assembly to the pinned upstream OPF source.

No score threshold is exposed here: the upstream ``default`` Viterbi operating
point is part of the model-raw artifact contract and must not be tuned on the
benchmark.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

OPENAI_PRIVACY_FILTER_MODEL_ID = "openai/privacy-filter"
OPENAI_PRIVACY_FILTER_MODEL_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
OPENAI_PRIVACY_FILTER_SOURCE_REVISION = "f7f00ca7fb869683eb732c010299d901457f19c3"

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

# Frozen before execution.  ``account_number`` is the upstream financial
# account identifier and is compared with the benchmark's only financial
# account-number category.  No other broad or cross-type mapping is allowed.
OPENAI_PRIVACY_FILTER_SOURCE_TO_TARGET: dict[str, str] = {
    "account_number": "BANK_CARD_NUMBER",
    "private_address": "ADDRESS",
    "private_email": "EMAIL_ADDRESS",
    "private_person": "PERSON_NAME",
    "private_phone": "PHONE_NUMBER",
}

OPENAI_PRIVACY_FILTER_PROTOCOL_EXCLUDED_SOURCE_TYPES: tuple[str, ...] = (
    "private_date",
    "private_url",
    "secret",
)

UNSUPPORTED_TARGET_LABELS: tuple[str, ...] = tuple(
    sorted(set(CLOSED_8_TARGET_LABELS) - set(OPENAI_PRIVACY_FILTER_SOURCE_TO_TARGET.values()))
)

EXPECTED_NATIVE_TOKEN_LABELS: tuple[str, ...] = (
    "O",
    "B-account_number",
    "I-account_number",
    "E-account_number",
    "S-account_number",
    "B-private_address",
    "I-private_address",
    "E-private_address",
    "S-private_address",
    "B-private_date",
    "I-private_date",
    "E-private_date",
    "S-private_date",
    "B-private_email",
    "I-private_email",
    "E-private_email",
    "S-private_email",
    "B-private_person",
    "I-private_person",
    "E-private_person",
    "S-private_person",
    "B-private_phone",
    "I-private_phone",
    "E-private_phone",
    "S-private_phone",
    "B-private_url",
    "I-private_url",
    "E-private_url",
    "S-private_url",
    "B-secret",
    "I-secret",
    "E-secret",
    "S-secret",
)

_MASK_SCORE = -1e9


class OpenAIPrivacyFilterError(RuntimeError):
    """Raised when the pinned model or decoder contract does not verify."""


@dataclass(frozen=True, slots=True)
class PredictedSpan:
    """One model-only, exact character-offset prediction."""

    start: int
    end: int
    label: str


def _ordered_native_labels(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("id2label")
    if not isinstance(raw, Mapping):
        raise OpenAIPrivacyFilterError("model config id2label must be an object")
    try:
        indexed = {int(key): str(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise OpenAIPrivacyFilterError("model config id2label indices are invalid") from exc
    expected_indices = tuple(range(len(EXPECTED_NATIVE_TOKEN_LABELS)))
    if tuple(sorted(indexed)) != expected_indices:
        raise OpenAIPrivacyFilterError("model config id2label indices are not contiguous")
    labels = tuple(indexed[index] for index in expected_indices)
    if labels != EXPECTED_NATIVE_TOKEN_LABELS:
        raise OpenAIPrivacyFilterError("model native BIOES label order changed")
    return labels


def protocol_allowed_token_label_ids(
    native_token_labels: Sequence[str] = EXPECTED_NATIVE_TOKEN_LABELS,
) -> tuple[int, ...]:
    """Return the frozen background plus five covered-source token classes."""

    allowed_source_types = set(OPENAI_PRIVACY_FILTER_SOURCE_TO_TARGET)
    allowed: list[int] = []
    for index, label in enumerate(native_token_labels):
        if label == "O":
            allowed.append(index)
            continue
        if "-" not in label:
            raise OpenAIPrivacyFilterError(f"invalid native token label at index {index}")
        _boundary, source_type = label.split("-", 1)
        if source_type in allowed_source_types:
            allowed.append(index)
    expected_count = 1 + 4 * len(allowed_source_types)
    if len(allowed) != expected_count:
        raise OpenAIPrivacyFilterError("covered source types do not have complete BIOES labels")
    return tuple(allowed)


def mask_non_protocol_scores(
    token_scores: torch.Tensor,
    *,
    allowed_token_label_ids: Sequence[int],
) -> torch.Tensor:
    """Mask non-protocol classes before the official Viterbi path search."""

    if token_scores.ndim < 2:
        raise OpenAIPrivacyFilterError("token scores must have at least two dimensions")
    if token_scores.shape[-1] != len(EXPECTED_NATIVE_TOKEN_LABELS):
        raise OpenAIPrivacyFilterError("token score class dimension changed")
    allowed = tuple(int(index) for index in allowed_token_label_ids)
    if not allowed or len(set(allowed)) != len(allowed):
        raise OpenAIPrivacyFilterError("allowed token labels must be unique and non-empty")
    if any(index < 0 or index >= token_scores.shape[-1] for index in allowed):
        raise OpenAIPrivacyFilterError("allowed token label index is out of range")
    if 0 not in allowed:
        raise OpenAIPrivacyFilterError("background token label must remain allowed")

    masked = token_scores.clone()
    excluded = sorted(set(range(token_scores.shape[-1])) - set(allowed))
    if excluded:
        masked[..., excluded] = _MASK_SCORE
    return masked


class Closed8ViterbiDecoder:
    """Pre-decode protocol mask around the official upstream decoder."""

    def __init__(self, native_decoder: Any, *, allowed_token_label_ids: Sequence[int]) -> None:
        self._native_decoder = native_decoder
        self.allowed_token_label_ids = tuple(int(value) for value in allowed_token_label_ids)

    def decode(self, token_scores: torch.Tensor) -> list[int]:
        masked = mask_non_protocol_scores(
            token_scores,
            allowed_token_label_ids=self.allowed_token_label_ids,
        )
        decoded = self._native_decoder.decode(masked)
        result = [int(value) for value in decoded]
        if any(value not in self.allowed_token_label_ids for value in result):
            raise OpenAIPrivacyFilterError("official decoder emitted a masked token label")
        return result


def _load_official_helpers(source_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve(strict=True)
    if not (source_root / "opf" / "_core" / "decoding.py").is_file():
        raise OpenAIPrivacyFilterError("official OPF source tree is incomplete")
    root_string = str(source_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        decoding = importlib.import_module("opf._core.decoding")
        sequence_labeling = importlib.import_module("opf._core.sequence_labeling")
        spans = importlib.import_module("opf._core.spans")
    except ImportError as exc:
        raise OpenAIPrivacyFilterError("official OPF decoder source is not importable") from exc
    return {
        "build_sequence_decoder": decoding.build_sequence_decoder,
        "build_label_info": sequence_labeling.build_label_info,
        "labels_to_spans": spans.labels_to_spans,
        "token_spans_to_char_spans": spans.token_spans_to_char_spans,
        "trim_char_spans_whitespace": spans.trim_char_spans_whitespace,
    }


class OpenAIPrivacyFilterClosed8Predictor:
    """Local-only Transformers forward pass with the official OPF decoder."""

    attention_mode = "native_bidirectional_sliding_window_128"
    tag_scheme = "BIOES"

    def __init__(
        self,
        *,
        model_path: Path,
        official_source_root: Path,
        device: str,
        max_tokens: int,
    ) -> None:
        if max_tokens < 1:
            raise OpenAIPrivacyFilterError("max_tokens must be positive")
        self.model_path = model_path.expanduser().resolve(strict=True)
        self.official_source_root = official_source_root.expanduser().resolve(strict=True)
        self.device = torch.device(device)
        self.max_tokens = int(max_tokens)
        self._helpers = _load_official_helpers(self.official_source_root)

        config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise OpenAIPrivacyFilterError("model config must be a JSON object")
        self.native_token_labels = _ordered_native_labels(config)
        self.allowed_token_label_ids = protocol_allowed_token_label_ids(self.native_token_labels)
        self.label_info = self._helpers["build_label_info"](self.native_token_labels)
        native_decoder, biases = self._helpers["build_sequence_decoder"](
            decode_mode="viterbi",
            label_info=self.label_info,
            viterbi_calibration_path=None,
            checkpoint_dir=str(self.model_path),
        )
        if native_decoder is None or not isinstance(biases, Mapping):
            raise OpenAIPrivacyFilterError("official Viterbi decoder did not initialize")
        if set(biases) != {
            "transition_bias_background_stay",
            "transition_bias_background_to_start",
            "transition_bias_inside_to_continue",
            "transition_bias_inside_to_end",
            "transition_bias_end_to_background",
            "transition_bias_end_to_start",
        } or any(float(value) != 0.0 for value in biases.values()):
            raise OpenAIPrivacyFilterError("default Viterbi operating point changed")
        self.viterbi_biases = {str(key): float(value) for key, value in biases.items()}
        self.decoder = Closed8ViterbiDecoder(
            native_decoder,
            allowed_token_label_ids=self.allowed_token_label_ids,
        )

        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise OpenAIPrivacyFilterError("Transformers runtime is unavailable") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        if not bool(getattr(self.tokenizer, "is_fast", False)):
            raise OpenAIPrivacyFilterError("offset-preserving fast tokenizer is required")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            use_safetensors=True,
            dtype=dtype,
        )
        configured = tuple(
            str(self.model.config.id2label[index]) for index in range(len(self.native_token_labels))
        )
        if configured != self.native_token_labels:
            raise OpenAIPrivacyFilterError("loaded model label order changed")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, texts: Sequence[str]) -> list[tuple[PredictedSpan, ...]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise OpenAIPrivacyFilterError("all prediction inputs must be strings")
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            raise OpenAIPrivacyFilterError("tokenizer did not return attention_mask")
        lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
        if any(length > self.max_tokens for length in lengths):
            raise OpenAIPrivacyFilterError("input exceeds frozen no-truncation token limit")
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**model_inputs).logits.float()
        protocol_logits = mask_non_protocol_scores(
            logits,
            allowed_token_label_ids=self.allowed_token_label_ids,
        )
        log_probs = torch.log_softmax(protocol_logits, dim=-1).cpu()

        results: list[tuple[PredictedSpan, ...]] = []
        for row, (text, length) in enumerate(zip(texts, lengths, strict=True)):
            if length == 0:
                results.append(())
                continue
            token_scores = log_probs[row, :length]
            decoded = self.decoder.decode(token_scores)
            if len(decoded) != length:
                raise OpenAIPrivacyFilterError("official decoder output length changed")
            predicted_by_index = {index: label for index, label in enumerate(decoded)}
            token_spans = self._helpers["labels_to_spans"](
                predicted_by_index,
                self.label_info,
            )
            row_offsets = offsets[row, :length].tolist()
            char_starts = [int(pair[0]) for pair in row_offsets]
            char_ends = [int(pair[1]) for pair in row_offsets]
            char_spans = self._helpers["token_spans_to_char_spans"](
                token_spans,
                char_starts,
                char_ends,
            )
            char_spans = self._helpers["trim_char_spans_whitespace"](char_spans, text)

            converted: list[PredictedSpan] = []
            seen: set[tuple[int, int, str]] = set()
            for label_index, start, end in char_spans:
                source_type = str(self.label_info.span_class_names[int(label_index)])
                target = OPENAI_PRIVACY_FILTER_SOURCE_TO_TARGET.get(source_type)
                if target is None:
                    raise OpenAIPrivacyFilterError("decoded span escaped the closed-eight mask")
                start_int = int(start)
                end_int = int(end)
                if not 0 <= start_int < end_int <= len(text):
                    raise OpenAIPrivacyFilterError("decoded character offset is invalid")
                key = (start_int, end_int, target)
                if key not in seen:
                    converted.append(PredictedSpan(start=start_int, end=end_int, label=target))
                    seen.add(key)
            results.append(
                tuple(sorted(converted, key=lambda span: (span.start, span.end, span.label)))
            )
        return results


__all__ = [
    "CLOSED_8_TARGET_LABELS",
    "EXPECTED_NATIVE_TOKEN_LABELS",
    "OPENAI_PRIVACY_FILTER_MODEL_ID",
    "OPENAI_PRIVACY_FILTER_MODEL_REVISION",
    "OPENAI_PRIVACY_FILTER_PROTOCOL_EXCLUDED_SOURCE_TYPES",
    "OPENAI_PRIVACY_FILTER_SOURCE_REVISION",
    "OPENAI_PRIVACY_FILTER_SOURCE_TO_TARGET",
    "OpenAIPrivacyFilterClosed8Predictor",
    "OpenAIPrivacyFilterError",
    "PredictedSpan",
    "UNSUPPORTED_TARGET_LABELS",
    "mask_non_protocol_scores",
    "protocol_allowed_token_label_ids",
]
