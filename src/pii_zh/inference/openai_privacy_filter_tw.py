"""Full native-eight OpenAI Privacy Filter inference for tw-PII-bench."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .openai_privacy_filter import (
    EXPECTED_NATIVE_TOKEN_LABELS,
    Closed8ViterbiDecoder,
    OpenAIPrivacyFilterError,
    PredictedSpan,
    _load_official_helpers,
    _ordered_native_labels,
)

TW_NATIVE_SOURCE_LABELS: tuple[str, ...] = (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)


class OpenAIPrivacyFilterTwPredictor:
    """Run the official Viterbi decoder at its untuned default operating point."""

    attention_mode = "native_bidirectional_full_document"
    tag_scheme = "BIOES"

    def __init__(
        self,
        *,
        model_path: Path,
        official_source_root: Path,
        device: str,
        max_tokens: int,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise OpenAIPrivacyFilterError("max_tokens must be a positive integer")
        self.model_path = model_path.expanduser().resolve(strict=True)
        self.official_source_root = official_source_root.expanduser().resolve(strict=True)
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self._helpers = _load_official_helpers(self.official_source_root)
        config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise OpenAIPrivacyFilterError("model config must be an object")
        self.native_token_labels = _ordered_native_labels(config)
        self.allowed_token_label_ids = tuple(range(len(EXPECTED_NATIVE_TOKEN_LABELS)))
        self.label_info = self._helpers["build_label_info"](self.native_token_labels)
        native_decoder, biases = self._helpers["build_sequence_decoder"](
            decode_mode="viterbi",
            label_info=self.label_info,
            viterbi_calibration_path=None,
            checkpoint_dir=str(self.model_path),
        )
        if native_decoder is None or not isinstance(biases, Mapping):
            raise OpenAIPrivacyFilterError("official Viterbi decoder did not initialize")
        if any(float(value) != 0.0 for value in biases.values()):
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
            trust_remote_code=False,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise OpenAIPrivacyFilterError("fast tokenizer is required")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
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
        if any(not isinstance(text, str) for text in texts):
            raise OpenAIPrivacyFilterError("prediction inputs must be strings")
        if not texts:
            return []
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
            raise OpenAIPrivacyFilterError("input exceeds frozen no-truncation limit")
        logits = self.model(
            **{key: value.to(self.device) for key, value in encoded.items()}
        ).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        results: list[tuple[PredictedSpan, ...]] = []
        for row, (text, length) in enumerate(zip(texts, lengths, strict=True)):
            if length == 0:
                results.append(())
                continue
            decoded = self.decoder.decode(log_probs[row, :length])
            if len(decoded) != length:
                raise OpenAIPrivacyFilterError("official decoder output length changed")
            token_spans = self._helpers["labels_to_spans"](
                dict(enumerate(decoded)), self.label_info
            )
            row_offsets = offsets[row, :length].tolist()
            char_spans = self._helpers["token_spans_to_char_spans"](
                token_spans,
                [int(pair[0]) for pair in row_offsets],
                [int(pair[1]) for pair in row_offsets],
            )
            char_spans = self._helpers["trim_char_spans_whitespace"](char_spans, text)
            converted: dict[tuple[int, int, str], PredictedSpan] = {}
            for label_index, start, end in char_spans:
                label = str(self.label_info.span_class_names[int(label_index)])
                if label not in TW_NATIVE_SOURCE_LABELS:
                    raise OpenAIPrivacyFilterError("decoded span label is outside native eight")
                start_int = int(start)
                end_int = int(end)
                if not 0 <= start_int < end_int <= len(text):
                    raise OpenAIPrivacyFilterError("decoded character offset is invalid")
                converted[(start_int, end_int, label)] = PredictedSpan(start_int, end_int, label)
            results.append(
                tuple(
                    converted[key]
                    for key in sorted(converted, key=lambda value: (value[0], value[1], value[2]))
                )
            )
        return results


__all__ = ["OpenAIPrivacyFilterTwPredictor", "TW_NATIVE_SOURCE_LABELS"]
