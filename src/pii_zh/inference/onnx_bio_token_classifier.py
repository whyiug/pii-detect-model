"""Closed-protocol inference for local ONNX BIO token classifiers.

The adapter is intentionally small and artifact-driven.  It accepts a standard
token-classification ONNX graph plus a fast Hugging Face tokenizer, masks every
non-protocol logit *before* softmax/argmax, and emits only normalized labels,
character offsets, scores, and non-sensitive protocol metadata.

``onnxruntime`` remains a lazy optional dependency so the core package and unit
tests do not require a platform-specific runtime wheel.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from transformers import AutoTokenizer

from pii_zh.models.decoding import decode_bio_ids

from .bio_token_classifier import (
    BioTokenClassifierError,
    closed_bio_label_ids,
    excluded_bio_source_types,
    normalize_bio_id2label,
)


class _OnnxValueInfo(Protocol):
    name: str
    shape: Sequence[Any]


class _OnnxSession(Protocol):
    def get_inputs(self) -> Sequence[_OnnxValueInfo]: ...

    def get_outputs(self) -> Sequence[_OnnxValueInfo]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, np.ndarray],
    ) -> Sequence[np.ndarray]: ...


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


def _fixed_batch_size(session: _OnnxSession) -> int | None:
    outputs = list(session.get_outputs())
    if len(outputs) != 1 or outputs[0].name != "logits":
        raise BioTokenClassifierError("ONNX graph must expose exactly one logits output")
    shape = list(outputs[0].shape)
    if len(shape) != 3:
        raise BioTokenClassifierError("ONNX logits output must have rank three")
    batch = shape[0]
    if isinstance(batch, bool):
        raise BioTokenClassifierError("ONNX logits batch dimension is invalid")
    if isinstance(batch, int):
        if batch < 1:
            raise BioTokenClassifierError("ONNX logits batch dimension must be positive")
        return batch
    return None


class ClosedOnnxBioTokenClassifierPredictor:
    """Decode a standard ONNX BIO classifier under a closed label protocol."""

    attention_mode = "checkpoint_native_bidirectional"
    tag_scheme = "BIO"

    def __init__(
        self,
        session: _OnnxSession,
        tokenizer: Any,
        *,
        id2label: Mapping[Any, Any],
        source_to_target: Mapping[str, str],
        protocol_id: str,
        micro_batch_size: int = 1,
        runtime_version: str = "unknown",
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

        self.id2label = normalize_bio_id2label(id2label)
        self.allowed_label_ids = closed_bio_label_ids(self.id2label, source_to_target)
        self.excluded_source_types = excluded_bio_source_types(self.id2label, source_to_target)
        self.source_to_target = dict(source_to_target)
        self.protocol_id = protocol_id
        self.session = session
        self.tokenizer = tokenizer
        self.runtime_version = runtime_version
        self.provider = str(session.get_providers()[0])
        self.input_names = {item.name for item in session.get_inputs()}
        if "input_ids" not in self.input_names:
            raise BioTokenClassifierError("ONNX graph lacks input_ids")
        self.fixed_batch_size = _fixed_batch_size(session)
        if self.fixed_batch_size is not None:
            self.micro_batch_size = self.fixed_batch_size
        else:
            self.micro_batch_size = micro_batch_size

    @staticmethod
    def _protocol_argmax(
        logits: np.ndarray,
        *,
        allowed_label_ids: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mask before softmax and return original label IDs plus probabilities."""

        allowed = np.asarray(allowed_label_ids, dtype=np.int64)
        protocol_logits = np.take(logits.astype(np.float64, copy=False), allowed, axis=-1)
        protocol_logits -= protocol_logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(protocol_logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        restricted_ids = probabilities.argmax(axis=-1)
        scores = np.take_along_axis(
            probabilities,
            restricted_ids[..., np.newaxis],
            axis=-1,
        )[..., 0]
        return allowed[restricted_ids], scores

    def _predict_micro_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        raw_offsets = np.asarray(encoded.pop("offset_mapping"))
        feed = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in encoded.items()
            if key in self.input_names
        }
        raw_outputs = self.session.run(["logits"], feed)
        if len(raw_outputs) != 1:
            raise BioTokenClassifierError("ONNX runtime returned an unexpected output count")
        logits = np.asarray(raw_outputs[0])
        expected_shape = (len(texts), raw_offsets.shape[1], len(self.id2label))
        if logits.shape != expected_shape:
            raise BioTokenClassifierError(
                f"ONNX logits shape {logits.shape!r} does not match {expected_shape!r}"
            )
        label_ids, token_scores = self._protocol_argmax(
            logits,
            allowed_label_ids=self.allowed_label_ids,
        )

        results: list[list[dict[str, Any]]] = []
        for row, text in enumerate(texts):
            offsets = _offset_row(raw_offsets[row].tolist(), text_length=len(text))
            decoded = decode_bio_ids(
                label_ids[row].tolist(),
                offsets,
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


def load_closed_onnx_bio_token_classifier(
    model_path: str | Path,
    *,
    source_to_target: Mapping[str, str],
    protocol_id: str,
    device: str = "cpu",
    micro_batch_size: int = 1,
) -> ClosedOnnxBioTokenClassifierPredictor:
    """Load a local ONNX BIO checkpoint without remote or executable model code."""

    path = Path(model_path).expanduser().resolve(strict=True)
    required = (
        "config.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise BioTokenClassifierError("model_path lacks required local ONNX files")
    forbidden = sorted(
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pth", "*.pt", "*.ckpt")
        for item in path.glob(pattern)
    )
    if forbidden:
        raise BioTokenClassifierError(
            "unsafe serialized model files are forbidden: " + ", ".join(forbidden)
        )
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BioTokenClassifierError("config.json is not readable JSON") from exc
    raw_id2label = config.get("id2label") if isinstance(config, Mapping) else None
    if not isinstance(raw_id2label, Mapping):
        raise BioTokenClassifierError("config.json must provide id2label")

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - platform-specific optional dependency
        raise ImportError("ONNX inference requires onnxruntime or onnxruntime-gpu") from exc

    if device in {"cuda", "cuda:0"}:
        # Importing torch preloads the CUDA/cuDNN libraries bundled with the
        # caller-selected compatible environment before ONNX Runtime starts.
        import torch

        if not torch.cuda.is_available():
            raise BioTokenClassifierError("CUDA was requested but is unavailable")
        providers: list[Any] = [
            ("CUDAExecutionProvider", {"device_id": "0"}),
            "CPUExecutionProvider",
        ]
        required_provider = "CUDAExecutionProvider"
    elif device == "cpu":
        providers = ["CPUExecutionProvider"]
        required_provider = "CPUExecutionProvider"
    else:
        raise ValueError("device must be cpu, cuda, or cuda:0")

    session = ort.InferenceSession(str(path / "model.onnx"), providers=providers)
    if list(session.get_providers())[0] != required_provider:
        raise BioTokenClassifierError(f"failed to initialize {required_provider}")
    session.disable_fallback()
    return ClosedOnnxBioTokenClassifierPredictor(
        session,
        tokenizer,
        id2label=raw_id2label,
        source_to_target=source_to_target,
        protocol_id=protocol_id,
        micro_batch_size=micro_batch_size,
        runtime_version=str(ort.__version__),
    )


__all__ = [
    "ClosedOnnxBioTokenClassifierPredictor",
    "load_closed_onnx_bio_token_classifier",
]
