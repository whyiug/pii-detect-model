"""Transformers adapter which emits raw-text-free character spans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoTokenizer, Qwen3ForTokenClassification
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError("Local inference requires torch and transformers.") from exc

from pii_zh.models.decoding import decode_bio_ids
from pii_zh.models.qwen3_bi import (
    ARCHITECTURE_VERSION,
    Qwen3BiForTokenClassification,
)
from pii_zh.models.qwen3_jpt import build_jpt_inputs
from pii_zh.tokenization import assert_character_boundary_tokenizer

_BOUND_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _verify_training_artifact_binding(path: Path) -> None:
    manifest_path = path / "training_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InferenceSafetyError("completed training_manifest.json is required") from exc
    if not isinstance(manifest, dict):
        raise InferenceSafetyError("training manifest must be a JSON object")
    expected_manifest_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(expected_manifest_hash, str)
        or _canonical_hash(unsigned) != expected_manifest_hash
        or manifest.get("status") != "completed"
    ):
        raise InferenceSafetyError("training manifest completion/hash check failed")
    recorded = manifest.get("output_artifact")
    if not isinstance(recorded, dict):
        raise InferenceSafetyError("training manifest lacks output artifact binding")
    weights = tuple(sorted(path.glob("model*.safetensors")))
    files = {item.name: _sha256_file(item) for item in weights if item.is_file()}
    for name in _BOUND_METADATA_FILES:
        candidate = path / name
        if candidate.is_file() and not candidate.is_symlink():
            files[name] = _sha256_file(candidate)
    weight_hashes = {name: files[name] for name in sorted(files) if name.endswith(".safetensors")}
    actual = {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": _canonical_hash(weight_hashes),
        "artifact_files_combined_sha256": _canonical_hash(files),
    }
    if actual != recorded:
        raise InferenceSafetyError("model files do not match the completed training manifest")


class InferenceSafetyError(RuntimeError):
    """Raised when a local model violates the release/inference contract."""


def _normalize_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for raw_id, raw_label in value.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise InferenceSafetyError("model id2label contains a non-integer key") from exc
        if not isinstance(raw_label, str) or not raw_label:
            raise InferenceSafetyError("model id2label contains an invalid label")
        result[label_id] = raw_label
    if not result or sorted(result) != list(range(len(result))):
        raise InferenceSafetyError("model id2label must be contiguous from zero")
    return result


class TransformersSpanPredictor:
    """Decode a local token classifier into relative character spans.

    Returned dictionaries intentionally contain no input text or token strings.
    They implement the ``WindowPredictor`` protocol used by
    :class:`pii_zh.presidio.QwenPiiRecognizer`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        device: str | torch.device | None = None,
        micro_batch_size: int = 16,
        ignored_labels: Sequence[str] = (),
        attention_mode: str = "full",
        jpt_sep_token_id: int | None = None,
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
        if config is None or not isinstance(getattr(config, "id2label", None), Mapping):
            raise ValueError("model config must provide id2label")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.id2label = _normalize_id2label(config.id2label)
        self.ignored_labels = frozenset(ignored_labels)
        self.micro_batch_size = micro_batch_size
        if attention_mode not in {"causal", "full", "jpt"}:
            raise ValueError("attention_mode must be causal, full, or jpt")
        if attention_mode == "jpt" and jpt_sep_token_id is None:
            raise ValueError("JPT inference requires a separator token id")
        self.attention_mode = attention_mode
        self.jpt_sep_token_id = jpt_sep_token_id
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
        source_width = model_inputs["input_ids"].shape[1]
        if self.attention_mode == "jpt":
            repeated = build_jpt_inputs(
                model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                sep_token_id=int(self.jpt_sep_token_id),
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            repeated.pop("second_copy_mask")
            model_inputs = repeated
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits.float()
        if self.attention_mode == "jpt":
            second_start = source_width + 1
            logits = logits[:, second_start : second_start + source_width]
        probabilities = torch.softmax(logits, dim=-1)
        token_scores, label_ids = probabilities.max(dim=-1)
        results: list[list[dict[str, Any]]] = []
        for row in range(len(texts)):
            row_offsets = [tuple(map(int, pair)) for pair in offsets[row].tolist()]
            spans = decode_bio_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                token_scores=token_scores[row].tolist(),
            )
            results.append(
                [
                    {
                        "entity_type": span.entity_type,
                        "start": span.start,
                        "end": span.end,
                        "score": float(span.score if span.score is not None else 1.0),
                        "offset_mode": "relative",
                        "metadata": {"boundary_refined": False},
                    }
                    for span in spans
                    if span.entity_type not in self.ignored_labels
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


def load_local_predictor(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    micro_batch_size: int = 16,
    ignored_labels: Sequence[str] = (),
) -> TransformersSpanPredictor:
    """Load a full-attention release or marked causal/JPT research artifact."""

    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise InferenceSafetyError("model_path must be a local model directory")
    if not (path / "model.safetensors").is_file():
        raise InferenceSafetyError("model package is missing model.safetensors")
    forbidden = [
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pth")
        for item in path.glob(pattern)
    ]
    if forbidden:
        raise InferenceSafetyError(f"unsafe weight files are forbidden: {sorted(forbidden)}")
    if any(path.glob("adapter_*")):
        raise InferenceSafetyError("adapter-only artifacts are forbidden in the model root")
    _verify_training_artifact_binding(path)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "weights_only": True,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        config_document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InferenceSafetyError("model config is not valid local JSON") from exc
    if not isinstance(config_document, dict):
        raise InferenceSafetyError("model config must be a JSON object")
    attention_mode = config_document.get("pii_attention_mode")
    model_type = config_document.get("model_type")
    if attention_mode == "full" and model_type == "qwen3_bi":
        model = Qwen3BiForTokenClassification.from_pretrained(path, **kwargs)
        if getattr(model.config, "architecture_version", None) != ARCHITECTURE_VERSION:
            raise InferenceSafetyError("unsupported Qwen3Bi architecture version")
    elif attention_mode in {"causal", "jpt"} and model_type == "qwen3":
        if config_document.get("pii_release_eligible") is not False:
            raise InferenceSafetyError("research attention artifact has invalid release marker")
        model = Qwen3ForTokenClassification.from_pretrained(path, **kwargs)
    else:
        raise InferenceSafetyError("model type and PII attention mode are inconsistent")
    jpt_sep_token_id = config_document.get("pii_jpt_sep_token_id")
    if attention_mode == "jpt" and (
        isinstance(jpt_sep_token_id, bool) or not isinstance(jpt_sep_token_id, int)
    ):
        raise InferenceSafetyError("JPT artifact is missing its separator token id")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    assert_character_boundary_tokenizer(tokenizer)
    return TransformersSpanPredictor(
        model,
        tokenizer,
        device=device,
        micro_batch_size=micro_batch_size,
        ignored_labels=ignored_labels,
        attention_mode=attention_mode,
        jpt_sep_token_id=jpt_sep_token_id,
    )
