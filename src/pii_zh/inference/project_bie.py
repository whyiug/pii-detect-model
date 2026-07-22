"""Reusable inference for the project's 73-logit native-causal BIE model."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import Qwen3ForTokenClassification
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError("Project BIE inference requires torch and transformers.") from exc

from pii_zh.inference.aiguard import AiguardProtocolError, decode_bie_ids, normalize_id2label
from pii_zh.models.aiguard24_bie import build_core_bie_label_maps
from pii_zh.tokenization import load_serialized_fast_tokenizer
from pii_zh.training.manifest import verify_output_artifact_binding, verify_training_manifest

PROJECT_BIE_PROTOCOL_ID = "pii_zh_core24_native_causal_bie_v1"
PROJECT_CLOSED8_LABELS: tuple[str, ...] = (
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ProjectBieInferenceError(ValueError):
    """Raised when a model or allowed-label protocol is not BIE73-compatible."""


def _verify_completed_candidate_manifest(manifest: Mapping[str, Any]) -> bool:
    """Require independent-dev feasibility and an exact best-checkpoint receipt."""

    update_mode = manifest.get("update_mode")
    recipe = manifest.get("recipe")
    validation = manifest.get("independent_validation")
    checkpoint = manifest.get("checkpoint_selection")
    benchmark = manifest.get("benchmark_isolation")
    label2id, _ = build_core_bie_label_maps()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_eligible") is not False
        or manifest.get("validation_split_name") != "dev"
        or update_mode not in {"head_only", "lora"}
        or manifest.get("label2id") != label2id
        or not isinstance(recipe, Mapping)
        or recipe.get("attention_mode") != "causal"
        or recipe.get("tag_scheme") != "BIE"
        or recipe.get("update_mode") != update_mode
        or recipe.get("checkpoint_selection_split") != "independent_validation"
        or recipe.get("checkpoint_selection_metric") != "independent_dev_selection_score"
        or not isinstance(validation, Mapping)
        or validation.get("eval_dev_gate_passed") != 1.0
        or not isinstance(benchmark, Mapping)
        or any(
            benchmark.get(field) is not False
            for field in (
                "public_test_read_for_training",
                "public_test_read_for_validation",
                "public_test_read_for_checkpoint_selection",
                "pii_bench_zh_read",
            )
        )
        or not isinstance(checkpoint, Mapping)
    ):
        return False

    step = checkpoint.get("selected_global_step")
    best_metric = checkpoint.get("best_metric")
    replay_metric = checkpoint.get("final_validation_replay_metric")
    hashes = checkpoint.get("safetensor_sha256")
    expected_weight = (
        "adapter_model.safetensors" if update_mode == "lora" else "model.safetensors"
    )
    return bool(
        checkpoint.get("selection_split") == "independent_validation"
        and checkpoint.get("selection_metric") == "independent_dev_selection_score"
        and checkpoint.get("update_mode", update_mode) == update_mode
        and isinstance(step, int)
        and not isinstance(step, bool)
        and step > 0
        and checkpoint.get("selected_checkpoint_id") == f"checkpoint-{step}"
        and isinstance(best_metric, (int, float))
        and not isinstance(best_metric, bool)
        and math.isfinite(float(best_metric))
        and isinstance(replay_metric, (int, float))
        and not isinstance(replay_metric, bool)
        and math.isfinite(float(replay_metric))
        and math.isclose(
            float(best_metric),
            float(replay_metric),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        )
        and isinstance(hashes, Mapping)
        and set(hashes) == {expected_weight}
        and all(isinstance(value, str) and _SHA256.fullmatch(value) for value in hashes.values())
    )


def normalize_project_bie_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    """Require the exact packaged ``O + 24 * B/I/E`` label inventory."""

    try:
        normalized = normalize_id2label(value)
    except AiguardProtocolError as exc:
        raise ProjectBieInferenceError("invalid project BIE id2label mapping") from exc
    _, expected = build_core_bie_label_maps()
    if normalized != expected:
        raise ProjectBieInferenceError("model does not expose the exact project BIE73 schema")
    return normalized


def project_bie_allowed_label_ids(
    id2label: Mapping[int, str],
    allowed_project_labels: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Return ``O`` plus B/I/E IDs for uppercase project entity labels."""

    normalized = normalize_project_bie_id2label(id2label)
    entity_types = {label.split("-", 1)[1] for label in normalized.values() if label != "O"}
    if allowed_project_labels is None:
        selected = entity_types
    else:
        if isinstance(allowed_project_labels, (str, bytes)):
            raise TypeError("allowed_project_labels must be a sequence")
        selected = set(allowed_project_labels)
        if (
            not selected
            or len(selected) != len(tuple(allowed_project_labels))
            or any(not isinstance(label, str) or not label for label in selected)
            or not selected <= entity_types
        ):
            raise ProjectBieInferenceError(
                "allowed_project_labels must be unique uppercase project labels"
            )
    expected_tags = {
        "O",
        *(f"{prefix}-{label}" for label in selected for prefix in _BIE_PREFIXES),
    }
    allowed = tuple(
        label_id for label_id, label in sorted(normalized.items()) if label in expected_tags
    )
    if len(allowed) != 1 + 3 * len(selected):
        raise ProjectBieInferenceError("allowed-label mask is incomplete")
    return allowed


def closed8_project_bie_label_ids(id2label: Mapping[int, str]) -> tuple[int, ...]:
    """Return the predecode ``O + 8 * B/I/E`` mask for closed-eight use."""

    return project_bie_allowed_label_ids(id2label, PROJECT_CLOSED8_LABELS)


def predecode_project_bie(
    logits: torch.Tensor,
    *,
    allowed_label_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict logits before softmax/argmax and return full-schema IDs/scores."""

    if not torch.is_floating_point(logits) or logits.ndim < 2 or logits.shape[-1] != 73:
        raise ProjectBieInferenceError("BIE predecode expects floating logits with width 73")
    if isinstance(allowed_label_ids, (str, bytes)):
        raise TypeError("allowed_label_ids must be an integer sequence")
    allowed_values = tuple(allowed_label_ids)
    if (
        not allowed_values
        or len(set(allowed_values)) != len(allowed_values)
        or 0 not in allowed_values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 73
            for value in allowed_values
        )
    ):
        raise ProjectBieInferenceError("allowed_label_ids must include O and unique IDs in [0, 73)")
    allowed = torch.tensor(allowed_values, dtype=torch.long, device=logits.device)
    restricted = logits.index_select(-1, allowed)
    probabilities = torch.softmax(restricted, dim=-1)
    token_scores, restricted_ids = probabilities.max(dim=-1)
    return allowed[restricted_ids], token_scores


class ProjectBieSpanPredictor:
    """Window-compatible predictor for a loaded project BIE73 model."""

    protocol_id = PROJECT_BIE_PROTOCOL_ID
    attention_mode = "causal"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        device: str | torch.device | None = None,
        micro_batch_size: int = 16,
        allowed_project_labels: Sequence[str] | None = None,
    ) -> None:
        if (
            isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or micro_batch_size < 1
        ):
            raise ProjectBieInferenceError("micro_batch_size must be a positive integer")
        if not getattr(tokenizer, "is_fast", False):
            raise ProjectBieInferenceError("exact character inference requires a fast tokenizer")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ProjectBieInferenceError("tokenizer lacks pad and EOS tokens")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        config = getattr(model, "config", None)
        raw_id2label = getattr(config, "id2label", None)
        if not isinstance(raw_id2label, Mapping):
            raise ProjectBieInferenceError("model config must provide id2label")
        if (
            getattr(config, "model_type", None) != "qwen3"
            or getattr(config, "pii_attention_mode", None) != "causal"
            or getattr(config, "pii_tagging_scheme", None) != "BIE"
        ):
            raise ProjectBieInferenceError("model is not a native-causal project BIE artifact")
        self.id2label = normalize_project_bie_id2label(raw_id2label)
        self.allowed_label_ids = project_bie_allowed_label_ids(
            self.id2label, allowed_project_labels
        )
        self.allowed_project_labels = (
            frozenset(allowed_project_labels)
            if allowed_project_labels is not None
            else frozenset(
                label.split("-", 1)[1] for label in self.id2label.values() if label != "O"
            )
        )
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
        label_ids, token_scores = predecode_project_bie(
            logits,
            allowed_label_ids=self.allowed_label_ids,
        )
        results: list[list[dict[str, Any]]] = []
        for row in range(len(texts)):
            row_offsets = [tuple(map(int, pair)) for pair in offsets[row].tolist()]
            if any(len(pair) != 2 for pair in row_offsets):
                raise ProjectBieInferenceError("tokenizer returned invalid offsets")
            decoded = decode_bie_ids(
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


def load_local_project_bie_predictor(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    micro_batch_size: int = 16,
    allowed_project_labels: Sequence[str] | None = None,
) -> ProjectBieSpanPredictor:
    """Load a completed, manifest-bound, safetensors-only BIE73 candidate."""

    candidate = Path(model_path).expanduser()
    if candidate.is_symlink():
        raise ProjectBieInferenceError("model root must be a regular local directory")
    path = candidate.resolve(strict=True)
    required = ("config.json", "model.safetensors", "tokenizer.json", "training_manifest.json")
    if not path.is_dir() or any(
        (path / name).is_symlink() or not (path / name).is_file() for name in required
    ):
        raise ProjectBieInferenceError("model path lacks required standalone files")
    if any(path.glob("adapter_*")) or any(
        any(path.glob(pattern)) for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pt", "*.pth")
    ):
        raise ProjectBieInferenceError("unsafe or adapter-only model artifacts are forbidden")
    try:
        manifest = json.loads((path / "training_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectBieInferenceError("training manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or not verify_training_manifest(manifest)
        or manifest.get("status") != "completed"
        or manifest.get("manifest_type") != "aiguard24_native_causal_bie_training_v1"
        or manifest.get("attention_mode") != "causal"
        or manifest.get("tag_scheme") != "BIE"
        or not _verify_completed_candidate_manifest(manifest)
        or not verify_output_artifact_binding(manifest, path, require_single_weight=True)
    ):
        raise ProjectBieInferenceError("training manifest or artifact binding failed")
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
    tokenizer = load_serialized_fast_tokenizer(path, require_boundary=True)
    return ProjectBieSpanPredictor(
        model,
        tokenizer,
        device=device,
        micro_batch_size=micro_batch_size,
        allowed_project_labels=allowed_project_labels,
    )


__all__ = [
    "PROJECT_BIE_PROTOCOL_ID",
    "PROJECT_CLOSED8_LABELS",
    "ProjectBieInferenceError",
    "ProjectBieSpanPredictor",
    "closed8_project_bie_label_ids",
    "load_local_project_bie_predictor",
    "normalize_project_bie_id2label",
    "predecode_project_bie",
    "project_bie_allowed_label_ids",
]
