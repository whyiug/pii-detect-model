"""Fail-closed loading of local Qwen3 Base CausalLM safetensors checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForTokenClassification
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "Checkpoint loading requires torch, safetensors, and Transformers with Qwen3 support."
    ) from exc

from pii_zh.models.qwen3_bi import Qwen3BiConfig, Qwen3BiForTokenClassification


class CheckpointSafetyError(RuntimeError):
    """Raised when a checkpoint or its state-dict transition fails audit."""


@dataclass(frozen=True, slots=True)
class BackboneLoadingAudit:
    attention_mode: str
    config_sha256: str
    weights_sha256: str
    safetensor_files: tuple[str, ...]
    lm_head_present_in_checkpoint: bool
    lm_head_discarded: bool
    newly_initialized_score_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]
    error_messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InitializationAudit:
    """Path-free evidence for a verified token-classifier weight transition."""

    strategy: str
    source_manifest_schema_version: int
    source_manifest_sha256: str
    source_manifest_file_sha256: str
    source_output_artifact_sha256: str
    source_attention_mode: str
    source_fine_tuning: str
    source_code_revision: str | None
    source_config_sha256: str
    source_weights_sha256: str
    source_safetensor_files: tuple[str, ...]
    source_architecture_sha256: str
    base_source_id: str
    base_config_sha256: str
    base_weights_sha256: str
    label_schema_sha256: str
    taxonomy_version: str
    tokenizer_effective_contract_sha256: str
    train_sha256: str
    validation_sha256: str
    tensor_count: int
    tensor_dtypes: tuple[str, ...]
    score_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(values: dict[str, str]) -> str:
    encoded = json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def inspect_local_qwen3_checkpoint(
    checkpoint: str | Path,
) -> tuple[Path, Qwen3Config, tuple[Path, ...], bool, str, str]:
    """Validate a local Qwen3 CausalLM checkpoint without executing repository code."""

    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise CheckpointSafetyError("base_model must resolve to a local checkpoint directory")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise CheckpointSafetyError("local checkpoint is missing config.json")
    safetensor_files = tuple(sorted(path.glob("*.safetensors")))
    if not safetensor_files:
        raise CheckpointSafetyError(
            "local checkpoint has no safetensors weights; pickle-based .bin loading is forbidden"
        )

    try:
        config = Qwen3Config.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise CheckpointSafetyError("failed to parse local checkpoint as Qwen3 config") from exc
    if config.model_type != "qwen3":
        raise CheckpointSafetyError(f"expected model_type='qwen3', got {config.model_type!r}")
    if getattr(config, "auto_map", None):
        raise CheckpointSafetyError("checkpoint auto_map is forbidden for safe local loading")
    architectures = set(config.architectures or [])
    if architectures and "Qwen3ForCausalLM" not in architectures:
        raise CheckpointSafetyError(
            "checkpoint must declare Qwen3ForCausalLM; a fine-tuned or unrelated architecture "
            f"was declared: {sorted(architectures)!r}"
        )

    lm_head_present = False
    for weight_path in safetensor_files:
        try:
            with safe_open(weight_path, framework="pt", device="cpu") as handle:
                if "lm_head.weight" in handle.keys():
                    lm_head_present = True
        except Exception as exc:
            raise CheckpointSafetyError(
                f"invalid safetensors header in {weight_path.name!r}"
            ) from exc

    weight_hashes = {item.name: _sha256_file(item) for item in safetensor_files}
    return (
        path,
        config,
        safetensor_files,
        lm_head_present,
        _sha256_file(config_path),
        _combined_hash(weight_hashes),
    )


def _normalise_loading_keys(
    value: object,
    *,
    field: str,
    structured: bool = False,
) -> tuple[str, ...]:
    """Normalize legacy lists and Transformers 5 ``set`` loading reports.

    Transformers 5.13 returns sets from ``LoadStateDictInfo.to_dict``.  Invalid
    or absent report fields are an audit failure, never an empty result.
    """

    if not isinstance(value, (list, tuple, set, frozenset)):
        raise CheckpointSafetyError(f"Transformers loading report {field} is not a collection")
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif structured and isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            normalized.append(item[0])
        else:
            raise CheckpointSafetyError(
                f"Transformers loading report {field} contains an invalid entry"
            )
    if len(normalized) != len(set(normalized)):
        raise CheckpointSafetyError(f"Transformers loading report {field} contains duplicates")
    return tuple(sorted(normalized))


def _safetensor_schema(safetensor_files: tuple[Path, ...]) -> dict[str, tuple[int, ...]]:
    """Read the complete checkpoint key/shape schema without loading tensors."""

    schema: dict[str, tuple[int, ...]] = {}
    for weight_path in safetensor_files:
        try:
            with safe_open(weight_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in schema:
                        raise CheckpointSafetyError(
                            "checkpoint repeats a tensor key across safetensor shards"
                        )
                    schema[key] = tuple(int(size) for size in handle.get_slice(key).get_shape())
        except CheckpointSafetyError:
            raise
        except Exception as exc:
            raise CheckpointSafetyError(
                f"invalid safetensors header in {weight_path.name!r}"
            ) from exc
    return schema


def _audit_loading_transition(
    *,
    model: torch.nn.Module,
    info: dict[str, Any],
    attention_mode: str,
    config_sha256: str,
    weights_sha256: str,
    safetensor_files: tuple[Path, ...],
    lm_head_present: bool,
) -> BackboneLoadingAudit:
    if not isinstance(info, dict):
        raise CheckpointSafetyError("Transformers did not return a structured loading report")
    missing = _normalise_loading_keys(info.get("missing_keys"), field="missing_keys")
    unexpected = _normalise_loading_keys(info.get("unexpected_keys"), field="unexpected_keys")
    mismatched = _normalise_loading_keys(
        info.get("mismatched_keys"), field="mismatched_keys", structured=True
    )
    errors = _normalise_loading_keys(info.get("error_msgs"), field="error_msgs")
    expected_score = tuple(sorted(key for key in model.state_dict() if key.startswith("score.")))
    if expected_score != ("score.bias", "score.weight"):
        raise CheckpointSafetyError(
            "token-classifier architecture does not expose the approved score head"
        )

    checkpoint_schema = _safetensor_schema(safetensor_files)
    model_schema = {
        key: tuple(int(size) for size in value.shape) for key, value in model.state_dict().items()
    }
    schema_missing = tuple(sorted(set(model_schema) - set(checkpoint_schema)))
    schema_unexpected = tuple(sorted(set(checkpoint_schema) - set(model_schema)))
    schema_mismatched = tuple(
        sorted(
            key
            for key in set(model_schema) & set(checkpoint_schema)
            if model_schema[key] != checkpoint_schema[key]
        )
    )
    expected_unexpected = ("lm_head.weight",) if lm_head_present else ()
    if schema_missing != expected_score or schema_unexpected != expected_unexpected:
        raise CheckpointSafetyError(
            "checkpoint/model state-dict schema transition is not approved: "
            f"missing={list(schema_missing)}, unexpected={list(schema_unexpected)}"
        )
    if schema_mismatched:
        raise CheckpointSafetyError(
            f"checkpoint/model state-dict shapes differ: mismatched={list(schema_mismatched)}"
        )
    if missing != schema_missing or unexpected != schema_unexpected:
        raise CheckpointSafetyError(
            "Transformers loading report disagrees with the safetensors schema audit"
        )

    unapproved_missing = sorted(set(missing) - set(expected_score))
    unapproved_unexpected = sorted(set(unexpected) - set(expected_unexpected))
    if unapproved_missing or unapproved_unexpected or mismatched or errors:
        raise CheckpointSafetyError(
            "CausalLM-to-token-classifier state-dict audit failed: "
            f"unapproved_missing={unapproved_missing}, "
            f"unapproved_unexpected={unapproved_unexpected}, "
            f"mismatched={list(mismatched)}, error_count={len(errors)}"
        )
    if set(missing) != set(expected_score):
        raise CheckpointSafetyError(
            "classification head was not cleanly initialized; expected missing keys "
            f"{list(expected_score)!r}, got {list(missing)!r}"
        )
    discarded_lm_head = "lm_head.weight" in unexpected
    if discarded_lm_head != lm_head_present:
        raise CheckpointSafetyError(
            "lm_head presence in safetensors disagrees with Transformers loading audit"
        )

    return BackboneLoadingAudit(
        attention_mode=attention_mode,
        config_sha256=config_sha256,
        weights_sha256=weights_sha256,
        safetensor_files=tuple(item.name for item in safetensor_files),
        lm_head_present_in_checkpoint=lm_head_present,
        lm_head_discarded=discarded_lm_head,
        newly_initialized_score_keys=expected_score,
        missing_keys=missing,
        unexpected_keys=unexpected,
        mismatched_keys=mismatched,
        error_messages=errors,
    )


def load_token_classifier_from_local_causal_lm(
    checkpoint: str | Path,
    *,
    attention_mode: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    classifier_dropout: float = 0.1,
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, BackboneLoadingAudit]:
    """Load a causal or full-attention classifier from local CausalLM weights.

    ``jpt`` uses the causal classifier architecture; only its input/collation
    differs.  No Hub lookup, remote code, or pickle weight fallback is allowed.
    """

    if attention_mode not in {"causal", "full", "jpt"}:
        raise ValueError("attention_mode must be causal, full, or jpt")
    (
        path,
        base_config,
        safetensor_files,
        lm_head_present,
        config_sha256,
        weights_sha256,
    ) = inspect_local_qwen3_checkpoint(checkpoint)

    base_config.num_labels = len(label2id)
    base_config.label2id = dict(label2id)
    base_config.id2label = dict(id2label)
    base_config.classifier_dropout = classifier_dropout
    base_config.use_cache = False
    common_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "weights_only": True,
        "output_loading_info": True,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        common_kwargs["dtype"] = dtype

    try:
        if attention_mode == "full":
            model_config = Qwen3BiConfig.from_qwen3_config(base_config)
            model, info = Qwen3BiForTokenClassification.from_pretrained(
                path,
                config=model_config,
                **common_kwargs,
            )
        else:
            model, info = Qwen3ForTokenClassification.from_pretrained(
                path,
                config=base_config,
                **common_kwargs,
            )
    except Exception as exc:
        raise CheckpointSafetyError(
            "Transformers rejected the local CausalLM-to-classifier transition"
        ) from exc
    audit = _audit_loading_transition(
        model=model,
        info=info,
        attention_mode=attention_mode,
        config_sha256=config_sha256,
        weights_sha256=weights_sha256,
        safetensor_files=safetensor_files,
        lm_head_present=lm_head_present,
    )
    return model, audit
