"""Verified, weights-only staged initialization for Full Attention training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen3Config
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "Staged initialization requires torch, safetensors, and Transformers."
    ) from exc

from pii_zh.models.qwen3_bi import Qwen3BiForTokenClassification
from pii_zh.tokenization import load_serialized_fast_tokenizer
from pii_zh.training.config import STAGED_INITIALIZATION_STRATEGY, TrainingConfig
from pii_zh.training.data import TrainingDataSummary
from pii_zh.training.loading import (
    BackboneLoadingAudit,
    CheckpointSafetyError,
    InitializationAudit,
)
from pii_zh.training.manifest import (
    canonical_json_hash,
    resolve_training_source_ids,
    sha256_file,
    tokenizer_fingerprint,
    training_data_summary_dict,
    verify_output_artifact_binding,
)

_SUPPORTED_SOURCE_MANIFEST_SCHEMAS = frozenset({2, 3, 4})
_SOURCE_ATTENTION_MODES = frozenset({"causal", "jpt"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARCHITECTURE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_sliding_window",
    "sliding_window",
    "max_window_layers",
    "rope_theta",
    "rope_scaling",
    "attention_bias",
    "attention_dropout",
    "tie_word_embeddings",
    "layer_types",
    "mlp_only_layers",
)
_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.bool: "BOOL",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointSafetyError(f"{description} must be a regular local file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointSafetyError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointSafetyError(f"{description} must be a JSON object")
    return value


def _object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointSafetyError(f"{description} must be an object")
    return value


def _sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CheckpointSafetyError(f"{description} must be a lowercase SHA-256")
    return value


def _architecture_signature(config: Qwen3Config) -> dict[str, Any]:
    values = config.to_dict()
    return {name: values.get(name) for name in _ARCHITECTURE_FIELDS}


def _validate_source_base(
    manifest: dict[str, Any],
    *,
    source_attention_mode: str,
    config: TrainingConfig,
    base_loading_audit: BackboneLoadingAudit,
) -> None:
    if manifest.get("base_source_id") != config.base_source_id:
        raise CheckpointSafetyError("initial model base_source_id disagrees with the target")
    base = _object(manifest.get("base_checkpoint"), "initial model base_checkpoint")
    expected = {
        "config_sha256": base_loading_audit.config_sha256,
        "weights_sha256": base_loading_audit.weights_sha256,
        "safetensor_files": list(base_loading_audit.safetensor_files),
    }
    if any(base.get(name) != value for name, value in expected.items()):
        raise CheckpointSafetyError(
            "initial model was not trained from the selected base checkpoint"
        )

    audit = _object(base.get("loading_audit"), "initial model base loading audit")
    for name, value in expected.items():
        if audit.get(name) != value:
            raise CheckpointSafetyError(
                "initial model base loading audit is internally inconsistent"
            )
    if audit.get("attention_mode") != source_attention_mode:
        raise CheckpointSafetyError("initial model attention mode disagrees with its loading audit")
    missing = audit.get("missing_keys")
    initialized = audit.get("newly_initialized_score_keys")
    if (
        not isinstance(missing, list)
        or not isinstance(initialized, list)
        or sorted(missing) != sorted(initialized)
        or not initialized
        or any(not isinstance(key, str) or not key.startswith("score.") for key in initialized)
        or audit.get("mismatched_keys") != []
        or audit.get("error_messages") != []
    ):
        raise CheckpointSafetyError("initial model base loading transition is not clean")
    unexpected = audit.get("unexpected_keys")
    if not isinstance(unexpected, list) or any(
        not isinstance(key, str) or not key.startswith("lm_head.") for key in unexpected
    ):
        raise CheckpointSafetyError("initial model base loading audit has unsafe unexpected keys")


def _validate_source_config(
    root: Path,
    manifest: dict[str, Any],
    *,
    source_attention_mode: str,
    target_config: TrainingConfig,
    label2id: dict[str, int],
    id2label: dict[int, str],
    target_tokenizer: Any,
) -> tuple[Qwen3Config, str]:
    try:
        source_config = Qwen3Config.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        base_config = Qwen3Config.from_pretrained(
            Path(target_config.base_model).expanduser().resolve(strict=True),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise CheckpointSafetyError("failed to parse staged initialization configs safely") from exc

    if source_config.model_type != "qwen3":
        raise CheckpointSafetyError("initial model must use the native qwen3 model type")
    if source_config.architectures != ["Qwen3ForTokenClassification"]:
        raise CheckpointSafetyError("initial model must be a standalone Qwen3 token classifier")
    if getattr(source_config, "auto_map", None):
        raise CheckpointSafetyError("initial model auto_map is forbidden")
    if getattr(source_config, "pii_attention_mode", None) != source_attention_mode:
        raise CheckpointSafetyError(
            "initial model config attention mode disagrees with its manifest"
        )
    if getattr(source_config, "pii_release_eligible", None) is not False:
        raise CheckpointSafetyError("only non-release causal/JPT checkpoints may initialize Full")
    if source_config.use_cache is not False:
        raise CheckpointSafetyError("initial model must have KV caching disabled")
    if source_config.label2id != label2id or source_config.id2label != id2label:
        raise CheckpointSafetyError("initial model label mapping disagrees with the target")
    if int(source_config.num_labels) != len(label2id):
        raise CheckpointSafetyError("initial model label count disagrees with the target")
    if float(getattr(source_config, "classifier_dropout", -1.0)) != float(
        target_config.classifier_dropout
    ):
        raise CheckpointSafetyError("initial model classifier dropout disagrees with the target")
    if source_config.pad_token_id != target_tokenizer.pad_token_id:
        raise CheckpointSafetyError("initial model pad token disagrees with the target tokenizer")
    if source_attention_mode == "jpt":
        separator = getattr(source_config, "pii_jpt_sep_token_id", None)
        if (
            isinstance(separator, bool)
            or not isinstance(separator, int)
            or not (0 <= separator < source_config.vocab_size)
        ):
            raise CheckpointSafetyError("initial JPT model has an invalid separator token")
    elif hasattr(source_config, "pii_jpt_sep_token_id"):
        raise CheckpointSafetyError("causal initial model unexpectedly declares a JPT separator")

    signature = _architecture_signature(source_config)
    if signature != _architecture_signature(base_config):
        raise CheckpointSafetyError("initial model architecture disagrees with the selected base")
    if manifest.get("label2id") != dict(sorted(label2id.items(), key=lambda item: item[1])):
        raise CheckpointSafetyError(
            "initial model manifest label mapping disagrees with the target"
        )
    return source_config, canonical_json_hash(signature)


def _validate_tokenizers(
    root: Path,
    manifest: dict[str, Any],
    *,
    config: TrainingConfig,
    target_tokenizer: Any,
) -> str:
    target = tokenizer_fingerprint(config.base_model, tokenizer=target_tokenizer)
    if manifest.get("tokenizer") != target:
        raise CheckpointSafetyError("initial model training tokenizer disagrees with the target")
    try:
        source_tokenizer = load_serialized_fast_tokenizer(root, require_boundary=True)
        source_effective = tokenizer_fingerprint(root, tokenizer=source_tokenizer).get("effective")
    except Exception as exc:
        raise CheckpointSafetyError(
            "initial model tokenizer failed its boundary-safe audit"
        ) from exc
    if source_effective != target.get("effective"):
        raise CheckpointSafetyError("initial model tokenizer contract disagrees with the target")
    return _sha256(target.get("effective_contract_sha256"), "target tokenizer contract")


def _validate_dataset(
    manifest: dict[str, Any],
    *,
    name: str,
    path: str,
    summary: TrainingDataSummary,
) -> str:
    datasets = _object(manifest.get("datasets"), "initial model datasets")
    source = _object(datasets.get(name), f"initial model {name} dataset")
    digest = sha256_file(path)
    if source.get("sha256") != digest:
        raise CheckpointSafetyError(f"initial model {name} data hash disagrees with the target")
    if source.get("summary") != training_data_summary_dict(summary):
        raise CheckpointSafetyError(
            f"initial model {name} provenance summary disagrees with target"
        )
    return digest


def _validate_state_header(
    model: torch.nn.Module,
    weights_path: Path,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    expected: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, tensor in model.state_dict().items():
        dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(tensor.dtype)
        if dtype is None:
            raise CheckpointSafetyError("target model contains an unsupported tensor dtype")
        expected[name] = (tuple(tensor.shape), dtype)
    try:
        # ``safe_open`` is a compiled extension without Python typing metadata.
        with safe_open(  # type: ignore[no-untyped-call]
            weights_path, framework="pt", device="cpu"
        ) as handle:
            actual = {
                name: (
                    tuple(handle.get_slice(name).get_shape()),
                    str(handle.get_slice(name).get_dtype()),
                )
                for name in handle.keys()
            }
    except Exception as exc:
        raise CheckpointSafetyError("initial model safetensors header is invalid") from exc

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        name for name in set(expected) & set(actual) if expected[name] != actual[name]
    )
    if missing or unexpected or mismatched:
        raise CheckpointSafetyError(
            "initial model state dict does not exactly match the Full target architecture"
        )
    dtypes = tuple(sorted({dtype for _, dtype in actual.values()}))
    score_keys = tuple(sorted(name for name in actual if name.startswith("score.")))
    if not score_keys:
        raise CheckpointSafetyError("initial model is missing its trained classification head")
    return len(actual), dtypes, score_keys


def initialize_full_attention_from_verified_model(
    model: torch.nn.Module,
    *,
    config: TrainingConfig,
    base_loading_audit: BackboneLoadingAudit,
    taxonomy_version: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    tokenizer: Any,
    train_summary: TrainingDataSummary,
    validation_summary: TrainingDataSummary,
) -> InitializationAudit:
    """Strictly overlay a completed causal/JPT classifier onto a Full target.

    Only model weights cross this boundary. Trainer, optimizer, scheduler, RNG,
    global-step, and epoch state are deliberately not read from the source run.
    """

    if config.initial_model is None:
        raise ValueError("initial_model is required for staged initialization")
    if config.attention_mode != "full" or config.resume is not False:
        raise CheckpointSafetyError("staged initialization requires a fresh Full run")
    if not isinstance(model, Qwen3BiForTokenClassification):
        raise CheckpointSafetyError("staged initialization target must be Qwen3Bi")

    candidate = Path(config.initial_model).expanduser()
    if candidate.is_symlink():
        raise CheckpointSafetyError("initial_model directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise CheckpointSafetyError("initial_model must resolve to a local directory") from exc
    if not root.is_dir():
        raise CheckpointSafetyError("initial_model must resolve to a local directory")
    manifest_path = root / "training_manifest.json"
    manifest = _load_json_object(manifest_path, "initial model training manifest")
    source_manifest_file_sha256 = sha256_file(manifest_path)
    schema_version = manifest.get("schema_version")
    if schema_version not in _SUPPORTED_SOURCE_MANIFEST_SCHEMAS:
        raise CheckpointSafetyError("initial model training manifest schema is unsupported")
    if manifest.get("status") != "completed" or not verify_output_artifact_binding(
        manifest, root, require_single_weight=True
    ):
        raise CheckpointSafetyError("initial model manifest/artifact binding is invalid")

    source_attention_mode = manifest.get("attention_mode")
    if source_attention_mode not in _SOURCE_ATTENTION_MODES:
        raise CheckpointSafetyError("initial model must be a causal or JPT training artifact")
    source_fine_tuning = manifest.get("fine_tuning")
    if source_fine_tuning not in {"full", "lora"}:
        raise CheckpointSafetyError("initial model fine-tuning mode is invalid")
    source_recipe = _object(manifest.get("recipe"), "initial model training recipe")
    if (
        source_recipe.get("attention_mode") != source_attention_mode
        or source_recipe.get("fine_tuning") != source_fine_tuning
    ):
        raise CheckpointSafetyError("initial model recipe disagrees with its manifest")
    if schema_version in {3, 4}:
        source_initialization = _object(
            manifest.get("initialization"), "initial model initialization audit"
        )
        if (
            source_recipe.get("initialization_strategy") != "base_causal_lm_v1"
            or source_initialization.get("strategy") != "base_causal_lm_v1"
        ):
            raise CheckpointSafetyError("causal/JPT initializer must originate from the Base model")
    if schema_version == 4:
        expected_source_ids = list(
            resolve_training_source_ids(config, train_summary, validation_summary)
        )
        if manifest.get("training_source_ids") != expected_source_ids:
            raise CheckpointSafetyError(
                "initial model training source lineage disagrees with the target"
            )
    source_manifest_sha256 = _sha256(manifest.get("manifest_sha256"), "initial model manifest hash")
    source_code_revision = manifest.get("code_revision")
    if source_code_revision is not None and (
        not isinstance(source_code_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_code_revision) is None
    ):
        raise CheckpointSafetyError("initial model code revision is invalid")

    _validate_source_base(
        manifest,
        source_attention_mode=source_attention_mode,
        config=config,
        base_loading_audit=base_loading_audit,
    )
    _, source_architecture_sha256 = _validate_source_config(
        root,
        manifest,
        source_attention_mode=source_attention_mode,
        target_config=config,
        label2id=label2id,
        id2label=id2label,
        target_tokenizer=tokenizer,
    )
    label_schema_sha256 = canonical_json_hash(label2id)
    if manifest.get("label_schema_sha256") != label_schema_sha256:
        raise CheckpointSafetyError("initial model label schema hash disagrees with the target")
    if manifest.get("taxonomy_version") != taxonomy_version:
        raise CheckpointSafetyError("initial model taxonomy version disagrees with the target")
    tokenizer_contract = _validate_tokenizers(
        root,
        manifest,
        config=config,
        target_tokenizer=tokenizer,
    )
    train_sha256 = _validate_dataset(
        manifest,
        name="train",
        path=config.train_file,
        summary=train_summary,
    )
    validation_sha256 = _validate_dataset(
        manifest,
        name="validation",
        path=config.validation_file,
        summary=validation_summary,
    )

    output_artifact = _object(manifest.get("output_artifact"), "initial model output artifact")
    files = _object(output_artifact.get("files"), "initial model output artifact files")
    source_config_sha256 = _sha256(files.get("config.json"), "initial model config hash")
    source_weights_sha256 = _sha256(
        output_artifact.get("weights_combined_sha256"), "initial model weights hash"
    )
    source_safetensor_files = output_artifact.get("weight_files")
    if source_safetensor_files != ["model.safetensors"]:
        raise CheckpointSafetyError("initial model must contain exactly model.safetensors")
    weights_path = root / "model.safetensors"
    tensor_count, tensor_dtypes, score_keys = _validate_state_header(model, weights_path)
    try:
        source_state = load_file(weights_path, device="cpu")
        incompatible = model.load_state_dict(source_state, strict=True)
    except Exception as exc:
        raise CheckpointSafetyError("strict staged state-dict loading failed") from exc
    finally:
        if "source_state" in locals():
            del source_state
    missing = tuple(sorted(incompatible.missing_keys))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    if missing or unexpected:
        raise CheckpointSafetyError("strict staged state-dict loading was incomplete")
    if not verify_output_artifact_binding(manifest, root, require_single_weight=True):
        raise CheckpointSafetyError("initial model changed while it was being loaded")
    if sha256_file(manifest_path) != source_manifest_file_sha256:
        raise CheckpointSafetyError("initial model manifest changed while it was being loaded")

    return InitializationAudit(
        strategy=STAGED_INITIALIZATION_STRATEGY,
        source_manifest_schema_version=int(schema_version),
        source_manifest_sha256=source_manifest_sha256,
        source_manifest_file_sha256=source_manifest_file_sha256,
        source_output_artifact_sha256=canonical_json_hash(output_artifact),
        source_attention_mode=str(source_attention_mode),
        source_fine_tuning=str(source_fine_tuning),
        source_code_revision=source_code_revision,
        source_config_sha256=source_config_sha256,
        source_weights_sha256=source_weights_sha256,
        source_safetensor_files=("model.safetensors",),
        source_architecture_sha256=source_architecture_sha256,
        base_source_id=config.base_source_id,
        base_config_sha256=base_loading_audit.config_sha256,
        base_weights_sha256=base_loading_audit.weights_sha256,
        label_schema_sha256=label_schema_sha256,
        taxonomy_version=taxonomy_version,
        tokenizer_effective_contract_sha256=tokenizer_contract,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
        tensor_count=tensor_count,
        tensor_dtypes=tensor_dtypes,
        score_keys=score_keys,
        missing_keys=missing,
        unexpected_keys=unexpected,
        mismatched_keys=(),
    )


__all__ = ["initialize_full_attention_from_verified_model"]
