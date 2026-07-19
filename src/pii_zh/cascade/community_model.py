"""Lightweight fail-closed contract for the community AIGuard24 artifact.

This module intentionally does not import torch or transformers.  It validates
the local artifact, exact core-24 BIO schema and completed full-attention
training receipt before the heavyweight loader is reached.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pii_zh.taxonomy import load_taxonomy

COMMUNITY_MODEL_IDENTITY_SCHEMA_VERSION = "pii-zh.community-model-identity.v1"
COMMUNITY_MANIFEST_TYPE = "aiguard24_community_training_v1"
COMMUNITY_BASE_SOURCE_ID = "ZJUICSR/AIguard-pii-detection-fast"
COMMUNITY_BASE_SOURCE_REVISION = "677a5ebc1600fef61e8973cafd3026be322b3a73"
COMMUNITY_MODEL_TYPE = "qwen3_bi"
COMMUNITY_ATTENTION_MODE = "full"
COMMUNITY_ARCHITECTURE_VERSION = "qwen3_bi_token_cls_v1"
COMMUNITY_ARCHITECTURE = "Qwen3BiForTokenClassification"
COMMUNITY_TRAINING_STATUS = "completed_candidate_not_benchmark_evaluated"

_OUTPUT_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CommunityModelContractError(RuntimeError):
    """Raised before loading when a community model violates its exact contract."""


@dataclass(frozen=True, slots=True)
class CommunityModelIdentity:
    """Path-free identity safe to expose in local CLI/API responses."""

    schema_version: str
    manifest_type: str
    manifest_sha256: str
    model_type: str
    attention_mode: str
    architecture_version: str
    taxonomy_version: str
    label_count: int
    label_schema_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerifiedCommunityModel:
    root: Path
    identity: CommunityModelIdentity


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CommunityModelContractError(f"community model {kind} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CommunityModelContractError(f"community model {kind} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CommunityModelContractError(f"community model {kind} must be a JSON object")
    return value


def expected_core24_label2id() -> dict[str, int]:
    taxonomy = load_taxonomy()
    labels = [taxonomy.outside_label]
    for entity in taxonomy.label_sets["core"]:
        labels.extend((f"B-{entity.name}", f"I-{entity.name}"))
    result = {label: index for index, label in enumerate(labels)}
    if len(result) != 49:
        raise CommunityModelContractError("packaged taxonomy is not exact core-24 BIO")
    return result


def _normalized_id2label(value: object) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise CommunityModelContractError("community model config id2label is missing")
    result: dict[int, str] = {}
    for raw_key, raw_label in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_label, str):
            raise CommunityModelContractError("community model config id2label is malformed")
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise CommunityModelContractError(
                "community model config id2label is malformed"
            ) from exc
        if key in result:
            raise CommunityModelContractError("community model config id2label is ambiguous")
        result[key] = raw_label
    return result


def _verify_output_artifact(root: Path, manifest: Mapping[str, Any]) -> None:
    weights = tuple(sorted(root.glob("model*.safetensors")))
    if len(weights) != 1 or weights[0].name != "model.safetensors":
        raise CommunityModelContractError(
            "community model must contain one merged model.safetensors"
        )
    if any(path.is_symlink() or not path.is_file() for path in weights):
        raise CommunityModelContractError("community model weight must be a regular file")
    files = {path.name: _sha256_file(path) for path in weights}
    for name in _OUTPUT_METADATA_FILES:
        path = root / name
        if path.is_symlink():
            raise CommunityModelContractError("community model metadata must not be symlinked")
        if path.is_file():
            files[name] = _sha256_file(path)
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= files.keys():
        raise CommunityModelContractError("community model metadata inventory is incomplete")
    weight_hashes = {
        name: files[name] for name in sorted(files) if name.endswith(".safetensors")
    }
    actual = {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": canonical_json_hash(files),
    }
    if manifest.get("output_artifact") != actual:
        raise CommunityModelContractError(
            "community model files do not match the completed manifest"
        )


def _verify_manifest(
    manifest: dict[str, Any], *, expected_labels: dict[str, int]
) -> CommunityModelIdentity:
    manifest_sha256 = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(manifest_sha256, str)
        or _HEX_SHA256.fullmatch(manifest_sha256) is None
        or canonical_json_hash(unsigned) != manifest_sha256
    ):
        raise CommunityModelContractError("community training manifest hash is invalid")
    exact = {
        "schema_version": 4,
        "manifest_type": COMMUNITY_MANIFEST_TYPE,
        "status": "completed",
        "release_eligible": False,
        "attention_mode": COMMUNITY_ATTENTION_MODE,
        "fine_tuning": "lora_merged",
        "base_source_id": COMMUNITY_BASE_SOURCE_ID,
        "source_revision": COMMUNITY_BASE_SOURCE_REVISION,
        "taxonomy_version": load_taxonomy().taxonomy_version,
    }
    if any(manifest.get(key) != value for key, value in exact.items()):
        raise CommunityModelContractError(
            "community model requires a completed full-attention AIGuard24 manifest"
        )
    if manifest.get("label2id") != expected_labels:
        raise CommunityModelContractError("community manifest is not exact core-24 BIO")
    label_schema_sha256 = manifest.get("label_schema_sha256")
    if (
        not isinstance(label_schema_sha256, str)
        or label_schema_sha256 != canonical_json_hash(expected_labels)
    ):
        raise CommunityModelContractError("community manifest label schema hash is invalid")
    initialization = manifest.get("initialization")
    if not isinstance(initialization, Mapping):
        raise CommunityModelContractError("community manifest initialization audit is missing")
    conversion = initialization.get("attention_conversion")
    if (
        initialization.get("strategy") != "aiguard_bie_to_pii_zh_core24_bio_v1"
        or initialization.get("target_label_count") != 49
        or initialization.get("target_label2id") != expected_labels
        or initialization.get("release_eligible") is not False
        or not isinstance(conversion, Mapping)
        or conversion.get("source_attention_mode") != "causal"
        or conversion.get("target_attention_mode") != "full"
        or conversion.get("conversion") != "qwen3_to_qwen3_bi_shared_state_dict_v1"
        or conversion.get("attention_backend") != "sdpa"
        or conversion.get("strict_state_dict_load") is not True
        or conversion.get("newly_initialized_parameter_keys") != []
        or conversion.get("discarded_parameter_keys") != []
    ):
        raise CommunityModelContractError(
            "community manifest full-attention conversion audit is invalid"
        )
    checkpoint = manifest.get("checkpoint_selection")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("selection_split") != "validation"
        or checkpoint.get("selection_metric") != "risk_weighted_score"
    ):
        raise CommunityModelContractError(
            "community manifest lacks a completed validation checkpoint receipt"
        )
    selected_step = checkpoint.get("selected_global_step")
    selected_id = checkpoint.get("selected_checkpoint_id")
    adapter_sha256 = checkpoint.get("adapter_model_sha256")
    best_metric = checkpoint.get("best_metric")
    replay_metric = checkpoint.get("final_validation_replay_metric")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step < 1
        or selected_id != f"checkpoint-{selected_step}"
        or not isinstance(adapter_sha256, str)
        or _HEX_SHA256.fullmatch(adapter_sha256) is None
        or isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(float(best_metric))
        or isinstance(replay_metric, bool)
        or not isinstance(replay_metric, (int, float))
        or not math.isfinite(float(replay_metric))
        or not math.isclose(
            float(best_metric), float(replay_metric), rel_tol=1.0e-6, abs_tol=1.0e-8
        )
    ):
        raise CommunityModelContractError(
            "community validation checkpoint receipt is malformed"
        )
    return CommunityModelIdentity(
        schema_version=COMMUNITY_MODEL_IDENTITY_SCHEMA_VERSION,
        manifest_type=COMMUNITY_MANIFEST_TYPE,
        manifest_sha256=manifest_sha256,
        model_type=COMMUNITY_MODEL_TYPE,
        attention_mode=COMMUNITY_ATTENTION_MODE,
        architecture_version=COMMUNITY_ARCHITECTURE_VERSION,
        taxonomy_version=exact["taxonomy_version"],
        label_count=49,
        label_schema_sha256=label_schema_sha256,
    )


def _verify_config(config: Mapping[str, Any], *, expected_labels: dict[str, int]) -> None:
    expected_id2label = {index: label for label, index in expected_labels.items()}
    if (
        config.get("model_type") != COMMUNITY_MODEL_TYPE
        or config.get("pii_attention_mode") != COMMUNITY_ATTENTION_MODE
        or config.get("architectures") != [COMMUNITY_ARCHITECTURE]
        or config.get("architecture_version") != COMMUNITY_ARCHITECTURE_VERSION
        or config.get("bi_attention_backend") != "sdpa"
        or config.get("use_cache") is not False
        or config.get("pii_training_status") != COMMUNITY_TRAINING_STATUS
        or config.get("pii_taxonomy_version") != load_taxonomy().taxonomy_version
        or config.get("pii_release_eligible") is not False
    ):
        raise CommunityModelContractError(
            "community model must be qwen3_bi/full with the completed candidate marker"
        )
    if config.get("label2id") != expected_labels or _normalized_id2label(
        config.get("id2label")
    ) != expected_id2label:
        raise CommunityModelContractError("community model config is not exact core-24 BIO")
    num_labels = config.get("num_labels")
    if num_labels is not None and num_labels != 49:
        raise CommunityModelContractError("community model config label count must be 49")


def verify_community_model_artifact(model_path: str | Path) -> VerifiedCommunityModel:
    """Verify and return one path-free-identified community model directory."""

    candidate = Path(model_path).expanduser()
    if candidate.is_symlink():
        raise CommunityModelContractError("community model root must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CommunityModelContractError(
            "community model path must be an existing local directory"
        ) from exc
    if not root.is_dir():
        raise CommunityModelContractError(
            "community model path must be an existing local directory"
        )
    forbidden = [
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pt", "*.pth")
        for item in root.glob(pattern)
    ]
    if forbidden or any(root.glob("adapter_*")):
        raise CommunityModelContractError(
            "community model root must be merged safetensors-only"
        )
    labels = expected_core24_label2id()
    manifest = _json_object(root / "training_manifest.json", kind="training manifest")
    identity = _verify_manifest(manifest, expected_labels=labels)
    config = _json_object(root / "config.json", kind="config")
    _verify_config(config, expected_labels=labels)
    _verify_output_artifact(root, manifest)
    return VerifiedCommunityModel(root=root, identity=identity)


__all__ = [
    "COMMUNITY_ARCHITECTURE",
    "COMMUNITY_ARCHITECTURE_VERSION",
    "COMMUNITY_ATTENTION_MODE",
    "COMMUNITY_BASE_SOURCE_ID",
    "COMMUNITY_BASE_SOURCE_REVISION",
    "COMMUNITY_MANIFEST_TYPE",
    "COMMUNITY_MODEL_IDENTITY_SCHEMA_VERSION",
    "COMMUNITY_MODEL_TYPE",
    "COMMUNITY_TRAINING_STATUS",
    "CommunityModelContractError",
    "CommunityModelIdentity",
    "VerifiedCommunityModel",
    "canonical_json_hash",
    "expected_core24_label2id",
    "verify_community_model_artifact",
]
