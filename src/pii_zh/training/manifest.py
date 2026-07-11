"""Privacy-safe, content-addressed training manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pii_zh.tokenization import BOUNDARY_MODE_CONFIG_KEY
from pii_zh.training.config import TrainingConfig
from pii_zh.training.data import TrainingDataSummary
from pii_zh.training.loading import BackboneLoadingAudit, InitializationAudit

MANIFEST_SCHEMA_VERSION = 3
OUTPUT_ARTIFACT_SCHEMA_VERSION = 1
OUTPUT_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_revision(worktree: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if len(revision) == 40 else None


def tokenizer_fingerprint(
    checkpoint: str | Path,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Hash source files and the effective runtime tokenizer graph.

    The effective graph matters because the PII model intentionally replaces
    Qwen's pre-tokenizer with a character-boundary-safe view before training.
    No local path or vocabulary content is copied into the manifest.
    """

    root = Path(checkpoint)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    )
    files = {name: sha256_file(root / name) for name in names if (root / name).is_file()}
    result: dict[str, Any] = {
        "files": files,
        "source_files_combined_sha256": canonical_json_hash(files),
    }
    if tokenizer is not None:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or not hasattr(backend, "to_str"):
            raise ValueError("effective tokenizer lacks a serializable fast backend")
        backend_json = backend.to_str()
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        boundary_mode = (
            init_kwargs.get(BOUNDARY_MODE_CONFIG_KEY) if isinstance(init_kwargs, dict) else None
        )
        effective = {
            "backend_sha256": hashlib.sha256(backend_json.encode("utf-8")).hexdigest(),
            "boundary_mode": boundary_mode,
            "vocab_size": int(len(tokenizer)),
            "special_token_ids": {
                name: getattr(tokenizer, name, None)
                for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
            },
        }
        result["effective"] = effective
        result["effective_contract_sha256"] = canonical_json_hash(effective)
    return result


def training_data_summary_dict(summary: TrainingDataSummary) -> dict[str, Any]:
    """Return the deterministic, privacy-safe data summary stored in manifests."""

    return {
        "document_count": summary.document_count,
        "entity_count": summary.entity_count,
        "token_count": summary.token_count,
        "pii_free_document_count": summary.pii_free_document_count,
        "unalignable_boundary_count": summary.unalignable_boundary_count,
        "label_counts": summary.label_counts,
        "split_counts": summary.split_counts,
        "data_pool_counts": summary.data_pool_counts,
        "quality_tier_counts": summary.quality_tier_counts,
        "quality_gate_passed_document_count": summary.quality_gate_passed_document_count,
        "validators_passed_document_count": summary.validators_passed_document_count,
        "public_weight_training_allowed_document_count": (
            summary.public_weight_training_allowed_document_count
        ),
        "sources": [
            {
                "source_id": source.source_id,
                "source_kind": source.source_kind,
                "revision": source.source_revision,
                "license": source.license,
                "data_pool": source.data_pool,
                "split": source.split,
                "document_count": source.document_count,
                "entity_count": source.entity_count,
                "quality_tier_counts": source.quality_tier_counts,
                "quality_gate_passed_document_count": (source.quality_gate_passed_document_count),
                "validators_passed_document_count": (source.validators_passed_document_count),
                "public_weight_training_allowed_document_count": (
                    source.public_weight_training_allowed_document_count
                ),
            }
            for source in summary.sources
        ],
    }


def output_artifact_fingerprint(output_dir: str | Path) -> dict[str, Any]:
    """Hash standalone model/config/tokenizer files without recording a path."""

    root = Path(output_dir)
    weights = tuple(sorted(root.glob("model*.safetensors")))
    if not weights:
        raise ValueError("output artifact contains no model safetensors")
    if any(path.is_symlink() or not path.is_file() for path in weights):
        raise ValueError("output artifact weights must be regular files")
    files = {path.name: sha256_file(path) for path in weights}
    for name in OUTPUT_METADATA_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError("output artifact metadata must not contain symlinks")
        if path.is_file():
            files[name] = sha256_file(path)
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= files.keys():
        raise ValueError("output artifact is missing required model/tokenizer metadata")
    weight_hashes = {name: files[name] for name in sorted(files) if name.endswith(".safetensors")}
    return {
        "schema_version": OUTPUT_ARTIFACT_SCHEMA_VERSION,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": canonical_json_hash(files),
    }


def verify_output_artifact_binding(
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    require_single_weight: bool = False,
) -> bool:
    """Verify the completed manifest is byte-bound to the supplied model root."""

    if not verify_training_manifest(manifest) or manifest.get("status") != "completed":
        return False
    recorded = manifest.get("output_artifact")
    if not isinstance(recorded, dict):
        return False
    try:
        actual = output_artifact_fingerprint(output_dir)
    except (OSError, ValueError):
        return False
    if require_single_weight and actual.get("weight_files") != ["model.safetensors"]:
        return False
    return recorded == actual


def finalize_training_manifest(
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    completed_at: str | None = None,
    runtime_environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically finalizable manifest value bound to the completed checkpoint."""

    if not verify_training_manifest(manifest):
        raise ValueError("training manifest hash is missing or invalid")
    finalized = dict(manifest)
    finalized.pop("manifest_sha256", None)
    finalized["status"] = "completed"
    finalized["completed_at"] = completed_at or datetime.now(timezone.utc).isoformat()
    finalized["output_artifact"] = output_artifact_fingerprint(output_dir)
    if runtime_environment is not None:
        finalized["runtime_environment"] = dict(runtime_environment)
    finalized["manifest_sha256"] = canonical_json_hash(finalized)
    return finalized


def build_training_manifest(
    *,
    config: TrainingConfig,
    loading_audit: BackboneLoadingAudit,
    train_summary: TrainingDataSummary,
    validation_summary: TrainingDataSummary,
    taxonomy_version: str,
    label2id: dict[str, int],
    tokenizer: Any | None = None,
    initialization_audit: InitializationAudit | None = None,
    created_at: str | None = None,
    worktree: str | Path = ".",
) -> dict[str, Any]:
    """Build a manifest containing hashes/counts but no raw text or entity values."""

    if (config.initial_model is None) != (initialization_audit is None):
        raise ValueError("training config and initialization audit presence disagree")
    if (
        initialization_audit is not None
        and initialization_audit.strategy != config.initialization_strategy
    ):
        raise ValueError("training config and initialization audit strategies disagree")

    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    recipe = config.manifest_recipe()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "in_progress",
        "created_at": timestamp,
        "seed": config.seed,
        "attention_mode": config.attention_mode,
        "fine_tuning": config.fine_tuning,
        "base_source_id": config.base_source_id,
        "recipe": recipe,
        "recipe_sha256": canonical_json_hash(recipe),
        "taxonomy_version": taxonomy_version,
        "label2id": dict(sorted(label2id.items(), key=lambda item: item[1])),
        "label_schema_sha256": canonical_json_hash(label2id),
        "base_checkpoint": {
            "config_sha256": loading_audit.config_sha256,
            "weights_sha256": loading_audit.weights_sha256,
            "safetensor_files": list(loading_audit.safetensor_files),
            "loading_audit": loading_audit.to_dict(),
        },
        "initialization": (
            initialization_audit.to_dict()
            if initialization_audit is not None
            else {"strategy": config.initialization_strategy}
        ),
        "tokenizer": tokenizer_fingerprint(config.base_model, tokenizer=tokenizer),
        "datasets": {
            "train": {
                "sha256": sha256_file(config.train_file),
                "summary": training_data_summary_dict(train_summary),
            },
            "validation": {
                "sha256": sha256_file(config.validation_file),
                "summary": training_data_summary_dict(validation_summary),
            },
        },
        "versions": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "accelerate": _package_version("accelerate"),
            "peft": _package_version("peft") if config.fine_tuning == "lora" else None,
            "pii_zh_qwen": _package_version("pii-zh-qwen"),
        },
        "code_revision": _git_revision(Path(worktree)),
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "trainer_reporting_integrations": [],
        },
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    return manifest


def verify_training_manifest(manifest: dict[str, Any]) -> bool:
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return isinstance(expected, str) and expected == canonical_json_hash(unsigned)


def write_training_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    """Atomically write ``training_manifest.json`` after self-hash verification."""

    if not verify_training_manifest(manifest):
        raise ValueError("training manifest hash is missing or invalid")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "training_manifest.json"
    temporary = directory / ".training_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
