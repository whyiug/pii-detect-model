"""Privacy-safe bindings for test-informed Qwen evaluation on tw-PII-bench.

The v3 protocol remains immutable and was frozen for an earlier MacBERT
checkpoint.  These helpers reuse only its pinned source, label projection, and
metric implementation.  A later Qwen checkpoint is bound separately and must
therefore be reported as posthoc diagnostic evidence, never as a confirmatory
v3 result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pii_zh.data.tw_pii_bench import TwPiiBenchSourceError, verify_source_manifest
from pii_zh.evaluation.provenance import canonical_json_hash, sha256_file
from pii_zh.evaluation.tw_pii_protocol import EFFECTIVE_8_LABELS, PROJECT_TO_EFFECTIVE_8
from pii_zh.training.manifest import verify_output_artifact_binding, verify_training_manifest

_PINNED_IMPLEMENTATIONS = (
    "src/pii_zh/data/tw_pii_bench.py",
    "src/pii_zh/evaluation/tw_pii_protocol.py",
)
_EXPECTED_MODEL_TYPES = {"jpt": "qwen3", "full": "qwen3_bi"}


class TwPiiPosthocError(RuntimeError):
    """Raised without reflecting local paths, source text, or entity values."""


def load_self_hashed_json(path: Path, *, kind: str) -> dict[str, Any]:
    """Load a regular, self-hashed JSON object with path-free errors."""

    if path.is_symlink() or not path.is_file():
        raise TwPiiPosthocError(f"{kind} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TwPiiPosthocError(f"{kind} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TwPiiPosthocError(f"{kind} must be a JSON object")
    claimed = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or canonical_json_hash(unsigned) != claimed:
        raise TwPiiPosthocError(f"{kind} self-hash does not verify")
    return value


def _load_json_object(path: Path, *, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TwPiiPosthocError(f"{kind} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TwPiiPosthocError(f"{kind} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TwPiiPosthocError(f"{kind} must be a JSON object")
    return value


def verify_v3_metric_contract(
    *,
    repository_root: Path,
    source_manifest_path: Path,
    protocol_freeze_path: Path,
) -> dict[str, Any]:
    """Verify the reusable v3 source/projection/metric contract.

    The v3 model binding is deliberately not reused: it identifies the earlier
    MacBERT checkpoint.  Returned metadata calls this distinction out so a
    downstream report cannot silently claim v3 preregistration.
    """

    source = load_self_hashed_json(source_manifest_path, kind="source manifest")
    try:
        verify_source_manifest(source)
    except TwPiiBenchSourceError as exc:
        raise TwPiiPosthocError("source manifest violates the frozen contract") from exc
    freeze = load_self_hashed_json(protocol_freeze_path, kind="v3 protocol freeze")
    if (
        freeze.get("manifest_type") != "tw_pii_bench_v3_protocol_preregistration"
        or freeze.get("status") != "frozen_before_project_model_inference"
    ):
        raise TwPiiPosthocError("v3 protocol freeze identity is invalid")
    source_binding = freeze.get("source_manifest")
    if not isinstance(source_binding, Mapping) or source_binding != {
        "file_sha256": sha256_file(source_manifest_path),
        "manifest_sha256": source["manifest_sha256"],
    }:
        raise TwPiiPosthocError("source manifest does not match the v3 freeze")

    protocol = freeze.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("effective_labels") != list(EFFECTIVE_8_LABELS)
        or protocol.get("project_to_effective")
        != dict(sorted(PROJECT_TO_EFFECTIVE_8.items()))
        or protocol.get("threshold_selection") != "none"
    ):
        raise TwPiiPosthocError("v3 label or threshold contract changed")
    frozen_implementations = freeze.get("implementation_files")
    if not isinstance(frozen_implementations, Mapping):
        raise TwPiiPosthocError("v3 implementation bindings are missing")
    implementation_sha256: dict[str, str] = {}
    root = repository_root.resolve(strict=True)
    for relative in _PINNED_IMPLEMENTATIONS:
        candidate = (root / relative).resolve(strict=True)
        if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
            raise TwPiiPosthocError("pinned implementation path is invalid")
        actual = sha256_file(candidate)
        if frozen_implementations.get(relative) != actual:
            raise TwPiiPosthocError("pinned v3 metric implementation changed")
        implementation_sha256[relative] = actual
    return {
        "source_manifest_sha256": source["manifest_sha256"],
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "v3_protocol_freeze_sha256": freeze["manifest_sha256"],
        "v3_protocol_freeze_file_sha256": sha256_file(protocol_freeze_path),
        "effective_labels": list(EFFECTIVE_8_LABELS),
        "project_to_effective": dict(sorted(PROJECT_TO_EFFECTIVE_8.items())),
        "implementation_sha256": implementation_sha256,
        "v3_model_binding_reused": False,
        "confirmatory_status_inherited": False,
    }


def _entity_labels(label2id: Mapping[str, Any]) -> tuple[str, ...]:
    if label2id.get("O") != 0:
        raise TwPiiPosthocError("model label schema must contain O at id zero")
    entities = {
        label[2:]
        for label in label2id
        if isinstance(label, str) and label.startswith(("B-", "I-"))
    }
    if not entities or any(
        f"B-{entity}" not in label2id or f"I-{entity}" not in label2id
        for entity in entities
    ):
        raise TwPiiPosthocError("model BIO label inventory is incomplete")
    missing_projection_sources = set(PROJECT_TO_EFFECTIVE_8) - entities
    if missing_projection_sources:
        raise TwPiiPosthocError("model lacks labels required by the effective-eight projection")
    return tuple(sorted(entities))


def verify_completed_qwen_model(model_path: Path) -> dict[str, Any]:
    """Byte-bind one completed JPT/Full Qwen artifact without loading weights."""

    if model_path.is_symlink() or not model_path.is_dir():
        raise TwPiiPosthocError("model root must be a regular directory")
    model = model_path.resolve(strict=True)
    manifest_path = model / "training_manifest.json"
    manifest = load_self_hashed_json(manifest_path, kind="training manifest")
    if not verify_training_manifest(manifest) or manifest.get("status") != "completed":
        raise TwPiiPosthocError("training manifest is not completed")
    if not verify_output_artifact_binding(manifest, model, require_single_weight=True):
        raise TwPiiPosthocError("model files do not match the completed training manifest")
    attention_mode = manifest.get("attention_mode")
    if attention_mode not in _EXPECTED_MODEL_TYPES:
        raise TwPiiPosthocError("only completed Qwen JPT or Full artifacts are accepted")
    config = _load_json_object(model / "config.json", kind="model config")
    if (
        config.get("pii_attention_mode") != attention_mode
        or config.get("model_type") != _EXPECTED_MODEL_TYPES[attention_mode]
    ):
        raise TwPiiPosthocError("model config and training attention mode disagree")
    if attention_mode == "jpt" and (
        config.get("pii_release_eligible") is not False
        or isinstance(config.get("pii_jpt_sep_token_id"), bool)
        or not isinstance(config.get("pii_jpt_sep_token_id"), int)
    ):
        raise TwPiiPosthocError("JPT model config lacks its research/separator contract")
    manifest_labels = manifest.get("label2id")
    config_labels = config.get("label2id")
    if (
        not isinstance(manifest_labels, Mapping)
        or not isinstance(config_labels, Mapping)
        or dict(manifest_labels) != dict(config_labels)
    ):
        raise TwPiiPosthocError("model config and training label schemas disagree")
    if manifest.get("label_schema_sha256") != canonical_json_hash(dict(manifest_labels)):
        raise TwPiiPosthocError("training label schema content hash does not verify")
    native_entities = _entity_labels(manifest_labels)
    output_artifact = manifest.get("output_artifact")
    if not isinstance(output_artifact, Mapping):  # already guarded by binding verification
        raise TwPiiPosthocError("training manifest lacks output artifact identity")
    return {
        "model_type": config["model_type"],
        "attention_mode": attention_mode,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "training_manifest_file_sha256": sha256_file(manifest_path),
        "output_artifact_sha256": canonical_json_hash(output_artifact),
        "label_schema_sha256": manifest.get("label_schema_sha256"),
        "native_entity_labels": list(native_entities),
        "full_attention_architecture": attention_mode == "full",
        "release_candidate_asserted": False,
    }


def inference_implementation_identity(repository_root: Path) -> dict[str, str]:
    """Hash every implementation component that can change inference bytes."""

    root = repository_root.resolve(strict=True)
    components = {
        "bio_decoder": root / "src/pii_zh/models/decoding.py",
        "contract": root / "src/pii_zh/evaluation/tw_pii_posthoc.py",
        "cross_window_deduplication": root / "src/pii_zh/presidio/qwen_recognizer.py",
        "qwen_recognizer": root / "src/pii_zh/presidio/qwen_recognizer.py",
        "runner": root / "scripts/run_tw_pii_completed_qwen_inference.py",
        "token_chunker": root / "src/pii_zh/presidio/token_chunker.py",
        "transformers_predictor": root / "src/pii_zh/inference/transformers_predictor.py",
    }
    if any(
        path.is_symlink() or not path.is_file() or root not in path.resolve().parents
        for path in components.values()
    ):
        raise TwPiiPosthocError("inference implementation component is invalid")
    return {name: sha256_file(path) for name, path in sorted(components.items())}


def evaluation_implementation_identity(repository_root: Path) -> dict[str, str]:
    """Hash posthoc scoring code without exposing local or repository paths."""

    root = repository_root.resolve(strict=True)
    components = {
        "contract": root / "src/pii_zh/evaluation/tw_pii_posthoc.py",
        "evaluator": root / "scripts/evaluate_tw_pii_completed_qwen_posthoc.py",
        "metric": root / "src/pii_zh/evaluation/tw_pii_protocol.py",
    }
    if any(
        path.is_symlink() or not path.is_file() or root not in path.resolve().parents
        for path in components.values()
    ):
        raise TwPiiPosthocError("evaluation implementation component is invalid")
    return {name: sha256_file(path) for name, path in sorted(components.items())}


def assert_public_aggregate(payload: Mapping[str, Any]) -> None:
    """Reject common record-level or local-path leaks before report writing."""

    forbidden_keys = {"doc_id", "record_id", "text", "entity_text", "entity_value"}

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in forbidden_keys:
                    raise TwPiiPosthocError("public report contains a record-level field")
                visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith("/"):
            raise TwPiiPosthocError("public report contains an absolute path")

    visit(payload)


__all__ = [
    "TwPiiPosthocError",
    "assert_public_aggregate",
    "evaluation_implementation_identity",
    "inference_implementation_identity",
    "load_self_hashed_json",
    "verify_completed_qwen_model",
    "verify_v3_metric_contract",
]
