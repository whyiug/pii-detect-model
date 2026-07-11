#!/usr/bin/env python3
"""Build an atomic, local-only Hugging Face model release directory.

Only an explicit allowlist of files is copied.  The command does not download,
train, upload, or invoke Hugging Face APIs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
OPTIONAL_CHECKPOINT_FILES = ("added_tokens.json",)
EVIDENCE_FILES = (
    "taxonomy.yaml",
    "id2label.json",
    "calibration.json",
    "thresholds.yaml",
    "training_manifest.json",
    "data_provenance.json",
    "teacher_provenance.json",
    "evaluation_report.json",
    "model-index.yml",
    "sbom.cdx.json",
)
PROJECT_FILES = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
    "SECURITY.md": "SECURITY.md",
}
REMOTE_CODE_FILES = ("configuration_qwen3_bi.py", "modeling_qwen3_bi.py")
REQUIRED_RELEASE_FILES = frozenset(
    {
        "README.md",
        *CHECKPOINT_FILES,
        *EVIDENCE_FILES,
        *PROJECT_FILES.values(),
        *REMOTE_CODE_FILES,
        "checksums.txt",
    }
)
ALLOWED_RELEASE_FILES = REQUIRED_RELEASE_FILES | frozenset(OPTIONAL_CHECKPOINT_FILES)
FORBIDDEN_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".joblib", ".npy", ".npz", ".pickle", ".pkl", ".pt", ".pth"}
)
RAW_DATA_SUFFIXES = frozenset({".arrow", ".csv", ".jsonl", ".parquet", ".tsv"})
FORBIDDEN_NAME_PARTS = (
    "optimizer",
    "scheduler",
    "trainer_state",
    "rng_state",
    "gradient",
    "tfevents",
    "api_prompt",
    "api_response",
    "customer_data",
)
AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
    "AutoModelForTokenClassification": ("modeling_qwen3_bi.Qwen3BiForTokenClassification"),
}
BOUNDARY_MODE_CONFIG_KEY = "pii_zh_boundary_mode"
BOUNDARY_MODE = "unicode_codepoint_v1"
BOUND_ARTIFACT_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)


class ReleaseBuildError(ValueError):
    """Raised when an input is unsafe or incomplete."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def output_artifact_fingerprint(root: Path) -> dict[str, Any]:
    weights = tuple(sorted(root.glob("model*.safetensors")))
    files = {path.name: sha256_file(path) for path in weights if path.is_file()}
    for name in BOUND_ARTIFACT_METADATA_FILES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            files[name] = sha256_file(path)
    weight_hashes = {name: files[name] for name in sorted(files) if name.endswith(".safetensors")}
    return {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": _canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": _canonical_json_hash(files),
    }


def validate_training_artifact_binding(manifest_path: Path, checkpoint_dir: Path) -> None:
    training = _load_json_object(manifest_path, "training manifest")
    expected = training.get("manifest_sha256")
    unsigned = dict(training)
    unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(expected, str)
        or _canonical_json_hash(unsigned) != expected
        or training.get("status") != "completed"
    ):
        raise ReleaseBuildError("training manifest completion/hash check failed")
    actual = output_artifact_fingerprint(checkpoint_dir)
    if actual.get("weight_files") != ["model.safetensors"]:
        raise ReleaseBuildError("release checkpoint must contain exactly model.safetensors")
    if training.get("output_artifact") != actual:
        raise ReleaseBuildError("checkpoint files do not match the completed training manifest")


def _require_regular_file(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise ReleaseBuildError(f"{description} must not be a symlink: {path}")
    if not path.is_file():
        raise ReleaseBuildError(f"missing {description}: {path}")
    return path


def _reject_contaminated_tree(root: Path, description: str) -> None:
    if not root.is_dir():
        raise ReleaseBuildError(f"{description} is not a directory: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        lowered_name = path.name.lower()
        if path.is_symlink():
            raise ReleaseBuildError(f"{description} contains symlink: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ReleaseBuildError(
                f"{description} contains forbidden pickle/checkpoint file: {relative}"
            )
        if path.suffix.lower() in RAW_DATA_SUFFIXES:
            raise ReleaseBuildError(f"{description} contains raw/tabular data file: {relative}")
        if any(fragment in lowered_name for fragment in FORBIDDEN_NAME_PARTS):
            raise ReleaseBuildError(
                f"{description} contains forbidden training/private artifact: {relative}"
            )
        if any(part.lower() in {"raw", "raw_data"} for part in relative.parts[:-1]):
            raise ReleaseBuildError(f"{description} contains a raw-data directory: {relative}")


def validate_safetensors(path: Path) -> None:
    """Perform a dependency-free structural validation of a safetensors file."""

    size = path.stat().st_size
    if size < 10:
        raise ReleaseBuildError("model.safetensors is too small to contain a valid header")
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length < 2 or header_length > size - 8 or header_length > 100 * 1024 * 1024:
            raise ReleaseBuildError("model.safetensors has an invalid header length")
        raw_header = stream.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError("model.safetensors header is not valid UTF-8 JSON") from exc
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ReleaseBuildError("model.safetensors contains no tensors")
    data_length = size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in tensors.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ReleaseBuildError("model.safetensors tensor metadata is malformed")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_length
        ):
            raise ReleaseBuildError(f"model.safetensors has invalid data offsets for {name!r}")
        if not isinstance(metadata.get("shape"), list) or not isinstance(
            metadata.get("dtype"), str
        ):
            raise ReleaseBuildError(f"model.safetensors has incomplete metadata for {name!r}")
        intervals.append((offsets[0], offsets[1], name))
    expected_start = 0
    for start, end, name in sorted(intervals):
        if start != expected_start:
            raise ReleaseBuildError(
                f"model.safetensors has overlapping or unreferenced data before {name!r}"
            )
        expected_start = end
    if expected_start != data_length:
        raise ReleaseBuildError("model.safetensors has trailing or unreferenced tensor data")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"{description} must be a JSON object: {path}")
    return value


def _normalized_id2label(path: Path) -> dict[str, str]:
    document = _load_json_object(path, "id2label")
    normalized: dict[str, str] = {}
    for key, value in document.items():
        if not isinstance(value, str) or not value.strip():
            raise ReleaseBuildError("id2label values must be non-empty strings")
        try:
            numeric_key = int(key)
        except (TypeError, ValueError) as exc:
            raise ReleaseBuildError("id2label keys must be integer-like") from exc
        if numeric_key < 0 or str(numeric_key) in normalized:
            raise ReleaseBuildError("id2label keys must be unique non-negative integers")
        normalized[str(numeric_key)] = value
    expected_keys = [str(index) for index in range(len(normalized))]
    if sorted(normalized, key=int) != expected_keys:
        raise ReleaseBuildError("id2label keys must be contiguous and start at zero")
    if len(set(normalized.values())) != len(normalized):
        raise ReleaseBuildError("id2label labels must be unique")
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def build_release_config(config_path: Path, id2label_path: Path) -> dict[str, Any]:
    config = _load_json_object(config_path, "checkpoint config")
    if config.get("model_type") != "qwen3_bi":
        raise ReleaseBuildError(
            "checkpoint model_type must already be qwen3_bi; causal checkpoints cannot "
            "be converted during release"
        )
    if config.get("architectures") != ["Qwen3BiForTokenClassification"]:
        raise ReleaseBuildError(
            "checkpoint architectures must already identify Qwen3BiForTokenClassification"
        )
    if config.get("pii_attention_mode") != "full" or config.get("pii_release_eligible") is not True:
        raise ReleaseBuildError(
            "checkpoint must explicitly declare full attention and release eligibility"
        )
    id2label = _normalized_id2label(id2label_path)
    existing = config.get("id2label")
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or {str(key): value for key, value in existing.items()} != id2label
        ):
            raise ReleaseBuildError(
                "checkpoint config id2label disagrees with release id2label.json"
            )
    existing_num_labels = config.get("num_labels")
    if existing_num_labels is not None and existing_num_labels != len(id2label):
        raise ReleaseBuildError("checkpoint config num_labels disagrees with id2label.json")

    expected_values = {
        "architectures": ["Qwen3BiForTokenClassification"],
        "auto_map": dict(AUTO_MAP),
        "id2label": id2label,
        "label2id": {label: int(index) for index, label in id2label.items()},
        "model_type": "qwen3_bi",
        "use_cache": False,
    }
    mismatched = sorted(key for key, value in expected_values.items() if config.get(key) != value)
    if mismatched:
        raise ReleaseBuildError(
            "checkpoint config is not already release-ready; mismatched fields: "
            + ", ".join(mismatched)
        )
    return config


def validate_bidirectional_release_contract(
    *,
    training_manifest_path: Path,
    tokenizer_config_path: Path,
    checkpoint_dir: Path,
) -> None:
    """Bind release inputs to the trained full-attention tokenizer contract."""

    training = _load_json_object(training_manifest_path, "training manifest")
    if training.get("attention_mode") != "full":
        raise ReleaseBuildError("training_manifest attention_mode must be full for publication")
    tokenizer_config = _load_json_object(tokenizer_config_path, "tokenizer config")
    if tokenizer_config.get(BOUNDARY_MODE_CONFIG_KEY) != BOUNDARY_MODE:
        raise ReleaseBuildError("tokenizer config is missing the approved character-boundary mode")
    tokenizer_evidence = training.get("tokenizer")
    effective = (
        tokenizer_evidence.get("effective") if isinstance(tokenizer_evidence, dict) else None
    )
    if not isinstance(effective, dict) or effective.get("boundary_mode") != BOUNDARY_MODE:
        raise ReleaseBuildError(
            "training manifest is not bound to the approved character-boundary tokenizer"
        )
    validate_training_artifact_binding(training_manifest_path, checkpoint_dir)


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ReleaseBuildError("could not extract reviewed remote-code definition")
    return segment


def render_remote_code(source_path: Path) -> tuple[str, str]:
    """Split the repository implementation into HF configuration/model modules."""

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = {
        getattr(node, "name", ""): node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef))
    }
    constants: dict[str, ast.AST] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "ARCHITECTURE_VERSION",
                "SUPPORTED_ATTENTION_BACKENDS",
            }:
                constants[target.id] = node
    required_definitions = {
        "Qwen3BiConfig",
        "build_full_attention_mask",
        "convert_qwen3_token_classifier_state_dict",
        "Qwen3BiForTokenClassification",
    }
    missing = sorted(required_definitions - definitions.keys())
    missing_constants = sorted(
        {"ARCHITECTURE_VERSION", "SUPPORTED_ATTENTION_BACKENDS"} - constants.keys()
    )
    if missing or missing_constants:
        raise ReleaseBuildError(
            "remote-code source is missing reviewed definitions: "
            + ", ".join([*missing, *missing_constants])
        )

    banner = (
        '"""Generated from src/pii_zh/models/qwen3_bi.py; do not edit in the model repo."""\n\n'
        "from __future__ import annotations\n\n"
    )
    configuration = banner + "from typing import Any\n\nfrom transformers import Qwen3Config\n\n"
    configuration += _source_segment(source, constants["SUPPORTED_ATTENTION_BACKENDS"]) + "\n"
    configuration += _source_segment(source, constants["ARCHITECTURE_VERSION"]) + "\n\n"
    configuration += _source_segment(source, definitions["Qwen3BiConfig"]) + "\n"

    modeling = banner
    modeling += "from collections.abc import Mapping\nfrom typing import Any\n\n"
    modeling += "import torch\nfrom torch import nn\n"
    modeling += "from transformers import Qwen3Config, Qwen3Model\n"
    modeling += "from transformers.modeling_outputs import TokenClassifierOutput\n"
    modeling += "from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel\n\n"
    modeling += (
        "from .configuration_qwen3_bi import (\n"
        "    Qwen3BiConfig,\n"
        "    SUPPORTED_ATTENTION_BACKENDS,\n"
        ")\n\n"
    )
    for name in (
        "build_full_attention_mask",
        "convert_qwen3_token_classifier_state_dict",
        "Qwen3BiForTokenClassification",
    ):
        modeling += _source_segment(source, definitions[name]) + "\n\n"

    for name, rendered in (
        ("configuration_qwen3_bi.py", configuration),
        ("modeling_qwen3_bi.py", modeling),
    ):
        try:
            compile(rendered, name, "exec")
        except SyntaxError as exc:
            raise ReleaseBuildError(f"generated {name} is invalid Python") from exc
    return configuration, modeling


def write_checksums(directory: Path) -> Path:
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "checksums.txt"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}" for path in files]
    destination = directory / "checksums.txt"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _copy_file(source: Path, destination: Path, description: str) -> None:
    _require_regular_file(source, description)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)


def build_release(
    *,
    checkpoint_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    model_card: Path,
    remote_code_source: Path,
    repository_root: Path = REPOSITORY_ROOT,
    project_file_overrides: dict[str, Path] | None = None,
    force: bool = False,
) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    evidence_dir = evidence_dir.resolve()
    output_dir = output_dir.resolve()
    repository_root = repository_root.resolve()
    protected_paths = {
        Path(output_dir.anchor),
        Path.home().resolve(),
        checkpoint_dir,
        evidence_dir,
        repository_root,
    }
    if output_dir in protected_paths or output_dir in checkpoint_dir.parents:
        raise ReleaseBuildError(f"refusing unsafe output directory: {output_dir}")
    if output_dir in evidence_dir.parents or output_dir in repository_root.parents:
        raise ReleaseBuildError(f"refusing unsafe output directory: {output_dir}")
    if checkpoint_dir in output_dir.parents or evidence_dir in output_dir.parents:
        raise ReleaseBuildError("output directory must not be nested inside checkpoint/evidence")
    if repository_root / ".git" in {output_dir, *output_dir.parents}:
        raise ReleaseBuildError("output directory must not be inside repository metadata")
    if force and repository_root / "release" not in output_dir.parents:
        raise ReleaseBuildError(
            "--force is restricted to a child of the repository release directory"
        )
    _reject_contaminated_tree(checkpoint_dir, "checkpoint directory")
    _reject_contaminated_tree(evidence_dir, "evidence directory")

    checkpoint_paths = {
        name: _require_regular_file(checkpoint_dir / name, f"checkpoint file {name}")
        for name in CHECKPOINT_FILES
    }
    optional_checkpoint_paths = {
        name: _require_regular_file(checkpoint_dir / name, f"checkpoint file {name}")
        for name in OPTIONAL_CHECKPOINT_FILES
        if (checkpoint_dir / name).exists()
    }
    evidence_paths = {
        name: _require_regular_file(evidence_dir / name, f"release evidence {name}")
        for name in EVIDENCE_FILES
    }
    _require_regular_file(model_card, "model card")
    _require_regular_file(remote_code_source, "Qwen3Bi remote-code source")
    validate_safetensors(checkpoint_paths["model.safetensors"])
    validate_bidirectional_release_contract(
        training_manifest_path=evidence_paths["training_manifest.json"],
        tokenizer_config_path=checkpoint_paths["tokenizer_config.json"],
        checkpoint_dir=checkpoint_dir,
    )
    build_release_config(checkpoint_paths["config.json"], evidence_paths["id2label.json"])
    configuration_code, modeling_code = render_remote_code(remote_code_source)

    if output_dir.exists() and not force:
        raise ReleaseBuildError(f"output already exists (use --force to replace it): {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _copy_file(model_card, staging / "README.md", "model card")
        for name, source in checkpoint_paths.items():
            _copy_file(source, staging / name, f"checkpoint file {name}")
        for name, source in optional_checkpoint_paths.items():
            _copy_file(source, staging / name, f"checkpoint file {name}")
        for name, source in evidence_paths.items():
            _copy_file(source, staging / name, f"release evidence {name}")
        overrides = project_file_overrides or {}
        for source_name, destination_name in PROJECT_FILES.items():
            source = overrides.get(destination_name, repository_root / source_name)
            _copy_file(
                source,
                staging / destination_name,
                f"project file {source_name}",
            )
        (staging / "configuration_qwen3_bi.py").write_text(configuration_code, encoding="utf-8")
        (staging / "modeling_qwen3_bi.py").write_text(modeling_code, encoding="utf-8")
        write_checksums(staging)
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        expected = REQUIRED_RELEASE_FILES | optional_checkpoint_paths.keys()
        if actual != expected:
            raise ReleaseBuildError(
                "internal error: release file set differs from the required allowlist: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        if output_dir.exists():
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise ReleaseBuildError(f"refusing to replace unsafe output path: {output_dir}")
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-card", type=Path, default=REPOSITORY_ROOT / "model_cards/README.md"
    )
    parser.add_argument(
        "--remote-code-source",
        type=Path,
        default=REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--license-file", type=Path)
    parser.add_argument("--notice-file", type=Path)
    parser.add_argument("--third-party-notices", type=Path)
    parser.add_argument("--security-policy", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        output = build_release(
            checkpoint_dir=args.checkpoint_dir,
            evidence_dir=args.evidence_dir,
            output_dir=args.output_dir,
            model_card=args.model_card,
            remote_code_source=args.remote_code_source,
            repository_root=args.repository_root,
            project_file_overrides={
                name: path
                for name, path in {
                    "LICENSE": args.license_file,
                    "NOTICE": args.notice_file,
                    "THIRD_PARTY_NOTICES.md": args.third_party_notices,
                    "SECURITY.md": args.security_policy,
                }.items()
                if path is not None
            },
            force=args.force,
        )
    except (OSError, ReleaseBuildError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"release build blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Built local release package: {output}")
    print("No upload was performed. Run scripts/release_gate.py before any publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
