#!/usr/bin/env python3
"""Verify the closed, deterministic B3 BIE73 Hugging Face package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if __package__:
    from scripts import build_b3_full_bie73_hf_package as package_builder
    from scripts import build_release
else:  # pragma: no cover - exercised through the command-line smoke test
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts import build_b3_full_bie73_hf_package as package_builder  # type: ignore[no-redef]
    from scripts import build_release  # type: ignore[no-redef]


class B3PackageVerificationError(RuntimeError):
    """Raised when a public B3 package does not verify exactly."""


def _fail_closed(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return callable_(*args, **kwargs)
    except package_builder.B3PackageError as exc:
        raise B3PackageVerificationError(str(exc)) from exc


def _inventory(root: Path, *, require_read_only: bool) -> dict[str, Path]:
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise B3PackageVerificationError("package directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise B3PackageVerificationError("package root must be a real directory")
    if require_read_only and stat.S_IMODE(metadata.st_mode) != 0o555:
        raise B3PackageVerificationError("package root mode must be exactly 0555")
    try:
        entries = sorted(os.scandir(root), key=lambda item: item.name)
    except OSError as exc:
        raise B3PackageVerificationError("package directory cannot be enumerated") from exc
    result: dict[str, Path] = {}
    for entry in entries:
        try:
            item = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise B3PackageVerificationError("package entry cannot be inspected") from exc
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
            raise B3PackageVerificationError("package may contain only regular files")
        if require_read_only and stat.S_IMODE(item.st_mode) != 0o444:
            raise B3PackageVerificationError(f"package file {entry.name} mode must be 0444")
        try:
            package_builder._safe_package_name(entry.name)
        except package_builder.B3PackageError as exc:
            raise B3PackageVerificationError("package contains a forbidden filename") from exc
        result[entry.name] = Path(entry.path)
    if set(result) != package_builder.PACKAGE_FILES:
        missing = sorted(package_builder.PACKAGE_FILES - set(result))
        extra = sorted(set(result) - package_builder.PACKAGE_FILES)
        raise B3PackageVerificationError(
            f"package differs from the closed allowlist: missing={missing}, extra={extra}"
        )
    return result


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        return package_builder._strict_json(payload, field=field)
    except package_builder.B3PackageError as exc:
        raise B3PackageVerificationError(str(exc)) from exc


def _read_text(path: Path, *, field: str, scan_local_paths: bool = True) -> bytes:
    try:
        payload = package_builder._read_regular(
            path, field=field, maximum=package_builder._MAX_TEXT_BYTES
        )
        if scan_local_paths:
            package_builder._assert_no_local_path(payload, field=field)
    except package_builder.B3PackageError as exc:
        raise B3PackageVerificationError(str(exc)) from exc
    return payload


def _verify_checksum_closure(files: Mapping[str, Path]) -> dict[str, str]:
    checksum_payload = _read_text(files[package_builder.CHECKSUMS_NAME], field="checksums.txt")
    try:
        declared = package_builder.parse_checksums(checksum_payload)
    except package_builder.B3PackageError as exc:
        raise B3PackageVerificationError(str(exc)) from exc
    expected_names = package_builder.PACKAGE_FILES - {package_builder.CHECKSUMS_NAME}
    if set(declared) != expected_names or list(declared) != sorted(declared):
        raise B3PackageVerificationError("checksums.txt is not the sorted complete closure")
    for name, expected in declared.items():
        try:
            observed, _size = package_builder._hash_regular(
                files[name], field=f"package file {name}"
            )
        except package_builder.B3PackageError as exc:
            raise B3PackageVerificationError(str(exc)) from exc
        if observed != expected:
            raise B3PackageVerificationError(f"checksum mismatch for {name}")
    return declared


def _verify_manifest(
    manifest: Mapping[str, Any],
    *,
    files: Mapping[str, Path],
    checksums: Mapping[str, str],
) -> None:
    if set(manifest) != {
        "schema_version",
        "status",
        "format",
        "model",
        "training_manifest",
        "selection_provenance",
        "files",
        "file_policy",
        "build_inputs",
        "manifest_sha256",
    }:
        raise B3PackageVerificationError("package manifest has an invalid closed shape")
    claimed = manifest.get("manifest_sha256")
    if (
        not isinstance(claimed, str)
        or package_builder.SHA256_PATTERN.fullmatch(claimed) is None
        or claimed != package_builder.canonical_json_hash(manifest, remove="manifest_sha256")
    ):
        raise B3PackageVerificationError("package manifest self-hash does not verify")
    if (
        manifest.get("schema_version") != package_builder.SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("format") != "huggingface_transformers_remote_code"
    ):
        raise B3PackageVerificationError("package manifest identity is invalid")

    inventory = manifest.get("files")
    if not isinstance(inventory, Mapping) or set(inventory) != package_builder.PAYLOAD_FILES:
        raise B3PackageVerificationError("package manifest payload inventory is incomplete")
    for name, raw in inventory.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw, Mapping)
            or set(raw)
            != {
                "file_sha256",
                "role",
                "size_bytes",
            }
        ):
            raise B3PackageVerificationError("package manifest contains an invalid file entry")
        size = raw.get("size_bytes")
        if (
            raw.get("file_sha256") != checksums.get(name)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size != files[name].stat().st_size
            or not isinstance(raw.get("role"), str)
            or not raw.get("role")
        ):
            raise B3PackageVerificationError(f"package manifest binding is invalid for {name}")

    expected_policy = {
        "closed_allowlist": True,
        "contains_internal_evaluation_artifacts": False,
        "contains_local_absolute_path_metadata": False,
        "contains_pickle_serialization": False,
        "contains_recovery_artifacts": False,
        "contains_symlinks": False,
    }
    if manifest.get("file_policy") != expected_policy:
        raise B3PackageVerificationError("package file policy is invalid")
    identity = package_builder.FROZEN_B3_IDENTITY
    expected_provenance = {
        "candidate_id": identity["candidate_id"],
        "seed": identity["seed"],
        "epoch": identity["epoch"],
        "global_step": identity["global_step"],
        "checkpoint_id": identity["checkpoint_id"],
        "selection_receipt_file_sha256": identity["selection_receipt_file_sha256"],
        "selection_receipt_sha256": identity["selection_receipt_sha256"],
        "model_export_receipt_file_sha256": identity["model_export_receipt_file_sha256"],
        "model_export_receipt_sha256": identity["model_export_receipt_sha256"],
    }
    if manifest.get("selection_provenance") != expected_provenance:
        raise B3PackageVerificationError("package is not anchored to the frozen B3 selection")
    build_inputs = manifest.get("build_inputs")
    if not isinstance(build_inputs, Mapping) or set(build_inputs) != {
        "builder_file_sha256",
        "remote_code_source_file_sha256",
        "taxonomy_source_file_sha256",
    }:
        raise B3PackageVerificationError("package build-input identity is incomplete")
    expected_build_inputs = {
        "builder_file_sha256": _fail_closed(
            package_builder._hash_regular,
            Path(package_builder.__file__),
            field="B3 package builder",
        )[0],
        "remote_code_source_file_sha256": _fail_closed(
            package_builder._hash_regular,
            REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
            field="remote code source",
        )[0],
        "taxonomy_source_file_sha256": _fail_closed(
            package_builder._hash_regular,
            REPOSITORY_ROOT / "src/pii_zh/taxonomy/taxonomy.yaml",
            field="taxonomy source",
        )[0],
    }
    if build_inputs != expected_build_inputs:
        raise B3PackageVerificationError("package was not produced by the current frozen inputs")


def _verify_generated_contracts(files: Mapping[str, Path]) -> tuple[str, dict[str, Any]]:
    gitattributes = _read_text(files[".gitattributes"], field=".gitattributes")
    if gitattributes != package_builder.GITATTRIBUTES_PAYLOAD:
        raise B3PackageVerificationError(
            ".gitattributes does not declare the frozen safetensors LFS contract"
        )
    expected_taxonomy = _fail_closed(package_builder.render_public_taxonomy)
    observed_taxonomy = _read_text(files["taxonomy.yaml"], field="taxonomy.yaml")
    if observed_taxonomy != expected_taxonomy:
        raise B3PackageVerificationError("taxonomy.yaml is not the deterministic BIE taxonomy")
    try:
        taxonomy = yaml.safe_load(observed_taxonomy.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise B3PackageVerificationError("taxonomy.yaml is invalid") from exc
    if not isinstance(taxonomy, Mapping):
        raise B3PackageVerificationError("taxonomy.yaml must be a mapping")
    taxonomy_version = taxonomy.get("taxonomy_version")
    contract = taxonomy.get("model_contract")
    _source, core = _fail_closed(package_builder._source_taxonomy)
    if (
        not isinstance(taxonomy_version, str)
        or taxonomy.get("tagging_scheme") != "BIE"
        or not isinstance(contract, Mapping)
        or contract.get("schema_version") != package_builder.TAXONOMY_CONTRACT_VERSION
        or contract.get("entity_type_count") != 24
        or contract.get("entity_types") != list(core)
        or contract.get("tag_prefixes") != ["B", "I", "E"]
        or contract.get("token_label_count") != 73
        or contract.get("single_token_entity_state") != "B"
    ):
        raise B3PackageVerificationError("taxonomy.yaml BIE73 contract is invalid")

    expected_id2label = _fail_closed(package_builder.render_id2label)
    observed_id2label = _read_text(files["id2label.json"], field="id2label.json")
    if observed_id2label != expected_id2label:
        raise B3PackageVerificationError("id2label.json is not the deterministic BIE73 map")
    expected_decoder = _fail_closed(
        package_builder.render_decoder_config,
        id2label_sha256=hashlib.sha256(expected_id2label).hexdigest(),
        taxonomy_version=taxonomy_version,
    )
    observed_decoder = _read_text(files["decoder_config.json"], field="decoder_config.json")
    if observed_decoder != expected_decoder:
        raise B3PackageVerificationError("decoder_config.json is not the frozen decoder contract")
    decoder = _strict_json(observed_decoder, field="decoder_config.json")
    if (
        decoder.get("schema_version") != package_builder.DECODER_SCHEMA_VERSION
        or decoder.get("decoder_id") != "constrained_viterbi"
        or decoder.get("allowed_label_ids") != list(range(73))
        or decoder.get("token_label_count") != 73
        or decoder.get("execution")
        != {
            "automatic_in_transformers_pipeline": False,
            "recommended_loader": "pii_zh.full_bie73.load_full_bie73_predictor",
            "raw_transformers_output": "token_logits_not_final_spans",
        }
    ):
        raise B3PackageVerificationError("decoder_config.json semantic contract is invalid")
    return taxonomy_version, decoder


def _verify_remote_code(files: Mapping[str, Path]) -> None:
    try:
        configuration, modeling = build_release.render_remote_code(
            REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
            community_v2_preauthorization=True,
        )
    except build_release.ReleaseBuildError as exc:
        raise B3PackageVerificationError("reviewed remote code cannot be rendered") from exc
    expected = {
        "configuration_qwen3_bi.py": configuration.encode("utf-8"),
        "modeling_qwen3_bi.py": modeling.encode("utf-8"),
    }
    for name, payload in expected.items():
        observed = _read_text(files[name], field=name)
        if observed != payload:
            raise B3PackageVerificationError(f"{name} differs from reviewed remote code")
        try:
            compile(observed.decode("utf-8"), name, "exec")
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise B3PackageVerificationError(f"{name} is not valid Python") from exc


def verify_package(
    package_dir: Path,
    *,
    expected_weight_sha256: str | None = None,
    expected_training_manifest_file_sha256: str | None = None,
    require_read_only: bool = True,
) -> dict[str, Any]:
    """Verify one package without loading tensors or contacting the network."""

    package_dir = _fail_closed(
        package_builder._reject_symlink_ancestry,
        package_dir,
        field="package directory",
    )
    files = _inventory(package_dir, require_read_only=require_read_only)
    checksums = _verify_checksum_closure(files)
    for name in sorted(package_builder.PACKAGE_FILES - {"model.safetensors"}):
        payload = _read_text(files[name], field=name, scan_local_paths=name != "tokenizer.json")
        if name == "tokenizer.json":
            _fail_closed(package_builder._assert_tokenizer_has_no_local_path_metadata, payload)

    manifest_payload = _read_text(
        files[package_builder.PACKAGE_MANIFEST_NAME], field=package_builder.PACKAGE_MANIFEST_NAME
    )
    manifest = _strict_json(manifest_payload, field=package_builder.PACKAGE_MANIFEST_NAME)
    _verify_manifest(manifest, files=files, checksums=checksums)
    taxonomy_version, _decoder = _verify_generated_contracts(files)

    config_payload = _read_text(files["config.json"], field="config.json")
    config = _strict_json(config_payload, field="config.json")
    try:
        package_builder.validate_bie73_config(config, taxonomy_version=taxonomy_version)
    except package_builder.B3PackageError as exc:
        raise B3PackageVerificationError(str(exc)) from exc
    training_payload = _read_text(files["training_manifest.json"], field="training_manifest.json")
    training = _strict_json(training_payload, field="training_manifest.json")
    try:
        package_builder.validate_training_manifest(
            training,
            source_model_dir=package_dir,
            taxonomy_version=taxonomy_version,
        )
    except package_builder.B3PackageError as exc:
        raise B3PackageVerificationError(str(exc)) from exc
    try:
        build_release.validate_safetensors(files["model.safetensors"])
    except build_release.ReleaseBuildError as exc:
        raise B3PackageVerificationError("model.safetensors is structurally invalid") from exc
    _verify_remote_code(files)

    model = manifest.get("model")
    training_binding = manifest.get("training_manifest")
    weight_hash = checksums["model.safetensors"]
    training_hash = checksums["training_manifest.json"]
    if (
        not isinstance(model, Mapping)
        or model
        != {
            "architecture": "Qwen3BiForTokenClassification",
            "attention_mode": "full",
            "decoder_id": "constrained_viterbi",
            "entity_type_count": 24,
            "model_type": "qwen3_bi",
            "tagging_scheme": "BIE",
            "taxonomy_version": taxonomy_version,
            "token_label_count": 73,
            "weight_file_sha256": weight_hash,
        }
        or not isinstance(training_binding, Mapping)
        or training_binding
        != {
            "file_sha256": training_hash,
            "manifest_sha256": training.get("manifest_sha256"),
            "preserved_byte_for_byte": True,
        }
        or weight_hash != package_builder.FROZEN_B3_IDENTITY["weight_file_sha256"]
        or training_hash != package_builder.FROZEN_B3_IDENTITY["training_manifest_file_sha256"]
        or training.get("manifest_sha256")
        != package_builder.FROZEN_B3_IDENTITY["training_manifest_sha256"]
    ):
        raise B3PackageVerificationError("package model/training identity is invalid")
    if expected_weight_sha256 is not None and expected_weight_sha256 != weight_hash:
        raise B3PackageVerificationError("model weight hash differs from the expected release hash")
    if (
        expected_training_manifest_file_sha256 is not None
        and expected_training_manifest_file_sha256 != training_hash
    ):
        raise B3PackageVerificationError(
            "training manifest bytes differ from the expected release hash"
        )
    return manifest


def _sha256_argument(value: str) -> str:
    if package_builder.SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--expected-weight-sha256", type=_sha256_argument)
    parser.add_argument("--expected-training-manifest-file-sha256", type=_sha256_argument)
    parser.add_argument(
        "--allow-portable-modes",
        action="store_true",
        help=(
            "allow ordinary modes in a materialized flat download; symlinked HF cache "
            "snapshots remain unsupported (local builds require exact 0555/0444)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = verify_package(
            args.package_dir,
            expected_weight_sha256=args.expected_weight_sha256,
            expected_training_manifest_file_sha256=(args.expected_training_manifest_file_sha256),
            require_read_only=not args.allow_portable_modes,
        )
    except (B3PackageVerificationError, OSError) as exc:
        print(f"B3 HF package verification failed: {exc}", file=__import__("sys").stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema_version": manifest["schema_version"],
                "manifest_sha256": manifest["manifest_sha256"],
                "weight_file_sha256": manifest["model"]["weight_file_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
