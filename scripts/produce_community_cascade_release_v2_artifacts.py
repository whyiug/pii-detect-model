#!/usr/bin/env python3
"""Produce typed, immutable local artifacts for community release contract v2.

The producer accepts file-system locations only as invocation inputs.  It never
serializes local paths, caller-provided hashes, caller-provided PASS states, or
publication claims.  Every JSON artifact is closed by a shared schema, binds
the current producer/schema bytes, and carries a canonical self hash.

This command does not upload or publish anything.  The dependency scan is the
only subcommand allowed to access the network; it invokes the fixed
``osv-scanner`` JSON/SBOM interface and records that fact explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any

from pii_zh.cascade import CommunityModelContractError, expected_core24_label2id
from pii_zh.evaluation import release_eval_v2_prediction_provenance as model_provenance

try:
    from scripts import build_release, generate_sbom, scan_public_artifacts
    from scripts.community_contract_utils import (
        CommunityContractError,
        canonical_json_hash,
        load_json_path,
        read_regular_file,
        sha256_bytes,
        strict_json_bytes,
        validate_schema,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    import build_release  # type: ignore[no-redef]
    import generate_sbom  # type: ignore[no-redef]
    import scan_public_artifacts  # type: ignore[no-redef]
    from community_contract_utils import (  # type: ignore[no-redef]
        CommunityContractError,
        canonical_json_hash,
        load_json_path,
        read_regular_file,
        sha256_bytes,
        strict_json_bytes,
        validate_schema,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATH = "scripts/produce_community_cascade_release_v2_artifacts.py"
SCHEMA_PATH = "configs/release/community_cascade_release_v2.artifact-evidence.schema.json"
SCHEMA_VERSION = "pii-zh.community-release-artifact-evidence.v2"

DISPLAY_NAME = "pii-zh-qwen3-0.6b-24class"
PACKAGE_NAME = "pii-zh-qwen"
PACKAGE_VERSION = "0.2.0rc1"
PUBLICATION_STATE = "unpublished_local_candidate"
CORE24_ORDERED_LABELS = tuple(
    label.removeprefix("B-")
    for label, _index in sorted(expected_core24_label2id().items(), key=lambda item: item[1])
    if label.startswith("B-")
)

ARTIFACT_IDS = (
    "model_card",
    "model_package_manifest",
    "technical_documentation_manifest",
    "wheel_manifest",
    "wheelhouse_manifest",
    "container_manifest",
    "sbom",
    "license_report",
    "public_artifact_scan",
    "dependency_scan",
)

TECHNICAL_DOCUMENTS: Mapping[str, str] = {
    "community_cascade_release_v2": "docs/community-cascade-release-v2.md",
    "community_model_service": "docs/community-model-service.md",
    "cascade_architecture": "docs/cascade-architecture.md",
    "cascade_deployment": "docs/cascade-deployment.md",
    "full_model_and_cascade_guide": "docs/aiguard24_full_model_and_cascade_technical_guide.md",
}
COMMUNITY_TEMPLATE_ASSET_PATH = "src/pii_zh/data/synthetic/assets/curated_templates_v1.json"
PII_BENCH_POSTHOC_DATASET_ID = "wan9yu/pii-bench-zh"
PII_BENCH_POSTHOC_DATASET_REVISION = "c350b94897af668517ff5de237d89f2ce2eaa6f0"
PII_BENCH_POSTHOC_METRICS: Mapping[str, Mapping[str, float | int]] = {
    "formal": {"documents": 5000, "strict_micro_f1": 0.59921050, "strict_macro_f1": 0.57994984},
    "chat": {"documents": 3000, "strict_micro_f1": 0.47628738, "strict_macro_f1": 0.41874684},
    "pooled": {"documents": 8000, "strict_micro_f1": 0.55750532, "strict_macro_f1": 0.52965259},
}

MODEL_PACKAGE_REQUIRED_IDS = frozenset(
    {
        "pkg:README.md",
        "pkg:LICENSE",
        "pkg:NOTICE",
        "pkg:SECURITY.md",
        "pkg:THIRD_PARTY_NOTICES.md",
        "pkg:config.json",
        "pkg:configuration_qwen3_bi.py",
        "pkg:id2label.json",
        "pkg:model.safetensors",
        "pkg:modeling_qwen3_bi.py",
        "pkg:special_tokens_map.json",
        "pkg:tokenizer.json",
        "pkg:tokenizer_config.json",
        "pkg:taxonomy.yaml",
        "pkg:training_manifest.json",
        "pkg:checksums.txt",
        "pkg:community_v2_preauthorization.json",
    }
)
WHEEL_REQUIRED_SUFFIXES = frozenset(
    {
        "pii_zh:cli.py",
        "pii_zh:service:app.py",
        "pii_zh:cascade:routing.py",
        "pii_zh:taxonomy:taxonomy.yaml",
    }
)
FORBIDDEN_PACKAGE_SUFFIXES = frozenset(
    {
        ".arrow",
        ".bin",
        ".ckpt",
        ".csv",
        ".jsonl",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".tsv",
    }
)
FORBIDDEN_MODEL_CARD_MARKERS = (
    "@@",
    "TODO",
    "TBD",
    "首个",
    "最强",
    "SOTA",
    "state-of-the-art",
    "production-ready",
    "生产就绪",
)
MAX_LARGE_FILE_BYTES = 32 * 1024 * 1024 * 1024
MAX_WHEEL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024

EXPECTED_RELEASE = {
    "display_name": DISPLAY_NAME,
    "package_name": PACKAGE_NAME,
    "package_version": PACKAGE_VERSION,
    "publication_state": PUBLICATION_STATE,
    "production_ready": False,
}
EXPECTED_PRIVACY_OFFLINE = {
    "contains_local_paths": False,
    "contains_raw_records": False,
    "contains_secrets": False,
    "network_used": False,
}


class CommunityReleaseArtifactError(CommunityContractError):
    """Raised when a release artifact is incomplete, unsafe, or inconsistent."""


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CommunityReleaseArtifactError(f"{field} is not a SHA-256 digest")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:@+\-]{0,191}", value
    ):
        raise CommunityReleaseArtifactError(f"{field} is not a safe logical ID")
    return value


def _exact(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CommunityReleaseArtifactError(f"{field} has an invalid closed shape")
    return value


def _producer_identity(repository_root: Path, artifact_id: str) -> dict[str, str]:
    producer_hash = sha256_bytes(
        read_regular_file(repository_root / PRODUCER_PATH, field="artifact producer")
    )
    schema_hash = sha256_bytes(
        read_regular_file(repository_root / SCHEMA_PATH, field="artifact evidence schema")
    )
    identity = {
        "artifact_id": artifact_id,
        "producer_file_sha256": producer_hash,
        "schema_file_sha256": schema_hash,
        "schema_version": SCHEMA_VERSION,
    }
    return {
        "file_sha256": producer_hash,
        "schema_file_sha256": schema_hash,
        "implementation_identity_sha256": canonical_json_hash(identity),
    }


def _open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _large_file_binding(path: Path, *, field: str) -> dict[str, int | str]:
    """Hash one stable regular file without imposing the metadata-only size cap."""

    try:
        descriptor = os.open(path, _open_flags())
    except OSError as exc:
        raise CommunityReleaseArtifactError(f"{field} is unavailable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_LARGE_FILE_BYTES
        ):
            raise CommunityReleaseArtifactError(f"{field} is not a bounded non-empty file")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or size != after.st_size:
            raise CommunityReleaseArtifactError(f"{field} changed while it was hashed")
        return {"file_sha256": digest.hexdigest(), "size_bytes": size}
    finally:
        os.close(descriptor)


def _metadata_binding(path: Path, *, field: str) -> dict[str, int | str]:
    payload = read_regular_file(path, field=field)
    if not payload:
        raise CommunityReleaseArtifactError(f"{field} is empty")
    return {"file_sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _base_document(
    artifact_id: str,
    *,
    inputs: Mapping[str, Mapping[str, int | str]],
    result: Mapping[str, Any],
    repository_root: Path,
    status: str = "PASS",
    network_used: bool = False,
) -> dict[str, Any]:
    if artifact_id not in ARTIFACT_IDS:
        raise CommunityReleaseArtifactError("unsupported release artifact ID")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "status": status,
        "release": dict(EXPECTED_RELEASE),
        "inputs": dict(sorted((key, dict(value)) for key, value in inputs.items())),
        "result": copy.deepcopy(dict(result)),
        "privacy": {**EXPECTED_PRIVACY_OFFLINE, "network_used": network_used},
        "producer": _producer_identity(repository_root, artifact_id),
        "artifact_sha256": "",
    }
    document["artifact_sha256"] = canonical_json_hash(document, remove="artifact_sha256")
    validate_artifact_document(
        document,
        artifact_id=artifact_id,
        repository_root=repository_root,
        verify_current_repository_inputs=False,
    )
    return document


def _validate_binding_map(value: object, *, field: str, allow_empty: bool = False) -> None:
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        raise CommunityReleaseArtifactError(f"{field} must be a non-empty binding map")
    for logical_id, binding in value.items():
        _safe_id(logical_id, field=f"{field} logical ID")
        item = _exact(binding, {"file_sha256", "size_bytes"}, field=f"{field}.{logical_id}")
        _digest(item["file_sha256"], field=f"{field}.{logical_id}.file_sha256")
        if (
            isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] <= 0
        ):
            raise CommunityReleaseArtifactError(f"{field}.{logical_id}.size_bytes is invalid")


def _validate_model_card_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "media_type",
            "markdown",
            "markdown_sha256",
            "entity_type_count",
            "token_label_count",
            "quality_receipt_sha256",
            "model_manifest_file_sha256",
            "service_manifest_file_sha256",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "base_model_id",
            "base_model_revision",
            "base_model_license",
            "head_projection_count",
            "ordered_entity_labels",
            "training_data_facts",
            "template_asset_file_sha256",
            "pii_bench_report_file_sha256",
            "pii_bench_report_sha256",
            "pii_bench_posthoc",
            "placeholder_count",
            "forbidden_claims_absent",
        },
        field="model card result",
    )
    markdown = result["markdown"]
    if not isinstance(markdown, str) or len(markdown.encode("utf-8")) < 512:
        raise CommunityReleaseArtifactError("model card markdown is missing or too small")
    if result["media_type"] != "text/markdown; charset=utf-8":
        raise CommunityReleaseArtifactError("model card media type is invalid")
    if sha256_bytes(markdown.encode("utf-8")) != result["markdown_sha256"]:
        raise CommunityReleaseArtifactError("model card content hash does not verify")
    for key in (
        "quality_receipt_sha256",
        "model_manifest_file_sha256",
        "service_manifest_file_sha256",
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "template_asset_file_sha256",
        "pii_bench_report_file_sha256",
        "pii_bench_report_sha256",
    ):
        _digest(result[key], field=f"model card {key}")
    lowered = markdown.casefold()
    forbidden = [marker for marker in FORBIDDEN_MODEL_CARD_MARKERS if marker.casefold() in lowered]
    ordered_labels = result["ordered_entity_labels"]
    training_data_facts = result["training_data_facts"]
    pii_bench_posthoc = result["pii_bench_posthoc"]
    expected_training_data_facts = {
        "training_document_count": 87_995,
        "training_pii_free_document_count": 39_600,
        "validation_document_count": 2_005,
        "validation_pii_free_document_count": 900,
        "entity_type_count": 24,
        "synthetic_only": True,
        "contains_real_customer_or_production_pii": False,
        "validation_blind": False,
        "template_group_overlap_count": 17,
        "model_assisted_accepted_template_count": 53,
        "human_authored_positive_template_count": 34,
        "human_authored_hard_negative_template_count": 20,
        "template_candidate_model_id": "Qwen/Qwen3-8B",
    }
    expected_pii_bench = {
        "dataset_id": PII_BENCH_POSTHOC_DATASET_ID,
        "dataset_revision": PII_BENCH_POSTHOC_DATASET_REVISION,
        "evidence_classification": "posthoc_descriptive_public_benchmark",
        "public_test_exposed": True,
        "selection_allowed": False,
        "descriptive_active_envelope_passed": False,
        "model_raw_evaluated": True,
        "full_system_evaluated": False,
        "suites": {key: dict(value) for key, value in PII_BENCH_POSTHOC_METRICS.items()},
    }
    if (
        result["placeholder_count"] != 0
        or result["forbidden_claims_absent"] is not True
        or forbidden
        or DISPLAY_NAME not in markdown
        or PACKAGE_VERSION not in markdown
        or PUBLICATION_STATE not in markdown
        or result["entity_type_count"] != 24
        or result["token_label_count"] != 49
        or result["base_model_id"] != build_release.COMMUNITY_V2_BASE_MODEL_ID
        or result["base_model_revision"] != build_release.COMMUNITY_V2_BASE_MODEL_REVISION
        or result["base_model_license"] != build_release.COMMUNITY_V2_BASE_MODEL_LICENSE
        or result["head_projection_count"] != len(build_release.COMMUNITY_V2_MAPPED_TARGET_LABELS)
        or ordered_labels != list(CORE24_ORDERED_LABELS)
        or training_data_facts != expected_training_data_facts
        or pii_bench_posthoc != expected_pii_bench
        or f"license: {build_release.COMMUNITY_V2_BASE_MODEL_LICENSE}" not in markdown
        or f"base_model: {build_release.COMMUNITY_V2_BASE_MODEL_ID}" not in markdown
        or f"base_model_revision: {build_release.COMMUNITY_V2_BASE_MODEL_REVISION}" not in markdown
        or build_release.COMMUNITY_V2_BASE_MODEL_REVISION not in markdown
        or "12 个源实体分类头" not in markdown
        or "87,995" not in markdown
        or "39,600" not in markdown
        or "2,005" not in markdown
        or "900" not in markdown
        or "100% 确定性合成" not in markdown
        or "不是盲验证" not in markdown
        or "0.59921050" not in markdown
        or "0.47628738" not in markdown
        or "0.55750532" not in markdown
        or "selection_allowed=false" not in markdown
        or "N/A（未评测）" not in markdown
        or "`, `".join(ordered_labels) not in markdown
        or "LOCAL_MODEL_DIR" in markdown
        or "from pii_zh.inference import load_local_predictor" not in markdown
        or "predictor.predict(synthetic_text)" not in markdown
        or '"label": span["entity_type"]' not in markdown
        or '"start": span["start"]' not in markdown
        or '"end": span["end"]' not in markdown
        or "demo@example.com" not in markdown
        or "user@example.test" in markdown
        or "不构成对任意输入必然命中的承诺" not in markdown
        or "AutoConfig.from_pretrained" not in markdown
        or markdown.count("local_files_only=True") != 3
        or markdown.count("trust_remote_code=True") != 3
        or "config.pii_release_eligible is False" not in markdown
        or "logits.shape[-1] == 49" not in markdown
        or "torch.isfinite(logits).all().item()" not in markdown
    ):
        raise CommunityReleaseArtifactError("model card contains a placeholder or forbidden claim")


def _validate_inventory_result(
    result: Mapping[str, Any], *, field: str, required_ids: frozenset[str] = frozenset()
) -> Mapping[str, Any]:
    result = _exact(
        result,
        {"format", "file_count", "inventory_sha256", "files"},
        field=field,
    )
    files = result["files"]
    _validate_binding_map(files, field=f"{field}.files")
    if result["file_count"] != len(files) or result["inventory_sha256"] != canonical_json_hash(
        files
    ):
        raise CommunityReleaseArtifactError(f"{field} inventory closure does not verify")
    if not required_ids.issubset(files):
        raise CommunityReleaseArtifactError(f"{field} required inventory is incomplete")
    return result


def _validate_model_package_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "format",
            "file_count",
            "inventory_sha256",
            "files",
            "entity_type_count",
            "token_label_count",
            "training_release_eligible",
            "community_candidate_eligibility_source",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "model_identity_sha256",
            "output_artifact_sha256",
            "weights_combined_sha256",
            "config_file_sha256",
            "remote_code_profile",
            "remote_code_source_file_sha256",
            "remote_code_generator_file_sha256",
            "remote_code_files",
            "package_support_profile",
            "package_support_files",
            "community_preauthorization_state",
            "community_preauthorization_sha256",
        },
        field="model package manifest",
    )
    files = result["files"]
    _validate_binding_map(files, field="model package manifest.files")
    if (
        result["format"] != "huggingface_safetensors_model_package_v1"
        or result["file_count"] != len(files)
        or result["inventory_sha256"] != canonical_json_hash(files)
        or set(files) != MODEL_PACKAGE_REQUIRED_IDS
        or result["entity_type_count"] != 24
        or result["token_label_count"] != 49
        or result["training_release_eligible"] is not False
        or result["community_candidate_eligibility_source"]
        != "community_cascade_release_v2_contract"
        or result["community_preauthorization_state"]
        != build_release.COMMUNITY_PREAUTHORIZATION_STATE
        or result["remote_code_profile"] != build_release.COMMUNITY_V2_REMOTE_CODE_PROFILE
        or result["package_support_profile"] != build_release.COMMUNITY_V2_PACKAGE_SUPPORT_PROFILE
    ):
        raise CommunityReleaseArtifactError("model package identity is incomplete")
    for key in (
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "model_identity_sha256",
        "output_artifact_sha256",
        "weights_combined_sha256",
        "config_file_sha256",
        "remote_code_source_file_sha256",
        "remote_code_generator_file_sha256",
        "community_preauthorization_sha256",
    ):
        _digest(result[key], field=f"model package {key}")
    remote_code_files = result["remote_code_files"]
    package_support_files = result["package_support_files"]
    _validate_binding_map(remote_code_files, field="model package remote_code_files")
    _validate_binding_map(package_support_files, field="model package package_support_files")
    if (
        set(remote_code_files) != set(build_release.REMOTE_CODE_FILES)
        or any(binding != files[f"pkg:{name}"] for name, binding in remote_code_files.items())
        or set(package_support_files) != {"NOTICE", "THIRD_PARTY_NOTICES.md"}
        or any(binding != files[f"pkg:{name}"] for name, binding in package_support_files.items())
    ):
        raise CommunityReleaseArtifactError(
            "model package generated support-file bindings are incomplete"
        )


def _validate_current_model_package_remote_code(
    result: Mapping[str, Any], *, repository_root: Path
) -> None:
    source_path = repository_root / "src/pii_zh/models/qwen3_bi.py"
    generator_path = repository_root / "scripts/build_release.py"
    try:
        configuration, modeling = build_release.render_remote_code(
            source_path,
            community_v2_preauthorization=True,
        )
    except (OSError, SyntaxError, ValueError) as exc:
        raise CommunityReleaseArtifactError(
            "current community remote-code profile cannot be rendered"
        ) from exc
    expected_files = {
        "configuration_qwen3_bi.py": configuration.encode("utf-8"),
        "modeling_qwen3_bi.py": modeling.encode("utf-8"),
    }
    expected_bindings = {
        name: {"file_sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        for name, payload in sorted(expected_files.items())
    }
    if (
        result["remote_code_profile"] != build_release.COMMUNITY_V2_REMOTE_CODE_PROFILE
        or result["remote_code_source_file_sha256"]
        != sha256_bytes(read_regular_file(source_path, field="remote-code source"))
        or result["remote_code_generator_file_sha256"]
        != sha256_bytes(read_regular_file(generator_path, field="remote-code generator"))
        or result["remote_code_files"] != expected_bindings
    ):
        raise CommunityReleaseArtifactError(
            "model package remote-code generation identity is stale"
        )


def _validate_wheelhouse_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "format",
            "wheel_count",
            "inventory_sha256",
            "files",
            "packages_sha256",
            "packages",
            "sbom_artifact_sha256",
        },
        field="wheelhouse manifest result",
    )
    files = result["files"]
    packages = result["packages"]
    _validate_binding_map(files, field="wheelhouse manifest.files")
    if not isinstance(packages, Mapping) or not packages:
        raise CommunityReleaseArtifactError("wheelhouse package inventory is empty")
    for logical_id, package in packages.items():
        _safe_id(logical_id, field="wheelhouse package ID")
        package = _exact(
            package,
            {
                "name",
                "version",
                "wheel_file_sha256",
                "member_count",
                "member_inventory_sha256",
            },
            field=f"wheelhouse package {logical_id}",
        )
        if (
            not isinstance(package["name"], str)
            or not package["name"]
            or not isinstance(package["version"], str)
            or not package["version"]
            or isinstance(package["member_count"], bool)
            or not isinstance(package["member_count"], int)
            or package["member_count"] < 1
        ):
            raise CommunityReleaseArtifactError("wheelhouse package metadata is invalid")
        _digest(package["wheel_file_sha256"], field="wheelhouse package file")
        _digest(package["member_inventory_sha256"], field="wheelhouse member inventory")
    if (
        result["format"] != "locked_python_wheelhouse_v1"
        or result["wheel_count"] != len(files)
        or result["wheel_count"] != len(packages)
        or result["inventory_sha256"] != canonical_json_hash(files)
        or result["packages_sha256"] != canonical_json_hash(packages)
    ):
        raise CommunityReleaseArtifactError("wheelhouse inventory closure does not verify")
    _digest(result["sbom_artifact_sha256"], field="wheelhouse SBOM artifact")


def _validate_container_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "format",
            "image_id",
            "content_digest_sha256",
            "repo_digest_sha256",
            "config_sha256",
            "labels_sha256",
            "offline_environment",
            "bound_wheel_artifact_sha256",
            "bound_wheel_file_sha256",
            "bound_wheelhouse_artifact_sha256",
            "external_model_package_artifact_sha256",
            "model_delivery",
        },
        field="container manifest result",
    )
    if result["format"] != "docker_image_inspect_v1":
        raise CommunityReleaseArtifactError("container manifest format is invalid")
    for key in (
        "content_digest_sha256",
        "repo_digest_sha256",
        "config_sha256",
        "labels_sha256",
        "bound_wheel_artifact_sha256",
        "bound_wheel_file_sha256",
        "bound_wheelhouse_artifact_sha256",
        "external_model_package_artifact_sha256",
    ):
        _digest(result[key], field=f"container {key}")
    if not isinstance(result["image_id"], str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", result["image_id"]
    ):
        raise CommunityReleaseArtifactError("container image ID is invalid")
    if result["offline_environment"] != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise CommunityReleaseArtifactError("container offline environment is not fixed")
    if result["model_delivery"] != "external_read_only_mount":
        raise CommunityReleaseArtifactError("container model delivery boundary is invalid")


def _validate_sbom_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {"format", "bom_sha256", "component_count", "dependency_edge_count", "bom"},
        field="SBOM result",
    )
    bom = result["bom"]
    if not isinstance(bom, Mapping):
        raise CommunityReleaseArtifactError("SBOM payload is not an object")
    components = bom.get("components")
    dependencies = bom.get("dependencies")
    metadata = bom.get("metadata")
    if (
        result["format"] != "CycloneDX-1.6"
        or bom.get("bomFormat") != "CycloneDX"
        or bom.get("specVersion") != "1.6"
        or not isinstance(components, list)
        or not components
        or not isinstance(dependencies, list)
        or not isinstance(metadata, Mapping)
        or not isinstance(metadata.get("component"), Mapping)
        or metadata["component"].get("name") != PACKAGE_NAME
        or metadata["component"].get("version") != PACKAGE_VERSION
        or result["component_count"] != len(components)
        or result["dependency_edge_count"]
        != sum(len(item.get("dependsOn", [])) for item in dependencies if isinstance(item, Mapping))
        or result["bom_sha256"] != canonical_json_hash(bom)
    ):
        raise CommunityReleaseArtifactError("SBOM semantic closure does not verify")


def _validate_license_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "format",
            "component_count",
            "declared_license_count",
            "unknown_license_count",
            "license_inventory_sha256",
            "license_inventory",
            "human_approval_status",
            "legal_review_status",
            "automated_release_clearance",
        },
        field="license report result",
    )
    inventory = result["license_inventory"]
    if not isinstance(inventory, Mapping) or not inventory:
        raise CommunityReleaseArtifactError("license inventory is empty")
    if (
        result["format"] != "factual_license_inventory_v1"
        or result["component_count"] != len(inventory)
        or result["declared_license_count"] + result["unknown_license_count"] != len(inventory)
        or result["license_inventory_sha256"] != canonical_json_hash(inventory)
        or result["human_approval_status"] != "pending"
        or result["legal_review_status"] != "pending"
        or result["automated_release_clearance"] is not False
    ):
        raise CommunityReleaseArtifactError("license report overstates or breaks its evidence")


def _validate_public_scan_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "format",
            "scanner_file_sha256",
            "deep_scanned_file_count",
            "deep_scanned_inventory_sha256",
            "deep_scanned_inventory",
            "hashed_dependency_wheel_count",
            "hashed_dependency_wheel_inventory_sha256",
            "hashed_dependency_wheel_inventory",
            "finding_count",
            "finding_kinds",
        },
        field="public artifact scan result",
    )
    inventory = result["deep_scanned_inventory"]
    dependency_inventory = result["hashed_dependency_wheel_inventory"]
    _validate_binding_map(inventory, field="public artifact deep-scanned inventory")
    _validate_binding_map(
        dependency_inventory, field="public artifact hashed dependency wheel inventory"
    )
    if (
        result["format"] != "pii_zh_public_artifact_scan_v3"
        or result["deep_scanned_file_count"] != len(inventory)
        or result["deep_scanned_inventory_sha256"] != canonical_json_hash(inventory)
        or result["hashed_dependency_wheel_count"] != len(dependency_inventory)
        or result["hashed_dependency_wheel_inventory_sha256"]
        != canonical_json_hash(dependency_inventory)
        or result["finding_count"] != 0
        or result["finding_kinds"] != []
    ):
        raise CommunityReleaseArtifactError("public artifact scan is not a complete PASS")
    _digest(result["scanner_file_sha256"], field="public scan implementation")


def _validate_dependency_scan_result(result: Mapping[str, Any]) -> None:
    result = _exact(
        result,
        {
            "format",
            "scanner_name",
            "scanner_version",
            "scanner_file_sha256",
            "command_id",
            "command_argv_sha256",
            "sbom_artifact_sha256",
            "sbom_content_sha256",
            "component_count",
            "result_count",
            "vulnerability_count",
            "raw_report_sha256",
        },
        field="dependency scan result",
    )
    if (
        result["format"] != "osv_scanner_json_v2"
        or result["scanner_name"] != "osv-scanner"
        or not isinstance(result["scanner_version"], str)
        or re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?",
            result["scanner_version"],
        )
        is None
        or result["command_id"] != "osv_scanner_v2_cyclonedx_json"
        or isinstance(result["component_count"], bool)
        or not isinstance(result["component_count"], int)
        or result["component_count"] <= 0
        or result["result_count"] < 0
        or result["vulnerability_count"] != 0
    ):
        raise CommunityReleaseArtifactError("dependency scan is not a fresh zero-finding PASS")
    for key in (
        "scanner_file_sha256",
        "command_argv_sha256",
        "sbom_artifact_sha256",
        "sbom_content_sha256",
        "raw_report_sha256",
    ):
        _digest(result[key], field=f"dependency scan {key}")


def validate_artifact_document(
    document: Mapping[str, Any],
    *,
    artifact_id: str,
    repository_root: Path = REPOSITORY_ROOT,
    verify_current_repository_inputs: bool = True,
) -> None:
    """Independently validate one typed artifact evidence document."""

    schema = load_json_path(repository_root / SCHEMA_PATH, field="artifact evidence schema")
    validate_schema(document, schema, field=f"{artifact_id} artifact evidence")
    if document["artifact_id"] != artifact_id or artifact_id not in ARTIFACT_IDS:
        raise CommunityReleaseArtifactError("artifact ID differs from its contract slot")
    if document["release"] != EXPECTED_RELEASE:
        raise CommunityReleaseArtifactError("artifact release identity is not the fixed candidate")
    _validate_binding_map(document["inputs"], field=f"{artifact_id} inputs", allow_empty=True)
    if document["producer"] != _producer_identity(repository_root, artifact_id):
        raise CommunityReleaseArtifactError("artifact producer identity is stale")
    if document["artifact_sha256"] != canonical_json_hash(document, remove="artifact_sha256"):
        raise CommunityReleaseArtifactError("artifact canonical self hash does not verify")
    privacy = document["privacy"]
    if {key: privacy[key] for key in EXPECTED_PRIVACY_OFFLINE} != {
        **EXPECTED_PRIVACY_OFFLINE,
        "network_used": privacy["network_used"],
    }:
        raise CommunityReleaseArtifactError("artifact privacy boundary is invalid")
    if artifact_id == "dependency_scan":
        if privacy["network_used"] is not True or document["status"] != "PASS":
            raise CommunityReleaseArtifactError("dependency scan network disclosure is invalid")
    elif privacy["network_used"] is not False:
        raise CommunityReleaseArtifactError("offline artifact unexpectedly reports network use")
    result = document["result"]
    if artifact_id == "model_card":
        _validate_model_card_result(result)
    elif artifact_id == "model_package_manifest":
        _validate_model_package_result(result)
        if verify_current_repository_inputs:
            _validate_current_model_package_remote_code(result, repository_root=repository_root)
    elif artifact_id == "technical_documentation_manifest":
        parsed = _validate_inventory_result(result, field="technical documentation manifest")
        if parsed["format"] != "community_release_documentation_v1" or set(parsed["files"]) != set(
            TECHNICAL_DOCUMENTS
        ):
            raise CommunityReleaseArtifactError("technical documentation inventory is incomplete")
        if verify_current_repository_inputs:
            expected_documents = {
                logical_id: _metadata_binding(
                    repository_root / relative, field=f"technical document {logical_id}"
                )
                for logical_id, relative in TECHNICAL_DOCUMENTS.items()
            }
            if parsed["files"] != expected_documents or document["inputs"] != expected_documents:
                raise CommunityReleaseArtifactError("technical documentation is stale")
    elif artifact_id == "wheel_manifest":
        parsed = _validate_inventory_result(result, field="wheel manifest")
        if parsed["format"] != "python_wheel_zip_v1" or not WHEEL_REQUIRED_SUFFIXES.issubset(
            parsed["files"]
        ):
            raise CommunityReleaseArtifactError("wheel manifest is incomplete")
    elif artifact_id == "wheelhouse_manifest":
        _validate_wheelhouse_result(result)
    elif artifact_id == "container_manifest":
        _validate_container_result(result)
    elif artifact_id == "sbom":
        _validate_sbom_result(result)
        if verify_current_repository_inputs:
            lockfile = repository_root / "uv.lock"
            pyproject = repository_root / "pyproject.toml"
            expected_inputs = {
                "uv_lock": _metadata_binding(lockfile, field="locked dependency graph"),
                "pyproject": _metadata_binding(pyproject, field="project metadata"),
            }
            try:
                expected_bom = generate_sbom.build_sbom(lockfile, pyproject)
            except (OSError, TypeError, ValueError) as exc:
                raise CommunityReleaseArtifactError("current SBOM inputs are invalid") from exc
            if document["inputs"] != expected_inputs or result["bom"] != expected_bom:
                raise CommunityReleaseArtifactError(
                    "SBOM differs from the current lock/project bytes"
                )
    elif artifact_id == "license_report":
        if document["status"] != "COMPLETE_HUMAN_APPROVAL_PENDING":
            raise CommunityReleaseArtifactError("license report must preserve pending approval")
        _validate_license_result(result)
    elif artifact_id == "public_artifact_scan":
        _validate_public_scan_result(result)
        if verify_current_repository_inputs:
            expected_scanner = sha256_bytes(
                read_regular_file(
                    repository_root / "scripts/scan_public_artifacts.py",
                    field="public artifact scanner implementation",
                )
            )
            if result["scanner_file_sha256"] != expected_scanner:
                raise CommunityReleaseArtifactError("public artifact scanner identity is stale")
    elif artifact_id == "dependency_scan":
        _validate_dependency_scan_result(result)


def load_and_validate_artifact(
    path: Path,
    *,
    artifact_id: str,
    repository_root: Path = REPOSITORY_ROOT,
    verify_current_repository_inputs: bool = True,
) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path, field=f"{artifact_id} artifact")
    document = strict_json_bytes(payload, field=f"{artifact_id} artifact")
    validate_artifact_document(
        document,
        artifact_id=artifact_id,
        repository_root=repository_root,
        verify_current_repository_inputs=verify_current_repository_inputs,
    )
    return document, payload


def publish_artifact(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically publish an immutable JSON artifact without clobbering."""

    if path.exists() or path.is_symlink():
        raise CommunityReleaseArtifactError("refusing to overwrite release artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise CommunityReleaseArtifactError("release artifact mode is not 0444")
    finally:
        temporary.unlink(missing_ok=True)


def _stage_immutable_payload(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        return temporary
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise CommunityReleaseArtifactError(f"failed to stage immutable {field}") from exc


def _unlink_if_same_inode(path: Path, staged: Path) -> None:
    try:
        if path.stat(follow_symlinks=False).st_ino == staged.stat(follow_symlinks=False).st_ino:
            path.unlink()
    except OSError:
        return


def publish_model_card_outputs(
    artifact_path: Path,
    markdown_path: Path,
    document: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Publish one validated model-card JSON/Markdown pair without clobbering."""

    validate_artifact_document(
        document,
        artifact_id="model_card",
        repository_root=repository_root,
    )
    if os.path.abspath(artifact_path) == os.path.abspath(markdown_path):
        raise CommunityReleaseArtifactError("model card artifact and Markdown outputs must differ")
    for path in (artifact_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise CommunityReleaseArtifactError("refusing to overwrite model card output")
    artifact_payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    markdown_payload = document["result"]["markdown"].encode("utf-8")
    artifact_staged: Path | None = None
    markdown_staged: Path | None = None
    artifact_linked = False
    markdown_linked = False
    try:
        artifact_staged = _stage_immutable_payload(
            artifact_path, artifact_payload, field="model card artifact"
        )
        markdown_staged = _stage_immutable_payload(
            markdown_path, markdown_payload, field="model card Markdown"
        )
        os.link(artifact_staged, artifact_path)
        artifact_linked = True
        os.link(markdown_staged, markdown_path)
        markdown_linked = True
        if (
            stat.S_IMODE(artifact_path.stat().st_mode) != 0o444
            or stat.S_IMODE(markdown_path.stat().st_mode) != 0o444
            or read_regular_file(artifact_path, field="published model card artifact")
            != artifact_payload
            or read_regular_file(markdown_path, field="published model card Markdown")
            != markdown_payload
        ):
            raise CommunityReleaseArtifactError("model card output pair did not publish exactly")
    except Exception as exc:
        if markdown_linked and markdown_staged is not None:
            _unlink_if_same_inode(markdown_path, markdown_staged)
        if artifact_linked and artifact_staged is not None:
            _unlink_if_same_inode(artifact_path, artifact_staged)
        if isinstance(exc, CommunityReleaseArtifactError):
            raise
        raise CommunityReleaseArtifactError("failed to publish model card output pair") from exc
    finally:
        if artifact_staged is not None:
            artifact_staged.unlink(missing_ok=True)
        if markdown_staged is not None:
            markdown_staged.unlink(missing_ok=True)


def _load_self_hashed_json(
    path: Path, *, field: str, self_field: str
) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path, field=field)
    document = strict_json_bytes(payload, field=field)
    expected = document.get(self_field)
    if not isinstance(expected, str) or expected != canonical_json_hash(
        document, remove=self_field
    ):
        raise CommunityReleaseArtifactError(f"{field} self hash does not verify")
    return document, payload


def _format_metric(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "未记录"
    return f"{float(value):.4f}"


def _quality_metric(track: Mapping[str, Any], group: str, key: str) -> str:
    values = track.get(group)
    return _format_metric(values.get(key) if isinstance(values, Mapping) else None)


def _quality_delta(track: Mapping[str, Any], key: str) -> tuple[str, str]:
    bootstrap = track.get("paired_bootstrap")
    delta = bootstrap.get(key) if isinstance(bootstrap, Mapping) else None
    if not isinstance(delta, Mapping):
        return "未记录", "未记录"
    point = _format_metric(delta.get("point_delta_candidate_minus_comparator"))
    lower = _format_metric(delta.get("ci_low"))
    upper = _format_metric(delta.get("ci_high"))
    interval = "未记录" if "未记录" in {lower, upper} else f"[{lower}, {upper}]"
    return point, interval


def _load_pii_bench_posthoc(
    path: Path,
    *,
    training_manifest: Mapping[str, Any],
    training_payload: bytes,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    report, payload = _load_self_hashed_json(
        path,
        field="PII Bench ZH posthoc report",
        self_field="report_sha256",
    )
    status = report.get("status")
    benchmark = report.get("benchmark")
    candidate = report.get("candidate")
    comparison = report.get("comparison_contract")
    envelope = report.get("active_comparator_envelope")
    results = report.get("results")
    if (
        report.get("report_type") != "bio24_model_raw_posthoc_evaluation"
        or report.get("candidate_id") != "aiguard24-v3-seed97"
        or not isinstance(status, Mapping)
        or status.get("evidence_classification") != "posthoc_descriptive_public_benchmark"
        or status.get("public_test_exposed") is not True
        or status.get("posthoc_lineage") is not True
        or status.get("selection_allowed") is not False
        or status.get("release_selection_eligible") is not False
        or status.get("confirmatory_claim_allowed") is not False
        or not isinstance(benchmark, Mapping)
        or benchmark.get("dataset_id") != PII_BENCH_POSTHOC_DATASET_ID
        or benchmark.get("dataset_revision") != PII_BENCH_POSTHOC_DATASET_REVISION
        or benchmark.get("fixed_gold_verified") is not True
        or not isinstance(candidate, Mapping)
        or candidate.get("training_manifest_file_sha256") != sha256_bytes(training_payload)
        or candidate.get("training_manifest_sha256") != training_manifest.get("manifest_sha256")
        or candidate.get("attention_mode") != "full"
        or candidate.get("core_entity_count") != 24
        or not isinstance(comparison, Mapping)
        or comparison.get("track") != "model_raw"
        or comparison.get("threshold_tuning") is not False
        or comparison.get("checkpoint_selection") is not False
        or not isinstance(envelope, Mapping)
        or envelope.get("descriptive_active_envelope_passed") is not False
        or envelope.get("confirmatory_claim_allowed") is not False
        or not isinstance(results, Mapping)
    ):
        raise CommunityReleaseArtifactError(
            "PII Bench ZH report is not the frozen seed-97 descriptive one-shot"
        )
    summary_suites: dict[str, dict[str, float | int]] = {}
    for suite, expected in PII_BENCH_POSTHOC_METRICS.items():
        observed = results.get(suite)
        micro = observed.get("strict_micro") if isinstance(observed, Mapping) else None
        macro = observed.get("strict_macro") if isinstance(observed, Mapping) else None
        summary = {
            "documents": observed.get("documents") if isinstance(observed, Mapping) else None,
            "strict_micro_f1": micro.get("f1") if isinstance(micro, Mapping) else None,
            "strict_macro_f1": macro.get("f1") if isinstance(macro, Mapping) else None,
        }
        if summary != expected:
            raise CommunityReleaseArtifactError(
                f"PII Bench ZH {suite} descriptive metrics differ from the frozen report"
            )
        summary_suites[suite] = dict(expected)
    summary = {
        "dataset_id": PII_BENCH_POSTHOC_DATASET_ID,
        "dataset_revision": PII_BENCH_POSTHOC_DATASET_REVISION,
        "evidence_classification": "posthoc_descriptive_public_benchmark",
        "public_test_exposed": True,
        "selection_allowed": False,
        "descriptive_active_envelope_passed": False,
        "model_raw_evaluated": True,
        "full_system_evaluated": False,
        "suites": summary_suites,
    }
    return report, payload, summary


def render_model_card(
    *,
    quality_receipt: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    service_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    template_asset: Mapping[str, Any],
    pii_bench_posthoc: Mapping[str, Any],
) -> str:
    """Render the fixed pre-authorization model card from admitted evidence."""

    if (
        quality_receipt.get("reported_status") != "PASS"
        or quality_receipt.get("claims", {}).get("claim_activation_allowed") is not False
        or quality_receipt.get("candidate", {}).get("model_id") != model_manifest.get("model_id")
        or quality_receipt.get("candidate", {}).get("service_id")
        != service_manifest.get("service_id")
    ):
        raise CommunityReleaseArtifactError("model card inputs are not one admitted PASS candidate")
    tracks = quality_receipt.get("tracks")
    if not isinstance(tracks, Mapping) or set(tracks) != {"model_raw", "full_system"}:
        raise CommunityReleaseArtifactError("model card quality tracks are incomplete")
    try:
        training_facts = build_release._community_training_initialization_facts(
            dict(training_manifest)
        )
        training_data_facts = build_release._community_training_data_facts(
            dict(training_manifest), dict(template_asset)
        )
    except (TypeError, ValueError) as exc:
        raise CommunityReleaseArtifactError(
            "model card training initialization is not the frozen AIguard migration"
        ) from exc
    ordered_labels = model_manifest.get("ordered_labels")
    if ordered_labels != list(CORE24_ORDERED_LABELS):
        raise CommunityReleaseArtifactError("model card final model has no exact core-24 taxonomy")
    pii_bench_suites = pii_bench_posthoc.get("suites")
    if not isinstance(pii_bench_suites, Mapping) or set(pii_bench_suites) != {
        "formal",
        "chat",
        "pooled",
    }:
        raise CommunityReleaseArtifactError("model card PII Bench summary is incomplete")

    rows: list[str] = []
    delta_rows: list[str] = []
    for track_id, display in (("model_raw", "单模型"), ("full_system", "集成服务")):
        track = tracks[track_id]
        if not isinstance(track, Mapping):
            raise CommunityReleaseArtifactError("model card quality track is malformed")
        comparator = track.get("comparator_aggregate")
        if not isinstance(comparator, Mapping):
            raise CommunityReleaseArtifactError("model card comparator aggregate is missing")
        rows.append(
            f"| {display} | 候选 | {track.get('declared_status', '未记录')} | "
            f"{_quality_metric(track, 'strict_span', 'micro_f1')} | "
            f"{_quality_metric(track, 'strict_span', 'micro_recall')} | "
            f"{_quality_metric(track, 'pii_free', 'document_fpr')} |"
        )
        rows.append(
            f"| {display} | 比较器 `{track.get('comparator_id', '未记录')}` | 对照 | "
            f"{_quality_metric(comparator, 'strict_span', 'micro_f1')} | "
            f"{_quality_metric(comparator, 'strict_span', 'micro_recall')} | "
            f"{_quality_metric(comparator, 'pii_free', 'document_fpr')} |"
        )
        micro_point, micro_ci = _quality_delta(track, "micro_f1_delta")
        macro_point, macro_ci = _quality_delta(track, "macro_f1_delta")
        fpr_point, fpr_ci = _quality_delta(track, "document_fpr_delta")
        delta_rows.append(
            f"| {display} | `{track.get('comparator_id', '未记录')}` | "
            f"{micro_point} | {micro_ci} | {macro_point} | {macro_ci} | "
            f"{fpr_point} | {fpr_ci} |"
        )
    pii_bench_rows = [
        "| "
        + " | ".join(
            (
                display,
                str(pii_bench_suites[suite]["documents"]),
                f"{pii_bench_suites[suite]['strict_micro_f1']:.8f}",
                f"{pii_bench_suites[suite]['strict_macro_f1']:.8f}",
                "N/A（未评测）",
            )
        )
        + " |"
        for suite, display in (("formal", "Formal"), ("chat", "Chat"), ("pooled", "Pooled"))
    ]

    text = f"""---
language:
- zh
library_name: transformers
pipeline_tag: token-classification
license: {training_facts["base_model_license"]}
base_model: {training_facts["base_model_id"]}
base_model_revision: {training_facts["base_model_revision"]}
tags:
- pii
- chinese
- token-classification
model_name: {DISPLAY_NAME}
package_version: {PACKAGE_VERSION}
publication_state: {PUBLICATION_STATE}
---

# {DISPLAY_NAME}

这是一个面向简体中文个人信息识别的 token-classification 模型候选，覆盖 24 个实体类型；
分类头包含 `O + 24×BIO`，因此共有 49 个 token 标签。模型配套规则、Presidio 兼容层与
级联服务。当前材料仅用于本地发布前审核，尚未发布，也不作行业领先性、现实世界泛化或
生产可用性声明。

## 使用方式

安装版本固定为 `{PACKAGE_NAME}=={PACKAGE_VERSION}`。模型包采用 Transformers 兼容的
safetensors 格式；服务模式使用同一模型、冻结校准配置和社区级联 profile。加载模型时应
启用离线选项，并从已审核的本地模型目录读取权重。

推荐通过发行 wheel 的字符级 span API 做检测。下面的输入使用仅用于文档示例的保留域名
`example.com`，不代表真实联系人或项目收件地址；输出只保留标签、字符起止位置和分数，不打印
原文或命中值。实际返回以当前已校验模型包为准，示例不构成对任意输入必然命中的承诺。命令应从
`model-package` 所在目录运行。

```python
from pathlib import Path

from pii_zh.inference import load_local_predictor

model_dir = Path("./model-package").resolve(strict=True)
predictor = load_local_predictor(model_dir, device="cpu", micro_batch_size=1)
synthetic_text = "测试邮箱 demo@example.com"
spans = predictor.predict(synthetic_text)
detections = [
    {{
        "label": span["entity_type"],
        "start": span["start"],
        "end": span["end"],
        "score": span["score"],
    }}
    for span in spans
]
print(detections)
```

如需验证原生 Transformers remote-code 兼容性，可运行下面的离线最小前向；这段低层示例返回
token logits，不负责字符 span 解码。只应对已经审核并校验过 checksum 的本地模型包启用
Transformers remote-code trust。

```python
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

model_dir = Path("./model-package").resolve(strict=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=True
)
config = AutoConfig.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=True
)
model = AutoModelForTokenClassification.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=True
)
assert config.pii_release_eligible is False
assert config.pii_attention_mode == "full"
assert config.num_labels == 49
model.eval()
inputs = tokenizer("这是合成测试文本。", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
assert logits.shape[-1] == 49 and torch.isfinite(logits).all().item()
print(tuple(logits.shape))
```

## 已绑定评测

| 轨道 | 系统 | 状态 | strict-span micro F1 | strict-span micro recall | PII-free document FPR |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

下表差值均为“候选减比较器”；区间是按文档配对 bootstrap 的 95% 置信区间。

| 轨道 | 比较器 | Δ micro F1 | 95% CI | Δ macro F1 | 95% CI | Δ document FPR | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

这些数字来自合成 Open-24 发布评测，并受独立重放门约束；它们不能替代真实业务数据、
跨域数据或人工标注盲测。Presidio、中文规则和大模型各有不同误报/漏报特征，推荐在应用侧保留
置信度、来源与审计信息。

### 附加公开测试已见结果（不参与选择或领先性声明）

PII Bench ZH 已被项目读取，下面是冻结后的 `model_raw` 单次 posthoc 描述性结果；公开测试暴露
为 `true`、`selection_allowed=false`，且没有通过描述性 active-comparator envelope。它与上面的
Open-24 轨道不是同一数据或 claim，不应混合比较。该次没有运行完整级联服务，因此服务结果为 N/A。

| suite | documents | strict micro F1 | strict macro F1 | full service |
|---|---:|---:|---:|---|
{chr(10).join(pii_bench_rows)}

## 24 类标签

精确且有序的 24 类为：`{"`, `".join(ordered_labels)}`。同一顺序由模型包内 `taxonomy.yaml`、
`id2label.json` 和 `config.json` 共同约束；模型卡不会用 taxonomy 之外的泛化类别替代它们。

## 基座模型与标签迁移

本候选的固定初始化来源是 `{training_facts["base_model_id"]}`，revision
`{training_facts["base_model_revision"]}`；其上游模型卡声明许可证为
`{training_facts["base_model_license"]}`。训练 manifest 记录了严格的 backbone state-dict
复制、`O` 标签行复制，以及 {training_facts["head_projection_count"]} 个源实体分类头到中文
core-24 标签的验证投影。具体源文件 SHA-256 与 12 类映射保留在模型包内
`training_manifest.json` 和 `THIRD_PARTY_NOTICES.md`。这份机械归属说明不替代发布前的人工
许可证复核。

## 训练数据事实与限制

训练集共 {training_data_facts["training_document_count"]:,} 条，其中 PII-free
{training_data_facts["training_pii_free_document_count"]:,} 条；开发验证集共
{training_data_facts["validation_document_count"]:,} 条，其中 PII-free
{training_data_facts["validation_pii_free_document_count"]:,} 条，两个 split 均覆盖 24 类。二者均为
100% 确定性合成、开源派生数据，不含真实、客户或生产 PII。模板资产包括本地 Qwen3-8B 辅助提出并
经人工筛选接受的 {training_data_facts["model_assisted_accepted_template_count"]} 个占位符模板、
{training_data_facts["human_authored_positive_template_count"]} 个人工正例模板和
{training_data_facts["human_authored_hard_negative_template_count"]} 个人工 hard-negative 模板。

这不是盲验证：冻结 manifest 明确记录 `validation_informed_template_family_overlap`，train/validation
存在 {training_data_facts["template_group_overlap_count"]} 个 template-group overlap；因此开发集数字
不能作为独立真实数据泛化证据。

## 限制与安全边界

- 当前证据仅覆盖公开开源与合成数据，且公开测试集存在暴露风险。
- 单独模型效果不是唯一目标；正式应用建议使用 Presidio、规则与模型级联。
- 输出需要按场景复核，不应直接作为法律、合规或访问控制结论。
- 发布、tag、Hub 上传、签名与许可证批准均需另行授权和留痕。

## 可复现性

候选身份由最终模型 manifest、服务配置 manifest、完整运行时源码清单、校准 bundle、质量回执、
两条轨道的候选/比较器/性能重放回执共同绑定。模型卡本身处于
`{PUBLICATION_STATE}`，不会预写 GitHub 或 Hugging Face 已发布状态。
"""
    lowered = text.casefold()
    if any(marker.casefold() in lowered for marker in FORBIDDEN_MODEL_CARD_MARKERS):
        raise CommunityReleaseArtifactError("rendered model card contains forbidden language")
    return text


def build_model_card_artifact(
    *,
    quality_receipt_path: Path,
    model_manifest_path: Path,
    service_manifest_path: Path,
    training_manifest_path: Path,
    pii_bench_report_path: Path,
    template_asset_path: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    if template_asset_path is None:
        template_asset_path = repository_root / COMMUNITY_TEMPLATE_ASSET_PATH
    quality_receipt, quality_payload = _load_self_hashed_json(
        quality_receipt_path, field="quality receipt", self_field="receipt_sha256"
    )
    model_manifest, model_payload = _load_self_hashed_json(
        model_manifest_path, field="final model manifest", self_field="manifest_sha256"
    )
    service_manifest, service_payload = _load_self_hashed_json(
        service_manifest_path,
        field="service configuration manifest",
        self_field="manifest_sha256",
    )
    training_manifest, training_payload = _load_self_hashed_json(
        training_manifest_path,
        field="seed-97 training manifest",
        self_field="manifest_sha256",
    )
    template_asset_payload = read_regular_file(
        template_asset_path,
        field="community synthetic template asset",
    )
    template_asset = strict_json_bytes(
        template_asset_payload,
        field="community synthetic template asset",
    )
    try:
        training_facts = build_release._community_training_initialization_facts(training_manifest)
        training_data_facts = build_release._community_training_data_facts(
            training_manifest, template_asset
        )
    except (TypeError, ValueError) as exc:
        raise CommunityReleaseArtifactError(
            "model card training initialization is not the frozen AIguard migration"
        ) from exc
    if model_manifest.get("training_manifest_file_sha256") != sha256_bytes(
        training_payload
    ) or model_manifest.get("training_manifest_sha256") != training_manifest.get("manifest_sha256"):
        raise CommunityReleaseArtifactError(
            "model card final-model/training-manifest binding is stale"
        )
    _pii_bench_report, pii_bench_payload, pii_bench_summary = _load_pii_bench_posthoc(
        pii_bench_report_path,
        training_manifest=training_manifest,
        training_payload=training_payload,
    )
    markdown = render_model_card(
        quality_receipt=quality_receipt,
        model_manifest=model_manifest,
        service_manifest=service_manifest,
        training_manifest=training_manifest,
        template_asset=template_asset,
        pii_bench_posthoc=pii_bench_summary,
    )
    return _base_document(
        "model_card",
        inputs={
            "quality_receipt": {
                "file_sha256": sha256_bytes(quality_payload),
                "size_bytes": len(quality_payload),
            },
            "final_model_manifest": {
                "file_sha256": sha256_bytes(model_payload),
                "size_bytes": len(model_payload),
            },
            "service_configuration_manifest": {
                "file_sha256": sha256_bytes(service_payload),
                "size_bytes": len(service_payload),
            },
            "training_manifest": {
                "file_sha256": sha256_bytes(training_payload),
                "size_bytes": len(training_payload),
            },
            "template_asset": {
                "file_sha256": sha256_bytes(template_asset_payload),
                "size_bytes": len(template_asset_payload),
            },
            "pii_bench_posthoc_report": {
                "file_sha256": sha256_bytes(pii_bench_payload),
                "size_bytes": len(pii_bench_payload),
            },
        },
        result={
            "media_type": "text/markdown; charset=utf-8",
            "markdown": markdown,
            "markdown_sha256": sha256_bytes(markdown.encode("utf-8")),
            "entity_type_count": 24,
            "token_label_count": 49,
            "quality_receipt_sha256": quality_receipt["receipt_sha256"],
            "model_manifest_file_sha256": sha256_bytes(model_payload),
            "service_manifest_file_sha256": sha256_bytes(service_payload),
            "training_manifest_file_sha256": sha256_bytes(training_payload),
            "training_manifest_sha256": training_manifest["manifest_sha256"],
            "base_model_id": training_facts["base_model_id"],
            "base_model_revision": training_facts["base_model_revision"],
            "base_model_license": training_facts["base_model_license"],
            "head_projection_count": training_facts["head_projection_count"],
            "ordered_entity_labels": list(model_manifest["ordered_labels"]),
            "training_data_facts": training_data_facts,
            "template_asset_file_sha256": sha256_bytes(template_asset_payload),
            "pii_bench_report_file_sha256": sha256_bytes(pii_bench_payload),
            "pii_bench_report_sha256": _pii_bench_report["report_sha256"],
            "pii_bench_posthoc": pii_bench_summary,
            "placeholder_count": 0,
            "forbidden_claims_absent": True,
        },
        repository_root=repository_root,
    )


def _logical_tree_id(relative: PurePosixPath, *, prefix: str) -> str:
    value = relative.as_posix()
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CommunityReleaseArtifactError("artifact tree contains an unsafe relative name")
    logical_id = f"{prefix}:{value.replace('/', ':')}"
    return _safe_id(logical_id, field="artifact tree logical ID")


def _inventory_tree(
    root: Path,
    *,
    prefix: str,
    reject_package_payloads: bool,
) -> dict[str, dict[str, int | str]]:
    try:
        root_meta = root.lstat()
        entries = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CommunityReleaseArtifactError("artifact tree is unavailable") from exc
    if stat.S_ISLNK(root_meta.st_mode) or not stat.S_ISDIR(root_meta.st_mode):
        raise CommunityReleaseArtifactError("artifact tree root is unsafe")
    inventory: dict[str, dict[str, int | str]] = {}
    for path in entries:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CommunityReleaseArtifactError("artifact tree contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CommunityReleaseArtifactError("artifact tree contains a special file")
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if reject_package_payloads and (
            path.suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES
            or any(part.lower() in {"raw", "raw_data", ".git"} for part in relative.parts)
        ):
            raise CommunityReleaseArtifactError("model package contains a private/unsafe payload")
        logical_id = _logical_tree_id(relative, prefix=prefix)
        if logical_id in inventory:
            raise CommunityReleaseArtifactError("artifact tree logical IDs collide")
        inventory[logical_id] = _large_file_binding(path, field=f"artifact file {logical_id}")
    if not inventory:
        raise CommunityReleaseArtifactError("artifact tree is empty")
    return dict(sorted(inventory.items()))


def _validate_model_package_support_files(
    model_package_root: Path,
    *,
    inventory: Mapping[str, Mapping[str, int | str]],
    config: Mapping[str, Any],
    expected_bio: Sequence[str],
    training_manifest: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    try:
        id2label = strict_json_bytes(
            read_regular_file(model_package_root / "id2label.json", field="model id2label"),
            field="model id2label",
        )
        special_tokens = strict_json_bytes(
            read_regular_file(
                model_package_root / "special_tokens_map.json",
                field="model special tokens",
            ),
            field="model special tokens",
        )
    except (CommunityContractError, OSError, TypeError, ValueError) as exc:
        raise CommunityReleaseArtifactError("model package metadata is invalid") from exc
    expected_id2label = {str(index): label for index, label in enumerate(expected_bio)}
    if (
        id2label != expected_id2label
        or config.get("id2label") != expected_id2label
        or config.get("label2id") != {label: index for index, label in enumerate(expected_bio)}
        or config.get("num_labels", len(expected_bio)) != len(expected_bio)
        or config.get("auto_map") != build_release.AUTO_MAP
        or not isinstance(special_tokens, Mapping)
    ):
        raise CommunityReleaseArtifactError("model package label/remote-code metadata drifted")

    remote_code_source = repository_root / "src/pii_zh/models/qwen3_bi.py"
    remote_code_generator = repository_root / "scripts/build_release.py"
    try:
        configuration_code, modeling_code = build_release.render_remote_code(
            remote_code_source,
            community_v2_preauthorization=True,
        )
    except (OSError, SyntaxError, ValueError) as exc:
        raise CommunityReleaseArtifactError("reviewed remote-code source is invalid") from exc
    expected_remote_code = {
        "configuration_qwen3_bi.py": configuration_code.encode("utf-8"),
        "modeling_qwen3_bi.py": modeling_code.encode("utf-8"),
    }
    for name, expected in expected_remote_code.items():
        if read_regular_file(model_package_root / name, field=f"model package {name}") != expected:
            raise CommunityReleaseArtifactError(
                f"model package {name} differs from reviewed repository source"
            )

    try:
        expected_package_support = {
            "NOTICE": build_release.render_community_v2_notice(dict(training_manifest)).encode(
                "utf-8"
            ),
            "THIRD_PARTY_NOTICES.md": build_release.render_community_v2_third_party_notices(
                dict(training_manifest)
            ).encode("utf-8"),
        }
    except (TypeError, ValueError) as exc:
        raise CommunityReleaseArtifactError(
            "community package attribution source is invalid"
        ) from exc
    for name, expected in expected_package_support.items():
        if read_regular_file(model_package_root / name, field=f"model package {name}") != expected:
            raise CommunityReleaseArtifactError(
                f"model package {name} differs from the seed-97 attribution renderer"
            )

    checksum_payload = read_regular_file(
        model_package_root / "checksums.txt", field="model package checksums"
    )
    try:
        checksum_text = checksum_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CommunityReleaseArtifactError("model package checksums are not UTF-8") from exc
    expected_lines: list[str] = []
    for logical_id, binding in sorted(inventory.items()):
        if logical_id == "pkg:checksums.txt":
            continue
        relative = logical_id.removeprefix("pkg:").replace(":", "/")
        expected_lines.append(f"{binding['file_sha256']}  {relative}")
    expected_checksum_text = "\n".join(expected_lines) + "\n"
    if checksum_text != expected_checksum_text:
        raise CommunityReleaseArtifactError("model package checksum manifest is incomplete")
    return {
        "remote_code_profile": build_release.COMMUNITY_V2_REMOTE_CODE_PROFILE,
        "remote_code_source_file_sha256": sha256_bytes(
            read_regular_file(remote_code_source, field="remote-code source")
        ),
        "remote_code_generator_file_sha256": sha256_bytes(
            read_regular_file(remote_code_generator, field="remote-code generator")
        ),
        "remote_code_files": {
            name: dict(inventory[f"pkg:{name}"]) for name in sorted(expected_remote_code)
        },
        "package_support_profile": build_release.COMMUNITY_V2_PACKAGE_SUPPORT_PROFILE,
        "package_support_files": {
            name: dict(inventory[f"pkg:{name}"]) for name in sorted(expected_package_support)
        },
    }


def build_model_package_manifest(
    *,
    model_package_root: Path,
    model_card_artifact_path: Path,
    final_model_manifest_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    model_card, model_card_payload = load_and_validate_artifact(
        model_card_artifact_path, artifact_id="model_card", repository_root=repository_root
    )
    final_model, final_model_payload = _load_self_hashed_json(
        final_model_manifest_path,
        field="final model manifest",
        self_field="manifest_sha256",
    )
    inventory = _inventory_tree(model_package_root, prefix="pkg", reject_package_payloads=True)
    if set(inventory) != MODEL_PACKAGE_REQUIRED_IDS:
        raise CommunityReleaseArtifactError(
            "model package must contain the exact seed-97 public-file inventory"
        )
    try:
        build_release.validate_safetensors(model_package_root / "model.safetensors")
    except (OSError, ValueError) as exc:
        raise CommunityReleaseArtifactError("model package safetensors is invalid") from exc
    readme_payload = read_regular_file(model_package_root / "README.md", field="model README")
    if sha256_bytes(readme_payload) != model_card["result"]["markdown_sha256"]:
        raise CommunityReleaseArtifactError("model package README differs from the model card")
    if readme_payload != model_card["result"]["markdown"].encode("utf-8"):
        raise CommunityReleaseArtifactError("model package README bytes are not canonical")
    for name, source in (
        ("LICENSE", repository_root / "LICENSE"),
        ("SECURITY.md", repository_root / "SECURITY.md"),
        ("taxonomy.yaml", repository_root / "src/pii_zh/taxonomy/taxonomy.yaml"),
    ):
        if read_regular_file(model_package_root / name, field=f"model package {name}") != (
            read_regular_file(source, field=f"repository {name}")
        ):
            raise CommunityReleaseArtifactError(f"model package {name} differs from the repository")
    config = strict_json_bytes(
        read_regular_file(model_package_root / "config.json", field="model package config"),
        field="model package config",
    )
    training_manifest = strict_json_bytes(
        read_regular_file(
            model_package_root / "training_manifest.json",
            field="model package training manifest",
        ),
        field="model package training manifest",
    )
    try:
        actual_model = model_provenance._model_identity(model_package_root)
    except (CommunityModelContractError, OSError, TypeError, ValueError) as exc:
        raise CommunityReleaseArtifactError(
            "model package is not a completed full-attention community model"
        ) from exc
    ordered_labels = final_model.get("ordered_labels")
    expected_bio = ["O"]
    if isinstance(ordered_labels, list):
        for label in ordered_labels:
            expected_bio.extend((f"B-{label}", f"I-{label}"))
    normalized_id2label = config.get("id2label")
    observed_bio = (
        [normalized_id2label.get(str(index)) for index in range(49)]
        if isinstance(normalized_id2label, Mapping)
        else []
    )
    generated_identity = _validate_model_package_support_files(
        model_package_root,
        inventory=inventory,
        config=config,
        expected_bio=expected_bio,
        training_manifest=training_manifest,
        repository_root=repository_root,
    )
    preauthorization = strict_json_bytes(
        read_regular_file(
            model_package_root / build_release.COMMUNITY_PREAUTHORIZATION_FILENAME,
            field="community model preauthorization",
        ),
        field="community model preauthorization",
    )
    preauthorization_sha256 = preauthorization.get("preauthorization_sha256")
    if (
        set(preauthorization)
        != {
            "schema_version",
            "status",
            "model_id",
            "seed",
            "entity_type_count",
            "token_label_count",
            "training_release_eligible",
            "config_release_eligible",
            "final_model_manifest_file_sha256",
            "final_model_manifest_sha256",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "model_identity_sha256",
            "output_artifact_sha256",
            "preauthorization_sha256",
        }
        or preauthorization.get("schema_version") != "pii-zh.community-v2-preauthorization.v1"
        or not isinstance(preauthorization_sha256, str)
        or preauthorization_sha256
        != canonical_json_hash(preauthorization, remove="preauthorization_sha256")
        or preauthorization.get("status") != build_release.COMMUNITY_PREAUTHORIZATION_STATE
        or preauthorization.get("model_id") != final_model.get("model_id")
        or preauthorization.get("seed") != 97
        or preauthorization.get("entity_type_count") != 24
        or preauthorization.get("token_label_count") != 49
        or preauthorization.get("training_release_eligible") is not False
        or preauthorization.get("config_release_eligible") is not False
        or preauthorization.get("final_model_manifest_file_sha256")
        != sha256_bytes(final_model_payload)
        or preauthorization.get("final_model_manifest_sha256") != final_model.get("manifest_sha256")
        or preauthorization.get("training_manifest_file_sha256")
        != actual_model.get("training_manifest_file_sha256")
        or preauthorization.get("training_manifest_sha256")
        != actual_model.get("training_manifest_sha256")
        or preauthorization.get("model_identity_sha256") != actual_model.get("identity_sha256")
        or preauthorization.get("output_artifact_sha256")
        != actual_model.get("output_artifact_sha256")
    ):
        raise CommunityReleaseArtifactError(
            "model package community preauthorization binding is invalid"
        )
    if (
        final_model.get("label_count") != 24
        or final_model.get("attention_mode") != "full"
        or len(ordered_labels or []) != 24
        or expected_bio != observed_bio
        or actual_model.get("label_count") != 49
        or config.get("pii_release_eligible") is not False
        or actual_model.get("training_manifest_file_sha256")
        != final_model.get("training_manifest_file_sha256")
        or actual_model.get("training_manifest_sha256")
        != final_model.get("training_manifest_sha256")
        or actual_model.get("identity_sha256") != final_model.get("model_identity_sha256")
        or actual_model.get("output_artifact_sha256") != final_model.get("artifact_sha256")
    ):
        raise CommunityReleaseArtifactError(
            "model package identity differs from the evaluated final model"
        )
    return _base_document(
        "model_package_manifest",
        inputs={
            "model_card": {
                "file_sha256": sha256_bytes(model_card_payload),
                "size_bytes": len(model_card_payload),
            },
            "final_model_manifest": {
                "file_sha256": sha256_bytes(final_model_payload),
                "size_bytes": len(final_model_payload),
            },
        },
        result={
            "format": "huggingface_safetensors_model_package_v1",
            "file_count": len(inventory),
            "inventory_sha256": canonical_json_hash(inventory),
            "files": inventory,
            "entity_type_count": 24,
            "token_label_count": 49,
            "training_release_eligible": False,
            "community_candidate_eligibility_source": "community_cascade_release_v2_contract",
            "training_manifest_file_sha256": actual_model["training_manifest_file_sha256"],
            "training_manifest_sha256": actual_model["training_manifest_sha256"],
            "model_identity_sha256": actual_model["identity_sha256"],
            "output_artifact_sha256": actual_model["output_artifact_sha256"],
            "weights_combined_sha256": actual_model["weights_combined_sha256"],
            "config_file_sha256": actual_model["config_file_sha256"],
            **generated_identity,
            "community_preauthorization_state": build_release.COMMUNITY_PREAUTHORIZATION_STATE,
            "community_preauthorization_sha256": preauthorization_sha256,
        },
        repository_root=repository_root,
    )


def build_technical_documentation_manifest(
    *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    inventory: dict[str, dict[str, int | str]] = {}
    document_paths = [repository_root / relative for relative in TECHNICAL_DOCUMENTS.values()]
    findings = scan_public_artifacts.scan_paths(document_paths)
    if findings:
        kinds = ",".join(sorted({finding.kind for finding in findings}))
        raise CommunityReleaseArtifactError(
            f"technical documentation contains public-scan finding(s): {kinds}"
        )
    for logical_id, relative in TECHNICAL_DOCUMENTS.items():
        inventory[logical_id] = _metadata_binding(
            repository_root / relative, field=f"technical document {logical_id}"
        )
    return _base_document(
        "technical_documentation_manifest",
        inputs=inventory,
        result={
            "format": "community_release_documentation_v1",
            "file_count": len(inventory),
            "inventory_sha256": canonical_json_hash(inventory),
            "files": inventory,
        },
        repository_root=repository_root,
    )


def _validated_wheel_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name.endswith("/")
        or path.is_absolute()
        or path.as_posix() != name
        or "\\" in name
        or ":" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CommunityReleaseArtifactError("wheel contains an unsafe member name")
    return name


def _wheel_logical_id(name: str) -> str:
    validated = _validated_wheel_member_name(name)
    return _safe_id(validated.replace("/", ":"), field="wheel member logical ID")


def _wheel_inventory(
    path: Path,
    *,
    expected_name: str | None = PACKAGE_NAME,
    expected_version: str | None = PACKAGE_VERSION,
    schema_safe_member_ids: bool = True,
    reject_release_data: bool = True,
    maximum_uncompressed_bytes: int = MAX_WHEEL_UNCOMPRESSED_BYTES,
) -> tuple[dict[str, dict[str, int | str]], str, str]:
    inventory: dict[str, dict[str, int | str]] = {}
    metadata_payload: bytes | None = None
    metadata_directory: str | None = None
    metadata_count = 0
    total_size = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                validated_name = _validated_wheel_member_name(info.filename)
                logical_id = (
                    _wheel_logical_id(validated_name) if schema_safe_member_ids else validated_name
                )
                if logical_id in inventory:
                    raise CommunityReleaseArtifactError("wheel contains duplicate logical members")
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG}:
                    raise CommunityReleaseArtifactError("wheel contains a non-regular member")
                if info.flag_bits & 0x1:
                    raise CommunityReleaseArtifactError("wheel contains an encrypted member")
                if (
                    reject_release_data
                    and PurePosixPath(info.filename).suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES
                ):
                    raise CommunityReleaseArtifactError("wheel contains forbidden release data")
                total_size += info.file_size
                if total_size > maximum_uncompressed_bytes:
                    raise CommunityReleaseArtifactError("wheel expands beyond the size limit")
                digest = hashlib.sha256()
                size = 0
                with archive.open(info, "r") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        size += len(block)
                if size != info.file_size:
                    raise CommunityReleaseArtifactError("wheel member size is unstable")
                inventory[logical_id] = {"file_sha256": digest.hexdigest(), "size_bytes": size}
                member_path = PurePosixPath(info.filename)
                if (
                    len(member_path.parts) == 2
                    and member_path.parts[0].endswith(".dist-info")
                    and member_path.parts[1] == "METADATA"
                ):
                    metadata_payload = archive.read(info)
                    metadata_directory = member_path.parts[0]
                    metadata_count += 1
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise CommunityReleaseArtifactError("wheel is unreadable or invalid") from exc
    if metadata_payload is None:
        raise CommunityReleaseArtifactError("wheel metadata is missing")
    if metadata_count != 1:
        raise CommunityReleaseArtifactError(
            "wheel contains multiple root distribution metadata files"
        )
    if metadata_directory is None:  # pragma: no cover - guarded by metadata_payload
        raise CommunityReleaseArtifactError("wheel metadata directory is missing")
    message = BytesParser(policy=email_policy).parsebytes(metadata_payload)
    name = message.get("Name")
    version = message.get("Version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise CommunityReleaseArtifactError("wheel name/version metadata is malformed")
    metadata_stem = metadata_directory.removesuffix(".dist-info")
    directory_name, separator, directory_version = metadata_stem.rpartition("-")
    if not separator or not directory_name or not directory_version:
        raise CommunityReleaseArtifactError("wheel root metadata directory identity is malformed")
    if _normalize_distribution_name(directory_name) != _normalize_distribution_name(name):
        raise CommunityReleaseArtifactError("wheel root metadata name differs from METADATA")
    if directory_version != version.replace("-", "_"):
        raise CommunityReleaseArtifactError("wheel root metadata version differs from METADATA")
    if not path.name.endswith(".whl") or not path.name.startswith(f"{metadata_stem}-"):
        raise CommunityReleaseArtifactError("wheel filename differs from root metadata identity")
    if (expected_name is not None and name != expected_name) or (
        expected_version is not None and version != expected_version
    ):
        raise CommunityReleaseArtifactError("wheel name/version differs from the successor release")
    return dict(sorted(inventory.items())), name, version


def build_wheel_manifest(
    *, wheel_path: Path, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    wheel_binding = _large_file_binding(wheel_path, field="built wheel")
    inventory, _name, _version = _wheel_inventory(wheel_path)
    return _base_document(
        "wheel_manifest",
        inputs={"wheel": wheel_binding},
        result={
            "format": "python_wheel_zip_v1",
            "file_count": len(inventory),
            "inventory_sha256": canonical_json_hash(inventory),
            "files": inventory,
        },
        repository_root=repository_root,
    )


def _normalize_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _wheelhouse_inventory(
    wheelhouse_root: Path,
) -> tuple[dict[str, dict[str, int | str]], dict[str, dict[str, int | str]]]:
    try:
        root_metadata = wheelhouse_root.lstat()
        entries = sorted(wheelhouse_root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise CommunityReleaseArtifactError("wheelhouse is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CommunityReleaseArtifactError("wheelhouse root is unsafe")
    if not entries:
        raise CommunityReleaseArtifactError("wheelhouse is empty")
    files: dict[str, dict[str, int | str]] = {}
    packages: dict[str, dict[str, int | str]] = {}
    for path in entries:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or path.suffix != ".whl"
        ):
            raise CommunityReleaseArtifactError("wheelhouse must contain only regular wheels")
        binding = _large_file_binding(path, field="wheelhouse wheel")
        member_inventory, name, version = _wheel_inventory(
            path,
            expected_name=None,
            expected_version=None,
            schema_safe_member_ids=False,
            reject_release_data=False,
            maximum_uncompressed_bytes=MAX_LARGE_FILE_BYTES,
        )
        normalized_name = _normalize_distribution_name(name)
        if normalized_name == _normalize_distribution_name(PACKAGE_NAME):
            raise CommunityReleaseArtifactError(
                "wheelhouse must not contain a second release-project wheel"
            )
        logical_id = _safe_id(f"wheel:{path.name}", field="wheelhouse wheel ID")
        package_id = _safe_id(f"package:{normalized_name}", field="wheelhouse package ID")
        if logical_id in files or package_id in packages:
            raise CommunityReleaseArtifactError("wheelhouse contains a duplicate package")
        files[logical_id] = binding
        packages[package_id] = {
            "name": name,
            "version": version,
            "wheel_file_sha256": binding["file_sha256"],
            "member_count": len(member_inventory),
            "member_inventory_sha256": canonical_json_hash(member_inventory),
        }
    return dict(sorted(files.items())), dict(sorted(packages.items()))


def build_wheelhouse_manifest(
    *,
    wheelhouse_root: Path,
    sbom_artifact_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    sbom, sbom_payload = _load_typed_input(
        sbom_artifact_path, artifact_id="sbom", repository_root=repository_root
    )
    files, packages = _wheelhouse_inventory(wheelhouse_root)
    sbom_versions = {
        (_normalize_distribution_name(str(component.get("name"))), str(component.get("version")))
        for component in sbom["result"]["bom"]["components"]
        if isinstance(component, Mapping)
    }
    if any(
        (_normalize_distribution_name(str(package["name"])), str(package["version"]))
        not in sbom_versions
        for package in packages.values()
    ):
        raise CommunityReleaseArtifactError("wheelhouse package/version is absent from the SBOM")
    return _base_document(
        "wheelhouse_manifest",
        inputs={
            "sbom": {
                "file_sha256": sha256_bytes(sbom_payload),
                "size_bytes": len(sbom_payload),
            }
        },
        result={
            "format": "locked_python_wheelhouse_v1",
            "wheel_count": len(files),
            "inventory_sha256": canonical_json_hash(files),
            "files": files,
            "packages_sha256": canonical_json_hash(packages),
            "packages": packages,
            "sbom_artifact_sha256": sbom["artifact_sha256"],
        },
        repository_root=repository_root,
    )


def _run_command(
    argv: Sequence[str], *, timeout_seconds: int, environment: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(argv),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=None if environment is None else dict(environment),
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommunityReleaseArtifactError("fixed artifact harness failed to execute") from exc


def _load_typed_input(
    path: Path,
    *,
    artifact_id: str,
    repository_root: Path,
    verify_current_repository_inputs: bool = True,
) -> tuple[dict[str, Any], bytes]:
    return load_and_validate_artifact(
        path,
        artifact_id=artifact_id,
        repository_root=repository_root,
        verify_current_repository_inputs=verify_current_repository_inputs,
    )


def build_container_manifest(
    *,
    image_ref: str,
    wheel_path: Path,
    wheel_manifest_path: Path,
    wheelhouse_manifest_path: Path,
    model_package_manifest_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    if not image_ref or len(image_ref) > 256 or any(character.isspace() for character in image_ref):
        raise CommunityReleaseArtifactError("container image reference is invalid")
    wheel, wheel_payload = _load_typed_input(
        wheel_manifest_path, artifact_id="wheel_manifest", repository_root=repository_root
    )
    actual_wheel = _large_file_binding(wheel_path, field="container release wheel")
    actual_wheel_inventory, _name, _version = _wheel_inventory(wheel_path)
    if actual_wheel != wheel["inputs"].get("wheel") or actual_wheel_inventory != wheel[
        "result"
    ].get("files"):
        raise CommunityReleaseArtifactError("container release wheel differs from its manifest")
    model_package, model_package_payload = _load_typed_input(
        model_package_manifest_path,
        artifact_id="model_package_manifest",
        repository_root=repository_root,
    )
    wheelhouse, wheelhouse_payload = _load_typed_input(
        wheelhouse_manifest_path,
        artifact_id="wheelhouse_manifest",
        repository_root=repository_root,
    )
    completed = _run_command(
        ["docker", "image", "inspect", "--format", "{{json .}}", "--", image_ref],
        timeout_seconds=60,
    )
    if completed.returncode != 0 or len(completed.stdout) > 16 * 1024 * 1024:
        raise CommunityReleaseArtifactError("docker image inspect did not succeed")
    image = strict_json_bytes(completed.stdout.strip(), field="docker image inspect")
    config = image.get("Config")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    environment = config.get("Env") if isinstance(config, Mapping) else None
    image_id = image.get("Id")
    repo_digests = image.get("RepoDigests", [])
    if (
        not isinstance(config, Mapping)
        or not isinstance(labels, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items()
        )
        or not isinstance(environment, list)
        or any(not isinstance(item, str) for item in environment)
        or not isinstance(repo_digests, list)
        or any(not isinstance(item, str) for item in repo_digests)
        or not isinstance(image_id, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
    ):
        raise CommunityReleaseArtifactError("docker image inspection shape is invalid")
    offline = {
        key: value
        for item in environment
        if "=" in item
        for key, value in [item.split("=", 1)]
        if key in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
    }
    required_labels = {
        "org.opencontainers.image.title": "pii-zh-community-cascade",
        "org.opencontainers.image.version": PACKAGE_VERSION,
        "org.pii-zh.wheel-artifact-sha256": wheel["artifact_sha256"],
        "org.pii-zh.wheel-file-sha256": actual_wheel["file_sha256"],
        "org.pii-zh.wheelhouse-artifact-sha256": wheelhouse["artifact_sha256"],
        "org.pii-zh.external-model-package-artifact-sha256": model_package["artifact_sha256"],
        "org.pii-zh.model-delivery": "external_read_only_mount",
    }
    if offline != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"} or any(
        labels.get(key) != value for key, value in required_labels.items()
    ):
        raise CommunityReleaseArtifactError("container lacks fixed offline or artifact labels")
    digest_values: list[str] = []
    for item in repo_digests:
        digest = item.rsplit("@", 1)[-1]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise CommunityReleaseArtifactError("container repo digest is invalid")
        digest_values.append(digest)
    return _base_document(
        "container_manifest",
        inputs={
            "wheel_manifest": {
                "file_sha256": sha256_bytes(wheel_payload),
                "size_bytes": len(wheel_payload),
            },
            "model_package_manifest": {
                "file_sha256": sha256_bytes(model_package_payload),
                "size_bytes": len(model_package_payload),
            },
            "wheelhouse_manifest": {
                "file_sha256": sha256_bytes(wheelhouse_payload),
                "size_bytes": len(wheelhouse_payload),
            },
        },
        result={
            "format": "docker_image_inspect_v1",
            "image_id": image_id,
            "content_digest_sha256": image_id.removeprefix("sha256:"),
            "repo_digest_sha256": canonical_json_hash({"digests": sorted(digest_values)}),
            "config_sha256": canonical_json_hash(config),
            "labels_sha256": canonical_json_hash(dict(sorted(labels.items()))),
            "offline_environment": offline,
            "bound_wheel_artifact_sha256": wheel["artifact_sha256"],
            "bound_wheel_file_sha256": actual_wheel["file_sha256"],
            "bound_wheelhouse_artifact_sha256": wheelhouse["artifact_sha256"],
            "external_model_package_artifact_sha256": model_package["artifact_sha256"],
            "model_delivery": "external_read_only_mount",
        },
        repository_root=repository_root,
    )


def build_sbom_artifact(
    *,
    lockfile_path: Path,
    pyproject_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    lock_binding = _metadata_binding(lockfile_path, field="locked dependency graph")
    project_binding = _metadata_binding(pyproject_path, field="project metadata")
    try:
        bom = generate_sbom.build_sbom(lockfile_path, pyproject_path)
    except (OSError, TypeError, ValueError) as exc:
        raise CommunityReleaseArtifactError("CycloneDX SBOM generation failed") from exc
    metadata = bom.get("metadata")
    root_component = metadata.get("component") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(root_component, Mapping)
        or root_component.get("name") != PACKAGE_NAME
        or root_component.get("version") != PACKAGE_VERSION
    ):
        raise CommunityReleaseArtifactError(
            "project metadata must be bumped to the fixed successor version before SBOM creation"
        )
    components = bom.get("components")
    dependencies = bom.get("dependencies")
    if not isinstance(components, list) or not isinstance(dependencies, list):
        raise CommunityReleaseArtifactError("generated CycloneDX graph is incomplete")
    edge_count = sum(
        len(item.get("dependsOn", [])) for item in dependencies if isinstance(item, Mapping)
    )
    return _base_document(
        "sbom",
        inputs={"uv_lock": lock_binding, "pyproject": project_binding},
        result={
            "format": "CycloneDX-1.6",
            "bom_sha256": canonical_json_hash(bom),
            "component_count": len(components),
            "dependency_edge_count": edge_count,
            "bom": bom,
        },
        repository_root=repository_root,
    )


def _license_expression(component: Mapping[str, Any]) -> str | None:
    licenses = component.get("licenses")
    if not isinstance(licenses, list):
        return None
    values: list[str] = []
    for item in licenses:
        if not isinstance(item, Mapping):
            continue
        expression = item.get("expression")
        if isinstance(expression, str) and expression.strip():
            values.append(expression.strip())
        license_item = item.get("license")
        if isinstance(license_item, Mapping):
            name = license_item.get("id", license_item.get("name"))
            if isinstance(name, str) and name.strip():
                values.append(name.strip())
    return " AND ".join(sorted(set(values))) if values else None


def build_license_report(
    *,
    sbom_artifact_path: Path,
    model_package_root: Path,
    repository_root: Path = REPOSITORY_ROOT,
    verify_current_repository_inputs: bool = True,
) -> dict[str, Any]:
    sbom, sbom_payload = _load_typed_input(
        sbom_artifact_path,
        artifact_id="sbom",
        repository_root=repository_root,
        verify_current_repository_inputs=verify_current_repository_inputs,
    )
    license_path = model_package_root / "LICENSE"
    notice_path = model_package_root / "NOTICE"
    third_party_notices_path = model_package_root / "THIRD_PARTY_NOTICES.md"
    documents = {
        "license_document": _metadata_binding(license_path, field="model package license"),
        "notice_document": _metadata_binding(notice_path, field="model package notice"),
        "third_party_notices": _metadata_binding(
            third_party_notices_path, field="model package third-party notices"
        ),
    }
    components = sbom["result"]["bom"]["components"]
    inventory: dict[str, str] = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise CommunityReleaseArtifactError("SBOM contains a malformed component")
        reference = component.get("bom-ref")
        if not isinstance(reference, str):
            raise CommunityReleaseArtifactError("SBOM component reference is missing")
        logical_id = "component:" + hashlib.sha256(reference.encode("utf-8")).hexdigest()
        inventory[logical_id] = _license_expression(component) or "UNKNOWN_REQUIRES_HUMAN_REVIEW"
    declared = sum(value != "UNKNOWN_REQUIRES_HUMAN_REVIEW" for value in inventory.values())
    unknown = len(inventory) - declared
    return _base_document(
        "license_report",
        inputs={
            "sbom": {"file_sha256": sha256_bytes(sbom_payload), "size_bytes": len(sbom_payload)},
            **documents,
        },
        result={
            "format": "factual_license_inventory_v1",
            "component_count": len(inventory),
            "declared_license_count": declared,
            "unknown_license_count": unknown,
            "license_inventory_sha256": canonical_json_hash(inventory),
            "license_inventory": dict(sorted(inventory.items())),
            "human_approval_status": "pending",
            "legal_review_status": "pending",
            "automated_release_clearance": False,
        },
        status="COMPLETE_HUMAN_APPROVAL_PENDING",
        repository_root=repository_root,
    )


def _merge_scan_inventory(
    *,
    model_package_root: Path,
    wheel_path: Path,
    bound_artifacts: Mapping[str, Path],
    repository_root: Path,
) -> dict[str, dict[str, int | str]]:
    inventory: dict[str, dict[str, int | str]] = {}
    inventory.update(
        _inventory_tree(model_package_root, prefix="model", reject_package_payloads=False)
    )
    wheel_inventory, _name, _version = _wheel_inventory(wheel_path)
    for logical_id, binding in wheel_inventory.items():
        inventory[_safe_id(f"wheel:{logical_id}", field="public wheel member ID")] = binding
    for logical_id, relative in TECHNICAL_DOCUMENTS.items():
        inventory[_safe_id(f"doc:{logical_id}", field="public document ID")] = _metadata_binding(
            repository_root / relative, field=f"public technical document {logical_id}"
        )
    for logical_id, path in sorted(bound_artifacts.items()):
        safe = _safe_id(f"artifact:{logical_id}", field="public scan artifact ID")
        if safe in inventory:
            raise CommunityReleaseArtifactError("public scan artifact inventory collides")
        inventory[safe] = _metadata_binding(path, field=f"public artifact {logical_id}")
    return dict(sorted(inventory.items()))


def build_public_artifact_scan(
    *,
    model_package_root: Path,
    wheel_path: Path,
    wheelhouse_root: Path,
    bound_artifacts: Mapping[str, Path],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    if not bound_artifacts:
        raise CommunityReleaseArtifactError("public scan requires the bound artifact set")
    if set(bound_artifacts) != {
        "model_card",
        "model_package_manifest",
        "technical_documentation_manifest",
        "wheel_manifest",
        "wheelhouse_manifest",
        "container_manifest",
        "sbom",
        "license_report",
        "dependency_scan",
    }:
        raise CommunityReleaseArtifactError("public scan artifact inventory is not exact")
    for artifact_id, path in bound_artifacts.items():
        load_and_validate_artifact(path, artifact_id=artifact_id, repository_root=repository_root)
    model_manifest = load_and_validate_artifact(
        bound_artifacts["model_package_manifest"],
        artifact_id="model_package_manifest",
        repository_root=repository_root,
    )[0]
    actual_model_inventory = _inventory_tree(
        model_package_root, prefix="pkg", reject_package_payloads=True
    )
    if actual_model_inventory != model_manifest["result"]["files"]:
        raise CommunityReleaseArtifactError("public scan model package differs from its manifest")
    license_report = load_and_validate_artifact(
        bound_artifacts["license_report"],
        artifact_id="license_report",
        repository_root=repository_root,
    )[0]
    expected_license_documents = {
        "license_document": model_manifest["result"]["files"]["pkg:LICENSE"],
        "notice_document": model_manifest["result"]["files"]["pkg:NOTICE"],
        "third_party_notices": model_manifest["result"]["files"]["pkg:THIRD_PARTY_NOTICES.md"],
    }
    if {
        key: license_report["inputs"].get(key) for key in expected_license_documents
    } != expected_license_documents:
        raise CommunityReleaseArtifactError(
            "public scan license report differs from model-package legal documents"
        )
    wheel_manifest = load_and_validate_artifact(
        bound_artifacts["wheel_manifest"],
        artifact_id="wheel_manifest",
        repository_root=repository_root,
    )[0]
    actual_wheel_binding = _large_file_binding(wheel_path, field="public release wheel")
    actual_wheel_inventory, _wheel_name, _wheel_version = _wheel_inventory(wheel_path)
    if actual_wheel_binding != wheel_manifest["inputs"].get("wheel"):
        raise CommunityReleaseArtifactError("public scan wheel differs from its manifest")
    if actual_wheel_inventory != wheel_manifest["result"]["files"]:
        raise CommunityReleaseArtifactError("public scan wheel inventory differs from its manifest")
    wheelhouse_manifest = load_and_validate_artifact(
        bound_artifacts["wheelhouse_manifest"],
        artifact_id="wheelhouse_manifest",
        repository_root=repository_root,
    )[0]
    wheelhouse_files, _wheelhouse_packages = _wheelhouse_inventory(wheelhouse_root)
    if wheelhouse_files != wheelhouse_manifest["result"]["files"]:
        raise CommunityReleaseArtifactError("public scan wheelhouse differs from its manifest")

    with tempfile.TemporaryDirectory(prefix="pii-zh-public-wheel-") as temporary_directory:
        extracted = Path(temporary_directory) / "release-wheel"
        extracted.mkdir()
        scan_roots = [extracted]
        try:
            with zipfile.ZipFile(wheel_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    _wheel_logical_id(info.filename)
                    destination = extracted.joinpath(*PurePosixPath(info.filename).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info, "r") as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise CommunityReleaseArtifactError("public wheel extraction failed") from exc
        findings = scan_public_artifacts.scan_paths(
            [
                model_package_root,
                *scan_roots,
                *(repository_root / relative for relative in TECHNICAL_DOCUMENTS.values()),
                *bound_artifacts.values(),
            ]
        )
    if findings:
        raise CommunityReleaseArtifactError(
            f"public artifact scan found {len(findings)} non-allowlisted finding(s)"
        )
    inventory = _merge_scan_inventory(
        model_package_root=model_package_root,
        wheel_path=wheel_path,
        bound_artifacts=bound_artifacts,
        repository_root=repository_root,
    )
    dependency_inventory = {
        _safe_id(f"wheelhouse:{logical_id}", field="public wheelhouse file ID"): binding
        for logical_id, binding in wheelhouse_files.items()
    }
    dependency_inventory = dict(sorted(dependency_inventory.items()))
    scanner_hash = sha256_bytes(
        read_regular_file(
            repository_root / "scripts/scan_public_artifacts.py",
            field="public artifact scanner implementation",
        )
    )
    inputs = {
        f"artifact:{logical_id}": _metadata_binding(path, field=f"public artifact {logical_id}")
        for logical_id, path in bound_artifacts.items()
    }
    return _base_document(
        "public_artifact_scan",
        inputs=inputs,
        result={
            "format": "pii_zh_public_artifact_scan_v3",
            "scanner_file_sha256": scanner_hash,
            "deep_scanned_file_count": len(inventory),
            "deep_scanned_inventory_sha256": canonical_json_hash(inventory),
            "deep_scanned_inventory": inventory,
            "hashed_dependency_wheel_count": len(dependency_inventory),
            "hashed_dependency_wheel_inventory_sha256": canonical_json_hash(dependency_inventory),
            "hashed_dependency_wheel_inventory": dependency_inventory,
            "finding_count": 0,
            "finding_kinds": [],
        },
        repository_root=repository_root,
    )


def _osv_vulnerability_count(report: Mapping[str, Any]) -> tuple[int, int]:
    results = report.get("results")
    if not isinstance(results, list):
        raise CommunityReleaseArtifactError("OSV JSON report has no results list")
    count = 0
    for result in results:
        if not isinstance(result, Mapping):
            raise CommunityReleaseArtifactError("OSV JSON result is malformed")
        packages = result.get("packages", [])
        if not isinstance(packages, list):
            raise CommunityReleaseArtifactError("OSV JSON packages are malformed")
        for package in packages:
            if not isinstance(package, Mapping):
                raise CommunityReleaseArtifactError("OSV JSON package is malformed")
            vulnerabilities = package.get("vulnerabilities", [])
            if not isinstance(vulnerabilities, list):
                raise CommunityReleaseArtifactError("OSV vulnerabilities are malformed")
            count += len(vulnerabilities)
    return len(results), count


def _parse_osv_scanner_version(stdout: bytes, stderr: bytes) -> str:
    """Parse the official four-line v2 version report into one SemVer value."""

    if len(stdout) > 1024 or len(stderr) > 1024 or stderr.strip():
        raise CommunityReleaseArtifactError("osv-scanner version output is unsafe")
    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise CommunityReleaseArtifactError("osv-scanner version output is unsafe") from exc
    patterns = (
        r"osv-scanner version: ([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?)",
        r"osv-scalibr version: [0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?",
        r"commit: [0-9a-f]{40}",
        r"built at: [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
    )
    if len(lines) != len(patterns):
        raise CommunityReleaseArtifactError("osv-scanner version output is unsafe")
    matches = [re.fullmatch(pattern, line) for pattern, line in zip(patterns, lines, strict=True)]
    if any(match is None for match in matches):
        raise CommunityReleaseArtifactError("osv-scanner version output is unsafe")
    version = matches[0].group(1) if matches[0] is not None else None
    if version is None:  # pragma: no cover - guarded by the exact matches above
        raise CommunityReleaseArtifactError("osv-scanner version output is unsafe")
    return version


def build_dependency_scan(
    *,
    sbom_artifact_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    sbom, sbom_payload = _load_typed_input(
        sbom_artifact_path, artifact_id="sbom", repository_root=repository_root
    )
    executable = shutil.which("osv-scanner")
    if executable is None:
        raise CommunityReleaseArtifactError("osv-scanner is not installed")
    executable_path = Path(executable)
    scanner_binding = _large_file_binding(executable_path, field="osv-scanner executable")
    version_run = _run_command([executable, "--version"], timeout_seconds=30)
    if version_run.returncode != 0:
        raise CommunityReleaseArtifactError("osv-scanner version query failed")
    version = _parse_osv_scanner_version(version_run.stdout, version_run.stderr)
    with tempfile.TemporaryDirectory(prefix="pii-zh-osv-") as temporary_directory:
        sbom_path = Path(temporary_directory) / "release.cdx.json"
        sbom_path.write_text(
            json.dumps(sbom["result"]["bom"], ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        argv = [executable, "scan", "--format", "json", "-L", str(sbom_path)]
        completed = _run_command(argv, timeout_seconds=900)
    if completed.returncode != 0 or len(completed.stdout) > 64 * 1024 * 1024:
        raise CommunityReleaseArtifactError(
            "OSV dependency scan did not return a zero-finding PASS"
        )
    report = strict_json_bytes(completed.stdout, field="OSV JSON report")
    result_count, vulnerability_count = _osv_vulnerability_count(report)
    if vulnerability_count:
        raise CommunityReleaseArtifactError("OSV dependency scan found known vulnerabilities")
    semantic_argv = ["osv-scanner", "scan", "--format", "json", "-L", "<CYCLONEDX_SBOM>"]
    return _base_document(
        "dependency_scan",
        inputs={
            "sbom": {"file_sha256": sha256_bytes(sbom_payload), "size_bytes": len(sbom_payload)}
        },
        result={
            "format": "osv_scanner_json_v2",
            "scanner_name": "osv-scanner",
            "scanner_version": version,
            "scanner_file_sha256": scanner_binding["file_sha256"],
            "command_id": "osv_scanner_v2_cyclonedx_json",
            "command_argv_sha256": canonical_json_hash({"argv": semantic_argv}),
            "sbom_artifact_sha256": sbom["artifact_sha256"],
            "sbom_content_sha256": sbom["result"]["bom_sha256"],
            "component_count": sbom["result"]["component_count"],
            "result_count": result_count,
            "vulnerability_count": 0,
            "raw_report_sha256": sha256_bytes(completed.stdout),
        },
        network_used=True,
        repository_root=repository_root,
    )


def _parse_named_paths(values: Sequence[str], *, field: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CommunityReleaseArtifactError(f"{field} must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        _safe_id(name, field=f"{field} name")
        if not raw_path or name in result:
            raise CommunityReleaseArtifactError(f"{field} contains an invalid entry")
        result[name] = Path(raw_path)
    return result


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_card = subparsers.add_parser("model-card", help="build the fixed preauth model card")
    model_card.add_argument("--quality-receipt", required=True, type=Path)
    model_card.add_argument("--final-model-manifest", required=True, type=Path)
    model_card.add_argument("--service-configuration-manifest", required=True, type=Path)
    model_card.add_argument("--training-manifest", required=True, type=Path)
    model_card.add_argument("--pii-bench-report", required=True, type=Path)
    model_card.add_argument(
        "--template-asset",
        type=Path,
        default=REPOSITORY_ROOT / COMMUNITY_TEMPLATE_ASSET_PATH,
    )
    model_card.add_argument("--markdown-output", required=True, type=Path)
    _add_output(model_card)

    model_package = subparsers.add_parser(
        "model-package-manifest", help="inventory an actual Hugging Face model package"
    )
    model_package.add_argument("--model-package-root", required=True, type=Path)
    model_package.add_argument("--model-card-artifact", required=True, type=Path)
    model_package.add_argument("--final-model-manifest", required=True, type=Path)
    _add_output(model_package)

    documentation = subparsers.add_parser(
        "technical-documentation-manifest", help="bind the fixed technical document set"
    )
    _add_output(documentation)

    wheel = subparsers.add_parser("wheel-manifest", help="inspect a built successor wheel")
    wheel.add_argument("--wheel", required=True, type=Path)
    _add_output(wheel)

    wheelhouse = subparsers.add_parser(
        "wheelhouse-manifest", help="inventory the exact offline dependency wheelhouse"
    )
    wheelhouse.add_argument("--wheelhouse", required=True, type=Path)
    wheelhouse.add_argument("--sbom-artifact", required=True, type=Path)
    _add_output(wheelhouse)

    container = subparsers.add_parser(
        "container-manifest", help="inspect a locally built, labeled offline image"
    )
    container.add_argument("--image-ref", required=True)
    container.add_argument("--wheel", required=True, type=Path)
    container.add_argument("--wheel-manifest", required=True, type=Path)
    container.add_argument("--wheelhouse-manifest", required=True, type=Path)
    container.add_argument("--model-package-manifest", required=True, type=Path)
    _add_output(container)

    sbom = subparsers.add_parser("sbom", help="build a CycloneDX 1.6 SBOM evidence artifact")
    sbom.add_argument("--lockfile", type=Path, default=REPOSITORY_ROOT / "uv.lock")
    sbom.add_argument("--pyproject", type=Path, default=REPOSITORY_ROOT / "pyproject.toml")
    _add_output(sbom)

    license_report = subparsers.add_parser(
        "license-report", help="build a factual license inventory with pending human review"
    )
    license_report.add_argument("--sbom-artifact", required=True, type=Path)
    license_report.add_argument(
        "--model-package-root",
        required=True,
        type=Path,
        help="exact community model package containing LICENSE/NOTICE/THIRD_PARTY_NOTICES.md",
    )
    _add_output(license_report)

    dependency = subparsers.add_parser(
        "dependency-scan", help="run the fixed OSV scanner against the bound CycloneDX SBOM"
    )
    dependency.add_argument("--sbom-artifact", required=True, type=Path)
    _add_output(dependency)

    public_scan = subparsers.add_parser(
        "public-artifact-scan", help="scan the complete public package and artifact set"
    )
    public_scan.add_argument("--model-package-root", required=True, type=Path)
    public_scan.add_argument("--wheel", required=True, type=Path)
    public_scan.add_argument("--wheelhouse", required=True, type=Path)
    public_scan.add_argument("--artifact", action="append", required=True)
    _add_output(public_scan)

    validate = subparsers.add_parser("validate", help="validate one typed artifact")
    validate.add_argument("--artifact-id", required=True, choices=ARTIFACT_IDS)
    validate.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            document, payload = load_and_validate_artifact(args.input, artifact_id=args.artifact_id)
            print(
                json.dumps(
                    {
                        "artifact_id": document["artifact_id"],
                        "artifact_sha256": document["artifact_sha256"],
                        "file_sha256": sha256_bytes(payload),
                        "status": document["status"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "model-card":
            document = build_model_card_artifact(
                quality_receipt_path=args.quality_receipt,
                model_manifest_path=args.final_model_manifest,
                service_manifest_path=args.service_configuration_manifest,
                training_manifest_path=args.training_manifest,
                pii_bench_report_path=args.pii_bench_report,
                template_asset_path=args.template_asset,
            )
        elif args.command == "model-package-manifest":
            document = build_model_package_manifest(
                model_package_root=args.model_package_root,
                model_card_artifact_path=args.model_card_artifact,
                final_model_manifest_path=args.final_model_manifest,
            )
        elif args.command == "technical-documentation-manifest":
            document = build_technical_documentation_manifest()
        elif args.command == "wheel-manifest":
            document = build_wheel_manifest(wheel_path=args.wheel)
        elif args.command == "wheelhouse-manifest":
            document = build_wheelhouse_manifest(
                wheelhouse_root=args.wheelhouse,
                sbom_artifact_path=args.sbom_artifact,
            )
        elif args.command == "container-manifest":
            document = build_container_manifest(
                image_ref=args.image_ref,
                wheel_path=args.wheel,
                wheel_manifest_path=args.wheel_manifest,
                wheelhouse_manifest_path=args.wheelhouse_manifest,
                model_package_manifest_path=args.model_package_manifest,
            )
        elif args.command == "sbom":
            document = build_sbom_artifact(
                lockfile_path=args.lockfile, pyproject_path=args.pyproject
            )
        elif args.command == "license-report":
            document = build_license_report(
                sbom_artifact_path=args.sbom_artifact,
                model_package_root=args.model_package_root,
            )
        elif args.command == "dependency-scan":
            document = build_dependency_scan(sbom_artifact_path=args.sbom_artifact)
        elif args.command == "public-artifact-scan":
            document = build_public_artifact_scan(
                model_package_root=args.model_package_root,
                wheel_path=args.wheel,
                wheelhouse_root=args.wheelhouse,
                bound_artifacts=_parse_named_paths(args.artifact, field="artifact"),
            )
        else:  # pragma: no cover - argparse closes the command set
            raise CommunityReleaseArtifactError("unsupported artifact command")
        if args.command == "model-card":
            publish_model_card_outputs(args.output, args.markdown_output, document)
        else:
            publish_artifact(args.output, document)
        print(
            json.dumps(
                {
                    "artifact_id": document["artifact_id"],
                    "artifact_sha256": document["artifact_sha256"],
                    "status": document["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, CommunityContractError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
