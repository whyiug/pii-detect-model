#!/usr/bin/env python3
"""Validate the path-free community model + cascade release contract v2.

The v2 gate consumes the aggregate Open-24 quality receipt, but never treats
the producer's ``*_replay_required`` booleans as proof.  A release candidate
can pass only when seven independently content-addressed replay receipts are
present: one quality strict replay plus candidate, comparator and performance
replays for both canonical tracks.

This validator never reads evaluation JSONL or raw records, uses a GPU, accesses
a network, or publishes anything.  Its physical artifact closure does read and
hash model-weight bytes without loading the model for inference.  Paths are
invocation-only inputs and are never serialized into the contract or receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts import produce_community_cascade_release_v2_artifacts as release_artifacts
    from scripts import produce_community_cascade_release_v2_verifications as release_verifications
    from scripts import produce_public_synthetic_service_quality_v2 as quality
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
    import produce_community_cascade_release_v2_artifacts as release_artifacts  # type: ignore[no-redef]
    import produce_community_cascade_release_v2_verifications as release_verifications  # type: ignore[no-redef]
    import produce_public_synthetic_service_quality_v2 as quality  # type: ignore[no-redef]
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
CONTRACT_PATH = "configs/release/community_cascade_release_v2.json"
CONTRACT_SCHEMA_PATH = "configs/release/community_cascade_release_v2.schema.json"
RECEIPT_SCHEMA_PATH = "configs/release/community_cascade_release_v2.receipt.schema.json"
REPLAY_SCHEMA_PATH = "configs/release/community_cascade_release_v2.replay-receipt.schema.json"
SOURCE_SCHEMA_PATH = "configs/release/community_service_source_identity_v4.schema.json"
PREDECESSOR_SOURCE_SCHEMA_PATH = "configs/release/community_service_source_identity_v3.schema.json"
QUALITY_REPLAY_EVIDENCE_SCHEMA_PATH = (
    "configs/release/community_cascade_release_v2.quality-replay-evidence.schema.json"
)
CANDIDATE_REPLAY_EVIDENCE_SCHEMA_PATH = (
    "configs/release/community_cascade_release_v2.candidate-replay-evidence.schema.json"
)
COMPARATOR_REPLAY_EVIDENCE_SCHEMA_PATH = (
    "configs/evaluation/open24_comparator_execution_replay_v1.schema.json"
)
VALIDATOR_PATH = "scripts/validate_community_cascade_release_v2.py"
SUPPORT_PATH = "scripts/community_contract_utils.py"
EVIDENCE_PRODUCER_PATH = "scripts/produce_community_cascade_release_v2_evidence.py"
ARTIFACT_PRODUCER_PATH = release_artifacts.PRODUCER_PATH
ARTIFACT_SCHEMA_PATH = release_artifacts.SCHEMA_PATH
VERIFICATION_PRODUCER_PATH = release_verifications.PRODUCER_PATH
VERIFICATION_SCHEMA_PATH = release_verifications.SCHEMA_PATH
CONTRACT_BUILDER_PATH = "scripts/build_community_cascade_release_v2_contract.py"
MODEL_PACKAGE_BUILDER_PATH = "scripts/build_release.py"
PUBLIC_ARTIFACT_SCANNER_PATH = "scripts/scan_public_artifacts.py"
PERFORMANCE_HARNESS_MODULE_PATH = "src/pii_zh/evaluation/release_eval_v2_performance.py"
PERFORMANCE_HARNESS_CLI_PATH = "scripts/benchmark_release_eval_v2_candidate.py"
STAGE_GATE_MODULE_PATH = "src/pii_zh/evaluation/release_eval_v2_stage_gate.py"
STAGE_GATE_CLI_PATH = "scripts/release_eval_v2_stage_gate.py"
CALIBRATION_AUTHORIZATION_SCHEMA_PATH = (
    "configs/evaluation/release_eval_v2_calibration_authorization_v1.schema.json"
)
INTERNAL_UNLOCK_SCHEMA_PATH = "configs/evaluation/release_eval_v2_internal_unlock_v1.schema.json"

CONTRACT_ID = "community_cascade_release_v2"
CONTRACT_SCHEMA_VERSION = "pii-zh.community-cascade-release-contract.v2"
RECEIPT_SCHEMA_VERSION = "pii-zh.community-cascade-release-receipt.v2"
REPLAY_SCHEMA_VERSION = "pii-zh.community-cascade-replay-receipt.v2"
SOURCE_SCHEMA_VERSION = "pii-zh.community-service-source-identity.v4"

_PACKAGE_SOURCE_ROOT = Path("src/pii_zh")
_PACKAGE_SOURCE_SUFFIXES = frozenset({".json", ".py", ".yaml", ".yml"})

TRACKS = ("model_raw", "full_system")
REPLAY_KINDS = (
    "candidate_prediction_full_replay",
    "comparator_generation_replay",
    "performance_harness_replay",
)
ARTIFACT_IDS = (
    "final_model_manifest",
    "model_card",
    "model_package_manifest",
    "service_configuration_manifest",
    "service_source_manifest",
    "technical_documentation_manifest",
    "wheel_manifest",
    "wheelhouse_manifest",
    "container_manifest",
    "sbom",
    "license_report",
    "benchmark_report",
    "public_artifact_scan",
    "dependency_scan",
)
VERIFICATION_IDS = (
    "unit_tests",
    "clean_wheel_smoke",
    "container_smoke",
    "offline_model_smoke",
    "offline_service_smoke",
)

IMPLEMENTATION_PATHS: Mapping[str, str] = {
    "contract_schema": CONTRACT_SCHEMA_PATH,
    "release_receipt_schema": RECEIPT_SCHEMA_PATH,
    "replay_receipt_schema": REPLAY_SCHEMA_PATH,
    "source_identity_schema": SOURCE_SCHEMA_PATH,
    "source_identity_predecessor_schema": PREDECESSOR_SOURCE_SCHEMA_PATH,
    "artifact_evidence_schema": ARTIFACT_SCHEMA_PATH,
    "verification_receipt_schema": VERIFICATION_SCHEMA_PATH,
    "quality_replay_evidence_schema": QUALITY_REPLAY_EVIDENCE_SCHEMA_PATH,
    "candidate_replay_evidence_schema": CANDIDATE_REPLAY_EVIDENCE_SCHEMA_PATH,
    "comparator_replay_evidence_schema": COMPARATOR_REPLAY_EVIDENCE_SCHEMA_PATH,
    "validator": VALIDATOR_PATH,
    "support": SUPPORT_PATH,
    "evidence_producer": EVIDENCE_PRODUCER_PATH,
    "artifact_producer": ARTIFACT_PRODUCER_PATH,
    "verification_producer": VERIFICATION_PRODUCER_PATH,
    "contract_builder": CONTRACT_BUILDER_PATH,
    "model_package_builder": MODEL_PACKAGE_BUILDER_PATH,
    "public_artifact_scanner": PUBLIC_ARTIFACT_SCANNER_PATH,
    "performance_harness_module": PERFORMANCE_HARNESS_MODULE_PATH,
    "performance_harness_cli": PERFORMANCE_HARNESS_CLI_PATH,
    "quality_producer": quality.PRODUCER_PATH,
    "quality_result_schema": quality.SCHEMA_PATH,
    "quality_registry_schema": quality.REGISTRY_SCHEMA_PATH,
    "candidate_provenance_schema": quality.CANDIDATE_PREDICTION_SCHEMA_PATH,
    "candidate_provenance_module": quality.CANDIDATE_PREDICTION_MODULE_PATH,
    "candidate_provenance_cli": quality.CANDIDATE_PREDICTION_CLI_PATH,
    "comparator_provenance_schema": quality.COMPARATOR_PREDICTION_SCHEMA_PATH,
    "stage_gate_module": STAGE_GATE_MODULE_PATH,
    "stage_gate_cli": STAGE_GATE_CLI_PATH,
    "calibration_authorization_schema": CALIBRATION_AUTHORIZATION_SCHEMA_PATH,
    "internal_unlock_schema": INTERNAL_UNLOCK_SCHEMA_PATH,
}

EXPECTED_SCOPE = {
    "allowed_data": ["public_open_source", "synthetic"],
    "community_release": True,
    "human_evidence_required": False,
    "production_ready": False,
    "public_test_exposed": True,
    "real_world_sota": False,
}
EXPECTED_CLAIMS = {
    "first_chinese_pii_model": False,
    "global_sota": False,
    "production_ready": False,
    "real_world_sota": False,
    "named_open24_leadership_requires_all_replays": True,
}
EXPECTED_PRIVACY = {
    "contract_contains_paths": False,
    "release_receipt_contains_paths": False,
    "record_level_data_read_allowed": False,
    "model_weight_read_allowed": True,
    "network_allowed": False,
    "gpu_query_or_use_allowed": False,
    "publication_allowed_without_authorization": False,
}
EXPECTED_HISTORICAL_POLICY = {
    "historical_v1_is_immutable": True,
    "production_gates_inherit_this_pass": False,
    "community_pass_is_not_real_world_evidence": True,
}


class CommunityCascadeReleaseV2Error(CommunityContractError):
    """A fail-closed v2 release validation error."""


@dataclass(frozen=True, slots=True)
class ReleaseEvidencePaths:
    """Invocation-only file locations; none are emitted into a receipt."""

    quality_receipt: Path | None = None
    quality_replay_receipt: Path | None = None
    quality_native_replay: Path | None = None
    final_model_manifest: Path | None = None
    service_configuration_manifest: Path | None = None
    pii_bench_posthoc_report: Path | None = None
    calibration_bundle: Path | None = None
    service_source_manifest: Path | None = None
    predecessor_service_source_manifest: Path | None = None
    internal_preopen_unlock: Path | None = None
    candidate_manifests: Mapping[str, Path] = field(default_factory=dict)
    comparator_manifests: Mapping[str, Path] = field(default_factory=dict)
    performance_manifests: Mapping[str, Path] = field(default_factory=dict)
    replay_receipts: Mapping[str, Path] = field(default_factory=dict)
    native_replay_evidence: Mapping[str, Path] = field(default_factory=dict)
    release_artifacts: Mapping[str, Path] = field(default_factory=dict)
    model_package_root: Path | None = None
    release_wheel: Path | None = None
    wheelhouse_root: Path | None = None
    verification_receipts: Mapping[str, Path] = field(default_factory=dict)
    publication_receipts: Mapping[str, Path] = field(default_factory=dict)


def _digest(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CommunityCascadeReleaseV2Error(f"{field_name} is not a SHA-256 digest")
    return value


def _safe_id(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 192:
        raise CommunityCascadeReleaseV2Error(f"{field_name} is invalid")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:@+-")
    if value[0] not in allowed or any(character not in allowed for character in value):
        raise CommunityCascadeReleaseV2Error(f"{field_name} is invalid")
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommunityCascadeReleaseV2Error(f"{field_name} must be an object")
    return value


def _exact_keys(value: object, expected: set[str], *, field_name: str) -> Mapping[str, Any]:
    result = _mapping(value, field_name=field_name)
    if set(result) != expected:
        raise CommunityCascadeReleaseV2Error(f"{field_name} has an invalid closed shape")
    return result


def _reject_paths(value: object, *, field_name: str = "metadata") -> None:
    """Reject serialized local paths while allowing stable logical identifiers."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                key in {"path", "paths"}
                or key.endswith("_file_path")
                or key.endswith("_directory_path")
            ):
                raise CommunityCascadeReleaseV2Error(f"{field_name} contains a path field")
            _reject_paths(item, field_name=f"{field_name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_paths(item, field_name=f"{field_name}[{index}]")
    elif isinstance(value, str) and ("/" in value or "\\" in value or value.startswith("file:")):
        raise CommunityCascadeReleaseV2Error(f"{field_name} contains a serialized path")


def _read_json(path: Path, *, field_name: str) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path, field=field_name)
    return strict_json_bytes(payload, field=field_name), payload


def _verify_hash_binding(
    path: Path,
    binding: Mapping[str, Any],
    *,
    field_name: str,
) -> tuple[dict[str, Any], bytes]:
    _exact_keys(binding, {"logical_id", "file_sha256"}, field_name=f"{field_name} binding")
    _safe_id(binding["logical_id"], field_name=f"{field_name}.logical_id")
    expected = _digest(binding["file_sha256"], field_name=f"{field_name}.file_sha256")
    document, payload = _read_json(path, field_name=field_name)
    if sha256_bytes(payload) != expected:
        raise CommunityCascadeReleaseV2Error(f"{field_name} byte hash does not match")
    return document, payload


def _verify_file_hash_binding(
    path: Path,
    binding: Mapping[str, Any],
    *,
    field_name: str,
) -> bytes:
    """Verify arbitrary release bytes without pretending every artifact is JSON."""

    _exact_keys(binding, {"logical_id", "file_sha256"}, field_name=f"{field_name} binding")
    _safe_id(binding["logical_id"], field_name=f"{field_name}.logical_id")
    expected = _digest(binding["file_sha256"], field_name=f"{field_name}.file_sha256")
    payload = read_regular_file(path, field=field_name)
    if sha256_bytes(payload) != expected:
        raise CommunityCascadeReleaseV2Error(f"{field_name} byte hash does not match")
    return payload


def _verify_self_binding(
    path: Path,
    binding: Mapping[str, Any],
    *,
    field_name: str,
    self_field: str,
) -> tuple[dict[str, Any], bytes]:
    _exact_keys(
        binding,
        {"logical_id", "file_sha256", "canonical_sha256"},
        field_name=f"{field_name} binding",
    )
    document, payload = _verify_hash_binding(
        path,
        {"logical_id": binding["logical_id"], "file_sha256": binding["file_sha256"]},
        field_name=field_name,
    )
    expected = _digest(binding["canonical_sha256"], field_name=f"{field_name}.canonical_sha256")
    if (
        document.get(self_field) != expected
        or canonical_json_hash(document, remove=self_field) != expected
    ):
        raise CommunityCascadeReleaseV2Error(f"{field_name} self hash does not verify")
    return document, payload


def _file_binding_matches(path: Path, binding: Mapping[str, Any], *, field_name: str) -> bytes:
    expected_keys = {"file_sha256", "size_bytes"}
    if set(binding) != expected_keys:
        raise CommunityCascadeReleaseV2Error(f"{field_name} quality binding shape is invalid")
    payload = read_regular_file(path, field=field_name)
    if (
        sha256_bytes(payload) != _digest(binding["file_sha256"], field_name=field_name)
        or len(payload) != binding["size_bytes"]
    ):
        raise CommunityCascadeReleaseV2Error(f"{field_name} differs from quality receipt")
    return payload


def _implementation_hashes(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for logical_id, relative in IMPLEMENTATION_PATHS.items():
        payload = read_regular_file(
            repository_root / relative, field=f"implementation {logical_id}"
        )
        result[logical_id] = sha256_bytes(payload)
    return result


def _verify_quality_implementation(receipt: Mapping[str, Any], *, repository_root: Path) -> None:
    """Require the replayed quality receipt to describe the current frozen code."""

    bindings = _mapping(
        receipt["implementation_bindings"], field_name="quality implementation bindings"
    )
    direct = {
        "producer_sha256": quality.PRODUCER_PATH,
        "result_schema_sha256": quality.SCHEMA_PATH,
        "registry_schema_sha256": quality.REGISTRY_SCHEMA_PATH,
    }
    for key, relative in direct.items():
        payload = read_regular_file(repository_root / relative, field=f"quality {key}")
        if bindings[key] != sha256_bytes(payload):
            raise CommunityCascadeReleaseV2Error(f"quality {key} is stale")
    for map_name in ("metric_source_sha256", "prediction_provenance_source_sha256"):
        source_map = _mapping(bindings[map_name], field_name=f"quality {map_name}")
        for relative, expected in source_map.items():
            if not isinstance(relative, str):
                raise CommunityCascadeReleaseV2Error(f"quality {map_name} key is invalid")
            payload = read_regular_file(
                repository_root / relative, field=f"quality implementation {relative}"
            )
            if sha256_bytes(payload) != expected:
                raise CommunityCascadeReleaseV2Error(f"quality implementation {relative} is stale")


def validate_contract(
    contract_path: Path | None = None,
    schema_path: Path | None = None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    contract_file = contract_path or repository_root / CONTRACT_PATH
    schema_file = schema_path or repository_root / CONTRACT_SCHEMA_PATH
    contract = load_json_path(contract_file, field="community cascade release v2 contract")
    schema = load_json_path(schema_file, field="community cascade release v2 schema")
    validate_schema(contract, schema, field="community cascade release v2 contract")
    _reject_paths(contract, field_name="community cascade release v2 contract")
    if contract["contract_sha256"] != canonical_json_hash(contract, remove="contract_sha256"):
        raise CommunityCascadeReleaseV2Error("community release contract self hash does not verify")
    if contract["scope"] != EXPECTED_SCOPE:
        raise CommunityCascadeReleaseV2Error("community release scope boundary is invalid")
    if contract["claims"] != EXPECTED_CLAIMS:
        raise CommunityCascadeReleaseV2Error("community release claim boundary is invalid")
    if contract["privacy"] != EXPECTED_PRIVACY:
        raise CommunityCascadeReleaseV2Error("community release privacy boundary is invalid")
    if contract["historical_policy"] != EXPECTED_HISTORICAL_POLICY:
        raise CommunityCascadeReleaseV2Error("historical release policy was weakened")
    observed = _implementation_hashes(repository_root)
    if set(contract["implementation"]) != set(IMPLEMENTATION_PATHS):
        raise CommunityCascadeReleaseV2Error("release implementation inventory is incomplete")
    for logical_id, binding in contract["implementation"].items():
        _exact_keys(
            binding,
            {"logical_id", "file_sha256"},
            field_name=f"implementation {logical_id}",
        )
        if binding["logical_id"] != logical_id or binding["file_sha256"] != observed[logical_id]:
            raise CommunityCascadeReleaseV2Error(
                f"release implementation {logical_id} does not match current bytes"
            )
    return contract


def _validate_model_manifest(document: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "model_id",
        "artifact_class",
        "label_count",
        "ordered_labels",
        "taxonomy_version",
        "attention_mode",
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "model_identity_sha256",
        "artifact_sha256",
        "manifest_sha256",
    }
    _exact_keys(document, expected, field_name="final model manifest")
    if (
        document["schema_version"] != quality.MODEL_MANIFEST_SCHEMA_VERSION
        or document["artifact_class"] != "24_label_zh_hans_token_classifier"
        or document["label_count"] != 24
        or document["ordered_labels"] != list(quality.PII_CORE_LABELS)
        or document["taxonomy_version"] != "1.0.0"
        or document["attention_mode"] != "full"
    ):
        raise CommunityCascadeReleaseV2Error(
            "final model manifest is not the selected full Open-24 model"
        )
    _safe_id(document["model_id"], field_name="final model ID")
    for name in (
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "model_identity_sha256",
        "artifact_sha256",
    ):
        _digest(document[name], field_name=f"final model {name}")
    if document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256"):
        raise CommunityCascadeReleaseV2Error("final model manifest self hash does not verify")


def _validate_service_manifest(
    document: Mapping[str, Any], *, model: Mapping[str, Any], model_file_sha256: str
) -> None:
    expected = {
        "schema_version",
        "service_id",
        "profile_id",
        "canonical_track",
        "final_model_id",
        "final_model_manifest_sha256",
        "model_identity_sha256",
        "calibration_bundle_file_sha256",
        "implementation_sha256",
        "configuration_sha256",
        "manifest_sha256",
    }
    _exact_keys(document, expected, field_name="service configuration manifest")
    if (
        document["schema_version"] != quality.SERVICE_MANIFEST_SCHEMA_VERSION
        or document["canonical_track"] != "full_system"
        or document["final_model_id"] != model["model_id"]
        or document["final_model_manifest_sha256"] != model_file_sha256
        or document["model_identity_sha256"] != model["model_identity_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error("service manifest does not bind the final model")
    _safe_id(document["service_id"], field_name="service ID")
    _safe_id(document["profile_id"], field_name="service profile ID")
    for name in (
        "calibration_bundle_file_sha256",
        "implementation_sha256",
        "configuration_sha256",
    ):
        _digest(document[name], field_name=f"service {name}")
    if document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256"):
        raise CommunityCascadeReleaseV2Error("service manifest self hash does not verify")


def _runtime_package_source_hashes(repository_root: Path) -> dict[str, str]:
    """Independently recompute the complete installed-package source projection."""

    source_root = repository_root / _PACKAGE_SOURCE_ROOT
    try:
        root_metadata = source_root.lstat()
        entries = sorted(source_root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CommunityCascadeReleaseV2Error(
            "runtime package source inventory is unavailable"
        ) from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CommunityCascadeReleaseV2Error("runtime package source root is unsafe")
    result: dict[str, str] = {}
    for path in entries:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CommunityCascadeReleaseV2Error(
                "runtime package source entry is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise CommunityCascadeReleaseV2Error(
                "runtime package source inventory contains a symlink"
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CommunityCascadeReleaseV2Error(
                "runtime package source inventory contains a special file"
            )
        if path.suffix not in _PACKAGE_SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(source_root).as_posix()
        logical_id = f"pii_zh.{relative.replace('/', '.')}"
        if logical_id in result:
            raise CommunityCascadeReleaseV2Error(
                "runtime package logical source identity is ambiguous"
            )
        result[logical_id] = sha256_bytes(
            read_regular_file(path, field=f"runtime package source {logical_id}")
        )
    if not result:
        raise CommunityCascadeReleaseV2Error("runtime package source inventory is empty")
    return dict(sorted(result.items()))


def _validate_source_manifest(
    document: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    predecessor_document: Mapping[str, Any],
    predecessor_payload: bytes,
    predecessor_schema: Mapping[str, Any],
    service: Mapping[str, Any],
    full_candidate: Mapping[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    validate_schema(document, schema, field="service source identity manifest")
    validate_schema(
        predecessor_document,
        predecessor_schema,
        field="predecessor service source identity manifest",
    )
    if document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256"):
        raise CommunityCascadeReleaseV2Error("service source identity self hash does not verify")
    if predecessor_document["manifest_sha256"] != canonical_json_hash(
        predecessor_document, remove="manifest_sha256"
    ):
        raise CommunityCascadeReleaseV2Error(
            "predecessor service source identity self hash does not verify"
        )
    candidate_service = _mapping(
        full_candidate["service"], field_name="full-system service identity"
    )
    runtime_sources = _runtime_package_source_hashes(repository_root)
    predecessor_sources = _mapping(
        predecessor_document["runtime_package_source_sha256"],
        field_name="predecessor runtime package source inventory",
    )
    predecessor = _mapping(document["predecessor"], field_name="source predecessor binding")
    predecessor_keys = set(predecessor_sources)
    runtime_keys = set(runtime_sources)
    added = {
        logical_id: runtime_sources[logical_id]
        for logical_id in sorted(runtime_keys - predecessor_keys)
    }
    changed = {
        logical_id: runtime_sources[logical_id]
        for logical_id in sorted(runtime_keys & predecessor_keys)
        if runtime_sources[logical_id] != predecessor_sources[logical_id]
    }
    removed = {
        logical_id: predecessor_sources[logical_id]
        for logical_id in sorted(predecessor_keys - runtime_keys)
    }
    delta = {
        "added": added,
        "changed": changed,
        "removed": removed,
        "added_count": len(added),
        "changed_count": len(changed),
        "removed_count": len(removed),
        "summary_sha256": "",
    }
    delta["summary_sha256"] = canonical_json_hash(delta, remove="summary_sha256")
    expected_changed_ids = {
        "pii_zh.evaluation.release_eval_v2_performance.py",
        "pii_zh.data.synthetic.sota_release_eval_v1.py",
        "pii_zh.data.synthetic.sota_release_eval_v2.py",
        "pii_zh.data.synthetic.sota_v1.py",
        "pii_zh.data.synthetic.sota_v2_resplit.py",
    }
    if (
        document["service_id"] != service["service_id"]
        or document["profile_id"] != service["profile_id"]
        or document["implementation_sha256"] != service["implementation_sha256"]
        or document["implementation_sha256"] != candidate_service["implementation_sha256"]
        or document["implementation_source_sha256"]
        != candidate_service["implementation_source_sha256"]
        or document["runtime_package_source_sha256"] != runtime_sources
        or document["runtime_package_inventory_sha256"] != canonical_json_hash(runtime_sources)
        or document["runtime_package_file_count"] != len(runtime_sources)
        or document["runtime_package_inventory_complete"] is not True
        or predecessor_document["schema_version"] != "pii-zh.community-service-source-identity.v3"
        or predecessor_document["service_id"] != document["service_id"]
        or predecessor_document["profile_id"] != document["profile_id"]
        or predecessor_document["implementation_source_sha256"]
        != document["implementation_source_sha256"]
        or predecessor_document["implementation_sha256"] != document["implementation_sha256"]
        or predecessor_document["runtime_package_inventory_sha256"]
        != canonical_json_hash(predecessor_sources)
        or predecessor_document["runtime_package_file_count"] != len(predecessor_sources)
        or predecessor_document["runtime_package_inventory_complete"] is not True
        or predecessor
        != {
            "schema_version": "pii-zh.community-service-source-identity.v3",
            "file_sha256": sha256_bytes(predecessor_payload),
            "manifest_sha256": predecessor_document["manifest_sha256"],
            "runtime_package_inventory_sha256": predecessor_document[
                "runtime_package_inventory_sha256"
            ],
            "runtime_package_file_count": predecessor_document["runtime_package_file_count"],
        }
        or document["successor_delta"] != delta
        or added
        or removed
        or set(changed) != expected_changed_ids
    ):
        raise CommunityCascadeReleaseV2Error(
            "service source identity differs from candidate or current runtime closure"
        )


def _validate_calibration(
    document: Mapping[str, Any],
    payload: bytes,
    *,
    model: Mapping[str, Any],
    service: Mapping[str, Any],
    full_candidate: Mapping[str, Any],
) -> None:
    try:
        from pii_zh.calibration import CalibrationBundle

        bundle = CalibrationBundle.from_dict(document)
    except (ImportError, TypeError, ValueError) as exc:
        raise CommunityCascadeReleaseV2Error("calibration bundle is invalid") from exc
    candidate_calibration = _mapping(
        full_candidate["calibration"], field_name="full-system calibration identity"
    )
    file_hash = sha256_bytes(payload)
    if (
        file_hash != service["calibration_bundle_file_sha256"]
        or file_hash != candidate_calibration["bundle_file_sha256"]
        or bundle.model_version != model["training_manifest_sha256"]
        or bundle.calibration_version != candidate_calibration["calibration_version"]
        or candidate_calibration["model_training_manifest_sha256"]
        != model["training_manifest_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error(
            "calibration identity differs from final model/service"
        )


def _validate_internal_unlock(
    unlock: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    service: Mapping[str, Any],
    quality_receipt: Mapping[str, Any],
    full_candidate: Mapping[str, Any],
) -> None:
    """Bind the upstream pre-open decision; this cannot repair a late guard."""

    try:
        from pii_zh.evaluation.release_eval_v2_stage_gate import (
            ReleaseEvalV2StageGateError,
            validate_internal_unlock_documents,
        )

        validate_internal_unlock_documents(unlock, model, service)
    except (ImportError, ReleaseEvalV2StageGateError) as exc:
        raise CommunityCascadeReleaseV2Error("internal pre-open unlock is invalid") from exc
    dataset = _mapping(unlock["dataset"], field_name="internal unlock dataset")
    bindings = _mapping(unlock["bindings"], field_name="internal unlock bindings")
    gold = quality_receipt["input_bindings"]["gold"]
    calibration = _mapping(
        full_candidate["calibration"], field_name="full-system calibration identity"
    )
    if (
        unlock["status"] != "INTERNAL_EVALUATION_AUTHORIZED_ONE_SHOT"
        or unlock["authorization"]["quality_production_allowed"] is not True
        or unlock["privacy"]["internal_evaluation_opened_hashed_or_decoded_during_unlock"]
        is not False
        or dataset["target_split"] != "internal_evaluation"
        or dataset["gold_file_sha256"] != gold["file_sha256"]
        or dataset["gold_size_bytes"] != gold["size_bytes"]
        or dataset["gold_document_count"] != quality_receipt["protocol"]["document_count"]
        or bindings["final_model_binding_sha256"] != model["manifest_sha256"]
        or bindings["service_configuration_binding_sha256"] != service["manifest_sha256"]
        or bindings["calibration_bundle_file_sha256"] != service["calibration_bundle_file_sha256"]
        or bindings["calibration_bundle_file_sha256"] != calibration["bundle_file_sha256"]
        or bindings["calibration_diagnostics_manifest_sha256"]
        != calibration["diagnostics_manifest_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error(
            "internal unlock identity differs from the quality-evaluated candidate"
        )
    provenance = quality_receipt["input_bindings"].get("gold_provenance")
    if provenance is not None:
        expected = {
            "dataset_manifest_file_sha256": provenance["dataset_manifest_file_sha256"],
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "materialization_receipt_file_sha256": provenance[
                "materialization_receipt_file_sha256"
            ],
            "materialization_receipt_sha256": provenance["materialization_receipt_sha256"],
            "freeze_receipt_file_sha256": provenance["freeze_receipt_file_sha256"],
        }
        if any(dataset[key] != value for key, value in expected.items()):
            raise CommunityCascadeReleaseV2Error(
                "internal unlock dataset chain differs from quality provenance"
            )


def _validate_candidate_manifest(
    path: Path,
    document: Mapping[str, Any],
    *,
    track: str,
    quality_receipt: Mapping[str, Any],
    model: Mapping[str, Any],
    service: Mapping[str, Any],
) -> None:
    schema = load_json_path(
        REPOSITORY_ROOT / quality.CANDIDATE_PREDICTION_SCHEMA_PATH,
        field="candidate prediction provenance schema",
    )
    validate_schema(document, schema, field=f"{track} candidate prediction manifest")
    if document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256"):
        raise CommunityCascadeReleaseV2Error(
            f"{track} candidate manifest self hash does not verify"
        )
    try:
        from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
            ReleaseEvalV2PredictionProvenanceError,
            validate_release_eval_v2_prediction_manifest,
        )

        metadata = validate_release_eval_v2_prediction_manifest(path)
    except (ImportError, ReleaseEvalV2PredictionProvenanceError) as exc:
        raise CommunityCascadeReleaseV2Error(
            f"{track} candidate prediction metadata validation failed"
        ) from exc
    track_binding = quality_receipt["input_bindings"]["tracks"][track]
    gold = quality_receipt["input_bindings"]["gold"]
    if (
        metadata["status"] != "PASS_METADATA_ONLY"
        or metadata["full_replay_required"] is not True
        or document["track"] != track
        or document["dataset"]["target_split"] != "internal_evaluation"
        or document["dataset"]["gold_file_sha256"] != gold["file_sha256"]
        or document["prediction"]["file_sha256"]
        != track_binding["candidate_predictions"]["file_sha256"]
        or document["model"]["training_manifest_sha256"] != model["training_manifest_sha256"]
        or document["model"]["identity_sha256"] != model["model_identity_sha256"]
        or document["model"]["output_artifact_sha256"] != model["artifact_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error(f"{track} candidate identity differs from quality")
    if track == "model_raw":
        if document["service"] is not None:
            raise CommunityCascadeReleaseV2Error("model_raw candidate unexpectedly binds a service")
    else:
        candidate_service = _mapping(document["service"], field_name="full-system service")
        if (
            candidate_service["profile_id"] != service["profile_id"]
            or candidate_service["configuration_sha256"] != service["configuration_sha256"]
            or candidate_service["implementation_sha256"] != service["implementation_sha256"]
            or candidate_service["model_identity_sha256"] != model["model_identity_sha256"]
            or candidate_service["calibration_bundle_file_sha256"]
            != service["calibration_bundle_file_sha256"]
        ):
            raise CommunityCascadeReleaseV2Error(
                "full-system service identity differs from quality"
            )


def _validate_comparator_manifest(
    document: Mapping[str, Any],
    *,
    track: str,
    quality_receipt: Mapping[str, Any],
) -> None:
    schema = load_json_path(
        REPOSITORY_ROOT / quality.COMPARATOR_PREDICTION_SCHEMA_PATH,
        field="comparator prediction provenance schema",
    )
    validate_schema(document, schema, field=f"{track} comparator prediction manifest")
    if document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256"):
        raise CommunityCascadeReleaseV2Error(f"{track} comparator self hash does not verify")
    quality_track = quality_receipt["tracks"][track]
    bindings = quality_receipt["input_bindings"]["tracks"][track]
    gold = quality_receipt["input_bindings"]["gold"]
    if (
        document["track"] != track
        or document["system_id"] != quality_track["comparator_id"]
        or document["dataset"]["target_split"] != "internal_evaluation"
        or document["dataset"]["gold_file_sha256"] != gold["file_sha256"]
        or document["prediction"]["file_sha256"]
        != bindings["comparator_predictions"]["file_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error(f"{track} comparator identity differs from quality")


def _validate_performance_manifest(
    document: Mapping[str, Any],
    *,
    track: str,
    quality_receipt: Mapping[str, Any],
    model_file_sha256: str,
    service_file_sha256: str,
) -> None:
    expected = {
        "schema_version",
        "track",
        "system_id",
        "gold_sha256",
        "predictions_sha256",
        "final_model_manifest_sha256",
        "service_configuration_manifest_sha256",
        "measurement",
        "manifest_sha256",
    }
    _exact_keys(document, expected, field_name=f"{track} performance manifest")
    quality_track = quality_receipt["tracks"][track]
    quality_bindings = quality_receipt["input_bindings"]["tracks"][track]
    if (
        document["schema_version"] != quality.PERFORMANCE_SCHEMA_VERSION
        or document["track"] != track
        or document["system_id"] != quality_track["candidate_id"]
        or document["gold_sha256"] != quality_receipt["input_bindings"]["gold"]["file_sha256"]
        or document["predictions_sha256"]
        != quality_bindings["candidate_predictions"]["file_sha256"]
        or document["final_model_manifest_sha256"] != model_file_sha256
        or document["service_configuration_manifest_sha256"] != service_file_sha256
        or document["measurement"] != quality_track["performance"]
        or document["manifest_sha256"] != canonical_json_hash(document, remove="manifest_sha256")
    ):
        raise CommunityCascadeReleaseV2Error(f"{track} performance identity differs from quality")


def _identity_payload(
    *,
    quality_receipt: Mapping[str, Any],
    model: Mapping[str, Any],
    service: Mapping[str, Any],
    source: Mapping[str, Any],
    internal_unlock: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "model_id": str(model["model_id"]),
        "service_id": str(service["service_id"]),
        "training_manifest_sha256": str(model["training_manifest_sha256"]),
        "model_identity_sha256": str(model["model_identity_sha256"]),
        "calibration_bundle_file_sha256": str(service["calibration_bundle_file_sha256"]),
        "service_configuration_sha256": str(service["configuration_sha256"]),
        "service_implementation_sha256": str(service["implementation_sha256"]),
        "source_manifest_sha256": str(source["manifest_sha256"]),
        "quality_receipt_sha256": str(quality_receipt["receipt_sha256"]),
        "internal_unlock_sha256": str(internal_unlock["receipt_sha256"]),
    }


def _wrapper_replay_implementation(
    *, receipt_type: str, target_document: Mapping[str, Any], repository_root: Path
) -> tuple[str, str]:
    """Bind the wrapper producer and the actual implementation it replayed."""

    wrapper_file_sha256 = sha256_bytes(
        read_regular_file(
            repository_root / EVIDENCE_PRODUCER_PATH,
            field="community replay evidence producer",
        )
    )
    if receipt_type == "quality_strict_replay":
        underlying = {"quality_implementation_bindings": target_document["implementation_bindings"]}
    elif receipt_type == "candidate_prediction_full_replay":
        generation = _mapping(
            target_document.get("generation"), field_name="candidate generation identity"
        )
        underlying = {
            "candidate_provenance_implementation": target_document["provenance_implementation"],
            "candidate_generator_implementation_sha256": _digest(
                generation.get("generator_implementation_sha256"),
                field_name="candidate generator implementation",
            ),
        }
    elif receipt_type == "comparator_generation_replay":
        underlying = {"comparator_system_bindings": target_document["system_bindings"]}
    else:
        raise CommunityCascadeReleaseV2Error("unsupported replay wrapper implementation")
    identity = {
        "receipt_type": receipt_type,
        "wrapper_file_sha256": wrapper_file_sha256,
        "underlying_implementation": underlying,
    }
    return wrapper_file_sha256, canonical_json_hash(identity)


def _performance_replay_implementation(track: str) -> tuple[str, str]:
    """Derive the performance implementation independently of contract values."""

    if track not in TRACKS:
        raise CommunityCascadeReleaseV2Error("unsupported performance replay track")
    try:
        from pii_zh.evaluation.release_eval_v2_performance import _implementation_identity

        implementation_file, implementation_identity = _implementation_identity(track)
    except (ImportError, TypeError, ValueError, OSError) as exc:
        raise CommunityCascadeReleaseV2Error(
            "performance replay implementation identity cannot be derived"
        ) from exc
    return (
        _digest(implementation_file, field_name="performance implementation file"),
        _digest(implementation_identity, field_name="performance implementation identity"),
    )


def _upstream_replay_binding(
    document: Mapping[str, Any], payload: bytes, *, field_name: str
) -> dict[str, str]:
    canonical = _digest(document.get("receipt_sha256"), field_name=f"{field_name}.receipt_sha256")
    if canonical_json_hash(document, remove="receipt_sha256") != canonical:
        raise CommunityCascadeReleaseV2Error(f"{field_name} self hash does not verify")
    schema_version = document.get("schema_version")
    _safe_id(schema_version, field_name=f"{field_name}.schema_version")
    return {
        "schema_version": str(schema_version),
        "file_sha256": sha256_bytes(payload),
        "canonical_sha256": canonical,
    }


def _verify_quality_native_replay(
    path: Path,
    *,
    quality_receipt: Mapping[str, Any],
    quality_payload: bytes,
    repository_root: Path,
) -> dict[str, str]:
    document, payload = _read_json(path, field_name="quality native replay evidence")
    schema = load_json_path(
        repository_root / QUALITY_REPLAY_EVIDENCE_SCHEMA_PATH,
        field="quality native replay evidence schema",
    )
    validate_schema(document, schema, field="quality native replay evidence")
    _reject_paths(document, field_name="quality native replay evidence")
    implementation = _mapping(
        quality_receipt["implementation_bindings"],
        field_name="quality implementation bindings",
    )
    expected = {
        "schema_version": "pii-zh.community-quality-strict-replay-evidence.v1",
        "receipt_type": "quality_strict_semantic_replay",
        "status": "PASS",
        "target": {
            "artifact_kind": "quality_result_receipt",
            "file_sha256": sha256_bytes(quality_payload),
            "canonical_sha256": quality_receipt["receipt_sha256"],
        },
        "replay": {
            "producer_file_sha256": implementation["producer_sha256"],
            "implementation_identity_sha256": canonical_json_hash(implementation),
            "rebuilt_receipt_sha256": quality_receipt["receipt_sha256"],
            "exact_semantic_equality": True,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "aggregate_only": True,
            "network_used": False,
        },
        "receipt_sha256": document.get("receipt_sha256"),
    }
    expected["receipt_sha256"] = canonical_json_hash(expected, remove="receipt_sha256")
    if document != expected:
        raise CommunityCascadeReleaseV2Error(
            "quality native replay evidence differs from the admitted quality result"
        )
    return _upstream_replay_binding(document, payload, field_name="quality native replay evidence")


def _candidate_execution_identity(
    document: Mapping[str, Any], *, manifest_file_sha256: str
) -> dict[str, Any]:
    prediction = _mapping(document.get("prediction"), field_name="candidate prediction")
    generation = _mapping(document.get("generation"), field_name="candidate generation")
    return {
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": document["manifest_sha256"],
        "predictions_file_sha256": prediction["file_sha256"],
        "prediction_document_count": prediction["document_count"],
        "generation_receipt_file_sha256": generation["receipt_file_sha256"],
        "generation_receipt_sha256": generation["receipt_sha256"],
    }


def _verify_candidate_native_replay(
    path: Path,
    *,
    track: str,
    candidate_document: Mapping[str, Any],
    candidate_file_sha256: str,
    repository_root: Path,
) -> dict[str, str]:
    document, payload = _read_json(path, field_name=f"{track} candidate native replay evidence")
    schema = load_json_path(
        repository_root / CANDIDATE_REPLAY_EVIDENCE_SCHEMA_PATH,
        field="candidate native replay evidence schema",
    )
    validate_schema(document, schema, field=f"{track} candidate native replay evidence")
    _reject_paths(document, field_name=f"{track} candidate native replay evidence")
    target = _candidate_execution_identity(
        candidate_document, manifest_file_sha256=candidate_file_sha256
    )
    generation = _mapping(candidate_document.get("generation"), field_name="candidate generation")
    provenance = _mapping(
        candidate_document.get("provenance_implementation"),
        field_name="candidate provenance implementation",
    )
    runtime_sources = _runtime_package_source_hashes(repository_root)
    expected = {
        "schema_version": "pii-zh.community-candidate-execution-replay-evidence.v1",
        "receipt_type": "candidate_prediction_full_execution_replay",
        "status": "PASS",
        "track": track,
        "target": target,
        "replayed": {
            **target,
            "predictions_byte_identical": True,
            "generation_receipt_exact_equal": True,
            "provenance_exact_equal": True,
        },
        "implementation": {
            "replay_producer_file_sha256": sha256_bytes(
                read_regular_file(
                    repository_root / EVIDENCE_PRODUCER_PATH,
                    field="candidate replay evidence producer",
                )
            ),
            "generator_implementation_sha256": generation["generator_implementation_sha256"],
            "provenance_implementation_sha256": provenance["implementation_sha256"],
            "runtime_package_inventory_sha256": canonical_json_hash(runtime_sources),
            "runtime_package_file_count": len(runtime_sources),
            "runtime_package_inventory_complete": True,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "aggregate_only": True,
            "network_used": False,
        },
        "receipt_sha256": document.get("receipt_sha256"),
    }
    expected["receipt_sha256"] = canonical_json_hash(expected, remove="receipt_sha256")
    if document != expected:
        raise CommunityCascadeReleaseV2Error(
            f"{track} candidate native replay is disconnected from the admitted candidate"
        )
    return _upstream_replay_binding(
        document, payload, field_name=f"{track} candidate native replay evidence"
    )


def _verify_comparator_native_replay(
    path: Path,
    *,
    track: str,
    comparator_document: Mapping[str, Any],
    comparator_file_sha256: str,
) -> dict[str, str]:
    try:
        from pii_zh.evaluation.open24_comparator_provenance import (
            validate_execution_replay_receipt,
        )

        prediction = _mapping(
            comparator_document.get("prediction"), field_name="comparator prediction"
        )
        consumed = validate_execution_replay_receipt(
            path,
            track=track,
            predictions_sha256=str(prediction["file_sha256"]),
            prediction_manifest_file_sha256=comparator_file_sha256,
            prediction_manifest_sha256=str(comparator_document["manifest_sha256"]),
            allow_fixture=False,
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise CommunityCascadeReleaseV2Error(
            f"{track} comparator native execution replay is invalid"
        ) from exc
    document, payload = _read_json(path, field_name=f"{track} comparator native execution replay")
    provenance = _mapping(document.get("provenance"), field_name="comparator replay provenance")
    replay_prediction = _mapping(
        document.get("prediction"), field_name="comparator replay prediction"
    )
    implementation = _mapping(
        document.get("implementation"), field_name="comparator replay implementation"
    )
    system_bindings = _mapping(
        comparator_document.get("system_bindings"), field_name="comparator system bindings"
    )
    generator = _mapping(
        system_bindings.get("generator_implementation"),
        field_name="comparator generator binding",
    )
    if (
        consumed.get("status") != "PASS_FULL_EXECUTION_REPLAY_RECEIPT"
        or consumed.get("fixture") is not False
        or replay_prediction.get("document_count") != prediction.get("document_count")
        or provenance.get("stage_and_support_strict_replay_passed") is not True
        or implementation.get("observed_file_sha256") != generator.get("file_sha256")
        or implementation.get("replayed_file_sha256") != generator.get("file_sha256")
    ):
        raise CommunityCascadeReleaseV2Error(
            f"{track} comparator native replay differs from the admitted comparator"
        )
    return _upstream_replay_binding(
        document, payload, field_name=f"{track} comparator native execution replay"
    )


def _expected_replay_receipt_core(
    *,
    receipt_type: str,
    track: str | None,
    quality_receipt: Mapping[str, Any],
    quality_file_sha256: str,
    identity: Mapping[str, str],
    target_document: Mapping[str, Any],
    target_file_sha256: str,
    target_kind: str,
    implementation_file_sha256: str,
    implementation_identity_sha256: str,
    upstream_replay_evidence: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    target_canonical = str(
        target_document.get("receipt_sha256") or target_document.get("manifest_sha256")
    )
    _digest(target_canonical, field_name=f"{receipt_type} target canonical hash")
    target = {
        "artifact_kind": target_kind,
        "file_sha256": target_file_sha256,
        "canonical_sha256": target_canonical,
    }
    input_material: dict[str, Any] = {
        "receipt_type": receipt_type,
        "track": track,
        "quality_file_sha256": quality_file_sha256,
        "quality_receipt_sha256": quality_receipt["receipt_sha256"],
        "identity": dict(identity),
        "quality_input_bindings": quality_receipt["input_bindings"],
        "target": target,
    }
    if upstream_replay_evidence is not None:
        input_material["upstream_replay_evidence"] = dict(upstream_replay_evidence)
    if track is not None:
        input_material["quality_track"] = quality_receipt["tracks"][track]
    output_material = {
        "target": target,
        "reported_status": quality_receipt["reported_status"],
        "track_status": None
        if track is None
        else quality_receipt["tracks"][track]["declared_status"],
    }
    core = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "receipt_type": receipt_type,
        "track": track,
        "status": "EXACT_REPLAY_PASS",
        "quality": {
            "file_sha256": quality_file_sha256,
            "receipt_sha256": quality_receipt["receipt_sha256"],
        },
        "target": target,
        "identity": dict(identity),
        "replay": {
            "implementation_file_sha256": implementation_file_sha256,
            "implementation_identity_sha256": implementation_identity_sha256,
            "input_set_sha256": canonical_json_hash(input_material),
            "output_set_sha256": canonical_json_hash(output_material),
            "mode": "full_exact_replay",
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "aggregate_only": True,
            "network_used": False,
        },
    }
    core["receipt_id"] = canonical_json_hash(core)
    return core


def _verify_replay_receipt(
    path: Path,
    contract_binding: Mapping[str, Any],
    *,
    expected_core: Mapping[str, Any],
    replay_schema: Mapping[str, Any],
    field_name: str,
) -> tuple[dict[str, Any], bytes]:
    _exact_keys(
        contract_binding,
        {
            "logical_id",
            "file_sha256",
            "canonical_sha256",
            "implementation_file_sha256",
            "implementation_identity_sha256",
        },
        field_name=f"{field_name} contract binding",
    )
    document, payload = _read_json(path, field_name=field_name)
    validate_schema(document, replay_schema, field=field_name)
    _reject_paths(document, field_name=field_name)
    if (
        sha256_bytes(payload) != contract_binding["file_sha256"]
        or document["receipt_sha256"] != contract_binding["canonical_sha256"]
        or document["receipt_sha256"] != canonical_json_hash(document, remove="receipt_sha256")
        or document["replay"]["implementation_file_sha256"]
        != contract_binding["implementation_file_sha256"]
        or document["replay"]["implementation_identity_sha256"]
        != contract_binding["implementation_identity_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error(f"{field_name} content binding does not verify")
    expected = dict(expected_core)
    expected["receipt_sha256"] = canonical_json_hash(expected_core)
    if document != expected:
        raise CommunityCascadeReleaseV2Error(
            f"{field_name} does not match independently derived replay context"
        )
    return document, payload


def _verify_typed_release_artifact(
    path: Path,
    binding: Mapping[str, Any],
    *,
    artifact_id: str,
    repository_root: Path,
) -> tuple[bytes, dict[str, Any] | None]:
    payload = _verify_file_hash_binding(path, binding, field_name=f"release artifact {artifact_id}")
    if artifact_id not in release_artifacts.ARTIFACT_IDS:
        return payload, None
    if binding["logical_id"] != artifact_id:
        raise CommunityCascadeReleaseV2Error(
            f"typed release artifact {artifact_id} has the wrong logical ID"
        )
    document = strict_json_bytes(payload, field=f"release artifact {artifact_id}")
    try:
        release_artifacts.validate_artifact_document(
            document,
            artifact_id=artifact_id,
            repository_root=repository_root,
        )
    except CommunityContractError as exc:
        raise CommunityCascadeReleaseV2Error(
            f"typed release artifact {artifact_id} is invalid"
        ) from exc
    return payload, document


def _file_subject(payload: bytes, *, canonical_sha256: str) -> dict[str, int | str]:
    return {
        "file_sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "canonical_sha256": canonical_sha256,
    }


def _validate_typed_artifact_graph(
    documents: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, bytes],
    *,
    core_subjects: Mapping[str, Mapping[str, int | str]],
) -> None:
    if set(documents) != set(release_artifacts.ARTIFACT_IDS):
        raise CommunityCascadeReleaseV2Error("typed release artifact inventory is incomplete")
    if len({sha256_bytes(payloads[item]) for item in documents}) != len(documents):
        raise CommunityCascadeReleaseV2Error("typed release artifacts reuse the same bytes")

    model_card = documents["model_card"]
    expected_model_card_inputs = {
        key: {
            "file_sha256": core_subjects[key]["file_sha256"],
            "size_bytes": core_subjects[key]["size_bytes"],
        }
        for key in (
            "quality_receipt",
            "final_model_manifest",
            "service_configuration_manifest",
        )
    }
    expected_model_card_inputs["training_manifest"] = documents["model_package_manifest"]["result"][
        "files"
    ]["pkg:training_manifest.json"]
    for logical_id, result_key in (
        ("template_asset", "template_asset_file_sha256"),
        ("pii_bench_posthoc_report", "pii_bench_report_file_sha256"),
    ):
        binding = model_card["inputs"].get(logical_id)
        if (
            not isinstance(binding, Mapping)
            or binding.get("file_sha256") != model_card["result"][result_key]
        ):
            raise CommunityCascadeReleaseV2Error(
                f"model card {logical_id} file binding does not verify"
            )
        expected_model_card_inputs[logical_id] = binding
    if model_card["inputs"] != expected_model_card_inputs:
        raise CommunityCascadeReleaseV2Error("model card inputs differ from the admitted release")

    def typed_input(artifact_id: str) -> dict[str, int | str]:
        return {
            "file_sha256": sha256_bytes(payloads[artifact_id]),
            "size_bytes": len(payloads[artifact_id]),
        }

    if documents["model_package_manifest"]["inputs"] != {
        "model_card": typed_input("model_card"),
        "final_model_manifest": {
            "file_sha256": core_subjects["final_model_manifest"]["file_sha256"],
            "size_bytes": core_subjects["final_model_manifest"]["size_bytes"],
        },
    }:
        raise CommunityCascadeReleaseV2Error(
            "model package does not bind the model card and evaluated final model"
        )
    if documents["container_manifest"]["inputs"] != {
        "model_package_manifest": typed_input("model_package_manifest"),
        "wheel_manifest": typed_input("wheel_manifest"),
        "wheelhouse_manifest": typed_input("wheelhouse_manifest"),
    }:
        raise CommunityCascadeReleaseV2Error("container does not bind the model package and wheel")
    container_result = documents["container_manifest"]["result"]
    if (
        container_result["bound_wheel_artifact_sha256"]
        != documents["wheel_manifest"]["artifact_sha256"]
        or container_result["bound_wheel_file_sha256"]
        != documents["wheel_manifest"]["inputs"]["wheel"]["file_sha256"]
        or container_result["external_model_package_artifact_sha256"]
        != documents["model_package_manifest"]["artifact_sha256"]
        or container_result["bound_wheelhouse_artifact_sha256"]
        != documents["wheelhouse_manifest"]["artifact_sha256"]
        or container_result["model_delivery"] != "external_read_only_mount"
    ):
        raise CommunityCascadeReleaseV2Error("container canonical artifact bindings differ")
    expected_license_inputs = {
        "sbom": typed_input("sbom"),
        "license_document": documents["model_package_manifest"]["result"]["files"]["pkg:LICENSE"],
        "notice_document": documents["model_package_manifest"]["result"]["files"]["pkg:NOTICE"],
        "third_party_notices": documents["model_package_manifest"]["result"]["files"][
            "pkg:THIRD_PARTY_NOTICES.md"
        ],
    }
    if documents["license_report"]["inputs"] != expected_license_inputs:
        raise CommunityCascadeReleaseV2Error(
            "license report does not bind the admitted SBOM and model-package legal documents"
        )
    dependency = documents["dependency_scan"]
    if (
        dependency["inputs"] != {"sbom": typed_input("sbom")}
        or dependency["result"]["sbom_artifact_sha256"] != documents["sbom"]["artifact_sha256"]
        or dependency["result"]["sbom_content_sha256"] != documents["sbom"]["result"]["bom_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error("dependency scan does not bind the admitted SBOM")
    wheelhouse = documents["wheelhouse_manifest"]
    if (
        wheelhouse["inputs"] != {"sbom": typed_input("sbom")}
        or wheelhouse["result"]["sbom_artifact_sha256"] != documents["sbom"]["artifact_sha256"]
    ):
        raise CommunityCascadeReleaseV2Error("wheelhouse does not bind the admitted SBOM")
    public_scan_inputs = documents["public_artifact_scan"]["inputs"]
    expected_public_ids = set(release_artifacts.ARTIFACT_IDS) - {"public_artifact_scan"}
    for artifact_id in expected_public_ids:
        if public_scan_inputs.get(f"artifact:{artifact_id}") != typed_input(artifact_id):
            raise CommunityCascadeReleaseV2Error(
                "public artifact scan does not bind the complete typed artifact set"
            )
    allowed_public_keys = {f"artifact:{item}" for item in expected_public_ids}
    if set(public_scan_inputs) != allowed_public_keys:
        raise CommunityCascadeReleaseV2Error("public artifact scan input inventory is not exact")


def _replay_typed_artifact_sources(
    documents: Mapping[str, Mapping[str, Any]],
    evidence: ReleaseEvidencePaths,
    *,
    repository_root: Path,
) -> None:
    """Verify typed artifacts against fixed physical or frozen source evidence.

    The dependency scan is deliberately validated as frozen typed evidence here:
    its producer performs the networked OSV scan once, while this final gate
    remains deterministic and never repeats a time-varying network query.
    """

    required_paths = {
        "quality_receipt": evidence.quality_receipt,
        "final_model_manifest": evidence.final_model_manifest,
        "service_configuration_manifest": evidence.service_configuration_manifest,
        "pii_bench_posthoc_report": evidence.pii_bench_posthoc_report,
        "model_package_root": evidence.model_package_root,
        "release_wheel": evidence.release_wheel,
        "wheelhouse_root": evidence.wheelhouse_root,
    }
    missing = sorted(name for name, path in required_paths.items() if path is None)
    if missing:
        raise CommunityCascadeReleaseV2Error("typed artifact source replay inputs are incomplete")
    assert evidence.quality_receipt is not None
    assert evidence.final_model_manifest is not None
    assert evidence.service_configuration_manifest is not None
    assert evidence.pii_bench_posthoc_report is not None
    assert evidence.model_package_root is not None
    assert evidence.release_wheel is not None
    assert evidence.wheelhouse_root is not None
    artifact_paths = evidence.release_artifacts
    if set(artifact_paths).intersection(release_artifacts.ARTIFACT_IDS) != set(
        release_artifacts.ARTIFACT_IDS
    ):
        raise CommunityCascadeReleaseV2Error("typed artifact paths are incomplete")

    rebuilt: dict[str, Mapping[str, Any]] = {}
    rebuilt["model_card"] = release_artifacts.build_model_card_artifact(
        quality_receipt_path=evidence.quality_receipt,
        model_manifest_path=evidence.final_model_manifest,
        service_manifest_path=evidence.service_configuration_manifest,
        training_manifest_path=evidence.model_package_root / "training_manifest.json",
        pii_bench_report_path=evidence.pii_bench_posthoc_report,
        template_asset_path=repository_root / release_artifacts.COMMUNITY_TEMPLATE_ASSET_PATH,
        repository_root=repository_root,
    )
    rebuilt["model_package_manifest"] = release_artifacts.build_model_package_manifest(
        model_package_root=evidence.model_package_root,
        model_card_artifact_path=artifact_paths["model_card"],
        final_model_manifest_path=evidence.final_model_manifest,
        repository_root=repository_root,
    )
    rebuilt["technical_documentation_manifest"] = (
        release_artifacts.build_technical_documentation_manifest(repository_root=repository_root)
    )
    rebuilt["wheel_manifest"] = release_artifacts.build_wheel_manifest(
        wheel_path=evidence.release_wheel,
        repository_root=repository_root,
    )
    rebuilt["wheelhouse_manifest"] = release_artifacts.build_wheelhouse_manifest(
        wheelhouse_root=evidence.wheelhouse_root,
        sbom_artifact_path=artifact_paths["sbom"],
        repository_root=repository_root,
    )
    rebuilt["container_manifest"] = release_artifacts.build_container_manifest(
        image_ref=str(documents["container_manifest"]["result"]["image_id"]),
        wheel_path=evidence.release_wheel,
        wheel_manifest_path=artifact_paths["wheel_manifest"],
        wheelhouse_manifest_path=artifact_paths["wheelhouse_manifest"],
        model_package_manifest_path=artifact_paths["model_package_manifest"],
        repository_root=repository_root,
    )
    rebuilt["sbom"] = release_artifacts.build_sbom_artifact(
        lockfile_path=repository_root / "uv.lock",
        pyproject_path=repository_root / "pyproject.toml",
        repository_root=repository_root,
    )
    rebuilt["license_report"] = release_artifacts.build_license_report(
        sbom_artifact_path=artifact_paths["sbom"],
        model_package_root=evidence.model_package_root,
        repository_root=repository_root,
    )
    rebuilt["dependency_scan"] = documents["dependency_scan"]
    rebuilt["public_artifact_scan"] = release_artifacts.build_public_artifact_scan(
        model_package_root=evidence.model_package_root,
        wheel_path=evidence.release_wheel,
        wheelhouse_root=evidence.wheelhouse_root,
        bound_artifacts={
            artifact_id: artifact_paths[artifact_id]
            for artifact_id in release_artifacts.ARTIFACT_IDS
            if artifact_id != "public_artifact_scan"
        },
        repository_root=repository_root,
    )
    if rebuilt != documents:
        differing = sorted(
            artifact_id
            for artifact_id in release_artifacts.ARTIFACT_IDS
            if rebuilt[artifact_id] != documents[artifact_id]
        )
        raise CommunityCascadeReleaseV2Error(
            "typed artifact source replay differs: " + ",".join(differing)
        )


def _verify_typed_verification(
    path: Path,
    binding: Mapping[str, Any],
    *,
    check_id: str,
    repository_root: Path,
) -> tuple[dict[str, Any], bytes]:
    document, payload = _verify_hash_binding(path, binding, field_name=f"verification {check_id}")
    if binding["logical_id"] != check_id:
        raise CommunityCascadeReleaseV2Error(f"verification {check_id} has the wrong logical ID")
    try:
        release_verifications.validate_verification_receipt(
            document,
            check_id=check_id,
            repository_root=repository_root,
        )
    except CommunityContractError as exc:
        raise CommunityCascadeReleaseV2Error(f"verification {check_id} is invalid") from exc
    return document, payload


def _validate_verification_subject_graph(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    artifact_subjects: Mapping[str, Mapping[str, int | str]],
    core_subjects: Mapping[str, Mapping[str, int | str]],
) -> None:
    expected = {
        "unit_tests": {"service_source_manifest": core_subjects["service_source_manifest"]},
        "clean_wheel_smoke": {
            "wheel_manifest": artifact_subjects["wheel_manifest"],
            "wheelhouse_manifest": artifact_subjects["wheelhouse_manifest"],
        },
        "container_smoke": {
            "container_manifest": artifact_subjects["container_manifest"],
            "wheel_manifest": artifact_subjects["wheel_manifest"],
            "wheelhouse_manifest": artifact_subjects["wheelhouse_manifest"],
            "model_package_manifest": artifact_subjects["model_package_manifest"],
            "service_configuration_manifest": core_subjects["service_configuration_manifest"],
            "service_source_manifest": core_subjects["service_source_manifest"],
            "calibration_bundle": core_subjects["calibration_bundle"],
        },
        "offline_model_smoke": {
            "model_package_manifest": artifact_subjects["model_package_manifest"],
            "wheel_manifest": artifact_subjects["wheel_manifest"],
            "wheelhouse_manifest": artifact_subjects["wheelhouse_manifest"],
        },
        "offline_service_smoke": {
            "model_package_manifest": artifact_subjects["model_package_manifest"],
            "wheel_manifest": artifact_subjects["wheel_manifest"],
            "wheelhouse_manifest": artifact_subjects["wheelhouse_manifest"],
            "service_configuration_manifest": core_subjects["service_configuration_manifest"],
            "service_source_manifest": core_subjects["service_source_manifest"],
            "calibration_bundle": core_subjects["calibration_bundle"],
        },
    }
    if set(receipts) != set(expected):
        raise CommunityCascadeReleaseV2Error("verification receipt inventory is incomplete")
    for check_id, subjects in expected.items():
        if receipts[check_id]["subjects"] != subjects:
            raise CommunityCascadeReleaseV2Error(
                f"verification {check_id} subjects differ from the admitted release"
            )


def _verify_publication(
    contract: Mapping[str, Any],
    evidence: ReleaseEvidencePaths,
) -> str:
    publication = contract["publication"]
    if publication["publication_requires_same_turn_user_authorization"] is not True:
        raise CommunityCascadeReleaseV2Error("publication authorization boundary is disabled")
    for target_name in ("github", "hugging_face"):
        target = publication[target_name]
        supplied = evidence.publication_receipts.get(target_name)
        if target != {
            "target_id": None,
            "authorized": False,
            "published": False,
            "publication_receipt": None,
        }:
            raise CommunityCascadeReleaseV2Error(
                f"{target_name} must remain an unpublished local target"
            )
        if supplied is not None:
            raise CommunityCascadeReleaseV2Error(
                f"{target_name} publication evidence belongs to a future successor contract"
            )
    return "READY_FOR_USER_AUTHORIZATION"


def build_report(
    *,
    contract_path: Path | None = None,
    schema_path: Path | None = None,
    evidence: ReleaseEvidencePaths | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    contract = validate_contract(contract_path, schema_path, repository_root=repository_root)
    supplied = evidence or ReleaseEvidencePaths()
    blockers: list[str] = []
    checks: dict[str, dict[str, str]] = {}
    core_subjects: dict[str, Mapping[str, int | str]] = {}

    required_top = (
        (
            "quality_result_receipt",
            contract["quality_gate"]["result_receipt"],
            supplied.quality_receipt,
        ),
        (
            "quality_strict_replay",
            contract["quality_gate"]["strict_replay_receipt"],
            supplied.quality_replay_receipt,
        ),
        (
            "final_model_manifest",
            None
            if contract["candidate_identity"] is None
            else contract["candidate_identity"]["final_model_manifest"],
            supplied.final_model_manifest,
        ),
        (
            "service_configuration_manifest",
            None
            if contract["candidate_identity"] is None
            else contract["candidate_identity"]["service_configuration_manifest"],
            supplied.service_configuration_manifest,
        ),
        (
            "calibration_bundle",
            None
            if contract["candidate_identity"] is None
            else contract["candidate_identity"]["calibration_bundle"],
            supplied.calibration_bundle,
        ),
        (
            "service_source_manifest",
            None
            if contract["candidate_identity"] is None
            else contract["candidate_identity"]["service_source_manifest"],
            supplied.service_source_manifest,
        ),
        (
            "predecessor_service_source_manifest",
            None
            if contract["candidate_identity"] is None
            else contract["candidate_identity"]["service_source_manifest"],
            supplied.predecessor_service_source_manifest,
        ),
        (
            "internal_preopen_unlock",
            None
            if contract["candidate_identity"] is None
            else contract["candidate_identity"]["internal_preopen_unlock"],
            supplied.internal_preopen_unlock,
        ),
    )
    for name, binding, path in required_top:
        if binding is None:
            blockers.append(name)
            checks[name] = {"status": "BLOCKED", "reason": "contract_binding_missing"}
        elif path is None:
            raise CommunityCascadeReleaseV2Error(
                f"{name} path was not supplied for a bound contract"
            )

    if (
        contract["quality_gate"]["strict_replay_receipt"] is not None
        and supplied.quality_native_replay is None
    ):
        raise CommunityCascadeReleaseV2Error(
            "bound quality strict replay requires native replay evidence"
        )

    expected_native_keys: set[str] = set()
    for track in TRACKS:
        for kind in REPLAY_KINDS:
            key = f"{track}:{kind}"
            binding = contract["replay_evidence"][track][kind]
            path = supplied.replay_receipts.get(key)
            if binding is None:
                blockers.append(f"replay:{key}")
                checks[f"replay:{key}"] = {
                    "status": "BLOCKED",
                    "reason": "contract_binding_missing",
                }
            elif path is None:
                raise CommunityCascadeReleaseV2Error(f"bound replay receipt {key} was not supplied")
            if (
                binding is not None
                and kind in {"candidate_prediction_full_replay", "comparator_generation_replay"}
                and key not in supplied.native_replay_evidence
            ):
                raise CommunityCascadeReleaseV2Error(
                    f"bound replay receipt {key} requires native replay evidence"
                )
            if binding is not None and kind in {
                "candidate_prediction_full_replay",
                "comparator_generation_replay",
            }:
                expected_native_keys.add(key)
    if set(supplied.native_replay_evidence) != expected_native_keys:
        raise CommunityCascadeReleaseV2Error(
            "native replay evidence inventory differs from the bound replay inventory"
        )

    identity_report: dict[str, str] | None = None
    replay_report: dict[str, Any] = {track: {} for track in TRACKS}
    quality_report: dict[str, Any] | None = None
    if not any(name in blockers for name, _binding, _path in required_top):
        assert supplied.quality_receipt is not None
        assert supplied.quality_replay_receipt is not None
        assert supplied.final_model_manifest is not None
        assert supplied.service_configuration_manifest is not None
        assert supplied.calibration_bundle is not None
        assert supplied.service_source_manifest is not None
        assert supplied.predecessor_service_source_manifest is not None
        assert supplied.internal_preopen_unlock is not None
        candidate_identity = _mapping(
            contract["candidate_identity"], field_name="candidate identity"
        )
        quality_receipt, quality_payload = _verify_self_binding(
            supplied.quality_receipt,
            contract["quality_gate"]["result_receipt"],
            field_name="quality result receipt",
            self_field="receipt_sha256",
        )
        quality.validate_receipt_schema(quality_receipt, repository_root=repository_root)
        _verify_quality_implementation(quality_receipt, repository_root=repository_root)
        if (
            quality_receipt["reported_status"] != "PASS"
            or quality_receipt["tracks"]["full_system"]["declared_status"] != "PASS"
            or quality_receipt["claims"]["claim_activation_allowed"] is not False
        ):
            raise CommunityCascadeReleaseV2Error(
                "quality v2 receipt is not a strict full-system PASS"
            )

        model, model_payload = _verify_self_binding(
            supplied.final_model_manifest,
            candidate_identity["final_model_manifest"],
            field_name="final model manifest",
            self_field="manifest_sha256",
        )
        service, service_payload = _verify_self_binding(
            supplied.service_configuration_manifest,
            candidate_identity["service_configuration_manifest"],
            field_name="service configuration manifest",
            self_field="manifest_sha256",
        )
        source, source_payload = _verify_self_binding(
            supplied.service_source_manifest,
            candidate_identity["service_source_manifest"],
            field_name="service source identity manifest",
            self_field="manifest_sha256",
        )
        predecessor_source, predecessor_source_payload = _read_json(
            supplied.predecessor_service_source_manifest,
            field_name="predecessor service source identity manifest",
        )
        try:
            predecessor_mode = stat.S_IMODE(
                supplied.predecessor_service_source_manifest.stat().st_mode
            )
        except OSError as exc:
            raise CommunityCascadeReleaseV2Error(
                "predecessor service source identity is unavailable"
            ) from exc
        if predecessor_mode != 0o444:
            raise CommunityCascadeReleaseV2Error(
                "predecessor service source identity must remain immutable mode 0444"
            )
        calibration, calibration_payload = _verify_hash_binding(
            supplied.calibration_bundle,
            candidate_identity["calibration_bundle"],
            field_name="calibration bundle",
        )
        internal_unlock, internal_unlock_payload = _verify_self_binding(
            supplied.internal_preopen_unlock,
            candidate_identity["internal_preopen_unlock"],
            field_name="internal pre-open unlock",
            self_field="receipt_sha256",
        )
        _validate_model_manifest(model)
        _validate_service_manifest(
            service, model=model, model_file_sha256=sha256_bytes(model_payload)
        )
        quality_bindings = quality_receipt["input_bindings"]
        _file_binding_matches(
            supplied.final_model_manifest,
            quality_bindings["final_model_manifest"],
            field_name="quality final model manifest",
        )
        _file_binding_matches(
            supplied.service_configuration_manifest,
            quality_bindings["service_configuration_manifest"],
            field_name="quality service configuration manifest",
        )
        if (
            quality_receipt["candidate"]["model_id"] != model["model_id"]
            or quality_receipt["candidate"]["service_id"] != service["service_id"]
        ):
            raise CommunityCascadeReleaseV2Error(
                "quality candidate IDs differ from release identity"
            )

        candidate_docs: dict[str, dict[str, Any]] = {}
        comparator_docs: dict[str, dict[str, Any]] = {}
        performance_docs: dict[str, dict[str, Any]] = {}
        for track in TRACKS:
            for collection_name, paths in (
                ("candidate", supplied.candidate_manifests),
                ("comparator", supplied.comparator_manifests),
                ("performance", supplied.performance_manifests),
            ):
                if track not in paths:
                    raise CommunityCascadeReleaseV2Error(
                        f"{track} {collection_name} manifest was not supplied"
                    )
            candidate_doc, _ = _read_json(
                supplied.candidate_manifests[track], field_name=f"{track} candidate manifest"
            )
            comparator_doc, _ = _read_json(
                supplied.comparator_manifests[track], field_name=f"{track} comparator manifest"
            )
            performance_doc, _ = _read_json(
                supplied.performance_manifests[track], field_name=f"{track} performance manifest"
            )
            track_bindings = quality_bindings["tracks"][track]
            _file_binding_matches(
                supplied.candidate_manifests[track],
                track_bindings["candidate_prediction_manifest"],
                field_name=f"quality {track} candidate manifest",
            )
            _file_binding_matches(
                supplied.comparator_manifests[track],
                track_bindings["comparator_prediction_manifest"],
                field_name=f"quality {track} comparator manifest",
            )
            _file_binding_matches(
                supplied.performance_manifests[track],
                track_bindings["performance_manifest"],
                field_name=f"quality {track} performance manifest",
            )
            if (
                candidate_doc["manifest_sha256"]
                != track_bindings["candidate_prediction_manifest_sha256"]
                or comparator_doc["manifest_sha256"]
                != track_bindings["comparator_prediction_manifest_sha256"]
            ):
                raise CommunityCascadeReleaseV2Error(
                    f"{track} manifest self hash differs from quality"
                )
            _validate_candidate_manifest(
                supplied.candidate_manifests[track],
                candidate_doc,
                track=track,
                quality_receipt=quality_receipt,
                model=model,
                service=service,
            )
            _validate_comparator_manifest(
                comparator_doc, track=track, quality_receipt=quality_receipt
            )
            _validate_performance_manifest(
                performance_doc,
                track=track,
                quality_receipt=quality_receipt,
                model_file_sha256=sha256_bytes(model_payload),
                service_file_sha256=sha256_bytes(service_payload),
            )
            candidate_docs[track] = candidate_doc
            comparator_docs[track] = comparator_doc
            performance_docs[track] = performance_doc

        source_schema = load_json_path(
            repository_root / SOURCE_SCHEMA_PATH, field="service source identity schema"
        )
        predecessor_source_schema = load_json_path(
            repository_root / PREDECESSOR_SOURCE_SCHEMA_PATH,
            field="predecessor service source identity schema",
        )
        _validate_source_manifest(
            source,
            schema=source_schema,
            predecessor_document=predecessor_source,
            predecessor_payload=predecessor_source_payload,
            predecessor_schema=predecessor_source_schema,
            service=service,
            full_candidate=candidate_docs["full_system"],
            repository_root=repository_root,
        )
        _validate_calibration(
            calibration,
            calibration_payload,
            model=model,
            service=service,
            full_candidate=candidate_docs["full_system"],
        )
        _validate_internal_unlock(
            internal_unlock,
            model=model,
            service=service,
            quality_receipt=quality_receipt,
            full_candidate=candidate_docs["full_system"],
        )
        identity = _identity_payload(
            quality_receipt=quality_receipt,
            model=model,
            service=service,
            source=source,
            internal_unlock=internal_unlock,
        )
        core_subjects = {
            "quality_receipt": _file_subject(
                quality_payload, canonical_sha256=quality_receipt["receipt_sha256"]
            ),
            "final_model_manifest": _file_subject(
                model_payload, canonical_sha256=model["manifest_sha256"]
            ),
            "service_configuration_manifest": _file_subject(
                service_payload, canonical_sha256=service["manifest_sha256"]
            ),
            "service_source_manifest": _file_subject(
                source_payload, canonical_sha256=source["manifest_sha256"]
            ),
            "calibration_bundle": _file_subject(
                calibration_payload, canonical_sha256=canonical_json_hash(calibration)
            ),
            "internal_preopen_unlock": _file_subject(
                internal_unlock_payload,
                canonical_sha256=internal_unlock["receipt_sha256"],
            ),
        }
        declared_identity = {
            key: candidate_identity[key]
            for key in (
                "model_id",
                "service_id",
                "training_manifest_sha256",
                "model_identity_sha256",
                "calibration_bundle_file_sha256",
                "service_configuration_sha256",
                "service_implementation_sha256",
                "source_manifest_sha256",
                "internal_unlock_sha256",
            )
        }
        if declared_identity != {key: identity[key] for key in declared_identity}:
            raise CommunityCascadeReleaseV2Error("contract candidate identity differs from quality")
        identity_report = identity

        replay_schema = load_json_path(
            repository_root / REPLAY_SCHEMA_PATH, field="community replay receipt schema"
        )
        assert supplied.quality_native_replay is not None
        quality_native_binding = _verify_quality_native_replay(
            supplied.quality_native_replay,
            quality_receipt=quality_receipt,
            quality_payload=quality_payload,
            repository_root=repository_root,
        )
        quality_implementation_file, quality_implementation_identity = (
            _wrapper_replay_implementation(
                receipt_type="quality_strict_replay",
                target_document=quality_receipt,
                repository_root=repository_root,
            )
        )
        quality_core = _expected_replay_receipt_core(
            receipt_type="quality_strict_replay",
            track=None,
            quality_receipt=quality_receipt,
            quality_file_sha256=sha256_bytes(quality_payload),
            identity=identity,
            target_document=quality_receipt,
            target_file_sha256=sha256_bytes(quality_payload),
            target_kind="quality_result_receipt",
            implementation_file_sha256=quality_implementation_file,
            implementation_identity_sha256=quality_implementation_identity,
            upstream_replay_evidence=quality_native_binding,
        )
        _verify_replay_receipt(
            supplied.quality_replay_receipt,
            contract["quality_gate"]["strict_replay_receipt"],
            expected_core=quality_core,
            replay_schema=replay_schema,
            field_name="quality strict replay receipt",
        )
        checks["quality_result_receipt"] = {"status": "PASS"}
        checks["native_replay:quality_strict_replay"] = {"status": "PASS"}
        checks["quality_strict_replay"] = {"status": "PASS"}

        for track in TRACKS:
            (
                performance_implementation_file,
                performance_implementation_identity,
            ) = _performance_replay_implementation(track)
            target_sets = {
                "candidate_prediction_full_replay": (
                    candidate_docs[track],
                    sha256_bytes(
                        read_regular_file(
                            supplied.candidate_manifests[track], field=f"{track} candidate manifest"
                        )
                    ),
                    "candidate_prediction_manifest",
                    candidate_docs[track]["provenance_implementation"]["module_file_sha256"],
                    candidate_docs[track]["provenance_implementation"]["implementation_sha256"],
                ),
                "comparator_generation_replay": (
                    comparator_docs[track],
                    sha256_bytes(
                        read_regular_file(
                            supplied.comparator_manifests[track],
                            field=f"{track} comparator manifest",
                        )
                    ),
                    "comparator_prediction_manifest",
                    comparator_docs[track]["system_bindings"]["generator_implementation"][
                        "file_sha256"
                    ],
                    canonical_json_hash(comparator_docs[track]["system_bindings"]),
                ),
                "performance_harness_replay": (
                    performance_docs[track],
                    sha256_bytes(
                        read_regular_file(
                            supplied.performance_manifests[track],
                            field=f"{track} performance manifest",
                        )
                    ),
                    "performance_manifest",
                    performance_implementation_file,
                    performance_implementation_identity,
                ),
            }
            for kind, (
                target_document,
                target_file_hash,
                target_kind,
                implementation_file_hash,
                implementation_identity_hash,
            ) in target_sets.items():
                replay_binding = contract["replay_evidence"][track][kind]
                if replay_binding is None:
                    continue
                upstream_replay_evidence: Mapping[str, str] | None = None
                key = f"{track}:{kind}"
                if kind == "candidate_prediction_full_replay":
                    implementation_file_hash, implementation_identity_hash = (
                        _wrapper_replay_implementation(
                            receipt_type=kind,
                            target_document=target_document,
                            repository_root=repository_root,
                        )
                    )
                    upstream_replay_evidence = _verify_candidate_native_replay(
                        supplied.native_replay_evidence[key],
                        track=track,
                        candidate_document=target_document,
                        candidate_file_sha256=target_file_hash,
                        repository_root=repository_root,
                    )
                elif kind == "comparator_generation_replay":
                    implementation_file_hash, implementation_identity_hash = (
                        _wrapper_replay_implementation(
                            receipt_type=kind,
                            target_document=target_document,
                            repository_root=repository_root,
                        )
                    )
                    upstream_replay_evidence = _verify_comparator_native_replay(
                        supplied.native_replay_evidence[key],
                        track=track,
                        comparator_document=target_document,
                        comparator_file_sha256=target_file_hash,
                    )
                core = _expected_replay_receipt_core(
                    receipt_type=kind,
                    track=track,
                    quality_receipt=quality_receipt,
                    quality_file_sha256=sha256_bytes(quality_payload),
                    identity=identity,
                    target_document=target_document,
                    target_file_sha256=target_file_hash,
                    target_kind=target_kind,
                    implementation_file_sha256=implementation_file_hash,
                    implementation_identity_sha256=implementation_identity_hash,
                    upstream_replay_evidence=upstream_replay_evidence,
                )
                document, _ = _verify_replay_receipt(
                    supplied.replay_receipts[key],
                    replay_binding,
                    expected_core=core,
                    replay_schema=replay_schema,
                    field_name=f"{track} {kind} receipt",
                )
                replay_report[track][kind] = {
                    "status": document["status"],
                    "receipt_sha256": document["receipt_sha256"],
                }
                if upstream_replay_evidence is not None:
                    checks[f"native_replay:{key}"] = {"status": "PASS"}
                checks[f"replay:{key}"] = {"status": "PASS"}

        if not any(item.startswith("replay:") for item in blockers):
            quality_report = {
                "reported_status": quality_receipt["reported_status"],
                "receipt_sha256": quality_receipt["receipt_sha256"],
                "full_system_status": quality_receipt["tracks"]["full_system"]["declared_status"],
                "all_required_replays": "PASS",
            }

    artifact_report: dict[str, str | None] = {}
    typed_artifact_documents: dict[str, Mapping[str, Any]] = {}
    typed_artifact_payloads: dict[str, bytes] = {}
    typed_artifact_sources_replayed = False
    for artifact_id in ARTIFACT_IDS:
        binding = contract["release_artifacts"][artifact_id]
        path = supplied.release_artifacts.get(artifact_id)
        if binding is None:
            blockers.append(f"artifact:{artifact_id}")
            checks[f"artifact:{artifact_id}"] = {
                "status": "BLOCKED",
                "reason": "contract_binding_missing",
            }
            artifact_report[artifact_id] = None
        else:
            if path is None:
                raise CommunityCascadeReleaseV2Error(
                    f"bound artifact {artifact_id} was not supplied"
                )
            payload, typed_document = _verify_typed_release_artifact(
                path,
                binding,
                artifact_id=artifact_id,
                repository_root=repository_root,
            )
            artifact_report[artifact_id] = sha256_bytes(payload)
            if typed_document is not None:
                typed_artifact_documents[artifact_id] = typed_document
                typed_artifact_payloads[artifact_id] = payload
            checks[f"artifact:{artifact_id}"] = {"status": "PASS"}

    if contract["release_artifacts"]["benchmark_report"] is not None:
        if contract["release_artifacts"]["benchmark_report"] != {
            "logical_id": contract["quality_gate"]["result_receipt"]["logical_id"],
            "file_sha256": contract["quality_gate"]["result_receipt"]["file_sha256"],
        }:
            raise CommunityCascadeReleaseV2Error(
                "benchmark artifact is not the admitted quality receipt"
            )
    if contract["candidate_identity"] is not None:
        for artifact_id, identity_id in (
            ("final_model_manifest", "final_model_manifest"),
            ("service_configuration_manifest", "service_configuration_manifest"),
            ("service_source_manifest", "service_source_manifest"),
        ):
            if contract["release_artifacts"][artifact_id] != {
                "logical_id": contract["candidate_identity"][identity_id]["logical_id"],
                "file_sha256": contract["candidate_identity"][identity_id]["file_sha256"],
            }:
                raise CommunityCascadeReleaseV2Error(
                    f"release artifact {artifact_id} differs from candidate identity"
                )

    if typed_artifact_documents:
        if not core_subjects:
            raise CommunityCascadeReleaseV2Error(
                "typed release artifacts cannot be admitted without candidate identity"
            )
        _validate_typed_artifact_graph(
            typed_artifact_documents,
            typed_artifact_payloads,
            core_subjects=core_subjects,
        )
        _replay_typed_artifact_sources(
            typed_artifact_documents,
            supplied,
            repository_root=repository_root,
        )
        typed_artifact_sources_replayed = True

    verification_report: dict[str, str | None] = {}
    verification_documents: dict[str, Mapping[str, Any]] = {}
    for check_id in VERIFICATION_IDS:
        binding = contract["verification_receipts"][check_id]
        path = supplied.verification_receipts.get(check_id)
        if binding is None:
            blockers.append(f"verification:{check_id}")
            checks[f"verification:{check_id}"] = {
                "status": "BLOCKED",
                "reason": "contract_binding_missing",
            }
            verification_report[check_id] = None
        else:
            if path is None:
                raise CommunityCascadeReleaseV2Error(
                    f"bound verification {check_id} was not supplied"
                )
            document, payload = _verify_typed_verification(
                path,
                binding,
                check_id=check_id,
                repository_root=repository_root,
            )
            verification_documents[check_id] = document
            verification_report[check_id] = sha256_bytes(payload)
            checks[f"verification:{check_id}"] = {"status": "PASS"}

    if verification_documents:
        if not core_subjects or set(typed_artifact_documents) != set(
            release_artifacts.ARTIFACT_IDS
        ):
            raise CommunityCascadeReleaseV2Error(
                "verification receipts require the complete typed artifact graph"
            )
        artifact_subjects = {
            artifact_id: _file_subject(
                typed_artifact_payloads[artifact_id],
                canonical_sha256=typed_artifact_documents[artifact_id]["artifact_sha256"],
            )
            for artifact_id in typed_artifact_documents
        }
        _validate_verification_subject_graph(
            verification_documents,
            artifact_subjects=artifact_subjects,
            core_subjects=core_subjects,
        )

    publication_state = _verify_publication(contract, supplied)
    blockers = sorted(set(blockers))
    if blockers:
        derived_status = "BLOCKED"
        expected_contract_status = "blocked_pending_replays_and_artifacts"
    else:
        derived_status = publication_state
        expected_contract_status = "candidate_complete_pending_user_authorization"
    if contract["status"] != expected_contract_status:
        raise CommunityCascadeReleaseV2Error(
            "contract status does not match verified release state"
        )

    contract_payload = read_regular_file(
        contract_path or repository_root / CONTRACT_PATH, field="community release contract"
    )
    report: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "report_type": "community_cascade_release_v2",
        "contract": {
            "file_sha256": sha256_bytes(contract_payload),
            "contract_sha256": contract["contract_sha256"],
        },
        "status": derived_status,
        "local_candidate_complete": not blockers,
        "publication_state": publication_state,
        "blocker_ids": blockers,
        "checks": checks,
        "quality": quality_report,
        "identity": identity_report,
        "replay_evidence": replay_report,
        "release_artifacts": artifact_report,
        "verification_receipts": verification_report,
        "claims": contract["claims"],
        "limitations": [
            "public_and_synthetic_only",
            "public_test_exposed",
            "not_production_ready",
            "no_real_world_sota_claim",
        ],
        "privacy": {
            "contains_paths": False,
            "contains_raw_records": False,
            "model_weights_read": typed_artifact_sources_replayed,
            "gpu_queried_or_used": False,
            "network_used": False,
        },
        "receipt_sha256": "",
    }
    report["receipt_sha256"] = canonical_json_hash(report, remove="receipt_sha256")
    receipt_schema = load_json_path(
        repository_root / RECEIPT_SCHEMA_PATH, field="community release receipt schema"
    )
    validate_schema(report, receipt_schema, field="community release receipt")
    _reject_paths(report, field_name="community release receipt")
    return report


def publish_report(path: Path, report: Mapping[str, Any]) -> None:
    """Publish one immutable report atomically and without clobbering."""

    if path.exists() or path.is_symlink():
        raise CommunityCascadeReleaseV2Error("refusing to overwrite release receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
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
            raise CommunityCascadeReleaseV2Error("release receipt mode is not 0444")
    finally:
        temporary.unlink(missing_ok=True)


def _parse_named(values: list[str], *, field_name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CommunityCascadeReleaseV2Error(f"{field_name} must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in result:
            raise CommunityCascadeReleaseV2Error(f"{field_name} contains an invalid entry")
        result[name] = Path(raw_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=REPOSITORY_ROOT / CONTRACT_PATH)
    parser.add_argument("--schema", type=Path, default=REPOSITORY_ROOT / CONTRACT_SCHEMA_PATH)
    parser.add_argument("--quality-receipt", type=Path)
    parser.add_argument("--quality-replay-receipt", type=Path)
    parser.add_argument("--quality-native-replay", type=Path)
    parser.add_argument("--final-model-manifest", type=Path)
    parser.add_argument("--service-configuration-manifest", type=Path)
    parser.add_argument("--pii-bench-posthoc-report", type=Path)
    parser.add_argument("--calibration-bundle", type=Path)
    parser.add_argument("--service-source-manifest", type=Path)
    parser.add_argument("--predecessor-service-source-manifest", type=Path)
    parser.add_argument("--internal-preopen-unlock", type=Path)
    parser.add_argument("--candidate-manifest", action="append", default=[])
    parser.add_argument("--comparator-manifest", action="append", default=[])
    parser.add_argument("--performance-manifest", action="append", default=[])
    parser.add_argument("--replay-receipt", action="append", default=[])
    parser.add_argument("--native-replay-evidence", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--model-package-root", type=Path)
    parser.add_argument("--release-wheel", type=Path)
    parser.add_argument("--wheelhouse-root", type=Path)
    parser.add_argument("--verification", action="append", default=[])
    parser.add_argument("--publication-receipt", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evidence = ReleaseEvidencePaths(
            quality_receipt=args.quality_receipt,
            quality_replay_receipt=args.quality_replay_receipt,
            quality_native_replay=args.quality_native_replay,
            final_model_manifest=args.final_model_manifest,
            service_configuration_manifest=args.service_configuration_manifest,
            pii_bench_posthoc_report=args.pii_bench_posthoc_report,
            calibration_bundle=args.calibration_bundle,
            service_source_manifest=args.service_source_manifest,
            predecessor_service_source_manifest=args.predecessor_service_source_manifest,
            internal_preopen_unlock=args.internal_preopen_unlock,
            candidate_manifests=_parse_named(
                args.candidate_manifest, field_name="candidate manifest"
            ),
            comparator_manifests=_parse_named(
                args.comparator_manifest, field_name="comparator manifest"
            ),
            performance_manifests=_parse_named(
                args.performance_manifest, field_name="performance manifest"
            ),
            replay_receipts=_parse_named(args.replay_receipt, field_name="replay receipt"),
            native_replay_evidence=_parse_named(
                args.native_replay_evidence, field_name="native replay evidence"
            ),
            release_artifacts=_parse_named(args.artifact, field_name="artifact"),
            model_package_root=args.model_package_root,
            release_wheel=args.release_wheel,
            wheelhouse_root=args.wheelhouse_root,
            verification_receipts=_parse_named(args.verification, field_name="verification"),
            publication_receipts=_parse_named(
                args.publication_receipt, field_name="publication receipt"
            ),
        )
        report = build_report(
            contract_path=args.contract,
            schema_path=args.schema,
            evidence=evidence,
        )
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            publish_report(args.output, report)
        return 0
    except (OSError, TypeError, ValueError, CommunityContractError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
