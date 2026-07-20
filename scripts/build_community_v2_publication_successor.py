#!/usr/bin/env python3
"""Build a fail-closed, offline publication successor for community v2.

The builder treats the pre-authorization model package as immutable input.  It
verifies its complete checksum closure, copies or reflinks only regular files
into a new no-clobber directory, replaces publication metadata, and emits a new
manifest and checksum closure.  It never loads model tensors, opens record-level
data, uses a GPU, contacts a network service, or mutates the source package.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_VERSION = "0.2.0rc1"
RELEASE_TAG = "v0.2.0rc1"
MANIFEST_SCHEMA_PATH = Path("configs/release/community_v2_publication_successor.schema.json")
LICENSE_APPROVAL_SCHEMA_PATH = MANIFEST_SCHEMA_PATH
SECURITY_CHANNEL_SCHEMA_PATH = MANIFEST_SCHEMA_PATH
HF_GITATTRIBUTES_TEMPLATE_PATH = Path("release/community-v2-rc1/huggingface.gitattributes")
FINAL_LOCAL_RECEIPT_SCHEMA_PATH = Path(
    "configs/release/community_cascade_release_v2.receipt.schema.json"
)

FINAL_LOCAL_RECEIPT_NAME = "community_v2_final_local_receipt.json"
LICENSE_APPROVAL_RECEIPT_NAME = "human_license_approval_receipt.json"
SECURITY_CHANNEL_RECEIPT_NAME = "tested_private_security_channel_receipt.json"
MANIFEST_NAME = "publication_manifest.json"
CHECKSUMS_NAME = "checksums.txt"
PREAUTHORIZATION_NAME = "community_v2_preauthorization.json"
HF_GITATTRIBUTES_NAME = ".gitattributes"

_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_CARD_TARGET_PLACEHOLDER = re.compile(
    r"(?:hf_namespace/|<(?:hf_)?namespace[^>]*>)", re.IGNORECASE
)
_GITHUB_URL = re.compile(r"https://github\.com/[^\s<>()\[\]{}\"']+")
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_READ_BLOCK_BYTES = 4 * 1024 * 1024
_FICLONE = 0x40049409

_REQUIRED_SOURCE_FILES = frozenset(
    {
        "README.md",
        "SECURITY.md",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "model.safetensors",
        PREAUTHORIZATION_NAME,
        "config.json",
        "training_manifest.json",
        "configuration_qwen3_bi.py",
        "modeling_qwen3_bi.py",
    }
)
_RESERVED_SUCCESSOR_FILES = frozenset(
    {
        FINAL_LOCAL_RECEIPT_NAME,
        LICENSE_APPROVAL_RECEIPT_NAME,
        SECURITY_CHANNEL_RECEIPT_NAME,
        MANIFEST_NAME,
        HF_GITATTRIBUTES_NAME,
    }
)
_OLD_PUBLICATION_MARKERS = (
    "unpublished",
    "pending",
    "withheld",
    "no tested private reporting route exists",
    "no-tested-route",
)
_LICENSE_HISTORY_MARKER = "COMPLETE_HUMAN_APPROVAL_PENDING"
_LICENSE_HISTORY_EXPLANATION = (
    "historical mechanical evidence",
    "human clearance comes",
    "separately validated approval receipt",
)
_FALSE_ONLY_CONTRACT_FILES = frozenset(
    {
        "config.json",
        "training_manifest.json",
        "configuration_qwen3_bi.py",
        "modeling_qwen3_bi.py",
    }
)
_SUPERSEDED_SOURCE_FILES = frozenset(
    {"README.md", "SECURITY.md", "NOTICE", "THIRD_PARTY_NOTICES.md"}
)
_GENERIC_STALE_MARKERS = tuple(marker.encode("ascii") for marker in _OLD_PUBLICATION_MARKERS)
_TOKENIZER_PUBLICATION_MARKERS = (
    b"unpublished_local",
    b"unpublished local release",
    b"pending_community",
    b"pending community contract",
    b"no tested private",
    b"no-tested-route",
    b"public upload is therefore withheld",
)
_FALSE_LINEAGE_MARKERS = (b"release_eligible", b"not_benchmark_evaluated")


class PublicationSuccessorError(RuntimeError):
    """Raised when successor construction cannot safely continue."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class RegularPayload:
    payload: bytes
    file_sha256: str


@dataclass(frozen=True)
class SourceEvidence:
    root: Path
    files: Mapping[str, Path]
    inventory: Mapping[str, Mapping[str, Any]]
    inventory_sha256: str
    checksums_file_sha256: str


@dataclass(frozen=True)
class BuildPlan:
    source: SourceEvidence
    output: Path
    model_card: RegularPayload
    security: RegularPayload
    notice: RegularPayload
    third_party_notices: RegularPayload
    huggingface_gitattributes: RegularPayload
    final_local_receipt: RegularPayload
    final_local_document: Mapping[str, Any]
    license_approval_receipt: RegularPayload
    license_approval_document: Mapping[str, Any]
    security_channel_receipt: RegularPayload
    security_channel_document: Mapping[str, Any]
    candidate_lineage_contract: Mapping[str, Any]
    stale_marker_scan: Mapping[str, Any]
    git_source_commit: str
    github_repository: str
    hugging_face_repository: str


def _canonical_json_bytes(value: Mapping[str, Any], *, remove: str | None = None) -> bytes:
    document = dict(value)
    if remove is not None:
        document.pop(remove, None)
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PublicationSuccessorError(
            "INVALID_CANONICAL_JSON", "document is not canonical-JSON serializable"
        ) from exc


def canonical_json_hash(value: Mapping[str, Any], *, remove: str | None = None) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationSuccessorError(
                    "DUPLICATE_JSON_KEY", f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise PublicationSuccessorError(
            "NONFINITE_JSON_NUMBER", f"{field} contains a non-finite JSON number"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise PublicationSuccessorError(
            "INVALID_JSON_ENCODING", f"{field} is not UTF-8 JSON"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PublicationSuccessorError("INVALID_JSON", f"{field} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PublicationSuccessorError("INVALID_JSON_SHAPE", f"{field} must be a JSON object")
    return value


def _open_read_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _read_regular_payload(
    path: Path, *, field: str, maximum_bytes: int = _MAX_METADATA_BYTES
) -> RegularPayload:
    try:
        descriptor = os.open(path, _open_read_flags())
    except OSError as exc:
        raise PublicationSuccessorError(
            "UNSAFE_INPUT_FILE", f"{field} is missing, non-regular, or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise PublicationSuccessorError(
                "UNSAFE_INPUT_FILE", f"{field} must be a bounded regular file"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, maximum_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise PublicationSuccessorError(
                    "INPUT_TOO_LARGE", f"{field} exceeds the metadata size limit"
                )
            digest.update(block)
            chunks.append(block)
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
        if before_identity != after_identity or total != before.st_size:
            raise PublicationSuccessorError(
                "INPUT_CHANGED_DURING_READ", f"{field} changed while it was read"
            )
        return RegularPayload(b"".join(chunks), digest.hexdigest())
    finally:
        os.close(descriptor)


def _hash_regular_file(path: Path, *, field: str) -> tuple[str, int]:
    try:
        descriptor = os.open(path, _open_read_flags())
    except OSError as exc:
        raise PublicationSuccessorError(
            "UNSAFE_SOURCE_FILE", f"{field} is unavailable or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationSuccessorError("UNSAFE_SOURCE_FILE", f"{field} is not a regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or total != before.st_size:
            raise PublicationSuccessorError(
                "SOURCE_CHANGED_DURING_READ", f"{field} changed while it was hashed"
            )
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _safe_relative_name(value: str, *, field: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or len(value) > 512
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(not part.isprintable() for part in pure.parts)
    ):
        raise PublicationSuccessorError(
            "UNSAFE_PACKAGE_PATH", f"{field} contains an unsafe package path"
        )
    return value


def _scan_tree(root: Path, *, field: str) -> dict[str, Path]:
    try:
        root_metadata = os.lstat(root)
    except OSError as exc:
        raise PublicationSuccessorError(
            "SOURCE_PACKAGE_MISSING", f"{field} is unavailable"
        ) from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise PublicationSuccessorError("UNSAFE_PACKAGE_ROOT", f"{field} must be a real directory")

    result: dict[str, Path] = {}

    def visit(directory: Path, parts: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise PublicationSuccessorError(
                "SOURCE_TREE_UNAVAILABLE", f"{field} cannot be enumerated"
            ) from exc
        for entry in entries:
            relative = _safe_relative_name(
                PurePosixPath(*parts, entry.name).as_posix(), field=field
            )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PublicationSuccessorError(
                    "SOURCE_TREE_UNAVAILABLE", f"{field} entry cannot be inspected"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise PublicationSuccessorError(
                    "SOURCE_SYMLINK_REJECTED", f"{field} contains symlink {relative!r}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                visit(Path(entry.path), (*parts, entry.name))
            elif stat.S_ISREG(metadata.st_mode):
                if relative in result:  # defensive: directory traversal must be injective
                    raise PublicationSuccessorError(
                        "DUPLICATE_PACKAGE_PATH", f"{field} contains duplicate path {relative!r}"
                    )
                result[relative] = Path(entry.path)
            else:
                raise PublicationSuccessorError(
                    "SOURCE_SPECIAL_FILE_REJECTED",
                    f"{field} contains non-regular file {relative!r}",
                )

    visit(root, ())
    return dict(sorted(result.items()))


def _parse_checksums(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PublicationSuccessorError(
            "INVALID_SOURCE_CHECKSUMS", "source checksums.txt is not UTF-8"
        ) from exc
    if not lines:
        raise PublicationSuccessorError("INVALID_SOURCE_CHECKSUMS", "source checksums.txt is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise PublicationSuccessorError(
                "INVALID_SOURCE_CHECKSUMS",
                f"invalid source checksums.txt line {line_number}",
            )
        digest, relative = match.groups()
        relative = _safe_relative_name(relative, field="source checksums.txt")
        if relative == CHECKSUMS_NAME:
            raise PublicationSuccessorError(
                "CHECKSUM_SELF_REFERENCE", "source checksums.txt must not list itself"
            )
        if relative in entries:
            raise PublicationSuccessorError(
                "DUPLICATE_CHECKSUM_PATH",
                f"source checksums.txt repeats path {relative!r}",
            )
        entries[relative] = digest
    return dict(sorted(entries.items()))


def verify_source_package(source_root: Path) -> SourceEvidence:
    files = _scan_tree(source_root, field="source package")
    checksums_path = files.get(CHECKSUMS_NAME)
    if checksums_path is None:
        raise PublicationSuccessorError(
            "SOURCE_CHECKSUMS_MISSING", "source package has no checksums.txt"
        )
    checksums_payload = _read_regular_payload(
        checksums_path, field="source checksums.txt", maximum_bytes=16 * 1024 * 1024
    )
    declared = _parse_checksums(checksums_payload.payload)
    actual_names = set(files) - {CHECKSUMS_NAME}
    declared_names = set(declared)
    missing = sorted(declared_names - actual_names)
    unlisted = sorted(actual_names - declared_names)
    if missing:
        raise PublicationSuccessorError(
            "SOURCE_CHECKSUM_FILE_MISSING",
            f"source package is missing checksum-listed file {missing[0]!r}",
        )
    if unlisted:
        raise PublicationSuccessorError(
            "SOURCE_UNLISTED_FILE",
            f"source package contains unlisted file {unlisted[0]!r}",
        )
    absent_required = sorted(_REQUIRED_SOURCE_FILES - actual_names)
    if absent_required:
        raise PublicationSuccessorError(
            "SOURCE_REQUIRED_FILE_MISSING",
            f"source package is missing required file {absent_required[0]!r}",
        )
    reserved = sorted(_RESERVED_SUCCESSOR_FILES & actual_names)
    if reserved:
        raise PublicationSuccessorError(
            "SOURCE_RESERVED_FILE_REJECTED",
            f"source package contains successor-only file {reserved[0]!r}",
        )
    jsonl_files = sorted(name for name in actual_names if name.lower().endswith(".jsonl"))
    if jsonl_files:
        raise PublicationSuccessorError(
            "SOURCE_JSONL_REJECTED",
            f"source package contains record-level JSONL file {jsonl_files[0]!r}",
        )

    inventory: dict[str, dict[str, Any]] = {}
    for relative, expected in declared.items():
        observed, size = _hash_regular_file(
            files[relative], field=f"source package file {relative}"
        )
        if observed != expected:
            raise PublicationSuccessorError(
                "SOURCE_CHECKSUM_MISMATCH",
                f"source checksum mismatch for {relative!r}",
            )
        inventory[relative] = {"file_sha256": observed, "size_bytes": size}
    second_names = set(_scan_tree(source_root, field="source package"))
    if second_names != set(files):
        raise PublicationSuccessorError(
            "SOURCE_TREE_CHANGED", "source package changed during verification"
        )
    return SourceEvidence(
        root=source_root,
        files=files,
        inventory=dict(sorted(inventory.items())),
        inventory_sha256=canonical_json_hash(inventory),
        checksums_file_sha256=checksums_payload.file_sha256,
    )


def _marker_hits(payload: bytes, markers: Sequence[bytes]) -> list[str]:
    lowered = payload.lower()
    return sorted(marker.decode("ascii") for marker in markers if marker in lowered)


def _release_eligibility_fields(value: object, *, prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "release_eligible" in str(key).lower():
                result[path] = item
            result.update(_release_eligibility_fields(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_release_eligibility_fields(item, prefix=f"{prefix}[{index}]"))
    return result


def _validate_candidate_lineage_contract(source: SourceEvidence) -> dict[str, Any]:
    config_payload = _read_regular_payload(source.files["config.json"], field="source config.json")
    training_payload = _read_regular_payload(
        source.files["training_manifest.json"], field="source training_manifest.json"
    )
    config = _strict_json(config_payload.payload, field="source config.json")
    training = _strict_json(training_payload.payload, field="source training_manifest.json")
    config_lineage = config.get("pii_lineage")
    initialization = training.get("initialization")
    config_release_fields = _release_eligibility_fields(config)
    training_release_fields = _release_eligibility_fields(training)
    if (
        config.get("pii_release_eligible") is not False
        or not isinstance(config_lineage, Mapping)
        or config_lineage.get("release_eligible") is not False
        or config.get("pii_training_status") != "completed_candidate_not_benchmark_evaluated"
        or not config_release_fields
        or any(value is not False for value in config_release_fields.values())
    ):
        raise PublicationSuccessorError(
            "CONFIG_FALSE_ONLY_CONTRACT_INVALID",
            "config.json does not preserve the exact false-only candidate-lineage contract",
        )
    if (
        training.get("release_eligible") is not False
        or not isinstance(initialization, Mapping)
        or initialization.get("release_eligible") is not False
        or not training_release_fields
        or any(value is not False for value in training_release_fields.values())
    ):
        raise PublicationSuccessorError(
            "TRAINING_FALSE_ONLY_CONTRACT_INVALID",
            "training_manifest.json does not preserve the false-only candidate lineage",
        )

    configuration_payload = _read_regular_payload(
        source.files["configuration_qwen3_bi.py"],
        field="source configuration_qwen3_bi.py",
    )
    modeling_payload = _read_regular_payload(
        source.files["modeling_qwen3_bi.py"], field="source modeling_qwen3_bi.py"
    )
    try:
        configuration = configuration_payload.payload.decode("utf-8")
        modeling = modeling_payload.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicationSuccessorError(
            "REMOTE_CODE_FALSE_ONLY_CONTRACT_INVALID",
            "false-only remote code must be UTF-8",
        ) from exc
    configuration_requirements = (
        'community_release_eligible = kwargs.get("pii_release_eligible")',
        "if community_release_eligible is not False:",
        '"pii_release_eligible=false contract."',
        "self.pii_release_eligible = community_release_eligible",
    )
    modeling_requirement = 'getattr(config, "pii_release_eligible", None) is False'
    if (
        any(item not in configuration for item in configuration_requirements)
        or "self.pii_release_eligible = True" in configuration
        or modeling_requirement not in modeling
        or 'getattr(config, "pii_release_eligible", None) is True' in modeling
    ):
        raise PublicationSuccessorError(
            "REMOTE_CODE_FALSE_ONLY_CONTRACT_INVALID",
            "remote code does not enforce the exact false-only runtime contract",
        )
    for relative, payload in (
        ("config.json", config_payload.payload),
        ("training_manifest.json", training_payload.payload),
        ("configuration_qwen3_bi.py", configuration_payload.payload),
        ("modeling_qwen3_bi.py", modeling_payload.payload),
    ):
        hits = _marker_hits(payload, _GENERIC_STALE_MARKERS)
        if hits:
            raise PublicationSuccessorError(
                "STALE_PUBLICATION_MARKER",
                f"false-only contract file {relative!r} also contains stale marker {hits[0]!r}",
            )

    return {
        "interpretation": "immutable_candidate_lineage_not_current_remote_publication_state",
        "byte_preserved": True,
        "authorization_does_not_mutate_runtime_contract": True,
        "files": {
            "config.json": {
                "file_sha256": source.inventory["config.json"]["file_sha256"],
                "contract": {
                    "pii_release_eligible": False,
                    "pii_lineage.release_eligible": False,
                    "pii_training_status": "completed_candidate_not_benchmark_evaluated",
                },
            },
            "training_manifest.json": {
                "file_sha256": source.inventory["training_manifest.json"]["file_sha256"],
                "contract": {
                    "release_eligible": False,
                    "initialization.release_eligible": False,
                },
            },
            "configuration_qwen3_bi.py": {
                "file_sha256": source.inventory["configuration_qwen3_bi.py"]["file_sha256"],
                "contract": "requires_explicit_pii_release_eligible_false",
            },
            "modeling_qwen3_bi.py": {
                "file_sha256": source.inventory["modeling_qwen3_bi.py"]["file_sha256"],
                "contract": "accepts_only_pii_release_eligible_false",
            },
        },
    }


def _scan_retained_source_metadata(source: SourceEvidence) -> dict[str, Any]:
    scanned: list[str] = []
    for relative, path in source.files.items():
        if relative in {
            "model.safetensors",
            CHECKSUMS_NAME,
            PREAUTHORIZATION_NAME,
            *_SUPERSEDED_SOURCE_FILES,
            *_FALSE_ONLY_CONTRACT_FILES,
        }:
            continue
        payload = _read_regular_payload(path, field=f"source metadata {relative}").payload
        markers = (
            _TOKENIZER_PUBLICATION_MARKERS + _FALSE_LINEAGE_MARKERS
            if relative == "tokenizer.json"
            else _GENERIC_STALE_MARKERS + _FALSE_LINEAGE_MARKERS
        )
        hits = _marker_hits(payload, markers)
        if hits:
            raise PublicationSuccessorError(
                "STALE_PUBLICATION_MARKER",
                f"retained source metadata {relative!r} contains stale marker {hits[0]!r}",
            )
        scanned.append(relative)
    return {
        "profile": "retained_nonbinary_publication_metadata_v1",
        "status": "PASS",
        "scanned_files": sorted(scanned),
        "false_only_contract_files": sorted(_FALSE_ONLY_CONTRACT_FILES),
        "superseded_publication_files": sorted(_SUPERSEDED_SOURCE_FILES),
        "validated_replacement_files": sorted(_SUPERSEDED_SOURCE_FILES),
        "removed_source_files": [PREAUTHORIZATION_NAME],
        "binary_files_not_text_scanned": ["model.safetensors"],
        "tokenizer_vocabulary_contextual_scan": "tokenizer.json" in source.files,
    }


def _load_schema(relative_path: Path) -> dict[str, Any]:
    payload = _read_regular_payload(
        REPOSITORY_ROOT / relative_path,
        field=f"schema {relative_path.as_posix()}",
        maximum_bytes=4 * 1024 * 1024,
    )
    return _strict_json(payload.payload, field=f"schema {relative_path.as_posix()}")


def _validate_schema(
    document: Mapping[str, Any],
    *,
    schema_path: Path,
    field: str,
    definition: str | None = None,
) -> None:
    schema = _load_schema(schema_path)
    if definition is not None:
        definitions = schema.get("$defs")
        if not isinstance(definitions, Mapping) or not isinstance(
            definitions.get(definition), Mapping
        ):
            raise PublicationSuccessorError(
                "SCHEMA_DEFINITION_MISSING", f"{field} schema definition is unavailable"
            )
        schema = {
            "$schema": schema.get("$schema"),
            "$defs": definitions,
            **dict(definitions[definition]),
        }
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    except Exception as exc:
        raise PublicationSuccessorError(
            "RECEIPT_SCHEMA_REJECTED", f"{field} does not satisfy its closed schema"
        ) from exc


def _load_self_hashed_receipt(
    path: Path | None,
    *,
    field: str,
    missing_blocker: str,
    schema_path: Path,
    schema_definition: str | None = None,
) -> tuple[RegularPayload, dict[str, Any]]:
    if path is None:
        raise PublicationSuccessorError(missing_blocker, f"{field} was not provided")
    payload = _read_regular_payload(path, field=field)
    document = _strict_json(payload.payload, field=field)
    _validate_schema(
        document,
        schema_path=schema_path,
        field=field,
        definition=schema_definition,
    )
    claimed = document.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256.fullmatch(claimed):
        raise PublicationSuccessorError(
            "RECEIPT_SELF_HASH_MISSING", f"{field} has no valid receipt_sha256"
        )
    if claimed != canonical_json_hash(document, remove="receipt_sha256"):
        raise PublicationSuccessorError(
            "RECEIPT_SELF_HASH_MISMATCH", f"{field} self-hash does not verify"
        )
    return payload, document


def _load_publication_text(
    path: Path,
    *,
    field: str,
    allowed_exact_markers: frozenset[str] = frozenset(),
) -> RegularPayload:
    payload = _read_regular_payload(path, field=field, maximum_bytes=8 * 1024 * 1024)
    try:
        text = payload.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicationSuccessorError(
            "INVALID_PUBLICATION_TEXT", f"{field} must be UTF-8"
        ) from exc
    if not text.strip() or "\x00" in text:
        raise PublicationSuccessorError(
            "INVALID_PUBLICATION_TEXT", f"{field} is empty or contains NUL"
        )
    scan_text = text
    for marker in allowed_exact_markers:
        if marker == _LICENSE_HISTORY_MARKER:
            explanatory_paragraphs = [
                paragraph
                for paragraph in re.split(r"\n\s*\n", text)
                if marker in paragraph
            ]
            if len(explanatory_paragraphs) != 1 or text.count(marker) != 1 or any(
                phrase not in explanatory_paragraphs[0].casefold()
                for phrase in _LICENSE_HISTORY_EXPLANATION
            ):
                raise PublicationSuccessorError(
                    "STALE_PREAUTHORIZATION_TEXT",
                    f"{field} does not explain the historical license-status marker",
                )
        scan_text = scan_text.replace(marker, "")
    lowered = scan_text.lower()
    stale = next((marker for marker in _OLD_PUBLICATION_MARKERS if marker in lowered), None)
    if stale is not None:
        raise PublicationSuccessorError(
            "STALE_PREAUTHORIZATION_TEXT",
            f"{field} retains stale pre-authorization marker {stale!r}",
        )
    return payload


def _validate_repository_id(value: str, *, field: str) -> str:
    if not _REPOSITORY_ID.fullmatch(value):
        raise PublicationSuccessorError(
            "INVALID_PUBLICATION_TARGET", f"{field} must be an owner/repository ID"
        )
    return value


def _validate_security_reporting_url(
    payload: RegularPayload, *, github_repository: str
) -> None:
    text = payload.payload.decode("utf-8")
    expected = f"https://github.com/{github_repository}/security/advisories/new"
    observed = {
        match.group(0).rstrip(".,;:!?") for match in _GITHUB_URL.finditer(text)
    }
    if expected not in observed:
        raise PublicationSuccessorError(
            "SECURITY_REPORTING_URL_MISSING",
            "current SECURITY does not contain the exact target private-report URL",
        )


def _model_card_target_values(text: str, *, key: str) -> list[str]:
    values: list[str] = []
    pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?[`\"']?{re.escape(key)}[`\"']?\s*[:=]\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match is None:
            continue
        value = match.group(1).strip().strip("`\"'").strip()
        values.append(value)
    return values


def _validate_model_card_targets(
    payload: RegularPayload,
    *,
    git_source_commit: str,
    github_repository: str,
    hugging_face_repository: str,
) -> None:
    text = payload.payload.decode("utf-8")
    if _MODEL_CARD_TARGET_PLACEHOLDER.search(text):
        raise PublicationSuccessorError(
            "MODEL_CARD_TARGET_PLACEHOLDER",
            "publication Model Card contains an unresolved Hugging Face namespace placeholder",
        )
    expected = {
        "github_repository": github_repository,
        "hugging_face_repository": hugging_face_repository,
        "git_source_commit": git_source_commit,
        "release_tag": RELEASE_TAG,
        "github_release_url": (
            f"https://github.com/{github_repository}/releases/tag/{RELEASE_TAG}"
        ),
    }
    for key, repository in expected.items():
        values = _model_card_target_values(text, key=key)
        if not values or any(value != repository for value in values):
            raise PublicationSuccessorError(
                "MODEL_CARD_TARGET_MISMATCH",
                f"publication Model Card {key} does not match the intended release binding",
            )


def _validate_output_location(source: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise PublicationSuccessorError(
            "OUTPUT_ALREADY_EXISTS", "publication successor output already exists"
        )
    try:
        parent_meta = os.lstat(output.parent)
    except OSError as exc:
        raise PublicationSuccessorError(
            "OUTPUT_PARENT_MISSING", "publication successor parent directory is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_meta.st_mode) or not stat.S_ISDIR(parent_meta.st_mode):
        raise PublicationSuccessorError(
            "UNSAFE_OUTPUT_PARENT", "publication successor parent must be a real directory"
        )
    source_resolved = source.resolve(strict=True)
    output_resolved = output.resolve(strict=False)
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise PublicationSuccessorError(
            "OUTPUT_INSIDE_SOURCE", "publication successor output cannot be inside source package"
        )


def prepare_build_plan(
    *,
    source_package: Path,
    output: Path,
    model_card: Path,
    security: Path,
    notice: Path,
    third_party_notices: Path,
    final_local_receipt: Path | None,
    human_license_approval_receipt: Path | None,
    tested_private_security_channel_receipt: Path | None,
    git_source_commit: str,
    github_repository: str,
    hugging_face_repository: str,
) -> BuildPlan:
    if not _GIT_COMMIT.fullmatch(git_source_commit):
        raise PublicationSuccessorError(
            "INVALID_GIT_SOURCE_COMMIT", "Git source commit must be a full 40- or 64-hex ID"
        )
    github_repository = _validate_repository_id(
        github_repository, field="intended GitHub repository"
    )
    hugging_face_repository = _validate_repository_id(
        hugging_face_repository, field="intended Hugging Face repository"
    )
    _validate_output_location(source_package, output)
    source = verify_source_package(source_package)
    candidate_lineage_contract = _validate_candidate_lineage_contract(source)
    stale_marker_scan = _scan_retained_source_metadata(source)
    publication_model_card = _load_publication_text(model_card, field="publication Model Card")
    if PACKAGE_VERSION not in publication_model_card.payload.decode("utf-8"):
        raise PublicationSuccessorError(
            "MODEL_CARD_VERSION_MISSING",
            f"publication Model Card must identify target version {PACKAGE_VERSION}",
        )
    _validate_model_card_targets(
        publication_model_card,
        git_source_commit=git_source_commit,
        github_repository=github_repository,
        hugging_face_repository=hugging_face_repository,
    )
    current_security = _load_publication_text(security, field="current SECURITY")
    _validate_security_reporting_url(
        current_security, github_repository=github_repository
    )
    publication_notice = _load_publication_text(notice, field="publication NOTICE")
    publication_third_party_notices = _load_publication_text(
        third_party_notices,
        field="publication THIRD_PARTY_NOTICES",
        allowed_exact_markers=frozenset({_LICENSE_HISTORY_MARKER}),
    )
    huggingface_gitattributes = _read_regular_payload(
        REPOSITORY_ROOT / HF_GITATTRIBUTES_TEMPLATE_PATH,
        field="reviewed Hugging Face .gitattributes",
        maximum_bytes=1024 * 1024,
    )
    if huggingface_gitattributes.payload != b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n":
        raise PublicationSuccessorError(
            "INVALID_HUGGING_FACE_GITATTRIBUTES",
            "reviewed Hugging Face .gitattributes template drifted",
        )
    for field, payload in (
        ("publication NOTICE", publication_notice),
        ("publication THIRD_PARTY_NOTICES", publication_third_party_notices),
    ):
        if PACKAGE_VERSION not in payload.payload.decode("utf-8"):
            raise PublicationSuccessorError(
                "PUBLICATION_NOTICE_VERSION_MISSING",
                f"{field} must identify target version {PACKAGE_VERSION}",
            )

    final_payload, final_document = _load_self_hashed_receipt(
        final_local_receipt,
        field="final local receipt",
        missing_blocker="FINAL_LOCAL_RECEIPT_MISSING",
        schema_path=FINAL_LOCAL_RECEIPT_SCHEMA_PATH,
    )
    if (
        final_document.get("status") != "READY_FOR_USER_AUTHORIZATION"
        or final_document.get("local_candidate_complete") is not True
        or final_document.get("blocker_ids") != []
    ):
        raise PublicationSuccessorError(
            "FINAL_LOCAL_RECEIPT_NOT_READY",
            "final local receipt does not close all local-candidate blockers",
        )

    license_payload, license_document = _load_self_hashed_receipt(
        human_license_approval_receipt,
        field="human license approval receipt",
        missing_blocker="HUMAN_LICENSE_APPROVAL_RECEIPT_MISSING",
        schema_path=LICENSE_APPROVAL_SCHEMA_PATH,
        schema_definition="humanLicenseApprovalReceipt",
    )
    security_payload, security_document = _load_self_hashed_receipt(
        tested_private_security_channel_receipt,
        field="tested private security channel receipt",
        missing_blocker="TESTED_PRIVATE_SECURITY_CHANNEL_RECEIPT_MISSING",
        schema_path=SECURITY_CHANNEL_SCHEMA_PATH,
        schema_definition="testedPrivateSecurityChannelReceipt",
    )

    expected_target = {
        "package_version": PACKAGE_VERSION,
        "github_repository": github_repository,
        "hugging_face_repository": hugging_face_repository,
    }
    if license_document.get("target") != expected_target:
        raise PublicationSuccessorError(
            "LICENSE_APPROVAL_TARGET_MISMATCH",
            "human license approval receipt targets a different release or repository",
        )
    if security_document.get("target") != expected_target:
        raise PublicationSuccessorError(
            "SECURITY_CHANNEL_TARGET_MISMATCH",
            "security channel receipt targets a different release or repository",
        )
    reviewed_files = license_document.get("reviewed_files")
    expected_reviewed = {
        "LICENSE": source.inventory["LICENSE"]["file_sha256"],
        "NOTICE": publication_notice.file_sha256,
        "THIRD_PARTY_NOTICES.md": publication_third_party_notices.file_sha256,
    }
    if reviewed_files != expected_reviewed:
        raise PublicationSuccessorError(
            "LICENSE_APPROVAL_BINDING_MISMATCH",
            "human license approval receipt does not bind the source license files",
        )
    if security_document.get("security_file_sha256") != current_security.file_sha256:
        raise PublicationSuccessorError(
            "SECURITY_CHANNEL_BINDING_MISMATCH",
            "security channel receipt does not bind the supplied current SECURITY file",
        )

    return BuildPlan(
        source=source,
        output=output,
        model_card=publication_model_card,
        security=current_security,
        notice=publication_notice,
        third_party_notices=publication_third_party_notices,
        huggingface_gitattributes=huggingface_gitattributes,
        final_local_receipt=final_payload,
        final_local_document=final_document,
        license_approval_receipt=license_payload,
        license_approval_document=license_document,
        security_channel_receipt=security_payload,
        security_channel_document=security_document,
        candidate_lineage_contract=candidate_lineage_contract,
        stale_marker_scan=stale_marker_scan,
        git_source_commit=git_source_commit,
        github_repository=github_repository,
        hugging_face_repository=hugging_face_repository,
    )


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_regular_file(source: Path, destination: Path, *, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_descriptor = os.open(source, _open_read_flags())
    except OSError as exc:
        raise PublicationSuccessorError(
            "SOURCE_CHANGED_DURING_BUILD", "source file became unavailable during copy"
        ) from exc
    destination_descriptor = -1
    try:
        metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PublicationSuccessorError(
                "SOURCE_CHANGED_DURING_BUILD", "source file is no longer regular"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        digest = hashlib.sha256()
        while True:
            block = os.read(source_descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        if digest.hexdigest() != expected_sha256:
            raise PublicationSuccessorError(
                "SOURCE_CHANGED_DURING_BUILD", "source checksum changed during copy"
            )
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _stage_model_weights(source: Path, destination: Path, *, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_descriptor = -1
    destination_descriptor = -1
    try:
        source_descriptor = os.open(source, _open_read_flags())
        source_metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise PublicationSuccessorError(
                "MODEL_WEIGHT_SOURCE_UNSAFE", "model weight source is not a regular file"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        fcntl.ioctl(destination_descriptor, _FICLONE, source_descriptor)
        os.fsync(destination_descriptor)
        method = "reflink"
    except OSError as exc:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
            destination_descriptor = -1
        if source_descriptor >= 0:
            os.close(source_descriptor)
            source_descriptor = -1
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if exc.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.ENOTTY,
            errno.EINVAL,
        }:
            raise PublicationSuccessorError(
                "MODEL_WEIGHT_STAGE_FAILED", "model weights could not be reflinked safely"
            ) from exc
        _copy_regular_file(source, destination, expected_sha256=expected_sha256)
        method = "copy"
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
    try:
        source_metadata = os.lstat(source)
        destination_metadata = os.lstat(destination)
    except OSError as exc:
        raise PublicationSuccessorError(
            "MODEL_WEIGHT_STAGE_FAILED", "staged model weights became unavailable"
        ) from exc
    if (
        not stat.S_ISREG(source_metadata.st_mode)
        or not stat.S_ISREG(destination_metadata.st_mode)
        or (source_metadata.st_dev, source_metadata.st_ino)
        == (destination_metadata.st_dev, destination_metadata.st_ino)
    ):
        raise PublicationSuccessorError(
            "MODEL_WEIGHT_INODE_NOT_ISOLATED",
            "staged model weights must have an inode independent from the source",
        )
    source_observed, _source_size = _hash_regular_file(source, field="source model.safetensors")
    observed, _size = _hash_regular_file(destination, field="staged model.safetensors")
    if source_observed != expected_sha256 or observed != expected_sha256:
        raise PublicationSuccessorError(
            "STAGED_MODEL_CHECKSUM_MISMATCH", "staged model.safetensors checksum drifted"
        )
    return method


def _binding(path: Path, *, provenance: str, transfer_method: str) -> dict[str, Any]:
    digest, size = _hash_regular_file(path, field=f"staged file {path.name}")
    return {
        "file_sha256": digest,
        "size_bytes": size,
        "provenance": provenance,
        "transfer_method": transfer_method,
        "output_type": "regular_file",
    }


def _make_manifest(plan: BuildPlan, files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "pii-zh.community-v2-publication-successor-manifest.v1",
        "package_version": PACKAGE_VERSION,
        "publication_state": "staged_not_uploaded",
        "publication_authorization_source": "external_receipts",
        "source_package": {
            "format": "immutable_checksum_closed_model_package",
            "inventory": plan.source.inventory,
            "inventory_sha256": plan.source.inventory_sha256,
            "checksums_file_sha256": plan.source.checksums_file_sha256,
            "verified_file_count": len(plan.source.inventory),
        },
        "source_control": {
            "git_source_commit": plan.git_source_commit,
        },
        "publication_targets": {
            "github_repository": plan.github_repository,
            "hugging_face_repository": plan.hugging_face_repository,
        },
        "remote_revisions": {
            "github_commit": None,
            "hugging_face_commit": None,
        },
        "approval_evidence": {
            "final_local_receipt": {
                "path": FINAL_LOCAL_RECEIPT_NAME,
                "file_sha256": plan.final_local_receipt.file_sha256,
                "receipt_sha256": plan.final_local_document["receipt_sha256"],
                "local_candidate_complete": True,
            },
            "human_license_approval": {
                "path": LICENSE_APPROVAL_RECEIPT_NAME,
                "file_sha256": plan.license_approval_receipt.file_sha256,
                "receipt_sha256": plan.license_approval_document["receipt_sha256"],
                "approved": True,
            },
            "tested_private_security_channel": {
                "path": SECURITY_CHANNEL_RECEIPT_NAME,
                "file_sha256": plan.security_channel_receipt.file_sha256,
                "receipt_sha256": plan.security_channel_document["receipt_sha256"],
                "tested": True,
            },
        },
        "candidate_lineage_contract": plan.candidate_lineage_contract,
        "stale_publication_marker_scan": plan.stale_marker_scan,
        "payload_files": dict(sorted(files.items())),
        "payload_inventory_sha256": canonical_json_hash(files),
        "excluded_source_files": [PREAUTHORIZATION_NAME],
        "superseded_source_files": sorted(_SUPERSEDED_SOURCE_FILES),
        "checksum_policy": {
            "algorithm": "sha256",
            "manifest_covers_payload_only": True,
            "checksums_covers_manifest_and_payload": True,
            "checksums_self_entry": False,
        },
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest, remove="manifest_sha256")
    _validate_schema(
        manifest,
        schema_path=MANIFEST_SCHEMA_PATH,
        field="publication successor manifest",
    )
    return manifest


def _json_file_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_checksums(directory: Path) -> None:
    files = _scan_tree(directory, field="publication successor")
    if CHECKSUMS_NAME in files:
        raise PublicationSuccessorError(
            "OUTPUT_CHECKSUMS_ALREADY_EXISTS", "output checksums.txt already exists"
        )
    lines: list[str] = []
    for relative, path in files.items():
        digest, _size = _hash_regular_file(path, field=f"publication successor {relative}")
        lines.append(f"{digest}  {relative}")
    _write_new_file(directory / CHECKSUMS_NAME, ("\n".join(lines) + "\n").encode("utf-8"))


def verify_successor_package(directory: Path) -> dict[str, Any]:
    files = _scan_tree(directory, field="publication successor")
    checksums_path = files.get(CHECKSUMS_NAME)
    if checksums_path is None:
        raise PublicationSuccessorError(
            "OUTPUT_CHECKSUMS_MISSING", "publication successor has no checksums.txt"
        )
    checksums_payload = _read_regular_payload(
        checksums_path, field="publication successor checksums.txt"
    )
    declared = _parse_checksums(checksums_payload.payload)
    actual = set(files) - {CHECKSUMS_NAME}
    if set(declared) != actual:
        raise PublicationSuccessorError(
            "OUTPUT_CHECKSUM_CLOSURE_MISMATCH",
            "publication successor checksums do not cover the package exactly",
        )
    for relative, expected in declared.items():
        observed, _size = _hash_regular_file(
            files[relative], field=f"publication successor {relative}"
        )
        if observed != expected:
            raise PublicationSuccessorError(
                "OUTPUT_CHECKSUM_MISMATCH", f"output checksum mismatch for {relative!r}"
            )
    if PREAUTHORIZATION_NAME in files:
        raise PublicationSuccessorError(
            "OUTPUT_PREAUTHORIZATION_RETAINED",
            "publication successor retained the pre-authorization receipt",
        )
    manifest_payload = _read_regular_payload(
        files[MANIFEST_NAME], field="publication successor manifest"
    )
    manifest = _strict_json(manifest_payload.payload, field="publication successor manifest")
    _validate_schema(
        manifest,
        schema_path=MANIFEST_SCHEMA_PATH,
        field="publication successor manifest",
    )
    if manifest.get("manifest_sha256") != canonical_json_hash(manifest, remove="manifest_sha256"):
        raise PublicationSuccessorError(
            "MANIFEST_SELF_HASH_MISMATCH", "publication successor manifest self-hash failed"
        )
    payload_files = manifest.get("payload_files")
    if not isinstance(payload_files, Mapping):  # schema validation is fail-closed above
        raise PublicationSuccessorError(
            "MANIFEST_PAYLOAD_BINDING_MISMATCH",
            "publication successor manifest has no payload inventory",
        )
    expected_payload_names = actual - {MANIFEST_NAME}
    if set(payload_files) != expected_payload_names:
        raise PublicationSuccessorError(
            "MANIFEST_PAYLOAD_BINDING_MISMATCH",
            "publication successor manifest payload paths differ from the package",
        )
    observed_payload_files: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_payload_names):
        digest, size = _hash_regular_file(
            files[relative], field=f"manifest-bound publication payload {relative}"
        )
        binding = payload_files.get(relative)
        if not isinstance(binding, Mapping) or (
            binding.get("file_sha256") != digest or binding.get("size_bytes") != size
        ):
            raise PublicationSuccessorError(
                "MANIFEST_PAYLOAD_BINDING_MISMATCH",
                f"publication successor manifest binding differs for {relative!r}",
            )
        observed_payload_files[relative] = dict(binding)
    observed_inventory_sha256 = canonical_json_hash(observed_payload_files)
    if manifest.get("payload_inventory_sha256") != observed_inventory_sha256:
        raise PublicationSuccessorError(
            "MANIFEST_PAYLOAD_INVENTORY_MISMATCH",
            "publication successor manifest payload inventory hash does not verify",
        )
    return {
        "file_count": len(files),
        "verified_file_count": len(declared),
        "checksums_file_sha256": checksums_payload.file_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _safe_remove_created_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = os.lstat(path)
    except OSError:
        return
    if (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == identity
    ):
        shutil.rmtree(path)


def build_publication_successor(plan: BuildPlan) -> dict[str, Any]:
    try:
        plan.output.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError as exc:
        raise PublicationSuccessorError(
            "OUTPUT_ALREADY_EXISTS", "publication successor output already exists"
        ) from exc
    output_meta = os.lstat(plan.output)
    output_identity = (output_meta.st_dev, output_meta.st_ino)
    payload_bindings: dict[str, dict[str, Any]] = {}
    try:
        replacements = {
            "README.md": (plan.model_card, "caller_publication_model_card"),
            "SECURITY.md": (plan.security, "caller_current_security"),
            "NOTICE": (plan.notice, "caller_publication_notice"),
            "THIRD_PARTY_NOTICES.md": (
                plan.third_party_notices,
                "caller_publication_third_party_notices",
            ),
            FINAL_LOCAL_RECEIPT_NAME: (plan.final_local_receipt, "final_local_receipt"),
            LICENSE_APPROVAL_RECEIPT_NAME: (
                plan.license_approval_receipt,
                "human_license_approval_receipt",
            ),
            SECURITY_CHANNEL_RECEIPT_NAME: (
                plan.security_channel_receipt,
                "tested_private_security_channel_receipt",
            ),
            HF_GITATTRIBUTES_NAME: (
                plan.huggingface_gitattributes,
                "reviewed_hugging_face_gitattributes",
            ),
        }
        excluded = {CHECKSUMS_NAME, PREAUTHORIZATION_NAME, *_SUPERSEDED_SOURCE_FILES}
        for relative, source_path in plan.source.files.items():
            if relative in excluded:
                continue
            destination = plan.output / PurePosixPath(relative)
            expected = plan.source.inventory[relative]["file_sha256"]
            if relative == "model.safetensors":
                method = _stage_model_weights(source_path, destination, expected_sha256=expected)
            else:
                _copy_regular_file(source_path, destination, expected_sha256=expected)
                method = "copy"
            payload_bindings[relative] = _binding(
                destination,
                provenance=(
                    "source_package_model_weights"
                    if relative == "model.safetensors"
                    else "source_package"
                ),
                transfer_method=method,
            )

        for relative, (payload, provenance) in replacements.items():
            destination = plan.output / relative
            _write_new_file(destination, payload.payload)
            payload_bindings[relative] = _binding(
                destination, provenance=provenance, transfer_method="copy"
            )

        manifest = _make_manifest(plan, payload_bindings)
        _write_new_file(plan.output / MANIFEST_NAME, _json_file_bytes(manifest))
        _write_checksums(plan.output)
        verification = verify_successor_package(plan.output)
        return {
            "status": "STAGED",
            "publication_state": "staged_not_uploaded",
            "package_version": PACKAGE_VERSION,
            "source_inventory_sha256": plan.source.inventory_sha256,
            **verification,
        }
    except Exception:
        _safe_remove_created_directory(plan.output, output_identity)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline, checksum-closed community-v2 publication successor."
    )
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-card", required=True, type=Path)
    parser.add_argument("--security", required=True, type=Path)
    parser.add_argument("--notice", required=True, type=Path)
    parser.add_argument("--third-party-notices", required=True, type=Path)
    parser.add_argument("--final-local-receipt", type=Path)
    parser.add_argument("--human-license-approval-receipt", type=Path)
    parser.add_argument("--tested-private-security-channel-receipt", type=Path)
    parser.add_argument("--git-source-commit", required=True)
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--hugging-face-repository", required=True)
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument("--dry-run", action="store_true")
    preflight.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "preflight" if args.dry_run or args.preflight else "build"
    try:
        plan = prepare_build_plan(
            source_package=args.source_package,
            output=args.output,
            model_card=args.model_card,
            security=args.security,
            notice=args.notice,
            third_party_notices=args.third_party_notices,
            final_local_receipt=args.final_local_receipt,
            human_license_approval_receipt=args.human_license_approval_receipt,
            tested_private_security_channel_receipt=(args.tested_private_security_channel_receipt),
            git_source_commit=args.git_source_commit,
            github_repository=args.github_repository,
            hugging_face_repository=args.hugging_face_repository,
        )
        if mode == "preflight":
            result = {
                "status": "READY",
                "mode": "preflight",
                "publication_state": "not_staged",
                "package_version": PACKAGE_VERSION,
                "source_inventory_sha256": plan.source.inventory_sha256,
                "source_verified_file_count": len(plan.source.inventory),
                "remote_write_performed": False,
            }
        else:
            result = {"mode": "build", **build_publication_successor(plan)}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except PublicationSuccessorError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "mode": mode,
                    "blocker_ids": [exc.blocker_id],
                    "reason": str(exc),
                    "remote_write_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    sys.exit(main())
