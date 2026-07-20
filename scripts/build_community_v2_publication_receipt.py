#!/usr/bin/env python3
"""Build and validate the community-v2 pre-publication gate receipt offline.

This module is deliberately not a remote collector.  It never reads token or
``.env`` files, imports a Hub/GitHub client, or opens a network connection.  A
caller must first use separately reviewed remote collectors, the reviewed HF
snapshot materializer, and the local GitHub Release asset verifier to produce
the four self-hashed evidence documents accepted here.  This producer validates
those documents, recomputes their relationships, and binds them into a gate
receipt.  It does not attest final publication.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path("configs/release/community_v2_publication_receipt.schema.json")
PACKAGE_VERSION = "0.2.0rc1"
TAG_NAME = "v0.2.0rc1"
EXPECTED_GITHUB_REPOSITORY = "whyiug/pii-detect-model"
EXPECTED_HUGGING_FACE_REPOSITORY = "Forrest20231206/pii-zh-qwen3-0.6b-24class"

GITHUB_EVIDENCE_DEFINITION = "githubRemoteEvidence"
HUGGING_FACE_EVIDENCE_DEFINITION = "huggingFaceRemoteEvidence"
HUGGING_FACE_DOWNLOAD_EVIDENCE_DEFINITION = "huggingFaceDownloadProvenanceEvidence"
GITHUB_EVIDENCE_SCHEMA_VERSION = "pii-zh.community-v2-github-publication-evidence.v1"
HUGGING_FACE_EVIDENCE_SCHEMA_VERSION = (
    "pii-zh.community-v2-hugging-face-publication-evidence.v1"
)
HUGGING_FACE_DOWNLOAD_EVIDENCE_SCHEMA_VERSION = (
    "pii-zh.community-v2-hf-download-provenance.v1"
)
HUGGING_FACE_DOWNLOAD_GENERATOR = Path(
    "scripts/materialize_community_v2_hf_snapshot.py"
)
REMOTE_EVIDENCE_COLLECTOR_GENERATOR = Path(
    "scripts/collect_community_v2_remote_evidence.py"
)
GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA = Path(
    "configs/release/community_v2_github_release_assets_receipt.schema.json"
)
GITHUB_RELEASE_ASSETS_RECEIPT_GENERATOR = Path(
    "scripts/build_community_v2_github_release_assets_receipt.py"
)
GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA_VERSION = (
    "pii-zh.community-v2-github-release-assets-receipt.v1"
)

REQUIRED_HUGGING_FACE_FILES = (
    ".gitattributes",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "checksums.txt",
    "community_v2_final_local_receipt.json",
    "config.json",
    "human_license_approval_receipt.json",
    "model.safetensors",
    "publication_manifest.json",
)
PRIVATE_SECURITY_CHANNEL_EVIDENCE_FILES = frozenset(
    {
        "private_security_channel_waiver_receipt.json",
        "tested_private_security_channel_receipt.json",
    }
)
REQUIRED_GITHUB_RELEASE_ASSETS = {
    "checksums": "checksums.txt",
    "sbom": "sbom.cdx.json",
    "wheel": "pii_zh_qwen-0.2.0rc1-py3-none-any.whl",
}
DRAFT_RECEIPT_ASSET_NAME = "community-v2-pre-publication-gate-receipt.json"

PRODUCER = {
    "name": "build_community_v2_publication_receipt.py",
    "network_accessed": False,
    "remote_mutation_performed": False,
    "input_policy": (
        "caller_supplied_self_hashed_remote_collector_download_provenance_and_local_release_assets_evidence"
    ),
    "remote_collector_still_required": True,
}
VERIFICATION = {
    "derivation": "relationships_recomputed_from_bound_remote_evidence",
    "caller_supplied_status_accepted": False,
    "reviewed_collector_implementation_bound": True,
    "github_tag_source_relationship_verified": True,
    "github_hosted_ci_exact_source_verified": True,
    "github_draft_release_relationship_verified": True,
    "github_draft_release_hf_revision_reference_verified": True,
    "release_asset_digests_match": True,
    "release_checksums_match_downloaded_assets": True,
    "github_release_assets_local_verification_bound": True,
    "github_release_assets_exact_remote_match_verified": True,
    "github_release_assets_checksum_closure_verified": True,
    "github_release_assets_wheel_inventory_verified": True,
    "github_release_assets_isolated_smoke_verified": True,
    "github_release_assets_public_scan_verified": True,
    "hugging_face_immutable_revision_verified": True,
    "hugging_face_inventory_unique_and_sorted": True,
    "hugging_face_download_verification_bound": True,
    "hugging_face_package_verification_bound": True,
    "remote_repository_references_verified": True,
}
LIMITATIONS = {
    "remote_collector_required": True,
    "producer_did_not_query_remote": True,
    "producer_did_not_modify_remote": True,
    "collector_authenticity_is_caller_responsibility": True,
    "does_not_attest_final_publication": True,
    "github_release_must_remain_draft": True,
    "hugging_face_repository_must_remain_private": True,
    "safe_to_attach_to_draft_release": True,
    "final_visibility_transition_requires_separate_confirmation": True,
    "statement": (
        "This offline producer validates caller-supplied pre-publication collector evidence; "
        "it does not prove collector authenticity or attest that the draft GitHub Release or "
        "private Hugging Face model has been finally published."
    ),
}

_MAX_JSON_BYTES = 16 * 1024 * 1024
_READ_BLOCK_BYTES = 1024 * 1024


class PublicationReceiptError(RuntimeError):
    """A stable, fail-closed publication receipt validation error."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class JsonPayload:
    """Stable bytes and parsed document from one regular local input file."""

    payload: bytes
    document: Mapping[str, Any]
    file_sha256: str
    mode: int


def _canonical_json_bytes(value: Any, *, remove: str | None = None) -> bytes:
    document = dict(value) if isinstance(value, Mapping) else value
    if remove is not None:
        if not isinstance(document, dict):
            raise PublicationReceiptError(
                "INVALID_CANONICAL_JSON",
                "a removable self-hash field requires a JSON object",
            )
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
        raise PublicationReceiptError(
            "INVALID_CANONICAL_JSON", "document is not canonical-JSON serializable"
        ) from exc


def canonical_json_hash(value: Any, *, remove: str | None = None) -> str:
    """Return the lowercase SHA-256 of canonical UTF-8 JSON bytes."""

    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PublicationReceiptError(
            "INVALID_OUTPUT_JSON", "receipt cannot be serialized as strict JSON"
        ) from exc


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationReceiptError(
                    "DUPLICATE_JSON_KEY", f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise PublicationReceiptError(
            "NONFINITE_JSON_NUMBER", f"{field} contains a non-finite JSON number"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise PublicationReceiptError(
            "INVALID_JSON_ENCODING", f"{field} is not UTF-8 JSON"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PublicationReceiptError("INVALID_JSON", f"{field} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PublicationReceiptError("INVALID_JSON_SHAPE", f"{field} must be a JSON object")
    return value


def _read_regular_json(path: Path, *, field: str) -> JsonPayload:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationReceiptError(
            "UNSAFE_INPUT_FILE", f"{field} is missing, non-regular, or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
            raise PublicationReceiptError(
                "UNSAFE_INPUT_FILE", f"{field} must be a bounded regular file"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, _MAX_JSON_BYTES + 1 - total))
            if not block:
                break
            total += len(block)
            if total > _MAX_JSON_BYTES:
                raise PublicationReceiptError(
                    "INPUT_TOO_LARGE", f"{field} exceeds the JSON input size limit"
                )
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        identities = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ), (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identities[0] != identities[1] or total != before.st_size:
            raise PublicationReceiptError(
                "INPUT_CHANGED_DURING_READ", f"{field} changed while it was read"
            )
        payload = b"".join(chunks)
        return JsonPayload(
            payload=payload,
            document=_strict_json(payload, field=field),
            file_sha256=digest.hexdigest(),
            mode=stat.S_IMODE(after.st_mode),
        )
    finally:
        os.close(descriptor)


def _schema() -> dict[str, Any]:
    return _read_regular_json(
        REPOSITORY_ROOT / SCHEMA_PATH,
        field=f"schema {SCHEMA_PATH.as_posix()}",
    ).document


def _definition_schema(name: str) -> dict[str, Any]:
    root = _schema()
    definitions = root.get("$defs")
    if not isinstance(definitions, Mapping) or not isinstance(definitions.get(name), Mapping):
        raise PublicationReceiptError(
            "SCHEMA_DEFINITION_MISSING", f"schema definition {name!r} is unavailable"
        )
    return {
        "$schema": root.get("$schema"),
        "$defs": definitions,
        **dict(definitions[name]),
    }


def _validate_schema(
    document: Mapping[str, Any], *, field: str, definition: str | None = None
) -> None:
    schema = _definition_schema(definition) if definition is not None else _schema()
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    except Exception as exc:
        raise PublicationReceiptError(
            "SCHEMA_REJECTED", f"{field} does not satisfy its closed schema"
        ) from exc


def _validate_self_hash(document: Mapping[str, Any], *, field: str, key: str) -> None:
    claimed = document.get(key)
    if claimed != canonical_json_hash(document, remove=key):
        raise PublicationReceiptError(
            "SELF_HASH_MISMATCH", f"{field} canonical self-hash does not verify"
        )


def _validate_repo_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise PublicationReceiptError("UNSAFE_REPOSITORY_PATH", f"{field} is not a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicationReceiptError(
            "UNSAFE_REPOSITORY_PATH", f"{field} is not a normalized relative repository path"
        )
    return value


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise PublicationReceiptError("INVALID_TIMESTAMP", f"{field} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationReceiptError("INVALID_TIMESTAMP", f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise PublicationReceiptError("INVALID_TIMESTAMP", f"{field} has no timezone")
    return parsed


def load_remote_evidence(path: Path, *, platform: str) -> JsonPayload:
    """Load and validate one caller-supplied remote collector evidence file."""

    definitions = {
        "github": GITHUB_EVIDENCE_DEFINITION,
        "hugging_face": HUGGING_FACE_EVIDENCE_DEFINITION,
    }
    if platform not in definitions:
        raise PublicationReceiptError("UNKNOWN_PLATFORM", f"unknown platform {platform!r}")
    payload = _read_regular_json(path, field=f"{platform} remote collector evidence")
    _validate_remote_evidence_payload(payload, platform=platform)
    return payload


def _validate_remote_evidence_payload(payload: JsonPayload, *, platform: str) -> None:
    definitions = {
        "github": GITHUB_EVIDENCE_DEFINITION,
        "hugging_face": HUGGING_FACE_EVIDENCE_DEFINITION,
    }
    if platform not in definitions:
        raise PublicationReceiptError("UNKNOWN_PLATFORM", f"unknown platform {platform!r}")
    if hashlib.sha256(payload.payload).hexdigest() != payload.file_sha256:
        raise PublicationReceiptError(
            "EVIDENCE_FILE_HASH_MISMATCH",
            f"{platform} evidence bytes do not match their file binding",
        )
    parsed = _strict_json(payload.payload, field=f"{platform} remote collector evidence")
    if parsed != payload.document:
        raise PublicationReceiptError(
            "EVIDENCE_DOCUMENT_MISMATCH",
            f"{platform} evidence document does not match its source bytes",
        )
    _validate_schema(
        payload.document,
        field=f"{platform} remote collector evidence",
        definition=definitions[platform],
    )
    _validate_self_hash(
        payload.document,
        field=f"{platform} remote collector evidence",
        key="evidence_sha256",
    )
    collector = payload.document["collector"]
    expected_generator_sha256 = _sha256_regular_file(
        REPOSITORY_ROOT / REMOTE_EVIDENCE_COLLECTOR_GENERATOR,
        field="reviewed remote evidence collector",
    )
    if collector["generator_file_sha256"] != expected_generator_sha256:
        raise PublicationReceiptError(
            "REMOTE_COLLECTOR_GENERATOR_MISMATCH",
            f"{platform} evidence was not produced by the reviewed collector implementation",
        )


def _sha256_regular_file(path: Path, *, field: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationReceiptError(
            "UNSAFE_LOCAL_IMPLEMENTATION", f"{field} is missing or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
            raise PublicationReceiptError(
                "UNSAFE_LOCAL_IMPLEMENTATION", f"{field} must be a bounded regular file"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if total > _MAX_JSON_BYTES:
                raise PublicationReceiptError(
                    "UNSAFE_LOCAL_IMPLEMENTATION", f"{field} exceeds the size limit"
                )
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or total != before.st_size:
            raise PublicationReceiptError(
                "LOCAL_IMPLEMENTATION_CHANGED_DURING_READ",
                f"{field} changed while it was read",
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _repository_head_sha() -> str:
    """Return the exact local HEAD commit without consulting user Git config."""

    git = shutil.which("git", path=os.defpath)
    if git is None:
        raise PublicationReceiptError(
            "LOCAL_GIT_UNAVAILABLE", "Git is required to verify the local source HEAD"
        )
    environment = {
        "PATH": os.defpath,
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        completed = subprocess.run(
            [git, "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicationReceiptError(
            "LOCAL_HEAD_VERIFICATION_FAILED", "could not verify the local source HEAD"
        ) from exc
    try:
        observed = completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise PublicationReceiptError(
            "LOCAL_HEAD_VERIFICATION_FAILED", "local source HEAD is not a full Git commit"
        ) from exc
    if (
        completed.returncode != 0
        or len(completed.stdout) > 128
        or len(completed.stderr) > 4096
        or len(observed) != 40
        or any(character not in "0123456789abcdef" for character in observed)
    ):
        raise PublicationReceiptError(
            "LOCAL_HEAD_VERIFICATION_FAILED", "local source HEAD is not a full Git commit"
        )
    return observed


def _github_release_assets_schema() -> JsonPayload:
    payload = _read_regular_json(
        REPOSITORY_ROOT / GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA,
        field="GitHub Release assets verification schema",
    )
    try:
        Draft202012Validator.check_schema(payload.document)
    except Exception as exc:
        raise PublicationReceiptError(
            "RELEASE_ASSETS_SCHEMA_INVALID",
            "GitHub Release assets verification schema is invalid",
        ) from exc
    return payload


def load_github_release_assets_verification(path: Path) -> JsonPayload:
    """Load the immutable receipt produced by the reviewed local asset verifier."""

    payload = _read_regular_json(path, field="GitHub Release assets verification receipt")
    _validate_github_release_assets_verification_payload(payload)
    return payload


def _validate_github_release_assets_verification_payload(payload: JsonPayload) -> None:
    field = "GitHub Release assets verification receipt"
    if payload.mode != 0o444:
        raise PublicationReceiptError(
            "RELEASE_ASSETS_RECEIPT_MODE_MISMATCH",
            "GitHub Release assets verification receipt must have mode 0444",
        )
    if hashlib.sha256(payload.payload).hexdigest() != payload.file_sha256:
        raise PublicationReceiptError(
            "EVIDENCE_FILE_HASH_MISMATCH",
            "GitHub Release assets verification bytes do not match their file binding",
        )
    parsed = _strict_json(payload.payload, field=field)
    if parsed != payload.document:
        raise PublicationReceiptError(
            "EVIDENCE_DOCUMENT_MISMATCH",
            "GitHub Release assets verification document differs from its source bytes",
        )

    schema_payload = _github_release_assets_schema()
    try:
        Draft202012Validator(
            schema_payload.document, format_checker=FormatChecker()
        ).validate(payload.document)
    except Exception as exc:
        raise PublicationReceiptError(
            "RELEASE_ASSETS_RECEIPT_SCHEMA_REJECTED",
            "GitHub Release assets verification receipt violates its independent schema",
        ) from exc
    _validate_self_hash(payload.document, field=field, key="receipt_sha256")

    implementation = payload.document["implementation"]
    expected_generator_sha256 = _sha256_regular_file(
        REPOSITORY_ROOT / GITHUB_RELEASE_ASSETS_RECEIPT_GENERATOR,
        field="reviewed GitHub Release assets verification generator",
    )
    if (
        implementation["generator_path"]
        != GITHUB_RELEASE_ASSETS_RECEIPT_GENERATOR.as_posix()
        or implementation["generator_file_sha256"] != expected_generator_sha256
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_GENERATOR_MISMATCH",
            "GitHub Release assets receipt was not produced by the reviewed generator",
        )
    if (
        implementation["schema_path"]
        != GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA.as_posix()
        or implementation["schema_file_sha256"] != schema_payload.file_sha256
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_SCHEMA_BINDING_MISMATCH",
            "GitHub Release assets receipt does not bind the reviewed independent schema",
        )

    source_commit = payload.document["source_commit"]
    if (
        payload.document["source"]["git_source_commit"] != source_commit
        or source_commit != _repository_head_sha()
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_HEAD_SOURCE_MISMATCH",
            "GitHub Release assets verification does not bind the exact local HEAD",
        )

    assets = payload.document["assets"]
    checksum_closure = payload.document["checksum_closure"]
    expected_checksum_entries = [
        {
            "name": assets[role]["name"],
            "file_sha256": assets[role]["file_sha256"],
        }
        for role in ("wheel", "sbom")
    ]
    if (
        checksum_closure["status"] != "PASS"
        or checksum_closure["line_count"] != len(expected_checksum_entries)
        or checksum_closure["ordered_entries"] != expected_checksum_entries
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_LOCAL_CHECKSUM_CLOSURE_MISMATCH",
            "GitHub Release assets receipt checksum closure does not bind its assets",
        )

    wheel_inventory = payload.document["wheel_inventory"]
    members = wheel_inventory["members"]
    if (
        wheel_inventory["status"] != "PASS"
        or wheel_inventory["member_count"] != len(members)
        or wheel_inventory["member_inventory_sha256"]
        != canonical_json_hash(members)
        or not set(wheel_inventory["required_member_ids"]).issubset(members)
        or wheel_inventory["required_members_present"] is not True
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_WHEEL_INVENTORY_MISMATCH",
            "GitHub Release wheel inventory does not independently self-verify",
        )
    if payload.document["isolated_wheel_smoke"]["status"] != "PASS":
        raise PublicationReceiptError(
            "RELEASE_ASSETS_SMOKE_NOT_PASS",
            "GitHub Release wheel smoke verification is not PASS",
        )
    public_scan = payload.document["public_artifact_scan"]
    if (
        public_scan["status"] != "PASS"
        or public_scan["finding_count"] != 0
        or public_scan["finding_kinds"] != []
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_PUBLIC_SCAN_NOT_PASS",
            "GitHub Release public artifact scan is not a clean PASS",
        )


def load_hugging_face_download_verification(path: Path) -> JsonPayload:
    """Load the self-hashed receipt produced by the reviewed HF materializer."""

    payload = _read_regular_json(path, field="Hugging Face download verification receipt")
    _validate_hugging_face_download_payload(payload)
    return payload


def _validate_hugging_face_download_payload(payload: JsonPayload) -> None:
    field = "Hugging Face download verification receipt"
    if hashlib.sha256(payload.payload).hexdigest() != payload.file_sha256:
        raise PublicationReceiptError(
            "EVIDENCE_FILE_HASH_MISMATCH",
            "Hugging Face download verification bytes do not match their file binding",
        )
    parsed = _strict_json(payload.payload, field=field)
    if parsed != payload.document:
        raise PublicationReceiptError(
            "EVIDENCE_DOCUMENT_MISMATCH",
            "Hugging Face download verification document differs from its source bytes",
        )
    _validate_schema(
        payload.document,
        field=field,
        definition=HUGGING_FACE_DOWNLOAD_EVIDENCE_DEFINITION,
    )
    _validate_self_hash(payload.document, field=field, key="receipt_sha256")

    local_root = payload.document["local_root"]
    files = local_root["files"]
    if local_root["file_count"] != len(files) or local_root[
        "inventory_sha256"
    ] != canonical_json_hash(files):
        raise PublicationReceiptError(
            "HF_DOWNLOAD_INVENTORY_SELF_HASH_MISMATCH",
            "Hugging Face download verification inventory does not self-verify",
        )
    remote_snapshot = payload.document["remote_snapshot"]
    remote_files = remote_snapshot["files"]
    observed_coverage = {
        "size_count": sum(item["size_bytes"] is not None for item in remote_files.values()),
        "git_blob_oid_count": sum(
            item["git_blob_oid"] is not None for item in remote_files.values()
        ),
        "lfs_oid_count": sum(
            item["lfs_oid_sha256"] is not None for item in remote_files.values()
        ),
        "content_verified_count": sum(
            item["content_verification"] != "immutable_revision_transport_only"
            for item in remote_files.values()
        ),
    }
    if (
        remote_snapshot["file_count"] != len(remote_files)
        or remote_snapshot["metadata_inventory_sha256"]
        != canonical_json_hash(remote_files)
        or remote_snapshot["metadata_coverage"] != observed_coverage
        or set(remote_files) != set(files)
        or remote_snapshot["visibility"]
        != ("private" if remote_snapshot["private"] else "public")
    ):
        raise PublicationReceiptError(
            "HF_DOWNLOAD_REMOTE_SNAPSHOT_SELF_HASH_MISMATCH",
            "Hugging Face download remote snapshot does not self-verify",
        )
    expected_generator_sha256 = _sha256_regular_file(
        REPOSITORY_ROOT / HUGGING_FACE_DOWNLOAD_GENERATOR,
        field="reviewed Hugging Face download generator",
    )
    if payload.document["generator"]["file_sha256"] != expected_generator_sha256:
        raise PublicationReceiptError(
            "HF_DOWNLOAD_GENERATOR_MISMATCH",
            "Hugging Face download verification was not produced by the reviewed generator",
        )


def _validate_github_relationships(document: Mapping[str, Any]) -> None:
    repository = document["repository"]
    source_sha = document["source_commit_sha"]
    tag = document["signed_tag"]
    hosted_ci = document["hosted_ci"]
    release = document["release"]
    cross_reference = document["cross_reference"]

    if repository["id"] != EXPECTED_GITHUB_REPOSITORY:
        raise PublicationReceiptError(
            "GITHUB_REPOSITORY_MISMATCH", "GitHub evidence targets a different repository"
        )
    if tag["ref_target_sha"] != tag["tag_object_sha"]:
        raise PublicationReceiptError(
            "GITHUB_TAG_OBJECT_MISMATCH", "signed tag ref does not resolve to its tag object"
        )
    if tag["tag_target_sha"] != source_sha:
        raise PublicationReceiptError(
            "GITHUB_TAG_SOURCE_MISMATCH", "signed tag object does not target the source commit"
        )
    if release["tag_name"] != tag["name"] or release["resolved_source_sha"] != source_sha:
        raise PublicationReceiptError(
            "GITHUB_RELEASE_SOURCE_MISMATCH",
            "GitHub Release does not resolve through the signed tag to the source commit",
        )
    if hashlib.sha256(release["body"].encode("utf-8")).hexdigest() != release[
        "body_sha256"
    ]:
        raise PublicationReceiptError(
            "GITHUB_RELEASE_BODY_HASH_MISMATCH",
            "GitHub draft Release body does not match its collector-supplied digest",
        )
    if hosted_ci["head_sha"] != source_sha:
        raise PublicationReceiptError(
            "GITHUB_HOSTED_CI_SOURCE_MISMATCH",
            "successful hosted CI did not run against the exact signed source commit",
        )
    if cross_reference["hugging_face_repository"] != EXPECTED_HUGGING_FACE_REPOSITORY:
        raise PublicationReceiptError(
            "GITHUB_HF_REFERENCE_MISMATCH",
            "GitHub source does not declare the intended Hugging Face repository",
        )
    _validate_repo_path(cross_reference["path"], field="GitHub cross-reference path")

    assets = release["assets"]
    names = [asset["name"] for asset in assets]
    identifiers = [asset["asset_id"] for asset in assets]
    if names != sorted(names) or len(names) != len(set(names)):
        raise PublicationReceiptError(
            "GITHUB_ASSET_INVENTORY_ORDER",
            "GitHub release assets must be unique and sorted by name",
        )
    if len(identifiers) != len(set(identifiers)):
        raise PublicationReceiptError(
            "GITHUB_ASSET_ID_DUPLICATE", "GitHub release asset IDs must be unique"
        )
    expected_names = sorted(REQUIRED_GITHUB_RELEASE_ASSETS.values())
    if names != expected_names or len(assets) != len(expected_names):
        raise PublicationReceiptError(
            "GITHUB_ASSET_SET_MISMATCH",
            "GitHub draft Release must contain exactly the fixed wheel, SBOM, and checksums",
        )
    optional_names = release["optional_asset_names"]
    if optional_names != []:
        raise PublicationReceiptError(
            "GITHUB_OPTIONAL_ASSET_SET_MISMATCH",
            "GitHub draft Release cannot contain optional publication assets",
        )
    role_names: dict[str, str] = {}
    for asset in assets:
        role = asset["role"]
        if role in role_names:
            raise PublicationReceiptError(
                "GITHUB_REQUIRED_ASSET_ROLE_DUPLICATE",
                f"GitHub release repeats required asset role {role!r}",
            )
        else:
            role_names[role] = asset["name"]
        if asset["github_digest_sha256"] != asset["downloaded_sha256"]:
            raise PublicationReceiptError(
                "GITHUB_ASSET_DIGEST_MISMATCH",
                f"GitHub and downloaded digests differ for asset {asset['name']!r}",
            )
    if role_names != REQUIRED_GITHUB_RELEASE_ASSETS:
        raise PublicationReceiptError(
            "GITHUB_REQUIRED_ASSET_SET_MISMATCH",
            "GitHub draft Release lacks the fixed wheel, SBOM, or checksums asset",
        )
    checksums = release["checksums"]
    assets_by_name = {asset["name"]: asset for asset in assets}
    checksums_asset = assets_by_name[REQUIRED_GITHUB_RELEASE_ASSETS["checksums"]]
    if checksums["file_sha256"] != checksums_asset["downloaded_sha256"]:
        raise PublicationReceiptError(
            "GITHUB_CHECKSUMS_FILE_MISMATCH",
            "parsed checksums evidence does not bind the downloaded checksums.txt bytes",
        )
    expected_checksum_names = (
        REQUIRED_GITHUB_RELEASE_ASSETS["wheel"],
        REQUIRED_GITHUB_RELEASE_ASSETS["sbom"],
    )
    if tuple(entry["name"] for entry in checksums["entries"]) != expected_checksum_names:
        raise PublicationReceiptError(
            "GITHUB_CHECKSUMS_ENTRY_SET_MISMATCH",
            "parsed checksums evidence does not contain the fixed wheel and SBOM entries",
        )
    for entry in checksums["entries"]:
        if entry["sha256"] != assets_by_name[entry["name"]]["downloaded_sha256"]:
            raise PublicationReceiptError(
                "GITHUB_CHECKSUMS_DIGEST_MISMATCH",
                "parsed checksums entries do not match downloaded wheel and SBOM bytes",
            )

    collected_at = _parse_timestamp(
        document["collector"]["collected_at"], field="GitHub collection timestamp"
    )
    for value, field in (
        (tag["verification"]["verified_at"], "GitHub tag verification timestamp"),
        (hosted_ci["completed_at"], "GitHub hosted CI completion timestamp"),
        (release["created_at"], "GitHub draft Release creation timestamp"),
        (release["updated_at"], "GitHub draft Release update timestamp"),
    ):
        if _parse_timestamp(value, field=field) > collected_at:
            raise PublicationReceiptError(
                "GITHUB_EVIDENCE_TIME_ORDER",
                f"{field} is later than the remote collection timestamp",
            )
    if _parse_timestamp(
        release["updated_at"], field="GitHub draft Release update timestamp"
    ) < _parse_timestamp(
        release["created_at"], field="GitHub draft Release creation timestamp"
    ):
        raise PublicationReceiptError(
            "GITHUB_RELEASE_TIME_ORDER",
            "GitHub draft Release update predates its creation",
        )


def _validate_github_release_assets_relationships(
    github_document: Mapping[str, Any],
    assets_receipt_document: Mapping[str, Any],
) -> None:
    source_sha = github_document["source_commit_sha"]
    if (
        assets_receipt_document["source_commit"] != source_sha
        or assets_receipt_document["source"]["git_source_commit"] != source_sha
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_GITHUB_SOURCE_MISMATCH",
            "local GitHub Release assets verification targets a different source commit",
        )

    remote_assets = {
        asset["role"]: asset for asset in github_document["release"]["assets"]
    }
    local_assets = assets_receipt_document["assets"]
    for role, expected_name in REQUIRED_GITHUB_RELEASE_ASSETS.items():
        remote = remote_assets.get(role)
        local = local_assets[role]
        if remote is None or local["name"] != expected_name or remote["name"] != expected_name:
            raise PublicationReceiptError(
                "RELEASE_ASSETS_NAME_MISMATCH",
                f"local and remote GitHub Release asset names differ for role {role!r}",
            )
        if local["size_bytes"] != remote["size_bytes"]:
            raise PublicationReceiptError(
                "RELEASE_ASSETS_SIZE_MISMATCH",
                f"local and remote GitHub Release asset sizes differ for {expected_name!r}",
            )
        if local["file_sha256"] != remote["downloaded_sha256"]:
            raise PublicationReceiptError(
                "RELEASE_ASSETS_DIGEST_MISMATCH",
                f"local and downloaded GitHub Release bytes differ for {expected_name!r}",
            )

    remote_checksums = github_document["release"]["checksums"]
    local_checksums = assets_receipt_document["checksum_closure"]
    if (
        local_assets["checksums"]["file_sha256"]
        != remote_checksums["file_sha256"]
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_CHECKSUMS_FILE_MISMATCH",
            "local verification and remote evidence bind different checksums.txt bytes",
        )
    local_entries = local_checksums["ordered_entries"]
    remote_entries = [
        {"name": entry["name"], "file_sha256": entry["sha256"]}
        for entry in remote_checksums["entries"]
    ]
    if (
        local_checksums["line_count"] != remote_checksums["line_count"]
        or local_entries != remote_entries
    ):
        raise PublicationReceiptError(
            "RELEASE_ASSETS_CHECKSUMS_ENTRIES_MISMATCH",
            "local and remote checksums entries do not match item by item",
        )


def _validate_hugging_face_relationships(document: Mapping[str, Any]) -> None:
    repository = document["repository"]
    revision = document["revision"]
    inventory = document["inventory"]
    cross_reference = document["cross_reference"]

    if repository["id"] != EXPECTED_HUGGING_FACE_REPOSITORY:
        raise PublicationReceiptError(
            "HF_REPOSITORY_MISMATCH", "Hugging Face evidence targets a different repository"
        )
    if revision["resolved_sha"] != revision["immutable_sha"]:
        raise PublicationReceiptError(
            "HF_IMMUTABLE_REVISION_MISMATCH",
            "requested Hugging Face revision did not resolve to the recorded immutable SHA",
        )

    paths = [item["path"] for item in inventory]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PublicationReceiptError(
            "HF_INVENTORY_ORDER", "Hugging Face inventory must be unique and sorted by path"
        )
    for path in paths:
        _validate_repo_path(path, field="Hugging Face inventory path")
    missing = sorted(set(REQUIRED_HUGGING_FACE_FILES) - set(paths))
    if missing:
        raise PublicationReceiptError(
            "HF_REQUIRED_FILES_MISSING",
            f"Hugging Face remote inventory lacks required file {missing[0]!r}",
        )
    private_security_evidence = sorted(
        set(paths) & PRIVATE_SECURITY_CHANNEL_EVIDENCE_FILES
    )
    if len(private_security_evidence) != 1:
        raise PublicationReceiptError(
            "HF_PRIVATE_SECURITY_CHANNEL_EVIDENCE_CARDINALITY",
            "Hugging Face remote inventory must contain exactly one tested-channel or "
            "maintainer-waiver security evidence receipt",
        )

    cross_path = _validate_repo_path(
        cross_reference["path"], field="Hugging Face cross-reference path"
    )
    by_path = {item["path"]: item for item in inventory}
    if cross_path not in by_path:
        raise PublicationReceiptError(
            "HF_CROSS_REFERENCE_FILE_MISSING",
            "Hugging Face GitHub cross-reference file is absent from the remote inventory",
        )
    if by_path[cross_path]["downloaded_sha256"] != cross_reference["file_sha256"]:
        raise PublicationReceiptError(
            "HF_CROSS_REFERENCE_HASH_MISMATCH",
            "Hugging Face cross-reference does not bind its downloaded remote file",
        )
    if (
        cross_reference["github_repository"] != EXPECTED_GITHUB_REPOSITORY
        or cross_reference["github_source_sha"] == ""
    ):
        raise PublicationReceiptError(
            "HF_GITHUB_REFERENCE_MISMATCH",
            "Hugging Face model card does not declare the intended GitHub source",
        )


def _validate_cross_platform_relationships(
    github_document: Mapping[str, Any], hugging_face_document: Mapping[str, Any]
) -> None:
    github_repository = github_document["repository"]["id"]
    github_source_sha = github_document["source_commit_sha"]
    hugging_face_repository = hugging_face_document["repository"]["id"]
    github_cross = github_document["cross_reference"]
    hugging_face_cross = hugging_face_document["cross_reference"]
    if github_cross["hugging_face_repository"] != hugging_face_repository:
        raise PublicationReceiptError(
            "BIDIRECTIONAL_HF_REPOSITORY_MISMATCH",
            "GitHub-to-Hugging-Face repository binding is inconsistent",
        )
    if (
        hugging_face_cross["github_repository"] != github_repository
        or hugging_face_cross["github_source_sha"] != github_source_sha
    ):
        raise PublicationReceiptError(
            "BIDIRECTIONAL_GITHUB_SOURCE_MISMATCH",
            "Hugging-Face-to-GitHub source binding is inconsistent",
        )
    expected_revision_url = (
        f"https://huggingface.co/{hugging_face_repository}/tree/"
        f"{hugging_face_document['revision']['immutable_sha']}"
    )
    release_body = github_document["release"]["body"]
    if hugging_face_repository not in release_body or expected_revision_url not in release_body:
        raise PublicationReceiptError(
            "GITHUB_RELEASE_HF_REVISION_REFERENCE_MISMATCH",
            "GitHub draft Release body lacks the exact Hugging Face repository and immutable URL",
        )


def _validate_hugging_face_download_relationships(
    hugging_face_document: Mapping[str, Any], download_document: Mapping[str, Any]
) -> None:
    revision = hugging_face_document["revision"]
    immutable_sha = revision["immutable_sha"]
    if download_document["repository"] != hugging_face_document["repository"]["id"]:
        raise PublicationReceiptError(
            "HF_DOWNLOAD_REPOSITORY_MISMATCH",
            "Hugging Face download verification targets a different repository",
        )
    if (
        download_document["requested_revision"] != immutable_sha
        or download_document["resolved_commit"] != immutable_sha
    ):
        raise PublicationReceiptError(
            "HF_DOWNLOAD_IMMUTABLE_SHA_MISMATCH",
            "Hugging Face download verification does not bind the collected immutable SHA",
        )

    remote_inventory = {
        item["path"]: {
            "file_sha256": item["downloaded_sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in hugging_face_document["inventory"]
    }
    downloaded_inventory = download_document["local_root"]["files"]
    if downloaded_inventory != remote_inventory:
        raise PublicationReceiptError(
            "HF_DOWNLOAD_INVENTORY_MISMATCH",
            "download verification files do not exactly match the immutable remote inventory",
        )
    remote_snapshot = download_document["remote_snapshot"]
    remote_files = remote_snapshot["files"]
    if remote_snapshot["private"] is not True or remote_snapshot[
        "visibility"
    ] != hugging_face_document["repository"]["visibility"]:
        raise PublicationReceiptError(
            "HF_DOWNLOAD_VISIBILITY_MISMATCH",
            "download provenance does not prove the required private Hugging Face state",
        )
    if (
        remote_snapshot["file_count"] != len(remote_inventory)
        or remote_snapshot["metadata_coverage"]["size_count"] != len(remote_inventory)
        or remote_snapshot["metadata_coverage"]["content_verified_count"]
        != len(remote_inventory)
    ):
        raise PublicationReceiptError(
            "HF_DOWNLOAD_REMOTE_COVERAGE_INCOMPLETE",
            "download provenance lacks complete immutable remote metadata verification",
        )
    hf_by_path = {item["path"]: item for item in hugging_face_document["inventory"]}
    for path, binding in remote_files.items():
        remote_oid = binding["lfs_oid_sha256"] or binding["git_blob_oid"]
        if (
            binding["size_bytes"] != hf_by_path[path]["size_bytes"]
            or remote_oid != hf_by_path[path]["remote_oid"]
            or binding["content_verification"]
            == "immutable_revision_transport_only"
        ):
            raise PublicationReceiptError(
                "HF_DOWNLOAD_REMOTE_METADATA_MISMATCH",
                f"download provenance metadata differs for Hugging Face file {path!r}",
            )


def _validate_package_verification_relationships(
    github_document: Mapping[str, Any],
    hugging_face_document: Mapping[str, Any],
    download_payload: JsonPayload,
) -> None:
    verification = hugging_face_document["package_verification"]
    download = download_payload.document
    if (
        verification["hugging_face_commit"]
        != hugging_face_document["revision"]["immutable_sha"]
        or verification["github_source_commit"]
        != github_document["source_commit_sha"]
    ):
        raise PublicationReceiptError(
            "HF_PACKAGE_VERIFICATION_SOURCE_MISMATCH",
            "package verification does not bind the exact HF and GitHub commits",
        )
    if (
        verification["hf_download_provenance_file_sha256"]
        != download_payload.file_sha256
        or verification["hf_download_provenance_receipt_sha256"]
        != download["receipt_sha256"]
    ):
        raise PublicationReceiptError(
            "HF_PACKAGE_VERIFICATION_PROVENANCE_MISMATCH",
            "package verification does not bind the supplied HF download provenance",
        )

    identity = verification["package_identity"]
    files = download["local_root"]["files"]
    expected_direct_hashes = {
        "checksums_file_sha256": files["checksums.txt"]["file_sha256"],
        "model_file_sha256": files["model.safetensors"]["file_sha256"],
    }
    if any(identity[key] != value for key, value in expected_direct_hashes.items()) or identity[
        "verified_file_count"
    ] != len(files) - 1:
        raise PublicationReceiptError(
            "HF_PACKAGE_IDENTITY_MISMATCH",
            "package verification identity differs from the immutable downloaded package",
        )


def _evidence_binding(payload: JsonPayload) -> dict[str, Any]:
    document = payload.document
    collector = document["collector"]
    binding = {
        "schema_version": document["schema_version"],
        "file_sha256": payload.file_sha256,
        "evidence_sha256": document["evidence_sha256"],
        "collected_at": collector["collected_at"],
        "collector_name": collector["name"],
        "generator_path": collector["generator_path"],
        "generator_file_sha256": collector["generator_file_sha256"],
        "collection_run_sha256": collector["collection_run_sha256"],
    }
    if document["schema_version"] == GITHUB_EVIDENCE_SCHEMA_VERSION:
        verification = document["signed_tag"]["verification"]
        binding["signing_key_fingerprint"] = verification[
            "signing_key_fingerprint"
        ]
        binding["local_cryptographic_verification"] = verification[
            "local_cryptographic_verification"
        ]
    return binding


def _download_evidence_binding(payload: JsonPayload) -> dict[str, Any]:
    document = payload.document
    return {
        "schema_version": document["schema_version"],
        "file_sha256": payload.file_sha256,
        "receipt_sha256": document["receipt_sha256"],
        "downloaded_at": document["downloaded_at"],
        "generator_path": document["generator"]["path"],
        "generator_file_sha256": document["generator"]["file_sha256"],
        "inventory_sha256": document["local_root"]["inventory_sha256"],
    }


def _github_release_assets_evidence_binding(payload: JsonPayload) -> dict[str, Any]:
    document = payload.document
    implementation = document["implementation"]
    wheel_inventory = document["wheel_inventory"]
    smoke = document["isolated_wheel_smoke"]
    public_scan = document["public_artifact_scan"]
    return {
        "schema_version": document["schema_version"],
        "file_sha256": payload.file_sha256,
        "receipt_sha256": document["receipt_sha256"],
        "source_commit": document["source_commit"],
        "generator_path": implementation["generator_path"],
        "generator_file_sha256": implementation["generator_file_sha256"],
        "schema_path": implementation["schema_path"],
        "schema_file_sha256": implementation["schema_file_sha256"],
        "verified_asset_count": document["release"]["asset_count"],
        "verified_asset_inventory_sha256": canonical_json_hash(document["assets"]),
        "checksums_file_sha256": document["assets"]["checksums"]["file_sha256"],
        "checksum_closure_status": document["checksum_closure"]["status"],
        "wheel_inventory_status": wheel_inventory["status"],
        "wheel_member_count": wheel_inventory["member_count"],
        "wheel_member_inventory_sha256": wheel_inventory[
            "member_inventory_sha256"
        ],
        "isolated_wheel_smoke_status": smoke["status"],
        "isolated_wheel_smoke_profile_version": smoke["profile_version"],
        "public_artifact_scan_status": public_scan["status"],
        "public_artifact_scan_finding_count": public_scan["finding_count"],
    }


def _release_assets(github_document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": asset["asset_id"],
            "role": asset["role"],
            "name": asset["name"],
            "size_bytes": asset["size_bytes"],
            "sha256": asset["downloaded_sha256"],
        }
        for asset in github_document["release"]["assets"]
    ]


def _hugging_face_inventory(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "remote_oid": item["remote_oid"],
            "sha256": item["downloaded_sha256"],
        }
        for item in document["inventory"]
    ]


def build_publication_receipt(
    *,
    github_evidence: JsonPayload,
    github_release_assets_verification: JsonPayload,
    hugging_face_evidence: JsonPayload,
    hugging_face_download_verification: JsonPayload,
    recorded_at: str,
) -> dict[str, Any]:
    """Derive a pre-publication gate receipt from four validated evidence files."""

    _validate_remote_evidence_payload(github_evidence, platform="github")
    _validate_github_release_assets_verification_payload(
        github_release_assets_verification
    )
    _validate_remote_evidence_payload(hugging_face_evidence, platform="hugging_face")
    _validate_hugging_face_download_payload(hugging_face_download_verification)
    github = github_evidence.document
    hugging_face = hugging_face_evidence.document
    hf_download = hugging_face_download_verification.document
    github_release_assets = github_release_assets_verification.document
    _validate_github_relationships(github)
    _validate_github_release_assets_relationships(github, github_release_assets)
    _validate_hugging_face_relationships(hugging_face)
    _validate_cross_platform_relationships(github, hugging_face)
    _validate_hugging_face_download_relationships(hugging_face, hf_download)
    _validate_package_verification_relationships(
        github, hugging_face, hugging_face_download_verification
    )

    recorded = _parse_timestamp(recorded_at, field="pre-publication gate receipt timestamp")
    for platform, document in (("GitHub", github), ("Hugging Face", hugging_face)):
        collected = _parse_timestamp(
            document["collector"]["collected_at"],
            field=f"{platform} collection timestamp",
        )
        if recorded < collected:
            raise PublicationReceiptError(
                "RECEIPT_TIME_ORDER",
                f"gate receipt timestamp predates {platform} evidence collection",
            )
    if recorded < _parse_timestamp(
        hf_download["downloaded_at"], field="Hugging Face download timestamp"
    ):
        raise PublicationReceiptError(
            "RECEIPT_TIME_ORDER",
            "gate receipt timestamp predates Hugging Face download verification",
        )

    source_sha = github["source_commit_sha"]
    tag = github["signed_tag"]
    hosted_ci = github["hosted_ci"]
    release = github["release"]
    hf_revision = hugging_face["revision"]
    github_cross = github["cross_reference"]
    hf_cross = hugging_face["cross_reference"]
    assets = _release_assets(github)
    inventory = _hugging_face_inventory(hugging_face)
    private_security_evidence_path = next(
        item["path"]
        for item in inventory
        if item["path"] in PRIVATE_SECURITY_CHANNEL_EVIDENCE_FILES
    )
    hf_revision_url = (
        f"https://huggingface.co/{hugging_face['repository']['id']}/tree/"
        f"{hf_revision['immutable_sha']}"
    )

    receipt: dict[str, Any] = {
        "schema_version": "pii-zh.community-v2-pre-publication-gate-receipt.v1",
        "receipt_type": "community_v2_pre_publication_gate",
        "package_version": PACKAGE_VERSION,
        "recorded_at": recorded_at,
        "status": "READY_FOR_FINAL_PUBLICATION_CONFIRMATION",
        "stage": "pre_publication_gate",
        "final_publication_confirmed": False,
        "producer": PRODUCER,
        "evidence_bindings": {
            "github": _evidence_binding(github_evidence),
            "github_release_assets_verification": (
                _github_release_assets_evidence_binding(
                    github_release_assets_verification
                )
            ),
            "hugging_face": _evidence_binding(hugging_face_evidence),
            "hugging_face_download_verification": _download_evidence_binding(
                hugging_face_download_verification
            ),
        },
        "github": {
            "repository": github["repository"]["id"],
            "visibility": github["repository"]["visibility"],
            "source_commit_sha": source_sha,
            "signed_tag": tag,
            "hosted_ci": hosted_ci,
            "release": {
                "release_id": release["release_id"],
                "tag_name": release["tag_name"],
                "resolved_source_sha": release["resolved_source_sha"],
                "created_at": release["created_at"],
                "updated_at": release["updated_at"],
                "body_sha256": release["body_sha256"],
                "hugging_face_revision_url": hf_revision_url,
                "draft": release["draft"],
                "prerelease": release["prerelease"],
                "optional_asset_names": release["optional_asset_names"],
                "checksums": release["checksums"],
            },
        },
        "hugging_face": {
            "repository": hugging_face["repository"]["id"],
            "visibility": hugging_face["repository"]["visibility"],
            "immutable_sha": hf_revision["immutable_sha"],
            "requested_revision": hf_revision["requested_revision"],
            "inventory": inventory,
            "inventory_sha256": canonical_json_hash(inventory),
            "required_files": [
                *REQUIRED_HUGGING_FACE_FILES,
                private_security_evidence_path,
            ],
            "download_verification": {
                "repository": hf_download["repository"],
                "immutable_sha": hf_download["resolved_commit"],
                "receipt_sha256": hf_download["receipt_sha256"],
                "evidence_file_sha256": hugging_face_download_verification.file_sha256,
                "downloaded_at": hf_download["downloaded_at"],
                "inventory_sha256": hf_download["local_root"]["inventory_sha256"],
            },
            "package_verification": hugging_face["package_verification"],
        },
        "required_release_assets": REQUIRED_GITHUB_RELEASE_ASSETS,
        "optional_release_asset_names": release["optional_asset_names"],
        "release_assets": assets,
        "release_asset_inventory_sha256": canonical_json_hash(assets),
        "draft_release_attachment": {
            "asset_name": DRAFT_RECEIPT_ASSET_NAME,
            "attach_after_receipt_generation": True,
            "excluded_from_bound_release_asset_inventory": True,
            "release_must_remain_draft": True,
        },
        "remote_reference_bindings": {
            "github_source_to_hugging_face_repository": {
                "reference_source": "github_source",
                "github_repository": github["repository"]["id"],
                "github_source_sha": source_sha,
                "evidence_path": github_cross["path"],
                "evidence_file_sha256": github_cross["file_sha256"],
                "hugging_face_repository": hugging_face["repository"]["id"],
            },
            "github_draft_release_to_hugging_face_revision": {
                "reference_source": "github_draft_release_body",
                "github_repository": github["repository"]["id"],
                "github_release_id": release["release_id"],
                "release_body_sha256": release["body_sha256"],
                "hugging_face_repository": hugging_face["repository"]["id"],
                "hugging_face_immutable_sha": hf_revision["immutable_sha"],
                "hugging_face_revision_url": hf_revision_url,
            },
            "hugging_face_model_card_to_github_source": {
                "reference_source": "hugging_face_model_card",
                "hugging_face_repository": hugging_face["repository"]["id"],
                "hugging_face_immutable_sha": hf_revision["immutable_sha"],
                "evidence_path": hf_cross["path"],
                "evidence_file_sha256": hf_cross["file_sha256"],
                "github_repository": github["repository"]["id"],
                "github_source_sha": source_sha,
                "github_signed_tag": tag["name"],
            },
            "receipt_staging_binding": {
                "binding_source": "pre_publication_gate_receipt",
                "github_source_sha": source_sha,
                "hugging_face_repository": hugging_face["repository"]["id"],
                "hugging_face_immutable_sha": hf_revision["immutable_sha"],
                "hugging_face_download_verification_receipt_sha256": hf_download[
                    "receipt_sha256"
                ],
            },
        },
        "verification": VERIFICATION,
        "limitations": LIMITATIONS,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, remove="receipt_sha256")
    _validate_schema(receipt, field="pre-publication gate receipt")
    return receipt


def validate_publication_receipt(
    document: Mapping[str, Any],
    *,
    github_evidence: JsonPayload,
    github_release_assets_verification: JsonPayload,
    hugging_face_evidence: JsonPayload,
    hugging_face_download_verification: JsonPayload,
) -> None:
    """Validate schema/self-hash and replay derivation from the exact evidence files."""

    _validate_schema(document, field="pre-publication gate receipt")
    _validate_self_hash(document, field="pre-publication gate receipt", key="receipt_sha256")
    expected = build_publication_receipt(
        github_evidence=github_evidence,
        github_release_assets_verification=github_release_assets_verification,
        hugging_face_evidence=hugging_face_evidence,
        hugging_face_download_verification=hugging_face_download_verification,
        recorded_at=document["recorded_at"],
    )
    if document != expected:
        raise PublicationReceiptError(
            "RECEIPT_DERIVATION_MISMATCH",
            "gate receipt is not the exact derivation of the supplied evidence files",
        )


def _write_new_file(path: Path, content: bytes) -> str:
    """Publish complete mode-0444 bytes using an atomic no-clobber hard link."""

    path = path.expanduser()
    if os.path.lexists(path):
        raise PublicationReceiptError("OUTPUT_EXISTS", "refusing to overwrite receipt output")
    try:
        parent = os.lstat(path.parent)
    except OSError as exc:
        raise PublicationReceiptError(
            "OUTPUT_PARENT_MISSING", "receipt output parent directory does not exist"
        ) from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise PublicationReceiptError(
            "UNSAFE_OUTPUT_PARENT", "receipt output parent must be a real directory"
        )

    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PublicationReceiptError(
                "OUTPUT_EXISTS", "refusing to overwrite receipt output"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except PublicationReceiptError:
        raise
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise PublicationReceiptError(
                "OUTPUT_EXISTS", "refusing to overwrite receipt output"
            ) from exc
        raise PublicationReceiptError(
            "OUTPUT_WRITE_FAILED", "could not publish receipt output"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    output = _read_regular_json(path, field="written publication receipt")
    if output.payload != content or output.mode != 0o444:
        raise PublicationReceiptError(
            "OUTPUT_VERIFICATION_FAILED", "written receipt bytes or mode failed verification"
        )
    return output.file_sha256


def write_publication_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    """Validate and publish one immutable no-clobber receipt file."""

    _validate_schema(receipt, field="publication receipt")
    _validate_self_hash(receipt, field="publication receipt", key="receipt_sha256")
    return _write_new_file(path, _pretty_json_bytes(receipt))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline producer/validator for the community-v2 pre-publication gate receipt. "
            "The caller must provide evidence from separate GitHub/Hugging Face collectors "
            "plus the reviewed local GitHub Release asset verifier and immutable HF download "
            "materializer; this command never queries or modifies either service."
        )
    )
    parser.add_argument("--github-evidence", required=True, type=Path)
    parser.add_argument(
        "--github-release-assets-verification-receipt", required=True, type=Path
    )
    parser.add_argument("--hugging-face-evidence", required=True, type=Path)
    parser.add_argument(
        "--hugging-face-download-verification-receipt", required=True, type=Path
    )
    parser.add_argument("--recorded-at")
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", type=Path, metavar="RECEIPT")
    return parser


def _validate_cli_shape(args: argparse.Namespace) -> str:
    if args.validate_only is not None:
        if args.recorded_at is not None or args.output is not None:
            raise PublicationReceiptError(
                "INVALID_CLI_MODE",
                "validate-only does not accept --recorded-at or --output",
            )
        return "validate-only"
    if args.recorded_at is None:
        raise PublicationReceiptError(
            "RECORDED_AT_REQUIRED", "build and dry-run require --recorded-at"
        )
    if args.dry_run:
        if args.output is not None:
            raise PublicationReceiptError(
                "INVALID_CLI_MODE", "dry-run refuses an output path"
            )
        return "dry-run"
    if args.output is None:
        raise PublicationReceiptError("OUTPUT_REQUIRED", "build requires --output")
    return "build"


def _summary(
    *, mode: str, receipt: Mapping[str, Any], file_sha256: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": mode,
        "result": (
            "VALIDATED_AND_WRITTEN"
            if mode == "build"
            else "VALIDATED_WITHOUT_WRITING"
        ),
        "package_version": PACKAGE_VERSION,
        "status": receipt["status"],
        "final_publication_confirmed": receipt["final_publication_confirmed"],
        "receipt_sha256": receipt["receipt_sha256"],
        "github_source_sha": receipt["github"]["source_commit_sha"],
        "hugging_face_immutable_sha": receipt["hugging_face"]["immutable_sha"],
        "release_asset_count": len(receipt["release_assets"]),
        "github_release_assets_verification_receipt_sha256": receipt[
            "evidence_bindings"
        ]["github_release_assets_verification"]["receipt_sha256"],
        "hugging_face_file_count": len(receipt["hugging_face"]["inventory"]),
        "local_write_performed": mode == "build",
        "remote_access_performed": False,
        "remote_mutation_performed": False,
        "remote_collector_still_required": True,
    }
    if file_sha256 is not None:
        result["receipt_file_sha256"] = file_sha256
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    mode = "preflight"
    try:
        mode = _validate_cli_shape(args)
        github = load_remote_evidence(args.github_evidence, platform="github")
        github_release_assets = load_github_release_assets_verification(
            args.github_release_assets_verification_receipt
        )
        hugging_face = load_remote_evidence(
            args.hugging_face_evidence, platform="hugging_face"
        )
        hf_download = load_hugging_face_download_verification(
            args.hugging_face_download_verification_receipt
        )
        if mode == "validate-only":
            receipt_payload = _read_regular_json(
                args.validate_only, field="publication receipt"
            )
            if receipt_payload.mode != 0o444:
                raise PublicationReceiptError(
                    "RECEIPT_MODE_MISMATCH", "publication receipt must have mode 0444"
                )
            validate_publication_receipt(
                receipt_payload.document,
                github_evidence=github,
                github_release_assets_verification=github_release_assets,
                hugging_face_evidence=hugging_face,
                hugging_face_download_verification=hf_download,
            )
            receipt = receipt_payload.document
            file_sha256 = receipt_payload.file_sha256
        else:
            receipt = build_publication_receipt(
                github_evidence=github,
                github_release_assets_verification=github_release_assets,
                hugging_face_evidence=hugging_face,
                hugging_face_download_verification=hf_download,
                recorded_at=args.recorded_at,
            )
            validate_publication_receipt(
                receipt,
                github_evidence=github,
                github_release_assets_verification=github_release_assets,
                hugging_face_evidence=hugging_face,
                hugging_face_download_verification=hf_download,
            )
            file_sha256 = (
                write_publication_receipt(args.output, receipt) if mode == "build" else None
            )
        print(
            json.dumps(
                _summary(mode=mode, receipt=receipt, file_sha256=file_sha256),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except PublicationReceiptError as exc:
        print(
            json.dumps(
                {
                    "mode": mode,
                    "result": "BLOCKED",
                    "blocker_ids": [exc.blocker_id],
                    "local_write_performed": False,
                    "remote_access_performed": False,
                    "remote_mutation_performed": False,
                    "remote_collector_still_required": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
