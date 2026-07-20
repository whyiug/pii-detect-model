#!/usr/bin/env python3
"""Recheck the draft release after attaching its pre-publication receipt.

This verifier is intentionally read-only.  It reads the three local evidence
documents, queries only the official GitHub and Hugging Face endpoints, and
downloads only the small pre-publication receipt attachment.  It never
downloads Hugging Face model weights and never mutates either remote.

The command does not read ``.env`` files.  The caller must place a private-model
read token in the process ``HF_TOKEN`` environment variable.  Token values,
authorization headers, raw API errors, release bodies, and downloaded receipt
bytes are never printed or written to the output receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from jsonschema import Draft202012Validator, FormatChecker

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - the release environment supplies it
    HfApi = None  # type: ignore[assignment,misc]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(
    "configs/release/community_v2_post_attachment_verification.schema.json"
)
VERIFIER_PATH = Path("scripts/verify_community_v2_post_attachment.py")
VERIFIER_VERSION = "1.0.0"

OFFICIAL_GITHUB_ENDPOINT = "https://api.github.com"
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
GITHUB_REPOSITORY = "whyiug/pii-detect-model"
HUGGING_FACE_REPOSITORY = "Forrest20231206/pii-zh-qwen3-0.6b-24class"
RELEASE_TAG = "v0.2.0rc1"
PACKAGE_VERSION = "0.2.0rc1"
PREPUBLICATION_RECEIPT_ASSET = "community-v2-pre-publication-gate-receipt.json"

GITHUB_EVIDENCE_SCHEMA_VERSION = (
    "pii-zh.community-v2-github-publication-evidence.v1"
)
HUGGING_FACE_EVIDENCE_SCHEMA_VERSION = (
    "pii-zh.community-v2-hugging-face-publication-evidence.v1"
)
PREPUBLICATION_RECEIPT_SCHEMA_VERSION = (
    "pii-zh.community-v2-pre-publication-gate-receipt.v1"
)
OUTPUT_SCHEMA_VERSION = "pii-zh.community-v2-post-attachment-verification.v1"

REQUIRED_RELEASE_ASSETS = {
    "checksums": "checksums.txt",
    "sbom": "sbom.cdx.json",
    "wheel": "pii_zh_qwen-0.2.0rc1-py3-none-any.whl",
}
EXPECTED_REMOTE_ASSET_NAMES = frozenset(
    [*REQUIRED_RELEASE_ASSETS.values(), PREPUBLICATION_RECEIPT_ASSET]
)

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_DOWNLOAD_BYTES = 16 * 1024 * 1024
READ_BLOCK_BYTES = 1024 * 1024

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GITHUB_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")


class PostAttachmentVerificationError(RuntimeError):
    """A fail-closed error with a stable, non-sensitive blocker identifier."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class JsonPayload:
    """Stable bytes and parsed JSON from one local regular file."""

    payload: bytes
    document: Mapping[str, Any]
    file_sha256: str
    mode: int


class GitHubTransport(Protocol):
    """The two read-only GitHub operations used by the verifier."""

    def get_json(self, endpoint: str) -> Any: ...

    def download_bytes(self, endpoint: str, *, maximum: int) -> bytes: ...


class HuggingFaceTransport(Protocol):
    """The metadata-only Hugging Face operations used by the verifier."""

    def repo_info(self, *, revision: str) -> object: ...

    def list_repo_files(self, *, revision: str) -> list[str]: ...


def _canonical_json_bytes(value: Any, *, remove: str | None = None) -> bytes:
    document = dict(value) if isinstance(value, Mapping) else value
    if remove is not None:
        if not isinstance(document, dict):
            raise PostAttachmentVerificationError(
                "INVALID_CANONICAL_JSON",
                "self-hash removal requires a JSON object",
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
        raise PostAttachmentVerificationError(
            "INVALID_CANONICAL_JSON",
            "document is not canonical-JSON serializable",
        ) from exc


def canonical_json_hash(value: Any, *, remove: str | None = None) -> str:
    """Return a lowercase SHA-256 over canonical UTF-8 JSON."""

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
        raise PostAttachmentVerificationError(
            "INVALID_OUTPUT_JSON", "output receipt is not strict JSON"
        ) from exc


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostAttachmentVerificationError(
                    "DUPLICATE_JSON_KEY", f"{field} repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise PostAttachmentVerificationError(
            "NONFINITE_JSON_NUMBER", f"{field} contains a non-finite JSON number"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostAttachmentVerificationError(
            "INVALID_JSON", f"{field} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PostAttachmentVerificationError(
            "INVALID_JSON_SHAPE", f"{field} must be a JSON object"
        )
    return value


def _file_state(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular(path: Path, *, field: str, maximum: int) -> tuple[bytes, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PostAttachmentVerificationError(
            "UNSAFE_INPUT_FILE", f"{field} is unavailable or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise PostAttachmentVerificationError(
                "UNSAFE_INPUT_FILE", f"{field} must be a bounded regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(READ_BLOCK_BYTES, maximum + 1 - total))
            if not block:
                break
            total += len(block)
            if total > maximum:
                raise PostAttachmentVerificationError(
                    "INPUT_TOO_LARGE", f"{field} exceeds its size limit"
                )
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            _file_state(before) != _file_state(after)
            or total != before.st_size
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise PostAttachmentVerificationError(
                "INPUT_CHANGED_DURING_READ", f"{field} changed while it was read"
            )
        return b"".join(chunks), stat.S_IMODE(after.st_mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_json_payload(path: Path, *, field: str) -> JsonPayload:
    """Read one immutable, regular, strict-JSON evidence file."""

    payload, mode = _read_regular(path, field=field, maximum=MAX_JSON_BYTES)
    if mode != 0o444:
        raise PostAttachmentVerificationError(
            "INPUT_MODE_INVALID", f"{field} must be immutable mode 0444"
        )
    return JsonPayload(
        payload=payload,
        document=_strict_json(payload, field=field),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        mode=mode,
    )


def _validate_payload_integrity(payload: JsonPayload, *, field: str) -> None:
    if payload.mode != 0o444:
        raise PostAttachmentVerificationError(
            "INPUT_MODE_INVALID", f"{field} must be immutable mode 0444"
        )
    if hashlib.sha256(payload.payload).hexdigest() != payload.file_sha256:
        raise PostAttachmentVerificationError(
            "INPUT_FILE_HASH_MISMATCH", f"{field} byte binding does not verify"
        )
    if _strict_json(payload.payload, field=field) != payload.document:
        raise PostAttachmentVerificationError(
            "INPUT_DOCUMENT_MISMATCH", f"{field} document differs from its source bytes"
        )


def _sha256_regular_file(path: Path, *, field: str) -> str:
    payload, _mode = _read_regular(path, field=field, maximum=MAX_JSON_BYTES)
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostAttachmentVerificationError(
            "INPUT_FIELD_INVALID", f"{field} must be an object"
        )
    return value


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PostAttachmentVerificationError(
            "INPUT_FIELD_INVALID", f"{field} must be an array"
        )
    return value


def _string(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise PostAttachmentVerificationError(
            "INPUT_FIELD_INVALID", f"{field} must be non-empty text"
        )
    if pattern is not None and pattern.fullmatch(value) is None:
        raise PostAttachmentVerificationError(
            "INPUT_FIELD_INVALID", f"{field} is malformed"
        )
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PostAttachmentVerificationError(
            "INPUT_FIELD_INVALID", f"{field} must be an integer >= {minimum}"
        )
    return value


def _timestamp(value: object, *, field: str) -> str:
    text = _string(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PostAttachmentVerificationError(
            "INVALID_TIMESTAMP", f"{field} is not a valid timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise PostAttachmentVerificationError(
            "INVALID_TIMESTAMP", f"{field} has no timezone"
        )
    return text


def _normalized_repo_path(value: object, *, field: str) -> str:
    text = _string(value, field=field)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PostAttachmentVerificationError(
            "UNSAFE_REPOSITORY_PATH", f"{field} is not a normalized relative path"
        )
    return text


def _validate_self_hash(
    document: Mapping[str, Any], *, key: str, field: str
) -> str:
    claimed = _string(document.get(key), field=f"{field}.{key}", pattern=SHA256_RE)
    if claimed != canonical_json_hash(document, remove=key):
        raise PostAttachmentVerificationError(
            "SELF_HASH_MISMATCH", f"{field} canonical self-hash does not verify"
        )
    return claimed


def _validate_github_evidence(payload: JsonPayload) -> Mapping[str, Any]:
    document = payload.document
    if document.get("schema_version") != GITHUB_EVIDENCE_SCHEMA_VERSION:
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_SCHEMA_MISMATCH",
            "GitHub evidence has an unexpected schema version",
        )
    _validate_self_hash(document, key="evidence_sha256", field="GitHub evidence")
    repository = _mapping(document.get("repository"), field="GitHub repository")
    if repository != {"id": GITHUB_REPOSITORY, "visibility": "public"}:
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_REPOSITORY_MISMATCH",
            "GitHub evidence targets another repository state",
        )
    source_sha = _string(
        document.get("source_commit_sha"), field="GitHub source SHA", pattern=SHA1_RE
    )
    tag = _mapping(document.get("signed_tag"), field="GitHub signed tag")
    tag_object_sha = _string(
        tag.get("tag_object_sha"), field="GitHub tag object SHA", pattern=SHA1_RE
    )
    if (
        tag.get("name") != RELEASE_TAG
        or tag.get("ref_target_type") != "tag"
        or tag.get("ref_target_sha") != tag_object_sha
        or tag.get("tag_target_type") != "commit"
        or tag.get("tag_target_sha") != source_sha
    ):
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_TAG_MISMATCH",
            "GitHub evidence does not bind the annotated tag to the source",
        )
    release = _mapping(document.get("release"), field="GitHub release evidence")
    body = _string(release.get("body"), field="GitHub release body")
    body_sha256 = _string(
        release.get("body_sha256"), field="GitHub release body SHA", pattern=SHA256_RE
    )
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != body_sha256:
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_BODY_HASH_MISMATCH",
            "GitHub evidence release body hash does not verify",
        )
    if (
        release.get("tag_name") != RELEASE_TAG
        or release.get("resolved_source_sha") != source_sha
        or release.get("draft") is not True
        or release.get("prerelease") is not True
        or release.get("optional_asset_names") != []
    ):
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_RELEASE_MISMATCH",
            "GitHub evidence is not the expected draft prerelease",
        )
    _integer(release.get("release_id"), field="GitHub release ID", minimum=1)
    assets = _list(release.get("assets"), field="GitHub evidence assets")
    names = [
        _string(
            _mapping(item, field="GitHub asset").get("name"), field="asset name"
        )
        for item in assets
    ]
    if names != sorted(names) or set(names) != set(REQUIRED_RELEASE_ASSETS.values()):
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_ASSET_SET_MISMATCH",
            "GitHub evidence must contain the exact three pre-attachment assets",
        )
    ids: set[int] = set()
    roles: dict[str, str] = {}
    for value in assets:
        asset = _mapping(value, field="GitHub evidence asset")
        asset_id = _integer(asset.get("asset_id"), field="GitHub asset ID", minimum=1)
        if asset_id in ids:
            raise PostAttachmentVerificationError(
                "GITHUB_EVIDENCE_ASSET_ID_DUPLICATE",
                "GitHub evidence repeats an asset ID",
            )
        ids.add(asset_id)
        role = _string(asset.get("role"), field="GitHub asset role")
        name = _string(asset.get("name"), field="GitHub asset name")
        roles[role] = name
        _integer(asset.get("size_bytes"), field="GitHub asset size")
        remote_digest = _string(
            asset.get("github_digest_sha256"),
            field="GitHub asset digest",
            pattern=SHA256_RE,
        )
        downloaded_digest = _string(
            asset.get("downloaded_sha256"),
            field="downloaded GitHub asset digest",
            pattern=SHA256_RE,
        )
        if asset.get("state") != "uploaded" or remote_digest != downloaded_digest:
            raise PostAttachmentVerificationError(
                "GITHUB_EVIDENCE_ASSET_DIGEST_MISMATCH",
                "GitHub evidence asset digest does not verify",
            )
    if roles != REQUIRED_RELEASE_ASSETS:
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_ASSET_ROLE_MISMATCH",
            "GitHub evidence asset roles are not the fixed release roles",
        )
    assets_by_name = {
        _mapping(value, field="GitHub evidence asset")["name"]: _mapping(
            value, field="GitHub evidence asset"
        )
        for value in assets
    }
    checksums = _mapping(
        release.get("checksums"), field="GitHub parsed checksums evidence"
    )
    entries = _list(
        checksums.get("entries"), field="GitHub parsed checksums entries"
    )
    expected_checksum_names = [
        REQUIRED_RELEASE_ASSETS["wheel"],
        REQUIRED_RELEASE_ASSETS["sbom"],
    ]
    if (
        checksums.get("asset_name") != REQUIRED_RELEASE_ASSETS["checksums"]
        or checksums.get("format") != "sha256sum_text_two_space_two_lines_lf"
        or checksums.get("line_count") != 2
        or checksums.get("matched_downloaded_assets") is not True
        or checksums.get("file_sha256")
        != assets_by_name[REQUIRED_RELEASE_ASSETS["checksums"]]["downloaded_sha256"]
        or len(entries) != 2
    ):
        raise PostAttachmentVerificationError(
            "GITHUB_EVIDENCE_CHECKSUMS_MISMATCH",
            "GitHub parsed checksums evidence is incomplete or unbound",
        )
    for entry, expected_name in zip(entries, expected_checksum_names, strict=True):
        binding = _mapping(entry, field="GitHub parsed checksum entry")
        if (
            binding.get("name") != expected_name
            or binding.get("sha256")
            != assets_by_name[expected_name]["downloaded_sha256"]
        ):
            raise PostAttachmentVerificationError(
                "GITHUB_EVIDENCE_CHECKSUMS_MISMATCH",
                "GitHub parsed checksums entries differ from downloaded asset evidence",
            )
    return document


def _evidence_hf_inventory(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = _list(document.get("inventory"), field="Hugging Face evidence inventory")
    inventory: list[dict[str, Any]] = []
    for value in values:
        item = _mapping(value, field="Hugging Face evidence file")
        path = _normalized_repo_path(item.get("path"), field="Hugging Face file path")
        oid = _string(item.get("remote_oid"), field="Hugging Face remote OID")
        if SHA1_RE.fullmatch(oid) is None and SHA256_RE.fullmatch(oid) is None:
            raise PostAttachmentVerificationError(
                "HF_EVIDENCE_OID_INVALID", "Hugging Face evidence OID is malformed"
            )
        if item.get("kind") != "file":
            raise PostAttachmentVerificationError(
                "HF_EVIDENCE_KIND_INVALID", "Hugging Face inventory contains a non-file"
            )
        inventory.append(
            {
                "path": path,
                "size_bytes": _integer(
                    item.get("size_bytes"), field="Hugging Face file size"
                ),
                "remote_oid": oid,
                "downloaded_sha256": _string(
                    item.get("downloaded_sha256"),
                    field="Hugging Face downloaded SHA",
                    pattern=SHA256_RE,
                ),
            }
        )
    paths = [item["path"] for item in inventory]
    if len(inventory) < 10 or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PostAttachmentVerificationError(
            "HF_EVIDENCE_INVENTORY_INVALID",
            "Hugging Face evidence inventory must be sorted, unique, and complete",
        )
    return inventory


def _validate_hugging_face_evidence(payload: JsonPayload) -> Mapping[str, Any]:
    document = payload.document
    if document.get("schema_version") != HUGGING_FACE_EVIDENCE_SCHEMA_VERSION:
        raise PostAttachmentVerificationError(
            "HF_EVIDENCE_SCHEMA_MISMATCH",
            "Hugging Face evidence has an unexpected schema version",
        )
    _validate_self_hash(document, key="evidence_sha256", field="Hugging Face evidence")
    repository = _mapping(document.get("repository"), field="Hugging Face repository")
    if repository != {"id": HUGGING_FACE_REPOSITORY, "visibility": "private"}:
        raise PostAttachmentVerificationError(
            "HF_EVIDENCE_REPOSITORY_MISMATCH",
            "Hugging Face evidence targets another repository state",
        )
    revision = _mapping(document.get("revision"), field="Hugging Face revision")
    immutable_sha = _string(
        revision.get("immutable_sha"), field="Hugging Face immutable SHA", pattern=SHA1_RE
    )
    if revision.get("resolved_sha") != immutable_sha:
        raise PostAttachmentVerificationError(
            "HF_EVIDENCE_REVISION_MISMATCH",
            "Hugging Face evidence revision is not immutable",
        )
    _evidence_hf_inventory(document)
    cross_reference = _mapping(
        document.get("cross_reference"), field="Hugging Face cross-reference"
    )
    if cross_reference.get("github_repository") != GITHUB_REPOSITORY:
        raise PostAttachmentVerificationError(
            "HF_EVIDENCE_SOURCE_REPOSITORY_MISMATCH",
            "Hugging Face evidence references another GitHub repository",
        )
    _string(
        cross_reference.get("github_source_sha"),
        field="Hugging Face GitHub source SHA",
        pattern=SHA1_RE,
    )
    return document


def _validate_prepublication_receipt(
    payload: JsonPayload,
    *,
    github_payload: JsonPayload,
    github: Mapping[str, Any],
    hugging_face_payload: JsonPayload,
    hugging_face: Mapping[str, Any],
) -> Mapping[str, Any]:
    document = payload.document
    if document.get("schema_version") != PREPUBLICATION_RECEIPT_SCHEMA_VERSION:
        raise PostAttachmentVerificationError(
            "PREPUBLICATION_RECEIPT_SCHEMA_MISMATCH",
            "pre-publication receipt has an unexpected schema version",
        )
    _validate_self_hash(
        document, key="receipt_sha256", field="pre-publication receipt"
    )
    if (
        document.get("receipt_type") != "community_v2_pre_publication_gate"
        or document.get("package_version") != PACKAGE_VERSION
        or document.get("status") != "READY_FOR_FINAL_PUBLICATION_CONFIRMATION"
        or document.get("stage") != "pre_publication_gate"
        or document.get("final_publication_confirmed") is not False
    ):
        raise PostAttachmentVerificationError(
            "PREPUBLICATION_RECEIPT_STATE_MISMATCH",
            "pre-publication receipt is not the expected unpublished gate",
        )
    _timestamp(document.get("recorded_at"), field="pre-publication receipt timestamp")
    bindings = _mapping(document.get("evidence_bindings"), field="evidence bindings")
    expected_bindings = (
        (
            "github",
            github_payload,
            github,
        ),
        (
            "hugging_face",
            hugging_face_payload,
            hugging_face,
        ),
    )
    for name, source_payload, source_document in expected_bindings:
        binding = _mapping(bindings.get(name), field=f"{name} evidence binding")
        if (
            binding.get("schema_version") != source_document.get("schema_version")
            or binding.get("file_sha256") != source_payload.file_sha256
            or binding.get("evidence_sha256") != source_document.get("evidence_sha256")
        ):
            raise PostAttachmentVerificationError(
                "PREPUBLICATION_EVIDENCE_BINDING_MISMATCH",
                "pre-publication receipt does not bind the exact collector evidence",
            )

    github_projection = _mapping(document.get("github"), field="receipt GitHub projection")
    github_release = _mapping(github.get("release"), field="GitHub release evidence")
    projected_release = _mapping(
        github_projection.get("release"), field="receipt GitHub release projection"
    )
    if (
        github_projection.get("repository") != GITHUB_REPOSITORY
        or github_projection.get("visibility") != "public"
        or github_projection.get("source_commit_sha") != github.get("source_commit_sha")
        or github_projection.get("signed_tag") != github.get("signed_tag")
        or projected_release.get("release_id") != github_release.get("release_id")
        or projected_release.get("tag_name") != github_release.get("tag_name")
        or projected_release.get("resolved_source_sha")
        != github_release.get("resolved_source_sha")
        or projected_release.get("body_sha256") != github_release.get("body_sha256")
        or projected_release.get("draft") is not True
        or projected_release.get("prerelease") is not True
    ):
        raise PostAttachmentVerificationError(
            "PREPUBLICATION_GITHUB_PROJECTION_MISMATCH",
            "pre-publication receipt GitHub projection differs from evidence",
        )

    evidence_assets = _list(github_release.get("assets"), field="GitHub evidence assets")
    projected_assets = _list(document.get("release_assets"), field="receipt release assets")
    expected_projected_assets = [
        {
            "asset_id": item["asset_id"],
            "role": item["role"],
            "name": item["name"],
            "size_bytes": item["size_bytes"],
            "sha256": item["downloaded_sha256"],
        }
        for item in evidence_assets
    ]
    if (
        projected_assets != expected_projected_assets
        or document.get("release_asset_inventory_sha256")
        != canonical_json_hash(projected_assets)
    ):
        raise PostAttachmentVerificationError(
            "PREPUBLICATION_ASSET_PROJECTION_MISMATCH",
            "pre-publication receipt asset projection differs from GitHub evidence",
        )
    attachment = _mapping(
        document.get("draft_release_attachment"), field="draft receipt attachment"
    )
    if (
        attachment.get("asset_name") != PREPUBLICATION_RECEIPT_ASSET
        or attachment.get("attach_after_receipt_generation") is not True
        or attachment.get("excluded_from_bound_release_asset_inventory") is not True
        or attachment.get("release_must_remain_draft") is not True
    ):
        raise PostAttachmentVerificationError(
            "PREPUBLICATION_ATTACHMENT_CONTRACT_MISMATCH",
            "pre-publication receipt attachment contract is invalid",
        )

    hf_projection = _mapping(
        document.get("hugging_face"), field="receipt Hugging Face projection"
    )
    hf_revision = _mapping(hugging_face.get("revision"), field="HF evidence revision")
    evidence_inventory = _evidence_hf_inventory(hugging_face)
    expected_hf_projection = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "remote_oid": item["remote_oid"],
            "sha256": item["downloaded_sha256"],
        }
        for item in evidence_inventory
    ]
    if (
        hf_projection.get("repository") != HUGGING_FACE_REPOSITORY
        or hf_projection.get("visibility") != "private"
        or hf_projection.get("immutable_sha") != hf_revision.get("immutable_sha")
        or hf_projection.get("inventory") != expected_hf_projection
        or hf_projection.get("inventory_sha256")
        != canonical_json_hash(expected_hf_projection)
    ):
        raise PostAttachmentVerificationError(
            "PREPUBLICATION_HF_PROJECTION_MISMATCH",
            "pre-publication receipt Hugging Face projection differs from evidence",
        )
    return document


class GhRestTransport:
    """Read only from official GitHub REST through an explicitly pinned ``gh`` host."""

    def __init__(self) -> None:
        executable = shutil.which("gh")
        if executable is None:
            raise PostAttachmentVerificationError(
                "GH_UNAVAILABLE", "gh CLI is unavailable"
            )
        self._executable = str(Path(executable).resolve(strict=True))

    @staticmethod
    def _environment() -> dict[str, str]:
        allowed = {
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "HOME",
            "XDG_CONFIG_HOME",
            "PATH",
            "LANG",
            "LC_ALL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    def _command(self, endpoint: str, *, accept: str) -> list[str]:
        prefix = f"/repos/{GITHUB_REPOSITORY}/"
        if (
            not endpoint.startswith(prefix)
            or any(character in endpoint for character in "\r\n?#")
        ):
            raise PostAttachmentVerificationError(
                "UNSAFE_GITHUB_ENDPOINT", "GitHub REST endpoint is outside the release repository"
            )
        return [
            self._executable,
            "api",
            "--hostname",
            "github.com",
            "--method",
            "GET",
            "-H",
            f"Accept: {accept}",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            endpoint,
        ]

    def get_json(self, endpoint: str) -> Any:
        completed = subprocess.run(
            self._command(endpoint, accept="application/vnd.github+json"),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=self._environment(),
            timeout=120,
        )
        if completed.returncode != 0:
            raise PostAttachmentVerificationError(
                "GITHUB_READ_FAILED",
                "GitHub REST read failed without exposing its response",
            )
        if len(completed.stdout) > MAX_JSON_BYTES:
            raise PostAttachmentVerificationError(
                "GITHUB_RESPONSE_TOO_LARGE", "GitHub REST response exceeded its limit"
            )
        return _strict_json(completed.stdout, field="GitHub REST response")

    def download_bytes(self, endpoint: str, *, maximum: int) -> bytes:
        if maximum <= 0 or maximum > MAX_RECEIPT_DOWNLOAD_BYTES:
            raise PostAttachmentVerificationError(
                "INVALID_DOWNLOAD_LIMIT", "receipt download limit is invalid"
            )
        with tempfile.TemporaryFile() as output:
            completed = subprocess.run(
                self._command(endpoint, accept="application/octet-stream"),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.PIPE,
                env=self._environment(),
                timeout=120,
            )
            if completed.returncode != 0:
                raise PostAttachmentVerificationError(
                    "GITHUB_RECEIPT_DOWNLOAD_FAILED",
                    "GitHub receipt download failed without exposing its response",
                )
            size = output.tell()
            if size <= 0 or size > maximum:
                raise PostAttachmentVerificationError(
                    "GITHUB_RECEIPT_DOWNLOAD_SIZE_INVALID",
                    "GitHub receipt download exceeded its strict bound",
                )
            output.seek(0)
            payload = output.read(maximum + 1)
            if len(payload) != size:
                raise PostAttachmentVerificationError(
                    "GITHUB_RECEIPT_DOWNLOAD_CHANGED",
                    "GitHub receipt download changed while it was read",
                )
            return payload


class OfficialHuggingFaceTransport:
    """Metadata-only client pinned to the official Hugging Face endpoint."""

    def __init__(self) -> None:
        if HfApi is None:
            raise PostAttachmentVerificationError(
                "HF_CLIENT_UNAVAILABLE", "huggingface_hub is unavailable"
            )
        token = os.environ.get("HF_TOKEN")
        if token is None or not token.strip() or token != token.strip():
            raise PostAttachmentVerificationError(
                "HF_TOKEN_UNAVAILABLE", "HF_TOKEN is absent or malformed"
            )
        self._api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)

    def repo_info(self, *, revision: str) -> object:
        try:
            return self._api.model_info(
                repo_id=HUGGING_FACE_REPOSITORY,
                revision=revision,
                files_metadata=True,
            )
        except Exception as exc:
            raise PostAttachmentVerificationError(
                "HF_READ_FAILED", "Hugging Face metadata read failed"
            ) from exc

    def list_repo_files(self, *, revision: str) -> list[str]:
        try:
            return self._api.list_repo_files(
                repo_id=HUGGING_FACE_REPOSITORY,
                revision=revision,
                repo_type="model",
            )
        except Exception as exc:
            raise PostAttachmentVerificationError(
                "HF_READ_FAILED", "Hugging Face inventory read failed"
            ) from exc


def _object_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _live_hf_inventory(info: object, paths: list[str]) -> list[dict[str, Any]]:
    siblings = _object_field(info, "siblings")
    if not isinstance(siblings, list):
        raise PostAttachmentVerificationError(
            "HF_METADATA_MISSING", "Hugging Face per-file metadata is unavailable"
        )
    by_path: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        path = _normalized_repo_path(
            _object_field(sibling, "rfilename"), field="remote Hugging Face path"
        )
        if path in by_path:
            raise PostAttachmentVerificationError(
                "HF_REMOTE_PATH_DUPLICATE", "Hugging Face metadata repeats a path"
            )
        size = _integer(
            _object_field(sibling, "size"), field=f"remote Hugging Face size for {path}"
        )
        lfs = _object_field(sibling, "lfs")
        lfs_sha = _object_field(lfs, "sha256") if lfs is not None else None
        oid = lfs_sha if lfs_sha is not None else _object_field(sibling, "blob_id")
        oid_text = _string(oid, field=f"remote Hugging Face OID for {path}")
        if SHA1_RE.fullmatch(oid_text) is None and SHA256_RE.fullmatch(oid_text) is None:
            raise PostAttachmentVerificationError(
                "HF_REMOTE_OID_INVALID", "Hugging Face remote OID is malformed"
            )
        by_path[path] = {
            "path": path,
            "size_bytes": size,
            "remote_oid": oid_text,
        }
    if sorted(by_path) != paths:
        raise PostAttachmentVerificationError(
            "HF_REMOTE_INVENTORY_MISMATCH",
            "Hugging Face metadata and file listing differ",
        )
    return [by_path[path] for path in paths]


def _hf_snapshot(
    transport: HuggingFaceTransport,
    *,
    revision: str,
) -> dict[str, Any]:
    info = transport.repo_info(revision=revision)
    if (
        _object_field(info, "id") != HUGGING_FACE_REPOSITORY
        or _object_field(info, "private") is not True
    ):
        raise PostAttachmentVerificationError(
            "HF_REMOTE_STATE_MISMATCH",
            "Hugging Face repository is not the exact private repository",
        )
    resolved_sha = _string(
        _object_field(info, "sha"), field="Hugging Face resolved SHA", pattern=SHA1_RE
    )
    raw_paths = transport.list_repo_files(revision=revision)
    paths = sorted(
        _normalized_repo_path(path, field="Hugging Face listed path") for path in raw_paths
    )
    if len(paths) != len(set(paths)):
        raise PostAttachmentVerificationError(
            "HF_REMOTE_PATH_DUPLICATE", "Hugging Face listing repeats a path"
        )
    return {
        "resolved_sha": resolved_sha,
        "inventory": _live_hf_inventory(info, paths),
    }


def _github_digest(value: object, *, field: str) -> str:
    text = _string(value, field=field)
    match = GITHUB_DIGEST_RE.fullmatch(text)
    if match is None:
        raise PostAttachmentVerificationError(
            "GITHUB_DIGEST_UNAVAILABLE", f"{field} lacks an official SHA-256 digest"
        )
    return match.group(1)


def _github_snapshot(
    transport: GitHubTransport,
    *,
    release_id: int,
) -> dict[str, Any]:
    release = _mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}/releases/{release_id}"),
        field="GitHub release response",
    )
    if _integer(release.get("id"), field="remote GitHub release ID", minimum=1) != release_id:
        raise PostAttachmentVerificationError(
            "GITHUB_RELEASE_ID_MISMATCH", "GitHub returned another release"
        )
    body = release.get("body")
    if not isinstance(body, str):
        raise PostAttachmentVerificationError(
            "GITHUB_RELEASE_BODY_INVALID", "GitHub release body is unavailable"
        )
    assets: list[dict[str, Any]] = []
    for value in _list(release.get("assets"), field="remote GitHub assets"):
        asset = _mapping(value, field="remote GitHub asset")
        if asset.get("state") != "uploaded":
            raise PostAttachmentVerificationError(
                "GITHUB_ASSET_STATE_INVALID", "GitHub release has an incomplete asset"
            )
        assets.append(
            {
                "asset_id": _integer(
                    asset.get("id"), field="remote GitHub asset ID", minimum=1
                ),
                "name": _string(asset.get("name"), field="remote GitHub asset name"),
                "size_bytes": _integer(
                    asset.get("size"), field="remote GitHub asset size"
                ),
                "github_digest_sha256": _github_digest(
                    asset.get("digest"), field="remote GitHub asset digest"
                ),
            }
        )
    assets.sort(key=lambda item: item["name"])
    names = [item["name"] for item in assets]
    ids = [item["asset_id"] for item in assets]
    if (
        set(names) != EXPECTED_REMOTE_ASSET_NAMES
        or len(names) != len(EXPECTED_REMOTE_ASSET_NAMES)
        or len(ids) != len(set(ids))
    ):
        raise PostAttachmentVerificationError(
            "GITHUB_POST_ATTACHMENT_ASSET_SET_MISMATCH",
            "GitHub draft release must contain exactly four fixed assets",
        )

    reference = _mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}/git/ref/tags/{RELEASE_TAG}"),
        field="GitHub tag ref response",
    )
    reference_object = _mapping(reference.get("object"), field="GitHub tag ref object")
    tag_object_sha = _string(
        reference_object.get("sha"), field="remote tag object SHA", pattern=SHA1_RE
    )
    if reference_object.get("type") != "tag":
        raise PostAttachmentVerificationError(
            "GITHUB_TAG_NOT_ANNOTATED", "GitHub tag ref is not an annotated tag"
        )
    tag_object = _mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}/git/tags/{tag_object_sha}"),
        field="GitHub annotated tag response",
    )
    target = _mapping(tag_object.get("object"), field="GitHub annotated tag target")
    return {
        "release": {
            "release_id": release_id,
            "tag_name": _string(release.get("tag_name"), field="remote release tag"),
            "draft": release.get("draft"),
            "prerelease": release.get("prerelease"),
            "created_at": _timestamp(
                release.get("created_at"), field="remote release creation time"
            ),
            "updated_at": _timestamp(
                release.get("updated_at"), field="remote release update time"
            ),
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        },
        "annotated_tag": {
            "ref_target_type": "tag",
            "ref_target_sha": tag_object_sha,
            "tag_object_sha": _string(
                tag_object.get("sha"), field="annotated tag object SHA", pattern=SHA1_RE
            ),
            "tag_target_type": target.get("type"),
            "tag_target_sha": _string(
                target.get("sha"), field="annotated tag target SHA", pattern=SHA1_RE
            ),
        },
        "assets": assets,
    }


def _validate_remote_github(
    snapshot: Mapping[str, Any],
    *,
    github_evidence: Mapping[str, Any],
    prepublication_receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    remote_release = _mapping(snapshot.get("release"), field="remote release snapshot")
    evidence_release = _mapping(github_evidence.get("release"), field="GitHub evidence release")
    projected_release = _mapping(
        _mapping(prepublication_receipt.get("github"), field="receipt GitHub").get("release"),
        field="receipt release projection",
    )
    source_sha = github_evidence["source_commit_sha"]
    if (
        remote_release.get("release_id") != evidence_release.get("release_id")
        or remote_release.get("tag_name") != RELEASE_TAG
        or remote_release.get("draft") is not True
        or remote_release.get("prerelease") is not True
        or remote_release.get("body_sha256") != evidence_release.get("body_sha256")
        or remote_release.get("body_sha256") != projected_release.get("body_sha256")
    ):
        raise PostAttachmentVerificationError(
            "GITHUB_RELEASE_DRIFT",
            "GitHub draft release identity, state, tag, or body changed",
        )
    remote_tag = _mapping(snapshot.get("annotated_tag"), field="remote annotated tag")
    evidence_tag = _mapping(github_evidence.get("signed_tag"), field="evidence signed tag")
    expected_tag = {
        "ref_target_type": "tag",
        "ref_target_sha": evidence_tag["ref_target_sha"],
        "tag_object_sha": evidence_tag["tag_object_sha"],
        "tag_target_type": "commit",
        "tag_target_sha": source_sha,
    }
    if remote_tag != expected_tag:
        raise PostAttachmentVerificationError(
            "GITHUB_ANNOTATED_TAG_DRIFT",
            "GitHub annotated tag no longer targets the exact source commit",
        )

    evidence_assets = {
        item["name"]: item
        for item in _list(evidence_release.get("assets"), field="GitHub evidence assets")
    }
    output_assets: list[dict[str, Any]] = []
    receipt_asset_id = 0
    for remote in _list(snapshot.get("assets"), field="remote asset snapshot"):
        asset = _mapping(remote, field="remote asset")
        name = asset["name"]
        if name == PREPUBLICATION_RECEIPT_ASSET:
            receipt_asset_id = asset["asset_id"]
            output_assets.append(
                {
                    **asset,
                    "role": "prepublication_receipt",
                    "expected_sha256": "",
                    "source_binding": "exact_local_prepublication_receipt_bytes",
                    "downloaded_sha256": "",
                    "downloaded_size_bytes": 0,
                }
            )
            continue
        expected = evidence_assets.get(name)
        if expected is None:
            raise PostAttachmentVerificationError(
                "GITHUB_POST_ATTACHMENT_ASSET_SET_MISMATCH",
                "GitHub release contains an unexpected asset",
            )
        if (
            asset["asset_id"] != expected["asset_id"]
            or asset["size_bytes"] != expected["size_bytes"]
            or asset["github_digest_sha256"] != expected["downloaded_sha256"]
            or asset["github_digest_sha256"] != expected["github_digest_sha256"]
        ):
            raise PostAttachmentVerificationError(
                "GITHUB_ORIGINAL_ASSET_DRIFT",
                "GitHub wheel, SBOM, or checksums digest/size changed",
            )
        output_assets.append(
            {
                **asset,
                "role": expected["role"],
                "expected_sha256": expected["downloaded_sha256"],
                "source_binding": "github_pre_attachment_collector_evidence",
                "downloaded_sha256": None,
                "downloaded_size_bytes": None,
            }
        )
    if receipt_asset_id <= 0:
        raise PostAttachmentVerificationError(
            "GITHUB_RECEIPT_ASSET_MISSING", "pre-publication receipt attachment is absent"
        )
    output_assets.sort(key=lambda item: item["name"])
    return output_assets, receipt_asset_id


def _validate_remote_hf(
    snapshot: Mapping[str, Any],
    *,
    revision: str,
    expected_inventory: list[dict[str, Any]],
) -> None:
    if snapshot.get("resolved_sha") != revision:
        raise PostAttachmentVerificationError(
            "HF_REMOTE_REVISION_DRIFT",
            "Hugging Face main or immutable revision moved",
        )
    live_inventory = _list(snapshot.get("inventory"), field="live HF inventory")
    expected_remote = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "remote_oid": item["remote_oid"],
        }
        for item in expected_inventory
    ]
    if live_inventory != expected_remote:
        raise PostAttachmentVerificationError(
            "HF_REMOTE_INVENTORY_DRIFT",
            "Hugging Face private inventory differs from collector evidence",
        )


def build_post_attachment_receipt(
    *,
    github_payload: JsonPayload,
    hugging_face_payload: JsonPayload,
    prepublication_payload: JsonPayload,
    github_transport: GitHubTransport,
    hugging_face_transport: HuggingFaceTransport,
    verified_at: str,
) -> dict[str, Any]:
    """Perform the complete read-only post-attachment verification."""

    _validate_payload_integrity(github_payload, field="GitHub collector evidence")
    _validate_payload_integrity(
        hugging_face_payload, field="Hugging Face collector evidence"
    )
    _validate_payload_integrity(
        prepublication_payload, field="pre-publication gate receipt"
    )
    github = _validate_github_evidence(github_payload)
    hugging_face = _validate_hugging_face_evidence(hugging_face_payload)
    prepublication = _validate_prepublication_receipt(
        prepublication_payload,
        github_payload=github_payload,
        github=github,
        hugging_face_payload=hugging_face_payload,
        hugging_face=hugging_face,
    )
    verified_at_text = _timestamp(verified_at, field="verification timestamp")
    recorded_at = datetime.fromisoformat(
        str(prepublication["recorded_at"]).replace("Z", "+00:00")
    )
    verification_time = datetime.fromisoformat(
        verified_at_text.replace("Z", "+00:00")
    )
    if verification_time < recorded_at:
        raise PostAttachmentVerificationError(
            "VERIFICATION_TIME_ORDER",
            "post-attachment verification predates the pre-publication receipt",
        )

    generator_sha256 = _sha256_regular_file(
        REPOSITORY_ROOT / VERIFIER_PATH, field="post-attachment verifier"
    )
    release_id = _integer(
        _mapping(github.get("release"), field="GitHub evidence release").get(
            "release_id"
        ),
        field="GitHub release ID",
        minimum=1,
    )
    github_start = _github_snapshot(github_transport, release_id=release_id)
    output_assets, receipt_asset_id = _validate_remote_github(
        github_start,
        github_evidence=github,
        prepublication_receipt=prepublication,
    )

    immutable_sha = _string(
        _mapping(hugging_face.get("revision"), field="HF evidence revision").get(
            "immutable_sha"
        ),
        field="HF immutable SHA",
        pattern=SHA1_RE,
    )
    expected_hf_inventory = _evidence_hf_inventory(hugging_face)
    hf_main_start = _hf_snapshot(hugging_face_transport, revision="main")
    hf_immutable_start = _hf_snapshot(
        hugging_face_transport, revision=immutable_sha
    )
    _validate_remote_hf(
        hf_main_start,
        revision=immutable_sha,
        expected_inventory=expected_hf_inventory,
    )
    _validate_remote_hf(
        hf_immutable_start,
        revision=immutable_sha,
        expected_inventory=expected_hf_inventory,
    )

    downloaded_receipt = github_transport.download_bytes(
        f"/repos/{GITHUB_REPOSITORY}/releases/assets/{receipt_asset_id}",
        maximum=MAX_RECEIPT_DOWNLOAD_BYTES,
    )
    downloaded_sha256 = hashlib.sha256(downloaded_receipt).hexdigest()
    if downloaded_receipt != prepublication_payload.payload:
        raise PostAttachmentVerificationError(
            "GITHUB_RECEIPT_BYTES_MISMATCH",
            "downloaded pre-publication receipt bytes differ from the local receipt",
        )
    for asset in output_assets:
        if asset["name"] == PREPUBLICATION_RECEIPT_ASSET:
            if (
                asset["size_bytes"] != len(prepublication_payload.payload)
                or asset["github_digest_sha256"] != prepublication_payload.file_sha256
                or downloaded_sha256 != prepublication_payload.file_sha256
            ):
                raise PostAttachmentVerificationError(
                    "GITHUB_RECEIPT_METADATA_MISMATCH",
                    "receipt attachment digest or size differs from local bytes",
                )
            asset["expected_sha256"] = prepublication_payload.file_sha256
            asset["downloaded_sha256"] = downloaded_sha256
            asset["downloaded_size_bytes"] = len(downloaded_receipt)

    github_end = _github_snapshot(github_transport, release_id=release_id)
    hf_main_end = _hf_snapshot(hugging_face_transport, revision="main")
    hf_immutable_end = _hf_snapshot(hugging_face_transport, revision=immutable_sha)
    if github_end != github_start:
        raise PostAttachmentVerificationError(
            "GITHUB_STATE_CHANGED_DURING_VERIFICATION",
            "GitHub release or tag changed during verification",
        )
    if hf_main_end != hf_main_start or hf_immutable_end != hf_immutable_start:
        raise PostAttachmentVerificationError(
            "HF_STATE_CHANGED_DURING_VERIFICATION",
            "Hugging Face repository changed during verification",
        )
    if generator_sha256 != _sha256_regular_file(
        REPOSITORY_ROOT / VERIFIER_PATH, field="post-attachment verifier"
    ):
        raise PostAttachmentVerificationError(
            "VERIFIER_CHANGED_DURING_RUN", "verifier implementation changed during the run"
        )

    output_assets.sort(key=lambda item: item["name"])
    output_hf_inventory = expected_hf_inventory
    receipt: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "receipt_type": "community_v2_post_attachment_verification",
        "package_version": PACKAGE_VERSION,
        "verified_at": verified_at_text,
        "status": "PASS",
        "stage": "draft_release_post_attachment",
        "final_publication_confirmed": False,
        "verifier": {
            "name": VERIFIER_PATH.name,
            "version": VERIFIER_VERSION,
            "generator_path": VERIFIER_PATH.as_posix(),
            "generator_file_sha256": generator_sha256,
            "github_endpoint": OFFICIAL_GITHUB_ENDPOINT,
            "hugging_face_endpoint": OFFICIAL_HF_ENDPOINT,
            "network_accessed": True,
            "remote_mutation_performed": False,
            "hugging_face_weight_download_performed": False,
            "prepublication_receipt_downloaded": True,
        },
        "evidence_bindings": {
            "github_collector": {
                "schema_version": github["schema_version"],
                "file_sha256": github_payload.file_sha256,
                "self_hash": github["evidence_sha256"],
            },
            "hugging_face_collector": {
                "schema_version": hugging_face["schema_version"],
                "file_sha256": hugging_face_payload.file_sha256,
                "self_hash": hugging_face["evidence_sha256"],
            },
            "prepublication_receipt": {
                "schema_version": prepublication["schema_version"],
                "file_sha256": prepublication_payload.file_sha256,
                "receipt_sha256": prepublication["receipt_sha256"],
                "size_bytes": len(prepublication_payload.payload),
            },
        },
        "github": {
            "repository": GITHUB_REPOSITORY,
            "source_commit_sha": github["source_commit_sha"],
            "release": github_start["release"],
            "annotated_tag": github_start["annotated_tag"],
            "release_assets": output_assets,
            "release_asset_inventory_sha256": canonical_json_hash(output_assets),
        },
        "hugging_face": {
            "repository": HUGGING_FACE_REPOSITORY,
            "visibility": "private",
            "private": True,
            "main_revision": "main",
            "main_resolved_sha": hf_main_start["resolved_sha"],
            "immutable_revision": immutable_sha,
            "immutable_resolved_sha": hf_immutable_start["resolved_sha"],
            "inventory": output_hf_inventory,
            "inventory_sha256": canonical_json_hash(output_hf_inventory),
        },
        "verification": {
            "github_release_identity_stable": True,
            "github_release_remains_draft_prerelease": True,
            "github_release_body_hash_matches_evidence": True,
            "github_annotated_tag_targets_exact_source": True,
            "github_asset_set_exact": True,
            "github_original_asset_digest_and_size_match_evidence": True,
            "github_receipt_asset_digest_and_size_match_local": True,
            "github_receipt_download_bytes_match_local": True,
            "hugging_face_repository_remains_private": True,
            "hugging_face_main_resolves_to_immutable_sha": True,
            "hugging_face_immutable_revision_stable": True,
            "hugging_face_inventory_matches_evidence": True,
            "remote_state_stable_during_verification": True,
        },
        "limitations": {
            "does_not_attest_final_publication": True,
            "github_release_must_remain_draft": True,
            "hugging_face_repository_must_remain_private": True,
            "final_visibility_transition_requires_separate_confirmation": True,
            "hugging_face_file_contents_not_redownloaded": True,
            "statement": (
                "This receipt verifies the attached pre-publication gate receipt and the "
                "still-private/draft remote state. It does not authorize or attest final "
                "publication."
            ),
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, remove="receipt_sha256")
    validate_output_receipt(receipt)
    return receipt


def _output_schema() -> Mapping[str, Any]:
    payload, _mode = _read_regular(
        REPOSITORY_ROOT / SCHEMA_PATH,
        field="post-attachment receipt schema",
        maximum=MAX_JSON_BYTES,
    )
    return _strict_json(payload, field="post-attachment receipt schema")


def validate_output_receipt(document: Mapping[str, Any]) -> None:
    """Validate the closed output schema and canonical self-hash."""

    schema = _output_schema()
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(document),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except Exception as exc:
        raise PostAttachmentVerificationError(
            "OUTPUT_SCHEMA_INVALID", "post-attachment receipt schema is invalid"
        ) from exc
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise PostAttachmentVerificationError(
            "OUTPUT_SCHEMA_REJECTED",
            f"post-attachment receipt failed its closed schema at {location}",
        )
    _validate_self_hash(document, key="receipt_sha256", field="output receipt")


def _write_new_receipt(path: Path, document: Mapping[str, Any]) -> str:
    payload = _pretty_json_bytes(document)
    path = path.expanduser()
    if path.name in {"", ".", ".."}:
        raise PostAttachmentVerificationError(
            "OUTPUT_PATH_UNSAFE", "output filename is unsafe"
        )
    try:
        parent_before = os.lstat(path.parent)
    except OSError as exc:
        raise PostAttachmentVerificationError(
            "OUTPUT_PARENT_MISSING", "output parent is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise PostAttachmentVerificationError(
            "OUTPUT_PARENT_UNSAFE", "output parent must be a real directory"
        )
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(path.parent, parent_flags)
    except OSError as exc:
        raise PostAttachmentVerificationError(
            "OUTPUT_PARENT_UNSAFE", "output parent is unsafe"
        ) from exc
    parent_identity = (parent_before.st_dev, parent_before.st_ino)
    opened_parent = os.fstat(parent_fd)
    if (opened_parent.st_dev, opened_parent.st_ino) != parent_identity:
        os.close(parent_fd)
        raise PostAttachmentVerificationError(
            "OUTPUT_PARENT_CHANGED", "output parent changed"
        )

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    owned_identity: tuple[int, int] | None = None

    def remove_owned() -> None:
        if owned_identity is None:
            return
        try:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            return
        if stat.S_ISREG(current.st_mode) and (
            current.st_dev,
            current.st_ino,
        ) == owned_identity:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except OSError:
                pass

    try:
        descriptor = os.open(path.name, flags, 0o444, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        owned_identity = (opened.st_dev, opened.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        final_descriptor = os.fstat(descriptor)
    except FileExistsError as exc:
        os.close(parent_fd)
        raise PostAttachmentVerificationError(
            "OUTPUT_EXISTS", "receipt output is no-clobber"
        ) from exc
    except BaseException as exc:
        remove_owned()
        os.close(parent_fd)
        if isinstance(exc, OSError):
            raise PostAttachmentVerificationError(
                "OUTPUT_WRITE_FAILED", "receipt output failed"
            ) from exc
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        final_parent = os.lstat(path.parent)
    except OSError as exc:
        remove_owned()
        os.close(parent_fd)
        raise PostAttachmentVerificationError(
            "OUTPUT_VERIFICATION_FAILED", "receipt output could not be finalized"
        ) from exc
    valid = (
        owned_identity is not None
        and stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == owned_identity
        and (final_descriptor.st_dev, final_descriptor.st_ino) == owned_identity
        and current.st_size == len(payload)
        and final_descriptor.st_size == len(payload)
        and stat.S_IMODE(current.st_mode) == 0o444
        and not stat.S_ISLNK(final_parent.st_mode)
        and (final_parent.st_dev, final_parent.st_ino) == parent_identity
    )
    if not valid:
        remove_owned()
        os.close(parent_fd)
        raise PostAttachmentVerificationError(
            "OUTPUT_VERIFICATION_FAILED", "receipt output drifted"
        )
    os.close(parent_fd)
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-evidence", required=True, type=Path)
    parser.add_argument("--hugging-face-evidence", required=True, type=Path)
    parser.add_argument("--prepublication-receipt", required=True, type=Path)
    parser.add_argument("--verified-at")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        github = load_json_payload(args.github_evidence, field="GitHub collector evidence")
        hugging_face = load_json_payload(
            args.hugging_face_evidence, field="Hugging Face collector evidence"
        )
        prepublication = load_json_payload(
            args.prepublication_receipt, field="pre-publication gate receipt"
        )
        github_document = _validate_github_evidence(github)
        hugging_face_document = _validate_hugging_face_evidence(hugging_face)
        _validate_prepublication_receipt(
            prepublication,
            github_payload=github,
            github=github_document,
            hugging_face_payload=hugging_face,
            hugging_face=hugging_face_document,
        )
        receipt = build_post_attachment_receipt(
            github_payload=github,
            hugging_face_payload=hugging_face,
            prepublication_payload=prepublication,
            github_transport=GhRestTransport(),
            hugging_face_transport=OfficialHuggingFaceTransport(),
            verified_at=args.verified_at or _utc_now(),
        )
        file_sha256 = _write_new_receipt(args.output, receipt)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "stage": receipt["stage"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "file_sha256": file_sha256,
                    "remote_mutation_performed": False,
                    "hugging_face_weight_download_performed": False,
                    "sensitive_values_logged": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except PostAttachmentVerificationError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "blocker_ids": [exc.blocker_id],
                    "remote_mutation_performed": False,
                    "hugging_face_weight_download_performed": False,
                    "sensitive_values_logged": False,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
