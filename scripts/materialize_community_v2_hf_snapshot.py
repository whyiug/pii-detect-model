#!/usr/bin/env python3
"""Materialize and bind one immutable Hugging Face model revision.

The command resolves a full commit against the official Hub, downloads every
remote file through ``huggingface_hub``, copies it into a no-clobber regular-file
tree, verifies the publication-successor closure, and emits a strict self-hashed
provenance receipt.  It never accepts caller-supplied hashes or PASS status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover - release environment supplies the client
    HfApi = None  # type: ignore[assignment,misc]
    hf_hub_download = None  # type: ignore[assignment]

try:
    from scripts import build_community_v2_publication_successor as successor
except ImportError:  # pragma: no cover - direct script execution
    import build_community_v2_publication_successor as successor  # type: ignore[no-redef]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = Path("scripts/materialize_community_v2_hf_snapshot.py")
SCHEMA_PATH = REPOSITORY_ROOT / (
    "configs/release/community_v2_hf_download_provenance.schema.json"
)
OFFICIAL_ENDPOINT = "https://huggingface.co"
SCHEMA_VERSION = "pii-zh.community-v2-hf-download-provenance.v1"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)
MAX_JSON_BYTES = 32 * 1024 * 1024
READ_BLOCK_BYTES = 1024 * 1024
REMOTE_EVIDENCE_BOUNDARY = (
    "official Hub repository metadata and list_repo_files bind the immutable remote "
    "inventory; local SHA-256 binds materialized bytes; an advertised LFS SHA-256 or "
    "non-LFS Git blob SHA-1 is content-verified when available, otherwise transport "
    "and immutable-revision trust are explicit"
)


class HfDownloadProvenanceError(RuntimeError):
    """Raised when an immutable HF download cannot be proven safely."""


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HfDownloadProvenanceError(f"{field} repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise HfDownloadProvenanceError(f"{field} contains a non-finite JSON number")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HfDownloadProvenanceError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise HfDownloadProvenanceError(f"{field} must be a JSON object")
    return document


def _read_regular(path: Path, *, field: str, maximum: int) -> bytes:
    parent, parent_identity = _real_directory_path(path.parent, field=f"{field} parent")
    path = parent / path.name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HfDownloadProvenanceError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HfDownloadProvenanceError(
                f"{field} must be a regular non-symlink file"
            )
        if before.st_size <= 0 or before.st_size > maximum:
            raise HfDownloadProvenanceError(f"{field} has an invalid size")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(
                descriptor,
                min(READ_BLOCK_BYTES, maximum + 1 - total),
            )
            if not block:
                break
            total += len(block)
            if total > maximum:
                raise HfDownloadProvenanceError(f"{field} exceeds its size limit")
            chunks.append(block)
        after = os.fstat(descriptor)
        path_metadata = os.lstat(path)
        _assert_directory_identity(
            parent,
            parent_identity,
            field=f"{field} parent",
        )
        before_identity = _file_state(before)
        if (
            before_identity != _file_state(after)
            or total != before.st_size
            or stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise HfDownloadProvenanceError(f"{field} changed while it was read")
        return b"".join(chunks)
    except HfDownloadProvenanceError:
        raise
    except OSError as exc:
        raise HfDownloadProvenanceError(f"{field} could not be read") from exc
    finally:
        os.close(descriptor)


def _file_state(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(path: Path, *, field: str) -> tuple[int, int]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise HfDownloadProvenanceError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HfDownloadProvenanceError(f"{field} must be a real directory")
    return metadata.st_dev, metadata.st_ino


def _real_directory_path(path: Path, *, field: str) -> tuple[Path, tuple[int, int]]:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    try:
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HfDownloadProvenanceError(f"{field} must have no symlink components")
        for part in absolute.parts[1:]:
            current /= part
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise HfDownloadProvenanceError(
                    f"{field} must have no symlink components"
                )
    except HfDownloadProvenanceError:
        raise
    except OSError as exc:
        raise HfDownloadProvenanceError(f"{field} is unavailable") from exc
    return absolute, (metadata.st_dev, metadata.st_ino)


def _assert_directory_identity(
    path: Path, identity: tuple[int, int], *, field: str
) -> None:
    _absolute, observed = _real_directory_path(path, field=field)
    if observed != identity:
        raise HfDownloadProvenanceError(f"{field} changed during materialization")


def _load_schema() -> Mapping[str, Any]:
    return _strict_json(
        _read_regular(SCHEMA_PATH, field="HF provenance schema", maximum=MAX_JSON_BYTES),
        field="HF provenance schema",
    )


def _generator_sha256() -> str:
    digest, _size = successor._hash_regular_file(
        REPOSITORY_ROOT / GENERATOR_PATH,
        field="HF provenance generator",
    )
    return digest


def _inventory(root: Path) -> dict[str, dict[str, int | str]]:
    root_identity = _directory_identity(root, field="materialized HF snapshot")
    try:
        files = successor._scan_tree(root, field="materialized HF snapshot")
    except successor.PublicationSuccessorError as exc:
        raise HfDownloadProvenanceError(
            "materialized HF snapshot cannot be safely inventoried"
        ) from exc
    result: dict[str, dict[str, int | str]] = {}
    states: dict[str, tuple[int, int, int, int, int]] = {}
    for relative, path in files.items():
        try:
            before = os.lstat(path)
            digest, size = successor._hash_regular_file(
                path, field=f"materialized HF snapshot {relative}"
            )
            after = os.lstat(path)
        except (OSError, successor.PublicationSuccessorError) as exc:
            raise HfDownloadProvenanceError(
                f"materialized HF snapshot file {relative!r} is unsafe"
            ) from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or _file_state(before) != _file_state(after)
            or size != after.st_size
        ):
            raise HfDownloadProvenanceError(
                f"materialized HF snapshot file {relative!r} changed during inventory"
            )
        result[relative] = {"file_sha256": digest, "size_bytes": size}
        states[relative] = _file_state(after)
    try:
        final_files = successor._scan_tree(root, field="materialized HF snapshot")
    except successor.PublicationSuccessorError as exc:
        raise HfDownloadProvenanceError(
            "materialized HF snapshot changed during inventory"
        ) from exc
    if set(final_files) != set(files):
        raise HfDownloadProvenanceError(
            "materialized HF snapshot changed during inventory"
        )
    for relative, path in final_files.items():
        try:
            final_metadata = os.lstat(path)
        except OSError as exc:
            raise HfDownloadProvenanceError(
                "materialized HF snapshot changed during inventory"
            ) from exc
        if _file_state(final_metadata) != states[relative]:
            raise HfDownloadProvenanceError(
                "materialized HF snapshot changed during inventory"
            )
    _assert_directory_identity(
        root,
        root_identity,
        field="materialized HF snapshot",
    )
    return dict(sorted(result.items()))


def _validate_document(document: Mapping[str, Any]) -> None:
    validator = Draft202012Validator(_load_schema(), format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise HfDownloadProvenanceError(
            f"HF provenance schema failed at {location}: {first.message}"
        )
    if document.get("requested_revision") != document.get("resolved_commit"):
        raise HfDownloadProvenanceError("HF provenance revision did not resolve immutably")
    local_root = document.get("local_root")
    if not isinstance(local_root, Mapping):
        raise HfDownloadProvenanceError("HF provenance has no local-root binding")
    files = local_root.get("files")
    if not isinstance(files, Mapping) or (
        local_root.get("file_count") != len(files)
        or local_root.get("inventory_sha256") != successor.canonical_json_hash(files)
    ):
        raise HfDownloadProvenanceError("HF provenance inventory does not self-verify")
    remote_snapshot = document.get("remote_snapshot")
    if not isinstance(remote_snapshot, Mapping):
        raise HfDownloadProvenanceError("HF provenance has no remote-snapshot binding")
    remote_files = remote_snapshot.get("files")
    coverage = remote_snapshot.get("metadata_coverage")
    if not isinstance(remote_files, Mapping) or not isinstance(coverage, Mapping):
        raise HfDownloadProvenanceError(
            "HF provenance remote metadata inventory is malformed"
        )
    observed_coverage = {
        "size_count": sum(
            isinstance(binding, Mapping) and binding.get("size_bytes") is not None
            for binding in remote_files.values()
        ),
        "git_blob_oid_count": sum(
            isinstance(binding, Mapping) and binding.get("git_blob_oid") is not None
            for binding in remote_files.values()
        ),
        "lfs_oid_count": sum(
            isinstance(binding, Mapping) and binding.get("lfs_oid_sha256") is not None
            for binding in remote_files.values()
        ),
        "content_verified_count": sum(
            isinstance(binding, Mapping)
            and binding.get("content_verification")
            != "immutable_revision_transport_only"
            for binding in remote_files.values()
        ),
    }
    if (
        remote_snapshot.get("file_count") != len(remote_files)
        or remote_snapshot.get("metadata_inventory_sha256")
        != successor.canonical_json_hash(remote_files)
        or dict(coverage) != observed_coverage
        or set(remote_files) != set(files)
        or remote_snapshot.get("visibility")
        != ("private" if remote_snapshot.get("private") else "public")
        or remote_snapshot.get("evidence_boundary") != REMOTE_EVIDENCE_BOUNDARY
    ):
        raise HfDownloadProvenanceError(
            "HF provenance remote metadata inventory does not self-verify"
        )
    generator = document.get("generator")
    if not isinstance(generator, Mapping) or generator.get("file_sha256") != (
        _generator_sha256()
    ):
        raise HfDownloadProvenanceError("HF provenance generator binding does not verify")
    if document.get("receipt_sha256") != successor.canonical_json_hash(
        document, remove="receipt_sha256"
    ):
        raise HfDownloadProvenanceError("HF provenance receipt self-hash failed")


def load_and_validate_provenance(path: Path) -> dict[str, Any]:
    document = _strict_json(
        _read_regular(path, field="HF download provenance", maximum=MAX_JSON_BYTES),
        field="HF download provenance",
    )
    _validate_document(document)
    return document


def verify_local_root_binding(root: Path, document: Mapping[str, Any]) -> None:
    _validate_document(document)
    local_root = document.get("local_root")
    if not isinstance(local_root, Mapping) or not isinstance(local_root.get("files"), Mapping):
        raise HfDownloadProvenanceError("HF provenance has no local-root inventory")
    observed = _inventory(root)
    if observed != local_root["files"] or successor.canonical_json_hash(observed) != (
        local_root.get("inventory_sha256")
    ):
        raise HfDownloadProvenanceError(
            "materialized package differs from its HF download provenance"
        )


def _required_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token is None or not token.strip():
        raise HfDownloadProvenanceError(
            "HF_TOKEN must be present and non-empty in the process environment"
        )
    return token


def _object_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _nullable_hex(
    value: object,
    *,
    length: int,
    field: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is None:
        raise HfDownloadProvenanceError(f"{field} is malformed")
    return value


def _remote_snapshot(info: object, paths: list[str]) -> dict[str, Any]:
    private = _object_field(info, "private")
    if not isinstance(private, bool):
        raise HfDownloadProvenanceError(
            "official Hub repository visibility metadata is unavailable"
        )
    siblings = _object_field(info, "siblings")
    if not isinstance(siblings, list):
        raise HfDownloadProvenanceError(
            "official Hub per-file metadata inventory is unavailable"
        )

    files: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        raw_path = _object_field(sibling, "rfilename")
        if not isinstance(raw_path, str):
            raise HfDownloadProvenanceError("official Hub file metadata is malformed")
        try:
            relative = successor._safe_relative_name(
                raw_path,
                field="HF remote metadata inventory",
            )
        except successor.PublicationSuccessorError as exc:
            raise HfDownloadProvenanceError(
                "official Hub file metadata contains an unsafe path"
            ) from exc
        if relative in files:
            raise HfDownloadProvenanceError(
                "official Hub file metadata repeats a path"
            )
        size = _object_field(sibling, "size")
        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
        ):
            raise HfDownloadProvenanceError(
                f"official Hub size metadata for {relative!r} is malformed"
            )
        git_blob_oid = _nullable_hex(
            _object_field(sibling, "blob_id"),
            length=40,
            field=f"official Hub Git blob OID for {relative!r}",
        )
        lfs = _object_field(sibling, "lfs")
        if lfs is None:
            lfs_oid_sha256 = None
        else:
            lfs_oid_sha256 = _nullable_hex(
                _object_field(lfs, "sha256"),
                length=64,
                field=f"official Hub LFS OID for {relative!r}",
            )
            if lfs_oid_sha256 is None:
                raise HfDownloadProvenanceError(
                    f"official Hub LFS metadata for {relative!r} has no SHA-256 OID"
                )
        if lfs_oid_sha256 is not None:
            content_verification = "lfs_sha256"
        elif git_blob_oid is not None:
            content_verification = "git_blob_sha1"
        else:
            content_verification = "immutable_revision_transport_only"
        files[relative] = {
            "size_bytes": size,
            "git_blob_oid": git_blob_oid,
            "lfs_oid_sha256": lfs_oid_sha256,
            "content_verification": content_verification,
        }

    if set(files) != set(paths):
        raise HfDownloadProvenanceError(
            "official Hub path inventory and per-file metadata inventory differ"
        )
    files = dict(sorted(files.items()))
    coverage = {
        "size_count": sum(binding["size_bytes"] is not None for binding in files.values()),
        "git_blob_oid_count": sum(
            binding["git_blob_oid"] is not None for binding in files.values()
        ),
        "lfs_oid_count": sum(
            binding["lfs_oid_sha256"] is not None for binding in files.values()
        ),
        "content_verified_count": sum(
            binding["content_verification"]
            != "immutable_revision_transport_only"
            for binding in files.values()
        ),
    }
    return {
        "private": private,
        "visibility": "private" if private else "public",
        "file_count": len(files),
        "metadata_inventory_sha256": successor.canonical_json_hash(files),
        "metadata_coverage": coverage,
        "files": files,
        "evidence_boundary": REMOTE_EVIDENCE_BOUNDARY,
    }


def _git_blob_sha1(path: Path, *, expected_size: int, field: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HfDownloadProvenanceError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise HfDownloadProvenanceError(f"{field} has an unexpected size or type")
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {expected_size}\0".encode("ascii"))
        total = 0
        while True:
            block = os.read(descriptor, READ_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if _file_state(before) != _file_state(after) or total != expected_size:
            raise HfDownloadProvenanceError(f"{field} changed while it was hashed")
        return digest.hexdigest()
    except OSError as exc:
        raise HfDownloadProvenanceError(f"{field} could not be hashed") from exc
    finally:
        os.close(descriptor)


def _verify_download_metadata(
    path: Path,
    *,
    relative: str,
    sha256: str,
    size: int,
    remote_binding: Mapping[str, Any],
) -> None:
    remote_size = remote_binding.get("size_bytes")
    if remote_size is not None and remote_size != size:
        raise HfDownloadProvenanceError(
            f"downloaded HF file {relative!r} differs from its advertised size"
        )
    method = remote_binding.get("content_verification")
    if method == "lfs_sha256":
        if remote_binding.get("lfs_oid_sha256") != sha256:
            raise HfDownloadProvenanceError(
                f"downloaded HF file {relative!r} differs from its LFS OID"
            )
    elif method == "git_blob_sha1":
        observed = _git_blob_sha1(
            path,
            expected_size=size,
            field=f"downloaded HF file {relative}",
        )
        if remote_binding.get("git_blob_oid") != observed:
            raise HfDownloadProvenanceError(
                f"downloaded HF file {relative!r} differs from its Git blob OID"
            )
    elif method != "immutable_revision_transport_only":
        raise HfDownloadProvenanceError(
            f"downloaded HF file {relative!r} has an invalid verification method"
        )


def _remote_paths(values: object) -> list[str]:
    if not isinstance(values, list) or not values:
        raise HfDownloadProvenanceError("HF revision has no file inventory")
    paths: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise HfDownloadProvenanceError("HF file inventory is malformed")
        try:
            paths.append(successor._safe_relative_name(value, field="HF remote inventory"))
        except successor.PublicationSuccessorError as exc:
            raise HfDownloadProvenanceError("HF file inventory contains an unsafe path") from exc
    if len(paths) != len(set(paths)):
        raise HfDownloadProvenanceError("HF file inventory repeats a path")
    return sorted(paths)


def materialize_snapshot(
    *,
    repository: str,
    revision: str,
    output: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    if REPO_ID_RE.fullmatch(repository) is None:
        raise HfDownloadProvenanceError("HF repository must be an owner/repository ID")
    if GIT_SHA_RE.fullmatch(revision) is None:
        raise HfDownloadProvenanceError("HF revision must be a full lowercase commit SHA")
    if HfApi is None or hf_hub_download is None:
        raise HfDownloadProvenanceError("huggingface_hub is unavailable")
    if output.exists() or output.is_symlink():
        raise HfDownloadProvenanceError("materialized HF output already exists")
    hf_token = _required_hf_token()
    cache_root, cache_identity = _real_directory_path(cache_dir, field="HF cache")
    output_parent, output_parent_identity = _real_directory_path(
        output.parent,
        field="materialized HF output parent",
    )
    output_candidate = output_parent / output.name
    if (
        output_candidate == cache_root
        or output_candidate.is_relative_to(cache_root)
        or cache_root.is_relative_to(output_candidate)
    ):
        raise HfDownloadProvenanceError(
            "HF cache and materialized output must be disjoint directory trees"
        )

    try:
        api = HfApi(endpoint=OFFICIAL_ENDPOINT, token=hf_token)
        info = api.repo_info(
            repository,
            revision=revision,
            repo_type="model",
            files_metadata=True,
        )
    except Exception as exc:
        raise HfDownloadProvenanceError(
            "official Hub repository resolution failed"
        ) from exc
    resolved = getattr(info, "sha", None)
    remote_id = getattr(info, "id", None)
    if resolved != revision or remote_id != repository:
        raise HfDownloadProvenanceError("official Hub resolved a different repository or commit")
    try:
        remote_paths = api.list_repo_files(
            repository,
            revision=resolved,
            repo_type="model",
        )
    except Exception as exc:
        raise HfDownloadProvenanceError(
            "official Hub file inventory resolution failed"
        ) from exc
    paths = _remote_paths(remote_paths)
    remote_snapshot = _remote_snapshot(info, paths)

    output_candidate.mkdir(mode=0o755, exist_ok=False)
    output = output_candidate
    output_identity = _directory_identity(output, field="materialized HF output")
    try:
        for relative in paths:
            _assert_directory_identity(cache_root, cache_identity, field="HF cache")
            _assert_directory_identity(
                output,
                output_identity,
                field="materialized HF output",
            )
            try:
                downloaded = hf_hub_download(
                    repo_id=repository,
                    filename=relative,
                    repo_type="model",
                    revision=resolved,
                    cache_dir=cache_dir,
                    endpoint=OFFICIAL_ENDPOINT,
                    token=hf_token,
                )
                cached = Path(downloaded)
                resolved_cached = cached.resolve(strict=True)
            except Exception as exc:
                raise HfDownloadProvenanceError(
                    f"official Hub download failed for {relative!r}"
                ) from exc
            _assert_directory_identity(cache_root, cache_identity, field="HF cache")
            if not resolved_cached.is_relative_to(cache_root):
                raise HfDownloadProvenanceError("HF cache returned a file outside its root")
            try:
                digest, _size = successor._hash_regular_file(
                    resolved_cached, field=f"downloaded HF file {relative}"
                )
                _verify_download_metadata(
                    resolved_cached,
                    relative=relative,
                    sha256=digest,
                    size=_size,
                    remote_binding=remote_snapshot["files"][relative],
                )
                successor._copy_regular_file(
                    resolved_cached,
                    output / relative,
                    expected_sha256=digest,
                )
                cached_metadata = os.lstat(resolved_cached)
                output_metadata = os.lstat(output / relative)
            except (OSError, successor.PublicationSuccessorError) as exc:
                raise HfDownloadProvenanceError(
                    f"downloaded HF file {relative!r} could not be copied safely"
                ) from exc
            if (
                not stat.S_ISREG(cached_metadata.st_mode)
                or not stat.S_ISREG(output_metadata.st_mode)
                or (cached_metadata.st_dev, cached_metadata.st_ino)
                == (output_metadata.st_dev, output_metadata.st_ino)
            ):
                raise HfDownloadProvenanceError(
                    f"materialized HF file {relative!r} is not an independent regular file"
                )
        _assert_directory_identity(cache_root, cache_identity, field="HF cache")
        _assert_directory_identity(
            output,
            output_identity,
            field="materialized HF output",
        )
        _assert_directory_identity(
            output_parent,
            output_parent_identity,
            field="materialized HF output parent",
        )
        inventory = _inventory(output)
        if set(inventory) != set(paths):
            raise HfDownloadProvenanceError(
                "materialized HF inventory differs from the remote inventory"
            )
        successor.verify_successor_package(output)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "provider": "huggingface_hub",
            "endpoint": OFFICIAL_ENDPOINT,
            "repository": repository,
            "requested_revision": revision,
            "resolved_commit": resolved,
            "downloaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "remote_snapshot": remote_snapshot,
            "generator": {
                "path": GENERATOR_PATH.as_posix(),
                "file_sha256": _generator_sha256(),
            },
            "local_root": {
                "format": "materialized_regular_file_tree_v1",
                "file_count": len(inventory),
                "inventory_sha256": successor.canonical_json_hash(inventory),
                "files": inventory,
            },
            "receipt_sha256": "",
        }
        receipt["receipt_sha256"] = successor.canonical_json_hash(
            receipt, remove="receipt_sha256"
        )
        _validate_document(receipt)
        return receipt
    except BaseException:
        successor._safe_remove_created_directory(output, output_identity)
        raise


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise HfDownloadProvenanceError("HF provenance output already exists")
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    parent, parent_identity = _real_directory_path(
        path.parent,
        field="HF provenance parent",
    )
    path = parent / path.name
    descriptor = -1
    created_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        before = os.fstat(descriptor)
        created_identity = before.st_dev, before.st_ino
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        path_metadata = os.lstat(path)
        _assert_directory_identity(
            path.parent,
            parent_identity,
            field="HF provenance parent",
        )
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_size != len(payload)
            or (after.st_dev, after.st_ino) != created_identity
            or stat.S_IMODE(after.st_mode) != 0o444
            or stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or (path_metadata.st_dev, path_metadata.st_ino) != created_identity
        ):
            raise HfDownloadProvenanceError(
                "HF provenance output changed while it was written"
            )
    except BaseException:
        if created_identity is not None:
            try:
                metadata = os.lstat(path)
                if (metadata.st_dev, metadata.st_ino) == created_identity:
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path, help="new materialized package root")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--provenance-output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    created_output_identity: tuple[int, int] | None = None
    try:
        if args.provenance_output.exists() or args.provenance_output.is_symlink():
            raise HfDownloadProvenanceError("HF provenance output already exists")
        output_candidate = args.output.resolve(strict=False)
        provenance_candidate = args.provenance_output.resolve(strict=False)
        if provenance_candidate == output_candidate or provenance_candidate.is_relative_to(
            output_candidate
        ):
            raise HfDownloadProvenanceError(
                "HF provenance output must be outside the materialized snapshot"
            )
        receipt = materialize_snapshot(
            repository=args.repository,
            revision=args.revision,
            output=args.output,
            cache_dir=args.cache_dir,
        )
        output_metadata = os.lstat(args.output)
        if stat.S_ISLNK(output_metadata.st_mode) or not stat.S_ISDIR(
            output_metadata.st_mode
        ):
            raise HfDownloadProvenanceError(
                "materialized HF output changed before provenance emission"
            )
        created_output_identity = output_metadata.st_dev, output_metadata.st_ino
        _write_new(args.provenance_output, receipt)
    except (
        OSError,
        ValueError,
        HfDownloadProvenanceError,
        successor.PublicationSuccessorError,
    ) as exc:
        if created_output_identity is not None:
            successor._safe_remove_created_directory(
                args.output,
                created_output_identity,
            )
        print(f"HF immutable download blocked: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "MATERIALIZED",
                "repository": receipt["repository"],
                "resolved_commit": receipt["resolved_commit"],
                "receipt_sha256": receipt["receipt_sha256"],
                "remote_write_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
