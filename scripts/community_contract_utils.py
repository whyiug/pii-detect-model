#!/usr/bin/env python3
"""Small fail-closed helpers shared by community release validators.

The helpers intentionally operate on bounded repository-relative manifest files.
They never follow symlinks, inspect model weights, read record-level data, use a
GPU, contact a network service, or query a Git remote.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

MAX_METADATA_BYTES = 64 * 1024 * 1024


class CommunityContractError(RuntimeError):
    """Raised when community evidence cannot be checked safely."""


def canonical_json_hash(value: Mapping[str, Any], *, remove: str | None = None) -> str:
    document = dict(value)
    if remove is not None:
        document.pop(remove, None)
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CommunityContractError("metadata is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def strict_json_bytes(payload: bytes, *, field: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CommunityContractError(f"{field} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise CommunityContractError(f"{field} has a non-finite JSON number")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommunityContractError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CommunityContractError(f"{field} must be a JSON object")
    return value


def safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise CommunityContractError(f"{field} is not a safe repository-relative path")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CommunityContractError(f"{field} is not a safe repository-relative path")
    return value


def _open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def read_regular_file(path: Path, *, field: str) -> bytes:
    try:
        descriptor = os.open(path, _open_flags())
    except OSError as exc:
        raise CommunityContractError(f"{field} is unavailable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA_BYTES:
            raise CommunityContractError(f"{field} is not a bounded regular file")
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
            if len(payload) > MAX_METADATA_BYTES:
                raise CommunityContractError(f"{field} exceeds the size limit")
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
        if identity_before != identity_after or len(payload) != after.st_size:
            raise CommunityContractError(f"{field} changed while it was read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def read_repository_file(root: Path, relative: object, *, field: str) -> bytes:
    safe = safe_relative_path(relative, field=field)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise CommunityContractError("repository root is unavailable") from exc
    if root.is_symlink() or not resolved_root.is_dir():
        raise CommunityContractError("repository root is unsafe")
    current = resolved_root
    for part in PurePosixPath(safe).parts[:-1]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise CommunityContractError(f"{field} ancestor is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CommunityContractError(f"{field} traverses an unsafe ancestor")
    return read_regular_file(
        resolved_root.joinpath(*PurePosixPath(safe).parts),
        field=field,
    )


def load_json_path(path: Path, *, field: str) -> dict[str, Any]:
    return strict_json_bytes(read_regular_file(path, field=field), field=field)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_binding(root: Path, binding: Mapping[str, Any], *, field: str) -> bytes:
    if set(binding) != {"logical_id", "path", "file_sha256"}:
        raise CommunityContractError(f"{field} has an invalid binding shape")
    logical_id = binding.get("logical_id")
    digest = binding.get("file_sha256")
    if not isinstance(logical_id, str) or not logical_id:
        raise CommunityContractError(f"{field}.logical_id is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CommunityContractError(f"{field}.file_sha256 is invalid")
    payload = read_repository_file(root, binding.get("path"), field=field)
    if sha256_bytes(payload) != digest:
        raise CommunityContractError(f"{field} byte hash does not match")
    return payload


def validate_schema(document: Mapping[str, Any], schema: Mapping[str, Any], *, field: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise CommunityContractError(f"{field} does not satisfy its closed schema") from exc


def require_finite_unit_interval(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommunityContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CommunityContractError(f"{field} must be in [0, 1]")
    return result


def require_nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommunityContractError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CommunityContractError(f"{field} must be a finite non-negative number")
    return result


def close_enough(actual: float, expected: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
