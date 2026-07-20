#!/usr/bin/env python3
"""Verify the fixed GitHub Release asset trio and write an offline receipt.

The command accepts only the release wheel, deterministic CycloneDX SBOM,
two-line ``checksums.txt``, and the exact local source commit.  It derives all
PASS states itself: the wheel is inventoried with the existing typed wheel
implementation, installed into a temporary venv and exercised by the existing
clean-wheel harness inside an unshared bubblewrap sandbox, and deep-scanned by
the existing public-artifact scanner.  It never contacts GitHub, Hugging Face,
or another network service and never accepts caller-supplied PASS evidence.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

try:
    from scripts import generate_sbom, scan_public_artifacts
    from scripts import produce_community_cascade_release_v2_artifacts as wheel_artifacts
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    import generate_sbom  # type: ignore[no-redef]
    import produce_community_cascade_release_v2_artifacts as wheel_artifacts  # type: ignore[no-redef]
    import scan_public_artifacts  # type: ignore[no-redef]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = Path("scripts/build_community_v2_github_release_assets_receipt.py")
SCHEMA_PATH = Path("configs/release/community_v2_github_release_assets_receipt.schema.json")
WHEEL_INVENTORY_PATH = Path("scripts/produce_community_cascade_release_v2_artifacts.py")
SBOM_GENERATOR_PATH = Path("scripts/generate_sbom.py")
PUBLIC_SCANNER_PATH = Path("scripts/scan_public_artifacts.py")
SMOKE_HARNESS_PATH = Path("scripts/run_successor_clean_wheel_smoke.py")
LOCKFILE_PATH = Path("uv.lock")
PYPROJECT_PATH = Path("pyproject.toml")

SCHEMA_VERSION = "pii-zh.community-v2-github-release-assets-receipt.v1"
GITHUB_REPOSITORY = "whyiug/pii-detect-model"
PACKAGE_NAME = "pii-zh-qwen"
PACKAGE_VERSION = "0.2.0rc1"
TAG_NAME = "v0.2.0rc1"
WHEEL_NAME = "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
SBOM_NAME = "sbom.cdx.json"
CHECKSUMS_NAME = "checksums.txt"
ASSET_NAMES = (WHEEL_NAME, SBOM_NAME, CHECKSUMS_NAME)
REQUIRED_WHEEL_MEMBERS = (
    "pii_zh:cli.py",
    "pii_zh:service:app.py",
    "pii_zh:cascade:routing.py",
    "pii_zh:taxonomy:taxonomy.yaml",
)
SOURCE_COMMIT_PATHS = (
    GENERATOR_PATH,
    SCHEMA_PATH,
    WHEEL_INVENTORY_PATH,
    SBOM_GENERATOR_PATH,
    PUBLIC_SCANNER_PATH,
    SMOKE_HARNESS_PATH,
    LOCKFILE_PATH,
    PYPROJECT_PATH,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_.+-]{0,191})$")
_MAX_WHEEL_BYTES = 256 * 1024 * 1024
_MAX_SBOM_BYTES = 64 * 1024 * 1024
_MAX_CHECKSUM_BYTES = 1024 * 1024
_MAX_REPOSITORY_FILE_BYTES = 64 * 1024 * 1024
_MAX_PROCESS_OUTPUT_BYTES = 1024 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class ReleaseAssetsReceiptError(RuntimeError):
    """Raised when the fixed release asset set cannot be verified safely."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class FileObservation:
    path: Path
    file_sha256: str
    size_bytes: int
    maximum_bytes: int
    field: str


@dataclass(frozen=True)
class AssetPayload:
    path: Path
    payload: bytes
    file_sha256: str
    size_bytes: int

    @property
    def binding(self) -> dict[str, int | str]:
        return {"file_sha256": self.file_sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ReceiptPlan:
    output: Path
    repository_root: Path
    source_commit: str
    document: Mapping[str, Any]
    serialized: bytes
    observations: tuple[FileObservation, ...]


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
        raise ReleaseAssetsReceiptError(
            "INVALID_CANONICAL_JSON", "receipt is not canonical-JSON serializable"
        ) from exc


def canonical_json_hash(value: Mapping[str, Any], *, remove: str | None = None) -> str:
    """Return a lowercase SHA-256 over canonical UTF-8 JSON."""

    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _lexical_absolute(path: Path, *, field: str) -> Path:
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ReleaseAssetsReceiptError(
            "UNSAFE_PATH", f"{field} cannot contain parent-directory traversal"
        )
    return expanded if expanded.is_absolute() else (Path.cwd() / expanded)


def _reject_symlink_ancestry(path: Path, *, field: str, include_leaf: bool = True) -> None:
    absolute = _lexical_absolute(path, field=field)
    candidates = list(absolute.parents)
    candidates.reverse()
    if include_leaf:
        candidates.append(absolute)
    for candidate in candidates:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            if candidate == absolute and not include_leaf:
                return
            raise ReleaseAssetsReceiptError(
                "PATH_UNAVAILABLE", f"{field} has an unavailable path component"
            ) from None
        except OSError as exc:
            raise ReleaseAssetsReceiptError(
                "PATH_INSPECTION_FAILED", f"{field} cannot be inspected safely"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseAssetsReceiptError(
                "SYMLINK_REJECTED", f"{field} cannot traverse a symlink"
            )


def _open_read_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _read_regular(path: Path, *, field: str, maximum_bytes: int) -> AssetPayload:
    absolute = _lexical_absolute(path, field=field)
    _reject_symlink_ancestry(absolute, field=field)
    try:
        before_path = os.lstat(absolute)
        descriptor = os.open(absolute, _open_read_flags())
    except OSError as exc:
        blocker = "SYMLINK_REJECTED" if exc.errno == errno.ELOOP else "INPUT_READ_FAILED"
        raise ReleaseAssetsReceiptError(blocker, f"{field} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise ReleaseAssetsReceiptError("INPUT_NOT_REGULAR", f"{field} must be a regular file")
        if (before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise ReleaseAssetsReceiptError(
                "INPUT_CHANGED_DURING_OPEN", f"{field} changed while opening"
            )
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ReleaseAssetsReceiptError(
                "INPUT_SIZE_INVALID", f"{field} is empty or exceeds its size limit"
            )
        payload = bytearray()
        while block := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(payload))):
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise ReleaseAssetsReceiptError(
                    "INPUT_SIZE_INVALID", f"{field} exceeds its size limit"
                )
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise ReleaseAssetsReceiptError(
                "INPUT_CHANGED_DURING_READ", f"{field} changed while reading"
            )
        if len(payload) != after.st_size:
            raise ReleaseAssetsReceiptError(
                "INPUT_CHANGED_DURING_READ", f"{field} size changed while reading"
            )
    finally:
        os.close(descriptor)
    try:
        final_path = os.lstat(absolute)
    except OSError as exc:
        raise ReleaseAssetsReceiptError(
            "INPUT_CHANGED_DURING_READ", f"{field} disappeared while reading"
        ) from exc
    if (final_path.st_dev, final_path.st_ino) != (before.st_dev, before.st_ino):
        raise ReleaseAssetsReceiptError(
            "INPUT_CHANGED_DURING_READ", f"{field} identity changed while reading"
        )
    data = bytes(payload)
    return AssetPayload(
        path=absolute,
        payload=data,
        file_sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _observation(asset: AssetPayload, *, maximum_bytes: int, field: str) -> FileObservation:
    return FileObservation(
        path=asset.path,
        file_sha256=asset.file_sha256,
        size_bytes=asset.size_bytes,
        maximum_bytes=maximum_bytes,
        field=field,
    )


def _read_repository_file(repository_root: Path, relative: Path, *, field: str) -> AssetPayload:
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseAssetsReceiptError(
            "INVALID_REPOSITORY_PATH", f"{field} path is not repository-relative"
        )
    return _read_regular(
        repository_root.joinpath(*relative.parts),
        field=field,
        maximum_bytes=_MAX_REPOSITORY_FILE_BYTES,
    )


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseAssetsReceiptError(
                    "DUPLICATE_JSON_KEY", f"{field} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ReleaseAssetsReceiptError(
            "NONFINITE_JSON_NUMBER", f"{field} contains a non-finite JSON number"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ReleaseAssetsReceiptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAssetsReceiptError(
            "INVALID_JSON", f"{field} must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseAssetsReceiptError("INVALID_JSON", f"{field} must be a JSON object")
    return value


def _schema(repository_root: Path) -> tuple[Mapping[str, Any], AssetPayload]:
    payload = _read_repository_file(
        repository_root, SCHEMA_PATH, field="release-assets receipt schema"
    )
    schema = _strict_json(payload.payload, field="release-assets receipt schema")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ReleaseAssetsReceiptError(
            "SCHEMA_INVALID", "release-assets receipt schema is invalid"
        ) from exc
    return schema, payload


def _validate_document(document: Mapping[str, Any], *, repository_root: Path) -> None:
    schema, _payload = _schema(repository_root)
    try:
        Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise ReleaseAssetsReceiptError(
            "RECEIPT_SCHEMA_REJECTED", "release-assets receipt violates its closed schema"
        ) from exc
    if document.get("receipt_sha256") != canonical_json_hash(document, remove="receipt_sha256"):
        raise ReleaseAssetsReceiptError(
            "RECEIPT_SELF_HASH_MISMATCH", "release-assets receipt self-hash does not verify"
        )


def _git_environment(git_binary: str) -> dict[str, str]:
    return {
        "PATH": str(Path(git_binary).parent),
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run_git(repository_root: Path, arguments: Sequence[str], *, maximum: int) -> bytes:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise ReleaseAssetsReceiptError("GIT_UNAVAILABLE", "local git executable is unavailable")
    try:
        completed = subprocess.run(
            [git_binary, "-C", str(repository_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
            env=_git_environment(git_binary),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseAssetsReceiptError(
            "GIT_READ_FAILED", "local git evidence could not be read"
        ) from exc
    if completed.returncode != 0 or len(completed.stdout) > maximum:
        raise ReleaseAssetsReceiptError(
            "GIT_READ_FAILED", "local git evidence is missing or exceeds its limit"
        )
    return completed.stdout


def _verify_source_context(repository_root: Path, source_commit: str) -> None:
    if _GIT_COMMIT.fullmatch(source_commit) is None:
        raise ReleaseAssetsReceiptError(
            "INVALID_SOURCE_COMMIT", "source commit must be a full lowercase commit SHA"
        )
    root = _lexical_absolute(repository_root, field="repository root")
    _reject_symlink_ancestry(root, field="repository root")
    if not root.is_dir():
        raise ReleaseAssetsReceiptError(
            "REPOSITORY_UNAVAILABLE", "repository root must be a real directory"
        )
    observed_root = _run_git(root, ("rev-parse", "--show-toplevel"), maximum=8192)
    try:
        declared_root = Path(observed_root.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise ReleaseAssetsReceiptError(
            "REPOSITORY_IDENTITY_INVALID", "git returned an invalid repository identity"
        ) from exc
    if declared_root != root:
        raise ReleaseAssetsReceiptError(
            "REPOSITORY_IDENTITY_INVALID", "repository root is not the exact git worktree root"
        )
    observed_head = (
        _run_git(root, ("rev-parse", "--verify", "HEAD^{commit}"), maximum=128)
        .decode("ascii", errors="strict")
        .strip()
    )
    if observed_head != source_commit:
        raise ReleaseAssetsReceiptError(
            "SOURCE_COMMIT_NOT_HEAD", "source commit is not the current repository HEAD"
        )
    for relative in SOURCE_COMMIT_PATHS:
        working = _read_repository_file(root, relative, field="source-bound implementation file")
        committed = _run_git(
            root,
            ("cat-file", "blob", f"{source_commit}:{relative.as_posix()}"),
            maximum=_MAX_REPOSITORY_FILE_BYTES,
        )
        if committed != working.payload:
            raise ReleaseAssetsReceiptError(
                "SOURCE_IMPLEMENTATION_DRIFT",
                "a verification implementation or SBOM input differs from the source commit",
            )


def _parse_checksums(
    checksums: AssetPayload, wheel: AssetPayload, sbom: AssetPayload
) -> list[dict[str, str]]:
    payload = checksums.payload
    if b"\r" in payload or not payload.endswith(b"\n") or payload.startswith(b"\xef\xbb\xbf"):
        raise ReleaseAssetsReceiptError(
            "CHECKSUM_FORMAT_INVALID", "checksums.txt must use exact UTF-8 LF formatting"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseAssetsReceiptError(
            "CHECKSUM_FORMAT_INVALID", "checksums.txt must be UTF-8"
        ) from exc
    lines = text.removesuffix("\n").split("\n")
    if len(lines) != 2:
        raise ReleaseAssetsReceiptError(
            "CHECKSUM_CLOSURE_INVALID", "checksums.txt must contain exactly two lines"
        )
    expected = ((WHEEL_NAME, wheel.file_sha256), (SBOM_NAME, sbom.file_sha256))
    entries: list[dict[str, str]] = []
    for line, (expected_name, expected_digest) in zip(lines, expected, strict=True):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ReleaseAssetsReceiptError(
                "CHECKSUM_FORMAT_INVALID", "checksums.txt has a non-canonical line"
            )
        digest, name = match.groups()
        if name != expected_name or digest != expected_digest:
            raise ReleaseAssetsReceiptError(
                "CHECKSUM_BINDING_MISMATCH",
                "checksums.txt does not bind the exact ordered wheel and SBOM bytes",
            )
        entries.append({"name": name, "file_sha256": digest})
    return entries


def _write_private_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _wheel_inventory(
    wheel: AssetPayload, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, int | str]]]:
    try:
        with tempfile.TemporaryDirectory(prefix="pii-release-wheel-inventory-") as temporary:
            wheel_copy = Path(temporary) / WHEEL_NAME
            _write_private_file(wheel_copy, wheel.payload)
            inventory, name, version = wheel_artifacts._wheel_inventory(wheel_copy)
    except (OSError, wheel_artifacts.CommunityReleaseArtifactError) as exc:
        raise ReleaseAssetsReceiptError(
            "WHEEL_INVENTORY_FAILED", "release wheel inventory validation failed"
        ) from exc
    if name != PACKAGE_NAME or version != PACKAGE_VERSION:
        raise ReleaseAssetsReceiptError(
            "WHEEL_IDENTITY_MISMATCH", "release wheel distribution identity is invalid"
        )
    if any(member not in inventory for member in REQUIRED_WHEEL_MEMBERS):
        raise ReleaseAssetsReceiptError(
            "WHEEL_REQUIRED_MEMBER_MISSING", "release wheel omits a required runtime member"
        )
    inventory_implementation = _read_repository_file(
        repository_root, WHEEL_INVENTORY_PATH, field="wheel inventory implementation"
    )
    result = {
        "status": "PASS",
        "format": "python_wheel_zip_v1",
        "distribution_name": name,
        "distribution_version": version,
        "member_count": len(inventory),
        "member_inventory_sha256": canonical_json_hash(inventory),
        "members": inventory,
        "required_member_ids": list(REQUIRED_WHEEL_MEMBERS),
        "required_members_present": True,
        "inventory_implementation_path": WHEEL_INVENTORY_PATH.as_posix(),
        "inventory_implementation_file_sha256": inventory_implementation.file_sha256,
    }
    return result, inventory


def _verify_sbom(sbom: AssetPayload, *, repository_root: Path) -> dict[str, Any]:
    bom = _strict_json(sbom.payload, field="CycloneDX SBOM")
    lockfile = _read_repository_file(repository_root, LOCKFILE_PATH, field="uv lockfile")
    pyproject = _read_repository_file(repository_root, PYPROJECT_PATH, field="project metadata")
    try:
        expected = generate_sbom.build_sbom(
            repository_root / LOCKFILE_PATH, repository_root / PYPROJECT_PATH
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseAssetsReceiptError(
            "SBOM_REGENERATION_FAILED", "deterministic SBOM regeneration failed"
        ) from exc
    expected_payload = (
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if sbom.payload != expected_payload or bom != expected:
        raise ReleaseAssetsReceiptError(
            "SBOM_SOURCE_MISMATCH",
            "SBOM bytes do not exactly match deterministic source regeneration",
        )
    metadata = bom.get("metadata")
    root_component = metadata.get("component") if isinstance(metadata, Mapping) else None
    components = bom.get("components")
    dependencies = bom.get("dependencies")
    if (
        bom.get("bomFormat") != "CycloneDX"
        or bom.get("specVersion") != "1.6"
        or not isinstance(root_component, Mapping)
        or root_component.get("name") != PACKAGE_NAME
        or root_component.get("version") != PACKAGE_VERSION
        or not isinstance(components, list)
        or not components
        or not isinstance(dependencies, list)
    ):
        raise ReleaseAssetsReceiptError(
            "SBOM_SEMANTIC_INVALID", "SBOM lacks the fixed CycloneDX package closure"
        )
    edge_count = 0
    for item in dependencies:
        if not isinstance(item, Mapping) or not isinstance(item.get("dependsOn"), list):
            raise ReleaseAssetsReceiptError(
                "SBOM_SEMANTIC_INVALID", "SBOM dependency graph is malformed"
            )
        edge_count += len(item["dependsOn"])
    generator = _read_repository_file(repository_root, SBOM_GENERATOR_PATH, field="SBOM generator")
    return {
        "status": "PASS",
        "format": "CycloneDX-1.6",
        "bom_sha256": canonical_json_hash(bom),
        "component_count": len(components),
        "dependency_edge_count": edge_count,
        "deterministic_source_regeneration_match": True,
        "lockfile_file_sha256": lockfile.file_sha256,
        "pyproject_file_sha256": pyproject.file_sha256,
        "generator_path": SBOM_GENERATOR_PATH.as_posix(),
        "generator_file_sha256": generator.file_sha256,
    }


def _extract_wheel_for_scan(wheel: AssetPayload, target: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(wheel.payload)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                wheel_artifacts._wheel_logical_id(info.filename)
                member = PurePosixPath(info.filename)
                destination = target.joinpath(*member.parts)
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                payload = archive.read(info)
                _write_private_file(destination, payload)
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseAssetsReceiptError(
            "PUBLIC_SCAN_EXTRACTION_FAILED", "release wheel could not be staged for deep scan"
        ) from exc


def _public_scan(
    wheel: AssetPayload,
    sbom: AssetPayload,
    checksums: AssetPayload,
    wheel_inventory: Mapping[str, Mapping[str, int | str]],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="pii-release-assets-scan-") as temporary:
            root = Path(temporary)
            extracted = root / "wheel"
            extracted.mkdir(mode=0o700)
            _extract_wheel_for_scan(wheel, extracted)
            _write_private_file(root / SBOM_NAME, sbom.payload)
            _write_private_file(root / CHECKSUMS_NAME, checksums.payload)
            findings = scan_public_artifacts.scan_paths(
                [extracted, root / SBOM_NAME, root / CHECKSUMS_NAME]
            )
    except ReleaseAssetsReceiptError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReleaseAssetsReceiptError(
            "PUBLIC_SCAN_FAILED", "public artifact scanner could not complete"
        ) from exc
    if findings:
        kinds = sorted({finding.kind for finding in findings})
        raise ReleaseAssetsReceiptError(
            "PUBLIC_SCAN_FINDINGS",
            f"public artifact scan found {len(findings)} redacted finding(s): {kinds}",
        )
    scan_inventory: dict[str, Mapping[str, int | str]] = {
        f"wheel:{key}": value for key, value in wheel_inventory.items()
    }
    scan_inventory[f"asset:{SBOM_NAME}"] = sbom.binding
    scan_inventory[f"asset:{CHECKSUMS_NAME}"] = checksums.binding
    scan_inventory = dict(sorted(scan_inventory.items()))
    scanner = _read_repository_file(
        repository_root, PUBLIC_SCANNER_PATH, field="public artifact scanner"
    )
    return {
        "status": "PASS",
        "format": "pii_zh_public_artifact_scan_v3",
        "scanner_path": PUBLIC_SCANNER_PATH.as_posix(),
        "scanner_file_sha256": scanner.file_sha256,
        "scanned_file_count": len(scan_inventory),
        "scanned_inventory_sha256": canonical_json_hash(scan_inventory),
        "finding_count": 0,
        "finding_kinds": [],
    }


def _offline_environment(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
    }


def _run_local_command(
    arguments: Sequence[str], *, cwd: Path, environment: Mapping[str, str], timeout: int
) -> tuple[bytes, bytes]:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_SETUP_FAILED", "isolated wheel smoke setup did not complete"
        ) from exc
    if (
        completed.returncode != 0
        or len(completed.stdout) > _MAX_PROCESS_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_PROCESS_OUTPUT_BYTES
    ):
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_SETUP_FAILED", "isolated wheel smoke setup failed closed"
        )
    return completed.stdout, completed.stderr


def _sandbox_parent_directories(paths: Sequence[Path]) -> list[str]:
    directories: set[str] = set()
    for path in paths:
        current = path.parent
        while current != current.parent and current != Path("/"):
            directories.add(str(current))
            current = current.parent
    arguments: list[str] = []
    for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
        arguments.extend(("--dir", directory))
    return arguments


def _run_sandboxed_harness(
    *, venv_root: Path, harness: Path, runtime_cwd: Path, temporary_root: Path
) -> tuple[bytes, bytes]:
    bubblewrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    if bubblewrap is None or prlimit is None:
        raise ReleaseAssetsReceiptError(
            "ISOLATION_TOOL_UNAVAILABLE", "bwrap and prlimit are required for wheel smoke"
        )
    runtime_prefixes = {
        Path(sys.prefix).resolve(strict=True),
        Path(sys.base_prefix).resolve(strict=True),
    }
    mounts = [path for path in (Path("/usr"), Path("/lib"), Path("/lib64")) if path.exists()]
    mounts.extend(sorted(runtime_prefixes, key=str))
    command = [
        prlimit,
        "--as=8589934592",
        "--cpu=300",
        "--nofile=512",
        "--core=0",
        "--",
        bubblewrap,
        "--unshare-all",
        "--die-with-parent",
    ]
    command.extend(_sandbox_parent_directories([*mounts, temporary_root]))
    for mount in mounts:
        command.extend(("--ro-bind", str(mount), str(mount)))
    command.extend(
        (
            "--bind",
            str(temporary_root),
            str(temporary_root),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            str(runtime_cwd),
            "--clearenv",
        )
    )
    sandbox_environment = _offline_environment(temporary_root / "home")
    for key, value in sandbox_environment.items():
        command.extend(("--setenv", key, value))
    command.extend((str(venv_root / "bin" / "python"), "-I", str(harness)))

    stdout_path = temporary_root / "smoke.stdout"
    stderr_path = temporary_root / "smoke.stderr"
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=runtime_cwd,
                env=_offline_environment(temporary_root / "home"),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=330)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
                raise ReleaseAssetsReceiptError(
                    "ISOLATED_SMOKE_TIMEOUT", "isolated wheel smoke exceeded its time limit"
                ) from exc
        except ReleaseAssetsReceiptError:
            raise
        except OSError as exc:
            raise ReleaseAssetsReceiptError(
                "ISOLATED_SMOKE_FAILED", "isolated wheel smoke could not start"
            ) from exc
    stdout = _read_regular(
        stdout_path, field="isolated smoke stdout", maximum_bytes=_MAX_PROCESS_OUTPUT_BYTES
    ).payload
    try:
        stderr = stderr_path.read_bytes()
    except OSError as exc:
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_FAILED", "isolated wheel smoke stderr could not be read"
        ) from exc
    if len(stderr) > _MAX_PROCESS_OUTPUT_BYTES or return_code != 0:
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_FAILED", "isolated wheel smoke returned a non-PASS status"
        )
    return stdout, stderr


def _validate_smoke_result(stdout: bytes, stderr: bytes) -> dict[str, Any]:
    if stderr:
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_STDERR", "isolated wheel smoke produced unexpected stderr"
        )
    result = _strict_json(stdout.strip(), field="isolated wheel smoke result")
    expected_keys = {
        "status",
        "profile_version",
        "historical_default_unchanged",
        "installed_from_clean_site_packages",
        "installed_module_path_recorded",
        "python",
        "distribution_versions",
        "paths",
        "installed_entrypoints",
        "historical_paths",
        "valid_case_count",
        "invalid_document_count",
        "raw_values_persisted",
        "model_packages_imported",
        "framework_torch_imported",
        "framework_transformers_imported",
        "gpu_visible",
    }
    versions = result.get("distribution_versions")
    if set(result) != expected_keys or not isinstance(versions, Mapping):
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_CONTRACT_DRIFT", "clean-wheel smoke result shape changed"
        )
    if (
        result.get("status") != "PASS"
        or result.get("profile_version") != "c1-conservative-v2"
        or result.get("historical_default_unchanged") is not True
        or result.get("installed_from_clean_site_packages") is not True
        or result.get("installed_module_path_recorded") is not False
        or result.get("paths")
        != ["python", "installed-cli", "installed-ablation-cli-help", "presidio", "http"]
        or result.get("historical_paths") != ["python", "installed-cli", "presidio", "http"]
        or result.get("installed_entrypoints")
        != {
            "pii-zh": "pii_zh.cli:main",
            "pii-zh-evaluate-ablation": "pii_zh.evaluation.cascade_ablation_cli:main",
        }
        or result.get("valid_case_count") != 3
        or result.get("invalid_document_count") != 1
        or result.get("raw_values_persisted") is not False
        or result.get("model_packages_imported") is not False
        or result.get("framework_torch_imported") is not False
        or result.get("framework_transformers_imported") is not False
        or result.get("gpu_visible") != ""
        or set(versions) != {"pii-zh-qwen", "presidio-analyzer", "fastapi", "httpx"}
        or versions.get("pii-zh-qwen") != PACKAGE_VERSION
    ):
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_ASSERTION_FAILED", "clean-wheel smoke assertions did not pass"
        )
    runtime_values = {"python": result.get("python"), **versions}
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(character in value for character in "\r\n\x00")
        for value in runtime_values.values()
    ):
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_VERSION_INVALID", "clean-wheel smoke runtime versions are invalid"
        )
    return {
        "status": "PASS",
        "profile_version": result["profile_version"],
        "harness_result_sha256": canonical_json_hash(result),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_empty": True,
        "runtime_versions": runtime_values,
        "assertions": {
            "historical_default_unchanged": True,
            "installed_from_clean_site_packages": True,
            "raw_values_persisted": False,
            "model_packages_imported": False,
            "torch_imported": False,
            "transformers_imported": False,
            "gpu_visible": "",
            "valid_case_count": 3,
            "invalid_document_count": 1,
        },
        "isolation": {
            "profile": "bwrap_unshare_all_prlimit_temporary_venv_v1",
            "temporary_venv": True,
            "source_tree_import_disabled": True,
            "pip_no_index": True,
            "pip_no_dependencies": True,
            "inherited_dependency_environment": True,
            "bubblewrap_unshare_all": True,
            "network_namespace_unshared": True,
            "gpu_devices_hidden": True,
            "resource_limits_enforced": True,
        },
    }


def _run_isolated_wheel_smoke(wheel: AssetPayload, *, repository_root: Path) -> dict[str, Any]:
    harness_payload = _read_repository_file(
        repository_root, SMOKE_HARNESS_PATH, field="clean-wheel smoke harness"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="pii-release-assets-smoke-") as temporary:
            root = Path(temporary)
            home = root / "home"
            runtime = root / "runtime"
            home.mkdir(mode=0o700)
            runtime.mkdir(mode=0o700)
            wheel_copy = root / WHEEL_NAME
            harness_copy = root / "run_successor_clean_wheel_smoke.py"
            _write_private_file(wheel_copy, wheel.payload)
            _write_private_file(harness_copy, harness_payload.payload)
            venv_root = root / "venv"
            environment = _offline_environment(home)
            _run_local_command(
                (
                    sys.executable,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(venv_root),
                ),
                cwd=runtime,
                environment=environment,
                timeout=300,
            )
            _run_local_command(
                (
                    str(venv_root / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--force-reinstall",
                    str(wheel_copy),
                ),
                cwd=runtime,
                environment=environment,
                timeout=300,
            )
            stdout, stderr = _run_sandboxed_harness(
                venv_root=venv_root,
                harness=harness_copy,
                runtime_cwd=runtime,
                temporary_root=root,
            )
    except ReleaseAssetsReceiptError:
        raise
    except OSError as exc:
        raise ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_FAILED", "isolated wheel smoke staging failed"
        ) from exc
    evidence = _validate_smoke_result(stdout, stderr)
    evidence["harness_path"] = SMOKE_HARNESS_PATH.as_posix()
    evidence["harness_file_sha256"] = harness_payload.file_sha256
    return evidence


def _validate_asset_paths(
    wheel: Path, sbom: Path, checksums: Path, output: Path
) -> tuple[Path, Path, Path, Path]:
    assets = (
        _lexical_absolute(wheel, field="release wheel"),
        _lexical_absolute(sbom, field="release SBOM"),
        _lexical_absolute(checksums, field="release checksums"),
    )
    expected_names = (WHEEL_NAME, SBOM_NAME, CHECKSUMS_NAME)
    if tuple(path.name for path in assets) != expected_names:
        raise ReleaseAssetsReceiptError(
            "ASSET_NAME_MISMATCH", "release asset filenames are not the fixed RC trio"
        )
    if len({path.parent for path in assets}) != 1:
        raise ReleaseAssetsReceiptError(
            "ASSET_DIRECTORY_MISMATCH", "release assets must share one staging directory"
        )
    if len(set(assets)) != len(assets):
        raise ReleaseAssetsReceiptError(
            "ASSET_PATH_COLLISION", "release asset paths must be distinct"
        )
    output_absolute = _lexical_absolute(output, field="receipt output")
    if output_absolute in set(assets):
        raise ReleaseAssetsReceiptError(
            "OUTPUT_INPUT_COLLISION", "receipt output cannot overwrite a release asset"
        )
    return (*assets, output_absolute)


def _validate_output(output: Path) -> None:
    parent = output.parent
    _reject_symlink_ancestry(parent, field="receipt output parent")
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise ReleaseAssetsReceiptError(
            "OUTPUT_PARENT_UNAVAILABLE", "receipt output parent is unavailable"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ReleaseAssetsReceiptError(
            "OUTPUT_PARENT_UNSAFE", "receipt output parent must be a real directory"
        )
    try:
        os.lstat(output)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ReleaseAssetsReceiptError(
            "OUTPUT_INSPECTION_FAILED", "receipt output cannot be inspected"
        ) from exc
    raise ReleaseAssetsReceiptError("OUTPUT_ALREADY_EXISTS", "receipt output is no-clobber")


def prepare_receipt(
    *,
    wheel: Path,
    sbom: Path,
    checksums: Path,
    source_commit: str,
    output: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> ReceiptPlan:
    """Execute all offline checks and prepare, but do not write, one receipt."""

    wheel_path, sbom_path, checksums_path, output_path = _validate_asset_paths(
        wheel, sbom, checksums, output
    )
    root = _lexical_absolute(repository_root, field="repository root")
    _validate_output(output_path)
    _verify_source_context(root, source_commit)

    wheel_payload = _read_regular(wheel_path, field="release wheel", maximum_bytes=_MAX_WHEEL_BYTES)
    sbom_payload = _read_regular(sbom_path, field="release SBOM", maximum_bytes=_MAX_SBOM_BYTES)
    checksums_payload = _read_regular(
        checksums_path, field="release checksums", maximum_bytes=_MAX_CHECKSUM_BYTES
    )
    checksum_entries = _parse_checksums(checksums_payload, wheel_payload, sbom_payload)
    wheel_evidence, wheel_members = _wheel_inventory(wheel_payload, repository_root=root)
    sbom_evidence = _verify_sbom(sbom_payload, repository_root=root)
    smoke_evidence = _run_isolated_wheel_smoke(wheel_payload, repository_root=root)
    scan_evidence = _public_scan(
        wheel_payload,
        sbom_payload,
        checksums_payload,
        wheel_members,
        repository_root=root,
    )

    generator = _read_repository_file(root, GENERATOR_PATH, field="receipt generator")
    schema, schema_payload = _schema(root)
    del schema
    implementation_assets = {
        path: _read_repository_file(root, path, field="source-bound implementation file")
        for path in SOURCE_COMMIT_PATHS
    }
    observations = (
        _observation(wheel_payload, maximum_bytes=_MAX_WHEEL_BYTES, field="release wheel"),
        _observation(sbom_payload, maximum_bytes=_MAX_SBOM_BYTES, field="release SBOM"),
        _observation(
            checksums_payload,
            maximum_bytes=_MAX_CHECKSUM_BYTES,
            field="release checksums",
        ),
        *(
            _observation(
                payload,
                maximum_bytes=_MAX_REPOSITORY_FILE_BYTES,
                field="source-bound implementation file",
            )
            for payload in implementation_assets.values()
        ),
    )
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "release": {
            "package_name": PACKAGE_NAME,
            "package_version": PACKAGE_VERSION,
            "tag_name": TAG_NAME,
            "github_repository": GITHUB_REPOSITORY,
            "asset_count": 3,
            "asset_names": list(ASSET_NAMES),
        },
        "source": {
            "git_source_commit": source_commit,
            "repository_head_verified": True,
            "implementation_files_match_commit": True,
            "binding_scope": (
                "verification_execution_context_at_exact_head_not_reproducible_wheel_build"
            ),
        },
        "assets": {
            "wheel": {
                "name": WHEEL_NAME,
                "media_type": "application/zip",
                **wheel_payload.binding,
            },
            "sbom": {
                "name": SBOM_NAME,
                "media_type": "application/vnd.cyclonedx+json",
                **sbom_payload.binding,
            },
            "checksums": {
                "name": CHECKSUMS_NAME,
                "media_type": "text/plain; charset=utf-8",
                **checksums_payload.binding,
            },
        },
        "checksum_closure": {
            "status": "PASS",
            "format": "sha256sum_two_space_lf_v1",
            "line_count": 2,
            "ordered_entries": checksum_entries,
            "exact_payload_asset_set": True,
            "declared_digests_match_assets": True,
            "self_entry_absent": True,
            "single_asset_directory_verified": True,
        },
        "wheel_inventory": wheel_evidence,
        "sbom_verification": sbom_evidence,
        "isolated_wheel_smoke": smoke_evidence,
        "public_artifact_scan": scan_evidence,
        "implementation": {
            "generator_path": GENERATOR_PATH.as_posix(),
            "generator_file_sha256": generator.file_sha256,
            "schema_path": SCHEMA_PATH.as_posix(),
            "schema_file_sha256": schema_payload.file_sha256,
        },
    }
    document = {**unsigned, "receipt_sha256": canonical_json_hash(unsigned)}
    _validate_document(document, repository_root=root)

    serialized = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return ReceiptPlan(
        output=output_path,
        repository_root=root,
        source_commit=source_commit,
        document=document,
        serialized=serialized,
        observations=tuple(observations),
    )


def _assert_plan_inputs_unchanged(plan: ReceiptPlan) -> None:
    _verify_source_context(plan.repository_root, plan.source_commit)
    for expected in plan.observations:
        observed = _read_regular(
            expected.path,
            field=expected.field,
            maximum_bytes=expected.maximum_bytes,
        )
        if (
            observed.file_sha256 != expected.file_sha256
            or observed.size_bytes != expected.size_bytes
        ):
            raise ReleaseAssetsReceiptError(
                "INPUT_CHANGED_BEFORE_WRITE", "a verified input changed before receipt creation"
            )
    _validate_document(plan.document, repository_root=plan.repository_root)


def _open_parent_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_write_flags() -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _remove_owned_output(parent_fd: int, name: str, identity: tuple[int, int]) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == identity:
        try:
            os.unlink(name, dir_fd=parent_fd)
        except OSError:
            pass


def write_receipt(plan: ReceiptPlan) -> None:
    """Recheck all evidence, create once through a directory fd, and freeze 0444."""

    _validate_output(plan.output)
    _assert_plan_inputs_unchanged(plan)
    try:
        parent_fd = os.open(plan.output.parent, _open_parent_flags())
    except OSError as exc:
        raise ReleaseAssetsReceiptError(
            "OUTPUT_PARENT_UNAVAILABLE", "receipt output parent cannot be opened safely"
        ) from exc
    parent_identity = os.fstat(parent_fd)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(
                plan.output.name,
                _open_write_flags(),
                0o444,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise ReleaseAssetsReceiptError(
                "OUTPUT_ALREADY_EXISTS", "receipt output is no-clobber"
            ) from exc
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        view = memoryview(plan.serialized)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = None

        parent_after = os.lstat(plan.output.parent)
        observed = os.stat(plan.output.name, dir_fd=parent_fd, follow_symlinks=False)
        path_observed = os.lstat(plan.output)
        if (
            (parent_after.st_dev, parent_after.st_ino)
            != (parent_identity.st_dev, parent_identity.st_ino)
            or (observed.st_dev, observed.st_ino) != identity
            or (path_observed.st_dev, path_observed.st_ino) != identity
            or not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o444
            or observed.st_size != len(plan.serialized)
        ):
            raise ReleaseAssetsReceiptError(
                "OUTPUT_FINALIZATION_FAILED", "receipt output identity, mode, or size is invalid"
            )
    except ReleaseAssetsReceiptError:
        if descriptor is not None:
            os.close(descriptor)
        if identity is not None:
            _remove_owned_output(parent_fd, plan.output.name, identity)
        raise
    except Exception as exc:
        if descriptor is not None:
            os.close(descriptor)
        if identity is not None:
            _remove_owned_output(parent_fd, plan.output.name, identity)
        raise ReleaseAssetsReceiptError(
            "OUTPUT_WRITE_FAILED", "receipt output could not be completed safely"
        ) from exc
    finally:
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sbom", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "dry-run" if args.dry_run else "build"
    try:
        plan = prepare_receipt(
            wheel=args.wheel,
            sbom=args.sbom,
            checksums=args.checksums,
            source_commit=args.source_commit,
            output=args.output,
        )
        if args.dry_run:
            status = "READY"
            written = False
        else:
            write_receipt(plan)
            status = "CREATED"
            written = True
        print(
            json.dumps(
                {
                    "status": status,
                    "mode": mode,
                    "receipt_written": written,
                    "output_name": plan.output.name,
                    "receipt_sha256": plan.document["receipt_sha256"],
                    "source_commit": plan.source_commit,
                    "asset_count": 3,
                    "checksum_line_count": 2,
                    "wheel_inventory_status": "PASS",
                    "isolated_wheel_smoke_status": "PASS",
                    "public_artifact_scan_status": "PASS",
                    "remote_state_queried": False,
                    "remote_write_performed": False,
                    "network_used": False,
                    "gpu_used": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except ReleaseAssetsReceiptError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "mode": mode,
                    "blocker_ids": [exc.blocker_id],
                    "message": str(exc),
                    "receipt_written": False,
                    "remote_state_queried": False,
                    "remote_write_performed": False,
                    "network_used": False,
                    "gpu_used": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
