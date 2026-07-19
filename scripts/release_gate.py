#!/usr/bin/env python3
"""Fail-closed gate for a local Hugging Face release candidate.

An exit code of zero means the machine-checkable gates passed; it does not
replace legal, privacy, model-card, or remote-code human review.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - CLI dependency error
    raise SystemExit("release_gate.py requires PyYAML; install the project's core extra") from exc

if __package__:
    from scripts import build_model_card_evidence_v2 as model_card_v2
    from scripts.build_release import (
        ALLOWED_RELEASE_FILES,
        AUTO_MAP,
        BOUNDARY_MODE,
        BOUNDARY_MODE_CONFIG_KEY,
        FORBIDDEN_NAME_PARTS,
        FORBIDDEN_SUFFIXES,
        OPTIONAL_EVIDENCE_FILES,
        RAW_DATA_SUFFIXES,
        REQUIRED_RELEASE_FILES,
        output_artifact_fingerprint,
        validate_safetensors,
    )
    from scripts.scan_public_artifacts import load_canary_allowlist, scan_paths
else:
    import build_model_card_evidence_v2 as model_card_v2
    from build_release import (
        ALLOWED_RELEASE_FILES,
        AUTO_MAP,
        BOUNDARY_MODE,
        BOUNDARY_MODE_CONFIG_KEY,
        FORBIDDEN_NAME_PARTS,
        FORBIDDEN_SUFFIXES,
        OPTIONAL_EVIDENCE_FILES,
        RAW_DATA_SUFFIXES,
        REQUIRED_RELEASE_FILES,
        output_artifact_fingerprint,
        validate_safetensors,
    )
    from scan_public_artifacts import load_canary_allowlist, scan_paths

from pii_zh.data.artifact_policy import ArtifactPolicyError, assert_document_allowed

CHECKSUM_PATTERN = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
PLACEHOLDER_PATTERN = re.compile(r"(?:\bTODO\b|\bTBD\b|<org>|<model|\{\{[^}]+\}\})", re.IGNORECASE)
UNSAFE_REMOTE_IMPORTS = frozenset(
    {
        "ctypes",
        "httpx",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
        "webbrowser",
    }
)
UNSAFE_REMOTE_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "open",
        "os.popen",
        "os.system",
        "subprocess.run",
        "subprocess.Popen",
    }
)
BLOCKING_SEVERITIES = frozenset({"critical", "high", "medium", "moderate", "unknown"})
BASE_INITIALIZATION_STRATEGY = "base_causal_lm_v1"
STAGED_INITIALIZATION_STRATEGY = "verified_token_classifier_to_full_v1"
SAFE_SOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str


@dataclass
class GateResult:
    name: str
    passed: bool
    issues: list[GateIssue]
    details: dict[str, Any]


@dataclass
class GateInputSnapshot:
    """Pinned, private view used by every gate and by the PASS receipt.

    The artifact tree is copied (using filesystem CoW when available) into a
    private temporary directory while the original root directory is held open.
    This makes an A/B rename or an in-place source write unable to change which
    bytes the gates read.  Small external evidence files are copied once.  A
    final integrity check rejects changes to either the pinned source tree or
    the private view.
    """

    args: argparse.Namespace
    original_artifact: Path | None = None
    original_root_identity: tuple[int, int] | None = None
    source_fd: int | None = None
    source_tree_identity: str | None = None
    snapshot_tree_identity: str | None = None

    @property
    def active(self) -> bool:
        return self.original_artifact is not None

    def integrity_gate(self) -> GateResult:
        if not self.active:
            return _result("artifact_snapshot_integrity", [], activated=False)

        issues: list[GateIssue] = []
        assert self.original_artifact is not None
        assert self.original_root_identity is not None
        assert self.source_fd is not None
        assert self.source_tree_identity is not None
        assert self.snapshot_tree_identity is not None
        try:
            current = os.lstat(self.original_artifact)
            current_root_identity = (current.st_dev, current.st_ino)
            if not stat.S_ISDIR(current.st_mode) or current_root_identity != (
                self.original_root_identity
            ):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_ARTIFACT_ROOT_REPLACED",
                        "artifact root identity changed while release gates were running",
                    )
                )
        except OSError:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_ARTIFACT_ROOT_REPLACED",
                    "artifact root became unavailable while release gates were running",
                )
            )
        try:
            if _fd_tree_metadata_identity(self.source_fd) != self.source_tree_identity:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_ARTIFACT_SOURCE_MUTATED",
                        "pinned artifact source tree changed while release gates were running",
                    )
                )
        except OSError:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_ARTIFACT_SOURCE_MUTATED",
                    "pinned artifact source tree could not be re-verified",
                )
            )
        try:
            if _artifact_tree_identity(self.args.artifact) != self.snapshot_tree_identity:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_ARTIFACT_SNAPSHOT_MUTATED",
                        "private artifact snapshot changed while release gates were running",
                    )
                )
        except OSError:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_ARTIFACT_SNAPSHOT_MUTATED",
                    "private artifact snapshot could not be re-verified",
                )
            )
        return _result(
            "artifact_snapshot_integrity",
            issues,
            activated=True,
            snapshot_manifest_sha256=self.snapshot_tree_identity,
        )


def _result(name: str, issues: Iterable[GateIssue], **details: Any) -> GateResult:
    materialized = list(issues)
    return GateResult(name=name, passed=not materialized, issues=materialized, details=details)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"forbidden non-finite JSON constant {value}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _fd_tree_metadata(fd: int, *, prefix: str = "") -> list[dict[str, Any]]:
    """Describe a directory through an already-open fd without path re-resolution."""

    entries: list[dict[str, Any]] = []
    for name in sorted(os.listdir(fd)):
        relative = f"{prefix}/{name}" if prefix else name
        item_stat = os.stat(name, dir_fd=fd, follow_symlinks=False)
        common: dict[str, Any] = {
            "path": relative,
            "device": item_stat.st_dev,
            "inode": item_stat.st_ino,
            "mode": stat.S_IFMT(item_stat.st_mode),
            "size": item_stat.st_size,
            "mtime_ns": item_stat.st_mtime_ns,
            "ctime_ns": item_stat.st_ctime_ns,
        }
        if stat.S_ISDIR(item_stat.st_mode):
            common["type"] = "directory"
            entries.append(common)
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=fd,
            )
            try:
                entries.extend(_fd_tree_metadata(child_fd, prefix=relative))
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(item_stat.st_mode):
            common["type"] = "file"
            file_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=fd,
            )
            try:
                before = os.fstat(file_fd)
                digest = hashlib.sha256()
                while chunk := os.read(file_fd, 1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(file_fd)
                if (
                    _unchanged_file_identity(before)
                    != _unchanged_file_identity(after)
                    or _unchanged_file_identity(before)
                    != _unchanged_file_identity(item_stat)
                ):
                    raise ValueError(
                        f"artifact file changed while its identity was read: {relative}"
                    )
                common["sha256"] = digest.hexdigest()
            finally:
                os.close(file_fd)
            entries.append(common)
        elif stat.S_ISLNK(item_stat.st_mode):
            common["type"] = "symlink"
            common["target"] = os.readlink(name, dir_fd=fd)
            entries.append(common)
        else:
            common["type"] = "unsupported"
            entries.append(common)
    return entries


def _fd_tree_metadata_identity(fd: int) -> str:
    return _canonical_json_hash(_fd_tree_metadata(fd))


_FICLONE = 0x40049409


def _unchanged_file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _copy_fd_file(source_directory_fd: int, name: str, destination: Path) -> None:
    """Copy one pinned regular file to an independent CoW/byte snapshot."""

    source_fd = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=source_directory_fd,
    )
    destination_fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o400,
    )
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact entry is not a regular file: {name}")
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        except OSError as exc:
            if exc.errno not in {
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            os.lseek(source_fd, 0, os.SEEK_SET)
            os.ftruncate(destination_fd, 0)
            while chunk := os.read(source_fd, 1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o400)
        after = os.fstat(source_fd)
        if _unchanged_file_identity(after) != _unchanged_file_identity(before):
            raise ValueError(f"artifact file changed while it was copied: {name}")
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def _copy_fd_tree(fd: int, destination: Path) -> None:
    """Copy one pinned tree without accepting links or special files."""

    destination.mkdir(mode=0o700)
    for name in sorted(os.listdir(fd)):
        item_stat = os.stat(name, dir_fd=fd, follow_symlinks=False)
        output = destination / name
        if stat.S_ISDIR(item_stat.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=fd,
            )
            try:
                _copy_fd_tree(child_fd, output)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(item_stat.st_mode):
            _copy_fd_file(fd, name, output)
        elif stat.S_ISLNK(item_stat.st_mode):
            raise ValueError(f"artifact snapshot refuses symlink: {name}")
        else:
            raise ValueError(f"unsupported artifact entry type: {name}")
    directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    destination.chmod(0o500)


def _artifact_tree_identity(root: Path) -> str:
    """Hash every entry and regular-file byte in one materialized gate view."""

    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        item_stat = path.lstat()
        if stat.S_ISDIR(item_stat.st_mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(item_stat.st_mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": item_stat.st_size,
                    "sha256": _sha256_file(path),
                }
            )
        elif stat.S_ISLNK(item_stat.st_mode):
            entries.append(
                {"path": relative, "type": "symlink", "target": os.readlink(path)}
            )
        else:
            entries.append({"path": relative, "type": "unsupported"})
    return _canonical_json_hash(entries)


def _validate_snapshot_matches_source_metadata(
    snapshot_root: Path, source_entries: list[dict[str, Any]]
) -> None:
    """Require the private view to contain exactly the pinned source entries."""

    expected_paths = {str(entry["path"]) for entry in source_entries}
    actual_paths = {
        path.relative_to(snapshot_root).as_posix() for path in snapshot_root.rglob("*")
    }
    if actual_paths != expected_paths:
        raise ValueError("artifact changed while its gate snapshot was being constructed")
    for entry in source_entries:
        path = snapshot_root / str(entry["path"])
        item_stat = path.lstat()
        entry_type = entry["type"]
        if entry_type == "file":
            if (
                not stat.S_ISREG(item_stat.st_mode)
                or item_stat.st_size != entry["size"]
                or (item_stat.st_dev, item_stat.st_ino)
                == (entry["device"], entry["inode"])
                or _sha256_file(path) != entry["sha256"]
            ):
                raise ValueError("artifact file identity changed during snapshot construction")
        elif entry_type == "directory":
            if not stat.S_ISDIR(item_stat.st_mode):
                raise ValueError("artifact directory identity changed during snapshot construction")
        elif entry_type == "symlink":
            if not stat.S_ISLNK(item_stat.st_mode) or os.readlink(path) != entry["target"]:
                raise ValueError("artifact symlink changed during snapshot construction")
        else:
            raise ValueError("unsupported artifact entry in snapshot construction")


def _copy_regular_input(source: Path, destination: Path) -> Path:
    """Copy one external evidence file once, refusing final symlinks."""

    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"gate input must be a regular file: {source}")
        with os.fdopen(source_fd, "rb", closefd=False) as input_stream:
            with destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        destination.chmod(0o400)
        final_stat = os.fstat(source_fd)
        if (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        ) != (
            source_stat.st_dev,
            source_stat.st_ino,
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        ):
            raise ValueError(f"gate input changed while it was snapshotted: {source}")
    finally:
        os.close(source_fd)
    return destination


def _remove_private_snapshot(root: Path) -> None:
    """Restore owner write permission only to delete this process's temp tree."""

    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        except OSError:
            pass
    try:
        root.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)


def _receipt_output_inside_artifact(args: argparse.Namespace) -> bool:
    if args.json_output is None:
        return False
    artifact = args.artifact.absolute()
    output = args.json_output.absolute()
    return output == artifact or artifact in output.parents


@contextlib.contextmanager
def snapshot_gate_inputs(args: argparse.Namespace) -> Iterable[GateInputSnapshot]:
    """Yield one immutable-enough, content-addressed view for all release gates."""

    original = args.artifact.absolute()
    if _receipt_output_inside_artifact(args):
        raise ValueError("release-gate receipt output must be outside the artifact tree")
    try:
        root_lstat = os.lstat(original)
    except OSError:
        yield GateInputSnapshot(args=args)
        return
    if not stat.S_ISDIR(root_lstat.st_mode):
        yield GateInputSnapshot(args=args)
        return

    source_fd = os.open(
        original,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    # Respect TMPDIR so operators can place the independent byte snapshot on
    # a capacious local filesystem.  tempfile creates the root mode 0700.
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{original.name}.gate-snapshot-"))
    os.chmod(temporary_root, 0o700)
    try:
        snapshot_artifact = temporary_root / "artifact"
        _copy_fd_tree(source_fd, snapshot_artifact)
        source_tree_metadata = _fd_tree_metadata(source_fd)
        _validate_snapshot_matches_source_metadata(
            snapshot_artifact, source_tree_metadata
        )
        source_tree_identity = _canonical_json_hash(source_tree_metadata)
        snapshot_tree_identity = _artifact_tree_identity(snapshot_artifact)

        snapshot_args = argparse.Namespace(**vars(args))
        snapshot_args.artifact = snapshot_artifact
        snapshot_args.artifact_snapshot_sha256 = snapshot_tree_identity
        inputs_dir = temporary_root / "inputs"
        inputs_dir.mkdir(mode=0o700)
        for attribute in (
            "source_registry",
            "canary_allowlist",
            "dependency_scan",
            "dependency_exceptions",
        ):
            source = getattr(args, attribute)
            if source is None:
                continue
            if source.is_symlink() or not source.is_file():
                # Invalid inputs cannot produce PASS; leave their original path
                # so the existing gate reports the field-specific blocker.
                continue
            setattr(
                snapshot_args,
                attribute,
                _copy_regular_input(source, inputs_dir / attribute),
            )
        root_stat = os.fstat(source_fd)
        yield GateInputSnapshot(
            args=snapshot_args,
            original_artifact=original,
            original_root_identity=(root_stat.st_dev, root_stat.st_ino),
            source_fd=source_fd,
            source_tree_identity=source_tree_identity,
            snapshot_tree_identity=snapshot_tree_identity,
        )
    finally:
        os.close(source_fd)
        _remove_private_snapshot(temporary_root)


def _regular_file_sha256(path: Path | None) -> str | None:
    """Hash a configured regular file without following a final symlink."""

    if path is None or path.is_symlink() or not path.is_file():
        return None
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _package_file_count(artifact: Path) -> int:
    if artifact.is_symlink() or not artifact.is_dir():
        return 0
    try:
        return sum(1 for path in artifact.rglob("*") if not path.is_symlink() and path.is_file())
    except OSError:
        return 0


def artifact_identity(args: argparse.Namespace) -> dict[str, str | int | None]:
    """Return the path-free content identity recorded in the gate receipt."""

    artifact = args.artifact
    identity: dict[str, str | int | None] = {
        "checksums_sha256": _regular_file_sha256(artifact / "checksums.txt"),
        "package_file_count": _package_file_count(artifact),
        "sbom_sha256": _regular_file_sha256(artifact / "sbom.cdx.json"),
        "source_registry_sha256": _regular_file_sha256(args.source_registry),
        "dependency_scan_sha256": _regular_file_sha256(args.dependency_scan),
        "dependency_exceptions_sha256": _regular_file_sha256(args.dependency_exceptions),
    }
    if args.v2_contract_sha256 is not None:
        identity["v2_contract_sha256"] = args.v2_contract_sha256
    if args.v2_selected_receipt_sha256 is not None:
        identity["v2_selected_receipt_sha256"] = args.v2_selected_receipt_sha256
    snapshot_sha256 = getattr(args, "artifact_snapshot_sha256", None)
    if snapshot_sha256 is not None:
        identity["artifact_snapshot_sha256"] = snapshot_sha256
    return identity


def gate_receipt_identity(
    identity: dict[str, str | int | None],
    *,
    dependency_exceptions_configured: bool,
) -> GateResult:
    """Require a complete content address before a receipt may report PASS."""

    required_hashes = [
        "checksums_sha256",
        "sbom_sha256",
        "source_registry_sha256",
        "dependency_scan_sha256",
    ]
    if "artifact_snapshot_sha256" in identity:
        required_hashes.append("artifact_snapshot_sha256")
    if dependency_exceptions_configured:
        required_hashes.append("dependency_exceptions_sha256")
    missing = [
        field
        for field in required_hashes
        if not isinstance(identity.get(field), str)
        or SHA256_PATTERN.fullmatch(str(identity.get(field))) is None
    ]
    issues: list[GateIssue] = []
    if missing:
        issues.append(
            GateIssue(
                "RC_BLOCKED_RECEIPT_IDENTITY",
                "release receipt lacks required content hashes: " + ", ".join(missing),
            )
        )
    file_count = identity.get("package_file_count")
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
        issues.append(
            GateIssue(
                "RC_BLOCKED_RECEIPT_FILE_COUNT",
                "release receipt cannot establish a positive package file count",
            )
        )
    return _result("receipt_identity", issues)


def gate_file_set(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    if artifact.is_symlink() or not artifact.is_dir():
        return _result(
            "file_set",
            [GateIssue("RC_BLOCKED_INVALID_ARTIFACT", "artifact must be a real directory")],
        )
    files: set[str] = set()
    for path in sorted(artifact.rglob("*")):
        relative = path.relative_to(artifact).as_posix()
        if path.is_symlink():
            issues.append(GateIssue("RC_BLOCKED_SYMLINK", f"symlink is forbidden: {relative}"))
            continue
        if not path.is_file():
            continue
        files.add(relative)
        lowered = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(
                GateIssue("RC_BLOCKED_UNSAFE_WEIGHT", f"unsafe checkpoint format: {relative}")
            )
        if path.suffix.lower() in RAW_DATA_SUFFIXES:
            issues.append(
                GateIssue("RC_BLOCKED_RAW_DATA", f"raw/tabular data is forbidden: {relative}")
            )
        if any(fragment in lowered for fragment in FORBIDDEN_NAME_PARTS):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TRAINING_STATE", f"private/training state is forbidden: {relative}"
                )
            )

    missing = sorted(REQUIRED_RELEASE_FILES - files)
    extras = sorted(files - ALLOWED_RELEASE_FILES)
    if missing:
        issues.append(
            GateIssue("RC_BLOCKED_MISSING_FILES", f"missing required files: {', '.join(missing)}")
        )
    if extras:
        issues.append(
            GateIssue("RC_BLOCKED_UNREVIEWED_FILES", f"unreviewed extra files: {', '.join(extras)}")
        )
    weight_suffixes = FORBIDDEN_SUFFIXES | {".safetensors"}
    unsafe_weights = sorted(
        name
        for name in files
        if name != "model.safetensors" and Path(name).suffix.lower() in weight_suffixes
    )
    if unsafe_weights:
        issues.append(
            GateIssue(
                "RC_BLOCKED_WEIGHT_ALLOWLIST",
                f"model.safetensors must be the only weight artifact: {', '.join(unsafe_weights)}",
            )
        )
    if (artifact / "model.safetensors").is_file():
        try:
            validate_safetensors(artifact / "model.safetensors")
        except ValueError as exc:
            issues.append(GateIssue("RC_BLOCKED_INVALID_SAFETENSORS", str(exc)))
    return _result("file_set", issues, file_count=len(files))


def gate_checksums(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    checksum_path = artifact / "checksums.txt"
    if not checksum_path.is_file():
        return _result(
            "checksums",
            [GateIssue("RC_BLOCKED_CHECKSUMS_MISSING", "checksums.txt is missing")],
        )
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = CHECKSUM_PATTERN.fullmatch(line)
        if not match:
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_FORMAT", f"invalid checksums.txt line {line_number}")
            )
            continue
        expected, relative = match.groups()
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or relative == "checksums.txt":
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_PATH", f"unsafe checksum path on line {line_number}")
            )
            continue
        if relative in entries:
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_DUPLICATE", f"duplicate checksum entry: {relative}")
            )
            continue
        entries[relative] = expected
    actual_files = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "checksums.txt"
    }
    if set(entries) != actual_files:
        missing = sorted(actual_files - entries.keys())
        extra = sorted(entries.keys() - actual_files)
        issues.append(
            GateIssue(
                "RC_BLOCKED_CHECKSUM_COVERAGE",
                f"checksum coverage mismatch; missing={missing}, unknown={extra}",
            )
        )
    for relative in sorted(actual_files & entries.keys()):
        if _sha256_file(artifact / relative) != entries[relative]:
            issues.append(
                GateIssue("RC_BLOCKED_CHECKSUM_MISMATCH", f"checksum mismatch: {relative}")
            )
    return _result("checksums", issues, verified_files=len(actual_files & entries.keys()))


def gate_config_and_remote_code(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        config = _load_json(artifact / "config.json")
        training = _load_json(artifact / "training_manifest.json")
        tokenizer_config = _load_json(artifact / "tokenizer_config.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("remote_code", [GateIssue("RC_BLOCKED_CONFIG", str(exc))])
    try:
        assert_document_allowed(training, purpose="release")
    except ArtifactPolicyError:
        issues.append(
            GateIssue(
                "RC_BLOCKED_ARTIFACT_LINEAGE",
                "training manifest is denied by artifact lineage policy",
            )
        )
    if config.get("architectures") != ["Qwen3BiForTokenClassification"]:
        issues.append(GateIssue("RC_BLOCKED_ARCHITECTURE", "config architectures is not Qwen3Bi"))
    if config.get("auto_map") != AUTO_MAP:
        issues.append(GateIssue("RC_BLOCKED_AUTO_MAP", "config auto_map is missing or unexpected"))
    if config.get("model_type") != "qwen3_bi" or config.get("use_cache") is not False:
        issues.append(
            GateIssue(
                "RC_BLOCKED_CONFIG_INVARIANT", "qwen3_bi model_type/use_cache invariant failed"
            )
        )
    if config.get("pii_attention_mode") != "full" or config.get("pii_release_eligible") is not True:
        issues.append(
            GateIssue(
                "RC_BLOCKED_CHECKPOINT_ATTENTION_CONTRACT",
                "checkpoint does not explicitly declare release-eligible full attention",
            )
        )
    if training.get("attention_mode") != "full":
        issues.append(
            GateIssue(
                "RC_BLOCKED_ATTENTION_MODE",
                "published training_manifest attention_mode must be full",
            )
        )
    if tokenizer_config.get(BOUNDARY_MODE_CONFIG_KEY) != BOUNDARY_MODE:
        issues.append(
            GateIssue(
                "RC_BLOCKED_BOUNDARY_TOKENIZER",
                "tokenizer_config is missing the approved character-boundary mode",
            )
        )
    tokenizer_evidence = training.get("tokenizer")
    effective = (
        tokenizer_evidence.get("effective") if isinstance(tokenizer_evidence, dict) else None
    )
    if not isinstance(effective, dict) or effective.get("boundary_mode") != BOUNDARY_MODE:
        issues.append(
            GateIssue(
                "RC_BLOCKED_TRAINING_TOKENIZER_BINDING",
                "training_manifest is not bound to the packaged boundary tokenizer",
            )
        )

    for filename in ("configuration_qwen3_bi.py", "modeling_qwen3_bi.py"):
        path = artifact / filename
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            issues.append(GateIssue("RC_BLOCKED_REMOTE_CODE_PARSE", f"{filename}: {exc}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name.split(".")[0] for alias in node.names}
                if imports & UNSAFE_REMOTE_IMPORTS:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_REMOTE_CODE_IMPORT",
                            f"{filename} imports network/process module",
                        )
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in UNSAFE_REMOTE_IMPORTS:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_REMOTE_CODE_IMPORT",
                            f"{filename} imports network/process module",
                        )
                    )
            elif isinstance(node, ast.Call):
                called = _call_name(node.func)
                if called in UNSAFE_REMOTE_CALLS:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_REMOTE_CODE_CALL",
                            f"{filename} contains unsafe call {called}",
                        )
                    )
    return _result("remote_code", issues)


def gate_training_artifact_binding(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        training = _load_json(artifact / "training_manifest.json")
        expected = training.get("manifest_sha256")
        unsigned = dict(training)
        unsigned.pop("manifest_sha256", None)
        encoded = json.dumps(
            unsigned,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        actual_manifest_hash = hashlib.sha256(encoded.encode()).hexdigest()
        if (
            not isinstance(expected, str)
            or actual_manifest_hash != expected
            or training.get("status") != "completed"
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TRAINING_MANIFEST_HASH",
                    "training manifest completion/hash check failed",
                )
            )
        actual = output_artifact_fingerprint(artifact)
        if actual.get("weight_files") != ["model.safetensors"]:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_OUTPUT_WEIGHT_SET",
                    "release must be bound to exactly model.safetensors",
                )
            )
        if training.get("output_artifact") != actual:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_OUTPUT_ARTIFACT_BINDING",
                    "packaged model/config/tokenizer files do not match training manifest",
                )
            )
        issues.extend(_initialization_contract_issues(training))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(GateIssue("RC_BLOCKED_OUTPUT_ARTIFACT_BINDING", str(exc)))
    return _result("training_artifact_binding", issues)


def _initialization_contract_issues(training: dict[str, Any]) -> list[GateIssue]:
    """Require a self-contained hash chain whenever staged initialization is declared."""

    issues: list[GateIssue] = []

    def block(message: str) -> None:
        issues.append(GateIssue("RC_BLOCKED_INITIALIZATION_AUDIT", message))

    recipe = training.get("recipe")
    initialization = training.get("initialization")
    schema_version = training.get("schema_version")
    if isinstance(recipe, dict):
        recipe_strategy = recipe.get("initialization_strategy")
    else:
        recipe_strategy = None
    if isinstance(initialization, dict):
        audit_strategy = initialization.get("strategy")
    else:
        audit_strategy = None

    if schema_version in {3, 4} and (recipe_strategy is None or audit_strategy is None):
        block(
            f"schema {schema_version} training manifests require an explicit "
            "initialization contract"
        )
        return issues
    if recipe_strategy is None and audit_strategy is None:
        return issues  # Legacy schema-2 release fixture or completed artifact.
    if recipe_strategy != audit_strategy:
        block("training recipe and initialization audit strategies disagree")
        return issues
    if audit_strategy == BASE_INITIALIZATION_STRATEGY:
        return issues
    if audit_strategy != STAGED_INITIALIZATION_STRATEGY:
        block("training manifest declares an unknown initialization strategy")
        return issues
    if schema_version not in {3, 4}:
        block("staged initialization requires training manifest schema 3 or 4")
    if not isinstance(recipe, dict) or recipe.get("resume") is not False:
        block("staged initialization must start with fresh trainer state")
    if training.get("attention_mode") != "full":
        block("staged initialization is release-eligible only for a Full target")
    if not isinstance(initialization, dict):
        block("staged initialization audit must be an object")
        return issues

    required_hashes = (
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_output_artifact_sha256",
        "source_config_sha256",
        "source_weights_sha256",
        "source_architecture_sha256",
        "base_config_sha256",
        "base_weights_sha256",
        "label_schema_sha256",
        "tokenizer_effective_contract_sha256",
        "train_sha256",
        "validation_sha256",
    )
    for field in required_hashes:
        value = initialization.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            block(f"staged initialization audit has an invalid {field}")

    if initialization.get("source_manifest_schema_version") not in {2, 3, 4}:
        block("staged initialization source manifest schema is unsupported")
    if initialization.get("source_attention_mode") not in {"causal", "jpt"}:
        block("staged initialization source must be causal or JPT")
    if initialization.get("source_fine_tuning") not in {"full", "lora"}:
        block("staged initialization source fine-tuning mode is invalid")
    source_revision = initialization.get("source_code_revision")
    if source_revision is not None and (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        block("staged initialization source code revision is invalid")
    if initialization.get("source_safetensor_files") != ["model.safetensors"]:
        block("staged initialization must be bound to exactly model.safetensors")
    if (
        isinstance(initialization.get("tensor_count"), bool)
        or not isinstance(initialization.get("tensor_count"), int)
        or initialization["tensor_count"] < 1
    ):
        block("staged initialization tensor count is invalid")
    tensor_dtypes = initialization.get("tensor_dtypes")
    if (
        not isinstance(tensor_dtypes, list)
        or not tensor_dtypes
        or any(not isinstance(value, str) or not value for value in tensor_dtypes)
    ):
        block("staged initialization tensor dtype audit is missing")
    score_keys = initialization.get("score_keys")
    if (
        not isinstance(score_keys, list)
        or not score_keys
        or any(not isinstance(value, str) or not value.startswith("score.") for value in score_keys)
    ):
        block("staged initialization classifier-head audit is missing")
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if initialization.get(field) != []:
            block(f"staged initialization {field} must be empty")

    base = training.get("base_checkpoint")
    tokenizer = training.get("tokenizer")
    datasets = training.get("datasets")
    if not isinstance(base, dict) or (
        initialization.get("base_config_sha256") != base.get("config_sha256")
        or initialization.get("base_weights_sha256") != base.get("weights_sha256")
    ):
        block("staged initialization base hashes disagree with the target manifest")
    if initialization.get("base_source_id") != training.get("base_source_id"):
        block("staged initialization base source ID disagrees with the target")
    if initialization.get("taxonomy_version") != training.get("taxonomy_version"):
        block("staged initialization taxonomy version disagrees with the target")
    if initialization.get("label_schema_sha256") != training.get("label_schema_sha256"):
        block("staged initialization label schema disagrees with the target")
    effective_hash = (
        tokenizer.get("effective_contract_sha256") if isinstance(tokenizer, dict) else None
    )
    if initialization.get("tokenizer_effective_contract_sha256") != effective_hash:
        block("staged initialization tokenizer contract disagrees with the target")
    train = datasets.get("train") if isinstance(datasets, dict) else None
    validation = datasets.get("validation") if isinstance(datasets, dict) else None
    if not isinstance(train, dict) or initialization.get("train_sha256") != train.get("sha256"):
        block("staged initialization train data hash disagrees with the target")
    if not isinstance(validation, dict) or (
        initialization.get("validation_sha256") != validation.get("sha256")
    ):
        block("staged initialization validation data hash disagrees with the target")
    return issues


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _registry_sources(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry must contain a sources list")
    result: dict[str, dict[str, Any]] = {}
    for entry in sources:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("every source registry entry must have an id")
        if entry["id"] in result:
            raise ValueError(f"duplicate source registry id: {entry['id']}")
        result[entry["id"]] = entry
    return result


def _source_ids_from_manifest(manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if lowered in {"source_id", "base_source_id", "teacher_source_id"} and isinstance(
            value, str
        ):
            result.add(value)
        elif lowered in {"source_ids", "training_source_ids"} and isinstance(value, list):
            result.update(item for item in value if isinstance(item, str))
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    visit(manifest)
    return result


def _schema4_training_source_lineage_issues(
    manifest: dict[str, Any],
) -> tuple[set[str], list[GateIssue]]:
    """Validate the canonical, top-level lineage required by training schema 4."""

    issues: list[GateIssue] = []

    def block(message: str) -> None:
        issues.append(GateIssue("RC_BLOCKED_TRAINING_SOURCE_LINEAGE", message))

    raw_source_ids = manifest.get("training_source_ids")
    if not isinstance(raw_source_ids, list) or not raw_source_ids:
        block("schema 4 training manifest requires a non-empty training_source_ids list")
        return set(), issues
    all_strings = all(isinstance(source_id, str) for source_id in raw_source_ids)
    if len(raw_source_ids) > 256:
        block("schema 4 training_source_ids exceeds the 256-entry safety limit")
    if any(
        not isinstance(source_id, str) or SAFE_SOURCE_ID_PATTERN.fullmatch(source_id) is None
        for source_id in raw_source_ids
    ):
        block("schema 4 training source IDs must be path-free identifiers")
    valid_source_ids = {
        source_id
        for source_id in raw_source_ids
        if isinstance(source_id, str) and SAFE_SOURCE_ID_PATTERN.fullmatch(source_id) is not None
    }
    valid_source_list = [
        source_id
        for source_id in raw_source_ids
        if isinstance(source_id, str) and SAFE_SOURCE_ID_PATTERN.fullmatch(source_id) is not None
    ]
    if len(set(valid_source_list)) != len(valid_source_list):
        block("schema 4 training source IDs must be unique")
    if not all_strings or raw_source_ids != sorted(raw_source_ids):
        block("schema 4 training source IDs must be sorted canonically")

    base_source_id = manifest.get("base_source_id")
    if not isinstance(base_source_id, str) or base_source_id not in valid_source_ids:
        block("schema 4 training lineage does not include its base_source_id")

    direct_source_ids: set[str] = set()
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        block("schema 4 training manifest has no dataset summaries")
    else:
        for split_name, split in datasets.items():
            if not isinstance(split, dict):
                block(f"schema 4 dataset {split_name!r} is not an object")
                continue
            summary = split.get("summary")
            sources = summary.get("sources") if isinstance(summary, dict) else None
            if not isinstance(sources, list):
                block(f"schema 4 dataset {split_name!r} has no direct source summary")
                continue
            for index, source in enumerate(sources):
                source_id = source.get("source_id") if isinstance(source, dict) else None
                if (
                    not isinstance(source_id, str)
                    or SAFE_SOURCE_ID_PATTERN.fullmatch(source_id) is None
                ):
                    block(
                        f"schema 4 dataset {split_name!r} source {index} has an invalid source_id"
                    )
                else:
                    direct_source_ids.add(source_id)
    missing_direct = direct_source_ids - valid_source_ids
    if missing_direct:
        block(
            "schema 4 training lineage omits direct dataset sources: "
            + ", ".join(sorted(missing_direct))
        )
    return valid_source_ids, issues


def _evaluation_only_references(document: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    def visit(value: Any, trail: str = "root") -> None:
        if isinstance(value, dict):
            if value.get("used_for_training") is False:
                return
            sample_count = value.get("sample_count", value.get("sampled_records"))
            if sample_count == 0:
                return
            for key, child in value.items():
                lowered = str(key).lower()
                child_trail = f"{trail}.{key}"
                if lowered in {"forbidden_training_pools", "excluded_pools"}:
                    continue
                if lowered in {"pool", "training_pool", "data_pool", "admitted_pool"}:
                    values = child if isinstance(child, list) else [child]
                    if any(str(item).lower() == "evaluation_only" for item in values):
                        failures.append(child_trail)
                if lowered.endswith(("_path", "_file", "_dir")) and isinstance(child, str):
                    if "evaluation_only" in child.lower():
                        failures.append(child_trail)
                visit(child, child_trail)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{trail}[{index}]")

    visit(document)
    return failures


def gate_provenance(artifact: Path, registry_path: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        registry = _registry_sources(_load_yaml(registry_path))
        training = _load_json(artifact / "training_manifest.json")
        data = _load_json(artifact / "data_provenance.json")
        teacher = _load_json(artifact / "teacher_provenance.json")
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return _result("provenance", [GateIssue("RC_BLOCKED_PROVENANCE_PARSE", str(exc))])

    for filename, document in (
        ("training_manifest.json", training),
        ("data_provenance.json", data),
        ("teacher_provenance.json", teacher),
    ):
        try:
            assert_document_allowed(document, purpose="release")
        except ArtifactPolicyError:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_ARTIFACT_LINEAGE",
                    f"{filename} is denied by artifact lineage policy",
                )
            )
        references = _evaluation_only_references(document)
        if references:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_EVALUATION_ONLY_TRAINING",
                    f"{filename} admits evaluation_only data at: {', '.join(references)}",
                )
            )

    if training.get("schema_version") == 4:
        training_source_ids, lineage_issues = _schema4_training_source_lineage_issues(training)
        issues.extend(lineage_issues)
    else:
        training_source_ids = _source_ids_from_manifest(training)
    actual_provenance_ids: set[str] = set()
    data_sources = data.get("sources")
    if not isinstance(data_sources, list) or not data_sources:
        issues.append(
            GateIssue("RC_BLOCKED_DATA_PROVENANCE_EMPTY", "data provenance has no sampled sources")
        )
    else:
        for index, source in enumerate(data_sources):
            if not isinstance(source, dict):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_DATA_PROVENANCE_SCHEMA",
                        f"data sources[{index}] is not an object",
                    )
                )
                continue
            count = source.get("sample_count", source.get("sampled_records"))
            if not isinstance(count, int) or count <= 0:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_DATA_PROVENANCE_COUNT",
                        f"data sources[{index}] lacks a positive sampled count",
                    )
                )
            source_id = source.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_DATA_PROVENANCE_SOURCE_ID",
                        f"data sources[{index}] lacks source_id",
                    )
                )
            elif isinstance(count, int) and count > 0:
                actual_provenance_ids.add(source_id)

    teachers = teacher.get("teachers")
    if not isinstance(teachers, list):
        if teacher.get("teacher_used") is not False:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TEACHER_PROVENANCE_SCHEMA",
                    "teacher provenance needs a teachers list or teacher_used=false",
                )
            )
    else:
        for index, entry in enumerate(teachers):
            if not isinstance(entry, dict) or not isinstance(entry.get("used_for_training"), bool):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_TEACHER_PROVENANCE_SCHEMA",
                        f"teachers[{index}] must explicitly state used_for_training",
                    )
                )
                continue
            if entry["used_for_training"]:
                source_id = entry.get("source_id")
                if not isinstance(source_id, str) or not source_id:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_TEACHER_PROVENANCE_SOURCE_ID",
                            f"teachers[{index}] lacks source_id",
                        )
                    )
                else:
                    actual_provenance_ids.add(source_id)

    unmanifested = actual_provenance_ids - training_source_ids
    if unmanifested:
        issues.append(
            GateIssue(
                "RC_BLOCKED_PROVENANCE_INCONSISTENT",
                "sampled data/teacher source IDs are absent from training_manifest.json: "
                + ", ".join(sorted(unmanifested)),
            )
        )
    source_ids = training_source_ids | actual_provenance_ids
    if not source_ids:
        issues.append(
            GateIssue(
                "RC_BLOCKED_SOURCE_REGISTRY_EMPTY",
                "no actual training/base/teacher source IDs were recorded",
            )
        )

    for source_id in sorted(source_ids):
        source = registry.get(source_id)
        if source is None:
            issues.append(
                GateIssue("RC_BLOCKED_SOURCE_UNREGISTERED", f"unregistered source: {source_id}")
            )
            continue
        if source.get("public_weight_training_allowed") is not True:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SOURCE_NOT_LICENSED",
                    f"source is not approved for public-weight training: {source_id}",
                )
            )
        license_value = source.get("declared_license", source.get("license"))
        if not isinstance(license_value, str) or license_value.strip().lower() in {
            "",
            "contract_specific",
            "pending",
            "proprietary",
            "unknown",
        }:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SOURCE_LICENSE",
                    f"source lacks an explicit distributable license: {source_id}",
                )
            )
        revision = source.get("revision", source.get("model_revision"))
        if (
            not isinstance(revision, str)
            or not revision.strip()
            or "<" in revision
            or revision.strip().lower() in {"head", "latest", "main", "master"}
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SOURCE_REVISION", f"source lacks an immutable revision: {source_id}"
                )
            )
        review = str(source.get("review_status", "")).lower()
        if not review or any(
            word in review
            for word in (
                "incomplete",
                "no_public",
                "not_approved",
                "not_available",
                "pending",
                "quarantined",
                "requires",
                "unreviewed",
            )
        ):
            issues.append(
                GateIssue("RC_BLOCKED_SOURCE_REVIEW", f"source review is incomplete: {source_id}")
            )
        if source.get("forced_pool") == "evaluation_only":
            issues.append(
                GateIssue(
                    "RC_BLOCKED_EVALUATION_ONLY_SOURCE",
                    f"evaluation-only source was referenced by training provenance: {source_id}",
                )
            )
        if source.get("kind") == "api_teacher":
            terms_snapshot = source.get("terms_snapshot")
            if (
                not source.get("legal_review_id")
                or not isinstance(terms_snapshot, str)
                or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", terms_snapshot)
            ):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_API_TEACHER_TERMS",
                        f"API teacher lacks legal-review/terms evidence: {source_id}",
                    )
                )
    if source_ids and not any(
        registry.get(source_id, {}).get("kind") == "base_model" for source_id in source_ids
    ):
        issues.append(
            GateIssue(
                "RC_BLOCKED_BASE_PROVENANCE",
                "training provenance does not reference a registered base_model source",
            )
        )
    return _result("provenance", issues, referenced_source_ids=sorted(source_ids))


def _seed_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = report.get("seeds", report.get("runs", []))
    if not isinstance(candidates, list):
        return []
    runs: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, int):
            runs.append({"seed": candidate})
        elif isinstance(candidate, dict):
            runs.append(candidate)
    return runs


def gate_quality(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    try:
        report = _load_json(artifact / "evaluation_report.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("quality", [GateIssue("RC_BLOCKED_QUALITY_PARSE", str(exc))])
    try:
        assert_document_allowed(report, purpose="release")
    except ArtifactPolicyError:
        issues.append(
            GateIssue(
                "RC_BLOCKED_ARTIFACT_LINEAGE",
                "evaluation report is denied by artifact lineage policy",
            )
        )
    runs = _seed_runs(report)
    unique_seeds = {run.get("seed") for run in runs if isinstance(run.get("seed"), int)}
    if len(unique_seeds) < 3:
        issues.append(
            GateIssue(
                "RC_BLOCKED_INSUFFICIENT_SEEDS",
                f"quality evidence has {len(unique_seeds)} unique seed(s); at least 3 are required",
            )
        )
    expected_metric_names: set[str] | None = None
    for run in runs:
        seed = run.get("seed")
        if not isinstance(seed, int):
            continue
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            issues.append(GateIssue("RC_BLOCKED_SEED_METRICS", f"seed {seed} has no metrics"))
        else:
            metric_names = set(metrics)
            if expected_metric_names is None:
                expected_metric_names = metric_names
            elif metric_names != expected_metric_names:
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_SEED_METRIC_MISMATCH",
                        f"seed {seed} reports a different metric set",
                    )
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in metrics.values()
            ):
                issues.append(
                    GateIssue(
                        "RC_BLOCKED_SEED_METRIC_VALUE",
                        f"seed {seed} contains a non-finite/non-numeric metric",
                    )
                )
        if run.get("quality_gate_passed") is not True:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SEED_QUALITY",
                    f"seed {seed} did not explicitly pass its quality gate",
                )
            )
    quality_gate = report.get("quality_gate")
    if (
        not isinstance(quality_gate, dict)
        or str(quality_gate.get("status", "")).lower() != "passed"
    ):
        issues.append(
            GateIssue("RC_BLOCKED_QUALITY_EVIDENCE", "aggregate quality_gate.status is not passed")
        )
    else:
        criteria = quality_gate.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            issues.append(
                GateIssue("RC_BLOCKED_QUALITY_CRITERIA", "quality gate has no explicit criteria")
            )
        else:
            for index, criterion in enumerate(criteria):
                if not isinstance(criterion, dict) or criterion.get("passed") is not True:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_QUALITY_CRITERION",
                            f"quality criterion {index} did not explicitly pass",
                        )
                    )
                    continue
                name = criterion.get("name")
                value = criterion.get("value")
                threshold = criterion.get("threshold")
                operator = criterion.get("operator")
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold)
                ):
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_QUALITY_CRITERION_EVIDENCE",
                            f"quality criterion {index} lacks a name/value/threshold",
                        )
                    )
                    continue
                numeric_value = float(value)
                numeric_threshold = float(threshold)
                comparisons = {
                    ">": numeric_value > numeric_threshold,
                    ">=": numeric_value >= numeric_threshold,
                    "<": numeric_value < numeric_threshold,
                    "<=": numeric_value <= numeric_threshold,
                }
                if operator not in comparisons or not comparisons[operator]:
                    issues.append(
                        GateIssue(
                            "RC_BLOCKED_QUALITY_CRITERION_COMPARISON",
                            f"quality criterion {index} does not satisfy its stated comparison",
                        )
                    )
    if str(report.get("release_decision", "")).lower() not in {"pass", "passed", "approved"}:
        issues.append(
            GateIssue("RC_BLOCKED_RELEASE_DECISION", "evaluation report does not approve release")
        )
    try:
        training_manifest = _load_json(artifact / "training_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(GateIssue("RC_BLOCKED_TRAINING_SEED", str(exc)))
    else:
        selected_seed = training_manifest.get("seed")
        if not isinstance(selected_seed, int) or selected_seed not in unique_seeds:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_TRAINING_SEED",
                    "training manifest seed is absent from the three-seed evidence",
                )
            )

    try:
        model_index = _load_yaml(artifact / "model-index.yml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        issues.append(GateIssue("RC_BLOCKED_MODEL_INDEX", str(exc)))
    else:
        indexed_models = model_index.get("model-index")
        indexed_results = []
        if isinstance(indexed_models, list):
            for model in indexed_models:
                if isinstance(model, dict) and isinstance(model.get("results"), list):
                    indexed_results.extend(model["results"])
        indexed_metrics = [
            metric
            for result in indexed_results
            if isinstance(result, dict) and isinstance(result.get("metrics"), list)
            for metric in result["metrics"]
            if isinstance(metric, dict)
        ]
        attributable_prefixes = ("Model Raw ", "Model Calibrated ")
        if (
            not indexed_results
            or not indexed_metrics
            or not all(
                not isinstance(metric.get("value"), bool)
                and isinstance(metric.get("value"), (int, float))
                and math.isfinite(metric["value"])
                for metric in indexed_metrics
            )
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_MODEL_INDEX",
                    "model-index.yml has no result with a finite numeric metric",
                )
            )
        elif not all(
            isinstance(metric.get("name"), str)
            and metric["name"].startswith(attributable_prefixes)
            for metric in indexed_metrics
        ):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_MODEL_INDEX_ATTRIBUTION",
                    "model-index.yml metrics must be explicitly attributed to Model Raw or "
                    "Model Calibrated output",
                )
            )
    return _result("quality", issues, unique_seed_count=len(unique_seeds))


def gate_public_scan(artifact: Path, canary_allowlist: Path | None) -> GateResult:
    try:
        allowed = load_canary_allowlist(canary_allowlist)
        findings = scan_paths([artifact], allowed_canaries=allowed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("public_artifact_scan", [GateIssue("RC_BLOCKED_SCAN_ERROR", str(exc))])
    issues = [
        GateIssue(
            "RC_BLOCKED_SECRET_OR_PII",
            f"{finding.path}:{finding.line} {finding.kind} {finding.fingerprint[:19]}...",
        )
        for finding in findings
    ]
    return _result("public_artifact_scan", issues, finding_count=len(findings))


def gate_release_text(artifact: Path) -> GateResult:
    issues: list[GateIssue] = []
    for filename in ("README.md", "SECURITY.md", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        path = artifact / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if PLACEHOLDER_PATTERN.search(text):
            issues.append(
                GateIssue("RC_BLOCKED_PLACEHOLDER", f"unresolved placeholder in {filename}")
            )
    security_path = artifact / "SECURITY.md"
    if security_path.is_file():
        lowered = security_path.read_text(encoding="utf-8").lower()
        if "release blocker" in lowered or "not yet configured" in lowered:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_SECURITY_CONTACT",
                    "SECURITY.md says the private reporting route is not configured",
                )
            )
    third_party_path = artifact / "THIRD_PARTY_NOTICES.md"
    if third_party_path.is_file():
        lowered = third_party_path.read_text(encoding="utf-8").lower()
        generic_markers = (
            "not a legal conclusion",
            "planned sources",
            "must generate a source-specific",
        )
        if any(marker in lowered for marker in generic_markers):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_GENERIC_NOTICES",
                    "THIRD_PARTY_NOTICES.md is a repository template, "
                    "not release-specific evidence",
                )
            )
    return _result("release_text", issues)


def gate_model_card_evidence_v2(
    artifact: Path,
    *,
    expected_contract_sha256: str | None,
    expected_selected_receipt_sha256: str | None,
) -> GateResult:
    """Activate the successor gate only for an explicit packaged v2 contract."""

    present_v2_files = {
        name
        for name in OPTIONAL_EVIDENCE_FILES
        if (artifact / name).exists() or (artifact / name).is_symlink()
    }
    activated = bool(present_v2_files)
    external_anchor_supplied = (
        expected_contract_sha256 is not None or expected_selected_receipt_sha256 is not None
    )
    if not activated:
        issues = []
        if external_anchor_supplied:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_MODEL_CARD_EVIDENCE_V2_MISSING",
                    "v2 trust anchors were supplied but no explicit v2 release contract exists",
                )
            )
        return _result("model_card_evidence_v2", issues, activated=False)

    issues: list[GateIssue] = []
    if present_v2_files != set(OPTIONAL_EVIDENCE_FILES):
        missing = sorted(set(OPTIONAL_EVIDENCE_FILES) - present_v2_files)
        issues.append(
            GateIssue(
                "RC_BLOCKED_MODEL_CARD_EVIDENCE_V2_INCOMPLETE",
                "explicit v2 release bundle is incomplete; missing: " + ", ".join(missing),
            )
        )
        return _result(
            "model_card_evidence_v2",
            issues,
            activated=True,
            externally_anchored=False,
        )
    anchors = {
        "contract": expected_contract_sha256,
        "selected_receipt": expected_selected_receipt_sha256,
    }
    invalid_anchors = sorted(
        name
        for name, value in anchors.items()
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
    )
    if invalid_anchors:
        issues.append(
            GateIssue(
                "RC_BLOCKED_MODEL_CARD_EVIDENCE_V2_TRUST_ANCHOR",
                "explicit v2 release requires external frozen SHA-256 anchors for: "
                + ", ".join(invalid_anchors),
            )
        )
    else:
        try:
            model_card_v2.validate_release_bundle(
                artifact,
                expected_contract_sha256=expected_contract_sha256,
                expected_selected_receipt_sha256=expected_selected_receipt_sha256,
            )
        except (model_card_v2.ModelCardEvidenceError, OSError, ValueError) as exc:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_MODEL_CARD_EVIDENCE_V2",
                    str(exc),
                )
            )
    return _result(
        "model_card_evidence_v2",
        issues,
        activated=True,
        externally_anchored=not invalid_anchors,
    )


def _normalized_vulnerabilities(scan: dict[str, Any]) -> list[dict[str, Any]]:
    findings = scan.get("findings")
    if isinstance(findings, list):
        return [item for item in findings if isinstance(item, dict)]
    dependencies = scan.get("dependencies")
    normalized: list[dict[str, Any]] = []
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                continue
            component = dependency.get("name")
            for vulnerability in dependency.get("vulns", []):
                if isinstance(vulnerability, dict):
                    normalized.append(
                        {
                            "id": vulnerability.get("id"),
                            "component": component,
                            "severity": vulnerability.get("severity", "unknown"),
                            "status": "open",
                        }
                    )
    return normalized


def _load_exceptions(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    document = _load_json(path)
    if document.get("schema_version") != 1 or not isinstance(document.get("exceptions"), list):
        raise ValueError("dependency exceptions must use schema_version 1 and contain exceptions")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(document["exceptions"]):
        if not isinstance(entry, dict):
            raise ValueError(f"dependency exception {index} must be an object")
        vulnerability_id = entry.get("vulnerability_id")
        component = entry.get("component")
        if not isinstance(vulnerability_id, str) or not isinstance(component, str):
            raise ValueError(
                f"dependency exception {index} must identify vulnerability_id and component"
            )
        if not isinstance(entry.get("approved_by"), str) or not entry["approved_by"].strip():
            raise ValueError(f"dependency exception {index} lacks approved_by")
        if (
            not isinstance(entry.get("justification"), str)
            or len(entry["justification"].strip()) < 20
        ):
            raise ValueError(f"dependency exception {index} needs a substantive justification")
        controls = entry.get("compensating_controls")
        if (
            not isinstance(controls, list)
            or not controls
            or any(not isinstance(control, str) or not control.strip() for control in controls)
        ):
            raise ValueError(f"dependency exception {index} needs compensating_controls")
        expires = entry.get("expires_on")
        if not isinstance(expires, str):
            raise ValueError(f"dependency exception {index} has invalid expires_on")
        try:
            expiration = date.fromisoformat(expires)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dependency exception {index} has invalid expires_on") from exc
        if expiration < datetime.now(timezone.utc).date():
            raise ValueError(f"dependency exception {index} expired on {expiration.isoformat()}")
        key = (vulnerability_id, component.lower())
        if key in result:
            raise ValueError(f"duplicate dependency exception for {vulnerability_id}/{component}")
        result[key] = entry
    return result


def gate_dependencies(
    scan_path: Path | None,
    exceptions_path: Path | None,
    sbom_path: Path,
) -> GateResult:
    if scan_path is None:
        return _result(
            "dependencies",
            [
                GateIssue(
                    "RC_BLOCKED_DEPENDENCY_SCAN_MISSING",
                    "a completed dependency vulnerability scan is required",
                )
            ],
        )
    issues: list[GateIssue] = []
    used_exceptions: list[str] = []
    try:
        scan = _load_json(scan_path)
        exceptions = _load_exceptions(exceptions_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("dependencies", [GateIssue("RC_BLOCKED_DEPENDENCY_EVIDENCE", str(exc))])
    if not isinstance(scan.get("scanner"), str) or not scan["scanner"].strip():
        issues.append(
            GateIssue("RC_BLOCKED_DEPENDENCY_SCANNER", "dependency scan does not name its scanner")
        )
    if not isinstance(scan.get("generated_at"), str):
        issues.append(
            GateIssue("RC_BLOCKED_DEPENDENCY_SCAN_TIME", "dependency scan lacks generated_at")
        )
    if scan.get("scan_complete") is not True:
        issues.append(
            GateIssue(
                "RC_BLOCKED_DEPENDENCY_SCAN_INCOMPLETE",
                "dependency scan is not explicitly complete",
            )
        )
    recorded_sbom = scan.get("sbom_sha256")
    if (
        not isinstance(recorded_sbom, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_sbom)
        or not sbom_path.is_file()
        or _sha256_file(sbom_path) != recorded_sbom
    ):
        issues.append(
            GateIssue(
                "RC_BLOCKED_DEPENDENCY_SBOM_MISMATCH",
                "dependency scan is not bound to the packaged SBOM SHA-256",
            )
        )
    for finding in _normalized_vulnerabilities(scan):
        vulnerability_id = finding.get("id")
        component = finding.get("component")
        severity = str(finding.get("severity", "unknown")).lower()
        status = str(finding.get("status", "open")).lower()
        if not isinstance(vulnerability_id, str) or not isinstance(component, str):
            issues.append(
                GateIssue(
                    "RC_BLOCKED_DEPENDENCY_FINDING_SCHEMA", "dependency finding lacks id/component"
                )
            )
            continue
        if status in {"fixed", "not_affected", "false_positive"}:
            continue
        exception = exceptions.get((vulnerability_id, component.lower()))
        if exception is not None:
            used_exceptions.append(str(exception.get("id", f"{vulnerability_id}/{component}")))
            continue
        if severity in BLOCKING_SEVERITIES:
            issues.append(
                GateIssue(
                    "RC_BLOCKED_DEPENDENCY_VULNERABILITY",
                    f"unresolved {severity} vulnerability {vulnerability_id} in {component}",
                )
            )
    return _result("dependencies", issues, used_exceptions=sorted(used_exceptions))


def gate_sbom(artifact: Path) -> GateResult:
    try:
        sbom = _load_json(artifact / "sbom.cdx.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result("sbom", [GateIssue("RC_BLOCKED_SBOM_PARSE", str(exc))])
    issues: list[GateIssue] = []
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") not in {"1.5", "1.6"}:
        issues.append(GateIssue("RC_BLOCKED_SBOM_FORMAT", "SBOM must be CycloneDX 1.5 or 1.6"))
    if not isinstance(sbom.get("components"), list) or not sbom["components"]:
        issues.append(GateIssue("RC_BLOCKED_SBOM_EMPTY", "SBOM has no dependency components"))
    return _result("sbom", issues, component_count=len(sbom.get("components", [])))


def run_gates(args: argparse.Namespace) -> list[GateResult]:
    artifact = args.artifact.resolve()
    return [
        gate_file_set(artifact),
        gate_checksums(artifact),
        gate_config_and_remote_code(artifact),
        gate_training_artifact_binding(artifact),
        gate_provenance(artifact, args.source_registry),
        gate_quality(artifact),
        gate_public_scan(artifact, args.canary_allowlist),
        gate_release_text(artifact),
        gate_model_card_evidence_v2(
            artifact,
            expected_contract_sha256=args.v2_contract_sha256,
            expected_selected_receipt_sha256=args.v2_selected_receipt_sha256,
        ),
        gate_sbom(artifact),
        gate_dependencies(
            args.dependency_scan,
            args.dependency_exceptions,
            artifact / "sbom.cdx.json",
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--canary-allowlist", type=Path)
    parser.add_argument("--dependency-scan", type=Path)
    parser.add_argument("--dependency-exceptions", type=Path)
    parser.add_argument("--v2-contract-sha256")
    parser.add_argument("--v2-selected-receipt-sha256")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def execute_gates(
    args: argparse.Namespace,
) -> tuple[list[GateResult], dict[str, str | int | None], bool]:
    """Run every check and build its receipt from one pinned input snapshot."""

    execution_error = False
    try:
        with snapshot_gate_inputs(args) as snapshot:
            try:
                results = run_gates(snapshot.args)
            except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
                print(f"release gate error: {exc}", file=sys.stderr)
                execution_error = True
                results = [
                    _result(
                        "gate_execution",
                        [GateIssue("RC_BLOCKED_GATE_EXECUTION", str(exc))],
                    )
                ]
            identity = artifact_identity(snapshot.args)
            results.append(snapshot.integrity_gate())
    except (OSError, ValueError) as exc:
        print(f"release gate snapshot error: {exc}", file=sys.stderr)
        execution_error = True
        results = [
            _result(
                "artifact_snapshot_integrity",
                [GateIssue("RC_BLOCKED_ARTIFACT_SNAPSHOT", str(exc))],
                activated=False,
            )
        ]
        identity = artifact_identity(args)
    results.append(
        gate_receipt_identity(
            identity,
            dependency_exceptions_configured=args.dependency_exceptions is not None,
        )
    )
    return results, identity, execution_error


def main() -> int:
    args = _parse_args()
    results, identity, execution_error = execute_gates(args)
    blockers = sum(len(result.issues) for result in results)
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "PASS" if blockers == 0 else "RC_BLOCKED",
        "blocker_count": blockers,
        "artifact_identity": identity,
        "gates": [
            {
                "name": result.name,
                "passed": result.passed,
                "issues": [asdict(issue) for issue in result.issues],
                "details": result.details,
            }
            for result in results
        ],
        "notice": "Machine checks do not replace required legal, privacy, and human reviews.",
    }
    report["manifest_sha256"] = _canonical_json_hash(report)
    if args.json_output and not _receipt_output_inside_artifact(args):
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"{report['status']}: {blockers} blocker(s)")
    for result in results:
        print(f"- {'PASS' if result.passed else 'BLOCKED'} {result.name}")
        for issue in result.issues:
            print(f"  {issue.code}: {issue.message}")
    if execution_error:
        return 2
    return 0 if blockers == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
