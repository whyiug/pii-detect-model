#!/usr/bin/env python3
"""Collect read-only GitHub or Hugging Face pre-publication evidence.

The GitHub collector uses the official REST API through ``gh`` and verifies the
annotated SSH tag locally against the release signing key.  The Hugging Face
collector consumes the reviewed immutable-download provenance and package
verification receipts, then performs a small read-only Hub recheck.  It never
downloads model weights.

No token, authentication header, raw API response, signature, or release asset
content is written to the evidence document or error output.
"""

from __future__ import annotations

import argparse
import base64
import binascii
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
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover - release environment supplies the client
    HfApi = None  # type: ignore[assignment,misc]
    hf_hub_download = None  # type: ignore[assignment]

try:
    from scripts import build_community_v2_publication_receipt as contract
    from scripts import materialize_community_v2_hf_snapshot as hf_provenance
except ImportError:  # pragma: no cover - direct script execution
    import build_community_v2_publication_receipt as contract  # type: ignore[no-redef]
    import materialize_community_v2_hf_snapshot as hf_provenance  # type: ignore[no-redef]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = Path("scripts/collect_community_v2_remote_evidence.py")
COLLECTOR_VERSION = "1.0.0"
OFFICIAL_GITHUB_ENDPOINT = "https://api.github.com"
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
GITHUB_REPOSITORY = "whyiug/pii-detect-model"
GITHUB_OWNER = "whyiug"
HUGGING_FACE_REPOSITORY = "Forrest20231206/pii-zh-qwen3-0.6b-24class"
RELEASE_TAG = "v0.2.0rc1"
GITHUB_MODEL_CARD_PATH = "model_cards/PII_ZH_QWEN3_0_6B_24CLASS_PUBLICATION.md"
HF_MODEL_CARD_PATH = "README.md"
WORKFLOW_PATH = ".github/workflows/ci.yml"
REQUIRED_WORKFLOW_JOBS = ("package-smoke", "quality")
EXPECTED_SIGNING_KEY_FINGERPRINT = (
    "SHA256:j3DjYC57yPDuXhn5qbIKYmc0OLyJrRtbqgvsLPIl4hg"
)
PACKAGE_VERIFICATION_SCHEMA_VERSION = (
    "pii-zh.community-v2-publication-package-verification.v1"
)
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_RELEASE_CHECKSUMS_BYTES = 4096
READ_BLOCK_BYTES = 1024 * 1024

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SSH_FINGERPRINT_RE = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")


class RemoteEvidenceError(RuntimeError):
    """Fail-closed remote evidence collection error."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class JsonPayload:
    path: Path
    payload: bytes
    document: Mapping[str, Any]
    file_sha256: str
    mode: int


class GitHubTransport(Protocol):
    def get_json(self, endpoint: str, *, params: Mapping[str, str] | None = None) -> Any: ...

    def download_digest(self, endpoint: str) -> tuple[str, int]: ...

    def download_bytes(self, endpoint: str, *, maximum: int) -> bytes: ...


class SignatureVerifier(Protocol):
    def fingerprint(self, public_key: str) -> str: ...

    def verify(self, *, public_key: str, payload: bytes, signature: bytes) -> bool: ...


class HuggingFaceTransport(Protocol):
    def repo_info(self, *, revision: str) -> object: ...

    def list_repo_files(self, *, revision: str) -> list[str]: ...

    def read_file(self, *, revision: str, path: str) -> bytes: ...


def _canonical_json_bytes(value: Any, *, remove: str | None = None) -> bytes:
    document = dict(value) if isinstance(value, Mapping) else value
    if remove is not None:
        if not isinstance(document, dict):
            raise RemoteEvidenceError(
                "INVALID_CANONICAL_JSON", "self-hash removal requires a JSON object"
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
        raise RemoteEvidenceError(
            "INVALID_CANONICAL_JSON", "evidence is not canonical-JSON serializable"
        ) from exc


def canonical_json_hash(value: Any, *, remove: str | None = None) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _strict_json_value(payload: bytes, *, field: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RemoteEvidenceError(
                    "DUPLICATE_JSON_KEY", f"{field} repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise RemoteEvidenceError("NONFINITE_JSON", f"{field} contains a non-finite number")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteEvidenceError("INVALID_JSON", f"{field} is not strict UTF-8 JSON") from exc


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    document = _strict_json_value(payload, field=field)
    if not isinstance(document, dict):
        raise RemoteEvidenceError("INVALID_JSON_SHAPE", f"{field} must be a JSON object")
    return document


def _file_state(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular(path: Path, *, field: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RemoteEvidenceError("UNSAFE_INPUT_FILE", f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum:
            raise RemoteEvidenceError(
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
                raise RemoteEvidenceError("INPUT_TOO_LARGE", f"{field} exceeds its size limit")
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
            raise RemoteEvidenceError("INPUT_CHANGED", f"{field} changed while it was read")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_json_payload(path: Path, *, field: str) -> JsonPayload:
    payload = _read_regular(path, field=field, maximum=MAX_JSON_BYTES)
    metadata = os.lstat(path)
    return JsonPayload(
        path=path,
        payload=payload,
        document=_strict_json(payload, field=field),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        mode=stat.S_IMODE(metadata.st_mode),
    )


def _sha256_regular_file(path: Path, *, field: str) -> str:
    payload = _read_regular(path, field=field, maximum=MAX_JSON_BYTES)
    return hashlib.sha256(payload).hexdigest()


def _contract_schema(definition: str) -> Mapping[str, Any]:
    root = _strict_json(
        _read_regular(
            REPOSITORY_ROOT / contract.SCHEMA_PATH,
            field="publication receipt schema",
            maximum=MAX_JSON_BYTES,
        ),
        field="publication receipt schema",
    )
    definitions = root.get("$defs")
    if not isinstance(definitions, Mapping) or not isinstance(
        definitions.get(definition), Mapping
    ):
        raise RemoteEvidenceError(
            "CONTRACT_DEFINITION_MISSING", f"contract definition {definition!r} is unavailable"
        )
    return {"$schema": root.get("$schema"), "$defs": definitions, **definitions[definition]}


def _validate_contract(document: Mapping[str, Any], *, definition: str) -> None:
    schema = _contract_schema(definition)
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except Exception as exc:
        raise RemoteEvidenceError("CONTRACT_SCHEMA_INVALID", "evidence schema is invalid") from exc
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise RemoteEvidenceError(
            "CONTRACT_REJECTED", f"remote evidence contract failed at {location}"
        )
    if document.get("evidence_sha256") != canonical_json_hash(
        document, remove="evidence_sha256"
    ):
        raise RemoteEvidenceError("EVIDENCE_SELF_HASH_MISMATCH", "evidence self-hash failed")


def _required_str(value: object, *, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise RemoteEvidenceError("REMOTE_FIELD_INVALID", f"{field} is unavailable")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise RemoteEvidenceError("REMOTE_FIELD_INVALID", f"{field} is malformed")
    return value


def _required_int(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RemoteEvidenceError("REMOTE_FIELD_INVALID", f"{field} is malformed")
    return value


def _required_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RemoteEvidenceError("REMOTE_FIELD_INVALID", f"{field} is unavailable")
    return value


def _required_list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise RemoteEvidenceError("REMOTE_FIELD_INVALID", f"{field} is unavailable")
    return value


def _repo_path(value: object, *, field: str) -> str:
    path_text = _required_str(value, field=field)
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or path_text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RemoteEvidenceError(
            "UNSAFE_REPOSITORY_PATH", f"{field} is not a normalized relative path"
        )
    return path_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _collector_hash(mode: str, facts: Mapping[str, Any]) -> str:
    return canonical_json_hash(
        {
            "collector_path": COLLECTOR_PATH.as_posix(),
            "collector_version": COLLECTOR_VERSION,
            "mode": mode,
            "facts": facts,
        }
    )


class GhRestTransport:
    """Small read-only GitHub REST transport implemented through ``gh api``."""

    def __init__(self) -> None:
        executable = shutil.which("gh")
        if executable is None:
            raise RemoteEvidenceError("GH_UNAVAILABLE", "gh CLI is unavailable")
        self._executable = executable

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
        if not endpoint.startswith("/") or "\n" in endpoint or "\r" in endpoint:
            raise RemoteEvidenceError("UNSAFE_GITHUB_ENDPOINT", "GitHub endpoint is unsafe")
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

    def get_json(self, endpoint: str, *, params: Mapping[str, str] | None = None) -> Any:
        command = self._command(endpoint, accept="application/vnd.github+json")
        for key, value in sorted((params or {}).items()):
            if not re.fullmatch(r"[A-Za-z0-9_]+", key) or any(ch in value for ch in "\r\n"):
                raise RemoteEvidenceError("UNSAFE_GITHUB_QUERY", "GitHub query is unsafe")
            command.extend(["-f", f"{key}={value}"])
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=self._environment(),
            timeout=120,
        )
        if completed.returncode != 0:
            raise RemoteEvidenceError(
                "GITHUB_READ_FAILED", "GitHub REST read failed without exposing its response"
            )
        if len(completed.stdout) > MAX_JSON_BYTES:
            raise RemoteEvidenceError("GITHUB_RESPONSE_TOO_LARGE", "GitHub JSON exceeded limit")
        return _strict_json_value(completed.stdout, field="GitHub REST response")

    def download_digest(self, endpoint: str) -> tuple[str, int]:
        command = self._command(endpoint, accept="application/octet-stream")
        with tempfile.TemporaryFile() as output:
            completed = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.PIPE,
                env=self._environment(),
                timeout=300,
            )
            if completed.returncode != 0:
                raise RemoteEvidenceError(
                    "GITHUB_ASSET_DOWNLOAD_FAILED",
                    "GitHub release asset download failed without exposing its response",
                )
            output.seek(0)
            digest = hashlib.sha256()
            size = 0
            while block := output.read(READ_BLOCK_BYTES):
                digest.update(block)
                size += len(block)
            return digest.hexdigest(), size

    def download_bytes(self, endpoint: str, *, maximum: int) -> bytes:
        if maximum <= 0:
            raise RemoteEvidenceError(
                "INVALID_DOWNLOAD_LIMIT", "GitHub asset download limit is invalid"
            )
        command = self._command(endpoint, accept="application/octet-stream")
        with tempfile.TemporaryFile() as output:
            completed = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.PIPE,
                env=self._environment(),
                timeout=120,
            )
            if completed.returncode != 0:
                raise RemoteEvidenceError(
                    "GITHUB_ASSET_DOWNLOAD_FAILED",
                    "GitHub release asset download failed without exposing its response",
                )
            size = output.tell()
            if size <= 0 or size > maximum:
                raise RemoteEvidenceError(
                    "GITHUB_ASSET_DOWNLOAD_SIZE_INVALID",
                    "GitHub release checksum asset exceeded its strict size bound",
                )
            output.seek(0)
            payload = output.read(maximum + 1)
            if len(payload) != size:
                raise RemoteEvidenceError(
                    "GITHUB_ASSET_DOWNLOAD_CHANGED",
                    "GitHub release checksum asset changed while it was read",
                )
            return payload


class OpenSshSignatureVerifier:
    """Identify and locally verify an SSH signature without logging signed bytes."""

    def __init__(self) -> None:
        executable = shutil.which("ssh-keygen")
        if executable is None:
            raise RemoteEvidenceError("SSH_KEYGEN_UNAVAILABLE", "ssh-keygen is unavailable")
        self._executable = str(Path(executable).resolve(strict=True))

    def _run(
        self, arguments: Sequence[str], *, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self._executable, *arguments],
            input=stdin,
            check=False,
            capture_output=True,
            env={"LANG": "C", "LC_ALL": "C"},
            timeout=30,
        )

    def fingerprint(self, public_key: str) -> str:
        with tempfile.TemporaryDirectory(prefix="pii-signing-key-") as directory:
            key_path = Path(directory) / "key.pub"
            key_path.write_text(public_key.strip() + "\n", encoding="utf-8")
            completed = self._run(["-lf", str(key_path), "-E", "sha256"])
        if completed.returncode != 0:
            raise RemoteEvidenceError("SSH_KEY_INVALID", "GitHub signing key is malformed")
        match = re.search(rb"\b(SHA256:[A-Za-z0-9+/]{43})\b", completed.stdout)
        if match is None:
            raise RemoteEvidenceError("SSH_FINGERPRINT_MISSING", "SSH fingerprint is unavailable")
        return match.group(1).decode("ascii")

    def verify(self, *, public_key: str, payload: bytes, signature: bytes) -> bool:
        with tempfile.TemporaryDirectory(prefix="pii-tag-verification-") as directory:
            root = Path(directory)
            allowed = root / "allowed_signers"
            signature_path = root / "signature"
            allowed.write_text(f"release-signer {public_key.strip()}\n", encoding="utf-8")
            signature_path.write_bytes(signature)
            completed = self._run(
                [
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed),
                    "-I",
                    "release-signer",
                    "-n",
                    "git",
                    "-s",
                    str(signature_path),
                ],
                stdin=payload,
            )
        return completed.returncode == 0


def _github_content(document: Mapping[str, Any], *, expected_path: str) -> bytes:
    if document.get("type") != "file" or document.get("path") != expected_path:
        raise RemoteEvidenceError("GITHUB_SOURCE_FILE_MISMATCH", "GitHub source path mismatched")
    if document.get("encoding") != "base64":
        raise RemoteEvidenceError("GITHUB_SOURCE_ENCODING", "GitHub source encoding is invalid")
    encoded = _required_str(document.get("content"), field="GitHub source content")
    encoded = "".join(encoded.split())
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RemoteEvidenceError("GITHUB_SOURCE_ENCODING", "GitHub source is not base64") from exc
    if not payload or len(payload) > MAX_TEXT_BYTES:
        raise RemoteEvidenceError("GITHUB_SOURCE_SIZE", "GitHub source file size is invalid")
    return payload


def _exact_binding(text: str, *, key: str) -> str:
    pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?[`\"']?{re.escape(key)}[`\"']?\s*[:=]\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    values = []
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match is not None:
            values.append(match.group(1).strip().strip("`\"'").strip())
    if len(values) != 1:
        raise RemoteEvidenceError(
            "REFERENCE_BINDING_INVALID", f"remote text must contain one exact {key} binding"
        )
    return values[0]


def _github_repository_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_name": value.get("full_name"),
        "private": value.get("private"),
        "visibility": value.get("visibility"),
    }


def _github_release_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "id",
            "tag_name",
            "target_commitish",
            "draft",
            "prerelease",
            "created_at",
            "updated_at",
            "body",
        )
    }


def _github_asset_states(values: Sequence[Any]) -> list[dict[str, Any]]:
    states = []
    for value in values:
        asset = _required_mapping(value, field="GitHub release asset state")
        states.append(
            {
                key: asset.get(key)
                for key in (
                    "id",
                    "name",
                    "size",
                    "state",
                    "digest",
                    "created_at",
                    "updated_at",
                )
            }
        )
    return sorted(states, key=lambda item: (str(item["name"]), str(item["id"])))


def _parse_release_checksums(payload: bytes) -> dict[str, Any]:
    """Parse the exact two-line ``sha256sum`` manifest used by this release."""

    if not payload or len(payload) > MAX_RELEASE_CHECKSUMS_BYTES:
        raise RemoteEvidenceError(
            "GITHUB_CHECKSUMS_FORMAT_INVALID",
            "GitHub checksums.txt is empty or exceeds its strict size bound",
        )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RemoteEvidenceError(
            "GITHUB_CHECKSUMS_FORMAT_INVALID",
            "GitHub checksums.txt is not strict ASCII sha256sum text",
        ) from exc
    if "\r" in text or not text.endswith("\n"):
        raise RemoteEvidenceError(
            "GITHUB_CHECKSUMS_FORMAT_INVALID",
            "GitHub checksums.txt must use LF and end with exactly one complete line",
        )
    lines = text[:-1].split("\n")
    expected_names = (
        contract.REQUIRED_GITHUB_RELEASE_ASSETS["wheel"],
        contract.REQUIRED_GITHUB_RELEASE_ASSETS["sbom"],
    )
    if len(lines) != len(expected_names):
        raise RemoteEvidenceError(
            "GITHUB_CHECKSUMS_FORMAT_INVALID",
            "GitHub checksums.txt must contain exactly the wheel and SBOM lines",
        )
    entries: list[dict[str, str]] = []
    for line, expected_name in zip(lines, expected_names, strict=True):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._+@ -]*)", line)
        if match is None or match.group(2) != expected_name:
            raise RemoteEvidenceError(
                "GITHUB_CHECKSUMS_FORMAT_INVALID",
                "GitHub checksums.txt does not use the fixed ordered sha256sum format",
            )
        entries.append({"name": expected_name, "sha256": match.group(1)})
    return {
        "asset_name": contract.REQUIRED_GITHUB_RELEASE_ASSETS["checksums"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "format": "sha256sum_text_two_space_two_lines_lf",
        "line_count": 2,
        "entries": entries,
        "matched_downloaded_assets": True,
    }


def _verify_github_signature(
    *,
    transport: GitHubTransport,
    verifier: SignatureVerifier,
    verification: Mapping[str, Any],
    expected_tag: str,
    expected_target: str,
) -> dict[str, Any]:
    if verification.get("verified") is not True or verification.get("reason") != "valid":
        raise RemoteEvidenceError("GITHUB_TAG_UNVERIFIED", "GitHub did not verify the tag")
    signature_text = _required_str(
        verification.get("signature"), field="GitHub tag signature"
    )
    payload_text = _required_str(verification.get("payload"), field="GitHub tag payload")
    expected_prefix = f"object {expected_target}\ntype commit\ntag {expected_tag}\n"
    if not payload_text.startswith(expected_prefix):
        raise RemoteEvidenceError(
            "GITHUB_TAG_PAYLOAD_MISMATCH",
            "GitHub verification payload does not bind the annotated tag target",
        )
    verified_at = _required_str(
        verification.get("verified_at"), field="GitHub tag verification time"
    )
    keys = _required_list(
        transport.get_json(
            f"/users/{GITHUB_OWNER}/ssh_signing_keys", params={"per_page": "100"}
        ),
        field="GitHub SSH signing keys",
    )
    if len(keys) >= 100:
        raise RemoteEvidenceError(
            "SIGNING_KEY_INVENTORY_INCOMPLETE",
            "GitHub signing-key inventory exceeded one complete bounded page",
        )
    matched_key: str | None = None
    for item in keys:
        key_document = _required_mapping(item, field="GitHub SSH signing key")
        public_key = _required_str(key_document.get("key"), field="GitHub SSH public key")
        if (
            len(public_key) > 16 * 1024
            or "\n" in public_key
            or "\r" in public_key
            or len(public_key.split()) < 2
        ):
            raise RemoteEvidenceError(
                "SSH_KEY_INVALID", "GitHub SSH public key is not a single key line"
            )
        if verifier.fingerprint(public_key) == EXPECTED_SIGNING_KEY_FINGERPRINT:
            if matched_key is not None:
                raise RemoteEvidenceError(
                    "SIGNING_KEY_AMBIGUOUS", "expected GitHub signing key appears more than once"
                )
            matched_key = public_key
    if matched_key is None:
        raise RemoteEvidenceError(
            "SIGNING_KEY_NOT_REGISTERED", "expected GitHub signing key is not registered"
        )
    signature = signature_text.encode("utf-8")
    payload = payload_text.encode("utf-8")
    if not verifier.verify(public_key=matched_key, payload=payload, signature=signature):
        raise RemoteEvidenceError(
            "LOCAL_TAG_SIGNATURE_INVALID", "annotated tag failed local SSH verification"
        )
    return {
        "provider": "github",
        "verified": True,
        "reason": "valid",
        "verified_at": verified_at,
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signing_key_fingerprint": EXPECTED_SIGNING_KEY_FINGERPRINT,
        "local_cryptographic_verification": True,
    }


def collect_github_evidence(
    *,
    source_commit: str,
    workflow_run_id: int,
    release_id: int,
    transport: GitHubTransport,
    signature_verifier: SignatureVerifier,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Collect one complete GitHub evidence document without remote mutation."""

    if SHA1_RE.fullmatch(source_commit) is None:
        raise RemoteEvidenceError("SOURCE_COMMIT_INVALID", "source commit must be full SHA-1")
    _required_int(workflow_run_id, field="workflow run ID")
    _required_int(release_id, field="release ID")
    generator_file_sha256 = _sha256_regular_file(
        REPOSITORY_ROOT / COLLECTOR_PATH, field="remote evidence collector"
    )

    repository = _required_mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}"), field="GitHub repository"
    )
    if (
        repository.get("full_name") != GITHUB_REPOSITORY
        or repository.get("private") is not False
        or repository.get("visibility") != "public"
    ):
        raise RemoteEvidenceError("GITHUB_NOT_PUBLIC", "GitHub repository is not public")

    ref_endpoint = f"/repos/{GITHUB_REPOSITORY}/git/ref/tags/{RELEASE_TAG}"
    ref = _required_mapping(transport.get_json(ref_endpoint), field="GitHub tag ref")
    ref_object = _required_mapping(ref.get("object"), field="GitHub tag ref target")
    tag_object_sha = _required_str(
        ref_object.get("sha"), field="GitHub tag object SHA", pattern=SHA1_RE
    )
    if ref_object.get("type") != "tag":
        raise RemoteEvidenceError("TAG_NOT_ANNOTATED", "release tag is not annotated")
    tag = _required_mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}/git/tags/{tag_object_sha}"),
        field="GitHub annotated tag",
    )
    tag_target = _required_mapping(tag.get("object"), field="GitHub tag target")
    tag_target_sha = _required_str(
        tag_target.get("sha"), field="GitHub tag target SHA", pattern=SHA1_RE
    )
    if tag.get("tag") != RELEASE_TAG or tag_target.get("type") != "commit":
        raise RemoteEvidenceError("TAG_TARGET_INVALID", "annotated tag target is invalid")
    if tag_target_sha != source_commit:
        raise RemoteEvidenceError("TAG_SOURCE_MISMATCH", "tag does not target source commit")
    tag_verification = _verify_github_signature(
        transport=transport,
        verifier=signature_verifier,
        verification=_required_mapping(tag.get("verification"), field="GitHub tag verification"),
        expected_tag=RELEASE_TAG,
        expected_target=source_commit,
    )

    run = _required_mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}/actions/runs/{workflow_run_id}"),
        field="GitHub Actions run",
    )
    run_id = _required_int(run.get("id"), field="GitHub Actions run ID")
    run_attempt = _required_int(run.get("run_attempt"), field="CI run attempt")
    if (
        run_id != workflow_run_id
        or run.get("event") != "workflow_dispatch"
        or run.get("path") != WORKFLOW_PATH
        or run.get("head_sha") != source_commit
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        raise RemoteEvidenceError(
            "HOSTED_CI_MISMATCH", "GitHub hosted CI is not successful for exact source"
        )
    jobs = _required_mapping(
        transport.get_json(
            f"/repos/{GITHUB_REPOSITORY}/actions/runs/{workflow_run_id}"
            f"/attempts/{run_attempt}/jobs",
            params={"per_page": "100"},
        ),
        field="GitHub Actions jobs",
    )
    job_values = _required_list(jobs.get("jobs"), field="GitHub Actions job list")
    job_count = _required_int(
        jobs.get("total_count"), field="GitHub Actions job count", minimum=0
    )
    if job_count != len(job_values) or len(job_values) >= 100:
        raise RemoteEvidenceError(
            "HOSTED_CI_JOBS_INCOMPLETE", "GitHub Actions jobs were not collected completely"
        )
    completed_times = []
    job_names: list[str] = []
    for item in job_values:
        job = _required_mapping(item, field="GitHub Actions job")
        job_names.append(_required_str(job.get("name"), field="GitHub Actions job name"))
        if (
            job.get("head_sha") != source_commit
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            raise RemoteEvidenceError("HOSTED_CI_JOB_FAILED", "GitHub Actions job did not pass")
        if isinstance(job.get("completed_at"), str):
            completed_times.append(job["completed_at"])
    if not completed_times:
        raise RemoteEvidenceError("HOSTED_CI_TIME_MISSING", "CI completion time is unavailable")
    if tuple(sorted(job_names)) != REQUIRED_WORKFLOW_JOBS:
        raise RemoteEvidenceError(
            "HOSTED_CI_JOB_SET_MISMATCH", "GitHub Actions required job set mismatched"
        )
    completed_at = max(completed_times)

    release_endpoint = f"/repos/{GITHUB_REPOSITORY}/releases/{release_id}"
    release = _required_mapping(
        transport.get_json(release_endpoint), field="GitHub draft release"
    )
    if (
        release.get("id") != release_id
        or release.get("tag_name") != RELEASE_TAG
        or release.get("draft") is not True
        or release.get("prerelease") is not True
    ):
        raise RemoteEvidenceError(
            "DRAFT_RELEASE_MISMATCH", "GitHub Release is not the expected draft prerelease"
        )
    body = _required_str(release.get("body"), field="GitHub draft Release body")
    created_at = _required_str(release.get("created_at"), field="release creation time")
    updated_at = _required_str(release.get("updated_at"), field="release update time")
    assets_endpoint = f"{release_endpoint}/assets"
    raw_assets = _required_list(
        transport.get_json(assets_endpoint, params={"per_page": "100"}),
        field="GitHub release assets",
    )
    if len(raw_assets) >= 100:
        raise RemoteEvidenceError(
            "GITHUB_ASSET_INVENTORY_INCOMPLETE",
            "GitHub release asset inventory exceeded one complete bounded page",
        )
    asset_names = [
        _required_str(
            _required_mapping(item, field="GitHub release asset").get("name"),
            field="GitHub release asset name",
        )
        for item in raw_assets
    ]
    if len(asset_names) != len(set(asset_names)):
        raise RemoteEvidenceError(
            "GITHUB_ASSET_NAME_DUPLICATE", "GitHub release repeats an asset name"
        )
    required_asset_names = set(contract.REQUIRED_GITHUB_RELEASE_ASSETS.values())
    if set(asset_names) != required_asset_names or len(asset_names) != len(
        required_asset_names
    ):
        raise RemoteEvidenceError(
            "GITHUB_ASSET_SET_MISMATCH",
            "GitHub draft Release must contain exactly the fixed wheel, SBOM, and checksums",
        )
    evidence_assets: list[dict[str, Any]] = []
    roles = {value: key for key, value in contract.REQUIRED_GITHUB_RELEASE_ASSETS.items()}
    checksums_payload: bytes | None = None
    for item in raw_assets:
        asset = _required_mapping(item, field="GitHub release asset")
        asset_id = _required_int(asset.get("id"), field="GitHub release asset ID")
        name = _required_str(asset.get("name"), field="GitHub release asset name")
        expected_size = _required_int(
            asset.get("size"), field="GitHub release asset size", minimum=0
        )
        digest_text = _required_str(asset.get("digest"), field="GitHub release asset digest")
        if not digest_text.startswith("sha256:") or SHA256_RE.fullmatch(digest_text[7:]) is None:
            raise RemoteEvidenceError(
                "GITHUB_ASSET_DIGEST_INVALID", "GitHub asset has no SHA-256 digest"
            )
        asset_endpoint = f"/repos/{GITHUB_REPOSITORY}/releases/assets/{asset_id}"
        if name == contract.REQUIRED_GITHUB_RELEASE_ASSETS["checksums"]:
            if expected_size <= 0 or expected_size > MAX_RELEASE_CHECKSUMS_BYTES:
                raise RemoteEvidenceError(
                    "GITHUB_CHECKSUMS_SIZE_INVALID",
                    "GitHub checksums.txt metadata exceeds its strict size bound",
                )
            checksums_payload = transport.download_bytes(
                asset_endpoint, maximum=MAX_RELEASE_CHECKSUMS_BYTES
            )
            observed_digest = hashlib.sha256(checksums_payload).hexdigest()
            observed_size = len(checksums_payload)
        else:
            observed_digest, observed_size = transport.download_digest(asset_endpoint)
        if observed_size != expected_size or observed_digest != digest_text[7:]:
            raise RemoteEvidenceError(
                "GITHUB_ASSET_BYTES_MISMATCH", "downloaded GitHub asset differs from metadata"
            )
        role = roles[name]
        evidence_assets.append(
            {
                "asset_id": asset_id,
                "role": role,
                "name": name,
                "size_bytes": observed_size,
                "state": _required_str(asset.get("state"), field="GitHub asset state"),
                "github_digest_sha256": digest_text[7:],
                "downloaded_sha256": observed_digest,
            }
        )
    evidence_assets.sort(key=lambda item: item["name"])
    if checksums_payload is None:  # defensive: exact inventory check above should make this moot
        raise RemoteEvidenceError(
            "GITHUB_CHECKSUMS_MISSING", "GitHub checksums.txt was not downloaded"
        )
    checksums = _parse_release_checksums(checksums_payload)
    assets_by_name = {item["name"]: item for item in evidence_assets}
    for entry in checksums["entries"]:
        if assets_by_name[entry["name"]]["downloaded_sha256"] != entry["sha256"]:
            raise RemoteEvidenceError(
                "GITHUB_CHECKSUMS_DIGEST_MISMATCH",
                "GitHub checksums.txt does not match downloaded wheel and SBOM bytes",
            )

    content_document = _required_mapping(
        transport.get_json(
            f"/repos/{GITHUB_REPOSITORY}/contents/{GITHUB_MODEL_CARD_PATH}",
            params={"ref": source_commit},
        ),
        field="GitHub source Model Card",
    )
    source_card = _github_content(content_document, expected_path=GITHUB_MODEL_CARD_PATH)
    try:
        source_card_text = source_card.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RemoteEvidenceError("GITHUB_MODEL_CARD_ENCODING", "Model Card is not UTF-8") from exc
    if _exact_binding(source_card_text, key="hugging_face_repository") != (
        HUGGING_FACE_REPOSITORY
    ):
        raise RemoteEvidenceError(
            "GITHUB_HF_REFERENCE_MISMATCH", "source Model Card references another HF repo"
        )

    final_ref = _required_mapping(transport.get_json(ref_endpoint), field="final GitHub tag ref")
    final_repository = _required_mapping(
        transport.get_json(f"/repos/{GITHUB_REPOSITORY}"),
        field="final GitHub repository",
    )
    final_release = _required_mapping(
        transport.get_json(release_endpoint), field="final GitHub draft release"
    )
    final_assets = _required_list(
        transport.get_json(assets_endpoint, params={"per_page": "100"}),
        field="final GitHub release assets",
    )
    if (
        final_ref != ref
        or _github_repository_state(final_repository)
        != _github_repository_state(repository)
        or _github_release_state(final_release) != _github_release_state(release)
        or _github_asset_states(final_assets) != _github_asset_states(raw_assets)
    ):
        raise RemoteEvidenceError(
            "GITHUB_STATE_CHANGED", "GitHub tag or draft Release changed during collection"
        )

    timestamp = collected_at or _utc_now()
    facts = {
        "repository": GITHUB_REPOSITORY,
        "source_commit": source_commit,
        "tag_object_sha": tag_object_sha,
        "workflow_run_id": workflow_run_id,
        "jobs_sha256": canonical_json_hash(jobs),
        "release_id": release_id,
        "release_sha256": canonical_json_hash(_github_release_state(release)),
        "release_assets_sha256": canonical_json_hash(_github_asset_states(raw_assets)),
        "source_card_sha256": hashlib.sha256(source_card).hexdigest(),
        "assets": evidence_assets,
        "checksums": checksums,
    }
    if generator_file_sha256 != _sha256_regular_file(
        REPOSITORY_ROOT / COLLECTOR_PATH, field="remote evidence collector"
    ):
        raise RemoteEvidenceError(
            "COLLECTOR_CHANGED", "remote evidence collector changed during collection"
        )
    document: dict[str, Any] = {
        "schema_version": contract.GITHUB_EVIDENCE_SCHEMA_VERSION,
        "collector": {
            "name": COLLECTOR_PATH.name,
            "version": COLLECTOR_VERSION,
            "collected_at": timestamp,
            "endpoint": OFFICIAL_GITHUB_ENDPOINT,
            "network_accessed": True,
            "remote_mutation_performed": False,
            "generator_path": COLLECTOR_PATH.as_posix(),
            "generator_file_sha256": generator_file_sha256,
            "collection_run_sha256": _collector_hash("github", facts),
        },
        "repository": {"id": GITHUB_REPOSITORY, "visibility": "public"},
        "source_commit_sha": source_commit,
        "signed_tag": {
            "name": RELEASE_TAG,
            "ref_target_type": "tag",
            "ref_target_sha": tag_object_sha,
            "tag_object_sha": tag_object_sha,
            "tag_target_type": "commit",
            "tag_target_sha": tag_target_sha,
            "verification": tag_verification,
        },
        "hosted_ci": {
            "provider": "github_actions",
            "workflow_path": WORKFLOW_PATH,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "head_sha": source_commit,
            "status": "completed",
            "conclusion": "success",
            "completed_at": completed_at,
            "jobs_receipt_sha256": canonical_json_hash(jobs),
        },
        "release": {
            "release_id": _required_int(release.get("id"), field="release ID"),
            "tag_name": RELEASE_TAG,
            "resolved_source_sha": source_commit,
            "draft": True,
            "prerelease": True,
            "created_at": created_at,
            "updated_at": updated_at,
            "body": body,
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "optional_asset_names": [],
            "checksums": checksums,
            "assets": evidence_assets,
        },
        "cross_reference": {
            "path": GITHUB_MODEL_CARD_PATH,
            "file_sha256": hashlib.sha256(source_card).hexdigest(),
            "hugging_face_repository": HUGGING_FACE_REPOSITORY,
        },
        "evidence_sha256": "",
    }
    document["evidence_sha256"] = canonical_json_hash(document, remove="evidence_sha256")
    _validate_contract(document, definition=contract.GITHUB_EVIDENCE_DEFINITION)
    return document


def _object_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


class OfficialHuggingFaceTransport:
    """Read-only official Hub client. HF_TOKEN remains only in process memory/env."""

    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN")
        if token is None or not token.strip():
            raise RemoteEvidenceError("HF_TOKEN_MISSING", "HF_TOKEN is required in process env")
        if HfApi is None or hf_hub_download is None:
            raise RemoteEvidenceError("HF_CLIENT_UNAVAILABLE", "huggingface_hub is unavailable")
        self._token = token
        self._api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)

    def repo_info(self, *, revision: str) -> object:
        try:
            return self._api.repo_info(
                HUGGING_FACE_REPOSITORY,
                repo_type="model",
                revision=revision,
                files_metadata=True,
            )
        except Exception as exc:
            raise RemoteEvidenceError("HF_READ_FAILED", "official HF repo read failed") from exc

    def list_repo_files(self, *, revision: str) -> list[str]:
        try:
            values = self._api.list_repo_files(
                HUGGING_FACE_REPOSITORY,
                repo_type="model",
                revision=revision,
            )
        except Exception as exc:
            raise RemoteEvidenceError(
                "HF_READ_FAILED", "official HF inventory read failed"
            ) from exc
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise RemoteEvidenceError("HF_INVENTORY_INVALID", "official HF inventory is invalid")
        return values

    def read_file(self, *, revision: str, path: str) -> bytes:
        try:
            downloaded = hf_hub_download(
                repo_id=HUGGING_FACE_REPOSITORY,
                repo_type="model",
                revision=revision,
                filename=path,
                endpoint=OFFICIAL_HF_ENDPOINT,
                token=self._token,
                force_download=True,
            )
        except Exception as exc:
            raise RemoteEvidenceError("HF_READ_FAILED", "official HF text read failed") from exc
        try:
            resolved = Path(downloaded).resolve(strict=True)
        except OSError as exc:
            raise RemoteEvidenceError(
                "HF_READ_FAILED", "official HF text cache target is unavailable"
            ) from exc
        return _read_regular(
            resolved, field="downloaded HF Model Card", maximum=MAX_TEXT_BYTES
        )


def _validate_provenance(payload: JsonPayload) -> None:
    try:
        hf_provenance._validate_document(payload.document)
    except Exception as exc:
        raise RemoteEvidenceError(
            "HF_PROVENANCE_INVALID", "HF download provenance receipt is invalid"
        ) from exc


def _validate_package_verification(payload: JsonPayload) -> None:
    schema_path = REPOSITORY_ROOT / (
        "configs/release/community_v2_publication_package_verification.schema.json"
    )
    schema = _strict_json(
        _read_regular(schema_path, field="package verification schema", maximum=MAX_JSON_BYTES),
        field="package verification schema",
    )
    try:
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(
                payload.document
            ),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except Exception as exc:
        raise RemoteEvidenceError(
            "PACKAGE_VERIFICATION_SCHEMA_INVALID", "package verification schema is invalid"
        ) from exc
    if errors or payload.document.get("receipt_sha256") != canonical_json_hash(
        payload.document, remove="receipt_sha256"
    ):
        raise RemoteEvidenceError(
            "PACKAGE_VERIFICATION_INVALID", "package verification receipt is invalid"
        )


def _hf_current_inventory(info: object, paths: list[str]) -> dict[str, dict[str, Any]]:
    siblings = _object_field(info, "siblings")
    if not isinstance(siblings, list):
        raise RemoteEvidenceError("HF_METADATA_MISSING", "HF per-file metadata is unavailable")
    result: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        path = _repo_path(_object_field(sibling, "rfilename"), field="HF remote path")
        if path in result:
            raise RemoteEvidenceError("HF_PATH_DUPLICATE", "HF metadata repeats a path")
        size = _object_field(sibling, "size")
        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
        ):
            raise RemoteEvidenceError("HF_SIZE_INVALID", "HF file size metadata is invalid")
        blob_id = _object_field(sibling, "blob_id")
        lfs = _object_field(sibling, "lfs")
        lfs_sha = _object_field(lfs, "sha256") if lfs is not None else None
        if lfs_sha is not None:
            oid = _required_str(lfs_sha, field="HF LFS OID", pattern=SHA256_RE)
        else:
            oid = _required_str(blob_id, field="HF Git blob OID", pattern=SHA1_RE)
        result[path] = {"size_bytes": size, "remote_oid": oid}
    if sorted(result) != paths:
        raise RemoteEvidenceError("HF_INVENTORY_MISMATCH", "HF metadata and path list differ")
    return dict(sorted(result.items()))


def collect_hugging_face_evidence(
    *,
    download_provenance: JsonPayload,
    package_verification: JsonPayload,
    transport: HuggingFaceTransport,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Collect HF remote evidence without repeating the 2.3 GB model download."""

    if download_provenance.mode != 0o444 or package_verification.mode != 0o444:
        raise RemoteEvidenceError(
            "INPUT_MODE_INVALID", "HF input receipts must be immutable mode 0444"
        )
    generator_file_sha256 = _sha256_regular_file(
        REPOSITORY_ROOT / COLLECTOR_PATH, field="remote evidence collector"
    )
    _validate_provenance(download_provenance)
    _validate_package_verification(package_verification)
    provenance = download_provenance.document
    verification = package_verification.document
    immutable_sha = _required_str(
        provenance.get("resolved_commit"), field="HF provenance commit", pattern=SHA1_RE
    )
    if (
        provenance.get("repository") != HUGGING_FACE_REPOSITORY
        or provenance.get("requested_revision") != immutable_sha
        or provenance.get("remote_snapshot", {}).get("private") is not True
    ):
        raise RemoteEvidenceError(
            "HF_PROVENANCE_TARGET_MISMATCH", "HF provenance targets another state"
        )
    if (
        verification.get("context") != "hugging_face_immutable_download"
        or verification.get("status") != "PASS"
    ):
        raise RemoteEvidenceError(
            "PACKAGE_VERIFICATION_CONTEXT_MISMATCH", "package verification is not immutable HF"
        )
    target = _required_mapping(verification.get("target"), field="verification target")
    source_control = _required_mapping(
        verification.get("source_control"), field="verification source control"
    )
    source_commit = _required_str(
        source_control.get("git_source_commit"), field="GitHub source commit", pattern=SHA1_RE
    )
    if (
        target.get("github_repository") != GITHUB_REPOSITORY
        or target.get("hugging_face_repository") != HUGGING_FACE_REPOSITORY
        or target.get("hugging_face_commit") != immutable_sha
    ):
        raise RemoteEvidenceError(
            "PACKAGE_VERIFICATION_TARGET_MISMATCH", "package verification target mismatched"
        )
    provenance_binding = _required_mapping(
        verification.get("hf_download_provenance"), field="verification provenance binding"
    )
    if (
        provenance_binding.get("file_sha256") != download_provenance.file_sha256
        or provenance_binding.get("receipt_sha256") != provenance.get("receipt_sha256")
        or provenance_binding.get("repository") != HUGGING_FACE_REPOSITORY
        or provenance_binding.get("resolved_commit") != immutable_sha
    ):
        raise RemoteEvidenceError(
            "PACKAGE_PROVENANCE_BINDING_MISMATCH",
            "package verification does not bind exact HF provenance",
        )

    info = transport.repo_info(revision=immutable_sha)
    if (
        _object_field(info, "id") != HUGGING_FACE_REPOSITORY
        or _object_field(info, "sha") != immutable_sha
        or _object_field(info, "private") is not True
    ):
        raise RemoteEvidenceError(
            "HF_REMOTE_STATE_MISMATCH", "HF remote state is not private exact"
        )
    paths = sorted(
        _repo_path(path, field="HF inventory path")
        for path in transport.list_repo_files(revision=immutable_sha)
    )
    if len(paths) != len(set(paths)):
        raise RemoteEvidenceError("HF_PATH_DUPLICATE", "HF inventory repeats a path")
    local_files = _required_mapping(
        _required_mapping(provenance.get("local_root"), field="HF local root").get("files"),
        field="HF local inventory",
    )
    if paths != sorted(local_files):
        raise RemoteEvidenceError(
            "HF_PROVENANCE_INVENTORY_MISMATCH", "current HF inventory differs from provenance"
        )
    current = _hf_current_inventory(info, paths)
    remote_snapshot = _required_mapping(
        provenance.get("remote_snapshot"), field="HF provenance remote snapshot"
    )
    remote_files = _required_mapping(
        remote_snapshot.get("files"), field="HF provenance remote files"
    )
    evidence_inventory: list[dict[str, Any]] = []
    for path in paths:
        local = _required_mapping(local_files[path], field=f"HF local binding {path}")
        old_remote = _required_mapping(remote_files[path], field=f"HF remote binding {path}")
        now_remote = current[path]
        local_size = _required_int(
            local.get("size_bytes"), field=f"downloaded size {path}"
        )
        if old_remote.get("size_bytes") != local_size or (
            now_remote["size_bytes"] is not None
            and now_remote["size_bytes"] != local_size
        ):
            raise RemoteEvidenceError("HF_FILE_SIZE_MISMATCH", "HF file size changed")
        expected_oid = old_remote.get("lfs_oid_sha256") or old_remote.get("git_blob_oid")
        if expected_oid != now_remote["remote_oid"]:
            raise RemoteEvidenceError("HF_FILE_OID_MISMATCH", "HF remote OID changed")
        digest = _required_str(
            local.get("file_sha256"), field=f"downloaded SHA-256 {path}", pattern=SHA256_RE
        )
        evidence_inventory.append(
            {
                "path": path,
                "kind": "file",
                "size_bytes": local_size,
                "remote_oid": now_remote["remote_oid"],
                "downloaded_sha256": digest,
            }
        )

    readme = transport.read_file(revision=immutable_sha, path=HF_MODEL_CARD_PATH)
    if hashlib.sha256(readme).hexdigest() != local_files[HF_MODEL_CARD_PATH].get(
        "file_sha256"
    ):
        raise RemoteEvidenceError("HF_README_HASH_MISMATCH", "HF README differs from provenance")
    try:
        readme_text = readme.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RemoteEvidenceError("HF_README_ENCODING", "HF README is not UTF-8") from exc
    if (
        _exact_binding(readme_text, key="github_repository") != GITHUB_REPOSITORY
        or _exact_binding(readme_text, key="git_source_commit") != source_commit
        or _exact_binding(readme_text, key="release_tag") != RELEASE_TAG
    ):
        raise RemoteEvidenceError(
            "HF_SOURCE_REFERENCE_MISMATCH", "HF README source binding drifted"
        )

    final_info = transport.repo_info(revision=immutable_sha)
    final_paths = sorted(
        _repo_path(path, field="final HF inventory path")
        for path in transport.list_repo_files(revision=immutable_sha)
    )
    if (
        _object_field(final_info, "id") != HUGGING_FACE_REPOSITORY
        or _object_field(final_info, "sha") != immutable_sha
        or _object_field(final_info, "private") is not True
        or final_paths != paths
        or _hf_current_inventory(final_info, final_paths) != current
    ):
        raise RemoteEvidenceError("HF_STATE_CHANGED", "HF state changed during collection")

    package_identity = _required_mapping(
        verification.get("package_identity"), field="package identity"
    )
    facts = {
        "repository": HUGGING_FACE_REPOSITORY,
        "immutable_sha": immutable_sha,
        "inventory": evidence_inventory,
        "readme_sha256": hashlib.sha256(readme).hexdigest(),
        "provenance_file_sha256": download_provenance.file_sha256,
        "verification_file_sha256": package_verification.file_sha256,
    }
    timestamp = collected_at or _utc_now()
    if generator_file_sha256 != _sha256_regular_file(
        REPOSITORY_ROOT / COLLECTOR_PATH, field="remote evidence collector"
    ):
        raise RemoteEvidenceError(
            "COLLECTOR_CHANGED", "remote evidence collector changed during collection"
        )
    document: dict[str, Any] = {
        "schema_version": contract.HUGGING_FACE_EVIDENCE_SCHEMA_VERSION,
        "collector": {
            "name": COLLECTOR_PATH.name,
            "version": COLLECTOR_VERSION,
            "collected_at": timestamp,
            "endpoint": OFFICIAL_HF_ENDPOINT,
            "network_accessed": True,
            "remote_mutation_performed": False,
            "generator_path": COLLECTOR_PATH.as_posix(),
            "generator_file_sha256": generator_file_sha256,
            "collection_run_sha256": _collector_hash("hugging_face", facts),
        },
        "repository": {"id": HUGGING_FACE_REPOSITORY, "visibility": "private"},
        "revision": {
            "requested_revision": immutable_sha,
            "resolved_sha": immutable_sha,
            "immutable_sha": immutable_sha,
        },
        "inventory": evidence_inventory,
        "cross_reference": {
            "path": HF_MODEL_CARD_PATH,
            "file_sha256": hashlib.sha256(readme).hexdigest(),
            "github_repository": GITHUB_REPOSITORY,
            "github_source_sha": source_commit,
        },
        "package_verification": {
            "schema_version": PACKAGE_VERIFICATION_SCHEMA_VERSION,
            "file_sha256": package_verification.file_sha256,
            "receipt_sha256": verification["receipt_sha256"],
            "context": "hugging_face_immutable_download",
            "status": "PASS",
            "hugging_face_commit": immutable_sha,
            "github_source_commit": source_commit,
            "hf_download_provenance_file_sha256": download_provenance.file_sha256,
            "hf_download_provenance_receipt_sha256": provenance["receipt_sha256"],
            "package_identity": dict(package_identity),
        },
        "evidence_sha256": "",
    }
    document["evidence_sha256"] = canonical_json_hash(document, remove="evidence_sha256")
    _validate_contract(document, definition=contract.HUGGING_FACE_EVIDENCE_DEFINITION)
    return document


def _write_evidence(path: Path, document: Mapping[str, Any]) -> str:
    collector_metadata = document.get("collector")
    if isinstance(collector_metadata, Mapping) and collector_metadata.get(
        "generator_path"
    ) == COLLECTOR_PATH.as_posix():
        current_generator_sha256 = _sha256_regular_file(
            REPOSITORY_ROOT / COLLECTOR_PATH, field="remote evidence collector"
        )
        if collector_metadata.get("generator_file_sha256") != current_generator_sha256:
            raise RemoteEvidenceError(
                "COLLECTOR_CHANGED", "collector changed before evidence write"
            )
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.name in {"", ".", ".."}:
        raise RemoteEvidenceError("OUTPUT_PATH_UNSAFE", "output name is unsafe")
    try:
        parent = os.lstat(path.parent)
    except OSError as exc:
        raise RemoteEvidenceError("OUTPUT_PARENT_MISSING", "output parent is unavailable") from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise RemoteEvidenceError("OUTPUT_PARENT_UNSAFE", "output parent must be a real directory")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = os.open(path.parent, parent_flags)
    except OSError as exc:
        raise RemoteEvidenceError("OUTPUT_PARENT_UNSAFE", "output parent is unsafe") from exc
    parent_identity = (parent.st_dev, parent.st_ino)
    opened_parent = os.fstat(parent_descriptor)
    if (opened_parent.st_dev, opened_parent.st_ino) != parent_identity:
        os.close(parent_descriptor)
        raise RemoteEvidenceError("OUTPUT_PARENT_CHANGED", "output parent changed")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    identity: tuple[int, int] | None = None

    def remove_owned_output() -> None:
        if identity is None:
            return
        try:
            current = os.stat(
                path.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except OSError:
            return
        if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == identity:
            try:
                os.unlink(path.name, dir_fd=parent_descriptor)
            except OSError:
                pass

    try:
        descriptor = os.open(path.name, flags, 0o444, dir_fd=parent_descriptor)
        metadata = os.fstat(descriptor)
        identity = metadata.st_dev, metadata.st_ino
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
        os.close(parent_descriptor)
        raise RemoteEvidenceError("OUTPUT_EXISTS", "evidence output is no-clobber") from exc
    except BaseException as exc:
        remove_owned_output()
        os.close(parent_descriptor)
        if isinstance(exc, OSError):
            raise RemoteEvidenceError("OUTPUT_WRITE_FAILED", "evidence output failed") from exc
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        final_parent = os.lstat(path.parent)
    except OSError as exc:
        remove_owned_output()
        os.close(parent_descriptor)
        raise RemoteEvidenceError(
            "OUTPUT_VERIFICATION_FAILED", "evidence output could not be finalized"
        ) from exc
    valid = (
        identity is not None
        and stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
        and (final_descriptor.st_dev, final_descriptor.st_ino) == identity
        and final_descriptor.st_size == len(payload)
        and stat.S_IMODE(current.st_mode) == 0o444
        and current.st_size == len(payload)
        and not stat.S_ISLNK(final_parent.st_mode)
        and (final_parent.st_dev, final_parent.st_ino) == parent_identity
    )
    if not valid:
        remove_owned_output()
        os.close(parent_descriptor)
        raise RemoteEvidenceError("OUTPUT_VERIFICATION_FAILED", "evidence output drifted")
    os.close(parent_descriptor)
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    github = subparsers.add_parser("github")
    github.add_argument("--source-commit", required=True)
    github.add_argument("--workflow-run-id", required=True, type=int)
    github.add_argument("--release-id", required=True, type=int)
    github.add_argument("--output", required=True, type=Path)
    hugging_face = subparsers.add_parser("hugging-face")
    hugging_face.add_argument("--download-provenance", required=True, type=Path)
    hugging_face.add_argument("--package-verification-receipt", required=True, type=Path)
    hugging_face.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "github":
            document = collect_github_evidence(
                source_commit=args.source_commit,
                workflow_run_id=args.workflow_run_id,
                release_id=args.release_id,
                transport=GhRestTransport(),
                signature_verifier=OpenSshSignatureVerifier(),
            )
        else:
            provenance = load_json_payload(
                args.download_provenance, field="HF download provenance"
            )
            verification = load_json_payload(
                args.package_verification_receipt,
                field="HF immutable package verification receipt",
            )
            document = collect_hugging_face_evidence(
                download_provenance=provenance,
                package_verification=verification,
                transport=OfficialHuggingFaceTransport(),
            )
        file_sha256 = _write_evidence(args.output, document)
        print(
            json.dumps(
                {
                    "status": "COLLECTED",
                    "mode": args.mode,
                    "evidence_sha256": document["evidence_sha256"],
                    "file_sha256": file_sha256,
                    "remote_mutation_performed": False,
                    "sensitive_values_logged": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except RemoteEvidenceError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "mode": args.mode,
                    "blocker_ids": [exc.blocker_id],
                    "remote_mutation_performed": False,
                    "sensitive_values_logged": False,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
