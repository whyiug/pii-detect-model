#!/usr/bin/env python3
"""Build an honest maintainer waiver for the private-channel RC gate.

The offline generator binds a human-authored waiver and ``SECURITY.md`` to an
exact publication target.  It records that GitHub Private Vulnerability
Reporting is enabled while the independent end-to-end test remains incomplete.
It never reads advisory content, queries remote state, or represents the test as
passed.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path("configs/release/private_security_channel_waiver_receipt.schema.json")
SCHEMA_VERSION = "pii-zh.community-v2-private-security-channel-waiver-receipt.v1"
PACKAGE_VERSION = "0.2.0rc1"
PROVIDER = "github_private_vulnerability_reporting"
DECISION = "maintainer_waived_for_release_candidate"
EVIDENCE_BASIS = "explicit_human_maintainer_attestation_and_bound_waiver_file"

_MAX_BOUND_FILE_BYTES = 8 * 1024 * 1024
_REPOSITORY_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)
_GITHUB_LOGIN = re.compile(r"^(?=.{1,39}$)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
_WAIVER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_GITHUB_URL = re.compile(r"https://github\.com/[^\s<>()\[\]{}\"']+")
_RFC3339 = re.compile(
    r"^(?:[0-9]{4})-(?:[0-9]{2})-(?:[0-9]{2})T"
    r"(?:[0-9]{2}):(?:[0-9]{2}):(?:[0-9]{2})"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_INCOMPLETE_SECURITY_MARKERS = (
    "has not yet been recorded",
    "not yet been recorded",
    "not been completed",
    "not completed",
    "not complete",
    "still pending",
    "remains outstanding",
)
_TEST_OR_CHANNEL_WORD = re.compile(r"\b(?:test(?:ed)?|check|channel)\b")
_POSITIVE_OUTCOME_WORD = re.compile(
    r"\b(?:pass(?:ed)?|success(?:ful(?:ly)?)?|succeed(?:ed)?|complete(?:d)?|accept(?:ed)?)\b"
)
_PVR_SUBJECT = (
    r"(?:github\s+private\s+vulnerability\s+reporting|"
    r"private\s+vulnerability\s+reporting|pvr)"
)
_CONTRADICTORY_PVR_PATTERNS = (
    re.compile(rf"\b{_PVR_SUBJECT}\b[^\r\n.!?]{{0,80}}\b(?:disabled|inactive|off)\b"),
    re.compile(rf"\b{_PVR_SUBJECT}\b[^\r\n.!?]{{0,80}}\bnot\s+enabled\b"),
    re.compile(rf"\b{_PVR_SUBJECT}\b[^\r\n.!?]{{0,80}}\benabled\s*:\s*false\b"),
    re.compile(
        rf"\bfalse\s+that\b[^\r\n.!?]{{0,80}}\b{_PVR_SUBJECT}\b"
        rf"[^\r\n.!?]{{0,40}}\benabled\b"
    ),
)


class SecurityChannelWaiverReceiptError(RuntimeError):
    """Raised when a waiver receipt cannot be prepared or written safely."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class BoundPayload:
    payload: bytes
    text: str
    file_sha256: str


@dataclass(frozen=True)
class ReceiptPlan:
    output: Path
    document: Mapping[str, Any]
    serialized: bytes
    security_file_sha256: str
    waiver_file_sha256: str


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
        raise SecurityChannelWaiverReceiptError(
            "INVALID_CANONICAL_JSON", "receipt is not canonical-JSON serializable"
        ) from exc


def canonical_json_hash(value: Mapping[str, Any], *, remove: str | None = None) -> str:
    """Return the lowercase SHA-256 of the canonical JSON representation."""

    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _open_read_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _read_bound_file(path: Path, *, kind: str) -> BoundPayload:
    blocker_prefix = kind.upper()
    try:
        path_metadata = os.lstat(path)
    except OSError as exc:
        raise SecurityChannelWaiverReceiptError(
            f"{blocker_prefix}_READ_FAILED", f"{kind} input cannot be inspected safely"
        ) from exc
    if stat.S_ISLNK(path_metadata.st_mode):
        raise SecurityChannelWaiverReceiptError(
            f"{blocker_prefix}_SYMLINK_REJECTED", f"{kind} input cannot be a symlink"
        )
    if not stat.S_ISREG(path_metadata.st_mode):
        raise SecurityChannelWaiverReceiptError(
            f"{blocker_prefix}_NOT_REGULAR", f"{kind} input must be a regular file"
        )
    try:
        descriptor = os.open(path, _open_read_flags())
    except OSError as exc:
        blocker = (
            f"{blocker_prefix}_SYMLINK_REJECTED"
            if exc.errno == errno.ELOOP
            else f"{blocker_prefix}_READ_FAILED"
        )
        raise SecurityChannelWaiverReceiptError(
            blocker, f"{kind} input cannot be opened safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SecurityChannelWaiverReceiptError(
                f"{blocker_prefix}_NOT_REGULAR", f"{kind} input must be a regular file"
            )
        if metadata.st_size > _MAX_BOUND_FILE_BYTES:
            raise SecurityChannelWaiverReceiptError(
                f"{blocker_prefix}_TOO_LARGE", f"{kind} input exceeds 8 MiB"
            )
        if (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino):
            raise SecurityChannelWaiverReceiptError(
                f"{blocker_prefix}_CHANGED_DURING_OPEN", f"{kind} input changed while opening"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor, min(1024 * 1024, _MAX_BOUND_FILE_BYTES + 1 - observed)
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > _MAX_BOUND_FILE_BYTES:
                raise SecurityChannelWaiverReceiptError(
                    f"{blocker_prefix}_TOO_LARGE", f"{kind} input exceeds 8 MiB"
                )
        payload = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(metadata, field) != getattr(final_metadata, field)
            for field in stable_fields
        ) or len(payload) != final_metadata.st_size:
            raise SecurityChannelWaiverReceiptError(
                f"{blocker_prefix}_CHANGED_DURING_READ", f"{kind} input changed while reading"
            )
    finally:
        os.close(descriptor)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecurityChannelWaiverReceiptError(
            f"{blocker_prefix}_NOT_UTF8", f"{kind} input must be UTF-8"
        ) from exc
    if not text.strip() or "\x00" in text:
        raise SecurityChannelWaiverReceiptError(
            f"{blocker_prefix}_INVALID_TEXT", f"{kind} input is empty or contains NUL"
        )
    return BoundPayload(
        payload=payload,
        text=text,
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _target_reporting_url(github_repository: str) -> str:
    if _REPOSITORY_ID.fullmatch(github_repository) is None:
        raise SecurityChannelWaiverReceiptError(
            "INVALID_GITHUB_REPOSITORY", "GitHub repository is not a valid owner/repository id"
        )
    return f"https://github.com/{github_repository}/security/advisories/new"


def _normalize_authorized_by(authorized_by: str) -> str:
    normalized = authorized_by.strip()
    if _GITHUB_LOGIN.fullmatch(normalized) is None:
        raise SecurityChannelWaiverReceiptError(
            "AUTHORIZED_BY_INVALID", "authorized_by must be a stable public GitHub login"
        )
    return normalized


def _normalize_waiver_id(waiver_id: str) -> str:
    normalized = waiver_id.strip()
    if _WAIVER_ID.fullmatch(normalized) is None:
        raise SecurityChannelWaiverReceiptError(
            "WAIVER_ID_INVALID", "waiver_id is not a stable machine-readable identifier"
        )
    return normalized


def _normalize_git_source_commit(git_source_commit: str) -> str:
    normalized = git_source_commit.strip()
    if _GIT_COMMIT.fullmatch(normalized) is None:
        raise SecurityChannelWaiverReceiptError(
            "INVALID_GIT_SOURCE_COMMIT",
            "git_source_commit must be a full lowercase 40- or 64-hex commit id",
        )
    return normalized


def _parse_authorized_at(value: str, *, now: datetime) -> datetime:
    if _RFC3339.fullmatch(value) is None:
        raise SecurityChannelWaiverReceiptError(
            "AUTHORIZED_AT_NOT_RFC3339",
            "authorized_at must be an RFC3339 timestamp with a timezone",
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise SecurityChannelWaiverReceiptError(
            "AUTHORIZED_AT_NOT_RFC3339", "authorized_at is not a valid RFC3339 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SecurityChannelWaiverReceiptError(
            "AUTHORIZED_AT_NOT_RFC3339", "authorized_at must include a timezone"
        )
    if now.tzinfo is None or now.utcoffset() is None:
        raise SecurityChannelWaiverReceiptError(
            "INVALID_CLOCK", "comparison clock must be timezone-aware"
        )
    if parsed.astimezone(timezone.utc) > now.astimezone(timezone.utc):
        raise SecurityChannelWaiverReceiptError(
            "AUTHORIZED_AT_IN_FUTURE", "authorized_at cannot be later than the current time"
        )
    return parsed


def _contains_positive_test_claim(line: str) -> bool:
    normalized = line.casefold()
    return bool(
        _TEST_OR_CHANNEL_WORD.search(normalized)
        and _POSITIVE_OUTCOME_WORD.search(normalized)
    )


def _validate_security(payload: BoundPayload, *, expected_reporting_url: str) -> None:
    observed_urls = {
        match.group(0).rstrip(".,;:!?") for match in _GITHUB_URL.finditer(payload.text)
    }
    if expected_reporting_url not in observed_urls:
        raise SecurityChannelWaiverReceiptError(
            "SECURITY_REPORTING_URL_MISSING",
            "SECURITY input does not contain the exact target private-report URL",
        )
    normalized = " ".join(payload.text.lower().split())
    if "github private vulnerability reporting is enabled" not in normalized:
        raise SecurityChannelWaiverReceiptError(
            "SECURITY_CHANNEL_NOT_ENABLED",
            "SECURITY input does not explicitly state that GitHub PVR is enabled",
        )
    if not any(marker in normalized for marker in _INCOMPLETE_SECURITY_MARKERS):
        raise SecurityChannelWaiverReceiptError(
            "SECURITY_INCOMPLETE_TEST_STATUS_MISSING",
            "SECURITY input does not explicitly retain the incomplete independent-test status",
        )
    contradictory_completion = any(
        _contains_positive_test_claim(line) for line in payload.text.splitlines()
    )
    contradictory_enabled_state = any(
        pattern.search(line.casefold())
        for line in payload.text.splitlines()
        for pattern in _CONTRADICTORY_PVR_PATTERNS
    )
    if contradictory_completion or contradictory_enabled_state:
        raise SecurityChannelWaiverReceiptError(
            "SECURITY_CONTRADICTORY_TEST_STATUS",
            "SECURITY input contains a contradictory reporting-channel or test-status claim",
        )


def _waiver_evidence_lines(
    *,
    package_version: str,
    github_repository: str,
    hugging_face_repository: str,
    authorized_by: str,
    authorized_at: str,
    waiver_id: str,
) -> set[str]:
    return {
        f"Waiver-ID: {waiver_id}",
        f"Authorized-by: {authorized_by}",
        f"Authorized-at: {authorized_at}",
        f"Package-version: {package_version}",
        f"GitHub-repository: {github_repository}",
        f"Hugging-Face-repository: {hugging_face_repository}",
        f"Provider: {PROVIDER}",
        "Enabled: true",
        "Independent-test-completed: false",
        f"Decision: {DECISION}",
        f"Evidence-basis: {EVIDENCE_BASIS}",
    }


def _machine_evidence_entry(raw_line: str) -> tuple[str, str] | None:
    """Normalize a Markdown evidence line without hiding key aliases."""

    line = raw_line.strip()
    while True:
        previous = line
        line = re.sub(r"^(?:[>+*-]\s*)+", "", line).lstrip()
        line = re.sub(r"^\d+[.)]\s*", "", line).lstrip()
        if line == previous:
            break
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    normalized_key = key.strip().strip("`*_~#[](){} ").casefold()
    if not normalized_key:
        return None
    return normalized_key, value.strip()


def _validate_waiver(payload: BoundPayload, *, required_lines: set[str]) -> None:
    required_by_key: dict[str, str] = {}
    for line in required_lines:
        entry = _machine_evidence_entry(line)
        if entry is None:
            raise SecurityChannelWaiverReceiptError(
                "WAIVER_EVIDENCE_CONTRACT_INVALID",
                "internal waiver evidence contract contains an invalid machine line",
            )
        key, value = entry
        required_by_key[key] = value
    observed_by_key: dict[str, list[str]] = {key: [] for key in required_by_key}
    observed_entries: list[tuple[str, str]] = []
    for raw_line in payload.text.splitlines():
        entry = _machine_evidence_entry(raw_line)
        if entry is None:
            continue
        key, value = entry
        observed_entries.append(entry)
        if key in observed_by_key:
            observed_by_key[key].append(value)
    missing_or_conflicting = sorted(
        key for key, expected_value in required_by_key.items()
        if observed_by_key[key] != [expected_value]
    )
    if missing_or_conflicting:
        raise SecurityChannelWaiverReceiptError(
            "WAIVER_EVIDENCE_INCOMPLETE",
            "waiver input must contain exactly one copy of each expected evidence key and value",
        )
    contradictory_machine_claim = any(
        (key == "tested" and value.casefold() == "true")
        or (key == "outcome" and value.casefold() == "accepted_private_test_report")
        or (key == "independent-test-completed" and value.casefold() != "false")
        or (key == "enabled" and value.casefold() != "true")
        for key, value in observed_entries
    )
    contradictory_prose_claim = False
    for line in payload.text.splitlines():
        entry = _machine_evidence_entry(line)
        if entry == ("independent-test-completed", "false"):
            continue
        if _contains_positive_test_claim(line):
            contradictory_prose_claim = True
            break
    if contradictory_machine_claim or contradictory_prose_claim:
        raise SecurityChannelWaiverReceiptError(
            "WAIVER_CONTRADICTORY_TEST_STATUS",
            "waiver input contains a contradictory tested-channel claim",
        )


def _schema() -> Mapping[str, Any]:
    schema_file = REPOSITORY_ROOT / SCHEMA_PATH
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SecurityChannelWaiverReceiptError(
            "RECEIPT_SCHEMA_UNAVAILABLE", "receipt schema cannot be loaded"
        ) from exc
    if not isinstance(schema, dict):
        raise SecurityChannelWaiverReceiptError(
            "RECEIPT_SCHEMA_INVALID", "receipt schema must be a JSON object"
        )
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise SecurityChannelWaiverReceiptError(
            "RECEIPT_SCHEMA_INVALID", "receipt schema is not valid Draft 2020-12"
        ) from exc
    return schema


def _validate_document(document: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(document)
    except SecurityChannelWaiverReceiptError:
        raise
    except Exception as exc:
        raise SecurityChannelWaiverReceiptError(
            "RECEIPT_SCHEMA_REJECTED", "receipt does not satisfy its closed schema"
        ) from exc


def _validate_output(output: Path) -> None:
    try:
        parent = os.lstat(output.parent)
    except OSError as exc:
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_PARENT_UNAVAILABLE", "receipt output parent does not exist"
        ) from exc
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_PARENT_UNSAFE", "receipt output parent must be a real directory"
        )
    try:
        os.lstat(output)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_INSPECTION_FAILED", "receipt output cannot be inspected"
        ) from exc
    raise SecurityChannelWaiverReceiptError("OUTPUT_ALREADY_EXISTS", "receipt output is no-clobber")


def prepare_receipt(
    *,
    output: Path,
    security: Path,
    waiver: Path,
    package_version: str,
    github_repository: str,
    hugging_face_repository: str,
    authorized_by: str,
    authorized_at: str,
    waiver_id: str,
    git_source_commit: str,
    independent_test_not_completed_attested: bool,
    maintainer_waiver_attested: bool,
    now: datetime | None = None,
) -> ReceiptPlan:
    """Validate all inputs and prepare, but do not write, the waiver receipt."""

    if not independent_test_not_completed_attested:
        raise SecurityChannelWaiverReceiptError(
            "INDEPENDENT_TEST_INCOMPLETE_ATTESTATION_MISSING",
            "a human must attest that the independent end-to-end test is not completed",
        )
    if not maintainer_waiver_attested:
        raise SecurityChannelWaiverReceiptError(
            "MAINTAINER_WAIVER_ATTESTATION_MISSING",
            "the named maintainer must explicitly attest the release-candidate waiver",
        )
    clock = now if now is not None else datetime.now(timezone.utc)
    _parse_authorized_at(authorized_at, now=clock)
    expected_reporting_url = _target_reporting_url(github_repository)
    normalized_authorized_by = _normalize_authorized_by(authorized_by)
    normalized_waiver_id = _normalize_waiver_id(waiver_id)
    normalized_git_source_commit = _normalize_git_source_commit(git_source_commit)
    _validate_output(output)
    security_payload = _read_bound_file(security, kind="security")
    _validate_security(security_payload, expected_reporting_url=expected_reporting_url)
    waiver_payload = _read_bound_file(waiver, kind="waiver")
    required_lines = _waiver_evidence_lines(
        package_version=package_version,
        github_repository=github_repository,
        hugging_face_repository=hugging_face_repository,
        authorized_by=normalized_authorized_by,
        authorized_at=authorized_at,
        waiver_id=normalized_waiver_id,
    )
    _validate_waiver(waiver_payload, required_lines=required_lines)
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": {
            "package_version": package_version,
            "github_repository": github_repository,
            "hugging_face_repository": hugging_face_repository,
            "git_source_commit": normalized_git_source_commit,
        },
        "security_channel_waiver": {
            "provider": PROVIDER,
            "enabled": True,
            "independent_test_completed": False,
            "decision": DECISION,
            "evidence_basis": EVIDENCE_BASIS,
            "authorized_by": normalized_authorized_by,
            "authorized_at": authorized_at,
            "waiver_id": normalized_waiver_id,
        },
        "security_file_sha256": security_payload.file_sha256,
        "waiver_file_sha256": waiver_payload.file_sha256,
    }
    document = {**unsigned, "receipt_sha256": canonical_json_hash(unsigned)}
    _validate_document(document)
    serialized = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return ReceiptPlan(
        output=output,
        document=document,
        serialized=serialized,
        security_file_sha256=security_payload.file_sha256,
        waiver_file_sha256=waiver_payload.file_sha256,
    )


def _open_write_flags() -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _remove_owned_output(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = os.lstat(path)
    except OSError:
        return
    if (metadata.st_dev, metadata.st_ino) == identity and stat.S_ISREG(metadata.st_mode):
        try:
            os.unlink(path)
        except OSError:
            pass


def write_receipt(plan: ReceiptPlan) -> None:
    """Write a prepared receipt exactly once and freeze it as mode 0444."""

    try:
        descriptor = os.open(plan.output, _open_write_flags(), 0o444)
    except FileExistsError as exc:
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_ALREADY_EXISTS", "receipt output is no-clobber"
        ) from exc
    except OSError as exc:
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_CREATE_FAILED", "receipt output cannot be created safely"
        ) from exc
    metadata = os.fstat(descriptor)
    identity = (metadata.st_dev, metadata.st_ino)
    try:
        view = memoryview(plan.serialized)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    except Exception as exc:
        os.close(descriptor)
        _remove_owned_output(plan.output, identity)
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_WRITE_FAILED", "receipt output could not be completed"
        ) from exc
    os.close(descriptor)
    observed = os.lstat(plan.output)
    if (observed.st_dev, observed.st_ino) != identity:
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_IDENTITY_CHANGED", "receipt output identity changed after writing"
        )
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o444
        or observed.st_size != len(plan.serialized)
    ):
        _remove_owned_output(plan.output, identity)
        raise SecurityChannelWaiverReceiptError(
            "OUTPUT_FINALIZATION_FAILED", "receipt output mode or size is invalid"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline receipt for a maintainer-waived private-channel RC gate. "
            "The receipt explicitly records that the independent test is incomplete."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--security", required=True, type=Path)
    parser.add_argument("--waiver", required=True, type=Path)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--hugging-face-repository", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-at", required=True)
    parser.add_argument("--waiver-id", required=True)
    parser.add_argument("--git-source-commit", required=True)
    parser.add_argument(
        "--attest-independent-test-not-completed", required=True, action="store_true"
    )
    parser.add_argument("--attest-maintainer-waiver", required=True, action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "dry-run" if args.dry_run else "build"
    try:
        plan = prepare_receipt(
            output=args.output,
            security=args.security,
            waiver=args.waiver,
            package_version=args.package_version,
            github_repository=args.github_repository,
            hugging_face_repository=args.hugging_face_repository,
            authorized_by=args.authorized_by,
            authorized_at=args.authorized_at,
            waiver_id=args.waiver_id,
            git_source_commit=args.git_source_commit,
            independent_test_not_completed_attested=(
                args.attest_independent_test_not_completed
            ),
            maintainer_waiver_attested=args.attest_maintainer_waiver,
        )
        if args.dry_run:
            status = "READY"
            receipt_written = False
        else:
            write_receipt(plan)
            status = "CREATED"
            receipt_written = True
        result = {
            "status": status,
            "mode": mode,
            "receipt_written": receipt_written,
            "output": str(plan.output),
            "receipt_sha256": plan.document["receipt_sha256"],
            "security_file_sha256": plan.security_file_sha256,
            "waiver_file_sha256": plan.waiver_file_sha256,
            "git_source_commit": plan.document["target"]["git_source_commit"],
            "provider": PROVIDER,
            "enabled": True,
            "independent_test_completed": False,
            "decision": DECISION,
            "evidence_basis": EVIDENCE_BASIS,
            "attestation_source": "explicit_human_cli_attestations_and_bound_waiver_file",
            "advisory_content_read": False,
            "remote_state_queried": False,
            "remote_state_verified_by_generator": False,
            "remote_write_performed": False,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except SecurityChannelWaiverReceiptError as exc:
        result = {
            "status": "BLOCKED",
            "mode": mode,
            "blocker_ids": [exc.blocker_id],
            "message": str(exc),
            "receipt_written": False,
            "independent_test_completed": False,
            "advisory_content_read": False,
            "remote_state_queried": False,
            "remote_state_verified_by_generator": False,
            "remote_write_performed": False,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
