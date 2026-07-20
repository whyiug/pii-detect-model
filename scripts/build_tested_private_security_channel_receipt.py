#!/usr/bin/env python3
"""Build a closed receipt for a human-tested private security channel.

This offline utility never contacts GitHub and has no input for an advisory
title, body, URL, reporter, or other report content.  It records an explicit
human attestation that a synthetic private report was accepted, binds that
attestation to the exact publication target and final ``SECURITY.md`` bytes,
and emits a canonical self-hash.  It cannot establish that the remote test
happened; callers must not invoke the write mode before completing the test.
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
SCHEMA_PATH = Path("configs/release/tested_private_security_channel_receipt.schema.json")
SCHEMA_VERSION = "pii-zh.community-v2-tested-private-security-channel-receipt.v1"
PACKAGE_VERSION = "0.2.0rc1"
PROVIDER = "github_private_vulnerability_reporting"
OUTCOME = "accepted_private_test_report"
EVIDENCE_BASIS = "human_attestation_not_remote_verified"

_MAX_SECURITY_BYTES = 8 * 1024 * 1024
_REPOSITORY_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)
_GITHUB_LOGIN = re.compile(r"^(?=.{1,39}$)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
_GITHUB_URL = re.compile(r"https://github\.com/[^\s<>()\[\]{}\"']+")
_RFC3339 = re.compile(
    r"^(?:[0-9]{4})-(?:[0-9]{2})-(?:[0-9]{2})T"
    r"(?:[0-9]{2}):(?:[0-9]{2}):(?:[0-9]{2})"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_STALE_SECURITY_MARKERS = (
    "unpublished",
    "pending",
    "withheld",
    "no tested private reporting route exists",
    "no-tested-route",
)


class SecurityChannelReceiptError(RuntimeError):
    """Raised when a receipt cannot be prepared or written safely."""

    def __init__(self, blocker_id: str, message: str) -> None:
        super().__init__(message)
        self.blocker_id = blocker_id


@dataclass(frozen=True)
class SecurityPayload:
    payload: bytes
    file_sha256: str


@dataclass(frozen=True)
class ReceiptPlan:
    output: Path
    document: Mapping[str, Any]
    serialized: bytes
    security_file_sha256: str


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
        raise SecurityChannelReceiptError(
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


def _read_security(path: Path, *, expected_reporting_url: str) -> SecurityPayload:
    try:
        path_metadata = os.lstat(path)
    except OSError as exc:
        raise SecurityChannelReceiptError(
            "SECURITY_READ_FAILED", "final SECURITY file cannot be inspected safely"
        ) from exc
    if stat.S_ISLNK(path_metadata.st_mode):
        raise SecurityChannelReceiptError(
            "SECURITY_SYMLINK_REJECTED", "final SECURITY input cannot be a symlink"
        )
    if not stat.S_ISREG(path_metadata.st_mode):
        raise SecurityChannelReceiptError(
            "SECURITY_NOT_REGULAR", "final SECURITY input must be a regular file"
        )
    try:
        descriptor = os.open(path, _open_read_flags())
    except OSError as exc:
        blocker = (
            "SECURITY_SYMLINK_REJECTED" if exc.errno == errno.ELOOP else "SECURITY_READ_FAILED"
        )
        raise SecurityChannelReceiptError(
            blocker, "final SECURITY file cannot be opened safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SecurityChannelReceiptError(
                "SECURITY_NOT_REGULAR", "final SECURITY input must be a regular file"
            )
        if metadata.st_size > _MAX_SECURITY_BYTES:
            raise SecurityChannelReceiptError(
                "SECURITY_TOO_LARGE", "final SECURITY input exceeds 8 MiB"
            )
        if (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino):
            raise SecurityChannelReceiptError(
                "SECURITY_CHANGED_DURING_OPEN", "final SECURITY input changed while opening"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_SECURITY_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > _MAX_SECURITY_BYTES:
                raise SecurityChannelReceiptError(
                    "SECURITY_TOO_LARGE", "final SECURITY input exceeds 8 MiB"
                )
        payload = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(metadata, field) != getattr(final_metadata, field) for field in stable_fields
        ) or len(payload) != final_metadata.st_size:
            raise SecurityChannelReceiptError(
                "SECURITY_CHANGED_DURING_READ", "final SECURITY input changed while reading"
            )
    finally:
        os.close(descriptor)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecurityChannelReceiptError(
            "SECURITY_NOT_UTF8", "final SECURITY input must be UTF-8"
        ) from exc
    if not text.strip() or "\x00" in text:
        raise SecurityChannelReceiptError(
            "SECURITY_INVALID_TEXT", "final SECURITY input is empty or contains NUL"
        )
    lowered = text.lower()
    stale = next((marker for marker in _STALE_SECURITY_MARKERS if marker in lowered), None)
    if stale is not None:
        raise SecurityChannelReceiptError(
            "SECURITY_NOT_FINAL",
            f"final SECURITY input retains blocked publication marker {stale!r}",
        )
    observed_urls = {
        match.group(0).rstrip(".,;:!?") for match in _GITHUB_URL.finditer(text)
    }
    if expected_reporting_url not in observed_urls:
        raise SecurityChannelReceiptError(
            "SECURITY_REPORTING_URL_MISSING",
            "final SECURITY input does not contain the exact target private-report URL",
        )
    return SecurityPayload(payload=payload, file_sha256=hashlib.sha256(payload).hexdigest())


def _target_reporting_url(github_repository: str) -> str:
    if _REPOSITORY_ID.fullmatch(github_repository) is None:
        raise SecurityChannelReceiptError(
            "INVALID_GITHUB_REPOSITORY", "GitHub repository is not a valid owner/repository id"
        )
    return f"https://github.com/{github_repository}/security/advisories/new"


def _normalize_tested_by(tested_by: str) -> str:
    normalized = tested_by.strip()
    if _GITHUB_LOGIN.fullmatch(normalized) is None:
        raise SecurityChannelReceiptError(
            "TESTED_BY_INVALID", "tested_by must be a stable public GitHub login"
        )
    return normalized


def _parse_tested_at(value: str, *, now: datetime) -> datetime:
    if _RFC3339.fullmatch(value) is None:
        raise SecurityChannelReceiptError(
            "TESTED_AT_NOT_RFC3339", "tested_at must be an RFC3339 timestamp with a timezone"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise SecurityChannelReceiptError(
            "TESTED_AT_NOT_RFC3339", "tested_at is not a valid RFC3339 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SecurityChannelReceiptError(
            "TESTED_AT_NOT_RFC3339", "tested_at must include a timezone"
        )
    if now.tzinfo is None or now.utcoffset() is None:
        raise SecurityChannelReceiptError(
            "INVALID_CLOCK", "comparison clock must be timezone-aware"
        )
    if parsed.astimezone(timezone.utc) > now.astimezone(timezone.utc):
        raise SecurityChannelReceiptError(
            "TESTED_AT_IN_FUTURE", "tested_at cannot be later than the current time"
        )
    return parsed


def _schema() -> Mapping[str, Any]:
    schema_file = REPOSITORY_ROOT / SCHEMA_PATH
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SecurityChannelReceiptError(
            "RECEIPT_SCHEMA_UNAVAILABLE", "receipt schema cannot be loaded"
        ) from exc
    if not isinstance(schema, dict):
        raise SecurityChannelReceiptError(
            "RECEIPT_SCHEMA_INVALID", "receipt schema must be a JSON object"
        )
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise SecurityChannelReceiptError(
            "RECEIPT_SCHEMA_INVALID", "receipt schema is not valid Draft 2020-12"
        ) from exc
    return schema


def _validate_document(document: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(document)
    except SecurityChannelReceiptError:
        raise
    except Exception as exc:
        raise SecurityChannelReceiptError(
            "RECEIPT_SCHEMA_REJECTED", "receipt does not satisfy its closed schema"
        ) from exc


def _validate_output(output: Path) -> None:
    try:
        parent = os.lstat(output.parent)
    except OSError as exc:
        raise SecurityChannelReceiptError(
            "OUTPUT_PARENT_UNAVAILABLE", "receipt output parent does not exist"
        ) from exc
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise SecurityChannelReceiptError(
            "OUTPUT_PARENT_UNSAFE", "receipt output parent must be a real directory"
        )
    try:
        os.lstat(output)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SecurityChannelReceiptError(
            "OUTPUT_INSPECTION_FAILED", "receipt output cannot be inspected"
        ) from exc
    raise SecurityChannelReceiptError("OUTPUT_ALREADY_EXISTS", "receipt output is no-clobber")


def prepare_receipt(
    *,
    output: Path,
    security: Path,
    package_version: str,
    github_repository: str,
    hugging_face_repository: str,
    tested_by: str,
    tested_at: str,
    test_case_id: str,
    outcome: str,
    accepted_private_report_attested: bool,
    no_real_sensitive_data_attested: bool,
    now: datetime | None = None,
) -> ReceiptPlan:
    """Validate inputs and prepare, but do not write, a receipt."""

    if not accepted_private_report_attested:
        raise SecurityChannelReceiptError(
            "ACCEPTED_REPORT_ATTESTATION_MISSING",
            "a human must attest that the synthetic private report was accepted",
        )
    if not no_real_sensitive_data_attested:
        raise SecurityChannelReceiptError(
            "NO_SENSITIVE_DATA_ATTESTATION_MISSING",
            "a human must attest that the channel test contained no real sensitive data",
        )
    if outcome != OUTCOME:
        raise SecurityChannelReceiptError(
            "INVALID_CHANNEL_OUTCOME", f"outcome must be exactly {OUTCOME!r}"
        )
    clock = now if now is not None else datetime.now(timezone.utc)
    _parse_tested_at(tested_at, now=clock)
    expected_reporting_url = _target_reporting_url(github_repository)
    normalized_tested_by = _normalize_tested_by(tested_by)
    _validate_output(output)
    security_payload = _read_security(
        security, expected_reporting_url=expected_reporting_url
    )
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": {
            "package_version": package_version,
            "github_repository": github_repository,
            "hugging_face_repository": hugging_face_repository,
        },
        "channel_test": {
            "tested": True,
            "tested_by": normalized_tested_by,
            "tested_at": tested_at,
            "provider": PROVIDER,
            "evidence_basis": EVIDENCE_BASIS,
            "test_case_id": test_case_id,
            "outcome": OUTCOME,
            "contains_real_sensitive_data": False,
        },
        "security_file_sha256": security_payload.file_sha256,
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
        raise SecurityChannelReceiptError(
            "OUTPUT_ALREADY_EXISTS", "receipt output is no-clobber"
        ) from exc
    except OSError as exc:
        raise SecurityChannelReceiptError(
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
        if isinstance(exc, SecurityChannelReceiptError):
            raise
        raise SecurityChannelReceiptError(
            "OUTPUT_WRITE_FAILED", "receipt output could not be completed"
        ) from exc
    os.close(descriptor)
    observed = os.lstat(plan.output)
    if (observed.st_dev, observed.st_ino) != identity:
        raise SecurityChannelReceiptError(
            "OUTPUT_IDENTITY_CHANGED", "receipt output identity changed after writing"
        )
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o444
        or observed.st_size != len(plan.serialized)
    ):
        _remove_owned_output(plan.output, identity)
        raise SecurityChannelReceiptError(
            "OUTPUT_FINALIZATION_FAILED", "receipt output mode or size is invalid"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline, human-attested receipt for an accepted synthetic GitHub private "
            "vulnerability report. This command never reads advisory content or verifies "
            "remote state."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--security", required=True, type=Path)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--hugging-face-repository", required=True)
    parser.add_argument("--tested-by", required=True)
    parser.add_argument("--tested-at", required=True)
    parser.add_argument("--test-case-id", required=True)
    parser.add_argument("--outcome", required=True, choices=(OUTCOME,))
    parser.add_argument("--attest-private-report-accepted", required=True, action="store_true")
    parser.add_argument("--attest-no-real-sensitive-data", required=True, action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "dry-run" if args.dry_run else "build"
    try:
        plan = prepare_receipt(
            output=args.output,
            security=args.security,
            package_version=args.package_version,
            github_repository=args.github_repository,
            hugging_face_repository=args.hugging_face_repository,
            tested_by=args.tested_by,
            tested_at=args.tested_at,
            test_case_id=args.test_case_id,
            outcome=args.outcome,
            accepted_private_report_attested=args.attest_private_report_accepted,
            no_real_sensitive_data_attested=args.attest_no_real_sensitive_data,
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
            "attestation_source": "explicit_human_cli_attestation",
            "advisory_content_read": False,
            "remote_state_queried": False,
            "remote_state_verified_by_generator": False,
            "remote_write_performed": False,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except SecurityChannelReceiptError as exc:
        result = {
            "status": "BLOCKED",
            "mode": mode,
            "blocker_ids": [exc.blocker_id],
            "message": str(exc),
            "receipt_written": False,
            "advisory_content_read": False,
            "remote_state_queried": False,
            "remote_state_verified_by_generator": False,
            "remote_write_performed": False,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
