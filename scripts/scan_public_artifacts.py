#!/usr/bin/env python3
"""Fail-closed secret and likely-real-PII scan for public release artifacts.

The scanner never prints a matched value.  Synthetic PII canaries may be
allowlisted by SHA-256 in a reviewed JSON file; secrets are never allowlisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_TEXT_BYTES = 64 * 1024 * 1024
SKIPPED_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
SKIPPED_BINARY_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".gif", ".jpeg", ".jpg", ".onnx", ".png", ".pt", ".pth", ".safetensors"}
)

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})\b"),
    ),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,255}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{20,})")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9._~+/=-]{12,})"
        ),
    ),
)

PHONE_PATTERN = re.compile(r"(?<![0-9A-Za-z])(1[3-9]\d{9})(?![0-9A-Za-z])")
RESIDENT_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])(\d{17}[0-9Xx])(?![0-9A-Za-z])")
LONG_DIGIT_PATTERN = re.compile(r"(?<![0-9A-Za-z])(\d{16,19})(?![0-9A-Za-z])")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
PLACEHOLDER_VALUES = frozenset({"changeme", "example", "placeholder", "redacted", "replace_me"})
RESERVED_EMAIL_SUFFIXES = (".example", ".invalid", ".localhost", ".test")
RESERVED_EMAIL_DOMAINS = frozenset({"example.com", "example.net", "example.org"})


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    fingerprint: str


def value_fingerprint(value: str) -> str:
    """Return a non-reversible identifier suitable for logs and allowlists."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_luhn_valid(value: str) -> bool:
    total = 0
    parity = len(value) % 2
    for index, character in enumerate(value):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_placeholder_secret(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered in PLACEHOLDER_VALUES
        or "example" in lowered
        or "placeholder" in lowered
        or "redact" in lowered
        or set(value) == {"x"}
    )


def load_canary_allowlist(path: Path | None) -> frozenset[str]:
    """Load reviewed synthetic-canary hashes.

    The file contains hashes rather than the canary values, so it is safe to
    retain as CI metadata.  Each entry must explicitly declare ``synthetic``
    and a non-empty purpose.
    """

    if path is None:
        return frozenset()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("canaries"), list):
        raise ValueError("canary allowlist must use schema_version 1 and contain a canaries list")
    allowed: set[str] = set()
    for index, entry in enumerate(document["canaries"]):
        if not isinstance(entry, dict):
            raise ValueError(f"canaries[{index}] must be an object")
        fingerprint = entry.get("sha256")
        purpose = entry.get("purpose")
        if (
            entry.get("synthetic") is not True
            or not isinstance(purpose, str)
            or not purpose.strip()
        ):
            raise ValueError(f"canaries[{index}] must explicitly be synthetic and state a purpose")
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", fingerprint
        ):
            raise ValueError(f"canaries[{index}].sha256 must be a lowercase SHA-256 fingerprint")
        allowed.add(fingerprint)
    return frozenset(allowed)


def _iter_files(paths: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    for supplied_path in paths:
        path = supplied_path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            yield path.parent, path
            continue
        for candidate in sorted(path.rglob("*")):
            if any(part in SKIPPED_DIRECTORIES for part in candidate.parts):
                continue
            if candidate.is_symlink():
                raise ValueError(f"public artifact scan refuses symlink: {candidate}")
            if candidate.is_file():
                yield path, candidate


def _scan_line(
    line: str,
    *,
    path: str,
    line_number: int,
    allowed_canaries: frozenset[str],
) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str, *, canary_allowed: bool) -> None:
        fingerprint = value_fingerprint(value)
        key = (kind, fingerprint)
        if key in seen or (canary_allowed and fingerprint in allowed_canaries):
            return
        seen.add(key)
        findings.append(Finding(path=path, line=line_number, kind=kind, fingerprint=fingerprint))

    for kind, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(line):
            value = match.group(1) if match.lastindex else match.group(0)
            if kind == "assigned_secret" and _is_placeholder_secret(value):
                continue
            add(kind, value, canary_allowed=False)

    for match in PHONE_PATTERN.finditer(line):
        add("potential_cn_mobile", match.group(1), canary_allowed=True)
    for match in RESIDENT_ID_PATTERN.finditer(line):
        add("potential_cn_resident_id", match.group(1), canary_allowed=True)
    for match in LONG_DIGIT_PATTERN.finditer(line):
        value = match.group(1)
        if _is_luhn_valid(value):
            add("potential_payment_card", value, canary_allowed=True)
    for match in EMAIL_PATTERN.finditer(line):
        domain = match.group(1).lower()
        if domain in RESERVED_EMAIL_DOMAINS or domain.endswith(RESERVED_EMAIL_SUFFIXES):
            continue
        add("potential_email", match.group(0), canary_allowed=True)
    return findings


def scan_paths(
    paths: Iterable[Path],
    *,
    allowed_canaries: frozenset[str] = frozenset(),
) -> list[Finding]:
    """Scan public text artifacts without returning matched content."""

    findings: list[Finding] = []
    for root, path in _iter_files(paths):
        if path.suffix.lower() in SKIPPED_BINARY_SUFFIXES:
            continue
        size = path.stat().st_size
        if size > MAX_TEXT_BYTES:
            raise ValueError(f"text artifact exceeds {MAX_TEXT_BYTES} bytes: {path}")
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ValueError(f"unexpected binary public artifact: {path}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"public text artifact is not UTF-8: {path}") from exc
        display_path = path.relative_to(root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(
                _scan_line(
                    line,
                    path=display_path,
                    line_number=line_number,
                    allowed_canaries=allowed_canaries,
                )
            )
    return findings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="files/directories to scan")
    parser.add_argument("--canary-allowlist", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        allowed = load_canary_allowlist(args.canary_allowlist)
        findings = scan_paths(args.paths, allowed_canaries=allowed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"artifact scan error: {exc}", file=sys.stderr)
        return 2

    report = {
        "schema_version": 1,
        "status": "blocked" if findings else "passed",
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if findings:
        print(f"BLOCKED: {len(findings)} potential secret/PII finding(s); values are redacted.")
        for finding in findings:
            print(f"- {finding.path}:{finding.line} {finding.kind} {finding.fingerprint[:19]}...")
        return 1
    print("PASS: no secrets or non-allowlisted potential real PII detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
