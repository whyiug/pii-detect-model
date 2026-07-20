from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from scripts import build_community_v2_publication_successor as successor
from scripts import build_tested_private_security_channel_receipt as receipt

GITHUB_REPOSITORY = "whyiug/pii-detect-model"
HUGGING_FACE_REPOSITORY = "Forrest20231206/pii-zh-qwen3-0.6b-24class"
TESTED_AT = "2026-07-20T04:00:00Z"
NOW = datetime(2026, 7, 20, 4, 5, tzinfo=timezone.utc)


def _security(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "SECURITY.final.md"
    path.write_text(
        text
        or (
            "# Security\n\n"
            "Use GitHub private vulnerability reporting at "
            "https://github.com/whyiug/pii-detect-model/security/advisories/new.\n"
        ),
        encoding="utf-8",
    )
    return path


def _prepare(
    tmp_path: Path,
    *,
    security: Path | None = None,
    output: Path | None = None,
    tested_at: str = TESTED_AT,
    now: datetime = NOW,
    accepted: bool = True,
    no_sensitive: bool = True,
    package_version: str = receipt.PACKAGE_VERSION,
    github_repository: str = GITHUB_REPOSITORY,
    hugging_face_repository: str = HUGGING_FACE_REPOSITORY,
    tested_by: str = "release-maintainer",
    test_case_id: str = "synthetic-private-report-001",
    outcome: str = receipt.OUTCOME,
) -> receipt.ReceiptPlan:
    return receipt.prepare_receipt(
        output=output or tmp_path / "tested-private-security-channel-receipt.json",
        security=security or _security(tmp_path),
        package_version=package_version,
        github_repository=github_repository,
        hugging_face_repository=hugging_face_repository,
        tested_by=tested_by,
        tested_at=tested_at,
        test_case_id=test_case_id,
        outcome=outcome,
        accepted_private_report_attested=accepted,
        no_real_sensitive_data_attested=no_sensitive,
        now=now,
    )


def _schema() -> dict[str, Any]:
    return json.loads(
        (receipt.REPOSITORY_ROOT / receipt.SCHEMA_PATH).read_text(encoding="utf-8")
    )


def test_standalone_schema_matches_successor_nested_contract() -> None:
    standalone = _schema()
    successor_schema = json.loads(
        (successor.REPOSITORY_ROOT / successor.MANIFEST_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    nested = successor_schema["$defs"]["testedPrivateSecurityChannelReceipt"]

    assert {
        key: standalone[key]
        for key in ("type", "additionalProperties", "required", "properties")
    } == nested
    for definition in (
        "sha256",
        "repoId",
        "humanIdentity",
        "githubLogin",
        "publicationTarget",
    ):
        assert standalone["$defs"][definition] == successor_schema["$defs"][definition]
    Draft202012Validator.check_schema(standalone)


def test_builds_closed_self_hashed_read_only_receipt(tmp_path: Path) -> None:
    security = _security(tmp_path)
    plan = _prepare(tmp_path, security=security)
    receipt.write_receipt(plan)

    output = plan.output
    document = json.loads(output.read_text(encoding="utf-8"))
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(document)
    assert document["target"] == {
        "package_version": "0.2.0rc1",
        "github_repository": GITHUB_REPOSITORY,
        "hugging_face_repository": HUGGING_FACE_REPOSITORY,
    }
    assert document["channel_test"] == {
        "tested": True,
        "tested_by": "release-maintainer",
        "tested_at": TESTED_AT,
        "provider": "github_private_vulnerability_reporting",
        "test_case_id": "synthetic-private-report-001",
        "outcome": "accepted_private_test_report",
        "contains_real_sensitive_data": False,
        "evidence_basis": "human_attestation_not_remote_verified",
    }
    assert document["security_file_sha256"] == hashlib.sha256(security.read_bytes()).hexdigest()
    assert document["receipt_sha256"] == receipt.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    assert document["receipt_sha256"] == successor.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert "security/advisories/new" not in output.read_text(encoding="utf-8")


def test_dry_run_validates_without_creating_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    security = _security(tmp_path)
    output = tmp_path / "receipt.json"
    result = receipt.main(
        [
            "--output",
            str(output),
            "--security",
            str(security),
            "--package-version",
            receipt.PACKAGE_VERSION,
            "--github-repository",
            GITHUB_REPOSITORY,
            "--hugging-face-repository",
            HUGGING_FACE_REPOSITORY,
            "--tested-by",
            "release-maintainer",
            "--tested-at",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "--test-case-id",
            "synthetic-private-report-001",
            "--outcome",
            receipt.OUTCOME,
            "--attest-private-report-accepted",
            "--attest-no-real-sensitive-data",
            "--dry-run",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["status"] == "READY"
    assert report["receipt_written"] is False
    assert report["advisory_content_read"] is False
    assert report["remote_state_queried"] is False
    assert report["remote_state_verified_by_generator"] is False
    assert report["remote_write_performed"] is False
    assert "PASS" not in json.dumps(report, sort_keys=True)
    assert not output.exists()


@pytest.mark.parametrize(
    ("tested_at", "blocker_id"),
    [
        ("2026-07-20 04:00:00Z", "TESTED_AT_NOT_RFC3339"),
        ("2026-07-20T04:00:00", "TESTED_AT_NOT_RFC3339"),
        ("2026-07-20T25:00:00Z", "TESTED_AT_NOT_RFC3339"),
        ("2026-07-20T04:05:00.000001Z", "TESTED_AT_IN_FUTURE"),
    ],
)
def test_tested_at_must_be_valid_rfc3339_and_not_future(
    tmp_path: Path, tested_at: str, blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, tested_at=tested_at)
    assert captured.value.blocker_id == blocker_id


@pytest.mark.parametrize(
    ("accepted", "no_sensitive", "blocker_id"),
    [
        (False, True, "ACCEPTED_REPORT_ATTESTATION_MISSING"),
        (True, False, "NO_SENSITIVE_DATA_ATTESTATION_MISSING"),
    ],
)
def test_both_explicit_human_attestations_are_required(
    tmp_path: Path, accepted: bool, no_sensitive: bool, blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, accepted=accepted, no_sensitive=no_sensitive)
    assert captured.value.blocker_id == blocker_id


@pytest.mark.parametrize(
    ("kwargs", "blocker_id"),
    [
        ({"package_version": "0.2.0"}, "RECEIPT_SCHEMA_REJECTED"),
        ({"github_repository": "not-a-repository"}, "INVALID_GITHUB_REPOSITORY"),
        ({"tested_by": "bad\nidentity"}, "TESTED_BY_INVALID"),
        ({"test_case_id": "bad test id"}, "RECEIPT_SCHEMA_REJECTED"),
        ({"outcome": "caller_reported_pass"}, "INVALID_CHANNEL_OUTCOME"),
    ],
)
def test_target_identity_case_and_outcome_are_strict(
    tmp_path: Path, kwargs: dict[str, str], blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, **kwargs)
    assert captured.value.blocker_id == blocker_id


@pytest.mark.parametrize(
    "marker",
    [
        "Independent synthetic test is pending.",
        "No tested private reporting route exists.",
        "Public upload is withheld.",
    ],
)
def test_security_file_must_be_final(tmp_path: Path, marker: str) -> None:
    security = _security(tmp_path, f"# Security\n\n{marker}\n")
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, security=security)
    assert captured.value.blocker_id == "SECURITY_NOT_FINAL"


@pytest.mark.parametrize(
    "text",
    [
        "# Security\n\nThe independent synthetic channel test is complete.\n",
        (
            "# Security\n\nUse "
            "https://github.com/another-owner/pii-detect-model/security/advisories/new.\n"
        ),
        (
            "# Security\n\nUse "
            "https://github.com/whyiug/pii-detect-model/security/advisories/new.evil.\n"
        ),
    ],
)
def test_security_requires_exact_target_private_reporting_url(tmp_path: Path, text: str) -> None:
    security = _security(tmp_path, text)
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, security=security)
    assert captured.value.blocker_id == "SECURITY_REPORTING_URL_MISSING"


@pytest.mark.parametrize(
    "tested_by",
    ["", "   ", "_maintainer", "maintainer_1", "-maintainer", "maintainer-", "a--b", "a" * 40],
)
def test_tested_by_must_be_stable_github_login(tmp_path: Path, tested_by: str) -> None:
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, tested_by=tested_by)
    assert captured.value.blocker_id == "TESTED_BY_INVALID"


def test_tested_by_is_stripped_before_receipt_is_sealed(tmp_path: Path) -> None:
    plan = _prepare(tmp_path, tested_by="  release-maintainer  ")
    assert plan.document["channel_test"]["tested_by"] == "release-maintainer"


def test_security_symlink_is_rejected(tmp_path: Path) -> None:
    real_security = _security(tmp_path)
    linked_security = tmp_path / "SECURITY.link.md"
    linked_security.symlink_to(real_security)
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, security=linked_security)
    assert captured.value.blocker_id == "SECURITY_SYMLINK_REJECTED"


def test_output_is_no_clobber_for_build_and_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("caller-owned\n", encoding="utf-8")
    with pytest.raises(receipt.SecurityChannelReceiptError) as captured:
        _prepare(tmp_path, output=output)
    assert captured.value.blocker_id == "OUTPUT_ALREADY_EXISTS"
    assert output.read_text(encoding="utf-8") == "caller-owned\n"

    result = receipt.main(
        [
            "--output",
            str(output),
            "--security",
            str(_security(tmp_path)),
            "--package-version",
            receipt.PACKAGE_VERSION,
            "--github-repository",
            GITHUB_REPOSITORY,
            "--hugging-face-repository",
            HUGGING_FACE_REPOSITORY,
            "--tested-by",
            "release-maintainer",
            "--tested-at",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "--test-case-id",
            "synthetic-private-report-001",
            "--outcome",
            receipt.OUTCOME,
            "--attest-private-report-accepted",
            "--attest-no-real-sensitive-data",
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 2
    assert report["blocker_ids"] == ["OUTPUT_ALREADY_EXISTS"]
    assert report["receipt_written"] is False
    assert output.read_text(encoding="utf-8") == "caller-owned\n"


def test_future_check_uses_timezone_equivalence(tmp_path: Path) -> None:
    equivalent = (NOW - timedelta(minutes=1)).astimezone(timezone(timedelta(hours=8)))
    plan = _prepare(tmp_path, tested_at=equivalent.isoformat(), now=NOW)
    assert plan.document["channel_test"]["tested_at"] == equivalent.isoformat()


def test_schema_rejects_additional_receipt_or_channel_fields(tmp_path: Path) -> None:
    plan = _prepare(tmp_path)
    top_level = dict(plan.document)
    top_level["remote_pass"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(top_level)

    channel = json.loads(json.dumps(plan.document))
    channel["channel_test"]["report_body"] = "must never be retained"
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(channel)


@pytest.mark.parametrize("evidence_basis", [None, "remote_verified", "caller_reported_pass"])
def test_schema_requires_explicit_non_remote_evidence_basis(
    tmp_path: Path, evidence_basis: str | None
) -> None:
    plan = _prepare(tmp_path)
    document = json.loads(json.dumps(plan.document))
    if evidence_basis is None:
        document["channel_test"].pop("evidence_basis")
    else:
        document["channel_test"]["evidence_basis"] = evidence_basis
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(document)


def test_cli_exposes_no_advisory_body_or_remote_pass_input() -> None:
    destinations = {action.dest for action in receipt._parser()._actions}
    assert "report_body" not in destinations
    assert "advisory" not in destinations
    assert "remote_pass" not in destinations
