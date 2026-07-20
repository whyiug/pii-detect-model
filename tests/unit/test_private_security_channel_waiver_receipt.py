from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from scripts import build_private_security_channel_waiver_receipt as receipt

GITHUB_REPOSITORY = "whyiug/pii-detect-model"
HUGGING_FACE_REPOSITORY = "Forrest20231206/pii-zh-qwen3-0.6b-24class"
AUTHORIZED_BY = "whyiug"
AUTHORIZED_AT = "2026-07-20T04:00:00Z"
WAIVER_ID = "pvr-independent-e2e-rc1-waiver-20260720"
GIT_SOURCE_COMMIT = "a" * 40
NOW = datetime(2026, 7, 20, 4, 5, tzinfo=timezone.utc)


def _security(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "SECURITY.final.md"
    path.write_text(
        text
        or (
            "# Security\n\n"
            "GitHub Private Vulnerability Reporting is enabled for this repository. "
            "Its independent synthetic end-to-end check has not yet been recorded.\n\n"
            "Use https://github.com/whyiug/pii-detect-model/security/advisories/new.\n"
        ),
        encoding="utf-8",
    )
    return path


def _waiver(
    tmp_path: Path,
    *,
    package_version: str = receipt.PACKAGE_VERSION,
    github_repository: str = GITHUB_REPOSITORY,
    hugging_face_repository: str = HUGGING_FACE_REPOSITORY,
    authorized_by: str = AUTHORIZED_BY,
    authorized_at: str = AUTHORIZED_AT,
    waiver_id: str = WAIVER_ID,
    omit: str | None = None,
) -> Path:
    values = [
        "# Private security channel RC waiver",
        f"Waiver-ID: {waiver_id}",
        f"Authorized-by: {authorized_by}",
        f"Authorized-at: {authorized_at}",
        f"Package-version: {package_version}",
        f"GitHub-repository: {github_repository}",
        f"Hugging-Face-repository: {hugging_face_repository}",
        f"Provider: {receipt.PROVIDER}",
        "Enabled: true",
        "Independent-test-completed: false",
        f"Decision: {receipt.DECISION}",
        f"Evidence-basis: {receipt.EVIDENCE_BASIS}",
        "",
        "This waiver retains the incomplete-test state and does not create tested evidence.",
    ]
    path = tmp_path / "private-security-channel-waiver.md"
    path.write_text("\n".join(line for line in values if line != omit) + "\n", encoding="utf-8")
    return path


def _prepare(
    tmp_path: Path,
    *,
    security: Path | None = None,
    waiver: Path | None = None,
    output: Path | None = None,
    package_version: str = receipt.PACKAGE_VERSION,
    github_repository: str = GITHUB_REPOSITORY,
    hugging_face_repository: str = HUGGING_FACE_REPOSITORY,
    authorized_by: str = AUTHORIZED_BY,
    authorized_at: str = AUTHORIZED_AT,
    waiver_id: str = WAIVER_ID,
    git_source_commit: str = GIT_SOURCE_COMMIT,
    incomplete: bool = True,
    waived: bool = True,
    now: datetime = NOW,
) -> receipt.ReceiptPlan:
    return receipt.prepare_receipt(
        output=output or tmp_path / "private-security-channel-waiver-receipt.json",
        security=security or _security(tmp_path),
        waiver=waiver
        or _waiver(
            tmp_path,
            package_version=package_version,
            github_repository=github_repository,
            hugging_face_repository=hugging_face_repository,
            authorized_by=authorized_by.strip(),
            authorized_at=authorized_at,
            waiver_id=waiver_id.strip(),
        ),
        package_version=package_version,
        github_repository=github_repository,
        hugging_face_repository=hugging_face_repository,
        authorized_by=authorized_by,
        authorized_at=authorized_at,
        waiver_id=waiver_id,
        git_source_commit=git_source_commit,
        independent_test_not_completed_attested=incomplete,
        maintainer_waiver_attested=waived,
        now=now,
    )


def _schema() -> dict[str, Any]:
    return json.loads(
        (receipt.REPOSITORY_ROOT / receipt.SCHEMA_PATH).read_text(encoding="utf-8")
    )


def _cli_args(
    tmp_path: Path,
    *,
    output: Path,
    security: Path,
    waiver: Path,
    authorized_at: str | None = None,
) -> list[str]:
    return [
        "--output",
        str(output),
        "--security",
        str(security),
        "--waiver",
        str(waiver),
        "--package-version",
        receipt.PACKAGE_VERSION,
        "--github-repository",
        GITHUB_REPOSITORY,
        "--hugging-face-repository",
        HUGGING_FACE_REPOSITORY,
        "--authorized-by",
        AUTHORIZED_BY,
        "--authorized-at",
        authorized_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "--waiver-id",
        WAIVER_ID,
        "--git-source-commit",
        GIT_SOURCE_COMMIT,
        "--attest-independent-test-not-completed",
        "--attest-maintainer-waiver",
    ]


def test_schema_is_closed_and_valid_draft_2020_12() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["security_channel_waiver"]["additionalProperties"] is False


def test_builds_honest_self_hashed_read_only_waiver_receipt(tmp_path: Path) -> None:
    security = _security(tmp_path)
    waiver = _waiver(tmp_path)
    plan = _prepare(tmp_path, security=security, waiver=waiver)
    receipt.write_receipt(plan)

    document = json.loads(plan.output.read_text(encoding="utf-8"))
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(document)
    assert document["target"] == {
        "package_version": "0.2.0rc1",
        "github_repository": GITHUB_REPOSITORY,
        "hugging_face_repository": HUGGING_FACE_REPOSITORY,
        "git_source_commit": GIT_SOURCE_COMMIT,
    }
    assert document["security_channel_waiver"] == {
        "provider": "github_private_vulnerability_reporting",
        "enabled": True,
        "independent_test_completed": False,
        "decision": "maintainer_waived_for_release_candidate",
        "evidence_basis": "explicit_human_maintainer_attestation_and_bound_waiver_file",
        "authorized_by": AUTHORIZED_BY,
        "authorized_at": AUTHORIZED_AT,
        "waiver_id": WAIVER_ID,
    }
    assert document["security_file_sha256"] == hashlib.sha256(security.read_bytes()).hexdigest()
    assert document["waiver_file_sha256"] == hashlib.sha256(waiver.read_bytes()).hexdigest()
    assert document["receipt_sha256"] == receipt.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    assert stat.S_IMODE(plan.output.stat().st_mode) == 0o444
    serialized = plan.output.read_text(encoding="utf-8")
    assert "advisories/new" not in serialized
    assert '"independent_test_completed": true' not in serialized


def test_dry_run_is_offline_and_does_not_create_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "receipt.json"
    authorized_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    waiver = _waiver(tmp_path, authorized_at=authorized_at)
    args = _cli_args(
        tmp_path,
        output=output,
        security=_security(tmp_path),
        waiver=waiver,
        authorized_at=authorized_at,
    )
    result = receipt.main([*args, "--dry-run"])

    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["status"] == "READY"
    assert report["receipt_written"] is False
    assert report["independent_test_completed"] is False
    assert report["advisory_content_read"] is False
    assert report["remote_state_queried"] is False
    assert report["remote_state_verified_by_generator"] is False
    assert report["remote_write_performed"] is False
    assert "PASS" not in json.dumps(report, sort_keys=True)
    assert not output.exists()


@pytest.mark.parametrize(
    ("incomplete", "waived", "blocker_id"),
    [
        (False, True, "INDEPENDENT_TEST_INCOMPLETE_ATTESTATION_MISSING"),
        (True, False, "MAINTAINER_WAIVER_ATTESTATION_MISSING"),
    ],
)
def test_both_explicit_human_attestations_are_required(
    tmp_path: Path, incomplete: bool, waived: bool, blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, incomplete=incomplete, waived=waived)
    assert captured.value.blocker_id == blocker_id


@pytest.mark.parametrize(
    ("authorized_at", "blocker_id"),
    [
        ("2026-07-20 04:00:00Z", "AUTHORIZED_AT_NOT_RFC3339"),
        ("2026-07-20T04:00:00", "AUTHORIZED_AT_NOT_RFC3339"),
        ("2026-07-20T25:00:00Z", "AUTHORIZED_AT_NOT_RFC3339"),
        ("2026-07-20T04:05:00.000001Z", "AUTHORIZED_AT_IN_FUTURE"),
    ],
)
def test_authorized_at_is_valid_rfc3339_and_not_future(
    tmp_path: Path, authorized_at: str, blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, authorized_at=authorized_at)
    assert captured.value.blocker_id == blocker_id


def test_future_check_uses_timezone_equivalence(tmp_path: Path) -> None:
    equivalent = (NOW - timedelta(minutes=1)).astimezone(timezone(timedelta(hours=8)))
    plan = _prepare(tmp_path, authorized_at=equivalent.isoformat(), now=NOW)
    assert plan.document["security_channel_waiver"]["authorized_at"] == equivalent.isoformat()


@pytest.mark.parametrize(
    ("kwargs", "blocker_id"),
    [
        ({"package_version": "0.2.0"}, "RECEIPT_SCHEMA_REJECTED"),
        ({"github_repository": "not-a-repository"}, "INVALID_GITHUB_REPOSITORY"),
        ({"authorized_by": "bad\nidentity"}, "AUTHORIZED_BY_INVALID"),
        ({"waiver_id": "bad waiver id"}, "WAIVER_ID_INVALID"),
        ({"git_source_commit": "A" * 40}, "INVALID_GIT_SOURCE_COMMIT"),
    ],
)
def test_target_and_human_identity_are_strict(
    tmp_path: Path, kwargs: dict[str, str], blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, **kwargs)
    assert captured.value.blocker_id == blocker_id


@pytest.mark.parametrize(
    ("text", "blocker_id"),
    [
        (
            "# Security\n\nIndependent test not completed.\n",
            "SECURITY_REPORTING_URL_MISSING",
        ),
        (
            "# Security\n\nUse https://github.com/whyiug/pii-detect-model/security/"
            "advisories/new. Independent test not completed.\n",
            "SECURITY_CHANNEL_NOT_ENABLED",
        ),
        (
            "# Security\n\nGitHub Private Vulnerability Reporting is enabled. Use "
            "https://github.com/whyiug/pii-detect-model/security/advisories/new.\n",
            "SECURITY_INCOMPLETE_TEST_STATUS_MISSING",
        ),
    ],
)
def test_security_must_support_only_truthful_waiver_state(
    tmp_path: Path, text: str, blocker_id: str
) -> None:
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, security=_security(tmp_path, text))
    assert captured.value.blocker_id == blocker_id


def test_waiver_requires_all_exact_human_evidence_lines(tmp_path: Path) -> None:
    missing_line = "Independent-test-completed: false"
    waiver = _waiver(tmp_path, omit=missing_line)
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, waiver=waiver)
    assert captured.value.blocker_id == "WAIVER_EVIDENCE_INCOMPLETE"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Enabled: true", "enabled: false"),
        ("Enabled: true", "Enabled : false"),
        ("Enabled: true", "> Enabled: false"),
        ("Enabled: true", "+ Enabled: false"),
        ("Enabled: true", "1. Enabled: false"),
        ("Enabled: true", "- > Enabled: false"),
        ("Enabled: true", "**Enabled**: false"),
        ("Enabled: true", "`Enabled`: false"),
        (
            "Independent-test-completed: false",
            "Independent-test-completed : true",
        ),
    ],
)
def test_waiver_rejects_case_spacing_and_markdown_aliases(
    tmp_path: Path, old: str, new: str
) -> None:
    waiver = _waiver(tmp_path)
    waiver.write_text(
        waiver.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, waiver=waiver)
    assert captured.value.blocker_id == "WAIVER_EVIDENCE_INCOMPLETE"


@pytest.mark.parametrize(
    "claim",
    [
        "The independent test: passed.",
        "The independent end-to-end test has successfully passed.",
        "The independent PVR test passed.",
        "The independent private-report-channel test completed.",
        "The independent test result was PASS.",
    ],
)
def test_waiver_rejects_positive_completion_prose(
    tmp_path: Path, claim: str
) -> None:
    waiver = _waiver(tmp_path)
    waiver.write_text(
        waiver.read_text(encoding="utf-8") + claim + "\n",
        encoding="utf-8",
    )
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, waiver=waiver)
    assert captured.value.blocker_id == "WAIVER_CONTRADICTORY_TEST_STATUS"


@pytest.mark.parametrize(
    "claim",
    [
        "The independent test: passed.",
        "The independent end-to-end test has successfully passed.",
        "The independent PVR test passed.",
        "The independent private-report-channel test completed.",
        "The independent test result was PASS.",
        "GitHub Private Vulnerability Reporting is disabled.",
        "It is false that GitHub Private Vulnerability Reporting is enabled.",
    ],
)
def test_security_rejects_contradictory_pass_claim(
    tmp_path: Path, claim: str
) -> None:
    text = (
        _security(tmp_path).read_text(encoding="utf-8")
        + claim
        + "\n"
    )
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, security=_security(tmp_path, text))
    assert captured.value.blocker_id == "SECURITY_CONTRADICTORY_TEST_STATUS"


@pytest.mark.parametrize(
    "extra_line",
    [
        "Independent-test-completed: true",
        "Outcome: accepted_private_test_report",
        "Tested: true",
    ],
)
def test_waiver_rejects_duplicate_or_contradictory_test_claims(
    tmp_path: Path, extra_line: str
) -> None:
    waiver = _waiver(tmp_path)
    waiver.write_text(
        waiver.read_text(encoding="utf-8") + extra_line + "\n", encoding="utf-8"
    )
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, waiver=waiver)
    assert captured.value.blocker_id in {
        "WAIVER_EVIDENCE_INCOMPLETE",
        "WAIVER_CONTRADICTORY_TEST_STATUS",
    }


@pytest.mark.parametrize(
    ("kind", "blocker_id"),
    [
        ("security", "SECURITY_SYMLINK_REJECTED"),
        ("waiver", "WAIVER_SYMLINK_REJECTED"),
    ],
)
def test_bound_input_symlinks_are_rejected(
    tmp_path: Path, kind: str, blocker_id: str
) -> None:
    real = _security(tmp_path) if kind == "security" else _waiver(tmp_path)
    linked = tmp_path / f"{kind}.link.md"
    linked.symlink_to(real)
    kwargs = {kind: linked}
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, **kwargs)
    assert captured.value.blocker_id == blocker_id


def test_output_is_no_clobber_for_build_and_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("caller-owned\n", encoding="utf-8")
    with pytest.raises(receipt.SecurityChannelWaiverReceiptError) as captured:
        _prepare(tmp_path, output=output)
    assert captured.value.blocker_id == "OUTPUT_ALREADY_EXISTS"

    authorized_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    waiver = _waiver(tmp_path, authorized_at=authorized_at)
    result = receipt.main(
        [
            *_cli_args(
                tmp_path,
                output=output,
                security=_security(tmp_path),
                waiver=waiver,
                authorized_at=authorized_at,
            ),
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 2
    assert report["blocker_ids"] == ["OUTPUT_ALREADY_EXISTS"]
    assert report["receipt_written"] is False
    assert output.read_text(encoding="utf-8") == "caller-owned\n"


def test_schema_rejects_pass_claims_and_additional_fields(tmp_path: Path) -> None:
    plan = _prepare(tmp_path)
    top_level = dict(plan.document)
    top_level["test_passed"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(top_level)

    nested = json.loads(json.dumps(plan.document))
    nested["security_channel_waiver"]["independent_test_completed"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(nested)

    nested = json.loads(json.dumps(plan.document))
    nested["security_channel_waiver"]["remote_verified"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(nested)


def test_cli_exposes_no_advisory_body_or_remote_pass_input() -> None:
    destinations = {action.dest for action in receipt._parser()._actions}
    assert "report_body" not in destinations
    assert "advisory" not in destinations
    assert "remote_pass" not in destinations
    assert "tested" not in destinations
