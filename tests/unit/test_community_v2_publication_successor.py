from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import build_community_v2_publication_successor as successor

GITHUB_REPOSITORY = "whyiug/pii-detect-model"
HUGGING_FACE_REPOSITORY = "maintainer/pii-zh-qwen3-0.6b-24class"
GIT_SOURCE_COMMIT = "a" * 40


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["receipt_sha256"] = ""
    result["receipt_sha256"] = successor.canonical_json_hash(result, remove="receipt_sha256")
    return result


def _write_checksums(source: Path) -> None:
    lines = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "checksums.txt":
            lines.append(f"{_sha256(path.read_bytes())}  {path.relative_to(source).as_posix()}")
    (source / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "immutable-source"
    source.mkdir()
    payloads = {
        "README.md": b"# old local candidate\nunpublished_local_candidate\n",
        "SECURITY.md": b"# old security placeholder\n",
        "LICENSE": b"Apache License fixture\n",
        "NOTICE": b"This unpublished local candidate must not be published.\n",
        "THIRD_PARTY_NOTICES.md": b"Pending human license review.\n",
        "model.safetensors": b"small-fake-safetensors-fixture",
        "config.json": json.dumps(
            {
                "architectures": ["Fixture"],
                "pii_release_eligible": False,
                "pii_lineage": {"release_eligible": False},
                "pii_training_status": "completed_candidate_not_benchmark_evaluated",
            },
            sort_keys=True,
        ).encode()
        + b"\n",
        "training_manifest.json": json.dumps(
            {
                "release_eligible": False,
                "initialization": {"release_eligible": False},
            },
            sort_keys=True,
        ).encode()
        + b"\n",
        "configuration_qwen3_bi.py": (
            b'community_release_eligible = kwargs.get("pii_release_eligible")\n'
            b"if community_release_eligible is not False:\n"
            b'    raise ValueError("pii_release_eligible=false contract.")\n'
            b"self.pii_release_eligible = community_release_eligible\n"
        ),
        "modeling_qwen3_bi.py": (b'getattr(config, "pii_release_eligible", None) is False\n'),
        "community_v2_preauthorization.json": b'{"publication":"withheld"}\n',
    }
    for relative, payload in payloads.items():
        (source / relative).write_bytes(payload)
    _write_checksums(source)
    return source


def _final_local_receipt() -> dict[str, Any]:
    sha = "1" * 64
    artifacts = {
        name: None
        for name in (
            "final_model_manifest",
            "model_card",
            "model_package_manifest",
            "service_configuration_manifest",
            "service_source_manifest",
            "technical_documentation_manifest",
            "wheel_manifest",
            "wheelhouse_manifest",
            "container_manifest",
            "sbom",
            "license_report",
            "benchmark_report",
            "public_artifact_scan",
            "dependency_scan",
        )
    }
    verifications = {
        name: None
        for name in (
            "unit_tests",
            "clean_wheel_smoke",
            "container_smoke",
            "offline_model_smoke",
            "offline_service_smoke",
        )
    }
    return _seal(
        {
            "schema_version": "pii-zh.community-cascade-release-receipt.v2",
            "report_type": "community_cascade_release_v2",
            "contract": {"file_sha256": sha, "contract_sha256": sha},
            "status": "READY_FOR_USER_AUTHORIZATION",
            "local_candidate_complete": True,
            "publication_state": "READY_FOR_USER_AUTHORIZATION",
            "blocker_ids": [],
            "checks": {"local_closure": {"status": "PASS"}},
            "quality": {
                "reported_status": "PASS",
                "receipt_sha256": sha,
                "full_system_status": "PASS",
                "all_required_replays": "PASS",
            },
            "identity": {
                "model_id": "fixture-model",
                "service_id": "fixture-service",
                "training_manifest_sha256": sha,
                "model_identity_sha256": sha,
                "calibration_bundle_file_sha256": sha,
                "service_configuration_sha256": sha,
                "service_implementation_sha256": sha,
                "source_manifest_sha256": sha,
                "quality_receipt_sha256": sha,
                "internal_unlock_sha256": sha,
            },
            "replay_evidence": {"model_raw": {}, "full_system": {}},
            "release_artifacts": artifacts,
            "verification_receipts": verifications,
            "claims": {
                "first_chinese_pii_model": False,
                "global_sota": False,
                "production_ready": False,
                "real_world_sota": False,
                "named_open24_leadership_requires_all_replays": True,
            },
            "limitations": [
                "public_and_synthetic_only",
                "public_test_exposed",
                "not_production_ready",
                "no_real_world_sota_claim",
            ],
            "privacy": {
                "contains_paths": False,
                "contains_raw_records": False,
                "model_weights_read": False,
                "gpu_queried_or_used": False,
                "network_used": False,
            },
        }
    )


def _fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    source = _make_source(tmp_path)
    model_card = tmp_path / "publication-model-card.md"
    model_card.write_text(
        "# Community v2\n\n"
        "Version 0.2.0rc1 publication candidate.\n\n"
        f"github_repository: {GITHUB_REPOSITORY}\n"
        f"hugging_face_repository: {HUGGING_FACE_REPOSITORY}\n"
        f"git_source_commit: {GIT_SOURCE_COMMIT}\n"
        f"release_tag: {successor.RELEASE_TAG}\n"
        "github_release_url: "
        f"https://github.com/{GITHUB_REPOSITORY}/releases/tag/{successor.RELEASE_TAG}\n",
        encoding="utf-8",
    )
    security = tmp_path / "SECURITY.current.md"
    security.write_text(
        "# Security\n\nUse the tested private vulnerability reporting channel at "
        f"https://github.com/{GITHUB_REPOSITORY}/security/advisories/new.\n",
        encoding="utf-8",
    )
    notice = tmp_path / "NOTICE.publication"
    notice.write_text("pii-zh-qwen 0.2.0rc1 community publication notice.\n", encoding="utf-8")
    third_party_notices = tmp_path / "THIRD_PARTY_NOTICES.publication.md"
    third_party_notices.write_text(
        "# Third-Party Notices for 0.2.0rc1\n\n"
        "The immutable license report retains status `COMPLETE_HUMAN_APPROVAL_PENDING` "
        "as historical mechanical evidence. Human clearance comes from the separately "
        "validated approval receipt.\n",
        encoding="utf-8",
    )
    final_receipt = tmp_path / "final-local-receipt.json"
    _write_json(final_receipt, _final_local_receipt())
    target = {
        "package_version": successor.PACKAGE_VERSION,
        "github_repository": GITHUB_REPOSITORY,
        "hugging_face_repository": HUGGING_FACE_REPOSITORY,
    }
    license_receipt = _seal(
        {
            "schema_version": "pii-zh.community-v2-human-license-approval-receipt.v1",
            "target": target,
            "decision": {
                "approved": True,
                "approved_by": "fixture-maintainer",
                "approved_at": "2026-07-19T08:00:00Z",
                "scope": "public_github_and_hugging_face_distribution",
                "exceptions": [],
            },
            "reviewed_files": {
                "LICENSE": _sha256((source / "LICENSE").read_bytes()),
                "NOTICE": _sha256(notice.read_bytes()),
                "THIRD_PARTY_NOTICES.md": _sha256(third_party_notices.read_bytes()),
            },
        }
    )
    license_path = tmp_path / "license-approval.json"
    _write_json(license_path, license_receipt)
    channel_receipt = _seal(
        {
            "schema_version": ("pii-zh.community-v2-tested-private-security-channel-receipt.v1"),
            "target": target,
            "channel_test": {
                "tested": True,
                "tested_by": "fixture-maintainer",
                "tested_at": "2026-07-19T08:05:00Z",
                "provider": "github_private_vulnerability_reporting",
                "evidence_basis": "human_attestation_not_remote_verified",
                "test_case_id": "synthetic-private-report-001",
                "outcome": "accepted_private_test_report",
                "contains_real_sensitive_data": False,
            },
            "security_file_sha256": _sha256(security.read_bytes()),
        }
    )
    channel_path = tmp_path / "security-channel-test.json"
    _write_json(channel_path, channel_receipt)
    return {
        "source": source,
        "model_card": model_card,
        "security": security,
        "notice": notice,
        "third_party_notices": third_party_notices,
        "final_receipt": final_receipt,
        "license_receipt": license_path,
        "channel_receipt": channel_path,
    }


def _prepare(tmp_path: Path, inputs: dict[str, Path]) -> successor.BuildPlan:
    return successor.prepare_build_plan(
        source_package=inputs["source"],
        output=tmp_path / "publication-successor",
        model_card=inputs["model_card"],
        security=inputs["security"],
        notice=inputs["notice"],
        third_party_notices=inputs["third_party_notices"],
        final_local_receipt=inputs["final_receipt"],
        human_license_approval_receipt=inputs["license_receipt"],
        tested_private_security_channel_receipt=inputs["channel_receipt"],
        git_source_commit=GIT_SOURCE_COMMIT,
        github_repository=GITHUB_REPOSITORY,
        hugging_face_repository=HUGGING_FACE_REPOSITORY,
    )


def test_builds_checksum_closed_successor_without_mutating_source(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    source_before = {
        path.relative_to(inputs["source"]).as_posix(): path.read_bytes()
        for path in inputs["source"].iterdir()
        if path.is_file()
    }
    plan = _prepare(tmp_path, inputs)
    result = successor.build_publication_successor(plan)

    output = plan.output
    assert result["status"] == "STAGED"
    assert result["publication_state"] == "staged_not_uploaded"
    assert (output / "README.md").read_bytes() == inputs["model_card"].read_bytes()
    assert (output / "SECURITY.md").read_bytes() == inputs["security"].read_bytes()
    assert (output / "NOTICE").read_bytes() == inputs["notice"].read_bytes()
    assert (output / "THIRD_PARTY_NOTICES.md").read_bytes() == inputs[
        "third_party_notices"
    ].read_bytes()
    assert (output / successor.HF_GITATTRIBUTES_NAME).read_bytes() == (
        successor.REPOSITORY_ROOT / successor.HF_GITATTRIBUTES_TEMPLATE_PATH
    ).read_bytes()
    assert b"unpublished local" not in (output / "NOTICE").read_bytes().lower()
    approval = json.loads(
        (output / successor.LICENSE_APPROVAL_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert approval["reviewed_files"] == {
        "LICENSE": _sha256((inputs["source"] / "LICENSE").read_bytes()),
        "NOTICE": _sha256(inputs["notice"].read_bytes()),
        "THIRD_PARTY_NOTICES.md": _sha256(inputs["third_party_notices"].read_bytes()),
    }
    assert not (output / successor.PREAUTHORIZATION_NAME).exists()
    assert stat.S_ISREG(os.lstat(output / "model.safetensors").st_mode)
    assert (output / "model.safetensors").read_bytes() == (
        inputs["source"] / "model.safetensors"
    ).read_bytes()
    assert source_before == {
        path.relative_to(inputs["source"]).as_posix(): path.read_bytes()
        for path in inputs["source"].iterdir()
        if path.is_file()
    }

    verification = successor.verify_successor_package(output)
    assert verification["verified_file_count"] == verification["file_count"] - 1
    manifest = json.loads((output / successor.MANIFEST_NAME).read_text(encoding="utf-8"))
    schema = json.loads(
        (successor.REPOSITORY_ROOT / successor.MANIFEST_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(manifest)
    assert manifest["manifest_sha256"] == successor.canonical_json_hash(
        manifest, remove="manifest_sha256"
    )
    assert manifest["source_package"]["inventory_sha256"] == plan.source.inventory_sha256
    assert manifest["source_control"]["git_source_commit"] == GIT_SOURCE_COMMIT
    assert manifest["publication_targets"] == {
        "github_repository": GITHUB_REPOSITORY,
        "hugging_face_repository": HUGGING_FACE_REPOSITORY,
    }
    assert manifest["remote_revisions"] == {
        "github_commit": None,
        "hugging_face_commit": None,
    }
    assert manifest["publication_authorization_source"] == "external_receipts"
    gitattributes_binding = manifest["payload_files"][successor.HF_GITATTRIBUTES_NAME]
    assert gitattributes_binding["provenance"] == "reviewed_hugging_face_gitattributes"
    assert gitattributes_binding["transfer_method"] == "copy"
    assert gitattributes_binding["output_type"] == "regular_file"
    lineage = manifest["candidate_lineage_contract"]
    assert lineage["interpretation"] == (
        "immutable_candidate_lineage_not_current_remote_publication_state"
    )
    assert lineage["byte_preserved"] is True
    assert lineage["authorization_does_not_mutate_runtime_contract"] is True
    assert lineage["files"]["config.json"]["contract"] == {
        "pii_release_eligible": False,
        "pii_lineage.release_eligible": False,
        "pii_training_status": "completed_candidate_not_benchmark_evaluated",
    }
    assert (output / "config.json").read_bytes() == (inputs["source"] / "config.json").read_bytes()
    assert (output / "training_manifest.json").read_bytes() == (
        inputs["source"] / "training_manifest.json"
    ).read_bytes()
    weight_binding = manifest["payload_files"]["model.safetensors"]
    assert weight_binding["transfer_method"] in {"reflink", "copy"}
    assert weight_binding["output_type"] == "regular_file"
    assert (output / "model.safetensors").stat().st_ino != (
        inputs["source"] / "model.safetensors"
    ).stat().st_ino


def test_staged_model_weights_do_not_follow_later_source_writes(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    plan = _prepare(tmp_path, inputs)
    successor.build_publication_successor(plan)
    source_weights = inputs["source"] / "model.safetensors"
    output_weights = plan.output / "model.safetensors"
    output_before = output_weights.read_bytes()

    source_weights.write_bytes(b"source-mutated-after-successor-build")

    assert output_weights.read_bytes() == output_before
    assert output_weights.stat().st_ino != source_weights.stat().st_ino


def test_reflink_unsupported_falls_back_to_independent_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture_inputs(tmp_path)

    def unsupported_reflink(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "fixture filesystem has no reflink")

    monkeypatch.setattr(successor.fcntl, "ioctl", unsupported_reflink)
    plan = _prepare(tmp_path, inputs)
    successor.build_publication_successor(plan)
    manifest = json.loads((plan.output / successor.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["payload_files"]["model.safetensors"]["transfer_method"] == "copy"
    assert (plan.output / "model.safetensors").stat().st_ino != (
        inputs["source"] / "model.safetensors"
    ).stat().st_ino


def test_source_checksum_tamper_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    (inputs["source"] / "config.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(successor.PublicationSuccessorError, match="checksum mismatch") as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "SOURCE_CHECKSUM_MISMATCH"
    assert not (tmp_path / "publication-successor").exists()


def test_verifier_rejects_manifest_payload_binding_lie_even_with_resealed_files(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    plan = _prepare(tmp_path, inputs)
    successor.build_publication_successor(plan)
    manifest_path = plan.output / successor.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_files"]["README.md"]["file_sha256"] = "0" * 64
    manifest["payload_inventory_sha256"] = successor.canonical_json_hash(
        manifest["payload_files"]
    )
    manifest["manifest_sha256"] = successor.canonical_json_hash(
        manifest, remove="manifest_sha256"
    )
    _write_json(manifest_path, manifest)
    _write_checksums(plan.output)

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        successor.verify_successor_package(plan.output)
    assert captured.value.blocker_id == "MANIFEST_PAYLOAD_BINDING_MISMATCH"


def test_verifier_rejects_resealed_manifest_payload_inventory_hash_lie(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    plan = _prepare(tmp_path, inputs)
    successor.build_publication_successor(plan)
    manifest_path = plan.output / successor.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_inventory_sha256"] = "0" * 64
    manifest["manifest_sha256"] = successor.canonical_json_hash(
        manifest, remove="manifest_sha256"
    )
    _write_json(manifest_path, manifest)
    _write_checksums(plan.output)

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        successor.verify_successor_package(plan.output)
    assert captured.value.blocker_id == "MANIFEST_PAYLOAD_INVENTORY_MISMATCH"


def test_hugging_face_gitattributes_template_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture_inputs(tmp_path)
    drifted = tmp_path / "huggingface.gitattributes"
    drifted.write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    monkeypatch.setattr(successor, "HF_GITATTRIBUTES_TEMPLATE_PATH", drifted)

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "INVALID_HUGGING_FACE_GITATTRIBUTES"
    assert not (tmp_path / "publication-successor").exists()


def test_source_symlink_fails_closed_even_when_unlisted(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    (inputs["source"] / "linked-model").symlink_to("model.safetensors")
    with pytest.raises(successor.PublicationSuccessorError, match="symlink") as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "SOURCE_SYMLINK_REJECTED"


def test_stale_marker_in_other_retained_metadata_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    (inputs["source"] / "extra-metadata.txt").write_text(
        "unpublished local candidate\n", encoding="utf-8"
    )
    _write_checksums(inputs["source"])
    with pytest.raises(successor.PublicationSuccessorError, match="stale marker") as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "STALE_PUBLICATION_MARKER"


def test_model_card_exact_repository_targets_are_bound_into_plan(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    plan = _prepare(tmp_path, inputs)
    text = plan.model_card.payload.decode("utf-8")
    assert f"github_repository: {GITHUB_REPOSITORY}" in text
    assert f"hugging_face_repository: {HUGGING_FACE_REPOSITORY}" in text
    assert f"git_source_commit: {GIT_SOURCE_COMMIT}" in text
    assert f"release_tag: {successor.RELEASE_TAG}" in text
    assert (
        f"github_release_url: https://github.com/{GITHUB_REPOSITORY}/releases/tag/"
        f"{successor.RELEASE_TAG}"
    ) in text
    assert plan.github_repository == GITHUB_REPOSITORY
    assert plan.hugging_face_repository == HUGGING_FACE_REPOSITORY


@pytest.mark.parametrize(
    ("hugging_face_target", "blocker_id"),
    [
        (
            "HF_NAMESPACE/pii-zh-qwen3-0.6b-24class",
            "MODEL_CARD_TARGET_PLACEHOLDER",
        ),
        ("<namespace>/pii-zh-qwen3-0.6b-24class", "MODEL_CARD_TARGET_PLACEHOLDER"),
        ("different-owner/pii-zh-qwen3-0.6b-24class", "MODEL_CARD_TARGET_MISMATCH"),
    ],
)
def test_model_card_placeholder_or_mismatched_target_fails_closed(
    tmp_path: Path, hugging_face_target: str, blocker_id: str
) -> None:
    inputs = _fixture_inputs(tmp_path)
    inputs["model_card"].write_text(
        "# Community v2\n\n"
        "Version 0.2.0rc1 publication candidate.\n\n"
        f"github_repository: {GITHUB_REPOSITORY}\n"
        f"hugging_face_repository: {hugging_face_target}\n"
        f"git_source_commit: {GIT_SOURCE_COMMIT}\n"
        f"release_tag: {successor.RELEASE_TAG}\n"
        "github_release_url: "
        f"https://github.com/{GITHUB_REPOSITORY}/releases/tag/{successor.RELEASE_TAG}\n",
        encoding="utf-8",
    )
    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == blocker_id
    assert not (tmp_path / "publication-successor").exists()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (GIT_SOURCE_COMMIT, "b" * 40),
        (successor.RELEASE_TAG, "v0.2.0"),
        (
            f"https://github.com/{GITHUB_REPOSITORY}/releases/tag/{successor.RELEASE_TAG}",
            "https://github.com/other/repository/releases/tag/v0.2.0rc1",
        ),
    ],
)
def test_model_card_source_tag_or_release_url_mismatch_fails_closed(
    tmp_path: Path, old: str, new: str
) -> None:
    inputs = _fixture_inputs(tmp_path)
    text = inputs["model_card"].read_text(encoding="utf-8")
    inputs["model_card"].write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "MODEL_CARD_TARGET_MISMATCH"


def test_explained_historical_license_pending_status_is_allowed_when_receipt_binds_it(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    plan = _prepare(tmp_path, inputs)
    assert successor._LICENSE_HISTORY_MARKER in plan.third_party_notices.payload.decode("utf-8")
    assert plan.license_approval_document["reviewed_files"]["THIRD_PARTY_NOTICES.md"] == (
        _sha256(plan.third_party_notices.payload)
    )


@pytest.mark.parametrize(
    "third_party_text",
    [
        "# Third-Party Notices for 0.2.0rc1\n\nPublication approval is pending.\n",
        (
            "# Third-Party Notices for 0.2.0rc1\n\n"
            "Status `COMPLETE_HUMAN_APPROVAL_PENDING` is retained.\n"
        ),
        (
            "# Third-Party Notices for 0.2.0rc1\n\n"
            "The immutable license report retains status `COMPLETE_HUMAN_APPROVAL_PENDING` "
            "as historical mechanical evidence. Human clearance comes from the separately "
            "validated approval receipt. Public distribution is still pending.\n"
        ),
    ],
)
def test_real_pending_publication_text_still_fails_with_a_binding_license_receipt(
    tmp_path: Path, third_party_text: str
) -> None:
    inputs = _fixture_inputs(tmp_path)
    inputs["third_party_notices"].write_text(third_party_text, encoding="utf-8")
    receipt = json.loads(inputs["license_receipt"].read_text(encoding="utf-8"))
    receipt["reviewed_files"]["THIRD_PARTY_NOTICES.md"] = _sha256(
        inputs["third_party_notices"].read_bytes()
    )
    _write_json(inputs["license_receipt"], _seal(receipt))

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "STALE_PREAUTHORIZATION_TEXT"


def test_license_receipt_must_bind_publication_not_old_source_notices(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    receipt = json.loads(inputs["license_receipt"].read_text(encoding="utf-8"))
    receipt["reviewed_files"]["NOTICE"] = _sha256((inputs["source"] / "NOTICE").read_bytes())
    receipt["reviewed_files"]["THIRD_PARTY_NOTICES.md"] = _sha256(
        (inputs["source"] / "THIRD_PARTY_NOTICES.md").read_bytes()
    )
    _write_json(inputs["license_receipt"], _seal(receipt))
    with pytest.raises(successor.PublicationSuccessorError, match="does not bind") as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "LICENSE_APPROVAL_BINDING_MISMATCH"


@pytest.mark.parametrize(
    "mutation",
    ["missing_evidence_basis", "remote_verified_evidence", "blank_tested_by"],
)
def test_security_receipt_requires_human_non_remote_evidence_contract(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _fixture_inputs(tmp_path)
    channel = json.loads(inputs["channel_receipt"].read_text(encoding="utf-8"))
    if mutation == "missing_evidence_basis":
        channel["channel_test"].pop("evidence_basis")
    elif mutation == "remote_verified_evidence":
        channel["channel_test"]["evidence_basis"] = "remote_verified"
    else:
        channel["channel_test"]["tested_by"] = "   "
    _write_json(inputs["channel_receipt"], _seal(channel))

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "RECEIPT_SCHEMA_REJECTED"


@pytest.mark.parametrize(
    "security_text",
    [
        "# Security\n\nThe synthetic private-report test is complete.\n",
        (
            "# Security\n\nUse "
            "https://github.com/different-owner/pii-detect-model/security/advisories/new.\n"
        ),
    ],
)
def test_successor_rejects_security_without_exact_target_reporting_url(
    tmp_path: Path, security_text: str
) -> None:
    inputs = _fixture_inputs(tmp_path)
    inputs["security"].write_text(security_text, encoding="utf-8")
    channel = json.loads(inputs["channel_receipt"].read_text(encoding="utf-8"))
    channel["security_file_sha256"] = _sha256(inputs["security"].read_bytes())
    _write_json(inputs["channel_receipt"], _seal(channel))

    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "SECURITY_REPORTING_URL_MISSING"


@pytest.mark.parametrize(
    ("mutation", "blocker_id"),
    [
        ("duplicate", "DUPLICATE_CHECKSUM_PATH"),
        ("escape", "UNSAFE_PACKAGE_PATH"),
        ("unlisted", "SOURCE_UNLISTED_FILE"),
    ],
)
def test_source_checksum_closure_rejects_duplicate_escape_and_unlisted_files(
    tmp_path: Path, mutation: str, blocker_id: str
) -> None:
    inputs = _fixture_inputs(tmp_path)
    checksums = inputs["source"] / "checksums.txt"
    if mutation == "duplicate":
        first = checksums.read_text(encoding="utf-8").splitlines()[0]
        checksums.write_text(checksums.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
    elif mutation == "escape":
        checksums.write_text(
            checksums.read_text(encoding="utf-8") + f"{'0' * 64}  ../escape\n",
            encoding="utf-8",
        )
    else:
        (inputs["source"] / "unlisted.txt").write_text("unlisted\n", encoding="utf-8")
    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == blocker_id


def test_missing_human_approval_receipt_reports_blocked_in_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "publication-successor"
    return_code = successor.main(
        [
            "--source-package",
            str(inputs["source"]),
            "--output",
            str(output),
            "--model-card",
            str(inputs["model_card"]),
            "--security",
            str(inputs["security"]),
            "--notice",
            str(inputs["notice"]),
            "--third-party-notices",
            str(inputs["third_party_notices"]),
            "--final-local-receipt",
            str(inputs["final_receipt"]),
            "--tested-private-security-channel-receipt",
            str(inputs["channel_receipt"]),
            "--git-source-commit",
            GIT_SOURCE_COMMIT,
            "--github-repository",
            GITHUB_REPOSITORY,
            "--hugging-face-repository",
            HUGGING_FACE_REPOSITORY,
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert return_code == 2
    assert report["status"] == "BLOCKED"
    assert report["mode"] == "preflight"
    assert report["blocker_ids"] == ["HUMAN_LICENSE_APPROVAL_RECEIPT_MISSING"]
    assert report["remote_write_performed"] is False
    assert not output.exists()


def test_receipt_cannot_replace_structured_human_approval_with_caller_pass(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    receipt = json.loads(inputs["license_receipt"].read_text(encoding="utf-8"))
    receipt["decision"]["approved"] = False
    receipt["status"] = "PASS"
    receipt = _seal(receipt)
    _write_json(inputs["license_receipt"], receipt)
    with pytest.raises(successor.PublicationSuccessorError) as captured:
        _prepare(tmp_path, inputs)
    assert captured.value.blocker_id == "RECEIPT_SCHEMA_REJECTED"


def test_build_is_no_clobber_and_preserves_existing_output(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    plan = _prepare(tmp_path, inputs)
    plan.output.mkdir()
    sentinel = plan.output / "belongs-to-caller.txt"
    sentinel.write_text("do not overwrite\n", encoding="utf-8")
    with pytest.raises(successor.PublicationSuccessorError) as captured:
        successor.build_publication_successor(plan)
    assert captured.value.blocker_id == "OUTPUT_ALREADY_EXISTS"
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite\n"
