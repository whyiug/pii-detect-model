from __future__ import annotations

import copy
import hashlib
import json
import socket
import stat
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import verify_community_v2_post_attachment as verifier

SOURCE_SHA = "a" * 40
TAG_OBJECT_SHA = "b" * 40
HF_SHA = "c" * 40
VERIFIED_AT = "2026-07-20T12:00:00Z"


def _sha256(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _seal(document: dict[str, Any], *, key: str) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result[key] = ""
    result[key] = verifier.canonical_json_hash(result, remove=key)
    return result


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _write_immutable(path: Path, document: dict[str, Any]) -> bytes:
    payload = _json_bytes(document)
    path.write_bytes(payload)
    path.chmod(0o444)
    return payload


def _release_assets() -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    contents = {
        "pii_zh_qwen-0.2.0rc1-py3-none-any.whl": b"wheel bytes",
        "sbom.cdx.json": b'{"bomFormat":"CycloneDX"}\n',
    }
    contents["checksums.txt"] = (
        f"{_sha256(contents['pii_zh_qwen-0.2.0rc1-py3-none-any.whl'])}  "
        "pii_zh_qwen-0.2.0rc1-py3-none-any.whl\n"
        f"{_sha256(contents['sbom.cdx.json'])}  sbom.cdx.json\n"
    ).encode()
    roles = {
        "checksums.txt": "checksums",
        "pii_zh_qwen-0.2.0rc1-py3-none-any.whl": "wheel",
        "sbom.cdx.json": "sbom",
    }
    assets = [
        {
            "asset_id": 1000 + index,
            "role": roles[name],
            "name": name,
            "size_bytes": len(contents[name]),
            "state": "uploaded",
            "github_digest_sha256": _sha256(contents[name]),
            "downloaded_sha256": _sha256(contents[name]),
        }
        for index, name in enumerate(sorted(contents), start=1)
    ]
    return assets, contents


def _github_document() -> dict[str, Any]:
    assets, contents = _release_assets()
    body = (
        "Pre-publication candidate.\n"
        f"Immutable model revision: https://huggingface.co/"
        f"{verifier.HUGGING_FACE_REPOSITORY}/tree/{HF_SHA}\n"
    )
    return _seal(
        {
            "schema_version": verifier.GITHUB_EVIDENCE_SCHEMA_VERSION,
            "repository": {
                "id": verifier.GITHUB_REPOSITORY,
                "visibility": "public",
            },
            "source_commit_sha": SOURCE_SHA,
            "signed_tag": {
                "name": verifier.RELEASE_TAG,
                "ref_target_type": "tag",
                "ref_target_sha": TAG_OBJECT_SHA,
                "tag_object_sha": TAG_OBJECT_SHA,
                "tag_target_type": "commit",
                "tag_target_sha": SOURCE_SHA,
            },
            "release": {
                "release_id": 12345,
                "tag_name": verifier.RELEASE_TAG,
                "resolved_source_sha": SOURCE_SHA,
                "draft": True,
                "prerelease": True,
                "created_at": "2026-07-20T10:00:00Z",
                "updated_at": "2026-07-20T10:10:00Z",
                "body": body,
                "body_sha256": _sha256(body),
                "optional_asset_names": [],
                "checksums": {
                    "asset_name": "checksums.txt",
                    "file_sha256": _sha256(contents["checksums.txt"]),
                    "format": "sha256sum_text_two_space_two_lines_lf",
                    "line_count": 2,
                    "entries": [
                        {
                            "name": "pii_zh_qwen-0.2.0rc1-py3-none-any.whl",
                            "sha256": _sha256(
                                contents[
                                    "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
                                ]
                            ),
                        },
                        {
                            "name": "sbom.cdx.json",
                            "sha256": _sha256(contents["sbom.cdx.json"]),
                        },
                    ],
                    "matched_downloaded_assets": True,
                },
                "assets": assets,
            },
            "evidence_sha256": "",
        },
        key="evidence_sha256",
    )


def _hf_paths() -> list[str]:
    return sorted(
        [
            ".gitattributes",
            "LICENSE",
            "NOTICE",
            "README.md",
            "SECURITY.md",
            "THIRD_PARTY_NOTICES.md",
            "checksums.txt",
            "community_v2_final_local_receipt.json",
            "config.json",
            "human_license_approval_receipt.json",
            "model.safetensors",
            "publication_manifest.json",
            "tested_private_security_channel_receipt.json",
        ]
    )


def _hf_document() -> dict[str, Any]:
    inventory = [
        {
            "path": path,
            "kind": "file",
            "size_bytes": index + 10,
            "remote_oid": (
                _sha256(f"lfs {path}")
                if path == "model.safetensors"
                else f"{index + 1:040x}"
            ),
            "downloaded_sha256": _sha256(f"downloaded {path}"),
        }
        for index, path in enumerate(_hf_paths())
    ]
    return _seal(
        {
            "schema_version": verifier.HUGGING_FACE_EVIDENCE_SCHEMA_VERSION,
            "repository": {
                "id": verifier.HUGGING_FACE_REPOSITORY,
                "visibility": "private",
            },
            "revision": {
                "requested_revision": HF_SHA,
                "resolved_sha": HF_SHA,
                "immutable_sha": HF_SHA,
            },
            "inventory": inventory,
            "cross_reference": {
                "path": "README.md",
                "file_sha256": _sha256("readme"),
                "github_repository": verifier.GITHUB_REPOSITORY,
                "github_source_sha": SOURCE_SHA,
            },
            "evidence_sha256": "",
        },
        key="evidence_sha256",
    )


def _prepublication_document(
    github: dict[str, Any],
    github_file_sha256: str,
    hugging_face: dict[str, Any],
    hugging_face_file_sha256: str,
) -> dict[str, Any]:
    projected_assets = [
        {
            "asset_id": item["asset_id"],
            "role": item["role"],
            "name": item["name"],
            "size_bytes": item["size_bytes"],
            "sha256": item["downloaded_sha256"],
        }
        for item in github["release"]["assets"]
    ]
    projected_inventory = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "remote_oid": item["remote_oid"],
            "sha256": item["downloaded_sha256"],
        }
        for item in hugging_face["inventory"]
    ]
    return _seal(
        {
            "schema_version": verifier.PREPUBLICATION_RECEIPT_SCHEMA_VERSION,
            "receipt_type": "community_v2_pre_publication_gate",
            "package_version": verifier.PACKAGE_VERSION,
            "recorded_at": "2026-07-20T11:00:00Z",
            "status": "READY_FOR_FINAL_PUBLICATION_CONFIRMATION",
            "stage": "pre_publication_gate",
            "final_publication_confirmed": False,
            "evidence_bindings": {
                "github": {
                    "schema_version": github["schema_version"],
                    "file_sha256": github_file_sha256,
                    "evidence_sha256": github["evidence_sha256"],
                },
                "hugging_face": {
                    "schema_version": hugging_face["schema_version"],
                    "file_sha256": hugging_face_file_sha256,
                    "evidence_sha256": hugging_face["evidence_sha256"],
                },
            },
            "github": {
                "repository": verifier.GITHUB_REPOSITORY,
                "visibility": "public",
                "source_commit_sha": SOURCE_SHA,
                "signed_tag": github["signed_tag"],
                "release": {
                    "release_id": github["release"]["release_id"],
                    "tag_name": verifier.RELEASE_TAG,
                    "resolved_source_sha": SOURCE_SHA,
                    "body_sha256": github["release"]["body_sha256"],
                    "draft": True,
                    "prerelease": True,
                },
            },
            "hugging_face": {
                "repository": verifier.HUGGING_FACE_REPOSITORY,
                "visibility": "private",
                "immutable_sha": HF_SHA,
                "inventory": projected_inventory,
                "inventory_sha256": verifier.canonical_json_hash(projected_inventory),
            },
            "release_assets": projected_assets,
            "release_asset_inventory_sha256": verifier.canonical_json_hash(
                projected_assets
            ),
            "draft_release_attachment": {
                "asset_name": verifier.PREPUBLICATION_RECEIPT_ASSET,
                "attach_after_receipt_generation": True,
                "excluded_from_bound_release_asset_inventory": True,
                "release_must_remain_draft": True,
            },
            "receipt_sha256": "",
        },
        key="receipt_sha256",
    )


class FakeGitHubTransport:
    def __init__(
        self,
        *,
        github: dict[str, Any],
        receipt_bytes: bytes,
    ) -> None:
        self.receipt_bytes = receipt_bytes
        self.download_endpoints: list[str] = []
        self.get_endpoints: list[str] = []
        self.release = {
            "id": github["release"]["release_id"],
            "tag_name": verifier.RELEASE_TAG,
            "draft": True,
            "prerelease": True,
            "created_at": github["release"]["created_at"],
            "updated_at": "2026-07-20T11:05:00Z",
            "body": github["release"]["body"],
            "assets": [
                {
                    "id": item["asset_id"],
                    "name": item["name"],
                    "size": item["size_bytes"],
                    "state": "uploaded",
                    "digest": f"sha256:{item['downloaded_sha256']}",
                }
                for item in github["release"]["assets"]
            ]
            + [
                {
                    "id": 2001,
                    "name": verifier.PREPUBLICATION_RECEIPT_ASSET,
                    "size": len(receipt_bytes),
                    "state": "uploaded",
                    "digest": f"sha256:{_sha256(receipt_bytes)}",
                }
            ],
        }
        self.reference = {"object": {"type": "tag", "sha": TAG_OBJECT_SHA}}
        self.tag_object = {
            "sha": TAG_OBJECT_SHA,
            "object": {"type": "commit", "sha": SOURCE_SHA},
        }

    def get_json(self, endpoint: str) -> Any:
        self.get_endpoints.append(endpoint)
        if "/releases/" in endpoint and "/assets/" not in endpoint:
            return copy.deepcopy(self.release)
        if "/git/ref/tags/" in endpoint:
            return copy.deepcopy(self.reference)
        if "/git/tags/" in endpoint:
            return copy.deepcopy(self.tag_object)
        raise AssertionError(f"unexpected GitHub endpoint: {endpoint}")

    def download_bytes(self, endpoint: str, *, maximum: int) -> bytes:
        self.download_endpoints.append(endpoint)
        assert maximum == verifier.MAX_RECEIPT_DOWNLOAD_BYTES
        assert endpoint.endswith("/releases/assets/2001")
        return self.receipt_bytes


class FakeHuggingFaceTransport:
    def __init__(self, hugging_face: dict[str, Any]) -> None:
        self.calls: list[tuple[str, str]] = []
        self.repo_id = verifier.HUGGING_FACE_REPOSITORY
        self.private = True
        self.sha = HF_SHA
        self.paths = [item["path"] for item in hugging_face["inventory"]]
        self.siblings = [
            {
                "rfilename": item["path"],
                "size": item["size_bytes"],
                "blob_id": (
                    None if len(item["remote_oid"]) == 64 else item["remote_oid"]
                ),
                "lfs": (
                    {"sha256": item["remote_oid"]}
                    if len(item["remote_oid"]) == 64
                    else None
                ),
            }
            for item in hugging_face["inventory"]
        ]

    def repo_info(self, *, revision: str) -> object:
        self.calls.append(("repo_info", revision))
        return {
            "id": self.repo_id,
            "private": self.private,
            "sha": self.sha,
            "siblings": copy.deepcopy(self.siblings),
        }

    def list_repo_files(self, *, revision: str) -> list[str]:
        self.calls.append(("list_repo_files", revision))
        return list(self.paths)


def _fixture(
    tmp_path: Path,
) -> tuple[
    verifier.JsonPayload,
    verifier.JsonPayload,
    verifier.JsonPayload,
    FakeGitHubTransport,
    FakeHuggingFaceTransport,
]:
    github = _github_document()
    hugging_face = _hf_document()
    github_path = tmp_path / "github.json"
    hf_path = tmp_path / "hf.json"
    github_bytes = _write_immutable(github_path, github)
    hf_bytes = _write_immutable(hf_path, hugging_face)
    prepublication = _prepublication_document(
        github,
        _sha256(github_bytes),
        hugging_face,
        _sha256(hf_bytes),
    )
    prepublication_path = tmp_path / verifier.PREPUBLICATION_RECEIPT_ASSET
    receipt_bytes = _write_immutable(prepublication_path, prepublication)
    github_payload = verifier.load_json_payload(
        github_path, field="GitHub collector evidence"
    )
    hf_payload = verifier.load_json_payload(
        hf_path, field="Hugging Face collector evidence"
    )
    receipt_payload = verifier.load_json_payload(
        prepublication_path, field="pre-publication receipt"
    )
    return (
        github_payload,
        hf_payload,
        receipt_payload,
        FakeGitHubTransport(github=github, receipt_bytes=receipt_bytes),
        FakeHuggingFaceTransport(hugging_face),
    )


def _build(
    fixture: tuple[
        verifier.JsonPayload,
        verifier.JsonPayload,
        verifier.JsonPayload,
        FakeGitHubTransport,
        FakeHuggingFaceTransport,
    ],
) -> dict[str, Any]:
    github, hf, receipt, github_transport, hf_transport = fixture
    return verifier.build_post_attachment_receipt(
        github_payload=github,
        hugging_face_payload=hf,
        prepublication_payload=receipt,
        github_transport=github_transport,
        hugging_face_transport=hf_transport,
        verified_at=VERIFIED_AT,
    )


def test_fake_transports_verify_exact_post_attachment_state_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def forbid_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unit test attempted real network access")

    monkeypatch.setattr(socket, "socket", forbid_network)
    receipt = _build(fixture)
    schema = json.loads(
        (verifier.REPOSITORY_ROOT / verifier.SCHEMA_PATH).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    verifier.validate_output_receipt(receipt)

    assert receipt["status"] == "PASS"
    assert receipt["final_publication_confirmed"] is False
    assert receipt["github"]["release"]["draft"] is True
    assert receipt["hugging_face"]["private"] is True
    assert receipt["hugging_face"]["main_resolved_sha"] == HF_SHA
    assert receipt["receipt_sha256"] == verifier.canonical_json_hash(
        receipt, remove="receipt_sha256"
    )
    assets = receipt["github"]["release_assets"]
    assert [item["name"] for item in assets] == sorted(
        verifier.EXPECTED_REMOTE_ASSET_NAMES
    )
    attached = next(
        item for item in assets if item["role"] == "prepublication_receipt"
    )
    assert attached["downloaded_sha256"] == fixture[2].file_sha256
    assert fixture[3].download_endpoints == [
        f"/repos/{verifier.GITHUB_REPOSITORY}/releases/assets/2001"
    ]
    assert all("model.safetensors" not in str(call) for call in fixture[4].calls)


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("draft", False, "GITHUB_RELEASE_DRIFT"),
        ("prerelease", False, "GITHUB_RELEASE_DRIFT"),
        ("tag_name", "v0.2.0", "GITHUB_RELEASE_DRIFT"),
        ("body", "changed body", "GITHUB_RELEASE_DRIFT"),
    ],
)
def test_release_identity_state_tag_and_body_are_exact(
    tmp_path: Path, field: str, value: object, blocker: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture[3].release[field] = value
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == blocker


def test_extra_remote_attachment_is_rejected_before_any_download(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture[3].release["assets"].append(
        {
            "id": 9999,
            "name": "unexpected.txt",
            "size": 1,
            "state": "uploaded",
            "digest": f"sha256:{'d' * 64}",
        }
    )
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == "GITHUB_POST_ATTACHMENT_ASSET_SET_MISMATCH"
    assert fixture[3].download_endpoints == []


@pytest.mark.parametrize("field", ["optional_asset_names", "checksums"])
def test_pre_attachment_evidence_preserves_closed_asset_contract(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    changed = copy.deepcopy(fixture[0].document)
    if field == "optional_asset_names":
        changed["release"][field] = ["unexpected.txt"]
    else:
        changed["release"][field]["entries"][0]["sha256"] = "d" * 64
    changed = _seal(changed, key="evidence_sha256")
    changed_bytes = _json_bytes(changed)
    changed_payload = verifier.JsonPayload(
        payload=changed_bytes,
        document=changed,
        file_sha256=_sha256(changed_bytes),
        mode=0o444,
    )
    changed_fixture = (changed_payload, *fixture[1:])
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(changed_fixture)
    assert captured.value.blocker_id in {
        "GITHUB_EVIDENCE_RELEASE_MISMATCH",
        "GITHUB_EVIDENCE_CHECKSUMS_MISMATCH",
    }


@pytest.mark.parametrize("field", ["size", "digest"])
def test_original_asset_size_and_digest_remain_bound_to_collector_evidence(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    original = next(
        item
        for item in fixture[3].release["assets"]
        if item["name"] == "sbom.cdx.json"
    )
    original[field] = (
        original["size"] + 1 if field == "size" else f"sha256:{'d' * 64}"
    )
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == "GITHUB_ORIGINAL_ASSET_DRIFT"


def test_downloaded_receipt_bytes_must_exactly_equal_local_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture[3].receipt_bytes = b"different but still JSON-looking bytes"
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == "GITHUB_RECEIPT_BYTES_MISMATCH"


def test_receipt_asset_metadata_must_match_local_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    attached = next(
        item
        for item in fixture[3].release["assets"]
        if item["name"] == verifier.PREPUBLICATION_RECEIPT_ASSET
    )
    attached["digest"] = f"sha256:{'d' * 64}"
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == "GITHUB_RECEIPT_METADATA_MISMATCH"


def test_annotated_tag_must_still_target_exact_source(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture[3].tag_object["object"]["sha"] = "d" * 40
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == "GITHUB_ANNOTATED_TAG_DRIFT"


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("public", "HF_REMOTE_STATE_MISMATCH"),
        ("main_moved", "HF_REMOTE_REVISION_DRIFT"),
        ("inventory_oid", "HF_REMOTE_INVENTORY_DRIFT"),
        ("inventory_size", "HF_REMOTE_INVENTORY_DRIFT"),
    ],
)
def test_hf_private_main_immutable_and_inventory_are_exact(
    tmp_path: Path, mutation: str, blocker: str
) -> None:
    fixture = _fixture(tmp_path)
    transport = fixture[4]
    if mutation == "public":
        transport.private = False
    elif mutation == "main_moved":
        transport.sha = "d" * 40
    elif mutation == "inventory_oid":
        transport.siblings[0]["blob_id"] = "d" * 40
    else:
        transport.siblings[0]["size"] += 1
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(fixture)
    assert captured.value.blocker_id == blocker


def test_input_self_hash_and_receipt_evidence_binding_are_rechecked(
    tmp_path: Path
) -> None:
    fixture = _fixture(tmp_path)
    tampered = copy.deepcopy(fixture[0].document)
    tampered["release"]["created_at"] = "2026-07-20T10:00:01Z"
    broken_payload = verifier.JsonPayload(
        payload=fixture[0].payload,
        document=tampered,
        file_sha256=fixture[0].file_sha256,
        mode=0o444,
    )
    broken_fixture = (broken_payload, *fixture[1:])
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(broken_fixture)
    assert captured.value.blocker_id == "INPUT_DOCUMENT_MISMATCH"

    resealed = _seal(tampered, key="evidence_sha256")
    resealed_bytes = _json_bytes(resealed)
    resealed_payload = verifier.JsonPayload(
        payload=resealed_bytes,
        document=resealed,
        file_sha256=_sha256(resealed_bytes),
        mode=0o444,
    )
    broken_fixture = (resealed_payload, *fixture[1:])
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        _build(broken_fixture)
    assert captured.value.blocker_id == "PREPUBLICATION_EVIDENCE_BINDING_MISMATCH"


def test_input_files_must_be_mode_0444_and_not_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "mutable.json"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        verifier.load_json_payload(path, field="mutable evidence")
    assert captured.value.blocker_id == "INPUT_MODE_INVALID"

    path.chmod(0o444)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        verifier.load_json_payload(link, field="symlink evidence")
    assert captured.value.blocker_id == "UNSAFE_INPUT_FILE"


def test_output_is_closed_self_hashed_0444_and_no_clobber(tmp_path: Path) -> None:
    receipt = _build(_fixture(tmp_path))
    output = tmp_path / "post-attachment.json"
    file_hash = verifier._write_new_receipt(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert file_hash == _sha256(output.read_bytes())
    original = output.read_bytes()
    with pytest.raises(verifier.PostAttachmentVerificationError) as captured:
        verifier._write_new_receipt(output, receipt)
    assert captured.value.blocker_id == "OUTPUT_EXISTS"
    assert output.read_bytes() == original


def test_github_transport_pins_official_host_and_ignores_gh_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = verifier.GhRestTransport.__new__(verifier.GhRestTransport)
    transport._executable = "/usr/bin/gh"
    monkeypatch.setenv("GH_HOST", "attacker.invalid")
    monkeypatch.setenv("GH_TOKEN", "secret")
    command = transport._command(
        f"/repos/{verifier.GITHUB_REPOSITORY}/releases/123", accept="application/json"
    )
    assert command[command.index("--hostname") + 1] == "github.com"
    assert "GH_HOST" not in transport._environment()
    assert transport._environment()["GH_TOKEN"] == "secret"


def test_hf_transport_pins_official_endpoint_and_uses_process_token_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str] = {}

    class FakeApi:
        def __init__(self, *, endpoint: str, token: str) -> None:
            observed["endpoint"] = endpoint
            observed["token"] = token

    monkeypatch.setattr(verifier, "HfApi", FakeApi)
    monkeypatch.setenv("HF_ENDPOINT", "https://attacker.invalid")
    monkeypatch.setenv("HF_TOKEN", "private-test-token")
    verifier.OfficialHuggingFaceTransport()
    assert observed == {
        "endpoint": "https://huggingface.co",
        "token": "private-test-token",
    }


def test_cli_logs_only_safe_summary_with_fake_transports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    github_path = tmp_path / "github-cli.json"
    hf_path = tmp_path / "hf-cli.json"
    receipt_path = tmp_path / "receipt-cli.json"
    for target, payload in zip(
        (github_path, hf_path, receipt_path), fixture[:3], strict=True
    ):
        target.write_bytes(payload.payload)
        target.chmod(0o444)
    monkeypatch.setattr(verifier, "GhRestTransport", lambda: fixture[3])
    monkeypatch.setattr(verifier, "OfficialHuggingFaceTransport", lambda: fixture[4])
    output = tmp_path / "post-cli.json"
    assert (
        verifier.main(
            [
                "--github-evidence",
                str(github_path),
                "--hugging-face-evidence",
                str(hf_path),
                "--prepublication-receipt",
                str(receipt_path),
                "--verified-at",
                VERIFIED_AT,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary_text = capsys.readouterr().out
    summary = json.loads(summary_text)
    assert summary["status"] == "PASS"
    assert summary["remote_mutation_performed"] is False
    assert summary["hugging_face_weight_download_performed"] is False
    assert "private-test-token" not in summary_text
    assert fixture[2].payload.decode("utf-8") not in summary_text
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_no_real_remote_clients_are_instantiated_when_input_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    invalid.chmod(0o444)

    def forbidden() -> None:
        raise AssertionError("remote client was instantiated before input validation")

    monkeypatch.setattr(verifier, "GhRestTransport", forbidden)
    monkeypatch.setattr(verifier, "OfficialHuggingFaceTransport", forbidden)
    assert (
        verifier.main(
            [
                "--github-evidence",
                str(invalid),
                "--hugging-face-evidence",
                str(invalid),
                "--prepublication-receipt",
                str(invalid),
                "--output",
                str(tmp_path / "never.json"),
            ]
        )
        == 2
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "BLOCKED"
    assert summary["sensitive_values_logged"] is False
