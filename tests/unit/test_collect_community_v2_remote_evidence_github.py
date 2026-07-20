from __future__ import annotations

import base64
import copy
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from scripts import collect_community_v2_remote_evidence as collector

SOURCE_SHA = "a" * 40
TAG_OBJECT_SHA = "b" * 40
HF_SHA = "c" * 40
RUN_ID = 98765
RUN_ATTEMPT = 2
RELEASE_ID = 12345
COLLECTED_AT = "2026-07-20T10:05:00Z"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FakeSignatureVerifier:
    def __init__(self, *, verifies: bool = True, fingerprint: str | None = None) -> None:
        self.verifies = verifies
        self.observed_fingerprint = fingerprint or collector.EXPECTED_SIGNING_KEY_FINGERPRINT
        self.calls: list[tuple[str, bytes, bytes]] = []

    def fingerprint(self, public_key: str) -> str:
        assert public_key == "ssh-ed25519 AAAATEST release@example.invalid"
        return self.observed_fingerprint

    def verify(self, *, public_key: str, payload: bytes, signature: bytes) -> bool:
        self.calls.append((public_key, payload, signature))
        return self.verifies


class FakeGitHubTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.download_calls: list[str] = []
        self.repository: dict[str, Any] = {
            "full_name": collector.GITHUB_REPOSITORY,
            "private": False,
            "visibility": "public",
        }
        self.ref: dict[str, Any] = {
            "ref": f"refs/tags/{collector.RELEASE_TAG}",
            "object": {"type": "tag", "sha": TAG_OBJECT_SHA},
        }
        self.tag: dict[str, Any] = {
            "sha": TAG_OBJECT_SHA,
            "tag": collector.RELEASE_TAG,
            "object": {"type": "commit", "sha": SOURCE_SHA},
            "verification": {
                "verified": True,
                "reason": "valid",
                "verified_at": "2026-07-20T10:00:00Z",
                "signature": "-----BEGIN SSH SIGNATURE-----\nsynthetic\n",
                "payload": (
                    f"object {SOURCE_SHA}\ntype commit\ntag {collector.RELEASE_TAG}\n"
                    "tagger Release Bot <release@example.invalid> 1784512800 +0800\n\n"
                    "Release candidate\n"
                ),
            },
        }
        self.signing_keys: list[dict[str, Any]] = [
            {
                "id": 71,
                "key": "ssh-ed25519 AAAATEST release@example.invalid",
                "title": "release signing key",
                "created_at": "2026-07-20T09:00:00Z",
            }
        ]
        self.run: dict[str, Any] = {
            "id": RUN_ID,
            "run_attempt": RUN_ATTEMPT,
            "event": "workflow_dispatch",
            "path": collector.WORKFLOW_PATH,
            "head_sha": SOURCE_SHA,
            "status": "completed",
            "conclusion": "success",
            "updated_at": "2026-07-20T10:02:01Z",
        }
        self.jobs: dict[str, Any] = {
            "total_count": 2,
            "jobs": [
                {
                    "id": 501,
                    "name": "quality",
                    "head_sha": SOURCE_SHA,
                    "status": "completed",
                    "conclusion": "success",
                    "completed_at": "2026-07-20T10:01:30Z",
                },
                {
                    "id": 502,
                    "name": "package-smoke",
                    "head_sha": SOURCE_SHA,
                    "status": "completed",
                    "conclusion": "success",
                    "completed_at": "2026-07-20T10:02:00Z",
                },
            ],
        }
        body = (
            "Pre-publication candidate.\n"
            f"Hugging Face repository: {collector.HUGGING_FACE_REPOSITORY}\n"
            f"Immutable model revision: https://huggingface.co/"
            f"{collector.HUGGING_FACE_REPOSITORY}/tree/{HF_SHA}\n"
        )
        wheel_name = collector.contract.REQUIRED_GITHUB_RELEASE_ASSETS["wheel"]
        sbom_name = collector.contract.REQUIRED_GITHUB_RELEASE_ASSETS["sbom"]
        wheel_payload = b"wheel bytes\n"
        sbom_payload = b'{"bomFormat":"CycloneDX"}\n'
        checksums_payload = (
            f"{_digest(wheel_payload)}  {wheel_name}\n"
            f"{_digest(sbom_payload)}  {sbom_name}\n"
        ).encode("ascii")
        asset_payloads = {
            "checksums.txt": checksums_payload,
            wheel_name: wheel_payload,
            sbom_name: sbom_payload,
        }
        self.asset_payloads: dict[int, bytes] = {}
        assets: list[dict[str, Any]] = []
        for asset_id, (name, payload) in enumerate(asset_payloads.items(), start=1001):
            self.asset_payloads[asset_id] = payload
            assets.append(
                {
                    "id": asset_id,
                    "name": name,
                    "size": len(payload),
                    "state": "uploaded",
                    "digest": f"sha256:{_digest(payload)}",
                }
            )
        self.release: dict[str, Any] = {
            "id": RELEASE_ID,
            "tag_name": collector.RELEASE_TAG,
            "target_commitish": SOURCE_SHA,
            "draft": True,
            "prerelease": True,
            "created_at": "2026-07-20T09:59:00Z",
            "updated_at": "2026-07-20T10:04:00Z",
            "body": body,
            "assets": assets,
        }
        card = (
            "# Publication source\n\n"
            f"- `hugging_face_repository: {collector.HUGGING_FACE_REPOSITORY}`\n"
        ).encode()
        self.content: dict[str, Any] = {
            "type": "file",
            "path": collector.GITHUB_MODEL_CARD_PATH,
            "encoding": "base64",
            "content": base64.b64encode(card).decode("ascii"),
            "size": len(card),
        }

    def get_json(
        self, endpoint: str, *, params: dict[str, str] | None = None
    ) -> Any:
        self.calls.append((endpoint, dict(params or {})))
        if endpoint == f"/repos/{collector.GITHUB_REPOSITORY}":
            value: Any = self.repository
        elif endpoint == (
            f"/repos/{collector.GITHUB_REPOSITORY}/git/ref/tags/{collector.RELEASE_TAG}"
        ):
            value = self.ref
        elif endpoint == (
            f"/repos/{collector.GITHUB_REPOSITORY}/git/tags/{TAG_OBJECT_SHA}"
        ):
            value = self.tag
        elif endpoint == f"/users/{collector.GITHUB_OWNER}/ssh_signing_keys":
            value = self.signing_keys
        elif endpoint == f"/repos/{collector.GITHUB_REPOSITORY}/actions/runs/{RUN_ID}":
            value = self.run
        elif endpoint in {
            f"/repos/{collector.GITHUB_REPOSITORY}/actions/runs/{RUN_ID}/jobs",
            (
                f"/repos/{collector.GITHUB_REPOSITORY}/actions/runs/{RUN_ID}/"
                f"attempts/{RUN_ATTEMPT}/jobs"
            ),
        }:
            value = self.jobs
        elif endpoint == f"/repos/{collector.GITHUB_REPOSITORY}/releases/{RELEASE_ID}":
            value = self.release
        elif endpoint == (
            f"/repos/{collector.GITHUB_REPOSITORY}/releases/{RELEASE_ID}/assets"
        ):
            value = self.release["assets"]
        elif endpoint == (
            f"/repos/{collector.GITHUB_REPOSITORY}/contents/"
            f"{collector.GITHUB_MODEL_CARD_PATH}"
        ):
            value = self.content
        else:  # pragma: no cover - makes an unexpected endpoint immediately diagnostic
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        return copy.deepcopy(value)

    def download_digest(self, endpoint: str) -> tuple[str, int]:
        self.download_calls.append(endpoint)
        prefix = f"/repos/{collector.GITHUB_REPOSITORY}/releases/assets/"
        assert endpoint.startswith(prefix)
        asset_id = int(endpoint.removeprefix(prefix))
        payload = self.asset_payloads[asset_id]
        return _digest(payload), len(payload)

    def download_bytes(self, endpoint: str, *, maximum: int) -> bytes:
        self.download_calls.append(endpoint)
        prefix = f"/repos/{collector.GITHUB_REPOSITORY}/releases/assets/"
        assert endpoint.startswith(prefix)
        asset_id = int(endpoint.removeprefix(prefix))
        payload = self.asset_payloads[asset_id]
        assert len(payload) <= maximum
        return payload


def _collect(
    transport: FakeGitHubTransport,
    verifier: FakeSignatureVerifier | None = None,
) -> dict[str, Any]:
    return collector.collect_github_evidence(
        source_commit=SOURCE_SHA,
        workflow_run_id=RUN_ID,
        release_id=RELEASE_ID,
        transport=transport,
        signature_verifier=verifier or FakeSignatureVerifier(),
        collected_at=COLLECTED_AT,
    )


def test_collect_github_evidence_binds_remote_facts_and_local_signature() -> None:
    transport = FakeGitHubTransport()
    verifier = FakeSignatureVerifier()

    document = _collect(transport, verifier)

    assert document["source_commit_sha"] == SOURCE_SHA
    assert document["signed_tag"]["tag_object_sha"] == TAG_OBJECT_SHA
    assert document["signed_tag"]["tag_target_sha"] == SOURCE_SHA
    assert document["signed_tag"]["verification"] == {
        "provider": "github",
        "verified": True,
        "reason": "valid",
        "verified_at": "2026-07-20T10:00:00Z",
        "signature_sha256": _digest(transport.tag["verification"]["signature"].encode()),
        "payload_sha256": _digest(transport.tag["verification"]["payload"].encode()),
        "signing_key_fingerprint": collector.EXPECTED_SIGNING_KEY_FINGERPRINT,
        "local_cryptographic_verification": True,
    }
    assert verifier.calls == [
        (
            transport.signing_keys[0]["key"],
            transport.tag["verification"]["payload"].encode(),
            transport.tag["verification"]["signature"].encode(),
        )
    ]
    assert document["hosted_ci"]["run_attempt"] == RUN_ATTEMPT
    assert document["hosted_ci"]["completed_at"] == "2026-07-20T10:02:00Z"
    assert document["hosted_ci"]["jobs_receipt_sha256"] == collector.canonical_json_hash(
        transport.jobs
    )
    assert (
        (
            f"/repos/{collector.GITHUB_REPOSITORY}/actions/runs/{RUN_ID}/"
            f"attempts/{RUN_ATTEMPT}/jobs"
        ),
        {"per_page": "100"},
    ) in transport.calls
    assert [(item["role"], item["name"]) for item in document["release"]["assets"]] == [
        ("checksums", "checksums.txt"),
        ("wheel", "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"),
        ("sbom", "sbom.cdx.json"),
    ]
    assets_by_name = {
        item["name"]: item for item in document["release"]["assets"]
    }
    assert document["release"]["optional_asset_names"] == []
    assert document["release"]["checksums"] == {
        "asset_name": "checksums.txt",
        "file_sha256": assets_by_name["checksums.txt"]["downloaded_sha256"],
        "format": "sha256sum_text_two_space_two_lines_lf",
        "line_count": 2,
        "entries": [
            {
                "name": "pii_zh_qwen-0.2.0rc1-py3-none-any.whl",
                "sha256": assets_by_name[
                    "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
                ]["downloaded_sha256"],
            },
            {
                "name": "sbom.cdx.json",
                "sha256": assets_by_name["sbom.cdx.json"]["downloaded_sha256"],
            },
        ],
        "matched_downloaded_assets": True,
    }
    assert document["cross_reference"]["hugging_face_repository"] == (
        collector.HUGGING_FACE_REPOSITORY
    )
    assert document["evidence_sha256"] == collector.canonical_json_hash(
        document, remove="evidence_sha256"
    )
    assert len(transport.download_calls) == 3


@pytest.mark.parametrize(
    ("mutation", "blocker_id"),
    [
        (lambda value: value.ref["object"].update(type="commit"), "TAG_NOT_ANNOTATED"),
        (
            lambda value: value.tag["object"].update(sha="d" * 40),
            "TAG_SOURCE_MISMATCH",
        ),
    ],
)
def test_collect_github_evidence_rejects_invalid_tag(mutation: Any, blocker_id: str) -> None:
    transport = FakeGitHubTransport()
    mutation(transport)

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == blocker_id


def test_collect_github_evidence_rejects_wrong_ci_result() -> None:
    transport = FakeGitHubTransport()
    transport.run["conclusion"] = "failure"

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "HOSTED_CI_MISMATCH"


def test_collect_github_evidence_rejects_asset_download_mismatch() -> None:
    transport = FakeGitHubTransport()
    asset_id = transport.release["assets"][0]["id"]
    transport.asset_payloads[asset_id] = b"different bytes"

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "GITHUB_ASSET_BYTES_MISMATCH"


def test_collect_github_evidence_rejects_any_extra_release_asset() -> None:
    transport = FakeGitHubTransport()
    payload = b"model weights must never be attached\n"
    asset_id = 1004
    transport.asset_payloads[asset_id] = payload
    transport.release["assets"].append(
        {
            "id": asset_id,
            "name": "model.safetensors",
            "size": len(payload),
            "state": "uploaded",
            "digest": f"sha256:{_digest(payload)}",
        }
    )

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "GITHUB_ASSET_SET_MISMATCH"
    assert transport.download_calls == []


@pytest.mark.parametrize(
    "payload",
    [
        b"not a checksum\n",
        (
            b"0" * 64
            + b"  pii_zh_qwen-0.2.0rc1-py3-none-any.whl\r\n"
            + b"0" * 64
            + b"  sbom.cdx.json\r\n"
        ),
        (
            b"0" * 64
            + b"  sbom.cdx.json\n"
            + b"0" * 64
            + b"  pii_zh_qwen-0.2.0rc1-py3-none-any.whl\n"
        ),
    ],
)
def test_collect_github_evidence_rejects_noncanonical_checksums(payload: bytes) -> None:
    transport = FakeGitHubTransport()
    checksums = next(
        item for item in transport.release["assets"] if item["name"] == "checksums.txt"
    )
    asset_id = checksums["id"]
    transport.asset_payloads[asset_id] = payload
    checksums["size"] = len(payload)
    checksums["digest"] = f"sha256:{_digest(payload)}"

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "GITHUB_CHECKSUMS_FORMAT_INVALID"


def test_collect_github_evidence_rejects_checksums_for_other_bytes() -> None:
    transport = FakeGitHubTransport()
    wheel_name = collector.contract.REQUIRED_GITHUB_RELEASE_ASSETS["wheel"]
    sbom_name = collector.contract.REQUIRED_GITHUB_RELEASE_ASSETS["sbom"]
    payload = f"{'0' * 64}  {wheel_name}\n{'1' * 64}  {sbom_name}\n".encode("ascii")
    checksums = next(
        item for item in transport.release["assets"] if item["name"] == "checksums.txt"
    )
    asset_id = checksums["id"]
    transport.asset_payloads[asset_id] = payload
    checksums["size"] = len(payload)
    checksums["digest"] = f"sha256:{_digest(payload)}"

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "GITHUB_CHECKSUMS_DIGEST_MISMATCH"


def test_collect_github_evidence_rejects_failed_local_signature_verification() -> None:
    transport = FakeGitHubTransport()

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport, FakeSignatureVerifier(verifies=False))

    assert caught.value.blocker_id == "LOCAL_TAG_SIGNATURE_INVALID"


def test_collect_github_evidence_rejects_signature_payload_for_another_target() -> None:
    transport = FakeGitHubTransport()
    transport.tag["verification"]["payload"] = transport.tag["verification"][
        "payload"
    ].replace(SOURCE_SHA, "d" * 40, 1)

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "GITHUB_TAG_PAYLOAD_MISMATCH"


def test_collect_github_evidence_requires_each_fixed_ci_job_to_succeed() -> None:
    transport = FakeGitHubTransport()
    transport.jobs["jobs"][1]["conclusion"] = "skipped"

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(transport)

    assert caught.value.blocker_id == "HOSTED_CI_JOB_FAILED"


def test_write_evidence_is_read_only_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "github-evidence.json"
    document = {"synthetic": True}

    expected_payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    assert collector._write_evidence(output, document) == _digest(expected_payload)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    before = output.read_bytes()

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        collector._write_evidence(output, {"synthetic": False})

    assert caught.value.blocker_id == "OUTPUT_EXISTS"
    assert output.read_bytes() == before
