from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts import build_community_v2_publication_receipt as publication
from scripts import collect_community_v2_remote_evidence as collector
from scripts import materialize_community_v2_hf_snapshot as hf_provenance

HF_SHA = "b" * 40
SOURCE_SHA = "a" * 40
COLLECTED_AT = "2026-07-20T11:30:00Z"


def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seal(document: dict[str, Any], *, key: str = "receipt_sha256") -> dict[str, Any]:
    document[key] = collector.canonical_json_hash(document, remove=key)
    return document


def _write_json(path: Path, document: dict[str, Any], *, mode: int = 0o444) -> bytes:
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    path.chmod(mode)
    return payload


def _file_payloads() -> dict[str, bytes]:
    readme = (
        "# PII Chinese model\n\n"
        f"github_repository: {collector.GITHUB_REPOSITORY}\n"
        f"git_source_commit: {SOURCE_SHA}\n"
        f"release_tag: {collector.RELEASE_TAG}\n"
    ).encode()
    result = {
        path: f"offline fixture for {path}\n".encode()
        for path in publication.REQUIRED_HUGGING_FACE_FILES
    }
    result[collector.HF_MODEL_CARD_PATH] = readme
    return result


def _provenance_document() -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads = _file_payloads()
    local_files = {
        path: {"file_sha256": _sha256(payload), "size_bytes": len(payload)}
        for path, payload in sorted(payloads.items())
    }
    remote_files = {
        path: {
            "size_bytes": len(payload),
            "git_blob_oid": f"{index + 1:040x}",
            "lfs_oid_sha256": None,
            "content_verification": "git_blob_sha1",
        }
        for index, (path, payload) in enumerate(sorted(payloads.items()))
    }
    generator = collector.REPOSITORY_ROOT / hf_provenance.GENERATOR_PATH
    document = {
        "schema_version": hf_provenance.SCHEMA_VERSION,
        "provider": "huggingface_hub",
        "endpoint": collector.OFFICIAL_HF_ENDPOINT,
        "repository": collector.HUGGING_FACE_REPOSITORY,
        "requested_revision": HF_SHA,
        "resolved_commit": HF_SHA,
        "downloaded_at": "2026-07-20T11:00:00Z",
        "remote_snapshot": {
            "private": True,
            "visibility": "private",
            "file_count": len(remote_files),
            "metadata_inventory_sha256": collector.canonical_json_hash(remote_files),
            "metadata_coverage": {
                "size_count": len(remote_files),
                "git_blob_oid_count": len(remote_files),
                "lfs_oid_count": 0,
                "content_verified_count": len(remote_files),
            },
            "files": remote_files,
            "evidence_boundary": hf_provenance.REMOTE_EVIDENCE_BOUNDARY,
        },
        "generator": {
            "path": hf_provenance.GENERATOR_PATH.as_posix(),
            "file_sha256": _sha256(generator.read_bytes()),
        },
        "local_root": {
            "format": "materialized_regular_file_tree_v1",
            "file_count": len(local_files),
            "inventory_sha256": collector.canonical_json_hash(local_files),
            "files": local_files,
        },
        "receipt_sha256": "",
    }
    return _seal(document), payloads


def _package_verification_document(
    provenance: dict[str, Any],
    provenance_file_sha256: str,
) -> dict[str, Any]:
    local_files = provenance["local_root"]["files"]
    remote_snapshot = provenance["remote_snapshot"]
    provenance_binding = {
        "file_sha256": provenance_file_sha256,
        "receipt_sha256": provenance["receipt_sha256"],
        "repository": collector.HUGGING_FACE_REPOSITORY,
        "requested_revision": HF_SHA,
        "resolved_commit": HF_SHA,
        "remote_visibility": "private",
        "remote_file_count": len(local_files),
        "remote_metadata_inventory_sha256": remote_snapshot[
            "metadata_inventory_sha256"
        ],
        "remote_content_verified_count": len(local_files),
        "local_root_inventory_sha256": provenance["local_root"]["inventory_sha256"],
        "generator_file_sha256": provenance["generator"]["file_sha256"],
    }
    document = {
        "schema_version": collector.PACKAGE_VERIFICATION_SCHEMA_VERSION,
        "context": "hugging_face_immutable_download",
        "package_version": publication.PACKAGE_VERSION,
        "target": {
            "github_repository": collector.GITHUB_REPOSITORY,
            "hugging_face_repository": collector.HUGGING_FACE_REPOSITORY,
            "hugging_face_commit": HF_SHA,
        },
        "source_control": {"git_source_commit": SOURCE_SHA},
        "hf_download_provenance": provenance_binding,
        "package_identity": {
            "manifest_sha256": _sha256("manifest"),
            "checksums_file_sha256": local_files["checksums.txt"]["file_sha256"],
            "payload_inventory_sha256": _sha256("payload inventory"),
            "model_file_sha256": local_files["model.safetensors"]["file_sha256"],
            "verified_file_count": len(local_files),
        },
        "checks": {
            "checksum_closure": True,
            "manifest_schema_and_self_hash": True,
            "target_and_source_binding": True,
            "hf_download_provenance_binding": True,
            "forbidden_file_absence": True,
            "remote_code_contract": True,
            "public_artifact_scan": True,
            "post_smoke_reverification": True,
            "offline_model_load": True,
            "finite_forward": True,
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.7.0",
            "transformers": "4.51.0",
            "network_disabled": True,
            "cuda_visible_devices": "",
        },
        "status": "PASS",
        "verified_at": "2026-07-20T11:15:00Z",
        "receipt_sha256": "",
    }
    return _seal(document)


class FakeOfficialHuggingFaceTransport:
    """Official-client-shaped read-only fake that refuses non-README downloads."""

    def __init__(self, provenance: dict[str, Any], payloads: dict[str, bytes]) -> None:
        self.repo_info_calls: list[str] = []
        self.list_calls: list[str] = []
        self.read_calls: list[tuple[str, str]] = []
        remote_files = provenance["remote_snapshot"]["files"]
        self.paths = sorted(remote_files)
        self.info: dict[str, Any] = {
            "id": collector.HUGGING_FACE_REPOSITORY,
            "sha": HF_SHA,
            "private": True,
            "siblings": [
                {
                    "rfilename": path,
                    "size": remote_files[path]["size_bytes"],
                    "blob_id": remote_files[path]["git_blob_oid"],
                    "lfs": None,
                }
                for path in self.paths
            ],
        }
        self.readme = payloads[collector.HF_MODEL_CARD_PATH]

    def repo_info(self, *, revision: str) -> object:
        self.repo_info_calls.append(revision)
        return copy.deepcopy(self.info)

    def list_repo_files(self, *, revision: str) -> list[str]:
        self.list_calls.append(revision)
        return list(self.paths)

    def read_file(self, *, revision: str, path: str) -> bytes:
        self.read_calls.append((revision, path))
        if path != collector.HF_MODEL_CARD_PATH:
            raise AssertionError("collector attempted to download a non-README model file")
        return self.readme


def _fixture(
    tmp_path: Path,
    *,
    provenance_mode: int = 0o444,
    verification_mode: int = 0o444,
    mutate_verification: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[
    collector.JsonPayload,
    collector.JsonPayload,
    FakeOfficialHuggingFaceTransport,
]:
    provenance, payloads = _provenance_document()
    provenance_path = tmp_path / "hf-download-provenance.json"
    provenance_bytes = _write_json(
        provenance_path, provenance, mode=provenance_mode
    )
    verification = _package_verification_document(
        provenance, _sha256(provenance_bytes)
    )
    if mutate_verification is not None:
        mutate_verification(verification)
        _seal(verification)
    verification_path = tmp_path / "hf-package-verification.json"
    _write_json(verification_path, verification, mode=verification_mode)
    loaded_provenance = collector.load_json_payload(
        provenance_path, field="test HF provenance"
    )
    loaded_verification = collector.load_json_payload(
        verification_path, field="test package verification"
    )
    return (
        loaded_provenance,
        loaded_verification,
        FakeOfficialHuggingFaceTransport(provenance, payloads),
    )


def _collect(
    provenance: collector.JsonPayload,
    verification: collector.JsonPayload,
    transport: FakeOfficialHuggingFaceTransport,
) -> dict[str, Any]:
    return collector.collect_hugging_face_evidence(
        download_provenance=provenance,
        package_verification=verification,
        transport=transport,
        collected_at=COLLECTED_AT,
    )


def test_collect_hf_evidence_rechecks_private_immutable_state_without_weights(
    tmp_path: Path,
) -> None:
    provenance, verification, transport = _fixture(tmp_path)

    document = _collect(provenance, verification, transport)

    expected_paths = sorted(publication.REQUIRED_HUGGING_FACE_FILES)
    assert provenance.mode == verification.mode == 0o444
    assert [item["path"] for item in document["inventory"]] == expected_paths
    assert len(document["inventory"]) == 13
    assert document["repository"] == {
        "id": collector.HUGGING_FACE_REPOSITORY,
        "visibility": "private",
    }
    assert document["revision"] == {
        "requested_revision": HF_SHA,
        "resolved_sha": HF_SHA,
        "immutable_sha": HF_SHA,
    }
    assert document["cross_reference"] == {
        "path": collector.HF_MODEL_CARD_PATH,
        "file_sha256": provenance.document["local_root"]["files"]["README.md"][
            "file_sha256"
        ],
        "github_repository": collector.GITHUB_REPOSITORY,
        "github_source_sha": SOURCE_SHA,
    }
    assert document["package_verification"]["file_sha256"] == (
        verification.file_sha256
    )
    assert document["package_verification"][
        "hf_download_provenance_file_sha256"
    ] == provenance.file_sha256
    assert document["evidence_sha256"] == collector.canonical_json_hash(
        document, remove="evidence_sha256"
    )
    assert transport.repo_info_calls == [HF_SHA, HF_SHA]
    assert transport.list_calls == [HF_SHA, HF_SHA]
    assert transport.read_calls == [(HF_SHA, collector.HF_MODEL_CARD_PATH)]
    assert all(path != "model.safetensors" for _revision, path in transport.read_calls)


@pytest.mark.parametrize(
    ("mutate", "blocker_id"),
    [
        (lambda value: value.info.__setitem__("private", False), "HF_REMOTE_STATE_MISMATCH"),
        (lambda value: value.info.__setitem__("sha", "c" * 40), "HF_REMOTE_STATE_MISMATCH"),
    ],
)
def test_collect_hf_evidence_rejects_visibility_or_commit_drift(
    tmp_path: Path,
    mutate: Callable[[FakeOfficialHuggingFaceTransport], None],
    blocker_id: str,
) -> None:
    provenance, verification, transport = _fixture(tmp_path)
    mutate(transport)

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(provenance, verification, transport)

    assert caught.value.blocker_id == blocker_id


def test_collect_hf_evidence_rejects_inventory_drift(tmp_path: Path) -> None:
    provenance, verification, transport = _fixture(tmp_path)
    transport.paths.append("unexpected.txt")

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(provenance, verification, transport)

    assert caught.value.blocker_id == "HF_PROVENANCE_INVENTORY_MISMATCH"


def test_collect_hf_evidence_rejects_readme_drift(tmp_path: Path) -> None:
    provenance, verification, transport = _fixture(tmp_path)
    transport.readme += b"drift\n"

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(provenance, verification, transport)

    assert caught.value.blocker_id == "HF_README_HASH_MISMATCH"
    assert transport.read_calls == [(HF_SHA, collector.HF_MODEL_CARD_PATH)]


def test_collect_hf_evidence_rejects_provenance_binding_drift(tmp_path: Path) -> None:
    def drift(document: dict[str, Any]) -> None:
        document["hf_download_provenance"]["file_sha256"] = "f" * 64

    provenance, verification, transport = _fixture(
        tmp_path, mutate_verification=drift
    )

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(provenance, verification, transport)

    assert caught.value.blocker_id == "PACKAGE_PROVENANCE_BINDING_MISMATCH"
    assert transport.repo_info_calls == []


@pytest.mark.parametrize("which", ["provenance", "verification"])
def test_collect_hf_evidence_requires_immutable_input_modes(
    tmp_path: Path, which: str
) -> None:
    provenance, verification, transport = _fixture(
        tmp_path,
        provenance_mode=0o644 if which == "provenance" else 0o444,
        verification_mode=0o644 if which == "verification" else 0o444,
    )
    assert stat.S_IMODE(os.lstat(provenance.path).st_mode) == provenance.mode
    assert stat.S_IMODE(os.lstat(verification.path).st_mode) == verification.mode

    with pytest.raises(collector.RemoteEvidenceError) as caught:
        _collect(provenance, verification, transport)

    assert caught.value.blocker_id == "INPUT_MODE_INVALID"
    assert transport.repo_info_calls == []
