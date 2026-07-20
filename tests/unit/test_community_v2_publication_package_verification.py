from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from scripts import build_community_v2_publication_successor as successor
from scripts import materialize_community_v2_hf_snapshot as hf_provenance
from scripts import verify_community_v2_publication_package as verification
from tests.unit.test_community_v2_publication_successor import (
    GIT_SOURCE_COMMIT,
    GITHUB_REPOSITORY,
    HUGGING_FACE_REPOSITORY,
    _fixture_inputs,
    _prepare,
    _write_checksums,
)


def _build_fixture(tmp_path: Path) -> Path:
    inputs = _fixture_inputs(tmp_path)
    config_path = inputs["source"] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "auto_map": verification.EXPECTED_AUTO_MAP,
            "id2label": verification.EXPECTED_ID2LABEL,
            "label2id": verification.EXPECTED_LABEL2ID,
            "pii_attention_mode": "full",
        }
    )
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    (inputs["source"] / "id2label.json").write_text(
        json.dumps(verification.EXPECTED_ID2LABEL, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums(inputs["source"])
    plan = _prepare(tmp_path, inputs)
    successor.build_publication_successor(plan)
    return plan.output


def _smoke_result() -> dict[str, object]:
    return {
        "status": "PASS",
        "offline_model_load": True,
        "finite_forward": True,
        "logit_shape": [1, 5, 49],
        "python": "3.12.0",
        "torch": "2.7.0",
        "transformers": "4.51.0",
    }


def _write_hf_provenance(
    tmp_path: Path,
    package: Path,
    *,
    repository: str = HUGGING_FACE_REPOSITORY,
    revision: str = "b" * 40,
) -> Path:
    inventory = hf_provenance._inventory(package)
    remote_files: dict[str, dict[str, object]] = {}
    for relative, local_binding in inventory.items():
        path = package / relative
        size = int(local_binding["size_bytes"])
        if relative == "model.safetensors":
            git_blob_oid = None
            lfs_oid_sha256 = local_binding["file_sha256"]
            method = "lfs_sha256"
        else:
            git_blob_oid = hf_provenance._git_blob_sha1(
                path,
                expected_size=size,
                field=f"fixture remote file {relative}",
            )
            lfs_oid_sha256 = None
            method = "git_blob_sha1"
        remote_files[relative] = {
            "size_bytes": size,
            "git_blob_oid": git_blob_oid,
            "lfs_oid_sha256": lfs_oid_sha256,
            "content_verification": method,
        }
    document: dict[str, object] = {
        "schema_version": hf_provenance.SCHEMA_VERSION,
        "provider": "huggingface_hub",
        "endpoint": hf_provenance.OFFICIAL_ENDPOINT,
        "repository": repository,
        "requested_revision": revision,
        "resolved_commit": revision,
        "downloaded_at": "2026-07-20T04:55:00Z",
        "remote_snapshot": {
            "private": False,
            "visibility": "public",
            "file_count": len(remote_files),
            "metadata_inventory_sha256": successor.canonical_json_hash(remote_files),
            "metadata_coverage": {
                "size_count": len(remote_files),
                "git_blob_oid_count": len(remote_files) - 1,
                "lfs_oid_count": 1,
                "content_verified_count": len(remote_files),
            },
            "files": remote_files,
            "evidence_boundary": hf_provenance.REMOTE_EVIDENCE_BOUNDARY,
        },
        "generator": {
            "path": hf_provenance.GENERATOR_PATH.as_posix(),
            "file_sha256": hf_provenance._generator_sha256(),
        },
        "local_root": {
            "format": "materialized_regular_file_tree_v1",
            "file_count": len(inventory),
            "inventory_sha256": successor.canonical_json_hash(inventory),
            "files": inventory,
        },
        "receipt_sha256": "",
    }
    document["receipt_sha256"] = successor.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    path = tmp_path / "hf-download-provenance.json"
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_schema_is_valid() -> None:
    schema = json.loads(verification.RECEIPT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_sha256_regular_hashes_without_loading_the_whole_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    payload = b"publication-verification" * 1024
    artifact.write_bytes(payload)

    assert verification._sha256_regular(
        artifact, field="test artifact", maximum=len(payload)
    ) == hashlib.sha256(payload).hexdigest()


def test_sha256_regular_refuses_a_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"publication-verification")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(artifact)

    with pytest.raises(verification.PublicationVerificationError, match="missing"):
        verification._sha256_regular(linked, field="test artifact", maximum=1024)


def test_offline_smoke_environment_does_not_inherit_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-propagate")
    monkeypatch.setenv("ARK_API_KEY", "must-not-propagate")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-propagate")
    monkeypatch.setenv("HTTPS_PROXY", "must-not-propagate")

    outer_environment = verification._offline_environment(tmp_path)
    sandbox_environment = verification._sandbox_environment()

    assert "must-not-propagate" not in outer_environment.values()
    assert "must-not-propagate" not in sandbox_environment.values()
    assert sandbox_environment["CUDA_VISIBLE_DEVICES"] == ""
    assert sandbox_environment["HF_HUB_OFFLINE"] == "1"
    assert sandbox_environment["TRANSFORMERS_OFFLINE"] == "1"
    assert sandbox_environment["OMP_NUM_THREADS"] == "1"


def test_model_smoke_uses_read_only_sandbox_and_resource_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 12345
        returncode = 0

        def communicate(self, timeout: int) -> tuple[bytes, bytes]:
            observed["timeout"] = timeout
            return json.dumps(_smoke_result(), sort_keys=True).encode(), b""

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(verification.subprocess, "Popen", fake_popen)
    result = verification._run_model_smoke(tmp_path)

    command = observed["command"]
    assert isinstance(command, list)
    assert "MemoryMax=16G" in command
    assert "TasksMax=64" in command
    assert "CPUQuota=100%" in command
    assert "--unshare-all" in command
    assert "--die-with-parent" in command
    assert "--clearenv" in command
    assert "/model" in command
    assert observed["timeout"] == 600
    assert result["status"] == "PASS"


def test_verifies_local_successor_and_writes_read_only_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    monkeypatch.setattr(verification, "_run_model_smoke", lambda _root: _smoke_result())
    monkeypatch.setattr(verification.scan_public_artifacts, "scan_paths", lambda _paths: [])

    document = verification.verify_package(
        package_root=package,
        context="local_publication_successor",
        source_commit=GIT_SOURCE_COMMIT,
        github_repository=GITHUB_REPOSITORY,
        hugging_face_repository=HUGGING_FACE_REPOSITORY,
        hugging_face_commit=None,
        verified_at="2026-07-20T05:00:00Z",
    )
    assert document["status"] == "PASS"
    assert document["target"]["hugging_face_commit"] is None
    assert document["hf_download_provenance"] is None
    assert document["checks"] == {
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
    }

    output = tmp_path / "verification-receipt.json"
    verification._write_new_receipt(output, document)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert verification.validate_receipt(output) == document
    with pytest.raises(verification.PublicationVerificationError, match="already exists"):
        verification._write_new_receipt(output, document)


def test_receipt_writer_rejects_symlink_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        verification.PublicationVerificationError, match="parent must be a real directory"
    ):
        verification._write_new_receipt(
            linked_parent / "verification-receipt.json", {"status": "synthetic"}
        )
    assert not (real_parent / "verification-receipt.json").exists()


def test_hf_download_context_requires_immutable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    monkeypatch.setattr(verification, "_run_model_smoke", lambda _root: _smoke_result())
    monkeypatch.setattr(verification.scan_public_artifacts, "scan_paths", lambda _paths: [])

    with pytest.raises(
        verification.PublicationVerificationError, match="requires a full lowercase commit"
    ):
        verification.verify_package(
            package_root=package,
            context="hugging_face_immutable_download",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit=None,
        )


def test_hf_download_context_requires_reviewed_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    monkeypatch.setattr(
        verification,
        "_run_model_smoke",
        lambda _root: pytest.fail("model smoke must not run without provenance"),
    )

    with pytest.raises(
        verification.PublicationVerificationError,
        match="requires reviewed download provenance",
    ):
        verification.verify_package(
            package_root=package,
            context="hugging_face_immutable_download",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit="b" * 40,
        )


def test_verifies_hf_download_only_with_matching_self_hashed_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    provenance = _write_hf_provenance(tmp_path, package)
    scan_calls = 0

    def scan(_paths: object) -> list[object]:
        nonlocal scan_calls
        scan_calls += 1
        return []

    monkeypatch.setattr(verification, "_run_model_smoke", lambda _root: _smoke_result())
    monkeypatch.setattr(verification.scan_public_artifacts, "scan_paths", scan)

    document = verification.verify_package(
        package_root=package,
        context="hugging_face_immutable_download",
        source_commit=GIT_SOURCE_COMMIT,
        github_repository=GITHUB_REPOSITORY,
        hugging_face_repository=HUGGING_FACE_REPOSITORY,
        hugging_face_commit="b" * 40,
        hugging_face_download_provenance=provenance,
        verified_at="2026-07-20T05:00:00Z",
    )

    assert scan_calls == 2
    assert document["hf_download_provenance"]["repository"] == HUGGING_FACE_REPOSITORY
    assert document["hf_download_provenance"]["resolved_commit"] == "b" * 40
    assert document["hf_download_provenance"]["remote_file_count"] == document[
        "hf_download_provenance"
    ]["remote_content_verified_count"]


def test_rejects_hf_provenance_without_complete_remote_content_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    provenance = _write_hf_provenance(tmp_path, package)
    document = json.loads(provenance.read_text(encoding="utf-8"))
    remote = document["remote_snapshot"]
    binding = remote["files"]["README.md"]
    binding["git_blob_oid"] = None
    binding["content_verification"] = "immutable_revision_transport_only"
    remote["metadata_coverage"]["git_blob_oid_count"] -= 1
    remote["metadata_coverage"]["content_verified_count"] -= 1
    remote["metadata_inventory_sha256"] = successor.canonical_json_hash(
        remote["files"]
    )
    document["receipt_sha256"] = successor.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    provenance.write_text(json.dumps(document) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        verification,
        "_run_model_smoke",
        lambda _root: pytest.fail("model smoke must not run for weak provenance"),
    )

    with pytest.raises(
        verification.PublicationVerificationError,
        match="lacks complete remote content verification",
    ):
        verification.verify_package(
            package_root=package,
            context="hugging_face_immutable_download",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit="b" * 40,
            hugging_face_download_provenance=provenance,
        )


def test_rejects_hf_provenance_for_a_different_revision_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    provenance = _write_hf_provenance(tmp_path, package, revision="c" * 40)
    monkeypatch.setattr(
        verification,
        "_run_model_smoke",
        lambda _root: pytest.fail("model smoke must not run after provenance drift"),
    )

    with pytest.raises(
        verification.PublicationVerificationError,
        match="repository or revision does not match",
    ):
        verification.verify_package(
            package_root=package,
            context="hugging_face_immutable_download",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit="b" * 40,
            hugging_face_download_provenance=provenance,
        )


def test_rejects_self_consistent_provenance_inventory_that_does_not_bind_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    provenance = _write_hf_provenance(tmp_path, package)
    document = json.loads(provenance.read_text(encoding="utf-8"))
    files = document["local_root"]["files"]
    files["README.md"]["file_sha256"] = "f" * 64
    document["local_root"]["inventory_sha256"] = successor.canonical_json_hash(files)
    document["receipt_sha256"] = successor.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    provenance.write_text(json.dumps(document) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        verification,
        "_run_model_smoke",
        lambda _root: pytest.fail("model smoke must not run after inventory drift"),
    )

    with pytest.raises(
        verification.PublicationVerificationError,
        match="does not bind the package",
    ):
        verification.verify_package(
            package_root=package,
            context="hugging_face_immutable_download",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit="b" * 40,
            hugging_face_download_provenance=provenance,
        )


def test_rejects_target_drift_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    monkeypatch.setattr(
        verification,
        "_run_model_smoke",
        lambda _root: pytest.fail("model smoke must not run after target drift"),
    )
    monkeypatch.setattr(verification.scan_public_artifacts, "scan_paths", lambda _paths: [])

    with pytest.raises(verification.PublicationVerificationError, match="target does not match"):
        verification.verify_package(
            package_root=package,
            context="local_publication_successor",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository="other/model",
            hugging_face_commit=None,
        )


def test_rejects_symlink_package_root_before_resolving_or_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    linked = tmp_path / "linked-package"
    linked.symlink_to(package, target_is_directory=True)
    monkeypatch.setattr(
        verification,
        "_run_model_smoke",
        lambda _root: pytest.fail("model smoke must not run through a symlink root"),
    )

    with pytest.raises(
        verification.PublicationVerificationError,
        match="non-symlink directory",
    ):
        verification.verify_package(
            package_root=linked,
            context="local_publication_successor",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit=None,
        )


def test_rejects_package_mutation_during_model_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)

    def mutating_smoke(root: Path) -> dict[str, object]:
        (root / "README.md").write_text("changed during smoke\n", encoding="utf-8")
        return _smoke_result()

    monkeypatch.setattr(verification, "_run_model_smoke", mutating_smoke)
    monkeypatch.setattr(verification.scan_public_artifacts, "scan_paths", lambda _paths: [])

    with pytest.raises(
        verification.PublicationVerificationError,
        match="post-smoke closure verification",
    ):
        verification.verify_package(
            package_root=package,
            context="local_publication_successor",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit=None,
        )


def test_remote_code_contract_locks_all_49_label_values(tmp_path: Path) -> None:
    package = _build_fixture(tmp_path)
    config_path = package / "config.json"
    labels_path = package / "id2label.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    old_label = labels["48"]
    fake_label = "I-UNREVIEWED_EXTRA"
    labels["48"] = fake_label
    config["id2label"]["48"] = fake_label
    del config["label2id"][old_label]
    config["label2id"][fake_label] = 48
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps(labels) + "\n", encoding="utf-8")

    with pytest.raises(
        verification.PublicationVerificationError,
        match="reviewed core-24 BIO contract",
    ):
        verification._verify_remote_code_contract(package)


def test_remote_code_contract_requires_config_mappings_to_match_id2label_file(
    tmp_path: Path,
) -> None:
    package = _build_fixture(tmp_path)
    config_path = package / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"]["1"] = "B-WRONG"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(
        verification.PublicationVerificationError,
        match="config id2label differs",
    ):
        verification._verify_remote_code_contract(package)


def test_validate_receipt_rejects_self_hash_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _build_fixture(tmp_path)
    monkeypatch.setattr(verification, "_run_model_smoke", lambda _root: _smoke_result())
    monkeypatch.setattr(verification.scan_public_artifacts, "scan_paths", lambda _paths: [])
    document = dict(
        verification.verify_package(
            package_root=package,
            context="local_publication_successor",
            source_commit=GIT_SOURCE_COMMIT,
            github_repository=GITHUB_REPOSITORY,
            hugging_face_repository=HUGGING_FACE_REPOSITORY,
            hugging_face_commit=None,
            verified_at="2026-07-20T05:00:00Z",
        )
    )
    document["status"] = "BLOCKED"
    receipt = tmp_path / "tampered.json"
    receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(verification.PublicationVerificationError):
        verification.validate_receipt(receipt)
