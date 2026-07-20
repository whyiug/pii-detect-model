from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import build_community_v2_publication_receipt as publication
from scripts import materialize_community_v2_hf_snapshot as hf_provenance

SOURCE_SHA = "a" * 40
TAG_OBJECT_SHA = "b" * 40
HF_SHA = "c" * 40
RECORDED_AT = "2026-07-20T10:07:00Z"
HF_REVISION_URL = (
    f"https://huggingface.co/{publication.EXPECTED_HUGGING_FACE_REPOSITORY}/tree/{HF_SHA}"
)
SIGNING_KEY_FINGERPRINT = f"SHA256:{'A' * 43}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal(document: dict[str, Any], *, key: str = "evidence_sha256") -> dict[str, Any]:
    sealed = copy.deepcopy(document)
    sealed[key] = ""
    sealed[key] = publication.canonical_json_hash(sealed, remove=key)
    return sealed


def _write_json(path: Path, document: dict[str, Any], *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _collector_generator_sha256() -> str:
    path = publication.REPOSITORY_ROOT / publication.REMOTE_EVIDENCE_COLLECTOR_GENERATOR
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_asset(
    *, asset_id: int, role: str, name: str, content: str
) -> dict[str, Any]:
    digest = _sha256(content)
    return {
        "asset_id": asset_id,
        "role": role,
        "name": name,
        "size_bytes": len(content.encode("utf-8")),
        "state": "uploaded",
        "github_digest_sha256": digest,
        "downloaded_sha256": digest,
    }


def _github_document() -> dict[str, Any]:
    body = (
        "Pre-publication candidate.\n"
        f"Hugging Face repository: {publication.EXPECTED_HUGGING_FACE_REPOSITORY}\n"
        f"Immutable model revision: {HF_REVISION_URL}\n"
    )
    wheel_name = publication.REQUIRED_GITHUB_RELEASE_ASSETS["wheel"]
    sbom_name = publication.REQUIRED_GITHUB_RELEASE_ASSETS["sbom"]
    wheel_content = "wheel bytes"
    sbom_content = "sbom bytes"
    checksums_content = (
        f"{_sha256(wheel_content)}  {wheel_name}\n"
        f"{_sha256(sbom_content)}  {sbom_name}\n"
    )
    assets = sorted(
        [
            _release_asset(
                asset_id=1001,
                role="checksums",
                name="checksums.txt",
                content=checksums_content,
            ),
            _release_asset(
                asset_id=1002,
                role="wheel",
                name=wheel_name,
                content=wheel_content,
            ),
            _release_asset(
                asset_id=1003,
                role="sbom",
                name=sbom_name,
                content=sbom_content,
            ),
        ],
        key=lambda item: item["name"],
    )
    return _seal(
        {
            "schema_version": publication.GITHUB_EVIDENCE_SCHEMA_VERSION,
            "collector": {
                "name": "github-release-readonly-collector",
                "version": "1.0.0",
                "collected_at": "2026-07-20T10:05:00Z",
                "endpoint": "https://api.github.com",
                "network_accessed": True,
                "remote_mutation_performed": False,
                "generator_path": publication.REMOTE_EVIDENCE_COLLECTOR_GENERATOR.as_posix(),
                "generator_file_sha256": _collector_generator_sha256(),
                "collection_run_sha256": _sha256("github collector run"),
            },
            "repository": {
                "id": publication.EXPECTED_GITHUB_REPOSITORY,
                "visibility": "public",
            },
            "source_commit_sha": SOURCE_SHA,
            "signed_tag": {
                "name": publication.TAG_NAME,
                "ref_target_type": "tag",
                "ref_target_sha": TAG_OBJECT_SHA,
                "tag_object_sha": TAG_OBJECT_SHA,
                "tag_target_type": "commit",
                "tag_target_sha": SOURCE_SHA,
                "verification": {
                    "provider": "github",
                    "verified": True,
                    "reason": "valid",
                    "verified_at": "2026-07-20T10:00:00Z",
                    "signature_sha256": _sha256("tag signature"),
                    "payload_sha256": _sha256("tag payload"),
                    "signing_key_fingerprint": SIGNING_KEY_FINGERPRINT,
                    "local_cryptographic_verification": True,
                },
            },
            "hosted_ci": {
                "provider": "github_actions",
                "workflow_path": ".github/workflows/ci.yml",
                "run_id": 98765,
                "run_attempt": 1,
                "head_sha": SOURCE_SHA,
                "status": "completed",
                "conclusion": "success",
                "completed_at": "2026-07-20T10:02:00Z",
                "jobs_receipt_sha256": _sha256("hosted CI jobs"),
            },
            "release": {
                "release_id": 12345,
                "tag_name": publication.TAG_NAME,
                "resolved_source_sha": SOURCE_SHA,
                "draft": True,
                "prerelease": True,
                "created_at": "2026-07-20T09:59:00Z",
                "updated_at": "2026-07-20T10:04:00Z",
                "body": body,
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "optional_asset_names": [],
                "checksums": {
                    "asset_name": "checksums.txt",
                    "file_sha256": _sha256(checksums_content),
                    "format": "sha256sum_text_two_space_two_lines_lf",
                    "line_count": 2,
                    "entries": [
                        {"name": wheel_name, "sha256": _sha256(wheel_content)},
                        {"name": sbom_name, "sha256": _sha256(sbom_content)},
                    ],
                    "matched_downloaded_assets": True,
                },
                "assets": assets,
            },
            "cross_reference": {
                "path": "model_cards/PII_ZH_QWEN3_0_6B_24CLASS_PUBLICATION.md",
                "file_sha256": _sha256("github publication model card"),
                "hugging_face_repository": publication.EXPECTED_HUGGING_FACE_REPOSITORY,
            },
        }
    )


def _github_release_assets_document(
    github: dict[str, Any], *, source_commit: str = SOURCE_SHA
) -> dict[str, Any]:
    by_role = {asset["role"]: asset for asset in github["release"]["assets"]}
    members = {
        "pii_zh:cli.py": {"file_sha256": _sha256("cli"), "size_bytes": 11},
        "pii_zh:service:app.py": {
            "file_sha256": _sha256("service"),
            "size_bytes": 12,
        },
        "pii_zh:cascade:routing.py": {
            "file_sha256": _sha256("routing"),
            "size_bytes": 13,
        },
        "pii_zh:taxonomy:taxonomy.yaml": {
            "file_sha256": _sha256("taxonomy"),
            "size_bytes": 14,
        },
    }
    implementation_generator = (
        publication.REPOSITORY_ROOT
        / publication.GITHUB_RELEASE_ASSETS_RECEIPT_GENERATOR
    )
    implementation_schema = (
        publication.REPOSITORY_ROOT
        / publication.GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA
    )
    assets = {
        role: {
            "name": remote["name"],
            "media_type": {
                "wheel": "application/zip",
                "sbom": "application/vnd.cyclonedx+json",
                "checksums": "text/plain; charset=utf-8",
            }[role],
            "file_sha256": remote["downloaded_sha256"],
            "size_bytes": remote["size_bytes"],
        }
        for role, remote in by_role.items()
    }
    document = {
        "schema_version": publication.GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA_VERSION,
        "source_commit": source_commit,
        "release": {
            "package_name": "pii-zh-qwen",
            "package_version": publication.PACKAGE_VERSION,
            "tag_name": publication.TAG_NAME,
            "github_repository": publication.EXPECTED_GITHUB_REPOSITORY,
            "asset_count": 3,
            "asset_names": [
                publication.REQUIRED_GITHUB_RELEASE_ASSETS["wheel"],
                publication.REQUIRED_GITHUB_RELEASE_ASSETS["sbom"],
                publication.REQUIRED_GITHUB_RELEASE_ASSETS["checksums"],
            ],
        },
        "source": {
            "git_source_commit": source_commit,
            "repository_head_verified": True,
            "implementation_files_match_commit": True,
            "binding_scope": (
                "verification_execution_context_at_exact_head_not_reproducible_wheel_build"
            ),
        },
        "assets": assets,
        "checksum_closure": {
            "status": "PASS",
            "format": "sha256sum_two_space_lf_v1",
            "line_count": 2,
            "ordered_entries": [
                {
                    "name": entry["name"],
                    "file_sha256": entry["sha256"],
                }
                for entry in github["release"]["checksums"]["entries"]
            ],
            "exact_payload_asset_set": True,
            "declared_digests_match_assets": True,
            "self_entry_absent": True,
            "single_asset_directory_verified": True,
        },
        "wheel_inventory": {
            "status": "PASS",
            "format": "python_wheel_zip_v1",
            "distribution_name": "pii-zh-qwen",
            "distribution_version": publication.PACKAGE_VERSION,
            "member_count": len(members),
            "member_inventory_sha256": publication.canonical_json_hash(members),
            "members": members,
            "required_member_ids": list(members),
            "required_members_present": True,
            "inventory_implementation_path": (
                "scripts/produce_community_cascade_release_v2_artifacts.py"
            ),
            "inventory_implementation_file_sha256": _sha256("wheel inventory"),
        },
        "sbom_verification": {
            "status": "PASS",
            "format": "CycloneDX-1.6",
            "bom_sha256": _sha256("bom"),
            "component_count": 10,
            "dependency_edge_count": 9,
            "deterministic_source_regeneration_match": True,
            "lockfile_file_sha256": _sha256("lock"),
            "pyproject_file_sha256": _sha256("pyproject"),
            "generator_path": "scripts/generate_sbom.py",
            "generator_file_sha256": _sha256("sbom generator"),
        },
        "isolated_wheel_smoke": {
            "status": "PASS",
            "profile_version": "c1-conservative-v2",
            "harness_path": "scripts/run_successor_clean_wheel_smoke.py",
            "harness_file_sha256": _sha256("harness"),
            "harness_result_sha256": _sha256("harness result"),
            "stdout_sha256": _sha256("stdout"),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_empty": True,
            "runtime_versions": {
                "python": "3.12.7",
                "pii-zh-qwen": publication.PACKAGE_VERSION,
                "presidio-analyzer": "2.2.362",
                "fastapi": "0.116.0",
                "httpx": "0.28.1",
            },
            "assertions": {
                "historical_default_unchanged": True,
                "installed_from_clean_site_packages": True,
                "raw_values_persisted": False,
                "model_packages_imported": False,
                "torch_imported": False,
                "transformers_imported": False,
                "gpu_visible": "",
                "valid_case_count": 3,
                "invalid_document_count": 1,
            },
            "isolation": {
                "profile": "bwrap_unshare_all_prlimit_temporary_venv_v1",
                "temporary_venv": True,
                "source_tree_import_disabled": True,
                "pip_no_index": True,
                "pip_no_dependencies": True,
                "inherited_dependency_environment": True,
                "bubblewrap_unshare_all": True,
                "network_namespace_unshared": True,
                "gpu_devices_hidden": True,
                "resource_limits_enforced": True,
            },
        },
        "public_artifact_scan": {
            "status": "PASS",
            "format": "pii_zh_public_artifact_scan_v3",
            "scanner_path": "scripts/scan_public_artifacts.py",
            "scanner_file_sha256": _sha256("scanner"),
            "scanned_file_count": 7,
            "scanned_inventory_sha256": _sha256("scan inventory"),
            "finding_count": 0,
            "finding_kinds": [],
        },
        "implementation": {
            "generator_path": (
                publication.GITHUB_RELEASE_ASSETS_RECEIPT_GENERATOR.as_posix()
            ),
            "generator_file_sha256": hashlib.sha256(
                implementation_generator.read_bytes()
            ).hexdigest(),
            "schema_path": publication.GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA.as_posix(),
            "schema_file_sha256": hashlib.sha256(
                implementation_schema.read_bytes()
            ).hexdigest(),
        },
    }
    return _seal(document, key="receipt_sha256")


def _hugging_face_document() -> dict[str, Any]:
    inventory = [
        {
            "path": path,
            "kind": "file",
            "size_bytes": index + 1,
            "remote_oid": (
                _sha256(f"remote file {path}")
                if path == "model.safetensors"
                else f"{index + 1:040x}"
            ),
            "downloaded_sha256": _sha256(f"remote file {path}"),
        }
        for index, path in enumerate(publication.REQUIRED_HUGGING_FACE_FILES)
    ]
    readme = next(item for item in inventory if item["path"] == "README.md")
    return _seal(
        {
            "schema_version": publication.HUGGING_FACE_EVIDENCE_SCHEMA_VERSION,
            "collector": {
                "name": "hugging-face-model-readonly-collector",
                "version": "1.0.0",
                "collected_at": "2026-07-20T10:06:00Z",
                "endpoint": "https://huggingface.co",
                "network_accessed": True,
                "remote_mutation_performed": False,
                "generator_path": publication.REMOTE_EVIDENCE_COLLECTOR_GENERATOR.as_posix(),
                "generator_file_sha256": _collector_generator_sha256(),
                "collection_run_sha256": _sha256("hugging face collector run"),
            },
            "repository": {
                "id": publication.EXPECTED_HUGGING_FACE_REPOSITORY,
                "visibility": "private",
            },
            "revision": {
                "requested_revision": "main",
                "resolved_sha": HF_SHA,
                "immutable_sha": HF_SHA,
            },
            "inventory": inventory,
            "cross_reference": {
                "path": "README.md",
                "file_sha256": readme["downloaded_sha256"],
                "github_repository": publication.EXPECTED_GITHUB_REPOSITORY,
                "github_source_sha": SOURCE_SHA,
            },
        }
    )


def _download_document(hugging_face: dict[str, Any]) -> dict[str, Any]:
    files = {
        item["path"]: {
            "file_sha256": item["downloaded_sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in hugging_face["inventory"]
    }
    remote_files: dict[str, dict[str, Any]] = {}
    for item in hugging_face["inventory"]:
        is_lfs = item["path"] == "model.safetensors"
        remote_files[item["path"]] = {
            "size_bytes": item["size_bytes"],
            "git_blob_oid": None if is_lfs else item["remote_oid"],
            "lfs_oid_sha256": item["remote_oid"] if is_lfs else None,
            "content_verification": "lfs_sha256" if is_lfs else "git_blob_sha1",
        }
    generator = publication.REPOSITORY_ROOT / publication.HUGGING_FACE_DOWNLOAD_GENERATOR
    document = {
        "schema_version": publication.HUGGING_FACE_DOWNLOAD_EVIDENCE_SCHEMA_VERSION,
        "provider": "huggingface_hub",
        "endpoint": "https://huggingface.co",
        "repository": publication.EXPECTED_HUGGING_FACE_REPOSITORY,
        "requested_revision": HF_SHA,
        "resolved_commit": HF_SHA,
        "downloaded_at": "2026-07-20T10:06:30Z",
        "remote_snapshot": {
            "private": True,
            "visibility": "private",
            "file_count": len(remote_files),
            "metadata_inventory_sha256": publication.canonical_json_hash(remote_files),
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
            "path": publication.HUGGING_FACE_DOWNLOAD_GENERATOR.as_posix(),
            "file_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        },
        "local_root": {
            "format": "materialized_regular_file_tree_v1",
            "file_count": len(files),
            "inventory_sha256": publication.canonical_json_hash(files),
            "files": files,
        },
    }
    return _seal(document, key="receipt_sha256")


def _package_verification(
    hugging_face: dict[str, Any], download: dict[str, Any], download_path: Path
) -> dict[str, Any]:
    files = download["local_root"]["files"]
    return {
        "schema_version": "pii-zh.community-v2-publication-package-verification.v1",
        "file_sha256": _sha256("package verification receipt file"),
        "receipt_sha256": _sha256("package verification receipt self hash"),
        "context": "hugging_face_immutable_download",
        "status": "PASS",
        "hugging_face_commit": hugging_face["revision"]["immutable_sha"],
        "github_source_commit": hugging_face["cross_reference"]["github_source_sha"],
        "hf_download_provenance_file_sha256": hashlib.sha256(
            download_path.read_bytes()
        ).hexdigest(),
        "hf_download_provenance_receipt_sha256": download["receipt_sha256"],
        "package_identity": {
            "manifest_sha256": _sha256("manifest canonical self hash"),
            "checksums_file_sha256": files["checksums.txt"]["file_sha256"],
            "payload_inventory_sha256": _sha256("manifest payload inventory"),
            "model_file_sha256": files["model.safetensors"]["file_sha256"],
            "verified_file_count": len(files) - 1,
        },
    }


def _evidence_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    github_path = tmp_path / "github-evidence.json"
    hugging_face_path = tmp_path / "hugging-face-evidence.json"
    download_path = tmp_path / "hf-download-verification.json"
    github_release_assets_path = tmp_path / "github-release-assets-verification.json"
    github = _github_document()
    hugging_face = _hugging_face_document()
    download = _download_document(hugging_face)
    _write_json(github_path, github)
    _write_json(download_path, download)
    _write_json(github_release_assets_path, _github_release_assets_document(github))
    os.chmod(github_release_assets_path, 0o444)
    hugging_face["package_verification"] = _package_verification(
        hugging_face, download, download_path
    )
    _write_json(hugging_face_path, _seal(hugging_face))
    return github_path, hugging_face_path, download_path, github_release_assets_path


def _load_all(
    github_path: Path,
    hugging_face_path: Path,
    download_path: Path,
    github_release_assets_path: Path,
) -> tuple[
    publication.JsonPayload,
    publication.JsonPayload,
    publication.JsonPayload,
    publication.JsonPayload,
]:
    return (
        publication.load_remote_evidence(github_path, platform="github"),
        publication.load_remote_evidence(hugging_face_path, platform="hugging_face"),
        publication.load_hugging_face_download_verification(download_path),
        publication.load_github_release_assets_verification(
            github_release_assets_path
        ),
    )


def _build(
    loaded: tuple[
        publication.JsonPayload,
        publication.JsonPayload,
        publication.JsonPayload,
        publication.JsonPayload,
    ],
    *,
    recorded_at: str = RECORDED_AT,
) -> dict[str, Any]:
    github, hugging_face, download, github_release_assets = loaded
    return publication.build_publication_receipt(
        github_evidence=github,
        github_release_assets_verification=github_release_assets,
        hugging_face_evidence=hugging_face,
        hugging_face_download_verification=download,
        recorded_at=recorded_at,
    )


def _receipt(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    publication.JsonPayload,
    publication.JsonPayload,
    publication.JsonPayload,
    publication.JsonPayload,
]:
    loaded = _load_all(*_evidence_files(tmp_path))
    return _build(loaded), *loaded


def _mutate_and_reseal(
    path: Path,
    mutation: Callable[[dict[str, Any]], None],
    *,
    key: str = "evidence_sha256",
) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    _write_json(path, _seal(document, key=key))


def _mutate_release_assets_and_reseal(
    path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    os.chmod(path, 0o644)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    _write_json(path, _seal(document, key="receipt_sha256"))
    os.chmod(path, 0o444)


@pytest.fixture(autouse=True)
def _fixed_local_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publication, "_repository_head_sha", lambda: SOURCE_SHA)


def test_build_is_prepublication_gate_with_exact_remote_bindings(tmp_path: Path) -> None:
    receipt, github, hugging_face, download, github_release_assets = _receipt(tmp_path)
    schema = json.loads(
        (publication.REPOSITORY_ROOT / publication.SCHEMA_PATH).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    publication.validate_publication_receipt(
        receipt,
        github_evidence=github,
        github_release_assets_verification=github_release_assets,
        hugging_face_evidence=hugging_face,
        hugging_face_download_verification=download,
    )

    assert receipt["status"] == "READY_FOR_FINAL_PUBLICATION_CONFIRMATION"
    assert receipt["stage"] == "pre_publication_gate"
    assert receipt["final_publication_confirmed"] is False
    assert receipt["github"]["visibility"] == "public"
    assert receipt["github"]["release"]["draft"] is True
    assert receipt["github"]["release"]["prerelease"] is True
    assert receipt["github"]["hosted_ci"]["head_sha"] == SOURCE_SHA
    assert receipt["github"]["signed_tag"]["verification"] == {
        **github.document["signed_tag"]["verification"]
    }
    assert receipt["evidence_bindings"]["github"]["signing_key_fingerprint"] == (
        SIGNING_KEY_FINGERPRINT
    )
    assert receipt["evidence_bindings"]["github"][
        "local_cryptographic_verification"
    ] is True
    assert receipt["evidence_bindings"]["github"]["generator_file_sha256"] == (
        _collector_generator_sha256()
    )
    release_assets_binding = receipt["evidence_bindings"][
        "github_release_assets_verification"
    ]
    assert release_assets_binding["file_sha256"] == github_release_assets.file_sha256
    assert release_assets_binding["receipt_sha256"] == (
        github_release_assets.document["receipt_sha256"]
    )
    assert release_assets_binding["schema_version"] == (
        publication.GITHUB_RELEASE_ASSETS_RECEIPT_SCHEMA_VERSION
    )
    assert release_assets_binding["source_commit"] == SOURCE_SHA
    assert release_assets_binding["generator_file_sha256"] == (
        github_release_assets.document["implementation"]["generator_file_sha256"]
    )
    assert release_assets_binding["schema_file_sha256"] == (
        github_release_assets.document["implementation"]["schema_file_sha256"]
    )
    assert release_assets_binding["checksum_closure_status"] == "PASS"
    assert release_assets_binding["wheel_inventory_status"] == "PASS"
    assert release_assets_binding["isolated_wheel_smoke_status"] == "PASS"
    assert release_assets_binding["public_artifact_scan_status"] == "PASS"
    assert release_assets_binding["public_artifact_scan_finding_count"] == 0
    assert receipt["hugging_face"]["visibility"] == "private"
    assert receipt["hugging_face"]["immutable_sha"] == HF_SHA
    assert receipt["hugging_face"]["download_verification"]["receipt_sha256"] == (
        download.document["receipt_sha256"]
    )
    assert receipt["hugging_face"]["package_verification"] == (
        hugging_face.document["package_verification"]
    )
    assert receipt["required_release_assets"] == publication.REQUIRED_GITHUB_RELEASE_ASSETS
    assert {item["role"] for item in receipt["release_assets"]} == {
        "checksums",
        "sbom",
        "wheel",
    }
    assert receipt["optional_release_asset_names"] == []
    assert receipt["github"]["release"]["checksums"] == (
        github.document["release"]["checksums"]
    )
    assert receipt["verification"]["release_checksums_match_downloaded_assets"] is True
    assert receipt["verification"][
        "github_release_assets_exact_remote_match_verified"
    ] is True
    github_source_reference = receipt["remote_reference_bindings"][
        "github_source_to_hugging_face_repository"
    ]
    assert "hugging_face_immutable_sha" not in github_source_reference
    release_reference = receipt["remote_reference_bindings"][
        "github_draft_release_to_hugging_face_revision"
    ]
    assert release_reference["reference_source"] == "github_draft_release_body"
    assert release_reference["hugging_face_revision_url"] == HF_REVISION_URL
    assert receipt["limitations"]["does_not_attest_final_publication"] is True
    assert receipt["receipt_sha256"] == publication.canonical_json_hash(
        receipt, remove="receipt_sha256"
    )


def test_write_is_no_clobber_and_validate_only_replays_all_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (
        github_path,
        hugging_face_path,
        download_path,
        github_release_assets_path,
    ) = _evidence_files(tmp_path)
    output = tmp_path / publication.DRAFT_RECEIPT_ASSET_NAME
    common = [
        "--github-evidence",
        str(github_path),
        "--github-release-assets-verification-receipt",
        str(github_release_assets_path),
        "--hugging-face-evidence",
        str(hugging_face_path),
        "--hugging-face-download-verification-receipt",
        str(download_path),
    ]
    argv = [*common, "--recorded-at", RECORDED_AT, "--output", str(output)]
    assert publication.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "READY_FOR_FINAL_PUBLICATION_CONFIRMATION"
    assert report["final_publication_confirmed"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    original = output.read_bytes()

    assert publication.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["blocker_ids"] == ["OUTPUT_EXISTS"]
    assert output.read_bytes() == original

    assert publication.main([*common, "--validate-only", str(output)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["mode"] == "validate-only"
    assert validated["local_write_performed"] is False


def test_dry_run_never_creates_a_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _evidence_files(tmp_path)
    assert publication.main(
        [
            "--github-evidence",
            str(paths[0]),
            "--github-release-assets-verification-receipt",
            str(paths[3]),
            "--hugging-face-evidence",
            str(paths[1]),
            "--hugging-face-download-verification-receipt",
            str(paths[2]),
            "--recorded-at",
            RECORDED_AT,
            "--dry-run",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "VALIDATED_WITHOUT_WRITING"
    assert report["remote_access_performed"] is False
    assert len(list(tmp_path.iterdir())) == 4


def test_real_standalone_hf_provenance_is_accepted_by_inline_contract(
    tmp_path: Path,
) -> None:
    paths = _evidence_files(tmp_path)
    standalone = hf_provenance.load_and_validate_provenance(paths[2])
    loaded = publication.load_hugging_face_download_verification(paths[2])
    assert loaded.document == standalone
    assert standalone["remote_snapshot"]["visibility"] == "private"
    assert standalone["remote_snapshot"]["metadata_coverage"][
        "content_verified_count"
    ] == len(standalone["remote_snapshot"]["files"])


def test_release_assets_receipt_requires_mode_0444(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    os.chmod(paths[3], 0o644)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_github_release_assets_verification(paths[3])
    assert captured.value.blocker_id == "RELEASE_ASSETS_RECEIPT_MODE_MISMATCH"


def test_release_assets_receipt_self_hash_must_verify(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    os.chmod(paths[3], 0o644)
    document = json.loads(paths[3].read_text(encoding="utf-8"))
    document["source"]["binding_scope"] = (
        "verification_execution_context_at_exact_head_not_reproducible_wheel_build"
    )
    document["receipt_sha256"] = "d" * 64
    _write_json(paths[3], document)
    os.chmod(paths[3], 0o444)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_github_release_assets_verification(paths[3])
    assert captured.value.blocker_id == "SELF_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("field", "blocker_id"),
    [
        ("generator_file_sha256", "RELEASE_ASSETS_GENERATOR_MISMATCH"),
        ("schema_file_sha256", "RELEASE_ASSETS_SCHEMA_BINDING_MISMATCH"),
    ],
)
def test_release_assets_receipt_binds_reviewed_implementation(
    tmp_path: Path, field: str, blocker_id: str
) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_release_assets_and_reseal(
        paths[3],
        lambda document: document["implementation"].__setitem__(field, "d" * 64),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_github_release_assets_verification(paths[3])
    assert captured.value.blocker_id == blocker_id


def test_release_assets_receipt_must_bind_exact_local_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evidence_files(tmp_path)
    monkeypatch.setattr(publication, "_repository_head_sha", lambda: "d" * 40)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_github_release_assets_verification(paths[3])
    assert captured.value.blocker_id == "RELEASE_ASSETS_HEAD_SOURCE_MISMATCH"


def test_release_assets_receipt_source_must_match_github_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["source_commit"] = "d" * 40
        document["source"]["git_source_commit"] = "d" * 40

    _mutate_release_assets_and_reseal(paths[3], mutate)
    monkeypatch.setattr(publication, "_repository_head_sha", lambda: "d" * 40)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "RELEASE_ASSETS_GITHUB_SOURCE_MISMATCH"


@pytest.mark.parametrize(
    ("field", "blocker_id"),
    [
        ("size_bytes", "RELEASE_ASSETS_SIZE_MISMATCH"),
        ("file_sha256", "RELEASE_ASSETS_DIGEST_MISMATCH"),
    ],
)
def test_release_assets_receipt_must_match_downloaded_remote_asset(
    tmp_path: Path, field: str, blocker_id: str
) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        asset = document["assets"]["wheel"]
        asset[field] = asset[field] + 1 if field == "size_bytes" else "d" * 64
        if field == "file_sha256":
            document["checksum_closure"]["ordered_entries"][0]["file_sha256"] = (
                asset[field]
            )

    _mutate_release_assets_and_reseal(paths[3], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == blocker_id


def test_release_assets_checksum_entries_are_cross_bound_item_by_item() -> None:
    github = _github_document()
    local = _github_release_assets_document(github)
    local["checksum_closure"]["ordered_entries"][0]["file_sha256"] = "d" * 64
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication._validate_github_release_assets_relationships(github, local)
    assert captured.value.blocker_id == "RELEASE_ASSETS_CHECKSUMS_ENTRIES_MISMATCH"


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        ("isolated_wheel_smoke", {"status": "FAIL"}),
        ("public_artifact_scan", {"status": "FAIL"}),
        ("wheel_inventory", None),
    ],
)
def test_release_assets_pass_evidence_is_required_by_independent_schema(
    tmp_path: Path, section: str, mutation: dict[str, str] | None
) -> None:
    paths = _evidence_files(tmp_path)

    def mutate_document(document: dict[str, Any]) -> None:
        if mutation is None:
            document.pop(section)
        else:
            document[section].update(mutation)

    _mutate_release_assets_and_reseal(paths[3], mutate_document)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_github_release_assets_verification(paths[3])
    assert captured.value.blocker_id == "RELEASE_ASSETS_RECEIPT_SCHEMA_REJECTED"


def test_caller_supplied_pass_status_is_schema_rejected(tmp_path: Path) -> None:
    document = _github_document()
    document["status"] = "PASS"
    path = tmp_path / "github.json"
    _write_json(path, _seal(document))
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(path, platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_remote_evidence_tamper_fails_self_hash(tmp_path: Path) -> None:
    document = _github_document()
    document["collector"]["version"] = "tampered"
    path = tmp_path / "github.json"
    _write_json(path, document)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(path, platform="github")
    assert captured.value.blocker_id == "SELF_HASH_MISMATCH"


def test_unverified_signed_tag_is_schema_rejected(tmp_path: Path) -> None:
    document = _github_document()
    document["signed_tag"]["verification"]["verified"] = False
    path = tmp_path / "github.json"
    _write_json(path, _seal(document))
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(path, platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("signing_key_fingerprint", "SHA256:invalid="),
        ("local_cryptographic_verification", False),
    ],
)
def test_local_signed_tag_verification_contract_is_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0],
        lambda doc: doc["signed_tag"]["verification"].__setitem__(field, value),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[0], platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_remote_evidence_must_bind_reviewed_collector_file(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1],
        lambda doc: doc["collector"].__setitem__("generator_file_sha256", "d" * 64),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[1], platform="hugging_face")
    assert captured.value.blocker_id == "REMOTE_COLLECTOR_GENERATOR_MISMATCH"


@pytest.mark.parametrize(
    ("field", "blocker_id"),
    [
        ("ref_target_sha", "GITHUB_TAG_OBJECT_MISMATCH"),
        ("tag_target_sha", "GITHUB_TAG_SOURCE_MISMATCH"),
    ],
)
def test_signed_tag_relationship_mismatch_fails(
    tmp_path: Path, field: str, blocker_id: str
) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(paths[0], lambda doc: doc["signed_tag"].__setitem__(field, "d" * 40))
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == blocker_id


def test_release_must_resolve_to_signed_source(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0], lambda doc: doc["release"].__setitem__("resolved_source_sha", "d" * 40)
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_RELEASE_SOURCE_MISMATCH"


def test_hosted_ci_must_run_on_exact_source(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0], lambda doc: doc["hosted_ci"].__setitem__("head_sha", "d" * 40)
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_HOSTED_CI_SOURCE_MISMATCH"


def test_failed_hosted_ci_is_schema_rejected(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0], lambda doc: doc["hosted_ci"].__setitem__("conclusion", "failure")
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[0], platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_non_draft_release_is_schema_rejected(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(paths[0], lambda doc: doc["release"].__setitem__("draft", False))
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[0], platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_release_body_hash_must_verify(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(paths[0], lambda doc: doc["release"].__setitem__("body", "changed"))
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_RELEASE_BODY_HASH_MISMATCH"


def test_release_body_must_contain_exact_hf_immutable_url(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        body = document["release"]["body"].replace(HF_SHA, "d" * 40)
        document["release"]["body"] = body
        document["release"]["body_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    _mutate_and_reseal(paths[0], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_RELEASE_HF_REVISION_REFERENCE_MISMATCH"


def test_release_asset_digests_must_match(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0],
        lambda doc: doc["release"]["assets"][0].__setitem__("downloaded_sha256", "d" * 64),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_ASSET_DIGEST_MISMATCH"


def test_release_asset_inventory_must_be_sorted(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        assets = document["release"]["assets"]
        assets[0], assets[1] = assets[1], assets[0]

    _mutate_and_reseal(paths[0], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_ASSET_INVENTORY_ORDER"


def test_fixed_required_release_asset_name_is_enforced(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        sbom = next(item for item in document["release"]["assets"] if item["role"] == "sbom")
        sbom["name"] = "wrong-sbom.json"
        document["release"]["assets"].sort(key=lambda item: item["name"])

    _mutate_and_reseal(paths[0], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_ASSET_SET_MISMATCH"


def test_required_release_asset_roles_cannot_be_duplicated(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        sbom = next(item for item in document["release"]["assets"] if item["role"] == "sbom")
        sbom["role"] = "wheel"

    _mutate_and_reseal(paths[0], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_REQUIRED_ASSET_ROLE_DUPLICATE"


def test_extra_release_asset_is_schema_rejected(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["release"]["assets"].append(
            _release_asset(asset_id=1004, role="optional", name="usage.pdf", content="pdf")
        )
        document["release"]["assets"].sort(key=lambda item: item["name"])

    _mutate_and_reseal(paths[0], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[0], platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_gate_receipt_cannot_self_reference_as_release_asset(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["release"]["assets"].append(
            _release_asset(
                asset_id=1004,
                role="optional",
                name=publication.DRAFT_RECEIPT_ASSET_NAME,
                content="impossible receipt bytes",
            )
        )
        document["release"]["assets"].sort(key=lambda item: item["name"])
        document["release"]["optional_asset_names"] = [publication.DRAFT_RECEIPT_ASSET_NAME]

    _mutate_and_reseal(paths[0], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[0], platform="github")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_checksums_file_digest_must_bind_downloaded_checksums_asset(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0],
        lambda doc: doc["release"]["checksums"].__setitem__("file_sha256", "d" * 64),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_CHECKSUMS_FILE_MISMATCH"


def test_checksums_entries_must_match_downloaded_wheel_and_sbom(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[0],
        lambda doc: doc["release"]["checksums"]["entries"][0].__setitem__(
            "sha256", "d" * 64
        ),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "GITHUB_CHECKSUMS_DIGEST_MISMATCH"


def test_public_hugging_face_repository_is_schema_rejected(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1], lambda doc: doc["repository"].__setitem__("visibility", "public")
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[1], platform="hugging_face")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_hugging_face_revision_must_be_immutable(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1], lambda doc: doc["revision"].__setitem__("resolved_sha", "d" * 40)
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_IMMUTABLE_REVISION_MISMATCH"


def test_hugging_face_inventory_requires_fixed_file_set(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["inventory"] = [
            item for item in document["inventory"] if item["path"] != ".gitattributes"
        ]

    _mutate_and_reseal(paths[1], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[1], platform="hugging_face")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_hugging_face_inventory_must_be_sorted(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["inventory"][0], document["inventory"][1] = (
            document["inventory"][1],
            document["inventory"][0],
        )

    _mutate_and_reseal(paths[1], mutate)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_INVENTORY_ORDER"


def test_hugging_face_cross_reference_hash_binds_inventory(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1], lambda doc: doc["cross_reference"].__setitem__("file_sha256", "d" * 64)
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_CROSS_REFERENCE_HASH_MISMATCH"


def test_hugging_face_model_card_must_reference_exact_github_source(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1], lambda doc: doc["cross_reference"].__setitem__("github_source_sha", "d" * 40)
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "BIDIRECTIONAL_GITHUB_SOURCE_MISMATCH"


def test_download_receipt_must_bind_hf_immutable_sha(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[2],
        lambda doc: (
            doc.__setitem__("requested_revision", "d" * 40),
            doc.__setitem__("resolved_commit", "d" * 40),
        ),
        key="receipt_sha256",
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_DOWNLOAD_IMMUTABLE_SHA_MISMATCH"


def test_download_receipt_files_must_match_remote_inventory(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["local_root"]["files"]["model.safetensors"]["file_sha256"] = "d" * 64
        document["local_root"]["inventory_sha256"] = publication.canonical_json_hash(
            document["local_root"]["files"]
        )

    _mutate_and_reseal(paths[2], mutate, key="receipt_sha256")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_DOWNLOAD_INVENTORY_MISMATCH"


def test_download_receipt_must_prove_private_remote_snapshot(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        document["remote_snapshot"]["private"] = False
        document["remote_snapshot"]["visibility"] = "public"

    _mutate_and_reseal(paths[2], mutate, key="receipt_sha256")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_DOWNLOAD_VISIBILITY_MISMATCH"


def test_download_receipt_requires_complete_remote_content_verification(
    tmp_path: Path,
) -> None:
    paths = _evidence_files(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        remote = document["remote_snapshot"]
        remote["files"]["README.md"][
            "content_verification"
        ] = "immutable_revision_transport_only"
        remote["metadata_coverage"]["content_verified_count"] -= 1
        remote["metadata_inventory_sha256"] = publication.canonical_json_hash(
            remote["files"]
        )

    _mutate_and_reseal(paths[2], mutate, key="receipt_sha256")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_DOWNLOAD_REMOTE_COVERAGE_INCOMPLETE"


def test_download_receipt_inventory_must_self_verify(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[2],
        lambda doc: doc["local_root"].__setitem__("inventory_sha256", "d" * 64),
        key="receipt_sha256",
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_hugging_face_download_verification(paths[2])
    assert captured.value.blocker_id == "HF_DOWNLOAD_INVENTORY_SELF_HASH_MISMATCH"


def test_download_receipt_generator_must_be_reviewed_implementation(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[2],
        lambda doc: doc["generator"].__setitem__("file_sha256", "d" * 64),
        key="receipt_sha256",
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_hugging_face_download_verification(paths[2])
    assert captured.value.blocker_id == "HF_DOWNLOAD_GENERATOR_MISMATCH"


def test_hf_package_verification_must_be_pass(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1],
        lambda doc: doc["package_verification"].__setitem__("status", "FAIL"),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(paths[1], platform="hugging_face")
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


@pytest.mark.parametrize("field", ["hugging_face_commit", "github_source_commit"])
def test_hf_package_verification_binds_exact_commits(tmp_path: Path, field: str) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1],
        lambda doc: doc["package_verification"].__setitem__(field, "d" * 40),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_PACKAGE_VERIFICATION_SOURCE_MISMATCH"


def test_hf_package_verification_binds_exact_provenance_file(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1],
        lambda doc: doc["package_verification"].__setitem__(
            "hf_download_provenance_file_sha256", "d" * 64
        ),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_PACKAGE_VERIFICATION_PROVENANCE_MISMATCH"


def test_hf_package_identity_binds_downloaded_model(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    _mutate_and_reseal(
        paths[1],
        lambda doc: doc["package_verification"]["package_identity"].__setitem__(
            "model_file_sha256", "d" * 64
        ),
    )
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(_load_all(*paths))
    assert captured.value.blocker_id == "HF_PACKAGE_IDENTITY_MISMATCH"


def test_receipt_timestamp_cannot_predate_any_evidence(tmp_path: Path) -> None:
    loaded = _load_all(*_evidence_files(tmp_path))
    with pytest.raises(publication.PublicationReceiptError) as captured:
        _build(loaded, recorded_at="2026-07-20T10:06:15Z")
    assert captured.value.blocker_id == "RECEIPT_TIME_ORDER"


def test_receipt_cannot_claim_published_status(tmp_path: Path) -> None:
    receipt, github, hugging_face, download, release_assets = _receipt(tmp_path)
    receipt["status"] = "PUBLISHED"
    receipt = _seal(receipt, key="receipt_sha256")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.validate_publication_receipt(
            receipt,
            github_evidence=github,
            github_release_assets_verification=release_assets,
            hugging_face_evidence=hugging_face,
            hugging_face_download_verification=download,
        )
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_github_source_reference_cannot_claim_hf_immutable_sha(tmp_path: Path) -> None:
    receipt, github, hugging_face, download, release_assets = _receipt(tmp_path)
    receipt["remote_reference_bindings"][
        "github_source_to_hugging_face_repository"
    ]["hugging_face_immutable_sha"] = HF_SHA
    receipt = _seal(receipt, key="receipt_sha256")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.validate_publication_receipt(
            receipt,
            github_evidence=github,
            github_release_assets_verification=release_assets,
            hugging_face_evidence=hugging_face,
            hugging_face_download_verification=download,
        )
    assert captured.value.blocker_id == "SCHEMA_REJECTED"


def test_validate_rejects_semantically_resealed_receipt_tamper(tmp_path: Path) -> None:
    receipt, github, hugging_face, download, release_assets = _receipt(tmp_path)
    receipt["release_assets"][0]["size_bytes"] += 1
    receipt = _seal(receipt, key="receipt_sha256")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.validate_publication_receipt(
            receipt,
            github_evidence=github,
            github_release_assets_verification=release_assets,
            hugging_face_evidence=hugging_face,
            hugging_face_download_verification=download,
        )
    assert captured.value.blocker_id == "RECEIPT_DERIVATION_MISMATCH"


def test_validate_binds_exact_evidence_file_bytes(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    github, hugging_face, download, release_assets = _load_all(*paths)
    receipt = _build((github, hugging_face, download, release_assets))
    document = json.loads(paths[0].read_text(encoding="utf-8"))
    _write_json(paths[0], document, compact=True)
    reformatted = publication.load_remote_evidence(paths[0], platform="github")
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.validate_publication_receipt(
            receipt,
            github_evidence=reformatted,
            github_release_assets_verification=release_assets,
            hugging_face_evidence=hugging_face,
            hugging_face_download_verification=download,
        )
    assert captured.value.blocker_id == "RECEIPT_DERIVATION_MISMATCH"


def test_symlinked_evidence_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "evidence.json"
    _write_json(target, _github_document())
    link.symlink_to(target)
    with pytest.raises(publication.PublicationReceiptError) as captured:
        publication.load_remote_evidence(link, platform="github")
    assert captured.value.blocker_id == "UNSAFE_INPUT_FILE"


def test_validate_only_requires_mode_0444(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _evidence_files(tmp_path)
    document = _build(_load_all(*paths))
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, document)
    os.chmod(receipt_path, 0o644)
    assert publication.main(
        [
            "--github-evidence",
            str(paths[0]),
            "--github-release-assets-verification-receipt",
            str(paths[3]),
            "--hugging-face-evidence",
            str(paths[1]),
            "--hugging-face-download-verification-receipt",
            str(paths[2]),
            "--validate-only",
            str(receipt_path),
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out)["blocker_ids"] == ["RECEIPT_MODE_MISMATCH"]


def test_producer_does_not_use_network_or_environment_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(publication.__file__).read_text(encoding="utf-8")
    assert "huggingface_hub" not in source
    assert "requests" not in source
    assert "urllib.request" not in source
    assert "HF_TOKEN" not in source
    assert "os.environ" not in source

    def deny_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", deny_socket)
    receipt, github, hugging_face, download, release_assets = _receipt(tmp_path)
    publication.validate_publication_receipt(
        receipt,
        github_evidence=github,
        github_release_assets_verification=release_assets,
        hugging_face_evidence=hugging_face,
        hugging_face_download_verification=download,
    )
