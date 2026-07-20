from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import build_community_v2_publication_successor as successor
from scripts import materialize_community_v2_hf_snapshot as materializer
from tests.unit.test_community_v2_publication_successor import (
    HUGGING_FACE_REPOSITORY,
    _fixture_inputs,
    _prepare,
)

REVISION = "b" * 40
FAKE_HF_TOKEN = "fixture-token-never-written-to-receipt"


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_remote_package(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "remote-fixture"
    fixture_root.mkdir()
    inputs = _fixture_inputs(fixture_root)
    plan = _prepare(fixture_root, inputs)
    successor.build_publication_successor(plan)
    return plan.output


def _cache_package(package: Path, cache_dir: Path) -> dict[str, Path]:
    cached: dict[str, Path] = {}
    for source in sorted(package.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(package).as_posix()
        destination = cache_dir / "fake-hub-blobs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        cached[relative] = destination
    return cached


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote_paths: list[str],
    cached: dict[str, Path],
    resolved_commit: str = REVISION,
    remote_id: str | None = HUGGING_FACE_REPOSITORY,
    download_override: Any | None = None,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "api_init": [],
        "repo_info": [],
        "list_repo_files": [],
        "download": [],
    }

    class FakeApi:
        def __init__(self, *, endpoint: str, token: str) -> None:
            calls["api_init"].append((endpoint, token))

        def repo_info(
            self,
            repository: str,
            *,
            revision: str,
            repo_type: str,
            files_metadata: bool,
        ) -> SimpleNamespace:
            calls["repo_info"].append(
                (repository, revision, repo_type, files_metadata)
            )
            siblings = []
            for relative in dict.fromkeys(remote_paths):
                cached_path = cached.get(relative)
                if cached_path is None:
                    continue
                payload = cached_path.read_bytes()
                if relative == "model.safetensors":
                    blob_id = "d" * 40
                    lfs = SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest())
                else:
                    blob = hashlib.sha1(usedforsecurity=False)
                    blob.update(f"blob {len(payload)}\0".encode("ascii"))
                    blob.update(payload)
                    blob_id = blob.hexdigest()
                    lfs = None
                siblings.append(
                    SimpleNamespace(
                        rfilename=relative,
                        size=len(payload),
                        blob_id=blob_id,
                        lfs=lfs,
                    )
                )
            return SimpleNamespace(
                sha=resolved_commit,
                id=remote_id,
                private=False,
                siblings=siblings,
            )

        def list_repo_files(
            self,
            repository: str,
            *,
            revision: str,
            repo_type: str,
        ) -> list[str]:
            calls["list_repo_files"].append((repository, revision, repo_type))
            return list(remote_paths)

    def fake_download(**kwargs: Any) -> str:
        calls["download"].append(dict(kwargs))
        if download_override is not None:
            return str(download_override(kwargs))
        return str(cached[kwargs["filename"]])

    monkeypatch.setattr(materializer, "HfApi", FakeApi)
    monkeypatch.setattr(materializer, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", FAKE_HF_TOKEN)
    return calls


def _materialize_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, list[Any]]]:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    remote_paths = sorted(cached, reverse=True)
    calls = _install_fake_hub(
        monkeypatch,
        remote_paths=remote_paths,
        cached=cached,
    )
    output = tmp_path / "materialized"
    receipt = materializer.materialize_snapshot(
        repository=HUGGING_FACE_REPOSITORY,
        revision=REVISION,
        output=output,
        cache_dir=cache_dir,
    )
    return output, cache_dir, receipt, cached, calls


def test_provenance_schema_is_valid() -> None:
    schema = json.loads(materializer.SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_materializes_exact_commit_and_complete_remote_inventory_without_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, cache_dir, receipt, cached, calls = _materialize_fixture(
        tmp_path,
        monkeypatch,
    )

    expected_paths = sorted(cached)
    assert calls["api_init"] == [(materializer.OFFICIAL_ENDPOINT, FAKE_HF_TOKEN)]
    assert calls["repo_info"] == [
        (HUGGING_FACE_REPOSITORY, REVISION, "model", True)
    ]
    assert calls["list_repo_files"] == [
        (HUGGING_FACE_REPOSITORY, REVISION, "model")
    ]
    assert [call["filename"] for call in calls["download"]] == expected_paths
    for call in calls["download"]:
        assert call == {
            "repo_id": HUGGING_FACE_REPOSITORY,
            "filename": call["filename"],
            "repo_type": "model",
            "revision": REVISION,
            "cache_dir": cache_dir,
            "endpoint": materializer.OFFICIAL_ENDPOINT,
            "token": FAKE_HF_TOKEN,
        }

    assert receipt["repository"] == HUGGING_FACE_REPOSITORY
    assert receipt["requested_revision"] == REVISION
    assert receipt["resolved_commit"] == REVISION
    assert receipt["remote_snapshot"]["private"] is False
    assert receipt["remote_snapshot"]["visibility"] == "public"
    assert sorted(receipt["remote_snapshot"]["files"]) == expected_paths
    assert receipt["remote_snapshot"]["file_count"] == len(expected_paths)
    assert receipt["remote_snapshot"]["metadata_coverage"] == {
        "size_count": len(expected_paths),
        "git_blob_oid_count": len(expected_paths),
        "lfs_oid_count": 1,
        "content_verified_count": len(expected_paths),
    }
    assert receipt["remote_snapshot"]["evidence_boundary"] == (
        materializer.REMOTE_EVIDENCE_BOUNDARY
    )
    assert sorted(receipt["local_root"]["files"]) == expected_paths
    assert receipt["local_root"]["file_count"] == len(expected_paths)
    assert receipt["local_root"]["inventory_sha256"] == successor.canonical_json_hash(
        receipt["local_root"]["files"]
    )
    assert receipt["receipt_sha256"] == successor.canonical_json_hash(
        receipt,
        remove="receipt_sha256",
    )
    materializer._validate_document(receipt)
    materializer.verify_local_root_binding(output, receipt)
    successor.verify_successor_package(output)
    assert FAKE_HF_TOKEN not in json.dumps(receipt, sort_keys=True)

    for relative, cached_path in cached.items():
        materialized_path = output / relative
        assert materialized_path.read_bytes() == cached_path.read_bytes()
        assert (materialized_path.stat().st_dev, materialized_path.stat().st_ino) != (
            cached_path.stat().st_dev,
            cached_path.stat().st_ino,
        )

    provenance_path = tmp_path / "hf-download-provenance.json"
    materializer._write_new(provenance_path, receipt)
    assert stat.S_IMODE(provenance_path.stat().st_mode) == 0o444
    assert materializer.load_and_validate_provenance(provenance_path) == receipt
    with pytest.raises(materializer.HfDownloadProvenanceError, match="already exists"):
        materializer._write_new(provenance_path, receipt)


@pytest.mark.parametrize("revision", ["main", "B" * 40, "b" * 39])
def test_rejects_nonimmutable_revision_before_hub_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
) -> None:
    class UnexpectedApi:
        def __init__(self, **_kwargs: Any) -> None:
            pytest.fail("Hub API must not be constructed for a mutable revision")

    monkeypatch.setattr(materializer, "HfApi", UnexpectedApi)
    monkeypatch.setattr(
        materializer,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("download must not run"),
    )

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="full lowercase commit SHA",
    ):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=revision,
            output=tmp_path / "output",
            cache_dir=tmp_path / "cache",
        )


def test_requires_explicit_process_hf_token_before_hub_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedApi:
        def __init__(self, **_kwargs: Any) -> None:
            pytest.fail("Hub API must not be constructed without HF_TOKEN")

    monkeypatch.setattr(materializer, "HfApi", UnexpectedApi)
    monkeypatch.setattr(materializer, "hf_hub_download", lambda **_kwargs: "unused")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="HF_TOKEN must be present and non-empty",
    ):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=tmp_path / "output",
            cache_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("resolved_commit", "remote_id"),
    [
        ("c" * 40, HUGGING_FACE_REPOSITORY),
        (REVISION, "other/model"),
        (REVISION, None),
    ],
)
def test_rejects_hub_resolution_drift_before_listing_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolved_commit: str,
    remote_id: str | None,
) -> None:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    calls = _install_fake_hub(
        monkeypatch,
        remote_paths=list(cached),
        cached=cached,
        resolved_commit=resolved_commit,
        remote_id=remote_id,
    )
    output = tmp_path / "materialized"

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="different repository or commit",
    ):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=output,
            cache_dir=cache_dir,
        )

    assert calls["list_repo_files"] == []
    assert calls["download"] == []
    assert not output.exists()


@pytest.mark.parametrize(
    "remote_paths",
    [[], ["../escape"], ["README.md", "README.md"]],
)
def test_rejects_empty_unsafe_or_duplicate_remote_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_paths: list[str],
) -> None:
    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir()
    calls = _install_fake_hub(
        monkeypatch,
        remote_paths=remote_paths,
        cached={},
    )
    output = tmp_path / "materialized"

    with pytest.raises(materializer.HfDownloadProvenanceError):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=output,
            cache_dir=cache_dir,
        )

    assert calls["download"] == []
    assert not output.exists()


def test_incomplete_remote_package_fails_successor_closure_and_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    remote_paths = [path for path in sorted(cached) if path != "README.md"]
    _install_fake_hub(
        monkeypatch,
        remote_paths=remote_paths,
        cached=cached,
    )
    output = tmp_path / "materialized"

    with pytest.raises(successor.PublicationSuccessorError):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=output,
            cache_dir=cache_dir,
        )

    assert not output.exists()


def test_rejects_download_path_outside_cache_and_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    outside = tmp_path / "outside-cache.bin"
    outside.write_bytes(b"not a Hub cache object")
    first = sorted(cached)[0]

    def outside_first(kwargs: dict[str, Any]) -> Path:
        if kwargs["filename"] == first:
            return outside
        return cached[kwargs["filename"]]

    _install_fake_hub(
        monkeypatch,
        remote_paths=list(cached),
        cached=cached,
        download_override=outside_first,
    )
    output = tmp_path / "materialized"

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="outside its root",
    ):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=output,
            cache_dir=cache_dir,
        )

    assert not output.exists()


def test_rejects_download_bytes_that_differ_from_advertised_lfs_oid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    mutated = False

    def mutate_weight(kwargs: dict[str, Any]) -> Path:
        nonlocal mutated
        relative = kwargs["filename"]
        cached_path = cached[relative]
        if relative == "model.safetensors" and not mutated:
            original = cached_path.read_bytes()
            cached_path.write_bytes(b"X" * len(original))
            mutated = True
        return cached_path

    _install_fake_hub(
        monkeypatch,
        remote_paths=list(cached),
        cached=cached,
        download_override=mutate_weight,
    )
    output = tmp_path / "materialized"

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="differs from its LFS OID",
    ):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=output,
            cache_dir=cache_dir,
        )

    assert not output.exists()


def test_rejects_cache_root_inode_replacement_during_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    moved_cache = tmp_path / "moved-hf-cache"
    replaced = False

    def replace_cache(kwargs: dict[str, Any]) -> Path:
        nonlocal replaced
        if not replaced:
            cache_dir.rename(moved_cache)
            cache_dir.mkdir()
            replaced = True
        relative = kwargs["filename"]
        return moved_cache / cached[relative].relative_to(cache_dir)

    _install_fake_hub(
        monkeypatch,
        remote_paths=list(cached),
        cached=cached,
        download_override=replace_cache,
    )
    output = tmp_path / "materialized"

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="HF cache changed during materialization",
    ):
        materializer.materialize_snapshot(
            repository=HUGGING_FACE_REPOSITORY,
            revision=REVISION,
            output=output,
            cache_dir=cache_dir,
        )

    assert not output.exists()


def test_provenance_self_hash_inventory_and_local_bytes_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _cache_dir, receipt, _cached, _calls = _materialize_fixture(
        tmp_path,
        monkeypatch,
    )

    self_hash_tamper = copy.deepcopy(receipt)
    self_hash_tamper["downloaded_at"] = "2026-07-20T00:00:00Z"
    self_hash_path = tmp_path / "self-hash-tamper.json"
    _write_json(self_hash_path, self_hash_tamper)
    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="self-hash failed",
    ):
        materializer.load_and_validate_provenance(self_hash_path)

    inventory_tamper = copy.deepcopy(receipt)
    inventory_tamper["local_root"]["files"]["README.md"]["file_sha256"] = "0" * 64
    inventory_tamper["receipt_sha256"] = successor.canonical_json_hash(
        inventory_tamper,
        remove="receipt_sha256",
    )
    inventory_path = tmp_path / "inventory-tamper.json"
    _write_json(inventory_path, inventory_tamper)
    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="inventory does not self-verify",
    ):
        materializer.load_and_validate_provenance(inventory_path)

    (output / "README.md").write_bytes(b"changed after provenance was emitted\n")
    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="differs from its HF download provenance",
    ):
        materializer.verify_local_root_binding(output, receipt)


def test_regular_reader_rejects_path_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "receipt.json"
    replacement = tmp_path / "replacement.json"
    artifact.write_bytes(b'{"original":true}\n')
    replacement.write_bytes(b'{"replacement":true}\n')
    real_read = os.read
    swapped = False

    def swap_after_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        block = real_read(descriptor, size)
        if block and not swapped:
            os.replace(replacement, artifact)
            swapped = True
        return block

    monkeypatch.setattr(materializer.os, "read", swap_after_read)
    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="changed while it was read",
    ):
        materializer._read_regular(
            artifact,
            field="fixture receipt",
            maximum=1024,
        )


def test_regular_reader_and_writer_reject_symlink_parent(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "input.json").write_text('{"ok":true}\n', encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="no symlink components",
    ):
        materializer._read_regular(
            linked_parent / "input.json",
            field="linked input",
            maximum=1024,
        )
    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="no symlink components",
    ):
        materializer._write_new(linked_parent / "receipt.json", {"ok": True})
    assert not (real_parent / "receipt.json").exists()


def test_provenance_writer_does_not_create_missing_parent(tmp_path: Path) -> None:
    missing_parent = tmp_path / "missing-parent"

    with pytest.raises(
        materializer.HfDownloadProvenanceError,
        match="parent is unavailable",
    ):
        materializer._write_new(missing_parent / "receipt.json", {"ok": True})

    assert not missing_parent.exists()


def test_main_removes_only_its_new_snapshot_when_provenance_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = _build_remote_package(tmp_path)
    cache_dir = tmp_path / "hf-cache"
    cached = _cache_package(package, cache_dir)
    _install_fake_hub(
        monkeypatch,
        remote_paths=list(cached),
        cached=cached,
    )
    output = tmp_path / "materialized"
    provenance = tmp_path / "missing-parent" / "provenance.json"
    monkeypatch.setattr(
        materializer.sys,
        "argv",
        [
            materializer.GENERATOR_PATH.as_posix(),
            "--repository",
            HUGGING_FACE_REPOSITORY,
            "--revision",
            REVISION,
            "--output",
            str(output),
            "--cache-dir",
            str(cache_dir),
            "--provenance-output",
            str(provenance),
        ],
    )

    assert materializer.main() == 2
    assert not output.exists()
    assert not provenance.exists()
    assert FAKE_HF_TOKEN not in capsys.readouterr().err
