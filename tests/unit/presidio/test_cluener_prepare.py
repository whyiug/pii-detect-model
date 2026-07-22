from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from pii_zh.presidio import cluener_prepare as prepare
from pii_zh.presidio.cluener_primary import (
    CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED,
    CLUENER_PRIMARY_REQUIRED_FILES,
    VerifiedCluenerPrimaryArtifact,
)
from pii_zh.training.safe_conversion import sha256_file


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    for name in prepare.CLUENER_PRIMARY_SOURCE_REQUIRED_FILES:
        (root / name).write_bytes(f"source::{name}".encode())
    expected = {
        name: sha256_file(root / name)
        for name in prepare.CLUENER_PRIMARY_SOURCE_REQUIRED_FILES
    }
    monkeypatch.setattr(
        prepare,
        "CLUENER_PRIMARY_SOURCE_REQUIRED_FILES",
        MappingProxyType(expected),
    )
    return root


def test_preparer_source_hash_is_the_public_receipt_converter_identity() -> None:
    assert sha256_file(Path(prepare.__file__).resolve(strict=True)) == (
        CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED["converter_sha256"]
    )


def test_prepare_builds_only_the_verified_eight_file_artifact_and_keeps_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared" / "cluener"
    conversion_calls: list[tuple[Path, Path, bool]] = []
    verification_roots: list[Path] = []

    def fake_materialize(vocab_path: Path, destination_path: Path) -> None:
        assert vocab_path == source / "vocab.txt"
        destination_path.write_bytes(b"fixture tokenizer")

    def fake_convert(
        source_path: str | Path,
        destination_path: str | Path,
        *,
        delete_source: bool,
    ) -> dict[str, object]:
        source_file = Path(source_path)
        destination_file = Path(destination_path)
        conversion_calls.append((source_file, destination_file, delete_source))
        destination_file.write_bytes(b"fixture safetensors")
        receipt = dict(CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED)
        receipt.pop("converter_sha256")
        return receipt

    def fake_verify(root: str | Path) -> VerifiedCluenerPrimaryArtifact:
        resolved = Path(root).resolve(strict=True)
        verification_roots.append(resolved)
        assert {item.name for item in resolved.iterdir()} == set(
            CLUENER_PRIMARY_REQUIRED_FILES
        )
        receipt = json.loads((resolved / "conversion_receipt.json").read_text())
        assert receipt["source_deleted"] is False
        assert receipt["converter_sha256"] == (
            CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED["converter_sha256"]
        )
        return VerifiedCluenerPrimaryArtifact(
            root=resolved,
            model_identity=MappingProxyType({"fixture": True}),
            _snapshot=(),
        )

    monkeypatch.setattr(prepare, "_materialize_tokenizer_json", fake_materialize)
    monkeypatch.setattr(prepare, "convert_restricted_state_dict", fake_convert)
    monkeypatch.setattr(prepare, "verify_cluener_primary_artifact", fake_verify)

    artifact = prepare.prepare_cluener_primary_artifact(source, output)

    assert artifact.root == output.resolve()
    assert len(verification_roots) == 2
    assert conversion_calls == [
        (
            source / prepare.CLUENER_PRIMARY_SOURCE_WEIGHT_NAME,
            verification_roots[0] / "model.safetensors",
            False,
        )
    ]
    assert (source / prepare.CLUENER_PRIMARY_SOURCE_WEIGHT_NAME).exists()
    assert not (output / prepare.CLUENER_PRIMARY_SOURCE_WEIGHT_NAME).exists()


def test_prepare_rejects_source_that_is_not_the_pinned_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in prepare.CLUENER_PRIMARY_SOURCE_REQUIRED_FILES:
        (source / name).write_bytes(b"not the pinned revision")

    with pytest.raises(prepare.CluenerPrimaryPreparationError, match="pinned revision"):
        prepare.prepare_cluener_primary_artifact(source, tmp_path / "output")
