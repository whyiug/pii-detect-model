from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conftest import run_script, write_json


def test_public_scan_redacts_and_allows_only_reviewed_synthetic_canary(
    repository_root: Path, tmp_path: Path
) -> None:
    value = "138" + "0013" + "8000"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "example.md").write_text(f"Synthetic canary: {value}\n", encoding="utf-8")
    result = run_script(repository_root, "scan_public_artifacts.py", str(artifact))
    assert result.returncode == 1
    assert value not in result.stdout
    fingerprint = "sha256:" + hashlib.sha256(value.encode()).hexdigest()
    allowlist = tmp_path / "canaries.json"
    write_json(
        allowlist,
        {
            "canaries": [
                {
                    "purpose": "Synthetic unit-test canary; not a real person's number.",
                    "sha256": fingerprint,
                    "synthetic": True,
                }
            ],
            "schema_version": 1,
        },
    )
    allowed = run_script(
        repository_root,
        "scan_public_artifacts.py",
        str(artifact),
        "--canary-allowlist",
        str(allowlist),
    )
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr


def test_canary_allowlist_cannot_suppress_secret(repository_root: Path, tmp_path: Path) -> None:
    secret = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "1234567890"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.md").write_text(f"token={secret}\n", encoding="utf-8")
    allowlist = tmp_path / "canaries.json"
    write_json(
        allowlist,
        {
            "canaries": [
                {
                    "purpose": "Attempted synthetic fixture allowlist.",
                    "sha256": "sha256:" + hashlib.sha256(secret.encode()).hexdigest(),
                    "synthetic": True,
                }
            ],
            "schema_version": 1,
        },
    )
    result = run_script(
        repository_root,
        "scan_public_artifacts.py",
        str(artifact),
        "--canary-allowlist",
        str(allowlist),
    )
    assert result.returncode == 1
    assert secret not in result.stdout
    assert "huggingface_token" in result.stdout


def test_sbom_generation_is_deterministic_and_cyclonedx(
    repository_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "sbom.cdx.json"
    first = run_script(
        repository_root,
        "generate_sbom.py",
        "--lockfile",
        str(repository_root / "uv.lock"),
        "--pyproject",
        str(repository_root / "pyproject.toml"),
        "--output",
        str(output),
    )
    assert first.returncode == 0, first.stderr
    before = output.read_bytes()
    check = run_script(
        repository_root,
        "generate_sbom.py",
        "--lockfile",
        str(repository_root / "uv.lock"),
        "--pyproject",
        str(repository_root / "pyproject.toml"),
        "--output",
        str(output),
        "--check",
    )
    assert check.returncode == 0, check.stderr
    assert output.read_bytes() == before
    document = json.loads(before)
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.6"
    assert document["components"]
