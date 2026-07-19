from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from tests.release.conftest import run_script, write_json


def test_public_scan_redacts_and_allows_only_reviewed_synthetic_canary(
    repository_root: Path, tmp_path: Path
) -> None:
    value = "138" + "0013" + "8000"
    fingerprint = "sha256:" + hashlib.sha256(value.encode()).hexdigest()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "example.md").write_text(f"Synthetic canary: {value}\n", encoding="utf-8")
    blocked_report = tmp_path / "blocked-scan.json"
    result = run_script(
        repository_root,
        "scan_public_artifacts.py",
        str(artifact),
        "--json-output",
        str(blocked_report),
    )
    assert result.returncode == 1
    assert value not in result.stdout
    assert fingerprint not in result.stdout
    serialized_report = blocked_report.read_text(encoding="utf-8")
    assert fingerprint not in serialized_report
    assert json.loads(serialized_report)["findings"] == [
        {"kind": "potential_cn_mobile", "line": 1, "path": "example.md"}
    ]
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


def test_scan_blocks_private_local_paths_without_echoing_values(
    repository_root: Path, tmp_path: Path
) -> None:
    private_values = (
        "/" + "data/ruiyang/code/project",
        "/" + "data1/models/private-checkpoint",
        "/" + "home/alice/.cache/model",
        "file:" + "///" + "data1/runs/private-result.json",
        "C:" + "\\Users\\alice\\models\\checkpoint",
        "D:" + "/" + "Users/bob/runs/result.json",
    )
    artifact = tmp_path / "private-paths.md"
    artifact.write_text(
        "\n".join(f"location={value}" for value in private_values) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "private-paths-scan.json"

    result = run_script(
        repository_root,
        "scan_public_artifacts.py",
        str(artifact),
        "--json-output",
        str(report),
    )

    assert result.returncode == 1
    serialized = report.read_text(encoding="utf-8")
    document = json.loads(serialized)
    assert document["finding_count"] == len(private_values)
    assert {finding["kind"] for finding in document["findings"]} == {"private_local_path"}
    for value in private_values:
        assert value not in result.stdout
        assert value not in result.stderr
        assert value not in serialized


def test_scan_allows_bare_tokenizer_root_tokens_but_blocks_real_descendants(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "/data": 1,
                        "/home": 2,
                        "/root": 3,
                        "/Users": 4,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    allowed = run_script(repository_root, "scan_public_artifacts.py", str(tokenizer))
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr

    private_path = "/" + "data1/runs/model"
    tokenizer.write_text(json.dumps({"model": {"vocab": {private_path: 1}}}))
    blocked = run_script(repository_root, "scan_public_artifacts.py", str(tokenizer))
    assert blocked.returncode == 1
    assert "private_local_path" in blocked.stdout


def test_scan_blocks_fixed_physical_gpu_but_allows_portable_gpu_selection(
    repository_root: Path, tmp_path: Path
) -> None:
    fixed_assignments = (
        "CUDA_VISIBLE_DEVICES=" + "5",
        "export CUDA_VISIBLE_DEVICES='" + "0, 2'",
        "CUDA_VISIBLE_DEVICES=" + "3; python -m pii_zh",
        'command="CUDA_VISIBLE_DEVICES=' + '4"',
        '{"CUDA_VISIBLE_DEVICES": "' + '7"}',
    )
    blocked = tmp_path / "fixed-gpu.txt"
    blocked.write_text("\n".join(fixed_assignments) + "\n", encoding="utf-8")
    report = tmp_path / "fixed-gpu-scan.json"

    result = run_script(
        repository_root,
        "scan_public_artifacts.py",
        str(blocked),
        "--json-output",
        str(report),
    )

    assert result.returncode == 1
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["finding_count"] == len(fixed_assignments)
    assert {finding["kind"] for finding in document["findings"]} == {"fixed_physical_gpu"}

    portable = tmp_path / "portable-roots-and-gpu.txt"
    portable.write_text(
        "\n".join(
            (
                'cd "$REPO_ROOT"',
                'model="$MODEL_ROOT/pii-model"',
                'dataset="$DATA_ROOT/evaluation.jsonl"',
                'output="$RUN_ROOT/replay"',
                "CUDA_VISIBLE_DEVICES=<FREE_GPU_ID>",
                'CUDA_VISIBLE_DEVICES="$GPU_ID"',
                "CUDA_VISIBLE_DEVICES=",
                "--device cuda:0",
                "curl http://127.0.0.1:8000/v1/analyze",
                "--tmpfs /tmp:rw,noexec",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    allowed = run_script(repository_root, "scan_public_artifacts.py", str(portable))
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr


def test_scan_allows_secret_taxonomy_probability_only_in_threshold_artifacts(
    repository_root: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "thresholds.yaml").write_text(
        "SECRET: 0.7840\x3220271531302\n"
        "strict_f1: 0.8643655065074376\n"
        "manifest_sha256: 9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d\n",
        encoding="utf-8",
    )
    (artifact / "calibration.json").write_text(
        '{\n  "SECRET": 0.7840220271531302,\n  "strict_f1": 0.8643655065074376\n}\n',
        encoding="utf-8",
    )

    result = run_script(repository_root, "scan_public_artifacts.py", str(artifact))

    assert result.returncode == 0, result.stdout + result.stderr


def test_scan_blocks_decimal_values_assigned_to_explicit_secret_keys(
    repository_root: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    values = (
        "password: 0.7840\x3220271531302\n"
        "secret: 0.8643\x3655065074376\n"
        "auth_token: 0.3141\x3592653589793\n"
        "api_key: 0.2718\x3281828459045\n"
    )
    (artifact / "metrics.yaml").write_text(values, encoding="utf-8")

    result = run_script(repository_root, "scan_public_artifacts.py", str(artifact))

    assert result.returncode == 1
    assert result.stdout.count("assigned_secret") == 4
    for value in (
        "0.7840220271531302",
        "0.8643655065074376",
        "0.3141592653589793",
        "0.2718281828459045",
    ):
        assert value not in result.stdout


def test_scan_still_blocks_integer_assigned_secret_and_luhn_card(
    repository_root: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "unsafe.txt").write_text(
        "password: 123456\x3789012\ncard: 411111\x31111111111\n",
        encoding="utf-8",
    )

    result = run_script(repository_root, "scan_public_artifacts.py", str(artifact))

    assert result.returncode == 1
    assert "assigned_secret" in result.stdout
    assert "potential_payment_card" in result.stdout
    assert "123456789012" not in result.stdout
    assert "411111\x31111111111" not in result.stdout


def _write_safetensors(path: Path, *, metadata: dict[str, str]) -> None:
    header = json.dumps(
        {
            "__metadata__": metadata,
            "weight": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]},
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)


def test_scan_parses_safetensors_metadata_and_redacts_secret_values(
    repository_root: Path, tmp_path: Path
) -> None:
    safe = tmp_path / "safe.safetensors"
    _write_safetensors(safe, metadata={"format": "pt", "model_name": "fixture"})
    safe_result = run_script(repository_root, "scan_public_artifacts.py", str(safe))
    assert safe_result.returncode == 0, safe_result.stdout + safe_result.stderr

    secret = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "1234567890"
    unsafe = tmp_path / "unsafe.safetensors"
    _write_safetensors(unsafe, metadata={"note": secret})
    report = tmp_path / "safetensors-scan.json"
    blocked = run_script(
        repository_root,
        "scan_public_artifacts.py",
        str(unsafe),
        "--json-output",
        str(report),
    )

    assert blocked.returncode == 1
    assert secret not in blocked.stdout
    assert hashlib.sha256(secret.encode()).hexdigest() not in blocked.stdout
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["findings"] == [
        {"kind": "huggingface_token", "line": 1, "path": "unsafe.safetensors"}
    ]


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
