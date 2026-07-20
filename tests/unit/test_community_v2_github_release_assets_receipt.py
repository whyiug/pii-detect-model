from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError
from scripts import build_community_v2_github_release_assets_receipt as receipt
from scripts import generate_sbom

SOURCE_COMMIT = "a" * 40


def _wheel(path: Path, *, unsafe_text: str | None = None) -> None:
    dist_info = "pii_zh_qwen-0.2.0rc1.dist-info"
    members = {
        "pii_zh/__init__.py": b'__version__ = "0.2.0rc1"\n',
        "pii_zh/cli.py": b"def main():\n    return 0\n",
        "pii_zh/service/app.py": b"APP = True\n",
        "pii_zh/cascade/routing.py": b"ROUTING = True\n",
        "pii_zh/taxonomy/taxonomy.yaml": b"labels: []\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.4\n"
            b"Name: pii-zh-qwen\n"
            b"Version: 0.2.0rc1\n"
            b"Summary: test fixture\n\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/RECORD": b"fixture,sha256=fixture,1\n",
    }
    if unsafe_text is not None:
        members["pii_zh/unsafe.py"] = unsafe_text.encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _smoke_result() -> dict[str, Any]:
    return {
        "status": "PASS",
        "profile_version": "c1-conservative-v2",
        "harness_path": receipt.SMOKE_HARNESS_PATH.as_posix(),
        "harness_file_sha256": "1" * 64,
        "harness_result_sha256": "2" * 64,
        "stdout_sha256": "3" * 64,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr_empty": True,
        "runtime_versions": {
            "python": "3.12.7",
            "pii-zh-qwen": "0.2.0rc1",
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
    }


def _assets(tmp_path: Path, *, unsafe_wheel_text: str | None = None) -> tuple[Path, Path, Path]:
    wheel = tmp_path / receipt.WHEEL_NAME
    sbom = tmp_path / receipt.SBOM_NAME
    checksums = tmp_path / receipt.CHECKSUMS_NAME
    _wheel(wheel, unsafe_text=unsafe_wheel_text)
    bom = generate_sbom.build_sbom(
        receipt.REPOSITORY_ROOT / receipt.LOCKFILE_PATH,
        receipt.REPOSITORY_ROOT / receipt.PYPROJECT_PATH,
    )
    sbom.write_text(json.dumps(bom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {receipt.WHEEL_NAME}\n"
        f"{hashlib.sha256(sbom.read_bytes()).hexdigest()}  {receipt.SBOM_NAME}\n",
        encoding="utf-8",
    )
    return wheel, sbom, checksums


@pytest.fixture(autouse=True)
def _local_source_and_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(receipt, "_verify_source_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        receipt,
        "_run_isolated_wheel_smoke",
        lambda *_args, **_kwargs: _smoke_result(),
    )


def _prepare(
    tmp_path: Path,
    *,
    wheel: Path | None = None,
    sbom: Path | None = None,
    checksums: Path | None = None,
    output: Path | None = None,
) -> receipt.ReceiptPlan:
    if wheel is None or sbom is None or checksums is None:
        wheel, sbom, checksums = _assets(tmp_path)
    return receipt.prepare_receipt(
        wheel=wheel,
        sbom=sbom,
        checksums=checksums,
        source_commit=SOURCE_COMMIT,
        output=output or tmp_path / "release-assets-receipt.json",
    )


def _schema() -> dict[str, Any]:
    return json.loads((receipt.REPOSITORY_ROOT / receipt.SCHEMA_PATH).read_text(encoding="utf-8"))


def test_builds_closed_self_hashed_read_only_receipt(tmp_path: Path) -> None:
    wheel, sbom, checksums = _assets(tmp_path)
    plan = _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    receipt.write_receipt(plan)

    document = json.loads(plan.output.read_text(encoding="utf-8"))
    Draft202012Validator(_schema()).validate(document)
    assert document["source_commit"] == SOURCE_COMMIT
    assert document["source"]["git_source_commit"] == SOURCE_COMMIT
    assert document["release"]["asset_names"] == list(receipt.ASSET_NAMES)
    assert document["assets"]["wheel"] == {
        "name": receipt.WHEEL_NAME,
        "media_type": "application/zip",
        "file_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "size_bytes": wheel.stat().st_size,
    }
    assert (
        document["assets"]["sbom"]["file_sha256"] == hashlib.sha256(sbom.read_bytes()).hexdigest()
    )
    assert (
        document["assets"]["checksums"]["file_sha256"]
        == hashlib.sha256(checksums.read_bytes()).hexdigest()
    )
    assert document["checksum_closure"]["ordered_entries"] == [
        {
            "name": receipt.WHEEL_NAME,
            "file_sha256": document["assets"]["wheel"]["file_sha256"],
        },
        {
            "name": receipt.SBOM_NAME,
            "file_sha256": document["assets"]["sbom"]["file_sha256"],
        },
    ]
    assert document["wheel_inventory"]["status"] == "PASS"
    assert document["wheel_inventory"]["member_count"] == len(
        document["wheel_inventory"]["members"]
    )
    assert document["wheel_inventory"]["member_inventory_sha256"] == (
        receipt.canonical_json_hash(document["wheel_inventory"]["members"])
    )
    assert document["isolated_wheel_smoke"]["status"] == "PASS"
    assert document["public_artifact_scan"] == {
        **document["public_artifact_scan"],
        "status": "PASS",
        "finding_count": 0,
        "finding_kinds": [],
    }
    assert (
        document["implementation"]["generator_file_sha256"]
        == hashlib.sha256(
            (receipt.REPOSITORY_ROOT / receipt.GENERATOR_PATH).read_bytes()
        ).hexdigest()
    )
    assert document["receipt_sha256"] == receipt.canonical_json_hash(
        document, remove="receipt_sha256"
    )
    assert stat.S_IMODE(plan.output.stat().st_mode) == 0o444
    assert str(tmp_path) not in plan.output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("payload", "blocker_id"),
    [
        ("{wheel}\n", "CHECKSUM_CLOSURE_INVALID"),
        ("{wheel}\n{sbom}\n{wheel}\n", "CHECKSUM_CLOSURE_INVALID"),
        ("{sbom}\n{wheel}\n", "CHECKSUM_BINDING_MISMATCH"),
        ("{wheel_single_space}\n{sbom}\n", "CHECKSUM_FORMAT_INVALID"),
        ("{wheel}\r\n{sbom}\r\n", "CHECKSUM_FORMAT_INVALID"),
        ("{wrong_wheel}\n{sbom}\n", "CHECKSUM_BINDING_MISMATCH"),
    ],
)
def test_checksums_are_an_exact_ordered_two_line_closure(
    tmp_path: Path, payload: str, blocker_id: str
) -> None:
    wheel, sbom, checksums = _assets(tmp_path)
    wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    sbom_digest = hashlib.sha256(sbom.read_bytes()).hexdigest()
    rendered = payload.format(
        wheel=f"{wheel_digest}  {receipt.WHEEL_NAME}",
        wheel_single_space=f"{wheel_digest} {receipt.WHEEL_NAME}",
        wrong_wheel=f"{'0' * 64}  {receipt.WHEEL_NAME}",
        sbom=f"{sbom_digest}  {receipt.SBOM_NAME}",
    )
    checksums.write_bytes(rendered.encode())

    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    assert captured.value.blocker_id == blocker_id


def test_rejects_symlink_inputs_and_nonfixed_names(tmp_path: Path) -> None:
    wheel, sbom, checksums = _assets(tmp_path)
    symlink = tmp_path / "link.whl"
    symlink.symlink_to(wheel)
    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        receipt.prepare_receipt(
            wheel=symlink,
            sbom=sbom,
            checksums=checksums,
            source_commit=SOURCE_COMMIT,
            output=tmp_path / "receipt.json",
        )
    assert captured.value.blocker_id == "ASSET_NAME_MISMATCH"

    wheel.unlink()
    wheel.symlink_to(symlink)
    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    assert captured.value.blocker_id == "SYMLINK_REJECTED"


def test_sbom_must_exactly_regenerate_from_source(tmp_path: Path) -> None:
    wheel, sbom, checksums = _assets(tmp_path)
    document = json.loads(sbom.read_text(encoding="utf-8"))
    document["version"] = 2
    sbom.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {receipt.WHEEL_NAME}\n"
        f"{hashlib.sha256(sbom.read_bytes()).hexdigest()}  {receipt.SBOM_NAME}\n",
        encoding="utf-8",
    )

    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    assert captured.value.blocker_id == "SBOM_SOURCE_MISMATCH"


def test_public_scan_executes_on_decompressed_wheel_without_echoing_secret(
    tmp_path: Path,
) -> None:
    secret = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    wheel, sbom, checksums = _assets(tmp_path, unsafe_wheel_text=f"token={secret}\n")

    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    assert captured.value.blocker_id == "PUBLIC_SCAN_FINDINGS"
    assert "huggingface_token" in str(captured.value)
    assert secret not in str(captured.value)


def test_smoke_failure_blocks_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel, sbom, checksums = _assets(tmp_path)

    def blocked(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise receipt.ReleaseAssetsReceiptError(
            "ISOLATED_SMOKE_FAILED", "isolated wheel smoke failed closed"
        )

    monkeypatch.setattr(receipt, "_run_isolated_wheel_smoke", blocked)
    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    assert captured.value.blocker_id == "ISOLATED_SMOKE_FAILED"


def test_write_rechecks_assets_and_refuses_clobber(
    tmp_path: Path,
) -> None:
    wheel, sbom, checksums = _assets(tmp_path)
    plan = _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        receipt.write_receipt(plan)
    assert captured.value.blocker_id == "INPUT_CHANGED_BEFORE_WRITE"
    assert not plan.output.exists()

    wheel, sbom, checksums = _assets(tmp_path)
    plan = _prepare(tmp_path, wheel=wheel, sbom=sbom, checksums=checksums)
    plan.output.write_text("owned by caller\n", encoding="utf-8")
    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        receipt.write_receipt(plan)
    assert captured.value.blocker_id == "OUTPUT_ALREADY_EXISTS"
    assert plan.output.read_text(encoding="utf-8") == "owned by caller\n"


def test_closed_schema_rejects_extra_property(tmp_path: Path) -> None:
    plan = _prepare(tmp_path)
    document = dict(plan.document)
    document["caller_pass"] = True
    document["receipt_sha256"] = receipt.canonical_json_hash(document, remove="receipt_sha256")
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(document)


def test_dry_run_executes_verification_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel, sbom, checksums = _assets(tmp_path)
    output = tmp_path / "receipt.json"
    result = receipt.main(
        [
            "--wheel",
            str(wheel),
            "--sbom",
            str(sbom),
            "--checksums",
            str(checksums),
            "--source-commit",
            SOURCE_COMMIT,
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["status"] == "READY"
    assert report["receipt_written"] is False
    assert report["wheel_inventory_status"] == "PASS"
    assert report["isolated_wheel_smoke_status"] == "PASS"
    assert report["public_artifact_scan_status"] == "PASS"
    assert report["network_used"] is False
    assert report["gpu_used"] is False
    assert not output.exists()


def test_smoke_result_parser_requires_exact_pass_contract() -> None:
    raw = {
        "status": "PASS",
        "profile_version": "c1-conservative-v2",
        "historical_default_unchanged": True,
        "installed_from_clean_site_packages": True,
        "installed_module_path_recorded": False,
        "python": "3.12.7",
        "distribution_versions": {
            "pii-zh-qwen": "0.2.0rc1",
            "presidio-analyzer": "2.2.362",
            "fastapi": "0.116.0",
            "httpx": "0.28.1",
        },
        "paths": [
            "python",
            "installed-cli",
            "installed-ablation-cli-help",
            "presidio",
            "http",
        ],
        "installed_entrypoints": {
            "pii-zh": "pii_zh.cli:main",
            "pii-zh-evaluate-ablation": "pii_zh.evaluation.cascade_ablation_cli:main",
        },
        "historical_paths": ["python", "installed-cli", "presidio", "http"],
        "valid_case_count": 3,
        "invalid_document_count": 1,
        "raw_values_persisted": False,
        "model_packages_imported": False,
        "framework_torch_imported": False,
        "framework_transformers_imported": False,
        "gpu_visible": "",
    }
    stdout = (json.dumps(raw, sort_keys=True) + "\n").encode()
    evidence = receipt._validate_smoke_result(stdout, b"")
    assert evidence["status"] == "PASS"
    assert evidence["harness_result_sha256"] == receipt.canonical_json_hash(raw)

    raw["framework_torch_imported"] = True
    with pytest.raises(receipt.ReleaseAssetsReceiptError) as captured:
        receipt._validate_smoke_result((json.dumps(raw) + "\n").encode(), b"")
    assert captured.value.blocker_id == "ISOLATED_SMOKE_ASSERTION_FAILED"
