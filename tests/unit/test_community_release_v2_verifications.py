from __future__ import annotations

import copy
import csv
import io
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import produce_community_cascade_release_v2_verifications as verification

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_subject(path: Path) -> None:
    document: dict[str, Any] = {
        "schema_version": "pii-zh.fixture-source.v1",
        "inventory_sha256": "1" * 64,
        "manifest_sha256": "",
    }
    document["manifest_sha256"] = verification.canonical_json_hash(
        document, remove="manifest_sha256"
    )
    path.write_text(json.dumps(document, sort_keys=True) + "\n")


def _unit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, return_code: int = 0
) -> dict[str, Any]:
    subject = tmp_path / "source.json"
    _write_subject(subject)
    manifest = verification._current_rc_regression_input_manifest(REPOSITORY_ROOT)
    monkeypatch.setattr(
        verification,
        "_execute_current_rc_regression",
        lambda **_kwargs: verification._ExecutionOutput(
            return_code,
            b"42 passed\n" if return_code == 0 else b"",
            b"failed",
            regression_input_manifest_sha256=manifest["manifest_sha256"],
            regression_input_manifest_file_count=manifest["file_count"],
        ),
    )
    return verification._execute_fixed_check_and_issue_receipt(
        SimpleNamespace(check_id="unit_tests", service_source_manifest=subject)
    )


def test_verification_schema_receipt_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = json.loads((REPOSITORY_ROOT / verification.SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    assert not hasattr(verification, "build_verification_receipt")
    assert not hasattr(verification, "ExecutionOutput")
    receipt = _unit_receipt(tmp_path, monkeypatch)
    Draft202012Validator(schema).validate(receipt)
    verification.validate_verification_receipt(receipt, check_id="unit_tests")
    assert receipt["status"] == "PASS"
    assert receipt["execution"]["return_code"] == 0
    assert receipt["execution"]["harness_id"] == "community_release_current_rc_regression_v2"
    assert receipt["execution"]["regression_input_manifest_file_count"] == len(
        verification.CURRENT_COMMUNITY_RC_TEST_PATHS
        + verification.CURRENT_COMMUNITY_RC_TEST_SUPPORT_PATHS
        + verification.CURRENT_COMMUNITY_RC_IMPLEMENTATION_PATHS
    )
    assert len(receipt["execution"]["regression_input_manifest_sha256"]) == 64
    assert receipt["assertions"]["caller_supplied_pass_or_hash"] is False

    output = tmp_path / "unit-tests.json"
    verification.publish_receipt(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(verification.CommunityVerificationError, match="overwrite"):
        verification.publish_receipt(output, receipt)


def test_old_handwritten_v1_or_forged_execution_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = {
        "schema_version": "pii-zh.community-verification-receipt.v1",
        "check_id": "unit_tests",
        "status": "PASS",
        "scope": "local_non_production",
        "contains_raw_records": False,
        "evidence_sha256": "0" * 64,
    }
    with pytest.raises(verification.CommunityContractError):
        verification.validate_verification_receipt(old, check_id="unit_tests")

    forged = copy.deepcopy(_unit_receipt(tmp_path, monkeypatch))
    forged["execution"]["harness_file_sha256"] = "0" * 64
    forged["execution"]["implementation_identity_sha256"] = "0" * 64
    forged["receipt_sha256"] = verification.canonical_json_hash(forged, remove="receipt_sha256")
    with pytest.raises(verification.CommunityVerificationError, match="identity"):
        verification.validate_verification_receipt(forged, check_id="unit_tests")


def test_nonzero_fixed_harness_cannot_emit_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(verification.CommunityVerificationError, match="return code"):
        _unit_receipt(tmp_path, monkeypatch, return_code=1)


def test_current_rc_regression_uses_closed_file_allowlist_and_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "tests" / "current" / "test_first.py"
    second = tmp_path / "tests" / "current" / "test_second.py"
    support = tmp_path / "pyproject.toml"
    first.parent.mkdir(parents=True)
    first.write_text("def test_first():\n    assert True\n", encoding="utf-8")
    second.write_text("def test_second():\n    assert True\n", encoding="utf-8")
    support.write_text("[tool.pytest.ini_options]\naddopts = '-q'\n", encoding="utf-8")
    monkeypatch.setattr(
        verification,
        "CURRENT_COMMUNITY_RC_TEST_PATHS",
        ("tests/current/test_first.py", "tests/current/test_second.py"),
    )
    monkeypatch.setattr(
        verification, "CURRENT_COMMUNITY_RC_TEST_SUPPORT_PATHS", ("pyproject.toml",)
    )

    monkeypatch.setattr(verification, "CURRENT_COMMUNITY_RC_IMPLEMENTATION_PATHS", ())

    before = verification._current_rc_regression_input_manifest(tmp_path)
    assert before["test_paths"] == [
        "tests/current/test_first.py",
        "tests/current/test_second.py",
    ]
    assert before["support_paths"] == ["pyproject.toml"]
    assert before["file_count"] == 3
    assert set(before["files"]) == {
        "tests/current/test_first.py",
        "tests/current/test_second.py",
        "pyproject.toml",
    }

    first.write_text("def test_first():\n    assert 1 == 1\n", encoding="utf-8")
    after = verification._current_rc_regression_input_manifest(tmp_path)
    assert after["manifest_sha256"] != before["manifest_sha256"]


def test_current_rc_regression_fails_closed_on_mid_run_input_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_path = tmp_path / "tests" / "current" / "test_release.py"
    support_path = tmp_path / "pyproject.toml"
    implementation_path = tmp_path / "scripts" / "release_control.py"
    test_path.parent.mkdir(parents=True)
    implementation_path.parent.mkdir(parents=True)
    test_path.write_text("def test_release():\n    assert True\n", encoding="utf-8")
    support_path.write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    implementation_path.write_text("STATUS = 'stable'\n", encoding="utf-8")
    monkeypatch.setattr(
        verification,
        "CURRENT_COMMUNITY_RC_TEST_PATHS",
        ("tests/current/test_release.py",),
    )
    monkeypatch.setattr(
        verification, "CURRENT_COMMUNITY_RC_TEST_SUPPORT_PATHS", ("pyproject.toml",)
    )
    monkeypatch.setattr(
        verification,
        "CURRENT_COMMUNITY_RC_IMPLEMENTATION_PATHS",
        ("scripts/release_control.py",),
    )

    def mutate(*_args: Any, **_kwargs: Any) -> verification._ExecutionOutput:
        implementation_path.write_text("STATUS = 'changed'\n", encoding="utf-8")
        return verification._ExecutionOutput(0, b"pass\n", b"")

    monkeypatch.setattr(verification, "_run", mutate)
    with pytest.raises(verification.CommunityVerificationError, match="changed during"):
        verification._execute_current_rc_regression(repository_root=tmp_path)


def test_current_rc_regression_argv_contains_only_explicit_test_files() -> None:
    argv = verification.CURRENT_RC_REGRESSION_SEMANTIC_ARGV
    assert argv[:4] == ("python", "-m", "pytest", "-q")
    assert argv[4:] == verification.CURRENT_COMMUNITY_RC_TEST_PATHS
    assert all(path.endswith(".py") for path in argv[4:])
    assert "tests/unit" not in argv
    assert "tests/service" not in argv
    assert "tests/release" not in argv


def test_subject_inventory_is_check_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = copy.deepcopy(_unit_receipt(tmp_path, monkeypatch))
    receipt["subjects"] = {"model_package_manifest": next(iter(receipt["subjects"].values()))}
    receipt["receipt_sha256"] = verification.canonical_json_hash(receipt, remove="receipt_sha256")
    with pytest.raises(verification.CommunityVerificationError, match="inventory"):
        verification.validate_verification_receipt(receipt, check_id="unit_tests")


@pytest.mark.parametrize(
    ("check_id", "expected_subjects"),
    [
        (
            "offline_model_smoke",
            {"model_package_manifest", "wheel_manifest", "wheelhouse_manifest"},
        ),
        (
            "offline_service_smoke",
            {
                "model_package_manifest",
                "wheel_manifest",
                "wheelhouse_manifest",
                "service_configuration_manifest",
                "service_source_manifest",
                "calibration_bundle",
            },
        ),
        (
            "container_smoke",
            {
                "container_manifest",
                "model_package_manifest",
                "wheel_manifest",
                "wheelhouse_manifest",
                "service_configuration_manifest",
                "service_source_manifest",
                "calibration_bundle",
            },
        ),
    ],
)
def test_fixed_runtime_branches_bind_complete_subject_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check_id: str,
    expected_subjects: set[str],
) -> None:
    paths: dict[str, Path] = {}
    for subject_id in expected_subjects:
        path = tmp_path / f"{subject_id}.json"
        _write_subject(path)
        paths[subject_id] = path
    wheel = tmp_path / "release.whl"
    wheel.write_bytes(b"wheel")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "dependency.whl").write_bytes(b"dependency")
    model_root = tmp_path / "model"
    model_root.mkdir()

    def typed(_path: Path, *, artifact_id: str) -> dict[str, Any]:
        if artifact_id == "container_manifest":
            return {
                "inputs": {
                    "wheelhouse_manifest": verification.artifacts._metadata_binding(
                        paths["wheelhouse_manifest"], field="fixture wheelhouse manifest"
                    )
                },
                "result": {
                    "image_id": "sha256:" + "1" * 64,
                    "bound_wheelhouse_artifact_sha256": "2" * 64,
                },
            }
        if artifact_id == "wheelhouse_manifest":
            return {"artifact_sha256": "2" * 64, "result": {}}
        return {"inputs": {"wheel": {}}, "result": {"files": {}}}

    monkeypatch.setattr(verification, "_verify_typed_artifact", typed)
    monkeypatch.setattr(verification, "_verify_wheel_against_manifest", lambda *args: None)
    monkeypatch.setattr(verification, "_verify_model_root_against_manifest", lambda *args: None)
    monkeypatch.setattr(verification, "_verify_service_inputs", lambda **kwargs: None)
    output = verification._ExecutionOutput(0, b"fixed command passed\n", b"")
    monkeypatch.setattr(verification, "_execute_model", lambda **kwargs: output)
    monkeypatch.setattr(verification, "_execute_service", lambda **kwargs: output)
    monkeypatch.setattr(verification, "_execute_container", lambda **kwargs: output)
    namespace = SimpleNamespace(
        check_id=check_id,
        model_package_manifest=paths.get("model_package_manifest"),
        wheel_manifest=paths.get("wheel_manifest"),
        wheelhouse_manifest=paths.get("wheelhouse_manifest"),
        container_manifest=paths.get("container_manifest"),
        service_configuration_manifest=paths.get("service_configuration_manifest"),
        service_source_manifest=paths.get("service_source_manifest"),
        calibration_bundle=paths.get("calibration_bundle"),
        wheel=wheel,
        wheelhouse=wheelhouse,
        model_root=model_root,
    )
    receipt = verification._execute_fixed_check_and_issue_receipt(namespace)
    assert set(receipt["subjects"]) == expected_subjects
    verification.validate_verification_receipt(receipt, check_id=check_id)


def test_model_and_service_smokes_use_fresh_wheel_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "release.whl"
    wheel.write_bytes(b"wheel")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "dependency.whl").write_bytes(b"dependency")
    model_root = tmp_path / "model"
    model_root.mkdir()
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    commands: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> verification._ExecutionOutput:
        commands.append(list(argv))
        return verification._ExecutionOutput(0, b"pass\n", b"")

    monkeypatch.setattr(verification, "_run", run)
    for check_id in ("offline_model_smoke", "offline_service_smoke"):
        result = verification._execute_in_clean_venv(
            check_id=check_id,
            wheel_path=wheel,
            wheelhouse=wheelhouse,
            model_root=model_root,
            calibration_bundle=(calibration if check_id == "offline_service_smoke" else None),
            repository_root=REPOSITORY_ROOT,
        )
        assert result.return_code == 0
    pip_commands = [command for command in commands if "pip" in command]
    assert len(pip_commands) == 2
    assert all("--no-index" in command and "--find-links" in command for command in pip_commands)
    assert pip_commands[0][-1].endswith("[inference]")
    assert pip_commands[1][-1].endswith("[cascade,service]")
    smoke_commands = [
        command
        for command in commands
        if verification.MODEL_SMOKE_CODE in command or verification.SERVICE_SMOKE_CODE in command
    ]
    assert len(smoke_commands) == 2
    assert all(command[0] != str(Path(verification.sys.executable)) for command in smoke_commands)


def test_offline_model_smoke_loads_packaged_transformers_remote_code() -> None:
    code = verification.MODEL_SMOKE_CODE
    assert "load_local_predictor" not in code
    assert "AutoTokenizer.from_pretrained" in code
    assert "AutoConfig.from_pretrained" in code
    assert "AutoModelForTokenClassification.from_pretrained" in code
    assert code.count("local_files_only=True") == 3
    assert code.count("trust_remote_code=True") == 3
    assert "pii_release_eligible is False" in code
    assert 'pii_attention_mode == "full"' in code
    assert "num_labels == 49" in code
    assert "logits.shape[-1] == 49" in code
    assert "torch.isfinite(logits).all().item()" in code


def test_offline_environment_isolates_transformers_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-source-tree")
    monkeypatch.setenv("PYTHONHOME", "/tmp/untrusted-python-home")
    monkeypatch.setenv("PYTHONUSERBASE", "/tmp/untrusted-user-base")
    environment = verification._offline_environment(cache_root=tmp_path / "cache")
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_HOME"].startswith(str(tmp_path))
    assert environment["HF_MODULES_CACHE"].startswith(str(tmp_path))
    assert environment["XDG_CACHE_HOME"].startswith(str(tmp_path))
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONUSERBASE" not in environment


def test_clean_wheel_harness_executes_real_offline_venv_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
    dist_info = "pii_zh_qwen-0.2.0rc1.dist-info"
    members = {
        "pii_zh/__init__.py": b"installed_fixture = True\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.4\nName: pii-zh-qwen\nVersion: 0.2.0rc1\n"
            b"Provides-Extra: cascade\nProvides-Extra: service\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: fixed-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record_stream = io.StringIO()
    writer = csv.writer(record_stream, lineterminator="\n")
    for name in members:
        writer.writerow((name, "", ""))
    writer.writerow((f"{dist_info}/RECORD", "", ""))
    members[f"{dist_info}/RECORD"] = record_stream.getvalue().encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with zipfile.ZipFile(wheelhouse / "fixture-1.0-py3-none-any.whl", "w") as archive:
        archive.writestr(
            "fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: fixture\nVersion: 1.0\n",
        )
    harness = tmp_path / "smoke.py"
    harness.write_text("import pii_zh\nassert pii_zh.installed_fixture\n")
    monkeypatch.setattr(verification, "CLEAN_WHEEL_HARNESS_PATH", str(harness))

    output = verification._execute_in_clean_venv(
        check_id="clean_wheel_smoke",
        wheel_path=wheel,
        wheelhouse=wheelhouse,
        model_root=None,
        calibration_bundle=None,
        repository_root=REPOSITORY_ROOT,
    )
    assert output.return_code == 0, output.stderr.decode(errors="replace")


def test_wheelhouse_physical_inventory_must_match_typed_manifest(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    dependency = wheelhouse / "demo_dependency-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(dependency, "w") as archive:
        archive.writestr(
            "demo_dependency-1.2.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: demo-dependency\nVersion: 1.2.3\n",
        )
        archive.writestr("demo_dependency/__init__.py", "safe = True\n")
    files, packages = verification.artifacts._wheelhouse_inventory(wheelhouse)
    manifest = {
        "result": {
            "files": files,
            "packages": packages,
            "inventory_sha256": verification.canonical_json_hash(files),
            "packages_sha256": verification.canonical_json_hash(packages),
        }
    }
    verification._verify_wheelhouse_against_manifest(wheelhouse, manifest)

    with zipfile.ZipFile(dependency, "a") as archive:
        archive.writestr("demo_dependency/new.py", "drift = True\n")
    with pytest.raises(verification.CommunityVerificationError, match="differs"):
        verification._verify_wheelhouse_against_manifest(wheelhouse, manifest)


def test_container_smoke_mounts_exact_public_inputs_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "release.whl"
    wheel.write_bytes(b"wheel")
    model = tmp_path / "model"
    model.mkdir()
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    wheelhouse_manifest = tmp_path / "wheelhouse-manifest.json"
    wheelhouse_manifest.write_text("{}")
    captured: list[str] = []

    def run(argv: list[str], **_kwargs: Any) -> verification._ExecutionOutput:
        captured.extend(argv)
        return verification._ExecutionOutput(0, b"pass\n", b"")

    monkeypatch.setattr(verification, "_run", run)
    result = verification._execute_container(
        image_ref="sha256:" + "1" * 64,
        wheel_path=wheel,
        wheelhouse_manifest_path=wheelhouse_manifest,
        model_root=model,
        calibration_bundle=calibration,
        repository_root=REPOSITORY_ROOT,
    )
    assert result.return_code == 0
    assert "--network" in captured and "none" in captured
    assert "--read-only" in captured
    assert "--tmpfs" in captured
    assert "/tmp:rw,noexec,nosuid,size=64m" in captured
    assert "--tmpfs" in verification.CONTAINER_SEMANTIC_ARGV
    assert "/tmp:rw,noexec,nosuid,size=64m" in verification.CONTAINER_SEMANTIC_ARGV
    assert sum("readonly" in item for item in captured) == 4
    assert "/release/wheelhouse-manifest.json" in " ".join(captured)
    assert verification.CONTAINER_SMOKE_CODE in captured
    assert "importlib.metadata.version" in verification.CONTAINER_SMOKE_CODE
