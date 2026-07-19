from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import validate_release_execution_surface_v2 as surface

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "configs/release/release_execution_surface_v2.json"
SCHEMA_PATH = REPOSITORY_ROOT / "configs/release/release_execution_surface_v2.schema.json"
RECEIPT_PATH = REPOSITORY_ROOT / "reports/release_audits/release_execution_surface_v2_pass.json"
EXPECTED_VALIDATOR_FILE_SHA256 = "26ab2fb6628d2db3217bf38f2ce1f3416eb7570ed2b541f73a655dd9839c0617"
EXPECTED_SCHEMA_FILE_SHA256 = "e14fe9147d45fe00f9549488cfdfa6001b97e57f58504cb8177ce1c38d88f162"
EXPECTED_ENTRYPOINTS = {
    "pii-zh": "pii_zh.cli:main",
    "pii-zh-evaluate-ablation": "pii_zh.evaluation.cascade_ablation_cli:main",
}
EXPECTED_DYNAMIC_EXTERNAL_TARGETS = {
    "opf._core.decoding",
    "opf._core.sequence_labeling",
    "opf._core.spans",
}
EXPECTED_ENTRYPOINT_RESOURCES = {
    "src/pii_zh/cascade/profiles/ablation_profiles_v1.json",
    "src/pii_zh/taxonomy/presidio_mapping.yaml",
    "src/pii_zh/taxonomy/taxonomy.yaml",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory() -> tuple[list[dict[str, Any]], dict[str, bytes], list[str], list[str]]:
    pyproject = surface._parse_toml((REPOSITORY_ROOT / "pyproject.toml").read_bytes())
    package_data = surface._package_data_configuration(pyproject)
    return surface._discover_wheel_inventory(
        REPOSITORY_ROOT,
        package_root="src/pii_zh",
        package_data=package_data,
    )


def test_implementation_and_schema_are_stable_v8_aware_successors() -> None:
    schema = _load(SCHEMA_PATH)
    contract = _load(CONTRACT_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    assert _sha256(Path(surface.__file__)) == EXPECTED_VALIDATOR_FILE_SHA256
    assert _sha256(SCHEMA_PATH) == EXPECTED_SCHEMA_FILE_SHA256
    assert contract["implementation"] == {
        "validator": {
            "path": "scripts/validate_release_execution_surface_v2.py",
            "file_sha256": EXPECTED_VALIDATOR_FILE_SHA256,
        },
        "schema": {
            "path": "configs/release/release_execution_surface_v2.schema.json",
            "file_sha256": EXPECTED_SCHEMA_FILE_SHA256,
        },
    }
    source = contract["source_closure"]
    assert source["artifact"]["path"] == (
        "configs/release/successor_rules_only_service_source_closure_v8.json"
    )
    assert source["artifact"]["closure_id"] == ("successor_rules_only_service_source_closure_v8")
    assert source["verifier"]["path"] == ("scripts/build_successor_service_source_closure_v8.py")
    assert source["current_tree_match_required"] is True
    assert source["strict_replay_required"] is True
    assert source["whole_wheel_inventory_binding_required"] is True
    harness = (REPOSITORY_ROOT / "scripts/run_successor_clean_wheel_smoke.py").read_text(
        encoding="utf-8"
    )
    assert '"installed_module": str(module_path)' not in harness
    assert '"installed_module_path_recorded": False' in harness
    assert 'assert "torch" not in sys.modules' not in harness
    assert 'name == "pii_zh.inference"' in harness
    assert 'name == "pii_zh.models"' in harness
    assert '"framework_torch_imported": "torch" in sys.modules' in harness
    assert '"framework_transformers_imported": "transformers" in sys.modules' in harness


def test_all_project_scripts_are_exactly_enumerated() -> None:
    contract = _load(CONTRACT_PATH)
    pyproject = surface._parse_toml((REPOSITORY_ROOT / "pyproject.toml").read_bytes())
    project_scripts = dict(pyproject["project"]["scripts"])
    declared = {item["name"]: item["target"] for item in contract["installed_entrypoints"]}

    assert project_scripts == declared == EXPECTED_ENTRYPOINTS


def test_historical_ast_closure_is_superseded_by_current_runtime_surface() -> None:
    contract = _load(CONTRACT_PATH)
    _, payloads, python_paths, _ = _inventory()
    module_to_path, _ = surface._module_index(python_paths)
    entrypoint_modules = [
        item["target"].partition(":")[0] for item in contract["installed_entrypoints"]
    ]
    api_modules = [item["module"] for item in contract["selected_service_apis"]]

    paths, _ = surface._runtime_import_closure(
        root_modules=[*entrypoint_modules, *api_modules],
        module_to_path=module_to_path,
        payloads=payloads,
    )

    assert len(paths) > 72
    assert "src/pii_zh/cli.py" in paths
    assert "src/pii_zh/evaluation/cascade_ablation_cli.py" in paths
    assert "src/pii_zh/data/tw_pii_bench.py" in paths
    assert "src/pii_zh/service/app.py" in paths
    assert "src/pii_zh/presidio/engine.py" in paths


def test_historical_whole_wheel_inventory_is_immutable_and_current_tree_is_larger() -> None:
    contract = _load(CONTRACT_PATH)
    inventory, payloads, python_paths, package_data_paths = _inventory()
    metadata_payloads = {
        "pyproject.toml": (REPOSITORY_ROOT / "pyproject.toml").read_bytes(),
        **{
            item["path"]: (REPOSITORY_ROOT / item["path"]).read_bytes()
            for item in contract["wheel_metadata_sources"]
        },
    }
    counts, affected = surface._pattern_counts(
        {**payloads, **metadata_payloads}, surface._PRIVATE_ABSOLUTE_PATH_PATTERNS
    )

    assert contract["whole_wheel"]["inventory_file_count"] == 143
    assert contract["whole_wheel"]["python_file_count"] == 136
    assert len(inventory) > contract["whole_wheel"]["inventory_file_count"]
    assert len(python_paths) > contract["whole_wheel"]["python_file_count"]
    assert len(package_data_paths) == contract["whole_wheel"]["package_data_file_count"] == 7
    assert len(metadata_payloads) == contract["whole_wheel"]["metadata_source_file_count"] == 3
    assert surface.wheel_inventory_sha256(inventory) != contract["whole_wheel"]["inventory_sha256"]
    assert sum(counts.values()) == 0
    assert affected == set()


def test_dynamic_import_and_package_data_boundaries_are_explicit() -> None:
    contract = _load(CONTRACT_PATH)
    _, payloads, python_paths, package_data_paths = _inventory()
    dynamic: list[str | None] = []
    for path in python_paths:
        dynamic.extend(surface._dynamic_imports(surface._parse_python(payloads[path], path=path)))

    assert None not in dynamic
    assert set(dynamic) == EXPECTED_DYNAMIC_EXTERNAL_TARGETS
    assert set(contract["dynamic_import_boundary"]["allowed_constant_external_targets"]) == (
        EXPECTED_DYNAMIC_EXTERNAL_TARGETS
    )
    required = {
        item["path"] for item in contract["package_data_boundary"]["entrypoint_required_resources"]
    }
    assert required == EXPECTED_ENTRYPOINT_RESOURCES
    assert required < set(package_data_paths)
    assert contract["package_data_boundary"]["runtime_trace_claimed"] is False


def test_schema_rejects_unenumerated_entrypoint_or_unbound_source_closure() -> None:
    schema = _load(SCHEMA_PATH)
    contract = copy.deepcopy(_load(CONTRACT_PATH))
    contract["installed_entrypoints"].pop()
    contract["source_closure"]["strict_replay_required"] = False
    contract["source_closure"]["artifact"]["path"] = (
        "configs/release/successor_rules_only_service_source_closure_v7.json"
    )
    contract["contract_sha256"] = surface.canonical_json_hash(contract, remove="contract_sha256")

    error_paths = {
        tuple(error.absolute_path) for error in Draft202012Validator(schema).iter_errors(contract)
    }

    assert ("installed_entrypoints",) in error_paths
    assert ("source_closure", "strict_replay_required") in error_paths
    assert ("source_closure", "artifact", "path") in error_paths
    assert contract["contract_sha256"] == surface.canonical_json_hash(
        contract, remove="contract_sha256"
    )


def test_historical_surface_fails_closed_against_current_successor_tree() -> None:
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="content hash"):
        surface.build_report(contract_path=CONTRACT_PATH, schema_path=SCHEMA_PATH)


def test_fake_source_closure_replay_cannot_bypass_historical_tree_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        surface, "_load_source_closure_verifier", lambda *args, **kwargs: lambda *a, **k: {}
    )

    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="content hash"):
        surface.build_report(contract_path=CONTRACT_PATH, schema_path=SCHEMA_PATH)


def test_repository_reader_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    (root / "link.py").symlink_to(target)

    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="unavailable"):
        surface._read_repository_file(root, "link.py", field="symlink")

    real_directory = root / "real"
    real_directory.mkdir()
    (real_directory / "nested.py").write_text("value = 2\n", encoding="utf-8")
    (root / "linked-directory").symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="ancestor"):
        surface._read_repository_file(root, "linked-directory/nested.py", field="ancestor symlink")


def test_contract_mode_gate_rejects_writable_or_symlinked_policy(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text("{}\n", encoding="utf-8")
    policy.chmod(0o644)
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="mode 0444"):
        surface._require_read_only_regular(policy, field="contract")

    policy.chmod(0o444)
    surface._require_read_only_regular(policy, field="contract")
    alias = tmp_path / "policy-link.json"
    alias.symlink_to(policy)
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="mode 0444"):
        surface._require_read_only_regular(alias, field="contract")


def test_build_report_rejects_contract_leaf_and_ancestor_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    schema = root / "schema.json"
    schema.write_text("{}\n", encoding="utf-8")
    real_directory = root / "real"
    real_directory.mkdir()
    real_contract = real_directory / "contract.json"
    real_contract.write_text("{}\n", encoding="utf-8")
    real_contract.chmod(0o444)

    leaf = root / "contract-link.json"
    leaf.symlink_to(real_contract)
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="contract is unavailable"):
        surface.build_report(
            contract_path=leaf,
            schema_path=schema,
            repository_root=root,
        )

    ancestor = root / "linked-directory"
    ancestor.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="contract traverses"):
        surface.build_report(
            contract_path=ancestor / "contract.json",
            schema_path=schema,
            repository_root=root,
        )


def test_build_report_rejects_writable_contract_and_schema_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    contract = root / "contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    schema_target = root / "schema-target.json"
    schema_target.write_text("{}\n", encoding="utf-8")
    schema_link = root / "schema.json"
    schema_link.symlink_to(schema_target)

    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="mode 0444"):
        surface.build_report(
            contract_path=contract,
            schema_path=schema_target,
            repository_root=root,
        )

    contract.chmod(0o444)
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="schema is unavailable"):
        surface.build_report(
            contract_path=contract,
            schema_path=schema_link,
            repository_root=root,
        )


def test_historical_pass_receipt_remains_immutable_but_is_not_current_replay_proof() -> None:
    receipt = _load(RECEIPT_PATH)

    assert stat.S_IMODE(RECEIPT_PATH.stat().st_mode) == 0o444
    assert receipt["receipt_sha256"] == surface.canonical_json_hash(
        receipt, remove="receipt_sha256"
    )
    with pytest.raises(surface.ReleaseExecutionSurfaceV2Error, match="content hash"):
        surface.build_report(contract_path=CONTRACT_PATH, schema_path=SCHEMA_PATH)
    serialized = RECEIPT_PATH.read_text(encoding="utf-8")
    assert "/data1/" not in serialized
    assert "/home/" not in serialized


def test_historical_cli_strict_replay_fails_closed_after_successor_drift() -> None:
    assert (
        surface.main(
            [
                "--contract",
                str(CONTRACT_PATH),
                "--schema",
                str(SCHEMA_PATH),
                "--verify-receipt",
                str(RECEIPT_PATH),
            ]
        )
        == 2
    )
