#!/usr/bin/env python3
"""Run fixed local release checks and emit immutable verification receipts.

Callers provide only the release subjects needed by a named check.  They cannot
provide a command, status, return code, log digest, or PASS value.  Harness
commands, semantic argv identities and output normalization are fixed here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts import produce_community_cascade_release_v2_artifacts as artifacts
    from scripts.community_contract_utils import (
        CommunityContractError,
        canonical_json_hash,
        load_json_path,
        read_regular_file,
        sha256_bytes,
        strict_json_bytes,
        validate_schema,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    import produce_community_cascade_release_v2_artifacts as artifacts  # type: ignore[no-redef]
    from community_contract_utils import (  # type: ignore[no-redef]
        CommunityContractError,
        canonical_json_hash,
        load_json_path,
        read_regular_file,
        sha256_bytes,
        strict_json_bytes,
        validate_schema,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATH = "scripts/produce_community_cascade_release_v2_verifications.py"
SCHEMA_PATH = "configs/release/community_cascade_release_v2.verification-receipt.schema.json"
SCHEMA_VERSION = "pii-zh.community-verification-receipt.v2"
CLEAN_WHEEL_HARNESS_PATH = "scripts/run_successor_clean_wheel_smoke.py"

CHECK_IDS = (
    "unit_tests",
    "clean_wheel_smoke",
    "container_smoke",
    "offline_model_smoke",
    "offline_service_smoke",
)
HARNESS_IDS = {
    "unit_tests": "community_release_current_rc_regression_v2",
    "clean_wheel_smoke": "community_release_clean_wheel_v1",
    "container_smoke": "community_release_offline_container_v1",
    "offline_model_smoke": "community_release_local_model_v1",
    "offline_service_smoke": "community_release_inprocess_http_v1",
}
EXPECTED_SUBJECT_IDS = {
    "unit_tests": frozenset({"service_source_manifest"}),
    "clean_wheel_smoke": frozenset({"wheel_manifest", "wheelhouse_manifest"}),
    "container_smoke": frozenset(
        {
            "container_manifest",
            "wheel_manifest",
            "wheelhouse_manifest",
            "model_package_manifest",
            "service_configuration_manifest",
            "service_source_manifest",
            "calibration_bundle",
        }
    ),
    "offline_model_smoke": frozenset(
        {"model_package_manifest", "wheel_manifest", "wheelhouse_manifest"}
    ),
    "offline_service_smoke": frozenset(
        {
            "model_package_manifest",
            "wheel_manifest",
            "wheelhouse_manifest",
            "service_configuration_manifest",
            "service_source_manifest",
            "calibration_bundle",
        }
    ),
}
EXPECTED_RELEASE = {
    "display_name": artifacts.DISPLAY_NAME,
    "package_version": artifacts.PACKAGE_VERSION,
    "publication_state": artifacts.PUBLICATION_STATE,
}
EXPECTED_ASSERTIONS = {
    "caller_supplied_command": False,
    "caller_supplied_pass_or_hash": False,
    "external_network_requested_by_harness": False,
    "formal_record_level_data_opened": False,
    "gpu_required": False,
    "synthetic_smoke_input_only": True,
}
EXPECTED_PRIVACY = {
    "contains_local_paths": False,
    "contains_raw_records": False,
    "contains_synthetic_fixture_text": False,
    "contains_secrets": False,
    "contains_unredacted_logs": False,
}
MAX_LOG_BYTES = 64 * 1024 * 1024

# This is deliberately an exact file allowlist rather than ``tests/unit`` (or
# the entire repository test tree).  Historical immutable gates remain useful
# evidence, but many of them intentionally reject the current successor source
# tree.  The community RC receipt therefore attests only this current-release
# regression surface and never claims that every historical test is green.
CURRENT_COMMUNITY_RC_TEST_PATHS = (
    "tests/cli/test_cli.py",
    "tests/service/test_local_service.py",
    "tests/unit/calibration/test_calibrate_predictions_cli.py",
    "tests/unit/calibration/test_calibration.py",
    "tests/unit/calibration/test_span_thresholds.py",
    "tests/unit/cascade/test_ablation_confirmatory_contracts.py",
    "tests/unit/cascade/test_ablation_profiles.py",
    "tests/unit/cascade/test_community_full24.py",
    "tests/unit/cascade/test_config.py",
    "tests/unit/cascade/test_import_boundary.py",
    "tests/unit/cascade/test_pipeline.py",
    "tests/unit/cascade/test_redaction_coverage.py",
    "tests/unit/cascade/test_result.py",
    "tests/unit/cascade/test_service_profiles.py",
    "tests/unit/fusion/test_deterministic.py",
    "tests/unit/presidio/test_context_enhancer.py",
    "tests/unit/presidio/test_qwen_recognizer.py",
    "tests/unit/presidio/test_token_chunker.py",
    "tests/unit/rules/test_cn_common.py",
    "tests/unit/rules/test_cn_common_v6.py",
    "tests/unit/rules/test_rules_refinement_cli.py",
    "tests/unit/test_community_rc_summary.py",
    "tests/unit/test_community_release_contracts_v2.py",
    "tests/unit/test_community_release_v2_artifacts.py",
    "tests/unit/test_community_release_v2_evidence.py",
    "tests/unit/test_community_release_v2_verifications.py",
    "tests/unit/test_benchmark_cascade.py",
    "tests/unit/test_build_service_quality_suite.py",
    "tests/unit/test_cascade_ablation_cli.py",
    "tests/unit/test_evaluate_cascade.py",
    "tests/unit/test_inference.py",
    "tests/unit/test_inference_postprocessing.py",
    "tests/unit/test_service_eval_cli.py",
    "tests/unit/test_service_quality_suite.py",
    "tests/unit/test_taxonomy.py",
    "tests/unit/test_tokenization.py",
    "tests/unit/test_release_eval_v2_calibration_service.py",
    "tests/unit/test_release_eval_v2_candidate_generator.py",
    "tests/unit/test_release_eval_v2_performance_harness.py",
    "tests/unit/test_release_eval_v2_prediction_provenance.py",
    "tests/unit/test_release_eval_v2_stage_gate.py",
    "tests/unit/data/test_synthetic_sota_release_eval_v1.py",
    "tests/unit/data/test_synthetic_sota_release_eval_v2.py",
    "tests/unit/data/test_synthetic_sota_v1.py",
    "tests/unit/data/test_synthetic_sota_v2_resplit.py",
    "tests/unit/data/test_validators.py",
    "tests/unit/data/test_windowing.py",
    "tests/unit/models/test_aiguard24.py",
    "tests/release/test_build_release.py",
    "tests/release/test_public_scan_and_sbom.py",
    "tests/release/test_release_execution_surface_v2.py",
    "tests/release/test_templates.py",
)
CURRENT_COMMUNITY_RC_TEST_SUPPORT_PATHS = (
    "pyproject.toml",
    "tests/unit/conftest.py",
    "tests/release/conftest.py",
)
CURRENT_COMMUNITY_RC_IMPLEMENTATION_PATHS = (
    "configs/evaluation/open24_comparator_execution_replay_v1.schema.json",
    "configs/evaluation/open24_comparator_prediction_provenance_v1.schema.json",
    "configs/evaluation/public_synthetic_quality_comparator_registry_v2.schema.json",
    "configs/evaluation/public_synthetic_service_quality_v1.json",
    "configs/evaluation/public_synthetic_service_quality_v1.schema.json",
    "configs/evaluation/public_synthetic_service_quality_result_v2.schema.json",
    "configs/evaluation/release_eval_v2_calibration_authorization_v1.schema.json",
    "configs/evaluation/release_eval_v2_internal_unlock_v1.schema.json",
    "configs/evaluation/release_eval_v2_prediction_provenance_v1.schema.json",
    "configs/release/community_cascade_release_v2.artifact-evidence.schema.json",
    "configs/release/community_cascade_release_v2.candidate-replay-evidence.schema.json",
    "configs/release/community_cascade_release_v2.quality-replay-evidence.schema.json",
    "configs/release/community_cascade_release_v2.receipt.schema.json",
    "configs/release/community_cascade_release_v2.replay-receipt.schema.json",
    "configs/release/community_cascade_release_v2.schema.json",
    "configs/release/community_cascade_release_v2.verification-receipt.schema.json",
    "configs/release/community_cascade_release_v2.json",
    "configs/release/community_service_source_identity_v3.schema.json",
    "configs/release/community_service_source_identity_v4.schema.json",
    "scripts/benchmark_release_eval_v2_candidate.py",
    "scripts/build_model_card_evidence_v2.py",
    "scripts/build_community_cascade_release_v2_contract.py",
    "scripts/build_release.py",
    "scripts/community_contract_utils.py",
    "scripts/generate_sbom.py",
    "scripts/produce_community_cascade_release_v2_artifacts.py",
    "scripts/produce_community_cascade_release_v2_evidence.py",
    "scripts/produce_community_cascade_release_v2_verifications.py",
    "scripts/produce_public_synthetic_service_quality_v2.py",
    "scripts/release_eval_v2_prediction_provenance.py",
    "scripts/release_eval_v2_stage_gate.py",
    "scripts/run_successor_clean_wheel_smoke.py",
    "scripts/scan_public_artifacts.py",
    "scripts/validate_community_cascade_release_v2.py",
    "scripts/validate_public_synthetic_service_quality_v1.py",
    "src/pii_zh/evaluation/release_eval_v2_performance.py",
    "src/pii_zh/evaluation/release_eval_v2_prediction_provenance.py",
    "src/pii_zh/evaluation/release_eval_v2_stage_gate.py",
)
CURRENT_RC_REGRESSION_SEMANTIC_ARGV = (
    "python",
    "-m",
    "pytest",
    "-q",
    *CURRENT_COMMUNITY_RC_TEST_PATHS,
)
CLEAN_WHEEL_SEMANTIC_ARGV = (
    ("python", "-m", "venv", "<TEMP_VENV>"),
    (
        "<TEMP_PYTHON>",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        "<BOUND_WHEELHOUSE>",
        "<RELEASE_WHEEL>[cascade,service]",
    ),
    ("<TEMP_PYTHON>", "-I", "-c", "<FIXED_INSTALLED_WHEEL_ASSERT>", "<RELEASE_WHEEL>"),
    ("<TEMP_PYTHON>", CLEAN_WHEEL_HARNESS_PATH),
)
CONTAINER_SEMANTIC_ARGV = (
    "docker",
    "run",
    "--rm",
    "--network",
    "none",
    "--read-only",
    "--tmpfs",
    "/tmp:rw,noexec,nosuid,size=64m",
    "--mount",
    "<BOUND_WHEEL_READ_ONLY>",
    "--mount",
    "<BOUND_MODEL_READ_ONLY>",
    "--mount",
    "<BOUND_CALIBRATION_READ_ONLY>",
    "--mount",
    "<BOUND_WHEELHOUSE_MANIFEST_READ_ONLY>",
    "--entrypoint",
    "python",
    "<IMAGE>",
    "-I",
    "-c",
    "<FIXED_INSTALLED_WHEEL_AND_SERVICE_SMOKE>",
)
MODEL_SEMANTIC_ARGV = (
    CLEAN_WHEEL_SEMANTIC_ARGV[0],
    (*CLEAN_WHEEL_SEMANTIC_ARGV[1][:-1], "<RELEASE_WHEEL>[inference]"),
    CLEAN_WHEEL_SEMANTIC_ARGV[2],
    ("<TEMP_PYTHON>", "-I", "-c", "<FIXED_LOCAL_MODEL_SMOKE>", "<MODEL_ROOT>"),
)
SERVICE_SEMANTIC_ARGV = (
    *CLEAN_WHEEL_SEMANTIC_ARGV[:-1],
    (
        "<TEMP_PYTHON>",
        "-I",
        "-c",
        "<FIXED_INPROCESS_HTTP_SMOKE>",
        "<MODEL_ROOT>",
        "<CALIBRATION_BUNDLE>",
    ),
)

INSTALLED_WHEEL_ASSERT_CODE = """
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

distribution = importlib.metadata.distribution("pii-zh-qwen")
assert distribution.version == "0.2.0rc1"
prefix = Path(sys.prefix).resolve()
with zipfile.ZipFile(sys.argv[1]) as archive:
    members = [
        name for name in archive.namelist()
        if name.startswith("pii_zh/") and not name.endswith("/")
    ]
    assert members
    for name in members:
        installed = Path(distribution.locate_file(name)).resolve()
        assert installed.is_relative_to(prefix)
        assert installed.read_bytes() == archive.read(name)
import pii_zh
assert Path(pii_zh.__file__).resolve().is_relative_to(prefix)
print(json.dumps({"status": "PASS", "installed_wheel_member_count": len(members)}, sort_keys=True))
""".strip()
MODEL_SMOKE_CODE = """
import json
import sys
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

model_root = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(
    model_root,
    local_files_only=True,
    trust_remote_code=True,
)
config = AutoConfig.from_pretrained(
    model_root,
    local_files_only=True,
    trust_remote_code=True,
)
assert config.pii_release_eligible is False
assert config.pii_attention_mode == "full"
assert config.num_labels == 49
model = AutoModelForTokenClassification.from_pretrained(
    model_root,
    local_files_only=True,
    trust_remote_code=True,
)
assert model.config.pii_release_eligible is False
assert model.config.pii_attention_mode == "full"
assert model.config.num_labels == 49
model.eval()
inputs = tokenizer("张三", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
assert logits.ndim == 3
assert logits.shape[0] == 1
assert logits.shape[-1] == 49
assert torch.isfinite(logits).all().item()
print(json.dumps({
    "status": "PASS",
    "input_token_count": int(logits.shape[1]),
    "logit_shape": list(logits.shape),
    "pii_attention_mode": model.config.pii_attention_mode,
    "pii_release_eligible": model.config.pii_release_eligible,
}, sort_keys=True))
""".strip()
SERVICE_SMOKE_CODE = """
import json
import sys
from fastapi.testclient import TestClient
from pii_zh.cascade import COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
from pii_zh.service import create_app

app = create_app(
    mode="cascade",
    profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    model_path=sys.argv[1],
    device="cpu",
    micro_batch_size=1,
    calibration=sys.argv[2],
)
with TestClient(app) as client:
    health = client.get("/healthz")
    response = client.post("/v1/analyze", json={"text": "张三的电话是一三八零零一三八零零零。"})
assert health.status_code == 200
assert response.status_code == 200
payload = response.json()
assert payload["profile_version"] == COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
assert isinstance(payload["detections"], list)
summary = {
    "status": "PASS",
    "http": True,
    "detection_count": len(payload["detections"]),
}
print(json.dumps(summary, sort_keys=True))
""".strip()
WHEELHOUSE_DISTRIBUTION_ASSERT_CODE = """
import importlib.metadata
import json
import sys

with open(sys.argv[4], encoding="utf-8") as handle:
    wheelhouse_manifest = json.load(handle)
packages = wheelhouse_manifest["result"]["packages"].values()
for package in packages:
    assert importlib.metadata.version(package["name"]) == package["version"]
""".strip()
CONTAINER_SMOKE_CODE = (
    INSTALLED_WHEEL_ASSERT_CODE
    + "\n"
    + WHEELHOUSE_DISTRIBUTION_ASSERT_CODE
    + "\n"
    + SERVICE_SMOKE_CODE.replace("sys.argv[2]", "sys.argv[3]").replace("sys.argv[1]", "sys.argv[2]")
)


class CommunityVerificationError(CommunityContractError):
    """Raised when a fixed verification harness or receipt fails closed."""


@dataclass(frozen=True, slots=True)
class _ExecutionOutput:
    return_code: int
    stdout: bytes
    stderr: bytes
    regression_input_manifest_sha256: str | None = None
    regression_input_manifest_file_count: int | None = None


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CommunityVerificationError(f"{field} is not a SHA-256 digest")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:@+\-]{0,191}", value
    ):
        raise CommunityVerificationError(f"{field} is not a safe logical ID")
    return value


def _exact(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CommunityVerificationError(f"{field} has an invalid closed shape")
    return value


def _semantic_argv(check_id: str) -> object:
    values: Mapping[str, object] = {
        "unit_tests": CURRENT_RC_REGRESSION_SEMANTIC_ARGV,
        "clean_wheel_smoke": CLEAN_WHEEL_SEMANTIC_ARGV,
        "container_smoke": CONTAINER_SEMANTIC_ARGV,
        "offline_model_smoke": MODEL_SEMANTIC_ARGV,
        "offline_service_smoke": SERVICE_SEMANTIC_ARGV,
    }
    try:
        return values[check_id]
    except KeyError as exc:
        raise CommunityVerificationError("unsupported verification check") from exc


def _current_rc_regression_input_manifest(repository_root: Path) -> dict[str, Any]:
    test_paths = CURRENT_COMMUNITY_RC_TEST_PATHS
    support_paths = CURRENT_COMMUNITY_RC_TEST_SUPPORT_PATHS
    implementation_paths = CURRENT_COMMUNITY_RC_IMPLEMENTATION_PATHS
    all_paths = (*test_paths, *support_paths, *implementation_paths)
    if len(set(all_paths)) != len(all_paths):
        raise CommunityVerificationError("current RC test manifest contains duplicate paths")
    files: dict[str, dict[str, int | str]] = {}
    for relative in all_paths:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise CommunityVerificationError("current RC test manifest path is unsafe")
        payload = read_regular_file(
            repository_root / path, field=f"current RC test manifest file {relative}"
        )
        if not payload:
            raise CommunityVerificationError("current RC test manifest contains an empty file")
        files[relative] = {
            "file_sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    manifest = {
        "test_paths": list(test_paths),
        "support_paths": list(support_paths),
        "implementation_paths": list(implementation_paths),
        "files": files,
        "file_count": len(files),
    }
    return {
        **manifest,
        "manifest_sha256": canonical_json_hash(manifest),
    }


def expected_execution_identity(
    check_id: str, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, int | str]:
    if check_id not in CHECK_IDS:
        raise CommunityVerificationError("unsupported verification check")
    producer_hash = sha256_bytes(
        read_regular_file(repository_root / PRODUCER_PATH, field="verification producer")
    )
    schema_hash = sha256_bytes(
        read_regular_file(repository_root / SCHEMA_PATH, field="verification schema")
    )
    harness_hash = (
        sha256_bytes(
            read_regular_file(
                repository_root / CLEAN_WHEEL_HARNESS_PATH, field="clean wheel harness"
            )
        )
        if check_id == "clean_wheel_smoke"
        else producer_hash
    )
    semantic_hash = canonical_json_hash({"argv": _semantic_argv(check_id)})
    identity = {
        "check_id": check_id,
        "harness_id": HARNESS_IDS[check_id],
        "semantic_argv_sha256": semantic_hash,
        "producer_file_sha256": producer_hash,
        "schema_file_sha256": schema_hash,
        "harness_file_sha256": harness_hash,
    }
    if check_id == "unit_tests":
        test_manifest = _current_rc_regression_input_manifest(repository_root)
        identity.update(
            {
                "regression_input_manifest_file_count": test_manifest["file_count"],
                "regression_input_manifest_sha256": test_manifest["manifest_sha256"],
            }
        )
    return {
        **{key: value for key, value in identity.items() if key != "check_id"},
        "implementation_identity_sha256": canonical_json_hash(identity),
    }


def _subject_binding(path: Path, *, field: str) -> dict[str, int | str]:
    payload = read_regular_file(path, field=field)
    if not payload:
        raise CommunityVerificationError(f"{field} is empty")
    document = strict_json_bytes(payload, field=field)
    self_fields = ("artifact_sha256", "manifest_sha256", "receipt_sha256", "contract_sha256")
    canonical: str | None = None
    for self_field in self_fields:
        if self_field not in document:
            continue
        value = document[self_field]
        if not isinstance(value, str) or value != canonical_json_hash(document, remove=self_field):
            raise CommunityVerificationError(f"{field} self hash does not verify")
        canonical = value
        break
    if canonical is None:
        canonical = canonical_json_hash(document)
    return {
        "file_sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "canonical_sha256": canonical,
    }


def _offline_environment(*, cache_root: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        lowered = key.casefold()
        if "proxy" in lowered or lowered in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "pythonhome",
            "pythonpath",
            "pythonuserbase",
        }:
            environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_PROXY": "*",
        }
    )
    if cache_root is not None:
        resolved_cache_root = cache_root.resolve()
        environment.update(
            {
                "HF_HOME": str(resolved_cache_root / "huggingface"),
                "HF_MODULES_CACHE": str(resolved_cache_root / "huggingface" / "modules"),
                "XDG_CACHE_HOME": str(resolved_cache_root / "xdg"),
            }
        )
    return environment


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    environment: Mapping[str, str],
) -> _ExecutionOutput:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            cwd=cwd,
            env=dict(environment),
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommunityVerificationError("fixed verification harness failed to execute") from exc
    if len(completed.stdout) > MAX_LOG_BYTES or len(completed.stderr) > MAX_LOG_BYTES:
        raise CommunityVerificationError("fixed verification harness log exceeds the limit")
    return _ExecutionOutput(completed.returncode, completed.stdout, completed.stderr)


def _combine_outputs(outputs: Sequence[_ExecutionOutput]) -> _ExecutionOutput:
    stdout = b"".join(
        f"step={index}\n".encode() + item.stdout for index, item in enumerate(outputs, start=1)
    )
    stderr = b"".join(
        f"step={index}\n".encode() + item.stderr for index, item in enumerate(outputs, start=1)
    )
    return _ExecutionOutput(
        next((item.return_code for item in outputs if item.return_code != 0), 0),
        stdout,
        stderr,
    )


def _verify_typed_artifact(path: Path, *, artifact_id: str) -> dict[str, Any]:
    document, _payload = artifacts.load_and_validate_artifact(path, artifact_id=artifact_id)
    return document


def _verify_model_root_against_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    inventory = artifacts._inventory_tree(root, prefix="pkg", reject_package_payloads=True)
    if (
        inventory != manifest["result"]["files"]
        or canonical_json_hash(inventory) != manifest["result"]["inventory_sha256"]
    ):
        raise CommunityVerificationError("model root differs from its typed package manifest")


def _verify_service_inputs(
    *,
    service_configuration_manifest: Path,
    service_source_manifest: Path,
    calibration_bundle: Path,
) -> None:
    _subject_binding(service_configuration_manifest, field="service configuration manifest")
    _subject_binding(service_source_manifest, field="service source manifest")
    service = strict_json_bytes(
        read_regular_file(service_configuration_manifest, field="service configuration manifest"),
        field="service configuration manifest",
    )
    source = strict_json_bytes(
        read_regular_file(service_source_manifest, field="service source manifest"),
        field="service source manifest",
    )
    calibration_binding = artifacts._metadata_binding(
        calibration_bundle, field="calibration bundle"
    )
    if (
        service.get("service_id") != source.get("service_id")
        or service.get("profile_id") != source.get("profile_id")
        or service.get("implementation_sha256") != source.get("implementation_sha256")
        or service.get("calibration_bundle_file_sha256") != calibration_binding["file_sha256"]
    ):
        raise CommunityVerificationError("service smoke subjects do not bind one runtime")


def _execute_current_rc_regression(*, repository_root: Path) -> _ExecutionOutput:
    before = _current_rc_regression_input_manifest(repository_root)
    output = _run(
        [sys.executable, *CURRENT_RC_REGRESSION_SEMANTIC_ARGV[1:]],
        cwd=repository_root,
        timeout_seconds=3600,
        environment=_offline_environment(),
    )
    after = _current_rc_regression_input_manifest(repository_root)
    if after != before:
        raise CommunityVerificationError(
            "current RC regression inputs changed during fixed harness execution"
        )
    return _ExecutionOutput(
        output.return_code,
        output.stdout,
        output.stderr,
        regression_input_manifest_sha256=before["manifest_sha256"],
        regression_input_manifest_file_count=before["file_count"],
    )


def _verify_wheelhouse_against_manifest(
    wheelhouse: Path, wheelhouse_manifest: Mapping[str, Any]
) -> None:
    try:
        files, packages = artifacts._wheelhouse_inventory(wheelhouse)
    except artifacts.CommunityReleaseArtifactError as exc:
        raise CommunityVerificationError("wheelhouse is unavailable or unsafe") from exc
    result = wheelhouse_manifest["result"]
    if (
        files != result.get("files")
        or packages != result.get("packages")
        or canonical_json_hash(files) != result.get("inventory_sha256")
        or canonical_json_hash(packages) != result.get("packages_sha256")
    ):
        raise CommunityVerificationError("wheelhouse differs from its typed manifest")


def _verify_wheel_against_manifest(wheel_path: Path, wheel_manifest: Mapping[str, Any]) -> None:
    actual_wheel = artifacts._large_file_binding(wheel_path, field="release wheel")
    actual_inventory, _name, _version = artifacts._wheel_inventory(wheel_path)
    if actual_wheel != wheel_manifest["inputs"].get("wheel") or actual_inventory != wheel_manifest[
        "result"
    ].get("files"):
        raise CommunityVerificationError("release wheel differs from its typed manifest")


def _execute_in_clean_venv(
    *,
    check_id: str,
    wheel_path: Path,
    wheelhouse: Path,
    model_root: Path | None,
    calibration_bundle: Path | None,
    repository_root: Path,
) -> _ExecutionOutput:
    install_extras = {
        "clean_wheel_smoke": "cascade,service",
        "offline_model_smoke": "inference",
        "offline_service_smoke": "cascade,service",
    }
    try:
        requested_wheel = f"{wheel_path}[{install_extras[check_id]}]"
    except KeyError as exc:
        raise CommunityVerificationError("clean runtime check ID is invalid") from exc
    with tempfile.TemporaryDirectory(prefix=f"pii-zh-{check_id}-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        prefix = temporary_root / "venv"
        runtime_cwd = temporary_root / "runtime"
        runtime_cwd.mkdir()
        offline_environment = _offline_environment(cache_root=temporary_root / "cache")
        outputs: list[_ExecutionOutput] = []
        create = _run(
            [sys.executable, "-m", "venv", str(prefix)],
            cwd=runtime_cwd,
            timeout_seconds=300,
            environment=offline_environment,
        )
        outputs.append(create)
        if create.return_code != 0:
            return _combine_outputs(outputs)
        python = prefix / "bin" / "python"
        install = _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                requested_wheel,
            ],
            cwd=runtime_cwd,
            timeout_seconds=900,
            environment=offline_environment,
        )
        outputs.append(install)
        if install.return_code != 0:
            return _combine_outputs(outputs)
        installed = _run(
            [str(python), "-I", "-c", INSTALLED_WHEEL_ASSERT_CODE, str(wheel_path)],
            cwd=runtime_cwd,
            timeout_seconds=300,
            environment=offline_environment,
        )
        outputs.append(installed)
        if installed.return_code != 0:
            return _combine_outputs(outputs)
        if check_id == "clean_wheel_smoke":
            smoke_argv = [str(python), str(repository_root / CLEAN_WHEEL_HARNESS_PATH)]
            timeout_seconds = 300
        elif check_id == "offline_model_smoke" and model_root is not None:
            smoke_argv = [str(python), "-I", "-c", MODEL_SMOKE_CODE, str(model_root)]
            timeout_seconds = 900
        elif (
            check_id == "offline_service_smoke"
            and model_root is not None
            and calibration_bundle is not None
        ):
            smoke_argv = [
                str(python),
                "-I",
                "-c",
                SERVICE_SMOKE_CODE,
                str(model_root),
                str(calibration_bundle),
            ]
            timeout_seconds = 1200
        else:
            raise CommunityVerificationError("clean runtime smoke inputs are incomplete")
        outputs.append(
            _run(
                smoke_argv,
                cwd=runtime_cwd,
                timeout_seconds=timeout_seconds,
                environment=offline_environment,
            )
        )
        return _combine_outputs(outputs)


def _execute_clean_wheel(
    *,
    wheel_path: Path,
    wheelhouse: Path,
    wheel_manifest: Mapping[str, Any],
    wheelhouse_manifest: Mapping[str, Any],
    repository_root: Path,
) -> _ExecutionOutput:
    _verify_wheel_against_manifest(wheel_path, wheel_manifest)
    _verify_wheelhouse_against_manifest(wheelhouse, wheelhouse_manifest)
    return _execute_in_clean_venv(
        check_id="clean_wheel_smoke",
        wheel_path=wheel_path,
        wheelhouse=wheelhouse,
        model_root=None,
        calibration_bundle=None,
        repository_root=repository_root,
    )


def _execute_container(
    *,
    image_ref: str,
    wheel_path: Path,
    wheelhouse_manifest_path: Path,
    model_root: Path,
    calibration_bundle: Path,
    repository_root: Path,
) -> _ExecutionOutput:
    if not image_ref or len(image_ref) > 256 or any(character.isspace() for character in image_ref):
        raise CommunityVerificationError("container image reference is invalid")
    bindings = (
        (wheel_path, "/release/release.whl"),
        (model_root, "/release/model"),
        (calibration_bundle, "/release/calibration.json"),
        (wheelhouse_manifest_path, "/release/wheelhouse-manifest.json"),
    )
    mounts: list[str] = []
    for source, destination in bindings:
        resolved = source.resolve()
        if "," in str(resolved) or not resolved.exists() or resolved.is_symlink():
            raise CommunityVerificationError("container smoke bind source is unsafe")
        mounts.extend(["--mount", f"type=bind,src={resolved},dst={destination},readonly"])
    return _run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            *mounts,
            "--entrypoint",
            "python",
            image_ref,
            "-I",
            "-c",
            CONTAINER_SMOKE_CODE,
            "/release/release.whl",
            "/release/model",
            "/release/calibration.json",
            "/release/wheelhouse-manifest.json",
        ],
        cwd=repository_root,
        timeout_seconds=300,
        environment=_offline_environment(),
    )


def _execute_model(
    *,
    wheel_path: Path,
    wheelhouse: Path,
    wheelhouse_manifest: Mapping[str, Any],
    model_root: Path,
    repository_root: Path,
) -> _ExecutionOutput:
    _verify_wheelhouse_against_manifest(wheelhouse, wheelhouse_manifest)
    return _execute_in_clean_venv(
        check_id="offline_model_smoke",
        wheel_path=wheel_path,
        wheelhouse=wheelhouse,
        model_root=model_root,
        calibration_bundle=None,
        repository_root=repository_root,
    )


def _execute_service(
    *,
    wheel_path: Path,
    wheelhouse: Path,
    wheelhouse_manifest: Mapping[str, Any],
    model_root: Path,
    calibration_bundle: Path,
    repository_root: Path,
) -> _ExecutionOutput:
    _verify_wheelhouse_against_manifest(wheelhouse, wheelhouse_manifest)
    return _execute_in_clean_venv(
        check_id="offline_service_smoke",
        wheel_path=wheel_path,
        wheelhouse=wheelhouse,
        model_root=model_root,
        calibration_bundle=calibration_bundle,
        repository_root=repository_root,
    )


def _execution_document(
    check_id: str, output: _ExecutionOutput, *, repository_root: Path
) -> dict[str, Any]:
    if output.return_code != 0:
        raise CommunityVerificationError(
            f"{check_id} fixed harness failed with return code {output.return_code}"
        )
    identity = expected_execution_identity(check_id, repository_root=repository_root)
    if check_id == "unit_tests":
        if (
            output.regression_input_manifest_sha256 != identity["regression_input_manifest_sha256"]
            or output.regression_input_manifest_file_count
            != identity["regression_input_manifest_file_count"]
        ):
            raise CommunityVerificationError(
                "current RC regression inputs changed before receipt issuance"
            )
    elif (
        output.regression_input_manifest_sha256 is not None
        or output.regression_input_manifest_file_count is not None
    ):
        raise CommunityVerificationError("non-regression harness carried a test manifest")
    stdout_hash = sha256_bytes(output.stdout)
    stderr_hash = sha256_bytes(output.stderr)
    log_hash = canonical_json_hash(
        {
            "return_code": output.return_code,
            "stdout_sha256": stdout_hash,
            "stderr_sha256": stderr_hash,
        }
    )
    return {
        **identity,
        "return_code": 0,
        "stdout_sha256": stdout_hash,
        "stderr_sha256": stderr_hash,
        "redacted_log_sha256": log_hash,
    }


def validate_verification_receipt(
    receipt: Mapping[str, Any],
    *,
    check_id: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    schema = load_json_path(repository_root / SCHEMA_PATH, field="verification receipt schema")
    validate_schema(receipt, schema, field=f"{check_id} verification receipt")
    if receipt["check_id"] != check_id or check_id not in CHECK_IDS:
        raise CommunityVerificationError("verification check ID differs from its contract slot")
    if (
        receipt["release"] != EXPECTED_RELEASE
        or receipt["assertions"] != EXPECTED_ASSERTIONS
        or receipt["privacy"] != EXPECTED_PRIVACY
        or receipt["status"] != "PASS"
        or receipt["scope"] != "local_non_production"
    ):
        raise CommunityVerificationError("verification scope or assertion boundary changed")
    if set(receipt["subjects"]) != EXPECTED_SUBJECT_IDS[check_id]:
        raise CommunityVerificationError("verification subject inventory is not exact")
    for logical_id, binding in receipt["subjects"].items():
        _safe_id(logical_id, field="verification subject ID")
        binding = _exact(
            binding,
            {"file_sha256", "size_bytes", "canonical_sha256"},
            field=f"verification subject {logical_id}",
        )
        _digest(binding["file_sha256"], field=f"verification subject {logical_id} file")
        _digest(binding["canonical_sha256"], field=f"verification subject {logical_id} canonical")
        if (
            isinstance(binding["size_bytes"], bool)
            or not isinstance(binding["size_bytes"], int)
            or binding["size_bytes"] <= 0
        ):
            raise CommunityVerificationError("verification subject size is invalid")
    execution = receipt["execution"]
    expected_identity = expected_execution_identity(check_id, repository_root=repository_root)
    if set(execution) != {
        *expected_identity,
        "return_code",
        "stdout_sha256",
        "stderr_sha256",
        "redacted_log_sha256",
    }:
        raise CommunityVerificationError("verification execution inventory is not exact")
    if any(execution[key] != value for key, value in expected_identity.items()):
        raise CommunityVerificationError("verification implementation identity is stale")
    if execution["return_code"] != 0:
        raise CommunityVerificationError("verification harness did not pass")
    for key in ("stdout_sha256", "stderr_sha256", "redacted_log_sha256"):
        _digest(execution[key], field=f"verification execution {key}")
    if execution["redacted_log_sha256"] != canonical_json_hash(
        {
            "return_code": 0,
            "stdout_sha256": execution["stdout_sha256"],
            "stderr_sha256": execution["stderr_sha256"],
        }
    ):
        raise CommunityVerificationError("verification redacted log binding is invalid")
    if receipt["receipt_sha256"] != canonical_json_hash(receipt, remove="receipt_sha256"):
        raise CommunityVerificationError("verification receipt self hash does not verify")


def load_and_validate_verification(
    path: Path,
    *,
    check_id: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path, field=f"{check_id} verification receipt")
    receipt = strict_json_bytes(payload, field=f"{check_id} verification receipt")
    validate_verification_receipt(receipt, check_id=check_id, repository_root=repository_root)
    return receipt, payload


def publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CommunityVerificationError("refusing to overwrite verification receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise CommunityVerificationError("verification receipt mode is not 0444")
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="check_id", required=True)

    unit = subparsers.add_parser("unit_tests")
    unit.add_argument("--service-source-manifest", required=True, type=Path)
    unit.add_argument("--output", required=True, type=Path)

    wheel = subparsers.add_parser("clean_wheel_smoke")
    wheel.add_argument("--wheel-manifest", required=True, type=Path)
    wheel.add_argument("--wheelhouse-manifest", required=True, type=Path)
    wheel.add_argument("--wheel", required=True, type=Path)
    wheel.add_argument("--wheelhouse", required=True, type=Path)
    wheel.add_argument("--output", required=True, type=Path)

    container = subparsers.add_parser("container_smoke")
    container.add_argument("--container-manifest", required=True, type=Path)
    container.add_argument("--wheel-manifest", required=True, type=Path)
    container.add_argument("--wheelhouse-manifest", required=True, type=Path)
    container.add_argument("--wheel", required=True, type=Path)
    container.add_argument("--model-package-manifest", required=True, type=Path)
    container.add_argument("--model-root", required=True, type=Path)
    container.add_argument("--service-configuration-manifest", required=True, type=Path)
    container.add_argument("--service-source-manifest", required=True, type=Path)
    container.add_argument("--calibration-bundle", required=True, type=Path)
    container.add_argument("--output", required=True, type=Path)

    model = subparsers.add_parser("offline_model_smoke")
    model.add_argument("--model-package-manifest", required=True, type=Path)
    model.add_argument("--model-root", required=True, type=Path)
    model.add_argument("--wheel-manifest", required=True, type=Path)
    model.add_argument("--wheelhouse-manifest", required=True, type=Path)
    model.add_argument("--wheel", required=True, type=Path)
    model.add_argument("--wheelhouse", required=True, type=Path)
    model.add_argument("--output", required=True, type=Path)

    service = subparsers.add_parser("offline_service_smoke")
    service.add_argument("--model-package-manifest", required=True, type=Path)
    service.add_argument("--model-root", required=True, type=Path)
    service.add_argument("--wheel-manifest", required=True, type=Path)
    service.add_argument("--wheelhouse-manifest", required=True, type=Path)
    service.add_argument("--wheel", required=True, type=Path)
    service.add_argument("--wheelhouse", required=True, type=Path)
    service.add_argument("--service-configuration-manifest", required=True, type=Path)
    service.add_argument("--service-source-manifest", required=True, type=Path)
    service.add_argument("--calibration-bundle", required=True, type=Path)
    service.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", required=True, type=Path)
    validate.add_argument("--receipt-check-id", required=True, choices=CHECK_IDS)
    return parser


def _execute_fixed_check_and_issue_receipt(args: argparse.Namespace) -> dict[str, Any]:
    check_id = args.check_id
    if check_id == "unit_tests":
        subjects = {"service_source_manifest": args.service_source_manifest}
        output = _execute_current_rc_regression(repository_root=REPOSITORY_ROOT)
    elif check_id == "clean_wheel_smoke":
        wheel_manifest = _verify_typed_artifact(args.wheel_manifest, artifact_id="wheel_manifest")
        wheelhouse_manifest = _verify_typed_artifact(
            args.wheelhouse_manifest, artifact_id="wheelhouse_manifest"
        )
        subjects = {
            "wheel_manifest": args.wheel_manifest,
            "wheelhouse_manifest": args.wheelhouse_manifest,
        }
        output = _execute_clean_wheel(
            wheel_path=args.wheel,
            wheelhouse=args.wheelhouse,
            wheel_manifest=wheel_manifest,
            wheelhouse_manifest=wheelhouse_manifest,
            repository_root=REPOSITORY_ROOT,
        )
    elif check_id == "container_smoke":
        container_manifest = _verify_typed_artifact(
            args.container_manifest, artifact_id="container_manifest"
        )
        wheel_manifest = _verify_typed_artifact(args.wheel_manifest, artifact_id="wheel_manifest")
        wheelhouse_manifest = _verify_typed_artifact(
            args.wheelhouse_manifest, artifact_id="wheelhouse_manifest"
        )
        model_manifest = _verify_typed_artifact(
            args.model_package_manifest, artifact_id="model_package_manifest"
        )
        _verify_wheel_against_manifest(args.wheel, wheel_manifest)
        if (
            container_manifest["inputs"].get("wheelhouse_manifest")
            != artifacts._metadata_binding(
                args.wheelhouse_manifest, field="container wheelhouse manifest"
            )
            or container_manifest["result"].get("bound_wheelhouse_artifact_sha256")
            != wheelhouse_manifest["artifact_sha256"]
        ):
            raise CommunityVerificationError("container does not bind the supplied wheelhouse")
        _verify_model_root_against_manifest(args.model_root, model_manifest)
        _verify_service_inputs(
            service_configuration_manifest=args.service_configuration_manifest,
            service_source_manifest=args.service_source_manifest,
            calibration_bundle=args.calibration_bundle,
        )
        subjects = {
            "container_manifest": args.container_manifest,
            "wheel_manifest": args.wheel_manifest,
            "wheelhouse_manifest": args.wheelhouse_manifest,
            "model_package_manifest": args.model_package_manifest,
            "service_configuration_manifest": args.service_configuration_manifest,
            "service_source_manifest": args.service_source_manifest,
            "calibration_bundle": args.calibration_bundle,
        }
        output = _execute_container(
            image_ref=container_manifest["result"]["image_id"],
            wheel_path=args.wheel,
            wheelhouse_manifest_path=args.wheelhouse_manifest,
            model_root=args.model_root,
            calibration_bundle=args.calibration_bundle,
            repository_root=REPOSITORY_ROOT,
        )
    elif check_id == "offline_model_smoke":
        model_manifest = _verify_typed_artifact(
            args.model_package_manifest, artifact_id="model_package_manifest"
        )
        _verify_model_root_against_manifest(args.model_root, model_manifest)
        wheel_manifest = _verify_typed_artifact(args.wheel_manifest, artifact_id="wheel_manifest")
        _verify_wheel_against_manifest(args.wheel, wheel_manifest)
        wheelhouse_manifest = _verify_typed_artifact(
            args.wheelhouse_manifest, artifact_id="wheelhouse_manifest"
        )
        subjects = {
            "model_package_manifest": args.model_package_manifest,
            "wheel_manifest": args.wheel_manifest,
            "wheelhouse_manifest": args.wheelhouse_manifest,
        }
        output = _execute_model(
            wheel_path=args.wheel,
            wheelhouse=args.wheelhouse,
            wheelhouse_manifest=wheelhouse_manifest,
            model_root=args.model_root,
            repository_root=REPOSITORY_ROOT,
        )
    elif check_id == "offline_service_smoke":
        model_manifest = _verify_typed_artifact(
            args.model_package_manifest, artifact_id="model_package_manifest"
        )
        _verify_model_root_against_manifest(args.model_root, model_manifest)
        wheel_manifest = _verify_typed_artifact(args.wheel_manifest, artifact_id="wheel_manifest")
        _verify_wheel_against_manifest(args.wheel, wheel_manifest)
        wheelhouse_manifest = _verify_typed_artifact(
            args.wheelhouse_manifest, artifact_id="wheelhouse_manifest"
        )
        _verify_service_inputs(
            service_configuration_manifest=args.service_configuration_manifest,
            service_source_manifest=args.service_source_manifest,
            calibration_bundle=args.calibration_bundle,
        )
        subjects = {
            "model_package_manifest": args.model_package_manifest,
            "wheel_manifest": args.wheel_manifest,
            "wheelhouse_manifest": args.wheelhouse_manifest,
            "service_configuration_manifest": args.service_configuration_manifest,
            "service_source_manifest": args.service_source_manifest,
            "calibration_bundle": args.calibration_bundle,
        }
        output = _execute_service(
            wheel_path=args.wheel,
            wheelhouse=args.wheelhouse,
            wheelhouse_manifest=wheelhouse_manifest,
            model_root=args.model_root,
            calibration_bundle=args.calibration_bundle,
            repository_root=REPOSITORY_ROOT,
        )
    else:  # pragma: no cover - argparse closes the check set
        raise CommunityVerificationError("unsupported verification check")
    if check_id not in CHECK_IDS or set(subjects) != EXPECTED_SUBJECT_IDS[check_id]:
        raise CommunityVerificationError("verification subject inventory is not exact")
    # PASS construction is deliberately kept inside this fixed-command path.
    # There is no callable receipt builder that accepts caller-supplied output,
    # return code, command, status or hash.
    bound_subjects = {
        logical_id: _subject_binding(path, field=f"verification subject {logical_id}")
        for logical_id, path in sorted(subjects.items())
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "check_id": check_id,
        "status": "PASS",
        "scope": "local_non_production",
        "release": dict(EXPECTED_RELEASE),
        "subjects": bound_subjects,
        "execution": _execution_document(check_id, output, repository_root=REPOSITORY_ROOT),
        "assertions": dict(EXPECTED_ASSERTIONS),
        "privacy": dict(EXPECTED_PRIVACY),
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, remove="receipt_sha256")
    validate_verification_receipt(receipt, check_id=check_id)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.check_id == "validate":
            receipt, payload = load_and_validate_verification(
                args.receipt, check_id=args.receipt_check_id
            )
            print(
                json.dumps(
                    {
                        "check_id": receipt["check_id"],
                        "file_sha256": sha256_bytes(payload),
                        "receipt_sha256": receipt["receipt_sha256"],
                        "status": receipt["status"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        receipt = _execute_fixed_check_and_issue_receipt(args)
        publish_receipt(args.output, receipt)
        print(
            json.dumps(
                {
                    "check_id": receipt["check_id"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "status": receipt["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, CommunityContractError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
