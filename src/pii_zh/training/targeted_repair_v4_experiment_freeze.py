"""Fail-closed verifier for the targeted-repair-v4 experiment freeze."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ..targeted_repair_v4_experiment_closure import (
    DATA_LINEAGE_ONLY_PATH_KEYS,
    EVIDENCE_ONLY_PATH_KEYS,
    EXPERIMENT_ANCHOR_KEY,
    EXPERIMENT_IMPLEMENTATION_PATHS,
    RUNTIME_PATH_KEYS,
)
from .targeted_repair_v4_experiment_freeze_anchor import (
    EXPERIMENT_RECEIPT_FILE_SHA256,
    EXPERIMENT_RECEIPT_LOGICAL_SHA256,
)

_ROOT = Path(__file__).resolve().parents[3]
_ANCHOR_KEY = EXPERIMENT_ANCHOR_KEY
_ANCHOR_PATH = Path(__file__).with_name("targeted_repair_v4_experiment_freeze_anchor.py")
_PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
    "PyYAML": "PyYAML",
    "numpy": "numpy",
}


class TargetedRepairV4ExperimentFreezeError(ValueError):
    """Raised without exposing paths, credentials, or row-level content."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TargetedRepairV4ExperimentFreezeError(f"{field} must be an object")
    return value


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TargetedRepairV4ExperimentFreezeError(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV4ExperimentFreezeError(f"{field} is unreadable") from exc
    return dict(_mapping(value, field=field))


def _runtime(config_lock: Mapping[str, Any], *, require_gpu: bool) -> dict[str, Any]:
    python = _mapping(config_lock.get("python"), field="python runtime")
    packages = _mapping(config_lock.get("packages"), field="package runtime")
    if (
        python.get("implementation") != "CPython"
        or python.get("version") != platform.python_version()
        or python.get("executable_sha256") != _sha(Path(sys.executable).resolve(strict=True))
        or sys.implementation.name != "cpython"
        or set(packages) != set(_PACKAGE_DISTRIBUTIONS)
        or any(
            importlib.metadata.version(_PACKAGE_DISTRIBUTIONS[name]) != expected
            for name, expected in packages.items()
        )
        or config_lock.get("torch_build") != torch.__version__
        or config_lock.get("torch_cuda_runtime") != torch.version.cuda
    ):
        raise TargetedRepairV4ExperimentFreezeError("runtime lock changed")
    offline = _mapping(config_lock.get("offline_model_loading"), field="offline runtime")
    if offline != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"} or any(
        os.environ.get(name) != value for name, value in offline.items()
    ):
        raise TargetedRepairV4ExperimentFreezeError("offline model-loading environment is required")
    gpu = _mapping(config_lock.get("gpu"), field="GPU runtime")
    if (
        gpu.get("model") != "NVIDIA A800-SXM4-80GB"
        or gpu.get("driver_version") != "550.144.03"
        or gpu.get("physical_id") != 0
        or gpu.get("cuda_visible_devices") != "0"
        or gpu.get("device_count_after_isolation") != 1
        or gpu.get("maximum_preflight_memory_mib") != 16
        or gpu.get("maximum_preflight_utilization_percent") != 0
        or gpu.get("maximum_preflight_compute_process_count") != 0
        or gpu.get("must_be_freshly_idle_before_launch") is not True
        or gpu.get("kill_or_reset_other_processes_allowed") is not False
    ):
        raise TargetedRepairV4ExperimentFreezeError("GPU isolation contract changed")
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "offline_model_loading": True,
        "gpu_required": require_gpu,
        "cuda_visible_devices": "0" if require_gpu else None,
        "gpu_verified": False,
    }
    if require_gpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise TargetedRepairV4ExperimentFreezeError("GPU runtime is not isolated to card zero")
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TargetedRepairV4ExperimentFreezeError("GPU preflight query failed") from exc
        selected: tuple[int, str, str, str, int, int] | None = None
        selected_rows = []
        for line in completed.stdout.splitlines():
            fields = tuple(field.strip() for field in line.split(","))
            if len(fields) != 6:
                raise TargetedRepairV4ExperimentFreezeError("GPU preflight output changed")
            try:
                row = (
                    int(fields[0]),
                    fields[1],
                    fields[2],
                    fields[3],
                    int(fields[4]),
                    int(fields[5]),
                )
            except ValueError as exc:
                raise TargetedRepairV4ExperimentFreezeError("GPU preflight output changed") from exc
            if row[0] == gpu["physical_id"]:
                selected_rows.append(row)
        if len(selected_rows) == 1:
            selected = selected_rows[0]
        if (
            selected is None
            or selected[2] != gpu["model"]
            or selected[3] != gpu["driver_version"]
            or selected[4] > gpu["maximum_preflight_memory_mib"]
            or selected[5] > gpu["maximum_preflight_utilization_percent"]
        ):
            raise TargetedRepairV4ExperimentFreezeError(
                "selected GPU is not freshly idle and compatible"
            )
        try:
            compute_apps = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TargetedRepairV4ExperimentFreezeError(
                "GPU process preflight query failed"
            ) from exc
        selected_process_count = 0
        for line in compute_apps.stdout.splitlines():
            if not line.strip():
                continue
            fields = tuple(field.strip() for field in line.split(","))
            if len(fields) != 2 or not fields[1].isdigit():
                raise TargetedRepairV4ExperimentFreezeError("GPU process preflight output changed")
            selected_process_count += fields[0] == selected[1]
        if selected_process_count > gpu["maximum_preflight_compute_process_count"]:
            raise TargetedRepairV4ExperimentFreezeError("selected GPU has a compute process")
        # CUDA may only be initialized after both read-only physical-card checks pass.
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != gpu["model"]
        ):
            raise TargetedRepairV4ExperimentFreezeError("GPU runtime is not isolated to card zero")
        result["gpu_verified"] = True
        result["gpu"] = {
            "physical_id": selected[0],
            "model": selected[2],
            "driver_version": selected[3],
            "preflight_memory_mib": selected[4],
            "preflight_utilization_percent": selected[5],
            "preflight_compute_process_count": selected_process_count,
            "single_visible_device": True,
            "kill_or_reset_performed": False,
        }
    return result


def verify_targeted_repair_v4_experiment_freeze(
    *, receipt_path: Path, config_path: Path, require_gpu: bool
) -> Mapping[str, Any]:
    if (
        EXPERIMENT_RECEIPT_FILE_SHA256 == "UNFROZEN"
        or EXPERIMENT_RECEIPT_LOGICAL_SHA256 == "UNFROZEN"
    ):
        raise TargetedRepairV4ExperimentFreezeError("experiment implementation is not frozen")
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or config_path.is_symlink()
        or not config_path.is_file()
        or _sha(receipt_path) != EXPERIMENT_RECEIPT_FILE_SHA256
    ):
        raise TargetedRepairV4ExperimentFreezeError("experiment freeze identity changed")
    receipt = _load_json(receipt_path, field="experiment receipt")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        claimed != EXPERIMENT_RECEIPT_LOGICAL_SHA256
        or _canonical(unsigned) != claimed
        or receipt.get("receipt_id") != "macbert_24_targeted_repair_v4_preregistration_freeze_v1"
        or receipt.get("status") != "preregistered_frozen_before_GPU"
    ):
        raise TargetedRepairV4ExperimentFreezeError("experiment receipt self-hash changed")
    config = _mapping(receipt.get("config"), field="receipt config")
    if config.get(
        "repository_relative_path"
    ) != "configs/experiments/macbert_24_targeted_repair_v4.yaml" or config.get(
        "file_sha256"
    ) != _sha(config_path):
        raise TargetedRepairV4ExperimentFreezeError("experiment config binding changed")
    try:
        config_document = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), field="experiment config"
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TargetedRepairV4ExperimentFreezeError("experiment config is unreadable") from exc
    if (
        config_document.get("experiment_id") != "macbert_24_targeted_repair_v4_d0_only_seed42"
        or config_document.get("status") != "preregistered_frozen"
        or config_document.get("release_eligible") is not False
        or config_document.get("synthetic_only") is not True
        or _mapping(config_document.get("freeze_contract"), field="config freeze contract")
        != {
            "receipt_id": "macbert_24_targeted_repair_v4_preregistration_freeze_v1",
            "repository_relative_path": (
                "reports/experiments/macbert_24_targeted_repair_v4_preregistration_freeze.json"
            ),
            "data_freeze_v4_bound": True,
            "experiment_anchor_externally_reported_before_GPU": True,
        }
    ):
        raise TargetedRepairV4ExperimentFreezeError("experiment config safety contract changed")
    declared = dict(EXPERIMENT_IMPLEMENTATION_PATHS)
    if declared.get(_ANCHOR_KEY) != (
        "src/pii_zh/training/targeted_repair_v4_experiment_freeze_anchor.py"
    ):
        raise TargetedRepairV4ExperimentFreezeError("experiment anchor inventory changed")
    implementations = _mapping(receipt.get("implementations"), field="implementations")
    expected_implementations = set(declared) - {_ANCHOR_KEY}
    if set(implementations) != expected_implementations:
        raise TargetedRepairV4ExperimentFreezeError("experiment implementation inventory changed")
    for name in expected_implementations:
        binding = _mapping(implementations[name], field=f"implementation {name}")
        relative = declared[name]
        path = _ROOT / relative
        if (
            binding.get("repository_relative_path") != relative
            or path.is_symlink()
            or not path.is_file()
            or binding.get("file_sha256") != _sha(path)
        ):
            raise TargetedRepairV4ExperimentFreezeError("experiment implementation changed")
    closure_audit = _mapping(receipt.get("closure_audit"), field="closure audit")
    if closure_audit != {
        "algorithm": (
            "recursive_ast_imports_with_parent_package_init_fixpoint_plus_runtime_assets"
        ),
        "authoritative_closure_module": ("src/pii_zh/targeted_repair_v4_experiment_closure.py"),
        "superseded_data_module_experiment_map_authoritative": False,
        "declared_total_path_count": len(declared),
        "runtime_path_count_including_external_anchor": len(RUNTIME_PATH_KEYS),
        "python_runtime_path_count_including_external_anchor": sum(
            Path(declared[key]).suffix == ".py" for key in RUNTIME_PATH_KEYS
        ),
        "non_python_runtime_asset_count": sum(
            Path(declared[key]).suffix != ".py" for key in RUNTIME_PATH_KEYS
        ),
        "data_lineage_only_path_count": len(DATA_LINEAGE_ONLY_PATH_KEYS),
        "evidence_only_path_count": len(EVIDENCE_ONLY_PATH_KEYS),
        "receipt_bound_path_count": len(implementations),
        "externally_bound_anchor_count": 1,
        "missing_or_extra_runtime_paths": 0,
    }:
        raise TargetedRepairV4ExperimentFreezeError("experiment closure audit changed")
    anchor = _mapping(receipt.get("external_trust_anchor"), field="external trust anchor")
    if anchor != {
        "key": _ANCHOR_KEY,
        "repository_relative_path": declared[_ANCHOR_KEY],
        "excluded_from_self_referential_receipt": True,
        "reason": "anchor embeds this receipt file and logical hash",
    }:
        raise TargetedRepairV4ExperimentFreezeError("experiment external anchor changed")
    tests = _mapping(receipt.get("pre_freeze_tests"), field="pre-freeze tests")
    if (
        tests.get("status") != "passed"
        or not isinstance(tests.get("passed_count"), int)
        or tests.get("passed_count") < 1
        or tests.get("test_file_sha256")
        != _mapping(implementations.get("trainer_unit_tests"), field="trainer tests").get(
            "file_sha256"
        )
        or tests.get("GPU_used") is not False
    ):
        raise TargetedRepairV4ExperimentFreezeError("pre-freeze tests changed")
    data = _mapping(receipt.get("data_freeze"), field="data freeze")
    targeted = _mapping(
        _mapping(config_document.get("data"), field="config data").get("targeted_dataset"),
        field="config targeted dataset",
    )
    if (
        data.get("receipt_id") != "targeted_repair_v4_implementation_freeze_v4"
        or data.get("receipt_file_sha256")
        != "d25e0309b990943cd5e324fb92c3371494809e679f49eefb8cb9b87bcae63502"
        or data.get("receipt_sha256")
        != "c01b3003ddc585f9549ca241d1d6dbdcb9e8136427287d3c0e3992262d37efe2"
        or data.get("anchor_file_sha256")
        != "d4343cd5a986f1e0201fab6e01e349eb2193aa501376c43b67cbc211ad6cf3c2"
        or data.get("receipt_file_sha256") != targeted.get("freeze_receipt_file_sha256")
        or data.get("receipt_sha256") != targeted.get("freeze_receipt_sha256")
        or data.get("anchor_file_sha256") != targeted.get("data_freeze_anchor_file_sha256")
    ):
        raise TargetedRepairV4ExperimentFreezeError("data freeze binding changed")
    if receipt.get("execution_boundary") != {
        "GPU_training_started": False,
        "model_checkpoint_created": False,
        "external_evaluation_count": 0,
        "PII_Bench_execution_count": 0,
    }:
        raise TargetedRepairV4ExperimentFreezeError("pre-GPU execution boundary changed")
    config_base = _mapping(config_document.get("base_model"), field="config base model")
    config_fingerprint = _mapping(
        config_base.get("frozen_fingerprint"), field="config base fingerprint"
    )
    base = _mapping(receipt.get("base_model"), field="receipt base model")
    if base != {
        "source_id": config_base.get("source_id"),
        "revision": config_base.get("revision"),
        "license": config_base.get("license"),
        "files": dict(_mapping(config_fingerprint.get("files"), field="base files")),
        "local_files_only": True,
        "trust_remote_code": False,
        "safe_weights_only": True,
        "v3_checkpoint_loading_allowed": False,
    }:
        raise TargetedRepairV4ExperimentFreezeError("base model receipt binding changed")
    runtime_lock = _mapping(receipt.get("runtime_lock"), field="runtime lock")
    if dict(runtime_lock) != dict(
        _mapping(config_document.get("runtime_lock"), field="config runtime lock")
    ):
        raise TargetedRepairV4ExperimentFreezeError("receipt runtime differs from config")
    runtime = _runtime(runtime_lock, require_gpu=require_gpu)
    return {
        "status": "verified",
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": claimed,
        "receipt_file_sha256": EXPERIMENT_RECEIPT_FILE_SHA256,
        "config_file_sha256": config["file_sha256"],
        "implementation_count": len(implementations),
        "declared_total_path_count": len(declared),
        "runtime_path_count_including_external_anchor": len(RUNTIME_PATH_KEYS),
        "data_lineage_only_path_count": len(DATA_LINEAGE_ONLY_PATH_KEYS),
        "evidence_only_path_count": len(EVIDENCE_ONLY_PATH_KEYS),
        "experiment_anchor_file_sha256": _sha(_ANCHOR_PATH),
        "data_freeze": dict(data),
        "runtime": runtime,
    }


__all__ = [
    "TargetedRepairV4ExperimentFreezeError",
    "verify_targeted_repair_v4_experiment_freeze",
]
