"""Fail-closed verifier for the MacBERT targeted-repair-v3 preregistration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pii_zh.targeted_repair_v3_closure import EXPERIMENT_IMPLEMENTATION_PATHS

from .targeted_repair_v3_experiment_anchor import (
    EXPERIMENT_RECEIPT_FILE_SHA256,
    EXPERIMENT_RECEIPT_LOGICAL_SHA256,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RECEIPT_ID = "macbert_24_targeted_repair_v3_preregistration_freeze_v1"
_SINGLE_GPU_ID = re.compile(r"[0-9]+")
_RUNTIME_PACKAGES = frozenset(
    {"torch", "transformers", "tokenizers", "safetensors", "PyYAML", "numpy"}
)


class TargetedRepairV3ExperimentFreezeError(ValueError):
    """Raised without reflecting local paths, environment values, or secrets."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _regular_file(path: Path, *, field: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or any(parent.is_symlink() for parent in path.parents)
    ):
        raise TargetedRepairV3ExperimentFreezeError(f"{field} must be a regular file")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TargetedRepairV3ExperimentFreezeError(f"{field} must be an object")
    return value


def _frozen_timestamp(receipt: Mapping[str, Any], receipt_path: Path) -> int:
    frozen_at_ns = receipt.get("frozen_at_ns")
    frozen_at = receipt.get("frozen_at")
    if (
        isinstance(frozen_at_ns, bool)
        or not isinstance(frozen_at_ns, int)
        or not isinstance(frozen_at, str)
    ):
        raise TargetedRepairV3ExperimentFreezeError("freeze timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(frozen_at)
    except ValueError as exc:
        raise TargetedRepairV3ExperimentFreezeError("freeze timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TargetedRepairV3ExperimentFreezeError("freeze timestamp must be timezone-aware")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    parsed_ns = (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
    receipt_mtime_ns = receipt_path.stat().st_mtime_ns
    now_ns = time.time_ns()
    if (
        parsed_ns != frozen_at_ns
        or receipt_mtime_ns < frozen_at_ns
        or now_ns < receipt_mtime_ns
        or now_ns < frozen_at_ns
    ):
        raise TargetedRepairV3ExperimentFreezeError("freeze timestamp ordering is inconsistent")
    return frozen_at_ns


def verify_targeted_repair_v3_experiment_freeze(
    *, receipt_path: Path, config_path: Path
) -> Mapping[str, Any]:
    """Verify the exact receipt, config, result code, and freeze ordering."""

    if (
        EXPERIMENT_RECEIPT_FILE_SHA256 == "UNFROZEN"
        or EXPERIMENT_RECEIPT_LOGICAL_SHA256 == "UNFROZEN"
    ):
        raise TargetedRepairV3ExperimentFreezeError("targeted repair v3 experiment is not frozen")
    _regular_file(receipt_path, field="experiment freeze receipt")
    if _sha256_file(receipt_path) != EXPERIMENT_RECEIPT_FILE_SHA256:
        raise TargetedRepairV3ExperimentFreezeError("experiment receipt byte identity changed")
    try:
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV3ExperimentFreezeError("experiment receipt is unreadable") from exc
    receipt = _mapping(document, field="experiment receipt")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        claimed != EXPERIMENT_RECEIPT_LOGICAL_SHA256
        or _canonical_hash(unsigned) != claimed
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_id") != _RECEIPT_ID
        or receipt.get("status") != "preregistered_frozen"
    ):
        raise TargetedRepairV3ExperimentFreezeError("experiment receipt identity changed")
    frozen_at_ns = _frozen_timestamp(receipt, receipt_path)

    _regular_file(config_path, field="experiment config")
    config = _mapping(receipt.get("config"), field="experiment config binding")
    expected_config_relative = "configs/experiments/macbert_24_targeted_repair_v3.yaml"
    if (
        config.get("repository_relative_path") != expected_config_relative
        or config_path.resolve(strict=True)
        != (_REPOSITORY_ROOT / expected_config_relative).resolve()
        or _sha256_file(config_path) != config.get("file_sha256")
        or config_path.stat().st_mtime_ns > frozen_at_ns
    ):
        raise TargetedRepairV3ExperimentFreezeError("experiment config identity changed")
    try:
        config_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TargetedRepairV3ExperimentFreezeError("experiment config is unreadable") from exc
    config_root = _mapping(config_document, field="experiment config")

    implementations = _mapping(receipt.get("implementations"), field="implementation bindings")
    if set(implementations) != set(EXPERIMENT_IMPLEMENTATION_PATHS):
        raise TargetedRepairV3ExperimentFreezeError("implementation inventory changed")
    for name, binding_value in implementations.items():
        binding = _mapping(binding_value, field="implementation binding")
        relative = binding.get("repository_relative_path")
        expected = binding.get("file_sha256")
        if (
            relative != EXPERIMENT_IMPLEMENTATION_PATHS[name]
            or not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
        ):
            raise TargetedRepairV3ExperimentFreezeError("implementation binding is unsafe")
        path = _REPOSITORY_ROOT / relative
        _regular_file(path, field="frozen implementation")
        if _sha256_file(path) != expected or path.stat().st_mtime_ns > frozen_at_ns:
            raise TargetedRepairV3ExperimentFreezeError("frozen implementation identity changed")
    config_implementations = _mapping(
        config_root.get("implementation_closure"), field="config implementation closure"
    )
    if dict(config_implementations) != dict(implementations):
        raise TargetedRepairV3ExperimentFreezeError("config implementation closure changed")

    anchor = _mapping(receipt.get("anchor"), field="anchor contract")
    if anchor != {
        "repository_relative_path": ("src/pii_zh/training/targeted_repair_v3_experiment_anchor.py"),
        "excluded_from_self_referential_receipt": True,
        "final_file_sha256_reported_before_gpu_execution": True,
    }:
        raise TargetedRepairV3ExperimentFreezeError("experiment anchor contract changed")
    cross_bindings = {
        "data": "data",
        "base_model": "base_model",
        "runtime": "runtime_lock",
        "checkpoint_selection": "checkpoint_selection",
    }
    for receipt_field, config_field in cross_bindings.items():
        receipt_value = _mapping(receipt.get(receipt_field), field=receipt_field)
        config_value = _mapping(config_root.get(config_field), field=config_field)
        if dict(receipt_value) != dict(config_value):
            raise TargetedRepairV3ExperimentFreezeError(
                f"{receipt_field} differs from the frozen config"
            )
    return {
        "receipt_id": _RECEIPT_ID,
        "receipt_sha256": claimed,
        "receipt_file_sha256": EXPERIMENT_RECEIPT_FILE_SHA256,
        "anchor_file_sha256": _sha256_file(
            Path(__file__).with_name("targeted_repair_v3_experiment_anchor.py")
        ),
        "config_file_sha256": config["file_sha256"],
        "frozen_at": receipt["frozen_at"],
        "frozen_at_ns": frozen_at_ns,
        "implementation_count": len(implementations),
        "implementations": dict(implementations),
        "data": receipt["data"],
        "base_model": receipt["base_model"],
        "runtime": receipt["runtime"],
        "checkpoint_selection": receipt["checkpoint_selection"],
    }


def verify_targeted_repair_v3_runtime(
    runtime: Mapping[str, Any], *, require_gpu: bool
) -> Mapping[str, Any]:
    """Verify result-determining Python packages and optional single-GPU isolation."""

    python = _mapping(runtime.get("python"), field="Python runtime")
    packages = _mapping(runtime.get("packages"), field="runtime packages")
    gpu = _mapping(runtime.get("gpu"), field="GPU runtime")
    if (
        set(python) != {"implementation", "version", "executable_sha256"}
        or set(packages) != _RUNTIME_PACKAGES
        or set(gpu)
        != {
            "model",
            "driver_version",
            "maximum_preflight_memory_mib",
            "maximum_preflight_utilization_percent",
        }
        or set(runtime)
        != {
            "python",
            "packages",
            "torch_build",
            "torch_cuda_runtime",
            "gpu",
        }
    ):
        raise TargetedRepairV3ExperimentFreezeError("runtime lock inventory changed")
    if (
        python.get("implementation") != platform.python_implementation()
        or python.get("version") != platform.python_version()
        or python.get("executable_sha256")
        != _sha256_file(Path(sys.executable).resolve(strict=True))
    ):
        raise TargetedRepairV3ExperimentFreezeError("Python runtime identity changed")
    observed_packages: dict[str, str] = {}
    for distribution, expected in packages.items():
        if not isinstance(expected, str):
            raise TargetedRepairV3ExperimentFreezeError("runtime package binding is malformed")
        try:
            observed_packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TargetedRepairV3ExperimentFreezeError("runtime package is missing") from exc
    if observed_packages != dict(packages):
        raise TargetedRepairV3ExperimentFreezeError("runtime package versions changed")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise TargetedRepairV3ExperimentFreezeError("offline model-loading environment is required")

    import torch

    if torch.__version__ != runtime.get("torch_build") or torch.version.cuda != runtime.get(
        "torch_cuda_runtime"
    ):
        raise TargetedRepairV3ExperimentFreezeError("PyTorch CUDA build identity changed")
    result: dict[str, Any] = {
        "python": dict(python),
        "packages": observed_packages,
        "torch_build": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "offline_model_loading": True,
        "gpu_verified": False,
    }
    if not require_gpu:
        return result
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _SINGLE_GPU_ID.fullmatch(visible) is None:
        raise TargetedRepairV3ExperimentFreezeError("exactly one physical GPU must be selected")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TargetedRepairV3ExperimentFreezeError("GPU preflight query failed") from exc
    selected: tuple[str, str, str, str, int, int] | None = None
    for line in completed.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 6 or fields[0] != visible:
            continue
        try:
            selected = (
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                int(fields[4]),
                int(fields[5]),
            )
        except ValueError as exc:
            raise TargetedRepairV3ExperimentFreezeError("GPU preflight output changed") from exc
        break
    for key in ("maximum_preflight_memory_mib", "maximum_preflight_utilization_percent"):
        if isinstance(gpu.get(key), bool) or not isinstance(gpu.get(key), int):
            raise TargetedRepairV3ExperimentFreezeError("GPU runtime lock is malformed")
    if (
        selected is None
        or selected[2] != gpu.get("model")
        or selected[3] != gpu.get("driver_version")
        or selected[4] > gpu["maximum_preflight_memory_mib"]
        or selected[5] > gpu["maximum_preflight_utilization_percent"]
    ):
        raise TargetedRepairV3ExperimentFreezeError("selected GPU is not idle and compatible")
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
        raise TargetedRepairV3ExperimentFreezeError("GPU process preflight query failed") from exc
    if any(
        tuple(field.strip() for field in line.split(","))[0] == selected[1]
        for line in compute_apps.stdout.splitlines()
        if line.strip()
    ):
        raise TargetedRepairV3ExperimentFreezeError("selected GPU has a compute process")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TargetedRepairV3ExperimentFreezeError("single visible CUDA device is unavailable")
    if torch.cuda.get_device_name(0) != gpu.get("model"):
        raise TargetedRepairV3ExperimentFreezeError("visible CUDA device identity changed")
    result["gpu_verified"] = True
    result["gpu"] = {
        "model": selected[2],
        "driver_version": selected[3],
        "preflight_memory_mib": selected[4],
        "preflight_utilization_percent": selected[5],
        "single_visible_device": True,
    }
    return result


__all__ = [
    "TargetedRepairV3ExperimentFreezeError",
    "verify_targeted_repair_v3_experiment_freeze",
    "verify_targeted_repair_v3_runtime",
]
