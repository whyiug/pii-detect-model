"""Narrow host-runtime isolation for tests of immutable historical freezes.

The v4/v5b evidence files intentionally bind the exact interpreter and package
versions used for their original launch.  The project uv environment is newer.
These fixtures keep the production verifier fail-closed while allowing three
unit tests to exercise their stated lineage/GPU behavior on the current host.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from pii_zh.training import targeted_repair_v4_experiment_freeze as experiment_freeze

_ROOT = Path(__file__).resolve().parents[2]
_V4_CONFIG = _ROOT / "configs/experiments/macbert_24_targeted_repair_v4.yaml"
_GPU_PREFLIGHT_NODEIDS = {
    (
        "tests/unit/test_train_macbert_targeted_repair_v4.py::"
        "test_gpu_busy_preflight_fails_before_any_cuda_call"
    ),
    (
        "tests/unit/test_train_macbert_targeted_repair_v4.py::"
        "test_gpu_idle_preflight_is_recorded_without_process_identifiers"
    ),
}
_PARENT_LINEAGE_NODEID = (
    "tests/unit/test_train_macbert_numeric_o_v5b_pilot.py::"
    "test_parent_v5a_failed_run_is_bound_and_weightless"
)


def _frozen_runtime_lock() -> Mapping[str, object]:
    document = yaml.safe_load(_V4_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(
        document.get("runtime_lock"), Mapping
    ):
        raise AssertionError("frozen v4 runtime lock is unavailable")
    return document["runtime_lock"]


def _require_unchanged_lock(runtime_lock: object) -> Mapping[str, object]:
    frozen = _frozen_runtime_lock()
    if not isinstance(runtime_lock, Mapping) or dict(runtime_lock) != dict(frozen):
        raise experiment_freeze.TargetedRepairV4ExperimentFreezeError(
            "runtime lock changed"
        )
    return frozen


def _host_compatible_copy(runtime_lock: object) -> dict[str, object]:
    frozen = _require_unchanged_lock(runtime_lock)
    adjusted = copy.deepcopy(dict(frozen))
    adjusted["python"] = {
        "implementation": "CPython",
        "version": platform.python_version(),
        "executable_sha256": experiment_freeze._sha(
            Path(sys.executable).resolve(strict=True)
        ),
    }
    adjusted["packages"] = {
        name: importlib.metadata.version(distribution)
        for name, distribution in experiment_freeze._PACKAGE_DISTRIBUTIONS.items()
    }
    adjusted["torch_build"] = experiment_freeze.torch.__version__
    adjusted["torch_cuda_runtime"] = experiment_freeze.torch.version.cuda
    return adjusted


@pytest.fixture(autouse=True)
def isolate_historical_runtime_attestation(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate only the three host-independent assertions named above."""

    if request.node.nodeid in _GPU_PREFLIGHT_NODEIDS:
        production_runtime = experiment_freeze._runtime

        def current_host_runtime(
            runtime_lock: object, *, require_gpu: bool
        ) -> dict[str, object]:
            return production_runtime(
                _host_compatible_copy(runtime_lock), require_gpu=require_gpu
            )

        monkeypatch.setattr(experiment_freeze, "_runtime", current_host_runtime)
        return

    if request.node.nodeid == _PARENT_LINEAGE_NODEID:

        def frozen_lineage_without_host_attestation(
            runtime_lock: object, *, require_gpu: bool
        ) -> dict[str, object]:
            frozen = _require_unchanged_lock(runtime_lock)
            if require_gpu:
                raise AssertionError("parent lineage test must remain CPU-only")
            offline = frozen.get("offline_model_loading")
            if not isinstance(offline, Mapping) or any(
                os.environ.get(name) != value for name, value in offline.items()
            ):
                raise experiment_freeze.TargetedRepairV4ExperimentFreezeError(
                    "offline model-loading environment is required"
                )
            return {
                "python": dict(frozen["python"])["version"],
                "torch": frozen["torch_build"],
                "torch_cuda_runtime": frozen["torch_cuda_runtime"],
                "offline_model_loading": True,
                "gpu_required": False,
                "cuda_visible_devices": None,
                "gpu_verified": False,
            }

        monkeypatch.setattr(
            experiment_freeze,
            "_runtime",
            frozen_lineage_without_host_attestation,
        )
