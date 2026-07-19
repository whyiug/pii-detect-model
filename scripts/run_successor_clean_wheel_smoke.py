#!/usr/bin/env python3
"""Verify installed v1/v2 rules-only service paths without importing repo source."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import pii_zh
from pii_zh.cascade import (
    LEGACY_SERVICE_PROFILE_VERSION,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    build_rules_only_service_pipeline,
)
from pii_zh.presidio import create_analyzer_engine
from pii_zh.service import create_app

ENTITY = "CN_VEHICLE_LICENSE_PLATE"


def _presidio_core(item: Any, schema_version: str) -> dict[str, Any]:
    metadata = item.recognition_metadata
    return {
        "schema_version": schema_version,
        "start": item.start,
        "end": item.end,
        "entity_type": item.entity_type,
        "score": item.score,
        "source": metadata["source"],
        "sources": metadata["sources"],
        "decision_process": metadata["decision_process"],
    }


def _run_cli(
    executable: Path,
    arguments: list[str],
    *,
    text: str,
    cwd: Path,
) -> tuple[dict[str, Any], str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": "",
        }
    )
    completed = subprocess.run(
        [str(executable), *arguments],
        input=text,
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
        env=environment,
        timeout=30,
    )
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError(
            f"installed CLI failed without a clean stderr (status={completed.returncode})"
        )
    return json.loads(completed.stdout), completed.stdout


def _run_help(executable: Path, *, cwd: Path) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": "",
        }
    )
    completed = subprocess.run(
        [str(executable), "--help"],
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
        env=environment,
        timeout=30,
    )
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError(
            "installed entry-point help failed without a clean stderr "
            f"(status={completed.returncode})"
        )
    return completed.stdout


def main() -> int:
    prefix = Path(sys.prefix).resolve()
    module_path = Path(pii_zh.__file__).resolve()
    assert sys.prefix != sys.base_prefix, "smoke must run inside an isolated virtual environment"
    assert module_path.is_relative_to(prefix), "pii_zh must import from the clean environment"
    assert "site-packages" in module_path.parts

    cli_executable = prefix / "bin" / "pii-zh"
    ablation_executable = prefix / "bin" / "pii-zh-evaluate-ablation"
    assert cli_executable.is_file()
    assert ablation_executable.is_file()
    runtime_cwd = prefix.parent / "runtime-cwd"
    runtime_cwd.mkdir(mode=0o700, exist_ok=True)

    distribution = importlib.metadata.distribution("pii-zh-qwen")
    installed_entrypoints = {
        entrypoint.name: entrypoint.value
        for entrypoint in distribution.entry_points
        if entrypoint.group == "console_scripts"
    }
    assert installed_entrypoints == {
        "pii-zh": "pii_zh.cli:main",
        "pii-zh-evaluate-ablation": "pii_zh.evaluation.cascade_ablation_cli:main",
    }
    ablation_help = _run_help(ablation_executable, cwd=runtime_cwd)
    assert "--adapter" in ablation_help
    assert "--run-intent" in ablation_help

    legacy_text = "车辆模型A12345已加载。"
    legacy = build_rules_only_service_pipeline()
    assert legacy.config.profile_version == LEGACY_SERVICE_PROFILE_VERSION
    legacy_detections = legacy.detect(legacy_text, entities=[ENTITY])
    assert len(legacy_detections) == 1
    assert legacy_detections[0].source == "rule:cn_common"
    legacy_direct = [item.to_dict() for item in legacy_detections]

    legacy_cli, legacy_cli_raw = _run_cli(
        cli_executable,
        ["detect", "--entity", ENTITY],
        text=legacy_text,
        cwd=runtime_cwd,
    )
    assert legacy_cli["detections"] == legacy_direct
    assert "A12345" not in legacy_cli_raw

    legacy_analyzer = create_analyzer_engine()
    legacy_presidio = legacy_analyzer.analyze(
        text=legacy_text,
        language="zh",
        entities=[ENTITY],
    )
    legacy_presidio_core = [
        _presidio_core(item, direct_item["schema_version"])
        for item, direct_item in zip(legacy_presidio, legacy_direct, strict=True)
    ]
    assert legacy_presidio_core == legacy_direct

    legacy_app = create_app()
    assert legacy_app.state.pipeline.config.profile_version == LEGACY_SERVICE_PROFILE_VERSION
    with TestClient(legacy_app) as client:
        legacy_http = client.post(
            "/v1/analyze",
            json={"text": legacy_text, "entities": [ENTITY]},
        )
    assert legacy_http.status_code == 200
    assert legacy_http.json()["detections"] == legacy_direct
    assert "A12345" not in legacy_http.text

    plates = ("京A12B34", "沪AD12345", "粤B12345F")
    valid_text = f"普通{plates[0]}，新能源{plates[1]}和{plates[2]}。"
    invalid_text = "车辆模型A12345已加载；车辆岚A·AB12CD3已核验。"
    successor = build_rules_only_service_pipeline(SUCCESSOR_SERVICE_PROFILE_VERSION)
    assert successor.config.profile_version == SUCCESSOR_SERVICE_PROFILE_VERSION
    assert successor.config.rule_source == "rule:cn_common_v6"
    route = successor.config.route_map[ENTITY]
    assert route.rule_enabled and not route.model_enabled
    assert route.validator_label == ENTITY

    direct = [item.to_dict() for item in successor.detect(valid_text, entities=[ENTITY])]
    assert len(direct) == len(plates)
    assert [valid_text[item["start"] : item["end"]] for item in direct] == list(plates)
    assert all(item["source"] == "rule:cn_common_v6" for item in direct)
    assert all("validator:passed" in item["decision_process"] for item in direct)
    assert successor.detect(invalid_text, entities=[ENTITY]) == []

    cli_payload, cli_raw = _run_cli(
        cli_executable,
        [
            "detect",
            "--profile",
            SUCCESSOR_SERVICE_PROFILE_VERSION,
            "--entity",
            ENTITY,
        ],
        text=valid_text,
        cwd=runtime_cwd,
    )
    assert cli_payload["detections"] == direct
    assert all(plate not in cli_raw for plate in plates)
    cli_invalid, _ = _run_cli(
        cli_executable,
        [
            "detect",
            "--profile",
            SUCCESSOR_SERVICE_PROFILE_VERSION,
            "--entity",
            ENTITY,
        ],
        text=invalid_text,
        cwd=runtime_cwd,
    )
    assert cli_invalid["detections"] == []
    cli_redact, cli_redact_raw = _run_cli(
        cli_executable,
        [
            "redact",
            "--profile",
            SUCCESSOR_SERVICE_PROFILE_VERSION,
            "--entity",
            ENTITY,
            "--replacement",
            "<PLATE>",
        ],
        text=valid_text,
        cwd=runtime_cwd,
    )
    assert cli_redact["redacted_text"] == "普通<PLATE>，新能源<PLATE>和<PLATE>。"
    assert all(plate not in cli_redact_raw for plate in plates)

    successor_app = create_app(profile_version=SUCCESSOR_SERVICE_PROFILE_VERSION)
    assert successor_app.state.pipeline.config.profile_version == SUCCESSOR_SERVICE_PROFILE_VERSION
    with TestClient(successor_app) as client:
        health = client.get("/healthz")
        analyze = client.post(
            "/v1/analyze",
            json={"text": valid_text, "entities": [ENTITY]},
        )
        redact = client.post(
            "/v1/redact",
            json={"text": valid_text, "entities": [ENTITY], "replacement": "<PLATE>"},
        )
        invalid = client.post(
            "/v1/analyze",
            json={"text": invalid_text, "entities": [ENTITY]},
        )
    assert health.status_code == 200 and health.json()["mode"] == "rules-only"
    assert analyze.status_code == 200 and analyze.json()["detections"] == direct
    assert redact.status_code == 200
    assert redact.json()["redacted_text"] == "普通<PLATE>，新能源<PLATE>和<PLATE>。"
    assert invalid.status_code == 200 and invalid.json()["detections"] == []
    assert all(plate not in analyze.text and plate not in redact.text for plate in plates)

    analyzer = create_analyzer_engine(cascade_pipeline=successor)
    presidio = analyzer.analyze(text=valid_text, language="zh", entities=[ENTITY])
    presidio_core = [
        _presidio_core(item, direct_item["schema_version"])
        for item, direct_item in zip(presidio, direct, strict=True)
    ]
    assert presidio_core == direct
    assert analyzer.analyze(text=invalid_text, language="zh", entities=[ENTITY]) == []
    assert all(
        item.recognition_metadata["cascade_profile_version"] == SUCCESSOR_SERVICE_PROFILE_VERSION
        for item in presidio
    )

    model_stack_modules = {
        name
        for name in sys.modules
        if name == "pii_zh.inference"
        or name.startswith("pii_zh.inference.")
        or name == "pii_zh.models"
        or name.startswith("pii_zh.models.")
    }
    assert not model_stack_modules, (
        f"rules-only smoke imported project model stacks: {sorted(model_stack_modules)}"
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    output = {
        "status": "PASS",
        "profile_version": SUCCESSOR_SERVICE_PROFILE_VERSION,
        "historical_default_unchanged": True,
        "installed_from_clean_site_packages": True,
        "installed_module_path_recorded": False,
        "python": sys.version.split()[0],
        "distribution_versions": {
            name: importlib.metadata.version(name)
            for name in ("pii-zh-qwen", "presidio-analyzer", "fastapi", "httpx")
        },
        "paths": [
            "python",
            "installed-cli",
            "installed-ablation-cli-help",
            "presidio",
            "http",
        ],
        "installed_entrypoints": installed_entrypoints,
        "historical_paths": ["python", "installed-cli", "presidio", "http"],
        "valid_case_count": len(plates),
        "invalid_document_count": 1,
        "raw_values_persisted": False,
        "model_packages_imported": False,
        "framework_torch_imported": "torch" in sys.modules,
        "framework_transformers_imported": "transformers" in sys.modules,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
