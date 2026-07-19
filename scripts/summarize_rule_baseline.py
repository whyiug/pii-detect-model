#!/usr/bin/env python3
"""Create a publication-safe, content-addressed rule-baseline summary."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import (  # noqa: E402
    PUBLIC_SCORE_DECIMALS,
    add_manifest_hash,
    canonical_json_hash,
    sha256_file,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return value


def _number(value: object, *, field: str, allow_none: bool = False) -> int | float | None:
    if allow_none and value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"invalid numeric {field}")
    return round(value, PUBLIC_SCORE_DECIMALS) if isinstance(value, float) else value


def _load_report(path: Path, *, expected_subset: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{expected_subset} evaluation report is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{expected_subset} evaluation report must be an object")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{expected_subset} report lacks bound evaluation provenance")
    gold = provenance.get("gold")
    dataset = gold.get("dataset") if isinstance(gold, Mapping) else None
    if not isinstance(dataset, Mapping) or dataset.get("subset") != expected_subset:
        raise ValueError(f"{expected_subset} report is not bound to that frozen subset")
    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or not bootstrap.get("samples"):
        raise ValueError(f"{expected_subset} report lacks bootstrap confidence intervals")
    return value


def _micro_metrics(value: object, *, field: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing {field} metrics")
    allowed = ("tp", "fp", "fn", "support", "predicted", "precision", "recall", "f1", "f2")
    result: dict[str, int | float] = {}
    for key in allowed:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"invalid {field}.{key} metric")
        normalized = _number(item, field=f"{field}.{key}")
        assert normalized is not None
        result[key] = normalized
    return result


def _macro_metrics(value: object, *, field: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing {field} metrics")
    result: dict[str, int | float] = {}
    for key in ("label_count", "precision", "recall", "f1", "f2"):
        normalized = _number(value.get(key), field=f"{field}.{key}")
        assert normalized is not None
        result[key] = normalized
    return result


def _selected_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    strict = report.get("strict")
    relaxed = report.get("relaxed")
    character = report.get("character")
    if not all(isinstance(value, Mapping) for value in (strict, relaxed, character)):
        raise ValueError("evaluation report is missing required metric families")
    calibration = report.get("confidence_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("evaluation report is missing confidence calibration evidence")
    calibration_summary = None
    if isinstance(calibration, Mapping):
        status = calibration.get("status")
        reason = calibration.get("reason")
        if status not in {"available", "not_applicable"}:
            raise ValueError("invalid confidence_calibration.status")
        if reason not in {None, "no_prediction_spans", "scores_absent", "scores_incomplete"}:
            raise ValueError("invalid confidence_calibration.reason")
        if calibration.get("scope") != "emitted_prediction_spans":
            raise ValueError("invalid confidence calibration scope")
        if calibration.get("target") != "strict_start_end_label_match":
            raise ValueError("invalid confidence calibration target")
        binning = calibration.get("binning")
        if not isinstance(binning, Mapping) or binning.get("method") != "equal_width":
            raise ValueError("invalid confidence calibration binning")
        calibration_summary = {
            "status": status,
            "reason": reason,
            "scope": "emitted_prediction_spans",
            "target": "strict_start_end_label_match",
            "predicted_span_count": _number(
                calibration.get("predicted_span_count"), field="predicted_span_count"
            ),
            "scored_span_count": _number(
                calibration.get("scored_span_count"), field="scored_span_count"
            ),
            "binning": {
                "method": "equal_width",
                "bin_count": _number(binning.get("bin_count"), field="bin_count"),
            },
            "ece": _number(calibration.get("ece"), field="ece", allow_none=True),
            "brier": _number(calibration.get("brier"), field="brier", allow_none=True),
        }
    pii_free = report.get("pii_free")
    if not isinstance(pii_free, Mapping):
        raise ValueError("evaluation report is missing pii_free metrics")
    span_count = report.get("span_count")
    if not isinstance(span_count, Mapping):
        raise ValueError("evaluation report is missing span_count metrics")
    bootstrap = report["bootstrap"]
    intervals = bootstrap.get("intervals")
    if not isinstance(intervals, Mapping):
        raise ValueError("bootstrap intervals must be an object")
    safe_intervals: dict[str, dict[str, int | float | None]] = {}
    allowed_intervals = {
        "strict_micro_f1",
        "strict_macro_f1",
        "relaxed_micro_f1",
        "character_micro_f1",
        "boundary_exact_f1",
        "pii_free_false_positive_rate",
    }
    missing_intervals = allowed_intervals - set(intervals)
    if missing_intervals:
        raise ValueError("bootstrap report is missing required intervals")
    for name in allowed_intervals:
        interval = intervals.get(name)
        if interval is None:
            continue
        if not isinstance(interval, Mapping):
            raise ValueError(f"invalid bootstrap interval {name}")
        safe_intervals[name] = {
            "point": _number(interval.get("point"), field=f"{name}.point", allow_none=True),
            "lower": _number(interval.get("lower"), field=f"{name}.lower", allow_none=True),
            "upper": _number(interval.get("upper"), field=f"{name}.upper", allow_none=True),
            "effective_samples": _number(
                interval.get("effective_samples"), field=f"{name}.effective_samples"
            ),
        }
    if bootstrap.get("method") != "document_level_percentile_bootstrap":
        raise ValueError("invalid bootstrap method")
    return {
        "document_count": _number(report.get("document_count"), field="document_count"),
        "span_count": {
            key: _number(span_count.get(key), field=f"span_count.{key}")
            for key in ("gold", "predicted")
        },
        "strict_micro": _micro_metrics(strict.get("micro"), field="strict.micro"),
        "strict_macro": _macro_metrics(strict.get("macro"), field="strict.macro"),
        "relaxed_micro": _micro_metrics(relaxed.get("micro"), field="relaxed.micro"),
        "character_micro": _micro_metrics(character.get("micro"), field="character.micro"),
        "pii_free": {
            "documents": _number(pii_free.get("documents"), field="pii_free.documents"),
            "false_positive_documents": _number(
                pii_free.get("false_positive_documents"),
                field="pii_free.false_positive_documents",
            ),
            "false_positive_rate": _number(
                pii_free.get("false_positive_rate"),
                field="pii_free.false_positive_rate",
                allow_none=True,
            ),
        },
        "confidence_calibration": calibration_summary,
        "bootstrap": {
            "method": "document_level_percentile_bootstrap",
            "confidence": _number(bootstrap.get("confidence"), field="bootstrap.confidence"),
            "samples": _number(bootstrap.get("samples"), field="bootstrap.samples"),
            "seed": _number(bootstrap.get("seed"), field="bootstrap.seed"),
            "intervals": safe_intervals,
        },
    }


def _safe_dataset_identity(value: object, *, subset: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("subset") != subset:
        raise ValueError(f"invalid {subset} dataset identity")
    if value.get("evaluation_only") is not True:
        raise ValueError(f"{subset} dataset identity is not evaluation_only")
    expected_identity = {
        "source_id": "pii_bench_zh",
        "upstream_source": "wan9yu/pii-bench-zh",
        "upstream_revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
        "license": "Apache-2.0",
    }
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        raise ValueError(f"{subset} report is not the frozen PII Bench ZH identity")
    return {
        **expected_identity,
        "subset": subset,
        "gold_sha256": _sha256(value.get("gold_sha256"), field=f"{subset}.gold"),
        "record_count": _number(value.get("record_count"), field=f"{subset}.record_count"),
        "span_count": _number(value.get("span_count"), field=f"{subset}.span_count"),
        "manifest_sha256": _sha256(
            value.get("manifest_sha256"), field=f"{subset}.dataset_manifest"
        ),
        "manifest_file_sha256": _sha256(
            value.get("manifest_file_sha256"), field=f"{subset}.dataset_manifest_file"
        ),
        "evaluation_only": True,
    }


def _summarize(path: Path, *, subset: str) -> tuple[dict[str, Any], str]:
    report = _load_report(path, expected_subset=subset)
    provenance = report["provenance"]
    dataset = _safe_dataset_identity(provenance["gold"]["dataset"], subset=subset)
    predictions = provenance.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError(f"{subset} report lacks prediction provenance")
    return (
        {
            "evaluation_report_sha256": sha256_file(path),
            "evaluation_sha256": _sha256(
                provenance.get("evaluation_sha256"), field=f"{subset}.evaluation"
            ),
            "dataset": dataset,
            "predictions_sha256": _sha256(predictions.get("sha256"), field=f"{subset}.predictions"),
            "metrics": _selected_metrics(report),
        },
        dataset["manifest_sha256"],
    )


def _write_json(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--chat-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: stdout")
    return parser


def main() -> int:
    args = _parser().parse_args()
    formal, formal_manifest = _summarize(args.formal_report, subset="formal")
    chat, chat_manifest = _summarize(args.chat_report, subset="chat")
    if formal_manifest != chat_manifest:
        raise ValueError("formal and chat reports use different dataset manifests")

    rule_files = {
        "cn_common.py": sha256_file(_REPOSITORY_ROOT / "src/pii_zh/rules/cn_common.py"),
        "predict_rules.py": sha256_file(_REPOSITORY_ROOT / "scripts/predict_rules.py"),
        "validators.py": sha256_file(_REPOSITORY_ROOT / "src/pii_zh/data/validators/__init__.py"),
    }
    summary = add_manifest_hash(
        {
            "schema_version": 1,
            "artifact_type": "rule_baseline_summary",
            "baseline": {
                "id": "cn_common_rules",
                "kind": "open_source_framework_rules",
                "implementation_sha256": canonical_json_hash(rule_files),
                "component_sha256": rule_files,
            },
            "comparison_contract": {
                "track": "rules_only",
                "metric": "strict_exact_start_end_label",
                "benchmark_protocol": "pii_bench_zh_closed_8_v1",
                "target_labels": [
                    "ADDRESS",
                    "BANK_CARD_NUMBER",
                    "CN_RESIDENT_ID",
                    "EMAIL_ADDRESS",
                    "PASSPORT_NUMBER",
                    "PERSON_NAME",
                    "PHONE_NUMBER",
                    "VEHICLE_LICENSE_PLATE",
                ],
                "supported_target_labels": [
                    "BANK_CARD_NUMBER",
                    "CN_RESIDENT_ID",
                    "EMAIL_ADDRESS",
                    "PASSPORT_NUMBER",
                    "PHONE_NUMBER",
                    "VEHICLE_LICENSE_PLATE",
                ],
                "unsupported_protocol_labels_count_as_false_negatives": True,
            },
            "dataset_manifest_sha256": formal_manifest,
            "subsets": {"formal": formal, "chat": chat},
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
            },
        }
    )
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        _write_json(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
