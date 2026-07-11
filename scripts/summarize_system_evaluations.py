#!/usr/bin/env python3
"""Aggregate three release-bound frozen system evaluations without record data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import add_manifest_hash, canonical_json_hash  # noqa: E402

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_MAXIMUM_REPORT_BYTES = 64 * 1024 * 1024
_SUITE_ORDER = ("synthetic_test", "pii_bench_formal", "pii_bench_chat")
_CALIBRATION_FIT_GOLD_SHA256 = "9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d"
_CALIBRATION_FIT_DOCUMENT_COUNT = 2_000
_REQUIRED_RULESET_ID = "cn_common_v5"
_REQUIRED_REFINEMENT_ID = "structured_refinement_v4"
_BOOTSTRAP_INTERVALS = {
    "boundary_exact_f1",
    "character_micro_f1",
    "pii_free_false_positive_rate",
    "relaxed_micro_f1",
    "strict_macro_f1",
    "strict_micro_f1",
}
_SUITE_IDENTITIES: dict[str, dict[str, Any]] = {
    "synthetic_test": {
        "source_id": "pii_zh_synthetic_v1",
        "upstream_source": "repo_curated_synthetic_templates",
        "upstream_revision": ("1dc82af3ac68e8c9c1f20e985304dd9121a01b9db5fe112ef82e3cdc59722d1e"),
        "license": "Apache-2.0",
        "subset": "test",
        "gold_sha256": "26e49949d0bd876b51c693b7b31135b5ce503ca6a5ab7cfca37177e0ccbd4384",
        "record_count": 2_000,
        "span_count": 3_267,
    },
    "pii_bench_formal": {
        "source_id": "pii_bench_zh",
        "upstream_source": "wan9yu/pii-bench-zh",
        "upstream_revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
        "license": "Apache-2.0",
        "subset": "formal",
        "gold_sha256": "bb978137711cc90ae0c16df0645983413f3b34943bf29469e58b795b183a38af",
        "record_count": 5_000,
        "span_count": 15_662,
        "manifest_sha256": "545c990c96f23118ddc097ec0bed1d7f180785ac72e9f2a6ed56eb1ba3ee5e9b",
    },
    "pii_bench_chat": {
        "source_id": "pii_bench_zh",
        "upstream_source": "wan9yu/pii-bench-zh",
        "upstream_revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
        "license": "Apache-2.0",
        "subset": "chat",
        "gold_sha256": "2d73326a7ea2da9966ba13336db1a791f03d5f18b7982fdc4495cb726c1180a5",
        "record_count": 3_000,
        "span_count": 7_544,
        "manifest_sha256": "545c990c96f23118ddc097ec0bed1d7f180785ac72e9f2a6ed56eb1ba3ee5e9b",
    },
}


class SystemEvaluationSummaryError(ValueError):
    """Raised when aggregate system evaluation evidence is incomplete or inconsistent."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemEvaluationSummaryError(f"{field} must be an object")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SystemEvaluationSummaryError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SystemEvaluationSummaryError(f"{field} must be a path-free stable identifier")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SystemEvaluationSummaryError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemEvaluationSummaryError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SystemEvaluationSummaryError(f"{field} must be a finite number")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise SystemEvaluationSummaryError(f"{field} must be between zero and one")
    return result


def _optional_probability(value: object, *, field: str) -> float | None:
    return None if value is None else _probability(value, field=field)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _read_report(path: Path, *, suite: str) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SystemEvaluationSummaryError(f"cannot open regular {suite} report") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemEvaluationSummaryError(f"{suite} report must be a regular file")
        if before.st_size > _MAXIMUM_REPORT_BYTES:
            raise SystemEvaluationSummaryError(f"{suite} report exceeds size limit")
        encoded = bytearray()
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            encoded.extend(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise SystemEvaluationSummaryError(f"{suite} report changed while read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(encoded).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemEvaluationSummaryError(f"{suite} report must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SystemEvaluationSummaryError(f"{suite} report must be a JSON object")
    _assert_public_safe(value)
    return value, digest.hexdigest()


def _assert_public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = {"doc_id", "document_id", "raw_text", "text", "entity_value"}
        if forbidden & {str(key).lower() for key in value}:
            raise SystemEvaluationSummaryError("report contains a forbidden record-level field")
        for item in value.values():
            _assert_public_safe(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_public_safe(item)
    elif isinstance(value, str):
        if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise SystemEvaluationSummaryError("report contains an absolute path")
        if ".." in value.replace("\\", "/").split("/"):
            raise SystemEvaluationSummaryError("report contains an unsafe relative locator")


def _model_identity(value: object, *, suite: str) -> dict[str, str]:
    raw = _mapping(value, field=f"{suite}.system.model_identity")
    allowed = {
        "architecture_version",
        "config_sha256",
        "identity_sha256",
        "label_schema_sha256",
        "model_type",
        "weights_sha256",
    }
    if set(raw) - allowed:
        raise SystemEvaluationSummaryError(f"{suite} model identity has unsupported fields")
    result: dict[str, str] = {}
    for name in ("config_sha256", "identity_sha256", "label_schema_sha256", "weights_sha256"):
        if name in raw:
            result[name] = _sha256(raw[name], field=f"{suite}.model_identity.{name}")
    for name in ("architecture_version", "model_type"):
        if name in raw:
            result[name] = _safe_id(raw[name], field=f"{suite}.model_identity.{name}")
    required = {"config_sha256", "identity_sha256", "model_type", "weights_sha256"}
    if not required <= set(result):
        raise SystemEvaluationSummaryError(f"{suite} model identity is incomplete")
    unsigned = dict(result)
    claimed = unsigned.pop("identity_sha256")
    if canonical_json_hash(unsigned) != claimed:
        raise SystemEvaluationSummaryError(f"{suite} model identity self-hash failed")
    return result


def _score(value: object, *, field: str) -> dict[str, int | float]:
    raw = _mapping(value, field=field)
    tp = _integer(raw.get("tp"), field=f"{field}.tp")
    fp = _integer(raw.get("fp"), field=f"{field}.fp")
    fn = _integer(raw.get("fn"), field=f"{field}.fn")
    support = _integer(raw.get("support"), field=f"{field}.support")
    predicted = _integer(raw.get("predicted"), field=f"{field}.predicted")
    if support != tp + fn or predicted != tp + fp:
        raise SystemEvaluationSummaryError(f"{field} support/predicted counts are inconsistent")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    expected = {"precision": precision, "recall": recall, "f1": f1, "f2": f2}
    for name, calculated in expected.items():
        reported = _probability(raw.get(name), field=f"{field}.{name}")
        if not math.isclose(reported, calculated, rel_tol=0.0, abs_tol=1e-12):
            raise SystemEvaluationSummaryError(f"{field}.{name} does not match its counts")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "predicted": predicted,
        **expected,
    }


def _macro_f1(strict: Mapping[str, Any], *, suite: str) -> float:
    macro = _mapping(strict.get("macro"), field=f"{suite}.strict.macro")
    reported = _probability(macro.get("f1"), field=f"{suite}.strict.macro.f1")
    per_class = _mapping(strict.get("per_class"), field=f"{suite}.strict.per_class")
    if not per_class:
        raise SystemEvaluationSummaryError(f"{suite}.strict.per_class must not be empty")
    f1_values = [
        _score(value, field=f"{suite}.strict.per_class.{label}")["f1"]
        for label, value in per_class.items()
    ]
    calculated = sum(float(item) for item in f1_values) / len(f1_values)
    if not math.isclose(reported, calculated, rel_tol=0.0, abs_tol=1e-12):
        raise SystemEvaluationSummaryError(f"{suite}.strict.macro.f1 does not match per-class F1")
    return reported


def _bootstrap(value: object, *, suite: str) -> dict[str, Any]:
    raw = _mapping(value, field=f"{suite}.bootstrap")
    if raw.get("method") != "document_level_percentile_bootstrap":
        raise SystemEvaluationSummaryError(f"{suite} bootstrap method is invalid")
    if _integer(raw.get("samples"), field=f"{suite}.bootstrap.samples") != 1_000:
        raise SystemEvaluationSummaryError(f"{suite} bootstrap samples must equal 1000")
    if _integer(raw.get("seed"), field=f"{suite}.bootstrap.seed") != 42:
        raise SystemEvaluationSummaryError(f"{suite} bootstrap seed must equal 42")
    confidence = _probability(raw.get("confidence"), field=f"{suite}.bootstrap.confidence")
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise SystemEvaluationSummaryError(f"{suite} bootstrap confidence must equal 0.95")
    intervals = _mapping(raw.get("intervals"), field=f"{suite}.bootstrap.intervals")
    if not _BOOTSTRAP_INTERVALS <= set(intervals):
        raise SystemEvaluationSummaryError(f"{suite} bootstrap intervals are incomplete")
    result: dict[str, Any] = {}
    for name in sorted(_BOOTSTRAP_INTERVALS):
        interval = _mapping(intervals.get(name), field=f"{suite}.bootstrap.{name}")
        point = _optional_probability(interval.get("point"), field=f"{suite}.{name}.point")
        lower = _optional_probability(interval.get("lower"), field=f"{suite}.{name}.lower")
        upper = _optional_probability(interval.get("upper"), field=f"{suite}.{name}.upper")
        effective = _integer(
            interval.get("effective_samples"), field=f"{suite}.{name}.effective_samples"
        )
        if None not in {point, lower, upper} and not float(lower) <= float(point) <= float(upper):
            raise SystemEvaluationSummaryError(f"{suite} bootstrap interval {name} is unordered")
        if effective > 1_000:
            raise SystemEvaluationSummaryError(
                f"{suite} bootstrap interval {name} has too many samples"
            )
        result[name] = {
            "point": point,
            "lower": lower,
            "upper": upper,
            "effective_samples": effective,
        }
    return {
        "method": "document_level_percentile_bootstrap",
        "samples": 1_000,
        "seed": 42,
        "confidence": 0.95,
        "intervals": result,
    }


def _suite_metrics(report: Mapping[str, Any], *, suite: str) -> dict[str, Any]:
    document_count = _integer(report.get("document_count"), field=f"{suite}.document_count")
    spans = _mapping(report.get("span_count"), field=f"{suite}.span_count")
    gold_spans = _integer(spans.get("gold"), field=f"{suite}.span_count.gold")
    predicted_spans = _integer(spans.get("predicted"), field=f"{suite}.span_count.predicted")
    strict = _mapping(report.get("strict"), field=f"{suite}.strict")
    relaxed = _mapping(report.get("relaxed"), field=f"{suite}.relaxed")
    character = _mapping(report.get("character"), field=f"{suite}.character")
    strict_micro = _score(strict.get("micro"), field=f"{suite}.strict.micro")
    relaxed_micro = _score(relaxed.get("micro"), field=f"{suite}.relaxed.micro")
    character_micro = _score(character.get("micro"), field=f"{suite}.character.micro")
    macro_f1 = _macro_f1(strict, suite=suite)
    if strict_micro["support"] != gold_spans or strict_micro["predicted"] != predicted_spans:
        raise SystemEvaluationSummaryError(f"{suite} span counts differ from strict micro counts")
    pii_free = _mapping(report.get("pii_free"), field=f"{suite}.pii_free")
    pii_free_documents = _integer(pii_free.get("documents"), field=f"{suite}.pii_free.documents")
    pii_free_fp_documents = _integer(
        pii_free.get("false_positive_documents"),
        field=f"{suite}.pii_free.false_positive_documents",
    )
    if pii_free_fp_documents > pii_free_documents:
        raise SystemEvaluationSummaryError(f"{suite} PII-free false positives exceed documents")
    fpr = _optional_probability(
        pii_free.get("false_positive_rate"), field=f"{suite}.pii_free.false_positive_rate"
    )
    expected_fpr = pii_free_fp_documents / pii_free_documents if pii_free_documents else None
    if (fpr is None) != (expected_fpr is None) or (
        fpr is not None
        and expected_fpr is not None
        and not math.isclose(fpr, expected_fpr, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise SystemEvaluationSummaryError(f"{suite} PII-free FPR has the wrong denominator")
    return {
        "document_count": document_count,
        "span_count": {"gold": gold_spans, "predicted": predicted_spans},
        "strict_micro": strict_micro,
        "strict_macro_f1": macro_f1,
        "relaxed_micro": relaxed_micro,
        "character_micro": character_micro,
        "pii_free": {
            "documents": pii_free_documents,
            "false_positive_documents": pii_free_fp_documents,
            "false_positive_rate": fpr,
        },
        "bootstrap": _bootstrap(report.get("bootstrap"), suite=suite),
    }


def _dataset_identity(provenance: Mapping[str, Any], *, suite: str) -> dict[str, Any]:
    gold = _mapping(provenance.get("gold"), field=f"{suite}.provenance.gold")
    dataset = _mapping(gold.get("dataset"), field=f"{suite}.provenance.gold.dataset")
    expected = _SUITE_IDENTITIES[suite]
    for field, expected_value in expected.items():
        if dataset.get(field) != expected_value:
            raise SystemEvaluationSummaryError(f"{suite} dataset {field} is not frozen identity")
    if dataset.get("evaluation_only") is not True:
        raise SystemEvaluationSummaryError(f"{suite} dataset is not evaluation_only")
    if gold.get("sha256") != expected["gold_sha256"]:
        raise SystemEvaluationSummaryError(f"{suite} gold SHA-256 differs from dataset identity")
    result = {
        field: dataset[field]
        for field in (
            "source_id",
            "upstream_source",
            "upstream_revision",
            "license",
            "subset",
            "gold_sha256",
            "record_count",
            "span_count",
            "manifest_sha256",
            "manifest_file_sha256",
            "evaluation_only",
        )
        if field in dataset
    }
    _sha256(result.get("manifest_sha256"), field=f"{suite}.dataset.manifest_sha256")
    _sha256(result.get("manifest_file_sha256"), field=f"{suite}.dataset.manifest_file_sha256")
    return result


def _common_system_identity(
    provenance: Mapping[str, Any],
    *,
    suite: str,
    document_count: int,
    dataset_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if provenance.get("mode") != "release":
        raise SystemEvaluationSummaryError(f"{suite} evaluation provenance must be release mode")
    parameters = _mapping(provenance.get("parameters"), field=f"{suite}.provenance.parameters")
    expected_parameters = {
        "bootstrap_samples": 1_000,
        "confidence": 0.95,
        "iou_threshold": 0.5,
        "seed": 42,
    }
    if dict(parameters) != expected_parameters:
        raise SystemEvaluationSummaryError(f"{suite} evaluation parameters are not frozen")
    privacy = _mapping(provenance.get("privacy"), field=f"{suite}.provenance.privacy")
    for name in ("contains_paths", "contains_raw_text", "contains_entity_values"):
        if privacy.get(name) is not False:
            raise SystemEvaluationSummaryError(f"{suite} provenance privacy is unsafe")
    evaluation_sha256 = _sha256(
        provenance.get("evaluation_sha256"), field=f"{suite}.provenance.evaluation_sha256"
    )
    unsigned = dict(provenance)
    unsigned.pop("evaluation_sha256", None)
    if canonical_json_hash(unsigned) != evaluation_sha256:
        raise SystemEvaluationSummaryError(f"{suite} evaluation provenance self-hash failed")

    training = _mapping(
        provenance.get("model_training"), field=f"{suite}.provenance.model_training"
    )
    training_sha256 = _sha256(
        training.get("manifest_sha256"), field=f"{suite}.training.manifest_sha256"
    )
    training_file_sha256 = _sha256(
        training.get("manifest_file_sha256"), field=f"{suite}.training.manifest_file_sha256"
    )
    if training.get("seed") != 42 or training.get("attention_mode") != "full":
        raise SystemEvaluationSummaryError(f"{suite} does not use the selected seed42 Full model")

    predictions = _mapping(provenance.get("predictions"), field=f"{suite}.provenance.predictions")
    predictions_sha256 = _sha256(predictions.get("sha256"), field=f"{suite}.predictions.sha256")
    system = _mapping(predictions.get("manifest"), field=f"{suite}.predictions.system_manifest")
    if system.get("predictor_type") != "system":
        raise SystemEvaluationSummaryError(f"{suite} final prediction is not predictor_type=system")
    if system.get("seed") != 42 or system.get("attention_mode") != "full":
        raise SystemEvaluationSummaryError(f"{suite} system identity is not seed42 Full")
    if system.get("predictions_sha256") != predictions_sha256:
        raise SystemEvaluationSummaryError(f"{suite} system manifest output hash differs")
    if system.get("prediction_document_count") != document_count:
        raise SystemEvaluationSummaryError(f"{suite} system document count differs")
    if system.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise SystemEvaluationSummaryError(f"{suite} system/dataset manifest identities differ")
    system_training_sha256 = _sha256(
        system.get("model_training_manifest_sha256"),
        field=f"{suite}.system.model_training_manifest_sha256",
    )
    if system_training_sha256 != training_sha256:
        raise SystemEvaluationSummaryError(f"{suite} system/training manifest identities differ")
    system_manifest_sha256 = _sha256(
        system.get("manifest_sha256"), field=f"{suite}.system.manifest_sha256"
    )
    system_manifest_file_sha256 = _sha256(
        system.get("manifest_file_sha256"), field=f"{suite}.system.manifest_file_sha256"
    )
    components = _mapping(system.get("components"), field=f"{suite}.system.components")
    expected_components = {
        "model_prediction",
        "rules_prediction",
        "fusion",
        "calibration_fit",
        "target_application",
        "refinement",
    }
    if set(components) != expected_components:
        raise SystemEvaluationSummaryError(f"{suite} system component set is incomplete")
    model = _mapping(components["model_prediction"], field=f"{suite}.system.model_prediction")
    rules = _mapping(components["rules_prediction"], field=f"{suite}.system.rules_prediction")
    fusion = _mapping(components["fusion"], field=f"{suite}.system.fusion")
    fit = _mapping(components["calibration_fit"], field=f"{suite}.system.calibration_fit")
    target = _mapping(components["target_application"], field=f"{suite}.system.target_application")
    refinement = _mapping(components["refinement"], field=f"{suite}.system.refinement")
    if model.get("model_training_manifest_sha256") != training_sha256:
        raise SystemEvaluationSummaryError(f"{suite} model component training identity differs")
    for component_name, component in (("model", model), ("rules", rules)):
        if component.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise SystemEvaluationSummaryError(
                f"{suite} {component_name} component dataset identity differs"
            )
    model_manifest_sha256 = _sha256(
        model.get("manifest_sha256"), field=f"{suite}.model.manifest_sha256"
    )
    model_predictions_sha256 = _sha256(
        model.get("predictions_sha256"), field=f"{suite}.model.predictions_sha256"
    )
    rules_manifest_sha256 = _sha256(
        rules.get("manifest_sha256"), field=f"{suite}.rules.manifest_sha256"
    )
    rules_predictions_sha256 = _sha256(
        rules.get("predictions_sha256"), field=f"{suite}.rules.predictions_sha256"
    )
    if (
        fusion.get("model_prediction_manifest_sha256") != model_manifest_sha256
        or fusion.get("model_predictions_sha256") != model_predictions_sha256
        or fusion.get("rules_prediction_manifest_sha256") != rules_manifest_sha256
        or fusion.get("rules_predictions_sha256") != rules_predictions_sha256
    ):
        raise SystemEvaluationSummaryError(f"{suite} fusion inputs differ from components")
    fusion_module_sha256 = _sha256(
        fusion.get("module_sha256"), field=f"{suite}.fusion.module_sha256"
    )
    fusion_cli_sha256 = _sha256(fusion.get("cli_sha256"), field=f"{suite}.fusion.cli_sha256")
    fusion_implementation_sha256 = _sha256(
        fusion.get("implementation_sha256"), field=f"{suite}.fusion.implementation_sha256"
    )
    if fusion_implementation_sha256 != canonical_json_hash(
        {"module_sha256": fusion_module_sha256, "cli_sha256": fusion_cli_sha256}
    ):
        raise SystemEvaluationSummaryError(f"{suite} fusion implementation identity is invalid")
    if target.get("bundle_sha256") != fit.get("bundle_sha256") or target.get(
        "calibration_version"
    ) != fit.get("calibration_version"):
        raise SystemEvaluationSummaryError(f"{suite} calibration fit/application identities differ")
    if target.get("input_predictions_sha256") != fusion.get("output_predictions_sha256"):
        raise SystemEvaluationSummaryError(f"{suite} target application input differs from fusion")
    if refinement.get("input_predictions_sha256") != target.get("output_predictions_sha256"):
        raise SystemEvaluationSummaryError(f"{suite} refinement input differs from calibration")
    if refinement.get("output_predictions_sha256") != predictions_sha256:
        raise SystemEvaluationSummaryError(f"{suite} refinement output differs from final output")
    counts = {
        item.get("prediction_document_count") for item in (model, rules, fusion, target, refinement)
    }
    if counts != {document_count}:
        raise SystemEvaluationSummaryError(f"{suite} system component counts differ")

    rules_identity = _mapping(rules.get("rules_identity"), field=f"{suite}.system.rules_identity")
    ruleset_id = _safe_id(rules_identity.get("ruleset_id"), field=f"{suite}.rules.ruleset_id")
    if ruleset_id != _REQUIRED_RULESET_ID:
        raise SystemEvaluationSummaryError(f"{suite} does not use the frozen ruleset")
    refinement_id = _safe_id(
        refinement.get("refinement_id"), field=f"{suite}.refinement.refinement_id"
    )
    if refinement_id != _REQUIRED_REFINEMENT_ID:
        raise SystemEvaluationSummaryError(f"{suite} does not use the frozen refinement")
    fit_gold_sha256 = _sha256(fit.get("gold_sha256"), field=f"{suite}.calibration.gold_sha256")
    fit_document_count = _integer(
        fit.get("document_count"), field=f"{suite}.calibration.document_count"
    )
    if (
        fit_gold_sha256 != _CALIBRATION_FIT_GOLD_SHA256
        or fit_document_count != _CALIBRATION_FIT_DOCUMENT_COUNT
    ):
        raise SystemEvaluationSummaryError(
            f"{suite} calibration was not fit on frozen synthetic validation"
        )
    common = {
        "selected_seed": 42,
        "attention_mode": "full",
        "training": {
            "manifest_sha256": training_sha256,
            "manifest_file_sha256": training_file_sha256,
        },
        "model_identity": _model_identity(system.get("model_identity"), suite=suite),
        "rules_identity": {
            "ruleset_id": ruleset_id,
            "implementation_sha256": _sha256(
                rules_identity.get("implementation_sha256"),
                field=f"{suite}.rules.implementation_sha256",
            ),
            "configuration_sha256": _sha256(
                rules_identity.get("configuration_sha256"),
                field=f"{suite}.rules.configuration_sha256",
            ),
        },
        "fusion_identity": {
            "implementation_sha256": fusion_implementation_sha256,
            "module_sha256": fusion_module_sha256,
            "cli_sha256": fusion_cli_sha256,
            "configuration_sha256": _sha256(
                fusion.get("configuration_sha256"),
                field=f"{suite}.fusion.configuration_sha256",
            ),
        },
        "calibration_fit": {
            "bundle_sha256": _sha256(
                fit.get("bundle_sha256"), field=f"{suite}.calibration.bundle_sha256"
            ),
            "diagnostics_manifest_sha256": _sha256(
                fit.get("diagnostics_manifest_sha256"),
                field=f"{suite}.calibration.diagnostics_manifest_sha256",
            ),
            "calibration_version": _safe_id(
                fit.get("calibration_version"),
                field=f"{suite}.calibration.calibration_version",
            ),
            "gold_sha256": fit_gold_sha256,
            "fused_predictions_sha256": _sha256(
                fit.get("fused_predictions_sha256"),
                field=f"{suite}.calibration.fused_predictions_sha256",
            ),
            "document_count": fit_document_count,
        },
        "refinement_identity": {
            "refinement_id": refinement_id,
            "implementation_sha256": _sha256(
                refinement.get("implementation_sha256"),
                field=f"{suite}.refinement.implementation_sha256",
            ),
        },
    }
    suite_identity = {
        "evaluation_sha256": evaluation_sha256,
        "predictions_sha256": predictions_sha256,
        "system_prediction_manifest_sha256": system_manifest_sha256,
        "system_prediction_manifest_file_sha256": system_manifest_file_sha256,
        "target_application": {
            "input_predictions_sha256": _sha256(
                target.get("input_predictions_sha256"),
                field=f"{suite}.target.input_predictions_sha256",
            ),
            "output_predictions_sha256": _sha256(
                target.get("output_predictions_sha256"),
                field=f"{suite}.target.output_predictions_sha256",
            ),
        },
        "refinement_audit_manifest_sha256": _sha256(
            refinement.get("audit_manifest_sha256"),
            field=f"{suite}.refinement.audit_manifest_sha256",
        ),
    }
    return common, suite_identity


def _summarize_report(path: Path, *, suite: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report, file_sha256 = _read_report(path, suite=suite)
    metrics = _suite_metrics(report, suite=suite)
    expected = _SUITE_IDENTITIES[suite]
    if metrics["document_count"] != expected["record_count"]:
        raise SystemEvaluationSummaryError(f"{suite} report document count differs")
    if metrics["span_count"]["gold"] != expected["span_count"]:
        raise SystemEvaluationSummaryError(f"{suite} report gold span count differs")
    provenance = _mapping(report.get("provenance"), field=f"{suite}.provenance")
    dataset = _dataset_identity(provenance, suite=suite)
    common, system = _common_system_identity(
        provenance,
        suite=suite,
        document_count=metrics["document_count"],
        dataset_manifest_sha256=dataset["manifest_sha256"],
    )
    return {
        "evaluation_report_file_sha256": file_sha256,
        "dataset": dataset,
        "system": system,
        "metrics": metrics,
    }, common


def _pooled_score(suites: Mapping[str, Mapping[str, Any]], *, metric: str) -> dict[str, Any]:
    totals = {
        name: sum(int(suite["metrics"][metric][name]) for suite in suites.values())
        for name in ("tp", "fp", "fn")
    }
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {**totals, "precision": precision, "recall": recall, "f1": f1, "f2": f2}


def _write_json_atomic(value: Mapping[str, Any], destination: Path) -> None:
    if destination.is_symlink():
        raise SystemEvaluationSummaryError("output must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            delete=False,
            prefix=f".{destination.name}.",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_stale_output(destination: Path) -> None:
    if destination.is_symlink():
        raise SystemEvaluationSummaryError("output must not be a symlink")
    if destination.exists():
        if not destination.is_file():
            raise SystemEvaluationSummaryError("output must be a regular file")
        destination.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-test", type=Path, required=True)
    parser.add_argument("--pii-bench-formal", type=Path, required=True)
    parser.add_argument("--pii-bench-chat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    raw_inputs = {
        "synthetic_test": args.synthetic_test,
        "pii_bench_formal": args.pii_bench_formal,
        "pii_bench_chat": args.pii_bench_chat,
    }
    output = args.output.expanduser().absolute()
    cleanup_output = False
    try:
        if output.is_symlink():
            raise SystemEvaluationSummaryError("output must not be a symlink")
        inputs = {
            suite: path.expanduser().resolve(strict=True) for suite, path in raw_inputs.items()
        }
        if len(set(inputs.values())) != len(inputs):
            raise SystemEvaluationSummaryError("evaluation report inputs must be distinct")
        if output in set(inputs.values()):
            raise SystemEvaluationSummaryError("output must differ from every input report")
        if output.exists() and any(os.path.samefile(output, path) for path in inputs.values()):
            raise SystemEvaluationSummaryError("output must not alias an input report")
        cleanup_output = True
        _remove_stale_output(output)
        suites: dict[str, Any] = {}
        common_identity: dict[str, Any] | None = None
        for suite in _SUITE_ORDER:
            summary, common = _summarize_report(inputs[suite], suite=suite)
            if common_identity is None:
                common_identity = common
            elif common != common_identity:
                raise SystemEvaluationSummaryError(
                    f"{suite} uses a different model/system/calibration identity"
                )
            suites[suite] = summary
        if common_identity is None:  # pragma: no cover - fixed suite list is non-empty
            raise SystemEvaluationSummaryError("no system evaluation reports were supplied")
        aggregate = {
            "document_count": sum(item["metrics"]["document_count"] for item in suites.values()),
            "gold_span_count": sum(
                item["metrics"]["span_count"]["gold"] for item in suites.values()
            ),
            "predicted_span_count": sum(
                item["metrics"]["span_count"]["predicted"] for item in suites.values()
            ),
            "strict_micro_pooled": _pooled_score(suites, metric="strict_micro"),
            "relaxed_micro_pooled": _pooled_score(suites, metric="relaxed_micro"),
            "character_micro_pooled": _pooled_score(suites, metric="character_micro"),
            "bootstrap_intervals": "reported_per_suite_not_naively_pooled",
        }
        summary = add_manifest_hash(
            {
                "schema_version": 1,
                "artifact_type": "system_frozen_evaluation_summary",
                "suite_order": list(_SUITE_ORDER),
                "system_identity": common_identity,
                "suites": suites,
                "aggregate": aggregate,
                "privacy": {
                    "contains_paths": False,
                    "contains_document_ids": False,
                    "contains_raw_text": False,
                    "contains_entity_values": False,
                    "contains_record_level_data": False,
                },
            }
        )
        _assert_public_safe(summary)
        _write_json_atomic(summary, output)
    except (OSError, SystemEvaluationSummaryError) as exc:
        if cleanup_output:
            try:
                _remove_stale_output(output)
            except (OSError, SystemEvaluationSummaryError):
                pass
        parser.error(str(exc))
    print(
        json.dumps(
            {"manifest_sha256": summary["manifest_sha256"], "suite_count": len(suites)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
