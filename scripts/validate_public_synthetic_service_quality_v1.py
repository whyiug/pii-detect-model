#!/usr/bin/env python3
"""Validate the public/synthetic community service-quality v1 contract.

This gate is deliberately separate from every production/human-evidence gate.
It can support only a named, public-test-exposed, non-production benchmark
claim.  A checked-in preregistration without a separately frozen result receipt
is expected to report ``BLOCKED``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from scripts.community_contract_utils import (
        CommunityContractError,
        canonical_json_hash,
        close_enough,
        load_json_path,
        read_regular_file,
        require_finite_unit_interval,
        require_nonnegative_number,
        sha256_bytes,
        strict_json_bytes,
        validate_schema,
        verify_binding,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    from community_contract_utils import (  # type: ignore[no-redef]
        CommunityContractError,
        canonical_json_hash,
        close_enough,
        load_json_path,
        read_regular_file,
        require_finite_unit_interval,
        require_nonnegative_number,
        sha256_bytes,
        strict_json_bytes,
        validate_schema,
        verify_binding,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = "configs/evaluation/public_synthetic_service_quality_v1.json"
SCHEMA_PATH = "configs/evaluation/public_synthetic_service_quality_v1.schema.json"
VALIDATOR_PATH = "scripts/validate_public_synthetic_service_quality_v1.py"
CONTRACT_ID = "public_synthetic_service_quality_v1"
CONTRACT_SCHEMA_VERSION = "pii-zh.public-synthetic-service-quality-contract.v1"
RESULT_SCHEMA_VERSION = "pii-zh.public-synthetic-service-quality-result.v1"
REPORT_SCHEMA_VERSION = "pii-zh.public-synthetic-service-quality-report.v1"

TRACKS = ("model_raw", "full_system")
CORE_LABELS = (
    "PERSON_NAME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "VEHICLE_LICENSE_PLATE",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER",
    "WECHAT_ID",
    "QQ_NUMBER",
    "ALIPAY_ACCOUNT",
    "USERNAME",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "DEVICE_ID",
    "GEO_COORDINATE",
    "SECRET",
)

REQUIRED_RESULT_BINDINGS = (
    "final_model_manifest",
    "model_raw_dataset_manifest",
    "full_system_dataset_manifest",
    "pii_free_dataset_manifest",
    "model_raw_prediction_manifest",
    "full_system_prediction_manifest",
    "source_snapshot_manifest",
    "service_configuration",
)

EXPECTED_SCOPE = {
    "allowed_data": ["public_open_source", "synthetic"],
    "human_evidence_required": False,
    "production_evidence": False,
    "public_test_exposed": True,
    "real_world_generalization_established": False,
    "release_class": "community_research_preview",
}
EXPECTED_MODEL_POLICY = {
    "artifact_class": "24_label_zh_hans_token_classifier",
    "allowed_attention_modes": ["causal", "full"],
    "causal": {
        "candidate_allowed": True,
        "loader_pii_release_eligible": False,
        "release_class": "research_preview_explicit_opt_in_only",
        "runtime_policy": "explicit_opt_in_causal_research_runtime",
    },
    "full": {
        "candidate_allowed": True,
        "loader_pii_release_eligible_may_be_true": True,
        "release_class": "community_research_preview_candidate",
        "runtime_policy": "standard_full_attention_runtime",
    },
    "evidence_requirements": {
        "full_system_release_blocking": True,
        "full_system_service_quality_required": True,
        "model_raw_and_full_system_bind_same_final_model": True,
        "model_raw_evaluation_required": True,
    },
    "production_ready": False,
}
EXPECTED_CLAIMS = {
    "first_chinese_pii_model": False,
    "global_sota": False,
    "production_ready": False,
    "real_world_sota": False,
    "permitted_claim": (
        "best_on_named_public_synthetic_benchmark_among_frozen_comparators_if_"
        "the_result_receipt_passes_the_paired_bootstrap_gate"
    ),
}
EXPECTED_PRIVACY = {
    "aggregate_receipt_only": True,
    "gpu_query_allowed": False,
    "model_weight_read_allowed": False,
    "network_allowed": False,
    "record_level_data_read_allowed": False,
    "result_bindings_are_manifests_not_raw_records": True,
}


class PublicSyntheticQualityError(CommunityContractError):
    """A public/synthetic quality evidence validation error."""


def _exact_keys(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PublicSyntheticQualityError(f"{field} has an invalid closed shape")
    return value


def _positive_integer(value: object, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicSyntheticQualityError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise PublicSyntheticQualityError(f"{field} is below its minimum")
    return value


def _load_contract_and_schema(
    *,
    contract_path: Path,
    schema_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    contract_payload = read_regular_file(contract_path, field="quality contract")
    contract = load_json_path(contract_path, field="quality contract")
    schema = load_json_path(schema_path, field="quality schema")
    validate_schema(contract, schema, field="quality contract")
    return contract, schema, contract_payload


def validate_preregistration(
    contract_path: Path | None = None,
    schema_path: Path | None = None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    contract_file = contract_path or repository_root / CONTRACT_PATH
    schema_file = schema_path or repository_root / SCHEMA_PATH
    contract, _schema, _payload = _load_contract_and_schema(
        contract_path=contract_file,
        schema_path=schema_file,
    )
    if contract["contract_sha256"] != canonical_json_hash(contract, remove="contract_sha256"):
        raise PublicSyntheticQualityError("quality contract canonical self hash does not verify")
    if contract["scope"] != EXPECTED_SCOPE:
        raise PublicSyntheticQualityError(
            "quality contract scope weakens the non-production boundary"
        )
    if contract["model_policy"] != EXPECTED_MODEL_POLICY:
        raise PublicSyntheticQualityError("quality contract model policy is invalid")
    if contract["claims"] != EXPECTED_CLAIMS:
        raise PublicSyntheticQualityError("quality contract claim boundary is invalid")
    if contract["privacy"] != EXPECTED_PRIVACY:
        raise PublicSyntheticQualityError("quality contract metadata-only boundary is invalid")
    if tuple(contract["taxonomy"]["labels"]) != CORE_LABELS:
        raise PublicSyntheticQualityError(
            "quality contract taxonomy is not the ordered 24-label core"
        )
    if tuple(contract["tracks"]) != TRACKS:
        raise PublicSyntheticQualityError(
            "quality contract must contain model_raw then full_system"
        )
    if tuple(contract["required_result_bindings"]) != REQUIRED_RESULT_BINDINGS:
        raise PublicSyntheticQualityError("quality contract result binding inventory drifted")

    frozen = contract["frozen_inputs"]
    verify_binding(repository_root, frozen["comparator_registry"], field="comparator registry")
    metric_paths: set[str] = set()
    for index, binding in enumerate(frozen["metric_code"]):
        verify_binding(repository_root, binding, field=f"metric code {index}")
        path = binding["path"]
        if path in metric_paths:
            raise PublicSyntheticQualityError("metric code bindings contain duplicate paths")
        metric_paths.add(path)

    implementation = contract["implementation"]
    verify_binding(repository_root, implementation["schema"], field="quality schema binding")
    verify_binding(repository_root, implementation["validator"], field="quality validator binding")
    verify_binding(repository_root, implementation["support"], field="quality support binding")

    policies = contract["track_policies"]
    if set(policies) != set(TRACKS):
        raise PublicSyntheticQualityError("quality contract track policies are incomplete")
    if policies["model_raw"]["canonical_track"] != "model_raw":
        raise PublicSyntheticQualityError("model_raw policy uses the wrong canonical track")
    if policies["full_system"]["canonical_track"] != "full_system":
        raise PublicSyntheticQualityError("full_system policy uses the wrong canonical track")
    if policies["model_raw"]["release_blocking"] is not False:
        raise PublicSyntheticQualityError("standalone model must remain non-blocking")
    if policies["full_system"]["release_blocking"] is not True:
        raise PublicSyntheticQualityError("integrated service must remain release-blocking")
    for track in TRACKS:
        thresholds = policies[track]["thresholds"]
        for key, value in thresholds.items():
            if key.endswith(("_min", "_max")):
                require_nonnegative_number(value, field=f"{track}.{key}")
        bootstrap = policies[track]["bootstrap"]
        if bootstrap["method"] != "paired_document_bootstrap_percentile":
            raise PublicSyntheticQualityError(f"{track} bootstrap method is invalid")
        if bootstrap["comparison_unit"] != "document":
            raise PublicSyntheticQualityError(f"{track} bootstrap unit is invalid")
        if bootstrap["confidence_level"] != 0.95:
            raise PublicSyntheticQualityError(f"{track} bootstrap confidence level is invalid")
        if bootstrap["resamples_min"] < 10000:
            raise PublicSyntheticQualityError(f"{track} bootstrap is underpowered")
    return contract


def _verify_strict_span(
    value: object,
    *,
    track: str,
    minimum_per_label_recall: float,
) -> tuple[dict[str, float], list[str]]:
    metrics = _exact_keys(
        value,
        {"tp", "fp", "fn", "micro_precision", "micro_recall", "micro_f1", "macro_f1", "per_label"},
        field=f"{track}.strict_span",
    )
    per_label = _exact_keys(metrics["per_label"], set(CORE_LABELS), field=f"{track}.per_label")
    total_tp = total_fp = total_fn = 0
    label_f1: list[float] = []
    low_recall: list[str] = []
    for label in CORE_LABELS:
        item = _exact_keys(
            per_label[label],
            {"tp", "fp", "fn", "precision", "recall", "f1"},
            field=f"{track}.per_label.{label}",
        )
        tp = _positive_integer(item["tp"], field=f"{track}.{label}.tp", allow_zero=True)
        fp = _positive_integer(item["fp"], field=f"{track}.{label}.fp", allow_zero=True)
        fn = _positive_integer(item["fn"], field=f"{track}.{label}.fn", allow_zero=True)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        for name, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
            actual = require_finite_unit_interval(
                item[name], field=f"{track}.{label}.{name}"
            )
            if not close_enough(actual, expected):
                raise PublicSyntheticQualityError(f"{track}.{label}.{name} is inconsistent")
        label_f1.append(f1)
        if tp + fn > 0 and recall < minimum_per_label_recall:
            low_recall.append(label)

    if (metrics["tp"], metrics["fp"], metrics["fn"]) != (total_tp, total_fp, total_fn):
        raise PublicSyntheticQualityError(f"{track} strict-span totals are inconsistent")
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro_f1 = sum(label_f1) / len(CORE_LABELS)
    derived = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }
    for name, expected in derived.items():
        actual = require_finite_unit_interval(metrics[name], field=f"{track}.{name}")
        if not close_enough(actual, expected):
            raise PublicSyntheticQualityError(f"{track}.{name} is inconsistent")
    return derived, low_recall


def _verify_pii_free(value: object, *, track: str) -> dict[str, float]:
    metrics = _exact_keys(
        value,
        {
            "document_count",
            "false_positive_document_count",
            "document_fpr",
            "text_character_count",
            "false_positive_span_count",
            "false_positive_spans_per_1000_characters",
        },
        field=f"{track}.pii_free",
    )
    documents = _positive_integer(metrics["document_count"], field=f"{track}.document_count")
    fp_documents = _positive_integer(
        metrics["false_positive_document_count"],
        field=f"{track}.false_positive_document_count",
        allow_zero=True,
    )
    characters = _positive_integer(
        metrics["text_character_count"], field=f"{track}.text_character_count"
    )
    fp_spans = _positive_integer(
        metrics["false_positive_span_count"],
        field=f"{track}.false_positive_span_count",
        allow_zero=True,
    )
    if fp_documents > documents:
        raise PublicSyntheticQualityError(f"{track} false-positive documents exceed documents")
    expected_fpr = fp_documents / documents
    expected_per_1000 = fp_spans * 1000.0 / characters
    actual_fpr = require_finite_unit_interval(
        metrics["document_fpr"], field=f"{track}.document_fpr"
    )
    actual_per_1000 = require_nonnegative_number(
        metrics["false_positive_spans_per_1000_characters"],
        field=f"{track}.false_positive_spans_per_1000_characters",
    )
    if not close_enough(actual_fpr, expected_fpr) or not close_enough(
        actual_per_1000, expected_per_1000
    ):
        raise PublicSyntheticQualityError(f"{track} PII-free metrics are inconsistent")
    return {"document_fpr": expected_fpr, "fp_spans_per_1000": expected_per_1000}


def _verify_character_error(value: object, *, track: str) -> dict[str, float]:
    metrics = _exact_keys(
        value,
        {
            "gold_pii_character_count",
            "missed_pii_character_count",
            "pii_character_miss_rate",
            "non_pii_character_count",
            "falsely_flagged_non_pii_character_count",
            "non_pii_character_false_positive_rate",
        },
        field=f"{track}.character_error",
    )
    gold = _positive_integer(
        metrics["gold_pii_character_count"], field=f"{track}.gold_pii_character_count"
    )
    missed = _positive_integer(
        metrics["missed_pii_character_count"],
        field=f"{track}.missed_pii_character_count",
        allow_zero=True,
    )
    non_pii = _positive_integer(
        metrics["non_pii_character_count"], field=f"{track}.non_pii_character_count"
    )
    falsely_flagged = _positive_integer(
        metrics["falsely_flagged_non_pii_character_count"],
        field=f"{track}.falsely_flagged_non_pii_character_count",
        allow_zero=True,
    )
    if missed > gold or falsely_flagged > non_pii:
        raise PublicSyntheticQualityError(f"{track} character errors exceed their denominators")
    miss_rate = missed / gold
    false_positive_rate = falsely_flagged / non_pii
    if not close_enough(
        require_finite_unit_interval(
            metrics["pii_character_miss_rate"], field=f"{track}.pii_character_miss_rate"
        ),
        miss_rate,
    ) or not close_enough(
        require_finite_unit_interval(
            metrics["non_pii_character_false_positive_rate"],
            field=f"{track}.non_pii_character_false_positive_rate",
        ),
        false_positive_rate,
    ):
        raise PublicSyntheticQualityError(f"{track} character metrics are inconsistent")
    return {"miss_rate": miss_rate, "false_positive_rate": false_positive_rate}


def _verify_performance(value: object, *, track: str) -> dict[str, float]:
    metrics = _exact_keys(
        value,
        {
            "sample_count",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "throughput_documents_per_second",
            "throughput_characters_per_second",
            "peak_cpu_rss_mb",
            "peak_gpu_memory_mb",
        },
        field=f"{track}.performance",
    )
    _positive_integer(metrics["sample_count"], field=f"{track}.performance.sample_count")
    numeric = {
        name: require_nonnegative_number(metrics[name], field=f"{track}.performance.{name}")
        for name in set(metrics) - {"sample_count"}
    }
    if not (
        numeric["latency_p50_ms"]
        <= numeric["latency_p95_ms"]
        <= numeric["latency_p99_ms"]
    ):
        raise PublicSyntheticQualityError(f"{track} latency percentiles are inconsistent")
    if numeric["throughput_documents_per_second"] <= 0.0:
        raise PublicSyntheticQualityError(f"{track} throughput must be positive")
    return numeric


def _verify_bootstrap(value: object, *, track: str) -> dict[str, float | int | str]:
    result = _exact_keys(
        value,
        {
            "method",
            "comparison_unit",
            "seed",
            "resamples",
            "confidence_level",
            "comparator_id",
            "comparator_track",
            "micro_f1_delta_ci_low",
            "macro_f1_delta_ci_low",
            "document_fpr_delta_ci_high",
        },
        field=f"{track}.bootstrap",
    )
    if result["method"] != "paired_document_bootstrap_percentile":
        raise PublicSyntheticQualityError(f"{track} bootstrap method drifted")
    if result["comparison_unit"] != "document":
        raise PublicSyntheticQualityError(f"{track} bootstrap unit drifted")
    _positive_integer(result["resamples"], field=f"{track}.bootstrap.resamples")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
        raise PublicSyntheticQualityError(f"{track} bootstrap seed is invalid")
    if result["confidence_level"] != 0.95:
        raise PublicSyntheticQualityError(f"{track} bootstrap confidence level drifted")
    if result["comparator_track"] != track:
        raise PublicSyntheticQualityError(f"{track} comparator crosses canonical tracks")
    if not isinstance(result["comparator_id"], str) or not result["comparator_id"]:
        raise PublicSyntheticQualityError(f"{track} comparator id is invalid")
    output: dict[str, float | int | str] = dict(result)
    for name in (
        "micro_f1_delta_ci_low",
        "macro_f1_delta_ci_low",
        "document_fpr_delta_ci_high",
    ):
        raw = result[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PublicSyntheticQualityError(f"{track}.{name} must be finite")
        number = float(raw)
        if number != number or number in (float("inf"), float("-inf")):
            raise PublicSyntheticQualityError(f"{track}.{name} must be finite")
        output[name] = number
    return output


def _verify_track(
    value: object,
    *,
    track: str,
    policy: Mapping[str, Any],
) -> tuple[str, list[str]]:
    result = _exact_keys(
        value,
        {
            "declared_status",
            "canonical_track",
            "strict_span",
            "pii_free",
            "character_error",
            "performance",
            "bootstrap",
        },
        field=track,
    )
    if result["canonical_track"] != track:
        raise PublicSyntheticQualityError(f"{track} result is assigned to the wrong track")
    thresholds = policy["thresholds"]
    strict, low_recall = _verify_strict_span(
        result["strict_span"],
        track=track,
        minimum_per_label_recall=thresholds["minimum_per_label_recall_min"],
    )
    pii_free = _verify_pii_free(result["pii_free"], track=track)
    character = _verify_character_error(result["character_error"], track=track)
    performance = _verify_performance(result["performance"], track=track)
    bootstrap = _verify_bootstrap(result["bootstrap"], track=track)
    bootstrap_policy = policy["bootstrap"]

    failures: list[str] = []
    comparisons = (
        (strict["micro_f1"] >= thresholds["strict_micro_f1_min"], "strict_micro_f1"),
        (strict["macro_f1"] >= thresholds["strict_macro_f1_min"], "strict_macro_f1"),
        (not low_recall, "minimum_per_label_recall"),
        (
            pii_free["document_fpr"] <= thresholds["pii_free_document_fpr_max"],
            "pii_free_document_fpr",
        ),
        (
            pii_free["fp_spans_per_1000"]
            <= thresholds["fp_spans_per_1000_characters_max"],
            "fp_spans_per_1000_characters",
        ),
        (
            character["miss_rate"] <= thresholds["pii_character_miss_rate_max"],
            "pii_character_miss_rate",
        ),
        (
            character["false_positive_rate"]
            <= thresholds["non_pii_character_false_positive_rate_max"],
            "non_pii_character_false_positive_rate",
        ),
        (performance["latency_p95_ms"] <= thresholds["latency_p95_ms_max"], "latency_p95_ms"),
        (
            performance["throughput_documents_per_second"]
            >= thresholds["throughput_documents_per_second_min"],
            "throughput_documents_per_second",
        ),
        (performance["peak_cpu_rss_mb"] <= thresholds["peak_cpu_rss_mb_max"], "peak_cpu_rss_mb"),
        (
            performance["peak_gpu_memory_mb"] <= thresholds["peak_gpu_memory_mb_max"],
            "peak_gpu_memory_mb",
        ),
        (bootstrap["resamples"] >= bootstrap_policy["resamples_min"], "bootstrap_resamples"),
        (
            bootstrap["micro_f1_delta_ci_low"]
            >= bootstrap_policy["micro_f1_delta_ci_low_min"],
            "bootstrap_micro_f1",
        ),
        (
            bootstrap["macro_f1_delta_ci_low"]
            >= bootstrap_policy["macro_f1_delta_ci_low_min"],
            "bootstrap_macro_f1",
        ),
        (
            bootstrap["document_fpr_delta_ci_high"]
            <= bootstrap_policy["document_fpr_delta_ci_high_max"],
            "bootstrap_document_fpr",
        ),
    )
    failures.extend(name for passed, name in comparisons if not passed)
    if low_recall:
        failures.extend(f"per_label_recall:{label}" for label in low_recall)
    expected_status = "PASS" if not failures else "FAIL"
    if result["declared_status"] != expected_status:
        raise PublicSyntheticQualityError(f"{track} declared status does not match metrics")
    return expected_status, sorted(set(failures))


def _verify_result_bindings(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    artifact_bindings = _exact_keys(
        receipt["artifact_bindings"],
        set(REQUIRED_RESULT_BINDINGS) | {"comparator_registry", "metric_code"},
        field="result artifact_bindings",
    )
    seen_paths: set[str] = set()
    payloads: dict[str, bytes] = {}
    for name in REQUIRED_RESULT_BINDINGS:
        binding = artifact_bindings[name]
        payloads[name] = verify_binding(
            repository_root, binding, field=f"result binding {name}"
        )
        path = binding["path"]
        if path in seen_paths:
            raise PublicSyntheticQualityError("result artifact bindings reuse a manifest path")
        seen_paths.add(path)
    if artifact_bindings["comparator_registry"] != contract["frozen_inputs"]["comparator_registry"]:
        raise PublicSyntheticQualityError(
            "result comparator registry is not the preregistered registry"
        )
    verify_binding(
        repository_root,
        artifact_bindings["comparator_registry"],
        field="result comparator registry",
    )
    if artifact_bindings["metric_code"] != contract["frozen_inputs"]["metric_code"]:
        raise PublicSyntheticQualityError("result metric code is not the preregistered code")
    for index, binding in enumerate(artifact_bindings["metric_code"]):
        verify_binding(repository_root, binding, field=f"result metric code {index}")

    model_manifest = strict_json_bytes(
        payloads["final_model_manifest"], field="final model manifest"
    )
    _exact_keys(
        model_manifest,
        {
            "schema_version",
            "model_id",
            "architecture",
            "attention_mode",
            "label_count",
            "taxonomy_version",
            "artifact_sha256",
            "config_sha256",
            "tokenizer_manifest_sha256",
            "training_data_scope",
            "public_test_exposed",
            "community_research_release_eligible",
            "production_release_eligible",
            "loader_pii_release_eligible",
            "runtime_policy",
        },
        field="final model manifest",
    )
    if model_manifest["schema_version"] != "pii-zh.community-model-manifest.v1":
        raise PublicSyntheticQualityError("final model manifest schema version is invalid")
    if not isinstance(model_manifest["model_id"], str) or not model_manifest["model_id"]:
        raise PublicSyntheticQualityError("final model manifest model_id is invalid")
    if not isinstance(model_manifest["architecture"], str) or not model_manifest["architecture"]:
        raise PublicSyntheticQualityError("final model manifest architecture is invalid")
    model_policy = contract["model_policy"]
    if model_manifest["label_count"] != 24 or model_manifest["taxonomy_version"] != "1.0.0":
        raise PublicSyntheticQualityError("final model manifest taxonomy is invalid")
    for digest_name in ("artifact_sha256", "config_sha256", "tokenizer_manifest_sha256"):
        digest = model_manifest[digest_name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PublicSyntheticQualityError(f"final model manifest {digest_name} is invalid")
    if model_manifest["training_data_scope"] != "public_and_synthetic_only":
        raise PublicSyntheticQualityError("final model manifest training scope is invalid")
    if model_manifest["public_test_exposed"] is not True:
        raise PublicSyntheticQualityError("final model manifest hides public test exposure")
    if model_manifest["community_research_release_eligible"] is not True:
        raise PublicSyntheticQualityError(
            "final model is not eligible for a community research preview"
        )
    if model_manifest["production_release_eligible"] is not False:
        raise PublicSyntheticQualityError("final model improperly enables production release")
    attention_mode = model_manifest["attention_mode"]
    if attention_mode not in model_policy["allowed_attention_modes"]:
        raise PublicSyntheticQualityError("final model attention mode is unsupported")
    if attention_mode == "causal":
        causal_policy = model_policy["causal"]
        if (
            causal_policy["candidate_allowed"] is not True
            or model_manifest["loader_pii_release_eligible"]
            is not causal_policy["loader_pii_release_eligible"]
            or model_manifest["runtime_policy"] != causal_policy["runtime_policy"]
        ):
            raise PublicSyntheticQualityError(
                "causal model must retain the non-production explicit-opt-in loader boundary"
            )
    elif attention_mode == "full":
        full_policy = model_policy["full"]
        if full_policy["candidate_allowed"] is not True:
            raise PublicSyntheticQualityError("full-attention candidate is not admitted")
        if model_manifest["runtime_policy"] != full_policy["runtime_policy"]:
            raise PublicSyntheticQualityError("full-attention model runtime policy is invalid")
        if not isinstance(model_manifest["loader_pii_release_eligible"], bool):
            raise PublicSyntheticQualityError("full-attention loader eligibility is invalid")
    return attention_mode


def validate_result_receipt(
    result_path: Path,
    *,
    contract_path: Path | None = None,
    schema_path: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    contract_file = contract_path or repository_root / CONTRACT_PATH
    contract = validate_preregistration(
        contract_file,
        schema_path,
        repository_root=repository_root,
    )
    receipt = load_json_path(result_path, field="quality result receipt")
    required_receipt_keys = {
        "schema_version",
        "receipt_id",
        "reported_status",
        "contract",
        "exposure",
        "artifact_bindings",
        "tracks",
        "claims",
        "receipt_sha256",
    }
    _exact_keys(receipt, required_receipt_keys, field="quality result receipt")
    if receipt["schema_version"] != RESULT_SCHEMA_VERSION:
        raise PublicSyntheticQualityError("quality result schema version is invalid")
    if not isinstance(receipt["receipt_id"], str) or not receipt["receipt_id"]:
        raise PublicSyntheticQualityError("quality result receipt id is invalid")
    if receipt["receipt_sha256"] != canonical_json_hash(receipt, remove="receipt_sha256"):
        raise PublicSyntheticQualityError("quality result canonical self hash does not verify")

    contract_payload = read_regular_file(contract_file, field="quality contract")
    expected_contract_binding = {
        "contract_id": CONTRACT_ID,
        "path": CONTRACT_PATH,
        "file_sha256": sha256_bytes(contract_payload),
        "canonical_sha256": contract["contract_sha256"],
    }
    if receipt["contract"] != expected_contract_binding:
        raise PublicSyntheticQualityError("quality result does not bind the exact preregistration")
    if receipt["exposure"] != {
        "data_scope": "public_and_synthetic_only",
        "evaluation_is_blind": False,
        "public_test_exposed": True,
        "real_world_evidence": False,
    }:
        raise PublicSyntheticQualityError("quality result exposure declaration is invalid")
    claims = _exact_keys(
        receipt["claims"],
        {
            "model_raw_benchmark_leading",
            "full_system_benchmark_leading",
            "first_chinese_pii_model",
            "global_sota",
            "production_ready",
            "real_world_sota",
        },
        field="quality result claims",
    )
    for prohibited in (
        "first_chinese_pii_model",
        "global_sota",
        "production_ready",
        "real_world_sota",
    ):
        if claims[prohibited] is not False:
            raise PublicSyntheticQualityError(
                f"quality result enables prohibited claim {prohibited}"
            )
    attention_mode = _verify_result_bindings(
        receipt, contract, repository_root=repository_root
    )

    tracks = _exact_keys(receipt["tracks"], set(TRACKS), field="quality result tracks")
    statuses: dict[str, str] = {}
    failures: dict[str, list[str]] = {}
    for track in TRACKS:
        status, track_failures = _verify_track(
            tracks[track], track=track, policy=contract["track_policies"][track]
        )
        statuses[track] = status
        failures[track] = track_failures
        if claims[f"{track}_benchmark_leading"] is True:
            bootstrap = tracks[track]["bootstrap"]
            if not (
                bootstrap["micro_f1_delta_ci_low"] > 0.0
                and bootstrap["macro_f1_delta_ci_low"] > 0.0
                and bootstrap["document_fpr_delta_ci_high"] <= 0.0
                and status == "PASS"
            ):
                raise PublicSyntheticQualityError(
                    f"{track} benchmark-leading claim lacks paired-bootstrap support"
                )
        elif claims[f"{track}_benchmark_leading"] is not False:
            raise PublicSyntheticQualityError(f"{track} benchmark-leading claim is not boolean")

    blocking_tracks = [
        track for track in TRACKS if contract["track_policies"][track]["release_blocking"]
    ]
    overall_status = (
        "PASS" if all(statuses[track] == "PASS" for track in blocking_tracks) else "FAIL"
    )
    if receipt["reported_status"] != overall_status:
        raise PublicSyntheticQualityError(
            "quality result reported status does not match gate outcome"
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "public_synthetic_service_quality",
        "status": overall_status,
        "scope": EXPECTED_SCOPE,
        "candidate_model": {
            "attention_mode": attention_mode,
            "label_count": 24,
            "release_class": contract["model_policy"][attention_mode]["release_class"],
            "production_ready": False,
        },
        "tracks": {
            track: {
                "status": statuses[track],
                "release_blocking": contract["track_policies"][track]["release_blocking"],
                "failure_ids": failures[track],
            }
            for track in TRACKS
        },
        "claims": dict(claims),
        "limitations": [
            "synthetic_and_public_data_only",
            "public_test_exposed",
            "non_production",
            "no_real_world_sota_claim",
        ],
    }


def build_report(
    *,
    contract_path: Path | None = None,
    schema_path: Path | None = None,
    result_path: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    contract = validate_preregistration(
        contract_path,
        schema_path,
        repository_root=repository_root,
    )
    if result_path is not None:
        return validate_result_receipt(
            result_path,
            contract_path=contract_path,
            schema_path=schema_path,
            repository_root=repository_root,
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "public_synthetic_service_quality",
        "status": "BLOCKED",
        "reason_codes": ["frozen_result_receipt_missing"],
        "contract_id": contract["contract_id"],
        "scope": contract["scope"],
        "claims": contract["claims"],
        "limitations": [
            "synthetic_and_public_data_only",
            "public_test_exposed",
            "non_production",
            "no_real_world_sota_claim",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=REPOSITORY_ROOT / CONTRACT_PATH)
    parser.add_argument("--schema", type=Path, default=REPOSITORY_ROOT / SCHEMA_PATH)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = build_report(
            contract_path=args.contract,
            schema_path=args.schema,
            result_path=args.result,
        )
    except CommunityContractError as exc:
        raise SystemExit(f"quality validation failed: {exc}") from exc
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report["status"] in {"PASS", "BLOCKED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
