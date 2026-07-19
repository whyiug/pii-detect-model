"""Aggregate-only paired diagnostics for the fixed production PII core.

The builder accepts already-loaded span records, delegates strict metrics and
the frozen statistical procedure to the existing three-metric implementation,
and retains no document identifiers or span keys in its result.  A later,
release-bound contract must activate any formal suite receipt or claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .io import EvaluationDataError, GoldRecord, PredictionRecord
from .paired_bootstrap_three_metric import (
    ThreeMetricPairedSuite,
    build_three_metric_paired_suite,
    frozen_three_metric_intersection_union_bootstrap,
)
from .provenance import canonical_json_hash

PII_CORE_LABELS = (
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


@dataclass(frozen=True, slots=True)
class ServiceQualityComparatorInput:
    """One named comparator's in-memory predictions."""

    comparator_id: str
    predictions: Sequence[PredictionRecord]


def _normalized_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationDataError(f"{field} must be a non-empty normalized string")
    return value


def _gold_records(values: Sequence[GoldRecord]) -> tuple[GoldRecord, ...]:
    if isinstance(values, (str, bytes)):
        raise EvaluationDataError("gold_records must be a sequence of GoldRecord values")
    records = tuple(values)
    if not records or any(not isinstance(record, GoldRecord) for record in records):
        raise EvaluationDataError("gold_records must contain GoldRecord values")
    return records


def _prediction_records(
    values: Sequence[PredictionRecord], *, field: str
) -> tuple[PredictionRecord, ...]:
    if isinstance(values, (str, bytes)):
        raise EvaluationDataError(f"{field} must be a sequence of PredictionRecord values")
    records = tuple(values)
    if not records or any(not isinstance(record, PredictionRecord) for record in records):
        raise EvaluationDataError(f"{field} must contain PredictionRecord values")
    return records


def _fixed_labels(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or tuple(values) != PII_CORE_LABELS:
        raise EvaluationDataError(
            "labels must be the exact ordered canonical 24-class PII core"
        )
    return PII_CORE_LABELS


def _strict_counts(suite: ThreeMetricPairedSuite, *, side: str) -> dict[str, int]:
    systems = [getattr(document, side) for document in suite.documents]
    return {
        "tp": sum(system.strict_micro.tp for system in systems),
        "fp": sum(system.strict_micro.fp for system in systems),
        "fn": sum(system.strict_micro.fn for system in systems),
    }


def _added_prediction_diagnostic(
    *,
    gold_records: Sequence[GoldRecord],
    candidate_records: Sequence[PredictionRecord],
    comparator_records: Sequence[PredictionRecord],
    suite: ThreeMetricPairedSuite,
) -> dict[str, Any]:
    """Decompose candidate-only predictions without retaining their keys."""

    gold_by_id = {record.doc_id: record for record in gold_records}
    candidate_by_id = {record.doc_id: record for record in candidate_records}
    comparator_by_id = {record.doc_id: record for record in comparator_records}

    recovered_comparator_fn = 0
    suppressed_comparator_tp = 0
    added_fp = 0
    suppressed_comparator_fp = 0
    for doc_id in sorted(gold_by_id):
        gold_keys = {span.metric_key() for span in gold_by_id[doc_id].spans}
        candidate_keys = {span.metric_key() for span in candidate_by_id[doc_id].spans}
        comparator_keys = {span.metric_key() for span in comparator_by_id[doc_id].spans}
        candidate_added = candidate_keys - comparator_keys
        candidate_suppressed = comparator_keys - candidate_keys
        recovered_comparator_fn += len(candidate_added & gold_keys)
        added_fp += len(candidate_added - gold_keys)
        suppressed_comparator_tp += len(candidate_suppressed & gold_keys)
        suppressed_comparator_fp += len(candidate_suppressed - gold_keys)

    added_prediction_count = recovered_comparator_fn + added_fp
    precision = (
        None
        if added_prediction_count == 0
        else Fraction(recovered_comparator_fn, added_prediction_count)
    )
    candidate_counts = _strict_counts(suite, side="candidate")
    comparator_counts = _strict_counts(suite, side="comparator")
    delta = {
        key: candidate_counts[key] - comparator_counts[key] for key in ("tp", "fp", "fn")
    }
    if delta != {
        "tp": recovered_comparator_fn - suppressed_comparator_tp,
        "fp": added_fp - suppressed_comparator_fp,
        "fn": suppressed_comparator_tp - recovered_comparator_fn,
    }:
        raise RuntimeError("candidate/comparator strict-count decomposition is inconsistent")

    return {
        "candidate_added_prediction_count": added_prediction_count,
        "candidate_added_unique_true_positive_count": recovered_comparator_fn,
        "candidate_added_unique_tp_precision": {
            "value": None if precision is None else float(precision),
            "numerator": recovered_comparator_fn,
            "denominator": added_prediction_count,
        },
        "recovered_comparator_false_negative_count": recovered_comparator_fn,
        "suppressed_comparator_true_positive_count": suppressed_comparator_tp,
        "added_false_positive_count": added_fp,
        "suppressed_comparator_false_positive_count": suppressed_comparator_fp,
        "strict_counts": {
            "candidate": candidate_counts,
            "comparator": comparator_counts,
            "delta_candidate_minus_comparator": delta,
        },
    }


def build_service_quality_suite(
    *,
    suite_name: str,
    candidate_id: str,
    gold_records: Sequence[GoldRecord],
    candidate_records: Sequence[PredictionRecord],
    comparators: Sequence[ServiceQualityComparatorInput],
    labels: Sequence[str],
    pii_free_eligible_doc_ids: Sequence[str],
    pii_free_eligibility_complete: bool,
) -> dict[str, Any]:
    """Build a non-activating, aggregate-only fixed-24 service suite.

    ``pii_free_eligibility_complete`` is an explicit caller assertion that the
    supplied eligible IDs and their complement classify the complete gold
    universe.  It is required in addition to the existing invariant that an
    eligible document has no protocol-gold span.
    """

    normalized_suite_name = _normalized_id(suite_name, field="suite_name")
    normalized_candidate_id = _normalized_id(candidate_id, field="candidate_id")
    protocol_labels = _fixed_labels(labels)
    if pii_free_eligibility_complete is not True:
        raise EvaluationDataError(
            "PII-free eligibility must be explicitly asserted complete"
        )
    if isinstance(pii_free_eligible_doc_ids, (str, bytes)):
        raise EvaluationDataError("PII-free eligible doc IDs must be an explicit sequence")
    pii_free_ids = tuple(pii_free_eligible_doc_ids)

    gold = _gold_records(gold_records)
    candidate = _prediction_records(candidate_records, field="candidate_records")
    if isinstance(comparators, (str, bytes)):
        raise EvaluationDataError("comparators must be a sequence")
    comparator_inputs = tuple(comparators)
    if not comparator_inputs:
        raise EvaluationDataError("at least one comparator is required")
    if any(
        not isinstance(comparator, ServiceQualityComparatorInput)
        for comparator in comparator_inputs
    ):
        raise EvaluationDataError(
            "comparators must contain ServiceQualityComparatorInput values"
        )

    normalized_comparators: list[
        tuple[str, tuple[PredictionRecord, ...]]
    ] = []
    for comparator in comparator_inputs:
        comparator_id = _normalized_id(comparator.comparator_id, field="comparator_id")
        if comparator_id == normalized_candidate_id:
            raise EvaluationDataError("candidate_id cannot equal a comparator_id")
        predictions = _prediction_records(
            comparator.predictions,
            field=f"comparator {comparator_id} predictions",
        )
        normalized_comparators.append((comparator_id, predictions))
    comparator_ids = [comparator_id for comparator_id, _ in normalized_comparators]
    if len(comparator_ids) != len(set(comparator_ids)):
        raise EvaluationDataError("comparator IDs must be unique")
    normalized_comparators.sort(key=lambda value: value[0])

    suites: dict[str, tuple[ThreeMetricPairedSuite, ...]] = {}
    comparator_records_by_id: dict[str, tuple[PredictionRecord, ...]] = {}
    for comparator_id, predictions in normalized_comparators:
        suite = build_three_metric_paired_suite(
            name=normalized_suite_name,
            gold_records=gold,
            candidate_records=candidate,
            comparator_records=predictions,
            labels=protocol_labels,
            pii_free_eligible_doc_ids=pii_free_ids,
        )
        suites[comparator_id] = (suite,)
        comparator_records_by_id[comparator_id] = predictions

    observed_gold_labels = {span.label for record in gold for span in record.spans}
    if observed_gold_labels != set(protocol_labels):
        raise EvaluationDataError(
            "gold must contain support for every canonical PII-core label"
        )

    admitted_comparator_ids = tuple(sorted(suites))
    bootstrap = frozen_three_metric_intersection_union_bootstrap(
        suites,
        candidate_id=normalized_candidate_id,
        admitted_comparator_ids=admitted_comparator_ids,
    )
    if bootstrap.get("claim_activation", {}).get("allowed") is not False:
        raise RuntimeError("frozen three-metric bootstrap unexpectedly activated a claim")

    candidate_alignment_sha256 = bootstrap["suite_alignment_sha256"][
        normalized_suite_name
    ]
    comparison_summaries: dict[str, Any] = {}
    comparator_alignment_hashes: dict[str, str] = {}
    for comparator_id in admitted_comparator_ids:
        comparator_input_sha256 = bootstrap["comparisons"][comparator_id][
            "comparator_input_sha256"
        ]
        comparator_alignment_sha256 = canonical_json_hash(
            {
                "domain": "pii-zh-service-quality-comparator-alignment-v1",
                "candidate_gold_alignment_sha256": candidate_alignment_sha256,
                "comparator_id": comparator_id,
                "comparator_input_sha256": comparator_input_sha256,
            }
        )
        comparator_alignment_hashes[comparator_id] = comparator_alignment_sha256
        diagnostic = _added_prediction_diagnostic(
            gold_records=gold,
            candidate_records=candidate,
            comparator_records=comparator_records_by_id[comparator_id],
            suite=suites[comparator_id][0],
        )
        comparison_summaries[comparator_id] = {
            "input_alignment_sha256": comparator_alignment_sha256,
            **diagnostic,
        }

    input_alignment_sha256 = canonical_json_hash(
        {
            "domain": "pii-zh-service-quality-suite-input-alignment-v1",
            "suite_name": normalized_suite_name,
            "candidate_id": normalized_candidate_id,
            "candidate_gold_alignment_sha256": candidate_alignment_sha256,
            "comparator_alignment_sha256": comparator_alignment_hashes,
            "labels": list(protocol_labels),
            "pii_free_eligibility_complete": True,
        }
    )
    return {
        "schema_version": 1,
        "suite_name": normalized_suite_name,
        "status": "DEVELOPMENT_AGGREGATE_ONLY_NON_ACTIVATING",
        "candidate_id": normalized_candidate_id,
        "comparator_ids": list(admitted_comparator_ids),
        "document_count": len(gold),
        "pii_free_eligible_document_count": len(pii_free_ids),
        "pii_free_ineligible_document_count": len(gold) - len(pii_free_ids),
        "pii_free_eligibility_explicit_complete": True,
        "protocol": {
            "scope": "fixed_canonical_pii_core_24",
            "label_count": len(protocol_labels),
            "ordered_labels": list(protocol_labels),
            "all_labels_have_gold_support": True,
        },
        "input_alignment_sha256": input_alignment_sha256,
        "candidate_gold_alignment_sha256": candidate_alignment_sha256,
        "comparisons": comparison_summaries,
        "frozen_three_metric_bootstrap": bootstrap,
        "formal_suite_receipt": {
            "present": False,
            "activated": False,
            "required_for_formal_use": True,
            "reason": "a later release-bound contract must bind and activate the suite",
        },
        "claim_activation": {
            "allowed": False,
            "reason": "development aggregation cannot activate first, best, or SOTA claims",
        },
        "privacy": {
            "aggregate_only": True,
            "raw_text_retained": False,
            "raw_doc_ids_retained": False,
            "entity_values_retained": False,
            "per_document_statistics_output": False,
        },
    }


__all__ = [
    "PII_CORE_LABELS",
    "ServiceQualityComparatorInput",
    "build_service_quality_suite",
]
