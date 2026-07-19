from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import pytest

from pii_zh.evaluation import EvaluationDataError, GoldRecord, PredictionRecord, Span
from pii_zh.evaluation.service_quality_suite import (
    PII_CORE_LABELS,
    ServiceQualityComparatorInput,
    build_service_quality_suite,
)


def _fixture(
    *, negative_count: int = 2, prefix: str = "quality"
) -> tuple[
    list[GoldRecord],
    dict[str, tuple[Span, ...]],
    list[str],
]:
    gold: list[GoldRecord] = []
    perfect: dict[str, tuple[Span, ...]] = {}
    for index, label in enumerate(PII_CORE_LABELS):
        doc_id = f"{prefix}-positive-{index:02d}"
        gold_span = Span(0, 1, label)
        gold.append(GoldRecord(doc_id, (gold_span,), text_length=8))
        perfect[doc_id] = (Span(0, 1, label, 0.9),)
    negative_ids = [f"{prefix}-negative-{index:02d}" for index in range(negative_count)]
    gold.extend(GoldRecord(doc_id, (), text_length=8) for doc_id in negative_ids)
    return gold, perfect, negative_ids


def _predictions(
    gold: Sequence[GoldRecord],
    by_doc_id: Mapping[str, Sequence[Span]],
) -> list[PredictionRecord]:
    return [
        PredictionRecord(record.doc_id, tuple(by_doc_id.get(record.doc_id, ())))
        for record in gold
    ]


def _build(
    *,
    gold: Sequence[GoldRecord],
    candidate: Sequence[PredictionRecord],
    comparator: Sequence[PredictionRecord],
    negative_ids: Sequence[str],
    comparator_id: str = "rules-only",
) -> dict[str, object]:
    return build_service_quality_suite(
        suite_name="production-pii-core-dev",
        candidate_id="candidate-cascade",
        gold_records=gold,
        candidate_records=candidate,
        comparators=(ServiceQualityComparatorInput(comparator_id, comparator),),
        labels=PII_CORE_LABELS,
        pii_free_eligible_doc_ids=negative_ids,
        pii_free_eligibility_complete=True,
    )


def test_added_unique_tp_duplicate_suppression_and_added_fp_are_aggregate_only() -> None:
    sentinel = "DO_NOT_OUTPUT_DOCUMENT_ID_734b9d"
    gold, perfect, negative_ids = _fixture(prefix=sentinel)
    candidate = dict(perfect)
    comparator = dict(perfect)

    recovered_id = gold[1].doc_id
    suppressed_id = gold[2].doc_id
    comparator[recovered_id] = ()
    candidate[suppressed_id] = ()
    candidate[negative_ids[0]] = (Span(2, 3, PII_CORE_LABELS[3], 0.7),)

    result = _build(
        gold=gold,
        candidate=_predictions(gold, candidate),
        comparator=_predictions(gold, comparator),
        negative_ids=negative_ids,
    )
    comparison = result["comparisons"]["rules-only"]  # type: ignore[index]
    assert comparison["candidate_added_prediction_count"] == 2
    assert comparison["candidate_added_unique_true_positive_count"] == 1
    assert comparison["candidate_added_unique_tp_precision"] == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert comparison["recovered_comparator_false_negative_count"] == 1
    assert comparison["suppressed_comparator_true_positive_count"] == 1
    assert comparison["added_false_positive_count"] == 1
    assert comparison["suppressed_comparator_false_positive_count"] == 0
    assert comparison["strict_counts"] == {
        "candidate": {"tp": 23, "fp": 1, "fn": 1},
        "comparator": {"tp": 23, "fp": 0, "fn": 1},
        "delta_candidate_minus_comparator": {"tp": 0, "fp": 1, "fn": 0},
    }

    # The 22 shared true positives are comparator duplicates, not candidate-added TPs.
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert sentinel not in serialized
    assert all(doc_id not in serialized for doc_id in negative_ids)
    assert re.fullmatch(r"[0-9a-f]{64}", result["input_alignment_sha256"])  # type: ignore[arg-type]
    assert result["privacy"] == {
        "aggregate_only": True,
        "raw_text_retained": False,
        "raw_doc_ids_retained": False,
        "entity_values_retained": False,
        "per_document_statistics_output": False,
    }
    assert result["claim_activation"]["allowed"] is False  # type: ignore[index]
    assert result["formal_suite_receipt"]["activated"] is False  # type: ignore[index]


def test_frozen_three_metric_fpr_gate_is_reused_without_claim_activation() -> None:
    gold, perfect, negative_ids = _fixture(negative_count=20, prefix="fpr")
    candidate = dict(perfect)
    for doc_id in negative_ids[:2]:
        candidate[doc_id] = (Span(2, 3, "PHONE_NUMBER", 0.8),)

    result = _build(
        gold=gold,
        candidate=_predictions(gold, candidate),
        comparator=_predictions(gold, perfect),
        negative_ids=negative_ids,
    )
    bootstrap = result["frozen_three_metric_bootstrap"]
    assert bootstrap["frozen_statistical_parameters"]["matches"] is True  # type: ignore[index]
    assert bootstrap["fpr_safety_gate"]["candidate_pooled_point_estimate"] == 0.1  # type: ignore[index]
    assert bootstrap["fpr_safety_gate"]["absolute_maximum_passed"] is False  # type: ignore[index]
    assert bootstrap["intersection_union"]["formal_statistical_component_passed"] is False  # type: ignore[index]
    assert bootstrap["claim_activation"]["allowed"] is False  # type: ignore[index]
    comparison = result["comparisons"]["rules-only"]  # type: ignore[index]
    assert comparison["added_false_positive_count"] == 2
    assert comparison["strict_counts"]["delta_candidate_minus_comparator"] == {  # type: ignore[index]
        "tp": 0,
        "fp": 2,
        "fn": 0,
    }


def test_each_comparator_gets_an_independent_aligned_increment_summary() -> None:
    gold, perfect, negative_ids = _fixture(prefix="multi")
    candidate = _predictions(gold, perfect)
    weaker = dict(perfect)
    weaker[gold[0].doc_id] = ()
    result = build_service_quality_suite(
        suite_name="quality",
        candidate_id="candidate",
        gold_records=gold,
        candidate_records=candidate,
        comparators=(
            ServiceQualityComparatorInput("same", candidate),
            ServiceQualityComparatorInput("weaker", _predictions(gold, weaker)),
        ),
        labels=PII_CORE_LABELS,
        pii_free_eligible_doc_ids=negative_ids,
        pii_free_eligibility_complete=True,
    )

    assert result["comparator_ids"] == ["same", "weaker"]
    comparisons = result["comparisons"]
    assert comparisons["same"]["candidate_added_unique_tp_precision"] == {  # type: ignore[index]
        "value": None,
        "numerator": 0,
        "denominator": 0,
    }
    assert comparisons["weaker"]["recovered_comparator_false_negative_count"] == 1  # type: ignore[index]
    assert (
        comparisons["same"]["input_alignment_sha256"]  # type: ignore[index]
        != comparisons["weaker"]["input_alignment_sha256"]  # type: ignore[index]
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "misaligned"])
def test_document_alignment_failures_are_rejected(mutation: str) -> None:
    gold, perfect, negative_ids = _fixture()
    candidate = _predictions(gold, perfect)
    comparator = _predictions(gold, perfect)
    if mutation == "missing":
        candidate.pop()
    elif mutation == "duplicate":
        candidate.append(candidate[0])
    else:
        comparator[-1] = PredictionRecord("unexpected-document", ())

    with pytest.raises(EvaluationDataError, match="duplicate|exactly"):
        _build(
            gold=gold,
            candidate=candidate,
            comparator=comparator,
            negative_ids=negative_ids,
        )


def test_fixed_24_protocol_and_gold_coverage_fail_closed() -> None:
    gold, perfect, negative_ids = _fixture()
    candidate = _predictions(gold, perfect)
    comparator = _predictions(gold, perfect)
    with pytest.raises(EvaluationDataError, match="exact ordered canonical 24-class"):
        build_service_quality_suite(
            suite_name="quality",
            candidate_id="candidate",
            gold_records=gold,
            candidate_records=candidate,
            comparators=(ServiceQualityComparatorInput("baseline", comparator),),
            labels=PII_CORE_LABELS[:-1],
            pii_free_eligible_doc_ids=negative_ids,
            pii_free_eligibility_complete=True,
        )

    reduced_gold = gold[1:]
    reduced_candidate = _predictions(reduced_gold, perfect)
    with pytest.raises(EvaluationDataError, match="support for every canonical"):
        build_service_quality_suite(
            suite_name="quality",
            candidate_id="candidate",
            gold_records=reduced_gold,
            candidate_records=reduced_candidate,
            comparators=(
                ServiceQualityComparatorInput("baseline", reduced_candidate),
            ),
            labels=PII_CORE_LABELS,
            pii_free_eligible_doc_ids=negative_ids,
            pii_free_eligibility_complete=True,
        )


def test_pii_free_eligibility_requires_explicit_complete_constraint() -> None:
    gold, perfect, negative_ids = _fixture()
    predictions = _predictions(gold, perfect)
    with pytest.raises(EvaluationDataError, match="explicitly asserted complete"):
        build_service_quality_suite(
            suite_name="quality",
            candidate_id="candidate",
            gold_records=gold,
            candidate_records=predictions,
            comparators=(ServiceQualityComparatorInput("baseline", predictions),),
            labels=PII_CORE_LABELS,
            pii_free_eligible_doc_ids=negative_ids,
            pii_free_eligibility_complete=False,
        )
    with pytest.raises(EvaluationDataError, match="must be unique"):
        build_service_quality_suite(
            suite_name="quality",
            candidate_id="candidate",
            gold_records=gold,
            candidate_records=predictions,
            comparators=(ServiceQualityComparatorInput("baseline", predictions),),
            labels=PII_CORE_LABELS,
            pii_free_eligible_doc_ids=(*negative_ids, negative_ids[0]),
            pii_free_eligibility_complete=True,
        )


def test_comparator_universe_is_nonempty_unique_and_disjoint_from_candidate() -> None:
    gold, perfect, negative_ids = _fixture()
    predictions = _predictions(gold, perfect)
    base = {
        "suite_name": "quality",
        "candidate_id": "candidate",
        "gold_records": gold,
        "candidate_records": predictions,
        "labels": PII_CORE_LABELS,
        "pii_free_eligible_doc_ids": negative_ids,
        "pii_free_eligibility_complete": True,
    }
    with pytest.raises(EvaluationDataError, match="at least one comparator"):
        build_service_quality_suite(comparators=(), **base)
    with pytest.raises(EvaluationDataError, match="comparator IDs must be unique"):
        build_service_quality_suite(
            comparators=(
                ServiceQualityComparatorInput("same", predictions),
                ServiceQualityComparatorInput("same", predictions),
            ),
            **base,
        )
    with pytest.raises(EvaluationDataError, match="cannot equal"):
        build_service_quality_suite(
            comparators=(ServiceQualityComparatorInput("candidate", predictions),),
            **base,
        )
    with pytest.raises(EvaluationDataError, match="must contain PredictionRecord"):
        build_service_quality_suite(
            comparators=(ServiceQualityComparatorInput("empty", ()),),
            **base,
        )
