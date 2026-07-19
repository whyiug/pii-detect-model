"""Reusable privacy-safe aggregates for model/framework baseline matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .io import EvaluationDataError

PUBLIC_SCORE_DECIMALS = 8
_COUNT_FIELDS = ("tp", "fp", "fn", "support", "predicted")
_SCORE_FIELDS = ("precision", "recall", "f1", "f2")


def rounded_score(value: object) -> float | None:
    """Normalize an evaluation score for publication-safe deterministic JSON."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationDataError("evaluation score must be numeric or null")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise EvaluationDataError("evaluation score must be between zero and one")
    return round(result, PUBLIC_SCORE_DECIMALS)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{field} must be an object")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationDataError(f"{field} must be a non-negative integer")
    return value


def _compact_counts(value: object, *, field: str) -> dict[str, int | float]:
    source = _mapping(value, field=field)
    counts = {
        name: _non_negative_int(source.get(name), field=f"{field}.{name}")
        for name in _COUNT_FIELDS
    }
    if counts["support"] != counts["tp"] + counts["fn"]:
        raise EvaluationDataError(f"{field}.support is inconsistent")
    if counts["predicted"] != counts["tp"] + counts["fp"]:
        raise EvaluationDataError(f"{field}.predicted is inconsistent")
    scores: dict[str, float] = {}
    for name in _SCORE_FIELDS:
        score = rounded_score(source.get(name))
        if score is None:
            raise EvaluationDataError(f"{field} scores must not be null")
        scores[name] = score
    return {**counts, **scores}


def compact_strict_result(
    evaluation: Mapping[str, Any],
    *,
    expected_labels: Sequence[str],
) -> dict[str, Any]:
    """Reduce a full evaluator result to aggregate-only strict matrix fields."""

    labels = tuple(expected_labels)
    if not labels or len(set(labels)) != len(labels):
        raise EvaluationDataError("expected_labels must be unique and non-empty")
    strict = _mapping(evaluation.get("strict"), field="strict")
    per_class = _mapping(strict.get("per_class"), field="strict.per_class")
    if set(per_class) != set(labels):
        raise EvaluationDataError("strict.per_class does not match the comparison protocol")
    macro = _mapping(strict.get("macro"), field="strict.macro")
    if _non_negative_int(macro.get("label_count"), field="strict.macro.label_count") != len(
        labels
    ):
        raise EvaluationDataError("strict.macro.label_count is inconsistent")
    span_count = _mapping(evaluation.get("span_count"), field="span_count")
    pii_free = _mapping(evaluation.get("pii_free"), field="pii_free")
    strict_micro = _compact_counts(strict.get("micro"), field="strict.micro")
    gold_spans = _non_negative_int(span_count.get("gold"), field="span_count.gold")
    predicted_spans = _non_negative_int(
        span_count.get("predicted"), field="span_count.predicted"
    )
    result: dict[str, Any] = {
        "documents": _non_negative_int(evaluation.get("document_count"), field="document_count"),
        "gold_spans": gold_spans,
        "predicted_spans": predicted_spans,
        "strict_micro": strict_micro,
        "strict_macro": {
            "label_count": len(labels),
            **{name: rounded_score(macro.get(name)) for name in _SCORE_FIELDS},
        },
        "strict_per_label": {
            label: _compact_counts(per_class[label], field=f"strict.per_class.{label}")
            for label in labels
        },
        "pii_free": {
            "documents": _non_negative_int(pii_free.get("documents"), field="pii_free.documents"),
            "false_positive_documents": _non_negative_int(
                pii_free.get("false_positive_documents"),
                field="pii_free.false_positive_documents",
            ),
            "false_positive_rate": rounded_score(pii_free.get("false_positive_rate")),
        },
    }
    if strict_micro["support"] != gold_spans:
        raise EvaluationDataError("strict support and gold span count disagree")
    if strict_micro["predicted"] != predicted_spans:
        raise EvaluationDataError("strict predicted count and span count disagree")
    return result


def _score_from_counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
        "precision": round(precision, PUBLIC_SCORE_DECIMALS),
        "recall": round(recall, PUBLIC_SCORE_DECIMALS),
        "f1": round(f1, PUBLIC_SCORE_DECIMALS),
        "f2": round(f2, PUBLIC_SCORE_DECIMALS),
    }


def pool_strict_results(
    results: Mapping[str, Mapping[str, Any]],
    *,
    expected_labels: Sequence[str],
) -> dict[str, Any]:
    """Count-pool compact suite results without accessing record-level data."""

    if not results:
        raise EvaluationDataError("at least one suite result is required")
    labels = tuple(expected_labels)
    per_label: dict[str, dict[str, int | float]] = {}
    for label in labels:
        per_label[label] = _score_from_counts(
            tp=sum(int(result["strict_per_label"][label]["tp"]) for result in results.values()),
            fp=sum(int(result["strict_per_label"][label]["fp"]) for result in results.values()),
            fn=sum(int(result["strict_per_label"][label]["fn"]) for result in results.values()),
        )
    micro = _score_from_counts(
        tp=sum(int(result["strict_micro"]["tp"]) for result in results.values()),
        fp=sum(int(result["strict_micro"]["fp"]) for result in results.values()),
        fn=sum(int(result["strict_micro"]["fn"]) for result in results.values()),
    )
    macro = {
        "label_count": len(labels),
        **{
            field: round(
                sum(float(per_label[label][field]) for label in labels) / len(labels),
                PUBLIC_SCORE_DECIMALS,
            )
            for field in _SCORE_FIELDS
        },
    }
    pii_free_documents = sum(int(result["pii_free"]["documents"]) for result in results.values())
    false_positive_documents = sum(
        int(result["pii_free"]["false_positive_documents"]) for result in results.values()
    )
    return {
        "documents": sum(int(result["documents"]) for result in results.values()),
        "gold_spans": micro["support"],
        "predicted_spans": micro["predicted"],
        "strict_micro": micro,
        "strict_macro": macro,
        "strict_per_label": per_label,
        "pii_free": {
            "documents": pii_free_documents,
            "false_positive_documents": false_positive_documents,
            "false_positive_rate": (
                round(
                    false_positive_documents / pii_free_documents,
                    PUBLIC_SCORE_DECIMALS,
                )
                if pii_free_documents
                else None
            ),
        },
    }


__all__ = [
    "PUBLIC_SCORE_DECIMALS",
    "compact_strict_result",
    "pool_strict_results",
    "rounded_score",
]
