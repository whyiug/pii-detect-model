"""Raw-text-free prediction-record adapter for deterministic system fusion."""

from __future__ import annotations

from pii_zh.evaluation.io import PredictionRecord, Span

from .deterministic import (
    DEFAULT_SOURCE_PRIORITY,
    DEFAULT_SPECIFICITY,
    DeterministicFusion,
)


def system_fusion_configuration() -> dict[str, object]:
    """Return the explicit fixed fusion configuration used by release manifests."""

    return {
        "schema_version": 1,
        "default_threshold": 0.0,
        "model_threshold": 0.0,
        "keep_nested_different_types": False,
        "model_source": "qwen",
        "rules_source": "rule:cn_common",
        "specificity": dict(sorted(DEFAULT_SPECIFICITY.items())),
        "source_priority": dict(sorted(DEFAULT_SOURCE_PRIORITY.items())),
    }


def _indexed(
    records: list[PredictionRecord], *, kind: str
) -> tuple[list[str], dict[str, PredictionRecord]]:
    order = [record.doc_id for record in records]
    indexed = {record.doc_id: record for record in records}
    if len(order) != len(indexed):
        raise ValueError(f"{kind} predictions contain duplicate document identifiers")
    return order, indexed


def fuse_prediction_records(
    rules_records: list[PredictionRecord],
    model_records: list[PredictionRecord],
    *,
    model_threshold: float = 0.0,
    keep_nested_different_types: bool = False,
) -> list[PredictionRecord]:
    """Fuse aligned model/rule records using the same adapter as the CLI."""

    if not 0.0 <= model_threshold <= 1.0:
        raise ValueError("model threshold must be between zero and one")
    rule_order, rules = _indexed(rules_records, kind="rule")
    _, model = _indexed(model_records, kind="model")
    if set(rules) != set(model):
        raise ValueError("rule and model prediction document sets differ")
    fusion = DeterministicFusion(
        default_threshold=0.0,
        specificity=DEFAULT_SPECIFICITY,
        source_priority=DEFAULT_SOURCE_PRIORITY,
        keep_nested_different_types=keep_nested_different_types,
    )
    output: list[PredictionRecord] = []
    for doc_id in rule_order:
        rule_items = [
            {
                "entity_type": span.label,
                "start": span.start,
                "end": span.end,
                "score": 1.0 if span.score is None else span.score,
                "source": "rule:cn_common",
                "metadata": {"validator_valid": True},
            }
            for span in rules[doc_id].spans
        ]
        model_items = [
            {
                "entity_type": span.label,
                "start": span.start,
                "end": span.end,
                "score": 1.0 if span.score is None else span.score,
                "source": "qwen",
            }
            for span in model[doc_id].spans
            if span.score is None or span.score >= model_threshold
        ]
        fused = fusion.fuse(rule_items, model_items)
        output.append(
            PredictionRecord(
                doc_id=doc_id,
                spans=tuple(
                    Span(
                        start=item.start,
                        end=item.end,
                        label=item.entity_type,
                        score=item.score,
                    )
                    for item in fused
                ),
            )
        )
    return output


__all__ = ["fuse_prediction_records", "system_fusion_configuration"]
