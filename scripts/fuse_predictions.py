#!/usr/bin/env python3
"""Fuse raw-text-free rule and model prediction JSONL deterministically."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import (  # noqa: E402
    PredictionRecord,
    Span,
    load_prediction_jsonl,
    write_prediction_jsonl,
)
from pii_zh.fusion import DeterministicFusion  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-threshold", type=float, default=0.0)
    parser.add_argument(
        "--keep-nested-different-types",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _indexed(
    records: list[PredictionRecord], *, kind: str
) -> tuple[list[str], dict[str, PredictionRecord]]:
    order = [record.doc_id for record in records]
    indexed = {record.doc_id: record for record in records}
    if len(order) != len(indexed):  # defensive; loader already rejects this
        raise ValueError(f"{kind} predictions contain duplicate document identifiers")
    return order, indexed


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.model_threshold <= 1.0:
        raise ValueError("model threshold must be between zero and one")
    rule_order, rules = _indexed(load_prediction_jsonl(args.rules), kind="rule")
    _, model = _indexed(load_prediction_jsonl(args.model), kind="model")
    if set(rules) != set(model):
        raise ValueError("rule and model prediction document sets differ")

    fusion = DeterministicFusion(
        default_threshold=0.0,
        keep_nested_different_types=args.keep_nested_different_types,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        delete=False,
        prefix=f".{args.output.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
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
                    if span.score is None or span.score >= args.model_threshold
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
            write_prediction_jsonl(output, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, args.output)
    print(json.dumps({"documents": len(rule_order), "output": args.output.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
