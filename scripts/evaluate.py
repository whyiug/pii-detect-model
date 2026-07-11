#!/usr/bin/env python3
"""Evaluate raw-text-free prediction JSONL against character-span gold JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import (  # noqa: E402
    EvaluationDataError,
    build_evaluation_provenance,
    evaluate_documents,
    load_gold_jsonl,
    load_prediction_jsonl,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate left-closed/right-open character spans. Prediction rows may "
            "contain only doc_id and spans; reports never contain raw text."
        )
    )
    parser.add_argument("--gold", required=True, type=Path, help="Gold JSONL path")
    parser.add_argument(
        "--predictions",
        "--pred",
        required=True,
        type=Path,
        help="Prediction JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report JSON path (default: stdout)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Content-addressed evaluation dataset manifest",
    )
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help="Content-addressed prediction manifest bound to this prediction JSONL",
    )
    parser.add_argument(
        "--model-training-manifest",
        type=Path,
        help="Content-addressed training manifest for the evaluated model",
    )
    parser.add_argument(
        "--release-mode",
        action="store_true",
        help="Require and cross-check all release provenance manifests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        gold = load_gold_jsonl(args.gold)
        predictions = load_prediction_jsonl(args.predictions)
        report = evaluate_documents(
            gold,
            predictions,
            iou_threshold=args.iou_threshold,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        report["provenance"] = build_evaluation_provenance(
            gold_path=args.gold,
            predictions_path=args.predictions,
            gold_records=gold,
            dataset_manifest_path=args.dataset_manifest,
            prediction_manifest_path=args.prediction_manifest,
            model_training_manifest_path=args.model_training_manifest,
            release_mode=args.release_mode,
            evaluation_parameters={
                "bootstrap_samples": args.bootstrap_samples,
                "confidence": args.confidence,
                "iou_threshold": args.iou_threshold,
                "seed": args.seed,
            },
        )
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            args.output.write_text(serialized, encoding="utf-8")
    except (EvaluationDataError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
