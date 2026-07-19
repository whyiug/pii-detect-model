#!/usr/bin/env python3
"""Evaluate raw-text-free prediction JSONL against character-span gold JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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


def _assert_safe_output(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.output is None:
        return
    output = args.output.expanduser().resolve(strict=False)
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite an existing evaluation output")
    inputs = [args.gold, args.predictions]
    inputs.extend(
        path
        for path in (
            args.dataset_manifest,
            args.prediction_manifest,
            args.model_training_manifest,
        )
        if path is not None
    )
    if output in {path.expanduser().resolve(strict=False) for path in inputs}:
        parser.error("evaluation output collides with an input artifact")
    model_manifest = args.model_training_manifest
    if model_manifest is not None:
        model_root = model_manifest.expanduser().resolve(strict=False).parent
        if output == model_root or model_root in output.parents:
            parser.error("evaluation output must not be written inside the model artifact")
def _write_new_text(serialized: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite an existing evaluation output")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination.chmod(0o444)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _assert_safe_output(args, parser)
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
            _write_new_text(serialized, args.output)
    except (EvaluationDataError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
