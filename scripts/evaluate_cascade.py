#!/usr/bin/env python3
"""Run the published cascade and evaluate privacy-safe character spans."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.cascade import CascadeConfig, CascadeMode, CascadePipeline  # noqa: E402
from pii_zh.data.schema import SchemaError, iter_jsonl  # noqa: E402
from pii_zh.evaluation import (  # noqa: E402
    EvaluationDataError,
    PredictionRecord,
    Span,
    build_evaluation_provenance,
    canonical_json_hash,
    evaluate_documents,
    load_gold_jsonl,
    load_prediction_jsonl,
    sha256_file,
    write_prediction_jsonl,
)
from pii_zh.evaluation.required_metrics import build_required_metrics  # noqa: E402
from pii_zh.taxonomy import load_presidio_mapping  # noqa: E402

CASCADE_EVALUATION_SCHEMA_VERSION = "pii-zh.cascade-evaluation.v1"


class SafeCliError(RuntimeError):
    """An allowlisted CLI failure message which contains no input values."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CascadePipeline on canonical JSONL, write raw-text-free predictions, "
            "and evaluate them with the shared character-span evaluator."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Canonical JSONL input")
    parser.add_argument("--predictions", required=True, type=Path, help="New prediction JSONL")
    parser.add_argument("--report", required=True, type=Path, help="New evaluation report JSON")
    parser.add_argument(
        "--mode",
        choices=("rules-only", "model-only", "cascade"),
        default="rules-only",
    )
    parser.add_argument("--model-path", type=Path, help="Existing local model directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--document-batch-size", type=int, default=32)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--stride-fraction", type=float, default=0.25)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--entity", action="append", dest="entities")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_args(args: argparse.Namespace) -> None:
    outputs = [_resolved(args.predictions), _resolved(args.report)]
    if len(set(outputs)) != len(outputs):
        raise SafeCliError("prediction and report outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in (args.predictions, args.report)):
        raise SafeCliError("refusing to overwrite an existing output")
    input_path = _resolved(args.input)
    if input_path in outputs:
        raise SafeCliError("an output collides with the canonical input")
    if not args.input.is_file():
        raise SafeCliError("canonical input must be an existing local file")
    if args.document_batch_size < 1 or args.window_batch_size < 1 or args.max_tokens < 1:
        raise SafeCliError("batch sizes and max tokens must be positive")
    if not 0.0 < args.stride_fraction < 1.0:
        raise SafeCliError("stride fraction must be between zero and one")

    mode = cast(CascadeMode, args.mode)
    if mode == "rules-only":
        if args.model_path is not None or args.calibration is not None:
            raise SafeCliError("model options are invalid in rules-only mode")
        return
    if args.model_path is None:
        raise SafeCliError("model-enabled modes require a local model directory")
    try:
        model_root = args.model_path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SafeCliError("model path must be an existing local directory") from exc
    if not model_root.is_dir():
        raise SafeCliError("model path must be an existing local directory")
    if any(output == model_root or model_root in output.parents for output in outputs):
        raise SafeCliError("outputs must not be written inside the model artifact")
    args.model_path = model_root
    if args.calibration is not None and not args.calibration.is_file():
        raise SafeCliError("calibration must be an existing local file")


def _config_payload(config: CascadeConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
        "routes": [asdict(route) for route in config.routes],
    }


def _implementation_identity() -> dict[str, object]:
    components = {
        relative: sha256_file(_REPOSITORY_ROOT / relative)
        for relative in (
            "scripts/evaluate_cascade.py",
            "src/pii_zh/cascade/config.py",
            "src/pii_zh/cascade/pipeline.py",
            "src/pii_zh/cascade/routing.py",
            "src/pii_zh/evaluation/required_metrics.py",
        )
    }
    return {
        "component_sha256": components,
        "implementation_sha256": canonical_json_hash(components),
    }


def _build_pipeline(args: argparse.Namespace, config: CascadeConfig) -> CascadePipeline:
    if config.mode == "rules-only":
        return CascadePipeline(config=config)
    assert args.model_path is not None
    try:
        return CascadePipeline.from_pretrained(
            args.model_path,
            config=config,
            device=args.device,
            micro_batch_size=args.window_batch_size,
            calibration=args.calibration,
            max_tokens=args.max_tokens,
            stride_fraction=args.stride_fraction,
            local_files_only=True,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SafeCliError("local model initialization failed") from exc


def _model_manifest_sha256(model_path: Path | None) -> str | None:
    if model_path is None:
        return None
    manifest = model_path / "training_manifest.json"
    try:
        return sha256_file(manifest)
    except OSError as exc:
        raise SafeCliError("local model manifest binding failed") from exc


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, prefix=f".{destination.name}."
    )
    path = Path(handle.name)
    handle.close()
    return path


def _generate_predictions(
    *,
    pipeline: CascadePipeline,
    input_path: Path,
    temporary: Path,
    batch_size: int,
    entities: Sequence[str] | None,
) -> int:
    mapping = load_presidio_mapping()
    presidio_to_model = {value: key for key, value in mapping.model_to_presidio.items()}
    seen: set[str] = set()
    count = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def flush(handle: Any) -> None:
        nonlocal count
        if not batch_texts:
            return
        try:
            detected = pipeline.detect_batch(batch_texts, entities=entities)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise SafeCliError("cascade inference failed") from exc
        predictions = []
        for doc_id, detections in zip(batch_ids, detected, strict=True):
            predictions.append(
                PredictionRecord(
                    doc_id=doc_id,
                    spans=tuple(
                        Span(
                            start=item.start,
                            end=item.end,
                            label=presidio_to_model.get(item.entity_type, item.entity_type),
                            score=item.score,
                        )
                        for item in detections
                    ),
                )
            )
        write_prediction_jsonl(predictions, handle)
        count += len(predictions)
        batch_ids.clear()
        batch_texts.clear()

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in iter_jsonl(input_path):
                if record.doc_id in seen:
                    raise SafeCliError("canonical input contains duplicate document identifiers")
                seen.add(record.doc_id)
                batch_ids.append(record.doc_id)
                batch_texts.append(record.text)
                if len(batch_texts) >= batch_size:
                    flush(handle)
            flush(handle)
    except SafeCliError:
        raise
    except (EvaluationDataError, OSError, SchemaError, TypeError, UnicodeError, ValueError) as exc:
        raise SafeCliError("canonical input or prediction generation failed") from exc
    if count == 0:
        raise SafeCliError("canonical input must contain at least one document")
    return count


def _report(
    *,
    args: argparse.Namespace,
    config: CascadeConfig,
    temporary_predictions: Path,
    document_count: int,
) -> dict[str, Any]:
    try:
        gold = load_gold_jsonl(args.input)
        predictions = load_prediction_jsonl(temporary_predictions)
        report = evaluate_documents(
            gold,
            predictions,
            iou_threshold=args.iou_threshold,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        report["required_metrics"] = build_required_metrics(gold, predictions)
        report["provenance"] = build_evaluation_provenance(
            gold_path=args.input,
            predictions_path=temporary_predictions,
            gold_records=gold,
            evaluation_parameters={
                "bootstrap_samples": args.bootstrap_samples,
                "confidence": args.confidence,
                "iou_threshold": args.iou_threshold,
                "seed": args.seed,
            },
        )
    except (EvaluationDataError, OSError, TypeError, ValueError) as exc:
        raise SafeCliError("strict evaluation failed") from exc

    config_payload = _config_payload(config)
    calibration_sha256 = sha256_file(args.calibration) if args.calibration is not None else None
    report["cascade_runtime"] = {
        "schema_version": CASCADE_EVALUATION_SCHEMA_VERSION,
        "mode": config.mode,
        "profile_version": config.profile_version,
        "configuration": config_payload,
        "configuration_sha256": canonical_json_hash(config_payload),
        "implementation": _implementation_identity(),
        "model_training_manifest_file_sha256": _model_manifest_sha256(args.model_path),
        "calibration_sha256": calibration_sha256,
        "document_batch_size": args.document_batch_size,
        "window_batch_size": args.window_batch_size if config.uses_model else None,
        "max_tokens": args.max_tokens if config.uses_model else None,
        "stride_fraction": args.stride_fraction if config.uses_model else None,
        "entity_filter": sorted(args.entities) if args.entities is not None else None,
        "prediction_document_count": document_count,
        "predictions_sha256": sha256_file(temporary_predictions),
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
        },
    }
    return report


def _publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
        destination.chmod(0o444)
    except OSError as exc:
        raise SafeCliError("failed to publish a new output") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    temporary_predictions: Path | None = None
    temporary_report: Path | None = None
    published: list[Path] = []
    try:
        _validate_args(args)
        config = CascadeConfig(mode=cast(CascadeMode, args.mode))
        pipeline = _build_pipeline(args, config)
        temporary_predictions = _temporary_path(args.predictions)
        document_count = _generate_predictions(
            pipeline=pipeline,
            input_path=args.input,
            temporary=temporary_predictions,
            batch_size=args.document_batch_size,
            entities=args.entities,
        )
        report = _report(
            args=args,
            config=config,
            temporary_predictions=temporary_predictions,
            document_count=document_count,
        )
        temporary_report = _temporary_path(args.report)
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish(temporary_predictions, args.predictions)
        published.append(args.predictions)
        _publish(temporary_report, args.report)
        published.append(args.report)
        print(
            json.dumps(
                {
                    "documents": document_count,
                    "mode": config.mode,
                    "predictions_sha256": sha256_file(args.predictions),
                    "report_sha256": sha256_file(args.report),
                },
                sort_keys=True,
            )
        )
    except SafeCliError as exc:
        for path in published:
            path.unlink(missing_ok=True)
        print(f"evaluate-cascade: error: {exc}", file=sys.stderr)
        return 2
    finally:
        for temporary in (temporary_predictions, temporary_report):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
