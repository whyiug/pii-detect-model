#!/usr/bin/env python3
"""Generate raw-text-free character-span predictions from a local Qwen3Bi model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.schema import iter_jsonl  # noqa: E402
from pii_zh.evaluation import (  # noqa: E402
    PredictionRecord,
    Span,
    build_model_prediction_manifest,
    canonical_json_hash,
    sha256_file,
    write_prediction_jsonl,
)
from pii_zh.inference import load_local_predictor  # noqa: E402
from pii_zh.presidio import QwenPiiRecognizer  # noqa: E402
from pii_zh.taxonomy import load_taxonomy  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--document-batch-size", type=int, default=32)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--training-manifest",
        "--model-training-manifest",
        dest="training_manifest",
        type=Path,
    )
    parser.add_argument("--prediction-manifest", type=Path)
    return parser


def _write_manifest_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.document_batch_size < 1 or args.window_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    provenance_values = (
        args.dataset_manifest,
        args.training_manifest,
        args.prediction_manifest,
    )
    if any(value is not None for value in provenance_values) and not all(
        value is not None for value in provenance_values
    ):
        parser.error(
            "--dataset-manifest, --training-manifest, and --prediction-manifest "
            "must be supplied together"
        )
    if args.prediction_manifest is not None and (
        args.prediction_manifest.resolve() == args.output.resolve()
    ):
        parser.error("--prediction-manifest and --output must be different files")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    taxonomy = load_taxonomy()
    output_entities = sorted(taxonomy.output_label_names)
    ignored = [entity.name for entity in taxonomy.label_sets["auxiliary"]]
    predictor = load_local_predictor(
        args.model,
        device=args.device,
        dtype=dtype,
        micro_batch_size=args.window_batch_size,
        ignored_labels=ignored,
    )
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        label_mapping={label: label for label in output_entities},
        ignored_labels=ignored,
        calibration=args.calibration,
        default_threshold=args.threshold,
        max_tokens=args.max_tokens,
        model_version="local-model",
        attention_mode=predictor.attention_mode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_doc_ids: set[str] = set()
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False, prefix=f".{args.output.name}."
    ) as handle:
        temporary = Path(handle.name)
        try:
            batch_ids: list[str] = []
            batch_texts: list[str] = []

            def flush() -> None:
                nonlocal count
                if not batch_texts:
                    return
                results = recognizer.analyze_batch(batch_texts, entities=output_entities)
                predictions = []
                for doc_id, spans in zip(batch_ids, results, strict=True):
                    predictions.append(
                        PredictionRecord(
                            doc_id=doc_id,
                            spans=tuple(
                                Span(
                                    start=result.start,
                                    end=result.end,
                                    label=result.entity_type,
                                    score=float(result.score),
                                )
                                for result in spans
                            ),
                        )
                    )
                count += write_prediction_jsonl(predictions, handle)
                batch_ids.clear()
                batch_texts.clear()

            for record in iter_jsonl(args.input):
                if record.doc_id in seen_doc_ids:
                    raise ValueError("input contains a duplicate doc_id")
                seen_doc_ids.add(record.doc_id)
                batch_ids.append(record.doc_id)
                batch_texts.append(record.text)
                if len(batch_texts) >= args.document_batch_size:
                    flush()
            flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, args.output)
    prediction_manifest_sha256 = None
    if args.prediction_manifest is not None:
        config = predictor.model.config
        label2id = dict(getattr(config, "label2id", {}))
        model_identity: dict[str, object] = {
            "model_type": str(config.model_type),
            "config_sha256": sha256_file(args.model / "config.json"),
            "weights_sha256": sha256_file(args.model / "model.safetensors"),
            "label_schema_sha256": canonical_json_hash(label2id),
        }
        architecture_version = getattr(config, "architecture_version", None)
        if architecture_version is not None:
            model_identity["architecture_version"] = str(architecture_version)
        manifest = build_model_prediction_manifest(
            predictions_path=args.output,
            dataset_manifest_path=args.dataset_manifest,
            model_training_manifest_path=args.training_manifest,
            prediction_document_count=count,
            attention_mode=predictor.attention_mode,
            model_identity=model_identity,
        )
        _write_manifest_atomic(manifest, args.prediction_manifest)
        prediction_manifest_sha256 = manifest["manifest_sha256"]
    print(
        json.dumps(
            {
                "documents": count,
                "output": args.output.name,
                "prediction_manifest_sha256": prediction_manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
