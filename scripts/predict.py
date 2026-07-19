#!/usr/bin/env python3
"""Generate raw-text-free character-span predictions from a local Qwen3Bi model."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--stride-fraction", type=float, default=0.25)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--allowed-label",
        action="append",
        dest="allowed_labels",
        help=(
            "Restrict logits before softmax/argmax to O plus this model entity label. "
            "Repeat for protocol-restricted evaluation."
        ),
    )
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--training-manifest",
        "--model-training-manifest",
        dest="training_manifest",
        type=Path,
    )
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument(
        "--evidence-class",
        choices=("development", "test_informed_posthoc"),
        help="Optional evidence limitation bound into the prediction manifest",
    )
    return parser


def _write_manifest_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite a prediction manifest")
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
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination.chmod(0o444)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _assert_new_outputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    outputs = [args.output]
    if args.prediction_manifest is not None:
        outputs.append(args.prediction_manifest)
    resolved_outputs = [_resolved(path) for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("prediction outputs must be distinct files")
    for path in outputs:
        if path.exists() or path.is_symlink():
            parser.error("refusing to overwrite an existing prediction output")
    protected_files = [args.input]
    protected_files.extend(
        path
        for path in (args.dataset_manifest, args.training_manifest, args.calibration)
        if path is not None
    )
    protected = {_resolved(path) for path in protected_files}
    model_root = _resolved(args.model)
    for output in resolved_outputs:
        if output in protected:
            parser.error("prediction output collides with an input artifact")
        if output == model_root or model_root in output.parents:
            parser.error("prediction output must not be written inside the model artifact")


def _inference_implementation() -> dict[str, str]:
    recognizer_sha256 = sha256_file(
        _REPOSITORY_ROOT / "src/pii_zh/presidio/qwen_recognizer.py"
    )
    return {
        "bio_decoder": sha256_file(_REPOSITORY_ROOT / "src/pii_zh/models/decoding.py"),
        "cross_window_deduplication": recognizer_sha256,
        "predict_cli": sha256_file(Path(__file__)),
        "qwen_recognizer": recognizer_sha256,
        "token_chunker": sha256_file(
            _REPOSITORY_ROOT / "src/pii_zh/presidio/token_chunker.py"
        ),
        "transformers_predictor": sha256_file(
            _REPOSITORY_ROOT / "src/pii_zh/inference/transformers_predictor.py"
        ),
    }


def _evidence_policy(evidence_class: str) -> dict[str, object]:
    is_test_informed = evidence_class == "test_informed_posthoc"
    return {
        "evidence_class": evidence_class,
        "test_informed": is_test_informed,
        "posthoc": is_test_informed,
        "confirmatory": False,
        "supports_release_claim": False,
    }


def _bind_evidence_manifest(
    manifest: dict[str, object],
    *,
    inference_configuration: dict[str, object],
    inference_implementation: dict[str, str],
    evidence_policy: dict[str, object],
) -> dict[str, object]:
    """Extend a model manifest without changing the frozen provenance module."""

    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    unsigned.pop("prediction_id", None)
    model_identity = unsigned.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("model prediction manifest lacks model identity")
    model_identity_sha256 = model_identity.get("identity_sha256")
    if not isinstance(model_identity_sha256, str):
        raise ValueError("model prediction manifest lacks model identity hash")
    attention_mode = unsigned.get("attention_mode")
    if not isinstance(attention_mode, str):
        raise ValueError("model prediction manifest lacks attention mode")
    unsigned["inference_configuration"] = {
        **inference_configuration,
        "attention_mode": attention_mode,
        "model_identity_sha256": model_identity_sha256,
    }
    unsigned["inference_implementation"] = dict(inference_implementation)
    unsigned["evidence_policy"] = dict(evidence_policy)
    unsigned["prediction_id"] = "model-" + canonical_json_hash(unsigned)[:24]
    unsigned["manifest_sha256"] = canonical_json_hash(unsigned)
    return unsigned


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.document_batch_size < 1 or args.window_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.max_tokens < 4:
        parser.error("--max-tokens must be at least four")
    if not math.isfinite(args.stride_fraction) or not 0.20 <= args.stride_fraction <= 0.25:
        parser.error("--stride-fraction must be between 0.20 and 0.25")
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between zero and one")
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
    if args.evidence_class is not None and args.prediction_manifest is None:
        parser.error("--evidence-class requires the complete prediction manifest chain")
    if args.prediction_manifest is not None and (
        args.prediction_manifest.resolve() == args.output.resolve()
    ):
        parser.error("--prediction-manifest and --output must be different files")
    _assert_new_outputs(args, parser)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    taxonomy = load_taxonomy()
    if args.allowed_labels is None:
        output_entities = sorted(taxonomy.output_label_names)
    else:
        output_entities = sorted(set(args.allowed_labels))
        if (
            len(output_entities) != len(args.allowed_labels)
            or not output_entities
            or not set(output_entities) <= taxonomy.core_label_names
        ):
            parser.error("--allowed-label values must be unique core taxonomy labels")
    ignored = [entity.name for entity in taxonomy.label_sets["auxiliary"]]
    predictor = load_local_predictor(
        args.model,
        device=args.device,
        dtype=dtype,
        micro_batch_size=args.window_batch_size,
        ignored_labels=ignored,
        allowed_labels=output_entities,
    )
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        label_mapping={label: label for label in output_entities},
        ignored_labels=ignored,
        calibration=args.calibration,
        default_threshold=args.threshold,
        max_tokens=args.max_tokens,
        stride_fraction=args.stride_fraction,
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
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    args.output.chmod(0o444)
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
        if args.evidence_class is not None:
            manifest = _bind_evidence_manifest(
                manifest,
                inference_configuration={
                    "allowed_labels": output_entities,
                    "calibration": (
                        {"mode": "none"}
                        if args.calibration is None
                        else {
                            "mode": "bundle",
                            "bundle_sha256": sha256_file(args.calibration),
                        }
                    ),
                    "document_batch_size": args.document_batch_size,
                    "max_tokens": args.max_tokens,
                    "stride_fraction": args.stride_fraction,
                    "threshold": args.threshold,
                    "window_batch_size": args.window_batch_size,
                },
                inference_implementation=_inference_implementation(),
                evidence_policy=_evidence_policy(args.evidence_class),
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
