"""Character-span evaluation and privacy-safe JSONL I/O."""

from .io import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    Span,
    dumps_prediction,
    load_gold_jsonl,
    load_prediction_jsonl,
    write_prediction_jsonl,
)
from .metrics import evaluate, evaluate_documents, span_iou
from .provenance import (
    add_manifest_hash,
    build_evaluation_provenance,
    build_model_prediction_manifest,
    build_rule_prediction_manifest,
    canonical_json_hash,
    sha256_file,
    verified_manifest_hash,
)

__all__ = [
    "EvaluationDataError",
    "GoldRecord",
    "PredictionRecord",
    "Span",
    "add_manifest_hash",
    "build_evaluation_provenance",
    "build_model_prediction_manifest",
    "build_rule_prediction_manifest",
    "canonical_json_hash",
    "dumps_prediction",
    "evaluate",
    "evaluate_documents",
    "load_gold_jsonl",
    "load_prediction_jsonl",
    "span_iou",
    "sha256_file",
    "verified_manifest_hash",
    "write_prediction_jsonl",
]
