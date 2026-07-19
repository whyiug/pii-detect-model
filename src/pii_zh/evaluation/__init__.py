"""Character-span evaluation and privacy-safe JSONL I/O."""

from .baseline_reporting import (
    PUBLIC_SCORE_DECIMALS,
    compact_strict_result,
    pool_strict_results,
    rounded_score,
)
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
    build_system_prediction_manifest,
    canonical_json_hash,
    sha256_file,
    validate_system_prediction_manifest,
    verified_manifest_hash,
)

__all__ = [
    "EvaluationDataError",
    "GoldRecord",
    "PUBLIC_SCORE_DECIMALS",
    "PredictionRecord",
    "Span",
    "add_manifest_hash",
    "build_evaluation_provenance",
    "build_model_prediction_manifest",
    "build_rule_prediction_manifest",
    "build_system_prediction_manifest",
    "canonical_json_hash",
    "compact_strict_result",
    "dumps_prediction",
    "evaluate",
    "evaluate_documents",
    "load_gold_jsonl",
    "load_prediction_jsonl",
    "pool_strict_results",
    "rounded_score",
    "span_iou",
    "sha256_file",
    "validate_system_prediction_manifest",
    "verified_manifest_hash",
    "write_prediction_jsonl",
]
