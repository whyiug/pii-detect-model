"""Local, reproducible Qwen3 token-classifier training."""

from pii_zh.training.config import (
    LoraSettings,
    TrainingConfig,
    TrainingConfigError,
    load_training_config,
)
from pii_zh.training.data import (
    DocumentTokenDataset,
    DynamicTokenCollator,
    EncodedDocument,
    GoldSpan,
    JPTTokenCollator,
    TrainingDataError,
    TrainingDataSummary,
    TrainingSourceSummary,
    assert_disjoint_training_splits,
    build_bio_label_maps,
    compute_class_weights,
    compute_document_sampling_weights,
    load_aligned_jsonl,
)
from pii_zh.training.loading import (
    BackboneLoadingAudit,
    CheckpointSafetyError,
    InitializationAudit,
    inspect_local_qwen3_checkpoint,
    load_token_classifier_from_local_causal_lm,
)
from pii_zh.training.manifest import (
    build_training_manifest,
    finalize_training_manifest,
    output_artifact_fingerprint,
    verify_output_artifact_binding,
    verify_training_manifest,
    write_training_manifest,
)
from pii_zh.training.pipeline import TrainingRunResult, run_training

__all__ = [
    "BackboneLoadingAudit",
    "CheckpointSafetyError",
    "DocumentTokenDataset",
    "DynamicTokenCollator",
    "EncodedDocument",
    "GoldSpan",
    "InitializationAudit",
    "JPTTokenCollator",
    "LoraSettings",
    "TrainingConfig",
    "TrainingConfigError",
    "TrainingDataError",
    "TrainingDataSummary",
    "TrainingSourceSummary",
    "TrainingRunResult",
    "build_bio_label_maps",
    "build_training_manifest",
    "finalize_training_manifest",
    "compute_class_weights",
    "compute_document_sampling_weights",
    "assert_disjoint_training_splits",
    "inspect_local_qwen3_checkpoint",
    "load_aligned_jsonl",
    "load_token_classifier_from_local_causal_lm",
    "load_training_config",
    "output_artifact_fingerprint",
    "run_training",
    "verify_training_manifest",
    "verify_output_artifact_binding",
    "write_training_manifest",
]
