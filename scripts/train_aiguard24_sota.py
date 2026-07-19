#!/usr/bin/env python3
"""Train a project-native 24-class BIO candidate from the pinned AIguard model.

This runner is intentionally separate from the historical closed-eight pilot.
It never reads a public benchmark and selects checkpoints only with the
declared synthetic validation split.  The saved model is a standalone merged
Qwen3 checkpoint with a modern, content-bound training manifest.  It remains
``release_eligible = false`` until the frozen community evaluation and release
gates promote it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import torch
from transformers import EarlyStoppingCallback, PreTrainedModel, TrainingArguments, set_seed

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.artifact_policy import (  # noqa: E402
    ArtifactPolicyError,
    assert_file_allowed,
)
from pii_zh.data.schema import DataPool, iter_jsonl  # noqa: E402
from pii_zh.models.aiguard24 import (  # noqa: E402
    AIGUARD_SOURCE_MODEL_ID,
    AIGUARD_SOURCE_REVISION,
    initialize_aiguard24,
)
from pii_zh.models.qwen3_bi import Qwen3BiForTokenClassification  # noqa: E402
from pii_zh.taxonomy import load_taxonomy  # noqa: E402
from pii_zh.tokenization import (  # noqa: E402
    assert_character_boundary_tokenizer,
    configure_character_boundary_tokenizer,
    load_serialized_fast_tokenizer,
)
from pii_zh.training.config import LoraSettings  # noqa: E402
from pii_zh.training.data import (  # noqa: E402
    DynamicTokenCollator,
    TrainingDataSummary,
    assert_disjoint_training_splits,
    build_bio_label_maps,
    load_aligned_jsonl,
)
from pii_zh.training.manifest import (  # noqa: E402
    canonical_json_hash,
    output_artifact_fingerprint,
    sha256_file,
    tokenizer_fingerprint,
    training_data_summary_dict,
    write_training_manifest,
)
from pii_zh.training.trainer import (  # noqa: E402
    CharacterValidationComputer,
    SecureTokenTrainer,
    merge_parameter_efficient_model_for_release,
    prepare_parameter_efficient_model,
)

_BLOCKED_EVALUATION_SOURCE_MARKERS = (
    "pii-bench",
    "pii_bench",
    "tw-pii",
    "tw_pii",
    "cluener",
    "chinese_common_ner",
)


class Aiguard24TrainingError(RuntimeError):
    """Raised without echoing source text or local sensitive paths."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-mode",
        choices=("full", "causal"),
        default="full",
        help=(
            "Use release-native bidirectional Qwen3Bi by default. Causal is retained "
            "only for explicitly identified research diagnostics."
        ),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--min-pii-free-ratio", type=float, default=0.40)
    parser.add_argument(
        "--split-isolation-policy",
        choices=("strict", "template-overlap-development-v1"),
        default="strict",
        help=(
            "Keep strict group isolation by default. The explicit development policy "
            "permits only template_group overlap while still requiring document, "
            "entity-value, and source-group disjointness."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        "max_length",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "lora_rank",
        "lora_alpha",
        "early_stopping_patience",
        "save_total_limit",
    )
    if any(
        isinstance(getattr(args, name), bool)
        or not isinstance(getattr(args, name), int)
        or getattr(args, name) < 1
        for name in positive_integers
    ):
        raise Aiguard24TrainingError("integer hyperparameters must be positive")
    if args.max_length < 8:
        raise Aiguard24TrainingError("max_length must be at least eight")
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed < 0
        or isinstance(args.max_steps, bool)
        or not isinstance(args.max_steps, int)
        or args.max_steps == 0
        or args.max_steps < -1
    ):
        raise Aiguard24TrainingError("seed/max_steps are invalid")
    if (
        not isinstance(args.epochs, (int, float))
        or isinstance(args.epochs, bool)
        or args.epochs <= 0
    ):
        raise Aiguard24TrainingError("epochs must be positive")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        for value in (args.learning_rate, args.classifier_learning_rate)
    ):
        raise Aiguard24TrainingError("learning rates must be positive")
    if (
        not isinstance(args.weight_decay, (int, float))
        or args.weight_decay < 0
        or not isinstance(args.warmup_ratio, (int, float))
        or not 0 <= args.warmup_ratio < 1
        or not isinstance(args.lora_dropout, (int, float))
        or not 0 <= args.lora_dropout < 1
        or not isinstance(args.min_pii_free_ratio, (int, float))
        or not 0.30 <= args.min_pii_free_ratio < 1
    ):
        raise Aiguard24TrainingError("ratio/dropout/decay hyperparameters are invalid")


def _require_isolated_cuda(device: str) -> torch.device:
    selected = torch.device(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected != torch.device("cuda:0") or re.fullmatch(r"\d+", visible) is None:
        raise Aiguard24TrainingError(
            "set exactly one numeric CUDA_VISIBLE_DEVICES id and use --device cuda:0"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Aiguard24TrainingError("the isolated CUDA device is unavailable")
    return selected


def _assert_split_isolation(
    train_dataset: Any,
    validation_dataset: Any,
    *,
    policy: str,
) -> dict[str, Any]:
    """Apply the frozen split policy without exposing group values."""

    if policy == "strict":
        assert_disjoint_training_splits(train_dataset, validation_dataset)
        return {
            "policy": "strict",
            "template_group_overlap_allowed": False,
            "collision_counts": {
                "document_id": 0,
                "template_group": 0,
                "entity_value_group": 0,
                "source_group": 0,
            },
            "release_test_eligible": False,
        }
    if policy != "template-overlap-development-v1":
        raise Aiguard24TrainingError("unknown split isolation policy")
    collision_counts = {
        group_type: len(
            train_dataset.leakage_fingerprints.get(group_type, frozenset())
            & validation_dataset.leakage_fingerprints.get(group_type, frozenset())
        )
        for group_type in (
            "document_id",
            "template_group",
            "entity_value_group",
            "source_group",
        )
    }
    forbidden = {
        name: count
        for name, count in collision_counts.items()
        if name != "template_group" and count
    }
    if forbidden:
        raise Aiguard24TrainingError(
            "development split violates identity/value/source isolation"
        )
    if collision_counts["template_group"] < 1:
        raise Aiguard24TrainingError(
            "template-overlap policy was selected but no overlap was observed"
        )
    return {
        "policy": policy,
        "template_group_overlap_allowed": True,
        "collision_counts": collision_counts,
        "release_test_eligible": False,
        "limitation": "validation_informed_template_family_overlap",
    }


def _safe_package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def _load_tokenizer(source_model: Path) -> Any:
    tokenizer = load_serialized_fast_tokenizer(source_model)
    configure_character_boundary_tokenizer(tokenizer)
    assert_character_boundary_tokenizer(tokenizer)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise Aiguard24TrainingError("source tokenizer lacks pad and EOS tokens")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _assert_synthetic_open_contract(path: Path, *, expected_split: str) -> None:
    count = 0
    for record in iter_jsonl(path):
        count += 1
        source_marker = record.provenance.source_id.casefold()
        generator_marker = (record.provenance.generator_name or "").casefold()
        if any(
            marker in source_marker or marker in generator_marker
            for marker in _BLOCKED_EVALUATION_SOURCE_MARKERS
        ):
            raise Aiguard24TrainingError("an evaluation/quarantined source entered training")
        if (
            record.split != expected_split
            or record.data_pool is not DataPool.PUBLIC_RELEASE
            or record.provenance.synthetic is not True
            or record.provenance.evaluation_only is True
            or record.public_weight_training_allowed is not True
            or record.quality.quality_gate is not True
            or record.quality.validators_passed is not True
            or record.provenance.license.upper() != "APACHE-2.0"
        ):
            raise Aiguard24TrainingError("a record violates the synthetic/open training contract")
    if count == 0:
        raise Aiguard24TrainingError("training input is empty")


def _assert_training_artifact_allowed(path: Path) -> None:
    try:
        assert_file_allowed(path, purpose="training")
    except ArtifactPolicyError:
        raise Aiguard24TrainingError("a training input is denied by artifact policy") from None


def _assert_summary_contract(
    summary: TrainingDataSummary,
    *,
    expected_labels: set[str],
    min_pii_free_ratio: float,
) -> None:
    if set(summary.label_counts) != expected_labels:
        raise Aiguard24TrainingError("a training split does not cover all 24 core labels")
    if summary.pii_free_document_count / summary.document_count < min_pii_free_ratio:
        raise Aiguard24TrainingError("a training split is below the PII-free document floor")
    if (
        summary.quality_gate_passed_document_count != summary.document_count
        or summary.validators_passed_document_count != summary.document_count
        or summary.public_weight_training_allowed_document_count != summary.document_count
        or any(
            "synthetic" not in source.source_kind.casefold()
            or source.data_pool != DataPool.PUBLIC_RELEASE.value
            or source.license.upper() != "APACHE-2.0"
            or source.document_count != source.quality_gate_passed_document_count
            or source.document_count != source.validators_passed_document_count
            or source.document_count != source.public_weight_training_allowed_document_count
            for source in summary.sources
        )
    ):
        raise Aiguard24TrainingError("data summary violates quality or release routing")


def _seal_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if "manifest_sha256" in value:
        raise Aiguard24TrainingError("manifest is already sealed")
    result = dict(value)
    result["manifest_sha256"] = canonical_json_hash(result)
    return result


def _runtime_environment() -> dict[str, Any]:
    return {
        "execution_device": "cuda",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "visible_device_count": int(torch.cuda.device_count()),
        "device_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _convert_attention_mode(
    model: PreTrainedModel,
    *,
    attention_mode: str,
) -> tuple[PreTrainedModel, dict[str, Any]]:
    """Convert the initialized AIguard classifier to the requested attention mode.

    Qwen3Bi deliberately keeps the upstream ``model.*``/``score.*`` state-dict
    schema.  The strict conversion therefore changes only the attention-mask
    contract; it does not discard or randomly reinitialize a learned tensor.
    """

    if attention_mode == "causal":
        return model, {
            "source_attention_mode": "causal",
            "target_attention_mode": "causal",
            "conversion": "identity",
            "strict_state_dict_load": True,
            "state_dict_key_count": len(model.state_dict()),
        }
    if attention_mode != "full":
        raise Aiguard24TrainingError("attention mode must be full or causal")

    source_state = model.state_dict()
    source_keys = tuple(source_state)
    converted = Qwen3BiForTokenClassification.from_qwen3_token_classifier_state_dict(
        model.config,
        source_state,
        strict=True,
        bi_attention_backend="sdpa",
    )
    target_keys = tuple(converted.state_dict())
    if target_keys != source_keys:
        raise Aiguard24TrainingError(
            "bidirectional conversion changed the initialized state-dict schema"
        )
    if (
        converted.config.model_type != "qwen3_bi"
        or converted.config.pii_attention_mode != "full"
        or converted.config.use_cache is not False
    ):
        raise Aiguard24TrainingError("bidirectional conversion contract is incomplete")
    return cast(PreTrainedModel, converted), {
        "source_attention_mode": "causal",
        "target_attention_mode": "full",
        "conversion": "qwen3_to_qwen3_bi_shared_state_dict_v1",
        "attention_backend": "sdpa",
        "strict_state_dict_load": True,
        "state_dict_key_count": len(source_keys),
        "newly_initialized_parameter_keys": [],
        "discarded_parameter_keys": [],
    }


def _training_source_ids(*summaries: TrainingDataSummary) -> list[str]:
    return sorted(
        {
            AIGUARD_SOURCE_MODEL_ID,
            *(source.source_id for summary in summaries for source in summary.sources),
        }
    )


def _checkpoint_selection_receipt(
    *,
    trainer_state_root: Path,
    best_model_checkpoint: str | None,
    best_global_step: int | None,
    best_metric: float | None,
    final_eval_metric: float | None,
) -> dict[str, Any]:
    """Bind the merged candidate to Trainer's validation-selected LoRA checkpoint."""

    if not isinstance(best_model_checkpoint, str) or not best_model_checkpoint:
        raise Aiguard24TrainingError("Trainer did not select a best checkpoint")
    if (
        isinstance(best_global_step, bool)
        or not isinstance(best_global_step, int)
        or best_global_step < 1
    ):
        raise Aiguard24TrainingError("Trainer best checkpoint step is invalid")
    if (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(float(best_metric))
        or isinstance(final_eval_metric, bool)
        or not isinstance(final_eval_metric, (int, float))
        or not math.isfinite(float(final_eval_metric))
    ):
        raise Aiguard24TrainingError("checkpoint selection metrics are invalid")

    root = trainer_state_root.resolve(strict=True)
    checkpoint = Path(best_model_checkpoint)
    if checkpoint.is_symlink():
        raise Aiguard24TrainingError("selected checkpoint must not be a symlink")
    checkpoint = checkpoint.resolve(strict=True)
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if checkpoint.parent != root or match is None:
        raise Aiguard24TrainingError("selected checkpoint escaped the dedicated trainer state")
    checkpoint_step = int(match.group(1))
    if checkpoint_step != best_global_step:
        raise Aiguard24TrainingError("selected checkpoint id and best step disagree")
    adapter_weights = checkpoint / "adapter_model.safetensors"
    if adapter_weights.is_symlink() or not adapter_weights.is_file():
        raise Aiguard24TrainingError("selected checkpoint lacks regular LoRA safetensors")
    if not math.isclose(
        float(best_metric),
        float(final_eval_metric),
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        raise Aiguard24TrainingError(
            "selected checkpoint metric disagrees with final validation replay"
        )
    return {
        "selection_split": "validation",
        "selection_metric": "risk_weighted_score",
        "selected_checkpoint_id": checkpoint.name,
        "selected_global_step": checkpoint_step,
        "best_metric": float(best_metric),
        "final_validation_replay_metric": float(final_eval_metric),
        "adapter_model_sha256": sha256_file(adapter_weights),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    _require_isolated_cuda(args.device)
    source_model = args.source_model.expanduser().resolve(strict=True)
    train_path = args.train.expanduser().resolve(strict=True)
    validation_path = args.validation.expanduser().resolve(strict=True)
    if train_path == validation_path:
        raise Aiguard24TrainingError("train and validation must be distinct files")
    output = args.output.expanduser().resolve()
    trainer_state = output.parent / f"{output.name}.trainer-state"
    if (
        output.exists()
        or output.is_symlink()
        or trainer_state.exists()
        or trainer_state.is_symlink()
    ):
        raise Aiguard24TrainingError("output and trainer-state paths must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    train_hash = sha256_file(train_path)
    validation_hash = sha256_file(validation_path)
    _assert_training_artifact_allowed(train_path)
    _assert_training_artifact_allowed(validation_path)
    _assert_synthetic_open_contract(train_path, expected_split="train")
    _assert_synthetic_open_contract(validation_path, expected_split="validation")
    set_seed(args.seed)
    taxonomy = load_taxonomy()
    label2id, id2label = build_bio_label_maps(taxonomy)
    expected_labels = {entity.name for entity in taxonomy.label_sets["core"]}
    tokenizer = _load_tokenizer(source_model)
    train_dataset, train_summary = load_aligned_jsonl(
        train_path,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        expected_split="train",
        allowed_pools=frozenset({DataPool.PUBLIC_RELEASE}),
    )
    validation_dataset, validation_summary = load_aligned_jsonl(
        validation_path,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        expected_split="validation",
        allowed_pools=frozenset({DataPool.PUBLIC_RELEASE}),
    )
    split_isolation = _assert_split_isolation(
        train_dataset,
        validation_dataset,
        policy=args.split_isolation_policy,
    )
    _assert_summary_contract(
        train_summary,
        expected_labels=expected_labels,
        min_pii_free_ratio=args.min_pii_free_ratio,
    )
    _assert_summary_contract(
        validation_summary,
        expected_labels=expected_labels,
        min_pii_free_ratio=args.min_pii_free_ratio,
    )
    if sha256_file(train_path) != train_hash or sha256_file(validation_path) != validation_hash:
        raise Aiguard24TrainingError("a dataset changed while it was being loaded")

    initialization = initialize_aiguard24(source_model, taxonomy=taxonomy, seed=args.seed)
    model = initialization.model
    if (
        model.config.label2id != label2id
        or {int(key): value for key, value in model.config.id2label.items()} != id2label
    ):
        raise Aiguard24TrainingError("initialized model label schema disagrees with the dataset")
    model, attention_conversion = _convert_attention_mode(
        model,
        attention_mode=args.attention_mode,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    lora = LoraSettings(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=float(args.lora_dropout),
    )
    model = cast(
        PreTrainedModel,
        prepare_parameter_efficient_model(model, fine_tuning="lora", lora=lora),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    gc.collect()

    label_to_risk_tiers = {
        entity.name: entity.risk_tiers
        for entity in taxonomy.entities
        if entity.name in expected_labels
    }
    metrics_computer = CharacterValidationComputer(
        validation_dataset,
        id2label=id2label,
        label_to_risk_tiers=label_to_risk_tiers,
        attention_mode=args.attention_mode,
    )
    step_limited = args.max_steps > 0
    strategy = "steps" if step_limited else "epoch"
    training_args = TrainingArguments(
        output_dir=str(trainer_state),
        do_train=True,
        do_eval=True,
        eval_strategy=strategy,
        save_strategy=strategy,
        eval_steps=args.max_steps if step_limited else None,
        save_steps=args.max_steps if step_limited else 500,
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="risk_weighted_score",
        greater_is_better=True,
        log_level="warning",
    )
    trainer = SecureTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DynamicTokenCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            pad_to_multiple_of=8,
        ),
        compute_metrics=metrics_computer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        class_weights=None,
        document_sampling_weights=None,
        sampler_seed=args.seed,
        classifier_learning_rate=args.classifier_learning_rate,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    recipe = {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "class_weighting": False,
        "document_sampling": False,
        "gradient_checkpointing": True,
        "bf16": True,
        "attention_mode": args.attention_mode,
        "initialization_strategy": "aiguard24_projection_then_attention_conversion_v1",
        "fine_tuning": "lora_merged",
        "lora": {
            "rank": lora.rank,
            "alpha": lora.alpha,
            "dropout": lora.dropout,
            "target_modules": list(lora.target_modules),
        },
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_metric": "risk_weighted_score",
        "min_pii_free_ratio": args.min_pii_free_ratio,
        "split_isolation_policy": args.split_isolation_policy,
    }
    manifest_base: dict[str, Any] = {
        "schema_version": 4,
        "manifest_type": "aiguard24_community_training_v1",
        "status": "in_progress",
        "release_eligible": False,
        "created_at": created_at,
        "seed": args.seed,
        "attention_mode": args.attention_mode,
        "fine_tuning": "lora",
        "base_source_id": AIGUARD_SOURCE_MODEL_ID,
        "training_source_ids": _training_source_ids(train_summary, validation_summary),
        "source_revision": AIGUARD_SOURCE_REVISION,
        "recipe": recipe,
        "recipe_sha256": canonical_json_hash(recipe),
        "taxonomy_version": taxonomy.taxonomy_version,
        "label2id": dict(sorted(label2id.items(), key=lambda item: item[1])),
        "label_schema_sha256": canonical_json_hash(label2id),
        "initialization": {
            **initialization.audit.to_dict(),
            "attention_conversion": attention_conversion,
        },
        "tokenizer": tokenizer_fingerprint(source_model, tokenizer=tokenizer),
        "datasets": {
            "train": {
                "sha256": train_hash,
                "summary": training_data_summary_dict(train_summary),
            },
            "validation": {
                "sha256": validation_hash,
                "summary": training_data_summary_dict(validation_summary),
            },
        },
        "benchmark_isolation": {
            "dataset_id": "wan9yu/pii-bench-zh",
            "read_for_training": False,
            "read_for_validation": False,
            "read_for_checkpoint_selection": False,
            "read_for_threshold_tuning": False,
        },
        "split_isolation": split_isolation,
        "versions": {
            "python": platform.python_version(),
            "torch": _safe_package_version("torch"),
            "transformers": _safe_package_version("transformers"),
            "accelerate": _safe_package_version("accelerate"),
            "peft": _safe_package_version("peft"),
            "pii_zh_qwen": _safe_package_version("pii-zh-qwen"),
        },
        "code_revision": _git_revision(),
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "synthetic_only": True,
            "trainer_reporting_integrations": [],
        },
    }
    output.mkdir(mode=0o700)
    write_training_manifest(_seal_manifest(manifest_base), output)

    train_output = trainer.train()
    eval_output = trainer.evaluate(metric_key_prefix="eval")
    numeric_train = {
        key: float(value)
        for key, value in train_output.metrics.items()
        if isinstance(value, (int, float))
    }
    numeric_eval = {
        key: float(value) for key, value in eval_output.items() if isinstance(value, (int, float))
    }
    checkpoint_selection = _checkpoint_selection_receipt(
        trainer_state_root=trainer_state,
        best_model_checkpoint=trainer.state.best_model_checkpoint,
        best_global_step=getattr(trainer.state, "best_global_step", None),
        best_metric=trainer.state.best_metric,
        final_eval_metric=numeric_eval.get("eval_risk_weighted_score"),
    )
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    final_model = merge_parameter_efficient_model_for_release(
        unwrapped,
        fine_tuning="lora",
        attention_mode=args.attention_mode,
    )
    final_model.config.pii_training_status = "completed_candidate_not_benchmark_evaluated"
    final_model.config.pii_taxonomy_version = taxonomy.taxonomy_version
    final_model.config.pii_release_eligible = False
    final_model.config._name_or_path = ""
    final_model.save_pretrained(output, safe_serialization=True)
    tokenizer.name_or_path = ""
    if isinstance(getattr(tokenizer, "init_kwargs", None), dict):
        tokenizer.init_kwargs["name_or_path"] = ""
    tokenizer.save_pretrained(output)
    if any(output.glob("adapter_*")) or not (output / "model.safetensors").is_file():
        raise Aiguard24TrainingError("final output is not a standalone merged safetensors model")
    assert_character_boundary_tokenizer(
        load_serialized_fast_tokenizer(output, require_boundary=True)
    )
    metrics_path = output / "final_eval_metrics.json"
    metrics_path.write_text(
        json.dumps(numeric_eval, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer.save_metrics("train", numeric_train)
    trainer.save_metrics("eval", numeric_eval)
    trainer.save_state()
    if sha256_file(train_path) != train_hash or sha256_file(validation_path) != validation_hash:
        raise Aiguard24TrainingError("a dataset changed during training")

    finalized = dict(manifest_base)
    finalized.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "fine_tuning": "lora_merged",
            "runtime_environment": _runtime_environment(),
            "checkpoint_selection": checkpoint_selection,
            "validation": numeric_eval,
            "output_artifact": output_artifact_fingerprint(output),
        }
    )
    manifest_path = write_training_manifest(_seal_manifest(finalized), output)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": output.name,
                "manifest": manifest_path.name,
                "manifest_sha256": json.loads(manifest_path.read_text())["manifest_sha256"],
                "eval_strict_micro_f1": numeric_eval.get("eval_strict_micro_f1"),
                "eval_strict_macro_f1": numeric_eval.get("eval_strict_macro_f1"),
                "eval_risk_weighted_score": numeric_eval.get("eval_risk_weighted_score"),
                "peak_memory_bytes": finalized["runtime_environment"]["peak_memory_bytes"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Aiguard24TrainingError, OSError, RuntimeError, ValueError, ImportError) as exc:
        raise SystemExit(f"training failed: {exc}") from None
