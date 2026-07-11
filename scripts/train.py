#!/usr/bin/env python3
"""Train a Chinese PII token classifier from local, approved inputs only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.training import (  # noqa: E402
    TrainingConfig,
    load_training_config,
    run_training,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-only Qwen3 PII token-classification training"
    )
    parser.add_argument("--config", type=Path, help="Strict training YAML")
    parser.add_argument("--base-model", help="Local Qwen3 Base CausalLM checkpoint")
    parser.add_argument("--base-source-id", help="Registered immutable base-model source ID")
    parser.add_argument("--train-file", help="Canonical training JSONL")
    parser.add_argument("--validation-file", help="Canonical validation JSONL")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--initial-model",
        help="Completed, bound causal/JPT standalone checkpoint used only to initialize Full",
    )
    parser.add_argument("--taxonomy-path")
    parser.add_argument("--attention-mode", choices=("causal", "full", "jpt"))
    parser.add_argument("--fine-tuning", choices=("full", "lora"))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--classifier-learning-rate", type=float)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-cpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--document-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume the latest checkpoint, or provide a local checkpoint path",
    )
    return parser


def _overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "base_model": arguments.base_model,
            "base_source_id": arguments.base_source_id,
            "train_file": arguments.train_file,
            "validation_file": arguments.validation_file,
            "output_dir": arguments.output_dir,
            "initial_model": arguments.initial_model,
            "taxonomy_path": arguments.taxonomy_path,
            "attention_mode": arguments.attention_mode,
            "fine_tuning": arguments.fine_tuning,
            "max_length": arguments.max_length,
            "per_device_train_batch_size": arguments.per_device_train_batch_size,
            "per_device_eval_batch_size": arguments.per_device_eval_batch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "learning_rate": arguments.learning_rate,
            "classifier_learning_rate": arguments.classifier_learning_rate,
            "epochs": arguments.epochs,
            "seed": arguments.seed,
            "bf16": arguments.bf16,
            "use_cpu": arguments.use_cpu,
            "gradient_checkpointing": arguments.gradient_checkpointing,
            "class_weighting": arguments.class_weighting,
            "document_sampling": arguments.document_sampling,
            "resume": arguments.resume,
        }.items()
        if value is not None
    }


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    values = _overrides(arguments)
    try:
        if arguments.config is not None:
            config = load_training_config(arguments.config, overrides=values)
        else:
            config = TrainingConfig.from_mapping(values)
        result = run_training(config)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        # Training/data exceptions are intentionally sanitized before reaching
        # this boundary; never print a record, prompt, or source line here.
        parser.exit(2, f"training failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": result.manifest_path.name,
                "output_artifact": result.output_dir.name,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
