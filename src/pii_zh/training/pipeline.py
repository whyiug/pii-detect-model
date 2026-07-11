"""End-to-end local training pipeline; no remote model or data access."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import (
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "Training pipeline dependencies are missing; install the training extra."
    ) from exc

from pii_zh.taxonomy import load_taxonomy
from pii_zh.tokenization import (
    configure_character_boundary_tokenizer,
    load_serialized_fast_tokenizer,
)
from pii_zh.training.config import TrainingConfig
from pii_zh.training.data import (
    DynamicTokenCollator,
    JPTTokenCollator,
    assert_disjoint_training_splits,
    build_bio_label_maps,
    compute_class_weights,
    compute_document_sampling_weights,
    load_aligned_jsonl,
)
from pii_zh.training.initialization import initialize_full_attention_from_verified_model
from pii_zh.training.loading import load_token_classifier_from_local_causal_lm
from pii_zh.training.manifest import (
    build_training_manifest,
    finalize_training_manifest,
    write_training_manifest,
)
from pii_zh.training.trainer import (
    CharacterValidationComputer,
    SecureTokenTrainer,
    merge_parameter_efficient_model_for_release,
    prepare_parameter_efficient_model,
)


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    output_dir: Path
    manifest_path: Path
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float]


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute()
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _assert_standalone_model_root(output_dir: Path) -> None:
    if not (output_dir / "model.safetensors").is_file():
        raise RuntimeError("final model root is missing model.safetensors")
    adapter_artifacts = sorted(path.name for path in output_dir.glob("adapter_*") if path.is_file())
    if adapter_artifacts:
        raise RuntimeError("final model root contains adapter-only artifacts")
    for filename in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = output_dir / filename
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("final model metadata is not valid local JSON") from exc
        if _contains_absolute_path(document):
            raise RuntimeError("final model metadata contains an absolute local path")


def _runtime_environment(config: TrainingConfig) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    visible_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    execution_device = "cuda" if cuda_available and not config.use_cpu else "cpu"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible_marker = "unset"
    elif re.fullmatch(r"(?:-1|\d+(?:,\d+)*)", visible.strip()):
        visible_marker = visible.strip()
    else:
        visible_marker = "non_numeric_selection"

    device_name: str | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    selected_device_index: int | None = None
    if execution_device == "cuda":
        selected_device_index = int(torch.cuda.current_device())
        device_name = str(torch.cuda.get_device_name(selected_device_index))
        major, minor = torch.cuda.get_device_capability(selected_device_index)
        compute_capability = f"{major}.{minor}"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            candidate = result.stdout.splitlines()[0].strip()
            if re.fullmatch(r"\d+(?:\.\d+)+", candidate):
                driver_version = candidate
        except (FileNotFoundError, IndexError, subprocess.SubprocessError):
            driver_version = None
    return {
        "execution_device": execution_device,
        "cuda_available": cuda_available,
        "visible_device_count": visible_device_count,
        "cuda_visible_devices": visible_marker,
        "selected_device_index": selected_device_index,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver_version,
        "bf16_enabled": bool(config.bf16 and execution_device == "cuda"),
    }


def _safe_tokenizer(checkpoint: str) -> Any:
    path = Path(checkpoint).expanduser().resolve(strict=True)
    tokenizer = load_serialized_fast_tokenizer(path)
    if not tokenizer.is_fast:
        raise RuntimeError("Qwen3 training requires a fast tokenizer with offset mappings")
    configure_character_boundary_tokenizer(tokenizer)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_training(config: TrainingConfig) -> TrainingRunResult:
    """Train and save a token classifier from local, content-bound inputs."""

    config.validate()
    set_seed(config.seed)
    taxonomy = load_taxonomy(config.taxonomy_path)
    label2id, id2label = build_bio_label_maps(
        taxonomy,
        include_extended=config.include_extended_labels,
        include_auxiliary=config.include_auxiliary_labels,
    )
    tokenizer = _safe_tokenizer(config.base_model)

    source_max_length = (
        (config.max_length - 1) // 2 if config.attention_mode == "jpt" else config.max_length
    )
    train_dataset, train_summary = load_aligned_jsonl(
        config.train_file,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=source_max_length,
        expected_split="train",
    )
    validation_dataset, validation_summary = load_aligned_jsonl(
        config.validation_file,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=source_max_length,
        expected_split="validation",
    )
    assert_disjoint_training_splits(train_dataset, validation_dataset)
    if train_summary.unalignable_boundary_count or validation_summary.unalignable_boundary_count:
        raise RuntimeError(
            "boundary-safe tokenizer invariant failed: training/validation contains "
            "unalignable character spans"
        )

    dtype = torch.bfloat16 if config.bf16 and not config.use_cpu else None
    model, loading_audit = load_token_classifier_from_local_causal_lm(
        config.base_model,
        attention_mode=config.attention_mode,
        label2id=label2id,
        id2label=id2label,
        classifier_dropout=config.classifier_dropout,
        dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    initialization_audit = None
    if config.initial_model is not None:
        initialization_audit = initialize_full_attention_from_verified_model(
            model,
            config=config,
            base_loading_audit=loading_audit,
            taxonomy_version=taxonomy.taxonomy_version,
            label2id=label2id,
            id2label=id2label,
            tokenizer=tokenizer,
            train_summary=train_summary,
            validation_summary=validation_summary,
        )
    model = prepare_parameter_efficient_model(
        model, fine_tuning=config.fine_tuning, lora=config.lora
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.fine_tuning == "lora" and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    separator: int | None = None
    if config.attention_mode == "jpt":
        separator = config.jpt_sep_token_id
        if separator is None:
            separator = tokenizer.eos_token_id
        if separator is None:
            raise RuntimeError("JPT requires an existing separator/EOS token id")
        collator = JPTTokenCollator(
            pad_token_id=tokenizer.pad_token_id,
            sep_token_id=separator,
            max_length=config.max_length,
        )
    else:
        pad_multiple = 8 if config.max_length % 8 == 0 else None
        collator = DynamicTokenCollator(
            pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=pad_multiple,
        )

    class_weights = None
    if config.class_weighting:
        class_weights = compute_class_weights(
            [document.labels for document in train_dataset.documents],
            num_labels=len(label2id),
            cap=config.class_weight_cap,
        )
    sampling_weights = None
    if config.document_sampling:
        sampling_weights = compute_document_sampling_weights(
            train_dataset.documents, cap=config.document_sampling_cap
        )

    selected_names = {
        label.removeprefix("B-").removeprefix("I-") for label in label2id if label != "O"
    }
    label_to_risk_tiers = {
        entity.name: entity.risk_tiers
        for entity in taxonomy.entities
        if entity.name in selected_names
    }
    metrics_computer = CharacterValidationComputer(
        validation_dataset,
        id2label=id2label,
        label_to_risk_tiers=label_to_risk_tiers,
        attention_mode=config.attention_mode,
    )

    output_dir = Path(config.output_dir)
    trainer_state_dir = output_dir.parent / f"{output_dir.name}.trainer-state"
    training_args = TrainingArguments(
        output_dir=str(trainer_state_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.epochs,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=config.bf16 and not config.use_cpu,
        use_cpu=config.use_cpu,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=True,
        save_total_limit=config.save_total_limit,
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
        data_collator=collator,
        compute_metrics=metrics_computer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        class_weights=class_weights,
        document_sampling_weights=sampling_weights,
        sampler_seed=config.seed,
        classifier_learning_rate=config.effective_classifier_learning_rate,
    )

    repository_root = Path(__file__).resolve().parents[3]
    manifest = build_training_manifest(
        config=config,
        loading_audit=loading_audit,
        train_summary=train_summary,
        validation_summary=validation_summary,
        taxonomy_version=taxonomy.taxonomy_version,
        label2id=label2id,
        tokenizer=tokenizer,
        initialization_audit=initialization_audit,
        worktree=repository_root,
    )
    manifest_path = write_training_manifest(manifest, output_dir)

    resume: bool | str = config.resume
    if isinstance(resume, str):
        resume = str(Path(resume).expanduser().resolve(strict=True))
    train_output = trainer.train(resume_from_checkpoint=resume)
    eval_output = trainer.evaluate(metric_key_prefix="eval")
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    final_model = merge_parameter_efficient_model_for_release(
        unwrapped_model,
        fine_tuning=config.fine_tuning,
        attention_mode=config.attention_mode,
        jpt_sep_token_id=separator,
    )
    for adapter_name in ("adapter_config.json", "adapter_model.safetensors"):
        adapter_path = output_dir / adapter_name
        if adapter_path.exists():
            adapter_path.unlink()
    final_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.name_or_path = ""
    if isinstance(getattr(tokenizer, "init_kwargs", None), dict):
        tokenizer.init_kwargs["name_or_path"] = ""
    tokenizer.save_pretrained(output_dir)
    _assert_standalone_model_root(output_dir)
    trainer.save_metrics("train", train_output.metrics)
    trainer.save_metrics("eval", eval_output)
    trainer.save_state()
    numeric_metrics = {
        key: float(value)
        for key, value in train_output.metrics.items()
        if isinstance(value, (int, float))
    }
    numeric_eval_metrics = {
        key: float(value) for key, value in eval_output.items() if isinstance(value, (int, float))
    }
    metrics_path = output_dir / "final_eval_metrics.json"
    temporary_metrics_path = output_dir / ".final_eval_metrics.json.tmp"
    temporary_metrics_path.write_text(
        json.dumps(numeric_eval_metrics, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metrics_path, metrics_path)
    manifest = finalize_training_manifest(
        manifest,
        output_dir,
        runtime_environment=_runtime_environment(config),
    )
    manifest_path = write_training_manifest(manifest, output_dir)
    return TrainingRunResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        train_metrics=numeric_metrics,
        eval_metrics=numeric_eval_metrics,
    )
