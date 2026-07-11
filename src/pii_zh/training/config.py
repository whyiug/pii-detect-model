"""Validated YAML/CLI configuration for local token-classifier training."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "Training configuration requires PyYAML (install the training extra)."
    ) from exc


class TrainingConfigError(ValueError):
    """Raised when a training configuration is incomplete or unsafe."""


DOCUMENT_SAMPLING_STRATEGY = "rare_positive_v1_preserve_negative_mass"
BASE_INITIALIZATION_STRATEGY = "base_causal_lm_v1"
STAGED_INITIALIZATION_STRATEGY = "verified_token_classifier_to_full_v1"
TRAINING_SOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}")
MAX_TRAINING_SOURCE_IDS = 256


def normalize_training_source_ids(value: object) -> tuple[str, ...]:
    """Return a canonical, path-free set of registered source identifiers."""

    if not isinstance(value, (list, tuple)):
        raise TrainingConfigError("training_source_ids must be a list of source identifiers")
    if len(value) > MAX_TRAINING_SOURCE_IDS:
        raise TrainingConfigError(
            f"training_source_ids must contain at most {MAX_TRAINING_SOURCE_IDS} entries"
        )
    if any(
        not isinstance(source_id, str) or TRAINING_SOURCE_ID_PATTERN.fullmatch(source_id) is None
        for source_id in value
    ):
        raise TrainingConfigError(
            "training_source_ids must contain only non-empty path-free identifiers"
        )
    normalized = tuple(sorted(value))
    if len(set(normalized)) != len(normalized):
        raise TrainingConfigError("training_source_ids must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class LoraSettings:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @classmethod
    def from_mapping(cls, value: object) -> LoraSettings:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TrainingConfigError("lora must be a mapping")
        unknown = set(value) - {"rank", "alpha", "dropout", "target_modules"}
        if unknown:
            raise TrainingConfigError(f"unknown lora fields: {sorted(unknown)}")
        defaults = cls()
        targets = value.get("target_modules", defaults.target_modules)
        if not isinstance(targets, (list, tuple)) or not all(
            isinstance(item, str) and item for item in targets
        ):
            raise TrainingConfigError("lora.target_modules must contain non-empty strings")
        result = cls(
            rank=value.get("rank", defaults.rank),
            alpha=value.get("alpha", defaults.alpha),
            dropout=value.get("dropout", defaults.dropout),
            target_modules=tuple(targets),
        )
        if isinstance(result.rank, bool) or not isinstance(result.rank, int) or result.rank < 1:
            raise TrainingConfigError("lora.rank must be a positive integer")
        if isinstance(result.alpha, bool) or not isinstance(result.alpha, int) or result.alpha < 1:
            raise TrainingConfigError("lora.alpha must be a positive integer")
        if not isinstance(result.dropout, (int, float)) or not 0 <= result.dropout < 1:
            raise TrainingConfigError("lora.dropout must be in [0, 1)")
        return result


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Complete training recipe.

    YAML uses these field names directly.  Paths are intentionally excluded
    from the public manifest; their byte hashes identify inputs instead.
    """

    base_model: str
    train_file: str
    validation_file: str
    output_dir: str
    base_source_id: str = "qwen3_0_6b_base"
    training_source_ids: tuple[str, ...] = ()
    attention_mode: str = "full"
    fine_tuning: str = "lora"
    taxonomy_path: str | None = None
    max_length: int = 512
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    classifier_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    epochs: float = 3.0
    bf16: bool = True
    use_cpu: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42
    resume: bool | str = False
    initial_model: str | None = None
    class_weighting: bool = True
    class_weight_cap: float = 5.0
    document_sampling: bool = True
    document_sampling_cap: float = 5.0
    early_stopping_patience: int = 2
    classifier_dropout: float = 0.1
    include_extended_labels: bool = False
    include_auxiliary_labels: bool = False
    jpt_sep_token_id: int | None = None
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    lora: LoraSettings = field(default_factory=LoraSettings)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_source_ids",
            normalize_training_source_ids(self.training_source_ids),
        )

    def validate(self) -> None:
        for name in (
            "base_model",
            "train_file",
            "validation_file",
            "output_dir",
            "base_source_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TrainingConfigError(f"{name} must be a non-empty string")
        if self.attention_mode not in {"causal", "full", "jpt"}:
            raise TrainingConfigError("attention_mode must be causal, full, or jpt")
        if self.fine_tuning not in {"full", "lora"}:
            raise TrainingConfigError("fine_tuning must be full or lora")
        for name in (
            "bf16",
            "use_cpu",
            "gradient_checkpointing",
            "class_weighting",
            "document_sampling",
            "include_extended_labels",
            "include_auxiliary_labels",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TrainingConfigError(f"{name} must be a boolean")
        for name in (
            "max_length",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "early_stopping_patience",
            "save_total_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TrainingConfigError(f"{name} must be a positive integer")
        if self.max_length < 4:
            raise TrainingConfigError("max_length must be at least 4")
        if (
            isinstance(self.dataloader_num_workers, bool)
            or not isinstance(self.dataloader_num_workers, int)
            or self.dataloader_num_workers < 0
        ):
            raise TrainingConfigError("dataloader_num_workers must be non-negative")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TrainingConfigError("seed must be an integer")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, (int, float))
            or self.epochs <= 0
        ):
            raise TrainingConfigError("epochs must be positive")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise TrainingConfigError("learning_rate must be positive")
        if self.classifier_learning_rate is not None and (
            isinstance(self.classifier_learning_rate, bool)
            or not isinstance(self.classifier_learning_rate, (int, float))
            or self.classifier_learning_rate <= 0
        ):
            raise TrainingConfigError("classifier_learning_rate must be positive or null")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or self.weight_decay < 0
        ):
            raise TrainingConfigError("weight_decay must be non-negative")
        if (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, (int, float))
            or not 0 <= self.warmup_ratio < 1
        ):
            raise TrainingConfigError("warmup_ratio must be in [0, 1)")
        if (
            isinstance(self.classifier_dropout, bool)
            or not isinstance(self.classifier_dropout, (int, float))
            or not 0 <= self.classifier_dropout < 1
        ):
            raise TrainingConfigError("classifier_dropout must be in [0, 1)")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 1
            for value in (self.class_weight_cap, self.document_sampling_cap)
        ):
            raise TrainingConfigError("class/document weight caps must be at least 1")
        if self.taxonomy_path is not None and (
            not isinstance(self.taxonomy_path, str) or not self.taxonomy_path.strip()
        ):
            raise TrainingConfigError("taxonomy_path must be a non-empty string or null")
        if self.jpt_sep_token_id is not None and (
            isinstance(self.jpt_sep_token_id, bool)
            or not isinstance(self.jpt_sep_token_id, int)
            or self.jpt_sep_token_id < 0
        ):
            raise TrainingConfigError("jpt_sep_token_id must be a non-negative integer or null")
        if not isinstance(self.resume, (bool, str)):
            raise TrainingConfigError("resume must be false, true, or a local checkpoint path")
        if isinstance(self.resume, str) and not self.resume.strip():
            raise TrainingConfigError("resume path cannot be empty")
        if self.initial_model is not None and (
            not isinstance(self.initial_model, str) or not self.initial_model.strip()
        ):
            raise TrainingConfigError("initial_model must be a non-empty local path or null")
        if self.initial_model is not None:
            if self.attention_mode != "full":
                raise TrainingConfigError(
                    "initial_model is supported only for a full-attention target"
                )
            if self.resume is not False:
                raise TrainingConfigError("initial_model and resume are mutually exclusive")
            initial = Path(self.initial_model).expanduser().resolve(strict=False)
            output = Path(self.output_dir).expanduser().resolve(strict=False)
            if (
                initial == output
                or initial.is_relative_to(output)
                or output.is_relative_to(initial)
            ):
                raise TrainingConfigError(
                    "initial_model and output_dir must be distinct, non-nested directories"
                )

    @classmethod
    def from_mapping(cls, value: object) -> TrainingConfig:
        if not isinstance(value, dict):
            raise TrainingConfigError("training YAML must contain a top-level mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise TrainingConfigError(f"unknown training fields: {sorted(unknown)}")
        required = {"base_model", "train_file", "validation_file", "output_dir"}
        missing = required - set(value)
        if missing:
            raise TrainingConfigError(f"missing training fields: {sorted(missing)}")
        values = dict(value)
        values["lora"] = LoraSettings.from_mapping(values.get("lora"))
        values["training_source_ids"] = normalize_training_source_ids(
            values.get("training_source_ids", ())
        )
        result = cls(**values)
        result.validate()
        return result

    def manifest_recipe(self) -> dict[str, Any]:
        """Return hyperparameters without local/private filesystem paths."""

        result = asdict(self)
        for key in (
            "base_model",
            "train_file",
            "validation_file",
            "output_dir",
            "taxonomy_path",
            "initial_model",
        ):
            result.pop(key, None)
        if isinstance(result["resume"], str):
            result["resume"] = True
        result["training_source_ids"] = list(self.training_source_ids)
        result["effective_classifier_learning_rate"] = self.effective_classifier_learning_rate
        result["document_sampling_strategy"] = (
            DOCUMENT_SAMPLING_STRATEGY if self.document_sampling else "disabled"
        )
        result["initialization_strategy"] = self.initialization_strategy
        return result

    @property
    def initialization_strategy(self) -> str:
        return (
            STAGED_INITIALIZATION_STRATEGY
            if self.initial_model is not None
            else BASE_INITIALIZATION_STRATEGY
        )

    @property
    def effective_classifier_learning_rate(self) -> float:
        """Use a higher LR only when the BIO head is newly initialized."""

        if self.classifier_learning_rate is not None:
            return float(self.classifier_learning_rate)
        if self.initial_model is not None:
            return float(self.learning_rate)
        if self.fine_tuning == "lora":
            return 5e-4
        return float(self.learning_rate) * 10.0


def load_training_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Load a strict YAML config and apply explicit CLI overrides."""

    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TrainingConfigError("training YAML must contain a top-level mapping")
    merged = dict(value)
    if overrides:
        merged.update({key: item for key, item in overrides.items() if item is not None})
    return TrainingConfig.from_mapping(merged)
