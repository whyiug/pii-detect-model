"""Weighted Trainer extensions and privacy-safe character validation metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sized
from inspect import signature
from typing import Any, cast

try:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
    from transformers import EvalPrediction, PreTrainedModel, Trainer
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "Training requires numpy, torch, and Transformers (install the training extra)."
    ) from exc

from pii_zh.models.decoding import decode_bio_ids
from pii_zh.training.config import LoraSettings
from pii_zh.training.data import DocumentTokenDataset

# Keep these fail-closed eligibility floors aligned with the validation-only
# span calibration defaults in ``scripts/calibrate_predictions.py``.  They do
# not change the existing early-stop score; they make a checkpoint that loses
# one risk-bearing label explicitly ineligible for downstream calibration.
_CALIBRATION_RECALL_FLOORS = {"T0": 0.90, "T1": 0.85}


def prepare_parameter_efficient_model(
    model: PreTrainedModel,
    *,
    fine_tuning: str,
    lora: LoraSettings,
) -> torch.nn.Module:
    """Return a full-finetune model or attach a PEFT token-classification LoRA."""

    if fine_tuning == "full":
        return model
    if fine_tuning != "lora":
        raise ValueError("fine_tuning must be full or lora")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional branch
        raise ImportError(
            "LoRA was requested but peft is unavailable; install the training extra "
            "or set fine_tuning: full."
        ) from exc

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=float(lora.dropout),
        target_modules=list(lora.target_modules),
        modules_to_save=["score"],
        bias="none",
    )
    return get_peft_model(model, peft_config)


def merge_parameter_efficient_model_for_release(
    model: torch.nn.Module,
    *,
    fine_tuning: str,
    attention_mode: str = "full",
    jpt_sep_token_id: int | None = None,
) -> PreTrainedModel:
    """Return a complete standalone model, merging adapters with safe checks."""

    if fine_tuning == "full":
        merged = model
    elif fine_tuning == "lora":
        merge = getattr(model, "merge_and_unload", None)
        if not callable(merge):
            raise RuntimeError("LoRA training did not produce a mergeable PEFT model")
        try:
            merged = merge(safe_merge=True)
        except Exception as exc:
            raise RuntimeError("LoRA safe merge failed") from exc
    else:
        raise ValueError("fine_tuning must be full or lora")

    if attention_mode not in {"causal", "full", "jpt"}:
        raise ValueError("attention_mode must be causal, full, or jpt")
    # Both native Transformers models and PEFT's merge result expose the
    # PreTrainedModel release API; PEFT does not declare that nominally.
    release_model = cast(PreTrainedModel, merged)
    config = getattr(release_model, "config", None)
    expected_model_type = "qwen3_bi" if attention_mode == "full" else "qwen3"
    if config is None or getattr(config, "model_type", None) != expected_model_type:
        raise RuntimeError("final model architecture disagrees with its attention mode")
    config.architectures = [
        "Qwen3BiForTokenClassification"
        if attention_mode == "full"
        else "Qwen3ForTokenClassification"
    ]
    config.use_cache = False
    config.pii_attention_mode = attention_mode
    config.pii_release_eligible = attention_mode == "full"
    if attention_mode == "jpt":
        if jpt_sep_token_id is None:
            raise RuntimeError("JPT finalization requires the effective separator token id")
        config.pii_jpt_sep_token_id = int(jpt_sep_token_id)
    elif hasattr(config, "pii_jpt_sep_token_id"):
        delattr(config, "pii_jpt_sep_token_id")
    # Transformers persists this private field in config.json.  A local base
    # directory is neither portable nor appropriate for a public artifact.
    config._name_or_path = ""
    return release_model


class SecureTokenTrainer(Trainer):
    """Trainer with class-weighted loss and optional document-level sampling."""

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        document_sampling_weights: torch.Tensor | None = None,
        sampler_seed: int = 42,
        classifier_learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.document_sampling_weights = document_sampling_weights
        self.sampler_seed = sampler_seed
        if classifier_learning_rate is not None and classifier_learning_rate <= 0:
            raise ValueError("classifier_learning_rate must be positive or null")
        self.classifier_learning_rate = classifier_learning_rate
        self.model_accepts_loss_kwargs = False

    @staticmethod
    def _is_classifier_parameter(name: str) -> bool:
        # Covers the full model's ``score.*`` and PEFT's
        # ``score.modules_to_save.default.*`` without relying on adapter names.
        return "score" in name.split(".")

    def create_optimizer(
        self,
        model: torch.nn.Module | None = None,
    ) -> torch.optim.Optimizer:
        """Assign the newly initialized classifier a separate learning rate."""

        if self.classifier_learning_rate is None:
            base_create_optimizer: Callable[..., object] = super().create_optimizer
            accepts_model = "model" in signature(base_create_optimizer).parameters
            if model is None:
                base_create_optimizer()
            elif accepts_model:
                # Transformers 5 accepts a model prepared for delayed optimizer
                # creation (for example FSDP); use that model directly.
                base_create_optimizer(model)
            else:
                # Transformers 4.57 has a self-only override point.  Preserve
                # its optimizer implementation while making an explicit model
                # call use the same model it would have received in 5.x.
                original_model = self.model
                original_model_wrapped = self.model_wrapped
                try:
                    self.model = model
                    self.model_wrapped = model
                    base_create_optimizer()
                finally:
                    self.model = original_model
                    self.model_wrapped = original_model_wrapped
            if self.optimizer is None:
                raise RuntimeError("Transformers did not create an optimizer")
            return self.optimizer
        if self.optimizer is not None:
            return self.optimizer

        optimizer_model = self.model_wrapped if model is None else model
        decay_names = set(self.get_decay_parameter_names(optimizer_model))
        grouped: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
        seen: set[int] = set()
        for name, parameter in optimizer_model.named_parameters():
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity in seen:
                raise RuntimeError("a trainable parameter appears more than once")
            seen.add(identity)
            key = (self._is_classifier_parameter(name), name in decay_names)
            grouped.setdefault(key, []).append(parameter)

        expected = {
            id(parameter) for parameter in optimizer_model.parameters() if parameter.requires_grad
        }
        if seen != expected:
            raise RuntimeError("optimizer parameter grouping omitted trainable parameters")
        if not any(is_head and parameters for (is_head, _), parameters in grouped.items()):
            raise RuntimeError("classifier learning rate was requested but score is not trainable")
        optimizer_groups = [
            {
                "params": parameters,
                "lr": self.classifier_learning_rate if is_head else self.args.learning_rate,
                "weight_decay": self.args.weight_decay if use_decay else 0.0,
            }
            for (is_head, use_decay), parameters in sorted(grouped.items())
            if parameters
        ]
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            # Trainer accepts arbitrary torch modules, but this helper's third-party
            # annotation is narrower than its implementation (PreTrainedModel only).
            pretrained_optimizer_model = cast(PreTrainedModel, optimizer_model)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, pretrained_optimizer_model
            )
        optimizer_kwargs = dict(optimizer_kwargs)
        if any(key in optimizer_kwargs for key in ("params", "model", "optimizer_dict")):
            raise RuntimeError("selected optimizer is incompatible with explicit LR groups")
        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        if "labels" not in inputs:
            raise ValueError("token-classification batches must contain labels")
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=weight,
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(
        self,
        train_dataset: Dataset[Any] | None = None,
    ) -> Sampler[int] | None:
        if self.document_sampling_weights is None:
            return super()._get_train_sampler(train_dataset)
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if not isinstance(dataset, Sized):
            raise TypeError("document-weighted sampling requires a sized training dataset")
        if self.args.world_size > 1:
            raise RuntimeError(
                "document-weighted sampling currently supports one training process; "
                "disable it before distributed training"
            )
        dataset_length = len(dataset)
        if len(self.document_sampling_weights) != dataset_length:
            raise ValueError("document sampling weights do not match the training dataset")
        generator = torch.Generator()
        generator.manual_seed(self.sampler_seed)
        weights = [float(value) for value in self.document_sampling_weights.tolist()]
        return WeightedRandomSampler(
            weights,
            num_samples=dataset_length,
            replacement=True,
            generator=generator,
        )


def _prf(tp: int, fp: int, fn: int, *, beta: float = 1.0) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    score = (1 + beta_squared) * precision * recall / denominator if denominator else 0.0
    return precision, recall, score


def _strict_counts(
    gold_documents: list[set[tuple[int, int, str]]],
    predicted_documents: list[set[tuple[int, int, str]]],
    *,
    allowed_labels: set[str] | None = None,
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for gold, predicted in zip(gold_documents, predicted_documents, strict=True):
        if allowed_labels is not None:
            gold = {item for item in gold if item[2] in allowed_labels}
            predicted = {item for item in predicted if item[2] in allowed_labels}
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    return tp, fp, fn


class CharacterValidationComputer:
    """Decode eval logits to spans and compute the risk-weighted early-stop score."""

    def __init__(
        self,
        dataset: DocumentTokenDataset,
        *,
        id2label: Mapping[int, str],
        label_to_risk_tiers: Mapping[str, tuple[str, ...]],
        attention_mode: str,
    ) -> None:
        self.dataset = dataset
        self.id2label = dict(id2label)
        self.label_to_risk_tiers = dict(label_to_risk_tiers)
        self.attention_mode = attention_mode

    def __call__(self, prediction: EvalPrediction) -> dict[str, float]:
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits_array = np.asarray(logits)
        if logits_array.shape[0] != len(self.dataset.documents):
            raise ValueError("validation predictions do not match document order")
        predicted_ids = logits_array.argmax(axis=-1)
        if self.attention_mode == "jpt":
            prediction_width = predicted_ids.shape[1]
            if prediction_width < 3 or prediction_width % 2 != 1:
                raise ValueError("JPT validation predictions must have an odd repeated width")
            jpt_second_copy_start = (prediction_width - 1) // 2 + 1
        else:
            jpt_second_copy_start = 0

        gold_documents: list[set[tuple[int, int, str]]] = []
        predicted_documents: list[set[tuple[int, int, str]]] = []
        for row, document in enumerate(self.dataset.documents):
            token_count = len(document.offset_mapping)
            if self.attention_mode == "jpt":
                row_ids = predicted_ids[
                    row,
                    jpt_second_copy_start : jpt_second_copy_start + token_count,
                ]
            else:
                row_ids = predicted_ids[row, :token_count]
            decoded = decode_bio_ids(row_ids.tolist(), document.offset_mapping, self.id2label)
            predicted_documents.append(
                {(span.start, span.end, span.entity_type) for span in decoded}
            )
            gold_documents.append(
                {(span.start, span.end, span.label) for span in document.gold_spans}
            )

        all_labels = set(self.label_to_risk_tiers)
        label_counts = {
            label: _strict_counts(
                gold_documents,
                predicted_documents,
                allowed_labels={label},
            )
            for label in sorted(all_labels)
        }
        macro_scores: list[float] = []
        for label in sorted(all_labels):
            tp, fp, fn = label_counts[label]
            if tp + fp + fn:
                macro_scores.append(_prf(tp, fp, fn)[2])
        strict_macro_f1 = sum(macro_scores) / len(macro_scores) if macro_scores else 0.0

        tier_scores: dict[str, float] = {}
        for tier in ("T0", "T1"):
            labels = {label for label, tiers in self.label_to_risk_tiers.items() if tier in tiers}
            tp, fp, fn = _strict_counts(gold_documents, predicted_documents, allowed_labels=labels)
            tier_scores[tier] = _prf(tp, fp, fn, beta=2.0)[2]

        # Threshold calibration cannot recover an exact span/label candidate
        # that the checkpoint never emits.  Mirror calibration's T0-first risk
        # precedence and require every configured T0/T1 label to meet the same
        # recall floor used by ``calibrate_predictions.py``.  A configured
        # label with no validation gold is deliberately assigned zero recall:
        # absence of evidence must not make a release candidate eligible.
        calibration_labels = {
            "T0": {label for label, tiers in self.label_to_risk_tiers.items() if "T0" in tiers},
            "T1": {
                label
                for label, tiers in self.label_to_risk_tiers.items()
                if "T0" not in tiers and "T1" in tiers
            },
        }
        minimum_label_recall: dict[str, float] = {}
        recall_floor_met: dict[str, float] = {}
        for tier, floor in _CALIBRATION_RECALL_FLOORS.items():
            labels = calibration_labels[tier]
            recalls = []
            for label in sorted(labels):
                tp, _, fn = label_counts[label]
                recalls.append(tp / (tp + fn) if tp + fn else 0.0)
            # A tier that is not part of a reduced label schema is vacuously
            # satisfied; an entirely risk-free schema remains ineligible below.
            minimum_label_recall[tier] = min(recalls) if recalls else 1.0
            recall_floor_met[tier] = float(minimum_label_recall[tier] >= floor)
        has_calibratable_label = any(calibration_labels.values())
        calibration_recall_eligible = float(
            has_calibratable_label and all(recall_floor_met.values())
        )

        pii_free_rows = [index for index, gold in enumerate(gold_documents) if not gold]
        if pii_free_rows:
            false_positive_rate = sum(
                bool(predicted_documents[index]) for index in pii_free_rows
            ) / len(pii_free_rows)
            pii_free_precision = 1.0 - false_positive_rate
        else:
            pii_free_precision = 1.0

        calibration_score = self._token_calibration_score(
            logits_array, np.asarray(prediction.label_ids)
        )
        risk_score = (
            0.45 * tier_scores["T0"]
            + 0.25 * tier_scores["T1"]
            + 0.15 * strict_macro_f1
            + 0.10 * pii_free_precision
            + 0.05 * calibration_score
        )
        return {
            "tier0_f2": tier_scores["T0"],
            "tier1_f2": tier_scores["T1"],
            "tier0_min_label_recall": minimum_label_recall["T0"],
            "tier1_min_label_recall": minimum_label_recall["T1"],
            "tier0_recall_floor_met": recall_floor_met["T0"],
            "tier1_recall_floor_met": recall_floor_met["T1"],
            "calibration_recall_eligible": calibration_recall_eligible,
            "strict_macro_f1": strict_macro_f1,
            "pii_free_precision": pii_free_precision,
            "calibration_score": calibration_score,
            "risk_weighted_score": risk_score,
        }

    @staticmethod
    def _token_calibration_score(
        logits: np.ndarray[Any, np.dtype[Any]],
        labels: np.ndarray[Any, np.dtype[Any]],
    ) -> float:
        valid = labels != -100
        if not valid.any():
            return 0.0
        selected_logits = logits[valid]
        selected_labels = labels[valid].astype(np.int64)
        shifted = selected_logits - selected_logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        targets = np.zeros_like(probabilities)
        targets[np.arange(len(selected_labels)), selected_labels] = 1.0
        brier = np.square(probabilities - targets).sum(axis=-1).mean()
        if not math.isfinite(float(brier)):
            return 0.0
        return max(0.0, 1.0 - float(brier))
