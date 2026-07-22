"""Strict causal-to-full attention conversion for project BIE73 classifiers.

The conversion is intentionally in-memory and architecture-only.  It preserves
the complete ``model.*``/``score.*`` state dict, the 73-label BIE schema, custom
lineage fields, parameter trainability, and the source model's train/eval mode.
Only the Qwen3 attention contract and architecture identifiers may change.

This module does not load datasets, checkpoints, or model artifacts.  Callers
must perform provenance and artifact verification before constructing the
source model passed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForTokenClassification
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "BIE attention conversion requires PyTorch and Transformers Qwen3 support."
    ) from exc

from pii_zh.models.qwen3_bi import (
    SUPPORTED_ATTENTION_BACKENDS,
    Qwen3BiForTokenClassification,
)
from pii_zh.taxonomy import load_taxonomy

BIE73_FULL_ATTENTION_CONVERSION_STRATEGY = "qwen3_causal_bie73_to_qwen3_bi_full_v1"
_INTENTIONAL_CONFIG_DIFFERENCES = frozenset(
    {
        "architecture_version",
        "architectures",
        "auto_map",
        "bi_attention_backend",
        "model_type",
        "pii_attention_mode",
    }
)


class BieAttentionConversionError(RuntimeError):
    """Raised when a BIE73 attention conversion is not exact and fail-closed."""


@dataclass(frozen=True, slots=True)
class BieAttentionConversionAudit:
    """Path-free evidence for an exact in-memory attention conversion."""

    strategy: str
    source_architecture: str
    target_architecture: str
    source_model_type: str
    target_model_type: str
    source_attention_mode: str
    target_attention_mode: str
    attention_backend: str
    tagging_scheme: str
    label_count: int
    state_dict_key_count: int
    state_dict_keys_identical: bool
    state_dict_values_identical: bool
    classifier_keys: tuple[str, ...]
    classifier_values_identical: bool
    config_payload_identical_except_attention_contract: bool
    preserved_config_key_count: int
    trainability_preserved: bool
    source_training_mode_preserved: bool
    source_device: str
    target_device: str
    lora_target_module_counts: tuple[tuple[str, int], ...]
    newly_initialized_parameter_keys: tuple[str, ...]
    discarded_parameter_keys: tuple[str, ...]
    release_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit payload."""

        return asdict(self)


def _normalize_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_id, raw_label in value.items():
        if isinstance(raw_id, bool) or not isinstance(raw_label, str):
            raise BieAttentionConversionError("BIE id2label contains an invalid entry")
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise BieAttentionConversionError("BIE id2label contains an invalid key") from exc
        if label_id < 0 or not raw_label or label_id in normalized:
            raise BieAttentionConversionError("BIE id2label is not a unique contiguous mapping")
        normalized[label_id] = raw_label
    if sorted(normalized) != list(range(len(normalized))):
        raise BieAttentionConversionError("BIE id2label is not a unique contiguous mapping")
    return normalized


def _expected_bie73_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    taxonomy = load_taxonomy()
    core_labels = tuple(entity.name for entity in taxonomy.label_sets["core"])
    if taxonomy.outside_label != "O" or len(core_labels) != 24:
        raise BieAttentionConversionError("packaged taxonomy does not define the 24-label core")
    labels = [taxonomy.outside_label]
    for entity in core_labels:
        labels.extend((f"B-{entity}", f"I-{entity}", f"E-{entity}"))
    label2id = {label: index for index, label in enumerate(labels)}
    return label2id, {index: label for label, index in label2id.items()}


def _validate_source(model: Qwen3ForTokenClassification) -> None:
    if type(model) is not Qwen3ForTokenClassification:
        raise BieAttentionConversionError(
            "source must be a standalone native Qwen3ForTokenClassification"
        )
    config = model.config
    if not isinstance(config, Qwen3Config) or config.model_type != "qwen3":
        raise BieAttentionConversionError("source config must be native Qwen3")
    if config.architectures != ["Qwen3ForTokenClassification"]:
        raise BieAttentionConversionError("source architecture declaration is not standalone")
    if getattr(config, "auto_map", None):
        raise BieAttentionConversionError("source auto_map is forbidden")
    if (
        getattr(config, "pii_attention_mode", None) != "causal"
        or getattr(config, "pii_tagging_scheme", None) != "BIE"
        or getattr(config, "pii_release_eligible", None) is not False
        or getattr(config, "pii_training_status", None) != "initialized_untrained"
        or config.use_cache is not False
    ):
        raise BieAttentionConversionError(
            "source does not satisfy the causal BIE research contract"
        )

    expected_label2id, expected_id2label = _expected_bie73_label_maps()
    observed_id2label = _normalize_id2label(config.id2label)
    if (
        int(config.num_labels) != 73
        or config.label2id != expected_label2id
        or observed_id2label != expected_id2label
    ):
        raise BieAttentionConversionError("source does not use the exact project BIE73 schema")
    if (
        model.num_labels != 73
        or model.score.in_features != int(config.hidden_size)
        or model.score.out_features != 73
        or model.score.bias is None
    ):
        raise BieAttentionConversionError("source BIE73 classifier head has an invalid shape")
    if any(tensor.device.type != "cpu" for tensor in model.state_dict().values()):
        raise BieAttentionConversionError("attention conversion must run on a CPU source model")


def _validated_lora_targets(target_modules: Sequence[str]) -> tuple[str, ...]:
    targets = tuple(target_modules)
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise BieAttentionConversionError("LoRA target modules must be unique non-empty names")
    return targets


def _lora_target_counts(
    model: torch.nn.Module,
    *,
    target_modules: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    counts = {target: 0 for target in target_modules}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in counts:
            if not isinstance(module, torch.nn.Linear):
                raise BieAttentionConversionError(
                    f"LoRA target {leaf!r} is not a linear projection"
                )
            counts[leaf] += 1
    if any(count == 0 for count in counts.values()):
        raise BieAttentionConversionError("at least one requested LoRA target is absent")
    return tuple((target, counts[target]) for target in target_modules)


def assert_lora_target_compatibility(
    source: torch.nn.Module,
    target: torch.nn.Module,
    *,
    target_modules: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    """Require identical, non-empty Qwen projection targets across conversion."""

    targets = _validated_lora_targets(target_modules)
    source_counts = _lora_target_counts(source, target_modules=targets)
    target_counts = _lora_target_counts(target, target_modules=targets)
    if source_counts != target_counts:
        raise BieAttentionConversionError("attention conversion changed LoRA target modules")
    return target_counts


def _comparable_config_payload(config: Qwen3Config) -> dict[str, Any]:
    payload = config.to_dict()
    for key in _INTENTIONAL_CONFIG_DIFFERENCES:
        payload.pop(key, None)
    return payload


def convert_causal_bie73_to_full_attention(
    model: Qwen3ForTokenClassification,
    *,
    bi_attention_backend: str = "sdpa",
    lora_target_modules: Sequence[str] | None = None,
) -> tuple[Qwen3BiForTokenClassification, BieAttentionConversionAudit]:
    """Convert a verified causal BIE73 classifier to Qwen3Bi without weight changes.

    Conversion is intended to run immediately after BIE73 initialization and
    before PEFT attachment or CUDA placement.  Passing ``lora_target_modules``
    additionally proves that the requested projection set is unchanged.
    """

    _validate_source(model)
    if bi_attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise BieAttentionConversionError("unsupported full-attention backend")

    source_state = model.state_dict()
    source_keys = tuple(source_state)
    source_trainability = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    source_training = model.training
    source_config_payload = _comparable_config_payload(model.config)
    try:
        converted = Qwen3BiForTokenClassification.from_qwen3_token_classifier_state_dict(
            model.config,
            source_state,
            strict=True,
            bi_attention_backend=bi_attention_backend,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BieAttentionConversionError("strict BIE73 state-dict conversion failed") from exc

    # Full attention alone does not make this research BIE artifact release
    # eligible.  A separate release gate may change the value only after all
    # model, inference, and service contracts support the new architecture.
    converted.config.pii_release_eligible = False
    converted.train(source_training)

    target_state = converted.state_dict()
    if any(tensor.device.type != "cpu" for tensor in target_state.values()):
        raise BieAttentionConversionError("converted BIE73 model unexpectedly left CPU")
    target_keys = tuple(target_state)
    if target_keys != source_keys:
        raise BieAttentionConversionError("conversion changed the state-dict key schema")
    mismatched_values = tuple(
        key for key in source_keys if not torch.equal(source_state[key], target_state[key])
    )
    if mismatched_values:
        raise BieAttentionConversionError("conversion changed one or more state-dict tensors")

    target_parameters = dict(converted.named_parameters())
    if tuple(target_parameters) != tuple(source_trainability):
        raise BieAttentionConversionError("conversion changed the parameter-name schema")
    for name, parameter in target_parameters.items():
        parameter.requires_grad_(source_trainability[name])
    trainability_preserved = all(
        target_parameters[name].requires_grad == required
        for name, required in source_trainability.items()
    )
    if not trainability_preserved:
        raise BieAttentionConversionError("conversion did not preserve parameter trainability")
    training_mode_preserved = converted.training == source_training
    if not training_mode_preserved:
        raise BieAttentionConversionError("conversion did not preserve train/eval mode")

    expected_label2id, expected_id2label = _expected_bie73_label_maps()
    if (
        type(converted) is not Qwen3BiForTokenClassification
        or converted.config.model_type != "qwen3_bi"
        or converted.config.architectures != ["Qwen3BiForTokenClassification"]
        or converted.config.pii_attention_mode != "full"
        or converted.config.pii_tagging_scheme != "BIE"
        or converted.config.pii_release_eligible is not False
        or converted.config.use_cache is not False
        or converted.config.num_labels != 73
        or converted.config.label2id != expected_label2id
        or _normalize_id2label(converted.config.id2label) != expected_id2label
    ):
        raise BieAttentionConversionError("converted model has an incomplete full BIE73 contract")

    target_config_payload = _comparable_config_payload(converted.config)
    if target_config_payload != source_config_payload:
        raise BieAttentionConversionError(
            "conversion changed config fields outside the attention contract"
        )

    lora_counts: tuple[tuple[str, int], ...] = ()
    if lora_target_modules is not None:
        lora_counts = assert_lora_target_compatibility(
            model,
            converted,
            target_modules=lora_target_modules,
        )

    classifier_keys = tuple(key for key in source_keys if key.startswith("score."))
    if classifier_keys != ("score.weight", "score.bias"):
        raise BieAttentionConversionError("BIE73 classifier state keys are not canonical")
    classifier_identical = all(
        torch.equal(source_state[key], target_state[key]) for key in classifier_keys
    )
    if not classifier_identical:
        raise BieAttentionConversionError("conversion changed the BIE73 classifier head")

    audit = BieAttentionConversionAudit(
        strategy=BIE73_FULL_ATTENTION_CONVERSION_STRATEGY,
        source_architecture="Qwen3ForTokenClassification",
        target_architecture="Qwen3BiForTokenClassification",
        source_model_type="qwen3",
        target_model_type="qwen3_bi",
        source_attention_mode="causal",
        target_attention_mode="full",
        attention_backend=bi_attention_backend,
        tagging_scheme="BIE",
        label_count=73,
        state_dict_key_count=len(source_keys),
        state_dict_keys_identical=True,
        state_dict_values_identical=True,
        classifier_keys=classifier_keys,
        classifier_values_identical=True,
        config_payload_identical_except_attention_contract=True,
        preserved_config_key_count=len(source_config_payload),
        trainability_preserved=trainability_preserved,
        source_training_mode_preserved=training_mode_preserved,
        source_device="cpu",
        target_device="cpu",
        lora_target_module_counts=lora_counts,
        newly_initialized_parameter_keys=(),
        discarded_parameter_keys=(),
        release_eligible=False,
    )
    return converted, audit


__all__ = [
    "BIE73_FULL_ATTENTION_CONVERSION_STRATEGY",
    "BieAttentionConversionAudit",
    "BieAttentionConversionError",
    "assert_lora_target_compatibility",
    "convert_causal_bie73_to_full_attention",
]
