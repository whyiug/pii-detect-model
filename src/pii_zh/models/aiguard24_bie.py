"""Pinned AIguard initialization for a native-causal 24-class BIE model.

The transition deliberately preserves AIguard's causal attention contract and
its three boundary states.  The target classifier is ``O + 24 * (B, I, E)``:

* ``O`` and all ``B/I/E`` rows for the twelve compatible project labels are
  copied from the pinned AIguard checkpoint;
* BANK_CARD_NUMBER averages the bank-card and credit-card rows separately for
  each of ``B``, ``I`` and ``E``; and
* all rows for the other twelve labels retain deterministic Qwen
  initialization controlled by ``seed``.

This module performs initialization only.  It never reads evaluation data and
always marks the returned model as an untrained, non-release candidate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForTokenClassification
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "AIguard24 BIE initialization requires PyTorch and Transformers with Qwen3 support."
    ) from exc

from pii_zh.models.aiguard24 import (
    AIGUARD24_TARGET_TO_SOURCE,
    AIGUARD_SOURCE_ENTITY_TYPES,
    AIGUARD_SOURCE_MODEL_ID,
    AIGUARD_SOURCE_REVISION,
)
from pii_zh.taxonomy import Taxonomy, load_taxonomy

AIGUARD24_BIE_INITIALIZATION_STRATEGY = "aiguard_bie_to_pii_zh_core24_bie_v1"

_EXPECTED_SOURCE_HASHES: dict[str, str] = {
    "config.json": "5f8b5cafa13310bd5327e90beabc919307f749cb00714d4e87bff4d22cb31225",
    "model.safetensors": "89c9b36aa98a155ea3255eb2c53280858c49b9fed790b43107c48a53158ec4b2",
    "tokenizer.json": "352a863cd2761388ccc58f1432467ba6a1037bf12df9069889b142fa246471f6",
    "tokenizer_config.json": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
}
_BIE_PREFIXES = ("B", "I", "E")


class Aiguard24BieInitializationError(RuntimeError):
    """Raised when the pinned source or BIE weight transition fails closed."""


@dataclass(frozen=True, slots=True)
class BieHeadProjectionAudit:
    """Explicit row mapping for one target entity."""

    target_label: str
    source_labels: tuple[str, ...]
    target_to_source_tags: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Aiguard24BieInitializationAudit:
    """Path-free evidence for the causal BIE initialization."""

    schema_version: int
    strategy: str
    source_model_id: str
    source_revision: str
    source_hashes: dict[str, str]
    source_architecture: str
    source_label_count: int
    source_entity_types: tuple[str, ...]
    taxonomy_version: str
    target_architecture: str
    target_tag_scheme: str
    target_label_count: int
    target_label2id: dict[str, int]
    initialization_seed: int
    mapped_target_labels: tuple[str, ...]
    unmapped_target_labels: tuple[str, ...]
    head_projections: tuple[BieHeadProjectionAudit, ...]
    backbone_copy_mode: str
    backbone_tensor_count: int
    backbone_parameter_count: int
    backbone_missing_keys: tuple[str, ...]
    backbone_unexpected_keys: tuple[str, ...]
    outside_row_copied: bool
    mapped_head_rows_verified: bool
    unmapped_head_rows_preserved: bool
    attention_mode: str
    release_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(json.dumps(asdict(self), ensure_ascii=True, sort_keys=True))
        if not isinstance(value, dict):  # pragma: no cover - dataclass root is an object
            raise Aiguard24BieInitializationError("initialization audit is not an object")
        return value


@dataclass(frozen=True, slots=True)
class Aiguard24BieInitializationResult:
    """The standalone Transformers model and its initialization audit."""

    model: Qwen3ForTokenClassification
    audit: Aiguard24BieInitializationAudit


def build_core_bie_label_maps(
    taxonomy: Taxonomy | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build stable ``O, B/I/E`` maps in packaged core-taxonomy order."""

    selected = taxonomy or load_taxonomy()
    labels = [selected.outside_label]
    for entity in selected.label_sets["core"]:
        labels.extend(f"{prefix}-{entity.name}" for prefix in _BIE_PREFIXES)
    label2id = {label: index for index, label in enumerate(labels)}
    return label2id, {index: label for label, index in label2id.items()}


def configure_head_only_training(model: Qwen3ForTokenClassification) -> dict[str, int]:
    """Freeze the complete causal backbone and leave only ``score`` trainable."""

    if not isinstance(model, Qwen3ForTokenClassification):
        raise Aiguard24BieInitializationError("head-only mode requires native Qwen3")
    if (
        int(model.config.num_labels) != 73
        or getattr(model.config, "pii_tagging_scheme", None) != "BIE"
    ):
        raise Aiguard24BieInitializationError("head-only mode requires the 73-label BIE model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.score.parameters():
        parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    classifier = sum(parameter.numel() for parameter in model.score.parameters())
    if trainable != classifier or trainable < 1:
        raise Aiguard24BieInitializationError("head-only freezing did not isolate the classifier")
    return {
        "trainable_parameter_count": int(trainable),
        "frozen_parameter_count": int(
            sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            )
        ),
    }


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_pinned_source(source_model: str | Path) -> tuple[Path, dict[str, str]]:
    candidate = Path(source_model).expanduser()
    if candidate.is_symlink():
        raise Aiguard24BieInitializationError("source model directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise Aiguard24BieInitializationError(
            "source model is not an available local directory"
        ) from exc
    if not root.is_dir():
        raise Aiguard24BieInitializationError("source model must be a local directory")
    for name in sorted(_EXPECTED_SOURCE_HASHES):
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file():
            raise Aiguard24BieInitializationError(
                "source model is missing a required regular safe artifact"
            )
    forbidden_suffixes = {".bin", ".pkl", ".pickle", ".pt", ".pth"}
    if any(item.is_file() and item.suffix.lower() in forbidden_suffixes for item in root.iterdir()):
        raise Aiguard24BieInitializationError("pickle-based source artifacts are forbidden")
    observed = {name: _sha256_file(root / name) for name in sorted(_EXPECTED_SOURCE_HASHES)}
    if observed != dict(sorted(_EXPECTED_SOURCE_HASHES.items())):
        raise Aiguard24BieInitializationError(
            "source model does not match the pinned AIguard revision"
        )
    return root, observed


def _expected_source_labels() -> tuple[str, ...]:
    return (
        "O",
        *(f"B-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
        *(f"I-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
        *(f"E-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
    )


def _normalize_id2label(value: Mapping[Any, Any]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_id, raw_label in value.items():
        if isinstance(raw_id, bool):
            raise Aiguard24BieInitializationError("source id2label contains an invalid key")
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise Aiguard24BieInitializationError(
                "source id2label contains an invalid key"
            ) from exc
        if label_id < 0 or not isinstance(raw_label, str) or not raw_label:
            raise Aiguard24BieInitializationError("source id2label contains an invalid entry")
        if label_id in normalized:
            raise Aiguard24BieInitializationError("source id2label contains duplicate keys")
        normalized[label_id] = raw_label
    if sorted(normalized) != list(range(len(normalized))):
        raise Aiguard24BieInitializationError("source id2label must be contiguous")
    return normalized


def _validate_source_model(model: Qwen3ForTokenClassification) -> dict[int, str]:
    config = model.config
    if not isinstance(config, Qwen3Config) or config.model_type != "qwen3":
        raise Aiguard24BieInitializationError("source config must be native Qwen3")
    if config.architectures != ["Qwen3ForTokenClassification"]:
        raise Aiguard24BieInitializationError(
            "source config must declare a standalone Qwen3 token classifier"
        )
    if getattr(config, "auto_map", None):
        raise Aiguard24BieInitializationError("source config auto_map is forbidden")
    id2label = _normalize_id2label(config.id2label)
    expected = _expected_source_labels()
    if tuple(id2label.values()) != expected:
        raise Aiguard24BieInitializationError(
            "source label inventory does not match the pinned AIguard protocol"
        )
    label2id = {label: index for index, label in id2label.items()}
    if config.label2id != label2id or int(config.num_labels) != len(expected):
        raise Aiguard24BieInitializationError("source label maps are not exact inverses")
    if (
        model.score.in_features != int(config.hidden_size)
        or model.score.out_features != len(expected)
        or model.score.bias is None
    ):
        raise Aiguard24BieInitializationError("source classifier shape is incompatible")
    return id2label


def _validate_taxonomy(taxonomy: Taxonomy | None) -> Taxonomy:
    packaged = load_taxonomy()
    selected = taxonomy or packaged
    packaged_core = tuple(entity.name for entity in packaged.label_sets["core"])
    selected_core = tuple(entity.name for entity in selected.label_sets["core"])
    if (
        selected.taxonomy_version != packaged.taxonomy_version
        or selected.outside_label != "O"
        or selected_core != packaged_core
        or len(selected_core) != 24
    ):
        raise Aiguard24BieInitializationError(
            "target taxonomy must exactly match the packaged 24-label core"
        )
    if set(AIGUARD24_TARGET_TO_SOURCE) - set(selected_core):
        raise Aiguard24BieInitializationError("projection contains an unknown target")
    referenced_sources = {
        source for sources in AIGUARD24_TARGET_TO_SOURCE.values() for source in sources
    }
    if not referenced_sources <= set(AIGUARD_SOURCE_ENTITY_TYPES):
        raise Aiguard24BieInitializationError("projection contains an unknown source")
    return selected


def _target_config(
    source_config: Qwen3Config,
    *,
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> Qwen3Config:
    values = source_config.to_dict()
    for name in (
        "_name_or_path",
        "architectures",
        "auto_map",
        "id2label",
        "label2id",
        "num_labels",
        "pii_attention_mode",
        "pii_lineage",
        "pii_release_eligible",
        "pii_tagging_scheme",
        "pii_training_status",
    ):
        values.pop(name, None)
    values.update(
        {
            "architectures": ["Qwen3ForTokenClassification"],
            "id2label": id2label,
            "label2id": label2id,
            "num_labels": len(label2id),
            "use_cache": False,
        }
    )
    config = Qwen3Config.from_dict(values)
    config._name_or_path = ""
    config.architectures = ["Qwen3ForTokenClassification"]
    config.use_cache = False
    config.pii_attention_mode = "causal"
    config.pii_tagging_scheme = "BIE"
    config.pii_release_eligible = False
    config.pii_training_status = "initialized_untrained"
    return config


def _mean_rows(parameter: torch.Tensor, row_ids: tuple[int, ...]) -> torch.Tensor:
    indexes = torch.tensor(row_ids, dtype=torch.long, device=parameter.device)
    return parameter.index_select(0, indexes).mean(dim=0)


def initialize_aiguard24_bie(
    source_model: str | Path,
    *,
    taxonomy: Taxonomy | None = None,
    seed: int = 42,
) -> Aiguard24BieInitializationResult:
    """Initialize a standard 73-label causal Qwen3 classifier from AIguard."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Aiguard24BieInitializationError("seed must be a non-negative integer")
    root, source_hashes = _inspect_pinned_source(source_model)
    try:
        source = Qwen3ForTokenClassification.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            weights_only=True,
        )
    except Exception as exc:
        raise Aiguard24BieInitializationError(
            "pinned AIguard model failed safe local loading"
        ) from exc
    if any(parameter.device.type != "cpu" for parameter in source.parameters()):
        raise Aiguard24BieInitializationError("source model must load on CPU")
    source_id2label = _validate_source_model(source)
    source_label2id = {label: index for index, label in source_id2label.items()}
    selected_taxonomy = _validate_taxonomy(taxonomy)
    label2id, id2label = build_core_bie_label_maps(selected_taxonomy)
    if len(label2id) != 73:
        raise Aiguard24BieInitializationError("target head must contain exactly 73 labels")

    config = _target_config(source.config, label2id=label2id, id2label=id2label)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        target = Qwen3ForTokenClassification(config)
    if any(parameter.device.type != "cpu" for parameter in target.parameters()):
        raise Aiguard24BieInitializationError("target initialization unexpectedly left CPU")

    source_backbone = source.model.state_dict()
    load_result = target.model.load_state_dict(source_backbone, strict=True)
    missing_keys = tuple(sorted(load_result.missing_keys))
    unexpected_keys = tuple(sorted(load_result.unexpected_keys))
    if missing_keys or unexpected_keys:
        raise Aiguard24BieInitializationError("strict backbone copy was incompatible")
    if target.score.bias is None or source.score.bias is None:
        raise Aiguard24BieInitializationError("classifier bias is required")

    target_entities = tuple(entity.name for entity in selected_taxonomy.label_sets["core"])
    mapped_targets = tuple(
        label for label in target_entities if label in AIGUARD24_TARGET_TO_SOURCE
    )
    unmapped_targets = tuple(
        label for label in target_entities if label not in AIGUARD24_TARGET_TO_SOURCE
    )
    unmapped_ids = tuple(
        label2id[f"{prefix}-{label}"] for label in unmapped_targets for prefix in _BIE_PREFIXES
    )
    unmapped_weight = target.score.weight[list(unmapped_ids)].detach().clone()
    unmapped_bias = target.score.bias[list(unmapped_ids)].detach().clone()

    projections: list[BieHeadProjectionAudit] = []
    with torch.no_grad():
        target.score.weight[label2id["O"]].copy_(source.score.weight[source_label2id["O"]])
        target.score.bias[label2id["O"]].copy_(source.score.bias[source_label2id["O"]])
        for target_label in mapped_targets:
            source_types = AIGUARD24_TARGET_TO_SOURCE[target_label]
            mapping: dict[str, tuple[str, ...]] = {}
            for prefix in _BIE_PREFIXES:
                target_tag = f"{prefix}-{target_label}"
                source_tags = tuple(f"{prefix}-{source_type}" for source_type in source_types)
                source_ids = tuple(source_label2id[tag] for tag in source_tags)
                target.score.weight[label2id[target_tag]].copy_(
                    _mean_rows(source.score.weight, source_ids)
                )
                target.score.bias[label2id[target_tag]].copy_(
                    _mean_rows(source.score.bias, source_ids)
                )
                mapping[target_tag] = source_tags
            projections.append(
                BieHeadProjectionAudit(
                    target_label=target_label,
                    source_labels=source_types,
                    target_to_source_tags=mapping,
                )
            )

    outside_copied = bool(
        torch.equal(target.score.weight[label2id["O"]], source.score.weight[source_label2id["O"]])
        and torch.equal(target.score.bias[label2id["O"]], source.score.bias[source_label2id["O"]])
    )
    mapped_verified = True
    for projection in projections:
        for target_tag, source_tags in projection.target_to_source_tags.items():
            source_ids = tuple(source_label2id[tag] for tag in source_tags)
            mapped_verified = mapped_verified and bool(
                torch.equal(
                    target.score.weight[label2id[target_tag]],
                    _mean_rows(source.score.weight, source_ids),
                )
                and torch.equal(
                    target.score.bias[label2id[target_tag]],
                    _mean_rows(source.score.bias, source_ids),
                )
            )
    unmapped_preserved = bool(
        torch.equal(target.score.weight[list(unmapped_ids)], unmapped_weight)
        and torch.equal(target.score.bias[list(unmapped_ids)], unmapped_bias)
    )
    if not outside_copied or not mapped_verified or not unmapped_preserved:
        raise Aiguard24BieInitializationError("classifier projection failed copy invariants")

    audit = Aiguard24BieInitializationAudit(
        schema_version=1,
        strategy=AIGUARD24_BIE_INITIALIZATION_STRATEGY,
        source_model_id=AIGUARD_SOURCE_MODEL_ID,
        source_revision=AIGUARD_SOURCE_REVISION,
        source_hashes=dict(sorted(source_hashes.items())),
        source_architecture="Qwen3ForTokenClassification",
        source_label_count=len(source_id2label),
        source_entity_types=AIGUARD_SOURCE_ENTITY_TYPES,
        taxonomy_version=selected_taxonomy.taxonomy_version,
        target_architecture="Qwen3ForTokenClassification",
        target_tag_scheme="BIE",
        target_label_count=len(label2id),
        target_label2id=dict(sorted(label2id.items(), key=lambda item: item[1])),
        initialization_seed=seed,
        mapped_target_labels=mapped_targets,
        unmapped_target_labels=unmapped_targets,
        head_projections=tuple(projections),
        backbone_copy_mode="strict_state_dict",
        backbone_tensor_count=len(source_backbone),
        backbone_parameter_count=sum(int(tensor.numel()) for tensor in source_backbone.values()),
        backbone_missing_keys=missing_keys,
        backbone_unexpected_keys=unexpected_keys,
        outside_row_copied=outside_copied,
        mapped_head_rows_verified=mapped_verified,
        unmapped_head_rows_preserved=unmapped_preserved,
        attention_mode="causal",
        release_eligible=False,
    )
    target.config.pii_lineage = audit.to_dict()
    json.dumps(target.config.pii_lineage, ensure_ascii=True, sort_keys=True)
    return Aiguard24BieInitializationResult(model=target, audit=audit)


__all__ = [
    "AIGUARD24_BIE_INITIALIZATION_STRATEGY",
    "Aiguard24BieInitializationAudit",
    "Aiguard24BieInitializationError",
    "Aiguard24BieInitializationResult",
    "BieHeadProjectionAudit",
    "build_core_bie_label_maps",
    "configure_head_only_training",
    "initialize_aiguard24_bie",
]
