"""Verified AIguard-to-project-24 token-classifier initialization.

This module performs one narrowly scoped weight transition:

* load the pinned, local AIguard Qwen3 token classifier without remote code or
  pickle weights;
* copy its complete Qwen3 backbone into a standard
  :class:`~transformers.Qwen3ForTokenClassification` model;
* construct the project's ``O + 24 x B/I`` core-taxonomy head; and
* project compatible AIguard ``B/I/E`` rows into that head.

The result is an *initialization*, not a trained or release-ready model.  Its
config therefore fails closed with ``pii_release_eligible = False``.  The
returned audit contains no local path and can be embedded in a later training
manifest.
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
except ImportError as exc:  # pragma: no cover - depends on optional training extras
    raise ImportError(
        "AIguard24 initialization requires PyTorch and Transformers with Qwen3 support."
    ) from exc

from pii_zh.taxonomy import Taxonomy, load_taxonomy

AIGUARD_SOURCE_MODEL_ID = "ZJUICSR/AIguard-pii-detection-fast"
AIGUARD_SOURCE_REVISION = "677a5ebc1600fef61e8973cafd3026be322b3a73"
AIGUARD24_INITIALIZATION_STRATEGY = "aiguard_bie_to_pii_zh_core24_bio_v1"

# These hashes pin the already-downloaded Apache-2.0 source revision.  They are
# intentionally duplicated here instead of importing an experimental script:
# this module is a reusable library boundary and must not depend on scripts/.
_EXPECTED_SOURCE_HASHES: dict[str, str] = {
    "config.json": "5f8b5cafa13310bd5327e90beabc919307f749cb00714d4e87bff4d22cb31225",
    "model.safetensors": "89c9b36aa98a155ea3255eb2c53280858c49b9fed790b43107c48a53158ec4b2",
    "tokenizer.json": "352a863cd2761388ccc58f1432467ba6a1037bf12df9069889b142fa246471f6",
    "tokenizer_config.json": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
}

AIGUARD_SOURCE_ENTITY_TYPES: tuple[str, ...] = (
    "address",
    "bank_card",
    "bank_password",
    "birth_date",
    "credit_card",
    "drivers_license",
    "email",
    "ems_tracking",
    "hkmtp_pass",
    "id_card",
    "insurance_policy",
    "jd_order",
    "mobile",
    "name",
    "passport",
    "pdd_order",
    "plate_number",
    "sf_tracking",
    "social_security",
    "taobao_order",
    "yto_tracking",
)

# A tuple allows a target row to average semantically duplicate source rows.
# bank_card and credit_card are deliberately combined into one project label.
AIGUARD24_TARGET_TO_SOURCE: dict[str, tuple[str, ...]] = {
    "ADDRESS": ("address",),
    "BANK_CARD_NUMBER": ("bank_card", "credit_card"),
    "CN_RESIDENT_ID": ("id_card",),
    "DATE_OF_BIRTH": ("birth_date",),
    "DRIVER_LICENSE_NUMBER": ("drivers_license",),
    "EMAIL_ADDRESS": ("email",),
    "PASSPORT_NUMBER": ("passport",),
    "PERSON_NAME": ("name",),
    "PHONE_NUMBER": ("mobile",),
    "SECRET": ("bank_password",),
    "SOCIAL_SECURITY_NUMBER": ("social_security",),
    "VEHICLE_LICENSE_PLATE": ("plate_number",),
}


class Aiguard24InitializationError(RuntimeError):
    """Raised when the local source or weight transition fails closed."""


@dataclass(frozen=True, slots=True)
class HeadProjectionAudit:
    """One target entity's explicit source-row projection."""

    target_label: str
    target_b_tag: str
    source_b_tags: tuple[str, ...]
    target_i_tag: str
    source_i_e_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Aiguard24InitializationAudit:
    """Path-free evidence for a verified backbone/head initialization."""

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
    target_label_count: int
    target_label2id: dict[str, int]
    initialization_seed: int
    mapped_target_labels: tuple[str, ...]
    unmapped_target_labels: tuple[str, ...]
    head_projections: tuple[HeadProjectionAudit, ...]
    backbone_copy_mode: str
    backbone_tensor_count: int
    backbone_parameter_count: int
    backbone_missing_keys: tuple[str, ...]
    backbone_unexpected_keys: tuple[str, ...]
    outside_row_copied: bool
    mapped_head_rows_verified: bool
    unmapped_head_rows_preserved: bool
    release_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible value suitable for a training manifest."""

        value = json.loads(json.dumps(asdict(self), ensure_ascii=True, sort_keys=True))
        if not isinstance(value, dict):  # pragma: no cover - dataclass root is always an object
            raise Aiguard24InitializationError("initialization audit is not a JSON object")
        return value


@dataclass(frozen=True, slots=True)
class Aiguard24InitializationResult:
    """The standard Transformers model and its path-free initialization audit."""

    model: Qwen3ForTokenClassification
    audit: Aiguard24InitializationAudit


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_pinned_source(source_model: str | Path) -> tuple[Path, dict[str, str]]:
    candidate = Path(source_model).expanduser()
    if candidate.is_symlink():
        raise Aiguard24InitializationError("source model directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise Aiguard24InitializationError(
            "source model is not an available local directory"
        ) from exc
    if not root.is_dir():
        raise Aiguard24InitializationError("source model must be a local directory")

    required = tuple(sorted(_EXPECTED_SOURCE_HASHES))
    for name in required:
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file():
            raise Aiguard24InitializationError(
                "source model is missing a required regular safe artifact"
            )
    forbidden_suffixes = {".bin", ".pkl", ".pickle", ".pt", ".pth"}
    if any(item.is_file() and item.suffix.lower() in forbidden_suffixes for item in root.iterdir()):
        raise Aiguard24InitializationError("pickle-based source model artifacts are forbidden")

    observed = {name: _sha256_file(root / name) for name in required}
    if observed != dict(sorted(_EXPECTED_SOURCE_HASHES.items())):
        raise Aiguard24InitializationError(
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
            raise Aiguard24InitializationError("source id2label contains an invalid key")
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise Aiguard24InitializationError("source id2label contains an invalid key") from exc
        if label_id < 0 or not isinstance(raw_label, str) or not raw_label:
            raise Aiguard24InitializationError("source id2label contains an invalid entry")
        if label_id in normalized:
            raise Aiguard24InitializationError("source id2label contains duplicate integer keys")
        normalized[label_id] = raw_label
    if sorted(normalized) != list(range(len(normalized))):
        raise Aiguard24InitializationError("source id2label must be contiguous from zero")
    return normalized


def _validate_source_model(model: Qwen3ForTokenClassification) -> dict[int, str]:
    config = model.config
    if not isinstance(config, Qwen3Config) or config.model_type != "qwen3":
        raise Aiguard24InitializationError("source config must be native Qwen3")
    if config.architectures != ["Qwen3ForTokenClassification"]:
        raise Aiguard24InitializationError(
            "source config must declare a standalone Qwen3 token classifier"
        )
    if getattr(config, "auto_map", None):
        raise Aiguard24InitializationError("source config auto_map is forbidden")

    id2label = _normalize_id2label(config.id2label)
    expected_labels = _expected_source_labels()
    if tuple(id2label.values()) != expected_labels:
        raise Aiguard24InitializationError(
            "source config label inventory does not match the pinned AIguard protocol"
        )
    expected_label2id = {label: label_id for label_id, label in id2label.items()}
    if config.label2id != expected_label2id or int(config.num_labels) != len(expected_labels):
        raise Aiguard24InitializationError("source config label maps are not exact inverses")
    if model.score.in_features != int(config.hidden_size):
        raise Aiguard24InitializationError("source classifier input width disagrees with config")
    if model.score.out_features != len(expected_labels) or model.score.bias is None:
        raise Aiguard24InitializationError("source classifier shape is incompatible")
    return id2label


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
    config.pii_release_eligible = False
    config.pii_training_status = "initialized_untrained"
    return config


def _core_bio_label_maps(taxonomy: Taxonomy) -> tuple[dict[str, int], dict[int, str]]:
    labels = ["O"]
    for entity in taxonomy.label_sets["core"]:
        labels.extend((f"B-{entity.name}", f"I-{entity.name}"))
    label2id = {label: label_id for label_id, label in enumerate(labels)}
    return label2id, {label_id: label for label, label_id in label2id.items()}


def _mean_rows(parameter: torch.Tensor, row_ids: tuple[int, ...]) -> torch.Tensor:
    indexes = torch.tensor(row_ids, dtype=torch.long, device=parameter.device)
    return parameter.index_select(0, indexes).mean(dim=0)


def initialize_aiguard24(
    source_model: str | Path,
    *,
    taxonomy: Taxonomy | None = None,
    seed: int = 42,
) -> Aiguard24InitializationResult:
    """Initialize the project's standard 24-class BIO Qwen3 from AIguard.

    The function is deliberately local-only and CPU-only.  ``seed`` controls
    the standard Qwen initialization retained by target labels with no safe
    source mapping.  PyTorch's caller-visible random state is preserved.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Aiguard24InitializationError("seed must be a non-negative integer")
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
        raise Aiguard24InitializationError(
            "pinned AIguard model failed safe local loading"
        ) from exc
    if any(parameter.device.type != "cpu" for parameter in source.parameters()):
        raise Aiguard24InitializationError("source model must load on CPU")
    source_id2label = _validate_source_model(source)
    source_label2id = {label: label_id for label_id, label in source_id2label.items()}

    packaged_taxonomy = load_taxonomy()
    selected_taxonomy = taxonomy or packaged_taxonomy
    packaged_core = tuple(entity.name for entity in packaged_taxonomy.label_sets["core"])
    selected_core = tuple(entity.name for entity in selected_taxonomy.label_sets["core"])
    if (
        selected_taxonomy.taxonomy_version != packaged_taxonomy.taxonomy_version
        or selected_taxonomy.tagging_scheme != "BIO"
        or selected_taxonomy.outside_label != "O"
        or selected_core != packaged_core
    ):
        raise Aiguard24InitializationError(
            "target taxonomy must exactly match the packaged project core taxonomy"
        )
    label2id, id2label = _core_bio_label_maps(selected_taxonomy)
    if len(selected_taxonomy.label_sets["core"]) != 24 or len(label2id) != 49:
        raise Aiguard24InitializationError("target taxonomy must contain exactly 24 core labels")
    target_entities = selected_core
    unknown_targets = set(AIGUARD24_TARGET_TO_SOURCE) - set(target_entities)
    if unknown_targets:
        raise Aiguard24InitializationError("head projection contains a target outside the taxonomy")
    referenced_sources = {
        source_type
        for source_types in AIGUARD24_TARGET_TO_SOURCE.values()
        for source_type in source_types
    }
    if not referenced_sources <= set(AIGUARD_SOURCE_ENTITY_TYPES):
        raise Aiguard24InitializationError("head projection references an unknown source type")

    config = _target_config(source.config, label2id=label2id, id2label=id2label)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        target = Qwen3ForTokenClassification(config)
    if any(parameter.device.type != "cpu" for parameter in target.parameters()):
        raise Aiguard24InitializationError("target model initialization unexpectedly left CPU")

    source_backbone = source.model.state_dict()
    load_result = target.model.load_state_dict(source_backbone, strict=True)
    missing_keys = tuple(sorted(load_result.missing_keys))
    unexpected_keys = tuple(sorted(load_result.unexpected_keys))
    if missing_keys or unexpected_keys:
        raise Aiguard24InitializationError("strict backbone copy reported incompatible keys")

    mapped_targets = tuple(
        label for label in target_entities if label in AIGUARD24_TARGET_TO_SOURCE
    )
    unmapped_targets = tuple(
        label for label in target_entities if label not in AIGUARD24_TARGET_TO_SOURCE
    )
    untouched_row_ids = tuple(
        label2id[f"{prefix}-{label}"] for label in unmapped_targets for prefix in ("B", "I")
    )
    untouched_weight = target.score.weight[list(untouched_row_ids)].detach().clone()
    if target.score.bias is None or source.score.bias is None:
        raise Aiguard24InitializationError("classifier bias is required for the projection")
    untouched_bias = target.score.bias[list(untouched_row_ids)].detach().clone()

    projections: list[HeadProjectionAudit] = []
    with torch.no_grad():
        outside_target = label2id["O"]
        outside_source = source_label2id["O"]
        target.score.weight[outside_target].copy_(source.score.weight[outside_source])
        target.score.bias[outside_target].copy_(source.score.bias[outside_source])

        for target_label in mapped_targets:
            source_types = AIGUARD24_TARGET_TO_SOURCE[target_label]
            source_b_tags = tuple(f"B-{source_type}" for source_type in source_types)
            source_i_e_tags = tuple(
                f"{prefix}-{source_type}" for source_type in source_types for prefix in ("I", "E")
            )
            target_b_tag = f"B-{target_label}"
            target_i_tag = f"I-{target_label}"
            source_b_ids = tuple(source_label2id[tag] for tag in source_b_tags)
            source_i_e_ids = tuple(source_label2id[tag] for tag in source_i_e_tags)
            target.score.weight[label2id[target_b_tag]].copy_(
                _mean_rows(source.score.weight, source_b_ids)
            )
            target.score.bias[label2id[target_b_tag]].copy_(
                _mean_rows(source.score.bias, source_b_ids)
            )
            target.score.weight[label2id[target_i_tag]].copy_(
                _mean_rows(source.score.weight, source_i_e_ids)
            )
            target.score.bias[label2id[target_i_tag]].copy_(
                _mean_rows(source.score.bias, source_i_e_ids)
            )
            projections.append(
                HeadProjectionAudit(
                    target_label=target_label,
                    target_b_tag=target_b_tag,
                    source_b_tags=source_b_tags,
                    target_i_tag=target_i_tag,
                    source_i_e_tags=source_i_e_tags,
                )
            )

    outside_copied = bool(
        torch.equal(target.score.weight[outside_target], source.score.weight[outside_source])
        and torch.equal(target.score.bias[outside_target], source.score.bias[outside_source])
    )
    mapped_verified = True
    for projection in projections:
        source_b_ids = tuple(source_label2id[tag] for tag in projection.source_b_tags)
        source_i_e_ids = tuple(source_label2id[tag] for tag in projection.source_i_e_tags)
        mapped_verified = mapped_verified and bool(
            torch.equal(
                target.score.weight[label2id[projection.target_b_tag]],
                _mean_rows(source.score.weight, source_b_ids),
            )
            and torch.equal(
                target.score.bias[label2id[projection.target_b_tag]],
                _mean_rows(source.score.bias, source_b_ids),
            )
            and torch.equal(
                target.score.weight[label2id[projection.target_i_tag]],
                _mean_rows(source.score.weight, source_i_e_ids),
            )
            and torch.equal(
                target.score.bias[label2id[projection.target_i_tag]],
                _mean_rows(source.score.bias, source_i_e_ids),
            )
        )
    unmapped_preserved = bool(
        torch.equal(target.score.weight[list(untouched_row_ids)], untouched_weight)
        and torch.equal(target.score.bias[list(untouched_row_ids)], untouched_bias)
    )
    if not outside_copied or not mapped_verified or not unmapped_preserved:
        raise Aiguard24InitializationError("classifier projection failed its copy invariants")

    audit = Aiguard24InitializationAudit(
        schema_version=1,
        strategy=AIGUARD24_INITIALIZATION_STRATEGY,
        source_model_id=AIGUARD_SOURCE_MODEL_ID,
        source_revision=AIGUARD_SOURCE_REVISION,
        source_hashes=dict(sorted(source_hashes.items())),
        source_architecture="Qwen3ForTokenClassification",
        source_label_count=len(source_id2label),
        source_entity_types=AIGUARD_SOURCE_ENTITY_TYPES,
        taxonomy_version=selected_taxonomy.taxonomy_version,
        target_architecture="Qwen3ForTokenClassification",
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
        release_eligible=False,
    )
    # Keep the persisted lineage path-free.  A later trainer can copy this
    # exact structure into its modern training manifest/output-artifact audit.
    target.config.pii_lineage = audit.to_dict()
    # Prove the structure is JSON-compatible before returning it to callers.
    json.dumps(target.config.pii_lineage, ensure_ascii=True, sort_keys=True)
    return Aiguard24InitializationResult(model=target, audit=audit)


__all__ = [
    "AIGUARD24_INITIALIZATION_STRATEGY",
    "AIGUARD24_TARGET_TO_SOURCE",
    "AIGUARD_SOURCE_ENTITY_TYPES",
    "AIGUARD_SOURCE_MODEL_ID",
    "AIGUARD_SOURCE_REVISION",
    "Aiguard24InitializationAudit",
    "Aiguard24InitializationError",
    "Aiguard24InitializationResult",
    "HeadProjectionAudit",
    "initialize_aiguard24",
]
