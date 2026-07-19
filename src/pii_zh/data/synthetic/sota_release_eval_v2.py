"""Frozen v2 release evaluation data with an auditable v1 supersession.

The implementation deliberately keeps JSONL contents out of logs.  It verifies
and indexes the frozen 90k training/development corpus and both withdrawn v1
evaluation splits, then rejects v2 candidates on exact or normalized text,
entity-value, or document-id collision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..schema import DataPool, DocumentRecord, iter_jsonl, stable_text_hash, write_jsonl
from .sota_release_eval_v1 import (
    COLLISION_DIMENSIONS,
    FingerprintIndex,
    build_fingerprint_index,
    load_reference_fingerprint_index,
)
from .sota_release_eval_v1 import _release_templates as _shared_release_templates
from .sota_v1 import (
    DATASET_ID as BASE_DATASET_ID,
)
from .sota_v1 import (
    DATASET_VERSION as BASE_DATASET_VERSION,
)
from .sota_v1 import (
    SOTA_GENERATOR_NAME,
    SOTA_GENERATOR_VERSION,
    SyntheticSotaGenerator,
    load_synthetic_sota_config,
)
from .templates import (
    CORE_LABELS,
    CURATED_ASSET_ID,
    CURATED_ASSET_SHA256,
    CURATED_ASSET_VERSION,
    CURATION_AUDIT_SHA256,
)

DATASET_ID = "pii_zh_synthetic_sota_release_eval_v2"
DATASET_VERSION = "2.0.0"
MATERIALIZER_VERSION = "2.0.0"
DEFAULT_CONFIG_PATH = Path("configs/data/synthetic_sota_release_eval_v2.yaml")
DEFAULT_DATA_ROOT = Path(os.environ.get("PII_ZH_DATA_ROOT", "data"))
DEFAULT_TRAINING_DEVELOPMENT_DIR = (
    DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_v1"
)
DEFAULT_WITHDRAWN_V1_DIR = (
    DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_release_eval_v1"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_release_eval_v2"
)
SUPERSESSION_FREEZE_RECEIPT_PATH = Path(
    "configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json"
)
SUPERSESSION_FREEZE_RECEIPT_FILE_SHA256 = (
    "06ebdce5e288b0e7699a8225e28c4c4f0a1762e76f6967cf990b14384101f0f0"
)
CONFIG_FILE_SHA256 = "23c5c15a27c92a2110b6bd5fc180f62dced81cdc17be61b44dbc30194e5dba30"
FROZEN_AT_UTC = "2026-07-18T00:50:01Z"
GENERATION_SALT_SHA256 = "69ecd313a414459ece55274a7d7ce881537fb8f57e51940068b4a190a6fa437e"
SPLITS = ("calibration", "internal_evaluation")
FROZEN_SPLITS = {
    "calibration": {"records": 10_000, "seed": 2_026_071_811},
    "internal_evaluation": {"records": 10_000, "seed": 2_026_071_812},
}
BASE_CONFIG_SHA256 = "817127bdd96858b3ecf190b6b500661b76feaa258f0a5aef5c4ef9f8d2a2e418"
WITHDRAWN_V1_DATASET_ID = "pii_zh_synthetic_sota_release_eval_v1"
WITHDRAWN_V1_DATASET_VERSION = "1.0.0"
WITHDRAWN_V1_MANIFEST_FILE_SHA256 = (
    "aa4817080097681688948104e867ebb510be463fb87d23f9f9e3f786f58dfa7b"
)
WITHDRAWN_V1_MANIFEST_SHA256 = "f0500a5c0672b5285c8d95782d9491b6b5dd3df6a14977e6ade3f53bbd7130ca"
WITHDRAWN_V1_FILES = {
    "calibration.jsonl": {
        "records": 10_000,
        "sha256": "70f7eac06f2ee77a90b4029cd4f2e5caad97a00daa4f38b2870f60b79fe6ed72",
    },
    "internal_evaluation.jsonl": {
        "records": 10_000,
        "sha256": "5dbda1f7e4ab749e4a008f24a8d08d5202660c69226a7e4da33f4794b1acb3a7",
    },
}
FORBIDDEN_EVALUATION_SOURCES = frozenset(
    {"pii_bench_zh", "tw_pii", "cluener", "chinese_common_ner", "evaluation_error_examples"}
)
CALIBRATION_POLICY = {
    "gradient_training_allowed": False,
    "checkpoint_or_model_selection_allowed": False,
    "threshold_tuning_allowed": True,
    "temperature_scaling_allowed": True,
    "fusion_calibration_allowed": True,
    "final_metric_claim_allowed": False,
    "content_read_before_cross_seed_selection_allowed": False,
}
INTERNAL_EVALUATION_POLICY = {
    "gradient_training_allowed": False,
    "checkpoint_or_model_selection_allowed": False,
    "threshold_tuning_allowed": False,
    "temperature_scaling_allowed": False,
    "fusion_calibration_allowed": False,
    "final_metric_claim_allowed": True,
    "result_informed_model_or_threshold_change_allowed": False,
    "content_read_before_final_freeze_allowed": False,
}
COMPARISONS = (
    "calibration_vs_training_development",
    "internal_evaluation_vs_training_development",
    "calibration_vs_withdrawn_release_eval_v1",
    "internal_evaluation_vs_withdrawn_release_eval_v1",
    "calibration_vs_internal_evaluation",
)


class SyntheticSotaReleaseEvalV2Error(ValueError):
    """Raised when the frozen v2 contract cannot be verified."""


@dataclass(frozen=True, slots=True)
class ReleaseEvalV2SplitConfig:
    records: int
    seed: int


@dataclass(frozen=True, slots=True)
class SyntheticSotaReleaseEvalV2Config:
    config_path: Path
    config_sha256: str
    repository_root: Path
    base_config_path: Path
    frozen_at_utc: str
    generation_salt_sha256: str
    split_configs: Mapping[str, ReleaseEvalV2SplitConfig]
    pii_free_ratio: float
    hard_negative_ratio: float
    max_candidates_per_split: int
    usage_policy: Mapping[str, Mapping[str, bool]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return _sha256_bytes(payload)


def _exact_hash(value: str) -> str:
    return "sha256:" + _sha256_bytes(value.encode("utf-8"))


def _record_fingerprints(record: DocumentRecord) -> dict[str, str | set[str]]:
    return {
        "doc_id": record.doc_id,
        "exact_text_sha256": _exact_hash(record.text),
        "normalized_text_sha256": stable_text_hash(record.text),
        "exact_entity_value_sha256": {_exact_hash(item.text) for item in record.entities},
        "normalized_entity_value_sha256": {stable_text_hash(item.text) for item in record.entities},
    }


def _require_mapping(value: Any, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticSotaReleaseEvalV2Error(f"{name} must be an object")
    if set(value) != keys:
        raise SyntheticSotaReleaseEvalV2Error(
            f"{name} fields differ from frozen schema: "
            f"expected={sorted(keys)}, observed={sorted(value)}"
        )
    return value


def _resolve_repo_path(root: Path, raw_path: Any, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SyntheticSotaReleaseEvalV2Error(f"{name} must be a non-empty relative path")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise SyntheticSotaReleaseEvalV2Error(f"{name} must be repository-relative")
    resolved = (root / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SyntheticSotaReleaseEvalV2Error(f"{name} escapes repository") from exc
    return resolved


def _repository_relative_or_redacted(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return f"external-config:{path.name}"


def load_release_eval_v2_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> SyntheticSotaReleaseEvalV2Config:
    """Load the exact pre-validation v2 freeze file and fail closed on drift."""

    config_path = config_path.expanduser().resolve(strict=True)
    root = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    raw_bytes = config_path.read_bytes()
    if _sha256_bytes(raw_bytes) != CONFIG_FILE_SHA256:
        raise SyntheticSotaReleaseEvalV2Error("frozen v2 config file hash mismatch")
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise SyntheticSotaReleaseEvalV2Error("invalid v2 YAML") from exc
    raw = _require_mapping(
        raw,
        name="config",
        keys={
            "schema_version",
            "dataset",
            "freeze",
            "generation",
            "base_recipe",
            "reference_corpora",
            "isolation",
            "usage_policy",
            "input_policy",
        },
    )
    if raw["schema_version"] != 2:
        raise SyntheticSotaReleaseEvalV2Error("schema_version must be 2")
    dataset = _require_mapping(
        raw["dataset"],
        name="dataset",
        keys={"id", "version", "license", "language", "publication_pool", "record_data_pool"},
    )
    if dict(dataset) != {
        "id": DATASET_ID,
        "version": DATASET_VERSION,
        "license": "Apache-2.0",
        "language": "zh-Hans-CN",
        "publication_pool": DataPool.PUBLIC_RELEASE.value,
        "record_data_pool": DataPool.EVALUATION_ONLY.value,
    }:
        raise SyntheticSotaReleaseEvalV2Error("dataset identity contract changed")
    freeze = _require_mapping(
        raw["freeze"],
        name="freeze",
        keys={"frozen_at_utc", "frozen_before_any_v3_seed42_validation_result"},
    )
    if dict(freeze) != {
        "frozen_at_utc": FROZEN_AT_UTC,
        "frozen_before_any_v3_seed42_validation_result": True,
    }:
        raise SyntheticSotaReleaseEvalV2Error("freeze chronology changed")
    generation = _require_mapping(
        raw["generation"],
        name="generation",
        keys={
            "splits",
            "generation_salt_sha256",
            "pii_free_ratio",
            "hard_negative_ratio",
            "max_candidates_per_split",
            "template_policy",
        },
    )
    split_raw = _require_mapping(generation["splits"], name="generation.splits", keys=set(SPLITS))
    if tuple(split_raw) != SPLITS:
        raise SyntheticSotaReleaseEvalV2Error("split order changed")
    split_configs: dict[str, ReleaseEvalV2SplitConfig] = {}
    for split in SPLITS:
        item = _require_mapping(
            split_raw[split], name=f"generation.splits.{split}", keys={"records", "seed"}
        )
        if dict(item) != FROZEN_SPLITS[split]:
            raise SyntheticSotaReleaseEvalV2Error(f"frozen {split} recipe changed")
        split_configs[split] = ReleaseEvalV2SplitConfig(
            records=int(item["records"]), seed=int(item["seed"])
        )
    if generation["generation_salt_sha256"] != GENERATION_SALT_SHA256:
        raise SyntheticSotaReleaseEvalV2Error("generation salt changed")
    if (
        generation["pii_free_ratio"] != 0.45
        or generation["hard_negative_ratio"] != 0.45
        or generation["max_candidates_per_split"] != 120_000
        or generation["template_policy"] != "synthetic_sota_v1_train_validation_templates"
    ):
        raise SyntheticSotaReleaseEvalV2Error("frozen generation policy changed")
    base = _require_mapping(
        raw["base_recipe"],
        name="base_recipe",
        keys={"config_path", "config_sha256", "dataset_id", "dataset_version"},
    )
    if (
        base["config_sha256"] != BASE_CONFIG_SHA256
        or base["dataset_id"] != BASE_DATASET_ID
        or base["dataset_version"] != BASE_DATASET_VERSION
    ):
        raise SyntheticSotaReleaseEvalV2Error("base recipe binding changed")
    base_path = _resolve_repo_path(root, base["config_path"], name="base_recipe.config_path")
    if _sha256_file(base_path) != BASE_CONFIG_SHA256:
        raise SyntheticSotaReleaseEvalV2Error("base recipe config hash mismatch")
    load_synthetic_sota_config(base_path, repository_root=root)
    references = _require_mapping(
        raw["reference_corpora"],
        name="reference_corpora",
        keys={"training_development", "withdrawn_release_eval_v1"},
    )
    training = _require_mapping(
        references["training_development"],
        name="reference_corpora.training_development",
        keys={"dataset_id", "dataset_version", "manifest_file_sha256", "manifest_sha256", "files"},
    )
    expected_training = {
        "dataset_id": BASE_DATASET_ID,
        "dataset_version": BASE_DATASET_VERSION,
        "manifest_file_sha256": "b18ddf2510fbac8f5484ef1e4c7fa7d388453187bc32516c72181e073695544a",
        "manifest_sha256": "b1625b2ccab60dacb1e24ed96b7015f0087961b3799d57c58d2cf48c42949f63",
        "files": {
            "train.jsonl": {
                "records": 80_000,
                "sha256": "b6c85dad3ff30c4182515fb8e015b7cde04b9cea12633b5b5c7947110deaf37c",
            },
            "validation.jsonl": {
                "records": 10_000,
                "sha256": "bb33092745c705ebdbf642feb071ab99621a324fbf2ed33ebba13a6a898e3cc0",
            },
        },
    }
    if dict(training) != expected_training:
        raise SyntheticSotaReleaseEvalV2Error("training/development binding changed")
    withdrawn = _require_mapping(
        references["withdrawn_release_eval_v1"],
        name="reference_corpora.withdrawn_release_eval_v1",
        keys={"dataset_id", "dataset_version", "manifest_file_sha256", "manifest_sha256", "files"},
    )
    if dict(withdrawn) != {
        "dataset_id": WITHDRAWN_V1_DATASET_ID,
        "dataset_version": WITHDRAWN_V1_DATASET_VERSION,
        "manifest_file_sha256": WITHDRAWN_V1_MANIFEST_FILE_SHA256,
        "manifest_sha256": WITHDRAWN_V1_MANIFEST_SHA256,
        "files": WITHDRAWN_V1_FILES,
    }:
        raise SyntheticSotaReleaseEvalV2Error("withdrawn v1 binding changed")
    isolation = _require_mapping(
        raw["isolation"],
        name="isolation",
        keys={"reject_candidate_on_collision", "comparisons"},
    )
    if isolation["reject_candidate_on_collision"] != list(COLLISION_DIMENSIONS) or isolation[
        "comparisons"
    ] != list(COMPARISONS):
        raise SyntheticSotaReleaseEvalV2Error("isolation contract changed")
    usage = _require_mapping(raw["usage_policy"], name="usage_policy", keys=set(SPLITS))
    expected_usage = {
        "calibration": CALIBRATION_POLICY,
        "internal_evaluation": INTERNAL_EVALUATION_POLICY,
    }
    if {split: dict(usage[split]) for split in SPLITS} != expected_usage:
        raise SyntheticSotaReleaseEvalV2Error("usage policy changed")
    inputs = _require_mapping(
        raw["input_policy"],
        name="input_policy",
        keys={"external_dataset_inputs", "forbidden_evaluation_sources"},
    )
    if (
        inputs["external_dataset_inputs"] != []
        or set(inputs["forbidden_evaluation_sources"]) != FORBIDDEN_EVALUATION_SOURCES
    ):
        raise SyntheticSotaReleaseEvalV2Error("input policy changed")
    return SyntheticSotaReleaseEvalV2Config(
        config_path=config_path,
        config_sha256=CONFIG_FILE_SHA256,
        repository_root=root,
        base_config_path=base_path,
        frozen_at_utc=FROZEN_AT_UTC,
        generation_salt_sha256=GENERATION_SALT_SHA256,
        split_configs=split_configs,
        pii_free_ratio=0.45,
        hard_negative_ratio=0.45,
        max_candidates_per_split=120_000,
        usage_policy=expected_usage,
    )


def load_withdrawn_v1_fingerprint_index(
    withdrawn_v1_dir: Path,
) -> tuple[FingerprintIndex, dict[str, Any]]:
    """Verify/index v1 without emitting row text or entity values."""

    directory = withdrawn_v1_dir.expanduser().resolve(strict=True)
    manifest_path = directory / "dataset_manifest.json"
    if _sha256_file(manifest_path) != WITHDRAWN_V1_MANIFEST_FILE_SHA256:
        raise SyntheticSotaReleaseEvalV2Error("withdrawn v1 manifest file hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SyntheticSotaReleaseEvalV2Error("withdrawn v1 manifest cannot be read") from exc
    if not isinstance(manifest, Mapping):
        raise SyntheticSotaReleaseEvalV2Error("withdrawn v1 manifest must be an object")
    unsigned = dict(manifest)
    self_hash = unsigned.pop("manifest_sha256", None)
    if (
        self_hash != WITHDRAWN_V1_MANIFEST_SHA256
        or _canonical_json_hash(unsigned) != WITHDRAWN_V1_MANIFEST_SHA256
        or manifest.get("dataset_id") != WITHDRAWN_V1_DATASET_ID
        or manifest.get("dataset_version") != WITHDRAWN_V1_DATASET_VERSION
        or manifest.get("record_data_pool") != DataPool.EVALUATION_ONLY.value
        or manifest.get("public_weight_training_allowed") is not False
    ):
        raise SyntheticSotaReleaseEvalV2Error("withdrawn v1 manifest identity failed")
    index = FingerprintIndex.empty()
    split_counts: dict[str, int] = {}
    for filename, expected in WITHDRAWN_V1_FILES.items():
        path = directory / filename
        if _sha256_file(path) != expected["sha256"]:
            raise SyntheticSotaReleaseEvalV2Error(f"withdrawn v1 file hash mismatch: {filename}")
        expected_split = filename.removesuffix(".jsonl")
        count = 0
        for record in iter_jsonl(path):
            if (
                record.split != expected_split
                or record.data_pool is not DataPool.EVALUATION_ONLY
                or record.public_weight_training_allowed is not False
                or not record.provenance.evaluation_only
            ):
                raise SyntheticSotaReleaseEvalV2Error(
                    f"withdrawn v1 row isolation failed: {filename}"
                )
            index.add(record)
            count += 1
        if count != expected["records"]:
            raise SyntheticSotaReleaseEvalV2Error(f"withdrawn v1 count failed: {filename}")
        split_counts[expected_split] = count
    if index.record_count != 20_000:
        raise SyntheticSotaReleaseEvalV2Error("withdrawn v1 must contain exactly 20000 rows")
    return index, {
        "dataset_id": WITHDRAWN_V1_DATASET_ID,
        "dataset_version": WITHDRAWN_V1_DATASET_VERSION,
        "confirmatory_calibration_status": "withdrawn_due_to_structural_pre_read",
        "manifest_file_sha256": WITHDRAWN_V1_MANIFEST_FILE_SHA256,
        "manifest_sha256": WITHDRAWN_V1_MANIFEST_SHA256,
        "file_sha256": {name: str(spec["sha256"]) for name, spec in WITHDRAWN_V1_FILES.items()},
        "split_counts": split_counts,
        "fingerprint_counts": index.counts(),
        "fingerprint_set_sha256": index.set_digests(),
        "raw_text_or_entity_values_in_audit": False,
        "local_path_recorded": False,
    }


def _derive_seed(config: SyntheticSotaReleaseEvalV2Config, split: str, purpose: str) -> int:
    material = (
        f"{DATASET_ID}\0{config.generation_salt_sha256}\0{split}\0"
        f"{config.split_configs[split].seed}\0{purpose}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _as_evaluation_record(
    record: DocumentRecord,
    *,
    split: str,
    config: SyntheticSotaReleaseEvalV2Config,
) -> DocumentRecord:
    provenance = replace(
        record.provenance,
        evaluation_only=True,
        metadata={
            **record.provenance.metadata,
            "release_evaluation_dataset_id": DATASET_ID,
            "release_evaluation_dataset_version": DATASET_VERSION,
            "release_evaluation_split": split,
            "generation_salt_sha256": config.generation_salt_sha256,
            "weight_training_allowed_by_protocol": False,
        },
    )
    converted = replace(
        record,
        provenance=provenance,
        public_weight_training_allowed=False,
        data_pool=DataPool.EVALUATION_ONLY,
        split=split,
        metadata={
            **record.metadata,
            "synthetic_release_eval_recipe": DATASET_VERSION,
            "release_evaluation_split": split,
            "generation_salt_sha256": config.generation_salt_sha256,
            "pii_free": not record.entities,
            "usage_policy": (
                "post_cross_seed_selection_calibration_only"
                if split == "calibration"
                else "one_shot_final_internal_evaluation_only"
            ),
        },
    )
    converted.validate()
    return converted


def _generate_split(
    *,
    split: str,
    config: SyntheticSotaReleaseEvalV2Config,
    training_development_index: FingerprintIndex,
    withdrawn_v1_index: FingerprintIndex,
    accepted_index: FingerprintIndex,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    spec = config.split_configs[split]
    negative_target = math.ceil(spec.records * config.hard_negative_ratio)
    positive_target = spec.records - negative_target
    positive_templates, negative_templates = _shared_release_templates()
    if positive_target < len(positive_templates) or negative_target < len(negative_templates):
        raise SyntheticSotaReleaseEvalV2Error("split too small to exercise every template")
    derived_generation_seed = _derive_seed(config, split, "generation-v1")
    generator = SyntheticSotaGenerator(derived_generation_seed)
    accepted: list[DocumentRecord] = []
    rejection_counts: Counter[str] = Counter()
    source_rejections: dict[str, Counter[str]] = {
        "training_development": Counter(),
        "withdrawn_release_eval_v1": Counter(),
        "accepted_v2": Counter(),
    }
    generated_by_kind: Counter[str] = Counter()
    for kind, target in (("positive", positive_target), ("hard_negative", negative_target)):
        accepted_for_kind = 0
        while accepted_for_kind < target:
            remaining = target - accepted_for_kind
            if sum(generated_by_kind.values()) + remaining > config.max_candidates_per_split:
                raise SyntheticSotaReleaseEvalV2Error(
                    f"{split} exhausted bounded collision-rejection budget"
                )
            candidates = generator.generate_partition(
                positive_count=remaining if kind == "positive" else 0,
                hard_negative_count=remaining if kind == "hard_negative" else 0,
                positive_templates=positive_templates,
                hard_negative_templates=negative_templates,
            )
            generated_by_kind[kind] += len(candidates)
            for raw in candidates:
                record = _as_evaluation_record(raw, split=split, config=config)
                collisions_by_source = {
                    "training_development": set(
                        training_development_index.collision_dimensions(record)
                    ),
                    "withdrawn_release_eval_v1": set(
                        withdrawn_v1_index.collision_dimensions(record)
                    ),
                    "accepted_v2": set(accepted_index.collision_dimensions(record)),
                }
                collisions = set().union(*collisions_by_source.values())
                if collisions:
                    rejection_counts["candidate_records"] += 1
                    for dimension in sorted(collisions):
                        rejection_counts[dimension] += 1
                    for source, source_dimensions in collisions_by_source.items():
                        if source_dimensions:
                            source_rejections[source]["candidate_records"] += 1
                            for dimension in sorted(source_dimensions):
                                source_rejections[source][dimension] += 1
                    continue
                accepted_index.add(record)
                accepted.append(record)
                accepted_for_kind += 1
    ordering_seed = _derive_seed(config, split, "ordering-v1")
    random.Random(ordering_seed).shuffle(accepted)
    rejection_dimensions = ("candidate_records", *COLLISION_DIMENSIONS)
    return accepted, {
        "declared_seed": spec.seed,
        "derived_generation_seed": derived_generation_seed,
        "generation_salt_sha256": config.generation_salt_sha256,
        "seed_derivation_algorithm": "sha256-namespace-first-u64-be-v1",
        "ordering_algorithm": "deterministic-whole-split-shuffle-v2",
        "derived_ordering_seed": ordering_seed,
        "requested_records": spec.records,
        "accepted_records": len(accepted),
        "accepted_positive": positive_target,
        "accepted_pii_free": negative_target,
        "accepted_hard_negative": negative_target,
        "pii_free_ratio": negative_target / spec.records,
        "hard_negative_ratio": negative_target / spec.records,
        "generated_candidates_by_kind": dict(sorted(generated_by_kind.items())),
        "generated_candidates_total": sum(generated_by_kind.values()),
        "collision_rejections": {
            dimension: rejection_counts.get(dimension, 0) for dimension in rejection_dimensions
        },
        "collision_rejections_by_reference": {
            source: {dimension: counts.get(dimension, 0) for dimension in rejection_dimensions}
            for source, counts in source_rejections.items()
        },
        "max_candidates_per_split": config.max_candidates_per_split,
        "positive_template_count": len(positive_templates),
        "variable_hard_negative_template_count": len(negative_templates),
        "static_hard_negative_policy": "excluded_due_to_reference_exact_text_collision",
    }


def _intersection_counts(left: FingerprintIndex, right: FingerprintIndex) -> dict[str, int]:
    return {
        dimension: len(getattr(left, dimension).intersection(getattr(right, dimension)))
        for dimension in COLLISION_DIMENSIONS
    }


def audit_release_eval_v2_isolation(
    records: Sequence[DocumentRecord],
    *,
    training_development_index: FingerprintIndex,
    withdrawn_v1_index: FingerprintIndex,
) -> dict[str, Any]:
    """Recompute every frozen exact/normalized cross-corpus comparison."""

    split_records = {
        split: [record for record in records if record.split == split] for split in SPLITS
    }
    if sum(map(len, split_records.values())) != len(records):
        raise SyntheticSotaReleaseEvalV2Error("records contain unsupported split")
    indexes = {split: build_fingerprint_index(items) for split, items in split_records.items()}
    comparisons = {
        "calibration_vs_training_development": _intersection_counts(
            indexes["calibration"], training_development_index
        ),
        "internal_evaluation_vs_training_development": _intersection_counts(
            indexes["internal_evaluation"], training_development_index
        ),
        "calibration_vs_withdrawn_release_eval_v1": _intersection_counts(
            indexes["calibration"], withdrawn_v1_index
        ),
        "internal_evaluation_vs_withdrawn_release_eval_v1": _intersection_counts(
            indexes["internal_evaluation"], withdrawn_v1_index
        ),
        "calibration_vs_internal_evaluation": _intersection_counts(
            indexes["calibration"], indexes["internal_evaluation"]
        ),
    }
    nonzero = {
        name: {dimension: count for dimension, count in values.items() if count}
        for name, values in comparisons.items()
        if any(values.values())
    }
    if nonzero:
        raise SyntheticSotaReleaseEvalV2Error(
            f"release-evaluation v2 collision audit failed: {nonzero}"
        )
    return {
        "algorithm": "exact-and-normalized-cross-corpus-fingerprint-v2",
        "required_exact_dimensions": [
            "exact_text_sha256",
            "exact_entity_value_sha256",
            "doc_id",
        ],
        "additional_stricter_dimensions": [
            "normalized_text_sha256",
            "normalized_entity_value_sha256",
        ],
        "comparisons": comparisons,
        "all_collision_counts_zero": True,
        "split_fingerprint_counts": {split: indexes[split].counts() for split in SPLITS},
        "split_fingerprint_set_sha256": {split: indexes[split].set_digests() for split in SPLITS},
        "raw_text_or_entity_values_in_audit": False,
    }


def build_release_eval_v2_records(
    config: SyntheticSotaReleaseEvalV2Config,
    *,
    training_development_index: FingerprintIndex,
    withdrawn_v1_index: FingerprintIndex,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    accepted_index = FingerprintIndex.empty()
    records: list[DocumentRecord] = []
    generation: dict[str, Any] = {}
    for split in SPLITS:
        selected, split_audit = _generate_split(
            split=split,
            config=config,
            training_development_index=training_development_index,
            withdrawn_v1_index=withdrawn_v1_index,
            accepted_index=accepted_index,
        )
        records.extend(selected)
        generation[split] = split_audit
    label_counts = {split: Counter[str]() for split in SPLITS}
    split_counts: Counter[str] = Counter()
    pii_free_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    for record in records:
        record.validate()
        if (
            record.data_pool is not DataPool.EVALUATION_ONLY
            or record.public_weight_training_allowed is not False
            or not record.provenance.evaluation_only
        ):
            raise SyntheticSotaReleaseEvalV2Error("v2 row escaped evaluation-only isolation")
        split = str(record.split)
        split_counts[split] += 1
        label_counts[split].update(item.label for item in record.entities)
        if not record.entities:
            pii_free_counts[split] += 1
        if record.metadata.get("hard_negative") is True:
            hard_negative_counts[split] += 1
    for split in SPLITS:
        expected = config.split_configs[split].records
        expected_negative = math.ceil(expected * config.pii_free_ratio)
        if split_counts[split] != expected:
            raise SyntheticSotaReleaseEvalV2Error(f"{split} count failed")
        if (
            pii_free_counts[split] != expected_negative
            or hard_negative_counts[split] != expected_negative
        ):
            raise SyntheticSotaReleaseEvalV2Error(f"{split} PII-free ratio failed")
        if set(label_counts[split]) != set(CORE_LABELS):
            raise SyntheticSotaReleaseEvalV2Error(f"{split} lacks core labels")
    isolation = audit_release_eval_v2_isolation(
        records,
        training_development_index=training_development_index,
        withdrawn_v1_index=withdrawn_v1_index,
    )
    return records, {
        "generation_by_split": generation,
        "counts_by_split": dict(split_counts),
        "pii_free_counts_by_split": dict(pii_free_counts),
        "hard_negative_counts_by_split": dict(hard_negative_counts),
        "label_counts_by_split": {
            split: dict(sorted(label_counts[split].items())) for split in SPLITS
        },
        "core_label_count": len(CORE_LABELS),
        "cross_corpus_isolation": isolation,
    }


def _write_split(
    records: Sequence[DocumentRecord], *, split: str, destination: Path
) -> dict[str, Any]:
    selected = [record for record in records if record.split == split]
    written = write_jsonl(selected, destination)
    destination.chmod(0o444)
    verified = 0
    for record in iter_jsonl(destination):
        if (
            record.split != split
            or record.data_pool is not DataPool.EVALUATION_ONLY
            or record.public_weight_training_allowed is not False
            or not record.provenance.evaluation_only
        ):
            raise SyntheticSotaReleaseEvalV2Error(f"{split} readback isolation failed")
        verified += 1
    if written != verified or verified != len(selected):
        raise SyntheticSotaReleaseEvalV2Error(f"{split} readback count failed")
    return {
        "name": destination.name,
        "split": split,
        "records": verified,
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "mode": "0444",
    }


def _verify_freeze_receipt(root: Path) -> dict[str, Any]:
    path = (root / SUPERSESSION_FREEZE_RECEIPT_PATH).resolve(strict=True)
    if _sha256_file(path) != SUPERSESSION_FREEZE_RECEIPT_FILE_SHA256:
        raise SyntheticSotaReleaseEvalV2Error("supersession freeze receipt hash mismatch")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SyntheticSotaReleaseEvalV2Error("freeze receipt cannot be read") from exc
    chronology = receipt.get("chronology", {})
    frozen = receipt.get("frozen_design", {})
    if (
        receipt.get("receipt_id") != "synthetic_sota_release_eval_v2_supersession_freeze"
        or receipt.get("frozen_at_utc") != FROZEN_AT_UTC
        or chronology
        != {
            "frozen_before_any_v3_seed42_validation_result": True,
            "v1_structural_pre_read_one_record_no_model_result": True,
            "v1_confirmatory_calibration_withdrawn": True,
            "v2_zero_content_read_before_cross_seed_selection": True,
        }
        or frozen.get("config_file_sha256") != CONFIG_FILE_SHA256
        or frozen.get("generation_salt_sha256") != GENERATION_SALT_SHA256
    ):
        raise SyntheticSotaReleaseEvalV2Error("freeze receipt semantics changed")
    return {
        "path": str(SUPERSESSION_FREEZE_RECEIPT_PATH),
        "file_sha256": SUPERSESSION_FREEZE_RECEIPT_FILE_SHA256,
        "receipt_id": receipt["receipt_id"],
        "frozen_at_utc": FROZEN_AT_UTC,
        "chronology": chronology,
    }


def _build_supersession_receipt(
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    freeze_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "receipt_schema_version": 1,
        "receipt_id": "synthetic_sota_release_eval_v2_supersession_materialization",
        "frozen_at_utc": FROZEN_AT_UTC,
        "freeze_receipt": dict(freeze_receipt),
        "chronology": {
            "v1_structural_pre_read_one_record_no_model_result": True,
            "v1_confirmatory_calibration_withdrawn": True,
            "v2_zero_content_read_before_cross_seed_selection": True,
            "frozen_before_any_v3_seed42_validation_result": True,
        },
        "replacement": {
            "withdrawn_dataset_id": WITHDRAWN_V1_DATASET_ID,
            "withdrawn_dataset_version": WITHDRAWN_V1_DATASET_VERSION,
            "replacement_dataset_id": DATASET_ID,
            "replacement_dataset_version": DATASET_VERSION,
            "config_file_sha256": CONFIG_FILE_SHA256,
            "generation_salt_sha256": GENERATION_SALT_SHA256,
            "dataset_manifest_file_sha256": manifest_file_sha256,
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "files": manifest["files"],
        },
        "verification": {
            "record_data_pool": "evaluation_only",
            "public_weight_training_allowed": False,
            "calibration_records": manifest["counts"]["by_split"]["calibration"],
            "internal_evaluation_records": manifest["counts"]["by_split"]["internal_evaluation"],
            "pii_free_counts_by_split": manifest["audits"]["pii_free_counts_by_split"],
            "hard_negative_counts_by_split": manifest["audits"]["hard_negative_counts_by_split"],
            "core_label_count_each_split": manifest["audits"]["core_label_count"],
            "all_required_collision_counts_zero": True,
            "jsonl_rows_printed_by_materializer": False,
            "protected_public_benchmark_read": False,
        },
        "protocol_effect": {
            "v1_zero_read_confirmatory_role_available": False,
            "v2_calibration_available_only_after_cross_seed_selection": True,
            "v2_internal_evaluation_is_one_shot_after_full_freeze": True,
            "frozen_v3_training_protocol_or_selector_modified": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_json_hash(receipt)
    return receipt


def materialize_release_eval_v2_corpus(
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    training_development_dir: Path = DEFAULT_TRAINING_DEVELOPMENT_DIR,
    withdrawn_v1_dir: Path = DEFAULT_WITHDRAWN_V1_DIR,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize immutable v2 JSONL, manifest, and supersession receipt."""

    config = load_release_eval_v2_config(config_path, repository_root=repository_root)
    freeze_receipt = _verify_freeze_receipt(config.repository_root)
    training_index, training_audit = load_reference_fingerprint_index(training_development_dir)
    withdrawn_index, withdrawn_audit = load_withdrawn_v1_fingerprint_index(withdrawn_v1_dir)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(mode=0o755)
    try:
        records, audit = build_release_eval_v2_records(
            config,
            training_development_index=training_index,
            withdrawn_v1_index=withdrawn_index,
        )
        files = [
            _write_split(records, split=split, destination=staging / f"{split}.jsonl")
            for split in SPLITS
        ]
        manifest: dict[str, Any] = {
            "manifest_schema_version": 2,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "license": "Apache-2.0",
            "language": "zh-Hans-CN",
            "publication_pool": DataPool.PUBLIC_RELEASE.value,
            "record_data_pool": DataPool.EVALUATION_ONLY.value,
            "public_weight_training_allowed": False,
            "public_artifact_release_allowed": True,
            "freeze": {
                "frozen_at_utc": config.frozen_at_utc,
                "frozen_before_any_v3_seed42_validation_result": True,
                "supersession_freeze_receipt": freeze_receipt,
            },
            "config": {
                "path": _repository_relative_or_redacted(
                    config.config_path, config.repository_root
                ),
                "sha256": config.config_sha256,
            },
            "generation": {
                "generator_name": SOTA_GENERATOR_NAME,
                "generator_version": SOTA_GENERATOR_VERSION,
                "materializer_version": MATERIALIZER_VERSION,
                "declared_split_seeds": {
                    split: config.split_configs[split].seed for split in SPLITS
                },
                "generation_salt_sha256": config.generation_salt_sha256,
                "seed_derivation_algorithm": "sha256-namespace-first-u64-be-v1",
                "seeds_and_salt_independent_from_v1": True,
                "requested_split_counts": {
                    split: config.split_configs[split].records for split in SPLITS
                },
                "requested_pii_free_ratio": config.pii_free_ratio,
                "requested_hard_negative_ratio": config.hard_negative_ratio,
                "contains_real_personal_data": False,
                "network_used": False,
                "gpu_used": False,
                "external_dataset_inputs": [],
                "protected_evaluation_data_read": False,
                "jsonl_rows_printed": False,
            },
            "source": {
                "source_id": "repo_curated_synthetic_templates",
                "template_asset_id": CURATED_ASSET_ID,
                "template_asset_version": CURATED_ASSET_VERSION,
                "template_asset_sha256": CURATED_ASSET_SHA256,
                "curation_audit_sha256": CURATION_AUDIT_SHA256,
                "base_recipe_dataset_id": BASE_DATASET_ID,
                "base_recipe_dataset_version": BASE_DATASET_VERSION,
                "base_recipe_config_sha256": BASE_CONFIG_SHA256,
                "declared_license": "Apache-2.0",
            },
            "reference_corpora": {
                "training_development": training_audit,
                "withdrawn_release_eval_v1": withdrawn_audit,
            },
            "supersession": {
                "v1_structural_pre_read_one_record_no_model_result": True,
                "v1_confirmatory_calibration_withdrawn": True,
                "v2_zero_content_read_before_cross_seed_selection": True,
                "replacement_scope": "confirmatory_calibration_and_internal_evaluation",
            },
            "usage_policy": {split: dict(config.usage_policy[split]) for split in SPLITS},
            "input_policy": {
                "external_dataset_inputs": [],
                "synthetic_reference_inputs": [BASE_DATASET_ID, WITHDRAWN_V1_DATASET_ID],
                "forbidden_evaluation_sources": sorted(FORBIDDEN_EVALUATION_SOURCES),
                "forbidden_sources_read": [],
            },
            "counts": {
                "total": len(records),
                "by_split": {split: config.split_configs[split].records for split in SPLITS},
                "pii_free": sum(1 for record in records if not record.entities),
                "hard_negative": sum(
                    1 for record in records if record.metadata.get("hard_negative") is True
                ),
                "core_labels": len(CORE_LABELS),
            },
            "audits": audit,
            "files": files,
        }
        manifest["manifest_sha256"] = _canonical_json_hash(manifest)
        manifest_path = staging / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o444)
        receipt = _build_supersession_receipt(
            manifest=manifest,
            manifest_file_sha256=_sha256_file(manifest_path),
            freeze_receipt=freeze_receipt,
        )
        receipt_path = staging / "supersession_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_path.chmod(0o444)
        staging.chmod(0o555)
        os.replace(staging, output_dir)
        return {"dataset_manifest": manifest, "supersession_receipt": receipt}
    except Exception:
        if staging.exists():
            staging.chmod(0o755)
            for child in staging.iterdir():
                child.chmod(0o644)
            shutil.rmtree(staging)
        raise


def _binary_line_count(path: Path) -> int:
    """Count rows without decoding, parsing, or emitting JSONL content."""

    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def validate_release_eval_v2_artifacts(
    output_dir: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly validate hashes/counts/policies without parsing JSONL rows."""

    root = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    load_release_eval_v2_config(root / DEFAULT_CONFIG_PATH, repository_root=root)
    freeze_receipt = _verify_freeze_receipt(root)
    directory = output_dir.expanduser().resolve(strict=True)
    manifest_path = directory / "dataset_manifest.json"
    receipt_path = directory / "supersession_receipt.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SyntheticSotaReleaseEvalV2Error("v2 metadata artifact cannot be read") from exc
    if not isinstance(manifest, Mapping) or not isinstance(receipt, Mapping):
        raise SyntheticSotaReleaseEvalV2Error("v2 metadata artifacts must be objects")
    unsigned_manifest = dict(manifest)
    manifest_self_hash = unsigned_manifest.pop("manifest_sha256", None)
    if (
        manifest_self_hash != _canonical_json_hash(unsigned_manifest)
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("record_data_pool") != DataPool.EVALUATION_ONLY.value
        or manifest.get("public_weight_training_allowed") is not False
        or manifest.get("config", {}).get("sha256") != CONFIG_FILE_SHA256
        or manifest.get("freeze", {}).get("supersession_freeze_receipt") != freeze_receipt
    ):
        raise SyntheticSotaReleaseEvalV2Error("v2 manifest identity/self-hash failed")
    expected_counts = {"calibration": 10_000, "internal_evaluation": 10_000}
    audits = manifest.get("audits", {})
    isolation = audits.get("cross_corpus_isolation", {})
    if (
        manifest.get("counts", {}).get("by_split") != expected_counts
        or manifest.get("counts", {}).get("total") != 20_000
        or manifest.get("counts", {}).get("pii_free") != 9_000
        or manifest.get("counts", {}).get("hard_negative") != 9_000
        or manifest.get("counts", {}).get("core_labels") != 24
        or audits.get("pii_free_counts_by_split")
        != {"calibration": 4_500, "internal_evaluation": 4_500}
        or audits.get("hard_negative_counts_by_split")
        != {"calibration": 4_500, "internal_evaluation": 4_500}
        or audits.get("core_label_count") != 24
        or any(
            set(audits.get("label_counts_by_split", {}).get(split, {})) != set(CORE_LABELS)
            for split in SPLITS
        )
        or isolation.get("all_collision_counts_zero") is not True
        or set(isolation.get("comparisons", {})) != set(COMPARISONS)
        or any(
            values != {dimension: 0 for dimension in COLLISION_DIMENSIONS}
            for values in isolation.get("comparisons", {}).values()
        )
    ):
        raise SyntheticSotaReleaseEvalV2Error("v2 counts/labels/isolation audit failed")
    file_specs = manifest.get("files")
    if not isinstance(file_specs, list) or len(file_specs) != 2:
        raise SyntheticSotaReleaseEvalV2Error("v2 manifest must bind two JSONL files")
    observed_files: dict[str, dict[str, Any]] = {}
    for item in file_specs:
        if not isinstance(item, Mapping):
            raise SyntheticSotaReleaseEvalV2Error("invalid v2 file entry")
        name = item.get("name")
        split = item.get("split")
        if name != f"{split}.jsonl" or split not in SPLITS:
            raise SyntheticSotaReleaseEvalV2Error("v2 file/split binding failed")
        path = directory / str(name)
        if (
            item.get("records") != expected_counts[str(split)]
            or _binary_line_count(path) != expected_counts[str(split)]
            or _sha256_file(path) != item.get("sha256")
            or path.stat().st_size != item.get("bytes")
            or path.stat().st_mode & 0o777 != 0o444
        ):
            raise SyntheticSotaReleaseEvalV2Error(f"v2 file verification failed: {name}")
        observed_files[str(split)] = dict(item)
    if set(observed_files) != set(SPLITS):
        raise SyntheticSotaReleaseEvalV2Error("v2 split files incomplete")
    unsigned_receipt = dict(receipt)
    receipt_self_hash = unsigned_receipt.pop("receipt_sha256", None)
    replacement = receipt.get("replacement", {})
    verification = receipt.get("verification", {})
    if (
        receipt_self_hash != _canonical_json_hash(unsigned_receipt)
        or receipt.get("receipt_id")
        != "synthetic_sota_release_eval_v2_supersession_materialization"
        or receipt.get("freeze_receipt") != freeze_receipt
        or receipt.get("chronology")
        != {
            "v1_structural_pre_read_one_record_no_model_result": True,
            "v1_confirmatory_calibration_withdrawn": True,
            "v2_zero_content_read_before_cross_seed_selection": True,
            "frozen_before_any_v3_seed42_validation_result": True,
        }
        or replacement.get("dataset_manifest_file_sha256") != _sha256_file(manifest_path)
        or replacement.get("dataset_manifest_sha256") != manifest_self_hash
        or replacement.get("files") != file_specs
        or verification.get("record_data_pool") != "evaluation_only"
        or verification.get("public_weight_training_allowed") is not False
        or verification.get("all_required_collision_counts_zero") is not True
        or verification.get("jsonl_rows_printed_by_materializer") is not False
        or verification.get("protected_public_benchmark_read") is not False
        or directory.stat().st_mode & 0o777 != 0o555
        or manifest_path.stat().st_mode & 0o777 != 0o444
        or receipt_path.stat().st_mode & 0o777 != 0o444
    ):
        raise SyntheticSotaReleaseEvalV2Error("supersession receipt validation failed")
    return {
        "status": "VALID",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "records": 20_000,
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_sha256": manifest_self_hash,
        "supersession_receipt_file_sha256": _sha256_file(receipt_path),
        "supersession_receipt_sha256": receipt_self_hash,
        "all_collision_counts_zero": True,
        "jsonl_content_parsed_or_printed": False,
    }


__all__ = [
    "COLLISION_DIMENSIONS",
    "CONFIG_FILE_SHA256",
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TRAINING_DEVELOPMENT_DIR",
    "DEFAULT_WITHDRAWN_V1_DIR",
    "FROZEN_AT_UTC",
    "GENERATION_SALT_SHA256",
    "ReleaseEvalV2SplitConfig",
    "SPLITS",
    "SyntheticSotaReleaseEvalV2Config",
    "SyntheticSotaReleaseEvalV2Error",
    "audit_release_eval_v2_isolation",
    "build_release_eval_v2_records",
    "load_release_eval_v2_config",
    "load_withdrawn_v1_fingerprint_index",
    "materialize_release_eval_v2_corpus",
    "validate_release_eval_v2_artifacts",
]
