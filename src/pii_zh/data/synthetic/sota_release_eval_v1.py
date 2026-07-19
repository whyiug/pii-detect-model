"""Frozen, benchmark-independent release evaluation data for AIguard24.

The corpus contains an explicitly tunable calibration split and a one-shot
internal-evaluation split.  Both are generated from independent seeds, routed
to ``evaluation_only``, and rejected on any text, entity-value, or document-id
collision with the frozen ``synthetic_sota_v1`` train/validation corpus.

No public benchmark, network service, GPU, or real personal data is read.
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
from .allocation import TEMPLATE_SPLITS
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
    HARD_NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    SyntheticTemplate,
)

DATASET_ID = "pii_zh_synthetic_sota_release_eval_v1"
DATASET_VERSION = "1.0.0"
MATERIALIZER_VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = Path("configs/data/synthetic_sota_release_eval_v1.yaml")
DEFAULT_DATA_ROOT = Path(os.environ.get("PII_ZH_DATA_ROOT", "data"))
DEFAULT_REFERENCE_DIR = DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_v1"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_release_eval_v1"
)
SPLITS = ("calibration", "internal_evaluation")
COLLISION_DIMENSIONS = (
    "exact_text_sha256",
    "normalized_text_sha256",
    "exact_entity_value_sha256",
    "normalized_entity_value_sha256",
    "doc_id",
)
FORBIDDEN_EVALUATION_SOURCES = frozenset(
    {
        "pii_bench_zh",
        "tw_pii",
        "cluener",
        "chinese_common_ner",
        "evaluation_error_examples",
    }
)
BASE_CONFIG_SHA256 = "817127bdd96858b3ecf190b6b500661b76feaa258f0a5aef5c4ef9f8d2a2e418"
REFERENCE_MANIFEST_FILE_SHA256 = "b18ddf2510fbac8f5484ef1e4c7fa7d388453187bc32516c72181e073695544a"
REFERENCE_MANIFEST_SHA256 = "b1625b2ccab60dacb1e24ed96b7015f0087961b3799d57c58d2cf48c42949f63"
REFERENCE_FILES = {
    "train.jsonl": {
        "records": 80_000,
        "sha256": "b6c85dad3ff30c4182515fb8e015b7cde04b9cea12633b5b5c7947110deaf37c",
    },
    "validation.jsonl": {
        "records": 10_000,
        "sha256": "bb33092745c705ebdbf642feb071ab99621a324fbf2ed33ebba13a6a898e3cc0",
    },
}
CALIBRATION_POLICY = {
    "gradient_training_allowed": False,
    "checkpoint_or_model_selection_allowed": False,
    "threshold_tuning_allowed": True,
    "temperature_scaling_allowed": True,
    "fusion_calibration_allowed": True,
    "final_metric_claim_allowed": False,
}
INTERNAL_EVALUATION_POLICY = {
    "gradient_training_allowed": False,
    "checkpoint_or_model_selection_allowed": False,
    "threshold_tuning_allowed": False,
    "temperature_scaling_allowed": False,
    "fusion_calibration_allowed": False,
    "final_metric_claim_allowed": True,
    "result_informed_model_or_threshold_change_allowed": False,
}


class SyntheticSotaReleaseEvalError(ValueError):
    """Raised when the release-evaluation corpus cannot fail closed."""


@dataclass(frozen=True, slots=True)
class ReleaseEvalSplitConfig:
    records: int
    seed: int


@dataclass(frozen=True, slots=True)
class SyntheticSotaReleaseEvalConfig:
    config_path: Path
    config_sha256: str
    repository_root: Path
    base_config_path: Path
    split_configs: Mapping[str, ReleaseEvalSplitConfig]
    pii_free_ratio: float
    hard_negative_ratio: float
    max_candidates_per_split: int
    usage_policy: Mapping[str, Mapping[str, bool]]


@dataclass(slots=True)
class FingerprintIndex:
    """Text-free collision index for one or more canonical corpora."""

    doc_id: set[str]
    exact_text_sha256: set[str]
    normalized_text_sha256: set[str]
    exact_entity_value_sha256: set[str]
    normalized_entity_value_sha256: set[str]
    record_count: int = 0
    entity_occurrence_count: int = 0

    @classmethod
    def empty(cls) -> FingerprintIndex:
        return cls(set(), set(), set(), set(), set())

    def collision_dimensions(self, record: DocumentRecord) -> tuple[str, ...]:
        values = _record_fingerprints(record)
        collisions: list[str] = []
        for dimension in COLLISION_DIMENSIONS:
            observed = getattr(self, dimension)
            candidate = values[dimension]
            if isinstance(candidate, set):
                if observed.intersection(candidate):
                    collisions.append(dimension)
            elif candidate in observed:
                collisions.append(dimension)
        return tuple(collisions)

    def add(self, record: DocumentRecord, *, require_unique: bool = True) -> None:
        collisions = self.collision_dimensions(record)
        if require_unique and collisions:
            raise SyntheticSotaReleaseEvalError(
                "duplicate record fingerprints: " + ", ".join(collisions)
            )
        values = _record_fingerprints(record)
        self.doc_id.add(str(values["doc_id"]))
        self.exact_text_sha256.add(str(values["exact_text_sha256"]))
        self.normalized_text_sha256.add(str(values["normalized_text_sha256"]))
        exact_entities = values["exact_entity_value_sha256"]
        normalized_entities = values["normalized_entity_value_sha256"]
        assert isinstance(exact_entities, set)
        assert isinstance(normalized_entities, set)
        self.exact_entity_value_sha256.update(exact_entities)
        self.normalized_entity_value_sha256.update(normalized_entities)
        self.record_count += 1
        self.entity_occurrence_count += len(record.entities)

    def counts(self) -> dict[str, int]:
        return {
            "records": self.record_count,
            "entity_occurrences": self.entity_occurrence_count,
            "doc_id": len(self.doc_id),
            "exact_text_sha256": len(self.exact_text_sha256),
            "normalized_text_sha256": len(self.normalized_text_sha256),
            "exact_entity_value_sha256": len(self.exact_entity_value_sha256),
            "normalized_entity_value_sha256": len(self.normalized_entity_value_sha256),
        }

    def set_digests(self) -> dict[str, str]:
        return {
            dimension: _canonical_json_hash(sorted(getattr(self, dimension)))
            for dimension in COLLISION_DIMENSIONS
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_hash(value: str) -> str:
    return "sha256:" + _sha256_bytes(value.encode("utf-8"))


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _repository_relative_or_redacted(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return f"external-config:{path.name}"


def _require_mapping(value: Any, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticSotaReleaseEvalError(f"{name} must be an object")
    if set(value) != keys:
        raise SyntheticSotaReleaseEvalError(
            f"{name} fields differ from the frozen schema: "
            f"expected={sorted(keys)}, observed={sorted(value)}"
        )
    return value


def _resolve_repository_path(repository_root: Path, raw_path: Any, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SyntheticSotaReleaseEvalError(f"{name} must be a non-empty relative path")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise SyntheticSotaReleaseEvalError(f"{name} must be repository-relative")
    resolved = (repository_root / candidate).resolve(strict=True)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise SyntheticSotaReleaseEvalError(f"{name} escapes the repository") from exc
    return resolved


def _record_fingerprints(record: DocumentRecord) -> dict[str, str | set[str]]:
    return {
        "doc_id": record.doc_id,
        "exact_text_sha256": _exact_hash(record.text),
        "normalized_text_sha256": stable_text_hash(record.text),
        "exact_entity_value_sha256": {_exact_hash(entity.text) for entity in record.entities},
        "normalized_entity_value_sha256": {
            stable_text_hash(entity.text) for entity in record.entities
        },
    }


def load_release_eval_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> SyntheticSotaReleaseEvalConfig:
    """Load and strictly validate the frozen release-evaluation recipe."""

    config_path = config_path.expanduser().resolve(strict=True)
    root = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    raw_bytes = config_path.read_bytes()
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise SyntheticSotaReleaseEvalError("invalid release-evaluation YAML") from exc
    raw = _require_mapping(
        raw,
        name="config",
        keys={
            "schema_version",
            "dataset",
            "generation",
            "base_recipe",
            "reference_corpus",
            "isolation",
            "usage_policy",
            "input_policy",
        },
    )
    if raw["schema_version"] != 1:
        raise SyntheticSotaReleaseEvalError("schema_version must be 1")

    dataset = _require_mapping(
        raw["dataset"],
        name="dataset",
        keys={
            "id",
            "version",
            "license",
            "language",
            "publication_pool",
            "record_data_pool",
        },
    )
    if dict(dataset) != {
        "id": DATASET_ID,
        "version": DATASET_VERSION,
        "license": "Apache-2.0",
        "language": "zh-Hans-CN",
        "publication_pool": DataPool.PUBLIC_RELEASE.value,
        "record_data_pool": DataPool.EVALUATION_ONLY.value,
    }:
        raise SyntheticSotaReleaseEvalError("dataset identity or isolation contract changed")

    generation = _require_mapping(
        raw["generation"],
        name="generation",
        keys={
            "splits",
            "pii_free_ratio",
            "hard_negative_ratio",
            "max_candidates_per_split",
            "template_policy",
        },
    )
    split_raw = _require_mapping(
        generation["splits"],
        name="generation.splits",
        keys=set(SPLITS),
    )
    if tuple(split_raw) != SPLITS:
        raise SyntheticSotaReleaseEvalError("split order must be calibration then evaluation")
    split_configs: dict[str, ReleaseEvalSplitConfig] = {}
    for split in SPLITS:
        item = _require_mapping(
            split_raw[split],
            name=f"generation.splits.{split}",
            keys={"records", "seed"},
        )
        count = item["records"]
        seed = item["seed"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != 10_000
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise SyntheticSotaReleaseEvalError(
                f"{split} must pin 10000 records and an integer seed"
            )
        split_configs[split] = ReleaseEvalSplitConfig(records=count, seed=seed)
    if len({item.seed for item in split_configs.values()}) != len(SPLITS):
        raise SyntheticSotaReleaseEvalError("release-evaluation seeds must be distinct")
    if 20260717 in {item.seed for item in split_configs.values()}:
        raise SyntheticSotaReleaseEvalError("release seeds must differ from the base corpus seed")
    if generation["pii_free_ratio"] != 0.45 or generation["hard_negative_ratio"] != 0.45:
        raise SyntheticSotaReleaseEvalError("both PII-free ratios must remain frozen at 0.45")
    maximum = generation["max_candidates_per_split"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum != 100_000:
        raise SyntheticSotaReleaseEvalError("max candidate budget must remain 100000")
    if generation["template_policy"] != "synthetic_sota_v1_train_validation_templates":
        raise SyntheticSotaReleaseEvalError("template policy changed")

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
        raise SyntheticSotaReleaseEvalError("base recipe binding changed")
    base_path = _resolve_repository_path(root, base["config_path"], name="base_recipe.path")
    if _sha256_file(base_path) != BASE_CONFIG_SHA256:
        raise SyntheticSotaReleaseEvalError("base recipe config hash mismatch")
    load_synthetic_sota_config(base_path, repository_root=root)

    reference = _require_mapping(
        raw["reference_corpus"],
        name="reference_corpus",
        keys={
            "dataset_id",
            "dataset_version",
            "manifest_file_sha256",
            "manifest_sha256",
            "files",
        },
    )
    if (
        reference["dataset_id"] != BASE_DATASET_ID
        or reference["dataset_version"] != BASE_DATASET_VERSION
        or reference["manifest_file_sha256"] != REFERENCE_MANIFEST_FILE_SHA256
        or reference["manifest_sha256"] != REFERENCE_MANIFEST_SHA256
        or reference["files"] != REFERENCE_FILES
    ):
        raise SyntheticSotaReleaseEvalError("reference corpus identity changed")

    isolation = _require_mapping(
        raw["isolation"],
        name="isolation",
        keys={"reject_candidate_on_collision", "comparisons"},
    )
    if isolation["reject_candidate_on_collision"] != list(COLLISION_DIMENSIONS) or isolation[
        "comparisons"
    ] != [
        "calibration_vs_reference",
        "internal_evaluation_vs_reference",
        "calibration_vs_internal_evaluation",
    ]:
        raise SyntheticSotaReleaseEvalError("cross-corpus collision contract changed")

    usage = _require_mapping(
        raw["usage_policy"],
        name="usage_policy",
        keys=set(SPLITS),
    )
    expected_usage = {
        "calibration": CALIBRATION_POLICY,
        "internal_evaluation": INTERNAL_EVALUATION_POLICY,
    }
    if (
        any(not isinstance(usage[split], Mapping) for split in SPLITS)
        or {split: dict(usage[split]) for split in SPLITS} != expected_usage
    ):
        raise SyntheticSotaReleaseEvalError("split usage policy changed")

    inputs = _require_mapping(
        raw["input_policy"],
        name="input_policy",
        keys={"external_dataset_inputs", "forbidden_evaluation_sources"},
    )
    if inputs["external_dataset_inputs"] != []:
        raise SyntheticSotaReleaseEvalError("external dataset inputs are forbidden")
    forbidden = inputs["forbidden_evaluation_sources"]
    if not isinstance(forbidden, list) or set(forbidden) != FORBIDDEN_EVALUATION_SOURCES:
        raise SyntheticSotaReleaseEvalError("protected evaluation-source denylist changed")

    return SyntheticSotaReleaseEvalConfig(
        config_path=config_path,
        config_sha256=_sha256_bytes(raw_bytes),
        repository_root=root,
        base_config_path=base_path,
        split_configs=split_configs,
        pii_free_ratio=0.45,
        hard_negative_ratio=0.45,
        max_candidates_per_split=maximum,
        usage_policy=expected_usage,
    )


def load_reference_fingerprint_index(
    reference_dir: Path,
) -> tuple[FingerprintIndex, dict[str, Any]]:
    """Verify and index the exact frozen 80k/10k base corpus."""

    reference_dir = reference_dir.expanduser().resolve(strict=True)
    manifest_path = reference_dir / "dataset_manifest.json"
    if _sha256_file(manifest_path) != REFERENCE_MANIFEST_FILE_SHA256:
        raise SyntheticSotaReleaseEvalError("reference manifest file hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SyntheticSotaReleaseEvalError("reference manifest cannot be read") from exc
    if not isinstance(manifest, Mapping):
        raise SyntheticSotaReleaseEvalError("reference manifest must be an object")
    unsigned = dict(manifest)
    observed_self_hash = unsigned.pop("manifest_sha256", None)
    if (
        observed_self_hash != REFERENCE_MANIFEST_SHA256
        or _canonical_json_hash(unsigned) != REFERENCE_MANIFEST_SHA256
        or manifest.get("dataset_id") != BASE_DATASET_ID
        or manifest.get("dataset_version") != BASE_DATASET_VERSION
        or manifest.get("data_pool") != DataPool.PUBLIC_RELEASE.value
        or manifest.get("public_weight_training_allowed") is not True
    ):
        raise SyntheticSotaReleaseEvalError("reference manifest identity/self-hash failed")

    index = FingerprintIndex.empty()
    split_counts: dict[str, int] = {}
    for filename, expected in REFERENCE_FILES.items():
        path = reference_dir / filename
        if _sha256_file(path) != expected["sha256"]:
            raise SyntheticSotaReleaseEvalError(f"reference file hash mismatch: {filename}")
        expected_split = filename.removesuffix(".jsonl")
        count = 0
        for record in iter_jsonl(path):
            if (
                record.split != expected_split
                or record.data_pool is not DataPool.PUBLIC_RELEASE
                or record.public_weight_training_allowed is not True
                or record.provenance.evaluation_only
            ):
                raise SyntheticSotaReleaseEvalError(
                    f"reference record isolation failed: {filename}"
                )
            index.add(record)
            count += 1
        if count != expected["records"]:
            raise SyntheticSotaReleaseEvalError(f"reference record count failed: {filename}")
        split_counts[expected_split] = count
    if index.record_count != 90_000:
        raise SyntheticSotaReleaseEvalError("reference corpus must contain exactly 90000 rows")
    audit = {
        "dataset_id": BASE_DATASET_ID,
        "dataset_version": BASE_DATASET_VERSION,
        "manifest_file_sha256": REFERENCE_MANIFEST_FILE_SHA256,
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "file_sha256": {name: str(spec["sha256"]) for name, spec in REFERENCE_FILES.items()},
        "split_counts": split_counts,
        "fingerprint_counts": index.counts(),
        "fingerprint_set_sha256": index.set_digests(),
        "local_path_recorded": False,
    }
    return index, audit


def build_fingerprint_index(records: Sequence[DocumentRecord]) -> FingerprintIndex:
    index = FingerprintIndex.empty()
    for record in records:
        record.validate()
        index.add(record)
    return index


def _release_templates() -> tuple[tuple[SyntheticTemplate, ...], tuple[SyntheticTemplate, ...]]:
    allowed_ids = {
        template_id
        for template_id, split in TEMPLATE_SPLITS.items()
        if split in {"train", "validation"}
    }
    positive = tuple(
        template for template in POSITIVE_TEMPLATES if template.template_id in allowed_ids
    )
    # Static negatives are byte-identical to rows in the reference corpus and
    # therefore cannot enter an exact-text-isolated release holdout.
    negative = tuple(
        template
        for template in HARD_NEGATIVE_TEMPLATES
        if template.template_id in allowed_ids and template.placeholders
    )
    if not positive or not negative:
        raise SyntheticSotaReleaseEvalError("release template pools cannot be empty")
    labels = {
        spec.label
        for template in positive
        for spec in template.placeholders
        if spec.label is not None
    }
    if labels != set(CORE_LABELS):
        raise SyntheticSotaReleaseEvalError("release templates do not cover all 24 labels")
    return positive, negative


def _as_evaluation_record(record: DocumentRecord, *, split: str) -> DocumentRecord:
    provenance = replace(
        record.provenance,
        evaluation_only=True,
        metadata={
            **record.provenance.metadata,
            "release_evaluation_dataset_id": DATASET_ID,
            "release_evaluation_dataset_version": DATASET_VERSION,
            "release_evaluation_split": split,
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
            "pii_free": not record.entities,
            "usage_policy": (
                "post_selection_calibration_only"
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
    config: SyntheticSotaReleaseEvalConfig,
    reference_index: FingerprintIndex,
    accepted_index: FingerprintIndex,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    split_config = config.split_configs[split]
    negative_target = math.ceil(split_config.records * config.hard_negative_ratio)
    positive_target = split_config.records - negative_target
    positive_templates, negative_templates = _release_templates()
    if positive_target < len(positive_templates) or negative_target < len(negative_templates):
        raise SyntheticSotaReleaseEvalError("split is too small to exercise every template")
    generator = SyntheticSotaGenerator(split_config.seed)
    accepted: list[DocumentRecord] = []
    rejection_counts: Counter[str] = Counter()
    generated_by_kind: Counter[str] = Counter()

    for kind, target in (
        ("positive", positive_target),
        ("hard_negative", negative_target),
    ):
        accepted_for_kind = 0
        while accepted_for_kind < target:
            remaining = target - accepted_for_kind
            if sum(generated_by_kind.values()) + remaining > config.max_candidates_per_split:
                raise SyntheticSotaReleaseEvalError(
                    f"{split} exhausted its bounded collision-rejection budget"
                )
            candidates = generator.generate_partition(
                positive_count=remaining if kind == "positive" else 0,
                hard_negative_count=remaining if kind == "hard_negative" else 0,
                positive_templates=positive_templates,
                hard_negative_templates=negative_templates,
            )
            generated_by_kind[kind] += len(candidates)
            for raw_record in candidates:
                record = _as_evaluation_record(raw_record, split=split)
                collisions = set(reference_index.collision_dimensions(record))
                collisions.update(accepted_index.collision_dimensions(record))
                if collisions:
                    rejection_counts["candidate_records"] += 1
                    for dimension in sorted(collisions):
                        rejection_counts[dimension] += 1
                    continue
                accepted_index.add(record)
                accepted.append(record)
                accepted_for_kind += 1

    ordering_seed = split_config.seed ^ 0x5EED
    random.Random(ordering_seed).shuffle(accepted)
    return accepted, {
        "seed": split_config.seed,
        "ordering_algorithm": "deterministic-whole-split-shuffle-v1",
        "ordering_seed": ordering_seed,
        "requested_records": split_config.records,
        "accepted_records": len(accepted),
        "accepted_positive": positive_target,
        "accepted_pii_free": negative_target,
        "accepted_hard_negative": negative_target,
        "pii_free_ratio": negative_target / split_config.records,
        "hard_negative_ratio": negative_target / split_config.records,
        "generated_candidates_by_kind": dict(sorted(generated_by_kind.items())),
        "generated_candidates_total": sum(generated_by_kind.values()),
        "collision_rejections": {
            dimension: rejection_counts.get(dimension, 0)
            for dimension in ("candidate_records", *COLLISION_DIMENSIONS)
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


def audit_release_eval_isolation(
    records: Sequence[DocumentRecord],
    *,
    reference_index: FingerprintIndex,
) -> dict[str, Any]:
    """Recompute all required cross-corpus and cross-split collision counts."""

    split_records = {
        split: [record for record in records if record.split == split] for split in SPLITS
    }
    if sum(len(values) for values in split_records.values()) != len(records):
        raise SyntheticSotaReleaseEvalError("records contain an unsupported split")
    indexes = {split: build_fingerprint_index(values) for split, values in split_records.items()}
    comparisons = {
        "calibration_vs_reference": _intersection_counts(indexes["calibration"], reference_index),
        "internal_evaluation_vs_reference": _intersection_counts(
            indexes["internal_evaluation"], reference_index
        ),
        "calibration_vs_internal_evaluation": _intersection_counts(
            indexes["calibration"], indexes["internal_evaluation"]
        ),
    }
    nonzero = {
        comparison: {dimension: count for dimension, count in counts.items() if count}
        for comparison, counts in comparisons.items()
        if any(counts.values())
    }
    if nonzero:
        raise SyntheticSotaReleaseEvalError(f"release-evaluation collision audit failed: {nonzero}")
    return {
        "algorithm": "exact-and-normalized-cross-corpus-fingerprint-v1",
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


def build_release_eval_records(
    config: SyntheticSotaReleaseEvalConfig,
    *,
    reference_index: FingerprintIndex,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    """Generate both splits and return a complete in-memory quality audit."""

    accepted_index = FingerprintIndex.empty()
    records: list[DocumentRecord] = []
    generation: dict[str, Any] = {}
    for split in SPLITS:
        selected, split_audit = _generate_split(
            split=split,
            config=config,
            reference_index=reference_index,
            accepted_index=accepted_index,
        )
        records.extend(selected)
        generation[split] = split_audit

    label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
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
            raise SyntheticSotaReleaseEvalError("release row escaped evaluation-only isolation")
        split = str(record.split)
        split_counts[split] += 1
        label_counts[split].update(entity.label for entity in record.entities)
        if not record.entities:
            pii_free_counts[split] += 1
        if record.metadata.get("hard_negative") is True:
            hard_negative_counts[split] += 1
    for split in SPLITS:
        expected = config.split_configs[split].records
        expected_negative = math.ceil(expected * config.pii_free_ratio)
        if split_counts[split] != expected:
            raise SyntheticSotaReleaseEvalError(f"{split} record count failed")
        if pii_free_counts[split] != expected_negative or hard_negative_counts[split] != (
            expected_negative
        ):
            raise SyntheticSotaReleaseEvalError(f"{split} PII-free ratio failed")
        if set(label_counts[split]) != set(CORE_LABELS):
            raise SyntheticSotaReleaseEvalError(f"{split} lacks one or more core labels")

    isolation = audit_release_eval_isolation(records, reference_index=reference_index)
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
    records: Sequence[DocumentRecord],
    *,
    split: str,
    destination: Path,
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
            raise SyntheticSotaReleaseEvalError(f"{split} readback isolation failed")
        verified += 1
    if written != verified or verified != len(selected):
        raise SyntheticSotaReleaseEvalError(f"{split} readback count failed")
    return {
        "name": destination.name,
        "split": split,
        "records": verified,
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "mode": "0444",
    }


def materialize_release_eval_corpus(
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize one immutable release-evaluation corpus without overwrite."""

    config = load_release_eval_config(config_path, repository_root=repository_root)
    reference_index, reference_audit = load_reference_fingerprint_index(reference_dir)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(mode=0o755)
    try:
        records, audit = build_release_eval_records(
            config,
            reference_index=reference_index,
        )
        files = [
            _write_split(records, split=split, destination=staging / f"{split}.jsonl")
            for split in SPLITS
        ]
        manifest: dict[str, Any] = {
            "manifest_schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "license": "Apache-2.0",
            "language": "zh-Hans-CN",
            "publication_pool": DataPool.PUBLIC_RELEASE.value,
            "record_data_pool": DataPool.EVALUATION_ONLY.value,
            "public_weight_training_allowed": False,
            "public_artifact_release_allowed": True,
            "config": {
                "path": _repository_relative_or_redacted(
                    config.config_path,
                    config.repository_root,
                ),
                "sha256": config.config_sha256,
            },
            "generation": {
                "generator_name": SOTA_GENERATOR_NAME,
                "generator_version": SOTA_GENERATOR_VERSION,
                "materializer_version": MATERIALIZER_VERSION,
                "split_seeds": {split: config.split_configs[split].seed for split in SPLITS},
                "seeds_independent": True,
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
            "reference_corpus": reference_audit,
            "usage_policy": {split: dict(config.usage_policy[split]) for split in SPLITS},
            "input_policy": {
                "external_dataset_inputs": [],
                "synthetic_reference_inputs": [BASE_DATASET_ID],
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
        staging.chmod(0o555)
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists():
            staging.chmod(0o755)
            for child in staging.iterdir():
                child.chmod(0o644)
            shutil.rmtree(staging)
        raise


__all__ = [
    "COLLISION_DIMENSIONS",
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REFERENCE_DIR",
    "FingerprintIndex",
    "MATERIALIZER_VERSION",
    "ReleaseEvalSplitConfig",
    "SPLITS",
    "SyntheticSotaReleaseEvalConfig",
    "SyntheticSotaReleaseEvalError",
    "audit_release_eval_isolation",
    "build_fingerprint_index",
    "build_release_eval_records",
    "load_reference_fingerprint_index",
    "load_release_eval_config",
    "materialize_release_eval_corpus",
]
