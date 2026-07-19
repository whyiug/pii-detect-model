"""Materialize the benchmark-independent ``synthetic_sota_v1`` training corpus.

This recipe intentionally has no dataset input.  It renders the repository's
versioned, Apache-2.0 template asset with the deterministic synthetic value
factory.  The train and validation partitions use disjoint pinned template
families and are audited for template, generator sequence, surrogate, entity
value, near-duplicate-template and exact-text leakage before any file is
published.

The historical ``synthetic_v1`` materializer and its frozen holdouts are not
imported or modified by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..schema import DataPool, DocumentRecord, iter_jsonl, stable_text_hash, write_jsonl
from .allocation import (
    FAMILY_ALLOCATION_ALGORITHM_VERSION,
    FAMILY_ALLOCATION_ASSET_SHA256,
    TEMPLATE_SPLITS,
)
from .generator import GENERATOR_NAME, GENERATOR_VERSION, SyntheticGenerator
from .templates import (
    ALL_TEMPLATES,
    CORE_LABELS,
    CURATED_ASSET_ID,
    CURATED_ASSET_METADATA,
    CURATED_ASSET_SHA256,
    CURATED_ASSET_VERSION,
    CURATION_AUDIT_SHA256,
    HARD_NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    SyntheticTemplate,
)
from .values import (
    _COMPOUND_SURNAMES,
    _GIVEN_NAME_CHARACTERS,
    _SINGLE_SURNAMES,
    SyntheticValueFactory,
)

DATASET_ID = "pii_zh_synthetic_sota_v1"
DATASET_VERSION = "1.0.1"
MATERIALIZER_VERSION = "1.1.0"
SOTA_GENERATOR_NAME = "pii_zh_synthetic_sota"
SOTA_GENERATOR_VERSION = "1.1.0"
DEFAULT_CONFIG_PATH = Path("configs/data/synthetic_sota_v1.yaml")
DEFAULT_DATA_ROOT = Path(os.environ.get("PII_ZH_DATA_ROOT", "data"))
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed/public_release_pool/synthetic_sota_v1"
SPLITS = ("train", "validation")
GROUP_KEYS = (
    "template_id_group",
    "template_family_group",
    "generator_sequence_group",
    "surrogate_group",
    "entity_value_groups",
    "near_duplicate_group",
    "exact_text_group",
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
_PLACEHOLDER_RE = re.compile(r"<<([A-Z0-9_]+)>>")
_NUMBER_SUFFIX_RE = re.compile(r"_[1-9][0-9]*\Z")


class SyntheticSotaMaterializationError(ValueError):
    """Raised when the new corpus cannot satisfy its fail-closed contract."""


class SyntheticSotaValueFactory(SyntheticValueFactory):
    """Scale-safe value factory without changing the frozen base generator.

    The historical name generator has deliberately small stylistic streams and
    starts repeating at this recipe's 90k-record scale.  This subclass expands
    only the PERSON_NAME namespace.  All other versioned value constructors
    remain byte-for-byte inherited from ``SyntheticValueFactory``.
    """

    def person_name(self) -> str:
        style = self._unique_number("sota_person_name_style", 10)
        if style == 0:
            return self._unique_product(
                "sota_person_name_two_character",
                (_SINGLE_SURNAMES, _GIVEN_NAME_CHARACTERS),
            )
        if style < 8:
            surnames = _SINGLE_SURNAMES[(style - 1) :: 7]
            return self._unique_product(
                f"sota_person_name_three_character_{style}",
                (surnames, _GIVEN_NAME_CHARACTERS, _GIVEN_NAME_CHARACTERS),
            )
        if style == 8:
            return self._unique_product(
                "sota_person_name_four_character",
                (
                    _SINGLE_SURNAMES,
                    _GIVEN_NAME_CHARACTERS,
                    _GIVEN_NAME_CHARACTERS,
                    _GIVEN_NAME_CHARACTERS,
                ),
            )
        raw = self._unique_product(
            "sota_person_name_compound",
            (_COMPOUND_SURNAMES, _GIVEN_NAME_CHARACTERS, _GIVEN_NAME_CHARACTERS),
        )
        return f"{raw[:2]}·{raw[2:]}"

    def masked_phone(self) -> str:
        return "199****" + self._unique_digits("sota_masked_phone", 4)

    def quad_version(self) -> str:
        serial = self._unique_number("sota_quad_version", 10_000)
        return "v" + ".".join(f"{serial:04d}")


class SyntheticSotaGenerator(SyntheticGenerator):
    """SyntheticGenerator with explicitly versioned scale-safe value provenance."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._factory = SyntheticSotaValueFactory(self._rng)

    def _select_templates(
        self,
        templates: tuple[SyntheticTemplate, ...],
        count: int,
    ) -> list[SyntheticTemplate]:
        static = tuple(
            template
            for template in templates
            if template.hard_negative and not template.placeholders
        )
        if not static:
            return super()._select_templates(templates, count)
        if count <= len(static):
            selected = list(static)
            self._rng.shuffle(selected)
            return selected[:count]
        variable = tuple(template for template in templates if template not in static)
        if not variable:
            raise SyntheticSotaMaterializationError(
                "a repeated hard-negative partition cannot contain only static templates"
            )
        # Every static statement is useful once, but repeating an identical
        # hotline sentence thousands of times would reduce the effective
        # dataset size.  Remaining capacity is assigned only to templates with
        # a versioned value generator.
        return [*static, *super()._select_templates(variable, count - len(static))]

    def generate_document(self, template: SyntheticTemplate) -> DocumentRecord:
        record = super().generate_document(template)
        provenance = replace(
            record.provenance,
            generator_name=SOTA_GENERATOR_NAME,
            generator_version=SOTA_GENERATOR_VERSION,
            metadata={
                **record.provenance.metadata,
                "base_generator_name": GENERATOR_NAME,
                "base_generator_version": GENERATOR_VERSION,
                "value_factory_override": "scale_safe_unique_values_v2",
                "static_hard_negative_policy": "emit-once-per-split-v1",
            },
        )
        return replace(
            record,
            provenance=provenance,
            generator_version=SOTA_GENERATOR_VERSION,
            metadata={
                **record.metadata,
                "base_generator_version": GENERATOR_VERSION,
                "value_factory_override": "scale_safe_unique_values_v2",
                "static_hard_negative_policy": "emit-once-per-split-v1",
            },
        )


@dataclass(frozen=True, slots=True)
class SyntheticSotaConfig:
    config_path: Path
    config_sha256: str
    repository_root: Path
    seed: int
    split_counts: Mapping[str, int]
    pii_free_ratio: float
    hard_negative_ratio: float
    source_registry_path: Path
    source_registry_sha256: str
    source_registry_entry: Mapping[str, Any]
    near_duplicate_threshold: float


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _group_hash(namespace: str, value: str) -> str:
    return "sha256:" + _sha256_bytes(f"{namespace}\0{value}".encode())


def _require_mapping(value: Any, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticSotaMaterializationError(f"{name} must be an object")
    if set(value) != keys:
        raise SyntheticSotaMaterializationError(
            f"{name} fields differ from the frozen schema: "
            f"expected={sorted(keys)}, observed={sorted(value)}"
        )
    return value


def _load_yaml_mapping(path: Path, *, description: str) -> tuple[Mapping[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SyntheticSotaMaterializationError(f"invalid {description} YAML") from exc
    if not isinstance(value, Mapping):
        raise SyntheticSotaMaterializationError(f"{description} must contain an object")
    return value, _sha256_bytes(raw)


def _resolve_repository_path(repository_root: Path, raw_path: Any, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SyntheticSotaMaterializationError(f"{name} must be a non-empty relative path")
    path = Path(raw_path)
    if path.is_absolute():
        raise SyntheticSotaMaterializationError(f"{name} must be repository-relative")
    resolved = (repository_root / path).resolve(strict=True)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise SyntheticSotaMaterializationError(f"{name} escapes the repository") from exc
    return resolved


def load_synthetic_sota_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> SyntheticSotaConfig:
    """Load the strict recipe and verify its source-registry admission entry."""

    config_path = config_path.expanduser().resolve(strict=True)
    root = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    raw, config_sha256 = _load_yaml_mapping(config_path, description="synthetic SOTA config")
    _require_mapping(
        raw,
        name="config",
        keys={
            "schema_version",
            "dataset",
            "generation",
            "source",
            "split_isolation",
            "input_policy",
        },
    )
    if raw["schema_version"] != 1:
        raise SyntheticSotaMaterializationError("config.schema_version must be 1")

    dataset = _require_mapping(
        raw["dataset"],
        name="dataset",
        keys={"id", "version", "license", "language", "data_pool"},
    )
    expected_dataset = {
        "id": DATASET_ID,
        "version": DATASET_VERSION,
        "license": "Apache-2.0",
        "language": "zh-Hans-CN",
        "data_pool": DataPool.PUBLIC_RELEASE.value,
    }
    if dict(dataset) != expected_dataset:
        raise SyntheticSotaMaterializationError(
            "dataset identity or public license contract changed"
        )

    generation = _require_mapping(
        raw["generation"],
        name="generation",
        keys={
            "seed",
            "splits",
            "pii_free_ratio",
            "hard_negative_ratio",
        },
    )
    seed = generation["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SyntheticSotaMaterializationError("generation.seed must be an integer")
    split_counts = _require_mapping(
        generation["splits"],
        name="generation.splits",
        keys=set(SPLITS),
    )
    if tuple(split_counts) != SPLITS or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in split_counts.values()
    ):
        raise SyntheticSotaMaterializationError(
            "generation.splits must contain positive train/validation integers in order"
        )
    pii_free_ratio = generation["pii_free_ratio"]
    hard_negative_ratio = generation["hard_negative_ratio"]
    if any(
        isinstance(ratio, bool) or not isinstance(ratio, (int, float))
        for ratio in (pii_free_ratio, hard_negative_ratio)
    ):
        raise SyntheticSotaMaterializationError("generation ratios must be numeric")
    pii_free_ratio = float(pii_free_ratio)
    hard_negative_ratio = float(hard_negative_ratio)
    if not 0.40 <= pii_free_ratio <= 0.50:
        raise SyntheticSotaMaterializationError("pii_free_ratio must be between 0.40 and 0.50")
    if not 0.30 <= hard_negative_ratio <= pii_free_ratio:
        raise SyntheticSotaMaterializationError(
            "hard_negative_ratio must be at least 0.30 and no larger than pii_free_ratio"
        )
    # This v1 recipe deliberately gives every PII-free row a reviewed
    # hard-negative template.  A future neutral-negative asset must receive a
    # new recipe version instead of silently weakening that provenance.
    if hard_negative_ratio != pii_free_ratio:
        raise SyntheticSotaMaterializationError(
            "synthetic_sota_v1 requires every PII-free row to be a hard negative"
        )

    source = _require_mapping(
        raw["source"],
        name="source",
        keys={
            "registry_path",
            "source_id",
            "expected_template_asset_sha256",
            "expected_curation_audit_sha256",
            "expected_family_allocation_sha256",
            "expected_generator_version",
        },
    )
    if source["source_id"] != "repo_curated_synthetic_templates":
        raise SyntheticSotaMaterializationError("only the curated synthetic source is allowed")
    expected_source_bindings = {
        "expected_template_asset_sha256": CURATED_ASSET_SHA256,
        "expected_curation_audit_sha256": CURATION_AUDIT_SHA256,
        "expected_family_allocation_sha256": FAMILY_ALLOCATION_ASSET_SHA256,
        "expected_generator_version": GENERATOR_VERSION,
    }
    if any(source[key] != value for key, value in expected_source_bindings.items()):
        raise SyntheticSotaMaterializationError("source asset or generator binding changed")
    registry_path = _resolve_repository_path(
        root,
        source["registry_path"],
        name="source.registry_path",
    )
    registry, registry_sha256 = _load_yaml_mapping(
        registry_path,
        description="source registry",
    )
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise SyntheticSotaMaterializationError("source registry has no sources array")
    entries = [
        item
        for item in sources
        if isinstance(item, Mapping) and item.get("id") == source["source_id"]
    ]
    if len(entries) != 1:
        raise SyntheticSotaMaterializationError("curated synthetic source must occur exactly once")
    entry = entries[0]
    if (
        entry.get("revision") != f"sha256:{CURATED_ASSET_SHA256}"
        or entry.get("curation_audit_sha256") != CURATION_AUDIT_SHA256
        or entry.get("generator_version") != GENERATOR_VERSION
        or entry.get("declared_license") != "Apache-2.0"
        or entry.get("public_weight_training_allowed") is not True
        or entry.get("forced_pool") != DataPool.PUBLIC_RELEASE.value
    ):
        raise SyntheticSotaMaterializationError(
            "curated source registry entry is not admitted for public weight training"
        )
    allocation = entry.get("family_allocation_asset")
    if not isinstance(allocation, Mapping) or allocation.get("revision") != (
        f"sha256:{FAMILY_ALLOCATION_ASSET_SHA256}"
    ):
        raise SyntheticSotaMaterializationError("source registry allocation binding changed")

    isolation = _require_mapping(
        raw["split_isolation"],
        name="split_isolation",
        keys={
            "algorithm",
            "near_duplicate_method",
            "near_duplicate_threshold",
            "group_keys",
        },
    )
    if (
        isolation["algorithm"] != "pinned-family-near-duplicate-v1"
        or isolation["near_duplicate_method"]
        != "normalized-placeholder-char-3gram-jaccard-components-v1"
        or isolation["group_keys"] != list(GROUP_KEYS)
    ):
        raise SyntheticSotaMaterializationError("split-isolation algorithm contract changed")
    threshold = isolation["near_duplicate_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise SyntheticSotaMaterializationError("near_duplicate_threshold must be numeric")
    threshold = float(threshold)
    if threshold != 0.75:
        raise SyntheticSotaMaterializationError(
            "synthetic_sota_v1 pins the audited near-duplicate threshold at 0.75"
        )

    input_policy = _require_mapping(
        raw["input_policy"],
        name="input_policy",
        keys={"external_dataset_inputs", "forbidden_evaluation_sources"},
    )
    if input_policy["external_dataset_inputs"] != []:
        raise SyntheticSotaMaterializationError("external dataset inputs are forbidden")
    forbidden = input_policy["forbidden_evaluation_sources"]
    if not isinstance(forbidden, list) or not FORBIDDEN_EVALUATION_SOURCES.issubset(forbidden):
        raise SyntheticSotaMaterializationError(
            "input policy must explicitly deny all protected evaluation sources"
        )

    return SyntheticSotaConfig(
        config_path=config_path,
        config_sha256=config_sha256,
        repository_root=root,
        seed=seed,
        split_counts={name: int(split_counts[name]) for name in SPLITS},
        pii_free_ratio=pii_free_ratio,
        hard_negative_ratio=hard_negative_ratio,
        source_registry_path=registry_path,
        source_registry_sha256=registry_sha256,
        source_registry_entry=dict(entry),
        near_duplicate_threshold=threshold,
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _normalized_template_skeleton(template: SyntheticTemplate) -> str:
    specs = {spec.key: spec for spec in template.placeholders}

    def replace_placeholder(match: re.Match[str]) -> str:
        key = match.group(1)
        spec = specs[key]
        role = spec.label or _NUMBER_SUFFIX_RE.sub("", spec.generator_key)
        return f"占位{role}占位"

    rendered = _PLACEHOLDER_RE.sub(replace_placeholder, template.body)
    normalized = unicodedata.normalize("NFKC", rendered).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _char_shingles(value: str, *, size: int = 3) -> frozenset[str]:
    if len(value) < size:
        return frozenset({value})
    return frozenset(value[index : index + size] for index in range(len(value) - size + 1))


def build_near_duplicate_groups(
    templates: Sequence[SyntheticTemplate] = ALL_TEMPLATES,
    *,
    threshold: float = 0.75,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Cluster normalized template skeletons by connected Jaccard components."""

    if not templates:
        raise SyntheticSotaMaterializationError("template catalog cannot be empty")
    if not 0.0 < threshold <= 1.0:
        raise SyntheticSotaMaterializationError("near-duplicate threshold must be in (0, 1]")
    if len({template.template_id for template in templates}) != len(templates):
        raise SyntheticSotaMaterializationError("template ids must be unique")
    shingles = [_char_shingles(_normalized_template_skeleton(template)) for template in templates]
    union_find = _UnionFind(len(templates))
    similar_pairs = 0
    for left in range(len(templates)):
        for right in range(left + 1, len(templates)):
            union = shingles[left] | shingles[right]
            similarity = len(shingles[left] & shingles[right]) / len(union)
            if similarity >= threshold:
                union_find.union(left, right)
                similar_pairs += 1
    components: dict[int, list[str]] = {}
    for index, template in enumerate(templates):
        components.setdefault(union_find.find(index), []).append(template.template_id)
    groups: dict[str, str] = {}
    component_sizes: Counter[int] = Counter()
    for members in components.values():
        stable_members = sorted(members)
        group = _group_hash("synthetic-sota-v1-near-duplicate", "|".join(stable_members))
        component_sizes[len(stable_members)] += 1
        for template_id in stable_members:
            groups[template_id] = group
    audit = {
        "method": "normalized-placeholder-char-3gram-jaccard-components-v1",
        "threshold": threshold,
        "template_count": len(templates),
        "similar_pair_count": similar_pairs,
        "component_count": len(components),
        "component_size_histogram": {
            str(size): count for size, count in sorted(component_sizes.items())
        },
        "template_to_group_sha256": _canonical_json_hash(sorted(groups.items())),
        "raw_template_text_in_manifest": False,
    }
    return groups, audit


def _template_pools(
    near_groups: Mapping[str, str],
) -> tuple[
    dict[str, tuple[tuple[SyntheticTemplate, ...], tuple[SyntheticTemplate, ...]]],
    dict[str, Any],
]:
    """Close pinned split anchors over near-duplicate components.

    Validation is the conservative owner of any component containing both a
    pinned train and pinned validation template.  This keeps the entire
    near-duplicate component out of training.  Components touching the
    historical test allocation fail closed instead of importing a test-family
    neighbor into this training recipe.
    """

    template_by_id = {template.template_id: template for template in ALL_TEMPLATES}
    output_ids = {template_id for template_id, split in TEMPLATE_SPLITS.items() if split in SPLITS}
    expected_ids = set(template_by_id)
    if set(TEMPLATE_SPLITS) != expected_ids:
        raise SyntheticSotaMaterializationError("pinned template catalog drifted")
    group_members: dict[str, set[str]] = {}
    for template_id, group in near_groups.items():
        group_members.setdefault(group, set()).add(template_id)

    owner: dict[str, str] = {}
    moved_to_validation: set[str] = set()
    for members in group_members.values():
        pinned_splits = {TEMPLATE_SPLITS[template_id] for template_id in members}
        output_members = members & output_ids
        if not output_members:
            continue
        if "test" in pinned_splits and pinned_splits & set(SPLITS):
            raise SyntheticSotaMaterializationError(
                "an output template is near-duplicate with a historical test template"
            )
        target = "validation" if "validation" in pinned_splits else "train"
        for template_id in output_members:
            owner[template_id] = target
            if target == "validation" and TEMPLATE_SPLITS[template_id] == "train":
                moved_to_validation.add(template_id)
    if set(owner) != output_ids:
        raise SyntheticSotaMaterializationError("near-duplicate closure lost an output template")

    positive = {
        split: tuple(
            template for template in POSITIVE_TEMPLATES if owner.get(template.template_id) == split
        )
        for split in SPLITS
    }
    negative = {
        split: tuple(
            template
            for template in HARD_NEGATIVE_TEMPLATES
            if owner.get(template.template_id) == split
        )
        for split in SPLITS
    }
    if any(not positive[split] or not negative[split] for split in SPLITS):
        raise SyntheticSotaMaterializationError(
            "every output split needs positive and hard-negative template components"
        )
    pools = {split: (positive[split], negative[split]) for split in SPLITS}
    audit = {
        "base_allocation": FAMILY_ALLOCATION_ALGORITHM_VERSION,
        "closure_policy": "validation-owns-cross-train-validation-near-component-v1",
        "output_template_count": len(output_ids),
        "template_counts_by_split": {
            split: len(positive[split]) + len(negative[split]) for split in SPLITS
        },
        "positive_template_counts_by_split": {split: len(positive[split]) for split in SPLITS},
        "hard_negative_template_counts_by_split": {split: len(negative[split]) for split in SPLITS},
        "moved_from_train_to_validation_count": len(moved_to_validation),
        "moved_from_train_to_validation_ids_sha256": _canonical_json_hash(
            sorted(moved_to_validation)
        ),
        "assignment_sha256": _canonical_json_hash(sorted(owner.items())),
        "historical_test_templates_used": 0,
    }
    return pools, audit


def _strict_group_values(record: DocumentRecord) -> tuple[tuple[str, str], ...]:
    contract = record.metadata.get("synthetic_sota_group_contract")
    if not isinstance(contract, Mapping):
        raise SyntheticSotaMaterializationError("record is missing its group contract")
    required = {
        "template_id_group",
        "template_family_group",
        "generator_sequence_group",
        "surrogate_group",
        "near_duplicate_group",
        "exact_text_group",
    }
    if set(contract) != required or any(
        not isinstance(contract[key], str) or not contract[key].startswith("sha256:")
        for key in required
    ):
        raise SyntheticSotaMaterializationError("record group contract is malformed")
    values = [(key, contract[key]) for key in sorted(required)]
    values.extend(("entity_value_groups", value) for value in record.entity_value_groups)
    return tuple(values)


def assert_synthetic_sota_isolation(records: Sequence[DocumentRecord]) -> dict[str, Any]:
    """Fail on any cross-split collision across every v1 group dimension."""

    if not records:
        raise SyntheticSotaMaterializationError("cannot audit an empty corpus")
    owner: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    for record in records:
        record.validate()
        if record.split not in SPLITS:
            raise SyntheticSotaMaterializationError("record has an unsupported or missing split")
        for group_type, group_value in _strict_group_values(record):
            counts[group_type] += 1
            previous = owner.setdefault((group_type, group_value), record.split)
            if previous != record.split:
                raise SyntheticSotaMaterializationError(
                    f"cross-split {group_type} collision detected"
                )
    unique_counts = Counter(group_type for group_type, _ in owner)
    return {
        "algorithm": "strict-cross-split-group-owner-v1",
        "group_keys": list(GROUP_KEYS),
        "records_checked": len(records),
        "observations_by_group_type": dict(sorted(counts.items())),
        "unique_groups_by_type": dict(sorted(unique_counts.items())),
        "cross_split_collisions": 0,
    }


def _decorate_record(
    record: DocumentRecord,
    *,
    split: str,
    sequence_group: str,
    near_duplicate_group: str,
) -> DocumentRecord:
    template_id = record.provenance.template_id
    if not template_id or not record.template_group:
        raise SyntheticSotaMaterializationError("synthetic record lacks template provenance")
    value_fingerprint = "|".join(sorted(record.entity_value_groups))
    surrogate_material = value_fingerprint or record.doc_id
    contract = {
        "template_id_group": _group_hash("synthetic-sota-v1-template", template_id),
        "template_family_group": _group_hash(
            "synthetic-sota-v1-template-family", record.template_group
        ),
        "generator_sequence_group": sequence_group,
        "surrogate_group": _group_hash("synthetic-sota-v1-surrogate", surrogate_material),
        "near_duplicate_group": near_duplicate_group,
        "exact_text_group": stable_text_hash(record.text),
    }
    return replace(
        record,
        split=split,
        metadata={
            **record.metadata,
            "synthetic_sota_recipe": DATASET_VERSION,
            "pii_free": not record.entities,
            "synthetic_sota_group_contract": contract,
        },
    )


def build_synthetic_sota_records(
    config: SyntheticSotaConfig,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    """Generate and fully audit the in-memory corpus described by ``config``."""

    near_groups, near_audit = build_near_duplicate_groups(threshold=config.near_duplicate_threshold)
    pools, template_allocation_audit = _template_pools(near_groups)
    near_group_splits: dict[str, set[str]] = {}
    for split in SPLITS:
        for pool in pools[split]:
            for template in pool:
                near_group_splits.setdefault(near_groups[template.template_id], set()).add(split)
    leaking_near_groups = {
        group: splits for group, splits in near_group_splits.items() if len(splits) > 1
    }
    if leaking_near_groups:
        raise SyntheticSotaMaterializationError(
            "closed template allocation crosses a near-duplicate component"
        )

    generator = SyntheticSotaGenerator(config.seed)
    records: list[DocumentRecord] = []
    sequence_start = 0
    split_generation: dict[str, Any] = {}
    for split in SPLITS:
        count = config.split_counts[split]
        negative_count = math.ceil(count * config.hard_negative_ratio)
        positive_count = count - negative_count
        positive_templates, negative_templates = pools[split]
        if positive_count < len(positive_templates):
            raise SyntheticSotaMaterializationError(
                f"split {split!r} cannot exercise every positive template"
            )
        if negative_count < len(negative_templates):
            raise SyntheticSotaMaterializationError(
                f"split {split!r} cannot exercise every hard-negative template"
            )
        sequence_end = sequence_start + count
        sequence_group = _group_hash(
            "synthetic-sota-v1-generator-sequence",
            f"{config.seed}:{sequence_start}:{sequence_end}",
        )
        generated = generator.generate_partition(
            positive_count=positive_count,
            hard_negative_count=negative_count,
            positive_templates=positive_templates,
            hard_negative_templates=negative_templates,
        )
        decorated = [
            _decorate_record(
                record,
                split=split,
                sequence_group=sequence_group,
                near_duplicate_group=near_groups[str(record.provenance.template_id)],
            )
            for record in generated
        ]
        records.extend(decorated)
        split_generation[split] = {
            "records": count,
            "positive": positive_count,
            "pii_free": negative_count,
            "hard_negative": negative_count,
            "pii_free_ratio": negative_count / count,
            "hard_negative_ratio": negative_count / count,
            "generator_sequence_start": sequence_start,
            "generator_sequence_end": sequence_end,
            "generator_sequence_group": sequence_group,
            "positive_template_count": len(positive_templates),
            "hard_negative_template_count": len(negative_templates),
        }
        sequence_start = sequence_end

    isolation = assert_synthetic_sota_isolation(records)
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    split_counts: Counter[str] = Counter()
    pii_free_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    observed_values: set[str] = set()
    observed_texts: set[str] = set()
    for record in records:
        split = str(record.split)
        split_counts[split] += 1
        label_counts[split].update(entity.label for entity in record.entities)
        if not record.entities:
            pii_free_counts[split] += 1
        if record.metadata.get("hard_negative") is True:
            hard_negative_counts[split] += 1
        text_group = stable_text_hash(record.text)
        if text_group in observed_texts:
            raise SyntheticSotaMaterializationError(
                "rendered records must be globally exact-text unique"
            )
        observed_texts.add(text_group)
        for value_group in record.entity_value_groups:
            if value_group in observed_values:
                raise SyntheticSotaMaterializationError(
                    "generated entity values are not globally unique"
                )
            observed_values.add(value_group)
    if dict(split_counts) != dict(config.split_counts):
        raise SyntheticSotaMaterializationError("generated split counts differ from config")
    for split in SPLITS:
        if set(label_counts[split]) != set(CORE_LABELS):
            raise SyntheticSotaMaterializationError(
                f"split {split!r} does not cover all 24 core labels"
            )
        observed_pii_free_ratio = pii_free_counts[split] / split_counts[split]
        observed_hard_negative_ratio = hard_negative_counts[split] / split_counts[split]
        if not 0.40 <= observed_pii_free_ratio <= 0.50:
            raise SyntheticSotaMaterializationError("observed PII-free ratio left its contract")
        if observed_hard_negative_ratio < 0.30:
            raise SyntheticSotaMaterializationError(
                "observed hard-negative ratio left its contract"
            )
    audit = {
        "split_generation": split_generation,
        "label_counts_by_split": {
            split: dict(sorted(label_counts[split].items())) for split in SPLITS
        },
        "entity_value_group_count": len(observed_values),
        "entity_value_groups_globally_unique": True,
        "exact_text_group_count": len(observed_texts),
        "exact_texts_globally_unique": True,
        "template_allocation": template_allocation_audit,
        "near_duplicate": near_audit,
        "isolation": isolation,
    }
    return records, audit


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
        if record.split != split or record.data_pool is not DataPool.PUBLIC_RELEASE:
            raise SyntheticSotaMaterializationError(f"{split} readback isolation failed")
        _strict_group_values(record)
        verified += 1
    if written != verified or verified != len(selected):
        raise SyntheticSotaMaterializationError(f"{split} readback count failed")
    return {
        "name": destination.name,
        "split": split,
        "records": verified,
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "mode": "0444",
    }


def materialize_synthetic_sota_corpus(
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize one immutable corpus; existing destinations are never overwritten."""

    config = load_synthetic_sota_config(config_path, repository_root=repository_root)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(mode=0o755)
    try:
        records, audit = build_synthetic_sota_records(config)
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
            "data_pool": DataPool.PUBLIC_RELEASE.value,
            "public_weight_training_allowed": True,
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
                "base_generator_name": GENERATOR_NAME,
                "base_generator_version": GENERATOR_VERSION,
                "value_factory_override": "scale_safe_unique_values_v2",
                "static_hard_negative_policy": "emit-once-per-split-v1",
                "materializer_version": MATERIALIZER_VERSION,
                "seed": config.seed,
                "requested_split_counts": dict(config.split_counts),
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
                "source_registry_path": str(
                    config.source_registry_path.relative_to(config.repository_root)
                ),
                "source_registry_sha256": config.source_registry_sha256,
                "source_registry_entry_sha256": _canonical_json_hash(config.source_registry_entry),
                "template_asset_id": CURATED_ASSET_ID,
                "template_asset_version": CURATED_ASSET_VERSION,
                "template_asset_sha256": CURATED_ASSET_SHA256,
                "curation_audit_sha256": CURATION_AUDIT_SHA256,
                "family_allocation_algorithm": FAMILY_ALLOCATION_ALGORITHM_VERSION,
                "family_allocation_asset_sha256": FAMILY_ALLOCATION_ASSET_SHA256,
                "declared_license": "Apache-2.0",
                "candidate_template_teacher": {
                    "model_id": CURATED_ASSET_METADATA["candidate_source"]["model_id"],
                    "declared_license": CURATED_ASSET_METADATA["candidate_source"][
                        "declared_license"
                    ],
                    "deployment": CURATED_ASSET_METADATA["candidate_source"]["deployment"],
                    "raw_outputs_included": False,
                    "accepted_templates_human_curated": True,
                },
            },
            "input_policy": {
                "external_dataset_inputs": [],
                "forbidden_evaluation_sources": sorted(FORBIDDEN_EVALUATION_SOURCES),
                "forbidden_sources_read": [],
            },
            "counts": {
                "total": len(records),
                "by_split": dict(config.split_counts),
                "pii_free": sum(1 for record in records if record.metadata.get("pii_free") is True),
                "hard_negative": sum(
                    1 for record in records if record.metadata.get("hard_negative") is True
                ),
                "core_labels": len(CORE_LABELS),
            },
            "audits": audit,
            "files": files,
        }
        unsigned_hash = _canonical_json_hash(manifest)
        manifest["manifest_sha256"] = unsigned_hash
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
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "GROUP_KEYS",
    "MATERIALIZER_VERSION",
    "SOTA_GENERATOR_NAME",
    "SOTA_GENERATOR_VERSION",
    "SyntheticSotaGenerator",
    "SyntheticSotaConfig",
    "SyntheticSotaMaterializationError",
    "SyntheticSotaValueFactory",
    "assert_synthetic_sota_isolation",
    "build_near_duplicate_groups",
    "build_synthetic_sota_records",
    "load_synthetic_sota_config",
    "materialize_synthetic_sota_corpus",
]
