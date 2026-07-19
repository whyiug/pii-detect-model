"""Fail-closed assembly of human context with split-isolated surrogate PII.

The input format is deliberately private: it can contain source text, internal
grouping material, and private relation material.  None of those values are
copied into public identifiers, manifests, exception messages, or object
representations.  A build is accepted only when its source registry, privacy
proof, annotation proof, negative-sample composition, grouped split, and
surrogate reconstruction all pass.

The module does not invent PII-shaped replacements.  Each private edit carries
one independently reviewed replacement per split.  After group assignment the
builder selects exactly one replacement, derives a split/label/domain-separated
opaque relation identifier, and rebuilds text plus character spans in one
forward pass.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TextIO

from pii_zh.taxonomy import load_taxonomy

from .context_contract import (
    DEFAULT_CONTEXT_CONTRACT_PATH,
    ContextContractError,
    ContextDatasetContract,
    load_context_dataset_contract,
)
from .schema import (
    DataPool,
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    SchemaError,
    dumps_document,
)
from .source_admission import (
    CandidateSourceRegistry,
    evaluate_registered_source,
)
from .surrogates import (
    OriginalSpan,
    SurrogateEdit,
    SurrogateKeyDeriver,
    rebuild_surrogate_text,
)
from .validators import validate_luhn

PRIVATE_INPUT_SCHEMA_VERSION = "pii-zh-context-v2-private-assembly-v1"
MANIFEST_SCHEMA_VERSION = "pii-zh-context-v2-assembly-manifest-v1"
RELEASE_MANIFEST_SCHEMA_VERSION = "pii-zh-context-v2-release-assembly-manifest-v1"
DEVELOPMENT_ASSEMBLY_MODE = "development"
RELEASE_ASSEMBLY_MODE = "release"
SPLITS = ("train", "validation", "public_test", "hidden_leaderboard")
PUBLIC_SPLITS = SPLITS[:-1]

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LABEL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_PUBLIC_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DOC_ID_PATTERN = re.compile(r"^doc_v2_[A-Za-z0-9_-]{24}$")
_GROUP_ID_PATTERN = re.compile(r"^grp_v2_[A-Za-z0-9_-]{24}$")
_ANNOTATION_ID_PATTERN = re.compile(r"^ann_v2_[A-Za-z0-9_-]{24}$")
_RELATION_ID_PATTERN = re.compile(r"^rel_v2_[A-Za-z0-9_-]{24}$")
_CONTEXT_ORIGINS = frozenset(
    {
        "commissioned_human",
        "open_human_dialogue",
        "wizard_of_oz",
        "licensed_natural_text",
        "production_derived",
    }
)
_SURFACE_STATES = frozenset({"surrogate", "surrogate_sentinel", "masked", "invalid", "partial"})
_SUBJECT_KINDS = frozenset({"person", "public_person", "organization", "service", "unknown"})
_VALIDATOR_STATES = frozenset({"passed", "failed", "not_applicable", "intentionally_invalid"})
_ADJUDICATION_STATES = frozenset({"adjudicated", "single_reviewed"})
_RISK_TIERS = frozenset({"T0", "T1", "T2"})
_MAX_RELATION_BYTES = 1024 * 1024
_MAX_PRIVATE_FIELD_BYTES = 1024 * 1024
_OPAQUE_ID_BYTES = 18
_ID_PROTOCOL = b"pii-zh/context-v2/public-id/v1"
_REVIEW_PROTOCOL = b"pii-zh/context-v2/review-evidence/v1"
_OUTPUT_SEAL_PROTOCOL = b"pii-zh/context-v2/output-seal/v1"
_RESIDUAL_SCAN_VERSION = "pii-zh-residual-scan-v1"
_BASELINE_MINIMUM_PII_FREE = 0.30
_BASELINE_MAXIMUM_PII_FREE = 0.40
_BASELINE_MINIMUM_HARD_NEGATIVE = 0.15
_FROZEN_SPLIT_SEED = 20260712
_FROZEN_RELEASE_CONTRACT_PATH = "configs/data/real_context_dataset_v2.yaml"
_FROZEN_RELEASE_CONTRACT_FILE_SHA256 = (
    "863e094d093db5367a0c67a9cbe53f5d416800174814372a508ad1e0d072db81"
)
_RELEASE_LINKAGE_KINDS = (
    "document",
    "tenant",
    "conversation",
    "template",
    "document_group",
    "source_item",
    "author",
    "near_duplicate",
    "exact_placeholder",
    "parent",
    "relation",
)

_TAXONOMY = load_taxonomy()
_CORE_LABELS = _TAXONOMY.core_label_names
_HARD_NEGATIVE_CATEGORIES = frozenset(label.casefold() for label in _TAXONOMY.auxiliary_label_names)
_PUBLIC_WEIGHT_LICENSES = frozenset(
    {"APACHE-2.0", "BSD-2-CLAUSE", "BSD-3-CLAUSE", "CC-BY-4.0", "CC0-1.0", "MIT"}
)

_RAW_SENTINEL_PATTERN = re.compile(r"(?i)(?:RAW[_ -]?PII|UNREDACTED[_ -]?PII)")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:(?:\+?86[- ]?)?1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})(?!\d)")
_CN_ID_PATTERN = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}"
)
_IP_TOKEN_PATTERN = re.compile(r"(?<![0-9A-Fa-f:.])(?:[0-9A-Fa-f:.]{3,45})(?![0-9A-Fa-f:.])")
_MAC_PATTERN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])")
_BANK_DIGITS_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){11,18}\d(?!\d)")
_SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


class AssemblyError(ValueError):
    """Raised without reflecting private record values or filesystem paths."""


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Deterministic document-level target ratios."""

    train: float = 0.70
    validation: float = 0.10
    public_test: float = 0.10
    hidden_leaderboard: float = 0.10

    def ratios(self) -> dict[str, float]:
        values = {
            "train": self.train,
            "validation": self.validation,
            "public_test": self.public_test,
            "hidden_leaderboard": self.hidden_leaderboard,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in values.values()
        ):
            raise AssemblyError("split policy contains an invalid ratio")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise AssemblyError("split policy ratios must sum to one")
        baseline = {
            "train": 0.70,
            "validation": 0.10,
            "public_test": 0.10,
            "hidden_leaderboard": 0.10,
        }
        if any(
            not math.isclose(float(values[name]), baseline[name], rel_tol=0.0, abs_tol=1e-9)
            for name in values
        ):
            raise AssemblyError("split policy cannot weaken the frozen v2 split contract")
        return {name: float(value) for name, value in values.items()}


@dataclass(frozen=True, slots=True)
class NegativeRatioPolicy:
    """Dataset-level PII-free and hard-negative composition gates."""

    minimum_pii_free: float = 0.30
    maximum_pii_free: float = 0.40
    minimum_hard_negative: float = 0.15

    def validate(self) -> None:
        values = (
            self.minimum_pii_free,
            self.maximum_pii_free,
            self.minimum_hard_negative,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise AssemblyError("negative-ratio policy is invalid")
        if not 0.0 <= self.minimum_pii_free <= self.maximum_pii_free <= 1.0:
            raise AssemblyError("PII-free ratio bounds are invalid")
        if not 0.0 <= self.minimum_hard_negative <= self.maximum_pii_free:
            raise AssemblyError("hard-negative ratio bound is invalid")
        if (
            self.minimum_pii_free < _BASELINE_MINIMUM_PII_FREE
            or self.maximum_pii_free > _BASELINE_MAXIMUM_PII_FREE
            or self.minimum_hard_negative < _BASELINE_MINIMUM_HARD_NEGATIVE
        ):
            raise AssemblyError("negative-ratio policy cannot weaken the v2 baseline")


def _normalize_assembly_mode(value: object) -> str:
    if value not in {DEVELOPMENT_ASSEMBLY_MODE, RELEASE_ASSEMBLY_MODE}:
        raise AssemblyError("assembly mode must be development or release")
    assert isinstance(value, str)
    return value


def _load_frozen_release_contract() -> ContextDatasetContract:
    try:
        payload = DEFAULT_CONTEXT_CONTRACT_PATH.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _FROZEN_RELEASE_CONTRACT_FILE_SHA256:
            raise AssemblyError("frozen release-scale contract identity changed")
        return load_context_dataset_contract(DEFAULT_CONTEXT_CONTRACT_PATH)
    except AssemblyError:
        raise
    except (ContextContractError, OSError):
        raise AssemblyError("frozen release-scale contract is unavailable or invalid") from None


def release_scale_zero_collision_counts() -> dict[str, int]:
    """Return the closed cross-split linkage universe for release-mode evidence."""

    return {kind: 0 for kind in _RELEASE_LINKAGE_KINDS}


def validate_release_scale_aggregates(
    *,
    assembly_mode: str,
    document_count: int,
    split_document_counts: Mapping[str, int],
    pii_free_document_count: int,
    hard_negative_document_count: int,
    cross_split_group_collision_counts: Mapping[str, int],
) -> None:
    """Validate aggregate-only release scale without claiming real materialization.

    This function is deliberately unable to activate training, release, or claims.  It
    validates only the frozen count/composition/group-isolation contract; a separate,
    independently anchored real-materialization receipt remains mandatory.
    """

    if _normalize_assembly_mode(assembly_mode) != RELEASE_ASSEMBLY_MODE:
        raise AssemblyError("release-scale aggregates require explicit release mode")
    contract = _load_frozen_release_contract()
    if isinstance(document_count, bool) or not isinstance(document_count, int):
        raise AssemblyError("release document count must be an integer")
    if (
        not isinstance(split_document_counts, Mapping)
        or set(split_document_counts) != set(SPLITS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in split_document_counts.values()
        )
    ):
        raise AssemblyError("release split document counts are invalid")
    expected_splits = dict(contract.release_split_documents)
    if document_count != contract.release_total_documents or dict(split_document_counts) != (
        expected_splits
    ):
        raise AssemblyError("release-scale document counts do not match the frozen 20K contract")
    if sum(split_document_counts.values()) != document_count:
        raise AssemblyError("release split document counts do not sum to the total")

    for value, field_name in (
        (pii_free_document_count, "PII-free"),
        (hard_negative_document_count, "hard-negative"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AssemblyError(f"release {field_name} document count is invalid")
    required_pii_free = int(document_count * contract.pii_free_document_ratio)
    minimum_hard_negative = math.ceil(
        document_count * contract.hard_negative_document_ratio_minimum
    )
    if pii_free_document_count != required_pii_free:
        raise AssemblyError("release PII-free composition must be exactly 0.40")
    if not minimum_hard_negative <= hard_negative_document_count <= pii_free_document_count:
        raise AssemblyError("release hard-negative composition must be at least 0.25")

    expected_collisions = release_scale_zero_collision_counts()
    if (
        not isinstance(cross_split_group_collision_counts, Mapping)
        or set(cross_split_group_collision_counts) != set(expected_collisions)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value != 0
            for value in cross_split_group_collision_counts.values()
        )
    ):
        raise AssemblyError("release cross-split group leakage must be exactly zero")


def _release_negative_ratio_policy(policy: NegativeRatioPolicy | None) -> NegativeRatioPolicy:
    resolved = policy or NegativeRatioPolicy(
        minimum_pii_free=0.40,
        maximum_pii_free=0.40,
        minimum_hard_negative=0.25,
    )
    resolved.validate()
    if (
        not math.isclose(resolved.minimum_pii_free, 0.40, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(resolved.maximum_pii_free, 0.40, rel_tol=0.0, abs_tol=1e-12)
        or resolved.minimum_hard_negative < 0.25
    ):
        raise AssemblyError("release negative-ratio policy weakens the frozen contract")
    return resolved


@dataclass(frozen=True, slots=True)
class InternalGroups:
    """Private linkage fields used only for connected-component splitting."""

    document_group: str = field(repr=False)
    source_item_group: str = field(repr=False)
    near_duplicate_group: str = field(repr=False)
    tenant_group: str | None = field(default=None, repr=False)
    author_group: str | None = field(default=None, repr=False)
    conversation_group: str | None = field(default=None, repr=False)
    parent_group: str | None = field(default=None, repr=False)
    template_group: str | None = field(default=None, repr=False)

    def keys(self) -> tuple[tuple[str, str], ...]:
        values = (
            ("document", self.document_group),
            ("source_item", self.source_item_group),
            ("near_duplicate", self.near_duplicate_group),
            ("tenant", self.tenant_group),
            ("author", self.author_group),
            ("conversation", self.conversation_group),
            ("parent", self.parent_group),
            ("template", self.template_group),
        )
        return tuple((kind, value) for kind, value in values if value is not None)


@dataclass(frozen=True, slots=True)
class PrivacyProof:
    potential_raw_pii: bool
    replacement_complete: bool
    residual_scan_passed: bool
    relation_review_passed: bool
    human_review_passed: bool
    contains_real_personal_data_after_replacement: bool
    expected_edit_count: int
    source_text_sha256: str = field(repr=False)
    review_evidence_sha256: str = field(repr=False)
    surrogate_policy_version: str
    residual_scan_version: str


@dataclass(frozen=True, slots=True)
class AnnotationProof:
    quality_tier: QualityTier
    adjudication_status: str
    validators_passed: bool
    pii_free: bool
    hard_negative_categories: tuple[str, ...]
    length_tokens: int
    length_bucket: str
    annotation_revision: str


@dataclass(frozen=True, slots=True)
class PlannedSurrogateEdit:
    start: int
    end: int
    label: str
    relation_material: bytes = field(repr=False)
    replacements: Mapping[str, str] = field(repr=False)
    surface_state: str
    subject_kind: str
    validator_state: str
    risk_tiers: tuple[str, ...]
    policy_review_passed: bool


@dataclass(frozen=True, slots=True)
class AssemblyCandidate:
    registry_source_id: str
    context_origin: str
    record: DocumentRecord = field(repr=False)
    groups: InternalGroups = field(repr=False)
    privacy: PrivacyProof = field(repr=False)
    annotation: AnnotationProof
    edits: tuple[PlannedSurrogateEdit, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    records_by_split: Mapping[str, tuple[DocumentRecord, ...]]
    manifest: Mapping[str, Any]
    split_policy: SplitPolicy = field(repr=False)
    review_seal: str = field(repr=False)
    assembly_mode: str = DEVELOPMENT_ASSEMBLY_MODE


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AssemblyError(f"{field_name} must be a string-keyed object")
    return value


def _exact_fields(value: Mapping[str, Any], *, expected: frozenset[str], field_name: str) -> None:
    if set(value) != expected:
        raise AssemblyError(f"{field_name} does not match the private assembly schema")


def _string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AssemblyError(f"{field_name} must be a canonical non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise AssemblyError(f"{field_name} is not valid Unicode") from None
    if len(encoded) > _MAX_PRIVATE_FIELD_BYTES:
        raise AssemblyError(f"{field_name} exceeds the supported length")
    return value


def _nullable_private_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name=field_name)


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AssemblyError(f"{field_name} must be a boolean")
    return value


def _integer(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssemblyError(f"{field_name} must be an integer in the supported range")
    return value


def _digest(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise AssemblyError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _public_token(value: object, *, field_name: str) -> str:
    normalized = _string(value, field_name=field_name)
    if _PUBLIC_TOKEN_PATTERN.fullmatch(normalized) is None:
        raise AssemblyError(f"{field_name} is not a safe public token")
    return normalized


def _parse_groups(value: object) -> InternalGroups:
    raw = _mapping(value, field_name="groups")
    expected = frozenset(
        {
            "document_group",
            "source_item_group",
            "near_duplicate_group",
            "tenant_group",
            "author_group",
            "conversation_group",
            "parent_group",
            "template_group",
        }
    )
    _exact_fields(raw, expected=expected, field_name="groups")
    return InternalGroups(
        document_group=_string(raw["document_group"], field_name="groups.document_group"),
        source_item_group=_string(raw["source_item_group"], field_name="groups.source_item_group"),
        near_duplicate_group=_string(
            raw["near_duplicate_group"], field_name="groups.near_duplicate_group"
        ),
        tenant_group=_nullable_private_string(
            raw["tenant_group"], field_name="groups.tenant_group"
        ),
        author_group=_nullable_private_string(
            raw["author_group"], field_name="groups.author_group"
        ),
        conversation_group=_nullable_private_string(
            raw["conversation_group"], field_name="groups.conversation_group"
        ),
        parent_group=_nullable_private_string(
            raw["parent_group"], field_name="groups.parent_group"
        ),
        template_group=_nullable_private_string(
            raw["template_group"], field_name="groups.template_group"
        ),
    )


def _parse_privacy(value: object) -> PrivacyProof:
    raw = _mapping(value, field_name="privacy_proof")
    expected = frozenset(
        {
            "potential_raw_pii",
            "replacement_complete",
            "residual_scan_passed",
            "relation_review_passed",
            "human_review_passed",
            "contains_real_personal_data_after_replacement",
            "expected_edit_count",
            "source_text_sha256",
            "review_evidence_sha256",
            "surrogate_policy_version",
            "residual_scan_version",
        }
    )
    _exact_fields(raw, expected=expected, field_name="privacy_proof")
    return PrivacyProof(
        potential_raw_pii=_boolean(
            raw["potential_raw_pii"], field_name="privacy_proof.potential_raw_pii"
        ),
        replacement_complete=_boolean(
            raw["replacement_complete"], field_name="privacy_proof.replacement_complete"
        ),
        residual_scan_passed=_boolean(
            raw["residual_scan_passed"], field_name="privacy_proof.residual_scan_passed"
        ),
        relation_review_passed=_boolean(
            raw["relation_review_passed"], field_name="privacy_proof.relation_review_passed"
        ),
        human_review_passed=_boolean(
            raw["human_review_passed"], field_name="privacy_proof.human_review_passed"
        ),
        contains_real_personal_data_after_replacement=_boolean(
            raw["contains_real_personal_data_after_replacement"],
            field_name="privacy_proof.contains_real_personal_data_after_replacement",
        ),
        expected_edit_count=_integer(
            raw["expected_edit_count"], field_name="privacy_proof.expected_edit_count"
        ),
        source_text_sha256=_digest(
            raw["source_text_sha256"], field_name="privacy_proof.source_text_sha256"
        ),
        review_evidence_sha256=_digest(
            raw["review_evidence_sha256"],
            field_name="privacy_proof.review_evidence_sha256",
        ),
        surrogate_policy_version=_public_token(
            raw["surrogate_policy_version"],
            field_name="privacy_proof.surrogate_policy_version",
        ),
        residual_scan_version=_public_token(
            raw["residual_scan_version"],
            field_name="privacy_proof.residual_scan_version",
        ),
    )


def _length_bucket(length_tokens: int) -> str:
    if length_tokens <= 128:
        return "le_128"
    if length_tokens <= 512:
        return "129_512"
    if length_tokens <= 2048:
        return "513_2048"
    return "gt_2048"


def _parse_annotation(value: object) -> AnnotationProof:
    raw = _mapping(value, field_name="annotation_proof")
    expected = frozenset(
        {
            "quality_tier",
            "adjudication_status",
            "validators_passed",
            "pii_free",
            "hard_negative_categories",
            "length_tokens",
            "length_bucket",
            "annotation_revision",
        }
    )
    _exact_fields(raw, expected=expected, field_name="annotation_proof")
    try:
        tier = QualityTier(raw["quality_tier"])
    except (TypeError, ValueError):
        raise AssemblyError("annotation_proof.quality_tier is unsupported") from None
    if tier not in {QualityTier.GOLD, QualityTier.HUMAN_REVIEWED, QualityTier.HARD_NEGATIVE}:
        raise AssemblyError("annotation_proof.quality_tier is not trainable human gold")
    adjudication_status = _string(
        raw["adjudication_status"], field_name="annotation_proof.adjudication_status"
    )
    if adjudication_status not in _ADJUDICATION_STATES:
        raise AssemblyError("annotation_proof.adjudication_status is unsupported")
    raw_categories = raw["hard_negative_categories"]
    if isinstance(raw_categories, (str, bytes)) or not isinstance(raw_categories, Sequence):
        raise AssemblyError("annotation_proof.hard_negative_categories must be an array")
    categories = tuple(raw_categories)
    if (
        any(
            not isinstance(item, str) or _CATEGORY_PATTERN.fullmatch(item) is None
            for item in categories
        )
        or len(categories) != len(set(categories))
        or not set(categories).issubset(_HARD_NEGATIVE_CATEGORIES)
    ):
        raise AssemblyError("annotation_proof.hard_negative_categories is invalid")
    length_tokens = _integer(
        raw["length_tokens"], field_name="annotation_proof.length_tokens", minimum=1
    )
    length_bucket = _string(raw["length_bucket"], field_name="annotation_proof.length_bucket")
    if length_bucket != _length_bucket(length_tokens):
        raise AssemblyError("annotation_proof.length_bucket does not match the token count")
    return AnnotationProof(
        quality_tier=tier,
        adjudication_status=adjudication_status,
        validators_passed=_boolean(
            raw["validators_passed"], field_name="annotation_proof.validators_passed"
        ),
        pii_free=_boolean(raw["pii_free"], field_name="annotation_proof.pii_free"),
        hard_negative_categories=categories,
        length_tokens=length_tokens,
        length_bucket=length_bucket,
        annotation_revision=_public_token(
            raw["annotation_revision"], field_name="annotation_proof.annotation_revision"
        ),
    )


def _decode_relation_material(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise AssemblyError("surrogate edit relation material is invalid")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError):
        raise AssemblyError("surrogate edit relation material is invalid") from None
    if not decoded or len(decoded) > _MAX_RELATION_BYTES:
        raise AssemblyError("surrogate edit relation material is invalid")
    return decoded


def _parse_edit(value: object) -> PlannedSurrogateEdit:
    raw = _mapping(value, field_name="surrogate_edit")
    expected = frozenset(
        {
            "start",
            "end",
            "label",
            "relation_material_b64",
            "replacements",
            "surface_state",
            "subject_kind",
            "validator_state",
            "risk_tiers",
            "policy_review_passed",
        }
    )
    _exact_fields(raw, expected=expected, field_name="surrogate_edit")
    label = _string(raw["label"], field_name="surrogate_edit.label")
    if _LABEL_PATTERN.fullmatch(label) is None or label not in _CORE_LABELS:
        raise AssemblyError("surrogate_edit.label is invalid")
    replacements_raw = _mapping(raw["replacements"], field_name="surrogate_edit.replacements")
    _exact_fields(
        replacements_raw,
        expected=frozenset(SPLITS),
        field_name="surrogate_edit.replacements",
    )
    replacements: dict[str, str] = {}
    for split in SPLITS:
        replacement = _string(replacements_raw[split], field_name="surrogate_edit replacement")
        if "\x00" in replacement:
            raise AssemblyError("surrogate edit replacement is invalid")
        replacements[split] = replacement
    surface_state = _string(raw["surface_state"], field_name="surrogate_edit.surface_state")
    if surface_state not in _SURFACE_STATES:
        raise AssemblyError("surrogate_edit.surface_state is unsupported")
    subject_kind = _string(raw["subject_kind"], field_name="surrogate_edit.subject_kind")
    if subject_kind not in _SUBJECT_KINDS:
        raise AssemblyError("surrogate_edit.subject_kind is unsupported")
    validator_state = _string(raw["validator_state"], field_name="surrogate_edit.validator_state")
    if validator_state not in _VALIDATOR_STATES:
        raise AssemblyError("surrogate_edit.validator_state is unsupported")
    raw_risk_tiers = raw["risk_tiers"]
    if isinstance(raw_risk_tiers, (str, bytes)) or not isinstance(raw_risk_tiers, Sequence):
        raise AssemblyError("surrogate_edit.risk_tiers must be an array")
    risk_tiers = tuple(raw_risk_tiers)
    if (
        not risk_tiers
        or any(not isinstance(item, str) or item not in _RISK_TIERS for item in risk_tiers)
        or len(risk_tiers) != len(set(risk_tiers))
    ):
        raise AssemblyError("surrogate_edit.risk_tiers is invalid")
    return PlannedSurrogateEdit(
        start=_integer(raw["start"], field_name="surrogate_edit.start"),
        end=_integer(raw["end"], field_name="surrogate_edit.end", minimum=1),
        label=label,
        relation_material=_decode_relation_material(raw["relation_material_b64"]),
        replacements=MappingProxyType(replacements),
        surface_state=surface_state,
        subject_kind=subject_kind,
        validator_state=validator_state,
        risk_tiers=risk_tiers,
        policy_review_passed=_boolean(
            raw["policy_review_passed"], field_name="surrogate_edit.policy_review_passed"
        ),
    )


def _validate_candidate(candidate: AssemblyCandidate) -> None:
    try:
        candidate.record.validate()
    except Exception:
        raise AssemblyError("canonical candidate record failed validation") from None
    record = candidate.record
    try:
        record.text.encode("utf-8")
    except UnicodeEncodeError:
        raise AssemblyError("canonical candidate text is not valid Unicode") from None
    for name, value in (
        ("doc_id", record.doc_id),
        ("provenance.source_id", record.provenance.source_id),
        ("provenance.source_revision", record.provenance.source_revision),
        ("provenance.license", record.provenance.license),
    ):
        _string(value, field_name=f"record.{name}")
    if record.entities or record.split is not None:
        raise AssemblyError("canonical candidate must be unannotated and unsplit")
    if record.provenance.synthetic or record.provenance.evaluation_only:
        raise AssemblyError("canonical candidate is not eligible human context")
    if record.provenance.private_enterprise:
        raise AssemblyError("private-enterprise candidates cannot enter the public build")
    for name, value in (
        ("language", record.language),
        ("domain", record.domain),
        ("scene", record.scene),
    ):
        _public_token(value, field_name=f"record.{name}")
    if not record.language.casefold().startswith("zh"):
        raise AssemblyError("canonical candidate language is outside the v2 scope")
    recorded_origin = record.metadata.get("context_origin")
    if recorded_origin is not None and recorded_origin != candidate.context_origin:
        raise AssemblyError("candidate context origin does not match canonical metadata")
    if record.conversation_group is not None and (
        candidate.groups.conversation_group != record.conversation_group
    ):
        raise AssemblyError("candidate conversation linkage is incomplete")
    if record.tenant_group is not None and candidate.groups.tenant_group != record.tenant_group:
        raise AssemblyError("candidate tenant linkage is incomplete")
    if record.template_group is not None and (
        candidate.groups.template_group != record.template_group
    ):
        raise AssemblyError("candidate template linkage is incomplete")
    source_group = record.metadata.get("source_group")
    if isinstance(source_group, str) and source_group not in {
        candidate.groups.source_item_group,
        candidate.groups.conversation_group,
        candidate.groups.document_group,
    }:
        raise AssemblyError("candidate source-item linkage is incomplete")

    privacy = candidate.privacy
    expected_text_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(privacy.source_text_sha256, expected_text_hash):
        raise AssemblyError("privacy proof is not bound to the candidate text")
    if (
        not all(
            (
                privacy.replacement_complete,
                privacy.residual_scan_passed,
                privacy.relation_review_passed,
                privacy.human_review_passed,
            )
        )
        or privacy.contains_real_personal_data_after_replacement
    ):
        raise AssemblyError("privacy and replacement proof is incomplete")
    if privacy.residual_scan_version != _RESIDUAL_SCAN_VERSION:
        raise AssemblyError("privacy proof uses an unsupported residual scan version")
    if privacy.expected_edit_count != len(candidate.edits):
        raise AssemblyError("privacy proof edit count does not match the replacement plan")
    if not candidate.annotation.validators_passed:
        raise AssemblyError("annotation validators have not passed")

    ordered = sorted(candidate.edits, key=lambda edit: (edit.start, edit.end))
    previous_end = 0
    for edit in ordered:
        if not 0 <= edit.start < edit.end <= len(record.text):
            raise AssemblyError("surrogate edit lies outside the candidate text")
        if edit.start < previous_end:
            raise AssemblyError("surrogate edits overlap")
        previous_end = edit.end
        source_value = record.text[edit.start : edit.end]
        if not source_value.strip():
            raise AssemblyError("surrogate edit selects an empty source value")
        if not edit.policy_review_passed:
            raise AssemblyError("surrogate edit policy review has not passed")
        if any(replacement == source_value for replacement in edit.replacements.values()):
            raise AssemblyError("surrogate replacement retains a source value")
        if any(_contains_unsafe_residual(value) for value in edit.replacements.values()):
            raise AssemblyError("surrogate replacement fails the safe-namespace scan")

    annotation = candidate.annotation
    if annotation.pii_free:
        if candidate.edits or annotation.quality_tier is not QualityTier.HARD_NEGATIVE:
            raise AssemblyError("PII-free annotation proof conflicts with entity edits")
    elif not candidate.edits or annotation.quality_tier not in {
        QualityTier.GOLD,
        QualityTier.HUMAN_REVIEWED,
    }:
        raise AssemblyError("positive annotation proof conflicts with entity edits")
    if annotation.hard_negative_categories and not annotation.pii_free:
        raise AssemblyError("hard-negative categories require a PII-free record")
    if record.metadata.get("requires_surrogate_replacement") is True and not candidate.edits:
        raise AssemblyError("candidate requires surrogate replacement but has no edit plan")


def _candidate_from_dict_unsealed(value: Mapping[str, Any]) -> AssemblyCandidate:
    """Parse and structurally validate one candidate before digest verification."""

    raw = _mapping(value, field_name="private candidate")
    expected = frozenset(
        {
            "schema_version",
            "registry_source_id",
            "context_origin",
            "record",
            "groups",
            "privacy_proof",
            "annotation_proof",
            "surrogate_edits",
        }
    )
    _exact_fields(raw, expected=expected, field_name="private candidate")
    if raw["schema_version"] != PRIVATE_INPUT_SCHEMA_VERSION:
        raise AssemblyError("private candidate schema version is unsupported")
    context_origin = _string(raw["context_origin"], field_name="context_origin")
    if context_origin not in _CONTEXT_ORIGINS:
        raise AssemblyError("context_origin is unsupported")
    try:
        record_raw = _mapping(raw["record"], field_name="record")
        record = DocumentRecord.from_dict(record_raw)
    except (AssemblyError, SchemaError, TypeError, ValueError, KeyError):
        raise AssemblyError("canonical candidate record failed validation") from None
    raw_edits = raw["surrogate_edits"]
    if isinstance(raw_edits, (str, bytes)) or not isinstance(raw_edits, Sequence):
        raise AssemblyError("surrogate_edits must be an array")
    candidate = AssemblyCandidate(
        registry_source_id=_public_token(
            raw["registry_source_id"], field_name="registry_source_id"
        ),
        context_origin=context_origin,
        record=record,
        groups=_parse_groups(raw["groups"]),
        privacy=_parse_privacy(raw["privacy_proof"]),
        annotation=_parse_annotation(raw["annotation_proof"]),
        edits=tuple(_parse_edit(item) for item in raw_edits),
    )
    _validate_candidate(candidate)
    return candidate


def _validate_review_attestation_key(value: object) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise AssemblyError("review attestation key does not satisfy the v2 policy")
    return value


def _review_evidence_sha256(candidate: AssemblyCandidate, review_attestation_key: bytes) -> str:
    """Bind the review receipt to every release-relevant private input field."""

    digest = hmac.new(review_attestation_key, _REVIEW_PROTOCOL, hashlib.sha256)

    def update(name: str, value: str | bytes | bool | int | None) -> None:
        name_bytes = name.encode("ascii")
        if value is None:
            value_bytes = b"<null>"
        elif isinstance(value, bytes):
            value_bytes = value
        elif isinstance(value, bool):
            value_bytes = b"true" if value else b"false"
        elif isinstance(value, int):
            value_bytes = str(value).encode("ascii")
        else:
            value_bytes = value.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)

    record = candidate.record
    try:
        canonical_record = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise AssemblyError("canonical candidate cannot be bound to review evidence") from None
    update("record.canonical_sha256", hashlib.sha256(canonical_record).hexdigest())
    update("registry_source_id", candidate.registry_source_id)
    update("context_origin", candidate.context_origin)
    update("record.doc_id", record.doc_id)
    update("record.text_sha256", candidate.privacy.source_text_sha256)
    update("record.language", record.language)
    update("record.domain", record.domain)
    update("record.scene", record.scene)
    update("record.provenance.source_id", record.provenance.source_id)
    update("record.provenance.source_revision", record.provenance.source_revision)
    update("record.provenance.license", record.provenance.license)
    for kind, value in (
        ("document", candidate.groups.document_group),
        ("source_item", candidate.groups.source_item_group),
        ("near_duplicate", candidate.groups.near_duplicate_group),
        ("tenant", candidate.groups.tenant_group),
        ("author", candidate.groups.author_group),
        ("conversation", candidate.groups.conversation_group),
        ("parent", candidate.groups.parent_group),
        ("template", candidate.groups.template_group),
    ):
        update(f"groups.{kind}", value)
    privacy = candidate.privacy
    for name, value in (
        ("potential_raw_pii", privacy.potential_raw_pii),
        ("replacement_complete", privacy.replacement_complete),
        ("residual_scan_passed", privacy.residual_scan_passed),
        ("relation_review_passed", privacy.relation_review_passed),
        ("human_review_passed", privacy.human_review_passed),
        (
            "contains_real_personal_data_after_replacement",
            privacy.contains_real_personal_data_after_replacement,
        ),
        ("expected_edit_count", privacy.expected_edit_count),
        ("surrogate_policy_version", privacy.surrogate_policy_version),
        ("residual_scan_version", privacy.residual_scan_version),
    ):
        update(f"privacy.{name}", value)
    annotation = candidate.annotation
    for name, value in (
        ("quality_tier", annotation.quality_tier.value),
        ("adjudication_status", annotation.adjudication_status),
        ("validators_passed", annotation.validators_passed),
        ("pii_free", annotation.pii_free),
        ("length_tokens", annotation.length_tokens),
        ("length_bucket", annotation.length_bucket),
        ("annotation_revision", annotation.annotation_revision),
    ):
        update(f"annotation.{name}", value)
    for index, category in enumerate(annotation.hard_negative_categories):
        update(f"annotation.hard_negative_categories.{index}", category)
    for index, edit in enumerate(sorted(candidate.edits, key=lambda item: (item.start, item.end))):
        prefix = f"edits.{index}"
        update(f"{prefix}.start", edit.start)
        update(f"{prefix}.end", edit.end)
        update(f"{prefix}.label", edit.label)
        update(f"{prefix}.relation_material", edit.relation_material)
        for split in SPLITS:
            update(f"{prefix}.replacements.{split}", edit.replacements[split])
        update(f"{prefix}.surface_state", edit.surface_state)
        update(f"{prefix}.subject_kind", edit.subject_kind)
        update(f"{prefix}.validator_state", edit.validator_state)
        update(f"{prefix}.policy_review_passed", edit.policy_review_passed)
        for risk_index, risk_tier in enumerate(edit.risk_tiers):
            update(f"{prefix}.risk_tiers.{risk_index}", risk_tier)
    return digest.hexdigest()


def compute_review_evidence_sha256(
    value: Mapping[str, Any], *, review_attestation_key: bytes
) -> str:
    """Compute the immutable review-plan digest before sealing a private row."""

    key = _validate_review_attestation_key(review_attestation_key)
    candidate = _candidate_from_dict_unsealed(value)
    return _review_evidence_sha256(candidate, key)


def candidate_from_dict(
    value: Mapping[str, Any], *, review_attestation_key: bytes
) -> AssemblyCandidate:
    """Parse one sealed candidate without reflecting record values on failure."""

    key = _validate_review_attestation_key(review_attestation_key)
    candidate = _candidate_from_dict_unsealed(value)
    expected = _review_evidence_sha256(candidate, key)
    if not hmac.compare_digest(candidate.privacy.review_evidence_sha256, expected):
        raise AssemblyError("review evidence is not bound to the complete replacement plan")
    return candidate


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError("private candidate JSON contains duplicate fields")
        result[key] = value
    return result


def _open_private_text(path: Path) -> TextIO:
    descriptor = -1
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        file_stat = os.fstat(descriptor)
    except OSError:
        raise AssemblyError("private candidate input is unavailable") from None
    try:
        if not stat.S_ISREG(file_stat.st_mode):
            raise AssemblyError("private candidate input must be a regular non-symlink file")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise AssemblyError("private candidate input permissions are too broad")
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def load_private_candidates(
    path: str | Path, *, review_attestation_key: bytes
) -> tuple[AssemblyCandidate, ...]:
    """Load a mode-0600-style private JSONL file with duplicate-key rejection."""

    source = Path(path)
    candidates: list[AssemblyCandidate] = []
    try:
        with _open_private_text(source) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                    if not isinstance(value, Mapping):
                        raise AssemblyError("private candidate row must be an object")
                    candidates.append(
                        candidate_from_dict(value, review_attestation_key=review_attestation_key)
                    )
                except (AssemblyError, json.JSONDecodeError, UnicodeError):
                    raise AssemblyError("private candidate row failed validation") from None
    except AssemblyError:
        raise
    except UnicodeError:
        raise AssemblyError("private candidate input is not valid UTF-8 JSONL") from None
    except OSError:
        raise AssemblyError("private candidate input could not be read") from None
    if not candidates:
        raise AssemblyError("private candidate input is empty")
    return tuple(candidates)


def ensure_registry_has_approved_sources(registry: CandidateSourceRegistry) -> None:
    """Reject the current all-pending registry before private inputs are opened."""

    try:
        allowed = any(
            evaluate_registered_source(registry, source_id).public_release_allowed
            for source_id in registry.sources
        )
    except Exception:
        raise AssemblyError("source registry admission evaluation failed") from None
    if not allowed:
        raise AssemblyError("raw_source_blocked; derived_source_pending")


def _validate_source_admission(
    candidate: AssemblyCandidate, registry: CandidateSourceRegistry
) -> None:
    try:
        source = registry.source(candidate.registry_source_id)
        decision = evaluate_registered_source(registry, candidate.registry_source_id)
    except Exception:
        raise AssemblyError("candidate source is not approved for public training") from None
    if not decision.public_release_allowed:
        raise AssemblyError("raw_source_blocked; derived_source_pending")
    record = candidate.record
    if (
        source.immutable_revision is None
        or record.provenance.source_id != candidate.registry_source_id
        or record.provenance.source_revision != source.immutable_revision
        or record.provenance.license.casefold() != source.declared_license.casefold()
        or record.license.casefold() != source.declared_license.casefold()
    ):
        raise AssemblyError("candidate provenance does not match the approved source revision")
    if record.public_weight_training_allowed is False:
        raise AssemblyError("candidate record explicitly forbids public weight training")
    if record.data_pool in {DataPool.EVALUATION_ONLY, DataPool.PRIVATE_ENTERPRISE}:
        raise AssemblyError("candidate record belongs to an ineligible data pool")


def _security_normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _placeholder_fingerprint(candidate: AssemblyCandidate) -> str:
    parts: list[str] = []
    cursor = 0
    for edit in sorted(candidate.edits, key=lambda item: (item.start, item.end)):
        parts.append(candidate.record.text[cursor : edit.start])
        parts.append(f"<PII:{edit.label}>")
        cursor = edit.end
    parts.append(candidate.record.text[cursor:])
    normalized = " ".join(_security_normalize("".join(parts)).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _candidate_linkage_keys(
    candidate: AssemblyCandidate,
) -> tuple[tuple[str, str | bytes], ...]:
    keys: list[tuple[str, str | bytes]] = [
        (
            "record",
            f"{candidate.registry_source_id}\0{candidate.record.doc_id}",
        ),
        ("exact_placeholder", _placeholder_fingerprint(candidate)),
    ]
    keys.extend(candidate.groups.keys())
    keys.extend(
        (
            f"relation:{candidate.registry_source_id}:{candidate.record.domain}",
            edit.relation_material,
        )
        for edit in candidate.edits
    )
    return tuple(keys)


def _component_signature(
    candidates: Sequence[AssemblyCandidate], component: Sequence[int], seed: int
) -> str:
    digest = hashlib.sha256()
    digest.update(b"pii-zh/context-v2/component/v1\0")
    digest.update(str(seed).encode("ascii"))
    identities = sorted(
        (
            candidates[index].registry_source_id.encode("utf-8"),
            candidates[index].record.doc_id.encode("utf-8"),
        )
        for index in component
    )
    for source_id, doc_id in identities:
        digest.update(len(source_id).to_bytes(8, "big"))
        digest.update(source_id)
        digest.update(len(doc_id).to_bytes(8, "big"))
        digest.update(doc_id)
    return digest.hexdigest()


def _connected_components(candidates: Sequence[AssemblyCandidate]) -> list[list[int]]:
    union_find = _UnionFind(len(candidates))
    first_by_key: dict[tuple[str, str | bytes], int] = {}
    for index, candidate in enumerate(candidates):
        for key in _candidate_linkage_keys(candidate):
            previous = first_by_key.setdefault(key, index)
            union_find.union(index, previous)
    components: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        components.setdefault(union_find.find(index), []).append(index)
    return list(components.values())


def _assign_splits(
    candidates: Sequence[AssemblyCandidate], *, split_policy: SplitPolicy, seed: int
) -> tuple[dict[int, str], int]:
    ratios = split_policy.ratios()
    components = _connected_components(candidates)
    largest_component = max(len(component) for component in components)
    signatures = {
        tuple(component): _component_signature(candidates, component, seed)
        for component in components
    }
    component_stats = {
        tuple(component): (
            sum(candidates[index].annotation.pii_free for index in component),
            sum(bool(candidates[index].annotation.hard_negative_categories) for index in component),
        )
        for component in components
    }
    components.sort(
        key=lambda component: (
            -int(component_stats[tuple(component)][1] > 0),
            -int(component_stats[tuple(component)][0] > 0),
            -len(component),
            signatures[tuple(component)],
        )
    )
    targets = {name: ratio * len(candidates) for name, ratio in ratios.items()}
    if largest_component > math.ceil(max(targets.values())):
        raise AssemblyError("a leakage component makes the configured split infeasible")
    assigned_counts = {name: 0 for name in ratios}
    total_pii_free = sum(candidate.annotation.pii_free for candidate in candidates)
    total_hard_negative = sum(
        bool(candidate.annotation.hard_negative_categories) for candidate in candidates
    )
    pii_free_targets = {name: ratio * total_pii_free for name, ratio in ratios.items()}
    hard_negative_targets = {name: ratio * total_hard_negative for name, ratio in ratios.items()}
    assigned_pii_free = {name: 0 for name in ratios}
    assigned_hard_negative = {name: 0 for name in ratios}
    split_tiebreakers = {
        name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest() for name in ratios
    }
    assignments: dict[int, str] = {}
    for component in components:
        pii_free_count, hard_negative_count = component_stats[tuple(component)]

        def assignment_cost(
            name: str,
            component_size: int = len(component),
            component_pii_free: int = pii_free_count,
            component_hard_negative: int = hard_negative_count,
        ) -> tuple[float, str]:
            cost = 0.0
            for split_name in ratios:
                added_documents = component_size if split_name == name else 0
                added_pii_free = component_pii_free if split_name == name else 0
                added_hard = component_hard_negative if split_name == name else 0
                document_error = (
                    assigned_counts[split_name] + added_documents - targets[split_name]
                ) / max(1.0, targets[split_name])
                pii_free_error = (
                    assigned_pii_free[split_name] + added_pii_free - pii_free_targets[split_name]
                ) / max(1.0, pii_free_targets[split_name])
                hard_error = (
                    assigned_hard_negative[split_name]
                    + added_hard
                    - hard_negative_targets[split_name]
                ) / max(1.0, hard_negative_targets[split_name])
                cost += 4.0 * document_error**2 + 2.0 * pii_free_error**2 + hard_error**2
            return cost, split_tiebreakers[name]

        split = min(
            ratios,
            key=assignment_cost,
        )
        for index in component:
            assignments[index] = split
        assigned_counts[split] += len(component)
        assigned_pii_free[split] += pii_free_count
        assigned_hard_negative[split] += hard_negative_count
    _assert_internal_no_leakage(candidates, assignments)
    if any(assigned_counts[name] == 0 for name in ratios):
        raise AssemblyError("configured validation/test/hidden splits must not be empty")
    if any(
        abs(assigned_counts[name] - targets[name]) > max(1, largest_component) for name in ratios
    ):
        raise AssemblyError("grouped split deviates beyond the component-size tolerance")
    return assignments, len(components)


def _assert_internal_no_leakage(
    candidates: Sequence[AssemblyCandidate], assignments: Mapping[int, str]
) -> None:
    observed: dict[tuple[str, str | bytes], str] = {}
    for index, candidate in enumerate(candidates):
        split = assignments[index]
        for key in _candidate_linkage_keys(candidate):
            previous = observed.setdefault(key, split)
            if previous != split:
                raise AssemblyError("grouped split leakage invariant failed")


def _opaque_id(key: bytes, prefix: str, *parts: str) -> str:
    framed = bytearray(_ID_PROTOCOL)
    framed.extend(len(prefix).to_bytes(4, "big"))
    framed.extend(prefix.encode("ascii"))
    for part in parts:
        encoded = part.encode("utf-8")
        framed.extend(len(encoded).to_bytes(8, "big"))
        framed.extend(encoded)
    digest = hmac.new(key, bytes(framed), hashlib.sha256).digest()[:_OPAQUE_ID_BYTES]
    encoded_digest = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"{prefix}_v2_{encoded_digest}"


def _opaque_group(key: bytes, *, split: str, kind: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _opaque_id(key, "grp", split, kind, value)


def _validate_negative_ratios(
    candidates: Sequence[AssemblyCandidate], policy: NegativeRatioPolicy
) -> tuple[float, float, float]:
    policy.validate()
    total = len(candidates)
    pii_free_count = sum(candidate.annotation.pii_free for candidate in candidates)
    hard_negative_count = sum(
        bool(candidate.annotation.hard_negative_categories) for candidate in candidates
    )
    pii_free_ratio = pii_free_count / total
    hard_negative_ratio = hard_negative_count / total
    ordinary_negative_ratio = (pii_free_count - hard_negative_count) / total
    tolerance = 1e-12
    if not (
        policy.minimum_pii_free - tolerance <= pii_free_ratio <= policy.maximum_pii_free + tolerance
    ):
        raise AssemblyError("PII-free document ratio is outside the approved range")
    if hard_negative_ratio + tolerance < policy.minimum_hard_negative:
        raise AssemblyError("hard-negative document ratio is below the approved minimum")
    return pii_free_ratio, hard_negative_ratio, ordinary_negative_ratio


def _validate_split_negative_ratios(
    candidates: Sequence[AssemblyCandidate],
    assignments: Mapping[int, str],
    policy: NegativeRatioPolicy,
) -> None:
    for split in SPLITS:
        indices = [index for index, assigned in assignments.items() if assigned == split]
        count = len(indices)
        if count < 1:
            raise AssemblyError("configured validation/test/hidden splits must not be empty")
        pii_free_count = sum(candidates[index].annotation.pii_free for index in indices)
        hard_negative_count = sum(
            bool(candidates[index].annotation.hard_negative_categories) for index in indices
        )
        minimum_pii_free = math.floor(policy.minimum_pii_free * count)
        maximum_pii_free = math.ceil(policy.maximum_pii_free * count)
        minimum_hard_negative = math.floor(policy.minimum_hard_negative * count)
        if not minimum_pii_free <= pii_free_count <= maximum_pii_free:
            raise AssemblyError("one split violates the rounded PII-free composition gate")
        if hard_negative_count < minimum_hard_negative:
            raise AssemblyError("one split violates the rounded hard-negative composition gate")


def _is_safe_documentation_ip(value: str) -> bool:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return True
    if parsed.is_loopback or parsed.is_unspecified:
        return True
    documentation_networks = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
        ipaddress.ip_network("2001:db8::/32"),
    )
    return any(parsed in network for network in documentation_networks)


def _contains_unsafe_residual(text: str) -> bool:
    """Conservative final-text scan independent of self-reported scan booleans."""

    text = unicodedata.normalize("NFKC", text)
    if (
        _RAW_SENTINEL_PATTERN.search(text)
        or _PHONE_PATTERN.search(text)
        or _CN_ID_PATTERN.search(text)
        or any(pattern.search(text) for pattern in _SECRET_PATTERNS)
    ):
        return True
    for match in _EMAIL_PATTERN.finditer(text):
        if not match.group(0).casefold().endswith(".invalid"):
            return True
    for match in _BANK_DIGITS_PATTERN.finditer(text):
        normalized = re.sub(r"[ -]", "", match.group(0))
        if validate_luhn(normalized).valid:
            return True
    for match in _IP_TOKEN_PATTERN.finditer(text):
        if not _is_safe_documentation_ip(match.group(0)):
            return True
    for match in _MAC_PATTERN.finditer(text):
        first_octet = int(match.group(0)[:2], 16)
        if not (first_octet & 0b10 or first_octet & 0b1):
            return True
    return False


def _create_public_document(**kwargs: Any) -> DocumentRecord:
    try:
        return DocumentRecord.create(**kwargs)
    except Exception:
        raise AssemblyError("assembled public record failed validation") from None


def _transform_candidate(
    candidate: AssemblyCandidate,
    *,
    split: str,
    key: bytes,
    deriver: SurrogateKeyDeriver,
    replacement_owners: dict[str, tuple[str, str, str, str]],
    relation_replacements: dict[tuple[str, str, str, str], str],
    registry_version: str,
) -> DocumentRecord:
    annotation = candidate.annotation
    if split != "train" and annotation.adjudication_status != "adjudicated":
        raise AssemblyError("evaluation split contains non-adjudicated gold")
    if split != "train" and annotation.quality_tier is QualityTier.HUMAN_REVIEWED:
        raise AssemblyError("evaluation split contains G1-only gold")

    ordered = sorted(candidate.edits, key=lambda edit: (edit.start, edit.end))
    surrogate_edits: list[SurrogateEdit] = []
    relation_ids: list[str] = []
    for edit in ordered:
        try:
            derivation = deriver.derive(
                split=split,
                label=edit.label,
                scope=(f"source:{candidate.registry_source_id}|domain:{candidate.record.domain}"),
                relation=edit.relation_material,
            )
        except Exception:
            raise AssemblyError("surrogate relation derivation failed") from None
        replacement = edit.replacements[split]
        namespace = (
            split,
            candidate.record.domain,
            edit.label,
            derivation.relation_id,
        )
        previous_owner = replacement_owners.setdefault(_security_normalize(replacement), namespace)
        if previous_owner != namespace:
            raise AssemblyError("surrogate replacement crosses a split/label/domain namespace")
        relation_key = (split, candidate.record.domain, edit.label, derivation.relation_id)
        previous_replacement = relation_replacements.setdefault(relation_key, replacement)
        if previous_replacement != replacement:
            raise AssemblyError("one surrogate relation has inconsistent replacements")
        source_value = candidate.record.text[edit.start : edit.end]
        surrogate_edits.append(
            SurrogateEdit(
                span=OriginalSpan(
                    start=edit.start,
                    end=edit.end,
                    label=edit.label,
                    text=source_value,
                ),
                replacement=replacement,
                relation_id=derivation.relation_id,
            )
        )
        relation_ids.append(derivation.relation_id)
    try:
        rebuilt = rebuild_surrogate_text(candidate.record.text, surrogate_edits)
    except Exception:
        raise AssemblyError("surrogate text reconstruction failed") from None
    if _contains_unsafe_residual(rebuilt.text):
        raise AssemblyError("independent final-text residual scan failed")
    if candidate.privacy.potential_raw_pii:
        source_values = {candidate.record.text[edit.start : edit.end] for edit in ordered}
        if any(source_value in rebuilt.text for source_value in source_values):
            raise AssemblyError("surrogate output retains a reviewed source value")

    public_doc_id = _opaque_id(
        key,
        "doc",
        split,
        candidate.registry_source_id,
        candidate.record.doc_id,
    )
    entities: list[EntitySpan] = []
    for index, (planned, span, relation_id) in enumerate(
        zip(ordered, rebuilt.spans, relation_ids, strict=True)
    ):
        annotation_id = _opaque_id(
            key,
            "ann",
            split,
            public_doc_id,
            str(index),
            planned.label,
        )
        entities.append(
            EntitySpan(
                start=span.start,
                end=span.end,
                label=span.label,
                text=span.text,
                source="surrogate_gold",
                risk_tier=planned.risk_tiers[0],
                validator_state=planned.validator_state,
                metadata={
                    "annotation_id": annotation_id,
                    "risk_tiers": list(planned.risk_tiers),
                    "surface_state": planned.surface_state,
                    "subject_kind": planned.subject_kind,
                    "validator_state": planned.validator_state,
                    "relation_group_id": relation_id,
                    "adjudication_status": annotation.adjudication_status,
                    "attributes": {},
                },
            )
        )

    groups = candidate.groups
    provenance = Provenance(
        source_id=candidate.registry_source_id,
        source_kind="human_authored_context",
        source_revision=candidate.record.provenance.source_revision,
        license=candidate.record.provenance.license,
        synthetic=False,
        metadata={
            "context_origin": candidate.context_origin,
            "contains_real_personal_data": False,
            "privacy_review_status": "passed",
            "source_registry_version": registry_version,
        },
    )
    quality = QualityMetadata(
        tier=annotation.quality_tier,
        quality_gate=True,
        validators_passed=True,
        review_status=annotation.adjudication_status,
        confidence=None,
        metadata={
            "annotation_revision": annotation.annotation_revision,
            "pii_free": annotation.pii_free,
        },
    )
    record = _create_public_document(
        doc_id=public_doc_id,
        text=rebuilt.text,
        entities=entities,
        language=candidate.record.language,
        domain=candidate.record.domain,
        scene=candidate.record.scene,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        tenant_group=_opaque_group(key, split=split, kind="tenant", value=groups.tenant_group),
        conversation_group=_opaque_group(
            key,
            split=split,
            kind="conversation",
            value=groups.conversation_group,
        ),
        template_group=_opaque_group(
            key, split=split, kind="template", value=groups.template_group
        ),
        entity_value_groups=(),
        split=split,
        metadata={
            "context_origin": candidate.context_origin,
            "document_group": _opaque_group(
                key, split=split, kind="document", value=groups.document_group
            ),
            "source_item_group": _opaque_group(
                key,
                split=split,
                kind="source_item",
                value=groups.source_item_group,
            ),
            "author_group": _opaque_group(
                key, split=split, kind="author", value=groups.author_group
            ),
            "near_duplicate_group": _opaque_group(
                key,
                split=split,
                kind="near_duplicate",
                value=groups.near_duplicate_group,
            ),
            "exact_placeholder_group": _opaque_group(
                key,
                split=split,
                kind="exact_placeholder",
                value=_placeholder_fingerprint(candidate),
            ),
            "parent_group": _opaque_group(
                key, split=split, kind="parent", value=groups.parent_group
            ),
            "surrogate_policy_version": candidate.privacy.surrogate_policy_version,
            "privacy_review_status": "passed",
            "residual_scan_version": _RESIDUAL_SCAN_VERSION,
            "contains_real_personal_data": False,
            "pii_free": annotation.pii_free,
            "hard_negative_categories": list(annotation.hard_negative_categories),
            "length_tokens": annotation.length_tokens,
            "length_bucket": annotation.length_bucket,
            "annotation_revision": annotation.annotation_revision,
        },
    )
    try:
        record.validate()
    except Exception:
        raise AssemblyError("assembled public record failed validation") from None
    return record


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[DocumentRecord]) -> bytes:
    try:
        return "".join(dumps_document(record) + "\n" for record in records).encode("utf-8")
    except Exception:
        raise AssemblyError("assembled JSONL serialization failed") from None


def _public_record_linkage_keys(record: DocumentRecord) -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = [("document", record.doc_id)]
    for kind, value in (
        ("tenant", record.tenant_group),
        ("conversation", record.conversation_group),
        ("template", record.template_group),
        ("document_group", record.metadata.get("document_group")),
        ("source_item", record.metadata.get("source_item_group")),
        ("author", record.metadata.get("author_group")),
        ("near_duplicate", record.metadata.get("near_duplicate_group")),
        ("exact_placeholder", record.metadata.get("exact_placeholder_group")),
        ("parent", record.metadata.get("parent_group")),
    ):
        if isinstance(value, str):
            keys.append((kind, value))
    for entity in record.entities:
        relation_id = entity.metadata.get("relation_group_id")
        if isinstance(relation_id, str):
            keys.append(("relation", relation_id))
    return tuple(keys)


def _public_component_count(records_by_split: Mapping[str, tuple[DocumentRecord, ...]]) -> int:
    records = [record for split in SPLITS for record in records_by_split[split]]
    union_find = _UnionFind(len(records))
    first_by_key: dict[tuple[str, str], int] = {}
    split_by_key: dict[tuple[str, str], str] = {}
    for index, record in enumerate(records):
        if record.split not in SPLITS:
            raise AssemblyError("assembled record has an invalid split")
        for key in _public_record_linkage_keys(record):
            previous_split = split_by_key.setdefault(key, record.split)
            if previous_split != record.split:
                raise AssemblyError("assembled public linkage crosses splits")
            previous = first_by_key.setdefault(key, index)
            union_find.union(index, previous)
    return len({union_find.find(index) for index in range(len(records))})


def _require_public_group(value: object, *, required: bool) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or _GROUP_ID_PATTERN.fullmatch(value) is None:
        raise AssemblyError("assembled public group identifier is invalid")


def _validate_release_record(record: DocumentRecord, *, split: str) -> None:
    if (
        record.data_pool is not DataPool.PUBLIC_RELEASE
        or record.public_weight_training_allowed is not True
        or not record.quality.quality_gate
        or not record.quality.validators_passed
    ):
        raise AssemblyError("assembled record is outside the public training pool")
    if _DOC_ID_PATTERN.fullmatch(record.doc_id) is None:
        raise AssemblyError("assembled public document identifier is invalid")
    if not record.text or _contains_unsafe_residual(record.text):
        raise AssemblyError("assembled record fails the independent final-text scan")
    for name, value in (
        ("language", record.language),
        ("domain", record.domain),
        ("scene", record.scene),
    ):
        _public_token(value, field_name=f"assembled record {name}")

    provenance = record.provenance
    if (
        provenance.synthetic
        or provenance.evaluation_only
        or provenance.private_enterprise
        or provenance.source_kind != "human_authored_context"
        or _PUBLIC_TOKEN_PATTERN.fullmatch(provenance.source_id) is None
        or _REVISION_PATTERN.fullmatch(provenance.source_revision) is None
        or provenance.license.upper() not in _PUBLIC_WEIGHT_LICENSES
        or any(
            value is not None
            for value in (
                provenance.generator_name,
                provenance.generator_version,
                provenance.generator_seed,
                provenance.template_id,
                provenance.terms_snapshot,
            )
        )
    ):
        raise AssemblyError("assembled public provenance is invalid")
    if set(provenance.metadata) != {
        "context_origin",
        "contains_real_personal_data",
        "privacy_review_status",
        "source_registry_version",
    }:
        raise AssemblyError("assembled public provenance metadata is invalid")
    if (
        provenance.metadata.get("context_origin") not in _CONTEXT_ORIGINS
        or provenance.metadata.get("contains_real_personal_data") is not False
        or provenance.metadata.get("privacy_review_status") != "passed"
    ):
        raise AssemblyError("assembled public provenance privacy gate failed")
    _public_token(
        provenance.metadata.get("source_registry_version"),
        field_name="assembled source registry version",
    )

    metadata = record.metadata
    expected_metadata = {
        "context_origin",
        "document_group",
        "source_item_group",
        "author_group",
        "near_duplicate_group",
        "exact_placeholder_group",
        "parent_group",
        "surrogate_policy_version",
        "privacy_review_status",
        "residual_scan_version",
        "contains_real_personal_data",
        "pii_free",
        "hard_negative_categories",
        "length_tokens",
        "length_bucket",
        "annotation_revision",
    }
    if set(metadata) != expected_metadata:
        raise AssemblyError("assembled public document metadata is invalid")
    if (
        metadata.get("context_origin") != provenance.metadata.get("context_origin")
        or metadata.get("privacy_review_status") != "passed"
        or metadata.get("residual_scan_version") != _RESIDUAL_SCAN_VERSION
        or metadata.get("contains_real_personal_data") is not False
    ):
        raise AssemblyError("assembled public document privacy gate failed")
    _public_token(
        metadata.get("surrogate_policy_version"),
        field_name="assembled surrogate policy version",
    )
    _public_token(
        metadata.get("annotation_revision"),
        field_name="assembled annotation revision",
    )
    length_tokens = metadata.get("length_tokens")
    if (
        isinstance(length_tokens, bool)
        or not isinstance(length_tokens, int)
        or length_tokens < 1
        or metadata.get("length_bucket") != _length_bucket(length_tokens)
    ):
        raise AssemblyError("assembled public length metadata is invalid")
    for value, required in (
        (record.tenant_group, False),
        (record.conversation_group, False),
        (record.template_group, False),
        (metadata.get("document_group"), True),
        (metadata.get("source_item_group"), True),
        (metadata.get("author_group"), False),
        (metadata.get("near_duplicate_group"), True),
        (metadata.get("exact_placeholder_group"), True),
        (metadata.get("parent_group"), False),
    ):
        _require_public_group(value, required=required)
    if (
        record.entity_value_groups
        or record.created_at_bucket is not None
        or record.generator_version
    ):
        raise AssemblyError("assembled public record retains unsupported grouping metadata")

    pii_free = metadata.get("pii_free")
    if not isinstance(pii_free, bool) or pii_free != (not record.entities):
        raise AssemblyError("assembled PII-free state conflicts with gold entities")
    categories = metadata.get("hard_negative_categories")
    if (
        not isinstance(categories, list)
        or any(not isinstance(item, str) for item in categories)
        or len(categories) != len(set(categories))
        or not set(categories).issubset(_HARD_NEGATIVE_CATEGORIES)
        or (categories and not pii_free)
    ):
        raise AssemblyError("assembled hard-negative categories are invalid")
    if set(record.quality.metadata) != {"annotation_revision", "pii_free"} or (
        record.quality.metadata.get("annotation_revision") != metadata.get("annotation_revision")
        or record.quality.metadata.get("pii_free") is not pii_free
        or record.quality.review_status not in _ADJUDICATION_STATES
        or record.quality.confidence is not None
    ):
        raise AssemblyError("assembled public quality metadata is invalid")
    if pii_free and record.quality.tier is not QualityTier.HARD_NEGATIVE:
        raise AssemblyError("assembled negative quality tier is invalid")
    if not pii_free and record.quality.tier not in {
        QualityTier.GOLD,
        QualityTier.HUMAN_REVIEWED,
    }:
        raise AssemblyError("assembled positive quality tier is invalid")
    if split != "train" and (
        record.quality.tier is QualityTier.HUMAN_REVIEWED
        or record.quality.review_status != "adjudicated"
    ):
        raise AssemblyError("assembled evaluation record is not adjudicated gold")

    expected_entity_metadata = {
        "annotation_id",
        "risk_tiers",
        "surface_state",
        "subject_kind",
        "validator_state",
        "relation_group_id",
        "adjudication_status",
        "attributes",
    }
    for entity in record.entities:
        entity_metadata = entity.metadata
        risk_tiers = entity_metadata.get("risk_tiers")
        if (
            entity.label not in _CORE_LABELS
            or entity.source != "surrogate_gold"
            or set(entity_metadata) != expected_entity_metadata
            or _ANNOTATION_ID_PATTERN.fullmatch(str(entity_metadata.get("annotation_id"))) is None
            or _RELATION_ID_PATTERN.fullmatch(str(entity_metadata.get("relation_group_id"))) is None
            or not isinstance(risk_tiers, list)
            or not risk_tiers
            or any(item not in _RISK_TIERS for item in risk_tiers)
            or len(risk_tiers) != len(set(risk_tiers))
            or entity.risk_tier != risk_tiers[0]
            or entity_metadata.get("surface_state") not in _SURFACE_STATES
            or entity_metadata.get("subject_kind") not in _SUBJECT_KINDS
            or entity_metadata.get("validator_state") not in _VALIDATOR_STATES
            or entity.validator_state != entity_metadata.get("validator_state")
            or entity_metadata.get("adjudication_status") != record.quality.review_status
            or entity_metadata.get("attributes") != {}
        ):
            raise AssemblyError("assembled public entity metadata is invalid")


def _validate_public_composition(
    records_by_split: Mapping[str, tuple[DocumentRecord, ...]],
) -> None:
    policy = NegativeRatioPolicy()
    all_records = [record for split in SPLITS for record in records_by_split[split]]
    total = len(all_records)
    pii_free_count = sum(record.metadata["pii_free"] is True for record in all_records)
    hard_count = sum(bool(record.metadata["hard_negative_categories"]) for record in all_records)
    pii_free_ratio = pii_free_count / total
    hard_ratio = hard_count / total
    if not policy.minimum_pii_free <= pii_free_ratio <= policy.maximum_pii_free:
        raise AssemblyError("assembled PII-free composition is outside the v2 baseline")
    if hard_ratio < policy.minimum_hard_negative:
        raise AssemblyError("assembled hard-negative composition is outside the v2 baseline")
    for split in SPLITS:
        records = records_by_split[split]
        count = len(records)
        split_pii_free = sum(record.metadata["pii_free"] is True for record in records)
        split_hard = sum(bool(record.metadata["hard_negative_categories"]) for record in records)
        if not (
            math.floor(policy.minimum_pii_free * count)
            <= split_pii_free
            <= math.ceil(policy.maximum_pii_free * count)
        ):
            raise AssemblyError("assembled split PII-free composition is invalid")
        if split_hard < math.floor(policy.minimum_hard_negative * count):
            raise AssemblyError("assembled split hard-negative composition is invalid")


def _validate_records_by_split(
    records_by_split: Mapping[str, tuple[DocumentRecord, ...]],
) -> None:
    if set(records_by_split) != set(SPLITS):
        raise AssemblyError("assembled split containers do not match the frozen contract")
    seen_doc_ids: set[str] = set()
    for split in SPLITS:
        records = records_by_split[split]
        if not isinstance(records, tuple):
            raise AssemblyError("assembled split container must be an immutable tuple")
        for record in records:
            if not isinstance(record, DocumentRecord):
                raise AssemblyError("assembled split contains an invalid record")
            if record.split != split:
                raise AssemblyError("assembled record split does not match its container")
            if record.doc_id in seen_doc_ids:
                raise AssemblyError("assembled dataset contains a duplicate public document")
            seen_doc_ids.add(record.doc_id)
            try:
                record.validate()
            except Exception:
                raise AssemblyError("assembled public record failed validation") from None
            _validate_release_record(record, split=split)
    _validate_public_composition(records_by_split)


def _build_manifest(
    *,
    records_by_split: Mapping[str, tuple[DocumentRecord, ...]],
    split_policy: SplitPolicy,
    assembly_mode: str = DEVELOPMENT_ASSEMBLY_MODE,
) -> dict[str, Any]:
    mode = _normalize_assembly_mode(assembly_mode)
    _validate_records_by_split(records_by_split)
    total = sum(len(records) for records in records_by_split.values())
    if total < 1:
        raise AssemblyError("assembled dataset is empty")
    split_counts = {split: len(records_by_split[split]) for split in SPLITS}
    if any(count < 1 for count in split_counts.values()):
        raise AssemblyError("assembled validation/test/hidden splits must not be empty")
    label_distribution = {
        split: dict(
            sorted(
                Counter(entity.label for record in records for entity in record.entities).items()
            )
        )
        for split, records in records_by_split.items()
    }
    artifacts = {
        split: {
            "record_count": len(records_by_split[split]),
            "sha256": hashlib.sha256(_jsonl_bytes(records_by_split[split])).hexdigest(),
        }
        for split in SPLITS
    }
    pii_free_count = sum(
        record.metadata.get("pii_free") is True
        for records in records_by_split.values()
        for record in records
    )
    hard_negative_count = sum(
        bool(record.metadata.get("hard_negative_categories"))
        for records in records_by_split.values()
        for record in records
    )
    pii_free_ratio = pii_free_count / total
    hard_negative_ratio = hard_negative_count / total
    ordinary_negative_ratio = (pii_free_count - hard_negative_count) / total
    pii_free_by_split = {
        split: sum(record.metadata.get("pii_free") is True for record in records) / len(records)
        for split, records in records_by_split.items()
    }
    hard_negative_by_split = {
        split: sum(bool(record.metadata.get("hard_negative_categories")) for record in records)
        / len(records)
        for split, records in records_by_split.items()
    }
    component_count = _public_component_count(records_by_split)
    manifest: dict[str, Any] = {
        "schema_version": (
            RELEASE_MANIFEST_SCHEMA_VERSION
            if mode == RELEASE_ASSEMBLY_MODE
            else MANIFEST_SCHEMA_VERSION
        ),
        "status": "completed",
        "counts": {
            "document_count": total,
            "connected_component_count": component_count,
            "split_document_counts": split_counts,
        },
        "ratios": {
            "configured_split_ratios": split_policy.ratios(),
            "observed_split_ratios": {split: split_counts[split] / total for split in SPLITS},
            "pii_free_document_ratio": pii_free_ratio,
            "hard_negative_document_ratio": hard_negative_ratio,
            "ordinary_negative_document_ratio": ordinary_negative_ratio,
            "pii_free_document_ratio_by_split": pii_free_by_split,
            "hard_negative_document_ratio_by_split": hard_negative_by_split,
        },
        "label_distribution": label_distribution,
        "artifacts": artifacts,
        "admission": {
            "raw_source_status": "raw_source_blocked",
            "derived_source_status": "approved",
            "approved_derived_document_count": total,
        },
        "privacy": {
            "contains_real_personal_data": False,
            "all_replacements_reviewed": True,
            "residual_scan_version": _RESIDUAL_SCAN_VERSION,
            "exact_placeholder_dedup_applied": True,
            "hidden_gold_isolated": True,
            "manifest_contains_raw_text": False,
            "manifest_contains_record_identifiers": False,
            "manifest_contains_entity_values": False,
            "manifest_contains_paths": False,
        },
    }
    if mode == RELEASE_ASSEMBLY_MODE:
        collision_counts = release_scale_zero_collision_counts()
        validate_release_scale_aggregates(
            assembly_mode=mode,
            document_count=total,
            split_document_counts=split_counts,
            pii_free_document_count=pii_free_count,
            hard_negative_document_count=hard_negative_count,
            cross_split_group_collision_counts=collision_counts,
        )
        contract = _load_frozen_release_contract()
        manifest["assembly_mode"] = RELEASE_ASSEMBLY_MODE
        manifest["release_scale"] = {
            "fixture_only": False,
            "contract": {
                "path": _FROZEN_RELEASE_CONTRACT_PATH,
                "file_sha256": _FROZEN_RELEASE_CONTRACT_FILE_SHA256,
                "contract_id": contract.contract_id,
                "contract_version": contract.contract_version,
                "dataset_id": contract.dataset_id,
            },
            "composition_counts": {
                "pii_free_document_count": pii_free_count,
                "hard_negative_document_count": hard_negative_count,
                "ordinary_negative_document_count": pii_free_count - hard_negative_count,
            },
            "group_isolation": {
                "scope": "real_assembly_runtime_verification",
                "verified_during_assembly": True,
                "cross_split_group_collision_counts": collision_counts,
            },
            "real_materialization_evidence": None,
            "activation": {
                "training_allowed": False,
                "release_allowed": False,
                "claims_allowed": False,
                "production_activation_allowed": False,
            },
            "claims": {
                "real_materialization_verified": False,
                "d1_gap_closed": False,
                "training_ready": False,
                "release_ready": False,
                "production_ready": False,
            },
        }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    return manifest


def _output_review_seal(
    records_by_split: Mapping[str, tuple[DocumentRecord, ...]],
    manifest: Mapping[str, Any],
    review_attestation_key: bytes,
) -> str:
    key = _validate_review_attestation_key(review_attestation_key)
    digest = hmac.new(key, _OUTPUT_SEAL_PROTOCOL, hashlib.sha256)
    for split in SPLITS:
        split_bytes = split.encode("ascii")
        payload = _jsonl_bytes(records_by_split[split])
        digest.update(len(split_bytes).to_bytes(8, "big"))
        digest.update(split_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    manifest_bytes = _canonical_json_bytes(dict(manifest))
    digest.update(len(manifest_bytes).to_bytes(8, "big"))
    digest.update(manifest_bytes)
    return digest.hexdigest()


def assemble_dataset(
    candidates: Iterable[AssemblyCandidate],
    *,
    registry: CandidateSourceRegistry,
    surrogate_key: bytes,
    review_attestation_key: bytes,
    seed: int = _FROZEN_SPLIT_SEED,
    split_policy: SplitPolicy | None = None,
    negative_ratio_policy: NegativeRatioPolicy | None = None,
    assembly_mode: str = DEVELOPMENT_ASSEMBLY_MODE,
) -> AssemblyResult:
    """Validate, split, replace, and assemble v2 records entirely in memory."""

    mode = _normalize_assembly_mode(assembly_mode)
    ensure_registry_has_approved_sources(registry)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AssemblyError("assembly seed must be an integer")
    if seed != _FROZEN_SPLIT_SEED:
        raise AssemblyError("assembly seed must match the frozen v2 split contract")
    if not isinstance(surrogate_key, bytes):
        raise AssemblyError("surrogate key must be bytes")
    review_key = _validate_review_attestation_key(review_attestation_key)
    if isinstance(surrogate_key, bytes) and hmac.compare_digest(surrogate_key, review_key):
        raise AssemblyError("review and surrogate keys must be independently scoped")
    try:
        deriver = SurrogateKeyDeriver(surrogate_key)
    except Exception:
        raise AssemblyError("surrogate key does not satisfy the v2 policy") from None
    try:
        normalized = tuple(candidates)
    except Exception:
        raise AssemblyError("candidate collection could not be materialized") from None
    if not normalized or any(not isinstance(item, AssemblyCandidate) for item in normalized):
        raise AssemblyError("candidate collection is empty or invalid")
    if mode == RELEASE_ASSEMBLY_MODE:
        contract = _load_frozen_release_contract()
        if len(normalized) != contract.release_total_documents:
            raise AssemblyError(
                "release assembly requires exactly 20,000 independently admitted documents"
            )
    seen_documents: set[tuple[str, str]] = set()
    for candidate in normalized:
        _validate_candidate(candidate)
        if not hmac.compare_digest(
            candidate.privacy.review_evidence_sha256,
            _review_evidence_sha256(candidate, review_key),
        ):
            raise AssemblyError("review evidence is not bound to the complete replacement plan")
        _validate_source_admission(candidate, registry)
        identity = (candidate.registry_source_id, candidate.record.doc_id)
        if identity in seen_documents:
            raise AssemblyError("candidate collection contains a duplicate document")
        seen_documents.add(identity)

    ratio_policy = (
        _release_negative_ratio_policy(negative_ratio_policy)
        if mode == RELEASE_ASSEMBLY_MODE
        else negative_ratio_policy or NegativeRatioPolicy()
    )
    _validate_negative_ratios(normalized, ratio_policy)
    policy = split_policy or SplitPolicy()
    assignments, component_count = _assign_splits(normalized, split_policy=policy, seed=seed)
    _validate_split_negative_ratios(normalized, assignments, ratio_policy)
    if mode == RELEASE_ASSEMBLY_MODE:
        assigned_counts = Counter(assignments.values())
        validate_release_scale_aggregates(
            assembly_mode=mode,
            document_count=len(normalized),
            split_document_counts={split: assigned_counts[split] for split in SPLITS},
            pii_free_document_count=sum(item.annotation.pii_free for item in normalized),
            hard_negative_document_count=sum(
                bool(item.annotation.hard_negative_categories) for item in normalized
            ),
            cross_split_group_collision_counts=release_scale_zero_collision_counts(),
        )
    replacement_owners: dict[str, tuple[str, str, str, str]] = {}
    relation_replacements: dict[tuple[str, str, str, str], str] = {}
    assembled: dict[str, list[DocumentRecord]] = {split: [] for split in SPLITS}
    for index, candidate in enumerate(normalized):
        split = assignments[index]
        assembled[split].append(
            _transform_candidate(
                candidate,
                split=split,
                key=surrogate_key,
                deriver=deriver,
                replacement_owners=replacement_owners,
                relation_replacements=relation_replacements,
                registry_version=registry.registry_version,
            )
        )
    records_by_split = {
        split: tuple(sorted(records, key=lambda record: record.doc_id))
        for split, records in assembled.items()
    }
    manifest = _build_manifest(
        records_by_split=records_by_split,
        split_policy=policy,
        assembly_mode=mode,
    )
    if manifest["counts"]["connected_component_count"] != component_count:
        raise AssemblyError("public component reconstruction does not match private grouping")
    return AssemblyResult(
        records_by_split=MappingProxyType(records_by_split),
        manifest=MappingProxyType(manifest),
        split_policy=policy,
        review_seal=_output_review_seal(records_by_split, manifest, review_key),
        assembly_mode=mode,
    )


def _assert_new_output_root(output_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise AssemblyError("output root must be a new non-symlink path")
    parent = output_root.parent
    try:
        parent_stat = parent.stat()
    except OSError:
        raise AssemblyError("output parent directory is unavailable") from None
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise AssemblyError("output parent is not a directory")
    absolute_parent = parent.absolute()
    current = Path(absolute_parent.anchor)
    for part in absolute_parent.parts[1:]:
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise AssemblyError("output path contains a symlink component")
        except AssemblyError:
            raise
        except OSError:
            raise AssemblyError("output path validation failed") from None


def _write_new_file(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_dataset(
    result: AssemblyResult,
    output_root: str | Path,
    *,
    review_attestation_key: bytes,
) -> None:
    """Atomically publish one root with read-only public and mode-0600 hidden data."""

    if not isinstance(result, AssemblyResult):
        raise AssemblyError("assembly result is invalid")
    destination = Path(output_root)
    _assert_new_output_root(destination)
    payloads = {split: _jsonl_bytes(result.records_by_split[split]) for split in SPLITS}
    try:
        expected_manifest = _build_manifest(
            records_by_split=result.records_by_split,
            split_policy=result.split_policy,
            assembly_mode=result.assembly_mode,
        )
        if not hmac.compare_digest(
            _canonical_json_bytes(dict(result.manifest)),
            _canonical_json_bytes(expected_manifest),
        ):
            raise AssemblyError("assembly manifest does not match reconstructed aggregates")
        expected_seal = _output_review_seal(
            result.records_by_split,
            result.manifest,
            review_attestation_key,
        )
        if not isinstance(result.review_seal, str) or not hmac.compare_digest(
            result.review_seal, expected_seal
        ):
            raise AssemblyError("assembly output does not match the reviewed release seal")
    except AssemblyError:
        raise
    except Exception:
        raise AssemblyError("assembly result verification failed") from None

    staging_path: Path | None = None
    try:
        staging_path = Path(tempfile.mkdtemp(prefix=".pii-zh-v2-assembly-", dir=destination.parent))
        os.chmod(staging_path, 0o700)
        public_directory = staging_path / "public"
        hidden_directory = staging_path / "hidden"
        public_directory.mkdir(mode=0o700)
        hidden_directory.mkdir(mode=0o700)
        for split in PUBLIC_SPLITS:
            _write_new_file(public_directory / f"{split}.jsonl", payloads[split], mode=0o444)
        _write_new_file(
            public_directory / "manifest.json",
            json.dumps(dict(result.manifest), ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n",
            mode=0o444,
        )
        _write_new_file(
            hidden_directory / "hidden_leaderboard.jsonl",
            payloads["hidden_leaderboard"],
            mode=0o600,
        )
        os.chmod(public_directory, 0o755)
        os.chmod(hidden_directory, 0o700)
        _fsync_directory(public_directory)
        _fsync_directory(hidden_directory)
        _fsync_directory(staging_path)
        if destination.exists() or destination.is_symlink():
            raise AssemblyError("output root changed during atomic materialization")
        os.rename(staging_path, destination)
        staging_path = None
        _fsync_directory(destination.parent)
    except AssemblyError:
        raise
    except OSError:
        raise AssemblyError("atomic dataset materialization failed") from None
    finally:
        if staging_path is not None:
            shutil.rmtree(staging_path, ignore_errors=True)


__all__ = [
    "AnnotationProof",
    "AssemblyCandidate",
    "AssemblyError",
    "AssemblyResult",
    "DEVELOPMENT_ASSEMBLY_MODE",
    "InternalGroups",
    "MANIFEST_SCHEMA_VERSION",
    "NegativeRatioPolicy",
    "PRIVATE_INPUT_SCHEMA_VERSION",
    "PUBLIC_SPLITS",
    "RELEASE_ASSEMBLY_MODE",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "PlannedSurrogateEdit",
    "PrivacyProof",
    "SPLITS",
    "SplitPolicy",
    "assemble_dataset",
    "candidate_from_dict",
    "compute_review_evidence_sha256",
    "ensure_registry_has_approved_sources",
    "load_private_candidates",
    "materialize_dataset",
    "release_scale_zero_collision_counts",
    "validate_release_scale_aggregates",
]
