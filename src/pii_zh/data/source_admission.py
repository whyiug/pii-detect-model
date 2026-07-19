"""Fail-closed admission for unbound v2 candidate data sources.

This module deliberately does not read or modify the source registry bound to
the synthetic-v1.3 release.  A candidate can enter the public-release pool
only after two independent explicit opt-ins (``admission_status`` and
``public_weight_training_allowed``) plus immutable source, terms, privacy,
quality, license, and legal evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .schema import DataPool

DEFAULT_CANDIDATE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "configs/data/source_registry_v2_candidates.yaml"
)

_SCHEMA_VERSION = 1
_REGISTRY_SCOPE = "v2_candidate_sources_unbound"
_DATAFOUNTAIN_SOURCE_ID = "datafountain_472"
_REQUIRED_SOURCE_IDS = frozenset(
    {
        "crosswoz",
        "oasst1",
        "wikipedia_zh",
        "dureader",
        _DATAFOUNTAIN_SOURCE_ID,
        "jddc",
        "csds",
        "risawoz",
        "datasir",
    }
)
_PUBLIC_WEIGHT_LICENSES = frozenset(
    {
        "APACHE-2.0",
        "BSD-2-CLAUSE",
        "BSD-3-CLAUSE",
        "CC-BY-4.0",
        "CC0-1.0",
        "MIT",
    }
)
_UNKNOWN_LICENSES = frozenset({"NOASSERTION", "UNKNOWN", "UNLICENSED", "UNSPECIFIED"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class SourceAdmissionError(ValueError):
    """Raised when a candidate registry violates the fail-closed schema."""


class ReviewStatus(str, Enum):
    PASS = "pass"
    PENDING = "pending"
    FAIL = "fail"
    UNKNOWN = "unknown"


class AdmissionStatus(str, Enum):
    PUBLIC_RELEASE = "public_release"
    EVALUATION_ONLY = "evaluation_only"
    LEGAL_REVIEW_REQUIRED = "legal_review_required"
    QUARANTINED = "quarantined"
    EXCLUDED_FROM_PUBLIC_WEIGHTS = "excluded_from_public_weights"


@dataclass(frozen=True, slots=True)
class TermsSnapshot:
    status: ReviewStatus
    sha256: str | None


@dataclass(frozen=True, slots=True)
class SourceReviews:
    license: ReviewStatus
    terms: ReviewStatus
    privacy: ReviewStatus
    quality: ReviewStatus
    legal: ReviewStatus

    @property
    def all_passed(self) -> bool:
        return all(
            status is ReviewStatus.PASS
            for status in (self.license, self.terms, self.privacy, self.quality, self.legal)
        )


@dataclass(frozen=True, slots=True)
class CandidateSource:
    source_id: str
    display_name: str
    kind: str
    locator: str
    immutable_revision: str | None
    declared_license: str
    terms_snapshot: TermsSnapshot
    reviews: SourceReviews
    public_weight_training_allowed: bool
    admission_status: AdmissionStatus


@dataclass(frozen=True, slots=True)
class CandidateSourceRegistry:
    schema_version: int
    registry_version: str
    registry_scope: str
    release_bound: bool
    default_admission: AdmissionStatus
    sources: Mapping[str, CandidateSource]

    def source(self, source_id: str) -> CandidateSource:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise SourceAdmissionError(
                "source is not registered in the v2 candidate registry"
            ) from exc


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    source_id: str
    status: AdmissionStatus
    pool: DataPool
    public_weight_training_allowed: bool
    reasons: tuple[str, ...]

    @property
    def public_release_allowed(self) -> bool:
        return (
            self.status is AdmissionStatus.PUBLIC_RELEASE
            and self.pool is DataPool.PUBLIC_RELEASE
            and self.public_weight_training_allowed
        )

    def to_dict(self) -> dict[str, str | bool | list[str]]:
        return {
            "source_id": self.source_id,
            "status": self.status.value,
            "pool": self.pool.value,
            "public_weight_training_allowed": self.public_weight_training_allowed,
            "public_release_allowed": self.public_release_allowed,
            "reasons": list(self.reasons),
        }


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SourceAdmissionError(f"{field} must be a string-keyed mapping")
    return value


def _exact_fields(value: Mapping[str, Any], *, expected: frozenset[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise SourceAdmissionError(
            f"{field} fields differ from the schema; missing={missing}, unknown={unknown}"
        )


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SourceAdmissionError(f"{field} must be a non-empty trimmed string")
    return value


def _review_status(value: object, *, field: str) -> ReviewStatus:
    try:
        return ReviewStatus(value)
    except (TypeError, ValueError) as exc:
        raise SourceAdmissionError(f"{field} has an unsupported review status") from exc


def _admission_status(value: object, *, field: str) -> AdmissionStatus:
    try:
        return AdmissionStatus(value)
    except (TypeError, ValueError) as exc:
        raise SourceAdmissionError(f"{field} has an unsupported admission status") from exc


def _parse_terms(value: object, *, field: str) -> TermsSnapshot:
    raw = _mapping(value, field=field)
    _exact_fields(raw, expected=frozenset({"status", "sha256"}), field=field)
    digest = raw["sha256"]
    if digest is not None and (
        not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise SourceAdmissionError(f"{field}.sha256 must be null or a lowercase SHA-256 digest")
    return TermsSnapshot(
        status=_review_status(raw["status"], field=f"{field}.status"),
        sha256=digest,
    )


def _parse_reviews(value: object, *, field: str) -> SourceReviews:
    raw = _mapping(value, field=field)
    names = frozenset({"license", "terms", "privacy", "quality", "legal"})
    _exact_fields(raw, expected=names, field=field)
    return SourceReviews(
        license=_review_status(raw["license"], field=f"{field}.license"),
        terms=_review_status(raw["terms"], field=f"{field}.terms"),
        privacy=_review_status(raw["privacy"], field=f"{field}.privacy"),
        quality=_review_status(raw["quality"], field=f"{field}.quality"),
        legal=_review_status(raw["legal"], field=f"{field}.legal"),
    )


def _parse_source(value: object, *, index: int) -> CandidateSource:
    field = f"sources[{index}]"
    raw = _mapping(value, field=field)
    expected = frozenset(
        {
            "id",
            "display_name",
            "kind",
            "locator",
            "immutable_revision",
            "declared_license",
            "terms_snapshot",
            "reviews",
            "public_weight_training_allowed",
            "admission_status",
        }
    )
    _exact_fields(raw, expected=expected, field=field)
    source_id = _string(raw["id"], field=f"{field}.id")
    if _SOURCE_ID_PATTERN.fullmatch(source_id) is None:
        raise SourceAdmissionError(f"{field}.id must be lowercase snake_case")
    revision = raw["immutable_revision"]
    if revision is not None:
        revision = _string(revision, field=f"{field}.immutable_revision")
        if _REVISION_PATTERN.fullmatch(revision) is None:
            raise SourceAdmissionError(
                f"{field}.immutable_revision must be a lowercase 40-character commit"
            )
    permission = raw["public_weight_training_allowed"]
    if not isinstance(permission, bool):
        raise SourceAdmissionError(f"{field}.public_weight_training_allowed must be a boolean")
    return CandidateSource(
        source_id=source_id,
        display_name=_string(raw["display_name"], field=f"{field}.display_name"),
        kind=_string(raw["kind"], field=f"{field}.kind"),
        locator=_string(raw["locator"], field=f"{field}.locator"),
        immutable_revision=revision,
        declared_license=_string(raw["declared_license"], field=f"{field}.declared_license"),
        terms_snapshot=_parse_terms(raw["terms_snapshot"], field=f"{field}.terms_snapshot"),
        reviews=_parse_reviews(raw["reviews"], field=f"{field}.reviews"),
        public_weight_training_allowed=permission,
        admission_status=_admission_status(
            raw["admission_status"], field=f"{field}.admission_status"
        ),
    )


def _parse_candidate_registry(document: object) -> CandidateSourceRegistry:
    raw = _mapping(document, field="registry")
    expected = frozenset(
        {
            "schema_version",
            "registry_version",
            "registry_scope",
            "release_bound",
            "default_admission",
            "sources",
        }
    )
    _exact_fields(raw, expected=expected, field="registry")
    schema_version = raw["schema_version"]
    if isinstance(schema_version, bool) or schema_version != _SCHEMA_VERSION:
        raise SourceAdmissionError(f"registry.schema_version must equal {_SCHEMA_VERSION}")
    if raw["registry_scope"] != _REGISTRY_SCOPE:
        raise SourceAdmissionError("registry is not the unbound v2 candidate scope")
    if raw["release_bound"] is not False:
        raise SourceAdmissionError("v2 candidate registry must not be release-bound")
    default_admission = _admission_status(
        raw["default_admission"], field="registry.default_admission"
    )
    if default_admission is not AdmissionStatus.QUARANTINED:
        raise SourceAdmissionError("registry.default_admission must be quarantined")
    raw_sources = raw["sources"]
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
        raise SourceAdmissionError("registry.sources must be a sequence")
    sources: dict[str, CandidateSource] = {}
    for index, value in enumerate(raw_sources):
        source = _parse_source(value, index=index)
        if source.source_id in sources:
            raise SourceAdmissionError("registry contains a duplicate source id")
        sources[source.source_id] = source
    missing_required = sorted(_REQUIRED_SOURCE_IDS - set(sources))
    if missing_required:
        raise SourceAdmissionError(
            f"registry is missing required v2 candidates: {missing_required}"
        )
    return CandidateSourceRegistry(
        schema_version=_SCHEMA_VERSION,
        registry_version=_string(raw["registry_version"], field="registry.registry_version"),
        registry_scope=_REGISTRY_SCOPE,
        release_bound=False,
        default_admission=default_admission,
        sources=MappingProxyType(sources),
    )


def load_candidate_registry_bytes(value: bytes) -> CandidateSourceRegistry:
    """Parse one already pinned byte snapshot without reopening a filesystem path."""

    if not isinstance(value, bytes) or not value:
        raise SourceAdmissionError("candidate registry bytes must be non-empty")
    try:
        document = yaml.safe_load(value.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise SourceAdmissionError("unable to parse the v2 candidate registry bytes") from exc
    return _parse_candidate_registry(document)


def load_candidate_registry(
    path: str | Path = DEFAULT_CANDIDATE_REGISTRY_PATH,
) -> CandidateSourceRegistry:
    """Load and strictly validate the release-unbound v2 candidate registry."""

    try:
        value = Path(path).read_bytes()
    except OSError as exc:
        raise SourceAdmissionError("unable to load the v2 candidate registry") from exc
    return load_candidate_registry_bytes(value)


def _license_reasons(declared_license: str) -> tuple[str, ...]:
    normalized = declared_license.upper()
    components = set(re.split(r"[^A-Z0-9]+", normalized))
    reasons: list[str] = []
    if normalized in _UNKNOWN_LICENSES:
        reasons.append("license_unknown")
    if "NC" in components:
        reasons.append("license_noncommercial")
    if "SA" in components:
        reasons.append("license_sharealike")
    if not reasons and normalized not in _PUBLIC_WEIGHT_LICENSES:
        reasons.append("license_not_allowlisted")
    return tuple(reasons)


def evaluate_source_admission(source: CandidateSource) -> AdmissionDecision:
    """Return a fail-closed decision for one parsed candidate source."""

    if source.source_id == _DATAFOUNTAIN_SOURCE_ID:
        return AdmissionDecision(
            source_id=source.source_id,
            status=AdmissionStatus.EVALUATION_ONLY,
            pool=DataPool.EVALUATION_ONLY,
            public_weight_training_allowed=False,
            reasons=("datafountain_472_forced_evaluation_only",),
        )
    if source.admission_status is AdmissionStatus.EVALUATION_ONLY:
        return AdmissionDecision(
            source_id=source.source_id,
            status=AdmissionStatus.EVALUATION_ONLY,
            pool=DataPool.EVALUATION_ONLY,
            public_weight_training_allowed=False,
            reasons=("source_declared_evaluation_only",),
        )

    reasons = list(_license_reasons(source.declared_license))
    if (
        not isinstance(source.immutable_revision, str)
        or _REVISION_PATTERN.fullmatch(source.immutable_revision) is None
    ):
        reasons.append("immutable_revision_missing_or_invalid")
    if source.terms_snapshot.status is not ReviewStatus.PASS:
        reasons.append("terms_snapshot_not_passed")
    if (
        not isinstance(source.terms_snapshot.sha256, str)
        or _SHA256_PATTERN.fullmatch(source.terms_snapshot.sha256) is None
    ):
        reasons.append("terms_snapshot_sha256_missing")
    for name, review_status in (
        ("license", source.reviews.license),
        ("terms", source.reviews.terms),
        ("privacy", source.reviews.privacy),
        ("quality", source.reviews.quality),
        ("legal", source.reviews.legal),
    ):
        if review_status is not ReviewStatus.PASS:
            reasons.append(f"{name}_review_not_passed")
    if source.public_weight_training_allowed is not True:
        reasons.append("public_weight_training_not_explicitly_allowed")
    if source.admission_status is not AdmissionStatus.PUBLIC_RELEASE:
        reasons.append("admission_status_not_public_release")

    if not reasons:
        return AdmissionDecision(
            source_id=source.source_id,
            status=AdmissionStatus.PUBLIC_RELEASE,
            pool=DataPool.PUBLIC_RELEASE,
            public_weight_training_allowed=True,
            reasons=(),
        )

    restricted = {"license_noncommercial", "license_sharealike"} & set(reasons)
    if source.admission_status is AdmissionStatus.EXCLUDED_FROM_PUBLIC_WEIGHTS or restricted:
        decision_status = AdmissionStatus.EXCLUDED_FROM_PUBLIC_WEIGHTS
    elif source.admission_status is AdmissionStatus.LEGAL_REVIEW_REQUIRED:
        decision_status = AdmissionStatus.LEGAL_REVIEW_REQUIRED
    else:
        decision_status = AdmissionStatus.QUARANTINED
    return AdmissionDecision(
        source_id=source.source_id,
        status=decision_status,
        pool=DataPool.QUARANTINED,
        public_weight_training_allowed=False,
        reasons=tuple(reasons),
    )


def evaluate_registered_source(
    registry: CandidateSourceRegistry, source_id: str
) -> AdmissionDecision:
    """Evaluate one source by its stable registry identifier."""

    return evaluate_source_admission(registry.source(source_id))


__all__ = [
    "AdmissionDecision",
    "AdmissionStatus",
    "CandidateSource",
    "CandidateSourceRegistry",
    "DEFAULT_CANDIDATE_REGISTRY_PATH",
    "ReviewStatus",
    "SourceAdmissionError",
    "SourceReviews",
    "TermsSnapshot",
    "evaluate_registered_source",
    "evaluate_source_admission",
    "load_candidate_registry",
    "load_candidate_registry_bytes",
]
