"""Load and enforce the immutable synthetic template-family split allocation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from .templates import ALL_TEMPLATES, CORE_LABELS, SyntheticTemplate

FAMILY_ALLOCATION_FILENAME = "stable_family_allocation_v3.json"
PRIOR_FAMILY_ALLOCATION_FILENAME = "stable_family_allocation_v2.json"
_SPLITS = frozenset({"train", "validation", "test"})
TRAIN_ONLY_ADDITION_ROLE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "VEHICLE_LICENSE_PLATE": 3,
        "EMPLOYEE_ID": 2,
        "DATE_OF_BIRTH": 2,
        "DRIVER_LICENSE_NUMBER": 1,
        "GEO_COORDINATE": 1,
        "MAC_ADDRESS": 1,
        "PASSPORT_NUMBER": 1,
    }
)


class FamilyAllocationAssetError(ValueError):
    """Raised when the packaged family allocation is malformed or stale."""


@dataclass(frozen=True, slots=True)
class FamilyAllocationEntry:
    template_id: str
    template_group: str
    split: str
    cohort: str
    coverage_role: str | None = None


def _required_text(item: Mapping[str, Any], key: str, *, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FamilyAllocationAssetError(f"{context}.{key} must be a non-empty string")
    return value


def _parse_entries(
    value: Any,
    *,
    cohort: str,
) -> tuple[FamilyAllocationEntry, ...]:
    if not isinstance(value, list):
        raise FamilyAllocationAssetError(f"{cohort} must be an array")
    entries: list[FamilyAllocationEntry] = []
    expected_fields = (
        {"template_id", "template_group", "split"}
        if cohort == "legacy"
        else {"template_id", "template_group", "split", "coverage_role"}
    )
    for index, item in enumerate(value):
        context = f"{cohort}[{index}]"
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise FamilyAllocationAssetError(
                f"{context} fields must be exactly {sorted(expected_fields)}"
            )
        template_id = _required_text(item, "template_id", context=context)
        template_group = _required_text(item, "template_group", context=context)
        if not template_group.startswith("synthetic-family:"):
            raise FamilyAllocationAssetError(
                f"{context}.template_group must use the synthetic-family namespace"
            )
        split = _required_text(item, "split", context=context)
        if split not in _SPLITS:
            raise FamilyAllocationAssetError(f"{context}.split is unsupported")
        coverage_role = (
            _required_text(item, "coverage_role", context=context)
            if cohort == "train_only_addition"
            else None
        )
        entries.append(
            FamilyAllocationEntry(
                template_id=template_id,
                template_group=template_group,
                split=split,
                cohort=cohort,
                coverage_role=coverage_role,
            )
        )
    return tuple(entries)


_RAW = (
    files("pii_zh.data.synthetic")
    .joinpath("assets")
    .joinpath(FAMILY_ALLOCATION_FILENAME)
    .read_bytes()
)
try:
    _ASSET = json.loads(_RAW)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise FamilyAllocationAssetError("invalid packaged family allocation JSON") from exc
if not isinstance(_ASSET, dict):
    raise FamilyAllocationAssetError("family allocation asset must contain an object")
_EXPECTED_TOP_LEVEL = {
    "schema_version",
    "asset_id",
    "asset_version",
    "algorithm_version",
    "source_snapshot",
    "policy",
    "legacy_assignments",
    "train_only_additions",
}
if set(_ASSET) != _EXPECTED_TOP_LEVEL:
    raise FamilyAllocationAssetError("family allocation top-level fields differ from schema")
if _ASSET.get("schema_version") != 2:
    raise FamilyAllocationAssetError("unsupported family allocation schema_version")

FAMILY_ALLOCATION_ASSET_ID = _required_text(_ASSET, "asset_id", context="asset")
FAMILY_ALLOCATION_ASSET_VERSION = _required_text(_ASSET, "asset_version", context="asset")
FAMILY_ALLOCATION_ALGORITHM_VERSION = _required_text(_ASSET, "algorithm_version", context="asset")
FAMILY_ALLOCATION_ASSET_SHA256 = hashlib.sha256(_RAW).hexdigest()
FAMILY_ALLOCATION_METADATA: Mapping[str, Any] = MappingProxyType(dict(_ASSET))

_source_snapshot = _ASSET["source_snapshot"]
if not isinstance(_source_snapshot, dict) or set(_source_snapshot) != {
    "dataset_id",
    "snapshot_role",
    "source_asset_filename",
    "source_asset_sha256",
    "fields_read",
    "raw_text_or_entity_values_read",
    "legacy_template_count",
}:
    raise FamilyAllocationAssetError("source_snapshot fields differ from schema")
if (
    _source_snapshot.get("fields_read")
    != [
        "provenance.template_id",
        "template_group",
        "split",
    ]
    or _source_snapshot.get("raw_text_or_entity_values_read") is not False
):
    raise FamilyAllocationAssetError("source_snapshot must preserve the metadata-only extraction")
if _source_snapshot.get("source_asset_filename") != PRIOR_FAMILY_ALLOCATION_FILENAME:
    raise FamilyAllocationAssetError("source_snapshot must name the immediately prior allocation")

_PRIOR_RAW = (
    files("pii_zh.data.synthetic")
    .joinpath("assets")
    .joinpath(PRIOR_FAMILY_ALLOCATION_FILENAME)
    .read_bytes()
)
PRIOR_FAMILY_ALLOCATION_SHA256 = hashlib.sha256(_PRIOR_RAW).hexdigest()
if _source_snapshot.get("source_asset_sha256") != PRIOR_FAMILY_ALLOCATION_SHA256:
    raise FamilyAllocationAssetError("source_snapshot prior allocation hash does not match")
try:
    _PRIOR_ASSET = json.loads(_PRIOR_RAW)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise FamilyAllocationAssetError("invalid prior family allocation JSON") from exc
if not isinstance(_PRIOR_ASSET, dict):
    raise FamilyAllocationAssetError("prior family allocation must contain an object")

_policy = _ASSET["policy"]
if _policy != {
    "legacy_assignments": "exactly_pinned",
    "new_assignments": "explicit_train_only",
    "unregistered_templates": "reject",
}:
    raise FamilyAllocationAssetError("family allocation policy must remain fail-closed")

LEGACY_FAMILY_ALLOCATIONS = _parse_entries(
    _ASSET["legacy_assignments"],
    cohort="legacy",
)
TRAIN_ONLY_FAMILY_ADDITIONS = _parse_entries(
    _ASSET["train_only_additions"],
    cohort="train_only_addition",
)
if len(LEGACY_FAMILY_ALLOCATIONS) != 96 or _source_snapshot.get("legacy_template_count") != 96:
    raise FamilyAllocationAssetError("family allocation must pin exactly 96 pre-v1.3 templates")
if Counter(entry.split for entry in LEGACY_FAMILY_ALLOCATIONS) != Counter(
    {"train": 65, "validation": 15, "test": 16}
):
    raise FamilyAllocationAssetError("legacy family split counts differ from the frozen snapshot")
if len(TRAIN_ONLY_FAMILY_ADDITIONS) != 11 or any(
    entry.split != "train" for entry in TRAIN_ONLY_FAMILY_ADDITIONS
):
    raise FamilyAllocationAssetError("all eleven v1.3 additions must remain train-only")
if Counter(entry.coverage_role for entry in TRAIN_ONLY_FAMILY_ADDITIONS) != Counter(
    TRAIN_ONLY_ADDITION_ROLE_COUNTS
):
    raise FamilyAllocationAssetError("v1.3 train-only coverage-role counts changed")

_prior_legacy = _PRIOR_ASSET.get("legacy_assignments")
_prior_additions = _PRIOR_ASSET.get("train_only_additions")
if not isinstance(_prior_legacy, list) or not isinstance(_prior_additions, list):
    raise FamilyAllocationAssetError("prior allocation arrays are missing")
_prior_items = _prior_legacy + _prior_additions
if not all(isinstance(item, dict) for item in _prior_items):
    raise FamilyAllocationAssetError("prior allocation entries must be objects")
_prior_assignments = {
    (item.get("template_id"), item.get("template_group"), item.get("split"))
    for item in _prior_items
}
_legacy_assignments = {
    (entry.template_id, entry.template_group, entry.split) for entry in LEGACY_FAMILY_ALLOCATIONS
}
if len(_prior_items) != 96 or len(_prior_assignments) != 96:
    raise FamilyAllocationAssetError("prior allocation must contain exactly 96 unique templates")
if _legacy_assignments != _prior_assignments:
    raise FamilyAllocationAssetError("v1.3 changed a pre-existing template split or family")

ALL_FAMILY_ALLOCATIONS = LEGACY_FAMILY_ALLOCATIONS + TRAIN_ONLY_FAMILY_ADDITIONS
_allocation_ids = [entry.template_id for entry in ALL_FAMILY_ALLOCATIONS]
_allocation_groups = [entry.template_group for entry in ALL_FAMILY_ALLOCATIONS]
if len(_allocation_ids) != len(set(_allocation_ids)):
    raise FamilyAllocationAssetError("family allocation template ids must be unique")
if len(_allocation_groups) != len(set(_allocation_groups)):
    raise FamilyAllocationAssetError("family allocation template groups must be unique")

LEGACY_TEMPLATE_SPLITS: Mapping[str, str] = MappingProxyType(
    {entry.template_id: entry.split for entry in LEGACY_FAMILY_ALLOCATIONS}
)
TRAIN_ONLY_ADDITION_TEMPLATE_IDS = frozenset(
    entry.template_id for entry in TRAIN_ONLY_FAMILY_ADDITIONS
)
TEMPLATE_SPLITS: Mapping[str, str] = MappingProxyType(
    {entry.template_id: entry.split for entry in ALL_FAMILY_ALLOCATIONS}
)


def validate_template_family_catalog(templates: Sequence[SyntheticTemplate]) -> None:
    """Reject unregistered templates and template-id-to-family drift."""

    by_id = {template.template_id: template for template in templates}
    expected_ids = set(TEMPLATE_SPLITS)
    observed_ids = set(by_id)
    if observed_ids != expected_ids or len(by_id) != len(templates):
        raise FamilyAllocationAssetError(
            "template catalog differs from the explicit family allocation registry"
        )
    entries = {entry.template_id: entry for entry in ALL_FAMILY_ALLOCATIONS}
    for template_id, template in by_id.items():
        entry = entries[template_id]
        observed_group = f"synthetic-family:{template.family}"
        if observed_group != entry.template_group:
            raise FamilyAllocationAssetError(
                f"template family changed for registered id {template_id!r}"
            )
        labels = {spec.label for spec in template.placeholders if spec.label is not None}
        if entry.cohort == "legacy" and any(
            spec.value_variant != "legacy" for spec in template.placeholders
        ):
            raise FamilyAllocationAssetError(
                f"pre-v1.3 template {template_id!r} must keep legacy value formats"
            )
        if entry.cohort == "train_only_addition" and (
            template.origin_kind != "human_authored" or template.source_candidate_id is not None
        ):
            raise FamilyAllocationAssetError(
                f"train-only addition {template_id!r} must remain human-authored"
            )
        if entry.coverage_role == "hard_negative" and not template.hard_negative:
            raise FamilyAllocationAssetError(
                f"train-only hard-negative role changed for {template_id!r}"
            )
        if entry.coverage_role is not None and entry.coverage_role not in {
            *CORE_LABELS,
            "hard_negative",
        }:
            raise FamilyAllocationAssetError(
                f"unknown train-only coverage role for {template_id!r}"
            )
        if entry.coverage_role in CORE_LABELS and (
            template.hard_negative or entry.coverage_role not in labels
        ):
            raise FamilyAllocationAssetError(
                f"train-only positive coverage role changed for {template_id!r}"
            )


validate_template_family_catalog(ALL_TEMPLATES)


__all__ = [
    "ALL_FAMILY_ALLOCATIONS",
    "FAMILY_ALLOCATION_ALGORITHM_VERSION",
    "FAMILY_ALLOCATION_ASSET_ID",
    "FAMILY_ALLOCATION_ASSET_SHA256",
    "FAMILY_ALLOCATION_ASSET_VERSION",
    "FAMILY_ALLOCATION_FILENAME",
    "FAMILY_ALLOCATION_METADATA",
    "FamilyAllocationAssetError",
    "FamilyAllocationEntry",
    "LEGACY_FAMILY_ALLOCATIONS",
    "LEGACY_TEMPLATE_SPLITS",
    "PRIOR_FAMILY_ALLOCATION_FILENAME",
    "PRIOR_FAMILY_ALLOCATION_SHA256",
    "TEMPLATE_SPLITS",
    "TRAIN_ONLY_ADDITION_TEMPLATE_IDS",
    "TRAIN_ONLY_ADDITION_ROLE_COUNTS",
    "TRAIN_ONLY_FAMILY_ADDITIONS",
    "validate_template_family_catalog",
]
