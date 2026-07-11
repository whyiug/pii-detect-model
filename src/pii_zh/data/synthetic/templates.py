"""Load and validate the versioned, human-curated synthetic template asset."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

TEMPLATE_ASSET_FILENAME = "curated_templates_v1.json"
CURATION_AUDIT_FILENAME = "curation_audit_v1.json"
_PLACEHOLDER_RE = re.compile(r"<<([A-Z][A-Z0-9_]*_[1-9][0-9]*)>>")
_NUMBERED_LABEL_RE = re.compile(r"([A-Z][A-Z0-9_]*?)_[1-9][0-9]*\Z")
_SYNTHETIC_SHORTCUT_WORDS = ("测试", "示例", "虚构", "演练")

CORE_LABELS: tuple[str, ...] = (
    "PERSON_NAME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "VEHICLE_LICENSE_PLATE",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER",
    "WECHAT_ID",
    "QQ_NUMBER",
    "ALIPAY_ACCOUNT",
    "USERNAME",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "DEVICE_ID",
    "GEO_COORDINATE",
    "SECRET",
)


class TemplateAssetError(ValueError):
    """Raised when the packaged curated template asset is malformed."""


@dataclass(frozen=True, slots=True)
class PlaceholderSpec:
    key: str
    generator_key: str
    label: str | None
    risk_tier: str | None = None


@dataclass(frozen=True, slots=True)
class SyntheticTemplate:
    template_id: str
    family: str
    domain: str
    scene: str
    body: str
    placeholders: tuple[PlaceholderSpec, ...]
    hard_negative: bool = False
    origin_kind: str = "human_authored"
    source_candidate_id: str | None = None


def _asset_bytes(filename: str) -> bytes:
    return files("pii_zh.data.synthetic").joinpath("assets", filename).read_bytes()


def _decode_json(filename: str, raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemplateAssetError(f"invalid packaged JSON asset {filename}") from exc
    if not isinstance(value, dict):
        raise TemplateAssetError(f"{filename} must contain a JSON object")
    return value


def _required_text(item: Mapping[str, Any], key: str, *, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TemplateAssetError(f"{context}.{key} must be a non-empty string")
    return value


def _placeholder_label(key: str, *, context: str) -> str:
    match = _NUMBERED_LABEL_RE.fullmatch(key)
    if match is None:
        raise TemplateAssetError(f"{context} has an invalid numbered placeholder {key!r}")
    return match.group(1)


def _parse_template(item: Any, *, hard_negative: bool) -> SyntheticTemplate:
    if not isinstance(item, dict):
        raise TemplateAssetError("each template entry must be an object")
    context = str(item.get("id", "<missing-template-id>"))
    expected = (
        {"id", "family", "domain", "scene", "text", "non_entities"}
        if hard_negative
        else {"id", "family", "domain", "scene", "text", "entities", "source_candidate_id"}
    )
    if set(item) != expected:
        raise TemplateAssetError(
            f"{context} fields differ from schema: expected={sorted(expected)}, "
            f"observed={sorted(item)}"
        )

    template_id = _required_text(item, "id", context=context)
    if re.fullmatch(r"[a-z][a-z0-9_]{5,100}", template_id) is None:
        raise TemplateAssetError(f"invalid template id {template_id!r}")
    family = _required_text(item, "family", context=context)
    domain = _required_text(item, "domain", context=context)
    scene = _required_text(item, "scene", context=context)
    body = _required_text(item, "text", context=context)
    shortcut = next((word for word in _SYNTHETIC_SHORTCUT_WORDS if word in body), None)
    if shortcut is not None:
        raise TemplateAssetError(
            f"{context} training text contains forbidden synthetic shortcut word {shortcut!r}"
        )

    config_key = "non_entities" if hard_negative else "entities"
    config = item[config_key]
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise TemplateAssetError(f"{context}.{config_key} must be an object with string keys")
    observed = _PLACEHOLDER_RE.findall(body)
    if "<<" in _PLACEHOLDER_RE.sub("", body) or set(observed) != set(config):
        raise TemplateAssetError(
            f"{context} placeholder mismatch: text={sorted(set(observed))}, "
            f"declared={sorted(config)}"
        )
    if len(observed) != len(set(observed)):
        raise TemplateAssetError(f"{context} repeats a placeholder key")
    if not hard_negative and not observed:
        raise TemplateAssetError(f"positive template {context} must contain an entity")

    specs: list[PlaceholderSpec] = []
    for key in observed:
        configured = config[key]
        if hard_negative:
            if not isinstance(configured, str) or not configured:
                raise TemplateAssetError(f"{context}.{key} generator key must be a string")
            specs.append(PlaceholderSpec(key=key, generator_key=configured, label=None))
            continue
        label = _placeholder_label(key, context=context)
        if label not in CORE_LABELS:
            raise TemplateAssetError(f"{context} uses non-core entity label {label}")
        if configured not in {"T0", "T1", "T2"}:
            raise TemplateAssetError(f"{context}.{key} must declare T0, T1, or T2")
        specs.append(
            PlaceholderSpec(
                key=key,
                generator_key=label,
                label=label,
                risk_tier=configured,
            )
        )

    source_candidate_id: str | None = None
    if not hard_negative:
        raw_source_id = item["source_candidate_id"]
        if raw_source_id is not None and (
            not isinstance(raw_source_id, str) or not raw_source_id.strip()
        ):
            raise TemplateAssetError(f"{context}.source_candidate_id must be a string or null")
        source_candidate_id = raw_source_id
    return SyntheticTemplate(
        template_id=template_id,
        family=family,
        domain=domain,
        scene=scene,
        body=body,
        placeholders=tuple(specs),
        hard_negative=hard_negative,
        origin_kind="qwen3_candidate" if source_candidate_id else "human_authored",
        source_candidate_id=source_candidate_id,
    )


def _template_list(asset: Mapping[str, Any], key: str) -> list[Any]:
    value = asset.get(key)
    if not isinstance(value, list):
        raise TemplateAssetError(f"asset.{key} must be an array")
    return value


def core_label_family_coverage(
    templates: tuple[SyntheticTemplate, ...] | None = None,
) -> Mapping[str, frozenset[str]]:
    selected = POSITIVE_TEMPLATES if templates is None else templates
    result = {label: set() for label in CORE_LABELS}
    for template in selected:
        for spec in template.placeholders:
            if spec.label in result:
                result[spec.label].add(template.family)
    return MappingProxyType({label: frozenset(families) for label, families in result.items()})


_ASSET_RAW = _asset_bytes(TEMPLATE_ASSET_FILENAME)
_ASSET = _decode_json(TEMPLATE_ASSET_FILENAME, _ASSET_RAW)
if _ASSET.get("schema_version") != 1:
    raise TemplateAssetError("unsupported curated template schema_version")
if _ASSET.get("license") != "Apache-2.0":
    raise TemplateAssetError("curated template asset must remain Apache-2.0")

CURATED_ASSET_ID = _required_text(_ASSET, "asset_id", context="asset")
CURATED_ASSET_VERSION = _required_text(_ASSET, "asset_version", context="asset")
CURATED_ASSET_SHA256 = hashlib.sha256(_ASSET_RAW).hexdigest()
CURATED_ASSET_METADATA: Mapping[str, Any] = MappingProxyType(dict(_ASSET))

POSITIVE_TEMPLATES: tuple[SyntheticTemplate, ...] = tuple(
    _parse_template(item, hard_negative=False)
    for item in _template_list(_ASSET, "positive_templates")
)
HARD_NEGATIVE_TEMPLATES: tuple[SyntheticTemplate, ...] = tuple(
    _parse_template(item, hard_negative=True)
    for item in _template_list(_ASSET, "hard_negative_templates")
)
ALL_TEMPLATES = POSITIVE_TEMPLATES + HARD_NEGATIVE_TEMPLATES

_curation = _ASSET.get("curation")
if not isinstance(_curation, Mapping):
    raise TemplateAssetError("asset.curation must be an object")
_candidate_template_count = sum(
    template.source_candidate_id is not None for template in POSITIVE_TEMPLATES
)
_human_positive_count = len(POSITIVE_TEMPLATES) - _candidate_template_count
_expected_asset_counts = {
    "accepted_candidate_count": _candidate_template_count,
    "human_authored_positive_count": _human_positive_count,
    "human_authored_hard_negative_count": len(HARD_NEGATIVE_TEMPLATES),
}
for _count_key, _observed_count in _expected_asset_counts.items():
    if _curation.get(_count_key) != _observed_count:
        raise TemplateAssetError(f"asset.curation.{_count_key} does not match packaged templates")

_template_ids = [template.template_id for template in ALL_TEMPLATES]
if len(_template_ids) != len(set(_template_ids)):
    raise TemplateAssetError("curated template ids must be unique")
_normalized_bodies = ["".join(template.body.split()).casefold() for template in ALL_TEMPLATES]
if len(_normalized_bodies) != len(set(_normalized_bodies)):
    raise TemplateAssetError("curated template bodies must not contain exact duplicates")
if len({template.domain for template in ALL_TEMPLATES}) < 12:
    raise TemplateAssetError("curated templates must cover at least twelve domains")
_coverage = core_label_family_coverage(POSITIVE_TEMPLATES)
_undercovered = {label: len(families) for label, families in _coverage.items() if len(families) < 3}
if _undercovered:
    raise TemplateAssetError(
        f"every core label needs three distinct template families: {_undercovered}"
    )

_AUDIT_RAW = _asset_bytes(CURATION_AUDIT_FILENAME)
_AUDIT = _decode_json(CURATION_AUDIT_FILENAME, _AUDIT_RAW)
CURATION_AUDIT_SHA256 = hashlib.sha256(_AUDIT_RAW).hexdigest()
CURATION_AUDIT_METADATA: Mapping[str, Any] = MappingProxyType(dict(_AUDIT))
_accepted = _AUDIT.get("accepted_candidate_ids")
_rejected = _AUDIT.get("rejected_candidates")
if not isinstance(_accepted, list) or not all(isinstance(item, str) for item in _accepted):
    raise TemplateAssetError("curation audit accepted_candidate_ids must be a string array")
if not isinstance(_rejected, list) or not all(isinstance(item, dict) for item in _rejected):
    raise TemplateAssetError("curation audit rejected_candidates must be an object array")
if _AUDIT.get("accepted_candidate_count") != len(_accepted) or _AUDIT.get(
    "rejected_candidate_count"
) != len(_rejected):
    raise TemplateAssetError("curation audit decision counts do not match the decision arrays")
if _AUDIT.get("reviewed_candidate_count") != len(_accepted) + len(_rejected):
    raise TemplateAssetError("curation audit reviewed count does not match all decisions")
_rejected_ids = {item.get("id") for item in _rejected}
if not all(isinstance(item, str) for item in _rejected_ids):
    raise TemplateAssetError("curation audit rejected candidate ids must be strings")
_packaged_candidate_ids = {
    template.source_candidate_id
    for template in POSITIVE_TEMPLATES
    if template.source_candidate_id is not None
}
if set(_accepted) != _packaged_candidate_ids:
    raise TemplateAssetError("curation audit accepted ids differ from packaged candidate ids")
if set(_accepted) & _rejected_ids or len(set(_accepted) | _rejected_ids) != 70:
    raise TemplateAssetError("curation audit must partition all seventy candidate ids")
