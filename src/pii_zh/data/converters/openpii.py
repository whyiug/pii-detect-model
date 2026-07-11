"""Offline converter for already-downloaded OpenPII zh-CN rows.

This module intentionally contains no Hub client or network code.  Converted
rows are quarantined by default until the required manual audit and legal gate
are recorded by the caller.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from ..schema import DocumentRecord, EntitySpan, Provenance, QualityMetadata, QualityTier
from ..validators import validate_cn_resident_id, validate_entity_value
from ._common import (
    ConversionError,
    SourceSpan,
    ensure_non_overlapping,
    entity_value_groups,
    get_spans,
    get_text,
    stable_doc_id,
)

OPENPII_SOURCE_ID = "ai4privacy/pii-masking-openpii-1.5m"
OPENPII_LICENSE = "CC-BY-4.0"


_LABEL_MAP = {
    "GIVENNAME": "PERSON_NAME",
    "SURNAME": "PERSON_NAME",
    "FIRSTNAME": "PERSON_NAME",
    "LASTNAME": "PERSON_NAME",
    "NAME": "PERSON_NAME",
    "PERSON": "PERSON_NAME",
    "TELEPHONENUM": "PHONE_NUMBER",
    "PHONE": "PHONE_NUMBER",
    "PHONENUMBER": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "EMAILADDRESS": "EMAIL_ADDRESS",
    "CREDITCARDNUMBER": "BANK_CARD_NUMBER",
    "BANKCARD": "BANK_CARD_NUMBER",
    "IDCARDNUM": "CN_RESIDENT_ID_OR_GENERIC",
    "IDCARD": "CN_RESIDENT_ID_OR_GENERIC",
    "STREET": "ADDRESS_COMPONENT",
    "CITY": "ADDRESS_COMPONENT",
    "BUILDINGNUM": "ADDRESS_COMPONENT",
    "ZIPCODE": "ADDRESS_COMPONENT",
    "ADDRESS": "ADDRESS",
    "DATE": "DATE_CONTEXTUAL",
    "BIRTHDATE": "DATE_OF_BIRTH",
    "DATEOFBIRTH": "DATE_OF_BIRTH",
    "PASSPORT": "PASSPORT_NUMBER",
    "PASSPORTNUM": "PASSPORT_NUMBER",
    "DRIVERLICENSENUM": "DRIVER_LICENSE_NUMBER",
    "SOCIALNUM": "SOCIAL_SECURITY_NUMBER",
    "IP": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "MAC": "MAC_ADDRESS",
    "MACADDRESS": "MAC_ADDRESS",
    "USERNAME": "USERNAME",
}


_RISK = {
    "PERSON_NAME": "T1",
    "PHONE_NUMBER": "T1",
    "EMAIL_ADDRESS": "T1",
    "ADDRESS": "T1",
    "DATE_OF_BIRTH": "T1",
    "CN_RESIDENT_ID": "T0",
    "GENERIC_DATE": "T2",
    "BANK_CARD_NUMBER": "T0",
    "PASSPORT_NUMBER": "T0",
    "DRIVER_LICENSE_NUMBER": "T0",
    "SOCIAL_SECURITY_NUMBER": "T0",
    "IP_ADDRESS": "T2",
    "MAC_ADDRESS": "T2",
    "USERNAME": "T1",
}


@dataclass(frozen=True, slots=True)
class _MappedSpan:
    start: int
    end: int
    label: str
    original_label: str


def _normalize_source_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", label.upper())


def _is_zh_cn_row(row: Mapping[str, Any]) -> bool:
    language = str(row.get("language", row.get("lang", row.get("locale", ""))))
    language = language.replace("_", "-").lower()
    explicit_region = row.get("region", row.get("country"))
    if explicit_region is not None:
        region = str(explicit_region).upper()
    elif language in {"zh-cn", "zh-hans-cn"}:
        region = "CN"
    else:
        # A bare "zh" without a region cannot prove the required zh-CN filter.
        return False
    return language in {"zh", "zh-cn", "zh-hans", "zh-hans-cn"} and region in {
        "CN",
        "CHN",
    }


def _is_birth_context(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 8) : min(len(text), end + 4)]
    return any(marker in context for marker in ("出生", "生日", "生于", "出生日期"))


def _map_span(span: SourceSpan, text: str) -> _MappedSpan | None:
    normalized = _normalize_source_label(span.label)
    mapped = _LABEL_MAP.get(normalized)
    if mapped is None:
        return None
    if mapped == "CN_RESIDENT_ID_OR_GENERIC":
        # The public source's generic IDCARDNUM label is not proof that a
        # value is a mainland resident ID. Invalid/non-CN values are excluded
        # from the core-label training view instead of inventing a label which
        # is absent from the frozen taxonomy.
        if not validate_cn_resident_id(span.text).valid:
            return None
        mapped = "CN_RESIDENT_ID"
    elif mapped == "DATE_CONTEXTUAL":
        mapped = (
            "DATE_OF_BIRTH" if _is_birth_context(text, span.start, span.end) else "GENERIC_DATE"
        )
    elif mapped == "ADDRESS_COMPONENT":
        mapped = "ADDRESS"
    return _MappedSpan(span.start, span.end, mapped, normalized)


def _merge_components(spans: list[_MappedSpan], text: str) -> list[_MappedSpan]:
    if not spans:
        return []
    merged: list[_MappedSpan] = []
    index = 0
    name_parts = {"GIVENNAME", "SURNAME", "FIRSTNAME", "LASTNAME"}
    address_parts = {"STREET", "CITY", "BUILDINGNUM", "ZIPCODE"}
    while index < len(spans):
        current = spans[index]
        end = current.end
        originals = {current.original_label}
        next_index = index + 1
        while next_index < len(spans):
            candidate = spans[next_index]
            gap = text[end : candidate.start]
            compatible_names = (
                current.label == candidate.label == "PERSON_NAME"
                and originals | {candidate.original_label} <= name_parts
            )
            compatible_address = (
                current.label == candidate.label == "ADDRESS"
                and originals | {candidate.original_label} <= address_parts
            )
            if candidate.start < end or not (compatible_names or compatible_address):
                break
            if len(gap) > 2 or any(char not in " \t,，-" for char in gap):
                break
            end = candidate.end
            originals.add(candidate.original_label)
            next_index += 1
        merged.append(
            _MappedSpan(
                current.start,
                end,
                current.label,
                "+".join(sorted(originals)),
            )
        )
        index = next_index
    return merged


def convert_openpii_record(
    row: Mapping[str, Any],
    *,
    source_revision: str,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> DocumentRecord:
    if not _is_zh_cn_row(row):
        raise ConversionError("OpenPII converter accepts only zh-CN rows")
    text = get_text(row)
    source_spans = get_spans(row, text)
    mapped = [item for span in source_spans if (item := _map_span(span, text)) is not None]
    mapped_source_count = len(mapped)
    mapped.sort(key=lambda item: (item.start, item.end, item.label))
    mapped = _merge_components(mapped, text)
    entities = []
    validators_passed = True
    for span in mapped:
        quote = text[span.start : span.end]
        validation = validate_entity_value(span.label, quote)
        state = (
            "not_applicable" if validation is None else ("passed" if validation.valid else "failed")
        )
        validators_passed = validators_passed and (validation is None or validation.valid)
        entities.append(
            EntitySpan(
                start=span.start,
                end=span.end,
                label=span.label,
                text=quote,
                source="openpii",
                risk_tier=_RISK.get(span.label, "T2"),
                validator_state=state,
                metadata={"original_label": span.original_label},
            )
        )
    canonical_entities = ensure_non_overlapping(entities)
    effective_gate = quality_gate and validators_passed
    provenance = Provenance(
        source_id=OPENPII_SOURCE_ID,
        source_kind="public_dataset",
        source_revision=source_revision,
        license=OPENPII_LICENSE,
        synthetic=True,
        generator_name="ai4privacy_openpii",
        generator_version=source_revision,
        template_id=str(row.get("template_id")) if row.get("template_id") is not None else None,
        metadata={"converted_offline": True, "requires_2000_sample_audit": True},
    )
    quality = QualityMetadata(
        tier=QualityTier.SINGLE_TEACHER,
        quality_gate=effective_gate,
        validators_passed=validators_passed,
        review_status="manual_audit_passed" if effective_gate else "pending_manual_audit",
        confidence=None,
        metadata={
            "source_span_count": len(source_spans),
            "mapped_span_count": len(canonical_entities),
            "dropped_unknown_labels": len(source_spans) - mapped_source_count,
            "merged_component_count": mapped_source_count - len(mapped),
        },
    )
    return DocumentRecord.create(
        doc_id=stable_doc_id(OPENPII_SOURCE_ID, row, text),
        text=text,
        entities=canonical_entities,
        language="zh-Hans-CN",
        domain=str(row.get("domain", "openpii_general")),
        scene=str(row.get("scene", "public_warmup")),
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=public_weight_training_allowed,
        template_group=(
            f"openpii:{row['template_id']}" if row.get("template_id") is not None else None
        ),
        entity_value_groups=entity_value_groups(canonical_entities),
        generator_version=source_revision,
        metadata={"training_role": "warmup_only", "source_dataset": OPENPII_SOURCE_ID},
    )


def iter_openpii_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_revision: str,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> Iterator[DocumentRecord]:
    for row in rows:
        if not _is_zh_cn_row(row):
            continue
        yield convert_openpii_record(
            row,
            source_revision=source_revision,
            quality_gate=quality_gate,
            public_weight_training_allowed=public_weight_training_allowed,
        )
