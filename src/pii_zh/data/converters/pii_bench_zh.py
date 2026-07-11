"""Offline, evaluation-only converter for already-downloaded pii-bench-zh rows."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ..schema import DocumentRecord, EntitySpan, Provenance, QualityMetadata, QualityTier
from ..validators import validate_entity_value
from ._common import (
    ConversionError,
    ensure_non_overlapping,
    entity_value_groups,
    get_spans,
    get_text,
    stable_doc_id,
)

PII_BENCH_SOURCE_ID = "wan9yu/pii-bench-zh"
PII_BENCH_LICENSE = "Apache-2.0"


_LABEL_MAP = {
    "PERSON": "PERSON_NAME",
    "PERSONNAME": "PERSON_NAME",
    "NAME": "PERSON_NAME",
    "PHONE": "PHONE_NUMBER",
    "PHONENUMBER": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "EMAILADDRESS": "EMAIL_ADDRESS",
    "ADDRESS": "ADDRESS",
    "IDCARD": "CN_RESIDENT_ID",
    "IDNUMBER": "CN_RESIDENT_ID",
    "CNRESIDENTID": "CN_RESIDENT_ID",
    "BANKCARD": "BANK_CARD_NUMBER",
    "BANKCARDNUMBER": "BANK_CARD_NUMBER",
    "PASSPORT": "PASSPORT_NUMBER",
    "PASSPORTNUMBER": "PASSPORT_NUMBER",
    "LICENSEPLATE": "VEHICLE_LICENSE_PLATE",
    "LICENCEPLATE": "VEHICLE_LICENSE_PLATE",
    "WECHAT": "WECHAT_ID",
    "WECHATID": "WECHAT_ID",
    "QQ": "QQ_NUMBER",
    "QQNUMBER": "QQ_NUMBER",
}


_RISK = {
    "CN_RESIDENT_ID": "T0",
    "BANK_CARD_NUMBER": "T0",
    "PASSPORT_NUMBER": "T0",
    "VEHICLE_LICENSE_PLATE": "T1",
    "PERSON_NAME": "T1",
    "PHONE_NUMBER": "T1",
    "EMAIL_ADDRESS": "T1",
    "ADDRESS": "T1",
    "WECHAT_ID": "T1",
    "QQ_NUMBER": "T1",
}


def _normalize_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def convert_pii_bench_zh_record(
    row: Mapping[str, Any],
    *,
    source_revision: str,
    subset: str | None = None,
) -> DocumentRecord:
    text = get_text(row)
    source_spans = get_spans(row, text)
    entities = []
    validators_passed = True
    for span in source_spans:
        normalized = _normalize_label(span.label)
        try:
            label = _LABEL_MAP[normalized]
        except KeyError as exc:
            raise ConversionError(f"unknown pii-bench-zh label: {span.label!r}") from exc
        validation = validate_entity_value(label, span.text)
        state = (
            "not_applicable" if validation is None else ("passed" if validation.valid else "failed")
        )
        validators_passed = validators_passed and (validation is None or validation.valid)
        entities.append(
            EntitySpan(
                start=span.start,
                end=span.end,
                label=label,
                text=span.text,
                source="pii_bench_zh",
                risk_tier=_RISK[label],
                validator_state=state,
                metadata={"original_label": span.label},
            )
        )
    canonical_entities = ensure_non_overlapping(entities)
    benchmark_subset = str(subset or row.get("subset", row.get("config", "formal"))).lower()
    if benchmark_subset not in {"formal", "chat"}:
        raise ConversionError("pii-bench-zh subset must be 'formal' or 'chat'")
    provenance = Provenance(
        source_id=PII_BENCH_SOURCE_ID,
        source_kind="frozen_external_benchmark",
        source_revision=source_revision,
        license=PII_BENCH_LICENSE,
        synthetic=True,
        evaluation_only=True,
        generator_name="wan9yu_pii_bench_zh",
        generator_version=source_revision,
        metadata={
            "converted_offline": True,
            "frozen": True,
            "forbidden_uses": [
                "training",
                "threshold_selection",
                "distillation",
                "prompt_development",
            ],
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.SYNTHETIC_VALIDATED,
        quality_gate=True,
        validators_passed=validators_passed,
        review_status="frozen_benchmark_import",
        confidence=1.0,
        metadata={"subset": benchmark_subset},
    )
    return DocumentRecord.create(
        doc_id=stable_doc_id(PII_BENCH_SOURCE_ID, row, text),
        text=text,
        entities=canonical_entities,
        language="zh-Hans-CN",
        domain="pii_bench_zh",
        scene=benchmark_subset,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=False,
        entity_value_groups=entity_value_groups(canonical_entities),
        generator_version=source_revision,
        split="test",
        metadata={
            "benchmark_subset": benchmark_subset,
            "evaluation_only": True,
            "never_train": True,
        },
    )


def iter_pii_bench_zh_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_revision: str,
    subset: str | None = None,
) -> Iterator[DocumentRecord]:
    for row in rows:
        yield convert_pii_bench_zh_record(
            row,
            source_revision=source_revision,
            subset=subset,
        )
