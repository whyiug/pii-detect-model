"""Fail-closed offline conversion of OpenAssistant OASST1 messages.

The converter accepts already-loaded upstream rows and performs no downloads.
Each eligible Chinese message becomes one unannotated quarantine candidate.
Raw upstream message, user, parent, and tree identifiers are never copied into
canonical metadata; only opaque hashes are used for document and conversation
grouping.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime
from typing import Any

from ..schema import DocumentRecord, Provenance, QualityMetadata, QualityTier
from ._common import ConversionError

OASST1_SOURCE_ID = "OpenAssistant/oasst1"
OASST1_CONTEXT_ORIGIN = "open_human_dialogue"

OASST1_FIELDS = frozenset(
    {
        "message_id",
        "parent_id",
        "user_id",
        "created_date",
        "text",
        "role",
        "lang",
        "review_count",
        "review_result",
        "deleted",
        "rank",
        "synthetic",
        "model_name",
        "detoxify",
        "message_tree_id",
        "tree_state",
        "emojis",
        "labels",
    }
)
_ROLES = frozenset({"prompter", "assistant"})


def _require_non_empty_string(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"OASST1 {field} failed type validation")
    return value


def _require_nullable_string(field: str, value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise ConversionError(f"OASST1 {field} failed type validation")


def _require_structured(field: str, value: object, *, mapping_only: bool = False) -> None:
    """Validate unused nested fields without reflecting their values in errors."""

    if value is None:
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConversionError(f"OASST1 {field} failed type validation")
        for item in value.values():
            _require_json_value(field, item)
        return
    if not mapping_only and isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ):
        for item in value:
            _require_json_value(field, item)
        return
    raise ConversionError(f"OASST1 {field} failed type validation")


def _require_json_value(field: str, value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversionError(f"OASST1 {field} failed type validation")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConversionError(f"OASST1 {field} failed type validation")
        for item in value.values():
            _require_json_value(field, item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _require_json_value(field, item)
        return
    raise ConversionError(f"OASST1 {field} failed type validation")


def _validate_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ConversionError("OASST1 row failed schema validation")
    if set(row) != OASST1_FIELDS:
        # Field names can originate outside the privacy boundary, so never echo
        # the unknown or missing names in an exception or log.
        raise ConversionError("OASST1 row failed schema validation")

    _require_non_empty_string("message_id", row["message_id"])
    _require_nullable_string("parent_id", row["parent_id"])
    _require_nullable_string("user_id", row["user_id"])
    created_date = row["created_date"]
    if not (
        (isinstance(created_date, str) and bool(created_date.strip()))
        or isinstance(created_date, (date, datetime))
    ):
        raise ConversionError("OASST1 created_date failed type validation")
    _require_non_empty_string("text", row["text"])

    role = _require_non_empty_string("role", row["role"])
    if role not in _ROLES:
        raise ConversionError("OASST1 role failed validation")
    _require_non_empty_string("lang", row["lang"])

    review_count = row["review_count"]
    if isinstance(review_count, bool) or not isinstance(review_count, int) or review_count < 0:
        raise ConversionError("OASST1 review_count failed type validation")
    if row["review_result"] is not None and not isinstance(row["review_result"], bool):
        raise ConversionError("OASST1 review_result failed type validation")
    if not isinstance(row["deleted"], bool):
        raise ConversionError("OASST1 deleted failed type validation")

    rank = row["rank"]
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, (int, float)):
            raise ConversionError("OASST1 rank failed type validation")
        if isinstance(rank, float) and not math.isfinite(rank):
            raise ConversionError("OASST1 rank failed type validation")
    if not isinstance(row["synthetic"], bool):
        raise ConversionError("OASST1 synthetic failed type validation")
    _require_nullable_string("model_name", row["model_name"])
    _require_structured("detoxify", row["detoxify"], mapping_only=True)
    _require_non_empty_string("message_tree_id", row["message_tree_id"])
    _require_non_empty_string("tree_state", row["tree_state"])
    _require_structured("emojis", row["emojis"])
    _require_structured("labels", row["labels"])
    return row


def oasst1_exclusion_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic default exclusion reasons after strict validation."""

    validated = _validate_row(row)
    reasons: list[str] = []
    if validated["lang"] != "zh":
        reasons.append("non_zh")
    if validated["deleted"] is True:
        reasons.append("deleted")
    if validated["synthetic"] is True:
        reasons.append("synthetic")
    return tuple(reasons)


def _opaque_identity(
    *, source_revision: str, message_id: str, message_tree_id: str
) -> tuple[str, str]:
    message_material = (
        f"{OASST1_SOURCE_ID}\0{source_revision}\0message\0{message_id}".encode()
    )
    tree_material = (
        f"{OASST1_SOURCE_ID}\0{source_revision}\0tree\0{message_tree_id}".encode()
    )
    doc_id = "oasst1-message:sha256:" + hashlib.sha256(message_material).hexdigest()
    conversation_group = "oasst1-tree:sha256:" + hashlib.sha256(tree_material).hexdigest()
    return doc_id, conversation_group


def convert_oasst1_record(
    row: Mapping[str, Any],
    *,
    source_revision: str,
    source_license: str,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> DocumentRecord:
    """Convert one eligible Chinese message into an unannotated quarantine record."""

    source_revision = _require_non_empty_string("source_revision", source_revision)
    source_license = _require_non_empty_string("source_license", source_license)
    if quality_gate is not False or public_weight_training_allowed is not None:
        raise ConversionError("OASST1 conversion emits quarantine candidates only")

    reasons = oasst1_exclusion_reasons(row)
    if reasons:
        raise ConversionError("OASST1 row is excluded by the quarantine import policy")

    message_id = row["message_id"]
    message_tree_id = row["message_tree_id"]
    role = row["role"]
    text = row["text"]
    assert isinstance(message_id, str)
    assert isinstance(message_tree_id, str)
    assert isinstance(role, str)
    assert isinstance(text, str)
    doc_id, conversation_group = _opaque_identity(
        source_revision=source_revision,
        message_id=message_id,
        message_tree_id=message_tree_id,
    )

    provenance = Provenance(
        source_id=OASST1_SOURCE_ID,
        source_kind="public_human_authored_dialogue",
        source_revision=source_revision,
        license=source_license,
        synthetic=False,
        metadata={
            "converted_offline": True,
            "source_schema": "oasst1-message-v1",
            "context_origin": OASST1_CONTEXT_ORIGIN,
            "contains_real_personal_data": None,
            "privacy_review_status": "pending",
            "source_message_identifiers_stored": False,
            "source_parent_identifiers_stored": False,
            "source_user_identifiers_stored": False,
            "source_tree_identifiers_stored": False,
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.UNCERTAIN,
        quality_gate=False,
        validators_passed=False,
        review_status="privacy_license_and_human_annotation_review_pending",
        confidence=None,
        metadata={
            "structural_schema_validated": True,
            "human_annotation_complete": False,
            "privacy_review_complete": False,
        },
    )
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(),
        language="zh-Hans-CN",
        domain="general_open_dialogue",
        scene=f"oasst1_{role}_message",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=None,
        conversation_group=conversation_group,
        entity_value_groups=(),
        metadata={
            "context_origin": OASST1_CONTEXT_ORIGIN,
            "contains_real_personal_data": None,
            "privacy_review_status": "pending",
            "annotation_status": "pending",
            "source_role": role,
            "source_review_count": row["review_count"],
            "source_review_result": row["review_result"],
            "source_message_identifiers_stored": False,
            "source_parent_identifiers_stored": False,
            "source_user_identifiers_stored": False,
            "source_tree_identifiers_stored": False,
            "source_group": conversation_group,
            "training_role": "annotation_candidate_only",
        },
    )


def iter_oasst1_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_revision: str,
    source_license: str,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> Iterator[DocumentRecord]:
    """Yield eligible zh messages while failing closed on schema or role drift."""

    _require_non_empty_string("source_revision", source_revision)
    _require_non_empty_string("source_license", source_license)
    if quality_gate is not False or public_weight_training_allowed is not None:
        raise ConversionError("OASST1 conversion emits quarantine candidates only")
    for row in rows:
        if oasst1_exclusion_reasons(row):
            continue
        yield convert_oasst1_record(
            row,
            source_revision=source_revision,
            source_license=source_license,
            quality_gate=quality_gate,
            public_weight_training_allowed=public_weight_training_allowed,
        )


__all__ = [
    "OASST1_CONTEXT_ORIGIN",
    "OASST1_FIELDS",
    "OASST1_SOURCE_ID",
    "convert_oasst1_record",
    "iter_oasst1_records",
    "oasst1_exclusion_reasons",
]
