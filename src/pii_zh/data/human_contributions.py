"""Fail-closed intake validation for human-authored Chinese context contributions.

The validator deliberately stops before surrogate insertion and publication.
Contributors write context around explicit, value-free slots; no submitted text
is allowed to contain an intended PII value.  Independent annotators label the
PII slots, disagreements are adjudicated, and grouping keys are checked before
any downstream split-specific surrogate is generated.

Passing this module is only an intake/quarantine gate.  It never grants source
admission, public-weight eligibility, or publication eligibility.  The report
contains aggregate counts only: source text, identifiers, paths, and hashes are
never returned or reflected in exceptions.
"""

from __future__ import annotations

import json
import os
import re
import stat
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from pii_zh.taxonomy import load_taxonomy

CONTRIBUTION_SCHEMA_VERSION = "pii-zh-human-context-contribution-v1"
AGGREGATE_REPORT_SCHEMA_VERSION = "pii-zh-human-context-contribution-report-v1"
REAL_SOURCE_ID = "community_human_context_contribution_v1"
FIXTURE_SOURCE_ID = "fixture_only_human_context_contribution"
AGREEMENT_VERSION = "pii-zh-human-context-terms-v1"
ANNOTATION_STAGE = "pre_surrogate_slot_intent"
SURROGATE_STATE = "not_inserted_post_group_split_required"
SPLITS = ("train", "validation", "public_test", "hidden_leaderboard")

_MAX_INPUT_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024
_MAX_TEXT_CHARACTERS = 100_000
_MIN_BATCH_RECORDS = 10
_MIN_PII_FREE_RATIO = 0.30
_MAX_PII_FREE_RATIO = 0.40
_MIN_EXACT_IAA = 0.92
_MIN_T0_IAA = 0.95
_MIN_PII_FREE_AGREEMENT = 0.95
_MIN_PER_LABEL_IAA = 0.85

_PUBLIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{7,127}$")
_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SLOT_ID_PATTERN = re.compile(r"^slot_[a-z0-9_]{2,63}$")
_PII_SLOT_PATTERN = re.compile(r"\[\[PII_SLOT:(slot_[a-z0-9_]{2,63})\]\]")
_NEGATIVE_SLOT_PATTERN = re.compile(
    r"\[\[NEG_SLOT:([A-Z][A-Z0-9_]{1,63}):(slot_[a-z0-9_]{2,63})\]\]"
)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PHONE_PATTERN = re.compile(r"(?<![0-9A-Za-z])(?:\+?86[- ]?)?1[3-9]\d{9}(?![0-9A-Za-z])")
_RESIDENT_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
_LONG_DIGIT_PATTERN = re.compile(r"(?<![0-9A-Za-z.])\d{12,19}(?![0-9A-Za-z.])")
_EMAIL_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@(?:[A-Z0-9-]+\.)+[A-Z]{2,63}"
)
_MAC_PATTERN = re.compile(
    r"(?i)(?<![0-9A-F])(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}(?![0-9A-F])"
)
_IPV4_PATTERN = re.compile(
    r"(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])"
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{12,}"
    ),
)
_ORIGINS = frozenset(
    {"human_authored", "human_authored_ai_assisted", "ai_generated", "unknown"}
)
_ADJUDICATION_REASONS = frozenset(
    {
        "boundary_disagreement",
        "label_disagreement",
        "pii_free_disagreement",
        "guideline_ambiguity",
        "other_documented_privately",
    }
)
_KNOWN_ERROR_CODES = frozenset(
    {
        "adjudication_invalid",
        "adjudication_required",
        "adjudicator_not_independent",
        "aggregate_report_privacy_failure",
        "annotation_independence_failed",
        "annotation_slot_coverage_incomplete",
        "annotation_span_duplicate",
        "annotation_span_invalid",
        "annotation_span_order_invalid",
        "annotation_stage_invalid",
        "annotators_not_independent",
        "batch_too_small",
        "blank_jsonl_line",
        "campaign_mixed",
        "context_text_invalid",
        "contribution_identifier_duplicate",
        "exact_iaa_below_threshold",
        "fixture_and_real_records_mixed",
        "fixture_input_forbidden",
        "group_cross_split",
        "hard_negative_category_invalid",
        "input_empty",
        "input_file_invalid",
        "input_permissions_too_broad",
        "input_size_limit_exceeded",
        "json_duplicate_key",
        "json_parse_failed",
        "language_invalid",
        "negative_slot_invalid",
        "nonhuman_origin_forbidden",
        "origin_invalid",
        "output_already_exists",
        "output_parent_invalid",
        "output_write_failed",
        "per_label_iaa_below_threshold",
        "pii_free_agreement_below_threshold",
        "pii_free_ratio_out_of_range",
        "pii_free_state_invalid",
        "potential_personal_data_detected",
        "potential_secret_detected",
        "privacy_declaration_failed",
        "privacy_review_not_passed",
        "record_schema_invalid",
        "review_roles_not_independent",
        "rights_declaration_failed",
        "schema_version_invalid",
        "slot_identifier_duplicate",
        "slot_marker_malformed",
        "slot_marker_overlap",
        "source_identifier_invalid",
        "source_mixed",
        "split_invalid",
        "surrogate_order_invalid",
        "t0_iaa_below_threshold",
        "validation_failed",
    }
)

_TAXONOMY = load_taxonomy()
_CORE_LABELS = _TAXONOMY.core_label_names
_AUXILIARY_LABELS = _TAXONOMY.auxiliary_label_names
_T0_LABELS = frozenset(
    entity.name
    for entity in _TAXONOMY.label_sets["core"]
    if "T0" in entity.risk_tiers
)


class HumanContributionValidationError(ValueError):
    """A fixed-code error that never reflects submitted text or identifiers."""

    def __init__(self, code: str) -> None:
        if code not in _KNOWN_ERROR_CODES:
            code = "validation_failed"
        self.code = code
        super().__init__(code)


def _fail(code: str) -> HumanContributionValidationError:
    return HumanContributionValidationError(code)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _fail("record_schema_invalid")
    return value


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail("record_schema_invalid")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str]) -> None:
    if set(value) != expected:
        raise _fail("record_schema_invalid")


def _bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise _fail("record_schema_invalid")
    return value


def _string(value: object, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _fail("record_schema_invalid")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise _fail("record_schema_invalid")
    return value


def _nullable_id(value: object) -> str | None:
    if value is None:
        return None
    return _string(value, pattern=_PUBLIC_ID_PATTERN)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail("json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise _fail("json_parse_failed")


@dataclass(frozen=True, slots=True)
class SlotMarker:
    start: int
    end: int
    slot_id: str
    kind: str
    label: str | None


@dataclass(frozen=True, slots=True)
class IntentSpan:
    start: int
    end: int
    label: str
    slot_id: str

    @property
    def key(self) -> tuple[int, int, str, str]:
        return (self.start, self.end, self.label, self.slot_id)


@dataclass(frozen=True, slots=True)
class AnnotationPacket:
    annotator_id: str
    independent: bool
    saw_other_annotation: bool
    spans: tuple[IntentSpan, ...]


@dataclass(frozen=True, slots=True)
class Adjudication:
    status: str
    adjudicator_id: str | None
    reason_codes: tuple[str, ...]
    spans: tuple[IntentSpan, ...]


@dataclass(frozen=True, slots=True)
class ContributionRecord:
    fixture_only: bool
    source_id: str
    campaign_id: str
    contribution_id: str
    split: str
    origin_type: str
    contributor_group_id: str
    privacy_reviewer_id: str
    group_ids: tuple[tuple[str, str], ...]
    language: str
    domain: str
    scene: str
    text: str = field(repr=False)
    pii_free: bool
    hard_negative_categories: tuple[str, ...]
    markers: tuple[SlotMarker, ...]
    annotation_a: AnnotationPacket = field(repr=False)
    annotation_b: AnnotationPacket = field(repr=False)
    adjudication: Adjudication = field(repr=False)


def _parse_markers(text: str) -> tuple[SlotMarker, ...]:
    markers: list[SlotMarker] = []
    occupied: list[tuple[int, int]] = []
    for match in _PII_SLOT_PATTERN.finditer(text):
        markers.append(
            SlotMarker(
                start=match.start(),
                end=match.end(),
                slot_id=match.group(1),
                kind="pii",
                label=None,
            )
        )
        occupied.append((match.start(), match.end()))
    for match in _NEGATIVE_SLOT_PATTERN.finditer(text):
        label = match.group(1)
        if label not in _AUXILIARY_LABELS:
            raise _fail("negative_slot_invalid")
        markers.append(
            SlotMarker(
                start=match.start(),
                end=match.end(),
                slot_id=match.group(2),
                kind="negative",
                label=label,
            )
        )
        occupied.append((match.start(), match.end()))
    markers.sort(key=lambda marker: (marker.start, marker.end))
    if len({marker.slot_id for marker in markers}) != len(markers):
        raise _fail("slot_identifier_duplicate")
    ordered_ranges = sorted(occupied)
    for previous, current in zip(ordered_ranges, ordered_ranges[1:], strict=False):
        if current[0] < previous[1]:
            raise _fail("slot_marker_overlap")

    scrubbed = _PII_SLOT_PATTERN.sub("", text)
    scrubbed = _NEGATIVE_SLOT_PATTERN.sub("", scrubbed)
    if "[[" in scrubbed or "]]" in scrubbed:
        raise _fail("slot_marker_malformed")
    return tuple(markers)


def _scan_context_without_slots(text: str) -> None:
    scrubbed = _PII_SLOT_PATTERN.sub(" ", text)
    scrubbed = _NEGATIVE_SLOT_PATTERN.sub(" ", scrubbed)
    if any(pattern.search(scrubbed) for pattern in _SECRET_PATTERNS):
        raise _fail("potential_secret_detected")
    if _PHONE_PATTERN.search(scrubbed):
        raise _fail("potential_personal_data_detected")
    if _RESIDENT_ID_PATTERN.search(scrubbed):
        raise _fail("potential_personal_data_detected")
    if _LONG_DIGIT_PATTERN.search(scrubbed):
        raise _fail("potential_personal_data_detected")
    if _EMAIL_PATTERN.search(scrubbed):
        raise _fail("potential_personal_data_detected")
    if _MAC_PATTERN.search(scrubbed):
        raise _fail("potential_personal_data_detected")
    if _IPV4_PATTERN.search(scrubbed):
        raise _fail("potential_personal_data_detected")


def _parse_span(value: object, *, marker_by_id: Mapping[str, SlotMarker]) -> IntentSpan:
    raw = _mapping(value)
    _exact_fields(raw, frozenset({"start", "end", "label", "slot_id"}))
    start = raw["start"]
    end = raw["end"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise _fail("annotation_span_invalid")
    label = _string(raw["label"])
    slot_id = _string(raw["slot_id"], pattern=_SLOT_ID_PATTERN)
    marker = marker_by_id.get(slot_id)
    if (
        marker is None
        or marker.kind != "pii"
        or start != marker.start
        or end != marker.end
        or label not in _CORE_LABELS
    ):
        raise _fail("annotation_span_invalid")
    return IntentSpan(start=start, end=end, label=label, slot_id=slot_id)


def _parse_spans(value: object, *, markers: tuple[SlotMarker, ...]) -> tuple[IntentSpan, ...]:
    marker_by_id = {marker.slot_id: marker for marker in markers}
    spans = tuple(_parse_span(item, marker_by_id=marker_by_id) for item in _sequence(value))
    if tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label))) != spans:
        raise _fail("annotation_span_order_invalid")
    if len({span.slot_id for span in spans}) != len(spans):
        raise _fail("annotation_span_duplicate")
    expected_slots = {marker.slot_id for marker in markers if marker.kind == "pii"}
    if {span.slot_id for span in spans} != expected_slots:
        raise _fail("annotation_slot_coverage_incomplete")
    return spans


def _parse_annotation(value: object, *, markers: tuple[SlotMarker, ...]) -> AnnotationPacket:
    raw = _mapping(value)
    _exact_fields(
        raw,
        frozenset({"annotator_id", "independent", "saw_other_annotation", "spans"}),
    )
    packet = AnnotationPacket(
        annotator_id=_string(raw["annotator_id"], pattern=_PUBLIC_ID_PATTERN),
        independent=_bool(raw["independent"]),
        saw_other_annotation=_bool(raw["saw_other_annotation"]),
        spans=_parse_spans(raw["spans"], markers=markers),
    )
    if not packet.independent or packet.saw_other_annotation:
        raise _fail("annotation_independence_failed")
    return packet


def _parse_adjudication(
    value: object,
    *,
    markers: tuple[SlotMarker, ...],
    annotation_a: AnnotationPacket,
    annotation_b: AnnotationPacket,
) -> Adjudication:
    raw = _mapping(value)
    _exact_fields(raw, frozenset({"status", "adjudicator_id", "reason_codes", "spans"}))
    status = _string(raw["status"])
    if status not in {"agreed", "adjudicated"}:
        raise _fail("adjudication_invalid")
    adjudicator_id = _nullable_id(raw["adjudicator_id"])
    reasons = tuple(_string(item) for item in _sequence(raw["reason_codes"]))
    if len(set(reasons)) != len(reasons) or any(
        reason not in _ADJUDICATION_REASONS for reason in reasons
    ):
        raise _fail("adjudication_invalid")
    spans = _parse_spans(raw["spans"], markers=markers)
    annotations_agree = tuple(span.key for span in annotation_a.spans) == tuple(
        span.key for span in annotation_b.spans
    )
    if annotations_agree:
        if status != "agreed" or adjudicator_id is not None or reasons:
            raise _fail("adjudication_invalid")
        if tuple(span.key for span in spans) != tuple(span.key for span in annotation_a.spans):
            raise _fail("adjudication_invalid")
    else:
        if status != "adjudicated" or adjudicator_id is None or not reasons:
            raise _fail("adjudication_required")
        if adjudicator_id in {annotation_a.annotator_id, annotation_b.annotator_id}:
            raise _fail("adjudicator_not_independent")
    return Adjudication(
        status=status,
        adjudicator_id=adjudicator_id,
        reason_codes=reasons,
        spans=spans,
    )


def parse_contribution_record(
    value: object,
    *,
    allow_fixture: bool = False,
) -> ContributionRecord:
    """Parse one exact-schema contribution without granting release admission."""

    raw = _mapping(value)
    _exact_fields(
        raw,
        frozenset(
            {
                "schema_version",
                "fixture_only",
                "source_id",
                "campaign_id",
                "contribution_id",
                "split",
                "origin",
                "rights",
                "privacy",
                "grouping",
                "context",
                "annotation",
                "surrogate_state",
            }
        ),
    )
    if raw["schema_version"] != CONTRIBUTION_SCHEMA_VERSION:
        raise _fail("schema_version_invalid")
    fixture_only = _bool(raw["fixture_only"])
    if fixture_only and not allow_fixture:
        raise _fail("fixture_input_forbidden")
    expected_source = FIXTURE_SOURCE_ID if fixture_only else REAL_SOURCE_ID
    if raw["source_id"] != expected_source:
        raise _fail("source_identifier_invalid")
    if raw["surrogate_state"] != SURROGATE_STATE:
        raise _fail("surrogate_order_invalid")

    origin = _mapping(raw["origin"])
    _exact_fields(
        origin,
        frozenset(
            {
                "type",
                "human_author_attested",
                "generative_ai_used",
                "machine_generated_text_used",
            }
        ),
    )
    origin_type = _string(origin["type"])
    if origin_type not in _ORIGINS:
        raise _fail("origin_invalid")
    if (
        origin_type != "human_authored"
        or not _bool(origin["human_author_attested"])
        or _bool(origin["generative_ai_used"])
        or _bool(origin["machine_generated_text_used"])
    ):
        raise _fail("nonhuman_origin_forbidden")

    rights = _mapping(raw["rights"])
    _exact_fields(
        rights,
        frozenset(
            {
                "original_author",
                "rights_to_license",
                "license_id",
                "agreement_version",
                "public_redistribution_allowed",
                "modification_allowed",
                "model_training_allowed",
                "derived_weight_distribution_allowed",
                "withdrawal_policy_acknowledged",
            }
        ),
    )
    if rights["license_id"] != "CC-BY-4.0" or rights["agreement_version"] != AGREEMENT_VERSION:
        raise _fail("rights_declaration_failed")
    rights_flags = (
        "original_author",
        "rights_to_license",
        "public_redistribution_allowed",
        "modification_allowed",
        "model_training_allowed",
        "derived_weight_distribution_allowed",
        "withdrawal_policy_acknowledged",
    )
    if any(not _bool(rights[name]) for name in rights_flags):
        raise _fail("rights_declaration_failed")

    privacy = _mapping(raw["privacy"])
    _exact_fields(
        privacy,
        frozenset(
            {
                "contains_real_personal_data",
                "contains_third_party_personal_data",
                "copied_from_production",
                "copied_from_private_communication",
                "copied_from_support_or_case_data",
                "copied_from_nonpublic_source",
                "pii_uses_project_slots_only",
                "contributor_self_review_passed",
                "independent_review_status",
                "independent_reviewer_id",
            }
        ),
    )
    prohibited_true = (
        "contains_real_personal_data",
        "contains_third_party_personal_data",
        "copied_from_production",
        "copied_from_private_communication",
        "copied_from_support_or_case_data",
        "copied_from_nonpublic_source",
    )
    if any(_bool(privacy[name]) for name in prohibited_true):
        raise _fail("privacy_declaration_failed")
    if not _bool(privacy["pii_uses_project_slots_only"]) or not _bool(
        privacy["contributor_self_review_passed"]
    ):
        raise _fail("privacy_declaration_failed")
    if privacy["independent_review_status"] != "passed":
        raise _fail("privacy_review_not_passed")
    privacy_reviewer_id = _string(
        privacy["independent_reviewer_id"], pattern=_PUBLIC_ID_PATTERN
    )

    grouping = _mapping(raw["grouping"])
    _exact_fields(
        grouping,
        frozenset(
            {
                "contributor_group_id",
                "document_group_id",
                "conversation_group_id",
                "parent_group_id",
                "near_duplicate_group_id",
            }
        ),
    )
    contributor_group_id = _string(
        grouping["contributor_group_id"], pattern=_PUBLIC_ID_PATTERN
    )
    group_pairs = (
        ("contributor", contributor_group_id),
        ("document", _string(grouping["document_group_id"], pattern=_PUBLIC_ID_PATTERN)),
        ("conversation", _nullable_id(grouping["conversation_group_id"])),
        ("parent", _nullable_id(grouping["parent_group_id"])),
        (
            "near_duplicate",
            _string(grouping["near_duplicate_group_id"], pattern=_PUBLIC_ID_PATTERN),
        ),
    )
    group_ids = tuple((kind, group_id) for kind, group_id in group_pairs if group_id is not None)

    context = _mapping(raw["context"])
    _exact_fields(
        context,
        frozenset(
            {
                "language",
                "domain",
                "scene",
                "text",
                "pii_free",
                "hard_negative_categories",
            }
        ),
    )
    language = _string(context["language"])
    if language != "zh-Hans-CN":
        raise _fail("language_invalid")
    domain = _string(context["domain"], pattern=_SLUG_PATTERN)
    scene = _string(context["scene"], pattern=_SLUG_PATTERN)
    text = context["text"]
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > _MAX_TEXT_CHARACTERS
        or unicodedata.normalize("NFC", text) != text
        or _CJK_PATTERN.search(text) is None
        or any(
            unicodedata.category(character) == "Cc" and character not in "\n\t"
            for character in text
        )
    ):
        raise _fail("context_text_invalid")
    markers = _parse_markers(text)
    _scan_context_without_slots(text)
    pii_free = _bool(context["pii_free"])
    pii_markers = tuple(marker for marker in markers if marker.kind == "pii")
    if pii_free == bool(pii_markers):
        raise _fail("pii_free_state_invalid")
    hard_negative_categories = tuple(
        _string(item) for item in _sequence(context["hard_negative_categories"])
    )
    if len(set(hard_negative_categories)) != len(hard_negative_categories) or any(
        category not in _AUXILIARY_LABELS for category in hard_negative_categories
    ):
        raise _fail("hard_negative_category_invalid")
    marker_categories = {
        marker.label for marker in markers if marker.kind == "negative" and marker.label is not None
    }
    if set(hard_negative_categories) != marker_categories:
        raise _fail("hard_negative_category_invalid")

    annotation = _mapping(raw["annotation"])
    _exact_fields(
        annotation,
        frozenset(
            {
                "stage",
                "guideline_version",
                "annotator_a",
                "annotator_b",
                "adjudication",
            }
        ),
    )
    if annotation["stage"] != ANNOTATION_STAGE:
        raise _fail("annotation_stage_invalid")
    _string(annotation["guideline_version"], pattern=_PUBLIC_ID_PATTERN)
    annotation_a = _parse_annotation(annotation["annotator_a"], markers=markers)
    annotation_b = _parse_annotation(annotation["annotator_b"], markers=markers)
    if annotation_a.annotator_id == annotation_b.annotator_id:
        raise _fail("annotators_not_independent")
    if privacy_reviewer_id in {annotation_a.annotator_id, annotation_b.annotator_id}:
        raise _fail("review_roles_not_independent")
    adjudication = _parse_adjudication(
        annotation["adjudication"],
        markers=markers,
        annotation_a=annotation_a,
        annotation_b=annotation_b,
    )
    if adjudication.adjudicator_id == privacy_reviewer_id:
        raise _fail("review_roles_not_independent")
    review_role_ids = {
        privacy_reviewer_id,
        annotation_a.annotator_id,
        annotation_b.annotator_id,
        adjudication.adjudicator_id,
    }
    if contributor_group_id in review_role_ids:
        raise _fail("review_roles_not_independent")

    split = _string(raw["split"])
    if split not in SPLITS:
        raise _fail("split_invalid")
    return ContributionRecord(
        fixture_only=fixture_only,
        source_id=expected_source,
        campaign_id=_string(raw["campaign_id"], pattern=_PUBLIC_ID_PATTERN),
        contribution_id=_string(raw["contribution_id"], pattern=_PUBLIC_ID_PATTERN),
        split=split,
        origin_type=origin_type,
        contributor_group_id=contributor_group_id,
        privacy_reviewer_id=privacy_reviewer_id,
        group_ids=group_ids,
        language=language,
        domain=domain,
        scene=scene,
        text=text,
        pii_free=pii_free,
        hard_negative_categories=hard_negative_categories,
        markers=markers,
        annotation_a=annotation_a,
        annotation_b=annotation_b,
        adjudication=adjudication,
    )


def _f1(matches: int, predicted: int, gold: int) -> float:
    if predicted == 0 and gold == 0:
        return 1.0
    if predicted == 0 or gold == 0:
        return 0.0
    precision = matches / predicted
    recall = matches / gold
    return 2.0 * precision * recall / (precision + recall)


def _rounded(value: float) -> float:
    return round(float(value), 8)


def _validate_batch_structure(records: tuple[ContributionRecord, ...]) -> None:
    if len(records) < _MIN_BATCH_RECORDS:
        raise _fail("batch_too_small")
    if len({record.contribution_id for record in records}) != len(records):
        raise _fail("contribution_identifier_duplicate")
    if len({record.campaign_id for record in records}) != 1:
        raise _fail("campaign_mixed")
    if len({record.source_id for record in records}) != 1:
        raise _fail("source_mixed")
    if len({record.fixture_only for record in records}) != 1:
        raise _fail("fixture_and_real_records_mixed")

    group_splits: dict[tuple[str, str], str] = {}
    for record in records:
        for group_key in record.group_ids:
            previous = group_splits.setdefault(group_key, record.split)
            if previous != record.split:
                raise _fail("group_cross_split")

    pii_free_count = sum(record.pii_free for record in records)
    pii_free_ratio = pii_free_count / len(records)
    if not _MIN_PII_FREE_RATIO <= pii_free_ratio <= _MAX_PII_FREE_RATIO:
        raise _fail("pii_free_ratio_out_of_range")


def _agreement_report(records: tuple[ContributionRecord, ...]) -> dict[str, Any]:
    a_count = 0
    b_count = 0
    exact_matches = 0
    t0_a_count = 0
    t0_b_count = 0
    t0_matches = 0
    label_counts: Counter[str] = Counter()
    label_a: Counter[str] = Counter()
    label_b: Counter[str] = Counter()
    label_matches: Counter[str] = Counter()
    document_agreements = 0
    adjudicated_documents = 0
    reason_counts: Counter[str] = Counter()
    pii_free_agreements = 0
    pii_free_documents = 0

    for record in records:
        a_keys = {span.key for span in record.annotation_a.spans}
        b_keys = {span.key for span in record.annotation_b.spans}
        matches = a_keys & b_keys
        a_count += len(a_keys)
        b_count += len(b_keys)
        exact_matches += len(matches)
        if a_keys == b_keys:
            document_agreements += 1
        if record.adjudication.status == "adjudicated":
            adjudicated_documents += 1
            reason_counts.update(record.adjudication.reason_codes)
        if record.pii_free:
            pii_free_documents += 1
            if not a_keys and not b_keys:
                pii_free_agreements += 1

        for span in record.annotation_a.spans:
            label_a[span.label] += 1
            if span.label in _T0_LABELS:
                t0_a_count += 1
        for span in record.annotation_b.spans:
            label_b[span.label] += 1
            if span.label in _T0_LABELS:
                t0_b_count += 1
        for key in matches:
            label_matches[key[2]] += 1
            if key[2] in _T0_LABELS:
                t0_matches += 1
        label_counts.update(span.label for span in record.adjudication.spans)

    exact_f1 = _f1(exact_matches, a_count, b_count)
    t0_f1 = _f1(t0_matches, t0_a_count, t0_b_count)
    pii_free_agreement = (
        pii_free_agreements / pii_free_documents if pii_free_documents else 1.0
    )
    per_label_f1 = {
        label: _rounded(_f1(label_matches[label], label_a[label], label_b[label]))
        for label in sorted(set(label_a) | set(label_b))
    }
    if exact_f1 < _MIN_EXACT_IAA:
        raise _fail("exact_iaa_below_threshold")
    if (t0_a_count or t0_b_count) and t0_f1 < _MIN_T0_IAA:
        raise _fail("t0_iaa_below_threshold")
    if pii_free_agreement < _MIN_PII_FREE_AGREEMENT:
        raise _fail("pii_free_agreement_below_threshold")
    if any(value < _MIN_PER_LABEL_IAA for value in per_label_f1.values()):
        raise _fail("per_label_iaa_below_threshold")

    return {
        "annotation_stage": ANNOTATION_STAGE,
        "annotator_a_span_count": a_count,
        "annotator_b_span_count": b_count,
        "exact_span_label_match_count": exact_matches,
        "exact_span_label_f1": _rounded(exact_f1),
        "t0_exact_span_label_f1": _rounded(t0_f1),
        "document_exact_agreement_rate": _rounded(document_agreements / len(records)),
        "pii_free_document_agreement_rate": _rounded(pii_free_agreement),
        "adjudicated_document_count": adjudicated_documents,
        "adjudication_reason_counts": dict(sorted(reason_counts.items())),
        "per_label_exact_f1": per_label_f1,
        "adjudicated_label_span_counts": dict(sorted(label_counts.items())),
        "thresholds": {
            "minimum_exact_span_label_f1": _MIN_EXACT_IAA,
            "minimum_t0_exact_span_label_f1": _MIN_T0_IAA,
            "minimum_pii_free_document_agreement": _MIN_PII_FREE_AGREEMENT,
            "minimum_per_label_exact_f1": _MIN_PER_LABEL_IAA,
        },
    }


def validate_contribution_batch(
    records: Iterable[ContributionRecord],
) -> dict[str, Any]:
    """Validate a complete campaign batch and return an aggregate-only report.

    The returned admission decision is intentionally always false.  A separate
    immutable source-registry decision, post-split surrogate build, residual
    privacy scan, and post-surrogate gold review are still required.
    """

    materialized = tuple(records)
    _validate_batch_structure(materialized)
    agreement = _agreement_report(materialized)
    fixture_only = materialized[0].fixture_only
    pii_free_count = sum(record.pii_free for record in materialized)
    hard_negative_count = sum(bool(record.hard_negative_categories) for record in materialized)
    split_counts = Counter(record.split for record in materialized)
    origin_counts = Counter(record.origin_type for record in materialized)
    report: dict[str, Any] = {
        "schema_version": AGGREGATE_REPORT_SCHEMA_VERSION,
        "status": "fixture_validated_not_releasable"
        if fixture_only
        else "validated_for_quarantine",
        "contains_text": False,
        "contains_record_identifiers": False,
        "contains_paths": False,
        "collected_count": 0 if fixture_only else len(materialized),
        "fixture_record_count": len(materialized) if fixture_only else 0,
        "record_count": len(materialized),
        "origin_counts": dict(sorted(origin_counts.items())),
        "split_document_counts": {split: split_counts[split] for split in SPLITS},
        "composition": {
            "positive_document_count": len(materialized) - pii_free_count,
            "pii_free_document_count": pii_free_count,
            "pii_free_document_ratio": _rounded(pii_free_count / len(materialized)),
            "hard_negative_document_count": hard_negative_count,
            "required_pii_free_ratio": {
                "minimum": _MIN_PII_FREE_RATIO,
                "maximum": _MAX_PII_FREE_RATIO,
            },
        },
        "annotation": agreement,
        "workflow_state": {
            "group_split_checked": True,
            "surrogate_state": SURROGATE_STATE,
            "post_surrogate_double_annotation_required": True,
        },
        "release_admission": {
            "source_registry_bound": False,
            "public_release_allowed": False,
            "public_weight_training_allowed": False,
            "reason": "intake_only_requires_independent_release_gates",
        },
    }
    _assert_aggregate_report(report)
    return report


def _assert_aggregate_report(report: Mapping[str, Any]) -> None:
    """Defense in depth: reject any accidental string resembling private content."""

    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    forbidden_keys = (
        '"text"',
        '"contribution_id"',
        '"campaign_id"',
        '"contributor_group_id"',
        '"annotator_id"',
        '"independent_reviewer_id"',
        '"path"',
        '"sha256"',
    )
    if any(key in serialized for key in forbidden_keys):
        raise _fail("aggregate_report_privacy_failure")


def parse_jsonl_stream(
    stream: TextIO,
    *,
    allow_fixtures: bool = False,
) -> tuple[ContributionRecord, ...]:
    """Parse bounded JSONL without reflecting a line or record in failures."""

    records: list[ContributionRecord] = []
    total_bytes = 0
    try:
        for line in stream:
            encoded_size = len(line.encode("utf-8"))
            total_bytes += encoded_size
            if encoded_size > _MAX_LINE_BYTES or total_bytes > _MAX_INPUT_BYTES:
                raise _fail("input_size_limit_exceeded")
            if not line.strip():
                raise _fail("blank_jsonl_line")
            try:
                document = json.loads(
                    line,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except HumanContributionValidationError:
                raise
            except (UnicodeError, json.JSONDecodeError, RecursionError):
                raise _fail("json_parse_failed") from None
            records.append(parse_contribution_record(document, allow_fixture=allow_fixtures))
    except UnicodeError:
        raise _fail("json_parse_failed") from None
    if not records:
        raise _fail("input_empty")
    return tuple(records)


def validate_jsonl_stream(
    stream: TextIO,
    *,
    allow_fixtures: bool = False,
) -> dict[str, Any]:
    return validate_contribution_batch(
        parse_jsonl_stream(stream, allow_fixtures=allow_fixtures)
    )


def validate_private_jsonl_path(path: str | Path) -> dict[str, Any]:
    """Validate a private regular JSONL file; fixture bypass is unavailable."""

    descriptor = -1
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(Path(path), flags)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise _fail("input_file_invalid")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise _fail("input_permissions_too_broad")
        if file_stat.st_size > _MAX_INPUT_BYTES:
            raise _fail("input_size_limit_exceeded")
        with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as stream:
            descriptor = -1
            return validate_jsonl_stream(stream, allow_fixtures=False)
    except HumanContributionValidationError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError):
        raise _fail("input_file_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_aggregate_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Create one aggregate report atomically with private file permissions."""

    _assert_aggregate_report(report)
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise _fail("output_already_exists")
    parent = output.parent
    if not parent.is_dir():
        raise _fail("output_parent_invalid")
    temporary = parent / f".{output.name}.tmp-{os.getpid()}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
    except HumanContributionValidationError:
        raise
    except (OSError, ValueError, TypeError):
        raise _fail("output_write_failed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def rejected_aggregate_report(code: str) -> dict[str, Any]:
    """Return a fixed, no-text CLI rejection receipt."""

    safe_code = HumanContributionValidationError(code).code
    report = {
        "schema_version": AGGREGATE_REPORT_SCHEMA_VERSION,
        "status": "rejected",
        "error_code": safe_code,
        "contains_text": False,
        "contains_record_identifiers": False,
        "contains_paths": False,
        "collected_count": 0,
        "release_admission": {
            "source_registry_bound": False,
            "public_release_allowed": False,
            "public_weight_training_allowed": False,
        },
    }
    _assert_aggregate_report(report)
    return report


__all__ = [
    "AGGREGATE_REPORT_SCHEMA_VERSION",
    "AGREEMENT_VERSION",
    "ANNOTATION_STAGE",
    "CONTRIBUTION_SCHEMA_VERSION",
    "FIXTURE_SOURCE_ID",
    "HumanContributionValidationError",
    "REAL_SOURCE_ID",
    "SPLITS",
    "SURROGATE_STATE",
    "ContributionRecord",
    "parse_contribution_record",
    "parse_jsonl_stream",
    "rejected_aggregate_report",
    "validate_contribution_batch",
    "validate_jsonl_stream",
    "validate_private_jsonl_path",
    "write_aggregate_report",
]
