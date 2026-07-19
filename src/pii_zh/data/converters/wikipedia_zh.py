"""Fail-closed conversion of fixed-revision Chinese Wikipedia Viewer rows.

The converter accepts already-loaded Dataset Viewer rows and performs no
downloads.  It only emits unannotated quarantine candidates after conservative
biography, living-person, contact, and identifier screening.  No source text is
generated, rewritten, or augmented by a model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any
from urllib.parse import urlparse

from ..schema import DocumentRecord, Provenance, QualityMetadata, QualityTier
from ._common import ConversionError

WIKIPEDIA_ZH_SOURCE_ID = "wikimedia/wikipedia"
WIKIPEDIA_ZH_CONFIG = "20231101.zh"
WIKIPEDIA_ZH_SPLIT = "train"
WIKIPEDIA_ZH_REVISION = "b60f7f34e19c4cc122335eeb0ebb775547057fb5"
WIKIPEDIA_ZH_DECLARED_LICENSE = "CC-BY-SA-3.0 AND GFDL"
WIKIPEDIA_ZH_CONTEXT_ORIGIN = "open_human_edited_encyclopedia"
WIKIPEDIA_ZH_FIELDS = frozenset({"id", "url", "title", "text"})

_MIN_TEXT_LENGTH = 140
_MAX_TEXT_LENGTH = 6000
_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PAGE_ID_PATTERN = re.compile(r"^[0-9]{1,20}$")
_EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z])"
)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)|"
    r"(?<!\d)(?:0\d{2,3}[-－— ]?)?\d{7,8}(?:[-－— ]\d{1,6})?(?!\d)"
)
_PRC_ID_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(?:\d{6})(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])"
)
_LONG_ID_PATTERN = re.compile(r"(?<!\d)\d{9,}(?!\d)")
_IPV4_PATTERN = re.compile(
    r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
)
_URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)")
_CONTACT_KEYWORD_PATTERN = re.compile(
    r"(?:联系方式|联系电话|联络电话|手机号码|电话号码|电子邮箱|电子邮件|"
    r"邮箱地址|微信号|QQ号|通讯地址|通信地址|邮政编码|身份证号|护照号码|"
    r"社会保障号|银行卡号|开户行)"
)
_BIOGRAPHY_TITLE_PATTERN = re.compile(
    r"(?:人物列表|人物名录|校友列表|成员列表|传记|生平|讣告|演员|歌手|"
    r"政治人物|运动员|球员|作家|诗人|画家|主持人|企业家)$"
)
_BIOGRAPHY_LEAD_PATTERN = re.compile(
    r"(?:出生于|生于|出生日期|逝世于|卒于|逝世日期|本名|原名|籍贯|"
    r"生卒年|享年|配偶|父亲|母亲|丈夫|妻子|儿子|女儿)|"
    r"[（(](?:18|19|20)\d{2}年\d{0,2}月?\d{0,2}日?\s*[—–－~至-]"
)
_LIVING_PERSON_PATTERN = re.compile(
    r"(?:仍然健在|仍健在|目前健在|现年\d{1,3}岁|现任.{0,30}(?:主席|总裁|"
    r"主任|教授|议员|部长|市长|校长|教练))"
)


def _non_empty_string(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"Wikipedia zh {field} failed type validation")
    return value


def _row_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversionError("Wikipedia zh row index failed type validation")
    return value


def _validate_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(row, Mapping) or set(row) != WIKIPEDIA_ZH_FIELDS:
        raise ConversionError("Wikipedia zh row failed schema validation")
    page_id = _non_empty_string("page id", row["id"])
    if _PAGE_ID_PATTERN.fullmatch(page_id) is None:
        raise ConversionError("Wikipedia zh page id failed validation")
    url = _non_empty_string("url", row["url"])
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ConversionError("Wikipedia zh url failed validation") from exc
    if (
        parsed.scheme != "https"
        or hostname != "zh.wikipedia.org"
        or not parsed.path.startswith("/wiki/")
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConversionError("Wikipedia zh url failed validation")
    _non_empty_string("title", row["title"])
    _non_empty_string("text", row["text"])
    return row


def wikipedia_zh_exclusion_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return conservative, deterministic quarantine-import exclusion reasons."""

    validated = _validate_row(row)
    title = str(validated["title"])
    text = str(validated["text"])
    lead = text[:1200]
    reasons: list[str] = []
    if len(text) < _MIN_TEXT_LENGTH:
        reasons.append("text_too_short")
    if len(text) > _MAX_TEXT_LENGTH:
        reasons.append("text_too_long")
    if _BIOGRAPHY_TITLE_PATTERN.search(title):
        reasons.append("biography_title")
    if _BIOGRAPHY_LEAD_PATTERN.search(lead):
        reasons.append("biography_or_personal_relation_lead")
    if _LIVING_PERSON_PATTERN.search(lead):
        reasons.append("living_person_lead")
    if _CONTACT_KEYWORD_PATTERN.search(text):
        reasons.append("contact_or_identity_keyword")
    if _EMAIL_PATTERN.search(text):
        reasons.append("email_pattern")
    if _PHONE_PATTERN.search(text):
        reasons.append("phone_pattern")
    if _PRC_ID_PATTERN.search(text):
        reasons.append("prc_identity_pattern")
    if _LONG_ID_PATTERN.search(text):
        reasons.append("long_numeric_identifier_pattern")
    if _IPV4_PATTERN.search(text):
        reasons.append("network_identifier_pattern")
    if _URL_PATTERN.search(text):
        reasons.append("embedded_url_pattern")
    return tuple(reasons)


def _raw_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _opaque_doc_id(*, source_revision: str, page_id: str, row_index: int) -> str:
    material = (
        f"{WIKIPEDIA_ZH_SOURCE_ID}\0{source_revision}\0{WIKIPEDIA_ZH_CONFIG}\0"
        f"{WIKIPEDIA_ZH_SPLIT}\0{row_index}\0{page_id}"
    ).encode()
    return "wikipedia-zh-page:sha256:" + hashlib.sha256(material).hexdigest()


def convert_wikipedia_zh_record(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source_revision: str = WIKIPEDIA_ZH_REVISION,
    source_license: str = WIKIPEDIA_ZH_DECLARED_LICENSE,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> DocumentRecord:
    """Convert one screened page into an unannotated quarantine candidate."""

    row_index = _row_index(row_index)
    source_revision = _non_empty_string("source revision", source_revision)
    source_license = _non_empty_string("source license", source_license)
    if _SHA1_PATTERN.fullmatch(source_revision) is None:
        raise ConversionError("Wikipedia zh source revision failed validation")
    if source_revision != WIKIPEDIA_ZH_REVISION:
        raise ConversionError("Wikipedia zh converter requires the audited fixed revision")
    if quality_gate is not False or public_weight_training_allowed is not None:
        raise ConversionError("Wikipedia zh conversion emits quarantine candidates only")
    reasons = wikipedia_zh_exclusion_reasons(row)
    if reasons:
        raise ConversionError("Wikipedia zh row is excluded by the quarantine import policy")

    page_id = str(row["id"])
    url = str(row["url"])
    title = str(row["title"])
    text = str(row["text"])
    doc_id = _opaque_doc_id(
        source_revision=source_revision,
        page_id=page_id,
        row_index=row_index,
    )
    attribution = {
        # These are public page-level attribution fields retained only inside
        # the 0600 quarantine record.  Public manifests contain aggregates only.
        "source_page_title": title,
        "source_page_url": url,
        "source_page_id_sha256": _raw_sha256(page_id),
        "source_page_title_sha256": _raw_sha256(title),
        "source_page_url_sha256": _raw_sha256(url),
        "source_page_revision_id": None,
        "source_page_revision_status": "unavailable_in_hf_snapshot_schema",
        "hf_viewer_row_index": row_index,
    }
    provenance = Provenance(
        source_id=WIKIPEDIA_ZH_SOURCE_ID,
        source_kind="public_human_edited_encyclopedia",
        source_revision=source_revision,
        license=source_license,
        synthetic=False,
        metadata={
            "converted_from_dataset_viewer": True,
            "source_schema": "hf-wikimedia-wikipedia-id-url-title-text",
            "dataset_config": WIKIPEDIA_ZH_CONFIG,
            "dataset_split": WIKIPEDIA_ZH_SPLIT,
            "context_origin": WIKIPEDIA_ZH_CONTEXT_ORIGIN,
            "human_authored_context": True,
            "model_generated_context_added": False,
            "source_per_page_ai_provenance_verified": False,
            "contains_real_personal_data": None,
            "privacy_review_status": "pending",
            "license_and_share_alike_review_status": "pending",
            **attribution,
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.UNCERTAIN,
        quality_gate=False,
        validators_passed=False,
        review_status="privacy_license_attribution_and_human_review_pending",
        confidence=None,
        metadata={
            "structural_schema_validated": True,
            "aggregate_privacy_prescreen_passed": True,
            "human_annotation_complete": False,
            "privacy_review_complete": False,
            "page_revision_attribution_complete": False,
        },
    )
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(),
        language="zh",
        domain="general_encyclopedia",
        scene="wikipedia_zh_article",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=None,
        conversation_group=None,
        entity_value_groups=(),
        metadata={
            "context_origin": WIKIPEDIA_ZH_CONTEXT_ORIGIN,
            "human_authored_context": True,
            "model_generated_context_added": False,
            "contains_real_personal_data": None,
            "privacy_review_status": "pending",
            "annotation_status": "pending",
            "training_role": "quarantine_candidate_only",
            "derived_source_status": "pending",
            **attribution,
        },
    )


def iter_wikipedia_zh_records(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    source_revision: str = WIKIPEDIA_ZH_REVISION,
    source_license: str = WIKIPEDIA_ZH_DECLARED_LICENSE,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> Iterator[DocumentRecord]:
    """Yield screened pages and fail closed on schema or source drift."""

    if quality_gate is not False or public_weight_training_allowed is not None:
        raise ConversionError("Wikipedia zh conversion emits quarantine candidates only")
    for row_index, row in rows:
        reasons = wikipedia_zh_exclusion_reasons(row)
        if reasons:
            continue
        yield convert_wikipedia_zh_record(
            row,
            row_index=row_index,
            source_revision=source_revision,
            source_license=source_license,
        )


__all__ = [
    "WIKIPEDIA_ZH_CONFIG",
    "WIKIPEDIA_ZH_CONTEXT_ORIGIN",
    "WIKIPEDIA_ZH_DECLARED_LICENSE",
    "WIKIPEDIA_ZH_FIELDS",
    "WIKIPEDIA_ZH_REVISION",
    "WIKIPEDIA_ZH_SOURCE_ID",
    "WIKIPEDIA_ZH_SPLIT",
    "convert_wikipedia_zh_record",
    "iter_wikipedia_zh_records",
    "wikipedia_zh_exclusion_reasons",
]
