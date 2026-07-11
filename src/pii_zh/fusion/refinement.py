"""Privacy-safe structured-validator refinement for model/rule predictions."""

from __future__ import annotations

import re
from collections import Counter

from pii_zh.data.validators import validate_entity_value
from pii_zh.evaluation.io import PredictionRecord

_IDENTITY_CONTEXT = ("身份证", "证件号", "公民身份", "无效证件", "证件串")
_BANK_ACCOUNT_CONTEXT = (
    "银行账户号码",
    "银行账户号",
    "银行账号",
    "银行账户",
    "工资账户",
    "薪资账户",
    "收款账户",
    "结算账户",
    "入账账户",
    "转账账户",
    "退款账户",
    "付款账户",
    "bank account",
    "account number",
    "account no",
)
_BANK_CARD_CONTEXT = (
    "银行卡号",
    "银行卡",
    "信用卡号",
    "信用卡",
    "借记卡号",
    "借记卡",
    "储蓄卡号",
    "储蓄卡",
    "工资卡",
    "卡号",
)
_STUDENT_CONTEXT = ("学号", "学生", "学籍", "校园卡", "在校")
_BIRTH_CONTEXT_RE = re.compile(
    r"date\s+of\s+birth|birth[_ -]?date|dob|出生(?:日期|年月|信息)?|生日|生于|年龄",
    re.IGNORECASE,
)
_BIRTH_NEGATING_PREFIX_RE = re.compile(
    r"(?:不是|并非|非|不属于|不代表|不含|不涉及|无需|不要|未提供|未登记|未知|无).{0,5}$",
    re.IGNORECASE,
)
_BIRTH_NEGATING_SUFFIX_RE = re.compile(
    r"^(?:信息|字段|日期|数据)?\s*(?:无关|不详|未知|缺失|不存在|不可用|不适用|否|未提供)",
    re.IGNORECASE,
)


def _nearest_context_distance(
    text: str,
    start: int,
    end: int,
    phrases: tuple[str, ...],
) -> int | None:
    window_start = max(0, start - 16)
    window_end = min(len(text), end + 16)
    window = text[window_start:window_end].lower()
    distances: list[int] = []
    for phrase in phrases:
        normalized = phrase.lower()
        offset = window.find(normalized)
        while offset >= 0:
            phrase_start = window_start + offset
            phrase_end = phrase_start + len(normalized)
            distances.append(
                (start - phrase_end) * 2
                if phrase_end <= start
                else (phrase_start - end) * 2 + 1
                if phrase_start >= end
                else 0
            )
            offset = window.find(normalized, offset + 1)
    return min(distances) if distances else None


def _is_in_a_stronger_context(
    text: str,
    start: int,
    end: int,
    *,
    stronger: tuple[str, ...],
    weaker: tuple[str, ...],
) -> bool:
    stronger_distance = _nearest_context_distance(text, start, end, stronger)
    weaker_distance = _nearest_context_distance(text, start, end, weaker)
    return stronger_distance is not None and (
        weaker_distance is None or stronger_distance < weaker_distance
    )


def _nearest_preceding_context_distance(
    text: str, start: int, phrases: tuple[str, ...]
) -> int | None:
    window_start = max(0, start - 16)
    prefix = text[window_start:start].lower()
    distances = [
        start - (window_start + offset + len(phrase))
        for phrase in (item.lower() for item in phrases)
        if (offset := prefix.rfind(phrase)) >= 0
    ]
    return min(distances) if distances else None


def _bank_card_is_in_a_stronger_identity_context(text: str, start: int, end: int) -> bool:
    identity_distance = _nearest_preceding_context_distance(text, start, _IDENTITY_CONTEXT)
    card_distance = _nearest_preceding_context_distance(text, start, _BANK_CARD_CONTEXT)
    preceding_identity_is_stronger = identity_distance is not None and (
        card_distance is None or identity_distance < card_distance
    )
    return preceding_identity_is_stronger or _is_in_a_stronger_context(
        text,
        start,
        end,
        stronger=_IDENTITY_CONTEXT,
        weaker=_BANK_CARD_CONTEXT,
    )


def _bank_card_is_in_a_stronger_account_context(text: str, start: int, end: int) -> bool:
    return _is_in_a_stronger_context(
        text,
        start,
        end,
        stronger=_BANK_ACCOUNT_CONTEXT,
        weaker=_BANK_CARD_CONTEXT,
    )


def _birth_context_is_negated(window: str, marker_start: int, marker_end: int) -> bool:
    prefix = window[max(0, marker_start - 10) : marker_start]
    suffix = window[marker_end : marker_end + 10]
    return (
        _BIRTH_NEGATING_PREFIX_RE.search(prefix) is not None
        or _BIRTH_NEGATING_SUFFIX_RE.match(suffix) is not None
    )


def _has_affirmative_birth_context(text: str, start: int, end: int) -> bool:
    window_start = max(0, start - 24)
    window_end = min(len(text), end + 24)
    window = text[window_start:window_end]
    candidates: list[tuple[int, bool]] = []
    for marker in _BIRTH_CONTEXT_RE.finditer(window):
        marker_start = window_start + marker.start()
        marker_end = window_start + marker.end()
        distance = (
            start - marker_end
            if marker_end <= start
            else marker_start - end
            if marker_start >= end
            else 0
        )
        candidates.append(
            (distance, not _birth_context_is_negated(window, marker.start(), marker.end()))
        )
    if not candidates:
        return False
    nearest_distance = min(distance for distance, _ in candidates)
    return any(affirmative for distance, affirmative in candidates if distance == nearest_distance)


def _enclosing_digit_run(text: str, start: int, end: int) -> tuple[int, int] | None:
    if not text[start:end].isdigit():
        return None
    run_start = start
    run_end = end
    while run_start > 0 and text[run_start - 1].isdigit():
        run_start -= 1
    while run_end < len(text) and text[run_end].isdigit():
        run_end += 1
    return run_start, run_end


def _student_id_is_an_identity_number_fragment(text: str, start: int, end: int) -> bool:
    digit_run = _enclosing_digit_run(text, start, end)
    if digit_run is None or digit_run[1] - digit_run[0] not in {15, 18}:
        return False
    return _is_in_a_stronger_context(
        text,
        *digit_run,
        stronger=_IDENTITY_CONTEXT,
        weaker=_STUDENT_CONTEXT,
    )


def suppress_invalid_structured_spans(
    record: PredictionRecord,
    text: str,
) -> tuple[PredictionRecord, Counter[str]]:
    """Remove validator-rejected spans while retaining semantic labels unchanged."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    record.validate(text_length=len(text))
    kept = []
    suppressed: Counter[str] = Counter()
    for span in record.spans:
        if span.label == "BANK_CARD_NUMBER":
            if _bank_card_is_in_a_stronger_identity_context(
                text, span.start, span.end
            ) or _bank_card_is_in_a_stronger_account_context(text, span.start, span.end):
                suppressed[span.label] += 1
                continue
        if span.label == "STUDENT_ID" and _student_id_is_an_identity_number_fragment(
            text, span.start, span.end
        ):
            suppressed[span.label] += 1
            continue
        if span.label == "DATE_OF_BIRTH" and not _has_affirmative_birth_context(
            text, span.start, span.end
        ):
            suppressed[span.label] += 1
            continue
        validation = validate_entity_value(span.label, text[span.start : span.end])
        if validation is not None and not validation.valid:
            suppressed[span.label] += 1
            continue
        kept.append(span)
    return PredictionRecord(doc_id=record.doc_id, spans=tuple(kept)), suppressed


__all__ = ["suppress_invalid_structured_spans"]
