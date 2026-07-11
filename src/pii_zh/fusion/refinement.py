"""Privacy-safe structured-validator refinement for model/rule predictions."""

from __future__ import annotations

from collections import Counter

from pii_zh.data.validators import validate_entity_value
from pii_zh.evaluation.io import PredictionRecord

_IDENTITY_CONTEXT = ("身份证", "证件号", "公民身份", "无效证件", "证件串")
_BANK_CARD_CONTEXT = ("银行卡", "卡号", "信用卡")


def _nearest_context_distance(
    text: str,
    start: int,
    end: int,
    phrases: tuple[str, ...],
) -> int | None:
    window_start = max(0, start - 16)
    window_end = min(len(text), end + 16)
    window = text[window_start:window_end]
    distances: list[int] = []
    for phrase in phrases:
        offset = window.find(phrase)
        while offset >= 0:
            phrase_start = window_start + offset
            phrase_end = phrase_start + len(phrase)
            distances.append(
                start - phrase_end
                if phrase_end <= start
                else phrase_start - end
                if phrase_start >= end
                else 0
            )
            offset = window.find(phrase, offset + 1)
    return min(distances) if distances else None


def _bank_card_is_in_a_stronger_identity_context(text: str, start: int, end: int) -> bool:
    identity_distance = _nearest_context_distance(text, start, end, _IDENTITY_CONTEXT)
    card_distance = _nearest_context_distance(text, start, end, _BANK_CARD_CONTEXT)
    return identity_distance is not None and (
        card_distance is None or identity_distance < card_distance
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
        if span.label == "BANK_CARD_NUMBER" and _bank_card_is_in_a_stronger_identity_context(
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
