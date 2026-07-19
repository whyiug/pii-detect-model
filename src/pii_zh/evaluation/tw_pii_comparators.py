"""Pre-result mappings and deterministic fusion for tw-PII open baselines."""

from __future__ import annotations

from collections.abc import Sequence

from .io import Span

ERNIE_EFFECTIVE_MAPPING: dict[str, str] = {
    "address": "private_address",
    "email": "private_email",
    "mobile": "private_phone",
    "name": "private_person",
}

CLUENER_EFFECTIVE_MAPPING: dict[str, str] = {
    "address": "private_address",
    "name": "private_person",
}

AIGUARD_EFFECTIVE_MAPPING: dict[str, str] = {
    "address": "private_address",
    "bank_card": "account_number",
    "bank_password": "secret",
    "birth_date": "private_date",
    "credit_card": "account_number",
    "drivers_license": "account_number",
    "email": "private_email",
    "hkmtp_pass": "account_number",
    "id_card": "account_number",
    "mobile": "private_phone",
    "name": "private_person",
    "passport": "account_number",
    "social_security": "account_number",
}

RULE_EFFECTIVE_MAPPING: dict[str, str] = {
    "BANK_ACCOUNT_NUMBER": "account_number",
    "BANK_CARD_NUMBER": "account_number",
    "CN_ID_CARD": "account_number",
    "DRIVER_LICENSE_NUMBER": "account_number",
    "EMAIL_ADDRESS": "private_email",
    "EMPLOYEE_ID": "account_number",
    "PASSPORT_NUMBER": "account_number",
    "PHONE_NUMBER": "private_phone",
    "POSTAL_CODE": "private_address",
    "SECRET": "secret",
    "STUDENT_ID": "account_number",
}


def _span_score(span: Span) -> float:
    return float(span.score if span.score is not None else 0.0)


def merge_framework_hybrid_spans(
    model_spans: Sequence[Span], rule_spans: Sequence[Span]
) -> tuple[Span, ...]:
    """Union model and rules with frozen same-label overlap resolution.

    Different labels remain visible.  Exact duplicates and same-label spans
    overlapping at least half of the shorter interval select by score, then
    longer length, then deterministic offsets.
    """

    exact: dict[tuple[int, int, str], Span] = {}
    for span in [*model_spans, *rule_spans]:
        span.validate()
        key = span.metric_key()
        existing = exact.get(key)
        if existing is None or _span_score(span) > _span_score(existing):
            exact[key] = span

    kept: list[Span] = []
    for candidate in sorted(
        exact.values(),
        key=lambda item: (item.label, item.start, item.end, -_span_score(item)),
    ):
        duplicate_indices: list[int] = []
        for index, existing in enumerate(kept):
            if existing.label != candidate.label:
                continue
            intersection = max(
                0,
                min(existing.end, candidate.end) - max(existing.start, candidate.start),
            )
            shorter = min(existing.end - existing.start, candidate.end - candidate.start)
            if intersection and intersection / shorter >= 0.5:
                duplicate_indices.append(index)
        if not duplicate_indices:
            kept.append(candidate)
            continue
        cluster = [candidate, *(kept[index] for index in duplicate_indices)]
        winner = max(
            cluster,
            key=lambda item: (
                _span_score(item),
                item.end - item.start,
                -item.start,
                -item.end,
            ),
        )
        for index in reversed(duplicate_indices):
            kept.pop(index)
        kept.append(winner)
    return tuple(
        sorted(kept, key=lambda item: (item.start, item.end, item.label, -_span_score(item)))
    )


__all__ = [
    "AIGUARD_EFFECTIVE_MAPPING",
    "CLUENER_EFFECTIVE_MAPPING",
    "ERNIE_EFFECTIVE_MAPPING",
    "RULE_EFFECTIVE_MAPPING",
    "merge_framework_hybrid_spans",
]
