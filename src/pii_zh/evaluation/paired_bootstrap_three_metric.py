"""Three-metric paired document bootstrap for a frozen comparator universe.

This module is a successor to :mod:`pii_zh.evaluation.paired_bootstrap`; the
historical strict-micro-only implementation remains unchanged so already
frozen evidence can still be replayed byte-for-byte.

The implementation operates on privacy-safe document sufficient statistics.
It reports strict micro F1, strict macro F1 over every protocol label, and
PII-free document false-positive rate. Candidate/comparator documents are
always sampled together. Mutually exclusive atomic suites are stratified by
explicit PII-free eligibility and protocol-gold support, preserving all
required denominators in every replicate. Possibly overlapping reporting
slices reuse that one global plan and never duplicate documents in the pooled
resample.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .io import EvaluationDataError, GoldRecord, PredictionRecord
from .paired_bootstrap import StrictCounts

FROZEN_SAMPLES = 1000
FROZEN_CONFIDENCE = 0.95
FROZEN_SEED = 42
FROZEN_FPR_NONINFERIORITY_MARGIN = 0.005
FROZEN_FPR_ABSOLUTE_MAXIMUM = 0.05

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_METRICS = (
    "strict_micro_f1",
    "strict_macro_f1",
    "pii_free_document_fpr",
)
_POOLED_SCOPE = "pooled"


@dataclass(frozen=True, slots=True)
class ThreeMetricSystemDocument:
    """One system's sufficient statistics for one document."""

    prediction_keys_sha256: str
    strict_micro: StrictCounts
    strict_per_label: tuple[StrictCounts, ...]
    pii_free_false_positive: bool


@dataclass(frozen=True, slots=True)
class ThreeMetricPairedDocument:
    """Candidate/comparator statistics for the same gold document."""

    document_key_sha256: str
    gold_keys_sha256: str
    reporting_slices: tuple[str, ...]
    pii_free_eligible: bool
    candidate: ThreeMetricSystemDocument
    comparator: ThreeMetricSystemDocument


@dataclass(frozen=True, slots=True)
class ThreeMetricPairedSuite:
    """Privacy-safe aligned documents for one evaluation suite."""

    name: str
    labels: tuple[str, ...]
    alignment_sha256: str
    documents: tuple[ThreeMetricPairedDocument, ...]


@dataclass(frozen=True, slots=True)
class _MetricValues:
    strict_micro_f1: Fraction
    strict_macro_f1: Fraction
    pii_free_document_fpr: Fraction | None
    protocol_gold_spans: int
    pii_free_documents: int

    def to_dict(self) -> dict[str, float | None]:
        return {
            "strict_micro_f1": float(self.strict_micro_f1),
            "strict_macro_f1": float(self.strict_macro_f1),
            "pii_free_document_fpr": (
                None if self.pii_free_document_fpr is None else float(self.pii_free_document_fpr)
            ),
        }


class _Sha256CounterRng:
    """Versioned cross-runtime deterministic rejection-sampling PRNG."""

    def __init__(self, seed: int) -> None:
        self._seed = str(seed).encode("ascii")
        self._counter = 0

    def randbelow(self, upper: int) -> int:
        if isinstance(upper, bool) or not isinstance(upper, int) or upper < 1:
            raise ValueError("PRNG upper bound must be a positive integer")
        modulus = 1 << 256
        limit = modulus - modulus % upper
        while True:
            digest = hashlib.sha256(
                b"pii-zh-three-metric-bootstrap-prng-v1\0"
                + len(self._seed).to_bytes(4, "big")
                + self._seed
                + self._counter.to_bytes(16, "big")
            ).digest()
            self._counter += 1
            value = int.from_bytes(digest, "big")
            if value < limit:
                return value % upper


def _normalized_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty normalized string")
    return value


def _protocol_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence of label strings, not a string")
    values = tuple(labels)
    if not values:
        raise ValueError("labels must be non-empty")
    for label in values:
        _normalized_identifier(label, field="label")
    if len(values) != len(set(values)):
        raise ValueError("labels must be unique")
    return values


def _reporting_slices(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("reporting slices must be a sequence of strings, not a string")
    slices = tuple(values)
    if not slices:
        raise ValueError("every document must belong to at least one reporting slice")
    for value in slices:
        _normalized_identifier(value, field="reporting slice")
        if value == _POOLED_SCOPE:
            raise ValueError("pooled is reserved and cannot be a reporting slice name")
    if len(slices) != len(set(slices)):
        raise ValueError("reporting slices must be unique per document")
    return tuple(sorted(slices))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _document_key_sha256(doc_id: str) -> str:
    return hashlib.sha256(
        b"pii-zh-three-metric-document-key-v1\0" + doc_id.encode("utf-8")
    ).hexdigest()


def _span_keys_sha256(
    *,
    domain: str,
    doc_id: str,
    keys: set[tuple[int, int, str]],
) -> str:
    return _canonical_sha256(
        {
            "domain": domain,
            "document_key_sha256": _document_key_sha256(doc_id),
            "span_metric_keys": [list(key) for key in sorted(keys)],
        }
    )


def _suite_alignment_sha256(
    *,
    name: str,
    labels: Sequence[str],
    documents: Sequence[ThreeMetricPairedDocument],
) -> str:
    return _canonical_sha256(
        {
            "domain": "pii-zh-three-metric-suite-alignment-v2",
            "suite": name,
            "labels": list(labels),
            "documents": [
                {
                    "document_key_sha256": document.document_key_sha256,
                    "gold_keys_sha256": document.gold_keys_sha256,
                    "reporting_slices": list(document.reporting_slices),
                    "pii_free_eligible": document.pii_free_eligible,
                    "candidate": {
                        "prediction_keys_sha256": (document.candidate.prediction_keys_sha256),
                        "strict_micro": {
                            "fn": document.candidate.strict_micro.fn,
                            "fp": document.candidate.strict_micro.fp,
                            "tp": document.candidate.strict_micro.tp,
                        },
                        "strict_per_label": [
                            {"fn": counts.fn, "fp": counts.fp, "tp": counts.tp}
                            for counts in document.candidate.strict_per_label
                        ],
                        "pii_free_false_positive": (document.candidate.pii_free_false_positive),
                    },
                }
                for document in documents
            ],
        }
    )


def _strict_counts_for_keys(
    gold_keys: set[tuple[int, int, str]],
    prediction_keys: set[tuple[int, int, str]],
) -> StrictCounts:
    matched = gold_keys & prediction_keys
    return StrictCounts(
        tp=len(matched),
        fp=len(prediction_keys - matched),
        fn=len(gold_keys - matched),
    )


def _system_document(
    *,
    doc_id: str,
    gold_keys: set[tuple[int, int, str]],
    prediction: PredictionRecord,
    text_length: int | None,
    labels: tuple[str, ...],
    pii_free_eligible: bool,
) -> ThreeMetricSystemDocument:
    prediction.validate(text_length=text_length)
    prediction_keys = {span.metric_key() for span in prediction.spans}
    label_set = frozenset(labels)
    unexpected = {label for _, _, label in prediction_keys} - label_set
    if unexpected:
        raise EvaluationDataError(
            "predictions contain labels outside the three-metric comparison protocol"
        )
    per_label = tuple(
        _strict_counts_for_keys(
            {key for key in gold_keys if key[2] == label},
            {key for key in prediction_keys if key[2] == label},
        )
        for label in labels
    )
    return ThreeMetricSystemDocument(
        prediction_keys_sha256=_span_keys_sha256(
            domain="pii-zh-three-metric-prediction-keys-v1",
            doc_id=doc_id,
            keys=prediction_keys,
        ),
        strict_micro=_strict_counts_for_keys(gold_keys, prediction_keys),
        strict_per_label=per_label,
        pii_free_false_positive=pii_free_eligible and bool(prediction_keys),
    )


def build_three_metric_paired_suite(
    *,
    name: str,
    gold_records: Sequence[GoldRecord],
    candidate_records: Sequence[PredictionRecord],
    comparator_records: Sequence[PredictionRecord],
    labels: Sequence[str],
    pii_free_eligible_doc_ids: Sequence[str],
    reporting_slices_by_doc_id: Mapping[str, Sequence[str]] | None = None,
) -> ThreeMetricPairedSuite:
    """Align two systems and discard everything except sufficient statistics."""

    suite_name = _normalized_identifier(name, field="suite name")
    protocol_labels = _protocol_labels(labels)
    label_set = frozenset(protocol_labels)

    gold_by_id: dict[str, GoldRecord] = {}
    for record in gold_records:
        record.validate()
        if record.doc_id != record.doc_id.strip():
            raise EvaluationDataError("three-metric document IDs must be normalized strings")
        if record.doc_id in gold_by_id:
            raise EvaluationDataError("duplicate gold doc_id in three-metric comparison")
        gold_labels = {span.label for span in record.spans}
        if gold_labels - label_set:
            raise EvaluationDataError(
                "gold contains labels outside the three-metric comparison protocol"
            )
        gold_by_id[record.doc_id] = record
    if not gold_by_id:
        raise EvaluationDataError("three-metric comparison requires at least one gold document")
    if isinstance(pii_free_eligible_doc_ids, (str, bytes)):
        raise EvaluationDataError("PII-free eligible doc_ids must be a sequence, not a string")
    pii_free_ids = tuple(pii_free_eligible_doc_ids)
    if any(
        not isinstance(doc_id, str) or not doc_id or doc_id != doc_id.strip()
        for doc_id in pii_free_ids
    ):
        raise EvaluationDataError("PII-free eligible doc_ids must be non-empty normalized strings")
    if len(pii_free_ids) != len(set(pii_free_ids)):
        raise EvaluationDataError("PII-free eligible doc_ids must be unique")
    if not set(pii_free_ids) <= set(gold_by_id):
        raise EvaluationDataError("PII-free eligible doc_ids must be a subset of gold documents")
    for doc_id in pii_free_ids:
        if gold_by_id[doc_id].spans:
            raise EvaluationDataError(
                "PII-free eligible documents must not contain protocol gold spans"
            )
    pii_free_id_set = frozenset(pii_free_ids)

    if reporting_slices_by_doc_id is None:
        slices_by_id = {doc_id: (suite_name,) for doc_id in gold_by_id}
    else:
        if not isinstance(reporting_slices_by_doc_id, Mapping) or set(
            reporting_slices_by_doc_id
        ) != set(gold_by_id):
            raise EvaluationDataError(
                "reporting slice membership must cover exactly the gold documents"
            )
        try:
            slices_by_id = {
                doc_id: _reporting_slices(reporting_slices_by_doc_id[doc_id])
                for doc_id in gold_by_id
            }
        except ValueError as exc:
            raise EvaluationDataError("reporting slice membership is invalid") from exc

    def prediction_map(
        records: Sequence[PredictionRecord], *, kind: str
    ) -> dict[str, PredictionRecord]:
        result: dict[str, PredictionRecord] = {}
        for record in records:
            record.validate()
            if record.doc_id in result:
                raise EvaluationDataError(f"duplicate {kind} doc_id in three-metric comparison")
            result[record.doc_id] = record
        if set(result) != set(gold_by_id):
            raise EvaluationDataError(
                f"{kind} predictions must contain exactly the three-metric gold documents"
            )
        return result

    candidate_by_id = prediction_map(candidate_records, kind="candidate")
    comparator_by_id = prediction_map(comparator_records, kind="comparator")
    ordered_doc_ids = tuple(sorted(gold_by_id))
    documents: list[ThreeMetricPairedDocument] = []
    for doc_id in ordered_doc_ids:
        gold = gold_by_id[doc_id]
        gold_keys = {span.metric_key() for span in gold.spans}
        pii_free_eligible = doc_id in pii_free_id_set
        documents.append(
            ThreeMetricPairedDocument(
                document_key_sha256=_document_key_sha256(doc_id),
                gold_keys_sha256=_span_keys_sha256(
                    domain="pii-zh-three-metric-gold-keys-v1",
                    doc_id=doc_id,
                    keys=gold_keys,
                ),
                reporting_slices=slices_by_id[doc_id],
                pii_free_eligible=pii_free_eligible,
                candidate=_system_document(
                    doc_id=doc_id,
                    gold_keys=gold_keys,
                    prediction=candidate_by_id[doc_id],
                    text_length=gold.text_length,
                    labels=protocol_labels,
                    pii_free_eligible=pii_free_eligible,
                ),
                comparator=_system_document(
                    doc_id=doc_id,
                    gold_keys=gold_keys,
                    prediction=comparator_by_id[doc_id],
                    text_length=gold.text_length,
                    labels=protocol_labels,
                    pii_free_eligible=pii_free_eligible,
                ),
            )
        )
    alignment_sha256 = _suite_alignment_sha256(
        name=suite_name,
        labels=protocol_labels,
        documents=documents,
    )
    suite = ThreeMetricPairedSuite(
        name=suite_name,
        labels=protocol_labels,
        alignment_sha256=alignment_sha256,
        documents=tuple(documents),
    )
    _validate_suite(suite)
    return suite


def _validate_counts(value: object, *, field: str) -> StrictCounts:
    if not isinstance(value, StrictCounts):
        raise ValueError(f"{field} must be StrictCounts")
    for name in ("tp", "fp", "fn"):
        count = getattr(value, name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field}.{name} must be a non-negative integer")
    return value


def _sum_strict_counts(values: Sequence[StrictCounts]) -> StrictCounts:
    return StrictCounts(
        tp=sum(value.tp for value in values),
        fp=sum(value.fp for value in values),
        fn=sum(value.fn for value in values),
    )


def _validate_system_document(
    value: object,
    *,
    label_count: int,
    pii_free_eligible: bool,
    field: str,
) -> ThreeMetricSystemDocument:
    if not isinstance(value, ThreeMetricSystemDocument):
        raise ValueError(f"{field} must be ThreeMetricSystemDocument")
    if (
        not isinstance(value.prediction_keys_sha256, str)
        or _SHA256_RE.fullmatch(value.prediction_keys_sha256) is None
    ):
        raise ValueError(f"{field}.prediction_keys_sha256 is invalid")
    micro = _validate_counts(value.strict_micro, field=f"{field}.strict_micro")
    if not isinstance(value.strict_per_label, tuple) or len(value.strict_per_label) != label_count:
        raise ValueError(f"{field}.strict_per_label does not match protocol labels")
    per_label = tuple(
        _validate_counts(item, field=f"{field}.strict_per_label[{index}]")
        for index, item in enumerate(value.strict_per_label)
    )
    if _sum_strict_counts(per_label) != micro:
        raise ValueError(f"{field} micro and per-label counts disagree")
    if not isinstance(value.pii_free_false_positive, bool):
        raise ValueError(f"{field}.pii_free_false_positive must be boolean")
    if pii_free_eligible:
        if micro.tp != 0 or micro.fn != 0:
            raise ValueError(f"{field} has gold support in a PII-free document")
        if value.pii_free_false_positive != (micro.fp > 0):
            raise ValueError(f"{field} PII-free false-positive indicator is inconsistent")
    elif value.pii_free_false_positive:
        raise ValueError(f"{field} sets a PII-free indicator on an FPR-ineligible document")
    return value


def _validate_suite(value: object) -> ThreeMetricPairedSuite:
    if not isinstance(value, ThreeMetricPairedSuite):
        raise ValueError("comparisons must contain ThreeMetricPairedSuite values")
    _normalized_identifier(value.name, field="suite name")
    labels = _protocol_labels(value.labels)
    if value.labels != labels:
        raise ValueError("suite labels must be a normalized tuple")
    if (
        not isinstance(value.alignment_sha256, str)
        or _SHA256_RE.fullmatch(value.alignment_sha256) is None
    ):
        raise ValueError("suite alignment SHA-256 is invalid")
    if not isinstance(value.documents, tuple) or not value.documents:
        raise ValueError("three-metric suites must contain documents")
    for index, document in enumerate(value.documents):
        if not isinstance(document, ThreeMetricPairedDocument):
            raise ValueError("suite documents must be ThreeMetricPairedDocument values")
        if (
            not isinstance(document.document_key_sha256, str)
            or _SHA256_RE.fullmatch(document.document_key_sha256) is None
        ):
            raise ValueError("document alignment key SHA-256 is invalid")
        if (
            not isinstance(document.gold_keys_sha256, str)
            or _SHA256_RE.fullmatch(document.gold_keys_sha256) is None
        ):
            raise ValueError("gold span-key SHA-256 is invalid")
        slices = _reporting_slices(document.reporting_slices)
        if document.reporting_slices != slices:
            raise ValueError("document reporting slices must be a canonical tuple")
        if not isinstance(document.pii_free_eligible, bool):
            raise ValueError("document.pii_free_eligible must be boolean")
        candidate = _validate_system_document(
            document.candidate,
            label_count=len(labels),
            pii_free_eligible=document.pii_free_eligible,
            field=f"documents[{index}].candidate",
        )
        comparator = _validate_system_document(
            document.comparator,
            label_count=len(labels),
            pii_free_eligible=document.pii_free_eligible,
            field=f"documents[{index}].comparator",
        )
        if candidate.strict_micro.tp + candidate.strict_micro.fn != (
            comparator.strict_micro.tp + comparator.strict_micro.fn
        ):
            raise ValueError("candidate and comparator gold support disagree")
        for candidate_label, comparator_label in zip(
            candidate.strict_per_label,
            comparator.strict_per_label,
            strict=True,
        ):
            if candidate_label.tp + candidate_label.fn != (
                comparator_label.tp + comparator_label.fn
            ):
                raise ValueError("candidate and comparator per-label gold support disagree")
        gold_support = candidate.strict_micro.tp + candidate.strict_micro.fn
        if document.pii_free_eligible and gold_support != 0:
            raise ValueError("PII-free eligible document contains protocol gold support")
    reporting_signatures = {document.reporting_slices for document in value.documents}
    if len(reporting_signatures) != 1:
        raise ValueError(
            "each mutually exclusive resampling suite must have one reporting-slice signature"
        )
    expected_alignment_sha256 = _suite_alignment_sha256(
        name=value.name,
        labels=value.labels,
        documents=value.documents,
    )
    if value.alignment_sha256 != expected_alignment_sha256:
        raise ValueError("suite alignment SHA-256 does not match suite contents")
    return value


def _validate_bootstrap_parameters(
    *,
    samples: int,
    confidence: float,
    seed: int,
    fpr_noninferiority_margin: float,
    fpr_absolute_maximum: float,
) -> None:
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if (
        type(confidence) not in {int, float}
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be finite and in (0, 1)")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for value, field in (
        (fpr_noninferiority_margin, "fpr_noninferiority_margin"),
        (fpr_absolute_maximum, "fpr_absolute_maximum"),
    ):
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{field} must be finite and in [0, 1]")


def _validate_comparisons(
    comparisons: Mapping[str, Sequence[ThreeMetricPairedSuite]],
    *,
    candidate_id: str,
    admitted_comparator_ids: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[ThreeMetricPairedSuite, ...]]]:
    normalized_candidate = _normalized_identifier(candidate_id, field="candidate_id")
    if isinstance(admitted_comparator_ids, (str, bytes)):
        raise ValueError("admitted_comparator_ids must be a sequence, not a string")
    supplied_admitted = tuple(admitted_comparator_ids)
    if not supplied_admitted:
        raise ValueError("admitted_comparator_ids must be non-empty")
    for comparator_id in supplied_admitted:
        _normalized_identifier(comparator_id, field="comparator_id")
        if comparator_id == normalized_candidate:
            raise ValueError("candidate_id cannot be an admitted comparator_id")
    if len(supplied_admitted) != len(set(supplied_admitted)):
        raise ValueError("admitted_comparator_ids must be unique")
    admitted = tuple(sorted(supplied_admitted))
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(admitted):
        raise ValueError("comparisons must exactly match the admitted comparator universe")

    normalized: dict[str, tuple[ThreeMetricPairedSuite, ...]] = {}
    for comparator_id in admitted:
        if isinstance(comparisons[comparator_id], (str, bytes)):
            raise ValueError("comparator suites must be a sequence, not a string")
        suites = tuple(comparisons[comparator_id])
        if not suites:
            raise ValueError("each comparator must contain at least one suite")
        validated = tuple(
            sorted(
                (_validate_suite(suite) for suite in suites),
                key=lambda suite: suite.name,
            )
        )
        names = tuple(suite.name for suite in validated)
        if len(names) != len(set(names)):
            raise ValueError("suite names must be unique within each comparator")
        reporting_signatures = tuple(suite.documents[0].reporting_slices for suite in validated)
        if len(reporting_signatures) != len(set(reporting_signatures)):
            raise ValueError("reporting-slice signatures must uniquely identify resampling suites")
        if any(suite.labels != validated[0].labels for suite in validated):
            raise ValueError("all suites must use the same ordered protocol labels")
        normalized[comparator_id] = validated

    reference = normalized[admitted[0]]
    for comparator_id in admitted[1:]:
        current = normalized[comparator_id]
        if len(current) != len(reference):
            raise ValueError("every comparator must contain the same suites")
        for expected, observed in zip(reference, current, strict=True):
            if (
                observed.name != expected.name
                or observed.labels != expected.labels
                or observed.alignment_sha256 != expected.alignment_sha256
                or len(observed.documents) != len(expected.documents)
            ):
                raise ValueError("comparator suite alignment or protocol changed")
            for expected_doc, observed_doc in zip(
                expected.documents,
                observed.documents,
                strict=True,
            ):
                if (
                    observed_doc.document_key_sha256 != expected_doc.document_key_sha256
                    or observed_doc.gold_keys_sha256 != expected_doc.gold_keys_sha256
                    or observed_doc.reporting_slices != expected_doc.reporting_slices
                    or observed_doc.pii_free_eligible != expected_doc.pii_free_eligible
                    or observed_doc.candidate != expected_doc.candidate
                ):
                    raise ValueError(
                        "candidate/gold document statistics changed across comparators"
                    )

    documents = [document for suite in reference for document in suite.documents]
    document_keys = [document.document_key_sha256 for document in documents]
    if len(document_keys) != len(set(document_keys)):
        raise ValueError(
            "resampling suites must be mutually exclusive; overlapping document keys found"
        )
    return admitted, normalized


def _reporting_slice_names(
    suites: Sequence[ThreeMetricPairedSuite],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                reporting_slice
                for suite in suites
                for document in suite.documents
                for reporting_slice in document.reporting_slices
            }
        )
    )


def _normalize_required_metrics(
    required_metrics_by_reporting_slice: Mapping[str, Sequence[str]] | None,
    *,
    suites: Sequence[ThreeMetricPairedSuite],
) -> tuple[dict[str, tuple[str, ...]], str]:
    if required_metrics_by_reporting_slice is None:
        return {_POOLED_SCOPE: _METRICS}, "default_pooled_only_diagnostic"
    expected_scopes = {_POOLED_SCOPE, *_reporting_slice_names(suites)}
    if (
        not isinstance(required_metrics_by_reporting_slice, Mapping)
        or set(required_metrics_by_reporting_slice) != expected_scopes
    ):
        raise ValueError("required metrics must cover pooled and every reporting slice exactly")
    normalized: dict[str, tuple[str, ...]] = {}
    for scope in sorted(expected_scopes):
        values = required_metrics_by_reporting_slice[scope]
        if isinstance(values, (str, bytes)):
            raise ValueError("required metrics must be a sequence, not a string")
        metrics = tuple(values)
        if (
            not metrics
            or any(not isinstance(metric, str) for metric in metrics)
            or len(metrics) != len(set(metrics))
            or set(metrics) - set(_METRICS)
        ):
            raise ValueError("required metrics contain an invalid or duplicate metric")
        normalized[scope] = tuple(metric for metric in _METRICS if metric in metrics)
    if normalized[_POOLED_SCOPE] != _METRICS:
        raise ValueError("pooled reporting scope must require all three metrics")
    return normalized, "explicit_runtime_mapping_requires_successor_preregistration_binding"


def _available_metrics(values: _MetricValues) -> tuple[str, ...]:
    available: list[str] = []
    if values.protocol_gold_spans > 0:
        available.extend(("strict_micro_f1", "strict_macro_f1"))
    if values.pii_free_documents > 0:
        available.append("pii_free_document_fpr")
    return tuple(available)


def _validate_required_metric_support(
    required: Mapping[str, Sequence[str]],
    *,
    point_values: Mapping[str, _MetricValues],
) -> None:
    for scope, metrics in required.items():
        values = point_values[scope]
        for metric in metrics:
            if metric in {"strict_micro_f1", "strict_macro_f1"} and (
                values.protocol_gold_spans < 1
            ):
                raise ValueError(
                    f"strict F1 requires at least one protocol gold span in scope {scope}"
                )
            if metric == "pii_free_document_fpr" and values.pii_free_documents < 1:
                raise ValueError(
                    "PII-free document FPR requires at least one PII-free document "
                    f"in scope {scope}"
                )


def _resampling_group_counts(suite: ThreeMetricPairedSuite) -> dict[str, int]:
    return {
        "fpr_ineligible_without_protocol_gold": sum(
            not document.pii_free_eligible
            and document.candidate.strict_micro.tp + document.candidate.strict_micro.fn == 0
            for document in suite.documents
        ),
        "fpr_ineligible_with_protocol_gold": sum(
            not document.pii_free_eligible
            and document.candidate.strict_micro.tp + document.candidate.strict_micro.fn > 0
            for document in suite.documents
        ),
        "pii_free_eligible_without_protocol_gold": sum(
            document.pii_free_eligible for document in suite.documents
        ),
    }


def _resampling_group_quality(
    suites: Sequence[ThreeMetricPairedSuite],
) -> dict[str, int | bool | None]:
    nonempty_sizes = [
        count for suite in suites for count in _resampling_group_counts(suite).values() if count > 0
    ]
    singleton_count = sum(size == 1 for size in nonempty_sizes)
    return {
        "nonempty_group_count": len(nonempty_sizes),
        "minimum_nonempty_group_size": min(nonempty_sizes, default=None),
        "singleton_group_count": singleton_count,
        "no_singleton_groups": singleton_count == 0,
    }


def _counts_payload(counts: StrictCounts) -> dict[str, int]:
    return {"fn": counts.fn, "fp": counts.fp, "tp": counts.tp}


def _comparator_input_sha256(suites: Sequence[ThreeMetricPairedSuite]) -> str:
    return _canonical_sha256(
        {
            "domain": "pii-zh-three-metric-comparator-input-v1",
            "suites": [
                {
                    "alignment_sha256": suite.alignment_sha256,
                    "name": suite.name,
                    "documents": [
                        {
                            "document_key_sha256": document.document_key_sha256,
                            "prediction_keys_sha256": (document.comparator.prediction_keys_sha256),
                            "strict_micro": _counts_payload(document.comparator.strict_micro),
                            "strict_per_label": [
                                _counts_payload(counts)
                                for counts in document.comparator.strict_per_label
                            ],
                            "pii_free_false_positive": (
                                document.comparator.pii_free_false_positive
                            ),
                        }
                        for document in suite.documents
                    ],
                }
                for suite in suites
            ],
        }
    )


def _percentile(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = position.numerator // position.denominator
    upper_index = -(-position.numerator // position.denominator)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _strict_f1_exact(counts: StrictCounts) -> Fraction:
    denominator = 2 * counts.tp + counts.fp + counts.fn
    return Fraction(2 * counts.tp, denominator) if denominator else Fraction(0)


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _stratified_sample_plan(
    suites: Sequence[ThreeMetricPairedSuite], rng: _Sha256CounterRng
) -> tuple[tuple[int, ...], ...]:
    plan: list[tuple[int, ...]] = []
    for suite in suites:
        groups = tuple(
            tuple(
                index
                for index, document in enumerate(suite.documents)
                if document.pii_free_eligible == pii_free_eligible
                and (document.candidate.strict_micro.tp + document.candidate.strict_micro.fn > 0)
                == has_protocol_gold_support
            )
            for pii_free_eligible, has_protocol_gold_support in (
                (False, False),
                (False, True),
                (True, False),
            )
        )
        sampled = tuple(
            source[rng.randbelow(len(source))] for source in groups if source for _ in source
        )
        if len(sampled) != len(suite.documents):
            raise RuntimeError("stratified bootstrap did not preserve suite size")
        plan.append(sampled)
    return tuple(plan)


def _update_plan_hash(
    hasher: Any,
    *,
    sample_index: int,
    plan: Sequence[Sequence[int]],
) -> None:
    hasher.update(sample_index.to_bytes(8, "big"))
    hasher.update(len(plan).to_bytes(4, "big"))
    for indices in plan:
        hasher.update(len(indices).to_bytes(8, "big"))
        for index in indices:
            hasher.update(index.to_bytes(8, "big"))


def _metric_values(
    suites: Sequence[ThreeMetricPairedSuite],
    *,
    side: str,
    plan: Sequence[Sequence[int]] | None,
    reporting_slice: str | None = None,
) -> _MetricValues:
    if side not in {"candidate", "comparator"}:
        raise ValueError("metric side is invalid")
    label_count = len(suites[0].labels)
    micro = StrictCounts(0, 0, 0)
    per_label = [StrictCounts(0, 0, 0) for _ in range(label_count)]
    pii_free_documents = 0
    pii_free_false_positives = 0
    for suite_index, suite in enumerate(suites):
        indices: Sequence[int] = (
            tuple(range(len(suite.documents))) if plan is None else plan[suite_index]
        )
        for index in indices:
            document = suite.documents[index]
            if reporting_slice is not None and reporting_slice not in document.reporting_slices:
                continue
            system = document.candidate if side == "candidate" else document.comparator
            micro += system.strict_micro
            for label_index, counts in enumerate(system.strict_per_label):
                per_label[label_index] += counts
            if document.pii_free_eligible:
                pii_free_documents += 1
                pii_free_false_positives += int(system.pii_free_false_positive)
    protocol_gold_spans = micro.tp + micro.fn
    return _MetricValues(
        strict_micro_f1=_strict_f1_exact(micro),
        strict_macro_f1=(
            sum((_strict_f1_exact(counts) for counts in per_label), start=Fraction(0)) / label_count
        ),
        pii_free_document_fpr=(
            None
            if pii_free_documents == 0
            else Fraction(
                pii_free_false_positives,
                pii_free_documents,
            )
        ),
        protocol_gold_spans=protocol_gold_spans,
        pii_free_documents=pii_free_documents,
    )


def _raw_delta(
    metric: str,
    candidate: _MetricValues,
    comparator: _MetricValues,
) -> Fraction:
    candidate_value = getattr(candidate, metric)
    comparator_value = getattr(comparator, metric)
    if candidate_value is None or comparator_value is None:
        raise ValueError(f"metric {metric} is unavailable for this reporting scope")
    return candidate_value - comparator_value


def _gate_oriented_delta(
    metric: str,
    candidate: _MetricValues,
    comparator: _MetricValues,
    *,
    fpr_noninferiority_margin: float,
) -> Fraction:
    if metric == "pii_free_document_fpr":
        return Fraction(str(fpr_noninferiority_margin)) - _raw_delta(
            metric,
            candidate,
            comparator,
        )
    return _raw_delta(metric, candidate, comparator)


def _metric_report(
    *,
    metric: str,
    candidate: _MetricValues,
    comparator: _MetricValues,
    deltas: Sequence[Fraction],
    confidence: float,
    fpr_noninferiority_margin: float,
) -> dict[str, Any]:
    tail = (Fraction(1) - Fraction(str(confidence))) / 2
    lower = _percentile(deltas, tail)
    upper = _percentile(deltas, Fraction(1) - tail)
    positive = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    negative = len(deltas) - positive - ties
    candidate_value = getattr(candidate, metric)
    comparator_value = getattr(comparator, metric)
    direction = "lower_is_better" if metric == "pii_free_document_fpr" else "higher_is_better"
    gate_definition = (
        "comparator_plus_noninferiority_margin_minus_candidate"
        if metric == "pii_free_document_fpr"
        else "candidate_minus_comparator"
    )
    is_fpr = metric == "pii_free_document_fpr"
    gate_passed = lower >= 0 if is_fpr else lower > 0
    outcomes = (
        {
            "candidate_within_noninferiority_margin": positive + ties,
            "boundary_noninferior": ties,
            "candidate_exceeds_noninferiority_margin": negative,
        }
        if is_fpr
        else {
            "candidate_superior": positive,
            "tie_not_superior": ties,
            "comparator_superior": negative,
        }
    )
    result: dict[str, Any] = {
        "direction": direction,
        "raw_delta_definition": "candidate_minus_comparator",
        "gate_oriented_delta_definition": gate_definition,
        "candidate_point_estimate": float(candidate_value),
        "comparator_point_estimate": float(comparator_value),
        "raw_point_delta_candidate_minus_comparator": float(
            _raw_delta(metric, candidate, comparator)
        ),
        "gate_oriented_point_delta": float(
            _gate_oriented_delta(
                metric,
                candidate,
                comparator,
                fpr_noninferiority_margin=fpr_noninferiority_margin,
            )
        ),
        "gate_oriented_confidence_interval": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "exact_rational": {
            "candidate_point_estimate": _fraction_payload(candidate_value),
            "comparator_point_estimate": _fraction_payload(comparator_value),
            "raw_point_delta_candidate_minus_comparator": _fraction_payload(
                _raw_delta(metric, candidate, comparator)
            ),
            "gate_oriented_point_delta": _fraction_payload(
                _gate_oriented_delta(
                    metric,
                    candidate,
                    comparator,
                    fpr_noninferiority_margin=fpr_noninferiority_margin,
                )
            ),
            "gate_oriented_confidence_interval": {
                "lower": _fraction_payload(lower),
                "upper": _fraction_payload(upper),
            },
        },
        "bootstrap_outcomes": outcomes,
        "pairwise_requirement_passed": gate_passed,
    }
    if is_fpr:
        result.update(
            {
                "noninferiority_margin": fpr_noninferiority_margin,
                "candidate_noninferiority_replicate_rate": (positive + ties) / len(deltas),
                "one_sided_margin_failure_tail_probability_approximation": (negative + 1.0)
                / (len(deltas) + 1.0),
                "boundary_counts_as_noninferior_not_superior": True,
            }
        )
    else:
        result.update(
            {
                "candidate_superiority_rate": positive / len(deltas),
                "one_sided_non_superiority_tail_probability_approximation": (negative + ties + 1.0)
                / (len(deltas) + 1.0),
                "zero_delta_counts_as_superiority": False,
            }
        )
    return result


def three_metric_intersection_union_bootstrap(
    comparisons: Mapping[str, Sequence[ThreeMetricPairedSuite]],
    *,
    candidate_id: str,
    admitted_comparator_ids: Sequence[str],
    samples: int,
    confidence: float,
    seed: int,
    fpr_noninferiority_margin: float,
    fpr_absolute_maximum: float,
    required_metrics_by_reporting_slice: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Evaluate paired metrics across pooled and overlapping reporting slices.

    Strict micro/macro F1 require a strictly positive lower bound for
    candidate-minus-comparator. FPR follows the plan's safety contract instead:
    the candidate point estimate must satisfy an absolute cap and the lower
    bound of ``comparator + margin - candidate`` must be non-negative. A zero
    F1 delta is never superiority; a zero margin-adjusted FPR delta is the
    pre-registered non-inferiority boundary.
    """

    _validate_bootstrap_parameters(
        samples=samples,
        confidence=confidence,
        seed=seed,
        fpr_noninferiority_margin=fpr_noninferiority_margin,
        fpr_absolute_maximum=fpr_absolute_maximum,
    )
    confidence = float(confidence)
    fpr_noninferiority_margin = float(fpr_noninferiority_margin)
    fpr_absolute_maximum = float(fpr_absolute_maximum)
    normalized_candidate = _normalized_identifier(candidate_id, field="candidate_id")
    admitted, normalized = _validate_comparisons(
        comparisons,
        candidate_id=normalized_candidate,
        admitted_comparator_ids=admitted_comparator_ids,
    )
    reference = normalized[admitted[0]]
    required_metrics, required_metrics_source = _normalize_required_metrics(
        required_metrics_by_reporting_slice,
        suites=reference,
    )
    scope_names = (_POOLED_SCOPE, *_reporting_slice_names(reference))

    def values_by_scope(
        suites: Sequence[ThreeMetricPairedSuite],
        *,
        side: str,
        plan: Sequence[Sequence[int]] | None,
    ) -> dict[str, _MetricValues]:
        return {
            scope: _metric_values(
                suites,
                side=side,
                plan=plan,
                reporting_slice=None if scope == _POOLED_SCOPE else scope,
            )
            for scope in scope_names
        }

    point_candidate_by_scope = values_by_scope(
        reference,
        side="candidate",
        plan=None,
    )
    _validate_required_metric_support(
        required_metrics,
        point_values=point_candidate_by_scope,
    )
    applicable_metrics_by_scope = {
        scope: _available_metrics(values) for scope, values in point_candidate_by_scope.items()
    }
    point_comparators_by_scope = {
        comparator_id: values_by_scope(
            normalized[comparator_id],
            side="comparator",
            plan=None,
        )
        for comparator_id in admitted
    }
    for comparator_id in admitted:
        for scope in scope_names:
            if (
                _available_metrics(point_comparators_by_scope[comparator_id][scope])
                != applicable_metrics_by_scope[scope]
            ):
                raise ValueError("candidate and comparator metric applicability changed")

    delta_samples: dict[str, dict[str, dict[str, list[Fraction]]]] = {
        comparator_id: {
            scope: {metric: [] for metric in applicable_metrics_by_scope[scope]}
            for scope in scope_names
        }
        for comparator_id in admitted
    }
    comparator_input_sha256 = {
        comparator_id: _comparator_input_sha256(normalized[comparator_id])
        for comparator_id in admitted
    }
    plan_binding = {
        "algorithm": "sha256_counter_rejection_v1",
        "candidate_id": normalized_candidate,
        "comparator_ids": list(admitted),
        "comparator_input_sha256": comparator_input_sha256,
        "confidence_decimal": str(confidence),
        "fpr_absolute_maximum_decimal": str(fpr_absolute_maximum),
        "fpr_noninferiority_margin_decimal": str(fpr_noninferiority_margin),
        "labels": list(reference[0].labels),
        "required_metrics_by_reporting_scope": {
            scope: list(metrics) for scope, metrics in required_metrics.items()
        },
        "samples": samples,
        "seed": seed,
        "suites": [
            {
                "alignment_sha256": suite.alignment_sha256,
                "documents": len(suite.documents),
                "name": suite.name,
                "reporting_slices": list(suite.documents[0].reporting_slices),
                "resampling_group_counts": _resampling_group_counts(suite),
            }
            for suite in reference
        ],
    }
    plan_binding_sha256 = _canonical_sha256(plan_binding)
    rng = _Sha256CounterRng(seed)
    plan_hasher = hashlib.sha256(
        b"pii-zh-three-metric-bootstrap-plan-v2\0" + bytes.fromhex(plan_binding_sha256)
    )
    for sample_index in range(samples):
        plan = _stratified_sample_plan(reference, rng)
        _update_plan_hash(plan_hasher, sample_index=sample_index, plan=plan)
        sampled_candidate_by_scope = values_by_scope(
            reference,
            side="candidate",
            plan=plan,
        )
        if any(
            _available_metrics(sampled_candidate_by_scope[scope])
            != applicable_metrics_by_scope[scope]
            for scope in scope_names
        ):
            raise RuntimeError("stratified bootstrap changed metric applicability")
        for comparator_id in admitted:
            sampled_comparator_by_scope = values_by_scope(
                normalized[comparator_id],
                side="comparator",
                plan=plan,
            )
            for scope in scope_names:
                for metric in applicable_metrics_by_scope[scope]:
                    delta_samples[comparator_id][scope][metric].append(
                        _gate_oriented_delta(
                            metric,
                            sampled_candidate_by_scope[scope],
                            sampled_comparator_by_scope[scope],
                            fpr_noninferiority_margin=float(fpr_noninferiority_margin),
                        )
                    )

    comparison_reports: dict[str, Any] = {}
    for comparator_id in admitted:
        scope_reports: dict[str, Any] = {}
        for scope in scope_names:
            metric_reports = {
                metric: _metric_report(
                    metric=metric,
                    candidate=point_candidate_by_scope[scope],
                    comparator=point_comparators_by_scope[comparator_id][scope],
                    deltas=delta_samples[comparator_id][scope][metric],
                    confidence=float(confidence),
                    fpr_noninferiority_margin=float(fpr_noninferiority_margin),
                )
                for metric in applicable_metrics_by_scope[scope]
            }
            scope_required_metrics = required_metrics.get(scope, ())
            metric_requirements_passed = all(
                metric_reports[metric]["pairwise_requirement_passed"]
                for metric in scope_required_metrics
            )
            fpr_cap_required = "pii_free_document_fpr" in scope_required_metrics
            fpr_value = point_candidate_by_scope[scope].pii_free_document_fpr
            fpr_cap_passed = (
                None
                if not fpr_cap_required
                else fpr_value is not None and fpr_value <= Fraction(str(fpr_absolute_maximum))
            )
            scope_reports[scope] = {
                "candidate_point_estimates": point_candidate_by_scope[scope].to_dict(),
                "comparator_point_estimates": point_comparators_by_scope[comparator_id][
                    scope
                ].to_dict(),
                "support": {
                    "pii_free_documents": point_candidate_by_scope[scope].pii_free_documents,
                    "protocol_gold_spans": point_candidate_by_scope[scope].protocol_gold_spans,
                },
                "applicable_metrics": list(applicable_metrics_by_scope[scope]),
                "required_metrics": list(scope_required_metrics),
                "metrics": metric_reports,
                "required_metric_requirements_passed": metric_requirements_passed,
                "fpr_absolute_cap_required": fpr_cap_required,
                "fpr_absolute_cap_passed": fpr_cap_passed,
                "scope_requirement_passed": (
                    metric_requirements_passed and (fpr_cap_passed is not False)
                ),
            }
        pooled_report = scope_reports[_POOLED_SCOPE]
        comparison_reports[comparator_id] = {
            "comparator_input_sha256": comparator_input_sha256[comparator_id],
            "candidate_point_estimates": pooled_report["candidate_point_estimates"],
            "comparator_point_estimates": pooled_report["comparator_point_estimates"],
            "metrics": pooled_report["metrics"],
            "all_three_metric_pairwise_requirements_passed": (
                set(_METRICS) <= set(pooled_report["metrics"])
                and all(
                    pooled_report["metrics"][metric]["pairwise_requirement_passed"]
                    for metric in _METRICS
                )
            ),
            "reporting_scopes": scope_reports,
            "all_required_scope_metric_requirements_passed": all(
                report["required_metric_requirements_passed"]
                for scope, report in scope_reports.items()
                if scope in required_metrics
            ),
            "all_required_scope_requirements_passed": all(
                report["scope_requirement_passed"]
                for scope, report in scope_reports.items()
                if scope in required_metrics
            ),
        }

    frozen_statistical_parameters_match = (
        samples == FROZEN_SAMPLES
        and float(confidence) == FROZEN_CONFIDENCE
        and seed == FROZEN_SEED
        and float(fpr_noninferiority_margin) == FROZEN_FPR_NONINFERIORITY_MARGIN
        and float(fpr_absolute_maximum) == FROZEN_FPR_ABSOLUTE_MAXIMUM
    )
    all_pairwise_requirements = all(
        value["all_required_scope_metric_requirements_passed"]
        for value in comparison_reports.values()
    )
    required_fpr_scopes = tuple(
        scope for scope, metrics in required_metrics.items() if "pii_free_document_fpr" in metrics
    )
    fpr_cap_by_scope = {
        scope: (
            point_candidate_by_scope[scope].pii_free_document_fpr is not None
            and point_candidate_by_scope[scope].pii_free_document_fpr
            <= Fraction(str(fpr_absolute_maximum))
        )
        for scope in required_fpr_scopes
    }
    absolute_fpr_passed = all(fpr_cap_by_scope.values())
    resampling_group_quality = _resampling_group_quality(reference)
    unbound_runtime_diagnostic_passed = (
        frozen_statistical_parameters_match
        and all_pairwise_requirements
        and absolute_fpr_passed
        and resampling_group_quality["no_singleton_groups"] is True
    )
    pooled_fpr = point_candidate_by_scope[_POOLED_SCOPE].pii_free_document_fpr
    return {
        "schema_version": 2,
        "method": "paired_atomic_stratified_document_bootstrap_intersection_union",
        "candidate_id": normalized_candidate,
        "runtime_supplied_admitted_comparator_ids": list(admitted),
        "evaluated_comparator_ids": list(admitted),
        "runtime_comparison_keys_match_supplied_ids": True,
        "successor_preregistered_comparator_universe_binding_present": False,
        "protocol_labels": list(reference[0].labels),
        "documents": sum(len(suite.documents) for suite in reference),
        "suite_document_counts": {suite.name: len(suite.documents) for suite in reference},
        "suite_pii_free_document_counts": {
            suite.name: sum(document.pii_free_eligible for document in suite.documents)
            for suite in reference
        },
        "suite_alignment_sha256": {suite.name: suite.alignment_sha256 for suite in reference},
        "resampling_group_counts": {
            suite.name: _resampling_group_counts(suite) for suite in reference
        },
        "resampling_group_quality": resampling_group_quality,
        "reporting_slice_document_counts": {
            scope: sum(
                scope in document.reporting_slices
                for suite in reference
                for document in suite.documents
            )
            for scope in _reporting_slice_names(reference)
        },
        "resampling_unit": "document",
        "resampling_strata": (
            "mutually_exclusive_suite_x_reporting_signature_x_"
            "pii_free_eligibility_x_protocol_gold_support"
        ),
        "overlapping_reporting_slices_are_resampling_strata": False,
        "candidate_comparator_pairing": "preserved",
        "suite_mixture": "preserved_exact_document_counts",
        "pii_free_and_gold_support_strata": "preserved_exact_within_each_suite",
        "metrics": list(_METRICS),
        "macro_label_policy": "all_ordered_protocol_labels_including_zero_support",
        "fpr_denominator_policy": "explicit_protocol_pii_free_eligible_documents_only",
        "required_metrics_by_reporting_scope": {
            scope: list(metrics) for scope, metrics in required_metrics.items()
        },
        "required_metrics_source": required_metrics_source,
        "runtime_required_metrics_mapping_preregistered": False,
        "tie_handling": {
            "strict_micro_f1": "zero_delta_is_not_superiority",
            "strict_macro_f1": "zero_delta_is_not_superiority",
            "pii_free_document_fpr": "zero_margin_adjusted_delta_is_noninferior_not_superior",
        },
        "confidence_interval_method": "percentile",
        "percentile_interpolation": "linear_type_7",
        "metric_arithmetic": "exact_rational_until_json_float_projection",
        "prng": "sha256_counter_rejection_v1",
        "resample_plan_binding_sha256": plan_binding_sha256,
        "shared_resample_plan_sha256": plan_hasher.hexdigest(),
        "effective_samples": samples,
        "confidence": float(confidence),
        "seed": seed,
        "frozen_statistical_parameters": {
            "samples": FROZEN_SAMPLES,
            "confidence": FROZEN_CONFIDENCE,
            "seed": FROZEN_SEED,
            "fpr_noninferiority_margin": FROZEN_FPR_NONINFERIORITY_MARGIN,
            "fpr_absolute_maximum": FROZEN_FPR_ABSOLUTE_MAXIMUM,
            "matches": frozen_statistical_parameters_match,
        },
        "fpr_safety_gate": {
            "candidate_pooled_point_estimate": (None if pooled_fpr is None else float(pooled_fpr)),
            "absolute_maximum": float(fpr_absolute_maximum),
            "absolute_maximum_passed": absolute_fpr_passed,
            "noninferiority_margin": float(fpr_noninferiority_margin),
            "required_fpr_scopes": list(required_fpr_scopes),
            "absolute_maximum_passed_by_scope": fpr_cap_by_scope,
            "absolute_cap_applies_to": "candidate_point_estimate_not_ci_upper",
        },
        "comparisons": comparison_reports,
        "intersection_union": {
            "rule": (
                "micro_macro_strict_superiority_and_fpr_noninferiority_against_"
                "every_admitted_comparator_plus_absolute_fpr_cap"
            ),
            "all_pairwise_metric_requirements_passed": all_pairwise_requirements,
            "absolute_fpr_requirement_passed": absolute_fpr_passed,
            "unbound_runtime_diagnostic_passed": unbound_runtime_diagnostic_passed,
            "unbound_runtime_diagnostic_is_claim_evidence": False,
            "nondegenerate_resampling_groups_required": True,
            "formal_statistical_component_passed": False,
            "formal_gate_blocker": (
                "successor registry has not bound Open-24, the exact comparator universe, "
                "candidate identity, eligibility manifest, the exact allowed reporting-slice "
                "universe, an outcome-independent atomic-partition manifest, and required "
                "reporting metrics"
            ),
        },
        "claim_activation": {
            "allowed": False,
            "reason": "statistical_component_alone_cannot_activate_first_best_or_sota_claims",
        },
        "privacy": {
            "raw_text_retained": False,
            "raw_doc_ids_retained": False,
            "per_document_alignment_hashes_output": False,
            "entity_values_retained": False,
            "document_sufficient_statistics_only": True,
        },
    }


def frozen_three_metric_intersection_union_bootstrap(
    comparisons: Mapping[str, Sequence[ThreeMetricPairedSuite]],
    *,
    candidate_id: str,
    admitted_comparator_ids: Sequence[str],
    required_metrics_by_reporting_slice: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Run frozen numeric parameters without activating a formal protocol claim."""

    return three_metric_intersection_union_bootstrap(
        comparisons,
        candidate_id=candidate_id,
        admitted_comparator_ids=admitted_comparator_ids,
        samples=FROZEN_SAMPLES,
        confidence=FROZEN_CONFIDENCE,
        seed=FROZEN_SEED,
        fpr_noninferiority_margin=FROZEN_FPR_NONINFERIORITY_MARGIN,
        fpr_absolute_maximum=FROZEN_FPR_ABSOLUTE_MAXIMUM,
        required_metrics_by_reporting_slice=required_metrics_by_reporting_slice,
    )


__all__ = [
    "FROZEN_CONFIDENCE",
    "FROZEN_FPR_ABSOLUTE_MAXIMUM",
    "FROZEN_FPR_NONINFERIORITY_MARGIN",
    "FROZEN_SAMPLES",
    "FROZEN_SEED",
    "ThreeMetricPairedDocument",
    "ThreeMetricPairedSuite",
    "ThreeMetricSystemDocument",
    "build_three_metric_paired_suite",
    "frozen_three_metric_intersection_union_bootstrap",
    "three_metric_intersection_union_bootstrap",
]
