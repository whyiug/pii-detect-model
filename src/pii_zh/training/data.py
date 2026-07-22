"""Document-level JSONL loading, token alignment, weighting, and dynamic padding."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError("Training data utilities require PyTorch.") from exc

from pii_zh.data.alignment import AlignmentError, align_document_to_bio
from pii_zh.data.schema import DataPool, SchemaError, iter_jsonl
from pii_zh.models.qwen3_jpt import build_jpt_inputs
from pii_zh.taxonomy import Taxonomy


class TrainingDataError(ValueError):
    """Raised without echoing raw source text or entity values."""


@dataclass(frozen=True, slots=True)
class GoldSpan:
    start: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class EncodedDocument:
    doc_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    gold_spans: tuple[GoldSpan, ...]


@dataclass(frozen=True, slots=True)
class TrainingDataSummary:
    document_count: int
    entity_count: int
    token_count: int
    pii_free_document_count: int
    unalignable_boundary_count: int
    label_counts: dict[str, int]
    split_counts: dict[str, int] = field(default_factory=dict)
    data_pool_counts: dict[str, int] = field(default_factory=dict)
    quality_tier_counts: dict[str, int] = field(default_factory=dict)
    quality_gate_passed_document_count: int = 0
    validators_passed_document_count: int = 0
    public_weight_training_allowed_document_count: int = 0
    sources: tuple[TrainingSourceSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingSourceSummary:
    """Non-content provenance and quality counts for one immutable source."""

    source_id: str
    source_kind: str
    source_revision: str
    license: str
    data_pool: str
    split: str
    document_count: int
    entity_count: int
    quality_tier_counts: dict[str, int]
    quality_gate_passed_document_count: int
    validators_passed_document_count: int
    public_weight_training_allowed_document_count: int


def build_bio_label_maps(
    taxonomy: Taxonomy,
    *,
    include_extended: bool = False,
    include_auxiliary: bool = False,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build stable O/B/I maps from taxonomy file order."""

    entities = list(taxonomy.label_sets["core"])
    if include_extended:
        entities.extend(taxonomy.label_sets["extended"])
    if include_auxiliary:
        entities.extend(taxonomy.label_sets["auxiliary"])
    labels = ["O"]
    for entity in entities:
        labels.extend((f"B-{entity.name}", f"I-{entity.name}"))
    label2id = {label: index for index, label in enumerate(labels)}
    return label2id, {index: label for label, index in label2id.items()}


class DocumentTokenDataset(Dataset[dict[str, list[int]]]):
    """One dataset item per source document; raw text is never retained."""

    def __init__(
        self,
        documents: list[EncodedDocument],
        *,
        leakage_fingerprints: dict[str, frozenset[str]] | None = None,
    ) -> None:
        if not documents:
            raise TrainingDataError("token dataset must contain at least one document")
        self.documents = tuple(documents)
        self.leakage_fingerprints = dict(leakage_fingerprints or {})

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.documents[index]
        return {
            "input_ids": list(item.input_ids),
            "attention_mask": list(item.attention_mask),
            "labels": list(item.labels),
        }


def _safe_record_token(index: int, doc_id: str) -> str:
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:12]
    return f"row={index},doc_hash={digest}"


def _leakage_fingerprint(group_type: str, value: str) -> str:
    payload = f"pii-zh-split-v1\x00{group_type}\x00{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _record_leakage_values(record: Any) -> dict[str, tuple[str, ...]]:
    source_groups: list[str] = []
    if record.tenant_group:
        source_groups.append(f"tenant:{record.tenant_group}")
    if record.conversation_group:
        source_groups.append(f"conversation:{record.conversation_group}")
    metadata_source_group = record.metadata.get("source_group")
    if metadata_source_group is not None:
        if not isinstance(metadata_source_group, str) or not metadata_source_group:
            raise TrainingDataError("source-group metadata must be a non-empty string")
        source_groups.append(f"metadata:{metadata_source_group}")
    return {
        "document_id": (record.doc_id,),
        "template_group": (record.template_group,) if record.template_group else (),
        "entity_value_group": tuple(record.entity_value_groups),
        "source_group": tuple(source_groups),
    }


def assert_disjoint_training_splits(
    train_dataset: DocumentTokenDataset,
    validation_dataset: DocumentTokenDataset,
) -> None:
    """Fail closed on identity/group overlap without exposing source values."""

    collisions = {
        group_type: len(
            train_dataset.leakage_fingerprints.get(group_type, frozenset())
            & validation_dataset.leakage_fingerprints.get(group_type, frozenset())
        )
        for group_type in (
            "document_id",
            "template_group",
            "entity_value_group",
            "source_group",
        )
    }
    collisions = {name: count for name, count in collisions.items() if count}
    if collisions:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(collisions.items()))
        raise TrainingDataError(f"cross-split leakage detected ({summary})")


def load_aligned_jsonl(
    path: str | Path,
    *,
    tokenizer: Any,
    label2id: dict[str, int],
    max_length: int,
    expected_split: str,
    allowed_pools: frozenset[DataPool] = frozenset(
        {DataPool.PUBLIC_RELEASE, DataPool.PRIVATE_ENTERPRISE}
    ),
) -> tuple[DocumentTokenDataset, TrainingDataSummary]:
    """Read canonical JSONL and derive a non-truncating BIO token view."""

    if not getattr(tokenizer, "is_fast", False):
        raise TrainingDataError("training requires a fast tokenizer with offset mappings")
    if expected_split not in {"train", "dev", "validation"}:
        raise TrainingDataError("expected_split must be train, dev, or validation")
    documents: list[EncodedDocument] = []
    seen_doc_ids: set[str] = set()
    entity_count = 0
    token_count = 0
    pii_free_count = 0
    unalignable_count = 0
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    data_pool_counts: Counter[str] = Counter()
    quality_tier_counts: Counter[str] = Counter()
    quality_gate_passed = 0
    validators_passed = 0
    public_weight_allowed = 0
    leakage_fingerprints: dict[str, set[str]] = {
        "document_id": set(),
        "template_group": set(),
        "entity_value_group": set(),
        "source_group": set(),
    }
    source_metadata: dict[str, tuple[str, str, str, str, str]] = {}
    source_document_counts: Counter[str] = Counter()
    source_entity_counts: Counter[str] = Counter()
    source_quality_tiers: dict[str, Counter[str]] = {}
    source_quality_gate_passed: Counter[str] = Counter()
    source_validators_passed: Counter[str] = Counter()
    source_public_weight_allowed: Counter[str] = Counter()
    try:
        records = iter_jsonl(path)
        for index, record in enumerate(records, start=1):
            marker = _safe_record_token(index, record.doc_id)
            if record.doc_id in seen_doc_ids:
                raise TrainingDataError(f"duplicate document identifier ({marker})")
            seen_doc_ids.add(record.doc_id)
            if record.split != expected_split:
                raise TrainingDataError(
                    f"record split does not match required {expected_split!r} split ({marker})"
                )
            if record.data_pool not in allowed_pools:
                raise TrainingDataError(
                    "record belongs to forbidden training pool "
                    f"{record.data_pool.value!r} ({marker})"
                )
            try:
                alignment = align_document_to_bio(
                    record,
                    tokenizer,
                    max_length=None,
                    truncation=False,
                    add_special_tokens=True,
                )
                label_ids = alignment.label_ids(label2id)
            except (AlignmentError, KeyError, ValueError) as exc:
                raise TrainingDataError(
                    f"token alignment failed ({marker}, error={type(exc).__name__})"
                ) from None
            if len(alignment.input_ids) > max_length:
                raise TrainingDataError(
                    f"document exceeds safe source token limit {max_length} ({marker}); "
                    "pre-chunk it without splitting entities"
                )

            gold_spans = tuple(
                GoldSpan(start=item.start, end=item.end, label=item.label)
                for item in record.entities
            )
            documents.append(
                EncodedDocument(
                    doc_id=record.doc_id,
                    input_ids=alignment.input_ids,
                    attention_mask=alignment.attention_mask,
                    labels=label_ids,
                    offset_mapping=alignment.offset_mapping,
                    gold_spans=gold_spans,
                )
            )
            entity_count += len(gold_spans)
            token_count += sum(alignment.attention_mask)
            pii_free_count += not gold_spans
            unalignable_count += alignment.statistics.unalignable_count
            label_counts.update(item.label for item in gold_spans)
            split_counts[record.split] += 1
            data_pool_counts[record.data_pool.value] += 1
            quality_tier_counts[record.quality.tier.value] += 1
            quality_gate_passed += int(record.quality.quality_gate)
            validators_passed += int(record.quality.validators_passed)
            public_weight_allowed += int(record.public_weight_training_allowed is True)
            for group_type, values in _record_leakage_values(record).items():
                leakage_fingerprints[group_type].update(
                    _leakage_fingerprint(group_type, value) for value in values
                )

            source_id = record.provenance.source_id
            metadata = (
                record.provenance.source_kind,
                record.provenance.source_revision,
                record.provenance.license,
                record.data_pool.value,
                expected_split,
            )
            previous_metadata = source_metadata.setdefault(source_id, metadata)
            if previous_metadata != metadata:
                raise TrainingDataError(
                    f"a source_id maps to inconsistent provenance within one split ({marker})"
                )
            source_document_counts[source_id] += 1
            source_entity_counts[source_id] += len(gold_spans)
            source_quality_tiers.setdefault(source_id, Counter())[record.quality.tier.value] += 1
            source_quality_gate_passed[source_id] += int(record.quality.quality_gate)
            source_validators_passed[source_id] += int(record.quality.validators_passed)
            source_public_weight_allowed[source_id] += int(
                record.public_weight_training_allowed is True
            )
    except SchemaError:
        raise TrainingDataError(
            "JSONL schema validation failed; inspect the source only in an approved secure context"
        ) from None

    dataset = DocumentTokenDataset(
        documents,
        leakage_fingerprints={
            name: frozenset(values) for name, values in leakage_fingerprints.items()
        },
    )
    sources = tuple(
        TrainingSourceSummary(
            source_id=source_id,
            source_kind=source_metadata[source_id][0],
            source_revision=source_metadata[source_id][1],
            license=source_metadata[source_id][2],
            data_pool=source_metadata[source_id][3],
            split=source_metadata[source_id][4],
            document_count=source_document_counts[source_id],
            entity_count=source_entity_counts[source_id],
            quality_tier_counts=dict(sorted(source_quality_tiers[source_id].items())),
            quality_gate_passed_document_count=source_quality_gate_passed[source_id],
            validators_passed_document_count=source_validators_passed[source_id],
            public_weight_training_allowed_document_count=source_public_weight_allowed[source_id],
        )
        for source_id in sorted(source_metadata)
    )
    return dataset, TrainingDataSummary(
        document_count=len(documents),
        entity_count=entity_count,
        token_count=token_count,
        pii_free_document_count=pii_free_count,
        unalignable_boundary_count=unalignable_count,
        label_counts=dict(sorted(label_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        data_pool_counts=dict(sorted(data_pool_counts.items())),
        quality_tier_counts=dict(sorted(quality_tier_counts.items())),
        quality_gate_passed_document_count=quality_gate_passed,
        validators_passed_document_count=validators_passed,
        public_weight_training_allowed_document_count=public_weight_allowed,
        sources=sources,
    )


def compute_class_weights(
    label_sequences: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    *,
    num_labels: int,
    cap: float = 5.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return ``min(sqrt(N / N_c), cap)`` token-class weights."""

    counts = torch.zeros(num_labels, dtype=torch.long)
    for sequence in label_sequences:
        for label in sequence:
            if label == ignore_index:
                continue
            if label < 0 or label >= num_labels:
                raise TrainingDataError(f"label id {label} is outside [0, {num_labels})")
            counts[label] += 1
    total = int(counts.sum().item())
    if total == 0:
        raise TrainingDataError("cannot compute class weights without supervised tokens")
    weights = torch.ones(num_labels, dtype=torch.float32)
    observed = counts > 0
    weights[observed] = torch.sqrt(total / counts[observed].to(dtype=torch.float32)).clamp(max=cap)
    return weights


def compute_document_sampling_weights(
    documents: tuple[EncodedDocument, ...],
    *,
    cap: float = 5.0,
) -> torch.Tensor:
    """Weight rare positive documents without diluting hard negatives.

    Rare-class weights are normalized to mean one *within positive
    documents*. Negative documents keep weight one, so their aggregate
    sampling probability remains equal to their corpus proportion. This
    satisfies the rare-label objective without silently undoing the configured
    hard-negative ratio.
    """

    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update({span.label for span in document.gold_spans})
    total_documents = len(documents)
    weights: list[float] = []
    positive_indices: list[int] = []
    for document in documents:
        labels = {span.label for span in document.gold_spans}
        if not labels:
            weights.append(1.0)
            continue
        positive_indices.append(len(weights))
        weights.append(
            min(
                cap,
                max((total_documents / document_frequency[label]) ** 0.5 for label in labels),
            )
        )
    if positive_indices:
        positive_mean = sum(weights[index] for index in positive_indices) / len(positive_indices)
        for index in positive_indices:
            weights[index] /= positive_mean
    return torch.tensor(weights, dtype=torch.double)


class DynamicTokenCollator:
    """Right-pad token-classification documents to the longest item in a batch."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        pad_to_multiple_of: int | None = 8,
        fixed_length: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.fixed_length = fixed_length

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise TrainingDataError("cannot collate an empty batch")
        length = max(len(item["input_ids"]) for item in features)
        if self.fixed_length is not None:
            if length > self.fixed_length:
                raise TrainingDataError("batch item exceeds the configured fixed length")
            length = self.fixed_length
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            length = ((length + multiple - 1) // multiple) * multiple

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            item_length = len(item["input_ids"])
            if not (item_length == len(item["attention_mask"]) == len(item["labels"])):
                raise TrainingDataError("input_ids, attention_mask, and labels must align")
            padding = length - item_length
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class JPTTokenCollator:
    """Dynamically pad documents, then construct the repeated JPT sequence."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        sep_token_id: int,
        max_length: int,
    ) -> None:
        self.base_collator = DynamicTokenCollator(
            pad_token_id=pad_token_id,
            pad_to_multiple_of=None,
            fixed_length=(max_length - 1) // 2,
        )
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = self.base_collator(features)
        repeated = build_jpt_inputs(
            cast(torch.LongTensor, batch["input_ids"]),
            attention_mask=batch["attention_mask"],
            labels=cast(torch.LongTensor, batch["labels"]),
            sep_token_id=self.sep_token_id,
            pad_token_id=self.pad_token_id,
            max_length=self.max_length,
        )
        # This mask is useful for tests/debugging, but is not a model argument.
        repeated.pop("second_copy_mask")
        return repeated
