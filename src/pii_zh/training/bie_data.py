"""Strict BIE training views derived from canonical character annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pii_zh.data.alignment import IGNORE_LABEL
from pii_zh.data.schema import DataPool, QualityTier, iter_jsonl
from pii_zh.taxonomy import Taxonomy
from pii_zh.training.data import (
    DocumentTokenDataset,
    EncodedDocument,
    TrainingDataError,
    TrainingDataSummary,
    build_bio_label_maps,
    load_aligned_jsonl,
)


def bio_labels_to_bie_labels(labels: tuple[str | int, ...]) -> tuple[str | int, ...]:
    """Convert a well-formed BIO sequence to AIguard-compatible BIE.

    A one-token entity remains ``B-X`` because the pinned source schema has no
    singleton ``S`` state.  For a multi-token entity the final ``I-X`` becomes
    ``E-X``.  Orphan/mismatched ``I`` tags fail closed.
    """

    converted: list[str | int] = []
    index = 0
    while index < len(labels):
        label = labels[index]
        if label == IGNORE_LABEL or label == "O":
            converted.append(label)
            index += 1
            continue
        if not isinstance(label, str):
            raise TrainingDataError("BIO labels must be strings or the ignore index")
        prefix, separator, entity_type = label.partition("-")
        if separator != "-" or prefix != "B" or not entity_type:
            raise TrainingDataError("BIO sequence contains an orphan or malformed continuation")
        run_end = index + 1
        while run_end < len(labels) and labels[run_end] == f"I-{entity_type}":
            run_end += 1
        converted.append(label)
        continuation_count = run_end - index - 1
        if continuation_count:
            converted.extend(f"I-{entity_type}" for _ in range(continuation_count - 1))
            converted.append(f"E-{entity_type}")
        index = run_end
    return tuple(converted)


class BieDocumentTokenDataset(DocumentTokenDataset):
    """Document dataset with privacy-safe hard-negative row membership."""

    def __init__(
        self,
        documents: list[EncodedDocument],
        *,
        leakage_fingerprints: dict[str, frozenset[str]],
        hard_negative_indices: tuple[int, ...],
    ) -> None:
        super().__init__(documents, leakage_fingerprints=leakage_fingerprints)
        if any(index < 0 or index >= len(documents) for index in hard_negative_indices):
            raise TrainingDataError("hard-negative membership is outside the dataset")
        self.hard_negative_indices = hard_negative_indices


def load_bie_aligned_jsonl(
    path: str | Path,
    *,
    tokenizer: Any,
    taxonomy: Taxonomy,
    bie_label2id: dict[str, int],
    max_length: int,
    expected_split: str,
    allowed_pools: frozenset[DataPool] = frozenset(
        {DataPool.PUBLIC_RELEASE, DataPool.PRIVATE_ENTERPRISE}
    ),
    require_exact_boundaries: bool = True,
) -> tuple[DocumentTokenDataset, TrainingDataSummary]:
    """Load canonical JSONL and supervise every non-special token in BIE space."""

    bio_label2id, bio_id2label = build_bio_label_maps(taxonomy)
    dataset, summary = load_aligned_jsonl(
        path,
        tokenizer=tokenizer,
        label2id=bio_label2id,
        max_length=max_length,
        expected_split=expected_split,
        allowed_pools=allowed_pools,
    )
    if require_exact_boundaries and summary.unalignable_boundary_count:
        raise TrainingDataError("BIE training requires exact tokenizer/entity boundaries")

    documents: list[EncodedDocument] = []
    for document in dataset.documents:
        bio_labels: tuple[str | int, ...] = tuple(
            IGNORE_LABEL if label_id == IGNORE_LABEL else bio_id2label[label_id]
            for label_id in document.labels
        )
        bie_labels = bio_labels_to_bie_labels(bio_labels)
        try:
            bie_ids = tuple(
                IGNORE_LABEL if label == IGNORE_LABEL else bie_label2id[str(label)]
                for label in bie_labels
            )
        except KeyError as exc:
            raise TrainingDataError("BIE label map does not cover the aligned taxonomy") from exc
        if not (
            len(document.input_ids)
            == len(document.attention_mask)
            == len(document.offset_mapping)
            == len(bie_ids)
        ):
            raise TrainingDataError("BIE conversion changed token alignment length")
        documents.append(
            EncodedDocument(
                doc_id=document.doc_id,
                input_ids=document.input_ids,
                attention_mask=document.attention_mask,
                labels=bie_ids,
                offset_mapping=document.offset_mapping,
                gold_spans=document.gold_spans,
            )
        )
    hard_negative_indices: list[int] = []
    for index, record in enumerate(iter_jsonl(path)):
        if index >= len(documents) or record.doc_id != documents[index].doc_id:
            raise TrainingDataError("BIE hard-negative membership changed document order")
        if record.quality.tier is QualityTier.HARD_NEGATIVE:
            if record.entities:
                raise TrainingDataError("a hard-negative document contains a gold entity")
            hard_negative_indices.append(index)
    if len(documents) != summary.document_count:
        raise TrainingDataError("BIE conversion changed the document count")
    return (
        BieDocumentTokenDataset(
            documents,
            leakage_fingerprints=dataset.leakage_fingerprints,
            hard_negative_indices=tuple(hard_negative_indices),
        ),
        summary,
    )


__all__ = [
    "BieDocumentTokenDataset",
    "bio_labels_to_bie_labels",
    "load_bie_aligned_jsonl",
]
