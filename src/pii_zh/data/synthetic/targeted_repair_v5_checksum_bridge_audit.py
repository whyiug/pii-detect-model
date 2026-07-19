"""CPU-only, content-addressed evidence for the v5a checksum rewrite."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..schema import DocumentRecord, iter_jsonl
from .targeted_repair_v5_checksum_bridge import (
    DATASET_ID,
    DATASET_VERSION,
    DEFAULT_RECIPE_PATH,
    SPLITS,
    TargetedRepairV5BridgeError,
    canonical_hash,
    sha256_file,
    validate_materialized,
)

_D0_DATASET_ID = "pii_zh_synthetic_v1"
_D0_DATASET_VERSION = "1.3.0"
_D0_MANIFEST_FILE_SHA256 = "8bb54e0f3d0ccadc420f8c212fc9cdc15a873714d3dd76cb18cb179723523682"
_D0_MANIFEST_SHA256 = "93bcbff0eb3db85c5b568dc0154387115498b573635fac3b51996ac64dbeb3a1"
_D0_FILES = {
    "train": (16000, "6a1c4356ee3de64f4fab7e811bcdc873c21db2a043aa5cbf600626ed0baadbad"),
    "validation": (
        2000,
        "9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d",
    ),
}
_CLEAN_MANIFEST_FILE_SHA256 = "4cc808ba5ae96725cdb080b7b0192a96820588912fe7765b44dc957006c3da06"
_CLEAN_MANIFEST_SHA256 = "1e1ea4e275280f2d85414b6122b58c1d2874340eb4a98cbf99167bbb920c32fe"
_CLEAN_TRAIN_SHA256 = "4243d3302a243ee735686c2e9986935fa82120bbe4d7a4a184c28d4376d343ad"
_CLEAN_TRAIN_RECORDS = 15650
_CLEAN_RECEIPT_FILE_SHA256 = "72f317de7bf57e42466ca18174c8ff1e261c119babc74c1ae5232060dda85353"
_CLEAN_RECEIPT_SHA256 = "ba5f12b0324134c50bc3a87005c9598f01f53a4cb247674c2421229e28db9db1"
_ASCII_SURFACE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9@._:/+*\-]{5,}[A-Za-z0-9])"
    r"(?![A-Za-z0-9])"
)
_CN_DATE_RE = re.compile(r"(?<!\d)((?:19|20|21)\d{2}年\d{1,2}月\d{1,2}日)(?!\d)")
_NEAR_THRESHOLD = 0.92
_D0_TEMPLATE_THRESHOLD = 0.75


class TargetedRepairV5AuditError(ValueError):
    """Raised without retaining or reflecting row-level content."""


def _signed_json(path: Path, logical_key: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TargetedRepairV5AuditError("signed input must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV5AuditError("signed input is unreadable") from exc
    if not isinstance(value, dict):
        raise TargetedRepairV5AuditError("signed input must be an object")
    unsigned = dict(value)
    claimed = unsigned.pop(logical_key, None)
    if not isinstance(claimed, str) or canonical_hash(unsigned) != claimed:
        raise TargetedRepairV5AuditError("signed input logical hash failed")
    return value


def _verify_d0(d0_dataset: Path) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    if d0_dataset.is_symlink() or not d0_dataset.is_dir():
        raise TargetedRepairV5AuditError("D0 dataset directory must be regular")
    manifest_path = d0_dataset / "dataset_manifest.json"
    if sha256_file(manifest_path) != _D0_MANIFEST_FILE_SHA256:
        raise TargetedRepairV5AuditError("D0 manifest file identity changed")
    manifest = _signed_json(manifest_path, "manifest_sha256")
    if (
        manifest.get("manifest_sha256") != _D0_MANIFEST_SHA256
        or manifest.get("dataset_id") != _D0_DATASET_ID
        or manifest.get("dataset_version") != _D0_DATASET_VERSION
    ):
        raise TargetedRepairV5AuditError("D0 manifest logical identity changed")
    result: dict[str, list[DocumentRecord]] = {}
    for split in SPLITS:
        path = d0_dataset / f"{split}.jsonl"
        if path.is_symlink() or not path.is_file() or sha256_file(path) != _D0_FILES[split][1]:
            raise TargetedRepairV5AuditError("D0 development split identity changed")
        records = list(iter_jsonl(path))
        if len(records) != _D0_FILES[split][0] or any(record.split != split for record in records):
            raise TargetedRepairV5AuditError("D0 development split count changed")
        result[split] = records
    return result["train"], result["validation"]


def _verify_clean_view(
    clean_view: Path, clean_receipt: Path
) -> tuple[list[DocumentRecord], dict[str, Any], dict[str, Any]]:
    if clean_view.is_symlink() or not clean_view.is_dir():
        raise TargetedRepairV5AuditError("D0 clean view must be regular")
    manifest_path = clean_view / "selection_manifest.json"
    train_path = clean_view / "train.jsonl"
    if (
        sha256_file(manifest_path) != _CLEAN_MANIFEST_FILE_SHA256
        or sha256_file(train_path) != _CLEAN_TRAIN_SHA256
        or sha256_file(clean_receipt) != _CLEAN_RECEIPT_FILE_SHA256
    ):
        raise TargetedRepairV5AuditError("D0 clean-view file identity changed")
    manifest = _signed_json(manifest_path, "manifest_sha256")
    receipt = _signed_json(clean_receipt, "receipt_sha256")
    if (
        manifest.get("manifest_sha256") != _CLEAN_MANIFEST_SHA256
        or receipt.get("receipt_sha256") != _CLEAN_RECEIPT_SHA256
    ):
        raise TargetedRepairV5AuditError("D0 clean-view logical identity changed")
    records = list(iter_jsonl(train_path))
    if len(records) != _CLEAN_TRAIN_RECORDS or any(record.split != "train" for record in records):
        raise TargetedRepairV5AuditError("D0 clean-view count changed")
    return records, manifest, receipt


def _normalize(text: str) -> str:
    return "".join(text.casefold().split())


def masked_skeleton(text: str) -> str:
    spans = [
        match.span(1)
        for pattern in (_ASCII_SURFACE_RE, _CN_DATE_RE)
        for match in pattern.finditer(text)
    ]
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + "占位" + text[end:]
    return _normalize(text)


def _ngrams(text: str) -> frozenset[str]:
    if len(text) < 5:
        return frozenset({text})
    return frozenset(text[index : index + 5] for index in range(len(text) - 4))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def _v5_records(dataset_dir: Path) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    result: list[list[DocumentRecord]] = []
    for split in SPLITS:
        records = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
        if any(record.split != split for record in records):
            raise TargetedRepairV5AuditError("v5a record escaped its split")
        result.append(records)
    return result[0], result[1]


def design_audit(
    *,
    dataset_dir: Path,
    d0_dataset: Path,
    recipe_path: Path = DEFAULT_RECIPE_PATH,
) -> Mapping[str, Any]:
    try:
        independent = validate_materialized(dataset_dir, recipe_path)
    except TargetedRepairV5BridgeError as exc:
        raise TargetedRepairV5AuditError("v5a independent validation failed") from exc
    d0_train, d0_validation = _verify_d0(d0_dataset)
    v5_train, v5_validation = _v5_records(dataset_dir)
    rewritten = [
        record
        for record in (*v5_train, *v5_validation)
        if record.metadata.get("targeted_repair_v5a_checksum_context_rewrite") is True
    ]
    if len(rewritten) != 1200:
        raise TargetedRepairV5AuditError("v5a rewritten-record count changed")
    d0_text_hashes = {
        hashlib.sha256(record.text.encode()).hexdigest() for record in (*d0_train, *d0_validation)
    }
    rewritten_text_hashes = {
        hashlib.sha256(record.text.encode()).hexdigest() for record in rewritten
    }
    exact_text_collisions = len(d0_text_hashes & rewritten_text_hashes)
    if exact_text_collisions:
        raise TargetedRepairV5AuditError("v5a copied an exact D0 sentence")
    d0_skeletons = {masked_skeleton(record.text) for record in (*d0_train, *d0_validation)}
    rewritten_skeletons = {masked_skeleton(record.text) for record in rewritten}
    exact_skeleton_collisions = len(d0_skeletons & rewritten_skeletons)
    if exact_skeleton_collisions:
        raise TargetedRepairV5AuditError("v5a D0 masked skeleton exact gate failed")
    d0_grams = [_ngrams(skeleton) for skeleton in d0_skeletons]
    maximum_similarity = 0.0
    candidate_pairs = 0
    for skeleton in rewritten_skeletons:
        left = _ngrams(skeleton)
        for right in d0_grams:
            candidate_pairs += 1
            maximum_similarity = max(maximum_similarity, _jaccard(left, right))
    if maximum_similarity >= _D0_TEMPLATE_THRESHOLD:
        raise TargetedRepairV5AuditError("v5a D0 masked skeleton near gate failed")
    required_flags = {
        "D0_validation_informed": True,
        "D0_raw_text_copied": False,
        "test-hidden-PIIBench": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "PII_Bench_read_or_executed": False,
        "external_benchmark_read": False,
        "confirmatory_evaluation_eligible": False,
    }
    if any(
        any(record.metadata.get(key) is not expected for key, expected in required_flags.items())
        for record in rewritten
    ):
        raise TargetedRepairV5AuditError("v5a rewritten lineage flags changed")
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "targeted_repair_v5a_design_audit",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "manifest_file_sha256": independent["manifest_file_sha256"],
        "manifest_sha256": independent["manifest_sha256"],
        "counts": independent["counts"],
        "rewrite_matrix": independent["rewrite_matrix"],
        "template_audit": independent["template_audit"],
        "validator_audit": independent["validator_audit"],
        "rewrite_contract_audit": independent["rewrite_contract_audit"],
        "D0_copy_and_masked_skeleton_audit": {
            "D0_train_records_read": len(d0_train),
            "D0_validation_records_read": len(d0_validation),
            "rewritten_texts": len(rewritten_text_hashes),
            "rewritten_masked_skeletons": len(rewritten_skeletons),
            "D0_masked_skeletons": len(d0_skeletons),
            "exact_raw_text_collisions": 0,
            "exact_masked_skeleton_collisions": 0,
            "char5gram_candidate_pairs": candidate_pairs,
            "char5gram_maximum_similarity": maximum_similarity,
            "char5gram_threshold_exclusive": _D0_TEMPLATE_THRESHOLD,
            "D0_raw_text_copied": False,
        },
        "development_boundary": {
            "D0_validation_informed": True,
            "confirmatory_evaluation_eligible": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "external_benchmark_read": False,
            "PII_Bench_execution_count": 0,
        },
        "publication_boundary": {
            "optimization_only": True,
            "best_open_model_claim_supported": False,
            "new_unseen_confirmatory_evaluation_required": True,
        },
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_identifiers": False,
            "contains_absolute_paths": False,
        },
    }
    report["report_sha256"] = canonical_hash(report)
    return MappingProxyType(report)


def _sets(records: Iterable[DocumentRecord]) -> dict[str, set[str]]:
    result = {
        "document_ids": set(),
        "text_hashes": set(),
        "template_groups": set(),
        "value_groups": set(),
        "relation_groups": set(),
    }
    for record in records:
        result["document_ids"].add(record.doc_id)
        result["text_hashes"].add(hashlib.sha256(record.text.encode()).hexdigest())
        if record.template_group:
            result["template_groups"].add(record.template_group)
        result["value_groups"].update(record.entity_value_groups)
        non_entity = record.metadata.get("non_entity_value_groups")
        if isinstance(non_entity, list):
            result["value_groups"].update(str(value) for value in non_entity)
        if record.conversation_group:
            result["relation_groups"].add(record.conversation_group)
    return result


def _near_audit(
    train: Sequence[DocumentRecord], validation: Sequence[DocumentRecord]
) -> dict[str, Any]:
    exact = len(
        {hashlib.sha256(record.text.encode()).hexdigest() for record in train}
        & {hashlib.sha256(record.text.encode()).hexdigest() for record in validation}
    )
    validation_grams = [_ngrams(_normalize(record.text)) for record in validation]
    inverted: dict[str, set[int]] = {}
    for index, grams in enumerate(validation_grams):
        for gram in grams:
            inverted.setdefault(gram, set()).add(index)
    near = 0
    compared = 0
    for record in train:
        left = _ngrams(_normalize(record.text))
        candidates: set[int] = set()
        for gram in left:
            candidates.update(inverted.get(gram, ()))
        for index in candidates:
            compared += 1
            if _jaccard(left, validation_grams[index]) >= _NEAR_THRESHOLD:
                near += 1
    if exact or near:
        raise TargetedRepairV5AuditError("combined exact or near-duplicate gate failed")
    return {
        "exact_duplicate_pair_count": 0,
        "near_duplicate_pair_count": 0,
        "candidate_pair_count": compared,
        "method": "normalized_character_5gram_jaccard",
        "threshold": _NEAR_THRESHOLD,
    }


def combined_isolation_audit(
    *,
    dataset_dir: Path,
    d0_dataset: Path,
    d0_clean_view: Path,
    d0_clean_receipt: Path,
    recipe_path: Path = DEFAULT_RECIPE_PATH,
) -> Mapping[str, Any]:
    try:
        independent = validate_materialized(dataset_dir, recipe_path)
    except TargetedRepairV5BridgeError as exc:
        raise TargetedRepairV5AuditError("v5a independent validation failed") from exc
    _, d0_validation = _verify_d0(d0_dataset)
    clean_train, clean_manifest, clean_receipt = _verify_clean_view(d0_clean_view, d0_clean_receipt)
    v5_train, v5_validation = _v5_records(dataset_dir)
    train = clean_train + v5_train
    validation = d0_validation + v5_validation
    if (len(v5_train), len(v5_validation), len(train), len(validation)) != (
        3840,
        960,
        19490,
        2960,
    ):
        raise TargetedRepairV5AuditError("combined input count changed")
    train_sets = _sets(train)
    validation_sets = _sets(validation)
    isolation: dict[str, Any] = {}
    for field in train_sets:
        collisions = train_sets[field] & validation_sets[field]
        if collisions:
            raise TargetedRepairV5AuditError(f"combined {field} isolation failed")
        isolation[field] = {
            "train": len(train_sets[field]),
            "validation": len(validation_sets[field]),
            "cross_split_collisions": 0,
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "targeted_repair_v5a_combined_isolation_audit",
        "status": "passed",
        "counts": {
            "D0_clean_train": len(clean_train),
            "D0_development_validation": len(d0_validation),
            "v5a_train": len(v5_train),
            "v5a_validation": len(v5_validation),
            "combined_train": len(train),
            "combined_validation": len(validation),
        },
        "D0_identity": {
            "manifest_file_sha256": _D0_MANIFEST_FILE_SHA256,
            "manifest_sha256": _D0_MANIFEST_SHA256,
            "clean_manifest_file_sha256": _CLEAN_MANIFEST_FILE_SHA256,
            "clean_manifest_sha256": clean_manifest["manifest_sha256"],
            "clean_receipt_file_sha256": _CLEAN_RECEIPT_FILE_SHA256,
            "clean_receipt_sha256": clean_receipt["receipt_sha256"],
        },
        "v5a_identity": {
            "manifest_file_sha256": independent["manifest_file_sha256"],
            "manifest_sha256": independent["manifest_sha256"],
        },
        "independent_replay": {
            "v5a_byte_identical": independent["independent_rematerialization"]["byte_identical"],
            "D0_clean_view_content_addressed": True,
        },
        "isolation": isolation,
        "duplicate_audit": _near_audit(train, validation),
        "development_boundary": {
            "D0_validation_is_development_only": True,
            "confirmatory_evaluation_eligible": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "external_benchmark_read": False,
            "PII_Bench_execution_count": 0,
        },
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_identifiers": False,
            "contains_absolute_paths": False,
        },
    }
    report["report_sha256"] = canonical_hash(report)
    return MappingProxyType(report)


__all__ = [
    "TargetedRepairV5AuditError",
    "combined_isolation_audit",
    "design_audit",
    "masked_skeleton",
]
