"""Train-only audit for the proposed MacBERT v5b numeric-O loss scope.

The audit deliberately accepts only the two frozen training inputs.  It has no
validation, test, hidden, or external-benchmark input surface and emits only
aggregate counts and cryptographic fingerprints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from stat import S_IMODE
from typing import Any

from transformers import AutoTokenizer

from pii_zh.taxonomy import load_taxonomy
from pii_zh.training.data import DocumentTokenDataset, build_bio_label_maps, load_aligned_jsonl
from pii_zh.training.safe_conversion import canonical_json_hash, sha256_file

NUMERIC_TOKEN_REGEX = r"(?:##)?[0-9]+"
NUMERIC_TOKEN_PATTERN = re.compile(NUMERIC_TOKEN_REGEX)
OUTSIDE_LABEL_ID = 0
IGNORE_INDEX = -100
NUMERIC_O_WEIGHT = 4
MAX_LENGTH = 192

EXPECTED_INVENTORY_COUNT = 1015
EXPECTED_INVENTORY_SHA256 = "aa02b19473d39a374cd5341a63ff66bf91c4ce01fec8323a8faabd77d6465386"
EXPECTED_COMBINED_COUNTS = {
    "valid_tokens": 632_679,
    "digit_gold_O_tokens": 24_891,
    "digit_entity_tokens": 94_011,
}
EXPECTED_INPUTS: dict[str, dict[str, Any]] = {
    "D0": {
        "sha256": "4243d3302a243ee735686c2e9986935fa82120bbe4d7a4a184c28d4376d343ad",
        "documents": 15_650,
        "counts": {
            "valid_tokens": 526_796,
            "digit_gold_O_tokens": 15_871,
            "digit_entity_tokens": 82_937,
        },
    },
    "targeted_v5a": {
        "sha256": "726a7209fd02f73e7da3330114ae00486efd2dcd31232bd4bfb421a47d714cd4",
        "documents": 3_840,
        "counts": {
            "valid_tokens": 105_883,
            "digit_gold_O_tokens": 9_020,
            "digit_entity_tokens": 11_074,
        },
    },
}
EXPECTED_MODEL_FILES = {
    "config.json": "648c8a2b51c12551f94bae19a66c21a59c9bd8aa0ebb99f2a0af7236cdb99daa",
    "tokenizer.json": "53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}


class NumericOLossScopeAuditError(RuntimeError):
    """Fail-closed audit error that never echoes source text or local paths."""


def _regular_file(path: Path, *, field: str) -> Path:
    try:
        candidate = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NumericOLossScopeAuditError(f"{field} is unavailable") from exc
    if path.is_symlink() or not candidate.is_file():
        raise NumericOLossScopeAuditError(f"{field} must be a regular non-symlink file")
    return candidate


def _regular_directory(path: Path, *, field: str) -> Path:
    try:
        candidate = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NumericOLossScopeAuditError(f"{field} is unavailable") from exc
    if path.is_symlink() or not candidate.is_dir():
        raise NumericOLossScopeAuditError(f"{field} must be a regular non-symlink directory")
    return candidate


def numeric_inventory(tokenizer: Any) -> list[list[str | int]]:
    """Return the frozen digits-only WordPiece inventory in canonical order."""

    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping):
        raise NumericOLossScopeAuditError("tokenizer vocabulary must be a mapping")
    items: list[list[str | int]] = []
    for token, raw_token_id in vocabulary.items():
        if (
            not isinstance(token, str)
            or isinstance(raw_token_id, bool)
            or not isinstance(raw_token_id, int)
        ):
            raise NumericOLossScopeAuditError("tokenizer vocabulary is malformed")
        if raw_token_id < 0:
            raise NumericOLossScopeAuditError("tokenizer vocabulary contains a negative ID")
        if NUMERIC_TOKEN_PATTERN.fullmatch(token):
            items.append([token, raw_token_id])
    items.sort(key=lambda item: (str(item[0]), int(item[1])))
    if len({int(item[1]) for item in items}) != len(items):
        raise NumericOLossScopeAuditError("numeric token inventory contains duplicate IDs")
    return items


def count_loss_scope(
    dataset: DocumentTokenDataset, *, numeric_token_ids: frozenset[int]
) -> dict[str, int]:
    """Count valid, digit-gold-O, and digit-entity tokens without retaining content."""

    counts = {name: 0 for name in EXPECTED_COMBINED_COUNTS}
    for document in dataset.documents:
        if not (len(document.input_ids) == len(document.attention_mask) == len(document.labels)):
            raise NumericOLossScopeAuditError("encoded training fields have inconsistent lengths")
        for token_id, attention, label in zip(
            document.input_ids,
            document.attention_mask,
            document.labels,
            strict=True,
        ):
            if label == IGNORE_INDEX:
                continue
            if label < 0:
                raise NumericOLossScopeAuditError(
                    "encoded training labels violate ignore semantics"
                )
            if attention != 1:
                raise NumericOLossScopeAuditError(
                    "a valid training label appears outside the attention mask"
                )
            counts["valid_tokens"] += 1
            if token_id in numeric_token_ids:
                field = (
                    "digit_gold_O_tokens" if label == OUTSIDE_LABEL_ID else "digit_entity_tokens"
                )
                counts[field] += 1
    return counts


def weight_four_explanation(counts: Mapping[str, int]) -> dict[str, Any]:
    """Explain why weight four approximately balances numeric O and entity mass."""

    try:
        valid = int(counts["valid_tokens"])
        digit_o = int(counts["digit_gold_O_tokens"])
        digit_entity = int(counts["digit_entity_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NumericOLossScopeAuditError("loss-scope counts are malformed") from exc
    if min(valid, digit_o, digit_entity) <= 0 or digit_o + digit_entity > valid:
        raise NumericOLossScopeAuditError("loss-scope counts violate the token partition")
    weighted_digit_o = NUMERIC_O_WEIGHT * digit_o
    weighted_denominator = valid + (NUMERIC_O_WEIGHT - 1) * digit_o
    return {
        "fixed_weight": NUMERIC_O_WEIGHT,
        "selection_rule": "nearest_integer_to_digit_entity_over_digit_gold_O",
        "digit_entity_over_digit_gold_O": round(digit_entity / digit_o, 12),
        "nearest_integer": round(digit_entity / digit_o),
        "unweighted_digit_gold_O_over_digit_entity": round(digit_o / digit_entity, 12),
        "weighted_digit_gold_O_mass": weighted_digit_o,
        "weighted_denominator": weighted_denominator,
        "weighted_digit_gold_O_over_digit_entity": round(weighted_digit_o / digit_entity, 12),
        "digit_gold_O_fraction_of_valid": round(digit_o / valid, 12),
        "digit_entity_fraction_of_valid": round(digit_entity / valid, 12),
        "weighted_digit_gold_O_fraction_of_denominator": round(
            weighted_digit_o / weighted_denominator, 12
        ),
        "weighted_denominator_over_unweighted": round(weighted_denominator / valid, 12),
    }


def _verify_source_model(source_model: Path) -> tuple[Any, dict[str, Any]]:
    source = _regular_directory(source_model, field="source model")
    hashes: dict[str, str] = {}
    for name, expected_hash in EXPECTED_MODEL_FILES.items():
        artifact = source / name
        if artifact.is_symlink() or not artifact.is_file():
            raise NumericOLossScopeAuditError("source model is missing a tokenizer artifact")
        observed = sha256_file(artifact)
        if observed != expected_hash:
            raise NumericOLossScopeAuditError("source tokenizer fingerprint changed")
        hashes[name] = observed
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as exc:
        raise NumericOLossScopeAuditError("offline tokenizer loading failed") from exc
    if not getattr(tokenizer, "is_fast", False) or tokenizer.pad_token_id is None:
        raise NumericOLossScopeAuditError("source model lacks the frozen fast tokenizer")
    return tokenizer, {
        "source_id": "hfl/chinese-macbert-base",
        "revision": "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files": hashes,
        "local_files_only": True,
        "trust_remote_code": False,
        "GPU_used": False,
    }


def _load_input(
    *,
    name: str,
    path: Path,
    tokenizer: Any,
    label2id: dict[str, int],
    numeric_token_ids: frozenset[int],
) -> tuple[DocumentTokenDataset, dict[str, Any]]:
    expected = EXPECTED_INPUTS[name]
    source = _regular_file(path, field=f"{name} train input")
    digest = sha256_file(source)
    if digest != expected["sha256"]:
        raise NumericOLossScopeAuditError(f"{name} train hash violates the frozen contract")
    try:
        dataset, summary = load_aligned_jsonl(
            source,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=MAX_LENGTH,
            expected_split="train",
        )
    except Exception as exc:
        raise NumericOLossScopeAuditError(f"{name} train alignment failed") from exc
    counts = count_loss_scope(dataset, numeric_token_ids=numeric_token_ids)
    if summary.document_count != expected["documents"] or counts != expected["counts"]:
        raise NumericOLossScopeAuditError(f"{name} train loss-scope counts changed")
    return dataset, {
        "dataset_id": name,
        "split": "train",
        "sha256": digest,
        "documents": summary.document_count,
        **counts,
    }


def build_report(*, source_model: Path, d0_train: Path, targeted_train: Path) -> dict[str, Any]:
    """Run the complete frozen train-only audit and return a self-hashed report."""

    tokenizer, source_summary = _verify_source_model(source_model)
    inventory = numeric_inventory(tokenizer)
    inventory_hash = canonical_json_hash(inventory)
    if len(inventory) != EXPECTED_INVENTORY_COUNT or inventory_hash != EXPECTED_INVENTORY_SHA256:
        raise NumericOLossScopeAuditError("numeric tokenizer inventory changed")
    numeric_token_ids = frozenset(int(item[1]) for item in inventory)
    label2id, _ = build_bio_label_maps(load_taxonomy())
    if label2id.get("O") != OUTSIDE_LABEL_ID:
        raise NumericOLossScopeAuditError("outside-label identity changed")
    _, d0 = _load_input(
        name="D0",
        path=d0_train,
        tokenizer=tokenizer,
        label2id=label2id,
        numeric_token_ids=numeric_token_ids,
    )
    _, targeted = _load_input(
        name="targeted_v5a",
        path=targeted_train,
        tokenizer=tokenizer,
        label2id=label2id,
        numeric_token_ids=numeric_token_ids,
    )
    combined = {field: int(d0[field]) + int(targeted[field]) for field in EXPECTED_COMBINED_COUNTS}
    if combined != EXPECTED_COMBINED_COUNTS:
        raise NumericOLossScopeAuditError("combined train loss-scope counts changed")
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "macbert_24_numeric_o_v5b_train_only_loss_scope_audit",
        "audit_id": "macbert_24_numeric_o_v5b_loss_scope_v1",
        "status": "passed",
        "gate_passed": True,
        "scope": {
            "split": "train_only",
            "inputs": ["D0", "targeted_v5a"],
            "validation_data_read": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "external_benchmark_read": False,
            "PII_Bench_execution_count": 0,
            "GPU_used": False,
        },
        "source_model": source_summary,
        "numeric_inventory": {
            "regex": NUMERIC_TOKEN_REGEX,
            "match_semantics": "fullmatch",
            "sort_order": ["token", "id"],
            "count": len(inventory),
            "canonical_sha256": inventory_hash,
            "tokens_disclosed": False,
        },
        "inputs": {"D0": d0, "targeted_v5a": targeted},
        "combined_train": combined,
        "loss_scope": {
            "outside_label": "O",
            "outside_label_id": OUTSIDE_LABEL_ID,
            "ignore_index": IGNORE_INDEX,
            "condition": ("attention=1 AND label=O AND token_fullmatches_(?:##)?[0-9]+"),
            "normalization": "weighted_mean_over_labels_not_equal_to_ignore_index",
            "weight_four_explanation": weight_four_explanation(combined),
        },
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "contains_local_paths": False,
        },
    }
    report["report_sha256"] = canonical_json_hash(report)
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    """Verify the report self-hash, frozen aggregates, and privacy contract."""

    claimed = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if not isinstance(claimed, str) or canonical_json_hash(unsigned) != claimed:
        raise NumericOLossScopeAuditError("report self-hash does not verify")
    if (
        report.get("gate_passed") is not True
        or report.get("combined_train") != EXPECTED_COMBINED_COUNTS
        or report.get("numeric_inventory", {}).get("canonical_sha256") != EXPECTED_INVENTORY_SHA256
        or report.get("privacy")
        != {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "contains_local_paths": False,
        }
    ):
        raise NumericOLossScopeAuditError("report violates the frozen audit contract")


def write_report(report: Mapping[str, Any], output: Path) -> None:
    """Atomically create a read-only report without overwriting prior evidence."""

    verify_report(report)
    destination = output.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise NumericOLossScopeAuditError("audit output must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if S_IMODE(destination.stat().st_mode) != 0o444:
        raise NumericOLossScopeAuditError("audit output mode verification failed")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source-model", required=True, type=Path)
    command.add_argument("--d0-train", required=True, type=Path)
    command.add_argument("--targeted-train", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = build_report(
        source_model=args.source_model,
        d0_train=args.d0_train,
        targeted_train=args.targeted_train,
    )
    write_report(report, args.output)
    return 0


__all__ = [
    "EXPECTED_COMBINED_COUNTS",
    "EXPECTED_INVENTORY_COUNT",
    "EXPECTED_INVENTORY_SHA256",
    "IGNORE_INDEX",
    "NUMERIC_O_WEIGHT",
    "NUMERIC_TOKEN_REGEX",
    "NumericOLossScopeAuditError",
    "build_report",
    "count_loss_scope",
    "main",
    "numeric_inventory",
    "parser",
    "verify_report",
    "weight_four_explanation",
    "write_report",
]
