#!/usr/bin/env python3
"""Run a raw-text-free structural audit and prepare a manual OpenPII queue.

The aggregate report contains no source text or entity values.  The separate
manual-review queue contains synthetic source rows and must remain in the
external quarantined data root; it is never written into Git or a training
pool by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.converters._common import ConversionError  # noqa: E402
from pii_zh.data.converters.openpii import convert_openpii_record  # noqa: E402
from pii_zh.data.schema import SchemaError  # noqa: E402

OPENPII_REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
_PLACEHOLDER_INDEX = re.compile(r"_\d+(?=\])")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_manual_queue(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        for row in rows:
            item = {
                "sample_id": "sha256:" + _stable_hash(str(row.get("uid", ""))),
                "source_uid": row.get("uid"),
                "text": row["source_text"],
                "source_entities": row["privacy_mask"],
                "review": {
                    "span_boundaries_correct": None,
                    "labels_correct": None,
                    "natural_zh_cn": None,
                    "contains_real_personal_data": None,
                    "notes": None,
                    "reviewer": None,
                    "reviewed_at": None,
                },
            }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o600)
    return _sha256(path)


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
    return rows


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_labels: Counter[str] = Counter()
    converted_labels: Counter[str] = Counter()
    validator_failures: Counter[str] = Counter()
    conversion_error_types: Counter[str] = Counter()
    text_hashes: Counter[str] = Counter()
    template_hashes: Counter[str] = Counter()
    invalid_source_spans = 0
    source_span_count = 0
    converted_span_count = 0
    dropped_source_spans = 0
    zero_entity_documents = 0

    for row in rows:
        text = row.get("source_text")
        masked = row.get("masked_text")
        spans = row.get("privacy_mask")
        if not isinstance(text, str) or not isinstance(masked, str) or not isinstance(spans, list):
            conversion_error_types["invalid_source_schema"] += 1
            continue
        text_hashes[_stable_hash(text)] += 1
        template = _PLACEHOLDER_INDEX.sub("", masked)
        template_hashes[_stable_hash(template)] += 1
        for span in spans:
            source_span_count += 1
            if not isinstance(span, dict):
                invalid_source_spans += 1
                continue
            label = str(span.get("label", "<missing>"))
            source_labels[label] += 1
            start, end, value = span.get("start"), span.get("end"), span.get("value")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(text)
                or text[start:end] != value
            ):
                invalid_source_spans += 1
        try:
            record = convert_openpii_record(row, source_revision=OPENPII_REVISION)
            record.validate()
        except (ConversionError, SchemaError, KeyError, TypeError, ValueError) as exc:
            conversion_error_types[type(exc).__name__] += 1
            continue
        if not record.entities:
            zero_entity_documents += 1
        converted_span_count += len(record.entities)
        dropped_source_spans += int(record.quality.metadata["dropped_unknown_labels"])
        for entity in record.entities:
            converted_labels[entity.label] += 1
            if entity.validator_state == "failed":
                validator_failures[entity.label] += 1

    duplicate_text_groups = [count for count in text_hashes.values() if count > 1]
    duplicate_template_groups = [count for count in template_hashes.values() if count > 1]
    structural_pass = not invalid_source_spans and not conversion_error_types
    quality_pass = structural_pass and not validator_failures
    return {
        "documents": len(rows),
        "source_spans": source_span_count,
        "converted_spans": converted_span_count,
        "dropped_source_spans": dropped_source_spans,
        "zero_mapped_entity_documents": zero_entity_documents,
        "invalid_source_spans": invalid_source_spans,
        "conversion_errors": dict(sorted(conversion_error_types.items())),
        "source_label_counts": dict(sorted(source_labels.items())),
        "converted_label_counts": dict(sorted(converted_labels.items())),
        "validator_failure_counts": dict(sorted(validator_failures.items())),
        "deduplication": {
            "exact_duplicate_groups": len(duplicate_text_groups),
            "rows_in_exact_duplicate_groups": sum(duplicate_text_groups),
            "max_exact_duplicate_group": max(duplicate_text_groups, default=1),
            "template_duplicate_groups": len(duplicate_template_groups),
            "rows_in_template_duplicate_groups": sum(duplicate_template_groups),
            "max_template_duplicate_group": max(duplicate_template_groups, default=1),
        },
        "automated_structural_gate": "pass" if structural_pass else "fail",
        "automated_quality_gate": "pass" if quality_pass else "fail",
        "quality_blockers": (
            []
            if quality_pass
            else [
                "structured_entity_validator_failures",
                "manual_2000_sample_review_pending",
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manual-queue", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("sample size must be positive")
    inputs = [path.expanduser().resolve(strict=True) for path in args.input]
    rows = _load_rows(inputs)
    report = _audit(rows)
    ranked = sorted(
        rows,
        key=lambda row: _stable_hash(f"{args.seed}:{row.get('uid', '')}"),
    )
    sample = ranked[: min(args.sample_size, len(ranked))]
    queue_sha256 = _write_manual_queue(args.manual_queue, sample)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "dataset": "ai4privacy/pii-masking-openpii-1.5m",
            "revision": OPENPII_REVISION,
            "license": "CC-BY-4.0",
            "input_files": [{"name": path.name, "sha256": _sha256(path)} for path in inputs],
        },
        "automated_audit": report,
        "manual_audit": {
            "status": "pending",
            "required": True,
            "sample_size": len(sample),
            "queue_sha256": queue_sha256,
            "admission_allowed": False,
        },
        "overall_status": "AUTOMATED_ONLY_MANUAL_REVIEW_PENDING",
    }
    _atomic_json(args.report, payload)
    print(
        f"documents={report['documents']} source_spans={report['source_spans']} "
        f"structural_gate={report['automated_structural_gate']} "
        f"quality_gate={report['automated_quality_gate']}"
    )
    print(f"manual_sample={len(sample)} queue_sha256={queue_sha256}")


if __name__ == "__main__":
    main()
