#!/usr/bin/env python3
"""Suppress validator-rejected structured spans without retaining source text."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.schema import iter_jsonl  # noqa: E402
from pii_zh.evaluation import (  # noqa: E402
    add_manifest_hash,
    load_prediction_jsonl,
    sha256_file,
    write_prediction_jsonl,
)
from pii_zh.fusion import suppress_invalid_structured_spans  # noqa: E402

REFINEMENT_ID = "structured_refinement_v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser


def _write_json_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.chmod(temporary, 0o444)
    os.replace(temporary, destination)


def main() -> int:
    args = _parser().parse_args()
    documents = args.documents.expanduser().resolve(strict=True)
    predictions = args.predictions.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    if len({documents, predictions, output, audit_path}) != 4:
        raise ValueError("documents, predictions, output, and audit must be distinct")

    prediction_records = load_prediction_jsonl(predictions)
    indexed = {record.doc_id: record for record in prediction_records}
    input_span_count = sum(len(record.spans) for record in prediction_records)
    suppressed: Counter[str] = Counter()
    refined = []
    for document in iter_jsonl(documents):
        prediction = indexed.pop(document.doc_id, None)
        if prediction is None:
            raise ValueError("document and prediction identifier sets differ")
        result, counts = suppress_invalid_structured_spans(prediction, document.text)
        refined.append(result)
        suppressed.update(counts)
    if indexed:
        raise ValueError("document and prediction identifier sets differ")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False, prefix=f".{output.name}."
    ) as handle:
        temporary = Path(handle.name)
        try:
            write_prediction_jsonl(refined, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.chmod(temporary, 0o444)
    os.replace(temporary, output)

    output_span_count = sum(len(record.spans) for record in refined)
    audit = add_manifest_hash(
        {
            "schema_version": 1,
            "manifest_type": "structured_prediction_refinement",
            "refinement_id": REFINEMENT_ID,
            "inputs": {
                "documents_sha256": sha256_file(documents),
                "predictions_sha256": sha256_file(predictions),
                "document_count": len(refined),
                "input_span_count": input_span_count,
            },
            "output": {
                "predictions_sha256": sha256_file(output),
                "span_count": output_span_count,
            },
            "suppressed_by_label": dict(sorted(suppressed.items())),
            "privacy": {
                "contains_document_ids": False,
                "contains_entity_values": False,
                "contains_paths": False,
                "contains_raw_text": False,
            },
        }
    )
    _write_json_atomic(audit, audit_path)
    print(
        json.dumps(
            {
                "documents": len(refined),
                "input_spans": input_span_count,
                "output": output.name,
                "output_spans": output_span_count,
                "suppressed": sum(suppressed.values()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
