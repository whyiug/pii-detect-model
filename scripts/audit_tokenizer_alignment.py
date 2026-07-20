#!/usr/bin/env python3
"""Audit exact entity-boundary alignment without retaining source text."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.alignment import align_document_to_bio  # noqa: E402
from pii_zh.data.schema import iter_jsonl  # noqa: E402
from pii_zh.tokenization import (  # noqa: E402
    configure_character_boundary_tokenizer,
    load_serialized_fast_tokenizer,
)
from pii_zh.training.manifest import (  # noqa: E402
    canonical_json_hash,
    sha256_file,
    tokenizer_fingerprint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Local planning/provenance document recorded by hash; it is not distributed.",
    )
    return parser


def _quantile(values: list[int], fraction: float) -> int:
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def _audit(tokenizer: Any, inputs: list[Path]) -> dict[str, Any]:
    documents = entities = unalignable = failures = 0
    token_lengths: list[int] = []
    label_counts: Counter[str] = Counter()
    unalignable_by_label: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    for path in inputs:
        for record in iter_jsonl(path):
            documents += 1
            try:
                alignment = align_document_to_bio(
                    record,
                    tokenizer,
                    add_special_tokens=True,
                    truncation=False,
                )
            except Exception as exc:  # sanitized aggregate evidence only
                failures += 1
                failure_types[type(exc).__name__] += 1
                continue
            token_lengths.append(len(alignment.input_ids))
            entities += alignment.statistics.entity_count
            unalignable += alignment.statistics.unalignable_count
            label_counts.update(entity.label for entity in record.entities)
            unalignable_by_label.update(item.label for item in alignment.unalignable_boundaries)
    if not token_lengths:
        raise RuntimeError("tokenizer audit produced no aligned documents")
    token_lengths.sort()
    return {
        "documents": documents,
        "entities": entities,
        "failures": failures,
        "failure_types": dict(sorted(failure_types.items())),
        "unalignable_boundaries": unalignable,
        "unalignable_boundary_rate": unalignable / entities if entities else 0.0,
        "label_counts": dict(sorted(label_counts.items())),
        "unalignable_by_label": dict(sorted(unalignable_by_label.items())),
        "token_lengths": {
            "minimum": token_lengths[0],
            "p50": _quantile(token_lengths, 0.50),
            "p95": _quantile(token_lengths, 0.95),
            "p99": _quantile(token_lengths, 0.99),
            "maximum": token_lengths[-1],
        },
    }


def main() -> int:
    args = _parser().parse_args()
    base_model = args.base_model.expanduser().resolve(strict=True)
    inputs = [path.expanduser().resolve(strict=True) for path in args.input]
    default_tokenizer = load_serialized_fast_tokenizer(base_model)
    safe_tokenizer = configure_character_boundary_tokenizer(copy.deepcopy(default_tokenizer))
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "privacy": {"contains_raw_text": False, "contains_entity_values": False},
        "plan_sha256": sha256_file(args.plan.expanduser().resolve(strict=True)),
        "inputs": [{"name": path.name, "sha256": sha256_file(path)} for path in inputs],
        "tokenizer": tokenizer_fingerprint(base_model, tokenizer=safe_tokenizer),
        "variants": {
            "qwen_default": _audit(default_tokenizer, inputs),
            "unicode_codepoint_v1": _audit(safe_tokenizer, inputs),
        },
    }
    safe_result = report["variants"]["unicode_codepoint_v1"]
    report["gate_passed"] = (
        safe_result["failures"] == 0 and safe_result["unalignable_boundaries"] == 0
    )
    report["report_sha256"] = canonical_json_hash(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        delete=False,
        prefix=f".{args.output.name}.",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o444)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "gate_passed": report["gate_passed"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
