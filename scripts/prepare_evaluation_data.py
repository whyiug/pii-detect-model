#!/usr/bin/env python3
"""Prepare and attest the frozen PII Bench ZH evaluation dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.converters.pii_bench_zh import (  # noqa: E402
    convert_pii_bench_zh_record,
)
from pii_zh.data.schema import DataPool, dumps_document, iter_jsonl  # noqa: E402
from pii_zh.evaluation import add_manifest_hash, sha256_file  # noqa: E402

PII_BENCH_REVISION = "c350b94897af668517ff5de237d89f2ce2eaa6f0"
PII_BENCH_CONVERTER_VERSION = "1.0.0"
PII_BENCH_SOURCE = "wan9yu/pii-bench-zh"
PII_BENCH_LICENSE = "Apache-2.0"

# These are the immutable files frozen for the public evaluation.  Changing a
# source or converter requires a deliberate new benchmark manifest/version.
_FROZEN_SUBSETS: dict[str, dict[str, object]] = {
    "formal": {
        "record_count": 5_000,
        "span_count": 15_662,
        "raw_sha256": "bcaa54909fd69c6737b8e442c4d740ce1bf5b95b7d4af70b2f2692010e1ff7a5",
        "canonical_sha256": "bb978137711cc90ae0c16df0645983413f3b34943bf29469e58b795b183a38af",
    },
    "chat": {
        "record_count": 3_000,
        "span_count": 7_544,
        "raw_sha256": "4ddacd78dfac49b723032de5e1c29c432a44049245e03a8ea960c86bbf3b765c",
        "canonical_sha256": "2d73326a7ea2da9966ba13336db1a791f03d5f18b7982fdc4495cb726c1180a5",
    },
}


def _convert(source: Path, destination: Path, subset: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("r", encoding="utf-8") as input_handle,
        tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            delete=False,
            prefix=f".{destination.name}.",
        ) as output_handle,
    ):
        temporary = Path(output_handle.name)
        try:
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number}: expected an object")
                value["subset"] = subset
                record = convert_pii_bench_zh_record(
                    value,
                    source_revision=PII_BENCH_REVISION,
                )
                if record.data_pool is not DataPool.EVALUATION_ONLY:
                    raise RuntimeError("benchmark converter violated evaluation-only isolation")
                output_handle.write(dumps_document(record) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def _summarize_and_verify(source: Path, destination: Path, subset: str) -> dict[str, object]:
    frozen = _FROZEN_SUBSETS[subset]
    raw_sha256 = sha256_file(source)
    if raw_sha256 != frozen["raw_sha256"]:
        raise RuntimeError(f"frozen {subset} raw SHA-256 mismatch")

    canonical_sha256 = sha256_file(destination)
    if canonical_sha256 != frozen["canonical_sha256"]:
        raise RuntimeError(f"frozen {subset} canonical SHA-256 mismatch")

    label_counts: Counter[str] = Counter()
    record_count = 0
    span_count = 0
    for record in iter_jsonl(destination):
        if record.data_pool is not DataPool.EVALUATION_ONLY:
            raise RuntimeError(f"canonical {subset} row escaped evaluation_only isolation")
        if record.metadata.get("benchmark_subset") != subset:
            raise RuntimeError(f"canonical {subset} row has inconsistent subset identity")
        record_count += 1
        span_count += len(record.entities)
        label_counts.update(entity.label for entity in record.entities)
    if record_count != frozen["record_count"]:
        raise RuntimeError(f"unexpected {subset} record_count: {record_count}")
    if span_count != frozen["span_count"]:
        raise RuntimeError(f"unexpected {subset} span_count: {span_count}")

    source.chmod(0o444)
    destination.chmod(0o444)
    return {
        "record_count": record_count,
        "span_count": span_count,
        "label_counts": dict(sorted(label_counts.items())),
        "raw": {"sha256": raw_sha256, "record_count": record_count},
        "canonical": {"sha256": canonical_sha256, "record_count": record_count},
    }


def _prepared_at(value: str | None) -> tuple[str, dict[str, str]]:
    strategy: dict[str, str]
    if value is not None:
        raw_value = value
        strategy = {"type": "explicit", "source": "--prepared-at"}
    else:
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if epoch is None:
            raise ValueError(
                "set --prepared-at or SOURCE_DATE_EPOCH; wall-clock timestamps are not reproducible"
            )
        try:
            timestamp = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        except (OverflowError, ValueError) as exc:
            raise ValueError("SOURCE_DATE_EPOCH must be a valid Unix timestamp") from exc
        raw_value = timestamp.isoformat()
        strategy = {"type": "source_date_epoch", "source": "SOURCE_DATE_EPOCH"}
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("prepared_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("prepared_at must include a timezone")
    normalized = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z"), strategy


def _write_manifest(manifest: dict[str, object], destination: Path) -> None:
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
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--chat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Default: OUTPUT_DIR/pii_bench_zh.dataset_manifest.json",
    )
    parser.add_argument(
        "--prepared-at",
        help="Explicit timezone-aware ISO-8601 timestamp (or set SOURCE_DATE_EPOCH)",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Attest existing canonical files without rewriting them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "formal": args.formal.expanduser().resolve(strict=True),
        "chat": args.chat.expanduser().resolve(strict=True),
    }
    output_dir = args.output_dir.expanduser().resolve()
    prepared_at, prepared_at_strategy = _prepared_at(args.prepared_at)
    results: dict[str, dict[str, object]] = {}
    for subset, source in inputs.items():
        destination = output_dir / f"pii_bench_zh_{subset}.canonical.jsonl"
        if args.verify_existing:
            destination.resolve(strict=True)
        else:
            if sha256_file(source) != _FROZEN_SUBSETS[subset]["raw_sha256"]:
                raise RuntimeError(f"frozen {subset} raw SHA-256 mismatch")
            _convert(source, destination, subset)
        result = _summarize_and_verify(source, destination, subset)
        results[subset] = result
        print(
            f"verified subset={subset} records={result['record_count']} "
            f"sha256={result['canonical']['sha256']}"
        )

    converter_path = _REPOSITORY_ROOT / "src/pii_zh/data/converters/pii_bench_zh.py"
    manifest = add_manifest_hash(
        {
            "schema_version": 1,
            "manifest_type": "evaluation_dataset",
            "source_id": "pii_bench_zh",
            "upstream": {
                "source": PII_BENCH_SOURCE,
                "revision": PII_BENCH_REVISION,
                "license": PII_BENCH_LICENSE,
            },
            "converter": {
                "name": "pii_zh.data.converters.pii_bench_zh",
                "version": PII_BENCH_CONVERTER_VERSION,
                "implementation_sha256": sha256_file(converter_path),
            },
            "prepared_at": prepared_at,
            "prepared_at_strategy": prepared_at_strategy,
            "isolation": {
                "pool": "evaluation_only",
                "training_allowed": False,
                "threshold_selection_allowed": False,
                "distillation_allowed": False,
            },
            "subsets": results,
            "totals": {
                "record_count": sum(int(item["record_count"]) for item in results.values()),
                "span_count": sum(int(item["span_count"]) for item in results.values()),
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
            },
        }
    )
    manifest_path = args.manifest_output or output_dir / "pii_bench_zh.dataset_manifest.json"
    _write_manifest(manifest, manifest_path.expanduser().resolve())
    print(
        f"manifest={manifest_path.name} manifest_sha256={manifest['manifest_sha256']} "
        "isolation=evaluation_only training_allowed=false"
    )


if __name__ == "__main__":
    main()
