#!/usr/bin/env python3
"""Freeze the public datasets used by the PII-ZH project.

The script deliberately keeps OpenPII in ``quarantined`` until its quality
audit and license gate are recorded.  PII Bench ZH is always written under
``evaluation_only`` and must never be admitted to a training pool.

No raw text is printed.  Logs contain only source IDs, counts, hashes, and
paths.  ``HF_ENDPOINT`` is honored without echoing the configured endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENPII_ID = "ai4privacy/pii-masking-openpii-1.5m"
OPENPII_REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
OPENPII_EXPECTED = {"train": 20_962, "validation": 5_310}
OPENPII_PARQUET_SHARD = "0000.parquet"

PII_BENCH_ID = "wan9yu/pii-bench-zh"
PII_BENCH_REVISION = "c350b94897af668517ff5de237d89f2ce2eaa6f0"
PII_BENCH_FILES = {
    "formal": ("data/pii_bench_zh.jsonl", 5_000),
    "chat": ("data/pii_bench_zh_chat.jsonl", 3_000),
}

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_DATA_ROOT_ENV = "PII_ZH_PUBLIC_DATA_ROOT"


def _default_data_root() -> Path:
    configured = os.environ.get(_PUBLIC_DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    return _REPOSITORY_ROOT / "data" / "raw" / "public_sources"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _endpoint() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("HF_ENDPOINT must be an HTTPS origin.")
    return endpoint


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=output.parent, delete=False, prefix=f".{output.name}."
    ) as handle:
        temporary = Path(handle.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pii-zh-qwen/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output)


def _count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def fetch_pii_bench(root: Path, endpoint: str) -> dict[str, Any]:
    destination = root / "evaluation_only" / "pii_bench_zh" / "raw"
    files: dict[str, Any] = {}
    for subset, (remote_path, expected_rows) in PII_BENCH_FILES.items():
        output = destination / Path(remote_path).name
        quoted_path = "/".join(urllib.parse.quote(part) for part in remote_path.split("/"))
        url = f"{endpoint}/datasets/{PII_BENCH_ID}/resolve/{PII_BENCH_REVISION}/{quoted_path}"
        _download(url, output)
        rows = _count_jsonl(output)
        if rows != expected_rows:
            raise RuntimeError(
                f"Unexpected {PII_BENCH_ID}/{subset} row count: {rows} != {expected_rows}."
            )
        output.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        files[subset] = {
            "path": str(output.relative_to(root)),
            "rows": rows,
            "sha256": _sha256(output),
            "mode": "evaluation_only",
        }
        print(f"frozen {PII_BENCH_ID}/{subset}: rows={rows} sha256={files[subset]['sha256']}")

    return {
        "dataset": PII_BENCH_ID,
        "revision": PII_BENCH_REVISION,
        "declared_license": "Apache-2.0",
        "pool": "evaluation_only",
        "training_allowed": False,
        "files": files,
    }


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
        raise RuntimeError(
            "OpenPII extraction requires duckdb. Run with `uv run --with duckdb "
            "python scripts/fetch_public_data.py ...`."
        ) from exc
    return duckdb


def fetch_openpii(root: Path, endpoint: str) -> dict[str, Any]:
    """Extract only zh-CN rows via Parquet range reads.

    Dataset Viewer statistics prove that, for the pinned release, all zh-CN
    rows occur in shard ``0000`` for each split.  Counts are fail-closed so a
    changed conversion cannot silently omit data.
    """

    duckdb = _duckdb()
    destination = root / "quarantined" / "openpii_zh_cn" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    files: dict[str, Any] = {}
    selected_columns = """
        source_text, masked_text, privacy_mask, split, uid, language,
        region, script, source_dataset
    """
    try:
        for split, expected_rows in OPENPII_EXPECTED.items():
            converted_ref = urllib.parse.quote("refs/convert/parquet", safe="")
            url = (
                f"{endpoint}/datasets/{OPENPII_ID}/resolve/{converted_ref}/"
                f"default/{split}/{OPENPII_PARQUET_SHARD}"
            )
            output = destination / f"openpii_zh_cn_{split}.parquet"
            temporary = output.with_suffix(".parquet.partial")
            temporary.unlink(missing_ok=True)
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE selected AS
                SELECT {selected_columns}
                FROM read_parquet(?)
                WHERE language = 'zh' AND region = 'CN'
                """,
                [url],
            )
            rows = int(connection.execute("SELECT count(*) FROM selected").fetchone()[0])
            if rows != expected_rows:
                raise RuntimeError(
                    f"Unexpected {OPENPII_ID}/{split} zh-CN count: {rows} != {expected_rows}."
                )
            connection.execute(
                "COPY selected TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)]
            )
            os.replace(temporary, output)
            files[split] = {
                "path": str(output.relative_to(root)),
                "rows": rows,
                "sha256": _sha256(output),
                "source_converted_shard": OPENPII_PARQUET_SHARD,
            }
            print(f"extracted {OPENPII_ID}/{split}: rows={rows} sha256={files[split]['sha256']}")
    finally:
        connection.close()

    return {
        "dataset": OPENPII_ID,
        "revision": OPENPII_REVISION,
        "declared_license": "CC-BY-4.0",
        "attribution": "Ai4Privacy / Ai Suisse SA",
        "pool": "quarantined",
        "training_allowed": False,
        "admission_blockers": ["quality_audit", "template_deduplication", "license_gate_record"],
        "filter": {"language": "zh", "region": "CN"},
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_data_root(),
        help=(
            "Data root (default: repository data/raw/public_sources; override with "
            f"{_PUBLIC_DATA_ROOT_ENV} or --root). Raw rows are never written into Git."
        ),
    )
    parser.add_argument("--skip-openpii", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = _endpoint()
    args.root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    if not args.skip_benchmark:
        sources["pii_bench_zh"] = fetch_pii_bench(args.root, endpoint)
    if not args.skip_openpii:
        sources["openpii_zh_cn"] = fetch_openpii(args.root, endpoint)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "hf_endpoint_configured": "HF_ENDPOINT" in os.environ,
        "sources": sources,
    }
    manifest_path = args.root / "manifests" / "public_sources.json"
    _atomic_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
