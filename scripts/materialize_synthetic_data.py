#!/usr/bin/env python3
"""Materialize the immutable synthetic-v1 public training corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.synthetic.materialize import materialize_synthetic_corpus  # noqa: E402

DEFAULT_OUTPUT = _REPOSITORY_ROOT / "data/processed/synthetic_v1_3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic Chinese PII train data, byte-copy the frozen "
            "v1.2 validation/test splits, and write a raw-text-free lineage manifest. "
            "Existing output directories are never overwritten."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--holdout-source-dir",
        type=Path,
        required=True,
        help="Frozen synthetic v1.2 directory; validation/test are hash-verified and byte-copied.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Local planning/provenance document recorded by hash; it is not distributed.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = materialize_synthetic_corpus(
        args.output_dir,
        plan_path=args.plan,
        holdout_source_dir=args.holdout_source_dir,
    )
    counts = manifest["counts"]
    print(
        f"dataset_id={manifest['dataset_id']} records={counts['total']} "
        f"hard_negative={counts['hard_negative']} "
        f"manifest_sha256={manifest['manifest_sha256']}"
    )
    print(f"split_counts={counts['by_split']}")


if __name__ == "__main__":
    main()
