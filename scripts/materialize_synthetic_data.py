#!/usr/bin/env python3
"""Materialize the immutable synthetic-v1 public training corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.synthetic.materialize import (  # noqa: E402
    DEFAULT_COUNT,
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_SEED,
    materialize_synthetic_corpus,
)

DEFAULT_OUTPUT = Path("/data1/datasets/pii-detect-model/processed/public_release_pool/synthetic_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic Chinese PII JSONL splits and a raw-text-free "
            "manifest. Existing output directories are never overwritten."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan", type=Path, default=_REPOSITORY_ROOT / "pii-detect-plan.md")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=DEFAULT_HARD_NEGATIVE_RATIO,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_synthetic_corpus(
        args.output_dir,
        plan_path=args.plan,
        count=args.count,
        seed=args.seed,
        hard_negative_ratio=args.hard_negative_ratio,
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
