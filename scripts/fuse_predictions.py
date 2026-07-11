#!/usr/bin/env python3
"""Fuse raw-text-free rule and model prediction JSONL deterministically."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import load_prediction_jsonl, write_prediction_jsonl  # noqa: E402
from pii_zh.fusion import fuse_prediction_records  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-threshold", type=float, default=0.0)
    parser.add_argument(
        "--keep-nested-different-types",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = fuse_prediction_records(
        load_prediction_jsonl(args.rules),
        load_prediction_jsonl(args.model),
        model_threshold=args.model_threshold,
        keep_nested_different_types=args.keep_nested_different_types,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        delete=False,
        prefix=f".{args.output.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            write_prediction_jsonl(output, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, args.output)
    print(json.dumps({"documents": len(output), "output": args.output.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
