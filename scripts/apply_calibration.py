#!/usr/bin/env python3
"""Apply frozen validation calibration to raw-text-free prediction JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.calibration import CalibrationBundle, apply_calibration  # noqa: E402
from pii_zh.evaluation import load_prediction_jsonl, write_prediction_jsonl  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.input.expanduser().resolve(strict=True)
    calibration_path = args.calibration.expanduser().resolve(strict=True)
    destination = args.output.expanduser().resolve()
    if destination in {source, calibration_path}:
        raise ValueError("output must differ from input and calibration")

    raw = load_prediction_jsonl(source)
    bundle = CalibrationBundle.from_json(calibration_path)
    calibrated = apply_calibration(raw, bundle)
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
            write_prediction_jsonl(calibrated, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.chmod(temporary, 0o444)
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "calibration_version": bundle.calibration_version,
                "documents": len(calibrated),
                "input_spans": sum(len(record.spans) for record in raw),
                "output": destination.name,
                "output_spans": sum(len(record.spans) for record in calibrated),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
