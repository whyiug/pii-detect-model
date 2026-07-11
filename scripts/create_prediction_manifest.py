#!/usr/bin/env python3
"""Create a deterministic manifest that binds predictions to model and data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import (  # noqa: E402
    add_manifest_hash,
    sha256_file,
    verified_manifest_hash,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-training-manifest", type=Path, required=True)
    parser.add_argument("--prediction-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if _SAFE_ID.fullmatch(args.prediction_id) is None:
        raise ValueError("prediction-id must be a path-free stable identifier")
    manifest = add_manifest_hash(
        {
            "schema_version": 1,
            "manifest_type": "prediction",
            "prediction_id": args.prediction_id,
            "predictions_sha256": sha256_file(args.predictions),
            "dataset_manifest_sha256": verified_manifest_hash(
                args.dataset_manifest, kind="dataset"
            ),
            "model_training_manifest_sha256": verified_manifest_hash(
                args.model_training_manifest, kind="model training"
            ),
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
            },
        }
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
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, args.output)
    args.output.chmod(0o444)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "prediction_id": args.prediction_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
