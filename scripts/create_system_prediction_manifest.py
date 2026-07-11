#!/usr/bin/env python3
"""Create a path-free manifest for fused, calibrated, and refined predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import (  # noqa: E402
    EvaluationDataError,
    build_system_prediction_manifest,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-document-count", type=int, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-training-manifest", type=Path, required=True)
    parser.add_argument("--model-predictions", type=Path, required=True)
    parser.add_argument("--model-prediction-manifest", type=Path, required=True)
    parser.add_argument("--rules-predictions", type=Path, required=True)
    parser.add_argument("--rules-prediction-manifest", type=Path, required=True)
    parser.add_argument("--fused-predictions", type=Path, required=True)
    parser.add_argument("--calibration-bundle", type=Path, required=True)
    parser.add_argument("--calibration-diagnostics", type=Path, required=True)
    parser.add_argument("--calibration-fit-gold", type=Path, required=True)
    parser.add_argument("--calibration-fit-predictions", type=Path, required=True)
    parser.add_argument("--calibrated-predictions", type=Path, required=True)
    parser.add_argument("--refinement-audit", type=Path, required=True)
    parser.add_argument("--refinement-implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    parser = _parser()
    args = parser.parse_args()
    input_paths = {
        value.expanduser().resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path) and name != "output"
    }
    output = args.output.expanduser().resolve()
    if output in input_paths:
        parser.error("output must differ from every input artifact")
    try:
        manifest = build_system_prediction_manifest(
            predictions_path=args.predictions,
            prediction_document_count=args.prediction_document_count,
            dataset_manifest_path=args.dataset_manifest,
            model_training_manifest_path=args.model_training_manifest,
            model_predictions_path=args.model_predictions,
            model_prediction_manifest_path=args.model_prediction_manifest,
            rules_predictions_path=args.rules_predictions,
            rules_prediction_manifest_path=args.rules_prediction_manifest,
            fused_predictions_path=args.fused_predictions,
            calibration_bundle_path=args.calibration_bundle,
            calibration_diagnostics_path=args.calibration_diagnostics,
            calibration_fit_gold_path=args.calibration_fit_gold,
            calibration_fit_predictions_path=args.calibration_fit_predictions,
            calibrated_predictions_path=args.calibrated_predictions,
            refinement_audit_path=args.refinement_audit,
            refinement_implementation_sha256=sha256_file(args.refinement_implementation),
        )
        _write_json_atomic(manifest, output)
    except (EvaluationDataError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "prediction_id": manifest["prediction_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
