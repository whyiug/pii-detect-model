#!/usr/bin/env python3
"""Build or strictly replay frozen release-eval-v2 prediction provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation.release_eval_v2_prediction_provenance import (  # noqa: E402
    AmendmentReplayInputs,
    CalibrationInputs,
    ModelRawInputs,
    PredictionProvenanceInputs,
    ReleaseEvalV2PredictionProvenanceError,
    build_release_eval_v2_prediction_manifest,
    replay_release_eval_v2_prediction_manifest,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track", required=True, choices=("model_raw", "model_calibrated", "full_system")
    )
    parser.add_argument(
        "--target-split", required=True, choices=("calibration", "internal_evaluation")
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--generation-receipt", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--v1-release-manifest", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--calibration-bundle", type=Path)
    parser.add_argument("--calibration-diagnostics", type=Path)
    parser.add_argument("--calibration-fit-gold", type=Path)
    parser.add_argument("--calibration-fit-predictions", type=Path)
    parser.add_argument("--calibration-fit-generation-receipt", type=Path)
    parser.add_argument("--calibration-fit-prediction-manifest", type=Path)
    parser.add_argument("--model-raw-predictions", type=Path)
    parser.add_argument("--model-raw-generation-receipt", type=Path)
    parser.add_argument("--model-raw-prediction-manifest", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a new read-only manifest")
    _common(build)
    build.add_argument("--output", required=True, type=Path)
    replay = subparsers.add_parser("replay", help="strictly replay an existing manifest")
    _common(replay)
    replay.add_argument("--manifest", required=True, type=Path)
    return parser


def _inputs(args: argparse.Namespace) -> PredictionProvenanceInputs:
    if len(args.candidate) != 3:
        raise ReleaseEvalV2PredictionProvenanceError(
            "exactly three frozen candidate roots are required"
        )
    calibration_values = (
        args.calibration_bundle,
        args.calibration_diagnostics,
        args.calibration_fit_gold,
        args.calibration_fit_predictions,
        args.calibration_fit_generation_receipt,
        args.calibration_fit_prediction_manifest,
    )
    if any(value is not None for value in calibration_values) and not all(
        value is not None for value in calibration_values
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "all calibration provenance arguments are required together"
        )
    model_raw_values = (
        args.model_raw_predictions,
        args.model_raw_generation_receipt,
        args.model_raw_prediction_manifest,
    )
    if any(value is not None for value in model_raw_values) and not all(
        value is not None for value in model_raw_values
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "all same-split model_raw arguments are required together"
        )
    calibration = (
        None
        if args.calibration_bundle is None
        else CalibrationInputs(
            bundle=args.calibration_bundle,
            diagnostics=args.calibration_diagnostics,
            fit_gold=args.calibration_fit_gold,
            fit_predictions=args.calibration_fit_predictions,
            fit_generation_receipt=args.calibration_fit_generation_receipt,
            fit_prediction_manifest=args.calibration_fit_prediction_manifest,
        )
    )
    model_raw = (
        None
        if args.model_raw_predictions is None
        else ModelRawInputs(
            predictions=args.model_raw_predictions,
            generation_receipt=args.model_raw_generation_receipt,
            prediction_manifest=args.model_raw_prediction_manifest,
        )
    )
    return PredictionProvenanceInputs(
        track=cast(Literal["model_raw", "model_calibrated", "full_system"], args.track),
        target_split=cast(Literal["calibration", "internal_evaluation"], args.target_split),
        predictions=args.predictions,
        generation_receipt=args.generation_receipt,
        gold=args.gold,
        dataset_manifest=args.dataset_manifest,
        materialization_receipt=args.materialization_receipt,
        freeze_receipt=args.freeze_receipt,
        model_artifact=args.model_artifact,
        selection_receipt=args.selection_receipt,
        amendment=AmendmentReplayInputs(
            amendment=args.amendment,
            protocol=args.protocol,
            development_manifest=args.development_manifest,
            v1_release_manifest=args.v1_release_manifest,
            candidate_roots=(args.candidate[0], args.candidate[1], args.candidate[2]),
        ),
        calibration=calibration,
        model_raw=model_raw,
    )


def _write_fresh(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ReleaseEvalV2PredictionProvenanceError("refusing to overwrite output")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        inputs = _inputs(args)
        if args.command == "build":
            manifest = build_release_eval_v2_prediction_manifest(inputs)
            _write_fresh(manifest, args.output)
            result = {
                "status": "PASS",
                "track": manifest["track"],
                "target_split": manifest["dataset"]["target_split"],
                "manifest_sha256": manifest["manifest_sha256"],
                "predictions_file_sha256": manifest["prediction"]["file_sha256"],
                "prediction_document_count": manifest["prediction"]["document_count"],
            }
        else:
            result = replay_release_eval_v2_prediction_manifest(args.manifest, inputs)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, ReleaseEvalV2PredictionProvenanceError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
