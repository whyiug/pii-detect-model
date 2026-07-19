#!/usr/bin/env python3
"""Build/replay release-eval-v2 calibration and internal pre-open gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation.release_eval_v2_prediction_provenance import (  # noqa: E402
    CalibrationInputs,
)
from pii_zh.evaluation.release_eval_v2_stage_gate import (  # noqa: E402
    CalibrationAuthorizationInputs,
    InternalUnlockInputs,
    ReleaseEvalV2StageGateError,
    SelectionReplayEvidence,
    build_calibration_authorization,
    build_internal_unlock,
    replay_calibration_authorization,
    replay_internal_unlock,
    write_read_only_json,
)


def _selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--v1-release-manifest", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    parser.add_argument("--calibration-gold", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)


def _calibration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-bundle", required=True, type=Path)
    parser.add_argument("--calibration-diagnostics", required=True, type=Path)
    parser.add_argument("--calibration-fit-predictions", required=True, type=Path)
    parser.add_argument("--calibration-fit-generation-receipt", required=True, type=Path)
    parser.add_argument("--calibration-fit-prediction-manifest", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_calibration = commands.add_parser("build-calibration-authorization")
    _selection_args(build_calibration)
    build_calibration.add_argument("--output", required=True, type=Path)

    replay_calibration = commands.add_parser("replay-calibration-authorization")
    _selection_args(replay_calibration)
    replay_calibration.add_argument("--authorization", required=True, type=Path)

    build_unlock = commands.add_parser("build-internal-unlock")
    _selection_args(build_unlock)
    _calibration_args(build_unlock)
    build_unlock.add_argument("--authorization", required=True, type=Path)
    build_unlock.add_argument("--model-binding-output", required=True, type=Path)
    build_unlock.add_argument("--service-binding-output", required=True, type=Path)
    build_unlock.add_argument("--unlock-output", required=True, type=Path)

    replay_unlock = commands.add_parser("replay-internal-unlock")
    _selection_args(replay_unlock)
    _calibration_args(replay_unlock)
    replay_unlock.add_argument("--authorization", required=True, type=Path)
    replay_unlock.add_argument("--model-binding", required=True, type=Path)
    replay_unlock.add_argument("--service-binding", required=True, type=Path)
    replay_unlock.add_argument("--unlock", required=True, type=Path)
    return parser


def _authorization_inputs(args: argparse.Namespace) -> CalibrationAuthorizationInputs:
    candidates = tuple(args.candidate)
    if len(candidates) != 3:
        raise ReleaseEvalV2StageGateError("exactly three candidate roots are required")
    return CalibrationAuthorizationInputs(
        evidence=SelectionReplayEvidence(
            protocol=args.protocol,
            development_manifest=args.development_manifest,
            v1_release_manifest=args.v1_release_manifest,
            candidates=(candidates[0], candidates[1], candidates[2]),
            selection_receipt=args.selection_receipt,
            amendment=args.amendment,
        ),
        dataset_manifest=args.dataset_manifest,
        materialization_receipt=args.materialization_receipt,
        freeze_receipt=args.freeze_receipt,
        calibration_gold=args.calibration_gold,
        model_artifact=args.model_artifact,
    )


def _unlock_inputs(args: argparse.Namespace) -> InternalUnlockInputs:
    return InternalUnlockInputs(
        authorization_inputs=_authorization_inputs(args),
        authorization_receipt=args.authorization,
        calibration=CalibrationInputs(
            bundle=args.calibration_bundle,
            diagnostics=args.calibration_diagnostics,
            fit_gold=args.calibration_gold,
            fit_predictions=args.calibration_fit_predictions,
            fit_generation_receipt=args.calibration_fit_generation_receipt,
            fit_prediction_manifest=args.calibration_fit_prediction_manifest,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-calibration-authorization":
            value = build_calibration_authorization(_authorization_inputs(args))
            file_sha256 = write_read_only_json(args.output, value)
            result = {
                "status": value["status"],
                "authorization_file_sha256": file_sha256,
                "authorization_sha256": value["receipt_sha256"],
                "internal_evaluation_opened_hashed_or_decoded": False,
            }
        elif args.command == "replay-calibration-authorization":
            value, file_sha256 = replay_calibration_authorization(
                args.authorization,
                _authorization_inputs(args),
            )
            result = {
                "status": "CALIBRATION_AUTHORIZATION_STRICT_REPLAY_PASS",
                "authorization_file_sha256": file_sha256,
                "authorization_sha256": value["receipt_sha256"],
                "internal_evaluation_opened_hashed_or_decoded": False,
            }
        elif args.command == "build-internal-unlock":
            model, service, unlock = build_internal_unlock(_unlock_inputs(args))
            model_file = write_read_only_json(args.model_binding_output, model)
            try:
                service_file = write_read_only_json(args.service_binding_output, service)
                unlock_file = write_read_only_json(args.unlock_output, unlock)
            except BaseException:
                # Published evidence is immutable; retain partial files so the
                # operator can audit and choose fresh output paths on retry.
                raise
            result = {
                "status": unlock["status"],
                "final_model_binding_file_sha256": model_file,
                "final_model_binding_sha256": model["manifest_sha256"],
                "service_configuration_binding_file_sha256": service_file,
                "service_configuration_binding_sha256": service["manifest_sha256"],
                "unlock_file_sha256": unlock_file,
                "unlock_sha256": unlock["receipt_sha256"],
                "internal_evaluation_opened_hashed_or_decoded": False,
            }
        else:
            result = replay_internal_unlock(
                args.unlock,
                args.model_binding,
                args.service_binding,
                _unlock_inputs(args),
            )
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, ReleaseEvalV2StageGateError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
