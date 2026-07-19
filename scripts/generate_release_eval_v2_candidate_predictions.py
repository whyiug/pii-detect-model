#!/usr/bin/env python3
"""Generate raw-text-free candidate predictions through the frozen local path.

This command is an execution surface, not an evaluator.  It supports raw model
diagnostics and the actual community model-only/cascade builder, writes a
content-bound aggregate generation receipt, and never prints input rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

# Make accidental Hub access fail even when the parent shell forgot its
# offline flags.  Every loader below additionally requires local files only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.cascade import (  # noqa: E402
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    build_community_model_service_pipeline,
)
from pii_zh.data.schema import iter_jsonl  # noqa: E402
from pii_zh.evaluation import PredictionRecord, Span, write_prediction_jsonl  # noqa: E402
from pii_zh.evaluation.release_eval_v2_prediction_provenance import (  # noqa: E402
    GenerationRuntime,
    build_release_eval_v2_generation_receipt,
)
from pii_zh.evaluation.release_eval_v2_stage_gate import (  # noqa: E402
    ReleaseEvalV2StageGateError,
    guard_calibration_preopen,
    guard_internal_preopen,
)
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS  # noqa: E402
from pii_zh.inference import CLOSED_8_TARGET_LABELS, load_local_predictor  # noqa: E402
from pii_zh.presidio import QwenPiiRecognizer  # noqa: E402
from pii_zh.taxonomy import load_presidio_mapping, load_taxonomy  # noqa: E402

MAX_TOKENS = 512
STRIDE_FRACTION = 0.25


class CandidateGenerationError(RuntimeError):
    """An aggregate-only setup/inference failure."""


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("model-raw", "model-only", "cascade"))
    parser.add_argument(
        "--target-split",
        required=True,
        choices=("fixture", "calibration", "internal_evaluation"),
        help="Explicit stage role; formal v2 roles require a pre-open gate.",
    )
    parser.add_argument("--scope", choices=("open24", "closed8"), default="open24")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-authorization", type=Path)
    parser.add_argument("--internal-unlock", type=Path)
    parser.add_argument("--final-model-binding", type=Path)
    parser.add_argument("--service-configuration-binding", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--document-batch-size", type=_positive, default=32)
    parser.add_argument("--micro-batch-size", type=_positive, default=16)
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_args(args: argparse.Namespace) -> None:
    outputs = (_resolved(args.predictions), _resolved(args.receipt))
    if len(set(outputs)) != 2 or any(path.exists() or path.is_symlink() for path in outputs):
        raise CandidateGenerationError("outputs must be distinct new files")
    inputs = {_resolved(args.input), _resolved(args.model)}
    if args.calibration is not None:
        inputs.add(_resolved(args.calibration))
    if any(output in inputs for output in outputs):
        raise CandidateGenerationError("an output collides with an input artifact")
    raw_model = args.model.expanduser()
    raw_source = args.input.expanduser()
    if raw_model.is_symlink() or raw_source.is_symlink():
        raise CandidateGenerationError("model/input must not be symlinks")
    try:
        model = raw_model.resolve(strict=True)
        source = raw_source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CandidateGenerationError("model/input must exist locally") from exc
    if not model.is_dir() or not source.is_file() or source.is_symlink():
        raise CandidateGenerationError("model/input types are invalid")
    args.model = model
    args.input = source
    gate_values = (
        args.internal_unlock,
        args.final_model_binding,
        args.service_configuration_binding,
    )
    if args.target_split == "fixture":
        if args.calibration_authorization is not None or any(
            value is not None for value in gate_values
        ):
            raise CandidateGenerationError("fixture generation must not claim a formal stage gate")
        args.stage_gate_identity = None
    elif args.target_split == "calibration":
        if args.calibration_authorization is None or any(
            value is not None for value in gate_values
        ):
            raise CandidateGenerationError(
                "calibration generation requires only the calibration authorization"
            )
        try:
            args.stage_gate_identity = guard_calibration_preopen(
                args.calibration_authorization,
                input_path=args.input,
                mode=args.mode,
                model_artifact=args.model,
            )
        except ReleaseEvalV2StageGateError as exc:
            raise CandidateGenerationError("calibration pre-open gate failed") from exc
    else:
        if args.calibration_authorization is not None or not all(
            value is not None for value in gate_values
        ):
            raise CandidateGenerationError(
                "internal generation requires unlock, model binding and service binding"
            )
        try:
            args.stage_gate_identity = guard_internal_preopen(
                args.internal_unlock,
                args.final_model_binding,
                args.service_configuration_binding,
                input_path=args.input,
                mode=args.mode,
            )
        except ReleaseEvalV2StageGateError as exc:
            raise CandidateGenerationError("internal pre-open gate failed") from exc
    if args.mode == "model-raw":
        if args.calibration is not None:
            raise CandidateGenerationError("model-raw must be threshold-zero and uncalibrated")
    else:
        if args.calibration is None:
            raise CandidateGenerationError("community model-only/cascade requires calibration")
        try:
            calibration = args.calibration.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CandidateGenerationError("calibration must exist locally") from exc
        if not calibration.is_file() or calibration.is_symlink():
            raise CandidateGenerationError("calibration must be a regular local file")
        args.calibration = calibration
    if args.device == "cuda:0":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        values = [item.strip() for item in visible.split(",") if item.strip()]
        if len(values) != 1:
            raise CandidateGenerationError(
                "CUDA mode requires exactly one CUDA_VISIBLE_DEVICES entry"
            )


def _snapshot_input(args: argparse.Namespace) -> Path:
    """Copy one stable input inode and bind inference to those exact bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source = os.open(args.input, flags)
    except OSError as exc:
        raise CandidateGenerationError("candidate input could not be snapshotted") from exc
    temporary: Path | None = None
    try:
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode):
            raise CandidateGenerationError("candidate input must be a regular file")
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=args.predictions.parent,
            delete=False,
            prefix=".release-eval-v2-input-snapshot.",
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            while block := os.read(source, 1024 * 1024):
                digest.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        after = os.fstat(source)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CandidateGenerationError("candidate input changed during snapshot")
        temporary.chmod(0o400)
        expected = (
            None
            if args.stage_gate_identity is None
            else args.stage_gate_identity["expected_input_sha256"]
        )
        if expected is not None and digest.hexdigest() != expected:
            raise CandidateGenerationError("candidate input bytes do not match the stage gate")
        return temporary
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(source)


def _scope_labels(scope: str) -> tuple[str, ...]:
    return tuple(PII_CORE_LABELS) if scope == "open24" else tuple(CLOSED_8_TARGET_LABELS)


def _model_raw_recognizer(args: argparse.Namespace, labels: Sequence[str]) -> QwenPiiRecognizer:
    import torch

    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise CandidateGenerationError("the isolated CUDA device is unavailable")
    dtype = torch.bfloat16 if args.device == "cuda:0" else None
    taxonomy = load_taxonomy()
    ignored = [entity.name for entity in taxonomy.label_sets["auxiliary"]]
    try:
        predictor = load_local_predictor(
            args.model,
            device=args.device,
            dtype=dtype,
            micro_batch_size=args.micro_batch_size,
            ignored_labels=ignored,
            allowed_labels=labels,
        )
        return QwenPiiRecognizer(
            predictor=predictor,
            tokenizer=predictor.tokenizer,
            label_mapping={label: label for label in labels},
            ignored_labels=ignored,
            default_threshold=0.0,
            max_tokens=MAX_TOKENS,
            stride_fraction=STRIDE_FRACTION,
            model_version="release-eval-v2-local-candidate",
            attention_mode=predictor.attention_mode,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CandidateGenerationError("local model-raw initialization failed") from exc


def _generate_model_raw(
    args: argparse.Namespace,
    *,
    labels: Sequence[str],
    temporary: Path,
) -> int:
    recognizer = _model_raw_recognizer(args, labels)

    def detect(texts: Sequence[str]) -> list[list[Any]]:
        return recognizer.analyze_batch(texts, entities=labels)

    return _write_batches(args, temporary=temporary, detect=detect, output_mapping=None)


def _generate_service(
    args: argparse.Namespace,
    *,
    labels: Sequence[str],
    temporary: Path,
) -> int:
    mode = cast(str, args.mode)
    dtype = None
    if args.device == "cuda:0":
        import torch

        if not torch.cuda.is_available():
            raise CandidateGenerationError("the isolated CUDA device is unavailable")
        dtype = torch.bfloat16
    try:
        pipeline = build_community_model_service_pipeline(
            args.model,
            mode=cast(Any, mode),
            device=args.device,
            dtype=dtype,
            micro_batch_size=args.micro_batch_size,
            calibration=args.calibration,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CandidateGenerationError("community service initialization failed") from exc
    if (
        pipeline.config.profile_version != COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
        or pipeline.config.mode != mode
    ):
        raise CandidateGenerationError("community builder returned a different profile/mode")
    mapping = load_presidio_mapping()
    requested = [mapping.model_to_presidio[label] for label in labels]
    output_mapping = {value: key for key, value in mapping.model_to_presidio.items()}

    def detect(texts: Sequence[str]) -> list[list[Any]]:
        return pipeline.detect_batch(texts, entities=requested)

    return _write_batches(
        args,
        temporary=temporary,
        detect=detect,
        output_mapping=output_mapping,
    )


def _write_batches(
    args: argparse.Namespace,
    *,
    temporary: Path,
    detect: Any,
    output_mapping: Mapping[str, str] | None,
) -> int:
    seen: set[str] = set()
    count = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def flush(handle: Any) -> None:
        nonlocal count
        if not batch_texts:
            return
        try:
            detected = detect(batch_texts)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CandidateGenerationError("candidate inference failed") from exc
        predictions: list[PredictionRecord] = []
        for doc_id, spans in zip(batch_ids, detected, strict=True):
            predictions.append(
                PredictionRecord(
                    doc_id=doc_id,
                    spans=tuple(
                        Span(
                            start=item.start,
                            end=item.end,
                            label=(
                                item.entity_type
                                if output_mapping is None
                                else output_mapping[item.entity_type]
                            ),
                            score=float(item.score),
                        )
                        for item in spans
                    ),
                )
            )
        count += write_prediction_jsonl(predictions, handle)
        batch_ids.clear()
        batch_texts.clear()

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in iter_jsonl(args.input):
                if record.doc_id in seen:
                    raise CandidateGenerationError("canonical input contains duplicate IDs")
                seen.add(record.doc_id)
                batch_ids.append(record.doc_id)
                batch_texts.append(record.text)
                if len(batch_texts) >= args.document_batch_size:
                    flush(handle)
            flush(handle)
    except CandidateGenerationError:
        raise
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise CandidateGenerationError("canonical input/prediction serialization failed") from exc
    if count < 1:
        raise CandidateGenerationError("canonical input is empty")
    return count


def _publish_json(payload: Mapping[str, Any], destination: Path) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    temporary: Path | None = None
    input_snapshot: Path | None = None
    published_prediction = False
    try:
        _validate_args(args)
        input_snapshot = _snapshot_input(args)
        args.input = input_snapshot
        labels = _scope_labels(args.scope)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=args.predictions.parent,
            delete=False,
            prefix=f".{args.predictions.name}.",
        ) as handle:
            temporary = Path(handle.name)
        if args.mode == "model-raw":
            count = _generate_model_raw(args, labels=labels, temporary=temporary)
            track = "model_raw"
        else:
            count = _generate_service(args, labels=labels, temporary=temporary)
            track = "model_calibrated" if args.mode == "model-only" else "full_system"
        os.link(temporary, args.predictions)
        args.predictions.chmod(0o444)
        published_prediction = True
        visible_count = 1 if args.device == "cuda:0" else 0
        receipt = build_release_eval_v2_generation_receipt(
            track=cast(Any, track),
            input_path=args.input,
            predictions_path=args.predictions,
            model_artifact=args.model,
            calibration_bundle=args.calibration,
            runtime=GenerationRuntime(
                document_batch_size=args.document_batch_size,
                micro_batch_size=args.micro_batch_size,
                max_tokens=MAX_TOKENS,
                stride_fraction=STRIDE_FRACTION,
                device_class="cuda" if args.device == "cuda:0" else "cpu",
                dtype="bfloat16" if args.device == "cuda:0" else "float32",
                visible_cuda_device_count=visible_count,
                scope=args.scope,
                target_split=args.target_split,
            ),
            stage_gate=args.stage_gate_identity,
        )
        _publish_json(receipt, args.receipt)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "documents": count,
                    "track": track,
                    "predictions_sha256": receipt["output"]["file_sha256"],
                    "generation_receipt_sha256": receipt["receipt_sha256"],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    except (CandidateGenerationError, OSError, TypeError, ValueError) as exc:
        if published_prediction:
            try:
                args.predictions.chmod(0o600)
                args.predictions.unlink()
            except OSError:
                pass
        parser.error(str(exc))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if input_snapshot is not None:
            input_snapshot.unlink(missing_ok=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
