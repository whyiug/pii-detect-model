"""Local-only command line interface for the Chinese PII cascade."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO, cast

from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    DEFAULT_SERVICE_PROFILE_VERSION,
    SERVICE_PROFILE_VERSIONS,
    CascadeDetection,
    CascadeMode,
    CascadePipeline,
    build_community_model_service_pipeline,
    build_rules_only_service_pipeline,
)
from pii_zh.full_bie73 import (
    COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
    FULL_BIE73_DEFAULT_SCOPE,
    FullBie73Scope,
    build_full_bie73_service_pipeline,
)

CLI_OUTPUT_SCHEMA_VERSION = "pii-zh.cli.output.v1"
MAX_JSONL_RECORDS = 10_000
MAX_JSONL_TEXT_CHARS = 100_000
DEFAULT_CPU_MAX_CONCURRENCY = 4
DEFAULT_CUDA_MAX_CONCURRENCY = 1
CLI_SERVICE_PROFILE_VERSIONS = (
    *SERVICE_PROFILE_VERSIONS,
    COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
)
_ENTITY_LABEL_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")


def _local_directory(value: str) -> Path:
    """Resolve an existing local directory without accepting Hub identifiers."""

    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise argparse.ArgumentTypeError("model path must be an existing local directory") from exc
    if not path.is_dir():
        raise argparse.ArgumentTypeError("model path must be an existing local directory")
    return path


def _local_file(value: str) -> Path:
    """Resolve one regular local input file without following a symlink."""

    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise argparse.ArgumentTypeError("input file must be a regular local file")
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise argparse.ArgumentTypeError("input file must be a regular local file") from exc
    if not path.is_file():
        raise argparse.ArgumentTypeError("input file must be a regular local file")
    return path


def _loopback_host(value: str) -> str:
    """Keep the packaged service command local by construction."""

    if value not in {"127.0.0.1", "::1", "localhost"}:
        raise argparse.ArgumentTypeError("the packaged serve command only binds to a loopback host")
    return value


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _threshold_override(value: str) -> tuple[str, float]:
    label, separator, raw_threshold = value.partition("=")
    if not separator or _ENTITY_LABEL_RE.fullmatch(label) is None:
        raise argparse.ArgumentTypeError("threshold must use LABEL=0..1")
    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must use LABEL=0..1") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must use LABEL=0..1")
    return label, threshold


def _thresholds(args: argparse.Namespace) -> dict[str, float] | None:
    entries = cast(list[tuple[str, float]] | None, getattr(args, "thresholds", None))
    if entries is None:
        return None
    result: dict[str, float] = {}
    for label, threshold in entries:
        if label in result:
            raise ValueError(f"--threshold repeats entity label {label}")
        result[label] = threshold
    return result


def _default_max_concurrency(device: str) -> int:
    return (
        DEFAULT_CUDA_MAX_CONCURRENCY
        if device.strip().casefold().startswith("cuda")
        else DEFAULT_CPU_MAX_CONCURRENCY
    )


def _shared_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--text",
        help="input text; when omitted, read the complete UTF-8 text from stdin",
    )
    source.add_argument(
        "--input-file",
        type=_local_file,
        help="read one local UTF-8 text or JSONL file",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help=(
            "treat stdin or --input-file as JSONL objects containing only a text field; "
            "emit one compact JSON object per record"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("rules-only", "model-only", "cascade"),
        default=None,
        help=(
            "recognizer mode (default: cascade for community-full-bie73-cascade-v1; "
            "rules-only otherwise); BIE73 model-only is a validated service ablation"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=CLI_SERVICE_PROFILE_VERSIONS,
        default=DEFAULT_SERVICE_PROFILE_VERSION,
        help=(
            "versioned detection profile (default: c1-conservative-v1); "
            "the selected BIE73 service profile has its own frozen model defaults"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=_local_directory,
        help="existing local model directory; required by model-only and cascade",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="local inference device passed to the model loader (default: cpu)",
    )
    parser.add_argument(
        "--scope",
        choices=("open24", "closed8"),
        help=(
            "BIE73 pre-decode label scope "
            f"(default for its service profile: {FULL_BIE73_DEFAULT_SCOPE})"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=_local_file,
        help="optional calibration JSON for the legacy community model profile",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=_threshold_override,
        dest="thresholds",
        metavar="LABEL=VALUE",
        help="legacy community model threshold; may be repeated",
    )
    parser.add_argument(
        "--entity",
        action="append",
        dest="entities",
        metavar="LABEL",
        help="restrict output to a configured entity label; may be repeated",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent JSON output for human inspection",
    )
    return parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-zh",
        description=(
            "Detect or redact Simplified Chinese PII locally. No command downloads a model."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    shared = _shared_parser()
    commands.add_parser(
        "detect",
        parents=[shared],
        help="emit raw-text-free detection spans as JSON",
    )
    redact = commands.add_parser(
        "redact",
        parents=[shared],
        help="replace detected spans and emit the redacted text as JSON",
    )
    redact.add_argument(
        "--replacement",
        default="<PII>",
        help="replacement string used for every selected span (default: <PII>)",
    )
    serve = commands.add_parser(
        "serve",
        help="run the bounded local HTTP API without access logging",
    )
    serve.add_argument(
        "--mode",
        choices=("rules-only", "model-only", "cascade"),
        default=None,
        help=(
            "recognizer mode (default: cascade for community-full-bie73-cascade-v1; "
            "rules-only otherwise); BIE73 model-only is a validated service ablation"
        ),
    )
    serve.add_argument(
        "--profile",
        choices=CLI_SERVICE_PROFILE_VERSIONS,
        default=DEFAULT_SERVICE_PROFILE_VERSION,
        help=(
            "versioned detection profile (default: c1-conservative-v1); "
            "the selected BIE73 service profile has its own frozen model defaults"
        ),
    )
    serve.add_argument(
        "--model-path",
        type=_local_directory,
        help="existing local model directory; required by model-only and cascade",
    )
    serve.add_argument(
        "--device",
        default="cpu",
        help="local inference device passed to the model loader (default: cpu)",
    )
    serve.add_argument(
        "--scope",
        choices=("open24", "closed8"),
        help=(
            "BIE73 pre-decode label scope "
            f"(default for its service profile: {FULL_BIE73_DEFAULT_SCOPE})"
        ),
    )
    serve.add_argument(
        "--micro-batch-size",
        type=_positive_integer,
        default=16,
        help="bounded model inference micro-batch size (default: 16)",
    )
    serve.add_argument(
        "--calibration",
        type=_local_file,
        help="optional calibration JSON for the legacy community model profile",
    )
    serve.add_argument(
        "--threshold",
        action="append",
        type=_threshold_override,
        dest="thresholds",
        metavar="LABEL=VALUE",
        help="legacy community model threshold; may be repeated",
    )
    serve.add_argument(
        "--max-concurrency",
        type=_positive_integer,
        help="request concurrency (default: CUDA 1, CPU 4)",
    )
    serve.add_argument(
        "--host",
        type=_loopback_host,
        default="127.0.0.1",
        help="loopback bind host (default: 127.0.0.1)",
    )
    serve.add_argument(
        "--port",
        type=_port,
        default=8000,
        help="local HTTP port (default: 8000)",
    )
    return parser


def _pipeline(args: argparse.Namespace) -> CascadePipeline:
    requested_mode = cast(CascadeMode | None, args.mode)
    model_path = cast(Path | None, args.model_path)
    profile = cast(str, args.profile)
    mode = (
        "cascade"
        if requested_mode is None and profile == COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION
        else "rules-only"
        if requested_mode is None
        else requested_mode
    )
    args.mode = mode
    scope = cast(FullBie73Scope | None, getattr(args, "scope", None))
    calibration = cast(Path | None, getattr(args, "calibration", None))
    thresholds = _thresholds(args)
    if profile == COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION:
        if mode == "rules-only":
            raise ValueError(
                f"{COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION} supports only "
                "model-only or cascade mode"
            )
        if model_path is None:
            raise ValueError(f"--model-path is required in {mode} mode")
        if calibration is not None or thresholds is not None:
            raise ValueError(
                "the stable BIE73 CLI uses its frozen thresholds and does not accept "
                "calibration or threshold overrides"
            )
        return build_full_bie73_service_pipeline(
            model_path,
            scope=FULL_BIE73_DEFAULT_SCOPE if scope is None else scope,
            mode=mode,
            device=cast(str, args.device),
            micro_batch_size=cast(int, getattr(args, "micro_batch_size", 16)),
        )
    if scope is not None:
        raise ValueError(
            f"--scope requires --profile {COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION}"
        )
    if mode == "rules-only":
        if model_path is not None:
            raise ValueError("--model-path is only valid in model-only or cascade mode")
        if calibration is not None or thresholds is not None:
            raise ValueError("calibration and thresholds require a model-enabled mode")
        return build_rules_only_service_pipeline(profile)
    if profile != COMMUNITY_MODEL_SERVICE_PROFILE_VERSION:
        raise ValueError(
            "the selected profile can only support rules-only mode; "
            f"{mode} mode requires --profile {COMMUNITY_MODEL_SERVICE_PROFILE_VERSION}"
        )
    if model_path is None:
        raise ValueError(f"--model-path is required in {mode} mode")
    options: dict[str, object] = {
        "mode": mode,
        "device": cast(str, args.device),
        "micro_batch_size": cast(int, getattr(args, "micro_batch_size", 16)),
    }
    if calibration is not None:
        options["calibration"] = calibration
    if thresholds is not None:
        options["thresholds"] = thresholds
    return build_community_model_service_pipeline(model_path, **options)  # type: ignore[arg-type]


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError:
        raise ValueError("input file must be valid UTF-8") from None
    except OSError:
        raise ValueError("input file could not be read") from None


def _input_text(args: argparse.Namespace, stdin: TextIO) -> str:
    value = cast(str | None, args.text)
    path = cast(Path | None, args.input_file)
    if value is not None:
        return value
    if path is not None:
        return _read_utf8(path)
    return stdin.read()


def _jsonl_texts(args: argparse.Namespace, stdin: TextIO) -> list[str]:
    if cast(str | None, args.text) is not None:
        raise ValueError("--jsonl cannot be combined with --text")
    path = cast(Path | None, args.input_file)
    content = _read_utf8(path) if path is not None else stdin.read()
    lines = content.splitlines()
    if not lines:
        raise ValueError("JSONL input must contain at least one record")
    if len(lines) > MAX_JSONL_RECORDS:
        raise ValueError(f"JSONL input exceeds the {MAX_JSONL_RECORDS}-record limit")
    texts: list[str] = []
    for record_index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"JSONL record {record_index} must not be blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"JSONL record {record_index} is invalid JSON") from None
        if not isinstance(value, dict) or set(value) != {"text"}:
            raise ValueError(f"JSONL record {record_index} must be an object containing only text")
        text = value["text"]
        if not isinstance(text, str):
            raise ValueError(f"JSONL record {record_index} text must be a string")
        if len(text) > MAX_JSONL_TEXT_CHARS:
            raise ValueError(f"JSONL record {record_index} exceeds the text-length limit")
        texts.append(text)
    return texts


def _write_json(payload: Mapping[str, object], *, pretty: bool, stdout: TextIO) -> None:
    indent = 2 if pretty else None
    separators = None if pretty else (",", ":")
    stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )
    )
    stdout.write("\n")


def _detections_payload(
    detections: Sequence[CascadeDetection],
    *,
    mode: str,
    profile_version: str,
    model_identity: Mapping[str, object] | None,
    record_index: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "mode": mode,
        "profile_version": profile_version,
        "model_identity": dict(model_identity) if model_identity is not None else None,
        "detections": [detection.to_dict() for detection in detections],
    }
    if record_index is not None:
        payload["record_index"] = record_index
    return payload


def _redact_payload(
    pipeline: CascadePipeline,
    text: str,
    *,
    mode: str,
    profile_version: str,
    model_identity: Mapping[str, object] | None,
    entities: Sequence[str] | None,
    replacement: str,
    record_index: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "mode": mode,
        "profile_version": profile_version,
        "model_identity": dict(model_identity) if model_identity is not None else None,
        "redacted_text": pipeline.redact(
            text,
            replacement,
            entities=entities,
        ),
    }
    if record_index is not None:
        payload["record_index"] = record_index
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        pipeline = _pipeline(args)
        command = cast(str, args.command)
        profile_version = cast(str, args.profile)
        raw_model_identity = getattr(pipeline, "model_identity", None)
        model_identity = (
            cast(Mapping[str, object], raw_model_identity)
            if isinstance(raw_model_identity, Mapping)
            else None
        )
        if command == "serve":
            from pii_zh.service import run

            requested_concurrency = cast(int | None, args.max_concurrency)
            run(
                pipeline=pipeline,
                host=cast(str, args.host),
                port=cast(int, args.port),
                max_concurrency=(
                    requested_concurrency
                    if requested_concurrency is not None
                    else _default_max_concurrency(cast(str, args.device))
                ),
            )
            return 0
        entities = cast(list[str] | None, args.entities)
        mode = cast(str, args.mode)
        replacement = cast(str, getattr(args, "replacement", "<PII>"))
        if cast(bool, args.jsonl):
            if cast(bool, args.pretty):
                raise ValueError("--pretty is not supported with --jsonl")
            texts = _jsonl_texts(args, sys.stdin)
            if command == "detect":
                batches = pipeline.detect_batch(texts, entities=entities)
                payloads = [
                    _detections_payload(
                        detections,
                        mode=mode,
                        profile_version=profile_version,
                        model_identity=model_identity,
                        record_index=index,
                    )
                    for index, detections in enumerate(batches)
                ]
            else:
                payloads = [
                    _redact_payload(
                        pipeline,
                        text,
                        mode=mode,
                        profile_version=profile_version,
                        model_identity=model_identity,
                        entities=entities,
                        replacement=replacement,
                        record_index=index,
                    )
                    for index, text in enumerate(texts)
                ]
            for payload in payloads:
                _write_json(payload, pretty=False, stdout=sys.stdout)
        else:
            text = _input_text(args, sys.stdin)
            payload = (
                _detections_payload(
                    pipeline.detect(text, entities=entities),
                    mode=mode,
                    profile_version=profile_version,
                    model_identity=model_identity,
                )
                if command == "detect"
                else _redact_payload(
                    pipeline,
                    text,
                    mode=mode,
                    profile_version=profile_version,
                    model_identity=model_identity,
                    entities=entities,
                    replacement=replacement,
                )
            )
            _write_json(
                payload,
                pretty=cast(bool, args.pretty),
                stdout=sys.stdout,
            )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"pii-zh: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - console script is the normal entry point
    raise SystemExit(main())


__all__ = [
    "CLI_SERVICE_PROFILE_VERSIONS",
    "CLI_OUTPUT_SCHEMA_VERSION",
    "MAX_JSONL_RECORDS",
    "MAX_JSONL_TEXT_CHARS",
    "main",
]
