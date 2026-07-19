#!/usr/bin/env python3
"""Measure the published rules-only cascade without retaining input content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on supported Linux releases
    resource = None  # type: ignore[assignment]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

# This entry point has no model option.  These flags additionally fail closed if
# a future import accidentally grows a Hub-aware code path.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pii_zh.cascade import CascadeConfig, CascadePipeline  # noqa: E402
from pii_zh.data.schema import SchemaError, iter_jsonl  # noqa: E402

SCHEMA_VERSION = "pii-zh.cascade-resource-benchmark.v1"
BUILTIN_FIXTURE_REVISION = "pii-zh.rules-only-synthetic-smoke.v1"
_BUILTIN_SYNTHETIC_TEXTS = (
    "测试邮箱 demo@example.com，测试地址 203.0.113.7。",
    "备用邮箱 qa@example.org；文档网段地址 198.51.100.8。",
    "本行是无敏感信息的中文合成负例。",
    "联系地址 support@example.net，服务端为 2001:db8::8。",
)
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_IMPLEMENTATION_FILES = (
    "scripts/benchmark_cascade.py",
    "src/pii_zh/cascade/__init__.py",
    "src/pii_zh/cascade/config.py",
    "src/pii_zh/cascade/context.py",
    "src/pii_zh/cascade/pipeline.py",
    "src/pii_zh/cascade/result.py",
    "src/pii_zh/cascade/routing.py",
    "src/pii_zh/data/validators/__init__.py",
    "src/pii_zh/fusion/__init__.py",
    "src/pii_zh/fusion/deterministic.py",
    "src/pii_zh/fusion/predictions.py",
    "src/pii_zh/fusion/refinement.py",
    "src/pii_zh/fusion/replacement.py",
    "src/pii_zh/rules/__init__.py",
    "src/pii_zh/rules/cn_common.py",
)


class SafeBenchmarkError(RuntimeError):
    """An allowlisted error whose message contains no input content or paths."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline, CPU/rules-only resource diagnostic through the published "
            "CascadePipeline. The aggregate report is not a release performance claim."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Existing local canonical JSONL")
    source.add_argument(
        "--builtin-synthetic",
        action="store_true",
        help="Use the bundled non-production synthetic smoke workload",
    )
    parser.add_argument("--output", required=True, type=Path, help="New aggregate JSON report")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3, help="Untimed full-corpus warmup passes")
    parser.add_argument("--repetitions", type=int, default=10, help="Timed full-corpus passes")
    parser.add_argument("--max-documents", type=int, default=100_000)
    parser.add_argument("--max-characters", type=int, default=100_000_000)
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(serialized)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.batch_size > 100_000:
        raise SafeBenchmarkError("batch size must be between 1 and 100000")
    if args.warmup < 0 or args.warmup > 10_000:
        raise SafeBenchmarkError("warmup must be between 0 and 10000")
    if args.repetitions < 1 or args.repetitions > 100_000:
        raise SafeBenchmarkError("repetitions must be between 1 and 100000")
    if args.max_documents < 1 or args.max_characters < 1:
        raise SafeBenchmarkError("input limits must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise SafeBenchmarkError("refusing to overwrite an existing output")
    if args.input is None:
        return
    if args.input.is_symlink() or not args.input.is_file():
        raise SafeBenchmarkError("canonical input must be an existing local regular file")
    try:
        size = args.input.stat().st_size
    except OSError as exc:
        raise SafeBenchmarkError("canonical input metadata could not be read") from exc
    if size > _MAX_INPUT_BYTES:
        raise SafeBenchmarkError("canonical input exceeds the fixed byte safety limit")
    if _resolved(args.input) == _resolved(args.output):
        raise SafeBenchmarkError("output must not collide with canonical input")


def _load_workload(args: argparse.Namespace) -> tuple[tuple[str, ...], dict[str, object]]:
    if args.input is None:
        texts = _BUILTIN_SYNTHETIC_TEXTS
        identity: dict[str, object] = {
            "source_kind": "builtin_synthetic",
            "fixture_revision": BUILTIN_FIXTURE_REVISION,
            "content_sha256": _canonical_sha256(list(texts)),
        }
    else:
        try:
            texts = tuple(record.text for record in iter_jsonl(args.input))
            content_sha256 = _sha256_file(args.input)
        except (OSError, SchemaError, TypeError, UnicodeError, ValueError) as exc:
            raise SafeBenchmarkError("canonical input validation failed") from exc
        identity = {
            "source_kind": "local_canonical_jsonl",
            "content_sha256": content_sha256,
        }
    if not texts:
        raise SafeBenchmarkError("benchmark workload must contain at least one document")
    character_count = sum(len(text) for text in texts)
    if len(texts) > args.max_documents or character_count > args.max_characters:
        raise SafeBenchmarkError("benchmark workload exceeds configured aggregate limits")
    identity.update(
        {
            "document_count": len(texts),
            "character_count": character_count,
            "contains_paths": False,
            "contains_document_ids": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
        }
    )
    return texts, identity


def _config_payload(config: CascadeConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
        "routes": [asdict(route) for route in config.routes],
    }


def _implementation_identity() -> dict[str, object]:
    try:
        components = {
            relative: _sha256_file(_REPOSITORY_ROOT / relative)
            for relative in _IMPLEMENTATION_FILES
        }
    except OSError as exc:
        raise SafeBenchmarkError("runtime implementation binding failed") from exc
    return {
        "component_sha256": components,
        "implementation_sha256": _canonical_sha256(components),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile needs at least one measurement")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary_ms(values_ns: Sequence[int]) -> dict[str, int | float]:
    values_ms = [value / 1_000_000 for value in values_ns]
    return {
        "sample_count": len(values_ms),
        "p50": round(_percentile(values_ms, 0.50), 6),
        "p95": round(_percentile(values_ms, 0.95), 6),
    }


def _batches(texts: Sequence[str], batch_size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)
    )


def _run_pass(
    pipeline: CascadePipeline,
    batches: Sequence[Sequence[str]],
    *,
    collect_call_latency: bool,
) -> tuple[int, tuple[int, ...], int]:
    call_latencies: list[int] = []
    emitted_spans = 0
    pass_start = time.perf_counter_ns()
    for batch in batches:
        call_start = time.perf_counter_ns()
        try:
            detections = pipeline.detect_batch(batch)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise SafeBenchmarkError("rules-only cascade execution failed") from exc
        call_end = time.perf_counter_ns()
        if len(detections) != len(batch):
            raise SafeBenchmarkError("rules-only cascade returned an invalid batch")
        emitted_spans += sum(len(document) for document in detections)
        if collect_call_latency:
            call_latencies.append(call_end - call_start)
    return time.perf_counter_ns() - pass_start, tuple(call_latencies), emitted_spans


def _max_rss() -> dict[str, object]:
    if resource is None:  # pragma: no cover - supported release environment is Linux
        return {
            "status": "unavailable",
            "bytes": None,
            "reason": "standard resource accounting is unavailable on this platform",
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {
        "status": "available",
        "bytes": int(usage.ru_maxrss * multiplier),
        "scope": "process_high_water_mark_including_interpreter_and_imports",
    }


def _cpu_times() -> dict[str, float] | None:
    if resource is None:  # pragma: no cover - supported release environment is Linux
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_seconds": round(float(usage.ru_utime), 6),
        "system_seconds": round(float(usage.ru_stime), 6),
    }


def _environment() -> dict[str, object]:
    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = None
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": affinity_count,
        "timer": "time.perf_counter_ns",
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }


def _build_report(args: argparse.Namespace) -> dict[str, object]:
    texts, workload = _load_workload(args)
    batches = _batches(texts, args.batch_size)
    config = CascadeConfig(mode="rules-only")
    config_payload = _config_payload(config)
    rss_before = _max_rss()
    cpu_before = _cpu_times()

    initialization_start = time.perf_counter_ns()
    pipeline = CascadePipeline(config=config)
    initialization_ns = time.perf_counter_ns() - initialization_start

    warmup_start = time.perf_counter_ns()
    warmup_spans = 0
    for _ in range(args.warmup):
        _, _, emitted = _run_pass(pipeline, batches, collect_call_latency=False)
        warmup_spans += emitted
    warmup_ns = time.perf_counter_ns() - warmup_start

    pass_latencies: list[int] = []
    call_latencies: list[int] = []
    measured_spans = 0
    for _ in range(args.repetitions):
        pass_ns, calls_ns, emitted = _run_pass(
            pipeline,
            batches,
            collect_call_latency=True,
        )
        pass_latencies.append(pass_ns)
        call_latencies.extend(calls_ns)
        measured_spans += emitted

    rss_after = _max_rss()
    cpu_after = _cpu_times()
    measured_ns = sum(pass_latencies)
    measured_seconds = measured_ns / 1_000_000_000
    measured_documents = len(texts) * args.repetitions
    measured_characters = int(workload["character_count"]) * args.repetitions
    batch_sizes = [len(batch) for _ in range(args.repetitions) for batch in batches]
    per_document_ns = [
        latency // batch_count
        for latency, batch_count in zip(call_latencies, batch_sizes, strict=True)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "claim_scope": {
            "release_performance_conclusion": False,
            "production_capacity_conclusion": False,
            "model_performance_conclusion": False,
            "reason": (
                "local workload and single-process rules-only measurements are not a "
                "controlled release performance study"
            ),
        },
        "runtime": {
            "mode": "rules-only",
            "device": "cpu",
            "gpu_used": False,
            "model_loaded": False,
            "remote_input_supported": False,
            "network_policy": "offline",
            "profile_version": config.profile_version,
            "configuration_sha256": _canonical_sha256(config_payload),
            "configuration": config_payload,
            "implementation": _implementation_identity(),
        },
        "workload": workload,
        "parameters": {
            "batch_size": args.batch_size,
            "warmup_passes": args.warmup,
            "repetitions": args.repetitions,
            "max_documents": args.max_documents,
            "max_characters": args.max_characters,
        },
        "measurements": {
            "startup": {
                "cold_process_start": {
                    "status": "unavailable",
                    "milliseconds": None,
                    "reason": (
                        "reliable process cold-start measurement requires an external "
                        "supervisor and controlled filesystem cache state"
                    ),
                },
                "first_pipeline_initialization": {
                    "status": "available",
                    "milliseconds": round(initialization_ns / 1_000_000, 6),
                    "scope": "in_process_rules_only_pipeline_construction",
                },
            },
            "warmup": {
                "passes": args.warmup,
                "documents": len(texts) * args.warmup,
                "elapsed_milliseconds": round(warmup_ns / 1_000_000, 6),
                "emitted_spans": warmup_spans,
                "included_in_hot_statistics": False,
            },
            "hot_path": {
                "status": "available",
                "definition": "post_warmup_full_corpus_CascadePipeline.detect_batch",
                "repetitions": args.repetitions,
                "pipeline_call_count": len(call_latencies),
                "document_count": measured_documents,
                "character_count": measured_characters,
                "emitted_spans": measured_spans,
                "latency_milliseconds": {
                    "per_batch_call": _summary_ms(call_latencies),
                    "per_document_normalized": _summary_ms(per_document_ns),
                    "per_full_corpus_pass": _summary_ms(pass_latencies),
                },
                "throughput": {
                    "docs_per_second": round(measured_documents / measured_seconds, 6),
                    "chars_per_second": round(measured_characters / measured_seconds, 6),
                    "elapsed_seconds": round(measured_seconds, 9),
                },
            },
            "model_call_rate": {
                "numerator": 0,
                "denominator_documents": measured_documents,
                "calls_per_document": 0.0,
                "basis": "CascadeConfig(mode=rules-only) has uses_model=false",
            },
            "cpu": {
                "max_rss_before": rss_before,
                "max_rss_after": rss_after,
                "cpu_times_before": cpu_before,
                "cpu_times_after": cpu_after,
            },
        },
        "environment": _environment(),
        "privacy": {
            "aggregate_only": True,
            "contains_absolute_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
        },
        "limitations": [
            "This report is not a release or production performance conclusion.",
            "The workload may not represent production text lengths or distributions.",
            (
                "Process cold start, GPU execution, model loading, and model inference "
                "are not measured."
            ),
            "CPU max RSS is a process-wide high-water mark, not isolated pipeline allocation.",
        ],
    }


def _publish_new_json(report: Mapping[str, object], output: Path) -> None:
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            delete=False,
            prefix=f".{output.name}.",
        )
        temporary = Path(handle.name)
        with handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, output)
    except OSError as exc:
        raise SafeBenchmarkError("failed to atomically publish a new report") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        report = _build_report(args)
        _publish_new_json(report, args.output)
        print(
            json.dumps(
                {
                    "document_count": report["workload"]["document_count"],  # type: ignore[index]
                    "mode": "rules-only",
                    "report_sha256": _sha256_file(args.output),
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
    except SafeBenchmarkError as exc:
        print(f"benchmark-cascade: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
