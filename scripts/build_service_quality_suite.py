#!/usr/bin/env python3
"""Build a private, aggregate-only development service-quality comparison.

This command consumes canonical gold plus outputs produced by
``pii-zh-evaluate-service``.  It deliberately has no confirmatory execution
path: any confirmatory intent is rejected before the gold or prediction files
are opened.  Empty fixed-24 gold is usable as a development-only PII-free
proxy, but is never represented as human-adjudicated or formal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

from pii_zh.evaluation import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    canonical_json_hash,
    load_gold_jsonl,
    load_prediction_jsonl,
)
from pii_zh.evaluation.service_eval_cli import (
    SERVICE_EVALUATION_DOCUMENT_ID_VERSION,
    SERVICE_EVALUATION_SCHEMA_VERSION,
    service_evaluation_doc_id,
)
from pii_zh.evaluation.service_quality_suite import (
    PII_CORE_LABELS,
    ServiceQualityComparatorInput,
    build_service_quality_suite,
)
from pii_zh.taxonomy import load_presidio_mapping

OUTPUT_SCHEMA_VERSION = "pii-zh.service-quality-suite-development.v1"
_RUN_INTENTS = ("development-non-claim", "frozen-confirmatory")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}\Z")
_SAFE_DOCUMENT_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_GOLD_BYTES = 512 * 1024 * 1024
_MAX_PREDICTION_BYTES = 512 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_OUTPUT_MODE = 0o400


class ServiceQualitySuiteCliError(RuntimeError):
    """An input-value-free failure safe to expose on stderr."""


@dataclass(frozen=True, slots=True)
class _SystemArguments:
    system_id: str
    predictions_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class _ValidatedSystem:
    system_id: str
    predictions: tuple[PredictionRecord, ...]
    predictions_sha256: str
    package_implementation_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-intent", required=True, choices=_RUN_INTENTS)
    parser.add_argument(
        "--suite-name",
        default="production-pii-core-development",
        help="Path-free stable identifier included in the aggregate result.",
    )
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument(
        "--comparator",
        action="append",
        nargs=3,
        metavar=("ID", "PREDICTIONS", "SERVICE_REPORT"),
        help="Repeat once per comparator.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ServiceQualitySuiteCliError(f"{field} must be a path-free stable identifier")
    return value


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _system_arguments(args: argparse.Namespace) -> tuple[_SystemArguments, ...]:
    if not args.comparator:
        raise ServiceQualitySuiteCliError("at least one comparator is required")
    systems: list[_SystemArguments] = [
        _SystemArguments(
            system_id=_safe_id(args.candidate_id, field="candidate ID"),
            predictions_path=args.candidate_predictions,
            report_path=args.candidate_report,
        )
    ]
    for raw in args.comparator:
        comparator_id, predictions, report = raw
        systems.append(
            _SystemArguments(
                system_id=_safe_id(comparator_id, field="comparator ID"),
                predictions_path=Path(predictions),
                report_path=Path(report),
            )
        )
    system_ids = [system.system_id for system in systems]
    if len(system_ids) != len(set(system_ids)):
        raise ServiceQualitySuiteCliError("candidate and comparator IDs must be unique")
    return tuple(systems)


def _validate_development_args(
    args: argparse.Namespace,
) -> tuple[str, tuple[_SystemArguments, ...]]:
    # This check must remain first: confirmatory intent may not inspect any data path.
    if args.run_intent != "development-non-claim":
        raise ServiceQualitySuiteCliError(
            "frozen-confirmatory service-quality admission contract unavailable"
        )
    suite_name = _safe_id(args.suite_name, field="suite name")
    systems = _system_arguments(args)
    if args.output.exists() or args.output.is_symlink():
        raise ServiceQualitySuiteCliError("refusing to overwrite an existing output")
    input_paths = {_resolved(args.gold)}
    for system in systems:
        input_paths.add(_resolved(system.predictions_path))
        input_paths.add(_resolved(system.report_path))
    if _resolved(args.output) in input_paths:
        raise ServiceQualitySuiteCliError("output must be distinct from every input")
    return suite_name, systems


def _read_regular(path: Path, *, field: str, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ServiceQualitySuiteCliError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ServiceQualitySuiteCliError(f"{field} must be a bounded regular file")
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise ServiceQualitySuiteCliError(f"{field} exceeds the size limit")
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(payload) != after.st_size:
            raise ServiceQualitySuiteCliError(f"{field} changed while it was read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ServiceQualitySuiteCliError("service report contains a duplicate JSON key")
        value[key] = item
    return value


def _load_report(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ServiceQualitySuiteCliError("service report contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceQualitySuiteCliError("service report is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ServiceQualitySuiteCliError("service report must be a JSON object")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServiceQualitySuiteCliError(f"{field} must be an object")
    return value


def _sha256_value(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ServiceQualitySuiteCliError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_component_path(value: object) -> str:
    if not isinstance(value, str):
        raise ServiceQualitySuiteCliError("package component path is invalid")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ServiceQualitySuiteCliError("package component path is invalid")
    return value


def _package_implementation(runtime: Mapping[str, Any]) -> str:
    closure = _mapping(runtime.get("package_source_closure"), field="package source closure")
    components = _mapping(closure.get("component_sha256"), field="package components")
    if not components:
        raise ServiceQualitySuiteCliError("package source closure is empty")
    normalized: dict[str, str] = {}
    for raw_path, raw_sha256 in components.items():
        path = _safe_component_path(raw_path)
        normalized[path] = _sha256_value(raw_sha256, field="package component hash")
    count = closure.get("source_file_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(normalized):
        raise ServiceQualitySuiteCliError("package source closure count is invalid")
    if "evaluation/service_eval_cli.py" not in normalized:
        raise ServiceQualitySuiteCliError("package source closure omits the service evaluator")
    implementation = _sha256_value(
        closure.get("implementation_sha256"), field="package implementation hash"
    )
    if implementation != canonical_json_hash(normalized):
        raise ServiceQualitySuiteCliError("package implementation hash does not verify")
    mode = closure.get("closure_mode")
    if not isinstance(mode, str) or not mode.startswith("full_pii_zh_"):
        raise ServiceQualitySuiteCliError("package source closure mode is invalid")
    return implementation


def _validate_scope(runtime: Mapping[str, Any]) -> None:
    scope = _mapping(runtime.get("canonical_evaluation_scope"), field="evaluation scope")
    mapping = load_presidio_mapping()
    expected_policy_labels = [
        mapping.model_to_presidio[label] for label in PII_CORE_LABELS
    ]
    if (
        scope.get("scope_mode") != "fixed_canonical_24_input_and_service_routing"
        or scope.get("entity_count") != 24
        or scope.get("metric_model_labels") != list(PII_CORE_LABELS)
        or scope.get("pipeline_policy_entities") != expected_policy_labels
        or scope.get("formal_fixed_24_macro_status")
        != "NOT_CLAIMED_IN_THIS_DEVELOPMENT_RUN"
    ):
        raise ServiceQualitySuiteCliError("service report is not fixed-24 development scope")


def _validate_report(
    report: Mapping[str, Any],
    *,
    gold_sha256: str,
    predictions_sha256: str,
    document_count: int,
) -> str:
    runtime = _mapping(
        report.get("service_evaluation_runtime"), field="service evaluation runtime"
    )
    if (
        runtime.get("schema_version") != SERVICE_EVALUATION_SCHEMA_VERSION
        or runtime.get("run_intent") != "development-non-claim"
        or runtime.get("claim_eligible") is not False
        or runtime.get("claim_status") != "DEVELOPMENT_NON_CLAIM"
        or runtime.get("human_gold_admission_contract_status") != "UNAVAILABLE"
    ):
        raise ServiceQualitySuiteCliError("service report development contract is invalid")
    if runtime.get("canonical_input_sha256") != gold_sha256:
        raise ServiceQualitySuiteCliError("service report canonical input hash mismatch")
    if runtime.get("predictions_sha256") != predictions_sha256:
        raise ServiceQualitySuiteCliError("service report prediction hash mismatch")
    count = runtime.get("prediction_document_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != document_count:
        raise ServiceQualitySuiteCliError("service report document count mismatch")

    provenance = _mapping(report.get("provenance"), field="service report provenance")
    provenance_gold = _mapping(provenance.get("gold"), field="provenance gold")
    provenance_predictions = _mapping(
        provenance.get("predictions"), field="provenance predictions"
    )
    if (
        provenance.get("schema_version") != 1
        or provenance.get("mode") != "development"
        or provenance_gold.get("sha256") != gold_sha256
        or provenance_predictions.get("sha256") != predictions_sha256
    ):
        raise ServiceQualitySuiteCliError("service report provenance binding mismatch")

    document_policy = _mapping(runtime.get("document_id_policy"), field="document ID policy")
    if (
        document_policy.get("schema_version") != SERVICE_EVALUATION_DOCUMENT_ID_VERSION
        or document_policy.get("algorithm") != "domain_separated_sha256"
        or document_policy.get("original_document_ids_persisted") is not False
    ):
        raise ServiceQualitySuiteCliError("service report document ID policy is invalid")
    privacy = _mapping(runtime.get("privacy"), field="service report privacy")
    for flag in (
        "contains_absolute_paths",
        "contains_raw_text",
        "contains_entity_values",
        "contains_original_document_ids",
    ):
        if privacy.get(flag) is not False:
            raise ServiceQualitySuiteCliError("service report privacy contract is invalid")
    _validate_scope(runtime)
    return _package_implementation(runtime)


def _load_gold(payload: bytes) -> tuple[GoldRecord, ...]:
    try:
        raw_records = load_gold_jsonl(StringIO(payload.decode("utf-8")))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ServiceQualitySuiteCliError("canonical gold validation failed") from exc
    if not raw_records:
        raise ServiceQualitySuiteCliError("canonical gold must contain documents")
    fixed_labels = frozenset(PII_CORE_LABELS)
    safe_records: list[GoldRecord] = []
    safe_ids: set[str] = set()
    for record in raw_records:
        if any(span.label not in fixed_labels for span in record.spans):
            raise ServiceQualitySuiteCliError("canonical gold is outside fixed-24 scope")
        safe_id = service_evaluation_doc_id(record.doc_id)
        if safe_id in safe_ids:
            raise ServiceQualitySuiteCliError("canonical gold document ID mapping collided")
        safe_ids.add(safe_id)
        safe_record = GoldRecord(
            doc_id=safe_id,
            spans=record.spans,
            text_length=record.text_length,
        )
        safe_record.validate()
        safe_records.append(safe_record)
    return tuple(safe_records)


def _load_predictions(payload: bytes) -> tuple[PredictionRecord, ...]:
    try:
        records = tuple(load_prediction_jsonl(StringIO(payload.decode("utf-8"))))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ServiceQualitySuiteCliError("prediction validation failed") from exc
    if not records or any(
        _SAFE_DOCUMENT_ID_RE.fullmatch(record.doc_id) is None for record in records
    ):
        raise ServiceQualitySuiteCliError("predictions must use service-safe document IDs")
    return records


def _load_system(
    system: _SystemArguments,
    *,
    gold_sha256: str,
    gold_document_ids: frozenset[str],
) -> _ValidatedSystem:
    predictions_payload = _read_regular(
        system.predictions_path,
        field="prediction input",
        maximum_bytes=_MAX_PREDICTION_BYTES,
    )
    predictions = _load_predictions(predictions_payload)
    prediction_ids = {record.doc_id for record in predictions}
    if prediction_ids != gold_document_ids or len(predictions) != len(gold_document_ids):
        raise ServiceQualitySuiteCliError("prediction document universe mismatch")
    report_payload = _read_regular(
        system.report_path,
        field="service report",
        maximum_bytes=_MAX_REPORT_BYTES,
    )
    predictions_sha256 = _sha256(predictions_payload)
    implementation_sha256 = _validate_report(
        _load_report(report_payload),
        gold_sha256=gold_sha256,
        predictions_sha256=predictions_sha256,
        document_count=len(predictions),
    )
    return _ValidatedSystem(
        system_id=system.system_id,
        predictions=predictions,
        predictions_sha256=predictions_sha256,
        package_implementation_sha256=implementation_sha256,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Validate development inputs and build one aggregate-only report."""

    suite_name, system_arguments = _validate_development_args(args)
    gold_payload = _read_regular(
        args.gold,
        field="canonical gold",
        maximum_bytes=_MAX_GOLD_BYTES,
    )
    gold_sha256 = _sha256(gold_payload)
    gold = _load_gold(gold_payload)
    gold_document_ids = frozenset(record.doc_id for record in gold)
    systems = tuple(
        _load_system(
            system,
            gold_sha256=gold_sha256,
            gold_document_ids=gold_document_ids,
        )
        for system in system_arguments
    )
    package_hashes = {system.package_implementation_sha256 for system in systems}
    if len(package_hashes) != 1:
        raise ServiceQualitySuiteCliError(
            "candidate and comparators use different package implementations"
        )

    candidate = systems[0]
    comparator_inputs = tuple(
        ServiceQualityComparatorInput(system.system_id, system.predictions)
        for system in systems[1:]
    )
    pii_free_ids = tuple(record.doc_id for record in gold if not record.spans)
    try:
        aggregate = build_service_quality_suite(
            suite_name=suite_name,
            candidate_id=candidate.system_id,
            gold_records=gold,
            candidate_records=candidate.predictions,
            comparators=comparator_inputs,
            labels=PII_CORE_LABELS,
            pii_free_eligible_doc_ids=pii_free_ids,
            pii_free_eligibility_complete=True,
        )
    except (EvaluationDataError, RuntimeError, TypeError, ValueError) as exc:
        raise ServiceQualitySuiteCliError(
            "aggregate service-quality construction failed"
        ) from exc

    report: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_intent": "development-non-claim",
        "status": "DEVELOPMENT_AGGREGATE_ONLY_NON_ACTIVATING",
        "input_bindings": {
            "canonical_input_sha256": gold_sha256,
            "document_count": len(gold),
            "service_package_implementation_sha256": next(iter(package_hashes)),
            "candidate": {
                "system_id": candidate.system_id,
                "predictions_sha256": candidate.predictions_sha256,
            },
            "comparators": [
                {
                    "system_id": system.system_id,
                    "predictions_sha256": system.predictions_sha256,
                }
                for system in systems[1:]
            ],
        },
        "pii_free_eligibility": {
            "status": "DEVELOPMENT_INFERRED_NON_ADJUDICATED_NONFORMAL",
            "basis": "empty_gold_within_fixed_canonical_24_scope",
            "document_count": len(pii_free_ids),
            "human_adjudicated": False,
            "formal_eligibility": False,
            "claim_eligible": False,
        },
        "aggregate_suite": aggregate,
        "claims": {
            "first": False,
            "best": False,
            "global_best": False,
            "sota": False,
            "claim_eligible": False,
        },
        "privacy": {
            "aggregate_only": True,
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "per_document_statistics_output": False,
        },
        "limitations": [
            "Empty fixed-24 gold is only a development PII-free proxy, not adjudication.",
            "This artifact cannot activate formal, first, best, global-best, or SOTA claims.",
        ],
    }
    report["artifact_sha256"] = canonical_json_hash(report)
    return report


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ServiceQualitySuiteCliError("aggregate report is not serializable") from exc


def _publish_new(report: Mapping[str, Any], output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ServiceQualitySuiteCliError("refusing to overwrite an existing output")
    temporary: Path | None = None
    linked = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temporary = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(_OUTPUT_MODE)
        os.link(temporary, output)
        linked = True
    except FileExistsError as exc:
        raise ServiceQualitySuiteCliError("refusing to overwrite an existing output") from exc
    except OSError as exc:
        if linked:
            try:
                output.chmod(0o600)
                output.unlink()
            except OSError:
                pass
        raise ServiceQualitySuiteCliError("could not atomically publish the output") from exc
    finally:
        if temporary is not None:
            try:
                if not linked:
                    temporary.chmod(0o600)
                temporary.unlink()
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(args)
        _publish_new(report, args.output)
        print(
            json.dumps(
                {
                    "artifact_sha256": report["artifact_sha256"],
                    "document_count": report["input_bindings"]["document_count"],
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "status": report["status"],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    except (ServiceQualitySuiteCliError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, ServiceQualitySuiteCliError):
            message = str(exc)
        else:
            message = "service-quality suite setup failed"
        print(f"service-quality-suite: error: {message}", file=sys.stderr)
        return 2


__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "ServiceQualitySuiteCliError",
    "build_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
