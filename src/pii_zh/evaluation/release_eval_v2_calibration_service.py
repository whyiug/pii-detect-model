"""Development-only service diagnostics on the authorized v2 calibration split.

This module never loads a model.  It consumes already-frozen ``model_raw``
predictions, applies the same calibration normalization/filtering used by the
community service, and injects the retained spans into the real cascade
pipeline.  The calibration stage gate is checked before the calibration JSONL
is opened; the internal-evaluation split is never accepted by this tool.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from pii_zh.calibration import CalibrationBundle, apply_calibration
from pii_zh.cascade import CascadePipeline
from pii_zh.cascade.routing import community_full24_routes
from pii_zh.cascade.service_profiles import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    _normalize_calibration,
    load_service_config,
)
from pii_zh.data.schema import DocumentRecord, SchemaError
from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
from pii_zh.evaluation.io import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    Span,
    dumps_prediction,
    load_prediction_jsonl,
)
from pii_zh.evaluation.metrics import evaluate_documents
from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
    AmendmentReplayInputs,
    PredictionProvenanceInputs,
    ReleaseEvalV2PredictionProvenanceError,
    _rules_identity,
    _source_hashes,
    _validator_identity,
    replay_release_eval_v2_prediction_manifest,
)
from pii_zh.evaluation.release_eval_v2_stage_gate import (
    CalibrationAuthorizationInputs,
    ReleaseEvalV2StageGateError,
    SelectionReplayEvidence,
    guard_calibration_preopen,
    replay_calibration_authorization,
)
from pii_zh.evaluation.required_metrics import build_required_metrics
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.rules.cn_common_v6 import CnCommonRulePackV6
from pii_zh.taxonomy import load_presidio_mapping

SCHEMA_VERSION = "pii-zh.release-eval-v2-calibration-service-diagnostic.v1"
SCHEMA_REPOSITORY_PATH = (
    "configs/evaluation/release_eval_v2_calibration_service_diagnostic_v1.schema.json"
)
MODULE_REPOSITORY_PATH = "src/pii_zh/evaluation/release_eval_v2_calibration_service.py"
CLI_REPOSITORY_PATH = "scripts/evaluate_release_eval_v2_calibration_service.py"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class CalibrationServiceDiagnosticError(RuntimeError):
    """Fail-closed diagnostic error which never includes record contents."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    payload: bytes
    sha256: str
    size_bytes: int
    nonempty_lines: int
    mode: str


@dataclass(frozen=True, slots=True)
class CalibrationServiceRequest:
    """Complete inputs for a deterministic development-only run or replay."""

    authorization: Path
    protocol: Path
    development_manifest: Path
    v1_release_manifest: Path
    candidate_roots: tuple[Path, Path, Path]
    selection_receipt: Path
    amendment: Path
    dataset_manifest: Path
    materialization_receipt: Path
    freeze_receipt: Path
    calibration_gold: Path
    model_artifact: Path
    model_raw_predictions: Path
    model_raw_generation_receipt: Path
    model_raw_prediction_manifest: Path
    calibration_bundle: Path
    model_calibrated_predictions: Path
    full_system_predictions: Path
    receipt: Path

    def authorization_inputs(self) -> CalibrationAuthorizationInputs:
        return CalibrationAuthorizationInputs(
            evidence=SelectionReplayEvidence(
                protocol=self.protocol,
                development_manifest=self.development_manifest,
                v1_release_manifest=self.v1_release_manifest,
                candidates=self.candidate_roots,
                selection_receipt=self.selection_receipt,
                amendment=self.amendment,
            ),
            dataset_manifest=self.dataset_manifest,
            materialization_receipt=self.materialization_receipt,
            freeze_receipt=self.freeze_receipt,
            calibration_gold=self.calibration_gold,
            model_artifact=self.model_artifact,
        )


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CalibrationServiceDiagnosticError("dataset manifest has a duplicate YAML key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CalibrationServiceDiagnosticError("diagnostic is not canonical JSON") from exc


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pretty_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CalibrationServiceDiagnosticError("diagnostic is not strict JSON") from exc


def _snapshot(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
    retain_payload: bool = True,
) -> FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CalibrationServiceDiagnosticError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise CalibrationServiceDiagnosticError(f"{field} must be a bounded regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        nonempty_lines = 0
        carry = b""
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            if retain_payload:
                chunks.append(block)
            values = (carry + block).split(b"\n")
            carry = values.pop()
            nonempty_lines += sum(bool(value.strip()) for value in values)
        nonempty_lines += bool(carry.strip())
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if not stable:
            raise CalibrationServiceDiagnosticError(f"{field} changed while being read")
    except OSError as exc:
        raise CalibrationServiceDiagnosticError(f"{field} cannot be read") from exc
    finally:
        os.close(descriptor)
    return FileSnapshot(
        payload=b"".join(chunks),
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
        nonempty_lines=int(nonempty_lines),
        mode=f"{stat.S_IMODE(before.st_mode):04o}",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationServiceDiagnosticError("JSON input contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise CalibrationServiceDiagnosticError("JSON input contains a non-finite number")


def _json_object(snapshot: FileSnapshot, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationServiceDiagnosticError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CalibrationServiceDiagnosticError(f"{field} must be a JSON object")
    return value


def _assert_path_free(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_path_free(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_path_free(nested)
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "~\\", "file:"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise CalibrationServiceDiagnosticError("diagnostic receipt contains a local path")


def _output_paths(request: CalibrationServiceRequest) -> tuple[Path, Path, Path]:
    paths = (
        request.model_calibrated_predictions,
        request.full_system_predictions,
        request.receipt,
    )
    resolved = [path.expanduser().resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise CalibrationServiceDiagnosticError("diagnostic outputs must be distinct")
    inputs = {
        path.expanduser().resolve(strict=False)
        for path in (
            request.authorization,
            request.protocol,
            request.development_manifest,
            request.v1_release_manifest,
            *request.candidate_roots,
            request.selection_receipt,
            request.amendment,
            request.dataset_manifest,
            request.materialization_receipt,
            request.freeze_receipt,
            request.calibration_gold,
            request.model_raw_predictions,
            request.model_raw_generation_receipt,
            request.model_raw_prediction_manifest,
            request.calibration_bundle,
        )
    }
    if inputs & set(resolved):
        raise CalibrationServiceDiagnosticError("diagnostic output aliases an input")
    return paths


def _preflight_new_outputs(request: CalibrationServiceRequest) -> None:
    for path in _output_paths(request):
        if path.exists() or path.is_symlink():
            raise CalibrationServiceDiagnosticError("refusing to overwrite a diagnostic output")


def _usage_policy(snapshot: FileSnapshot, *, expected_sha256: str) -> dict[str, Any]:
    if snapshot.sha256 != expected_sha256:
        raise CalibrationServiceDiagnosticError(
            "dataset manifest differs from calibration authorization"
        )
    try:
        document = yaml.load(snapshot.payload.decode("utf-8"), Loader=_UniqueSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise CalibrationServiceDiagnosticError(
            "dataset manifest is not strict UTF-8 YAML"
        ) from exc
    if not isinstance(document, Mapping):
        raise CalibrationServiceDiagnosticError("dataset manifest root must be an object")
    usage = document.get("usage_policy")
    policy = usage.get("calibration") if isinstance(usage, Mapping) else None
    expected = {
        "gradient_training_allowed": False,
        "checkpoint_or_model_selection_allowed": False,
        "threshold_tuning_allowed": True,
        "temperature_scaling_allowed": True,
        "fusion_calibration_allowed": True,
        "final_metric_claim_allowed": False,
        "content_read_before_cross_seed_selection_allowed": False,
    }
    if not isinstance(policy, Mapping) or dict(policy) != expected:
        raise CalibrationServiceDiagnosticError(
            "calibration usage policy is not the development-only tuning policy"
        )
    return {"target_split": "calibration", **expected}


def _parse_documents(snapshot: FileSnapshot) -> tuple[list[DocumentRecord], list[GoldRecord]]:
    documents: list[DocumentRecord] = []
    gold: list[GoldRecord] = []
    seen: set[str] = set()
    try:
        decoded = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationServiceDiagnosticError("calibration gold is not UTF-8") from exc
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, Mapping):
                raise SchemaError("record must be an object")
            record = DocumentRecord.from_dict(value)
            if record.doc_id in seen:
                raise SchemaError("duplicate document identifier")
            seen.add(record.doc_id)
            gold_record = GoldRecord(
                doc_id=record.doc_id,
                spans=tuple(
                    Span(start=entity.start, end=entity.end, label=entity.label)
                    for entity in record.entities
                ),
                text_length=len(record.text),
            )
            gold_record.validate()
        except (EvaluationDataError, SchemaError, TypeError, ValueError) as exc:
            raise CalibrationServiceDiagnosticError(
                f"calibration gold line {line_number} violates the canonical contract"
            ) from exc
        documents.append(record)
        gold.append(gold_record)
    if not documents or len(documents) != snapshot.nonempty_lines:
        raise CalibrationServiceDiagnosticError("calibration gold is empty or line count changed")
    allowed = frozenset(PII_CORE_LABELS)
    if any(span.label not in allowed for record in gold for span in record.spans):
        raise CalibrationServiceDiagnosticError("calibration gold contains a non-Open24 label")
    return documents, gold


def _parse_raw_predictions(
    snapshot: FileSnapshot,
    *,
    documents: Sequence[DocumentRecord],
) -> list[PredictionRecord]:
    try:
        predictions = load_prediction_jsonl(io.StringIO(snapshot.payload.decode("utf-8")))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise CalibrationServiceDiagnosticError(
            "model_raw predictions violate the text-free prediction contract"
        ) from exc
    if [record.doc_id for record in predictions] != [record.doc_id for record in documents]:
        raise CalibrationServiceDiagnosticError(
            "model_raw predictions do not exactly align by document identity and order"
        )
    allowed = frozenset(PII_CORE_LABELS)
    for document, prediction in zip(documents, predictions, strict=True):
        try:
            prediction.validate(text_length=len(document.text))
        except EvaluationDataError as exc:
            raise CalibrationServiceDiagnosticError(
                "model_raw prediction span is outside its document"
            ) from exc
        for span in prediction.spans:
            if span.label not in allowed or span.score is None:
                raise CalibrationServiceDiagnosticError(
                    "model_raw spans require an Open24 label and raw score"
                )
    return predictions


class _PrecomputedRecognizer:
    """Stateless recognizer adapter over calibrated, output-label spans."""

    def __init__(
        self,
        *,
        texts: Sequence[str],
        records: Sequence[PredictionRecord],
    ) -> None:
        self._texts = tuple(texts)
        self._records = tuple(records)

    def analyze_batch(
        self,
        texts: Sequence[str],
        entities: Sequence[str] | None,
    ) -> list[list[dict[str, object]]]:
        if tuple(texts) != self._texts:
            raise ValueError("precomputed recognizer input order changed")
        requested = None if entities is None else frozenset(entities)
        return [
            [
                {
                    "entity_type": span.label,
                    "start": span.start,
                    "end": span.end,
                    "score": span.score,
                }
                for span in record.spans
                if requested is None or span.label in requested
            ]
            for record in self._records
        ]


def _calibrated_output_label_records(
    raw_predictions: Sequence[PredictionRecord],
    bundle: CalibrationBundle,
) -> tuple[list[PredictionRecord], CalibrationBundle, dict[str, str]]:
    mapping = load_presidio_mapping()
    raw_to_output = {
        label: mapping.model_to_presidio[label] for label in PII_CORE_LABELS
    }
    if len(set(raw_to_output.values())) != len(raw_to_output):
        raise CalibrationServiceDiagnosticError("Open24 service mapping is not one-to-one")
    normalized = _normalize_calibration(bundle)
    if normalized is None:  # pragma: no cover - non-null input is explicit above
        raise CalibrationServiceDiagnosticError("calibration normalization returned null")
    mapped = [
        PredictionRecord(
            doc_id=record.doc_id,
            spans=tuple(
                Span(
                    start=span.start,
                    end=span.end,
                    label=raw_to_output[span.label],
                    score=span.score,
                )
                for span in record.spans
            ),
        )
        for record in raw_predictions
    ]
    try:
        calibrated = apply_calibration(mapped, normalized)
    except (EvaluationDataError, TypeError, ValueError) as exc:
        raise CalibrationServiceDiagnosticError("frozen calibration application failed") from exc
    return calibrated, normalized, raw_to_output


def _run_service(
    *,
    mode: str,
    documents: Sequence[DocumentRecord],
    calibrated: Sequence[PredictionRecord],
    raw_to_output: Mapping[str, str],
) -> tuple[list[PredictionRecord], dict[str, Any]]:
    if mode not in {"model-only", "cascade"}:
        raise CalibrationServiceDiagnosticError("unsupported service diagnostic mode")
    config = load_service_config(
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode=cast(Any, mode),
    )
    config = replace(
        config,
        routes=community_full24_routes(recognizer_thresholds_authoritative=True),
    )
    texts = [record.text for record in documents]
    pipeline = CascadePipeline(
        config=config,
        rule_recognizer=CnCommonRulePackV6() if mode == "cascade" else None,
        model_recognizer=_PrecomputedRecognizer(texts=texts, records=calibrated),
        validators=COMMUNITY_STRUCTURED_VALIDATORS,
        model_identity={"precomputed_model_raw": True, "model_loaded": False},
    )
    try:
        detections = pipeline.detect_batch(texts, entities=tuple(raw_to_output.values()))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CalibrationServiceDiagnosticError("community cascade diagnostic failed") from exc
    output_to_raw = {output: raw for raw, output in raw_to_output.items()}
    predictions: list[PredictionRecord] = []
    for document, items in zip(documents, detections, strict=True):
        spans: list[Span] = []
        for item in items:
            raw_label = output_to_raw.get(item.entity_type)
            if raw_label is None:
                raise CalibrationServiceDiagnosticError("service emitted a non-Open24 label")
            span = Span(
                start=item.start,
                end=item.end,
                label=raw_label,
                score=item.score,
            )
            try:
                span.validate(text_length=len(document.text))
            except EvaluationDataError as exc:
                raise CalibrationServiceDiagnosticError("service emitted an invalid span") from exc
            spans.append(span)
        record = PredictionRecord(doc_id=document.doc_id, spans=tuple(spans))
        try:
            record.validate(text_length=len(document.text))
        except EvaluationDataError as exc:
            raise CalibrationServiceDiagnosticError(
                "service emitted duplicate or invalid spans"
            ) from exc
        predictions.append(record)
    configuration = {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
        "routes": [asdict(route) for route in config.routes],
    }
    return predictions, {
        "mode": mode,
        "configuration_sha256": _canonical_hash(configuration),
        "routes_sha256": _canonical_hash(configuration["routes"]),
    }


def _zero_score() -> dict[str, int | float]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "support": 0,
        "predicted": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "f2": 0.0,
    }


def _fixed_metric_section(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = cast(Mapping[str, Any], report[name])
    observed = cast(Mapping[str, Mapping[str, int | float]], section["per_class"])
    per_label = {
        label: dict(observed.get(label, _zero_score())) for label in PII_CORE_LABELS
    }
    fields = ("precision", "recall", "f1", "f2")
    macro = {
        "label_count": len(PII_CORE_LABELS),
        **{
            field: sum(float(per_label[label][field]) for label in PII_CORE_LABELS)
            / len(PII_CORE_LABELS)
            for field in fields
        },
    }
    return {
        "micro": dict(cast(Mapping[str, Any], section["micro"])),
        "macro": macro,
        "per_label": per_label,
    }


def _aggregate_metrics(
    gold: Sequence[GoldRecord],
    predictions: Sequence[PredictionRecord],
) -> dict[str, Any]:
    report = evaluate_documents(gold, predictions, bootstrap_samples=0)
    required = build_required_metrics(gold, predictions)
    supported = sorted({span.label for record in gold for span in record.spans})
    density = required["pii_free_false_positive_span_density"]
    return {
        "document_count": report["document_count"],
        "span_count": report["span_count"],
        "fixed_label_count": len(PII_CORE_LABELS),
        "gold_supported_label_count": len(supported),
        "gold_supported_labels": supported,
        "strict": _fixed_metric_section(report, "strict"),
        "character": _fixed_metric_section(report, "character"),
        "pii_free": {
            **dict(cast(Mapping[str, Any], report["pii_free"])),
            "false_positive_span_count": density["false_positive_span_count"],
            "character_count": density["character_count"],
            "false_positive_spans_per_1000_characters": density[
                "false_positive_spans_per_1000_characters"
            ],
            "span_density_status": density["status"],
            "span_density_reason": density["reason"],
        },
    }


def _prediction_bytes(records: Sequence[PredictionRecord]) -> bytes:
    return ("".join(dumps_prediction(record) + "\n" for record in records)).encode("utf-8")


def _artifact_identity(payload: bytes, *, document_count: int) -> dict[str, Any]:
    return {
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "document_count": document_count,
        "mode": "0444",
        "contains_raw_text": False,
        "contains_entity_values": False,
        "contains_document_ids": True,
    }


def _implementation_identity() -> dict[str, Any]:
    root = _repository_root()
    files = {
        "module": root / MODULE_REPOSITORY_PATH,
        "cli": root / CLI_REPOSITORY_PATH,
        "schema": root / SCHEMA_REPOSITORY_PATH,
    }
    hashes = {
        logical: _snapshot(
            path,
            field=f"diagnostic implementation {logical}",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        ).sha256
        for logical, path in files.items()
    }
    return {"source_file_sha256": hashes, "implementation_sha256": _canonical_hash(hashes)}


def _service_identity(
    *,
    model_only_configuration: Mapping[str, Any],
    cascade_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    root = _repository_root()
    try:
        source_hashes = _source_hashes(root)
        validators = _validator_identity(root)
        rules = _rules_identity(root)
    except (OSError, ReleaseEvalV2PredictionProvenanceError, TypeError, ValueError) as exc:
        raise CalibrationServiceDiagnosticError("service source identity cannot be closed") from exc
    return {
        "profile_id": COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        "model_calibrated": dict(model_only_configuration),
        "full_system": dict(cascade_configuration),
        "rule_pack": {
            "ruleset_id": rules["ruleset_id"],
            "rule_source": rules["rule_source"],
            "rule_count": rules["rule_count"],
            "definitions_sha256": rules["definitions_sha256"],
            "implementation_sha256": rules["implementation_sha256"],
        },
        "structured_validators": {
            "validator_count": validators["validator_count"],
            "configuration_sha256": validators["configuration_sha256"],
        },
        "implementation_source_sha256": source_hashes,
        "implementation_sha256": _canonical_hash(source_hashes),
    }


def _validate_receipt(value: Mapping[str, Any]) -> None:
    schema_snapshot = _snapshot(
        _repository_root() / SCHEMA_REPOSITORY_PATH,
        field="calibration service diagnostic schema",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    schema = _json_object(schema_snapshot, field="calibration service diagnostic schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except Exception as exc:
        raise CalibrationServiceDiagnosticError(
            "calibration service diagnostic violates its schema"
        ) from exc
    claimed = value.get("receipt_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise CalibrationServiceDiagnosticError("diagnostic receipt self hash is invalid")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256")
    if _canonical_hash(unsigned) != claimed:
        raise CalibrationServiceDiagnosticError("diagnostic receipt self hash differs")
    _assert_path_free(value)


def _build(
    request: CalibrationServiceRequest,
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    # The stage guard only lstats the target split.  It must run before any
    # operation below snapshots, hashes or decodes calibration content.
    try:
        preopen = guard_calibration_preopen(
            request.authorization,
            input_path=request.calibration_gold,
            mode="model-raw",
            model_artifact=request.model_artifact,
        )
    except ReleaseEvalV2StageGateError as exc:
        raise CalibrationServiceDiagnosticError("calibration pre-open gate failed") from exc

    authorization_snapshot = _snapshot(
        request.authorization,
        field="calibration authorization",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    authorization_document = _json_object(
        authorization_snapshot,
        field="calibration authorization",
    )
    if preopen.get("authorization_file_sha256") != authorization_snapshot.sha256:
        raise CalibrationServiceDiagnosticError("calibration authorization changed after guard")
    dataset_binding = authorization_document.get("dataset")
    if not isinstance(dataset_binding, Mapping):
        raise CalibrationServiceDiagnosticError("calibration authorization lacks dataset binding")
    manifest_snapshot = _snapshot(
        request.dataset_manifest,
        field="release-eval-v2 dataset manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    usage_policy = _usage_policy(
        manifest_snapshot,
        expected_sha256=cast(str, dataset_binding.get("dataset_manifest_file_sha256")),
    )

    authorization_inputs = request.authorization_inputs()
    try:
        authorization, authorization_file_sha256 = replay_calibration_authorization(
            request.authorization,
            authorization_inputs,
        )
    except ReleaseEvalV2StageGateError as exc:
        raise CalibrationServiceDiagnosticError(
            "calibration authorization strict replay failed"
        ) from exc
    if (
        authorization != authorization_document
        or authorization_file_sha256 != authorization_snapshot.sha256
    ):
        raise CalibrationServiceDiagnosticError("calibration authorization replay changed")

    gold_snapshot = _snapshot(
        request.calibration_gold,
        field="authorized calibration gold",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    bound_gold = cast(Mapping[str, Any], authorization["dataset"])
    if (
        gold_snapshot.sha256 != bound_gold.get("gold_file_sha256")
        or gold_snapshot.size_bytes != bound_gold.get("gold_size_bytes")
        or gold_snapshot.nonempty_lines != bound_gold.get("gold_document_count")
        or preopen.get("expected_input_sha256") != gold_snapshot.sha256
    ):
        raise CalibrationServiceDiagnosticError("calibration snapshot differs from authorization")

    raw_snapshot = _snapshot(
        request.model_raw_predictions,
        field="model_raw predictions",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    try:
        raw_replay = replay_release_eval_v2_prediction_manifest(
            request.model_raw_prediction_manifest,
            PredictionProvenanceInputs(
                track="model_raw",
                target_split="calibration",
                predictions=request.model_raw_predictions,
                gold=request.calibration_gold,
                dataset_manifest=request.dataset_manifest,
                materialization_receipt=request.materialization_receipt,
                freeze_receipt=request.freeze_receipt,
                model_artifact=request.model_artifact,
                selection_receipt=request.selection_receipt,
                generation_receipt=request.model_raw_generation_receipt,
                amendment=AmendmentReplayInputs(
                    amendment=request.amendment,
                    protocol=request.protocol,
                    development_manifest=request.development_manifest,
                    v1_release_manifest=request.v1_release_manifest,
                    candidate_roots=request.candidate_roots,
                ),
            ),
        )
    except ReleaseEvalV2PredictionProvenanceError as exc:
        raise CalibrationServiceDiagnosticError(
            "model_raw prediction provenance strict replay failed"
        ) from exc
    if raw_replay.get("predictions_file_sha256") != raw_snapshot.sha256:
        raise CalibrationServiceDiagnosticError("model_raw provenance differs from snapshot")

    documents, gold = _parse_documents(gold_snapshot)
    raw_predictions = _parse_raw_predictions(raw_snapshot, documents=documents)
    bundle_snapshot = _snapshot(
        request.calibration_bundle,
        field="frozen calibration bundle",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    bundle_document = _json_object(bundle_snapshot, field="frozen calibration bundle")
    try:
        bundle = CalibrationBundle.from_dict(bundle_document)
    except (TypeError, ValueError) as exc:
        raise CalibrationServiceDiagnosticError("frozen calibration bundle is invalid") from exc
    model_binding = cast(Mapping[str, Any], authorization["model"])
    if (
        bundle.model_version != model_binding.get("training_manifest_sha256")
        or bundle.calibration_version is None
    ):
        raise CalibrationServiceDiagnosticError(
            "calibration bundle is not bound to the authorized selected model"
        )

    calibrated, normalized_bundle, raw_to_output = _calibrated_output_label_records(
        raw_predictions,
        bundle,
    )
    model_predictions, model_config = _run_service(
        mode="model-only",
        documents=documents,
        calibrated=calibrated,
        raw_to_output=raw_to_output,
    )
    full_predictions, full_config = _run_service(
        mode="cascade",
        documents=documents,
        calibrated=calibrated,
        raw_to_output=raw_to_output,
    )
    model_payload = _prediction_bytes(model_predictions)
    full_payload = _prediction_bytes(full_predictions)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "release_eval_v2_calibration_service_development_diagnostic",
        "status": "DEVELOPMENT_ONLY_DIAGNOSTIC_COMPLETE_NOT_RELEASE",
        "scope": {
            "target_split": "calibration",
            "ordered_labels": list(PII_CORE_LABELS),
            "model_loaded": False,
            "gpu_used": False,
            "bootstrap_samples": 0,
        },
        "authorization": {
            "preopen_status": preopen["status"],
            "authorization_file_sha256": authorization_file_sha256,
            "authorization_sha256": authorization["receipt_sha256"],
            "strict_replay": True,
            "gate_applied_before_calibration_open": True,
        },
        "usage_policy": usage_policy,
        "dataset": {
            "dataset_id": bound_gold["dataset_id"],
            "dataset_version": bound_gold["dataset_version"],
            "dataset_manifest_file_sha256": bound_gold[
                "dataset_manifest_file_sha256"
            ],
            "dataset_manifest_sha256": bound_gold["dataset_manifest_sha256"],
            "gold_file_sha256": gold_snapshot.sha256,
            "gold_size_bytes": gold_snapshot.size_bytes,
            "gold_document_count": len(documents),
        },
        "model": {
            "training_manifest_file_sha256": model_binding[
                "training_manifest_file_sha256"
            ],
            "training_manifest_sha256": model_binding["training_manifest_sha256"],
            "model_identity_sha256": model_binding["model_identity_sha256"],
        },
        "model_raw": {
            "predictions_file_sha256": raw_snapshot.sha256,
            "predictions_size_bytes": raw_snapshot.size_bytes,
            "prediction_document_count": len(raw_predictions),
            "prediction_manifest_file_sha256": raw_replay["manifest_file_sha256"],
            "prediction_manifest_sha256": raw_replay["manifest_sha256"],
            "strict_provenance_replay": True,
        },
        "calibration": {
            "bundle_file_sha256": bundle_snapshot.sha256,
            "calibration_version": bundle.calibration_version,
            "model_version": bundle.model_version,
            "normalized_bundle_sha256": _canonical_hash(normalized_bundle.to_dict()),
            "apply_calibration_reused": True,
            "serving_normalization_reused": True,
        },
        "service": _service_identity(
            model_only_configuration=model_config,
            cascade_configuration=full_config,
        ),
        "outputs": {
            "model_calibrated": _artifact_identity(
                model_payload,
                document_count=len(model_predictions),
            ),
            "full_system": _artifact_identity(
                full_payload,
                document_count=len(full_predictions),
            ),
        },
        "metrics": {
            "model_calibrated": _aggregate_metrics(gold, model_predictions),
            "full_system": _aggregate_metrics(gold, full_predictions),
        },
        "implementation": _implementation_identity(),
        "claim_policy": {
            "formal_provenance": False,
            "formal_quality_receipt": False,
            "release_gate_eligible": False,
            "first_best_or_sota_claim_allowed": False,
            "automatic_production_configuration_change_allowed": False,
            "purpose": "fusion_calibration_and_threshold_tuning_diagnostic_only",
        },
        "privacy": {
            "receipt_contains_paths": False,
            "receipt_contains_raw_text": False,
            "receipt_contains_entity_values": False,
            "receipt_contains_document_ids": False,
            "prediction_outputs_are_text_free": True,
            "internal_evaluation_opened_hashed_or_decoded": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    _validate_receipt(receipt)
    return receipt, model_payload, full_payload, _pretty_bytes(receipt)


def _publish_many(payloads: Mapping[Path, bytes]) -> None:
    temporaries: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination, payload in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written < 1:
                        raise OSError("short write")
                    offset += written
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            temporaries[destination] = temporary
        for destination, temporary in temporaries.items():
            os.link(temporary, destination, follow_symlinks=False)
            published.append(destination)
    except OSError as exc:
        for destination in published:
            temporary = temporaries[destination]
            try:
                if destination.lstat().st_ino == temporary.lstat().st_ino:
                    destination.unlink()
            except OSError:
                pass
        raise CalibrationServiceDiagnosticError("cannot publish diagnostic outputs") from exc
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def produce_calibration_service_diagnostic(
    request: CalibrationServiceRequest,
) -> dict[str, Any]:
    """Build and atomically publish three new mode-0444 diagnostic artifacts."""

    _preflight_new_outputs(request)
    receipt, model_payload, full_payload, receipt_payload = _build(request)
    _preflight_new_outputs(request)
    _publish_many(
        {
            request.model_calibrated_predictions: model_payload,
            request.full_system_predictions: full_payload,
            request.receipt: receipt_payload,
        }
    )
    return {
        "status": receipt["status"],
        "receipt_sha256": receipt["receipt_sha256"],
        "document_count": receipt["dataset"]["gold_document_count"],
        "formal_claim_allowed": False,
    }


def replay_calibration_service_diagnostic(
    request: CalibrationServiceRequest,
) -> dict[str, Any]:
    """Rebuild in memory and require byte-identical read-only artifacts."""

    _output_paths(request)
    receipt, model_payload, full_payload, receipt_payload = _build(request)
    expected = {
        request.model_calibrated_predictions: model_payload,
        request.full_system_predictions: full_payload,
        request.receipt: receipt_payload,
    }
    for destination, payload in expected.items():
        observed = _snapshot(
            destination,
            field="persisted diagnostic output",
            maximum_bytes=_MAX_JSONL_BYTES,
        )
        if observed.mode != "0444" or observed.payload != payload:
            raise CalibrationServiceDiagnosticError(
                "persisted diagnostic output is not byte-identical mode-0444"
            )
    return {
        "status": "DEVELOPMENT_ONLY_REPLAY_PASS_NOT_RELEASE",
        "receipt_sha256": receipt["receipt_sha256"],
        "prediction_byte_identical": True,
        "receipt_byte_identical": True,
        "formal_claim_allowed": False,
    }


__all__ = [
    "CalibrationServiceDiagnosticError",
    "CalibrationServiceRequest",
    "SCHEMA_VERSION",
    "produce_calibration_service_diagnostic",
    "replay_calibration_service_diagnostic",
]
