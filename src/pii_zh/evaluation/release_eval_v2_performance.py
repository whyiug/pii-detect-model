"""Exact, aggregate-only performance replay for release-eval-v2 candidates.

The formal path strictly rebuilds the complete internal-unlock chain before it
opens, hashes, or decodes the internal split.  Inference is then executed
through the same local model recognizer or community service builder used for
candidate generation.  A run is accepted only if its prediction JSONL is byte
identical to the already frozen candidate prediction file.

Performance numbers are measured in process.  They are never accepted as CLI
arguments.  Replay measures again and checks metric invariants and broad
same-runtime plausibility; noisy wall-clock values are intentionally not
required to be byte-for-byte identical.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from pii_zh.calibration import CalibrationBundle
from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    build_community_model_service_pipeline,
)
from pii_zh.data.schema import DocumentRecord, SchemaError
from pii_zh.evaluation.io import (
    EvaluationDataError,
    PredictionRecord,
    Span,
    load_prediction_jsonl,
    write_prediction_jsonl,
)
from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
    CalibrationInputs,
    _model_identity,
    _service_identity,
    generation_implementation_identity,
)
from pii_zh.evaluation.release_eval_v2_stage_gate import (
    CalibrationAuthorizationInputs,
    InternalUnlockInputs,
    ReleaseEvalV2StageGateError,
    SelectionReplayEvidence,
    guard_internal_preopen,
    replay_internal_unlock,
)
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.inference import load_local_predictor
from pii_zh.presidio import QwenPiiRecognizer
from pii_zh.taxonomy import load_presidio_mapping, load_taxonomy

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PerformanceTrack = Literal["model_raw", "model_calibrated", "full_system"]
CanonicalReleaseTrack = Literal["model_raw", "full_system"]
Device = Literal["cpu", "cuda:0"]
Detector = Callable[[Sequence[str]], Sequence[Sequence[Any]]]

PERFORMANCE_SCHEMA_VERSION = "pii-zh.community-quality-performance-binding.v2"
REPLAY_SCHEMA_VERSION = "pii-zh.community-cascade-replay-receipt.v2"
PERFORMANCE_SCHEMA_PATH = "configs/evaluation/community_quality_performance_binding_v2.schema.json"
REPLAY_SCHEMA_PATH = "configs/release/community_cascade_release_v2.replay-receipt.schema.json"
QUALITY_SCHEMA_PATH = "configs/evaluation/public_synthetic_service_quality_result_v2.schema.json"
SOURCE_SCHEMA_PATH = "configs/release/community_service_source_identity_v4.schema.json"
SOURCE_SCHEMA_VERSION = "pii-zh.community-service-source-identity.v4"
HARNESS_CLI_PATH = "scripts/benchmark_release_eval_v2_candidate.py"
HARNESS_MODULE_PATH = "src/pii_zh/evaluation/release_eval_v2_performance.py"
EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_ID = "pii_zh.evaluation.release_eval_v2_performance.py"
EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS = frozenset(
    {
        EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_ID,
        "pii_zh.data.synthetic.sota_release_eval_v1.py",
        "pii_zh.data.synthetic.sota_release_eval_v2.py",
        "pii_zh.data.synthetic.sota_v1.py",
        "pii_zh.data.synthetic.sota_v2_resplit.py",
    }
)

_PACKAGE_SOURCE_ROOT = Path("src/pii_zh")
_PACKAGE_SOURCE_SUFFIXES = frozenset({".json", ".py", ".yaml", ".yml"})

# Match the frozen release-eval-v2 candidate generation runtime.  Prediction
# receipts include calibrated floating-point scores, so changing the batch
# layout can change serialized bytes even when labels and boundaries agree.
DOCUMENT_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 8
WARMUP_BATCHES = 2
MAX_TOKENS = 512
STRIDE_FRACTION = 0.25
MAX_REPLAY_MEASUREMENT_RATIO = 100.0

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}\Z")


class ReleaseEvalV2PerformanceError(RuntimeError):
    """A fail-closed performance/replay error with no record contents or paths."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Bytes read from one stable, non-symlink regular-file descriptor."""

    payload: bytes
    sha256: str
    size_bytes: int
    nonempty_lines: int


@dataclass(frozen=True, slots=True)
class FormalStageReplayPaths:
    """Complete upstream evidence needed to strictly rebuild internal unlock."""

    protocol: Path
    development_manifest: Path
    v1_release_manifest: Path
    candidates: tuple[Path, Path, Path]
    selection_receipt: Path
    amendment: Path
    dataset_manifest: Path
    materialization_receipt: Path
    freeze_receipt: Path
    calibration_gold: Path
    calibration_authorization: Path
    calibration_diagnostics: Path
    calibration_fit_predictions: Path
    calibration_fit_generation_receipt: Path
    calibration_fit_prediction_manifest: Path
    internal_unlock: Path


@dataclass(frozen=True, slots=True)
class HarnessRequest:
    """Invocation-only paths; none are serialized into release evidence."""

    track: PerformanceTrack
    target_split: Literal["fixture", "internal_evaluation"]
    model: Path
    calibration: Path
    source: Path
    expected_predictions: Path
    final_model_binding: Path
    service_configuration_binding: Path
    device: Device
    stage_replay: FormalStageReplayPaths | None = None


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    """In-memory execution result; prediction bytes are never put in a receipt."""

    measurement: dict[str, int | float]
    input_sha256: str
    predictions_sha256: str
    document_count: int
    character_count: int


@dataclass(frozen=True, slots=True)
class PreparedIdentity:
    """Validated path-free model/service identity and exact file bindings."""

    model: dict[str, Any]
    model_payload: bytes
    service: dict[str, Any]
    service_payload: bytes
    calibration: dict[str, Any]
    calibration_sha256: str
    unlock: dict[str, Any] | None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise ReleaseEvalV2PerformanceError("value is not canonical strict JSON") from exc


def canonical_json_hash(value: object) -> str:
    return _sha256(_canonical_bytes(value))


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
        raise ReleaseEvalV2PerformanceError("value is not strict JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseEvalV2PerformanceError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ReleaseEvalV2PerformanceError("JSON contains a non-finite number")


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvalV2PerformanceError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseEvalV2PerformanceError(f"{field} must be a JSON object")
    return value


def _read_regular(path: Path, *, field: str, maximum_bytes: int) -> FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseEvalV2PerformanceError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ReleaseEvalV2PerformanceError(f"{field} must be a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            total += len(block)
            if total > maximum_bytes:
                raise ReleaseEvalV2PerformanceError(f"{field} exceeds the byte limit")
            chunks.append(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ReleaseEvalV2PerformanceError(f"{field} changed during snapshot")
    except OSError as exc:
        raise ReleaseEvalV2PerformanceError(f"{field} could not be snapshotted") from exc
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return FileSnapshot(
        payload=payload,
        sha256=_sha256(payload),
        size_bytes=len(payload),
        nonempty_lines=sum(bool(line.strip()) for line in payload.splitlines()),
    )


def _read_json(path: Path, *, field: str) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = _read_regular(path, field=field, maximum_bytes=_MAX_JSON_BYTES)
    return _strict_json(snapshot.payload, field=field), snapshot


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseEvalV2PerformanceError(f"{field} must be an object")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReleaseEvalV2PerformanceError(f"{field} is not a lowercase SHA-256")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ReleaseEvalV2PerformanceError(f"{field} is not a safe identifier")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], *, field: str) -> None:
    if set(value) != keys:
        raise ReleaseEvalV2PerformanceError(f"{field} has an unexpected closed shape")


def _verify_self_hash(value: Mapping[str, Any], *, field: str, key: str) -> str:
    claimed = _digest(value.get(key), field=f"{field}.{key}")
    unsigned = dict(value)
    unsigned.pop(key, None)
    if canonical_json_hash(unsigned) != claimed:
        raise ReleaseEvalV2PerformanceError(f"{field} self hash does not verify")
    return claimed


def _assert_path_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"path", "paths"} or key.endswith("_path"):
                raise ReleaseEvalV2PerformanceError("receipt contains a path field")
            _assert_path_free(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_path_free(nested)
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "~\\", "file:")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ReleaseEvalV2PerformanceError("receipt contains a local path")


def _validate_schema(value: Mapping[str, Any], relative: str, *, field: str) -> None:
    schema, _snapshot = _read_json(_repository_root() / relative, field=f"{field} schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except Exception as exc:
        raise ReleaseEvalV2PerformanceError(f"{field} violates its closed schema") from exc


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_package_source_hashes(repository_root: Path) -> dict[str, str]:
    """Independently rebuild the complete installed-package source projection."""

    source_root = repository_root / _PACKAGE_SOURCE_ROOT
    try:
        root_metadata = source_root.lstat()
        entries = sorted(source_root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise ReleaseEvalV2PerformanceError(
            "runtime package source inventory is unavailable"
        ) from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ReleaseEvalV2PerformanceError("runtime package source root is unsafe")

    result: dict[str, str] = {}
    for path in entries:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ReleaseEvalV2PerformanceError(
                "runtime package source entry is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseEvalV2PerformanceError(
                "runtime package source inventory contains a symlink"
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseEvalV2PerformanceError(
                "runtime package source inventory contains a special file"
            )
        if path.suffix not in _PACKAGE_SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(source_root).as_posix()
        logical_id = f"pii_zh.{relative.replace('/', '.')}"
        if logical_id in result:
            raise ReleaseEvalV2PerformanceError(
                "runtime package logical source identity is ambiguous"
            )
        result[logical_id] = _read_regular(
            path,
            field=f"runtime package source {logical_id}",
            maximum_bytes=_MAX_JSON_BYTES,
        ).sha256
    if not result:
        raise ReleaseEvalV2PerformanceError("runtime package source inventory is empty")
    return dict(sorted(result.items()))


def _validate_source_successor(source: Mapping[str, Any]) -> None:
    """Validate the v4 current-tree closure and its narrowly approved v3 delta."""

    _validate_schema(source, SOURCE_SCHEMA_PATH, field="service source manifest")
    _verify_self_hash(source, field="service source manifest", key="manifest_sha256")
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ReleaseEvalV2PerformanceError("service source successor version differs")

    runtime = _mapping(
        source.get("runtime_package_source_sha256"),
        field="service source runtime inventory",
    )
    current_runtime = _runtime_package_source_hashes(_repository_root())
    if (
        dict(runtime) != current_runtime
        or source.get("runtime_package_inventory_sha256") != canonical_json_hash(current_runtime)
        or source.get("runtime_package_file_count") != len(current_runtime)
        or source.get("runtime_package_inventory_complete") is not True
    ):
        raise ReleaseEvalV2PerformanceError(
            "service source successor differs from the current runtime closure"
        )

    predecessor = _mapping(source.get("predecessor"), field="service source predecessor")
    _exact(
        predecessor,
        {
            "schema_version",
            "file_sha256",
            "manifest_sha256",
            "runtime_package_inventory_sha256",
            "runtime_package_file_count",
        },
        field="service source predecessor",
    )
    if predecessor.get("schema_version") != "pii-zh.community-service-source-identity.v3":
        raise ReleaseEvalV2PerformanceError("service source predecessor version differs")
    for field in ("file_sha256", "manifest_sha256", "runtime_package_inventory_sha256"):
        _digest(predecessor.get(field), field=f"service source predecessor {field}")

    delta = _mapping(source.get("successor_delta"), field="service source successor delta")
    _exact(
        delta,
        {
            "added",
            "changed",
            "removed",
            "added_count",
            "changed_count",
            "removed_count",
            "summary_sha256",
        },
        field="service source successor delta",
    )
    added = _mapping(delta.get("added"), field="service source added delta")
    changed = _mapping(delta.get("changed"), field="service source changed delta")
    removed = _mapping(delta.get("removed"), field="service source removed delta")
    if not EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS.issubset(current_runtime):
        raise ReleaseEvalV2PerformanceError(
            "approved performance/portability source is absent from the runtime closure"
        )
    expected_changed = {
        logical_id: current_runtime[logical_id]
        for logical_id in sorted(EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS)
    }
    if (
        dict(added) != {}
        or dict(changed) != expected_changed
        or dict(removed) != {}
        or delta.get("added_count") != 0
        or delta.get("changed_count") != len(EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS)
        or delta.get("removed_count") != 0
        or predecessor.get("runtime_package_file_count") != len(current_runtime)
        or predecessor.get("runtime_package_inventory_sha256")
        == source.get("runtime_package_inventory_sha256")
    ):
        raise ReleaseEvalV2PerformanceError(
            "service source successor delta is outside the approved performance/portability change"
        )
    summary = dict(delta)
    claimed_summary = _digest(
        summary.pop("summary_sha256", None), field="service source delta summary hash"
    )
    if canonical_json_hash(summary) != claimed_summary:
        raise ReleaseEvalV2PerformanceError("service source successor delta hash does not verify")


def _formal_replay_inputs(
    request: HarnessRequest,
    paths: FormalStageReplayPaths,
) -> InternalUnlockInputs:
    return InternalUnlockInputs(
        authorization_inputs=CalibrationAuthorizationInputs(
            evidence=SelectionReplayEvidence(
                protocol=paths.protocol,
                development_manifest=paths.development_manifest,
                v1_release_manifest=paths.v1_release_manifest,
                candidates=paths.candidates,
                selection_receipt=paths.selection_receipt,
                amendment=paths.amendment,
            ),
            dataset_manifest=paths.dataset_manifest,
            materialization_receipt=paths.materialization_receipt,
            freeze_receipt=paths.freeze_receipt,
            calibration_gold=paths.calibration_gold,
            model_artifact=request.model,
        ),
        authorization_receipt=paths.calibration_authorization,
        calibration=CalibrationInputs(
            bundle=request.calibration,
            diagnostics=paths.calibration_diagnostics,
            fit_gold=paths.calibration_gold,
            fit_predictions=paths.calibration_fit_predictions,
            fit_generation_receipt=paths.calibration_fit_generation_receipt,
            fit_prediction_manifest=paths.calibration_fit_prediction_manifest,
        ),
    )


def _formal_preopen_replay(request: HarnessRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly replay every upstream stage before touching the internal split."""

    paths = request.stage_replay
    if paths is None:
        raise ReleaseEvalV2PerformanceError("formal performance requires full stage replay inputs")
    try:
        replay = replay_internal_unlock(
            paths.internal_unlock,
            request.final_model_binding,
            request.service_configuration_binding,
            _formal_replay_inputs(request, paths),
        )
        mode = (
            "model-raw"
            if request.track == "model_raw"
            else "model-only"
            if request.track == "model_calibrated"
            else "cascade"
        )
        gate = guard_internal_preopen(
            paths.internal_unlock,
            request.final_model_binding,
            request.service_configuration_binding,
            input_path=request.source,
            mode=mode,
        )
    except ReleaseEvalV2StageGateError as exc:
        raise ReleaseEvalV2PerformanceError(
            f"formal internal unlock strict replay failed before input open: {exc}"
        ) from exc
    if (
        replay.get("status") != "INTERNAL_UNLOCK_STRICT_REPLAY_PASS"
        or replay.get("internal_evaluation_opened_hashed_or_decoded") is not False
        or gate.get("status") != "INTERNAL_PREOPEN_PASS"
    ):
        raise ReleaseEvalV2PerformanceError("formal stage replay did not authorize input open")
    return replay, gate


def _load_binding_pair(request: HarnessRequest) -> PreparedIdentity:
    model, model_file = _read_json(request.final_model_binding, field="final model binding")
    service, service_file = _read_json(
        request.service_configuration_binding,
        field="service configuration binding",
    )
    model_keys = {
        "schema_version",
        "model_id",
        "artifact_class",
        "label_count",
        "ordered_labels",
        "taxonomy_version",
        "attention_mode",
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "model_identity_sha256",
        "artifact_sha256",
        "manifest_sha256",
    }
    service_keys = {
        "schema_version",
        "service_id",
        "profile_id",
        "canonical_track",
        "final_model_id",
        "final_model_manifest_sha256",
        "model_identity_sha256",
        "calibration_bundle_file_sha256",
        "implementation_sha256",
        "configuration_sha256",
        "manifest_sha256",
    }
    _exact(model, model_keys, field="final model binding")
    _exact(service, service_keys, field="service configuration binding")
    _verify_self_hash(model, field="final model binding", key="manifest_sha256")
    _verify_self_hash(service, field="service configuration binding", key="manifest_sha256")
    if (
        model.get("schema_version") != "pii-zh.community-final-model-binding.v2"
        or model.get("artifact_class") != "24_label_zh_hans_token_classifier"
        or model.get("label_count") != 24
        or model.get("ordered_labels") != list(PII_CORE_LABELS)
        or model.get("attention_mode") != "full"
        or service.get("schema_version") != "pii-zh.community-service-configuration-binding.v2"
        or service.get("profile_id") != COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
        or service.get("canonical_track") != "full_system"
        or service.get("final_model_id") != model.get("model_id")
        or service.get("final_model_manifest_sha256") != model_file.sha256
        or service.get("model_identity_sha256") != model.get("model_identity_sha256")
    ):
        raise ReleaseEvalV2PerformanceError("final model/service bindings are inconsistent")
    _safe_id(model.get("model_id"), field="model ID")
    _safe_id(service.get("service_id"), field="service ID")
    for field in (
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "model_identity_sha256",
        "artifact_sha256",
    ):
        _digest(model.get(field), field=f"model {field}")
    for field in (
        "calibration_bundle_file_sha256",
        "implementation_sha256",
        "configuration_sha256",
    ):
        _digest(service.get(field), field=f"service {field}")

    calibration_file = _read_regular(
        request.calibration,
        field="calibration bundle",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    calibration = _strict_json(calibration_file.payload, field="calibration bundle")
    try:
        CalibrationBundle.from_dict(calibration)
    except (TypeError, ValueError) as exc:
        raise ReleaseEvalV2PerformanceError("calibration bundle is invalid") from exc
    if service.get("calibration_bundle_file_sha256") != calibration_file.sha256:
        raise ReleaseEvalV2PerformanceError("service binding does not match calibration bytes")

    try:
        actual_model = _model_identity(request.model)
        actual_service = _service_identity(
            model=actual_model,
            calibration={
                "bundle_file_sha256": calibration_file.sha256,
                "diagnostics_manifest_sha256": "0" * 64,
            },
            mode="cascade",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PerformanceError("actual model/service identity is invalid") from exc
    if (
        model.get("training_manifest_file_sha256")
        != actual_model.get("training_manifest_file_sha256")
        or model.get("training_manifest_sha256") != actual_model.get("training_manifest_sha256")
        or model.get("model_identity_sha256") != actual_model.get("identity_sha256")
        or model.get("artifact_sha256") != actual_model.get("output_artifact_sha256")
        or service.get("model_identity_sha256") != actual_model.get("identity_sha256")
        or service.get("implementation_sha256") != actual_service.get("implementation_sha256")
        or service.get("configuration_sha256") != actual_service.get("configuration_sha256")
    ):
        raise ReleaseEvalV2PerformanceError(
            "actual model/community builder differs from final bindings"
        )
    unlock: dict[str, Any] | None = None
    if request.stage_replay is not None:
        unlock, _unlock_file = _read_json(
            request.stage_replay.internal_unlock,
            field="internal unlock",
        )
    return PreparedIdentity(
        model=model,
        model_payload=model_file.payload,
        service=service,
        service_payload=service_file.payload,
        calibration=calibration,
        calibration_sha256=calibration_file.sha256,
        unlock=unlock,
    )


def _parse_workload(snapshot: FileSnapshot) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        decoded = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseEvalV2PerformanceError("candidate workload is not UTF-8") from exc
    document_ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
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
                raise ReleaseEvalV2PerformanceError("workload row must be an object")
            record = DocumentRecord.from_dict(value)
        except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
            raise ReleaseEvalV2PerformanceError(
                f"candidate workload line {line_number} violates canonical JSONL"
            ) from exc
        if record.doc_id in seen:
            raise ReleaseEvalV2PerformanceError("candidate workload has duplicate document IDs")
        seen.add(record.doc_id)
        document_ids.append(record.doc_id)
        texts.append(record.text)
    if not texts or len(texts) != snapshot.nonempty_lines:
        raise ReleaseEvalV2PerformanceError("candidate workload is empty or line count differs")
    return tuple(document_ids), tuple(texts)


def _validate_expected_predictions(
    snapshot: FileSnapshot,
    *,
    document_ids: Sequence[str],
) -> None:
    try:
        records = load_prediction_jsonl(io.StringIO(snapshot.payload.decode("utf-8")))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PerformanceError(
            "expected candidate predictions violate safe JSONL"
        ) from exc
    if [record.doc_id for record in records] != list(document_ids):
        raise ReleaseEvalV2PerformanceError(
            "expected candidate predictions differ in document identity/order"
        )


def _validate_device(device: Device) -> None:
    if device == "cuda:0":
        visible = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if len(visible) != 1:
            raise ReleaseEvalV2PerformanceError(
                "CUDA execution requires exactly one CUDA_VISIBLE_DEVICES entry"
            )


def _build_detector(request: HarnessRequest, identity: PreparedIdentity) -> Detector:
    labels = tuple(PII_CORE_LABELS)
    _validate_device(request.device)
    dtype: Any | None = None
    if request.device == "cuda:0":
        try:
            import torch

            if not torch.cuda.is_available():
                raise ReleaseEvalV2PerformanceError("isolated CUDA device is unavailable")
            dtype = torch.bfloat16
        except ImportError as exc:
            raise ReleaseEvalV2PerformanceError("CUDA runtime is unavailable") from exc
    if request.track == "model_raw":
        taxonomy = load_taxonomy()
        ignored = [entity.name for entity in taxonomy.label_sets["auxiliary"]]
        try:
            predictor = load_local_predictor(
                request.model,
                device=request.device,
                dtype=dtype,
                micro_batch_size=MICRO_BATCH_SIZE,
                ignored_labels=ignored,
                allowed_labels=labels,
            )
            recognizer = QwenPiiRecognizer(
                predictor=predictor,
                tokenizer=predictor.tokenizer,
                label_mapping={label: label for label in labels},
                ignored_labels=ignored,
                default_threshold=0.0,
                max_tokens=MAX_TOKENS,
                stride_fraction=STRIDE_FRACTION,
                model_version="release-eval-v2-performance-replay",
                attention_mode=predictor.attention_mode,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ReleaseEvalV2PerformanceError("model-raw detector initialization failed") from exc

        def model_raw(texts: Sequence[str]) -> Sequence[Sequence[Any]]:
            return recognizer.analyze_batch(texts, entities=labels)

        return model_raw

    mode = "model-only" if request.track == "model_calibrated" else "cascade"
    try:
        pipeline = build_community_model_service_pipeline(
            request.model,
            mode=cast(Any, mode),
            device=request.device,
            dtype=dtype,
            micro_batch_size=MICRO_BATCH_SIZE,
            calibration=identity.calibration,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PerformanceError(
            "community service detector initialization failed"
        ) from exc
    if (
        pipeline.config.profile_version != COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
        or pipeline.config.mode != mode
    ):
        raise ReleaseEvalV2PerformanceError("community builder returned a different profile")
    mapping = load_presidio_mapping()
    requested = tuple(mapping.model_to_presidio[label] for label in labels)

    def service(texts: Sequence[str]) -> Sequence[Sequence[Any]]:
        return pipeline.detect_batch(texts, entities=requested)

    return service


def _item_field(item: object, name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _normalize_predictions(
    *,
    document_ids: Sequence[str],
    texts: Sequence[str],
    detections: Sequence[Sequence[Any]],
    track: PerformanceTrack,
) -> list[PredictionRecord]:
    if len(detections) != len(texts):
        raise ReleaseEvalV2PerformanceError("detector did not return one result per document")
    reverse = {output: raw for raw, output in load_presidio_mapping().model_to_presidio.items()}
    allowed = frozenset(PII_CORE_LABELS)
    result: list[PredictionRecord] = []
    for doc_id, text, items in zip(document_ids, texts, detections, strict=True):
        spans: list[Span] = []
        for item in items:
            raw_label = _item_field(item, "entity_type")
            label = (
                raw_label
                if track == "model_raw"
                else reverse.get(raw_label)
                if isinstance(raw_label, str)
                else None
            )
            if not isinstance(label, str) or label not in allowed:
                raise ReleaseEvalV2PerformanceError("detector emitted a non-Open24 label")
            start = _item_field(item, "start")
            end = _item_field(item, "end")
            score = _item_field(item, "score")
            span = Span(
                start=cast(Any, start),
                end=cast(Any, end),
                label=label,
                score=cast(Any, score),
            )
            try:
                span.validate(text_length=len(text))
            except EvaluationDataError as exc:
                raise ReleaseEvalV2PerformanceError("detector emitted an invalid span") from exc
            spans.append(span)
        record = PredictionRecord(doc_id=doc_id, spans=tuple(spans))
        try:
            record.validate(text_length=len(text))
        except EvaluationDataError as exc:
            raise ReleaseEvalV2PerformanceError("detector output is not serializable") from exc
        result.append(record)
    return result


def _gpu_sync(device: Device) -> None:
    if device == "cuda:0":
        try:
            import torch

            torch.cuda.synchronize()
        except (ImportError, RuntimeError) as exc:
            raise ReleaseEvalV2PerformanceError("CUDA synchronization failed") from exc


def _gpu_reset(device: Device) -> None:
    if device == "cuda:0":
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError) as exc:
            raise ReleaseEvalV2PerformanceError("CUDA peak reset failed") from exc


def _gpu_peak_mb(device: Device) -> float:
    if device == "cpu":
        return 0.0
    try:
        import torch

        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except (ImportError, RuntimeError) as exc:
        raise ReleaseEvalV2PerformanceError("CUDA peak measurement failed") from exc


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if platform.system() == "Linux" else 1024.0 * 1024.0
    return float(value) / divisor


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ReleaseEvalV2PerformanceError("latency sample is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _batches(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(values[start : start + DOCUMENT_BATCH_SIZE])
        for start in range(0, len(values), DOCUMENT_BATCH_SIZE)
    )


def _measure_execution(
    request: HarnessRequest,
    *,
    identity: PreparedIdentity,
    input_snapshot: FileSnapshot,
    expected_snapshot: FileSnapshot,
) -> ExecutionObservation:
    document_ids, texts = _parse_workload(input_snapshot)
    _validate_expected_predictions(expected_snapshot, document_ids=document_ids)
    character_count = sum(len(text) for text in texts)
    if character_count < 1:
        raise ReleaseEvalV2PerformanceError("performance workload has no characters")
    detector = _build_detector(request, identity)
    text_batches = _batches(texts)
    id_batches = _batches(document_ids)
    _gpu_reset(request.device)
    for batch in text_batches[:WARMUP_BATCHES]:
        try:
            warmup = detector(batch)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ReleaseEvalV2PerformanceError("performance warmup inference failed") from exc
        if len(warmup) != len(batch):
            raise ReleaseEvalV2PerformanceError("warmup result count differs")
    _gpu_sync(request.device)

    output = io.StringIO()
    per_document_latency_ms: list[float] = []
    measured_count = 0
    overall_start = time.perf_counter_ns()
    for ids, batch in zip(id_batches, text_batches, strict=True):
        _gpu_sync(request.device)
        started = time.perf_counter_ns()
        try:
            raw = detector(batch)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ReleaseEvalV2PerformanceError("performance inference failed") from exc
        _gpu_sync(request.device)
        elapsed_ns = time.perf_counter_ns() - started
        normalized = (elapsed_ns / 1_000_000.0) / len(batch)
        per_document_latency_ms.extend(normalized for _ in batch)
        predictions = _normalize_predictions(
            document_ids=ids,
            texts=batch,
            detections=raw,
            track=request.track,
        )
        measured_count += write_prediction_jsonl(predictions, output)
    wall_time_ns = time.perf_counter_ns() - overall_start
    generated = output.getvalue().encode("utf-8")
    if measured_count != len(texts) or generated != expected_snapshot.payload:
        raise ReleaseEvalV2PerformanceError(
            "performance execution predictions are not byte-identical to frozen predictions"
        )
    seconds = wall_time_ns / 1_000_000_000.0
    measurement: dict[str, int | float] = {
        "sample_count": measured_count,
        "latency_p50_ms": round(_percentile(per_document_latency_ms, 0.50), 9),
        "latency_p95_ms": round(_percentile(per_document_latency_ms, 0.95), 9),
        "latency_p99_ms": round(_percentile(per_document_latency_ms, 0.99), 9),
        "throughput_documents_per_second": round(measured_count / seconds, 9),
        "throughput_characters_per_second": round(character_count / seconds, 9),
        "peak_cpu_rss_mb": round(_rss_mb(), 6),
        "peak_gpu_memory_mb": round(_gpu_peak_mb(request.device), 6),
    }
    _validate_measurement(
        measurement,
        document_count=measured_count,
        character_count=character_count,
        device=request.device,
    )
    return ExecutionObservation(
        measurement=measurement,
        input_sha256=input_snapshot.sha256,
        predictions_sha256=expected_snapshot.sha256,
        document_count=measured_count,
        character_count=character_count,
    )


def _number(value: object, *, field: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseEvalV2PerformanceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ReleaseEvalV2PerformanceError(f"{field} is outside its finite range")
    return result


def _validate_measurement(
    measurement: Mapping[str, Any],
    *,
    document_count: int,
    character_count: int,
    device: Device,
) -> None:
    expected = {
        "sample_count",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_documents_per_second",
        "throughput_characters_per_second",
        "peak_cpu_rss_mb",
        "peak_gpu_memory_mb",
    }
    _exact(measurement, expected, field="performance measurement")
    if measurement.get("sample_count") != document_count:
        raise ReleaseEvalV2PerformanceError("performance sample count is not full-corpus")
    p50 = _number(measurement.get("latency_p50_ms"), field="latency p50")
    p95 = _number(measurement.get("latency_p95_ms"), field="latency p95")
    p99 = _number(measurement.get("latency_p99_ms"), field="latency p99")
    docs = _number(measurement.get("throughput_documents_per_second"), field="document throughput")
    chars = _number(
        measurement.get("throughput_characters_per_second"), field="character throughput"
    )
    _number(measurement.get("peak_cpu_rss_mb"), field="peak CPU RSS")
    gpu = _number(
        measurement.get("peak_gpu_memory_mb"),
        field="peak GPU memory",
        positive=device == "cuda:0",
    )
    if not p50 <= p95 <= p99:
        raise ReleaseEvalV2PerformanceError("latency percentiles are not monotonic")
    if not math.isclose(
        chars / docs,
        character_count / document_count,
        rel_tol=1.0e-6,
        abs_tol=1.0e-9,
    ):
        raise ReleaseEvalV2PerformanceError("throughput metrics do not share one wall clock")
    if (device == "cpu" and gpu != 0.0) or (device == "cuda:0" and gpu <= 0.0):
        raise ReleaseEvalV2PerformanceError("GPU peak is inconsistent with runtime device")


def _validate_remeasurement(
    recorded: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    document_count: int,
    character_count: int,
    device: Device,
) -> None:
    _validate_measurement(
        recorded,
        document_count=document_count,
        character_count=character_count,
        device=device,
    )
    _validate_measurement(
        observed,
        document_count=document_count,
        character_count=character_count,
        device=device,
    )
    for field in (
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_documents_per_second",
        "throughput_characters_per_second",
        "peak_cpu_rss_mb",
    ):
        left = float(recorded[field])
        right = float(observed[field])
        ratio = right / left
        if not 1.0 / MAX_REPLAY_MEASUREMENT_RATIO <= ratio <= MAX_REPLAY_MEASUREMENT_RATIO:
            raise ReleaseEvalV2PerformanceError(
                "remeasured performance is implausibly far from the bound execution"
            )
    recorded_gpu = float(recorded["peak_gpu_memory_mb"])
    observed_gpu = float(observed["peak_gpu_memory_mb"])
    if (recorded_gpu == 0.0) != (observed_gpu == 0.0):
        raise ReleaseEvalV2PerformanceError("remeasured GPU resource class differs")
    if recorded_gpu > 0.0:
        ratio = observed_gpu / recorded_gpu
        if not 1.0 / MAX_REPLAY_MEASUREMENT_RATIO <= ratio <= MAX_REPLAY_MEASUREMENT_RATIO:
            raise ReleaseEvalV2PerformanceError("remeasured GPU peak is implausibly different")


def _implementation_identity(track: PerformanceTrack) -> tuple[str, str]:
    root = _repository_root()
    components: dict[str, str] = {}
    runtime_sources = _runtime_package_source_hashes(root)
    inventory = {
        "harness_cli": HARNESS_CLI_PATH,
        "harness_module": HARNESS_MODULE_PATH,
        "performance_schema": PERFORMANCE_SCHEMA_PATH,
        "release_replay_schema": REPLAY_SCHEMA_PATH,
        "quality_schema": QUALITY_SCHEMA_PATH,
        "service_source_schema": SOURCE_SCHEMA_PATH,
        "stage_gate": "src/pii_zh/evaluation/release_eval_v2_stage_gate.py",
        "prediction_provenance": ("src/pii_zh/evaluation/release_eval_v2_prediction_provenance.py"),
        "evaluation_io": "src/pii_zh/evaluation/io.py",
        "service_builder": "src/pii_zh/cascade/service_profiles.py",
        "service_pipeline": "src/pii_zh/cascade/pipeline.py",
        "service_routing": "src/pii_zh/cascade/routing.py",
        "rules": "src/pii_zh/rules/cn_common_v6.py",
        "validators": "src/pii_zh/data/validators/community.py",
        "model_loader": "src/pii_zh/inference/transformers_predictor.py",
        "recognizer": "src/pii_zh/presidio/qwen_recognizer.py",
    }
    for logical, relative in sorted(inventory.items()):
        snapshot = _read_regular(
            root / relative,
            field=f"performance implementation {logical}",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        components[logical] = snapshot.sha256
    generation = generation_implementation_identity(track)
    identity = canonical_json_hash(
        {
            "schema_version": "pii-zh.release-eval-v2-performance-implementation.v1",
            "fixed_runtime": {
                "document_batch_size": DOCUMENT_BATCH_SIZE,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "warmup_batches": WARMUP_BATCHES,
                "max_tokens": MAX_TOKENS,
                "stride_fraction": STRIDE_FRACTION,
            },
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "runtime_package_source_sha256": runtime_sources,
            "runtime_package_inventory_sha256": canonical_json_hash(runtime_sources),
            "runtime_package_file_count": len(runtime_sources),
            "runtime_package_inventory_complete": True,
            "component_sha256": components,
            "candidate_generation_identity": generation,
        }
    )
    return components["harness_cli"], identity


def _system_id(track: PerformanceTrack, identity: PreparedIdentity) -> str:
    if track == "full_system":
        return _safe_id(identity.service.get("service_id"), field="service ID")
    return _safe_id(identity.model.get("model_id"), field="model ID")


def build_performance_manifest(
    request: HarnessRequest,
    identity: PreparedIdentity,
    observation: ExecutionObservation,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "track": request.track,
        "system_id": _system_id(request.track, identity),
        "gold_sha256": observation.input_sha256,
        "predictions_sha256": observation.predictions_sha256,
        "final_model_manifest_sha256": _sha256(identity.model_payload),
        "service_configuration_manifest_sha256": _sha256(identity.service_payload),
        "measurement": observation.measurement,
    }
    value["manifest_sha256"] = canonical_json_hash(value)
    _validate_schema(value, PERFORMANCE_SCHEMA_PATH, field="performance manifest")
    _assert_path_free(value)
    return value


def _validate_performance_manifest(
    value: Mapping[str, Any],
    *,
    request: HarnessRequest,
    identity: PreparedIdentity,
    observation: ExecutionObservation,
) -> None:
    _validate_schema(value, PERFORMANCE_SCHEMA_PATH, field="performance manifest")
    _verify_self_hash(value, field="performance manifest", key="manifest_sha256")
    if (
        value.get("track") != request.track
        or value.get("system_id") != _system_id(request.track, identity)
        or value.get("gold_sha256") != observation.input_sha256
        or value.get("predictions_sha256") != observation.predictions_sha256
        or value.get("final_model_manifest_sha256") != _sha256(identity.model_payload)
        or value.get("service_configuration_manifest_sha256") != _sha256(identity.service_payload)
    ):
        raise ReleaseEvalV2PerformanceError("performance manifest identity differs")
    _validate_remeasurement(
        _mapping(value.get("measurement"), field="recorded measurement"),
        observation.measurement,
        document_count=observation.document_count,
        character_count=observation.character_count,
        device=request.device,
    )


def _execute(request: HarnessRequest) -> tuple[PreparedIdentity, ExecutionObservation]:
    if request.target_split == "internal_evaluation":
        _formal_replay, gate = _formal_preopen_replay(request)
    else:
        if request.stage_replay is not None:
            raise ReleaseEvalV2PerformanceError("fixture must not claim formal replay evidence")
        gate = None
    identity = _load_binding_pair(request)
    implementation_before = _implementation_identity(request.track)
    # This is the first open/hash/decode of the formal internal split.  Every
    # upstream strict replay and final identity check above has already passed.
    input_snapshot = _read_regular(
        request.source,
        field="candidate performance input",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    if gate is not None and input_snapshot.sha256 != gate.get("expected_input_sha256"):
        raise ReleaseEvalV2PerformanceError("input snapshot differs from internal unlock")
    expected_snapshot = _read_regular(
        request.expected_predictions,
        field="frozen candidate predictions",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    observation = _measure_execution(
        request,
        identity=identity,
        input_snapshot=input_snapshot,
        expected_snapshot=expected_snapshot,
    )
    try:
        model_after = _model_identity(request.model)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PerformanceError("model identity changed during execution") from exc
    if model_after.get("identity_sha256") != identity.model.get("model_identity_sha256"):
        raise ReleaseEvalV2PerformanceError("model identity changed during execution")
    if _implementation_identity(request.track) != implementation_before:
        raise ReleaseEvalV2PerformanceError("performance implementation changed during execution")
    return identity, observation


def measure(request: HarnessRequest) -> dict[str, Any]:
    """Measure one full candidate pass and return a quality-compatible manifest."""

    identity, observation = _execute(request)
    return build_performance_manifest(request, identity, observation)


def _validate_quality_context(
    *,
    quality: Mapping[str, Any],
    quality_snapshot: FileSnapshot,
    source: Mapping[str, Any],
    performance: Mapping[str, Any],
    performance_snapshot: FileSnapshot,
    request: HarnessRequest,
    identity: PreparedIdentity,
    observation: ExecutionObservation,
) -> dict[str, str]:
    if request.track not in {"model_raw", "full_system"}:
        raise ReleaseEvalV2PerformanceError(
            "model_calibrated is diagnostic-only and has no community release replay slot"
        )
    track = cast(CanonicalReleaseTrack, request.track)
    _validate_schema(quality, QUALITY_SCHEMA_PATH, field="quality receipt")
    _verify_self_hash(quality, field="quality receipt", key="receipt_sha256")
    _validate_source_successor(source)
    quality_bindings = _mapping(quality.get("input_bindings"), field="quality bindings")
    track_bindings = _mapping(
        _mapping(quality_bindings.get("tracks"), field="quality track bindings").get(track),
        field="quality canonical track bindings",
    )
    gold_binding = _mapping(quality_bindings.get("gold"), field="quality gold binding")
    prediction_binding = _mapping(
        track_bindings.get("candidate_predictions"),
        field="quality candidate prediction binding",
    )
    performance_binding = _mapping(
        track_bindings.get("performance_manifest"),
        field="quality performance binding",
    )
    model_binding = _mapping(
        quality_bindings.get("final_model_manifest"),
        field="quality model binding",
    )
    service_binding = _mapping(
        quality_bindings.get("service_configuration_manifest"),
        field="quality service binding",
    )
    quality_tracks = _mapping(quality.get("tracks"), field="quality tracks")
    quality_track = _mapping(quality_tracks.get(track), field="quality canonical track")
    candidate = _mapping(quality.get("candidate"), field="quality candidate")
    if (
        quality.get("reported_status") != "PASS"
        or quality_snapshot.sha256 != _sha256(quality_snapshot.payload)
        or gold_binding.get("file_sha256") != observation.input_sha256
        or prediction_binding.get("file_sha256") != observation.predictions_sha256
        or performance_binding
        != {
            "file_sha256": performance_snapshot.sha256,
            "size_bytes": performance_snapshot.size_bytes,
        }
        or model_binding
        != {
            "file_sha256": _sha256(identity.model_payload),
            "size_bytes": len(identity.model_payload),
        }
        or service_binding
        != {
            "file_sha256": _sha256(identity.service_payload),
            "size_bytes": len(identity.service_payload),
        }
        or quality_track.get("performance") != performance.get("measurement")
        or quality_track.get("candidate_id") != performance.get("system_id")
        or candidate.get("model_id") != identity.model.get("model_id")
        or candidate.get("service_id") != identity.service.get("service_id")
        or source.get("service_id") != identity.service.get("service_id")
        or source.get("profile_id") != identity.service.get("profile_id")
        or source.get("implementation_sha256") != identity.service.get("implementation_sha256")
        or identity.unlock is None
    ):
        raise ReleaseEvalV2PerformanceError(
            "quality/source/performance identities do not form one release chain"
        )
    unlock = identity.unlock
    assert unlock is not None
    return {
        "model_id": _safe_id(identity.model.get("model_id"), field="model ID"),
        "service_id": _safe_id(identity.service.get("service_id"), field="service ID"),
        "training_manifest_sha256": _digest(
            identity.model.get("training_manifest_sha256"),
            field="training manifest self hash",
        ),
        "model_identity_sha256": _digest(
            identity.model.get("model_identity_sha256"), field="model identity hash"
        ),
        "calibration_bundle_file_sha256": _digest(
            identity.service.get("calibration_bundle_file_sha256"),
            field="calibration bundle hash",
        ),
        "service_configuration_sha256": _digest(
            identity.service.get("configuration_sha256"),
            field="service configuration hash",
        ),
        "service_implementation_sha256": _digest(
            identity.service.get("implementation_sha256"),
            field="service implementation hash",
        ),
        "source_manifest_sha256": _digest(
            source.get("manifest_sha256"), field="source manifest hash"
        ),
        "quality_receipt_sha256": _digest(
            quality.get("receipt_sha256"), field="quality receipt hash"
        ),
        "internal_unlock_sha256": _digest(
            unlock.get("receipt_sha256"), field="internal unlock hash"
        ),
    }


def build_release_replay_receipt(
    *,
    track: CanonicalReleaseTrack,
    quality: Mapping[str, Any],
    quality_file_sha256: str,
    performance: Mapping[str, Any],
    performance_file_sha256: str,
    identity: Mapping[str, str],
    implementation_file_sha256: str,
    implementation_identity_sha256: str,
) -> dict[str, Any]:
    """Build the exact core independently re-derived by the v2 release gate."""

    target = {
        "artifact_kind": "performance_manifest",
        "file_sha256": performance_file_sha256,
        "canonical_sha256": _digest(
            performance.get("manifest_sha256"), field="performance self hash"
        ),
    }
    input_material = {
        "receipt_type": "performance_harness_replay",
        "track": track,
        "quality_file_sha256": quality_file_sha256,
        "quality_receipt_sha256": quality["receipt_sha256"],
        "identity": dict(identity),
        "quality_input_bindings": quality["input_bindings"],
        "target": target,
        "quality_track": quality["tracks"][track],
    }
    output_material = {
        "target": target,
        "reported_status": quality["reported_status"],
        "track_status": quality["tracks"][track]["declared_status"],
    }
    receipt: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "receipt_type": "performance_harness_replay",
        "track": track,
        "status": "EXACT_REPLAY_PASS",
        "quality": {
            "file_sha256": quality_file_sha256,
            "receipt_sha256": quality["receipt_sha256"],
        },
        "target": target,
        "identity": dict(identity),
        "replay": {
            "implementation_file_sha256": implementation_file_sha256,
            "implementation_identity_sha256": implementation_identity_sha256,
            "input_set_sha256": canonical_json_hash(input_material),
            "output_set_sha256": canonical_json_hash(output_material),
            "mode": "full_exact_replay",
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "aggregate_only": True,
            "network_used": False,
        },
    }
    receipt["receipt_id"] = canonical_json_hash(receipt)
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    _validate_schema(receipt, REPLAY_SCHEMA_PATH, field="performance replay receipt")
    _assert_path_free(receipt)
    return receipt


def replay(
    request: HarnessRequest,
    *,
    performance_manifest: Path,
    quality_receipt: Path,
    service_source_manifest: Path,
) -> dict[str, Any]:
    """Rerun inference/measurement and return a release-v2 replay receipt."""

    if request.target_split != "internal_evaluation":
        raise ReleaseEvalV2PerformanceError("release replay is only valid for formal internal")
    performance, performance_snapshot = _read_json(
        performance_manifest,
        field="performance manifest",
    )
    quality, quality_snapshot = _read_json(quality_receipt, field="quality receipt")
    source, _source_snapshot = _read_json(
        service_source_manifest,
        field="service source manifest",
    )
    # Fail before the expensive inference pass when the successor source
    # closure is stale, malformed, or from the predecessor schema.
    _validate_source_successor(source)
    identity, observation = _execute(request)
    _validate_performance_manifest(
        performance,
        request=request,
        identity=identity,
        observation=observation,
    )
    release_identity = _validate_quality_context(
        quality=quality,
        quality_snapshot=quality_snapshot,
        source=source,
        performance=performance,
        performance_snapshot=performance_snapshot,
        request=request,
        identity=identity,
        observation=observation,
    )
    implementation_file, implementation_identity = _implementation_identity(request.track)
    return build_release_replay_receipt(
        track=cast(CanonicalReleaseTrack, request.track),
        quality=quality,
        quality_file_sha256=quality_snapshot.sha256,
        performance=performance,
        performance_file_sha256=performance_snapshot.sha256,
        identity=release_identity,
        implementation_file_sha256=implementation_file,
        implementation_identity_sha256=implementation_identity,
    )


def write_read_only_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a new mode-0444 JSON file with no overwrite window."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ReleaseEvalV2PerformanceError("refusing to overwrite performance evidence")
    payload = _pretty_bytes(value)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, raw_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        temporary = Path(raw_name)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("short write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, destination, follow_symlinks=False)
    except OSError as exc:
        raise ReleaseEvalV2PerformanceError("cannot publish performance evidence") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256(payload)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track",
        required=True,
        choices=("model_raw", "model_calibrated", "full_system"),
    )
    parser.add_argument(
        "--target-split",
        required=True,
        choices=("fixture", "internal_evaluation"),
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-predictions", required=True, type=Path)
    parser.add_argument("--final-model-binding", required=True, type=Path)
    parser.add_argument("--service-configuration-binding", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")

    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--v1-release-manifest", type=Path)
    parser.add_argument("--candidate", action="append", type=Path)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--materialization-receipt", type=Path)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--calibration-gold", type=Path)
    parser.add_argument("--calibration-authorization", type=Path)
    parser.add_argument("--calibration-diagnostics", type=Path)
    parser.add_argument("--calibration-fit-predictions", type=Path)
    parser.add_argument("--calibration-fit-generation-receipt", type=Path)
    parser.add_argument("--calibration-fit-prediction-manifest", type=Path)
    parser.add_argument("--internal-unlock", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    _add_common_arguments(measure_parser)
    measure_parser.add_argument("--output", required=True, type=Path)

    replay_parser = commands.add_parser("replay")
    _add_common_arguments(replay_parser)
    replay_parser.add_argument("--performance-manifest", required=True, type=Path)
    replay_parser.add_argument("--quality-receipt", required=True, type=Path)
    replay_parser.add_argument("--service-source-manifest", required=True, type=Path)
    replay_parser.add_argument("--output", required=True, type=Path)
    return parser


def _stage_paths(args: argparse.Namespace) -> FormalStageReplayPaths | None:
    names = (
        "protocol",
        "development_manifest",
        "v1_release_manifest",
        "selection_receipt",
        "amendment",
        "dataset_manifest",
        "materialization_receipt",
        "freeze_receipt",
        "calibration_gold",
        "calibration_authorization",
        "calibration_diagnostics",
        "calibration_fit_predictions",
        "calibration_fit_generation_receipt",
        "calibration_fit_prediction_manifest",
        "internal_unlock",
    )
    values = [getattr(args, name) for name in names]
    candidates = tuple(args.candidate or ())
    if args.target_split == "fixture":
        if any(value is not None for value in values) or candidates:
            raise ReleaseEvalV2PerformanceError("fixture must not receive formal stage inputs")
        return None
    if any(value is None for value in values) or len(candidates) != 3:
        raise ReleaseEvalV2PerformanceError(
            "formal internal requires every stage input and exactly three candidates"
        )
    return FormalStageReplayPaths(
        protocol=cast(Path, args.protocol),
        development_manifest=cast(Path, args.development_manifest),
        v1_release_manifest=cast(Path, args.v1_release_manifest),
        candidates=cast(tuple[Path, Path, Path], candidates),
        selection_receipt=cast(Path, args.selection_receipt),
        amendment=cast(Path, args.amendment),
        dataset_manifest=cast(Path, args.dataset_manifest),
        materialization_receipt=cast(Path, args.materialization_receipt),
        freeze_receipt=cast(Path, args.freeze_receipt),
        calibration_gold=cast(Path, args.calibration_gold),
        calibration_authorization=cast(Path, args.calibration_authorization),
        calibration_diagnostics=cast(Path, args.calibration_diagnostics),
        calibration_fit_predictions=cast(Path, args.calibration_fit_predictions),
        calibration_fit_generation_receipt=cast(Path, args.calibration_fit_generation_receipt),
        calibration_fit_prediction_manifest=cast(Path, args.calibration_fit_prediction_manifest),
        internal_unlock=cast(Path, args.internal_unlock),
    )


def _request(args: argparse.Namespace) -> HarnessRequest:
    return HarnessRequest(
        track=cast(PerformanceTrack, args.track),
        target_split=cast(Literal["fixture", "internal_evaluation"], args.target_split),
        model=args.model,
        calibration=args.calibration,
        source=args.input,
        expected_predictions=args.expected_predictions,
        final_model_binding=args.final_model_binding,
        service_configuration_binding=args.service_configuration_binding,
        device=cast(Device, args.device),
        stage_replay=_stage_paths(args),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        request = _request(args)
        if args.output.exists() or args.output.is_symlink():
            raise ReleaseEvalV2PerformanceError("refusing to overwrite performance evidence")
        if args.command == "measure":
            value = measure(request)
            file_sha256 = write_read_only_json(args.output, value)
            result = {
                "status": "PASS",
                "track": request.track,
                "performance_manifest_file_sha256": file_sha256,
                "performance_manifest_sha256": value["manifest_sha256"],
            }
        else:
            value = replay(
                request,
                performance_manifest=args.performance_manifest,
                quality_receipt=args.quality_receipt,
                service_source_manifest=args.service_source_manifest,
            )
            file_sha256 = write_read_only_json(args.output, value)
            result = {
                "status": value["status"],
                "track": request.track,
                "replay_receipt_file_sha256": file_sha256,
                "replay_receipt_sha256": value["receipt_sha256"],
                "implementation_file_sha256": value["replay"]["implementation_file_sha256"],
                "implementation_identity_sha256": value["replay"]["implementation_identity_sha256"],
            }
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


__all__ = [
    "DOCUMENT_BATCH_SIZE",
    "FormalStageReplayPaths",
    "HarnessRequest",
    "MICRO_BATCH_SIZE",
    "PERFORMANCE_SCHEMA_PATH",
    "PERFORMANCE_SCHEMA_VERSION",
    "ReleaseEvalV2PerformanceError",
    "WARMUP_BATCHES",
    "build_performance_manifest",
    "build_release_replay_receipt",
    "canonical_json_hash",
    "main",
    "measure",
    "replay",
    "write_read_only_json",
]
