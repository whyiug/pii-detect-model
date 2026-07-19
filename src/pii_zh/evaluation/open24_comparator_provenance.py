"""Reproducible Open-24 comparators for frozen release-eval v2.

This module owns two deliberately narrow comparison systems:

* the pinned, unmodified AIguard checkpoint at threshold zero on the Open-24
  metric (only the twelve core labels with an explicit upstream mapping can be
  emitted); and
* native Presidio with the pinned CLUENER checkpoint plus ``cn_common_v6``.

The generation API is evidence-first.  It strictly replays the v1-to-v2
amendment and the internal-evaluation final freeze before opening the target
JSONL, snapshots that JSONL through one ``O_NOFOLLOW`` descriptor, writes a
raw-text-free prediction file, and publishes a closed content-addressed run
bundle.  Full execution replay reruns inference and compares prediction bytes;
metadata validation alone is never reported as generation replay.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from jsonschema import Draft202012Validator

from pii_zh.data.schema import DocumentRecord, iter_jsonl
from pii_zh.evaluation.io import (
    EvaluationDataError,
    PredictionRecord,
    Span,
    load_prediction_jsonl,
    write_prediction_jsonl,
)
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.taxonomy import load_presidio_mapping

ComparatorTrack = Literal["model_raw", "full_system"]
DeviceClass = Literal["cpu", "cuda"]

SCHEMA_VERSION = "pii-zh.open24-comparator-prediction-provenance.v1"
MANIFEST_TYPE = "open24_comparator_prediction_provenance"
GENERATION_RECEIPT_SCHEMA_VERSION = "pii-zh.open24-comparator-generation-receipt.v1"
MEASUREMENT_SCHEMA_VERSION = "pii-zh.open24-comparator-execution-measurement.v1"
SUPPORT_SCHEMA_VERSION = "pii-zh.open24-comparator-support-document.v1"
STAGE_SPEC_SCHEMA_VERSION = "pii-zh.release-eval-v2-comparator-stage-replay-spec.v1"
EXECUTION_REPLAY_SCHEMA_VERSION = "pii-zh.open24-comparator-execution-replay.v1"

MANIFEST_SCHEMA_PATH = Path(
    "configs/evaluation/open24_comparator_prediction_provenance_v1.schema.json"
)
GENERATION_SCHEMA_PATH = Path(
    "configs/evaluation/open24_comparator_generation_receipt_v1.schema.json"
)
MEASUREMENT_SCHEMA_PATH = Path(
    "configs/evaluation/open24_comparator_execution_measurement_v1.schema.json"
)
EXECUTION_REPLAY_SCHEMA_PATH = Path(
    "configs/evaluation/open24_comparator_execution_replay_v1.schema.json"
)
GENERATOR_PATH = Path("scripts/generate_release_eval_v2_open24_comparators.py")

AIGUARD_SYSTEM_ID = "aiguard_pii_detection_fast_open24_model_raw_v1"
FRAMEWORK_SYSTEM_ID = "native_presidio_cluener_cn_common_v6_open24_v1"

AIGUARD_MODEL_ID = "ZJUICSR/AIguard-pii-detection-fast"
AIGUARD_REVISION = "677a5ebc1600fef61e8973cafd3026be322b3a73"
AIGUARD_LICENSE = "Apache-2.0"
AIGUARD_REQUIRED_FILES: Mapping[str, str] = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "README.md": "15e234add5b4640307e85523af2bab4b8ac87c564181e71e2aa9742f1c8475d1",
    "added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "chat_template.jinja": (
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ),
    "config.json": "5f8b5cafa13310bd5327e90beabc919307f749cb00714d4e87bff4d22cb31225",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "89c9b36aa98a155ea3255eb2c53280858c49b9fed790b43107c48a53158ec4b2",
    "special_tokens_map.json": (
        "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"
    ),
    "tokenizer.json": "352a863cd2761388ccc58f1432467ba6a1037bf12df9069889b142fa246471f6",
    "tokenizer_config.json": (
        "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
    ),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
AIGUARD_SOURCE_TO_CORE24: Mapping[str, str] = {
    "address": "ADDRESS",
    "bank_card": "BANK_CARD_NUMBER",
    "bank_password": "SECRET",
    "birth_date": "DATE_OF_BIRTH",
    "credit_card": "BANK_CARD_NUMBER",
    "drivers_license": "DRIVER_LICENSE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "id_card": "CN_RESIDENT_ID",
    "mobile": "PHONE_NUMBER",
    "name": "PERSON_NAME",
    "passport": "PASSPORT_NUMBER",
    "plate_number": "VEHICLE_LICENSE_PLATE",
    "social_security": "SOCIAL_SECURITY_NUMBER",
}

CLUENER_MODEL_ID = "uer/roberta-base-finetuned-cluener2020-chinese"
CLUENER_REVISION = "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b"
CLUENER_LICENSE = "NOASSERTION-upstream-model-card"
CLUENER_REQUIRED_FILES: Mapping[str, str] = {
    "README.md": "a24882ff7dc060a5ec4b0d376d510bfacab57a00973915cc53255f8261f41865",
    "config.json": "a80e368cebeaeb01ed195f068b01a65f5b01b87e8059434710a01f3b97a38598",
    "conversion_receipt.json": (
        "1a387e51fa9be2cc3c507527b154f95a8bdc1f442c0e3fa37b4e89dce77ada70"
    ),
    "model.safetensors": (
        "0902a7ced1d20dd65a1862f8cd03484495a6b322e59c1d0e294dc66c67723177"
    ),
    "special_tokens_map.json": (
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
    ),
    "tokenizer.json": "7dfbf1966ebf99d471c3796e9b457329d2b2182b817e144f1e904b957745c839",
    "tokenizer_config.json": (
        "21b0a2fba3d74b521cdbd631b2936c2c063c856c3a05533fabee04005c02cd23"
    ),
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}
CLUENER_SOURCE_TO_CORE24: Mapping[str, str] = {
    "address": "ADDRESS",
    "name": "PERSON_NAME",
}
CLUENER_CONVERSION_RECEIPT: Mapping[str, Any] = {
    "converter_sha256": "fe9ba3d29c151688ba04e6cf9993d1bf8c0e9cd50f1679701d271f4ce766651a",
    "destination_sha256": CLUENER_REQUIRED_FILES["model.safetensors"],
    "loader": "torch.load(weights_only=True,map_location=cpu)",
    "payload_contract": "flat_str_to_tensor_state_dict",
    "source_deleted": True,
    "source_sha256": "b865252516115c46bc508167fa2258f198bcce520eb7a1ac4fbf8d50dc361368",
    "verification": "safetensors_full_tensor_equality",
}

MAX_TOKENS = 512
STRIDE_FRACTION = 0.25
THRESHOLD = 0.0
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 2 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}\Z")
_UNSAFE_MODEL_SUFFIXES = frozenset({".bin", ".ckpt", ".pickle", ".pkl", ".pt", ".pth"})

_SUPPORT_FILENAMES: Mapping[str, str] = {
    "source_closure_manifest": "source_closure.json",
    "artifact_or_configuration": "artifact_identity.json",
    "generator_implementation": "generator_implementation.json",
    "runtime_configuration": "runtime_configuration.json",
    "environment_lock": "environment_lock.json",
}


class Open24ComparatorError(RuntimeError):
    """A fail-closed comparator error which never includes row contents."""


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Frozen inference controls for one comparator run."""

    device: str = "cpu"
    document_batch_size: int = 16
    micro_batch_size: int = 16
    max_tokens: int = MAX_TOKENS
    stride_fraction: float = STRIDE_FRACTION


@dataclass(frozen=True, slots=True)
class StageReplaySpec:
    """Local-only paths needed to replay the final pre-open gate."""

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
    selected_model_artifact: Path
    calibration_authorization: Path
    calibration_bundle: Path
    calibration_diagnostics: Path
    calibration_fit_gold: Path
    calibration_fit_predictions: Path
    calibration_fit_generation_receipt: Path
    calibration_fit_prediction_manifest: Path
    internal_unlock: Path
    final_model_binding: Path
    service_configuration_binding: Path


@dataclass(frozen=True, slots=True)
class ComparatorRunInputs:
    """Inputs for generation or strict execution replay."""

    track: ComparatorTrack
    source_model: Path
    input_path: Path
    output_dir: Path
    stage_spec: StageReplaySpec
    runtime: RuntimeOptions


@dataclass(frozen=True, slots=True)
class FileIdentity:
    sha256: str
    size_bytes: int
    nonempty_lines: int | None = None


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    payload: bytes
    identity: FileIdentity


@dataclass(frozen=True, slots=True)
class StageEvidence:
    expected_input_sha256: str
    expected_input_size_bytes: int
    expected_document_count: int
    dataset: Mapping[str, Any]
    bindings: Mapping[str, Any]


class BatchDetector(Protocol):
    """Return one raw span sequence per input string."""

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[Any]]: ...


StageAuthorizer = Callable[[StageReplaySpec, Path, ComparatorTrack], StageEvidence]
ArtifactInspector = Callable[[ComparatorTrack, Path], dict[str, Any]]
DetectorFactory = Callable[[ComparatorTrack, Path, RuntimeOptions], BatchDetector]


@dataclass(frozen=True, slots=True)
class ComparatorHooks:
    """Injectable execution boundaries used by deterministic fixture replay."""

    stage_authorizer: StageAuthorizer
    artifact_inspector: ArtifactInspector
    detector_factory: DetectorFactory


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
        raise Open24ComparatorError("comparator evidence is not canonical JSON") from exc


def canonical_hash(value: object) -> str:
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
        raise Open24ComparatorError("comparator evidence is not strict JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Open24ComparatorError("JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise Open24ComparatorError("JSON contains a non-finite number")


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Open24ComparatorError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Open24ComparatorError(f"{field} must be a JSON object")
    return value


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
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


def _read_regular(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
    count_lines: bool = False,
    retain_payload: bool = True,
) -> tuple[bytes, FileIdentity]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Open24ComparatorError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise Open24ComparatorError(f"{field} must be a bounded regular file")
        payload = bytearray()
        digest = hashlib.sha256()
        total_bytes = 0
        nonempty_lines = 0
        pending = b""
        while block := os.read(descriptor, 1024 * 1024):
            total_bytes += len(block)
            if retain_payload:
                payload.extend(block)
            digest.update(block)
            if total_bytes > maximum_bytes:
                raise Open24ComparatorError(f"{field} exceeds its size limit")
            if count_lines:
                pieces = (pending + block).split(b"\n")
                pending = pieces.pop()
                nonempty_lines += sum(bool(piece.strip()) for piece in pieces)
        if count_lines and pending.strip():
            nonempty_lines += 1
        after = os.fstat(descriptor)
        if not _same_file(before, after) or total_bytes != after.st_size:
            raise Open24ComparatorError(f"{field} changed while read")
        return bytes(payload), FileIdentity(
            sha256=digest.hexdigest(),
            size_bytes=total_bytes,
            nonempty_lines=nonempty_lines if count_lines else None,
        )
    except OSError as exc:
        raise Open24ComparatorError(f"{field} could not be read") from exc
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, field: str) -> tuple[dict[str, Any], FileIdentity]:
    payload, identity = _read_regular(path, field=field, maximum_bytes=_MAX_JSON_BYTES)
    return _strict_json(payload, field=field), identity


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Open24ComparatorError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise Open24ComparatorError(f"{field} must be a path-free stable identifier")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Open24ComparatorError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, field: str) -> None:
    if set(value) != keys:
        raise Open24ComparatorError(f"{field} has an invalid closed shape")


def _self_hashed(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = canonical_hash(result)
    return result


def _verify_self_hash(document: Mapping[str, Any], *, field: str, key: str) -> str:
    claimed = _require_digest(document.get(key), field=f"{field}.{key}")
    unsigned = dict(document)
    unsigned.pop(key, None)
    if canonical_hash(unsigned) != claimed:
        raise Open24ComparatorError(f"{field} self hash does not verify")
    return claimed


def _assert_no_local_paths(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_no_local_paths(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_local_paths(nested)
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "~\\", "file:"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise Open24ComparatorError("public comparator evidence contains a local path")


def system_id_for_track(track: ComparatorTrack) -> str:
    if track == "model_raw":
        return AIGUARD_SYSTEM_ID
    if track == "full_system":
        return FRAMEWORK_SYSTEM_ID
    raise Open24ComparatorError("unsupported comparator track")


def _source_metadata(track: ComparatorTrack) -> dict[str, Any]:
    if track == "model_raw":
        return {
            "kind": "open_source_model",
            "repository": f"https://huggingface.co/{AIGUARD_MODEL_ID}",
            "revision": AIGUARD_REVISION,
            "license": AIGUARD_LICENSE,
            "public_source": True,
        }
    return {
        "kind": "open_source_framework",
        "repository": (
            "https://github.com/microsoft/presidio + "
            f"https://huggingface.co/{CLUENER_MODEL_ID}"
        ),
        "revision": f"presidio-analyzer-2.2.x+cluener-{CLUENER_REVISION}+cn_common_v6",
        "license": "Apache-2.0-framework-and-rules;NOASSERTION-cluener-checkpoint",
        "public_source": True,
    }


def _inspect_model_directory(
    path: Path,
    *,
    required: Mapping[str, str],
    field: str,
) -> tuple[Path, dict[str, str]]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise Open24ComparatorError(f"{field} directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Open24ComparatorError(f"{field} directory is unavailable") from exc
    if not root.is_dir():
        raise Open24ComparatorError(f"{field} must be a local directory")
    top_level_files = {item.name for item in root.iterdir() if item.is_file()}
    if top_level_files != set(required):
        raise Open24ComparatorError(f"{field} top-level file inventory changed")
    observed: dict[str, str] = {}
    for name, expected in sorted(required.items()):
        artifact = root / name
        if artifact.is_symlink():
            raise Open24ComparatorError(f"{field} contains a symlink artifact")
        _payload, identity = _read_regular(
            artifact,
            field=f"{field} artifact",
            maximum_bytes=64 * 1024 * 1024 * 1024,
            retain_payload=False,
        )
        if identity.sha256 != expected:
            raise Open24ComparatorError(f"{field} does not match the pinned revision")
        observed[name] = identity.sha256
    unsafe = sorted(
        item.name
        for item in root.iterdir()
        if item.is_file() and item.suffix.casefold() in _UNSAFE_MODEL_SUFFIXES
    )
    if unsafe:
        raise Open24ComparatorError(f"{field} contains unsafe serialized weights")
    return root, observed


def inspect_comparator_artifact(track: ComparatorTrack, path: Path) -> dict[str, Any]:
    """Verify actual local bytes against the one admitted source revision."""

    if track == "model_raw":
        _root, observed = _inspect_model_directory(
            path,
            required=AIGUARD_REQUIRED_FILES,
            field="pinned AIguard model",
        )
        payload: dict[str, Any] = {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "document_type": "comparator_artifact_identity",
            "system_id": AIGUARD_SYSTEM_ID,
            "artifact_type": "huggingface_safetensors_token_classifier",
            "source_model_id": AIGUARD_MODEL_ID,
            "source_revision": AIGUARD_REVISION,
            "files": observed,
            "files_combined_sha256": canonical_hash(observed),
            "conversion": None,
            "local_path_included": False,
        }
    else:
        root, observed = _inspect_model_directory(
            path,
            required=CLUENER_REQUIRED_FILES,
            field="pinned CLUENER model",
        )
        if any((root / name).exists() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
            raise Open24ComparatorError(
                "CLUENER license evidence changed and requires a new source review"
            )
        receipt, _identity = _read_json(
            root / "conversion_receipt.json",
            field="CLUENER safe conversion receipt",
        )
        if any(
            receipt.get(name) != expected
            for name, expected in CLUENER_CONVERSION_RECEIPT.items()
        ):
            raise Open24ComparatorError("CLUENER safe conversion receipt changed")
        payload = {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "document_type": "comparator_artifact_identity",
            "system_id": FRAMEWORK_SYSTEM_ID,
            "artifact_type": "huggingface_safetensors_bios_token_classifier",
            "source_model_id": CLUENER_MODEL_ID,
            "source_revision": CLUENER_REVISION,
            "files": observed,
            "files_combined_sha256": canonical_hash(observed),
            "conversion": receipt,
            "local_path_included": False,
        }
    result = _self_hashed(payload, field="document_sha256")
    _assert_no_local_paths(result)
    return result


def _model_stat_snapshot(track: ComparatorTrack, path: Path) -> tuple[tuple[Any, ...], ...]:
    """Retain local inode metadata across verification and Transformers load."""

    required = AIGUARD_REQUIRED_FILES if track == "model_raw" else CLUENER_REQUIRED_FILES
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise Open24ComparatorError("comparator model directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Open24ComparatorError("comparator model directory is unavailable") from exc
    snapshot: list[tuple[Any, ...]] = []
    for name in sorted(required):
        artifact = root / name
        try:
            observed = artifact.lstat()
        except OSError as exc:
            raise Open24ComparatorError("comparator model artifact is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise Open24ComparatorError("comparator model artifact type changed")
        snapshot.append(
            (
                name,
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
        )
    return tuple(snapshot)


def load_stage_replay_spec(path: Path) -> StageReplaySpec:
    """Load a private local path inventory; its paths are never published."""

    document, _identity = _read_json(path, field="comparator stage replay spec")
    _exact_keys(
        document,
        {
            "schema_version",
            "protocol",
            "development_manifest",
            "v1_release_manifest",
            "candidates",
            "selection_receipt",
            "amendment",
            "dataset_manifest",
            "materialization_receipt",
            "freeze_receipt",
            "calibration_gold",
            "selected_model_artifact",
            "calibration_authorization",
            "calibration_bundle",
            "calibration_diagnostics",
            "calibration_fit_gold",
            "calibration_fit_predictions",
            "calibration_fit_generation_receipt",
            "calibration_fit_prediction_manifest",
            "internal_unlock",
            "final_model_binding",
            "service_configuration_binding",
        },
        field="comparator stage replay spec",
    )
    if document.get("schema_version") != STAGE_SPEC_SCHEMA_VERSION:
        raise Open24ComparatorError("comparator stage replay spec version changed")
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise Open24ComparatorError("stage replay requires exactly three candidate paths")

    def local_path(name: str) -> Path:
        value = document.get(name)
        if not isinstance(value, str) or not value or "\x00" in value:
            raise Open24ComparatorError(f"stage replay field {name} must be a local path")
        return Path(value).expanduser()

    candidate_paths: list[Path] = []
    for value in candidates:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise Open24ComparatorError("stage replay candidate paths are invalid")
        candidate_paths.append(Path(value).expanduser())
    return StageReplaySpec(
        protocol=local_path("protocol"),
        development_manifest=local_path("development_manifest"),
        v1_release_manifest=local_path("v1_release_manifest"),
        candidates=cast(tuple[Path, Path, Path], tuple(candidate_paths)),
        selection_receipt=local_path("selection_receipt"),
        amendment=local_path("amendment"),
        dataset_manifest=local_path("dataset_manifest"),
        materialization_receipt=local_path("materialization_receipt"),
        freeze_receipt=local_path("freeze_receipt"),
        calibration_gold=local_path("calibration_gold"),
        selected_model_artifact=local_path("selected_model_artifact"),
        calibration_authorization=local_path("calibration_authorization"),
        calibration_bundle=local_path("calibration_bundle"),
        calibration_diagnostics=local_path("calibration_diagnostics"),
        calibration_fit_gold=local_path("calibration_fit_gold"),
        calibration_fit_predictions=local_path("calibration_fit_predictions"),
        calibration_fit_generation_receipt=local_path(
            "calibration_fit_generation_receipt"
        ),
        calibration_fit_prediction_manifest=local_path(
            "calibration_fit_prediction_manifest"
        ),
        internal_unlock=local_path("internal_unlock"),
        final_model_binding=local_path("final_model_binding"),
        service_configuration_binding=local_path("service_configuration_binding"),
    )


def authorize_internal_preopen(
    spec: StageReplaySpec,
    input_path: Path,
    track: ComparatorTrack,
) -> StageEvidence:
    """Strictly replay amendment/final freeze, then only ``lstat`` the target."""

    try:
        from scripts.build_aiguard24_v3_release_eval_v2_amendment import (
            validate_amendment,
        )

        from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
            CalibrationInputs,
        )
        from pii_zh.evaluation.release_eval_v2_stage_gate import (
            CalibrationAuthorizationInputs,
            InternalUnlockInputs,
            SelectionReplayEvidence,
            guard_internal_preopen,
            replay_internal_unlock,
        )

        amendment, amendment_file = _read_json(spec.amendment, field="formal v2 amendment")
        amendment_self = validate_amendment(amendment)
        authorization_inputs = CalibrationAuthorizationInputs(
            evidence=SelectionReplayEvidence(
                protocol=spec.protocol,
                development_manifest=spec.development_manifest,
                v1_release_manifest=spec.v1_release_manifest,
                candidates=spec.candidates,
                selection_receipt=spec.selection_receipt,
                amendment=spec.amendment,
            ),
            dataset_manifest=spec.dataset_manifest,
            materialization_receipt=spec.materialization_receipt,
            freeze_receipt=spec.freeze_receipt,
            calibration_gold=spec.calibration_gold,
            model_artifact=spec.selected_model_artifact,
        )
        calibration = CalibrationInputs(
            bundle=spec.calibration_bundle,
            diagnostics=spec.calibration_diagnostics,
            fit_gold=spec.calibration_fit_gold,
            fit_predictions=spec.calibration_fit_predictions,
            fit_generation_receipt=spec.calibration_fit_generation_receipt,
            fit_prediction_manifest=spec.calibration_fit_prediction_manifest,
        )
        unlock_inputs = InternalUnlockInputs(
            authorization_inputs=authorization_inputs,
            authorization_receipt=spec.calibration_authorization,
            calibration=calibration,
        )
        replay = replay_internal_unlock(
            spec.internal_unlock,
            spec.final_model_binding,
            spec.service_configuration_binding,
            unlock_inputs,
        )
        mode = "comparator-model-raw" if track == "model_raw" else "comparator-full-system"
        guard = guard_internal_preopen(
            spec.internal_unlock,
            spec.final_model_binding,
            spec.service_configuration_binding,
            input_path=input_path,
            mode=mode,
        )
        unlock, unlock_file = _read_json(spec.internal_unlock, field="internal unlock")
        upstream = _mapping(unlock.get("upstream"), field="internal unlock upstream")
        dataset = _mapping(unlock.get("dataset"), field="internal unlock dataset")
        if (
            replay.get("status") != "INTERNAL_UNLOCK_STRICT_REPLAY_PASS"
            or replay.get("internal_evaluation_opened_hashed_or_decoded") is not False
            or guard.get("status") != "INTERNAL_PREOPEN_PASS"
            or guard.get("expected_input_sha256") != dataset.get("gold_file_sha256")
            or upstream.get("amendment_file_sha256") != amendment_file.sha256
            or upstream.get("amendment_sha256") != amendment_self
            or replay.get("unlock_file_sha256") != unlock_file.sha256
        ):
            raise Open24ComparatorError(
                "amendment, internal unlock and comparator pre-open gate are inconsistent"
            )
        expected_size = dataset.get("gold_size_bytes")
        expected_count = dataset.get("gold_document_count")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 1
        ):
            raise Open24ComparatorError("internal unlock target counts are invalid")
        bindings = {
            "amendment_file_sha256": amendment_file.sha256,
            "amendment_sha256": amendment_self,
            "internal_unlock_file_sha256": replay["unlock_file_sha256"],
            "internal_unlock_sha256": replay["unlock_sha256"],
            "final_model_binding_file_sha256": replay[
                "final_model_binding_file_sha256"
            ],
            "final_model_binding_sha256": replay["final_model_binding_sha256"],
            "service_configuration_binding_file_sha256": replay[
                "service_configuration_binding_file_sha256"
            ],
            "service_configuration_binding_sha256": replay[
                "service_configuration_binding_sha256"
            ],
            "strict_replay_passed": True,
            "target_opened_hashed_or_decoded_during_stage_replay": False,
        }
        return StageEvidence(
            expected_input_sha256=_require_digest(
                dataset.get("gold_file_sha256"), field="internal gold hash"
            ),
            expected_input_size_bytes=expected_size,
            expected_document_count=expected_count,
            dataset={
                "dataset_id": _safe_id(dataset.get("dataset_id"), field="dataset ID"),
                "dataset_version": _safe_id(
                    dataset.get("dataset_version"), field="dataset version"
                ),
                "target_split": "internal_evaluation",
                "gold_file_sha256": _require_digest(
                    dataset.get("gold_file_sha256"), field="internal gold hash"
                ),
                "gold_size_bytes": expected_size,
                "gold_document_count": expected_count,
                "dataset_manifest_file_sha256": _require_digest(
                    dataset.get("dataset_manifest_file_sha256"),
                    field="dataset manifest file hash",
                ),
                "dataset_manifest_sha256": _require_digest(
                    dataset.get("dataset_manifest_sha256"),
                    field="dataset manifest self hash",
                ),
                "materialization_receipt_file_sha256": _require_digest(
                    dataset.get("materialization_receipt_file_sha256"),
                    field="materialization receipt file hash",
                ),
                "materialization_receipt_sha256": _require_digest(
                    dataset.get("materialization_receipt_sha256"),
                    field="materialization receipt self hash",
                ),
                "freeze_receipt_file_sha256": _require_digest(
                    dataset.get("freeze_receipt_file_sha256"),
                    field="freeze receipt file hash",
                ),
            },
            bindings=bindings,
        )
    except Open24ComparatorError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Open24ComparatorError(
            "strict amendment/internal-unlock pre-open replay failed"
        ) from exc


def snapshot_internal_input(path: Path, evidence: StageEvidence) -> InputSnapshot:
    """Open the formal input exactly once after authorization and retain bytes."""

    payload, identity = _read_regular(
        path,
        field="authorized internal-evaluation input",
        maximum_bytes=_MAX_JSONL_BYTES,
        count_lines=True,
    )
    if (
        identity.sha256 != evidence.expected_input_sha256
        or identity.size_bytes != evidence.expected_input_size_bytes
        or identity.nonempty_lines != evidence.expected_document_count
    ):
        raise Open24ComparatorError(
            "authorized input bytes/count differ from the replayed final freeze"
        )
    return InputSnapshot(payload=payload, identity=identity)


def _source_closure(track: ComparatorTrack) -> dict[str, Any]:
    mapping = (
        dict(sorted(AIGUARD_SOURCE_TO_CORE24.items()))
        if track == "model_raw"
        else dict(sorted(CLUENER_SOURCE_TO_CORE24.items()))
    )
    if not set(mapping.values()) <= set(PII_CORE_LABELS):
        raise Open24ComparatorError("comparator mapping escapes the Open-24 taxonomy")
    model_labels = sorted(set(mapping.values()), key=PII_CORE_LABELS.index)
    if track == "model_raw":
        rule_labels: list[str] = []
        semantics = "pinned_original_checkpoint_mapped_logits_threshold_zero"
    else:
        from pii_zh.rules.cn_common_v6 import CnCommonRulePackV6

        presidio = load_presidio_mapping()
        reverse = {value: key for key, value in presidio.model_to_presidio.items()}
        rule_labels = sorted(
            {
                reverse[definition.entity_type]
                for definition in CnCommonRulePackV6().rules
                if definition.entity_type in reverse
                and reverse[definition.entity_type] in PII_CORE_LABELS
            },
            key=PII_CORE_LABELS.index,
        )
        semantics = "native_presidio_analyzer_cluener_plus_cn_common_v6"
    emitted = sorted(set(model_labels) | set(rule_labels), key=PII_CORE_LABELS.index)
    payload = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "document_type": "comparator_source_closure",
        "system_id": system_id_for_track(track),
        "track": track,
        "source": _source_metadata(track),
        "scope": {
            "scope_id": "open24",
            "ordered_labels": list(PII_CORE_LABELS),
            "strict_metric_includes_unsupported_labels": True,
        },
        "prediction_semantics": semantics,
        "source_to_core24": mapping,
        "model_supported_core24": model_labels,
        "rule_supported_core24": rule_labels,
        "emittable_core24": emitted,
        "unsupported_core24": [label for label in PII_CORE_LABELS if label not in emitted],
        "threshold": THRESHOLD,
        "post_result_mapping_or_threshold_change_allowed": False,
    }
    result = _self_hashed(payload, field="document_sha256")
    _assert_no_local_paths(result)
    return result


def _source_file_hash(relative: str) -> str:
    _payload, identity = _read_regular(
        _repository_root() / relative,
        field="comparator implementation source",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    return identity.sha256


def _implementation_identity(track: ComparatorTrack) -> dict[str, Any]:
    sources = {
        "comparator_provenance": (
            "src/pii_zh/evaluation/open24_comparator_provenance.py"
        ),
        "generator_cli": str(GENERATOR_PATH),
        "prediction_io": "src/pii_zh/evaluation/io.py",
        "schema_parser": "src/pii_zh/data/schema.py",
        "stage_gate": "src/pii_zh/evaluation/release_eval_v2_stage_gate.py",
    }
    if track == "model_raw":
        sources.update(
            {
                "aiguard_decoder": "src/pii_zh/inference/aiguard.py",
                "qwen_recognizer": "src/pii_zh/presidio/qwen_recognizer.py",
                "token_chunker": "src/pii_zh/presidio/token_chunker.py",
            }
        )
    else:
        sources.update(
            {
                "bios_decoder": "src/pii_zh/inference/bios_token_classifier.py",
                "bio_recognizer": "src/pii_zh/presidio/bio_token_recognizer.py",
                "cn_common_adapter": "src/pii_zh/presidio/cn_common_recognizer.py",
                "native_presidio": "src/pii_zh/presidio/native_framework.py",
                "cn_common_v6": "src/pii_zh/rules/cn_common_v6.py",
            }
        )
    hashes = {name: _source_file_hash(relative) for name, relative in sorted(sources.items())}
    payload = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "document_type": "comparator_generator_implementation",
        "system_id": system_id_for_track(track),
        "factory_id": (
            "Open24AiguardSpanPredictor+QwenPiiRecognizer"
            if track == "model_raw"
            else "NativePresidioFrameworkRecognizer+CLUENER+CnCommonRulePackV6"
        ),
        "source_file_sha256": hashes,
        "implementation_sha256": canonical_hash(hashes),
        "repository_relative_names_only": True,
    }
    result = _self_hashed(payload, field="document_sha256")
    _assert_no_local_paths(result)
    return result


def _validate_runtime(runtime: RuntimeOptions, track: ComparatorTrack) -> dict[str, Any]:
    if (
        isinstance(runtime.document_batch_size, bool)
        or runtime.document_batch_size < 1
        or isinstance(runtime.micro_batch_size, bool)
        or runtime.micro_batch_size < 1
        or runtime.max_tokens != MAX_TOKENS
        or not math.isclose(runtime.stride_fraction, STRIDE_FRACTION, rel_tol=0.0, abs_tol=0.0)
    ):
        raise Open24ComparatorError("comparator runtime is outside the frozen protocol")
    if runtime.device == "cpu":
        device_class: DeviceClass = "cpu"
        visible_count = 0
        dtype = "float32"
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
            raise Open24ComparatorError(
                "CPU comparator requires CUDA_VISIBLE_DEVICES to be empty or -1"
            )
    elif runtime.device in {"cuda", "cuda:0"}:
        device_class = "cuda"
        values = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if len(values) != 1 or not values[0].isdigit():
            raise Open24ComparatorError(
                "CUDA comparator requires exactly one numeric CUDA_VISIBLE_DEVICES entry"
            )
        visible_count = 1
        dtype = "bfloat16"
    else:
        raise Open24ComparatorError("device must be cpu, cuda, or cuda:0")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise Open24ComparatorError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    payload = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "document_type": "comparator_runtime_configuration",
        "system_id": system_id_for_track(track),
        "document_batch_size": runtime.document_batch_size,
        "micro_batch_size": runtime.micro_batch_size,
        "max_tokens": runtime.max_tokens,
        "stride_fraction": runtime.stride_fraction,
        "threshold": THRESHOLD,
        "scope_id": "open24",
        "device_class": device_class,
        "dtype": dtype,
        "visible_cuda_device_count": visible_count,
        "local_files_only": True,
        "offline_environment": True,
        "remote_code_allowed": False,
    }
    return _self_hashed(payload, field="document_sha256")


def _distribution_version(name: str) -> str:
    try:
        result = version(name)
    except PackageNotFoundError:
        return "not-installed"
    if not result or len(result) > 128 or any(character.isspace() for character in result):
        raise Open24ComparatorError("runtime distribution version is not a safe token")
    return result


def _presidio_installed_source_identity() -> dict[str, Any]:
    try:
        installed = distribution("presidio-analyzer")
    except PackageNotFoundError as exc:
        raise Open24ComparatorError("native Presidio distribution is not installed") from exc
    files = installed.files
    if files is None:
        raise Open24ComparatorError("native Presidio distribution lacks a file inventory")
    hashes: dict[str, str] = {}
    for item in files:
        name = str(item).replace("\\", "/")
        if not name.startswith("presidio_analyzer/") or not name.endswith(".py"):
            continue
        path = Path(str(installed.locate_file(item)))
        if path.is_symlink():
            raise Open24ComparatorError("native Presidio source contains a symlink")
        _payload, identity = _read_regular(
            path,
            field="installed native Presidio source",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        )
        hashes[name] = identity.sha256
    if not hashes:
        raise Open24ComparatorError("native Presidio source inventory is empty")
    return {
        "distribution": "presidio-analyzer",
        "version": _distribution_version("presidio-analyzer"),
        "python_source_file_count": len(hashes),
        "python_source_file_sha256": dict(sorted(hashes.items())),
        "combined_sha256": canonical_hash(dict(sorted(hashes.items()))),
        "repository_relative_distribution_names_only": True,
    }


def _environment_lock(track: ComparatorTrack, runtime: Mapping[str, Any]) -> dict[str, Any]:
    packages = {
        name: _distribution_version(name)
        for name in ("pii-zh-qwen", "presidio-analyzer", "torch", "transformers")
    }
    payload = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "document_type": "comparator_environment_lock",
        "system_id": system_id_for_track(track),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": packages,
        "native_framework_source": (
            None if track == "model_raw" else _presidio_installed_source_identity()
        ),
        "device_class": runtime["device_class"],
        "visible_cuda_device_count": runtime["visible_cuda_device_count"],
        "hf_hub_offline": True,
        "transformers_offline": True,
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "credentials_or_environment_values_included": False,
    }
    result = _self_hashed(payload, field="document_sha256")
    _assert_no_local_paths(result)
    return result


class _Open24AiguardSpanPredictor:
    """AIguard B/I/E predictor restricted to explicitly mapped core24 labels."""

    protocol_id = "public_synthetic_chinese_pii_open24_v2"
    attention_mode = "checkpoint_native_causal"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        micro_batch_size: int,
    ) -> None:
        import torch

        from pii_zh.inference.aiguard import normalize_id2label

        if isinstance(micro_batch_size, bool) or micro_batch_size < 1:
            raise Open24ComparatorError("AIguard micro batch size must be positive")
        if not getattr(tokenizer, "is_fast", False):
            raise Open24ComparatorError("AIguard exact offsets require a fast tokenizer")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise Open24ComparatorError("AIguard tokenizer has no padding token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        raw_id2label = getattr(getattr(model, "config", None), "id2label", None)
        if not isinstance(raw_id2label, Mapping):
            raise Open24ComparatorError("AIguard config lacks id2label")
        self.id2label = normalize_id2label(raw_id2label)
        allowed_tags = {
            "O",
            *(
                f"{prefix}-{source}"
                for source in AIGUARD_SOURCE_TO_CORE24
                for prefix in ("B", "I", "E")
            ),
        }
        missing = allowed_tags - set(self.id2label.values())
        if missing:
            raise Open24ComparatorError("AIguard checkpoint lacks a mapped Open-24 tag")
        self.allowed_label_ids = tuple(
            label_id for label_id, label in self.id2label.items() if label in allowed_tags
        )
        self.model = model.eval().to(torch.device(device))
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.micro_batch_size = micro_batch_size

    def _predict_micro_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        import torch

        from pii_zh.inference.aiguard import decode_bie_ids

        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {
            name: value.to(self.device)
            for name, value in encoded.items()
            if name in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits.float()
        if logits.ndim != 3 or logits.shape[-1] != len(self.id2label):
            raise Open24ComparatorError("AIguard logits do not match id2label")
        allowed = torch.tensor(
            self.allowed_label_ids,
            device=logits.device,
            dtype=torch.long,
        )
        probabilities = torch.softmax(logits.index_select(-1, allowed), dim=-1)
        token_scores, restricted_ids = probabilities.max(dim=-1)
        label_ids = allowed[restricted_ids]
        batches: list[list[dict[str, Any]]] = []
        for row, text in enumerate(texts):
            row_offsets: list[tuple[int, int]] = []
            for pair in offsets[row].tolist():
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or isinstance(pair[0], bool)
                    or isinstance(pair[1], bool)
                ):
                    raise Open24ComparatorError("AIguard tokenizer returned invalid offsets")
                start, end = int(pair[0]), int(pair[1])
                if start < 0 or end < start or end > len(text):
                    raise Open24ComparatorError("AIguard tokenizer offset exceeds the window")
                row_offsets.append((start, end))
            decoded = decode_bie_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                token_scores=token_scores[row].tolist(),
            )
            batches.append(
                [
                    {
                        "entity_type": AIGUARD_SOURCE_TO_CORE24[span.entity_type],
                        "start": span.start,
                        "end": span.end,
                        "score": span.score,
                        "offset_mode": "relative",
                        "metadata": {
                            "boundary_refined": False,
                            "protocol_id": self.protocol_id,
                        },
                    }
                    for span in decoded
                ]
            )
        return batches

    def predict_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if isinstance(texts, (str, bytes)) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        results: list[list[dict[str, Any]]] = [[] for _ in values]
        nonempty = [(index, text) for index, text in enumerate(values) if text]
        for start in range(0, len(nonempty), self.micro_batch_size):
            batch = nonempty[start : start + self.micro_batch_size]
            predicted = self._predict_micro_batch([text for _, text in batch])
            for (index, _text), spans in zip(batch, predicted, strict=True):
                results[index] = spans
        return results


def _build_aiguard_detector(path: Path, runtime: RuntimeOptions) -> BatchDetector:
    import torch
    from transformers import AutoTokenizer, Qwen3ForTokenClassification

    from pii_zh.presidio import QwenPiiRecognizer

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "weights_only": True,
        "low_cpu_mem_usage": True,
    }
    if runtime.device != "cpu":
        kwargs["dtype"] = torch.bfloat16
    model = Qwen3ForTokenClassification.from_pretrained(path, **kwargs)
    predictor = _Open24AiguardSpanPredictor(
        model,
        tokenizer,
        device=runtime.device,
        micro_batch_size=runtime.micro_batch_size,
    )
    supported = tuple(sorted(set(AIGUARD_SOURCE_TO_CORE24.values()), key=PII_CORE_LABELS.index))
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=tokenizer,
        label_mapping={label: label for label in supported},
        ignored_labels=(),
        default_threshold=THRESHOLD,
        max_tokens=runtime.max_tokens,
        stride_fraction=runtime.stride_fraction,
        model_version=f"{AIGUARD_MODEL_ID}@{AIGUARD_REVISION}",
        attention_mode=predictor.attention_mode,
        deduplication_policy="exact_only_v1",
        name="AIguardOpen24Comparator",
    )

    def detect(texts: Sequence[str]) -> Sequence[Sequence[Any]]:
        return cast(Sequence[Sequence[Any]], recognizer.analyze_batch(texts, entities=supported))

    return detect


def _build_framework_detector(path: Path, runtime: RuntimeOptions) -> BatchDetector:
    import torch

    from pii_zh.inference.bios_token_classifier import load_closed_bios_token_classifier
    from pii_zh.presidio.bio_token_recognizer import BioTokenClassifierRecognizer
    from pii_zh.presidio.cn_common_recognizer import CnCommonRecognizer
    from pii_zh.presidio.native_framework import (
        NativePresidioFrameworkRecognizer,
        presidio_analyzer_version,
    )
    from pii_zh.rules.cn_common_v6 import CnCommonRulePackV6

    actual_presidio = presidio_analyzer_version()
    if actual_presidio != "2.2.363":
        raise Open24ComparatorError("native Presidio version must be exactly 2.2.363")
    mapping = load_presidio_mapping()
    source_to_presidio = {
        source: mapping.model_to_presidio[target]
        for source, target in CLUENER_SOURCE_TO_CORE24.items()
    }
    dtype = torch.bfloat16 if runtime.device != "cpu" else None
    predictor = load_closed_bios_token_classifier(
        path,
        source_to_target=source_to_presidio,
        protocol_id="public_synthetic_chinese_pii_open24_v2",
        device=runtime.device,
        dtype=dtype,
        micro_batch_size=runtime.micro_batch_size,
    )
    ner = BioTokenClassifierRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        supported_entities=sorted(source_to_presidio.values()),
        default_threshold=THRESHOLD,
        max_tokens=runtime.max_tokens,
        stride_fraction=runtime.stride_fraction,
        model_version=f"{CLUENER_MODEL_ID}@{CLUENER_REVISION}",
        source_name="native_presidio:cluener_open24",
        deduplication_policy="exact_only_v1",
        name="CluenerOpen24NativeFrameworkNerRecognizer",
    )
    framework = NativePresidioFrameworkRecognizer(
        ner_recognizer=ner,
        rule_recognizer=CnCommonRecognizer(
            rule_pack=CnCommonRulePackV6(),
            name="CnCommonV6Open24ComparatorRecognizer",
            version="6.0.0",
        ),
    )
    identity = framework.runtime_identity()
    if (
        identity.get("framework") != "presidio-analyzer"
        or identity.get("framework_version") != "2.2.363"
        or type(framework.engine).__name__ != "AnalyzerEngine"
        or not type(framework.engine).__module__.startswith("presidio_analyzer")
    ):
        raise Open24ComparatorError("full-system comparator is not native Presidio")
    requested = [mapping.model_to_presidio[label] for label in PII_CORE_LABELS]

    def detect(texts: Sequence[str]) -> Sequence[Sequence[Any]]:
        return [framework.analyze(text, entities=requested) for text in texts]

    return detect


def build_comparator_detector(
    track: ComparatorTrack,
    path: Path,
    runtime: RuntimeOptions,
) -> BatchDetector:
    """Load the exact offline comparator declared for ``track``."""

    if track == "model_raw":
        return _build_aiguard_detector(path, runtime)
    if track == "full_system":
        return _build_framework_detector(path, runtime)
    raise Open24ComparatorError("unsupported comparator track")


def default_hooks() -> ComparatorHooks:
    return ComparatorHooks(
        stage_authorizer=authorize_internal_preopen,
        artifact_inspector=inspect_comparator_artifact,
        detector_factory=build_comparator_detector,
    )


def _raw_span_value(item: Any, name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _normalize_detected_spans(
    values: Sequence[Any],
    *,
    track: ComparatorTrack,
    text_length: int,
) -> tuple[Span, ...]:
    reverse_presidio = {
        value: key for key, value in load_presidio_mapping().model_to_presidio.items()
    }
    spans: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for item in values:
        start = _raw_span_value(item, "start")
        end = _raw_span_value(item, "end")
        entity = _raw_span_value(item, "entity_type")
        score = _raw_span_value(item, "score")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise Open24ComparatorError("comparator emitted an invalid span")
        if not isinstance(entity, str):
            raise Open24ComparatorError("comparator emitted an invalid entity label")
        label = entity if track == "model_raw" else reverse_presidio.get(entity)
        if label not in PII_CORE_LABELS:
            raise Open24ComparatorError("comparator emitted a label outside Open-24")
        span = Span(start=start, end=end, label=label, score=float(score))
        try:
            span.validate(text_length=text_length)
        except EvaluationDataError as exc:
            raise Open24ComparatorError("comparator emitted an out-of-range span") from exc
        if span.metric_key() in seen:
            raise Open24ComparatorError("comparator emitted a duplicate metric span")
        seen.add(span.metric_key())
        spans.append(span)
    return tuple(
        sorted(
            spans,
            key=lambda span: (
                span.start,
                span.end,
                span.label,
                -float(span.score or 0),
            ),
        )
    )


def _gpu_prepare(runtime: Mapping[str, Any]) -> None:
    if runtime["device_class"] != "cuda":
        return
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError) as exc:
        raise Open24ComparatorError("CUDA measurement initialization failed") from exc


def _gpu_synchronize(runtime: Mapping[str, Any]) -> None:
    if runtime["device_class"] != "cuda":
        return
    try:
        import torch

        torch.cuda.synchronize()
    except (ImportError, RuntimeError) as exc:
        raise Open24ComparatorError("CUDA measurement synchronization failed") from exc


def _gpu_peak_mb(runtime: Mapping[str, Any]) -> float:
    if runtime["device_class"] != "cuda":
        return 0.0
    try:
        import torch

        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except (ImportError, RuntimeError) as exc:
        raise Open24ComparatorError("CUDA peak-memory measurement failed") from exc


def _rss_mb(value: int) -> float:
    # Linux reports KiB and macOS reports bytes.  The formal environment is
    # Linux, but retaining the branch keeps fixture replay honest elsewhere.
    divisor = 1024.0 if platform.system() == "Linux" else 1024.0 * 1024.0
    return float(value) / divisor


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise Open24ComparatorError("latency measurement contains no batches")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _measurement_document(
    *,
    track: ComparatorTrack,
    runtime: Mapping[str, Any],
    batch_latencies_ms: Sequence[float],
    document_count: int,
    character_count: int,
    wall_time_ns: int,
    peak_rss_mb: float,
    peak_gpu_memory_mb: float,
    implementation_sha256: str,
) -> dict[str, Any]:
    if (
        document_count < 1
        or character_count < 0
        or wall_time_ns < 1
        or len(batch_latencies_ms) < 1
        or any(not math.isfinite(value) or value < 0 for value in batch_latencies_ms)
        or not math.isfinite(peak_rss_mb)
        or peak_rss_mb < 0
        or not math.isfinite(peak_gpu_memory_mb)
        or peak_gpu_memory_mb < 0
    ):
        raise Open24ComparatorError("execution measurement is invalid")
    seconds = wall_time_ns / 1_000_000_000.0
    payload = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "receipt_type": "open24_comparator_execution_measurement",
        "system_id": system_id_for_track(track),
        "track": track,
        "sample_count": document_count,
        "character_count": character_count,
        "batch_count": len(batch_latencies_ms),
        "configured_document_batch_size": runtime["document_batch_size"],
        "latency_semantics": "end_to_end_batch_wall_time",
        "latency_ms": {
            "p50": _percentile(batch_latencies_ms, 0.50),
            "p95": _percentile(batch_latencies_ms, 0.95),
            "p99": _percentile(batch_latencies_ms, 0.99),
            "maximum": max(batch_latencies_ms),
        },
        "throughput": {
            "documents_per_second": document_count / seconds,
            "characters_per_second": character_count / seconds,
            "wall_time_seconds": seconds,
        },
        "resources": {
            "peak_process_rss_mb": peak_rss_mb,
            "peak_process_rss_semantics": "getrusage_process_lifetime_high_water_mark",
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "peak_gpu_memory_semantics": (
                "torch_cuda_max_memory_allocated_after_reset"
                if runtime["device_class"] == "cuda"
                else "cpu_run_zero_by_definition"
            ),
        },
        "measurement_sources": {
            "latency_and_throughput": "time.perf_counter_ns",
            "peak_rss": "resource.getrusage_RUSAGE_SELF_ru_maxrss",
            "peak_gpu_memory": (
                "torch.cuda.max_memory_allocated"
                if runtime["device_class"] == "cuda"
                else "not_applicable"
            ),
            "implementation_sha256": implementation_sha256,
            "caller_supplied_measurement_values": False,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "aggregate_only": True,
        },
    }
    result = _self_hashed(payload, field="receipt_sha256")
    _assert_no_local_paths(result)
    _validate_schema(result, MEASUREMENT_SCHEMA_PATH, field="execution measurement")
    return result


def _generate_predictions(
    *,
    snapshot: InputSnapshot,
    destination: Path,
    track: ComparatorTrack,
    detector: BatchDetector,
    runtime: Mapping[str, Any],
    implementation_sha256: str,
) -> tuple[FileIdentity, dict[str, Any]]:
    seen: set[str] = set()
    ordered_ids: list[str] = []
    batch_records: list[DocumentRecord] = []
    batch_latencies: list[float] = []
    count = 0
    characters = 0
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _gpu_prepare(runtime)
    overall_start = time.perf_counter_ns()

    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:

            def flush() -> None:
                nonlocal count, characters
                if not batch_records:
                    return
                texts = [record.text for record in batch_records]
                _gpu_synchronize(runtime)
                started = time.perf_counter_ns()
                try:
                    raw_batches = list(detector(texts))
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise Open24ComparatorError("comparator inference failed") from exc
                _gpu_synchronize(runtime)
                elapsed = time.perf_counter_ns() - started
                if len(raw_batches) != len(batch_records):
                    raise Open24ComparatorError(
                        "comparator must emit one span sequence per document"
                    )
                predictions: list[PredictionRecord] = []
                for record, raw_spans in zip(batch_records, raw_batches, strict=True):
                    if isinstance(raw_spans, (str, bytes)) or not isinstance(
                        raw_spans, Sequence
                    ):
                        raise Open24ComparatorError(
                            "comparator document output must be a span sequence"
                        )
                    spans = _normalize_detected_spans(
                        raw_spans,
                        track=track,
                        text_length=len(record.text),
                    )
                    prediction = PredictionRecord(doc_id=record.doc_id, spans=spans)
                    prediction.validate(text_length=len(record.text))
                    predictions.append(prediction)
                    ordered_ids.append(record.doc_id)
                    characters += len(record.text)
                count += write_prediction_jsonl(predictions, handle)
                batch_latencies.append(elapsed / 1_000_000.0)
                batch_records.clear()

            try:
                decoded = snapshot.payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise Open24ComparatorError("authorized input is not UTF-8") from exc
            try:
                for record in iter_jsonl(io.StringIO(decoded)):
                    if record.doc_id in seen:
                        raise Open24ComparatorError(
                            "authorized input contains a duplicate document ID"
                        )
                    seen.add(record.doc_id)
                    batch_records.append(record)
                    if len(batch_records) >= runtime["document_batch_size"]:
                        flush()
                flush()
            except Open24ComparatorError:
                raise
            except (TypeError, UnicodeError, ValueError) as exc:
                raise Open24ComparatorError("authorized input violates canonical JSONL") from exc
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    wall_time_ns = time.perf_counter_ns() - overall_start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if count != snapshot.identity.nonempty_lines or len(ordered_ids) != count:
        destination.unlink(missing_ok=True)
        raise Open24ComparatorError("prediction count differs from the input snapshot")
    payload, identity = _read_regular(
        destination,
        field="generated comparator predictions",
        maximum_bytes=_MAX_JSONL_BYTES,
        count_lines=True,
    )
    try:
        records = load_prediction_jsonl(io.StringIO(payload.decode("utf-8")))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise Open24ComparatorError("generated predictions violate the safe JSONL schema") from exc
    if [record.doc_id for record in records] != ordered_ids or identity.nonempty_lines != count:
        destination.unlink(missing_ok=True)
        raise Open24ComparatorError("prediction ordering/count differs from the input")
    os.chmod(destination, 0o444)
    measurement = _measurement_document(
        track=track,
        runtime=runtime,
        batch_latencies_ms=batch_latencies,
        document_count=count,
        character_count=characters,
        wall_time_ns=wall_time_ns,
        peak_rss_mb=_rss_mb(max(rss_before, rss_after)),
        peak_gpu_memory_mb=_gpu_peak_mb(runtime),
        implementation_sha256=implementation_sha256,
    )
    return identity, measurement


def _write_new(path: Path, payload: bytes, *, mode: int = 0o444) -> FileIdentity:
    descriptor = os.open(
        path,
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
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return FileIdentity(hashlib.sha256(payload).hexdigest(), len(payload))


def _write_json(path: Path, document: Mapping[str, Any]) -> FileIdentity:
    return _write_new(path, _pretty_bytes(document))


def _binding(identity: FileIdentity) -> dict[str, Any]:
    return {"file_sha256": identity.sha256, "size_bytes": identity.size_bytes}


def _validate_schema(document: Mapping[str, Any], schema_path: Path, *, field: str) -> None:
    schema, _identity = _read_json(_repository_root() / schema_path, field=f"{field} schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise Open24ComparatorError(f"{field} violates its strict schema") from exc


def _generation_receipt(
    *,
    track: ComparatorTrack,
    stage: StageEvidence,
    snapshot: InputSnapshot,
    prediction: FileIdentity,
    support_documents: Mapping[str, tuple[Mapping[str, Any], FileIdentity]],
    measurement: tuple[Mapping[str, Any], FileIdentity],
    fixture: bool,
) -> dict[str, Any]:
    support = {
        name: {
            **_binding(identity),
            "document_sha256": document["document_sha256"],
        }
        for name, (document, identity) in sorted(support_documents.items())
    }
    measurement_document, measurement_file = measurement
    payload = {
        "schema_version": GENERATION_RECEIPT_SCHEMA_VERSION,
        "receipt_type": "open24_comparator_generation",
        "system_id": system_id_for_track(track),
        "track": track,
        "scope": {
            "scope_id": "open24",
            "ordered_labels": list(PII_CORE_LABELS),
            "threshold": THRESHOLD,
        },
        "stage_gate": dict(stage.bindings),
        "input": {
            "file_sha256": snapshot.identity.sha256,
            "size_bytes": snapshot.identity.size_bytes,
            "document_count": snapshot.identity.nonempty_lines,
            "single_descriptor_snapshot": True,
            "symlink_followed": False,
            "changed_while_read": False,
        },
        "output": {
            "file_sha256": prediction.sha256,
            "size_bytes": prediction.size_bytes,
            "document_count": prediction.nonempty_lines,
            "raw_text_free_prediction_schema": True,
            "actual_generated_bytes_bound": True,
        },
        "support_documents": support,
        "measurement": {
            **_binding(measurement_file),
            "receipt_sha256": measurement_document["receipt_sha256"],
            "caller_supplied_measurement_values": False,
            "execution_replay_required": True,
        },
        "execution": {
            "exact_gold_document_set": True,
            "prediction_file_generated_by_bound_system": True,
            "hand_edited": False,
            "fixture": fixture,
            "full_execution_replay_required": True,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "aggregate_only": True,
        },
    }
    result = _self_hashed(payload, field="receipt_sha256")
    _assert_no_local_paths(result)
    _validate_schema(result, GENERATION_SCHEMA_PATH, field="generation receipt")
    return result


def _prediction_manifest(
    *,
    track: ComparatorTrack,
    stage: StageEvidence,
    prediction: FileIdentity,
    support_identities: Mapping[str, FileIdentity],
    generation_identity: FileIdentity,
    fixture: bool,
) -> dict[str, Any]:
    system_bindings = {
        name: _binding(support_identities[name]) for name in _SUPPORT_FILENAMES
    }
    system_bindings["generation_receipt"] = _binding(generation_identity)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "role": "comparator",
        "track": track,
        "system_id": system_id_for_track(track),
        "prediction_semantics": "strict_exact_character_spans_before_quality_aggregation",
        "source": _source_metadata(track),
        "dataset": dict(stage.dataset),
        "prediction": {
            "file_sha256": prediction.sha256,
            "size_bytes": prediction.size_bytes,
            "document_count": prediction.nonempty_lines,
        },
        "system_bindings": system_bindings,
        "execution": {
            "exact_gold_document_set": True,
            "prediction_file_generated_by_bound_system": True,
            "hand_edited": False,
            "fixture": fixture,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "aggregate_only": True,
        },
    }
    result = _self_hashed(payload, field="manifest_sha256")
    _assert_no_local_paths(result)
    _validate_schema(result, MANIFEST_SCHEMA_PATH, field="prediction provenance manifest")
    return result


def _validate_support_document(
    document: Mapping[str, Any],
    *,
    track: ComparatorTrack,
    document_type: str,
) -> str:
    if (
        document.get("schema_version") != SUPPORT_SCHEMA_VERSION
        or document.get("document_type") != document_type
        or document.get("system_id") != system_id_for_track(track)
    ):
        raise Open24ComparatorError("comparator support document identity changed")
    self_hash = _verify_self_hash(
        document,
        field="comparator support document",
        key="document_sha256",
    )
    _assert_no_local_paths(document)
    return self_hash


def _validate_measurement(
    document: Mapping[str, Any],
    *,
    track: ComparatorTrack,
    implementation_sha256: str,
    expected_documents: int,
) -> str:
    _validate_schema(document, MEASUREMENT_SCHEMA_PATH, field="execution measurement")
    _exact_keys(
        document,
        {
            "schema_version",
            "receipt_type",
            "system_id",
            "track",
            "sample_count",
            "character_count",
            "batch_count",
            "configured_document_batch_size",
            "latency_semantics",
            "latency_ms",
            "throughput",
            "resources",
            "measurement_sources",
            "privacy",
            "receipt_sha256",
        },
        field="execution measurement",
    )
    latency = _mapping(document.get("latency_ms"), field="measurement latency")
    throughput = _mapping(document.get("throughput"), field="measurement throughput")
    resources = _mapping(document.get("resources"), field="measurement resources")
    sources = _mapping(document.get("measurement_sources"), field="measurement sources")
    privacy = _mapping(document.get("privacy"), field="measurement privacy")
    numeric = [
        *(latency.get(name) for name in ("p50", "p95", "p99", "maximum")),
        *(throughput.get(name) for name in (
            "documents_per_second",
            "characters_per_second",
            "wall_time_seconds",
        )),
        resources.get("peak_process_rss_mb"),
        resources.get("peak_gpu_memory_mb"),
    ]
    if (
        document.get("schema_version") != MEASUREMENT_SCHEMA_VERSION
        or document.get("receipt_type") != "open24_comparator_execution_measurement"
        or document.get("system_id") != system_id_for_track(track)
        or document.get("track") != track
        or document.get("sample_count") != expected_documents
        or isinstance(document.get("character_count"), bool)
        or not isinstance(document.get("character_count"), int)
        or int(document["character_count"]) < 0
        or isinstance(document.get("batch_count"), bool)
        or not isinstance(document.get("batch_count"), int)
        or int(document["batch_count"]) < 1
        or document.get("latency_semantics") != "end_to_end_batch_wall_time"
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in numeric
        )
        or sources.get("implementation_sha256") != implementation_sha256
        or sources.get("caller_supplied_measurement_values") is not False
        or any(
            privacy.get(name) is not False
            for name in (
                "contains_paths",
                "contains_raw_text",
                "contains_entity_values",
                "contains_document_ids",
            )
        )
        or privacy.get("aggregate_only") is not True
    ):
        raise Open24ComparatorError("execution measurement contract is invalid")
    _assert_no_local_paths(document)
    return _verify_self_hash(
        document,
        field="execution measurement",
        key="receipt_sha256",
    )


def _normalized_destination(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.name or candidate.name in {".", ".."}:
        raise Open24ComparatorError("output directory name is invalid")
    try:
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Open24ComparatorError("output parent must already exist") from exc
    if not parent.is_dir() or parent.is_symlink():
        raise Open24ComparatorError("output parent must be a real directory")
    destination = parent / candidate.name
    if os.path.lexists(destination):
        raise Open24ComparatorError("refusing to overwrite comparator output directory")
    return destination


def _publish_run_directory(staging: Path, destination: Path) -> None:
    names = sorted(item.name for item in staging.iterdir())
    expected = {
        "predictions.jsonl",
        "execution_measurement.json",
        "generation_receipt.json",
        "prediction_provenance.json",
        *_SUPPORT_FILENAMES.values(),
    }
    if set(names) != expected:
        raise Open24ComparatorError("staging bundle inventory is incomplete")
    created = False
    linked: list[Path] = []
    try:
        destination.mkdir(mode=0o700)
        created = True
        for name in names:
            source = staging / name
            target = destination / name
            os.link(source, target, follow_symlinks=False)
            linked.append(target)
        directory_fd = os.open(
            destination,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.chmod(destination, 0o555)
    except OSError as exc:
        for target in reversed(linked):
            target.unlink(missing_ok=True)
        if created:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise Open24ComparatorError("cannot publish comparator run directory") from exc


def generate_comparator_run(
    inputs: ComparatorRunInputs,
    *,
    hooks: ComparatorHooks | None = None,
) -> dict[str, Any]:
    """Generate and no-clobber publish one complete comparator bundle."""

    active_hooks = default_hooks() if hooks is None else hooks
    fixture = hooks is not None
    destination = _normalized_destination(inputs.output_dir)
    runtime_document = _validate_runtime(inputs.runtime, inputs.track)
    model_snapshot = (
        _model_stat_snapshot(inputs.track, inputs.source_model) if hooks is None else None
    )
    artifact_document = active_hooks.artifact_inspector(inputs.track, inputs.source_model)
    _validate_support_document(
        artifact_document,
        track=inputs.track,
        document_type="comparator_artifact_identity",
    )
    source_document = _source_closure(inputs.track)
    implementation_document = _implementation_identity(inputs.track)
    environment_document = _environment_lock(inputs.track, runtime_document)
    detector = active_hooks.detector_factory(
        inputs.track,
        inputs.source_model,
        inputs.runtime,
    )
    if model_snapshot is not None and _model_stat_snapshot(
        inputs.track, inputs.source_model
    ) != model_snapshot:
        raise Open24ComparatorError(
            "comparator model artifacts changed between verification and load"
        )

    # This is the hard pre-open boundary.  Neither source/artifact validation
    # nor model loading above opens the target JSONL.
    stage = active_hooks.stage_authorizer(inputs.stage_spec, inputs.input_path, inputs.track)
    snapshot = snapshot_internal_input(inputs.input_path, stage)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    os.chmod(staging, 0o700)
    try:
        support_documents: dict[str, Mapping[str, Any]] = {
            "source_closure_manifest": source_document,
            "artifact_or_configuration": artifact_document,
            "generator_implementation": implementation_document,
            "runtime_configuration": runtime_document,
            "environment_lock": environment_document,
        }
        support_identities: dict[str, FileIdentity] = {}
        for name, filename in _SUPPORT_FILENAMES.items():
            support_identities[name] = _write_json(
                staging / filename,
                support_documents[name],
            )
        prediction_identity, measurement_document = _generate_predictions(
            snapshot=snapshot,
            destination=staging / "predictions.jsonl",
            track=inputs.track,
            detector=detector,
            runtime=runtime_document,
            implementation_sha256=str(implementation_document["implementation_sha256"]),
        )
        measurement_identity = _write_json(
            staging / "execution_measurement.json",
            measurement_document,
        )
        generation_document = _generation_receipt(
            track=inputs.track,
            stage=stage,
            snapshot=snapshot,
            prediction=prediction_identity,
            support_documents={
                name: (support_documents[name], support_identities[name])
                for name in support_documents
            },
            measurement=(measurement_document, measurement_identity),
            fixture=fixture,
        )
        generation_identity = _write_json(
            staging / "generation_receipt.json",
            generation_document,
        )
        manifest = _prediction_manifest(
            track=inputs.track,
            stage=stage,
            prediction=prediction_identity,
            support_identities=support_identities,
            generation_identity=generation_identity,
            fixture=fixture,
        )
        manifest_identity = _write_json(
            staging / "prediction_provenance.json",
            manifest,
        )
        if model_snapshot is not None and _model_stat_snapshot(
            inputs.track, inputs.source_model
        ) != model_snapshot:
            raise Open24ComparatorError("comparator model artifacts changed during generation")
        _publish_run_directory(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "status": (
            "PASS_FIXTURE_GENERATED_REPLAY_REQUIRED"
            if fixture
            else "PASS_GENERATED_REPLAY_REQUIRED"
        ),
        "track": inputs.track,
        "system_id": system_id_for_track(inputs.track),
        "document_count": prediction_identity.nonempty_lines,
        "predictions_sha256": prediction_identity.sha256,
        "prediction_manifest_file_sha256": manifest_identity.sha256,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "generation_receipt_sha256": generation_document["receipt_sha256"],
        "measurement_receipt_sha256": measurement_document["receipt_sha256"],
        "full_execution_replay_required": True,
    }


def _bundle_inventory(path: Path) -> dict[str, tuple[bytes, FileIdentity]]:
    try:
        root = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Open24ComparatorError("comparator bundle is unavailable") from exc
    if path.expanduser().is_symlink() or not root.is_dir():
        raise Open24ComparatorError("comparator bundle must be a non-symlink directory")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise Open24ComparatorError("comparator bundle directory mode changed")
    expected = {
        "predictions.jsonl",
        "execution_measurement.json",
        "generation_receipt.json",
        "prediction_provenance.json",
        *_SUPPORT_FILENAMES.values(),
    }
    observed = {item.name for item in root.iterdir()}
    if observed != expected:
        raise Open24ComparatorError("comparator bundle has an unexpected inventory")
    result: dict[str, tuple[bytes, FileIdentity]] = {}
    for name in sorted(expected):
        artifact = root / name
        if artifact.is_symlink() or stat.S_IMODE(artifact.stat().st_mode) != 0o444:
            raise Open24ComparatorError("comparator bundle artifact mode changed")
        maximum = _MAX_JSONL_BYTES if name == "predictions.jsonl" else _MAX_JSON_BYTES
        result[name] = _read_regular(
            artifact,
            field="comparator bundle artifact",
            maximum_bytes=maximum,
            count_lines=name == "predictions.jsonl",
        )
    return result


def validate_comparator_run(
    inputs: ComparatorRunInputs,
    *,
    hooks: ComparatorHooks | None = None,
) -> dict[str, Any]:
    """Strictly replay provenance from actual files without rerunning inference."""

    active_hooks = default_hooks() if hooks is None else hooks
    fixture = hooks is not None
    inventory = _bundle_inventory(inputs.output_dir)
    runtime_expected = _validate_runtime(inputs.runtime, inputs.track)
    artifact_expected = active_hooks.artifact_inspector(inputs.track, inputs.source_model)
    source_expected = _source_closure(inputs.track)
    implementation_expected = _implementation_identity(inputs.track)
    environment_expected = _environment_lock(inputs.track, runtime_expected)
    expected_support: dict[str, Mapping[str, Any]] = {
        "source_closure_manifest": source_expected,
        "artifact_or_configuration": artifact_expected,
        "generator_implementation": implementation_expected,
        "runtime_configuration": runtime_expected,
        "environment_lock": environment_expected,
    }
    support_identities: dict[str, FileIdentity] = {}
    for name, filename in _SUPPORT_FILENAMES.items():
        document = _strict_json(inventory[filename][0], field="comparator support document")
        if document != expected_support[name]:
            raise Open24ComparatorError("comparator support document strict replay differs")
        support_identities[name] = inventory[filename][1]

    stage = active_hooks.stage_authorizer(inputs.stage_spec, inputs.input_path, inputs.track)
    snapshot = snapshot_internal_input(inputs.input_path, stage)
    prediction_payload, prediction_identity = inventory["predictions.jsonl"]
    try:
        predictions = load_prediction_jsonl(
            io.StringIO(prediction_payload.decode("utf-8"))
        )
        source_ids = [
            record.doc_id
            for record in iter_jsonl(io.StringIO(snapshot.payload.decode("utf-8")))
        ]
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise Open24ComparatorError("bundle prediction/input replay failed") from exc
    if (
        [record.doc_id for record in predictions] != source_ids
        or prediction_identity.nonempty_lines != snapshot.identity.nonempty_lines
    ):
        raise Open24ComparatorError("bundle predictions do not cover the exact input order")

    measurement_document = _strict_json(
        inventory["execution_measurement.json"][0],
        field="execution measurement",
    )
    measurement_self = _validate_measurement(
        measurement_document,
        track=inputs.track,
        implementation_sha256=str(implementation_expected["implementation_sha256"]),
        expected_documents=int(snapshot.identity.nonempty_lines or 0),
    )
    generation_observed = _strict_json(
        inventory["generation_receipt.json"][0],
        field="generation receipt",
    )
    generation_expected = _generation_receipt(
        track=inputs.track,
        stage=stage,
        snapshot=snapshot,
        prediction=prediction_identity,
        support_documents={
            name: (expected_support[name], support_identities[name])
            for name in expected_support
        },
        measurement=(measurement_document, inventory["execution_measurement.json"][1]),
        fixture=fixture,
    )
    if generation_observed != generation_expected:
        raise Open24ComparatorError("generation receipt strict replay differs")
    manifest_observed = _strict_json(
        inventory["prediction_provenance.json"][0],
        field="prediction provenance manifest",
    )
    manifest_expected = _prediction_manifest(
        track=inputs.track,
        stage=stage,
        prediction=prediction_identity,
        support_identities=support_identities,
        generation_identity=inventory["generation_receipt.json"][1],
        fixture=fixture,
    )
    if manifest_observed != manifest_expected:
        raise Open24ComparatorError("prediction provenance strict replay differs")
    return {
        "status": (
            "PASS_FIXTURE_PROVENANCE_REPLAY_EXECUTION_REPLAY_REQUIRED"
            if fixture
            else "PASS_PROVENANCE_REPLAY_EXECUTION_REPLAY_REQUIRED"
        ),
        "track": inputs.track,
        "system_id": system_id_for_track(inputs.track),
        "document_count": prediction_identity.nonempty_lines,
        "predictions_sha256": prediction_identity.sha256,
        "prediction_manifest_file_sha256": inventory["prediction_provenance.json"][1].sha256,
        "prediction_manifest_sha256": manifest_expected["manifest_sha256"],
        "generation_receipt_sha256": generation_expected["receipt_sha256"],
        "measurement_receipt_sha256": measurement_self,
        "prediction_origin_execution_replayed": False,
        "full_execution_replay_required": True,
    }


def replay_comparator_execution(
    inputs: ComparatorRunInputs,
    *,
    hooks: ComparatorHooks | None = None,
) -> dict[str, Any]:
    """Rerun the comparator and require byte-identical prediction output."""

    fixture = hooks is not None
    observed = validate_comparator_run(inputs, hooks=hooks)
    parent = inputs.output_dir.expanduser().resolve(strict=True).parent
    scratch = Path(tempfile.mkdtemp(prefix=".open24-comparator-replay.", dir=parent))
    replay_dir = scratch / "replayed"
    replay_inputs = ComparatorRunInputs(
        track=inputs.track,
        source_model=inputs.source_model,
        input_path=inputs.input_path,
        output_dir=replay_dir,
        stage_spec=inputs.stage_spec,
        runtime=inputs.runtime,
    )
    try:
        generated = generate_comparator_run(replay_inputs, hooks=hooks)
        validated = validate_comparator_run(replay_inputs, hooks=hooks)
        observed_predictions, observed_identity = _read_regular(
            inputs.output_dir / "predictions.jsonl",
            field="observed comparator predictions",
            maximum_bytes=_MAX_JSONL_BYTES,
            count_lines=True,
        )
        replayed_predictions, replayed_identity = _read_regular(
            replay_dir / "predictions.jsonl",
            field="replayed comparator predictions",
            maximum_bytes=_MAX_JSONL_BYTES,
            count_lines=True,
        )
        if observed_predictions != replayed_predictions or observed_identity != replayed_identity:
            raise Open24ComparatorError(
                "full comparator execution replay produced different prediction bytes"
            )
        observed_measurement, _ = _read_json(
            inputs.output_dir / "execution_measurement.json",
            field="observed execution measurement",
        )
        replayed_measurement, _ = _read_json(
            replay_dir / "execution_measurement.json",
            field="replayed execution measurement",
        )
        observed_generator, observed_generator_file = _read_json(
            inputs.output_dir / "generator_implementation.json",
            field="observed generator implementation",
        )
        replayed_generator, replayed_generator_file = _read_json(
            replay_dir / "generator_implementation.json",
            field="replayed generator implementation",
        )
        observed_generation, _ = _read_json(
            inputs.output_dir / "generation_receipt.json",
            field="observed generation receipt",
        )
        replayed_generation, _ = _read_json(
            replay_dir / "generation_receipt.json",
            field="replayed generation receipt",
        )
        if observed_generator != replayed_generator:
            raise Open24ComparatorError("generator implementation changed during replay")
        if (
            observed_generation.get("stage_gate") != replayed_generation.get("stage_gate")
            or observed_generation.get("input") != replayed_generation.get("input")
            or observed_generation.get("output") != replayed_generation.get("output")
        ):
            raise Open24ComparatorError("generation identity changed during execution replay")
        receipt = {
            "schema_version": EXECUTION_REPLAY_SCHEMA_VERSION,
            "receipt_type": "open24_comparator_full_execution_replay",
            "status": "PASS",
            "track": inputs.track,
            "system_id": system_id_for_track(inputs.track),
            "prediction": {
                "observed_file_sha256": observed_identity.sha256,
                "replayed_file_sha256": replayed_identity.sha256,
                "byte_identical": True,
                "document_count": observed_identity.nonempty_lines,
            },
            "provenance": {
                "observed_manifest_file_sha256": observed[
                    "prediction_manifest_file_sha256"
                ],
                "observed_manifest_sha256": observed["prediction_manifest_sha256"],
                "replayed_manifest_file_sha256": validated[
                    "prediction_manifest_file_sha256"
                ],
                "replayed_manifest_sha256": validated["prediction_manifest_sha256"],
                "stage_and_support_strict_replay_passed": True,
            },
            "generation": {
                "observed_receipt_sha256": observed_generation["receipt_sha256"],
                "replayed_receipt_sha256": replayed_generation["receipt_sha256"],
                "stage_gate_equal": True,
                "input_identity_equal": True,
                "output_identity_equal": True,
            },
            "implementation": {
                "observed_file_sha256": observed_generator_file.sha256,
                "replayed_file_sha256": replayed_generator_file.sha256,
                "document_sha256": observed_generator["document_sha256"],
                "implementation_sha256": observed_generator["implementation_sha256"],
                "byte_identical": True,
            },
            "measurement": {
                "observed_receipt_sha256": observed_measurement["receipt_sha256"],
                "replayed_receipt_sha256": replayed_measurement["receipt_sha256"],
                "fresh_measurement_collected": True,
                "numeric_equality_required": False,
                "harness_replayed": True,
            },
            "execution": {
                "inference_rerun": True,
                "generated_status": generated["status"],
                "fixture": fixture,
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
                "aggregate_only": True,
            },
        }
        result = _self_hashed(receipt, field="receipt_sha256")
        _assert_no_local_paths(result)
        _validate_schema(
            result,
            EXECUTION_REPLAY_SCHEMA_PATH,
            field="execution replay receipt",
        )
        return result
    finally:
        if replay_dir.exists() and not replay_dir.is_symlink():
            try:
                os.chmod(replay_dir, 0o700)
            except OSError:
                pass
        shutil.rmtree(scratch, ignore_errors=True)


def write_execution_replay_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    """No-clobber publish a path-free replay receipt after full execution."""

    if receipt.get("schema_version") != EXECUTION_REPLAY_SCHEMA_VERSION:
        raise Open24ComparatorError("execution replay receipt version changed")
    _validate_schema(
        receipt,
        EXECUTION_REPLAY_SCHEMA_PATH,
        field="execution replay receipt",
    )
    self_hash = _verify_self_hash(
        receipt,
        field="execution replay receipt",
        key="receipt_sha256",
    )
    _assert_no_local_paths(receipt)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise Open24ComparatorError("refusing to overwrite execution replay receipt")
    _write_new(destination, _pretty_bytes(receipt))
    return self_hash


def validate_execution_replay_receipt(
    path: Path,
    *,
    track: ComparatorTrack,
    predictions_sha256: str,
    prediction_manifest_file_sha256: str,
    prediction_manifest_sha256: str,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate a consumed replay receipt against the observed run bundle."""

    document, file_identity = _read_json(path, field="execution replay receipt")
    _validate_schema(
        document,
        EXECUTION_REPLAY_SCHEMA_PATH,
        field="execution replay receipt",
    )
    self_hash = _verify_self_hash(
        document,
        field="execution replay receipt",
        key="receipt_sha256",
    )
    prediction = _mapping(document.get("prediction"), field="replay prediction")
    provenance = _mapping(document.get("provenance"), field="replay provenance")
    implementation = _mapping(
        document.get("implementation"), field="replay implementation"
    )
    measurement = _mapping(document.get("measurement"), field="replay measurement")
    execution = _mapping(document.get("execution"), field="replay execution")
    fixture = execution.get("fixture") is True
    if (
        document.get("track") != track
        or document.get("system_id") != system_id_for_track(track)
        or prediction.get("observed_file_sha256") != predictions_sha256
        or prediction.get("replayed_file_sha256") != predictions_sha256
        or provenance.get("observed_manifest_file_sha256")
        != prediction_manifest_file_sha256
        or provenance.get("observed_manifest_sha256") != prediction_manifest_sha256
        or implementation.get("byte_identical") is not True
        or measurement.get("fresh_measurement_collected") is not True
        or measurement.get("harness_replayed") is not True
        or (fixture and allow_fixture is not True)
    ):
        raise Open24ComparatorError(
            "execution replay receipt is bound to a different comparator run"
        )
    _assert_no_local_paths(document)
    return {
        "status": (
            "PASS_FIXTURE_FULL_EXECUTION_REPLAY_RECEIPT"
            if fixture
            else "PASS_FULL_EXECUTION_REPLAY_RECEIPT"
        ),
        "track": track,
        "system_id": system_id_for_track(track),
        "receipt_file_sha256": file_identity.sha256,
        "receipt_sha256": self_hash,
        "predictions_sha256": predictions_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "prediction_byte_identical": True,
        "measurement_harness_replayed": True,
        "inference_rerun": True,
        "fixture": fixture,
    }


__all__ = [
    "AIGUARD_SYSTEM_ID",
    "ComparatorHooks",
    "ComparatorRunInputs",
    "ComparatorTrack",
    "EXECUTION_REPLAY_SCHEMA_VERSION",
    "FRAMEWORK_SYSTEM_ID",
    "GENERATION_RECEIPT_SCHEMA_VERSION",
    "MEASUREMENT_SCHEMA_VERSION",
    "Open24ComparatorError",
    "RuntimeOptions",
    "STAGE_SPEC_SCHEMA_VERSION",
    "StageEvidence",
    "StageReplaySpec",
    "authorize_internal_preopen",
    "build_comparator_detector",
    "canonical_hash",
    "default_hooks",
    "generate_comparator_run",
    "inspect_comparator_artifact",
    "load_stage_replay_spec",
    "replay_comparator_execution",
    "snapshot_internal_input",
    "system_id_for_track",
    "validate_comparator_run",
    "validate_execution_replay_receipt",
    "write_execution_replay_receipt",
]
