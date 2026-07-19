#!/usr/bin/env python3
"""Produce fail-closed native replay evidence and release-v2 wrappers.

The producer has six deliberately narrow commands:

* ``service-source`` builds a path-free v4 service source identity from the
  immutable v3 predecessor, final service binding, full-system candidate and
  the one approved performance-harness source delta;
* ``quality-replay`` executes the quality producer's strict semantic replay;
* ``candidate-native-replay`` reruns the frozen candidate generator, requires
  byte-identical predictions and generation receipts, then rebuilds the full
  provenance manifest from the replayed files without waiting for quality;
* ``candidate-wrapper`` binds an already-published native candidate replay to
  the admitted quality receipt;
* ``candidate-replay`` performs those two candidate steps atomically when the
  quality receipt is already available;
* ``comparator-replay`` consumes a native comparator execution-replay receipt
  only after validating it against the admitted comparator manifest.

Every published file is strict JSON, path-free, self-hashed, mode 0444 and
no-clobber.  A community wrapper is never produced from a caller-supplied
boolean or CLI stdout status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import produce_public_synthetic_service_quality_v2 as quality  # noqa: E402
from scripts import validate_community_cascade_release_v2 as release  # noqa: E402
from scripts.community_contract_utils import (  # noqa: E402
    CommunityContractError,
    canonical_json_hash,
    load_json_path,
    read_regular_file,
    sha256_bytes,
    strict_json_bytes,
    validate_schema,
)

from pii_zh.evaluation.open24_comparator_provenance import (  # noqa: E402
    Open24ComparatorError,
    validate_execution_replay_receipt,
)
from pii_zh.evaluation.release_eval_v2_prediction_provenance import (  # noqa: E402
    AmendmentReplayInputs,
    CalibrationInputs,
    ModelRawInputs,
    PredictionProvenanceInputs,
    ReleaseEvalV2PredictionProvenanceError,
    _source_hashes,
    replay_release_eval_v2_prediction_manifest,
    validate_release_eval_v2_prediction_manifest,
)

PRODUCER_PATH = "scripts/produce_community_cascade_release_v2_evidence.py"
CANDIDATE_GENERATOR_PATH = "scripts/generate_release_eval_v2_candidate_predictions.py"
QUALITY_EVIDENCE_SCHEMA_PATH = (
    "configs/release/community_cascade_release_v2.quality-replay-evidence.schema.json"
)
CANDIDATE_EVIDENCE_SCHEMA_PATH = (
    "configs/release/community_cascade_release_v2.candidate-replay-evidence.schema.json"
)
REPLAY_SCHEMA_PATH = "configs/release/community_cascade_release_v2.replay-receipt.schema.json"
SOURCE_SCHEMA_PATH = "configs/release/community_service_source_identity_v4.schema.json"
PREDECESSOR_SOURCE_SCHEMA_PATH = "configs/release/community_service_source_identity_v3.schema.json"

QUALITY_EVIDENCE_SCHEMA_VERSION = "pii-zh.community-quality-strict-replay-evidence.v1"
CANDIDATE_EVIDENCE_SCHEMA_VERSION = "pii-zh.community-candidate-execution-replay-evidence.v1"
REPLAY_SCHEMA_VERSION = "pii-zh.community-cascade-replay-receipt.v2"
SOURCE_SCHEMA_VERSION = "pii-zh.community-service-source-identity.v4"
PREDECESSOR_SOURCE_SCHEMA_VERSION = "pii-zh.community-service-source-identity.v3"

_PACKAGE_SOURCE_ROOT = Path("src/pii_zh")
_PACKAGE_SOURCE_SUFFIXES = frozenset({".json", ".py", ".yaml", ".yml"})
_EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_ID = "pii_zh.evaluation.release_eval_v2_performance.py"
_EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS = frozenset(
    {
        _EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_ID,
        "pii_zh.data.synthetic.sota_release_eval_v1.py",
        "pii_zh.data.synthetic.sota_release_eval_v2.py",
        "pii_zh.data.synthetic.sota_v1.py",
        "pii_zh.data.synthetic.sota_v2_resplit.py",
    }
)

CanonicalTrack = Literal["model_raw", "full_system"]


class CommunityReleaseEvidenceError(CommunityContractError):
    """Raised when replay evidence cannot be produced without ambiguity."""


@dataclass(frozen=True, slots=True)
class ReleaseContext:
    quality: dict[str, Any]
    quality_payload: bytes
    model: dict[str, Any]
    service: dict[str, Any]
    source: dict[str, Any]
    internal_unlock: dict[str, Any]
    full_candidate: dict[str, Any]
    identity: dict[str, str]


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CommunityReleaseEvidenceError(f"{field} is not a SHA-256 digest")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommunityReleaseEvidenceError(f"{field} must be an object")
    return value


def _load_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path, field=field)
    return strict_json_bytes(payload, field=field), payload


def _load_immutable_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    """Load an exact mode-0444 predecessor without following a symlink."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise CommunityReleaseEvidenceError(f"{field} is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise CommunityReleaseEvidenceError(f"{field} must be an immutable mode-0444 file")
    document, payload = _load_json(path, field=field)
    try:
        after = path.lstat()
    except OSError as exc:
        raise CommunityReleaseEvidenceError(f"{field} changed while it was read") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CommunityReleaseEvidenceError(f"{field} changed while it was read")
    return document, payload


def _verify_self_hash(document: Mapping[str, Any], *, key: str, field: str) -> str:
    observed = _digest(document.get(key), field=f"{field}.{key}")
    if canonical_json_hash(document, remove=key) != observed:
        raise CommunityReleaseEvidenceError(f"{field} self hash does not verify")
    return observed


def _path_free(value: object, *, field: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"path", "paths"} or key.endswith(("_path", "_directory")):
                raise CommunityReleaseEvidenceError(f"{field} contains a path field")
            _path_free(nested, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _path_free(nested, field=f"{field}[{index}]")
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "~\\", "file:"))
        or "\\" in value
        or (len(value) > 2 and value[1:3] in {":/", ":\\"})
    ):
        raise CommunityReleaseEvidenceError(f"{field} contains a local path")


def _pretty_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CommunityReleaseEvidenceError("evidence is not serializable strict JSON") from exc


def _ensure_fresh(path: Path, *, field: str) -> None:
    if path.exists() or path.is_symlink() or os.path.lexists(path):
        raise CommunityReleaseEvidenceError(f"refusing to overwrite {field}")


def publish_read_only_json(path: Path, document: Mapping[str, Any]) -> str:
    """Publish one immutable JSON document atomically without clobbering."""

    _path_free(document)
    payload = _pretty_bytes(document)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ensure_fresh(destination, field="evidence output")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise CommunityReleaseEvidenceError("refusing to overwrite evidence output") from exc
        if stat.S_IMODE(destination.stat().st_mode) != 0o444:
            raise CommunityReleaseEvidenceError("evidence output is not mode 0444")
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


def _validate_document_schema(document: Mapping[str, Any], *, schema_path: str, field: str) -> None:
    schema = load_json_path(REPOSITORY_ROOT / schema_path, field=f"{field} schema")
    validate_schema(document, schema, field=field)
    _path_free(document, field=field)


def _load_release_context(
    *,
    quality_receipt: Path,
    final_model_manifest: Path,
    service_configuration_manifest: Path,
    service_source_manifest: Path,
    predecessor_service_source_manifest: Path,
    internal_preopen_unlock: Path,
    full_system_candidate_manifest: Path,
) -> ReleaseContext:
    quality_document, quality_payload = _load_json(quality_receipt, field="quality result receipt")
    quality.validate_receipt_schema(quality_document, repository_root=REPOSITORY_ROOT)
    release._verify_quality_implementation(quality_document, repository_root=REPOSITORY_ROOT)
    if (
        quality_document.get("reported_status") != "PASS"
        or _mapping(quality_document.get("tracks"), field="quality tracks")
        .get("full_system", {})
        .get("declared_status")
        != "PASS"
        or _mapping(quality_document.get("claims"), field="quality claims").get(
            "claim_activation_allowed"
        )
        is not False
    ):
        raise CommunityReleaseEvidenceError("quality result is not an admissible strict PASS")

    model, model_payload = _load_json(final_model_manifest, field="final model manifest")
    service, service_payload = _load_json(
        service_configuration_manifest, field="service configuration manifest"
    )
    source, _source_payload = _load_json(
        service_source_manifest, field="service source identity manifest"
    )
    predecessor_source, predecessor_source_payload = _load_immutable_json(
        predecessor_service_source_manifest,
        field="predecessor service source identity manifest",
    )
    internal_unlock, _unlock_payload = _load_json(
        internal_preopen_unlock, field="internal pre-open unlock"
    )
    full_candidate, _candidate_payload = _load_json(
        full_system_candidate_manifest, field="full-system candidate manifest"
    )
    _verify_self_hash(model, key="manifest_sha256", field="final model manifest")
    _verify_self_hash(service, key="manifest_sha256", field="service configuration manifest")
    _verify_self_hash(source, key="manifest_sha256", field="service source identity manifest")
    _verify_self_hash(internal_unlock, key="receipt_sha256", field="internal pre-open unlock")
    _verify_self_hash(full_candidate, key="manifest_sha256", field="full-system candidate")
    release._validate_model_manifest(model)
    release._validate_service_manifest(
        service, model=model, model_file_sha256=sha256_bytes(model_payload)
    )
    release._file_binding_matches(
        final_model_manifest,
        quality_document["input_bindings"]["final_model_manifest"],
        field_name="quality final model manifest",
    )
    release._file_binding_matches(
        service_configuration_manifest,
        quality_document["input_bindings"]["service_configuration_manifest"],
        field_name="quality service configuration manifest",
    )
    release._file_binding_matches(
        full_system_candidate_manifest,
        quality_document["input_bindings"]["tracks"]["full_system"][
            "candidate_prediction_manifest"
        ],
        field_name="quality full-system candidate manifest",
    )
    release._validate_candidate_manifest(
        full_system_candidate_manifest,
        full_candidate,
        track="full_system",
        quality_receipt=quality_document,
        model=model,
        service=service,
    )
    source_schema = load_json_path(
        REPOSITORY_ROOT / SOURCE_SCHEMA_PATH, field="service source identity schema"
    )
    predecessor_source_schema = load_json_path(
        REPOSITORY_ROOT / PREDECESSOR_SOURCE_SCHEMA_PATH,
        field="predecessor service source identity schema",
    )
    release._validate_source_manifest(
        source,
        schema=source_schema,
        predecessor_document=predecessor_source,
        predecessor_payload=predecessor_source_payload,
        predecessor_schema=predecessor_source_schema,
        service=service,
        full_candidate=full_candidate,
    )
    release._validate_internal_unlock(
        internal_unlock,
        model=model,
        service=service,
        quality_receipt=quality_document,
        full_candidate=full_candidate,
    )
    identity = release._identity_payload(
        quality_receipt=quality_document,
        model=model,
        service=service,
        source=source,
        internal_unlock=internal_unlock,
    )
    return ReleaseContext(
        quality=quality_document,
        quality_payload=quality_payload,
        model=model,
        service=service,
        source=source,
        internal_unlock=internal_unlock,
        full_candidate=full_candidate,
        identity=identity,
    )


def _runtime_package_source_hashes(repository_root: Path) -> dict[str, str]:
    """Return the complete installed-package source projection under ``src/pii_zh``."""

    source_root = repository_root / _PACKAGE_SOURCE_ROOT
    try:
        root_metadata = source_root.lstat()
    except OSError as exc:
        raise CommunityReleaseEvidenceError("runtime package source root is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CommunityReleaseEvidenceError("runtime package source root is unsafe")

    result: dict[str, str] = {}
    try:
        entries = sorted(source_root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CommunityReleaseEvidenceError("runtime package inventory failed") from exc
    for path in entries:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CommunityReleaseEvidenceError("runtime package entry is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise CommunityReleaseEvidenceError("runtime package inventory contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CommunityReleaseEvidenceError("runtime package inventory contains a special file")
        if path.suffix not in _PACKAGE_SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(source_root).as_posix()
        logical_id = f"pii_zh.{relative.replace('/', '.')}"
        if logical_id in result:
            raise CommunityReleaseEvidenceError(
                "runtime package logical source identity is ambiguous"
            )
        result[logical_id] = sha256_bytes(
            read_regular_file(path, field=f"runtime package source {logical_id}")
        )
    if not result:
        raise CommunityReleaseEvidenceError("runtime package source inventory is empty")
    return dict(sorted(result.items()))


def _source_successor_delta(
    predecessor: Mapping[str, str], current: Mapping[str, str]
) -> dict[str, Any]:
    """Return the only admitted v3-to-v4 runtime inventory transition."""

    predecessor_keys = set(predecessor)
    current_keys = set(current)
    added = {
        logical_id: current[logical_id] for logical_id in sorted(current_keys - predecessor_keys)
    }
    changed = {
        logical_id: current[logical_id]
        for logical_id in sorted(current_keys & predecessor_keys)
        if current[logical_id] != predecessor[logical_id]
    }
    removed = {
        logical_id: predecessor[logical_id]
        for logical_id in sorted(predecessor_keys - current_keys)
    }
    expected_changed = {
        logical_id: current[logical_id]
        for logical_id in sorted(_EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS)
        if logical_id in current
    }
    if (
        added
        or removed
        or set(expected_changed) != _EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS
        or changed != expected_changed
    ):
        raise CommunityReleaseEvidenceError(
            "source successor delta is not the approved performance-and-portability amendment"
        )
    result: dict[str, Any] = {
        "added": added,
        "changed": changed,
        "removed": removed,
        "added_count": len(added),
        "changed_count": len(changed),
        "removed_count": len(removed),
        "summary_sha256": "",
    }
    result["summary_sha256"] = canonical_json_hash(result, remove="summary_sha256")
    return result


def build_service_source_manifest(
    *,
    final_model_manifest: Path,
    service_configuration_manifest: Path,
    full_system_candidate_manifest: Path,
    predecessor_source_manifest: Path,
) -> dict[str, Any]:
    """Build the current closure as the narrow performance/portability successor to v3."""

    model, model_payload = _load_json(final_model_manifest, field="final model manifest")
    service, _service_payload = _load_json(
        service_configuration_manifest, field="service configuration manifest"
    )
    candidate, _candidate_payload = _load_json(
        full_system_candidate_manifest, field="full-system candidate manifest"
    )
    predecessor, predecessor_payload = _load_immutable_json(
        predecessor_source_manifest,
        field="v3 predecessor service source identity",
    )
    _verify_self_hash(model, key="manifest_sha256", field="final model manifest")
    _verify_self_hash(service, key="manifest_sha256", field="service configuration manifest")
    _verify_self_hash(candidate, key="manifest_sha256", field="full-system candidate")
    _verify_self_hash(
        predecessor,
        key="manifest_sha256",
        field="v3 predecessor service source identity",
    )
    release._validate_model_manifest(model)
    release._validate_service_manifest(
        service, model=model, model_file_sha256=sha256_bytes(model_payload)
    )
    candidate_schema = load_json_path(
        REPOSITORY_ROOT / quality.CANDIDATE_PREDICTION_SCHEMA_PATH,
        field="candidate provenance schema",
    )
    validate_schema(candidate, candidate_schema, field="full-system candidate manifest")
    _validate_document_schema(
        predecessor,
        schema_path=PREDECESSOR_SOURCE_SCHEMA_PATH,
        field="v3 predecessor service source identity",
    )
    if predecessor.get("schema_version") != PREDECESSOR_SOURCE_SCHEMA_VERSION:
        raise CommunityReleaseEvidenceError("service source predecessor is not v3")
    metadata = validate_release_eval_v2_prediction_manifest(full_system_candidate_manifest)
    candidate_service = _mapping(candidate.get("service"), field="candidate service")
    sources = dict(
        sorted(
            _mapping(
                candidate_service.get("implementation_source_sha256"),
                field="candidate service source map",
            ).items()
        )
    )
    current_sources = _source_hashes(REPOSITORY_ROOT)
    predecessor_runtime_sources = dict(
        sorted(
            _mapping(
                predecessor.get("runtime_package_source_sha256"),
                field="v3 predecessor runtime package source map",
            ).items()
        )
    )
    if (
        metadata.get("status") != "PASS_METADATA_ONLY"
        or metadata.get("full_replay_required") is not True
        or candidate.get("track") != "full_system"
        or candidate_service.get("profile_id") != service.get("profile_id")
        or candidate_service.get("implementation_sha256") != service.get("implementation_sha256")
        or candidate_service.get("model_identity_sha256") != service.get("model_identity_sha256")
        or sources != current_sources
    ):
        raise CommunityReleaseEvidenceError(
            "full-system candidate does not bind the current final service sources"
        )
    if (
        predecessor.get("service_id") != service.get("service_id")
        or predecessor.get("profile_id") != service.get("profile_id")
        or predecessor.get("implementation_sha256") != service.get("implementation_sha256")
        or predecessor.get("implementation_source_sha256") != sources
        or predecessor.get("runtime_package_inventory_sha256")
        != canonical_json_hash(predecessor_runtime_sources)
        or predecessor.get("runtime_package_file_count") != len(predecessor_runtime_sources)
        or predecessor.get("runtime_package_inventory_complete") is not True
    ):
        raise CommunityReleaseEvidenceError(
            "v3 predecessor does not bind the same service and a complete runtime inventory"
        )
    runtime_sources = _runtime_package_source_hashes(REPOSITORY_ROOT)
    if runtime_sources != _runtime_package_source_hashes(REPOSITORY_ROOT):
        raise CommunityReleaseEvidenceError(
            "runtime package sources changed during identity construction"
        )
    successor_delta = _source_successor_delta(predecessor_runtime_sources, runtime_sources)
    result: dict[str, Any] = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "service_id": service["service_id"],
        "profile_id": service["profile_id"],
        "implementation_source_sha256": sources,
        "implementation_sha256": service["implementation_sha256"],
        "runtime_package_source_sha256": runtime_sources,
        "runtime_package_inventory_sha256": canonical_json_hash(runtime_sources),
        "runtime_package_file_count": len(runtime_sources),
        "runtime_package_inventory_complete": True,
        "predecessor": {
            "schema_version": PREDECESSOR_SOURCE_SCHEMA_VERSION,
            "file_sha256": sha256_bytes(predecessor_payload),
            "manifest_sha256": predecessor["manifest_sha256"],
            "runtime_package_inventory_sha256": predecessor["runtime_package_inventory_sha256"],
            "runtime_package_file_count": predecessor["runtime_package_file_count"],
        },
        "successor_delta": successor_delta,
        "contains_paths": False,
        "manifest_sha256": "",
    }
    result["manifest_sha256"] = canonical_json_hash(result, remove="manifest_sha256")
    _validate_document_schema(
        result, schema_path=SOURCE_SCHEMA_PATH, field="service source identity manifest"
    )
    return result


def _wrapper_implementation(
    *, receipt_type: str, target_document: Mapping[str, Any]
) -> tuple[str, str]:
    wrapper_file_sha256 = sha256_bytes(
        read_regular_file(REPOSITORY_ROOT / PRODUCER_PATH, field="replay wrapper producer")
    )
    if receipt_type == "quality_strict_replay":
        underlying = {"quality_implementation_bindings": target_document["implementation_bindings"]}
    elif receipt_type == "candidate_prediction_full_replay":
        generation = _mapping(target_document.get("generation"), field="candidate generation")
        underlying = {
            "candidate_provenance_implementation": target_document["provenance_implementation"],
            "candidate_generator_implementation_sha256": _digest(
                generation.get("generator_implementation_sha256"),
                field="candidate generator implementation",
            ),
        }
    elif receipt_type == "comparator_generation_replay":
        underlying = {"comparator_system_bindings": target_document["system_bindings"]}
    else:
        raise CommunityReleaseEvidenceError("unsupported replay wrapper type")
    implementation = {
        "receipt_type": receipt_type,
        "wrapper_file_sha256": wrapper_file_sha256,
        "underlying_implementation": underlying,
    }
    return wrapper_file_sha256, canonical_json_hash(implementation)


def _native_binding(document: Mapping[str, Any], payload: bytes, *, field: str) -> dict[str, str]:
    return {
        "schema_version": str(document["schema_version"]),
        "file_sha256": sha256_bytes(payload),
        "canonical_sha256": _verify_self_hash(document, key="receipt_sha256", field=field),
    }


def build_release_wrapper(
    *,
    receipt_type: Literal[
        "quality_strict_replay",
        "candidate_prediction_full_replay",
        "comparator_generation_replay",
    ],
    track: CanonicalTrack | None,
    context: ReleaseContext,
    target_document: Mapping[str, Any],
    target_file_sha256: str,
    target_kind: str,
    upstream_replay_evidence: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact gate wrapper while binding its validated native replay."""

    target_canonical = _digest(
        target_document.get("receipt_sha256") or target_document.get("manifest_sha256"),
        field="replay target canonical hash",
    )
    target = {
        "artifact_kind": target_kind,
        "file_sha256": _digest(target_file_sha256, field="replay target file hash"),
        "canonical_sha256": target_canonical,
    }
    upstream = {
        "schema_version": str(upstream_replay_evidence["schema_version"]),
        "file_sha256": _digest(
            upstream_replay_evidence["file_sha256"], field="native replay file hash"
        ),
        "canonical_sha256": _digest(
            upstream_replay_evidence["canonical_sha256"],
            field="native replay canonical hash",
        ),
    }
    implementation_file, implementation_identity = _wrapper_implementation(
        receipt_type=receipt_type, target_document=target_document
    )
    quality_file_sha256 = sha256_bytes(context.quality_payload)
    input_material: dict[str, Any] = {
        "receipt_type": receipt_type,
        "track": track,
        "quality_file_sha256": quality_file_sha256,
        "quality_receipt_sha256": context.quality["receipt_sha256"],
        "identity": dict(context.identity),
        "quality_input_bindings": context.quality["input_bindings"],
        "target": target,
        "upstream_replay_evidence": upstream,
    }
    if track is not None:
        input_material["quality_track"] = context.quality["tracks"][track]
    output_material = {
        "target": target,
        "reported_status": context.quality["reported_status"],
        "track_status": (
            None if track is None else context.quality["tracks"][track]["declared_status"]
        ),
    }
    receipt: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "receipt_type": receipt_type,
        "track": track,
        "status": "EXACT_REPLAY_PASS",
        "quality": {
            "file_sha256": quality_file_sha256,
            "receipt_sha256": context.quality["receipt_sha256"],
        },
        "target": target,
        "identity": dict(context.identity),
        "replay": {
            "implementation_file_sha256": implementation_file,
            "implementation_identity_sha256": implementation_identity,
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
    _validate_document_schema(
        receipt, schema_path=REPLAY_SCHEMA_PATH, field="community replay wrapper"
    )
    return receipt


def _candidate_execution_identity(
    candidate_document: Mapping[str, Any], *, manifest_file_sha256: str
) -> dict[str, Any]:
    prediction = _mapping(candidate_document.get("prediction"), field="candidate prediction")
    generation = _mapping(candidate_document.get("generation"), field="candidate generation")
    return {
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": candidate_document["manifest_sha256"],
        "predictions_file_sha256": prediction["file_sha256"],
        "prediction_document_count": prediction["document_count"],
        "generation_receipt_file_sha256": generation["receipt_file_sha256"],
        "generation_receipt_sha256": generation["receipt_sha256"],
    }


def _quality_inputs(args: argparse.Namespace) -> quality.ProducerInputs:
    tracks = tuple(
        quality.TrackInput(
            track=raw[0],
            candidate_predictions=Path(raw[1]),
            candidate_prediction_manifest=Path(raw[2]),
            comparator_predictions=Path(raw[3]),
            comparator_prediction_manifest=Path(raw[4]),
            performance_manifest=Path(raw[5]),
        )
        for raw in args.track_input
    )
    return quality.ProducerInputs(
        gold=args.gold,
        gold_dataset_manifest=args.gold_dataset_manifest,
        gold_materialization_receipt=args.gold_materialization_receipt,
        gold_freeze_receipt=args.gold_freeze_receipt,
        gold_split="internal_evaluation",
        final_model_manifest=args.final_model_manifest,
        service_configuration=args.service_configuration_manifest,
        comparator_registry=args.comparator_registry,
        tracks=tracks,
        internal_unlock=args.internal_preopen_unlock,
        final_model_binding=args.final_model_manifest,
        service_configuration_binding=args.service_configuration_manifest,
    )


def _candidate_inputs(
    args: argparse.Namespace, *, predictions: Path, generation_receipt: Path
) -> PredictionProvenanceInputs:
    if len(args.candidate) != 3:
        raise CommunityReleaseEvidenceError("exactly three frozen candidates are required")
    calibration = CalibrationInputs(
        bundle=args.calibration_bundle,
        diagnostics=args.calibration_diagnostics,
        fit_gold=args.calibration_fit_gold,
        fit_predictions=args.calibration_fit_predictions,
        fit_generation_receipt=args.calibration_fit_generation_receipt,
        fit_prediction_manifest=args.calibration_fit_prediction_manifest,
    )
    model_raw = (
        None
        if args.track == "model_raw"
        else ModelRawInputs(
            predictions=args.model_raw_predictions,
            generation_receipt=args.model_raw_generation_receipt,
            prediction_manifest=args.model_raw_prediction_manifest,
        )
    )
    return PredictionProvenanceInputs(
        track=cast(CanonicalTrack, args.track),
        target_split="internal_evaluation",
        predictions=predictions,
        generation_receipt=generation_receipt,
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


def _compare_regular_files(left: Path, right: Path, *, field: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        for path in (left, right):
            descriptors.append(os.open(path, flags))
        before = [os.fstat(descriptor) for descriptor in descriptors]
        if any(not stat.S_ISREG(item.st_mode) for item in before):
            raise CommunityReleaseEvidenceError(f"{field} is not a regular file pair")
        digest = hashlib.sha256()
        while True:
            first = os.read(descriptors[0], 1024 * 1024)
            second = os.read(descriptors[1], 1024 * 1024)
            if first != second:
                raise CommunityReleaseEvidenceError(f"{field} replay bytes differ")
            if not first:
                break
            digest.update(first)
        after = [os.fstat(descriptor) for descriptor in descriptors]
        for old, new in zip(before, after, strict=True):
            if (
                old.st_dev,
                old.st_ino,
                old.st_size,
                old.st_mtime_ns,
                old.st_ctime_ns,
            ) != (
                new.st_dev,
                new.st_ino,
                new.st_size,
                new.st_mtime_ns,
                new.st_ctime_ns,
            ):
                raise CommunityReleaseEvidenceError(f"{field} changed during comparison")
        return digest.hexdigest()
    except OSError as exc:
        raise CommunityReleaseEvidenceError(f"{field} could not be compared safely") from exc
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _run_candidate_generator(
    args: argparse.Namespace, *, predictions: Path, generation_receipt: Path
) -> None:
    observed_generation, _payload = _load_json(
        args.generation_receipt, field="observed candidate generation receipt"
    )
    runtime = _mapping(observed_generation.get("runtime"), field="candidate generation runtime")
    scope = _mapping(observed_generation.get("scope"), field="candidate generation scope")
    expected_device = "cuda:0" if runtime.get("device_class") == "cuda" else "cpu"
    if args.device != expected_device:
        raise CommunityReleaseEvidenceError(
            "candidate replay device class differs from the frozen generation receipt"
        )
    mode = "model-raw" if args.track == "model_raw" else "cascade"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / CANDIDATE_GENERATOR_PATH),
        "--mode",
        mode,
        "--target-split",
        "internal_evaluation",
        "--scope",
        str(scope.get("scope_id")),
        "--model",
        str(args.model_artifact),
        "--input",
        str(args.gold),
        "--predictions",
        str(predictions),
        "--receipt",
        str(generation_receipt),
        "--internal-unlock",
        str(args.internal_preopen_unlock),
        "--final-model-binding",
        str(args.final_model_manifest),
        "--service-configuration-binding",
        str(args.service_configuration_manifest),
        "--device",
        args.device,
        "--document-batch-size",
        str(runtime.get("document_batch_size")),
        "--micro-batch-size",
        str(runtime.get("micro_batch_size")),
    ]
    if args.track == "full_system":
        command.extend(("--calibration", str(args.calibration_bundle)))
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CommunityReleaseEvidenceError("candidate inference replay failed")


def _release_context_from_args(args: argparse.Namespace) -> ReleaseContext:
    return _load_release_context(
        quality_receipt=args.quality_receipt,
        final_model_manifest=args.final_model_manifest,
        service_configuration_manifest=args.service_configuration_manifest,
        service_source_manifest=args.service_source_manifest,
        predecessor_service_source_manifest=args.predecessor_service_source_manifest,
        internal_preopen_unlock=args.internal_preopen_unlock,
        full_system_candidate_manifest=args.full_system_candidate_manifest,
    )


def _produce_quality_replay(args: argparse.Namespace) -> dict[str, str]:
    _ensure_fresh(args.native_output, field="quality native replay evidence")
    _ensure_fresh(args.output, field="quality replay wrapper")
    if args.native_output.resolve(strict=False) == args.output.resolve(strict=False):
        raise CommunityReleaseEvidenceError("quality replay outputs must be distinct")
    context = _release_context_from_args(args)
    rebuilt = quality.replay_receipt(args.quality_receipt, _quality_inputs(args))
    if rebuilt != context.quality:
        raise CommunityReleaseEvidenceError("quality replay result changed after validation")
    implementation = _mapping(
        context.quality.get("implementation_bindings"), field="quality implementation"
    )
    native: dict[str, Any] = {
        "schema_version": QUALITY_EVIDENCE_SCHEMA_VERSION,
        "receipt_type": "quality_strict_semantic_replay",
        "status": "PASS",
        "target": {
            "artifact_kind": "quality_result_receipt",
            "file_sha256": sha256_bytes(context.quality_payload),
            "canonical_sha256": context.quality["receipt_sha256"],
        },
        "replay": {
            "producer_file_sha256": implementation["producer_sha256"],
            "implementation_identity_sha256": canonical_json_hash(implementation),
            "rebuilt_receipt_sha256": context.quality["receipt_sha256"],
            "exact_semantic_equality": True,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "aggregate_only": True,
            "network_used": False,
        },
        "receipt_sha256": "",
    }
    native["receipt_sha256"] = canonical_json_hash(native, remove="receipt_sha256")
    _validate_document_schema(
        native,
        schema_path=QUALITY_EVIDENCE_SCHEMA_PATH,
        field="quality native replay evidence",
    )
    publish_read_only_json(args.native_output, native)
    native_document, native_payload = _load_json(
        args.native_output, field="quality native replay evidence"
    )
    _validate_document_schema(
        native_document,
        schema_path=QUALITY_EVIDENCE_SCHEMA_PATH,
        field="quality native replay evidence",
    )
    native_binding = _native_binding(
        native_document, native_payload, field="quality native replay evidence"
    )
    wrapper = build_release_wrapper(
        receipt_type="quality_strict_replay",
        track=None,
        context=context,
        target_document=context.quality,
        target_file_sha256=sha256_bytes(context.quality_payload),
        target_kind="quality_result_receipt",
        upstream_replay_evidence=native_binding,
    )
    publish_read_only_json(args.output, wrapper)
    return {
        "native_receipt_sha256": native["receipt_sha256"],
        "wrapper_receipt_sha256": wrapper["receipt_sha256"],
    }


def _load_candidate_target(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bytes]:
    if len(args.candidate) != 3:
        raise CommunityReleaseEvidenceError("exactly three frozen candidates are required")
    candidate_document, candidate_payload = _load_json(
        args.manifest, field=f"{args.track} candidate manifest"
    )
    _verify_self_hash(
        candidate_document, key="manifest_sha256", field=f"{args.track} candidate manifest"
    )
    metadata = validate_release_eval_v2_prediction_manifest(args.manifest)
    if (
        metadata.get("status") != "PASS_METADATA_ONLY"
        or metadata.get("full_replay_required") is not True
        or metadata.get("track") != args.track
        or metadata.get("target_split") != "internal_evaluation"
        or metadata.get("manifest_file_sha256") != sha256_bytes(candidate_payload)
        or metadata.get("manifest_sha256") != candidate_document["manifest_sha256"]
    ):
        raise CommunityReleaseEvidenceError(
            "candidate manifest is not an admissible formal replay target"
        )
    return candidate_document, candidate_payload


def _execute_candidate_native_replay(
    args: argparse.Namespace,
    *,
    candidate_document: Mapping[str, Any],
    candidate_payload: bytes,
) -> dict[str, Any]:
    runtime_sources_before = _runtime_package_source_hashes(REPOSITORY_ROOT)
    output_parent = args.native_output.parent.expanduser()
    output_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".candidate-release-replay.", dir=output_parent))
    replayed_predictions = scratch / "predictions.jsonl"
    replayed_generation = scratch / "generation.json"
    try:
        _run_candidate_generator(
            args,
            predictions=replayed_predictions,
            generation_receipt=replayed_generation,
        )
        observed_prediction_sha = _compare_regular_files(
            args.predictions,
            replayed_predictions,
            field="candidate prediction execution replay",
        )
        if observed_prediction_sha != candidate_document["prediction"]["file_sha256"]:
            raise CommunityReleaseEvidenceError(
                "candidate replayed prediction hash differs from provenance"
            )
        _compare_regular_files(
            args.generation_receipt,
            replayed_generation,
            field="candidate generation receipt execution replay",
        )
        replay_result = replay_release_eval_v2_prediction_manifest(
            args.manifest,
            _candidate_inputs(
                args,
                predictions=replayed_predictions,
                generation_receipt=replayed_generation,
            ),
        )
        if replay_result != {
            "status": "PASS",
            "track": args.track,
            "target_split": "internal_evaluation",
            "manifest_file_sha256": sha256_bytes(candidate_payload),
            "manifest_sha256": candidate_document["manifest_sha256"],
            "predictions_file_sha256": candidate_document["prediction"]["file_sha256"],
            "prediction_document_count": candidate_document["prediction"]["document_count"],
        }:
            raise CommunityReleaseEvidenceError(
                "candidate provenance replay returned a disconnected aggregate result"
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    runtime_sources_after = _runtime_package_source_hashes(REPOSITORY_ROOT)
    if runtime_sources_after != runtime_sources_before:
        raise CommunityReleaseEvidenceError(
            "runtime package sources changed during candidate execution replay"
        )
    track = candidate_document.get("track")
    if track not in {"model_raw", "full_system"}:
        raise CommunityReleaseEvidenceError("candidate replay track is not canonical")
    target = _candidate_execution_identity(
        candidate_document, manifest_file_sha256=sha256_bytes(candidate_payload)
    )
    generation = _mapping(candidate_document.get("generation"), field="candidate generation")
    provenance = _mapping(
        candidate_document.get("provenance_implementation"),
        field="candidate provenance implementation",
    )
    result: dict[str, Any] = {
        "schema_version": CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        "receipt_type": "candidate_prediction_full_execution_replay",
        "status": "PASS",
        "track": track,
        "target": target,
        "replayed": {
            **target,
            "predictions_byte_identical": True,
            "generation_receipt_exact_equal": True,
            "provenance_exact_equal": True,
        },
        "implementation": {
            "replay_producer_file_sha256": sha256_bytes(
                read_regular_file(REPOSITORY_ROOT / PRODUCER_PATH, field="replay producer")
            ),
            "generator_implementation_sha256": generation["generator_implementation_sha256"],
            "provenance_implementation_sha256": provenance["implementation_sha256"],
            "runtime_package_inventory_sha256": canonical_json_hash(runtime_sources_after),
            "runtime_package_file_count": len(runtime_sources_after),
            "runtime_package_inventory_complete": True,
        },
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_document_ids": False,
            "contains_entity_values": False,
            "aggregate_only": True,
            "network_used": False,
        },
        "receipt_sha256": "",
    }
    result["receipt_sha256"] = canonical_json_hash(result, remove="receipt_sha256")
    _validate_document_schema(
        result,
        schema_path=CANDIDATE_EVIDENCE_SCHEMA_PATH,
        field="candidate native replay evidence",
    )
    return result


def _validate_candidate_against_quality(
    args: argparse.Namespace,
    *,
    context: ReleaseContext,
    candidate_document: Mapping[str, Any],
) -> None:
    release._file_binding_matches(
        args.manifest,
        context.quality["input_bindings"]["tracks"][args.track]["candidate_prediction_manifest"],
        field_name=f"quality {args.track} candidate manifest",
    )
    release._validate_candidate_manifest(
        args.manifest,
        candidate_document,
        track=args.track,
        quality_receipt=context.quality,
        model=context.model,
        service=context.service,
    )


def _produce_candidate_native_replay(args: argparse.Namespace) -> dict[str, str]:
    _ensure_fresh(args.native_output, field="candidate native replay evidence")
    candidate_document, candidate_payload = _load_candidate_target(args)
    native = _execute_candidate_native_replay(
        args,
        candidate_document=candidate_document,
        candidate_payload=candidate_payload,
    )
    publish_read_only_json(args.native_output, native)
    return {"native_receipt_sha256": native["receipt_sha256"]}


def _build_candidate_wrapper_from_native(
    args: argparse.Namespace,
    *,
    native_path: Path,
    context: ReleaseContext,
    candidate_document: Mapping[str, Any],
    candidate_payload: bytes,
) -> dict[str, Any]:
    _validate_candidate_against_quality(
        args, context=context, candidate_document=candidate_document
    )
    native_document, native_payload = _load_json(
        native_path, field="candidate native replay evidence"
    )
    native_implementation = _mapping(
        native_document.get("implementation"), field="candidate native replay implementation"
    )
    if (
        native_implementation.get("runtime_package_inventory_sha256")
        != context.source.get("runtime_package_inventory_sha256")
        or native_implementation.get("runtime_package_file_count")
        != context.source.get("runtime_package_file_count")
        or native_implementation.get("runtime_package_inventory_complete") is not True
    ):
        raise CommunityReleaseEvidenceError(
            "candidate native replay predates the admitted runtime source closure"
        )
    native_binding = release._verify_candidate_native_replay(
        native_path,
        track=args.track,
        candidate_document=candidate_document,
        candidate_file_sha256=sha256_bytes(candidate_payload),
        repository_root=REPOSITORY_ROOT,
    )
    del native_payload
    wrapper = build_release_wrapper(
        receipt_type="candidate_prediction_full_replay",
        track=cast(CanonicalTrack, args.track),
        context=context,
        target_document=candidate_document,
        target_file_sha256=sha256_bytes(candidate_payload),
        target_kind="candidate_prediction_manifest",
        upstream_replay_evidence=native_binding,
    )
    return wrapper


def _produce_candidate_wrapper(args: argparse.Namespace) -> dict[str, str]:
    _ensure_fresh(args.output, field="candidate replay wrapper")
    context = _release_context_from_args(args)
    candidate_document, candidate_payload = _load_json(
        args.manifest, field=f"{args.track} candidate manifest"
    )
    _verify_self_hash(
        candidate_document, key="manifest_sha256", field=f"{args.track} candidate manifest"
    )
    wrapper = _build_candidate_wrapper_from_native(
        args,
        native_path=args.native_replay,
        context=context,
        candidate_document=candidate_document,
        candidate_payload=candidate_payload,
    )
    publish_read_only_json(args.output, wrapper)
    return {"wrapper_receipt_sha256": wrapper["receipt_sha256"]}


def _produce_candidate_replay(args: argparse.Namespace) -> dict[str, str]:
    _ensure_fresh(args.native_output, field="candidate native replay evidence")
    _ensure_fresh(args.output, field="candidate replay wrapper")
    if args.native_output.resolve(strict=False) == args.output.resolve(strict=False):
        raise CommunityReleaseEvidenceError("candidate replay outputs must be distinct")
    context = _release_context_from_args(args)
    candidate_document, candidate_payload = _load_candidate_target(args)
    _validate_candidate_against_quality(
        args, context=context, candidate_document=candidate_document
    )
    native = _execute_candidate_native_replay(
        args,
        candidate_document=candidate_document,
        candidate_payload=candidate_payload,
    )
    publish_read_only_json(args.native_output, native)
    wrapper = _build_candidate_wrapper_from_native(
        args,
        native_path=args.native_output,
        context=context,
        candidate_document=candidate_document,
        candidate_payload=candidate_payload,
    )
    publish_read_only_json(args.output, wrapper)
    return {
        "native_receipt_sha256": native["receipt_sha256"],
        "wrapper_receipt_sha256": wrapper["receipt_sha256"],
    }


def _validate_comparator_native(
    *,
    path: Path,
    track: CanonicalTrack,
    comparator_document: Mapping[str, Any],
    comparator_file_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    document, payload = _load_json(path, field=f"{track} comparator execution replay")
    prediction = _mapping(comparator_document.get("prediction"), field="comparator prediction")
    consumed = validate_execution_replay_receipt(
        path,
        track=track,
        predictions_sha256=str(prediction["file_sha256"]),
        prediction_manifest_file_sha256=comparator_file_sha256,
        prediction_manifest_sha256=str(comparator_document["manifest_sha256"]),
        allow_fixture=False,
    )
    provenance = _mapping(document.get("provenance"), field="comparator replay provenance")
    replay_prediction = _mapping(document.get("prediction"), field="comparator replay prediction")
    implementation = _mapping(
        document.get("implementation"), field="comparator replay implementation"
    )
    system_bindings = _mapping(
        comparator_document.get("system_bindings"), field="comparator system bindings"
    )
    generator_binding = _mapping(
        system_bindings.get("generator_implementation"),
        field="comparator generator binding",
    )
    if (
        consumed.get("status") != "PASS_FULL_EXECUTION_REPLAY_RECEIPT"
        or consumed.get("fixture") is not False
        or replay_prediction.get("document_count") != prediction.get("document_count")
        or provenance.get("stage_and_support_strict_replay_passed") is not True
        or implementation.get("observed_file_sha256") != generator_binding.get("file_sha256")
        or implementation.get("replayed_file_sha256") != generator_binding.get("file_sha256")
    ):
        raise CommunityReleaseEvidenceError(
            "comparator native replay is not fully bound to the admitted bundle"
        )
    return document, payload


def _produce_comparator_replay(args: argparse.Namespace) -> dict[str, str]:
    _ensure_fresh(args.output, field="comparator replay wrapper")
    context = _release_context_from_args(args)
    comparator_document, comparator_payload = _load_json(
        args.comparator_manifest, field=f"{args.track} comparator manifest"
    )
    _verify_self_hash(
        comparator_document,
        key="manifest_sha256",
        field=f"{args.track} comparator manifest",
    )
    release._file_binding_matches(
        args.comparator_manifest,
        context.quality["input_bindings"]["tracks"][args.track]["comparator_prediction_manifest"],
        field_name=f"quality {args.track} comparator manifest",
    )
    release._validate_comparator_manifest(
        comparator_document, track=args.track, quality_receipt=context.quality
    )
    native_document, native_payload = _validate_comparator_native(
        path=args.execution_replay,
        track=cast(CanonicalTrack, args.track),
        comparator_document=comparator_document,
        comparator_file_sha256=sha256_bytes(comparator_payload),
    )
    native_binding = _native_binding(
        native_document, native_payload, field="comparator native execution replay"
    )
    wrapper = build_release_wrapper(
        receipt_type="comparator_generation_replay",
        track=cast(CanonicalTrack, args.track),
        context=context,
        target_document=comparator_document,
        target_file_sha256=sha256_bytes(comparator_payload),
        target_kind="comparator_prediction_manifest",
        upstream_replay_evidence=native_binding,
    )
    publish_read_only_json(args.output, wrapper)
    return {
        "native_receipt_sha256": native_document["receipt_sha256"],
        "wrapper_receipt_sha256": wrapper["receipt_sha256"],
    }


def _add_release_context(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quality-receipt", required=True, type=Path)
    parser.add_argument("--final-model-manifest", required=True, type=Path)
    parser.add_argument("--service-configuration-manifest", required=True, type=Path)
    parser.add_argument("--service-source-manifest", required=True, type=Path)
    parser.add_argument("--predecessor-service-source-manifest", required=True, type=Path)
    parser.add_argument("--internal-preopen-unlock", required=True, type=Path)
    parser.add_argument("--full-system-candidate-manifest", required=True, type=Path)


def _add_candidate_stage_bindings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-model-manifest", required=True, type=Path)
    parser.add_argument("--service-configuration-manifest", required=True, type=Path)
    parser.add_argument("--internal-preopen-unlock", required=True, type=Path)


def _add_candidate_execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track", required=True, choices=("model_raw", "full_system"))
    parser.add_argument("--manifest", required=True, type=Path)
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
    parser.add_argument("--calibration-bundle", required=True, type=Path)
    parser.add_argument("--calibration-diagnostics", required=True, type=Path)
    parser.add_argument("--calibration-fit-gold", required=True, type=Path)
    parser.add_argument("--calibration-fit-predictions", required=True, type=Path)
    parser.add_argument("--calibration-fit-generation-receipt", required=True, type=Path)
    parser.add_argument("--calibration-fit-prediction-manifest", required=True, type=Path)
    parser.add_argument("--model-raw-predictions", type=Path)
    parser.add_argument("--model-raw-generation-receipt", type=Path)
    parser.add_argument("--model-raw-prediction-manifest", type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--native-output", required=True, type=Path)


def _validate_candidate_arguments(args: argparse.Namespace) -> None:
    model_raw_values = (
        args.model_raw_predictions,
        args.model_raw_generation_receipt,
        args.model_raw_prediction_manifest,
    )
    if args.track == "full_system" and any(value is None for value in model_raw_values):
        raise CommunityReleaseEvidenceError(
            "full_system replay requires all same-split model_raw inputs"
        )
    if args.track == "model_raw" and any(value is not None for value in model_raw_values):
        raise CommunityReleaseEvidenceError(
            "model_raw replay must not bind recursive model_raw inputs"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("service-source")
    source.add_argument("--final-model-manifest", required=True, type=Path)
    source.add_argument("--service-configuration-manifest", required=True, type=Path)
    source.add_argument("--full-system-candidate-manifest", required=True, type=Path)
    source.add_argument("--predecessor-source-manifest", required=True, type=Path)
    source.add_argument("--output", required=True, type=Path)

    quality_replay = commands.add_parser("quality-replay")
    _add_release_context(quality_replay)
    quality_replay.add_argument("--gold", required=True, type=Path)
    quality_replay.add_argument("--gold-dataset-manifest", required=True, type=Path)
    quality_replay.add_argument("--gold-materialization-receipt", required=True, type=Path)
    quality_replay.add_argument("--gold-freeze-receipt", required=True, type=Path)
    quality_replay.add_argument("--comparator-registry", required=True, type=Path)
    quality_replay.add_argument(
        "--track-input",
        action="append",
        nargs=6,
        required=True,
        metavar=(
            "TRACK",
            "CANDIDATE_PREDICTIONS",
            "CANDIDATE_MANIFEST",
            "COMPARATOR_PREDICTIONS",
            "COMPARATOR_MANIFEST",
            "PERFORMANCE_MANIFEST",
        ),
    )
    quality_replay.add_argument("--native-output", required=True, type=Path)
    quality_replay.add_argument("--output", required=True, type=Path)

    candidate = commands.add_parser("candidate-replay")
    _add_release_context(candidate)
    _add_candidate_execution(candidate)
    candidate.add_argument("--output", required=True, type=Path)

    candidate_native = commands.add_parser("candidate-native-replay")
    _add_candidate_stage_bindings(candidate_native)
    _add_candidate_execution(candidate_native)

    candidate_wrapper = commands.add_parser("candidate-wrapper")
    _add_release_context(candidate_wrapper)
    candidate_wrapper.add_argument("--track", required=True, choices=("model_raw", "full_system"))
    candidate_wrapper.add_argument("--manifest", required=True, type=Path)
    candidate_wrapper.add_argument("--native-replay", required=True, type=Path)
    candidate_wrapper.add_argument("--output", required=True, type=Path)

    comparator = commands.add_parser("comparator-replay")
    _add_release_context(comparator)
    comparator.add_argument("--track", required=True, choices=("model_raw", "full_system"))
    comparator.add_argument("--comparator-manifest", required=True, type=Path)
    comparator.add_argument("--execution-replay", required=True, type=Path)
    comparator.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "service-source":
            _ensure_fresh(args.output, field="service source manifest")
            source = build_service_source_manifest(
                final_model_manifest=args.final_model_manifest,
                service_configuration_manifest=args.service_configuration_manifest,
                full_system_candidate_manifest=args.full_system_candidate_manifest,
                predecessor_source_manifest=args.predecessor_source_manifest,
            )
            publish_read_only_json(args.output, source)
            result = {"manifest_sha256": source["manifest_sha256"], "status": "PASS"}
        elif args.command == "quality-replay":
            result = {**_produce_quality_replay(args), "status": "PASS"}
        elif args.command in {"candidate-replay", "candidate-native-replay"}:
            _validate_candidate_arguments(args)
            if args.command == "candidate-replay":
                result = {**_produce_candidate_replay(args), "status": "PASS"}
            else:
                result = {**_produce_candidate_native_replay(args), "status": "PASS"}
        elif args.command == "candidate-wrapper":
            result = {**_produce_candidate_wrapper(args), "status": "PASS"}
        elif args.command == "comparator-replay":
            result = {**_produce_comparator_replay(args), "status": "PASS"}
        else:  # pragma: no cover - argparse owns the command vocabulary
            raise CommunityReleaseEvidenceError("unsupported evidence command")
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (
        CommunityContractError,
        Open24ComparatorError,
        ReleaseEvalV2PredictionProvenanceError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
