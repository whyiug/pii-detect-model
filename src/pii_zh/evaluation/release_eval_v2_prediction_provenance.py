"""Strict, path-free prediction provenance for frozen release-eval v2.

This module is deliberately separate from :mod:`pii_zh.evaluation.provenance`.
The historical helpers accept only ``manifest_type=evaluation_dataset`` and
must not be weakened for the release-eval-v2 materialization contract.

The builder never runs inference.  It content-addresses an already-produced
prediction file and verifies the complete frozen dataset, selected model and
post-selection calibration chain.  The full-system variant additionally
derives the community service identity from the installed repository sources;
callers cannot supply a service, model, ruleset, route or validator identity.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from pii_zh.calibration import CalibrationBundle
from pii_zh.cascade.community_model import (
    CommunityModelContractError,
    verify_community_model_artifact,
)
from pii_zh.cascade.community_model import (
    canonical_json_hash as model_canonical_json_hash,
)
from pii_zh.cascade.routing import community_full24_routes
from pii_zh.cascade.service_profiles import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    load_service_config,
)
from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
from pii_zh.evaluation.io import EvaluationDataError, load_prediction_jsonl
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.rules.cn_common_v6 import (
    CN_COMMON_RULES_V6,
    CN_COMMON_V6_RULESET_ID,
    CN_COMMON_V6_SOURCE,
)

PredictionTrack = Literal["model_raw", "model_calibrated", "full_system"]
TargetSplit = Literal["calibration", "internal_evaluation"]

SCHEMA_VERSION = "pii-zh.release-eval-v2-prediction-provenance.v1"
MANIFEST_TYPE = "release_eval_v2_prediction_provenance"
SCHEMA_REPOSITORY_PATH = Path(
    "configs/evaluation/release_eval_v2_prediction_provenance_v1.schema.json"
)
CLI_REPOSITORY_PATH = Path("scripts/release_eval_v2_prediction_provenance.py")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_PREDICTION_BYTES = 2 * 1024 * 1024 * 1024

# These are immutable identities of the already-materialized v2 corpus.  They
# are not caller arguments: a newly self-signed manifest/receipt pair cannot
# impersonate the frozen release split.
FROZEN_V2: Mapping[str, Any] = {
    "dataset_id": "pii_zh_synthetic_sota_release_eval_v2",
    "dataset_version": "2.0.0",
    "dataset_manifest_file_sha256": (
        "a09c9f502c2fef64a1db944150b45ab83179df01df8fa9f33a0c009b2a5faa42"
    ),
    "dataset_manifest_sha256": ("bd99b9a858c07fe96f1effcc93d07defa17f7ce39a00712bcfe6b9fda56c19ad"),
    "materialization_receipt_file_sha256": (
        "e3ba0a965bdfcf05aad53a0bdf79868f4b8bb9b0adf4d414d840e68d89904c58"
    ),
    "materialization_receipt_sha256": (
        "342bf8672e211a6dd08078d7d12fd00551ec54710a6d6e921d178039f62110c4"
    ),
    "freeze_receipt_file_sha256": (
        "06ebdce5e288b0e7699a8225e28c4c4f0a1762e76f6967cf990b14384101f0f0"
    ),
    "generation_salt_sha256": ("69ecd313a414459ece55274a7d7ce881537fb8f57e51940068b4a190a6fa437e"),
    "splits": {
        "calibration": {
            "sha256": "cb5cdfeff816f1338e26a471f1407246b9718bc60b059a70bb9d882e1fe97935",
            "bytes": 33_450_063,
            "records": 10_000,
        },
        "internal_evaluation": {
            "sha256": "43704ece40d733b1609093499a1d0b76ab7910ac1ba7ff60d7c3f2359ba6d4cf",
            "bytes": 33_626_033,
            "records": 10_000,
        },
    },
}

_SERVICE_SOURCE_FILES: Mapping[str, str] = {
    "ablation_runtime": "src/pii_zh/cascade/ablation_runtime.py",
    "calibration_filtering": "src/pii_zh/calibration/filtering.py",
    "calibration_thresholds": "src/pii_zh/calibration/thresholds.py",
    "calibration_temperature": "src/pii_zh/calibration/temperature.py",
    "candidate_contract": "src/pii_zh/cascade/result.py",
    "community_model_contract": "src/pii_zh/cascade/community_model.py",
    "community_validators": "src/pii_zh/data/validators/community.py",
    "configuration": "src/pii_zh/cascade/config.py",
    "core_validators": "src/pii_zh/data/validators/__init__.py",
    "model_predictor": "src/pii_zh/inference/transformers_predictor.py",
    "pipeline": "src/pii_zh/cascade/pipeline.py",
    "pipeline_stages": "src/pii_zh/cascade/stages.py",
    "routing": "src/pii_zh/cascade/routing.py",
    "rules_base": "src/pii_zh/rules/cn_common.py",
    "rules_successor": "src/pii_zh/rules/cn_common_v6.py",
    "service_profile": "src/pii_zh/cascade/service_profiles.py",
    "vehicle_plate_validator": "src/pii_zh/data/validators/cn_vehicle_plate.py",
}

GENERATION_RECEIPT_SCHEMA_VERSION = "pii-zh.release-eval-v2-generation-receipt.v1"
GENERATOR_REPOSITORY_PATH = Path("scripts/generate_release_eval_v2_candidate_predictions.py")


class ReleaseEvalV2PredictionProvenanceError(RuntimeError):
    """A fail-closed provenance error that never includes record contents."""


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    """Files proving the frozen post-selection calibration chain."""

    bundle: Path
    diagnostics: Path
    fit_gold: Path
    fit_predictions: Path
    fit_generation_receipt: Path
    fit_prediction_manifest: Path


@dataclass(frozen=True, slots=True)
class ModelRawInputs:
    """Internal-evaluation raw-model output used by a full-system manifest."""

    predictions: Path
    generation_receipt: Path
    prediction_manifest: Path


@dataclass(frozen=True, slots=True)
class AmendmentReplayInputs:
    """Evidence needed to replay the frozen selector and v1-to-v2 amendment."""

    amendment: Path
    protocol: Path
    development_manifest: Path
    v1_release_manifest: Path
    candidate_roots: tuple[Path, Path, Path]


@dataclass(frozen=True, slots=True)
class PredictionProvenanceInputs:
    """Complete input inventory for deterministic build or strict replay."""

    track: PredictionTrack
    target_split: TargetSplit
    predictions: Path
    gold: Path
    dataset_manifest: Path
    materialization_receipt: Path
    freeze_receipt: Path
    model_artifact: Path
    selection_receipt: Path
    generation_receipt: Path
    amendment: AmendmentReplayInputs | None = None
    calibration: CalibrationInputs | None = None
    model_raw: ModelRawInputs | None = None


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    sha256: str
    size_bytes: int
    nonempty_lines: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationRuntime:
    """Bound inference controls emitted by the dedicated generator."""

    document_batch_size: int
    micro_batch_size: int
    max_tokens: int
    stride_fraction: float
    device_class: Literal["cpu", "cuda"]
    dtype: Literal["float32", "bfloat16"]
    visible_cuda_device_count: int
    scope: Literal["open24", "closed8"]
    target_split: Literal["fixture", "calibration", "internal_evaluation"] = "fixture"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json_hash(value: Any, *, ensure_ascii: bool = True) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_no_absolute_paths(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_no_absolute_paths(nested, field=field)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_paths(nested, field=field)
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "~\\", "file:"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} contains a local path")


def _strict_object(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseEvalV2PredictionProvenanceError(
                    f"{field} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} contains a non-finite JSON number")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} must be a JSON object")
    return value


def _read_regular(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
    count_lines: bool = False,
    retain_payload: bool = True,
) -> tuple[bytes, _FileIdentity]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ReleaseEvalV2PredictionProvenanceError(f"{field} must be a bounded regular file")
        digest = hashlib.sha256()
        payload = bytearray()
        total_bytes = 0
        nonempty_lines = 0
        pending = b""
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            total_bytes += len(block)
            if retain_payload:
                payload.extend(block)
            if total_bytes > maximum_bytes:
                raise ReleaseEvalV2PredictionProvenanceError(f"{field} exceeds the size limit")
            if count_lines:
                chunks = (pending + block).split(b"\n")
                pending = chunks.pop()
                nonempty_lines += sum(bool(line.strip()) for line in chunks)
        if count_lines and pending.strip():
            nonempty_lines += 1
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
        if before_identity != after_identity or total_bytes != after.st_size:
            raise ReleaseEvalV2PredictionProvenanceError(f"{field} changed while read")
        identity = _FileIdentity(
            sha256=digest.hexdigest(),
            size_bytes=total_bytes,
            nonempty_lines=nonempty_lines if count_lines else None,
        )
        return bytes(payload), identity
    finally:
        os.close(descriptor)


def _json_file(path: Path, *, field: str) -> tuple[dict[str, Any], _FileIdentity]:
    payload, identity = _read_regular(path, field=field, maximum_bytes=_MAX_JSON_BYTES)
    return _strict_object(payload, field=field), identity


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} must be a lowercase SHA-256")
    return value


def _require_safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ReleaseEvalV2PredictionProvenanceError(
            f"{field} must be a path-free stable identifier"
        )
    return value


def _exact_keys(value: object, *, field: str, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} has an invalid closed shape")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} must be an object")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} must be a positive integer")
    return value


def _verify_self_hash(
    document: Mapping[str, Any],
    *,
    field: str,
    hash_field: str,
    ensure_ascii: bool,
) -> str:
    claimed = _require_digest(document.get(hash_field), field=f"{field}.{hash_field}")
    unsigned = dict(document)
    unsigned.pop(hash_field, None)
    if _canonical_json_hash(unsigned, ensure_ascii=ensure_ascii) != claimed:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} self hash does not verify")
    return claimed


def _split_entries(document: Mapping[str, Any], *, field: str) -> dict[str, Mapping[str, Any]]:
    raw = document.get("files")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ReleaseEvalV2PredictionProvenanceError(f"{field} must bind exactly two splits")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        value = _mapping(item, field=f"{field} split entry")
        split = value.get("split")
        if split not in {"calibration", "internal_evaluation"} or split in result:
            raise ReleaseEvalV2PredictionProvenanceError(f"{field} split inventory is invalid")
        result[str(split)] = value
    return result


def _dataset_identity(inputs: PredictionProvenanceInputs) -> dict[str, Any]:
    expected = FROZEN_V2
    manifest, manifest_file = _json_file(
        inputs.dataset_manifest, field="release-eval-v2 dataset manifest"
    )
    materialization, materialization_file = _json_file(
        inputs.materialization_receipt, field="release-eval-v2 materialization receipt"
    )
    freeze, freeze_file = _json_file(inputs.freeze_receipt, field="release-eval-v2 freeze receipt")
    exact_files = {
        "dataset_manifest_file_sha256": manifest_file.sha256,
        "materialization_receipt_file_sha256": materialization_file.sha256,
        "freeze_receipt_file_sha256": freeze_file.sha256,
    }
    for name, actual in exact_files.items():
        if actual != expected[name]:
            raise ReleaseEvalV2PredictionProvenanceError(f"frozen release-eval-v2 {name} changed")
    manifest_self = _verify_self_hash(
        manifest,
        field="release-eval-v2 dataset manifest",
        hash_field="manifest_sha256",
        ensure_ascii=False,
    )
    materialization_self = _verify_self_hash(
        materialization,
        field="release-eval-v2 materialization receipt",
        hash_field="receipt_sha256",
        ensure_ascii=False,
    )
    if (
        manifest_self != expected["dataset_manifest_sha256"]
        or materialization_self != expected["materialization_receipt_sha256"]
        or manifest.get("dataset_id") != expected["dataset_id"]
        or manifest.get("dataset_version") != expected["dataset_version"]
        or manifest.get("manifest_schema_version") != 2
        or manifest.get("record_data_pool") != "evaluation_only"
        or manifest.get("public_weight_training_allowed") is not False
        or manifest.get("public_artifact_release_allowed") is not True
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "release-eval-v2 dataset identity or isolation policy changed"
        )
    generation = _mapping(manifest.get("generation"), field="dataset generation")
    if (
        generation.get("generation_salt_sha256") != expected["generation_salt_sha256"]
        or generation.get("contains_real_personal_data") is not False
        or generation.get("protected_evaluation_data_read") is not False
        or generation.get("jsonl_rows_printed") is not False
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "release-eval-v2 generation/privacy contract changed"
        )
    audits = _mapping(manifest.get("audits"), field="dataset audits")
    isolation = _mapping(
        audits.get("cross_corpus_isolation"), field="dataset cross-corpus isolation"
    )
    if (
        audits.get("core_label_count") != 24
        or isolation.get("all_collision_counts_zero") is not True
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "release-eval-v2 coverage/isolation audit is invalid"
        )

    manifest_entries = _split_entries(manifest, field="dataset manifest")
    receipt_replacement = _mapping(
        materialization.get("replacement"), field="materialization replacement"
    )
    receipt_entries = _split_entries(receipt_replacement, field="materialization receipt")
    for split, frozen in _mapping(expected["splits"], field="frozen splits").items():
        for source, entry in (
            ("dataset manifest", manifest_entries[split]),
            ("materialization receipt", receipt_entries[split]),
        ):
            if (
                entry.get("split") != split
                or entry.get("name") != f"{split}.jsonl"
                or entry.get("sha256") != frozen["sha256"]
                or entry.get("bytes") != frozen["bytes"]
                or entry.get("records") != frozen["records"]
                or entry.get("mode") != "0444"
            ):
                raise ReleaseEvalV2PredictionProvenanceError(f"{source} {split} binding changed")
    if (
        receipt_replacement.get("dataset_manifest_file_sha256") != manifest_file.sha256
        or receipt_replacement.get("dataset_manifest_sha256") != manifest_self
        or receipt_replacement.get("replacement_dataset_id") != expected["dataset_id"]
        or receipt_replacement.get("replacement_dataset_version") != expected["dataset_version"]
        or receipt_replacement.get("generation_salt_sha256") != expected["generation_salt_sha256"]
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "materialization receipt does not bind the exact dataset manifest"
        )
    freeze_binding = _mapping(
        materialization.get("freeze_receipt"), field="materialization freeze binding"
    )
    manifest_freeze = _mapping(manifest.get("freeze"), field="dataset freeze")
    manifest_freeze_binding = _mapping(
        manifest_freeze.get("supersession_freeze_receipt"),
        field="dataset freeze receipt binding",
    )
    if (
        freeze.get("receipt_id") != "synthetic_sota_release_eval_v2_supersession_freeze"
        or freeze.get("receipt_schema_version") != 1
        or freeze_binding.get("file_sha256") != freeze_file.sha256
        or manifest_freeze_binding.get("file_sha256") != freeze_file.sha256
        or freeze_binding.get("receipt_id") != freeze.get("receipt_id")
        or manifest_freeze_binding.get("receipt_id") != freeze.get("receipt_id")
        or freeze_binding.get("chronology") != freeze.get("chronology")
        or manifest_freeze_binding.get("chronology") != freeze.get("chronology")
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "supersession freeze/materialization cross-binding is invalid"
        )

    split = inputs.target_split
    usage = _mapping(manifest.get("usage_policy"), field="dataset usage policy")
    policy = _mapping(usage.get(split), field=f"dataset {split} policy")
    if (
        policy.get("gradient_training_allowed") is not False
        or policy.get("checkpoint_or_model_selection_allowed") is not False
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "target split permits model training or selection"
        )
    if split == "calibration":
        if (
            policy.get("final_metric_claim_allowed") is not False
            or policy.get("threshold_tuning_allowed") is not True
            or policy.get("content_read_before_cross_seed_selection_allowed") is not False
        ):
            raise ReleaseEvalV2PredictionProvenanceError(
                "calibration split role is not the frozen post-selection role"
            )
    elif (
        policy.get("final_metric_claim_allowed") is not True
        or policy.get("threshold_tuning_allowed") is not False
        or policy.get("temperature_scaling_allowed") is not False
        or policy.get("fusion_calibration_allowed") is not False
        or policy.get("result_informed_model_or_threshold_change_allowed") is not False
        or policy.get("content_read_before_final_freeze_allowed") is not False
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "internal evaluation split is not frozen one-shot evaluation"
        )

    _gold_payload, gold_file = _read_regular(
        inputs.gold,
        field=f"{split} gold",
        maximum_bytes=_MAX_PREDICTION_BYTES,
        count_lines=True,
        retain_payload=False,
    )
    frozen_split = _mapping(expected["splits"], field="frozen splits")[split]
    if (
        gold_file.sha256 != frozen_split["sha256"]
        or gold_file.size_bytes != frozen_split["bytes"]
        or gold_file.nonempty_lines != frozen_split["records"]
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "target gold bytes/count do not match the frozen split"
        )
    return {
        "dataset_id": expected["dataset_id"],
        "dataset_version": expected["dataset_version"],
        "dataset_manifest_file_sha256": manifest_file.sha256,
        "dataset_manifest_sha256": manifest_self,
        "materialization_receipt_file_sha256": materialization_file.sha256,
        "materialization_receipt_sha256": materialization_self,
        "freeze_receipt_file_sha256": freeze_file.sha256,
        "generation_salt_sha256": expected["generation_salt_sha256"],
        "target_split": split,
        "gold_file_sha256": gold_file.sha256,
        "gold_size_bytes": gold_file.size_bytes,
        "gold_document_count": frozen_split["records"],
        "evaluation_only": True,
    }


def _model_identity(model_artifact: Path) -> dict[str, Any]:
    try:
        verified = verify_community_model_artifact(model_artifact)
    except (CommunityModelContractError, OSError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PredictionProvenanceError(
            "final community model artifact contract is invalid"
        ) from exc
    manifest, manifest_file = _json_file(
        verified.root / "training_manifest.json", field="final training manifest"
    )
    config, config_file = _json_file(verified.root / "config.json", field="actual model config")
    manifest_self = _verify_self_hash(
        manifest,
        field="final training manifest",
        hash_field="manifest_sha256",
        ensure_ascii=True,
    )
    if manifest_self != verified.identity.manifest_sha256:
        raise ReleaseEvalV2PredictionProvenanceError(
            "community model verifier and training manifest identity differ"
        )
    output = _mapping(manifest.get("output_artifact"), field="training output artifact")
    files = _mapping(output.get("files"), field="training output files")
    weight_names = output.get("weight_files")
    if not isinstance(weight_names, list) or not weight_names:
        raise ReleaseEvalV2PredictionProvenanceError("training output weight inventory is empty")
    weight_hashes: list[str] = []
    for index, name in enumerate(weight_names):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ReleaseEvalV2PredictionProvenanceError(
                "training output weight inventory is malformed"
            )
        _payload, identity = _read_regular(
            verified.root / name,
            field=f"actual model weight shard {index}",
            maximum_bytes=64 * 1024 * 1024 * 1024,
            retain_payload=False,
        )
        if identity.sha256 != files.get(name):
            raise ReleaseEvalV2PredictionProvenanceError(
                "actual model weight differs from training manifest"
            )
        weight_hashes.append(identity.sha256)
    if config_file.sha256 != files.get("config.json"):
        raise ReleaseEvalV2PredictionProvenanceError(
            "actual model config differs from training manifest"
        )
    label2id = config.get("label2id")
    label_schema_sha256 = _canonical_json_hash(label2id, ensure_ascii=True)
    if (
        label2id != manifest.get("label2id")
        or label_schema_sha256 != manifest.get("label_schema_sha256")
        or label_schema_sha256 != verified.identity.label_schema_sha256
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "actual model label schema differs from final training manifest"
        )
    core = {
        "training_manifest_file_sha256": manifest_file.sha256,
        "training_manifest_sha256": manifest_self,
        "training_manifest_schema_version": manifest.get("schema_version"),
        "training_manifest_type": manifest.get("manifest_type"),
        "seed": manifest.get("seed"),
        "attention_mode": verified.identity.attention_mode,
        "model_type": verified.identity.model_type,
        "architecture_version": verified.identity.architecture_version,
        "taxonomy_version": verified.identity.taxonomy_version,
        "label_count": verified.identity.label_count,
        "label_schema_sha256": label_schema_sha256,
        "config_file_sha256": config_file.sha256,
        "weight_file_count": len(weight_hashes),
        "weight_file_sha256": weight_hashes,
        "weights_combined_sha256": output.get("weights_combined_sha256"),
        "artifact_files_combined_sha256": output.get("artifact_files_combined_sha256"),
        "output_artifact_sha256": _canonical_json_hash(output),
    }
    if (
        isinstance(core["seed"], bool)
        or not isinstance(core["seed"], int)
        or core["training_manifest_schema_version"] != 4
        or core["training_manifest_type"] != "aiguard24_community_training_v1"
        or manifest.get("status") != "completed"
        or manifest.get("fine_tuning") != "lora_merged"
        or _require_digest(core["weights_combined_sha256"], field="training weights combined hash")
        != model_canonical_json_hash(
            {name: digest for name, digest in zip(weight_names, weight_hashes, strict=True)}
        )
        or _require_digest(
            core["artifact_files_combined_sha256"],
            field="training artifact files combined hash",
        )
        != output.get("artifact_files_combined_sha256")
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "final training/model artifact identity is incomplete"
        )
    core["identity_sha256"] = _canonical_json_hash(core)
    return core


def _selection_identity(
    selection_path: Path,
    *,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    freeze_receipt_path: Path,
    amendment_replay: AmendmentReplayInputs | None,
) -> dict[str, Any]:
    """Prove that the supplied artifact is the frozen cross-seed winner.

    The v3 selector was frozen against release-eval v1.  The separately frozen
    v2 supersession explicitly preserves that selector while withdrawing v1;
    both immutable receipts are therefore required and cross-checked here.
    """

    selection, selection_file = _json_file(selection_path, field="v3 cross-seed selection receipt")
    selection_self = _verify_self_hash(
        selection,
        field="v3 cross-seed selection receipt",
        hash_field="selection_receipt_sha256",
        ensure_ascii=True,
    )
    selected = _mapping(selection.get("selected"), field="selection selected candidate")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ReleaseEvalV2PredictionProvenanceError(
            "selection receipt must contain all three frozen candidates"
        )
    matching = [
        _mapping(item, field="selection candidate")
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("training_manifest_sha256") == selected.get("training_manifest_sha256")
    ]
    protocol = _mapping(selection.get("protocol"), field="selection protocol")
    post = _mapping(
        selection.get("post_selection_evaluation"), field="selection post-selection evaluation"
    )
    if (
        selection.get("schema_version") != 1
        or selection.get("receipt_type") != "aiguard24_v3_cross_seed_calibration_selection_v1"
        or selection.get("status") != "SELECTED_FOR_CALIBRATION_NOT_RELEASE"
        or selection.get("release_eligible") is not False
        or selection.get("candidate_count") != 3
        or protocol.get("protocol_id") != "aiguard24_synthetic_sota_v3_resplit_dev_selection"
        or protocol.get("file_sha256")
        != "9f330eca4dbc6e522fc52b06f6bfb7c4047e455a0b3be555a94a03e6dd4ca51e"
        or selected.get("training_manifest_sha256") != model["training_manifest_sha256"]
        or selected.get("seed") != model["seed"]
        or selected.get("weights_combined_sha256") != model["weights_combined_sha256"]
        or selected.get("output_artifact_sha256") != model["output_artifact_sha256"]
        or len(matching) != 1
        or matching[0].get("training_manifest_file_sha256")
        != model["training_manifest_file_sha256"]
        or matching[0].get("label_schema_sha256") != model["label_schema_sha256"]
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "model artifact is not the exact frozen cross-seed selection"
        )
    # The immutable selector still names v1.  Accept it only together with the
    # exact v2 freeze that withdraws that same v1 identity without changing the
    # selector, then names the exact v2 replacement.
    freeze, freeze_file = _json_file(freeze_receipt_path, field="v2 supersession freeze receipt")
    supersession = _mapping(freeze.get("supersession"), field="v2 supersession")
    effect = _mapping(freeze.get("protocol_effect"), field="v2 protocol effect")
    if (
        freeze_file.sha256 != FROZEN_V2["freeze_receipt_file_sha256"]
        or post.get("dataset_id") != "pii_zh_synthetic_sota_release_eval_v1"
        or post.get("dataset_version") != "1.0.0"
        or post.get("file_sha256")
        != "aa4817080097681688948104e867ebb510be463fb87d23f9f9e3f786f58dfa7b"
        or post.get("manifest_sha256")
        != "f0500a5c0672b5285c8d95782d9491b6b5dd3df6a14977e6ade3f53bbd7130ca"
        or supersession.get("withdrawn_dataset_id") != post.get("dataset_id")
        or supersession.get("withdrawn_dataset_version") != post.get("dataset_version")
        or supersession.get("replacement_dataset_id") != FROZEN_V2["dataset_id"]
        or supersession.get("replacement_dataset_version") != FROZEN_V2["dataset_version"]
        or effect.get("does_not_modify_frozen_v3_training_protocol_or_selector") is not True
        or effect.get("v1_may_not_be_used_for_zero-read_confirmatory_calibration") is not True
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "v3 selection and v1-to-v2 supersession do not form a closed chronology"
        )
    if amendment_replay is None:
        raise ReleaseEvalV2PredictionProvenanceError(
            "prediction provenance requires strict v1-to-v2 amendment replay"
        )
    try:
        from scripts.build_aiguard24_v3_release_eval_v2_amendment import (
            replay_amendment_from_selection_evidence,
        )

        amendment, amendment_file_sha256 = replay_amendment_from_selection_evidence(
            amendment_replay.amendment,
            protocol_path=amendment_replay.protocol,
            development_manifest_path=amendment_replay.development_manifest,
            v1_release_manifest_path=amendment_replay.v1_release_manifest,
            candidate_roots=amendment_replay.candidate_roots,
            selection_receipt_path=selection_path,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PredictionProvenanceError(
            "frozen selector/amendment strict replay failed"
        ) from exc
    amendment_self = _verify_self_hash(
        amendment,
        field="v1-to-v2 amendment",
        hash_field="amendment_sha256",
        ensure_ascii=False,
    )
    replay = _mapping(amendment.get("selection_replay"), field="amendment selection replay")
    replacement = _mapping(amendment.get("replacement_v2"), field="amendment v2 replacement")
    amendment_selected = _mapping(replay.get("selected"), field="amendment selected model")
    if (
        replay.get("receipt_file_sha256") != selection_file.sha256
        or replay.get("receipt_sha256") != selection_self
        or amendment_selected.get("training_manifest_sha256")
        != model["training_manifest_sha256"]
        or replacement.get("dataset_id") != dataset["dataset_id"]
        or replacement.get("dataset_version") != dataset["dataset_version"]
        or replacement.get("manifest_file_sha256")
        != dataset["dataset_manifest_file_sha256"]
        or replacement.get("manifest_sha256") != dataset["dataset_manifest_sha256"]
        or replacement.get("calibration_sha256")
        != dataset["calibration_gold_file_sha256"]
        or amendment.get("status") != "V2_CALIBRATION_NEXT_NOT_RELEASE"
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "strict amendment replay does not bind this model/dataset"
        )
    return {
        "selection_receipt_file_sha256": selection_file.sha256,
        "selection_receipt_sha256": selection_self,
        "status": selection["status"],
        "selected_seed": model["seed"],
        "selected_training_manifest_sha256": model["training_manifest_sha256"],
        "selected_training_manifest_file_sha256": model["training_manifest_file_sha256"],
        "selected_weights_combined_sha256": model["weights_combined_sha256"],
        "selected_output_artifact_sha256": model["output_artifact_sha256"],
        "supersession_freeze_file_sha256": freeze_file.sha256,
        "supersession": "release_eval_v1_withdrawn_and_replaced_by_v2",
        "amendment_file_sha256": amendment_file_sha256,
        "amendment_sha256": amendment_self,
        "amendment_status": amendment["status"],
        "amendment_strict_replay": True,
    }


def _prediction_identity(path: Path, *, track: PredictionTrack) -> dict[str, Any]:
    payload, identity = _read_regular(
        path,
        field=f"{track} predictions",
        maximum_bytes=_MAX_PREDICTION_BYTES,
        count_lines=True,
    )
    try:
        records = load_prediction_jsonl(StringIO(payload.decode("utf-8")))
    except (EvaluationDataError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ReleaseEvalV2PredictionProvenanceError(
            f"{track} predictions violate the raw-text-free JSONL contract"
        ) from exc
    if track == "model_raw" and any(
        span.score is None for record in records for span in record.spans
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "model_raw prediction spans must retain raw scores"
        )
    if len(records) != identity.nonempty_lines:
        raise ReleaseEvalV2PredictionProvenanceError(
            f"{track} prediction line count is inconsistent"
        )
    return {
        "file_sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
        "document_count": identity.nonempty_lines,
        "raw_text_free_contract": True,
    }


def generation_implementation_identity(track: str) -> dict[str, Any]:
    """Derive the actual generator and inference source identity from files."""

    if track not in {"model_raw", "model_calibrated", "full_system"}:
        raise ReleaseEvalV2PredictionProvenanceError("unsupported generation track")
    root = _repository_root()
    common = {
        "generator": str(GENERATOR_REPOSITORY_PATH),
        "model_loader": "src/pii_zh/inference/transformers_predictor.py",
        "bio_decoder": "src/pii_zh/models/decoding.py",
        "recognizer": "src/pii_zh/presidio/qwen_recognizer.py",
        "token_chunker": "src/pii_zh/presidio/token_chunker.py",
    }
    if track != "model_raw":
        common.update(
            {
                "community_builder": "src/pii_zh/cascade/service_profiles.py",
                "pipeline": "src/pii_zh/cascade/pipeline.py",
                "routing": "src/pii_zh/cascade/routing.py",
                "rules": "src/pii_zh/rules/cn_common_v6.py",
                "validators": "src/pii_zh/data/validators/community.py",
            }
        )
    sources: dict[str, str] = {}
    for logical, relative in sorted(common.items()):
        _payload, identity = _read_regular(
            root / relative,
            field=f"generation source {logical}",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        )
        sources[logical] = identity.sha256
    return {
        "generator_id": "release_eval_v2_candidate_generator_v1",
        "factory_id": (
            "pii_zh.inference.load_local_predictor+QwenPiiRecognizer"
            if track == "model_raw"
            else "pii_zh.cascade.build_community_model_service_pipeline"
        ),
        "source_file_sha256": sources,
        "implementation_sha256": _canonical_json_hash(sources),
    }


def build_release_eval_v2_generation_receipt(
    *,
    track: Literal["model_raw", "model_calibrated", "full_system"],
    input_path: Path,
    predictions_path: Path,
    model_artifact: Path,
    calibration_bundle: Path | None,
    runtime: GenerationRuntime,
    stage_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the aggregate receipt emitted by the dedicated inference CLI."""

    input_payload, input_file = _read_regular(
        input_path,
        field="generation canonical input",
        maximum_bytes=_MAX_PREDICTION_BYTES,
        count_lines=True,
        retain_payload=False,
    )
    del input_payload
    prediction = _prediction_identity(predictions_path, track=track)
    if prediction["document_count"] != input_file.nonempty_lines:
        raise ReleaseEvalV2PredictionProvenanceError(
            "generation input/output document counts differ"
        )
    model = _model_identity(model_artifact)
    if (
        isinstance(runtime.document_batch_size, bool)
        or runtime.document_batch_size < 1
        or isinstance(runtime.micro_batch_size, bool)
        or runtime.micro_batch_size < 1
        or runtime.max_tokens != 512
        or not math.isclose(runtime.stride_fraction, 0.25, rel_tol=0.0, abs_tol=0.0)
        or runtime.visible_cuda_device_count < 0
        or (runtime.device_class == "cuda" and runtime.visible_cuda_device_count != 1)
        or (runtime.device_class == "cpu" and runtime.visible_cuda_device_count != 0)
        or (runtime.device_class == "cuda" and runtime.dtype != "bfloat16")
        or (runtime.device_class == "cpu" and runtime.dtype != "float32")
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "generation runtime is outside the frozen local/offline bounds"
        )
    if track == "model_raw":
        if calibration_bundle is not None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "model_raw generation must be uncalibrated"
            )
        calibration_sha256 = None
        mode = "model_raw"
    else:
        if calibration_bundle is None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "community model-only/cascade generation requires calibration"
            )
        _bundle, calibration_file = _json_file(
            calibration_bundle, field="generation calibration bundle"
        )
        calibration_sha256 = calibration_file.sha256
        mode = "model-only" if track == "model_calibrated" else "cascade"
    if runtime.target_split == "fixture":
        if stage_gate is not None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "fixture generation must not carry a formal stage gate"
            )
        stage = {
            "target_split": "fixture",
            "gate_applied_before_input_open": False,
            "gate_status": None,
            "gate_file_sha256": None,
            "gate_sha256": None,
        }
    else:
        if not isinstance(stage_gate, Mapping):
            raise ReleaseEvalV2PredictionProvenanceError(
                "formal generation requires a pre-open stage gate"
            )
        expected_status = (
            "CALIBRATION_PREOPEN_PASS"
            if runtime.target_split == "calibration"
            else "INTERNAL_PREOPEN_PASS"
        )
        file_key = (
            "authorization_file_sha256"
            if runtime.target_split == "calibration"
            else "unlock_file_sha256"
        )
        self_key = (
            "authorization_sha256"
            if runtime.target_split == "calibration"
            else "unlock_sha256"
        )
        if (
            set(stage_gate)
            != {"status", file_key, self_key, "expected_input_sha256"}
            or stage_gate.get("status") != expected_status
            or stage_gate.get("expected_input_sha256") != input_file.sha256
        ):
            raise ReleaseEvalV2PredictionProvenanceError(
                "formal generation stage gate does not bind the input snapshot"
            )
        stage = {
            "target_split": runtime.target_split,
            "gate_applied_before_input_open": True,
            "gate_status": expected_status,
            "gate_file_sha256": _require_digest(
                stage_gate.get(file_key), field="generation stage gate file hash"
            ),
            "gate_sha256": _require_digest(
                stage_gate.get(self_key), field="generation stage gate self hash"
            ),
        }
    receipt: dict[str, Any] = {
        "schema_version": GENERATION_RECEIPT_SCHEMA_VERSION,
        "receipt_type": "release_eval_v2_candidate_generation",
        "track": track,
        "mode": mode,
        "scope": {
            "scope_id": runtime.scope,
            "ordered_labels": (
                list(PII_CORE_LABELS)
                if runtime.scope == "open24"
                else [
                    "ADDRESS",
                    "BANK_CARD_NUMBER",
                    "CN_RESIDENT_ID",
                    "EMAIL_ADDRESS",
                    "PASSPORT_NUMBER",
                    "PERSON_NAME",
                    "PHONE_NUMBER",
                    "VEHICLE_LICENSE_PLATE",
                ]
            ),
        },
        "input": {
            "file_sha256": input_file.sha256,
            "size_bytes": input_file.size_bytes,
            "document_count": input_file.nonempty_lines,
        },
        "output": dict(prediction),
        "model": {
            "training_manifest_sha256": model["training_manifest_sha256"],
            "training_manifest_file_sha256": model["training_manifest_file_sha256"],
            "model_identity_sha256": model["identity_sha256"],
        },
        "calibration": {
            "applied": track != "model_raw",
            "bundle_file_sha256": calibration_sha256,
        },
        "runtime": {
            "document_batch_size": runtime.document_batch_size,
            "micro_batch_size": runtime.micro_batch_size,
            "max_tokens": runtime.max_tokens,
            "stride_fraction": runtime.stride_fraction,
            "device_class": runtime.device_class,
            "dtype": runtime.dtype,
            "visible_cuda_device_count": runtime.visible_cuda_device_count,
            "local_files_only": True,
            "offline_environment": True,
            "remote_code_allowed": False,
        },
        "stage_gate": stage,
        "generator": generation_implementation_identity(track),
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "aggregate_only": True,
        },
    }
    _assert_no_absolute_paths(receipt, field="generation receipt")
    receipt["receipt_sha256"] = _canonical_json_hash(receipt)
    return receipt


def _generation_identity(
    receipt_path: Path,
    *,
    track: PredictionTrack,
    dataset: Mapping[str, Any],
    prediction: Mapping[str, Any],
    model: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    receipt, receipt_file = _json_file(receipt_path, field="generation receipt")
    receipt_self = _verify_self_hash(
        receipt,
        field="generation receipt",
        hash_field="receipt_sha256",
        ensure_ascii=True,
    )
    _assert_no_absolute_paths(receipt, field="generation receipt")
    _exact_keys(
        receipt,
        field="generation receipt",
        expected={
            "schema_version",
            "receipt_type",
            "track",
            "mode",
            "scope",
            "input",
            "output",
            "model",
            "calibration",
            "runtime",
            "stage_gate",
            "generator",
            "privacy",
            "receipt_sha256",
        },
    )
    input_identity = _mapping(receipt.get("input"), field="generation input")
    output_identity = _mapping(receipt.get("output"), field="generation output")
    model_identity = _mapping(receipt.get("model"), field="generation model")
    calibration_identity = _mapping(receipt.get("calibration"), field="generation calibration")
    runtime = _mapping(receipt.get("runtime"), field="generation runtime")
    scope = _mapping(receipt.get("scope"), field="generation scope")
    privacy = _mapping(receipt.get("privacy"), field="generation privacy")
    stage_gate = _mapping(receipt.get("stage_gate"), field="generation stage gate")
    _exact_keys(
        scope,
        field="generation scope",
        expected={"scope_id", "ordered_labels"},
    )
    _exact_keys(
        input_identity,
        field="generation input",
        expected={"file_sha256", "size_bytes", "document_count"},
    )
    _exact_keys(
        output_identity,
        field="generation output",
        expected={"file_sha256", "size_bytes", "document_count", "raw_text_free_contract"},
    )
    _exact_keys(
        model_identity,
        field="generation model",
        expected={
            "training_manifest_sha256",
            "training_manifest_file_sha256",
            "model_identity_sha256",
        },
    )
    _exact_keys(
        calibration_identity,
        field="generation calibration",
        expected={"applied", "bundle_file_sha256"},
    )
    _exact_keys(
        runtime,
        field="generation runtime",
        expected={
            "document_batch_size",
            "micro_batch_size",
            "max_tokens",
            "stride_fraction",
            "device_class",
            "dtype",
            "visible_cuda_device_count",
            "local_files_only",
            "offline_environment",
            "remote_code_allowed",
        },
    )
    _exact_keys(
        stage_gate,
        field="generation stage gate",
        expected={
            "target_split",
            "gate_applied_before_input_open",
            "gate_status",
            "gate_file_sha256",
            "gate_sha256",
        },
    )
    _exact_keys(
        privacy,
        field="generation privacy",
        expected={
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
            "aggregate_only",
        },
    )
    expected_generator = generation_implementation_identity(track)
    expected_calibration = None if calibration is None else calibration["bundle_file_sha256"]
    if (
        receipt.get("schema_version") != GENERATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("receipt_type") != "release_eval_v2_candidate_generation"
        or receipt.get("track") != track
        or receipt.get("mode")
        != (
            "model_raw"
            if track == "model_raw"
            else "model-only"
            if track == "model_calibrated"
            else "cascade"
        )
        or scope.get("scope_id") != "open24"
        or scope.get("ordered_labels") != list(PII_CORE_LABELS)
        or input_identity.get("file_sha256") != dataset["gold_file_sha256"]
        or input_identity.get("size_bytes") != dataset["gold_size_bytes"]
        or input_identity.get("document_count") != dataset["gold_document_count"]
        or output_identity != prediction
        or model_identity.get("training_manifest_sha256") != model["training_manifest_sha256"]
        or model_identity.get("training_manifest_file_sha256")
        != model["training_manifest_file_sha256"]
        or model_identity.get("model_identity_sha256") != model["identity_sha256"]
        or calibration_identity.get("applied") is not (track != "model_raw")
        or calibration_identity.get("bundle_file_sha256")
        != (expected_calibration if track != "model_raw" else None)
        or runtime.get("max_tokens") != 512
        or runtime.get("stride_fraction") != 0.25
        or runtime.get("local_files_only") is not True
        or runtime.get("offline_environment") is not True
        or runtime.get("remote_code_allowed") is not False
        or _positive_int(runtime.get("document_batch_size"), field="generation document batch size")
        < 1
        or _positive_int(runtime.get("micro_batch_size"), field="generation micro batch size") < 1
        or receipt.get("generator") != expected_generator
        or stage_gate.get("target_split") != dataset["target_split"]
        or stage_gate.get("gate_applied_before_input_open") is not True
        or stage_gate.get("gate_status")
        != (
            "CALIBRATION_PREOPEN_PASS"
            if dataset["target_split"] == "calibration"
            else "INTERNAL_PREOPEN_PASS"
        )
        or _require_digest(
            stage_gate.get("gate_file_sha256"), field="generation stage gate file hash"
        )
        != stage_gate.get("gate_file_sha256")
        or _require_digest(
            stage_gate.get("gate_sha256"), field="generation stage gate self hash"
        )
        != stage_gate.get("gate_sha256")
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
        raise ReleaseEvalV2PredictionProvenanceError(
            "generation receipt is not bound to the actual frozen inference path"
        )
    device = runtime.get("device_class")
    visible = runtime.get("visible_cuda_device_count")
    dtype = runtime.get("dtype")
    if (
        device not in {"cpu", "cuda"}
        or isinstance(visible, bool)
        or not isinstance(visible, int)
        or (device, visible, dtype) not in {("cpu", 0, "float32"), ("cuda", 1, "bfloat16")}
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "generation receipt violates single-device runtime isolation"
        )
    return {
        "receipt_file_sha256": receipt_file.sha256,
        "receipt_sha256": receipt_self,
        "generator_implementation_sha256": expected_generator["implementation_sha256"],
        "mode": receipt["mode"],
        "scope_id": scope["scope_id"],
        "runtime_configuration_sha256": _canonical_json_hash(runtime),
        "stage_gate_target_split": stage_gate["target_split"],
        "stage_gate_file_sha256": stage_gate["gate_file_sha256"],
        "stage_gate_sha256": stage_gate["gate_sha256"],
    }


def _calibration_identity(
    calibration: CalibrationInputs,
    *,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    base_inputs: PredictionProvenanceInputs,
) -> dict[str, Any]:
    if base_inputs.target_split != "internal_evaluation":
        raise ReleaseEvalV2PredictionProvenanceError(
            "calibration bundle is valid only for internal evaluation targets"
        )
    fit_inputs = PredictionProvenanceInputs(
        track="model_raw",
        target_split="calibration",
        predictions=calibration.fit_predictions,
        gold=calibration.fit_gold,
        dataset_manifest=base_inputs.dataset_manifest,
        materialization_receipt=base_inputs.materialization_receipt,
        freeze_receipt=base_inputs.freeze_receipt,
        model_artifact=base_inputs.model_artifact,
        selection_receipt=base_inputs.selection_receipt,
        generation_receipt=calibration.fit_generation_receipt,
        amendment=base_inputs.amendment,
    )
    expected_fit = build_release_eval_v2_prediction_manifest(fit_inputs)
    observed_fit, fit_manifest_file = _json_file(
        calibration.fit_prediction_manifest,
        field="calibration fit prediction manifest",
    )
    if observed_fit != expected_fit:
        raise ReleaseEvalV2PredictionProvenanceError(
            "calibration fit prediction manifest does not strictly replay"
        )
    bundle_document, bundle_file = _json_file(calibration.bundle, field="calibration bundle")
    try:
        bundle = CalibrationBundle.from_dict(bundle_document)
    except (TypeError, ValueError) as exc:
        raise ReleaseEvalV2PredictionProvenanceError("calibration bundle is invalid") from exc
    if (
        bundle.calibration_version is None
        or bundle.model_version != model["training_manifest_sha256"]
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "calibration bundle does not bind the final training manifest"
        )
    diagnostics, diagnostics_file = _json_file(
        calibration.diagnostics, field="calibration diagnostics"
    )
    diagnostics_self = _verify_self_hash(
        diagnostics,
        field="calibration diagnostics",
        hash_field="manifest_sha256",
        ensure_ascii=True,
    )
    required = {
        "schema_version",
        "manifest_type",
        "calibration_version",
        "calibration_bundle_sha256",
        "taxonomy_version",
        "inputs",
        "dataset_contract",
        "parameters",
        "temperature",
        "confidence_calibration",
        "per_label",
        "privacy",
        "manifest_sha256",
    }
    _exact_keys(diagnostics, field="calibration diagnostics", expected=required)
    diag_inputs = _mapping(diagnostics.get("inputs"), field="calibration diagnostics inputs")
    contract = _mapping(diagnostics.get("dataset_contract"), field="calibration dataset contract")
    privacy = _mapping(diagnostics.get("privacy"), field="calibration diagnostics privacy")
    fit_prediction = expected_fit["prediction"]
    if (
        diagnostics.get("schema_version") != 1
        or diagnostics.get("manifest_type") != "calibration_diagnostics"
        or diagnostics.get("calibration_version") != bundle.calibration_version
        or diagnostics.get("calibration_bundle_sha256") != bundle_file.sha256
        or diagnostics.get("taxonomy_version") != model["taxonomy_version"]
        or diag_inputs.get("gold_sha256") != dataset["calibration_gold_file_sha256"]
        or diag_inputs.get("predictions_sha256") != fit_prediction["file_sha256"]
        or diag_inputs.get("document_count") != dataset["calibration_document_count"]
        or contract.get("dataset_id") != dataset["dataset_id"]
        or contract.get("dataset_version") != dataset["dataset_version"]
        or contract.get("gold_sha256") != dataset["calibration_gold_file_sha256"]
        or contract.get("manifest_sha256") != dataset["dataset_manifest_sha256"]
        or contract.get("manifest_file_sha256") != dataset["dataset_manifest_file_sha256"]
        or contract.get("supersession_freeze_file_sha256") != dataset["freeze_receipt_file_sha256"]
        or contract.get("records") != dataset["calibration_document_count"]
        or contract.get("role") != "post_cross_seed_selection_calibration_only"
        or any(
            privacy.get(name) is not False
            for name in (
                "contains_paths",
                "contains_raw_text",
                "contains_entity_values",
                "contains_document_ids",
            )
        )
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "calibration bundle/diagnostics/fit input cross-binding is invalid"
        )
    return {
        "bundle_file_sha256": bundle_file.sha256,
        "diagnostics_file_sha256": diagnostics_file.sha256,
        "diagnostics_manifest_sha256": diagnostics_self,
        "calibration_version": bundle.calibration_version,
        "model_training_manifest_sha256": model["training_manifest_sha256"],
        "fit_gold_file_sha256": dataset["calibration_gold_file_sha256"],
        "fit_predictions_file_sha256": fit_prediction["file_sha256"],
        "fit_document_count": dataset["calibration_document_count"],
        "fit_prediction_manifest_file_sha256": fit_manifest_file.sha256,
        "fit_prediction_manifest_sha256": expected_fit["manifest_sha256"],
    }


def _route_payload(route: Any) -> dict[str, Any]:
    return asdict(route)


def _source_hashes(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for logical_id, relative in _SERVICE_SOURCE_FILES.items():
        _payload, identity = _read_regular(
            repository_root / relative,
            field=f"service implementation source {logical_id}",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        )
        result[logical_id] = identity.sha256
    return dict(sorted(result.items()))


def _validator_identity(repository_root: Path) -> dict[str, Any]:
    validators: dict[str, Any] = {}
    for label, validator in sorted(COMMUNITY_STRUCTURED_VALIDATORS.items()):
        source = inspect.getsourcefile(validator)
        if source is None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "community validator lacks inspectable source"
            )
        source_path = Path(source).resolve(strict=True)
        try:
            source_path.relative_to(repository_root)
        except ValueError as exc:
            raise ReleaseEvalV2PredictionProvenanceError(
                "community validator source is outside the repository"
            ) from exc
        _payload, identity = _read_regular(
            source_path,
            field="community validator implementation",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        )
        validators[label] = {
            "callable_id": _require_safe_id(
                f"{validator.__module__}.{validator.__qualname__}",
                field="validator callable ID",
            ),
            "source_file_sha256": identity.sha256,
        }
    return {
        "validator_count": len(validators),
        "validators": validators,
        "configuration_sha256": _canonical_json_hash(validators),
    }


def _rules_identity(repository_root: Path) -> dict[str, Any]:
    definitions = [
        {
            "entity_type": item.entity_type,
            "pattern_id": item.pattern_id,
            "pattern": item.pattern.pattern,
            "flags": item.pattern.flags,
            "score": item.score,
            "validator_label": item.validator_label,
            "required_context": list(item.required_context),
            "already_masked": item.already_masked,
            "value_group": item.value_group,
        }
        for item in CN_COMMON_RULES_V6
    ]
    sources: dict[str, str] = {}
    for logical, relative in {
        "base": "src/pii_zh/rules/cn_common.py",
        "successor": "src/pii_zh/rules/cn_common_v6.py",
        "plate_validator": "src/pii_zh/data/validators/cn_vehicle_plate.py",
    }.items():
        _payload, identity = _read_regular(
            repository_root / relative,
            field=f"rules source {logical}",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        )
        sources[logical] = identity.sha256
    return {
        "ruleset_id": CN_COMMON_V6_RULESET_ID,
        "rule_source": CN_COMMON_V6_SOURCE,
        "rule_count": len(definitions),
        "definitions_sha256": _canonical_json_hash(definitions),
        "source_file_sha256": sources,
        "implementation_sha256": _canonical_json_hash(
            {"definitions": definitions, "sources": sources}
        ),
    }


def _service_identity(
    *,
    model: Mapping[str, Any],
    calibration: Mapping[str, Any],
    mode: Literal["model-only", "cascade"] = "cascade",
) -> dict[str, Any]:
    repository_root = _repository_root()
    config = load_service_config(
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode=mode,
    )
    config = replace(
        config,
        routes=community_full24_routes(recognizer_thresholds_authoritative=True),
    )
    routes = [_route_payload(route) for route in config.routes]
    validators = _validator_identity(repository_root)
    required_validator_labels = {
        route.validator_label
        for route in config.routes
        if route.category == "structured" and route.model_enabled
    }
    if None in required_validator_labels or not required_validator_labels <= set(
        validators["validators"]
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "community service routes are not closed by actual validators"
        )
    config_payload = {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
        "routes": routes,
        "calibration_bundle_file_sha256": calibration["bundle_file_sha256"],
        "model_identity_sha256": model["identity_sha256"],
    }
    sources = _source_hashes(repository_root)
    rules = _rules_identity(repository_root)
    return {
        "profile_id": config.profile_version,
        "mode": config.mode,
        "configuration_sha256": _canonical_json_hash(config_payload),
        "routes_sha256": _canonical_json_hash(routes),
        "route_count": len(routes),
        "validators": validators,
        "rules": rules,
        "rules_enabled": mode == "cascade",
        "implementation_source_sha256": sources,
        "implementation_sha256": _canonical_json_hash(sources),
        "model_identity_sha256": model["identity_sha256"],
        "calibration_bundle_file_sha256": calibration["bundle_file_sha256"],
        "calibration_diagnostics_manifest_sha256": calibration["diagnostics_manifest_sha256"],
    }


def _implementation_identity() -> dict[str, str]:
    root = _repository_root()
    identities: dict[str, str] = {}
    for name, path in {
        "module_file_sha256": Path(__file__).resolve(),
        "schema_file_sha256": root / SCHEMA_REPOSITORY_PATH,
        "cli_file_sha256": root / CLI_REPOSITORY_PATH,
    }.items():
        _payload, identity = _read_regular(
            path,
            field=f"provenance {name}",
            maximum_bytes=_MAX_JSON_BYTES,
            retain_payload=False,
        )
        identities[name] = identity.sha256
    identities["implementation_sha256"] = _canonical_json_hash(identities)
    return identities


def _dataset_with_calibration_split(
    dataset: dict[str, Any], inputs: PredictionProvenanceInputs
) -> dict[str, Any]:
    """Add the frozen calibration split identity without changing target identity."""

    expected_split = _mapping(FROZEN_V2["splits"], field="frozen splits")["calibration"]
    if inputs.target_split == "calibration":
        calibration_file_sha = dataset["gold_file_sha256"]
        calibration_count = dataset["gold_document_count"]
    else:
        calibration_gold = replace(
            inputs,
            target_split="calibration",
            gold=(inputs.calibration.fit_gold if inputs.calibration is not None else inputs.gold),
        )
        calibration_dataset = _dataset_identity(calibration_gold)
        calibration_file_sha = calibration_dataset["gold_file_sha256"]
        calibration_count = calibration_dataset["gold_document_count"]
    if (
        calibration_file_sha != expected_split["sha256"]
        or calibration_count != expected_split["records"]
    ):
        raise ReleaseEvalV2PredictionProvenanceError("frozen calibration split identity changed")
    result = dict(dataset)
    result["calibration_gold_file_sha256"] = calibration_file_sha
    result["calibration_document_count"] = calibration_count
    return result


def build_release_eval_v2_prediction_manifest(
    inputs: PredictionProvenanceInputs,
) -> dict[str, Any]:
    """Build one deterministic model-raw or full-system provenance manifest."""

    if inputs.track not in {"model_raw", "model_calibrated", "full_system"}:
        raise ReleaseEvalV2PredictionProvenanceError("unsupported prediction track")
    if inputs.target_split not in {"calibration", "internal_evaluation"}:
        raise ReleaseEvalV2PredictionProvenanceError("unsupported release-eval-v2 split")
    if inputs.target_split == "calibration":
        if inputs.track != "model_raw" or inputs.calibration is not None or inputs.model_raw:
            raise ReleaseEvalV2PredictionProvenanceError(
                "calibration split accepts only uncalibrated model_raw provenance"
            )
    else:
        if inputs.calibration is None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "internal evaluation requires a frozen calibration chain"
            )
        if inputs.track in {"model_calibrated", "full_system"} and inputs.model_raw is None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "calibrated/system tracks require the same-split model_raw manifest"
            )
        if inputs.track == "model_raw" and inputs.model_raw is not None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "model_raw must not recursively bind another model_raw manifest"
            )

    dataset = _dataset_identity(inputs)
    dataset = _dataset_with_calibration_split(dataset, inputs)
    model = _model_identity(inputs.model_artifact)
    selection = _selection_identity(
        inputs.selection_receipt,
        model=model,
        dataset=dataset,
        freeze_receipt_path=inputs.freeze_receipt,
        amendment_replay=inputs.amendment,
    )
    prediction = _prediction_identity(inputs.predictions, track=inputs.track)
    if prediction["document_count"] != dataset["gold_document_count"]:
        raise ReleaseEvalV2PredictionProvenanceError(
            "prediction document count does not match target gold"
        )
    calibration_identity = (
        None
        if inputs.calibration is None
        else _calibration_identity(
            inputs.calibration,
            dataset=dataset,
            model=model,
            base_inputs=inputs,
        )
    )
    service: dict[str, Any] | None = None
    upstream_model_raw: dict[str, Any] | None = None
    if inputs.track in {"model_calibrated", "full_system"}:
        assert calibration_identity is not None
        assert inputs.model_raw is not None
        raw_inputs = replace(
            inputs,
            track="model_raw",
            predictions=inputs.model_raw.predictions,
            generation_receipt=inputs.model_raw.generation_receipt,
            model_raw=None,
        )
        expected_raw = build_release_eval_v2_prediction_manifest(raw_inputs)
        observed_raw, raw_manifest_file = _json_file(
            inputs.model_raw.prediction_manifest,
            field="same-split model_raw prediction manifest",
        )
        if observed_raw != expected_raw:
            raise ReleaseEvalV2PredictionProvenanceError(
                "same-split model_raw prediction manifest does not strictly replay"
            )
        if expected_raw["calibration"] != calibration_identity:
            raise ReleaseEvalV2PredictionProvenanceError(
                "full_system and model_raw do not bind the same calibration"
            )
        upstream_model_raw = {
            "manifest_file_sha256": raw_manifest_file.sha256,
            "manifest_sha256": expected_raw["manifest_sha256"],
            "predictions_file_sha256": expected_raw["prediction"]["file_sha256"],
        }
        service = _service_identity(
            model=model,
            calibration=calibration_identity,
            mode="model-only" if inputs.track == "model_calibrated" else "cascade",
        )

    generation = _generation_identity(
        inputs.generation_receipt,
        track=inputs.track,
        dataset=dataset,
        prediction=prediction,
        model=model,
        calibration=calibration_identity,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "role": "project_candidate",
        "track": inputs.track,
        "prediction_semantics": (
            "raw_model_scores_pre_service"
            if inputs.track == "model_raw"
            else "community_model_only_calibrated_output_no_rules"
            if inputs.track == "model_calibrated"
            else "community_cascade_final_output"
        ),
        "dataset": dataset,
        "prediction": prediction,
        "model": model,
        "selection": selection,
        "calibration": calibration_identity,
        "service": service,
        "upstream_model_raw": upstream_model_raw,
        "generation": generation,
        "provenance_implementation": _implementation_identity(),
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_document_ids": False,
            "aggregate_only": True,
        },
    }
    payload["manifest_sha256"] = _canonical_json_hash(payload)
    _validate_schema(payload)
    return payload


def _validate_schema(manifest: Mapping[str, Any]) -> None:
    schema, _identity = _json_file(
        _repository_root() / SCHEMA_REPOSITORY_PATH,
        field="release-eval-v2 prediction provenance schema",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
    except Exception as exc:
        raise ReleaseEvalV2PredictionProvenanceError(
            "prediction provenance manifest violates its strict schema"
        ) from exc


def replay_release_eval_v2_prediction_manifest(
    manifest_path: Path,
    inputs: PredictionProvenanceInputs,
) -> dict[str, Any]:
    """Strictly recompute a manifest and require complete field equality."""

    observed, identity = _json_file(
        manifest_path, field="release-eval-v2 prediction provenance manifest"
    )
    _validate_schema(observed)
    expected = build_release_eval_v2_prediction_manifest(inputs)
    if observed != expected:
        raise ReleaseEvalV2PredictionProvenanceError(
            "prediction provenance strict replay differs from the recorded manifest"
        )
    return {
        "status": "PASS",
        "track": expected["track"],
        "target_split": expected["dataset"]["target_split"],
        "manifest_file_sha256": identity.sha256,
        "manifest_sha256": expected["manifest_sha256"],
        "predictions_file_sha256": expected["prediction"]["file_sha256"],
        "prediction_document_count": expected["prediction"]["document_count"],
    }


def validate_release_eval_v2_prediction_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate aggregate metadata for a consumer which lacks replay inputs.

    The quality producer uses this narrow interface.  Release acceptance must
    still run the full replay function with every source artifact.
    """

    manifest, file_identity = _json_file(
        manifest_path, field="release-eval-v2 prediction provenance manifest"
    )
    _validate_schema(manifest)
    manifest_self = _verify_self_hash(
        manifest,
        field="release-eval-v2 prediction provenance manifest",
        hash_field="manifest_sha256",
        ensure_ascii=True,
    )
    track = manifest.get("track")
    if track not in {"model_raw", "model_calibrated", "full_system"}:
        raise ReleaseEvalV2PredictionProvenanceError("prediction provenance track is invalid")
    if manifest.get("provenance_implementation") != _implementation_identity():
        raise ReleaseEvalV2PredictionProvenanceError(
            "prediction provenance implementation identity changed"
        )
    generation = _mapping(manifest.get("generation"), field="prediction generation")
    if (
        generation.get("generator_implementation_sha256")
        != (generation_implementation_identity(str(track))["implementation_sha256"])
    ):
        raise ReleaseEvalV2PredictionProvenanceError(
            "prediction generator implementation identity changed"
        )
    dataset = _mapping(manifest.get("dataset"), field="prediction dataset")
    prediction = _mapping(manifest.get("prediction"), field="prediction artifact")
    model = _mapping(manifest.get("model"), field="prediction model")
    service = manifest.get("service")
    if track in {"model_calibrated", "full_system"}:
        service_mapping = _mapping(service, field="calibrated/system service")
        service_identity: dict[str, Any] | None = {
            "profile_id": service_mapping.get("profile_id"),
            "configuration_sha256": service_mapping.get("configuration_sha256"),
            "implementation_sha256": service_mapping.get("implementation_sha256"),
            "model_identity_sha256": service_mapping.get("model_identity_sha256"),
            "calibration_bundle_file_sha256": service_mapping.get("calibration_bundle_file_sha256"),
        }
    else:
        if service is not None:
            raise ReleaseEvalV2PredictionProvenanceError(
                "model_raw manifest unexpectedly contains a service identity"
            )
        service_identity = None
    return {
        "status": "PASS_METADATA_ONLY",
        "full_replay_required": True,
        "manifest_file_sha256": file_identity.sha256,
        "manifest_sha256": manifest_self,
        "track": track,
        "target_split": dataset.get("target_split"),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("dataset_version"),
        "dataset_manifest_file_sha256": dataset.get("dataset_manifest_file_sha256"),
        "dataset_manifest_sha256": dataset.get("dataset_manifest_sha256"),
        "materialization_receipt_file_sha256": dataset.get("materialization_receipt_file_sha256"),
        "materialization_receipt_sha256": dataset.get("materialization_receipt_sha256"),
        "freeze_receipt_file_sha256": dataset.get("freeze_receipt_file_sha256"),
        "gold_file_sha256": dataset.get("gold_file_sha256"),
        "gold_size_bytes": dataset.get("gold_size_bytes"),
        "gold_document_count": dataset.get("gold_document_count"),
        "predictions_file_sha256": prediction.get("file_sha256"),
        "predictions_size_bytes": prediction.get("size_bytes"),
        "prediction_document_count": prediction.get("document_count"),
        "training_manifest_file_sha256": model.get("training_manifest_file_sha256"),
        "training_manifest_sha256": model.get("training_manifest_sha256"),
        "model_identity_sha256": model.get("identity_sha256"),
        "output_artifact_sha256": model.get("output_artifact_sha256"),
        "service": service_identity,
    }


__all__ = [
    "AmendmentReplayInputs",
    "CalibrationInputs",
    "FROZEN_V2",
    "GENERATION_RECEIPT_SCHEMA_VERSION",
    "GenerationRuntime",
    "MANIFEST_TYPE",
    "ModelRawInputs",
    "PredictionProvenanceInputs",
    "ReleaseEvalV2PredictionProvenanceError",
    "SCHEMA_REPOSITORY_PATH",
    "SCHEMA_VERSION",
    "build_release_eval_v2_generation_receipt",
    "build_release_eval_v2_prediction_manifest",
    "generation_implementation_identity",
    "replay_release_eval_v2_prediction_manifest",
    "validate_release_eval_v2_prediction_manifest",
]
