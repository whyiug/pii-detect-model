"""Frozen local CLUENER + native Presidio primary detector.

The public builder in this module reproduces the admitted Presidio comparator
without importing the evaluation package.  It accepts only one byte-for-byte
pinned, safetensors-only local checkpoint and never resolves a Hub model ID.
The checkpoint remains a separate third-party dependency: this package neither
downloads nor redistributes its weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from pii_zh.taxonomy import load_presidio_mapping

if TYPE_CHECKING:  # pragma: no cover - avoids a package import cycle at runtime
    from pii_zh.cascade.config import CascadeConfig
    from pii_zh.cascade.result import CascadeDetection

CLUENER_PRIMARY_SCHEMA_VERSION = "pii-zh.cluener-primary.v1"
CLUENER_PRIMARY_PROFILE_VERSION = "native-presidio-cluener-cn-common-v5-v1"
CLUENER_PRIMARY_MODEL_ID = "uer/roberta-base-finetuned-cluener2020-chinese"
CLUENER_PRIMARY_MODEL_REVISION = "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b"
CLUENER_PRIMARY_PRESIDIO_VERSION = "2.2.363"
CLUENER_PRIMARY_PROTOCOL_ID = "public_synthetic_chinese_pii_open24_v2"
CLUENER_PRIMARY_SOURCE = "presidio:cluener-cn-common-v5-primary"
CLUENER_PRIMARY_DEFAULT_THRESHOLD = 0.0
CLUENER_PRIMARY_MAX_TOKENS = 512
CLUENER_PRIMARY_STRIDE_FRACTION = 0.25
CLUENER_PRIMARY_DEFAULT_MICRO_BATCH_SIZE = 16

CLUENER_PRIMARY_PROJECT_CLOSED8: tuple[str, ...] = (
    "ADDRESS",
    "BANK_CARD_NUMBER",
    "CN_RESIDENT_ID",
    "EMAIL_ADDRESS",
    "PASSPORT_NUMBER",
    "PERSON_NAME",
    "PHONE_NUMBER",
    "VEHICLE_LICENSE_PLATE",
)
CLUENER_PRIMARY_CLOSED8_ENTITIES: tuple[str, ...] = (
    "CN_ADDRESS",
    "BANK_CARD_NUMBER",
    "CN_ID_CARD",
    "EMAIL_ADDRESS",
    "PASSPORT_NUMBER",
    "PERSON",
    "PHONE_NUMBER",
    "CN_VEHICLE_LICENSE_PLATE",
)

CLUENER_PRIMARY_REQUIRED_FILES: Mapping[str, str] = MappingProxyType(
    {
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
        "tokenizer.json": (
            "7dfbf1966ebf99d471c3796e9b457329d2b2182b817e144f1e904b957745c839"
        ),
        "tokenizer_config.json": (
            "21b0a2fba3d74b521cdbd631b2936c2c063c856c3a05533fabee04005c02cd23"
        ),
        "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
    }
)
CLUENER_PRIMARY_CONVERSION_RECEIPT_EXPECTED: Mapping[str, str | bool] = MappingProxyType(
    {
        "converter_sha256": (
            "fe9ba3d29c151688ba04e6cf9993d1bf8c0e9cd50f1679701d271f4ce766651a"
        ),
        "destination_sha256": CLUENER_PRIMARY_REQUIRED_FILES["model.safetensors"],
        "loader": "torch.load(weights_only=True,map_location=cpu)",
        "payload_contract": "flat_str_to_tensor_state_dict",
        "source_deleted": True,
        "source_sha256": (
            "b865252516115c46bc508167fa2258f198bcce520eb7a1ac4fbf8d50dc361368"
        ),
        "verification": "safetensors_full_tensor_equality",
    }
)
CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED: Mapping[str, str | bool] = MappingProxyType(
    {
        "converter_sha256": (
            "418be7a56aad52c56a37cd93454de6eacb9eafd8712bc1ea38c24a2339def474"
        ),
        "destination_sha256": CLUENER_PRIMARY_REQUIRED_FILES["model.safetensors"],
        "loader": "torch.load(weights_only=True,map_location=cpu)",
        "payload_contract": "flat_str_to_tensor_state_dict",
        "source_deleted": False,
        "source_sha256": (
            "b865252516115c46bc508167fa2258f198bcce520eb7a1ac4fbf8d50dc361368"
        ),
        "verification": "safetensors_full_tensor_equality",
    }
)
CLUENER_PRIMARY_ACCEPTED_CONVERSION_RECEIPT_SHA256: frozenset[str] = frozenset(
    {
        CLUENER_PRIMARY_REQUIRED_FILES["conversion_receipt.json"],
        "84d8af71c97f95b33cf44fbc28e7e91e7461b0002e2680ce6780a66347f2b92a",
    }
)
CLUENER_PRIMARY_CONVERSION_RECEIPT_BY_CONVERTER: Mapping[
    str, Mapping[str, str | bool]
] = MappingProxyType(
    {
        str(CLUENER_PRIMARY_CONVERSION_RECEIPT_EXPECTED["converter_sha256"]): (
            CLUENER_PRIMARY_CONVERSION_RECEIPT_EXPECTED
        ),
        str(CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED["converter_sha256"]): (
            CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED
        ),
    }
)

_UNSAFE_WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"})
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
_IdentityValue = str | int | bool | None
_StatSnapshot = tuple[tuple[str, int, int, int, int, int], ...]


class CluenerPrimaryContractError(RuntimeError):
    """Raised before inference when the frozen primary contract is not met."""


def _canonical_json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stat_signature(name: str, value: os.stat_result) -> tuple[str, int, int, int, int, int]:
    return (
        name,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _artifact_snapshot(root: Path) -> _StatSnapshot:
    snapshot: list[tuple[str, int, int, int, int, int]] = []
    for name in sorted(CLUENER_PRIMARY_REQUIRED_FILES):
        artifact = root / name
        try:
            observed = artifact.lstat()
        except OSError as exc:
            raise CluenerPrimaryContractError("required CLUENER artifact is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise CluenerPrimaryContractError(
                "required CLUENER artifacts must be regular, non-symlink files"
            )
        if observed.st_size > _MAX_ARTIFACT_BYTES:
            raise CluenerPrimaryContractError("CLUENER artifact exceeds the supported size limit")
        snapshot.append(_stat_signature(name, observed))
    return tuple(snapshot)


def _hash_regular_file(path: Path) -> str:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise CluenerPrimaryContractError(
                "required CLUENER artifacts must be regular, non-symlink files"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if _stat_signature(path.name, opened) != _stat_signature(path.name, before):
                raise CluenerPrimaryContractError("CLUENER artifact changed before verification")
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise CluenerPrimaryContractError("cannot read required CLUENER artifact") from exc
    if _stat_signature(path.name, after) != _stat_signature(path.name, before):
        raise CluenerPrimaryContractError("CLUENER artifact changed during verification")
    return digest.hexdigest()


def _conversion_receipt(root: Path) -> Mapping[str, Any]:
    receipt_path = root / "conversion_receipt.json"
    try:
        payload = receipt_path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CluenerPrimaryContractError("CLUENER conversion receipt is invalid") from exc
    if not isinstance(value, Mapping):
        raise CluenerPrimaryContractError("CLUENER safe-conversion receipt changed")
    converter_sha256 = value.get("converter_sha256")
    expected = (
        CLUENER_PRIMARY_CONVERSION_RECEIPT_BY_CONVERTER.get(converter_sha256)
        if isinstance(converter_sha256, str)
        else None
    )
    if expected is None or any(value.get(key) != item for key, item in expected.items()):
        raise CluenerPrimaryContractError("CLUENER safe-conversion receipt changed")
    return value


def _identity(observed: Mapping[str, str]) -> Mapping[str, _IdentityValue]:
    closed8_hash = _canonical_json_hash(CLUENER_PRIMARY_CLOSED8_ENTITIES)
    identity: dict[str, _IdentityValue] = {
        "schema_version": CLUENER_PRIMARY_SCHEMA_VERSION,
        "profile_version": CLUENER_PRIMARY_PROFILE_VERSION,
        "artifact_type": "huggingface_safetensors_bios_token_classifier",
        "source_model_id": CLUENER_PRIMARY_MODEL_ID,
        "source_revision": CLUENER_PRIMARY_MODEL_REVISION,
        "presidio_analyzer_version": CLUENER_PRIMARY_PRESIDIO_VERSION,
        "protocol_id": CLUENER_PRIMARY_PROTOCOL_ID,
        "rule_pack_id": "cn_common_v5",
        "ner_source_labels": "address,name",
        "ner_output_entities": "CN_ADDRESS,PERSON",
        "closed8_entity_count": len(CLUENER_PRIMARY_CLOSED8_ENTITIES),
        "closed8_entities_sha256": closed8_hash,
        "artifact_file_count": len(observed),
        "artifact_files_combined_sha256": _canonical_json_hash(dict(sorted(observed.items()))),
        "model_safetensors_sha256": observed["model.safetensors"],
        "conversion_receipt_sha256": observed["conversion_receipt.json"],
        "local_files_only": True,
        "safetensors_only": True,
        "trust_remote_code": False,
        "weights_bundled": False,
    }
    identity["identity_sha256"] = _canonical_json_hash(identity)
    # Assert the public mapping remains flat and JSON-safe before exposing it.
    json.dumps(identity, ensure_ascii=True, allow_nan=False, sort_keys=True)
    return MappingProxyType(identity)


@dataclass(frozen=True, slots=True)
class VerifiedCluenerPrimaryArtifact:
    """A verified local root plus a path-free public model identity."""

    root: Path
    model_identity: Mapping[str, _IdentityValue]
    _snapshot: _StatSnapshot = field(repr=False, compare=False)


def verify_cluener_primary_artifact(
    model_path: str | Path,
) -> VerifiedCluenerPrimaryArtifact:
    """Verify the exact admitted local checkpoint without loading a model."""

    candidate = Path(model_path).expanduser()
    if candidate.is_symlink():
        raise CluenerPrimaryContractError("CLUENER model directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CluenerPrimaryContractError("CLUENER model directory is unavailable") from exc
    if not root.is_dir():
        raise CluenerPrimaryContractError("CLUENER model_path must be a local directory")

    try:
        top_level_files = {item.name for item in root.iterdir() if item.is_file()}
    except OSError as exc:
        raise CluenerPrimaryContractError("cannot inspect CLUENER model directory") from exc
    if top_level_files != set(CLUENER_PRIMARY_REQUIRED_FILES):
        raise CluenerPrimaryContractError("CLUENER top-level file inventory changed")
    unsafe = sorted(
        name
        for name in top_level_files
        if Path(name).suffix.casefold() in _UNSAFE_WEIGHT_SUFFIXES
    )
    if unsafe:
        raise CluenerPrimaryContractError("unsafe serialized CLUENER weights are forbidden")

    before = _artifact_snapshot(root)
    observed: dict[str, str] = {}
    for name, expected in sorted(CLUENER_PRIMARY_REQUIRED_FILES.items()):
        actual = _hash_regular_file(root / name)
        accepted = actual == expected
        if name == "conversion_receipt.json":
            accepted = actual in CLUENER_PRIMARY_ACCEPTED_CONVERSION_RECEIPT_SHA256
        if not accepted:
            raise CluenerPrimaryContractError(
                "local CLUENER artifact does not match the pinned revision"
            )
        observed[name] = actual
    _conversion_receipt(root)
    after = _artifact_snapshot(root)
    if after != before:
        raise CluenerPrimaryContractError("CLUENER artifact changed during verification")
    return VerifiedCluenerPrimaryArtifact(
        root=root,
        model_identity=_identity(observed),
        _snapshot=after,
    )


def _require_exact_presidio_version() -> str:
    from pii_zh.presidio.native_framework import presidio_analyzer_version

    actual = presidio_analyzer_version()
    if actual != CLUENER_PRIMARY_PRESIDIO_VERSION:
        raise CluenerPrimaryContractError(
            "native Presidio version must be exactly " + CLUENER_PRIMARY_PRESIDIO_VERSION
        )
    return actual


def _source_to_presidio() -> dict[str, str]:
    mapping = load_presidio_mapping().model_to_presidio
    expected_closed8 = tuple(mapping[label] for label in CLUENER_PRIMARY_PROJECT_CLOSED8)
    if expected_closed8 != CLUENER_PRIMARY_CLOSED8_ENTITIES:
        raise CluenerPrimaryContractError("packaged Presidio closed8 mapping changed")
    return {
        "address": mapping["ADDRESS"],
        "name": mapping["PERSON_NAME"],
    }


def _load_native_framework(
    artifact: VerifiedCluenerPrimaryArtifact,
    *,
    device: str,
    micro_batch_size: int,
) -> object:
    import torch

    from pii_zh.inference.bios_token_classifier import load_closed_bios_token_classifier
    from pii_zh.presidio.bio_token_recognizer import BioTokenClassifierRecognizer
    from pii_zh.presidio.cn_common_recognizer import CnCommonRecognizer
    from pii_zh.presidio.native_framework import NativePresidioFrameworkRecognizer
    from pii_zh.rules import CnCommonRulePack

    source_to_presidio = _source_to_presidio()
    dtype = None if device == "cpu" else torch.bfloat16
    predictor = load_closed_bios_token_classifier(
        artifact.root,
        source_to_target=source_to_presidio,
        protocol_id=CLUENER_PRIMARY_PROTOCOL_ID,
        device=device,
        dtype=dtype,
        micro_batch_size=micro_batch_size,
    )
    ner = BioTokenClassifierRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        supported_entities=sorted(source_to_presidio.values()),
        default_threshold=CLUENER_PRIMARY_DEFAULT_THRESHOLD,
        max_tokens=CLUENER_PRIMARY_MAX_TOKENS,
        stride_fraction=CLUENER_PRIMARY_STRIDE_FRACTION,
        model_version=(
            f"{CLUENER_PRIMARY_MODEL_ID}@{CLUENER_PRIMARY_MODEL_REVISION}"
        ),
        source_name="native_presidio:cluener_open24",
        deduplication_policy="exact_only_v1",
        name="CluenerOpen24NativeFrameworkNerRecognizer",
    )
    return NativePresidioFrameworkRecognizer(
        ner_recognizer=ner,
        rule_recognizer=CnCommonRecognizer(
            rule_pack=CnCommonRulePack(),
            name="CnCommonV5Open24ComparatorRecognizer",
            version="5.0.0",
        ),
    )


def _validate_runtime_options(device: str, micro_batch_size: int) -> None:
    if not isinstance(device, str) or not device.strip():
        raise TypeError("device must be a non-empty string")
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int):
        raise TypeError("micro_batch_size must be an integer")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")


def build_cluener_primary_native_framework(
    model_path: str | Path,
    *,
    device: str = "cpu",
    micro_batch_size: int = CLUENER_PRIMARY_DEFAULT_MICRO_BATCH_SIZE,
) -> object:
    """Build the frozen CLUENER + cn_common_v5 native AnalyzerEngine branch."""

    _validate_runtime_options(device, micro_batch_size)
    _require_exact_presidio_version()
    artifact = verify_cluener_primary_artifact(model_path)
    framework = _load_native_framework(
        artifact,
        device=device,
        micro_batch_size=micro_batch_size,
    )
    runtime_identity = getattr(framework, "runtime_identity", None)
    if not callable(runtime_identity):
        raise CluenerPrimaryContractError("primary detector is not a native Presidio framework")
    identity = runtime_identity()
    engine = getattr(framework, "engine", None)
    if (
        not isinstance(identity, Mapping)
        or identity.get("framework") != "presidio-analyzer"
        or identity.get("framework_version") != CLUENER_PRIMARY_PRESIDIO_VERSION
        or type(engine).__name__ != "AnalyzerEngine"
        or not type(engine).__module__.startswith("presidio_analyzer")
    ):
        raise CluenerPrimaryContractError("primary detector is not the pinned native Presidio")
    if _artifact_snapshot(artifact.root) != artifact._snapshot:
        raise CluenerPrimaryContractError("CLUENER artifact changed while loading")
    cast(Any, framework).model_identity = artifact.model_identity
    return framework


def _flat_identity(
    value: Mapping[str, _IdentityValue],
) -> Mapping[str, _IdentityValue]:
    if not value or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, (str, int, bool, type(None)))
        for key, item in value.items()
    ):
        raise TypeError("model_identity must be a non-empty flat JSON-safe mapping")
    copied = dict(value)
    json.dumps(copied, ensure_ascii=True, allow_nan=False, sort_keys=True)
    return MappingProxyType(copied)


def _cluener_primary_config() -> CascadeConfig:
    """Describe exact closed8 ownership for the outer composition layer."""

    from pii_zh.cascade.config import CascadeConfig
    from pii_zh.cascade.routing import EntityRoute, default_routes

    defaults = {route.entity_type: route for route in default_routes()}
    routes = tuple(
        EntityRoute(
            entity_type=entity,
            category=defaults[entity].category,
            rule_enabled=True,
            model_enabled=False,
            validator_label=defaults[entity].validator_label,
            model_requires_validator=False,
            rule_threshold=0.0,
            model_threshold=0.0,
        )
        for entity in CLUENER_PRIMARY_CLOSED8_ENTITIES
    )
    if {route.entity_type for route in routes} != set(CLUENER_PRIMARY_CLOSED8_ENTITIES):
        raise CluenerPrimaryContractError("primary config does not own exactly closed8")
    return CascadeConfig(
        profile_version=CLUENER_PRIMARY_PROFILE_VERSION,
        mode="rules-only",
        routes=routes,
        rule_source=CLUENER_PRIMARY_SOURCE,
        model_source="qwen:disabled-in-cluener-primary",
        keep_nested_different_types=False,
    )


class CluenerPrimaryPipeline:
    """Narrow, no-fusion adapter for one verified native Presidio framework."""

    __slots__ = ("_config", "_framework", "_model_identity")

    def __init__(
        self,
        *,
        native_framework: object,
        model_identity: Mapping[str, _IdentityValue],
    ) -> None:
        if not callable(getattr(native_framework, "analyze", None)):
            raise TypeError("native_framework must expose analyze")
        self._framework = native_framework
        self._model_identity = _flat_identity(model_identity)
        self._config = _cluener_primary_config()

    @property
    def config(self) -> CascadeConfig:
        return self._config

    @property
    def native_framework(self) -> object:
        return self._framework

    @property
    def model_identity(self) -> Mapping[str, _IdentityValue]:
        return self._model_identity

    @staticmethod
    def _requested_entities(entities: Sequence[str] | None) -> tuple[str, ...]:
        if entities is None:
            return CLUENER_PRIMARY_CLOSED8_ENTITIES
        if isinstance(entities, (str, bytes)):
            raise TypeError("entities must be a sequence of closed8 labels")
        aliases = dict(
            zip(
                CLUENER_PRIMARY_PROJECT_CLOSED8,
                CLUENER_PRIMARY_CLOSED8_ENTITIES,
                strict=True,
            )
        )
        supported = frozenset(CLUENER_PRIMARY_CLOSED8_ENTITIES)
        requested: list[str] = []
        for entity in entities:
            if not isinstance(entity, str):
                raise TypeError("entities must contain strings")
            normalized = aliases.get(entity, entity)
            if normalized not in supported:
                raise ValueError("entities must be limited to the canonical closed8")
            if normalized not in requested:
                requested.append(normalized)
        return tuple(requested)

    def _detect(self, text: str, requested: tuple[str, ...]) -> list[CascadeDetection]:
        if not requested:
            return []
        raw_results = cast(Any, self._framework).analyze(text, entities=requested)
        if isinstance(raw_results, (str, bytes, Mapping)):
            raise TypeError("native Presidio analyze must return a sequence")
        try:
            values = list(raw_results)
        except TypeError as exc:
            raise TypeError("native Presidio analyze must return a sequence") from exc

        # Delay this import until invocation so importing pii_zh.presidio stays
        # lightweight and cannot create a presidio <-> cascade package cycle.
        from pii_zh.cascade.result import CascadeDetection

        detections: list[CascadeDetection] = []
        allowed = frozenset(requested)
        for result in values:
            entity_type = getattr(result, "entity_type", None)
            start = getattr(result, "start", None)
            end = getattr(result, "end", None)
            score = getattr(result, "score", None)
            if not isinstance(entity_type, str) or entity_type not in allowed:
                raise CluenerPrimaryContractError(
                    "native Presidio emitted an entity outside the requested closed8"
                )
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
            ):
                raise CluenerPrimaryContractError("native Presidio emitted invalid offsets")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise CluenerPrimaryContractError("native Presidio emitted an invalid score")
            normalized_score = float(score)
            if not math.isfinite(normalized_score):
                raise CluenerPrimaryContractError("native Presidio emitted an invalid score")
            try:
                detection = CascadeDetection(
                    start=start,
                    end=end,
                    entity_type=entity_type,
                    score=normalized_score,
                    source=CLUENER_PRIMARY_SOURCE,
                    sources=(CLUENER_PRIMARY_SOURCE,),
                    decision_process=("primary:native_presidio",),
                )
            except (TypeError, ValueError) as exc:
                raise CluenerPrimaryContractError(
                    "native Presidio emitted an invalid detection"
                ) from exc
            detections.append(detection)
        return sorted(
            detections,
            key=lambda item: (item.start, item.end, item.entity_type, -item.score),
        )

    def detect(
        self,
        text: str,
        *,
        entities: Sequence[str] | None = None,
    ) -> list[CascadeDetection]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._detect(text, self._requested_entities(entities))

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> list[list[CascadeDetection]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        requested = self._requested_entities(entities)
        return [self._detect(text, requested) for text in values]


def build_cluener_primary_pipeline(
    model_path: str | Path,
    *,
    device: str = "cpu",
    micro_batch_size: int = CLUENER_PRIMARY_DEFAULT_MICRO_BATCH_SIZE,
) -> CluenerPrimaryPipeline:
    """Build the no-fusion primary adapter from one verified local checkpoint."""

    framework = build_cluener_primary_native_framework(
        model_path,
        device=device,
        micro_batch_size=micro_batch_size,
    )
    identity = getattr(framework, "model_identity", None)
    if not isinstance(identity, Mapping):  # pragma: no cover - builder invariant
        raise CluenerPrimaryContractError("native primary lost its model identity")
    return CluenerPrimaryPipeline(
        native_framework=framework,
        model_identity=identity,
    )


__all__ = [
    "CLUENER_PRIMARY_CLOSED8_ENTITIES",
    "CLUENER_PRIMARY_CONVERSION_RECEIPT_EXPECTED",
    "CLUENER_PRIMARY_DEFAULT_MICRO_BATCH_SIZE",
    "CLUENER_PRIMARY_DEFAULT_THRESHOLD",
    "CLUENER_PRIMARY_MAX_TOKENS",
    "CLUENER_PRIMARY_MODEL_ID",
    "CLUENER_PRIMARY_MODEL_REVISION",
    "CLUENER_PRIMARY_PRESIDIO_VERSION",
    "CLUENER_PRIMARY_PROFILE_VERSION",
    "CLUENER_PRIMARY_PROJECT_CLOSED8",
    "CLUENER_PRIMARY_PROTOCOL_ID",
    "CLUENER_PRIMARY_REQUIRED_FILES",
    "CLUENER_PRIMARY_SCHEMA_VERSION",
    "CLUENER_PRIMARY_SOURCE",
    "CLUENER_PRIMARY_STRIDE_FRACTION",
    "CluenerPrimaryContractError",
    "CluenerPrimaryPipeline",
    "VerifiedCluenerPrimaryArtifact",
    "build_cluener_primary_native_framework",
    "build_cluener_primary_pipeline",
    "verify_cluener_primary_artifact",
]
