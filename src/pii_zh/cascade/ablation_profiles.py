"""Load the packaged six-way cascade ablation profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal, cast

from .config import CascadeMode
from .stages import PipelineStagePolicy

AblationRecognizerKind = Literal[
    "none",
    "project_model",
    "native_presidio_cn_common_closed_bio_ner",
]
ABLATION_PROFILE_SET_SCHEMA_VERSION = "pii-zh.cascade-ablation-profiles.v1"
ABLATION_PROFILE_SET_ID = "human-hidden-six-way-ablation-v1"
ABLATION_PROFILE_RESOURCE = "profiles/ablation_profiles_v1.json"
REQUIRED_ABLATION_IDS = (
    "rules_only",
    "presidio_zh_ner_cn_common",
    "project_model_raw",
    "simple_union",
    "validator_threshold_fusion",
    "final_cascade",
)
_MODES = frozenset({"rules-only", "model-only", "cascade"})
_RECOGNIZER_KINDS = frozenset(
    {"none", "project_model", "native_presidio_cn_common_closed_bio_ner"}
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AblationRuntimeProfile:
    """One immutable adapter definition from the packaged profile set."""

    adapter_id: str
    mode: CascadeMode
    recognizer_kind: AblationRecognizerKind
    rule_source: str
    model_source: str
    stage_policy: PipelineStagePolicy
    profile_sha256: str
    profile_set_file_sha256: str


def _profile(
    adapter_id: str,
    raw: object,
    *,
    profile_set_file_sha256: str,
) -> AblationRuntimeProfile:
    if not isinstance(raw, Mapping):
        raise TypeError("ablation profile entries must be mappings")
    allowed = {
        "adapter_id",
        "mode",
        "recognizer_kind",
        "rule_source",
        "model_source",
        "stage_policy",
    }
    if set(raw) != allowed:
        raise ValueError("ablation profile fields do not match the frozen schema")
    if raw.get("adapter_id") != adapter_id:
        raise ValueError("ablation profile key and adapter_id differ")
    mode = raw.get("mode")
    recognizer_kind = raw.get("recognizer_kind")
    if mode not in _MODES:
        raise ValueError("ablation profile mode is invalid")
    if recognizer_kind not in _RECOGNIZER_KINDS:
        raise ValueError("ablation recognizer_kind is invalid")
    for name in ("rule_source", "model_source"):
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ablation {name} must be a non-empty string")
    stage = raw.get("stage_policy")
    if not isinstance(stage, Mapping):
        raise TypeError("ablation stage_policy must be a mapping")
    stage_policy = PipelineStagePolicy.from_mapping(cast(Mapping[str, Any], stage))
    if stage_policy.purpose != "evaluation_ablation":
        raise ValueError("packaged ablation policies must be evaluation-only")
    return AblationRuntimeProfile(
        adapter_id=adapter_id,
        mode=cast(CascadeMode, mode),
        recognizer_kind=cast(AblationRecognizerKind, recognizer_kind),
        rule_source=cast(str, raw["rule_source"]),
        model_source=cast(str, raw["model_source"]),
        stage_policy=stage_policy,
        profile_sha256=_canonical_hash(raw),
        profile_set_file_sha256=profile_set_file_sha256,
    )


def load_ablation_profiles() -> dict[str, AblationRuntimeProfile]:
    """Return all six profiles after strict packaged-resource validation."""

    resource = files("pii_zh.cascade").joinpath(ABLATION_PROFILE_RESOURCE)
    payload = resource.read_bytes()
    profile_set_file_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("packaged ablation profile set is invalid JSON") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version",
        "profile_set_id",
        "profiles",
    }:
        raise ValueError("packaged ablation profile set has an invalid shape")
    if (
        document.get("schema_version") != ABLATION_PROFILE_SET_SCHEMA_VERSION
        or document.get("profile_set_id") != ABLATION_PROFILE_SET_ID
    ):
        raise ValueError("packaged ablation profile identity is invalid")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, Mapping) or tuple(raw_profiles) != REQUIRED_ABLATION_IDS:
        raise ValueError("packaged ablation profile order/set is invalid")
    profiles = {
        adapter_id: _profile(
            adapter_id,
            raw_profiles[adapter_id],
            profile_set_file_sha256=profile_set_file_sha256,
        )
        for adapter_id in REQUIRED_ABLATION_IDS
    }
    return profiles


def load_ablation_profile(adapter_id: str) -> AblationRuntimeProfile:
    if not isinstance(adapter_id, str):
        raise TypeError("adapter_id must be a string")
    try:
        return load_ablation_profiles()[adapter_id]
    except KeyError:
        raise ValueError("unknown cascade ablation adapter_id") from None


__all__ = [
    "ABLATION_PROFILE_RESOURCE",
    "ABLATION_PROFILE_SET_ID",
    "ABLATION_PROFILE_SET_SCHEMA_VERSION",
    "REQUIRED_ABLATION_IDS",
    "AblationRecognizerKind",
    "AblationRuntimeProfile",
    "load_ablation_profile",
    "load_ablation_profiles",
]
