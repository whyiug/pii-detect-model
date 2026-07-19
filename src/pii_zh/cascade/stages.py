"""Explicit stage policy for release and evaluation cascade profiles.

The policy is consumed by :class:`~pii_zh.cascade.pipeline.CascadePipeline`;
evaluation scripts do not implement their own validator, threshold, or fusion
logic.  The default policy preserves the release runtime.  Ablation policies
are deliberately marked evaluation-only and require an explicit constructor
opt-in so they cannot be selected by the service or ordinary ``pii-zh`` CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from .routing import CandidateBranch, canonicalize_entity_type

StagePolicyPurpose = Literal["release", "evaluation_ablation"]
ConflictPolicy = Literal["deterministic", "exact_duplicates_only"]

PIPELINE_STAGE_POLICY_SCHEMA_VERSION = "pii-zh.pipeline-stage-policy.v1"
_PURPOSES = frozenset({"release", "evaluation_ablation"})
_CONFLICT_POLICIES = frozenset({"deterministic", "exact_duplicates_only"})


def _entity_tuple(name: str, value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be null or a sequence of canonical labels")
    entities = tuple(canonicalize_entity_type(str(entity)) for entity in value)
    if len(entities) != len(set(entities)):
        raise ValueError(f"{name} must not contain duplicate labels")
    return entities


@dataclass(frozen=True, slots=True)
class PipelineStagePolicy:
    """Versioned switches for one pass through the shared stage graph."""

    schema_version: str = PIPELINE_STAGE_POLICY_SCHEMA_VERSION
    policy_id: str = "release-default-v1"
    purpose: StagePolicyPurpose = "release"
    context_enabled: bool = True
    context_required: bool = False
    validators_enabled: bool = True
    thresholds_enabled: bool = True
    conflict_policy: ConflictPolicy = "deterministic"
    rule_branch_entities: tuple[str, ...] | None = None
    model_branch_entities: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PIPELINE_STAGE_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"stage policy schema_version must be {PIPELINE_STAGE_POLICY_SCHEMA_VERSION}"
            )
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("stage policy_id must be a non-empty string")
        if self.purpose not in _PURPOSES:
            raise ValueError("stage policy purpose is invalid")
        for name in (
            "context_enabled",
            "context_required",
            "validators_enabled",
            "thresholds_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.context_required and not self.context_enabled:
            raise ValueError("context_required needs context_enabled")
        if self.conflict_policy not in _CONFLICT_POLICIES:
            raise ValueError("stage conflict_policy is invalid")
        for name in ("rule_branch_entities", "model_branch_entities"):
            value = getattr(self, name)
            if value is not None:
                normalized = _entity_tuple(name, value)
                object.__setattr__(self, name, normalized)

        # Release profiles may not weaken a safety stage.  A future release
        # policy can tune routes/context, but it must retain validation,
        # thresholding, and deterministic conflict ownership.
        if self.purpose == "release" and (
            not self.validators_enabled
            or not self.thresholds_enabled
            or self.conflict_policy != "deterministic"
            or self.rule_branch_entities is not None
            or self.model_branch_entities is not None
        ):
            raise ValueError("release stage policies may not weaken safety stages or routes")

    def branch_override(self, branch: CandidateBranch) -> frozenset[str] | None:
        values = self.rule_branch_entities if branch == "rule" else self.model_branch_entities
        return None if values is None else frozenset(values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PipelineStagePolicy:
        allowed = {
            "schema_version",
            "policy_id",
            "purpose",
            "context_enabled",
            "context_required",
            "validators_enabled",
            "thresholds_enabled",
            "conflict_policy",
            "rule_branch_entities",
            "model_branch_entities",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"stage policy contains unknown fields: {sorted(unknown)}")
        document = dict(value)
        for name in ("rule_branch_entities", "model_branch_entities"):
            if name in document:
                document[name] = _entity_tuple(name, document[name])
        return cls(**cast(dict[str, Any], document))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "purpose": self.purpose,
            "context_enabled": self.context_enabled,
            "context_required": self.context_required,
            "validators_enabled": self.validators_enabled,
            "thresholds_enabled": self.thresholds_enabled,
            "conflict_policy": self.conflict_policy,
            "rule_branch_entities": (
                list(self.rule_branch_entities) if self.rule_branch_entities is not None else None
            ),
            "model_branch_entities": (
                list(self.model_branch_entities) if self.model_branch_entities is not None else None
            ),
        }


DEFAULT_RELEASE_STAGE_POLICY = PipelineStagePolicy()


__all__ = [
    "DEFAULT_RELEASE_STAGE_POLICY",
    "PIPELINE_STAGE_POLICY_SCHEMA_VERSION",
    "ConflictPolicy",
    "PipelineStagePolicy",
    "StagePolicyPurpose",
]
