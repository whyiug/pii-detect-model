"""Research-only closed-8 cascade for the manifest-bound project BIE73 model.

This module is deliberately separate from the public community service
profile.  It reuses that profile's routes, rules, validators, calibration and
fusion, while keeping the BIE artifact contract and loader out of the release
BIO/Open24 construction path.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from pii_zh.calibration import CalibrationBundle
from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
from pii_zh.inference.project_bie import (
    PROJECT_BIE_PROTOCOL_ID,
    PROJECT_CLOSED8_LABELS,
    load_local_project_bie_predictor,
    normalize_project_bie_id2label,
    project_bie_allowed_label_ids,
)
from pii_zh.inference.project_full_bie import (
    FULL_BIE73_MANIFEST_TYPE,
    FULL_BIE73_MODEL_PROTOCOL,
    FULL_BIE73_PROTOCOL_ID,
    load_local_full_bie73_predictor,
    verify_full_bie73_artifact,
)
from pii_zh.models.aiguard24 import AIGUARD_SOURCE_MODEL_ID, AIGUARD_SOURCE_REVISION
from pii_zh.models.aiguard24_bie import (
    AIGUARD24_BIE_INITIALIZATION_STRATEGY,
    build_core_bie_label_maps,
)
from pii_zh.presidio import QwenPiiRecognizer
from pii_zh.rules.cn_common_v6 import CnCommonRulePackV6
from pii_zh.taxonomy import load_taxonomy
from pii_zh.training.manifest import (
    canonical_json_hash,
    verify_output_artifact_binding,
    verify_training_manifest,
)

from .community_model import canonical_json_hash as service_policy_hash
from .config import CascadeMode
from .pipeline import CascadePipeline
from .routing import (
    COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES,
    EntityRoute,
    community_full24_routes,
)
from .service_profiles import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    _normalize_calibration,
    _normalize_thresholds,
    load_service_config,
)

RESEARCH_BIE_MANIFEST_TYPE = "aiguard24_native_causal_bie_training_v1"
RESEARCH_BIE_MODEL_PROTOCOL = "BIE73"
RESEARCH_BIE_MODEL_IDENTITY_SCHEMA_VERSION = "pii-zh.research-bie-model-identity.v1"
RESEARCH_FULL_BIE_DECODER_ID = "constrained_viterbi"
ResearchBieRulePolicy = Literal[
    "preserve_all_validated_rules",
    "community_calibrated_rule_suppression",
    "fpr_guarded_weak5",
    "fpr_guarded_all6",
    "model_only_rules_disabled",
]
PRESERVE_ALL_VALIDATED_RULES: ResearchBieRulePolicy = "preserve_all_validated_rules"
COMMUNITY_CALIBRATED_RULE_SUPPRESSION: ResearchBieRulePolicy = (
    "community_calibrated_rule_suppression"
)
FPR_GUARDED_WEAK5: ResearchBieRulePolicy = "fpr_guarded_weak5"
FPR_GUARDED_ALL6: ResearchBieRulePolicy = "fpr_guarded_all6"
MODEL_ONLY_RULES_DISABLED: ResearchBieRulePolicy = "model_only_rules_disabled"
DEFAULT_RESEARCH_BIE_RULE_POLICY = COMMUNITY_CALIBRATED_RULE_SUPPRESSION
# Historical v1 grid.  Keep this tuple byte-for-byte semantically stable so the
# sealed six-candidate receipt remains independently interpretable.
RESEARCH_BIE_RULE_POLICIES: tuple[ResearchBieRulePolicy, ...] = (
    PRESERVE_ALL_VALIDATED_RULES,
    COMMUNITY_CALIBRATED_RULE_SUPPRESSION,
)
# Adaptive v2 grid, preregistered only after the v1 aggregate receipt reported
# no feasible candidate.  Neither policy disables a model branch.
ADAPTIVE_RESEARCH_BIE_RULE_POLICIES: tuple[ResearchBieRulePolicy, ...] = (
    FPR_GUARDED_WEAK5,
    FPR_GUARDED_ALL6,
)
ALL_RESEARCH_BIE_RULE_POLICIES: tuple[ResearchBieRulePolicy, ...] = (
    *RESEARCH_BIE_RULE_POLICIES,
    *ADAPTIVE_RESEARCH_BIE_RULE_POLICIES,
    MODEL_ONLY_RULES_DISABLED,
)

# V1 diagnostics (not used for v1 selection) showed threshold-insensitive
# aggregate false positives in these six labels (590/605 at t=.5 and 595/600
# at t=.8), while their rule routes remained enabled.  That is a plausible
# rule-association hypothesis, not span-level causal attribution: no row,
# prediction, or entity value was inspected.  ``weak5`` retains the checksum-
# validated CN ID rule; ``all6`` also suppresses it.  Both retain every neural
# route and are evaluated under a per-label recall guard.
FPR_GUARDED_WEAK5_ADDITIONAL_RULE_DISABLED_ENTITIES = frozenset(
    {
        "EMAIL_ADDRESS",
        "GEO_COORDINATE",
        "MAC_ADDRESS",
        "PHONE_NUMBER",
        "SECRET",
    }
)
FPR_GUARDED_WEAK5_RULE_DISABLED_ENTITIES = (
    COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES
    | FPR_GUARDED_WEAK5_ADDITIONAL_RULE_DISABLED_ENTITIES
)
FPR_GUARDED_ALL6_RULE_DISABLED_ENTITIES = FPR_GUARDED_WEAK5_RULE_DISABLED_ENTITIES | {"CN_ID_CARD"}
MODEL_ONLY_RULE_DISABLED_ENTITIES = frozenset(
    route.entity_type for route in community_full24_routes()
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "status",
        "release_eligible",
        "created_at",
        "completed_at",
        "seed",
        "attention_mode",
        "tag_scheme",
        "update_mode",
        "base_source_id",
        "source_revision",
        "training_source_ids",
        "taxonomy_version",
        "label2id",
        "label_schema_sha256",
        "recipe",
        "recipe_sha256",
        "initialization",
        "parameter_update_audit",
        "tokenizer",
        "datasets",
        "split_isolation",
        "dev_selection_gates",
        "benchmark_isolation",
        "versions",
        "code_revision",
        "privacy",
        "runtime_environment",
        "checkpoint_selection",
        "independent_validation",
        "output_artifact",
        "manifest_sha256",
    }
)


class ResearchBieArtifactError(RuntimeError):
    """Raised before loading when a BIE73 research artifact fails closed."""


@dataclass(frozen=True, slots=True)
class ResearchBieModelIdentity:
    """Path-free identity for a verified research-only BIE73 artifact."""

    schema_version: str
    manifest_type: str
    manifest_sha256: str
    model_protocol: str
    protocol_id: str
    model_type: str
    attention_mode: str
    tag_scheme: str
    taxonomy_version: str
    label_count: int
    label_schema_sha256: str
    weights_combined_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerifiedResearchBieModel:
    root: Path
    identity: ResearchBieModelIdentity


def normalize_research_bie_rule_policy(value: str) -> ResearchBieRulePolicy:
    """Validate the closed research rule-policy enum without fallback."""

    if not isinstance(value, str):
        raise TypeError("research BIE rule_policy must be a string")
    if value not in ALL_RESEARCH_BIE_RULE_POLICIES:
        raise ValueError("research BIE rule_policy is unsupported")
    return value


def research_bie_disabled_rule_entities(
    rule_policy: ResearchBieRulePolicy,
) -> frozenset[str]:
    """Return the exact policy-facing rule branches disabled by one policy."""

    normalized = normalize_research_bie_rule_policy(rule_policy)
    if normalized == PRESERVE_ALL_VALIDATED_RULES:
        return frozenset()
    if normalized == COMMUNITY_CALIBRATED_RULE_SUPPRESSION:
        return COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES
    if normalized == FPR_GUARDED_WEAK5:
        return FPR_GUARDED_WEAK5_RULE_DISABLED_ENTITIES
    if normalized == FPR_GUARDED_ALL6:
        return FPR_GUARDED_ALL6_RULE_DISABLED_ENTITIES
    if normalized == MODEL_ONLY_RULES_DISABLED:
        return MODEL_ONLY_RULE_DISABLED_ENTITIES
    raise AssertionError("normalized research rule policy is unhandled")


def _research_routes(
    *,
    recognizer_thresholds_authoritative: bool,
    rule_policy: ResearchBieRulePolicy,
) -> tuple[EntityRoute, ...]:
    """Return community routes with an explicitly bound research rule policy."""

    if not recognizer_thresholds_authoritative:
        return community_full24_routes()
    authoritative = community_full24_routes(recognizer_thresholds_authoritative=True)
    disabled = research_bie_disabled_rule_entities(rule_policy)
    resolved = tuple(
        replace(route, rule_enabled=route.entity_type not in disabled) for route in authoritative
    )
    if any(route.model_threshold != 0.0 for route in resolved):
        raise RuntimeError("research authoritative model thresholds must remain zero")
    if any(not route.model_enabled for route in resolved):
        raise RuntimeError("research rule policies must retain every neural route")
    return resolved


def _json_object(path: Path, *, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ResearchBieArtifactError(f"BIE73 {kind} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ResearchBieArtifactError(f"BIE73 {kind} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ResearchBieArtifactError(f"BIE73 {kind} must be a JSON object")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ResearchBieArtifactError(f"BIE73 {field} must be a SHA-256 value")
    return value


def _verify_checkpoint_receipt(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ResearchBieArtifactError("BIE73 checkpoint selection receipt is missing")
    step = value.get("selected_global_step")
    best = value.get("best_metric")
    replay = value.get("final_validation_replay_metric")
    hashes = value.get("safetensor_sha256")
    if (
        value.get("selection_split") != "independent_validation"
        or value.get("selection_metric") != "independent_dev_selection_score"
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 1
        or value.get("selected_checkpoint_id") != f"checkpoint-{step}"
        or isinstance(best, bool)
        or not isinstance(best, (int, float))
        or not math.isfinite(float(best))
        or isinstance(replay, bool)
        or not isinstance(replay, (int, float))
        or not math.isfinite(float(replay))
        or not math.isclose(float(best), float(replay), rel_tol=1.0e-6, abs_tol=1.0e-8)
        or not isinstance(hashes, Mapping)
        or not hashes
        or any(not isinstance(name, str) or not name.endswith(".safetensors") for name in hashes)
        or any(
            not isinstance(item, str) or _HEX_SHA256.fullmatch(item) is None
            for item in hashes.values()
        )
    ):
        raise ResearchBieArtifactError("BIE73 checkpoint selection receipt is invalid")


def _verify_manifest(manifest: dict[str, Any]) -> ResearchBieModelIdentity:
    label2id, _ = build_core_bie_label_maps()
    taxonomy = load_taxonomy()
    if not _REQUIRED_MANIFEST_FIELDS <= manifest.keys():
        raise ResearchBieArtifactError("BIE73 completed training manifest is incomplete")
    if (
        not verify_training_manifest(manifest)
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_type") != RESEARCH_BIE_MANIFEST_TYPE
        or manifest.get("status") != "completed"
        or manifest.get("release_eligible") is not False
        or manifest.get("attention_mode") != "causal"
        or manifest.get("tag_scheme") != "BIE"
        or manifest.get("update_mode") not in {"head_only", "lora"}
        or manifest.get("base_source_id") != AIGUARD_SOURCE_MODEL_ID
        or manifest.get("source_revision") != AIGUARD_SOURCE_REVISION
        or manifest.get("taxonomy_version") != taxonomy.taxonomy_version
        or manifest.get("label2id") != label2id
        or manifest.get("label_schema_sha256") != canonical_json_hash(label2id)
    ):
        raise ResearchBieArtifactError("BIE73 training manifest protocol is invalid")
    recipe = manifest.get("recipe")
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("attention_mode") != "causal"
        or recipe.get("tag_scheme") != "BIE"
        or recipe.get("label_count") != 73
        or recipe.get("initialization_strategy") != AIGUARD24_BIE_INITIALIZATION_STRATEGY
        or recipe.get("checkpoint_selection_split") != "independent_validation"
        or recipe.get("checkpoint_selection_metric") != "independent_dev_selection_score"
        or manifest.get("recipe_sha256") != canonical_json_hash(recipe)
    ):
        raise ResearchBieArtifactError("BIE73 training recipe receipt is invalid")
    initialization = manifest.get("initialization")
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("strategy") != AIGUARD24_BIE_INITIALIZATION_STRATEGY
        or initialization.get("source_model_id") != AIGUARD_SOURCE_MODEL_ID
        or initialization.get("source_revision") != AIGUARD_SOURCE_REVISION
        or initialization.get("target_tag_scheme") != "BIE"
        or initialization.get("target_label_count") != 73
        or initialization.get("target_label2id") != label2id
        or initialization.get("attention_mode") != "causal"
        or initialization.get("release_eligible") is not False
    ):
        raise ResearchBieArtifactError("BIE73 initialization receipt is invalid")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {"train", "independent_validation"}:
        raise ResearchBieArtifactError("BIE73 training dataset receipts are incomplete")
    for receipt in datasets.values():
        dataset_sha256 = receipt.get("sha256") if isinstance(receipt, Mapping) else None
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(dataset_sha256, str)
            or _HEX_SHA256.fullmatch(dataset_sha256) is None
            or not isinstance(receipt.get("summary"), Mapping)
            or not isinstance(receipt.get("admission"), Mapping)
        ):
            raise ResearchBieArtifactError("BIE73 training dataset receipt is invalid")
    isolation = manifest.get("split_isolation")
    expected_collisions = {
        "document_id": 0,
        "template_group": 0,
        "entity_value_group": 0,
        "source_group": 0,
    }
    if (
        not isinstance(isolation, Mapping)
        or isolation.get("policy") != "strict_all_groups"
        or isolation.get("collision_counts") != expected_collisions
        or isolation.get("template_group_overlap_allowed") is not False
    ):
        raise ResearchBieArtifactError("BIE73 split-isolation receipt is invalid")
    benchmark = manifest.get("benchmark_isolation")
    if not isinstance(benchmark, Mapping) or any(
        benchmark.get(field) is not False
        for field in (
            "public_test_read_for_training",
            "public_test_read_for_validation",
            "public_test_read_for_checkpoint_selection",
            "pii_bench_zh_read",
        )
    ):
        raise ResearchBieArtifactError("BIE73 benchmark-isolation receipt is invalid")
    privacy = manifest.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("contains_raw_text") is not False
        or privacy.get("contains_entity_values") is not False
        or privacy.get("trainer_reporting_integrations") != []
    ):
        raise ResearchBieArtifactError("BIE73 privacy receipt is invalid")
    independent_validation = manifest.get("independent_validation")
    if (
        not isinstance(independent_validation, Mapping)
        or independent_validation.get("eval_dev_gate_passed") != 1.0
    ):
        raise ResearchBieArtifactError("BIE73 independent-validation gate is incomplete")
    _verify_checkpoint_receipt(manifest.get("checkpoint_selection"))
    output = manifest.get("output_artifact")
    if not isinstance(output, Mapping):
        raise ResearchBieArtifactError("BIE73 output artifact receipt is missing")
    return ResearchBieModelIdentity(
        schema_version=RESEARCH_BIE_MODEL_IDENTITY_SCHEMA_VERSION,
        manifest_type=RESEARCH_BIE_MANIFEST_TYPE,
        manifest_sha256=_sha256(manifest.get("manifest_sha256"), field="manifest hash"),
        model_protocol=RESEARCH_BIE_MODEL_PROTOCOL,
        protocol_id=PROJECT_BIE_PROTOCOL_ID,
        model_type="qwen3",
        attention_mode="causal",
        tag_scheme="BIE",
        taxonomy_version=taxonomy.taxonomy_version,
        label_count=73,
        label_schema_sha256=_sha256(manifest.get("label_schema_sha256"), field="label schema hash"),
        weights_combined_sha256=_sha256(
            output.get("weights_combined_sha256"), field="combined weights hash"
        ),
    )


def _verify_config(config: Mapping[str, Any]) -> None:
    label2id, expected_id2label = build_core_bie_label_maps()
    try:
        id2label = normalize_project_bie_id2label(config.get("id2label", {}))
    except (TypeError, ValueError) as exc:
        raise ResearchBieArtifactError("BIE73 model config label schema is invalid") from exc
    num_labels = config.get("num_labels")
    if (
        config.get("model_type") != "qwen3"
        or config.get("architectures") != ["Qwen3ForTokenClassification"]
        or config.get("pii_attention_mode") != "causal"
        or config.get("pii_tagging_scheme") != "BIE"
        or config.get("pii_training_status") != "completed_independent_dev_feasible_candidate"
        or config.get("pii_taxonomy_version") != load_taxonomy().taxonomy_version
        or config.get("pii_release_eligible") is not False
        or config.get("use_cache") is not False
        or bool(config.get("auto_map"))
        or config.get("label2id") != label2id
        or id2label != expected_id2label
        or (num_labels is not None and num_labels != 73)
    ):
        raise ResearchBieArtifactError("BIE73 model config protocol is invalid")


def verify_research_bie_artifact(model_path: str | Path) -> VerifiedResearchBieModel:
    """Verify a complete, standalone BIE73 artifact without release fallback."""

    candidate = Path(model_path).expanduser()
    if candidate.is_symlink():
        raise ResearchBieArtifactError("BIE73 model root must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ResearchBieArtifactError("BIE73 model root must be a local directory") from exc
    if not root.is_dir():
        raise ResearchBieArtifactError("BIE73 model root must be a local directory")
    forbidden = [
        item.name
        for pattern in ("*.bin", "*.pkl", "*.pickle", "*.pt", "*.pth")
        for item in root.glob(pattern)
    ]
    if forbidden or any(root.glob("adapter_*")):
        raise ResearchBieArtifactError("BIE73 model root must be merged safetensors-only")
    for name in (
        "training_manifest.json",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ResearchBieArtifactError("BIE73 standalone artifact inventory is incomplete")
    manifest = _json_object(root / "training_manifest.json", kind="training manifest")
    identity = _verify_manifest(manifest)
    _verify_config(_json_object(root / "config.json", kind="model config"))
    if not verify_output_artifact_binding(manifest, root, require_single_weight=True):
        raise ResearchBieArtifactError("BIE73 manifest-to-safetensors binding failed")
    return VerifiedResearchBieModel(root=root, identity=identity)


def build_research_bie_closed8_pipeline(
    model_path: str | Path,
    *,
    mode: CascadeMode = "cascade",
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
    calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    rule_policy: ResearchBieRulePolicy = DEFAULT_RESEARCH_BIE_RULE_POLICY,
) -> CascadePipeline:
    """Build the fixed closed-8 BIE73 research cascade on verified v6 stages."""

    if mode != "cascade":
        raise ValueError("the research BIE73 profile supports cascade mode only")
    normalized_rule_policy = normalize_research_bie_rule_policy(rule_policy)
    verified = verify_research_bie_artifact(model_path)
    normalized_calibration = _normalize_calibration(calibration)
    normalized_thresholds = _normalize_thresholds(thresholds)
    config = load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode="cascade")
    if calibration is not None or thresholds is not None:
        config = replace(
            config,
            routes=_research_routes(
                recognizer_thresholds_authoritative=True,
                rule_policy=normalized_rule_policy,
            ),
        )
    loader_options: dict[str, Any] = {
        "device": device,
        "micro_batch_size": micro_batch_size,
        "allowed_project_labels": PROJECT_CLOSED8_LABELS,
    }
    if dtype is not None:
        loader_options["dtype"] = dtype
    predictor = load_local_project_bie_predictor(verified.root, **loader_options)
    if predictor.allowed_project_labels != frozenset(PROJECT_CLOSED8_LABELS):
        raise ResearchBieArtifactError("BIE73 predictor did not install the closed-8 mask")
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        calibration=normalized_calibration,
        thresholds=normalized_thresholds,
        max_tokens=512,
        stride_fraction=0.25,
        model_version=verified.root.name,
        attention_mode=predictor.attention_mode,
        deduplication_policy="high_overlap_v1",
    )
    model_identity: dict[str, str | int | bool | None] = {
        **verified.identity.to_dict(),
        "calibration_sha256": (
            service_policy_hash(normalized_calibration.to_dict())
            if normalized_calibration is not None
            else None
        ),
        "thresholds_sha256": (
            service_policy_hash(normalized_thresholds)
            if normalized_thresholds is not None
            else None
        ),
        "allowed_model_labels_sha256": service_policy_hash(PROJECT_CLOSED8_LABELS),
        "rule_policy": normalized_rule_policy,
    }
    return CascadePipeline(
        config=config,
        rule_recognizer=CnCommonRulePackV6(),
        model_recognizer=recognizer,
        validators=COMMUNITY_STRUCTURED_VALIDATORS,
        model_identity=model_identity,
    )


def _build_research_full_bie_pipeline(
    model_path: str | Path,
    *,
    mode: CascadeMode = "cascade",
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
    calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    rule_policy: ResearchBieRulePolicy = DEFAULT_RESEARCH_BIE_RULE_POLICY,
    decoder_id: str,
    allowed_project_labels: tuple[str, ...],
) -> CascadePipeline:
    """Build one strict full-attention BIE73 research cascade view.

    The decoder is deliberately part of the public constructor contract.  The
    research profile accepts only constrained Viterbi, and the exact Open-24
    inventory is installed before decoding rather than projected afterwards.
    """

    if mode != "cascade":
        raise ValueError("the research full-BIE73 profile supports cascade mode only")
    if decoder_id != RESEARCH_FULL_BIE_DECODER_ID:
        raise ValueError("the research full-BIE73 profile requires constrained_viterbi")
    if allowed_project_labels not in {PII_CORE_LABELS, PROJECT_CLOSED8_LABELS}:
        raise ValueError("the research full-BIE73 profile accepts only Open-24 or closed-8")
    normalized_rule_policy = normalize_research_bie_rule_policy(rule_policy)
    verified = verify_full_bie73_artifact(model_path)
    normalized_calibration = _normalize_calibration(calibration)
    normalized_thresholds = _normalize_thresholds(thresholds)
    config = load_service_config(COMMUNITY_MODEL_SERVICE_PROFILE_VERSION, mode="cascade")
    if calibration is not None or thresholds is not None:
        config = replace(
            config,
            routes=_research_routes(
                recognizer_thresholds_authoritative=True,
                rule_policy=normalized_rule_policy,
            ),
        )
    loader_options: dict[str, Any] = {
        "decoder_id": decoder_id,
        "device": device,
        "micro_batch_size": micro_batch_size,
        "allowed_project_labels": allowed_project_labels,
    }
    if dtype is not None:
        loader_options["dtype"] = dtype
    predictor = load_local_full_bie73_predictor(verified.root, **loader_options)
    expected_ids = project_bie_allowed_label_ids(predictor.id2label, allowed_project_labels)
    if (
        predictor.protocol_id != FULL_BIE73_PROTOCOL_ID
        or predictor.attention_mode != "full"
        or predictor.decoder_id != RESEARCH_FULL_BIE_DECODER_ID
        or predictor.allowed_project_labels != frozenset(allowed_project_labels)
        or tuple(predictor.allowed_label_ids) != expected_ids
    ):
        raise ResearchBieArtifactError(
            "full BIE73 predictor did not install the bound decoder and Open-24 mask"
        )
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        calibration=normalized_calibration,
        thresholds=normalized_thresholds,
        max_tokens=512,
        stride_fraction=0.25,
        model_version=verified.root.name,
        attention_mode=predictor.attention_mode,
        deduplication_policy="high_overlap_v1",
    )
    model_identity: dict[str, str | int | bool | None] = {
        **verified.identity.to_dict(),
        "calibration_sha256": (
            service_policy_hash(normalized_calibration.to_dict())
            if normalized_calibration is not None
            else None
        ),
        "thresholds_sha256": (
            service_policy_hash(normalized_thresholds)
            if normalized_thresholds is not None
            else None
        ),
        "allowed_model_labels_sha256": service_policy_hash(allowed_project_labels),
        "decoder_id": decoder_id,
        "model_scope": "open24" if allowed_project_labels == PII_CORE_LABELS else "closed8",
        "rule_policy": normalized_rule_policy,
    }
    if (
        model_identity.get("manifest_type") != FULL_BIE73_MANIFEST_TYPE
        or model_identity.get("model_protocol") != FULL_BIE73_MODEL_PROTOCOL
    ):
        raise ResearchBieArtifactError("full BIE73 verifier returned a different protocol")
    return CascadePipeline(
        config=config,
        rule_recognizer=CnCommonRulePackV6(),
        model_recognizer=recognizer,
        validators=COMMUNITY_STRUCTURED_VALIDATORS,
        model_identity=model_identity,
    )


def build_research_full_bie_open24_pipeline(
    model_path: str | Path,
    *,
    mode: CascadeMode = "cascade",
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
    calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    rule_policy: ResearchBieRulePolicy = DEFAULT_RESEARCH_BIE_RULE_POLICY,
    decoder_id: str,
) -> CascadePipeline:
    """Build the strict full-attention Open-24 BIE73 research cascade."""

    return _build_research_full_bie_pipeline(
        model_path,
        mode=mode,
        device=device,
        dtype=dtype,
        micro_batch_size=micro_batch_size,
        calibration=calibration,
        thresholds=thresholds,
        rule_policy=rule_policy,
        decoder_id=decoder_id,
        allowed_project_labels=PII_CORE_LABELS,
    )


def build_research_full_bie_closed8_pipeline(
    model_path: str | Path,
    *,
    mode: CascadeMode = "cascade",
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
    calibration: CalibrationBundle | Mapping[str, Any] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    rule_policy: ResearchBieRulePolicy = DEFAULT_RESEARCH_BIE_RULE_POLICY,
    decoder_id: str,
) -> CascadePipeline:
    """Build the closed-8 pre-decode view of the same strict full-BIE73 artifact."""

    return _build_research_full_bie_pipeline(
        model_path,
        mode=mode,
        device=device,
        dtype=dtype,
        micro_batch_size=micro_batch_size,
        calibration=calibration,
        thresholds=thresholds,
        rule_policy=rule_policy,
        decoder_id=decoder_id,
        allowed_project_labels=PROJECT_CLOSED8_LABELS,
    )


__all__ = [
    "ADAPTIVE_RESEARCH_BIE_RULE_POLICIES",
    "ALL_RESEARCH_BIE_RULE_POLICIES",
    "COMMUNITY_CALIBRATED_RULE_SUPPRESSION",
    "DEFAULT_RESEARCH_BIE_RULE_POLICY",
    "FPR_GUARDED_ALL6",
    "FPR_GUARDED_ALL6_RULE_DISABLED_ENTITIES",
    "FPR_GUARDED_WEAK5",
    "FPR_GUARDED_WEAK5_ADDITIONAL_RULE_DISABLED_ENTITIES",
    "FPR_GUARDED_WEAK5_RULE_DISABLED_ENTITIES",
    "FULL_BIE73_MANIFEST_TYPE",
    "FULL_BIE73_MODEL_PROTOCOL",
    "FULL_BIE73_PROTOCOL_ID",
    "MODEL_ONLY_RULE_DISABLED_ENTITIES",
    "MODEL_ONLY_RULES_DISABLED",
    "PRESERVE_ALL_VALIDATED_RULES",
    "RESEARCH_BIE_MANIFEST_TYPE",
    "RESEARCH_BIE_MODEL_IDENTITY_SCHEMA_VERSION",
    "RESEARCH_BIE_MODEL_PROTOCOL",
    "RESEARCH_BIE_RULE_POLICIES",
    "RESEARCH_FULL_BIE_DECODER_ID",
    "ResearchBieArtifactError",
    "ResearchBieModelIdentity",
    "ResearchBieRulePolicy",
    "VerifiedResearchBieModel",
    "build_research_bie_closed8_pipeline",
    "build_research_full_bie_open24_pipeline",
    "build_research_full_bie_closed8_pipeline",
    "normalize_research_bie_rule_policy",
    "research_bie_disabled_rule_entities",
    "verify_research_bie_artifact",
]
