"""Stable, lazy-loading public helpers for the full-attention BIE73 model.

The implementation intentionally delegates inference and cascade construction
to the manifest-bound full-BIE research runtime.  Importing this module itself
loads only Python's standard library; PyTorch, Transformers, Presidio and the
project runtime are imported only when a builder is called.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

FullBie73Scope = Literal["open24", "closed8"]
FullBie73ServiceMode = Literal["model-only", "cascade"]
FullBie73AdaptiveRulePolicy = Literal["fpr_guarded_weak5", "fpr_guarded_all6"]

COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION = "community-presidio-bie73-cascade-v1"
FULL_BIE73_DECODER_ID = "constrained_viterbi"
FULL_BIE73_PUBLIC_PROFILE_SCHEMA_VERSION = "pii-zh.full-bie73-public-profile.v2"
FULL_BIE73_PUBLIC_PROFILE_VERSION = COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION
FULL_BIE73_THRESHOLD_POLICY = "core24-fixed-0.5-v1"
FULL_BIE73_FIXED_THRESHOLD = 0.5
FULL_BIE73_MODEL_ONLY_RULE_POLICY = "model_only_rules_disabled"
FULL_BIE73_DEFAULT_RULE_POLICY: FullBie73AdaptiveRulePolicy = "fpr_guarded_all6"
FULL_BIE73_DEFAULT_SCOPE: FullBie73Scope = "open24"
FULL_BIE73_DEFAULT_MODE: FullBie73ServiceMode = "cascade"
FULL_BIE73_PRIMARY_PRESIDIO_VERSION = "2.2.363"
FULL_BIE73_PRIMARY_MODEL_ID = "uer/roberta-base-finetuned-cluener2020-chinese"
FULL_BIE73_PRIMARY_MODEL_REVISION = "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b"
FULL_BIE73_SELECTED_CANDIDATE_ID = "b3-v2-t050-fpr-guarded-all6"
FULL_BIE73_SELECTION_STATUS = "SELECTED_DEVELOPMENT_CASCADE_POLICY_NOT_BENCHMARK_EVALUATED"
FULL_BIE73_SELECTION_RECEIPT_FILE_SHA256 = (
    "28b37a0bfb2b6e902a76297300b2bd47fbe3be7569c0dedc7b43e4210a5802e2"
)
FULL_BIE73_SELECTION_RECEIPT_SHA256 = (
    "eeaa0f80593f5a661ed72ecc3e21292cedf19819c52cce645fbd0db9855cfd37"
)
FULL_BIE73_SELECTION_RECEIPT_PHYSICAL_SHA256 = FULL_BIE73_SELECTION_RECEIPT_FILE_SHA256
FULL_BIE73_SELECTION_RECEIPT_LOGICAL_SHA256 = FULL_BIE73_SELECTION_RECEIPT_SHA256
FULL_BIE73_ADAPTIVE_RULE_POLICIES: tuple[FullBie73AdaptiveRulePolicy, ...] = (
    "fpr_guarded_weak5",
    "fpr_guarded_all6",
)
_FULL_BIE73_SCOPES: tuple[FullBie73Scope, ...] = ("open24", "closed8")
_FULL_BIE73_SERVICE_MODES: tuple[FullBie73ServiceMode, ...] = ("model-only", "cascade")


class FullBie73ProfileError(ValueError):
    """Raised when the stable public BIE73 profile cannot be reproduced."""


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_scope(value: str) -> FullBie73Scope:
    if not isinstance(value, str):
        raise TypeError("scope must be a string")
    if value not in _FULL_BIE73_SCOPES:
        raise FullBie73ProfileError("scope must be 'open24' or 'closed8'")
    return value


def _normalize_service_mode(value: str) -> FullBie73ServiceMode:
    if not isinstance(value, str):
        raise TypeError("mode must be a string")
    if value not in _FULL_BIE73_SERVICE_MODES:
        raise FullBie73ProfileError("mode must be 'model-only' or 'cascade'")
    return value


def _normalize_rule_policy(value: str) -> FullBie73AdaptiveRulePolicy:
    if not isinstance(value, str):
        raise TypeError("rule_policy must be a string")
    if value not in FULL_BIE73_ADAPTIVE_RULE_POLICIES:
        raise FullBie73ProfileError("rule_policy must be an evaluated adaptive full-BIE73 policy")
    return value


def _positive_micro_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("micro_batch_size must be an integer")
    if value < 1:
        raise FullBie73ProfileError("micro_batch_size must be positive")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FullBie73ProfileError("threshold JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_json_constant(_: str) -> None:
    raise FullBie73ProfileError("threshold JSON contains a non-finite number")


def _threshold_mapping(
    value: Mapping[str, float] | str | Path,
    *,
    open24_labels: Sequence[str],
) -> dict[str, float]:
    labels = tuple(open24_labels)
    if (
        len(labels) != 24
        or len(set(labels)) != 24
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise RuntimeError("the installed runtime does not expose the canonical Open-24 scope")

    if isinstance(value, Mapping):
        raw: object = dict(value)
    elif isinstance(value, (str, Path)):
        candidate = Path(value).expanduser()
        if candidate.is_symlink():
            raise FullBie73ProfileError("thresholds must be a non-symlink local JSON file")
        try:
            path = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FullBie73ProfileError("thresholds must be an existing local JSON file") from exc
        if not path.is_file():
            raise FullBie73ProfileError("thresholds must be an existing local JSON file")
        try:
            raw = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except FullBie73ProfileError:
            raise
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise FullBie73ProfileError("thresholds file is invalid JSON") from exc
    else:
        raise TypeError("thresholds must be a mapping or local JSON path")

    if not isinstance(raw, Mapping):
        raise FullBie73ProfileError("thresholds must be a JSON object")
    if set(raw) != set(labels) or len(raw) != 24:
        raise FullBie73ProfileError("thresholds must cover the exact canonical Open-24 labels")
    for key, item in raw.items():
        if not isinstance(key, str):
            raise FullBie73ProfileError("threshold labels must be strings")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("threshold values must be numeric")
        number = float(item)
        if not math.isfinite(number) or number != FULL_BIE73_FIXED_THRESHOLD:
            raise FullBie73ProfileError("all Open-24 thresholds must equal 0.5")
    return {label: FULL_BIE73_FIXED_THRESHOLD for label in labels}


def _load_predictor_runtime() -> dict[str, Any]:
    from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
    from pii_zh.inference.project_bie import (
        PROJECT_CLOSED8_LABELS,
        project_bie_allowed_label_ids,
    )
    from pii_zh.inference.project_full_bie import (
        FULL_BIE73_MODEL_PROTOCOL,
        FULL_BIE73_PROTOCOL_ID,
        load_local_full_bie73_predictor,
    )

    return {
        "closed8_labels": PROJECT_CLOSED8_LABELS,
        "loader": load_local_full_bie73_predictor,
        "model_protocol": FULL_BIE73_MODEL_PROTOCOL,
        "open24_labels": PII_CORE_LABELS,
        "project_allowed_label_ids": project_bie_allowed_label_ids,
        "protocol_id": FULL_BIE73_PROTOCOL_ID,
    }


def _load_service_runtime() -> dict[str, Any]:
    from pii_zh.cascade.research_bie_profile import (
        ADAPTIVE_RESEARCH_BIE_RULE_POLICIES,
        FULL_BIE73_MANIFEST_TYPE,
        FULL_BIE73_MODEL_PROTOCOL,
        FULL_BIE73_PROTOCOL_ID,
        MODEL_ONLY_RULES_DISABLED,
        RESEARCH_FULL_BIE_DECODER_ID,
        build_research_full_bie_closed8_pipeline,
        build_research_full_bie_open24_pipeline,
    )
    from pii_zh.cascade.service_profiles import _normalize_thresholds
    from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
    from pii_zh.inference.project_bie import PROJECT_CLOSED8_LABELS

    return {
        "adaptive_rule_policies": ADAPTIVE_RESEARCH_BIE_RULE_POLICIES,
        "closed8_builder": build_research_full_bie_closed8_pipeline,
        "closed8_labels": PROJECT_CLOSED8_LABELS,
        "decoder_id": RESEARCH_FULL_BIE_DECODER_ID,
        "manifest_type": FULL_BIE73_MANIFEST_TYPE,
        "model_protocol": FULL_BIE73_MODEL_PROTOCOL,
        "model_only_rule_policy": MODEL_ONLY_RULES_DISABLED,
        "open24_builder": build_research_full_bie_open24_pipeline,
        "open24_labels": PII_CORE_LABELS,
        "protocol_id": FULL_BIE73_PROTOCOL_ID,
        "threshold_normalizer": _normalize_thresholds,
    }


def _runtime_labels(runtime: Mapping[str, Any], scope: FullBie73Scope) -> tuple[str, ...]:
    open24 = tuple(runtime.get("open24_labels", ()))
    closed8 = tuple(runtime.get("closed8_labels", ()))
    if (
        len(open24) != 24
        or len(set(open24)) != 24
        or len(closed8) != 8
        or len(set(closed8)) != 8
        or not set(closed8) < set(open24)
    ):
        raise RuntimeError("the installed BIE73 runtime label scopes changed")
    return open24 if scope == "open24" else closed8


def load_full_bie73_predictor(
    model_path: str | Path,
    *,
    scope: FullBie73Scope,
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
) -> Any:
    """Load one verified full-attention BIE73 predictor with a fixed decoder.

    ``scope`` is applied before BIE decoding, so ``closed8`` is not a
    post-processing projection of Open-24 predictions.
    """

    selected_scope = _normalize_scope(scope)
    selected_batch_size = _positive_micro_batch_size(micro_batch_size)
    runtime = _load_predictor_runtime()
    labels = _runtime_labels(runtime, selected_scope)
    loader = runtime.get("loader")
    allowed_ids = runtime.get("project_allowed_label_ids")
    if not callable(loader) or not callable(allowed_ids):
        raise RuntimeError("the installed full-BIE73 predictor runtime is incomplete")
    options: dict[str, Any] = {
        "decoder_id": FULL_BIE73_DECODER_ID,
        "device": device,
        "micro_batch_size": selected_batch_size,
        "allowed_project_labels": labels,
    }
    if dtype is not None:
        options["dtype"] = dtype
    predictor = loader(model_path, **options)
    try:
        expected_ids = tuple(allowed_ids(predictor.id2label, labels))
        actual_ids = tuple(predictor.allowed_label_ids)
        actual_labels = frozenset(predictor.allowed_project_labels)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("the loaded predictor does not expose the BIE73 contract") from exc
    if (
        runtime.get("model_protocol") != "FULL_BIE73"
        or predictor.protocol_id != runtime.get("protocol_id")
        or predictor.attention_mode != "full"
        or predictor.decoder_id != FULL_BIE73_DECODER_ID
        or actual_labels != frozenset(labels)
        or actual_ids != expected_ids
    ):
        raise RuntimeError("the loaded predictor differs from the public BIE73 profile")
    return predictor


def _install_public_profile_identity(
    pipeline: Any,
    *,
    runtime: Mapping[str, Any],
    scope: FullBie73Scope,
    labels: tuple[str, ...],
    thresholds: Mapping[str, float],
    rule_policy: str,
    mode: FullBie73ServiceMode,
) -> Any:
    current = getattr(pipeline, "model_identity", None)
    if not isinstance(current, Mapping):
        raise RuntimeError("the full-BIE73 service builder did not expose model identity")
    identity = dict(current)
    expected_thresholds_sha256 = _canonical_json_hash(dict(thresholds))
    if (
        identity.get("manifest_type") != runtime.get("manifest_type")
        or identity.get("model_protocol") != runtime.get("model_protocol")
        or identity.get("protocol_id") != runtime.get("protocol_id")
        or identity.get("attention_mode") != "full"
        or identity.get("tag_scheme") != "BIE"
        or identity.get("label_count") != 73
        or identity.get("decoder_id") != FULL_BIE73_DECODER_ID
        or identity.get("model_scope") != scope
        or identity.get("rule_policy") != rule_policy
        or identity.get("thresholds_sha256") != expected_thresholds_sha256
        or identity.get("allowed_model_labels_sha256") != _canonical_json_hash(labels)
    ):
        raise RuntimeError("the service builder returned a different BIE73 profile identity")

    identity.update(
        {
            "public_profile_schema_version": FULL_BIE73_PUBLIC_PROFILE_SCHEMA_VERSION,
            "public_profile_version": FULL_BIE73_PUBLIC_PROFILE_VERSION,
            "service_mode": mode,
            "profile_purpose": (
                "model_only_service_ablation_with_validators"
                if mode == "model-only"
                else "adaptive_rule_model_cascade"
            ),
            "validators_enabled": True,
            "raw_model_benchmark_equivalent": False,
            "threshold_policy": FULL_BIE73_THRESHOLD_POLICY,
            "selected_rule_policy": FULL_BIE73_DEFAULT_RULE_POLICY,
            "selected_scope": FULL_BIE73_DEFAULT_SCOPE,
            "selected_candidate_id": FULL_BIE73_SELECTED_CANDIDATE_ID,
            "selection_status": FULL_BIE73_SELECTION_STATUS,
            "selection_receipt_file_sha256": FULL_BIE73_SELECTION_RECEIPT_FILE_SHA256,
            "selection_receipt_sha256": FULL_BIE73_SELECTION_RECEIPT_SHA256,
            "selection_receipt_physical_sha256": (FULL_BIE73_SELECTION_RECEIPT_PHYSICAL_SHA256),
            "selection_receipt_logical_sha256": (FULL_BIE73_SELECTION_RECEIPT_LOGICAL_SHA256),
            "matches_selected_cascade": (
                mode == FULL_BIE73_DEFAULT_MODE
                and scope == FULL_BIE73_DEFAULT_SCOPE
                and rule_policy == FULL_BIE73_DEFAULT_RULE_POLICY
            ),
        }
    )
    identity["public_profile_identity_sha256"] = _canonical_json_hash(identity)
    # Only identity metadata changes.  Returning the same pipeline object keeps
    # recognizers, routes, validators, stage ordering and fusion byte-for-byte
    # aligned with the evaluated research builder.
    pipeline.model_identity = MappingProxyType(identity)
    return pipeline


def _adapt_public_service_pipeline(
    pipeline: Any,
    *,
    mode: FullBie73ServiceMode,
) -> Any:
    """Change only public mode metadata and the rule execution switch.

    The research builder owns recognizer construction, validators, routes,
    stage ordering and deterministic fusion.  Replacing the frozen config
    dataclass keeps those objects intact while ``mode='model-only'`` prevents
    the already-disabled rule recognizer from entering the execution path.
    """

    config = getattr(pipeline, "config", None)
    model_recognizer = getattr(pipeline, "model_recognizer", None)
    validators = getattr(pipeline, "validators", None)
    fusion = getattr(pipeline, "_fusion", None)
    if (
        config is None
        or getattr(config, "mode", None) != "cascade"
        or model_recognizer is None
        or not isinstance(validators, Mapping)
        or fusion is None
    ):
        raise RuntimeError("the research builder returned an incomplete cascade pipeline")
    try:
        public_config = replace(
            config,
            profile_version=FULL_BIE73_PUBLIC_PROFILE_VERSION,
            mode=mode,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("the research builder returned an incompatible cascade config") from exc
    if (
        getattr(public_config, "mode", None) != mode
        or getattr(public_config, "profile_version", None) != FULL_BIE73_PUBLIC_PROFILE_VERSION
    ):
        raise RuntimeError("the public BIE73 service config adaptation failed")
    pipeline.config = public_config
    if mode == "model-only":
        pipeline.rule_recognizer = None
    if (
        pipeline.model_recognizer is not model_recognizer
        or pipeline.validators is not validators
        or pipeline._fusion is not fusion
    ):
        raise RuntimeError("the public service adaptation changed evaluated runtime components")
    return pipeline


def _build_bie73_augmentation_pipeline(
    model_path: str | Path,
    *,
    device: str,
    dtype: Any | None,
    micro_batch_size: int,
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    """Build the model branch with only Open-24 labels not owned by primary.

    The label restriction is passed to the predictor before constrained BIE
    decoding.  It is therefore impossible for this branch to emit a closed-8
    label and later have it hidden by post-processing.
    """

    from dataclasses import replace as dataclass_replace

    from pii_zh.cascade.config import CascadeConfig
    from pii_zh.cascade.pipeline import CascadePipeline
    from pii_zh.cascade.routing import community_full24_routes
    from pii_zh.data.validators import COMMUNITY_STRUCTURED_VALIDATORS
    from pii_zh.evaluation.service_quality_suite import PII_CORE_LABELS
    from pii_zh.inference.project_bie import (
        PROJECT_CLOSED8_LABELS,
        project_bie_allowed_label_ids,
    )
    from pii_zh.inference.project_full_bie import (
        FULL_BIE73_PROTOCOL_ID,
        load_local_full_bie73_predictor,
        verify_full_bie73_artifact,
    )
    from pii_zh.presidio import QwenPiiRecognizer
    from pii_zh.taxonomy import load_presidio_mapping

    open24 = tuple(PII_CORE_LABELS)
    closed8 = tuple(PROJECT_CLOSED8_LABELS)
    augmentation = tuple(label for label in open24 if label not in set(closed8))
    if len(open24) != 24 or len(closed8) != 8 or len(augmentation) != 16:
        raise RuntimeError("the installed Open-24 ownership contract changed")

    verified = verify_full_bie73_artifact(model_path)
    loader_options: dict[str, Any] = {
        "decoder_id": FULL_BIE73_DECODER_ID,
        "device": device,
        "micro_batch_size": micro_batch_size,
        "allowed_project_labels": augmentation,
    }
    if dtype is not None:
        loader_options["dtype"] = dtype
    predictor = load_local_full_bie73_predictor(verified.root, **loader_options)
    expected_ids = project_bie_allowed_label_ids(predictor.id2label, augmentation)
    if (
        predictor.protocol_id != FULL_BIE73_PROTOCOL_ID
        or predictor.attention_mode != "full"
        or predictor.decoder_id != FULL_BIE73_DECODER_ID
        or predictor.allowed_project_labels != frozenset(augmentation)
        or tuple(predictor.allowed_label_ids) != expected_ids
    ):
        raise RuntimeError("the BIE73 augmentation pre-decode mask was not installed")

    mapping = load_presidio_mapping().model_to_presidio
    label_mapping = {label: mapping[label] for label in augmentation}
    canonical_augmentation = tuple(label_mapping[label] for label in augmentation)
    canonical_set = set(canonical_augmentation)
    routes = tuple(
        dataclass_replace(route, rule_enabled=False)
        for route in community_full24_routes(recognizer_thresholds_authoritative=True)
        if route.entity_type in canonical_set
    )
    if len(routes) != 16 or {route.entity_type for route in routes} != canonical_set:
        raise RuntimeError("the installed service routes do not cover augmentation-16")

    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        label_mapping=label_mapping,
        ignored_labels=(),
        default_threshold=FULL_BIE73_FIXED_THRESHOLD,
        max_tokens=512,
        stride_fraction=0.25,
        model_version=verified.root.name,
        attention_mode=predictor.attention_mode,
        deduplication_policy="high_overlap_v1",
        name="Bie73Open24AugmentationRecognizer",
    )
    config = CascadeConfig(
        profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        mode="model-only",
        routes=routes,
        rule_source="presidio:primary-disabled-in-augmentation",
        model_source="qwen:bie73-augmentation16",
        keep_nested_different_types=False,
    )
    model_identity: dict[str, str | int | bool | None] = {
        **verified.identity.to_dict(),
        "decoder_id": FULL_BIE73_DECODER_ID,
        "model_scope": "open24_minus_closed8",
        "allowed_model_labels_sha256": _canonical_json_hash(augmentation),
        "allowed_model_label_count": len(augmentation),
        "threshold_policy": FULL_BIE73_THRESHOLD_POLICY,
    }
    return (
        CascadePipeline(
            config=config,
            rule_recognizer=None,
            model_recognizer=recognizer,
            validators=COMMUNITY_STRUCTURED_VALIDATORS,
            model_identity=model_identity,
        ),
        closed8,
        augmentation,
    )


def _build_presidio_primary_service(
    model_path: str | Path,
    *,
    primary_model_path: str | Path,
    device: str,
    dtype: Any | None,
    micro_batch_size: int,
) -> Any:
    from pii_zh.cascade.config import CascadeConfig
    from pii_zh.cascade.primary_augmentation import (
        PrimaryPreservingAugmentationPipeline,
    )
    from pii_zh.presidio.cluener_primary import (
        CLUENER_PRIMARY_MODEL_ID,
        CLUENER_PRIMARY_MODEL_REVISION,
        CLUENER_PRIMARY_PRESIDIO_VERSION,
        CLUENER_PRIMARY_SOURCE,
        build_cluener_primary_pipeline,
    )
    from pii_zh.taxonomy import load_presidio_mapping

    primary = build_cluener_primary_pipeline(
        primary_model_path,
        device=device,
        micro_batch_size=micro_batch_size,
    )
    augmentation, closed8, augmentation16 = _build_bie73_augmentation_pipeline(
        model_path,
        device=device,
        dtype=dtype,
        micro_batch_size=micro_batch_size,
    )
    mapping = load_presidio_mapping().model_to_presidio
    canonical_closed8 = tuple(mapping[label] for label in closed8)
    canonical_augmentation = tuple(mapping[label] for label in augmentation16)
    routes = tuple(
        route
        for route in (*primary.config.routes, *augmentation.config.routes)
        if route.entity_type in set((*canonical_closed8, *canonical_augmentation))
    )
    if len(routes) != 24 or len({route.entity_type for route in routes}) != 24:
        raise RuntimeError("the combined service does not own exactly Open-24")
    config = CascadeConfig(
        profile_version=COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
        mode="cascade",
        routes=routes,
        rule_source=CLUENER_PRIMARY_SOURCE,
        model_source="qwen:bie73-augmentation16",
        keep_nested_different_types=False,
    )
    primary_identity = getattr(primary, "model_identity", None)
    model_identity = getattr(augmentation, "model_identity", None)
    if not isinstance(primary_identity, Mapping) or not isinstance(model_identity, Mapping):
        raise RuntimeError("the combined service branches do not expose identity")
    if (
        CLUENER_PRIMARY_MODEL_ID != FULL_BIE73_PRIMARY_MODEL_ID
        or CLUENER_PRIMARY_MODEL_REVISION != FULL_BIE73_PRIMARY_MODEL_REVISION
        or CLUENER_PRIMARY_PRESIDIO_VERSION != FULL_BIE73_PRIMARY_PRESIDIO_VERSION
        or primary_identity.get("source_model_id") != FULL_BIE73_PRIMARY_MODEL_ID
        or primary_identity.get("source_revision") != FULL_BIE73_PRIMARY_MODEL_REVISION
        or primary_identity.get("presidio_analyzer_version")
        != FULL_BIE73_PRIMARY_PRESIDIO_VERSION
        or primary_identity.get("rule_pack_id") != "cn_common_v5"
        or not isinstance(primary_identity.get("identity_sha256"), str)
        or not isinstance(primary_identity.get("model_safetensors_sha256"), str)
        or not isinstance(model_identity.get("manifest_sha256"), str)
        or not isinstance(model_identity.get("weights_combined_sha256"), str)
    ):
        raise RuntimeError("the Presidio primary identity does not match the public profile")
    identity: dict[str, str | int | bool | None] = {
        "public_profile_schema_version": FULL_BIE73_PUBLIC_PROFILE_SCHEMA_VERSION,
        "public_profile_version": FULL_BIE73_PUBLIC_PROFILE_VERSION,
        "profile_purpose": "presidio_primary_bie73_augmentation",
        "service_mode": "cascade",
        "primary_outputs_preserved": True,
        "primary_overlap_wins": True,
        "primary_label_count": 8,
        "primary_labels_sha256": _canonical_json_hash(canonical_closed8),
        "augmentation_label_count": 16,
        "augmentation_labels_sha256": _canonical_json_hash(canonical_augmentation),
        "augmentation_predecode_masked": True,
        "primary_model_id": FULL_BIE73_PRIMARY_MODEL_ID,
        "primary_model_revision": FULL_BIE73_PRIMARY_MODEL_REVISION,
        "presidio_version": FULL_BIE73_PRIMARY_PRESIDIO_VERSION,
        "primary_artifact_identity_sha256": primary_identity["identity_sha256"],
        "primary_model_safetensors_sha256": primary_identity["model_safetensors_sha256"],
        "bie73_manifest_sha256": model_identity["manifest_sha256"],
        "bie73_weights_sha256": model_identity["weights_combined_sha256"],
        "bie73_decoder_id": FULL_BIE73_DECODER_ID,
        "threshold_policy": FULL_BIE73_THRESHOLD_POLICY,
        "raw_model_benchmark_equivalent": False,
    }
    identity["public_profile_identity_sha256"] = _canonical_json_hash(identity)
    return PrimaryPreservingAugmentationPipeline(
        primary=primary,
        augmentation=augmentation,
        config=config,
        model_identity=identity,
    )


def build_full_bie73_service_pipeline(
    model_path: str | Path,
    *,
    primary_model_path: str | Path | None = None,
    thresholds: Mapping[str, float] | str | Path | None = None,
    scope: FullBie73Scope = FULL_BIE73_DEFAULT_SCOPE,
    mode: FullBie73ServiceMode = FULL_BIE73_DEFAULT_MODE,
    rule_policy: FullBie73AdaptiveRulePolicy | None = None,
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
) -> Any:
    """Build the public Presidio-primary service or a BIE73-only ablation.

    Cascade mode requires a second, local CLUENER checkpoint.  Native
    Presidio exclusively owns the frozen closed-8 labels; BIE73 is restricted
    before decoding to the other sixteen Open-24 labels.  Primary detections
    are returned unchanged and win every cross-branch overlap.  Model-only is
    retained as an explicit service ablation and is not the raw benchmark
    protocol.  No construction failure silently falls back to another mode.
    """

    selected_scope = _normalize_scope(scope)
    selected_mode = _normalize_service_mode(mode)
    if selected_mode == "cascade":
        if primary_model_path is None:
            raise FullBie73ProfileError(
                "cascade mode requires primary_model_path for the pinned CLUENER model"
            )
        if selected_scope != "open24":
            raise FullBie73ProfileError(
                "Presidio-primary cascade mode has the fixed open24 output scope"
            )
        if rule_policy is not None:
            raise FullBie73ProfileError(
                "Presidio-primary cascade mode does not accept a research rule_policy"
            )
        if thresholds is not None:
            raise FullBie73ProfileError(
                "Presidio-primary cascade mode uses its frozen thresholds"
            )
        selected_batch_size = _positive_micro_batch_size(micro_batch_size)
        return _build_presidio_primary_service(
            model_path,
            primary_model_path=primary_model_path,
            device=device,
            dtype=dtype,
            micro_batch_size=selected_batch_size,
        )
    else:
        if primary_model_path is not None:
            raise FullBie73ProfileError(
                "primary_model_path is only valid in Presidio-primary cascade mode"
            )
        if rule_policy is not None:
            raise FullBie73ProfileError(
                "model-only mode selects its internal evaluated rule policy automatically"
            )
        selected_policy = FULL_BIE73_MODEL_ONLY_RULE_POLICY
    selected_batch_size = _positive_micro_batch_size(micro_batch_size)
    runtime = _load_service_runtime()
    if (
        tuple(runtime.get("adaptive_rule_policies", ())) != FULL_BIE73_ADAPTIVE_RULE_POLICIES
        or runtime.get("model_only_rule_policy") != FULL_BIE73_MODEL_ONLY_RULE_POLICY
        or runtime.get("decoder_id") != FULL_BIE73_DECODER_ID
        or runtime.get("model_protocol") != "FULL_BIE73"
    ):
        raise RuntimeError("the installed research runtime changed the adaptive BIE73 contract")
    labels = _runtime_labels(runtime, selected_scope)
    open24_labels = tuple(runtime.get("open24_labels", ()))
    threshold_input = (
        {label: FULL_BIE73_FIXED_THRESHOLD for label in open24_labels}
        if thresholds is None
        else thresholds
    )
    validated_thresholds = _threshold_mapping(
        threshold_input,
        open24_labels=open24_labels,
    )
    threshold_normalizer = runtime.get("threshold_normalizer")
    if not callable(threshold_normalizer):
        raise RuntimeError("the installed full-BIE73 threshold runtime is incomplete")
    normalized_thresholds = threshold_normalizer(validated_thresholds)
    if (
        not isinstance(normalized_thresholds, Mapping)
        or len(normalized_thresholds) != 24
        or any(value != FULL_BIE73_FIXED_THRESHOLD for value in normalized_thresholds.values())
    ):
        raise RuntimeError("the installed runtime changed Open-24 threshold normalization")
    normalized_thresholds = dict(normalized_thresholds)
    builder_key = "open24_builder" if selected_scope == "open24" else "closed8_builder"
    builder = runtime.get(builder_key)
    if not callable(builder):
        raise RuntimeError("the installed full-BIE73 service runtime is incomplete")
    options: dict[str, Any] = {
        "mode": "cascade",
        "device": device,
        "micro_batch_size": selected_batch_size,
        "thresholds": normalized_thresholds,
        "rule_policy": selected_policy,
        "decoder_id": FULL_BIE73_DECODER_ID,
    }
    if dtype is not None:
        options["dtype"] = dtype
    pipeline = _adapt_public_service_pipeline(
        builder(model_path, **options),
        mode=selected_mode,
    )
    return _install_public_profile_identity(
        pipeline,
        runtime=runtime,
        scope=selected_scope,
        labels=labels,
        thresholds=normalized_thresholds,
        rule_policy=selected_policy,
        mode=selected_mode,
    )


__all__ = [
    "COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION",
    "FULL_BIE73_ADAPTIVE_RULE_POLICIES",
    "FULL_BIE73_DECODER_ID",
    "FULL_BIE73_DEFAULT_MODE",
    "FULL_BIE73_DEFAULT_RULE_POLICY",
    "FULL_BIE73_DEFAULT_SCOPE",
    "FULL_BIE73_FIXED_THRESHOLD",
    "FULL_BIE73_MODEL_ONLY_RULE_POLICY",
    "FULL_BIE73_PRIMARY_MODEL_ID",
    "FULL_BIE73_PRIMARY_MODEL_REVISION",
    "FULL_BIE73_PRIMARY_PRESIDIO_VERSION",
    "FULL_BIE73_PUBLIC_PROFILE_SCHEMA_VERSION",
    "FULL_BIE73_PUBLIC_PROFILE_VERSION",
    "FULL_BIE73_SELECTED_CANDIDATE_ID",
    "FULL_BIE73_SELECTION_RECEIPT_FILE_SHA256",
    "FULL_BIE73_SELECTION_RECEIPT_LOGICAL_SHA256",
    "FULL_BIE73_SELECTION_RECEIPT_PHYSICAL_SHA256",
    "FULL_BIE73_SELECTION_RECEIPT_SHA256",
    "FULL_BIE73_SELECTION_STATUS",
    "FULL_BIE73_THRESHOLD_POLICY",
    "FullBie73AdaptiveRulePolicy",
    "FullBie73ProfileError",
    "FullBie73Scope",
    "FullBie73ServiceMode",
    "build_full_bie73_service_pipeline",
    "load_full_bie73_predictor",
]
