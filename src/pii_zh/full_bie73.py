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

FULL_BIE73_DECODER_ID = "constrained_viterbi"
FULL_BIE73_PUBLIC_PROFILE_SCHEMA_VERSION = "pii-zh.full-bie73-public-profile.v1"
FULL_BIE73_PUBLIC_PROFILE_VERSION = "full-bie73-service-v1"
FULL_BIE73_THRESHOLD_POLICY = "core24-fixed-0.5-v1"
FULL_BIE73_MODEL_ONLY_RULE_POLICY = "model_only_rules_disabled"
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
        if not math.isfinite(number) or number != 0.5:
            raise FullBie73ProfileError("all Open-24 thresholds must equal 0.5")
    return {label: 0.5 for label in labels}


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


def build_full_bie73_service_pipeline(
    model_path: str | Path,
    *,
    thresholds: Mapping[str, float] | str | Path,
    scope: FullBie73Scope,
    mode: FullBie73ServiceMode,
    rule_policy: FullBie73AdaptiveRulePolicy | None = None,
    device: str = "cpu",
    dtype: Any | None = None,
    micro_batch_size: int = 16,
) -> Any:
    """Build an evaluated full-BIE73 service view without fallback.

    Cascade mode requires one explicit evaluated adaptive rule policy; this
    scaffold deliberately has no provisional default winner.  Model-only mode
    selects the evaluated ``model_only_rules_disabled`` research policy
    internally, skips the rule branch, and still applies service validators,
    thresholds and fusion.  It is therefore a service ablation, not the final
    raw-model benchmark protocol.
    """

    selected_scope = _normalize_scope(scope)
    selected_mode = _normalize_service_mode(mode)
    if selected_mode == "cascade":
        if rule_policy is None:
            raise FullBie73ProfileError("cascade mode requires an explicit adaptive rule_policy")
        selected_policy: str = _normalize_rule_policy(rule_policy)
    else:
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
    validated_thresholds = _threshold_mapping(
        thresholds,
        open24_labels=tuple(runtime.get("open24_labels", ())),
    )
    threshold_normalizer = runtime.get("threshold_normalizer")
    if not callable(threshold_normalizer):
        raise RuntimeError("the installed full-BIE73 threshold runtime is incomplete")
    normalized_thresholds = threshold_normalizer(validated_thresholds)
    if (
        not isinstance(normalized_thresholds, Mapping)
        or len(normalized_thresholds) != 24
        or any(value != 0.5 for value in normalized_thresholds.values())
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
    "FULL_BIE73_ADAPTIVE_RULE_POLICIES",
    "FULL_BIE73_DECODER_ID",
    "FULL_BIE73_MODEL_ONLY_RULE_POLICY",
    "FULL_BIE73_PUBLIC_PROFILE_SCHEMA_VERSION",
    "FULL_BIE73_PUBLIC_PROFILE_VERSION",
    "FULL_BIE73_THRESHOLD_POLICY",
    "FullBie73AdaptiveRulePolicy",
    "FullBie73ProfileError",
    "FullBie73Scope",
    "FullBie73ServiceMode",
    "build_full_bie73_service_pipeline",
    "load_full_bie73_predictor",
]
