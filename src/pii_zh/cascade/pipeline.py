"""In-memory Chinese PII cascade with one deterministic stage ordering."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from pii_zh.data.validators import VALIDATORS, ValidationResult
from pii_zh.fusion.deterministic import DeterministicFusion, FusedDetection, as_detection
from pii_zh.fusion.replacement import (
    ReplacementSpec,
    apply_replacements,
    merge_replacement_coverage,
)
from pii_zh.rules import CnCommonRulePack

from .config import CascadeConfig
from .context import ContextCandidate, ContextDecision, ContextEnhancer
from .result import CascadeDetection
from .routing import CandidateBranch, canonicalize_entity_type
from .stages import DEFAULT_RELEASE_STAGE_POLICY, PipelineStagePolicy


class Recognizer(Protocol):
    def analyze(self, text: str, entities: Sequence[str] | None) -> Iterable[Any]: ...


ValidatorResult = ValidationResult | bool
Validator = Callable[[str], ValidatorResult]


def _sequence(value: object, *, location: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{location} must be a sequence of recognizer results")
    try:
        return list(cast(Iterable[Any], value))
    except TypeError as exc:
        raise TypeError(f"{location} must be a sequence of recognizer results") from exc


def _validation_passed(value: ValidatorResult) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, ValidationResult):
        raise TypeError("validators must return ValidationResult or bool")
    return value.valid


class CascadePipeline:
    """Compose rule/model recognizers without retaining source text in results.

    Recognizers are injected as already-created local objects.  Construction
    never calls ``load`` or otherwise touches a model.  In ``rules-only`` mode
    the model reference is neither inspected nor invoked.
    """

    def __init__(
        self,
        *,
        config: CascadeConfig | None = None,
        rule_recognizer: Recognizer | Callable[..., Iterable[Any]] | None = None,
        model_recognizer: Recognizer | Callable[..., Iterable[Any]] | None = None,
        validators: Mapping[str, Validator] | None = None,
        context_enhancer: ContextEnhancer | None = None,
        stage_policy: PipelineStagePolicy | None = None,
        allow_evaluation_profile: bool = False,
        model_identity: Mapping[str, str | int | bool | None] | None = None,
    ) -> None:
        self.config = config or CascadeConfig()
        self.stage_policy = stage_policy or DEFAULT_RELEASE_STAGE_POLICY
        if not isinstance(self.stage_policy, PipelineStagePolicy):
            raise TypeError("stage_policy must be a PipelineStagePolicy")
        if not isinstance(allow_evaluation_profile, bool):
            raise TypeError("allow_evaluation_profile must be a boolean")
        if self.stage_policy.purpose == "evaluation_ablation" and not allow_evaluation_profile:
            raise ValueError("evaluation-only stage policies require explicit authorization")
        self._evaluation_profile_authorized = allow_evaluation_profile
        self.rule_recognizer = (
            rule_recognizer
            if rule_recognizer is not None
            else CnCommonRulePack()
            if self.config.uses_rules
            else None
        )
        self.model_recognizer = model_recognizer
        if model_identity is not None:
            if not self.config.uses_model:
                raise ValueError("model_identity requires a model-enabled mode")
            if (
                not isinstance(model_identity, Mapping)
                or not model_identity
                or any(
                    not isinstance(key, str)
                    or not key
                    or not isinstance(value, (str, int, bool, type(None)))
                    for key, value in model_identity.items()
                )
            ):
                raise TypeError("model_identity must be a non-empty flat JSON-safe mapping")
            self.model_identity: Mapping[str, str | int | bool | None] | None = MappingProxyType(
                dict(model_identity)
            )
        else:
            self.model_identity = None
        if context_enhancer is not None and not callable(
            getattr(context_enhancer, "enhance_candidate", None)
        ):
            raise TypeError("context_enhancer must expose enhance_candidate")
        if not self.stage_policy.context_enabled and context_enhancer is not None:
            raise ValueError("the selected stage policy disables context")
        if self.stage_policy.context_required and context_enhancer is None:
            raise ValueError("the selected stage policy requires an explicit context enhancer")
        self.context_enhancer = context_enhancer
        if self.config.uses_rules and self.rule_recognizer is None:
            raise ValueError("the selected mode requires a rule recognizer")
        if self.config.uses_model and self.model_recognizer is None:
            raise ValueError("the selected mode requires an injected model recognizer")
        registry: dict[str, Validator] = dict(VALIDATORS)
        if validators is not None:
            for label, validator in validators.items():
                canonicalize_entity_type(label)
                if not callable(validator):
                    raise TypeError("validator registry values must be callable")
                registry[label] = validator
        self.validators: Mapping[str, Validator] = registry
        configured_entities = frozenset(self.config.route_map)
        for branch in ("rule", "model"):
            override = self.stage_policy.branch_override(branch)
            if override is not None and not override <= configured_entities:
                raise ValueError(f"{branch} branch override contains an unconfigured entity")
            if override and (
                (branch == "rule" and not self.config.uses_rules)
                or (branch == "model" and not self.config.uses_model)
            ):
                raise ValueError(f"{branch} branch override conflicts with cascade mode")
        for route in self.config.routes:
            if route.validator_label is not None and route.validator_label not in registry:
                raise ValueError(
                    f"route {route.entity_type} requires unavailable validator "
                    f"{route.validator_label}"
                )
        self._fusion = DeterministicFusion(
            default_threshold=0.0,
            source_priority={
                self.config.rule_source: 40,
                self.config.model_source: 30,
            },
            keep_nested_different_types=self.config.keep_nested_different_types,
        )

    def with_context_enhancer(self, context_enhancer: ContextEnhancer | None) -> CascadePipeline:
        """Return an equivalent pipeline with one replacement context policy.

        Recognizer and validator objects are reused; the returned pipeline owns
        the complete stage ordering.  This is primarily used by protocol
        adapters which historically accepted context separately.
        """

        return CascadePipeline(
            config=self.config,
            rule_recognizer=self.rule_recognizer,
            model_recognizer=self.model_recognizer,
            validators=self.validators,
            context_enhancer=context_enhancer,
            stage_policy=self.stage_policy,
            allow_evaluation_profile=self._evaluation_profile_authorized,
            model_identity=self.model_identity,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        config: CascadeConfig | None = None,
        rule_recognizer: Recognizer | Callable[..., Iterable[Any]] | None = None,
        validators: Mapping[str, Validator] | None = None,
        device: str = "cpu",
        dtype: Any | None = None,
        micro_batch_size: int = 16,
        calibration: Mapping[str, Any] | str | Path | None = None,
        thresholds: Mapping[str, float] | None = None,
        max_tokens: int = 512,
        stride_fraction: float = 0.25,
        local_files_only: bool = True,
        stage_policy: PipelineStagePolicy | None = None,
        allow_evaluation_profile: bool = False,
        context_enhancer: ContextEnhancer | None = None,
        recognizer_deduplication_policy: str = "high_overlap_v1",
        allowed_model_labels: Sequence[str] | None = None,
        model_identity: Mapping[str, str | int | bool | None] | None = None,
    ) -> CascadePipeline:
        """Build a cascade from one manifest-bound local safetensors model.

        Remote identifiers and implicit Hub downloads are deliberately not
        supported.  Callers must materialize and verify the model directory
        before construction.
        """

        if local_files_only is not True:
            raise ValueError("CascadePipeline only supports local_files_only=True")
        resolved_config = config or CascadeConfig(mode="cascade")
        if not resolved_config.uses_model:
            raise ValueError("from_pretrained requires a model-enabled cascade mode")
        # Keep optional heavyweight dependencies outside module import time so
        # rules-only users never import or initialize torch/transformers.
        from pii_zh.inference.transformers_predictor import load_local_predictor
        from pii_zh.presidio import QwenPiiRecognizer

        predictor_options: dict[str, Any] = {
            "device": device,
            "dtype": dtype,
            "micro_batch_size": micro_batch_size,
        }
        if allowed_model_labels is not None:
            # This must reach TransformersSpanPredictor before argmax/BIO
            # decoding. Filtering recognizer output later is not equivalent:
            # a disallowed logit can otherwise suppress an allowed label.
            predictor_options["allowed_labels"] = allowed_model_labels
        predictor = load_local_predictor(model_path, **predictor_options)
        recognizer = QwenPiiRecognizer(
            predictor=predictor,
            tokenizer=predictor.tokenizer,
            calibration=calibration,
            thresholds=thresholds,
            max_tokens=max_tokens,
            stride_fraction=stride_fraction,
            model_version=Path(model_path).name,
            attention_mode=predictor.attention_mode,
            deduplication_policy=recognizer_deduplication_policy,
        )
        return cls(
            config=resolved_config,
            rule_recognizer=rule_recognizer,
            model_recognizer=recognizer,
            validators=validators,
            context_enhancer=context_enhancer,
            stage_policy=stage_policy,
            allow_evaluation_profile=allow_evaluation_profile,
            model_identity=model_identity,
        )

    def _requested_entities(self, entities: Sequence[str] | None) -> frozenset[str]:
        available = self.config.route_map
        if entities is None:
            return frozenset(available)
        if isinstance(entities, (str, bytes)):
            raise TypeError("entities must be a sequence of labels, not one string")
        canonical: list[str] = []
        for entity in entities:
            normalized = canonicalize_entity_type(entity)
            if normalized not in available:
                raise ValueError(f"entity filter is not configured: {normalized}")
            canonical.append(normalized)
        return frozenset(canonical)

    def _branch_entities(
        self, requested: frozenset[str], *, branch: CandidateBranch
    ) -> tuple[str, ...]:
        override = self.stage_policy.branch_override(branch)
        if override is not None:
            return tuple(sorted(requested & override))
        return tuple(
            sorted(
                entity
                for entity in requested
                if (
                    self.config.route_map[entity].rule_enabled
                    if branch == "rule"
                    else self.config.route_map[entity].model_enabled
                )
            )
        )

    @staticmethod
    def _recognize_many(
        recognizer: object,
        texts: Sequence[str],
        entities: Sequence[str],
        *,
        branch: CandidateBranch,
    ) -> list[list[Any]]:
        if not texts or not entities:
            return [[] for _ in texts]
        batch_method = getattr(recognizer, "analyze_batch", None)
        if callable(batch_method):
            try:
                raw_batches = batch_method(texts=texts, entities=entities)
            except Exception:
                raise RuntimeError(f"{branch} recognizer failed") from None
            batches = _sequence(raw_batches, location=f"{branch} batch output")
            if len(batches) != len(texts):
                raise ValueError(f"{branch} recognizer must return one batch per input text")
            return [_sequence(batch, location=f"{branch} batch item") for batch in batches]
        method = getattr(recognizer, "analyze", None)
        if callable(method):
            output: list[list[Any]] = []
            for text in texts:
                try:
                    raw = method(text=text, entities=entities)
                except Exception:
                    raise RuntimeError(f"{branch} recognizer failed") from None
                output.append(_sequence(raw, location=f"{branch} recognizer output"))
            return output
        if callable(recognizer):
            output = []
            for text in texts:
                try:
                    raw = recognizer(text, entities)
                except Exception:
                    raise RuntimeError(f"{branch} recognizer failed") from None
                output.append(_sequence(raw, location=f"{branch} recognizer output"))
            return output
        raise TypeError(f"{branch} recognizer must be callable or expose analyze/analyze_batch")

    def _validated_candidate(
        self,
        raw: Any,
        text: str,
        *,
        branch: CandidateBranch,
        requested: frozenset[str],
    ) -> FusedDetection | None:
        source = self.config.rule_source if branch == "rule" else self.config.model_source
        candidate = as_detection(raw, default_source=source)
        entity_type = canonicalize_entity_type(candidate.entity_type)
        route = self.config.route_map.get(entity_type)
        # Recognizer output is an untrusted allow-list input.  Unknown or
        # unrequested labels fail closed rather than leaking into fusion.
        if route is None or entity_type not in requested:
            return None
        if candidate.end > len(text):
            raise ValueError(f"{branch} recognizer returned a span outside the input length")
        override = self.stage_policy.branch_override(branch)
        branch_enabled = (
            entity_type in override
            if override is not None
            else route.rule_enabled
            if branch == "rule"
            else route.model_enabled
        )
        if not branch_enabled:
            return None
        score = candidate.score
        context_step: str | None = None
        if self.stage_policy.context_enabled and self.context_enhancer is not None:
            context_candidate = ContextCandidate(
                entity_type=entity_type,
                start=candidate.start,
                end=candidate.end,
                score=score,
            )
            try:
                context_decision = self.context_enhancer.enhance_candidate(
                    text,
                    context_candidate,
                )
            except Exception:
                raise RuntimeError("context enhancer failed") from None
            if not isinstance(context_decision, ContextDecision):
                raise RuntimeError("context enhancer returned an invalid decision")
            if (
                context_decision.outcome == "no_match"
                and context_decision.score != context_candidate.score
            ):
                raise RuntimeError("context enhancer returned an invalid decision")
            if (
                context_decision.outcome == "positive"
                and context_decision.score < context_candidate.score
            ):
                raise RuntimeError("context enhancer returned an invalid decision")
            if (
                context_decision.outcome == "negative"
                and context_decision.score > context_candidate.score
            ):
                raise RuntimeError("context enhancer returned an invalid decision")
            score = context_decision.score
            context_step = context_decision.decision_step
        metadata: dict[str, object] = {"branch": branch}
        validator = (
            self.validators.get(route.validator_label)
            if route.validator_label is not None
            else None
        )
        if not self.stage_policy.validators_enabled:
            validator_step = "validator:disabled_by_profile"
        elif validator is not None:
            try:
                validation = validator(text[candidate.start : candidate.end])
            except Exception:
                raise RuntimeError("structured validator failed") from None
            if not _validation_passed(validation):
                return None
            metadata["validator_valid"] = True
            validator_step = "validator:passed"
        elif branch == "model" and route.model_requires_validator:
            # Constructor validation should make this unreachable; retaining
            # the guard preserves the ordering invariant if state is adapted.
            return None
        else:
            validator_step = "validator:not_required"
        if self.stage_policy.thresholds_enabled:
            decision = route.evaluate(
                branch=branch,
                score=score,
                span_length=candidate.end - candidate.start,
            )
            if not decision.accepted:
                return None
            route_steps = decision.decision_process
        else:
            route_steps = (
                f"candidate:{branch}",
                f"route:{route.category}",
                "threshold:disabled_by_profile",
            )
        # Routing owns the stable candidate/route/threshold tokens.  Place the
        # shared context and validator decisions between its prefix and final
        # threshold token to preserve the frozen C0 execution order.
        steps = list(route_steps[:2])
        if context_step is not None:
            steps.append(context_step)
        steps.append(validator_step)
        steps.extend(route_steps[2:])
        metadata["decision_process"] = tuple(steps)
        return FusedDetection(
            entity_type=entity_type,
            start=candidate.start,
            end=candidate.end,
            score=score,
            source=source,
            metadata=metadata,
        )

    def _candidate_groups_many(
        self, texts: Sequence[str], *, entities: Sequence[str] | None
    ) -> list[tuple[list[FusedDetection], list[FusedDetection]]]:
        """Return candidates that passed every pre-fusion acceptance stage."""

        requested = self._requested_entities(entities)
        rule_batches: list[list[Any]] = [[] for _ in texts]
        model_batches: list[list[Any]] = [[] for _ in texts]
        if self.config.uses_rules:
            assert self.rule_recognizer is not None
            rule_batches = self._recognize_many(
                self.rule_recognizer,
                texts,
                self._branch_entities(requested, branch="rule"),
                branch="rule",
            )
        if self.config.uses_model:
            assert self.model_recognizer is not None
            model_batches = self._recognize_many(
                self.model_recognizer,
                texts,
                self._branch_entities(requested, branch="model"),
                branch="model",
            )
        output: list[tuple[list[FusedDetection], list[FusedDetection]]] = []
        for text, raw_rules, raw_model in zip(texts, rule_batches, model_batches, strict=True):
            rules = [
                candidate
                for raw in raw_rules
                if (
                    candidate := self._validated_candidate(
                        raw, text, branch="rule", requested=requested
                    )
                )
                is not None
            ]
            model = [
                candidate
                for raw in raw_model
                if (
                    candidate := self._validated_candidate(
                        raw, text, branch="model", requested=requested
                    )
                )
                is not None
            ]
            output.append((rules, model))
        return output

    def _run_many(
        self, texts: Sequence[str], *, entities: Sequence[str] | None
    ) -> list[list[CascadeDetection]]:
        groups = self._candidate_groups_many(texts, entities=entities)
        output: list[list[CascadeDetection]] = []
        for rules, model in groups:
            if self.stage_policy.conflict_policy == "deterministic":
                fused = self._fusion.fuse(rules, model)
            else:
                # ``simple_union`` and raw-model evaluation retain both
                # overlapping labels.  They still reuse DeterministicFusion's
                # exact same-label collapse so identical spans from two
                # branches are emitted once with both sources recorded.
                fused = sorted(
                    self._fusion._collapse_exact([*rules, *model]),
                    key=lambda item: (item.start, item.end, item.entity_type),
                )
            detections: list[CascadeDetection] = []
            for item in fused:
                raw_sources = item.metadata.get("sources", [item.source])
                if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
                    raise TypeError("fusion sources must be a sequence of strings")
                sources = tuple(sorted({str(source) for source in raw_sources}))
                raw_steps = item.metadata.get("decision_process", ())
                if not isinstance(raw_steps, tuple) or any(
                    not isinstance(step, str) for step in raw_steps
                ):
                    raise TypeError("fusion decision_process must be a tuple of strings")
                fusion_decision = item.metadata.get("fusion_decision", "selected")
                steps = (*raw_steps, f"fusion:{fusion_decision}")
                if len(sources) > 1:
                    steps = (*steps, "fusion:exact_agreement")
                detections.append(
                    CascadeDetection(
                        start=item.start,
                        end=item.end,
                        entity_type=item.entity_type,
                        score=item.score,
                        source=item.source,
                        sources=sources,
                        decision_process=steps,
                    )
                )
            output.append(detections)
        return output

    def detect(self, text: str, *, entities: Sequence[str] | None = None) -> list[CascadeDetection]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._run_many([text], entities=entities)[0]

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> list[list[CascadeDetection]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return self._run_many(values, entities=entities)

    def redact(
        self,
        text: str,
        replacement: ReplacementSpec = "<PII>",
        *,
        default_replacement: str | None = None,
        entities: Sequence[str] | None = None,
    ) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        detect_method = self.detect
        run_many_method = self._run_many
        if not (
            getattr(detect_method, "__func__", None) is CascadePipeline.detect
            and getattr(run_many_method, "__func__", None) is CascadePipeline._run_many
        ):
            raise RuntimeError(
                "custom detection paths must implement an explicit safe redact method"
            )
        rules, model = self._candidate_groups_many([text], entities=entities)[0]
        coverage = merge_replacement_coverage([*rules, *model])
        return apply_replacements(
            text,
            coverage,
            replacement,
            default_replacement=default_replacement,
        )


__all__ = ["CascadePipeline", "Recognizer", "Validator"]
