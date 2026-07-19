"""Fail-closed CLI for the six packaged ``CascadePipeline`` ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from pii_zh.cascade import CascadePipeline, load_ablation_profile
from pii_zh.cascade.ablation_profiles import REQUIRED_ABLATION_IDS, AblationRuntimeProfile
from pii_zh.cascade.ablation_runtime import (
    ContextConfigIdentity,
    ExecutionEvidenceIdentity,
    ProjectRunIntent,
    ablation_config,
    build_ablation_pipeline,
    load_closed_bio_mapping,
    load_closed_bio_ner_recognizer,
    load_context_config,
    verify_execution_evidence,
    verify_loaded_project_predictor,
    verify_ner_admission_receipt,
    verify_ner_artifact_binding,
    verify_project_model_for_ablation,
)
from pii_zh.cascade.config import CascadeConfig
from pii_zh.data.schema import SchemaError, iter_jsonl
from pii_zh.evaluation.io import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    Span,
    load_gold_jsonl,
    load_prediction_jsonl,
    write_prediction_jsonl,
)
from pii_zh.evaluation.metrics import evaluate_documents
from pii_zh.evaluation.provenance import (
    build_evaluation_provenance,
    canonical_json_hash,
    sha256_file,
)
from pii_zh.evaluation.required_metrics import build_required_metrics
from pii_zh.taxonomy import load_presidio_mapping

ABLATION_EVALUATION_SCHEMA_VERSION = "pii-zh.cascade-ablation-evaluation.v2"
DOCUMENT_ID_POLICY_VERSION = "pii-zh.cascade-ablation-doc-id.v1"
_DOCUMENT_ID_DOMAIN = b"pii-zh.cascade-ablation-doc-id.v1\0"
_SINGLE_VISIBLE_GPU = re.compile(r"^[0-9]+$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_PROJECT_RECEIPT_FIELDS = (
    "candidate_freeze",
    "candidate_selection_receipt",
    "release_gate_receipt",
)


class SafeAblationCliError(RuntimeError):
    """A fixed, raw-input-free CLI failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-zh-evaluate-ablation",
        description="Evaluate one packaged fixed-24-class ablation through CascadePipeline.",
    )
    parser.add_argument("--adapter", required=True, choices=REQUIRED_ABLATION_IDS)
    parser.add_argument(
        "--run-intent",
        required=True,
        choices=("development-non-claim", "frozen-confirmatory"),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--candidate-freeze", type=Path)
    parser.add_argument("--candidate-selection-receipt", type=Path)
    parser.add_argument("--release-gate-receipt", type=Path)
    parser.add_argument("--ner-model-path", type=Path)
    parser.add_argument("--ner-mapping-config", type=Path)
    parser.add_argument("--ner-artifact-binding", type=Path)
    parser.add_argument("--ner-admission-receipt", type=Path)
    parser.add_argument("--context-config", type=Path)
    parser.add_argument("--source-closure", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--document-batch-size", type=int, default=32)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--stride-fraction", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _existing_directory(path: Path | None, *, field: str) -> Path:
    if path is None:
        raise SafeAblationCliError(f"{field} is required by the selected adapter")
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise SafeAblationCliError(f"{field} must be an existing local directory")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SafeAblationCliError(f"{field} must be an existing local directory") from exc
    if not resolved.is_dir():
        raise SafeAblationCliError(f"{field} must be an existing local directory")
    return resolved


def _validate_args(args: argparse.Namespace, profile: AblationRuntimeProfile) -> None:
    outputs = [_resolved(args.predictions), _resolved(args.report)]
    if len(set(outputs)) != 2:
        raise SafeAblationCliError("prediction and report outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in (args.predictions, args.report)):
        raise SafeAblationCliError("refusing to overwrite an existing output")
    if not args.input.is_file() or _resolved(args.input) in outputs:
        raise SafeAblationCliError("canonical input must be a distinct existing local file")
    if args.document_batch_size < 1 or args.window_batch_size < 1 or args.max_tokens < 1:
        raise SafeAblationCliError("batch sizes and max tokens must be positive")
    if not 0.0 < args.stride_fraction < 1.0:
        raise SafeAblationCliError("stride fraction must be between zero and one")
    if args.device == "cpu":
        args.device = "cpu"
    elif args.device in {"cuda", "cuda:0"}:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or _SINGLE_VISIBLE_GPU.fullmatch(visible) is None:
            raise SafeAblationCliError(
                "CUDA evaluation requires one numeric CUDA_VISIBLE_DEVICES selection"
            )
        args.device = "cuda:0"
    else:
        raise SafeAblationCliError("device must be cpu, cuda, or cuda:0")

    run_intent = cast(ProjectRunIntent, args.run_intent)
    confirmatory = run_intent == "frozen-confirmatory"
    execution_receipts = (args.source_closure, args.runtime_lock)
    if confirmatory and any(path is None for path in execution_receipts):
        raise SafeAblationCliError(
            "frozen-confirmatory runs require source closure and runtime lock receipts"
        )
    if not confirmatory and any(path is not None for path in execution_receipts):
        raise SafeAblationCliError(
            "development runs must not present confirmatory execution receipts"
        )

    project_model = profile.recognizer_kind == "project_model"
    ner_model = profile.recognizer_kind == "native_presidio_cn_common_closed_bio_ner"
    project_receipts = tuple(getattr(args, field) for field in _PROJECT_RECEIPT_FIELDS)
    if not project_model and not ner_model and args.device != "cpu":
        raise SafeAblationCliError("rules-only evaluation must execute on cpu")
    if project_model:
        args.model_path = _existing_directory(args.model_path, field="project model path")
        if any(
            value is not None
            for value in (
                args.ner_model_path,
                args.ner_mapping_config,
                args.ner_artifact_binding,
                args.ner_admission_receipt,
            )
        ):
            raise SafeAblationCliError("NER options are invalid for a project-model adapter")
        if profile.adapter_id in {"project_model_raw", "simple_union"} and (
            args.calibration is not None
        ):
            raise SafeAblationCliError("raw and simple-union adapters forbid calibration")
        if confirmatory and any(path is None for path in project_receipts):
            raise SafeAblationCliError(
                "frozen project runs require freeze, selection, and release-gate receipts"
            )
        if not confirmatory and any(path is not None for path in project_receipts):
            raise SafeAblationCliError(
                "development project runs must not present confirmatory model receipts"
            )
    elif (
        args.model_path is not None
        or args.calibration is not None
        or any(path is not None for path in project_receipts)
    ):
        raise SafeAblationCliError("project-model options are invalid for the selected adapter")

    if ner_model:
        args.ner_model_path = _existing_directory(args.ner_model_path, field="NER model path")
        if args.ner_mapping_config is None or args.ner_artifact_binding is None:
            raise SafeAblationCliError(
                "NER mapping config and artifact binding are required by the NER adapter"
            )
        if confirmatory and args.ner_admission_receipt is None:
            raise SafeAblationCliError(
                "frozen NER runs require a public-source admission receipt"
            )
        if not confirmatory and args.ner_admission_receipt is not None:
            raise SafeAblationCliError(
                "development NER runs must not present a confirmatory admission receipt"
            )
    elif any(
        value is not None
        for value in (
            args.ner_model_path,
            args.ner_mapping_config,
            args.ner_artifact_binding,
            args.ner_admission_receipt,
        )
    ):
        raise SafeAblationCliError("NER options are invalid for the selected adapter")

    if profile.stage_policy.context_required:
        if args.context_config is None:
            raise SafeAblationCliError("final_cascade requires an explicit context config")
    elif args.context_config is not None:
        raise SafeAblationCliError("context config is invalid for a context-disabled adapter")
    if args.calibration is not None and (
        args.calibration.is_symlink() or not args.calibration.is_file()
    ):
        raise SafeAblationCliError("calibration must be an existing non-symlink local file")
    for model_root in (args.model_path, args.ner_model_path):
        if model_root is not None and any(
            output == model_root or model_root in output.parents for output in outputs
        ):
            raise SafeAblationCliError("outputs must not be written inside a model artifact")


def _build_pipeline(
    args: argparse.Namespace,
    profile: AblationRuntimeProfile,
) -> tuple[CascadePipeline, dict[str, object] | None, ContextConfigIdentity | None]:
    context_enhancer = None
    context_identity = None
    if args.context_config is not None:
        try:
            context_enhancer, context_identity = load_context_config(args.context_config)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SafeAblationCliError("context config verification failed") from exc

    if profile.recognizer_kind == "none":
        return (
            build_ablation_pipeline(profile, context_enhancer=context_enhancer),
            None,
            context_identity,
        )
    if profile.recognizer_kind == "native_presidio_cn_common_closed_bio_ner":
        try:
            from pii_zh.presidio.native_framework import NativePresidioFrameworkRecognizer

            mapping = load_closed_bio_mapping(args.ner_mapping_config)
            model_root, artifact = verify_ner_artifact_binding(
                args.ner_model_path,
                args.ner_artifact_binding,
                mapping=mapping,
            )
            ner_recognizer = load_closed_bio_ner_recognizer(
                model_root,
                mapping=mapping,
                artifact=artifact,
                device=args.device,
                micro_batch_size=args.window_batch_size,
                max_tokens=args.max_tokens,
                stride_fraction=args.stride_fraction,
                deduplication_policy="exact_only_v1",
            )
            framework = NativePresidioFrameworkRecognizer(ner_recognizer=ner_recognizer)
            admission = verify_ner_admission_receipt(
                args.ner_admission_receipt,
                run_intent=cast(ProjectRunIntent, args.run_intent),
                artifact=artifact,
                mapping=mapping,
            )
            pipeline = build_ablation_pipeline(profile, recognizer=framework)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SafeAblationCliError("native Presidio NER adapter initialization failed") from exc
        identity: dict[str, object] = {
            "artifact": artifact.to_report(),
            "mapping_id": mapping.mapping_id,
            "mapping_config_file_sha256": mapping.config_file_sha256,
            "mapping_config_sha256": mapping.config_sha256,
            "allowed_target_labels": ["PERSON"],
            "recognizer_deduplication_policy": "exact_only_v1",
            "admission": admission.to_report(),
            "native_framework": framework.runtime_identity(),
        }
        return pipeline, identity, context_identity

    deduplication_policy = (
        "exact_only_v1"
        if profile.adapter_id in {"project_model_raw", "simple_union"}
        else "high_overlap_v1"
    )
    try:
        assert args.model_path is not None
        project_identity = verify_project_model_for_ablation(
            args.model_path,
            run_intent=cast(ProjectRunIntent, args.run_intent),
            candidate_freeze_path=args.candidate_freeze,
            candidate_selection_path=args.candidate_selection_receipt,
            release_gate_path=args.release_gate_receipt,
        )
        pipeline = CascadePipeline.from_pretrained(
            args.model_path,
            config=ablation_config(profile),
            device=args.device,
            micro_batch_size=args.window_batch_size,
            calibration=args.calibration,
            max_tokens=args.max_tokens,
            stride_fraction=args.stride_fraction,
            local_files_only=True,
            stage_policy=profile.stage_policy,
            allow_evaluation_profile=True,
            context_enhancer=context_enhancer,
            recognizer_deduplication_policy=deduplication_policy,
        )
        verify_loaded_project_predictor(pipeline, project_identity)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SafeAblationCliError("local project-model initialization failed") from exc
    identity = {
        **project_identity.to_report(),
        "recognizer_deduplication_policy": deduplication_policy,
    }
    return pipeline, identity, context_identity


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    )
    path = Path(handle.name)
    handle.close()
    return path


def safe_evaluation_doc_id(doc_id: str) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise EvaluationDataError("doc_id must be a non-empty string")
    digest = hashlib.sha256(_DOCUMENT_ID_DOMAIN + doc_id.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _fixed_evaluation_scope(config: CascadeConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    policy_scope = tuple(config.route_map)
    mapping = load_presidio_mapping()
    model_scope = tuple(
        model_label
        for model_label, policy_label in mapping.model_to_presidio.items()
        if policy_label in policy_scope
    )
    if (
        len(policy_scope) != 24
        or len(model_scope) != 24
        or len(set(model_scope)) != 24
        or {mapping.model_to_presidio[label] for label in model_scope} != set(policy_scope)
    ):
        raise SafeAblationCliError("packaged evaluation scope is not the frozen 24 classes")
    return policy_scope, model_scope


def _generate_predictions(
    *,
    pipeline: CascadePipeline,
    input_path: Path,
    temporary: Path,
    batch_size: int,
) -> int:
    policy_scope, model_scope = _fixed_evaluation_scope(pipeline.config)
    mapping = load_presidio_mapping()
    presidio_to_model = {
        mapping.model_to_presidio[model_label]: model_label for model_label in model_scope
    }
    if set(presidio_to_model) != set(policy_scope):
        raise SafeAblationCliError("packaged evaluation mapping is not one-to-one")
    seen_raw: set[str] = set()
    seen_safe: set[str] = set()
    count = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def flush(handle: Any) -> None:
        nonlocal count
        if not batch_texts:
            return
        try:
            detected = pipeline.detect_batch(batch_texts, entities=None)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise SafeAblationCliError("cascade inference failed") from exc
        predictions = [
            PredictionRecord(
                doc_id=doc_id,
                spans=tuple(
                    Span(
                        start=item.start,
                        end=item.end,
                        label=presidio_to_model.get(item.entity_type, item.entity_type),
                        score=item.score,
                    )
                    for item in detections
                ),
            )
            for doc_id, detections in zip(batch_ids, detected, strict=True)
        ]
        write_prediction_jsonl(predictions, handle)
        count += len(predictions)
        batch_ids.clear()
        batch_texts.clear()

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in iter_jsonl(input_path):
                if record.doc_id in seen_raw:
                    raise SafeAblationCliError(
                        "canonical input contains duplicate document identifiers"
                    )
                safe_id = safe_evaluation_doc_id(record.doc_id)
                if safe_id in seen_safe:
                    raise SafeAblationCliError("document identifier hash collision")
                seen_raw.add(record.doc_id)
                seen_safe.add(safe_id)
                batch_ids.append(safe_id)
                batch_texts.append(record.text)
                if len(batch_texts) >= batch_size:
                    flush(handle)
            flush(handle)
    except SafeAblationCliError:
        raise
    except (EvaluationDataError, OSError, SchemaError, TypeError, UnicodeError, ValueError) as exc:
        raise SafeAblationCliError("canonical input or prediction generation failed") from exc
    if count == 0:
        raise SafeAblationCliError("canonical input must contain at least one document")
    return count


def _config_payload(config: CascadeConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
        "routes": [asdict(route) for route in config.routes],
    }


def _implementation_identity(profile: AblationRuntimeProfile) -> dict[str, object]:
    package_root = Path(__file__).resolve().parents[1]
    allowed_suffixes = {".json", ".py", ".yaml", ".yml"}
    sources: list[Path] = []
    for candidate in package_root.rglob("*"):
        if candidate.is_symlink():
            raise SafeAblationCliError("package source closure contains an unsafe file")
        if candidate.suffix not in allowed_suffixes:
            continue
        if not candidate.is_file():
            raise SafeAblationCliError("package source closure contains an unsafe file")
        sources.append(candidate)
    if not sources:
        raise SafeAblationCliError("package source closure is empty")
    components = {
        candidate.relative_to(package_root).as_posix(): sha256_file(candidate)
        for candidate in sorted(sources)
    }
    return {
        "closure_mode": "full_pii_zh_python_and_packaged_resource_superset_v1",
        "adapter_id": profile.adapter_id,
        "source_file_count": len(components),
        "component_sha256": components,
        "implementation_sha256": canonical_json_hash(components),
        "entrypoint_boundary": {
            "console_script": "pii-zh-evaluate-ablation",
            "module": "pii_zh.evaluation.cascade_ablation_cli:main",
            "repository_wrapper_and_packaging_bound_by_suite_readiness": True,
        },
    }


def _distribution_version(distribution: str) -> str:
    try:
        value = version(distribution)
    except PackageNotFoundError:
        return "not-installed"
    return value if _SAFE_VERSION.fullmatch(value) is not None else "unreportable"


def _runtime_versions() -> dict[str, str]:
    python_version = platform.python_version()
    if _SAFE_VERSION.fullmatch(python_version) is None:
        raise SafeAblationCliError("Python runtime version is not safely reportable")
    return {
        "python": python_version,
        "pii-zh-qwen": _distribution_version("pii-zh-qwen"),
        "torch": _distribution_version("torch"),
        "transformers": _distribution_version("transformers"),
        "presidio-analyzer": _distribution_version("presidio-analyzer"),
    }


def _runtime_input_identity(
    *,
    args: argparse.Namespace,
    profile: AblationRuntimeProfile,
    recognizer_identity: Mapping[str, object] | None,
    context_identity: ContextConfigIdentity | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ablation_id": profile.adapter_id,
        "profile_sha256": profile.profile_sha256,
        "recognizer_artifact": dict(recognizer_identity) if recognizer_identity else None,
        "calibration_file_sha256": (
            sha256_file(args.calibration) if args.calibration is not None else None
        ),
        "context_config": context_identity.to_report() if context_identity else None,
    }
    return {"payload": payload, "runtime_input_identity_sha256": canonical_json_hash(payload)}


def _verify_implementation_cross_bindings(
    *,
    profile: AblationRuntimeProfile,
    recognizer_identity: Mapping[str, object] | None,
    implementation_identity: Mapping[str, object],
) -> None:
    components = implementation_identity.get("component_sha256")
    if not isinstance(components, Mapping) or components.get(
        "cascade/profiles/ablation_profiles_v1.json"
    ) != profile.profile_set_file_sha256:
        raise SafeAblationCliError("profile resource differs from the package source closure")
    if profile.recognizer_kind != "native_presidio_cn_common_closed_bio_ner":
        return
    native = recognizer_identity.get("native_framework") if recognizer_identity else None
    if not isinstance(native, Mapping) or native.get(
        "framework_factory_file_sha256"
    ) != components.get("presidio/native_framework.py"):
        raise SafeAblationCliError("native Presidio factory differs from the source closure")


def _privacy_safe_gold(
    records: Sequence[GoldRecord],
    *,
    allowed_labels: frozenset[str],
) -> list[GoldRecord]:
    remapped: list[GoldRecord] = []
    seen: set[str] = set()
    for record in records:
        if any(span.label not in allowed_labels for span in record.spans):
            raise EvaluationDataError("gold contains a label outside the fixed 24-class scope")
        safe_id = safe_evaluation_doc_id(record.doc_id)
        if safe_id in seen:
            raise EvaluationDataError("document identifier hash collision")
        seen.add(safe_id)
        remapped.append(
            GoldRecord(
                doc_id=safe_id,
                spans=record.spans,
                text_length=record.text_length,
            )
        )
    return remapped


def _report(
    *,
    args: argparse.Namespace,
    profile: AblationRuntimeProfile,
    pipeline: CascadePipeline,
    temporary_predictions: Path,
    document_count: int,
    recognizer_identity: Mapping[str, object] | None,
    context_identity: ContextConfigIdentity | None,
    implementation_identity: Mapping[str, object],
    runtime_versions: Mapping[str, str],
    runtime_input_identity: Mapping[str, object],
    execution_evidence: ExecutionEvidenceIdentity,
) -> dict[str, Any]:
    policy_scope, model_scope = _fixed_evaluation_scope(pipeline.config)
    try:
        gold = _privacy_safe_gold(
            load_gold_jsonl(args.input),
            allowed_labels=frozenset(model_scope),
        )
        predictions = load_prediction_jsonl(temporary_predictions)
        if any(
            span.label not in model_scope
            for prediction in predictions
            for span in prediction.spans
        ):
            raise EvaluationDataError(
                "predictions contain a label outside the fixed 24-class scope"
            )
        report = evaluate_documents(
            gold,
            predictions,
            iou_threshold=args.iou_threshold,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        report["required_metrics"] = build_required_metrics(gold, predictions)
        report["provenance"] = build_evaluation_provenance(
            gold_path=args.input,
            predictions_path=temporary_predictions,
            gold_records=gold,
            evaluation_parameters={
                "bootstrap_samples": args.bootstrap_samples,
                "confidence": args.confidence,
                "iou_threshold": args.iou_threshold,
                "seed": args.seed,
            },
        )
    except (EvaluationDataError, OSError, TypeError, ValueError) as exc:
        raise SafeAblationCliError("strict evaluation failed") from exc

    stage_payload = profile.stage_policy.to_dict()
    config_payload = _config_payload(pipeline.config)
    scope_payload = {
        "scope_mode": "fixed_canonical_24",
        "entity_count": 24,
        "metric_model_labels": list(model_scope),
        "pipeline_policy_entities": list(policy_scope),
    }
    confirmatory = args.run_intent == "frozen-confirmatory"
    report["cascade_ablation_runtime"] = {
        "schema_version": ABLATION_EVALUATION_SCHEMA_VERSION,
        "ablation_id": profile.adapter_id,
        "run_intent": args.run_intent,
        "profile_sha256": profile.profile_sha256,
        "profile_set_file_sha256": profile.profile_set_file_sha256,
        "stage_policy": stage_payload,
        "stage_policy_sha256": canonical_json_hash(stage_payload),
        "configuration": config_payload,
        "configuration_sha256": canonical_json_hash(config_payload),
        "canonical_evaluation_scope": scope_payload,
        "canonical_evaluation_scope_sha256": canonical_json_hash(scope_payload),
        "implementation": dict(implementation_identity),
        "runtime_versions": dict(runtime_versions),
        "runtime_input_identity": dict(runtime_input_identity),
        "execution_evidence": execution_evidence.to_report(),
        "recognizer_artifact": dict(recognizer_identity) if recognizer_identity else None,
        "calibration_file_sha256": (
            sha256_file(args.calibration) if args.calibration is not None else None
        ),
        "context_config": context_identity.to_report() if context_identity else None,
        "execution_device": args.device,
        "document_batch_size": args.document_batch_size,
        "window_batch_size": (
            args.window_batch_size if profile.recognizer_kind != "none" else None
        ),
        "max_tokens": args.max_tokens if profile.recognizer_kind != "none" else None,
        "stride_fraction": (
            args.stride_fraction if profile.recognizer_kind != "none" else None
        ),
        "prediction_document_count": document_count,
        "predictions_sha256": sha256_file(temporary_predictions),
        "document_id_policy": {
            "schema_version": DOCUMENT_ID_POLICY_VERSION,
            "algorithm": "domain_separated_sha256",
            "serialized_format": "sha256:<64-lowercase-hex>",
            "original_document_ids_persisted": False,
        },
        "local_confirmatory_gate_status": "PASS" if confirmatory else "NOT_APPLICABLE",
        "claim_eligible": False,
        "claim_status": (
            "LOCAL_CONFIRMATORY_GATES_PASS_SUITE_CLAIMS_DISABLED"
            if confirmatory
            else "DEVELOPMENT_NON_CLAIM"
        ),
        "remaining_claim_blockers": [
            "human_hidden_dataset_admission_and_freeze_receipt",
            "complete_six_adapter_suite_receipt",
            "suite_level_statistical_claim_review",
        ],
        "comparator_threshold_status": (
            "CLOSED_PERSON_ONLY_NATIVE_FRAMEWORK_COMPARATOR"
            if profile.adapter_id == "presidio_zh_ner_cn_common"
            else None
        ),
        "privacy": {
            "contains_absolute_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_original_document_ids": False,
        },
    }
    return report


def _publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
        destination.chmod(0o400)
    except OSError as exc:
        raise SafeAblationCliError("failed to publish a new output") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    temporary_predictions: Path | None = None
    temporary_report: Path | None = None
    published: list[Path] = []
    try:
        profile = load_ablation_profile(cast(str, args.adapter))
        _validate_args(args, profile)
        pipeline, recognizer_identity, context_identity = _build_pipeline(args, profile)
        implementation_identity = _implementation_identity(profile)
        _verify_implementation_cross_bindings(
            profile=profile,
            recognizer_identity=recognizer_identity,
            implementation_identity=implementation_identity,
        )
        runtime_versions = _runtime_versions()
        runtime_input_identity = _runtime_input_identity(
            args=args,
            profile=profile,
            recognizer_identity=recognizer_identity,
            context_identity=context_identity,
        )
        execution_evidence = verify_execution_evidence(
            run_intent=cast(ProjectRunIntent, args.run_intent),
            ablation_id=profile.adapter_id,
            profile_set_file_sha256=profile.profile_set_file_sha256,
            implementation_sha256=cast(
                str, implementation_identity["implementation_sha256"]
            ),
            runtime_input_identity_sha256=cast(
                str, runtime_input_identity["runtime_input_identity_sha256"]
            ),
            runtime_versions=runtime_versions,
            source_closure_path=args.source_closure,
            runtime_lock_path=args.runtime_lock,
        )
        temporary_predictions = _temporary_path(args.predictions)
        temporary_report = _temporary_path(args.report)
        document_count = _generate_predictions(
            pipeline=pipeline,
            input_path=args.input,
            temporary=temporary_predictions,
            batch_size=args.document_batch_size,
        )
        report = _report(
            args=args,
            profile=profile,
            pipeline=pipeline,
            temporary_predictions=temporary_predictions,
            document_count=document_count,
            recognizer_identity=recognizer_identity,
            context_identity=context_identity,
            implementation_identity=implementation_identity,
            runtime_versions=runtime_versions,
            runtime_input_identity=runtime_input_identity,
            execution_evidence=execution_evidence,
        )
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(temporary_predictions, args.predictions)
        published.append(args.predictions)
        _publish(temporary_report, args.report)
        published.append(args.report)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "schema_version": ABLATION_EVALUATION_SCHEMA_VERSION,
                    "ablation_id": profile.adapter_id,
                    "run_intent": args.run_intent,
                    "document_count": document_count,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    except SafeAblationCliError as exc:
        for destination in published:
            try:
                destination.chmod(0o600)
                destination.unlink()
            except OSError:
                pass
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError):
        print("error: cascade ablation setup failed", file=os.sys.stderr)
        return 2
    finally:
        for temporary in (temporary_predictions, temporary_report):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


__all__ = [
    "ABLATION_EVALUATION_SCHEMA_VERSION",
    "DOCUMENT_ID_POLICY_VERSION",
    "SafeAblationCliError",
    "main",
    "safe_evaluation_doc_id",
]
