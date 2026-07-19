"""Fail-closed rules-only evaluator for versioned production service profiles.

This entry point deliberately supports only development, non-claim execution in
its first revision.  A future confirmatory successor must bind independently
approved human-gold admission and custody evidence before it may read hidden
data.  Both supported profiles are constructed through the public service
factory so evaluation cannot silently substitute a bare ``CascadeConfig``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pii_zh.cascade import (
    SERVICE_PROFILE_VERSIONS,
    CascadeConfig,
    CascadePipeline,
    build_rules_only_service_pipeline,
)
from pii_zh.data.schema import SchemaError, iter_jsonl
from pii_zh.evaluation.io import (
    EvaluationDataError,
    GoldRecord,
    PredictionRecord,
    Span,
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

SERVICE_EVALUATION_SCHEMA_VERSION = "pii-zh.service-evaluation.v1"
SERVICE_EVALUATION_DOCUMENT_ID_VERSION = "pii-zh.service-evaluation-doc-id.v1"
_DOCUMENT_ID_DOMAIN = b"pii-zh.service-evaluation-doc-id.v1\0"
_RUN_INTENTS = ("development-non-claim", "frozen-confirmatory")
_SOURCE_SUFFIXES = frozenset({".json", ".py", ".yaml", ".yml"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SafeServiceEvaluationError(RuntimeError):
    """A fixed, input-value-free service-evaluation failure."""


@dataclass(frozen=True, slots=True)
class _EvaluationDocument:
    safe_id: str
    text: str
    gold: GoldRecord


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-zh-evaluate-service",
        description="Evaluate one fixed-24-class rules-only production service profile.",
    )
    parser.add_argument(
        "--service-profile",
        required=True,
        choices=SERVICE_PROFILE_VERSIONS,
    )
    parser.add_argument(
        "--run-intent",
        required=True,
        choices=_RUN_INTENTS,
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--document-batch-size", type=int, default=32)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def service_evaluation_doc_id(doc_id: str) -> str:
    """Return a stable domain-separated ID without retaining its source value."""

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise EvaluationDataError("doc_id must be a non-empty string")
    digest = hashlib.sha256(_DOCUMENT_ID_DOMAIN + doc_id.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_development_args(args: argparse.Namespace) -> None:
    if args.run_intent != "development-non-claim":
        raise SafeServiceEvaluationError("frozen-confirmatory admission contract unavailable")

    outputs = (_resolved(args.predictions), _resolved(args.report))
    if len(set(outputs)) != 2:
        raise SafeServiceEvaluationError("prediction and report outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in (args.predictions, args.report)):
        raise SafeServiceEvaluationError("refusing to overwrite an existing output")
    if args.input.is_symlink() or not args.input.is_file():
        raise SafeServiceEvaluationError("canonical input must be an existing regular file")
    if _resolved(args.input) in outputs:
        raise SafeServiceEvaluationError("canonical input must be distinct from outputs")
    if (
        isinstance(args.document_batch_size, bool)
        or not isinstance(args.document_batch_size, int)
        or args.document_batch_size < 1
    ):
        raise SafeServiceEvaluationError("document batch size must be positive")


def _fixed_evaluation_scope(
    config: CascadeConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    policy_scope = tuple(config.route_map)
    mapping = load_presidio_mapping()
    model_scope = tuple(
        model_label
        for model_label, policy_label in mapping.model_to_presidio.items()
        if policy_label in policy_scope
    )
    policy_to_model = {
        mapping.model_to_presidio[model_label]: model_label for model_label in model_scope
    }
    if (
        len(policy_scope) != 24
        or len(model_scope) != 24
        or len(set(model_scope)) != 24
        or set(policy_to_model) != set(policy_scope)
    ):
        raise SafeServiceEvaluationError("service profile is not the fixed canonical 24 classes")
    return policy_scope, model_scope, policy_to_model


def _load_documents(input_path: Path, *, model_scope: Sequence[str]) -> list[_EvaluationDocument]:
    allowed_labels = frozenset(model_scope)
    seen: set[str] = set()
    documents: list[_EvaluationDocument] = []
    try:
        for record in iter_jsonl(input_path):
            safe_id = service_evaluation_doc_id(record.doc_id)
            if safe_id in seen:
                raise SafeServiceEvaluationError(
                    "canonical input contains duplicate or colliding document identifiers"
                )
            seen.add(safe_id)
            if any(entity.label not in allowed_labels for entity in record.entities):
                raise SafeServiceEvaluationError(
                    "canonical gold contains a label outside the fixed 24-class scope"
                )
            gold = GoldRecord(
                doc_id=safe_id,
                spans=tuple(
                    Span(start=entity.start, end=entity.end, label=entity.label)
                    for entity in record.entities
                ),
                text_length=len(record.text),
            )
            gold.validate()
            documents.append(_EvaluationDocument(safe_id=safe_id, text=record.text, gold=gold))
    except SafeServiceEvaluationError:
        raise
    except (EvaluationDataError, OSError, SchemaError, TypeError, UnicodeError, ValueError) as exc:
        raise SafeServiceEvaluationError("canonical input validation failed") from exc
    if not documents:
        raise SafeServiceEvaluationError("canonical input must contain at least one document")
    return documents


def _temporary_path(destination: Path) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            delete=False,
            prefix=f".{destination.name}.",
        )
    except OSError as exc:
        raise SafeServiceEvaluationError("failed to create a private temporary output") from exc
    path = Path(handle.name)
    handle.close()
    return path


def _generate_predictions(
    *,
    pipeline: CascadePipeline,
    documents: Sequence[_EvaluationDocument],
    temporary: Path,
    batch_size: int,
    policy_to_model: Mapping[str, str],
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for offset in range(0, len(documents), batch_size):
                batch = documents[offset : offset + batch_size]
                detected = pipeline.detect_batch(
                    [document.text for document in batch],
                    entities=None,
                )
                batch_predictions = [
                    PredictionRecord(
                        doc_id=document.safe_id,
                        spans=tuple(
                            Span(
                                start=item.start,
                                end=item.end,
                                label=policy_to_model[item.entity_type],
                                score=item.score,
                            )
                            for item in detections
                        ),
                    )
                    for document, detections in zip(batch, detected, strict=True)
                ]
                write_prediction_jsonl(batch_predictions, handle)
                predictions.extend(batch_predictions)
    except (EvaluationDataError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SafeServiceEvaluationError("rules-only service inference failed") from exc
    return predictions


def _configuration_payload(config: CascadeConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "profile_version": config.profile_version,
        "mode": config.mode,
        "rule_source": config.rule_source,
        "model_source": config.model_source,
        "keep_nested_different_types": config.keep_nested_different_types,
        "routes": [asdict(route) for route in config.routes],
    }


def _package_source_closure() -> dict[str, object]:
    package_root = Path(__file__).resolve().parents[1]
    sources: list[Path] = []
    try:
        for candidate in package_root.rglob("*"):
            if candidate.is_symlink():
                raise SafeServiceEvaluationError("package source closure contains an unsafe file")
            if candidate.suffix not in _SOURCE_SUFFIXES:
                continue
            if not candidate.is_file():
                raise SafeServiceEvaluationError("package source closure contains an unsafe file")
            sources.append(candidate)
        components = {
            candidate.relative_to(package_root).as_posix(): sha256_file(candidate)
            for candidate in sorted(sources)
        }
    except SafeServiceEvaluationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SafeServiceEvaluationError("package source closure verification failed") from exc
    if not components or any(_SHA256_RE.fullmatch(value) is None for value in components.values()):
        raise SafeServiceEvaluationError("package source closure is empty or invalid")
    return {
        "closure_mode": "full_pii_zh_python_and_packaged_resource_superset_v1",
        "source_file_count": len(components),
        "component_sha256": components,
        "implementation_sha256": canonical_json_hash(components),
    }


def _build_report(
    *,
    args: argparse.Namespace,
    pipeline: CascadePipeline,
    documents: Sequence[_EvaluationDocument],
    predictions: Sequence[PredictionRecord],
    temporary_predictions: Path,
    input_sha256: str,
    policy_scope: Sequence[str],
    model_scope: Sequence[str],
) -> dict[str, Any]:
    gold = [document.gold for document in documents]
    try:
        persisted_predictions = load_prediction_jsonl(temporary_predictions)
        if list(predictions) != persisted_predictions:
            raise EvaluationDataError("persisted predictions differ from in-memory predictions")
        report = evaluate_documents(
            gold,
            persisted_predictions,
            iou_threshold=args.iou_threshold,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        report["required_metrics"] = build_required_metrics(gold, persisted_predictions)
        provenance = build_evaluation_provenance(
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
        raise SafeServiceEvaluationError("strict service evaluation failed") from exc
    if provenance["gold"]["sha256"] != input_sha256:
        raise SafeServiceEvaluationError("canonical input changed during evaluation")

    configuration = _configuration_payload(pipeline.config)
    stage_policy = pipeline.stage_policy.to_dict()
    scope = {
        "scope_mode": "fixed_canonical_24_input_and_service_routing",
        "entity_count": 24,
        "metric_model_labels": list(model_scope),
        "pipeline_policy_entities": list(policy_scope),
        "core_evaluator_macro_denominator": (
            "implementation_defined_observed_label_scope_may_apply"
        ),
        "formal_fixed_24_macro_status": "NOT_CLAIMED_IN_THIS_DEVELOPMENT_RUN",
        "formal_fixed_24_macro_owner": "suite_core_recomputation",
    }
    predictions_sha256 = sha256_file(temporary_predictions)
    report["provenance"] = provenance
    report["service_evaluation_runtime"] = {
        "schema_version": SERVICE_EVALUATION_SCHEMA_VERSION,
        "run_intent": args.run_intent,
        "service_profile": pipeline.config.profile_version,
        "service_factory": "pii_zh.cascade.build_rules_only_service_pipeline",
        "execution_device": "cpu",
        "mode": "rules-only",
        "configuration": configuration,
        "configuration_sha256": canonical_json_hash(configuration),
        "stage_policy": stage_policy,
        "stage_policy_sha256": canonical_json_hash(stage_policy),
        "canonical_evaluation_scope": scope,
        "canonical_evaluation_scope_sha256": canonical_json_hash(scope),
        "package_source_closure": _package_source_closure(),
        "canonical_input_sha256": input_sha256,
        "predictions_sha256": predictions_sha256,
        "prediction_document_count": len(predictions),
        "document_batch_size": args.document_batch_size,
        "document_id_policy": {
            "schema_version": SERVICE_EVALUATION_DOCUMENT_ID_VERSION,
            "algorithm": "domain_separated_sha256",
            "serialized_format": "sha256:<64-lowercase-hex>",
            "original_document_ids_persisted": False,
        },
        "human_gold_admission_contract_status": "UNAVAILABLE",
        "claim_eligible": False,
        "claim_status": "DEVELOPMENT_NON_CLAIM",
        "privacy": {
            "contains_absolute_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_original_document_ids": False,
        },
    }
    return report


def _publish(temporary: Path, destination: Path) -> None:
    linked = False
    try:
        os.link(temporary, destination)
        linked = True
        destination.chmod(0o400)
    except OSError as exc:
        if linked:
            try:
                destination.chmod(0o600)
                destination.unlink()
            except OSError:
                pass
        raise SafeServiceEvaluationError("failed to publish a new output") from exc


def _remove_published(path: Path) -> None:
    try:
        path.chmod(0o600)
        path.unlink()
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    temporary_predictions: Path | None = None
    temporary_report: Path | None = None
    published: list[Path] = []
    try:
        _validate_development_args(args)
        pipeline = build_rules_only_service_pipeline(args.service_profile)
        if pipeline.config.mode != "rules-only":
            raise SafeServiceEvaluationError("service factory did not return rules-only mode")
        policy_scope, model_scope, policy_to_model = _fixed_evaluation_scope(pipeline.config)
        input_sha256 = sha256_file(args.input)
        documents = _load_documents(args.input, model_scope=model_scope)
        if sha256_file(args.input) != input_sha256:
            raise SafeServiceEvaluationError("canonical input changed during evaluation")

        temporary_predictions = _temporary_path(args.predictions)
        predictions = _generate_predictions(
            pipeline=pipeline,
            documents=documents,
            temporary=temporary_predictions,
            batch_size=args.document_batch_size,
            policy_to_model=policy_to_model,
        )
        report = _build_report(
            args=args,
            pipeline=pipeline,
            documents=documents,
            predictions=predictions,
            temporary_predictions=temporary_predictions,
            input_sha256=input_sha256,
            policy_scope=policy_scope,
            model_scope=model_scope,
        )
        temporary_report = _temporary_path(args.report)
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
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
                    "schema_version": SERVICE_EVALUATION_SCHEMA_VERSION,
                    "run_intent": args.run_intent,
                    "service_profile": args.service_profile,
                    "document_count": len(documents),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    except SafeServiceEvaluationError as exc:
        for destination in published:
            _remove_published(destination)
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError):
        for destination in published:
            _remove_published(destination)
        print("error: service evaluation setup failed", file=os.sys.stderr)
        return 2
    finally:
        for temporary in (temporary_predictions, temporary_report):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


__all__ = [
    "SERVICE_EVALUATION_DOCUMENT_ID_VERSION",
    "SERVICE_EVALUATION_SCHEMA_VERSION",
    "SafeServiceEvaluationError",
    "main",
    "service_evaluation_doc_id",
]


if __name__ == "__main__":
    raise SystemExit(main())
