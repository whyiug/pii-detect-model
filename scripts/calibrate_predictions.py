#!/usr/bin/env python3
"""Fit validation-only span calibration without retaining source text."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.calibration import (  # noqa: E402
    CalibrationBundle,
    CalibrationExample,
    TemperatureScaler,
    fit_temperature,
    select_span_thresholds,
)
from pii_zh.data.schema import DataPool, SchemaError, iter_jsonl  # noqa: E402
from pii_zh.evaluation import (  # noqa: E402
    EvaluationDataError,
    add_manifest_hash,
    canonical_json_hash,
    load_prediction_jsonl,
    sha256_file,
)
from pii_zh.taxonomy import load_taxonomy  # noqa: E402

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select span thresholds from synthetic validation gold and raw-score "
            "predictions. Frozen test/evaluation-only data is rejected."
        )
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="CalibrationBundle JSON")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--model-version")
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--t0-recall-floor", type=float, default=0.90)
    parser.add_argument("--t1-recall-floor", type=float, default=0.85)
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--minimum-temperature-samples", type=int, default=30)
    parser.add_argument("--minimum-temperature-class-count", type=int, default=5)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser


def _probability(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


def _write_json_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def _load_validation_gold(
    path: Path, *, allowed_labels: frozenset[str]
) -> tuple[dict[str, frozenset[tuple[int, int, str]]], Counter[str], int]:
    gold_by_document: dict[str, frozenset[tuple[int, int, str]]] = {}
    label_counts: Counter[str] = Counter()
    document_count = 0
    for record in iter_jsonl(path):
        if record.split != "validation":
            raise ValueError("calibration gold must contain only the validation split")
        if not record.provenance.synthetic:
            raise ValueError("calibration gold must contain only synthetic records")
        if record.provenance.evaluation_only or record.data_pool is not DataPool.PUBLIC_RELEASE:
            raise ValueError("evaluation-only and non-public-release records cannot calibrate")
        if not (
            record.public_weight_training_allowed is True
            and record.quality.quality_gate
            and record.quality.validators_passed
        ):
            raise ValueError("calibration gold must pass public, quality, and validator gates")
        if record.doc_id in gold_by_document:
            raise ValueError("calibration gold contains duplicate document identifiers")
        keys = [(entity.start, entity.end, entity.label) for entity in record.entities]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration gold contains duplicate spans")
        unknown = {label for _, _, label in keys} - allowed_labels
        if unknown:
            raise ValueError("calibration gold contains labels outside the output taxonomy")
        gold_by_document[record.doc_id] = frozenset(keys)
        label_counts.update(label for _, _, label in keys)
        document_count += 1
    if not document_count:
        raise ValueError("calibration gold must not be empty")
    return gold_by_document, label_counts, document_count


def _emitted_examples(
    *,
    predictions_path: Path,
    gold_by_document: dict[str, frozenset[tuple[int, int, str]]],
    allowed_labels: frozenset[str],
) -> list[CalibrationExample]:
    predictions = load_prediction_jsonl(predictions_path)
    prediction_ids = {record.doc_id for record in predictions}
    if prediction_ids != set(gold_by_document) or len(predictions) != len(gold_by_document):
        raise ValueError("prediction document set must exactly equal validation gold")
    examples: list[CalibrationExample] = []
    for record in predictions:
        gold_keys = gold_by_document[record.doc_id]
        for span in record.spans:
            if span.label not in allowed_labels:
                raise ValueError("predictions contain labels outside the output taxonomy")
            if span.score is None:
                raise ValueError("every emitted prediction span must have a raw score")
            examples.append(
                CalibrationExample(
                    entity_type=span.label,
                    score=float(span.score),
                    is_positive=span.metric_key() in gold_keys,
                )
            )
    return examples


def _fit_emitted_temperature(
    examples: Sequence[CalibrationExample],
    *,
    enabled: bool,
    minimum_samples: int,
    minimum_class_count: int,
) -> tuple[TemperatureScaler, dict[str, object]]:
    positive_count = sum(example.is_positive for example in examples)
    negative_count = len(examples) - positive_count
    base: dict[str, object] = {
        "enabled": enabled,
        "scope": "emitted_prediction_spans_only",
        "target": "strict_start_end_label_match",
        "un_emitted_gold_used_for_temperature": False,
        "sample_count": len(examples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "minimum_samples": minimum_samples,
        "minimum_class_count": minimum_class_count,
        "probability_clipping_epsilon": 1e-6,
    }
    if not enabled:
        return TemperatureScaler(1.0), {**base, "status": "disabled", "temperature": 1.0}
    if (
        len(examples) < minimum_samples
        or positive_count < minimum_class_count
        or negative_count < minimum_class_count
    ):
        return TemperatureScaler(1.0), {
            **base,
            "status": "insufficient_statistics",
            "temperature": 1.0,
        }
    epsilon = 1e-6
    logits: list[list[float]] = []
    targets: list[int] = []
    for example in examples:
        score = min(max(example.score, epsilon), 1.0 - epsilon)
        logits.append([math.log1p(-score), math.log(score)])
        targets.append(int(example.is_positive))
    scaler = fit_temperature(logits, targets)
    return scaler, {**base, "status": "fitted", "temperature": scaler.temperature}


def _score_diagnostics(
    examples: Sequence[CalibrationExample], *, bin_count: int
) -> dict[str, object]:
    if not examples:
        return {
            "status": "not_applicable",
            "reason": "no_emitted_prediction_spans",
            "scope": "emitted_prediction_spans_only",
            "sample_count": 0,
            "ece": None,
            "brier": None,
        }
    buckets: list[list[CalibrationExample]] = [[] for _ in range(bin_count)]
    for example in examples:
        index = min(int(example.score * bin_count), bin_count - 1)
        buckets[index].append(example)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_score = sum(example.score for example in bucket) / len(bucket)
        accuracy = sum(example.is_positive for example in bucket) / len(bucket)
        ece += len(bucket) / len(examples) * abs(accuracy - mean_score)
    brier = sum((example.score - float(example.is_positive)) ** 2 for example in examples) / len(
        examples
    )
    return {
        "status": "available",
        "reason": None,
        "scope": "emitted_prediction_spans_only",
        "sample_count": len(examples),
        "binning": {
            "method": "equal_width",
            "bin_count": bin_count,
            "intervals": "left_closed_right_open_except_last_closed",
        },
        "ece": ece,
        "brier": brier,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        default_threshold = _probability("default-threshold", args.default_threshold)
        t0_recall_floor = _probability("t0-recall-floor", args.t0_recall_floor)
        t1_recall_floor = _probability("t1-recall-floor", args.t1_recall_floor)
        if args.minimum_temperature_samples < 1:
            raise ValueError("minimum-temperature-samples must be positive")
        if args.minimum_temperature_class_count < 1:
            raise ValueError("minimum-temperature-class-count must be positive")
        if args.ece_bins < 2:
            raise ValueError("ece-bins must be at least two")
        if args.model_version is not None and _SAFE_ID.fullmatch(args.model_version) is None:
            raise ValueError("model-version must be a path-free stable identifier")
        if args.output.resolve() == args.diagnostics.resolve():
            raise ValueError("output and diagnostics must be different files")

        taxonomy = load_taxonomy(args.taxonomy)
        allowed_labels = taxonomy.output_label_names
        gold_by_document, gold_support, document_count = _load_validation_gold(
            args.gold, allowed_labels=allowed_labels
        )
        raw_examples = _emitted_examples(
            predictions_path=args.predictions,
            gold_by_document=gold_by_document,
            allowed_labels=allowed_labels,
        )
        scaler, temperature_diagnostics = _fit_emitted_temperature(
            raw_examples,
            enabled=args.fit_temperature,
            minimum_samples=args.minimum_temperature_samples,
            minimum_class_count=args.minimum_temperature_class_count,
        )
        calibrated_examples = [
            CalibrationExample(
                entity_type=example.entity_type,
                score=scaler.transform_probability(example.score),
                is_positive=example.is_positive,
            )
            for example in raw_examples
        ]

        entities = {entity.name: entity for entity in taxonomy.entities if entity.output}
        beta_by_entity: dict[str, float] = {}
        recall_by_entity: dict[str, float] = {}
        risk_by_entity: dict[str, str] = {}
        for label in set(gold_support) | {example.entity_type for example in raw_examples}:
            risk_tiers = entities[label].risk_tiers
            risk = "T0" if "T0" in risk_tiers else "T1" if "T1" in risk_tiers else "T2"
            risk_by_entity[label] = risk
            beta_by_entity[label] = 2.0 if risk in {"T0", "T1"} else 1.0
            recall_by_entity[label] = (
                t0_recall_floor if risk == "T0" else t1_recall_floor if risk == "T1" else 0.0
            )
        selections = select_span_thresholds(
            calibrated_examples,
            gold_support=gold_support,
            beta_by_entity=beta_by_entity,
            minimum_recall=recall_by_entity,
        )

        input_identity = {
            "gold_sha256": sha256_file(args.gold),
            "predictions_sha256": sha256_file(args.predictions),
        }
        parameters = {
            "default_threshold": default_threshold,
            "t0_recall_floor": t0_recall_floor,
            "t1_recall_floor": t1_recall_floor,
            "temperature_enabled": args.fit_temperature,
            "minimum_temperature_samples": args.minimum_temperature_samples,
            "minimum_temperature_class_count": args.minimum_temperature_class_count,
            "ece_bins": args.ece_bins,
            "threshold_matching": "strict_start_end_label",
            "threshold_objective": "F2_for_T0_T1_F1_for_T2",
        }
        calibration_version = (
            "cal-" + canonical_json_hash({"inputs": input_identity, "parameters": parameters})[:24]
        )
        bundle = CalibrationBundle(
            global_temperature=scaler.temperature,
            entity_thresholds={
                label: selection.threshold for label, selection in selections.items()
            },
            default_threshold=default_threshold,
            model_version=args.model_version,
            calibration_version=calibration_version,
        )
        _write_json_atomic(bundle.to_dict(), args.output)

        emitted_match_count = sum(example.is_positive for example in raw_examples)
        diagnostics = add_manifest_hash(
            {
                "schema_version": 1,
                "manifest_type": "calibration_diagnostics",
                "calibration_version": calibration_version,
                "calibration_bundle_sha256": sha256_file(args.output),
                "taxonomy_version": taxonomy.taxonomy_version,
                "inputs": {
                    **input_identity,
                    "document_count": document_count,
                    "gold_span_count": sum(gold_support.values()),
                    "emitted_prediction_span_count": len(raw_examples),
                    "emitted_strict_match_count": emitted_match_count,
                    "un_emitted_gold_span_count": sum(gold_support.values()) - emitted_match_count,
                },
                "parameters": parameters,
                "temperature": temperature_diagnostics,
                "confidence_calibration": {
                    "before_temperature": _score_diagnostics(raw_examples, bin_count=args.ece_bins),
                    "after_temperature": _score_diagnostics(
                        calibrated_examples, bin_count=args.ece_bins
                    ),
                    "false_negatives_excluded_from_ece_brier": True,
                },
                "per_label": {
                    label: {
                        "risk_tier": risk_by_entity[label],
                        **selection.to_dict(),
                    }
                    for label, selection in sorted(selections.items())
                },
                "privacy": {
                    "contains_paths": False,
                    "contains_raw_text": False,
                    "contains_entity_values": False,
                    "contains_document_ids": False,
                },
            }
        )
        _write_json_atomic(diagnostics, args.diagnostics)
    except (EvaluationDataError, SchemaError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "calibration_version": calibration_version,
                "diagnostics_manifest_sha256": diagnostics["manifest_sha256"],
                "documents": document_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
