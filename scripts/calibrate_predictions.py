#!/usr/bin/env python3
"""Fit post-selection span calibration without retaining source text.

The default path remains development-validation calibration.  A frozen
``evaluation_only`` calibration split is accepted only when the caller binds
the exact self-hashed dataset manifest and that manifest explicitly permits
post-selection calibration while forbidding model/checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
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
_RELEASE_EVAL_V2_DATASET_ID = "pii_zh_synthetic_sota_release_eval_v2"
_RELEASE_EVAL_V2_DATASET_VERSION = "2.0.0"
_RELEASE_EVAL_V2_CONFIG_SHA256 = "23c5c15a27c92a2110b6bd5fc180f62dced81cdc17be61b44dbc30194e5dba30"
_RELEASE_EVAL_V2_GENERATION_SALT_SHA256 = (
    "69ecd313a414459ece55274a7d7ce881537fb8f57e51940068b4a190a6fa437e"
)
_RELEASE_EVAL_V2_FREEZE_FILE_SHA256 = (
    "06ebdce5e288b0e7699a8225e28c4c4f0a1762e76f6967cf990b14384101f0f0"
)
_RELEASE_EVAL_V2_MANIFEST_SHA256 = (
    "bd99b9a858c07fe96f1effcc93d07defa17f7ce39a00712bcfe6b9fda56c19ad"
)
_RELEASE_EVAL_V2_FROZEN_AT_UTC = "2026-07-18T00:50:01Z"
_RELEASE_EVAL_V2_SPLIT_COUNTS = {"calibration": 10_000, "internal_evaluation": 10_000}
_WITHDRAWN_RELEASE_EVAL_V1_DATASET_ID = "pii_zh_synthetic_sota_release_eval_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select span thresholds from synthetic validation gold or an explicitly "
            "bound frozen calibration split and raw-score predictions."
        )
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="CalibrationBundle JSON")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--model-version")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help=(
            "Self-hashed release-evaluation dataset manifest. Required only when "
            "--gold is an evaluation_only calibration split."
        ),
    )
    parser.add_argument(
        "--expected-dataset-manifest-sha256",
        help=(
            "Frozen canonical manifest SHA-256. Must be supplied together with --dataset-manifest."
        ),
    )
    parser.add_argument(
        "--supersession-freeze",
        type=Path,
        help=(
            "Frozen v2 supersession receipt. Required together with the release "
            "evaluation dataset manifest."
        ),
    )
    parser.add_argument(
        "--expected-supersession-freeze-sha256",
        help="Pre-registered physical SHA-256 of the frozen v2 supersession receipt.",
    )
    parser.add_argument(
        "--release-eval-v2-authorization",
        type=Path,
        help="Strictly replayed calibration pre-open authorization for the formal v2 split.",
    )
    parser.add_argument(
        "--fit-prediction-manifest",
        type=Path,
        help="Strict-replayed model_raw calibration prediction provenance manifest.",
    )
    parser.add_argument(
        "--allowed-label",
        action="append",
        dest="allowed_labels",
        help=(
            "Calibrate only this output label and ignore other valid taxonomy gold. "
            "Repeat for a frozen reduced-label protocol."
        ),
    )
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
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination.name}")
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
    temporary.chmod(0o444)
    try:
        # A same-filesystem hard link publishes the fully-written temporary file
        # atomically and, unlike os.replace(), fails when the destination exists.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_frozen_calibration_contract(
    manifest_path: Path,
    *,
    gold_path: Path,
    expected_manifest_sha256: str,
    supersession_freeze_path: Path,
    expected_supersession_freeze_sha256: str,
) -> dict[str, object]:
    """Verify the exact v2 supersession and post-selection calibration contract."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError("expected dataset manifest hash must be a lowercase SHA-256")
    if re.fullmatch(r"[0-9a-f]{64}", expected_supersession_freeze_sha256) is None:
        raise ValueError("expected supersession freeze hash must be a lowercase SHA-256")
    if expected_supersession_freeze_sha256 != _RELEASE_EVAL_V2_FREEZE_FILE_SHA256:
        raise ValueError("supersession freeze hash is not the pre-registered v2 hash")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("dataset manifest must be a regular file")
    if supersession_freeze_path.is_symlink() or not supersession_freeze_path.is_file():
        raise ValueError("supersession freeze must be a regular file")
    try:
        freeze_bytes = supersession_freeze_path.read_bytes()
        freeze_receipt = json.loads(
            freeze_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("supersession freeze must be valid UTF-8 JSON") from exc
    if not isinstance(freeze_receipt, dict):
        raise ValueError("supersession freeze must be a JSON object")
    observed_freeze_file_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    if observed_freeze_file_sha256 != expected_supersession_freeze_sha256:
        raise ValueError("supersession freeze does not match the pre-registered hash")
    expected_chronology = {
        "frozen_before_any_v3_seed42_validation_result": True,
        "v1_confirmatory_calibration_withdrawn": True,
        "v1_structural_pre_read_one_record_no_model_result": True,
        "v2_zero_content_read_before_cross_seed_selection": True,
    }
    expected_protocol_effect = {
        "does_not_modify_frozen_v3_training_protocol_or_selector": True,
        "v1_may_not_be_used_for_zero-read_confirmatory_calibration": True,
        "v2_calibration_content_must_remain_unread_until_cross-seed_model_selection": True,
        (
            "v2_internal_evaluation_content_must_remain_unread_until_model_service_"
            "and_calibration_are_frozen"
        ): True,
    }
    frozen_design = freeze_receipt.get("frozen_design")
    supersession = freeze_receipt.get("supersession")
    if (
        freeze_receipt.get("receipt_schema_version") != 1
        or freeze_receipt.get("receipt_id") != "synthetic_sota_release_eval_v2_supersession_freeze"
        or freeze_receipt.get("frozen_at_utc") != _RELEASE_EVAL_V2_FROZEN_AT_UTC
        or freeze_receipt.get("chronology") != expected_chronology
        or freeze_receipt.get("protocol_effect") != expected_protocol_effect
        or not isinstance(frozen_design, dict)
        or frozen_design.get("config_file_sha256") != _RELEASE_EVAL_V2_CONFIG_SHA256
        or frozen_design.get("generation_salt_sha256") != _RELEASE_EVAL_V2_GENERATION_SALT_SHA256
        or frozen_design.get("calibration_records") != _RELEASE_EVAL_V2_SPLIT_COUNTS["calibration"]
        or frozen_design.get("internal_evaluation_records")
        != _RELEASE_EVAL_V2_SPLIT_COUNTS["internal_evaluation"]
        or frozen_design.get("core_label_count") != 24
        or frozen_design.get("record_data_pool") != "evaluation_only"
        or frozen_design.get("materialization_pending_at_freeze") is not True
        or not isinstance(supersession, dict)
        or supersession.get("withdrawn_dataset_id") != _WITHDRAWN_RELEASE_EVAL_V1_DATASET_ID
        or supersession.get("withdrawn_dataset_version") != "1.0.0"
        or supersession.get("withdrawn_role") != "zero-read confirmatory calibration"
        or supersession.get("replacement_dataset_id") != _RELEASE_EVAL_V2_DATASET_ID
        or supersession.get("replacement_dataset_version") != _RELEASE_EVAL_V2_DATASET_VERSION
        or supersession.get("replacement_role")
        != "unread post-selection calibration plus one-shot internal evaluation"
    ):
        raise ValueError("supersession freeze contract is invalid")

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("dataset manifest must be valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a JSON object")
    claimed_hash = manifest.get("manifest_sha256")
    if claimed_hash != expected_manifest_sha256:
        raise ValueError("dataset manifest does not match the frozen canonical hash")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if canonical_json_hash(unsigned) != claimed_hash:
        raise ValueError("dataset manifest self-hash is invalid")
    dataset_id = manifest.get("dataset_id")
    dataset_version = manifest.get("dataset_version")
    if dataset_id == _WITHDRAWN_RELEASE_EVAL_V1_DATASET_ID:
        raise ValueError("release evaluation v1 is withdrawn and cannot calibrate")
    schema_version = manifest.get("manifest_schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("dataset manifest schema version must be integer 2")
    if (
        dataset_id != _RELEASE_EVAL_V2_DATASET_ID
        or dataset_version != _RELEASE_EVAL_V2_DATASET_VERSION
    ):
        raise ValueError("dataset manifest is not the frozen release evaluation v2 identity")

    usage = manifest.get("usage_policy")
    generation = manifest.get("generation")
    audits = manifest.get("audits")
    if (
        not isinstance(usage, dict)
        or set(usage) != {"calibration", "internal_evaluation"}
        or not isinstance(usage.get("calibration"), dict)
    ):
        raise ValueError("dataset manifest lacks a calibration usage policy")
    calibration = usage["calibration"]
    expected_calibration_policy = {
        "checkpoint_or_model_selection_allowed": False,
        "final_metric_claim_allowed": False,
        "fusion_calibration_allowed": True,
        "gradient_training_allowed": False,
        "temperature_scaling_allowed": True,
        "threshold_tuning_allowed": True,
        "content_read_before_cross_seed_selection_allowed": False,
    }
    if calibration != expected_calibration_policy:
        raise ValueError("dataset manifest calibration policy is not fail-closed")
    internal = usage.get("internal_evaluation")
    expected_internal_policy = {
        "checkpoint_or_model_selection_allowed": False,
        "final_metric_claim_allowed": True,
        "fusion_calibration_allowed": False,
        "gradient_training_allowed": False,
        "result_informed_model_or_threshold_change_allowed": False,
        "temperature_scaling_allowed": False,
        "threshold_tuning_allowed": False,
        "content_read_before_final_freeze_allowed": False,
    }
    if internal != expected_internal_policy:
        raise ValueError("dataset manifest does not isolate internal evaluation")
    cross_corpus = audits.get("cross_corpus_isolation") if isinstance(audits, dict) else None
    expected_manifest_supersession = {
        "v1_structural_pre_read_one_record_no_model_result": True,
        "v1_confirmatory_calibration_withdrawn": True,
        "v2_zero_content_read_before_cross_seed_selection": True,
        "replacement_scope": "confirmatory_calibration_and_internal_evaluation",
    }
    expected_manifest_freeze = {
        "frozen_at_utc": _RELEASE_EVAL_V2_FROZEN_AT_UTC,
        "frozen_before_any_v3_seed42_validation_result": True,
        "supersession_freeze_receipt": {
            "path": "configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json",
            "file_sha256": observed_freeze_file_sha256,
            "receipt_id": "synthetic_sota_release_eval_v2_supersession_freeze",
            "frozen_at_utc": _RELEASE_EVAL_V2_FROZEN_AT_UTC,
            "chronology": expected_chronology,
        },
    }
    counts = manifest.get("counts")
    if (
        manifest.get("record_data_pool") != "evaluation_only"
        or manifest.get("publication_pool") != "public_release_pool"
        or manifest.get("public_weight_training_allowed") is not False
        or manifest.get("public_artifact_release_allowed") is not True
        or manifest.get("freeze") != expected_manifest_freeze
        or manifest.get("config")
        != {
            "path": "configs/data/synthetic_sota_release_eval_v2.yaml",
            "sha256": _RELEASE_EVAL_V2_CONFIG_SHA256,
        }
        or manifest.get("supersession") != expected_manifest_supersession
        or not isinstance(generation, dict)
        or generation.get("generation_salt_sha256") != _RELEASE_EVAL_V2_GENERATION_SALT_SHA256
        or generation.get("requested_split_counts") != _RELEASE_EVAL_V2_SPLIT_COUNTS
        or generation.get("seeds_and_salt_independent_from_v1") is not True
        or generation.get("contains_real_personal_data") is not False
        or generation.get("protected_evaluation_data_read") is not False
        or generation.get("external_dataset_inputs") != []
        or generation.get("network_used") is not False
        or generation.get("gpu_used") is not False
        or generation.get("jsonl_rows_printed") is not False
        or not isinstance(counts, dict)
        or counts.get("by_split") != _RELEASE_EVAL_V2_SPLIT_COUNTS
        or counts.get("total") != sum(_RELEASE_EVAL_V2_SPLIT_COUNTS.values())
        or counts.get("pii_free") != 9_000
        or counts.get("hard_negative") != 9_000
        or counts.get("core_labels") != 24
        or not isinstance(cross_corpus, dict)
        or cross_corpus.get("all_collision_counts_zero") is not True
    ):
        raise ValueError("dataset manifest v2 freeze/privacy/isolation contract is invalid")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(_RELEASE_EVAL_V2_SPLIT_COUNTS):
        raise ValueError("dataset manifest must bind exactly both v2 splits")
    files_by_split: dict[str, dict[str, object]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("every dataset manifest file entry must be an object")
        name = item.get("name")
        split = item.get("split")
        digest = item.get("sha256")
        records = item.get("records")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(split, str)
            or split not in _RELEASE_EVAL_V2_SPLIT_COUNTS
            or name != f"{split}.jsonl"
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(records) is not int
            or records != _RELEASE_EVAL_V2_SPLIT_COUNTS[split]
        ):
            raise ValueError("dataset manifest file entry has malformed identity or counts")
        if split in files_by_split:
            raise ValueError("dataset manifest contains a duplicate split file")
        files_by_split[split] = item
    if set(files_by_split) != set(_RELEASE_EVAL_V2_SPLIT_COUNTS):
        raise ValueError("dataset manifest v2 split binding is incomplete")
    entry = files_by_split["calibration"]
    records = entry.get("records")
    if entry.get("sha256") != sha256_file(gold_path) or type(records) is not int or records < 1:
        raise ValueError("calibration gold disagrees with the frozen dataset manifest")
    return {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "generation_salt_sha256": _RELEASE_EVAL_V2_GENERATION_SALT_SHA256,
        "gold_sha256": entry["sha256"],
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_sha256": claimed_hash,
        "records": records,
        "role": "post_cross_seed_selection_calibration_only",
        "supersession_freeze_file_sha256": observed_freeze_file_sha256,
    }


def _load_validation_gold(
    path: Path,
    *,
    taxonomy_labels: frozenset[str],
    selected_labels: frozenset[str],
    frozen_contract: dict[str, object] | None = None,
) -> tuple[dict[str, frozenset[tuple[int, int, str]]], Counter[str], int]:
    gold_by_document: dict[str, frozenset[tuple[int, int, str]]] = {}
    label_counts: Counter[str] = Counter()
    document_count = 0
    for record in iter_jsonl(path):
        expected_split = "calibration" if frozen_contract is not None else "validation"
        if record.split != expected_split:
            raise ValueError(f"calibration gold must contain only the {expected_split} split")
        if not record.provenance.synthetic:
            raise ValueError("calibration gold must contain only synthetic records")
        if frozen_contract is None:
            if record.provenance.evaluation_only or record.data_pool is not DataPool.PUBLIC_RELEASE:
                raise ValueError("evaluation-only calibration requires a frozen dataset manifest")
            if record.public_weight_training_allowed is not True:
                raise ValueError("development calibration gold must allow public training")
        else:
            metadata = record.metadata
            provenance_metadata = record.provenance.metadata
            if (
                not record.provenance.evaluation_only
                or record.data_pool is not DataPool.EVALUATION_ONLY
                or record.public_weight_training_allowed is not False
                or metadata.get("release_evaluation_split") != "calibration"
                or metadata.get("usage_policy") != "post_cross_seed_selection_calibration_only"
                or metadata.get("synthetic_release_eval_recipe")
                != frozen_contract["dataset_version"]
                or metadata.get("generation_salt_sha256")
                != frozen_contract["generation_salt_sha256"]
                or provenance_metadata.get("release_evaluation_dataset_id")
                != frozen_contract["dataset_id"]
                or provenance_metadata.get("release_evaluation_dataset_version")
                != frozen_contract["dataset_version"]
                or provenance_metadata.get("release_evaluation_split") != "calibration"
                or provenance_metadata.get("generation_salt_sha256")
                != frozen_contract["generation_salt_sha256"]
                or provenance_metadata.get("weight_training_allowed_by_protocol") is not False
            ):
                raise ValueError("calibration record disagrees with the frozen role contract")
        if not (record.quality.quality_gate and record.quality.validators_passed):
            raise ValueError("calibration gold must pass quality and validator gates")
        if record.doc_id in gold_by_document:
            raise ValueError("calibration gold contains duplicate document identifiers")
        all_keys = [(entity.start, entity.end, entity.label) for entity in record.entities]
        if len(all_keys) != len(set(all_keys)):
            raise ValueError("calibration gold contains duplicate spans")
        unknown = {label for _, _, label in all_keys} - taxonomy_labels
        if unknown:
            raise ValueError("calibration gold contains labels outside the output taxonomy")
        keys = [item for item in all_keys if item[2] in selected_labels]
        gold_by_document[record.doc_id] = frozenset(keys)
        label_counts.update(label for _, _, label in keys)
        document_count += 1
    if not document_count:
        raise ValueError("calibration gold must not be empty")
    if frozen_contract is not None and document_count != frozen_contract["records"]:
        raise ValueError("calibration document count disagrees with the frozen manifest")
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
        frozen_arguments = (
            args.dataset_manifest,
            args.expected_dataset_manifest_sha256,
            args.supersession_freeze,
            args.expected_supersession_freeze_sha256,
        )
        if any(item is not None for item in frozen_arguments) and not all(
            item is not None for item in frozen_arguments
        ):
            raise ValueError(
                "dataset-manifest, expected-dataset-manifest-sha256, supersession-freeze, "
                "and expected-supersession-freeze-sha256 are required together"
            )
        if args.output.resolve() == args.diagnostics.resolve():
            raise ValueError("output and diagnostics must be different files")
        for destination in (args.output, args.diagnostics):
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"refusing to overwrite existing output: {destination.name}")

        frozen_contract = None
        if args.dataset_manifest is not None:
            if (
                not isinstance(args.expected_dataset_manifest_sha256, str)
                or not isinstance(args.supersession_freeze, Path)
                or not isinstance(args.expected_supersession_freeze_sha256, str)
            ):
                raise ValueError("frozen calibration arguments have malformed types")
            frozen_contract = _load_frozen_calibration_contract(
                args.dataset_manifest,
                gold_path=args.gold,
                expected_manifest_sha256=args.expected_dataset_manifest_sha256,
                supersession_freeze_path=args.supersession_freeze,
                expected_supersession_freeze_sha256=args.expected_supersession_freeze_sha256,
            )
        formal_authorization: dict[str, object] | None = None
        if args.expected_dataset_manifest_sha256 == _RELEASE_EVAL_V2_MANIFEST_SHA256:
            if (
                args.release_eval_v2_authorization is None
                or args.fit_prediction_manifest is None
                or args.model_version is None
            ):
                raise ValueError(
                    "formal v2 calibration requires authorization, fit prediction manifest, "
                    "and selected training-manifest model-version"
                )
            from pii_zh.evaluation.release_eval_v2_prediction_provenance import (
                validate_release_eval_v2_prediction_manifest,
            )
            from pii_zh.evaluation.release_eval_v2_stage_gate import (
                validate_calibration_authorization,
            )

            authorization_bytes = args.release_eval_v2_authorization.read_bytes()
            authorization = json.loads(
                authorization_bytes.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(authorization, dict):
                raise ValueError("formal calibration authorization must be a JSON object")
            authorization_self = validate_calibration_authorization(authorization)
            prediction_provenance = validate_release_eval_v2_prediction_manifest(
                args.fit_prediction_manifest
            )
            authorization_dataset = authorization["dataset"]
            authorization_model = authorization["model"]
            if (
                not isinstance(authorization_dataset, dict)
                or not isinstance(authorization_model, dict)
                or authorization_dataset.get("gold_file_sha256") != sha256_file(args.gold)
                or authorization_dataset.get("dataset_manifest_sha256")
                != args.expected_dataset_manifest_sha256
                or authorization_model.get("training_manifest_sha256") != args.model_version
                or prediction_provenance.get("track") != "model_raw"
                or prediction_provenance.get("target_split") != "calibration"
                or prediction_provenance.get("predictions_file_sha256")
                != sha256_file(args.predictions)
                or prediction_provenance.get("training_manifest_sha256") != args.model_version
            ):
                raise ValueError(
                    "formal calibration authorization/prediction/model chain is inconsistent"
                )
            formal_authorization = {
                "authorization_file_sha256": hashlib.sha256(
                    authorization_bytes
                ).hexdigest(),
                "authorization_sha256": authorization_self,
                "fit_prediction_manifest_file_sha256": sha256_file(
                    args.fit_prediction_manifest
                ),
                "fit_prediction_manifest_sha256": prediction_provenance["manifest_sha256"],
                "selected_training_manifest_sha256": args.model_version,
            }
        elif (
            args.release_eval_v2_authorization is not None
            or args.fit_prediction_manifest is not None
        ):
            raise ValueError(
                "formal v2 authorization arguments require the exact frozen v2 manifest"
            )

        taxonomy = load_taxonomy(args.taxonomy)
        taxonomy_labels = taxonomy.output_label_names
        if args.allowed_labels is None:
            allowed_labels = taxonomy_labels
        else:
            allowed_labels = frozenset(args.allowed_labels)
            if (
                not allowed_labels
                or len(allowed_labels) != len(args.allowed_labels)
                or not allowed_labels <= taxonomy.core_label_names
            ):
                raise ValueError("allowed-label values must be unique core taxonomy labels")
        gold_by_document, gold_support, document_count = _load_validation_gold(
            args.gold,
            taxonomy_labels=taxonomy_labels,
            selected_labels=allowed_labels,
            frozen_contract=frozen_contract,
        )
        if frozen_contract is not None and sha256_file(args.gold) != frozen_contract["gold_sha256"]:
            raise ValueError("calibration gold changed after manifest verification")
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

        final_gold_sha256 = sha256_file(args.gold)
        if frozen_contract is not None and final_gold_sha256 != frozen_contract["gold_sha256"]:
            raise ValueError("calibration gold changed before output publication")
        input_identity = {
            "gold_sha256": final_gold_sha256,
            "predictions_sha256": sha256_file(args.predictions),
        }
        if frozen_contract is not None:
            input_identity["dataset_manifest_sha256"] = frozen_contract["manifest_sha256"]
            input_identity["dataset_manifest_file_sha256"] = frozen_contract["manifest_file_sha256"]
            input_identity["supersession_freeze_file_sha256"] = frozen_contract[
                "supersession_freeze_file_sha256"
            ]
        if formal_authorization is not None:
            input_identity.update(formal_authorization)
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
            "selected_labels": sorted(allowed_labels),
            "gold_role": (
                "frozen_post_selection_calibration"
                if frozen_contract is not None
                else "development_validation"
            ),
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
                "dataset_contract": frozen_contract,
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
