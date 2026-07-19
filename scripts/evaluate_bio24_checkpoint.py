#!/usr/bin/env python3
"""Freeze and run a post-hoc 24-class BIO checkpoint evaluation.

The benchmark view is PII Bench ZH closed-eight.  A candidate checkpoint must
still expose the project's complete 24-core-label BIO head; the other sixteen
labels are masked at the logits level before softmax, argmax, and BIO decoding.

This entrypoint deliberately has two phases:

``freeze``
    Bind the completed local model artifact, frozen dataset manifest, active
    model-raw comparator reports/predictions, inference parameters, and paired
    bootstrap parameters without reading gold records.

``evaluate``
    Re-verify the content-addressed freeze plan, read the fixed gold exactly for
    scoring, write private record-level predictions to a fresh run directory,
    and emit aggregate-only report and receipt artifacts.

PII Bench ZH is public and this project has prior benchmark exposure.  Every
result is therefore marked ``public_test_exposed`` and ``posthoc_lineage`` and
is prohibited from checkpoint, threshold, or recipe selection.  The active
comparator envelope is descriptive evidence, never an uncontaminated claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from scripts import evaluate_aiguard_baseline as baseline_helpers  # noqa: E402

from pii_zh.evaluation import (  # noqa: E402
    canonical_json_hash,
    compact_strict_result,
    evaluate_documents,
    load_gold_jsonl,
    load_prediction_jsonl,
    pool_strict_results,
    rounded_score,
    sha256_file,
)
from pii_zh.evaluation.comparator_registry import (  # noqa: E402
    load_comparator_registry,
    validate_comparator_registry,
)
from pii_zh.evaluation.paired_bootstrap import (  # noqa: E402
    build_paired_suite,
    paired_bootstrap_delta,
)
from pii_zh.inference.transformers_predictor import load_local_predictor  # noqa: E402
from pii_zh.presidio import QwenPiiRecognizer  # noqa: E402
from pii_zh.taxonomy import load_taxonomy  # noqa: E402
from pii_zh.training.data import build_bio_label_maps  # noqa: E402

PROTOCOL_ID = "pii_bench_zh_closed_8_v1"
PLAN_TYPE = "bio24_pii_bench_zh_model_raw_frozen_evaluation"
REPORT_TYPE = "bio24_model_raw_posthoc_evaluation"
RECEIPT_TYPE = "bio24_model_raw_content_addressed_run_receipt"
DATASET_ID = "wan9yu/pii-bench-zh"
DATASET_REVISION = "c350b94897af668517ff5de237d89f2ce2eaa6f0"
DATASET_MANIFEST_SHA256 = "545c990c96f23118ddc097ec0bed1d7f180785ac72e9f2a6ed56eb1ba3ee5e9b"
SUITES = ("formal", "chat")
COMPARISON_SUITES = (*SUITES, "pooled")
CLOSED_8_LABELS = (
    "ADDRESS",
    "BANK_CARD_NUMBER",
    "CN_RESIDENT_ID",
    "EMAIL_ADDRESS",
    "PASSPORT_NUMBER",
    "PERSON_NAME",
    "PHONE_NUMBER",
    "VEHICLE_LICENSE_PLATE",
)
FIXED_SUBSETS: dict[str, dict[str, Any]] = {
    "formal": {
        "canonical_sha256": ("bb978137711cc90ae0c16df0645983413f3b34943bf29469e58b795b183a38af"),
        "record_count": 5_000,
        "gold_span_count": 15_662,
    },
    "chat": {
        "canonical_sha256": ("2d73326a7ea2da9966ba13336db1a791f03d5f18b7982fdc4495cb726c1180a5"),
        "record_count": 3_000,
        "gold_span_count": 7_544,
    },
}
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 20_260_717

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FORBIDDEN_WEIGHT_SUFFIXES = {".bin", ".pkl", ".pickle", ".pt", ".pth"}
_BOUND_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)
_FORBIDDEN_PUBLIC_KEYS = {
    "doc_id",
    "document_id",
    "entity_text",
    "entity_value",
    "raw_text",
    "text",
}


class Bio24EvaluationError(RuntimeError):
    """Raised when a frozen input, model artifact, or report fails closed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="write a content-addressed plan")
    _add_shared_identity_arguments(freeze)
    freeze.add_argument("--candidate-id", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--max-tokens", type=int, default=512)
    freeze.add_argument("--stride-fraction", type=float, default=0.25)
    freeze.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    freeze.add_argument("--bootstrap-confidence", type=float, default=DEFAULT_BOOTSTRAP_CONFIDENCE)
    freeze.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)

    evaluate = commands.add_parser("evaluate", help="run one frozen evaluation")
    _add_shared_identity_arguments(evaluate)
    evaluate.add_argument("--freeze-plan", required=True, type=Path)
    evaluate.add_argument("--formal", required=True, type=Path)
    evaluate.add_argument("--chat", required=True, type=Path)
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--document-batch-size", type=int, default=64)
    evaluate.add_argument("--window-batch-size", type=int, default=32)
    evaluate.add_argument("--log-every-documents", type=int, default=500)
    return parser


def _add_shared_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--comparator-registry", required=True, type=Path)
    parser.add_argument(
        "--comparator-prediction",
        action="append",
        nargs=3,
        default=[],
        metavar=("COMPARATOR_ID", "SUITE", "PATH"),
        help="repeat once per active comparator and formal/chat suite",
    )


def _load_object(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Bio24EvaluationError(f"{kind} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise Bio24EvaluationError(f"{kind} must be a JSON object")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Bio24EvaluationError(f"{field} must be an object")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise Bio24EvaluationError(f"{field} must be a safe non-empty identifier")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Bio24EvaluationError(f"{field} must be a lowercase SHA-256")
    return value


def _score(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Bio24EvaluationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise Bio24EvaluationError(f"{field} must be finite and in [0, 1]")
    return result


def _nested(value: Mapping[str, Any], selector: str, *, field: str) -> Mapping[str, Any]:
    current: object = value
    for component in selector.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise Bio24EvaluationError(f"{field} selector does not exist")
        current = current[component]
    return _mapping(current, field=field)


def _canonical_file_payload(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[hash_field] = canonical_json_hash(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], *, hash_field: str, kind: str) -> str:
    claimed = _sha256(payload.get(hash_field), field=f"{kind}.{hash_field}")
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    if canonical_json_hash(unsigned) != claimed:
        raise Bio24EvaluationError(f"{kind} canonical self-hash does not verify")
    return claimed


def _expected_bio_maps() -> tuple[dict[str, int], dict[int, str]]:
    label2id, id2label = build_bio_label_maps(load_taxonomy())
    if len(label2id) != 49 or len(id2label) != 49:
        raise Bio24EvaluationError("packaged core taxonomy is not the expected 24-label BIO view")
    return label2id, id2label


def _normalize_id2label(value: object) -> dict[int, str]:
    raw = _mapping(value, field="config.id2label")
    result: dict[int, str] = {}
    for key, label in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise Bio24EvaluationError("config.id2label has a non-integer key") from exc
        if isinstance(index, bool) or not isinstance(label, str) or not label:
            raise Bio24EvaluationError("config.id2label has an invalid entry")
        result[index] = label
    return result


def _normalize_label2id(value: object) -> dict[str, int]:
    raw = _mapping(value, field="config.label2id")
    result: dict[str, int] = {}
    for label, index in raw.items():
        if (
            not isinstance(label, str)
            or not label
            or isinstance(index, bool)
            or not isinstance(index, int)
        ):
            raise Bio24EvaluationError("config.label2id has an invalid entry")
        result[label] = index
    return result


def _verify_bio24_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_label2id, expected_id2label = _expected_bio_maps()
    if _normalize_id2label(config.get("id2label")) != expected_id2label:
        raise Bio24EvaluationError("model is not the exact project 24-core-label BIO head")
    if _normalize_label2id(config.get("label2id")) != expected_label2id:
        raise Bio24EvaluationError("model label2id does not match the exact BIO head")
    attention_mode = config.get("pii_attention_mode")
    model_type = config.get("model_type")
    supported = (
        (attention_mode == "full" and model_type == "qwen3_bi")
        or (attention_mode in {"causal", "jpt"} and model_type == "qwen3")
        or (attention_mode == "checkpoint_native_bidirectional" and model_type == "bert")
    )
    if not supported:
        raise Bio24EvaluationError("model type and attention mode are unsupported or inconsistent")
    return {
        "tag_scheme": "BIO",
        "core_entity_count": 24,
        "classifier_label_count": 49,
        "label_schema_sha256": canonical_json_hash(expected_label2id),
        "attention_mode": attention_mode,
        "model_type": model_type,
    }


def _verify_benchmark_isolation(manifest: Mapping[str, Any]) -> None:
    isolation = _mapping(manifest.get("benchmark_isolation"), field="manifest.benchmark_isolation")
    expected = {
        "dataset_id": DATASET_ID,
        "read_for_training": False,
        "read_for_validation": False,
        "read_for_checkpoint_selection": False,
        "read_for_threshold_tuning": False,
    }
    if any(isolation.get(key) != value for key, value in expected.items()):
        raise Bio24EvaluationError(
            "training manifest does not prove PII Bench was excluded from training and selection"
        )


def _verified_bio24_model(model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = model_path.expanduser()
    if raw.is_symlink():
        raise Bio24EvaluationError("model root must not be a symlink")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise Bio24EvaluationError("model must be a local directory")
    for item in root.iterdir():
        if item.is_symlink():
            raise Bio24EvaluationError("model artifact must not contain symlinks")
        if item.is_file() and item.suffix.lower() in _FORBIDDEN_WEIGHT_SUFFIXES:
            raise Bio24EvaluationError("model artifact contains a forbidden serialized weight")
        if item.name.startswith("adapter_"):
            raise Bio24EvaluationError("adapter-only artifacts are not accepted")

    manifest_path = root / "training_manifest.json"
    manifest = _load_object(manifest_path, kind="training manifest")
    manifest_sha256 = _verify_self_hash(
        manifest, hash_field="manifest_sha256", kind="training manifest"
    )
    if manifest.get("status") != "completed":
        raise Bio24EvaluationError("training manifest is not completed")
    _verify_benchmark_isolation(manifest)

    recorded = _mapping(manifest.get("output_artifact"), field="manifest.output_artifact")
    if recorded.get("schema_version") != 1 or recorded.get("format") != "huggingface_safetensors":
        raise Bio24EvaluationError("training manifest output format is unsupported")
    files = _mapping(recorded.get("files"), field="manifest.output_artifact.files")
    file_hashes: dict[str, str] = {}
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name
            or name == "training_manifest.json"
        ):
            raise Bio24EvaluationError("output artifact contains an unsafe file name")
        file_hashes[name] = _sha256(digest, field=f"output_artifact.files.{name}")
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    if not required <= set(file_hashes):
        raise Bio24EvaluationError("model artifact lacks bound config, safetensors, or tokenizer")
    for name, digest in file_hashes.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise Bio24EvaluationError("one or more bound model artifact hashes do not verify")
    weight_files = recorded.get("weight_files")
    if weight_files != ["model.safetensors"]:
        raise Bio24EvaluationError("evaluation requires one bound model.safetensors file")
    weight_hashes = {"model.safetensors": file_hashes["model.safetensors"]}
    if recorded.get("weights_combined_sha256") != canonical_json_hash(weight_hashes):
        raise Bio24EvaluationError("combined model weight identity does not verify")
    if recorded.get("artifact_files_combined_sha256") != canonical_json_hash(file_hashes):
        raise Bio24EvaluationError("combined model artifact identity does not verify")

    # Reject unbound files which the local Transformers/tokenizer loader can consume.
    for name in _BOUND_METADATA_FILES:
        path = root / name
        if path.is_file() and name not in file_hashes:
            raise Bio24EvaluationError("a model-consumed metadata file is not manifest-bound")
    config = _load_object(root / "config.json", kind="model config")
    schema = _verify_bio24_config(config)
    return manifest, {
        **schema,
        "training_manifest_sha256": manifest_sha256,
        "training_manifest_file_sha256": sha256_file(manifest_path),
        "artifact_files_combined_sha256": str(recorded["artifact_files_combined_sha256"]),
        "weights_combined_sha256": str(recorded["weights_combined_sha256"]),
        "all_model_artifacts_verified": True,
        "tokenizer_files_manifest_bound": True,
        "local_files_only_required": True,
        "safetensors_only": True,
    }


def _verified_dataset_manifest(
    manifest_path: Path,
    suite_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw_manifest = manifest_path.expanduser()
    if raw_manifest.is_symlink():
        raise Bio24EvaluationError("dataset manifest must not be a symlink")
    resolved = raw_manifest.resolve(strict=True)
    manifest = _load_object(resolved, kind="dataset manifest")
    logical_hash = _verify_self_hash(
        manifest, hash_field="manifest_sha256", kind="dataset manifest"
    )
    if logical_hash != DATASET_MANIFEST_SHA256:
        raise Bio24EvaluationError("dataset is not the frozen PII Bench ZH manifest")
    if manifest.get("manifest_type") != "evaluation_dataset":
        raise Bio24EvaluationError("dataset manifest is not evaluation-only")
    isolation = _mapping(manifest.get("isolation"), field="dataset.isolation")
    if (
        isolation.get("training_allowed") is not False
        or isolation.get("threshold_selection_allowed") is not False
    ):
        raise Bio24EvaluationError("dataset isolation permits training or threshold selection")
    upstream = _mapping(manifest.get("upstream"), field="dataset.upstream")
    if upstream.get("source") != DATASET_ID or upstream.get("revision") != DATASET_REVISION:
        raise Bio24EvaluationError("dataset upstream identity changed")
    subsets = _mapping(manifest.get("subsets"), field="dataset.subsets")
    verified: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        subset = _mapping(subsets.get(suite), field=f"dataset.subsets.{suite}")
        canonical = _mapping(subset.get("canonical"), field=f"dataset.subsets.{suite}.canonical")
        actual = {
            "canonical_sha256": canonical.get("sha256"),
            "record_count": canonical.get("record_count"),
            "gold_span_count": subset.get("span_count"),
            "label_counts": subset.get("label_counts"),
        }
        expected = FIXED_SUBSETS[suite]
        if any(actual.get(field) != value for field, value in expected.items()):
            raise Bio24EvaluationError(f"{suite} frozen subset identity changed")
        if suite_paths is not None:
            raw_path = suite_paths[suite].expanduser()
            if raw_path.is_symlink():
                raise Bio24EvaluationError(f"{suite} gold file must not be a symlink")
            path = raw_path.resolve(strict=True)
            if sha256_file(path) != expected["canonical_sha256"]:
                raise Bio24EvaluationError(f"{suite} gold file hash does not verify")
        verified[suite] = actual
    return manifest, verified


def _prediction_hash_from_report(
    report: Mapping[str, Any], *, suite: str, report_track: str
) -> str:
    benchmark = _mapping(report.get("benchmark"), field="comparator report benchmark")
    subsets = _mapping(benchmark.get("subsets"), field="comparator report subsets")
    subset = _mapping(subsets.get(suite), field=f"comparator report {suite}")
    value = subset.get("prediction_sha256")
    if isinstance(value, Mapping):
        value = value.get(report_track)
    return _sha256(value, field=f"comparator.{suite}.prediction_sha256")


def _strict_scores(results: Mapping[str, Any], *, comparator_id: str) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for suite in COMPARISON_SUITES:
        suite_result = _mapping(results.get(suite), field=f"{comparator_id}.{suite}")
        micro = _mapping(suite_result.get("strict_micro"), field=f"{suite}.strict_micro")
        macro = _mapping(suite_result.get("strict_macro"), field=f"{suite}.strict_macro")
        compact[suite] = {
            "strict_micro_f1": rounded_score(
                _score(micro.get("f1"), field=f"{comparator_id}.{suite}.micro")
            ),
            "strict_macro_f1": rounded_score(
                _score(macro.get("f1"), field=f"{comparator_id}.{suite}.macro")
            ),
        }
    return compact


def _verified_active_comparators(
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw_registry = registry_path.expanduser()
    if raw_registry.is_symlink():
        raise Bio24EvaluationError("comparator registry must not be a symlink")
    resolved = raw_registry.resolve(strict=True)
    registry = load_comparator_registry(resolved)
    receipt = validate_comparator_registry(registry, repository_root=_REPOSITORY_ROOT)
    raw_comparators = _mapping(registry.get("comparators"), field="registry.comparators")
    active: dict[str, dict[str, Any]] = {}
    for comparator_id, raw in sorted(raw_comparators.items()):
        comparator = _mapping(raw, field=f"registry.comparators.{comparator_id}")
        if comparator.get("canonical_track") != "model_raw":
            continue
        safe_id = _safe_id(comparator_id, field="comparator_id")
        if comparator.get("candidate") is not False:
            raise Bio24EvaluationError("active comparator set contains a project candidate")
        report_relative = comparator.get("report")
        if not isinstance(report_relative, str):
            raise Bio24EvaluationError("active comparator report path is invalid")
        report_path = (_REPOSITORY_ROOT / report_relative).resolve(strict=True)
        expected_report_hash = _sha256(
            comparator.get("report_file_sha256"), field=f"{safe_id}.report_file_sha256"
        )
        if sha256_file(report_path) != expected_report_hash:
            raise Bio24EvaluationError("active comparator report bytes changed")
        report = _load_object(report_path, kind=f"{safe_id} report")
        report_track = str(comparator.get("report_track"))
        selector = str(comparator.get("result_selector"))
        results = _nested(report, selector, field=f"{safe_id}.result_selector")
        active[safe_id] = {
            "report_file_sha256": expected_report_hash,
            "report_track": report_track,
            "result_selector": selector,
            "scores": _strict_scores(results, comparator_id=safe_id),
            "prediction_sha256": {
                suite: _prediction_hash_from_report(report, suite=suite, report_track=report_track)
                for suite in SUITES
            },
        }
    if not active:
        raise Bio24EvaluationError("comparator registry has no active model_raw comparators")
    return registry, {
        "registry_id": str(registry.get("registry_id")),
        "registry_sha256": receipt.registry_sha256,
        "registry_file_sha256": sha256_file(resolved),
        "active": active,
    }


def _parse_prediction_arguments(
    values: Sequence[Sequence[str]], *, expected_ids: Sequence[str]
) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for raw in values:
        if len(raw) != 3:
            raise Bio24EvaluationError("comparator prediction requires ID SUITE PATH")
        comparator_id = _safe_id(raw[0], field="comparator prediction ID")
        suite = raw[1].lower()
        if suite not in SUITES:
            raise Bio24EvaluationError("comparator prediction suite must be formal or chat")
        if suite in result.setdefault(comparator_id, {}):
            raise Bio24EvaluationError("duplicate comparator prediction argument")
        path = Path(raw[2]).expanduser()
        if path.is_symlink():
            raise Bio24EvaluationError("comparator prediction must not be a symlink")
        result[comparator_id][suite] = path.resolve(strict=True)
    expected = set(expected_ids)
    if set(result) != expected or any(set(result[item]) != set(SUITES) for item in expected):
        raise Bio24EvaluationError(
            "provide exactly formal and chat predictions for every active comparator"
        )
    return result


def _verify_comparator_prediction_files(
    paths: Mapping[str, Mapping[str, Path]],
    active: Mapping[str, Mapping[str, Any]],
) -> None:
    for comparator_id, by_suite in paths.items():
        expected = _mapping(active[comparator_id]["prediction_sha256"], field="prediction hashes")
        for suite in SUITES:
            if sha256_file(by_suite[suite]) != expected[suite]:
                raise Bio24EvaluationError(
                    f"{comparator_id} {suite} prediction hash does not verify"
                )


def _validate_freeze_parameters(args: argparse.Namespace) -> None:
    if not isinstance(args.max_tokens, int) or isinstance(args.max_tokens, bool):
        raise Bio24EvaluationError("max tokens must be an integer")
    if args.max_tokens < 32:
        raise Bio24EvaluationError("max tokens is implausibly small")
    if not 0.0 < float(args.stride_fraction) < 1.0:
        raise Bio24EvaluationError("stride fraction must be in (0, 1)")
    if args.bootstrap_samples < 1_000:
        raise Bio24EvaluationError("paired bootstrap requires at least 1000 samples")
    if float(args.bootstrap_confidence) != 0.95:
        raise Bio24EvaluationError("the frozen comparison requires 95% intervals")
    if isinstance(args.bootstrap_seed, bool) or not isinstance(args.bootstrap_seed, int):
        raise Bio24EvaluationError("bootstrap seed must be an integer")


def _build_freeze_plan(args: argparse.Namespace) -> dict[str, Any]:
    _validate_freeze_parameters(args)
    candidate_id = _safe_id(args.candidate_id, field="candidate_id")
    _manifest, model = _verified_bio24_model(args.model)
    dataset, subsets = _verified_dataset_manifest(args.dataset_manifest)
    _registry, comparator_set = _verified_active_comparators(args.comparator_registry)
    active = _mapping(comparator_set["active"], field="active comparators")
    paths = _parse_prediction_arguments(args.comparator_prediction, expected_ids=tuple(active))
    _verify_comparator_prediction_files(paths, active)
    upstream = _mapping(dataset.get("upstream"), field="dataset.upstream")
    plan: dict[str, Any] = {
        "schema_version": 1,
        "plan_type": PLAN_TYPE,
        "candidate_id": candidate_id,
        "status": {
            "frozen_before_candidate_benchmark_inference": True,
            "public_test_exposed": True,
            "posthoc_lineage": True,
            "selection_allowed": False,
            "checkpoint_selection_allowed": False,
            "threshold_tuning_allowed": False,
            "recipe_selection_allowed": False,
            "confirmatory_claim_allowed": False,
        },
        "candidate": model,
        "dataset": {
            "dataset_id": str(upstream.get("source")),
            "dataset_revision": str(upstream.get("revision")),
            "dataset_manifest_sha256": str(dataset["manifest_sha256"]),
            "dataset_manifest_file_sha256": sha256_file(
                args.dataset_manifest.expanduser().resolve(strict=True)
            ),
            "subsets": subsets,
        },
        "comparison_contract": {
            "track": "model_raw",
            "metric": "strict_exact_start_end_label",
            "protocol_id": PROTOCOL_ID,
            "candidate_head": "project_core_24_bio",
            "evaluated_labels": list(CLOSED_8_LABELS),
            "non_protocol_gold_policy": "not_present_in_closed8_gold",
            "closed8_logit_mask": True,
            "mask_timing": "before_softmax_argmax_and_standard_bio_decoding",
            "score_threshold": 0.0,
            "threshold_tuning": False,
            "checkpoint_selection": False,
            "deduplication_policy": "high_overlap_v1",
        },
        "inference": {
            "max_tokens": args.max_tokens,
            "stride_fraction": float(args.stride_fraction),
            "local_files_only": True,
            "trust_remote_code": False,
            "safetensors_only": True,
        },
        "bootstrap": {
            "unit": "document",
            "suite_resampling": "stratified_preserve_formal_chat_sizes",
            "samples": args.bootstrap_samples,
            "confidence": float(args.bootstrap_confidence),
            "base_seed": args.bootstrap_seed,
            "analysis": "posthoc_descriptive",
        },
        "comparator_registry": comparator_set,
        "implementation": {
            "evaluator_sha256": sha256_file(Path(__file__)),
            "transformers_predictor_sha256": sha256_file(
                _REPOSITORY_ROOT / "src/pii_zh/inference/transformers_predictor.py"
            ),
            "bio_decoder_sha256": sha256_file(_REPOSITORY_ROOT / "src/pii_zh/models/decoding.py"),
            "paired_bootstrap_sha256": sha256_file(
                _REPOSITORY_ROOT / "src/pii_zh/evaluation/paired_bootstrap.py"
            ),
        },
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_local_paths": False,
            "record_level_predictions_public": False,
        },
    }
    return _canonical_file_payload(plan, "freeze_plan_sha256")


def _verify_freeze_plan(
    path: Path,
    *,
    candidate_id: str,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    comparator_set: Mapping[str, Any],
) -> dict[str, Any]:
    raw_plan = path.expanduser()
    if raw_plan.is_symlink():
        raise Bio24EvaluationError("freeze plan must not be a symlink")
    plan_path = raw_plan.resolve(strict=True)
    plan = _load_object(plan_path, kind="freeze plan")
    _verify_self_hash(plan, hash_field="freeze_plan_sha256", kind="freeze plan")
    expected_root_fields = {
        "schema_version",
        "plan_type",
        "candidate_id",
        "status",
        "candidate",
        "dataset",
        "comparison_contract",
        "inference",
        "bootstrap",
        "comparator_registry",
        "implementation",
        "privacy",
        "freeze_plan_sha256",
    }
    if set(plan) != expected_root_fields:
        raise Bio24EvaluationError("freeze plan root schema changed")
    if (
        plan.get("schema_version") != 1
        or plan.get("plan_type") != PLAN_TYPE
        or plan.get("candidate_id") != candidate_id
    ):
        raise Bio24EvaluationError("freeze plan identity does not match the candidate")
    status = _mapping(plan.get("status"), field="freeze plan status")
    required_status = {
        "frozen_before_candidate_benchmark_inference": True,
        "public_test_exposed": True,
        "posthoc_lineage": True,
        "selection_allowed": False,
        "checkpoint_selection_allowed": False,
        "threshold_tuning_allowed": False,
        "recipe_selection_allowed": False,
        "confirmatory_claim_allowed": False,
    }
    if dict(status) != required_status:
        raise Bio24EvaluationError("freeze plan exposure or no-selection contract changed")
    if plan.get("candidate") != dict(model):
        raise Bio24EvaluationError("freeze plan candidate artifact identity changed")
    frozen_dataset = _mapping(plan.get("dataset"), field="freeze plan dataset")
    expected_dataset = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_manifest_file_sha256": dataset["dataset_manifest_file_sha256"],
        "subsets": dataset["subsets"],
    }
    if dict(frozen_dataset) != expected_dataset:
        raise Bio24EvaluationError("freeze plan dataset identity changed")
    if plan.get("comparator_registry") != dict(comparator_set):
        raise Bio24EvaluationError("freeze plan active comparator set changed")
    contract = _mapping(plan.get("comparison_contract"), field="comparison contract")
    threshold = _score(contract.get("score_threshold"), field="score_threshold")
    if (
        contract.get("track") != "model_raw"
        or contract.get("metric") != "strict_exact_start_end_label"
        or contract.get("protocol_id") != PROTOCOL_ID
        or tuple(contract.get("evaluated_labels", ())) != CLOSED_8_LABELS
        or contract.get("closed8_logit_mask") is not True
        or contract.get("mask_timing") != "before_softmax_argmax_and_standard_bio_decoding"
        or threshold != 0.0
        or contract.get("threshold_tuning") is not False
        or contract.get("checkpoint_selection") is not False
    ):
        raise Bio24EvaluationError("freeze plan comparison protocol changed")
    implementation = _mapping(plan.get("implementation"), field="implementation")
    expected_implementations = {
        "evaluator_sha256": sha256_file(Path(__file__)),
        "transformers_predictor_sha256": sha256_file(
            _REPOSITORY_ROOT / "src/pii_zh/inference/transformers_predictor.py"
        ),
        "bio_decoder_sha256": sha256_file(_REPOSITORY_ROOT / "src/pii_zh/models/decoding.py"),
        "paired_bootstrap_sha256": sha256_file(
            _REPOSITORY_ROOT / "src/pii_zh/evaluation/paired_bootstrap.py"
        ),
    }
    if dict(implementation) != expected_implementations:
        raise Bio24EvaluationError("freeze plan implementation hashes changed")

    inference = _mapping(plan.get("inference"), field="freeze plan inference")
    max_tokens = inference.get("max_tokens")
    stride_fraction = inference.get("stride_fraction")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 32
        or isinstance(stride_fraction, bool)
        or not isinstance(stride_fraction, (int, float))
        or not 0.0 < float(stride_fraction) < 1.0
        or inference.get("local_files_only") is not True
        or inference.get("trust_remote_code") is not False
        or inference.get("safetensors_only") is not True
    ):
        raise Bio24EvaluationError("freeze plan inference contract changed")
    bootstrap = _mapping(plan.get("bootstrap"), field="freeze plan bootstrap")
    samples = bootstrap.get("samples")
    seed = bootstrap.get("base_seed")
    if (
        bootstrap.get("unit") != "document"
        or bootstrap.get("suite_resampling") != "stratified_preserve_formal_chat_sizes"
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 1_000
        or bootstrap.get("confidence") != 0.95
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or bootstrap.get("analysis") != "posthoc_descriptive"
    ):
        raise Bio24EvaluationError("freeze plan paired bootstrap contract changed")
    privacy = _mapping(plan.get("privacy"), field="freeze plan privacy")
    if privacy != {
        "contains_raw_text": False,
        "contains_entity_values": False,
        "contains_local_paths": False,
        "record_level_predictions_public": False,
    }:
        raise Bio24EvaluationError("freeze plan privacy contract changed")
    _validate_public_payload(plan)
    return plan


def _private_fresh_run_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise Bio24EvaluationError(
                "run directory must be new and empty; benchmark reruns cannot select a candidate"
            )
    else:
        resolved.mkdir(parents=True)
    resolved.chmod(0o700)
    return resolved


def _public_bootstrap(value: Mapping[str, Any]) -> dict[str, Any]:
    interval = _mapping(value.get("confidence_interval"), field="bootstrap interval")
    outcomes = _mapping(value.get("bootstrap_outcomes"), field="bootstrap outcomes")
    return {
        "documents": int(value["documents"]),
        "suite_document_counts": dict(value["suite_document_counts"]),
        "candidate_strict_micro_f1": rounded_score(value["candidate_strict_micro_f1"]),
        "baseline_strict_micro_f1": rounded_score(value["baseline_strict_micro_f1"]),
        "delta_candidate_minus_baseline": round(float(value["delta_candidate_minus_baseline"]), 8),
        "delta_percentile_95_ci": {
            "lower": round(float(interval["lower"]), 8),
            "upper": round(float(interval["upper"]), 8),
        },
        "bootstrap_outcomes": {
            "candidate_wins": int(outcomes["candidate_wins"]),
            "ties": int(outcomes["ties"]),
            "baseline_wins": int(outcomes["baseline_wins"]),
        },
        "candidate_win_probability": round(float(value["candidate_win_probability"]), 8),
        "two_sided_tail_probability_approximation": round(
            float(value["two_sided_tail_probability_approximation"]), 8
        ),
        "effective_samples": int(value["effective_samples"]),
        "seed": int(value["seed"]),
    }


def _paired_comparisons(
    *,
    gold: Mapping[str, Sequence[Any]],
    candidate: Mapping[str, Sequence[Any]],
    comparator_predictions: Mapping[str, Mapping[str, Sequence[Any]]],
    samples: int,
    confidence: float,
    base_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for comparator_offset, comparator_id in enumerate(sorted(comparator_predictions)):
        suites = {
            suite: build_paired_suite(
                name=suite,
                gold_records=gold[suite],
                candidate_records=candidate[suite],
                baseline_records=comparator_predictions[comparator_id][suite],
                labels=CLOSED_8_LABELS,
            )
            for suite in SUITES
        }
        by_suite: dict[str, Any] = {}
        for suite_offset, suite in enumerate(COMPARISON_SUITES):
            selected = [suites[suite]] if suite in SUITES else [suites[item] for item in SUITES]
            seed = base_seed + comparator_offset * 100 + suite_offset
            by_suite[suite] = _public_bootstrap(
                paired_bootstrap_delta(
                    selected,
                    samples=samples,
                    confidence=confidence,
                    seed=seed,
                )
            )
        result[comparator_id] = by_suite
    return result


def _active_comparator_envelope(
    candidate_results: Mapping[str, Mapping[str, Any]],
    active: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    point_metrics: dict[str, Any] = {}
    all_point_passes: list[bool] = []
    for suite in COMPARISON_SUITES:
        suite_metrics: dict[str, Any] = {}
        for metric, result_key in (
            ("strict_micro_f1", "strict_micro"),
            ("strict_macro_f1", "strict_macro"),
        ):
            scores = {
                comparator_id: _score(
                    _mapping(info.get("scores"), field="comparator scores")[suite][metric],
                    field=f"{comparator_id}.{suite}.{metric}",
                )
                for comparator_id, info in active.items()
            }
            maximum = max(scores.values())
            owners = sorted(
                comparator_id for comparator_id, score in scores.items() if score == maximum
            )
            candidate_score = _score(
                candidate_results[suite][result_key]["f1"],
                field=f"candidate.{suite}.{metric}",
            )
            passed = candidate_score >= maximum
            all_point_passes.append(passed)
            suite_metrics[metric] = {
                "candidate": rounded_score(candidate_score),
                "active_envelope": rounded_score(maximum),
                "envelope_owner_ids": owners,
                "delta": round(candidate_score - maximum, 8),
                "meets_or_exceeds": passed,
            }
        point_metrics[suite] = suite_metrics

    paired_checks: dict[str, Any] = {}
    all_lower_bounds: list[bool] = []
    for comparator_id, by_suite in paired.items():
        paired_checks[comparator_id] = {}
        for suite in COMPARISON_SUITES:
            lower = float(by_suite[suite]["delta_percentile_95_ci"]["lower"])
            passed = lower >= 0.0
            all_lower_bounds.append(passed)
            paired_checks[comparator_id][suite] = {
                "delta_percentile_95_ci_lower_nonnegative": passed
            }
    point_passed = all(all_point_passes)
    bootstrap_passed = all(all_lower_bounds)
    return {
        "decision_scope": "posthoc_descriptive_active_model_raw_comparators",
        "confirmatory_claim_allowed": False,
        "point_estimate_envelope": point_metrics,
        "meets_all_formal_chat_pooled_micro_macro_point_estimates": point_passed,
        "paired_micro_bootstrap_checks": paired_checks,
        "all_paired_micro_95pct_lower_bounds_nonnegative": bootstrap_passed,
        "descriptive_active_envelope_passed": point_passed and bootstrap_passed,
    }


def _validate_public_payload(payload: Mapping[str, Any]) -> None:
    status = _mapping(payload.get("status"), field="public status")
    if (
        status.get("public_test_exposed") is not True
        or status.get("posthoc_lineage") is not True
        or status.get("selection_allowed") is not False
        or status.get("confirmatory_claim_allowed") is not False
    ):
        raise Bio24EvaluationError("public report lacks mandatory post-hoc warnings")
    privacy = _mapping(payload.get("privacy"), field="public privacy")
    if (
        privacy.get("contains_raw_text") is not False
        or privacy.get("contains_entity_values") is not False
        or privacy.get("contains_local_paths") is not False
        or privacy.get("record_level_predictions_public") is not False
    ):
        raise Bio24EvaluationError("public report privacy contract is incomplete")

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if not isinstance(key, str) or key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                    raise Bio24EvaluationError("public report contains a forbidden record field")
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and (
            value.startswith(("/", "~/", "file://"))
            or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        ):
            raise Bio24EvaluationError("public report contains a local path")

    visit(payload)


def _freeze(args: argparse.Namespace) -> int:
    plan = _build_freeze_plan(args)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise Bio24EvaluationError("freeze output already exists; immutable plans are not replaced")
    baseline_helpers._write_json_atomic(plan, output, mode=0o644)
    print(
        json.dumps(
            {
                "candidate_id": plan["candidate_id"],
                "freeze_plan_file_sha256": sha256_file(output),
                "freeze_plan_sha256": plan["freeze_plan_sha256"],
                "public_test_exposed": True,
                "posthoc_lineage": True,
                "selection_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if args.document_batch_size < 1 or args.window_batch_size < 1:
        raise Bio24EvaluationError("batch sizes must be positive")
    if args.log_every_documents < 1:
        raise Bio24EvaluationError("log interval must be positive")
    baseline_helpers._require_gpu_isolation(args.device)

    manifest, model = _verified_bio24_model(args.model)
    candidate_id = _safe_id(
        _load_object(args.freeze_plan.expanduser().resolve(strict=True), kind="freeze plan").get(
            "candidate_id"
        ),
        field="candidate_id",
    )
    suite_paths = {
        suite: getattr(args, suite).expanduser().resolve(strict=True) for suite in SUITES
    }
    dataset_manifest, subsets = _verified_dataset_manifest(args.dataset_manifest, suite_paths)
    _registry, comparator_set = _verified_active_comparators(args.comparator_registry)
    active = _mapping(comparator_set["active"], field="active comparators")
    comparator_paths = _parse_prediction_arguments(
        args.comparator_prediction, expected_ids=tuple(active)
    )
    _verify_comparator_prediction_files(comparator_paths, active)
    dataset_identity = {
        "dataset_manifest_file_sha256": sha256_file(
            args.dataset_manifest.expanduser().resolve(strict=True)
        ),
        "subsets": subsets,
    }
    plan = _verify_freeze_plan(
        args.freeze_plan,
        candidate_id=candidate_id,
        model=model,
        dataset=dataset_identity,
        comparator_set=comparator_set,
    )
    inference = _mapping(plan.get("inference"), field="freeze inference")
    bootstrap = _mapping(plan.get("bootstrap"), field="freeze bootstrap")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    predictor = load_local_predictor(
        args.model,
        device=args.device,
        dtype=dtype,
        micro_batch_size=args.window_batch_size,
        ignored_labels=(),
        allowed_labels=CLOSED_8_LABELS,
    )
    if predictor.allowed_labels != frozenset(CLOSED_8_LABELS):
        raise Bio24EvaluationError("predictor did not activate the frozen pre-decode label mask")
    recognizer = QwenPiiRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        label_mapping={label: label for label in CLOSED_8_LABELS},
        ignored_labels=(),
        default_threshold=0.0,
        max_tokens=int(inference["max_tokens"]),
        stride_fraction=float(inference["stride_fraction"]),
        model_version=candidate_id,
        attention_mode=predictor.attention_mode,
        deduplication_policy="high_overlap_v1",
        name="Bio24FrozenClosed8ModelRaw",
    )

    run_dir = _private_fresh_run_dir(args.run_dir)
    compact_results: dict[str, dict[str, Any]] = {}
    prediction_hashes: dict[str, str] = {}
    gold_records: dict[str, Sequence[Any]] = {}
    candidate_records: dict[str, Sequence[Any]] = {}
    for suite in SUITES:
        prediction_path = run_dir / f"{suite}.predictions.jsonl"
        baseline_helpers._write_predictions(
            input_path=suite_paths[suite],
            output_path=prediction_path,
            recognizer=recognizer,
            document_batch_size=args.document_batch_size,
            expected_documents=int(subsets[suite]["record_count"]),
            log_every_documents=args.log_every_documents,
            suite=suite,
        )
        gold_records[suite] = load_gold_jsonl(suite_paths[suite])
        candidate_records[suite] = load_prediction_jsonl(prediction_path)
        evaluation = evaluate_documents(
            gold_records[suite], candidate_records[suite], bootstrap_samples=0
        )
        compact_results[suite] = compact_strict_result(evaluation, expected_labels=CLOSED_8_LABELS)
        prediction_hashes[suite] = sha256_file(prediction_path)
    compact_results["pooled"] = pool_strict_results(
        {suite: compact_results[suite] for suite in SUITES},
        expected_labels=CLOSED_8_LABELS,
    )

    comparator_predictions = {
        comparator_id: {
            suite: load_prediction_jsonl(comparator_paths[comparator_id][suite]) for suite in SUITES
        }
        for comparator_id in active
    }
    paired = _paired_comparisons(
        gold=gold_records,
        candidate=candidate_records,
        comparator_predictions=comparator_predictions,
        samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        base_seed=int(bootstrap["base_seed"]),
    )
    envelope = _active_comparator_envelope(compact_results, active, paired)
    upstream = _mapping(dataset_manifest.get("upstream"), field="dataset upstream")
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "candidate_id": candidate_id,
        "status": {
            "evidence_classification": "posthoc_descriptive_public_benchmark",
            "public_test_exposed": True,
            "posthoc_lineage": True,
            "selection_allowed": False,
            "checkpoint_selection_allowed": False,
            "threshold_tuning_allowed": False,
            "recipe_selection_allowed": False,
            "confirmatory_claim_allowed": False,
            "release_selection_eligible": False,
            "reason": (
                "PII Bench ZH is public and the project lineage has already observed it; "
                "this report is descriptive evidence only"
            ),
        },
        "comparison_contract": dict(plan["comparison_contract"]),
        "candidate": {
            **model,
            "benchmark_isolation_verified": True,
            "training_manifest_release_eligible": manifest.get("release_eligible"),
            "inference": {
                **dict(inference),
                "attention_mode": predictor.attention_mode,
                "closed8_mask_applied_before_decode": True,
            },
        },
        "benchmark": {
            "dataset_id": str(upstream.get("source")),
            "dataset_revision": str(upstream.get("revision")),
            "dataset_manifest_sha256": str(dataset_manifest["manifest_sha256"]),
            "fixed_gold_verified": True,
            "subsets": {
                suite: {
                    **subsets[suite],
                    "candidate_prediction_sha256": prediction_hashes[suite],
                }
                for suite in SUITES
            },
        },
        "results": compact_results,
        "paired_document_bootstrap": {
            "methodology": dict(bootstrap),
            "comparators": paired,
        },
        "active_comparator_envelope": envelope,
        "evidence_binding": {
            "freeze_plan_sha256": plan["freeze_plan_sha256"],
            "freeze_plan_file_sha256": sha256_file(
                args.freeze_plan.expanduser().resolve(strict=True)
            ),
            "comparator_registry_sha256": comparator_set["registry_sha256"],
            "comparator_registry_file_sha256": comparator_set["registry_file_sha256"],
        },
        "implementation": dict(plan["implementation"]),
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_local_paths": False,
            "record_level_predictions_public": False,
            "record_level_predictions_mode": "0600",
        },
    }
    _validate_public_payload(report)
    report = _canonical_file_payload(report, "report_sha256")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise Bio24EvaluationError("evaluation output already exists and will not be replaced")
    baseline_helpers._write_json_atomic(report, output, mode=0o644)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "candidate_id": candidate_id,
        "report_sha256": report["report_sha256"],
        "report_file_sha256": sha256_file(output),
        "freeze_plan_sha256": plan["freeze_plan_sha256"],
        "freeze_plan_file_sha256": sha256_file(args.freeze_plan.expanduser().resolve(strict=True)),
        "training_manifest_sha256": model["training_manifest_sha256"],
        "artifact_files_combined_sha256": model["artifact_files_combined_sha256"],
        "candidate_prediction_sha256": prediction_hashes,
        "comparator_prediction_sha256": {
            comparator_id: dict(active[comparator_id]["prediction_sha256"])
            for comparator_id in sorted(active)
        },
        "public_test_exposed": True,
        "posthoc_lineage": True,
        "selection_allowed": False,
    }
    receipt = _canonical_file_payload(receipt, "receipt_sha256")
    baseline_helpers._write_json_atomic(receipt, run_dir / "run_receipt.json", mode=0o600)
    print(
        json.dumps(
            {
                "report_file_sha256": sha256_file(output),
                "report_sha256": report["report_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "pooled_strict_micro_f1": compact_results["pooled"]["strict_micro"]["f1"],
                "descriptive_active_envelope_passed": envelope[
                    "descriptive_active_envelope_passed"
                ],
                "public_test_exposed": True,
                "posthoc_lineage": True,
                "selection_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        return _freeze(args)
    if args.command == "evaluate":
        return _evaluate(args)
    raise Bio24EvaluationError("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
