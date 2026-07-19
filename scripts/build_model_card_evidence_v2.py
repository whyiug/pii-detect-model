#!/usr/bin/env python3
"""Build a closed, model-only Model Card evidence bundle.

This successor is deliberately opt-in: a release is on the v2 path only when
``model_card_evidence_v2.json`` is present in the package.  The builder consumes
one explicit release contract and exactly one release-eligible selection
receipt, then projects only frozen ``model_raw``/``model_calibrated`` evidence.
It never publishes or uploads a model.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "configs/release/model_card_evidence_v2.schema.json"
EVIDENCE_FILENAME = "model_card_evidence_v2.json"
CONTRACT_FILENAME = "release_contract_v2.json"
SELECTED_RECEIPT_FILENAME = "selected_candidate_receipt_v2.json"
MODEL_INDEX_FILENAME = "model-index.yml"
README_FILENAME = "README.md"
BOUND_RELEASE_TEXT_FILES = (
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
)

_MAXIMUM_INPUT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(r"^(?:/|~(?:/|$)|[A-Za-z]:[\\/]|file:" r"//)")
_MODEL_TRACKS = ("model_raw", "model_calibrated")
_TRACK_ORDER = {track: index for index, track in enumerate(_MODEL_TRACKS)}
_METRIC_PREFIX = {
    "model_raw": "Model Raw",
    "model_calibrated": "Model Calibrated",
}
_CHECKPOINT_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)
_EXPECTED_ATTRIBUTION_FLAGS = (
    "checkpoint_output",
    "decoder_applied",
    "calibration_applied",
    "rules_applied",
    "framework_applied",
    "cascade_applied",
    "fusion_applied",
    "refinement_applied",
)
_METRIC_CONSISTENCY_TOLERANCE = 5e-4
_UNAUTHORIZED_CLAIM_LANGUAGE = re.compile(
    r"(?:\b(?:first|best|sota|state[ -]of[ -]the[ -]art|production|compliance)\b|"
    r"首个|首款|第一款|最强|最佳|领先|生产级|合规保证)",
    re.IGNORECASE,
)


class ModelCardEvidenceError(ValueError):
    """Raised when v2 release evidence cannot be safely constructed."""


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ModelCardEvidenceError(f"{field} must be an object")
    return value


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ModelCardEvidenceError(f"{description} must be a regular non-symlink file")
    if path.stat().st_size > _MAXIMUM_INPUT_BYTES:
        raise ModelCardEvidenceError(f"{description} exceeds the input size limit")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ModelCardEvidenceError(
                    f"{description} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        raise ModelCardEvidenceError(
            f"{description} contains forbidden non-finite JSON constant {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except ModelCardEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelCardEvidenceError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelCardEvidenceError(f"{description} must be a JSON object")
    return value


def _verify_self_hash(
    document: Mapping[str, Any], *, field: str, description: str
) -> str:
    claimed = document.get(field)
    unsigned = dict(document)
    unsigned.pop(field, None)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise ModelCardEvidenceError(f"{description} has no valid {field}")
    if canonical_json_hash(unsigned) != claimed:
        raise ModelCardEvidenceError(f"{description} {field} is stale or invalid")
    return claimed


def _assert_no_absolute_paths(value: object, *, field: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_absolute_paths(item, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, field=f"{field}[{index}]")
    elif isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        raise ModelCardEvidenceError(f"absolute/local path is forbidden in evidence: {field}")


def _load_schema() -> dict[str, Any]:
    return _load_json(SCHEMA_PATH, description="model-card evidence v2 schema")


def validate_evidence_schema(document: Mapping[str, Any]) -> None:
    validator = jsonschema.Draft202012Validator(_load_schema())
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ModelCardEvidenceError(f"v2 evidence schema violation at {location}: {error.message}")


def _validate_schema_definition(
    document: Mapping[str, Any], *, definition: str, description: str
) -> None:
    root = _load_schema()
    validator = jsonschema.Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": f"#/$defs/{definition}",
            "$defs": root["$defs"],
        }
    )
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ModelCardEvidenceError(
            f"{description} schema violation at {location}: {error.message}"
        )


def _artifact_fingerprint(checkpoint_dir: Path) -> dict[str, Any]:
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ModelCardEvidenceError("checkpoint directory must be a real directory")
    weights = tuple(sorted(checkpoint_dir.glob("model*.safetensors")))
    if len(weights) != 1 or weights[0].name != "model.safetensors" or weights[0].is_symlink():
        raise ModelCardEvidenceError("checkpoint must contain exactly model.safetensors")
    files = {weights[0].name: sha256_file(weights[0])}
    for name in _CHECKPOINT_METADATA_FILES:
        path = checkpoint_dir / name
        if path.is_symlink():
            raise ModelCardEvidenceError(f"checkpoint metadata must not be a symlink: {name}")
        if path.is_file():
            files[name] = sha256_file(path)
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    if not required.issubset(files):
        raise ModelCardEvidenceError(
            "checkpoint is missing release-bound files: " + ", ".join(sorted(required - files))
        )
    files = dict(sorted(files.items()))
    weight_hashes = {"model.safetensors": files["model.safetensors"]}
    return {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": files,
        "weight_files": ["model.safetensors"],
        "weights_combined_sha256": canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": canonical_json_hash(files),
    }


def _finite_probability(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ModelCardEvidenceError(f"{field} must be a finite probability")
    return float(value)


def _dataset_contracts(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = contract.get("datasets")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, dict) for item in raw):
        raise ModelCardEvidenceError("release contract datasets must be a non-empty object list")
    datasets = [item for item in raw if isinstance(item, dict)]
    suite_ids = [item.get("suite_id") for item in datasets]
    if any(not isinstance(suite_id, str) for suite_id in suite_ids):
        raise ModelCardEvidenceError("release contract dataset suite_id must be text")
    if len(set(suite_ids)) != len(suite_ids):
        raise ModelCardEvidenceError("release contract dataset suite_id values must be unique")
    return datasets


def _validate_rendered_contract_text(contract: Mapping[str, Any]) -> None:
    publication = _mapping(contract.get("publication"), field="release_contract.publication")
    datasets = _dataset_contracts(contract)
    rendered_fields = {
        "publication.name": publication.get("name"),
        "publication.model_id": publication.get("model_id"),
        **{
            f"datasets[{index}].name": dataset.get("name")
            for index, dataset in enumerate(datasets)
        },
    }
    for field, value in rendered_fields.items():
        if not isinstance(value, str):
            raise ModelCardEvidenceError(f"{field} must be text")
        if any(character in value for character in ("\r", "\n", "`", "|", "<", ">")):
            raise ModelCardEvidenceError(f"unsafe Markdown/front-matter text in {field}")
        if _UNAUTHORIZED_CLAIM_LANGUAGE.search(value):
            raise ModelCardEvidenceError(
                f"BLOCKED: unauthorized first/best/SOTA/production claim language in {field}"
            )


def validate_bound_release_text_files(
    contract: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    """Bind every non-generated public text file to the frozen v2 contract."""

    expected = _mapping(contract.get("release_files"), field="release_contract.release_files")
    required = set(BOUND_RELEASE_TEXT_FILES)
    if set(expected) != required or set(paths) != required:
        raise ModelCardEvidenceError(
            "v2 bound release text file set differs from the frozen contract"
        )
    for name in BOUND_RELEASE_TEXT_FILES:
        path = paths[name]
        if path.is_symlink() or not path.is_file():
            raise ModelCardEvidenceError(
                f"bound release text must be a regular non-symlink file: {name}"
            )
        if sha256_file(path) != expected[name]:
            raise ModelCardEvidenceError(
                f"bound release text SHA-256 differs from the frozen contract: {name}"
            )


def _model_summaries(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    plural = report.get("frozen_model_evaluations")
    singular = report.get("frozen_model_evaluation")
    if plural is not None and singular is not None:
        raise ModelCardEvidenceError(
            "evaluation report must use one frozen model evidence container"
        )
    raw: object = plural if plural is not None else ([singular] if singular is not None else None)
    if not isinstance(raw, list) or not raw or any(not isinstance(item, dict) for item in raw):
        raise ModelCardEvidenceError("evaluation report has no frozen model-track evidence")
    summaries = [item for item in raw if isinstance(item, dict)]
    tracks = [item.get("canonical_track") for item in summaries]
    if any(not isinstance(track, str) for track in tracks):
        raise ModelCardEvidenceError("frozen model evidence canonical_track must be text")
    if len(set(tracks)) != len(tracks):
        raise ModelCardEvidenceError("frozen model evidence contains a duplicate track")
    if len(summaries) != 2 or set(tracks) != set(_MODEL_TRACKS):
        raise ModelCardEvidenceError(
            "frozen evidence must contain exactly model_raw and model_calibrated"
        )
    return summaries


def _expected_attribution(track: str) -> dict[str, object]:
    return {
        "output_stage": track,
        "checkpoint_output": True,
        "decoder_applied": True,
        "calibration_applied": track == "model_calibrated",
        "rules_applied": False,
        "framework_applied": False,
        "cascade_applied": False,
        "fusion_applied": False,
        "refinement_applied": False,
    }


def _validated_strict_metrics(
    strict: Mapping[str, Any], *, field: str
) -> tuple[float, float, float]:
    precision = _finite_probability(strict.get("precision"), field=f"{field}.precision")
    recall = _finite_probability(strict.get("recall"), field=f"{field}.recall")
    f1 = _finite_probability(strict.get("f1"), field=f"{field}.f1")
    count_names = ("tp", "fp", "fn")
    present_counts = [name for name in count_names if name in strict]
    if present_counts:
        if len(present_counts) != len(count_names):
            raise ModelCardEvidenceError(f"{field} has an incomplete tp/fp/fn count triple")
        counts: dict[str, int] = {}
        for name in count_names:
            value = strict[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelCardEvidenceError(f"{field}.{name} must be a non-negative integer")
            counts[name] = value
        expected_precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"]
            else 0.0
        )
        expected_recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"]
            else 0.0
        )
    else:
        expected_precision = precision
        expected_recall = recall
    expected_f1 = (
        2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    expected = (expected_precision, expected_recall, expected_f1)
    actual = (precision, recall, f1)
    if any(
        abs(actual_value - expected_value) > _METRIC_CONSISTENCY_TOLERANCE
        for actual_value, expected_value in zip(actual, expected, strict=True)
    ):
        raise ModelCardEvidenceError(f"{field} precision/recall/F1 are not self-consistent")
    return precision, recall, f1


def extract_frozen_model_evidence(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    training_manifest_sha256: str,
) -> list[dict[str, Any]]:
    datasets = _dataset_contracts(contract)
    suite_order = [str(item["suite_id"]) for item in datasets]
    dataset_contract_by_id = {str(item["suite_id"]): item for item in datasets}
    results: list[dict[str, Any]] = []
    seen_tracks: set[str] = set()
    for index, summary in enumerate(_model_summaries(report)):
        _verify_self_hash(
            summary,
            field="manifest_sha256",
            description=f"frozen model evaluation {index}",
        )
        if summary.get("artifact_type") != "frozen_model_evaluation_summary":
            raise ModelCardEvidenceError("frozen evidence is not a model evaluation summary")
        track = summary.get("canonical_track")
        if not isinstance(track, str) or track not in _MODEL_TRACKS:
            raise ModelCardEvidenceError(
                "frozen evidence track must be model_raw or model_calibrated"
            )
        seen_tracks.add(track)
        attribution = _mapping(summary.get("attribution"), field=f"{track}.attribution")
        if set(attribution) != {"output_stage", *_EXPECTED_ATTRIBUTION_FLAGS}:
            raise ModelCardEvidenceError(f"{track} attribution is not closed")
        if dict(attribution) != _expected_attribution(track):
            raise ModelCardEvidenceError(
                f"{track} evidence includes a system/rule component or wrong model stage"
            )
        identity = _mapping(summary.get("model_identity"), field=f"{track}.model_identity")
        training = _mapping(identity.get("training"), field=f"{track}.model_identity.training")
        if training.get("manifest_sha256") != training_manifest_sha256:
            raise ModelCardEvidenceError(f"{track} is not bound to the selected training manifest")
        actual_suite_order = summary.get("suite_order")
        if actual_suite_order != suite_order:
            raise ModelCardEvidenceError(f"{track} suite order differs from the v2 contract")
        suites = _mapping(summary.get("suites"), field=f"{track}.suites")
        if set(suites) != set(suite_order):
            raise ModelCardEvidenceError(f"{track} suite set differs from the v2 contract")
        for suite_id in suite_order:
            suite = _mapping(suites[suite_id], field=f"{track}.{suite_id}")
            dataset = _mapping(suite.get("dataset"), field=f"{track}.{suite_id}.dataset")
            if dataset.get("evaluation_only") is not True:
                raise ModelCardEvidenceError(f"{track}.{suite_id} is not frozen evaluation-only")
            record_count = dataset.get("record_count")
            if (
                isinstance(record_count, bool)
                or not isinstance(record_count, int)
                or record_count < 1
            ):
                raise ModelCardEvidenceError(f"{track}.{suite_id} has invalid record_count")
            manifest_sha256 = dataset.get("manifest_sha256")
            gold_sha256 = dataset.get("gold_sha256")
            if not isinstance(manifest_sha256, str) or _SHA256.fullmatch(manifest_sha256) is None:
                raise ModelCardEvidenceError(f"{track}.{suite_id} has invalid manifest hash")
            if not isinstance(gold_sha256, str) or _SHA256.fullmatch(gold_sha256) is None:
                raise ModelCardEvidenceError(f"{track}.{suite_id} has invalid gold hash")
            metrics = _mapping(suite.get("metrics"), field=f"{track}.{suite_id}.metrics")
            if metrics.get("document_count") != record_count:
                raise ModelCardEvidenceError(f"{track}.{suite_id} metric count is not bound")
            strict_field = f"{track}.{suite_id}.strict_micro"
            strict = _mapping(
                metrics.get("strict_micro"), field=f"{track}.{suite_id}.strict_micro"
            )
            precision, recall, f1 = _validated_strict_metrics(strict, field=strict_field)
            display = dataset_contract_by_id[suite_id]
            results.append(
                {
                    "track": track,
                    "suite_id": suite_id,
                    "dataset": {
                        "name": display["name"],
                        "type": display["type"],
                        "split": display["split"],
                        "manifest_sha256": manifest_sha256,
                        "gold_sha256": gold_sha256,
                        "record_count": record_count,
                    },
                    "metrics": {
                        "strict_span_precision": precision,
                        "strict_span_recall": recall,
                        "strict_span_f1": f1,
                    },
                }
            )
    if not seen_tracks:
        raise ModelCardEvidenceError("no model-attributable frozen track was selected")
    suite_rank = {suite_id: index for index, suite_id in enumerate(suite_order)}
    expected_pairs = {(track, suite_id) for track in _MODEL_TRACKS for suite_id in suite_order}
    actual_pairs = {(str(item["track"]), str(item["suite_id"])) for item in results}
    if actual_pairs != expected_pairs or len(results) != len(expected_pairs):
        raise ModelCardEvidenceError(
            "every contracted dataset must have exactly one raw and one calibrated result"
        )
    return sorted(
        results,
        key=lambda item: (_TRACK_ORDER[str(item["track"])], suite_rank[str(item["suite_id"])]),
    )


def render_model_index_document(evidence: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(evidence.get("release_contract"), field="release_contract")
    publication = _mapping(contract.get("publication"), field="release_contract.publication")
    raw_results = evidence.get("frozen_model_evidence")
    if not isinstance(raw_results, list):
        raise ModelCardEvidenceError("frozen_model_evidence must be a list")
    results: list[dict[str, Any]] = []
    for item in raw_results:
        result = _mapping(item, field="frozen_model_evidence[]")
        track = str(result["track"])
        dataset = _mapping(result["dataset"], field="frozen_model_evidence[].dataset")
        metrics = _mapping(result["metrics"], field="frozen_model_evidence[].metrics")
        prefix = _METRIC_PREFIX[track]
        results.append(
            {
                "task": {"name": "Token Classification", "type": "token-classification"},
                "dataset": {
                    "name": dataset["name"],
                    "type": dataset["type"],
                    "split": dataset["split"],
                },
                "metrics": [
                    {
                        "name": f"{prefix} Strict Span Micro F1",
                        "type": "strict_span_f1",
                        "value": metrics["strict_span_f1"],
                    },
                    {
                        "name": f"{prefix} Strict Span Precision",
                        "type": "strict_span_precision",
                        "value": metrics["strict_span_precision"],
                    },
                    {
                        "name": f"{prefix} Strict Span Recall",
                        "type": "strict_span_recall",
                        "value": metrics["strict_span_recall"],
                    },
                ],
            }
        )
    return {"model-index": [{"name": publication["name"], "results": results}]}


def render_model_index(evidence: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        render_model_index_document(evidence),
        allow_unicode=True,
        sort_keys=False,
        width=1_000,
    )


def _metric_text(value: object) -> str:
    return format(float(value), ".10g")


def render_readme(evidence: Mapping[str, Any]) -> str:
    contract = _mapping(evidence.get("release_contract"), field="release_contract")
    publication = _mapping(contract.get("publication"), field="release_contract.publication")
    base = _mapping(publication.get("base_model"), field="publication.base_model")
    receipt = _mapping(evidence.get("selected_receipt"), field="selected_receipt")
    bindings = _mapping(evidence.get("source_bindings"), field="source_bindings")
    evaluation = _mapping(bindings.get("evaluation_report"), field="evaluation_report")
    claims = _mapping(contract.get("claims"), field="release_contract.claims")
    raw_results = evidence.get("frozen_model_evidence")
    if not isinstance(raw_results, list):
        raise ModelCardEvidenceError("frozen_model_evidence must be a list")
    lines = [
        "---",
        "language:",
        "  - zh",
        f"license: {publication['license']}",
        "pipeline_tag: token-classification",
        f"base_model: {base['id']}",
        "---",
        f"# {publication['name']}",
        "",
        "This Model Card is generated from the closed v2 release contract. Manual metric or ",
        "claim edits invalidate the release gate.",
        "",
        "<!-- pii-zh-model-card-evidence-v2:start -->",
        "## Release evidence",
        "",
        f"- Status: `{publication['release_status']}`",
        f"- Model ID: `{publication['model_id']}`",
        f"- License: `{publication['license']}`",
        f"- Base model: `{base['id']}` at revision `{base['revision']}`",
        f"- Base-model license: `{base['license']}`",
        f"- Selected candidate: `{receipt['candidate_id']}`",
        f"- Selection receipt: `{receipt['receipt_id']}` / `{receipt['receipt_sha256']}`",
        f"- Model weights SHA-256: `{receipt['artifact']['files']['model.safetensors']}`",
        f"- Frozen evaluation SHA-256: `{evaluation['manifest_sha256']}`",
        "",
        "## Frozen model-only results",
        "",
        "| Track | Dataset | Split | Precision | Recall | F1 | Records |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in raw_results:
        result = _mapping(item, field="frozen_model_evidence[]")
        dataset = _mapping(result["dataset"], field="frozen_model_evidence[].dataset")
        metrics = _mapping(result["metrics"], field="frozen_model_evidence[].metrics")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(result["track"]),
                    str(dataset["name"]),
                    str(dataset["split"]),
                    _metric_text(metrics["strict_span_precision"]),
                    _metric_text(metrics["strict_span_recall"]),
                    _metric_text(metrics["strict_span_f1"]),
                    str(dataset["record_count"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Only checkpoint output is reported above. Rules, Presidio, cascade, framework and ",
            "full-system results are excluded from model attribution.",
            "",
            "## Claim boundary",
            "",
            "- First Simplified Chinese PII model claimed: `"
            + str(claims["first_simplified_chinese_pii_model"]).lower()
            + "`",
            f"- Best Chinese PII model claimed: `{str(claims['best_chinese_pii_model']).lower()}`",
            f"- State of the art claimed: `{str(claims['state_of_the_art']).lower()}`",
            f"- Production compliance claimed: `{str(claims['production_compliance']).lower()}`",
            "",
            "This evidence does not constitute a production compliance guarantee.",
            "<!-- pii-zh-model-card-evidence-v2:end -->",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_contract_and_receipt(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    training_manifest_sha256: str,
    evaluation_report_sha256: str,
) -> None:
    _verify_self_hash(contract, field="contract_sha256", description="release contract")
    _verify_self_hash(receipt, field="receipt_sha256", description="selected candidate receipt")
    if receipt.get("selection_count") != 1 or receipt.get("selected") is not True:
        raise ModelCardEvidenceError("BLOCKED: selected receipt is not uniquely selected")
    if receipt.get("release_eligible") is not True:
        raise ModelCardEvidenceError("BLOCKED: selected candidate is not release eligible")
    publication = _mapping(contract.get("publication"), field="release_contract.publication")
    if receipt.get("release_status") != publication.get("release_status"):
        raise ModelCardEvidenceError("selected receipt release status differs from contract")
    if receipt.get("license") != publication.get("license"):
        raise ModelCardEvidenceError("selected receipt release license differs from contract")
    if receipt.get("base_model") != publication.get("base_model"):
        raise ModelCardEvidenceError("selected receipt base model differs from contract")
    if receipt.get("artifact") != artifact:
        raise ModelCardEvidenceError("selected receipt artifact hash differs from checkpoint")
    if receipt.get("training_manifest_sha256") != training_manifest_sha256:
        raise ModelCardEvidenceError("selected receipt is not bound to training manifest")
    if receipt.get("evaluation_report_sha256") != evaluation_report_sha256:
        raise ModelCardEvidenceError("selected receipt is not bound to frozen evaluation")
    claims = _mapping(contract.get("claims"), field="release_contract.claims")
    if any(value is not False for value in claims.values()):
        raise ModelCardEvidenceError(
            "BLOCKED: first/best/SOTA/production claims have no v2 authorization receipt"
        )
    _validate_rendered_contract_text(contract)


def build_evidence(
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    selected_receipt: Mapping[str, Any],
    selected_receipt_file_sha256: str,
    evaluation_report: Mapping[str, Any],
    evaluation_report_file_sha256: str,
    training_manifest: Mapping[str, Any],
    training_manifest_file_sha256: str,
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    _validate_schema_definition(
        contract,
        definition="release_contract",
        description="release contract",
    )
    _validate_schema_definition(
        selected_receipt,
        definition="selected_receipt",
        description="selected candidate receipt",
    )
    _assert_no_absolute_paths(contract)
    _assert_no_absolute_paths(selected_receipt)
    training_manifest_sha256 = _verify_self_hash(
        training_manifest, field="manifest_sha256", description="training manifest"
    )
    if training_manifest.get("status") != "completed":
        raise ModelCardEvidenceError("BLOCKED: training manifest is not completed")
    if training_manifest.get("output_artifact") != artifact:
        raise ModelCardEvidenceError("training manifest artifact hash differs from checkpoint")
    evaluation_report_sha256 = _verify_self_hash(
        evaluation_report, field="manifest_sha256", description="evaluation report"
    )
    _validate_contract_and_receipt(
        contract,
        selected_receipt,
        artifact=artifact,
        training_manifest_sha256=training_manifest_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
    )
    frozen = extract_frozen_model_evidence(
        evaluation_report,
        contract,
        training_manifest_sha256=training_manifest_sha256,
    )
    evidence: dict[str, Any] = {
        "artifact_type": "pii_zh_model_card_release_evidence",
        "schema_version": 2,
        "release_eligible": True,
        "release_contract": dict(contract),
        "selected_receipt": dict(selected_receipt),
        "source_bindings": {
            "release_contract": {
                "file_sha256": contract_file_sha256,
                "contract_sha256": contract["contract_sha256"],
            },
            "selected_receipt": {
                "file_sha256": selected_receipt_file_sha256,
                "receipt_sha256": selected_receipt["receipt_sha256"],
            },
            "training_manifest": {
                "file_sha256": training_manifest_file_sha256,
                "manifest_sha256": training_manifest_sha256,
            },
            "evaluation_report": {
                "file_sha256": evaluation_report_file_sha256,
                "manifest_sha256": evaluation_report_sha256,
            },
        },
        "frozen_model_evidence": frozen,
    }
    model_index = render_model_index(evidence)
    readme = render_readme(evidence)
    evidence["rendered_artifacts"] = {
        README_FILENAME: hashlib.sha256(readme.encode()).hexdigest(),
        MODEL_INDEX_FILENAME: hashlib.sha256(model_index.encode()).hexdigest(),
    }
    evidence["evidence_sha256"] = canonical_json_hash(evidence)
    validate_evidence_document(evidence)
    return evidence, model_index, readme


def validate_evidence_document(evidence: Mapping[str, Any]) -> None:
    validate_evidence_schema(evidence)
    _assert_no_absolute_paths(evidence)
    _verify_self_hash(evidence, field="evidence_sha256", description="v2 evidence")
    contract = _mapping(evidence.get("release_contract"), field="release_contract")
    receipt = _mapping(evidence.get("selected_receipt"), field="selected_receipt")
    _verify_self_hash(contract, field="contract_sha256", description="release contract")
    _verify_self_hash(receipt, field="receipt_sha256", description="selected candidate receipt")
    _validate_rendered_contract_text(contract)


def validate_release_components(
    *,
    evidence_path: Path,
    contract_path: Path,
    selected_receipt_path: Path,
    training_path: Path,
    evaluation_path: Path,
    checkpoint_dir: Path,
    model_index_path: Path,
    readme_path: Path,
    bound_release_text_paths: Mapping[str, Path],
    expected_contract_sha256: str | None = None,
    expected_selected_receipt_sha256: str | None = None,
) -> None:
    """Rebuild v2 projections and optionally enforce external trust anchors."""

    evidence = _load_json(evidence_path, description="model card evidence v2")
    validate_evidence_document(evidence)
    packaged_contract = _load_json(contract_path, description="packaged v2 release contract")
    packaged_receipt = _load_json(
        selected_receipt_path, description="packaged selected candidate receipt"
    )
    contract = _mapping(evidence.get("release_contract"), field="release_contract")
    receipt = _mapping(evidence.get("selected_receipt"), field="selected_receipt")
    if packaged_contract != contract:
        raise ModelCardEvidenceError("packaged v2 contract differs from embedded evidence")
    if packaged_receipt != receipt:
        raise ModelCardEvidenceError("packaged selected receipt differs from embedded evidence")
    validate_bound_release_text_files(contract, bound_release_text_paths)
    contract_sha256 = _verify_self_hash(
        contract, field="contract_sha256", description="release contract"
    )
    receipt_sha256 = _verify_self_hash(
        receipt, field="receipt_sha256", description="selected candidate receipt"
    )
    if expected_contract_sha256 is not None and contract_sha256 != expected_contract_sha256:
        raise ModelCardEvidenceError("external frozen v2 contract SHA-256 does not match")
    if (
        expected_selected_receipt_sha256 is not None
        and receipt_sha256 != expected_selected_receipt_sha256
    ):
        raise ModelCardEvidenceError("external frozen selected-receipt SHA-256 does not match")
    training = _load_json(training_path, description="packaged training manifest")
    evaluation = _load_json(evaluation_path, description="packaged evaluation report")
    rebuilt, expected_model_index, expected_readme = build_evidence(
        contract=contract,
        contract_file_sha256=sha256_file(contract_path),
        selected_receipt=receipt,
        selected_receipt_file_sha256=sha256_file(selected_receipt_path),
        evaluation_report=evaluation,
        evaluation_report_file_sha256=sha256_file(evaluation_path),
        training_manifest=training,
        training_manifest_file_sha256=sha256_file(training_path),
        artifact=_artifact_fingerprint(checkpoint_dir),
    )
    if rebuilt != evidence:
        raise ModelCardEvidenceError("v2 evidence differs from its frozen source projection")
    try:
        actual_model_index = model_index_path.read_text(encoding="utf-8")
        actual_readme = readme_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelCardEvidenceError("generated Model Card artifacts are unreadable") from exc
    if actual_model_index != expected_model_index:
        raise ModelCardEvidenceError(
            "model-index.yml differs from frozen model-only v2 evidence"
        )
    if actual_readme != expected_readme:
        raise ModelCardEvidenceError("README.md differs from deterministic v2 evidence")


def validate_release_bundle(
    artifact_dir: Path,
    *,
    expected_contract_sha256: str | None = None,
    expected_selected_receipt_sha256: str | None = None,
) -> None:
    """Rebuild packaged v2 evidence projections from their frozen sources."""

    validate_release_components(
        evidence_path=artifact_dir / EVIDENCE_FILENAME,
        contract_path=artifact_dir / CONTRACT_FILENAME,
        selected_receipt_path=artifact_dir / SELECTED_RECEIPT_FILENAME,
        training_path=artifact_dir / "training_manifest.json",
        evaluation_path=artifact_dir / "evaluation_report.json",
        checkpoint_dir=artifact_dir,
        model_index_path=artifact_dir / MODEL_INDEX_FILENAME,
        readme_path=artifact_dir / README_FILENAME,
        bound_release_text_paths={
            name: artifact_dir / name for name in BOUND_RELEASE_TEXT_FILES
        },
        expected_contract_sha256=expected_contract_sha256,
        expected_selected_receipt_sha256=expected_selected_receipt_sha256,
    )


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without an existence-check race."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ModelCardEvidenceError(
            "atomic no-clobber directory publication requires renameat2(RENAME_NOREPLACE)"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ModelCardEvidenceError(
            f"output appeared during publication; refusing to clobber: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise ModelCardEvidenceError(
            "atomic no-clobber directory publication is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def write_bundle_no_clobber(
    output_dir: Path,
    *,
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    selected_receipt: Mapping[str, Any],
    model_index: str,
    readme: str,
) -> Path:
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise ModelCardEvidenceError(f"output already exists; refusing to clobber: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cursor = output_dir.parent
    while True:
        if cursor.is_symlink():
            raise ModelCardEvidenceError("output directory ancestry must not contain symlinks")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        _write_text(
            staging / EVIDENCE_FILENAME,
            _render_json(evidence),
        )
        _write_text(staging / CONTRACT_FILENAME, _render_json(contract))
        _write_text(staging / SELECTED_RECEIPT_FILENAME, _render_json(selected_receipt))
        _write_text(staging / MODEL_INDEX_FILENAME, model_index)
        _write_text(staging / README_FILENAME, readme)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_directory_no_replace(staging, output_dir)
        published = True
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return output_dir


def assemble(
    *,
    contract_path: Path,
    selected_receipt_path: Path | None,
    evaluation_report_path: Path,
    training_manifest_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Path:
    contract = _load_json(contract_path, description="v2 release contract")
    if selected_receipt_path is None:
        raise ModelCardEvidenceError(
            "BLOCKED: explicit v2 contract has no selected candidate receipt"
        )
    receipt = _load_json(selected_receipt_path, description="selected candidate receipt")
    evaluation = _load_json(evaluation_report_path, description="frozen evaluation report")
    training = _load_json(training_manifest_path, description="training manifest")
    evidence, model_index, readme = build_evidence(
        contract=contract,
        contract_file_sha256=hashlib.sha256(_render_json(contract).encode()).hexdigest(),
        selected_receipt=receipt,
        selected_receipt_file_sha256=hashlib.sha256(
            _render_json(receipt).encode()
        ).hexdigest(),
        evaluation_report=evaluation,
        evaluation_report_file_sha256=sha256_file(evaluation_report_path),
        training_manifest=training,
        training_manifest_file_sha256=sha256_file(training_manifest_path),
        artifact=_artifact_fingerprint(checkpoint_dir),
    )
    return write_bundle_no_clobber(
        output_dir,
        evidence=evidence,
        contract=contract,
        selected_receipt=receipt,
        model_index=model_index,
        readme=readme,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--selected-receipt", type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = assemble(
            contract_path=args.contract,
            selected_receipt_path=args.selected_receipt,
            evaluation_report_path=args.evaluation_report,
            training_manifest_path=args.training_manifest,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
        )
    except (ModelCardEvidenceError, OSError, jsonschema.SchemaError) as exc:
        print(f"model card evidence v2 BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Built local model-card evidence v2 bundle: {output}")
    print("No flagship Model Card was published or uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
