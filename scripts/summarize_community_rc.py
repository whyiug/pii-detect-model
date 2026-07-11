#!/usr/bin/env python3
"""Build fail-closed, path-free three-seed community RC quality evidence.

This command consumes aggregate validation reports only.  It deliberately has
no gold, prediction JSONL, or frozen-test argument, so it cannot open those
artifacts while assembling release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - CLI dependency error
    raise SystemExit("summarize_community_rc.py requires PyYAML") from exc

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import add_manifest_hash, canonical_json_hash  # noqa: E402
from pii_zh.fusion import system_fusion_configuration  # noqa: E402

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_REQUIRED_SEEDS = (13, 42, 97)
_METRIC_NAMES = (
    "strict_micro_f1",
    "strict_macro_f1",
    "pii_free_false_positive_rate",
    "tier0_min_label_recall",
    "tier1_min_label_recall",
    "calibration_completed",
)
_OUTPUT_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
)
_VALIDATION_RERUN_HINT = (
    "rerun validation only: python scripts/evaluate.py "
    '--gold "$VALIDATION_JSONL" --predictions "$FINAL_VALIDATION_JSONL" '
    '--model-training-manifest "$TRAINING_MANIFEST" '
    '--prediction-manifest "$SYSTEM_PREDICTION_MANIFEST" '
    '--output "$FINAL_EVALUATION_JSON"'
)


class CommunityRCEvidenceError(ValueError):
    """Raised when a source cannot safely support an RC decision."""


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CommunityRCEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_PATTERN.fullmatch(value) is None:
        raise CommunityRCEvidenceError(f"{field} must be a path-free stable identifier")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommunityRCEvidenceError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CommunityRCEvidenceError(f"{field} must be a finite number")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise CommunityRCEvidenceError(f"{field} must be between zero and one")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CommunityRCEvidenceError(f"{field} must be an integer >= {minimum}")
    return value


def _read_regular_bytes(
    path: Path,
    *,
    description: str,
    maximum_bytes: int | None = None,
) -> tuple[bytes, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CommunityRCEvidenceError(f"cannot open regular {description}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CommunityRCEvidenceError(f"{description} must be a regular file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise CommunityRCEvidenceError(f"{description} exceeds the safe size limit")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise CommunityRCEvidenceError(f"{description} changed while it was read")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def _sha256_regular_file(path: Path, *, description: str) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CommunityRCEvidenceError(f"cannot open regular {description}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CommunityRCEvidenceError(f"{description} must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise CommunityRCEvidenceError(f"{description} changed while it was hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    encoded, file_sha256 = _read_regular_bytes(
        path,
        description=description,
        maximum_bytes=64 * 1024 * 1024,
    )
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommunityRCEvidenceError(f"{description} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CommunityRCEvidenceError(f"{description} must be a JSON object")
    return value, file_sha256


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    encoded, file_sha256 = _read_regular_bytes(
        path,
        description="decision config",
        maximum_bytes=1024 * 1024,
    )
    try:
        value = yaml.safe_load(encoded.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CommunityRCEvidenceError("decision config must be UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise CommunityRCEvidenceError("decision config must be a YAML mapping")
    return value, file_sha256


def _verify_self_hash(document: Mapping[str, Any], *, description: str) -> str:
    expected = _sha256(document.get("manifest_sha256"), field=f"{description}.manifest_sha256")
    unsigned = dict(document)
    unsigned.pop("manifest_sha256", None)
    if canonical_json_hash(unsigned) != expected:
        raise CommunityRCEvidenceError(f"{description} self-hash does not verify")
    return expected


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommunityRCEvidenceError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CommunityRCEvidenceError(f"{field} must be a sequence")
    return value


def _verify_decision_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1 or config.get("status") != "amended_before_holdout_access":
        raise CommunityRCEvidenceError(
            "decision config must be a pre-holdout amendment with schema version 1"
        )
    decision_id = _safe_id(config.get("decision_id"), field="decision_id")
    if decision_id != "synthetic_v1_3_community_rc_v3":
        raise CommunityRCEvidenceError("decision config must be community RC amendment v3")
    if (
        config.get("candidate_scope") != "community_research_release_candidate"
        or config.get("production_ready") is not False
    ):
        raise CommunityRCEvidenceError("decision config has the wrong release scope")
    parent_decision = _mapping(config.get("parent_decision"), field="parent_decision")
    parent_decision_id = _safe_id(
        parent_decision.get("decision_id"), field="parent_decision.decision_id"
    )
    parent_config_sha256 = _sha256(
        parent_decision.get("config_sha256"), field="parent_decision.config_sha256"
    )
    _safe_id(parent_decision.get("code_revision"), field="parent_decision.code_revision")
    amendment = _mapping(config.get("amendment"), field="amendment")
    unchanged_flags = (
        "holdout_accessed",
        "thresholds_changed",
        "seeds_changed",
        "selected_seed_changed",
    )
    if any(amendment.get(field) is not False for field in unchanged_flags):
        raise CommunityRCEvidenceError("amendment must precede holdout access and preserve gates")
    allowed_changes = _sequence(amendment.get("allowed_changes"), field="amendment.allowed_changes")
    if tuple(allowed_changes) != ("require_affirmative_qq_context_during_refinement",):
        raise CommunityRCEvidenceError("amendment contains an unapproved system change")
    forbidden_changes = set(
        _sequence(amendment.get("forbidden_changes"), field="amendment.forbidden_changes")
    )
    required_forbidden = {
        "lower_quality_thresholds",
        "change_seed_set",
        "change_model_or_calibration",
        "inspect_or_tune_on_frozen_test",
        "refit_calibration_on_any_test_subset",
    }
    if not required_forbidden <= forbidden_changes:
        raise CommunityRCEvidenceError("amendment is missing required holdout prohibitions")
    dataset = _mapping(config.get("dataset"), field="dataset")
    validation_sha256 = _sha256(dataset.get("validation_sha256"), field="dataset.validation_sha256")
    _sha256(dataset.get("frozen_test_sha256"), field="dataset.frozen_test_sha256")
    if dataset.get("frozen_test_access_before_validation_gate") != "forbidden":
        raise CommunityRCEvidenceError("frozen test access must remain forbidden")

    system = _mapping(config.get("system"), field="system")
    seeds = _sequence(system.get("seeds"), field="system.seeds")
    if tuple(seeds) != _REQUIRED_SEEDS:
        raise CommunityRCEvidenceError("decision config seeds must be exactly [13, 42, 97]")
    if (
        system.get("selected_seed") != 42
        or system.get("selection_rule") != "fixed_conventional_seed_before_holdout_access"
    ):
        raise CommunityRCEvidenceError("selected seed must remain fixed at 42 before holdout")
    attention_mode = _safe_id(system.get("attention_mode"), field="system.attention_mode")
    initializer_attention_mode = _safe_id(
        system.get("initializer_attention_mode"), field="system.initializer_attention_mode"
    )
    ruleset_id = _safe_id(system.get("ruleset_id"), field="system.ruleset_id")
    if ruleset_id != "cn_common_v5":
        raise CommunityRCEvidenceError("community RC v3 requires cn_common_v5")
    rules_implementation_sha256 = _sha256(
        system.get("rules_implementation_sha256"), field="system.rules_implementation_sha256"
    )
    fusion_id = _safe_id(system.get("fusion"), field="system.fusion")
    if fusion_id != "deterministic_fusion_v1":
        raise CommunityRCEvidenceError("community RC v3 requires deterministic_fusion_v1")
    fusion_implementation_sha256 = _sha256(
        system.get("fusion_implementation_sha256"),
        field="system.fusion_implementation_sha256",
    )
    calibration = _mapping(system.get("calibration"), field="system.calibration")
    if (
        calibration.get("fit_per_seed_on_validation") is not True
        or calibration.get("selected_seed_bundle_frozen_before_holdout_access") is not True
        or calibration.get("holdout_policy") != "apply_only_no_refit"
    ):
        raise CommunityRCEvidenceError("calibration fit/apply holdout policy is invalid")
    t0_floor = _probability(
        calibration.get("t0_recall_floor"), field="system.calibration.t0_recall_floor"
    )
    t1_floor = _probability(
        calibration.get("t1_recall_floor"), field="system.calibration.t1_recall_floor"
    )
    temperature_enabled = calibration.get("temperature_scaling")
    if not isinstance(temperature_enabled, bool):
        raise CommunityRCEvidenceError("calibration.temperature_scaling must be boolean")
    refinement_contract = _safe_id(system.get("refinement"), field="system.refinement")
    if refinement_contract != "structured_prediction_refinement_v4":
        raise CommunityRCEvidenceError("community RC v3 requires refinement v4")
    refinement_implementation_sha256 = _sha256(
        system.get("refinement_implementation_sha256"),
        field="system.refinement_implementation_sha256",
    )

    holdout_unlock = _mapping(config.get("holdout_unlock"), field="holdout_unlock")
    if (
        holdout_unlock.get("requires_release_decision") != "passed"
        or tuple(
            _sequence(
                holdout_unlock.get("required_seed_evidence"),
                field="holdout_unlock.required_seed_evidence",
            )
        )
        != _REQUIRED_SEEDS
        or holdout_unlock.get("calibration_refit_on_holdout") != "forbidden"
        or holdout_unlock.get("post_unlock_changes_to_model_or_system") != "forbidden"
    ):
        raise CommunityRCEvidenceError("holdout unlock policy is incomplete")

    per_seed_criteria = _mapping(config.get("per_seed_criteria"), field="per_seed_criteria")
    if set(per_seed_criteria) != set(_METRIC_NAMES):
        raise CommunityRCEvidenceError("per_seed_criteria must define the approved metric set")
    aggregate_criteria = _mapping(config.get("aggregate_criteria"), field="aggregate_criteria")
    if set(aggregate_criteria) != {"unique_seed_count", "every_seed_passes"}:
        raise CommunityRCEvidenceError("aggregate_criteria must define both approved gates")
    calibration_criterion = _mapping(
        per_seed_criteria["calibration_completed"],
        field="per_seed_criteria.calibration_completed",
    )
    every_seed_criterion = _mapping(
        aggregate_criteria["every_seed_passes"],
        field="aggregate_criteria.every_seed_passes",
    )
    unique_seed_criterion = _mapping(
        aggregate_criteria["unique_seed_count"],
        field="aggregate_criteria.unique_seed_count",
    )
    if calibration_criterion != {"operator": "==", "threshold": True}:
        raise CommunityRCEvidenceError("calibration_completed must require equality to true")
    if every_seed_criterion != {"operator": "==", "threshold": True}:
        raise CommunityRCEvidenceError("every_seed_passes must require equality to true")
    if unique_seed_criterion != {"operator": ">=", "threshold": 3}:
        raise CommunityRCEvidenceError("unique_seed_count must require at least three")

    return {
        "decision_id": decision_id,
        "parent_decision_id": parent_decision_id,
        "parent_config_sha256": parent_config_sha256,
        "validation_sha256": validation_sha256,
        "attention_mode": attention_mode,
        "initializer_attention_mode": initializer_attention_mode,
        "t0_floor": t0_floor,
        "t1_floor": t1_floor,
        "temperature_enabled": temperature_enabled,
        "ruleset_id": ruleset_id,
        "rules_implementation_sha256": rules_implementation_sha256,
        "fusion_id": fusion_id,
        "fusion_implementation_sha256": fusion_implementation_sha256,
        "refinement_contract": refinement_contract,
        "refinement_id": "structured_refinement_v4",
        "refinement_implementation_sha256": refinement_implementation_sha256,
        "per_seed_criteria": per_seed_criteria,
        "aggregate_criteria": aggregate_criteria,
    }


def _output_artifact_fingerprint(root: Path) -> dict[str, Any]:
    weights = tuple(sorted(root.glob("model*.safetensors")))
    if not weights:
        raise CommunityRCEvidenceError("training output contains no model safetensors")
    files: dict[str, str] = {}
    for path in weights:
        files[path.name] = _sha256_regular_file(path, description="training output weight")
    for name in _OUTPUT_METADATA_FILES:
        path = root / name
        if path.is_symlink():
            raise CommunityRCEvidenceError("training output metadata must not be a symlink")
        if path.is_file():
            files[name] = _sha256_regular_file(path, description="training output metadata")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= files.keys():
        raise CommunityRCEvidenceError("training output is missing required model metadata")
    weight_hashes = {name: files[name] for name in sorted(files) if name.endswith(".safetensors")}
    return {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": sorted(weight_hashes),
        "weights_combined_sha256": canonical_json_hash(weight_hashes),
        "artifact_files_combined_sha256": canonical_json_hash(files),
    }


def _verify_training_manifest(
    path: Path,
    *,
    seed: int,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    manifest, file_sha256 = _load_json(path, description=f"seed {seed} training manifest")
    logical_sha256 = _verify_self_hash(manifest, description=f"seed {seed} training manifest")
    if manifest.get("status") != "completed" or manifest.get("seed") != seed:
        raise CommunityRCEvidenceError(f"seed {seed} training manifest is not completed/bound")
    if manifest.get("attention_mode") != contract["attention_mode"]:
        raise CommunityRCEvidenceError(f"seed {seed} training attention mode is inconsistent")
    recipe = _mapping(manifest.get("recipe"), field=f"seed {seed} training.recipe")
    if recipe.get("seed") != seed or recipe.get("attention_mode") != contract["attention_mode"]:
        raise CommunityRCEvidenceError(f"seed {seed} training recipe identity is inconsistent")
    initialization = _mapping(
        manifest.get("initialization"), field=f"seed {seed} training.initialization"
    )
    if initialization.get("source_attention_mode") != contract["initializer_attention_mode"]:
        raise CommunityRCEvidenceError(f"seed {seed} initializer attention mode is inconsistent")
    datasets = _mapping(manifest.get("datasets"), field=f"seed {seed} training.datasets")
    validation = _mapping(
        datasets.get("validation"), field=f"seed {seed} training.datasets.validation"
    )
    if validation.get("sha256") != contract["validation_sha256"]:
        raise CommunityRCEvidenceError(f"seed {seed} training validation hash is inconsistent")

    recorded_output = _mapping(
        manifest.get("output_artifact"), field=f"seed {seed} training.output_artifact"
    )
    actual_output = _output_artifact_fingerprint(path.parent)
    if actual_output.get("weight_files") != ["model.safetensors"]:
        raise CommunityRCEvidenceError(f"seed {seed} must use exactly model.safetensors")
    if dict(recorded_output) != actual_output:
        raise CommunityRCEvidenceError(f"seed {seed} training output artifact does not verify")
    return manifest, logical_sha256, file_sha256


def _score_from_counts(value: Mapping[str, Any], *, field: str) -> tuple[float, float]:
    true_positive = _integer(value.get("tp"), field=f"{field}.tp")
    false_positive = _integer(value.get("fp"), field=f"{field}.fp")
    false_negative = _integer(value.get("fn"), field=f"{field}.fn")
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    reported_recall = _probability(value.get("recall"), field=f"{field}.recall")
    reported_f1 = _probability(value.get("f1"), field=f"{field}.f1")
    if not math.isclose(reported_recall, recall, rel_tol=0.0, abs_tol=1e-12):
        raise CommunityRCEvidenceError(f"{field}.recall does not match its counts")
    if not math.isclose(reported_f1, f1, rel_tol=0.0, abs_tol=1e-12):
        raise CommunityRCEvidenceError(f"{field}.f1 does not match its counts")
    return recall, f1


def _verify_evaluation_provenance(
    report: Mapping[str, Any],
    *,
    seed: int,
    training_manifest_sha256: str,
    training_manifest_file_sha256: str,
    contract: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    provenance = _mapping(report.get("provenance"), field=f"seed {seed} evaluation.provenance")
    expected = _sha256(
        provenance.get("evaluation_sha256"),
        field=f"seed {seed} evaluation.provenance.evaluation_sha256",
    )
    unsigned = dict(provenance)
    unsigned.pop("evaluation_sha256", None)
    if canonical_json_hash(unsigned) != expected:
        raise CommunityRCEvidenceError(f"seed {seed} evaluation provenance self-hash failed")
    privacy = _mapping(
        provenance.get("privacy"), field=f"seed {seed} evaluation.provenance.privacy"
    )
    if any(
        privacy.get(field) is not False
        for field in ("contains_paths", "contains_raw_text", "contains_entity_values")
    ):
        raise CommunityRCEvidenceError(f"seed {seed} evaluation provenance is not public-safe")
    gold = _mapping(provenance.get("gold"), field=f"seed {seed} evaluation.provenance.gold")
    if gold.get("sha256") != contract["validation_sha256"]:
        raise CommunityRCEvidenceError(f"seed {seed} evaluation is not bound to validation")

    training = provenance.get("model_training")
    if not isinstance(training, Mapping):
        raise CommunityRCEvidenceError(
            f"seed {seed} evaluation lacks model_training provenance; {_VALIDATION_RERUN_HINT}"
        )
    expected_training = {
        "manifest_sha256": training_manifest_sha256,
        "manifest_file_sha256": training_manifest_file_sha256,
        "seed": seed,
        "attention_mode": contract["attention_mode"],
    }
    if any(training.get(field) != value for field, value in expected_training.items()):
        raise CommunityRCEvidenceError(
            f"seed {seed} evaluation/training provenance is inconsistent"
        )

    predictions = _mapping(
        provenance.get("predictions"), field=f"seed {seed} evaluation.provenance.predictions"
    )
    predictions_sha256 = _sha256(
        predictions.get("sha256"), field=f"seed {seed} evaluation predictions"
    )
    prediction_manifest = predictions.get("manifest")
    if not isinstance(prediction_manifest, Mapping):
        raise CommunityRCEvidenceError(
            f"seed {seed} evaluation lacks a system prediction manifest; {_VALIDATION_RERUN_HINT}"
        )
    bound = prediction_manifest
    if bound.get("predictor_type") != "system":
        raise CommunityRCEvidenceError(
            f"seed {seed} final evaluation must use predictor_type=system"
        )
    for field in ("manifest_sha256", "manifest_file_sha256", "dataset_manifest_sha256"):
        _sha256(bound.get(field), field=f"seed {seed} system prediction {field}")
    if bound.get("predictions_sha256") != predictions_sha256:
        raise CommunityRCEvidenceError(f"seed {seed} prediction output hash is inconsistent")
    if bound.get("model_training_manifest_sha256") != training_manifest_sha256:
        raise CommunityRCEvidenceError(f"seed {seed} prediction/training binding is inconsistent")
    if bound.get("seed") != seed:
        raise CommunityRCEvidenceError(f"seed {seed} prediction seed is inconsistent")
    if bound.get("attention_mode") != contract["attention_mode"]:
        raise CommunityRCEvidenceError(f"seed {seed} prediction attention is inconsistent")
    prediction_document_count = _integer(
        bound.get("prediction_document_count"),
        field=f"seed {seed} system prediction document count",
        minimum=1,
    )
    report_document_count = _integer(
        report.get("document_count"), field=f"seed {seed} evaluation document count", minimum=1
    )
    if prediction_document_count != report_document_count:
        raise CommunityRCEvidenceError(f"seed {seed} prediction/evaluation counts differ")

    components = _mapping(
        bound.get("components"), field=f"seed {seed} system prediction components"
    )
    expected_components = {
        "model_prediction",
        "rules_prediction",
        "fusion",
        "calibration_fit",
        "target_application",
        "refinement",
    }
    if set(components) != expected_components:
        raise CommunityRCEvidenceError(f"seed {seed} system component set is incomplete")
    model = _mapping(
        components.get("model_prediction"),
        field=f"seed {seed} system prediction model component",
    )
    if model.get("model_training_manifest_sha256") != training_manifest_sha256:
        raise CommunityRCEvidenceError(
            f"seed {seed} system model component/training binding is inconsistent"
        )
    rules = _mapping(
        components.get("rules_prediction"),
        field=f"seed {seed} system prediction rules component",
    )
    rules_identity = _mapping(
        rules.get("rules_identity"),
        field=f"seed {seed} system prediction rules identity",
    )
    if rules_identity.get("ruleset_id") != contract["ruleset_id"]:
        raise CommunityRCEvidenceError(f"seed {seed} does not use cn_common_v5")
    if rules_identity.get("implementation_sha256") != contract["rules_implementation_sha256"]:
        raise CommunityRCEvidenceError(f"seed {seed} rules implementation hash differs")
    _sha256(
        rules_identity.get("configuration_sha256"),
        field=f"seed {seed} rules configuration",
    )

    fusion = _mapping(
        components.get("fusion"), field=f"seed {seed} system prediction fusion component"
    )
    if fusion.get("implementation_sha256") != contract["fusion_implementation_sha256"]:
        raise CommunityRCEvidenceError(f"seed {seed} fusion implementation hash differs")
    expected_fusion_configuration = canonical_json_hash(system_fusion_configuration())
    if fusion.get("configuration_sha256") != expected_fusion_configuration:
        raise CommunityRCEvidenceError(f"seed {seed} fusion configuration hash differs")
    for field in ("module_sha256", "cli_sha256", "output_predictions_sha256"):
        _sha256(fusion.get(field), field=f"seed {seed} fusion {field}")

    calibration_fit = _mapping(
        components.get("calibration_fit"),
        field=f"seed {seed} system prediction calibration-fit component",
    )
    target_application = _mapping(
        components.get("target_application"),
        field=f"seed {seed} system prediction target-application component",
    )
    refinement = _mapping(
        components.get("refinement"),
        field=f"seed {seed} system prediction refinement component",
    )
    if refinement.get("refinement_id") != contract["refinement_id"]:
        raise CommunityRCEvidenceError(f"seed {seed} does not use refinement v4")
    if refinement.get("implementation_sha256") != contract["refinement_implementation_sha256"]:
        raise CommunityRCEvidenceError(f"seed {seed} refinement implementation hash differs")
    _sha256(
        refinement.get("audit_manifest_sha256"),
        field=f"seed {seed} refinement audit manifest",
    )
    if refinement.get("output_predictions_sha256") != predictions_sha256:
        raise CommunityRCEvidenceError(
            f"seed {seed} refinement output is not the evaluated prediction output"
        )
    if (
        calibration_fit.get("gold_sha256") != contract["validation_sha256"]
        or calibration_fit.get("fused_predictions_sha256")
        != target_application.get("input_predictions_sha256")
        or fusion.get("output_predictions_sha256")
        != target_application.get("input_predictions_sha256")
    ):
        raise CommunityRCEvidenceError(
            f"seed {seed} validation calibration-fit/target binding differs"
        )
    if calibration_fit.get("bundle_sha256") != target_application.get(
        "bundle_sha256"
    ) or calibration_fit.get("calibration_version") != target_application.get(
        "calibration_version"
    ):
        raise CommunityRCEvidenceError(
            f"seed {seed} calibration-fit and target application bundle differ"
        )
    if target_application.get("output_predictions_sha256") != refinement.get(
        "input_predictions_sha256"
    ):
        raise CommunityRCEvidenceError(f"seed {seed} target application/refinement binding differs")
    for name, component in (
        ("model", model),
        ("rules", rules),
        ("fusion", fusion),
        ("target application", target_application),
        ("refinement", refinement),
    ):
        count = _integer(
            component.get("prediction_document_count"),
            field=f"seed {seed} system {name} document count",
            minimum=1,
        )
        if count != report_document_count:
            raise CommunityRCEvidenceError(f"seed {seed} system component counts differ")
    calibration_fit_count = _integer(
        calibration_fit.get("document_count"),
        field=f"seed {seed} calibration-fit document count",
        minimum=1,
    )
    if calibration_fit_count != report_document_count:
        raise CommunityRCEvidenceError(
            f"seed {seed} validation calibration-fit/evaluation counts differ"
        )
    return (
        expected,
        predictions_sha256,
        {
            "diagnostics_manifest_sha256": _sha256(
                calibration_fit.get("diagnostics_manifest_sha256"),
                field=f"seed {seed} system calibration diagnostics",
            ),
            "calibration_bundle_sha256": _sha256(
                calibration_fit.get("bundle_sha256"),
                field=f"seed {seed} system calibration bundle",
            ),
            "calibration_version": _safe_id(
                calibration_fit.get("calibration_version"),
                field=f"seed {seed} system calibration version",
            ),
            "calibration_fit_predictions_sha256": _sha256(
                calibration_fit.get("fused_predictions_sha256"),
                field=f"seed {seed} system calibration-fit predictions",
            ),
            "document_count": report_document_count,
        },
    )


def _verify_calibration(
    path: Path,
    *,
    seed: int,
    expected_labels: set[str],
    expected_document_count: int,
    contract: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    diagnostics, file_sha256 = _load_json(path, description=f"seed {seed} calibration diagnostics")
    logical_sha256 = _verify_self_hash(
        diagnostics, description=f"seed {seed} calibration diagnostics"
    )
    if diagnostics.get("manifest_type") != "calibration_diagnostics":
        raise CommunityRCEvidenceError(f"seed {seed} has the wrong calibration manifest type")
    _safe_id(
        diagnostics.get("calibration_version"),
        field=f"seed {seed} calibration.calibration_version",
    )
    bundle_sha256 = _sha256(
        diagnostics.get("calibration_bundle_sha256"),
        field=f"seed {seed} calibration.calibration_bundle_sha256",
    )
    privacy = _mapping(diagnostics.get("privacy"), field=f"seed {seed} calibration.privacy")
    if any(
        privacy.get(field) is not False
        for field in (
            "contains_paths",
            "contains_raw_text",
            "contains_entity_values",
            "contains_document_ids",
        )
    ):
        raise CommunityRCEvidenceError(f"seed {seed} calibration diagnostics are not public-safe")
    inputs = _mapping(diagnostics.get("inputs"), field=f"seed {seed} calibration.inputs")
    if inputs.get("gold_sha256") != contract["validation_sha256"]:
        raise CommunityRCEvidenceError(f"seed {seed} calibration is not bound to validation")
    fit_predictions_sha256 = _sha256(
        inputs.get("predictions_sha256"), field=f"seed {seed} calibration predictions"
    )
    if inputs.get("document_count") != expected_document_count:
        raise CommunityRCEvidenceError(f"seed {seed} calibration/evaluation counts differ")
    parameters = _mapping(
        diagnostics.get("parameters"), field=f"seed {seed} calibration.parameters"
    )
    expected_parameters = {
        "t0_recall_floor": contract["t0_floor"],
        "t1_recall_floor": contract["t1_floor"],
        "temperature_enabled": contract["temperature_enabled"],
    }
    for name, expected in expected_parameters.items():
        actual = parameters.get(name)
        if isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = isinstance(actual, (int, float)) and not isinstance(actual, bool)
            matches = matches and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        if not matches:
            raise CommunityRCEvidenceError(f"seed {seed} calibration parameter {name} differs")

    temperature = _mapping(
        diagnostics.get("temperature"), field=f"seed {seed} calibration.temperature"
    )
    expected_status = "fitted" if contract["temperature_enabled"] else "disabled"
    if (
        temperature.get("enabled") is not contract["temperature_enabled"]
        or temperature.get("status") != expected_status
    ):
        raise CommunityRCEvidenceError(f"seed {seed} calibration did not complete")

    per_label = _mapping(diagnostics.get("per_label"), field=f"seed {seed} calibration.per_label")
    if set(per_label) != expected_labels:
        raise CommunityRCEvidenceError(f"seed {seed} calibration/evaluation labels differ")
    risk_by_label: dict[str, str] = {}
    for label, raw in per_label.items():
        value = _mapping(raw, field=f"seed {seed} calibration.per_label.{label}")
        if value.get("entity_type") != label:
            raise CommunityRCEvidenceError(f"seed {seed} calibration label identity differs")
        risk_tier = value.get("risk_tier")
        if risk_tier not in {"T0", "T1", "T2"}:
            raise CommunityRCEvidenceError(f"seed {seed} calibration risk tier is invalid")
        _probability(value.get("threshold"), field=f"seed {seed} {label} threshold")
        recall = _probability(value.get("recall"), field=f"seed {seed} {label} calibration recall")
        expected_floor = (
            contract["t0_floor"]
            if risk_tier == "T0"
            else contract["t1_floor"]
            if risk_tier == "T1"
            else 0.0
        )
        recall_floor = _probability(
            value.get("recall_floor"), field=f"seed {seed} {label} calibration recall floor"
        )
        if not math.isclose(recall_floor, expected_floor, rel_tol=0.0, abs_tol=1e-12):
            raise CommunityRCEvidenceError(f"seed {seed} {label} recall floor differs")
        if value.get("recall_floor_met") is not True or recall < recall_floor:
            raise CommunityRCEvidenceError(
                f"seed {seed} {label} calibration recall floor was not met"
            )
        risk_by_label[label] = risk_tier
    if not any(value == "T0" for value in risk_by_label.values()) or not any(
        value == "T1" for value in risk_by_label.values()
    ):
        raise CommunityRCEvidenceError(f"seed {seed} calibration lacks T0 or T1 labels")
    return risk_by_label, {
        "manifest_sha256": logical_sha256,
        "file_sha256": file_sha256,
        "calibration_bundle_sha256": bundle_sha256,
        "calibration_version": diagnostics["calibration_version"],
        "fit_gold_sha256": inputs["gold_sha256"],
        "fit_predictions_sha256": fit_predictions_sha256,
    }


def _evaluation_metrics(
    report: Mapping[str, Any],
    *,
    seed: int,
    risk_by_label: Mapping[str, str],
) -> dict[str, float]:
    strict = _mapping(report.get("strict"), field=f"seed {seed} evaluation.strict")
    micro = _mapping(strict.get("micro"), field=f"seed {seed} evaluation.strict.micro")
    _, micro_f1 = _score_from_counts(micro, field=f"seed {seed} strict.micro")
    macro = _mapping(strict.get("macro"), field=f"seed {seed} evaluation.strict.macro")
    macro_f1 = _probability(macro.get("f1"), field=f"seed {seed} strict.macro.f1")
    per_class = _mapping(strict.get("per_class"), field=f"seed {seed} evaluation.strict.per_class")
    if set(per_class) != set(risk_by_label):
        raise CommunityRCEvidenceError(f"seed {seed} strict/calibration labels differ")
    recalls: dict[str, float] = {}
    class_f1: list[float] = []
    for label, raw in per_class.items():
        value = _mapping(raw, field=f"seed {seed} strict.per_class.{label}")
        recall, f1 = _score_from_counts(value, field=f"seed {seed} strict.per_class.{label}")
        recalls[label] = recall
        class_f1.append(f1)
    recomputed_macro_f1 = sum(class_f1) / len(class_f1)
    if not math.isclose(macro_f1, recomputed_macro_f1, rel_tol=0.0, abs_tol=1e-12):
        raise CommunityRCEvidenceError(f"seed {seed} strict macro F1 does not match per-label F1")

    pii_free = _mapping(report.get("pii_free"), field=f"seed {seed} evaluation.pii_free")
    pii_free_documents = _integer(
        pii_free.get("documents"), field=f"seed {seed} pii_free.documents", minimum=1
    )
    false_positive_documents = _integer(
        pii_free.get("false_positive_documents"),
        field=f"seed {seed} pii_free.false_positive_documents",
    )
    if false_positive_documents > pii_free_documents:
        raise CommunityRCEvidenceError(f"seed {seed} PII-free false positives exceed documents")
    fpr = _probability(
        pii_free.get("false_positive_rate"), field=f"seed {seed} pii_free.false_positive_rate"
    )
    expected_fpr = false_positive_documents / pii_free_documents
    if not math.isclose(fpr, expected_fpr, rel_tol=0.0, abs_tol=1e-12):
        raise CommunityRCEvidenceError(f"seed {seed} PII-free FPR has the wrong denominator")

    t0_recalls = [recalls[label] for label, tier in risk_by_label.items() if tier == "T0"]
    t1_recalls = [recalls[label] for label, tier in risk_by_label.items() if tier == "T1"]
    return {
        "strict_micro_f1": micro_f1,
        "strict_macro_f1": macro_f1,
        "pii_free_false_positive_rate": fpr,
        "tier0_min_label_recall": min(t0_recalls),
        "tier1_min_label_recall": min(t1_recalls),
        # release_gate requires a common, finite numeric metric set.  One is
        # the machine-readable encoding of successful calibration completion.
        "calibration_completed": 1.0,
    }


def _compare(value: object, *, operator: object, threshold: object, field: str) -> bool:
    if operator == "==":
        if isinstance(threshold, bool):
            if not isinstance(value, bool):
                raise CommunityRCEvidenceError(f"{field} must compare boolean values")
            return value is threshold
        left = _number(value, field=f"{field}.value")
        right = _number(threshold, field=f"{field}.threshold")
        return math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
    left = _number(value, field=f"{field}.value")
    right = _number(threshold, field=f"{field}.threshold")
    comparisons = {
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }
    if operator not in comparisons:
        raise CommunityRCEvidenceError(f"{field}.operator is unsupported")
    return comparisons[operator]


def _seed_criteria(
    metrics: Mapping[str, float], criteria: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    evidence: list[dict[str, Any]] = []
    for name in _METRIC_NAMES:
        raw = _mapping(criteria[name], field=f"per_seed_criteria.{name}")
        value: object = metrics[name]
        if name == "calibration_completed":
            value = metrics[name] == 1.0
        passed = _compare(
            value,
            operator=raw.get("operator"),
            threshold=raw.get("threshold"),
            field=f"per_seed_criteria.{name}",
        )
        evidence.append(
            {
                "name": name,
                "operator": raw.get("operator"),
                "threshold": raw.get("threshold"),
                "value": value,
                "passed": passed,
            }
        )
    return evidence, all(item["passed"] is True for item in evidence)


def _aggregate_criteria(
    criteria: Mapping[str, Any], *, unique_seed_count: int, every_seed_passes: bool
) -> tuple[list[dict[str, Any]], bool]:
    unique = _mapping(criteria["unique_seed_count"], field="aggregate_criteria.unique_seed_count")
    unique_passed = _compare(
        unique_seed_count,
        operator=unique.get("operator"),
        threshold=unique.get("threshold"),
        field="aggregate_criteria.unique_seed_count",
    )
    every = _mapping(criteria["every_seed_passes"], field="aggregate_criteria.every_seed_passes")
    every_passed = _compare(
        every_seed_passes,
        operator=every.get("operator"),
        threshold=every.get("threshold"),
        field="aggregate_criteria.every_seed_passes",
    )
    # gate_quality accepts numeric inequalities, so the preregistered boolean
    # equality is encoded as the equivalent all-passed indicator >= 1.
    result = [
        {
            "name": "unique_seed_count",
            "operator": unique.get("operator"),
            "threshold": unique.get("threshold"),
            "value": unique_seed_count,
            "passed": unique_passed,
        },
        {
            "name": "every_seed_passes",
            "operator": ">=",
            "threshold": 1.0,
            "value": 1.0 if every_seed_passes else 0.0,
            "passed": every_passed,
        },
    ]
    return result, unique_passed and every_passed


def _assert_path_free(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden_keys = {"doc_id", "document_id", "raw_text", "text", "entity_value"}
        if forbidden_keys & {str(key).lower() for key in value}:
            raise CommunityRCEvidenceError("output contains a forbidden record-level field")
        for item in value.values():
            _assert_path_free(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_path_free(item)
    elif isinstance(value, str):
        if value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATTERN.match(value):
            raise CommunityRCEvidenceError("output contains an absolute path")
        if ".." in value.split("/"):
            raise CommunityRCEvidenceError("output contains an unsafe relative locator")


def _write_json_atomic(value: Mapping[str, Any], destination: Path) -> None:
    if destination.is_symlink():
        raise CommunityRCEvidenceError("output must not be a symlink")
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
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def _remove_stale_output(destination: Path) -> None:
    """Ensure a failed rerun cannot leave an older PASS report in place."""

    if destination.is_symlink():
        raise CommunityRCEvidenceError("output must not be a symlink")
    if not destination.exists():
        return
    if not destination.is_file():
        raise CommunityRCEvidenceError("output must be a regular file")
    destination.unlink()


def _ensure_distinct_output(destination: Path, inputs: Sequence[Path]) -> None:
    if destination.is_symlink():
        raise CommunityRCEvidenceError("output must not be a symlink")
    try:
        destination_identity = destination.resolve(strict=False)
        input_identities = {path.resolve(strict=False) for path in inputs}
    except OSError as exc:
        raise CommunityRCEvidenceError("cannot resolve input/output identities") from exc
    if destination_identity in input_identities:
        raise CommunityRCEvidenceError("output must differ from every input artifact")
    if destination.exists():
        for path in inputs:
            try:
                if path.exists() and os.path.samefile(destination, path):
                    raise CommunityRCEvidenceError(
                        "output must not be a hard link to an input artifact"
                    )
            except OSError as exc:
                raise CommunityRCEvidenceError("cannot compare input/output identities") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize exactly seeds 13/42/97 from aggregate validation JSON. "
            "No gold, prediction JSONL, or frozen-test input is accepted."
        )
    )
    parser.add_argument("--decision-config", required=True, type=Path)
    parser.add_argument(
        "--seed-evidence",
        required=True,
        action="append",
        nargs=4,
        metavar=("SEED", "EVALUATION_JSON", "CALIBRATION_DIAGNOSTICS", "TRAINING_MANIFEST"),
        help="Repeat once for each preregistered seed",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        raw_inputs = [
            args.decision_config,
            *(Path(value) for evidence in args.seed_evidence for value in evidence[1:]),
        ]
        _ensure_distinct_output(args.output, raw_inputs)
        _remove_stale_output(args.output)
        config, config_file_sha256 = _load_yaml(args.decision_config)
        contract = _verify_decision_config(config)
        supplied: dict[int, tuple[Path, Path, Path]] = {}
        for raw_seed, evaluation, diagnostics, training in args.seed_evidence:
            try:
                seed = int(raw_seed)
            except ValueError as exc:
                raise CommunityRCEvidenceError("seed must be an integer") from exc
            if seed in supplied:
                raise CommunityRCEvidenceError(f"seed {seed} evidence was supplied twice")
            supplied[seed] = (Path(evaluation), Path(diagnostics), Path(training))
        if tuple(sorted(supplied)) != _REQUIRED_SEEDS:
            raise CommunityRCEvidenceError("evidence seeds must be exactly 13, 42, and 97")

        seed_reports: list[dict[str, Any]] = []
        evaluation_file_hashes: set[str] = set()
        training_manifest_hashes: set[str] = set()
        for seed in _REQUIRED_SEEDS:
            evaluation_path, diagnostics_path, training_path = supplied[seed]
            training, training_sha256, training_file_sha256 = _verify_training_manifest(
                training_path, seed=seed, contract=contract
            )
            training_manifest_hashes.add(training_sha256)
            evaluation, evaluation_file_sha256 = _load_json(
                evaluation_path, description=f"seed {seed} evaluation report"
            )
            evaluation_sha256, predictions_sha256, system_binding = _verify_evaluation_provenance(
                evaluation,
                seed=seed,
                training_manifest_sha256=training_sha256,
                training_manifest_file_sha256=training_file_sha256,
                contract=contract,
            )
            datasets = _mapping(training.get("datasets"), field=f"seed {seed} training.datasets")
            validation = _mapping(
                datasets.get("validation"), field=f"seed {seed} training validation"
            )
            validation_summary = _mapping(
                validation.get("summary"), field=f"seed {seed} training validation summary"
            )
            if validation_summary.get("document_count") != system_binding["document_count"]:
                raise CommunityRCEvidenceError(
                    f"seed {seed} training/evaluation document counts differ"
                )
            evaluation_file_hashes.add(evaluation_file_sha256)
            strict = _mapping(evaluation.get("strict"), field=f"seed {seed} evaluation.strict")
            per_class = _mapping(
                strict.get("per_class"), field=f"seed {seed} evaluation.strict.per_class"
            )
            risk_by_label, calibration_identity = _verify_calibration(
                diagnostics_path,
                seed=seed,
                expected_labels=set(per_class),
                expected_document_count=system_binding["document_count"],
                contract=contract,
            )
            expected_calibration_identity = {
                "diagnostics_manifest_sha256": calibration_identity["manifest_sha256"],
                "calibration_bundle_sha256": calibration_identity["calibration_bundle_sha256"],
                "calibration_version": calibration_identity["calibration_version"],
                "calibration_fit_predictions_sha256": calibration_identity[
                    "fit_predictions_sha256"
                ],
            }
            if any(
                system_binding[field] != value
                for field, value in expected_calibration_identity.items()
            ):
                raise CommunityRCEvidenceError(
                    f"seed {seed} system/calibration provenance is inconsistent"
                )
            metrics = _evaluation_metrics(evaluation, seed=seed, risk_by_label=risk_by_label)
            criteria, quality_gate_passed = _seed_criteria(metrics, contract["per_seed_criteria"])
            output_artifact = _mapping(
                training.get("output_artifact"), field=f"seed {seed} training.output_artifact"
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "criteria": criteria,
                    "quality_gate_passed": quality_gate_passed,
                    "provenance": {
                        "evaluation": {
                            "artifact_id": f"seed-{seed}/validation-evaluation",
                            "evaluation_sha256": evaluation_sha256,
                            "file_sha256": evaluation_file_sha256,
                            "predictions_sha256": predictions_sha256,
                        },
                        "calibration": {
                            "artifact_id": f"seed-{seed}/calibration-diagnostics",
                            **calibration_identity,
                        },
                        "training": {
                            "artifact_id": f"seed-{seed}/training-manifest",
                            "manifest_sha256": training_sha256,
                            "file_sha256": training_file_sha256,
                            "output_artifact_sha256": _sha256(
                                output_artifact.get("artifact_files_combined_sha256"),
                                field=f"seed {seed} output artifact",
                            ),
                        },
                    },
                }
            )
        if len(evaluation_file_hashes) != len(_REQUIRED_SEEDS):
            raise CommunityRCEvidenceError("evaluation report files must be distinct per seed")
        if len(training_manifest_hashes) != len(_REQUIRED_SEEDS):
            raise CommunityRCEvidenceError("training manifests must be distinct per seed")

        every_seed_passes = all(run["quality_gate_passed"] is True for run in seed_reports)
        aggregate, aggregate_passed = _aggregate_criteria(
            contract["aggregate_criteria"],
            unique_seed_count=len({run["seed"] for run in seed_reports}),
            every_seed_passes=every_seed_passes,
        )
        passed = every_seed_passes and aggregate_passed
        report = add_manifest_hash(
            {
                "schema_version": 1,
                "artifact_type": "community_rc_evaluation_report",
                "decision_id": contract["decision_id"],
                "decision_config": {
                    "artifact_id": f"decision-config/{contract['decision_id']}",
                    "file_sha256": config_file_sha256,
                    "parent_decision_id": contract["parent_decision_id"],
                    "parent_config_sha256": contract["parent_config_sha256"],
                },
                "dataset": {
                    "split": "validation",
                    "sha256": contract["validation_sha256"],
                    "frozen_test_access": "not_accessed_by_summarizer",
                },
                "system_contract": {
                    "ruleset_id": contract["ruleset_id"],
                    "rules_implementation_sha256": contract["rules_implementation_sha256"],
                    "fusion_id": contract["fusion_id"],
                    "fusion_implementation_sha256": contract["fusion_implementation_sha256"],
                    "refinement": contract["refinement_contract"],
                    "refinement_id": contract["refinement_id"],
                    "refinement_implementation_sha256": contract[
                        "refinement_implementation_sha256"
                    ],
                    "calibration_holdout_policy": "apply_only_no_refit",
                },
                "metric_set": list(_METRIC_NAMES),
                "seeds": seed_reports,
                "quality_gate": {
                    "status": "passed" if passed else "failed",
                    "criteria": aggregate,
                },
                "release_decision": "passed" if passed else "blocked",
                "release_scope": "community_research_release_candidate",
                "production_ready": False,
                "privacy": {
                    "contains_paths": False,
                    "contains_document_ids": False,
                    "contains_raw_text": False,
                    "contains_entity_values": False,
                    "contains_record_level_data": False,
                },
            }
        )
        _assert_path_free(report)
        _write_json_atomic(report, args.output)
    except (CommunityRCEvidenceError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "release_decision": report["release_decision"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["release_decision"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
