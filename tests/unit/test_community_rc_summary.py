from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from pii_zh.evaluation import canonical_json_hash, sha256_file
from pii_zh.fusion import system_fusion_configuration

_VALIDATION_SHA256 = "a" * 64
_SEEDS = (13, 42, 97)
_RULES_IMPLEMENTATION_SHA256 = hashlib.sha256(b"rules-v5").hexdigest()
_FUSION_MODULE_SHA256 = hashlib.sha256(b"fusion-module").hexdigest()
_FUSION_CLI_SHA256 = hashlib.sha256(b"fusion-cli").hexdigest()
_FUSION_IMPLEMENTATION_SHA256 = canonical_json_hash(
    {"module_sha256": _FUSION_MODULE_SHA256, "cli_sha256": _FUSION_CLI_SHA256}
)
_REFINEMENT_IMPLEMENTATION_SHA256 = hashlib.sha256(b"refinement-v4").hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["manifest_sha256"] = canonical_json_hash(result)
    return result


def _score(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f1,
    }


def _output_fingerprint(root: Path) -> dict[str, Any]:
    names = (
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    files = {name: sha256_file(root / name) for name in names}
    weights = {"model.safetensors": files["model.safetensors"]}
    return {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(files.items())),
        "weight_files": ["model.safetensors"],
        "weights_combined_sha256": canonical_json_hash(weights),
        "artifact_files_combined_sha256": canonical_json_hash(files),
    }


def _training_manifest(root: Path, seed: int) -> Path:
    root.mkdir()
    (root / "model.safetensors").write_bytes(f"safe-{seed}".encode())
    _write_json(root / "config.json", {"model_type": "qwen3_bi"})
    _write_json(root / "tokenizer.json", {"model": {"type": "WordLevel"}})
    _write_json(root / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(root / "special_tokens_map.json", {"unk_token": "[UNK]"})
    manifest = _self_hashed(
        {
            "schema_version": 4,
            "status": "completed",
            "seed": seed,
            "attention_mode": "full",
            "fine_tuning": "lora",
            "training_source_ids": [
                "qwen3_0_6b_base",
                "qwen3_8b_placeholder_template_generator",
                "repo_curated_synthetic_templates",
            ],
            "provenance_migration": {
                "migration_id": "training_source_lineage_v1",
                "migration_code_revision": "a" * 40,
                "previous_schema_version": 3,
                "previous_manifest_sha256": "b" * 64,
                "previous_manifest_file_sha256": "c" * 64,
                "reason": "record_reviewed_upstream_placeholder_template_generator",
                "model_weights_changed": False,
                "tokenizer_changed": False,
                "training_data_changed": False,
                "training_recipe_changed": False,
            },
            "recipe": {"seed": seed, "attention_mode": "full"},
            "initialization": {"source_attention_mode": "jpt"},
            "datasets": {
                "validation": {
                    "sha256": _VALIDATION_SHA256,
                    "summary": {"document_count": 2000},
                }
            },
            "output_artifact": _output_fingerprint(root),
            "privacy": {"contains_raw_text": False, "contains_entity_values": False},
        }
    )
    path = root / "training_manifest.json"
    _write_json(path, manifest)
    return path


def _calibration_diagnostics(path: Path, seed: int) -> Path:
    per_label = {
        "BANK_ACCOUNT_NUMBER": {
            "entity_type": "BANK_ACCOUNT_NUMBER",
            "risk_tier": "T0",
            "threshold": 0.5,
            "recall": 0.95,
            "recall_floor": 0.90,
            "recall_floor_met": True,
        },
        "ADDRESS": {
            "entity_type": "ADDRESS",
            "risk_tier": "T1",
            "threshold": 0.4,
            "recall": 0.95,
            "recall_floor": 0.85,
            "recall_floor_met": True,
        },
    }
    diagnostics = _self_hashed(
        {
            "schema_version": 1,
            "manifest_type": "calibration_diagnostics",
            "calibration_version": f"cal-seed-{seed}",
            "calibration_bundle_sha256": hashlib.sha256(f"bundle-{seed}".encode()).hexdigest(),
            "inputs": {
                "gold_sha256": _VALIDATION_SHA256,
                "predictions_sha256": hashlib.sha256(f"candidate-{seed}".encode()).hexdigest(),
                "document_count": 2000,
            },
            "parameters": {
                "t0_recall_floor": 0.90,
                "t1_recall_floor": 0.85,
                "temperature_enabled": True,
            },
            "temperature": {"enabled": True, "status": "fitted"},
            "per_label": per_label,
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
            },
        }
    )
    _write_json(path, diagnostics)
    return path


def _evaluation_report(path: Path, seed: int, training_path: Path, diagnostics_path: Path) -> Path:
    training = json.loads(training_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    t0 = _score(95, 0, 5)
    t1 = _score(95, 0, 5)
    micro = _score(190, 0, 10)
    prediction_sha256 = hashlib.sha256(f"final-predictions-{seed}".encode()).hexdigest()
    fused_sha256 = diagnostics["inputs"]["predictions_sha256"]
    calibrated_sha256 = hashlib.sha256(f"calibrated-{seed}".encode()).hexdigest()
    provenance = {
        "schema_version": 1,
        "mode": "development",
        "gold": {"sha256": _VALIDATION_SHA256, "dataset": None},
        "predictions": {
            "sha256": prediction_sha256,
            "manifest": {
                "predictor_type": "system",
                "predictions_sha256": prediction_sha256,
                "manifest_sha256": hashlib.sha256(f"system-{seed}".encode()).hexdigest(),
                "manifest_file_sha256": hashlib.sha256(f"system-file-{seed}".encode()).hexdigest(),
                "dataset_manifest_sha256": hashlib.sha256(b"dataset-manifest").hexdigest(),
                "model_training_manifest_sha256": training["manifest_sha256"],
                "seed": seed,
                "attention_mode": "full",
                "prediction_document_count": 2000,
                "components": {
                    "model_prediction": {
                        "model_training_manifest_sha256": training["manifest_sha256"],
                        "prediction_document_count": 2000,
                    },
                    "rules_prediction": {
                        "prediction_document_count": 2000,
                        "rules_identity": {
                            "ruleset_id": "cn_common_v5",
                            "implementation_sha256": _RULES_IMPLEMENTATION_SHA256,
                            "configuration_sha256": hashlib.sha256(
                                b"rules-configuration"
                            ).hexdigest(),
                        },
                    },
                    "fusion": {
                        "module_sha256": _FUSION_MODULE_SHA256,
                        "cli_sha256": _FUSION_CLI_SHA256,
                        "implementation_sha256": _FUSION_IMPLEMENTATION_SHA256,
                        "configuration_sha256": canonical_json_hash(system_fusion_configuration()),
                        "output_predictions_sha256": fused_sha256,
                        "prediction_document_count": 2000,
                    },
                    "calibration_fit": {
                        "diagnostics_manifest_sha256": diagnostics["manifest_sha256"],
                        "bundle_sha256": diagnostics["calibration_bundle_sha256"],
                        "calibration_version": diagnostics["calibration_version"],
                        "gold_sha256": _VALIDATION_SHA256,
                        "fused_predictions_sha256": fused_sha256,
                        "document_count": 2000,
                    },
                    "target_application": {
                        "bundle_sha256": diagnostics["calibration_bundle_sha256"],
                        "calibration_version": diagnostics["calibration_version"],
                        "input_predictions_sha256": fused_sha256,
                        "output_predictions_sha256": calibrated_sha256,
                        "prediction_document_count": 2000,
                    },
                    "refinement": {
                        "audit_manifest_sha256": hashlib.sha256(
                            f"refinement-audit-{seed}".encode()
                        ).hexdigest(),
                        "refinement_id": "structured_refinement_v4",
                        "implementation_sha256": _REFINEMENT_IMPLEMENTATION_SHA256,
                        "input_predictions_sha256": calibrated_sha256,
                        "output_predictions_sha256": prediction_sha256,
                        "prediction_document_count": 2000,
                    },
                },
            },
        },
        "model_training": {
            "manifest_sha256": training["manifest_sha256"],
            "manifest_file_sha256": sha256_file(training_path),
            "seed": seed,
            "attention_mode": "full",
        },
        "parameters": {"bootstrap_samples": 1000},
        "privacy": {
            "contains_paths": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
        },
    }
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    report = {
        "document_count": 2000,
        "strict": {
            "micro": micro,
            "macro": {"label_count": 2, "f1": (t0["f1"] + t1["f1"]) / 2},
            "per_class": {"BANK_ACCOUNT_NUMBER": t0, "ADDRESS": t1},
        },
        "pii_free": {
            "documents": 600,
            "false_positive_documents": 0,
            "false_positive_rate": 0.0,
        },
        "provenance": provenance,
    }
    _write_json(path, report)
    return path


def _decision_config(path: Path) -> Path:
    value = {
        "schema_version": 1,
        "decision_id": "synthetic_v1_3_community_rc_v6",
        "status": "amended_before_holdout_access",
        "candidate_scope": "community_research_release_candidate",
        "production_ready": False,
        "parent_decision": {
            "decision_id": "synthetic_v1_3_community_rc_v5",
            "config_sha256": "c" * 64,
            "code_revision": "d" * 40,
        },
        "amendment": {
            "holdout_accessed": False,
            "thresholds_changed": False,
            "quality_gate_thresholds_changed": False,
            "seeds_changed": False,
            "selected_seed_changed": False,
            "model_weights_changed": False,
            "calibration_bundle_changed": False,
            "training_manifest_schema_changed": True,
            "training_source_lineage_changed": True,
            "allowed_changes": [
                "migrate_full_training_manifests_to_schema4",
                "record_reviewed_upstream_template_generator",
                "regenerate_prediction_system_and_evaluation_provenance",
            ],
            "forbidden_changes": [
                "lower_quality_thresholds",
                "change_seed_set",
                "change_selected_seed",
                "change_model_weights",
                "change_training_data_or_recipe",
                "change_rules_fusion_or_refinement_behavior",
                "change_calibration_bundle",
                "inspect_or_tune_on_frozen_test",
                "refit_calibration_on_any_test_subset",
            ],
        },
        "dataset": {
            "validation_sha256": _VALIDATION_SHA256,
            "frozen_test_sha256": "b" * 64,
            "frozen_test_access_before_validation_gate": "forbidden",
        },
        "system": {
            "attention_mode": "full",
            "initializer_attention_mode": "jpt",
            "seeds": [13, 42, 97],
            "selected_seed": 42,
            "selection_rule": "fixed_conventional_seed_before_holdout_access",
            "training_lineage": {
                "manifest_schema_version": 4,
                "required_training_source_ids": [
                    "qwen3_0_6b_base",
                    "qwen3_8b_placeholder_template_generator",
                    "repo_curated_synthetic_templates",
                ],
                "source_registry_sha256": "7" * 64,
                "migration_id": "training_source_lineage_v1",
                "migration_implementation_sha256": "8" * 64,
            },
            "ruleset_id": "cn_common_v5",
            "rules_implementation_sha256": _RULES_IMPLEMENTATION_SHA256,
            "fusion": "deterministic_fusion_v1",
            "fusion_implementation_sha256": _FUSION_IMPLEMENTATION_SHA256,
            "runtime": {
                "transformers_version": "5.13.1",
                "tokenizer_loader": "serialized_fast_tokenizer_v1",
                "tokenizer_boundary_mode": "unicode_codepoint_v1",
                "tokenizer_backend_sha256": "4" * 64,
                "tokenizer_implementation_sha256": "5" * 64,
                "dependency_lock_sha256": "6" * 64,
            },
            "calibration": {
                "fit_per_seed_on_validation": True,
                "selected_seed_bundle_frozen_before_holdout_access": True,
                "holdout_policy": "apply_only_no_refit",
                "refit_reason": "release_runtime_compatibility_migration",
                "t0_recall_floor": 0.90,
                "t1_recall_floor": 0.85,
                "temperature_scaling": True,
            },
            "refinement": "structured_prediction_refinement_v4",
            "refinement_implementation_sha256": _REFINEMENT_IMPLEMENTATION_SHA256,
        },
        "per_seed_criteria": {
            "strict_micro_f1": {"operator": ">=", "threshold": 0.94},
            "strict_macro_f1": {"operator": ">=", "threshold": 0.94},
            "pii_free_false_positive_rate": {"operator": "<=", "threshold": 0.02},
            "tier0_min_label_recall": {"operator": ">=", "threshold": 0.90},
            "tier1_min_label_recall": {"operator": ">=", "threshold": 0.85},
            "calibration_completed": {"operator": "==", "threshold": True},
        },
        "aggregate_criteria": {
            "unique_seed_count": {"operator": ">=", "threshold": 3},
            "every_seed_passes": {"operator": "==", "threshold": True},
        },
        "holdout_unlock": {
            "requires_release_decision": "passed",
            "required_seed_evidence": [13, 42, 97],
            "calibration_refit_on_holdout": "forbidden",
            "post_unlock_changes_to_model_or_system": "forbidden",
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, list[tuple[int, Path, Path, Path]]]:
    config = _decision_config(tmp_path / "decision.yaml")
    evidence = []
    for seed in _SEEDS:
        training = _training_manifest(tmp_path / f"checkpoint-{seed}", seed)
        diagnostics = _calibration_diagnostics(tmp_path / f"diagnostics-{seed}.json", seed)
        evaluation = _evaluation_report(
            tmp_path / f"evaluation-{seed}.json", seed, training, diagnostics
        )
        evidence.append((seed, evaluation, diagnostics, training))
    return config, evidence


def _run(
    repository_root: Path,
    config: Path,
    evidence: list[tuple[int, Path, Path, Path]],
    output: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(repository_root / "scripts/summarize_community_rc.py"),
        "--decision-config",
        str(config),
        "--output",
        str(output),
    ]
    for seed, evaluation, diagnostics, training in evidence:
        command.extend(
            [
                "--seed-evidence",
                str(seed),
                str(evaluation),
                str(diagnostics),
                str(training),
            ]
        )
    return subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _rewrite_evaluation_provenance(path: Path, evaluation: dict[str, Any]) -> None:
    provenance = evaluation["provenance"]
    provenance.pop("evaluation_sha256")
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    _write_json(path, evaluation)


def test_summary_is_path_free_self_hashed_and_release_gate_compatible(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(report)
    manifest_sha256 = unsigned.pop("manifest_sha256")
    assert manifest_sha256 == canonical_json_hash(unsigned)
    assert report["release_decision"] == "passed"
    assert report["quality_gate"]["status"] == "passed"
    assert [run["seed"] for run in report["seeds"]] == [13, 42, 97]
    metric_sets = {tuple(sorted(run["metrics"])) for run in report["seeds"]}
    assert len(metric_sets) == 1
    assert all(run["quality_gate_passed"] is True for run in report["seeds"])
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert '"doc_id"' not in serialized
    assert output.stat().st_mode & 0o777 == 0o444

    artifact = tmp_path / "release-artifact"
    artifact.mkdir()
    shutil.copyfile(output, artifact / "evaluation_report.json")
    shutil.copyfile(evidence[1][3], artifact / "training_manifest.json")
    (artifact / "model-index.yml").write_text(
        "model-index:\n"
        "  - name: community-rc-fixture\n"
        "    results:\n"
        "      - metrics:\n"
        "          - {name: Strict Micro F1, type: strict_micro_f1, value: 0.97}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(repository_root / "scripts"))
    release_gate = importlib.import_module("release_gate")
    gate = release_gate.gate_quality(artifact)
    assert gate.passed is True, gate.issues


def test_summary_fails_closed_without_training_provenance_binding(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    evaluation_path = evidence[0][1]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    provenance = evaluation["provenance"]
    provenance["model_training"] = None
    provenance.pop("evaluation_sha256")
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    _write_json(evaluation_path, evaluation)
    output = tmp_path / "evaluation_report.json"
    output.write_text('{"release_decision":"passed"}\n', encoding="utf-8")

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "--model-training-manifest" in completed.stderr
    assert "frozen" not in completed.stdout.lower()


def test_summary_rejects_model_only_final_evaluation(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    evaluation_path = evidence[0][1]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    provenance = evaluation["provenance"]
    provenance["predictions"]["manifest"]["predictor_type"] = "model"
    provenance.pop("evaluation_sha256")
    provenance["evaluation_sha256"] = canonical_json_hash(provenance)
    _write_json(evaluation_path, evaluation)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "predictor_type=system" in completed.stderr


def test_summary_rejects_system_manifest_without_rules_v5(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    evaluation_path = evidence[0][1]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    components = evaluation["provenance"]["predictions"]["manifest"]["components"]
    components["rules_prediction"]["rules_identity"]["ruleset_id"] = "cn_common_v4"
    _rewrite_evaluation_provenance(evaluation_path, evaluation)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "does not use cn_common_v5" in completed.stderr


def test_summary_rejects_system_manifest_without_refinement_v4(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    evaluation_path = evidence[0][1]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    components = evaluation["provenance"]["predictions"]["manifest"]["components"]
    components["refinement"]["refinement_id"] = "structured_refinement_v3"
    _rewrite_evaluation_provenance(evaluation_path, evaluation)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "does not use refinement v4" in completed.stderr


def test_summary_writes_blocked_report_when_a_preregistered_metric_fails(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    evaluation_path = evidence[2][1]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    t0 = evaluation["strict"]["per_class"]["BANK_ACCOUNT_NUMBER"]
    t1 = _score(80, 0, 20)
    evaluation["strict"]["per_class"]["ADDRESS"] = t1
    evaluation["strict"]["micro"] = _score(175, 0, 25)
    evaluation["strict"]["macro"]["f1"] = (t0["f1"] + t1["f1"]) / 2
    _write_json(evaluation_path, evaluation)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 1, completed.stdout + completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["release_decision"] == "blocked"
    assert report["quality_gate"]["status"] == "failed"
    seed97 = next(run for run in report["seeds"] if run["seed"] == 97)
    assert seed97["quality_gate_passed"] is False
    assert seed97["metrics"]["tier1_min_label_recall"] == 0.8


def test_summary_rejects_tampered_training_manifest(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    training_path = evidence[1][3]
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["status"] = "tampered"
    _write_json(training_path, training)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "self-hash does not verify" in completed.stderr


def test_summary_rejects_calibration_that_missed_a_risk_tier_floor(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    diagnostics_path = evidence[2][2]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    selection = diagnostics["per_label"]["BANK_ACCOUNT_NUMBER"]
    selection["recall"] = 0.70
    selection["recall_floor_met"] = False
    diagnostics.pop("manifest_sha256")
    diagnostics = _self_hashed(diagnostics)
    _write_json(diagnostics_path, diagnostics)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "calibration recall floor was not met" in completed.stderr


def test_summary_rejects_system_manifest_bound_to_different_calibration(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    diagnostics_path = evidence[1][2]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["taxonomy_version"] = "fixture-v2"
    diagnostics.pop("manifest_sha256")
    diagnostics = _self_hashed(diagnostics)
    _write_json(diagnostics_path, diagnostics)
    output = tmp_path / "evaluation_report.json"

    completed = _run(repository_root, config, evidence, output)

    assert completed.returncode == 2
    assert not output.exists()
    assert "system/calibration provenance is inconsistent" in completed.stderr


def test_summary_never_overwrites_an_input_artifact(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config, evidence = _fixture(tmp_path)
    training_path = evidence[0][3]
    before = training_path.read_bytes()

    completed = _run(repository_root, config, evidence, training_path)

    assert completed.returncode == 2
    assert training_path.read_bytes() == before
    assert "output must differ from every input artifact" in completed.stderr
