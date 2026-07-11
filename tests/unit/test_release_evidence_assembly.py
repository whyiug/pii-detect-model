from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts import assemble_release_evidence as assembly

from pii_zh.evaluation import canonical_json_hash
from pii_zh.taxonomy import load_taxonomy

EXPECTED_SEEDS = (13, 42, 97)
EXPECTED_SUITES = ("synthetic_test", "pii_bench_formal", "pii_bench_chat")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = canonical_json_hash(sealed)
    return sealed


def _community_report(*, validation_sha256: str) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": 1,
            "artifact_type": "community_rc_evaluation_report",
            "decision_id": "fixture-decision-v1",
            "release_scope": "community_research_release_candidate",
            "production_ready": False,
            "release_decision": "passed",
            "dataset": {"sha256": validation_sha256, "split": "validation"},
            "seeds": [
                {
                    "seed": seed,
                    "quality_gate_passed": True,
                    "metrics": {
                        "strict_macro_f1": 0.91 + index / 100,
                        "strict_micro_f1": 0.92 + index / 100,
                    },
                }
                for index, seed in enumerate(EXPECTED_SEEDS)
            ],
            "quality_gate": {
                "status": "passed",
                "criteria": [
                    {
                        "name": "fixture minimum strict micro F1",
                        "operator": ">=",
                        "passed": True,
                        "threshold": 0.90,
                        "value": 0.92,
                    }
                ],
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
                "contains_record_level_data": False,
            },
        }
    )


def _system_summary(
    *,
    training_manifest_sha256: str,
    training_file_sha256: str,
    calibration_file_sha256: str,
    calibration_version: str,
    diagnostics_manifest_sha256: str,
) -> dict[str, Any]:
    suites = {
        suite: {
            "metrics": {
                "document_count": count,
                "strict_micro": {
                    "precision": 0.94 - index / 100,
                    "recall": 0.93 - index / 100,
                    "f1": 0.935 - index / 100,
                },
            },
        }
        for index, (suite, count) in enumerate(
            zip(EXPECTED_SUITES, (2_000, 5_000, 3_000), strict=True)
        )
    }
    return _seal(
        {
            "schema_version": 1,
            "artifact_type": "frozen_system_evaluation_summary",
            "suite_order": list(EXPECTED_SUITES),
            "suites": suites,
            "aggregate": {
                "strict_micro_pooled": {
                    "precision": 0.92,
                    "recall": 0.91,
                    "f1": 0.915,
                }
            },
            "system_identity": {
                "selected_seed": 42,
                "attention_mode": "full",
                "training": {
                    "manifest_sha256": training_manifest_sha256,
                    "manifest_file_sha256": training_file_sha256,
                },
                "calibration_fit": {
                    "bundle_sha256": calibration_file_sha256,
                    "calibration_version": calibration_version,
                    "diagnostics_manifest_sha256": diagnostics_manifest_sha256,
                },
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_document_ids": False,
                "contains_record_level_data": False,
            },
        }
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    taxonomy_path = Path("src/pii_zh/taxonomy/taxonomy.yaml").resolve()
    taxonomy = load_taxonomy(taxonomy_path)
    core_labels = sorted(taxonomy.core_label_names)
    id2label = {"0": "O"}
    for entity in taxonomy.label_sets["core"]:
        id2label[str(len(id2label))] = f"B-{entity.name}"
        id2label[str(len(id2label))] = f"I-{entity.name}"

    checkpoint_dir = tmp_path / "checkpoint"
    _write_json(checkpoint_dir / "config.json", {"id2label": id2label})

    training = _seal(
        {
            "schema_version": 4,
            "status": "completed",
            "seed": 42,
            "attention_mode": "full",
            "privacy": {
                "contains_entity_values": False,
                "contains_raw_text": False,
                "trainer_reporting_integrations": [],
            },
        }
    )
    training_path = tmp_path / "training_manifest.json"
    _write_json(training_path, training)
    training_file_sha256 = assembly.sha256_file(training_path)

    calibration_version = "cal-fixture-validation-v1"
    calibration = {
        "schema_version": 1,
        "calibration_version": calibration_version,
        "global_temperature": 1.0,
        "default_threshold": 0.5,
        "entity_thresholds": {
            label: 0.50 + index / 1_000 for index, label in enumerate(reversed(core_labels))
        },
    }
    calibration_path = tmp_path / "calibration.json"
    _write_json(calibration_path, calibration)
    calibration_file_sha256 = assembly.sha256_file(calibration_path)

    validation_sha256 = "1" * 64
    diagnostics = _seal(
        {
            "schema_version": 1,
            "manifest_type": "calibration_diagnostics",
            "calibration_version": calibration_version,
            "calibration_bundle_sha256": calibration_file_sha256,
            "inputs": {"document_count": 2_000, "gold_sha256": validation_sha256},
        }
    )
    diagnostics_path = tmp_path / "calibration.diagnostics.json"
    _write_json(diagnostics_path, diagnostics)

    community = _community_report(validation_sha256=validation_sha256)
    community_path = tmp_path / "community_report.json"
    _write_json(community_path, community)

    system_summary = _system_summary(
        training_manifest_sha256=training["manifest_sha256"],
        training_file_sha256=training_file_sha256,
        calibration_file_sha256=calibration_file_sha256,
        calibration_version=calibration_version,
        diagnostics_manifest_sha256=diagnostics["manifest_sha256"],
    )
    system_summary_path = tmp_path / "system_summary.json"
    _write_json(system_summary_path, system_summary)

    provenance_binding = {
        "training_manifest_sha256": training["manifest_sha256"],
        "training_manifest_file_sha256": training_file_sha256,
    }
    data_provenance_path = tmp_path / "data_provenance.json"
    _write_json(
        data_provenance_path,
        _seal(
            {
                "schema_version": 1,
                "artifact_type": "data_provenance",
                "privacy": {
                    "contains_document_ids": False,
                    "contains_entity_values": False,
                    "contains_paths": False,
                    "contains_prompts_or_responses": False,
                    "contains_raw_text": False,
                },
                **provenance_binding,
            }
        ),
    )
    teacher_provenance_path = tmp_path / "teacher_provenance.json"
    _write_json(
        teacher_provenance_path,
        _seal(
            {
                "schema_version": 1,
                "artifact_type": "teacher_provenance",
                "privacy": {
                    "contains_document_ids": False,
                    "contains_entity_values": False,
                    "contains_paths": False,
                    "contains_prompts_or_responses": False,
                    "contains_raw_text": False,
                },
                **provenance_binding,
            }
        ),
    )

    output_dir = tmp_path / "assembled"
    return {
        "assemble_kwargs": {
            "checkpoint_dir": checkpoint_dir,
            "taxonomy_path": taxonomy_path,
            "training_manifest_path": training_path,
            "calibration_path": calibration_path,
            "calibration_diagnostics_path": diagnostics_path,
            "community_report_path": community_path,
            "system_summary_path": system_summary_path,
            "data_provenance_path": data_provenance_path,
            "teacher_provenance_path": teacher_provenance_path,
            "output_dir": output_dir,
            "release_name": "fixture-community-rc",
        },
        "calibration": calibration,
        "community": community,
        "core_labels": core_labels,
        "data_provenance_path": data_provenance_path,
        "output_dir": output_dir,
        "system_summary": system_summary,
        "teacher_provenance_path": teacher_provenance_path,
        "training": training,
        "training_file_sha256": training_file_sha256,
    }


def test_assemble_emits_gate_compatible_three_seed_report_and_deterministic_indexes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    first_hashes = assembly.assemble(**fixture["assemble_kwargs"])
    second_hashes = assembly.assemble(**fixture["assemble_kwargs"])

    assert first_hashes == second_hashes
    output_dir = fixture["output_dir"]
    report = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    unsigned = dict(report)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert tuple(run["seed"] for run in report["seeds"]) == EXPECTED_SEEDS
    assert report["quality_gate"] == fixture["community"]["quality_gate"]
    assert report["release_decision"] == "passed"
    assert "runs" not in report

    thresholds = yaml.safe_load((output_dir / "thresholds.yaml").read_text(encoding="utf-8"))
    assert thresholds["calibration_version"] == "cal-fixture-validation-v1"
    assert thresholds["default_threshold"] == 0.5
    assert thresholds["holdout_policy"] == "apply_only_no_refit"
    assert list(thresholds["entity_thresholds"]) == fixture["core_labels"]

    model_index = yaml.safe_load((output_dir / "model-index.yml").read_text(encoding="utf-8"))
    indexed = model_index["model-index"][0]
    assert indexed["name"] == "fixture-community-rc"
    assert [result["dataset"]["type"] for result in indexed["results"]] == list(EXPECTED_SUITES)
    assert all(
        [metric["name"] for metric in result["metrics"]]
        == [
            "System Strict Span Micro F1",
            "System Strict Span Precision",
            "System Strict Span Recall",
        ]
        for result in indexed["results"]
    )
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".json", ".yaml", ".yml"}
    )
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    "mutation, expected_message",
    [
        (
            lambda report: report["seeds"].__setitem__(2, {**report["seeds"][2], "seed": 42}),
            "frozen three seeds",
        ),
        (
            lambda report: report["seeds"][2]["metrics"].__setitem__("unexpected_metric", 0.5),
            "metric sets differ",
        ),
    ],
)
def test_combined_report_fails_closed_on_incompatible_three_seed_evidence(
    mutation: Any, expected_message: str
) -> None:
    community = _community_report(validation_sha256="1" * 64)
    mutation(community)

    with pytest.raises(assembly.ReleaseEvidenceError, match=expected_message):
        assembly.build_evaluation_report(
            community,
            {},
            community_file_sha256="2" * 64,
            system_summary_file_sha256="3" * 64,
        )


def test_combined_report_rejects_absolute_and_parent_traversal_paths() -> None:
    community = _community_report(validation_sha256="1" * 64)
    unsafe_summary = {"aggregate_source": "/private/evaluations/summary.json"}

    with pytest.raises(assembly.ReleaseEvidenceError, match="absolute path"):
        assembly.build_evaluation_report(
            community,
            unsafe_summary,
            community_file_sha256="2" * 64,
            system_summary_file_sha256="3" * 64,
        )

    unsafe_summary["aggregate_source"] = "reports/../private.json"
    with pytest.raises(assembly.ReleaseEvidenceError, match="unsafe relative path"):
        assembly.build_evaluation_report(
            community,
            unsafe_summary,
            community_file_sha256="2" * 64,
            system_summary_file_sha256="3" * 64,
        )


def test_assemble_rejects_self_hashed_unsafe_provenance_before_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    provenance = json.loads(fixture["data_provenance_path"].read_text(encoding="utf-8"))
    provenance["source_review"] = {
        "raw_text": "a record-level value that must never enter release evidence",
        "review_path": "/private/reviewer/worktree/source.jsonl",
    }
    _write_json(fixture["data_provenance_path"], _seal(provenance))

    with pytest.raises(assembly.ReleaseEvidenceError, match="forbidden|absolute path"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


@pytest.mark.parametrize(
    ("fixture_key", "extension"),
    [
        ("data_provenance_path", {"reviewer_note": "short private review detail"}),
        ("data_provenance_path", {"source_text": "short source excerpt"}),
        ("teacher_provenance_path", {"api_credential": "synthetic-credential-value"}),
    ],
)
def test_assemble_rejects_self_hashed_unapproved_extension_fields(
    tmp_path: Path,
    fixture_key: str,
    extension: dict[str, str],
) -> None:
    fixture = _fixture(tmp_path)
    provenance_path = fixture[fixture_key]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(extension)
    _write_json(provenance_path, _seal(provenance))

    with pytest.raises(assembly.ReleaseEvidenceError, match="unapproved field"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


def test_assemble_rejects_an_allowlisted_name_at_an_unapproved_object_path(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    provenance_path = fixture["teacher_provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    # source_id is valid only inside a reviewed teacher entry, never at the
    # artifact root where it could smuggle an unrelated private identifier.
    provenance["source_id"] = "synthetic-sensitive-marker"
    _write_json(provenance_path, _seal(provenance))

    with pytest.raises(assembly.ReleaseEvidenceError, match="unapproved field.*source_id"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


@pytest.mark.parametrize(
    "input_key",
    [
        "training_manifest_path",
        "community_report_path",
        "system_summary_path",
        "data_provenance_path",
        "teacher_provenance_path",
    ],
)
@pytest.mark.parametrize("remove_entire_declaration", [False, True])
def test_assemble_requires_complete_artifact_specific_privacy_declarations(
    tmp_path: Path,
    input_key: str,
    remove_entire_declaration: bool,
) -> None:
    fixture = _fixture(tmp_path)
    evidence_path = fixture["assemble_kwargs"][input_key]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if remove_entire_declaration:
        evidence.pop("privacy")
    else:
        evidence["privacy"].pop("contains_raw_text")
    _write_json(evidence_path, _seal(evidence))

    with pytest.raises(assembly.ReleaseEvidenceError, match="privacy"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entity_temperature", float("nan")),
        ("entity_temperature", float("inf")),
        ("entity_temperature", 0.0),
        ("entity_threshold", float("nan")),
        ("entity_threshold", float("inf")),
    ],
)
def test_assemble_rejects_nonfinite_or_nonpositive_calibration_values(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    fixture = _fixture(tmp_path)
    calibration_path = fixture["assemble_kwargs"]["calibration_path"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    label = fixture["core_labels"][0]
    if field == "entity_temperature":
        calibration["entity_temperatures"] = {label: value}
    else:
        calibration["entity_thresholds"][label] = value
    _write_json(calibration_path, calibration)

    with pytest.raises(assembly.ReleaseEvidenceError, match="finite|positive|between"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


def test_assemble_rejects_a_broken_input_self_hash_before_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    community_path = fixture["assemble_kwargs"]["community_report_path"]
    community = json.loads(community_path.read_text(encoding="utf-8"))
    community["release_decision"] = "tampered-after-sealing"
    _write_json(community_path, community)

    with pytest.raises(assembly.ReleaseEvidenceError, match="self-hash verification failed"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


def test_assemble_rejects_an_incomplete_checkpoint_bio_contract(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config_path = fixture["assemble_kwargs"]["checkpoint_dir"] / "config.json"
    incomplete = {"0": "O"}
    for entity in fixture["core_labels"]:
        incomplete[str(len(incomplete))] = f"B-{entity}"
    _write_json(config_path, {"id2label": incomplete})

    with pytest.raises(assembly.ReleaseEvidenceError, match="BIO|id2label|taxonomy"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


def test_assemble_rejects_duplicate_json_keys_before_copying_original_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    provenance_path = fixture["data_provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["review_path"] = "reviewed/source.json"
    serialized = json.dumps(_seal(provenance), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    safe_member = '  "review_path": "reviewed/source.json",'
    duplicate_members = (
        '  "review_path": "/private/reviewer/raw/source.jsonl",\n'
        '  "review_path": "reviewed/source.json",'
    )
    assert safe_member in serialized
    provenance_path.write_text(
        serialized.replace(safe_member, duplicate_members),
        encoding="utf-8",
    )
    parsed_last_wins = json.loads(provenance_path.read_text(encoding="utf-8"))
    unsigned = dict(parsed_last_wins)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert parsed_last_wins["review_path"] == "reviewed/source.json"

    with pytest.raises(assembly.ReleaseEvidenceError, match="duplicate.*review_path"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not fixture["output_dir"].exists() or not any(fixture["output_dir"].iterdir())


def test_assemble_rejects_a_symlinked_output_directory_ancestor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    escaped_output = linked_parent / "assembled"
    fixture["assemble_kwargs"]["output_dir"] = escaped_output

    with pytest.raises(assembly.ReleaseEvidenceError, match="symlink"):
        assembly.assemble(**fixture["assemble_kwargs"])

    assert not (real_parent / "assembled").exists()
