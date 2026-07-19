from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.cascade.ablation_runtime import (
    EXECUTION_RUNTIME_LOCK_SCHEMA_VERSION,
    EXECUTION_SOURCE_CLOSURE_SCHEMA_VERSION,
    NER_ADMISSION_RECEIPT_SCHEMA_VERSION,
    PROJECT_CANDIDATE_FREEZE_SCHEMA_VERSION,
    PROJECT_RELEASE_GATE_RECEIPT_SCHEMA_VERSION,
    PROJECT_SELECTION_RECEIPT_SCHEMA_VERSION,
    ClosedBioMapping,
    NerArtifactIdentity,
    ProjectModelIdentity,
    _project_label_contract,
    verify_execution_evidence,
    verify_loaded_project_predictor,
    verify_ner_admission_receipt,
    verify_project_model_for_ablation,
)
from pii_zh.evaluation.provenance import canonical_json_hash, sha256_file


def _write_receipt(path: Path, document: dict[str, object]) -> tuple[str, str]:
    logical_sha256 = canonical_json_hash(document)
    path.write_text(
        json.dumps(
            {**document, "receipt_sha256": logical_sha256},
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return logical_sha256, sha256_file(path)


def _project_model(
    root: Path,
    *,
    attention_mode: str = "full",
    release_eligible: bool = True,
) -> Path:
    root.mkdir()
    label2id, id2label, _, _ = _project_label_contract()
    output_artifact = {
        "artifact_files_combined_sha256": "a" * 64,
    }
    manifest: dict[str, object] = {
        "status": "completed",
        "attention_mode": attention_mode,
        "release_eligible": release_eligible,
        "label2id": label2id,
        "label_schema_sha256": canonical_json_hash(label2id),
        "output_artifact": output_artifact,
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    (root / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "label2id": label2id,
                "id2label": {str(index): label for index, label in id2label.items()},
                "pii_attention_mode": attention_mode,
                "pii_release_eligible": release_eligible,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _mock_completed_model_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    import pii_zh.evaluation.tw_pii_posthoc as module

    def verifier(model_root: Path) -> dict[str, object]:
        manifest = json.loads(
            (model_root / "training_manifest.json").read_text(encoding="utf-8")
        )
        return {
            "attention_mode": manifest["attention_mode"],
            "training_manifest_file_sha256": sha256_file(
                model_root / "training_manifest.json"
            ),
            "training_manifest_sha256": manifest["manifest_sha256"],
            "output_artifact_sha256": canonical_json_hash(manifest["output_artifact"]),
            "label_schema_sha256": manifest["label_schema_sha256"],
        }

    monkeypatch.setattr(module, "verify_completed_qwen_model", verifier)


def _project_receipts(
    tmp_path: Path,
    identity: ProjectModelIdentity,
) -> tuple[Path, Path, Path]:
    model = identity
    selection_path = tmp_path / "selection.json"
    selection: dict[str, object] = {
        "schema_version": PROJECT_SELECTION_RECEIPT_SCHEMA_VERSION,
        "report_type": "pii_zh_cascade_checkpoint_selection",
        "selection_id": "selection-v1",
        "status": "selected_release_checkpoint",
        "selected_checkpoint_id": "candidate-v7",
        "training_manifest_file_sha256": model.training_manifest_file_sha256,
        "training_manifest_sha256": model.training_manifest_sha256,
        "artifact_files_combined_sha256": model.artifact_files_combined_sha256,
        "label_schema_sha256": model.label_schema_sha256,
        "taxonomy_file_sha256": model.taxonomy_file_sha256,
        "taxonomy_version": model.taxonomy_version,
        "attention_mode": "full",
        "release_eligible": True,
        "all_selection_gates_passed": True,
    }
    selection_sha256, selection_file_sha256 = _write_receipt(
        selection_path, selection
    )

    release_path = tmp_path / "release-gate.json"
    release_gate: dict[str, object] = {
        "schema_version": PROJECT_RELEASE_GATE_RECEIPT_SCHEMA_VERSION,
        "report_type": "pii_zh_cascade_release_gate",
        "release_gate_id": "release-gate-v1",
        "status": "PASS",
        "selection_id": "selection-v1",
        "selected_checkpoint_id": "candidate-v7",
        "selection_file_sha256": selection_file_sha256,
        "selection_sha256": selection_sha256,
        "training_manifest_file_sha256": model.training_manifest_file_sha256,
        "training_manifest_sha256": model.training_manifest_sha256,
        "artifact_files_combined_sha256": model.artifact_files_combined_sha256,
        "label_schema_sha256": model.label_schema_sha256,
        "taxonomy_file_sha256": model.taxonomy_file_sha256,
        "taxonomy_version": model.taxonomy_version,
        "attention_mode": "full",
        "release_eligible": True,
        "all_release_gates_passed": True,
    }
    release_sha256, release_file_sha256 = _write_receipt(release_path, release_gate)

    freeze_path = tmp_path / "freeze.json"
    freeze: dict[str, object] = {
        "schema_version": PROJECT_CANDIDATE_FREEZE_SCHEMA_VERSION,
        "report_type": "pii_zh_cascade_candidate_freeze",
        "candidate_freeze_id": "candidate-freeze-v1",
        "selection_id": "selection-v1",
        "release_gate_id": "release-gate-v1",
        "selected_checkpoint_id": "candidate-v7",
        "status": "frozen_release_candidate",
        "selection_file_sha256": selection_file_sha256,
        "selection_sha256": selection_sha256,
        "release_gate_file_sha256": release_file_sha256,
        "release_gate_sha256": release_sha256,
        "training_manifest_file_sha256": model.training_manifest_file_sha256,
        "training_manifest_sha256": model.training_manifest_sha256,
        "artifact_files_combined_sha256": model.artifact_files_combined_sha256,
        "label_schema_sha256": model.label_schema_sha256,
        "taxonomy_file_sha256": model.taxonomy_file_sha256,
        "taxonomy_version": model.taxonomy_version,
        "attention_mode": "full",
        "release_eligible": True,
        "all_release_gates_passed": True,
        "frozen_before_hidden_access": True,
        "hidden_access_count_at_freeze": 0,
    }
    _write_receipt(freeze_path, freeze)
    return freeze_path, selection_path, release_path


def _development_model_identity(model_root: Path) -> ProjectModelIdentity:
    return verify_project_model_for_ablation(
        model_root,
        run_intent="development-non-claim",
        candidate_freeze_path=None,
        candidate_selection_path=None,
        release_gate_path=None,
    )


def test_confirmatory_project_model_requires_and_cross_binds_three_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_completed_model_verifier(monkeypatch)
    model_root = _project_model(tmp_path / "model")
    development = _development_model_identity(model_root)
    freeze, selection, release = _project_receipts(tmp_path, development)

    identity = verify_project_model_for_ablation(
        model_root,
        run_intent="frozen-confirmatory",
        candidate_freeze_path=freeze,
        candidate_selection_path=selection,
        release_gate_path=release,
    )

    assert identity.attention_mode == "full"
    assert identity.release_eligible is True
    assert identity.selected_checkpoint_id == "candidate-v7"
    assert identity.selection_id == "selection-v1"
    assert identity.release_gate_id == "release-gate-v1"
    assert len(identity.selection_file_sha256 or "") == 64
    assert len(identity.release_gate_sha256 or "") == 64


def test_confirmatory_project_model_rejects_release_gate_fail_or_receipt_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_completed_model_verifier(monkeypatch)
    model_root = _project_model(tmp_path / "model")
    development = _development_model_identity(model_root)
    freeze, selection, release = _project_receipts(tmp_path, development)
    document = json.loads(release.read_text(encoding="utf-8"))
    document.pop("receipt_sha256")
    document["status"] = "FAIL"
    _write_receipt(release, document)

    with pytest.raises(ValueError, match="release-gate PASS"):
        verify_project_model_for_ablation(
            model_root,
            run_intent="frozen-confirmatory",
            candidate_freeze_path=freeze,
            candidate_selection_path=selection,
            release_gate_path=release,
        )


def test_project_model_rejects_non_exact_24_class_config_and_loaded_predictor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_completed_model_verifier(monkeypatch)
    model_root = _project_model(tmp_path / "model")
    identity = _development_model_identity(model_root)
    good_predictor = SimpleNamespace(
        id2label=dict(identity.expected_id2label),
        attention_mode="full",
    )
    pipeline = SimpleNamespace(
        model_recognizer=SimpleNamespace(predictor=good_predictor)
    )
    verify_loaded_project_predictor(pipeline, identity)  # type: ignore[arg-type]
    good_predictor.id2label[48] = "I-NOT_CANONICAL"
    with pytest.raises(ValueError, match="differs"):
        verify_loaded_project_predictor(pipeline, identity)  # type: ignore[arg-type]

    config_path = model_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"]["48"] = "I-NOT_CANONICAL"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="24-class"):
        _development_model_identity(model_root)


def test_confirmatory_project_model_rejects_non_full_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_completed_model_verifier(monkeypatch)
    model_root = _project_model(
        tmp_path / "model",
        attention_mode="jpt",
        release_eligible=False,
    )
    development = _development_model_identity(model_root)
    freeze, selection, release = _project_receipts(tmp_path, development)

    with pytest.raises(ValueError, match="checkpoint selection"):
        verify_project_model_for_ablation(
            model_root,
            run_intent="frozen-confirmatory",
            candidate_freeze_path=freeze,
            candidate_selection_path=selection,
            release_gate_path=release,
        )


def _ner_identities() -> tuple[ClosedBioMapping, NerArtifactIdentity]:
    mapping = ClosedBioMapping(
        mapping_id="person-only-v1",
        source_to_target={"PER": "PERSON"},
        config_file_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    artifact = NerArtifactIdentity(
        artifact_id="ner-v1",
        binding_file_sha256="3" * 64,
        binding_sha256="4" * 64,
        files_combined_sha256="5" * 64,
        mapping_config_sha256=mapping.config_sha256,
    )
    return mapping, artifact


def test_ner_confirmatory_admission_binds_public_revision_license_and_artifact(
    tmp_path: Path,
) -> None:
    mapping, artifact = _ner_identities()
    receipt_path = tmp_path / "ner-admission.json"
    _write_receipt(
        receipt_path,
        {
            "schema_version": NER_ADMISSION_RECEIPT_SCHEMA_VERSION,
            "report_type": "pii_zh_ner_comparator_admission",
            "admission_id": "openmed-admission-v1",
            "status": "approved_for_confirmatory_comparison",
            "model_id": "OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1",
            "immutable_revision": "a" * 40,
            "license_id": "Apache-2.0",
            "license_review_status": "approved",
            "redistribution_status": "byo_not_redistributed",
            "artifact_files_combined_sha256": artifact.files_combined_sha256,
            "mapping_config_sha256": mapping.config_sha256,
        },
    )

    identity = verify_ner_admission_receipt(
        receipt_path,
        run_intent="frozen-confirmatory",
        artifact=artifact,
        mapping=mapping,
    )

    assert identity.model_id.startswith("OpenMed/")
    assert identity.license_review_status == "approved"
    assert identity.to_report()["confirmatory_admitted"] is True

    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    document.pop("receipt_sha256")
    document["license_review_status"] = "pending"
    _write_receipt(receipt_path, document)
    with pytest.raises(ValueError, match="provenance binding"):
        verify_ner_admission_receipt(
            receipt_path,
            run_intent="frozen-confirmatory",
            artifact=artifact,
            mapping=mapping,
        )


def test_execution_evidence_is_optional_only_for_explicit_non_claim_mode(
    tmp_path: Path,
) -> None:
    runtime_versions = {
        "python": "3.12.0",
        "pii-zh-qwen": "0.1.0",
        "torch": "2.6.0+cu124",
        "transformers": "4.51.0",
        "presidio-analyzer": "2.2.363",
    }
    development = verify_execution_evidence(
        run_intent="development-non-claim",
        ablation_id="rules_only",
        profile_set_file_sha256="a" * 64,
        implementation_sha256="b" * 64,
        runtime_input_identity_sha256="c" * 64,
        runtime_versions=runtime_versions,
        source_closure_path=None,
        runtime_lock_path=None,
    )
    assert development.to_report()["confirmatory_execution_evidence_verified"] is False

    closure_path = tmp_path / "closure.json"
    closure_sha256, _ = _write_receipt(
        closure_path,
        {
            "schema_version": EXECUTION_SOURCE_CLOSURE_SCHEMA_VERSION,
            "report_type": "pii_zh_cascade_source_closure",
            "closure_id": "rules-closure-v1",
            "status": "frozen_before_hidden_access",
            "ablation_id": "rules_only",
            "profile_set_file_sha256": "a" * 64,
            "implementation_sha256": "b" * 64,
            "runtime_input_identity_sha256": "c" * 64,
            "frozen_before_hidden_access": True,
            "hidden_access_count_at_freeze": 0,
        },
    )
    runtime_path = tmp_path / "runtime.json"
    _write_receipt(
        runtime_path,
        {
            "schema_version": EXECUTION_RUNTIME_LOCK_SCHEMA_VERSION,
            "report_type": "pii_zh_cascade_runtime_lock",
            "runtime_lock_id": "rules-runtime-v1",
            "status": "frozen",
            "source_closure_sha256": closure_sha256,
            "runtime_versions": runtime_versions,
        },
    )

    frozen = verify_execution_evidence(
        run_intent="frozen-confirmatory",
        ablation_id="rules_only",
        profile_set_file_sha256="a" * 64,
        implementation_sha256="b" * 64,
        runtime_input_identity_sha256="c" * 64,
        runtime_versions=runtime_versions,
        source_closure_path=closure_path,
        runtime_lock_path=runtime_path,
    )
    assert frozen.source_closure_status == "VERIFIED"
    assert frozen.runtime_lock_status == "VERIFIED"

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.pop("receipt_sha256")
    runtime["runtime_versions"]["torch"] = "0.0.0"
    _write_receipt(runtime_path, runtime)
    with pytest.raises(ValueError, match="runtime lock"):
        verify_execution_evidence(
            run_intent="frozen-confirmatory",
            ablation_id="rules_only",
            profile_set_file_sha256="a" * 64,
            implementation_sha256="b" * 64,
            runtime_input_identity_sha256="c" * 64,
            runtime_versions=runtime_versions,
            source_closure_path=closure_path,
            runtime_lock_path=runtime_path,
        )
