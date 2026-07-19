from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts import build_release as release_builder
from tests.release.conftest import (
    ReleaseFixture,
    rewrite_checksums,
    rewrite_training_manifest_binding,
    run_script,
)
from transformers import AutoConfig, AutoModelForTokenClassification

from pii_zh.models.qwen3_bi import Qwen3BiConfig, Qwen3BiForTokenClassification

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_builder_creates_exact_hf_package_and_auto_map(
    built_release: ReleaseFixture, repository_root: Path
) -> None:
    artifact = built_release.artifact
    expected = {
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "calibration.json",
        "checksums.txt",
        "config.json",
        "configuration_qwen3_bi.py",
        "data_provenance.json",
        "evaluation_report.json",
        "id2label.json",
        "model-index.yml",
        "model.safetensors",
        "modeling_qwen3_bi.py",
        "sbom.cdx.json",
        "special_tokens_map.json",
        "taxonomy.yaml",
        "teacher_provenance.json",
        "thresholds.yaml",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_manifest.json",
    }
    assert {path.name for path in artifact.iterdir()} == expected
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen3BiForTokenClassification"]
    assert config["auto_map"] == {
        "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
        "AutoModelForTokenClassification": "modeling_qwen3_bi.Qwen3BiForTokenClassification",
    }
    assert config["model_type"] == "qwen3_bi"
    assert config["use_cache"] is False
    compile((artifact / "configuration_qwen3_bi.py").read_text(encoding="utf-8"), "config", "exec")
    compile((artifact / "modeling_qwen3_bi.py").read_text(encoding="utf-8"), "model", "exec")

    result = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(artifact),
        "--source-registry",
        str(built_release.registry),
        "--dependency-scan",
        str(built_release.dependency_scan),
        "--dependency-exceptions",
        str(built_release.dependency_exceptions),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 0 blocker(s)" in result.stdout


def test_legacy_release_gate_rejects_community_preauthorization_file(
    built_release: ReleaseFixture,
    repository_root: Path,
) -> None:
    preauthorization = built_release.artifact / release_builder.COMMUNITY_PREAUTHORIZATION_FILENAME
    preauthorization.write_text("{}\n", encoding="utf-8")
    rewrite_checksums(built_release.artifact)
    result = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(built_release.artifact),
        "--source-registry",
        str(built_release.registry),
        "--dependency-scan",
        str(built_release.dependency_scan),
        "--dependency-exceptions",
        str(built_release.dependency_exceptions),
    )
    assert result.returncode == 1
    assert "RC_BLOCKED_UNREVIEWED_FILES" in result.stdout


def test_builder_copies_manifest_bound_optional_added_tokens(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    added_tokens = {"<synthetic-test-token>": 8}
    (release_fixture.checkpoint / "added_tokens.json").write_text(
        json.dumps(added_tokens, sort_keys=True), encoding="utf-8"
    )
    rewrite_training_manifest_binding(release_fixture.checkpoint, release_fixture.evidence)

    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )

    assert result.returncode == 0, result.stderr
    assert (
        json.loads((release_fixture.artifact / "added_tokens.json").read_text(encoding="utf-8"))
        == added_tokens
    )
    assert "  added_tokens.json\n" in (release_fixture.artifact / "checksums.txt").read_text(
        encoding="utf-8"
    )

    gate = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(release_fixture.artifact),
        "--source-registry",
        str(release_fixture.registry),
        "--dependency-scan",
        str(release_fixture.dependency_scan),
        "--dependency-exceptions",
        str(release_fixture.dependency_exceptions),
    )
    assert gate.returncode == 0, gate.stdout + gate.stderr

    (release_fixture.artifact / "added_tokens.json").write_text(
        json.dumps({"<synthetic-test-token>": 9}, sort_keys=True), encoding="utf-8"
    )
    rewrite_checksums(release_fixture.artifact)
    gate = run_script(
        repository_root,
        "release_gate.py",
        "--artifact",
        str(release_fixture.artifact),
        "--source-registry",
        str(release_fixture.registry),
        "--dependency-scan",
        str(release_fixture.dependency_scan),
        "--dependency-exceptions",
        str(release_fixture.dependency_exceptions),
    )
    assert gate.returncode == 1
    assert "RC_BLOCKED_OUTPUT_ARTIFACT_BINDING" in gate.stdout


def test_builder_non_force_publish_does_not_clobber_concurrently_created_target(
    release_fixture: ReleaseFixture,
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_publish = release_builder.model_card_v2._rename_directory_no_replace
    sentinel = release_fixture.artifact / "concurrent-owner.txt"

    def create_target_then_publish(source: Path, destination: Path) -> None:
        destination.mkdir()
        sentinel.write_text("owned by concurrent publisher", encoding="utf-8")
        real_publish(source, destination)

    monkeypatch.setattr(
        release_builder.model_card_v2,
        "_rename_directory_no_replace",
        create_target_then_publish,
    )

    with pytest.raises(release_builder.ReleaseBuildError, match="refusing to clobber"):
        release_builder.build_release(
            checkpoint_dir=release_fixture.checkpoint,
            evidence_dir=release_fixture.evidence,
            output_dir=release_fixture.artifact,
            model_card=release_fixture.model_card,
            remote_code_source=repository_root / "src/pii_zh/models/qwen3_bi.py",
            repository_root=repository_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "owned by concurrent publisher"
    assert not list(release_fixture.artifact.parent.glob(".*.staging-*"))


def test_builder_rejects_pickle_optimizer_and_raw_data(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    for filename in ("pytorch_model.bin", "optimizer.pt", "train.jsonl"):
        contaminated = release_fixture.checkpoint / filename
        contaminated.write_bytes(b"unsafe synthetic fixture")
        result = run_script(
            repository_root,
            "build_release.py",
            "--checkpoint-dir",
            str(release_fixture.checkpoint),
            "--evidence-dir",
            str(release_fixture.evidence),
            "--model-card",
            str(release_fixture.model_card),
            "--repository-root",
            str(release_fixture.repository_root),
            "--output-dir",
            str(release_fixture.artifact),
        )
        assert result.returncode == 1
        assert "release build blocked" in result.stderr
        contaminated.unlink()


def test_builder_rejects_invalid_safetensors(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    (release_fixture.checkpoint / "model.safetensors").write_bytes(b"not-a-pickle-either")
    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )
    assert result.returncode == 1
    assert "model.safetensors" in result.stderr


def test_builder_rejects_checkpoint_manifest_mismatch(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    tokenizer_path = release_fixture.checkpoint / "tokenizer_config.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model_max_length"] = 64
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )
    assert result.returncode == 1
    assert "do not match the completed training manifest" in result.stderr


def test_builder_rejects_unknown_generator_defect_semantics_in_evidence(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    provenance_path = release_fixture.evidence / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["lineage"] = {
        "dataset_id": "unknown-dataset-id",
        "generator_defect_ancestry": True,
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )

    assert result.returncode == 1
    assert "denied by artifact lineage policy" in result.stderr


def test_builder_rejects_non_boolean_release_eligibility(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    provenance_path = release_fixture.evidence / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["release_eligible"] = "false"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )

    assert result.returncode == 1
    assert "denied by artifact lineage policy" in result.stderr


def test_builder_rejects_causal_jpt_or_unmarked_checkpoint(
    release_fixture: ReleaseFixture, repository_root: Path
) -> None:
    config_path = release_fixture.checkpoint / "config.json"
    training_path = release_fixture.evidence / "training_manifest.json"
    tokenizer_path = release_fixture.checkpoint / "tokenizer_config.json"
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    original_training = json.loads(training_path.read_text(encoding="utf-8"))
    original_tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    mutations = (
        (config_path, original_config | {"model_type": "qwen3"}),
        (training_path, original_training | {"attention_mode": "jpt"}),
        (
            tokenizer_path,
            {
                key: value
                for key, value in original_tokenizer.items()
                if key != "pii_zh_boundary_mode"
            },
        ),
        (
            training_path,
            original_training | {"tokenizer": {"effective": {"boundary_mode": "missing"}}},
        ),
    )
    for path, mutation in mutations:
        path.write_text(json.dumps(mutation), encoding="utf-8")
        result = run_script(
            repository_root,
            "build_release.py",
            "--checkpoint-dir",
            str(release_fixture.checkpoint),
            "--evidence-dir",
            str(release_fixture.evidence),
            "--model-card",
            str(release_fixture.model_card),
            "--repository-root",
            str(release_fixture.repository_root),
            "--output-dir",
            str(release_fixture.artifact),
        )
        assert result.returncode == 1
        assert "release build blocked" in result.stderr
        config_path.write_text(json.dumps(original_config), encoding="utf-8")
        training_path.write_text(json.dumps(original_training), encoding="utf-8")
        tokenizer_path.write_text(json.dumps(original_tokenizer), encoding="utf-8")


def test_generated_remote_code_loads_tiny_local_safetensors(
    release_fixture: ReleaseFixture,
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = Qwen3BiConfig(
        attention_dropout=0.0,
        bi_attention_backend="eager",
        classifier_dropout=0.0,
        head_dim=8,
        hidden_size=16,
        id2label={0: "O", 1: "B-PERSON_NAME"},
        intermediate_size=24,
        label2id={"O": 0, "B-PERSON_NAME": 1},
        max_position_embeddings=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_labels=2,
        vocab_size=32,
    )
    model = Qwen3BiForTokenClassification(config).eval()
    model.save_pretrained(release_fixture.checkpoint, safe_serialization=True)
    rewrite_training_manifest_binding(release_fixture.checkpoint, release_fixture.evidence)
    result = run_script(
        repository_root,
        "build_release.py",
        "--checkpoint-dir",
        str(release_fixture.checkpoint),
        "--evidence-dir",
        str(release_fixture.evidence),
        "--model-card",
        str(release_fixture.model_card),
        "--repository-root",
        str(release_fixture.repository_root),
        "--output-dir",
        str(release_fixture.artifact),
    )
    assert result.returncode == 0, result.stderr

    loaded_config = AutoConfig.from_pretrained(
        release_fixture.artifact,
        local_files_only=True,
        trust_remote_code=True,
    )
    loaded = AutoModelForTokenClassification.from_pretrained(
        release_fixture.artifact,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
    ).eval()
    assert loaded_config.model_type == "qwen3_bi"
    assert type(loaded).__name__ == "Qwen3BiForTokenClassification"
    with torch.no_grad():
        output = loaded(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
    assert output.logits.shape == (1, 3, 2)


def test_community_remote_code_preserves_false_and_rejects_true_or_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    source = REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py"
    legacy_default = release_builder.render_remote_code(source)
    legacy_explicit = release_builder.render_remote_code(
        source, community_v2_preauthorization=False
    )
    assert legacy_default == legacy_explicit
    assert "self.pii_release_eligible = True" in legacy_default[0]
    assert "has_no_defaults_at_init = True" not in legacy_default[0]

    configuration, modeling = release_builder.render_remote_code(
        source,
        community_v2_preauthorization=True,
    )
    assert "has_no_defaults_at_init = True" in configuration
    assert 'getattr(config, "pii_release_eligible", None) is False' in modeling
    assert 'getattr(config, "pii_release_eligible", None) is True' not in modeling

    package = tmp_path / "community-model"
    package.mkdir()
    config = Qwen3BiConfig(
        attention_dropout=0.0,
        bi_attention_backend="eager",
        classifier_dropout=0.0,
        head_dim=8,
        hidden_size=16,
        id2label={0: "O", 1: "B-PERSON_NAME"},
        intermediate_size=24,
        label2id={"O": 0, "B-PERSON_NAME": 1},
        max_position_embeddings=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_labels=2,
        vocab_size=32,
    )
    local_model = Qwen3BiForTokenClassification(config).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    with torch.no_grad():
        local_logits = local_model(**inputs).logits
    local_model.config.pii_release_eligible = False
    local_model.save_pretrained(package, safe_serialization=True)
    (package / "configuration_qwen3_bi.py").write_text(configuration, encoding="utf-8")
    (package / "modeling_qwen3_bi.py").write_text(modeling, encoding="utf-8")
    assert json.loads((package / "config.json").read_text())["pii_release_eligible"] is False

    for invalid_value in (True, None):
        invalid = tmp_path / f"invalid-{invalid_value}"
        invalid.mkdir()
        for name in ("configuration_qwen3_bi.py", "modeling_qwen3_bi.py"):
            (invalid / name).write_bytes((package / name).read_bytes())
        invalid_config = json.loads((package / "config.json").read_text())
        if invalid_value is None:
            invalid_config.pop("pii_release_eligible")
        else:
            invalid_config["pii_release_eligible"] = invalid_value
        (invalid / "config.json").write_text(json.dumps(invalid_config), encoding="utf-8")
        with pytest.raises(ValueError, match="pii_release_eligible=false"):
            AutoConfig.from_pretrained(invalid, local_files_only=True, trust_remote_code=True)

    loaded_config = AutoConfig.from_pretrained(
        package,
        local_files_only=True,
        trust_remote_code=True,
    )
    loaded = AutoModelForTokenClassification.from_pretrained(
        package,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
    ).eval()
    assert loaded_config.pii_release_eligible is False
    assert loaded.config.pii_release_eligible is False
    assert set(loaded.state_dict()) == set(local_model.state_dict())
    with torch.no_grad():
        remote_logits = loaded(**inputs).logits
    assert torch.equal(remote_logits, local_logits)

    roundtrip = tmp_path / "roundtrip"
    loaded.save_pretrained(roundtrip, safe_serialization=True)
    reloaded_config = AutoConfig.from_pretrained(
        roundtrip,
        local_files_only=True,
        trust_remote_code=True,
    )
    assert reloaded_config.pii_release_eligible is False
