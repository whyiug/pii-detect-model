from __future__ import annotations

import copy
import json
import re
import shutil
import stat
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import build_release
from scripts import produce_community_cascade_release_v2_artifacts as artifacts

from pii_zh.cascade import expected_core24_label2id
from pii_zh.evaluation import release_eval_v2_prediction_provenance as model_provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, document: dict[str, Any]) -> bytes:
    payload = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return payload


def _self(document: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result[field] = artifacts.canonical_json_hash(result, remove=field)
    return result


def _quality_track(
    *, candidate_f1: float, candidate_recall: float, candidate_fpr: float, comparator_id: str
) -> dict[str, Any]:
    comparator_f1 = candidate_f1 - 0.04
    comparator_recall = candidate_recall - 0.03
    comparator_fpr = candidate_fpr + 0.02
    return {
        "declared_status": "PASS",
        "comparator_id": comparator_id,
        "strict_span": {"micro_f1": candidate_f1, "micro_recall": candidate_recall},
        "pii_free": {"document_fpr": candidate_fpr},
        "comparator_aggregate": {
            "strict_span": {
                "micro_f1": comparator_f1,
                "micro_recall": comparator_recall,
            },
            "pii_free": {"document_fpr": comparator_fpr},
        },
        "paired_bootstrap": {
            "micro_f1_delta": {
                "point_delta_candidate_minus_comparator": candidate_f1 - comparator_f1,
                "ci_low": 0.01,
                "ci_high": 0.07,
            },
            "macro_f1_delta": {
                "point_delta_candidate_minus_comparator": 0.03,
                "ci_low": 0.005,
                "ci_high": 0.06,
            },
            "document_fpr_delta": {
                "point_delta_candidate_minus_comparator": candidate_fpr - comparator_fpr,
                "ci_low": -0.04,
                "ci_high": 0.0,
            },
        },
    }


def _training_initialization() -> dict[str, Any]:
    labels = list(build_release.COMMUNITY_V2_MAPPED_TARGET_LABELS)
    return {
        "source_architecture": "Qwen3ForTokenClassification",
        "source_hashes": {
            "config.json": "a" * 64,
            "model.safetensors": "b" * 64,
            "tokenizer.json": "c" * 64,
            "tokenizer_config.json": "d" * 64,
        },
        "head_projections": [{"target_label": label} for label in labels],
        "mapped_target_labels": labels,
        "outside_row_copied": True,
        "mapped_head_rows_verified": True,
    }


def _training_data_fields() -> dict[str, Any]:
    label_counts = {
        label.removeprefix("B-"): 1
        for label in expected_core24_label2id()
        if label.startswith("B-")
    }

    def split(document_count: int, pii_free: int) -> dict[str, Any]:
        return {
            "summary": {
                "document_count": document_count,
                "pii_free_document_count": pii_free,
                "label_counts": label_counts,
                "sources": [
                    {
                        "source_id": "repo_curated_synthetic_templates",
                        "source_kind": "curated_deterministic_synthetic",
                        "document_count": document_count,
                    }
                ],
            }
        }

    return {
        "datasets": {
            "train": split(87_995, 39_600),
            "validation": split(2_005, 900),
        },
        "privacy": {
            "contains_entity_values": False,
            "contains_raw_text": False,
            "synthetic_only": True,
            "trainer_reporting_integrations": [],
        },
        "split_isolation": {
            "collision_counts": {"template_group": 17},
            "limitation": "validation_informed_template_family_overlap",
            "policy": "template-overlap-development-v1",
            "release_test_eligible": False,
            "template_group_overlap_allowed": True,
        },
    }


def _pii_bench_report(
    path: Path,
    *,
    training: dict[str, Any],
    training_payload: bytes,
) -> Path:
    results = {
        suite: {
            "documents": metrics["documents"],
            "strict_micro": {"f1": metrics["strict_micro_f1"]},
            "strict_macro": {"f1": metrics["strict_macro_f1"]},
        }
        for suite, metrics in artifacts.PII_BENCH_POSTHOC_METRICS.items()
    }
    report = _self(
        {
            "report_type": "bio24_model_raw_posthoc_evaluation",
            "candidate_id": "aiguard24-v3-seed97",
            "benchmark": {
                "dataset_id": artifacts.PII_BENCH_POSTHOC_DATASET_ID,
                "dataset_revision": artifacts.PII_BENCH_POSTHOC_DATASET_REVISION,
                "fixed_gold_verified": True,
            },
            "candidate": {
                "training_manifest_file_sha256": artifacts.sha256_bytes(training_payload),
                "training_manifest_sha256": training["manifest_sha256"],
                "attention_mode": "full",
                "core_entity_count": 24,
            },
            "comparison_contract": {
                "track": "model_raw",
                "threshold_tuning": False,
                "checkpoint_selection": False,
            },
            "active_comparator_envelope": {
                "descriptive_active_envelope_passed": False,
                "confirmatory_claim_allowed": False,
            },
            "results": results,
            "status": {
                "evidence_classification": "posthoc_descriptive_public_benchmark",
                "public_test_exposed": True,
                "posthoc_lineage": True,
                "selection_allowed": False,
                "release_selection_eligible": False,
                "confirmatory_claim_allowed": False,
            },
            "report_sha256": "",
        },
        "report_sha256",
    )
    _write_json(path, report)
    return path


def _identity_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    training = _self(
        {
            "base_source_id": build_release.COMMUNITY_V2_BASE_MODEL_ID,
            "source_revision": build_release.COMMUNITY_V2_BASE_MODEL_REVISION,
            "initialization": _training_initialization(),
            **_training_data_fields(),
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    training_path = tmp_path / "training.json"
    training_payload = _write_json(training_path, training)
    pii_bench_path = _pii_bench_report(
        tmp_path / "pii-bench-posthoc.json",
        training=training,
        training_payload=training_payload,
    )
    model = _self(
        {
            "model_id": "aiguard24-full-seed97",
            "ordered_labels": [
                label.removeprefix("B-")
                for label in expected_core24_label2id()
                if label.startswith("B-")
            ],
            "training_manifest_file_sha256": artifacts.sha256_bytes(training_payload),
            "training_manifest_sha256": training["manifest_sha256"],
            "model_identity_sha256": "2" * 64,
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    service = _self(
        {
            "service_id": "seed97-service",
            "profile_id": "community-cascade",
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    quality = _self(
        {
            "reported_status": "PASS",
            "candidate": {"model_id": model["model_id"], "service_id": service["service_id"]},
            "tracks": {
                "model_raw": _quality_track(
                    candidate_f1=0.91,
                    candidate_recall=0.92,
                    candidate_fpr=0.03,
                    comparator_id="aiguard-open24",
                ),
                "full_system": _quality_track(
                    candidate_f1=0.95,
                    candidate_recall=0.97,
                    candidate_fpr=0.01,
                    comparator_id="presidio-cncommon-v6",
                ),
            },
            "claims": {"claim_activation_allowed": False},
            "receipt_sha256": "",
        },
        "receipt_sha256",
    )
    quality_path = tmp_path / "quality.json"
    model_path = tmp_path / "model.json"
    service_path = tmp_path / "service.json"
    _write_json(quality_path, quality)
    _write_json(model_path, model)
    _write_json(service_path, service)
    return quality_path, model_path, service_path, training_path, pii_bench_path


def _model_card(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    quality, model, service, training, pii_bench = _identity_inputs(tmp_path)
    document = artifacts.build_model_card_artifact(
        quality_receipt_path=quality,
        model_manifest_path=model,
        service_manifest_path=service,
        training_manifest_path=training,
        pii_bench_report_path=pii_bench,
    )
    path = tmp_path / "model-card.json"
    _write_json(path, document)
    return document, path


def _valid_safetensors() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00"


def _canonical_hash(value: object) -> str:
    return artifacts.canonical_json_hash(value)


def _dependency_wheel(path: Path, *, binary: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "demo_dependency-1.2.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: demo-dependency\nVersion: 1.2.3\n",
        )
        archive.writestr(
            "demo_dependency/native.so" if binary else "demo_dependency/__init__.py",
            b"\x00\x01native-binary" if binary else b"__version__ = '1.2.3'\n",
        )
        if binary:
            archive.writestr("demo_dependency/model.bin", b"\x00\x02third-party-data")


def _sbom_with_demo_dependency(tmp_path: Path) -> tuple[Path, Path]:
    repository_root = tmp_path / "fixture-repository"
    (repository_root / "scripts").mkdir(parents=True)
    (repository_root / "configs/release").mkdir(parents=True)
    shutil.copyfile(
        REPOSITORY_ROOT / artifacts.PRODUCER_PATH,
        repository_root / artifacts.PRODUCER_PATH,
    )
    shutil.copyfile(
        REPOSITORY_ROOT / artifacts.SCHEMA_PATH,
        repository_root / artifacts.SCHEMA_PATH,
    )
    pyproject = repository_root / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nname = "{artifacts.PACKAGE_NAME}"\n'
        f'version = "{artifacts.PACKAGE_VERSION}"\n'
        'dependencies = ["demo-dependency==1.2.3"]\n'
    )
    lock = repository_root / "uv.lock"
    lock.write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.11"\n\n'
        f'[[package]]\nname = "{artifacts.PACKAGE_NAME}"\n'
        f'version = "{artifacts.PACKAGE_VERSION}"\nsource = {{ editable = "." }}\n'
        'dependencies = [{ name = "demo-dependency" }]\n\n'
        '[[package]]\nname = "demo-dependency"\nversion = "1.2.3"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    document = artifacts.build_sbom_artifact(
        lockfile_path=lock,
        pyproject_path=pyproject,
        repository_root=repository_root,
    )
    path = tmp_path / "sbom.json"
    _write_json(path, document)
    return path, repository_root


def _release_model_package(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    package = tmp_path / "model-package"
    package.mkdir()
    labels = expected_core24_label2id()
    config = {
        "model_type": "qwen3_bi",
        "pii_attention_mode": "full",
        "architectures": ["Qwen3BiForTokenClassification"],
        "architecture_version": "qwen3_bi_token_cls_v1",
        "bi_attention_backend": "sdpa",
        "use_cache": False,
        "pii_training_status": "completed_candidate_not_benchmark_evaluated",
        "pii_taxonomy_version": "1.0.0",
        "pii_release_eligible": False,
        "num_labels": 49,
        "label2id": labels,
        "id2label": {str(index): label for label, index in labels.items()},
        "auto_map": dict(build_release.AUTO_MAP),
    }
    (package / "config.json").write_text(json.dumps(config, sort_keys=True))
    (package / "id2label.json").write_text(
        json.dumps({str(index): label for label, index in labels.items()}, sort_keys=True)
    )
    (package / "tokenizer.json").write_text("{}")
    (package / "tokenizer_config.json").write_text(
        json.dumps({build_release.BOUNDARY_MODE_CONFIG_KEY: build_release.BOUNDARY_MODE})
    )
    (package / "special_tokens_map.json").write_text("{}")
    (package / "model.safetensors").write_bytes(_valid_safetensors())
    output_files = {
        name: artifacts.sha256_bytes((package / name).read_bytes())
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    }
    weights = {"model.safetensors": output_files["model.safetensors"]}
    output_artifact = {
        "schema_version": 1,
        "format": "huggingface_safetensors",
        "files": dict(sorted(output_files.items())),
        "weight_files": ["model.safetensors"],
        "weights_combined_sha256": _canonical_hash(weights),
        "artifact_files_combined_sha256": _canonical_hash(output_files),
    }
    training = {
        "schema_version": 4,
        "manifest_type": "aiguard24_community_training_v1",
        "status": "completed",
        "release_eligible": False,
        "attention_mode": "full",
        "fine_tuning": "lora_merged",
        "base_source_id": "ZJUICSR/AIguard-pii-detection-fast",
        "source_revision": "677a5ebc1600fef61e8973cafd3026be322b3a73",
        "taxonomy_version": "1.0.0",
        "seed": 97,
        "label2id": labels,
        "label_schema_sha256": _canonical_hash(labels),
        "tokenizer": {"effective": {"boundary_mode": build_release.BOUNDARY_MODE}},
        "initialization": {
            **_training_initialization(),
            "strategy": "aiguard_bie_to_pii_zh_core24_bio_v1",
            "target_label_count": 49,
            "target_label2id": labels,
            "release_eligible": False,
            "attention_conversion": {
                "source_attention_mode": "causal",
                "target_attention_mode": "full",
                "conversion": "qwen3_to_qwen3_bi_shared_state_dict_v1",
                "attention_backend": "sdpa",
                "strict_state_dict_load": True,
                "newly_initialized_parameter_keys": [],
                "discarded_parameter_keys": [],
            },
        },
        **_training_data_fields(),
        "checkpoint_selection": {
            "selection_split": "validation",
            "selection_metric": "risk_weighted_score",
            "selected_checkpoint_id": "checkpoint-10",
            "selected_global_step": 10,
            "best_metric": 0.5,
            "final_validation_replay_metric": 0.5,
            "adapter_model_sha256": "a" * 64,
        },
        "output_artifact": output_artifact,
    }
    training["manifest_sha256"] = _canonical_hash(training)
    training_path = package / "training_manifest.json"
    training_path.write_text(json.dumps(training, sort_keys=True))
    training_payload = training_path.read_bytes()
    pii_bench_path = _pii_bench_report(
        tmp_path / "pii-bench-posthoc-real.json",
        training=training,
        training_payload=training_payload,
    )
    actual = model_provenance._model_identity(package)
    ordered_labels = [label.removeprefix("B-") for label in labels if label.startswith("B-")]
    final_model = _self(
        {
            "model_id": "aiguard24-full-seed97",
            "artifact_class": "24_label_zh_hans_token_classifier",
            "label_count": 24,
            "ordered_labels": ordered_labels,
            "attention_mode": "full",
            "training_manifest_file_sha256": actual["training_manifest_file_sha256"],
            "training_manifest_sha256": actual["training_manifest_sha256"],
            "model_identity_sha256": actual["identity_sha256"],
            "artifact_sha256": actual["output_artifact_sha256"],
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    final_path = tmp_path / "final-model.json"
    _write_json(final_path, final_model)
    community_preauthorization = build_release.build_community_v2_preauthorization(
        checkpoint_dir=package,
        training_manifest_path=package / "training_manifest.json",
        id2label_path=package / "id2label.json",
        final_model_manifest_path=final_path,
    )
    _write_json(
        package / build_release.COMMUNITY_PREAUTHORIZATION_FILENAME,
        community_preauthorization,
    )
    service = _self(
        {"service_id": "seed97-service", "profile_id": "community-cascade", "manifest_sha256": ""},
        "manifest_sha256",
    )
    quality = _self(
        {
            "reported_status": "PASS",
            "candidate": {
                "model_id": "aiguard24-full-seed97",
                "service_id": "seed97-service",
            },
            "tracks": {
                "model_raw": _quality_track(
                    candidate_f1=0.90,
                    candidate_recall=0.91,
                    candidate_fpr=0.03,
                    comparator_id="aiguard-open24",
                ),
                "full_system": _quality_track(
                    candidate_f1=0.94,
                    candidate_recall=0.96,
                    candidate_fpr=0.01,
                    comparator_id="presidio-cncommon-v6",
                ),
            },
            "claims": {"claim_activation_allowed": False},
            "receipt_sha256": "",
        },
        "receipt_sha256",
    )
    service_path = tmp_path / "service.json"
    quality_path = tmp_path / "quality.json"
    _write_json(service_path, service)
    _write_json(quality_path, quality)
    model_card = artifacts.build_model_card_artifact(
        quality_receipt_path=quality_path,
        model_manifest_path=final_path,
        service_manifest_path=service_path,
        training_manifest_path=package / "training_manifest.json",
        pii_bench_report_path=pii_bench_path,
    )
    model_card_path = tmp_path / "model-card-real.json"
    _write_json(model_card_path, model_card)
    (package / "README.md").write_text(model_card["result"]["markdown"])
    (package / "LICENSE").write_bytes((REPOSITORY_ROOT / "LICENSE").read_bytes())
    (package / "NOTICE").write_text(build_release.render_community_v2_notice(training))
    (package / "SECURITY.md").write_bytes((REPOSITORY_ROOT / "SECURITY.md").read_bytes())
    (package / "THIRD_PARTY_NOTICES.md").write_text(
        build_release.render_community_v2_third_party_notices(training)
    )
    (package / "taxonomy.yaml").write_bytes(
        (REPOSITORY_ROOT / "src/pii_zh/taxonomy/taxonomy.yaml").read_bytes()
    )
    configuration, modeling = build_release.render_remote_code(
        REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
        community_v2_preauthorization=True,
    )
    (package / "configuration_qwen3_bi.py").write_text(configuration)
    (package / "modeling_qwen3_bi.py").write_text(modeling)
    build_release.write_checksums(package)
    return package, final_path, model_card_path, pii_bench_path


def _community_builder_arguments(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    checkpoint, final_model, model_card_artifact, pii_bench = _release_model_package(tmp_path)
    output = tmp_path / "community-output"
    arguments: dict[str, Any] = {
        "checkpoint_dir": checkpoint,
        "output_dir": output,
        "model_card": checkpoint / "README.md",
        "model_card_artifact": model_card_artifact,
        "pii_bench_report": pii_bench,
        "final_model_manifest": final_model,
        "remote_code_source": REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
    }
    return arguments, checkpoint, final_model, model_card_artifact, pii_bench


def test_artifact_schema_and_model_card_are_closed_and_path_free(tmp_path: Path) -> None:
    from pii_zh.inference import load_local_predictor

    schema = json.loads((REPOSITORY_ROOT / artifacts.SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    document, _path = _model_card(tmp_path)
    Draft202012Validator(schema).validate(document)
    artifacts.validate_artifact_document(document, artifact_id="model_card")

    assert document["release"] == artifacts.EXPECTED_RELEASE
    assert document["status"] == "PASS"
    markdown = document["result"]["markdown"]
    assert artifacts.DISPLAY_NAME in markdown
    assert artifacts.PACKAGE_VERSION in markdown
    assert all(
        marker.casefold() not in markdown.casefold()
        for marker in artifacts.FORBIDDEN_MODEL_CARD_MARKERS
    )
    assert "LOCAL_MODEL_DIR" not in markdown
    assert "predictor.predict(synthetic_text)" in markdown
    assert 'synthetic_text = "测试邮箱 demo@example.com"' in markdown
    assert "user@example.test" not in markdown
    assert "不构成对任意输入必然命中的承诺" in markdown
    assert '"label": span["entity_type"]' in markdown
    assert "AutoConfig.from_pretrained" in markdown
    assert "config.pii_release_eligible is False" in markdown
    assert "torch.isfinite(logits).all().item()" in markdown
    python_examples = re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)
    assert len(python_examples) == 2
    for index, example in enumerate(python_examples):
        compile(example, f"model-card-example-{index}.py", "exec")
    assert callable(load_local_predictor)
    assert str(tmp_path) not in json.dumps(document, ensure_ascii=False)


def test_model_card_tamper_and_no_clobber_fail_closed(tmp_path: Path) -> None:
    document, _path = _model_card(tmp_path)
    tampered = copy.deepcopy(document)
    tampered["result"]["markdown"] += "\n首个中文模型\n"
    tampered["result"]["markdown_sha256"] = artifacts.sha256_bytes(
        tampered["result"]["markdown"].encode()
    )
    tampered["artifact_sha256"] = artifacts.canonical_json_hash(tampered, remove="artifact_sha256")
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="forbidden"):
        artifacts.validate_artifact_document(tampered, artifact_id="model_card")

    output = tmp_path / "published.json"
    artifacts.publish_artifact(output, document)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="overwrite"):
        artifacts.publish_artifact(output, document)


def test_model_card_companion_outputs_are_exact_immutable_and_no_clobber(
    tmp_path: Path,
) -> None:
    document, _path = _model_card(tmp_path)
    artifact_output = tmp_path / "release" / "model-card.evidence.json"
    markdown_output = tmp_path / "model-package" / "README.md"

    artifacts.publish_model_card_outputs(artifact_output, markdown_output, document)

    assert json.loads(artifact_output.read_text(encoding="utf-8")) == document
    assert markdown_output.read_bytes() == document["result"]["markdown"].encode("utf-8")
    assert stat.S_IMODE(artifact_output.stat().st_mode) == 0o444
    assert stat.S_IMODE(markdown_output.stat().st_mode) == 0o444
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="overwrite"):
        artifacts.publish_model_card_outputs(artifact_output, markdown_output, document)


def test_model_card_companion_pair_rolls_back_on_second_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _path = _model_card(tmp_path)
    artifact_output = tmp_path / "model-card.evidence.json"
    markdown_output = tmp_path / "README.md"
    real_link = artifacts.os.link

    def fail_markdown_link(source: Path, target: Path) -> None:
        if Path(target) == markdown_output:
            raise FileExistsError("simulated race")
        real_link(source, target)

    monkeypatch.setattr(artifacts.os, "link", fail_markdown_link)
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="output pair"):
        artifacts.publish_model_card_outputs(artifact_output, markdown_output, document)
    assert not artifact_output.exists()
    assert not markdown_output.exists()


def test_model_card_cli_materializes_bound_markdown_companion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    quality, model, service, training, pii_bench = _identity_inputs(tmp_path)
    artifact_output = tmp_path / "model-card.evidence.json"
    markdown_output = tmp_path / "README.md"
    assert (
        artifacts.main(
            [
                "model-card",
                "--quality-receipt",
                str(quality),
                "--final-model-manifest",
                str(model),
                "--service-configuration-manifest",
                str(service),
                "--training-manifest",
                str(training),
                "--pii-bench-report",
                str(pii_bench),
                "--output",
                str(artifact_output),
                "--markdown-output",
                str(markdown_output),
            ]
        )
        == 0
    )
    reported = json.loads(capsys.readouterr().out)
    document = json.loads(artifact_output.read_text(encoding="utf-8"))
    assert reported["artifact_sha256"] == document["artifact_sha256"]
    assert (
        artifacts.sha256_bytes(markdown_output.read_bytes())
        == document["result"]["markdown_sha256"]
    )


def test_model_package_manifest_binds_exact_readme_and_complete_inventory(tmp_path: Path) -> None:
    package, final_model_path, model_card_path, _pii_bench_path = _release_model_package(tmp_path)

    document = artifacts.build_model_package_manifest(
        model_package_root=package,
        model_card_artifact_path=model_card_path,
        final_model_manifest_path=final_model_path,
    )
    artifacts.validate_artifact_document(document, artifact_id="model_package_manifest")
    assert set(document["result"]["files"]) == artifacts.MODEL_PACKAGE_REQUIRED_IDS
    assert document["result"]["entity_type_count"] == 24
    assert document["result"]["token_label_count"] == 49
    assert document["result"]["training_release_eligible"] is False

    (package / "README.md").write_text("different\n")
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="README"):
        artifacts.build_model_package_manifest(
            model_package_root=package,
            model_card_artifact_path=model_card_path,
            final_model_manifest_path=final_model_path,
        )


def test_community_seed97_builder_uses_minimal_checkpoint_evidence_and_no_clobber(
    tmp_path: Path,
) -> None:
    checkpoint, final_model_path, model_card_path, pii_bench_path = _release_model_package(tmp_path)
    output = tmp_path / "community-seed97-package"

    built = build_release.build_community_v2_release(
        checkpoint_dir=checkpoint,
        output_dir=output,
        model_card=checkpoint / "README.md",
        model_card_artifact=model_card_path,
        pii_bench_report=pii_bench_path,
        final_model_manifest=final_model_path,
        remote_code_source=REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
    )

    assert built == output
    assert {path.name for path in output.iterdir()} == set(
        build_release.COMMUNITY_V2_REQUIRED_RELEASE_FILES
    )
    assert not {
        "calibration.json",
        "data_provenance.json",
        "evaluation_report.json",
        "model-index.yml",
        "sbom.cdx.json",
        "teacher_provenance.json",
        "thresholds.yaml",
    } & {path.name for path in output.iterdir()}
    assert (output / "training_manifest.json").read_bytes() == (
        checkpoint / "training_manifest.json"
    ).read_bytes()
    manifest = artifacts.build_model_package_manifest(
        model_package_root=output,
        model_card_artifact_path=model_card_path,
        final_model_manifest_path=final_model_path,
    )
    artifacts.validate_artifact_document(manifest, artifact_id="model_package_manifest")
    with pytest.raises(build_release.ReleaseBuildError, match="no-clobber"):
        build_release.build_community_v2_release(
            checkpoint_dir=checkpoint,
            output_dir=output,
            model_card=checkpoint / "README.md",
            model_card_artifact=model_card_path,
            pii_bench_report=pii_bench_path,
            final_model_manifest=final_model_path,
            remote_code_source=REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
        )


def test_community_builder_rejects_typed_card_companion_or_producer_drift(
    tmp_path: Path,
) -> None:
    arguments, checkpoint, _final_model, model_card_artifact, _pii_bench = (
        _community_builder_arguments(tmp_path)
    )
    bad_markdown = tmp_path / "different-README.md"
    bad_markdown.write_text(checkpoint.joinpath("README.md").read_text() + "\ndrift\n")
    with pytest.raises(build_release.ReleaseBuildError, match="Markdown/training binding"):
        build_release.build_community_v2_release(**(arguments | {"model_card": bad_markdown}))

    document = json.loads(model_card_artifact.read_text())
    document["producer"]["file_sha256"] = "0" * 64
    document["artifact_sha256"] = artifacts.canonical_json_hash(document, remove="artifact_sha256")
    tampered = tmp_path / "tampered-model-card.json"
    _write_json(tampered, document)
    with pytest.raises(build_release.ReleaseBuildError, match="producer identity"):
        build_release.build_community_v2_release(**(arguments | {"model_card_artifact": tampered}))


def test_model_card_rejects_wrong_taxonomy_or_pii_bench_metrics(tmp_path: Path) -> None:
    document, _path = _model_card(tmp_path)
    taxonomy_drift = copy.deepcopy(document)
    taxonomy_drift["result"]["ordered_entity_labels"][0] = "ORGANIZATION"
    taxonomy_drift["artifact_sha256"] = artifacts.canonical_json_hash(
        taxonomy_drift, remove="artifact_sha256"
    )
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="model card"):
        artifacts.validate_artifact_document(taxonomy_drift, artifact_id="model_card")
    assert "ORGANIZATION" not in document["result"]["ordered_entity_labels"]
    assert "组织、" not in document["result"]["markdown"]

    metric_drift_root = tmp_path / "metric-drift"
    metric_drift_root.mkdir()
    quality, model, service, training, pii_bench = _identity_inputs(metric_drift_root)
    report = json.loads(pii_bench.read_text())
    report["results"]["formal"]["strict_micro"]["f1"] = 0.99999999
    report["report_sha256"] = artifacts.canonical_json_hash(report, remove="report_sha256")
    _write_json(pii_bench, report)
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="descriptive metrics"):
        artifacts.build_model_card_artifact(
            quality_receipt_path=quality,
            model_manifest_path=model,
            service_manifest_path=service,
            training_manifest_path=training,
            pii_bench_report_path=pii_bench,
        )


def test_model_package_rejects_extra_file_and_open_preauthorization_shape(
    tmp_path: Path,
) -> None:
    package, final_model, model_card, _pii_bench = _release_model_package(tmp_path)
    (package / "EXTRA.md").write_text("benign extra\n")
    build_release.write_checksums(package)
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="exact seed-97"):
        artifacts.build_model_package_manifest(
            model_package_root=package,
            model_card_artifact_path=model_card,
            final_model_manifest_path=final_model,
        )

    (package / "EXTRA.md").unlink()
    preauth_path = package / build_release.COMMUNITY_PREAUTHORIZATION_FILENAME
    preauth = json.loads(preauth_path.read_text())
    preauth["unexpected"] = "accepted-by-open-shape"
    preauth["preauthorization_sha256"] = artifacts.canonical_json_hash(
        preauth, remove="preauthorization_sha256"
    )
    _write_json(preauth_path, preauth)
    build_release.write_checksums(package)
    with pytest.raises(
        artifacts.CommunityReleaseArtifactError,
        match="preauthorization binding",
    ):
        artifacts.build_model_package_manifest(
            model_package_root=package,
            model_card_artifact_path=model_card,
            final_model_manifest_path=final_model,
        )


def test_community_preauthorization_rejects_nonfull_final_manifest(tmp_path: Path) -> None:
    package, final_model_path, _model_card, _pii_bench = _release_model_package(tmp_path)
    final_model = json.loads(final_model_path.read_text())
    final_model["attention_mode"] = "causal"
    final_model["manifest_sha256"] = artifacts.canonical_json_hash(
        final_model, remove="manifest_sha256"
    )
    _write_json(final_model_path, final_model)
    with pytest.raises(build_release.ReleaseBuildError, match="exact frozen seed-97"):
        build_release.build_community_v2_preauthorization(
            checkpoint_dir=package,
            training_manifest_path=package / "training_manifest.json",
            id2label_path=package / "id2label.json",
            final_model_manifest_path=final_model_path,
        )


def test_community_builder_rejects_symlinks_traversal_and_source_overrides(
    tmp_path: Path,
) -> None:
    arguments, checkpoint, final_model, _model_card, _pii_bench = _community_builder_arguments(
        tmp_path
    )
    dangling_target = tmp_path / "dangling-target"
    dangling_output = tmp_path / "dangling-output"
    dangling_output.symlink_to(dangling_target, target_is_directory=True)
    with pytest.raises(build_release.ReleaseBuildError, match="symlink"):
        build_release.build_community_v2_release(**(arguments | {"output_dir": dangling_output}))
    assert not dangling_target.exists()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(build_release.ReleaseBuildError, match="symlink"):
        build_release.build_community_v2_release(
            **(arguments | {"output_dir": linked_parent / "output"})
        )

    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(build_release.ReleaseBuildError, match="symlink"):
        build_release.build_community_v2_release(
            **(arguments | {"checkpoint_dir": checkpoint_link})
        )

    final_link = tmp_path / "final-link.json"
    final_link.symlink_to(final_model)
    with pytest.raises(build_release.ReleaseBuildError, match="symlink"):
        build_release.build_community_v2_release(
            **(arguments | {"final_model_manifest": final_link})
        )

    traversal = tmp_path / "lexical" / ".." / "traversal-output"
    with pytest.raises(build_release.ReleaseBuildError, match="traversal"):
        build_release.build_community_v2_release(**(arguments | {"output_dir": traversal}))

    alternate_source = tmp_path / "qwen3_bi.py"
    alternate_source.write_bytes((REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py").read_bytes())
    with pytest.raises(build_release.ReleaseBuildError, match="alternate remote-code"):
        build_release.build_community_v2_release(
            **(arguments | {"remote_code_source": alternate_source})
        )
    with pytest.raises(build_release.ReleaseBuildError, match="overrides"):
        build_release.build_community_v2_release(
            **(arguments | {"project_file_overrides": {"NOTICE": tmp_path / "notice"}})
        )


def test_model_package_rejects_different_model_even_with_valid_public_files(
    tmp_path: Path,
) -> None:
    package, final_model_path, model_card_path, _pii_bench_path = _release_model_package(tmp_path)
    weight_path = package / "model.safetensors"
    weight_path.write_bytes(_valid_safetensors() + b"different")

    with pytest.raises(
        artifacts.CommunityReleaseArtifactError,
        match="safetensors|completed full-attention community model",
    ):
        artifacts.build_model_package_manifest(
            model_package_root=package,
            model_card_artifact_path=model_card_path,
            final_model_manifest_path=final_model_path,
        )


def test_community_preauthorization_is_explicit_and_rejects_true_or_wrong_seed(
    tmp_path: Path,
) -> None:
    package, final_model_path, _model_card_path, _pii_bench_path = _release_model_package(tmp_path)
    id2label_path = package / "id2label.json"
    with pytest.raises(build_release.ReleaseBuildError, match="eligibility state"):
        build_release.build_release_config(package / "config.json", id2label_path)
    build_release.build_release_config(
        package / "config.json",
        id2label_path,
        community_v2_preauthorization=True,
    )

    config = json.loads((package / "config.json").read_text())
    config["pii_release_eligible"] = True
    (package / "config.json").write_text(json.dumps(config, sort_keys=True))
    with pytest.raises(build_release.ReleaseBuildError, match="eligibility state"):
        build_release.build_release_config(
            package / "config.json",
            id2label_path,
            community_v2_preauthorization=True,
        )

    config["pii_release_eligible"] = False
    (package / "config.json").write_text(json.dumps(config, sort_keys=True))
    training_path = package / "training_manifest.json"
    training = json.loads(training_path.read_text())
    training["seed"] = 98
    training["manifest_sha256"] = artifacts.canonical_json_hash(training, remove="manifest_sha256")
    training_path.write_text(json.dumps(training, sort_keys=True))
    with pytest.raises(build_release.ReleaseBuildError, match="exact frozen seed-97"):
        build_release.build_community_v2_preauthorization(
            checkpoint_dir=package,
            training_manifest_path=training_path,
            id2label_path=id2label_path,
            final_model_manifest_path=final_model_path,
        )


def test_community_preauthorization_rejects_wrong_final_model_hash(tmp_path: Path) -> None:
    package, final_model_path, _model_card_path, _pii_bench_path = _release_model_package(tmp_path)
    final_model = json.loads(final_model_path.read_text())
    final_model["model_identity_sha256"] = "0" * 64
    final_model["manifest_sha256"] = artifacts.canonical_json_hash(
        final_model, remove="manifest_sha256"
    )
    _write_json(final_model_path, final_model)
    with pytest.raises(build_release.ReleaseBuildError, match="exact frozen seed-97"):
        build_release.build_community_v2_preauthorization(
            checkpoint_dir=package,
            training_manifest_path=package / "training_manifest.json",
            id2label_path=package / "id2label.json",
            final_model_manifest_path=final_model_path,
        )


def test_wheel_manifest_rejects_old_version_and_accepts_successor(tmp_path: Path) -> None:
    def make(path: Path, version: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name in artifacts.WHEEL_REQUIRED_SUFFIXES:
                archive.writestr(name.replace(":", "/"), "fixture\n")
            archive.writestr(
                f"pii_zh_qwen-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.4\nName: pii-zh-qwen\nVersion: {version}\n",
            )

    old = tmp_path / "pii_zh_qwen-0.1.0-py3-none-any.whl"
    make(old, "0.1.0")
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="version"):
        artifacts.build_wheel_manifest(wheel_path=old)

    successor = tmp_path / "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
    make(successor, artifacts.PACKAGE_VERSION)
    document = artifacts.build_wheel_manifest(wheel_path=successor)
    artifacts.validate_artifact_document(document, artifact_id="wheel_manifest")


def test_wheelhouse_manifest_binds_exact_dependency_wheels(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    dependency = wheelhouse / "demo_dependency-1.2.3-py3-none-any.whl"
    _dependency_wheel(dependency, binary=True)
    sbom_path, repository_root = _sbom_with_demo_dependency(tmp_path)

    document = artifacts.build_wheelhouse_manifest(
        wheelhouse_root=wheelhouse,
        sbom_artifact_path=sbom_path,
        repository_root=repository_root,
    )
    artifacts.validate_artifact_document(
        document,
        artifact_id="wheelhouse_manifest",
        repository_root=repository_root,
    )
    assert document["result"]["wheel_count"] == 1
    assert document["result"]["packages"]["package:demo-dependency"]["version"] == "1.2.3"

    _dependency_wheel(dependency, binary=False)
    actual_files, _packages = artifacts._wheelhouse_inventory(wheelhouse)
    assert actual_files != document["result"]["files"]


def test_wheelhouse_uses_only_root_distribution_metadata(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    dependency = wheelhouse / "demo_dependency-1.2.3-py3-none-any.whl"
    _dependency_wheel(dependency)
    with zipfile.ZipFile(dependency, "a") as archive:
        archive.writestr(
            "demo_dependency/_vendor/vendored-9.9.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: vendored\nVersion: 9.9\n",
        )
    sbom_path, repository_root = _sbom_with_demo_dependency(tmp_path)

    document = artifacts.build_wheelhouse_manifest(
        wheelhouse_root=wheelhouse,
        sbom_artifact_path=sbom_path,
        repository_root=repository_root,
    )

    assert document["result"]["wheel_count"] == 1
    assert set(document["result"]["packages"]) == {"package:demo-dependency"}
    assert document["result"]["packages"]["package:demo-dependency"]["version"] == "1.2.3"
    members, name, version = artifacts._wheel_inventory(
        dependency,
        expected_name=None,
        expected_version=None,
        schema_safe_member_ids=False,
        reject_release_data=False,
    )
    assert "demo_dependency/_vendor/vendored-9.9.dist-info/METADATA" in members
    assert (name, version) == ("demo-dependency", "1.2.3")


def test_wheelhouse_rejects_multiple_root_distribution_metadata_files(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    dependency = wheelhouse / "demo_dependency-1.2.3-py3-none-any.whl"
    _dependency_wheel(dependency)
    with zipfile.ZipFile(dependency, "a") as archive:
        archive.writestr(
            "other_dependency-4.5.6.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: other-dependency\nVersion: 4.5.6\n",
        )

    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="multiple root"):
        artifacts._wheelhouse_inventory(wheelhouse)


@pytest.mark.parametrize(
    ("filename", "metadata_member", "metadata", "expected_error"),
    [
        (
            "demo_dependency-1.2.3-py3-none-any.whl",
            "demo_dependency/_vendor/vendored-9.9.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: vendored\nVersion: 9.9\n",
            "metadata is missing",
        ),
        (
            "demo_dependency-1.2.3-py3-none-any.whl",
            ".dist-info/METADATA",
            "Metadata-Version: 2.4\nName: demo-dependency\nVersion: 1.2.3\n",
            "directory identity is malformed",
        ),
        (
            "demo_dependency-1.2.3-py3-none-any.whl",
            "wrong_name-1.2.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: demo-dependency\nVersion: 1.2.3\n",
            "name differs",
        ),
        (
            "demo_dependency-1.2.3-py3-none-any.whl",
            "demo_dependency-9.9.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: demo-dependency\nVersion: 1.2.3\n",
            "version differs",
        ),
        (
            "other_dependency-1.2.3-py3-none-any.whl",
            "demo_dependency-1.2.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: demo-dependency\nVersion: 1.2.3\n",
            "filename differs",
        ),
    ],
)
def test_wheelhouse_rejects_ambiguous_distribution_identity(
    tmp_path: Path,
    filename: str,
    metadata_member: str,
    metadata: str,
    expected_error: str,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with zipfile.ZipFile(wheelhouse / filename, "w") as archive:
        archive.writestr(metadata_member, metadata)

    with pytest.raises(artifacts.CommunityReleaseArtifactError, match=expected_error):
        artifacts._wheelhouse_inventory(wheelhouse)


def test_public_scan_does_not_deep_scan_third_party_binary_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "README.md").write_text("safe model documentation\n")
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        (model_root / name).write_text(f"safe {name}\n")
    model_files = artifacts._inventory_tree(model_root, prefix="pkg", reject_package_payloads=True)
    wheel = tmp_path / "pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in artifacts.WHEEL_REQUIRED_SUFFIXES:
            archive.writestr(name.replace(":", "/"), "safe fixture\n")
        archive.writestr(
            "pii_zh_qwen-0.2.0rc1.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: pii-zh-qwen\nVersion: 0.2.0rc1\n",
        )
    wheel_binding = artifacts._large_file_binding(wheel, field="fixture wheel")
    wheel_files, _name, _version = artifacts._wheel_inventory(wheel)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _dependency_wheel(wheelhouse / "demo_dependency-1.2.3-py3-none-any.whl", binary=True)
    wheelhouse_files, wheelhouse_packages = artifacts._wheelhouse_inventory(wheelhouse)

    artifact_ids = {
        "model_card",
        "model_package_manifest",
        "technical_documentation_manifest",
        "wheel_manifest",
        "wheelhouse_manifest",
        "container_manifest",
        "sbom",
        "license_report",
        "dependency_scan",
    }
    bound: dict[str, Path] = {}
    for artifact_id in artifact_ids:
        path = tmp_path / f"{artifact_id}.json"
        path.write_text("{}\n")
        bound[artifact_id] = path
    license_inputs = {
        "license_document": dict(model_files["pkg:LICENSE"]),
        "notice_document": dict(model_files["pkg:NOTICE"]),
        "third_party_notices": dict(model_files["pkg:THIRD_PARTY_NOTICES.md"]),
    }

    def load(_path: Path, *, artifact_id: str, **_kwargs: Any) -> tuple[dict[str, Any], bytes]:
        if artifact_id == "model_package_manifest":
            document = {"result": {"files": model_files}}
        elif artifact_id == "wheel_manifest":
            document = {"inputs": {"wheel": wheel_binding}, "result": {"files": wheel_files}}
        elif artifact_id == "wheelhouse_manifest":
            document = {"result": {"files": wheelhouse_files, "packages": wheelhouse_packages}}
        elif artifact_id == "license_report":
            document = {"inputs": license_inputs}
        else:
            document = {}
        return document, b"{}\n"

    def scan(paths: list[Path]) -> list[object]:
        assert not any(
            candidate.suffix == ".so"
            for root in paths
            if root.is_dir()
            for candidate in root.rglob("*")
        )
        return []

    monkeypatch.setattr(artifacts, "load_and_validate_artifact", load)
    monkeypatch.setattr(artifacts.scan_public_artifacts, "scan_paths", scan)
    document = artifacts.build_public_artifact_scan(
        model_package_root=model_root,
        wheel_path=wheel,
        wheelhouse_root=wheelhouse,
        bound_artifacts=bound,
    )
    assert document["status"] == "PASS"
    assert not any(
        key.startswith("wheelhouse:") for key in document["result"]["deep_scanned_inventory"]
    )
    assert any(
        key.startswith("wheelhouse:")
        for key in document["result"]["hashed_dependency_wheel_inventory"]
    )

    license_inputs["notice_document"] = {
        **license_inputs["notice_document"],
        "file_sha256": "0" * 64,
    }
    with pytest.raises(
        artifacts.CommunityReleaseArtifactError,
        match="license report differs from model-package legal documents",
    ):
        artifacts.build_public_artifact_scan(
            model_package_root=model_root,
            wheel_path=wheel,
            wheelhouse_root=wheelhouse,
            bound_artifacts=bound,
        )


def test_sbom_and_license_report_keep_human_approval_pending(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nname = "{artifacts.PACKAGE_NAME}"\nversion = "{artifacts.PACKAGE_VERSION}"\n'
    )
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.11"\n\n'
        f'[[package]]\nname = "{artifacts.PACKAGE_NAME}"\n'
        f'version = "{artifacts.PACKAGE_VERSION}"\nsource = {{ editable = "." }}\n\n'
        '[[package]]\nname = "demo-dependency"\nversion = "1.2.3"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    sbom = artifacts.build_sbom_artifact(lockfile_path=lock, pyproject_path=pyproject)
    sbom_path = tmp_path / "sbom.json"
    _write_json(sbom_path, sbom)
    artifacts.validate_artifact_document(
        sbom, artifact_id="sbom", verify_current_repository_inputs=False
    )

    model_package = tmp_path / "model-package"
    model_package.mkdir()
    license_path = model_package / "LICENSE"
    notice_path = model_package / "NOTICE"
    third_party = model_package / "THIRD_PARTY_NOTICES.md"
    for path in (license_path, notice_path, third_party):
        path.write_text("review fixture\n")
    report = artifacts.build_license_report(
        sbom_artifact_path=sbom_path,
        model_package_root=model_package,
        verify_current_repository_inputs=False,
    )
    artifacts.validate_artifact_document(report, artifact_id="license_report")
    assert report["status"] == "COMPLETE_HUMAN_APPROVAL_PENDING"
    assert report["result"]["human_approval_status"] == "pending"
    assert report["result"]["automated_release_clearance"] is False


def test_sbom_rejects_historical_package_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "pii-zh-qwen"\nversion = "0.1.0"\n')
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.11"\n\n'
        '[[package]]\nname = "pii-zh-qwen"\nversion = "0.1.0"\n'
        'source = { editable = "." }\n'
    )
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="successor version"):
        artifacts.build_sbom_artifact(lockfile_path=lock, pyproject_path=pyproject)


def test_osv_scanner_official_multiline_version_is_normalized() -> None:
    output = (
        b"osv-scanner version: 2.3.8\n"
        b"osv-scalibr version: 0.4.5\n"
        b"commit: 408fcd6f8707999a29e7ba45e15809764cf24f67\n"
        b"built at: 2026-05-08T04:54:35Z\n"
    )
    assert artifacts._parse_osv_scanner_version(output, b"") == "2.3.8"


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"osv-scanner version: 2.3.8\n", b""),
        (
            b"osv-scanner version: 2.3.8\n"
            b"osv-scalibr version: 0.4.5\n"
            b"commit: 408fcd6f8707999a29e7ba45e15809764cf24f67\n"
            b"built at: 2026-05-08T04:54:35Z\n"
            b"unexpected: injected\n",
            b"",
        ),
        (b"x" * 1025, b""),
        (b"\xff\n", b""),
        (
            b"osv-scanner version: 2.3.8\n"
            b"osv-scalibr version: 0.4.5\n"
            b"commit: 408fcd6f8707999a29e7ba45e15809764cf24f67\n"
            b"built at: 2026-05-08T04:54:35Z\n",
            b"warning\n",
        ),
    ],
)
def test_osv_scanner_version_rejects_nonofficial_or_ambiguous_output(
    stdout: bytes, stderr: bytes
) -> None:
    with pytest.raises(artifacts.CommunityReleaseArtifactError, match="version output"):
        artifacts._parse_osv_scanner_version(stdout, stderr)
