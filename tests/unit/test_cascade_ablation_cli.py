from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_zh.cascade import load_ablation_profile
from pii_zh.cascade.ablation_runtime import (
    CONTEXT_CONFIG_SCHEMA_VERSION,
    NER_ARTIFACT_BINDING_SCHEMA_VERSION,
    NER_MAPPING_SCHEMA_VERSION,
    load_closed_bio_mapping,
    load_context_config,
    verify_ner_artifact_binding,
)
from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)
from pii_zh.evaluation import cascade_ablation_cli
from pii_zh.evaluation.cascade_ablation_cli import SafeAblationCliError, main
from pii_zh.evaluation.io import EvaluationDataError, GoldRecord, Span


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_self_hashed(path: Path, document: dict[str, object], field: str) -> str:
    digest = _canonical_hash(document)
    path.write_text(
        json.dumps({**document, field: digest}, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def _mapping(path: Path, *, target: str = "PERSON") -> str:
    return _write_self_hashed(
        path,
        {
            "schema_version": NER_MAPPING_SCHEMA_VERSION,
            "mapping_id": "synthetic-per-only-v1",
            "source_to_target": {"PER": target},
        },
        "config_sha256",
    )


def _artifact(model_root: Path, binding_path: Path, mapping_sha256: str) -> None:
    model_root.mkdir()
    for name, payload in {
        "config.json": b"{}\n",
        "model.safetensors": b"synthetic-not-real-weights",
        "tokenizer.json": b"{}\n",
    }.items():
        (model_root / name).write_bytes(payload)
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(model_root.iterdir())
    }
    document: dict[str, object] = {
        "schema_version": NER_ARTIFACT_BINDING_SCHEMA_VERSION,
        "artifact_id": "synthetic-ner-fixture-v1",
        "artifact_kind": "closed_bio_token_classifier",
        "files": files,
        "files_combined_sha256": _canonical_hash(files),
        "mapping_config_sha256": mapping_sha256,
    }
    _write_self_hashed(binding_path, document, "binding_sha256")


def _record(text: str, *, doc_id: str = "synthetic-doc") -> DocumentRecord:
    value = "demo@example.com"
    start = text.index(value)
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(
            EntitySpan(
                start=start,
                end=start + len(value),
                label="EMAIL_ADDRESS",
                text=value,
                source="unit-test",
            ),
        ),
        language="zh-Hans-CN",
        domain="synthetic-test",
        scene="unit-test",
        provenance=Provenance(
            source_id="unit-test",
            source_kind="synthetic",
            source_revision="v1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit-test",
            generator_version="v1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="synthetic_fixture",
        ),
        public_weight_training_allowed=True,
        generator_version="v1",
    )


def test_ner_mapping_and_artifact_are_content_addressed(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_sha256 = _mapping(mapping_path)
    model_root = tmp_path / "model"
    binding_path = tmp_path / "binding.json"
    _artifact(model_root, binding_path, mapping_sha256)

    mapping = load_closed_bio_mapping(mapping_path)
    resolved, identity = verify_ner_artifact_binding(model_root, binding_path, mapping=mapping)

    assert resolved == model_root.resolve()
    assert identity.mapping_config_sha256 == mapping_sha256
    assert identity.artifact_id == "synthetic-ner-fixture-v1"
    assert len(identity.files_combined_sha256) == 64

    symlink_root = tmp_path / "model-link"
    symlink_root.symlink_to(model_root, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        verify_ner_artifact_binding(symlink_root, binding_path, mapping=mapping)

    (model_root / "config.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        verify_ner_artifact_binding(model_root, binding_path, mapping=mapping)


def test_ner_mapping_refuses_loc_to_address_or_any_non_person_target(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    _mapping(path, target="CN_ADDRESS")
    with pytest.raises(ValueError, match="only map"):
        load_closed_bio_mapping(path)


def test_reported_binding_ids_are_restricted_safe_tokens(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_document: dict[str, object] = {
        "schema_version": NER_MAPPING_SCHEMA_VERSION,
        "mapping_id": "../unsafe",
        "source_to_target": {"PER": "PERSON"},
    }
    _write_self_hashed(mapping_path, mapping_document, "config_sha256")
    with pytest.raises(ValueError, match="identity/hash"):
        load_closed_bio_mapping(mapping_path)

    valid_mapping_path = tmp_path / "valid-mapping.json"
    mapping_sha256 = _mapping(valid_mapping_path)
    mapping = load_closed_bio_mapping(valid_mapping_path)
    model_root = tmp_path / "model"
    binding_path = tmp_path / "binding.json"
    _artifact(model_root, binding_path, mapping_sha256)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding.pop("binding_sha256")
    binding["artifact_id"] = "unsafe/artifact"
    _write_self_hashed(binding_path, binding, "binding_sha256")
    with pytest.raises(ValueError, match="identity/hash"):
        verify_ner_artifact_binding(model_root, binding_path, mapping=mapping)

    context_path = tmp_path / "context.json"
    _write_self_hashed(
        context_path,
        {
            "schema_version": CONTEXT_CONFIG_SCHEMA_VERSION,
            "context_id": "unsafe context id",
            "context": {"positive_context": {}, "negative_context": {}},
        },
        "config_sha256",
    )
    with pytest.raises(ValueError, match="identity/hash"):
        load_context_config(context_path)


def test_context_config_is_self_hashed_and_report_identity_omits_phrases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.json"
    _write_self_hashed(
        path,
        {
            "schema_version": CONTEXT_CONFIG_SCHEMA_VERSION,
            "context_id": "synthetic-context-v1",
            "context": {
                "positive_context": {"PERSON": ["姓名"]},
                "negative_context": {"PERSON": ["产品名"]},
                "context_window": 12,
                "positive_delta": 0.1,
                "negative_delta": 0.2,
            },
        },
        "config_sha256",
    )

    enhancer, identity = load_context_config(path)

    assert enhancer is not None
    serialized = json.dumps(identity.to_report(), ensure_ascii=False)
    assert "姓名" not in serialized
    assert "产品名" not in serialized
    assert len(identity.config_sha256) == 64


@pytest.mark.parametrize(
    "context",
    [
        {
            "positive_context": {"PERSNO": ["姓名"]},
            "negative_context": {},
            "context_window": 12,
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON_NAME": ["姓名"]},
            "negative_context": {},
            "context_window": 12,
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON": []},
            "negative_context": {},
            "context_window": 12,
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON": ["姓名"]},
            "negative_context": {"PERSON": ["姓名"]},
            "context_window": 12,
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {},
            "negative_context": {},
            "context_window": 12,
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON": ["姓名"]},
            "negative_context": {},
            "context_window": True,
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON": ["姓名"]},
            "negative_context": {},
            "context_window": 12,
            "positive_delta": False,
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON": ["姓名"]},
            "negative_context": {},
            "context_window": 12,
            "positive_delta": float("nan"),
            "negative_delta": 0.2,
        },
        {
            "positive_context": {"PERSON": ["姓名"]},
            "negative_context": {},
            "positive_delta": 0.1,
            "negative_delta": 0.2,
        },
    ],
)
def test_context_config_rejects_typos_empty_values_and_non_finite_or_bool_types(
    tmp_path: Path,
    context: dict[str, object],
) -> None:
    path = tmp_path / "context.json"
    _write_self_hashed(
        path,
        {
            "schema_version": CONTEXT_CONFIG_SCHEMA_VERSION,
            "context_id": "strict-negative-fixture-v1",
            "context": context,
        },
        "config_sha256",
    )

    with pytest.raises((TypeError, ValueError)):
        load_context_config(path)


def test_rules_only_ablation_cli_emits_bound_raw_text_free_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "测试邮箱 demo@example.com"
    original_doc_id = "/" + "home/alice/private/病例-张三.txt"
    input_path = tmp_path / "input.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl([_record(sentinel, doc_id=original_doc_id)], input_path)

    status = main(
        [
            "--adapter",
            "rules_only",
            "--run-intent",
            "development-non-claim",
            "--input",
            str(input_path),
            "--predictions",
            str(predictions_path),
            "--report",
            str(report_path),
            "--bootstrap-samples",
            "0",
        ]
    )

    assert status == 0
    report_text = report_path.read_text(encoding="utf-8")
    predictions_text = predictions_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert sentinel not in report_text
    assert sentinel not in predictions_text
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert original_doc_id not in report_text
    assert original_doc_id not in predictions_text
    assert original_doc_id not in captured.out
    assert original_doc_id not in captured.err
    prediction_row = json.loads(predictions_text)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", prediction_row["doc_id"])
    report = json.loads(report_text)
    runtime = report["cascade_ablation_runtime"]
    assert runtime["ablation_id"] == "rules_only"
    assert runtime["stage_policy"]["schema_version"] == "pii-zh.pipeline-stage-policy.v1"
    assert len(runtime["stage_policy_sha256"]) == 64
    assert len(runtime["profile_sha256"]) == 64
    assert runtime["recognizer_artifact"] is None
    assert runtime["context_config"] is None
    assert runtime["execution_device"] == "cpu"
    assert runtime["run_intent"] == "development-non-claim"
    assert runtime["claim_status"] == "DEVELOPMENT_NON_CLAIM"
    assert runtime["claim_eligible"] is False
    assert runtime["canonical_evaluation_scope"]["entity_count"] == 24
    assert runtime["document_id_policy"]["original_document_ids_persisted"] is False
    implementation = runtime["implementation"]
    assert implementation["closure_mode"].startswith("full_pii_zh_")
    components = implementation["component_sha256"]
    for required in (
        "__init__.py",
        "cascade/__init__.py",
        "evaluation/__init__.py",
        "evaluation/baseline_reporting.py",
        "evaluation/metrics.py",
        "data/__init__.py",
        "fusion/deterministic.py",
        "rules/cn_common.py",
        "data/validators/__init__.py",
        "calibration/filtering.py",
        "inference/__init__.py",
        "inference/aiguard.py",
        "models/__init__.py",
        "presidio/__init__.py",
        "presidio/cascade_recognizer.py",
        "presidio/engine.py",
        "presidio/context_enhancer.py",
    ):
        assert required in components
    assert implementation["entrypoint_boundary"]["module"].endswith(":main")
    assert predictions_path.stat().st_mode & 0o777 == 0o400
    assert report_path.stat().st_mode & 0o777 == 0o400


def test_final_cli_fails_closed_without_context_before_model_initialization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl([_record("测试邮箱 demo@example.com")], input_path)
    model_root = tmp_path / "model"
    model_root.mkdir()

    status = main(
        [
            "--adapter",
            "final_cascade",
            "--run-intent",
            "development-non-claim",
            "--input",
            str(input_path),
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--model-path",
            str(model_root),
            "--bootstrap-samples",
            "0",
        ]
    )

    assert status == 2
    assert "requires an explicit context config" in capsys.readouterr().err
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


def test_ner_cli_fails_closed_without_mapping_and_artifact_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl([_record("测试邮箱 demo@example.com")], input_path)
    model_root = tmp_path / "ner"
    model_root.mkdir()

    status = main(
        [
            "--adapter",
            "presidio_zh_ner_cn_common",
            "--run-intent",
            "development-non-claim",
            "--input",
            str(input_path),
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--ner-model-path",
            str(model_root),
        ]
    )

    assert status == 2
    assert "mapping config and artifact binding are required" in capsys.readouterr().err


def test_cli_forbids_ad_hoc_entity_scope() -> None:
    with pytest.raises(SystemExit):
        cascade_ablation_cli._parser().parse_args(
            [
                "--adapter",
                "rules_only",
                "--run-intent",
                "development-non-claim",
                "--input",
                "input.jsonl",
                "--predictions",
                "predictions.jsonl",
                "--report",
                "report.json",
                "--entity",
                "PERSON",
            ]
        )


def test_gold_scope_rejects_unknown_or_extended_labels() -> None:
    record = GoldRecord(
        doc_id="safe-fixture",
        spans=(Span(start=0, end=1, label="HEALTH_INFORMATION"),),
        text_length=1,
    )
    with pytest.raises(EvaluationDataError, match="outside the fixed 24-class scope"):
        cascade_ablation_cli._privacy_safe_gold(
            [record],
            allowed_labels=frozenset({"PERSON_NAME"}),
        )


def test_cuda_device_is_single_visible_gpu_and_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl([_record("测试邮箱 demo@example.com")], input_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    profile = load_ablation_profile("project_model_raw")

    def args_for(device: str) -> object:
        return cascade_ablation_cli._parser().parse_args(
            [
                "--adapter",
                "project_model_raw",
                "--run-intent",
                "development-non-claim",
                "--input",
                str(input_path),
                "--predictions",
                str(tmp_path / "predictions.jsonl"),
                "--report",
                str(tmp_path / "report.json"),
                "--model-path",
                str(model_root),
                "--device",
                device,
            ]
        )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SafeAblationCliError, match="CUDA_VISIBLE_DEVICES"):
        cascade_ablation_cli._validate_args(args_for("cuda"), profile)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SafeAblationCliError, match="CUDA_VISIBLE_DEVICES"):
        cascade_ablation_cli._validate_args(args_for("cuda:0"), profile)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    args = args_for("cuda")
    cascade_ablation_cli._validate_args(args, profile)
    assert args.device == "cuda:0"
    with pytest.raises(SafeAblationCliError, match="device must be"):
        cascade_ablation_cli._validate_args(args_for("cuda:1"), profile)


@pytest.mark.parametrize(
    ("adapter_id", "expected_policy"),
    [
        ("project_model_raw", "exact_only_v1"),
        ("simple_union", "exact_only_v1"),
        ("validator_threshold_fusion", "high_overlap_v1"),
    ],
)
def test_project_adapter_selects_and_reports_explicit_recognizer_dedup_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_id: str,
    expected_policy: str,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    args = cascade_ablation_cli._parser().parse_args(
        [
            "--adapter",
            adapter_id,
            "--run-intent",
            "development-non-claim",
            "--input",
            str(tmp_path / "unused.jsonl"),
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--model-path",
            str(model_root),
        ]
    )
    seen: list[str] = []

    monkeypatch.setattr(
        cascade_ablation_cli,
        "verify_project_model_for_ablation",
        lambda *args, **kwargs: SimpleNamespace(to_report=lambda: {"model": "fixture"}),
    )

    def fake_from_pretrained(*args: object, **kwargs: object) -> object:
        seen.append(str(kwargs["recognizer_deduplication_policy"]))
        return SimpleNamespace()

    monkeypatch.setattr(
        cascade_ablation_cli.CascadePipeline,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        cascade_ablation_cli,
        "verify_loaded_project_predictor",
        lambda *args, **kwargs: None,
    )

    _, identity, _ = cascade_ablation_cli._build_pipeline(
        args,
        load_ablation_profile(adapter_id),
    )

    assert seen == [expected_policy]
    assert identity is not None
    assert identity["recognizer_deduplication_policy"] == expected_policy
