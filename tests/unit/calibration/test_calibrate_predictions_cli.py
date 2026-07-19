from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pii_zh.calibration import CalibrationBundle
from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)
from pii_zh.evaluation import (
    PredictionRecord,
    Span,
    add_manifest_hash,
    canonical_json_hash,
    sha256_file,
    write_prediction_jsonl,
)


def _synthetic_record(
    *,
    doc_id: str,
    text: str,
    entity_text: str,
    split: str,
    label: str = "PERSON_NAME",
    frozen_calibration: bool = False,
    release_dataset_id: str = "pii_zh_synthetic_sota_release_eval_v2",
    release_dataset_version: str = "2.0.0",
    release_usage_policy: str = "post_cross_seed_selection_calibration_only",
) -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=[
            EntitySpan(
                start=0,
                end=len(entity_text),
                label=label,
                text=entity_text,
                source="synthetic",
            )
        ],
        language="zh-Hans-CN",
        domain="fixture",
        scene="calibration",
        provenance=Provenance(
            source_id="synthetic-fixture",
            source_kind="synthetic",
            source_revision="v1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="fixture",
            generator_version="v1",
            evaluation_only=frozen_calibration,
            metadata=(
                {
                    "release_evaluation_dataset_id": release_dataset_id,
                    "release_evaluation_dataset_version": release_dataset_version,
                    "release_evaluation_split": "calibration",
                    "generation_salt_sha256": (
                        "69ecd313a414459ece55274a7d7ce881537fb8f57e51940068b4a190a6fa437e"
                    ),
                    "weight_training_allowed_by_protocol": False,
                }
                if frozen_calibration
                else {}
            ),
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="validated",
        ),
        public_weight_training_allowed=not frozen_calibration,
        generator_version="v1",
        split=split,
        metadata=(
            {
                "release_evaluation_split": "calibration",
                "synthetic_release_eval_recipe": release_dataset_version,
                "generation_salt_sha256": (
                    "69ecd313a414459ece55274a7d7ce881537fb8f57e51940068b4a190a6fa437e"
                ),
                "usage_policy": release_usage_policy,
            }
            if frozen_calibration
            else {}
        ),
    )


def _run_calibration(
    tmp_path: Path,
    *,
    split: str = "validation",
    allowed_labels: tuple[str, ...] | None = None,
    frozen_calibration: bool = False,
    release_dataset_id: str = "pii_zh_synthetic_sota_release_eval_v2",
    release_dataset_version: str = "2.0.0",
    record_release_dataset_id: str | None = None,
    record_release_dataset_version: str | None = None,
    record_usage_policy: str = "post_cross_seed_selection_calibration_only",
    manifest_mutator: Callable[[dict[str, Any]], None] | None = None,
    post_hash_manifest_mutator: Callable[[dict[str, Any]], None] | None = None,
    expected_manifest_sha256: str | None = None,
    serialized_manifest_mutator: Callable[[str], str] | None = None,
    bind_manifest: bool | None = None,
    frozen_record_count: int = 2,
    expected_freeze_sha256: str | None = None,
    omit_supersession_freeze_arguments: bool = False,
) -> subprocess.CompletedProcess[str]:
    record_dataset_id = record_release_dataset_id or release_dataset_id
    record_dataset_version = record_release_dataset_version or release_dataset_version
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "calibration.json"
    diagnostics_path = tmp_path / "calibration.diagnostics.json"
    records = [
        _synthetic_record(
            doc_id="validation-one",
            text="张三登记",
            entity_text="张三",
            split=split,
            frozen_calibration=frozen_calibration,
            release_dataset_id=record_dataset_id,
            release_dataset_version=record_dataset_version,
            release_usage_policy=record_usage_policy,
        ),
        _synthetic_record(
            doc_id="validation-two",
            text="李四登记",
            entity_text="李四",
            split=split,
            frozen_calibration=frozen_calibration,
            release_dataset_id=record_dataset_id,
            release_dataset_version=record_dataset_version,
            release_usage_policy=record_usage_policy,
        ),
    ]
    predictions = [
        PredictionRecord(
            "validation-one",
            (
                Span(0, 2, "PERSON_NAME", score=0.9),
                Span(2, 4, "PERSON_NAME", score=0.8),
            ),
        ),
        PredictionRecord("validation-two", ()),
    ]
    assert frozen_record_count >= 2
    if frozen_calibration:
        for index in range(2, frozen_record_count):
            entity_text = f"合成人名{index:05d}"
            doc_id = f"calibration-{index:05d}"
            records.append(
                _synthetic_record(
                    doc_id=doc_id,
                    text=f"{entity_text}登记",
                    entity_text=entity_text,
                    split=split,
                    frozen_calibration=True,
                    release_dataset_id=record_dataset_id,
                    release_dataset_version=record_dataset_version,
                    release_usage_policy=record_usage_policy,
                )
            )
            predictions.append(
                PredictionRecord(
                    doc_id,
                    (Span(0, len(entity_text), "PERSON_NAME", score=0.9),),
                )
            )
    if allowed_labels is not None:
        records.append(
            _synthetic_record(
                doc_id="validation-email",
                text="user@example.invalid 登记",
                entity_text="user@example.invalid",
                split=split,
                label="EMAIL_ADDRESS",
                frozen_calibration=frozen_calibration,
                release_dataset_id=record_dataset_id,
                release_dataset_version=record_dataset_version,
                release_usage_policy=record_usage_policy,
            )
        )
        predictions.append(PredictionRecord("validation-email", ()))
    write_jsonl(records, gold_path)
    write_prediction_jsonl(predictions, prediction_path)
    repository_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        str(repository_root / "scripts/calibrate_predictions.py"),
        "--gold",
        str(gold_path),
        "--predictions",
        str(prediction_path),
        "--output",
        str(output_path),
        "--diagnostics",
        str(diagnostics_path),
        "--model-version",
        "fixture-model-v1",
        "--t0-recall-floor",
        "0.5",
        "--t1-recall-floor",
        "0.5",
        "--fit-temperature",
    ]
    for label in allowed_labels or ():
        command.extend(("--allowed-label", label))
    if bind_manifest if bind_manifest is not None else frozen_calibration:
        freeze_source = (
            repository_root
            / "configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json"
        )
        freeze_bytes = freeze_source.read_bytes()
        freeze_path = tmp_path / "supersession_freeze.json"
        freeze_path.write_bytes(freeze_bytes)
        manifest_path = tmp_path / "dataset_manifest.json"
        manifest_payload: dict[str, Any] = {
            "manifest_schema_version": 2,
            "dataset_id": release_dataset_id,
            "dataset_version": release_dataset_version,
            "publication_pool": "public_release_pool",
            "record_data_pool": "evaluation_only",
            "public_weight_training_allowed": False,
            "public_artifact_release_allowed": True,
            "freeze": {
                "frozen_at_utc": "2026-07-18T00:50:01Z",
                "frozen_before_any_v3_seed42_validation_result": True,
                "supersession_freeze_receipt": {
                    "path": (
                        "configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json"
                    ),
                    "file_sha256": (
                        "06ebdce5e288b0e7699a8225e28c4c4f0a1762e76f6967cf990b14384101f0f0"
                    ),
                    "receipt_id": "synthetic_sota_release_eval_v2_supersession_freeze",
                    "frozen_at_utc": "2026-07-18T00:50:01Z",
                    "chronology": {
                        "frozen_before_any_v3_seed42_validation_result": True,
                        "v1_confirmatory_calibration_withdrawn": True,
                        "v1_structural_pre_read_one_record_no_model_result": True,
                        "v2_zero_content_read_before_cross_seed_selection": True,
                    },
                },
            },
            "config": {
                "path": "configs/data/synthetic_sota_release_eval_v2.yaml",
                "sha256": ("23c5c15a27c92a2110b6bd5fc180f62dced81cdc17be61b44dbc30194e5dba30"),
            },
            "supersession": {
                "v1_structural_pre_read_one_record_no_model_result": True,
                "v1_confirmatory_calibration_withdrawn": True,
                "v2_zero_content_read_before_cross_seed_selection": True,
                "replacement_scope": "confirmatory_calibration_and_internal_evaluation",
            },
            "files": [
                {
                    "name": "calibration.jsonl",
                    "records": 10_000,
                    "sha256": sha256_file(gold_path),
                    "split": "calibration",
                },
                {
                    "name": "internal_evaluation.jsonl",
                    "records": 10_000,
                    "sha256": "1" * 64,
                    "split": "internal_evaluation",
                },
            ],
            "generation": {
                "generation_salt_sha256": (
                    "69ecd313a414459ece55274a7d7ce881537fb8f57e51940068b4a190a6fa437e"
                ),
                "requested_split_counts": {
                    "calibration": 10_000,
                    "internal_evaluation": 10_000,
                },
                "seeds_and_salt_independent_from_v1": True,
                "contains_real_personal_data": False,
                "protected_evaluation_data_read": False,
                "external_dataset_inputs": [],
                "network_used": False,
                "gpu_used": False,
                "jsonl_rows_printed": False,
            },
            "counts": {
                "by_split": {
                    "calibration": 10_000,
                    "internal_evaluation": 10_000,
                },
                "total": 20_000,
                "pii_free": 9_000,
                "hard_negative": 9_000,
                "core_labels": 24,
            },
            "audits": {"cross_corpus_isolation": {"all_collision_counts_zero": True}},
            "usage_policy": {
                "calibration": {
                    "checkpoint_or_model_selection_allowed": False,
                    "final_metric_claim_allowed": False,
                    "fusion_calibration_allowed": True,
                    "gradient_training_allowed": False,
                    "temperature_scaling_allowed": True,
                    "threshold_tuning_allowed": True,
                    "content_read_before_cross_seed_selection_allowed": False,
                },
                "internal_evaluation": {
                    "checkpoint_or_model_selection_allowed": False,
                    "final_metric_claim_allowed": True,
                    "fusion_calibration_allowed": False,
                    "gradient_training_allowed": False,
                    "result_informed_model_or_threshold_change_allowed": False,
                    "temperature_scaling_allowed": False,
                    "threshold_tuning_allowed": False,
                    "content_read_before_final_freeze_allowed": False,
                },
            },
        }
        if manifest_mutator is not None:
            manifest_mutator(manifest_payload)
        manifest = add_manifest_hash(manifest_payload)
        frozen_expected_hash = expected_manifest_sha256 or manifest["manifest_sha256"]
        if post_hash_manifest_mutator is not None:
            post_hash_manifest_mutator(manifest)
        serialized_manifest = json.dumps(manifest)
        if serialized_manifest_mutator is not None:
            serialized_manifest = serialized_manifest_mutator(serialized_manifest)
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
        command.extend(
            (
                "--dataset-manifest",
                str(manifest_path),
                "--expected-dataset-manifest-sha256",
                frozen_expected_hash,
            )
        )
        if not omit_supersession_freeze_arguments:
            command.extend(
                (
                    "--supersession-freeze",
                    str(freeze_path),
                    "--expected-supersession-freeze-sha256",
                    expected_freeze_sha256 or sha256_file(freeze_path),
                )
            )
    return subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_writes_bundle_and_self_hashed_raw_free_diagnostics(tmp_path: Path) -> None:
    completed = _run_calibration(tmp_path)
    assert completed.returncode == 0, completed.stderr

    output_path = tmp_path / "calibration.json"
    diagnostics_path = tmp_path / "calibration.diagnostics.json"
    bundle = CalibrationBundle.from_json(output_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    unsigned = dict(diagnostics)
    logical_hash = unsigned.pop("manifest_sha256")
    assert canonical_json_hash(unsigned) == logical_hash
    assert bundle.global_temperature == 1.0
    assert bundle.entity_thresholds["PERSON_NAME"] == 0.9
    assert diagnostics["temperature"]["status"] == "insufficient_statistics"
    assert diagnostics["inputs"]["un_emitted_gold_span_count"] == 1
    person = diagnostics["per_label"]["PERSON_NAME"]
    assert person["un_emitted_gold_fixed_fn"] == 1
    assert person["tp"] == 1
    assert person["fp"] == 0
    assert person["fn"] == 1
    serialized = diagnostics_path.read_text(encoding="utf-8")
    assert "张三" not in serialized
    assert "李四" not in serialized
    assert str(tmp_path) not in serialized
    assert output_path.stat().st_mode & 0o777 == 0o444
    assert diagnostics_path.stat().st_mode & 0o777 == 0o444


def test_cli_rejects_test_split_without_writing_outputs(tmp_path: Path) -> None:
    completed = _run_calibration(tmp_path, split="test")
    assert completed.returncode != 0
    assert "only the validation split" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_calibrates_a_frozen_reduced_label_protocol(tmp_path: Path) -> None:
    completed = _run_calibration(tmp_path, allowed_labels=("PERSON_NAME",))
    assert completed.returncode == 0, completed.stderr

    diagnostics = json.loads(
        (tmp_path / "calibration.diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["parameters"]["selected_labels"] == ["PERSON_NAME"]
    assert set(diagnostics["per_label"]) == {"PERSON_NAME"}
    assert diagnostics["inputs"]["gold_span_count"] == 2


def test_cli_accepts_only_a_manifest_bound_frozen_calibration_split(tmp_path: Path) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        frozen_record_count=10_000,
    )
    assert completed.returncode == 0, completed.stderr

    diagnostics = json.loads(
        (tmp_path / "calibration.diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["parameters"]["gold_role"] == "frozen_post_selection_calibration"
    assert diagnostics["dataset_contract"]["dataset_id"] == "pii_zh_synthetic_sota_release_eval_v2"
    assert (
        diagnostics["inputs"]["dataset_manifest_sha256"]
        == diagnostics["dataset_contract"]["manifest_sha256"]
    )
    serialized = "\n".join(
        (
            (tmp_path / "calibration.json").read_text(encoding="utf-8"),
            (tmp_path / "calibration.diagnostics.json").read_text(encoding="utf-8"),
        )
    )
    assert "张三" not in serialized
    assert "李四" not in serialized
    assert str(tmp_path) not in serialized


def test_cli_rejects_unbound_evaluation_only_calibration(tmp_path: Path) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        bind_manifest=False,
    )
    assert completed.returncode != 0
    assert "only the validation split" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_requires_the_callers_exact_frozen_manifest_hash(tmp_path: Path) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        expected_manifest_sha256="0" * 64,
    )
    assert completed.returncode != 0
    assert "does not match the frozen canonical hash" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_requires_the_exact_preregistered_supersession_freeze_hash(
    tmp_path: Path,
) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        expected_freeze_sha256="0" * 64,
    )
    assert completed.returncode != 0
    assert "not the pre-registered v2 hash" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_requires_supersession_freeze_with_the_dataset_manifest(tmp_path: Path) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        omit_supersession_freeze_arguments=True,
    )
    assert completed.returncode != 0
    assert "supersession-freeze" in completed.stderr
    assert "required together" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_explicitly_rejects_the_withdrawn_release_evaluation_v1(
    tmp_path: Path,
) -> None:
    def make_v1(payload: dict[str, Any]) -> None:
        payload["manifest_schema_version"] = 1

    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        release_dataset_id="pii_zh_synthetic_sota_release_eval_v1",
        release_dataset_version="1.0.0",
        manifest_mutator=make_v1,
    )
    assert completed.returncode != 0
    assert "release evaluation v1 is withdrawn" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_rejects_a_manifest_changed_after_self_hashing(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["dataset_version"] = "1.0.1"

    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        post_hash_manifest_mutator=mutate,
    )
    assert completed.returncode != 0
    assert "self-hash is invalid" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        ("model_selection", "calibration policy is not fail-closed"),
        ("gold_hash", "gold disagrees with the frozen dataset manifest"),
        ("record_count", "file entry has malformed identity or counts"),
        ("bool_record_count", "file entry has malformed identity or counts"),
        ("bool_schema", "schema version must be integer 2"),
        ("non_object_file", "file entry must be an object"),
        ("extra_usage_role", "lacks a calibration usage policy"),
        ("embedded_freeze", "v2 freeze/privacy/isolation contract is invalid"),
    ),
)
def test_cli_fails_closed_on_manifest_policy_identity_and_type_tampering(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        if tamper == "model_selection":
            payload["usage_policy"]["calibration"][  # type: ignore[index]
                "checkpoint_or_model_selection_allowed"
            ] = True
        elif tamper == "gold_hash":
            payload["files"][0]["sha256"] = "0" * 64
        elif tamper == "record_count":
            payload["files"][0]["records"] += 1
        elif tamper == "bool_record_count":
            payload["files"][0]["records"] = True
        elif tamper == "bool_schema":
            payload["manifest_schema_version"] = True
        elif tamper == "non_object_file":
            payload["files"][1] = "not-an-object"
        elif tamper == "extra_usage_role":
            payload["usage_policy"]["model_selection"] = {}
        elif tamper == "embedded_freeze":
            payload["freeze"]["frozen_before_any_v3_seed42_validation_result"] = False
        else:  # pragma: no cover - exhaustive fixture guard
            raise AssertionError(tamper)

    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        manifest_mutator=mutate,
    )
    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_id", "different-release-eval"),
        ("dataset_version", "2.0.1"),
    ),
)
def test_cli_requires_record_dataset_identity_to_match_manifest(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        record_release_dataset_id=(value if field == "dataset_id" else None),
        record_release_dataset_version=(value if field == "dataset_version" else None),
    )
    assert completed.returncode != 0
    assert "record disagrees with the frozen role contract" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_requires_frozen_record_role_metadata(tmp_path: Path) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        record_usage_policy="one_shot_final_internal_evaluation_only",
    )
    assert completed.returncode != 0
    assert "record disagrees with the frozen role contract" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_requires_gold_record_count_to_match_manifest(tmp_path: Path) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
    )
    assert completed.returncode != 0
    assert "document count disagrees with the frozen manifest" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_rejects_path_like_manifest_identity_even_when_records_match(
    tmp_path: Path,
) -> None:
    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        release_dataset_id="private/path/release-eval",
    )
    assert completed.returncode != 0
    assert "not the frozen release evaluation v2 identity" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    def duplicate_dataset_id(serialized: str) -> str:
        assert serialized.endswith("}")
        return serialized[:-1] + ', "dataset_id": "duplicate-release-eval"}'

    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        serialized_manifest_mutator=duplicate_dataset_id,
    )
    assert completed.returncode != 0
    assert "duplicate JSON key is forbidden" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


def test_cli_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    def add_nan(serialized: str) -> str:
        assert serialized.endswith("}")
        return serialized[:-1] + ', "malformed": NaN}'

    completed = _run_calibration(
        tmp_path,
        split="calibration",
        frozen_calibration=True,
        serialized_manifest_mutator=add_nan,
    )
    assert completed.returncode != 0
    assert "non-standard JSON constant is forbidden" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()


@pytest.mark.parametrize(
    "existing_name",
    ("calibration.json", "calibration.diagnostics.json"),
)
def test_cli_never_overwrites_existing_outputs(
    tmp_path: Path,
    existing_name: str,
) -> None:
    existing = tmp_path / existing_name
    existing.write_text("do-not-overwrite", encoding="utf-8")

    completed = _run_calibration(tmp_path)

    assert completed.returncode != 0
    assert "refusing to overwrite existing output" in completed.stderr
    assert existing.read_text(encoding="utf-8") == "do-not-overwrite"
    other_name = (
        "calibration.diagnostics.json"
        if existing_name == "calibration.json"
        else "calibration.json"
    )
    assert not (tmp_path / other_name).exists()
