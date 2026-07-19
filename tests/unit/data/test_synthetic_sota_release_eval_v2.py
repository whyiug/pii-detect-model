from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pii_zh.data.synthetic.sota_release_eval_v2 as release_eval_v2
from pii_zh.data.schema import DataPool
from pii_zh.data.synthetic.sota_release_eval_v1 import (
    ReleaseEvalSplitConfig,
    build_fingerprint_index,
    build_release_eval_records,
    load_release_eval_config,
)
from pii_zh.data.synthetic.sota_release_eval_v2 import (
    COLLISION_DIMENSIONS,
    CONFIG_FILE_SHA256,
    DATASET_ID,
    FROZEN_AT_UTC,
    GENERATION_SALT_SHA256,
    ReleaseEvalV2SplitConfig,
    SyntheticSotaReleaseEvalV2Error,
    audit_release_eval_v2_isolation,
    build_release_eval_v2_records,
    load_release_eval_v2_config,
    materialize_release_eval_v2_corpus,
)
from pii_zh.data.synthetic.sota_v1 import (
    build_synthetic_sota_records,
    load_synthetic_sota_config,
)

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs/data/synthetic_sota_release_eval_v2.yaml"
_V1_CONFIG = _ROOT / "configs/data/synthetic_sota_release_eval_v1.yaml"
_BASE_CONFIG = _ROOT / "configs/data/synthetic_sota_v1.yaml"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def small_v2_fixture():
    base = load_synthetic_sota_config(_BASE_CONFIG, repository_root=_ROOT)
    base = replace(base, split_counts={"train": 400, "validation": 300})
    training_records, _ = build_synthetic_sota_records(base)
    training_index = build_fingerprint_index(training_records)

    v1 = load_release_eval_config(_V1_CONFIG, repository_root=_ROOT)
    v1 = replace(
        v1,
        split_configs={
            "calibration": ReleaseEvalSplitConfig(records=300, seed=2026071801),
            "internal_evaluation": ReleaseEvalSplitConfig(records=300, seed=2026071802),
        },
        max_candidates_per_split=5_000,
    )
    v1_records, _ = build_release_eval_records(v1, reference_index=training_index)
    withdrawn_v1_index = build_fingerprint_index(v1_records)

    config = load_release_eval_v2_config(_CONFIG, repository_root=_ROOT)
    config = replace(
        config,
        split_configs={
            "calibration": ReleaseEvalV2SplitConfig(records=300, seed=2026071811),
            "internal_evaluation": ReleaseEvalV2SplitConfig(records=300, seed=2026071812),
        },
        max_candidates_per_split=5_000,
    )
    records, audit = build_release_eval_v2_records(
        config,
        training_development_index=training_index,
        withdrawn_v1_index=withdrawn_v1_index,
    )
    return config, training_index, withdrawn_v1_index, records, audit


def test_config_is_exact_pre_validation_freeze() -> None:
    config = load_release_eval_v2_config(_CONFIG, repository_root=_ROOT)
    assert config.config_sha256 == CONFIG_FILE_SHA256
    assert config.frozen_at_utc == FROZEN_AT_UTC
    assert config.generation_salt_sha256 == GENERATION_SALT_SHA256
    assert {split: (item.records, item.seed) for split, item in config.split_configs.items()} == {
        "calibration": (10_000, 2026071811),
        "internal_evaluation": (10_000, 2026071812),
    }
    assert config.pii_free_ratio == config.hard_negative_ratio == 0.45
    assert (
        config.usage_policy["calibration"]["content_read_before_cross_seed_selection_allowed"]
        is False
    )
    assert (
        config.usage_policy["internal_evaluation"]["content_read_before_final_freeze_allowed"]
        is False
    )


def test_config_fails_closed_on_any_file_drift(tmp_path: Path) -> None:
    tampered = tmp_path / "v2.yaml"
    tampered.write_bytes(_CONFIG.read_bytes() + b"\n")
    with pytest.raises(SyntheticSotaReleaseEvalV2Error, match="file hash mismatch"):
        load_release_eval_v2_config(tampered, repository_root=_ROOT)


def test_small_v2_is_deterministic_eval_only_full24_and_fully_isolated(
    small_v2_fixture,
) -> None:
    config, training_index, withdrawn_v1_index, records, audit = small_v2_fixture
    replay, replay_audit = build_release_eval_v2_records(
        config,
        training_development_index=training_index,
        withdrawn_v1_index=withdrawn_v1_index,
    )
    assert [item.to_dict() for item in records] == [item.to_dict() for item in replay]
    assert audit == replay_audit
    assert len(records) == 600
    assert audit["counts_by_split"] == {
        "calibration": 300,
        "internal_evaluation": 300,
    }
    assert audit["pii_free_counts_by_split"] == {
        "calibration": 135,
        "internal_evaluation": 135,
    }
    assert audit["hard_negative_counts_by_split"] == {
        "calibration": 135,
        "internal_evaluation": 135,
    }
    assert all(
        set(audit["label_counts_by_split"][split])
        == set(audit["label_counts_by_split"]["calibration"])
        for split in ("calibration", "internal_evaluation")
    )
    assert len(audit["label_counts_by_split"]["calibration"]) == 24
    isolation = audit["cross_corpus_isolation"]
    assert isolation["all_collision_counts_zero"] is True
    assert len(isolation["comparisons"]) == 5
    for comparison in isolation["comparisons"].values():
        assert comparison == {dimension: 0 for dimension in COLLISION_DIMENSIONS}
    assert all(
        item.data_pool is DataPool.EVALUATION_ONLY
        and item.public_weight_training_allowed is False
        and item.provenance.evaluation_only is True
        and item.metadata["generation_salt_sha256"] == GENERATION_SALT_SHA256
        for item in records
    )


def test_audit_detects_withdrawn_v1_doc_id_collision(small_v2_fixture) -> None:
    _, training_index, withdrawn_v1_index, records, _ = small_v2_fixture
    withdrawn_doc_id = next(iter(withdrawn_v1_index.doc_id))
    leaked = [replace(records[0], doc_id=withdrawn_doc_id), *records[1:]]
    with pytest.raises(SyntheticSotaReleaseEvalV2Error, match="collision audit failed"):
        audit_release_eval_v2_isolation(
            leaked,
            training_development_index=training_index,
            withdrawn_v1_index=withdrawn_v1_index,
        )


def test_small_materializer_binds_supersession_and_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_v2_fixture,
) -> None:
    config, training_index, withdrawn_v1_index, _, _ = small_v2_fixture
    training_audit = {
        "dataset_id": "pii_zh_synthetic_sota_v1",
        "dataset_version": "1.0.1",
        "manifest_file_sha256": "0" * 64,
        "manifest_sha256": "1" * 64,
        "file_sha256": {},
        "split_counts": {"train": 400, "validation": 300},
        "fingerprint_counts": training_index.counts(),
        "fingerprint_set_sha256": training_index.set_digests(),
        "local_path_recorded": False,
    }
    withdrawn_audit = {
        "dataset_id": "pii_zh_synthetic_sota_release_eval_v1",
        "dataset_version": "1.0.0",
        "manifest_file_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "file_sha256": {},
        "split_counts": {"calibration": 300, "internal_evaluation": 300},
        "fingerprint_counts": withdrawn_v1_index.counts(),
        "fingerprint_set_sha256": withdrawn_v1_index.set_digests(),
        "raw_text_or_entity_values_in_audit": False,
        "local_path_recorded": False,
    }
    monkeypatch.setattr(
        release_eval_v2, "load_release_eval_v2_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        release_eval_v2,
        "load_reference_fingerprint_index",
        lambda *args, **kwargs: (training_index, training_audit),
    )
    monkeypatch.setattr(
        release_eval_v2,
        "load_withdrawn_v1_fingerprint_index",
        lambda *args, **kwargs: (withdrawn_v1_index, withdrawn_audit),
    )
    output = tmp_path / "release-eval-v2"
    result = materialize_release_eval_v2_corpus(
        output,
        config_path=_CONFIG,
        training_development_dir=tmp_path,
        withdrawn_v1_dir=tmp_path,
        repository_root=_ROOT,
    )
    manifest = result["dataset_manifest"]
    receipt = result["supersession_receipt"]
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["record_data_pool"] == "evaluation_only"
    assert manifest["audits"]["cross_corpus_isolation"]["all_collision_counts_zero"]
    assert manifest["supersession"] == {
        "v1_structural_pre_read_one_record_no_model_result": True,
        "v1_confirmatory_calibration_withdrawn": True,
        "v2_zero_content_read_before_cross_seed_selection": True,
        "replacement_scope": "confirmatory_calibration_and_internal_evaluation",
    }
    assert receipt["chronology"]["v1_structural_pre_read_one_record_no_model_result"]
    assert receipt["chronology"]["v1_confirmatory_calibration_withdrawn"]
    assert receipt["chronology"]["v2_zero_content_read_before_cross_seed_selection"]
    disk_manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    unsigned_manifest = dict(disk_manifest)
    manifest_hash = unsigned_manifest.pop("manifest_sha256")
    assert manifest_hash == _canonical_hash(unsigned_manifest)
    disk_receipt = json.loads((output / "supersession_receipt.json").read_text(encoding="utf-8"))
    unsigned_receipt = dict(disk_receipt)
    receipt_hash = unsigned_receipt.pop("receipt_sha256")
    assert receipt_hash == _canonical_hash(unsigned_receipt)
    assert output.stat().st_mode & 0o777 == 0o555
    assert all(
        (output / name).stat().st_mode & 0o777 == 0o444
        for name in (
            "calibration.jsonl",
            "internal_evaluation.jsonl",
            "dataset_manifest.json",
            "supersession_receipt.json",
        )
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_release_eval_v2_corpus(
            output,
            config_path=_CONFIG,
            training_development_dir=tmp_path,
            withdrawn_v1_dir=tmp_path,
            repository_root=_ROOT,
        )
