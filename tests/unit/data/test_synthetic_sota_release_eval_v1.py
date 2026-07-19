from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pii_zh.data.synthetic.sota_release_eval_v1 as release_eval
from pii_zh.data.schema import DataPool
from pii_zh.data.synthetic.sota_release_eval_v1 import (
    COLLISION_DIMENSIONS,
    DATASET_ID,
    ReleaseEvalSplitConfig,
    SyntheticSotaReleaseEvalError,
    audit_release_eval_isolation,
    build_fingerprint_index,
    build_release_eval_records,
    load_release_eval_config,
    materialize_release_eval_corpus,
)
from pii_zh.data.synthetic.sota_v1 import (
    build_synthetic_sota_records,
    load_synthetic_sota_config,
)

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs/data/synthetic_sota_release_eval_v1.yaml"
_BASE_CONFIG = _ROOT / "configs/data/synthetic_sota_v1.yaml"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def small_release_fixture():
    base = load_synthetic_sota_config(_BASE_CONFIG, repository_root=_ROOT)
    base = replace(base, split_counts={"train": 400, "validation": 300})
    reference_records, _ = build_synthetic_sota_records(base)
    reference_index = build_fingerprint_index(reference_records)

    config = load_release_eval_config(_CONFIG, repository_root=_ROOT)
    config = replace(
        config,
        split_configs={
            "calibration": ReleaseEvalSplitConfig(records=300, seed=2026071801),
            "internal_evaluation": ReleaseEvalSplitConfig(records=300, seed=2026071802),
        },
        max_candidates_per_split=5_000,
    )
    records, audit = build_release_eval_records(config, reference_index=reference_index)
    return config, reference_records, reference_index, records, audit


def test_config_freezes_two_roles_reference_and_fail_closed_usage() -> None:
    config = load_release_eval_config(_CONFIG, repository_root=_ROOT)
    assert {
        split: (item.records, item.seed) for split, item in config.split_configs.items()
    } == {
        "calibration": (10_000, 2026071801),
        "internal_evaluation": (10_000, 2026071802),
    }
    assert config.pii_free_ratio == config.hard_negative_ratio == 0.45
    assert config.usage_policy["calibration"]["threshold_tuning_allowed"] is True
    assert (
        config.usage_policy["calibration"]["checkpoint_or_model_selection_allowed"]
        is False
    )
    assert config.usage_policy["internal_evaluation"]["threshold_tuning_allowed"] is False
    assert (
        config.usage_policy["internal_evaluation"][
            "result_informed_model_or_threshold_change_allowed"
        ]
        is False
    )


def test_config_rejects_internal_evaluation_threshold_tuning(tmp_path: Path) -> None:
    raw = _CONFIG.read_text(encoding="utf-8").replace(
        "  internal_evaluation:\n    gradient_training_allowed: false",
        "  internal_evaluation:\n    gradient_training_allowed: true",
    )
    path = tmp_path / "tampered.yaml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(SyntheticSotaReleaseEvalError, match="usage policy"):
        load_release_eval_config(path, repository_root=_ROOT)


def test_small_generation_is_deterministic_eval_only_24_class_and_isolated(
    small_release_fixture,
) -> None:
    config, _, reference_index, records, audit = small_release_fixture
    replay, replay_audit = build_release_eval_records(config, reference_index=reference_index)
    assert [record.to_dict() for record in records] == [record.to_dict() for record in replay]
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
    assert audit["cross_corpus_isolation"]["all_collision_counts_zero"] is True
    for comparison in audit["cross_corpus_isolation"]["comparisons"].values():
        assert comparison == {dimension: 0 for dimension in COLLISION_DIMENSIONS}
    assert all(
        record.data_pool is DataPool.EVALUATION_ONLY
        and record.public_weight_training_allowed is False
        and record.provenance.evaluation_only is True
        for record in records
    )


def test_cross_corpus_audit_detects_doc_id_leak(small_release_fixture) -> None:
    _, reference_records, reference_index, records, _ = small_release_fixture
    leaked = [replace(records[0], doc_id=reference_records[0].doc_id), *records[1:]]
    with pytest.raises(SyntheticSotaReleaseEvalError, match="collision audit failed"):
        audit_release_eval_isolation(leaked, reference_index=reference_index)


def test_small_materializer_is_immutable_and_manifest_binds_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_release_fixture,
) -> None:
    config, _, reference_index, _, _ = small_release_fixture
    reference_audit = {
        "dataset_id": "pii_zh_synthetic_sota_v1",
        "dataset_version": "1.0.1",
        "manifest_file_sha256": "0" * 64,
        "manifest_sha256": "1" * 64,
        "file_sha256": {},
        "split_counts": {"train": 400, "validation": 300},
        "fingerprint_counts": reference_index.counts(),
        "fingerprint_set_sha256": reference_index.set_digests(),
        "local_path_recorded": False,
    }
    monkeypatch.setattr(release_eval, "load_release_eval_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        release_eval,
        "load_reference_fingerprint_index",
        lambda *args, **kwargs: (reference_index, reference_audit),
    )
    output = tmp_path / "release-eval"
    manifest = materialize_release_eval_corpus(
        output,
        config_path=_CONFIG,
        reference_dir=tmp_path,
        repository_root=_ROOT,
    )
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["record_data_pool"] == "evaluation_only"
    assert manifest["public_weight_training_allowed"] is False
    assert manifest["public_artifact_release_allowed"] is True
    assert manifest["usage_policy"]["calibration"]["threshold_tuning_allowed"] is True
    assert (
        manifest["usage_policy"]["internal_evaluation"]["threshold_tuning_allowed"]
        is False
    )
    assert manifest["audits"]["cross_corpus_isolation"]["all_collision_counts_zero"]
    disk = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    unsigned = dict(disk)
    self_hash = unsigned.pop("manifest_sha256")
    assert self_hash == _canonical_hash(unsigned)
    assert output.stat().st_mode & 0o777 == 0o555
    assert all(
        (output / name).stat().st_mode & 0o777 == 0o444
        for name in (
            "calibration.jsonl",
            "internal_evaluation.jsonl",
            "dataset_manifest.json",
        )
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_release_eval_corpus(
            output,
            config_path=_CONFIG,
            reference_dir=tmp_path,
            repository_root=_ROOT,
        )
