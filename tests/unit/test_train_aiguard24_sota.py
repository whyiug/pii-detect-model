from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts import train_aiguard24_sota as runner

from pii_zh.data.schema import DataPool
from pii_zh.training.manifest import sha256_file


def _synthetic_record(
    *,
    source_id: str = "synthetic_sota_v1",
    license_name: str = "Apache-2.0",
) -> SimpleNamespace:
    return SimpleNamespace(
        split="train",
        data_pool=DataPool.PUBLIC_RELEASE,
        provenance=SimpleNamespace(
            source_id=source_id,
            generator_name="pii_zh_synthetic_sota_v1",
            synthetic=True,
            evaluation_only=False,
            license=license_name,
        ),
        public_weight_training_allowed=True,
        quality=SimpleNamespace(quality_gate=True, validators_passed=True),
    )


def test_requires_one_explicit_isolated_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert runner._require_isolated_cuda("cuda:0") == torch.device("cuda:0")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,0")
    with pytest.raises(runner.Aiguard24TrainingError, match="exactly one"):
        runner._require_isolated_cuda("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(runner.Aiguard24TrainingError, match="exactly one"):
        runner._require_isolated_cuda("cuda:1")


def test_training_defaults_to_release_native_full_attention(tmp_path: Path) -> None:
    args = runner._parser().parse_args(
        [
            "--source-model",
            str(tmp_path / "source"),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--validation",
            str(tmp_path / "validation.jsonl"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.attention_mode == "full"


def test_causal_attention_requires_explicit_opt_in(tmp_path: Path) -> None:
    args = runner._parser().parse_args(
        [
            "--source-model",
            str(tmp_path / "source"),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--validation",
            str(tmp_path / "validation.jsonl"),
            "--output",
            str(tmp_path / "output"),
            "--attention-mode",
            "causal",
        ]
    )
    assert args.attention_mode == "causal"


def test_template_overlap_policy_keeps_identity_and_value_groups_disjoint() -> None:
    train = SimpleNamespace(
        leakage_fingerprints={
            "document_id": frozenset({"train-doc"}),
            "template_group": frozenset({"shared-template"}),
            "entity_value_group": frozenset({"train-value"}),
            "source_group": frozenset(),
        }
    )
    validation = SimpleNamespace(
        leakage_fingerprints={
            "document_id": frozenset({"validation-doc"}),
            "template_group": frozenset({"shared-template"}),
            "entity_value_group": frozenset({"validation-value"}),
            "source_group": frozenset(),
        }
    )

    receipt = runner._assert_split_isolation(
        train,
        validation,
        policy="template-overlap-development-v1",
    )

    assert receipt["template_group_overlap_allowed"] is True
    assert receipt["collision_counts"]["template_group"] == 1
    assert receipt["release_test_eligible"] is False

    validation.leakage_fingerprints["entity_value_group"] = frozenset({"train-value"})
    with pytest.raises(runner.Aiguard24TrainingError, match="identity/value/source"):
        runner._assert_split_isolation(
            train,
            validation,
            policy="template-overlap-development-v1",
        )


def test_training_source_contract_rejects_benchmark_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "iter_jsonl", lambda _: iter([_synthetic_record()]))
    runner._assert_synthetic_open_contract(tmp_path / "unused", expected_split="train")

    monkeypatch.setattr(
        runner,
        "iter_jsonl",
        lambda _: iter([_synthetic_record(source_id="wan9yu/pii-bench-zh")]),
    )
    with pytest.raises(runner.Aiguard24TrainingError, match="quarantined"):
        runner._assert_synthetic_open_contract(tmp_path / "unused", expected_split="train")


def test_training_artifact_policy_failure_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_path = tmp_path / "sensitive-name.jsonl"

    def reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise runner.ArtifactPolicyError(str(sensitive_path))

    monkeypatch.setattr(runner, "assert_file_allowed", reject)
    with pytest.raises(runner.Aiguard24TrainingError, match="denied") as caught:
        runner._assert_training_artifact_allowed(sensitive_path)
    assert str(sensitive_path) not in str(caught.value)

    monkeypatch.setattr(
        runner,
        "iter_jsonl",
        lambda _: iter([_synthetic_record(license_name="Proprietary")]),
    )
    with pytest.raises(runner.Aiguard24TrainingError, match="synthetic/open"):
        runner._assert_synthetic_open_contract(tmp_path / "unused", expected_split="train")


def test_checkpoint_selection_receipt_is_path_free_and_content_bound(tmp_path: Path) -> None:
    root = tmp_path / "trainer-state"
    checkpoint = root / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"safe fixture adapter")

    receipt = runner._checkpoint_selection_receipt(
        trainer_state_root=root,
        best_model_checkpoint=str(checkpoint),
        best_global_step=20,
        best_metric=0.75,
        final_eval_metric=0.75,
    )

    assert receipt == {
        "selection_split": "validation",
        "selection_metric": "risk_weighted_score",
        "selected_checkpoint_id": "checkpoint-20",
        "selected_global_step": 20,
        "best_metric": 0.75,
        "final_validation_replay_metric": 0.75,
        "adapter_model_sha256": sha256_file(adapter),
    }
    assert str(tmp_path) not in str(receipt)


@pytest.mark.parametrize(
    ("checkpoint_name", "best_step", "best_metric", "final_metric", "message"),
    [
        ("checkpoint-20", 19, 0.75, 0.75, "step disagree"),
        ("checkpoint-20", 20, 0.75, 0.70, "metric disagrees"),
        ("not-a-checkpoint", 20, 0.75, 0.75, "escaped"),
    ],
)
def test_checkpoint_selection_receipt_fails_closed(
    tmp_path: Path,
    checkpoint_name: str,
    best_step: int,
    best_metric: float,
    final_metric: float,
    message: str,
) -> None:
    root = tmp_path / "trainer-state"
    checkpoint = root / checkpoint_name
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"safe fixture adapter")

    with pytest.raises(runner.Aiguard24TrainingError, match=message):
        runner._checkpoint_selection_receipt(
            trainer_state_root=root,
            best_model_checkpoint=str(checkpoint),
            best_global_step=best_step,
            best_metric=best_metric,
            final_eval_metric=final_metric,
        )
