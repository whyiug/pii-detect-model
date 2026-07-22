from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts import train_aiguard24_bie as runner
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    EvalPrediction,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForTokenClassification,
)

from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)
from pii_zh.inference import project_bie
from pii_zh.inference.aiguard import decode_bie_ids
from pii_zh.inference.project_bie import (
    PROJECT_CLOSED8_LABELS,
    ProjectBieInferenceError,
    ProjectBieSpanPredictor,
    closed8_project_bie_label_ids,
    load_local_project_bie_predictor,
    predecode_project_bie,
)
from pii_zh.models import aiguard24_bie
from pii_zh.models.aiguard24 import (
    AIGUARD24_TARGET_TO_SOURCE,
    AIGUARD_SOURCE_ENTITY_TYPES,
)
from pii_zh.models.aiguard24_bie import (
    Aiguard24BieInitializationError,
    build_core_bie_label_maps,
    configure_head_only_training,
    initialize_aiguard24_bie,
)
from pii_zh.taxonomy import load_taxonomy
from pii_zh.tokenization import configure_character_boundary_tokenizer
from pii_zh.training.bie_data import bio_labels_to_bie_labels, load_bie_aligned_jsonl
from pii_zh.training.config import LoraSettings
from pii_zh.training.manifest import (
    canonical_json_hash,
    output_artifact_fingerprint,
    write_training_manifest,
)
from pii_zh.training.trainer import (
    merge_parameter_efficient_model_for_release,
    prepare_parameter_efficient_model,
)


def _source_labels() -> list[str]:
    return [
        "O",
        *(f"B-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
        *(f"I-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
        *(f"E-{entity}" for entity in AIGUARD_SOURCE_ENTITY_TYPES),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(path: Path) -> dict[str, str]:
    return {
        name: _sha256(path / name)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }


def _make_source(path: Path) -> Qwen3ForTokenClassification:
    labels = _source_labels()
    id2label = dict(enumerate(labels))
    label2id = {label: index for index, label in id2label.items()}
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        architectures=["Qwen3ForTokenClassification"],
        classifier_dropout=0.0,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    torch.manual_seed(17)
    model = Qwen3ForTokenClassification(config)
    with torch.no_grad():
        values = torch.arange(len(labels) * config.hidden_size, dtype=torch.float32)
        model.score.weight.copy_(values.reshape(len(labels), config.hidden_size))
        model.score.bias.copy_(torch.arange(len(labels), dtype=torch.float32) + 0.25)
    model.save_pretrained(path, safe_serialization=True)
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return model


def _patch_fixture_hashes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aiguard24_bie, "_EXPECTED_SOURCE_HASHES", _source_hashes(path))


def test_builds_standard_73_label_inventory() -> None:
    label2id, id2label = build_core_bie_label_maps()
    core = [entity.name for entity in load_taxonomy().label_sets["core"]]
    expected = ["O"]
    for entity in core:
        expected.extend((f"B-{entity}", f"I-{entity}", f"E-{entity}"))
    assert list(label2id) == expected
    assert len(label2id) == 73
    assert id2label == {index: label for label, index in label2id.items()}


def test_initializes_causal_bie_and_copies_every_compatible_boundary_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    source = _make_source(source_path)
    _patch_fixture_hashes(source_path, monkeypatch)
    result = initialize_aiguard24_bie(source_path, seed=73)
    target = result.model
    label2id, id2label = build_core_bie_label_maps()

    assert type(target) is Qwen3ForTokenClassification
    assert target.config.label2id == label2id
    assert target.config.id2label == id2label
    assert target.config.num_labels == 73
    assert target.config.pii_attention_mode == "causal"
    assert target.config.pii_tagging_scheme == "BIE"
    assert target.config.pii_release_eligible is False
    assert target.config.pii_training_status == "initialized_untrained"
    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[key], value, rtol=0, atol=0)

    source_label2id = source.config.label2id
    torch.testing.assert_close(
        target.score.weight[label2id["O"]],
        source.score.weight[source_label2id["O"]],
        rtol=0,
        atol=0,
    )
    for prefix in ("B", "I", "E"):
        torch.testing.assert_close(
            target.score.weight[label2id[f"{prefix}-PERSON_NAME"]],
            source.score.weight[source_label2id[f"{prefix}-name"]],
            rtol=0,
            atol=0,
        )
        expected_bank = torch.stack(
            [
                source.score.weight[source_label2id[f"{prefix}-bank_card"]],
                source.score.weight[source_label2id[f"{prefix}-credit_card"]],
            ]
        ).mean(dim=0)
        torch.testing.assert_close(
            target.score.weight[label2id[f"{prefix}-BANK_CARD_NUMBER"]],
            expected_bank,
            rtol=0,
            atol=0,
        )

    audit = result.audit
    assert audit.mapped_target_labels == tuple(
        entity.name
        for entity in load_taxonomy().label_sets["core"]
        if entity.name in AIGUARD24_TARGET_TO_SOURCE
    )
    assert len(audit.mapped_target_labels) == 12
    assert len(audit.unmapped_target_labels) == 12
    assert audit.target_label_count == 73
    assert audit.target_tag_scheme == "BIE"
    assert audit.attention_mode == "causal"
    assert audit.outside_row_copied
    assert audit.mapped_head_rows_verified
    assert audit.unmapped_head_rows_preserved
    assert "source_path" not in json.dumps(audit.to_dict(), sort_keys=True)


def test_unmapped_rows_are_deterministic_and_output_is_standard_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    _make_source(source_path)
    _patch_fixture_hashes(source_path, monkeypatch)
    first = initialize_aiguard24_bie(source_path, seed=101).model
    second = initialize_aiguard24_bie(source_path, seed=101).model
    different = initialize_aiguard24_bie(source_path, seed=102).model
    unmapped_ids = [
        first.config.label2id[f"{prefix}-{label}"]
        for label in first.config.pii_lineage["unmapped_target_labels"]
        for prefix in ("B", "I", "E")
    ]
    torch.testing.assert_close(first.score.weight[unmapped_ids], second.score.weight[unmapped_ids])
    assert not torch.equal(first.score.weight[unmapped_ids], different.score.weight[unmapped_ids])

    output = tmp_path / "output"
    first.save_pretrained(output, safe_serialization=True)
    assert (output / "model.safetensors").is_file()
    reloaded = Qwen3ForTokenClassification.from_pretrained(
        output,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        weights_only=True,
    )
    assert reloaded.config.num_labels == 73
    assert reloaded.config.pii_tagging_scheme == "BIE"


def test_head_only_mode_freezes_everything_except_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    _make_source(source_path)
    _patch_fixture_hashes(source_path, monkeypatch)
    model = initialize_aiguard24_bie(source_path).model
    audit = configure_head_only_training(model)
    assert audit["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in model.score.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.score.parameters())
    assert not any(parameter.requires_grad for parameter in model.model.parameters())


def test_bio_to_bie_and_decoder_preserve_exact_character_spans() -> None:
    bio = (
        -100,
        "B-PERSON_NAME",
        "O",
        "B-PHONE_NUMBER",
        "I-PHONE_NUMBER",
        "B-ADDRESS",
        "I-ADDRESS",
        "I-ADDRESS",
        -100,
    )
    bie = bio_labels_to_bie_labels(bio)
    assert bie == (
        -100,
        "B-PERSON_NAME",
        "O",
        "B-PHONE_NUMBER",
        "E-PHONE_NUMBER",
        "B-ADDRESS",
        "I-ADDRESS",
        "E-ADDRESS",
        -100,
    )
    label2id, id2label = build_core_bie_label_maps()
    ids = [label2id["O"] if label == -100 else label2id[str(label)] for label in bie]
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 0)]
    spans = decode_bie_ids(ids, offsets, id2label)
    assert [(span.entity_type, span.start, span.end) for span in spans] == [
        ("PERSON_NAME", 0, 1),
        ("PHONE_NUMBER", 2, 4),
        ("ADDRESS", 4, 7),
    ]


class _CharacterTokenizer:
    is_fast = True

    def __call__(self, text: str, **_: object) -> dict[str, list[object]]:
        return {
            "input_ids": [1, *range(10, 10 + len(text)), 2],
            "attention_mask": [1] * (len(text) + 2),
            "offset_mapping": [(0, 0), *[(i, i + 1) for i in range(len(text))], (0, 0)],
            "special_tokens_mask": [1, *([0] * len(text)), 1],
        }


def _record(*, doc_id: str, split: str, hard_negative: bool) -> DocumentRecord:
    text = "普通文本" if hard_negative else "张三到访"
    entities = (
        ()
        if hard_negative
        else (
            EntitySpan(
                start=0,
                end=2,
                label="PERSON_NAME",
                text="张三",
                source="unit",
            ),
        )
    )
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=entities,
        language="zh-Hans-CN",
        domain="synthetic",
        scene="unit",
        provenance=Provenance(
            source_id="unit",
            source_kind="synthetic",
            source_revision="1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit",
            generator_version="1",
        ),
        quality=QualityMetadata(
            tier=(QualityTier.HARD_NEGATIVE if hard_negative else QualityTier.SYNTHETIC_VALIDATED),
            quality_gate=True,
            validators_passed=True,
            review_status="validated",
        ),
        public_weight_training_allowed=True,
        generator_version="1",
        split=split,
    )


def _preflight_records(*, split: str, prefix: str) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    provenance = Provenance(
        source_id=f"unit-preflight-{split}",
        source_kind="synthetic",
        source_revision="1",
        license="Apache-2.0",
        synthetic=True,
        generator_name="unit-preflight",
        generator_version="1",
    )
    for index, entity in enumerate(load_taxonomy().label_sets["core"]):
        text = "甲乙丙尾"
        records.append(
            DocumentRecord.create(
                doc_id=f"{prefix}-positive-{index}",
                text=text,
                entities=(
                    EntitySpan(
                        start=0,
                        end=3,
                        label=entity.name,
                        text=text[:3],
                        source="unit",
                    ),
                ),
                language="zh-Hans-CN",
                domain="synthetic",
                scene="preflight-unit",
                provenance=provenance,
                quality=QualityMetadata(
                    tier=QualityTier.SYNTHETIC_VALIDATED,
                    quality_gate=True,
                    validators_passed=True,
                    review_status="validated",
                ),
                public_weight_training_allowed=True,
                generator_version="1",
                split=split,
            )
        )
    for index in range(16):
        records.append(
            DocumentRecord.create(
                doc_id=f"{prefix}-hard-negative-{index}",
                text="绝密原文",
                entities=(),
                language="zh-Hans-CN",
                domain="synthetic",
                scene="preflight-unit",
                provenance=provenance,
                quality=QualityMetadata(
                    tier=QualityTier.HARD_NEGATIVE,
                    quality_gate=True,
                    validators_passed=True,
                    review_status="validated",
                ),
                public_weight_training_allowed=True,
                generator_version="1",
                split=split,
            )
        )
    return records


def test_bie_runner_defaults_to_dev_and_preflight_needs_no_output(tmp_path: Path) -> None:
    args = runner._parser().parse_args(
        [
            "--source-model",
            str(tmp_path / "source"),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--validation",
            str(tmp_path / "dev.jsonl"),
            "--dataset-manifest",
            str(tmp_path / "dataset_manifest.json"),
            "--preflight-only",
        ]
    )
    runner._validate_args(args)
    assert args.validation_split_name == "dev"
    assert args.output is None

    training_args = runner._parser().parse_args(
        [
            "--source-model",
            str(tmp_path / "source"),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--validation",
            str(tmp_path / "dev.jsonl"),
            "--dataset-manifest",
            str(tmp_path / "dataset_manifest.json"),
            "--output",
            str(tmp_path / "output"),
            "--preflight-receipt",
            str(tmp_path / "receipt.json"),
        ]
    )
    with pytest.raises(runner.Aiguard24BieTrainingError, match="only with"):
        runner._validate_args(training_args)


def test_preflight_dev_is_raw_text_free_and_never_uses_cuda_model_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    dataset_manifest = tmp_path / "dataset_manifest.json"
    output = tmp_path / "must-not-exist"
    preflight_receipt = tmp_path / "preflight-receipt.json"
    write_jsonl(_preflight_records(split="train", prefix="train"), train)
    write_jsonl(_preflight_records(split="dev", prefix="dev"), dev)
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_load_tokenizer", lambda _: _CharacterTokenizer())
    artifact_checks: list[str] = []
    monkeypatch.setattr(
        runner,
        "_assert_training_artifact_allowed",
        lambda path: artifact_checks.append(path.name),
    )
    manifest_binding = {
        "dataset_id": "pii_zh_domain_randomized_v7",
        "dataset_version": "7.1.0",
        "generator_version": "7.1.0",
        "materializer_version": "7.1.0",
        "manifest_file_sha256": runner.sha256_file(dataset_manifest),
        "logical_manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        runner,
        "_dataset_manifest_contract",
        lambda *args, **kwargs: manifest_binding,
    )

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("preflight entered a training-only operation")

    monkeypatch.setattr(runner, "_require_isolated_cuda", forbidden)
    monkeypatch.setattr(runner, "initialize_aiguard24_bie", forbidden)

    assert (
        runner.main(
            [
                "--source-model",
                str(source),
                "--train",
                str(train),
                "--validation",
                str(dev),
                "--dataset-manifest",
                str(dataset_manifest),
                "--output",
                str(output),
                "--preflight-only",
                "--preflight-receipt",
                str(preflight_receipt),
            ]
        )
        == 0
    )

    rendered = capsys.readouterr().out
    stdout_summary = json.loads(rendered)
    receipt_bytes = preflight_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    unsigned = dict(receipt)
    claimed_hash = unsigned.pop("receipt_sha256")
    assert claimed_hash == runner.canonical_json_hash(unsigned)
    assert stdout_summary == {
        "label_count": 73,
        "receipt_file_sha256": runner.sha256_file(preflight_receipt),
        "receipt_sha256": claimed_hash,
        "receipt_written": True,
        "status": "preflight_passed",
        "validation_split_name": "dev",
    }
    assert receipt["status"] == "passed"
    assert receipt["validation_split_name"] == "dev"
    assert receipt["label_count"] == receipt["supervised_label_count"] == 73
    assert receipt["datasets"]["independent_validation"]["record_split"] == "dev"
    assert receipt["dataset_manifest"] == manifest_binding
    assert receipt["checks"] == {
        "artifact_policy": "passed",
        "bie_tokenizer_alignment": "passed",
        "dataset_hashes_stable": True,
        "full_73_logit_supervision": True,
        "strict_split_isolation": "passed",
        "train_validation_contracts": "passed",
    }
    assert receipt["privacy"] == {
        "contains_document_ids": False,
        "contains_entity_values": False,
        "contains_paths": False,
        "contains_raw_text": False,
    }
    assert receipt["execution"] == {
        "cuda_required": False,
        "model_initialized": False,
        "preflight_receipt_persisted": True,
        "training_output_created": False,
    }
    assert "绝密原文" not in rendered
    assert "甲乙丙" not in rendered
    assert "绝密原文".encode() not in receipt_bytes
    assert "甲乙丙".encode() not in receipt_bytes
    assert not output.exists()
    assert preflight_receipt.stat().st_mode & 0o777 == 0o444
    assert sorted(artifact_checks) == ["dataset_manifest.json", "dev.jsonl", "train.jsonl"]
    with pytest.raises(runner.Aiguard24BieTrainingError, match="already exists"):
        runner._write_preflight_receipt(receipt, preflight_receipt)
    assert preflight_receipt.read_bytes() == receipt_bytes


def _write_v7_manifest(path: Path, *, train: Path, dev: Path) -> dict[str, object]:
    manifest: dict[str, object] = {
        "dataset_id": "pii_zh_domain_randomized_v7",
        "dataset_version": "7.1.0",
        "materializer_version": "7.1.0",
        "data_pool": "public_release_pool",
        "public_weight_training_allowed": True,
        "canonicalization_fixture": "中文",
        "generation": {
            "version": "7.1.0",
            "external_dataset_inputs": [],
            "pii_free_ratio": 0.40,
            "natural_positive_ratio": 0.50,
        },
        "clean_room": {
            "benchmark_rows_read": False,
            "benchmark_predictions_read": False,
            "benchmark_templates_read": False,
            "external_dataset_inputs": [],
            "repository_authored_grammars_only": True,
        },
        "counts": {
            "total": 56_000,
            "by_split": {"train": 48_000, "dev": 8_000},
            "pii_free_by_split": {"train": 19_200, "dev": 3_200},
        },
        "audits": {
            "isolation": {
                "cross_split_collisions": 0,
                "all_required_collision_counts_zero": True,
            },
            "coverage": {
                "natural_positive": {
                    "required_ratio": 0.50,
                    "by_split": {
                        "train": {"natural_positive_ratio": 0.50},
                        "dev": {"natural_positive_ratio": 0.50},
                    },
                }
            },
        },
        "files": {
            "train": {
                "name": train.name,
                "records": 48_000,
                "bytes": train.stat().st_size,
                "sha256": runner.sha256_file(train),
            },
            "dev": {
                "name": dev.name,
                "records": 8_000,
                "bytes": dev.stat().st_size,
                "sha256": runner.sha256_file(dev),
            },
        },
    }
    manifest["manifest_sha256"] = runner._dataset_manifest_logical_hash(manifest)
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def test_v7_dataset_manifest_binds_logical_and_file_identity(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    train.write_bytes(b"train fixture\n")
    dev.write_bytes(b"dev fixture\n")
    manifest = _write_v7_manifest(manifest_path, train=train, dev=dev)

    receipt = runner._dataset_manifest_contract(
        manifest_path,
        train_path=train,
        validation_path=dev,
        train_hash=runner.sha256_file(train),
        validation_hash=runner.sha256_file(dev),
        train_contract={"documents": 48_000, "pii_free": 19_200},
        validation_contract={"documents": 8_000, "pii_free": 3_200},
        validation_split_name="dev",
    )

    assert receipt["manifest_file_sha256"] == runner.sha256_file(manifest_path)
    assert receipt["logical_manifest_sha256"] == manifest["manifest_sha256"]
    assert receipt["dataset_version"] == "7.1.0"
    assert receipt["generator_version"] == "7.1.0"
    assert receipt["materializer_version"] == "7.1.0"
    assert receipt["files"]["train"] == {
        "name": "train.jsonl",
        "records": 48_000,
        "sha256": runner.sha256_file(train),
    }
    manifest["dataset_version"] = "7.0.0"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = runner._dataset_manifest_logical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.Aiguard24BieTrainingError, match="approved v7"):
        runner._dataset_manifest_contract(
            manifest_path,
            train_path=train,
            validation_path=dev,
            train_hash=runner.sha256_file(train),
            validation_hash=runner.sha256_file(dev),
            train_contract={"documents": 48_000, "pii_free": 19_200},
            validation_contract={"documents": 8_000, "pii_free": 3_200},
            validation_split_name="dev",
        )

    manifest["dataset_version"] = "7.1.0"
    manifest["clean_room"]["benchmark_rows_read"] = True  # type: ignore[index]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = runner._dataset_manifest_logical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.Aiguard24BieTrainingError, match="clean-room"):
        runner._dataset_manifest_contract(
            manifest_path,
            train_path=train,
            validation_path=dev,
            train_hash=runner.sha256_file(train),
            validation_hash=runner.sha256_file(dev),
            train_contract={"documents": 48_000, "pii_free": 19_200},
            validation_contract={"documents": 8_000, "pii_free": 3_200},
            validation_split_name="dev",
        )


def test_bie_loader_uses_exact_boundaries_and_tracks_hard_negative_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        [
            _record(doc_id="positive", split="train", hard_negative=False),
            _record(doc_id="negative", split="train", hard_negative=True),
        ],
        path,
    )
    label2id, _ = build_core_bie_label_maps()
    dataset, summary = load_bie_aligned_jsonl(
        path,
        tokenizer=_CharacterTokenizer(),
        taxonomy=load_taxonomy(),
        bie_label2id=label2id,
        max_length=32,
        expected_split="train",
    )
    assert dataset.documents[0].labels == (
        -100,
        label2id["B-PERSON_NAME"],
        label2id["E-PERSON_NAME"],
        label2id["O"],
        label2id["O"],
        -100,
    )
    assert dataset.hard_negative_indices == (1,)
    assert summary.unalignable_boundary_count == 0


def test_independent_dev_selection_uses_closed8_primary_and_open24_tie_break(
    tmp_path: Path,
) -> None:
    records = _preflight_records(split="dev", prefix="metrics")
    selected = [
        next(record for record in records if record.entities[0].label == "PERSON_NAME"),
        next(record for record in records if record.entities[0].label == "WECHAT_ID"),
        next(record for record in records if not record.entities),
    ]
    path = tmp_path / "metrics-dev.jsonl"
    write_jsonl(selected, path)
    label2id, id2label = build_core_bie_label_maps()
    dataset, _ = load_bie_aligned_jsonl(
        path,
        tokenizer=_CharacterTokenizer(),
        taxonomy=load_taxonomy(),
        bie_label2id=label2id,
        max_length=32,
        expected_split="dev",
    )
    metric = runner.IndependentBieDevMetrics(
        dataset,
        id2label=id2label,
        labels={entity.name for entity in load_taxonomy().label_sets["core"]},
        min_precision=0.0,
        min_recall=0.0,
        max_pii_free_fpr=1.0,
        max_hard_negative_fpr=1.0,
    )

    def prediction(*, person_correct: bool, wechat_correct: bool) -> EvalPrediction:
        width = max(len(document.labels) for document in dataset.documents)
        logits = np.full((len(dataset.documents), width, 73), -100.0, dtype=np.float32)
        logits[..., label2id["O"]] = 0.0
        for row, enabled in enumerate((person_correct, wechat_correct, False)):
            if not enabled:
                continue
            for column, label_id in enumerate(dataset.documents[row].labels):
                if label_id != -100:
                    logits[row, column, label_id] = 10.0
        return EvalPrediction(
            predictions=logits,
            label_ids=np.zeros((len(dataset.documents), width), dtype=np.int64),
        )

    closed_primary = metric(prediction(person_correct=True, wechat_correct=False))
    open_only = metric(prediction(person_correct=False, wechat_correct=True))
    perfect = metric(prediction(person_correct=True, wechat_correct=True))

    assert (
        closed_primary["independent_dev_selection_score"]
        > open_only["independent_dev_selection_score"]
    )
    assert open_only["strict_closed8_f1"] == 0.0
    assert perfect["strict_closed8_precision"] == 1.0
    assert perfect["strict_closed8_recall"] == 1.0
    assert perfect["strict_closed8_f1"] == 1.0
    assert perfect["strict_open24_precision"] == 1.0
    assert perfect["strict_open24_recall"] == 1.0
    assert perfect["strict_open24_f1"] == 1.0
    assert perfect["strict_open24_macro_f1"] == pytest.approx(2 / 24)
    assert perfect["independent_dev_selection_score"] == pytest.approx(
        2.0 + 1.0 + 1.0e-6 * (2 / 24)
    )
    assert runner._SELECTION_FORMULA == (
        "2*dev_gate_passed + strict_closed8_f1 + "
        "1e-6*strict_open24_macro_f1 - "
        "1e-9*(pii_free_document_fpr + hard_negative_document_fpr)"
    )


@pytest.mark.parametrize(
    ("update_mode", "weight_name", "metadata_name"),
    [
        ("head_only", "model.safetensors", "config.json"),
        ("lora", "adapter_model.safetensors", "adapter_config.json"),
    ],
)
def test_checkpoint_receipt_binds_exact_update_mode_inventory(
    tmp_path: Path,
    update_mode: str,
    weight_name: str,
    metadata_name: str,
) -> None:
    trainer_state = tmp_path / "trainer-state"
    checkpoint = trainer_state / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / weight_name).write_bytes(b"selected weights")
    (checkpoint / metadata_name).write_text("{}\n", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    receipt = runner._checkpoint_receipt(
        trainer_state,
        update_mode=update_mode,
        checkpoint=str(checkpoint),
        best_metric=3.25,
        replay_metric=3.25,
    )

    assert receipt["update_mode"] == update_mode
    assert set(receipt["safetensor_sha256"]) == {weight_name}
    assert set(receipt["metadata_sha256"]) == {metadata_name, "trainer_state.json"}

    with pytest.raises(runner.Aiguard24BieTrainingError, match="dev replay"):
        runner._checkpoint_receipt(
            trainer_state,
            update_mode=update_mode,
            checkpoint=str(checkpoint),
            best_metric=float("inf"),
            replay_metric=float("inf"),
        )

    (checkpoint / weight_name).rename(checkpoint / "unrelated.safetensors")
    with pytest.raises(runner.Aiguard24BieTrainingError, match="inventory"):
        runner._checkpoint_receipt(
            trainer_state,
            update_mode=update_mode,
            checkpoint=str(checkpoint),
            best_metric=3.25,
            replay_metric=3.25,
        )


def _completed_candidate_manifest_gate() -> dict[str, object]:
    label2id, _ = build_core_bie_label_maps()
    return {
        "schema_version": 1,
        "release_eligible": False,
        "validation_split_name": "dev",
        "update_mode": "lora",
        "label2id": label2id,
        "recipe": {
            "attention_mode": "causal",
            "tag_scheme": "BIE",
            "update_mode": "lora",
            "checkpoint_selection_split": "independent_validation",
            "checkpoint_selection_metric": "independent_dev_selection_score",
        },
        "independent_validation": {"eval_dev_gate_passed": 1.0},
        "benchmark_isolation": {
            "public_test_read_for_training": False,
            "public_test_read_for_validation": False,
            "public_test_read_for_checkpoint_selection": False,
            "pii_bench_zh_read": False,
        },
        "checkpoint_selection": {
            "selection_split": "independent_validation",
            "selection_metric": "independent_dev_selection_score",
            "update_mode": "lora",
            "selected_checkpoint_id": "checkpoint-7",
            "selected_global_step": 7,
            "best_metric": 3.25,
            "final_validation_replay_metric": 3.25,
            "safetensor_sha256": {"adapter_model.safetensors": "a" * 64},
        },
    }


def test_project_bie_manifest_gate_fails_closed_on_infeasible_or_wrong_checkpoint() -> None:
    manifest = _completed_candidate_manifest_gate()
    assert project_bie._verify_completed_candidate_manifest(manifest)

    manifest["independent_validation"] = {"eval_dev_gate_passed": 0.0}
    assert not project_bie._verify_completed_candidate_manifest(manifest)

    manifest = _completed_candidate_manifest_gate()
    checkpoint = manifest["checkpoint_selection"]
    assert isinstance(checkpoint, dict)
    checkpoint["safetensor_sha256"] = {"model.safetensors": "a" * 64}
    assert not project_bie._verify_completed_candidate_manifest(manifest)

    manifest = _completed_candidate_manifest_gate()
    checkpoint = manifest["checkpoint_selection"]
    assert isinstance(checkpoint, dict)
    checkpoint["final_validation_replay_metric"] = 3.0
    assert not project_bie._verify_completed_candidate_manifest(manifest)


def test_project_bie_loader_rejects_symlink_root_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "real-model"
    root.mkdir()
    alias = tmp_path / "model-link"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(
        project_bie.Qwen3ForTokenClassification,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("unsafe model deserialization was reached"),
    )

    with pytest.raises(ProjectBieInferenceError, match="regular local directory"):
        load_local_project_bie_predictor(alias)


def test_merged_lora_bie73_package_is_standalone_and_loadable_on_cpu(tmp_path: Path) -> None:
    label2id, id2label = build_core_bie_label_maps()
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_labels=73,
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager",
    )
    config.pii_attention_mode = "causal"
    config.pii_tagging_scheme = "BIE"
    model = Qwen3ForTokenClassification(config)
    before = model.score.weight.detach().clone()
    peft_model = prepare_parameter_efficient_model(
        model,
        fine_tuning="lora",
        lora=LoraSettings(rank=2, alpha=4, target_modules=("q_proj",)),
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if parameter.requires_grad and "score" in name:
                parameter.add_(1.0)
    final_model = merge_parameter_efficient_model_for_release(
        peft_model,
        fine_tuning="lora",
        attention_mode="causal",
    )
    final_model.config.pii_tagging_scheme = "BIE"
    assert not torch.equal(final_model.score.weight, before)
    final_model.save_pretrained(tmp_path, safe_serialization=True)

    backend = Tokenizer(BPE(vocab={"<unk>": 0}, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.eos_token = tokenizer.unk_token
    configure_character_boundary_tokenizer(tokenizer).save_pretrained(tmp_path)

    manifest = _completed_candidate_manifest_gate()
    manifest.update(
        {
            "manifest_type": "aiguard24_native_causal_bie_training_v1",
            "status": "completed",
            "attention_mode": "causal",
            "tag_scheme": "BIE",
            "output_artifact": output_artifact_fingerprint(tmp_path),
        }
    )
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_training_manifest(manifest, tmp_path)

    loaded = load_local_project_bie_predictor(tmp_path, device="cpu", micro_batch_size=1)
    assert type(loaded.model) is Qwen3ForTokenClassification
    assert loaded.model.config.num_labels == 73
    assert not any(tmp_path.glob("adapter_*"))
    torch.testing.assert_close(loaded.model.score.weight, final_model.score.weight)


def test_closed8_predecode_mask_contains_o_and_all_three_boundary_states() -> None:
    label2id, id2label = build_core_bie_label_maps()
    allowed = closed8_project_bie_label_ids(id2label)
    assert len(allowed) == 25
    assert {id2label[label_id] for label_id in allowed} == {
        "O",
        *(f"{prefix}-{label}" for label in PROJECT_CLOSED8_LABELS for prefix in ("B", "I", "E")),
    }
    logits = torch.zeros((1, 1, 73), dtype=torch.float32)
    logits[..., label2id["B-WECHAT_ID"]] = 100
    logits[..., label2id["B-PERSON_NAME"]] = 10
    predicted, scores = predecode_project_bie(logits, allowed_label_ids=allowed)
    assert predicted.item() == label2id["B-PERSON_NAME"]
    assert 0 < scores.item() <= 1


class _FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        assert all(len(text) == 2 for text in texts)
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
            "offset_mapping": torch.tensor([[[0, 1], [1, 2]]] * batch, dtype=torch.long),
        }


class _FakeBieModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        label2id, id2label = build_core_bie_label_maps()
        self.label2id = label2id
        self.config = SimpleNamespace(
            model_type="qwen3",
            pii_attention_mode="causal",
            pii_tagging_scheme="BIE",
            id2label=id2label,
        )

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        logits = torch.zeros((*input_ids.shape, 73), device=input_ids.device)
        logits[..., self.label2id["B-WECHAT_ID"]] = 100
        logits[:, 0, self.label2id["B-PERSON_NAME"]] = 10
        logits[:, 1, self.label2id["E-PERSON_NAME"]] = 10
        return SimpleNamespace(logits=logits)


def test_project_bie_predictor_masks_before_decode_and_emits_project_labels() -> None:
    predictor = ProjectBieSpanPredictor(
        _FakeBieModel(),
        _FakeTokenizer(),
        allowed_project_labels=PROJECT_CLOSED8_LABELS,
    )
    prediction = predictor.predict("张三")
    assert prediction[0].pop("score") > 0.99
    assert prediction == [
        {
            "entity_type": "PERSON_NAME",
            "start": 0,
            "end": 2,
            "offset_mode": "relative",
            "metadata": {
                "boundary_refined": False,
                "protocol_id": predictor.protocol_id,
            },
        }
    ]


def test_rejects_hash_mismatch_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    _make_source(source_path)
    hashes = _source_hashes(source_path)
    hashes["config.json"] = "0" * 64
    monkeypatch.setattr(aiguard24_bie, "_EXPECTED_SOURCE_HASHES", hashes)
    with pytest.raises(Aiguard24BieInitializationError, match="pinned AIguard revision"):
        initialize_aiguard24_bie(source_path)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_rejects_invalid_seed(seed: object, tmp_path: Path) -> None:
    with pytest.raises(Aiguard24BieInitializationError, match="seed"):
        initialize_aiguard24_bie(tmp_path, seed=seed)  # type: ignore[arg-type]
