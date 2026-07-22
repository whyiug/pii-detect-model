from __future__ import annotations

import hashlib
import json
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file
from scripts import build_b3_full_bie73_hf_package as package_builder
from scripts import build_release
from scripts import verify_b3_full_bie73_hf_package as package_verifier
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from pii_zh.models.qwen3_bi import Qwen3BiConfig, Qwen3BiForTokenClassification


@dataclass(frozen=True)
class PackageInputs:
    source: Path
    selection_receipt: Path
    model_export_receipt: Path
    readme: Path
    license_file: Path
    notice: Path
    third_party_notices: Path
    security: Path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_kwargs(inputs: PackageInputs, output: Path) -> dict[str, Path]:
    return {
        "source_model_dir": inputs.source,
        "selection_receipt": inputs.selection_receipt,
        "model_export_receipt": inputs.model_export_receipt,
        "output_dir": output,
        "readme": inputs.readme,
        "license_file": inputs.license_file,
        "notice": inputs.notice,
        "third_party_notices": inputs.third_party_notices,
        "security": inputs.security,
    }


@pytest.fixture
def package_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PackageInputs:
    source = tmp_path / "source-model"
    source.mkdir()
    label2id, id2label = package_builder.expected_label_maps()
    model_config = Qwen3BiConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        num_labels=73,
        id2label={int(key): value for key, value in id2label.items()},
        label2id=label2id,
        pad_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3BiForTokenClassification(model_config).cpu().eval()
    save_file(model.state_dict(), source / "model.safetensors")
    config_document = model_config.to_dict()
    config_document.update(
        {
            "architecture_version": "qwen3_bi_token_cls_v1",
            "architectures": ["Qwen3BiForTokenClassification"],
            "auto_map": package_builder.EXPECTED_AUTO_MAP,
            "bi_attention_backend": "sdpa",
            "id2label": id2label,
            "label2id": label2id,
            "layer_types": ["full_attention"],
            "model_type": "qwen3_bi",
            "num_labels": 73,
            "pii_attention_mode": "full",
            "pii_release_eligible": False,
            "pii_tagging_scheme": "BIE",
            "pii_taxonomy_version": "1.0.0",
            "use_cache": False,
        }
    )
    _write_json(
        source / "config.json",
        config_document,
    )
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "社区": 3,
                "/data-token": 4,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        model_max_length=32,
    )
    tokenizer.save_pretrained(source)
    # Transformers 5 may omit this legacy sidecar for a fast tokenizer, while
    # the frozen BIE73 package contract still requires the explicit map.
    _write_json(source / "special_tokens_map.json", tokenizer.special_tokens_map)
    training_manifest = {
        "attention_mode": "full",
        "label2id": label2id,
        "manifest_type": "aiguard24_full_attention_bie_training_v1",
        "output_artifact": build_release.output_artifact_fingerprint(source),
        "release_eligible": False,
        "status": "completed",
        "tag_scheme": "BIE",
        "taxonomy_version": "1.0.0",
    }
    training_manifest["manifest_sha256"] = package_builder.canonical_json_hash(training_manifest)
    _write_json(source / "training_manifest.json", training_manifest)

    training_file_sha256 = hashlib.sha256(
        (source / "training_manifest.json").read_bytes()
    ).hexdigest()
    selected = {
        "candidate_id": "tiny_b3_fixture",
        "seed": 42,
        "epoch": 1,
        "global_step": 7,
        "training_manifest": {
            "file_sha256": training_file_sha256,
            "manifest_sha256": training_manifest["manifest_sha256"],
            "status": "completed",
        },
        "selected_checkpoint": {
            "checkpoint_id": "checkpoint-7",
            "update_mode": "lora",
            "artifact_sha256": {
                "adapter_config.json": "1" * 64,
                "adapter_model.safetensors": "2" * 64,
                "trainer_state.json": "3" * 64,
            },
        },
    }
    selection = {
        "schema_version": "pii-zh.b3-full-attention-bie-model-selection.v1",
        "receipt_type": "b3_full_attention_bie_development_model_selection",
        "status": "selected_development_checkpoint_before_independent_test",
        "release_eligible": False,
        "selected": selected,
    }
    selection["selection_receipt_sha256"] = package_builder.canonical_json_hash(selection)
    selection_receipt = tmp_path / "selection.json"
    _write_json(selection_receipt, selection)
    selection_receipt.chmod(0o444)
    selection_file_sha256 = hashlib.sha256(selection_receipt.read_bytes()).hexdigest()

    runtime_files = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in package_builder.SOURCE_MODEL_FILES
        if name != "training_manifest.json"
    }
    model_export = {
        "schema_version": "pii-zh.b3-full-bie73-model-export.v1",
        "receipt_type": "b3_selected_checkpoint_merged_model_export",
        "status": "frozen_for_independent_test",
        "model_selection_receipt": {
            "file_sha256": selection_file_sha256,
            "selection_receipt_sha256": selection["selection_receipt_sha256"],
        },
        "selected": selected,
        "model": {
            "path": str(source.resolve()),
            "training_manifest_sha256": training_file_sha256,
            "training_manifest_logical_sha256": training_manifest["manifest_sha256"],
            "runtime_files": runtime_files,
        },
        "benchmark_content_read": False,
        "independent_test_content_read": False,
    }
    model_export["model_export_receipt_sha256"] = package_builder.canonical_json_hash(model_export)
    model_export_receipt = tmp_path / "model-export.json"
    _write_json(model_export_receipt, model_export)
    model_export_receipt.chmod(0o444)
    export_file_sha256 = hashlib.sha256(model_export_receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(
        package_builder,
        "FROZEN_B3_IDENTITY",
        {
            "selection_receipt_file_sha256": selection_file_sha256,
            "selection_receipt_sha256": selection["selection_receipt_sha256"],
            "model_export_receipt_file_sha256": export_file_sha256,
            "model_export_receipt_sha256": model_export["model_export_receipt_sha256"],
            "weight_file_sha256": runtime_files["model.safetensors"],
            "training_manifest_file_sha256": training_file_sha256,
            "training_manifest_sha256": training_manifest["manifest_sha256"],
            "candidate_id": "tiny_b3_fixture",
            "seed": 42,
            "epoch": 1,
            "global_step": 7,
            "checkpoint_id": "checkpoint-7",
        },
    )

    # These exist in the real recovered directory but must never enter the
    # public model package.
    _write_json(source / "recovery_attempt.json", {"status": "fixture-only"})
    _write_json(source / "final_eval_metrics.json", {"strict_f1": 0.5})

    readme = tmp_path / "MODEL_CARD.md"
    readme.write_text("# B3 BIE73\n\nPublic model card fixture.\n", encoding="utf-8")
    license_file = tmp_path / "LICENSE.input"
    license_file.write_text("Apache License 2.0 fixture\n", encoding="utf-8")
    notice = tmp_path / "NOTICE.input"
    notice.write_text("B3 BIE73 fixture notice\n", encoding="utf-8")
    third_party_notices = tmp_path / "THIRD_PARTY.input.md"
    third_party_notices.write_text("# Third-party notices\n\nFixture.\n", encoding="utf-8")
    security = tmp_path / "SECURITY.input.md"
    security.write_text(
        "# Security\n\nUse the repository private reporting page.\n", encoding="utf-8"
    )
    return PackageInputs(
        source=source,
        selection_receipt=selection_receipt,
        model_export_receipt=model_export_receipt,
        readme=readme,
        license_file=license_file,
        notice=notice,
        third_party_notices=third_party_notices,
        security=security,
    )


def _make_tree_writable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    root.chmod(0o755)
    for path in root.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(0o644)


def _seal_tree(root: Path) -> None:
    for path in root.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    root.chmod(0o555)


def test_builder_is_deterministic_minimal_read_only_and_verifiable(
    package_inputs: PackageInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    first = tmp_path / "package-first"
    second = tmp_path / "package-second"
    original_training = (package_inputs.source / "training_manifest.json").read_bytes()
    try:
        package_builder.build_package(**_package_kwargs(package_inputs, first))
        package_builder.build_package(**_package_kwargs(package_inputs, second))

        assert {path.name for path in first.iterdir()} == package_builder.PACKAGE_FILES
        assert not any(
            fragment in path.name.lower()
            for path in first.iterdir()
            for fragment in ("recovery", "attempt", "metric", "freeze")
        )
        assert stat.S_IMODE(first.stat().st_mode) == 0o555
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in first.iterdir())
        assert (first / "training_manifest.json").read_bytes() == original_training
        assert (first / "config.json").read_bytes() == (
            package_inputs.source / "config.json"
        ).read_bytes()
        assert (first / ".gitattributes").read_bytes() == (
            b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        )
        assert {path.name: path.read_bytes() for path in first.iterdir()} == {
            path.name: path.read_bytes() for path in second.iterdir()
        }

        manifest = package_verifier.verify_package(first)
        assert manifest["model"]["token_label_count"] == 73
        assert manifest["model"]["entity_type_count"] == 24
        assert manifest["model"]["decoder_id"] == "constrained_viterbi"
        assert manifest["training_manifest"]["preserved_byte_for_byte"] is True
        assert manifest["files"]["model.safetensors"]["size_bytes"] < 2 * 1024 * 1024

        taxonomy = yaml.safe_load((first / "taxonomy.yaml").read_text(encoding="utf-8"))
        assert taxonomy["tagging_scheme"] == "BIE"
        assert taxonomy["model_contract"]["token_label_count"] == 73
        labels = json.loads((first / "id2label.json").read_text(encoding="utf-8"))
        assert len(labels) == 73
        assert labels["0"] == "O"
        assert labels["72"] == "E-SECRET"
        decoder = json.loads((first / "decoder_config.json").read_text(encoding="utf-8"))
        assert decoder["allowed_label_ids"] == list(range(73))
        assert decoder["execution"] == {
            "automatic_in_transformers_pipeline": False,
            "recommended_loader": "pii_zh.full_bie73.load_full_bie73_predictor",
            "raw_transformers_output": "token_logits_not_final_spans",
        }

        configuration, modeling = build_release.render_remote_code(
            package_builder.REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
            community_v2_preauthorization=True,
        )
        assert (first / "configuration_qwen3_bi.py").read_text() == configuration
        assert (first / "modeling_qwen3_bi.py").read_text() == modeling

        loaded_config = AutoConfig.from_pretrained(
            first, local_files_only=True, trust_remote_code=True
        )
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            first, local_files_only=True, trust_remote_code=True, use_fast=True
        )
        loaded_model = (
            AutoModelForTokenClassification.from_pretrained(
                first,
                local_files_only=True,
                trust_remote_code=True,
                use_safetensors=True,
            )
            .cpu()
            .eval()
        )
        assert loaded_config.pii_release_eligible is False
        assert loaded_config.num_labels == 73
        assert loaded_tokenizer.is_fast
        encoded = loaded_tokenizer("社区", return_tensors="pt")
        with torch.inference_mode():
            logits = loaded_model(**encoded).logits
        assert logits.shape[0] == 1
        assert logits.shape[-1] == 73
        assert torch.isfinite(logits).all().item()
    finally:
        _make_tree_writable(first)
        _make_tree_writable(second)


def test_builder_rejects_local_absolute_path_without_publishing(
    package_inputs: PackageInputs, tmp_path: Path
) -> None:
    package_inputs.readme.write_text(
        "Do not publish local path /data1/runs/private/model.\n", encoding="utf-8"
    )
    output = tmp_path / "blocked-package"
    with pytest.raises(package_builder.B3PackageError, match="local absolute path"):
        package_builder.build_package(**_package_kwargs(package_inputs, output))
    assert not output.exists()
    assert not list(tmp_path.glob(".blocked-package.staging-*"))


def test_builder_distinguishes_token_content_from_local_path_metadata(
    package_inputs: PackageInputs, tmp_path: Path
) -> None:
    merge_fixture = json.dumps(
        {
            "model": {
                "vocab": {"/data-token": 0},
                "merges": [["/data-piece", "token"]],
            }
        }
    ).encode()
    package_builder._assert_tokenizer_has_no_local_path_metadata(merge_fixture)
    output = tmp_path / "token-content-package"
    try:
        package_builder.build_package(**_package_kwargs(package_inputs, output))
        package_verifier.verify_package(output)
    finally:
        _make_tree_writable(output)

    tokenizer_path = package_inputs.source / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["source_path"] = "/data1/private/tokenizer.json"
    _write_json(tokenizer_path, tokenizer)
    blocked = tmp_path / "tokenizer-metadata-blocked"
    with pytest.raises(package_builder.B3PackageError, match="local absolute path"):
        package_builder.build_package(**_package_kwargs(package_inputs, blocked))
    assert not blocked.exists()


def test_builder_requires_exact_frozen_selection_and_export_receipts(
    package_inputs: PackageInputs, tmp_path: Path
) -> None:
    package_inputs.model_export_receipt.chmod(0o644)
    export = json.loads(package_inputs.model_export_receipt.read_text(encoding="utf-8"))
    export["status"] = "different-export"
    export.pop("model_export_receipt_sha256")
    export["model_export_receipt_sha256"] = package_builder.canonical_json_hash(export)
    _write_json(package_inputs.model_export_receipt, export)
    package_inputs.model_export_receipt.chmod(0o444)
    output = tmp_path / "receipt-blocked"
    with pytest.raises(package_builder.B3PackageError, match="frozen B3 identity"):
        package_builder.build_package(**_package_kwargs(package_inputs, output))
    assert not output.exists()


def test_builder_rejects_pickle_source_and_required_symlink(
    package_inputs: PackageInputs, tmp_path: Path
) -> None:
    pickle_file = package_inputs.source / "optimizer.pt"
    pickle_file.write_bytes(b"not actually a pickle")
    with pytest.raises(package_builder.B3PackageError, match="pickle-based"):
        package_builder.build_package(
            **_package_kwargs(package_inputs, tmp_path / "pickle-blocked")
        )
    pickle_file.unlink()

    tokenizer = package_inputs.source / "tokenizer.json"
    tokenizer.unlink()
    tokenizer.symlink_to(package_inputs.source / "config.json")
    with pytest.raises(package_builder.B3PackageError, match="symlink"):
        package_builder.build_package(
            **_package_kwargs(package_inputs, tmp_path / "symlink-blocked")
        )


def test_builder_never_clobbers_existing_or_concurrently_created_output(
    package_inputs: PackageInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "owned"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(package_builder.B3PackageError, match="already exists"):
        package_builder.build_package(**_package_kwargs(package_inputs, output))
    assert sentinel.read_text(encoding="utf-8") == "keep"

    raced = tmp_path / "raced"

    def create_owner_then_fail(_source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel").write_text("concurrent owner", encoding="utf-8")
        raise package_builder.B3PackageError("output appeared during publication")

    monkeypatch.setattr(package_builder, "_rename_noreplace", create_owner_then_fail)
    with pytest.raises(package_builder.B3PackageError, match="appeared during publication"):
        package_builder.build_package(**_package_kwargs(package_inputs, raced))
    assert (raced / "sentinel").read_text(encoding="utf-8") == "concurrent owner"
    assert not list(tmp_path.glob(".raced.staging-*"))


def test_post_rename_fsync_failure_retains_verifiable_output_with_explicit_state(
    package_inputs: PackageInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "post-rename"
    real_fsync = package_builder._fsync_directory

    def fail_parent_only(path: Path) -> None:
        if path == output.parent:
            raise OSError("injected parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(package_builder, "_fsync_directory", fail_parent_only)
    try:
        with pytest.raises(
            package_builder.B3PackagePublicationAmbiguousError,
            match="retained.*verify.*retry",
        ):
            package_builder.build_package(**_package_kwargs(package_inputs, output))
        assert output.is_dir()
        assert stat.S_IMODE(output.stat().st_mode) == 0o555
        package_verifier.verify_package(output)
    finally:
        _make_tree_writable(output)


def test_verifier_rejects_tamper_extra_file_symlink_and_wrong_expected_hash(
    package_inputs: PackageInputs, tmp_path: Path
) -> None:
    package = tmp_path / "package"
    package_builder.build_package(**_package_kwargs(package_inputs, package))
    weight_hash = hashlib.sha256((package / "model.safetensors").read_bytes()).hexdigest()
    training_hash = hashlib.sha256((package / "training_manifest.json").read_bytes()).hexdigest()
    try:
        package_verifier.verify_package(
            package,
            expected_weight_sha256=weight_hash,
            expected_training_manifest_file_sha256=training_hash,
        )
        with pytest.raises(
            package_verifier.B3PackageVerificationError, match="expected release hash"
        ):
            package_verifier.verify_package(package, expected_weight_sha256="0" * 64)

        _make_tree_writable(package)
        config = package / "config.json"
        config.write_bytes(config.read_bytes() + b"\n")
        _seal_tree(package)
        with pytest.raises(package_verifier.B3PackageVerificationError, match="checksum mismatch"):
            package_verifier.verify_package(package)

        _make_tree_writable(package)
        config.write_bytes((package_inputs.source / "config.json").read_bytes())
        config.chmod(0o444)
        extra = package / "recovery_metrics.json"
        extra.write_text("{}\n", encoding="utf-8")
        _seal_tree(package)
        with pytest.raises(
            package_verifier.B3PackageVerificationError,
            match="forbidden filename|closed allowlist",
        ):
            package_verifier.verify_package(package)

        _make_tree_writable(package)
        extra.unlink()
        original = package / "README.md"
        original.unlink()
        original.symlink_to(package / "LICENSE")
        _seal_tree(package)
        with pytest.raises(package_verifier.B3PackageVerificationError, match="regular files"):
            package_verifier.verify_package(package)
    finally:
        _make_tree_writable(package)
        readme = package / "README.md"
        if readme.is_symlink():
            readme.unlink()


def test_cli_build_and_verify_tiny_fixture(
    package_inputs: PackageInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli-package"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_b3_full_bie73_hf_package.py",
            "--source-model-dir",
            str(package_inputs.source),
            "--selection-receipt",
            str(package_inputs.selection_receipt),
            "--model-export-receipt",
            str(package_inputs.model_export_receipt),
            "--output-dir",
            str(output),
            "--readme",
            str(package_inputs.readme),
            "--license",
            str(package_inputs.license_file),
            "--notice",
            str(package_inputs.notice),
            "--third-party-notices",
            str(package_inputs.third_party_notices),
            "--security",
            str(package_inputs.security),
        ],
    )
    try:
        assert package_builder.main() == 0
        capsys.readouterr()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_b3_full_bie73_hf_package.py",
                "--package-dir",
                str(output),
            ],
        )
        assert package_verifier.main() == 0
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["status"] == "PASS"
        assert receipt["schema_version"] == package_builder.SCHEMA_VERSION
    finally:
        _make_tree_writable(output)


def test_verifier_portable_mode_only_relaxes_permissions(
    package_inputs: PackageInputs, tmp_path: Path
) -> None:
    package = tmp_path / "portable-package"
    package_builder.build_package(**_package_kwargs(package_inputs, package))
    try:
        _make_tree_writable(package)
        with pytest.raises(package_verifier.B3PackageVerificationError, match="mode"):
            package_verifier.verify_package(package)
        manifest = package_verifier.verify_package(package, require_read_only=False)
        assert manifest["schema_version"] == package_builder.SCHEMA_VERSION
    finally:
        _make_tree_writable(package)
