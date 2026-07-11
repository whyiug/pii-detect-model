from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from pii_zh.data.alignment import align_document_to_bio
from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
)
from pii_zh.models.decoding import decode_bio_ids
from pii_zh.tokenization import (
    BOUNDARY_MODE,
    BOUNDARY_MODE_CONFIG_KEY,
    TokenizerBoundaryError,
    assert_character_boundary_tokenizer,
    configure_character_boundary_tokenizer,
    load_serialized_fast_tokenizer,
)


def _byte_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(BPE(vocab={"<unk>": 0}, merges=[], unk_token="<unk>", fuse_unk=False))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")


def _record() -> DocumentRecord:
    return DocumentRecord.create(
        doc_id="boundary-oracle",
        text="前缀张三后缀",
        entities=[
            EntitySpan(
                start=2,
                end=4,
                label="PERSON_NAME",
                text="张三",
                source="synthetic",
            )
        ],
        language="zh-Hans-CN",
        domain="unit",
        scene="boundary",
        provenance=Provenance(
            source_id="unit",
            source_kind="synthetic",
            source_revision="1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit",
            generator_version="1",
            generator_seed=1,
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="validated",
        ),
        public_weight_training_allowed=True,
        generator_version="1",
    )


def test_character_boundary_mode_is_marked_persisted_and_fail_closed(tmp_path) -> None:
    tokenizer = _byte_tokenizer()
    with pytest.raises(TokenizerBoundaryError, match="boundary mode"):
        assert_character_boundary_tokenizer(tokenizer)

    configure_character_boundary_tokenizer(tokenizer)
    assert tokenizer.init_kwargs[BOUNDARY_MODE_CONFIG_KEY] == BOUNDARY_MODE
    tokenizer.save_pretrained(tmp_path)
    loaded = AutoTokenizer.from_pretrained(
        tmp_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        fix_mistral_regex=False,
    )
    assert_character_boundary_tokenizer(loaded)
    assert (
        json.loads(loaded.backend_tokenizer.to_str())["pre_tokenizer"]
        == json.loads(tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]
    )


def test_serialized_loader_preserves_model_specific_backend_graph(tmp_path) -> None:
    tokenizer = configure_character_boundary_tokenizer(_byte_tokenizer())
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tokenizer_class"] = "Qwen2Tokenizer"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_serialized_fast_tokenizer(tmp_path, require_boundary=True)

    assert (
        json.loads(loaded.backend_tokenizer.to_str())["pre_tokenizer"]
        == json.loads(tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]
    )


def test_perfect_bio_oracle_decodes_to_exact_character_boundary() -> None:
    tokenizer = configure_character_boundary_tokenizer(_byte_tokenizer())
    alignment = align_document_to_bio(_record(), tokenizer, add_special_tokens=False)
    assert alignment.statistics.unalignable_count == 0
    label2id = {"O": 0, "B-PERSON_NAME": 1, "I-PERSON_NAME": 2}
    id2label = {value: key for key, value in label2id.items()}
    spans = decode_bio_ids(alignment.label_ids(label2id), alignment.offset_mapping, id2label)
    assert [(span.start, span.end, span.entity_type) for span in spans] == [(2, 4, "PERSON_NAME")]
