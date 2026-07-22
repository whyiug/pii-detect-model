#!/usr/bin/env python3
"""Build the deterministic public-file package for the frozen B3 BIE73 model.

The builder is intentionally local-only.  It copies a closed allowlist from a
completed model directory, derives the public BIE taxonomy/decoder metadata,
renders the reviewed Transformers remote code, and atomically publishes a
read-only directory.  It never loads tensors, reads evaluation rows, or copies
training/recovery diagnostics.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if __package__:
    from scripts import build_release
else:  # pragma: no cover - exercised through the command-line smoke test
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts import build_release  # type: ignore[no-redef]

from pii_zh.taxonomy import validate_taxonomy_document  # noqa: E402

SCHEMA_VERSION = "pii-zh.b3-full-bie73-hf-package.v1"
TAXONOMY_CONTRACT_VERSION = "pii-zh.bie73-taxonomy.v1"
DECODER_SCHEMA_VERSION = "pii-zh.bie73-decoder.v1"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
CHECKSUMS_NAME = "checksums.txt"
EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
    "AutoModelForTokenClassification": ("modeling_qwen3_bi.Qwen3BiForTokenClassification"),
}
SOURCE_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "training_manifest.json",
)
PUBLIC_INPUT_FILES = {
    "README.md": "readme",
    "LICENSE": "license",
    "NOTICE": "notice",
    "THIRD_PARTY_NOTICES.md": "third_party_notices",
    "SECURITY.md": "security",
}
GENERATED_FILES = (
    ".gitattributes",
    "configuration_qwen3_bi.py",
    "modeling_qwen3_bi.py",
    "id2label.json",
    "taxonomy.yaml",
    "decoder_config.json",
)
GITATTRIBUTES_PAYLOAD = b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
PAYLOAD_FILES = frozenset((*SOURCE_MODEL_FILES, *PUBLIC_INPUT_FILES, *GENERATED_FILES))
PACKAGE_FILES = frozenset((*PAYLOAD_FILES, PACKAGE_MANIFEST_NAME, CHECKSUMS_NAME))
FORBIDDEN_SERIALIZATION_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".joblib", ".pickle", ".pkl", ".pt", ".pth"}
)
FORBIDDEN_PUBLIC_NAME_PARTS = (
    "attempt",
    "checkpoint-",
    "eval_metric",
    "final_eval",
    "freeze",
    "recovery",
    "trainer_state",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
CHECKSUM_LINE_PATTERN = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)\Z")
_LOCAL_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9:])/(?:data\d*|home|root|tmp|var/(?:tmp|lib)|Users)(?:/[^\s`'\"<>]*)?"
)
_LOCAL_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:\\(?:[^\s`'\"<>]+\\?)+")
_FILE_URI = re.compile(r"(?i)file://")
_FICLONE = 0x40049409
_READ_BLOCK_BYTES = 8 * 1024 * 1024
_MAX_TEXT_BYTES = 64 * 1024 * 1024

# Exact, preregistered B3 selection/export anchors.  Unit tests replace this
# entire mapping with a tiny-fixture identity; the production CLI has no
# override and therefore cannot package an arbitrary self-consistent BIE73
# checkpoint under the B3 name.
FROZEN_B3_IDENTITY: dict[str, str | int] = {
    "selection_receipt_file_sha256": (
        "c24eebb51ead26f724e9d37bfef47bbca91fb2e98f054bc0596f9d19b4531103"
    ),
    "selection_receipt_sha256": (
        "00a49558cb691c146712a2e8a2cb14f2cd3772a518b68adf9d1b15b50918b26f"
    ),
    "model_export_receipt_file_sha256": (
        "b9e28ef2a54a3427024abe7e38aa8fc4891efbdfaa301dca97de13c09d4962eb"
    ),
    "model_export_receipt_sha256": (
        "049003c09aa1c0da72ebcc6cc5d4dd9496dc545e6ee54d00285df4c2d374da2b"
    ),
    "weight_file_sha256": ("cde3ab7366c5f733c13e58333429655ed84f0257f9ad5d8c716f2fc739ef4039"),
    "training_manifest_file_sha256": (
        "85ad2667f802562b45e05b6dd69a253f2956c401387d6ae2670a95fa74824526"
    ),
    "training_manifest_sha256": (
        "f1972b5a9ded56baa66c9bcd1a0c232bd3c1fca0f7b1a10190c8f3123a49ae48"
    ),
    "candidate_id": "b3_full_attention_bie73_v7_2_seed42",
    "seed": 42,
    "epoch": 1,
    "global_step": 1125,
    "checkpoint_id": "checkpoint-1125",
}


class B3PackageError(RuntimeError):
    """Raised when B3 package construction cannot fail closed."""


class B3PackagePublicationAmbiguousError(B3PackageError):
    """Raised after rename when parent-directory durability cannot be confirmed."""


def _canonical_json_bytes(value: Mapping[str, Any], *, remove: str | None = None) -> bytes:
    document = dict(value)
    if remove is not None:
        document.pop(remove, None)
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise B3PackageError("document is not canonical-JSON serializable") from exc


def canonical_json_hash(value: Mapping[str, Any], *, remove: str | None = None) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, remove=remove)).hexdigest()


def _render_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(payload: bytes, *, field: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise B3PackageError(f"{field} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise B3PackageError(f"{field} contains a non-finite JSON number")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except B3PackageError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B3PackageError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise B3PackageError(f"{field} must be a JSON object")
    return value


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise B3PackageError(f"path traversal is forbidden: {path}")
    return expanded.absolute()


def _reject_symlink_ancestry(path: Path, *, field: str) -> Path:
    lexical = _lexical_absolute(path)
    cursor = lexical
    while True:
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise B3PackageError(f"cannot inspect {field}: {cursor}") from exc
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise B3PackageError(f"{field} must not use symlinks: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return lexical


def _read_regular(path: Path, *, field: str, maximum: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise B3PackageError(f"{field} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise B3PackageError(f"{field} must be a regular non-symlink file")
        if maximum is not None and before.st_size > maximum:
            raise B3PackageError(f"{field} exceeds its size limit")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if maximum is not None and total > maximum:
                raise B3PackageError(f"{field} exceeds its size limit")
            blocks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        identity = lambda item: (  # noqa: E731 - keeps the race comparison compact
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if (
            identity(before) != identity(after)
            or identity(before) != identity(current)
            or not stat.S_ISREG(current.st_mode)
            or total != before.st_size
        ):
            raise B3PackageError(f"{field} changed while it was read")
        return b"".join(blocks)
    except B3PackageError:
        raise
    except OSError as exc:
        raise B3PackageError(f"{field} could not be read safely") from exc
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, *, field: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise B3PackageError(f"{field} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise B3PackageError(f"{field} must be a regular non-symlink file")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or before_identity != current_identity
            or not stat.S_ISREG(current.st_mode)
            or total != before.st_size
        ):
            raise B3PackageError(f"{field} changed while it was hashed")
        return digest.hexdigest(), total
    except B3PackageError:
        raise
    except OSError as exc:
        raise B3PackageError(f"{field} could not be hashed safely") from exc
    finally:
        os.close(descriptor)


def _assert_no_local_path(payload: bytes, *, field: str) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise B3PackageError(f"{field} must be UTF-8 text") from exc
    if _LOCAL_POSIX_PATH.search(text) or _LOCAL_WINDOWS_PATH.search(text) or _FILE_URI.search(text):
        raise B3PackageError(f"{field} contains a local absolute path")


def _assert_tokenizer_has_no_local_path_metadata(payload: bytes) -> None:
    """Scan tokenizer metadata while treating vocabulary strings as tokens.

    Byte-level tokenizers legitimately contain strings such as ``/data`` in
    vocab keys or merge pieces.  Those are model symbols, not filesystem
    provenance, so only non-token metadata is subject to the path policy.
    """

    document = _strict_json(payload, field="tokenizer.json")

    def is_token_content(path: tuple[str | int, ...]) -> bool:
        if len(path) >= 2 and path[:2] in {("model", "vocab"), ("model", "merges")}:
            return True
        if len(path) >= 3 and path[0] == "added_tokens" and path[-1] == "content":
            return True
        return bool(path and path[-1] in {"token", "tokens", "unk_token"})

    def visit(value: object, path: tuple[str | int, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise B3PackageError("tokenizer.json contains a non-string object key")
                visit(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, index))
        elif isinstance(value, str) and not is_token_content(path):
            _assert_no_local_path(value.encode("utf-8"), field="tokenizer.json metadata")

    visit(document, ())


def _source_taxonomy() -> tuple[dict[str, Any], tuple[str, ...]]:
    path = REPOSITORY_ROOT / "src/pii_zh/taxonomy/taxonomy.yaml"
    payload = _read_regular(path, field="repository taxonomy", maximum=_MAX_TEXT_BYTES)
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise B3PackageError("repository taxonomy is invalid YAML") from exc
    if not isinstance(raw, dict):
        raise B3PackageError("repository taxonomy must be a mapping")
    try:
        taxonomy = validate_taxonomy_document(raw)
    except (TypeError, ValueError) as exc:
        raise B3PackageError("repository taxonomy contract is invalid") from exc
    core = tuple(entity.name for entity in taxonomy.label_sets["core"])
    if taxonomy.outside_label != "O" or len(core) != 24 or len(set(core)) != 24:
        raise B3PackageError("repository taxonomy is not the exact core-24 contract")
    return deepcopy(raw), core


def expected_label_maps() -> tuple[dict[str, int], dict[str, str]]:
    _taxonomy, core = _source_taxonomy()
    labels = ["O"]
    for entity in core:
        labels.extend((f"B-{entity}", f"I-{entity}", f"E-{entity}"))
    if len(labels) != 73:
        raise B3PackageError("BIE label derivation did not produce 73 labels")
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {str(index): label for index, label in enumerate(labels)}
    return label2id, id2label


def render_public_taxonomy() -> bytes:
    document, core = _source_taxonomy()
    document["tagging_scheme"] = "BIE"
    document["model_contract"] = {
        "schema_version": TAXONOMY_CONTRACT_VERSION,
        "entity_type_count": 24,
        "entity_types": list(core),
        "tag_prefixes": ["B", "I", "E"],
        "token_label_count": 73,
        "single_token_entity_state": "B",
    }
    try:
        rendered = yaml.safe_dump(
            document,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, yaml.YAMLError) as exc:
        raise B3PackageError("BIE taxonomy could not be rendered deterministically") from exc
    _assert_no_local_path(rendered, field="generated taxonomy.yaml")
    return rendered


def render_id2label() -> bytes:
    _label2id, id2label = expected_label_maps()
    return _render_json(id2label)


def render_decoder_config(*, id2label_sha256: str, taxonomy_version: str) -> bytes:
    if SHA256_PATTERN.fullmatch(id2label_sha256) is None:
        raise B3PackageError("id2label hash is invalid")
    document: dict[str, Any] = {
        "schema_version": DECODER_SCHEMA_VERSION,
        "decoder_id": "constrained_viterbi",
        "tagging_scheme": "BIE",
        "taxonomy_version": taxonomy_version,
        "token_label_count": 73,
        "outside_label_id": 0,
        "allowed_label_ids": list(range(73)),
        "label_scope": "all_core24_labels_before_decode",
        "single_token_entity_state": "B",
        "multi_token_entity_states": ["B", "I", "E"],
        "invalid_transition_policy": "reject_by_constrained_viterbi",
        "span_score": "minimum_decoded_token_probability",
        "id2label_file_sha256": id2label_sha256,
        "execution": {
            "automatic_in_transformers_pipeline": False,
            "recommended_loader": "pii_zh.full_bie73.load_full_bie73_predictor",
            "raw_transformers_output": "token_logits_not_final_spans",
        },
    }
    return _render_json(document)


def _normalize_id2label(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise B3PackageError("config id2label must be an object")
    normalized: dict[str, str] = {}
    for raw_key, raw_label in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_label, str) or not raw_label:
            raise B3PackageError("config id2label contains an invalid entry")
        try:
            label_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise B3PackageError("config id2label contains an invalid key") from exc
        key = str(label_id)
        if label_id < 0 or key in normalized:
            raise B3PackageError("config id2label contains a duplicate or negative key")
        normalized[key] = raw_label
    if sorted(map(int, normalized)) != list(range(len(normalized))):
        raise B3PackageError("config id2label must be contiguous from zero")
    return {str(index): normalized[str(index)] for index in range(len(normalized))}


def validate_bie73_config(config: Mapping[str, Any], *, taxonomy_version: str) -> None:
    label2id, id2label = expected_label_maps()
    normalized = _normalize_id2label(config.get("id2label"))
    layer_types = config.get("layer_types")
    if (
        config.get("architectures") != ["Qwen3BiForTokenClassification"]
        or config.get("architecture_version") != "qwen3_bi_token_cls_v1"
        or config.get("auto_map") != EXPECTED_AUTO_MAP
        or config.get("bi_attention_backend") != "sdpa"
        or config.get("model_type") != "qwen3_bi"
        or config.get("num_labels") != 73
        or config.get("pii_attention_mode") != "full"
        or config.get("pii_release_eligible") is not False
        or config.get("pii_tagging_scheme") != "BIE"
        or config.get("pii_taxonomy_version") != taxonomy_version
        or config.get("use_cache") is not False
        or not isinstance(layer_types, list)
        or not layer_types
        or set(layer_types) != {"full_attention"}
        or normalized != id2label
        or config.get("label2id") != label2id
    ):
        raise B3PackageError("config.json is not the exact full-attention BIE73 contract")


def validate_training_manifest(
    manifest: Mapping[str, Any],
    *,
    source_model_dir: Path,
    taxonomy_version: str,
) -> None:
    label2id, _id2label = expected_label_maps()
    if (
        manifest.get("manifest_type") != "aiguard24_full_attention_bie_training_v1"
        or manifest.get("status") != "completed"
        or manifest.get("release_eligible") is not False
        or manifest.get("attention_mode") != "full"
        or manifest.get("tag_scheme") != "BIE"
        or manifest.get("taxonomy_version") != taxonomy_version
        or manifest.get("label2id") != label2id
    ):
        raise B3PackageError("training_manifest.json is not the completed B3 BIE73 contract")
    claimed = manifest.get("manifest_sha256")
    if (
        not isinstance(claimed, str)
        or SHA256_PATTERN.fullmatch(claimed) is None
        or claimed != canonical_json_hash(manifest, remove="manifest_sha256")
    ):
        raise B3PackageError("training_manifest.json self-hash does not verify")
    try:
        build_release.validate_training_artifact_binding(
            source_model_dir / "training_manifest.json", source_model_dir
        )
    except build_release.ReleaseBuildError as exc:
        raise B3PackageError("training manifest does not bind the source checkpoint") from exc


def _read_frozen_receipt(
    path: Path,
    *,
    field: str,
    logical_field: str,
    expected_file_sha256: str,
    expected_logical_sha256: str,
) -> tuple[dict[str, Any], str]:
    payload = _read_regular(path, field=field, maximum=_MAX_TEXT_BYTES)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise B3PackageError(f"{field} mode must be exactly 0444")
    file_sha256 = hashlib.sha256(payload).hexdigest()
    document = _strict_json(payload, field=field)
    logical_sha256 = document.get(logical_field)
    if (
        file_sha256 != expected_file_sha256
        or logical_sha256 != expected_logical_sha256
        or not isinstance(logical_sha256, str)
        or canonical_json_hash(document, remove=logical_field) != logical_sha256
    ):
        raise B3PackageError(f"{field} does not match the frozen B3 identity")
    return document, file_sha256


def verify_frozen_b3_provenance(
    *,
    selection_receipt: Path,
    model_export_receipt: Path,
    source_model_dir: Path,
    source_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Bind the package to the exact selected and exported B3 model."""

    identity = FROZEN_B3_IDENTITY
    required_identity = {
        "selection_receipt_file_sha256",
        "selection_receipt_sha256",
        "model_export_receipt_file_sha256",
        "model_export_receipt_sha256",
        "weight_file_sha256",
        "training_manifest_file_sha256",
        "training_manifest_sha256",
        "candidate_id",
        "seed",
        "epoch",
        "global_step",
        "checkpoint_id",
    }
    if set(identity) != required_identity:
        raise B3PackageError("frozen B3 identity constant has an invalid closed shape")
    selection, selection_file_sha256 = _read_frozen_receipt(
        selection_receipt,
        field="B3 model selection receipt",
        logical_field="selection_receipt_sha256",
        expected_file_sha256=str(identity["selection_receipt_file_sha256"]),
        expected_logical_sha256=str(identity["selection_receipt_sha256"]),
    )
    export, export_file_sha256 = _read_frozen_receipt(
        model_export_receipt,
        field="B3 model export receipt",
        logical_field="model_export_receipt_sha256",
        expected_file_sha256=str(identity["model_export_receipt_file_sha256"]),
        expected_logical_sha256=str(identity["model_export_receipt_sha256"]),
    )
    selected = selection.get("selected")
    exported_selected = export.get("selected")
    export_model = export.get("model")
    export_selection = export.get("model_selection_receipt")
    if not all(
        isinstance(item, Mapping)
        for item in (selected, exported_selected, export_model, export_selection)
    ):
        raise B3PackageError("B3 selection/export receipt structure is incomplete")
    assert isinstance(selected, Mapping)
    assert isinstance(exported_selected, Mapping)
    assert isinstance(export_model, Mapping)
    assert isinstance(export_selection, Mapping)
    selected_checkpoint = selected.get("selected_checkpoint")
    selected_training = selected.get("training_manifest")
    exported_training = exported_selected.get("training_manifest")
    runtime_files = export_model.get("runtime_files")
    if not all(
        isinstance(item, Mapping)
        for item in (
            selected_checkpoint,
            selected_training,
            exported_training,
            runtime_files,
        )
    ):
        raise B3PackageError("B3 selected model bindings are incomplete")
    assert isinstance(selected_checkpoint, Mapping)
    assert isinstance(selected_training, Mapping)
    assert isinstance(exported_training, Mapping)
    assert isinstance(runtime_files, Mapping)

    source_training_file_sha256 = hashlib.sha256(
        source_payloads["training_manifest.json"]
    ).hexdigest()
    source_training = _strict_json(
        source_payloads["training_manifest.json"], field="training_manifest.json"
    )
    source_runtime_hashes = {
        name: (
            str(identity["weight_file_sha256"])
            if name == "model.safetensors"
            else hashlib.sha256(source_payloads[name]).hexdigest()
        )
        for name in SOURCE_MODEL_FILES
        if name != "training_manifest.json"
    }
    expected_selected_identity = {
        "candidate_id": identity["candidate_id"],
        "seed": identity["seed"],
        "epoch": identity["epoch"],
        "global_step": identity["global_step"],
    }
    exported_model_path = export_model.get("path")
    try:
        exported_root = Path(str(exported_model_path)).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise B3PackageError("B3 export receipt model path is unavailable") from None
    mismatch = (
        selection.get("schema_version") != "pii-zh.b3-full-attention-bie-model-selection.v1"
        or selection.get("receipt_type") != "b3_full_attention_bie_development_model_selection"
        or selection.get("status") != "selected_development_checkpoint_before_independent_test"
        or selection.get("release_eligible") is not False
        or export.get("schema_version") != "pii-zh.b3-full-bie73-model-export.v1"
        or export.get("receipt_type") != "b3_selected_checkpoint_merged_model_export"
        or export.get("status") != "frozen_for_independent_test"
        or export.get("benchmark_content_read") is not False
        or export.get("independent_test_content_read") is not False
        or export_selection.get("file_sha256") != selection_file_sha256
        or export_selection.get("selection_receipt_sha256") != identity["selection_receipt_sha256"]
        or any(selected.get(key) != value for key, value in expected_selected_identity.items())
        or any(
            exported_selected.get(key) != value for key, value in expected_selected_identity.items()
        )
        or selected_checkpoint.get("checkpoint_id") != identity["checkpoint_id"]
        or selected_checkpoint.get("update_mode") != "lora"
        or exported_selected.get("selected_checkpoint") != selected_checkpoint
        or selected_training.get("file_sha256") != identity["training_manifest_file_sha256"]
        or selected_training.get("manifest_sha256") != identity["training_manifest_sha256"]
        or exported_training != selected_training
        or source_training_file_sha256 != identity["training_manifest_file_sha256"]
        or source_training.get("manifest_sha256") != identity["training_manifest_sha256"]
        or export_model.get("training_manifest_sha256") != identity["training_manifest_file_sha256"]
        or export_model.get("training_manifest_logical_sha256")
        != identity["training_manifest_sha256"]
        or exported_root != source_model_dir
        or any(runtime_files.get(name) != digest for name, digest in source_runtime_hashes.items())
    )
    if mismatch:
        raise B3PackageError("source model does not match the frozen selected B3 export")
    return {
        "candidate_id": identity["candidate_id"],
        "seed": identity["seed"],
        "epoch": identity["epoch"],
        "global_step": identity["global_step"],
        "checkpoint_id": identity["checkpoint_id"],
        "selection_receipt_file_sha256": selection_file_sha256,
        "selection_receipt_sha256": identity["selection_receipt_sha256"],
        "model_export_receipt_file_sha256": export_file_sha256,
        "model_export_receipt_sha256": identity["model_export_receipt_sha256"],
    }


def _validate_source_tree(source_model_dir: Path) -> None:
    try:
        metadata = os.lstat(source_model_dir)
    except OSError as exc:
        raise B3PackageError("source model directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise B3PackageError("source model directory must be a real directory")
    try:
        entries = sorted(os.scandir(source_model_dir), key=lambda item: item.name)
    except OSError as exc:
        raise B3PackageError("source model directory cannot be enumerated") from exc
    for entry in entries:
        try:
            item = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise B3PackageError("source model entry cannot be inspected") from exc
        if stat.S_ISLNK(item.st_mode):
            raise B3PackageError("source model directory contains a symlink")
        if stat.S_ISREG(item.st_mode) and Path(entry.name).suffix.lower() in (
            FORBIDDEN_SERIALIZATION_SUFFIXES
        ):
            raise B3PackageError("source model directory contains pickle-based weights")
        if not stat.S_ISREG(item.st_mode) and not stat.S_ISDIR(item.st_mode):
            raise B3PackageError("source model directory contains a special file")


def _write_new_file(path: Path, payload: bytes) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise B3PackageError(f"failed to write {path.name}")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _clone_or_copy_regular(source: Path, destination: Path, *, field: str) -> tuple[str, int]:
    expected_hash, expected_size = _hash_regular(source, field=field)
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    source_fd = -1
    destination_fd = -1
    try:
        source_fd = os.open(source, source_flags)
        destination_fd = os.open(destination, destination_flags, 0o600)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise B3PackageError(f"{field} must be a regular file")
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        except OSError as exc:
            if exc.errno not in {
                errno.EBADF,
                errno.EINVAL,
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            os.lseek(source_fd, 0, os.SEEK_SET)
            while True:
                block = os.read(source_fd, _READ_BLOCK_BYTES)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    if written < 1:
                        raise B3PackageError(f"failed to copy {field}") from exc
                    view = view[written:]
        os.fchmod(destination_fd, 0o444)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise B3PackageError(f"{field} changed while it was copied")
    except Exception:
        if source_fd >= 0:
            os.close(source_fd)
            source_fd = -1
        if destination_fd >= 0:
            os.close(destination_fd)
            destination_fd = -1
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
    observed_hash, observed_size = _hash_regular(destination, field=f"copied {field}")
    if (observed_hash, observed_size) != (expected_hash, expected_size):
        raise B3PackageError(f"copied {field} differs from its source")
    return observed_hash, observed_size


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise B3PackageError("atomic no-clobber publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise B3PackageError("output appeared during publication; refusing to clobber")
    if error_number in {errno.EINVAL, errno.ENOSYS}:
        raise B3PackageError("atomic no-clobber publication is unavailable")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_package_name(name: str) -> str:
    pure = PurePosixPath(name)
    if pure.is_absolute() or pure.as_posix() != name or len(pure.parts) != 1:
        raise B3PackageError("internal package filename is unsafe")
    lowered = name.lower()
    if Path(name).suffix.lower() in FORBIDDEN_SERIALIZATION_SUFFIXES or any(
        marker in lowered for marker in FORBIDDEN_PUBLIC_NAME_PARTS
    ):
        raise B3PackageError("internal package filename violates the public allowlist")
    return name


def _taxonomy_version(taxonomy_payload: bytes) -> str:
    try:
        value = yaml.safe_load(taxonomy_payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise B3PackageError("generated taxonomy is invalid") from exc
    version = value.get("taxonomy_version") if isinstance(value, Mapping) else None
    if not isinstance(version, str) or not version:
        raise B3PackageError("generated taxonomy has no version")
    return version


def build_package(
    *,
    source_model_dir: Path,
    selection_receipt: Path,
    model_export_receipt: Path,
    output_dir: Path,
    readme: Path,
    license_file: Path,
    notice: Path,
    third_party_notices: Path,
    security: Path,
) -> Path:
    """Build and atomically publish one immutable B3 BIE73 HF package."""

    source_model_dir = _reject_symlink_ancestry(source_model_dir, field="source model directory")
    selection_receipt = _reject_symlink_ancestry(
        selection_receipt, field="B3 model selection receipt"
    )
    model_export_receipt = _reject_symlink_ancestry(
        model_export_receipt, field="B3 model export receipt"
    )
    output_dir = _reject_symlink_ancestry(output_dir, field="output directory")
    public_inputs = {
        "README.md": _reject_symlink_ancestry(readme, field="README input"),
        "LICENSE": _reject_symlink_ancestry(license_file, field="LICENSE input"),
        "NOTICE": _reject_symlink_ancestry(notice, field="NOTICE input"),
        "THIRD_PARTY_NOTICES.md": _reject_symlink_ancestry(
            third_party_notices, field="third-party notices input"
        ),
        "SECURITY.md": _reject_symlink_ancestry(security, field="security input"),
    }
    if output_dir == source_model_dir or output_dir in source_model_dir.parents:
        raise B3PackageError("output directory must not replace an input ancestor")
    if source_model_dir in output_dir.parents:
        raise B3PackageError("output directory must not be inside the source model")
    if output_dir.exists() or output_dir.is_symlink():
        raise B3PackageError("output already exists; no-clobber is mandatory")
    _validate_source_tree(source_model_dir)

    source_paths = {name: source_model_dir / name for name in SOURCE_MODEL_FILES}
    source_payloads: dict[str, bytes] = {}
    for name, path in source_paths.items():
        if name == "model.safetensors":
            _hash_regular(path, field=name)
            continue
        payload = _read_regular(path, field=name, maximum=_MAX_TEXT_BYTES)
        if name == "tokenizer.json":
            _assert_tokenizer_has_no_local_path_metadata(payload)
        else:
            _assert_no_local_path(payload, field=name)
        source_payloads[name] = payload
    public_payloads: dict[str, bytes] = {}
    for name, path in public_inputs.items():
        payload = _read_regular(path, field=f"public input {name}", maximum=_MAX_TEXT_BYTES)
        if not payload:
            raise B3PackageError(f"public input {name} must not be empty")
        _assert_no_local_path(payload, field=f"public input {name}")
        public_payloads[name] = payload

    provenance = verify_frozen_b3_provenance(
        selection_receipt=selection_receipt,
        model_export_receipt=model_export_receipt,
        source_model_dir=source_model_dir,
        source_payloads=source_payloads,
    )

    taxonomy_payload = render_public_taxonomy()
    taxonomy_version = _taxonomy_version(taxonomy_payload)
    config = _strict_json(source_payloads["config.json"], field="config.json")
    validate_bie73_config(config, taxonomy_version=taxonomy_version)
    manifest = _strict_json(
        source_payloads["training_manifest.json"], field="training_manifest.json"
    )
    validate_training_manifest(
        manifest,
        source_model_dir=source_model_dir,
        taxonomy_version=taxonomy_version,
    )
    try:
        build_release.validate_safetensors(source_paths["model.safetensors"])
    except build_release.ReleaseBuildError as exc:
        raise B3PackageError("model.safetensors is structurally invalid") from exc

    id2label_payload = render_id2label()
    decoder_payload = render_decoder_config(
        id2label_sha256=hashlib.sha256(id2label_payload).hexdigest(),
        taxonomy_version=taxonomy_version,
    )
    try:
        configuration, modeling = build_release.render_remote_code(
            REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
            community_v2_preauthorization=True,
        )
    except build_release.ReleaseBuildError as exc:
        raise B3PackageError("reviewed remote code could not be rendered") from exc
    generated = {
        ".gitattributes": GITATTRIBUTES_PAYLOAD,
        "configuration_qwen3_bi.py": configuration.encode("utf-8"),
        "modeling_qwen3_bi.py": modeling.encode("utf-8"),
        "id2label.json": id2label_payload,
        "taxonomy.yaml": taxonomy_payload,
        "decoder_config.json": decoder_payload,
    }
    for name, payload in generated.items():
        _assert_no_local_path(payload, field=f"generated {name}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = _reject_symlink_ancestry(output_dir, field="output directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    inventory: dict[str, dict[str, Any]] = {}
    try:
        for name, source in source_paths.items():
            _safe_package_name(name)
            digest, size = _clone_or_copy_regular(
                source, staging / name, field=f"source model {name}"
            )
            if (
                name != "model.safetensors"
                and digest != hashlib.sha256(source_payloads[name]).hexdigest()
            ):
                raise B3PackageError(f"source model {name} changed after validation")
            inventory[name] = {
                "file_sha256": digest,
                "role": "model_weight" if name == "model.safetensors" else "model_runtime",
                "size_bytes": size,
            }
        for name, source in public_inputs.items():
            _safe_package_name(name)
            digest, size = _clone_or_copy_regular(
                source, staging / name, field=f"public input {name}"
            )
            if digest != hashlib.sha256(public_payloads[name]).hexdigest():
                raise B3PackageError(f"public input {name} changed after validation")
            inventory[name] = {
                "file_sha256": digest,
                "role": PUBLIC_INPUT_FILES[name],
                "size_bytes": size,
            }
        for name, payload in generated.items():
            _safe_package_name(name)
            digest, size = _write_new_file(staging / name, payload)
            inventory[name] = {
                "file_sha256": digest,
                "role": (
                    "huggingface_large_file_transport"
                    if name == ".gitattributes"
                    else "generated_runtime_contract"
                ),
                "size_bytes": size,
            }
        if set(inventory) != PAYLOAD_FILES:
            raise B3PackageError("internal package inventory differs from the closed allowlist")

        output_artifact = manifest.get("output_artifact")
        files = output_artifact.get("files") if isinstance(output_artifact, Mapping) else None
        expected_weight_hash = (
            files.get("model.safetensors") if isinstance(files, Mapping) else None
        )
        if expected_weight_hash != inventory["model.safetensors"]["file_sha256"]:
            raise B3PackageError("published weight hash differs from the training manifest")
        if expected_weight_hash != FROZEN_B3_IDENTITY["weight_file_sha256"]:
            raise B3PackageError("published weight hash differs from the selected B3 export")
        package_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "format": "huggingface_transformers_remote_code",
            "model": {
                "architecture": "Qwen3BiForTokenClassification",
                "attention_mode": "full",
                "decoder_id": "constrained_viterbi",
                "entity_type_count": 24,
                "model_type": "qwen3_bi",
                "tagging_scheme": "BIE",
                "taxonomy_version": taxonomy_version,
                "token_label_count": 73,
                "weight_file_sha256": expected_weight_hash,
            },
            "training_manifest": {
                "file_sha256": inventory["training_manifest.json"]["file_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "preserved_byte_for_byte": True,
            },
            "selection_provenance": provenance,
            "files": dict(sorted(inventory.items())),
            "file_policy": {
                "closed_allowlist": True,
                "contains_internal_evaluation_artifacts": False,
                "contains_local_absolute_path_metadata": False,
                "contains_pickle_serialization": False,
                "contains_recovery_artifacts": False,
                "contains_symlinks": False,
            },
            "build_inputs": {
                "builder_file_sha256": _hash_regular(Path(__file__), field="B3 package builder")[0],
                "remote_code_source_file_sha256": _hash_regular(
                    REPOSITORY_ROOT / "src/pii_zh/models/qwen3_bi.py",
                    field="remote code source",
                )[0],
                "taxonomy_source_file_sha256": _hash_regular(
                    REPOSITORY_ROOT / "src/pii_zh/taxonomy/taxonomy.yaml",
                    field="taxonomy source",
                )[0],
            },
        }
        package_manifest["manifest_sha256"] = canonical_json_hash(package_manifest)
        manifest_payload = _render_json(package_manifest)
        _assert_no_local_path(manifest_payload, field=PACKAGE_MANIFEST_NAME)
        _write_new_file(staging / PACKAGE_MANIFEST_NAME, manifest_payload)

        checksum_entries: list[str] = []
        for name in sorted(PACKAGE_FILES - {CHECKSUMS_NAME}):
            digest, _size = _hash_regular(staging / name, field=f"package file {name}")
            checksum_entries.append(f"{digest}  {name}")
        _write_new_file(
            staging / CHECKSUMS_NAME,
            ("\n".join(checksum_entries) + "\n").encode("utf-8"),
        )
        actual = {entry.name for entry in os.scandir(staging)}
        if actual != PACKAGE_FILES:
            raise B3PackageError("published package differs from the closed file allowlist")
        _fsync_directory(staging)
        os.chmod(staging, 0o555)
        _fsync_directory(staging)
        _rename_noreplace(staging, output_dir)
        try:
            _fsync_directory(output_dir.parent)
        except OSError as exc:
            raise B3PackagePublicationAmbiguousError(
                "package was atomically renamed and retained, but parent fsync failed; "
                "verify the existing output before any retry"
            ) from exc
    except Exception:
        if staging.exists():
            try:
                os.chmod(staging, 0o755)
            except OSError:
                pass
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def parse_checksums(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise B3PackageError("checksums.txt must be UTF-8") from exc
    if not lines:
        raise B3PackageError("checksums.txt must not be empty")
    result: dict[str, str] = {}
    for line in lines:
        match = CHECKSUM_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise B3PackageError("checksums.txt contains an invalid line")
        digest, name = match.groups()
        _safe_package_name(name)
        if name == CHECKSUMS_NAME or name in result:
            raise B3PackageError("checksums.txt contains a duplicate or self-reference")
        result[name] = digest
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    parser.add_argument("--model-export-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--readme", required=True, type=Path)
    parser.add_argument("--license", dest="license_file", required=True, type=Path)
    parser.add_argument("--notice", required=True, type=Path)
    parser.add_argument("--third-party-notices", required=True, type=Path)
    parser.add_argument("--security", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        output = build_package(
            source_model_dir=args.source_model_dir,
            selection_receipt=args.selection_receipt,
            model_export_receipt=args.model_export_receipt,
            output_dir=args.output_dir,
            readme=args.readme,
            license_file=args.license_file,
            notice=args.notice,
            third_party_notices=args.third_party_notices,
            security=args.security,
        )
    except (B3PackageError, OSError) as exc:
        print(f"B3 HF package build blocked: {exc}", file=__import__("sys").stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
