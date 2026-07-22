"""Prepare the pinned CLUENER primary checkpoint for local service use."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pii_zh.presidio.cluener_primary import (
    CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED,
    CLUENER_PRIMARY_REQUIRED_FILES,
    VerifiedCluenerPrimaryArtifact,
    verify_cluener_primary_artifact,
)
from pii_zh.training.safe_conversion import (
    RestrictedConversionError,
    convert_restricted_state_dict,
    sha256_file,
)

CLUENER_PRIMARY_PREPARATION_SCHEMA_VERSION = "pii-zh.cluener-primary.prepare.v1"
CLUENER_PRIMARY_SOURCE_WEIGHT_NAME = "pytorch_model.bin"
CLUENER_PRIMARY_SOURCE_REQUIRED_FILES: Mapping[str, str] = MappingProxyType(
    {
        "README.md": CLUENER_PRIMARY_REQUIRED_FILES["README.md"],
        "config.json": CLUENER_PRIMARY_REQUIRED_FILES["config.json"],
        CLUENER_PRIMARY_SOURCE_WEIGHT_NAME: (
            "b865252516115c46bc508167fa2258f198bcce520eb7a1ac4fbf8d50dc361368"
        ),
        "special_tokens_map.json": CLUENER_PRIMARY_REQUIRED_FILES[
            "special_tokens_map.json"
        ],
        "tokenizer_config.json": CLUENER_PRIMARY_REQUIRED_FILES[
            "tokenizer_config.json"
        ],
        "vocab.txt": CLUENER_PRIMARY_REQUIRED_FILES["vocab.txt"],
    }
)
_COPIED_SOURCE_FILES = tuple(
    name
    for name in CLUENER_PRIMARY_SOURCE_REQUIRED_FILES
    if name != CLUENER_PRIMARY_SOURCE_WEIGHT_NAME
)
_MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024 * 1024


class CluenerPrimaryPreparationError(RuntimeError):
    """Raised when the pinned source cannot be prepared without changing it."""


def _regular_source_file(root: Path, name: str) -> Path:
    path = root / name
    try:
        observed = path.lstat()
    except OSError as exc:
        raise CluenerPrimaryPreparationError(
            "the pinned CLUENER source download is incomplete"
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise CluenerPrimaryPreparationError(
            "CLUENER source files must be regular, non-symlink files"
        )
    if observed.st_size > _MAX_SOURCE_FILE_BYTES:
        raise CluenerPrimaryPreparationError("a CLUENER source file is unexpectedly large")
    return path


def _verify_source(source_dir: str | Path) -> tuple[Path, Mapping[str, Path]]:
    candidate = Path(source_dir).expanduser()
    if candidate.is_symlink():
        raise CluenerPrimaryPreparationError("CLUENER source directory must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CluenerPrimaryPreparationError(
            "CLUENER source directory is unavailable"
        ) from exc
    if not root.is_dir():
        raise CluenerPrimaryPreparationError("CLUENER source path must be a local directory")

    files: dict[str, Path] = {}
    for name, expected_sha256 in CLUENER_PRIMARY_SOURCE_REQUIRED_FILES.items():
        path = _regular_source_file(root, name)
        try:
            actual_sha256 = sha256_file(path)
        except OSError as exc:
            raise CluenerPrimaryPreparationError("cannot read a CLUENER source file") from exc
        if actual_sha256 != expected_sha256:
            raise CluenerPrimaryPreparationError(
                "CLUENER source files do not match the pinned revision"
            )
        files[name] = path
    return root, MappingProxyType(files)


def _destination_path(output_dir: str | Path) -> Path:
    candidate = Path(output_dir).expanduser()
    if candidate.is_symlink() or candidate.exists():
        raise CluenerPrimaryPreparationError("output directory must not already exist")
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CluenerPrimaryPreparationError("output parent directory is unavailable") from exc
    destination = parent / candidate.name
    if destination.is_symlink() or destination.exists():
        raise CluenerPrimaryPreparationError("output directory must not already exist")
    return destination


def _write_receipt(payload: Mapping[str, Any], destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _materialize_tokenizer_json(vocab_path: Path, destination: Path) -> None:
    try:
        from tokenizers import (  # type: ignore[import-untyped]
            AddedToken,
            Tokenizer,
            decoders,
            models,
            normalizers,
            pre_tokenizers,
            processors,
        )

        # Split only on LF. ``str.splitlines`` would incorrectly treat the
        # vocabulary's literal U+2028 tokens as line separators.
        vocabulary_lines = vocab_path.read_bytes().decode("utf-8").split("\n")
        if vocabulary_lines and vocabulary_lines[-1] == "":
            vocabulary_lines.pop()
        vocabulary = {
            token[:-1] if token.endswith("\r") else token: index
            for index, token in enumerate(vocabulary_lines)
        }
        if len(vocabulary) != len(vocabulary_lines):
            raise CluenerPrimaryPreparationError("CLUENER vocabulary contains duplicate tokens")

        tokenizer = Tokenizer(
            models.WordPiece(
                vocab=vocabulary,
                unk_token="[UNK]",
                continuing_subword_prefix="##",
                max_input_chars_per_word=100,
            )
        )
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=None,
            lowercase=True,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", vocabulary["[CLS]"]),
                ("[SEP]", vocabulary["[SEP]"]),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=True)
        tokenizer.add_special_tokens(
            [
                AddedToken(
                    token,
                    single_word=False,
                    lstrip=False,
                    rstrip=False,
                    normalized=False,
                    special=True,
                )
                for token in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
            ]
        )
        tokenizer.save(str(destination), pretty=True)
    except CluenerPrimaryPreparationError:
        raise
    except (ImportError, KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise CluenerPrimaryPreparationError(
            "could not materialize the pinned CLUENER fast tokenizer"
        ) from exc
    if sha256_file(destination) != CLUENER_PRIMARY_REQUIRED_FILES["tokenizer.json"]:
        raise CluenerPrimaryPreparationError(
            "materialized CLUENER tokenizer does not match the pinned artifact"
        )
    os.chmod(destination, 0o644)


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    expected = CLUENER_PRIMARY_PREPARATION_RECEIPT_EXPECTED
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise CluenerPrimaryPreparationError(
            "converted CLUENER weights do not match the pinned artifact"
        )


def prepare_cluener_primary_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
) -> VerifiedCluenerPrimaryArtifact:
    """Convert one exact Hub download without modifying the downloaded source.

    ``source_dir`` must contain the fixed upstream revision and its legacy
    ``pytorch_model.bin``. The output is installed only after the eight-file,
    safetensors-only service artifact passes the same verifier used at runtime.
    """

    _, source_files = _verify_source(source_dir)
    destination = _destination_path(output_dir)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
    )
    renamed = False
    succeeded = False
    try:
        for name in _COPIED_SOURCE_FILES:
            target = temporary_root / name
            shutil.copyfile(source_files[name], target)
            os.chmod(target, 0o644)
        _materialize_tokenizer_json(
            source_files["vocab.txt"],
            temporary_root / "tokenizer.json",
        )

        try:
            receipt = convert_restricted_state_dict(
                source_files[CLUENER_PRIMARY_SOURCE_WEIGHT_NAME],
                temporary_root / "model.safetensors",
                delete_source=False,
            )
        except RestrictedConversionError as exc:
            raise CluenerPrimaryPreparationError("restricted CLUENER conversion failed") from exc
        os.chmod(temporary_root / "model.safetensors", 0o644)

        converter_sha256 = sha256_file(Path(__file__).resolve(strict=True))
        receipt["converter_sha256"] = converter_sha256
        _validate_receipt(receipt)
        _write_receipt(receipt, temporary_root / "conversion_receipt.json")
        os.chmod(temporary_root, 0o755)

        verify_cluener_primary_artifact(temporary_root)
        if destination.exists() or destination.is_symlink():
            raise CluenerPrimaryPreparationError("output directory appeared during preparation")
        temporary_root.rename(destination)
        renamed = True
        artifact = verify_cluener_primary_artifact(destination)
        succeeded = True
        return artifact
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, CluenerPrimaryPreparationError):
            raise
        raise CluenerPrimaryPreparationError("CLUENER primary preparation failed") from exc
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
        if renamed and not succeeded and destination.exists():
            shutil.rmtree(destination, ignore_errors=True)


__all__ = [
    "CLUENER_PRIMARY_PREPARATION_SCHEMA_VERSION",
    "CLUENER_PRIMARY_SOURCE_REQUIRED_FILES",
    "CLUENER_PRIMARY_SOURCE_WEIGHT_NAME",
    "CluenerPrimaryPreparationError",
    "prepare_cluener_primary_artifact",
]
