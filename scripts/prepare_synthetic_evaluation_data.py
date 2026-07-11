#!/usr/bin/env python3
"""Attest one existing synthetic validation/test split for evaluation use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation import add_manifest_hash  # noqa: E402

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_ALLOWED_SUBSETS = ("validation", "test")
_MAXIMUM_MANIFEST_BYTES = 64 * 1024 * 1024


class SyntheticEvaluationDataError(ValueError):
    """Raised when the synthetic evaluation attestation cannot be trusted."""


def _synthetic_manifest_hash(value: Mapping[str, Any]) -> str:
    """Use the schema-2 materializer's UTF-8 canonical JSON contract."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SyntheticEvaluationDataError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticEvaluationDataError(f"{field} must be an object")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SyntheticEvaluationDataError(f"{field} must be a non-negative integer")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SyntheticEvaluationDataError(f"{field} must be a path-free stable identifier")
    return value


def _assert_public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = {"doc_id", "document_id", "raw_text", "text", "entity_value"}
        if forbidden & {str(key).lower() for key in value}:
            raise SyntheticEvaluationDataError("manifest contains a forbidden record-level field")
        for item in value.values():
            _assert_public_safe(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_public_safe(item)
    elif isinstance(value, str):
        if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise SyntheticEvaluationDataError("manifest contains an absolute path")
        if ".." in value.replace("\\", "/").split("/"):
            raise SyntheticEvaluationDataError("manifest contains an unsafe relative locator")


def _open_regular(path: Path, *, description: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SyntheticEvaluationDataError(f"cannot open regular {description}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise SyntheticEvaluationDataError(f"{description} must be a regular file")
    return descriptor


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    descriptor = _open_regular(path, description="synthetic source manifest")
    try:
        before = os.fstat(descriptor)
        if before.st_size > _MAXIMUM_MANIFEST_BYTES:
            raise SyntheticEvaluationDataError("synthetic source manifest exceeds size limit")
        encoded = bytearray()
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            encoded.extend(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise SyntheticEvaluationDataError("synthetic source manifest changed while read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(encoded).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticEvaluationDataError(
            "synthetic source manifest must be a UTF-8 JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise SyntheticEvaluationDataError("synthetic source manifest must be a JSON object")
    claimed = _sha256(value.get("manifest_sha256"), field="source manifest manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if _synthetic_manifest_hash(unsigned) != claimed:
        raise SyntheticEvaluationDataError("synthetic source manifest self-hash does not verify")
    return value, digest.hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _hash_and_count_records(path: Path) -> tuple[str, int, int]:
    descriptor = _open_regular(path, description="canonical synthetic split")
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        record_count = 0
        pending = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            parts = (pending + chunk).split(b"\n")
            pending = parts.pop()
            record_count += sum(bool(line.strip()) for line in parts)
        if pending.strip():
            record_count += 1
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise SyntheticEvaluationDataError("canonical synthetic split changed while read")
        return digest.hexdigest(), record_count, before.st_size
    finally:
        os.close(descriptor)


def _selected_file(source: Mapping[str, Any], *, subset: str) -> Mapping[str, Any]:
    files = source.get("files")
    if not isinstance(files, list):
        raise SyntheticEvaluationDataError("source manifest files must be an array")
    matches = [item for item in files if isinstance(item, Mapping) and item.get("split") == subset]
    if len(matches) != 1:
        raise SyntheticEvaluationDataError(f"source manifest must contain one {subset} file")
    return matches[0]


def _label_counts(source: Mapping[str, Any], *, subset: str) -> dict[str, int]:
    coverage = _mapping(source.get("coverage"), field="source manifest coverage")
    by_split = _mapping(
        coverage.get("labels_by_split"), field="source manifest coverage.labels_by_split"
    )
    raw = _mapping(by_split.get(subset), field=f"source manifest coverage.labels_by_split.{subset}")
    result: dict[str, int] = {}
    for label, count in raw.items():
        name = _safe_id(label, field="label name")
        result[name] = _non_negative_int(count, field=f"label count {name}")
    if not result:
        raise SyntheticEvaluationDataError("selected subset must contain at least one label")
    return dict(sorted(result.items()))


def _frozen_identity(source: Mapping[str, Any], *, subset: str) -> dict[str, Any]:
    frozen = _mapping(source.get("frozen_splits"), field="source manifest frozen_splits")
    if frozen.get("copy_mode") != "byte_for_byte":
        raise SyntheticEvaluationDataError("selected split is not a byte-for-byte frozen copy")
    if frozen.get("raw_jsonl_records_parsed") is not False:
        raise SyntheticEvaluationDataError("source manifest lacks the frozen access attestation")
    files = _mapping(frozen.get("files"), field="source manifest frozen_splits.files")
    item = _mapping(files.get(subset), field=f"source manifest frozen_splits.files.{subset}")
    output_sha256 = _sha256(item.get("output_sha256"), field=f"frozen {subset} output_sha256")
    source_sha256 = _sha256(item.get("source_sha256"), field=f"frozen {subset} source_sha256")
    if output_sha256 != source_sha256:
        raise SyntheticEvaluationDataError("frozen source/output SHA-256 values differ")
    return {
        "policy_version": _safe_id(frozen.get("policy_version"), field="frozen policy_version"),
        "copy_mode": "byte_for_byte",
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "record_count": _non_negative_int(item.get("records"), field=f"frozen {subset} records"),
        "bytes": _non_negative_int(item.get("bytes"), field=f"frozen {subset} bytes"),
    }


def _subset_policy(subset: str) -> dict[str, Any]:
    if subset == "validation":
        return {
            "subset_role": "calibration_validation",
            "used_for_threshold_selection": True,
            "frozen_holdout": True,
            "test_access": {
                "access_before_validation_gate": "not_applicable",
                "calibration_refit": "validation_only",
                "post_access_system_changes": "not_applicable",
            },
        }
    return {
        "subset_role": "frozen_test",
        "used_for_threshold_selection": False,
        "frozen_holdout": True,
        "test_access": {
            "access_before_validation_gate": "forbidden",
            "calibration_refit": "forbidden",
            "post_access_system_changes": "forbidden",
        },
    }


def _build_manifest(
    *,
    source: Mapping[str, Any],
    source_file_sha256: str,
    subset: str,
    canonical_sha256: str,
    record_count: int,
    byte_count: int,
) -> dict[str, Any]:
    if source.get("manifest_schema_version") != 2:
        raise SyntheticEvaluationDataError("synthetic source manifest schema must be version 2")
    dataset_id = _safe_id(source.get("dataset_id"), field="source dataset_id")
    dataset_version = _safe_id(source.get("dataset_version"), field="source dataset_version")
    file_identity = _selected_file(source, subset=subset)
    expected_sha256 = _sha256(
        file_identity.get("sha256"), field=f"source manifest {subset} SHA-256"
    )
    expected_records = _non_negative_int(
        file_identity.get("records"), field=f"source manifest {subset} records"
    )
    expected_bytes = _non_negative_int(
        file_identity.get("bytes"), field=f"source manifest {subset} bytes"
    )
    frozen = _frozen_identity(source, subset=subset)
    if len({expected_sha256, frozen["source_sha256"], frozen["output_sha256"]}) != 1:
        raise SyntheticEvaluationDataError("source and frozen split SHA-256 values differ")
    if expected_records != frozen["record_count"] or expected_bytes != frozen["bytes"]:
        raise SyntheticEvaluationDataError("source and frozen split counts differ")
    if canonical_sha256 != expected_sha256:
        raise SyntheticEvaluationDataError("canonical synthetic split SHA-256 mismatch")
    if record_count != expected_records:
        raise SyntheticEvaluationDataError("canonical synthetic split record count mismatch")
    if byte_count != expected_bytes:
        raise SyntheticEvaluationDataError("canonical synthetic split byte count mismatch")

    policy = _subset_policy(subset)
    label_counts = _label_counts(source, subset=subset)
    provenance = _mapping(source.get("provenance"), field="source manifest provenance")
    validation_test = _mapping(
        provenance.get("validation_test"), field="source manifest provenance.validation_test"
    )
    source_manifest_sha256 = _sha256(
        source.get("manifest_sha256"), field="source manifest manifest_sha256"
    )
    source_revision = _sha256(
        validation_test.get("parent_provenance_snapshot_sha256"),
        field="parent provenance snapshot SHA-256",
    )
    manifest = {
        "schema_version": 1,
        "manifest_type": "evaluation_dataset",
        "source_id": dataset_id,
        "upstream": {
            "source": "repo_curated_synthetic_templates",
            "revision": source_revision,
            "license": "Apache-2.0",
        },
        "source_dataset": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "manifest_schema_version": 2,
            "manifest_sha256": source_manifest_sha256,
            "manifest_file_sha256": source_file_sha256,
        },
        "isolation": {
            "pool": "evaluation_only",
            "training_allowed": False,
            "distillation_allowed": False,
            "threshold_selection_allowed": policy["used_for_threshold_selection"],
        },
        "subsets": {
            subset: {
                "record_count": record_count,
                "span_count": sum(label_counts.values()),
                "label_counts": label_counts,
                "canonical": {"sha256": canonical_sha256, "record_count": record_count},
                **policy,
                "frozen_source": frozen,
            }
        },
        "privacy": {
            "contains_paths": False,
            "contains_document_ids": False,
            "contains_raw_text": False,
            "contains_entity_values": False,
        },
    }
    result = add_manifest_hash(manifest)
    _assert_public_safe(result)
    return result


def _write_json_atomic(value: Mapping[str, Any], destination: Path) -> None:
    if destination.is_symlink():
        raise SyntheticEvaluationDataError("output must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            delete=False,
            prefix=f".{destination.name}.",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_stale_output(destination: Path) -> None:
    if destination.is_symlink():
        raise SyntheticEvaluationDataError("output must not be a symlink")
    if destination.exists():
        if not destination.is_file():
            raise SyntheticEvaluationDataError("output must be a regular file")
        destination.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=_ALLOWED_SUBSETS, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", "--manifest-output", dest="output", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Required safety acknowledgement; this command never rewrites canonical JSONL",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.verify_existing:
        parser.error("--verify-existing is required; canonical data is never generated here")
    output_path = args.output.expanduser().absolute()
    cleanup_output = False
    try:
        if output_path.is_symlink():
            raise SyntheticEvaluationDataError("output must not be a symlink")
        source_path = args.source_manifest.expanduser().resolve(strict=True)
        canonical_path = args.canonical.expanduser().resolve(strict=True)
        if source_path == canonical_path:
            raise SyntheticEvaluationDataError(
                "source manifest and canonical split must be distinct"
            )
        if output_path in {source_path, canonical_path}:
            raise SyntheticEvaluationDataError(
                "output must differ from source manifest and canonical split"
            )
        if output_path.exists() and any(
            os.path.samefile(output_path, source) for source in (source_path, canonical_path)
        ):
            raise SyntheticEvaluationDataError(
                "output must not alias the source manifest or canonical split"
            )
        cleanup_output = True
        _remove_stale_output(output_path)
        source, source_file_sha256 = _read_manifest(source_path)
        canonical_sha256, records, byte_count = _hash_and_count_records(canonical_path)
        manifest = _build_manifest(
            source=source,
            source_file_sha256=source_file_sha256,
            subset=args.subset,
            canonical_sha256=canonical_sha256,
            record_count=records,
            byte_count=byte_count,
        )
        _write_json_atomic(manifest, output_path)
    except (OSError, SyntheticEvaluationDataError) as exc:
        if cleanup_output:
            try:
                _remove_stale_output(output_path)
            except (OSError, SyntheticEvaluationDataError):
                pass
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "record_count": records,
                "subset": args.subset,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
