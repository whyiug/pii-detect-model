#!/usr/bin/env python3
"""Export deterministic, public-safe data and teacher release provenance.

The exporter consumes only a completed Full training manifest and reviewed
repository metadata.  It never reads a training/evaluation JSONL and never
copies template bodies, prompts, paths, document IDs, or entity values into
the release artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64}|local-fingerprint-[0-9a-f]{64})$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_MAXIMUM_INPUT_BYTES = 64 * 1024 * 1024

_BASE_SOURCE_ID = "qwen3_0_6b_base"
_DATA_SOURCE_ID = "repo_curated_synthetic_templates"
_TEACHER_SOURCE_ID = "qwen3_8b_placeholder_template_generator"
_EXPECTED_TRAINING_SOURCE_IDS = (
    _BASE_SOURCE_ID,
    _TEACHER_SOURCE_ID,
    _DATA_SOURCE_ID,
)
_EXPECTED_SPLIT_COUNTS = {"train": 16_000, "validation": 2_000}
_EXPECTED_DATA_POOL = "public_release_pool"
_TEACHER_ROLE = "upstream_template_candidate_generator"


class ReleaseProvenanceError(ValueError):
    """Raised when release provenance cannot be exported without ambiguity."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--template-asset", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    return parser


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_regular_bytes(path: Path, *, description: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseProvenanceError(f"cannot open regular {description}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseProvenanceError(f"{description} must be a regular file")
        if before.st_size > _MAXIMUM_INPUT_BYTES:
            raise ReleaseProvenanceError(f"{description} exceeds the input size limit")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            if total > _MAXIMUM_INPUT_BYTES:
                raise ReleaseProvenanceError(f"{description} exceeds the input size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ReleaseProvenanceError(f"{description} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    encoded = _read_regular_bytes(path, description=description)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseProvenanceError(f"{description} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseProvenanceError(f"{description} must be a JSON object")
    return value, hashlib.sha256(encoded).hexdigest()


def _load_yaml(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    encoded = _read_regular_bytes(path, description=description)
    try:
        value = yaml.safe_load(encoded.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReleaseProvenanceError(f"{description} must be valid UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise ReleaseProvenanceError(f"{description} must be a YAML object")
    return value, hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseProvenanceError(f"{field} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseProvenanceError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReleaseProvenanceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ReleaseProvenanceError(f"{field} must be an immutable content revision")
    return value


def _exact_int(value: object, expected: int, *, field: str) -> int:
    if isinstance(value, bool) or value != expected:
        raise ReleaseProvenanceError(f"{field} must equal {expected}")
    return expected


def _verify_self_hash(value: Mapping[str, Any], *, description: str) -> str:
    claimed = _sha256(value.get("manifest_sha256"), field=f"{description}.manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if _canonical_json_hash(unsigned) != claimed:
        raise ReleaseProvenanceError(f"{description} self-hash does not verify")
    return claimed


def _registry_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if registry.get("schema_version") != 1:
        raise ReleaseProvenanceError("source registry schema_version must equal 1")
    _string(registry.get("registry_version"), field="source registry.registry_version")
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ReleaseProvenanceError("source registry.sources must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(sources):
        entry = _mapping(source, field=f"source registry.sources[{index}]")
        source_id = _string(entry.get("id"), field=f"source registry.sources[{index}].id")
        if _SAFE_ID.fullmatch(source_id) is None:
            raise ReleaseProvenanceError("source registry contains an unsafe source ID")
        if source_id in result:
            raise ReleaseProvenanceError(f"source registry contains duplicate ID {source_id}")
        result[source_id] = entry
    return result


def _validate_registry_source(
    source: Mapping[str, Any],
    *,
    source_id: str,
    expected_kind: str,
) -> None:
    if source.get("id") != source_id or source.get("kind") != expected_kind:
        raise ReleaseProvenanceError(f"registry entry {source_id} has the wrong identity/kind")
    _revision(source.get("revision"), field=f"registry.{source_id}.revision")
    _string(source.get("declared_license"), field=f"registry.{source_id}.declared_license")
    review_status = _string(
        source.get("review_status"), field=f"registry.{source_id}.review_status"
    )
    if review_status.startswith(("pending", "not_", "no_public")):
        raise ReleaseProvenanceError(f"registry entry {source_id} has incomplete review")
    if source.get("public_weight_training_allowed") is not True:
        raise ReleaseProvenanceError(f"registry entry {source_id} is not release-admitted")


def _validate_training_manifest(manifest: Mapping[str, Any]) -> str:
    manifest_sha256 = _verify_self_hash(manifest, description="training manifest")
    if manifest.get("schema_version") != 4:
        raise ReleaseProvenanceError("training manifest schema_version must equal 4")
    if manifest.get("status") != "completed":
        raise ReleaseProvenanceError("training manifest must be completed")
    if manifest.get("attention_mode") != "full":
        raise ReleaseProvenanceError("selected checkpoint must use Full attention")
    if manifest.get("base_source_id") != _BASE_SOURCE_ID:
        raise ReleaseProvenanceError("training manifest has the wrong base source")
    source_ids = manifest.get("training_source_ids")
    if source_ids != list(_EXPECTED_TRAINING_SOURCE_IDS):
        raise ReleaseProvenanceError(
            "training_source_ids must canonically contain only base, direct data, "
            "and reviewed Qwen3-8B upstream lineage"
        )
    output_artifact = _mapping(
        manifest.get("output_artifact"), field="training manifest.output_artifact"
    )
    _sha256(
        output_artifact.get("weights_combined_sha256"),
        field="training manifest.output_artifact.weights_combined_sha256",
    )
    return manifest_sha256


def _validate_split_summary(
    split_name: str,
    split: Mapping[str, Any],
    *,
    count: int,
    data_revision: str,
    data_license: str,
) -> str:
    split_sha256 = _sha256(split.get("sha256"), field=f"datasets.{split_name}.sha256")
    summary = _mapping(split.get("summary"), field=f"datasets.{split_name}.summary")
    _exact_int(summary.get("document_count"), count, field=f"{split_name}.document_count")
    expected_split_counts = {split_name: count}
    if summary.get("split_counts") != expected_split_counts:
        raise ReleaseProvenanceError(
            f"{split_name}.split_counts must equal {expected_split_counts!r}"
        )
    expected_pool_counts = {_EXPECTED_DATA_POOL: count}
    if summary.get("data_pool_counts") != expected_pool_counts:
        raise ReleaseProvenanceError(
            f"{split_name}.data_pool_counts must equal {expected_pool_counts!r}"
        )
    for field in (
        "quality_gate_passed_document_count",
        "validators_passed_document_count",
        "public_weight_training_allowed_document_count",
    ):
        _exact_int(summary.get(field), count, field=f"{split_name}.{field}")
    sources = summary.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ReleaseProvenanceError(f"{split_name} must contain exactly one direct data source")
    source = _mapping(sources[0], field=f"datasets.{split_name}.summary.sources[0]")
    expected_fields: dict[str, object] = {
        "source_id": _DATA_SOURCE_ID,
        "source_kind": "curated_deterministic_synthetic",
        "revision": data_revision,
        "license": data_license,
        "data_pool": _EXPECTED_DATA_POOL,
        "split": split_name,
        "document_count": count,
    }
    for field, expected in expected_fields.items():
        if source.get(field) != expected:
            raise ReleaseProvenanceError(
                f"{split_name} direct source {field} must equal {expected!r}"
            )
    for field in (
        "quality_gate_passed_document_count",
        "validators_passed_document_count",
        "public_weight_training_allowed_document_count",
    ):
        _exact_int(source.get(field), count, field=f"{split_name}.source.{field}")
    return split_sha256


def _validate_template_asset(
    asset: Mapping[str, Any],
    *,
    asset_file_sha256: str,
    data_source: Mapping[str, Any],
    teacher_source: Mapping[str, Any],
) -> dict[str, Any]:
    expected_asset_revision = f"sha256:{asset_file_sha256}"
    if data_source.get("revision") != expected_asset_revision:
        raise ReleaseProvenanceError("template asset bytes differ from the registered revision")
    if asset.get("schema_version") != data_source.get("template_schema_version"):
        raise ReleaseProvenanceError("template asset schema differs from the registry")
    for asset_field, registry_field in (
        ("asset_version", "template_asset_version"),
        ("license", "declared_license"),
    ):
        if asset.get(asset_field) != data_source.get(registry_field):
            raise ReleaseProvenanceError(f"template asset {asset_field} differs from the registry")
    asset_id = _string(asset.get("asset_id"), field="template asset.asset_id")
    asset_version = _string(asset.get("asset_version"), field="template asset.asset_version")
    candidate = _mapping(asset.get("candidate_source"), field="template asset.candidate_source")
    curation = _mapping(asset.get("curation"), field="template asset.curation")
    evidence = _mapping(
        teacher_source.get("review_evidence"), field="teacher registry.review_evidence"
    )

    candidate_payload_sha256 = _sha256(
        candidate.get("candidate_payload_sha256"),
        field="template asset.candidate_source.candidate_payload_sha256",
    )
    if candidate_payload_sha256 != evidence.get("candidate_payload_sha256"):
        raise ReleaseProvenanceError("candidate payload hash differs from teacher review evidence")
    if expected_asset_revision != evidence.get("template_asset_revision"):
        raise ReleaseProvenanceError("template asset revision differs from teacher review evidence")
    curation_audit_sha256 = _sha256(
        data_source.get("curation_audit_sha256"),
        field="data registry.curation_audit_sha256",
    )
    if curation_audit_sha256 != evidence.get("curation_audit_sha256"):
        raise ReleaseProvenanceError("curation audit hash differs across registry entries")

    if candidate.get("deployment") != "local":
        raise ReleaseProvenanceError("template candidates must come from the reviewed local run")
    if candidate.get("model_id") != teacher_source.get("locator"):
        raise ReleaseProvenanceError("candidate model identity differs from the teacher registry")
    if candidate.get("declared_license") != teacher_source.get("declared_license"):
        raise ReleaseProvenanceError("candidate model license differs from the teacher registry")
    model_hashes = _mapping(
        candidate.get("model_file_hashes"),
        field="template asset.candidate_source.model_file_hashes",
    )
    model_index_sha256 = _sha256(
        model_hashes.get("model.safetensors.index.json"),
        field="template asset candidate model index",
    )
    if teacher_source.get("revision") != f"local-fingerprint-{model_index_sha256}":
        raise ReleaseProvenanceError("candidate model fingerprint differs from the registry")

    count_fields = (
        "reviewed_candidate_count",
        "accepted_candidate_count",
        "rejected_candidate_count",
    )
    for field in count_fields:
        value = curation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReleaseProvenanceError(f"template asset curation.{field} is invalid")
        if value != evidence.get(field):
            raise ReleaseProvenanceError(f"curation {field} differs from teacher review evidence")
    if curation.get("reviewed_candidate_count") != (
        curation.get("accepted_candidate_count", 0) + curation.get("rejected_candidate_count", 0)
    ):
        raise ReleaseProvenanceError("curation reviewed count does not partition decisions")
    if curation.get("review_method") != evidence.get("review_method"):
        raise ReleaseProvenanceError("curation review method differs from teacher evidence")
    if curation.get("raw_rejected_outputs_included") is not False:
        raise ReleaseProvenanceError("raw rejected teacher outputs must not enter the asset")
    if evidence.get("raw_rejected_outputs_included") is not False:
        raise ReleaseProvenanceError("teacher review evidence admits raw rejected outputs")

    positive = asset.get("positive_templates")
    hard_negative = asset.get("hard_negative_templates")
    if not isinstance(positive, list) or not isinstance(hard_negative, list):
        raise ReleaseProvenanceError("template asset must contain reviewed template arrays")
    candidate_ids = [
        item.get("source_candidate_id")
        for item in positive
        if isinstance(item, Mapping) and item.get("source_candidate_id") is not None
    ]
    human_positive_count = sum(
        isinstance(item, Mapping) and item.get("source_candidate_id") is None for item in positive
    )
    if not all(isinstance(value, str) and value for value in candidate_ids):
        raise ReleaseProvenanceError("accepted template candidate IDs are malformed")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReleaseProvenanceError("accepted template candidate IDs are not unique")
    _exact_int(
        len(candidate_ids),
        int(curation["accepted_candidate_count"]),
        field="accepted candidate template count",
    )
    _exact_int(
        human_positive_count,
        int(curation.get("human_authored_positive_count", -1)),
        field="human-authored positive template count",
    )
    _exact_int(
        len(hard_negative),
        int(curation.get("human_authored_hard_negative_count", -1)),
        field="human-authored hard-negative template count",
    )
    return {
        "asset_id": asset_id,
        "asset_version": asset_version,
        "asset_revision": expected_asset_revision,
        "candidate_payload_sha256": candidate_payload_sha256,
        "curation_audit_sha256": curation_audit_sha256,
        "reviewed_candidate_count": curation["reviewed_candidate_count"],
        "accepted_candidate_count": curation["accepted_candidate_count"],
        "rejected_candidate_count": curation["rejected_candidate_count"],
        "review_method": curation["review_method"],
    }


def _assert_public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden_keys = {
            "doc_id",
            "document_id",
            "entity_value",
            "raw_text",
            "text",
            "path",
            "file",
            "directory",
            "locator",
            "prompt",
            "response",
        }
        if forbidden_keys & {str(key).lower() for key in value}:
            raise ReleaseProvenanceError("release provenance contains a forbidden content field")
        for child in value.values():
            _assert_public_safe(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public_safe(child)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
            raise ReleaseProvenanceError("release provenance contains an absolute path")
        if ".." in normalized.split("/"):
            raise ReleaseProvenanceError("release provenance contains path traversal")


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["manifest_sha256"] = _canonical_json_hash(result)
    _verify_self_hash(result, description=str(result.get("artifact_type", "artifact")))
    _assert_public_safe(result)
    return result


def build_release_provenance(
    *,
    training_manifest_path: Path,
    source_registry_path: Path,
    template_asset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate reviewed lineage and return data/teacher provenance artifacts."""

    training, training_file_sha256 = _load_json(
        training_manifest_path, description="training manifest"
    )
    registry, registry_file_sha256 = _load_yaml(source_registry_path, description="source registry")
    asset, asset_file_sha256 = _load_json(template_asset_path, description="template asset")

    training_manifest_sha256 = _validate_training_manifest(training)
    registry_sources = _registry_index(registry)
    try:
        base_source = registry_sources[_BASE_SOURCE_ID]
        data_source = registry_sources[_DATA_SOURCE_ID]
        teacher_source = registry_sources[_TEACHER_SOURCE_ID]
    except KeyError as exc:
        raise ReleaseProvenanceError(
            f"source registry is missing required source {exc.args[0]}"
        ) from None
    _validate_registry_source(base_source, source_id=_BASE_SOURCE_ID, expected_kind="base_model")
    _validate_registry_source(
        data_source,
        source_id=_DATA_SOURCE_ID,
        expected_kind="deterministic_synthetic_dataset",
    )
    _validate_registry_source(
        teacher_source,
        source_id=_TEACHER_SOURCE_ID,
        expected_kind="self_hosted_teacher",
    )
    if data_source.get("forced_pool") != _EXPECTED_DATA_POOL:
        raise ReleaseProvenanceError("direct data source is not admitted to public_release_pool")
    if teacher_source.get("input_class") != "placeholder_only":
        raise ReleaseProvenanceError("upstream teacher input must remain placeholder-only")
    if teacher_source.get("output_role") != "synthetic_template_candidates":
        raise ReleaseProvenanceError("upstream teacher has an unexpected output role")
    if teacher_source.get("admission_scope") != "accepted_placeholder_templates_only":
        raise ReleaseProvenanceError("upstream teacher admission scope is too broad")
    if teacher_source.get("raw_output_training_allowed") is not False:
        raise ReleaseProvenanceError("raw upstream teacher output must not be training-admitted")

    asset_summary = _validate_template_asset(
        asset,
        asset_file_sha256=asset_file_sha256,
        data_source=data_source,
        teacher_source=teacher_source,
    )
    datasets = _mapping(training.get("datasets"), field="training manifest.datasets")
    if set(datasets) != set(_EXPECTED_SPLIT_COUNTS):
        raise ReleaseProvenanceError("training manifest must contain only train and validation")
    split_hashes = {
        split_name: _validate_split_summary(
            split_name,
            _mapping(datasets.get(split_name), field=f"datasets.{split_name}"),
            count=count,
            data_revision=str(data_source["revision"]),
            data_license=str(data_source["declared_license"]),
        )
        for split_name, count in _EXPECTED_SPLIT_COUNTS.items()
    }
    total_sample_count = sum(_EXPECTED_SPLIT_COUNTS.values())
    registry_identity = {
        "schema_version": registry["schema_version"],
        "registry_version": registry["registry_version"],
        "sha256": registry_file_sha256,
    }
    privacy = {
        "contains_document_ids": False,
        "contains_entity_values": False,
        "contains_paths": False,
        "contains_prompts_or_responses": False,
        "contains_raw_text": False,
    }
    data_provenance = _self_hashed(
        {
            "schema_version": 1,
            "artifact_type": "release_data_provenance",
            "training_manifest_sha256": training_manifest_sha256,
            "training_manifest_file_sha256": training_file_sha256,
            "source_registry": registry_identity,
            "sources": [
                {
                    "source_id": _DATA_SOURCE_ID,
                    "source_kind": data_source["kind"],
                    "revision": data_source["revision"],
                    "declared_license": data_source["declared_license"],
                    "pool": _EXPECTED_DATA_POOL,
                    "sample_count": total_sample_count,
                    "usage_counts": {
                        "gradient_training": _EXPECTED_SPLIT_COUNTS["train"],
                        "validation_early_stopping_and_calibration": (
                            _EXPECTED_SPLIT_COUNTS["validation"]
                        ),
                        "frozen_test_training_use": 0,
                    },
                    "split_sha256": split_hashes,
                    "synthetic": True,
                    "contains_real_personal_data": False,
                    "public_weight_training_allowed": True,
                    "upstream_generation": {
                        "source_id": _TEACHER_SOURCE_ID,
                        "role": _TEACHER_ROLE,
                        "accepted_candidate_count": asset_summary["accepted_candidate_count"],
                        "direct_span_annotation": False,
                        "direct_pseudo_labeling": False,
                    },
                    "template_asset": {
                        "asset_id": asset_summary["asset_id"],
                        "asset_version": asset_summary["asset_version"],
                        "revision": asset_summary["asset_revision"],
                        "curation_audit_sha256": asset_summary["curation_audit_sha256"],
                    },
                }
            ],
            "total_sample_count": total_sample_count,
            "privacy": privacy,
        }
    )
    teacher_provenance = _self_hashed(
        {
            "schema_version": 1,
            "artifact_type": "release_teacher_provenance",
            "training_manifest_sha256": training_manifest_sha256,
            "training_manifest_file_sha256": training_file_sha256,
            "source_registry": registry_identity,
            "teacher_used": True,
            "external_api_used": False,
            "external_api_teacher_used": False,
            "teachers": [
                {
                    "source_id": _TEACHER_SOURCE_ID,
                    "revision": teacher_source["revision"],
                    "declared_license": teacher_source["declared_license"],
                    "used_for_training": True,
                    "role": _TEACHER_ROLE,
                    "input_class": "placeholder_only",
                    "accepted_placeholder_templates_only": True,
                    "raw_outputs_used_for_training": False,
                    "span_teacher": False,
                    "pseudo_label_teacher": False,
                    "external_api": False,
                    "candidate_payload_sha256": asset_summary["candidate_payload_sha256"],
                    "template_asset_revision": asset_summary["asset_revision"],
                    "curation_audit_sha256": asset_summary["curation_audit_sha256"],
                    "review_method": asset_summary["review_method"],
                    "reviewed_candidate_count": asset_summary["reviewed_candidate_count"],
                    "accepted_candidate_count": asset_summary["accepted_candidate_count"],
                    "rejected_candidate_count": asset_summary["rejected_candidate_count"],
                }
            ],
            "privacy": privacy,
        }
    )
    return data_provenance, teacher_provenance


def _encode_artifact(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json_atomic(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            delete=False,
            prefix=f".{destination.name}.",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_encode_artifact(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def export_release_provenance(
    *,
    training_manifest_path: Path,
    source_registry_path: Path,
    template_asset_path: Path,
    data_output: Path,
    teacher_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and atomically write both deterministic provenance artifacts."""

    destinations = {data_output.expanduser().resolve(), teacher_output.expanduser().resolve()}
    if len(destinations) != 2:
        raise ReleaseProvenanceError("data and teacher outputs must be distinct")
    inputs = {
        training_manifest_path.expanduser().resolve(),
        source_registry_path.expanduser().resolve(),
        template_asset_path.expanduser().resolve(),
    }
    if destinations & inputs:
        raise ReleaseProvenanceError("provenance outputs must not replace an input")
    data, teacher = build_release_provenance(
        training_manifest_path=training_manifest_path,
        source_registry_path=source_registry_path,
        template_asset_path=template_asset_path,
    )
    _write_json_atomic(data, data_output)
    _write_json_atomic(teacher, teacher_output)
    return data, teacher


def main() -> int:
    args = _parser().parse_args()
    try:
        data, teacher = export_release_provenance(
            training_manifest_path=args.training_manifest,
            source_registry_path=args.source_registry,
            template_asset_path=args.template_asset,
            data_output=args.data_output,
            teacher_output=args.teacher_output,
        )
    except (OSError, ReleaseProvenanceError, TypeError, ValueError) as exc:
        print(f"release provenance export blocked: {exc}", file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "data_provenance_sha256": data["manifest_sha256"],
                "teacher_provenance_sha256": teacher["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
