"""Factories and content-addressed inputs for six-way cascade ablations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

from .ablation_profiles import AblationRuntimeProfile
from .config import CascadeConfig
from .context import ContextEnhancer
from .pipeline import CascadePipeline, Recognizer, Validator
from .routing import canonicalize_entity_type, default_routes

NER_MAPPING_SCHEMA_VERSION = "pii-zh.closed-bio-mapping.v1"
NER_ARTIFACT_BINDING_SCHEMA_VERSION = "pii-zh.local-ner-artifact-binding.v1"
CONTEXT_CONFIG_SCHEMA_VERSION = "pii-zh.context-config.v1"
PROJECT_CANDIDATE_FREEZE_SCHEMA_VERSION = "pii-zh.cascade-candidate-freeze.v1"
PROJECT_SELECTION_RECEIPT_SCHEMA_VERSION = "pii-zh.cascade-checkpoint-selection.v1"
PROJECT_RELEASE_GATE_RECEIPT_SCHEMA_VERSION = "pii-zh.cascade-release-gate.v1"
NER_ADMISSION_RECEIPT_SCHEMA_VERSION = "pii-zh.ner-comparator-admission.v1"
EXECUTION_SOURCE_CLOSURE_SCHEMA_VERSION = "pii-zh.cascade-source-closure.v1"
EXECUTION_RUNTIME_LOCK_SCHEMA_VERSION = "pii-zh.cascade-runtime-lock.v1"
ProjectRunIntent = Literal["development-non-claim", "frozen-confirmatory"]
_SAFE_SOURCE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_PUBLIC_MODEL_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
_SAFE_LICENSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
_SAFE_RUNTIME_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, field: str) -> tuple[Path, dict[str, Any], str]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{field} must be a regular local JSON file")
    try:
        resolved = candidate.resolve(strict=True)
        payload = resolved.read_bytes()
        document = json.loads(payload)
    except (OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid local JSON file") from exc
    if not resolved.is_file() or not isinstance(document, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return resolved, document, hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ClosedBioMapping:
    mapping_id: str
    source_to_target: Mapping[str, str]
    config_file_sha256: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class NerArtifactIdentity:
    artifact_id: str
    binding_file_sha256: str
    binding_sha256: str
    files_combined_sha256: str
    mapping_config_sha256: str

    def to_report(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "binding_file_sha256": self.binding_file_sha256,
            "binding_sha256": self.binding_sha256,
            "files_combined_sha256": self.files_combined_sha256,
            "mapping_config_sha256": self.mapping_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextConfigIdentity:
    context_id: str
    config_file_sha256: str
    config_sha256: str

    def to_report(self) -> dict[str, str]:
        return {
            "context_id": self.context_id,
            "config_file_sha256": self.config_file_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProjectModelIdentity:
    run_intent: ProjectRunIntent
    training_manifest_file_sha256: str
    training_manifest_sha256: str
    artifact_files_combined_sha256: str
    label_schema_sha256: str
    taxonomy_file_sha256: str
    taxonomy_version: str
    attention_mode: str
    release_eligible: bool
    expected_id2label: tuple[tuple[int, str], ...]
    candidate_freeze_id: str | None = None
    candidate_freeze_file_sha256: str | None = None
    candidate_freeze_sha256: str | None = None
    selection_id: str | None = None
    selection_file_sha256: str | None = None
    selection_sha256: str | None = None
    release_gate_id: str | None = None
    release_gate_file_sha256: str | None = None
    release_gate_sha256: str | None = None
    selected_checkpoint_id: str | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "run_intent": self.run_intent,
            "training_manifest_file_sha256": self.training_manifest_file_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
            "artifact_files_combined_sha256": self.artifact_files_combined_sha256,
            "label_schema_sha256": self.label_schema_sha256,
            "taxonomy_file_sha256": self.taxonomy_file_sha256,
            "taxonomy_version": self.taxonomy_version,
            "attention_mode": self.attention_mode,
            "release_eligible": self.release_eligible,
            "model_bio_label_count": len(self.expected_id2label),
            "canonical_entity_label_count": (len(self.expected_id2label) - 1) // 2,
            "candidate_freeze_id": self.candidate_freeze_id,
            "candidate_freeze_file_sha256": self.candidate_freeze_file_sha256,
            "candidate_freeze_sha256": self.candidate_freeze_sha256,
            "selection_id": self.selection_id,
            "selection_file_sha256": self.selection_file_sha256,
            "selection_sha256": self.selection_sha256,
            "release_gate_id": self.release_gate_id,
            "release_gate_file_sha256": self.release_gate_file_sha256,
            "release_gate_sha256": self.release_gate_sha256,
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "confirmatory_gate_status": (
                "PASS" if self.run_intent == "frozen-confirmatory" else "NOT_APPLICABLE"
            ),
        }


@dataclass(frozen=True, slots=True)
class NerAdmissionIdentity:
    run_intent: ProjectRunIntent
    admission_status: str
    admission_id: str | None = None
    admission_file_sha256: str | None = None
    admission_sha256: str | None = None
    model_id: str | None = None
    immutable_revision: str | None = None
    license_id: str | None = None
    license_review_status: str | None = None
    redistribution_status: str | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "run_intent": self.run_intent,
            "admission_status": self.admission_status,
            "admission_id": self.admission_id,
            "admission_file_sha256": self.admission_file_sha256,
            "admission_sha256": self.admission_sha256,
            "model_id": self.model_id,
            "immutable_revision": self.immutable_revision,
            "license_id": self.license_id,
            "license_review_status": self.license_review_status,
            "redistribution_status": self.redistribution_status,
            "confirmatory_admitted": self.run_intent == "frozen-confirmatory",
        }


@dataclass(frozen=True, slots=True)
class ExecutionEvidenceIdentity:
    run_intent: ProjectRunIntent
    source_closure_status: str
    runtime_lock_status: str
    source_closure_id: str | None = None
    source_closure_file_sha256: str | None = None
    source_closure_sha256: str | None = None
    runtime_lock_id: str | None = None
    runtime_lock_file_sha256: str | None = None
    runtime_lock_sha256: str | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "run_intent": self.run_intent,
            "source_closure_status": self.source_closure_status,
            "runtime_lock_status": self.runtime_lock_status,
            "source_closure_id": self.source_closure_id,
            "source_closure_file_sha256": self.source_closure_file_sha256,
            "source_closure_sha256": self.source_closure_sha256,
            "runtime_lock_id": self.runtime_lock_id,
            "runtime_lock_file_sha256": self.runtime_lock_file_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "confirmatory_execution_evidence_verified": (
                self.run_intent == "frozen-confirmatory"
            ),
        }


def _project_label_contract() -> tuple[dict[str, int], dict[int, str], str, str]:
    from pii_zh.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    labels = ["O"]
    for entity in taxonomy.label_sets["core"]:
        labels.extend((f"B-{entity.name}", f"I-{entity.name}"))
    label2id = {label: index for index, label in enumerate(labels)}
    if len(taxonomy.label_sets["core"]) != 24 or len(label2id) != 49:
        raise RuntimeError("packaged taxonomy no longer defines the frozen 24-class BIO scope")
    taxonomy_bytes = files("pii_zh.taxonomy").joinpath("taxonomy.yaml").read_bytes()
    return (
        label2id,
        {index: label for label, index in label2id.items()},
        hashlib.sha256(taxonomy_bytes).hexdigest(),
        taxonomy.taxonomy_version,
    )


def _normalize_json_id2label(value: object) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise ValueError("project model config must contain id2label")
    normalized: dict[int, str] = {}
    for raw_index, raw_label in value.items():
        if isinstance(raw_index, bool) or not isinstance(raw_index, str | int):
            raise ValueError("project model id2label keys must be integers")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("project model id2label keys must be integers") from exc
        if str(index) != str(raw_index) or index in normalized:
            raise ValueError("project model id2label keys must be canonical and unique")
        if not isinstance(raw_label, str):
            raise ValueError("project model id2label values must be strings")
        normalized[index] = raw_label
    return normalized


def verify_project_model_for_ablation(
    model_root: Path,
    *,
    run_intent: ProjectRunIntent,
    candidate_freeze_path: Path | None,
    candidate_selection_path: Path | None,
    release_gate_path: Path | None,
) -> ProjectModelIdentity:
    """Verify 24-class model identity and optional confirmatory release gate."""

    from pii_zh.evaluation.tw_pii_posthoc import verify_completed_qwen_model

    if run_intent not in {"development-non-claim", "frozen-confirmatory"}:
        raise ValueError("project run intent is invalid")
    confirmatory_paths = (
        candidate_freeze_path,
        candidate_selection_path,
        release_gate_path,
    )
    if run_intent == "development-non-claim" and any(
        path is not None for path in confirmatory_paths
    ):
        raise ValueError("development runs must not present confirmatory model receipts")
    if run_intent == "frozen-confirmatory" and any(
        path is None for path in confirmatory_paths
    ):
        raise ValueError(
            "frozen-confirmatory runs require candidate freeze, selection, "
            "and release-gate receipts"
        )
    try:
        resolved_root = model_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("project model path must be an existing local directory") from exc
    if model_root.expanduser().is_symlink() or not resolved_root.is_dir():
        raise ValueError("project model path must be a non-symlink local directory")
    verified = verify_completed_qwen_model(resolved_root)
    _, manifest, manifest_file_sha256 = _load_json_object(
        resolved_root / "training_manifest.json", field="project training manifest"
    )
    _, config, _ = _load_json_object(resolved_root / "config.json", field="project model config")
    claimed_manifest_sha = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    output_artifact = manifest.get("output_artifact")
    artifact_combined = (
        output_artifact.get("artifact_files_combined_sha256")
        if isinstance(output_artifact, Mapping)
        else None
    )
    label2id, expected_id2label, taxonomy_file_sha256, taxonomy_version = (
        _project_label_contract()
    )
    label_schema_sha256 = _canonical_hash(label2id)
    config_id2label = _normalize_json_id2label(config.get("id2label"))
    if (
        not isinstance(claimed_manifest_sha, str)
        or _canonical_hash(unsigned_manifest) != claimed_manifest_sha
        or manifest.get("status") != "completed"
        or manifest.get("label2id") != label2id
        or config.get("label2id") != label2id
        or config_id2label != expected_id2label
        or manifest.get("label_schema_sha256") != label_schema_sha256
        or verified.get("label_schema_sha256") != label_schema_sha256
        or verified.get("training_manifest_file_sha256") != manifest_file_sha256
        or verified.get("training_manifest_sha256") != claimed_manifest_sha
        or not isinstance(artifact_combined, str)
        or _HEX_SHA256.fullmatch(artifact_combined) is None
        or verified.get("output_artifact_sha256") != _canonical_hash(output_artifact)
        or verified.get("attention_mode") != manifest.get("attention_mode")
        or config.get("pii_attention_mode") != manifest.get("attention_mode")
    ):
        raise ValueError("project model does not match the frozen 24-class manifest contract")
    attention_mode = cast(str, manifest["attention_mode"])
    release_eligible = bool(
        manifest.get("release_eligible") is True
        and config.get("pii_release_eligible") is True
        and attention_mode == "full"
    )

    freeze_identity: dict[str, str | None] = {
        "candidate_freeze_id": None,
        "candidate_freeze_file_sha256": None,
        "candidate_freeze_sha256": None,
        "selection_id": None,
        "selection_file_sha256": None,
        "selection_sha256": None,
        "release_gate_id": None,
        "release_gate_file_sha256": None,
        "release_gate_sha256": None,
        "selected_checkpoint_id": None,
    }
    if run_intent == "frozen-confirmatory":
        assert candidate_freeze_path is not None
        assert candidate_selection_path is not None
        assert release_gate_path is not None

        _, selection, selection_file_sha256 = _load_json_object(
            candidate_selection_path, field="checkpoint selection receipt"
        )
        selection_fields = {
            "schema_version",
            "report_type",
            "selection_id",
            "status",
            "selected_checkpoint_id",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "artifact_files_combined_sha256",
            "label_schema_sha256",
            "taxonomy_file_sha256",
            "taxonomy_version",
            "attention_mode",
            "release_eligible",
            "all_selection_gates_passed",
            "receipt_sha256",
        }
        if set(selection) != selection_fields:
            raise ValueError("checkpoint selection fields do not match the frozen schema")
        selection_sha256 = selection.pop("receipt_sha256")
        selection_id = selection.get("selection_id")
        selected_checkpoint_id = selection.get("selected_checkpoint_id")
        if (
            selection.get("schema_version") != PROJECT_SELECTION_RECEIPT_SCHEMA_VERSION
            or selection.get("report_type") != "pii_zh_cascade_checkpoint_selection"
            or selection.get("status") != "selected_release_checkpoint"
            or not isinstance(selection_id, str)
            or _SAFE_ID.fullmatch(selection_id) is None
            or not isinstance(selected_checkpoint_id, str)
            or _SAFE_ID.fullmatch(selected_checkpoint_id) is None
            or selection.get("training_manifest_file_sha256") != manifest_file_sha256
            or selection.get("training_manifest_sha256") != claimed_manifest_sha
            or selection.get("artifact_files_combined_sha256") != artifact_combined
            or selection.get("label_schema_sha256") != label_schema_sha256
            or selection.get("taxonomy_file_sha256") != taxonomy_file_sha256
            or selection.get("taxonomy_version") != taxonomy_version
            or selection.get("attention_mode") != "full"
            or selection.get("release_eligible") is not True
            or selection.get("all_selection_gates_passed") is not True
            or not isinstance(selection_sha256, str)
            or _canonical_hash(selection) != selection_sha256
            or not release_eligible
        ):
            raise ValueError("checkpoint selection does not bind the loaded release model")

        _, release_gate, release_gate_file_sha256 = _load_json_object(
            release_gate_path, field="release-gate receipt"
        )
        release_gate_fields = {
            "schema_version",
            "report_type",
            "release_gate_id",
            "status",
            "selection_id",
            "selected_checkpoint_id",
            "selection_file_sha256",
            "selection_sha256",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "artifact_files_combined_sha256",
            "label_schema_sha256",
            "taxonomy_file_sha256",
            "taxonomy_version",
            "attention_mode",
            "release_eligible",
            "all_release_gates_passed",
            "receipt_sha256",
        }
        if set(release_gate) != release_gate_fields:
            raise ValueError("release-gate fields do not match the frozen schema")
        release_gate_sha256 = release_gate.pop("receipt_sha256")
        release_gate_id = release_gate.get("release_gate_id")
        if (
            release_gate.get("schema_version")
            != PROJECT_RELEASE_GATE_RECEIPT_SCHEMA_VERSION
            or release_gate.get("report_type") != "pii_zh_cascade_release_gate"
            or release_gate.get("status") != "PASS"
            or not isinstance(release_gate_id, str)
            or _SAFE_ID.fullmatch(release_gate_id) is None
            or release_gate.get("selection_id") != selection_id
            or release_gate.get("selected_checkpoint_id") != selected_checkpoint_id
            or release_gate.get("selection_file_sha256") != selection_file_sha256
            or release_gate.get("selection_sha256") != selection_sha256
            or release_gate.get("training_manifest_file_sha256") != manifest_file_sha256
            or release_gate.get("training_manifest_sha256") != claimed_manifest_sha
            or release_gate.get("artifact_files_combined_sha256") != artifact_combined
            or release_gate.get("label_schema_sha256") != label_schema_sha256
            or release_gate.get("taxonomy_file_sha256") != taxonomy_file_sha256
            or release_gate.get("taxonomy_version") != taxonomy_version
            or release_gate.get("attention_mode") != "full"
            or release_gate.get("release_eligible") is not True
            or release_gate.get("all_release_gates_passed") is not True
            or not isinstance(release_gate_sha256, str)
            or _canonical_hash(release_gate) != release_gate_sha256
        ):
            raise ValueError("release-gate PASS receipt does not bind the selected model")

        _, freeze, freeze_file_sha256 = _load_json_object(
            candidate_freeze_path, field="candidate freeze receipt"
        )
        expected_fields = {
            "schema_version",
            "report_type",
            "candidate_freeze_id",
            "selection_id",
            "release_gate_id",
            "selected_checkpoint_id",
            "status",
            "selection_file_sha256",
            "selection_sha256",
            "release_gate_file_sha256",
            "release_gate_sha256",
            "training_manifest_file_sha256",
            "training_manifest_sha256",
            "artifact_files_combined_sha256",
            "label_schema_sha256",
            "taxonomy_file_sha256",
            "taxonomy_version",
            "attention_mode",
            "release_eligible",
            "all_release_gates_passed",
            "frozen_before_hidden_access",
            "hidden_access_count_at_freeze",
            "receipt_sha256",
        }
        if set(freeze) != expected_fields:
            raise ValueError("candidate freeze fields do not match the frozen schema")
        receipt_sha256 = freeze.pop("receipt_sha256")
        safe_ids = (
            freeze.get("candidate_freeze_id"),
            freeze.get("selection_id"),
            freeze.get("release_gate_id"),
            freeze.get("selected_checkpoint_id"),
        )
        if (
            freeze.get("schema_version") != PROJECT_CANDIDATE_FREEZE_SCHEMA_VERSION
            or freeze.get("report_type") != "pii_zh_cascade_candidate_freeze"
            or freeze.get("status") != "frozen_release_candidate"
            or any(
                not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None
                for value in safe_ids
            )
            or not isinstance(receipt_sha256, str)
            or _canonical_hash(freeze) != receipt_sha256
            or freeze.get("selection_id") != selection_id
            or freeze.get("release_gate_id") != release_gate_id
            or freeze.get("selected_checkpoint_id") != selected_checkpoint_id
            or freeze.get("selection_file_sha256") != selection_file_sha256
            or freeze.get("selection_sha256") != selection_sha256
            or freeze.get("release_gate_file_sha256") != release_gate_file_sha256
            or freeze.get("release_gate_sha256") != release_gate_sha256
            or freeze.get("training_manifest_file_sha256") != manifest_file_sha256
            or freeze.get("training_manifest_sha256") != claimed_manifest_sha
            or freeze.get("artifact_files_combined_sha256") != artifact_combined
            or freeze.get("label_schema_sha256") != label_schema_sha256
            or freeze.get("taxonomy_file_sha256") != taxonomy_file_sha256
            or freeze.get("taxonomy_version") != taxonomy_version
            or freeze.get("attention_mode") != "full"
            or freeze.get("release_eligible") is not True
            or freeze.get("all_release_gates_passed") is not True
            or freeze.get("frozen_before_hidden_access") is not True
            or freeze.get("hidden_access_count_at_freeze") != 0
            or not release_eligible
        ):
            raise ValueError("candidate freeze release-gate cross-binding is invalid")
        freeze_identity = {
            "candidate_freeze_id": cast(str, freeze["candidate_freeze_id"]),
            "candidate_freeze_file_sha256": freeze_file_sha256,
            "candidate_freeze_sha256": receipt_sha256,
            "selection_id": selection_id,
            "selection_file_sha256": selection_file_sha256,
            "selection_sha256": selection_sha256,
            "release_gate_id": release_gate_id,
            "release_gate_file_sha256": release_gate_file_sha256,
            "release_gate_sha256": release_gate_sha256,
            "selected_checkpoint_id": selected_checkpoint_id,
        }

    return ProjectModelIdentity(
        run_intent=run_intent,
        training_manifest_file_sha256=manifest_file_sha256,
        training_manifest_sha256=claimed_manifest_sha,
        artifact_files_combined_sha256=artifact_combined,
        label_schema_sha256=label_schema_sha256,
        taxonomy_file_sha256=taxonomy_file_sha256,
        taxonomy_version=taxonomy_version,
        attention_mode=attention_mode,
        release_eligible=release_eligible,
        expected_id2label=tuple(expected_id2label.items()),
        **freeze_identity,
    )


def verify_loaded_project_predictor(
    pipeline: CascadePipeline,
    identity: ProjectModelIdentity,
) -> None:
    """Cross-check the actually loaded predictor against the pre-load contract."""

    recognizer = pipeline.model_recognizer
    predictor = getattr(recognizer, "predictor", None)
    raw_id2label = getattr(predictor, "id2label", None)
    if not isinstance(raw_id2label, Mapping):
        raise ValueError("loaded project predictor lacks id2label")
    normalized: dict[int, str] = {}
    for raw_index, raw_label in raw_id2label.items():
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("loaded predictor id2label keys must be integers")
        if not isinstance(raw_label, str):
            raise ValueError("loaded predictor id2label values must be strings")
        normalized[raw_index] = raw_label
    if (
        tuple(sorted(normalized.items())) != identity.expected_id2label
        or getattr(predictor, "attention_mode", None) != identity.attention_mode
    ):
        raise ValueError("loaded predictor identity differs from the frozen model contract")


def verify_ner_admission_receipt(
    receipt_path: Path | None,
    *,
    run_intent: ProjectRunIntent,
    artifact: NerArtifactIdentity,
    mapping: ClosedBioMapping,
) -> NerAdmissionIdentity:
    """Bind a confirmatory BYO NER to public, immutable, license-reviewed provenance."""

    if run_intent not in {"development-non-claim", "frozen-confirmatory"}:
        raise ValueError("NER run intent is invalid")
    if run_intent == "development-non-claim":
        if receipt_path is not None:
            raise ValueError("development NER runs must not present an admission receipt")
        return NerAdmissionIdentity(
            run_intent=run_intent,
            admission_status="DEVELOPMENT_NON_CLAIM_UNREVIEWED",
        )
    if receipt_path is None:
        raise ValueError("frozen-confirmatory NER runs require an admission receipt")

    _, receipt, receipt_file_sha256 = _load_json_object(
        receipt_path, field="NER admission receipt"
    )
    expected_fields = {
        "schema_version",
        "report_type",
        "admission_id",
        "status",
        "model_id",
        "immutable_revision",
        "license_id",
        "license_review_status",
        "redistribution_status",
        "artifact_files_combined_sha256",
        "mapping_config_sha256",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("NER admission receipt fields do not match the frozen schema")
    receipt_sha256 = receipt.pop("receipt_sha256")
    admission_id = receipt.get("admission_id")
    model_id = receipt.get("model_id")
    immutable_revision = receipt.get("immutable_revision")
    license_id = receipt.get("license_id")
    if (
        receipt.get("schema_version") != NER_ADMISSION_RECEIPT_SCHEMA_VERSION
        or receipt.get("report_type") != "pii_zh_ner_comparator_admission"
        or receipt.get("status") != "approved_for_confirmatory_comparison"
        or not isinstance(admission_id, str)
        or _SAFE_ID.fullmatch(admission_id) is None
        or not isinstance(model_id, str)
        or _SAFE_PUBLIC_MODEL_ID.fullmatch(model_id) is None
        or not isinstance(immutable_revision, str)
        or _IMMUTABLE_REVISION.fullmatch(immutable_revision) is None
        or not isinstance(license_id, str)
        or _SAFE_LICENSE_ID.fullmatch(license_id) is None
        or receipt.get("license_review_status") != "approved"
        or receipt.get("redistribution_status") != "byo_not_redistributed"
        or receipt.get("artifact_files_combined_sha256")
        != artifact.files_combined_sha256
        or receipt.get("mapping_config_sha256") != mapping.config_sha256
        or not isinstance(receipt_sha256, str)
        or _canonical_hash(receipt) != receipt_sha256
    ):
        raise ValueError("NER admission receipt provenance binding is invalid")
    return NerAdmissionIdentity(
        run_intent=run_intent,
        admission_status="APPROVED_FOR_CONFIRMATORY_COMPARISON",
        admission_id=admission_id,
        admission_file_sha256=receipt_file_sha256,
        admission_sha256=receipt_sha256,
        model_id=model_id,
        immutable_revision=immutable_revision,
        license_id=license_id,
        license_review_status="approved",
        redistribution_status="byo_not_redistributed",
    )


def verify_execution_evidence(
    *,
    run_intent: ProjectRunIntent,
    ablation_id: str,
    profile_set_file_sha256: str,
    implementation_sha256: str,
    runtime_input_identity_sha256: str,
    runtime_versions: Mapping[str, str],
    source_closure_path: Path | None,
    runtime_lock_path: Path | None,
) -> ExecutionEvidenceIdentity:
    """Fail closed on source/runtime freezing for a confirmatory adapter run."""

    if run_intent not in {"development-non-claim", "frozen-confirmatory"}:
        raise ValueError("execution run intent is invalid")
    if run_intent == "development-non-claim":
        if source_closure_path is not None or runtime_lock_path is not None:
            raise ValueError("development runs must not present confirmatory execution evidence")
        return ExecutionEvidenceIdentity(
            run_intent=run_intent,
            source_closure_status="NOT_PROVIDED_DEVELOPMENT_NON_CLAIM",
            runtime_lock_status="NOT_PROVIDED_DEVELOPMENT_NON_CLAIM",
        )
    if source_closure_path is None or runtime_lock_path is None:
        raise ValueError(
            "frozen-confirmatory runs require source closure and runtime lock receipts"
        )
    if (
        not isinstance(ablation_id, str)
        or _SAFE_ID.fullmatch(ablation_id) is None
        or _HEX_SHA256.fullmatch(profile_set_file_sha256) is None
        or _HEX_SHA256.fullmatch(implementation_sha256) is None
        or _HEX_SHA256.fullmatch(runtime_input_identity_sha256) is None
    ):
        raise ValueError("execution evidence input identities are invalid")
    expected_runtime_keys = {
        "python",
        "pii-zh-qwen",
        "torch",
        "transformers",
        "presidio-analyzer",
    }
    if set(runtime_versions) != expected_runtime_keys or any(
        not isinstance(value, str) or _SAFE_RUNTIME_VERSION.fullmatch(value) is None
        for value in runtime_versions.values()
    ):
        raise ValueError("runtime versions do not match the safe frozen schema")

    _, closure, closure_file_sha256 = _load_json_object(
        source_closure_path, field="cascade source closure"
    )
    closure_fields = {
        "schema_version",
        "report_type",
        "closure_id",
        "status",
        "ablation_id",
        "profile_set_file_sha256",
        "implementation_sha256",
        "runtime_input_identity_sha256",
        "frozen_before_hidden_access",
        "hidden_access_count_at_freeze",
        "receipt_sha256",
    }
    if set(closure) != closure_fields:
        raise ValueError("cascade source closure fields do not match the frozen schema")
    closure_sha256 = closure.pop("receipt_sha256")
    closure_id = closure.get("closure_id")
    if (
        closure.get("schema_version") != EXECUTION_SOURCE_CLOSURE_SCHEMA_VERSION
        or closure.get("report_type") != "pii_zh_cascade_source_closure"
        or closure.get("status") != "frozen_before_hidden_access"
        or not isinstance(closure_id, str)
        or _SAFE_ID.fullmatch(closure_id) is None
        or closure.get("ablation_id") != ablation_id
        or closure.get("profile_set_file_sha256") != profile_set_file_sha256
        or closure.get("implementation_sha256") != implementation_sha256
        or closure.get("runtime_input_identity_sha256")
        != runtime_input_identity_sha256
        or closure.get("frozen_before_hidden_access") is not True
        or closure.get("hidden_access_count_at_freeze") != 0
        or not isinstance(closure_sha256, str)
        or _canonical_hash(closure) != closure_sha256
    ):
        raise ValueError("cascade source closure cross-binding is invalid")

    _, runtime_lock, runtime_lock_file_sha256 = _load_json_object(
        runtime_lock_path, field="cascade runtime lock"
    )
    runtime_fields = {
        "schema_version",
        "report_type",
        "runtime_lock_id",
        "status",
        "source_closure_sha256",
        "runtime_versions",
        "receipt_sha256",
    }
    if set(runtime_lock) != runtime_fields:
        raise ValueError("cascade runtime lock fields do not match the frozen schema")
    runtime_lock_sha256 = runtime_lock.pop("receipt_sha256")
    runtime_lock_id = runtime_lock.get("runtime_lock_id")
    if (
        runtime_lock.get("schema_version") != EXECUTION_RUNTIME_LOCK_SCHEMA_VERSION
        or runtime_lock.get("report_type") != "pii_zh_cascade_runtime_lock"
        or runtime_lock.get("status") != "frozen"
        or not isinstance(runtime_lock_id, str)
        or _SAFE_ID.fullmatch(runtime_lock_id) is None
        or runtime_lock.get("source_closure_sha256") != closure_sha256
        or runtime_lock.get("runtime_versions") != dict(runtime_versions)
        or not isinstance(runtime_lock_sha256, str)
        or _canonical_hash(runtime_lock) != runtime_lock_sha256
    ):
        raise ValueError("cascade runtime lock cross-binding is invalid")
    return ExecutionEvidenceIdentity(
        run_intent=run_intent,
        source_closure_status="VERIFIED",
        runtime_lock_status="VERIFIED",
        source_closure_id=closure_id,
        source_closure_file_sha256=closure_file_sha256,
        source_closure_sha256=closure_sha256,
        runtime_lock_id=runtime_lock_id,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
    )


def load_closed_bio_mapping(path: Path) -> ClosedBioMapping:
    """Load a self-hashed closed mapping; only PERSON is admitted for this baseline."""

    _, document, file_sha256 = _load_json_object(path, field="NER mapping config")
    if set(document) != {
        "schema_version",
        "mapping_id",
        "source_to_target",
        "config_sha256",
    }:
        raise ValueError("NER mapping config fields do not match the frozen schema")
    expected_hash = document.pop("config_sha256")
    if (
        document.get("schema_version") != NER_MAPPING_SCHEMA_VERSION
        or not isinstance(document.get("mapping_id"), str)
        or _SAFE_ID.fullmatch(cast(str, document["mapping_id"])) is None
        or not isinstance(expected_hash, str)
        or _canonical_hash(document) != expected_hash
    ):
        raise ValueError("NER mapping config identity/hash is invalid")
    raw_mapping = document.get("source_to_target")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ValueError("NER source_to_target must be a non-empty mapping")
    mapping: dict[str, str] = {}
    for source, target in raw_mapping.items():
        if (
            not isinstance(source, str)
            or _SAFE_SOURCE_LABEL.fullmatch(source) is None
            or target != "PERSON"
        ):
            raise ValueError("NER mapping may only map safe source labels to PERSON")
        mapping[source] = target
    return ClosedBioMapping(
        mapping_id=cast(str, document["mapping_id"]),
        source_to_target=mapping,
        config_file_sha256=file_sha256,
        config_sha256=expected_hash,
    )


def _artifact_inventory(model_root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for candidate in sorted(model_root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError("NER artifact must not contain symlinks")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError("NER artifact must contain regular files only")
        relative = candidate.relative_to(model_root).as_posix()
        inventory[relative] = _sha256_file(candidate)
    required = {"config.json", "model.safetensors", "tokenizer.json"}
    if not required <= set(inventory):
        raise ValueError("NER artifact lacks required safetensors/tokenizer files")
    if any(
        name.lower().endswith((".bin", ".pkl", ".pickle", ".pth", ".pt", ".ckpt"))
        for name in inventory
    ):
        raise ValueError("NER artifact contains a forbidden serialized model file")
    return inventory


def verify_ner_artifact_binding(
    model_root: Path,
    binding_path: Path,
    *,
    mapping: ClosedBioMapping,
) -> tuple[Path, NerArtifactIdentity]:
    """Verify an exact local file inventory and its mapping-config binding."""

    candidate_root = model_root.expanduser()
    if candidate_root.is_symlink():
        raise ValueError("NER model path must be an existing non-symlink directory")
    try:
        resolved_root = candidate_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("NER model path must be an existing local directory") from exc
    if not resolved_root.is_dir() or resolved_root.is_symlink():
        raise ValueError("NER model path must be an existing non-symlink directory")
    resolved_binding, document, binding_file_sha256 = _load_json_object(
        binding_path, field="NER artifact binding"
    )
    if resolved_root == resolved_binding.parent or resolved_root in resolved_binding.parents:
        raise ValueError("NER artifact binding must be stored outside the model artifact")
    expected_fields = {
        "schema_version",
        "artifact_id",
        "artifact_kind",
        "files",
        "files_combined_sha256",
        "mapping_config_sha256",
        "binding_sha256",
    }
    if set(document) != expected_fields:
        raise ValueError("NER artifact binding fields do not match the frozen schema")
    expected_binding_hash = document.pop("binding_sha256")
    artifact_id = document.get("artifact_id")
    recorded_files = document.get("files")
    recorded_combined = document.get("files_combined_sha256")
    if (
        document.get("schema_version") != NER_ARTIFACT_BINDING_SCHEMA_VERSION
        or document.get("artifact_kind") != "closed_bio_token_classifier"
        or not isinstance(artifact_id, str)
        or _SAFE_ID.fullmatch(artifact_id) is None
        or not isinstance(recorded_files, Mapping)
        or not isinstance(recorded_combined, str)
        or _HEX_SHA256.fullmatch(recorded_combined) is None
        or document.get("mapping_config_sha256") != mapping.config_sha256
        or not isinstance(expected_binding_hash, str)
        or _canonical_hash(document) != expected_binding_hash
    ):
        raise ValueError("NER artifact binding identity/hash is invalid")
    normalized_recorded: dict[str, str] = {}
    for name, digest in recorded_files.items():
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or _HEX_SHA256.fullmatch(digest) is None
        ):
            raise ValueError("NER artifact binding contains an invalid file entry")
        normalized_recorded[name] = digest
    actual_files = _artifact_inventory(resolved_root)
    if normalized_recorded != actual_files or _canonical_hash(actual_files) != recorded_combined:
        raise ValueError("NER artifact files do not match the content-addressed binding")
    return resolved_root, NerArtifactIdentity(
        artifact_id=artifact_id,
        binding_file_sha256=binding_file_sha256,
        binding_sha256=expected_binding_hash,
        files_combined_sha256=recorded_combined,
        mapping_config_sha256=mapping.config_sha256,
    )


def load_context_config(path: Path) -> tuple[ContextEnhancer, ContextConfigIdentity]:
    """Load one self-hashed context policy without exposing phrases in reports."""

    from pii_zh.presidio.context_enhancer import ChinesePhraseContextEnhancer

    _, document, file_sha256 = _load_json_object(path, field="context config")
    if set(document) != {"schema_version", "context_id", "context", "config_sha256"}:
        raise ValueError("context config fields do not match the frozen schema")
    expected_hash = document.pop("config_sha256")
    context_id = document.get("context_id")
    context = document.get("context")
    if (
        document.get("schema_version") != CONTEXT_CONFIG_SCHEMA_VERSION
        or not isinstance(context_id, str)
        or _SAFE_ID.fullmatch(context_id) is None
        or not isinstance(context, Mapping)
        or not isinstance(expected_hash, str)
        or _canonical_hash(document) != expected_hash
    ):
        raise ValueError("context config identity/hash is invalid")
    expected_context_fields = {
        "positive_context",
        "negative_context",
        "context_window",
        "positive_delta",
        "negative_delta",
    }
    if set(context) != expected_context_fields:
        raise ValueError("context inner fields do not match the frozen schema")
    allowed_entities = frozenset(route.entity_type for route in default_routes())
    phrase_count = 0
    entity_phrases: set[tuple[str, str]] = set()
    normalized_context: dict[str, Any] = {}
    for field in ("positive_context", "negative_context"):
        raw_phrases = context.get(field)
        if not isinstance(raw_phrases, Mapping):
            raise TypeError(f"context {field} must be a mapping")
        normalized_phrases: dict[str, list[str]] = {}
        for raw_entity, raw_values in raw_phrases.items():
            if not isinstance(raw_entity, str):
                raise TypeError("context entity keys must be strings")
            entity = canonicalize_entity_type(raw_entity)
            if entity != raw_entity or entity not in allowed_entities:
                raise ValueError("context entity keys must use the canonical 24-label scope")
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError("each context entity must contain at least one phrase")
            phrases: list[str] = []
            for phrase in raw_values:
                if (
                    not isinstance(phrase, str)
                    or phrase != phrase.strip()
                    or not 1 <= len(phrase) <= 20
                ):
                    raise ValueError("context phrases must be non-empty trimmed strings")
                key = (entity, phrase)
                if key in entity_phrases:
                    raise ValueError(
                        "context phrases must not conflict across positive and negative maps"
                    )
                entity_phrases.add(key)
                phrases.append(phrase)
            if len(phrases) != len(set(phrases)):
                raise ValueError("context phrases must not contain duplicates")
            normalized_phrases[entity] = phrases
            phrase_count += len(phrases)
        normalized_context[field] = normalized_phrases
    if phrase_count == 0:
        raise ValueError("context config must contain at least one phrase")
    context_window = context.get("context_window")
    if isinstance(context_window, bool) or not isinstance(context_window, int):
        raise TypeError("context_window must be an integer")
    if context_window < 1:
        raise ValueError("context_window must be positive")
    normalized_context["context_window"] = context_window
    for field in ("positive_delta", "negative_delta"):
        value = context.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{field} must be finite and between zero and one")
        normalized_context[field] = normalized
    enhancer = ChinesePhraseContextEnhancer.from_config(normalized_context)
    return enhancer, ContextConfigIdentity(
        context_id=context_id,
        config_file_sha256=file_sha256,
        config_sha256=expected_hash,
    )


def ablation_config(profile: AblationRuntimeProfile) -> CascadeConfig:
    if not isinstance(profile, AblationRuntimeProfile):
        raise TypeError("profile must be an AblationRuntimeProfile")
    config = CascadeConfig(
        profile_version=profile.stage_policy.policy_id,
        mode=profile.mode,
        rule_source=profile.rule_source,
        model_source=profile.model_source,
    )
    _validate_profile_scope(profile, config)
    return config


def _validate_profile_scope(profile: AblationRuntimeProfile, config: CascadeConfig) -> None:
    all_entities = frozenset(config.route_map)
    if profile.adapter_id in {"project_model_raw", "simple_union"}:
        if profile.stage_policy.branch_override("model") != all_entities:
            raise ValueError("raw/union model branch must admit all configured output labels")
    if profile.adapter_id == "simple_union":
        if profile.stage_policy.branch_override("rule") != all_entities:
            raise ValueError("simple_union rule branch must request the full configured scope")
    if profile.adapter_id == "presidio_zh_ner_cn_common":
        if profile.stage_policy.branch_override("model") != all_entities:
            raise ValueError("the native Presidio framework branch must expose all labels")
        if (
            profile.mode != "model-only"
            or profile.stage_policy.validators_enabled
            or profile.stage_policy.thresholds_enabled
            or profile.stage_policy.conflict_policy != "exact_duplicates_only"
        ):
            raise ValueError("the native Presidio profile must be Pipeline pass-through")


def load_closed_bio_ner_recognizer(
    model_root: Path,
    *,
    mapping: ClosedBioMapping,
    artifact: NerArtifactIdentity,
    device: str = "cpu",
    micro_batch_size: int = 16,
    max_tokens: int = 512,
    stride_fraction: float = 0.25,
    deduplication_policy: str = "high_overlap_v1",
) -> Recognizer:
    """Load the only supported BYO NER adapter from an already-verified root."""

    from pii_zh.inference.bio_token_classifier import load_closed_bio_token_classifier
    from pii_zh.presidio.bio_token_recognizer import BioTokenClassifierRecognizer

    predictor = load_closed_bio_token_classifier(
        model_root,
        source_to_target=mapping.source_to_target,
        protocol_id=mapping.mapping_id,
        device=device,
        micro_batch_size=micro_batch_size,
    )
    return BioTokenClassifierRecognizer(
        predictor=predictor,
        tokenizer=predictor.tokenizer,
        supported_entities=("PERSON",),
        default_threshold=0.0,
        max_tokens=max_tokens,
        stride_fraction=stride_fraction,
        model_version=artifact.artifact_id,
        source_name="ner:byo-closed-bio",
        deduplication_policy=deduplication_policy,
        name="BoundClosedBioChineseNerRecognizer",
    )


def build_ablation_pipeline(
    profile: AblationRuntimeProfile,
    *,
    recognizer: Recognizer | None = None,
    context_enhancer: ContextEnhancer | None = None,
    validators: Mapping[str, Validator] | None = None,
) -> CascadePipeline:
    """Build an injected ablation through the release ``CascadePipeline`` graph."""

    needs_recognizer = profile.recognizer_kind != "none"
    if needs_recognizer != (recognizer is not None):
        raise ValueError("profile recognizer requirement does not match injected recognizer")
    config = ablation_config(profile)
    return CascadePipeline(
        config=config,
        model_recognizer=recognizer,
        validators=validators,
        context_enhancer=context_enhancer,
        stage_policy=profile.stage_policy,
        allow_evaluation_profile=True,
    )


__all__ = [
    "CONTEXT_CONFIG_SCHEMA_VERSION",
    "EXECUTION_RUNTIME_LOCK_SCHEMA_VERSION",
    "EXECUTION_SOURCE_CLOSURE_SCHEMA_VERSION",
    "NER_ADMISSION_RECEIPT_SCHEMA_VERSION",
    "NER_ARTIFACT_BINDING_SCHEMA_VERSION",
    "NER_MAPPING_SCHEMA_VERSION",
    "PROJECT_CANDIDATE_FREEZE_SCHEMA_VERSION",
    "PROJECT_RELEASE_GATE_RECEIPT_SCHEMA_VERSION",
    "PROJECT_SELECTION_RECEIPT_SCHEMA_VERSION",
    "ClosedBioMapping",
    "ContextConfigIdentity",
    "ExecutionEvidenceIdentity",
    "NerAdmissionIdentity",
    "NerArtifactIdentity",
    "ProjectModelIdentity",
    "ProjectRunIntent",
    "ablation_config",
    "build_ablation_pipeline",
    "load_closed_bio_ner_recognizer",
    "load_closed_bio_mapping",
    "load_context_config",
    "verify_execution_evidence",
    "verify_loaded_project_predictor",
    "verify_ner_admission_receipt",
    "verify_ner_artifact_binding",
    "verify_project_model_for_ablation",
]
