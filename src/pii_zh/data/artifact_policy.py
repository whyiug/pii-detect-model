"""Fail-closed policy for quarantined or semantically retired artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DENYLIST_PATH = _REPOSITORY_ROOT / "configs/data/test_informed_artifact_denylist.yaml"
DEFAULT_ERRATUM_PATH = (
    _REPOSITORY_ROOT / "reports/baselines/macbert_24_targeted_repair_pilot_lineage_erratum.json"
)
DEFAULT_DENYLIST_FILE_SHA256 = "4b7f9f54d19e9932081dc31dabc423e54d6f1bae0ce03d6ad8b945f09b87cf4d"
DEFAULT_GENERATOR_DEFECT_DENYLIST_PATH = (
    _REPOSITORY_ROOT / "configs/data/generator_defect_artifact_denylist.yaml"
)
DEFAULT_GENERATOR_DEFECT_RECEIPT_PATH = (
    _REPOSITORY_ROOT
    / "reports/data_recipe_audits/targeted_repair_v2_generator_defect_retirement_receipt.json"
)
DEFAULT_GENERATOR_DEFECT_DIAGNOSTIC_PATH = (
    _REPOSITORY_ROOT
    / "reports/diagnostics/macbert_24_v2_dispatch_and_d0_gap_diagnostic.json"
)
DEFAULT_GENERATOR_DEFECT_DENYLIST_FILE_SHA256 = (
    "1484e64a422dc1acea84d971d2fcc13b182fa80108d96b18e146f2c423983cd8"
)
_DECLARATION_ONLY_KEYS = frozenset(
    {
        "denied_artifacts",
        "denied_ids",
        "forbidden_parent_artifacts",
        "prohibited_inputs",
    }
)
_PURPOSES = frozenset({"training", "release", "confirmatory_evaluation", "ancestry"})
_ALWAYS_DENIED_TRUE_KEYS = frozenset(
    {
        "artifact_test_informed_ancestry",
        "generator_defect_ancestry",
        "generator_defect_retired",
        "retired_due_to_generator_defect",
        "test_informed_ancestry",
    }
)
_PURPOSE_DENIED_FALSE_KEYS = MappingProxyType(
    {
        "training": frozenset(
            {
                "training_eligible",
                "public_weight_training_allowed",
                "weight_training_eligible",
            }
        ),
        "release": frozenset({"release_eligible", "pii_release_eligible"}),
        "confirmatory_evaluation": frozenset(
            {
                "confirmatory_evaluation_eligible",
                "external_benchmark_confirmatory_eligible",
                "PII_Bench_confirmatory_eligible",
            }
        ),
        "ancestry": frozenset({"ancestry_eligible", "v3_parent_eligible"}),
    }
)
_CONTROLLED_BOOLEAN_KEYS = frozenset(
    _ALWAYS_DENIED_TRUE_KEYS
    | frozenset(
        key for keys in _PURPOSE_DENIED_FALSE_KEYS.values() for key in keys
    )
)
_SEMANTIC_RETIREMENT_MARKERS = (
    "diagnostic_generator_defect",
    "generator_defect_retired",
    "retired_generator_defect",
    "test_informed_diagnostic",
)


class ArtifactPolicyError(ValueError):
    """Raised without reflecting paths, document content, or denied identifiers."""


@dataclass(frozen=True, slots=True)
class ArtifactDenylist:
    denylist_id: str
    denied_values: frozenset[str]
    erratum_file_sha256: str
    erratum_report_sha256: str
    raw: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ArtifactPolicyError(f"{field} must be a string-keyed mapping")
    return value


def _regular_file(path: Path, *, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArtifactPolicyError(f"{field} must be a regular non-symlink file")


def load_artifact_denylist(
    path: Path = DEFAULT_DENYLIST_PATH,
    *,
    erratum_path: Path = DEFAULT_ERRATUM_PATH,
) -> ArtifactDenylist:
    _regular_file(path, field="artifact denylist")
    if path == DEFAULT_DENYLIST_PATH and _sha256_file(path) != DEFAULT_DENYLIST_FILE_SHA256:
        raise ArtifactPolicyError("default artifact denylist byte identity changed")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ArtifactPolicyError("artifact denylist is not readable YAML") from exc
    root = _mapping(document, field="artifact denylist")
    if (
        root.get("schema_version") != 1
        or root.get("denylist_id") != "pii_zh_test_informed_artifacts_v1"
        or root.get("policy") != "fail_closed_for_training_release_and_confirmatory_evaluation"
    ):
        raise ArtifactPolicyError("artifact denylist identity changed")
    erratum = _mapping(root.get("erratum"), field="artifact denylist erratum")
    erratum_file_sha256 = str(erratum.get("file_sha256", ""))
    erratum_report_sha256 = str(erratum.get("report_sha256", ""))
    _regular_file(erratum_path, field="lineage erratum")
    if _sha256_file(erratum_path) != erratum_file_sha256:
        raise ArtifactPolicyError("lineage erratum byte identity changed")
    try:
        erratum_document = json.loads(erratum_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactPolicyError("lineage erratum is not readable JSON") from exc
    erratum_mapping = _mapping(erratum_document, field="lineage erratum")
    unsigned = dict(erratum_mapping)
    claimed = unsigned.pop("report_sha256", None)
    if claimed != erratum_report_sha256 or _canonical_hash(unsigned) != claimed:
        raise ArtifactPolicyError("lineage erratum logical identity changed")
    denied: set[str] = set()
    artifacts = _mapping(root.get("denied_artifacts"), field="denied artifacts")
    identifiers = _mapping(root.get("denied_ids"), field="denied ids")
    for group in (artifacts, identifiers):
        for values in group.values():
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ArtifactPolicyError("denylist entries must be arrays")
            if any(not isinstance(value, str) or not value for value in values):
                raise ArtifactPolicyError("denylist entry is invalid")
            denied.update(values)
    if not denied:
        raise ArtifactPolicyError("artifact denylist cannot be empty")
    return ArtifactDenylist(
        denylist_id=str(root["denylist_id"]),
        denied_values=frozenset(denied),
        erratum_file_sha256=erratum_file_sha256,
        erratum_report_sha256=erratum_report_sha256,
        raw=MappingProxyType(dict(root)),
    )


def load_generator_defect_denylist(
    path: Path = DEFAULT_GENERATOR_DEFECT_DENYLIST_PATH,
    *,
    receipt_path: Path = DEFAULT_GENERATOR_DEFECT_RECEIPT_PATH,
    diagnostic_path: Path = DEFAULT_GENERATOR_DEFECT_DIAGNOSTIC_PATH,
) -> ArtifactDenylist:
    """Load the independent generator-defect retirement policy."""

    _regular_file(path, field="generator-defect artifact denylist")
    if (
        path == DEFAULT_GENERATOR_DEFECT_DENYLIST_PATH
        and _sha256_file(path) != DEFAULT_GENERATOR_DEFECT_DENYLIST_FILE_SHA256
    ):
        raise ArtifactPolicyError("default generator-defect denylist byte identity changed")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ArtifactPolicyError("generator-defect denylist is not readable YAML") from exc
    root = _mapping(document, field="generator-defect artifact denylist")
    if (
        root.get("schema_version") != 1
        or root.get("denylist_id") != "pii_zh_generator_defect_artifacts_v1"
        or root.get("policy")
        != "fail_closed_for_training_release_confirmatory_evaluation_and_ancestry"
        or root.get("classification") != "diagnostic_generator_defect"
    ):
        raise ArtifactPolicyError("generator-defect denylist identity changed")

    receipt_contract = _mapping(
        root.get("retirement_receipt"), field="generator-defect retirement receipt"
    )
    _regular_file(receipt_path, field="generator-defect retirement receipt")
    if _sha256_file(receipt_path) != receipt_contract.get("file_sha256"):
        raise ArtifactPolicyError("generator-defect retirement receipt byte identity changed")
    try:
        receipt_document = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactPolicyError(
            "generator-defect retirement receipt is not readable JSON"
        ) from exc
    receipt = _mapping(receipt_document, field="generator-defect retirement receipt")
    unsigned_receipt = dict(receipt)
    claimed_receipt = unsigned_receipt.pop("receipt_sha256", None)
    if (
        claimed_receipt != receipt_contract.get("receipt_sha256")
        or _canonical_hash(unsigned_receipt) != claimed_receipt
        or receipt.get("status") != "atomically_relocated_and_retired"
    ):
        raise ArtifactPolicyError("generator-defect retirement receipt logical identity changed")
    effective = _mapping(receipt.get("effective_policy"), field="retirement effective policy")
    if any(
        effective.get(key) is not False
        for key in (
            "training_eligible",
            "release_eligible",
            "confirmatory_evaluation_eligible",
            "ancestry_eligible",
            "v3_parent_eligible",
        )
    ):
        raise ArtifactPolicyError("generator-defect retirement eligibility changed")

    diagnostic_contract = _mapping(root.get("diagnostic"), field="generator-defect diagnostic")
    _regular_file(diagnostic_path, field="generator-defect diagnostic")
    if _sha256_file(diagnostic_path) != diagnostic_contract.get("file_sha256"):
        raise ArtifactPolicyError("generator-defect diagnostic byte identity changed")
    try:
        diagnostic_document = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactPolicyError("generator-defect diagnostic is not readable JSON") from exc
    diagnostic = _mapping(diagnostic_document, field="generator-defect diagnostic")
    unsigned_diagnostic = dict(diagnostic)
    claimed_diagnostic = unsigned_diagnostic.pop("report_sha256", None)
    if (
        claimed_diagnostic != diagnostic_contract.get("report_sha256")
        or _canonical_hash(unsigned_diagnostic) != claimed_diagnostic
    ):
        raise ArtifactPolicyError("generator-defect diagnostic logical identity changed")

    denied: set[str] = set()
    artifacts = _mapping(root.get("denied_artifacts"), field="generator-defect artifacts")
    identifiers = _mapping(root.get("denied_ids"), field="generator-defect ids")
    for group in (artifacts, identifiers):
        for values in group.values():
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ArtifactPolicyError("generator-defect denylist entries must be arrays")
            if any(not isinstance(value, str) or not value for value in values):
                raise ArtifactPolicyError("generator-defect denylist entry is invalid")
            denied.update(values)
    if not denied:
        raise ArtifactPolicyError("generator-defect denylist cannot be empty")
    return ArtifactDenylist(
        denylist_id=str(root["denylist_id"]),
        denied_values=frozenset(denied),
        erratum_file_sha256=str(receipt_contract["file_sha256"]),
        erratum_report_sha256=str(receipt_contract["receipt_sha256"]),
        raw=MappingProxyType(dict(root)),
    )


def load_combined_artifact_denylist() -> ArtifactDenylist:
    """Load every independently frozen denylist and return their union."""

    test_informed = load_artifact_denylist()
    generator_defect = load_generator_defect_denylist()
    return ArtifactDenylist(
        denylist_id="pii_zh_combined_artifact_policy_v1",
        denied_values=frozenset(
            test_informed.denied_values | generator_defect.denied_values
        ),
        erratum_file_sha256=test_informed.erratum_file_sha256,
        erratum_report_sha256=test_informed.erratum_report_sha256,
        raw=MappingProxyType(
            {
                "component_denylist_ids": [
                    test_informed.denylist_id,
                    generator_defect.denylist_id,
                ]
            }
        ),
    )


def _string_values(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, str):
        result.add(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            result.add(str(key))
            if str(key) in _DECLARATION_ONLY_KEYS:
                continue
            result.update(_string_values(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            result.update(_string_values(child))
    return result


def _semantic_denial(document: Mapping[str, Any], *, purpose: str) -> bool:
    """Reject explicit unsafe state even when an identity was not pre-enumerated."""

    denied_false_keys = _PURPOSE_DENIED_FALSE_KEYS[purpose]

    def visit(value: object) -> bool:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                if key in _DECLARATION_ONLY_KEYS:
                    continue
                if key in _CONTROLLED_BOOLEAN_KEYS and not isinstance(child, bool):
                    return True
                if key in _ALWAYS_DENIED_TRUE_KEYS and child is True:
                    return True
                if key in denied_false_keys and child is False:
                    return True
                if key in {"classification", "storage_class", "status", "artifact_status"} and (
                    isinstance(child, str)
                    and any(marker in child.casefold() for marker in _SEMANTIC_RETIREMENT_MARKERS)
                ):
                    return True
                if visit(child):
                    return True
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return any(visit(child) for child in value)
        return False

    return visit(document)


def assert_document_allowed(
    document: Mapping[str, Any],
    *,
    purpose: str,
    denylist: ArtifactDenylist | None = None,
) -> None:
    if purpose not in _PURPOSES:
        raise ArtifactPolicyError("artifact policy purpose is unsupported")
    policy = load_combined_artifact_denylist() if denylist is None else denylist
    if _string_values(document) & policy.denied_values or _semantic_denial(
        document, purpose=purpose
    ):
        raise ArtifactPolicyError("artifact is denied by artifact lineage policy")


def assert_file_allowed(
    path: Path,
    *,
    purpose: str,
    document: Mapping[str, Any] | None = None,
    denylist: ArtifactDenylist | None = None,
) -> None:
    if purpose not in _PURPOSES:
        raise ArtifactPolicyError("artifact policy purpose is unsupported")
    _regular_file(path, field="candidate artifact")
    policy = load_combined_artifact_denylist() if denylist is None else denylist
    if _sha256_file(path) in policy.denied_values:
        raise ArtifactPolicyError("artifact is denied by artifact lineage policy")
    if document is not None:
        assert_document_allowed(document, purpose=purpose, denylist=policy)


__all__ = [
    "ArtifactDenylist",
    "ArtifactPolicyError",
    "DEFAULT_DENYLIST_FILE_SHA256",
    "DEFAULT_DENYLIST_PATH",
    "DEFAULT_ERRATUM_PATH",
    "DEFAULT_GENERATOR_DEFECT_DENYLIST_FILE_SHA256",
    "DEFAULT_GENERATOR_DEFECT_DENYLIST_PATH",
    "DEFAULT_GENERATOR_DEFECT_DIAGNOSTIC_PATH",
    "DEFAULT_GENERATOR_DEFECT_RECEIPT_PATH",
    "assert_document_allowed",
    "assert_file_allowed",
    "load_artifact_denylist",
    "load_combined_artifact_denylist",
    "load_generator_defect_denylist",
]
