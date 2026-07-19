"""Fail-closed validation for the preregistered comparator registry."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .provenance import canonical_json_hash, sha256_file

CANONICAL_TRACKS: tuple[str, ...] = (
    "model_raw",
    "model_calibrated",
    "rules_only",
    "framework_hybrid",
    "full_system",
)
TRACK_ALIASES: dict[str, str] = {"model_only": "model_raw"}
PROTOCOL_ID = "pii_bench_zh_closed_8_v1"

GENERATION_1_COMPARATOR_IDS: frozenset[str] = frozenset(
    {
        "aiguard_qwen3_0_6b_model_raw",
        "openai_privacy_filter_model_raw",
        "pii_engineer_chinese_ner_model_raw",
        "ernie_chinese_ner_model_raw",
        "presidio_ernie_cn_common_framework_hybrid",
        "cluener2020_roberta_model_raw",
        "presidio_cluener_cn_common_framework_hybrid",
        "cn_common_v5_rules_only",
        "argus_redact_fast_rules_only",
    }
)
GENERATION_1_DEFERRED_IDS: frozenset[str] = frozenset(
    {
        "privacy_filter_tw",
        "gliner2_privacy_filter_pii_multi",
        "datafountain_472",
        "piibench_english",
        "multipriv",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ComparatorRegistryError(ValueError):
    """Raised when registry evidence or freeze policy does not verify."""


@dataclass(frozen=True, slots=True)
class ComparatorRegistryReceipt:
    """Privacy-safe aggregate receipt for one validated registry."""

    registry_sha256: str
    schema_file_sha256: str
    comparator_count: int
    deferred_or_excluded_count: int
    unique_report_count: int
    report_file_sha256: dict[str, str]


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparatorRegistryError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ComparatorRegistryError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComparatorRegistryError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    digest = _text(value, field=field)
    if _SHA256.fullmatch(digest) is None:
        raise ComparatorRegistryError(f"{field} must be a lowercase SHA-256")
    return digest


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComparatorRegistryError(f"{field} must be an integer")
    return value


def _resolve_repository_file(
    repository_root: Path,
    relative_path: object,
    *,
    field: str,
) -> Path:
    relative = Path(_text(relative_path, field=field))
    if relative.is_absolute():
        raise ComparatorRegistryError(f"{field} must be repository-relative")
    root = repository_root.expanduser().resolve(strict=True)
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ComparatorRegistryError(f"{field} escapes or is missing from the repository") from exc
    if not resolved.is_file():
        raise ComparatorRegistryError(f"{field} must identify a file")
    return resolved


def _load_json_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparatorRegistryError(f"{field} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ComparatorRegistryError(f"{field} must contain a JSON object")
    return value


def load_comparator_registry(path: str | Path) -> dict[str, Any]:
    """Load one YAML registry without resolving any report artifacts."""

    registry_path = Path(path).expanduser().resolve(strict=True)
    try:
        value = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ComparatorRegistryError("registry is not readable YAML") from exc
    if not isinstance(value, dict):
        raise ComparatorRegistryError("registry must contain a YAML object")
    return value


def _verify_lightweight_schema(
    registry: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    schema_path = _resolve_repository_file(
        repository_root,
        registry.get("schema_file"),
        field="schema_file",
    )
    expected_schema_hash = _sha256(registry.get("schema_file_sha256"), field="schema_file_sha256")
    if sha256_file(schema_path) != expected_schema_hash:
        raise ComparatorRegistryError("schema file byte hash does not verify")
    schema = _load_json_object(schema_path, field="schema_file")
    required = set(_sequence(schema.get("required"), field="schema.required"))
    allowed = set(_mapping(schema.get("properties"), field="schema.properties"))
    if set(registry) != required or required != allowed:
        raise ComparatorRegistryError("registry root fields do not match the frozen schema")

    definitions = _mapping(schema.get("$defs"), field="schema.$defs")
    comparator_schema = _mapping(definitions.get("comparator"), field="schema.$defs.comparator")
    comparator_required = set(
        _sequence(comparator_schema.get("required"), field="schema comparator required")
    )
    comparator_allowed = set(
        _mapping(comparator_schema.get("properties"), field="schema comparator properties")
    )
    comparators = _mapping(registry.get("comparators"), field="comparators")
    for comparator_id, raw in comparators.items():
        comparator = _mapping(raw, field=f"comparators.{comparator_id}")
        if not comparator_required <= set(comparator) or not set(comparator) <= comparator_allowed:
            raise ComparatorRegistryError(
                f"comparators.{comparator_id} does not match the frozen schema"
            )
    return expected_schema_hash


def _verify_self_hash(registry: Mapping[str, Any]) -> str:
    claimed = _sha256(registry.get("registry_sha256"), field="registry_sha256")
    unsigned = dict(registry)
    unsigned.pop("registry_sha256", None)
    if canonical_json_hash(unsigned) != claimed:
        raise ComparatorRegistryError("registry canonical self-hash does not verify")
    return claimed


def _resolve_selector(report: Mapping[str, Any], selector: str, *, comparator_id: str) -> None:
    current: object = report
    for component in selector.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ComparatorRegistryError(
                f"comparators.{comparator_id}.result_selector does not exist in its report"
            )
        current = current[component]
    if not isinstance(current, Mapping):
        raise ComparatorRegistryError(
            f"comparators.{comparator_id}.result_selector must resolve to an object"
        )


def _verify_report_contract(
    report: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    comparator_id: str,
    dataset_manifest_sha256: str,
) -> None:
    contract = _mapping(report.get("comparison_contract"), field="report.comparison_contract")
    if contract.get("benchmark_protocol") != PROTOCOL_ID:
        raise ComparatorRegistryError(f"comparators.{comparator_id} protocol changed")
    report_track = _text(
        comparator.get("report_track"), field=f"comparators.{comparator_id}.report_track"
    )
    direct_track = contract.get("track")
    track_map = contract.get("tracks")
    if direct_track != report_track and (
        not isinstance(track_map, Mapping) or report_track not in track_map
    ):
        raise ComparatorRegistryError(
            f"comparators.{comparator_id}.report_track is absent from the report"
        )

    manifest_candidates: set[str] = set()
    for field in ("dataset_manifest_sha256",):
        value = report.get(field)
        if isinstance(value, str):
            manifest_candidates.add(value)
    benchmark = report.get("benchmark")
    if isinstance(benchmark, Mapping):
        value = benchmark.get("dataset_manifest_sha256")
        if isinstance(value, str):
            manifest_candidates.add(value)
    subsets = report.get("subsets")
    if isinstance(subsets, Mapping):
        for raw_subset in subsets.values():
            if not isinstance(raw_subset, Mapping):
                continue
            dataset = raw_subset.get("dataset")
            if isinstance(dataset, Mapping):
                value = dataset.get("manifest_sha256")
                if isinstance(value, str):
                    manifest_candidates.add(value)
    if dataset_manifest_sha256 not in manifest_candidates:
        raise ComparatorRegistryError(
            f"comparators.{comparator_id} report is not bound to the frozen dataset manifest"
        )

    privacy = report.get("privacy")
    if isinstance(privacy, Mapping) and any(value is True for value in privacy.values()):
        raise ComparatorRegistryError(f"comparators.{comparator_id} report privacy contract failed")


def _verify_comparator(
    comparator_id: str,
    raw: object,
    *,
    repository_root: Path,
    dataset_manifest_sha256: str,
) -> tuple[str, str]:
    comparator = _mapping(raw, field=f"comparators.{comparator_id}")
    canonical_track = _text(
        comparator.get("canonical_track"),
        field=f"comparators.{comparator_id}.canonical_track",
    )
    if canonical_track not in CANONICAL_TRACKS:
        raise ComparatorRegistryError(f"comparators.{comparator_id} has invalid track")
    if comparator.get("candidate") is not False:
        raise ComparatorRegistryError(f"comparators.{comparator_id} must not be a candidate")
    coverage = _integer(
        comparator.get("closed_8_coverage"),
        field=f"comparators.{comparator_id}.closed_8_coverage",
    )
    if not 1 <= coverage <= 8:
        raise ComparatorRegistryError(f"comparators.{comparator_id} coverage is out of range")
    if comparator.get("protocol_id") != PROTOCOL_ID:
        raise ComparatorRegistryError(f"comparators.{comparator_id} protocol changed")
    _text(comparator.get("repository"), field=f"comparators.{comparator_id}.repository")
    revision = comparator.get("revision")
    if revision is not None:
        _text(revision, field=f"comparators.{comparator_id}.revision")
    _text(
        comparator.get("revision_status"),
        field=f"comparators.{comparator_id}.revision_status",
    )
    version = comparator.get("version")
    if version is not None:
        _text(version, field=f"comparators.{comparator_id}.version")

    license_value = _mapping(
        comparator.get("license"), field=f"comparators.{comparator_id}.license"
    )
    _text(license_value.get("status"), field=f"comparators.{comparator_id}.license.status")
    _sequence(
        license_value.get("declarations"),
        field=f"comparators.{comparator_id}.license.declarations",
    )
    policy = license_value.get("project_redistribution_policy")
    if policy not in {"enabled_by_project", "disabled_by_project", "not_applicable"}:
        raise ComparatorRegistryError(
            f"comparators.{comparator_id}.license.project_redistribution_policy is invalid"
        )

    report_path = _resolve_repository_file(
        repository_root,
        comparator.get("report"),
        field=f"comparators.{comparator_id}.report",
    )
    expected_hash = _sha256(
        comparator.get("report_file_sha256"),
        field=f"comparators.{comparator_id}.report_file_sha256",
    )
    if sha256_file(report_path) != expected_hash:
        raise ComparatorRegistryError(
            f"comparators.{comparator_id} report_file_sha256 does not verify exact bytes"
        )
    report = _load_json_object(report_path, field=f"comparators.{comparator_id}.report")
    selector = _text(
        comparator.get("result_selector"),
        field=f"comparators.{comparator_id}.result_selector",
    )
    _resolve_selector(report, selector, comparator_id=comparator_id)
    _verify_report_contract(
        report,
        comparator,
        comparator_id=comparator_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return str(comparator.get("report")), expected_hash


def validate_comparator_registry(
    registry: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> ComparatorRegistryReceipt:
    """Validate schema, freeze policy, canonical hash, and exact report bytes."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    schema_hash = _verify_lightweight_schema(registry, repository_root=root)
    registry_hash = _verify_self_hash(registry)
    if registry.get("schema_version") != 1 or registry.get("leaderboard_generation") != 1:
        raise ComparatorRegistryError("registry generation identity changed")
    if registry.get("registry_id") != "comparator_registry_v1_generation_1":
        raise ComparatorRegistryError("registry_id changed")
    if tuple(_sequence(registry.get("tracks"), field="tracks")) != CANONICAL_TRACKS:
        raise ComparatorRegistryError("canonical five-track order changed")
    if dict(_mapping(registry.get("track_aliases"), field="track_aliases")) != TRACK_ALIASES:
        raise ComparatorRegistryError("model_only to model_raw alias changed")

    freeze = _mapping(registry.get("freeze"), field="freeze")
    if (
        freeze.get("registry_frozen_before_any_hidden_gold_access") is not True
        or freeze.get("hidden_gold_status_at_freeze") != "not_materialized_never_accessed"
        or freeze.get("hidden_gold_record_count_at_freeze") != 0
        or freeze.get("hidden_gold_access_count_at_freeze") != 0
    ):
        raise ComparatorRegistryError("registry was not frozen before hidden-gold access")
    _text(freeze.get("frozen_at"), field="freeze.frozen_at")

    candidate_policy = _mapping(registry.get("candidate_policy"), field="candidate_policy")
    if candidate_policy.get("project_candidates_are_comparators") is not False:
        raise ComparatorRegistryError("project candidates must not be comparators")
    amendment = _mapping(registry.get("amendment_policy"), field="amendment_policy")
    if (
        amendment.get("generation_1_comparator_set_closed") is not True
        or amendment.get("new_comparator_after_freeze") != "next_leaderboard_generation_only"
        or amendment.get("retroactive_generation_1_admission") != "prohibited"
    ):
        raise ComparatorRegistryError("generation amendment policy changed")

    benchmark = _mapping(registry.get("benchmark_contract"), field="benchmark_contract")
    if benchmark.get("protocol_id") != PROTOCOL_ID:
        raise ComparatorRegistryError("benchmark protocol changed")
    dataset_manifest_hash = _sha256(
        benchmark.get("dataset_manifest_sha256"),
        field="benchmark_contract.dataset_manifest_sha256",
    )
    if benchmark.get("threshold_tuning_on_benchmark") is not False:
        raise ComparatorRegistryError("benchmark threshold tuning must remain prohibited")

    comparators = _mapping(registry.get("comparators"), field="comparators")
    if set(comparators) != GENERATION_1_COMPARATOR_IDS:
        raise ComparatorRegistryError("generation-1 comparator set changed")
    report_hashes: dict[str, str] = {}
    for comparator_id, raw in comparators.items():
        report_path, report_hash = _verify_comparator(
            str(comparator_id),
            raw,
            repository_root=root,
            dataset_manifest_sha256=dataset_manifest_hash,
        )
        previous = report_hashes.get(report_path)
        if previous is not None and previous != report_hash:
            raise ComparatorRegistryError("shared report path has inconsistent byte hashes")
        report_hashes[report_path] = report_hash

    deferred = _mapping(registry.get("deferred_or_excluded"), field="deferred_or_excluded")
    if set(deferred) != GENERATION_1_DEFERRED_IDS:
        raise ComparatorRegistryError("deferred/excluded resource set changed")
    for resource_id, raw in deferred.items():
        resource = _mapping(raw, field=f"deferred_or_excluded.{resource_id}")
        if resource.get("disposition") not in {
            "deferred_next_generation",
            "excluded_not_same_protocol",
        }:
            raise ComparatorRegistryError(f"deferred_or_excluded.{resource_id} is invalid")
        _sequence(
            resource.get("reason_codes"),
            field=f"deferred_or_excluded.{resource_id}.reason_codes",
        )
        _text(
            resource.get("admission_requirement"),
            field=f"deferred_or_excluded.{resource_id}.admission_requirement",
        )

    return ComparatorRegistryReceipt(
        registry_sha256=registry_hash,
        schema_file_sha256=schema_hash,
        comparator_count=len(comparators),
        deferred_or_excluded_count=len(deferred),
        unique_report_count=len(report_hashes),
        report_file_sha256=dict(sorted(report_hashes.items())),
    )


__all__ = [
    "CANONICAL_TRACKS",
    "GENERATION_1_COMPARATOR_IDS",
    "GENERATION_1_DEFERRED_IDS",
    "PROTOCOL_ID",
    "TRACK_ALIASES",
    "ComparatorRegistryError",
    "ComparatorRegistryReceipt",
    "load_comparator_registry",
    "validate_comparator_registry",
]
