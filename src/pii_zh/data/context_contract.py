"""Fail-closed loader for the frozen human-context dataset scale contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

DEFAULT_CONTEXT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "configs/data/real_context_dataset_v2.yaml"
)
SPLITS = ("train", "validation", "public_test", "hidden_leaderboard")
PUBLIC_SPLITS = SPLITS[:-1]


class ContextContractError(ValueError):
    """Raised when the release-scale contract is missing or weakened."""


@dataclass(frozen=True, slots=True)
class ContextDatasetContract:
    """Validated release and pilot counts without any record-level data."""

    contract_id: str
    contract_version: str
    status: str
    dataset_id: str
    release_total_documents: int
    release_split_documents: Mapping[str, int]
    private_real_audit_minimum_documents: int
    pii_free_document_ratio: float
    hard_negative_document_ratio_minimum: float
    pilot_campaign_id: str
    pilot_total_task_cards: int
    pilot_split_task_cards: Mapping[str, int]

    @property
    def release_split_ratios(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                split: count / self.release_total_documents
                for split, count in self.release_split_documents.items()
            }
        )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextContractError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ContextContractError(f"{field} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], *, field: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise ContextContractError(f"{field} keys do not match the frozen contract")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextContractError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextContractError(f"{field} must be a non-negative integer")
    return value


def _ratio(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContextContractError(f"{field} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ContextContractError(f"{field} must be between zero and one")
    return result


def _false(value: object, *, field: str) -> None:
    if value is not False:
        raise ContextContractError(f"{field} must remain false")


def _true(value: object, *, field: str) -> None:
    if value is not True:
        raise ContextContractError(f"{field} must remain true")


def _split_counts(value: object, *, field: str) -> dict[str, int]:
    raw = _mapping(value, field=field)
    _exact_keys(raw, field=field, expected=set(SPLITS))
    return {split: _integer(raw[split], field=f"{field}.{split}") for split in SPLITS}


def validate_context_dataset_contract(document: object) -> ContextDatasetContract:
    """Validate exact v2 release/pilot scale and non-weakening isolation rules."""

    root = _mapping(document, field="contract")
    _exact_keys(
        root,
        field="contract",
        expected={
            "schema_version",
            "contract_id",
            "contract_version",
            "status",
            "authority_document",
            "release_dataset",
            "pilot_campaign",
            "evaluation_isolation",
        },
    )
    if root["schema_version"] != 1:
        raise ContextContractError("unsupported context contract schema version")
    if root["authority_document"] != "docs/real_context_dataset_v2.md":
        raise ContextContractError("authority document does not match the frozen contract")

    release = _mapping(root["release_dataset"], field="release_dataset")
    _exact_keys(
        release,
        field="release_dataset",
        expected={
            "dataset_id",
            "total_documents",
            "split_documents",
            "public_splits",
            "hidden_splits",
            "private_real_audit_minimum_documents",
            "public_contains_real_personal_data",
            "public_identifier_surface",
            "synthetic_tail_reported_separately",
            "pii_free_document_ratio",
            "hard_negative_document_ratio_minimum",
        },
    )
    release_total = _integer(release["total_documents"], field="release_dataset.total_documents")
    release_splits = _split_counts(
        release["split_documents"], field="release_dataset.split_documents"
    )
    if release_total != 20_000 or release_splits != {
        "train": 14_000,
        "validation": 2_000,
        "public_test": 2_000,
        "hidden_leaderboard": 2_000,
    }:
        raise ContextContractError("release counts do not match the frozen 20K contract")
    if sum(release_splits.values()) != release_total:
        raise ContextContractError("release split counts do not sum to the declared total")
    if release["public_splits"] != list(PUBLIC_SPLITS):
        raise ContextContractError("public split list does not match the frozen contract")
    if release["hidden_splits"] != ["hidden_leaderboard"]:
        raise ContextContractError("hidden split list does not match the frozen contract")
    private_minimum = _integer(
        release["private_real_audit_minimum_documents"],
        field="release_dataset.private_real_audit_minimum_documents",
    )
    if private_minimum < 2_000:
        raise ContextContractError("private real audit minimum cannot be weakened")
    _false(
        release["public_contains_real_personal_data"],
        field="release_dataset.public_contains_real_personal_data",
    )
    if release["public_identifier_surface"] != "surrogate_only":
        raise ContextContractError("public identifier surface must remain surrogate-only")
    _true(
        release["synthetic_tail_reported_separately"],
        field="release_dataset.synthetic_tail_reported_separately",
    )
    pii_free_ratio = _ratio(
        release["pii_free_document_ratio"], field="release_dataset.pii_free_document_ratio"
    )
    hard_negative_ratio = _ratio(
        release["hard_negative_document_ratio_minimum"],
        field="release_dataset.hard_negative_document_ratio_minimum",
    )
    if pii_free_ratio != 0.40 or hard_negative_ratio < 0.25:
        raise ContextContractError("release negative-document composition was weakened")

    pilot = _mapping(root["pilot_campaign"], field="pilot_campaign")
    _exact_keys(
        pilot,
        field="pilot_campaign",
        expected={
            "campaign_id",
            "total_task_cards",
            "split_task_cards",
            "release_dataset_equivalent",
            "human_record_counts_start_at_zero",
        },
    )
    pilot_total = _integer(pilot["total_task_cards"], field="pilot_campaign.total_task_cards")
    pilot_splits = _split_counts(
        pilot["split_task_cards"], field="pilot_campaign.split_task_cards"
    )
    if pilot_total != 1_200 or pilot_splits != {
        "train": 720,
        "validation": 180,
        "public_test": 150,
        "hidden_leaderboard": 150,
    }:
        raise ContextContractError("pilot counts do not match the frozen campaign")
    if sum(pilot_splits.values()) != pilot_total:
        raise ContextContractError("pilot split counts do not sum to the declared total")
    _false(
        pilot["release_dataset_equivalent"], field="pilot_campaign.release_dataset_equivalent"
    )
    _true(
        pilot["human_record_counts_start_at_zero"],
        field="pilot_campaign.human_record_counts_start_at_zero",
    )

    isolation = _mapping(root["evaluation_isolation"], field="evaluation_isolation")
    expected_isolation = {
        "public_test_allowed_for_model_selection": False,
        "hidden_allowed_for_model_selection": False,
        "hidden_accessible_to_training_team": False,
        "private_real_audit_enters_public_weights": False,
        "seen_pii_bench_is_confirmatory": False,
    }
    _exact_keys(
        isolation, field="evaluation_isolation", expected=set(expected_isolation)
    )
    if dict(isolation) != expected_isolation:
        raise ContextContractError("evaluation isolation cannot be weakened")

    return ContextDatasetContract(
        contract_id=_string(root["contract_id"], field="contract_id"),
        contract_version=_string(root["contract_version"], field="contract_version"),
        status=_string(root["status"], field="status"),
        dataset_id=_string(release["dataset_id"], field="release_dataset.dataset_id"),
        release_total_documents=release_total,
        release_split_documents=MappingProxyType(release_splits),
        private_real_audit_minimum_documents=private_minimum,
        pii_free_document_ratio=pii_free_ratio,
        hard_negative_document_ratio_minimum=hard_negative_ratio,
        pilot_campaign_id=_string(pilot["campaign_id"], field="pilot_campaign.campaign_id"),
        pilot_total_task_cards=pilot_total,
        pilot_split_task_cards=MappingProxyType(pilot_splits),
    )


def load_context_dataset_contract(
    path: str | Path | None = None,
) -> ContextDatasetContract:
    """Load the repository contract without exposing any record-level input."""

    resolved = DEFAULT_CONTEXT_CONTRACT_PATH if path is None else Path(path)
    try:
        document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContextContractError("unable to load the context dataset contract") from exc
    return validate_context_dataset_contract(document)


__all__ = [
    "DEFAULT_CONTEXT_CONTRACT_PATH",
    "ContextContractError",
    "ContextDatasetContract",
    "load_context_dataset_contract",
    "validate_context_dataset_contract",
]
