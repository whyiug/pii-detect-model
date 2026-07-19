"""Zero-record human-context campaign planning and starter-pack validation.

The module creates task and reviewer allocation metadata only.  It never
creates, accepts, labels, or admits a human-authored record.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from pii_zh.taxonomy import load_taxonomy

from .human_contributions import AGREEMENT_VERSION
from .human_contributions import SPLITS as INTAKE_SPLITS

CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_ID = "pii_zh_human_context_campaign_v1"
CAMPAIGN_VERSION = "1.0.0"
DEFAULT_CAMPAIGN_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/data/human_context_campaign_v1.yaml"
)
DEFAULT_CAMPAIGN_DOC_PATH = (
    Path(__file__).resolve().parents[3] / "docs/human_context_campaign_starter_v1.md"
)
_DOMAINS = (
    "customer_service",
    "finance",
    "medical",
    "education",
    "government",
    "developer_logs",
    "open_chat",
)
_MODES = (
    "positive_slot_context",
    "pii_free_hard_negative",
    "pii_free_counterfactual",
    "pii_free_plain",
)
_PII_FREE_MODES = frozenset(_MODES[1:])
_INSTRUCTION_CODES = (
    "write_from_blank",
    "no_real_personal_data",
    "no_production_or_nonpublic_copy",
    "no_generative_ai_or_machine_translation",
    "use_only_assigned_slot_tokens",
)
_GROUPING_FIELDS = frozenset(
    {
        "contributor_group_slot",
        "document_group_slot",
        "conversation_group_slot",
        "parent_group_slot",
        "near_duplicate_group_status",
    }
)
_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_version",
        "task_id",
        "contributor_slot_id",
        "split",
        "domain",
        "mode",
        "target_core_label",
        "negative_category",
        "slot_tokens",
        "instruction_codes",
        "grouping_plan",
        "answer_status",
    }
)
_BLIND_ASSIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "annotator_a_slot",
        "annotator_b_slot",
        "adjudicator_slot",
        "privacy_reviewer_slot",
        "blind_to_target_label",
        "blind_to_contributor_identity",
        "annotator_a_sees_b",
        "annotator_b_sees_a",
        "annotation_payload_status",
        "adjudication_payload_status",
    }
)
_FORBIDDEN_ANSWER_KEYS = frozenset(
    {
        "answer",
        "answer_text",
        "submission",
        "submission_text",
        "context_text",
        "human_text",
        "annotation",
        "annotations",
        "spans",
        "entities",
        "gold",
        "adjudicated_answer",
    }
)
_ZERO_STATUS = {
    "enrolled_contributors": 0,
    "collected_count": 0,
    "admitted_count": 0,
    "public_weight_training_count": 0,
}
_TAXONOMY = load_taxonomy()
_CORE_LABELS = tuple(entity.name for entity in _TAXONOMY.label_sets["core"])
_AUXILIARY_LABELS = frozenset(_TAXONOMY.auxiliary_label_names)


class HumanCampaignError(ValueError):
    """Raised with a fixed planning error that cannot reflect human content."""


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    campaign_id: str
    campaign_version: str
    seed: int
    governance: Mapping[str, Any]
    status: Mapping[str, int]
    task_quota: Mapping[str, int]
    split_quotas: Mapping[str, Mapping[str, int]]
    domain_quotas: Mapping[str, Mapping[str, int]]
    positive_label_quotas: Mapping[str, Mapping[str, int]]
    hard_negative_categories: tuple[str, ...]
    contributor_slots: tuple[Mapping[str, Any], ...]
    reviewer_slots: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class CampaignPack:
    task_cards: tuple[Mapping[str, Any], ...]
    blind_assignments: tuple[Mapping[str, Any], ...]
    consent_template: Mapping[str, Any]
    signoff_template: Mapping[str, Any]
    aggregate_status: Mapping[str, Any]
    plan_validation: Mapping[str, Any]


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise HumanCampaignError(f"{field} must be a string-keyed mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HumanCampaignError(f"{field} must be a sequence")
    return value


def _exact_fields(value: Mapping[str, Any], *, expected: frozenset[str], field: str) -> None:
    if set(value) != expected:
        raise HumanCampaignError(f"{field} fields differ from the campaign schema")


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HumanCampaignError(f"{field} must be an integer of at least {minimum}")
    return value


def _safe_token(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
    ):
        raise HumanCampaignError(f"{field} must be a lowercase safe token")
    return value


def load_campaign_config(
    path: Path = DEFAULT_CAMPAIGN_CONFIG_PATH,
) -> CampaignConfig:
    if path.is_symlink() or not path.is_file():
        raise HumanCampaignError("campaign config must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise HumanCampaignError("campaign config is invalid YAML") from exc
    root = _mapping(document, field="campaign")
    _exact_fields(
        root,
        expected=frozenset(
            {
                "schema_version",
                "campaign_id",
                "campaign_version",
                "starter_pack_only",
                "record_generation_allowed",
                "seed",
                "governance",
                "status",
                "task_quota",
                "split_quotas",
                "domain_quotas",
                "positive_label_quotas",
                "hard_negative_categories",
                "contributor_slots",
                "reviewer_slots",
            }
        ),
        field="campaign",
    )
    if (
        root["schema_version"] != CAMPAIGN_SCHEMA_VERSION
        or root["campaign_id"] != CAMPAIGN_ID
        or root["campaign_version"] != CAMPAIGN_VERSION
        or root["starter_pack_only"] is not True
        or root["record_generation_allowed"] is not False
    ):
        raise HumanCampaignError("campaign identity or zero-record safeguards changed")
    seed = _integer(root["seed"], field="seed")
    governance = _mapping(root["governance"], field="governance")
    expected_governance = {
        "contribution_license": "CC-BY-4.0",
        "real_personal_data_allowed": False,
        "production_content_allowed": False,
        "nonpublic_source_copy_allowed": False,
        "generative_ai_allowed": False,
        "machine_translation_allowed": False,
        "withdrawal_notice_required": True,
        "independent_privacy_review_required": True,
        "legal_signoff_required_before_collection": True,
        "legal_signoff_status": "pending",
    }
    if dict(governance) != expected_governance:
        raise HumanCampaignError("campaign governance safeguards changed")
    status = _mapping(root["status"], field="status")
    _exact_fields(
        status,
        expected=frozenset({"planned_contributor_slots", *_ZERO_STATUS}),
        field="status",
    )
    if any(status[key] != value for key, value in _ZERO_STATUS.items()):
        raise HumanCampaignError("campaign human-record counters must remain zero")
    planned_contributors = _integer(
        status["planned_contributor_slots"], field="planned_contributor_slots", minimum=20
    )
    task_quota = _mapping(root["task_quota"], field="task_quota")
    _exact_fields(
        task_quota,
        expected=frozenset({"total", *_MODES}),
        field="task_quota",
    )
    task_counts = {
        key: _integer(value, field=f"task_quota.{key}") for key, value in task_quota.items()
    }
    if task_counts["total"] != sum(task_counts[mode] for mode in _MODES):
        raise HumanCampaignError("global task mode quotas do not sum to total")
    split_quotas = _parse_split_matrix(
        root["split_quotas"],
        field="split_quotas",
        columns=("total", *_MODES),
    )
    if tuple(split_quotas) != INTAKE_SPLITS:
        raise HumanCampaignError("campaign split set differs from intake SPLITS")
    for _split, quotas in split_quotas.items():
        if quotas["total"] != sum(quotas[mode] for mode in _MODES):
            raise HumanCampaignError("split task mode quotas do not sum to total")
        ratio = sum(quotas[mode] for mode in _PII_FREE_MODES) / quotas["total"]
        if not 0.30 <= ratio <= 0.40:
            raise HumanCampaignError("split PII-free ratio escaped [0.30, 0.40]")
    for key in ("total", *_MODES):
        if sum(split_quotas[split][key] for split in INTAKE_SPLITS) != task_counts[key]:
            raise HumanCampaignError("split task quotas differ from global quotas")
    domain_quotas = _parse_split_matrix(
        root["domain_quotas"], field="domain_quotas", columns=_DOMAINS
    )
    if tuple(domain_quotas) != INTAKE_SPLITS:
        raise HumanCampaignError("domain quota split set differs from intake SPLITS")
    for split in INTAKE_SPLITS:
        if sum(domain_quotas[split].values()) != split_quotas[split]["total"]:
            raise HumanCampaignError("domain quotas do not sum to split total")
    positive_label_quotas = _parse_split_matrix(
        root["positive_label_quotas"],
        field="positive_label_quotas",
        columns=_CORE_LABELS,
    )
    if tuple(positive_label_quotas) != INTAKE_SPLITS:
        raise HumanCampaignError("label quota split set differs from intake SPLITS")
    for split in INTAKE_SPLITS:
        if sum(positive_label_quotas[split].values()) != split_quotas[split][
            "positive_slot_context"
        ] or any(value < 1 for value in positive_label_quotas[split].values()):
            raise HumanCampaignError("core label quotas are incomplete")
    categories_raw = _sequence(root["hard_negative_categories"], field="hard_negative_categories")
    categories = tuple(categories_raw)
    if (
        len(categories) != 9
        or len(set(categories)) != len(categories)
        or any(not isinstance(item, str) or item not in _AUXILIARY_LABELS for item in categories)
    ):
        raise HumanCampaignError("hard-negative categories are invalid")
    contributors_raw = _sequence(root["contributor_slots"], field="contributor_slots")
    contributors: list[Mapping[str, Any]] = []
    contributor_ids: set[str] = set()
    contributor_quota_by_split: Counter[str] = Counter()
    for index, item in enumerate(contributors_raw):
        contributor = _mapping(item, field=f"contributor_slots[{index}]")
        _exact_fields(
            contributor,
            expected=frozenset({"id", "split", "task_quota", "primary_domain"}),
            field=f"contributor_slots[{index}]",
        )
        contributor_id = _safe_token(contributor["id"], field="contributor id")
        split = contributor["split"]
        domain = contributor["primary_domain"]
        quota = _integer(contributor["task_quota"], field="contributor task_quota", minimum=1)
        if (
            contributor_id in contributor_ids
            or split not in INTAKE_SPLITS
            or domain not in _DOMAINS
        ):
            raise HumanCampaignError("contributor slot assignment is invalid")
        contributor_ids.add(contributor_id)
        contributor_quota_by_split[str(split)] += quota
        contributors.append(MappingProxyType(dict(contributor)))
    if len(contributors) != planned_contributors or len(contributors) < 20:
        raise HumanCampaignError("planned contributor slot count is inconsistent")
    if any(
        contributor_quota_by_split[split] != split_quotas[split]["total"] for split in INTAKE_SPLITS
    ):
        raise HumanCampaignError("contributor quotas do not sum to split totals")
    reviewers_raw = _sequence(root["reviewer_slots"], field="reviewer_slots")
    reviewers = tuple(_safe_token(item, field="reviewer slot") for item in reviewers_raw)
    if (
        len(reviewers) < 4
        or len(set(reviewers)) != len(reviewers)
        or set(reviewers) & contributor_ids
    ):
        raise HumanCampaignError("reviewer slots must contain at least four unique IDs")
    return CampaignConfig(
        campaign_id=CAMPAIGN_ID,
        campaign_version=CAMPAIGN_VERSION,
        seed=seed,
        governance=MappingProxyType(dict(governance)),
        status=MappingProxyType(
            {key: _integer(value, field=f"status.{key}") for key, value in status.items()}
        ),
        task_quota=MappingProxyType(task_counts),
        split_quotas=MappingProxyType(split_quotas),
        domain_quotas=MappingProxyType(domain_quotas),
        positive_label_quotas=MappingProxyType(positive_label_quotas),
        hard_negative_categories=categories,
        contributor_slots=tuple(contributors),
        reviewer_slots=reviewers,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _parse_split_matrix(
    value: object, *, field: str, columns: Sequence[str]
) -> dict[str, Mapping[str, int]]:
    matrix = _mapping(value, field=field)
    if tuple(matrix) != INTAKE_SPLITS:
        raise HumanCampaignError(f"{field} must use the ordered intake split set")
    result = {}
    for split in INTAKE_SPLITS:
        row = _mapping(matrix[split], field=f"{field}.{split}")
        _exact_fields(row, expected=frozenset(columns), field=f"{field}.{split}")
        result[split] = MappingProxyType(
            {column: _integer(row[column], field=f"{field}.{split}.{column}") for column in columns}
        )
    return result


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _expanded(mapping: Mapping[str, int]) -> list[str]:
    return [name for name, count in mapping.items() for _ in range(count)]


def _hard_negative_allocation(config: CampaignConfig) -> Mapping[str, list[str]]:
    categories = config.hard_negative_categories
    total = config.task_quota["pii_free_hard_negative"]
    base, remainder = divmod(total, len(categories))
    remaining = {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }
    allocation: dict[str, list[str]] = {split: [] for split in INTAKE_SPLITS}
    for split_index, split in enumerate(INTAKE_SPLITS):
        target = config.split_quotas[split]["pii_free_hard_negative"]
        while len(allocation[split]) < target:
            available = [category for category in categories if remaining[category] > 0]
            if not available:
                raise HumanCampaignError("hard-negative category capacity was exhausted")
            category = max(
                available,
                key=lambda item: (
                    remaining[item],
                    -((categories.index(item) - split_index) % len(categories)),
                ),
            )
            allocation[split].append(category)
            remaining[category] -= 1
    for split in INTAKE_SPLITS:
        if len(allocation[split]) != config.split_quotas[split]["pii_free_hard_negative"]:
            raise HumanCampaignError("hard-negative split allocation is inconsistent")
    expected = Counter(
        {
            category: base + (1 if index < remainder else 0)
            for index, category in enumerate(categories)
        }
    )
    if Counter(item for values in allocation.values() for item in values) != expected:
        raise HumanCampaignError("hard-negative global category allocation is inconsistent")
    return allocation


def _build_task_cards(config: CampaignConfig) -> tuple[Mapping[str, Any], ...]:
    hard_negatives = _hard_negative_allocation(config)
    cards: list[Mapping[str, Any]] = []
    task_number = 1
    for split in INTAKE_SPLITS:
        rng = _stable_rng(config.seed, f"cards:{split}")
        contributor_ids = [
            str(item["id"])
            for item in config.contributor_slots
            if item["split"] == split
            for _ in range(int(item["task_quota"]))
        ]
        domains = _expanded(config.domain_quotas[split])
        modes: list[tuple[str, str | None, str | None]] = []
        modes.extend(
            ("positive_slot_context", label, None)
            for label in _expanded(config.positive_label_quotas[split])
        )
        modes.extend(
            ("pii_free_hard_negative", None, category) for category in hard_negatives[split]
        )
        counter_count = config.split_quotas[split]["pii_free_counterfactual"]
        modes.extend(
            (
                "pii_free_counterfactual",
                _CORE_LABELS[index % len(_CORE_LABELS)],
                config.hard_negative_categories[index % len(config.hard_negative_categories)],
            )
            for index in range(counter_count)
        )
        modes.extend(
            ("pii_free_plain", None, None)
            for _ in range(config.split_quotas[split]["pii_free_plain"])
        )
        for values in (contributor_ids, domains, modes):
            rng.shuffle(values)
        contributor_seen: Counter[str] = Counter()
        for contributor_id, domain, (mode, target_label, negative_category) in zip(
            contributor_ids, domains, modes, strict=True
        ):
            local_index = contributor_seen[contributor_id]
            contributor_seen[contributor_id] += 1
            task_id = f"campaign_task_{task_number:04d}"
            task_number += 1
            session_number = local_index // 5 + 1
            slot_tokens = []
            if mode == "positive_slot_context":
                slot_tokens = ["[[PII_SLOT:slot_01]]"]
            elif mode in {"pii_free_hard_negative", "pii_free_counterfactual"}:
                slot_tokens = [f"[[NEG_SLOT:{negative_category}:slot_01]]"]
            cards.append(
                MappingProxyType(
                    {
                        "schema_version": "pii-zh-human-campaign-task-v1",
                        "campaign_id": config.campaign_id,
                        "campaign_version": config.campaign_version,
                        "task_id": task_id,
                        "contributor_slot_id": contributor_id,
                        "split": split,
                        "domain": domain,
                        "mode": mode,
                        "target_core_label": target_label,
                        "negative_category": negative_category,
                        "slot_tokens": slot_tokens,
                        "instruction_codes": list(_INSTRUCTION_CODES),
                        "grouping_plan": {
                            "contributor_group_slot": contributor_id,
                            "document_group_slot": f"document_group_{task_id}",
                            "conversation_group_slot": (
                                f"session_group_{contributor_id}_{session_number:02d}"
                            ),
                            "parent_group_slot": None,
                            "near_duplicate_group_status": "assign_after_collection_before_freeze",
                        },
                        "answer_status": "not_collected",
                    }
                )
            )
    return tuple(cards)


def _build_blind_assignments(
    config: CampaignConfig, task_cards: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    reviewers = config.reviewer_slots
    assignments = []
    for index, card in enumerate(task_cards):
        roles = (
            reviewers[index % len(reviewers)],
            reviewers[(index + 1) % len(reviewers)],
            reviewers[(index + 4) % len(reviewers)],
            reviewers[(index + 7) % len(reviewers)],
        )
        if len(set(roles)) != 4:
            raise HumanCampaignError("reviewer role rotation is not independent")
        assignments.append(
            MappingProxyType(
                {
                    "schema_version": "pii-zh-human-campaign-blind-assignment-v1",
                    "task_id": card["task_id"],
                    "annotator_a_slot": roles[0],
                    "annotator_b_slot": roles[1],
                    "adjudicator_slot": roles[2],
                    "privacy_reviewer_slot": roles[3],
                    "blind_to_target_label": True,
                    "blind_to_contributor_identity": True,
                    "annotator_a_sees_b": False,
                    "annotator_b_sees_a": False,
                    "annotation_payload_status": "not_created",
                    "adjudication_payload_status": "not_created",
                }
            )
        )
    return tuple(assignments)


def _consent_template(config: CampaignConfig) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema_version": "pii-zh-human-campaign-consent-template-v1",
            "campaign_id": config.campaign_id,
            "agreement_version": AGREEMENT_VERSION,
            "license_id": "CC-BY-4.0",
            "template_only": True,
            "prechecked": False,
            "contributor_slot_id": None,
            "consent_receipt_id": None,
            "signed_at": None,
            "checklist": {
                "original_author_and_rights_to_license": False,
                "cc_by_4_public_redistribution": False,
                "modification_and_model_training": False,
                "derived_weight_distribution": False,
                "no_generative_ai": False,
                "no_machine_translation": False,
                "no_real_or_third_party_pii": False,
                "no_production_or_nonpublic_content": False,
                "assigned_slots_only": False,
                "contributor_privacy_self_review": False,
                "withdrawal_limits_acknowledged": False,
            },
        }
    )


def _signoff_template(config: CampaignConfig) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema_version": "pii-zh-human-campaign-signoff-template-v1",
            "campaign_id": config.campaign_id,
            "template_only": True,
            "collection_open": False,
            "legal_review": {"status": "pending", "reviewer": None, "signed_at": None},
            "privacy_review": {"status": "pending", "reviewer": None, "signed_at": None},
            "security_review": {"status": "pending", "reviewer": None, "signed_at": None},
            "withdrawal_channel_verified": False,
            "minor_participation_policy_verified": False,
            "public_weight_training_allowed": False,
        }
    )


def build_campaign_pack(config: CampaignConfig) -> CampaignPack:
    cards = _build_task_cards(config)
    assignments = _build_blind_assignments(config, cards)
    consent = _consent_template(config)
    signoff = _signoff_template(config)
    status = MappingProxyType(
        {
            "schema_version": "pii-zh-human-campaign-status-v1",
            "campaign_id": config.campaign_id,
            "campaign_version": config.campaign_version,
            "planned_contributor_slots": config.status["planned_contributor_slots"],
            "task_card_count": len(cards),
            **_ZERO_STATUS,
            "human_record_count": 0,
            "legal_signoff_status": "pending",
            "privacy_signoff_status": "pending",
            "collection_open": False,
            "public_release_allowed": False,
            "public_weight_training_allowed": False,
        }
    )
    provisional = CampaignPack(
        task_cards=cards,
        blind_assignments=assignments,
        consent_template=consent,
        signoff_template=signoff,
        aggregate_status=status,
        plan_validation=MappingProxyType({}),
    )
    validation = validate_campaign_pack(provisional, config=config)
    return CampaignPack(
        task_cards=cards,
        blind_assignments=assignments,
        consent_template=consent,
        signoff_template=signoff,
        aggregate_status=status,
        plan_validation=validation,
    )


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def validate_campaign_pack(pack: CampaignPack, *, config: CampaignConfig) -> Mapping[str, Any]:
    cards = list(pack.task_cards)
    if len(cards) != config.task_quota["total"]:
        raise HumanCampaignError("task card count differs from frozen quota")
    if any(set(card) != _TASK_FIELDS for card in cards):
        raise HumanCampaignError("task card schema changed")
    if any(_walk_keys(card) & _FORBIDDEN_ANSWER_KEYS for card in cards):
        raise HumanCampaignError("task card contains a forbidden answer field")
    task_ids = [str(card["task_id"]) for card in cards]
    if len(set(task_ids)) != len(task_ids):
        raise HumanCampaignError("task IDs are not unique")
    split_counts = Counter(str(card["split"]) for card in cards)
    if tuple(config.split_quotas) != INTAKE_SPLITS or set(split_counts) != set(INTAKE_SPLITS):
        raise HumanCampaignError("campaign split set differs from intake SPLITS")
    mode_counts = Counter(str(card["mode"]) for card in cards)
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_modes: dict[str, Counter[str]] = defaultdict(Counter)
    contributor_counts: Counter[str] = Counter()
    contributor_splits: dict[str, set[str]] = defaultdict(set)
    document_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[str, set[str]] = defaultdict(set)
    hard_categories: Counter[str] = Counter()
    for card in cards:
        split = str(card["split"])
        mode = str(card["mode"])
        contributor = str(card["contributor_slot_id"])
        domain = str(card["domain"])
        if split not in INTAKE_SPLITS or mode not in _MODES or domain not in _DOMAINS:
            raise HumanCampaignError("task card uses an unknown split, mode, or domain")
        if (
            card["schema_version"] != "pii-zh-human-campaign-task-v1"
            or card["campaign_id"] != config.campaign_id
            or card["campaign_version"] != config.campaign_version
            or card["instruction_codes"] != list(_INSTRUCTION_CODES)
        ):
            raise HumanCampaignError("task card identity or instruction contract changed")
        contributor_counts[contributor] += 1
        contributor_splits[contributor].add(split)
        domain_counts[split][domain] += 1
        split_modes[split][mode] += 1
        grouping = _mapping(card["grouping_plan"], field="task grouping plan")
        _exact_fields(grouping, expected=_GROUPING_FIELDS, field="task grouping plan")
        if (
            grouping["contributor_group_slot"] != contributor
            or grouping["parent_group_slot"] is not None
            or grouping["near_duplicate_group_status"] != "assign_after_collection_before_freeze"
        ):
            raise HumanCampaignError("task grouping plan changed")
        document = _safe_token(grouping["document_group_slot"], field="document group slot")
        session = _safe_token(grouping["conversation_group_slot"], field="conversation group slot")
        document_splits[document].add(split)
        session_splits[session].add(split)
        target = card["target_core_label"]
        negative = card["negative_category"]
        tokens = card["slot_tokens"]
        if card["answer_status"] != "not_collected" or not isinstance(tokens, list):
            raise HumanCampaignError("task card answer state or slot list is invalid")
        if mode == "positive_slot_context":
            if (
                target not in _CORE_LABELS
                or negative is not None
                or tokens != ["[[PII_SLOT:slot_01]]"]
            ):
                raise HumanCampaignError("positive task slot contract failed")
            label_counts[split][str(target)] += 1
        elif mode == "pii_free_plain":
            if target is not None or negative is not None or tokens:
                raise HumanCampaignError("plain PII-free task contains a target or slot")
        else:
            if negative not in config.hard_negative_categories or tokens != [
                f"[[NEG_SLOT:{negative}:slot_01]]"
            ]:
                raise HumanCampaignError("negative task slot contract failed")
            if mode == "pii_free_hard_negative":
                if target is not None:
                    raise HumanCampaignError("hard-negative task unexpectedly has a core target")
                hard_categories[str(negative)] += 1
            elif target not in _CORE_LABELS:
                raise HumanCampaignError("counterfactual task lacks a core target")
    if (
        any(len(splits) != 1 for splits in contributor_splits.values())
        or any(len(splits) != 1 for splits in document_splits.values())
        or any(len(splits) != 1 for splits in session_splits.values())
    ):
        raise HumanCampaignError("contributor, document, or session group crosses splits")
    expected_contributors = {
        str(item["id"]): int(item["task_quota"]) for item in config.contributor_slots
    }
    if contributor_counts != Counter(expected_contributors):
        raise HumanCampaignError("contributor task counts differ from frozen quotas")
    for split in INTAKE_SPLITS:
        if split_counts[split] != config.split_quotas[split]["total"]:
            raise HumanCampaignError("split task count differs from quota")
        if split_modes[split] != Counter(
            {mode: config.split_quotas[split][mode] for mode in _MODES}
        ):
            raise HumanCampaignError("split mode counts differ from quota")
        if domain_counts[split] != Counter(config.domain_quotas[split]):
            raise HumanCampaignError("split domain counts differ from quota")
        if label_counts[split] != Counter(config.positive_label_quotas[split]):
            raise HumanCampaignError("split core-label counts differ from quota")
        pii_free = sum(split_modes[split][mode] for mode in _PII_FREE_MODES)
        if not 0.30 <= pii_free / split_counts[split] <= 0.40:
            raise HumanCampaignError("split PII-free ratio escaped [0.30, 0.40]")
    if mode_counts != Counter({mode: config.task_quota[mode] for mode in _MODES}):
        raise HumanCampaignError("global mode counts differ from quota")
    hard_total = config.task_quota["pii_free_hard_negative"]
    hard_base, hard_remainder = divmod(hard_total, len(config.hard_negative_categories))
    expected_hard_categories = Counter(
        {
            category: hard_base + (1 if index < hard_remainder else 0)
            for index, category in enumerate(config.hard_negative_categories)
        }
    )
    if hard_categories != expected_hard_categories:
        raise HumanCampaignError("hard-negative category counts differ from quota")
    assignments = list(pack.blind_assignments)
    if len(assignments) != len(cards) or {item["task_id"] for item in assignments} != set(task_ids):
        raise HumanCampaignError("blind assignment coverage is incomplete")
    reviewer_slots = set(config.reviewer_slots)
    for assignment in assignments:
        if (
            set(assignment) != _BLIND_ASSIGNMENT_FIELDS
            or assignment["schema_version"] != "pii-zh-human-campaign-blind-assignment-v1"
        ):
            raise HumanCampaignError("blind reviewer assignment schema changed")
        roles = {
            assignment["annotator_a_slot"],
            assignment["annotator_b_slot"],
            assignment["adjudicator_slot"],
            assignment["privacy_reviewer_slot"],
        }
        if (
            len(roles) != 4
            or not roles <= reviewer_slots
            or assignment["blind_to_target_label"] is not True
            or assignment["blind_to_contributor_identity"] is not True
            or assignment["annotator_a_sees_b"] is not False
            or assignment["annotator_b_sees_a"] is not False
            or assignment["annotation_payload_status"] != "not_created"
            or assignment["adjudication_payload_status"] != "not_created"
            or _walk_keys(assignment) & _FORBIDDEN_ANSWER_KEYS
        ):
            raise HumanCampaignError("blind reviewer assignment is invalid")
    consent = pack.consent_template
    if (
        consent.get("template_only") is not True
        or consent.get("prechecked") is not False
        or consent.get("contributor_slot_id") is not None
        or consent.get("consent_receipt_id") is not None
        or consent.get("signed_at") is not None
        or any(value is not False for value in consent.get("checklist", {}).values())
    ):
        raise HumanCampaignError("consent template is not blank")
    signoff = pack.signoff_template
    if (
        signoff.get("template_only") is not True
        or signoff.get("collection_open") is not False
        or signoff.get("public_weight_training_allowed") is not False
        or any(
            signoff[name]["status"] != "pending"
            for name in ("legal_review", "privacy_review", "security_review")
        )
    ):
        raise HumanCampaignError("legal/privacy signoff template is not pending")
    status = pack.aggregate_status
    if any(status.get(key) != value for key, value in _ZERO_STATUS.items()) or (
        status.get("human_record_count") != 0
        or status.get("public_release_allowed") is not False
        or status.get("public_weight_training_allowed") is not False
        or status.get("collection_open") is not False
    ):
        raise HumanCampaignError("aggregate campaign status escaped zero-record state")
    return MappingProxyType(
        {
            "status": "passed",
            "campaign_split_set": list(INTAKE_SPLITS),
            "task_card_count": len(cards),
            "planned_contributor_slots": len(config.contributor_slots),
            "reviewer_slots": len(config.reviewer_slots),
            "pii_free_count": sum(mode_counts[mode] for mode in _PII_FREE_MODES),
            "pii_free_ratio": sum(mode_counts[mode] for mode in _PII_FREE_MODES) / len(cards),
            "core_label_count": len(_CORE_LABELS),
            "hard_negative_category_count": len(config.hard_negative_categories),
            "contributor_cross_split_collisions": 0,
            "document_cross_split_collisions": 0,
            "session_cross_split_collisions": 0,
            "blind_assignment_missing_count": 0,
            "answer_payload_count": 0,
            "human_record_count": 0,
            **_ZERO_STATUS,
            "public_release_allowed": False,
            "public_weight_training_allowed": False,
        }
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(child) for child in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")
    path.chmod(0o444)


def _artifact_metadata(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HumanCampaignError("campaign artifact must be a regular non-symlink file")
    file_stat = path.stat()
    return {
        "name": path.name,
        "bytes": file_stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
    }


def materialize_campaign_pack(
    *,
    pack: CampaignPack,
    config: CampaignConfig,
    output_dir: Path,
    config_path: Path = DEFAULT_CAMPAIGN_CONFIG_PATH,
    doc_path: Path = DEFAULT_CAMPAIGN_DOC_PATH,
) -> Mapping[str, Any]:
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise HumanCampaignError("campaign output must be a new absolute directory")
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or doc_path.is_symlink()
        or not doc_path.is_file()
    ):
        raise HumanCampaignError("campaign config or instructions are unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        task_path = stage / "task_cards.jsonl"
        assignment_path = stage / "blind_annotation_assignments.jsonl"
        _write_jsonl(task_path, pack.task_cards)
        _write_jsonl(assignment_path, pack.blind_assignments)
        _write_json(stage / "consent_checklist.template.json", pack.consent_template)
        _write_json(stage / "legal_privacy_signoff.template.json", pack.signoff_template)
        _write_json(stage / "aggregate_status.json", pack.aggregate_status)
        config_snapshot = stage / "campaign_config.snapshot.yaml"
        config_snapshot.write_bytes(config_path.read_bytes())
        config_snapshot.chmod(0o444)
        instructions = stage / "CAMPAIGN_INSTRUCTIONS.md"
        instructions.write_bytes(doc_path.read_bytes())
        instructions.chmod(0o444)
        files = [_artifact_metadata(path) for path in sorted(stage.iterdir())]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "human_context_campaign_zero_record_starter_pack",
            "campaign_id": config.campaign_id,
            "campaign_version": config.campaign_version,
            "config_sha256": config.sha256,
            "instructions_sha256": hashlib.sha256(doc_path.read_bytes()).hexdigest(),
            "starter_pack_only": True,
            "record_generation_allowed": False,
            "task_card_count": len(pack.task_cards),
            "blind_assignment_count": len(pack.blind_assignments),
            "human_record_count": 0,
            **_ZERO_STATUS,
            "public_release_allowed": False,
            "public_weight_training_allowed": False,
            "plan_validation": dict(pack.plan_validation),
            "files": files,
            "paths_recorded": False,
        }
        manifest["manifest_sha256"] = _canonical_hash(manifest)
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


def validate_materialized_campaign_pack(
    campaign_dir: Path,
) -> Mapping[str, Any]:
    if campaign_dir.is_symlink() or not campaign_dir.is_dir():
        raise HumanCampaignError("campaign directory is invalid")
    config_path = campaign_dir / "campaign_config.snapshot.yaml"
    config = load_campaign_config(config_path)
    task_cards = tuple(_read_jsonl(campaign_dir / "task_cards.jsonl"))
    assignments = tuple(_read_jsonl(campaign_dir / "blind_annotation_assignments.jsonl"))
    consent = _read_json(campaign_dir / "consent_checklist.template.json")
    signoff = _read_json(campaign_dir / "legal_privacy_signoff.template.json")
    status = _read_json(campaign_dir / "aggregate_status.json")
    pack = CampaignPack(
        task_cards=task_cards,
        blind_assignments=assignments,
        consent_template=consent,
        signoff_template=signoff,
        aggregate_status=status,
        plan_validation=MappingProxyType({}),
    )
    validation = validate_campaign_pack(pack, config=config)
    manifest = _read_json(campaign_dir / "manifest.json")
    claimed_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "human_context_campaign_zero_record_starter_pack"
        or manifest.get("campaign_id") != config.campaign_id
        or manifest.get("campaign_version") != config.campaign_version
        or claimed_hash != _canonical_hash(unsigned)
        or manifest.get("config_sha256") != config.sha256
        or manifest.get("starter_pack_only") is not True
        or manifest.get("record_generation_allowed") is not False
        or manifest.get("task_card_count") != len(task_cards)
        or manifest.get("blind_assignment_count") != len(assignments)
        or manifest.get("human_record_count") != 0
        or any(manifest.get(key) != value for key, value in _ZERO_STATUS.items())
        or manifest.get("public_release_allowed") is not False
        or manifest.get("public_weight_training_allowed") is not False
        or manifest.get("paths_recorded") is not False
        or manifest.get("plan_validation") != dict(validation)
    ):
        raise HumanCampaignError("campaign manifest failed zero-record validation")
    expected_files = {
        "CAMPAIGN_INSTRUCTIONS.md",
        "aggregate_status.json",
        "blind_annotation_assignments.jsonl",
        "campaign_config.snapshot.yaml",
        "consent_checklist.template.json",
        "legal_privacy_signoff.template.json",
        "manifest.json",
        "task_cards.jsonl",
    }
    if {path.name for path in campaign_dir.iterdir()} != expected_files:
        raise HumanCampaignError("campaign directory contains an unexpected file")
    payload_names = expected_files - {"manifest.json"}
    expected_metadata = [
        dict(_artifact_metadata(campaign_dir / name)) for name in sorted(payload_names)
    ]
    if manifest.get("files") != expected_metadata:
        raise HumanCampaignError("campaign artifact metadata differs from manifest")
    if any(_artifact_metadata(campaign_dir / name)["mode"] != "0444" for name in expected_files):
        raise HumanCampaignError("campaign artifacts must remain read-only")
    return MappingProxyType(
        {
            **dict(validation),
            "manifest_sha256": claimed_hash,
            "materialized_file_count": len(expected_files),
        }
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HumanCampaignError("campaign JSON artifact is invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HumanCampaignError("campaign JSON artifact failed parsing") from exc
    return _mapping(value, field="campaign JSON artifact")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise HumanCampaignError("campaign JSONL artifact is invalid")
    values = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                raise HumanCampaignError("campaign JSONL contains a blank line")
            values.append(_mapping(json.loads(line), field="campaign JSONL row"))
    except HumanCampaignError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HumanCampaignError("campaign JSONL artifact failed parsing") from exc
    return values


__all__ = [
    "CAMPAIGN_ID",
    "CAMPAIGN_SCHEMA_VERSION",
    "CAMPAIGN_VERSION",
    "DEFAULT_CAMPAIGN_CONFIG_PATH",
    "DEFAULT_CAMPAIGN_DOC_PATH",
    "CampaignConfig",
    "CampaignPack",
    "HumanCampaignError",
    "build_campaign_pack",
    "load_campaign_config",
    "materialize_campaign_pack",
    "validate_campaign_pack",
    "validate_materialized_campaign_pack",
]
