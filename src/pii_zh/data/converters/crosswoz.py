"""Fail-closed offline conversion of original CrossWOZ dialogue objects.

The official release is a JSON object mapping ``task_id`` to a full dialogue.
This module accepts one already-loaded dialogue at a time (or an iterable of
``(task_id, dialogue)`` pairs), performs no network access, and emits exactly
one canonical document per full dialogue.

CrossWOZ dialogue acts and states are useful annotation candidates, not PII
gold.  The converter therefore never promotes its output beyond quarantine.
Original utterances may contain source slot values, public phone numbers, or
addresses.  Callers can supply a complete replacement map before creating an
annotation candidate; applying replacements still does not constitute a
privacy review or make the record public-release eligible.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from ..schema import DocumentRecord, Provenance, QualityMetadata, QualityTier
from ._common import ConversionError

CROSSWOZ_SOURCE_ID = "thu-coai/CrossWOZ"
CROSSWOZ_CONTEXT_ORIGIN = "wizard_of_oz"

_DIALOGUE_FIELDS = frozenset(
    {"sys-usr", "goal", "messages", "final_goal", "task description", "type"}
)
_TURN_REQUIRED_FIELDS = frozenset({"content", "role", "dialog_act"})
_TURN_OPTIONAL_FIELDS = frozenset({"user_state", "sys_state", "sys_state_init"})
_ROLES = frozenset({"usr", "sys"})
_ACT_TYPES = frozenset({"General", "Inform", "Request", "Recommend", "Select", "NoOffer"})

_HOTEL_FACILITY_SUFFIXES = frozenset(
    {
        "24小时热水",
        "SPA",
        "中式餐厅",
        "会议室",
        "健身房",
        "免费国内长途电话",
        "免费市内电话",
        "公共区域和部分房间提供wifi",
        "公共区域提供wifi",
        "叫醒服务",
        "吹风机",
        "商务中心",
        "国际长途电话",
        "室内游泳池",
        "室外游泳池",
        "宽带上网",
        "所有房间提供wifi",
        "接待外宾",
        "接机服务",
        "接站服务",
        "收费停车位",
        "无烟房",
        "早餐服务",
        "早餐服务免费",
        "暖气",
        "桑拿",
        "棋牌室",
        "残疾人设施",
        "洗衣服务",
        "温泉",
        "看护小孩服务",
        "租车",
        "行李寄存",
        "西式餐厅",
        "部分房间提供wifi",
        "酒吧",
        "酒店各处提供wifi",
    }
)

_SLOTS_BY_DOMAIN: Mapping[str, frozenset[str]] = {
    "景点": frozenset(
        {
            "名称",
            "门票",
            "游玩时间",
            "评分",
            "周边景点",
            "周边餐馆",
            "周边酒店",
            "地址",
            "电话",
            "源领域",
        }
    ),
    "餐馆": frozenset(
        {
            "名称",
            "推荐菜",
            "人均消费",
            "评分",
            "周边景点",
            "周边餐馆",
            "周边酒店",
            "营业时间",
            "地址",
            "电话",
            "源领域",
        }
    ),
    "酒店": frozenset(
        {
            "名称",
            "酒店类型",
            "酒店设施",
            "价格",
            "评分",
            "周边景点",
            "周边餐馆",
            "周边酒店",
            "地址",
            "电话",
            "源领域",
        }
    ),
    "地铁": frozenset(
        {"出发地", "目的地", "出发地附近地铁站", "目的地附近地铁站"}
    ),
    "出租": frozenset({"出发地", "目的地", "车型", "车牌"}),
}

_EMPTY_VALUES = frozenset({"", "none"})
_PHONE_CANDIDATE = re.compile(
    r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})(?!\d)"
)


def _require_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"CrossWOZ {name} must be a non-empty string")
    return value


def _require_fields(
    value: object,
    *,
    context: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConversionError(f"CrossWOZ {context} must be an object")
    fields = set(value)
    if fields - (required | optional):
        # Do not echo an attacker-controlled field name: conversion errors are
        # allowed to cross the raw-text privacy boundary.
        raise ConversionError(f"CrossWOZ {context} contains unknown fields")
    if required - fields:
        raise ConversionError(f"CrossWOZ {context} is missing required fields")
    return value


def _require_sequence(value: object, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConversionError(f"CrossWOZ {context} must be an array")
    return value


def _slot_is_allowed(domain: str, slot: str) -> bool:
    if domain not in _SLOTS_BY_DOMAIN:
        return False
    if slot in _SLOTS_BY_DOMAIN[domain]:
        return True
    prefix = "酒店设施-"
    return (
        domain == "酒店"
        and slot.startswith(prefix)
        and slot.removeprefix(prefix) in _HOTEL_FACILITY_SUFFIXES
    )


def _collect_value(value: object, candidates: set[str], *, context: str) -> None:
    if isinstance(value, str):
        if value.strip().casefold() not in _EMPTY_VALUES:
            candidates.add(value)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if not isinstance(item, str):
                raise ConversionError(f"CrossWOZ {context} has an unsupported slot value")
            if item.strip().casefold() not in _EMPTY_VALUES:
                candidates.add(item)
        return
    raise ConversionError(f"CrossWOZ {context} has an unsupported slot value")


def _validate_goal_items(
    value: object,
    *,
    context: str,
    candidates: set[str],
) -> None:
    items = _require_sequence(value, context=context)
    for index, item in enumerate(items):
        fields = _require_sequence(item, context=f"{context} item {index}")
        if len(fields) != 5:
            raise ConversionError(f"CrossWOZ {context} item {index} has invalid arity")
        subgoal_id, domain, slot, slot_value, mentioned = fields
        if isinstance(subgoal_id, bool) or not isinstance(subgoal_id, int):
            raise ConversionError(f"CrossWOZ {context} item {index} has invalid sub-goal id")
        if not isinstance(domain, str) or not isinstance(slot, str):
            raise ConversionError(f"CrossWOZ {context} item {index} has invalid domain or slot")
        if not _slot_is_allowed(domain, slot):
            raise ConversionError(f"CrossWOZ {context} item {index} has an unsupported slot")
        if not isinstance(mentioned, bool):
            raise ConversionError(f"CrossWOZ {context} item {index} has invalid mention state")
        _collect_value(slot_value, candidates, context=f"{context} item {index}")


def _validate_dialog_acts(
    value: object,
    *,
    turn_index: int,
    candidates: set[str],
) -> None:
    acts = _require_sequence(value, context=f"message {turn_index} dialog_act")
    for act_index, act in enumerate(acts):
        fields = _require_sequence(
            act,
            context=f"message {turn_index} dialog_act item {act_index}",
        )
        if len(fields) != 4:
            raise ConversionError(
                f"CrossWOZ message {turn_index} dialog_act item {act_index} has invalid arity"
            )
        act_type, domain, slot, slot_value = fields
        if not all(isinstance(item, str) for item in fields):
            raise ConversionError(
                f"CrossWOZ message {turn_index} dialog_act item {act_index} has invalid fields"
            )
        if act_type not in _ACT_TYPES:
            raise ConversionError(
                f"CrossWOZ message {turn_index} dialog_act item {act_index} "
                "has an unsupported act type"
            )
        if act_type == "General":
            if slot != "none" or slot_value != "none":
                raise ConversionError(
                    f"CrossWOZ message {turn_index} dialog_act item {act_index} "
                    "has an invalid General act"
                )
            continue
        if act_type == "NoOffer" and domain in {"景点", "酒店", "餐馆"}:
            if slot != "none" or slot_value != "none":
                raise ConversionError(
                    f"CrossWOZ message {turn_index} dialog_act item {act_index} "
                    "has an invalid NoOffer act"
                )
            continue
        if not _slot_is_allowed(domain, slot):
            raise ConversionError(
                f"CrossWOZ message {turn_index} dialog_act item {act_index} "
                "has an unsupported slot"
            )
        _collect_value(
            slot_value,
            candidates,
            context=f"message {turn_index} dialog_act item {act_index}",
        )


def _validate_system_state(
    value: object,
    *,
    context: str,
    candidates: set[str],
) -> None:
    if not isinstance(value, Mapping):
        raise ConversionError(f"CrossWOZ {context} must be an object")
    for domain, raw_state in value.items():
        if not isinstance(domain, str) or domain not in _SLOTS_BY_DOMAIN:
            raise ConversionError(f"CrossWOZ {context} has an unsupported domain")
        if not isinstance(raw_state, Mapping):
            raise ConversionError(f"CrossWOZ {context} domain state must be an object")
        allowed = _SLOTS_BY_DOMAIN[domain] | {"selectedResults"}
        if set(raw_state) - allowed:
            raise ConversionError(f"CrossWOZ {context} domain state has an unsupported slot")
        for slot, slot_value in raw_state.items():
            if slot == "selectedResults":
                selected = _require_sequence(slot_value, context=f"{context} selectedResults")
                if any(
                    isinstance(item, bool) or not isinstance(item, (str, int))
                    for item in selected
                ):
                    raise ConversionError(
                        f"CrossWOZ {context} selectedResults has an unsupported item"
                    )
                continue
            _collect_value(slot_value, candidates, context=f"{context} domain state")


def _validate_dialogue(
    dialogue: object,
) -> tuple[list[tuple[str, str]], set[str]]:
    row = _require_fields(
        dialogue,
        context="dialogue",
        required=_DIALOGUE_FIELDS,
    )

    annotators = _require_sequence(row["sys-usr"], context="sys-usr")
    if len(annotators) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in annotators
    ):
        raise ConversionError("CrossWOZ sys-usr must contain exactly two integer identifiers")

    candidates: set[str] = set()
    _validate_goal_items(row["goal"], context="goal", candidates=candidates)
    _validate_goal_items(row["final_goal"], context="final_goal", candidates=candidates)

    task_description = _require_sequence(
        row["task description"], context="task description"
    )
    if any(not isinstance(item, str) for item in task_description):
        raise ConversionError("CrossWOZ task description must contain only strings")
    _require_non_empty_string("type", row["type"])

    raw_messages = _require_sequence(row["messages"], context="messages")
    if not raw_messages:
        raise ConversionError("CrossWOZ messages cannot be empty")
    messages: list[tuple[str, str]] = []
    for turn_index, raw_turn in enumerate(raw_messages):
        turn = _require_fields(
            raw_turn,
            context=f"message {turn_index}",
            required=_TURN_REQUIRED_FIELDS,
            optional=_TURN_OPTIONAL_FIELDS,
        )
        content = turn["content"]
        role = turn["role"]
        if not isinstance(content, str):
            raise ConversionError(f"CrossWOZ message {turn_index} content must be a string")
        if not content.strip():
            raise ConversionError(f"CrossWOZ message {turn_index} content cannot be empty")
        if not isinstance(role, str) or role not in _ROLES:
            raise ConversionError(f"CrossWOZ message {turn_index} has an unsupported role")
        _validate_dialog_acts(
            turn["dialog_act"],
            turn_index=turn_index,
            candidates=candidates,
        )
        if "user_state" in turn:
            _validate_goal_items(
                turn["user_state"],
                context=f"message {turn_index} user_state",
                candidates=candidates,
            )
        for state_field in ("sys_state_init", "sys_state"):
            if state_field in turn:
                _validate_system_state(
                    turn[state_field],
                    context=f"message {turn_index} {state_field}",
                    candidates=candidates,
                )
        messages.append((role, content))
    return messages, candidates


def _validate_replacement_map(
    replacements: Mapping[str, str],
    *,
    messages: Sequence[tuple[str, str]],
    required_values: frozenset[str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    source_texts = [content for _, content in messages]
    for original, replacement in replacements.items():
        if (
            not isinstance(original, str)
            or not original
            or not isinstance(replacement, str)
            or not replacement
            or original == replacement
        ):
            raise ConversionError("CrossWOZ replacement map contains an invalid entry")
        if not any(original in content for content in source_texts):
            raise ConversionError("CrossWOZ replacement map contains an unmatched source value")
        normalized[original] = replacement
    if required_values - normalized.keys():
        raise ConversionError("CrossWOZ replacement map does not cover every source slot value")
    return normalized


def _required_replacements(
    messages: Sequence[tuple[str, str]],
    slot_values: set[str],
) -> frozenset[str]:
    contents = [content for _, content in messages]
    required = {
        value
        for value in slot_values
        if value and any(value in content for content in contents)
    }
    for content in contents:
        required.update(match.group(0) for match in _PHONE_CANDIDATE.finditer(content))
    return frozenset(required)


def _replace_content(content: str, replacements: Mapping[str, str]) -> tuple[str, int]:
    if not replacements:
        return content, 0
    ordered = sorted(replacements, key=lambda item: (-len(item), item))
    pattern = re.compile(
        "|".join(re.escape(value) for value in ordered)
    )
    rendered, count = pattern.subn(lambda match: replacements[match.group(0)], content)
    if any(original in rendered for original in replacements):
        raise ConversionError("CrossWOZ replacement output retains a source value")
    return rendered, count


def _render_dialogue(
    messages: Sequence[tuple[str, str]],
    replacements: Mapping[str, str],
) -> tuple[str, list[dict[str, int | str]], int]:
    parts: list[str] = []
    boundaries: list[dict[str, int | str]] = []
    output_length = 0
    replacement_count = 0
    for turn_index, (role, content) in enumerate(messages):
        if parts:
            parts.append("\n")
            output_length += 1
        marker = "[USR]" if role == "usr" else "[SYS]"
        prefix = marker + "\n"
        parts.append(prefix)
        output_length += len(prefix)
        start = output_length
        rendered, count = _replace_content(content, replacements)
        parts.append(rendered)
        output_length += len(rendered)
        replacement_count += count
        boundaries.append(
            {
                "turn_index": turn_index,
                "role": role,
                "start": start,
                "end": output_length,
            }
        )
    return "".join(parts), boundaries, replacement_count


def _opaque_dialogue_identity(source_revision: str, dialogue_id: str) -> tuple[str, str]:
    material = f"{CROSSWOZ_SOURCE_ID}\0{source_revision}\0{dialogue_id}".encode()
    digest = hashlib.sha256(material).hexdigest()
    return "sha256:" + digest, "crosswoz-dialogue:sha256:" + digest


def convert_crosswoz_record(
    dialogue: Mapping[str, Any],
    *,
    dialogue_id: str,
    source_revision: str,
    source_license: str,
    replacement_map: Mapping[str, str] | None = None,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> DocumentRecord:
    """Convert one complete original CrossWOZ dialogue into a quarantine candidate.

    ``quality_gate`` and ``public_weight_training_allowed`` are accepted only
    to make the fail-closed boundary explicit.  This source converter cannot
    certify annotation quality, privacy review, or public-weight rights and
    therefore rejects any attempt to open either gate.
    """

    source_revision = _require_non_empty_string("source_revision", source_revision)
    source_license = _require_non_empty_string("source_license", source_license)
    dialogue_id = _require_non_empty_string("dialogue_id", dialogue_id)
    if quality_gate is not False or public_weight_training_allowed is not None:
        raise ConversionError("CrossWOZ conversion emits quarantine candidates only")

    messages, slot_values = _validate_dialogue(dialogue)
    required_values = _required_replacements(messages, slot_values)
    if replacement_map is None:
        replacements: dict[str, str] = {}
    else:
        if not isinstance(replacement_map, Mapping):
            raise ConversionError("CrossWOZ replacement_map must be an object")
        replacements = _validate_replacement_map(
            replacement_map,
            messages=messages,
            required_values=required_values,
        )
    text, turn_boundaries, replacement_count = _render_dialogue(messages, replacements)
    doc_id, conversation_group = _opaque_dialogue_identity(source_revision, dialogue_id)

    provenance = Provenance(
        source_id=CROSSWOZ_SOURCE_ID,
        source_kind="public_human_wizard_of_oz_dialogue",
        source_revision=source_revision,
        license=source_license,
        synthetic=False,
        metadata={
            "converted_offline": True,
            "source_schema": "crosswoz-original-dialogue-v1",
            "context_origin": CROSSWOZ_CONTEXT_ORIGIN,
            "contains_real_personal_data": None,
            "privacy_review_status": "pending",
            "source_slot_values_stored_in_metadata": False,
            "source_annotator_identifiers_stored": False,
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.UNCERTAIN,
        quality_gate=False,
        validators_passed=False,
        review_status="privacy_license_and_human_annotation_review_pending",
        confidence=None,
        metadata={
            "structural_schema_validated": True,
            "human_annotation_complete": False,
            "privacy_review_complete": False,
        },
    )
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(),
        language="zh-Hans-CN",
        domain="travel_and_local_services",
        scene="task_oriented_dialogue",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=None,
        conversation_group=conversation_group,
        entity_value_groups=(),
        metadata={
            "context_origin": CROSSWOZ_CONTEXT_ORIGIN,
            "contains_real_personal_data": None,
            "privacy_review_status": "pending",
            "annotation_status": "pending",
            "full_dialogue_preserved": True,
            "turn_count": len(messages),
            "turn_boundaries": turn_boundaries,
            "replacement_map_applied": replacement_map is not None,
            "replacement_count": replacement_count,
            "source_slot_value_candidate_count": len(required_values),
            "requires_surrogate_replacement": bool(required_values - replacements.keys()),
            "source_slot_values_stored_in_metadata": False,
            "source_group": conversation_group,
            "training_role": "annotation_candidate_only",
        },
    )


def iter_crosswoz_records(
    dialogues: Mapping[str, Mapping[str, Any]] | Iterable[tuple[str, Mapping[str, Any]]],
    *,
    source_revision: str,
    source_license: str,
    replacement_maps: Mapping[str, Mapping[str, str]] | None = None,
    quality_gate: bool = False,
    public_weight_training_allowed: bool | None = None,
) -> Iterator[DocumentRecord]:
    """Yield one quarantine record per full dialogue without downloading data."""

    items: Iterable[tuple[str, Mapping[str, Any]]]
    items = dialogues.items() if isinstance(dialogues, Mapping) else dialogues
    for dialogue_id, dialogue in items:
        replacement_map = (
            replacement_maps.get(dialogue_id) if replacement_maps is not None else None
        )
        yield convert_crosswoz_record(
            dialogue,
            dialogue_id=dialogue_id,
            source_revision=source_revision,
            source_license=source_license,
            replacement_map=replacement_map,
            quality_gate=quality_gate,
            public_weight_training_allowed=public_weight_training_allowed,
        )


__all__ = [
    "CROSSWOZ_CONTEXT_ORIGIN",
    "CROSSWOZ_SOURCE_ID",
    "convert_crosswoz_record",
    "iter_crosswoz_records",
]
