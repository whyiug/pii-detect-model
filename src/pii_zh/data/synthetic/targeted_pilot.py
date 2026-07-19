"""Train/validation-only targeted synthetic augmentation pilot.

This module never opens test or evaluation data.  It profiles only canonical
D0 train/validation records, creates new deterministic synthetic text, and
validates template-family, entity-value, and relation-group isolation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ..schema import (
    DataPool,
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    iter_jsonl,
    stable_text_hash,
    write_jsonl,
)
from .generator import DEFAULT_LICENSE, render_template
from .materialize import assert_surface_contract, sha256_file
from .templates import CORE_LABELS, PlaceholderSpec, SyntheticTemplate
from .values import SyntheticValueFactory, generate_value

PILOT_DATASET_ID = "pii_zh_synthetic_targeted_v2_pilot"
PILOT_DATASET_VERSION = "0.1.0"
PILOT_GENERATOR_NAME = "pii_zh_targeted_synthetic_pilot"
PILOT_GENERATOR_VERSION = "0.1.0"
PILOT_MATERIALIZER_VERSION = "0.1.0"
DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[4] / "configs/data/targeted_augmentation_v2_pilot.yaml"
)
_SPLITS = ("train", "validation")
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_TARGET_LABELS = frozenset({"ADDRESS", "PERSON_NAME", "PHONE_NUMBER", "VEHICLE_LICENSE_PLATE"})
_SAFE_TOKEN = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


class TargetedPilotError(ValueError):
    """Fail-closed error that never includes source text or entity values."""


@dataclass(frozen=True, slots=True)
class PilotRecipe:
    recipe_id: str
    recipe_version: str
    seed: int
    counts: Mapping[str, int]
    pii_free_ratio: float
    required_core_labels: tuple[str, ...]
    target_labels: Mapping[str, int]
    default_label_weight: int
    minimum_chat_context_ratio: float
    validation_only_signal: Mapping[str, Any]
    pii_free_gate: Mapping[str, Any]
    input_contract: Mapping[str, Any]
    generation_contract: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceSplitProfile:
    split: str
    sha256: str
    size_bytes: int
    record_count: int
    pii_free_count: int
    label_counts: Mapping[str, int]
    template_groups: frozenset[str]
    entity_value_groups: frozenset[str]
    relation_groups: frozenset[str]
    text_hashes: frozenset[str]


@dataclass(frozen=True, slots=True)
class PilotBuild:
    records: tuple[DocumentRecord, ...]
    manifest: Mapping[str, Any]


_LABEL_DISPLAY: Mapping[str, str] = MappingProxyType(
    {
        "PERSON_NAME": "姓名",
        "PHONE_NUMBER": "联系电话",
        "EMAIL_ADDRESS": "邮箱",
        "ADDRESS": "详细地址",
        "DATE_OF_BIRTH": "出生日期",
        "CN_RESIDENT_ID": "居民身份证号",
        "PASSPORT_NUMBER": "护照号",
        "DRIVER_LICENSE_NUMBER": "驾驶证号",
        "SOCIAL_SECURITY_NUMBER": "社保号",
        "BANK_CARD_NUMBER": "银行卡号",
        "BANK_ACCOUNT_NUMBER": "银行账户",
        "VEHICLE_LICENSE_PLATE": "车牌",
        "EMPLOYEE_ID": "员工号",
        "STUDENT_ID": "学号",
        "MEDICAL_RECORD_NUMBER": "病历号",
        "WECHAT_ID": "微信号",
        "QQ_NUMBER": "QQ号",
        "ALIPAY_ACCOUNT": "支付账号",
        "USERNAME": "用户名",
        "IP_ADDRESS": "IP地址",
        "MAC_ADDRESS": "MAC地址",
        "DEVICE_ID": "设备号",
        "GEO_COORDINATE": "坐标",
        "SECRET": "访问密钥",
    }
)


_TARGET_BODIES: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "train": {
            "ADDRESS": (
                "用户：收货地改成“<<ADDRESS_1>>”，引号外的备注不要并入地址。\n客服：已只更新地址值。",
                "用户：请送到<<ADDRESS_1>>；分号后写的是配送说明。\n客服：地址边界已确认。",
                "群聊：地址栏为（<<ADDRESS_1>>），括号只用于分隔字段。",
            ),
            "PERSON_NAME": (
                "用户：请联系<<PERSON_NAME_1>>女士，称谓不属于姓名。\n客服：明白。",
                "群聊：负责人写作“<<PERSON_NAME_1>>”，只处理引号内姓名。",
                "客服：登记人为<<PERSON_NAME_1>>，后面的老师是称呼。",
            ),
            "PHONE_NUMBER": (
                "用户：回拨号码是（<<PHONE_NUMBER_1>>），两侧括号不属于号码。\n客服：收到。",
                "群聊：联系电话：<<PHONE_NUMBER_1>>；分号后是备注。",
                "客服：只更新<<PHONE_NUMBER_1>>这个号码，句末标点不要包含。",
            ),
            "VEHICLE_LICENSE_PLATE": (
                "用户：进场车辆写作“<<VEHICLE_LICENSE_PLATE_1>>”，引号不属于车牌。",
                "客服：车牌为（<<VEHICLE_LICENSE_PLATE_1>>），括号只作分隔。",
                "群聊：请核对车牌<<VEHICLE_LICENSE_PLATE_1>>；后面是停车说明。",
            ),
        },
        "validation": {
            "ADDRESS": (
                "客服：请确认地址字段【<<ADDRESS_1>>】。\n用户：方括号之外不是地址。",
                "用户：新地址是<<ADDRESS_1>>，逗号后的文字只是留言。",
                "客服：地址：『<<ADDRESS_1>>』；书名号和字段名均不属于地址值。",
            ),
            "PERSON_NAME": (
                "客服：收件人是（<<PERSON_NAME_1>>）先生吗？\n用户：姓名不含括号和先生。",
                "用户：人员字段填『<<PERSON_NAME_1>>』，只保护姓名本身。",
                "客服：我会转告<<PERSON_NAME_1>>同学，称呼不并入姓名。",
            ),
            "PHONE_NUMBER": (
                "客服：号码【<<PHONE_NUMBER_1>>】是否正确？\n用户：方括号不是号码的一部分。",
                "用户：联系电话填<<PHONE_NUMBER_1>>，后面的逗号不包含在内。",
                "客服：回拨字段是『<<PHONE_NUMBER_1>>』，只处理字段值。",
            ),
            "VEHICLE_LICENSE_PLATE": (
                "客服：车辆标识【<<VEHICLE_LICENSE_PLATE_1>>】已录入，方括号不属于车牌。",
                "用户：车牌填<<VEHICLE_LICENSE_PLATE_1>>，逗号以后是颜色说明。",
                "客服：请确认『<<VEHICLE_LICENSE_PLATE_1>>』，只保护车牌字符。",
            ),
        },
    }
)


_HARD_NEGATIVE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "generic_region",
        "ORG_ADDRESS_1",
        "ORG_ADDRESS",
        "机构办公点<<ORG_ADDRESS_1>>属于组织地址，不作为个人详细住址。",
    ),
    (
        "public_hotline",
        "ORDER_ID_1",
        "ORDER_ID",
        "公共便民热线12345关联工单<<ORDER_ID_1>>，热线不是个人联系电话。",
    ),
    (
        "short_code",
        "PRODUCT_CODE_1",
        "PRODUCT_CODE",
        "取件短码<<PRODUCT_CODE_1>>是业务代码，不是手机号码。",
    ),
    (
        "product_name",
        "DEVICE_MODEL_1",
        "DEVICE_MODEL",
        "产品名<<DEVICE_MODEL_1>>是设备型号，不是自然人姓名。",
    ),
    (
        "plate_confusable",
        "BUILD_ID_1",
        "BUILD_ID",
        "标记<<BUILD_ID_1>>是构建编号，不是车辆号牌。",
    ),
    (
        "masked_account",
        "MASKED_PHONE_1",
        "MASKED_PHONE",
        "页面只显示<<MASKED_PHONE_1>>，掩码值不是完整联系电话。",
    ),
    (
        "generic_date",
        "GENERIC_DATE_1",
        "GENERIC_DATE",
        "日期<<GENERIC_DATE_1>>是软件发布日期，不是出生日期。",
    ),
    (
        "version_number",
        "BUILD_ID_1",
        "BUILD_ID",
        "版本构建号<<BUILD_ID_1>>不是IP地址或设备标识。",
    ),
    (
        "organization",
        "ORGANIZATION_1",
        "ORGANIZATION",
        "字段值<<ORGANIZATION_1>>是机构名称，不是个人姓名。",
    ),
    (
        "order_suffix",
        "ORDER_ID_1",
        "ORDER_ID",
        "业务工单<<ORDER_ID_1>>不是银行卡、证件号或银行账户。",
    ),
    (
        "invalid_bank_card",
        "INVALID_CARD_1",
        "INVALID_CARD",
        "号码<<INVALID_CARD_1>>未通过Luhn校验，不作为银行卡号。",
    ),
    (
        "invalid_resident_id",
        "INVALID_CN_ID_1",
        "INVALID_CN_ID",
        "号码<<INVALID_CN_ID_1>>校验位错误，不作为居民身份证号。",
    ),
)

_NEGATIVE_PREFIXES: tuple[str, ...] = (
    "用户在群聊里补充：",
    "客服复核后说明：",
    "工单备注记录：",
    "同事在消息中确认：",
    "值班对话提到：",
    "服务线程里写着：",
)
_NEGATIVE_SUFFIXES: tuple[str, ...] = (
    "\n客服：请保留它的非个人信息语义。",
    "\n用户：不要把这一项扩展成个人字段。",
    "\n客服：系统应按业务字段处理。",
    "\n群聊：该值只用于当前业务流程。",
)
_NEAR_DUPLICATE_NGRAM = 5
_NEAR_DUPLICATE_JACCARD = 0.92


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _exact_mapping(value: object, *, field: str, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TargetedPilotError(f"{field} must be a string-keyed mapping")
    if set(value) != fields:
        raise TargetedPilotError(f"{field} fields differ from the recipe schema")
    return value


def _token(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character not in _SAFE_TOKEN for character in value)
    ):
        raise TargetedPilotError(f"{field} must be a lowercase safe token")
    return value


def _integer(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TargetedPilotError(f"{field} must be an integer of at least {minimum}")
    return value


def load_pilot_recipe(path: Path = DEFAULT_RECIPE_PATH) -> PilotRecipe:
    """Load one strict recipe byte snapshot."""

    if path.is_symlink() or not path.is_file():
        raise TargetedPilotError("pilot recipe must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TargetedPilotError("pilot recipe is invalid YAML") from exc
    root = _exact_mapping(
        document,
        field="recipe",
        fields=frozenset(
            {
                "schema_version",
                "recipe_id",
                "recipe_version",
                "seed",
                "counts",
                "pii_free_ratio",
                "required_core_labels",
                "target_labels",
                "default_label_weight",
                "minimum_chat_context_ratio",
                "validation_only_signal",
                "pii_free_gate",
                "input_contract",
                "generation_contract",
            }
        ),
    )
    if root["schema_version"] != 1:
        raise TargetedPilotError("pilot recipe schema_version must equal 1")
    recipe_id = _token(root["recipe_id"], field="recipe_id")
    version = root["recipe_version"]
    if not isinstance(version, str) or version != PILOT_DATASET_VERSION:
        raise TargetedPilotError("pilot recipe_version is unsupported")
    seed = _integer(root["seed"], field="seed", minimum=0)
    counts_raw = _exact_mapping(root["counts"], field="counts", fields=frozenset(_SPLITS))
    counts = {split: _integer(counts_raw[split], field=f"counts.{split}") for split in _SPLITS}
    ratio = root["pii_free_ratio"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not 0.30 <= float(ratio) <= 0.40
    ):
        raise TargetedPilotError("pii_free_ratio must be within [0.30, 0.40]")
    labels = root["required_core_labels"]
    if (
        isinstance(labels, (str, bytes))
        or not isinstance(labels, Sequence)
        or tuple(labels) != CORE_LABELS
    ):
        raise TargetedPilotError("required_core_labels must equal the ordered 24-label contract")
    targets_raw = _exact_mapping(
        root["target_labels"], field="target_labels", fields=_TARGET_LABELS
    )
    targets = {
        label: _integer(targets_raw[label], field=f"target_labels.{label}")
        for label in sorted(_TARGET_LABELS)
    }
    if any(weight != 4 for weight in targets.values()):
        raise TargetedPilotError("target label weight must remain four in the pilot")
    default_weight = _integer(root["default_label_weight"], field="default_label_weight")
    if default_weight != 1:
        raise TargetedPilotError("default_label_weight must equal one")
    chat_ratio = root["minimum_chat_context_ratio"]
    if (
        isinstance(chat_ratio, bool)
        or not isinstance(chat_ratio, (int, float))
        or not 0.0 <= float(chat_ratio) <= 1.0
    ):
        raise TargetedPilotError("minimum_chat_context_ratio must be a probability")
    signal = _exact_mapping(
        root["validation_only_signal"],
        field="validation_only_signal",
        fields=frozenset(
            {
                "source",
                "external_benchmark",
                "document_count",
                "false_positive_document_count",
                "pii_free_document_fpr",
                "concentrated_false_positive_labels",
            }
        ),
    )
    signal_labels = _exact_mapping(
        signal["concentrated_false_positive_labels"],
        field="validation_only_signal.concentrated_false_positive_labels",
        fields=frozenset({"BANK_CARD_NUMBER", "CN_RESIDENT_ID", "DATE_OF_BIRTH"}),
    )
    signal_documents = _integer(
        signal["document_count"], field="validation_only_signal.document_count"
    )
    signal_false_positives = _integer(
        signal["false_positive_document_count"],
        field="validation_only_signal.false_positive_document_count",
    )
    signal_fpr = signal["pii_free_document_fpr"]
    if (
        signal["source"] != "user_reported_local_validation_only"
        or signal["external_benchmark"] is not False
        or not isinstance(signal_fpr, (int, float))
        or isinstance(signal_fpr, bool)
        or signal_false_positives > signal_documents
        or abs(float(signal_fpr) - signal_false_positives / signal_documents) > 1e-12
        or sum(
            _integer(value, field="validation_only_signal.label_count")
            for value in signal_labels.values()
        )
        != signal_false_positives
    ):
        raise TargetedPilotError("validation-only optimization signal is inconsistent")
    pii_free_gate = _exact_mapping(
        root["pii_free_gate"],
        field="pii_free_gate",
        fields=frozenset(
            {
                "minimum_data_ratio",
                "maximum_data_ratio",
                "required_structural_negative_categories",
                "minimum_category_documents",
                "model_document_fpr_maximum",
            }
        ),
    )
    required_negative_categories = _exact_mapping(
        pii_free_gate["required_structural_negative_categories"],
        field="pii_free_gate.required_structural_negative_categories",
        fields=frozenset(signal_labels),
    )
    if required_negative_categories != {
        "BANK_CARD_NUMBER": "invalid_bank_card",
        "CN_RESIDENT_ID": "invalid_resident_id",
        "DATE_OF_BIRTH": "generic_date",
    }:
        raise TargetedPilotError("required structural hard-negative mapping changed")
    category_minimums = _exact_mapping(
        pii_free_gate["minimum_category_documents"],
        field="pii_free_gate.minimum_category_documents",
        fields=frozenset(_SPLITS),
    )
    gate_minimum = pii_free_gate["minimum_data_ratio"]
    gate_maximum = pii_free_gate["maximum_data_ratio"]
    model_fpr_maximum = pii_free_gate["model_document_fpr_maximum"]
    if (
        gate_minimum != 0.30
        or gate_maximum != 0.40
        or recipe_id != "pii_zh_synthetic_targeted_augmentation_v2_pilot"
        or _integer(category_minimums["train"], field="minimum_category_documents.train") != 16
        or _integer(
            category_minimums["validation"],
            field="minimum_category_documents.validation",
        )
        != 4
        or isinstance(model_fpr_maximum, bool)
        or not isinstance(model_fpr_maximum, (int, float))
        or not 0.0 <= float(model_fpr_maximum) <= 1.0
    ):
        raise TargetedPilotError("PII-free gate changed or is invalid")
    input_contract = _exact_mapping(
        root["input_contract"],
        field="input_contract",
        fields=frozenset(
            {
                "allowed_split_filenames",
                "forbidden_path_markers",
                "require_same_parent_directory",
                "require_synthetic_provenance",
                "copy_source_text",
            }
        ),
    )
    allowed_names = _exact_mapping(
        input_contract["allowed_split_filenames"],
        field="input_contract.allowed_split_filenames",
        fields=frozenset(_SPLITS),
    )
    if allowed_names != {"train": "train.jsonl", "validation": "validation.jsonl"}:
        raise TargetedPilotError("allowed split filenames changed")
    markers = input_contract["forbidden_path_markers"]
    if (
        isinstance(markers, (str, bytes))
        or not isinstance(markers, Sequence)
        or any(not isinstance(item, str) or not item for item in markers)
    ):
        raise TargetedPilotError("forbidden_path_markers must be non-empty strings")
    if (
        input_contract["require_same_parent_directory"] is not True
        or input_contract["require_synthetic_provenance"] is not True
        or input_contract["copy_source_text"] is not False
    ):
        raise TargetedPilotError("pilot input safeguards cannot be relaxed")
    generation_contract = _exact_mapping(
        root["generation_contract"],
        field="generation_contract",
        fields=frozenset(
            {
                "context_origin",
                "output_is_synthetic",
                "human_authored_context",
                "llm_generated_context",
                "external_benchmark_used_for_recipe",
                "selection_basis",
                "allowed_output_splits",
            }
        ),
    )
    if (
        generation_contract["context_origin"]
        != "deterministic_repository_authored_synthetic_template"
        or generation_contract["output_is_synthetic"] is not True
        or generation_contract["human_authored_context"] is not False
        or generation_contract["llm_generated_context"] is not False
        or generation_contract["external_benchmark_used_for_recipe"] is not False
        or generation_contract["allowed_output_splits"] != list(_SPLITS)
    ):
        raise TargetedPilotError("pilot generation safeguards cannot be relaxed")
    if generation_contract["selection_basis"] != [
        "user_specified_target_labels",
        "d0_train_validation_schema_and_group_profile",
    ]:
        raise TargetedPilotError("pilot selection basis changed")
    return PilotRecipe(
        recipe_id=recipe_id,
        recipe_version=version,
        seed=seed,
        counts=MappingProxyType(counts),
        pii_free_ratio=float(ratio),
        required_core_labels=tuple(labels),
        target_labels=MappingProxyType(targets),
        default_label_weight=default_weight,
        minimum_chat_context_ratio=float(chat_ratio),
        validation_only_signal=MappingProxyType(dict(signal)),
        pii_free_gate=MappingProxyType(dict(pii_free_gate)),
        input_contract=MappingProxyType(dict(input_contract)),
        generation_contract=MappingProxyType(dict(generation_contract)),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _safe_source_path(path: Path, *, split: str, recipe: PilotRecipe) -> Path:
    expected = recipe.input_contract["allowed_split_filenames"][split]
    if not path.is_absolute() or path.name != expected:
        raise TargetedPilotError("D0 source path must be the expected absolute split filename")
    lowered = path.as_posix().casefold()
    if any(
        str(marker).casefold() in lowered
        for marker in recipe.input_contract["forbidden_path_markers"]
    ):
        raise TargetedPilotError("D0 source path contains a forbidden evaluation marker")
    if path.is_symlink() or not path.is_file():
        raise TargetedPilotError("D0 source must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_regular_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_SOURCE_BYTES:
            raise TargetedPilotError("D0 source is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(_MAX_SOURCE_BYTES + 1)
        after = os.fstat(descriptor)
    except TargetedPilotError:
        raise
    except OSError as exc:
        raise TargetedPilotError("D0 source could not be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_SOURCE_BYTES or len(raw) != before.st_size:
        raise TargetedPilotError("D0 source size changed during read")
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise TargetedPilotError("D0 source changed during read")
    return raw, before


def profile_d0_split(path: Path, *, split: str, recipe: PilotRecipe) -> SourceSplitProfile:
    """Profile one canonical D0 split without retaining or returning raw text."""

    if split not in _SPLITS:
        raise TargetedPilotError("D0 source split must be train or validation")
    path = _safe_source_path(path, split=split, recipe=recipe)
    raw, source_stat = _read_regular_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise TargetedPilotError("D0 source is not UTF-8") from exc
    labels: Counter[str] = Counter()
    templates: set[str] = set()
    values: set[str] = set()
    relations: set[str] = set()
    text_hashes: set[str] = set()
    record_count = 0
    pii_free_count = 0
    try:
        records = iter_jsonl(io.StringIO(text))
        for record in records:
            record_count += 1
            if (
                record.split != split
                or record.data_pool is not DataPool.PUBLIC_RELEASE
                or record.public_weight_training_allowed is not True
                or record.provenance.synthetic is not True
                or record.provenance.evaluation_only
                or record.provenance.private_enterprise
            ):
                raise TargetedPilotError("D0 source record violated the train/validation contract")
            observed_labels = {entity.label for entity in record.entities}
            if not observed_labels <= set(CORE_LABELS):
                raise TargetedPilotError("D0 source contains a non-core label")
            labels.update(entity.label for entity in record.entities)
            if not record.entities:
                pii_free_count += 1
            if not record.template_group:
                raise TargetedPilotError("D0 source record lacks a template family")
            templates.add(record.template_group)
            for value_group in record.entity_value_groups:
                if value_group in values:
                    raise TargetedPilotError("D0 source repeats an entity value group")
                values.add(value_group)
            if record.conversation_group:
                relations.add(record.conversation_group)
            text_hashes.add(hashlib.sha256(record.text.encode("utf-8")).hexdigest())
    except TargetedPilotError:
        raise
    except Exception as exc:
        raise TargetedPilotError("D0 source record failed canonical validation") from exc
    if record_count < 1 or set(labels) != set(CORE_LABELS):
        raise TargetedPilotError("D0 source does not cover all 24 core labels")
    return SourceSplitProfile(
        split=split,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=source_stat.st_size,
        record_count=record_count,
        pii_free_count=pii_free_count,
        label_counts=MappingProxyType(dict(sorted(labels.items()))),
        template_groups=frozenset(templates),
        entity_value_groups=frozenset(values),
        relation_groups=frozenset(relations),
        text_hashes=frozenset(text_hashes),
    )


def profile_d0_sources(
    *, train_path: Path, validation_path: Path, recipe: PilotRecipe
) -> Mapping[str, SourceSplitProfile]:
    train = _safe_source_path(train_path, split="train", recipe=recipe)
    validation = _safe_source_path(validation_path, split="validation", recipe=recipe)
    if recipe.input_contract["require_same_parent_directory"] and train.parent != validation.parent:
        raise TargetedPilotError("D0 train and validation must share one source directory")
    profiles = {
        "train": profile_d0_split(train, split="train", recipe=recipe),
        "validation": profile_d0_split(validation, split="validation", recipe=recipe),
    }
    _assert_profile_isolation(profiles)
    return MappingProxyType(profiles)


def _assert_profile_isolation(profiles: Mapping[str, SourceSplitProfile]) -> None:
    train, validation = profiles["train"], profiles["validation"]
    for field in ("template_groups", "entity_value_groups", "relation_groups", "text_hashes"):
        if getattr(train, field) & getattr(validation, field):
            raise TargetedPilotError(f"D0 {field} leak across train and validation")


def _generic_template(split: str, label: str) -> SyntheticTemplate:
    display = _LABEL_DISPLAY[label]
    if split == "train":
        body = (
            f"群聊：资料里的{display}字段更新为「<<{label}_1>>」。\n"
            "客服：只处理括号内字段值，外围符号不属于值。"
        )
    else:
        body = f"客服：请确认{display}字段（<<{label}_1>>）。\n用户：仅处理圆括号里的内容。"
    return SyntheticTemplate(
        template_id=f"target_v2_{split}_{label.casefold()}_generic_chat",
        family=f"target-v2-{split}-{label.casefold()}-generic-chat",
        domain="customer_support",
        scene="synthetic_chat_field_update",
        body=body,
        placeholders=(PlaceholderSpec(f"{label}_1", label, label),),
        hard_negative=False,
        origin_kind="repository_authored_targeted_synthetic",
    )


def _positive_templates(split: str) -> tuple[SyntheticTemplate, ...]:
    templates = [_generic_template(split, label) for label in CORE_LABELS]
    for label in sorted(_TARGET_LABELS):
        for index, body in enumerate(_TARGET_BODIES[split][label], start=1):
            templates.append(
                SyntheticTemplate(
                    template_id=f"target_v2_{split}_{label.casefold()}_boundary_{index}",
                    family=f"target-v2-{split}-{label.casefold()}-boundary-{index}",
                    domain="customer_support",
                    scene="synthetic_chat_strict_boundary",
                    body=body,
                    placeholders=(PlaceholderSpec(f"{label}_1", label, label),),
                    hard_negative=False,
                    origin_kind="repository_authored_targeted_synthetic",
                )
            )
    return tuple(templates)


def _negative_templates(split: str, count: int) -> tuple[SyntheticTemplate, ...]:
    """Create one surface-diverse template per requested negative document."""

    base, remainder = divmod(count, len(_HARD_NEGATIVE_SPECS))
    templates: list[SyntheticTemplate] = []
    split_offset = 0 if split == "train" else 3
    for category_index, (category, key, generator_key, middle) in enumerate(_HARD_NEGATIVE_SPECS):
        category_count = base + (1 if category_index < remainder else 0)
        for variant in range(category_count):
            prefix = _NEGATIVE_PREFIXES[
                (variant + category_index + split_offset) % len(_NEGATIVE_PREFIXES)
            ]
            suffix = _NEGATIVE_SUFFIXES[
                (variant // len(_NEGATIVE_PREFIXES) + category_index + split_offset)
                % len(_NEGATIVE_SUFFIXES)
            ]
            templates.append(
                SyntheticTemplate(
                    template_id=(f"target_v2_{split}_negative_{category}_{variant + 1:02d}"),
                    family=f"target-v2-{split}-negative-{category}-{variant + 1:02d}",
                    domain="customer_support",
                    scene=f"synthetic_chat_hard_negative_{category}",
                    body=prefix + middle + suffix,
                    placeholders=(PlaceholderSpec(key, generator_key, None),),
                    hard_negative=True,
                    origin_kind="repository_authored_targeted_synthetic",
                )
            )
    if len(templates) != count:
        raise TargetedPilotError("hard-negative template allocation differs from its count")
    return tuple(templates)


def _select_cycles(
    templates: tuple[SyntheticTemplate, ...], count: int, *, rng: random.Random
) -> list[SyntheticTemplate]:
    selected: list[SyntheticTemplate] = []
    while len(selected) < count:
        cycle = list(templates)
        rng.shuffle(cycle)
        selected.extend(cycle[: count - len(selected)])
    return selected


def _generate_record(
    *,
    template: SyntheticTemplate,
    split: str,
    index: int,
    recipe: PilotRecipe,
    factory: SyntheticValueFactory,
) -> DocumentRecord:
    values = {
        spec.key: generate_value(factory, spec.generator_key, value_variant=spec.value_variant)
        for spec in template.placeholders
    }
    rendered = render_template(template, values)
    if not rendered.validators_passed:
        raise TargetedPilotError("generated pilot value failed a deterministic validator")
    identity = (
        f"{recipe.sha256}\0{recipe.seed}\0{split}\0{index}\0{template.template_id}\0{rendered.text}"
    )
    identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    relation_group = (
        "synthetic-target-relation:sha256:"
        + hashlib.sha256(f"relation\0{identity_digest}".encode()).hexdigest()
    )
    labels = {entity.label for entity in rendered.entities}
    focus_labels = sorted(labels & _TARGET_LABELS)
    non_entity_value_groups = tuple(
        sorted(
            stable_text_hash(rendered.values[spec.key])
            for spec in template.placeholders
            if spec.label is None
        )
    )
    provenance = Provenance(
        source_id="repo_targeted_synthetic_v2_pilot",
        source_kind="deterministic_repository_authored_synthetic",
        source_revision=f"sha256:{recipe.sha256}",
        license=DEFAULT_LICENSE,
        synthetic=True,
        generator_name=PILOT_GENERATOR_NAME,
        generator_version=PILOT_GENERATOR_VERSION,
        generator_seed=recipe.seed,
        template_id=template.template_id,
        metadata={
            "contains_real_personal_data": False,
            "context_origin": recipe.generation_contract["context_origin"],
            "human_authored_context": False,
            "llm_generated_context": False,
            "external_benchmark_used_for_recipe": False,
            "source_text_copied": False,
            "recipe_sha256": recipe.sha256,
        },
    )
    quality = QualityMetadata(
        tier=(
            QualityTier.HARD_NEGATIVE if template.hard_negative else QualityTier.SYNTHETIC_VALIDATED
        ),
        quality_gate=True,
        validators_passed=True,
        review_status="repository_recipe_and_deterministic_validators_passed",
        confidence=1.0,
        metadata={"validator_results": dict(rendered.validator_results)},
    )
    return DocumentRecord.create(
        doc_id="sha256:" + identity_digest,
        text=rendered.text,
        entities=rendered.entities,
        language="zh-Hans-CN",
        domain=template.domain,
        scene=template.scene,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        conversation_group=relation_group,
        template_group=f"synthetic-target-v2-family:{template.family}",
        entity_value_groups=tuple(
            sorted(stable_text_hash(entity.text) for entity in rendered.entities)
        ),
        generator_version=PILOT_GENERATOR_VERSION,
        split=split,
        metadata={
            "synthetic": True,
            "targeted_augmentation": True,
            "context_origin": recipe.generation_contract["context_origin"],
            "human_authored_context": False,
            "llm_generated_context": False,
            "source_text_copied": False,
            "external_benchmark_used_for_recipe": False,
            "chat_context": True,
            "strict_boundary_focus": "ADDRESS" in labels,
            "target_focus_labels": focus_labels,
            "hard_negative": template.hard_negative,
            "hard_negative_category": (
                template.scene.removeprefix("synthetic_chat_hard_negative_")
                if template.hard_negative
                else None
            ),
            "non_entity_value_groups": list(non_entity_value_groups),
            "hard_negative_generator_keys": sorted(
                spec.generator_key for spec in template.placeholders if spec.label is None
            ),
            "relation_group": relation_group,
            "training_role": "synthetic_targeted_augmentation_pilot",
        },
    )


def generate_pilot_records(recipe: PilotRecipe) -> tuple[DocumentRecord, ...]:
    rng = random.Random(recipe.seed)
    factory = SyntheticValueFactory(rng)
    records: list[DocumentRecord] = []
    global_index = 0
    for split in _SPLITS:
        total = recipe.counts[split]
        negative_count = int(round(total * recipe.pii_free_ratio))
        if negative_count / total != recipe.pii_free_ratio:
            raise TargetedPilotError("pilot counts cannot represent the exact PII-free ratio")
        positive_count = total - negative_count
        positive_templates = _positive_templates(split)
        negative_templates = _negative_templates(split, negative_count)
        expected_template_count = len(CORE_LABELS) + sum(
            weight - recipe.default_label_weight for weight in recipe.target_labels.values()
        )
        if len(positive_templates) != expected_template_count:
            raise TargetedPilotError("positive template weighting differs from the recipe")
        selected = _select_cycles(positive_templates, positive_count, rng=rng)
        selected.extend(negative_templates)
        rng.shuffle(selected)
        for template in selected:
            records.append(
                _generate_record(
                    template=template,
                    split=split,
                    index=global_index,
                    recipe=recipe,
                    factory=factory,
                )
            )
            global_index += 1
    return tuple(records)


def _group_sets(records: Sequence[DocumentRecord]) -> Mapping[str, Mapping[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {
        "template_groups": {split: set() for split in _SPLITS},
        "entity_value_groups": {split: set() for split in _SPLITS},
        "relation_groups": {split: set() for split in _SPLITS},
        "text_hashes": {split: set() for split in _SPLITS},
    }
    for record in records:
        if (
            record.split not in _SPLITS
            or not record.template_group
            or not record.conversation_group
        ):
            raise TargetedPilotError("generated pilot record lacks a required split group")
        split = record.split
        result["template_groups"][split].add(record.template_group)
        result["entity_value_groups"][split].update(record.entity_value_groups)
        non_entity_groups = record.metadata.get("non_entity_value_groups")
        if not isinstance(non_entity_groups, list) or any(
            not isinstance(group, str) or not group.startswith("sha256:")
            for group in non_entity_groups
        ):
            raise TargetedPilotError("generated pilot non-entity value groups are invalid")
        result["entity_value_groups"][split].update(non_entity_groups)
        result["relation_groups"][split].add(record.conversation_group)
        result["text_hashes"][split].add(hashlib.sha256(record.text.encode("utf-8")).hexdigest())
    return result


def _assert_combined_isolation(
    profiles: Mapping[str, SourceSplitProfile], records: Sequence[DocumentRecord]
) -> Mapping[str, Any]:
    generated = _group_sets(records)
    counts: dict[str, Any] = {}
    for field in ("template_groups", "entity_value_groups", "relation_groups", "text_hashes"):
        combined = {
            split: set(getattr(profiles[split], field)) | generated[field][split]
            for split in _SPLITS
        }
        collisions = combined["train"] & combined["validation"]
        if collisions:
            raise TargetedPilotError(f"combined {field} leak across train and validation")
        counts[field] = {
            "train": len(combined["train"]),
            "validation": len(combined["validation"]),
            "cross_split_collisions": 0,
        }
    if any(generated["text_hashes"][split] & profiles[split].text_hashes for split in _SPLITS):
        raise TargetedPilotError("generated pilot copied a D0 source document")
    counts["source_text_copied_count"] = 0
    return counts


def _character_ngrams(value: str, size: int) -> frozenset[str]:
    normalized = "".join(value.casefold().split())
    if len(normalized) < size:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    )


def _negative_duplicate_audit(records: Sequence[DocumentRecord]) -> Mapping[str, Any]:
    texts = [record.text for record in records if not record.entities]
    exact_count = len(texts) - len(set(texts))
    grams = [_character_ngrams(text, _NEAR_DUPLICATE_NGRAM) for text in texts]
    near_count = 0
    for left in range(len(grams)):
        for right in range(left + 1, len(grams)):
            union = grams[left] | grams[right]
            similarity = len(grams[left] & grams[right]) / len(union) if union else 1.0
            if similarity >= _NEAR_DUPLICATE_JACCARD:
                near_count += 1
    if exact_count or near_count:
        raise TargetedPilotError("PII-free hard negatives contain exact or near duplicates")
    return {
        "scope": "pii_free_hard_negative_surface_text",
        "document_count": len(texts),
        "exact_duplicate_pair_count": exact_count,
        "near_duplicate_pair_count": near_count,
        "near_duplicate_method": (f"character_{_NEAR_DUPLICATE_NGRAM}gram_jaccard"),
        "near_duplicate_threshold": _NEAR_DUPLICATE_JACCARD,
    }


def validate_pilot(
    *,
    records: Sequence[DocumentRecord],
    profiles: Mapping[str, SourceSplitProfile],
    recipe: PilotRecipe,
) -> Mapping[str, Any]:
    if len(records) != sum(recipe.counts.values()):
        raise TargetedPilotError("generated pilot count differs from the recipe")
    assert_surface_contract(records)
    by_split = {split: [record for record in records if record.split == split] for split in _SPLITS}
    labels_by_split: dict[str, dict[str, int]] = {}
    pii_free_by_split: dict[str, int] = {}
    hard_negative_categories: dict[str, dict[str, int]] = {}
    chat_by_split: dict[str, int] = {}
    address_boundary_by_split: dict[str, int] = {}
    duplicate_audit_by_split: dict[str, Mapping[str, Any]] = {}
    required_structural = dict(recipe.pii_free_gate["required_structural_negative_categories"])
    minimum_categories = dict(recipe.pii_free_gate["minimum_category_documents"])
    for split, selected in by_split.items():
        if len(selected) != recipe.counts[split]:
            raise TargetedPilotError("generated split count differs from the recipe")
        label_counts = Counter(entity.label for record in selected for entity in record.entities)
        if set(label_counts) != set(CORE_LABELS) or any(
            label_counts[label] < 1 for label in CORE_LABELS
        ):
            raise TargetedPilotError("generated split does not preserve all 24 core labels")
        non_target_max = max(
            count for label, count in label_counts.items() if label not in _TARGET_LABELS
        )
        if any(label_counts[label] <= non_target_max for label in _TARGET_LABELS):
            raise TargetedPilotError("target label augmentation is not stronger than core coverage")
        labels_by_split[split] = dict(sorted(label_counts.items()))
        pii_free = sum(not record.entities for record in selected)
        if not 0.30 <= pii_free / len(selected) <= 0.40:
            raise TargetedPilotError("generated PII-free ratio escaped the required range")
        if pii_free / len(selected) != recipe.pii_free_ratio:
            raise TargetedPilotError("generated PII-free ratio differs from the recipe")
        pii_free_by_split[split] = pii_free
        category_counts = Counter(
            str(record.metadata["hard_negative_category"])
            for record in selected
            if not record.entities
        )
        if set(category_counts) != {category for category, _, _, _ in _HARD_NEGATIVE_SPECS}:
            raise TargetedPilotError("hard-negative category coverage is incomplete")
        for label, category in required_structural.items():
            if category_counts[category] < minimum_categories[split]:
                raise TargetedPilotError(f"structural hard-negative minimum is not met for {label}")
        hard_negative_categories[split] = dict(sorted(category_counts.items()))
        duplicate_audit_by_split[split] = _negative_duplicate_audit(selected)
        chat_count = sum(record.metadata.get("chat_context") is True for record in selected)
        if chat_count / len(selected) < recipe.minimum_chat_context_ratio:
            raise TargetedPilotError("chat-context ratio is below the recipe minimum")
        chat_by_split[split] = chat_count
        boundary_count = sum(
            record.metadata.get("strict_boundary_focus") is True for record in selected
        )
        if boundary_count != label_counts["ADDRESS"]:
            raise TargetedPilotError("ADDRESS strict-boundary metadata is incomplete")
        address_boundary_by_split[split] = boundary_count
        for record in selected:
            if (
                record.data_pool is not DataPool.PUBLIC_RELEASE
                or record.provenance.synthetic is not True
                or record.metadata.get("synthetic") is not True
                or record.metadata.get("human_authored_context") is not False
                or record.metadata.get("llm_generated_context") is not False
                or record.metadata.get("source_text_copied") is not False
                or record.metadata.get("external_benchmark_used_for_recipe") is not False
                or record.public_weight_training_allowed is not True
            ):
                raise TargetedPilotError("generated record provenance escaped the synthetic pilot")
    isolation = _assert_combined_isolation(profiles, records)
    return {
        "counts": {
            "total": len(records),
            "by_split": {split: len(by_split[split]) for split in _SPLITS},
            "pii_free_by_split": pii_free_by_split,
            "pii_free_total": sum(pii_free_by_split.values()),
        },
        "ratios": {
            "pii_free_by_split": {
                split: pii_free_by_split[split] / len(by_split[split]) for split in _SPLITS
            },
            "chat_context_by_split": {
                split: chat_by_split[split] / len(by_split[split]) for split in _SPLITS
            },
        },
        "coverage": {
            "labels_by_split": labels_by_split,
            "required_core_label_count": len(CORE_LABELS),
            "target_labels": sorted(_TARGET_LABELS),
            "address_strict_boundary_documents_by_split": address_boundary_by_split,
            "hard_negative_categories_by_split": hard_negative_categories,
            "required_structural_negative_categories": required_structural,
        },
        "isolation": isolation,
        "pii_free_gate": {
            "data_ratio_minimum": recipe.pii_free_gate["minimum_data_ratio"],
            "data_ratio_maximum": recipe.pii_free_gate["maximum_data_ratio"],
            "data_ratio_passed": True,
            "required_structural_categories_passed": True,
            "duplicate_audit_by_split": duplicate_audit_by_split,
            "model_document_fpr_maximum": recipe.pii_free_gate["model_document_fpr_maximum"],
            "model_measurement_status": "pending_no_gpu_training_or_inference_run",
        },
    }


def build_targeted_pilot(
    *, train_path: Path, validation_path: Path, recipe_path: Path = DEFAULT_RECIPE_PATH
) -> PilotBuild:
    recipe = load_pilot_recipe(recipe_path)
    profiles = profile_d0_sources(
        train_path=train_path, validation_path=validation_path, recipe=recipe
    )
    records = generate_pilot_records(recipe)
    validation = validate_pilot(records=records, profiles=profiles, recipe=recipe)
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "artifact_type": "synthetic_targeted_augmentation_v2_pilot",
        "dataset_id": PILOT_DATASET_ID,
        "dataset_version": PILOT_DATASET_VERSION,
        "release_status": "experimental_local_pilot_not_release_approved",
        "data_pool": DataPool.PUBLIC_RELEASE.value,
        "public_weight_training_allowed": True,
        "recipe": {
            "id": recipe.recipe_id,
            "version": recipe.recipe_version,
            "sha256": recipe.sha256,
            "seed": recipe.seed,
            "selection_basis": list(recipe.generation_contract["selection_basis"]),
            "external_benchmark_used": False,
        },
        "optimization_signal": {
            "source": recipe.validation_only_signal["source"],
            "external_benchmark": False,
            "document_count": recipe.validation_only_signal["document_count"],
            "false_positive_document_count": recipe.validation_only_signal[
                "false_positive_document_count"
            ],
            "pii_free_document_fpr": recipe.validation_only_signal["pii_free_document_fpr"],
            "concentrated_false_positive_labels": dict(
                recipe.validation_only_signal["concentrated_false_positive_labels"]
            ),
            "used_only_for_training_recipe_design": True,
        },
        "source_contract": {
            "source_dataset_role": "D0_synthetic_train_validation_profile_only",
            "source_splits_read": list(_SPLITS),
            "test_split_read": False,
            "evaluation_sources_read": [],
            "source_text_copied_count": 0,
            "inputs": [
                {
                    "split": split,
                    "sha256": profiles[split].sha256,
                    "bytes": profiles[split].size_bytes,
                    "records": profiles[split].record_count,
                    "pii_free_records": profiles[split].pii_free_count,
                    "label_counts": dict(profiles[split].label_counts),
                    "path_recorded": False,
                }
                for split in _SPLITS
            ],
        },
        "generation": {
            "generator_name": PILOT_GENERATOR_NAME,
            "generator_version": PILOT_GENERATOR_VERSION,
            "materializer_version": PILOT_MATERIALIZER_VERSION,
            "context_origin": recipe.generation_contract["context_origin"],
            "synthetic": True,
            "human_authored_context": False,
            "llm_generated_context": False,
            "source_text_copied": False,
            "output_splits": list(_SPLITS),
        },
        **validation,
        "risks": {
            "human_context_claim_allowed": False,
            "benchmark_improvement_not_measured": True,
            "gpu_training_run": False,
            "release_gate_completed": False,
            "synthetic_to_real_domain_gap_remains": True,
        },
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    return PilotBuild(records=records, manifest=MappingProxyType(manifest))


def materialize_targeted_pilot(build: PilotBuild, output_dir: Path) -> Mapping[str, Any]:
    """Write a validated two-split pilot atomically to a new directory."""

    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise TargetedPilotError("pilot output directory must be a new absolute path")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise TargetedPilotError("pilot output parent failed validation")
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        files = []
        for split in _SPLITS:
            path = stage / f"{split}.jsonl"
            selected = [record for record in build.records if record.split == split]
            written = write_jsonl(selected, path)
            path.chmod(0o444)
            verified = list(iter_jsonl(path))
            if written != len(selected) or len(verified) != len(selected):
                raise TargetedPilotError("pilot split readback count failed")
            files.append(
                {
                    "name": path.name,
                    "split": split,
                    "records": written,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mode": "0444",
                }
            )
        manifest = dict(build.manifest)
        manifest["files"] = files
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = _canonical_hash(manifest)
        manifest_path = stage / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o444)
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


def validate_materialized_targeted_pilot(
    *,
    pilot_dir: Path,
    train_path: Path,
    validation_path: Path,
    recipe_path: Path = DEFAULT_RECIPE_PATH,
) -> Mapping[str, Any]:
    """Revalidate a materialized pilot against D0 train/validation and its recipe."""

    if pilot_dir.is_symlink() or not pilot_dir.is_dir():
        raise TargetedPilotError("pilot directory must be a regular directory")
    expected_names = {"train.jsonl", "validation.jsonl", "dataset_manifest.json"}
    observed_names = {path.name for path in pilot_dir.iterdir()}
    if observed_names != expected_names or any(path.is_symlink() for path in pilot_dir.iterdir()):
        raise TargetedPilotError("pilot directory file set failed validation")
    recipe = load_pilot_recipe(recipe_path)
    profiles = profile_d0_sources(
        train_path=train_path,
        validation_path=validation_path,
        recipe=recipe,
    )
    records: list[DocumentRecord] = []
    file_contracts = []
    for split in _SPLITS:
        path = pilot_dir / f"{split}.jsonl"
        selected = list(iter_jsonl(path))
        if any(record.split != split for record in selected):
            raise TargetedPilotError("materialized pilot record escaped its split file")
        records.extend(selected)
        file_contracts.append(
            {
                "name": path.name,
                "split": split,
                "records": len(selected),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mode": "0444",
            }
        )
    recomputed = validate_pilot(records=records, profiles=profiles, recipe=recipe)
    manifest_path = pilot_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedPilotError("pilot manifest failed JSON validation") from exc
    if not isinstance(manifest, dict):
        raise TargetedPilotError("pilot manifest must contain an object")
    claimed_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed_hash != _canonical_hash(unsigned):
        raise TargetedPilotError("pilot manifest logical hash failed validation")
    if (
        manifest.get("dataset_id") != PILOT_DATASET_ID
        or manifest.get("dataset_version") != PILOT_DATASET_VERSION
        or manifest.get("files") != file_contracts
        or manifest.get("recipe", {}).get("sha256") != recipe.sha256
        or manifest.get("source_contract", {}).get("source_splits_read") != list(_SPLITS)
        or manifest.get("source_contract", {}).get("test_split_read") is not False
        or manifest.get("source_contract", {}).get("evaluation_sources_read") != []
        or manifest.get("generation", {}).get("synthetic") is not True
        or manifest.get("generation", {}).get("human_authored_context") is not False
        or manifest.get("generation", {}).get("llm_generated_context") is not False
    ):
        raise TargetedPilotError("pilot manifest identity or provenance failed validation")
    for section in ("counts", "ratios", "coverage", "isolation", "pii_free_gate"):
        if manifest.get(section) != recomputed[section]:
            raise TargetedPilotError(f"pilot manifest {section} differs from recomputation")
    expected_sources = {
        split: (profiles[split].sha256, profiles[split].record_count) for split in _SPLITS
    }
    manifest_sources = manifest["source_contract"].get("inputs")
    if (
        not isinstance(manifest_sources, list)
        or {
            item.get("split"): (item.get("sha256"), item.get("records"))
            for item in manifest_sources
            if isinstance(item, Mapping)
        }
        != expected_sources
    ):
        raise TargetedPilotError("pilot manifest D0 input bindings changed")
    strings: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, Mapping):
            for key, child in value.items():
                strings.append(str(key))
                collect(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                collect(child)

    collect(manifest)
    if any(value.startswith("/") for value in strings):
        raise TargetedPilotError("pilot manifest contains a local absolute path")
    return MappingProxyType(
        {
            "status": "passed",
            "manifest_sha256": claimed_hash,
            "record_count": len(records),
            "source_splits_read": list(_SPLITS),
            "test_split_read": False,
            "evaluation_sources_read": [],
            "pii_free_gate": recomputed["pii_free_gate"],
        }
    )


__all__ = [
    "DEFAULT_RECIPE_PATH",
    "PILOT_DATASET_ID",
    "PILOT_DATASET_VERSION",
    "PILOT_GENERATOR_NAME",
    "PILOT_GENERATOR_VERSION",
    "PILOT_MATERIALIZER_VERSION",
    "PilotBuild",
    "PilotRecipe",
    "SourceSplitProfile",
    "TargetedPilotError",
    "build_targeted_pilot",
    "generate_pilot_records",
    "load_pilot_recipe",
    "materialize_targeted_pilot",
    "profile_d0_sources",
    "profile_d0_split",
    "validate_pilot",
    "validate_materialized_targeted_pilot",
]
