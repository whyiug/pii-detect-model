"""Independent deterministic D0-development repair dataset v3.

This generator consumes a frozen D0 aggregate report, an immutable abstract
diagnostic, and repository-authored v3 templates.  It has no code path for v2
rows/templates/values/model state, test/hidden data, external benchmarks, or
human-context sources.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ...targeted_repair_v3_closure import DATA_IMPLEMENTATION_PATHS
from ..schema import (
    DataPool,
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    iter_jsonl,
    stable_text_hash,
    write_jsonl,
)
from ..validators import (
    cn_resident_id_check_code,
    luhn_check_digit,
    validate_cn_resident_id,
    validate_entity_value,
    validate_luhn,
)
from .targeted_repair_v3_freeze import (
    TargetedRepairV3FreezeError,
    verify_targeted_repair_v3_freeze,
)
from .values import LEGACY_VALUE_VARIANT, SyntheticValueFactory, generate_value

DATASET_ID = "pii_zh_synthetic_targeted_repair_v3"
DATASET_VERSION = "0.3.0"
GENERATOR_NAME = "pii_zh_targeted_repair_v3"
GENERATOR_VERSION = "0.3.0"
MATERIALIZER_VERSION = "0.3.0"
DEFAULT_LICENSE = "Apache-2.0"
DEFAULT_RECIPE_PATH = Path(__file__).resolve().parents[4] / "configs/data/targeted_repair_v3.yaml"
DEFAULT_FREEZE_RECEIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "reports/data_recipe_audits/targeted_repair_v3_implementation_freeze_v3.json"
)
_SPLITS = ("train", "validation")
_NGRAM_SIZE = 5
_PLACEHOLDER_RE = re.compile(r"<<([A-Z0-9_]+)>>")
_SYNTHETIC_CARD_IIN = "999999"
_SYNTHETIC_ID_REGION_PREFIX = "990000"
_SURFACE_SHORTCUTS = ("test", "syn", "测试", "示例", "合成", "虚构", "演练")
_EXPLICIT_NEGATIVE_CUES = (
    "不得",
    "无效",
    "失败",
    "错误",
    "拒绝",
    "未通过",
    "未被采纳",
    "异常区",
    "异常清单",
    "校验",
)

CORE_LABELS: tuple[str, ...] = (
    "PERSON_NAME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "VEHICLE_LICENSE_PLATE",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER",
    "WECHAT_ID",
    "QQ_NUMBER",
    "ALIPAY_ACCOUNT",
    "USERNAME",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "DEVICE_ID",
    "GEO_COORDINATE",
    "SECRET",
)

_LABEL_DISPLAY = MappingProxyType(
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


@dataclass(frozen=True, slots=True)
class _CategorySpec:
    generator_key: str
    structural_role: str
    label_family: str | None
    context_style: str


_CATEGORY_SPECS = MappingProxyType(
    {
        "card_invalid_workflow": _CategorySpec(
            "INVALID_CARD", "invalid_card_negative_only", "BANK_CARD_NUMBER", "explicit"
        ),
        "resident_id_invalid_workflow": _CategorySpec(
            "INVALID_CN_ID", "invalid_id_negative_only", "CN_RESIDENT_ID", "explicit"
        ),
        "card_invalid_natural": _CategorySpec(
            "INVALID_CARD", "invalid_card_negative_only", "BANK_CARD_NUMBER", "natural"
        ),
        "card_valid_counterfactual": _CategorySpec(
            "BANK_CARD_NUMBER", "valid_card_pair", "BANK_CARD_NUMBER", "natural"
        ),
        "resident_id_invalid_natural": _CategorySpec(
            "INVALID_CN_ID", "invalid_id_negative_only", "CN_RESIDENT_ID", "natural"
        ),
        "resident_id_valid_counterfactual": _CategorySpec(
            "CN_RESIDENT_ID", "valid_id_pair", "CN_RESIDENT_ID", "natural"
        ),
        "date_business_guard": _CategorySpec(
            "GENERIC_DATE", "date_guard", "DATE_OF_BIRTH", "natural"
        ),
        "organization": _CategorySpec("ORGANIZATION", "generic_negative", None, "natural"),
        "order_reference": _CategorySpec("ORDER_ID", "generic_negative", None, "natural"),
    }
)

_EXPECTED_NEGATIVE_ALLOCATION = MappingProxyType(
    {
        "train": MappingProxyType(
            {
                "card_invalid_workflow": 160,
                "resident_id_invalid_workflow": 80,
                "card_invalid_natural": 464,
                "card_valid_counterfactual": 96,
                "resident_id_invalid_natural": 256,
                "resident_id_valid_counterfactual": 64,
                "date_business_guard": 160,
                "organization": 32,
                "order_reference": 32,
            }
        ),
        "validation": MappingProxyType(
            {
                "card_invalid_workflow": 40,
                "resident_id_invalid_workflow": 20,
                "card_invalid_natural": 116,
                "card_valid_counterfactual": 24,
                "resident_id_invalid_natural": 64,
                "resident_id_valid_counterfactual": 16,
                "date_business_guard": 40,
                "organization": 8,
                "order_reference": 8,
            }
        ),
    }
)

_EXPECTED_PAIRS = MappingProxyType(
    {
        "train": MappingProxyType(
            {"BANK_CARD_NUMBER": 96, "CN_RESIDENT_ID": 64, "DATE_OF_BIRTH": 104}
        ),
        "validation": MappingProxyType(
            {"BANK_CARD_NUMBER": 24, "CN_RESIDENT_ID": 16, "DATE_OF_BIRTH": 26}
        ),
    }
)


class TargetedRepairV3Error(ValueError):
    """Fail-closed error that never reflects document text or generated values."""


@dataclass(frozen=True, slots=True)
class _Placeholder:
    key: str
    generator_key: str
    label: str | None
    value_variant: str = LEGACY_VALUE_VARIANT


@dataclass(frozen=True, slots=True)
class _Template:
    template_id: str
    family: str
    body: str
    category: str | None
    label: str | None
    context_style: str | None
    placeholders: tuple[_Placeholder, ...]


@dataclass(frozen=True, slots=True)
class _Plan:
    template: _Template
    surface_role: str
    surface_index: int | None


@dataclass(frozen=True, slots=True)
class _Rendered:
    text: str
    entities: tuple[EntitySpan, ...]
    values: Mapping[str, str]
    spans: Mapping[str, tuple[int, int]]
    validators_passed: bool
    validator_results: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RepairRecipe:
    recipe_id: str
    recipe_version: str
    status: str
    seed: int
    counts: Mapping[str, int]
    pii_free_ratio: float
    negative_allocation: Mapping[str, Mapping[str, int]]
    positive_per_label: Mapping[str, int]
    negative_contexts: Mapping[str, Mapping[str, tuple[str, ...]]]
    generic_positive_contexts: Mapping[str, str]
    allowed_aggregate_signals: Mapping[str, Any]
    repair_signal_derivation: Mapping[str, Any]
    retired_v2_boundary: Mapping[str, Any]
    structural_contract: Mapping[str, Any]
    validation_gates: Mapping[str, Any]
    external_benchmark_influence: Mapping[str, Any]
    ancestry: Mapping[str, Any]
    generation_contract: Mapping[str, Any]
    implementation_bindings: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True, slots=True)
class RepairBuild:
    records: tuple[DocumentRecord, ...]
    manifest: Mapping[str, Any]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TargetedRepairV3Error(f"{field} must be a string-keyed object")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TargetedRepairV3Error(f"{field} must be an integer >= {minimum}")
    return value


def _template_array(
    value: object, *, field: str, required_placeholders: frozenset[str]
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TargetedRepairV3Error(f"{field} must be a non-empty string array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise TargetedRepairV3Error(f"{field} contains an invalid template")
        found = _PLACEHOLDER_RE.findall(item)
        if set(found) != set(required_placeholders) or any(found.count(key) != 1 for key in found):
            raise TargetedRepairV3Error(f"{field} template placeholder contract changed")
        result.append(item)
    if len(result) != len(set(result)):
        raise TargetedRepairV3Error(f"{field} contains duplicate templates")
    return tuple(result)


def load_recipe(path: Path = DEFAULT_RECIPE_PATH) -> RepairRecipe:
    if path.is_symlink() or not path.is_file():
        raise TargetedRepairV3Error("recipe must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode())
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TargetedRepairV3Error("recipe is invalid YAML") from exc
    root = _mapping(document, field="recipe")
    if (
        root.get("schema_version") != 1
        or root.get("recipe_id") != DATASET_ID
        or root.get("recipe_version") != DATASET_VERSION
        or root.get("status") not in {"draft", "preregistered_frozen"}
    ):
        raise TargetedRepairV3Error("recipe identity is unsupported")
    counts_raw = _mapping(root.get("counts"), field="counts")
    counts = {split: _integer(counts_raw.get(split), field=f"counts.{split}") for split in _SPLITS}
    if counts != {"train": 3840, "validation": 960}:
        raise TargetedRepairV3Error("v3 split counts changed")
    if root.get("pii_free_ratio") != 0.35:
        raise TargetedRepairV3Error("v3 PII-free ratio changed")

    negative_raw = _mapping(root.get("negative_allocation"), field="negative_allocation")
    negative_allocation: dict[str, Mapping[str, int]] = {}
    for split in _SPLITS:
        allocation = _mapping(negative_raw.get(split), field=f"negative_allocation.{split}")
        parsed = {
            key: _integer(value, field=f"negative_allocation.{split}.{key}")
            for key, value in allocation.items()
        }
        if parsed != dict(_EXPECTED_NEGATIVE_ALLOCATION[split]):
            raise TargetedRepairV3Error("v3 negative allocation changed")
        if sum(parsed.values()) != int(counts[split] * 0.35):
            raise TargetedRepairV3Error("v3 negative allocation does not fill its ratio")
        negative_allocation[split] = MappingProxyType(parsed)

    positive_raw = _mapping(root.get("positive_allocation"), field="positive_allocation")
    if positive_raw.get("policy") != "exact_equal_all_24_core_labels":
        raise TargetedRepairV3Error("positive allocation is not neutral")
    positive_per_label = {
        split: _integer(
            _mapping(positive_raw.get(split), field=f"positive.{split}").get("per_core_label"),
            field=f"positive.{split}.per_core_label",
            minimum=1,
        )
        for split in _SPLITS
    }
    if positive_per_label != {"train": 104, "validation": 26}:
        raise TargetedRepairV3Error("v3 positive allocation changed")
    for split in _SPLITS:
        if (
            positive_per_label[split] * len(CORE_LABELS) + sum(negative_allocation[split].values())
            != counts[split]
        ):
            raise TargetedRepairV3Error("v3 allocation does not fill the split")

    styles = _mapping(root.get("negative_context_styles"), field="negative_context_styles")
    expected_styles = {
        category: (
            "explicit_verification_workflow"
            if spec.context_style == "explicit"
            else "natural_business_context"
        )
        for category, spec in _CATEGORY_SPECS.items()
    }
    if dict(styles) != expected_styles:
        raise TargetedRepairV3Error("negative context style inventory changed")
    contexts_raw = _mapping(root.get("negative_contexts"), field="negative_contexts")
    if set(contexts_raw) != set(_CATEGORY_SPECS):
        raise TargetedRepairV3Error("negative context inventory changed")
    negative_contexts: dict[str, Mapping[str, tuple[str, ...]]] = {}
    all_train_templates: set[str] = set()
    all_validation_templates: set[str] = set()
    for category in _CATEGORY_SPECS:
        by_split = _mapping(contexts_raw[category], field=f"negative_contexts.{category}")
        if set(by_split) != set(_SPLITS):
            raise TargetedRepairV3Error("negative context split inventory changed")
        parsed = {
            split: _template_array(
                by_split[split],
                field=f"negative_contexts.{category}.{split}",
                required_placeholders=frozenset({"CANDIDATE", "CASE"}),
            )
            for split in _SPLITS
        }
        all_train_templates.update(parsed["train"])
        all_validation_templates.update(parsed["validation"])
        negative_contexts[category] = MappingProxyType(parsed)
    if all_train_templates & all_validation_templates:
        raise TargetedRepairV3Error("semantic negative templates collide across splits")

    generic_raw = _mapping(root.get("generic_positive_contexts"), field="positive contexts")
    generic_positive_contexts = {
        split: _template_array(
            [generic_raw.get(split)],
            field=f"generic_positive_contexts.{split}",
            required_placeholders=frozenset({"FIELD", "ENTITY", "CASE"}),
        )[0]
        for split in _SPLITS
    }
    if generic_positive_contexts["train"] == generic_positive_contexts["validation"]:
        raise TargetedRepairV3Error("positive semantic templates collide across splits")

    signals = _mapping(root.get("allowed_aggregate_signals"), field="aggregate signals")
    if set(signals) != {"D0_validation_report", "immutable_abstract_diagnostic"}:
        raise TargetedRepairV3Error("v3 signal inventory changed")
    d0_signal = _mapping(signals["D0_validation_report"], field="D0 signal")
    diagnostic_signal = _mapping(
        signals["immutable_abstract_diagnostic"], field="diagnostic signal"
    )
    if (
        d0_signal.get("file_sha256")
        != "67aa55bc3c4b60d4adfe1009cfe0c5d6b0670b0536259e47f44e76bb1f8f33f6"
        or d0_signal.get("report_sha256")
        != "1959a96356f0524b33e375d1ba89adba78d1e5f58349e16bf4f929f5bb7eb357"
        or diagnostic_signal.get("file_sha256")
        != "d18300224841ebd8e960e6a394fa3de272f78a469af08d3ad2fe304ce7f9bdb7"
        or diagnostic_signal.get("report_sha256")
        != "d22efb766ed0a59c66cb7256f862f1f99cd3e26d6d036c7a2e2e5ee4178ce64f"
    ):
        raise TargetedRepairV3Error("v3 signal hash binding changed")

    structural = _mapping(root.get("structural_contract"), field="structural contract")
    pairs = _mapping(
        structural.get("valid_exact_surface_counterfactual_pairs"), field="pair contract"
    )
    for label, category in (
        ("BANK_CARD_NUMBER", "card_valid_counterfactual"),
        ("CN_RESIDENT_ID", "resident_id_valid_counterfactual"),
        ("DATE_OF_BIRTH", "date_business_guard"),
    ):
        contract = _mapping(pairs.get(label), field=f"pair contract {label}")
        if (
            contract.get("negative_category") != category
            or contract.get("train_pairs") != _EXPECTED_PAIRS["train"][label]
            or contract.get("validation_pairs") != _EXPECTED_PAIRS["validation"][label]
        ):
            raise TargetedRepairV3Error("counterfactual pair contract changed")
    invalid = _mapping(
        structural.get("invalid_checksum_negative_only"), field="invalid checksum contract"
    )
    expected_invalid = {
        "BANK_CARD_NUMBER": (624, 156),
        "CN_RESIDENT_ID": (336, 84),
    }
    for label, expected in expected_invalid.items():
        contract = _mapping(invalid.get(label), field=f"invalid checksum {label}")
        if (
            contract.get("train_documents") != expected[0]
            or contract.get("validation_documents") != expected[1]
            or contract.get("exact_positive_surface_pairs") != 0
        ):
            raise TargetedRepairV3Error("invalid checksum count contract changed")
    if structural.get("candidate_occurrences_per_negative_document") != 1:
        raise TargetedRepairV3Error("single-candidate structural contract changed")

    boundary = _mapping(root.get("retired_v2_boundary"), field="retired v2 boundary")
    receipt = _mapping(boundary.get("retirement_receipt"), field="retirement receipt")
    denylist = _mapping(boundary.get("generator_defect_denylist"), field="defect denylist")
    if (
        receipt.get("file_sha256")
        != "d036b4c16712f402caf058be98d7ccc863f9aba53423fc2c560feceab6d49d3d"
        or receipt.get("receipt_sha256")
        != "390e2fd8e7a948f269da977252dbe7ddc8e23f8127169b0bcbfa771b6ccf3b30"
        or denylist.get("file_sha256")
        != "1484e64a422dc1acea84d971d2fcc13b182fa80108d96b18e146f2c423983cd8"
        or denylist.get("denylist_id") != "pii_zh_generator_defect_artifacts_v1"
        or any(
            boundary.get(key) is not False
            for key in (
                "v2_training_eligible",
                "v2_release_eligible",
                "v2_confirmatory_evaluation_eligible",
                "v2_ancestry_eligible",
            )
        )
        or boundary.get("v3_uses_abstract_diagnostic_only") is not True
    ):
        raise TargetedRepairV3Error("retired v2 boundary changed")

    influence = _mapping(root.get("external_benchmark_influence"), field="benchmark influence")
    expected_influence = {
        "designer_has_prior_PII_Bench_result_exposure": True,
        "direct_bound_PII_Bench_signal_used": False,
        "PII_Bench_examples_used": False,
        "PII_Bench_confirmatory_eligible": False,
        "new_unseen_v3_hidden_eligible": True,
        "allowed_future_confirmatory_benchmark": "new_unseen_v3_hidden_only",
    }
    if dict(influence) != expected_influence:
        raise TargetedRepairV3Error("external benchmark influence contract changed")

    ancestry = _mapping(root.get("ancestry"), field="ancestry")
    if (
        ancestry.get("allowed_parent_artifacts")
        != [
            "D0_validation_aggregate_report",
            "immutable_abstract_v2_dispatch_and_D0_gap_diagnostic",
            "repository_authored_targeted_repair_v3_recipe",
        ]
        or ancestry.get("repair_v1_recipe_report_data_model_ancestry") is not False
        or ancestry.get("v2_row_template_value_recipe_model_ancestry") is not False
        or ancestry.get("artifact_test_informed_ancestry") is not False
        or ancestry.get("generator_defect_ancestry") is not False
        or ancestry.get("designer_external_benchmark_exposure_recorded") is not True
    ):
        raise TargetedRepairV3Error("v3 ancestry contract changed")
    generation = _mapping(root.get("generation_contract"), field="generation contract")
    expected_generation_flags = {
        "synthetic": True,
        "human_authored_context": False,
        "llm_generated_context": False,
        "source_text_copied": False,
        "D0_row_level_data_read_during_generation": False,
        "D0_abstract_diagnostic_bound": True,
        "v2_row_level_data_read_during_generation": False,
        "v2_aggregate_report_bound": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "external_benchmark_read": False,
        "direct_bound_PII_Bench_signal_used": False,
        "PII_Bench_confirmatory_eligible": False,
        "new_unseen_v3_hidden_eligible": True,
        "repair_v1_ancestry": False,
        "v2_row_template_value_recipe_model_ancestry": False,
        "generator_defect_ancestry": False,
        "human_isolation_source_read": False,
    }
    if (
        generation.get("context_origin")
        != "deterministic_repository_authored_v3_synthetic_template"
        or generation.get("allowed_output_splits") != ["train", "validation"]
        or any(generation.get(key) is not value for key, value in expected_generation_flags.items())
    ):
        raise TargetedRepairV3Error("v3 generation safety contract changed")

    gates = _mapping(root.get("validation_gates"), field="validation gates")
    if (
        gates.get("required_core_label_count") != 24
        or gates.get("char_5gram_near_duplicate_threshold") != 0.92
        or gates.get("minimum_natural_business_negative_ratio") != 0.80
        or gates.get("maximum_explicit_workflow_negative_ratio") != 0.20
        or gates.get("structural_candidate_occurrences_per_negative_document") != 1
        or gates.get("invalid_checksum_negative_exact_positive_surface_pairs") != 0
        or gates.get("independent_rematerialization_byte_identical") is not True
    ):
        raise TargetedRepairV3Error("v3 validation gates changed")

    freeze_contract = _mapping(root.get("freeze_contract"), field="freeze contract")
    if freeze_contract != {
        "receipt_id": "targeted_repair_v3_implementation_freeze_v3",
        "repository_relative_path": (
            "reports/data_recipe_audits/targeted_repair_v3_implementation_freeze_v3.json"
        ),
        "anchor_module_bound_by_experiment_preregistration": True,
    }:
        raise TargetedRepairV3Error("v3 freeze contract changed")
    bindings = _mapping(root.get("implementation_bindings"), field="implementation bindings")
    expected_binding_paths = DATA_IMPLEMENTATION_PATHS
    if set(bindings) != set(expected_binding_paths):
        raise TargetedRepairV3Error("implementation binding inventory changed")
    for name, relative in expected_binding_paths.items():
        binding = _mapping(bindings[name], field=f"implementation binding {name}")
        path = Path(__file__).resolve().parents[4] / relative
        if binding.get("repository_relative_path") != relative or binding.get(
            "file_sha256"
        ) != sha256_file(path):
            raise TargetedRepairV3Error("frozen implementation binding changed")

    seed = _integer(root.get("seed"), field="seed")
    if seed != 20260714:
        raise TargetedRepairV3Error("v3 seed changed")
    return RepairRecipe(
        recipe_id=str(root["recipe_id"]),
        recipe_version=str(root["recipe_version"]),
        status=str(root["status"]),
        seed=seed,
        counts=MappingProxyType(counts),
        pii_free_ratio=0.35,
        negative_allocation=MappingProxyType(negative_allocation),
        positive_per_label=MappingProxyType(positive_per_label),
        negative_contexts=MappingProxyType(negative_contexts),
        generic_positive_contexts=MappingProxyType(generic_positive_contexts),
        allowed_aggregate_signals=MappingProxyType(dict(signals)),
        repair_signal_derivation=MappingProxyType(
            dict(_mapping(root.get("repair_signal_derivation"), field="repair derivation"))
        ),
        retired_v2_boundary=MappingProxyType(dict(boundary)),
        structural_contract=MappingProxyType(dict(structural)),
        validation_gates=MappingProxyType(dict(gates)),
        external_benchmark_influence=MappingProxyType(dict(influence)),
        ancestry=MappingProxyType(dict(ancestry)),
        generation_contract=MappingProxyType(dict(generation)),
        implementation_bindings=MappingProxyType(dict(bindings)),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _load_signed_report(
    path: Path, *, expected_file_hash: str, logical_key: str, field: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_file_hash:
        raise TargetedRepairV3Error(f"{field} file hash does not verify")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV3Error(f"{field} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise TargetedRepairV3Error(f"{field} must be an object")
    claimed = value.get(logical_key)
    unsigned = dict(value)
    unsigned.pop(logical_key, None)
    if not isinstance(claimed, str) or _canonical_hash(unsigned) != claimed:
        raise TargetedRepairV3Error(f"{field} logical hash does not verify")
    privacy = _mapping(value.get("privacy"), field=f"{field}.privacy")
    if (
        privacy.get("contains_raw_text") is not False
        or privacy.get("contains_entity_values") is not False
    ):
        raise TargetedRepairV3Error(f"{field} is not aggregate-only")
    return value


def verify_bound_signals(
    recipe: RepairRecipe, *, d0_report_path: Path, diagnostic_path: Path
) -> Mapping[str, Any]:
    d0_contract = _mapping(
        recipe.allowed_aggregate_signals["D0_validation_report"], field="D0 signal"
    )
    diagnostic_contract = _mapping(
        recipe.allowed_aggregate_signals["immutable_abstract_diagnostic"],
        field="diagnostic signal",
    )
    d0 = _load_signed_report(
        d0_report_path,
        expected_file_hash=str(d0_contract["file_sha256"]),
        logical_key="report_sha256",
        field="D0 aggregate report",
    )
    diagnostic = _load_signed_report(
        diagnostic_path,
        expected_file_hash=str(diagnostic_contract["file_sha256"]),
        logical_key="report_sha256",
        field="immutable abstract diagnostic",
    )
    if d0.get("report_sha256") != d0_contract.get("report_sha256") or diagnostic.get(
        "report_sha256"
    ) != diagnostic_contract.get("report_sha256"):
        raise TargetedRepairV3Error("bound aggregate identity changed")
    d0_validation = _mapping(
        _mapping(d0.get("results"), field="D0.results").get("validation"),
        field="D0.validation",
    )
    d0_per_label = _mapping(d0_validation.get("strict_per_label"), field="D0.per_label")
    if (
        _mapping(d0_validation.get("pii_free"), field="D0.pii_free").get("documents") != 600
        or _mapping(d0_validation.get("pii_free"), field="D0.pii_free").get("false_positive_rate")
        != 0.75
        or any(
            d0_per_label[label]["fp"] != 150
            for label in (
                "BANK_CARD_NUMBER",
                "CN_RESIDENT_ID",
                "DATE_OF_BIRTH",
            )
        )
    ):
        raise TargetedRepairV3Error("D0 aggregate signal values changed")
    root_cause = _mapping(diagnostic.get("root_cause"), field="diagnostic.root_cause")
    residual = _mapping(diagnostic.get("v2_residual_error"), field="diagnostic.residual")
    epoch = _mapping(residual.get("epoch_3_and_4"), field="diagnostic.epoch_3_and_4")
    if (
        root_cause.get("classification") != "synthetic_generator_placeholder_dispatch_defect"
        or root_cause.get("D0_mismatch")
        != (
            "Each relevant D0 held-out structural negative contains one candidate occurrence, "
            "while every affected v2 structural negative contains two identical candidate "
            "occurrences."
        )
        or epoch.get("D0_false_positives")
        != {"BANK_CARD_NUMBER": 150, "CN_RESIDENT_ID": 150, "DATE_OF_BIRTH": 0}
        or _mapping(diagnostic.get("v3_lineage_boundary"), field="diagnostic.v3 boundary").get(
            "required_v3_properties"
        )
        is None
    ):
        raise TargetedRepairV3Error("abstract diagnostic contract changed")
    return MappingProxyType(
        {
            "D0_report_sha256": d0["report_sha256"],
            "D0_report_file_sha256": sha256_file(d0_report_path),
            "abstract_diagnostic_sha256": diagnostic["report_sha256"],
            "abstract_diagnostic_file_sha256": sha256_file(diagnostic_path),
            "aggregate_reports_read": 2,
            "D0_row_level_data_read_during_generation": False,
            "v2_row_level_data_read_during_generation": False,
            "v2_aggregate_report_read_during_generation": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "external_benchmark_read": False,
            "human_context_source_read": False,
            "abstract_diagnostic_signal_only": True,
        }
    )


def _family(*, role: str, subtype: str, body: str) -> str:
    digest = hashlib.sha256(body.encode()).hexdigest()[:20]
    return f"repair-v3-{role}-{subtype}-{digest}"


def _positive_template(recipe: RepairRecipe, split: str, label: str) -> _Template:
    body = recipe.generic_positive_contexts[split]
    return _Template(
        template_id=f"repair_v3_{split}_positive_{label.casefold()}",
        family=_family(role="positive", subtype=label.casefold(), body=body),
        body=body,
        category=None,
        label=label,
        context_style=None,
        placeholders=(
            _Placeholder("FIELD", "LITERAL_FIELD", None),
            _Placeholder("ENTITY", label, label),
            _Placeholder("CASE", "BUILD_ID", None),
        ),
    )


def _negative_templates(recipe: RepairRecipe, split: str, category: str) -> tuple[_Template, ...]:
    spec = _CATEGORY_SPECS[category]
    return tuple(
        _Template(
            template_id=f"repair_v3_{split}_negative_{category}_{index:02d}",
            family=_family(role="negative", subtype=category, body=body),
            body=body,
            category=category,
            label=None,
            context_style=spec.context_style,
            placeholders=(
                _Placeholder("CANDIDATE", spec.generator_key, None),
                _Placeholder("CASE", "BUILD_ID", None),
            ),
        )
        for index, body in enumerate(recipe.negative_contexts[category][split], start=1)
    )


def _cycle(templates: Sequence[_Template], count: int, rng: random.Random) -> list[_Template]:
    result: list[_Template] = []
    while len(result) < count:
        batch = list(templates)
        rng.shuffle(batch)
        result.extend(batch[: count - len(result)])
    return result


def _split_seed(seed: int, split: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"repair-v3\0{seed}\0{split}".encode()).digest()[:8], "big"
    )


class _ValueFactory(SyntheticValueFactory):
    """V3-specific structural namespaces with split and role isolation."""

    def __init__(self, rng: random.Random, *, split: str) -> None:
        super().__init__(rng)
        self._split_bit = _SPLITS.index(split)
        self._name_prefix = ("闻舟·", "景澜·")[self._split_bit]
        self._indexed_offsets = {
            "card": rng.randrange(1_000_000_000 // 8),
            "resident": rng.randrange((30 * 365 * 999) // 8),
            "date": rng.randrange((10 * 365) // 8),
        }

    def _indexed(self, key: str, index: int, modulus: int, *, mode: int) -> int:
        if index < 0 or mode not in {0, 1, 2}:
            raise TargetedRepairV3Error("indexed structural value contract is invalid")
        slots = modulus // 8
        code = mode * 2 + self._split_bit
        return 8 * ((self._indexed_offsets[key] + index) % slots) + code

    def person_name(self) -> str:
        return self._name_prefix + self._unique_token("repair_v3_person", 8).upper()

    def build_id(self) -> str:
        serial = 2 * self._unique_number("repair_v3_case", 500_000_000_000) + self._split_bit
        return f"JOB-{serial:012d}"

    def organization(self) -> str:
        return "澄川文献协作组-" + self._unique_token("repair_v3_org", 10).upper()

    def order_id(self) -> str:
        return "REQ-" + self._unique_digits("repair_v3_order", 12)

    def card_surface(self, index: int, *, mode: int, valid: bool) -> str:
        suffix = self._indexed("card", index, 1_000_000_000, mode=mode)
        first_fifteen = _SYNTHETIC_CARD_IIN + f"{suffix:09d}"
        check = luhn_check_digit(first_fifteen)
        if not valid:
            check = "0" if check != "0" else "1"
        return first_fifteen + check

    def resident_surface(self, index: int, *, mode: int, valid: bool) -> str:
        serial = self._indexed("resident", index, 30 * 365 * 999, mode=mode)
        birth_offset, sequence_offset = divmod(serial, 999)
        birth = date(1980, 1, 1) + timedelta(days=birth_offset)
        first_seventeen = (
            _SYNTHETIC_ID_REGION_PREFIX + birth.strftime("%Y%m%d") + f"{sequence_offset + 1:03d}"
        )
        check = cn_resident_id_check_code(first_seventeen)
        if not valid:
            check = "0" if check != "0" else "1"
        return first_seventeen + check

    def date_surface(self, index: int, *, mode: int) -> str:
        offset = self._indexed("date", index, 10 * 365, mode=mode)
        value = date(1960, 1, 1) + timedelta(days=offset)
        style = (offset // 8) % 3
        if style == 0:
            return value.isoformat()
        if style == 1:
            return f"{value.year}年{value.month}月{value.day}日"
        return f"{value.year}/{value.month:02d}/{value.day:02d}"


def _surface_value(factory: _ValueFactory, plan: _Plan, placeholder: _Placeholder) -> str:
    if placeholder.generator_key == "LITERAL_FIELD":
        if plan.template.label is None:
            raise TargetedRepairV3Error("field display is missing its label")
        return _LABEL_DISPLAY[plan.template.label]
    if placeholder.key == "CASE":
        return generate_value(factory, "BUILD_ID")
    index = plan.surface_index
    if plan.surface_role == "valid_card_pair":
        if index is None:
            raise TargetedRepairV3Error("card pair index is missing")
        return factory.card_surface(index, mode=0, valid=True)
    if plan.surface_role == "invalid_card_negative_only":
        if index is None:
            raise TargetedRepairV3Error("invalid card index is missing")
        return factory.card_surface(index, mode=1, valid=False)
    if plan.surface_role == "valid_card_unpaired":
        if index is None:
            raise TargetedRepairV3Error("unpaired card index is missing")
        return factory.card_surface(index, mode=2, valid=True)
    if plan.surface_role == "valid_id_pair":
        if index is None:
            raise TargetedRepairV3Error("resident ID pair index is missing")
        return factory.resident_surface(index, mode=0, valid=True)
    if plan.surface_role == "invalid_id_negative_only":
        if index is None:
            raise TargetedRepairV3Error("invalid resident ID index is missing")
        return factory.resident_surface(index, mode=1, valid=False)
    if plan.surface_role == "valid_id_unpaired":
        if index is None:
            raise TargetedRepairV3Error("unpaired resident ID index is missing")
        return factory.resident_surface(index, mode=2, valid=True)
    if plan.surface_role == "date_pair":
        if index is None:
            raise TargetedRepairV3Error("date pair index is missing")
        return factory.date_surface(index, mode=0)
    if plan.surface_role == "date_guard_unpaired":
        if index is None:
            raise TargetedRepairV3Error("unpaired date index is missing")
        return factory.date_surface(index, mode=1)
    if plan.surface_role == "date_positive_unpaired":
        if index is None:
            raise TargetedRepairV3Error("positive date index is missing")
        return factory.date_surface(index, mode=2)
    return generate_value(
        factory, placeholder.generator_key, value_variant=placeholder.value_variant
    )


def _render(plan: _Plan, factory: _ValueFactory) -> _Rendered:
    template = plan.template
    values = {
        placeholder.key: _surface_value(factory, plan, placeholder)
        for placeholder in template.placeholders
    }
    declarations = {placeholder.key: placeholder for placeholder in template.placeholders}
    output: list[str] = []
    entities: list[EntitySpan] = []
    spans: dict[str, tuple[int, int]] = {}
    validator_results: dict[str, str] = {}
    cursor = 0
    length = 0
    validators_passed = True
    for match in _PLACEHOLDER_RE.finditer(template.body):
        literal = template.body[cursor : match.start()]
        output.append(literal)
        length += len(literal)
        key = match.group(1)
        if key not in declarations or key not in values:
            raise TargetedRepairV3Error("placeholder declaration is incomplete")
        value = values[key]
        start = length
        output.append(value)
        length += len(value)
        spans[key] = (start, length)
        placeholder = declarations[key]
        if placeholder.label is not None:
            result = validate_entity_value(placeholder.label, value)
            state = "not_applicable" if result is None else ("passed" if result.valid else "failed")
            if result is not None:
                validator_results[f"{key}:{placeholder.label}"] = result.reason
                validators_passed = validators_passed and result.valid
            entities.append(
                EntitySpan(
                    start=start,
                    end=length,
                    label=placeholder.label,
                    text=value,
                    source="synthetic",
                    validator_state=state,
                    metadata={"synthetic": True, "placeholder": key},
                )
            )
        cursor = match.end()
    output.append(template.body[cursor:])
    text = "".join(output)
    if set(declarations) != set(_PLACEHOLDER_RE.findall(template.body)):
        raise TargetedRepairV3Error("placeholder inventory changed")
    for entity in entities:
        entity.validate(text)
    if template.category is not None and entities:
        raise TargetedRepairV3Error("negative template emitted an entity")
    return _Rendered(
        text=text,
        entities=tuple(entities),
        values=MappingProxyType(values),
        spans=MappingProxyType(spans),
        validators_passed=validators_passed,
        validator_results=MappingProxyType(validator_results),
    )


def _relation(rendered: _Rendered, plan: _Plan, identity: str) -> tuple[str, str]:
    if plan.surface_role == "valid_card_pair":
        kind = "card_valid_exact_surface"
        material = rendered.values.get("ENTITY") or rendered.values.get("CANDIDATE")
    elif plan.surface_role == "valid_id_pair":
        kind = "resident_id_valid_exact_surface"
        material = rendered.values.get("ENTITY") or rendered.values.get("CANDIDATE")
    elif plan.surface_role == "date_pair":
        kind = "date_exact_surface"
        material = rendered.values.get("ENTITY") or rendered.values.get("CANDIDATE")
    elif plan.surface_role == "invalid_card_negative_only":
        kind = "invalid_card_negative_only"
        material = identity
    elif plan.surface_role == "invalid_id_negative_only":
        kind = "invalid_resident_id_negative_only"
        material = identity
    else:
        kind = "unique"
        material = identity
    if not isinstance(material, str):
        raise TargetedRepairV3Error("relation material is missing")
    digest = hashlib.sha256(f"{kind}\0{material}".encode()).hexdigest()
    return "synthetic-repair-v3-relation:sha256:" + digest, kind


def _record(
    *, plan: _Plan, split: str, index: int, recipe: RepairRecipe, factory: _ValueFactory
) -> DocumentRecord:
    rendered = _render(plan, factory)
    if not rendered.validators_passed:
        raise TargetedRepairV3Error("generated positive value failed validation")
    candidate_span = rendered.spans.get("CANDIDATE")
    case_span = rendered.spans.get("CASE")
    if case_span is None:
        raise TargetedRepairV3Error("every v3 record requires an independent case value")
    candidate_occurrences: int | None = None
    checksum_state: str | None = None
    if candidate_span is not None:
        candidate = rendered.text[candidate_span[0] : candidate_span[1]]
        case_value = rendered.text[case_span[0] : case_span[1]]
        candidate_occurrences = rendered.text.count(candidate)
        if candidate_occurrences != 1 or candidate == case_value:
            raise TargetedRepairV3Error("negative record violates the one-candidate dispatch gate")
        if plan.surface_role == "invalid_card_negative_only":
            if validate_luhn(candidate).valid:
                raise TargetedRepairV3Error("invalid-card negative unexpectedly passed Luhn")
            checksum_state = "invalid_luhn_negative_only"
        elif plan.surface_role == "valid_card_pair":
            if not validate_luhn(candidate).valid:
                raise TargetedRepairV3Error("valid-card counterfactual failed Luhn")
            checksum_state = "valid_luhn_semantic_non_entity"
        elif plan.surface_role == "invalid_id_negative_only":
            if validate_cn_resident_id(candidate).valid:
                raise TargetedRepairV3Error("invalid-ID negative unexpectedly passed checksum")
            checksum_state = "invalid_mod11_2_negative_only"
        elif plan.surface_role == "valid_id_pair":
            if not validate_cn_resident_id(candidate).valid:
                raise TargetedRepairV3Error("valid-ID counterfactual failed checksum")
            checksum_state = "valid_mod11_2_semantic_non_entity"
        else:
            checksum_state = "not_applicable"
    identity = hashlib.sha256(
        f"{DATASET_VERSION}\0{split}\0{index}\0{plan.template.template_id}\0{rendered.text}".encode()
    ).hexdigest()
    relation_group, relation_kind = _relation(rendered, plan, identity)
    non_entity_groups = tuple(
        sorted(
            stable_text_hash(rendered.values[item.key])
            for item in plan.template.placeholders
            if item.label is None and item.generator_key != "LITERAL_FIELD"
        )
    )
    provenance_flags = {
        "contains_real_personal_data": False,
        "context_origin": recipe.generation_contract["context_origin"],
        "human_authored_context": False,
        "llm_generated_context": False,
        "source_text_copied": False,
        "D0_row_level_data_read_during_generation": False,
        "v2_row_level_data_read_during_generation": False,
        "v2_row_template_value_recipe_model_ancestry": False,
        "generator_defect_ancestry": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "external_benchmark_read": False,
        "human_isolation_source_read": False,
        "designer_has_prior_PII_Bench_result_exposure": True,
        "direct_bound_PII_Bench_signal_used": False,
        "PII_Bench_confirmatory_eligible": False,
        "new_unseen_v3_hidden_eligible": True,
        "repair_v1_ancestry": False,
        "recipe_sha256": recipe.sha256,
    }
    provenance = Provenance(
        source_id="repo_targeted_repair_v3",
        source_kind="deterministic_repository_authored_v3_synthetic",
        source_revision=f"sha256:{recipe.sha256}",
        license=DEFAULT_LICENSE,
        synthetic=True,
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=recipe.seed,
        template_id=plan.template.template_id,
        metadata=provenance_flags,
    )
    negative = plan.template.category is not None
    quality = QualityMetadata(
        tier=QualityTier.HARD_NEGATIVE if negative else QualityTier.SYNTHETIC_VALIDATED,
        quality_gate=True,
        validators_passed=True,
        review_status="frozen_v3_recipe_and_deterministic_validators_passed",
        confidence=1.0,
        metadata={"validator_results": dict(rendered.validator_results)},
    )
    metadata = {
        "synthetic": True,
        "targeted_repair_v3": True,
        **provenance_flags,
        "hard_negative": negative,
        "hard_negative_category": plan.template.category,
        "negative_context_style": plan.template.context_style,
        "structural_surface_role": plan.surface_role,
        "checksum_state": checksum_state,
        "candidate_occurrence_count": candidate_occurrences,
        "candidate_span": list(candidate_span) if candidate_span else None,
        "case_span": list(case_span),
        "non_entity_value_groups": list(non_entity_groups),
        "relation_kind": relation_kind,
        "pair_allocation_method": "atomic_same_split_exact_surface_plan_before_shuffle",
        "training_role": "synthetic_targeted_data_repair_v3",
    }
    return DocumentRecord.create(
        doc_id="sha256:" + identity,
        text=rendered.text,
        entities=rendered.entities,
        language="zh-Hans-CN",
        domain=("synthetic_business_operations" if negative else "synthetic_personal_record"),
        scene=(
            f"synthetic_v3_negative_{plan.template.category}"
            if negative
            else "synthetic_v3_positive"
        ),
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        conversation_group=relation_group,
        template_group=f"synthetic-repair-v3-family:{plan.template.family}",
        entity_value_groups=tuple(
            sorted(stable_text_hash(entity.text) for entity in rendered.entities)
        ),
        generator_version=GENERATOR_VERSION,
        split=split,
        metadata=metadata,
    )


def _positive_role(label: str, index: int, pair_count: int) -> tuple[str, int | None]:
    if label == "BANK_CARD_NUMBER":
        return (
            ("valid_card_pair", index)
            if index < pair_count
            else (
                "valid_card_unpaired",
                index - pair_count,
            )
        )
    if label == "CN_RESIDENT_ID":
        return (
            ("valid_id_pair", index)
            if index < pair_count
            else (
                "valid_id_unpaired",
                index - pair_count,
            )
        )
    if label == "DATE_OF_BIRTH":
        return (
            ("date_pair", index)
            if index < pair_count
            else (
                "date_positive_unpaired",
                index - pair_count,
            )
        )
    return "generic_positive", None


def generate_records(recipe: RepairRecipe) -> tuple[DocumentRecord, ...]:
    records: list[DocumentRecord] = []
    for split in _SPLITS:
        rng = random.Random(_split_seed(recipe.seed, split))
        factory = _ValueFactory(rng, split=split)
        plans: list[_Plan] = []
        for label in CORE_LABELS:
            pair_count = int(_EXPECTED_PAIRS[split].get(label, 0))
            template = _positive_template(recipe, split, label)
            for index in range(recipe.positive_per_label[split]):
                role, surface_index = _positive_role(label, index, pair_count)
                plans.append(_Plan(template, role, surface_index))
        invalid_card_index = 0
        invalid_id_index = 0
        for category, count in recipe.negative_allocation[split].items():
            templates = _cycle(_negative_templates(recipe, split, category), count, rng)
            spec = _CATEGORY_SPECS[category]
            for index, template in enumerate(templates):
                role = spec.structural_role
                surface_index: int | None
                if role == "invalid_card_negative_only":
                    surface_index = invalid_card_index
                    invalid_card_index += 1
                elif role == "invalid_id_negative_only":
                    surface_index = invalid_id_index
                    invalid_id_index += 1
                elif role in {"valid_card_pair", "valid_id_pair"}:
                    surface_index = index
                elif role == "date_guard":
                    pair_count = int(_EXPECTED_PAIRS[split]["DATE_OF_BIRTH"])
                    if index < pair_count:
                        role = "date_pair"
                        surface_index = index
                    else:
                        role = "date_guard_unpaired"
                        surface_index = index - pair_count
                else:
                    surface_index = None
                plans.append(_Plan(template, role, surface_index))
        if len(plans) != recipe.counts[split]:
            raise TargetedRepairV3Error("planned records do not fill split")
        rng.shuffle(plans)
        for index, plan in enumerate(plans):
            records.append(
                _record(plan=plan, split=split, index=index, recipe=recipe, factory=factory)
            )
    return tuple(records)


def _ngrams(value: str) -> frozenset[str]:
    normalized = "".join(value.casefold().split())
    if len(normalized) < _NGRAM_SIZE:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + _NGRAM_SIZE]
        for index in range(len(normalized) - _NGRAM_SIZE + 1)
    )


def _duplicate_audit(
    records: Sequence[DocumentRecord], threshold: float, *, scope: str
) -> dict[str, Any]:
    texts = [record.text for record in records]
    counts = Counter(texts)
    exact = sum(count * (count - 1) // 2 for count in counts.values())
    grams = [_ngrams(text) for text in texts]
    inverted: dict[str, list[int]] = {}
    near = 0
    compared = 0
    for right, right_grams in enumerate(grams):
        candidates: set[int] = set()
        for gram in right_grams:
            candidates.update(inverted.get(gram, ()))
        for left in candidates:
            left_grams = grams[left]
            larger = max(len(left_grams), len(right_grams))
            smaller = min(len(left_grams), len(right_grams))
            if larger and smaller / larger < threshold:
                continue
            compared += 1
            intersection = len(left_grams & right_grams)
            union = len(left_grams) + len(right_grams) - intersection
            near += (intersection / union if union else 1.0) >= threshold
        for gram in right_grams:
            inverted.setdefault(gram, []).append(right)
    if exact or near:
        raise TargetedRepairV3Error(f"{scope} contains exact or near duplicates")
    return {
        "scope": scope,
        "document_count": len(records),
        "exact_duplicate_pair_count": exact,
        "near_duplicate_pair_count": near,
        "candidate_pair_count": compared,
        "near_duplicate_method": "character_5gram_jaccard",
        "near_duplicate_threshold": threshold,
    }


def _isolation(records: Sequence[DocumentRecord]) -> dict[str, Any]:
    fields = (
        "document_ids",
        "template_groups",
        "value_groups",
        "relation_groups",
        "text_hashes",
    )
    sets = {field: {split: set() for split in _SPLITS} for field in fields}
    for record in records:
        split = str(record.split)
        sets["document_ids"][split].add(record.doc_id)
        sets["template_groups"][split].add(record.template_group)
        values = set(record.entity_value_groups)
        non_entity = record.metadata.get("non_entity_value_groups")
        if not isinstance(non_entity, list) or not non_entity:
            raise TargetedRepairV3Error("record parameter value groups are missing")
        values.update(non_entity)
        sets["value_groups"][split].update(values)
        sets["relation_groups"][split].add(record.conversation_group)
        sets["text_hashes"][split].add(hashlib.sha256(record.text.encode()).hexdigest())
    result: dict[str, Any] = {}
    for field, values in sets.items():
        if values["train"] & values["validation"]:
            raise TargetedRepairV3Error(f"{field} collide across train and validation")
        result[field] = {
            "train": len(values["train"]),
            "validation": len(values["validation"]),
            "cross_split_collisions": 0,
        }
    return result


def _validate_candidate(record: DocumentRecord) -> str | None:
    span = record.metadata.get("candidate_span")
    if span is None:
        return None
    if (
        not isinstance(span, list)
        or len(span) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in span)
        or not 0 <= span[0] < span[1] <= len(record.text)
    ):
        raise TargetedRepairV3Error("candidate span metadata is invalid")
    candidate = record.text[span[0] : span[1]]
    case_span = record.metadata.get("case_span")
    if not isinstance(case_span, list) or len(case_span) != 2:
        raise TargetedRepairV3Error("case span metadata is invalid")
    case_value = record.text[case_span[0] : case_span[1]]
    if (
        record.text.count(candidate) != 1
        or candidate == case_value
        or record.metadata.get("candidate_occurrence_count") != 1
    ):
        raise TargetedRepairV3Error("materialized negative violates single-candidate dispatch")
    return candidate


def validate_records(records: Sequence[DocumentRecord], recipe: RepairRecipe) -> Mapping[str, Any]:
    if len(records) != sum(recipe.counts.values()):
        raise TargetedRepairV3Error("record count differs from recipe")
    by_split = {split: [record for record in records if record.split == split] for split in _SPLITS}
    label_counts: dict[str, dict[str, int]] = {}
    negative_counts: dict[str, dict[str, int]] = {}
    anti_shortcut: dict[str, Any] = {}
    structural_counts: dict[str, Any] = {}
    positive_value_groups: dict[str, set[str]] = {split: set() for split in _SPLITS}
    seen_positive_values: set[str] = set()
    for record in records:
        record.validate()
        if any(shortcut in record.text.casefold() for shortcut in _SURFACE_SHORTCUTS):
            raise TargetedRepairV3Error("generated text contains a synthetic shortcut")
        for group in record.entity_value_groups:
            if group in seen_positive_values:
                raise TargetedRepairV3Error("positive entity surfaces are not unique")
            seen_positive_values.add(group)
            positive_value_groups[str(record.split)].add(group)
    for split, selected in by_split.items():
        if len(selected) != recipe.counts[split] or len({r.doc_id for r in selected}) != len(
            selected
        ):
            raise TargetedRepairV3Error("split count or document identity is invalid")
        labels = Counter(entity.label for record in selected for entity in record.entities)
        expected_labels = {label: recipe.positive_per_label[split] for label in CORE_LABELS}
        if dict(labels) != expected_labels:
            raise TargetedRepairV3Error("positive label allocation changed")
        label_counts[split] = dict(sorted(labels.items()))
        negatives = [record for record in selected if not record.entities]
        categories = Counter(
            str(record.metadata.get("hard_negative_category")) for record in negatives
        )
        if dict(categories) != dict(recipe.negative_allocation[split]):
            raise TargetedRepairV3Error("negative category allocation changed")
        negative_counts[split] = dict(sorted(categories.items()))
        if len(negatives) / len(selected) != recipe.pii_free_ratio:
            raise TargetedRepairV3Error("PII-free ratio changed")
        explicit = [
            record
            for record in negatives
            if record.metadata.get("negative_context_style") == "explicit"
        ]
        natural = [
            record
            for record in negatives
            if record.metadata.get("negative_context_style") == "natural"
        ]
        if len(explicit) + len(natural) != len(negatives):
            raise TargetedRepairV3Error("negative context style is missing")
        if any(
            not any(cue in record.text for cue in _EXPLICIT_NEGATIVE_CUES) for record in explicit
        ):
            raise TargetedRepairV3Error("explicit workflow lacks a verification/failure cue")
        if any(any(cue in record.text for cue in _EXPLICIT_NEGATIVE_CUES) for record in natural):
            raise TargetedRepairV3Error("natural business context leaks an explicit shortcut cue")
        natural_ratio = len(natural) / len(negatives)
        explicit_ratio = len(explicit) / len(negatives)
        if natural_ratio < 0.80 or explicit_ratio > 0.20:
            raise TargetedRepairV3Error("negative context ratio gate failed")
        anti_shortcut[split] = {
            "negative_documents": len(negatives),
            "natural_business_context_documents": len(natural),
            "explicit_verification_workflow_documents": len(explicit),
            "natural_business_context_ratio": natural_ratio,
            "explicit_verification_workflow_ratio": explicit_ratio,
            "minimum_natural_ratio": 0.80,
            "maximum_explicit_ratio": 0.20,
            "gate_passed": True,
        }
        state_counts: Counter[str] = Counter()
        invalid_groups: set[str] = set()
        for record in selected:
            if (
                record.data_pool is not DataPool.PUBLIC_RELEASE
                or record.public_weight_training_allowed is not True
                or record.provenance.synthetic is not True
                or record.metadata.get("targeted_repair_v3") is not True
                or record.metadata.get("D0_row_level_data_read_during_generation") is not False
                or record.metadata.get("v2_row_level_data_read_during_generation") is not False
                or record.metadata.get("v2_row_template_value_recipe_model_ancestry") is not False
                or record.metadata.get("generator_defect_ancestry") is not False
                or record.metadata.get("test_data_read") is not False
                or record.metadata.get("hidden_data_read") is not False
                or record.metadata.get("external_benchmark_read") is not False
                or record.metadata.get("human_isolation_source_read") is not False
                or record.metadata.get("direct_bound_PII_Bench_signal_used") is not False
                or record.metadata.get("PII_Bench_confirmatory_eligible") is not False
            ):
                raise TargetedRepairV3Error("record provenance escaped the v3 contract")
            candidate = _validate_candidate(record)
            role = str(record.metadata.get("structural_surface_role"))
            if candidate is not None:
                candidate_group = stable_text_hash(candidate)
                if role == "invalid_card_negative_only":
                    if validate_luhn(candidate).valid:
                        raise TargetedRepairV3Error("invalid card materialization passed Luhn")
                    state_counts["invalid_card_negative_only"] += 1
                    invalid_groups.add(candidate_group)
                elif role == "invalid_id_negative_only":
                    if validate_cn_resident_id(candidate).valid:
                        raise TargetedRepairV3Error("invalid resident ID materialization passed")
                    state_counts["invalid_id_negative_only"] += 1
                    invalid_groups.add(candidate_group)
                elif role == "valid_card_pair":
                    if not validate_luhn(candidate).valid:
                        raise TargetedRepairV3Error("valid card counterfactual failed")
                    state_counts["valid_card_counterfactual_negative"] += 1
                elif role == "valid_id_pair":
                    if not validate_cn_resident_id(candidate).valid:
                        raise TargetedRepairV3Error("valid resident ID counterfactual failed")
                    state_counts["valid_id_counterfactual_negative"] += 1
                elif role in {"date_pair", "date_guard_unpaired"}:
                    state_counts["date_business_guard_negative"] += 1
        if invalid_groups & positive_value_groups[split]:
            raise TargetedRepairV3Error("invalid checksum negative shares a positive exact surface")
        expected_states = {
            "invalid_card_negative_only": (
                recipe.negative_allocation[split]["card_invalid_workflow"]
                + recipe.negative_allocation[split]["card_invalid_natural"]
            ),
            "invalid_id_negative_only": (
                recipe.negative_allocation[split]["resident_id_invalid_workflow"]
                + recipe.negative_allocation[split]["resident_id_invalid_natural"]
            ),
            "valid_card_counterfactual_negative": recipe.negative_allocation[split][
                "card_valid_counterfactual"
            ],
            "valid_id_counterfactual_negative": recipe.negative_allocation[split][
                "resident_id_valid_counterfactual"
            ],
            "date_business_guard_negative": recipe.negative_allocation[split][
                "date_business_guard"
            ],
        }
        if dict(state_counts) != expected_states:
            raise TargetedRepairV3Error("structural negative counts changed")
        structural_counts[split] = {
            "invalid_checksum_negative_only": {
                "BANK_CARD_NUMBER": state_counts["invalid_card_negative_only"],
                "CN_RESIDENT_ID": state_counts["invalid_id_negative_only"],
                "exact_positive_surface_pair_count": 0,
            },
            "valid_exact_surface_counterfactual_negative_documents": {
                "BANK_CARD_NUMBER": state_counts["valid_card_counterfactual_negative"],
                "CN_RESIDENT_ID": state_counts["valid_id_counterfactual_negative"],
                "DATE_OF_BIRTH": min(
                    state_counts["date_business_guard_negative"],
                    _EXPECTED_PAIRS[split]["DATE_OF_BIRTH"],
                ),
            },
            "single_candidate_negative_documents": len(negatives),
            "candidate_occurrences_per_negative_document": 1,
        }

    relation_members: dict[str, dict[str, list[DocumentRecord]]] = {split: {} for split in _SPLITS}
    for split, selected in by_split.items():
        for record in selected:
            relation_members[split].setdefault(str(record.conversation_group), []).append(record)
    pair_counts: dict[str, dict[str, int]] = {}
    relation_to_label = {
        "card_valid_exact_surface": "BANK_CARD_NUMBER",
        "resident_id_valid_exact_surface": "CN_RESIDENT_ID",
        "date_exact_surface": "DATE_OF_BIRTH",
    }
    for split in _SPLITS:
        observed = {label: 0 for label in relation_to_label.values()}
        for members in relation_members[split].values():
            kind = members[0].metadata.get("relation_kind")
            if kind not in relation_to_label:
                continue
            if len(members) != 2 or sum(bool(member.entities) for member in members) != 1:
                raise TargetedRepairV3Error("counterfactual relation is not an atomic pair")
            positive = next(member for member in members if member.entities)
            negative = next(member for member in members if not member.entities)
            if not set(positive.entity_value_groups) & set(
                negative.metadata["non_entity_value_groups"]
            ):
                raise TargetedRepairV3Error("counterfactual pair does not share its exact surface")
            observed[relation_to_label[str(kind)]] += 1
        if observed != dict(_EXPECTED_PAIRS[split]):
            raise TargetedRepairV3Error("counterfactual pair count changed")
        pair_counts[split] = observed

    threshold = float(recipe.validation_gates["char_5gram_near_duplicate_threshold"])
    duplicate_audit = {
        "all_records": _duplicate_audit(records, threshold, scope="all_records_cross_split"),
        "negative_by_split": {
            split: _duplicate_audit(
                [record for record in by_split[split] if not record.entities],
                threshold,
                scope=f"negative_{split}",
            )
            for split in _SPLITS
        },
    }
    return MappingProxyType(
        {
            "counts": {
                "total": len(records),
                "by_split": {split: len(by_split[split]) for split in _SPLITS},
                "pii_free_by_split": {
                    split: sum(not record.entities for record in by_split[split])
                    for split in _SPLITS
                },
            },
            "coverage": {
                "required_core_label_count": len(CORE_LABELS),
                "labels_by_split": label_counts,
                "hard_negative_categories_by_split": negative_counts,
                "positive_allocation_policy": "exact_equal_all_24_core_labels",
            },
            "structural_counts": structural_counts,
            "counterfactual_pairing": {
                "method": "atomic_same_split_valid_exact_surface_pairs",
                "pairs_by_split": pair_counts,
            },
            "anti_shortcut": {
                "explicit_negative_cues_retained_in_report": False,
                "by_split": anti_shortcut,
            },
            "duplicate_audit": duplicate_audit,
            "isolation": _isolation(records),
            "parameterized_record_count": len(records),
        }
    )


def build_repair_dataset(
    *,
    recipe_path: Path,
    freeze_receipt_path: Path,
    d0_report_path: Path,
    diagnostic_path: Path,
) -> RepairBuild:
    recipe = load_recipe(recipe_path)
    if recipe.status != "preregistered_frozen":
        raise TargetedRepairV3Error("dataset materialization requires a frozen recipe")
    try:
        freeze_receipt = verify_targeted_repair_v3_freeze(
            receipt_path=freeze_receipt_path, recipe_path=recipe_path
        )
    except TargetedRepairV3FreezeError as exc:
        raise TargetedRepairV3Error("dataset implementation freeze does not verify") from exc
    evidence = verify_bound_signals(
        recipe, d0_report_path=d0_report_path, diagnostic_path=diagnostic_path
    )
    records = generate_records(recipe)
    validation = validate_records(records, recipe)
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "artifact_type": "synthetic_targeted_data_repair_v3",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "release_status": "experimental_local_pilot_not_release_approved",
        "data_pool": DataPool.PUBLIC_RELEASE.value,
        "public_weight_training_allowed": True,
        "recipe": {
            "id": recipe.recipe_id,
            "version": recipe.recipe_version,
            "sha256": recipe.sha256,
            "seed": recipe.seed,
        },
        "freeze_receipt": dict(freeze_receipt),
        "aggregate_signal_sources": dict(evidence),
        "repair_signal_derivation": dict(recipe.repair_signal_derivation),
        "retired_v2_boundary": dict(recipe.retired_v2_boundary),
        "external_benchmark_influence": dict(recipe.external_benchmark_influence),
        "ancestry": dict(recipe.ancestry),
        "source_contract": {
            "aggregate_reports_read": 2,
            "D0_row_level_data_read_during_generation": False,
            "v2_row_level_data_read_during_generation": False,
            "v2_aggregate_report_read_during_generation": False,
            "v2_row_template_value_recipe_model_ancestry": False,
            "generator_defect_ancestry": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "external_benchmark_read": False,
            "human_context_source_read": False,
            "LLM_generation_used": False,
            "abstract_diagnostic_signal_only": True,
        },
        "generation": {
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "context_origin": recipe.generation_contract["context_origin"],
            "synthetic": True,
            "split_independent_rng": True,
            "placeholder_dispatch": "placeholder_key_first_never_category_override",
            "semantic_template_families_disjoint": True,
            "output_splits": list(_SPLITS),
        },
        **dict(validation),
        "risks": {
            "human_context_claim_allowed": False,
            "external_benchmark_improvement_measured": False,
            "release_gate_completed": False,
            "synthetic_to_real_domain_gap_remains": True,
        },
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    return RepairBuild(tuple(records), MappingProxyType(manifest))


def materialize(build: RepairBuild, output_dir: Path) -> Mapping[str, Any]:
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise TargetedRepairV3Error("output directory must be a new absolute directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        files: list[dict[str, Any]] = []
        for split in _SPLITS:
            path = stage / f"{split}.jsonl"
            selected = [record for record in build.records if record.split == split]
            if write_jsonl(selected, path) != len(selected) or len(list(iter_jsonl(path))) != len(
                selected
            ):
                raise TargetedRepairV3Error("materialized split readback failed")
            path.chmod(0o444)
            files.append(
                {
                    "name": path.name,
                    "split": split,
                    "records": len(selected),
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
        stage.chmod(0o700)
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


def _read_materialized(dataset_dir: Path) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
    records: list[DocumentRecord] = []
    files: list[dict[str, Any]] = []
    for split in _SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        selected = list(iter_jsonl(path))
        if any(record.split != split for record in selected):
            raise TargetedRepairV3Error("record escaped its materialized split")
        records.extend(selected)
        files.append(
            {
                "name": path.name,
                "split": split,
                "records": len(selected),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mode": "0444",
            }
        )
    return records, files


def validate_materialized(
    *,
    dataset_dir: Path,
    recipe_path: Path,
    freeze_receipt_path: Path,
    d0_report_path: Path,
    diagnostic_path: Path,
) -> Mapping[str, Any]:
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise TargetedRepairV3Error("dataset directory must be regular")
    expected_names = {"train.jsonl", "validation.jsonl", "dataset_manifest.json"}
    if {path.name for path in dataset_dir.iterdir()} != expected_names or any(
        path.is_symlink() for path in dataset_dir.iterdir()
    ):
        raise TargetedRepairV3Error("dataset file set is invalid")
    recipe = load_recipe(recipe_path)
    try:
        freeze_receipt = verify_targeted_repair_v3_freeze(
            receipt_path=freeze_receipt_path, recipe_path=recipe_path
        )
    except TargetedRepairV3FreezeError as exc:
        raise TargetedRepairV3Error("dataset implementation freeze does not verify") from exc
    evidence = verify_bound_signals(
        recipe, d0_report_path=d0_report_path, diagnostic_path=diagnostic_path
    )
    records, files = _read_materialized(dataset_dir)
    recomputed = validate_records(records, recipe)
    manifest_path = dataset_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV3Error("manifest is not readable JSON") from exc
    if not isinstance(manifest, dict):
        raise TargetedRepairV3Error("manifest must be an object")
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_hash(unsigned):
        raise TargetedRepairV3Error("manifest logical hash failed")
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("files") != files
        or manifest.get("recipe", {}).get("sha256") != recipe.sha256
        or manifest.get("freeze_receipt") != dict(freeze_receipt)
        or manifest.get("aggregate_signal_sources") != dict(evidence)
        or manifest.get("retired_v2_boundary") != dict(recipe.retired_v2_boundary)
    ):
        raise TargetedRepairV3Error("manifest identity or input binding changed")
    for section in (
        "counts",
        "coverage",
        "structural_counts",
        "counterfactual_pairing",
        "anti_shortcut",
        "duplicate_audit",
        "isolation",
        "parameterized_record_count",
    ):
        if manifest.get(section) != recomputed[section]:
            raise TargetedRepairV3Error(f"manifest {section} differs from recomputation")

    with tempfile.TemporaryDirectory(
        prefix="repair-v3-independent-", dir=dataset_dir.parent
    ) as root:
        replay_dir = Path(root) / "rematerialized"
        replay_build = build_repair_dataset(
            recipe_path=recipe_path,
            freeze_receipt_path=freeze_receipt_path,
            d0_report_path=d0_report_path,
            diagnostic_path=diagnostic_path,
        )
        materialize(replay_build, replay_dir)
        byte_identity: dict[str, str] = {}
        for name in sorted(expected_names):
            original = dataset_dir / name
            replay = replay_dir / name
            if original.read_bytes() != replay.read_bytes():
                raise TargetedRepairV3Error("independent rematerialization is not byte-identical")
            byte_identity[name] = sha256_file(original)
    return MappingProxyType(
        {
            "schema_version": 1,
            "artifact_type": "synthetic_targeted_data_repair_v3_independent_audit",
            "status": "passed",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "manifest_sha256": claimed,
            "manifest_file_sha256": sha256_file(manifest_path),
            "recipe_sha256": recipe.sha256,
            "freeze_receipt": dict(freeze_receipt),
            "files": files,
            **dict(recomputed),
            "independent_rematerialization": {
                "executed": True,
                "byte_identical": True,
                "files": byte_identity,
            },
            "source_contract": dict(manifest["source_contract"]),
            "aggregate_signal_sources": dict(evidence),
            "retired_v2_boundary": dict(recipe.retired_v2_boundary),
            "external_benchmark_influence": dict(recipe.external_benchmark_influence),
            "ancestry": dict(recipe.ancestry),
            "privacy": {
                "contains_raw_text": False,
                "contains_entity_values": False,
                "contains_absolute_paths": False,
            },
        }
    )


__all__ = [
    "CORE_LABELS",
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_RECIPE_PATH",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "MATERIALIZER_VERSION",
    "RepairBuild",
    "RepairRecipe",
    "TargetedRepairV3Error",
    "build_repair_dataset",
    "generate_records",
    "load_recipe",
    "materialize",
    "sha256_file",
    "validate_materialized",
    "validate_records",
    "verify_bound_signals",
]
