"""Deterministic train/validation-only synthetic data repair v2.

The generator consumes two aggregate-only public reports and repository-authored
templates. It has no argument or code path for D0 rows, test data, external
benchmarks, human-isolation sources, or LLM generation.
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
from .values import (
    LEGACY_VALUE_VARIANT,
    SYNTHETIC_CARD_IIN,
    SYNTHETIC_ID_REGION_PREFIX,
    SyntheticValueFactory,
    generate_value,
)

DATASET_ID = "pii_zh_synthetic_targeted_repair_v2"
DATASET_VERSION = "0.2.0"
GENERATOR_NAME = "pii_zh_targeted_repair_v2"
GENERATOR_VERSION = "0.2.0"
MATERIALIZER_VERSION = "0.2.0"
DEFAULT_LICENSE = "Apache-2.0"
DEFAULT_RECIPE_PATH = Path(__file__).resolve().parents[4] / "configs/data/targeted_repair_v2.yaml"
_SPLITS = ("train", "validation")
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
_STRUCTURAL_LABELS = ("BANK_CARD_NUMBER", "DATE_OF_BIRTH", "CN_RESIDENT_ID")
_NEGATIVE_GENERATORS = MappingProxyType(
    {
        "bank_card_business_identifier": "BANK_CARD_NUMBER",
        "generic_business_date": "GENERIC_DATE",
        "resident_id_business_identifier": "INVALID_CN_ID",
        "organization": "ORGANIZATION",
        "order_transaction": "TRANSACTION_ID",
        "public_hotline": "ORDER_ID",
        "masked_account": "MASKED_PHONE",
        "product_build": "DEVICE_MODEL",
    }
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
_NGRAM_SIZE = 5
_PLACEHOLDER_RE = re.compile(r"<<([A-Z0-9_]+)>>")
_SURFACE_SHORTCUTS = ("test", "syn", "测试", "示例", "合成", "虚构", "演练")
_EXPLICIT_NEGATIVE_CUES = (
    "不是",
    "不得",
    "无效",
    "失败",
    "错误",
    "拒绝",
    "不对应",
    "不属于",
    "不可",
    "只表示",
    "仅用于",
    "校验",
)


class TargetedRepairV2Error(ValueError):
    """Fail-closed error which never includes source text or generated values."""


@dataclass(frozen=True, slots=True)
class _PlaceholderSpec:
    key: str
    generator_key: str
    label: str | None
    risk_tier: str | None = None
    value_variant: str = LEGACY_VALUE_VARIANT


@dataclass(frozen=True, slots=True)
class _RepairTemplate:
    template_id: str
    family: str
    domain: str
    scene: str
    body: str
    placeholders: tuple[_PlaceholderSpec, ...]
    hard_negative: bool


@dataclass(frozen=True, slots=True)
class _PlannedRecord:
    template: _RepairTemplate
    pair_index: int | None


@dataclass(frozen=True, slots=True)
class _RenderedTemplate:
    text: str
    entities: tuple[EntitySpan, ...]
    values: Mapping[str, str]
    validators_passed: bool
    validator_results: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RepairRecipe:
    recipe_id: str
    recipe_version: str
    seed: int
    counts: Mapping[str, int]
    pii_free_ratio: float
    negative_allocation: Mapping[str, Mapping[str, int]]
    positive_allocation: Mapping[str, Mapping[str, Any]]
    negative_contexts: Mapping[str, Mapping[str, tuple[str, ...]]]
    generic_positive_contexts: Mapping[str, str]
    allowed_aggregate_signals: Mapping[str, Any]
    repair_signal_derivation: Mapping[str, Any]
    external_benchmark_influence: Mapping[str, Any]
    ancestry: Mapping[str, Any]
    validation_gates: Mapping[str, Any]
    generation_contract: Mapping[str, Any]
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
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TargetedRepairV2Error(f"{field} must be a string-keyed object")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TargetedRepairV2Error(f"{field} must be an integer >= {minimum}")
    return value


def _contexts(value: object, *, field: str, placeholders: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TargetedRepairV2Error(f"{field} must be a non-empty string array")
    result = tuple(value)
    for text in result:
        if not isinstance(text, str) or not text.strip() or text != text.strip():
            raise TargetedRepairV2Error(f"{field} contains an invalid template")
        if any(text.count(placeholder) != 1 for placeholder in placeholders):
            raise TargetedRepairV2Error(f"{field} template placeholders changed")
    if len(set(result)) != len(result):
        raise TargetedRepairV2Error(f"{field} contains duplicate templates")
    return result


def _split_contexts(
    value: object,
    *,
    field: str,
    placeholders: tuple[str, ...],
) -> Mapping[str, tuple[str, ...]]:
    raw = _mapping(value, field=field)
    if set(raw) != set(_SPLITS):
        raise TargetedRepairV2Error(f"{field} must define train and validation")
    result = {
        split: _contexts(
            raw[split],
            field=f"{field}.{split}",
            placeholders=placeholders,
        )
        for split in _SPLITS
    }
    if set(result["train"]) & set(result["validation"]):
        raise TargetedRepairV2Error(f"{field} reuses semantic templates across splits")
    return MappingProxyType(result)


def _single_context(
    value: object,
    *,
    field: str,
    placeholders: tuple[str, ...],
) -> str:
    parsed = _contexts([value], field=field, placeholders=placeholders)
    return parsed[0]


def load_recipe(path: Path = DEFAULT_RECIPE_PATH) -> RepairRecipe:
    if path.is_symlink() or not path.is_file():
        raise TargetedRepairV2Error("recipe must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode())
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TargetedRepairV2Error("recipe is invalid YAML") from exc
    root = _mapping(document, field="recipe")
    if (
        root.get("schema_version") != 1
        or root.get("recipe_id") != "pii_zh_synthetic_targeted_repair_v2"
        or root.get("recipe_version") != DATASET_VERSION
    ):
        raise TargetedRepairV2Error("recipe identity is unsupported")
    seed = _integer(root.get("seed"), field="seed")
    counts_raw = _mapping(root.get("counts"), field="counts")
    if set(counts_raw) != set(_SPLITS):
        raise TargetedRepairV2Error("counts must contain train and validation")
    counts = {
        split: _integer(counts_raw[split], field=f"counts.{split}", minimum=1) for split in _SPLITS
    }
    if counts != {"train": 2400, "validation": 600}:
        raise TargetedRepairV2Error("v2 split counts changed")
    ratio = root.get("pii_free_ratio")
    if ratio != 0.35:
        raise TargetedRepairV2Error("v2 PII-free ratio must equal 0.35")
    negative_raw = _mapping(root.get("negative_allocation"), field="negative_allocation")
    if set(negative_raw) != set(_SPLITS):
        raise TargetedRepairV2Error("negative allocation splits changed")
    negative_allocation: dict[str, Mapping[str, int]] = {}
    for split in _SPLITS:
        allocation = _mapping(negative_raw[split], field=f"negative_allocation.{split}")
        if set(allocation) != set(_NEGATIVE_GENERATORS):
            raise TargetedRepairV2Error("negative category inventory changed")
        parsed = {
            category: _integer(value, field=f"negative.{split}.{category}")
            for category, value in allocation.items()
        }
        if sum(parsed.values()) != int(counts[split] * float(ratio)):
            raise TargetedRepairV2Error("negative allocation does not match split ratio")
        minimum = 300 if split == "train" else 75
        if (
            parsed["bank_card_business_identifier"] < minimum
            or parsed["generic_business_date"] < minimum
        ):
            raise TargetedRepairV2Error("card/date negative minimum was weakened")
        negative_allocation[split] = MappingProxyType(parsed)
    positive_raw = _mapping(root.get("positive_allocation"), field="positive_allocation")
    if set(positive_raw) != {*_SPLITS, "extra_label_allocation"}:
        raise TargetedRepairV2Error("positive allocation schema changed")
    if positive_raw["extra_label_allocation"] != "sha256_sorted_core_labels":
        raise TargetedRepairV2Error("positive remainder allocation is not neutral")
    positive_allocation: dict[str, Mapping[str, Any]] = {}
    for split in _SPLITS:
        allocation = _mapping(positive_raw.get(split), field=f"positive_allocation.{split}")
        if set(allocation) != {"base_per_label", "extra_label_count"}:
            raise TargetedRepairV2Error("positive split allocation schema changed")
        base = _integer(
            allocation.get("base_per_label"),
            field=f"positive.{split}.base_per_label",
            minimum=1,
        )
        extra = _integer(
            allocation.get("extra_label_count"),
            field=f"positive.{split}.extra_label_count",
        )
        if extra >= len(CORE_LABELS):
            raise TargetedRepairV2Error("positive extra-label count is out of range")
        expected_positive = counts[split] - sum(negative_allocation[split].values())
        if base * len(CORE_LABELS) + extra != expected_positive:
            raise TargetedRepairV2Error("positive allocation does not fill split")
        positive_allocation[split] = MappingProxyType(
            {"base_per_label": base, "extra_label_count": extra}
        )
    negative_contexts_raw = _mapping(root.get("negative_contexts"), field="negative_contexts")
    if set(negative_contexts_raw) != set(_NEGATIVE_GENERATORS):
        raise TargetedRepairV2Error("negative context inventory changed")
    negative_contexts = {
        category: _split_contexts(
            value,
            field=f"negative_contexts.{category}",
            placeholders=("<<VALUE_1>>", "<<CASE_1>>"),
        )
        for category, value in negative_contexts_raw.items()
    }
    generic_raw = _mapping(root.get("generic_positive_contexts"), field="generic_positive_contexts")
    if set(generic_raw) != set(_SPLITS):
        raise TargetedRepairV2Error("generic positive contexts must define train and validation")
    generic_contexts = {
        split: _single_context(
            generic_raw[split],
            field=f"generic_positive_contexts.{split}",
            placeholders=("<<FIELD_1>>", "<<ENTITY_1>>", "<<CASE_1>>"),
        )
        for split in _SPLITS
    }
    if generic_contexts["train"] == generic_contexts["validation"]:
        raise TargetedRepairV2Error("generic semantic template reuses a split family")
    derivation = _mapping(root.get("repair_signal_derivation"), field="repair_signal_derivation")
    expected_derivation = {
        "source": "D0_validation_aggregate_only",
        "structural_hard_negatives": {
            "bank_card_business_identifier": {
                "observed_false_positive_label": "BANK_CARD_NUMBER",
                "observed_false_positive_count": 150,
            },
            "generic_business_date": {
                "observed_false_positive_label": "DATE_OF_BIRTH",
                "observed_false_positive_count": 150,
            },
            "resident_id_business_identifier": {
                "observed_false_positive_label": "CN_RESIDENT_ID",
                "observed_false_positive_count": 150,
            },
        },
    }
    if dict(derivation) != expected_derivation:
        raise TargetedRepairV2Error("hard-negative derivation is not D0-only")
    influence = _mapping(
        root.get("external_benchmark_influence"),
        field="external_benchmark_influence",
    )
    expected_influence = {
        "designer_has_prior_PII_Bench_result_exposure": True,
        "direct_bound_PII_Bench_signal_used": False,
        "PII_Bench_examples_used": False,
        "PII_Bench_confirmatory_eligible": False,
        "new_unseen_v2_hidden_eligible": True,
        "allowed_future_confirmatory_benchmark": "new_unseen_v2_hidden_only",
    }
    if dict(influence) != expected_influence:
        raise TargetedRepairV2Error("external benchmark influence contract changed")
    ancestry = _mapping(root.get("ancestry"), field="ancestry")
    forbidden_ancestors = ancestry.get("forbidden_parent_artifacts")
    if isinstance(forbidden_ancestors, (str, bytes)) or not isinstance(
        forbidden_ancestors, Sequence
    ):
        raise TargetedRepairV2Error("forbidden ancestry list is invalid")
    if (
        ancestry.get("repair_v1_recipe_report_data_model_ancestry") is not False
        or ancestry.get("artifact_test_informed_ancestry") is not False
        or ancestry.get("designer_external_benchmark_exposure_recorded") is not True
        or ancestry.get("allowed_parent_artifacts")
        != [
            "D0_validation_aggregate_report",
            "repository_authored_targeted_repair_v2_recipe",
        ]
        or set(forbidden_ancestors)
        != {
            "pii_zh_synthetic_targeted_v2_pilot",
            "macbert_24_targeted_repair_v1_seed42",
            "targeted_augmentation_v2_pilot_recipe",
            "targeted_repair_v1_report",
            "PII_Bench",
            "test_split",
            "external_benchmark_examples",
        }
    ):
        raise TargetedRepairV2Error("v2 ancestry contract changed")
    generation = _mapping(root.get("generation_contract"), field="generation_contract")
    expected_safety = {
        "context_origin": "deterministic_repository_authored_synthetic_template",
        "synthetic": True,
        "human_authored_context": False,
        "llm_generated_context": False,
        "source_text_copied": False,
        "D0_row_level_data_read": False,
        "test_data_read": False,
        "external_benchmark_read": False,
        "direct_bound_PII_Bench_signal_used": False,
        "PII_Bench_confirmatory_eligible": False,
        "new_unseen_v2_hidden_eligible": True,
        "repair_v1_ancestry": False,
        "human_isolation_source_read": False,
        "allowed_output_splits": ["train", "validation"],
    }
    if dict(generation) != expected_safety:
        raise TargetedRepairV2Error("generation safety contract changed")
    validation_gates = _mapping(root.get("validation_gates"), field="validation_gates")
    if validation_gates.get("minimum_natural_business_negative_ratio") != 0.80:
        raise TargetedRepairV2Error("natural-business anti-shortcut gate changed")
    return RepairRecipe(
        recipe_id=str(root["recipe_id"]),
        recipe_version=str(root["recipe_version"]),
        seed=seed,
        counts=MappingProxyType(counts),
        pii_free_ratio=float(ratio),
        negative_allocation=MappingProxyType(negative_allocation),
        positive_allocation=MappingProxyType(positive_allocation),
        negative_contexts=MappingProxyType(negative_contexts),
        generic_positive_contexts=MappingProxyType(generic_contexts),
        allowed_aggregate_signals=MappingProxyType(
            dict(_mapping(root.get("allowed_aggregate_signals"), field="aggregate signals"))
        ),
        repair_signal_derivation=MappingProxyType(dict(derivation)),
        external_benchmark_influence=MappingProxyType(dict(influence)),
        ancestry=MappingProxyType(dict(ancestry)),
        validation_gates=MappingProxyType(dict(validation_gates)),
        generation_contract=MappingProxyType(dict(generation)),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _load_report(path: Path, *, expected_file_hash: str, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_file_hash:
        raise TargetedRepairV2Error(f"{field} file hash does not verify")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV2Error(f"{field} is not readable aggregate JSON") from exc
    if not isinstance(value, dict):
        raise TargetedRepairV2Error(f"{field} must be an aggregate object")
    claimed = value.get("report_sha256")
    unsigned = dict(value)
    unsigned.pop("report_sha256", None)
    if claimed != _canonical_hash(unsigned):
        raise TargetedRepairV2Error(f"{field} logical hash does not verify")
    privacy = _mapping(value.get("privacy"), field=f"{field}.privacy")
    if (
        privacy.get("contains_raw_text") is not False
        or privacy.get("contains_entity_values") is not False
    ):
        raise TargetedRepairV2Error(f"{field} is not aggregate-only")
    return value


def verify_aggregate_signals(recipe: RepairRecipe, *, d0_report_path: Path) -> Mapping[str, Any]:
    signals = recipe.allowed_aggregate_signals
    if set(signals) != {"D0_validation_report"}:
        raise TargetedRepairV2Error("only the D0 aggregate report may influence v2")
    d0_contract = _mapping(signals.get("D0_validation_report"), field="D0 signal")
    d0 = _load_report(
        d0_report_path,
        expected_file_hash=str(d0_contract["file_sha256"]),
        field="D0 aggregate report",
    )
    if d0.get("report_sha256") != d0_contract.get("report_sha256"):
        raise TargetedRepairV2Error("aggregate signal logical identity changed")
    d0_validation = _mapping(
        _mapping(d0.get("results"), field="D0.results").get("validation"),
        field="D0.validation",
    )
    d0_per_label = _mapping(d0_validation.get("strict_per_label"), field="D0.per_label")
    if (
        _mapping(d0_validation.get("pii_free"), field="D0.pii_free").get("documents")
        != d0_contract.get("pii_free_documents")
        or _mapping(d0_validation.get("pii_free"), field="D0.pii_free").get("false_positive_rate")
        != d0_contract.get("pii_free_document_fpr")
        or any(
            d0_per_label[label]["fp"] != count
            for label, count in _mapping(
                d0_contract.get("false_positives"), field="D0 false positives"
            ).items()
        )
    ):
        raise TargetedRepairV2Error("D0 aggregate signal values changed")
    return MappingProxyType(
        {
            "D0_report_sha256": d0["report_sha256"],
            "D0_report_file_sha256": sha256_file(d0_report_path),
            "aggregate_reports_read": 1,
            "row_level_D0_data_read": False,
            "test_data_read": False,
            "external_benchmark_read": False,
            "designer_has_prior_PII_Bench_result_exposure": True,
            "direct_bound_PII_Bench_signal_used": False,
            "PII_Bench_confirmatory_eligible": False,
            "new_unseen_v2_hidden_eligible": True,
            "repair_v1_ancestry": False,
            "human_isolation_source_read": False,
        }
    )


def _family(*, role: str, subtype: str, body: str) -> str:
    """Bind split isolation to the actual semantic skeleton, not a split prefix."""

    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]
    return f"repair-v2-{role}-{subtype}-{body_hash}"


def _generic_positive_template(recipe: RepairRecipe, split: str, label: str) -> _RepairTemplate:
    body = recipe.generic_positive_contexts[split]
    return _RepairTemplate(
        template_id=f"repair_v2_{split}_positive_{label.casefold()}_generic",
        family=_family(role="positive", subtype=label.casefold(), body=body),
        domain="synthetic_customer_support",
        scene="synthetic_personal_record_update",
        body=body,
        placeholders=(
            _PlaceholderSpec("FIELD_1", "LITERAL_FIELD_DISPLAY", None),
            _PlaceholderSpec("ENTITY_1", label, label),
            _PlaceholderSpec("CASE_1", "BUILD_ID", None),
        ),
        hard_negative=False,
    )


def _positive_templates(
    recipe: RepairRecipe, split: str, label: str
) -> tuple[_RepairTemplate, ...]:
    return (_generic_positive_template(recipe, split, label),)


def _negative_templates(
    recipe: RepairRecipe, split: str, category: str
) -> tuple[_RepairTemplate, ...]:
    return tuple(
        _RepairTemplate(
            template_id=f"repair_v2_{split}_negative_{category}_{index:02d}",
            family=_family(role="negative", subtype=category, body=body),
            domain="synthetic_business_operations",
            scene=f"synthetic_v2_hard_negative_{category}",
            body=body,
            placeholders=(
                _PlaceholderSpec("VALUE_1", _NEGATIVE_GENERATORS[category], None),
                _PlaceholderSpec("CASE_1", "BUILD_ID", None),
            ),
            hard_negative=True,
        )
        for index, body in enumerate(recipe.negative_contexts[category][split], start=1)
    )


def _cycle(
    templates: Sequence[_RepairTemplate], count: int, rng: random.Random
) -> list[_RepairTemplate]:
    result: list[_RepairTemplate] = []
    while len(result) < count:
        cycle = list(templates)
        rng.shuffle(cycle)
        result.extend(cycle[: count - len(result)])
    return result


def _render_template(template: _RepairTemplate, values: Mapping[str, str]) -> _RenderedTemplate:
    specs = {spec.key: spec for spec in template.placeholders}
    output: list[str] = []
    entities: list[EntitySpan] = []
    validator_results: dict[str, str] = {}
    cursor = 0
    output_length = 0
    validators_passed = True
    for match in _PLACEHOLDER_RE.finditer(template.body):
        literal = template.body[cursor : match.start()]
        output.append(literal)
        output_length += len(literal)
        key = match.group(1)
        if key not in specs or key not in values:
            raise TargetedRepairV2Error("template placeholder declaration is incomplete")
        value = values[key]
        if not isinstance(value, str) or not value:
            raise TargetedRepairV2Error("generated placeholder value is empty")
        start = output_length
        output.append(value)
        output_length += len(value)
        spec = specs[key]
        if spec.label is not None:
            result = validate_entity_value(spec.label, value)
            state = "not_applicable" if result is None else ("passed" if result.valid else "failed")
            if result is not None:
                validator_results[f"{key}:{spec.label}"] = result.reason
                validators_passed = validators_passed and result.valid
            entities.append(
                EntitySpan(
                    start=start,
                    end=output_length,
                    label=spec.label,
                    text=value,
                    source="synthetic",
                    risk_tier=spec.risk_tier,
                    validator_state=state,
                    metadata={"synthetic": True, "placeholder": key},
                )
            )
        cursor = match.end()
    output.append(template.body[cursor:])
    text = "".join(output)
    if set(specs) != set(_PLACEHOLDER_RE.findall(template.body)):
        raise TargetedRepairV2Error("template placeholder declaration changed")
    for entity in entities:
        entity.validate(text)
    if template.hard_negative and entities:
        raise TargetedRepairV2Error("hard-negative template emitted an entity")
    return _RenderedTemplate(
        text=text,
        entities=tuple(entities),
        values=dict(values),
        validators_passed=validators_passed,
        validator_results=MappingProxyType(validator_results),
    )


def _split_seed(seed: int, split: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"repair-v2\0{seed}\0{split}".encode()).digest()[:8], "big"
    )


class _RepairValueFactory(SyntheticValueFactory):
    """Split-independent values with within-split positive/negative pairs."""

    def __init__(self, rng: random.Random, *, split: str) -> None:
        super().__init__(rng)
        self._split_bit = _SPLITS.index(split)
        self._paired_offsets = {
            "card": rng.randrange(500_000_000),
            "date": rng.randrange((10 * 365) // 2),
            "resident": rng.randrange((30 * 365 * 999) // 2),
        }

    def person_name(self) -> str:
        # The middle dot makes this v2 namespace provably disjoint from every
        # repository D0 person-name construction while remaining a natural
        # transliterated-name surface.
        return "安澜·" + self._unique_token("repair_v2_person_name", 8).upper()

    def _paired_serial(self, kind: str, index: int, modulus: int) -> int:
        if index < 0:
            raise TargetedRepairV2Error("pair index cannot be negative")
        half = modulus // 2
        return 2 * ((self._paired_offsets[kind] + index) % half) + self._split_bit

    def paired_bank_card(self, index: int) -> str:
        suffix = self._paired_serial("card", index, 1_000_000_000)
        without_check = SYNTHETIC_CARD_IIN + f"{suffix:09d}"
        return without_check + luhn_check_digit(without_check)

    def paired_date(self, index: int) -> str:
        offset = self._paired_serial("date", index, 10 * 365)
        value = date(1970, 1, 1) + timedelta(days=offset)
        style = (offset // 2) % 3
        if style == 0:
            return value.isoformat()
        if style == 1:
            return f"{value.year}年{value.month}月{value.day}日"
        return f"{value.year}/{value.month:02d}/{value.day:02d}"

    def paired_resident_id(self, index: int, *, valid: bool) -> str:
        serial = self._paired_serial("resident", index, 30 * 365 * 999)
        birth_offset, sequence_offset = divmod(serial, 999)
        birth = date(1980, 1, 1) + timedelta(days=birth_offset)
        first_seventeen = (
            SYNTHETIC_ID_REGION_PREFIX + birth.strftime("%Y%m%d") + f"{sequence_offset + 1:03d}"
        )
        check = cn_resident_id_check_code(first_seventeen)
        if valid:
            return first_seventeen + check
        invalid_check = "0" if check != "0" else "1"
        return first_seventeen + invalid_check


def _value_for(
    factory: _RepairValueFactory,
    spec: _PlaceholderSpec,
    *,
    label: str | None,
    category: str | None,
    pair_index: int | None,
) -> str:
    if spec.generator_key == "LITERAL_FIELD_DISPLAY":
        if label is None:
            raise TargetedRepairV2Error("field display is missing its entity label")
        return _LABEL_DISPLAY[label]
    if pair_index is not None:
        if label == "BANK_CARD_NUMBER" or category == "bank_card_business_identifier":
            return factory.paired_bank_card(pair_index)
        if label == "DATE_OF_BIRTH" or category == "generic_business_date":
            return factory.paired_date(pair_index)
        if label == "CN_RESIDENT_ID":
            return factory.paired_resident_id(pair_index, valid=True)
        if category == "resident_id_business_identifier":
            return factory.paired_resident_id(pair_index, valid=False)
    return generate_value(factory, spec.generator_key, value_variant=spec.value_variant)


def _relation_key(
    template: _RepairTemplate, rendered: _RenderedTemplate, identity: str
) -> tuple[str, str]:
    category = (
        template.scene.removeprefix("synthetic_v2_hard_negative_") if template.hard_negative else ""
    )
    labels = {entity.label for entity in rendered.entities}
    value = rendered.values.get("VALUE_1") or rendered.values.get("ENTITY_1")
    kind = "unique"
    pair_material = identity
    if value and (category == "bank_card_business_identifier" or "BANK_CARD_NUMBER" in labels):
        kind = "bank_card_surface"
        pair_material = value
    elif value and (category == "generic_business_date" or "DATE_OF_BIRTH" in labels):
        kind = "date_surface"
        pair_material = value
    elif value and (category == "resident_id_business_identifier" or "CN_RESIDENT_ID" in labels):
        kind = "resident_id_stem"
        pair_material = value[:-1]
    digest = hashlib.sha256(f"{kind}\0{pair_material}".encode()).hexdigest()
    return "synthetic-repair-v2-relation:sha256:" + digest, kind


def _generate_record(
    *,
    template: _RepairTemplate,
    split: str,
    index: int,
    pair_index: int | None,
    recipe: RepairRecipe,
    factory: _RepairValueFactory,
) -> DocumentRecord:
    entity_label = next(
        (spec.label for spec in template.placeholders if spec.label is not None), None
    )
    category = (
        template.scene.removeprefix("synthetic_v2_hard_negative_")
        if template.hard_negative
        else None
    )
    values = {
        spec.key: _value_for(
            factory,
            spec,
            label=entity_label,
            category=category,
            pair_index=pair_index,
        )
        for spec in template.placeholders
    }
    rendered = _render_template(template, values)
    if not rendered.validators_passed:
        raise TargetedRepairV2Error("generated value failed deterministic validation")
    non_entity_state: str | None = None
    if category == "bank_card_business_identifier":
        if not validate_luhn(rendered.values["VALUE_1"]).valid:
            raise TargetedRepairV2Error("same-surface card confusable failed Luhn")
        non_entity_state = "valid_luhn_surface_semantic_non_entity"
    elif category == "resident_id_business_identifier":
        if validate_cn_resident_id(rendered.values["VALUE_1"]).valid:
            raise TargetedRepairV2Error("invalid-ID negative passed checksum validation")
        non_entity_state = "intentionally_invalid_mod11_2_checksum"
    elif template.hard_negative:
        non_entity_state = "semantic_non_entity_confusable"
    identity = hashlib.sha256(
        f"{DATASET_VERSION}\0{split}\0{index}\0{template.template_id}\0{rendered.text}".encode()
    ).hexdigest()
    relation_group, relation_kind = _relation_key(template, rendered, identity)
    non_entity_groups = tuple(
        sorted(
            stable_text_hash(rendered.values[spec.key])
            for spec in template.placeholders
            if spec.label is None and spec.generator_key != "LITERAL_FIELD_DISPLAY"
        )
    )
    labels = {entity.label for entity in rendered.entities}
    provenance = Provenance(
        source_id="repo_targeted_repair_v2",
        source_kind="deterministic_repository_authored_synthetic",
        source_revision=f"sha256:{recipe.sha256}",
        license=DEFAULT_LICENSE,
        synthetic=True,
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=recipe.seed,
        template_id=template.template_id,
        metadata={
            "contains_real_personal_data": False,
            "context_origin": recipe.generation_contract["context_origin"],
            "human_authored_context": False,
            "llm_generated_context": False,
            "source_text_copied": False,
            "D0_row_level_data_read": False,
            "test_data_read": False,
            "external_benchmark_read": False,
            "designer_has_prior_PII_Bench_result_exposure": True,
            "direct_bound_PII_Bench_signal_used": False,
            "PII_Bench_confirmatory_eligible": False,
            "new_unseen_v2_hidden_eligible": True,
            "repair_v1_ancestry": False,
            "human_isolation_source_read": False,
            "recipe_sha256": recipe.sha256,
            "curated_template_definitions_reused": False,
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
        doc_id="sha256:" + identity,
        text=rendered.text,
        entities=rendered.entities,
        language="zh-Hans-CN",
        domain=template.domain,
        scene=template.scene,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        conversation_group=relation_group,
        template_group=f"synthetic-repair-v2-family:{template.family}",
        entity_value_groups=tuple(
            sorted(stable_text_hash(entity.text) for entity in rendered.entities)
        ),
        generator_version=GENERATOR_VERSION,
        split=split,
        metadata={
            "synthetic": True,
            "targeted_repair_v2": True,
            "context_origin": recipe.generation_contract["context_origin"],
            "human_authored_context": False,
            "llm_generated_context": False,
            "source_text_copied": False,
            "D0_row_level_data_read": False,
            "test_data_read": False,
            "external_benchmark_read": False,
            "designer_has_prior_PII_Bench_result_exposure": True,
            "direct_bound_PII_Bench_signal_used": False,
            "PII_Bench_confirmatory_eligible": False,
            "new_unseen_v2_hidden_eligible": True,
            "repair_v1_ancestry": False,
            "human_isolation_source_read": False,
            "hard_negative": template.hard_negative,
            "hard_negative_category": category,
            "non_entity_validation_state": non_entity_state,
            "non_entity_value_groups": list(non_entity_groups),
            "relation_group": relation_group,
            "relation_kind": relation_kind,
            "pair_allocation_method": "atomic_same_split_plan_before_record_shuffle",
            "structural_repair_labels": sorted(labels & set(_STRUCTURAL_LABELS)),
            "training_role": "synthetic_targeted_data_repair_v2",
        },
    )


def _positive_label_counts(recipe: RepairRecipe, split: str) -> Mapping[str, int]:
    allocation = recipe.positive_allocation[split]
    base = int(allocation["base_per_label"])
    extra = int(allocation["extra_label_count"])
    neutral_order = sorted(
        CORE_LABELS,
        key=lambda label: hashlib.sha256(
            f"repair-v2-neutral-positive-allocation\0{label}".encode()
        ).hexdigest(),
    )
    extra_labels = set(neutral_order[:extra])
    return MappingProxyType({label: base + (label in extra_labels) for label in CORE_LABELS})


def generate_records(recipe: RepairRecipe) -> tuple[DocumentRecord, ...]:
    records: list[DocumentRecord] = []
    for split in _SPLITS:
        rng = random.Random(_split_seed(recipe.seed, split))
        factory = _RepairValueFactory(rng, split=split)
        selected: list[_PlannedRecord] = []
        for label, count in _positive_label_counts(recipe, split).items():
            templates = _cycle(_positive_templates(recipe, split, label), count, rng)
            selected.extend(
                _PlannedRecord(
                    template=template,
                    pair_index=index if label in _STRUCTURAL_LABELS else None,
                )
                for index, template in enumerate(templates)
            )
        for category, count in recipe.negative_allocation[split].items():
            templates = _cycle(_negative_templates(recipe, split, category), int(count), rng)
            paired_category = category in {
                "bank_card_business_identifier",
                "generic_business_date",
                "resident_id_business_identifier",
            }
            selected.extend(
                _PlannedRecord(
                    template=template,
                    pair_index=index if paired_category else None,
                )
                for index, template in enumerate(templates)
            )
        if len(selected) != recipe.counts[split]:
            raise TargetedRepairV2Error("template allocation does not fill split")
        rng.shuffle(selected)
        for index, planned in enumerate(selected):
            records.append(
                _generate_record(
                    template=planned.template,
                    split=split,
                    index=index,
                    pair_index=planned.pair_index,
                    recipe=recipe,
                    factory=factory,
                )
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


def _assert_surface_contract(records: Sequence[DocumentRecord]) -> None:
    seen_entity_values: set[str] = set()
    for record in records:
        record.validate()
        lowered = record.text.casefold()
        if any(shortcut in lowered for shortcut in _SURFACE_SHORTCUTS):
            raise TargetedRepairV2Error("generated text contains a synthetic shortcut marker")
        for value_group in record.entity_value_groups:
            if value_group in seen_entity_values:
                raise TargetedRepairV2Error("generated positive entity values must be unique")
            seen_entity_values.add(value_group)


def _duplicate_audit(
    records: Sequence[DocumentRecord], threshold: float, *, scope: str
) -> dict[str, Any]:
    texts = [record.text for record in records]
    counts = Counter(texts)
    exact = sum(count * (count - 1) // 2 for count in counts.values())
    grams = [_ngrams(text) for text in texts]
    inverted: dict[str, list[int]] = {}
    near = 0
    compared_pairs = 0
    for right, right_grams in enumerate(grams):
        candidates: set[int] = set()
        for gram in right_grams:
            candidates.update(inverted.get(gram, ()))
        for left in candidates:
            left_grams = grams[left]
            smaller = min(len(left_grams), len(right_grams))
            larger = max(len(left_grams), len(right_grams))
            if larger and smaller / larger < threshold:
                continue
            compared_pairs += 1
            intersection = len(left_grams & right_grams)
            union_size = len(left_grams) + len(right_grams) - intersection
            similarity = intersection / union_size if union_size else 1.0
            near += similarity >= threshold
        for gram in right_grams:
            inverted.setdefault(gram, []).append(right)
    if exact or near:
        raise TargetedRepairV2Error(f"{scope} contains exact or char-5gram near duplicates")
    return {
        "scope": scope,
        "document_count": len(texts),
        "exact_duplicate_pair_count": exact,
        "near_duplicate_pair_count": near,
        "candidate_pair_count": compared_pairs,
        "near_duplicate_method": "character_5gram_jaccard",
        "near_duplicate_threshold": threshold,
    }


def _isolation(records: Sequence[DocumentRecord]) -> dict[str, Any]:
    sets = {
        field: {split: set() for split in _SPLITS}
        for field in (
            "document_ids",
            "template_groups",
            "value_groups",
            "relation_groups",
            "text_hashes",
        )
    }
    for record in records:
        split = str(record.split)
        sets["document_ids"][split].add(record.doc_id)
        sets["template_groups"][split].add(record.template_group)
        values = set(record.entity_value_groups)
        non_entity = record.metadata.get("non_entity_value_groups")
        if not isinstance(non_entity, list) or not non_entity:
            raise TargetedRepairV2Error("every record must contain parameter value groups")
        values.update(non_entity)
        sets["value_groups"][split].update(values)
        sets["relation_groups"][split].add(record.conversation_group)
        sets["text_hashes"][split].add(hashlib.sha256(record.text.encode()).hexdigest())
    result: dict[str, Any] = {}
    for field, values in sets.items():
        collision = values["train"] & values["validation"]
        if collision:
            raise TargetedRepairV2Error(f"{field} collide across train and validation")
        result[field] = {
            "train": len(values["train"]),
            "validation": len(values["validation"]),
            "cross_split_collisions": 0,
        }
    return result


def validate_records(records: Sequence[DocumentRecord], recipe: RepairRecipe) -> Mapping[str, Any]:
    if len(records) != sum(recipe.counts.values()):
        raise TargetedRepairV2Error("record count differs from recipe")
    _assert_surface_contract(records)
    by_split = {split: [record for record in records if record.split == split] for split in _SPLITS}
    label_counts: dict[str, dict[str, int]] = {}
    negative_counts: dict[str, dict[str, int]] = {}
    negative_duplicate_audits: dict[str, Any] = {}
    anti_shortcut_by_split: dict[str, Any] = {}
    ratio_counts: dict[str, int] = {}
    threshold = float(recipe.validation_gates["char_5gram_near_duplicate_threshold"])
    for split, selected in by_split.items():
        if len(selected) != recipe.counts[split] or len({r.doc_id for r in selected}) != len(
            selected
        ):
            raise TargetedRepairV2Error("split count or document identity is invalid")
        labels = Counter(entity.label for record in selected for entity in record.entities)
        expected_labels = dict(_positive_label_counts(recipe, split))
        if dict(labels) != expected_labels:
            raise TargetedRepairV2Error("positive 24-label allocation changed")
        if max(labels.values()) - min(labels.values()) > 1:
            raise TargetedRepairV2Error("positive labels are not neutrally balanced")
        label_counts[split] = dict(sorted(labels.items()))
        negatives = Counter(
            str(record.metadata["hard_negative_category"])
            for record in selected
            if not record.entities
        )
        if dict(negatives) != dict(recipe.negative_allocation[split]):
            raise TargetedRepairV2Error("negative allocation changed")
        negative_counts[split] = dict(sorted(negatives.items()))
        pii_free = sum(not record.entities for record in selected)
        if pii_free / len(selected) != recipe.pii_free_ratio:
            raise TargetedRepairV2Error("PII-free ratio changed")
        ratio_counts[split] = pii_free
        explicit_cue_documents = sum(
            any(cue in record.text for cue in _EXPLICIT_NEGATIVE_CUES)
            for record in selected
            if not record.entities
        )
        natural_business_documents = pii_free - explicit_cue_documents
        natural_ratio = natural_business_documents / pii_free
        minimum_natural_ratio = float(
            recipe.validation_gates["minimum_natural_business_negative_ratio"]
        )
        if natural_ratio < minimum_natural_ratio:
            raise TargetedRepairV2Error(
                "hard negatives rely too heavily on explicit non-entity cues"
            )
        anti_shortcut_by_split[split] = {
            "negative_documents": pii_free,
            "natural_business_context_documents": natural_business_documents,
            "explicit_negative_cue_documents": explicit_cue_documents,
            "natural_business_context_ratio": natural_ratio,
            "minimum_required_ratio": minimum_natural_ratio,
            "gate_passed": True,
        }
        negative_duplicate_audits[split] = _duplicate_audit(
            [record for record in selected if not record.entities],
            threshold,
            scope=f"hard_negatives_{split}",
        )
        for record in selected:
            if (
                record.data_pool is not DataPool.PUBLIC_RELEASE
                or record.provenance.synthetic is not True
                or record.public_weight_training_allowed is not True
                or record.metadata.get("synthetic") is not True
                or record.metadata.get("human_authored_context") is not False
                or record.metadata.get("llm_generated_context") is not False
                or record.metadata.get("D0_row_level_data_read") is not False
                or record.metadata.get("test_data_read") is not False
                or record.metadata.get("external_benchmark_read") is not False
                or record.metadata.get("designer_has_prior_PII_Bench_result_exposure") is not True
                or record.metadata.get("direct_bound_PII_Bench_signal_used") is not False
                or record.metadata.get("PII_Bench_confirmatory_eligible") is not False
                or record.metadata.get("new_unseen_v2_hidden_eligible") is not True
                or record.metadata.get("repair_v1_ancestry") is not False
                or record.metadata.get("human_isolation_source_read") is not False
            ):
                raise TargetedRepairV2Error("record provenance escaped the synthetic contract")
            category = record.metadata.get("hard_negative_category")
            validation_state = record.metadata.get("non_entity_validation_state")
            if category == "bank_card_business_identifier" and validation_state != (
                "valid_luhn_surface_semantic_non_entity"
            ):
                raise TargetedRepairV2Error("card-confusable semantic state is not explicit")
            if category == "resident_id_business_identifier" and validation_state != (
                "intentionally_invalid_mod11_2_checksum"
            ):
                raise TargetedRepairV2Error("invalid-ID state is not explicit")
    duplicate_audits = {
        "all_records": _duplicate_audit(records, threshold, scope="all_records_cross_split"),
        "hard_negatives_by_split": negative_duplicate_audits,
    }
    isolation = _isolation(records)
    paired_relations: dict[str, dict[str, int]] = {}
    for split, selected in by_split.items():
        relation_members: dict[str, list[DocumentRecord]] = {}
        for record in selected:
            relation_members.setdefault(str(record.conversation_group), []).append(record)
        paired_relations[split] = {}
        for kind in ("bank_card_surface", "date_surface", "resident_id_stem"):
            paired = 0
            for members in relation_members.values():
                if not members or members[0].metadata.get("relation_kind") != kind:
                    continue
                has_positive = any(member.entities for member in members)
                has_negative = any(not member.entities for member in members)
                if has_positive and has_negative:
                    if len(members) != 2 or any(
                        member.metadata.get("pair_allocation_method")
                        != "atomic_same_split_plan_before_record_shuffle"
                        for member in members
                    ):
                        raise TargetedRepairV2Error(
                            "counterfactual relation was not atomically allocated"
                        )
                    if kind in {"bank_card_surface", "date_surface"}:
                        positive_member = next(member for member in members if member.entities)
                        negative_member = next(member for member in members if not member.entities)
                        negative_groups = set(negative_member.metadata["non_entity_value_groups"])
                        if not set(positive_member.entity_value_groups) & negative_groups:
                            raise TargetedRepairV2Error(
                                "counterfactual pair does not share the exact surface"
                            )
                    paired += 1
            paired_relations[split][kind] = paired
        expected_pairs = {
            "bank_card_surface": min(
                int(_positive_label_counts(recipe, split)["BANK_CARD_NUMBER"]),
                int(recipe.negative_allocation[split]["bank_card_business_identifier"]),
            ),
            "date_surface": min(
                int(_positive_label_counts(recipe, split)["DATE_OF_BIRTH"]),
                int(recipe.negative_allocation[split]["generic_business_date"]),
            ),
            "resident_id_stem": min(
                int(_positive_label_counts(recipe, split)["CN_RESIDENT_ID"]),
                int(recipe.negative_allocation[split]["resident_id_business_identifier"]),
            ),
        }
        if paired_relations[split] != expected_pairs:
            raise TargetedRepairV2Error("counterfactual relation pairing changed")
    return MappingProxyType(
        {
            "counts": {
                "total": len(records),
                "by_split": {split: len(by_split[split]) for split in _SPLITS},
                "pii_free_by_split": ratio_counts,
            },
            "ratios": {
                "pii_free_by_split": {
                    split: ratio_counts[split] / len(by_split[split]) for split in _SPLITS
                }
            },
            "coverage": {
                "required_core_label_count": len(CORE_LABELS),
                "labels_by_split": label_counts,
                "hard_negative_categories_by_split": negative_counts,
                "positive_allocation_policy": "neutral_balanced_max_difference_one",
                "structural_hard_negative_labels": list(_STRUCTURAL_LABELS),
            },
            "duplicate_audit": duplicate_audits,
            "isolation": isolation,
            "counterfactual_pairing": {
                "method": (
                    "atomic_same_split_exact_surface_for_card_date_and_"
                    "identifier_stem_for_resident_id"
                ),
                "paired_relation_groups_by_split": paired_relations,
            },
            "anti_shortcut": {
                "explicit_negative_cues_retained_in_report": False,
                "by_split": anti_shortcut_by_split,
            },
            "parameterized_record_count": len(records),
        }
    )


def build_repair_dataset(*, recipe_path: Path, d0_report_path: Path) -> RepairBuild:
    recipe = load_recipe(recipe_path)
    aggregate_evidence = verify_aggregate_signals(recipe, d0_report_path=d0_report_path)
    records = generate_records(recipe)
    validation = validate_records(records, recipe)
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "artifact_type": "synthetic_targeted_data_repair_v2",
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
        "aggregate_signal_sources": dict(aggregate_evidence),
        "repair_signal_derivation": dict(recipe.repair_signal_derivation),
        "external_benchmark_influence": dict(recipe.external_benchmark_influence),
        "ancestry": dict(recipe.ancestry),
        "source_contract": {
            "aggregate_reports_read": 1,
            "curated_template_definitions_reused": False,
            "D0_row_level_data_read": False,
            "test_data_read": False,
            "external_benchmark_read": False,
            "designer_has_prior_PII_Bench_result_exposure": True,
            "direct_bound_PII_Bench_signal_used": False,
            "PII_Bench_confirmatory_eligible": False,
            "new_unseen_v2_hidden_eligible": True,
            "repair_v1_ancestry": False,
            "human_isolation_source_read": False,
            "LLM_generation_used": False,
        },
        "generation": {
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "context_origin": recipe.generation_contract["context_origin"],
            "synthetic": True,
            "human_authored_context": False,
            "llm_generated_context": False,
            "source_text_copied": False,
            "split_independent_rng": True,
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
        raise TargetedRepairV2Error("output directory must be a new absolute directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        files: list[dict[str, Any]] = []
        for split in _SPLITS:
            path = stage / f"{split}.jsonl"
            selected = [record for record in build.records if record.split == split]
            count = write_jsonl(selected, path)
            path.chmod(0o444)
            if count != len(selected) or len(list(iter_jsonl(path))) != count:
                raise TargetedRepairV2Error("materialized split readback failed")
            files.append(
                {
                    "name": path.name,
                    "split": split,
                    "records": count,
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


def validate_materialized(
    *,
    dataset_dir: Path,
    recipe_path: Path,
    d0_report_path: Path,
) -> Mapping[str, Any]:
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise TargetedRepairV2Error("dataset directory must be regular")
    expected_names = {"train.jsonl", "validation.jsonl", "dataset_manifest.json"}
    if {path.name for path in dataset_dir.iterdir()} != expected_names or any(
        path.is_symlink() for path in dataset_dir.iterdir()
    ):
        raise TargetedRepairV2Error("dataset file set is invalid")
    recipe = load_recipe(recipe_path)
    aggregate_evidence = verify_aggregate_signals(recipe, d0_report_path=d0_report_path)
    records: list[DocumentRecord] = []
    files: list[dict[str, Any]] = []
    for split in _SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        selected = list(iter_jsonl(path))
        if any(record.split != split for record in selected):
            raise TargetedRepairV2Error("record escaped its materialized split")
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
    recomputed = validate_records(records, recipe)
    manifest_path = dataset_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV2Error("manifest is not readable JSON") from exc
    if not isinstance(manifest, dict):
        raise TargetedRepairV2Error("manifest must be an object")
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_hash(unsigned):
        raise TargetedRepairV2Error("manifest logical hash failed")
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("files") != files
        or manifest.get("recipe", {}).get("sha256") != recipe.sha256
        or manifest.get("aggregate_signal_sources") != dict(aggregate_evidence)
        or manifest.get("repair_signal_derivation") != dict(recipe.repair_signal_derivation)
        or manifest.get("external_benchmark_influence") != dict(recipe.external_benchmark_influence)
        or manifest.get("ancestry") != dict(recipe.ancestry)
    ):
        raise TargetedRepairV2Error("manifest identity or input binding changed")
    for section in (
        "counts",
        "ratios",
        "coverage",
        "duplicate_audit",
        "isolation",
        "counterfactual_pairing",
        "anti_shortcut",
        "parameterized_record_count",
    ):
        if manifest.get(section) != recomputed[section]:
            raise TargetedRepairV2Error(f"manifest {section} differs from recomputation")
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
        raise TargetedRepairV2Error("manifest contains an absolute path")
    return MappingProxyType(
        {
            "schema_version": 1,
            "artifact_type": "synthetic_targeted_data_repair_v2_audit",
            "status": "passed",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "manifest_sha256": claimed,
            "manifest_file_sha256": sha256_file(manifest_path),
            "recipe_sha256": recipe.sha256,
            "files": files,
            **dict(recomputed),
            "source_contract": dict(manifest["source_contract"]),
            "aggregate_signal_sources": dict(aggregate_evidence),
            "repair_signal_derivation": dict(recipe.repair_signal_derivation),
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
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_RECIPE_PATH",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "MATERIALIZER_VERSION",
    "RepairBuild",
    "RepairRecipe",
    "TargetedRepairV2Error",
    "build_repair_dataset",
    "generate_records",
    "load_recipe",
    "materialize",
    "validate_materialized",
    "validate_records",
    "verify_aggregate_signals",
]
