"""Clean-room natural-context Chinese PII corpus v7.2.

This module is deliberately independent from the v7.1 materializer.  It reads
only its closed repository configuration, its own source, and the repository
taxonomy.  It has no benchmark, prediction, prior-corpus, or external-dataset
input.  The independent test partition is generated first and sealed before
the development and training files are written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from pii_zh.data.schema import (
    SCHEMA_VERSION,
    DataPool,
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    iter_jsonl,
    stable_text_hash,
    write_jsonl,
)
from pii_zh.data.synthetic.generator import render_template
from pii_zh.data.synthetic.templates import CORE_LABELS, PlaceholderSpec, SyntheticTemplate
from pii_zh.data.synthetic.values import SyntheticValueFactory, generate_value
from pii_zh.data.validators import validate_entity_value

DATASET_ID = "pii_zh_domain_randomized_v7"
DATASET_VERSION = "7.2.0"
GENERATOR_NAME = "pii_zh_domain_randomized_natural_context"
GENERATOR_VERSION = "7.2.0"
MATERIALIZER_VERSION = "7.2.0"
DEFAULT_CONFIG_PATH = Path("configs/data/domain_randomized_v7_2.yaml")
DEFAULT_TAXONOMY_PATH = Path("src/pii_zh/taxonomy/taxonomy.yaml")
DEFAULT_DATA_ROOT = Path(os.environ.get("PII_ZH_DATA_ROOT", "data"))
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed/public_release_pool/domain_randomized_v7_2"

SPLITS = ("train", "dev", "independent_test")
GENERATION_ORDER = ("independent_test", "train", "dev")
POSITIVE_COMPOSITIONS = (
    "natural_dialogue",
    "natural_narrative",
    "natural_workflow",
    "natural_multi_entity",
    "structured_label_first",
    "structured_value_first",
)
NATURAL_POSITIVE_COMPOSITIONS = POSITIVE_COMPOSITIONS[:4]
COMPOSITIONS = (*POSITIVE_COMPOSITIONS, "counterfactual", "pii_free")
CUE_PLACEMENTS = ("left_only", "right_only", "bilateral")
_CUE_ASSIGNMENT_ORDER = ("right_only", "bilateral", "left_only")
FIELD_POSITIONS = ("early", "middle", "late")
LENGTH_BANDS = ("short", "medium", "long")
NOISE_STYLES = ("clean", "ocr", "asr")
GROUP_KEYS = (
    "grammar_groups",
    "template_families",
    "semantic_skeletons",
    "value_groups",
    "source_groups",
    "text_hashes",
    "doc_ids",
)
GRAMMAR_FEATURE_KEYS = (
    "composition",
    "frame_variant",
    "field_position",
    "length_band",
    "noise_style",
    "punctuation_variant",
    "primary_label",
    "secondary_label",
    "context_signature",
)

DOMAINS = (
    "customer_service",
    "ecommerce",
    "finance",
    "healthcare",
    "education",
    "human_resources",
    "government",
    "transport",
    "messaging",
    "email_contract",
    "developer_operations",
    "semi_structured_forms",
)

_DOMAIN_TITLES = {
    "customer_service": "客户服务",
    "ecommerce": "电子商务",
    "finance": "金融业务",
    "healthcare": "医疗服务",
    "education": "教育管理",
    "human_resources": "人力资源",
    "government": "政务办理",
    "transport": "交通出行",
    "messaging": "即时通信",
    "email_contract": "邮件与合同",
    "developer_operations": "研发运维",
    "semi_structured_forms": "半结构化表单",
}

_LABEL_ALIASES: Mapping[str, tuple[str, str, str]] = {
    "PERSON_NAME": ("本人姓名", "登记人称呼", "联系人名字"),
    "PHONE_NUMBER": ("联系电话", "回拨号码", "手机号码"),
    "EMAIL_ADDRESS": ("电子邮箱", "收件邮箱", "邮件地址"),
    "ADDRESS": ("联系地址", "收件地点", "邮寄地址"),
    "DATE_OF_BIRTH": ("出生日期", "出生年月日", "生日信息"),
    "CN_RESIDENT_ID": ("居民身份证号", "身份证件号码", "本人证件号"),
    "PASSPORT_NUMBER": ("护照号码", "出境证件号", "护照本编号"),
    "DRIVER_LICENSE_NUMBER": ("驾驶证号码", "驾照证件号", "驾驶资格编号"),
    "SOCIAL_SECURITY_NUMBER": ("社会保障号码", "社保登记号", "社会保险编号"),
    "BANK_CARD_NUMBER": ("银行卡号码", "付款卡号", "退款卡编号"),
    "BANK_ACCOUNT_NUMBER": ("银行账户号", "收款账户", "结算账号"),
    "VEHICLE_LICENSE_PLATE": ("车辆号牌", "车牌号码", "机动车牌照"),
    "EMPLOYEE_ID": ("员工编号", "内部工号", "人员工牌号"),
    "STUDENT_ID": ("学生学号", "教务编号", "在校生编号"),
    "MEDICAL_RECORD_NUMBER": ("病历号码", "就诊档案号", "医疗记录编号"),
    "WECHAT_ID": ("微信账号", "微信联系号", "微信用户标识"),
    "QQ_NUMBER": ("QQ号码", "QQ联系号", "QQ账号编号"),
    "ALIPAY_ACCOUNT": ("支付宝账号", "支付宝收款号", "支付宝账户标识"),
    "USERNAME": ("平台用户名", "登录用户名", "系统用户标识"),
    "IP_ADDRESS": ("网络IP地址", "接入IP", "主机网络地址"),
    "MAC_ADDRESS": ("网卡MAC地址", "设备物理地址", "网络适配器地址"),
    "DEVICE_ID": ("设备标识", "终端编号", "设备唯一编号"),
    "GEO_COORDINATE": ("地理坐标", "定位经纬度", "位置坐标值"),
    "SECRET": ("访问密钥", "认证秘钥串", "部署凭据"),
}

_T2_LABELS = {
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "MEDICAL_RECORD_NUMBER",
    "SECRET",
}
_T0_LABELS = {"IP_ADDRESS", "MAC_ADDRESS"}

_PUNCTUATION_STYLES = (
    ("：", "；", "。"),
    ("——", "，", "。"),
    ("（", "）", "。"),
    ("：\n", "\n", "。"),
)

_SPLIT_SURFACES: Mapping[str, Mapping[str, Any]] = {
    "train": {
        "roles": (("来访人", "受理员"), ("客户", "坐席"), ("申请人", "窗口人员"), ("用户", "专员")),
        "openers": (
            "刚才的沟通还缺一项",
            "办理过程中需要补全内容",
            "为了继续流转当前事项",
            "这次登记要核清上下文",
        ),
        "intents": ("补交材料", "确认联系", "完成核验", "安排后续", "修改登记", "恢复访问"),
        "closers": (
            "确认后继续办理",
            "随后转入下一环节",
            "记录员将整句话一并留存",
            "双方复述无误后提交",
        ),
        "fillers": {
            "short": "当前事项仍在受理中。",
            "medium": "工作人员先核对说话人的目的，再把整句话连同上下文写入受理记录。",
            "long": (
                "工作人员先确认说话人的目的和前后指代，再复述完整句子，"
                "最后把上下文一并写入受理记录。"
            ),
        },
        "noise": {"ocr": ("扫描转写", "页尾"), "asr": ("语音抄录", "转写结束")},
        "left": (
            "我把{alias}补充为{value}",
            "请按{alias}{value}登记",
            "这次使用的{alias}是{value}",
            "系统需要的{alias}填{value}",
        ),
        "right": (
            "{value}，这是这次要留的{alias}",
            "{value}，后面请作为{alias}使用",
            "{value}，我说的是本次{alias}",
            "{value}，这一项对应{alias}",
        ),
        "bilateral": (
            "关于{alias}，我填{value}，后续按这项{alias}核对",
            "这次{alias}用{value}，请把它作为{alias}留存",
            "需要更新{alias}为{value}，之后仍按{alias}联系",
            "我的{alias}就是{value}，该值用于本次{alias}登记",
        ),
    },
    "dev": {
        "roles": (
            ("来电人", "复核员"),
            ("办事人", "记录员"),
            ("寄件人", "归档员"),
            ("访客", "值班员"),
        ),
        "openers": (
            "回看录音时发现信息散落",
            "复核历史条目时需要还原语义",
            "交接前要补写一处内容",
            "归档人员正在整理前后叙述",
        ),
        "intents": ("核对归档", "补写交接", "确认回执", "复查留存", "整理附件", "校正抄录"),
        "closers": (
            "复核完成后封存条目",
            "归档员按原话完成抄录",
            "上下文核对一致后签收",
            "交接人确认语义后关闭记录",
        ),
        "fillers": {
            "short": "这段记录等待复核。",
            "medium": "归档员对照前后叙述检查指代，并保留能够解释该值用途的完整句子。",
            "long": (
                "归档员对照前后叙述、业务目的和说话人角色逐项检查，"
                "再保留能够解释该值用途的完整句子。"
            ),
        },
        "noise": {"ocr": ("归档扫描", "复核页末"), "asr": ("录音回听", "抄录完毕")},
        "left": (
            "复核到的{alias}记作{value}",
            "原话把{alias}说成{value}",
            "归档时应将{alias}抄为{value}",
            "交接记录中的{alias}为{value}",
        ),
        "right": (
            "{value}，录音随后说明这是{alias}",
            "{value}，归档员确认它指代{alias}",
            "{value}，后一句把该值解释为{alias}",
            "{value}，交接备注称其用途是{alias}",
        ),
        "bilateral": (
            "复核{alias}时听到{value}，后文再次称它为{alias}",
            "归档的{alias}写成{value}，交接页也按{alias}引用",
            "原记录把{alias}记为{value}，复述时仍称作{alias}",
            "需确认的{alias}是{value}，上下文都在讨论{alias}",
        ),
    },
    "independent_test": {
        "roles": (
            ("咨询人", "审核岗"),
            ("提交人", "收件岗"),
            ("当事人", "查验岗"),
            ("联络人", "回访岗"),
        ),
        "openers": (
            "整理独立查验材料时出现一段转述",
            "收件岗需要理解完整语句再录入",
            "回访记录保留了自然说法",
            "审核岗正在判断该值的实际用途",
        ),
        "intents": ("完成查验", "安排回访", "确认投递", "恢复流程", "补齐凭据", "核清身份"),
        "closers": (
            "审核岗依据整句语义完成查验",
            "收件岗确认用途后封存材料",
            "回访岗复述原意后结束通话",
            "查验岗保留前后文作为依据",
        ),
        "fillers": {
            "short": "独立查验仍需保留语境。",
            "medium": "审核人员不会只看字段形状，而是连同说话人的目的和后续解释一起判断。",
            "long": (
                "审核人员不会只看字段形状，而是结合说话人的目的、后续解释"
                "和业务动作还原完整语义后再判断。"
            ),
        },
        "noise": {"ocr": ("材料识别", "查验页终"), "asr": ("回访转录", "通话转录结束")},
        "left": (
            "提交材料中的{alias}写的是{value}",
            "查验人说明{alias}应取{value}",
            "这份材料把{alias}列为{value}",
            "回访确认的{alias}就是{value}",
        ),
        "right": (
            "{value}，对方接着解释这是{alias}",
            "{value}，材料后文将其归为{alias}",
            "{value}，回访答复表明它是{alias}",
            "{value}，查验结论把该项认作{alias}",
        ),
        "bilateral": (
            "提交的{alias}为{value}，材料末尾仍以{alias}指代",
            "查验{alias}得到{value}，回访时又确认这是{alias}",
            "材料列出的{alias}是{value}，结论继续引用该{alias}",
            "回访要确认{alias}为{value}，答复也明确这是{alias}",
        ),
    },
}

_COUNTERFACTUAL_CONTEXTS = {
    "PERSON_NAME": "会议室代称",
    "PHONE_NUMBER": "公共语音总机",
    "EMAIL_ADDRESS": "无人值守工单入口",
    "ADDRESS": "机构直营网点位置",
    "DATE_OF_BIRTH": "设备投产日期",
    "CN_RESIDENT_ID": "风控规则保留编号",
    "PASSPORT_NUMBER": "运输批次编号",
    "DRIVER_LICENSE_NUMBER": "设备许可编号",
    "SOCIAL_SECURITY_NUMBER": "接口字段保留值",
    "BANK_CARD_NUMBER": "收单路由保留号",
    "BANK_ACCOUNT_NUMBER": "清算服务账号",
    "VEHICLE_LICENSE_PLATE": "停车工位标识",
    "EMPLOYEE_ID": "组织岗位编号",
    "STUDENT_ID": "课程批次编号",
    "MEDICAL_RECORD_NUMBER": "医疗耗材批次号",
    "WECHAT_ID": "企业公共服务号",
    "QQ_NUMBER": "公开交流群标识",
    "ALIPAY_ACCOUNT": "商户结算别名",
    "USERNAME": "共享机器人账号",
    "IP_ADDRESS": "协议文档保留地址",
    "MAC_ADDRESS": "实验台网卡地址",
    "DEVICE_ID": "库存设备编号",
    "GEO_COORDINATE": "公共设施坐标",
    "SECRET": "已经撤销的配置标记",
}

_PII_FREE_GENERATORS = (
    ("ORDER_ID", "订单编号"),
    ("TRACE_ID", "链路追踪号"),
    ("TRANSACTION_ID", "交易流水号"),
    ("INVOICE_CODE", "发票代码"),
    ("PRODUCT_CODE", "商品编码"),
    ("ORGANIZATION", "机构名称"),
    ("JOB_TITLE", "岗位名称"),
    ("GENERIC_DATE", "业务日期"),
    ("REGION_CODE", "区域代码"),
    ("BUILD_ID", "构建编号"),
    ("CONFIG_KEY_NAME", "配置项名称"),
    ("RELEASE_DATE", "发布日期"),
    ("DEVICE_MODEL", "设备型号"),
)

_NEGATIVE_FRAMES = {
    "train": (
        "业务目录把{value}作为{context}维护，值由公共流程统一使用",
        "运行台账中的{value}对应{context}，后续由系统任务继续引用",
        "服务清单记录{value}这项{context}，它服务于机构级流程",
    ),
    "dev": (
        "历史索引将{value}归入{context}，归档说明其用于公共流程",
        "复核附件里的{value}表示{context}，交接页沿用同一含义",
        "留存目录把{value}解释成{context}，审核记录未改变其用途",
    ),
    "independent_test": (
        "独立查验材料把{value}列作{context}，后文说明它属于公共配置",
        "收件清册中的{value}承担{context}用途，审核结论按此归类",
        "回访附件引用{value}这项{context}，整段都在描述机构流程",
    ),
}
_PLAIN_NEGATIVE_FRAMES = {
    "train": "业务台账把{value}登记为{description}，随后同步处理批次。",
    "dev": "归档清册将{value}抄录为{description}，复核后完成留存。",
    "independent_test": "独立查验材料列出{value}这项{description}，审核岗据此归档。",
}
_EXPLICIT_NEGATIVE_FRAMES = {
    "train": "核对备注：{value}属于{context}，不是个人{alias}。",
    "dev": "归档注记：{value}表示{context}，并非用户的{alias}。",
    "independent_test": "查验说明：{value}承担{context}用途，不对应自然人的{alias}。",
}
_SHORTCUT_TOKENS = ("不是个人", "并非用户", "不对应自然人")
_PLACEHOLDER_RE = re.compile(r"<<[A-Z0-9_]+>>")


class DomainRandomizedV72Error(ValueError):
    """Raised when the closed v7.2 generation contract is violated."""


@dataclass(frozen=True, slots=True)
class DomainRandomizedV72Config:
    config_path: Path
    config_sha256: str
    repository_root: Path
    taxonomy_path: Path
    seed: int
    split_counts: Mapping[str, int]
    pii_free_hard_negative_ratio: float
    natural_positive_ratio: float
    counterfactual_share_of_hard_negative: float
    positive_composition: Mapping[str, float]
    cue_placements: tuple[str, ...]
    domains: tuple[str, ...]
    diversity_gates: Mapping[str, Any]
    usage_policy: Mapping[str, Mapping[str, bool]]
    clean_room: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_hash(value: Any, *, remove: str | None = None) -> str:
    document = dict(value) if isinstance(value, Mapping) else value
    if remove is not None and isinstance(document, dict):
        document.pop(remove, None)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _group_hash(namespace: str, value: str) -> str:
    return "sha256:" + _sha256_bytes(f"{namespace}\0{value}".encode())


def _require_mapping(value: Any, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise DomainRandomizedV72Error(
            f"{name} must have the closed fields {sorted(keys)}; observed={observed}"
        )
    return value


def _risk_tier(label: str) -> str:
    if label in _T2_LABELS:
        return "T2"
    if label in _T0_LABELS:
        return "T0"
    return "T1"


def _allocate(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    exact = {name: total * weight for name, weight in weights.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(result.values())
    ranked = sorted(weights, key=lambda name: (-(exact[name] - result[name]), name))
    for name in ranked[:remaining]:
        result[name] += 1
    return result


def _validate_runtime_config(config: DomainRandomizedV72Config) -> None:
    if tuple(config.split_counts) != SPLITS:
        raise DomainRandomizedV72Error("split order must be train/dev/independent_test")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 640
        for count in config.split_counts.values()
    ):
        raise DomainRandomizedV72Error("every split count must be an integer of at least 640")
    if not math.isclose(config.pii_free_hard_negative_ratio, 0.40, abs_tol=1e-12):
        raise DomainRandomizedV72Error("v7.2 pins hard negatives to 40 percent")
    if not math.isclose(config.natural_positive_ratio, 0.75, abs_tol=1e-12):
        raise DomainRandomizedV72Error("v7.2 pins natural positives to 75 percent")
    if not math.isclose(config.counterfactual_share_of_hard_negative, 0.625, abs_tol=1e-12):
        raise DomainRandomizedV72Error("counterfactual share must remain 0.625")
    if tuple(config.positive_composition) != POSITIVE_COMPOSITIONS or not math.isclose(
        sum(config.positive_composition.values()), 1.0, abs_tol=1e-12
    ):
        raise DomainRandomizedV72Error("positive composition contract changed")
    expected_weights = {
        "natural_dialogue": 0.25,
        "natural_narrative": 0.25,
        "natural_workflow": 0.125,
        "natural_multi_entity": 0.125,
        "structured_label_first": 0.125,
        "structured_value_first": 0.125,
    }
    if dict(config.positive_composition) != expected_weights:
        raise DomainRandomizedV72Error("positive composition weights changed")
    natural_weight = sum(
        config.positive_composition[name] for name in NATURAL_POSITIVE_COMPOSITIONS
    )
    if not math.isclose(natural_weight, config.natural_positive_ratio, abs_tol=1e-12):
        raise DomainRandomizedV72Error("natural-positive weight drifted")
    if config.cue_placements != CUE_PLACEMENTS:
        raise DomainRandomizedV72Error("cue-placement inventory changed")
    if config.domains != DOMAINS:
        raise DomainRandomizedV72Error("domain inventory changed")
    if set(_LABEL_ALIASES) != set(CORE_LABELS) or any(
        len(set(aliases)) != 3 for aliases in _LABEL_ALIASES.values()
    ):
        raise DomainRandomizedV72Error("three-alias core24 inventory is incomplete")
    if set(_SPLIT_SURFACES) != set(SPLITS):
        raise DomainRandomizedV72Error("split-specific surface inventory is incomplete")
    for split in SPLITS:
        for placement in ("left", "right", "bilateral"):
            patterns = _SPLIT_SURFACES[split].get(placement)
            if not isinstance(patterns, tuple) or len(patterns) != 4 or len(set(patterns)) != 4:
                raise DomainRandomizedV72Error(
                    f"{split}/{placement} must have four unique surface patterns"
                )
    for split, count in config.split_counts.items():
        hard_negative = round(count * config.pii_free_hard_negative_ratio)
        positive = count - hard_negative
        if hard_negative + positive != count:
            raise DomainRandomizedV72Error(f"{split} ratio allocation drifted")
        composition_counts = _allocate(positive, config.positive_composition)
        if any(value % len(CORE_LABELS) for value in composition_counts.values()):
            raise DomainRandomizedV72Error(
                f"{split} composition counts must divide exactly across core24"
            )


def load_domain_randomized_v7_2_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    repository_root: Path | None = None,
) -> DomainRandomizedV72Config:
    """Load a closed recipe without resolving any corpus or benchmark input."""

    root = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    path = (
        (config_path if config_path.is_absolute() else root / config_path)
        .expanduser()
        .resolve(strict=True)
    )
    taxonomy_path = (root / DEFAULT_TAXONOMY_PATH).resolve(strict=True)
    raw_bytes = path.read_bytes()
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise DomainRandomizedV72Error("invalid domain-randomized v7.2 YAML") from exc
    top = _require_mapping(
        raw,
        name="config",
        keys={
            "schema_version",
            "dataset",
            "generation",
            "diversity_gates",
            "usage_policy",
            "clean_room",
            "isolation",
        },
    )
    if top["schema_version"] != 1:
        raise DomainRandomizedV72Error("config.schema_version must be 1")

    dataset = _require_mapping(
        top["dataset"],
        name="dataset",
        keys={"id", "version", "license", "language", "publication_pool"},
    )
    if dict(dataset) != {
        "id": DATASET_ID,
        "version": DATASET_VERSION,
        "license": "Apache-2.0",
        "language": "zh-Hans-CN",
        "publication_pool": DataPool.PUBLIC_RELEASE.value,
    }:
        raise DomainRandomizedV72Error("dataset identity changed")

    generation = _require_mapping(
        top["generation"],
        name="generation",
        keys={
            "seed",
            "splits",
            "generation_order",
            "pii_free_hard_negative_ratio",
            "natural_positive_ratio",
            "counterfactual_share_of_hard_negative",
            "positive_composition",
            "cue_placements",
            "domains",
        },
    )
    split_counts = _require_mapping(
        generation["splits"], name="generation.splits", keys=set(SPLITS)
    )
    positive_composition = _require_mapping(
        generation["positive_composition"],
        name="generation.positive_composition",
        keys=set(POSITIVE_COMPOSITIONS),
    )
    if generation["generation_order"] != list(GENERATION_ORDER):
        raise DomainRandomizedV72Error("independent_test must be generated first")
    if not isinstance(generation["seed"], int) or isinstance(generation["seed"], bool):
        raise DomainRandomizedV72Error("generation.seed must be an integer")
    numeric = (
        generation["pii_free_hard_negative_ratio"],
        generation["natural_positive_ratio"],
        generation["counterfactual_share_of_hard_negative"],
        *positive_composition.values(),
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric):
        raise DomainRandomizedV72Error("generation ratios must be numeric")
    if generation["cue_placements"] != list(CUE_PLACEMENTS):
        raise DomainRandomizedV72Error("cue placements changed")
    if generation["domains"] != list(DOMAINS):
        raise DomainRandomizedV72Error("domains changed")

    diversity = _require_mapping(
        top["diversity_gates"],
        name="diversity_gates",
        keys={
            "aliases_per_label",
            "surface_variants_per_cue_placement",
            "minimum_natural_semantic_skeletons_per_label",
            "maximum_cue_placement_count_gap_per_label",
            "minimum_right_visible_cue_ratio",
            "maximum_shortcut_cue_ratio_of_hard_negative",
        },
    )
    minimum_skeletons = _require_mapping(
        diversity["minimum_natural_semantic_skeletons_per_label"],
        name="minimum_natural_semantic_skeletons_per_label",
        keys=set(SPLITS),
    )
    expected_diversity = {
        "aliases_per_label": 3,
        "surface_variants_per_cue_placement": 4,
        "minimum_natural_semantic_skeletons_per_label": {
            "train": 36,
            "dev": 24,
            "independent_test": 24,
        },
        "maximum_cue_placement_count_gap_per_label": 1,
        "minimum_right_visible_cue_ratio": 0.666666,
        "maximum_shortcut_cue_ratio_of_hard_negative": 0.05,
    }
    if {
        **diversity,
        "minimum_natural_semantic_skeletons_per_label": dict(minimum_skeletons),
    } != expected_diversity:
        raise DomainRandomizedV72Error("diversity gates changed")

    usage = _require_mapping(top["usage_policy"], name="usage_policy", keys=set(SPLITS))
    usage_keys = {
        "evaluation_only",
        "public_weight_training_allowed",
        "gradient_training_allowed",
        "checkpoint_or_model_selection_allowed",
        "threshold_or_fusion_selection_allowed",
        "final_metric_claim_allowed",
        "candidate_must_be_frozen_before_content_access",
        "one_shot_only",
    }
    parsed_usage = {
        split: dict(_require_mapping(usage[split], name=f"usage_policy.{split}", keys=usage_keys))
        for split in SPLITS
    }
    if any(
        not isinstance(value, bool) for policy in parsed_usage.values() for value in policy.values()
    ):
        raise DomainRandomizedV72Error("usage policies must contain booleans")
    expected_usage = {
        "train": {
            "evaluation_only": False,
            "public_weight_training_allowed": True,
            "gradient_training_allowed": True,
            "checkpoint_or_model_selection_allowed": False,
            "threshold_or_fusion_selection_allowed": False,
            "final_metric_claim_allowed": False,
            "candidate_must_be_frozen_before_content_access": False,
            "one_shot_only": False,
        },
        "dev": {
            "evaluation_only": True,
            "public_weight_training_allowed": False,
            "gradient_training_allowed": False,
            "checkpoint_or_model_selection_allowed": True,
            "threshold_or_fusion_selection_allowed": True,
            "final_metric_claim_allowed": False,
            "candidate_must_be_frozen_before_content_access": False,
            "one_shot_only": False,
        },
        "independent_test": {
            "evaluation_only": True,
            "public_weight_training_allowed": False,
            "gradient_training_allowed": False,
            "checkpoint_or_model_selection_allowed": False,
            "threshold_or_fusion_selection_allowed": False,
            "final_metric_claim_allowed": True,
            "candidate_must_be_frozen_before_content_access": True,
            "one_shot_only": True,
        },
    }
    if parsed_usage != expected_usage:
        raise DomainRandomizedV72Error("split usage policy changed")

    clean_room = _require_mapping(
        top["clean_room"],
        name="clean_room",
        keys={
            "benchmark_rows_read",
            "benchmark_predictions_read",
            "benchmark_templates_read",
            "prior_dataset_rows_read",
            "prior_training_rows_read",
            "external_dataset_inputs",
            "repository_authored_grammars_only",
        },
    )
    expected_clean_room = {
        "benchmark_rows_read": False,
        "benchmark_predictions_read": False,
        "benchmark_templates_read": False,
        "prior_dataset_rows_read": False,
        "prior_training_rows_read": False,
        "external_dataset_inputs": [],
        "repository_authored_grammars_only": True,
    }
    if dict(clean_room) != expected_clean_room:
        raise DomainRandomizedV72Error("clean-room policy must reject all corpus inputs")

    isolation = _require_mapping(
        top["isolation"],
        name="isolation",
        keys={
            "split_specific_grammar_namespaces",
            "split_specific_template_family_namespaces",
            "split_specific_source_namespaces",
            "global_nonrepeating_value_stream",
            "pairwise_all_splits",
            "required_zero_collision_groups",
        },
    )
    if any(
        isolation[name] is not True
        for name in (
            "split_specific_grammar_namespaces",
            "split_specific_template_family_namespaces",
            "split_specific_source_namespaces",
            "global_nonrepeating_value_stream",
            "pairwise_all_splits",
        )
    ) or isolation["required_zero_collision_groups"] != list(GROUP_KEYS):
        raise DomainRandomizedV72Error("split-isolation policy changed")

    config = DomainRandomizedV72Config(
        config_path=path,
        config_sha256=_sha256_bytes(raw_bytes),
        repository_root=root,
        taxonomy_path=taxonomy_path,
        seed=generation["seed"],
        split_counts={split: int(split_counts[split]) for split in SPLITS},
        pii_free_hard_negative_ratio=float(generation["pii_free_hard_negative_ratio"]),
        natural_positive_ratio=float(generation["natural_positive_ratio"]),
        counterfactual_share_of_hard_negative=float(
            generation["counterfactual_share_of_hard_negative"]
        ),
        positive_composition={
            name: float(positive_composition[name]) for name in POSITIVE_COMPOSITIONS
        },
        cue_placements=tuple(generation["cue_placements"]),
        domains=tuple(generation["domains"]),
        diversity_gates={
            **dict(diversity),
            "minimum_natural_semantic_skeletons_per_label": dict(minimum_skeletons),
        },
        usage_policy=parsed_usage,
        clean_room=dict(clean_room),
    )
    _validate_runtime_config(config)
    return config


def _seeded_rng(seed: int, split: str, purpose: str) -> random.Random:
    digest = _sha256_bytes(f"{seed}\0{split}\0{purpose}".encode())
    return random.Random(int(digest[:16], 16))


def _cycled(items: Sequence[Any], count: int, rng: random.Random) -> list[Any]:
    selected: list[Any] = []
    while len(selected) < count:
        cycle = list(items)
        rng.shuffle(cycle)
        selected.extend(cycle[: count - len(selected)])
    return selected


def _positive_label_composition_pairs(
    config: DomainRandomizedV72Config,
    *,
    split: str,
    positive_count: int,
) -> list[tuple[str, str]]:
    """Allocate every positive composition exactly and balance primary labels."""

    composition_counts = _allocate(positive_count, config.positive_composition)
    pairs: list[tuple[str, str]] = []
    for composition in POSITIVE_COMPOSITIONS:
        count = composition_counts[composition]
        if count % len(CORE_LABELS):
            raise DomainRandomizedV72Error("composition is not exactly core24-balanced")
        labels = list(CORE_LABELS) * (count // len(CORE_LABELS))
        _seeded_rng(config.seed, split, f"labels:{composition}").shuffle(labels)
        pairs.extend((composition, label) for label in labels)
    _seeded_rng(config.seed, split, "positive-document-order").shuffle(pairs)
    return pairs


def _context_spec(
    *,
    split: str,
    label: str,
    placeholder: str,
    occurrence: int,
) -> dict[str, Any]:
    label_index = CORE_LABELS.index(label)
    # Put any non-divisible remainder on right-visible placements.  This keeps
    # right-only + bilateral at or above two thirds without unbalancing the
    # three placement counts by more than one.
    placement = _CUE_ASSIGNMENT_ORDER[occurrence % len(_CUE_ASSIGNMENT_ORDER)]
    alias_index = (occurrence // len(CUE_PLACEMENTS) + label_index) % 3
    surface_variant = (occurrence // (len(CUE_PLACEMENTS) * 3) + label_index) % 4
    semantic_family = (
        f"v7.2:{split}:{label}:{placement}:alias-{alias_index}:surface-{surface_variant}"
    )
    return {
        "placeholder": placeholder,
        "label": label,
        "cue_placement": placement,
        "alias_index": alias_index,
        "surface_variant": surface_variant,
        "semantic_family": semantic_family,
    }


def _semantic_clause(split: str, spec: Mapping[str, Any]) -> str:
    label = str(spec["label"])
    alias_index = int(spec["alias_index"])
    variant = int(spec["surface_variant"])
    placement = str(spec["cue_placement"])
    placeholder = f"<<{spec['placeholder']}>>"
    alias = _LABEL_ALIASES[label][alias_index]
    surface_key = {
        "left_only": "left",
        "right_only": "right",
        "bilateral": "bilateral",
    }[placement]
    pattern = _SPLIT_SURFACES[split][surface_key][variant]
    clause = pattern.format(alias=alias, value=placeholder)
    before, after = clause.split(placeholder, 1)
    has_left = alias in before
    has_right = alias in after
    expected = {
        "left_only": (True, False),
        "right_only": (False, True),
        "bilateral": (True, True),
    }[placement]
    if (has_left, has_right) != expected:
        raise DomainRandomizedV72Error("semantic cue placement is not textually true")
    return clause


def _positive_features(
    *,
    index: int,
    composition: str,
    primary_label: str,
    secondary_label: str | None,
    occurrence: int,
    context_specs: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    label_index = CORE_LABELS.index(primary_label)
    return {
        "composition": composition,
        "frame_variant": str((index + occurrence + label_index) % 4),
        "field_position": FIELD_POSITIONS[(occurrence + label_index) % 3],
        "length_band": LENGTH_BANDS[(occurrence // 3 + label_index) % 3],
        "noise_style": NOISE_STYLES[(occurrence // 9 + label_index) % 3],
        "punctuation_variant": str((occurrence + label_index * 2) % 4),
        "primary_label": primary_label,
        "secondary_label": secondary_label or "none",
        "context_signature": "+".join(str(spec["semantic_family"]) for spec in context_specs)
        or "structured",
    }


def _negative_features(
    *,
    index: int,
    composition: str,
    subject: str,
) -> dict[str, str]:
    return {
        "composition": composition,
        "frame_variant": str(index % 4),
        "field_position": FIELD_POSITIONS[index % 3],
        "length_band": LENGTH_BANDS[(index // 3) % 3],
        "noise_style": NOISE_STYLES[(index // 9) % 3],
        "punctuation_variant": str(index % 4),
        "primary_label": subject,
        "secondary_label": "none",
        "context_signature": f"negative:{subject}:{index % 12}",
    }


def _arrange_segments(
    head: str,
    detail: str,
    tail: str,
    *,
    position: str,
) -> tuple[str, str, str]:
    if position == "early":
        return detail, head, tail
    if position == "middle":
        return head, detail, tail
    if position == "late":
        return head, tail, detail
    raise DomainRandomizedV72Error("unknown field position")


def _apply_noise(body: str, noise_style: str, *, split: str) -> str:
    if noise_style == "clean":
        return body
    marker = _SPLIT_SURFACES[split]["noise"].get(noise_style)
    if marker is None:
        raise DomainRandomizedV72Error("unknown noise style")
    if noise_style == "ocr":
        return f"{marker[0]}｜{body}｜{marker[1]}"
    return f"{marker[0]}，{body}，嗯，{marker[1]}"


def _natural_body(
    *,
    split: str,
    composition: str,
    domain: str,
    features: Mapping[str, str],
    context_specs: Sequence[Mapping[str, Any]],
) -> str:
    if composition not in NATURAL_POSITIVE_COMPOSITIONS or not context_specs:
        raise DomainRandomizedV72Error("natural renderer contract failed")
    surface = _SPLIT_SURFACES[split]
    frame = int(features["frame_variant"])
    speaker, staff = surface["roles"][frame]
    opener = surface["openers"][frame]
    intent = surface["intents"][(frame + CORE_LABELS.index(context_specs[0]["label"])) % 6]
    closer = surface["closers"][frame]
    clauses = [_semantic_clause(split, spec) for spec in context_specs]

    if composition == "natural_dialogue":
        detail = f"{speaker}：{clauses[0]}。{staff}复述后询问是否准确。"
    elif composition == "natural_narrative":
        detail = f"{speaker}为了{intent}补充道：“{clauses[0]}。”"
    elif composition == "natural_workflow":
        detail = f"记录员在流转说明中写下“{clauses[0]}”，并保留说话目的。"
    elif composition == "natural_multi_entity":
        if len(clauses) != 2:
            raise DomainRandomizedV72Error("multi-entity natural row needs two clauses")
        connector = ("随后又补充", "接着说明", "同一句里还提到", "停顿后继续说")[frame]
        detail = f"{speaker}先说“{clauses[0]}”，{connector}“{clauses[1]}”。"
    else:
        raise DomainRandomizedV72Error("unknown natural composition")

    head = f"{_DOMAIN_TITLES[domain]}场景中，{staff}说明：{opener}，当前要{intent}。"
    tail = f"{surface['fillers'][features['length_band']]}{closer}。"
    segments = _arrange_segments(
        head,
        detail,
        tail,
        position=features["field_position"],
    )
    punctuation = _PUNCTUATION_STYLES[int(features["punctuation_variant"])]
    body = punctuation[1].join(segments) + punctuation[2]
    return _apply_noise(body, features["noise_style"], split=split)


def _structured_body(
    *,
    split: str,
    composition: str,
    domain: str,
    primary_label: str,
    occurrence: int,
    features: Mapping[str, str],
) -> str:
    if composition not in {"structured_label_first", "structured_value_first"}:
        raise DomainRandomizedV72Error("unknown structured composition")
    alias = _LABEL_ALIASES[primary_label][occurrence % 3]
    punctuation = _PUNCTUATION_STYLES[int(features["punctuation_variant"])]
    assignment, separator, ending = punctuation
    if composition == "structured_label_first":
        detail = f"{alias}{assignment}<<VALUE_1>>"
    else:
        detail = f"<<VALUE_1>>{assignment}{alias}"
    surface = _SPLIT_SURFACES[split]
    frame = int(features["frame_variant"])
    head = f"{_DOMAIN_TITLES[domain]}结构化摘录{assignment}{surface['openers'][frame]}"
    tail = surface["fillers"][features["length_band"]]
    body = (
        separator.join(_arrange_segments(head, detail, tail, position=features["field_position"]))
        + ending
    )
    return _apply_noise(body, features["noise_style"], split=split)


def _family_name(
    split: str,
    composition: str,
    features: Mapping[str, str],
) -> str:
    feature_hash = _canonical_json_hash(dict(sorted(features.items())))[:20]
    return f"v7.2-{split}-{composition}-{feature_hash}"


def _positive_template(
    *,
    split: str,
    composition: str,
    domain: str,
    primary_label: str,
    secondary_label: str | None,
    occurrence: int,
    features: Mapping[str, str],
    context_specs: Sequence[Mapping[str, Any]],
) -> SyntheticTemplate:
    if composition in NATURAL_POSITIVE_COMPOSITIONS:
        body = _natural_body(
            split=split,
            composition=composition,
            domain=domain,
            features=features,
            context_specs=context_specs,
        )
    else:
        body = _structured_body(
            split=split,
            composition=composition,
            domain=domain,
            primary_label=primary_label,
            occurrence=occurrence,
            features=features,
        )
    placeholders = [
        PlaceholderSpec(
            key="VALUE_1",
            generator_key=primary_label,
            label=primary_label,
            risk_tier=_risk_tier(primary_label),
        )
    ]
    if secondary_label is not None:
        placeholders.append(
            PlaceholderSpec(
                key="VALUE_2",
                generator_key=secondary_label,
                label=secondary_label,
                risk_tier=_risk_tier(secondary_label),
            )
        )
    family = _family_name(split, composition, features)
    return SyntheticTemplate(
        template_id=f"{family}-{domain}-{primary_label.lower()}",
        family=family,
        domain=domain,
        scene=composition,
        body=body,
        placeholders=tuple(placeholders),
        origin_kind="repository_authored_clean_room_v7_2",
    )


def _counterfactual_template(
    *,
    split: str,
    domain: str,
    target_label: str,
    alias_index: int,
    features: Mapping[str, str],
    shortcut_cue: bool,
) -> SyntheticTemplate:
    alias = _LABEL_ALIASES[target_label][alias_index]
    context = _COUNTERFACTUAL_CONTEXTS[target_label]
    if shortcut_cue:
        body = _EXPLICIT_NEGATIVE_FRAMES[split].format(
            value="<<VALUE_1>>",
            context=context,
            alias=alias,
        )
    else:
        frames = _NEGATIVE_FRAMES[split]
        body = frames[int(features["frame_variant"]) % len(frames)].format(
            value="<<VALUE_1>>",
            context=context,
        )
    body = (
        f"{_DOMAIN_TITLES[domain]}：{body}"
        f"{_SPLIT_SURFACES[split]['fillers'][features['length_band']]}"
    )
    body = _apply_noise(body, features["noise_style"], split=split)
    family = _family_name(split, "counterfactual", features)
    return SyntheticTemplate(
        template_id=f"{family}-{domain}-{target_label.lower()}",
        family=family,
        domain=domain,
        scene="counterfactual",
        body=body,
        placeholders=(PlaceholderSpec(key="VALUE_1", generator_key=target_label, label=None),),
        hard_negative=True,
        origin_kind="repository_authored_clean_room_v7_2",
    )


def _pii_free_template(
    *,
    split: str,
    domain: str,
    generator_key: str,
    description: str,
    features: Mapping[str, str],
) -> SyntheticTemplate:
    body = _PLAIN_NEGATIVE_FRAMES[split].format(
        value="<<VALUE_1>>",
        description=description,
    )
    body = (
        f"{_DOMAIN_TITLES[domain]}：{body}"
        f"{_SPLIT_SURFACES[split]['fillers'][features['length_band']]}"
    )
    body = _apply_noise(body, features["noise_style"], split=split)
    family = _family_name(split, "pii_free", features)
    return SyntheticTemplate(
        template_id=f"{family}-{domain}-{generator_key.lower()}",
        family=family,
        domain=domain,
        scene="pii_free",
        body=body,
        placeholders=(PlaceholderSpec(key="VALUE_1", generator_key=generator_key, label=None),),
        hard_negative=True,
        origin_kind="repository_authored_clean_room_v7_2",
    )


def _semantic_skeleton(template: SyntheticTemplate) -> str:
    value = _PLACEHOLDER_RE.sub("", template.body)
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def _unique_values(
    factory: SyntheticValueFactory,
    template: SyntheticTemplate,
    seen_value_groups: set[str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    values: dict[str, str] = {}
    groups: list[str] = []
    for spec in template.placeholders:
        for _ in range(200):
            value = generate_value(factory, spec.generator_key, value_variant=spec.value_variant)
            group = stable_text_hash(value)
            if group not in seen_value_groups:
                break
        else:
            raise DomainRandomizedV72Error("global non-repeating value stream exhausted")
        seen_value_groups.add(group)
        values[spec.key] = value
        groups.append(group)
    return values, tuple(sorted(groups))


def _make_record(
    *,
    config: DomainRandomizedV72Config,
    split: str,
    index: int,
    composition: str,
    template: SyntheticTemplate,
    factory: SyntheticValueFactory,
    seen_value_groups: set[str],
    grammar_features: Mapping[str, str],
    context_specs: Sequence[Mapping[str, Any]] = (),
    counterfactual_target_label: str | None = None,
    shortcut_cue: bool = False,
) -> DocumentRecord:
    values, all_value_groups = _unique_values(factory, template, seen_value_groups)
    rendered = render_template(template, values)
    validator_results = dict(rendered.validator_results)
    if counterfactual_target_label is not None:
        result = validate_entity_value(counterfactual_target_label, values["VALUE_1"])
        if result is not None:
            validator_results[f"counterfactual:{counterfactual_target_label}"] = result.reason
            if not result.valid:
                raise DomainRandomizedV72Error("counterfactual structured value failed validation")
    if not rendered.validators_passed:
        raise DomainRandomizedV72Error("generated structured value failed validation")
    observed_shortcut = any(token in rendered.text for token in _SHORTCUT_TOKENS)
    if observed_shortcut is not shortcut_cue:
        raise DomainRandomizedV72Error("shortcut-cue declaration drifted")

    natural_positive = bool(rendered.entities and composition in NATURAL_POSITIVE_COMPOSITIONS)
    if natural_positive is not bool(context_specs):
        raise DomainRandomizedV72Error("natural context metadata drifted")
    if context_specs:
        expected_placeholders = {
            str(entity.metadata.get("placeholder")) for entity in rendered.entities
        }
        observed_placeholders = {str(spec["placeholder"]) for spec in context_specs}
        if expected_placeholders != observed_placeholders:
            raise DomainRandomizedV72Error("context specs do not cover rendered entities")

    grammar_group = _group_hash("v7.2-grammar", template.family)
    skeleton_group = _group_hash("v7.2-semantic-skeleton", _semantic_skeleton(template))
    source_group = _group_hash("v7.2-source", f"{split}|{template.domain}")
    template_group = _group_hash("v7.2-template-family", template.family)
    doc_id = _group_hash(
        "v7.2-document",
        f"{config.seed}|{split}|{index}|{template.template_id}|{rendered.text}",
    )
    policy = config.usage_policy[split]
    pii_free = not rendered.entities
    provenance = Provenance(
        source_id=f"{DATASET_ID}_{DATASET_VERSION}_{split}",
        source_kind="repository_authored_clean_room_synthetic",
        source_revision=f"sha256:{config.config_sha256}",
        license="Apache-2.0",
        synthetic=True,
        evaluation_only=policy["evaluation_only"],
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=config.seed,
        template_id=template.template_id,
        metadata={
            "contains_real_personal_data": False,
            "benchmark_content_used": False,
            "prior_dataset_content_used": False,
            "external_dataset_inputs": [],
            "grammar_origin": "repository_authored_clean_room_v7_2",
            "split_specific_namespace": split,
            "weight_training_allowed_by_protocol": policy["public_weight_training_allowed"],
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.HARD_NEGATIVE if pii_free else QualityTier.SYNTHETIC_VALIDATED,
        quality_gate=True,
        validators_passed=True,
        review_status=(
            "clean_room_natural_context_and_span_validated"
            if natural_positive
            else (
                "clean_room_contextual_hard_negative_validated"
                if pii_free
                else "clean_room_structured_span_validated"
            )
        ),
        confidence=1.0,
        metadata={"validator_results": validator_results},
    )
    entity_value_groups = tuple(
        sorted(stable_text_hash(entity.text) for entity in rendered.entities)
    )
    return DocumentRecord.create(
        doc_id=doc_id,
        text=rendered.text,
        entities=rendered.entities,
        language="zh-Hans-CN",
        domain=template.domain,
        scene=composition,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=policy["public_weight_training_allowed"],
        split=split,
        template_group=template_group,
        entity_value_groups=entity_value_groups,
        generator_version=GENERATOR_VERSION,
        metadata={
            "composition": composition,
            "pii_free": pii_free,
            "hard_negative": pii_free,
            "counterfactual": composition == "counterfactual",
            "natural_positive": natural_positive,
            "primary_label": grammar_features["primary_label"],
            "secondary_label": grammar_features["secondary_label"],
            "context_specs": [dict(spec) for spec in context_specs],
            "contains_real_personal_data": False,
            "benchmark_content_used": False,
            "prior_dataset_content_used": False,
            "domain_randomized": True,
            "grammar_group": grammar_group,
            "template_family": template.family,
            "semantic_skeleton_group": skeleton_group,
            "source_group": source_group,
            "grammar_features": dict(grammar_features),
            "shortcut_cue": shortcut_cue,
            "all_value_groups": list(all_value_groups),
            "non_entity_value_groups": list(all_value_groups) if pii_free else [],
            "usage_policy": dict(policy),
        },
    )


def _split_records(
    config: DomainRandomizedV72Config,
    *,
    split: str,
    factory: SyntheticValueFactory,
    seen_value_groups: set[str],
) -> list[DocumentRecord]:
    count = config.split_counts[split]
    hard_negative_count = round(count * config.pii_free_hard_negative_ratio)
    positive_count = count - hard_negative_count
    counterfactual_count = round(hard_negative_count * config.counterfactual_share_of_hard_negative)
    plain_count = hard_negative_count - counterfactual_count
    positive_pairs = _positive_label_composition_pairs(
        config,
        split=split,
        positive_count=positive_count,
    )
    records: list[DocumentRecord] = []
    primary_occurrences: Counter[str] = Counter()
    composition_label_occurrences: Counter[tuple[str, str]] = Counter()
    context_occurrences: Counter[str] = Counter()
    split_offset = {"train": 0, "dev": 1, "independent_test": 2}[split]

    for index, (composition, primary_label) in enumerate(positive_pairs):
        occurrence = primary_occurrences[primary_label]
        primary_occurrences[primary_label] += 1
        composition_occurrence = composition_label_occurrences[(composition, primary_label)]
        composition_label_occurrences[(composition, primary_label)] += 1
        label_index = CORE_LABELS.index(primary_label)
        domain = config.domains[
            (label_index * 7 + occurrence * 5 + split_offset * 3) % len(config.domains)
        ]
        secondary_label = None
        if composition == "natural_multi_entity":
            pair_offset = (5, 7, 11)[composition_occurrence % 3]
            secondary_label = CORE_LABELS[(label_index + pair_offset) % len(CORE_LABELS)]

        context_specs: list[dict[str, Any]] = []
        if composition in NATURAL_POSITIVE_COMPOSITIONS:
            context_specs.append(
                _context_spec(
                    split=split,
                    label=primary_label,
                    placeholder="VALUE_1",
                    occurrence=context_occurrences[primary_label],
                )
            )
            context_occurrences[primary_label] += 1
            if secondary_label is not None:
                context_specs.append(
                    _context_spec(
                        split=split,
                        label=secondary_label,
                        placeholder="VALUE_2",
                        occurrence=context_occurrences[secondary_label],
                    )
                )
                context_occurrences[secondary_label] += 1
        features = _positive_features(
            index=index,
            composition=composition,
            primary_label=primary_label,
            secondary_label=secondary_label,
            occurrence=occurrence,
            context_specs=context_specs,
        )
        template = _positive_template(
            split=split,
            composition=composition,
            domain=domain,
            primary_label=primary_label,
            secondary_label=secondary_label,
            occurrence=occurrence,
            features=features,
            context_specs=context_specs,
        )
        records.append(
            _make_record(
                config=config,
                split=split,
                index=index,
                composition=composition,
                template=template,
                factory=factory,
                seen_value_groups=seen_value_groups,
                grammar_features=features,
                context_specs=context_specs,
            )
        )

    target_labels = _cycled(
        CORE_LABELS,
        counterfactual_count,
        _seeded_rng(config.seed, split, "counterfactual-labels"),
    )
    for offset, target_label in enumerate(target_labels):
        index = positive_count + offset
        label_index = CORE_LABELS.index(target_label)
        domain = config.domains[
            (label_index * 5 + offset * 7 + split_offset * 2) % len(config.domains)
        ]
        features = _negative_features(
            index=index,
            composition="counterfactual",
            subject=target_label,
        )
        shortcut_cue = offset % 20 == 0
        template = _counterfactual_template(
            split=split,
            domain=domain,
            target_label=target_label,
            alias_index=(offset + label_index) % 3,
            features=features,
            shortcut_cue=shortcut_cue,
        )
        records.append(
            _make_record(
                config=config,
                split=split,
                index=index,
                composition="counterfactual",
                template=template,
                factory=factory,
                seen_value_groups=seen_value_groups,
                grammar_features=features,
                counterfactual_target_label=target_label,
                shortcut_cue=shortcut_cue,
            )
        )

    free_specs = _cycled(
        _PII_FREE_GENERATORS,
        plain_count,
        _seeded_rng(config.seed, split, "pii-free-generators"),
    )
    for offset, (generator_key, description) in enumerate(free_specs):
        index = positive_count + counterfactual_count + offset
        domain = config.domains[
            (offset * 5 + _PII_FREE_GENERATORS.index((generator_key, description)) + split_offset)
            % len(config.domains)
        ]
        features = _negative_features(
            index=index,
            composition="pii_free",
            subject=generator_key,
        )
        template = _pii_free_template(
            split=split,
            domain=domain,
            generator_key=generator_key,
            description=description,
            features=features,
        )
        records.append(
            _make_record(
                config=config,
                split=split,
                index=index,
                composition="pii_free",
                template=template,
                factory=factory,
                seen_value_groups=seen_value_groups,
                grammar_features=features,
            )
        )
    if len(records) != count:
        raise DomainRandomizedV72Error(f"{split} generation count drifted")
    _seeded_rng(config.seed, split, "whole-split-order").shuffle(records)
    return records


def _record_groups(record: DocumentRecord) -> dict[str, set[str]]:
    values = record.metadata.get("all_value_groups")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise DomainRandomizedV72Error("record all_value_groups metadata is malformed")
    scalar_names = {
        "grammar_groups": "grammar_group",
        "template_families": "template_family",
        "semantic_skeletons": "semantic_skeleton_group",
        "source_groups": "source_group",
    }
    groups = {
        "value_groups": set(values),
        "text_hashes": {stable_text_hash(record.text)},
        "doc_ids": {record.doc_id},
    }
    for group_name, metadata_name in scalar_names.items():
        value = record.metadata.get(metadata_name)
        if not isinstance(value, str) or not value:
            raise DomainRandomizedV72Error(f"record {metadata_name} is malformed")
        groups[group_name] = {value}
    return groups


def audit_domain_randomized_v7_2_records(
    records: Sequence[DocumentRecord],
    config: DomainRandomizedV72Config,
) -> dict[str, Any]:
    """Fail closed on balance, natural context, policy, and three-way isolation."""

    split_groups = {split: {key: set() for key in GROUP_KEYS} for split in SPLITS}
    counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    natural_positive_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    counterfactual_counts: Counter[str] = Counter()
    shortcut_counts: Counter[str] = Counter()
    compositions = {split: Counter[str]() for split in SPLITS}
    domains = {split: Counter[str]() for split in SPLITS}
    labels = {split: Counter[str]() for split in SPLITS}
    primary_labels = {
        split: {composition: Counter[str]() for composition in POSITIVE_COMPOSITIONS}
        for split in SPLITS
    }
    template_families = {split: {"positive": set(), "hard_negative": set()} for split in SPLITS}
    semantic_skeletons = {split: {"positive": set(), "hard_negative": set()} for split in SPLITS}
    natural_context = {
        split: {
            label: {
                "records": 0,
                "cue_placements": Counter[str](),
                "aliases": set(),
                "surface_variants": set(),
                "semantic_families": set(),
                "semantic_skeletons": set(),
                "right_visible": 0,
                "minimum_right_context_chars": None,
            }
            for label in CORE_LABELS
        }
        for split in SPLITS
    }
    feature_coverage = {
        split: {
            label: {
                "domains": set(),
                "positions": set(),
                "lengths": set(),
                "noise": set(),
            }
            for label in CORE_LABELS
        }
        for split in SPLITS
    }
    all_value_occurrences = 0

    context_keys = {
        "placeholder",
        "label",
        "cue_placement",
        "alias_index",
        "surface_variant",
        "semantic_family",
    }
    for record in records:
        record.validate()
        split = record.split
        if split not in SPLITS:
            raise DomainRandomizedV72Error("record has an unknown split")
        policy = config.usage_policy[split]
        expected_pool = (
            DataPool.EVALUATION_ONLY if policy["evaluation_only"] else DataPool.PUBLIC_RELEASE
        )
        if (
            record.data_pool is not expected_pool
            or record.public_weight_training_allowed is not policy["public_weight_training_allowed"]
            or record.provenance.evaluation_only is not policy["evaluation_only"]
        ):
            raise DomainRandomizedV72Error(f"{split} record escaped its data-pool policy")
        if (
            record.provenance.metadata.get("benchmark_content_used") is not False
            or record.provenance.metadata.get("prior_dataset_content_used") is not False
            or record.provenance.metadata.get("external_dataset_inputs") != []
        ):
            raise DomainRandomizedV72Error("record lacks clean-room provenance")
        if record.metadata.get("usage_policy") != dict(policy):
            raise DomainRandomizedV72Error("record usage policy drifted")

        counts[split] += 1
        domains[split][record.domain] += 1
        labels[split].update(entity.label for entity in record.entities)
        composition = record.metadata.get("composition")
        if composition not in COMPOSITIONS:
            raise DomainRandomizedV72Error("record has an unknown composition")
        compositions[split][composition] += 1
        features = record.metadata.get("grammar_features")
        if not isinstance(features, Mapping) or set(features) != set(GRAMMAR_FEATURE_KEYS):
            raise DomainRandomizedV72Error("grammar feature metadata is malformed")
        if any(not isinstance(value, str) or not value for value in features.values()):
            raise DomainRandomizedV72Error("grammar features must be non-empty strings")
        if features["composition"] != composition:
            raise DomainRandomizedV72Error("composition feature drifted")

        role = "positive" if record.entities else "hard_negative"
        skeleton_group = record.metadata.get("semantic_skeleton_group")
        template_family = record.metadata.get("template_family")
        if not isinstance(skeleton_group, str) or not isinstance(template_family, str):
            raise DomainRandomizedV72Error("grouping metadata is malformed")
        semantic_skeletons[split][role].add(skeleton_group)
        template_families[split][role].add(template_family)

        natural_positive = record.metadata.get("natural_positive")
        expected_natural = bool(record.entities and composition in NATURAL_POSITIVE_COMPOSITIONS)
        if not isinstance(natural_positive, bool) or natural_positive is not expected_natural:
            raise DomainRandomizedV72Error("natural-positive declaration drifted")
        contexts = record.metadata.get("context_specs")
        if not isinstance(contexts, list):
            raise DomainRandomizedV72Error("context specs must be an array")

        if record.entities:
            positive_counts[split] += 1
            natural_positive_counts[split] += int(expected_natural)
            primary_label = record.metadata.get("primary_label")
            if primary_label not in CORE_LABELS:
                raise DomainRandomizedV72Error("positive row lacks a core24 primary label")
            primary_labels[split][composition][primary_label] += 1
            for entity in record.entities:
                coverage = feature_coverage[split][entity.label]
                coverage["domains"].add(record.domain)
                coverage["positions"].add(features["field_position"])
                coverage["lengths"].add(features["length_band"])
                coverage["noise"].add(features["noise_style"])

            if expected_natural:
                by_placeholder = {
                    str(entity.metadata.get("placeholder")): entity for entity in record.entities
                }
                if len(contexts) != len(by_placeholder):
                    raise DomainRandomizedV72Error("natural contexts do not match entity count")
                for raw_spec in contexts:
                    if not isinstance(raw_spec, Mapping) or set(raw_spec) != context_keys:
                        raise DomainRandomizedV72Error("natural context spec is malformed")
                    placeholder = raw_spec["placeholder"]
                    label = raw_spec["label"]
                    placement = raw_spec["cue_placement"]
                    alias_index = raw_spec["alias_index"]
                    surface_variant = raw_spec["surface_variant"]
                    semantic_family = raw_spec["semantic_family"]
                    if (
                        placeholder not in by_placeholder
                        or label not in CORE_LABELS
                        or by_placeholder[placeholder].label != label
                        or placement not in CUE_PLACEMENTS
                        or isinstance(alias_index, bool)
                        or not isinstance(alias_index, int)
                        or alias_index not in range(3)
                        or isinstance(surface_variant, bool)
                        or not isinstance(surface_variant, int)
                        or surface_variant not in range(4)
                        or not isinstance(semantic_family, str)
                        or not semantic_family.startswith(f"v7.2:{split}:{label}:")
                    ):
                        raise DomainRandomizedV72Error("natural context values are invalid")
                    entity = by_placeholder[placeholder]
                    alias = _LABEL_ALIASES[label][alias_index]
                    has_left = alias in record.text[: entity.start]
                    has_right = alias in record.text[entity.end :]
                    expected_sides = {
                        "left_only": (True, False),
                        "right_only": (False, True),
                        "bilateral": (True, True),
                    }[placement]
                    if (has_left, has_right) != expected_sides:
                        raise DomainRandomizedV72Error("rendered semantic cue placement is false")
                    context = natural_context[split][label]
                    context["records"] += 1
                    context["cue_placements"][placement] += 1
                    context["aliases"].add(alias_index)
                    context["surface_variants"].add(surface_variant)
                    context["semantic_families"].add(semantic_family)
                    context["semantic_skeletons"].add(skeleton_group)
                    if placement in {"right_only", "bilateral"}:
                        context["right_visible"] += 1
                        right_chars = len(record.text) - entity.end
                        previous = context["minimum_right_context_chars"]
                        context["minimum_right_context_chars"] = (
                            right_chars if previous is None else min(previous, right_chars)
                        )
            elif contexts:
                raise DomainRandomizedV72Error(
                    "structured positive unexpectedly has natural context specs"
                )
        else:
            if contexts:
                raise DomainRandomizedV72Error("hard negative has context specs")
            hard_negative_counts[split] += 1
            counterfactual_counts[split] += int(composition == "counterfactual")
            shortcut = record.metadata.get("shortcut_cue")
            if not isinstance(shortcut, bool):
                raise DomainRandomizedV72Error("shortcut-cue metadata is malformed")
            shortcut_counts[split] += int(shortcut)
            observed_shortcut = any(token in record.text for token in _SHORTCUT_TOKENS)
            if observed_shortcut is not shortcut:
                raise DomainRandomizedV72Error("shortcut-cue text audit failed")

        record_groups = _record_groups(record)
        all_value_occurrences += len(record_groups["value_groups"])
        for key in GROUP_KEYS:
            split_groups[split][key].update(record_groups[key])

    if len(records) != sum(config.split_counts.values()) or any(
        counts[split] != config.split_counts[split] for split in SPLITS
    ):
        raise DomainRandomizedV72Error("record counts do not match the recipe")

    expected_composition: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        count = config.split_counts[split]
        expected_hard_negative = round(count * config.pii_free_hard_negative_ratio)
        expected_positive = count - expected_hard_negative
        expected_counterfactual = round(
            expected_hard_negative * config.counterfactual_share_of_hard_negative
        )
        positive_composition = _allocate(expected_positive, config.positive_composition)
        expected_composition[split] = {
            **positive_composition,
            "counterfactual": expected_counterfactual,
            "pii_free": expected_hard_negative - expected_counterfactual,
        }
        if hard_negative_counts[split] != expected_hard_negative:
            raise DomainRandomizedV72Error(f"{split} lost the 40 percent hard-negative ratio")
        if dict(compositions[split]) != expected_composition[split]:
            raise DomainRandomizedV72Error(f"{split} composition drifted")
        expected_natural = round(expected_positive * config.natural_positive_ratio)
        if natural_positive_counts[split] != expected_natural:
            raise DomainRandomizedV72Error(f"{split} natural-positive ratio drifted")
        if set(domains[split]) != set(DOMAINS):
            raise DomainRandomizedV72Error(f"{split} lost domain coverage")
        if set(labels[split]) != set(CORE_LABELS) or len(set(labels[split].values())) != 1:
            raise DomainRandomizedV72Error(f"{split} core24 entity counts are not exact")
        for composition in POSITIVE_COMPOSITIONS:
            per_label = primary_labels[split][composition]
            if set(per_label) != set(CORE_LABELS) or len(set(per_label.values())) != 1:
                raise DomainRandomizedV72Error(
                    f"{split}/{composition} primary labels are not exact"
                )
        if shortcut_counts[split] / hard_negative_counts[split] > float(
            config.diversity_gates["maximum_shortcut_cue_ratio_of_hard_negative"]
        ):
            raise DomainRandomizedV72Error(f"{split} overuses shortcut negative cues")

        minimum_skeletons = config.diversity_gates["minimum_natural_semantic_skeletons_per_label"][
            split
        ]
        for label in CORE_LABELS:
            context = natural_context[split][label]
            placement_counts = context["cue_placements"]
            if set(placement_counts) != set(CUE_PLACEMENTS):
                raise DomainRandomizedV72Error(
                    f"{split}/{label} lacks left/right/bilateral cue support"
                )
            if max(placement_counts.values()) - min(placement_counts.values()) > int(
                config.diversity_gates["maximum_cue_placement_count_gap_per_label"]
            ):
                raise DomainRandomizedV72Error(f"{split}/{label} cue placement is imbalanced")
            right_ratio = context["right_visible"] / context["records"]
            if right_ratio < float(config.diversity_gates["minimum_right_visible_cue_ratio"]):
                raise DomainRandomizedV72Error(
                    f"{split}/{label} lacks right-visible semantic context"
                )
            if (
                context["aliases"] != set(range(3))
                or context["surface_variants"] != set(range(4))
                or len(context["semantic_skeletons"]) < minimum_skeletons
                or not isinstance(context["minimum_right_context_chars"], int)
                or context["minimum_right_context_chars"] < 4
            ):
                raise DomainRandomizedV72Error(
                    f"{split}/{label} natural template diversity is insufficient"
                )
            feature = feature_coverage[split][label]
            if (
                len(feature["domains"]) < 6
                or feature["positions"] != set(FIELD_POSITIONS)
                or feature["lengths"] != set(LENGTH_BANDS)
                or feature["noise"] != set(NOISE_STYLES)
            ):
                raise DomainRandomizedV72Error(f"{split}/{label} feature coverage is insufficient")

    if sum(len(split_groups[split]["value_groups"]) for split in SPLITS) != all_value_occurrences:
        raise DomainRandomizedV72Error("a synthetic value was repeated")

    pairwise_collisions: dict[str, dict[str, int]] = {}
    for left, right in combinations(SPLITS, 2):
        pair = f"{left}__{right}"
        pairwise_collisions[pair] = {
            key: len(split_groups[left][key] & split_groups[right][key]) for key in GROUP_KEYS
        }
    pairwise_collision_total = sum(
        count for collision in pairwise_collisions.values() for count in collision.values()
    )
    if pairwise_collision_total:
        raise DomainRandomizedV72Error(f"cross-split group collision: {pairwise_collisions}")

    group_summary = {
        split: {
            key: {
                "unique": len(split_groups[split][key]),
                "set_sha256": _canonical_json_hash(sorted(split_groups[split][key])),
            }
            for key in GROUP_KEYS
        }
        for split in SPLITS
    }
    natural_summary = {
        "required_document_ratio": config.natural_positive_ratio,
        "by_split": {
            split: {
                "documents": natural_positive_counts[split],
                "document_ratio": natural_positive_counts[split] / positive_counts[split],
                "by_label": {
                    label: {
                        "entity_contexts": natural_context[split][label]["records"],
                        "cue_placements": dict(
                            sorted(natural_context[split][label]["cue_placements"].items())
                        ),
                        "right_visible_ratio": (
                            natural_context[split][label]["right_visible"]
                            / natural_context[split][label]["records"]
                        ),
                        "alias_count": len(natural_context[split][label]["aliases"]),
                        "surface_variant_count": len(
                            natural_context[split][label]["surface_variants"]
                        ),
                        "semantic_family_count": len(
                            natural_context[split][label]["semantic_families"]
                        ),
                        "semantic_skeleton_count": len(
                            natural_context[split][label]["semantic_skeletons"]
                        ),
                        "minimum_right_context_chars": natural_context[split][label][
                            "minimum_right_context_chars"
                        ],
                    }
                    for label in CORE_LABELS
                },
            }
            for split in SPLITS
        },
    }
    return {
        "counts": {
            "total": len(records),
            "by_split": {split: counts[split] for split in SPLITS},
            "positive_by_split": {split: positive_counts[split] for split in SPLITS},
            "hard_negative_by_split": {split: hard_negative_counts[split] for split in SPLITS},
            "counterfactual_by_split": {split: counterfactual_counts[split] for split in SPLITS},
        },
        "composition": {
            "by_split": expected_composition,
            "domain_counts_by_split": {
                split: dict(sorted(domains[split].items())) for split in SPLITS
            },
            "label_counts_by_split": {
                split: dict(sorted(labels[split].items())) for split in SPLITS
            },
            "primary_label_counts_by_split_and_composition": {
                split: {
                    composition: dict(sorted(primary_labels[split][composition].items()))
                    for composition in POSITIVE_COMPOSITIONS
                }
                for split in SPLITS
            },
        },
        "coverage": {
            "natural_context": natural_summary,
            "template_family_counts_by_split": {
                split: {
                    role: len(template_families[split][role])
                    for role in ("positive", "hard_negative")
                }
                for split in SPLITS
            },
            "semantic_skeleton_counts_by_split": {
                split: {
                    role: len(semantic_skeletons[split][role])
                    for role in ("positive", "hard_negative")
                }
                for split in SPLITS
            },
        },
        "hard_negative_audit": {
            "required_ratio": config.pii_free_hard_negative_ratio,
            "by_split": {
                split: {
                    "records": hard_negative_counts[split],
                    "counterfactual_records": counterfactual_counts[split],
                    "shortcut_cue_records": shortcut_counts[split],
                    "shortcut_cue_ratio": shortcut_counts[split] / hard_negative_counts[split],
                    "natural_context_ratio": 1.0
                    - shortcut_counts[split] / hard_negative_counts[split],
                }
                for split in SPLITS
            },
        },
        "isolation": {
            "group_keys": list(GROUP_KEYS),
            "by_split": group_summary,
            "pairwise_collisions_by_group": pairwise_collisions,
            "pairwise_collision_total": pairwise_collision_total,
            "all_required_collision_counts_zero": True,
            "pairwise_all_splits": True,
            "global_nonrepeating_value_stream": True,
            "semantic_skeleton_normalization": (
                "NFKC+casefold+remove-placeholders+retain-alphanumeric"
            ),
        },
    }


def build_domain_randomized_v7_2_records(
    config: DomainRandomizedV72Config,
) -> tuple[list[DocumentRecord], dict[str, Any]]:
    """Build all three partitions from the closed recipe, test first."""

    _validate_runtime_config(config)
    factory = SyntheticValueFactory(random.Random(config.seed))
    seen_value_groups: set[str] = set()
    records: list[DocumentRecord] = []
    for split in GENERATION_ORDER:
        records.extend(
            _split_records(
                config,
                split=split,
                factory=factory,
                seen_value_groups=seen_value_groups,
            )
        )
    audit = audit_domain_randomized_v7_2_records(records, config)
    return records, audit


def _write_split(
    records: Sequence[DocumentRecord],
    path: Path,
    *,
    split: str,
    config: DomainRandomizedV72Config,
) -> dict[str, Any]:
    selected = [record for record in records if record.split == split]
    count = write_jsonl(selected, path)
    path.chmod(0o444)
    expected_policy = config.usage_policy[split]
    expected_pool = (
        DataPool.EVALUATION_ONLY if expected_policy["evaluation_only"] else DataPool.PUBLIC_RELEASE
    )
    verified = 0
    for record in iter_jsonl(path):
        if (
            record.split != split
            or record.data_pool is not expected_pool
            or record.public_weight_training_allowed
            is not expected_policy["public_weight_training_allowed"]
            or record.provenance.evaluation_only is not expected_policy["evaluation_only"]
        ):
            raise DomainRandomizedV72Error(f"{split} readback policy failed")
        verified += 1
    if count != len(selected) or verified != count:
        raise DomainRandomizedV72Error(f"{split} readback count failed")
    return {
        "name": path.name,
        "split": split,
        "records": count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "mode": "0444",
        "record_data_pool": expected_pool.value,
        "public_weight_training_allowed": expected_policy["public_weight_training_allowed"],
    }


def _write_read_only_json(path: Path, document: Mapping[str, Any]) -> str:
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return _sha256_file(path)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return f"repository-file:{path.name}"


def materialize_domain_randomized_v7_2(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically publish v7.2 and refuse every overwrite or staging reuse."""

    config = load_domain_randomized_v7_2_config(
        config_path,
        repository_root=repository_root,
    )
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite an existing v7.2 dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("refusing to reuse an existing staging directory")
    staging.mkdir(mode=0o700)
    try:
        records, audit = build_domain_randomized_v7_2_records(config)
        source_path = Path(__file__).resolve(strict=True)
        generator_source_sha256 = _sha256_file(source_path)
        taxonomy_sha256 = _sha256_file(config.taxonomy_path)
        files: dict[str, dict[str, Any]] = {}

        # The independent test bytes and seal are finalized before train/dev are written.
        test_path = staging / "independent_test.jsonl"
        files["independent_test"] = _write_split(
            records,
            test_path,
            split="independent_test",
            config=config,
        )
        test_policy = dict(config.usage_policy["independent_test"])
        seal: dict[str, Any] = {
            "schema_version": "pii-zh.domain-randomized-v7.2-independent-test-seal.v1",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "split": "independent_test",
            "seal_created_before_train_and_dev_files": True,
            "generation_order": list(GENERATION_ORDER),
            "usage_policy": test_policy,
            "bindings": {
                "config": {
                    "path": _display_path(config.config_path, config.repository_root),
                    "sha256": config.config_sha256,
                },
                "generator_source": {
                    "path": _display_path(source_path, config.repository_root),
                    "sha256": generator_source_sha256,
                },
                "taxonomy": {
                    "path": _display_path(config.taxonomy_path, config.repository_root),
                    "sha256": taxonomy_sha256,
                    "ordered_core24_sha256": _canonical_json_hash(list(CORE_LABELS)),
                },
                "independent_test_file": dict(files["independent_test"]),
            },
            "clean_room": dict(config.clean_room),
            "content_access_boundary": {
                "candidate_must_be_frozen": True,
                "training_forbidden": True,
                "checkpoint_model_threshold_and_fusion_selection_forbidden": True,
                "one_shot_final_evaluation_only": True,
            },
        }
        seal["seal_sha256"] = _canonical_json_hash(seal)
        seal_path = staging / "independent_test.seal.json"
        seal_file_sha256 = _write_read_only_json(seal_path, seal)

        for split in ("train", "dev"):
            files[split] = _write_split(
                records,
                staging / f"{split}.jsonl",
                split=split,
                config=config,
            )

        manifest: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "license": "Apache-2.0",
            "language": "zh-Hans-CN",
            "publication_pool": DataPool.PUBLIC_RELEASE.value,
            "record_data_pool_by_split": {
                split: files[split]["record_data_pool"] for split in SPLITS
            },
            "config": {
                "path": _display_path(config.config_path, config.repository_root),
                "sha256": config.config_sha256,
            },
            "source_bindings": {
                "generator_source": {
                    "path": _display_path(source_path, config.repository_root),
                    "sha256": generator_source_sha256,
                },
                "taxonomy": {
                    "path": _display_path(config.taxonomy_path, config.repository_root),
                    "sha256": taxonomy_sha256,
                    "ordered_core24_sha256": _canonical_json_hash(list(CORE_LABELS)),
                },
            },
            "generation": {
                "name": GENERATOR_NAME,
                "version": GENERATOR_VERSION,
                "seed": config.seed,
                "generation_order": list(GENERATION_ORDER),
                "test_was_written_and_sealed_before_train_dev": True,
                "pii_free_hard_negative_ratio": config.pii_free_hard_negative_ratio,
                "natural_positive_ratio": config.natural_positive_ratio,
                "counterfactual_share_of_hard_negative": (
                    config.counterfactual_share_of_hard_negative
                ),
                "positive_composition": dict(config.positive_composition),
                "external_dataset_inputs": [],
            },
            "clean_room": dict(config.clean_room),
            "usage_policy": {split: dict(config.usage_policy[split]) for split in SPLITS},
            "labels": {
                "count": len(CORE_LABELS),
                "names": list(CORE_LABELS),
                "exactly_balanced_by_split": True,
            },
            "counts": audit["counts"],
            "composition": audit["composition"],
            "audits": {
                "coverage": audit["coverage"],
                "hard_negative": audit["hard_negative_audit"],
                "isolation": audit["isolation"],
            },
            "files": {split: files[split] for split in SPLITS},
            "write_order": [
                "independent_test.jsonl",
                "independent_test.seal.json",
                "train.jsonl",
                "dev.jsonl",
                "dataset_manifest.json",
            ],
            "independent_test_seal": {
                "name": seal_path.name,
                "mode": "0444",
                "file_sha256": seal_file_sha256,
                "seal_sha256": seal["seal_sha256"],
                "generator_source_sha256": generator_source_sha256,
                "taxonomy_sha256": taxonomy_sha256,
            },
        }
        manifest["manifest_sha256"] = _canonical_json_hash(manifest)
        manifest_path = staging / "dataset_manifest.json"
        _write_read_only_json(manifest_path, manifest)
        staging.chmod(0o555)
        os.replace(staging, output)
        return manifest
    except Exception:
        if staging.exists():
            staging.chmod(0o700)
            for child in staging.iterdir():
                if child.is_file():
                    child.chmod(0o600)
            shutil.rmtree(staging)
        raise


__all__ = [
    "COMPOSITIONS",
    "CUE_PLACEMENTS",
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "GENERATION_ORDER",
    "GENERATOR_VERSION",
    "GROUP_KEYS",
    "NATURAL_POSITIVE_COMPOSITIONS",
    "POSITIVE_COMPOSITIONS",
    "SPLITS",
    "DomainRandomizedV72Config",
    "DomainRandomizedV72Error",
    "audit_domain_randomized_v7_2_records",
    "build_domain_randomized_v7_2_records",
    "load_domain_randomized_v7_2_config",
    "materialize_domain_randomized_v7_2",
]
