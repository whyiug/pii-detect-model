"""Independent deterministic v4 repair data with one structured surface per negative.

V4 is a clean restart.  It imports only the stable canonical schema, validators,
and generic synthetic value factory.  No v2/v3 row, template, value, recipe, or
model artifact is accepted as an input.  The sole v3 input is a content-addressed
aggregate diagnostic, which is recorded as design evidence and is never a parent
artifact.
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
from .values import SyntheticValueFactory, generate_value

DATASET_ID = "pii_zh_synthetic_targeted_repair_v4"
DATASET_VERSION = "0.4.0"
GENERATOR_NAME = "pii_zh_targeted_repair_v4_single_surface"
GENERATOR_VERSION = "0.4.0"
MATERIALIZER_VERSION = "0.4.0"
DEFAULT_LICENSE = "Apache-2.0"
DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[4] / "configs/data/targeted_repair_v4.yaml"
)
DEFAULT_FREEZE_RECEIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "reports/data_recipe_audits/targeted_repair_v4_implementation_freeze_v4.json"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_V3_DENYLIST_PATH = _REPOSITORY_ROOT / "configs/data/v3_diagnostic_only_artifact_denylist.yaml"
_V3_RETENTION_PATH = (
    _REPOSITORY_ROOT / "reports/data_recipe_audits/v3_diagnostic_only_retention_receipt.json"
)
_V3_DIAGNOSTIC_PATH = (
    _REPOSITORY_ROOT / "reports/diagnostics/macbert_24_v3_single_surface_transfer_failure.json"
)
_PRE_FREEZE_AMENDMENT_PATH = (
    _REPOSITORY_ROOT
    / "reports/data_recipe_audits/targeted_repair_v4_pre_freeze_recipe_amendment_receipt.json"
)
_PRE_FREEZE_AMENDMENT_FILE_SHA256 = "eb57c59e84230e3ba82445dd422b56825482017c2f9655203c825e81b4df69f4"
_PRE_FREEZE_AMENDMENT_LOGICAL_SHA256 = "98fabdf6a7c497504e05923e6595e48d32a1462981bdbf69f90d24e3b6cbedc3"
_SUPERSEDED_RECIPE_FILE_SHA256 = "34935246675353ba64c8d267542eaf64386a9992b771b869bad2fdfb88aab7fa"
_V3_DENYLIST_FILE_SHA256 = "58f650a366a431b620ff0e463883ed50d31702884712ffcacb0002538733576a"
_V3_RETENTION_FILE_SHA256 = "fa57b47dc5898c7ad142ae74dac66a076ad7d3c007c7b575f24c23a93bad3a2c"
_V3_RETENTION_LOGICAL_SHA256 = "fba33cfd513f922d82cd04b8cf4d1c673656d64419f2bd60b59c49d49686889f"
_V3_DIAGNOSTIC_FILE_SHA256 = "40b23767e1d1989264d8b9374f1d45f93bb631ac01f993b8593da8f6e52e477d"
_V3_DIAGNOSTIC_LOGICAL_SHA256 = "f2e91ec725bdfa61b2e5190f3d24ed78533d573520f5815763e9175c80fa207a"
_SPLITS = ("train", "validation")
_BUCKETS = ("short", "medium", "long")
_BUCKET_RANGES = {"short": (18, 30), "medium": (31, 45), "long": (46, 60)}
_NGRAM_SIZE = 5
_NEAR_THRESHOLD = 0.92
_ASCII_STRUCTURED_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9@._:/+*\-]{5,}[A-Za-z0-9])(?![A-Za-z0-9])"
)
_CHINESE_DATE_RE = re.compile(r"(?<!\d)((?:19|20|21)\d{2}年\d{1,2}月\d{1,2}日)(?!\d)")
_PLACEHOLDER = "<<CANDIDATE>>"
_EXPLICIT_CUES = ("失败", "拒绝", "未通过", "错误")
_BANNED_SHORTCUTS = ("测试", "示例", "合成", "虚构", "演练")
_SYNTHETIC_CARD_IIN = "999999"
_SYNTHETIC_ID_REGION_PREFIX = "990000"
_REFERENCE_DATE = date(2026, 7, 12)

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

_LABEL_CONTEXT = MappingProxyType(
    {
        "PERSON_NAME": ("人员名册", "姓名", "身份核对", "human_resources"),
        "PHONE_NUMBER": ("联络清单", "手机号", "回访联络", "customer_service"),
        "EMAIL_ADDRESS": ("通知渠道", "电子邮箱", "消息投递", "communications"),
        "ADDRESS": ("配送资料", "收件地址", "路线安排", "logistics"),
        "DATE_OF_BIRTH": ("年龄资料", "出生日期", "资格核验", "identity"),
        "CN_RESIDENT_ID": ("实名资料", "居民证件号", "身份核验", "identity"),
        "PASSPORT_NUMBER": ("出行资料", "护照号码", "行程办理", "travel"),
        "DRIVER_LICENSE_NUMBER": ("驾驶档案", "驾驶证号", "准驾核验", "mobility"),
        "SOCIAL_SECURITY_NUMBER": ("社保档案", "社会保障号", "待遇办理", "benefits"),
        "BANK_CARD_NUMBER": ("结算资料", "银行卡号", "付款核对", "finance"),
        "BANK_ACCOUNT_NUMBER": ("开户资料", "银行账户", "资金结算", "finance"),
        "VEHICLE_LICENSE_PLATE": ("车辆台账", "车牌号码", "通行登记", "mobility"),
        "EMPLOYEE_ID": ("人事花名册", "员工工号", "入职管理", "human_resources"),
        "STUDENT_ID": ("学籍名册", "学生学号", "教务办理", "education"),
        "MEDICAL_RECORD_NUMBER": ("诊疗档案", "病历号码", "病案调阅", "healthcare"),
        "WECHAT_ID": ("服务渠道", "微信账号", "在线联络", "customer_service"),
        "QQ_NUMBER": ("即时通讯录", "QQ号码", "在线联络", "customer_service"),
        "ALIPAY_ACCOUNT": ("支付宝资料", "支付宝账户", "支付宝账务核对", "finance"),
        "USERNAME": ("访问目录", "登录用户名", "权限配置", "information_technology"),
        "IP_ADDRESS": ("网络日志", "终端地址", "连接排查", "information_technology"),
        "MAC_ADDRESS": ("终端台账", "网卡地址", "设备接入", "information_technology"),
        "DEVICE_ID": ("设备清单", "设备标识", "资产管理", "information_technology"),
        "GEO_COORDINATE": ("定位记录", "地理坐标", "路线研判", "geospatial"),
        "SECRET": ("密钥清单", "访问密钥", "凭据轮换", "information_security"),
    }
)

_NEGATIVE_COUNTS = MappingProxyType(
    {
        "train": MappingProxyType(
            {
                "invalid_card_single_surface": 576,
                "invalid_resident_id_single_surface": 384,
                "ordinary_date_single_surface": 160,
                "valid_card_non_pii_counterfactual": 96,
                "valid_resident_id_non_pii_counterfactual": 64,
                "organization_decoy": 32,
                "order_reference_decoy": 32,
            }
        ),
        "validation": MappingProxyType(
            {
                "invalid_card_single_surface": 144,
                "invalid_resident_id_single_surface": 96,
                "ordinary_date_single_surface": 40,
                "valid_card_non_pii_counterfactual": 24,
                "valid_resident_id_non_pii_counterfactual": 16,
                "organization_decoy": 8,
                "order_reference_decoy": 8,
            }
        ),
    }
)

# Exact category/bucket/cue allocation.  This closes both the split totals and
# the global length/cue marginals rather than checking only percentages.
_NEGATIVE_MATRIX: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]] = MappingProxyType(
    {
        "train": {
            "invalid_card_single_surface": {
                "short": {"explicit": 88, "natural": 264},
                "medium": {"explicit": 36, "natural": 108},
                "long": {"explicit": 20, "natural": 60},
            },
            "invalid_resident_id_single_surface": {
                "short": {"explicit": 56, "natural": 168},
                "medium": {"explicit": 24, "natural": 72},
                "long": {"explicit": 16, "natural": 48},
            },
            "ordinary_date_single_surface": {
                "short": {"natural": 96}, "medium": {"natural": 40}, "long": {"natural": 24}
            },
            "valid_card_non_pii_counterfactual": {
                "short": {"natural": 56}, "medium": {"natural": 24}, "long": {"natural": 16}
            },
            "valid_resident_id_non_pii_counterfactual": {
                "short": {"natural": 40}, "medium": {"natural": 16}, "long": {"natural": 8}
            },
            "organization_decoy": {
                "short": {"natural": 20}, "medium": {"natural": 8}, "long": {"natural": 4}
            },
            "order_reference_decoy": {
                "short": {"natural": 20}, "medium": {"natural": 8}, "long": {"natural": 4}
            },
        },
        "validation": {
            "invalid_card_single_surface": {
                "short": {"explicit": 22, "natural": 66},
                "medium": {"explicit": 9, "natural": 27},
                "long": {"explicit": 5, "natural": 15},
            },
            "invalid_resident_id_single_surface": {
                "short": {"explicit": 14, "natural": 42},
                "medium": {"explicit": 6, "natural": 18},
                "long": {"explicit": 4, "natural": 12},
            },
            "ordinary_date_single_surface": {
                "short": {"natural": 24}, "medium": {"natural": 10}, "long": {"natural": 6}
            },
            "valid_card_non_pii_counterfactual": {
                "short": {"natural": 14}, "medium": {"natural": 6}, "long": {"natural": 4}
            },
            "valid_resident_id_non_pii_counterfactual": {
                "short": {"natural": 10}, "medium": {"natural": 4}, "long": {"natural": 2}
            },
            "organization_decoy": {
                "short": {"natural": 5}, "medium": {"natural": 2}, "long": {"natural": 1}
            },
            "order_reference_decoy": {
                "short": {"natural": 5}, "medium": {"natural": 2}, "long": {"natural": 1}
            },
        },
    }
)

_EXPECTED_LENGTH_COUNTS = {
    "train": {"short": 808, "medium": 336, "long": 200},
    "validation": {"short": 202, "medium": 84, "long": 50},
}
_EXPECTED_PAIRS = {
    "train": {"BANK_CARD_NUMBER": 96, "CN_RESIDENT_ID": 64, "DATE_OF_BIRTH": 104},
    "validation": {"BANK_CARD_NUMBER": 24, "CN_RESIDENT_ID": 16, "DATE_OF_BIRTH": 26},
}
_PAIR_BUCKET_COUNTS = {
    "train": {
        "BANK_CARD_NUMBER": {"short": 56, "medium": 24, "long": 16},
        "CN_RESIDENT_ID": {"short": 40, "medium": 16, "long": 8},
        "DATE_OF_BIRTH": {"short": 64, "medium": 24, "long": 16},
    },
    "validation": {
        "BANK_CARD_NUMBER": {"short": 14, "medium": 6, "long": 4},
        "CN_RESIDENT_ID": {"short": 10, "medium": 4, "long": 2},
        "DATE_OF_BIRTH": {"short": 16, "medium": 6, "long": 4},
    },
}
_PAIR_POSITIVE_STARTS = {
    "train": {
        "BANK_CARD_NUMBER": {"short": 0, "medium": 60, "long": 86},
        "CN_RESIDENT_ID": {"short": 0, "medium": 64, "long": 90},
        "DATE_OF_BIRTH": {"short": 0, "medium": 64, "long": 88},
    },
    "validation": {
        label: {"short": 0, "medium": 16, "long": 22}
        for label in ("BANK_CARD_NUMBER", "CN_RESIDENT_ID", "DATE_OF_BIRTH")
    },
}
_ROLE_FAMILIES = (
    "laboratory_sample",
    "physical_asset",
    "shipment_or_seal",
    "archive_index",
    "software_build",
    "business_order",
)
_ROLE_PREFIXES = {
    "train": {
        "laboratory_sample": "实验室样本条码",
        "physical_asset": "固定资产标签",
        "shipment_or_seal": "物流封签编号",
        "archive_index": "档案索引代码",
        "software_build": "软件构建标记",
        "business_order": "业务订单引用",
    },
    "validation": {
        "laboratory_sample": "检验样本追踪码",
        "physical_asset": "设备资产铭牌",
        "shipment_or_seal": "运输封条标识",
        "archive_index": "卷宗检索代号",
        "software_build": "程序发布标记",
        "business_order": "采购单据引用",
    },
}

_SPECIAL_POSITIVE_BODIES = MappingProxyType(
    {
        "BANK_CARD_NUMBER": {
            "train": (
                "本人银行卡号为<<ENTITY>>。",
                "持卡人的卡片号码是<<ENTITY>>。",
                "该账户持有人用于工资入账的银行卡号是<<ENTITY>>。",
                "银行客服复核个人借记卡资料时，确认这张卡由本人持有且完整卡号为<<ENTITY>>。",
            ),
            "validation": (
                "个人借记卡号登记为<<ENTITY>>。",
                "办理退款时，收款人确认本人使用的银行卡完整号码是<<ENTITY>>。",
                "银行在核对持卡人实名资料后，将这张个人银行卡的完整卡片号码记录为<<ENTITY>>。",
            ),
        },
        "CN_RESIDENT_ID": {
            "train": (
                "居民身份证号为<<ENTITY>>。",
                "本人身份证号码是<<ENTITY>>。",
                "实名认证材料登记的居民身份证证件号码为<<ENTITY>>。",
                "业务人员复核自然人户籍身份资料后，确认当事人所持居民身份证的完整号码为<<ENTITY>>。",
            ),
            "validation": (
                "申请人的身份证号码为<<ENTITY>>。",
                "核对证件后，居民身份证完整号码为<<ENTITY>>。",
                "完成实名核验后，工作人员在身份档案中记录该自然人居民身份证正面的完整号码为<<ENTITY>>。",
            ),
        },
        "DATE_OF_BIRTH": {
            "train": (
                "个人档案登记的出生日期是<<ENTITY>>。",
                "该当事人申报的生日日期为<<ENTITY>>。",
                "办理个人身份资料时，申请人确认自己的出生年月日为<<ENTITY>>。",
                "客服复核自然人身份档案中的相关基本资料后，确认当事人的出生日期已经完整登记为<<ENTITY>>。",
            ),
            "validation": (
                "申请表中的出生年月日为<<ENTITY>>。",
                "工作人员查阅个人基本信息，并确认当事人的生日日期记录为<<ENTITY>>。",
                "核对自然人身份资料及相关申请记录后，受理人员将该申请人完整的出生年月日正式登记为<<ENTITY>>。",
            ),
        },
    }
)

_SPECIAL_POSITIVE_VARIANT_COUNTS = MappingProxyType(
    {
        "BANK_CARD_NUMBER": {"train": (30, 30, 26, 18), "validation": (16, 6, 4)},
        "CN_RESIDENT_ID": {"train": (32, 32, 26, 14), "validation": (16, 6, 4)},
        "DATE_OF_BIRTH": {"train": (32, 32, 24, 16), "validation": (16, 6, 4)},
    }
)
_SPECIAL_POSITIVE_VARIANT_BUCKETS = {
    "train": ("short", "short", "medium", "long"),
    "validation": ("short", "medium", "long"),
}
_PRIORITY_SHAPE_LABELS = frozenset(
    {
        "EMPLOYEE_ID",
        "QQ_NUMBER",
        "MEDICAL_RECORD_NUMBER",
        "BANK_ACCOUNT_NUMBER",
        "SOCIAL_SECURITY_NUMBER",
    }
)


class TargetedRepairV4Error(ValueError):
    """Fail-closed error that never reflects generated text or local paths."""


@dataclass(frozen=True, slots=True)
class RepairRecipe:
    seed: int
    counts: Mapping[str, int]
    positive_per_label: Mapping[str, int]
    sha256: str
    v3_boundary: Mapping[str, Any]
    raw: Mapping[str, Any]


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


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_v3_diagnostic_boundary() -> Mapping[str, Any]:
    bindings = (
        (_V3_DENYLIST_PATH, _V3_DENYLIST_FILE_SHA256),
        (_V3_RETENTION_PATH, _V3_RETENTION_FILE_SHA256),
        (_V3_DIAGNOSTIC_PATH, _V3_DIAGNOSTIC_FILE_SHA256),
        (_PRE_FREEZE_AMENDMENT_PATH, _PRE_FREEZE_AMENDMENT_FILE_SHA256),
    )
    if any(path.is_symlink() or not path.is_file() or sha256_file(path) != expected for path, expected in bindings):
        raise TargetedRepairV4Error("v3 diagnostic boundary file identity changed")
    try:
        denylist = _mapping(yaml.safe_load(_V3_DENYLIST_PATH.read_text(encoding="utf-8")), "v3 denylist")
        retention = _mapping(json.loads(_V3_RETENTION_PATH.read_text(encoding="utf-8")), "v3 retention")
        diagnostic = _mapping(json.loads(_V3_DIAGNOSTIC_PATH.read_text(encoding="utf-8")), "v3 diagnostic")
        amendment = _mapping(
            json.loads(_PRE_FREEZE_AMENDMENT_PATH.read_text(encoding="utf-8")),
            "v4 pre-freeze amendment",
        )
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise TargetedRepairV4Error("v3 diagnostic boundary is unreadable") from exc
    retention_unsigned = dict(retention)
    retention_claimed = retention_unsigned.pop("receipt_sha256", None)
    diagnostic_unsigned = dict(diagnostic)
    diagnostic_claimed = diagnostic_unsigned.pop("report_sha256", None)
    amendment_unsigned = dict(amendment)
    amendment_claimed = amendment_unsigned.pop("receipt_sha256", None)
    denied_ids = _mapping(denylist.get("denied_ids"), "v3 denied ids")
    effective = _mapping(denylist.get("effective_policy"), "v3 effective policy")
    if (
        denylist.get("denylist_id") != "pii_zh_v3_diagnostic_only_artifacts_v1"
        or denylist.get("status") != "active"
        or retention.get("receipt_id") != "pii_zh_v3_diagnostic_only_retention_v1"
        or retention_claimed != _V3_RETENTION_LOGICAL_SHA256
        or canonical_hash(retention_unsigned) != retention_claimed
        or diagnostic_claimed != _V3_DIAGNOSTIC_LOGICAL_SHA256
        or canonical_hash(diagnostic_unsigned) != diagnostic_claimed
        or amendment.get("amendment_id")
        != "targeted_repair_v4_pre_freeze_recipe_amendment_v1"
        or amendment_claimed != _PRE_FREEZE_AMENDMENT_LOGICAL_SHA256
        or canonical_hash(amendment_unsigned) != amendment_claimed
        or amendment.get("superseded_recipe_file_sha256") != _SUPERSEDED_RECIPE_FILE_SHA256
        or any(
            _mapping(amendment.get("execution_boundary"), "amendment execution boundary").get(key)
            is not False
            for key in (
                "dataset_materialized_from_withdrawn_recipe",
                "GPU_training_started",
                "implementation_freeze_receipt_existed",
                "model_checkpoint_created",
            )
        )
        or denied_ids.get("dataset_ids") != ["pii_zh_synthetic_targeted_repair_v3"]
        or denied_ids.get("experiment_ids") != ["macbert_24_targeted_repair_v3_d0_only_seed42"]
        or any(
            effective.get(key) is not False
            for key in (
                "v4_training_eligible",
                "release_eligible",
                "confirmatory_evaluation_eligible",
                "v4_ancestry_eligible",
                "v4_parent_eligible",
            )
        )
        or effective.get("bytes_preserved") is not True
        or effective.get("diagnostic_inspection_allowed") is not True
    ):
        raise TargetedRepairV4Error("v3 diagnostic-only policy failed closed")
    diagnostic_privacy = _mapping(diagnostic.get("privacy"), "v3 diagnostic privacy")
    if (
        diagnostic_privacy.get("contains_raw_text") is not False
        or diagnostic_privacy.get("contains_entity_values") is not False
        or diagnostic_privacy.get("contains_absolute_paths") is not False
    ):
        raise TargetedRepairV4Error("v3 diagnostic is not aggregate-only")
    return MappingProxyType(
        {
            "denylist_id": denylist["denylist_id"],
            "denylist_file_sha256": _V3_DENYLIST_FILE_SHA256,
            "retention_receipt_id": retention["receipt_id"],
            "retention_receipt_file_sha256": _V3_RETENTION_FILE_SHA256,
            "retention_receipt_sha256": retention_claimed,
            "diagnostic_file_sha256": _V3_DIAGNOSTIC_FILE_SHA256,
            "diagnostic_report_sha256": diagnostic_claimed,
            "diagnostic_inspection_allowed": True,
            "v3_training_release_confirmatory_ancestry_parent_eligible": False,
            "bytes_preserved": True,
            "pre_freeze_amendment_file_sha256": _PRE_FREEZE_AMENDMENT_FILE_SHA256,
            "pre_freeze_amendment_receipt_sha256": amendment_claimed,
            "superseded_recipe_file_sha256": _SUPERSEDED_RECIPE_FILE_SHA256,
        }
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TargetedRepairV4Error(f"{field} must be a string-keyed object")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TargetedRepairV4Error(f"{field} must be an integer >= {minimum}")
    return value


def load_recipe(path: Path = DEFAULT_RECIPE_PATH) -> RepairRecipe:
    if path.is_symlink() or not path.is_file():
        raise TargetedRepairV4Error("recipe must be a regular non-symlink file")
    raw_bytes = path.read_bytes()
    try:
        root = _mapping(yaml.safe_load(raw_bytes.decode()), "recipe")
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TargetedRepairV4Error("recipe is invalid YAML") from exc
    if (
        root.get("schema_version") != 1
        or root.get("recipe_id") != "pii_zh_synthetic_targeted_repair_v4_single_surface_transfer"
        or root.get("recipe_version") != DATASET_VERSION
        or root.get("status") != "preregistered_frozen"
        or root.get("seed") != 20260715
    ):
        raise TargetedRepairV4Error("recipe identity is not frozen v4")
    freeze_history = _mapping(root.get("freeze_history"), "freeze history")
    withdrawn_v1 = _mapping(freeze_history.get("withdrawn_v1"), "withdrawn v1 freeze")
    withdrawn_v2 = _mapping(freeze_history.get("withdrawn_v2"), "withdrawn v2 freeze")
    withdrawn_v3 = _mapping(freeze_history.get("withdrawn_v3"), "withdrawn v3 freeze")
    if (
        freeze_history.get("active_receipt_id")
        != "targeted_repair_v4_implementation_freeze_v4"
        or withdrawn_v1.get("receipt_id") != "targeted_repair_v4_implementation_freeze_v1"
        or withdrawn_v1.get("withdrawal_file_sha256")
        != "70a412c4368f8f74ad0a0a95043d577ecc7eaf44e40dce33451ea4b1a2b7992c"
        or withdrawn_v1.get("withdrawal_receipt_sha256")
        != "740fb7e47e9167bfba4802412bc97031dc43e58a74e078672508cd1b19b7eee4"
        or withdrawn_v1.get("GPU_training_started") is not False
        or withdrawn_v2.get("receipt_id") != "targeted_repair_v4_implementation_freeze_v2"
        or withdrawn_v2.get("withdrawal_file_sha256")
        != "be9ac4180a08676259b032c637397cb7f9c251e66425fbce7fedcd863fb800a7"
        or withdrawn_v2.get("withdrawal_receipt_sha256")
        != "6015e50b0a96e3f2c79c5f3434a73d7d4dbae7d6230f768f8ad097e1de3c0121"
        or withdrawn_v2.get("GPU_training_started") is not False
        or withdrawn_v3.get("receipt_id") != "targeted_repair_v4_implementation_freeze_v3"
        or withdrawn_v3.get("withdrawal_file_sha256")
        != "f33ff4f13f50d1834c1059faccd74ff9527b9961022a6903f6851f949b0ed2d0"
        or withdrawn_v3.get("withdrawal_receipt_sha256")
        != "ab9c9de3aa94d9d73d95b31c539fdb520ae2d81f40e7bb441b86a8743e846a8d"
        or withdrawn_v3.get("GPU_training_started") is not False
    ):
        raise TargetedRepairV4Error("v4 freeze history changed")
    counts_raw = _mapping(root.get("counts"), "counts")
    counts = {split: _integer(counts_raw.get(split), f"counts.{split}") for split in _SPLITS}
    if counts != {"train": 3840, "validation": 960}:
        raise TargetedRepairV4Error("split counts changed")
    positive = _mapping(root.get("positive_allocation"), "positive_allocation")
    per_label = {
        "train": _integer(positive.get("train_per_core_label"), "positive.train", 1),
        "validation": _integer(
            positive.get("validation_per_core_label"), "positive.validation", 1
        ),
    }
    if per_label != {"train": 104, "validation": 26} or positive.get(
        "exact_equal_all_24_core_labels"
    ) is not True:
        raise TargetedRepairV4Error("positive allocation changed")
    positive_contract = _mapping(root.get("positive_template_contract"), "positive template contract")
    email_alipay = _mapping(
        positive_contract.get("email_alipay_same_shape_semantic_pairs"),
        "email/Alipay semantic pair contract",
    )
    birth_contract = _mapping(
        positive_contract.get("date_of_birth_semantics"), "birth-date semantic contract"
    )
    if (
        email_alipay.get("exact_surface_with_distinct_explicit_context") is not True
        or email_alipay.get("train_pairs") != 104
        or email_alipay.get("validation_pairs") != 26
        or email_alipay.get("cross_split_pairs") != 0
        or birth_contract.get("frozen_reference_date") != _REFERENCE_DATE.isoformat()
        or birth_contract.get("minimum_age_years") != 18
        or birth_contract.get("maximum_age_years") != 80
        or birth_contract.get("generated_birth_start_year_inclusive") != 1980
        or birth_contract.get("generated_birth_end_year_exclusive") != 2000
        or birth_contract.get("future_birth_dates_allowed") is not False
    ):
        raise TargetedRepairV4Error("positive semantic contract changed")
    negative = _mapping(root.get("negative_allocation"), "negative_allocation")
    for split in _SPLITS:
        if dict(_mapping(negative.get(split), f"negative.{split}")) != dict(
            _NEGATIVE_COUNTS[split]
        ):
            raise TargetedRepairV4Error("negative allocation changed")
        if per_label[split] * len(CORE_LABELS) + sum(_NEGATIVE_COUNTS[split].values()) != counts[
            split
        ]:
            raise TargetedRepairV4Error("allocation does not close")
    surface = _mapping(root.get("surface_contract"), "surface_contract")
    if (
        surface.get("structured_surfaces_per_negative_document") != 1
        or surface.get("candidate_occurrences_per_negative_document") != 1
        or surface.get("CASE_structured_identifier_allowed") is not False
        or surface.get("auxiliary_ascii_or_digit_identifier_allowed") is not False
        or surface.get("cross_split_surface_overlap") != 0
    ):
        raise TargetedRepairV4Error("single-surface contract changed")
    ancestry = _mapping(root.get("ancestry"), "ancestry")
    if (
        ancestry.get("v3_rows_templates_surface_values_model_state_inherited") is not False
        or ancestry.get("v2_rows_templates_surface_values_model_state_inherited") is not False
        or ancestry.get("test_data_read") is not False
        or ancestry.get("hidden_data_read") is not False
        or ancestry.get("external_benchmark_read") is not False
        or ancestry.get("PII_Bench_execution_count") != 0
        or ancestry.get("v3_dataset_id_denied") != "pii_zh_synthetic_targeted_repair_v3"
        or ancestry.get("v3_experiment_id_denied")
        != "macbert_24_targeted_repair_v3_d0_only_seed42"
    ):
        raise TargetedRepairV4Error("v4 ancestry contract changed")
    signals = _mapping(root.get("allowed_development_signals"), "development signals")
    amendment_binding = _mapping(signals.get("pre_freeze_recipe_amendment"), "amendment binding")
    diagnostic_binding = _mapping(signals.get("v3_failure_diagnostic"), "diagnostic binding")
    denylist_binding = _mapping(signals.get("v3_diagnostic_only_denylist"), "denylist binding")
    retention_binding = _mapping(signals.get("v3_retention_receipt"), "retention binding")
    if (
        diagnostic_binding.get("file_sha256") != _V3_DIAGNOSTIC_FILE_SHA256
        or diagnostic_binding.get("report_sha256") != _V3_DIAGNOSTIC_LOGICAL_SHA256
        or diagnostic_binding.get("aggregate_only") is not True
        or denylist_binding.get("file_sha256") != _V3_DENYLIST_FILE_SHA256
        or retention_binding.get("file_sha256") != _V3_RETENTION_FILE_SHA256
        or retention_binding.get("receipt_sha256") != _V3_RETENTION_LOGICAL_SHA256
        or amendment_binding.get("file_sha256") != _PRE_FREEZE_AMENDMENT_FILE_SHA256
        or amendment_binding.get("receipt_sha256") != _PRE_FREEZE_AMENDMENT_LOGICAL_SHA256
        or amendment_binding.get("superseded_recipe_file_sha256")
        != _SUPERSEDED_RECIPE_FILE_SHA256
    ):
        raise TargetedRepairV4Error("v3 diagnostic boundary recipe binding changed")
    gates = _mapping(root.get("development_gates"), "development_gates")
    if (
        gates.get("targeted_each_core_label_f1_minimum") != 0.8
        or gates.get("targeted_each_core_label_recall_minimum") != 0.8
    ):
        raise TargetedRepairV4Error("per-label training gates changed")
    boundary = verify_v3_diagnostic_boundary()
    return RepairRecipe(
        seed=20260715,
        counts=MappingProxyType(counts),
        positive_per_label=MappingProxyType(per_label),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        v3_boundary=boundary,
        raw=MappingProxyType(dict(root)),
    )


class _V4ValueFactory(SyntheticValueFactory):
    def __init__(self, rng: random.Random, split: str) -> None:
        super().__init__(rng)
        self.split = split
        self.split_bit = 0 if split == "train" else 1
        self._offsets_v4 = {
            "card": rng.randrange(100_000_000),
            "id": rng.randrange(30 * 365 * 999),
            "date": rng.randrange(20 * 365),
        }

    def _partitioned(self, key: str, index: int, modulus: int, lane: int) -> int:
        # Eight disjoint lanes: split bit plus positive/invalid/unpaired modes.
        slots = modulus // 8
        return 8 * ((self._offsets_v4[key] + index) % slots) + 2 * lane + self.split_bit

    def card(self, index: int, lane: int, valid: bool) -> str:
        suffix = self._partitioned("card", index, 1_000_000_000, lane)
        first = _SYNTHETIC_CARD_IIN + f"{suffix:09d}"
        check = luhn_check_digit(first)
        if not valid:
            check = "0" if check != "0" else "1"
        return first + check

    def resident_id(self, index: int, lane: int, valid: bool) -> str:
        serial = self._partitioned("id", index, 30 * 365 * 999, lane)
        birth_offset, sequence = divmod(serial, 999)
        birth = date(1980, 1, 1) + timedelta(days=birth_offset)
        first = _SYNTHETIC_ID_REGION_PREFIX + birth.strftime("%Y%m%d") + f"{sequence + 1:03d}"
        check = cn_resident_id_check_code(first)
        if not valid:
            check = "0" if check != "0" else "1"
        return first + check

    @staticmethod
    def _format_date(value: date, style_key: int) -> str:
        styles = (
            value.isoformat(),
            f"{value.year}/{value.month:02d}/{value.day:02d}",
            f"{value.year}年{value.month}月{value.day}日",
        )
        return styles[style_key % len(styles)]

    def birth_date(self, index: int) -> str:
        offset = self._partitioned("date", index, 20 * 365, 0)
        value = date(1980, 1, 1) + timedelta(days=offset)
        return self._format_date(value, offset // 8)

    def ordinary_date(self, index: int) -> str:
        offset = self._partitioned("date", index, 20 * 365, 1)
        value = date(2035, 1, 1) + timedelta(days=offset)
        return self._format_date(value, offset // 8)

    def organization_code(self, index: int) -> str:
        return f"ORG-{self.split_bit}{index:011d}"

    def order_code(self, index: int) -> str:
        return f"REQ-{self.split_bit}{index:011d}"


def _split_seed(seed: int, split: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"repair-v4\0{seed}\0{split}".encode()).digest()[:8], "big"
    )


def _doc_id(split: str, kind: str, index: int) -> str:
    suffix = hashlib.sha256(f"v4\0{split}\0{kind}\0{index}".encode()).hexdigest()[:24]
    return f"v4-{split}-{suffix}"


def _relation_id(split: str, label: str, index: int) -> str:
    return "v4-rel-" + hashlib.sha256(f"{split}\0{label}\0{index}".encode()).hexdigest()[:24]


def _positive_variant(split: str, label: str, index: int) -> int:
    if label not in _SPECIAL_POSITIVE_VARIANT_COUNTS:
        return index % (4 if split == "train" else 2)
    running = 0
    for variant, count in enumerate(_SPECIAL_POSITIVE_VARIANT_COUNTS[label][split]):
        running += count
        if index < running:
            return variant
    raise TargetedRepairV4Error("special positive variant allocation does not close")


def _positive_body(split: str, label: str, variant: int) -> str:
    if label in _SPECIAL_POSITIVE_BODIES:
        return _SPECIAL_POSITIVE_BODIES[label][split][variant]
    container, field, action, _ = _LABEL_CONTEXT[label]
    train_patterns = (
        f"{container}中{field}登记为<<ENTITY>>，用于{action}。",
        f"办理{action}时，{field}按授权记录为<<ENTITY>>。",
        f"{container}的{field}写作<<ENTITY>>，请依流程处理。",
        f"经本人授权，{action}所需{field}为<<ENTITY>>。",
    )
    validation_patterns = (
        f"{action}资料内，{field}留存为<<ENTITY>>。",
        f"受权查阅{container}，其中{field}是<<ENTITY>>。",
    )
    patterns = train_patterns if split == "train" else validation_patterns
    return patterns[variant % len(patterns)]


def _paired_positive_indices(split: str, label: str) -> frozenset[int]:
    if label not in _EXPECTED_PAIRS[split]:
        return frozenset()
    return frozenset(
        _PAIR_POSITIVE_STARTS[split][label][bucket] + offset
        for bucket in _BUCKETS
        for offset in range(_PAIR_BUCKET_COUNTS[split][label][bucket])
    )


def _render_positive(
    *, split: str, label: str, index: int, value: str, recipe: RepairRecipe
) -> DocumentRecord:
    variant = _positive_variant(split, label, index)
    body = _positive_body(split, label, variant)
    if body.count("<<ENTITY>>") != 1:
        raise TargetedRepairV4Error("positive template placeholder contract failed")
    start = body.index("<<ENTITY>>")
    text = body.replace("<<ENTITY>>", value)
    end = start + len(value)
    validation = validate_entity_value(label, value)
    if validation is not None and not validation.valid:
        raise TargetedRepairV4Error("generated positive failed its structured validator")
    relation_kind = {
        "BANK_CARD_NUMBER": "card_valid_exact_surface",
        "CN_RESIDENT_ID": "resident_id_valid_exact_surface",
        "DATE_OF_BIRTH": "date_exact_surface",
    }.get(label)
    if label in {"EMAIL_ADDRESS", "ALIPAY_ACCOUNT"}:
        relation = _relation_id(split, "EMAIL_ALIPAY", index)
        relation_kind = "email_alipay_exact_surface_semantic_pair"
    else:
        relation = (
            _relation_id(split, label, index)
            if index in _paired_positive_indices(split, label)
            else None
        )
    template_group = f"repair-v4-{split}-positive-{label.casefold()}-{variant}"
    skeleton_hash = stable_text_hash(body)
    declared_length_bucket = (
        _SPECIAL_POSITIVE_VARIANT_BUCKETS[split][variant]
        if label in _SPECIAL_POSITIVE_BODIES
        else None
    )
    if declared_length_bucket is not None:
        low, high = _BUCKET_RANGES[declared_length_bucket]
        if not low <= len(text) <= high:
            raise TargetedRepairV4Error("special positive template misses its length bucket")
    metadata = {
        "targeted_repair_v4": True,
        "record_role": "positive",
        "label_specific_context": True,
        "positive_label": label,
        "positive_template_variant": variant,
        "positive_shape_variant": (
            (index // (4 if split == "train" else 2)) % 4
            if label in _PRIORITY_SHAPE_LABELS
            else None
        ),
        "positive_length_bucket": declared_length_bucket,
        "template_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "authored_template_id": template_group,
        "template_semantic_skeleton_sha256": skeleton_hash,
        "relation_kind": relation_kind if relation is not None else None,
        "D0_development_only": True,
        "D0_raw_row_text_accessed_during_design": True,
        "D0_raw_text_copied": False,
        "D0_entity_value_read": False,
        "D0_confirmatory_evaluation_eligible": False,
        "v3_aggregate_diagnostic_only": True,
        "v3_row_template_value_model_ancestry": False,
        "v2_row_template_value_model_ancestry": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "external_benchmark_read": False,
        "PII_Bench_execution_count": 0,
        "non_entity_value_groups": [
            stable_text_hash(f"v4-positive-parameter\0{split}\0{label}\0{index}")
        ],
    }
    provenance = Provenance(
        source_id=DATASET_ID,
        source_kind="deterministic_synthetic_development_repair",
        source_revision=DATASET_VERSION,
        license=DEFAULT_LICENSE,
        synthetic=True,
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=recipe.seed,
        template_id=template_group,
        metadata={"clean_restart": True, "external_evaluation_count": 0},
    )
    quality = QualityMetadata(
        tier=QualityTier.SYNTHETIC_VALIDATED,
        quality_gate=True,
        validators_passed=True,
        review_status="deterministic_contract_validated",
        confidence=1.0,
        metadata={"optimization_only_not_confirmatory_evidence": True},
    )
    _, _, _, domain = _LABEL_CONTEXT[label]
    return DocumentRecord.create(
        doc_id=_doc_id(split, f"positive-{label}", index),
        text=text,
        entities=(
            EntitySpan(
                start=start,
                end=end,
                label=label,
                text=value,
                source=GENERATOR_NAME,
                validator_state=validation.reason if validation is not None else "semantic",
                metadata={"synthetic": True, "label_specific_context": True},
            ),
        ),
        language="zh-CN",
        domain=domain,
        scene=str(_LABEL_CONTEXT[label][2]),
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        template_group=template_group,
        conversation_group=relation or f"v4-single-{split}-{label}-{index}",
        entity_value_groups=(stable_text_hash(value),),
        generator_version=GENERATOR_VERSION,
        split=split,
        metadata=metadata,
    )


def _negative_base(
    split: str, category: str, cue: str, role: str, bucket: str
) -> tuple[str, str]:
    role_index = _ROLE_FAMILIES.index(role)
    if cue == "explicit":
        stem = (
            ("卡号" if "card" in category else "证件号")
            if bucket == "short"
            else ("银行卡号" if "card" in category else "证件号码")
        )
        variants = (
            ("校验失败", "核验错误", "复查失败", "审核错误", "校核失败", "复核错误")
            if split == "train"
            else ("复核拒绝", "复验失败", "核查错误", "复审拒绝", "查验失败", "审阅错误")
        )
        prefix = stem + variants[role_index]
        suffix = (
            "拒收"
            if split == "train" and bucket == "short"
            else "退回"
            if split == "validation" and bucket == "short"
            else "不予采纳"
            if split == "train"
            else "已被退回"
        )
        return prefix, suffix
    role_prefix = _ROLE_PREFIXES[split][role]
    if category in {
        "invalid_card_single_surface",
        "invalid_resident_id_single_surface",
        "valid_card_non_pii_counterfactual",
        "valid_resident_id_non_pii_counterfactual",
    }:
        if bucket == "short":
            short_roles = {
                "train": {
                    "laboratory_sample": "样本码",
                    "physical_asset": "资产码",
                    "shipment_or_seal": "封签码",
                    "archive_index": "索引码",
                    "software_build": "构建码",
                    "business_order": "订单码",
                },
                "validation": {
                    "laboratory_sample": "检样码",
                    "physical_asset": "器材码",
                    "shipment_or_seal": "运封码",
                    "archive_index": "卷宗码",
                    "software_build": "发布码",
                    "business_order": "采购码",
                },
            }
            return short_roles[split][role], "留存" if split == "train" else "归档"
        return role_prefix, "已纳入资料索引" if split == "train" else "已写入查阅目录"
    if category == "ordinary_date_single_surface":
        prefixes = {
            "train": ("样本日期", "资产日期", "封签日期", "归档日期", "发布日期", "订单日期"),
            "validation": ("检样日期", "器材日期", "运封日期", "卷宗日期", "程序日期", "采购日期"),
        }
        suffix = (
            "留存"
            if split == "train" and bucket == "short"
            else "登记"
            if split == "validation" and bucket == "short"
            else "已列入工作日程"
            if split == "train"
            else "已写入进度表"
        )
        return prefixes[split][role_index], suffix
    if category == "organization_decoy":
        prefixes = {
            "train": ("实验伙伴", "资产供应商", "运输承运方", "档案保管方", "软件供应方", "订单相对方"),
            "validation": ("检验协作方", "器材供货方", "物流承接方", "卷宗托管方", "程序提供方", "采购合作方"),
        }
    else:
        prefixes = {
            "train": ("实验采购单", "资产申领单", "运输流转单", "档案调阅单", "构建工单", "业务订单"),
            "validation": ("检验申购单", "器材领用单", "物流交接单", "卷宗借阅单", "发布任务单", "采购单据"),
        }
    suffix = (
        "入册"
        if split == "train" and bucket == "short"
        else "收录"
        if split == "validation" and bucket == "short"
        else "用于内部目录"
        if split == "train"
        else "已进入查阅目录"
    )
    return prefixes[split][role_index], suffix


def _render_negative_text(
    *,
    split: str,
    category: str,
    cue: str,
    bucket: str,
    index: int,
    candidate: str,
    role: str,
) -> tuple[str, str, str, int, int]:
    prefix, suffix = _negative_base(split, category, cue, role, bucket)
    low, high = _BUCKET_RANGES[bucket]
    clauses = {
        "train": {
            "medium": "相关资料按流程留痕",
            "long_natural": "相关资料已完整留痕并仅供授权人员按照权限查阅",
            "long_explicit": "相关资料已完整留痕并仅供授权人员按权限查阅",
        },
        "validation": {
            "medium": "该项记录依权限存档",
            "long_natural": "该项记录已进入业务台账且仅供授权人员按规定调阅",
            "long_explicit": "该项记录已进入业务台账且仅供授权人员调阅",
        },
    }
    clause = (
        clauses[split]["medium"]
        if bucket == "medium"
        else clauses[split][f"long_{cue}"]
        if bucket == "long"
        else None
    )
    body = (
        f"{prefix}：{_PLACEHOLDER}，{suffix}。"
        if bucket == "short"
        else f"{prefix}：{_PLACEHOLDER}，{clause}，{suffix}。"
    )
    text = body.replace(_PLACEHOLDER, candidate)
    start = body.index(_PLACEHOLDER)
    end = start + len(candidate)
    if not low <= len(text) <= high:
        raise TargetedRepairV4Error("negative authored template misses its length bucket")
    authored_skeleton = body.replace(_PLACEHOLDER, "<CANDIDATE>")
    return text, body, authored_skeleton, start, end


def structured_matches(text: str) -> tuple[tuple[int, int, str], ...]:
    matches = {
        (match.start(1), match.end(1), match.group(1))
        for pattern in (_ASCII_STRUCTURED_RE, _CHINESE_DATE_RE)
        for match in pattern.finditer(text)
    }
    return tuple(sorted(matches))


def _parse_frozen_date(value: str) -> date:
    normalized = value.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-")
    try:
        year, month, day = (int(part) for part in normalized.split("-"))
        return date(year, month, day)
    except (TypeError, ValueError) as exc:
        raise TargetedRepairV4Error("generated date syntax is invalid") from exc


def _render_negative(
    *,
    split: str,
    category: str,
    cue: str,
    bucket: str,
    index: int,
    candidate: str,
    role: str,
    relation: tuple[str, str] | None,
    recipe: RepairRecipe,
) -> DocumentRecord:
    text, body, authored_skeleton, start, end = _render_negative_text(
        split=split,
        category=category,
        cue=cue,
        bucket=bucket,
        index=index,
        candidate=candidate,
        role=role,
    )
    matches = structured_matches(text)
    if matches != ((start, end, candidate),):
        raise TargetedRepairV4Error("negative does not contain exactly one structured surface")
    variant_hash = hashlib.sha256(authored_skeleton.encode()).hexdigest()[:20]
    template_group = f"repair-v4-{split}-negative-{category}-{variant_hash}"
    relation_id, relation_kind = relation or (
        f"v4-negative-{split}-{category}-{index}",
        "unpaired_negative",
    )
    metadata = {
        "targeted_repair_v4": True,
        "record_role": "pii_free_hard_negative",
        "hard_negative_category": category,
        "negative_context_style": cue,
        "length_bucket": bucket,
        "structured_surface_count": 1,
        "candidate_span": [start, end],
        "candidate_occurrence_count": text.count(candidate),
        "CASE_structured_identifier_present": False,
        "auxiliary_ascii_or_digit_identifier_present": False,
        "decoy_role_family": role,
        "relation_kind": relation_kind,
        "template_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "authored_template_id": template_group,
        "template_semantic_skeleton_sha256": stable_text_hash(authored_skeleton),
        "D0_development_only": True,
        "D0_raw_row_text_accessed_during_design": True,
        "D0_raw_text_copied": False,
        "D0_entity_value_read": False,
        "D0_confirmatory_evaluation_eligible": False,
        "v3_aggregate_diagnostic_only": True,
        "v3_row_template_value_model_ancestry": False,
        "v2_row_template_value_model_ancestry": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "external_benchmark_read": False,
        "PII_Bench_execution_count": 0,
        "non_entity_value_groups": [stable_text_hash(candidate)],
    }
    provenance = Provenance(
        source_id=DATASET_ID,
        source_kind="deterministic_synthetic_development_repair",
        source_revision=DATASET_VERSION,
        license=DEFAULT_LICENSE,
        synthetic=True,
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=recipe.seed,
        template_id=template_group,
        metadata={"clean_restart": True, "single_surface_contract": True},
    )
    quality = QualityMetadata(
        tier=QualityTier.HARD_NEGATIVE,
        quality_gate=True,
        validators_passed=True,
        review_status="single_surface_contract_validated",
        confidence=1.0,
        metadata={"optimization_only_not_confirmatory_evidence": True},
    )
    return DocumentRecord.create(
        doc_id=_doc_id(split, category, index),
        text=text,
        entities=(),
        language="zh-CN",
        domain="cross_domain_operations",
        scene=role,
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        template_group=template_group,
        conversation_group=relation_id,
        entity_value_groups=(),
        generator_version=GENERATOR_VERSION,
        split=split,
        metadata=metadata,
    )


def _break_unintended_numeric_validators(value: str, label: str) -> str:
    if not value.isdigit():
        return value
    if label in {"BANK_CARD_NUMBER", "CN_RESIDENT_ID"}:
        return value
    prefix = value[:-1]
    original = int(value[-1])
    for offset in range(10):
        candidate = prefix + str((original + offset) % 10)
        if not validate_luhn(candidate).valid and not validate_cn_resident_id(candidate).valid:
            return candidate
    raise TargetedRepairV4Error("could not reject unintended numeric validators")


def _priority_value(factory: _V4ValueFactory, label: str, index: int) -> str | None:
    split_digit = str(factory.split_bit)
    serial = f"{factory.split_bit}{index:011d}"
    # Orthogonal to the template cycle: every label-specific template sees all
    # four shapes with per-cell counts differing by at most one.
    variant = (index // (4 if factory.split == "train" else 2)) % 4
    if label == "EMPLOYEE_ID":
        return (
            f"E{split_digit}{index:09d}",
            f"DPT-ZZ-{factory.split_bit}{index:07d}",
            f"STAFF/ZZ/{factory.split_bit}{index:07d}",
            f"EMP-{factory.split_bit}{index:08d}",
        )[variant]
    if label == "MEDICAL_RECORD_NUMBER":
        return (
            f"M{split_digit}{index:011d}",
            f"MR-{factory.split_bit}{index:09d}",
            f"HIS/{factory.split_bit}{index:09d}",
            f"病历{factory.split_bit}{index:08d}",
        )[variant]
    if label == "QQ_NUMBER":
        lengths = (7, 8, 10, 12)
        length = lengths[variant]
        digits = "9" + split_digit + f"{index:0{length - 3}d}" + "0"
        return _break_unintended_numeric_validators(digits, label)
    if label == "BANK_ACCOUNT_NUMBER":
        lengths = (16, 17, 18, 19)
        length = lengths[variant]
        digits = "88" + split_digit + f"{index:0{length - 4}d}" + "0"
        return _break_unintended_numeric_validators(digits, label)
    if label == "SOCIAL_SECURITY_NUMBER":
        return (
            _break_unintended_numeric_validators("77" + serial + "0000", label),
            f"SB-{factory.split_bit}{index:014d}",
            f"SS/{factory.split_bit}{index:014d}",
            f"社保{factory.split_bit}{index:012d}",
        )[variant]
    return None


def _positive_values(factory: _V4ValueFactory, split: str, per_label: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for label in CORE_LABELS:
        values: list[str] = []
        for index in range(per_label):
            if label == "ALIPAY_ACCOUNT":
                value = result["EMAIL_ADDRESS"][index]
            elif label == "BANK_CARD_NUMBER":
                value = factory.card(index, lane=0, valid=True)
            elif label == "CN_RESIDENT_ID":
                value = factory.resident_id(index, lane=0, valid=True)
            elif label == "DATE_OF_BIRTH":
                value = factory.birth_date(index)
            else:
                priority = _priority_value(factory, label, index)
                value = priority if priority is not None else generate_value(factory, label)
                value = _break_unintended_numeric_validators(value, label)
            values.append(value)
        if len(values) != len(set(values)):
            raise TargetedRepairV4Error("positive values are not unique within a label")
        result[label] = values
    return result


def _negative_candidate(
    *,
    factory: _V4ValueFactory,
    split: str,
    category: str,
    category_index: int,
    bucket: str,
    bucket_index: int,
    positives: Mapping[str, Sequence[str]],
) -> tuple[str, tuple[str, str] | None]:
    if category == "invalid_card_single_surface":
        return factory.card(category_index, lane=1, valid=False), None
    if category == "invalid_resident_id_single_surface":
        return factory.resident_id(category_index, lane=1, valid=False), None
    if category == "valid_card_non_pii_counterfactual":
        label = "BANK_CARD_NUMBER"
        positive_index = _PAIR_POSITIVE_STARTS[split][label][bucket] + bucket_index
        return positives[label][positive_index], (
            _relation_id(split, label, positive_index),
            "card_valid_exact_surface",
        )
    if category == "valid_resident_id_non_pii_counterfactual":
        label = "CN_RESIDENT_ID"
        positive_index = _PAIR_POSITIVE_STARTS[split][label][bucket] + bucket_index
        return positives[label][positive_index], (
            _relation_id(split, label, positive_index),
            "resident_id_valid_exact_surface",
        )
    if category == "ordinary_date_single_surface":
        label = "DATE_OF_BIRTH"
        if bucket_index < _PAIR_BUCKET_COUNTS[split][label][bucket]:
            positive_index = _PAIR_POSITIVE_STARTS[split][label][bucket] + bucket_index
            return positives[label][positive_index], (
                _relation_id(split, label, positive_index),
                "date_exact_surface",
            )
        return factory.ordinary_date(category_index), None
    if category == "organization_decoy":
        return factory.organization_code(category_index), None
    if category == "order_reference_decoy":
        return factory.order_code(category_index), None
    raise TargetedRepairV4Error("unknown negative category")


def generate_records(recipe: RepairRecipe) -> tuple[DocumentRecord, ...]:
    all_records: list[DocumentRecord] = []
    for split in _SPLITS:
        rng = random.Random(_split_seed(recipe.seed, split))
        factory = _V4ValueFactory(rng, split)
        positives = _positive_values(factory, split, recipe.positive_per_label[split])
        selected: list[DocumentRecord] = []
        for label in CORE_LABELS:
            for index, value in enumerate(positives[label]):
                selected.append(
                    _render_positive(
                        split=split, label=label, index=index, value=value, recipe=recipe
                    )
                )
        category_indices = {category: 0 for category in _NEGATIVE_COUNTS[split]}
        category_bucket_indices = {
            category: {bucket: 0 for bucket in _BUCKETS}
            for category in _NEGATIVE_COUNTS[split]
        }
        global_negative_index = 0
        for category, by_bucket in _NEGATIVE_MATRIX[split].items():
            for bucket in _BUCKETS:
                for cue, count in by_bucket[bucket].items():
                    for _ in range(count):
                        category_index = category_indices[category]
                        bucket_index = category_bucket_indices[category][bucket]
                        candidate, relation = _negative_candidate(
                            factory=factory,
                            split=split,
                            category=category,
                            category_index=category_index,
                            bucket=bucket,
                            bucket_index=bucket_index,
                            positives=positives,
                        )
                        role = _ROLE_FAMILIES[category_index % len(_ROLE_FAMILIES)]
                        selected.append(
                            _render_negative(
                                split=split,
                                category=category,
                                cue=cue,
                                bucket=bucket,
                                index=global_negative_index,
                                candidate=candidate,
                                role=role,
                                relation=relation,
                                recipe=recipe,
                            )
                        )
                        category_indices[category] += 1
                        category_bucket_indices[category][bucket] += 1
                        global_negative_index += 1
        if category_indices != dict(_NEGATIVE_COUNTS[split]):
            raise TargetedRepairV4Error("negative generation matrix did not close")
        rng.shuffle(selected)
        all_records.extend(selected)
    return tuple(all_records)


def _ngrams(text: str) -> frozenset[str]:
    normalized = "".join(text.casefold().split())
    if len(normalized) < _NGRAM_SIZE:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + _NGRAM_SIZE]
        for index in range(len(normalized) - _NGRAM_SIZE + 1)
    )


def _cross_split_duplicate_audit(records: Sequence[DocumentRecord]) -> dict[str, Any]:
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    train_hashes = {hashlib.sha256(record.text.encode()).hexdigest() for record in train}
    validation_hashes = {hashlib.sha256(record.text.encode()).hexdigest() for record in validation}
    exact = len(train_hashes & validation_hashes)
    inverted: dict[str, list[frozenset[str]]] = {}
    for record in validation:
        grams = _ngrams(record.text)
        for gram in grams:
            inverted.setdefault(gram, []).append(grams)
    near = 0
    compared = 0
    for record in train:
        left = _ngrams(record.text)
        candidates: set[frozenset[str]] = set()
        for gram in left:
            candidates.update(inverted.get(gram, ()))
        for right in candidates:
            compared += 1
            union = len(left | right)
            if (len(left & right) / union if union else 1.0) >= _NEAR_THRESHOLD:
                near += 1
    if exact or near:
        raise TargetedRepairV4Error("cross-split exact or near duplicate gate failed")
    return {
        "exact_duplicate_pair_count": exact,
        "near_duplicate_pair_count": near,
        "candidate_pair_count": compared,
        "method": "character_5gram_jaccard",
        "threshold": _NEAR_THRESHOLD,
    }


def _cross_split_isolation(records: Sequence[DocumentRecord]) -> dict[str, Any]:
    fields: dict[str, dict[str, set[str]]] = {
        key: {split: set() for split in _SPLITS}
        for key in (
            "document_ids",
            "template_groups",
            "value_groups",
            "structured_surface_hashes",
            "relation_groups",
            "text_hashes",
        )
    }
    for record in records:
        split = str(record.split)
        fields["document_ids"][split].add(record.doc_id)
        fields["template_groups"][split].add(str(record.template_group))
        fields["relation_groups"][split].add(str(record.conversation_group))
        fields["text_hashes"][split].add(hashlib.sha256(record.text.encode()).hexdigest())
        fields["value_groups"][split].update(record.entity_value_groups)
        non_entity = record.metadata.get("non_entity_value_groups")
        if not isinstance(non_entity, list) or not non_entity:
            raise TargetedRepairV4Error("non-entity grouping hashes are missing")
        fields["value_groups"][split].update(str(item) for item in non_entity)
        fields["structured_surface_hashes"][split].update(
            stable_text_hash(value) for _, _, value in structured_matches(record.text)
        )
    result: dict[str, Any] = {}
    for name, by_split in fields.items():
        collisions = by_split["train"] & by_split["validation"]
        if collisions:
            raise TargetedRepairV4Error(f"cross-split {name} isolation failed")
        result[name] = {
            "train": len(by_split["train"]),
            "validation": len(by_split["validation"]),
            "cross_split_collisions": 0,
        }
    return result


def validate_records(records: Sequence[DocumentRecord], recipe: RepairRecipe) -> Mapping[str, Any]:
    if len(records) != sum(recipe.counts.values()):
        raise TargetedRepairV4Error("record count changed")
    by_split = {split: [record for record in records if record.split == split] for split in _SPLITS}
    coverage: dict[str, Any] = {}
    surface_audit: dict[str, Any] = {}
    template_audit: dict[str, Any] = {}
    pairing_audit: dict[str, Any] = {}
    positive_groups = {
        split: {
            entity.label: set()
            for record in by_split[split]
            for entity in record.entities
        }
        for split in _SPLITS
    }
    for split in _SPLITS:
        selected = by_split[split]
        if len(selected) != recipe.counts[split] or len({r.doc_id for r in selected}) != len(selected):
            raise TargetedRepairV4Error("split count or document identity changed")
        for record in selected:
            record.validate()
            if (
                record.data_pool is not DataPool.PUBLIC_RELEASE
                or record.public_weight_training_allowed is not True
                or record.metadata.get("targeted_repair_v4") is not True
                or record.metadata.get("D0_development_only") is not True
                or record.metadata.get("D0_raw_row_text_accessed_during_design") is not True
                or record.metadata.get("D0_raw_text_copied") is not False
                or record.metadata.get("D0_entity_value_read") is not False
                or record.metadata.get("D0_confirmatory_evaluation_eligible") is not False
                or record.metadata.get("v3_aggregate_diagnostic_only") is not True
                or record.metadata.get("v3_row_template_value_model_ancestry") is not False
                or record.metadata.get("v2_row_template_value_model_ancestry") is not False
                or record.metadata.get("test_data_read") is not False
                or record.metadata.get("hidden_data_read") is not False
                or record.metadata.get("external_benchmark_read") is not False
                or record.metadata.get("PII_Bench_execution_count") != 0
            ):
                raise TargetedRepairV4Error("record provenance escaped v4 boundary")
            if any(token in record.text for token in _BANNED_SHORTCUTS):
                raise TargetedRepairV4Error("text contains a synthetic shortcut token")
            for entity in record.entities:
                positive_groups[split][entity.label].add(stable_text_hash(entity.text))
                if entity.label == "DATE_OF_BIRTH":
                    parsed = _parse_frozen_date(entity.text)
                    age_days = (_REFERENCE_DATE - parsed).days
                    if parsed > _REFERENCE_DATE or not 18 * 365 <= age_days <= 80 * 366:
                        raise TargetedRepairV4Error("birth date is not plausible at frozen reference date")
        labels = Counter(entity.label for record in selected for entity in record.entities)
        expected_labels = {label: recipe.positive_per_label[split] for label in CORE_LABELS}
        if dict(labels) != expected_labels:
            raise TargetedRepairV4Error("positive label allocation changed")
        negatives = [record for record in selected if not record.entities]
        categories = Counter(str(record.metadata.get("hard_negative_category")) for record in negatives)
        if dict(categories) != dict(_NEGATIVE_COUNTS[split]):
            raise TargetedRepairV4Error("negative category allocation changed")
        lengths = Counter(str(record.metadata.get("length_bucket")) for record in negatives)
        if dict(lengths) != _EXPECTED_LENGTH_COUNTS[split]:
            raise TargetedRepairV4Error("negative length allocation changed")
        observed_matrix: dict[str, dict[str, Counter[str]]] = {
            category: {bucket: Counter() for bucket in _BUCKETS}
            for category in _NEGATIVE_COUNTS[split]
        }
        role_shape_counts = {"16": Counter(), "18": Counter()}
        template_counts: dict[str, set[str]] = {category: set() for category in _NEGATIVE_COUNTS[split]}
        total_matches = 0
        for record in negatives:
            category = str(record.metadata["hard_negative_category"])
            bucket = str(record.metadata["length_bucket"])
            cue = str(record.metadata["negative_context_style"])
            observed_matrix[category][bucket][cue] += 1
            low, high = _BUCKET_RANGES[bucket]
            if not low <= len(record.text) <= high:
                raise TargetedRepairV4Error("declared length bucket differs from text")
            span = record.metadata.get("candidate_span")
            if not isinstance(span, list) or len(span) != 2:
                raise TargetedRepairV4Error("candidate span is missing")
            candidate = record.text[span[0] : span[1]]
            matches = structured_matches(record.text)
            total_matches += len(matches)
            if (
                matches != ((span[0], span[1], candidate),)
                or record.text.count(candidate) != 1
                or record.metadata.get("candidate_occurrence_count") != 1
                or record.metadata.get("structured_surface_count") != 1
                or record.metadata.get("CASE_structured_identifier_present") is not False
                or record.metadata.get("auxiliary_ascii_or_digit_identifier_present") is not False
            ):
                raise TargetedRepairV4Error("single structured surface gate failed")
            if cue == "explicit" and not any(token in record.text for token in _EXPLICIT_CUES):
                raise TargetedRepairV4Error("explicit negative lacks its cue")
            if cue == "natural" and any(token in record.text for token in _EXPLICIT_CUES):
                raise TargetedRepairV4Error("natural negative leaks an explicit cue")
            if len(candidate) in {16, 18}:
                role_shape_counts[str(len(candidate))][str(record.metadata["decoy_role_family"])] += 1
            template_counts[category].add(str(record.template_group))
            if category == "invalid_card_single_surface" and validate_luhn(candidate).valid:
                raise TargetedRepairV4Error("invalid card unexpectedly passes Luhn")
            if category == "invalid_resident_id_single_surface" and validate_cn_resident_id(candidate).valid:
                raise TargetedRepairV4Error("invalid resident ID unexpectedly passes")
            if category == "valid_card_non_pii_counterfactual" and not validate_luhn(candidate).valid:
                raise TargetedRepairV4Error("valid card counterfactual is invalid")
            if category == "valid_resident_id_non_pii_counterfactual" and not validate_cn_resident_id(candidate).valid:
                raise TargetedRepairV4Error("valid resident ID counterfactual is invalid")
            if category.startswith("invalid_") and stable_text_hash(candidate) in {
                group for groups in positive_groups[split].values() for group in groups
            }:
                raise TargetedRepairV4Error("invalid checksum surface is paired with a positive")
        if observed_matrix != {
            category: {bucket: Counter(cues) for bucket, cues in buckets.items()}
            for category, buckets in _NEGATIVE_MATRIX[split].items()
        }:
            raise TargetedRepairV4Error("negative category/bucket/cue cross-count changed")
        if total_matches != len(negatives):
            raise TargetedRepairV4Error("aggregate structured surface total is not one per negative")
        minimum = 12 if split == "train" else 4
        if any(len(groups) < minimum for groups in template_counts.values()):
            raise TargetedRepairV4Error("negative template diversity gate failed")
        for shape in ("16", "18"):
            if set(role_shape_counts[shape]) != set(_ROLE_FAMILIES) or min(
                role_shape_counts[shape].values()
            ) < 6:
                raise TargetedRepairV4Error("confusable role-family coverage gate failed")
        positive_template_counts = {
            label: len(
                {
                    str(record.template_group)
                    for record in selected
                    if record.entities and record.entities[0].label == label
                }
            )
            for label in CORE_LABELS
        }
        positive_minimum = 4 if split == "train" else 2
        if any(
            count
            < (
                4
                if split == "train"
                else (3 if label in _SPECIAL_POSITIVE_BODIES else positive_minimum)
            )
            for label, count in positive_template_counts.items()
        ):
            raise TargetedRepairV4Error("label-specific positive template gate failed")
        special_positive_lengths: dict[str, dict[str, int]] = {}
        for label in _SPECIAL_POSITIVE_BODIES:
            observed = Counter(
                str(record.metadata.get("positive_length_bucket"))
                for record in selected
                if record.entities and record.entities[0].label == label
            )
            expected = (
                {"short": 60, "medium": 26, "long": 18}
                if split == "train" and label == "BANK_CARD_NUMBER"
                else {"short": 64, "medium": 26, "long": 14}
                if split == "train" and label == "CN_RESIDENT_ID"
                else {"short": 64, "medium": 24, "long": 16}
                if split == "train"
                else {"short": 16, "medium": 6, "long": 4}
            )
            if dict(observed) != expected:
                raise TargetedRepairV4Error("special positive length allocation changed")
            special_positive_lengths[label] = dict(sorted(observed.items()))
        priority_template_shape_counts: dict[str, dict[str, dict[str, int]]] = {}
        for label in sorted(_PRIORITY_SHAPE_LABELS):
            joint: dict[str, Counter[str]] = {}
            for record in selected:
                if not record.entities or record.entities[0].label != label:
                    continue
                template_variant = str(record.metadata.get("positive_template_variant"))
                shape_variant = str(record.metadata.get("positive_shape_variant"))
                joint.setdefault(template_variant, Counter())[shape_variant] += 1
            expected_template_count = 4 if split == "train" else 2
            if set(joint) != {str(index) for index in range(expected_template_count)}:
                raise TargetedRepairV4Error("priority template inventory changed")
            for counts in joint.values():
                if set(counts) != {"0", "1", "2", "3"} or max(counts.values()) - min(
                    counts.values()
                ) > 1:
                    raise TargetedRepairV4Error("priority label template/shape balance failed")
            priority_template_shape_counts[label] = {
                template: dict(sorted(counts.items())) for template, counts in sorted(joint.items())
            }
        coverage[split] = {
            "records": len(selected),
            "positive_labels": dict(sorted(labels.items())),
            "negative_categories": dict(sorted(categories.items())),
            "negative_length_buckets": dict(sorted(lengths.items())),
            "negative_category_bucket_cue_matrix": {
                category: {
                    bucket: dict(sorted(cues.items())) for bucket, cues in buckets.items()
                }
                for category, buckets in observed_matrix.items()
            },
        }
        surface_audit[split] = {
            "negative_documents": len(negatives),
            "structured_surface_total": total_matches,
            "expected_structured_surface_total": len(negatives),
            "documents_with_exactly_one_structured_surface": len(negatives),
            "candidate_occurrence_count_exactly_one": len(negatives),
            "CASE_structured_identifier_documents": 0,
            "auxiliary_structured_identifier_documents": 0,
            "sixteen_character_role_counts": dict(sorted(role_shape_counts["16"].items())),
            "eighteen_character_role_counts": dict(sorted(role_shape_counts["18"].items())),
            "gate_passed": True,
        }
        template_audit[split] = {
            "negative_unique_templates_by_category": {
                key: len(value) for key, value in sorted(template_counts.items())
            },
            "positive_unique_label_specific_templates": positive_template_counts,
            "special_positive_length_buckets": special_positive_lengths,
            "priority_label_template_shape_joint_counts": priority_template_shape_counts,
            "negative_minimum_required": minimum,
            "positive_minimum_required": positive_minimum,
            "gate_passed": True,
        }

    relation_members: dict[str, list[DocumentRecord]] = {}
    for record in records:
        relation_members.setdefault(str(record.conversation_group), []).append(record)
    observed_pairs = {split: Counter() for split in _SPLITS}
    observed_pair_buckets = {
        split: {label: Counter() for label in _EXPECTED_PAIRS[split]} for split in _SPLITS
    }
    observed_email_alipay_pairs = Counter()
    relation_label = {
        "card_valid_exact_surface": "BANK_CARD_NUMBER",
        "resident_id_valid_exact_surface": "CN_RESIDENT_ID",
        "date_exact_surface": "DATE_OF_BIRTH",
    }
    for members in relation_members.values():
        kind = str(members[0].metadata.get("relation_kind"))
        if kind == "email_alipay_exact_surface_semantic_pair":
            if (
                len(members) != 2
                or len({member.split for member in members}) != 1
                or {member.entities[0].label for member in members} != {
                    "EMAIL_ADDRESS",
                    "ALIPAY_ACCOUNT",
                }
                or len({member.entity_value_groups[0] for member in members}) != 1
            ):
                raise TargetedRepairV4Error("email/Alipay semantic pair is not atomic")
            email = next(member for member in members if member.entities[0].label == "EMAIL_ADDRESS")
            alipay = next(
                member for member in members if member.entities[0].label == "ALIPAY_ACCOUNT"
            )
            if not any(token in email.text for token in ("邮箱", "电子邮件")) or "支付宝" not in alipay.text:
                raise TargetedRepairV4Error("email/Alipay pair lacks explicit label semantics")
            observed_email_alipay_pairs[str(members[0].split)] += 1
            continue
        if kind not in relation_label:
            continue
        if len(members) != 2 or {member.split for member in members} not in ({"train"}, {"validation"}):
            raise TargetedRepairV4Error("counterfactual relation crosses split or is not atomic")
        positive = [member for member in members if member.entities]
        negative = [member for member in members if not member.entities]
        if len(positive) != 1 or len(negative) != 1:
            raise TargetedRepairV4Error("counterfactual relation lacks one positive and one negative")
        if not set(positive[0].entity_value_groups) & set(
            negative[0].metadata["non_entity_value_groups"]
        ):
            raise TargetedRepairV4Error("counterfactual relation does not share exact surface")
        split = str(members[0].split)
        label = relation_label[kind]
        positive_bucket = positive[0].metadata.get("positive_length_bucket")
        negative_bucket = negative[0].metadata.get("length_bucket")
        if positive_bucket != negative_bucket or positive_bucket not in _BUCKETS:
            raise TargetedRepairV4Error("counterfactual pair length buckets do not match")
        observed_pairs[split][label] += 1
        observed_pair_buckets[split][label][str(positive_bucket)] += 1
    for split in _SPLITS:
        if dict(observed_pairs[split]) != _EXPECTED_PAIRS[split]:
            raise TargetedRepairV4Error("counterfactual pair counts changed")
        for label in _EXPECTED_PAIRS[split]:
            if dict(observed_pair_buckets[split][label]) != _PAIR_BUCKET_COUNTS[split][label]:
                raise TargetedRepairV4Error("counterfactual pair bucket allocation changed")
        pairing_audit[split] = {
            "labels": dict(sorted(observed_pairs[split].items())),
            "label_length_buckets": {
                label: dict(sorted(counts.items()))
                for label, counts in observed_pair_buckets[split].items()
            },
        }
        if observed_email_alipay_pairs[split] != recipe.positive_per_label[split]:
            raise TargetedRepairV4Error("email/Alipay semantic pair count changed")

    train_templates = {str(record.template_group) for record in by_split["train"]}
    validation_templates = {str(record.template_group) for record in by_split["validation"]}
    if train_templates & validation_templates:
        raise TargetedRepairV4Error("train/validation template overlap")
    train_body_hashes = {
        str(record.metadata.get("template_body_sha256")) for record in by_split["train"]
    }
    validation_body_hashes = {
        str(record.metadata.get("template_body_sha256")) for record in by_split["validation"]
    }
    train_skeleton_hashes = {
        str(record.metadata.get("template_semantic_skeleton_sha256"))
        for record in by_split["train"]
    }
    validation_skeleton_hashes = {
        str(record.metadata.get("template_semantic_skeleton_sha256"))
        for record in by_split["validation"]
    }
    if train_body_hashes & validation_body_hashes or train_skeleton_hashes & validation_skeleton_hashes:
        raise TargetedRepairV4Error("train/validation body or semantic skeleton overlap")
    isolation = _cross_split_isolation(records)
    duplicates = _cross_split_duplicate_audit(records)
    return MappingProxyType(
        {
            "counts": {
                "total": len(records),
                "by_split": {split: len(by_split[split]) for split in _SPLITS},
                "pii_free_by_split": {
                    split: sum(not record.entities for record in by_split[split]) for split in _SPLITS
                },
            },
            "coverage": coverage,
            "single_surface_audit": surface_audit,
            "template_audit": template_audit,
            "counterfactual_pairing": {
                "scope": "within_split_only",
                "pairs_by_split": pairing_audit,
                "cross_split_pairs": 0,
            },
            "email_alipay_same_shape_semantic_pairs": {
                "method": "within_split_exact_surface_distinct_explicit_semantics",
                "pairs_by_split": dict(sorted(observed_email_alipay_pairs.items())),
                "cross_split_pairs": 0,
                "gate_passed": True,
            },
            "duplicate_audit": duplicates,
            "isolation": isolation,
            "template_split_isolation": {
                "template_group_collisions": 0,
                "body_hash_collisions": 0,
                "semantic_skeleton_hash_collisions": 0,
            },
            "lineage_gate": {
                "v3_rows_templates_surface_values_model_state_inherited": False,
                "v3_aggregate_diagnostic_only": True,
                "D0_development_only": True,
                "D0_confirmatory_evaluation_eligible": False,
                "test_data_read": False,
                "hidden_data_read": False,
                "external_benchmark_read": False,
                "PII_Bench_execution_count": 0,
                "gate_passed": True,
            },
        }
    )


def build_repair_dataset(
    *, recipe_path: Path, freeze_receipt_path: Path
) -> RepairBuild:
    recipe = load_recipe(recipe_path)
    from .targeted_repair_v4_freeze import (
        TargetedRepairV4FreezeError,
        verify_targeted_repair_v4_freeze,
    )

    try:
        freeze_receipt = verify_targeted_repair_v4_freeze(
            receipt_path=freeze_receipt_path, recipe_path=recipe_path
        )
    except TargetedRepairV4FreezeError as exc:
        raise TargetedRepairV4Error("verified v4 freeze receipt is required") from exc
    records = generate_records(recipe)
    validation = validate_records(records, recipe)
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "artifact_type": "synthetic_targeted_data_repair_v4_single_surface",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "release_status": "development_optimization_only_not_release_or_claim_evidence",
        "data_pool": DataPool.PUBLIC_RELEASE.value,
        "public_weight_training_allowed": True,
        "recipe": {
            "id": "pii_zh_synthetic_targeted_repair_v4_single_surface_transfer",
            "version": DATASET_VERSION,
            "sha256": recipe.sha256,
            "seed": recipe.seed,
        },
        "freeze_receipt": dict(freeze_receipt),
        "v3_diagnostic_only_boundary": dict(recipe.v3_boundary),
        "source_contract": {
            "context_origin": "independently_authored_deterministic_v4_templates",
            "D0_development_template_shape_influenced_design": True,
            "D0_raw_row_text_accessed_for_masked_template_derivation": True,
            "D0_raw_text_copied": False,
            "D0_entity_value_read": False,
            "D0_confirmatory_evaluation_eligible": False,
            "v3_aggregate_diagnostic_read": True,
            "v3_row_template_value_recipe_model_artifact_read": False,
            "v3_ancestry_eligible": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "external_benchmark_read": False,
            "PII_Bench_execution_count": 0,
            "LLM_generation_used": False,
        },
        "generation": {
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "base_model_restart_required": True,
            "split_independent_rng": True,
            "output_splits": list(_SPLITS),
        },
        **dict(validation),
        "publication_boundary": {
            "optimization_only": True,
            "supports_best_open_model_claim": False,
            "real_or_consented_context_evaluation_required": True,
            "new_unseen_hidden_confirmatory_evaluation_required": True,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return RepairBuild(tuple(records), MappingProxyType(manifest))


def materialize(build: RepairBuild, output_dir: Path) -> Mapping[str, Any]:
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise TargetedRepairV4Error("output must be a new absolute directory")
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
                raise TargetedRepairV4Error("materialized split readback failed")
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
        manifest["manifest_sha256"] = canonical_hash(manifest)
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


def read_materialized(dataset_dir: Path) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
    records: list[DocumentRecord] = []
    files: list[dict[str, Any]] = []
    for split in _SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        selected = list(iter_jsonl(path))
        if any(record.split != split for record in selected):
            raise TargetedRepairV4Error("materialized record escaped its split")
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
    *, dataset_dir: Path, recipe_path: Path, freeze_receipt_path: Path
) -> Mapping[str, Any]:
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise TargetedRepairV4Error("dataset directory must be regular")
    expected_names = {"train.jsonl", "validation.jsonl", "dataset_manifest.json"}
    children = list(dataset_dir.iterdir())
    if {path.name for path in children} != expected_names or any(path.is_symlink() for path in children):
        raise TargetedRepairV4Error("dataset file inventory changed")
    recipe = load_recipe(recipe_path)
    expected_build = build_repair_dataset(
        recipe_path=recipe_path, freeze_receipt_path=freeze_receipt_path
    )
    records, files = read_materialized(dataset_dir)
    recomputed = validate_records(records, recipe)
    manifest_path = dataset_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV4Error("dataset manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise TargetedRepairV4Error("dataset manifest must be an object")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != canonical_hash(unsigned):
        raise TargetedRepairV4Error("dataset manifest logical hash failed")
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("files") != files
        or manifest.get("recipe", {}).get("sha256") != recipe.sha256
        or manifest.get("freeze_receipt") != dict(expected_build.manifest["freeze_receipt"])
    ):
        raise TargetedRepairV4Error("dataset manifest input binding changed")
    sections = (
        "counts",
        "coverage",
        "single_surface_audit",
        "template_audit",
        "counterfactual_pairing",
        "email_alipay_same_shape_semantic_pairs",
        "duplicate_audit",
        "isolation",
        "template_split_isolation",
        "lineage_gate",
    )
    for section in sections:
        if manifest.get(section) != recomputed.get(section):
            raise TargetedRepairV4Error(f"dataset manifest {section} differs from recomputation")
    with tempfile.TemporaryDirectory(prefix="repair-v4-independent-", dir=dataset_dir.parent) as root:
        replay_dir = Path(root) / "replay"
        materialize(expected_build, replay_dir)
        byte_hashes: dict[str, str] = {}
        for name in sorted(expected_names):
            original = dataset_dir / name
            replay = replay_dir / name
            if original.read_bytes() != replay.read_bytes():
                raise TargetedRepairV4Error("independent rematerialization is not byte-identical")
            byte_hashes[name] = sha256_file(original)
    return MappingProxyType(
        {
            "schema_version": 1,
            "artifact_type": "targeted_repair_v4_independent_validation",
            "status": "passed",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "manifest_sha256": claimed,
            "manifest_file_sha256": sha256_file(manifest_path),
            "recipe_file_sha256": recipe.sha256,
            "files": files,
            **dict(recomputed),
            "independent_rematerialization": {
                "executed": True,
                "byte_identical": True,
                "files": byte_hashes,
            },
            "source_contract": dict(manifest["source_contract"]),
            "v3_diagnostic_only_boundary": dict(recipe.v3_boundary),
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
    "TargetedRepairV4Error",
    "build_repair_dataset",
    "canonical_hash",
    "generate_records",
    "load_recipe",
    "materialize",
    "read_materialized",
    "sha256_file",
    "structured_matches",
    "validate_records",
    "validate_materialized",
]
