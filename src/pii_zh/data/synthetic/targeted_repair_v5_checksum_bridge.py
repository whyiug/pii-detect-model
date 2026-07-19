"""Deterministic v5a checksum-context rewrite over the frozen v4 dataset.

The output has exactly the same row count and ordering as v4. Only checksum-
invalid card and resident-ID negatives are replaced. All other ``DocumentRecord``
objects serialize byte-for-byte as before. This is development-only data and is
not confirmatory or publication-claim evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ..schema import (
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    dumps_document,
    iter_jsonl,
    stable_text_hash,
    write_jsonl,
)
from ..validators import validate_cn_resident_id, validate_luhn

DATASET_ID = "pii_zh_synthetic_targeted_repair_v5a_checksum_context_rewrite"
DATASET_VERSION = "0.5.0-dev1"
GENERATOR_NAME = "pii_zh_targeted_repair_v5a_checksum_context_rewrite"
GENERATOR_VERSION = DATASET_VERSION
MATERIALIZER_VERSION = DATASET_VERSION
DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs/data/targeted_repair_v5_checksum_context_bridge.yaml"
)
SPLITS = ("train", "validation")
CATEGORIES = (
    "invalid_card_single_surface",
    "invalid_resident_id_single_surface",
)
BUCKETS = ("short", "medium", "long")
ORDER_MODES = (
    "A_all_before",
    "B_surface_first",
    "C_split_verdict",
    "D_domain_first",
)
BUCKET_RANGES = {"short": (18, 30), "medium": (31, 45), "long": (46, 60)}
PLACEHOLDER = "<<CANDIDATE>>"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONTEXT_ASCII_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9]")
_STRUCTURED_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9@._:/+*\-]{5,}[A-Za-z0-9])"
    r"(?![A-Za-z0-9])"
)


class TargetedRepairV5BridgeError(ValueError):
    """Raised without reflecting candidate values or absolute paths."""


@dataclass(frozen=True, slots=True)
class SourceFileContract:
    name: str
    records: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BridgeRecipe:
    path: Path
    sha256: str
    seed: int
    source_root: Path
    source_dataset_id: str
    source_dataset_version: str
    source_manifest_file_sha256: str
    source_manifest_sha256: str
    source_files: Mapping[str, SourceFileContract]
    rewrite_matrix: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]]
    rewritten_counts: Mapping[str, int]
    unchanged_counts: Mapping[str, int]
    output_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    body: str
    domain_marker: str
    action_marker: str
    verdict_before: str | None
    verdict_after: str | None
    cue_phrase: str


@dataclass(frozen=True, slots=True)
class BridgeBuild:
    source_records: tuple[DocumentRecord, ...]
    records: tuple[DocumentRecord, ...]
    rewritten_records: tuple[DocumentRecord, ...]
    manifest: Mapping[str, Any]
    recipe: BridgeRecipe


def _spec(
    body: str,
    domain: str,
    action: str,
    *,
    before: str | None,
    after: str | None,
    cue: str,
) -> TemplateSpec:
    return TemplateSpec(body, domain, action, before, after, cue)


# Every family/order/length cell has one explicit authored skeleton. The two
# split banks use disjoint checksum terminology and cue phrases.
_TEMPLATES: Mapping[str, Mapping[str, Mapping[str, Mapping[str, TemplateSpec]]]] = MappingProxyType(
    {
        "train": {
            "invalid_card_single_surface": {
                "A_all_before": {
                    "short": _spec(
                        "卡号校验位失败，待查：<<CANDIDATE>>",
                        "卡号",
                        "校验位",
                        before="失败",
                        after=None,
                        cue="卡号校验位失败",
                    ),
                    "medium": _spec(
                        "结算卡号的校验位未通过，当前待核对象：<<CANDIDATE>>，请暂缓办理。",
                        "结算卡号",
                        "校验位",
                        before="未通过",
                        after=None,
                        cue="结算卡号的校验位未通过",
                    ),
                    "long": _spec(
                        "账务登记所载卡号的校验位不成立，当前待核对象：<<CANDIDATE>>，请留在内部复查队列等待后续处置。",
                        "账务登记所载卡号",
                        "校验位",
                        before="不成立",
                        after=None,
                        cue="账务登记所载卡号的校验位不成立",
                    ),
                },
                "B_surface_first": {
                    "short": _spec(
                        "<<CANDIDATE>>卡号校验位失败。",
                        "卡号",
                        "校验位",
                        before=None,
                        after="失败",
                        cue="卡号校验位失败",
                    ),
                    "medium": _spec(
                        "<<CANDIDATE>>作为结算卡号，经校验位核验后判定未通过，请暂缓办理。",
                        "结算卡号",
                        "校验位核验",
                        before=None,
                        after="未通过",
                        cue="校验位核验后判定未通过",
                    ),
                    "long": _spec(
                        "<<CANDIDATE>>作为账务登记中的卡号，经校验位复核后判定无效，请留在内部复查队列等待后续处置。",
                        "账务登记中的卡号",
                        "校验位复核",
                        before=None,
                        after="无效",
                        cue="校验位复核后判定无效",
                    ),
                },
                "C_split_verdict": {
                    "short": _spec(
                        "校验位失败，<<CANDIDATE>>卡号无效。",
                        "卡号",
                        "校验位",
                        before="失败",
                        after="无效",
                        cue="校验位失败后仍无效",
                    ),
                    "medium": _spec(
                        "校验位核验初判未通过，<<CANDIDATE>>作为结算卡号复查后仍属无效。",
                        "结算卡号",
                        "校验位核验",
                        before="未通过",
                        after="无效",
                        cue="校验位核验初判未通过后仍无效",
                    ),
                    "long": _spec(
                        "校验位核验初判不成立，<<CANDIDATE>>作为账务登记中的卡号经专员复查后仍判无效，请暂缓后续办理。",
                        "账务登记中的卡号",
                        "校验位核验",
                        before="不成立",
                        after="无效",
                        cue="校验位核验初判不成立后仍判无效",
                    ),
                },
                "D_domain_first": {
                    "short": _spec(
                        "卡号<<CANDIDATE>>校验位失败。",
                        "卡号",
                        "校验位",
                        before=None,
                        after="失败",
                        cue="卡号之后校验位失败",
                    ),
                    "medium": _spec(
                        "结算卡号<<CANDIDATE>>经校验位核验后判定未通过，请暂缓办理。",
                        "结算卡号",
                        "校验位核验",
                        before=None,
                        after="未通过",
                        cue="结算卡号之后校验位核验未通过",
                    ),
                    "long": _spec(
                        "账务登记中的卡号<<CANDIDATE>>经校验位复核后判定不成立，请留在内部复查队列等待后续处置。",
                        "账务登记中的卡号",
                        "校验位复核",
                        before=None,
                        after="不成立",
                        cue="账务登记卡号之后校验位复核不成立",
                    ),
                },
            },
            "invalid_resident_id_single_surface": {
                "A_all_before": {
                    "short": _spec(
                        "证号校验位失败，待查：<<CANDIDATE>>",
                        "证号",
                        "校验位",
                        before="失败",
                        after=None,
                        cue="证号校验位失败",
                    ),
                    "medium": _spec(
                        "实名证号的校验位未通过，当前待核对象：<<CANDIDATE>>，请暂缓办理。",
                        "实名证号",
                        "校验位",
                        before="未通过",
                        after=None,
                        cue="实名证号的校验位未通过",
                    ),
                    "long": _spec(
                        "身份登记所载证号的校验位不成立，当前待核对象：<<CANDIDATE>>，请留在内部复查队列等待后续处置。",
                        "身份登记所载证号",
                        "校验位",
                        before="不成立",
                        after=None,
                        cue="身份登记所载证号的校验位不成立",
                    ),
                },
                "B_surface_first": {
                    "short": _spec(
                        "<<CANDIDATE>>证号校验位失败。",
                        "证号",
                        "校验位",
                        before=None,
                        after="失败",
                        cue="证号校验位失败",
                    ),
                    "medium": _spec(
                        "<<CANDIDATE>>作为实名证号，经校验位核验后判定未通过，请暂缓办理。",
                        "实名证号",
                        "校验位核验",
                        before=None,
                        after="未通过",
                        cue="实名证号校验位核验后判定未通过",
                    ),
                    "long": _spec(
                        "<<CANDIDATE>>作为身份登记中的证号，经校验位复核后判定无效，请留在内部复查队列等待后续处置。",
                        "身份登记中的证号",
                        "校验位复核",
                        before=None,
                        after="无效",
                        cue="身份登记证号校验位复核后判定无效",
                    ),
                },
                "C_split_verdict": {
                    "short": _spec(
                        "校验位失败，<<CANDIDATE>>证号无效。",
                        "证号",
                        "校验位",
                        before="失败",
                        after="无效",
                        cue="证号校验位失败后仍无效",
                    ),
                    "medium": _spec(
                        "校验位核验初判未通过，<<CANDIDATE>>作为实名证号复查后仍属无效。",
                        "实名证号",
                        "校验位核验",
                        before="未通过",
                        after="无效",
                        cue="实名证号校验位初判未通过后仍无效",
                    ),
                    "long": _spec(
                        "校验位核验初判不成立，<<CANDIDATE>>作为身份登记中的证号经专员复查后仍判无效，请暂缓后续办理。",
                        "身份登记中的证号",
                        "校验位核验",
                        before="不成立",
                        after="无效",
                        cue="身份登记证号校验位不成立后仍判无效",
                    ),
                },
                "D_domain_first": {
                    "short": _spec(
                        "证号<<CANDIDATE>>校验位失败。",
                        "证号",
                        "校验位",
                        before=None,
                        after="失败",
                        cue="证号之后校验位失败",
                    ),
                    "medium": _spec(
                        "实名证号<<CANDIDATE>>经校验位核验后判定未通过，请暂缓办理。",
                        "实名证号",
                        "校验位核验",
                        before=None,
                        after="未通过",
                        cue="实名证号之后校验位核验未通过",
                    ),
                    "long": _spec(
                        "身份登记中的证号<<CANDIDATE>>经校验位复核后判定不成立，请留在内部复查队列等待后续处置。",
                        "身份登记中的证号",
                        "校验位复核",
                        before=None,
                        after="不成立",
                        cue="身份登记证号之后校验位复核不成立",
                    ),
                },
            },
        },
        "validation": {
            "invalid_card_single_surface": {
                "A_all_before": {
                    "short": _spec(
                        "支付卡核验码不符，待审：<<CANDIDATE>>",
                        "支付卡",
                        "核验码",
                        before="不符",
                        after=None,
                        cue="支付卡核验码不符",
                    ),
                    "medium": _spec(
                        "支付卡号核验码有误，当前待审对象：<<CANDIDATE>>，本轮暂不受理。",
                        "支付卡号",
                        "核验码",
                        before="有误",
                        after=None,
                        cue="支付卡号核验码有误",
                    ),
                    "long": _spec(
                        "清分资料所列支付卡号的核验码未获认可，当前待审对象：<<CANDIDATE>>，本轮转交后台人员复验后再作决定。",
                        "清分资料所列支付卡号",
                        "核验码",
                        before="未获认可",
                        after=None,
                        cue="清分资料支付卡号核验码未获认可",
                    ),
                },
                "B_surface_first": {
                    "short": _spec(
                        "<<CANDIDATE>>支付卡核验码有误。",
                        "支付卡",
                        "核验码",
                        before=None,
                        after="有误",
                        cue="支付卡核验码有误",
                    ),
                    "medium": _spec(
                        "<<CANDIDATE>>列作付款卡号，经核验码审查确认不符，本轮暂不受理。",
                        "付款卡号",
                        "核验码审查",
                        before=None,
                        after="不符",
                        cue="付款卡号核验码审查确认不符",
                    ),
                    "long": _spec(
                        "<<CANDIDATE>>列作清分资料中的支付卡号，经核验码审查确认遭拒，本轮转交后台人员复验后再作决定。",
                        "清分资料中的支付卡号",
                        "核验码审查",
                        before=None,
                        after="遭拒",
                        cue="清分支付卡号核验码审查确认遭拒",
                    ),
                },
                "C_split_verdict": {
                    "short": _spec(
                        "核验码遭拒，<<CANDIDATE>>支付卡无效。",
                        "支付卡",
                        "核验码",
                        before="遭拒",
                        after="无效",
                        cue="支付卡核验码遭拒后仍无效",
                    ),
                    "medium": _spec(
                        "核验码审查先判有误，<<CANDIDATE>>列作付款卡号复验后仍然不符。",
                        "付款卡号",
                        "核验码审查",
                        before="有误",
                        after="不符",
                        cue="付款卡号核验码先判有误后仍不符",
                    ),
                    "long": _spec(
                        "核验码审查先判未获认可，<<CANDIDATE>>列作清分资料中的支付卡号经后台人员复验后依旧不符，本轮暂不受理。",
                        "清分资料中的支付卡号",
                        "核验码审查",
                        before="未获认可",
                        after="不符",
                        cue="清分支付卡号核验码未获认可后仍不符",
                    ),
                },
                "D_domain_first": {
                    "short": _spec(
                        "支付卡<<CANDIDATE>>核验码不符。",
                        "支付卡",
                        "核验码",
                        before=None,
                        after="不符",
                        cue="支付卡之后核验码不符",
                    ),
                    "medium": _spec(
                        "付款卡号<<CANDIDATE>>经核验码审查确认有误，本轮暂不受理。",
                        "付款卡号",
                        "核验码审查",
                        before=None,
                        after="有误",
                        cue="付款卡号之后核验码审查有误",
                    ),
                    "long": _spec(
                        "清分资料中的支付卡号<<CANDIDATE>>经核验码审查确认未获认可，本轮转交后台人员复验后再作决定。",
                        "清分资料中的支付卡号",
                        "核验码审查",
                        before=None,
                        after="未获认可",
                        cue="清分支付卡号之后核验码未获认可",
                    ),
                },
            },
            "invalid_resident_id_single_surface": {
                "A_all_before": {
                    "short": _spec(
                        "居民证核验码不符，待审：<<CANDIDATE>>",
                        "居民证",
                        "核验码",
                        before="不符",
                        after=None,
                        cue="居民证核验码不符",
                    ),
                    "medium": _spec(
                        "居民证号核验码有误，待审对象：<<CANDIDATE>>，本轮暂不受理。",
                        "居民证号",
                        "核验码",
                        before="有误",
                        after=None,
                        cue="居民证号核验码有误",
                    ),
                    "long": _spec(
                        "认证资料所列居民证号的核验码未获认可，待审对象：<<CANDIDATE>>，转交后台人员复验后再决定。",
                        "认证资料所列居民证号",
                        "核验码",
                        before="未获认可",
                        after=None,
                        cue="认证资料居民证号核验码未获认可",
                    ),
                },
                "B_surface_first": {
                    "short": _spec(
                        "<<CANDIDATE>>居民证核验码有误。",
                        "居民证",
                        "核验码",
                        before=None,
                        after="有误",
                        cue="居民证核验码有误",
                    ),
                    "medium": _spec(
                        "<<CANDIDATE>>列作身份凭号，经核验码审查确认不符，本轮暂不受理。",
                        "身份凭号",
                        "核验码审查",
                        before=None,
                        after="不符",
                        cue="身份凭号核验码审查确认不符",
                    ),
                    "long": _spec(
                        "<<CANDIDATE>>列作认证资料中的居民证号，经核验码审查确认遭拒，本轮转交后台人员复验后再作决定。",
                        "认证资料中的居民证号",
                        "核验码审查",
                        before=None,
                        after="遭拒",
                        cue="认证居民证号核验码审查确认遭拒",
                    ),
                },
                "C_split_verdict": {
                    "short": _spec(
                        "核验码遭拒，<<CANDIDATE>>居民证无效。",
                        "居民证",
                        "核验码",
                        before="遭拒",
                        after="无效",
                        cue="居民证核验码遭拒后仍无效",
                    ),
                    "medium": _spec(
                        "核验码审查先判有误，<<CANDIDATE>>列作身份凭号复验后仍然不符。",
                        "身份凭号",
                        "核验码审查",
                        before="有误",
                        after="不符",
                        cue="身份凭号核验码先判有误后仍不符",
                    ),
                    "long": _spec(
                        "核验码审查先判未获认可，<<CANDIDATE>>列作认证资料中的居民证号经人员复验后仍不符，本轮不受理。",
                        "认证资料中的居民证号",
                        "核验码审查",
                        before="未获认可",
                        after="不符",
                        cue="认证居民证号核验码未获认可后仍不符",
                    ),
                },
                "D_domain_first": {
                    "short": _spec(
                        "居民证<<CANDIDATE>>核验码不符。",
                        "居民证",
                        "核验码",
                        before=None,
                        after="不符",
                        cue="居民证之后核验码不符",
                    ),
                    "medium": _spec(
                        "身份凭号<<CANDIDATE>>经核验码审查确认有误，本轮暂不受理。",
                        "身份凭号",
                        "核验码审查",
                        before=None,
                        after="有误",
                        cue="身份凭号之后核验码审查有误",
                    ),
                    "long": _spec(
                        "认证资料中的居民证号<<CANDIDATE>>经核验码审查确认未获认可，本轮转交后台人员复验后再作决定。",
                        "认证资料中的居民证号",
                        "核验码审查",
                        before=None,
                        after="未获认可",
                        cue="认证居民证号之后核验码未获认可",
                    ),
                },
            },
        },
    }
)


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetedRepairV5BridgeError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TargetedRepairV5BridgeError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TargetedRepairV5BridgeError(f"{field} must be a lowercase SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise TargetedRepairV5BridgeError(f"{field} fields changed")


def _parse_split_counts(value: Any, *, field: str) -> dict[str, int]:
    raw = _mapping(value, field=field)
    _exact_keys(raw, set(SPLITS), field=field)
    return {split: _positive_int(raw[split], field=f"{field}.{split}") for split in SPLITS}


def _parse_matrix(
    value: Any,
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    raw = _mapping(value, field="rewrite_matrix.expected")
    _exact_keys(raw, set(SPLITS), field="rewrite_matrix.expected")
    result: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for split in SPLITS:
        by_family = _mapping(raw[split], field=f"matrix.{split}")
        _exact_keys(by_family, set(CATEGORIES), field=f"matrix.{split}")
        result[split] = {}
        for category in CATEGORIES:
            by_bucket = _mapping(by_family[category], field=f"matrix.{split}.{category}")
            _exact_keys(by_bucket, set(BUCKETS), field=f"matrix.{split}.{category}")
            result[split][category] = {}
            for bucket in BUCKETS:
                by_order = _mapping(by_bucket[bucket], field=f"matrix.{split}.{category}.{bucket}")
                _exact_keys(
                    by_order,
                    set(ORDER_MODES),
                    field=f"matrix.{split}.{category}.{bucket}",
                )
                result[split][category][bucket] = {
                    order: _positive_int(
                        by_order[order],
                        field=f"matrix.{split}.{category}.{bucket}.{order}",
                    )
                    for order in ORDER_MODES
                }
    return result


def load_recipe(path: Path = DEFAULT_RECIPE_PATH) -> BridgeRecipe:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise TargetedRepairV5BridgeError("recipe must be an absolute regular file")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TargetedRepairV5BridgeError("recipe is unreadable") from exc
    root = _mapping(raw, field="recipe")
    if (
        root.get("schema_version") != 1
        or root.get("recipe_id") != DATASET_ID
        or root.get("recipe_version") != DATASET_VERSION
        or root.get("status") != "development_only_unfrozen"
    ):
        raise TargetedRepairV5BridgeError("recipe identity or status changed")
    source = _mapping(root.get("source_v4"), field="source_v4")
    source_root_raw = source.get("dataset_root")
    if not isinstance(source_root_raw, str) or not Path(source_root_raw).is_absolute():
        raise TargetedRepairV5BridgeError("source v4 root must be an absolute string")
    raw_files = _mapping(source.get("files"), field="source_v4.files")
    _exact_keys(raw_files, set(SPLITS), field="source_v4.files")
    source_files: dict[str, SourceFileContract] = {}
    for split in SPLITS:
        item = _mapping(raw_files[split], field=f"source_v4.files.{split}")
        _exact_keys(item, {"name", "records", "sha256"}, field=f"source files {split}")
        if item.get("name") != f"{split}.jsonl":
            raise TargetedRepairV5BridgeError("source split filename changed")
        source_files[split] = SourceFileContract(
            name=str(item["name"]),
            records=_positive_int(item.get("records"), field=f"source {split} records"),
            sha256=_sha256(item.get("sha256"), field=f"source {split} sha256"),
        )
    matrix_root = _mapping(root.get("rewrite_matrix"), field="rewrite_matrix")
    if matrix_root.get("order_modes") != list(ORDER_MODES) or matrix_root.get("length_buckets") != {
        bucket: list(BUCKET_RANGES[bucket]) for bucket in BUCKETS
    }:
        raise TargetedRepairV5BridgeError("order or length-bucket contract changed")
    matrix = _parse_matrix(matrix_root.get("expected"))
    rewritten = _parse_split_counts(matrix_root.get("rewritten_records"), field="rewritten_records")
    unchanged = _parse_split_counts(matrix_root.get("unchanged_records"), field="unchanged_records")
    output = _parse_split_counts(matrix_root.get("output_records"), field="output_records")
    for split in SPLITS:
        matrix_total = sum(
            matrix[split][category][bucket][order]
            for category in CATEGORIES
            for bucket in BUCKETS
            for order in ORDER_MODES
        )
        if (
            matrix_total != rewritten[split]
            or rewritten[split] + unchanged[split] != output[split]
            or output[split] != source_files[split].records
        ):
            raise TargetedRepairV5BridgeError("rewrite matrix does not close")
    contract = _mapping(root.get("rewrite_contract"), field="rewrite_contract")
    if contract != {
        "preserve_candidate_surface": True,
        "preserve_split": True,
        "preserve_hard_negative_category": True,
        "preserve_non_entity_value_group": True,
        "checksum_semantics_required": True,
        "candidate_occurrences_per_document": 1,
        "extra_digit_or_ascii_context_allowed": False,
        "train_validation_template_overlap_allowed": False,
        "train_validation_cue_phrase_overlap_allowed": False,
        "D0_masked_skeleton_exact_collision_allowed": False,
        "D0_masked_skeleton_char5gram_max_similarity_exclusive": 0.75,
    }:
        raise TargetedRepairV5BridgeError("rewrite contract changed")
    boundary = _mapping(root.get("lineage_and_evaluation_boundary"), field="lineage boundary")
    if boundary != {
        "D0_validation_informed": True,
        "D0_raw_text_copied": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "PII_Bench_read_or_executed": False,
        "external_benchmark_read": False,
        "confirmatory_evaluation_eligible": False,
        "supports_public_best_model_claim": False,
    }:
        raise TargetedRepairV5BridgeError("lineage boundary changed")
    training = _mapping(root.get("training_contract"), field="training contract")
    if training != {
        "dataset_cardinality_changed_from_v4": False,
        "optimizer_steps_per_epoch_changed_from_v4": False,
        "optimizer_steps_per_epoch": 305,
        "loss_weighting_changed": False,
    }:
        raise TargetedRepairV5BridgeError("training contract changed")
    return BridgeRecipe(
        path=path,
        sha256=sha256_file(path),
        seed=_positive_int(root.get("seed"), field="seed"),
        source_root=Path(source_root_raw),
        source_dataset_id=str(source.get("dataset_id")),
        source_dataset_version=str(source.get("dataset_version")),
        source_manifest_file_sha256=_sha256(
            source.get("manifest_file_sha256"), field="source manifest file hash"
        ),
        source_manifest_sha256=_sha256(
            source.get("manifest_sha256"), field="source manifest logical hash"
        ),
        source_files=MappingProxyType(source_files),
        rewrite_matrix=MappingProxyType(matrix),
        rewritten_counts=MappingProxyType(rewritten),
        unchanged_counts=MappingProxyType(unchanged),
        output_counts=MappingProxyType(output),
    )


def _signed_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV5BridgeError("source manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise TargetedRepairV5BridgeError("source manifest must be an object")
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or claimed != canonical_hash(unsigned):
        raise TargetedRepairV5BridgeError("source manifest logical hash failed")
    return value


def read_frozen_v4(recipe: BridgeRecipe) -> tuple[DocumentRecord, ...]:
    root = recipe.source_root
    if root.is_symlink() or not root.is_dir():
        raise TargetedRepairV5BridgeError("source v4 directory must be regular")
    expected_names = {"train.jsonl", "validation.jsonl", "dataset_manifest.json"}
    children = list(root.iterdir())
    if {child.name for child in children} != expected_names or any(
        child.is_symlink() or not child.is_file() for child in children
    ):
        raise TargetedRepairV5BridgeError("source v4 file inventory changed")
    manifest_path = root / "dataset_manifest.json"
    if sha256_file(manifest_path) != recipe.source_manifest_file_sha256:
        raise TargetedRepairV5BridgeError("source v4 manifest file hash changed")
    manifest = _signed_manifest(manifest_path)
    if (
        manifest.get("manifest_sha256") != recipe.source_manifest_sha256
        or manifest.get("dataset_id") != recipe.source_dataset_id
        or manifest.get("dataset_version") != recipe.source_dataset_version
    ):
        raise TargetedRepairV5BridgeError("source v4 manifest identity changed")
    manifest_files = {
        item.get("split"): item for item in manifest.get("files", []) if isinstance(item, Mapping)
    }
    records: list[DocumentRecord] = []
    for split in SPLITS:
        contract = recipe.source_files[split]
        path = root / contract.name
        if sha256_file(path) != contract.sha256:
            raise TargetedRepairV5BridgeError("source v4 split hash changed")
        bound = manifest_files.get(split)
        if not isinstance(bound, Mapping) or (
            bound.get("name") != contract.name
            or bound.get("records") != contract.records
            or bound.get("sha256") != contract.sha256
        ):
            raise TargetedRepairV5BridgeError("source v4 file binding changed")
        selected = list(iter_jsonl(path))
        if len(selected) != contract.records or any(record.split != split for record in selected):
            raise TargetedRepairV5BridgeError("source v4 split count or assignment changed")
        records.extend(selected)
    return tuple(records)


def _candidate(record: DocumentRecord, category: str) -> str:
    if record.entities:
        raise TargetedRepairV5BridgeError("rewrite source unexpectedly has entities")
    span = record.metadata.get("candidate_span")
    if (
        not isinstance(span, list)
        or len(span) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
    ):
        raise TargetedRepairV5BridgeError("rewrite source candidate span is invalid")
    start, end = span
    if not 0 <= start < end <= len(record.text):
        raise TargetedRepairV5BridgeError("rewrite source candidate span is out of bounds")
    candidate = record.text[start:end]
    if record.text.count(candidate) != 1:
        raise TargetedRepairV5BridgeError("rewrite source candidate is not unique")
    if record.metadata.get("non_entity_value_groups") != [stable_text_hash(candidate)]:
        raise TargetedRepairV5BridgeError("rewrite source value group changed")
    validation = (
        validate_luhn(candidate)
        if category == "invalid_card_single_surface"
        else validate_cn_resident_id(candidate)
    )
    if validation.valid:
        raise TargetedRepairV5BridgeError("rewrite source candidate unexpectedly validates")
    return candidate


def _validate_template(
    spec: TemplateSpec, *, split: str, category: str, order: str, bucket: str
) -> None:
    if spec.body.count(PLACEHOLDER) != 1:
        raise TargetedRepairV5BridgeError("template placeholder count changed")
    context = spec.body.replace(PLACEHOLDER, "")
    if _CONTEXT_ASCII_OR_DIGIT_RE.search(context):
        raise TargetedRepairV5BridgeError("template context contains ASCII or a digit")
    for marker in (spec.domain_marker, spec.action_marker, spec.cue_phrase):
        if not marker or (marker != spec.cue_phrase and marker not in spec.body):
            raise TargetedRepairV5BridgeError("template semantic marker changed")
    candidate_length = 16 if category == "invalid_card_single_surface" else 18
    rendered_length = len(spec.body.replace(PLACEHOLDER, "数" * candidate_length))
    low, high = BUCKET_RANGES[bucket]
    if not low <= rendered_length <= high:
        raise TargetedRepairV5BridgeError("template misses its length bucket")
    placeholder = spec.body.index(PLACEHOLDER)
    domain = spec.body.index(spec.domain_marker)
    action = spec.body.index(spec.action_marker)
    before = spec.body.index(spec.verdict_before) if spec.verdict_before else None
    after = spec.body.rindex(spec.verdict_after) if spec.verdict_after else None
    if order == "A_all_before":
        valid = (
            domain < placeholder
            and action < placeholder
            and before is not None
            and before < placeholder
        )
    elif order == "B_surface_first":
        valid = placeholder < domain < action and after is not None and action < after
    elif order == "C_split_verdict":
        valid = (
            action < placeholder
            and before is not None
            and action <= before < placeholder < domain
            and after is not None
            and domain < after
        )
    else:
        valid = domain < placeholder < action and after is not None and action < after
    if not valid:
        raise TargetedRepairV5BridgeError("template cue-order contract changed")
    expected_checksum_marker = "校验位" if split == "train" else "核验码"
    if expected_checksum_marker not in spec.body:
        raise TargetedRepairV5BridgeError("template lacks split-specific checksum semantics")


def template_contract() -> Mapping[str, Any]:
    template_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    cue_phrases: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split in SPLITS:
        for category in CATEGORIES:
            for order in ORDER_MODES:
                for bucket in BUCKETS:
                    spec = _TEMPLATES[split][category][order][bucket]
                    _validate_template(
                        spec,
                        split=split,
                        category=category,
                        order=order,
                        bucket=bucket,
                    )
                    template_hashes[split].add(hashlib.sha256(spec.body.encode()).hexdigest())
                    cue_phrases[split].add(spec.cue_phrase)
    if template_hashes["train"] & template_hashes["validation"]:
        raise TargetedRepairV5BridgeError("train and validation templates overlap")
    if cue_phrases["train"] & cue_phrases["validation"]:
        raise TargetedRepairV5BridgeError("train and validation cue phrases overlap")
    return MappingProxyType(
        {
            "authored_templates": {split: len(template_hashes[split]) for split in SPLITS},
            "cue_phrases": {split: len(cue_phrases[split]) for split in SPLITS},
            "train_validation_template_hash_collisions": 0,
            "train_validation_cue_phrase_collisions": 0,
            "all_templates_checksum_semantic": True,
            "all_templates_match_declared_order": True,
            "all_templates_match_declared_length_bucket": True,
            "all_template_contexts_exclude_extra_ascii_and_digits": True,
        }
    )


def _rewrite_record(
    *,
    source: DocumentRecord,
    candidate: str,
    category: str,
    bucket: str,
    order: str,
    source_row_index: int,
    recipe: BridgeRecipe,
) -> DocumentRecord:
    split = str(source.split)
    spec = _TEMPLATES[split][category][order][bucket]
    text = spec.body.replace(PLACEHOLDER, candidate)
    start = text.index(candidate)
    end = start + len(candidate)
    if text.count(candidate) != 1:
        raise TargetedRepairV5BridgeError("rewritten candidate occurrence count changed")
    context = text[:start] + text[end:]
    if _CONTEXT_ASCII_OR_DIGIT_RE.search(context):
        raise TargetedRepairV5BridgeError("rewritten context contains extra ASCII or digits")
    if tuple(match.span(1) for match in _STRUCTURED_RE.finditer(text)) != ((start, end),):
        raise TargetedRepairV5BridgeError("rewritten structured-surface count changed")
    low, high = BUCKET_RANGES[bucket]
    if not low <= len(text) <= high:
        raise TargetedRepairV5BridgeError("rewritten text misses its length bucket")
    source_doc_hash = hashlib.sha256(source.doc_id.encode()).hexdigest()
    template_id = f"repair-v5a-{split}-{category}-{order}-{bucket}"
    metadata = {
        "targeted_repair_v5a_checksum_context_rewrite": True,
        "record_role": "pii_free_checksum_invalid_negative",
        "hard_negative_category": category,
        "length_bucket": bucket,
        "checksum_cue_order": order,
        "checksum_semantics_explicit": True,
        "checksum_action_marker": spec.action_marker,
        "checksum_domain_marker": spec.domain_marker,
        "checksum_cue_phrase": spec.cue_phrase,
        "candidate_span": [start, end],
        "candidate_occurrence_count": 1,
        "structured_surface_count": 1,
        "non_entity_value_groups": [stable_text_hash(candidate)],
        "authored_template_id": template_id,
        "template_body_sha256": hashlib.sha256(spec.body.encode()).hexdigest(),
        "source_v4_doc_id_sha256": source_doc_hash,
        "source_v4_row_index": source_row_index,
        "source_v4_candidate_surface_preserved": True,
        "source_v4_split_preserved": True,
        "source_v4_hard_negative_category_preserved": True,
        "role_decoy_semantics_removed": True,
        "D0_validation_informed": True,
        "D0_raw_text_copied": False,
        "test-hidden-PIIBench": False,
        "test_data_read": False,
        "hidden_data_read": False,
        "PII_Bench_read_or_executed": False,
        "external_benchmark_read": False,
        "confirmatory_evaluation_eligible": False,
        "supports_public_best_model_claim": False,
        "auxiliary_ascii_or_digit_context_present": False,
    }
    provenance = Provenance(
        source_id=DATASET_ID,
        source_kind="deterministic_synthetic_development_checksum_context_rewrite",
        source_revision=DATASET_VERSION,
        license="Apache-2.0",
        synthetic=True,
        generator_name=GENERATOR_NAME,
        generator_version=GENERATOR_VERSION,
        generator_seed=recipe.seed,
        template_id=template_id,
        metadata={
            "frozen_v4_candidate_surface_parent": True,
            "D0_validation_informed": True,
            "D0_raw_text_copied": False,
            "optimization_only": True,
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.HARD_NEGATIVE,
        quality_gate=True,
        validators_passed=True,
        review_status="checksum_invalid_context_rewrite_validated",
        confidence=1.0,
        metadata={"optimization_only_not_confirmatory_evidence": True},
    )
    rewritten_identity = hashlib.sha256((source.doc_id + chr(0) + order).encode()).hexdigest()[:24]
    return DocumentRecord.create(
        doc_id=(f"v5a-{split}-{rewritten_identity}"),
        text=text,
        entities=(),
        language="zh-CN",
        domain="finance" if category == "invalid_card_single_surface" else "identity",
        scene="checksum_validation_followup",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=True,
        conversation_group=source.conversation_group,
        template_group=template_id,
        entity_value_groups=(),
        created_at_bucket=source.created_at_bucket,
        generator_version=GENERATOR_VERSION,
        split=split,
        metadata=metadata,
    )


def rewrite_records(
    source_records: Sequence[DocumentRecord], recipe: BridgeRecipe
) -> tuple[DocumentRecord, ...]:
    cell_indices: Counter[tuple[str, str, str]] = Counter()
    output: list[DocumentRecord] = []
    for row_index, source in enumerate(source_records):
        category_raw = source.metadata.get("hard_negative_category")
        if category_raw not in CATEGORIES:
            output.append(source)
            continue
        category = str(category_raw)
        split = str(source.split)
        bucket = str(source.metadata.get("length_bucket"))
        if split not in SPLITS or bucket not in BUCKETS:
            raise TargetedRepairV5BridgeError("rewrite source split or length bucket changed")
        candidate = _candidate(source, category)
        cell = (split, category, bucket)
        local_index = cell_indices[cell]
        cell_indices[cell] += 1
        order = ORDER_MODES[local_index % len(ORDER_MODES)]
        output.append(
            _rewrite_record(
                source=source,
                candidate=candidate,
                category=category,
                bucket=bucket,
                order=order,
                source_row_index=row_index,
                recipe=recipe,
            )
        )
    return tuple(output)


def _record_sets(records: Sequence[DocumentRecord]) -> dict[str, set[str]]:
    result = {
        "document_ids": set(),
        "text_hashes": set(),
        "template_groups": set(),
        "value_groups": set(),
        "relation_groups": set(),
    }
    for record in records:
        result["document_ids"].add(record.doc_id)
        result["text_hashes"].add(hashlib.sha256(record.text.encode()).hexdigest())
        if record.template_group:
            result["template_groups"].add(record.template_group)
        result["value_groups"].update(record.entity_value_groups)
        non_entity = record.metadata.get("non_entity_value_groups")
        if isinstance(non_entity, list):
            result["value_groups"].update(str(group) for group in non_entity)
        if record.conversation_group:
            result["relation_groups"].add(record.conversation_group)
    return result


def validate_rewrite(
    source_records: Sequence[DocumentRecord],
    output_records: Sequence[DocumentRecord],
    recipe: BridgeRecipe,
) -> Mapping[str, Any]:
    if len(source_records) != len(output_records):
        raise TargetedRepairV5BridgeError("rewrite changed dataset cardinality")
    template_audit = template_contract()
    matrix: Counter[tuple[str, str, str, str]] = Counter()
    rewritten_counts: Counter[str] = Counter()
    unchanged_counts: Counter[str] = Counter()
    validator_counts: Counter[str] = Counter()
    unchanged_digest = hashlib.sha256()
    candidate_pairs: set[tuple[str, str]] = set()
    for source, output in zip(source_records, output_records, strict=True):
        output.validate()
        category_raw = source.metadata.get("hard_negative_category")
        eligible = category_raw in CATEGORIES
        if not eligible:
            if dumps_document(source) != dumps_document(output):
                raise TargetedRepairV5BridgeError("non-target v4 row changed")
            unchanged_counts[str(source.split)] += 1
            unchanged_digest.update(dumps_document(output).encode())
            unchanged_digest.update(b"\n")
            continue
        category = str(category_raw)
        split = str(source.split)
        bucket = str(source.metadata.get("length_bucket"))
        order = str(output.metadata.get("checksum_cue_order"))
        if (
            output.split != source.split
            or output.metadata.get("hard_negative_category") != category
        ):
            raise TargetedRepairV5BridgeError("rewritten split or category changed")
        source_candidate = _candidate(source, category)
        span = output.metadata.get("candidate_span")
        if not isinstance(span, list) or len(span) != 2:
            raise TargetedRepairV5BridgeError("rewritten candidate span changed")
        start, end = span
        candidate = output.text[start:end]
        if candidate != source_candidate:
            raise TargetedRepairV5BridgeError("rewritten candidate surface changed")
        if output.metadata.get("non_entity_value_groups") != [stable_text_hash(candidate)]:
            raise TargetedRepairV5BridgeError("rewritten non-entity value group changed")
        validation = (
            validate_luhn(candidate)
            if category == "invalid_card_single_surface"
            else validate_cn_resident_id(candidate)
        )
        if validation.valid:
            raise TargetedRepairV5BridgeError("rewritten candidate unexpectedly validates")
        validator_counts[category] += 1
        matrix[(split, category, bucket, order)] += 1
        rewritten_counts[split] += 1
        candidate_pairs.add((split, stable_text_hash(candidate)))
        required = {
            "checksum_semantics_explicit": True,
            "source_v4_candidate_surface_preserved": True,
            "source_v4_split_preserved": True,
            "source_v4_hard_negative_category_preserved": True,
            "role_decoy_semantics_removed": True,
            "D0_validation_informed": True,
            "D0_raw_text_copied": False,
            "test-hidden-PIIBench": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "PII_Bench_read_or_executed": False,
            "external_benchmark_read": False,
            "confirmatory_evaluation_eligible": False,
        }
        if any(output.metadata.get(key) is not value for key, value in required.items()):
            raise TargetedRepairV5BridgeError("rewritten development-boundary flag changed")
    matrix_result = {
        split: {
            category: {
                bucket: {order: matrix[(split, category, bucket, order)] for order in ORDER_MODES}
                for bucket in BUCKETS
            }
            for category in CATEGORIES
        }
        for split in SPLITS
    }
    if matrix_result != recipe.rewrite_matrix:
        raise TargetedRepairV5BridgeError("observed rewrite matrix changed")
    for split in SPLITS:
        if (
            rewritten_counts[split] != recipe.rewritten_counts[split]
            or unchanged_counts[split] != recipe.unchanged_counts[split]
            or rewritten_counts[split] + unchanged_counts[split] != recipe.output_counts[split]
        ):
            raise TargetedRepairV5BridgeError("rewrite counts changed")
    by_split = {
        split: [record for record in output_records if record.split == split] for split in SPLITS
    }
    sets = {split: _record_sets(by_split[split]) for split in SPLITS}
    isolation: dict[str, Any] = {}
    for field in sets["train"]:
        collisions = sets["train"][field] & sets["validation"][field]
        if collisions:
            raise TargetedRepairV5BridgeError(f"cross-split {field} isolation failed")
        isolation[field] = {
            "train": len(sets["train"][field]),
            "validation": len(sets["validation"][field]),
            "cross_split_collisions": 0,
        }
    return MappingProxyType(
        {
            "counts": {
                "source_v4_by_split": {
                    split: recipe.source_files[split].records for split in SPLITS
                },
                "rewritten_by_split": {split: rewritten_counts[split] for split in SPLITS},
                "unchanged_by_split": {split: unchanged_counts[split] for split in SPLITS},
                "output_by_split": {
                    split: rewritten_counts[split] + unchanged_counts[split] for split in SPLITS
                },
            },
            "rewrite_matrix": matrix_result,
            "template_audit": dict(template_audit),
            "validator_audit": {
                "invalid_card_single_surface": validator_counts["invalid_card_single_surface"],
                "invalid_resident_id_single_surface": validator_counts[
                    "invalid_resident_id_single_surface"
                ],
                "all_rewritten_candidates_checksum_invalid": True,
            },
            "rewrite_contract_audit": {
                "rewritten_candidate_value_groups": len(candidate_pairs),
                "candidate_surfaces_preserved": True,
                "split_assignments_preserved": True,
                "hard_negative_categories_preserved": True,
                "non_entity_value_groups_preserved": True,
                "non_target_records_byte_canonical_unchanged": True,
                "non_target_records_aggregate_sha256": unchanged_digest.hexdigest(),
                "all_four_order_modes_covered_per_family_and_length": True,
                "all_rewritten_contexts_checksum_semantic": True,
                "role_decoy_semantics_removed_from_rewritten_records": True,
                "extra_ascii_or_digit_context_count": 0,
                "dataset_cardinality_changed_from_v4": False,
                "optimizer_steps_per_epoch": 305,
                "loss_weighting_changed": False,
            },
            "isolation": isolation,
        }
    )


def build_bridge_dataset(recipe_path: Path = DEFAULT_RECIPE_PATH) -> BridgeBuild:
    recipe = load_recipe(recipe_path)
    source_records = read_frozen_v4(recipe)
    output_records = rewrite_records(source_records, recipe)
    validation = validate_rewrite(source_records, output_records, recipe)
    rewritten = tuple(
        record
        for record in output_records
        if record.metadata.get("targeted_repair_v5a_checksum_context_rewrite") is True
    )
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "artifact_type": "synthetic_targeted_repair_v5a_checksum_context_rewrite",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "release_status": "development_optimization_only_not_claim_evidence",
        "data_pool": "public_release_pool",
        "public_weight_training_allowed": True,
        "recipe": {
            "id": DATASET_ID,
            "version": DATASET_VERSION,
            "sha256": recipe.sha256,
            "seed": recipe.seed,
        },
        "source_v4": {
            "dataset_id": recipe.source_dataset_id,
            "dataset_version": recipe.source_dataset_version,
            "manifest_file_sha256": recipe.source_manifest_file_sha256,
            "manifest_sha256": recipe.source_manifest_sha256,
            "files": {
                split: {
                    "name": recipe.source_files[split].name,
                    "records": recipe.source_files[split].records,
                    "sha256": recipe.source_files[split].sha256,
                }
                for split in SPLITS
            },
        },
        "generation": {
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "rewrite_in_place": True,
            "row_order_preserved": True,
            "output_splits": list(SPLITS),
        },
        **dict(validation),
        "lineage_and_evaluation_boundary": {
            "D0_validation_informed": True,
            "D0_raw_text_copied": False,
            "test-hidden-PIIBench": False,
            "test_data_read": False,
            "hidden_data_read": False,
            "PII_Bench_read_or_executed": False,
            "external_benchmark_read": False,
            "confirmatory_evaluation_eligible": False,
            "supports_public_best_model_claim": False,
        },
        "publication_boundary": {
            "optimization_only": True,
            "supports_first_chinese_PII_model_claim": False,
            "supports_best_open_model_claim": False,
            "real_or_consented_context_evaluation_required": True,
            "new_unseen_hidden_confirmatory_evaluation_required": True,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return BridgeBuild(
        source_records=source_records,
        records=output_records,
        rewritten_records=rewritten,
        manifest=MappingProxyType(manifest),
        recipe=recipe,
    )


def materialize(build: BridgeBuild, output_dir: Path) -> Mapping[str, Any]:
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise TargetedRepairV5BridgeError("output must be a new absolute directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        files: list[dict[str, Any]] = []
        for split in SPLITS:
            path = stage / f"{split}.jsonl"
            selected = [record for record in build.records if record.split == split]
            if write_jsonl(selected, path) != build.recipe.output_counts[split]:
                raise TargetedRepairV5BridgeError("materialized split count changed")
            if len(list(iter_jsonl(path))) != len(selected):
                raise TargetedRepairV5BridgeError("materialized split readback failed")
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


def validate_materialized(
    dataset_dir: Path, recipe_path: Path = DEFAULT_RECIPE_PATH
) -> Mapping[str, Any]:
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise TargetedRepairV5BridgeError("dataset directory must be regular")
    names = {"train.jsonl", "validation.jsonl", "dataset_manifest.json"}
    children = list(dataset_dir.iterdir())
    if {child.name for child in children} != names or any(
        child.is_symlink() or not child.is_file() for child in children
    ):
        raise TargetedRepairV5BridgeError("materialized dataset inventory changed")
    expected = build_bridge_dataset(recipe_path)
    records: list[DocumentRecord] = []
    files: list[dict[str, Any]] = []
    for split in SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        selected = list(iter_jsonl(path))
        if len(selected) != expected.recipe.output_counts[split] or any(
            record.split != split for record in selected
        ):
            raise TargetedRepairV5BridgeError("materialized split count or assignment changed")
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
    validation = validate_rewrite(expected.source_records, records, expected.recipe)
    for section in (
        "counts",
        "rewrite_matrix",
        "template_audit",
        "validator_audit",
        "rewrite_contract_audit",
        "isolation",
    ):
        if validation[section] != expected.manifest[section]:
            raise TargetedRepairV5BridgeError("materialized record audit changed")
    manifest_path = dataset_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV5BridgeError("materialized manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise TargetedRepairV5BridgeError("materialized manifest must be an object")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != canonical_hash(unsigned):
        raise TargetedRepairV5BridgeError("materialized manifest logical hash failed")
    expected_manifest = dict(expected.manifest)
    expected_manifest["files"] = files
    expected_manifest.pop("manifest_sha256", None)
    expected_manifest["manifest_sha256"] = canonical_hash(expected_manifest)
    if manifest != expected_manifest:
        raise TargetedRepairV5BridgeError("materialized manifest differs from reconstruction")
    with tempfile.TemporaryDirectory(prefix="repair-v5a-replay-", dir=dataset_dir.parent) as root:
        replay_dir = Path(root) / "replay"
        materialize(expected, replay_dir)
        replay_hashes: dict[str, str] = {}
        for name in sorted(names):
            original = dataset_dir / name
            replay = replay_dir / name
            if original.read_bytes() != replay.read_bytes():
                raise TargetedRepairV5BridgeError("independent replay is not byte-identical")
            replay_hashes[name] = sha256_file(original)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "targeted_repair_v5a_independent_validation",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": claimed,
        "recipe_file_sha256": expected.recipe.sha256,
        "files": files,
        "counts": manifest["counts"],
        "rewrite_matrix": manifest["rewrite_matrix"],
        "template_audit": manifest["template_audit"],
        "validator_audit": manifest["validator_audit"],
        "rewrite_contract_audit": manifest["rewrite_contract_audit"],
        "isolation": manifest["isolation"],
        "independent_rematerialization": {
            "executed": True,
            "byte_identical": True,
            "files": replay_hashes,
        },
        "development_boundary": manifest["lineage_and_evaluation_boundary"],
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_values": False,
            "contains_absolute_paths": False,
        },
    }
    report["report_sha256"] = canonical_hash(report)
    return MappingProxyType(report)


__all__ = [
    "BUCKETS",
    "BUCKET_RANGES",
    "BridgeBuild",
    "BridgeRecipe",
    "CATEGORIES",
    "DATASET_ID",
    "DATASET_VERSION",
    "DEFAULT_RECIPE_PATH",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "MATERIALIZER_VERSION",
    "ORDER_MODES",
    "SPLITS",
    "TargetedRepairV5BridgeError",
    "build_bridge_dataset",
    "canonical_hash",
    "load_recipe",
    "materialize",
    "read_frozen_v4",
    "rewrite_records",
    "sha256_file",
    "template_contract",
    "validate_materialized",
    "validate_rewrite",
]
