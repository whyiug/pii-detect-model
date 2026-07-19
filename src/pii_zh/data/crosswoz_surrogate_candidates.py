"""Fail-closed CrossWOZ natural-context surrogate annotation candidates.

The upstream dialogue context is human-authored, while every identifiable
slot value retained in a selected turn is replaced with a deterministic,
non-contactable surrogate.  Upstream task IDs, source values, annotator IDs,
and source text never enter the aggregate report.

Passing this pipeline does not establish human PII annotation, privacy review,
license suitability, or permission to train public weights.  Every record is
routed to quarantine and the upstream test split is permanently marked as
never-train.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from pii_zh.data.converters._common import ConversionError
from pii_zh.data.converters.crosswoz import (
    CROSSWOZ_SOURCE_ID,
    convert_crosswoz_record,
)
from pii_zh.data.schema import (
    DataPool,
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    dumps_document,
    stable_text_hash,
)
from pii_zh.data.surrogates import (
    OriginalSpan,
    SurrogateEdit,
    SurrogateError,
    rebuild_surrogate_text,
)
from pii_zh.data.validators import validate_cn_resident_id, validate_luhn, validate_phone

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without the core/dev extra
    yaml = None  # type: ignore[assignment]


SOURCE_REVISION = "df82c9fdff91b9b130f2d6b89110d3870ba6260e"
PIPELINE_ID = "crosswoz_human_context_surrogate_candidate_v1"
DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs/data/crosswoz_surrogate_candidate_v1.yaml"
)
SPLITS = ("train", "validation", "test")
CANDIDATE_KINDS = (
    "vehicle_plate_semantic_candidate",
    "organization_place_person_name_hard_negative",
)

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_GIT_REF_PATTERN = re.compile(r"refs/[A-Za-z0-9._/-]+\Z")
_MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
_NGRAM_SIZE = 5

_PHONE_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(?:(?:\+?86[- ()]?)?1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})"
    r"(?![0-9A-Za-z])"
)
_CN_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
_CARD_PATTERN = re.compile(r"(?<![0-9A-Za-z.])(?:\d[ -]?){11,18}\d(?![0-9A-Za-z.])")
_EMAIL_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@(?:[A-Z0-9-]+\.)+[A-Z]{2,63}"
)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9._~+/=-]{12,})"
    ),
)

_SLOT_CATEGORY: Mapping[tuple[str, str], str] = {
    **{
        (domain, slot): "organization_or_place_name"
        for domain in ("景点", "餐馆", "酒店")
        for slot in ("名称", "周边景点", "周边餐馆", "周边酒店")
    },
    **{
        (domain, "地址"): "organization_address"
        for domain in ("景点", "餐馆", "酒店")
    },
    **{
        (domain, "电话"): "public_contact_phone"
        for domain in ("景点", "餐馆", "酒店")
    },
    **{
        (domain, slot): "location_or_route_endpoint"
        for domain in ("地铁", "出租")
        for slot in ("出发地", "目的地")
    },
    ("地铁", "出发地附近地铁站"): "location_or_route_endpoint",
    ("地铁", "目的地附近地铁站"): "location_or_route_endpoint",
    ("出租", "车型"): "vehicle_model",
    ("出租", "车牌"): "vehicle_license_plate",
}

_CATEGORY_PRIORITY = {
    "pattern_secret": 0,
    "pattern_cn_resident_id": 1,
    "pattern_email": 2,
    "pattern_phone": 3,
    "pattern_bank_card": 4,
    "public_contact_phone": 5,
    "vehicle_license_plate": 6,
    "organization_address": 7,
    "organization_or_place_name": 8,
    "location_or_route_endpoint": 9,
    "vehicle_model": 10,
}
_CJK_TOKEN_ALPHABET = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄宇宙洪荒"
_ASCII_UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_ASCII_LOWER = _ASCII_UPPER.casefold()


class CrossWozSurrogateCandidateError(ValueError):
    """Raised without reflecting source values, source IDs, or absolute paths."""


@dataclass(frozen=True, slots=True)
class FileExpectation:
    relative_path: str
    sha256: str
    member: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateRecipe:
    pipeline_id: str
    pipeline_version: str
    source_id: str
    source_revision: str
    source_license: str
    archives: Mapping[str, FileExpectation]
    evidence: Mapping[str, FileExpectation]
    existing_quarantine: Mapping[str, FileExpectation]
    quotas: Mapping[str, Mapping[str, int]]
    recipe_namespace: str
    minimum_source_value_length: int
    near_duplicate_threshold: float
    training_roles: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SlotEvidence:
    value: str = field(repr=False)
    category: str
    domain: str
    slot: str
    origin: str


@dataclass(frozen=True, slots=True)
class TurnDescriptor:
    split: str
    dialogue_group: str
    dialogue_id: str = field(repr=False)
    turn_index: int
    role: str
    content: str = field(repr=False)
    candidate_kind: str
    evidence_by_value: Mapping[str, tuple[SlotEvidence, ...]] = field(repr=False)
    direct_plate_values: frozenset[str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _SourceSpan:
    start: int
    end: int
    value: str = field(repr=False)
    category: str
    evidence: tuple[SlotEvidence, ...]
    direct_plate: bool


@dataclass(frozen=True, slots=True)
class _TransformedTurn:
    record: DocumentRecord
    replacement_hashes: frozenset[str]
    scanner_counts: Mapping[str, int]
    permitted_surrogate_scan_count: int
    replacement_count: int


@dataclass(frozen=True, slots=True)
class CandidateBuild:
    recipe: CandidateRecipe
    records: tuple[DocumentRecord, ...]
    report: Mapping[str, Any]


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossWozSurrogateCandidateError(f"{context} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise CrossWozSurrogateCandidateError(f"{context} fields failed validation")


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CrossWozSurrogateCandidateError(f"{context} must be a canonical string")
    return value


def _hash(value: object, *, context: str) -> str:
    result = _string(value, context=context)
    if _HASH_PATTERN.fullmatch(result) is None:
        raise CrossWozSurrogateCandidateError(f"{context} must be a lowercase SHA-256")
    return result


def _relative_path(value: object, *, context: str) -> str:
    result = _string(value, context=context)
    pure = PurePosixPath(result)
    if pure.is_absolute() or ".." in pure.parts or "\\" in result:
        raise CrossWozSurrogateCandidateError(f"{context} must be a safe relative path")
    return result


def _file_expectation(value: object, *, archive: bool, context: str) -> FileExpectation:
    row = _mapping(value, context=context)
    expected = {"relative_path", "sha256", "member"} if archive else {
        "relative_path",
        "sha256",
    }
    _exact_keys(row, expected, context=context)
    member = _relative_path(row["member"], context=f"{context}.member") if archive else None
    return FileExpectation(
        relative_path=_relative_path(row["relative_path"], context=f"{context}.relative_path"),
        sha256=_hash(row["sha256"], context=f"{context}.sha256"),
        member=member,
    )


def load_recipe(path: Path = DEFAULT_RECIPE_PATH) -> CandidateRecipe:
    """Load the fixed fail-closed recipe without accepting weakened gates."""

    if yaml is None:
        raise CrossWozSurrogateCandidateError("PyYAML is required to load the candidate recipe")
    if path.is_symlink() or not path.is_file():
        raise CrossWozSurrogateCandidateError("candidate recipe must be a regular file")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise CrossWozSurrogateCandidateError("candidate recipe failed parsing") from None
    root = _mapping(raw, context="recipe")
    _exact_keys(
        root,
        {
            "schema_version",
            "pipeline_id",
            "pipeline_version",
            "source",
            "existing_quarantine",
            "selection",
            "surrogates",
            "gates",
            "admission",
            "lineage_exclusions",
        },
        context="recipe",
    )
    if root["schema_version"] != 1 or root["pipeline_id"] != PIPELINE_ID:
        raise CrossWozSurrogateCandidateError("candidate recipe identity failed validation")

    source = _mapping(root["source"], context="source")
    _exact_keys(
        source,
        {"source_id", "revision", "declared_license", "archives", "evidence"},
        context="source",
    )
    if source["source_id"] != CROSSWOZ_SOURCE_ID or source["revision"] != SOURCE_REVISION:
        raise CrossWozSurrogateCandidateError("fixed upstream identity failed validation")
    if _REVISION_PATTERN.fullmatch(str(source["revision"])) is None:
        raise CrossWozSurrogateCandidateError("fixed upstream revision failed validation")
    archives_raw = _mapping(source["archives"], context="source.archives")
    _exact_keys(archives_raw, set(SPLITS), context="source.archives")
    archives = {
        split: _file_expectation(
            archives_raw[split], archive=True, context=f"source.archives.{split}"
        )
        for split in SPLITS
    }
    evidence_raw = _mapping(source["evidence"], context="source.evidence")
    _exact_keys(evidence_raw, {"license", "root_readme", "data_readme"}, context="source.evidence")
    evidence = {
        name: _file_expectation(value, archive=False, context=f"source.evidence.{name}")
        for name, value in evidence_raw.items()
    }

    existing_raw = _mapping(root["existing_quarantine"], context="existing_quarantine")
    _exact_keys(existing_raw, set(SPLITS), context="existing_quarantine")
    existing: dict[str, FileExpectation] = {}
    for split in SPLITS:
        item = _mapping(existing_raw[split], context=f"existing_quarantine.{split}")
        _exact_keys(item, {"filename", "sha256"}, context=f"existing_quarantine.{split}")
        existing[split] = FileExpectation(
            relative_path=_relative_path(
                item["filename"], context=f"existing_quarantine.{split}.filename"
            ),
            sha256=_hash(item["sha256"], context=f"existing_quarantine.{split}.sha256"),
        )

    selection = _mapping(root["selection"], context="selection")
    _exact_keys(
        selection,
        set(SPLITS) | {"rank_method", "maximum_records_per_dialogue"},
        context="selection",
    )
    if (
        selection["rank_method"] != "sha256_split_group_turn_category"
        or selection["maximum_records_per_dialogue"] != 1
    ):
        raise CrossWozSurrogateCandidateError("selection isolation contract failed validation")
    quotas: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        row = _mapping(selection[split], context=f"selection.{split}")
        _exact_keys(row, set(CANDIDATE_KINDS), context=f"selection.{split}")
        quotas[split] = {}
        for kind in CANDIDATE_KINDS:
            count = row[kind]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise CrossWozSurrogateCandidateError("selection quota failed validation")
            quotas[split][kind] = count

    surrogates = _mapping(root["surrogates"], context="surrogates")
    _exact_keys(
        surrogates,
        {
            "recipe_namespace",
            "deterministic",
            "non_contactable",
            "format_preserving",
            "source_value_derived_public_hashes",
        },
        context="surrogates",
    )
    if (
        surrogates["deterministic"] is not True
        or surrogates["non_contactable"] is not True
        or surrogates["format_preserving"] is not True
        or surrogates["source_value_derived_public_hashes"] is not False
    ):
        raise CrossWozSurrogateCandidateError("surrogate safety contract cannot be weakened")

    gates = _mapping(root["gates"], context="gates")
    _exact_keys(
        gates,
        {
            "reject_ambiguous_identifiable_values_shorter_than",
            "exact_duplicate_pair_count",
            "char_5gram_near_duplicate_threshold",
            "char_5gram_near_duplicate_pair_count",
            "cross_split_group_collision_count",
            "cross_split_surrogate_value_collision_count",
            "source_value_residual_count",
            "uncovered_phone_email_id_card_secret_scan_count",
            "offset_alignment_error_count",
            "output_directory_mode",
            "output_file_mode",
        },
        context="gates",
    )
    if any(
        gates[name] != 0
        for name in (
            "exact_duplicate_pair_count",
            "char_5gram_near_duplicate_pair_count",
            "cross_split_group_collision_count",
            "cross_split_surrogate_value_collision_count",
            "source_value_residual_count",
            "uncovered_phone_email_id_card_secret_scan_count",
            "offset_alignment_error_count",
        )
    ) or gates["output_directory_mode"] != "0700" or gates["output_file_mode"] != "0600":
        raise CrossWozSurrogateCandidateError("validation gates cannot be weakened")
    minimum_length = gates["reject_ambiguous_identifiable_values_shorter_than"]
    threshold = gates["char_5gram_near_duplicate_threshold"]
    if (
        isinstance(minimum_length, bool)
        or not isinstance(minimum_length, int)
        or minimum_length < 2
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.9 <= float(threshold) < 1.0
    ):
        raise CrossWozSurrogateCandidateError("validation gate parameters failed validation")

    admission = _mapping(root["admission"], context="admission")
    _exact_keys(
        admission,
        {
            "status",
            "public_weight_training_allowed",
            "human_authored_context",
            "surrogate_values",
            "human_annotation",
            "human_gold",
            "contains_real_personal_data",
            "legal_review",
            "privacy_review",
            "human_review",
            "quality_review",
            "training_role",
        },
        context="admission",
    )
    if (
        admission["status"] != "quarantined"
        or admission["public_weight_training_allowed"] is not False
        or admission["human_authored_context"] is not True
        or admission["surrogate_values"] is not True
        or admission["human_annotation"] is not False
        or admission["human_gold"] is not False
        or admission["contains_real_personal_data"] != "not_claimed"
        or any(
            admission[name] != "pending"
            for name in ("legal_review", "privacy_review", "human_review", "quality_review")
        )
    ):
        raise CrossWozSurrogateCandidateError("admission must remain fail closed")
    roles = _mapping(admission["training_role"], context="admission.training_role")
    _exact_keys(roles, set(SPLITS), context="admission.training_role")
    if roles["test"] != "upstream_test_never_train":
        raise CrossWozSurrogateCandidateError("upstream test must remain never-train")

    lineage = _mapping(root["lineage_exclusions"], context="lineage_exclusions")
    _exact_keys(
        lineage,
        {
            "crosswoz_upstream_test_read_for_split_preserving_quarantine",
            "pii_bench_read",
            "external_benchmark_test_read",
            "confirmatory_hidden_test_read",
            "llm_or_api_used",
            "gpu_used",
        },
        context="lineage_exclusions",
    )
    if (
        lineage["crosswoz_upstream_test_read_for_split_preserving_quarantine"] is not True
        or any(
            lineage[name] is not False
            for name in (
                "pii_bench_read",
                "external_benchmark_test_read",
                "confirmatory_hidden_test_read",
                "llm_or_api_used",
                "gpu_used",
            )
        )
    ):
        raise CrossWozSurrogateCandidateError("forbidden lineage input was enabled")

    return CandidateRecipe(
        pipeline_id=PIPELINE_ID,
        pipeline_version=_string(root["pipeline_version"], context="pipeline_version"),
        source_id=CROSSWOZ_SOURCE_ID,
        source_revision=SOURCE_REVISION,
        source_license=_string(source["declared_license"], context="declared_license"),
        archives=archives,
        evidence=evidence,
        existing_quarantine=existing,
        quotas=quotas,
        recipe_namespace=_string(
            surrogates["recipe_namespace"], context="surrogates.recipe_namespace"
        ),
        minimum_source_value_length=minimum_length,
        near_duplicate_threshold=float(threshold),
        training_roles={split: _string(roles[split], context="training_role") for split in SPLITS},
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(root: Path, expectation: FileExpectation) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / expectation.relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise CrossWozSurrogateCandidateError("input path escaped its declared root") from None
    if candidate.is_symlink() or not candidate.is_file():
        raise CrossWozSurrogateCandidateError("input artifact must be a regular file")
    return candidate


def _git_head(source_root: Path) -> str:
    git_dir = source_root.resolve(strict=True) / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise CrossWozSurrogateCandidateError("upstream git metadata failed validation")
    head_path = git_dir / "HEAD"
    if head_path.is_symlink() or not head_path.is_file() or head_path.stat().st_size > 4096:
        raise CrossWozSurrogateCandidateError("upstream git HEAD failed validation")
    head = head_path.read_text(encoding="ascii").strip()
    if _REVISION_PATTERN.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        raise CrossWozSurrogateCandidateError("upstream git HEAD failed validation")
    reference = head.removeprefix("ref: ")
    if _GIT_REF_PATTERN.fullmatch(reference) is None or ".." in PurePosixPath(reference).parts:
        raise CrossWozSurrogateCandidateError("upstream git reference failed validation")
    loose = git_dir / reference
    if loose.is_file() and not loose.is_symlink():
        value = loose.read_text(encoding="ascii").strip()
        if _REVISION_PATTERN.fullmatch(value):
            return value
    packed = git_dir / "packed-refs"
    if packed.is_file() and not packed.is_symlink():
        for line in packed.read_text(encoding="ascii").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            fields = line.split(" ", 1)
            if (
                len(fields) == 2
                and fields[1] == reference
                and _REVISION_PATTERN.fullmatch(fields[0])
            ):
                return fields[0]
    raise CrossWozSurrogateCandidateError("upstream git reference could not be resolved")


def _verify_source_inputs(recipe: CandidateRecipe, source_root: Path) -> dict[str, Path]:
    if _git_head(source_root) != recipe.source_revision:
        raise CrossWozSurrogateCandidateError("upstream git revision mismatch")
    paths: dict[str, Path] = {}
    for split, expectation in recipe.archives.items():
        path = _regular_file(source_root, expectation)
        if _sha256_file(path) != expectation.sha256:
            raise CrossWozSurrogateCandidateError("upstream archive hash mismatch")
        paths[f"archive:{split}"] = path
    for name, expectation in recipe.evidence.items():
        path = _regular_file(source_root, expectation)
        if _sha256_file(path) != expectation.sha256:
            raise CrossWozSurrogateCandidateError("source evidence hash mismatch")
        paths[f"evidence:{name}"] = path
    return paths


def _verify_existing_quarantine(recipe: CandidateRecipe, root: Path) -> Mapping[str, str]:
    hashes: dict[str, str] = {}
    for split, expectation in recipe.existing_quarantine.items():
        path = _regular_file(root, expectation)
        mode = stat.S_IMODE(path.stat().st_mode)
        digest = _sha256_file(path)
        if digest != expectation.sha256 or mode != 0o600:
            raise CrossWozSurrogateCandidateError("existing quarantine integrity mismatch")
        hashes[split] = digest
    return hashes


def _load_archive(path: Path, expectation: FileExpectation) -> Mapping[str, Mapping[str, Any]]:
    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            if (
                len(infos) != 1
                or expectation.member is None
                or infos[0].filename != expectation.member
                or infos[0].file_size > _MAX_ARCHIVE_MEMBER_BYTES
                or infos[0].is_dir()
            ):
                raise CrossWozSurrogateCandidateError("upstream archive layout mismatch")
            value = json.loads(bundle.read(infos[0]))
    except CrossWozSurrogateCandidateError:
        raise
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
        raise CrossWozSurrogateCandidateError("upstream archive failed parsing") from None
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and key and isinstance(item, Mapping)
        for key, item in value.items()
    ):
        raise CrossWozSurrogateCandidateError("upstream archive schema failed validation")
    return value


def _dialogue_group(revision: str, dialogue_id: str) -> str:
    digest = hashlib.sha256(
        f"{CROSSWOZ_SOURCE_ID}\0{revision}\0{dialogue_id}".encode()
    ).hexdigest()
    return f"crosswoz-dialogue:sha256:{digest}"


def _doc_id(group: str, turn_index: int) -> str:
    digest = hashlib.sha256(f"{PIPELINE_ID}\0{group}\0{turn_index}".encode()).hexdigest()
    return f"crosswoz-turn:sha256:{digest}"


def _slot_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip().casefold() not in {"", "none"}:
            yield value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, str) and item.strip().casefold() not in {"", "none"}:
                yield item


def _add_evidence(
    result: dict[str, list[SlotEvidence]],
    *,
    domain: object,
    slot: object,
    value: object,
    origin: str,
) -> None:
    if not isinstance(domain, str) or not isinstance(slot, str):
        return
    category = _SLOT_CATEGORY.get((domain, slot))
    if category is None:
        return
    for item in _slot_values(value):
        result[item].append(
            SlotEvidence(
                value=item,
                category=category,
                domain=domain,
                slot=slot,
                origin=origin,
            )
        )


def _dialogue_evidence(dialogue: Mapping[str, Any]) -> Mapping[str, tuple[SlotEvidence, ...]]:
    result: dict[str, list[SlotEvidence]] = defaultdict(list)
    for field_name in ("goal", "final_goal"):
        for row in dialogue[field_name]:
            _add_evidence(
                result,
                domain=row[1],
                slot=row[2],
                value=row[3],
                origin=field_name,
            )
    for message in dialogue["messages"]:
        for act in message["dialog_act"]:
            _add_evidence(
                result,
                domain=act[1],
                slot=act[2],
                value=act[3],
                origin="dialog_act",
            )
        for row in message.get("user_state", ()):
            _add_evidence(
                result,
                domain=row[1],
                slot=row[2],
                value=row[3],
                origin="user_state",
            )
        for field_name in ("sys_state_init", "sys_state"):
            for domain, state in message.get(field_name, {}).items():
                for slot, value in state.items():
                    if slot != "selectedResults":
                        _add_evidence(
                            result,
                            domain=domain,
                            slot=slot,
                            value=value,
                            origin=field_name,
                        )
    return {
        value: tuple(
            sorted(
                set(items),
                key=lambda item: (item.category, item.domain, item.slot, item.origin),
            )
        )
        for value, items in result.items()
    }


def _direct_values(message: Mapping[str, Any], *, category: str) -> frozenset[str]:
    values = {
        act[3]
        for act in message["dialog_act"]
        if act[0] in {"Inform", "Recommend", "Select"}
        and _SLOT_CATEGORY.get((act[1], act[2])) == category
        and isinstance(act[3], str)
        and act[3]
        and act[3].casefold() != "none"
        and act[3] in message["content"]
    }
    return frozenset(values)


def _descriptors_for_split(
    *,
    split: str,
    dialogues: Mapping[str, Mapping[str, Any]],
    recipe: CandidateRecipe,
) -> tuple[TurnDescriptor, ...]:
    descriptors: list[TurnDescriptor] = []
    for dialogue_id, dialogue in dialogues.items():
        try:
            convert_crosswoz_record(
                dialogue,
                dialogue_id=dialogue_id,
                source_revision=recipe.source_revision,
                source_license=recipe.source_license,
            )
        except (ConversionError, TypeError, ValueError):
            raise CrossWozSurrogateCandidateError(
                "upstream dialogue failed strict schema validation"
            ) from None
        group = _dialogue_group(recipe.source_revision, dialogue_id)
        evidence = _dialogue_evidence(dialogue)
        for turn_index, message in enumerate(dialogue["messages"]):
            direct_plates = _direct_values(message, category="vehicle_license_plate")
            direct_names = _direct_values(message, category="organization_or_place_name")
            if direct_plates:
                kind = "vehicle_plate_semantic_candidate"
            elif direct_names:
                kind = "organization_place_person_name_hard_negative"
            else:
                continue
            descriptors.append(
                TurnDescriptor(
                    split=split,
                    dialogue_group=group,
                    dialogue_id=dialogue_id,
                    turn_index=turn_index,
                    role=message["role"],
                    content=message["content"],
                    candidate_kind=kind,
                    evidence_by_value=evidence,
                    direct_plate_values=direct_plates,
                )
            )
    return tuple(descriptors)


def _scan_high_risk(text: str) -> tuple[tuple[int, int, str, str], ...]:
    findings: list[tuple[int, int, str, str]] = []
    for match in _PHONE_PATTERN.finditer(text):
        findings.append((match.start(), match.end(), "pattern_phone", match.group(0)))
    for match in _EMAIL_PATTERN.finditer(text):
        findings.append((match.start(), match.end(), "pattern_email", match.group(0)))
    for match in _CN_ID_PATTERN.finditer(text):
        findings.append((match.start(), match.end(), "pattern_cn_resident_id", match.group(0)))
    for match in _CARD_PATTERN.finditer(text):
        value = match.group(0)
        if validate_luhn(value).valid:
            findings.append((match.start(), match.end(), "pattern_bank_card", value))
    for pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if "value" in match.groupdict() and match.group("value") is not None:
                start, end = match.span("value")
                value = match.group("value")
            else:
                start, end = match.span()
                value = match.group(0)
            findings.append((start, end, "pattern_secret", value))
    unique = {(start, end, category, value) for start, end, category, value in findings}
    return tuple(sorted(unique, key=lambda item: (item[0], item[1], item[2])))


def _source_spans(
    descriptor: TurnDescriptor,
    *,
    minimum_length: int,
) -> tuple[_SourceSpan, ...]:
    text = descriptor.content
    for value in descriptor.evidence_by_value:
        if len(value) < minimum_length and value in text:
            raise CrossWozSurrogateCandidateError("ambiguous short source slot value")
    values = sorted(
        (
            value
            for value in descriptor.evidence_by_value
            if len(value) >= minimum_length and value in text
        ),
        key=lambda value: (-len(value), value),
    )
    spans: dict[tuple[int, int], _SourceSpan] = {}
    if values:
        pattern = re.compile("|".join(re.escape(value) for value in values))
        for match in pattern.finditer(text):
            value = match.group(0)
            evidence = descriptor.evidence_by_value[value]
            category = min(
                (item.category for item in evidence), key=lambda item: _CATEGORY_PRIORITY[item]
            )
            spans[(match.start(), match.end())] = _SourceSpan(
                start=match.start(),
                end=match.end(),
                value=value,
                category=category,
                evidence=evidence,
                direct_plate=(value in descriptor.direct_plate_values),
            )

    for start, end, category, value in _scan_high_risk(text):
        existing = spans.get((start, end))
        if existing is not None:
            chosen = min((existing.category, category), key=lambda item: _CATEGORY_PRIORITY[item])
            spans[(start, end)] = _SourceSpan(
                start=start,
                end=end,
                value=value,
                category=chosen,
                evidence=existing.evidence,
                direct_plate=existing.direct_plate,
            )
            continue
        if any(start < other.end and other.start < end for other in spans.values()):
            raise CrossWozSurrogateCandidateError("overlapping source evidence is ambiguous")
        spans[(start, end)] = _SourceSpan(
            start=start,
            end=end,
            value=value,
            category=category,
            evidence=(),
            direct_plate=False,
        )

    ordered = tuple(sorted(spans.values(), key=lambda item: (item.start, item.end)))
    if not ordered:
        raise CrossWozSurrogateCandidateError("selected turn has no replaceable source evidence")
    if descriptor.candidate_kind == "vehicle_plate_semantic_candidate" and not any(
        span.category == "vehicle_license_plate" and span.direct_plate for span in ordered
    ):
        raise CrossWozSurrogateCandidateError("direct vehicle-plate span was not isolated")
    if descriptor.candidate_kind == "organization_place_person_name_hard_negative" and not any(
        span.category == "organization_or_place_name" for span in ordered
    ):
        raise CrossWozSurrogateCandidateError(
            "organization/place hard-negative span was not isolated"
        )
    return ordered


def _digest_stream(material: str) -> Iterable[int]:
    counter = 0
    while True:
        block = hashlib.sha256(f"{material}\0{counter}".encode()).digest()
        yield from block
        counter += 1


def _choose(stream: Iterable[int], alphabet: str, count: int) -> str:
    iterator = iter(stream)
    return "".join(alphabet[next(iterator) % len(alphabet)] for _ in range(count))


def _shape_chars(value: str, stream: Iterable[int], *, plate: bool = False) -> str:
    iterator = iter(stream)
    output: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if character.isdigit():
            output.append(str(next(iterator) % 10))
        elif "A" <= character <= "Z":
            output.append(_ASCII_UPPER[next(iterator) % len(_ASCII_UPPER)])
        elif "a" <= character <= "z":
            output.append(_ASCII_LOWER[next(iterator) % len(_ASCII_LOWER)])
        elif "\u3400" <= character <= "\u9fff":
            output.append(
                "测"
                if plate
                else _CJK_TOKEN_ALPHABET[next(iterator) % len(_CJK_TOKEN_ALPHABET)]
            )
        elif category.startswith("P"):
            output.append("※" if plate else character)
        else:
            output.append("X")
    return "".join(output)


def _phone_surrogate(value: str, stream: Iterable[int]) -> str:
    iterator = iter(stream)
    digit_index = 0
    output: list[str] = []
    for character in value:
        if character.isdigit():
            output.append("2" if digit_index == 0 else str(next(iterator) % 10))
            digit_index += 1
        else:
            output.append(character)
    result = "".join(output)
    if validate_phone(result).valid:
        raise CrossWozSurrogateCandidateError("phone surrogate remained contactable")
    return result


def _email_surrogate(value: str, stream: Iterable[int]) -> str:
    local_length = max(4, min(value.partition("@")[0].__len__(), 16))
    return _choose(stream, _ASCII_LOWER, local_length) + "@example.invalid"


def _id_surrogate(value: str, stream: Iterable[int]) -> str:
    iterator = iter(stream)
    output: list[str] = []
    digit_index = 0
    for character in value:
        if character.isdigit():
            output.append("0" if digit_index < 6 else str(next(iterator) % 10))
            digit_index += 1
        elif character in "Xx":
            output.append("X")
        else:
            output.append(character)
    result = "".join(output)
    if validate_cn_resident_id(result).valid:
        raise CrossWozSurrogateCandidateError("resident-ID surrogate remained valid")
    return result


def _card_surrogate(value: str, stream: Iterable[int]) -> str:
    iterator = iter(stream)
    digits = [str(next(iterator) % 10) for character in value if character.isdigit()]
    if len(digits) >= 12:
        normalized = "".join(digits)
        if validate_luhn(normalized).valid:
            digits[-1] = str((int(digits[-1]) + 1) % 10)
    digit_iterator = iter(digits)
    result = "".join(
        next(digit_iterator) if character.isdigit() else character for character in value
    )
    if validate_luhn(result).valid:
        raise CrossWozSurrogateCandidateError("bank-card surrogate retained a valid checksum")
    return result


def _secret_surrogate(value: str) -> str:
    result = "".join(
        "X"
        if "A" <= character <= "Z"
        else "x"
        if "a" <= character <= "z"
        else "0"
        if character.isdigit()
        else character
        for character in value
    )
    return result or "placeholder"


def _replacement(
    *,
    category: str,
    value: str,
    material: str,
    evidence: Sequence[SlotEvidence],
) -> str:
    stream = _digest_stream(material)
    token = _choose(_digest_stream(material + "\0token"), _CJK_TOKEN_ALPHABET, 3)
    if category in {"pattern_phone", "public_contact_phone"}:
        return _phone_surrogate(value, stream)
    if category == "pattern_email":
        return _email_surrogate(value, stream)
    if category == "pattern_cn_resident_id":
        return _id_surrogate(value, stream)
    if category == "pattern_bank_card":
        return _card_surrogate(value, stream)
    if category == "pattern_secret":
        return _secret_surrogate(value)
    if category == "vehicle_license_plate":
        return _shape_chars(value, stream, plate=True)
    if category == "organization_address":
        return f"示例大道{token}段"
    if category == "location_or_route_endpoint":
        return f"示例地点{token}"
    if category == "vehicle_model":
        return f"示例车型{token}"
    if category == "organization_or_place_name":
        domain_slots = {(item.domain, item.slot) for item in evidence}
        if any(slot == "周边景点" or domain == "景点" for domain, slot in domain_slots):
            prefix = "示例景点"
        elif any(slot == "周边餐馆" or domain == "餐馆" for domain, slot in domain_slots):
            prefix = "示例餐馆"
        elif any(slot == "周边酒店" or domain == "酒店" for domain, slot in domain_slots):
            prefix = "示例酒店"
        else:
            prefix = "示例地点"
        return prefix + token
    raise CrossWozSurrogateCandidateError("unsupported surrogate category")


def _relation_id(material: str) -> str:
    digest = hashlib.sha256((material + "\0relation").encode()).digest()[:18]
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"rel_v2_{encoded}"


def _known_safe_surrogate_scan(
    *,
    start: int,
    end: int,
    category: str,
    value: str,
    spans: Sequence[tuple[int, int, str]],
) -> bool:
    compatible = {
        "pattern_phone": {"pattern_phone", "public_contact_phone"},
        "pattern_email": {"pattern_email"},
        "pattern_cn_resident_id": {"pattern_cn_resident_id"},
        "pattern_bank_card": {"pattern_bank_card"},
        "pattern_secret": {"pattern_secret"},
    }
    if not any(
        span_start <= start and end <= span_end and span_category in compatible[category]
        for span_start, span_end, span_category in spans
    ):
        return False
    if category == "pattern_phone":
        return not validate_phone(value).valid
    if category == "pattern_email":
        return value.casefold().endswith(".invalid")
    if category == "pattern_cn_resident_id":
        return not validate_cn_resident_id(value).valid
    if category == "pattern_bank_card":
        return not validate_luhn(value).valid
    return category == "pattern_secret"


def _transform_turn(
    descriptor: TurnDescriptor,
    *,
    recipe: CandidateRecipe,
    attempt_ordinal: int,
) -> _TransformedTurn:
    source_spans = _source_spans(
        descriptor, minimum_length=recipe.minimum_source_value_length
    )
    unique_values: dict[str, tuple[str, tuple[SlotEvidence, ...], int]] = {}
    for span in source_spans:
        if span.value not in unique_values:
            unique_values[span.value] = (span.category, span.evidence, len(unique_values))

    replacements: dict[str, tuple[str, str]] = {}
    for value, (category, evidence, ordinal) in unique_values.items():
        material = (
            f"{recipe.recipe_namespace}\0{descriptor.split}\0{descriptor.dialogue_group}\0"
            f"{descriptor.turn_index}\0{descriptor.candidate_kind}\0{attempt_ordinal}\0"
            f"{category}\0{ordinal}"
        )
        replacements[value] = (
            _replacement(category=category, value=value, material=material, evidence=evidence),
            _relation_id(material),
        )

    edits = [
        SurrogateEdit(
            span=OriginalSpan(span.start, span.end, span.category, span.value),
            replacement=replacements[span.value][0],
            relation_id=replacements[span.value][1],
        )
        for span in source_spans
    ]
    try:
        rebuilt = rebuild_surrogate_text(descriptor.content, edits)
    except (SurrogateError, TypeError, ValueError):
        raise CrossWozSurrogateCandidateError("surrogate reconstruction failed") from None
    if any(value in rebuilt.text for value in unique_values):
        raise CrossWozSurrogateCandidateError("source value residual detected")
    if len(rebuilt.spans) != len(source_spans):
        raise CrossWozSurrogateCandidateError("surrogate span count mismatch")

    marker = "[USR]\n" if descriptor.role == "usr" else "[SYS]\n"
    shifted: list[tuple[int, int, str]] = []
    annotations: list[dict[str, Any]] = []
    entities: list[EntitySpan] = []
    value_groups: list[str] = []
    for source, surrogate in zip(source_spans, rebuilt.spans, strict=True):
        start = len(marker) + surrogate.start
        end = len(marker) + surrogate.end
        shifted.append((start, end, source.category))
        evidence_origins = sorted({item.origin for item in source.evidence})
        domain_slots = sorted({f"{item.domain}:{item.slot}" for item in source.evidence})
        annotations.append(
            {
                "start": start,
                "end": end,
                "candidate_type": source.category,
                "evidence_origins": evidence_origins or ["independent_pattern_scan"],
                "domain_slot_evidence": domain_slots,
                "surrogate_relation_id": surrogate.relation_id,
                "human_annotation": False,
                "human_gold": False,
                "person_name_gold": False,
            }
        )
        if source.category == "vehicle_license_plate" and source.direct_plate:
            entities.append(
                EntitySpan(
                    start=start,
                    end=end,
                    label="VEHICLE_LICENSE_PLATE",
                    text=surrogate.text,
                    source="crosswoz_dialog_act_semantic_surrogate_candidate",
                    risk_tier="T1",
                    validator_state="direct_slot_semantics_unreviewed_surrogate",
                    metadata={
                        "human_annotation": False,
                        "human_gold": False,
                        "real_personal_data": False,
                        "semantic_candidate_only": True,
                    },
                )
            )
            value_groups.append(stable_text_hash(surrogate.text))

    final_text = marker + rebuilt.text
    permitted_scan_count = 0
    scanner_counts = Counter()
    for start, end, category, value in _scan_high_risk(final_text):
        scanner_counts[category] += 1
        if not _known_safe_surrogate_scan(
            start=start,
            end=end,
            category=category,
            value=value,
            spans=shifted,
        ):
            raise CrossWozSurrogateCandidateError("uncovered high-risk residual detected")
        permitted_scan_count += 1

    provenance = Provenance(
        source_id=CROSSWOZ_SOURCE_ID,
        source_kind="public_human_wizard_of_oz_context_with_deterministic_surrogates",
        source_revision=recipe.source_revision,
        license=recipe.source_license,
        synthetic=False,
        evaluation_only=False,
        private_enterprise=False,
        metadata={
            "context_origin": "wizard_of_oz",
            "human_authored_context": True,
            "surrogate_values": True,
            "human_annotation": False,
            "human_gold": False,
            "contains_real_personal_data": "not_claimed",
            "source_values_stored": False,
            "source_annotator_identifiers_stored": False,
            "legal_review_status": "pending",
            "privacy_review_status": "pending",
            "human_review_status": "pending",
            "modified_from_upstream": True,
        },
    )
    quality = QualityMetadata(
        tier=QualityTier.UNCERTAIN,
        quality_gate=False,
        validators_passed=True,
        review_status="legal_privacy_and_human_annotation_review_pending",
        confidence=None,
        metadata={
            "automated_structural_validation": True,
            "automated_source_residual_scan": True,
            "automated_high_risk_pattern_scan": True,
            "human_annotation_complete": False,
        },
    )
    record = DocumentRecord.create(
        doc_id=_doc_id(descriptor.dialogue_group, descriptor.turn_index),
        text=final_text,
        entities=entities,
        language="zh-Hans-CN",
        domain="travel_and_local_services",
        scene="task_oriented_dialogue_turn",
        provenance=provenance,
        quality=quality,
        public_weight_training_allowed=False,
        conversation_group=descriptor.dialogue_group,
        entity_value_groups=tuple(sorted(set(value_groups))),
        split=descriptor.split,
        metadata={
            "candidate_kind": descriptor.candidate_kind,
            "annotation_candidates": annotations,
            "human_authored_context": True,
            "surrogate_values": True,
            "human_annotation": False,
            "human_gold": False,
            "contains_real_personal_data": "not_claimed",
            "source_values_stored": False,
            "source_value_residual_count": 0,
            "uncovered_high_risk_scan_count": 0,
            "replacement_count": len(source_spans),
            "source_turn_index": descriptor.turn_index,
            "upstream_split_preserved": True,
            "training_role": recipe.training_roles[descriptor.split],
            "person_name_gold_from_organization_or_place": False,
        },
    )
    if record.data_pool is not DataPool.QUARANTINED:
        raise CrossWozSurrogateCandidateError("candidate escaped quarantine")
    replacement_hashes = frozenset(
        hashlib.sha256(replacement.encode("utf-8")).hexdigest()
        for replacement, _ in replacements.values()
    )
    return _TransformedTurn(
        record=record,
        replacement_hashes=replacement_hashes,
        scanner_counts=dict(scanner_counts),
        permitted_surrogate_scan_count=permitted_scan_count,
        replacement_count=len(source_spans),
    )


def _ngrams(value: str) -> frozenset[str]:
    normalized = "".join(value.casefold().split())
    if len(normalized) < _NGRAM_SIZE:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(
        normalized[index : index + _NGRAM_SIZE]
        for index in range(len(normalized) - _NGRAM_SIZE + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _rank(descriptor: TurnDescriptor) -> str:
    return hashlib.sha256(
        f"{descriptor.split}\0{descriptor.dialogue_group}\0{descriptor.turn_index}\0"
        f"{descriptor.candidate_kind}".encode()
    ).hexdigest()


def _cross_split_collision_count(values: Mapping[str, set[str]]) -> int:
    count = 0
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            count += len(values[left] & values[right])
    return count


def _duplicate_audit(records: Sequence[DocumentRecord], threshold: float) -> Mapping[str, Any]:
    counts = Counter(record.text for record in records)
    exact = sum(count * (count - 1) // 2 for count in counts.values())
    grams = [_ngrams(record.text) for record in records]
    near = 0
    compared = 0
    for right, right_grams in enumerate(grams):
        for left in range(right):
            left_grams = grams[left]
            larger = max(len(left_grams), len(right_grams))
            smaller = min(len(left_grams), len(right_grams))
            if larger and smaller / larger < threshold:
                continue
            compared += 1
            near += _jaccard(left_grams, right_grams) >= threshold
    return {
        "exact_duplicate_pair_count": exact,
        "near_duplicate_pair_count": near,
        "candidate_pair_count": compared,
        "near_duplicate_method": "character_5gram_jaccard",
        "near_duplicate_threshold": threshold,
    }


def build_candidate_dataset(
    *,
    recipe: CandidateRecipe,
    source_root: Path,
    existing_quarantine_dir: Path,
) -> CandidateBuild:
    """Validate fixed inputs and construct a small deterministic in-memory candidate."""

    paths = _verify_source_inputs(recipe, source_root)
    existing_before = _verify_existing_quarantine(recipe, existing_quarantine_dir)
    dialogues_by_split = {
        split: _load_archive(paths[f"archive:{split}"], recipe.archives[split])
        for split in SPLITS
    }

    all_groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split, dialogues in dialogues_by_split.items():
        all_groups[split] = {
            _dialogue_group(recipe.source_revision, dialogue_id) for dialogue_id in dialogues
        }
    if _cross_split_collision_count(all_groups):
        raise CrossWozSurrogateCandidateError("upstream dialogue group crosses splits")

    descriptors = {
        split: _descriptors_for_split(
            split=split,
            dialogues=dialogues_by_split[split],
            recipe=recipe,
        )
        for split in SPLITS
    }
    selected: list[DocumentRecord] = []
    selected_grams: list[frozenset[str]] = []
    selected_texts: set[str] = set()
    selected_groups: set[str] = set()
    replacement_hash_splits: dict[str, set[str]] = {split: set() for split in SPLITS}
    rejection_counts: Counter[str] = Counter()
    scanner_counts: Counter[str] = Counter()
    permitted_surrogate_scan_count = 0
    replacement_count = 0
    selected_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    for split in SPLITS:
        attempted = 0
        for descriptor in sorted(descriptors[split], key=_rank):
            if all(
                selected_counts[split][kind] >= recipe.quotas[split][kind]
                for kind in CANDIDATE_KINDS
            ):
                break
            if (
                selected_counts[split][descriptor.candidate_kind]
                >= recipe.quotas[split][descriptor.candidate_kind]
            ):
                continue
            if descriptor.dialogue_group in selected_groups:
                rejection_counts["dialogue_already_selected"] += 1
                continue
            attempted += 1
            try:
                transformed = _transform_turn(
                    descriptor,
                    recipe=recipe,
                    attempt_ordinal=attempted,
                )
            except CrossWozSurrogateCandidateError as exc:
                reason = str(exc).replace(" ", "_")
                rejection_counts[reason] += 1
                continue
            text = transformed.record.text
            grams = _ngrams(text)
            if text in selected_texts:
                rejection_counts["exact_duplicate"] += 1
                continue
            if any(
                _jaccard(grams, prior) >= recipe.near_duplicate_threshold
                for prior in selected_grams
            ):
                rejection_counts["near_duplicate"] += 1
                continue
            other_split_hashes = set().union(
                *(replacement_hash_splits[other] for other in SPLITS if other != split)
            )
            if transformed.replacement_hashes & other_split_hashes:
                rejection_counts["cross_split_surrogate_value_collision"] += 1
                continue
            selected.append(transformed.record)
            selected_groups.add(descriptor.dialogue_group)
            selected_texts.add(text)
            selected_grams.append(grams)
            replacement_hash_splits[split].update(transformed.replacement_hashes)
            selected_counts[split][descriptor.candidate_kind] += 1
            scanner_counts.update(transformed.scanner_counts)
            permitted_surrogate_scan_count += transformed.permitted_surrogate_scan_count
            replacement_count += transformed.replacement_count

        for kind in CANDIDATE_KINDS:
            if selected_counts[split][kind] != recipe.quotas[split][kind]:
                raise CrossWozSurrogateCandidateError(
                    "safe candidate quota could not be filled; no output may be materialized"
                )

    duplicate_audit = _duplicate_audit(selected, recipe.near_duplicate_threshold)
    if (
        duplicate_audit["exact_duplicate_pair_count"]
        or duplicate_audit["near_duplicate_pair_count"]
    ):
        raise CrossWozSurrogateCandidateError("final duplicate audit failed")
    record_groups = {split: set() for split in SPLITS}
    for record in selected:
        if record.split not in record_groups or record.conversation_group is None:
            raise CrossWozSurrogateCandidateError("record split/group invariant failed")
        record_groups[record.split].add(record.conversation_group)
        if (
            record.public_weight_training_allowed is not False
            or record.data_pool is not DataPool.QUARANTINED
        ):
            raise CrossWozSurrogateCandidateError("record admission invariant failed")
        if (
            record.split == "test"
            and record.metadata.get("training_role") != "upstream_test_never_train"
        ):
            raise CrossWozSurrogateCandidateError("upstream test training-role invariant failed")
        if any(entity.label == "PERSON_NAME" for entity in record.entities):
            raise CrossWozSurrogateCandidateError("organization/place was promoted to PERSON_NAME")
    group_collisions = _cross_split_collision_count(record_groups)
    surrogate_collisions = _cross_split_collision_count(replacement_hash_splits)
    if group_collisions or surrogate_collisions:
        raise CrossWozSurrogateCandidateError("final cross-split collision audit failed")
    if _verify_existing_quarantine(recipe, existing_quarantine_dir) != existing_before:
        raise CrossWozSurrogateCandidateError("existing quarantine changed during build")

    by_split = {
        split: {
            "document_count": sum(selected_counts[split].values()),
            "candidate_kind_counts": {
                kind: selected_counts[split][kind] for kind in CANDIDATE_KINDS
            },
            "training_role": recipe.training_roles[split],
        }
        for split in SPLITS
    }
    entity_counts = Counter(entity.label for record in selected for entity in record.entities)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "crosswoz_human_context_surrogate_annotation_candidate",
        "pipeline_id": recipe.pipeline_id,
        "pipeline_version": recipe.pipeline_version,
        "source": {
            "source_id": recipe.source_id,
            "revision": recipe.source_revision,
            "declared_license": recipe.source_license,
            "git_revision_verified": True,
            "archive_hashes_verified": True,
            "license_snapshot_hash_verified": True,
            "root_readme_hash_verified": True,
            "data_readme_hash_verified": True,
            "upstream_split_groups_preserved": True,
        },
        "admission": {
            "status": "quarantined",
            "public_weight_training_allowed": False,
            "human_authored_context": True,
            "surrogate_values": True,
            "human_annotation": False,
            "human_gold": False,
            "contains_real_personal_data": "not_claimed",
            "legal_review": "pending",
            "privacy_review": "pending",
            "human_review": "pending",
            "quality_review": "pending",
        },
        "counts": {
            "document_count": len(selected),
            "by_split": by_split,
            "candidate_entity_label_counts": dict(sorted(entity_counts.items())),
            "replacement_span_count": replacement_count,
            "post_scan_permitted_surrogate_finding_count": permitted_surrogate_scan_count,
            "post_scan_finding_counts": dict(sorted(scanner_counts.items())),
            "rejected_candidate_counts": dict(sorted(rejection_counts.items())),
        },
        "validation": {
            "source_value_residual_count": 0,
            "uncovered_phone_email_id_card_secret_scan_count": 0,
            "offset_alignment_error_count": 0,
            "organization_or_place_promoted_to_person_name_count": 0,
            "cross_split_group_collision_count": group_collisions,
            "cross_split_surrogate_value_collision_count": surrogate_collisions,
            "duplicate_audit": duplicate_audit,
            "existing_quarantine_unchanged": True,
            "existing_quarantine_bytes_rewritten": False,
            "output_directory_mode": "0700",
            "output_file_mode": "0600",
        },
        "annotation_semantics": {
            "vehicle_plate_labels": "direct_dialog_act_semantic_candidate_only",
            "organization_place_names": "annotation_candidate_and_person_name_hard_negative_only",
            "uncertain_values": "annotation_candidate_or_excluded",
            "human_review_required_before_any_gold_claim": True,
        },
        "license_and_attribution": {
            "technical_evidence_gate": "pass",
            "license_applicability_legal_conclusion": "pending",
            "attribution_required": True,
            "modification_notice_required": True,
            "publication_allowed": False,
        },
        "lineage_exclusions": {
            "crosswoz_upstream_test_read_for_split_preserving_quarantine": True,
            "pii_bench_read": False,
            "external_benchmark_test_read": False,
            "confirmatory_hidden_test_read": False,
            "llm_or_api_used": False,
            "gpu_used": False,
        },
        "privacy": {
            "aggregate_report_contains_raw_text": False,
            "aggregate_report_contains_document_ids": False,
            "aggregate_report_contains_entity_values": False,
            "aggregate_report_contains_source_values": False,
            "aggregate_report_contains_absolute_paths": False,
            "quarantined_records_use_surrogates_for_extracted_identifiable_values": True,
        },
    }
    _assert_aggregate_report_safe(report)
    return CandidateBuild(recipe=recipe, records=tuple(selected), report=report)


def _assert_aggregate_report_safe(report: Mapping[str, Any]) -> None:
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if re.search(r'(?<![A-Za-z0-9_.-])/(?:[^\s"}]+)', serialized):
        raise CrossWozSurrogateCandidateError("aggregate report contains an absolute path")
    if '"doc_id"' in serialized or '"conversation_group"' in serialized:
        raise CrossWozSurrogateCandidateError("aggregate report contains record identifiers")
    for value in _walk_strings(report):
        if "[USR]" in value or "[SYS]" in value:
            raise CrossWozSurrogateCandidateError("aggregate report contains record text")


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_strings(item)


def _write_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _write_external_report(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CrossWozSurrogateCandidateError("aggregate report already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def materialize_candidate_dataset(
    build: CandidateBuild,
    *,
    output_dir: Path,
    aggregate_report_path: Path,
    existing_quarantine_dir: Path,
) -> Mapping[str, Any]:
    """Atomically write new mode-600 quarantine files without touching old bytes."""

    if output_dir.exists() or output_dir.is_symlink():
        raise CrossWozSurrogateCandidateError("candidate output already exists")
    if aggregate_report_path.exists() or aggregate_report_path.is_symlink():
        raise CrossWozSurrogateCandidateError("aggregate report already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    before = _verify_existing_quarantine(build.recipe, existing_quarantine_dir)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    os.chmod(temporary, 0o700)
    finalized = False
    try:
        outputs: list[dict[str, Any]] = []
        for split in SPLITS:
            destination = temporary / f"{split}.quarantined.jsonl"
            split_records = [record for record in build.records if record.split == split]
            with destination.open("x", encoding="utf-8") as stream:
                for record in split_records:
                    stream.write(dumps_document(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(destination, 0o600)
            outputs.append(
                {
                    "split": split,
                    "filename": destination.name,
                    "document_count": len(split_records),
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            )
        if _verify_existing_quarantine(build.recipe, existing_quarantine_dir) != before:
            raise CrossWozSurrogateCandidateError("existing quarantine changed during write")
        report = dict(build.report)
        report["outputs"] = outputs
        _assert_aggregate_report_safe(report)
        _write_json(temporary / "aggregate_manifest.json", report)
        os.replace(temporary, output_dir)
        finalized = True
        if _verify_existing_quarantine(build.recipe, existing_quarantine_dir) != before:
            raise CrossWozSurrogateCandidateError("existing quarantine changed after write")
        for path in output_dir.iterdir():
            if path.is_file() and stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise CrossWozSurrogateCandidateError("candidate output mode validation failed")
        if stat.S_IMODE(output_dir.stat().st_mode) != 0o700:
            raise CrossWozSurrogateCandidateError("candidate directory mode validation failed")
        _write_external_report(aggregate_report_path, report)
        return report
    except Exception:
        if finalized and output_dir.exists():
            shutil.rmtree(output_dir)
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "CANDIDATE_KINDS",
    "DEFAULT_RECIPE_PATH",
    "PIPELINE_ID",
    "SOURCE_REVISION",
    "CandidateBuild",
    "CandidateRecipe",
    "CrossWozSurrogateCandidateError",
    "build_candidate_dataset",
    "load_recipe",
    "materialize_candidate_dataset",
]
