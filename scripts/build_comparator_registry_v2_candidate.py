#!/usr/bin/env python3
"""Freeze and verify the blocked Generation-2 comparator candidate set.

This builder is intentionally metadata-only and CPU-only.  It binds already
public aggregate reports, frozen source closures, exact model revisions, label
mapping payloads, and current replay sources.  It does not open benchmark rows,
prediction JSONL, model weights, hidden gold, GPU state, or the network.

The output is a *blocked candidate-set freeze*, not an active leaderboard or a
best/first/blind/SOTA claim.  Normal successful construction therefore exits 1;
validation or publication failures exit 2.
"""

# ruff: noqa: E501 -- immutable hashes, logical paths, and receipt text are intentionally explicit.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_RELATIVE_PATH = "configs/evaluation/comparator_registry_v2_candidate.schema.json"
REGISTRY_RELATIVE_PATH = "configs/evaluation/comparator_registry_v2_generation_2_candidate.json"
RECEIPT_RELATIVE_PATH = (
    "reports/evaluation/comparator_registry_v2_generation_2_candidate_blocked.json"
)
BUILDER_RELATIVE_PATH = "scripts/build_comparator_registry_v2_candidate.py"

REGISTRY_ID = "comparator_registry_v2_generation_2"
RECEIPT_ID = "comparator_registry_v2_generation_2_candidate_freeze_receipt"
SEARCH_COMPLETED_AT = "2026-07-16T03:19:26+08:00"
EXPECTED_FORMAL_FROZEN_AT = "2026-07-16T04:26:44+08:00"
EXPECTED_SCHEMA_FILE_SHA256 = "8daf14beaaf88a70ff49bd229b322e76714518ea939cb51b35d3f837c80336b4"

HASH_CONTRACT = (
    "registry_sha256 is SHA-256 of canonical UTF-8 JSON after removing the "
    "top-level registry_sha256 field; keys are sorted, separators are compact, "
    "ensure_ascii=false, allow_nan=false"
)
RECEIPT_HASH_CONTRACT = (
    "receipt_sha256 is SHA-256 of canonical UTF-8 JSON after removing the "
    "top-level receipt_sha256 field under the same canonical JSON rules"
)

CLOSED_8_LABELS = (
    "ADDRESS",
    "BANK_CARD_NUMBER",
    "CN_RESIDENT_ID",
    "EMAIL_ADDRESS",
    "PASSPORT_NUMBER",
    "PERSON_NAME",
    "PHONE_NUMBER",
    "VEHICLE_LICENSE_PLATE",
)
ALL_SYSTEM_IDS = (
    "aiguard_qwen3_0_6b",
    "openai_privacy_filter",
    "pii_engineer_chinese_ner_v1_0",
    "pii_engineer_multi_ner_v2_1",
    "ernie_chinese_ner",
    "cluener2020_roberta",
    "cn_common_v5",
    "argus_redact_fast",
    "openmed_snowflake",
    "openmed_qwenmed",
    "openmed_bigmed",
    "openmed_multilingual_privacy_filter",
)

EXPECTED_FILES: dict[str, str] = {
    SCHEMA_RELATIVE_PATH: EXPECTED_SCHEMA_FILE_SHA256,
    "configs/evaluation/comparator_registry_v1.schema.json": (
        "2955138a1ffeec6024df97715c67a598dad14182a6446e9a7a6d759f4e8bbf99"
    ),
    "configs/evaluation/comparator_registry_v1.yaml": (
        "b82b6492bf5fc3c57a3053f846138924bf61930876d2acf83bec40547de80cfb"
    ),
    "configs/evaluation/comparator_registry_v1_withdrawal.yaml": (
        "5601606a625b09bda0ccf13af7461996e7f6b5cf9f0f8cbc5f56edab33117bf3"
    ),
    "reports/baselines/aiguard_pii_bench_zh.json": (
        "c10fa63beaebff42ad402fcdfca0642c7d899e332a4da3ce3eadc8d52bfecd97"
    ),
    "reports/baselines/openai_privacy_filter_pii_bench_zh.json": (
        "0461126c4f4b94b2657337325f72ba39f1b341fdc23de19d980947f1dbf5777f"
    ),
    "reports/baselines/pii_engineer_chinese_ner_pii_bench_zh.json": (
        "0d7a8e974c4b20afb3272208ba3a280e21710f9b207ee707ae6ebbc6791b2177"
    ),
    "reports/baselines/presidio_ernie_pii_bench_zh.json": (
        "b5f9442503d7e0acc8914e2d001f357df7f44212a206760d06210e2f634c7406"
    ),
    "reports/baselines/presidio_cluener_pii_bench_zh.json": (
        "a32b5d87c39604ea37f5f5dfd39f294cb560f6e1ff30c40ebbceac82cd718957"
    ),
    "reports/baselines/presidio_rules_closed8_pii_bench_zh.json": (
        "7932f7566268a17f5ff741cfd07df054445f12b290706506850d9135142bbec8"
    ),
    "reports/baselines/argus_redact_pii_bench_zh.json": (
        "2cd12c69aec08544f99aa6e9af21dbd0bfcfc8701e7620d92213021d24e262bd"
    ),
    "reports/baselines/openmed_generation2_repair_v2_results.json": (
        "95c7fc0c5c910a3d985222014613bafb8102f5c39d8aeff66117ecb30b540328"
    ),
    "reports/evaluation_readiness/evaluation_readiness_post_repair_v5_blocked.json": (
        "033ca89c9d70a2864078f837ff74217331e67563bfccaeea421d13ce5d44adf6"
    ),
    "scripts/evaluate_aiguard_baseline.py": (
        "4ede18a1548ab0e66fee7979ff3797829ffb7025a4401f863dcade45b0119f4c"
    ),
    "scripts/evaluate_openai_privacy_filter_baseline.py": (
        "53211458cde24ff0abd28692739762fb2e198bf096683a7bf50b5aa085da9adc"
    ),
    "scripts/evaluate_pii_engineer_baseline.py": (
        "93902073a81e115a673fdfdfe66b874bdae00da7d0bfc0a18bd2f2e1573654bd"
    ),
    "scripts/evaluate_presidio_ernie_baseline.py": (
        "0fe286f041fb0f12560d0da05dc464d3f9a741fbc6b4995c443538366e6e93f5"
    ),
    "scripts/evaluate_presidio_cluener_baseline.py": (
        "c8d09febff00a3734e2660fa8019eb20e6fe4d2bc774d8eb4a472a7f96dfd1cc"
    ),
    "scripts/predict_rules.py": (
        "b4df98fe9ad82e5d7fbca7d446be46a33d530455e6b9123550f25faa13d8abac"
    ),
    "scripts/evaluate_argus_redact_baseline.py": (
        "d0ab1c76021763916576bc4937eaac5a2d110bd7263f5766e5faadf55c153db5"
    ),
    "scripts/evaluate_openmed_snowflake_repair_v2.py": (
        "4902e8bde96e836bb5dc4f58e1f432896ca59206f569ce4bcbdbb49993c5324d"
    ),
    "scripts/evaluate_openmed_qwenmed_baseline_v2.py": (
        "73eef1f15567e3d402fa3693c64d5b9c70f3bdbb1044d5412e6ae1abc30b61b2"
    ),
    "scripts/evaluate_openmed_bigmed_baseline_v2.py": (
        "8fb493ac7dea3eacfa98bf2160c1246978b9aa3df833b081983935101738ec8e"
    ),
    "scripts/evaluate_openmed_privacy_filter_gen2_repair_v2.py": (
        "863e37c93ef1b2c754fc43d16e63d4aa160986d9ba899b46f95bad360f7f624b"
    ),
    "configs/evaluation/openmed_snowflake_closed8_model_raw_hybrid_v6.yaml": (
        "c1b3897eb3f946309789471277ff8553d098555142aa16d610f4acde5f75debf"
    ),
    "configs/evaluation/openmed_qwenmed_closed8_model_raw_hybrid_v1.yaml": (
        "c028e8eae45780d10416a1efcdc1159664bf08ae5b739ec045a35490935552f7"
    ),
    "configs/evaluation/openmed_bigmed_closed8_model_raw_hybrid_v1.json": (
        "335e787ab5d1ec4e3cc40dbc71d8daba0fb3a4769d6b233491fa493383d544fd"
    ),
    "configs/evaluation/openmed_privacy_filter_gen2_v1.final.yaml": (
        "443da78ffc10c032250c211ea2b2b58b0b13238feba00256b14584b4d82d6549"
    ),
    "configs/evaluation/openmed_generation2_closed8_repair_umbrella_v2.freeze.json": (
        "1ad600d73845912b4bf770798f51b66cb2e2b7fd3b1f524aac0840e718d2161e"
    ),
    "configs/evaluation/openmed_snowflake_repair_gate_v3.freeze.json": (
        "38dc7ba308fe4ed1cf6f22ac706b182788a5e631622e27274242cf49457d509a"
    ),
    "configs/evaluation/openmed_qwenmed_repair_gate_v2.freeze.json": (
        "80f8e3eafd077b9189ccbc3e34b152f365de3adff3ab74fe130caacb07d17714"
    ),
    "configs/evaluation/openmed_bigmed_repair_gate_v2.freeze.json": (
        "58491379d68ca20b6f7d4fac87d0c62408f2dff7ee4264a6932a56f108dd2a29"
    ),
    "configs/evaluation/openmed_privacy_filter_repair_gate_v3.freeze.json": (
        "8a0e9f0a92591eb0ef1d6db8cb81a28355fe9a43b19f97cb2fffb3c7ecbe6f73"
    ),
    "configs/evaluation/openmed_eval_source_closure_v5.json": (
        "aebfc518cf191255895ed55ba06af6dad36190810ae52663cca7d539a1a8a876"
    ),
    "configs/evaluation/openmed_qwenmed_eval_source_closure_v2.json": (
        "a26149fba5fc0607630a44d1880519e540d3a45e8f53c7955af4b428994d0f18"
    ),
    "configs/evaluation/openmed_bigmed_eval_source_closure_v2.json": (
        "1c679f3f21914e963f2c9584c91077d6d4ea04f11c322121936a0b42fe886a45"
    ),
    "configs/evaluation/openmed_privacy_filter_eval_source_closure_v2.json": (
        "eade95d508bcc7462ccee17f68271bcf245aa4d860cf268cdaba1711acce72b8"
    ),
    "scripts/evaluate.py": ("99078f06a7dbf40884aa376e439b343b5c97c654afcef8571265a53bc7412e08"),
    "src/pii_zh/evaluation/__init__.py": (
        "376e0560ce2d22c1faf43e57c68b12a4bd5b049c2ba15c9eb4b76e5cacace5ab"
    ),
    "src/pii_zh/evaluation/io.py": (
        "63d23d4418e73e9ebbe0ad0b5fe1a2c5895ba2e3ea38559c964cef6c0882750a"
    ),
    "src/pii_zh/evaluation/metrics.py": (
        "1e9623ac4be80bb3bf8218b81c8079c8843b096680f1a2e1de8ce723226ed271"
    ),
    "src/pii_zh/evaluation/provenance.py": (
        "c5c97e2e319c00322a0a1ceff00217c7fbed77144bf4c714ea8314d3d67a63ca"
    ),
    "src/pii_zh/evaluation/required_metrics.py": (
        "1d362661c53b5282302dd980f4c8875cf55252ea5017b3ef1a9457677a4a87e8"
    ),
    "src/pii_zh/evaluation/paired_bootstrap.py": (
        "af3852bbfe9cf8af98a0343a34bab2f3b5088e53a410f25b026af82564143ae6"
    ),
    "src/pii_zh/evaluation/baseline_reporting.py": (
        "3eae7efa5a7f8dee4f7a9d0528c16a33ca90e1e13a69110e275ccfad5312b8e1"
    ),
    "src/pii_zh/evaluation/openmed_generation2_runtime.py": (
        "f9295ef2e82497563770605748b90f91a2f56bd07a57650d103bb7b5d81caa1c"
    ),
    "src/pii_zh/taxonomy/__init__.py": (
        "83a787764dd49e00027493ef75e1987e1befaf93cee8cbd686114a6fc1822674"
    ),
    "src/pii_zh/taxonomy/loader.py": (
        "7a66ad992a58f6f598633e841d3d99f636cdca2de3d5e96c4648c8a1d2ee842c"
    ),
    "src/pii_zh/taxonomy/taxonomy.yaml": (
        "8e3db18bb769ade3cb2487ae4d95c7c3863abc36845e99391ce55603e4409c13"
    ),
    "src/pii_zh/taxonomy/presidio_mapping.yaml": (
        "9b750a50c14d65fd63473dd9b2af9dc617375831e1014dbff13518dd837b6c4a"
    ),
}

V1_REGISTRY_CANONICAL_SHA256 = "37565121cffb708a87812bc2d3a86a8988b54f458c38f12d87055bfb36d4a5f2"
V1_WITHDRAWAL_CANONICAL_SHA256 = "cc0e4ba05a3735f072a21b519729c41bac85a596322fbf0ee89d77919904d00c"
OPENMED_RESULTS_CANONICAL_SHA256 = (
    "6101c085cfd0e47c32e611f119466f8983a93ce322bb7e757eef72d5794bc478"
)

BENCHMARK = {
    "repository": "wan9yu/pii-bench-zh",
    "revision": "c350b94897af668517ff5de237d89f2ce2eaa6f0",
    "manifest_file_sha256": ("62c03dc8c5ba343a883498cf92d272ab9f240252e92ea06a665022f5ce7df6f1"),
    "manifest_logical_sha256": ("545c990c96f23118ddc097ec0bed1d7f180785ac72e9f2a6ed56eb1ba3ee5e9b"),
    "subsets": {
        "formal": {
            "record_count": 5000,
            "gold_span_count": 15662,
            "canonical_sha256": (
                "bb978137711cc90ae0c16df0645983413f3b34943bf29469e58b795b183a38af"
            ),
        },
        "chat": {
            "record_count": 3000,
            "gold_span_count": 7544,
            "canonical_sha256": (
                "2d73326a7ea2da9966ba13336db1a791f03d5f18b7982fdc4495cb726c1180a5"
            ),
        },
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_FORMAL_TIME_FUTURE_SKEW = timedelta(seconds=5)
_ALLOWED_REPOSITORY_ROOTS = frozenset({"configs", "reports", "scripts", "src", "docs"})


class ComparatorCandidateError(RuntimeError):
    """Raised when the candidate freeze cannot be proven safely."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """One stable, no-follow repository file snapshot."""

    relative_path: str
    content: bytes
    file_sha256: str
    mode: int
    mtime_ns: int


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    value: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in value
        except TypeError as exc:
            raise ComparatorCandidateError("YAML mapping key is not hashable") from exc
        if duplicate:
            raise ComparatorCandidateError(f"duplicate YAML key: {key!r}")
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical byte representation used by every self-hash."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ComparatorCandidateError("value is not canonical JSON") from exc


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise ComparatorCandidateError(f"duplicate JSON key: {key}")
        value[key] = nested
    return value


def _parse_json_bytes(content: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ComparatorCandidateError(f"non-finite JSON constant in {field}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparatorCandidateError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ComparatorCandidateError(f"{field} must contain a JSON object")
    return value


def _parse_yaml_bytes(content: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = yaml.load(content.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ComparatorCandidateError(f"{field} is not strict UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise ComparatorCandidateError(f"{field} must contain a YAML object")
    if not all(isinstance(key, str) for key in value):
        raise ComparatorCandidateError(f"{field} root keys must be strings")
    return cast(dict[str, Any], value)


def _validate_repo_relative_path(relative_path: object) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path:
        raise ComparatorCandidateError("repository path must be a non-empty string")
    if (
        "\\" in relative_path
        or "\x00" in relative_path
        or relative_path.startswith("/")
        or relative_path.startswith("//")
        or _WINDOWS_ABSOLUTE_RE.match(relative_path)
    ):
        raise ComparatorCandidateError("repository path is absolute or non-POSIX")
    raw_parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ComparatorCandidateError("repository path is not normalized")
    parsed = PurePosixPath(relative_path)
    if parsed.is_absolute() or tuple(parsed.parts) != tuple(raw_parts):
        raise ComparatorCandidateError("repository path is not canonical")
    if raw_parts[0] not in _ALLOWED_REPOSITORY_ROOTS:
        raise ComparatorCandidateError("repository path root is not allowed")
    return tuple(raw_parts)


def _same_stat(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and stat.S_IMODE(before.st_mode) == stat.S_IMODE(after.st_mode)
    )


def _read_repository_artifact(repository_root: Path, relative_path: str) -> Artifact:
    """Read one file through an openat/O_NOFOLLOW descriptor chain."""

    parts = _validate_repo_relative_path(relative_path)
    root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(repository_root, root_flags)
    except OSError as exc:
        raise ComparatorCandidateError("repository root is not a no-follow directory") from exc
    current_fd = root_fd
    try:
        for component in parts[:-1]:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise ComparatorCandidateError(
                    f"repository path ancestor is missing or a symlink: {relative_path}"
                ) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        leaf_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            leaf_fd = os.open(parts[-1], leaf_flags, dir_fd=current_fd)
        except OSError as exc:
            raise ComparatorCandidateError(
                f"repository path leaf is missing or a symlink: {relative_path}"
            ) from exc
        try:
            before = os.fstat(leaf_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ComparatorCandidateError(f"artifact is not a regular file: {relative_path}")
            if before.st_size > _MAX_INPUT_BYTES:
                raise ComparatorCandidateError(f"artifact is too large: {relative_path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(leaf_fd, min(1024 * 1024, _MAX_INPUT_BYTES - total + 1))
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if total > _MAX_INPUT_BYTES:
                    raise ComparatorCandidateError(f"artifact exceeded size limit: {relative_path}")
            after = os.fstat(leaf_fd)
        finally:
            os.close(leaf_fd)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    if not _same_stat(before, after):
        raise ComparatorCandidateError(f"artifact changed during read: {relative_path}")
    content = b"".join(chunks)
    return Artifact(
        relative_path=relative_path,
        content=content,
        file_sha256=hashlib.sha256(content).hexdigest(),
        mode=stat.S_IMODE(after.st_mode),
        mtime_ns=after.st_mtime_ns,
    )


def _parse_aware_time(value: object, *, field: str, require_shanghai_offset: bool) -> datetime:
    if not isinstance(value, str) or not value:
        raise ComparatorCandidateError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ComparatorCandidateError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ComparatorCandidateError(f"{field} must include an offset")
    if require_shanghai_offset and parsed.utcoffset() != timedelta(hours=8):
        raise ComparatorCandidateError(f"{field} must use the Asia/Shanghai +08:00 offset")
    return parsed


def _validate_registry_schema(registry: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(dict(schema))
        validator = Draft202012Validator(dict(schema), format_checker=FormatChecker())
        errors = sorted(
            validator.iter_errors(dict(registry)),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        raise ComparatorCandidateError("Generation-2 JSON Schema validation failed") from exc
    if errors:
        location = "/".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise ComparatorCandidateError(
            f"Generation-2 registry violates its schema at {location}: {errors[0].message}"
        )


def _datetime_from_unix_ns(value: int) -> datetime:
    """Convert an integer Unix nanosecond value without float rounding."""

    seconds, nanoseconds = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanoseconds // 1_000
    )


def _mtime_text_from_ns(value: int) -> str:
    return _datetime_from_unix_ns(value).isoformat(timespec="microseconds")


def _unix_ns_from_datetime(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = value.astimezone(timezone.utc) - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _reject_future_formal_time(value: datetime, *, field: str) -> None:
    if value.astimezone(timezone.utc) > (datetime.now(timezone.utc) + _MAX_FORMAL_TIME_FUTURE_SKEW):
        raise ComparatorCandidateError(f"{field} is implausibly in the future")


def _select(value: object, selector: str, *, field: str) -> object:
    current = value
    for component in selector.split("."):
        if not component or not isinstance(current, Mapping) or component not in current:
            raise ComparatorCandidateError(f"{field} selector does not resolve: {selector}")
        current = current[component]
    return current


def _file_binding(
    path: str, *, mode: str | None = None, canonical: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {"path": path, "file_sha256": EXPECTED_FILES[path]}
    if canonical is not None:
        value["canonical_sha256"] = canonical
    if mode is not None:
        value["file_mode"] = mode
    return value


def _load_and_verify_inputs(
    repository_root: Path,
    *,
    enforce_local_freeze_modes: bool = False,
) -> tuple[dict[str, Artifact], dict[str, dict[str, Any]]]:
    artifacts: dict[str, Artifact] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for relative_path, expected_sha256 in EXPECTED_FILES.items():
        artifact = _read_repository_artifact(repository_root, relative_path)
        if artifact.file_sha256 != expected_sha256:
            raise ComparatorCandidateError(f"artifact byte hash changed: {relative_path}")
        artifacts[relative_path] = artifact
        if relative_path.endswith(".json"):
            parsed[relative_path] = _parse_json_bytes(artifact.content, field=relative_path)
        elif relative_path.endswith((".yaml", ".yml")):
            parsed[relative_path] = _parse_yaml_bytes(artifact.content, field=relative_path)
    builder = _read_repository_artifact(repository_root, BUILDER_RELATIVE_PATH)
    artifacts[BUILDER_RELATIVE_PATH] = builder

    if enforce_local_freeze_modes:
        if artifacts[SCHEMA_RELATIVE_PATH].mode != 0o444:
            raise ComparatorCandidateError("Generation-2 schema must be mode 0444")
        for historical in (
            "configs/evaluation/comparator_registry_v1.schema.json",
            "configs/evaluation/comparator_registry_v1.yaml",
            "configs/evaluation/comparator_registry_v1_withdrawal.yaml",
        ):
            if artifacts[historical].mode != 0o600:
                raise ComparatorCandidateError("historical Generation-1 bytes or mode changed")
        for immutable in (
            "reports/baselines/openmed_generation2_repair_v2_results.json",
            "reports/evaluation_readiness/evaluation_readiness_post_repair_v5_blocked.json",
            "configs/evaluation/openmed_generation2_closed8_repair_umbrella_v2.freeze.json",
            "configs/evaluation/openmed_snowflake_repair_gate_v3.freeze.json",
            "configs/evaluation/openmed_qwenmed_repair_gate_v2.freeze.json",
            "configs/evaluation/openmed_bigmed_repair_gate_v2.freeze.json",
            "configs/evaluation/openmed_privacy_filter_repair_gate_v3.freeze.json",
            "configs/evaluation/openmed_eval_source_closure_v5.json",
            "configs/evaluation/openmed_qwenmed_eval_source_closure_v2.json",
            "configs/evaluation/openmed_bigmed_eval_source_closure_v2.json",
            "configs/evaluation/openmed_privacy_filter_eval_source_closure_v2.json",
        ):
            if artifacts[immutable].mode != 0o444:
                raise ComparatorCandidateError(
                    f"immutable bound artifact mode changed: {immutable}"
                )
    return artifacts, parsed


def _verify_predecessor(parsed: Mapping[str, dict[str, Any]]) -> None:
    registry = parsed["configs/evaluation/comparator_registry_v1.yaml"]
    withdrawal = parsed["configs/evaluation/comparator_registry_v1_withdrawal.yaml"]
    unsigned_registry = dict(registry)
    claimed_registry = unsigned_registry.pop("registry_sha256", None)
    if (
        registry.get("registry_id") != "comparator_registry_v1_generation_1"
        or claimed_registry != V1_REGISTRY_CANONICAL_SHA256
        or canonical_json_hash(unsigned_registry) != V1_REGISTRY_CANONICAL_SHA256
    ):
        raise ComparatorCandidateError("Generation-1 registry identity or self-hash changed")
    unsigned_withdrawal = dict(withdrawal)
    claimed_withdrawal = unsigned_withdrawal.pop("withdrawal_sha256", None)
    decision = withdrawal.get("decision")
    if (
        withdrawal.get("withdrawal_id")
        != "comparator_registry_v1_generation_1_freeze_time_withdrawal"
        or claimed_withdrawal != V1_WITHDRAWAL_CANONICAL_SHA256
        or canonical_json_hash(unsigned_withdrawal) != V1_WITHDRAWAL_CANONICAL_SHA256
        or not isinstance(decision, Mapping)
        or decision.get("status") != "withdrawn"
        or decision.get("generation_1_claims_eligible") is not False
        or decision.get("superseded_by") != "comparator_registry_v2_generation_2_pending"
    ):
        raise ComparatorCandidateError("Generation-1 withdrawal identity or policy changed")


def _verify_benchmark_contract(report: Mapping[str, Any], *, report_id: str) -> None:
    contract = report.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise ComparatorCandidateError(f"{report_id} has no comparison contract")
    if contract.get("benchmark_protocol") != "pii_bench_zh_closed_8_v1":
        raise ComparatorCandidateError(f"{report_id} protocol changed")
    benchmark = report.get("benchmark")
    if isinstance(benchmark, Mapping):
        if (
            benchmark.get("dataset_id") != BENCHMARK["repository"]
            or benchmark.get("dataset_revision") != BENCHMARK["revision"]
            or benchmark.get("dataset_manifest_sha256") != BENCHMARK["manifest_logical_sha256"]
        ):
            raise ComparatorCandidateError(f"{report_id} benchmark binding changed")
    elif report.get("dataset_manifest_sha256") != BENCHMARK["manifest_logical_sha256"]:
        raise ComparatorCandidateError(f"{report_id} benchmark manifest changed")
    privacy = report.get("privacy")
    if not isinstance(privacy, Mapping) or any(item is True for item in privacy.values()):
        raise ComparatorCandidateError(f"{report_id} privacy attestation failed")


def _verify_legacy_reports(parsed: Mapping[str, dict[str, Any]]) -> None:
    paths = {spec["report_path"] for spec in LEGACY_TRACK_SPECS.values()}
    for path in sorted(paths):
        report = parsed[path]
        _verify_benchmark_contract(report, report_id=path)
    for variant_id, spec in LEGACY_TRACK_SPECS.items():
        report = parsed[spec["report_path"]]
        selected = _select(report, spec["selector"], field=variant_id)
        if not isinstance(selected, Mapping) or not selected:
            raise ComparatorCandidateError(f"{variant_id} public result selector is empty")
        mapping = _select(report, spec["mapping_selector"], field=f"{variant_id}.mapping")
        if (
            not isinstance(mapping, (Mapping, Sequence))
            or isinstance(mapping, (str, bytes))
            or not mapping
        ):
            raise ComparatorCandidateError(f"{variant_id} mapping selector is empty")
        if canonical_json_hash(mapping) != spec["mapping_sha256"]:
            raise ComparatorCandidateError(f"{variant_id} mapping payload changed")


def _verify_openmed(parsed: Mapping[str, dict[str, Any]]) -> None:
    path = "reports/baselines/openmed_generation2_repair_v2_results.json"
    results = parsed[path]
    unsigned = dict(results)
    claimed = unsigned.pop("results_receipt_sha256", None)
    if (
        results.get("receipt_id") != "openmed_generation2_repair_v2_aggregate_results"
        or claimed != OPENMED_RESULTS_CANONICAL_SHA256
        or canonical_json_hash(unsigned) != OPENMED_RESULTS_CANONICAL_SHA256
        or results.get("claim_eligible") is not False
        or results.get("first_execution_claim") is not False
        or results.get("blind_evaluation_claim") is not False
    ):
        raise ComparatorCandidateError("OpenMed strict result receipt identity changed")
    claims = results.get("claim_policy")
    if not isinstance(claims, Mapping) or any(
        claims.get(field) is not False
        for field in (
            "best_chinese_pii_model_claim_allowed",
            "first_or_blind_run_claim_allowed",
            "global_best_claim_allowed",
            "sota_claim_allowed",
        )
    ):
        raise ComparatorCandidateError("OpenMed claim policy changed")
    for variant_id, spec in OPENMED_TRACK_SPECS.items():
        selected = _select(results, spec["selector"], field=variant_id)
        if not isinstance(selected, Mapping) or set(selected) != {"formal", "chat", "pooled"}:
            raise ComparatorCandidateError(f"{variant_id} strict result selector is incomplete")
        for suite in ("formal", "chat", "pooled"):
            cell = selected[suite]
            if not isinstance(cell, Mapping) or not isinstance(cell.get("strict_micro"), Mapping):
                raise ComparatorCandidateError(f"{variant_id}.{suite} metrics are missing")
    for member in OPENMED_MEMBERS.values():
        prereg = parsed[member["mapping_path"]]
        mapping = _select(prereg, member["mapping_selector"], field=member["system_id"])
        if (
            not isinstance(mapping, Mapping)
            or canonical_json_hash(mapping) != member["mapping_sha256"]
        ):
            raise ComparatorCandidateError(f"{member['system_id']} native mapping changed")


def _license(
    *,
    status: str,
    declarations: Sequence[str],
    redistribute: bool,
    evidence: str,
    evaluation_allowed: bool = True,
) -> dict[str, Any]:
    return {
        "status": status,
        "declarations": list(declarations),
        "evaluation_allowed": evaluation_allowed,
        "redistribution_allowed_by_project": redistribute,
        "release_bundle_included": False,
        "evidence": evidence,
    }


def _systems() -> dict[str, dict[str, Any]]:
    seen = "fixed public synthetic benchmark already seen; descriptive only"
    return {
        "aiguard_qwen3_0_6b": {
            "display_name": "AIguard PII Detection Fast Qwen3-0.6B",
            "resource_kind": "public_model",
            "repository": "ZJUICSR/AIguard-pii-detection-fast",
            "revision": "677a5ebc1600fef61e8973cafd3026be322b3a73",
            "revision_status": "pinned_model_revision",
            "language_scope": ["zh"],
            "artifact_binding_status": "pinned_revision_plus_aggregate_report",
            "license": _license(
                status="upstream_declared",
                declarations=["Apache-2.0"],
                redistribute=True,
                evidence="Generation-1 registry and independently reproduced aggregate report",
            ),
            "benchmark_exposure": seen,
            "caveats": ["Aggregate report does not embed its historical evaluator byte hash."],
        },
        "openai_privacy_filter": {
            "display_name": "OpenAI Privacy Filter",
            "resource_kind": "public_model",
            "repository": "openai/privacy-filter",
            "revision": "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
            "revision_status": "pinned_model_revision_plus_decoder_source_f7f00ca7fb869683eb732c010299d901457f19c3",
            "language_scope": ["en_primary", "zh_transfer"],
            "artifact_binding_status": "pinned_revision_decoder_and_aggregate_report",
            "license": _license(
                status="upstream_declared",
                declarations=["Apache-2.0"],
                redistribute=True,
                evidence="Pinned model and official decoder source revisions",
            ),
            "benchmark_exposure": seen,
            "caveats": [
                "English-primary transfer baseline; three closed-eight labels are unsupported."
            ],
        },
        "pii_engineer_chinese_ner_v1_0": {
            "display_name": "PII Engineer Chinese NER v1.0",
            "resource_kind": "public_model",
            "repository": "pii-engineer/PII-Engineer-Chinese-NER-v1.0",
            "revision": "d16aa1c54f34f993b2d4c30c218ee9d712959b2d",
            "revision_status": "pinned_model_revision",
            "language_scope": ["zh"],
            "artifact_binding_status": "pinned_revision_plus_aggregate_report_current_runner_drift",
            "license": _license(
                status="conflicting_upstream_declarations",
                declarations=["Apache-2.0-frontmatter", "AGPL-3.0-body"],
                redistribute=False,
                evidence="Pinned model card contains conflicting declarations",
            ),
            "benchmark_exposure": seen,
            "caveats": [
                "Current replay runner hash differs from the historical runner hash embedded in the report.",
                "Weight redistribution is disabled by this project.",
            ],
        },
        "pii_engineer_multi_ner_v2_1": {
            "display_name": "PII Engineer Multi NER v2.1",
            "resource_kind": "public_model_required_pending_closure",
            "repository": "pii-engineer/PII-Engineer-Multi-NER-v2.1",
            "revision": "8ce12a2da5c333483ca2bd9a85def9d60609f244",
            "revision_status": "external_public_metadata_observed_pending_repo_local_closure",
            "language_scope": ["multilingual", "zh"],
            "artifact_binding_status": "pending_local_snapshot_mapping_runner_and_same_protocol_report",
            "license": _license(
                status="conflicting_upstream_declarations_pending_review",
                declarations=["Apache-2.0-frontmatter", "AGPL-3.0-body"],
                redistribute=False,
                evidence="External public metadata observation only; no repository-local snapshot yet",
                evaluation_allowed=False,
            ),
            "benchmark_exposure": "not_run_under_repository_protocol",
            "caveats": [
                "Required candidate discovered before freeze but not locally auditable yet.",
                "No report, mapping, runner, artifact manifest, or local license snapshot is admitted.",
            ],
        },
        "ernie_chinese_ner": {
            "display_name": "ERNIE 3.0 Chinese NER",
            "resource_kind": "public_model_with_framework_adapter",
            "repository": "gyr66/Ernie-3.0-base-chinese-finetuned-ner",
            "revision": "3f3c3fa40cf87298586dbbed55fd7ca1988719a8",
            "revision_status": "pinned_model_revision",
            "language_scope": ["zh"],
            "artifact_binding_status": "pinned_revision_plus_aggregate_report_current_runner_drift",
            "license": _license(
                status="not_declared_in_pinned_model_card",
                declarations=[],
                redistribute=False,
                evidence="Pinned model card has no declared model license",
            ),
            "benchmark_exposure": seen,
            "caveats": [
                "Current replay runner hash differs from the historical runner hash embedded in the report.",
                "Historical framework track uses custom fusion, not native AnalyzerEngine semantics.",
            ],
        },
        "cluener2020_roberta": {
            "display_name": "RoBERTa CLUENER2020 Chinese NER",
            "resource_kind": "public_model_with_framework_adapter",
            "repository": "uer/roberta-base-finetuned-cluener2020-chinese",
            "revision": "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b",
            "revision_status": "pinned_model_revision",
            "language_scope": ["zh"],
            "artifact_binding_status": "pinned_revision_plus_aggregate_report",
            "license": _license(
                status="not_declared_in_pinned_model_card",
                declarations=[],
                redistribute=False,
                evidence="Pinned model card and upstream data redistribution remain unresolved",
            ),
            "benchmark_exposure": seen,
            "caveats": [
                "Historical framework track uses custom fusion, not native AnalyzerEngine semantics.",
                "Only ADDRESS and PERSON_NAME are directly mapped in model_raw.",
            ],
        },
        "cn_common_v5": {
            "display_name": "pii_zh cn_common v5 rules",
            "resource_kind": "project_open_source_rule_pack",
            "repository": "local:pii_zh.rules.cn_common",
            "revision": "sha256:41e06d41dcc8eabe59584cd27b64ecfd234bcb034dc6d2268d87095c1a8a8bbb",
            "revision_status": "implementation_content_sha256",
            "language_scope": ["zh"],
            "artifact_binding_status": "source_and_aggregate_report_bound",
            "license": _license(
                status="project_declared",
                declarations=["Apache-2.0"],
                redistribute=True,
                evidence="Project source and report component hashes",
            ),
            "benchmark_exposure": seen,
            "caveats": ["ADDRESS and PERSON_NAME are unsupported and count as false negatives."],
        },
        "argus_redact_fast": {
            "display_name": "Argus Redact fast rules",
            "resource_kind": "open_source_framework_rules",
            "repository": "wan9yu/argus-redact",
            "revision": "88d0342012b43e6c71fe79a02b90a1432db0f9a9",
            "revision_status": "source_revision_not_cryptographically_bound_to_wheel_0_7_18",
            "language_scope": ["zh"],
            "artifact_binding_status": "package_version_report_and_declared_source_revision",
            "license": _license(
                status="upstream_declared",
                declarations=["Apache-2.0"],
                redistribute=True,
                evidence="Package metadata and declared corresponding source revision",
            ),
            "benchmark_exposure": "development_coupled_synthetic_benchmark_same_author",
            "caveats": [
                "Framework repository and benchmark share an author and development artifacts.",
                "Wheel-to-source revision is not cryptographically proven.",
            ],
        },
        "openmed_snowflake": _openmed_system(
            "OpenMed SnowflakeMed Large 568M",
            "OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1",
            "a7f9f4bbc43961ce1a7cdc38ade5545fcfcfbe64",
            "a5f795813b6bbbcf85e82a6cfda80b396aa9300132a64a0455e5a9c10bf0cc27",
        ),
        "openmed_qwenmed": _openmed_system(
            "OpenMed QwenMed XLarge 600M",
            "OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1",
            "b230cddef42b3ed6ee2dc24fe065a130fdac6653",
            "ce3d9310f2327558855388fb36c9f3d901f54f642f20c5263e1cd6d7540a7d68",
        ),
        "openmed_bigmed": _openmed_system(
            "OpenMed BigMed Large 560M",
            "OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1",
            "5cf65bca4760d6efbea015a65b9cf9116aa3519c",
            "246578f95cd936065d036701493092e17c1e0c101388370822520f70c5be4619",
        ),
        "openmed_multilingual_privacy_filter": _openmed_system(
            "OpenMed Privacy Filter Multilingual v2",
            "OpenMed/privacy-filter-multilingual-v2",
            "0d0c0430fa386435ace1d9f842f0f966903a9ac8",
            "8757300f46dfc6f02dc8a570541d83739227e0e37b79b7016fdffb8198ee3d3c",
            license_status="upstream_card_observed_project_prereg_did_not_bind_license",
        ),
    }


def _openmed_system(
    display_name: str,
    repository: str,
    revision: str,
    readme_sha256: str,
    *,
    license_status: str = "upstream_declared_project_redistribution_disabled",
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "resource_kind": "public_model",
        "repository": repository,
        "revision": revision,
        "revision_status": "pinned_model_revision_gate_and_source_closure_bound",
        "language_scope": ["zh"],
        "artifact_binding_status": "pinned_revision_prereg_gate_source_closure_and_strict_receipt",
        "license": _license(
            status=license_status,
            declarations=["Apache-2.0-upstream-card"],
            redistribute=False,
            evidence=f"Pinned upstream README SHA-256 {readme_sha256}; weights excluded from project bundle",
        ),
        "benchmark_exposure": "post_freeze_repair_public_synthetic_benchmark_descriptive_only",
        "caveats": [
            "Public source aggregate report is represented by its hash inside the consolidated receipt.",
            "Project does not redistribute comparator weights.",
        ],
    }


LEGACY_TRACK_SPECS: dict[str, dict[str, Any]] = {
    "aiguard_qwen3_0_6b_model_raw": {
        "system_id": "aiguard_qwen3_0_6b",
        "canonical_track": "model_raw",
        "legacy_report_track": "model_only",
        "components": ["pinned_checkpoint", "native_token_classifier_decoder"],
        "supported": list(CLOSED_8_LABELS),
        "report_path": "reports/baselines/aiguard_pii_bench_zh.json",
        "selector": "results",
        "mapping_selector": "comparison_contract.closed_8.source_to_target",
        "mapping_sha256": "d9cd9a547a79833209e7b8ee1769870ddbee0fa32876e150e08f0f65a49c1d4a",
        "runner_path": "scripts/evaluate_aiguard_baseline.py",
        "runner_status": "bound_current_source",
        "decoder": "masked_logits_then_argmax_and_BIO_decode",
        "operating_point": "upstream_default_no_test_threshold_tuning",
    },
    "openai_privacy_filter_model_raw": {
        "system_id": "openai_privacy_filter",
        "canonical_track": "model_raw",
        "legacy_report_track": "model_only",
        "components": ["pinned_checkpoint", "official_constrained_viterbi"],
        "supported": [
            "ADDRESS",
            "BANK_CARD_NUMBER",
            "EMAIL_ADDRESS",
            "PERSON_NAME",
            "PHONE_NUMBER",
        ],
        "report_path": "reports/baselines/openai_privacy_filter_pii_bench_zh.json",
        "selector": "results",
        "mapping_selector": "comparison_contract.closed_8.source_to_target",
        "mapping_sha256": "d1774737261163a1b22eff3457bbe606320d66c767ad477db1e3d829b00d05eb",
        "runner_path": "scripts/evaluate_openai_privacy_filter_baseline.py",
        "runner_status": "bound_exact_historical_source",
        "decoder": "official_constrained_viterbi",
        "operating_point": "upstream_default",
    },
    "pii_engineer_chinese_ner_model_raw": {
        "system_id": "pii_engineer_chinese_ner_v1_0",
        "canonical_track": "model_raw",
        "legacy_report_track": "model_only",
        "components": ["pinned_ONNX_BERT_BIO_checkpoint", "BIO_repair_decoder"],
        "supported": ["ADDRESS", "CN_RESIDENT_ID", "PERSON_NAME", "PHONE_NUMBER"],
        "report_path": "reports/baselines/pii_engineer_chinese_ner_pii_bench_zh.json",
        "selector": "results",
        "mapping_selector": "comparison_contract.closed_8.source_to_target",
        "mapping_sha256": "c9e9e462d372b1f541fa192637cb7d366ff60e3c29e5b1b78809f4e0d6f973d6",
        "runner_path": "scripts/evaluate_pii_engineer_baseline.py",
        "runner_status": "bound_current_source",
        "decoder": "masked_logits_argmax_BIO_repair",
        "operating_point": "upstream_default_no_test_threshold_tuning",
    },
    "ernie_chinese_ner_model_raw": {
        "system_id": "ernie_chinese_ner",
        "canonical_track": "model_raw",
        "legacy_report_track": "model_only",
        "components": ["pinned_ERNIE_BIO_checkpoint", "LocalRecognizer_adapter"],
        "supported": ["ADDRESS", "EMAIL_ADDRESS", "PERSON_NAME", "PHONE_NUMBER"],
        "report_path": "reports/baselines/presidio_ernie_pii_bench_zh.json",
        "selector": "results.model_only",
        "mapping_selector": "comparison_contract.closed_8.model_source_to_target",
        "mapping_sha256": "bc717cf58f066fb19ab63658f3dfba4016665fd589165cbcf1eb9d397d982cf4",
        "runner_path": "scripts/evaluate_presidio_ernie_baseline.py",
        "runner_status": "bound_current_source",
        "decoder": "masked_logits_argmax_BIO_repair",
        "operating_point": "upstream_default_no_test_threshold_tuning",
    },
    "presidio_ernie_cn_common_framework_hybrid": {
        "system_id": "ernie_chinese_ner",
        "canonical_track": "framework_hybrid",
        "legacy_report_track": "framework_hybrid",
        "components": ["ERNIE_LocalRecognizer", "cn_common_v5", "custom_historical_fusion"],
        "supported": list(CLOSED_8_LABELS),
        "report_path": "reports/baselines/presidio_ernie_pii_bench_zh.json",
        "selector": "results.framework_hybrid",
        "mapping_selector": "comparison_contract.closed_8.model_source_to_target",
        "mapping_sha256": "bc717cf58f066fb19ab63658f3dfba4016665fd589165cbcf1eb9d397d982cf4",
        "runner_path": "scripts/evaluate_presidio_ernie_baseline.py",
        "runner_status": "bound_current_source",
        "decoder": "model_BIO_decode_plus_custom_exact_dedup_overlap_fusion",
        "operating_point": "historical_custom_fusion_not_native_AnalyzerEngine",
    },
    "cluener2020_roberta_model_raw": {
        "system_id": "cluener2020_roberta",
        "canonical_track": "model_raw",
        "legacy_report_track": "model_only",
        "components": ["pinned_CLUENER_BIOS_checkpoint", "LocalRecognizer_adapter"],
        "supported": ["ADDRESS", "PERSON_NAME"],
        "report_path": "reports/baselines/presidio_cluener_pii_bench_zh.json",
        "selector": "results.model_only",
        "mapping_selector": "comparison_contract.closed_8.model_source_to_target",
        "mapping_sha256": "d3da91edb1b064d0efe8519f5229c2046ad472addd3592d8e73bbde2d827e740",
        "runner_path": "scripts/evaluate_presidio_cluener_baseline.py",
        "runner_status": "bound_exact_historical_source",
        "decoder": "masked_logits_argmax_BIOS_repair",
        "operating_point": "upstream_default_no_test_threshold_tuning",
    },
    "presidio_cluener_cn_common_framework_hybrid": {
        "system_id": "cluener2020_roberta",
        "canonical_track": "framework_hybrid",
        "legacy_report_track": "framework_hybrid",
        "components": ["CLUENER_LocalRecognizer", "cn_common_v5", "custom_historical_fusion"],
        "supported": list(CLOSED_8_LABELS),
        "report_path": "reports/baselines/presidio_cluener_pii_bench_zh.json",
        "selector": "results.framework_hybrid",
        "mapping_selector": "comparison_contract.closed_8.model_source_to_target",
        "mapping_sha256": "d3da91edb1b064d0efe8519f5229c2046ad472addd3592d8e73bbde2d827e740",
        "runner_path": "scripts/evaluate_presidio_cluener_baseline.py",
        "runner_status": "bound_exact_historical_source",
        "decoder": "model_BIOS_decode_plus_custom_exact_dedup_overlap_fusion",
        "operating_point": "historical_custom_fusion_not_native_AnalyzerEngine",
    },
    "cn_common_v5_rules_only": {
        "system_id": "cn_common_v5",
        "canonical_track": "rules_only",
        "legacy_report_track": "rules_only",
        "components": ["cn_common_v5", "structured_validators"],
        "supported": [
            "BANK_CARD_NUMBER",
            "CN_RESIDENT_ID",
            "EMAIL_ADDRESS",
            "PASSPORT_NUMBER",
            "PHONE_NUMBER",
            "VEHICLE_LICENSE_PLATE",
        ],
        "report_path": "reports/baselines/presidio_rules_closed8_pii_bench_zh.json",
        "selector": "subsets",
        "mapping_selector": "comparison_contract.supported_target_labels",
        "mapping_sha256": "4e776a14578b21e122112b4563be8a5fa6eafe15f1bf9b9df95fce7efea778a4",
        "runner_path": "scripts/predict_rules.py",
        "runner_status": "bound_exact_historical_source",
        "decoder": "deterministic_rules_and_validators",
        "operating_point": "cn_common_v5_fixed",
        "mapping_identity_scope": "native_supported_label_contract",
    },
    "argus_redact_fast_rules_only": {
        "system_id": "argus_redact_fast",
        "canonical_track": "rules_only",
        "legacy_report_track": "rules_only",
        "components": ["argus_redact_0_7_18_fast_mode"],
        "supported": list(CLOSED_8_LABELS),
        "report_path": "reports/baselines/argus_redact_pii_bench_zh.json",
        "selector": "results",
        "mapping_selector": "comparison_contract.closed_8.source_to_target",
        "mapping_sha256": "a56547675b633ecba282fdfff551933b53955746addd9239106f8863f2428d9f",
        "runner_path": "scripts/evaluate_argus_redact_baseline.py",
        "runner_status": "bound_exact_historical_source",
        "decoder": "argus_fast_rules_type_filter",
        "operating_point": "package_default_fast_mode",
    },
}


OPENMED_MEMBERS: dict[str, dict[str, Any]] = {
    "snowflake": {
        "system_id": "openmed_snowflake",
        "runner_path": "scripts/evaluate_openmed_snowflake_repair_v2.py",
        "mapping_path": "configs/evaluation/openmed_snowflake_closed8_model_raw_hybrid_v6.yaml",
        "mapping_selector": "source_to_benchmark",
        "mapping_sha256": "e54c629ee369c16a61e83676aa0df242a90914410a6920cd4de8bec4312246b5",
    },
    "qwenmed": {
        "system_id": "openmed_qwenmed",
        "runner_path": "scripts/evaluate_openmed_qwenmed_baseline_v2.py",
        "mapping_path": "configs/evaluation/openmed_qwenmed_closed8_model_raw_hybrid_v1.yaml",
        "mapping_selector": "source_to_benchmark",
        "mapping_sha256": "e54c629ee369c16a61e83676aa0df242a90914410a6920cd4de8bec4312246b5",
    },
    "bigmed": {
        "system_id": "openmed_bigmed",
        "runner_path": "scripts/evaluate_openmed_bigmed_baseline_v2.py",
        "mapping_path": "configs/evaluation/openmed_bigmed_closed8_model_raw_hybrid_v1.json",
        "mapping_selector": "source_to_benchmark",
        "mapping_sha256": "e54c629ee369c16a61e83676aa0df242a90914410a6920cd4de8bec4312246b5",
    },
    "multilingual_privacy_filter": {
        "system_id": "openmed_multilingual_privacy_filter",
        "runner_path": "scripts/evaluate_openmed_privacy_filter_gen2_repair_v2.py",
        "mapping_path": "configs/evaluation/openmed_privacy_filter_gen2_v1.final.yaml",
        "mapping_selector": "mapping.source_to_benchmark",
        "mapping_sha256": "e54c629ee369c16a61e83676aa0df242a90914410a6920cd4de8bec4312246b5",
    },
}


OPENMED_TRACK_SPECS: dict[str, dict[str, Any]] = {}
for _member_id, _member in OPENMED_MEMBERS.items():
    for _track in ("model_raw", "framework_hybrid"):
        OPENMED_TRACK_SPECS[f"openmed_{_member_id}_{_track}"] = {
            **_member,
            "member_id": _member_id,
            "track": _track,
            "selector": f"results.{_member_id}.tracks.{_track}",
        }


def _legacy_variant(spec: Mapping[str, Any]) -> dict[str, Any]:
    supported = list(spec["supported"])
    unsupported = [label for label in CLOSED_8_LABELS if label not in supported]
    return {
        "system_id": spec["system_id"],
        "canonical_track": spec["canonical_track"],
        "legacy_report_track": spec["legacy_report_track"],
        "components": list(spec["components"]),
        "supported_closed_8_labels": supported,
        "unsupported_closed_8_labels": unsupported,
        "closed_8_coverage": len(supported),
        "mapping": {
            "status": "embedded_in_bound_report",
            "identity_scope": spec.get(
                "mapping_identity_scope", "native_source_to_closed_8_mapping"
            ),
            "native_source_mapping_status": "embedded_in_bound_report",
            "path": spec["report_path"],
            "file_sha256": EXPECTED_FILES[spec["report_path"]],
            "selector": spec["mapping_selector"],
            "canonical_sha256": spec["mapping_sha256"],
        },
        "decoder": spec["decoder"],
        "operating_point": spec["operating_point"],
        "benchmark_threshold_tuning": False,
        "runner": {
            "status": spec["runner_status"],
            "path": spec["runner_path"],
            "file_sha256": EXPECTED_FILES[spec["runner_path"]],
        },
        "public_diagnostic": {
            "status": "available_descriptive_only",
            "path": spec["report_path"],
            "file_sha256": EXPECTED_FILES[spec["report_path"]],
            "selector": spec["selector"],
            "claim_eligible": False,
        },
        "human_hidden_execution_status": "pending_not_executed",
        "candidate": False,
    }


def _openmed_variant(spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = spec["track"] == "model_raw"
    supported = [label for label in CLOSED_8_LABELS if not raw or label != "PASSPORT_NUMBER"]
    return {
        "system_id": spec["system_id"],
        "canonical_track": spec["track"],
        "legacy_report_track": None,
        "components": (
            ["pinned_OpenMed_checkpoint", "shared_sparse_BIO_runtime"]
            if raw
            else [
                "pinned_OpenMed_checkpoint",
                "shared_sparse_BIO_runtime",
                "presidio_analyzer_2_2_363",
                "cn_common_v5",
                "native_AnalyzerEngine",
            ]
        ),
        "supported_closed_8_labels": supported,
        "unsupported_closed_8_labels": [
            label for label in CLOSED_8_LABELS if label not in supported
        ],
        "closed_8_coverage": len(supported),
        "mapping": {
            "status": "bound_via_frozen_source_closure",
            "identity_scope": "native_source_to_closed_8_mapping",
            "native_source_mapping_status": "source_closure_bound_no_standalone_declared_hash",
            "path": spec["mapping_path"],
            "file_sha256": EXPECTED_FILES[spec["mapping_path"]],
            "selector": spec["mapping_selector"],
            "canonical_sha256": spec["mapping_sha256"],
        },
        "decoder": "sparse_BIO_exact_offset_merge_and_deterministic_repair_v2",
        "operating_point": (
            "fixed_native_model_raw" if raw else "native_Presidio_AnalyzerEngine_2_2_363_fixed"
        ),
        "benchmark_threshold_tuning": False,
        "runner": {
            "status": "bound_exact_historical_source",
            "path": spec["runner_path"],
            "file_sha256": EXPECTED_FILES[spec["runner_path"]],
        },
        "public_diagnostic": {
            "status": "available_descriptive_only",
            "path": "reports/baselines/openmed_generation2_repair_v2_results.json",
            "file_sha256": EXPECTED_FILES[
                "reports/baselines/openmed_generation2_repair_v2_results.json"
            ],
            "selector": spec["selector"],
            "claim_eligible": False,
        },
        "human_hidden_execution_status": "pending_not_executed",
        "candidate": False,
    }


def _pending_multi_variant() -> dict[str, Any]:
    return {
        "system_id": "pii_engineer_multi_ner_v2_1",
        "canonical_track": "model_raw",
        "legacy_report_track": None,
        "components": ["upstream_multilingual_ONNX_modules_pending_local_closure"],
        "supported_closed_8_labels": [],
        "unsupported_closed_8_labels": list(CLOSED_8_LABELS),
        "closed_8_coverage": 0,
        "mapping": {
            "status": "pending",
            "identity_scope": "pending",
            "native_source_mapping_status": "pending",
            "path": None,
            "file_sha256": None,
            "selector": None,
            "canonical_sha256": None,
        },
        "decoder": "pending_same_protocol_adapter_and_decoder_closure",
        "operating_point": "pending_upstream_default_freeze",
        "benchmark_threshold_tuning": False,
        "runner": {"status": "pending", "path": None, "file_sha256": None},
        "public_diagnostic": {
            "status": "pending_same_protocol_reproduction",
            "path": None,
            "file_sha256": None,
            "selector": None,
            "claim_eligible": False,
        },
        "human_hidden_execution_status": "pending_not_executed",
        "candidate": False,
    }


def _track_variants() -> dict[str, dict[str, Any]]:
    variants = {
        variant_id: _legacy_variant(spec) for variant_id, spec in LEGACY_TRACK_SPECS.items()
    }
    variants.update(
        {variant_id: _openmed_variant(spec) for variant_id, spec in OPENMED_TRACK_SPECS.items()}
    )
    variants["pii_engineer_multi_ner_v2_1_model_raw_pending"] = _pending_multi_variant()
    return dict(sorted(variants.items()))


def _search_resource(
    repository: str,
    revision: str,
    language_scope: Sequence[str],
    decision: str,
    reason: str,
    observation_sha256: str,
) -> dict[str, Any]:
    return {
        "repository": repository,
        "revision": revision,
        "language_scope": list(language_scope),
        "decision": decision,
        "reason": reason,
        "observation_sha256": observation_sha256,
    }


def _search_inventory() -> dict[str, Any]:
    return {
        "method": "public_hugging_face_metadata_plus_existing_repository_evidence",
        "network_snapshot_archived": False,
        "observation_digest_scope": (
            "SHA-256 of public API response bytes observed during the pre-freeze search; "
            "responses are not repository-archived evidence and cannot activate claims"
        ),
        "required_system_ids": list(ALL_SYSTEM_IDS),
        "newly_discovered_required_system_ids": ["pii_engineer_multi_ner_v2_1"],
        "observed_public_resources": {
            "pii_engineer_multi_ner_v2_1": _search_resource(
                "pii-engineer/PII-Engineer-Multi-NER-v2.1",
                "8ce12a2da5c333483ca2bd9a85def9d60609f244",
                ["multilingual", "zh"],
                "required_candidate",
                "Chinese-capable recent public PII NER model; local closure remains pending",
                "a1e1ffc8915c2c86ce8e01285c06f7f24855502b74fe7edcccf9cb1e53e660d0",
            ),
            "pii_engineer_chinese_ner_v1_0": _search_resource(
                "pii-engineer/PII-Engineer-Chinese-NER-v1.0",
                "d16aa1c54f34f993b2d4c30c218ee9d712959b2d",
                ["zh"],
                "required_candidate",
                "Existing reproduced Chinese-specific comparator",
                "48d3b812f70710367dffed307cb04dd9dc06b4bb3ceefe2952a4ee2866f72d61",
            ),
            "openmed_snowflake": _search_resource(
                "OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1",
                "a7f9f4bbc43961ce1a7cdc38ade5545fcfcfbe64",
                ["zh"],
                "required_candidate",
                "Existing Generation-2 reproduced comparator",
                "de4b24c3307704a010e70fb0168d2ff5e48244be1b8d63b53fb5aef91ba48fa3",
            ),
            "openmed_qwenmed": _search_resource(
                "OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1",
                "b230cddef42b3ed6ee2dc24fe065a130fdac6653",
                ["zh"],
                "required_candidate",
                "Existing Generation-2 reproduced comparator",
                "10d3e4e750cc5814d01ace6fce12792b00fb1412f471e70a06326a03ff0af325",
            ),
            "openmed_bigmed": _search_resource(
                "OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1",
                "5cf65bca4760d6efbea015a65b9cf9116aa3519c",
                ["zh"],
                "required_candidate",
                "Existing Generation-2 reproduced comparator",
                "d6d176d7682879e8abe3c2f3de5dc1a8f13c539fb3c447c5303f1e7d0168777d",
            ),
            "openmed_multilingual_privacy_filter": _search_resource(
                "OpenMed/privacy-filter-multilingual-v2",
                "0d0c0430fa386435ace1d9f842f0f966903a9ac8",
                ["multilingual", "zh"],
                "required_candidate",
                "Existing Generation-2 reproduced comparator",
                "6352be61403b30ef901995504ce6c3b337e794de2d5412435c6eb2555229d43a",
            ),
        },
        "excluded_public_resources": {
            "ossredact_large": _search_resource(
                "ZenSystemAI/OSSRedact-PII-Detection-Large",
                "1341549fc6d3fd57ffd87743a89030a6c3829328",
                ["fr", "en"],
                "excluded_language_scope",
                "No Simplified-Chinese language scope",
                "6dd3ec7a77f24be7bc79d84a584c8ac735158fd7b5a237daea60e1281a01dab1",
            ),
            "ar86bat_multi": _search_resource(
                "Ar86Bat/PII-multilingual-ner",
                "25111047e70291ad9e6194547296846784951fb1",
                ["en", "de", "fr", "it"],
                "excluded_language_scope",
                "No Chinese language scope",
                "27e9d5291d7f2dde5fcba279607dcebcd9efcec76083c6cfb29ecb6fccbb9700",
            ),
            "piiranha": _search_resource(
                "iiiorg/piiranha-v1-detect-personal-information",
                "255acde67a2f34cf452eb42e365b24d2957352fc",
                ["en", "it", "fr", "de", "nl", "es"],
                "excluded_language_scope",
                "No Chinese language scope",
                "55f44cc4dede6ea704f71f225551d331f48bc6e79efcad61e71f8e61fcfdbd03",
            ),
            "urchade_gliner": _search_resource(
                "urchade/gliner_multi_pii-v1",
                "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
                ["en", "fr", "de", "es", "pt", "it"],
                "excluded_language_scope",
                "No Chinese language scope",
                "2077ef7142d96d102232016983c1924538727078429e07cb27838f31770fe3cd",
            ),
            "fastino_gliner2": _search_resource(
                "fastino/gliner2-multilingual-pii",
                "e40819b10177c9b9cdea62a3ece5cfdebd921e01",
                ["en", "fr", "es", "de", "it", "pt", "nl"],
                "excluded_language_scope",
                "No Chinese language scope; synthetic training is also disclosed",
                "0ed09b25cde818205078c855c7f6adaf2e8772e8ebc0ba98fa74af5874c8654a",
            ),
            "privacy_filter_tw": _search_resource(
                "lianghsun/privacy-filter-tw",
                "89bc359c7cd130707992495c4347154ba7796e3c",
                ["zh_TW", "en"],
                "deferred_cross_locale",
                "Traditional-Chinese Taiwan 19-label protocol requires a separate cross-locale track",
                "898d5ce9d27ab4b4e581e514b9971208ed4efd4781d3ef231700237cfb33c574",
            ),
        },
    }


def _source_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "release_evaluator": "scripts/evaluate.py",
        "evaluation_api": "src/pii_zh/evaluation/__init__.py",
        "evaluation_io": "src/pii_zh/evaluation/io.py",
        "evaluation_metrics": "src/pii_zh/evaluation/metrics.py",
        "evaluation_provenance": "src/pii_zh/evaluation/provenance.py",
        "required_metrics": "src/pii_zh/evaluation/required_metrics.py",
        "baseline_reporting": "src/pii_zh/evaluation/baseline_reporting.py",
        "taxonomy_api": "src/pii_zh/taxonomy/__init__.py",
        "taxonomy_loader": "src/pii_zh/taxonomy/loader.py",
        "taxonomy": "src/pii_zh/taxonomy/taxonomy.yaml",
        "presidio_mapping": "src/pii_zh/taxonomy/presidio_mapping.yaml",
        "openmed_runtime": "src/pii_zh/evaluation/openmed_generation2_runtime.py",
        "openmed_strict_results": "reports/baselines/openmed_generation2_repair_v2_results.json",
        "strict_readiness_v5": "reports/evaluation_readiness/evaluation_readiness_post_repair_v5_blocked.json",
        "openmed_repair_umbrella": "configs/evaluation/openmed_generation2_closed8_repair_umbrella_v2.freeze.json",
        "openmed_snowflake_gate": "configs/evaluation/openmed_snowflake_repair_gate_v3.freeze.json",
        "openmed_qwenmed_gate": "configs/evaluation/openmed_qwenmed_repair_gate_v2.freeze.json",
        "openmed_bigmed_gate": "configs/evaluation/openmed_bigmed_repair_gate_v2.freeze.json",
        "openmed_privacy_filter_gate": "configs/evaluation/openmed_privacy_filter_repair_gate_v3.freeze.json",
        "openmed_snowflake_closure": "configs/evaluation/openmed_eval_source_closure_v5.json",
        "openmed_qwenmed_closure": "configs/evaluation/openmed_qwenmed_eval_source_closure_v2.json",
        "openmed_bigmed_closure": "configs/evaluation/openmed_bigmed_eval_source_closure_v2.json",
        "openmed_privacy_filter_closure": "configs/evaluation/openmed_privacy_filter_eval_source_closure_v2.json",
    }
    bindings: dict[str, dict[str, Any]] = {}
    for logical_id, path in paths.items():
        mode = "0444" if path.startswith(("configs/evaluation/openmed_", "reports/")) else None
        canonical = None
        if logical_id == "openmed_strict_results":
            canonical = OPENMED_RESULTS_CANONICAL_SHA256
        bindings[logical_id] = _file_binding(path, mode=mode, canonical=canonical)
    return bindings


def _pending_requirements(track_ids: Sequence[str]) -> dict[str, Any]:
    all_tracks = list(track_ids)
    all_systems = list(ALL_SYSTEM_IDS)
    return {
        "pii_engineer_multi_same_protocol_reproduction": {
            "status": "pending",
            "blocking": True,
            "reason": "Recent Chinese-capable model is material but has no repo-local immutable closure or same-protocol result.",
            "required_before": "registry_activation",
            "affected_system_ids": ["pii_engineer_multi_ner_v2_1"],
            "affected_track_variant_ids": ["pii_engineer_multi_ner_v2_1_model_raw_pending"],
            "acceptance_criteria": [
                "Pin artifact files and reconcile the Apache-2.0 versus AGPL-3.0 declarations.",
                "Freeze source mapping, decoder, operating point, runner, and unsupported-FN policy.",
                "Produce a same-protocol aggregate report without using hidden gold for selection.",
            ],
        },
        "open24_and_human_hidden_mapping_closure": {
            "status": "pending",
            "blocking": True,
            "reason": "Existing evidence covers public synthetic closed-8 only; member-level open-24 and human-hidden contracts do not exist.",
            "required_before": "human_hidden_execution",
            "affected_system_ids": all_systems,
            "affected_track_variant_ids": all_tracks,
            "acceptance_criteria": [
                "Freeze every member's open-24 mapping, unsupported-FN handling, pre-decode policy, and suite support.",
                "Materialize, sign off, seal, and keep the human hidden test at zero access before execution.",
                "Generate a successor active registry rather than mutating this blocked freeze.",
            ],
        },
        "three_metric_paired_bootstrap_implementation": {
            "status": "pending",
            "blocking": True,
            "reason": "Current paired bootstrap implements strict micro F1 only and cannot support macro F1 or PII-free document FPR claims.",
            "required_before": "human_hidden_execution",
            "affected_system_ids": all_systems,
            "affected_track_variant_ids": all_tracks,
            "acceptance_criteria": [
                "Implement paired document bootstrap for strict micro F1, strict macro F1, and PII-free document FPR.",
                "Preserve suite mixture and the frozen 1000-sample seed-42 percentile contract.",
                "Require intersection-union superiority against every admitted comparator without post-hoc deletion.",
            ],
        },
        "legacy_hybrid_semantics_source_closure": {
            "status": "pending",
            "blocking": True,
            "reason": "ERNIE and CLUENER historical hybrids used LocalRecognizer plus custom fusion, not native AnalyzerEngine execution.",
            "required_before": "registry_activation",
            "affected_system_ids": ["ernie_chinese_ner", "cluener2020_roberta"],
            "affected_track_variant_ids": [
                "presidio_ernie_cn_common_framework_hybrid",
                "presidio_cluener_cn_common_framework_hybrid",
            ],
            "acceptance_criteria": [
                "Either reproduce through the release-native AnalyzerEngine path or rename the track as custom fusion.",
                "Close historical/current runner drift and bind exact fusion source bytes.",
                "Do not label historical custom fusion as stock Presidio behavior.",
            ],
        },
    }


def _expected_registry_unsigned(
    *,
    frozen_at: str,
    builder_sha256: str,
    latest_bound_input_mtime: str,
    latest_bound_input_mtime_ns: int,
) -> dict[str, Any]:
    variants = _track_variants()
    return {
        "schema_version": 2,
        "registry_id": REGISTRY_ID,
        "leaderboard_generation": 2,
        "registry_state": "frozen_candidate_set_blocked",
        "frozen_at": frozen_at,
        "timezone": "Asia/Shanghai",
        "hash_contract": HASH_CONTRACT,
        "schema_binding": _file_binding(
            SCHEMA_RELATIVE_PATH,
            mode="0444",
        ),
        "predecessor_chain": {
            "generation_1_registry": {
                "path": "configs/evaluation/comparator_registry_v1.yaml",
                "file_sha256": EXPECTED_FILES["configs/evaluation/comparator_registry_v1.yaml"],
                "canonical_sha256": V1_REGISTRY_CANONICAL_SHA256,
                "file_mode": "0600",
                "logical_id": "comparator_registry_v1_generation_1",
                "status": "withdrawn_not_active",
                "immutability_status": "legacy_not_immutable_by_mode",
            },
            "generation_1_withdrawal": {
                "path": "configs/evaluation/comparator_registry_v1_withdrawal.yaml",
                "file_sha256": EXPECTED_FILES[
                    "configs/evaluation/comparator_registry_v1_withdrawal.yaml"
                ],
                "canonical_sha256": V1_WITHDRAWAL_CANONICAL_SHA256,
                "file_mode": "0600",
                "logical_id": "comparator_registry_v1_generation_1_freeze_time_withdrawal",
                "status": "effective_claims_disabled",
                "immutability_status": "legacy_not_immutable_by_mode",
            },
            "historical_claims_enabled": False,
        },
        "freeze_attestation": {
            "scope": "public_open_source_simplified_chinese_pii_comparator_candidate_set",
            "search_completed_at": SEARCH_COMPLETED_AT,
            "search_completed_before_freeze": True,
            "builder_binding": {
                "path": BUILDER_RELATIVE_PATH,
                "file_sha256": builder_sha256,
            },
            "all_bound_input_mtimes_not_after_freeze": True,
            "latest_bound_input_mtime": latest_bound_input_mtime,
            "latest_bound_input_mtime_ns": latest_bound_input_mtime_ns,
            "human_hidden_gold_status": "not_materialized_never_accessed",
            "human_hidden_record_count": 0,
            "human_hidden_access_count": 0,
            "candidate_set_mutation_after_hidden_access_prohibited": True,
            "freeze_does_not_activate_execution_or_claims": True,
        },
        "search_inventory": _search_inventory(),
        "protocol_contracts": {
            "public_closed_8_diagnostic": {
                "protocol_id": "pii_bench_zh_closed_8_v1",
                "status": "existing_descriptive_public_diagnostic",
                "repository": BENCHMARK["repository"],
                "revision": BENCHMARK["revision"],
                "manifest_file_sha256": BENCHMARK["manifest_file_sha256"],
                "manifest_logical_sha256": BENCHMARK["manifest_logical_sha256"],
                "labels": list(CLOSED_8_LABELS),
                "metric": "strict_exact_start_end_label",
                "unsupported_labels_count_as_false_negatives": True,
                "non_protocol_outputs_excluded_before_decode_or_metric_serialization": True,
                "pii_free_document_fpr_estimable": False,
                "benchmark_threshold_tuning": False,
                "benchmark_exposure": "test_already_seen_not_for_selection_or_claim_activation",
                "claim_eligible": False,
            },
            "generation_2_confirmatory": {
                "protocol_id": "pii_zh_generation_2_open_24_human_hidden_v1",
                "status": "blocked_not_materialized",
                "taxonomy_binding": _file_binding("src/pii_zh/taxonomy/taxonomy.yaml"),
                "tracks": [
                    "model_raw",
                    "model_calibrated",
                    "rules_only",
                    "framework_hybrid",
                    "full_system",
                ],
                "required_suites": ["formal", "chat", "pii_free", "hard_negative", "long_document"],
                "unsupported_labels_count_as_false_negatives": True,
                "closed_8_non_protocol_outputs_excluded_before_decode": True,
                "human_hidden_access_count": 0,
                "pii_free_documents_required": True,
                "execution_allowed": False,
            },
        },
        "evaluator_and_statistics": {
            "source_bindings": _source_bindings(),
            "paired_bootstrap": {
                "implementation": _file_binding("src/pii_zh/evaluation/paired_bootstrap.py"),
                "current_capability": "strict_micro_f1_only_insufficient",
                "required_metrics": ["strict_micro_f1", "strict_macro_f1", "pii_free_document_fpr"],
                "samples": 1000,
                "seed": 42,
                "confidence": 0.95,
                "resampling_unit": "document",
                "suite_mixture": "preserved",
                "ci_method": "percentile",
                "tie_handling": "zero_delta_is_not_superiority",
                "multiple_comparison": "intersection_union_must_beat_every_admitted_comparator",
                "status": "blocked_missing_macro_and_fpr",
            },
            "training_seeds": [13, 42, 97],
            "claim_statistic_status": "blocked_incomplete_three_metric_implementation",
        },
        "candidate_policy": {
            "project_candidate_models_may_be_comparators": False,
            "material_known_system_omission_prohibited": True,
            "pending_systems_must_remain_visible": True,
            "candidate_flag_required_false": True,
            "same_protocol_closure_required_before_activation": True,
        },
        "claim_policy": {
            "first_claim_enabled": False,
            "blind_claim_enabled": False,
            "best_claim_enabled": False,
            "global_best_claim_enabled": False,
            "sota_claim_enabled": False,
            "registry_claim_eligible": False,
            "scope_limited_descriptive_reporting_allowed": True,
            "activation_requires_receipts": [
                "activated_comparator_registry_receipt",
                "sealed_human_hidden_execution_receipt",
                "separate_claim_receipt",
            ],
        },
        "systems": _systems(),
        "track_variants": variants,
        "pending_requirements": _pending_requirements(tuple(variants)),
        "amendment_policy": {
            "in_place_mutation_prohibited": True,
            "new_system_requires_new_registry_id_and_hash": True,
            "post_hidden_amendment_requires_new_generation": True,
            "factual_error_requires_withdrawal_sidecar": True,
            "blocker_resolution_requires_successor_registry_id_and_hash": True,
        },
    }


def _validate_relational_invariants(registry: Mapping[str, Any]) -> None:
    systems = registry.get("systems")
    variants = registry.get("track_variants")
    pending = registry.get("pending_requirements")
    if not isinstance(systems, Mapping) or set(systems) != set(ALL_SYSTEM_IDS):
        raise ComparatorCandidateError("system set is not the exact frozen 12-system set")
    expected_variant_ids = (
        set(LEGACY_TRACK_SPECS)
        | set(OPENMED_TRACK_SPECS)
        | {"pii_engineer_multi_ner_v2_1_model_raw_pending"}
    )
    if (
        not isinstance(variants, Mapping)
        or set(variants) != expected_variant_ids
        or len(variants) != 18
    ):
        raise ComparatorCandidateError("track variant set is not the exact frozen 18-variant set")
    referenced: set[str] = set()
    labels = set(CLOSED_8_LABELS)
    for variant_id, raw in variants.items():
        if not isinstance(raw, Mapping):
            raise ComparatorCandidateError(f"{variant_id} is not an object")
        system_id = raw.get("system_id")
        if system_id not in systems:
            raise ComparatorCandidateError(f"{variant_id} references an unknown system")
        referenced.add(cast(str, system_id))
        supported = raw.get("supported_closed_8_labels")
        unsupported = raw.get("unsupported_closed_8_labels")
        if not isinstance(supported, list) or not isinstance(unsupported, list):
            raise ComparatorCandidateError(f"{variant_id} label sets are not arrays")
        if (
            len(supported) != len(set(supported))
            or len(unsupported) != len(set(unsupported))
            or set(supported) & set(unsupported)
            or set(supported) | set(unsupported) != labels
            or raw.get("closed_8_coverage") != len(supported)
        ):
            raise ComparatorCandidateError(f"{variant_id} label coverage is inconsistent")
    if referenced != set(systems):
        raise ComparatorCandidateError("registry contains an orphan system")
    exact_pending = {
        "pii_engineer_multi_same_protocol_reproduction",
        "open24_and_human_hidden_mapping_closure",
        "three_metric_paired_bootstrap_implementation",
        "legacy_hybrid_semantics_source_closure",
    }
    if not isinstance(pending, Mapping) or set(pending) != exact_pending:
        raise ComparatorCandidateError("pending blocker set changed")
    for requirement_id, raw in pending.items():
        if not isinstance(raw, Mapping):
            raise ComparatorCandidateError(f"{requirement_id} must be an object")
        affected_systems = raw.get("affected_system_ids")
        affected_variants = raw.get("affected_track_variant_ids")
        criteria = raw.get("acceptance_criteria")
        if (
            not isinstance(affected_systems, list)
            or not affected_systems
            or not set(affected_systems) <= set(systems)
            or not isinstance(affected_variants, list)
            or not affected_variants
            or not set(affected_variants) <= set(variants)
            or not isinstance(criteria, list)
            or not criteria
        ):
            raise ComparatorCandidateError(f"{requirement_id} references invalid affected IDs")


def build_candidate_registry(
    *,
    repository_root: Path,
    frozen_at: str,
) -> dict[str, Any]:
    """Build one self-hashed blocked registry after verifying every local input."""

    try:
        if repository_root.resolve(strict=True) != REPOSITORY_ROOT.resolve(strict=True):
            raise ComparatorCandidateError(
                "new freezes must execute from the repository whose builder is bound"
            )
    except OSError as exc:
        raise ComparatorCandidateError("repository root cannot be resolved safely") from exc

    frozen = _parse_aware_time(frozen_at, field="frozen_at", require_shanghai_offset=True)
    _reject_future_formal_time(frozen, field="frozen_at")
    if frozen_at != EXPECTED_FORMAL_FROZEN_AT:
        raise ComparatorCandidateError("frozen_at does not match the sealed formal freeze time")
    searched = _parse_aware_time(
        SEARCH_COMPLETED_AT,
        field="search_completed_at",
        require_shanghai_offset=True,
    )
    if searched > frozen:
        raise ComparatorCandidateError("freeze predates comparator search completion")
    artifacts, parsed = _load_and_verify_inputs(
        repository_root,
        enforce_local_freeze_modes=True,
    )
    latest_mtime_ns = max(artifact.mtime_ns for artifact in artifacts.values())
    if latest_mtime_ns > _unix_ns_from_datetime(frozen):
        raise ComparatorCandidateError("freeze predates a bound input file mtime")
    _verify_predecessor(parsed)
    _verify_legacy_reports(parsed)
    _verify_openmed(parsed)
    unsigned = _expected_registry_unsigned(
        frozen_at=frozen_at,
        builder_sha256=artifacts[BUILDER_RELATIVE_PATH].file_sha256,
        latest_bound_input_mtime=_mtime_text_from_ns(latest_mtime_ns),
        latest_bound_input_mtime_ns=latest_mtime_ns,
    )
    registry = {**unsigned, "registry_sha256": canonical_json_hash(unsigned)}
    validate_candidate_registry(
        registry,
        repository_root=repository_root,
    )
    return registry


def validate_candidate_registry(
    registry: Mapping[str, Any],
    *,
    repository_root: Path,
    enforce_current_mtimes: bool = True,
    externally_anchored_registry_file_sha256: str | None = None,
) -> str:
    """Validate self-hash, exact policy, source closure, and cross references."""

    claimed = registry.get("registry_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise ComparatorCandidateError("registry self-hash is invalid")
    unsigned = dict(registry)
    unsigned.pop("registry_sha256", None)
    if canonical_json_hash(unsigned) != claimed:
        raise ComparatorCandidateError("registry self-hash does not verify")
    frozen_at = registry.get("frozen_at")
    frozen = _parse_aware_time(frozen_at, field="frozen_at", require_shanghai_offset=True)
    _reject_future_formal_time(frozen, field="frozen_at")
    if frozen_at != EXPECTED_FORMAL_FROZEN_AT:
        raise ComparatorCandidateError("frozen_at does not match the sealed formal freeze time")
    if not enforce_current_mtimes:
        if (
            not isinstance(externally_anchored_registry_file_sha256, str)
            or _SHA256_RE.fullmatch(externally_anchored_registry_file_sha256) is None
        ):
            raise ComparatorCandidateError(
                "portable registry validation requires an external file SHA-256 anchor"
            )
        actual_file_sha256 = hashlib.sha256(_pretty_json_bytes(registry)).hexdigest()
        if actual_file_sha256 != externally_anchored_registry_file_sha256:
            raise ComparatorCandidateError("portable registry file SHA-256 anchor does not match")
    attestation = registry.get("freeze_attestation")
    if not isinstance(attestation, Mapping):
        raise ComparatorCandidateError("freeze attestation is missing")
    latest_text = attestation.get("latest_bound_input_mtime")
    latest = _parse_aware_time(
        latest_text,
        field="latest_bound_input_mtime",
        require_shanghai_offset=False,
    )
    latest_ns = attestation.get("latest_bound_input_mtime_ns")
    if isinstance(latest_ns, bool) or not isinstance(latest_ns, int) or latest_ns < 0:
        raise ComparatorCandidateError("latest bound input mtime nanoseconds are invalid")
    if latest_text != _mtime_text_from_ns(latest_ns):
        raise ComparatorCandidateError("recorded latest bound input mtimes disagree")
    if latest_ns > _unix_ns_from_datetime(frozen) or latest > frozen:
        raise ComparatorCandidateError("freeze predates its recorded latest bound input")
    searched = _parse_aware_time(
        attestation.get("search_completed_at"),
        field="search_completed_at",
        require_shanghai_offset=True,
    )
    if searched > frozen:
        raise ComparatorCandidateError("freeze predates comparator search completion")
    artifacts, parsed = _load_and_verify_inputs(
        repository_root,
        enforce_local_freeze_modes=enforce_current_mtimes,
    )
    current_latest_ns = max(artifact.mtime_ns for artifact in artifacts.values())
    current_latest_text = _mtime_text_from_ns(current_latest_ns)
    if enforce_current_mtimes:
        if latest_ns != current_latest_ns or latest_text != current_latest_text:
            raise ComparatorCandidateError("recorded latest bound input mtime changed")
        if current_latest_ns > _unix_ns_from_datetime(frozen):
            raise ComparatorCandidateError("freeze predates a current bound input file mtime")
    _verify_predecessor(parsed)
    _verify_legacy_reports(parsed)
    _verify_openmed(parsed)
    expected = _expected_registry_unsigned(
        frozen_at=cast(str, frozen_at),
        builder_sha256=artifacts[BUILDER_RELATIVE_PATH].file_sha256,
        latest_bound_input_mtime=(
            current_latest_text if enforce_current_mtimes else cast(str, latest_text)
        ),
        latest_bound_input_mtime_ns=(current_latest_ns if enforce_current_mtimes else latest_ns),
    )
    if unsigned != expected:
        raise ComparatorCandidateError("registry differs from the frozen exact policy")
    _validate_registry_schema(registry, parsed[SCHEMA_RELATIVE_PATH])
    _validate_relational_invariants(registry)
    claims = registry.get("claim_policy")
    if not isinstance(claims, Mapping) or any(
        claims.get(field) is not False
        for field in (
            "first_claim_enabled",
            "blind_claim_enabled",
            "best_claim_enabled",
            "global_best_claim_enabled",
            "sota_claim_enabled",
            "registry_claim_eligible",
        )
    ):
        raise ComparatorCandidateError("registry enables a prohibited claim")
    return claimed


def _expected_checks() -> dict[str, dict[str, Any]]:
    return {
        "candidate_set_identity": {
            "status": "PASS",
            "evidence": "12 systems and 18 track variants pass the bound Draft 2020-12 schema; project candidate models are excluded.",
        },
        "predecessor_withdrawal_chain": {
            "status": "PASS",
            "evidence": "Generation-1 exact bytes and effective claims-disabled withdrawal are bound.",
        },
        "public_search_inventory": {
            "status": "PASS",
            "evidence": "Recent material Chinese-capable PII Engineer Multi v2.1 remains visible as required pending.",
        },
        "closed8_public_diagnostics": {
            "status": "PASS",
            "evidence": "Nine legacy and eight OpenMed aggregate-only track selectors resolve; all remain descriptive.",
        },
        "source_mapping_and_runner_bindings": {
            "status": "PASS",
            "evidence": "Container bytes, mapping selectors/payload hashes, current runner bytes, and disclosed drift are bound.",
        },
        "claims_disabled": {
            "status": "PASS",
            "evidence": "first, blind, best, global-best, SOTA, and registry claim eligibility are false.",
        },
        "pii_engineer_multi_same_protocol_reproduction": {
            "status": "BLOCKED",
            "evidence": "No repository-local artifact, mapping, runner, license closure, or same-protocol result exists.",
        },
        "open24_and_human_hidden_mapping_closure": {
            "status": "BLOCKED",
            "evidence": "Open-24 member contracts and the zero-access human hidden test are not materialized.",
        },
        "three_metric_paired_bootstrap_implementation": {
            "status": "BLOCKED",
            "evidence": "Current implementation is strict-micro-only; macro and PII-free document FPR are missing.",
        },
        "legacy_hybrid_semantics_source_closure": {
            "status": "BLOCKED",
            "evidence": "ERNIE/CLUENER historical custom fusion is not yet closed as release-native AnalyzerEngine execution.",
        },
    }


def build_candidate_receipt(
    *,
    registry: Mapping[str, Any],
    registry_file_sha256: str,
    generated_at: str,
) -> dict[str, Any]:
    frozen_at = registry.get("frozen_at")
    frozen = _parse_aware_time(frozen_at, field="registry.frozen_at", require_shanghai_offset=True)
    generated = _parse_aware_time(generated_at, field="generated_at", require_shanghai_offset=False)
    _reject_future_formal_time(generated, field="generated_at")
    if generated != frozen or generated_at != frozen_at:
        raise ComparatorCandidateError(
            "receipt generated_at must exactly equal the registry freeze"
        )
    if _SHA256_RE.fullmatch(registry_file_sha256) is None:
        raise ComparatorCandidateError("registry file SHA-256 is invalid")
    checks = _expected_checks()
    blocked = sorted(key for key, value in checks.items() if value["status"] == "BLOCKED")
    passed = sorted(key for key, value in checks.items() if value["status"] == "PASS")
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "receipt_id": RECEIPT_ID,
        "status": "BLOCKED",
        "generated_at": generated_at,
        "timezone": "Asia/Shanghai",
        "hash_contract": RECEIPT_HASH_CONTRACT,
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "file_sha256": registry_file_sha256,
            "canonical_sha256": registry["registry_sha256"],
            "file_mode": "0444",
            "registry_id": REGISTRY_ID,
            "registry_state": "frozen_candidate_set_blocked",
        },
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(passed),
            "blocked_check_count": len(blocked),
            "passed_check_ids": passed,
            "blocked_check_ids": blocked,
            "system_count": 12,
            "track_variant_count": 18,
            "existing_public_diagnostic_track_count": 17,
            "required_pending_track_count": 1,
        },
        "checks": checks,
        "activation": {
            "registry_activation_allowed": False,
            "human_hidden_execution_allowed": False,
            "claim_activation_allowed": False,
            "blocked_freeze_may_be_mutated_in_place": False,
            "successor_registry_required": True,
        },
        "claims": {
            "first": False,
            "blind": False,
            "best": False,
            "global_best": False,
            "sota": False,
            "scope_limited_descriptive_reporting_only": True,
        },
        "verification_scope": {
            "strict_local_replay": "bytes_policy_schema_modes_and_original_mtimes",
            "portable_git_checkout_replay": "bytes_policy_schema_and_self_consistent_build_time_mtime_attestation",
            "portable_external_registry_and_receipt_file_hashes_required": True,
            "portable_replay_alone_does_not_reconstruct_original_mtimes": True,
        },
        "privacy": {
            "benchmark_rows_read": 0,
            "hidden_gold_read": 0,
            "prediction_files_read": 0,
            "model_weights_loaded": False,
            "gpu_queried_or_used": False,
            "network_used": False,
            "raw_text_or_entity_values_copied": False,
            "aggregate_and_metadata_only": True,
        },
    }
    receipt = {**unsigned, "receipt_sha256": canonical_json_hash(unsigned)}
    validate_candidate_receipt(
        receipt, registry=registry, registry_file_sha256=registry_file_sha256
    )
    return receipt


def validate_candidate_receipt(
    receipt: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    registry_file_sha256: str,
) -> str:
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise ComparatorCandidateError("receipt self-hash is invalid")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if canonical_json_hash(unsigned) != claimed:
        raise ComparatorCandidateError("receipt self-hash does not verify")
    generated_at = receipt.get("generated_at")
    expected = build_candidate_receipt_unsigned(
        registry=registry,
        registry_file_sha256=registry_file_sha256,
        generated_at=generated_at,
    )
    if unsigned != expected:
        raise ComparatorCandidateError("receipt differs from the exact blocked policy")
    return claimed


def build_candidate_receipt_unsigned(
    *,
    registry: Mapping[str, Any],
    registry_file_sha256: str,
    generated_at: object,
) -> dict[str, Any]:
    if not isinstance(generated_at, str):
        raise ComparatorCandidateError("receipt generated_at is invalid")
    # Build once through the same implementation but stop before recursive validation.
    _parse_aware_time(generated_at, field="generated_at", require_shanghai_offset=False)
    frozen_at = registry.get("frozen_at")
    frozen = _parse_aware_time(frozen_at, field="registry.frozen_at", require_shanghai_offset=True)
    generated = _parse_aware_time(generated_at, field="generated_at", require_shanghai_offset=False)
    _reject_future_formal_time(generated, field="generated_at")
    if generated != frozen or generated_at != frozen_at:
        raise ComparatorCandidateError(
            "receipt generated_at must exactly equal the registry freeze"
        )
    checks = _expected_checks()
    blocked = sorted(key for key, value in checks.items() if value["status"] == "BLOCKED")
    passed = sorted(key for key, value in checks.items() if value["status"] == "PASS")
    return {
        "schema_version": 1,
        "receipt_id": RECEIPT_ID,
        "status": "BLOCKED",
        "generated_at": generated_at,
        "timezone": "Asia/Shanghai",
        "hash_contract": RECEIPT_HASH_CONTRACT,
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "file_sha256": registry_file_sha256,
            "canonical_sha256": registry["registry_sha256"],
            "file_mode": "0444",
            "registry_id": REGISTRY_ID,
            "registry_state": "frozen_candidate_set_blocked",
        },
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(passed),
            "blocked_check_count": len(blocked),
            "passed_check_ids": passed,
            "blocked_check_ids": blocked,
            "system_count": 12,
            "track_variant_count": 18,
            "existing_public_diagnostic_track_count": 17,
            "required_pending_track_count": 1,
        },
        "checks": checks,
        "activation": {
            "registry_activation_allowed": False,
            "human_hidden_execution_allowed": False,
            "claim_activation_allowed": False,
            "blocked_freeze_may_be_mutated_in_place": False,
            "successor_registry_required": True,
        },
        "claims": {
            "first": False,
            "blind": False,
            "best": False,
            "global_best": False,
            "sota": False,
            "scope_limited_descriptive_reporting_only": True,
        },
        "verification_scope": {
            "strict_local_replay": "bytes_policy_schema_modes_and_original_mtimes",
            "portable_git_checkout_replay": "bytes_policy_schema_and_self_consistent_build_time_mtime_attestation",
            "portable_external_registry_and_receipt_file_hashes_required": True,
            "portable_replay_alone_does_not_reconstruct_original_mtimes": True,
        },
        "privacy": {
            "benchmark_rows_read": 0,
            "hidden_gold_read": 0,
            "prediction_files_read": 0,
            "model_weights_loaded": False,
            "gpu_queried_or_used": False,
            "network_used": False,
            "raw_text_or_entity_values_copied": False,
            "aggregate_and_metadata_only": True,
        },
    }


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        content = (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ComparatorCandidateError("publication cannot be serialized") from exc
    _parse_json_bytes(content, field="publication snapshot")
    return content


def _has_symlink_ancestor(path: Path) -> bool:
    current = path.parent
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _prepare_publication(path: Path, content: bytes) -> Path:
    if path.exists() or path.is_symlink():
        raise ComparatorCandidateError("refusing to overwrite an existing output")
    if _has_symlink_ancestor(path):
        raise ComparatorCandidateError("refusing to publish through a symlinked directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_ancestor(path):
        raise ComparatorCandidateError("refusing to publish through a symlinked directory")
    handle = tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        snapshot = _read_plain_file_no_follow(temporary)
        if snapshot != content:
            raise ComparatorCandidateError("temporary publication bytes changed")
        temporary.chmod(0o444)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_plain_file_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ComparatorCandidateError(
            "published file cannot be opened without following symlinks"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ComparatorCandidateError("published file is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _same_stat(before, after):
        raise ComparatorCandidateError("published file changed during read")
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_pair_no_clobber(
    *,
    registry: Mapping[str, Any],
    receipt: Mapping[str, Any],
    repository_root: Path,
    registry_output: Path,
    receipt_output: Path,
) -> tuple[str, str]:
    """Publish a registry followed by its receipt commit marker without clobbering."""

    if registry_output == receipt_output:
        raise ComparatorCandidateError("registry and receipt outputs must differ")
    validate_candidate_registry(
        registry,
        repository_root=repository_root,
        enforce_current_mtimes=True,
    )
    registry_bytes = _pretty_json_bytes(registry)
    receipt_bytes = _pretty_json_bytes(receipt)
    registry_file_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    validate_candidate_receipt(
        receipt,
        registry=registry,
        registry_file_sha256=registry_file_sha256,
    )
    registry_temp: Path | None = None
    receipt_temp: Path | None = None
    created_registry = False
    created_receipt = False
    try:
        registry_preexists = registry_output.exists() or registry_output.is_symlink()
        receipt_preexists = receipt_output.exists() or receipt_output.is_symlink()
        if receipt_preexists:
            raise ComparatorCandidateError("refusing to overwrite an existing output")
        if registry_preexists:
            if registry_output.is_symlink() or _has_symlink_ancestor(registry_output):
                raise ComparatorCandidateError("orphan registry is reachable through a symlink")
            if stat.S_IMODE(registry_output.stat().st_mode) != 0o444:
                raise ComparatorCandidateError("orphan registry mode is not 0444")
            if _read_plain_file_no_follow(registry_output) != registry_bytes:
                raise ComparatorCandidateError("orphan registry bytes do not match this freeze")
        else:
            registry_temp = _prepare_publication(registry_output, registry_bytes)
        receipt_temp = _prepare_publication(receipt_output, receipt_bytes)
        if registry_temp is not None:
            os.link(registry_temp, registry_output)
            created_registry = True
            _fsync_directory(registry_output.parent)
        os.link(receipt_temp, receipt_output)
        created_receipt = True
        _fsync_directory(receipt_output.parent)
        if stat.S_IMODE(registry_output.stat().st_mode) != 0o444:
            raise ComparatorCandidateError("published registry mode is not 0444")
        if stat.S_IMODE(receipt_output.stat().st_mode) != 0o444:
            raise ComparatorCandidateError("published receipt mode is not 0444")
        actual_registry = _read_plain_file_no_follow(registry_output)
        actual_receipt = _read_plain_file_no_follow(receipt_output)
        if actual_registry != registry_bytes or actual_receipt != receipt_bytes:
            raise ComparatorCandidateError("published bytes do not match snapshots")
        return (
            registry_file_sha256,
            hashlib.sha256(actual_receipt).hexdigest(),
        )
    except BaseException as exc:
        if created_receipt:
            receipt_output.unlink(missing_ok=True)
            try:
                _fsync_directory(receipt_output.parent)
            except OSError:
                pass
        if created_registry:
            registry_output.unlink(missing_ok=True)
            try:
                _fsync_directory(registry_output.parent)
            except OSError:
                pass
        if isinstance(exc, ComparatorCandidateError):
            raise
        if isinstance(exc, OSError):
            raise ComparatorCandidateError("atomic no-clobber publication failed") from exc
        raise
    finally:
        if receipt_temp is not None:
            receipt_temp.unlink(missing_ok=True)
        if registry_temp is not None:
            registry_temp.unlink(missing_ok=True)


def verify_published_pair(
    *,
    repository_root: Path,
    registry_relative_path: str = REGISTRY_RELATIVE_PATH,
    receipt_relative_path: str = RECEIPT_RELATIVE_PATH,
    portable_git_checkout: bool = False,
    expected_registry_file_sha256: str | None = None,
    expected_receipt_file_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    registry_artifact = _read_repository_artifact(repository_root, registry_relative_path)
    receipt_artifact = _read_repository_artifact(repository_root, receipt_relative_path)
    if portable_git_checkout:
        if (
            not isinstance(expected_registry_file_sha256, str)
            or _SHA256_RE.fullmatch(expected_registry_file_sha256) is None
            or not isinstance(expected_receipt_file_sha256, str)
            or _SHA256_RE.fullmatch(expected_receipt_file_sha256) is None
        ):
            raise ComparatorCandidateError(
                "portable replay requires external registry and receipt file SHA-256 anchors"
            )
        if registry_artifact.file_sha256 != expected_registry_file_sha256:
            raise ComparatorCandidateError("portable registry file SHA-256 anchor does not match")
        if receipt_artifact.file_sha256 != expected_receipt_file_sha256:
            raise ComparatorCandidateError("portable receipt file SHA-256 anchor does not match")
    allowed_modes = {0o444, 0o644} if portable_git_checkout else {0o444}
    if registry_artifact.mode not in allowed_modes or receipt_artifact.mode not in allowed_modes:
        expected = "0444 locally or 0644 after a Git checkout" if portable_git_checkout else "0444"
        raise ComparatorCandidateError(f"published artifact modes must be {expected}")
    registry = _parse_json_bytes(registry_artifact.content, field=registry_relative_path)
    receipt = _parse_json_bytes(receipt_artifact.content, field=receipt_relative_path)
    if registry_artifact.content != _pretty_json_bytes(registry):
        raise ComparatorCandidateError("published registry is not deterministic pretty JSON")
    if receipt_artifact.content != _pretty_json_bytes(receipt):
        raise ComparatorCandidateError("published receipt is not deterministic pretty JSON")
    validate_candidate_registry(
        registry,
        repository_root=repository_root,
        enforce_current_mtimes=not portable_git_checkout,
        externally_anchored_registry_file_sha256=(
            expected_registry_file_sha256 if portable_git_checkout else None
        ),
    )
    validate_candidate_receipt(
        receipt,
        registry=registry,
        registry_file_sha256=registry_artifact.file_sha256,
    )
    return registry, receipt, registry_artifact.file_sha256, receipt_artifact.file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--frozen-at")
    parser.add_argument("--generated-at")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--portable-git-checkout", action="store_true")
    parser.add_argument("--expected-registry-file-sha256")
    parser.add_argument("--expected-receipt-file-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repository_root = args.repository_root
        registry_output = repository_root / REGISTRY_RELATIVE_PATH
        receipt_output = repository_root / RECEIPT_RELATIVE_PATH
        if args.portable_git_checkout and not args.verify_existing:
            raise ComparatorCandidateError(
                "--portable-git-checkout is valid only with --verify-existing"
            )
        if (
            args.expected_registry_file_sha256 or args.expected_receipt_file_sha256
        ) and not args.portable_git_checkout:
            raise ComparatorCandidateError(
                "external file SHA-256 anchors are valid only for portable replay"
            )
        if args.verify_existing and (args.frozen_at or args.generated_at):
            raise ComparatorCandidateError(
                "freeze timestamps are invalid when verifying an existing pair"
            )
        if args.verify_existing:
            registry, receipt, registry_file_sha256, receipt_file_sha256 = verify_published_pair(
                repository_root=repository_root,
                portable_git_checkout=args.portable_git_checkout,
                expected_registry_file_sha256=args.expected_registry_file_sha256,
                expected_receipt_file_sha256=args.expected_receipt_file_sha256,
            )
        else:
            if not args.frozen_at:
                raise ComparatorCandidateError("--frozen-at is required for a new formal freeze")
            registry = build_candidate_registry(
                repository_root=repository_root,
                frozen_at=args.frozen_at,
            )
            registry_bytes = _pretty_json_bytes(registry)
            registry_file_sha256 = hashlib.sha256(registry_bytes).hexdigest()
            receipt = build_candidate_receipt(
                registry=registry,
                registry_file_sha256=registry_file_sha256,
                generated_at=args.generated_at or args.frozen_at,
            )
            registry_file_sha256, receipt_file_sha256 = publish_pair_no_clobber(
                registry=registry,
                receipt=receipt,
                repository_root=repository_root,
                registry_output=registry_output,
                receipt_output=receipt_output,
            )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "registry_state": registry["registry_state"],
                    "system_count": receipt["summary"]["system_count"],
                    "track_variant_count": receipt["summary"]["track_variant_count"],
                    "passed_check_count": receipt["summary"]["passed_check_count"],
                    "blocked_check_count": receipt["summary"]["blocked_check_count"],
                    "verification_mode": (
                        "portable_hash_anchored_policy_replay"
                        if args.portable_git_checkout
                        else "strict_local_original_mtime_replay"
                    ),
                    "registry_sha256": registry["registry_sha256"],
                    "registry_file_sha256": registry_file_sha256,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "receipt_file_sha256": receipt_file_sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        message = str(exc) if isinstance(exc, ComparatorCandidateError) else "build failed safely"
        print(f"comparator-registry-v2-candidate: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
