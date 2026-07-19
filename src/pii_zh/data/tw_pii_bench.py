"""Read-only validation and import for the frozen tw-PII-bench source snapshot.

Only the three pinned Parquet files are accepted.  The public record loader
returns immutable Python objects for evaluation protocol code; the manifest
path contains aggregate metadata only and never serializes source text,
record identifiers, or entity surfaces.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from stat import S_IMODE
from typing import Any

from pii_zh.training.safe_conversion import canonical_json_hash, sha256_file

SOURCE_DATASET_ID = "lianghsun/tw-PII-bench"
SOURCE_REVISION = "62bdad506803b15e1a4193f4493200ad431b1af7"
SOURCE_DECLARED_LICENSE = "Apache-2.0"
SOURCE_ROOT_ENVIRONMENT_VARIABLE = "PII_ZH_TW_PII_BENCH_SOURCE_ROOT"
SOURCE_CARD_SHA256 = "7055db1baacaff4e1b3c5213db766113c7de8a40a56a329ac1fb2a36d0778756"

V5B_FREEZE_RECEIPT_ID = "macbert_24_numeric_o_v5b_diagnostic_preregistration_v1"
V5B_FREEZE_RECEIPT_SHA256 = "0e6fe1a3244a30471eeeb51aa07e3f1b0e589d65939f8e9ab78257c40a6646c3"
V5B_FREEZE_RECEIPT_FILE_SHA256 = "bb7d950d5580adebd0cf33c77650b816530c5df27a620abd507c38fe77222bdb"
V5B_TRAINING_REPORT_FILE_SHA256 = "a9eb5dee06a8674d241a737f2e0e4db5703214f7e7bd1e14df1d490e81e0a3d3"

SPLITS = ("short", "mid", "long")
EXPECTED_SOURCE_FILES: dict[str, dict[str, int | str]] = {
    "short": {
        "filename": "short.parquet",
        "sha256": "03e4850a0a97a2b5cb046c1dcd3996b42c5eea3766ae2618d4bbf64391e73693",
        "size_bytes": 40_062,
        "rows": 310,
    },
    "mid": {
        "filename": "mid.parquet",
        "sha256": "5d05e1c7fc6adaf7c66591de0d76e5ac5d175fb182f67a19da6d23896b3b02e0",
        "size_bytes": 275_197,
        "rows": 300,
    },
    "long": {
        "filename": "long.parquet",
        "sha256": "b30ca2a1012254f164893d1aad491a778d2fe49cfef71239f8c934e0544883e2",
        "size_bytes": 1_896_519,
        "rows": 300,
    },
}
EXPECTED_BLOCK_COUNTS: dict[str, dict[str, int]] = {
    "short": {"A": 160, "B": 110, "C": 40},
    "mid": {"M": 300},
    "long": {"L": 300},
}
EXPECTED_TOTAL_ROWS = 910
EXPECTED_TOTAL_SPANS = 7_915
EXPECTED_PII_FREE_ROWS = 40
EXPECTED_NO_FALLBACK_SPANS = 545

IN_SCHEMA_LABELS = frozenset(
    {
        "private_person",
        "private_phone",
        "private_email",
        "private_address",
        "private_date",
        "private_url",
        "account_number",
        "secret",
    }
)
EXPECTED_MODEL_LABELS = IN_SCHEMA_LABELS | {""}
EXPECTED_MODEL_LABEL_BY_SOURCE_LABEL: dict[str, str] = {
    **{label: label for label in IN_SCHEMA_LABELS},
    "tw_national_id": "account_number",
    "tw_nhi_card": "account_number",
    "tw_company_id": "account_number",
    "tw_license_plate": "",
    "tw_passport": "account_number",
    "tw_driver_license": "account_number",
    "tw_line_id": "private_url",
    "tw_ptt_id": "",
    "tw_household_no": "account_number",
    "tw_medical_license": "account_number",
    "tw_military_id": "account_number",
}
SOURCE_LABELS = frozenset(EXPECTED_MODEL_LABEL_BY_SOURCE_LABEL)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_V5B_FREEZE_RECEIPT = (
    _REPOSITORY_ROOT / "reports/experiments/macbert_24_numeric_o_v5b_preregistration_freeze.json"
)
_DEFAULT_OUTPUT = _REPOSITORY_ROOT / "reports/evaluation/tw_pii_bench_source_manifest.json"
_RECORD_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
_SPAN_FIELDS = frozenset({"start", "end", "label", "text", "expected_model_label"})


class TwPiiBenchSourceError(RuntimeError):
    """Fail-closed source error that never reflects text, IDs, or local paths."""


def _configured_source_root(source_root: Path | None) -> Path:
    """Resolve an explicit argument or the documented environment setting.

    There is intentionally no machine-specific fallback.  The environment
    value must be an absolute path so an installed entry point cannot silently
    change datasets when its working directory changes.
    """

    if source_root is not None:
        return source_root
    configured = os.environ.get(SOURCE_ROOT_ENVIRONMENT_VARIABLE)
    if configured is None or not configured.strip():
        raise TwPiiBenchSourceError(
            "source root is required explicitly or through the documented environment setting"
        )
    candidate = Path(configured)
    if not candidate.is_absolute():
        raise TwPiiBenchSourceError("configured source root must be an absolute path")
    return candidate


@dataclass(frozen=True, slots=True)
class TwPiiSpan:
    """One validated source span; an absent fallback is normalized to ``None``."""

    start: int
    end: int
    label: str
    expected_model_label: str | None


@dataclass(frozen=True, slots=True)
class TwPiiRecord:
    """Immutable validated record exposed to the evaluation protocol layer."""

    record_id: str
    split: str
    block: str
    category: str
    scenario: str
    text: str
    spans: tuple[TwPiiSpan, ...]
    is_negative: bool

    @property
    def is_pii_free(self) -> bool:
        return not self.spans


@dataclass(frozen=True, slots=True)
class _ValidatedSource:
    records: tuple[TwPiiRecord, ...]
    files: tuple[dict[str, int | str], ...]


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional local evaluation dependency
        raise TwPiiBenchSourceError("tw-PII-bench validation requires pyarrow") from exc
    return pa, parquet


def _expected_arrow_schema(pa: Any) -> Any:
    span = pa.struct(
        [
            pa.field("start", pa.int64()),
            pa.field("end", pa.int64()),
            pa.field("label", pa.string()),
            pa.field("text", pa.string()),
            pa.field("expected_model_label", pa.string()),
        ]
    )
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("split", pa.string()),
            pa.field("block", pa.string()),
            pa.field("category", pa.string()),
            pa.field("scenario", pa.string()),
            pa.field("text", pa.string()),
            pa.field("spans", pa.list_(pa.field("element", span))),
            pa.field("is_negative", pa.bool_()),
            pa.field("char_length", pa.int64()),
            pa.field("n_spans", pa.int64()),
        ]
    )


def _schema_contract() -> list[dict[str, Any]]:
    return [
        {"name": "id", "type": "utf8"},
        {"name": "split", "type": "utf8"},
        {"name": "block", "type": "utf8"},
        {"name": "category", "type": "utf8"},
        {"name": "scenario", "type": "utf8"},
        {"name": "text", "type": "utf8"},
        {
            "name": "spans",
            "type": "list<struct>",
            "fields": [
                {"name": "start", "type": "int64"},
                {"name": "end", "type": "int64"},
                {"name": "label", "type": "utf8"},
                {"name": "text", "type": "utf8"},
                {"name": "expected_model_label", "type": "utf8"},
            ],
        },
        {"name": "is_negative", "type": "bool"},
        {"name": "char_length", "type": "int64"},
        {"name": "n_spans", "type": "int64"},
    ]


def _regular_source_root(source_root: Path) -> Path:
    try:
        root = source_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TwPiiBenchSourceError("source root is unavailable") from exc
    if source_root.is_symlink() or not root.is_dir():
        raise TwPiiBenchSourceError("source root must be a regular non-symlink directory")
    observed = {path.name for path in root.glob("*.parquet")}
    expected = {str(contract["filename"]) for contract in EXPECTED_SOURCE_FILES.values()}
    if observed != expected:
        raise TwPiiBenchSourceError("source Parquet inventory violates the frozen contract")
    return root


def _utf8_string(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise TwPiiBenchSourceError(f"{field} must be {requirement}")
    try:
        if value.encode("utf-8").decode("utf-8") != value:
            raise UnicodeError
    except UnicodeError as exc:
        raise TwPiiBenchSourceError(f"{field} is not canonical UTF-8 text") from exc
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TwPiiBenchSourceError(f"{field} must be an integer >= {minimum}")
    return value


def _parse_span(value: object, *, text: str, previous_end: int) -> tuple[TwPiiSpan, int]:
    if not isinstance(value, Mapping) or set(value) != _SPAN_FIELDS:
        raise TwPiiBenchSourceError("span schema violates the frozen contract")
    start = _integer(value["start"], field="span.start")
    end = _integer(value["end"], field="span.end", minimum=1)
    label = _utf8_string(value["label"], field="span.label")
    surface = _utf8_string(value["text"], field="span.text")
    expected = value["expected_model_label"]
    if not isinstance(expected, str) or expected not in EXPECTED_MODEL_LABELS:
        raise TwPiiBenchSourceError("expected_model_label is outside the frozen value domain")
    if label not in SOURCE_LABELS:
        raise TwPiiBenchSourceError("span label is outside the frozen source taxonomy")
    if EXPECTED_MODEL_LABEL_BY_SOURCE_LABEL[label] != expected:
        raise TwPiiBenchSourceError("expected_model_label is inconsistent with its source label")
    if not 0 <= start < end <= len(text) or start < previous_end:
        raise TwPiiBenchSourceError("span offsets are invalid, overlapping, or unsorted")
    if text[start:end] != surface:
        raise TwPiiBenchSourceError("span entity text does not match UTF-8 character offsets")
    return (
        TwPiiSpan(
            start=start,
            end=end,
            label=label,
            expected_model_label=expected or None,
        ),
        end,
    )


def _parse_record(value: object, *, file_split: str) -> TwPiiRecord:
    if not isinstance(value, Mapping):
        raise TwPiiBenchSourceError("source row must be an object")
    record_id = _utf8_string(value.get("id"), field="id")
    if _RECORD_ID_PATTERN.fullmatch(record_id) is None:
        raise TwPiiBenchSourceError("record identifier violates the frozen public format")
    split = _utf8_string(value.get("split"), field="split")
    if split != file_split:
        raise TwPiiBenchSourceError("row split does not match its frozen source file")
    block = _utf8_string(value.get("block"), field="block")
    category = _utf8_string(value.get("category"), field="category")
    scenario = _utf8_string(value.get("scenario"), field="scenario", allow_empty=True)
    text = _utf8_string(value.get("text"), field="text")
    char_length = _integer(value.get("char_length"), field="char_length", minimum=1)
    if char_length != len(text):
        raise TwPiiBenchSourceError("char_length differs from Python Unicode character count")
    raw_spans = value.get("spans")
    if not isinstance(raw_spans, list):
        raise TwPiiBenchSourceError("spans must be a list")
    n_spans = _integer(value.get("n_spans"), field="n_spans")
    if n_spans != len(raw_spans):
        raise TwPiiBenchSourceError("n_spans differs from the span list length")
    parsed_spans: list[TwPiiSpan] = []
    previous_end = 0
    for raw_span in raw_spans:
        span, previous_end = _parse_span(raw_span, text=text, previous_end=previous_end)
        parsed_spans.append(span)
    is_negative = value.get("is_negative")
    if not isinstance(is_negative, bool) or is_negative != (not parsed_spans):
        raise TwPiiBenchSourceError("PII-free/negative state is inconsistent with spans")
    return TwPiiRecord(
        record_id=record_id,
        split=split,
        block=block,
        category=category,
        scenario=scenario,
        text=text,
        spans=tuple(parsed_spans),
        is_negative=is_negative,
    )


def _validate_source(source_root: Path) -> _ValidatedSource:
    root = _regular_source_root(source_root)
    pa, parquet = _pyarrow_modules()
    expected_schema = _expected_arrow_schema(pa)
    records: list[TwPiiRecord] = []
    files: list[dict[str, int | str]] = []
    seen_ids: set[str] = set()
    total_spans = 0
    total_pii_free = 0
    total_no_fallback = 0
    for split in SPLITS:
        contract = EXPECTED_SOURCE_FILES[split]
        source = root / str(contract["filename"])
        if source.is_symlink() or not source.is_file():
            raise TwPiiBenchSourceError("source Parquet must be a regular non-symlink file")
        if (
            source.stat().st_size != contract["size_bytes"]
            or sha256_file(source) != contract["sha256"]
        ):
            raise TwPiiBenchSourceError("source Parquet fingerprint violates the frozen contract")
        try:
            parquet_file = parquet.ParquetFile(source)
            if not parquet_file.schema_arrow.equals(expected_schema, check_metadata=True):
                raise TwPiiBenchSourceError("source Parquet schema violates the frozen contract")
            if parquet_file.metadata.num_rows != contract["rows"]:
                raise TwPiiBenchSourceError("source Parquet row count violates the frozen contract")
            rows = parquet_file.read().to_pylist()
        except TwPiiBenchSourceError:
            raise
        except Exception as exc:
            raise TwPiiBenchSourceError("source Parquet could not be read safely") from exc
        block_counts: Counter[str] = Counter()
        split_records: list[TwPiiRecord] = []
        for value in rows:
            record = _parse_record(value, file_split=split)
            if record.record_id in seen_ids:
                raise TwPiiBenchSourceError("record identifiers are not globally unique")
            seen_ids.add(record.record_id)
            block_counts[record.block] += 1
            total_spans += len(record.spans)
            total_pii_free += int(record.is_pii_free)
            total_no_fallback += sum(span.expected_model_label is None for span in record.spans)
            split_records.append(record)
        if dict(sorted(block_counts.items())) != EXPECTED_BLOCK_COUNTS[split]:
            raise TwPiiBenchSourceError("source block counts violate the frozen contract")
        if len(split_records) != contract["rows"]:
            raise TwPiiBenchSourceError("decoded row count violates the frozen contract")
        records.extend(split_records)
        files.append(
            {
                "split": split,
                "filename": str(contract["filename"]),
                "sha256": str(contract["sha256"]),
                "size_bytes": int(contract["size_bytes"]),
                "rows": len(split_records),
            }
        )
    if (
        len(records) != EXPECTED_TOTAL_ROWS
        or total_spans != EXPECTED_TOTAL_SPANS
        or total_pii_free != EXPECTED_PII_FREE_ROWS
        or total_no_fallback != EXPECTED_NO_FALLBACK_SPANS
    ):
        raise TwPiiBenchSourceError("combined source statistics violate the frozen contract")
    return _ValidatedSource(records=tuple(records), files=tuple(files))


def load_tw_pii_bench_records(source_root: Path | None = None) -> tuple[TwPiiRecord, ...]:
    """Return all 910 validated records in short/mid/long source order.

    The loader is strictly read-only, validates the pinned file hashes and
    schema, and performs no model loading or inference.
    """

    return _validate_source(_configured_source_root(source_root)).records


def _verify_v5b_freeze() -> dict[str, str]:
    if (
        _V5B_FREEZE_RECEIPT.is_symlink()
        or not _V5B_FREEZE_RECEIPT.is_file()
        or sha256_file(_V5B_FREEZE_RECEIPT) != V5B_FREEZE_RECEIPT_FILE_SHA256
    ):
        raise TwPiiBenchSourceError("v5b freeze receipt fingerprint changed")
    try:
        receipt = json.loads(_V5B_FREEZE_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TwPiiBenchSourceError("v5b freeze receipt is unreadable") from exc
    boundary = receipt.get("execution_boundary")
    if (
        not isinstance(boundary, Mapping)
        or receipt.get("receipt_id") != V5B_FREEZE_RECEIPT_ID
        or receipt.get("receipt_sha256") != V5B_FREEZE_RECEIPT_SHA256
        or receipt.get("status") != "development_only_frozen_before_GPU"
        or boundary.get("external_evaluation_count") != 0
        or boundary.get("PII_Bench_execution_count") != 0
    ):
        raise TwPiiBenchSourceError("v5b freeze receipt contract changed")
    return {
        "receipt_id": V5B_FREEZE_RECEIPT_ID,
        "receipt_sha256": V5B_FREEZE_RECEIPT_SHA256,
        "receipt_file_sha256": V5B_FREEZE_RECEIPT_FILE_SHA256,
    }


def _statistics(records: Sequence[TwPiiRecord]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    split_stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        lengths = [len(record.text) for record in selected]
        span_counts = [len(record.spans) for record in selected]
        blocks = Counter(record.block for record in selected)
        for record in selected:
            for span in record.spans:
                label_counts[span.label] += 1
                expected_counts[span.expected_model_label or "none"] += 1
        split_stats[split] = {
            "rows": len(selected),
            "pii_free_rows": sum(record.is_pii_free for record in selected),
            "span_count": sum(span_counts),
            "character_count": sum(lengths),
            "minimum_characters": min(lengths),
            "maximum_characters": max(lengths),
            "minimum_spans_per_row": min(span_counts),
            "maximum_spans_per_row": max(span_counts),
            "block_counts": dict(sorted(blocks.items())),
        }
    return {
        "rows": len(records),
        "pii_free_rows": sum(record.is_pii_free for record in records),
        "span_count": sum(len(record.spans) for record in records),
        "character_count": sum(len(record.text) for record in records),
        "split_counts": {split: split_stats[split]["rows"] for split in SPLITS},
        "splits": split_stats,
        "source_label_counts": dict(sorted(label_counts.items())),
        "expected_model_label_counts": dict(sorted(expected_counts.items())),
    }


def build_source_manifest(source_root: Path | None = None) -> dict[str, Any]:
    """Validate the fixed snapshot and build an aggregate-only self-hashed manifest."""

    validated = _validate_source(_configured_source_root(source_root))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "tw_pii_bench_frozen_source_validation",
        "status": "passed",
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "revision": SOURCE_REVISION,
            "declared_license": SOURCE_DECLARED_LICENSE,
            "dataset_card_sha256": SOURCE_CARD_SHA256,
            "license_review_status": "source_declared_not_independently_adjudicated",
            "source_root_recorded": False,
        },
        "source_properties": {
            "language": "Traditional Chinese (zh-TW)",
            "synthetic": True,
            "human_authored_context": False,
            "source_declared_all_PII_fictional": True,
            "generation_method": "Gemini_3.1_Pro_Preview_with_Search_grounding",
            "annotation_origin": "generation_time_delimiters_parsed_to_character_spans",
            "designed_against_model": "openai/privacy-filter",
        },
        "files": list(validated.files),
        "schema": {
            "fields": _schema_contract(),
            "canonical_sha256": canonical_json_hash(_schema_contract()),
            "expected_model_label_domain": [None, *sorted(IN_SCHEMA_LABELS)],
            "missing_expected_model_label_source_encoding": "empty_string_normalized_to_null",
        },
        "statistics": _statistics(validated.records),
        "validation": {
            "exact_schema": True,
            "global_record_id_uniqueness": True,
            "UTF8_round_trip": True,
            "python_unicode_character_offsets": True,
            "entity_text_exact_match": True,
            "spans_sorted_and_non_overlapping": True,
            "char_length_exact": True,
            "n_spans_exact": True,
            "PII_free_negative_consistency": True,
            "expected_model_label_domain_and_mapping": True,
        },
        "policy": {
            "data_pool": "evaluation_only",
            "model_training_allowed": False,
            "model_selection_or_tuning_allowed": False,
            "source_file_mutation_allowed": False,
            "model_execution_count": 0,
            "importer_model_result_read_count": 0,
        },
        "lineage": {
            "discovered_after_v5b_freeze": True,
            "source_revision_predates_local_discovery_possible": True,
            "used_for_v5b_training": False,
            "used_for_v5b_checkpoint_selection": False,
            "benchmark_result_blind": False,
            "project_model_result_blind": True,
            "source_card_contains_preexisting_model_results": True,
            "source_card_result_exposure_timing": (
                "after_v5b_training_completed_before_evaluation_protocol_freeze"
            ),
            "source_card_result_permitted_methodology_influence": [
                "upstream_effective_gold_method",
                "upstream_native_decoder_method",
            ],
            "source_card_results_used_for_v5b_training_or_selection": False,
            "source_card_results_used_for_threshold_tuning": False,
            "v5b_training_completion_attested": True,
            "v5b_training_report_file_sha256": V5B_TRAINING_REPORT_FILE_SHA256,
            "v5b_freeze": _verify_v5b_freeze(),
        },
        "privacy": {
            "contains_raw_text": False,
            "contains_entity_text": False,
            "contains_record_ids": False,
            "contains_absolute_paths": False,
        },
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    return manifest


def verify_source_manifest(manifest: Mapping[str, Any]) -> None:
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or canonical_json_hash(unsigned) != claimed:
        raise TwPiiBenchSourceError("source manifest self-hash does not verify")
    if (
        manifest.get("status") != "passed"
        or manifest.get("statistics", {}).get("rows") != EXPECTED_TOTAL_ROWS
        or manifest.get("statistics", {}).get("pii_free_rows") != EXPECTED_PII_FREE_ROWS
        or manifest.get("policy", {}).get("data_pool") != "evaluation_only"
        or manifest.get("policy", {}).get("model_training_allowed") is not False
        or manifest.get("privacy")
        != {
            "contains_raw_text": False,
            "contains_entity_text": False,
            "contains_record_ids": False,
            "contains_absolute_paths": False,
        }
    ):
        raise TwPiiBenchSourceError("source manifest violates the frozen contract")


def write_source_manifest(manifest: Mapping[str, Any], output: Path) -> None:
    verify_source_manifest(manifest)
    destination = output.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TwPiiBenchSourceError("source manifest output must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if S_IMODE(destination.stat().st_mode) != 0o444:
        raise TwPiiBenchSourceError("source manifest output mode verification failed")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local directory containing the three hash-pinned Parquet files.",
    )
    command.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = build_source_manifest(args.source_root)
    write_source_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": manifest["statistics"]["rows"],
                "manifest_sha256": manifest["manifest_sha256"],
                "model_execution_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "EXPECTED_MODEL_LABELS",
    "EXPECTED_NO_FALLBACK_SPANS",
    "EXPECTED_SOURCE_FILES",
    "IN_SCHEMA_LABELS",
    "SOURCE_LABELS",
    "SOURCE_REVISION",
    "SOURCE_ROOT_ENVIRONMENT_VARIABLE",
    "SPLITS",
    "TwPiiBenchSourceError",
    "TwPiiRecord",
    "TwPiiSpan",
    "build_source_manifest",
    "load_tw_pii_bench_records",
    "main",
    "parser",
    "verify_source_manifest",
    "write_source_manifest",
]
