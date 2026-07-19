"""Trust anchor and verifier for the targeted-repair-v3 freeze receipt."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...targeted_repair_v3_closure import DATA_IMPLEMENTATION_PATHS
from .targeted_repair_v3_freeze_anchor import (
    FREEZE_RECEIPT_FILE_SHA256,
    FREEZE_RECEIPT_LOGICAL_SHA256,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_FREEZE_ANCHOR_PATH = Path(__file__).with_name("targeted_repair_v3_freeze_anchor.py")


class TargetedRepairV3FreezeError(ValueError):
    """Raised without reflecting local paths or sensitive values."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_targeted_repair_v3_freeze(*, receipt_path: Path, recipe_path: Path) -> Mapping[str, Any]:
    """Verify the exact receipt, recipe, implementations, and freeze ordering."""

    if FREEZE_RECEIPT_FILE_SHA256 == "UNFROZEN":
        raise TargetedRepairV3FreezeError("targeted repair v3 has not been frozen")
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or any(parent.is_symlink() for parent in receipt_path.parents)
    ):
        raise TargetedRepairV3FreezeError("freeze receipt must be a regular file")
    if _sha256_file(receipt_path) != FREEZE_RECEIPT_FILE_SHA256:
        raise TargetedRepairV3FreezeError("freeze receipt byte identity changed")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV3FreezeError("freeze receipt is not readable JSON") from exc
    if not isinstance(receipt, dict):
        raise TargetedRepairV3FreezeError("freeze receipt must be an object")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        claimed != FREEZE_RECEIPT_LOGICAL_SHA256
        or _canonical_hash(unsigned) != claimed
        or receipt.get("status") != "preregistered_frozen"
        or receipt.get("receipt_id") != "targeted_repair_v3_implementation_freeze_v3"
    ):
        raise TargetedRepairV3FreezeError("freeze receipt logical identity changed")
    if (
        recipe_path.is_symlink()
        or not recipe_path.is_file()
        or any(parent.is_symlink() for parent in recipe_path.parents)
    ):
        raise TargetedRepairV3FreezeError("recipe must be a regular file")
    recipe = receipt.get("recipe")
    if not isinstance(recipe, dict) or _sha256_file(recipe_path) != recipe.get("file_sha256"):
        raise TargetedRepairV3FreezeError("recipe differs from the freeze receipt")
    frozen_at_ns = receipt.get("frozen_at_ns")
    if isinstance(frozen_at_ns, bool) or not isinstance(frozen_at_ns, int):
        raise TargetedRepairV3FreezeError("freeze ordering timestamp is invalid")
    frozen_at = receipt.get("frozen_at")
    if not isinstance(frozen_at, str):
        raise TargetedRepairV3FreezeError("freeze timestamp is missing")
    try:
        parsed_frozen_at = datetime.fromisoformat(frozen_at)
    except ValueError as exc:
        raise TargetedRepairV3FreezeError("freeze timestamp is invalid") from exc
    if parsed_frozen_at.tzinfo is None or parsed_frozen_at.utcoffset() is None:
        raise TargetedRepairV3FreezeError("freeze timestamp must be timezone-aware")
    utc_value = parsed_frozen_at.astimezone(timezone.utc)
    epoch_delta = utc_value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    parsed_ns = (
        epoch_delta.days * 86_400 + epoch_delta.seconds
    ) * 1_000_000_000 + epoch_delta.microseconds * 1000
    receipt_mtime_ns = receipt_path.stat().st_mtime_ns
    now_ns = time.time_ns()
    if (
        parsed_ns != frozen_at_ns
        or receipt_mtime_ns < frozen_at_ns
        or now_ns < receipt_mtime_ns
        or now_ns < frozen_at_ns
    ):
        raise TargetedRepairV3FreezeError("freeze timestamp ordering is inconsistent")
    if recipe_path.stat().st_mtime_ns > frozen_at_ns:
        raise TargetedRepairV3FreezeError("recipe modification time is after the freeze")
    implementations = receipt.get("implementations")
    if not isinstance(implementations, dict) or set(implementations) != set(
        DATA_IMPLEMENTATION_PATHS
    ):
        raise TargetedRepairV3FreezeError("implementation binding inventory changed")
    for name, binding in implementations.items():
        if not isinstance(binding, dict):
            raise TargetedRepairV3FreezeError("implementation binding is malformed")
        relative = binding.get("repository_relative_path")
        expected = binding.get("file_sha256")
        if (
            relative != DATA_IMPLEMENTATION_PATHS[name]
            or not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
        ):
            raise TargetedRepairV3FreezeError("implementation binding is unsafe")
        path = _REPOSITORY_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != expected
            or path.stat().st_mtime_ns > frozen_at_ns
        ):
            raise TargetedRepairV3FreezeError("frozen implementation identity changed")
    base_model = dict(receipt["base_model"])
    base_model.pop("local_directory", None)
    return {
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": claimed,
        "receipt_file_sha256": FREEZE_RECEIPT_FILE_SHA256,
        "freeze_anchor_file_sha256": _sha256_file(_FREEZE_ANCHOR_PATH.resolve(strict=True)),
        "freeze_verifier_file_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "frozen_at": receipt["frozen_at"],
        "frozen_at_ns": frozen_at_ns,
        "recipe_file_sha256": recipe["file_sha256"],
        "implementation_count": len(implementations),
        "D0": receipt["D0"],
        "base_model": base_model,
    }


__all__ = [
    "FREEZE_RECEIPT_FILE_SHA256",
    "FREEZE_RECEIPT_LOGICAL_SHA256",
    "TargetedRepairV3FreezeError",
    "verify_targeted_repair_v3_freeze",
]
