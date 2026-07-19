"""Fail-closed verifier for the targeted-repair-v4 data freeze."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...targeted_repair_v4_closure import DATA_IMPLEMENTATION_PATHS
from .targeted_repair_v4_freeze_anchor import (
    FREEZE_RECEIPT_FILE_SHA256,
    FREEZE_RECEIPT_LOGICAL_SHA256,
)

_ROOT = Path(__file__).resolve().parents[4]
_ANCHOR = Path(__file__).with_name("targeted_repair_v4_freeze_anchor.py")
_V3_WITHDRAWAL = (
    _ROOT / "reports/data_recipe_audits/targeted_repair_v4_freeze_v3_recursive_closure_withdrawal.json"
)
_V3_WITHDRAWAL_FILE_SHA256 = "f33ff4f13f50d1834c1059faccd74ff9527b9961022a6903f6851f949b0ed2d0"
_V3_WITHDRAWAL_LOGICAL_SHA256 = "ab9c9de3aa94d9d73d95b31c539fdb520ae2d81f40e7bb441b86a8743e846a8d"
_ANCHOR_KEY = "v4_data_freeze_anchor"
_BASE_MODEL_FILES = {
    "README.md": "1516e1e24fff28dfeaed65f714bc676e4739d4c70e6c9f356fbf41d2712251e5",
    "added_tokens.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "config.json": "648c8a2b51c12551f94bae19a66c21a59c9bd8aa0ebb99f2a0af7236cdb99daa",
    "conversion_receipt.json": "c1068e5239e8938d9aa27d78dc16255724b882ae8b7eadd8647efd0aa3aaa6c1",
    "model.safetensors": "e7a57c48dff5971df0e904bf8dc6b94158a704e9410896b4e2ad09c893599c17",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6",
    "tokenizer_config.json": "61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}


class TargetedRepairV4FreezeError(ValueError):
    """Raised without exposing local paths."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_targeted_repair_v4_freeze(
    *, receipt_path: Path, recipe_path: Path
) -> Mapping[str, Any]:
    if FREEZE_RECEIPT_FILE_SHA256 == "UNFROZEN":
        raise TargetedRepairV4FreezeError("v4 data implementation is not frozen")
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or recipe_path.is_symlink()
        or not recipe_path.is_file()
        or _sha(receipt_path) != FREEZE_RECEIPT_FILE_SHA256
    ):
        raise TargetedRepairV4FreezeError("v4 freeze file identity changed")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV4FreezeError("v4 freeze receipt is unreadable") from exc
    if not isinstance(receipt, dict):
        raise TargetedRepairV4FreezeError("v4 freeze receipt must be an object")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        claimed != FREEZE_RECEIPT_LOGICAL_SHA256
        or _canonical(unsigned) != claimed
        or receipt.get("receipt_id") != "targeted_repair_v4_implementation_freeze_v4"
        or receipt.get("status") != "preregistered_frozen"
    ):
        raise TargetedRepairV4FreezeError("v4 freeze logical identity changed")
    recipe = receipt.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("file_sha256") != _sha(recipe_path):
        raise TargetedRepairV4FreezeError("v4 recipe differs from freeze")
    implementations = receipt.get("implementations")
    expected_implementations = set(DATA_IMPLEMENTATION_PATHS) - {_ANCHOR_KEY}
    if (
        not isinstance(implementations, dict)
        or set(implementations) != expected_implementations
        or set(DATA_IMPLEMENTATION_PATHS) - set(implementations) != {_ANCHOR_KEY}
        or len(DATA_IMPLEMENTATION_PATHS) != 39
    ):
        raise TargetedRepairV4FreezeError("v4 closure inventory changed")
    external_anchor = receipt.get("external_trust_anchor")
    if external_anchor != {
        "closure_key": _ANCHOR_KEY,
        "repository_relative_path": DATA_IMPLEMENTATION_PATHS[_ANCHOR_KEY],
        "excluded_from_self_referential_receipt": True,
        "reason": "anchor_contains_this_receipt_file_and_logical_hash",
        "actual_file_sha256_reported_by_verifier_and_bound_before_GPU": True,
    }:
        raise TargetedRepairV4FreezeError("external anchor contract changed")
    for name in sorted(expected_implementations):
        relative = DATA_IMPLEMENTATION_PATHS[name]
        binding = implementations.get(name)
        if (
            not isinstance(binding, dict)
            or binding.get("repository_relative_path") != relative
            or not isinstance(binding.get("file_sha256"), str)
        ):
            raise TargetedRepairV4FreezeError("v4 closure binding is malformed")
        path = _ROOT / relative
        if path.is_symlink() or not path.is_file() or _sha(path) != binding["file_sha256"]:
            raise TargetedRepairV4FreezeError("v4 closure file identity changed")
    boundary = receipt.get("v3_diagnostic_only_boundary")
    if not isinstance(boundary, dict) or (
        boundary.get("denylist_file_sha256")
        != "58f650a366a431b620ff0e463883ed50d31702884712ffcacb0002538733576a"
        or boundary.get("retention_receipt_file_sha256")
        != "fa57b47dc5898c7ad142ae74dac66a076ad7d3c007c7b575f24c23a93bad3a2c"
        or boundary.get("retention_receipt_sha256")
        != "fba33cfd513f922d82cd04b8cf4d1c673656d64419f2bd60b59c49d49686889f"
        or boundary.get("diagnostic_file_sha256")
        != "40b23767e1d1989264d8b9374f1d45f93bb631ac01f993b8593da8f6e52e477d"
        or boundary.get("diagnostic_report_sha256")
        != "f2e91ec725bdfa61b2e5190f3d24ed78533d573520f5815763e9175c80fa207a"
        or boundary.get("v3_training_release_confirmatory_ancestry_parent_eligible") is not False
        or boundary.get("pre_freeze_amendment_file_sha256")
        != "eb57c59e84230e3ba82445dd422b56825482017c2f9655203c825e81b4df69f4"
        or boundary.get("pre_freeze_amendment_receipt_sha256")
        != "98fabdf6a7c497504e05923e6595e48d32a1462981bdbf69f90d24e3b6cbedc3"
        or boundary.get("superseded_recipe_file_sha256")
        != "34935246675353ba64c8d267542eaf64386a9992b771b869bad2fdfb88aab7fa"
    ):
        raise TargetedRepairV4FreezeError("v3 diagnostic-only boundary changed")
    amendment = receipt.get("pre_freeze_amendment")
    if not isinstance(amendment, dict) or amendment != {
        "amendment_id": "targeted_repair_v4_pre_freeze_recipe_amendment_v1",
        "repository_relative_path": (
            "reports/data_recipe_audits/targeted_repair_v4_pre_freeze_recipe_amendment_receipt.json"
        ),
        "file_sha256": "eb57c59e84230e3ba82445dd422b56825482017c2f9655203c825e81b4df69f4",
        "receipt_sha256": "98fabdf6a7c497504e05923e6595e48d32a1462981bdbf69f90d24e3b6cbedc3",
        "superseded_recipe_file_sha256": (
            "34935246675353ba64c8d267542eaf64386a9992b771b869bad2fdfb88aab7fa"
        ),
        "amended_before_freeze_materialization_or_GPU": True,
    }:
        raise TargetedRepairV4FreezeError("pre-freeze amendment binding changed")
    supersedes = receipt.get("supersedes")
    if not isinstance(supersedes, dict) or supersedes != {
        "withdrawn_receipt_id": "targeted_repair_v4_implementation_freeze_v3",
        "withdrawal_repository_relative_path": (
            "reports/data_recipe_audits/targeted_repair_v4_freeze_v3_recursive_closure_withdrawal.json"
        ),
        "withdrawal_file_sha256": _V3_WITHDRAWAL_FILE_SHA256,
        "withdrawal_receipt_sha256": _V3_WITHDRAWAL_LOGICAL_SHA256,
        "defect": "incomplete_recursive_import_and_parent_package_closure",
        "GPU_training_started": False,
    }:
        raise TargetedRepairV4FreezeError("v3 withdrawal binding changed")
    if _V3_WITHDRAWAL.is_symlink() or not _V3_WITHDRAWAL.is_file() or _sha(
        _V3_WITHDRAWAL
    ) != _V3_WITHDRAWAL_FILE_SHA256:
        raise TargetedRepairV4FreezeError("v3 withdrawal file identity changed")
    try:
        withdrawal = json.loads(_V3_WITHDRAWAL.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetedRepairV4FreezeError("v3 withdrawal is unreadable") from exc
    withdrawal_unsigned = dict(withdrawal)
    withdrawal_claimed = withdrawal_unsigned.pop("receipt_sha256", None)
    if (
        withdrawal_claimed != _V3_WITHDRAWAL_LOGICAL_SHA256
        or _canonical(withdrawal_unsigned) != withdrawal_claimed
        or withdrawal.get("status")
        != "withdrawn_before_GPU_due_to_incomplete_recursive_closure"
        or withdrawal.get("execution_boundary", {}).get("GPU_training_started") is not False
    ):
        raise TargetedRepairV4FreezeError("v3 withdrawal logical identity changed")
    base = receipt.get("base_model_restart")
    if not isinstance(base, dict) or (
        base.get("source_id") != "hfl/chinese-macbert-base"
        or base.get("revision") != "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
        or base.get("v3_model_or_checkpoint_parent_allowed") is not False
        or base.get("local_files_only") is not True
        or base.get("safe_serialization_only") is not True
        or base.get("trust_remote_code") is not False
        or base.get("files") != _BASE_MODEL_FILES
        or "local_directory" in base
    ):
        raise TargetedRepairV4FreezeError("base restart contract changed")
    return {
        "status": "verified",
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": claimed,
        "receipt_file_sha256": FREEZE_RECEIPT_FILE_SHA256,
        "recipe_file_sha256": recipe["file_sha256"],
        "implementation_count": len(implementations),
        "recursive_closure_file_count_including_external_anchor": len(DATA_IMPLEMENTATION_PATHS),
        "freeze_anchor_file_sha256": _sha(_ANCHOR),
        "v3_diagnostic_only_boundary": dict(boundary),
        "base_model_restart": {key: value for key, value in base.items() if key != "local_directory"},
    }


__all__ = ["TargetedRepairV4FreezeError", "verify_targeted_repair_v4_freeze"]
