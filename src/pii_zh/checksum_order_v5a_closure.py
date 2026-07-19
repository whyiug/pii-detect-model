"""Authoritative new-file closure layered over the verified v4 experiment."""

from types import MappingProxyType

V5A_IMPLEMENTATION_PATHS = MappingProxyType(
    {
        "pilot_runner": "scripts/train_macbert_checksum_bridge_v5_pilot.py",
        "bridge_builder": ("src/pii_zh/data/synthetic/targeted_repair_v5_checksum_bridge.py"),
        "bridge_auditor": ("src/pii_zh/data/synthetic/targeted_repair_v5_checksum_bridge_audit.py"),
        "bridge_materializer_cli": "scripts/materialize_targeted_repair_v5_checksum_bridge.py",
        "bridge_audit_cli": "scripts/audit_targeted_repair_v5_checksum_bridge.py",
        "data_recipe": "configs/data/targeted_repair_v5_checksum_context_bridge.yaml",
        "pilot_anchor": "src/pii_zh/training/checksum_bridge_v5_pilot_anchor.py",
        "pilot_unit_tests": "tests/unit/test_train_macbert_checksum_bridge_v5_pilot.py",
        "bridge_unit_tests": "tests/unit/data/test_targeted_repair_v5_checksum_bridge.py",
        "authoritative_v5a_closure": "src/pii_zh/checksum_order_v5a_closure.py",
    }
)
V5A_ANCHOR_KEY = "pilot_anchor"
V5A_RUNTIME_PATH_KEYS = frozenset({"pilot_runner", "pilot_anchor", "authoritative_v5a_closure"})
V5A_DATA_LINEAGE_PATH_KEYS = frozenset(
    {
        "bridge_builder",
        "bridge_auditor",
        "bridge_materializer_cli",
        "bridge_audit_cli",
        "data_recipe",
    }
)
V5A_EVIDENCE_PATH_KEYS = frozenset({"pilot_unit_tests", "bridge_unit_tests"})

__all__ = [
    "V5A_ANCHOR_KEY",
    "V5A_DATA_LINEAGE_PATH_KEYS",
    "V5A_EVIDENCE_PATH_KEYS",
    "V5A_IMPLEMENTATION_PATHS",
    "V5A_RUNTIME_PATH_KEYS",
]
