"""Authoritative new-file closure for the v5b numeric-O loss pilot."""

from types import MappingProxyType

V5B_IMPLEMENTATION_PATHS = MappingProxyType(
    {
        "pilot_runner": "scripts/train_macbert_numeric_o_v5b_pilot.py",
        "pilot_anchor": "src/pii_zh/training/numeric_o_v5b_pilot_anchor.py",
        "pilot_unit_tests": "tests/unit/test_train_macbert_numeric_o_v5b_pilot.py",
        "loss_scope_auditor": "src/pii_zh/training/numeric_o_loss_scope_audit.py",
        "loss_scope_audit_cli": "scripts/audit_macbert_numeric_o_v5b_loss_scope.py",
        "loss_scope_unit_tests": "tests/unit/training/test_numeric_o_loss_scope_audit.py",
        "authoritative_v5b_closure": "src/pii_zh/checksum_numeric_o_v5b_closure.py",
    }
)
V5B_ANCHOR_KEY = "pilot_anchor"
V5B_RUNTIME_PATH_KEYS = frozenset({"pilot_runner", "pilot_anchor", "authoritative_v5b_closure"})
V5B_AUDIT_IMPLEMENTATION_PATH_KEYS = frozenset({"loss_scope_auditor", "loss_scope_audit_cli"})
V5B_EVIDENCE_PATH_KEYS = frozenset({"pilot_unit_tests", "loss_scope_unit_tests"})

__all__ = [
    "V5B_ANCHOR_KEY",
    "V5B_AUDIT_IMPLEMENTATION_PATH_KEYS",
    "V5B_EVIDENCE_PATH_KEYS",
    "V5B_IMPLEMENTATION_PATHS",
    "V5B_RUNTIME_PATH_KEYS",
]
