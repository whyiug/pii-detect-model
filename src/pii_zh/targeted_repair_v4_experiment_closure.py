"""Authoritative transitive runtime and evidence closure for the v4 experiment.

The older ``EXPERIMENT_IMPLEMENTATION_PATHS`` symbol in the data-closure module
is a superseded pre-implementation sketch.  It is retained byte-for-byte because
that module belongs to the frozen data lineage.  Only the mapping below is
authoritative for experiment freeze and launch.
"""

from types import MappingProxyType

from .targeted_repair_v4_closure import DATA_IMPLEMENTATION_PATHS

_EXPERIMENT_ADDITIONS = {
    "base_macbert_trainer": "scripts/train_macbert_24_pilot.py",
    "targeted_repair_engine": "scripts/train_macbert_targeted_repair_pilot.py",
    "targeted_repair_v4_trainer": "scripts/train_macbert_targeted_repair_v4.py",
    "calibration_init": "src/pii_zh/calibration/__init__.py",
    "calibration_filtering": "src/pii_zh/calibration/filtering.py",
    "calibration_span_thresholds": "src/pii_zh/calibration/span_thresholds.py",
    "calibration_temperature": "src/pii_zh/calibration/temperature.py",
    "calibration_thresholds": "src/pii_zh/calibration/thresholds.py",
    "evaluation_init": "src/pii_zh/evaluation/__init__.py",
    "evaluation_baseline_reporting": "src/pii_zh/evaluation/baseline_reporting.py",
    "evaluation_io": "src/pii_zh/evaluation/io.py",
    "evaluation_metrics": "src/pii_zh/evaluation/metrics.py",
    "evaluation_provenance": "src/pii_zh/evaluation/provenance.py",
    "fusion_init": "src/pii_zh/fusion/__init__.py",
    "fusion_deterministic": "src/pii_zh/fusion/deterministic.py",
    "fusion_predictions": "src/pii_zh/fusion/predictions.py",
    "fusion_refinement": "src/pii_zh/fusion/refinement.py",
    "fusion_replacement": "src/pii_zh/fusion/replacement.py",
    "experiment_freeze_verifier": ("src/pii_zh/training/targeted_repair_v4_experiment_freeze.py"),
    "experiment_freeze_anchor": (
        "src/pii_zh/training/targeted_repair_v4_experiment_freeze_anchor.py"
    ),
    "authoritative_experiment_closure": ("src/pii_zh/targeted_repair_v4_experiment_closure.py"),
    "taxonomy_runtime_asset": "src/pii_zh/taxonomy/taxonomy.yaml",
    "trainer_unit_tests": "tests/unit/test_train_macbert_targeted_repair_v4.py",
}

EXPERIMENT_IMPLEMENTATION_PATHS = MappingProxyType(
    {**dict(DATA_IMPLEMENTATION_PATHS), **_EXPERIMENT_ADDITIONS}
)
EVIDENCE_ONLY_PATH_KEYS = frozenset({"trainer_unit_tests"})
DATA_LINEAGE_ONLY_PATH_KEYS = frozenset(
    {
        "combined_isolation_audit_v3_dependency",
        "combined_isolation_audit",
        "design_auditor",
        "d0_decontamination_builder",
        "materializer",
        "validator",
    }
)
RUNTIME_PATH_KEYS = (
    frozenset(EXPERIMENT_IMPLEMENTATION_PATHS)
    - EVIDENCE_ONLY_PATH_KEYS
    - DATA_LINEAGE_ONLY_PATH_KEYS
)
EXPERIMENT_ANCHOR_KEY = "experiment_freeze_anchor"

__all__ = [
    "DATA_LINEAGE_ONLY_PATH_KEYS",
    "EVIDENCE_ONLY_PATH_KEYS",
    "EXPERIMENT_ANCHOR_KEY",
    "EXPERIMENT_IMPLEMENTATION_PATHS",
    "RUNTIME_PATH_KEYS",
]
