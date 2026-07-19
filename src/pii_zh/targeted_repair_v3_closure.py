"""Exact result-determining repository closure for targeted-repair-v3.

Receipt-anchor modules are handled separately.  The data anchor is bound by
the later experiment receipt; the experiment anchor is bound by the external
pre-GPU approval record.  Including an anchor in the receipt whose content
hash it stores would create a self-referential hash cycle.
"""

from types import MappingProxyType

_RESULT_IMPLEMENTATIONS = {
    "artifact_policy": "src/pii_zh/data/artifact_policy.py",
    "base_trainer": "scripts/train_macbert_24_pilot.py",
    "combined_isolation_audit": "scripts/audit_targeted_repair_v3_combined_isolation.py",
    "d0_decontamination_builder": ("scripts/build_d0_train_structured_decontaminated_view.py"),
    "data_alignment": "src/pii_zh/data/alignment.py",
    "data_init": "src/pii_zh/data/__init__.py",
    "data_schema": "src/pii_zh/data/schema.py",
    "data_splitting": "src/pii_zh/data/splitting.py",
    "decoder": "src/pii_zh/models/decoding.py",
    "design_auditor": "scripts/audit_targeted_repair_v3_design.py",
    "evaluation_baseline_reporting": "src/pii_zh/evaluation/baseline_reporting.py",
    "evaluation_init": "src/pii_zh/evaluation/__init__.py",
    "evaluation_io": "src/pii_zh/evaluation/io.py",
    "evaluation_metrics": "src/pii_zh/evaluation/metrics.py",
    "evaluation_provenance": "src/pii_zh/evaluation/provenance.py",
    "experiment_freeze_verifier": ("src/pii_zh/training/targeted_repair_v3_experiment_freeze.py"),
    "generator": "src/pii_zh/data/synthetic/targeted_repair_v3.py",
    "materializer": "scripts/materialize_targeted_repair_v3.py",
    "models_init": "src/pii_zh/models/__init__.py",
    "models_qwen3_bi": "src/pii_zh/models/qwen3_bi.py",
    "models_qwen3_jpt": "src/pii_zh/models/qwen3_jpt.py",
    "package_init": "src/pii_zh/__init__.py",
    "result_closure": "src/pii_zh/targeted_repair_v3_closure.py",
    "safe_conversion": "src/pii_zh/training/safe_conversion.py",
    "synthetic_allocation": "src/pii_zh/data/synthetic/allocation.py",
    "synthetic_generator": "src/pii_zh/data/synthetic/generator.py",
    "synthetic_init": "src/pii_zh/data/synthetic/__init__.py",
    "synthetic_materialize": "src/pii_zh/data/synthetic/materialize.py",
    "synthetic_templates": "src/pii_zh/data/synthetic/templates.py",
    "taxonomy_config": "src/pii_zh/taxonomy/taxonomy.yaml",
    "taxonomy_init": "src/pii_zh/taxonomy/__init__.py",
    "taxonomy_loader": "src/pii_zh/taxonomy/loader.py",
    "tokenization": "src/pii_zh/tokenization.py",
    "training_config": "src/pii_zh/training/config.py",
    "training_data": "src/pii_zh/training/data.py",
    "training_engine": "scripts/train_macbert_targeted_repair_pilot.py",
    "training_init": "src/pii_zh/training/__init__.py",
    "training_initialization": "src/pii_zh/training/initialization.py",
    "training_loading": "src/pii_zh/training/loading.py",
    "training_manifest": "src/pii_zh/training/manifest.py",
    "training_pipeline": "src/pii_zh/training/pipeline.py",
    "training_trainer": "src/pii_zh/training/trainer.py",
    "v3_data_freeze_verifier": ("src/pii_zh/data/synthetic/targeted_repair_v3_freeze.py"),
    "v3_trainer": "scripts/train_macbert_targeted_repair_v3.py",
    "validator": "scripts/validate_targeted_repair_v3.py",
    "validators_init": "src/pii_zh/data/validators/__init__.py",
    "value_generators": "src/pii_zh/data/synthetic/values.py",
}

DATA_IMPLEMENTATION_PATHS = MappingProxyType(dict(_RESULT_IMPLEMENTATIONS))
EXPERIMENT_IMPLEMENTATION_PATHS = MappingProxyType(
    {
        **_RESULT_IMPLEMENTATIONS,
        "v3_data_freeze_anchor": ("src/pii_zh/data/synthetic/targeted_repair_v3_freeze_anchor.py"),
    }
)

__all__ = ["DATA_IMPLEMENTATION_PATHS", "EXPERIMENT_IMPLEMENTATION_PATHS"]
