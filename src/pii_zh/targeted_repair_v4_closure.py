"""Exact data and experiment source closures for targeted-repair v4."""

from types import MappingProxyType

_DATA = {
    "combined_isolation_audit_v3_dependency": "scripts/audit_targeted_repair_v3_combined_isolation.py",
    "combined_isolation_audit": "scripts/audit_targeted_repair_v4_combined_isolation.py",
    "design_auditor": "scripts/audit_targeted_repair_v4_design.py",
    "d0_decontamination_builder": "scripts/build_d0_train_structured_decontaminated_view.py",
    "materializer": "scripts/materialize_targeted_repair_v4.py",
    "validator": "scripts/validate_targeted_repair_v4.py",
    "package_init": "src/pii_zh/__init__.py",
    "data_init": "src/pii_zh/data/__init__.py",
    "data_alignment": "src/pii_zh/data/alignment.py",
    "artifact_policy": "src/pii_zh/data/artifact_policy.py",
    "data_schema": "src/pii_zh/data/schema.py",
    "data_splitting": "src/pii_zh/data/splitting.py",
    "synthetic_init": "src/pii_zh/data/synthetic/__init__.py",
    "synthetic_allocation": "src/pii_zh/data/synthetic/allocation.py",
    "synthetic_generator": "src/pii_zh/data/synthetic/generator.py",
    "synthetic_materialize": "src/pii_zh/data/synthetic/materialize.py",
    "generator": "src/pii_zh/data/synthetic/targeted_repair_v4.py",
    "v4_data_freeze_verifier": "src/pii_zh/data/synthetic/targeted_repair_v4_freeze.py",
    "v4_data_freeze_anchor": "src/pii_zh/data/synthetic/targeted_repair_v4_freeze_anchor.py",
    "synthetic_templates": "src/pii_zh/data/synthetic/templates.py",
    "value_generators": "src/pii_zh/data/synthetic/values.py",
    "validators_init": "src/pii_zh/data/validators/__init__.py",
    "models_init": "src/pii_zh/models/__init__.py",
    "decoder": "src/pii_zh/models/decoding.py",
    "models_qwen3_bi": "src/pii_zh/models/qwen3_bi.py",
    "models_qwen3_jpt": "src/pii_zh/models/qwen3_jpt.py",
    "result_closure": "src/pii_zh/targeted_repair_v4_closure.py",
    "taxonomy_init": "src/pii_zh/taxonomy/__init__.py",
    "taxonomy_loader": "src/pii_zh/taxonomy/loader.py",
    "tokenization": "src/pii_zh/tokenization.py",
    "training_init": "src/pii_zh/training/__init__.py",
    "training_config": "src/pii_zh/training/config.py",
    "training_data": "src/pii_zh/training/data.py",
    "training_initialization": "src/pii_zh/training/initialization.py",
    "training_loading": "src/pii_zh/training/loading.py",
    "training_manifest": "src/pii_zh/training/manifest.py",
    "training_pipeline": "src/pii_zh/training/pipeline.py",
    "safe_conversion": "src/pii_zh/training/safe_conversion.py",
    "training_trainer": "src/pii_zh/training/trainer.py",
}

DATA_IMPLEMENTATION_PATHS = MappingProxyType(dict(_DATA))
EXPERIMENT_IMPLEMENTATION_PATHS = MappingProxyType(
    {
        **_DATA,
        "data_freeze_anchor": "src/pii_zh/data/synthetic/targeted_repair_v4_freeze_anchor.py",
        "decoder": "src/pii_zh/models/decoding.py",
        "evaluation_metrics": "src/pii_zh/evaluation/metrics.py",
        "experiment_freeze_verifier": "src/pii_zh/training/targeted_repair_v4_experiment_freeze.py",
        "targeted_repair_v4_trainer": "scripts/train_macbert_targeted_repair_v4.py",
        "tokenization": "src/pii_zh/tokenization.py",
        "training_data": "src/pii_zh/training/data.py",
        "training_initialization": "src/pii_zh/training/initialization.py",
        "training_manifest": "src/pii_zh/training/manifest.py",
    }
)

__all__ = ["DATA_IMPLEMENTATION_PATHS", "EXPERIMENT_IMPLEMENTATION_PATHS"]
