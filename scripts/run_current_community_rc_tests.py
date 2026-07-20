#!/usr/bin/env python3
"""Run the self-contained public community-RC regression allowlist.

The immutable local verification receipt covers a larger historical evidence
tree.  A public GitHub checkout intentionally omits private-path build records,
record-level data, and superseded frozen gates, so hosted CI uses this explicit
product/release surface instead of depending on an unpublished working tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PUBLIC_CLEAN_CHECKOUT_TEST_PATHS = (
    "tests/cli/test_cli.py",
    "tests/service/test_local_service.py",
    "tests/unit/calibration/test_calibration.py",
    "tests/unit/calibration/test_span_thresholds.py",
    "tests/unit/cascade/test_ablation_confirmatory_contracts.py",
    "tests/unit/cascade/test_ablation_profiles.py",
    "tests/unit/cascade/test_community_full24.py",
    "tests/unit/cascade/test_config.py",
    "tests/unit/cascade/test_import_boundary.py",
    "tests/unit/cascade/test_pipeline.py",
    "tests/unit/cascade/test_redaction_coverage.py",
    "tests/unit/cascade/test_result.py",
    "tests/unit/cascade/test_service_profiles.py",
    "tests/unit/fusion/test_deterministic.py",
    "tests/unit/presidio/test_context_enhancer.py",
    "tests/unit/presidio/test_qwen_recognizer.py",
    "tests/unit/presidio/test_token_chunker.py",
    "tests/unit/rules/test_cn_common.py",
    "tests/unit/rules/test_cn_common_v6.py",
    "tests/unit/rules/test_rules_refinement_cli.py",
    "tests/unit/test_community_release_v2_evidence.py",
    "tests/unit/test_community_release_v2_verifications.py",
    "tests/unit/test_collect_community_v2_remote_evidence_github.py",
    "tests/unit/test_collect_community_v2_remote_evidence_hf.py",
    "tests/unit/test_community_v2_github_release_assets_receipt.py",
    "tests/unit/test_community_v2_hf_snapshot_materialization.py",
    "tests/unit/test_community_v2_publication_package_verification.py",
    "tests/unit/test_community_v2_publication_receipt.py",
    "tests/unit/test_community_v2_publication_successor.py",
    "tests/unit/test_community_v2_post_attachment_verification.py",
    "tests/unit/test_benchmark_cascade.py",
    "tests/unit/test_build_service_quality_suite.py",
    "tests/unit/test_cascade_ablation_cli.py",
    "tests/unit/test_evaluate_cascade.py",
    "tests/unit/test_evaluation.py",
    "tests/unit/test_inference.py",
    "tests/unit/test_inference_postprocessing.py",
    "tests/unit/test_service_eval_cli.py",
    "tests/unit/test_service_quality_suite.py",
    "tests/unit/test_taxonomy.py",
    "tests/unit/test_tokenization.py",
    "tests/unit/test_release_eval_v2_candidate_generator.py",
    "tests/unit/test_release_eval_v2_stage_gate.py",
    "tests/unit/test_render_community_v2_publication_model_card.py",
    "tests/unit/test_render_community_v2_release_notes.py",
    "tests/unit/test_tested_private_security_channel_receipt.py",
    "tests/unit/data/test_synthetic_sota_v2_resplit.py",
    "tests/unit/data/test_validators.py",
    "tests/unit/data/test_windowing.py",
    "tests/unit/models/test_aiguard24.py",
    "tests/release/test_public_scan_and_sbom.py",
    "tests/release/test_templates.py",
)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *PUBLIC_CLEAN_CHECKOUT_TEST_PATHS],
        cwd=repository_root,
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
