#!/usr/bin/env python3
"""Materialize the immutable clean-room domain-randomized v7.2 corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.synthetic.domain_randomized_v7_2 import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    materialize_domain_randomized_v7_2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = materialize_domain_randomized_v7_2(
        args.output_dir,
        config_path=args.config,
        repository_root=_REPOSITORY_ROOT,
    )
    counts = manifest["counts"]["by_split"]
    print(
        f"dataset_id={manifest['dataset_id']} "
        f"version={manifest['dataset_version']} "
        f"train={counts['train']} dev={counts['dev']} "
        f"independent_test={counts['independent_test']} "
        f"pairwise_collisions="
        f"{manifest['audits']['isolation']['pairwise_collision_total']} "
        f"test_seal_sha256="
        f"{manifest['independent_test_seal']['seal_sha256']} "
        f"manifest_sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
