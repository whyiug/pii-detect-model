#!/usr/bin/env python3
"""Materialize the immutable validation-informed v2 development split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.synthetic.sota_v2_resplit import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    materialize_synthetic_sota_v2_resplit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = materialize_synthetic_sota_v2_resplit(
        args.output_dir,
        source_dir=args.source_dir,
        config_path=args.config,
        repository_root=_REPOSITORY_ROOT,
    )
    print(
        f"dataset_id={manifest['dataset_id']} "
        f"train={manifest['counts']['by_split']['train']} "
        f"validation={manifest['counts']['by_split']['validation']} "
        f"manifest_sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
