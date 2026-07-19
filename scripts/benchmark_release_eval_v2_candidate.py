#!/usr/bin/env python3
"""Measure or exactly replay release-eval-v2 candidate performance."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.evaluation.release_eval_v2_performance import main


if __name__ == "__main__":
    raise SystemExit(main())
