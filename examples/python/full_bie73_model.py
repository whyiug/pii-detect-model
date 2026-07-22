"""Run the validated model-only BIE73 service ablation.

This path skips rules but retains service thresholds, validators and fusion;
it is not the raw-model benchmark protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pii_zh.full_bie73 import build_full_bie73_service_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--scope", choices=("open24", "closed8"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pipeline = build_full_bie73_service_pipeline(
        args.model_path.expanduser(),
        thresholds=args.thresholds.expanduser(),
        scope=args.scope,
        mode="model-only",
        device=args.device,
    )
    text = "测试邮箱 demo@example.com，测试地址 203.0.113.7。"
    detections = pipeline.detect(text)
    print(
        json.dumps(
            {
                "detections": [item.to_dict() for item in detections],
                "profile": dict(pipeline.model_identity or {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
