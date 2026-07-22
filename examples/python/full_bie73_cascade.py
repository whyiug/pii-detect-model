"""Run the evaluated full-BIE73 rule/model cascade with explicit policy inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pii_zh.full_bie73 import (
    FULL_BIE73_ADAPTIVE_RULE_POLICIES,
    build_full_bie73_service_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--scope", choices=("open24", "closed8"), required=True)
    parser.add_argument(
        "--rule-policy",
        choices=FULL_BIE73_ADAPTIVE_RULE_POLICIES,
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pipeline = build_full_bie73_service_pipeline(
        args.model_path.expanduser(),
        rule_policy=args.rule_policy,
        thresholds=args.thresholds.expanduser(),
        scope=args.scope,
        mode="cascade",
        device=args.device,
    )
    text = "测试邮箱 demo@example.com，测试地址 203.0.113.7。"
    detections = pipeline.detect(text)
    print(
        json.dumps(
            {
                "detections": [item.to_dict() for item in detections],
                "profile": dict(pipeline.model_identity or {}),
                "redacted_text": pipeline.redact(text),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
