"""Run the formally selected full-BIE73 rule/model cascade.

The public defaults are Open-24, ``fpr_guarded_all6`` and fixed 0.5
thresholds.  Advanced Python integrations can override evaluated options
through :func:`build_full_bie73_service_pipeline` with strict validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pii_zh.full_bie73 import build_full_bie73_service_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scope", choices=("open24", "closed8"), default="open24")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pipeline = build_full_bie73_service_pipeline(
        args.model_path.expanduser(),
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
