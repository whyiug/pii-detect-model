"""Run the rules-only cascade on reserved, synthetic example values."""

from __future__ import annotations

import json

from pii_zh.cascade import CascadeConfig, CascadePipeline


def main() -> None:
    pipeline = CascadePipeline(config=CascadeConfig(mode="rules-only"))
    text = "测试邮箱 demo@example.com，测试地址 203.0.113.7。"
    detections = pipeline.detect(text)
    print(
        json.dumps(
            {"detections": [item.to_dict() for item in detections]},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(json.dumps({"redacted_text": pipeline.redact(text)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
