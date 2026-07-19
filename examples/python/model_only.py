"""Load one manifest-bound local model and emit raw-text-free spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pii_zh.cascade import CascadeConfig, CascadePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve(strict=True)
    pipeline = CascadePipeline.from_pretrained(
        model_path,
        config=CascadeConfig(mode="model-only"),
        device="cpu",
        local_files_only=True,
    )
    detections = pipeline.detect("示例联系人位于虚构地址。")
    print(
        json.dumps(
            {"detections": [item.to_dict() for item in detections]},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
