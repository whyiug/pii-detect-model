"""Run the same rules-only cascade through Presidio AnalyzerEngine."""

from __future__ import annotations

import json

from pii_zh.presidio import create_analyzer_engine


def main() -> None:
    analyzer = create_analyzer_engine()
    results = analyzer.analyze(
        text="测试邮箱 demo@example.com，测试地址 203.0.113.7。",
        language="zh",
    )
    payload = [
        {
            "start": item.start,
            "end": item.end,
            "entity_type": item.entity_type,
            "score": item.score,
            "sources": item.recognition_metadata.get("sources", []),
        }
        for item in results
    ]
    print(json.dumps({"detections": payload}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
