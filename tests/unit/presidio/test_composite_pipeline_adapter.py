from __future__ import annotations

from collections.abc import Sequence

from pii_zh.cascade import CascadeConfig, CascadeDetection
from pii_zh.cascade.primary_augmentation import PrimaryPreservingAugmentationPipeline
from pii_zh.presidio import CascadeRecognizer, create_analyzer_engine


def _detection(entity_type: str, start: int, end: int, source: str) -> CascadeDetection:
    return CascadeDetection(
        start=start,
        end=end,
        entity_type=entity_type,
        score=0.9,
        source=source,
        sources=(source,),
        decision_process=(f"candidate:{source}",),
    )


class _StaticBatch:
    def __init__(self, output: list[CascadeDetection]) -> None:
        self.output = output
        self.config = CascadeConfig(mode="cascade")
        self.model_identity = None

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> list[list[CascadeDetection]]:
        del entities
        return [list(self.output) for _text in texts]


def _composite() -> PrimaryPreservingAugmentationPipeline:
    return PrimaryPreservingAugmentationPipeline(
        primary=_StaticBatch([_detection("PERSON", 0, 2, "primary")]),
        augmentation=_StaticBatch([_detection("DEVICE_ID", 3, 8, "augmentation")]),
        config=CascadeConfig(
            profile_version="community-presidio-bie73-cascade-v1",
            mode="cascade",
        ),
        model_identity={"profile": "community-presidio-bie73-cascade-v1"},
    )


def test_cascade_recognizer_accepts_structural_composite_pipeline() -> None:
    recognizer = CascadeRecognizer(_composite())

    results = recognizer.analyze("张三 dev-1", entities=None)

    assert [(item.entity_type, item.start, item.end) for item in results] == [
        ("PERSON", 0, 2),
        ("DEVICE_ID", 3, 8),
    ]


def test_analyzer_engine_accepts_the_same_composite_pipeline() -> None:
    analyzer = create_analyzer_engine(cascade_pipeline=_composite())

    results = analyzer.analyze(text="张三 dev-1", language="zh")

    assert [(item.entity_type, item.start, item.end) for item in results] == [
        ("PERSON", 0, 2),
        ("DEVICE_ID", 3, 8),
    ]
