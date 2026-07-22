from __future__ import annotations

from collections.abc import Sequence

import pytest

from pii_zh.cascade.config import CascadeConfig
from pii_zh.cascade.primary_augmentation import PrimaryPreservingAugmentationPipeline
from pii_zh.cascade.result import CascadeDetection


def _detection(
    entity_type: str,
    start: int,
    end: int,
    *,
    source: str,
    score: float = 0.9,
) -> CascadeDetection:
    return CascadeDetection(
        start=start,
        end=end,
        entity_type=entity_type,
        score=score,
        source=source,
        sources=(source,),
        decision_process=(f"candidate:{source}",),
    )


class StaticBatchPipeline:
    def __init__(self, by_text: dict[str, list[CascadeDetection]]) -> None:
        self.by_text = by_text
        self.calls: list[tuple[tuple[str, ...], Sequence[str] | None]] = []
        self.config = CascadeConfig(mode="cascade")
        self.model_identity: dict[str, str] | None = None

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> list[list[CascadeDetection]]:
        self.calls.append((tuple(texts), entities))
        return [list(self.by_text.get(text, ())) for text in texts]


class FailingBatchPipeline:
    def __init__(self, message: str) -> None:
        self.message = message
        self.config = CascadeConfig(mode="cascade")
        self.model_identity: dict[str, str] | None = None

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> list[list[CascadeDetection]]:
        del texts, entities
        raise RuntimeError(self.message)


def test_primary_is_unchanged_and_owns_every_overlap() -> None:
    text = "0123456789ABCDEFGHIJ"
    authoritative = _detection("PERSON", 5, 10, source="primary", score=0.1)
    before = _detection("USERNAME", 0, 2, source="augmentation")
    adjacent = _detection("DEVICE_ID", 10, 12, source="augmentation")
    after = _detection("SECRET", 15, 18, source="augmentation")
    augmentation = [
        before,
        _detection("EMAIL_ADDRESS", 5, 10, source="augmentation", score=1.0),
        _detection("CN_ADDRESS", 4, 6, source="augmentation", score=1.0),
        _detection("PHONE_NUMBER", 6, 9, source="augmentation", score=1.0),
        _detection("BANK_CARD_NUMBER", 9, 11, source="augmentation", score=1.0),
        adjacent,
        after,
    ]
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=StaticBatchPipeline({text: [authoritative]}),
        augmentation=StaticBatchPipeline({text: augmentation}),
    )

    detections = pipeline.detect(text)

    assert detections == [before, authoritative, adjacent, after]
    assert detections[1] is authoritative
    assert authoritative.score == 0.1
    assert authoritative.decision_process == ("candidate:primary",)


def test_closed8_projection_is_exactly_the_frozen_primary_output() -> None:
    text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    closed8 = (
        "CN_ADDRESS",
        "BANK_CARD_NUMBER",
        "CN_ID_CARD",
        "EMAIL_ADDRESS",
        "PASSPORT_NUMBER",
        "PERSON",
        "PHONE_NUMBER",
        "CN_VEHICLE_LICENSE_PLATE",
    )
    primary_output = [
        _detection(entity, index * 2, index * 2 + 1, source="primary", score=0.1 + index / 10)
        for index, entity in enumerate(closed8)
    ]
    augmentation_output = [
        # Every overlap must lose even with a higher augmentation score.
        _detection("SECRET", item.start, item.end, source="augmentation", score=1.0)
        for item in primary_output
    ]
    augmentation_output.append(
        _detection("DEVICE_ID", 20, 24, source="augmentation", score=1.0)
    )
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=StaticBatchPipeline({text: primary_output}),
        augmentation=StaticBatchPipeline({text: augmentation_output}),
    )

    projected = pipeline.detect(text, entities=closed8)

    assert projected == primary_output
    assert all(
        observed is expected
        for observed, expected in zip(projected, primary_output, strict=True)
    )


def test_augmentation_results_are_not_refused_with_each_other() -> None:
    text = "abcdefghij"
    outer = _detection("CN_ADDRESS", 1, 8, source="augmentation")
    inner = _detection("PERSON", 3, 5, source="augmentation")
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=StaticBatchPipeline({text: []}),
        augmentation=StaticBatchPipeline({text: [outer, inner]}),
    )

    assert pipeline.detect(text) == [outer, inner]


def test_entity_filter_is_canonical_final_filter_and_is_not_pushed_down() -> None:
    text = "张三的账号user-7"
    person = _detection("PERSON", 0, 2, source="primary")
    username = _detection("USERNAME", 5, len(text), source="augmentation")
    primary = StaticBatchPipeline({text: [person]})
    augmentation = StaticBatchPipeline({text: [username]})
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=primary,
        augmentation=augmentation,
    )

    complete = pipeline.detect(text)
    filtered = pipeline.detect(text, entities=["PERSON_NAME"])

    assert filtered == [item for item in complete if item.entity_type == "PERSON"]
    assert primary.calls == [((text,), None), ((text,), None)]
    assert augmentation.calls == [((text,), None), ((text,), None)]


def test_filter_does_not_resurrect_augmentation_hidden_by_filtered_primary() -> None:
    text = "张三"
    person = _detection("PERSON", 0, 2, source="primary")
    overlapping_secret = _detection("SECRET", 0, 2, source="augmentation")
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=StaticBatchPipeline({text: [person]}),
        augmentation=StaticBatchPipeline({text: [overlapping_secret]}),
    )

    complete = pipeline.detect(text)

    assert complete == [person]
    assert pipeline.detect(text, entities=["SECRET"]) == [
        item for item in complete if item.entity_type == "SECRET"
    ]


def test_single_and_batch_share_the_same_child_batch_path() -> None:
    texts = ["张三", "设备dev-1"]
    person = _detection("PERSON", 0, 2, source="primary")
    device = _detection("DEVICE_ID", 2, len(texts[1]), source="augmentation")
    primary = StaticBatchPipeline({texts[0]: [person]})
    augmentation = StaticBatchPipeline({texts[1]: [device]})
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=primary,
        augmentation=augmentation,
    )

    singles = [pipeline.detect(text) for text in texts]
    batched = pipeline.detect_batch(texts)

    assert batched == singles == [[person], [device]]
    assert primary.calls[-1] == (tuple(texts), None)
    assert augmentation.calls[-1] == (tuple(texts), None)


@pytest.mark.parametrize("failing_branch", ["primary", "augmentation"])
def test_child_failure_is_not_converted_to_partial_output(failing_branch: str) -> None:
    healthy = StaticBatchPipeline(
        {"张三": [_detection("PERSON", 0, 2, source="healthy")]}
    )
    failing = FailingBatchPipeline(f"{failing_branch} failed")
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=failing if failing_branch == "primary" else healthy,
        augmentation=failing if failing_branch == "augmentation" else healthy,
    )

    with pytest.raises(RuntimeError, match=f"{failing_branch} failed"):
        pipeline.detect("张三")


def test_malformed_batch_shape_fails_instead_of_falling_back() -> None:
    class WrongLengthPipeline:
        def detect_batch(
            self,
            texts: Sequence[str],
            *,
            entities: Sequence[str] | None = None,
        ) -> list[list[CascadeDetection]]:
            del texts, entities
            return []

    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=WrongLengthPipeline(),
        augmentation=StaticBatchPipeline({}),
    )

    with pytest.raises(ValueError, match="primary pipeline must return one batch"):
        pipeline.detect("张三")


def test_redact_uses_only_composed_coverage_and_respects_final_filter() -> None:
    text = "张三的邮箱a@b.cn，设备dev-1"
    person_end = len("张三")
    email_start = text.index("a@b.cn")
    email_end = email_start + len("a@b.cn")
    device_start = text.index("dev-1")
    primary = StaticBatchPipeline(
        {
            text: [
                _detection("PERSON", 0, person_end, source="primary"),
                _detection("EMAIL_ADDRESS", email_start, email_end, source="primary"),
            ]
        }
    )
    augmentation = StaticBatchPipeline(
        {
            text: [
                # This higher-score overlap must not affect redaction coverage.
                _detection(
                    "SECRET",
                    email_start - 1,
                    email_end,
                    source="augmentation",
                    score=1.0,
                ),
                _detection(
                    "DEVICE_ID",
                    device_start,
                    len(text),
                    source="augmentation",
                ),
            ]
        }
    )
    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=primary,
        augmentation=augmentation,
    )

    assert pipeline.redact(
        text,
        {
            "PERSON": "<人名>",
            "EMAIL_ADDRESS": "<邮箱>",
            "DEVICE_ID": "<设备>",
        },
    ) == "<人名>的邮箱<邮箱>，设备<设备>"
    assert pipeline.redact(text, "<PII>", entities=["DEVICE_ID"]) == (
        "张三的邮箱a@b.cn，设备<PII>"
    )


def test_constructor_and_input_validation_fail_closed() -> None:
    healthy = StaticBatchPipeline({})
    with pytest.raises(TypeError, match="primary must expose detect_batch"):
        PrimaryPreservingAugmentationPipeline(primary=object(), augmentation=healthy)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="augmentation must expose detect_batch"):
        PrimaryPreservingAugmentationPipeline(primary=healthy, augmentation=object())  # type: ignore[arg-type]

    pipeline = PrimaryPreservingAugmentationPipeline(
        primary=healthy,
        augmentation=healthy,
    )
    with pytest.raises(TypeError, match="not one string"):
        pipeline.detect("text", entities="PERSON")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence of strings"):
        pipeline.detect_batch("text")  # type: ignore[arg-type]


def test_service_metadata_can_be_inherited_or_explicitly_overridden() -> None:
    primary = StaticBatchPipeline({})
    augmentation = StaticBatchPipeline({})
    augmentation.model_identity = {"model_id": "augmentation-v1"}
    inherited = PrimaryPreservingAugmentationPipeline(
        primary=primary,
        augmentation=augmentation,
    )

    assert inherited.config is augmentation.config
    assert inherited.model_identity == {"model_id": "augmentation-v1"}

    public_config = CascadeConfig(profile_version="primary-plus-augmentation-v1", mode="cascade")
    explicit = PrimaryPreservingAugmentationPipeline(
        primary=primary,
        augmentation=augmentation,
        config=public_config,
        model_identity={"profile": "primary-plus-augmentation-v1", "model_enabled": True},
    )

    assert explicit.config is public_config
    assert explicit.model_identity == {
        "profile": "primary-plus-augmentation-v1",
        "model_enabled": True,
    }
    with pytest.raises(TypeError, match="model_identity"):
        PrimaryPreservingAugmentationPipeline(
            primary=primary,
            augmentation=augmentation,
            model_identity={"nested": {"not": "flat"}},  # type: ignore[dict-item]
        )
