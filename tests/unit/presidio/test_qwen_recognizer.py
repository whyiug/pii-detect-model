from __future__ import annotations

from pathlib import Path

import pytest

from pii_zh.calibration import CalibrationBundle
from pii_zh.presidio import LocalModelBundle, QwenPiiRecognizer


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


class FakePredictor:
    def predict_batch(self, texts: list[str]) -> list[list[dict[str, object]]]:
        outputs: list[list[dict[str, object]]] = []
        for text in texts:
            predictions: list[dict[str, object]] = []
            start = text.find("张三")
            if start >= 0:
                predictions.append(
                    {
                        "entity": "B-PERSON_NAME",
                        "start": start,
                        "end": start + 2,
                        "score": 0.90,
                        "boundary_refined": True,
                    }
                )
            ignored = text.find("订单")
            if ignored >= 0:
                predictions.append(
                    {
                        "entity": "B-ORDER_ID",
                        "start": ignored,
                        "end": ignored + 2,
                        "score": 0.99,
                    }
                )
            outputs.append(predictions)
        return outputs


def _result_signature(results: list[object]) -> list[tuple[object, ...]]:
    return [
        (result.entity_type, result.start, result.end, result.score)  # type: ignore[attr-defined]
        for result in results
    ]


def test_cross_window_dedup_label_mapping_calibration_and_metadata() -> None:
    calibration = CalibrationBundle(
        entity_temperatures={"PERSON": 2.0},
        entity_thresholds={"PERSON": 0.70},
        default_threshold=0.95,
    )
    recognizer = QwenPiiRecognizer(
        predictor=FakePredictor(),
        tokenizer=CharacterTokenizer(),
        label_mapping={"PERSON_NAME": "PERSON"},
        ignored_labels=["ORDER_ID"],
        calibration=calibration,
        max_tokens=8,
        model_version="fake-v1",
        attention_mode="bidirectional",
    )
    text = "甲乙丙丁张三戊己订单庚辛"

    results = recognizer.analyze(text, ["PERSON"])

    assert len(results) == 1
    result = results[0]
    assert (result.entity_type, result.start, result.end) == ("PERSON", 4, 6)
    assert result.score == pytest.approx(0.75)
    assert result.recognition_metadata["raw_score"] == 0.90
    assert result.recognition_metadata["model_version"] == "fake-v1"
    assert result.recognition_metadata["boundary_refined"] is True


def test_analyze_batch_has_single_item_parity() -> None:
    recognizer = QwenPiiRecognizer(
        predictor=FakePredictor(),
        tokenizer=CharacterTokenizer(),
        label_mapping={"PERSON_NAME": "PERSON"},
        ignored_labels=["ORDER_ID"],
        max_tokens=8,
    )
    texts = ["甲乙张三", "无个人信息", "张三订单"]

    batched = recognizer.analyze_batch(texts, ["PERSON"])
    singles = [recognizer.analyze(text, ["PERSON"]) for text in texts]

    assert [_result_signature(results) for results in batched] == [
        _result_signature(results) for results in singles
    ]


def test_custom_local_loader_is_local_and_pluggable(tmp_path: Path) -> None:
    model_path = tmp_path / "custom-model"
    model_path.mkdir()
    seen: list[Path] = []

    def loader(path: Path) -> LocalModelBundle:
        seen.append(path)
        return LocalModelBundle(FakePredictor(), CharacterTokenizer(), {"loader": "fake"})

    recognizer = QwenPiiRecognizer(
        model_path=model_path,
        model_loader=loader,
        label_mapping={"PERSON_NAME": "PERSON"},
        max_tokens=8,
    )

    result = recognizer.analyze("张三", ["PERSON"])[0]
    assert seen == [model_path]
    assert result.recognition_metadata["loader"] == "fake"
