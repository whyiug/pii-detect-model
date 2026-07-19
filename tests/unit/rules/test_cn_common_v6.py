from __future__ import annotations

from pii_zh.rules import CnCommonRulePack
from pii_zh.rules.cn_common_v6 import CN_COMMON_V6_SOURCE, CnCommonRulePackV6


def _surfaces(text: str, *, v6: bool = True) -> list[tuple[str, str, str]]:
    pack = CnCommonRulePackV6() if v6 else CnCommonRulePack()
    return [
        (match.entity_type, text[match.start : match.end], match.source)
        for match in pack.analyze(text, entities=["CN_VEHICLE_LICENSE_PLATE"])
    ]


def test_v6_accepts_validated_ordinary_and_new_energy_plate_subset() -> None:
    text = "登记车牌京A12345，新能源沪AD12345，大型苏A12345F已核验。"

    matches = CnCommonRulePackV6().analyze(
        text,
        entities=["CN_VEHICLE_LICENSE_PLATE"],
    )

    assert [text[match.start : match.end] for match in matches] == [
        "京A12345",
        "沪AD12345",
        "苏A12345F",
    ]
    assert all(match.pattern_id == "cn_vehicle_plate_strict_v6" for match in matches)
    assert all(match.source == CN_COMMON_V6_SOURCE for match in matches)
    assert all(match.metadata["validator_valid"] is True for match in matches)
    assert all(match.metadata["validator_label"] == "CN_VEHICLE_LICENSE_PLATE" for match in matches)


def test_v6_rejects_v5_high_recall_fallback_false_positives_and_special_suffixes() -> None:
    cases = (
        "车辆模型A12345已加载",
        "车辆版本岚A-ABC12345已发布",
        "车辆岚A·AB12CD3已核验",
        "车牌京I12345待核验",
        "车牌京A12345警待核验",
        "候选车牌京A12",
    )

    for text in cases:
        assert _surfaces(text) == [], text


def test_v5_history_remains_broad_while_v6_is_explicitly_narrower() -> None:
    text = "车辆岚A·AB12CD3已核验。"

    assert _surfaces(text, v6=False) == [
        ("CN_VEHICLE_LICENSE_PLATE", "岚A·AB12CD3", "rule:cn_common")
    ]
    assert _surfaces(text) == []
