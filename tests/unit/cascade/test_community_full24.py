from __future__ import annotations

from collections.abc import Sequence

import pytest

from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    CascadePipeline,
    load_service_config,
)
from pii_zh.cascade.routing import (
    COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES,
    community_full24_routes,
)
from pii_zh.data.validators import (
    COMMUNITY_STRUCTURED_VALIDATORS,
    cn_resident_id_check_code,
    luhn_check_digit,
)
from pii_zh.taxonomy import load_presidio_mapping, load_taxonomy


def _resident_id() -> str:
    first_seventeen = "99000019900101001"
    return first_seventeen + cn_resident_id_check_code(first_seventeen)


def _bank_card() -> str:
    without_check = "999999000000000"
    return without_check + luhn_check_digit(without_check)


CORE24_MODEL_FIXTURES: dict[str, str] = {
    "PERSON": "赵安",
    "PHONE_NUMBER": "19900000000",
    "EMAIL_ADDRESS": "pii-test@example.invalid",
    "CN_ADDRESS": "宁川省云汀市栖霞区清和路1号",
    "DATE_OF_BIRTH": "1990-01-01",
    "CN_ID_CARD": _resident_id(),
    "PASSPORT_NUMBER": "PZ00000000",
    "DRIVER_LICENSE_NUMBER": "DL0000000000",
    "CN_SOCIAL_SECURITY_NUMBER": "990000000000000001",
    "BANK_CARD_NUMBER": _bank_card(),
    "BANK_ACCOUNT_NUMBER": "990000000000000001",
    "CN_VEHICLE_LICENSE_PLATE": "岚A·ABC1234",
    "EMPLOYEE_ID": "E0000000001",
    "STUDENT_ID": "S000000000001",
    "MEDICAL_RECORD_NUMBER": "M000000000001",
    "WECHAT_ID": "wx_demo000001",
    "QQ_NUMBER": "900000000001",
    "ALIPAY_ACCOUNT": "pay-demo0001@example.invalid",
    "USERNAME": "u_demo000001",
    "IP_ADDRESS": "2001:db8::1",
    "MAC_ADDRESS": "02:00:00:00:00:01",
    "DEVICE_ID": "DEV-ABC00000000001",
    "GEO_COORDINATE": "0.123456,0.654321",
    "SECRET": "key_0123456789abcdefghjkmnp",
}


class _OneFixtureModel:
    def __init__(self, entity_type: str, value: str) -> None:
        self.entity_type = entity_type
        self.value = value
        self.requested: list[tuple[str, ...]] = []

    def analyze(self, text: str, entities: Sequence[str]) -> list[dict[str, object]]:
        self.requested.append(tuple(entities))
        if self.entity_type not in entities:
            return []
        start = text.index(self.value)
        return [
            {
                "entity_type": self.entity_type,
                "start": start,
                "end": start + len(self.value),
                "score": 1.0,
            }
        ]


def test_community_route_table_matches_exact_taxonomy_core24_outputs() -> None:
    taxonomy = load_taxonomy()
    mapping = load_presidio_mapping()
    expected = {
        mapping.model_to_presidio[entity.name]
        for entity in taxonomy.label_sets["core"]
    }
    config = load_service_config(
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="model-only",
    )

    assert len(expected) == 24
    assert set(config.route_map) == expected == set(CORE24_MODEL_FIXTURES)
    assert all(route.model_enabled for route in config.routes)
    assert all(
        route.category == "semantic"
        or (route.model_requires_validator and route.validator_label is not None)
        for route in config.routes
    )


def test_authoritative_calibration_disables_only_frozen_harmful_rule_branches() -> None:
    default = {route.entity_type: route for route in community_full24_routes()}
    calibrated = {
        route.entity_type: route
        for route in community_full24_routes(recognizer_thresholds_authoritative=True)
    }

    assert all(route.rule_enabled for route in default.values())
    assert {
        label for label, route in calibrated.items() if not route.rule_enabled
    } == COMMUNITY_CALIBRATED_RULE_DISABLED_ENTITIES
    assert all(route.model_enabled for route in calibrated.values())


@pytest.mark.parametrize(
    ("entity_type", "value"),
    tuple(CORE24_MODEL_FIXTURES.items()),
)
def test_every_core24_model_fixture_reaches_final_service_output(
    entity_type: str,
    value: str,
) -> None:
    config = load_service_config(
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="model-only",
    )
    model = _OneFixtureModel(entity_type, value)
    pipeline = CascadePipeline(
        config=config,
        model_recognizer=model,
        validators=COMMUNITY_STRUCTURED_VALIDATORS,
    )
    text = f"字段：{value}。"

    detections = pipeline.detect(text, entities=[entity_type])

    assert len(detections) == 1
    detection = detections[0]
    assert detection.entity_type == entity_type
    assert text[detection.start : detection.end] == value
    assert detection.source == "model:pii-zh-24"
    assert detection.sources == ("model:pii-zh-24",)
    route = config.route_map[entity_type]
    expected_step = (
        "validator:not_required" if route.category == "semantic" else "validator:passed"
    )
    assert expected_step in detection.decision_process
    assert "threshold:passed" in detection.decision_process
    assert model.requested == [(entity_type,)]


@pytest.mark.parametrize(
    ("validator_label", "sentinel", "common_format"),
    [
        ("PHONE_NUMBER", "19900000000", "+8619900000001"),
        ("EMAIL_ADDRESS", "pii@example.invalid", "demo@example.com"),
        ("CN_RESIDENT_ID", _resident_id(), _resident_id()),
        ("PASSPORT_NUMBER", "PZ00000000", "E00000000"),
        ("DRIVER_LICENSE_NUMBER", "DL0000000000", _resident_id()),
        ("CN_SOCIAL_SECURITY_NUMBER", "990000000000000001", "AB1234567890"),
        ("BANK_CARD_NUMBER", _bank_card(), _bank_card()),
        ("BANK_ACCOUNT_NUMBER", "990000000000000001", "6222000000000001"),
        ("CN_VEHICLE_LICENSE_PLATE", "岚A·ABC1234", "京A00000"),
        ("EMPLOYEE_ID", "E0000000001", "DPT-HR-00001234"),
        ("STUDENT_ID", "S000000000001", "202600000001"),
        ("MEDICAL_RECORD_NUMBER", "M000000000001", "MR-00000001"),
        ("WECHAT_ID", "wx_demo000001", "wechat_001"),
        ("QQ_NUMBER", "900000000001", "100001"),
        ("ALIPAY_ACCOUNT", "pay-demo0001@example.invalid", "19900000000"),
        ("IP_ADDRESS", "2001:db8::1", "192.0.2.1"),
        ("MAC_ADDRESS", "02:00:00:00:00:01", "02-00-00-00-00-02"),
        ("DEVICE_ID", "DEV-ABC00000000001", "123E4567-E89B-12D3-A456-426614174000"),
        ("GEO_COORDINATE", "0.123456,0.654321", "31.2304,121.4737"),
        ("SECRET", "key_0123456789abcdefghjkmnp", "sk" "-0123456789ABCDefghij"),
    ],
)
def test_community_structured_validators_cover_sentinels_and_common_formats(
    validator_label: str,
    sentinel: str,
    common_format: str,
) -> None:
    validator = COMMUNITY_STRUCTURED_VALIDATORS[validator_label]

    assert validator(sentinel).valid is True
    assert validator(common_format).valid is True
    assert validator("x").valid is False


def test_community_structured_validator_registry_is_exact_for_all_structured_routes() -> None:
    config = load_service_config(
        COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
        mode="cascade",
    )
    required = {
        route.validator_label
        for route in config.routes
        if route.category == "structured"
    }

    assert None not in required
    assert required == set(COMMUNITY_STRUCTURED_VALIDATORS)
