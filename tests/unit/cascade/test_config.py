from __future__ import annotations

import pytest

from pii_zh.cascade import CascadeConfig, EntityRoute, canonicalize_entity_type


def test_config_mapping_rejects_unknown_fields_and_duplicate_routes() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        CascadeConfig.from_mapping({"mode": "rules-only", "modle_source": "typo"})

    route = EntityRoute("PERSON", "semantic", model_enabled=True)
    with pytest.raises(ValueError, match="duplicate"):
        CascadeConfig(routes=(route, route))


def test_structured_model_route_requires_an_explicit_validator() -> None:
    with pytest.raises(ValueError, match="named validator"):
        EntityRoute(
            "PASSPORT_NUMBER",
            "structured",
            rule_enabled=False,
            model_enabled=True,
        )

    route = EntityRoute.from_mapping(
        {
            "entity_type": "PASSPORT_NUMBER",
            "category": "structured",
            "rule_enabled": False,
            "model_enabled": True,
            "validator_label": "PASSPORT_NUMBER",
            "model_requires_validator": True,
        }
    )
    assert route.validator_label == "PASSPORT_NUMBER"


def test_short_span_threshold_is_never_weaker_than_base_threshold() -> None:
    with pytest.raises(ValueError, match="must not be below"):
        EntityRoute(
            "PERSON",
            "semantic",
            model_enabled=True,
            model_threshold=0.9,
            short_span_max_chars=2,
            short_model_threshold=0.8,
        )

    route = EntityRoute(
        "PERSON",
        "semantic",
        model_enabled=True,
        model_threshold=0.5,
        short_span_max_chars=2,
        short_model_threshold=0.98,
    )
    assert not route.evaluate(branch="model", score=0.97, span_length=2).accepted
    assert route.evaluate(branch="model", score=0.97, span_length=3).accepted


def test_model_taxonomy_aliases_normalize_to_policy_facing_names() -> None:
    assert canonicalize_entity_type("PERSON_NAME") == "PERSON"
    assert canonicalize_entity_type("ADDRESS") == "CN_ADDRESS"
    assert canonicalize_entity_type("CN_RESIDENT_ID") == "CN_ID_CARD"

    with pytest.raises(ValueError, match="canonical"):
        EntityRoute("PERSON_NAME", "semantic", model_enabled=True)


@pytest.mark.parametrize("mode", ["rules", "hybrid", "CASCADE", ""])
def test_config_mode_is_fail_closed(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        CascadeConfig(mode=mode)  # type: ignore[arg-type]
