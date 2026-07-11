"""Reproducible placeholder-first synthetic document generation."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ..schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    stable_text_hash,
)
from ..validators import validate_entity_value
from .templates import (
    CURATED_ASSET_ID,
    CURATED_ASSET_METADATA,
    CURATED_ASSET_SHA256,
    CURATED_ASSET_VERSION,
    CURATION_AUDIT_SHA256,
    HARD_NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    SyntheticTemplate,
)
from .values import SyntheticValueFactory, generate_value

GENERATOR_NAME = "pii_zh_deterministic_synthetic"
GENERATOR_VERSION = "2.3.0"
DEFAULT_LICENSE = "Apache-2.0"
_PLACEHOLDER_RE = re.compile(r"<<([A-Z0-9_]+)>>")


class SyntheticGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RenderedTemplate:
    text: str
    entities: tuple[EntitySpan, ...]
    values: Mapping[str, str]
    validators_passed: bool
    validator_results: Mapping[str, str]


def render_template(
    template: SyntheticTemplate,
    values: Mapping[str, str],
) -> RenderedTemplate:
    """Replace placeholders while computing exact offsets in one forward pass."""

    specs = {spec.key: spec for spec in template.placeholders}
    output: list[str] = []
    cursor = 0
    output_length = 0
    entities: list[EntitySpan] = []
    validator_results: dict[str, str] = {}
    validators_passed = True

    for match in _PLACEHOLDER_RE.finditer(template.body):
        literal = template.body[cursor : match.start()]
        output.append(literal)
        output_length += len(literal)
        key = match.group(1)
        if key not in specs:
            raise SyntheticGenerationError(
                f"template {template.template_id} has undeclared placeholder {key}"
            )
        if key not in values:
            raise SyntheticGenerationError(f"missing value for placeholder {key}")
        value = values[key]
        if not isinstance(value, str) or not value:
            raise SyntheticGenerationError(f"placeholder {key} must have a non-empty string")
        start = output_length
        output.append(value)
        output_length += len(value)
        end = output_length
        spec = specs[key]
        if spec.label is not None:
            result = validate_entity_value(spec.label, value)
            state = "not_applicable" if result is None else ("passed" if result.valid else "failed")
            if result is not None:
                validator_results[f"{key}:{spec.label}"] = result.reason
                validators_passed = validators_passed and result.valid
            entities.append(
                EntitySpan(
                    start=start,
                    end=end,
                    label=spec.label,
                    text=value,
                    source="synthetic",
                    risk_tier=spec.risk_tier,
                    validator_state=state,
                    metadata={"synthetic": True, "placeholder": key},
                )
            )
        cursor = match.end()

    trailing = template.body[cursor:]
    output.append(trailing)
    rendered_text = "".join(output)
    declared = set(specs)
    used = set(_PLACEHOLDER_RE.findall(template.body))
    if declared != used:
        raise SyntheticGenerationError(
            f"template {template.template_id} placeholder declaration mismatch: "
            f"declared={sorted(declared)}, used={sorted(used)}"
        )
    for entity in entities:
        entity.validate(rendered_text)
    if template.hard_negative and entities:
        raise SyntheticGenerationError("hard-negative templates cannot emit PII entities")
    return RenderedTemplate(
        text=rendered_text,
        entities=tuple(entities),
        values=dict(values),
        validators_passed=validators_passed,
        validator_results=validator_results,
    )


class SyntheticGenerator:
    """Generate deterministic documents without any network or external PII source."""

    def __init__(self, seed: int, *, license: str = DEFAULT_LICENSE) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.seed = seed
        self.license = license
        self._rng = random.Random(seed)
        self._factory = SyntheticValueFactory(self._rng)
        self._counter = 0

    def _values_for(self, template: SyntheticTemplate) -> dict[str, str]:
        return {
            spec.key: generate_value(self._factory, spec.generator_key)
            for spec in template.placeholders
        }

    def _select_templates(
        self,
        templates: tuple[SyntheticTemplate, ...],
        count: int,
    ) -> list[SyntheticTemplate]:
        """Select complete shuffled cycles so every family is used before reuse."""

        selected: list[SyntheticTemplate] = []
        while len(selected) < count:
            cycle = list(templates)
            self._rng.shuffle(cycle)
            selected.extend(cycle[: count - len(selected)])
        return selected

    def generate_document(self, template: SyntheticTemplate) -> DocumentRecord:
        index = self._counter
        self._counter += 1
        rendered = render_template(template, self._values_for(template))
        if not rendered.validators_passed:
            raise SyntheticGenerationError(
                f"generated structured value failed validation in {template.template_id}"
            )
        digest_input = f"{self.seed}:{index}:{template.template_id}:{rendered.text}"
        doc_id = "sha256:" + hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        provenance = Provenance(
            source_id="repo_curated_synthetic_templates",
            source_kind="curated_deterministic_synthetic",
            source_revision=f"sha256:{CURATED_ASSET_SHA256}",
            license=self.license,
            synthetic=True,
            generator_name=GENERATOR_NAME,
            generator_version=GENERATOR_VERSION,
            generator_seed=self.seed,
            template_id=template.template_id,
            metadata={
                "contains_real_personal_data": False,
                "reserved_test_identifiers": True,
                "template_asset_id": CURATED_ASSET_ID,
                "template_asset_version": CURATED_ASSET_VERSION,
                "template_asset_sha256": CURATED_ASSET_SHA256,
                "curation_audit_sha256": CURATION_AUDIT_SHA256,
                "template_origin_kind": template.origin_kind,
                "source_candidate_id": template.source_candidate_id,
                "candidate_model_id": CURATED_ASSET_METADATA["candidate_source"]["model_id"],
            },
        )
        quality = QualityMetadata(
            tier=QualityTier.HARD_NEGATIVE
            if template.hard_negative
            else QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="human_curated_and_deterministic_validator_passed",
            confidence=1.0,
            metadata={"validator_results": dict(rendered.validator_results)},
        )
        value_groups = tuple(
            sorted({stable_text_hash(entity.text) for entity in rendered.entities})
        )
        return DocumentRecord.create(
            doc_id=doc_id,
            text=rendered.text,
            entities=rendered.entities,
            language="zh-Hans-CN",
            domain=template.domain,
            scene=template.scene,
            provenance=provenance,
            quality=quality,
            public_weight_training_allowed=True,
            template_group=f"synthetic-family:{template.family}",
            entity_value_groups=value_groups,
            generator_version=GENERATOR_VERSION,
            metadata={
                "hard_negative": template.hard_negative,
                "contains_real_personal_data": False,
                "reserved_test_identifiers": True,
                "template_family": template.family,
                "template_asset_id": CURATED_ASSET_ID,
                "template_asset_version": CURATED_ASSET_VERSION,
            },
        )

    def generate_dataset(
        self,
        count: int,
        *,
        hard_negative_ratio: float = 0.30,
    ) -> list[DocumentRecord]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if not 0.30 <= hard_negative_ratio < 1.0:
            raise ValueError("hard_negative_ratio must be at least 0.30 and below 1.0")
        negative_count = max(1, math.ceil(count * hard_negative_ratio))
        positive_count = count - negative_count
        selected = self._select_templates(POSITIVE_TEMPLATES, positive_count)
        selected.extend(self._select_templates(HARD_NEGATIVE_TEMPLATES, negative_count))
        self._rng.shuffle(selected)
        return [self.generate_document(template) for template in selected]

    def generate_partition(
        self,
        *,
        positive_count: int,
        hard_negative_count: int,
        positive_templates: tuple[SyntheticTemplate, ...],
        hard_negative_templates: tuple[SyntheticTemplate, ...],
    ) -> list[DocumentRecord]:
        """Generate one deterministic split from disjoint template-family pools.

        A materializer first assigns every template family to exactly one split,
        then calls this method with that split's pools.  Keeping a single
        ``SyntheticGenerator`` instance across calls also guarantees that core
        entity generators do not restart their uniqueness sequences at split
        boundaries.
        """

        counts = (positive_count, hard_negative_count)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts
        ):
            raise ValueError("partition counts must be non-negative integers")
        if positive_count and not positive_templates:
            raise ValueError("positive_templates cannot be empty when positive_count is non-zero")
        if hard_negative_count and not hard_negative_templates:
            raise ValueError(
                "hard_negative_templates cannot be empty when hard_negative_count is non-zero"
            )
        if any(template.hard_negative for template in positive_templates):
            raise ValueError("positive_templates must not contain hard-negative templates")
        if any(not template.hard_negative for template in hard_negative_templates):
            raise ValueError("hard_negative_templates must contain only hard-negative templates")

        selected = self._select_templates(positive_templates, positive_count)
        selected.extend(self._select_templates(hard_negative_templates, hard_negative_count))
        self._rng.shuffle(selected)
        return [self.generate_document(template) for template in selected]


def generate_synthetic_dataset(
    count: int,
    *,
    seed: int,
    hard_negative_ratio: float = 0.30,
) -> list[DocumentRecord]:
    return SyntheticGenerator(seed).generate_dataset(
        count,
        hard_negative_ratio=hard_negative_ratio,
    )
