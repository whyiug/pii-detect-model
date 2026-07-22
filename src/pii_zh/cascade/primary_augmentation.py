"""Compose one authoritative detector with a non-destructive augmentation detector."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Protocol, cast

from pii_zh.fusion.replacement import (
    ReplacementSpec,
    apply_replacements,
    merge_replacement_coverage,
)

from .config import CascadeConfig
from .result import CascadeDetection
from .routing import canonicalize_entity_type


class BatchDetectionPipeline(Protocol):
    """Minimum child-pipeline contract used by the composition layer."""

    @property
    def config(self) -> CascadeConfig: ...

    @property
    def model_identity(self) -> Mapping[str, str | int | bool | None] | None: ...

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> Iterable[Iterable[CascadeDetection]]: ...


def _as_sequence(value: object, *, location: str) -> list[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{location} must be a sequence")
    try:
        return list(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{location} must be a sequence") from exc


def _validated_batches(
    value: object,
    *,
    expected: int,
    branch: str,
) -> list[list[CascadeDetection]]:
    batches = _as_sequence(value, location=f"{branch} batch output")
    if len(batches) != expected:
        raise ValueError(f"{branch} pipeline must return one batch per input text")
    output: list[list[CascadeDetection]] = []
    for raw_batch in batches:
        batch = _as_sequence(raw_batch, location=f"{branch} batch item")
        if any(not isinstance(item, CascadeDetection) for item in batch):
            raise TypeError(f"{branch} pipeline must return CascadeDetection values")
        output.append(cast(list[CascadeDetection], batch))
    return output


def _overlaps(left: CascadeDetection, right: CascadeDetection) -> bool:
    return left.start < right.end and right.start < left.end


def _requested_entities(entities: Sequence[str] | None) -> frozenset[str] | None:
    if entities is None:
        return None
    if isinstance(entities, (str, bytes)):
        raise TypeError("entities must be a sequence of labels, not one string")
    return frozenset(canonicalize_entity_type(entity) for entity in entities)


class PrimaryPreservingAugmentationPipeline:
    """Add only augmentation spans that do not overlap authoritative output.

    Both child pipelines run without an entity filter.  Applying the caller's
    filter only after composition makes filtered detection exactly equivalent
    to filtering the complete result and avoids branch-specific filter
    semantics.  Child failures and malformed outputs are deliberately allowed
    to propagate rather than turning a two-stage service into a silent
    primary-only fallback.
    """

    def __init__(
        self,
        *,
        primary: BatchDetectionPipeline,
        augmentation: BatchDetectionPipeline,
        config: CascadeConfig | None = None,
        model_identity: Mapping[str, str | int | bool | None] | None = None,
    ) -> None:
        if not callable(getattr(primary, "detect_batch", None)):
            raise TypeError("primary must expose detect_batch")
        if not callable(getattr(augmentation, "detect_batch", None)):
            raise TypeError("augmentation must expose detect_batch")
        resolved_config = config if config is not None else getattr(augmentation, "config", None)
        if not isinstance(resolved_config, CascadeConfig):
            raise TypeError("config must be a CascadeConfig or be exposed by augmentation")
        inherited_identity = getattr(augmentation, "model_identity", None)
        resolved_identity = model_identity if model_identity is not None else inherited_identity
        if resolved_identity is not None:
            if (
                not isinstance(resolved_identity, Mapping)
                or not resolved_identity
                or any(
                    not isinstance(key, str)
                    or not key
                    or not isinstance(value, (str, int, bool, type(None)))
                    for key, value in resolved_identity.items()
                )
            ):
                raise TypeError("model_identity must be a non-empty flat JSON-safe mapping")
            self.model_identity: Mapping[str, str | int | bool | None] | None = (
                MappingProxyType(dict(resolved_identity))
            )
        else:
            self.model_identity = None
        self.config = resolved_config
        self.primary = primary
        self.augmentation = augmentation

    @staticmethod
    def _compose(
        primary: Sequence[CascadeDetection],
        augmentation: Sequence[CascadeDetection],
        *,
        requested: frozenset[str] | None,
    ) -> list[CascadeDetection]:
        # Preserve every primary object and every augmentation object whose
        # half-open span is disjoint from all primary spans.  Augmentation is
        # assumed to have completed its own thresholding and conflict fusion.
        accepted_augmentation = [
            candidate
            for candidate in augmentation
            if not any(_overlaps(candidate, authoritative) for authoritative in primary)
        ]
        combined = sorted(
            [*primary, *accepted_augmentation],
            key=lambda item: (item.start, item.end),
        )
        if requested is None:
            return combined
        return [
            item
            for item in combined
            if canonicalize_entity_type(item.entity_type) in requested
        ]

    def _run_many(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None,
    ) -> list[list[CascadeDetection]]:
        requested = _requested_entities(entities)
        # No try/except or fallback is intentional: either branch failing
        # means the composed service failed to produce its declared result.
        primary = _validated_batches(
            self.primary.detect_batch(texts, entities=None),
            expected=len(texts),
            branch="primary",
        )
        augmentation = _validated_batches(
            self.augmentation.detect_batch(texts, entities=None),
            expected=len(texts),
            branch="augmentation",
        )
        return [
            self._compose(primary_batch, augmentation_batch, requested=requested)
            for primary_batch, augmentation_batch in zip(primary, augmentation, strict=True)
        ]

    def detect(
        self,
        text: str,
        *,
        entities: Sequence[str] | None = None,
    ) -> list[CascadeDetection]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._run_many([text], entities=entities)[0]

    def detect_batch(
        self,
        texts: Sequence[str],
        *,
        entities: Sequence[str] | None = None,
    ) -> list[list[CascadeDetection]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return self._run_many(values, entities=entities)

    def redact(
        self,
        text: str,
        replacement: ReplacementSpec = "<PII>",
        *,
        default_replacement: str | None = None,
        entities: Sequence[str] | None = None,
    ) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        detections = self._run_many([text], entities=entities)[0]
        coverage = merge_replacement_coverage(detections)
        return apply_replacements(
            text,
            coverage,
            replacement,
            default_replacement=default_replacement,
        )


__all__ = ["BatchDetectionPipeline", "PrimaryPreservingAugmentationPipeline"]
