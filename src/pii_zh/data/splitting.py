"""Leakage-resistant group splitting for document/template/entity-value groups."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .schema import DocumentRecord


class SplitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LeakageCollision:
    group_type: str
    group_value: str
    splits: tuple[str, ...]
    doc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeakageReport:
    collisions: tuple[LeakageCollision, ...]
    records_checked: int

    @property
    def has_leakage(self) -> bool:
        return bool(self.collisions)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def record_group_keys(record: DocumentRecord) -> tuple[tuple[str, str], ...]:
    """Return all group identities that must remain in one split."""

    keys: list[tuple[str, str]] = [("doc", record.doc_id)]
    if record.template_group:
        keys.append(("template", record.template_group))
    keys.extend(("value", group) for group in record.entity_value_groups)
    return tuple(keys)


def _connected_components(records: Sequence[DocumentRecord]) -> list[list[int]]:
    union_find = _UnionFind(len(records))
    first_by_key: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        record.validate()
        for key in record_group_keys(record):
            previous = first_by_key.setdefault(key, index)
            union_find.union(index, previous)
    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(union_find.find(index), []).append(index)
    return list(components.values())


def group_split(
    records: Sequence[DocumentRecord],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
    required_label_splits: Sequence[str] = (),
    required_labels: Sequence[str] | None = None,
) -> list[DocumentRecord]:
    """Assign connected group components to deterministic approximate splits.

    ``required_label_splits`` optionally seeds a group-safe, multi-label
    stratification pass.  ``required_labels`` can pin the exact contract that
    must appear in each requested split; otherwise all labels observed in the
    input are covered.  This is intended for curated synthetic corpora where
    every core label has several independent template families.  The default
    remains the original capacity-only behavior for arbitrary source data.
    """

    if not records:
        return []
    ratios = dict(ratios or {"train": 0.8, "validation": 0.1, "test": 0.1})
    if not ratios or any(
        not isinstance(name, str)
        or not name
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or weight <= 0
        for name, weight in ratios.items()
    ):
        raise SplitError("ratios must map non-empty split names to positive numbers")
    total_weight = float(sum(ratios.values()))
    if abs(total_weight - 1.0) > 1e-9:
        raise SplitError(f"split ratios must sum to 1.0, got {total_weight}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    if isinstance(required_label_splits, (str, bytes)) or any(
        not isinstance(name, str) or name not in ratios for name in required_label_splits
    ):
        raise SplitError("required_label_splits must contain configured split names")
    if len(required_label_splits) != len(set(required_label_splits)):
        raise SplitError("required_label_splits cannot contain duplicates")
    if required_labels is not None and (
        isinstance(required_labels, (str, bytes))
        or any(not isinstance(label, str) or not label for label in required_labels)
        or len(required_labels) != len(set(required_labels))
    ):
        raise SplitError("required_labels must contain unique non-empty label names")
    if required_labels is not None and not required_label_splits:
        raise SplitError("required_labels requires at least one required_label_split")

    components = _connected_components(records)

    def component_order(component: list[int]) -> tuple[int, str]:
        identity = "|".join(sorted(records[index].doc_id for index in component))
        digest = hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
        return (-len(component), digest)

    components.sort(key=component_order)
    target = {name: weight * len(records) for name, weight in ratios.items()}
    assigned = {name: 0 for name in ratios}
    assignments: dict[int, str] = {}
    stable_split_order = {
        name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest() for name in ratios
    }
    component_keys = {tuple(component): component_order(component)[1] for component in components}
    component_labels = {
        tuple(component): frozenset(
            entity.label for index in component for entity in records[index].entities
        )
        for component in components
    }
    labels_by_component: dict[str, list[tuple[int, ...]]] = {}
    for component in components:
        component_key = tuple(component)
        for label in component_labels[component_key]:
            labels_by_component.setdefault(label, []).append(component_key)

    labels_to_cover = set(labels_by_component) if required_labels is None else set(required_labels)
    absent_labels = labels_to_cover - set(labels_by_component)
    if absent_labels:
        raise SplitError(f"required labels are absent from the input: {sorted(absent_labels)}")

    coverage: dict[str, set[str]] = {name: set() for name in required_label_splits}
    if required_label_splits:
        insufficient = {
            label: len(labels_by_component[label])
            for label in labels_to_cover
            if len(labels_by_component[label]) < len(required_label_splits)
        }
        if insufficient:
            raise SplitError(
                "not enough independent group components for requested label coverage: "
                f"{insufficient}"
            )

        # Rare labels are assigned first.  Candidate components that cover
        # several labels missing from the same split are preferred, which keeps
        # small validation/test targets close to their requested capacities.
        for label in sorted(
            labels_to_cover,
            key=lambda item: (len(labels_by_component[item]), item),
        ):
            split_order = sorted(
                required_label_splits,
                key=lambda name: (target[name], stable_split_order[name]),
            )
            for split in split_order:
                if label in coverage[split]:
                    continue
                candidates = [
                    component_key
                    for component_key in labels_by_component[label]
                    if not any(index in assignments for index in component_key)
                ]
                if not candidates:
                    raise SplitError(
                        f"cannot satisfy label {label!r} coverage in split {split!r} "
                        "without breaking an existing group assignment"
                    )

                def coverage_score(
                    component_key: tuple[int, ...], target_split: str = split
                ) -> tuple[float, ...]:
                    newly_covered = (component_labels[component_key] & labels_to_cover) - coverage[
                        target_split
                    ]
                    scarcity = sum(1.0 / len(labels_by_component[item]) for item in newly_covered)
                    remaining_after = (
                        target[target_split] - assigned[target_split] - len(component_key)
                    )
                    digest_score = int(component_keys[component_key], 16) / (16**64)
                    return (
                        float(len(newly_covered)),
                        scarcity,
                        min(remaining_after, 0.0),
                        -float(len(component_key)),
                        digest_score,
                    )

                chosen = max(candidates, key=coverage_score)
                for index in chosen:
                    assignments[index] = split
                assigned[split] += len(chosen)
                coverage[split].update(component_labels[chosen] & labels_to_cover)

    for component in components:
        if any(index in assignments for index in component):
            continue
        # Prefer the split with the greatest remaining capacity.  This is stable
        # and keeps an indivisible leakage component intact.
        split = max(
            ratios,
            key=lambda name: (
                target[name] - assigned[name],
                -assigned[name],
                stable_split_order[name],
            ),
        )
        for index in component:
            assignments[index] = split
        assigned[split] += len(component)

    result = [replace(record, split=assignments[index]) for index, record in enumerate(records)]
    assert_no_group_leakage(result)
    for split in required_label_splits:
        observed = {
            entity.label for record in result if record.split == split for entity in record.entities
        }
        missing = labels_to_cover - observed
        if missing:
            raise SplitError(f"split {split!r} is missing required labels: {sorted(missing)}")
    return result


def detect_group_leakage(records: Iterable[DocumentRecord]) -> LeakageReport:
    records_list = list(records)
    by_key: dict[tuple[str, str], dict[str, set[str]]] = {}
    for record in records_list:
        record.validate()
        if not record.split:
            continue
        for key in record_group_keys(record):
            split_map = by_key.setdefault(key, {})
            split_map.setdefault(record.split, set()).add(record.doc_id)
    collisions = []
    for (group_type, group_value), split_map in sorted(by_key.items()):
        if len(split_map) <= 1:
            continue
        collisions.append(
            LeakageCollision(
                group_type=group_type,
                group_value=group_value,
                splits=tuple(sorted(split_map)),
                doc_ids=tuple(sorted({doc for docs in split_map.values() for doc in docs})),
            )
        )
    return LeakageReport(tuple(collisions), len(records_list))


def assert_no_group_leakage(records: Iterable[DocumentRecord]) -> None:
    report = detect_group_leakage(records)
    if report.has_leakage:
        summary = ", ".join(
            f"{item.group_type}:{item.group_value} -> {item.splits}"
            for item in report.collisions[:5]
        )
        raise SplitError(f"group leakage detected ({len(report.collisions)} groups): {summary}")
