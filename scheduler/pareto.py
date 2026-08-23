"""Deterministic Pareto-front assignment for named risk vectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import TypeVar

T = TypeVar("T")


def dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = sorted(set(left) | set(right))
    if not keys:
        return False

    def value(vector: Mapping[str, float], key: str) -> float:
        raw = float(vector.get(key, float("inf")))
        return raw if isfinite(raw) else float("inf")

    return all(value(left, key) <= value(right, key) for key in keys) and any(
        value(left, key) < value(right, key) for key in keys
    )


def pareto_fronts(
    items: Sequence[T],
    risk_vector: Callable[[T], Mapping[str, float]],
    *,
    stable_key: Callable[[T], str],
) -> dict[str, int]:
    remaining = sorted(items, key=stable_key)
    assigned: dict[str, int] = {}
    front_index = 0
    while remaining:
        front = [
            item
            for item in remaining
            if not any(
                other is not item and dominates(risk_vector(other), risk_vector(item))
                for other in remaining
            )
        ]
        # Stable keys are required to make identical risk vectors deterministic.
        front = sorted(front, key=stable_key)
        for item in front:
            assigned[stable_key(item)] = front_index
        front_ids = {id(item) for item in front}
        remaining = [item for item in remaining if id(item) not in front_ids]
        front_index += 1
    return assigned
