"""Priority-based content layer shedding.

shed(layers, budget) removes lowest-priority layers until the total
token count of the survivors is within the given budget.

Priority: lower number = lower priority (shed first).
Original declaration order is preserved among survivors.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HasPriorityAndTokens(Protocol):
    priority: int
    token_count: int


def shed(layers: list, budget: int) -> list:
    """Return a subset of *layers* whose total token_count <= *budget*.

    Layers with the lowest priority are shed first. When priorities tie,
    all tied layers at that priority may be shed if necessary to fit the
    budget.  The original order of the surviving layers is preserved.

    Args:
        layers: Objects with `.priority` (int) and `.token_count` (int).
        budget:  Maximum total token count allowed across survivors.

    Returns:
        A new list containing the surviving layers in original order.
    """
    if not layers:
        return []

    total = sum(layer.token_count for layer in layers)
    if total <= budget:
        return list(layers)

    # Sort by priority ascending (shed lowest first), preserving original
    # index as tiebreaker so we have a stable ordering.
    indexed = sorted(enumerate(layers), key=lambda pair: (pair[1].priority, pair[0]))

    shed_indices: set[int] = set()
    for original_idx, layer in indexed:
        if total <= budget:
            break
        shed_indices.add(original_idx)
        total -= layer.token_count

    return [layer for idx, layer in enumerate(layers) if idx not in shed_indices]
