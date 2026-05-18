"""Position strategies for content placement within layers.

FR15: PositionStrategy protocol with PrefixStrategy and AppendStrategy built-ins.
FR20: Protocol-based — new position strategies addable without modifying engine code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PositionStrategy(Protocol):
    """Determines how new content is placed relative to existing content."""

    def place(self, existing: list, new: list) -> list: ...


class AppendStrategy:
    """Appends new items after existing items."""

    def place(self, existing: list, new: list) -> list:
        return existing + new


class PrefixStrategy:
    """Prepends new items before existing items."""

    def place(self, existing: list, new: list) -> list:
        return new + existing
