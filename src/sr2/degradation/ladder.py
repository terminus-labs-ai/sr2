"""Degradation ladder — 5 levels from full capability to system-prompt-only.

Levels (in order of increasing severity):
  FULL              — all providers active
  REDUCED_MEMORY    — memory-intensive features scaled back
  TOOLS_DISABLED    — tool providers removed
  MEMORY_DISABLED   — memory providers removed
  SYSTEM_PROMPT_ONLY — only the system prompt layer remains
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class DegradationLevel(Enum):
    FULL = 0
    REDUCED_MEMORY = 1
    TOOLS_DISABLED = 2
    MEMORY_DISABLED = 3
    SYSTEM_PROMPT_ONLY = 4


# Which provider categories are active at each level.
# Listed from most-capable (FULL) to least-capable (SYSTEM_PROMPT_ONLY).
_ACTIVE_PROVIDERS: dict[DegradationLevel, list[str]] = {
    DegradationLevel.FULL: ["system", "memory", "tools", "context", "history"],
    DegradationLevel.REDUCED_MEMORY: ["system", "memory", "tools", "context"],
    DegradationLevel.TOOLS_DISABLED: ["system", "memory", "context"],
    DegradationLevel.MEMORY_DISABLED: ["system", "context"],
    DegradationLevel.SYSTEM_PROMPT_ONLY: ["system"],
}

_ORDERED_LEVELS = list(DegradationLevel)


class DegradationLadder:
    """Tracks the active degradation level and enforces monotonic step-down."""

    def __init__(self, initial_level: DegradationLevel = DegradationLevel.FULL) -> None:
        self._level = initial_level

    @property
    def current_level(self) -> DegradationLevel:
        return self._level

    def step_down(self) -> None:
        """Move one step toward greater degradation. No-op at the lowest level."""
        idx = _ORDERED_LEVELS.index(self._level)
        if idx < len(_ORDERED_LEVELS) - 1:
            self._level = _ORDERED_LEVELS[idx + 1]

    def reset(self) -> None:
        """Return to FULL capability."""
        self._level = DegradationLevel.FULL

    def is_at_full(self) -> bool:
        """Return True iff the current level is FULL."""
        return self._level == DegradationLevel.FULL

    def active_providers(self) -> list[str]:
        """Return the list of provider categories active at the current level."""
        return list(_ACTIVE_PROVIDERS[self._level])
