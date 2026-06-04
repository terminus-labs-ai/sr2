"""Degradation ladder — tracks which degradation level is active.

The ladder can be constructed in two ways:

1. **Config-built** (preferred, FR3): ``DegradationLadder.from_config(cfg)``
   builds the level table from ``DegradationConfig``.  The hardcoded enum and
   category map are replaced by config data.

2. **Legacy constructor**: ``DegradationLadder()`` or
   ``DegradationLadder(initial_level=DegradationLevel.FULL)`` still works
   with the built-in 5-level ladder for backwards compatibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# Legacy constants — kept for backwards compatibility
# ---------------------------------------------------------------------------


class DegradationLevel(Enum):
    """Legacy degradation levels.

    .. deprecated:: Use ``DegradationLadder.from_config()`` with a
       ``DegradationConfig`` instead.  The enum is retained so that old
       code (``DegradationLadder(initial_level=DegradationLevel.FULL)``)
       continues to work.
    """

    FULL = 0
    REDUCED_MEMORY = 1
    TOOLS_DISABLED = 2
    MEMORY_DISABLED = 3
    SYSTEM_PROMPT_ONLY = 4


# Which provider categories are active at each legacy level.
_ACTIVE_PROVIDERS: dict[DegradationLevel, list[str]] = {
    DegradationLevel.FULL: ["system", "memory", "tools", "context", "history"],
    DegradationLevel.REDUCED_MEMORY: ["system", "memory", "tools", "context"],
    DegradationLevel.TOOLS_DISABLED: ["system", "memory", "context"],
    DegradationLevel.MEMORY_DISABLED: ["system", "context"],
    DegradationLevel.SYSTEM_PROMPT_ONLY: ["system"],
}

_ORDERED_LEVELS = list(DegradationLevel)


# ---------------------------------------------------------------------------
# Config types (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------


def _import_degradation_config():
    """Lazily import DegradationConfig to avoid import-time circular deps."""
    from sr2.config.models import DegradationConfig

    return DegradationConfig


# ===========================================================================
# DegradationLadder
# ===========================================================================


class DegradationLadder:
    """Tracks the active degradation level and enforces monotonic step-down.

    Can be constructed via ``from_config()`` (config-driven) or via the
    legacy ``__init__(initial_level=DegradationLevel.FULL)``.

    Attributes
    ----------
    current_level : int
        Index into the config-defined level table (0 = FULL).
        For legacy construction, returns the ``DegradationLevel`` enum value
        for backwards compatibility.
    """

    def __init__(
        self,
        initial_level: DegradationLevel | int | None = None,
        *,
        _levels: list[list[str]] | None = None,
    ) -> None:
        """Construct a DegradationLadder.

        Parameters
        ----------
        initial_level : DegradationLevel | int | None
            For legacy construction: a ``DegradationLevel`` enum value or an
            integer index.  Defaults to ``DegradationLevel.FULL`` (index 0).
            For config-built ladders: passed as ``0`` internally.
        _levels : list[list[str]] | None
            Internal use only.  A list of category-lists, one per level,
            from most to least capable.  Set by ``from_config()``.
        """
        if _levels is not None:
            # Config-built ladder
            self._levels: list[list[str]] = _levels
            self._level_index: int = (
                0 if initial_level is None or initial_level == 0 else initial_level
            )
            self._legacy = False
        else:
            # Legacy construction using hardcoded enum
            self._levels = [
                _ACTIVE_PROVIDERS[level] for level in _ORDERED_LEVELS
            ]
            if isinstance(initial_level, DegradationLevel):
                self._level_index = initial_level.value
            elif isinstance(initial_level, int):
                self._level_index = initial_level
            else:
                self._level_index = 0  # FULL
            self._legacy = True

    @classmethod
    def from_config(cls, cfg: "DegradationConfig") -> "DegradationLadder":
        """Build a ladder from a ``DegradationConfig``.

        The ordered ``levels`` list (FULL → most-degraded) becomes the
        level table.  Each level's ``keep_categories`` is stored as-is.

        Parameters
        ----------
        cfg : DegradationConfig
            Configuration with at least one level defined.

        Returns
        -------
        DegradationLadder
            A ladder starting at index 0 (FULL) with the configured levels.
        """
        # Import type lazily to avoid circular deps
        DegradationConfig = _import_degradation_config()

        levels = [[cat for cat in level_cfg.keep_categories] for level_cfg in cfg.levels]
        return cls(_levels=levels)

    @property
    def current_level(self) -> int | DegradationLevel:
        """Current level.

        For config-built ladders: returns the integer index (0 = FULL).

        For legacy construction: returns the ``DegradationLevel`` enum
        value for backwards compatibility with existing code that
        compares against ``DegradationLevel.FULL``, etc.
        """
        if self._legacy:
            return _ORDERED_LEVELS[self._level_index]
        return self._level_index

    def step_down(self) -> None:
        """Move one step toward greater degradation. No-op at the lowest level."""
        if self._level_index < len(self._levels) - 1:
            self._level_index += 1

    def reset(self) -> None:
        """Return to FULL capability (level 0)."""
        self._level_index = 0

    def is_at_full(self) -> bool:
        """Return True iff the current level is FULL (index 0)."""
        return self._level_index == 0

    def active_categories(self) -> set[str]:
        """Return the set of category strings active at the current level.

        Categories not in this set should be excluded when compiling the
        request at the current degradation level.
        """
        return set(self._levels[self._level_index])

    def active_providers(self) -> list[str]:
        """Return the list of provider categories active at the current level.

        .. deprecated:: Use ``active_categories()`` instead.
           This method is retained for backwards compatibility.
        """
        return list(self._levels[self._level_index])
