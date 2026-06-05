"""Run-context dependencies for resolvers and transformers.

This module defines the typed seam through which a harness (e.g. spectre)
supplies execution metadata — mode (interactive vs headless) and source —
down to pipeline components at resolution time.

Core MUST NOT import spectre types.  These are minimal primitives that
spectre's RunContext can populate without a reverse dependency.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias


class RunMode(enum.StrEnum):
    """Execution mode: interactive (can ask user) or headless (must self-resolve).

    Defined in core so resolvers can match on it without importing the harness.
    Spectre's RunMode is a separate enum that maps 1:1.
    """

    INTERACTIVE = "interactive"
    HEADLESS = "headless"


@dataclass(frozen=True)
class RunContext:
    """Run metadata injected by the harness into ResolverContext.

    When the harness does not supply a provider this field remains None and
    resolver behavior is unchanged (regression-safe).

    Attributes:
        mode: Interactive or headless — controls agent proactivity.
        source: Where the run originated (working directory, channel name, etc.).
    """

    mode: RunMode
    source: str | None = None


#: Callable that returns the current run's context, or None if unavailable.
#: Mirrors the active_frame_provider pattern: generic, no harness import.
RunContextProvider: TypeAlias = Callable[[], RunContext | None]


__all__ = [
    "RunContext",
    "RunContextProvider",
    "RunMode",
]
