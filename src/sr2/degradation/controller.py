"""DegradationController — trigger evaluation (FR4).

The controller evaluates configured triggers against runtime signals and
drives ``DegradationLadder.step_down()`` when a condition is met.

v1 trigger set (D1):
  - ``overflow``: budget pressure detected at compile time (pre-LLM).
  - ``context_limit``: LLM provider rejected the request as too long.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2.config.models import DegradationTriggerConfig


# Patterns that indicate the LLM provider rejected the request due to
# context-length limits.  Matched case-insensitively against the
# exception message.
_CONTEXT_LIMIT_PATTERNS: list[str] = [
    r"context\s*(window|length|size)",
    r"exceeds?\s*(context|limit|budget)",
    r"too\s*(long|large|big)",
    r"max(?:imum)?\s*(token|context|length)",
    r"prompt\s*exceeds",
    r"input\s*is\s*too\s*long",
]

_CONTEXT_LIMIT_RE: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _CONTEXT_LIMIT_PATTERNS
]


class DegradationController:
    """Evaluates triggers and drives ladder step-downs.

    Parameters
    ----------
    ladder : DegradationLadder
        The ladder whose level this controller may step down.
    triggers : list[DegradationTriggerConfig]
        The configured trigger definitions.
    """

    def __init__(
        self,
        ladder: "DegradationLadder",
        triggers: list["DegradationTriggerConfig"],
    ) -> None:
        self._ladder = ladder
        self._triggers = triggers

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def ladder(self) -> "DegradationLadder":
        """The degradation ladder this controller drives."""
        return self._ladder

    def has_trigger(self, trigger_type: str) -> bool:
        """Return True if a trigger of *trigger_type* is configured."""
        return any(t.type == trigger_type for t in self._triggers)

    # ------------------------------------------------------------------
    # Trigger predicates
    # ------------------------------------------------------------------

    def over_budget(self, total_tokens: int, budget: int) -> bool:
        """Evaluate the *overflow* trigger predicate.

        Returns True when ``total_tokens > budget`` **and** an ``overflow``
        trigger is configured.  Returns False when no overflow trigger is
        active (the controller should not fire on behalf of a trigger the
        user never configured).

        Parameters
        ----------
        total_tokens : int
            Total tokens in the compiled CompletionRequest.
        budget : int
            The pipeline's configured token budget.
        """
        if not self.has_trigger("overflow"):
            return False
        return total_tokens > budget

    def is_context_limit_error(self, exc: Exception | str) -> bool:
        """Evaluate the *context_limit* trigger recogniser.

        Returns True when the exception message matches known context-length
        error patterns **and** a ``context_limit`` trigger is configured.

        Parameters
        ----------
        exc : Exception | str
            The exception raised by the LLM provider, or a raw error string.
        """
        if not self.has_trigger("context_limit"):
            return False

        message: str = exc if isinstance(exc, str) else str(exc)
        return any(pattern.search(message) for pattern in _CONTEXT_LIMIT_RE)

    # ------------------------------------------------------------------
    # Ladder control helpers
    # ------------------------------------------------------------------

    def step_down_if_needed(self) -> None:
        """Step the ladder down one level.

        This is a thin wrapper around ``ladder.step_down()`` that acts as a
        single call-site for the orchestrator/engine.  The actual decision to
        step down is made by the trigger predicates (``over_budget`` /
        ``is_context_limit_error``); this method is the effect side.

        No-op when already at the most-degraded level.
        """
        self._ladder.step_down()

    def reset(self) -> None:
        """Reset the ladder to FULL.

        Called at the start of every turn (FR7).  This delegates to the
        ladder but provides a single entry point in the controller.
        """
        self._ladder.reset()
