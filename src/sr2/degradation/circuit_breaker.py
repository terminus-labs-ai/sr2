"""Circuit breaker for per-provider fault isolation.

States:
  CLOSED    — normal operation; requests allowed.
  OPEN      — tripping threshold exceeded; requests rejected.
  HALF_OPEN — recovery probe; one request allowed to test the provider.

Transitions:
  CLOSED  → OPEN      : consecutive failures reach failure_threshold
  OPEN    → HALF_OPEN : recovery_timeout seconds have elapsed
  HALF_OPEN → CLOSED  : record_success() called
  HALF_OPEN → OPEN    : record_failure() called
"""

from __future__ import annotations

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker with configurable failure threshold and recovery timeout.

    Args:
        failure_threshold: Number of consecutive failures required to open.
        recovery_timeout:  Seconds to wait in OPEN state before probing again.
    """

    def __init__(self, failure_threshold: int, recovery_timeout: float) -> None:
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    def is_open(self) -> bool:
        """Return True iff the breaker is in the OPEN state."""
        return self._state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through.

        Side-effects:
          - OPEN → HALF_OPEN if the recovery timeout has elapsed.
        """
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.HALF_OPEN:
            return True
        # OPEN: check whether the recovery timeout has elapsed
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
        return False

    def record_success(self) -> None:
        """Record a successful call.

        HALF_OPEN → CLOSED (resets failure count).
        CLOSED    → resets consecutive failure count.
        """
        if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call.

        CLOSED    → increments failure count; OPEN if threshold reached.
        HALF_OPEN → OPEN immediately.
        OPEN      → no-op (already open).
        """
        if self._state == CircuitState.OPEN:
            return
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            return
        # CLOSED
        self._failure_count += 1
        if self._failure_count >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
