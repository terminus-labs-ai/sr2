"""Circuit breaker for content providers.

Per-provider circuit breaker pattern:
closed (normal) -> open (failing, skip provider) -> half-open (try one request) -> closed or back to open.

Design principles:
- SRP: Only tracks failure state — doesn't know about content or providers.
- OCP: New breaker policies added via subclasses, not by modifying state machine.
"""

from __future__ import annotations

import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, skip provider
    HALF_OPEN = "half_open" # Testing if recovery happened


class CircuitBreaker:
    """Per-content-provider circuit breaker.

    Sits on individual providers within a layer, not on the layer itself.
    A memory retrieval failure doesn't kill the whole conversation layer.

    Args:
        threshold: Consecutive failures before opening (default: 3).
        cooldown: Seconds to wait before transitioning to half-open (default: 300).
    """

    def __init__(self, threshold: int = 3, cooldown: int = 300) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0

    @property
    def state(self) -> CircuitState:
        """Current breaker state, with automatic half-open transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.cooldown:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        """Record a successful provider call. Resets failure count."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed provider call. Opens if threshold reached."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.threshold:
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if the provider should be called, False if it should be skipped.
        """
        current_state = self.state  # Triggers half-open check

        if current_state == CircuitState.CLOSED:
            return True
        if current_state == CircuitState.HALF_OPEN:
            return True  # Allow one test request
        # OPEN state
        return False

    def reset(self) -> None:
        """Reset to initial state. Useful for testing."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
