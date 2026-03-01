import time


class CircuitBreaker:
    """Per-stage circuit breaker. Opens after N consecutive failures,
    closes after cooldown period."""

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 300.0):
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._failure_counts: dict[str, int] = {}
        self._open_since: dict[str, float] = {}

    def record_success(self, stage: str) -> None:
        """Reset failure count for a stage."""
        self._failure_counts.pop(stage, None)
        self._open_since.pop(stage, None)

    def record_failure(self, stage: str) -> None:
        """Record a failure. Opens breaker if threshold reached."""
        self._failure_counts[stage] = self._failure_counts.get(stage, 0) + 1
        if self._failure_counts[stage] >= self._threshold:
            self._open_since[stage] = time.monotonic()

    def is_open(self, stage: str) -> bool:
        """Return True if the breaker is open (stage should be skipped)."""
        if stage not in self._open_since:
            return False
        elapsed = time.monotonic() - self._open_since[stage]
        if elapsed >= self._cooldown:
            self._open_since.pop(stage)
            self._failure_counts.pop(stage, None)
            return False
        return True

    def status(self) -> dict[str, dict]:
        """Return status of all tracked stages."""
        result = {}
        for stage in set(list(self._failure_counts) + list(self._open_since)):
            result[stage] = {
                "failures": self._failure_counts.get(stage, 0),
                "is_open": self.is_open(stage),
            }
        return result
