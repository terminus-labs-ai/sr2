import time

import pytest

from sr2.degradation.circuit_breaker import CircuitBreaker


class TestCircuitBreakerNoFailures:
    """Test 1: No failures -> is_open returns False."""

    def test_is_open_returns_false_with_no_failures(self):
        cb = CircuitBreaker(threshold=3, cooldown_seconds=300.0)
        assert cb.is_open("stage_a") is False


class TestCircuitBreakerBelowThreshold:
    """Test 2: Failures below threshold -> is_open returns False."""

    def test_is_open_returns_false_below_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown_seconds=300.0)
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        assert cb.is_open("stage_a") is False


class TestCircuitBreakerAtThreshold:
    """Test 3: Failures at threshold -> is_open returns True."""

    def test_is_open_returns_true_at_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown_seconds=300.0)
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        assert cb.is_open("stage_a") is True


class TestCircuitBreakerRecordSuccess:
    """Test 4: record_success resets failure count."""

    def test_record_success_resets_failure_count(self):
        cb = CircuitBreaker(threshold=3, cooldown_seconds=300.0)
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        assert cb.is_open("stage_a") is True

        cb.record_success("stage_a")
        assert cb.is_open("stage_a") is False
        # Confirm internal state is cleared
        assert cb._failure_counts.get("stage_a") is None
        assert cb._open_since.get("stage_a") is None


class TestCircuitBreakerCooldownExpiry:
    """Test 5: After cooldown expires -> is_open returns False."""

    def test_is_open_returns_false_after_cooldown(self):
        cb = CircuitBreaker(threshold=3, cooldown_seconds=0.1)
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        assert cb.is_open("stage_a") is True

        time.sleep(0.15)
        assert cb.is_open("stage_a") is False


class TestCircuitBreakerStatus:
    """Test 6: status() returns correct info for multiple stages."""

    def test_status_returns_correct_info(self):
        cb = CircuitBreaker(threshold=2, cooldown_seconds=300.0)
        # stage_a: 1 failure, not open
        cb.record_failure("stage_a")
        # stage_b: 2 failures, open
        cb.record_failure("stage_b")
        cb.record_failure("stage_b")

        result = cb.status()

        assert "stage_a" in result
        assert result["stage_a"]["failures"] == 1
        assert result["stage_a"]["is_open"] is False

        assert "stage_b" in result
        assert result["stage_b"]["failures"] == 2
        assert result["stage_b"]["is_open"] is True


class TestCircuitBreakerIndependentStages:
    """Test 7: Independent stages don't affect each other."""

    def test_independent_stages(self):
        cb = CircuitBreaker(threshold=3, cooldown_seconds=300.0)
        # Open circuit for stage_a
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        assert cb.is_open("stage_a") is True

        # stage_b should remain unaffected
        assert cb.is_open("stage_b") is False

        # Adding failures to stage_b doesn't change stage_a
        cb.record_failure("stage_b")
        assert cb.is_open("stage_a") is True
        assert cb.is_open("stage_b") is False

        # Resetting stage_a doesn't affect stage_b
        cb.record_success("stage_a")
        assert cb.is_open("stage_a") is False
        assert cb._failure_counts.get("stage_b") == 1


class TestCircuitBreakerHalfOpen:
    """Half-open state machine: after cooldown, one trial is allowed."""

    def _open_breaker(self, cb: CircuitBreaker, stage: str = "stage") -> None:
        """Record enough failures to open the breaker."""
        for _ in range(cb._threshold):
            cb.record_failure(stage)
        assert cb.is_open(stage) is True

    def test_half_open_single_failure_reopens(self):
        """After cooldown expires, a single failure re-opens the breaker."""
        cb = CircuitBreaker(threshold=2, cooldown_seconds=0.05)
        self._open_breaker(cb)

        time.sleep(0.1)
        # Cooldown expired — breaker should now be closed (half-open trial)
        assert cb.is_open("stage") is False

        # Trial fails — one failure shouldn't open (threshold=2),
        # but record enough to re-open
        cb.record_failure("stage")
        cb.record_failure("stage")
        assert cb.is_open("stage") is True

    def test_half_open_success_closes(self):
        """After cooldown expires, a success closes the breaker and resets counters."""
        cb = CircuitBreaker(threshold=2, cooldown_seconds=0.05)
        self._open_breaker(cb)

        time.sleep(0.1)
        # Cooldown expired — is_open returns False (clears state internally)
        assert cb.is_open("stage") is False

        # Record success — breaker stays closed, counters cleared
        cb.record_success("stage")
        assert cb.is_open("stage") is False
        assert cb._failure_counts.get("stage") is None
        assert cb._open_since.get("stage") is None

    def test_half_open_failure_resets_cooldown(self):
        """After half-open failure re-opens, cooldown restarts from scratch."""
        cb = CircuitBreaker(threshold=1, cooldown_seconds=0.05)
        self._open_breaker(cb)

        time.sleep(0.1)
        # Cooldown expired — half-open
        assert cb.is_open("stage") is False

        # Fail again — re-opens immediately (threshold=1)
        cb.record_failure("stage")
        assert cb.is_open("stage") is True

        # Should still be open immediately (cooldown just restarted)
        assert cb.is_open("stage") is True

        # Wait for second cooldown
        time.sleep(0.1)
        assert cb.is_open("stage") is False


class TestCircuitBreakerThresholdParametrized:
    """Verify breaker opens at exactly the configured threshold."""

    @pytest.mark.parametrize("threshold", [1, 2, 3, 5])
    def test_threshold_opens_at_exact_count(self, threshold):
        cb = CircuitBreaker(threshold=threshold, cooldown_seconds=300.0)
        for i in range(threshold - 1):
            cb.record_failure("stage")
            assert cb.is_open("stage") is False, f"Opened early at failure {i+1}/{threshold}"
        cb.record_failure("stage")
        assert cb.is_open("stage") is True, f"Should open at failure {threshold}"
