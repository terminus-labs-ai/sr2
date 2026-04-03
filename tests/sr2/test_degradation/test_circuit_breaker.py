import time

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
