"""Tests for sr2-56: Circuit breaker + timeout wiring around LLM calls in SR2.turn().

Requirements:
  CB1 — LLM exceptions are recorded on the circuit breaker as failures.
  CB2 — After `failure_threshold` consecutive failures the circuit opens.
  CB3 — When the circuit is open, turn() raises CircuitBreakerOpenError without
         calling the LLM.
  CB4 — After `recovery_timeout` seconds the circuit transitions to HALF_OPEN
         and allows one probe call through.
  CB5 — A successful LLM call resets the failure count (circuit stays / returns
         to CLOSED).

  TO1 — PipelineConfig gains `llm_timeout_seconds: float | None = None`.
  TO2 — When set, each LLM stream call is bounded by that timeout.
  TO3 — When the LLM stream exceeds the timeout, turn() raises
         asyncio.TimeoutError (or a subclass / wrapper thereof).
  TO4 — When `llm_timeout_seconds=None` (default), no timeout is applied
         and slow LLMs complete normally.

All tests are expected to FAIL until the feature is implemented.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, StreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_user_input(text: str = "Hello") -> list:
    return [TextBlock(text=text)]


def make_minimal_config(**overrides: Any) -> PipelineConfig:
    """Minimal two-layer config, accepting keyword overrides for PipelineConfig fields."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ],
        **overrides,
    )


class SimpleMockLLM:
    """LLM that returns a fixed sequence of events on every stream() call."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events
        self.stream_calls: int = 0

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        for event in self._events:
            yield event


class FailingLLM:
    """LLM that always raises an exception from stream()."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("LLM backend error")
        self.stream_calls: int = 0

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        raise self._exc
        # unreachable, but satisfies the async generator protocol
        yield  # type: ignore[misc]


class SlowLLM:
    """LLM that sleeps for `slow_seconds` before yielding events.

    Use this to simulate a slow/hanging backend for timeout tests.
    """

    def __init__(self, slow_seconds: float, events: list[StreamEvent] | None = None) -> None:
        self._slow_seconds = slow_seconds
        self._events = events or [
            StreamEvent(type="text", text="slow response"),
            StreamEvent(type="end"),
        ]
        self.stream_calls: int = 0

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        await asyncio.sleep(self._slow_seconds)
        for event in self._events:
            yield event


class SequentialLLM:
    """LLM that returns a different event sequence on each successive call."""

    def __init__(self, call_sequences: list[list[StreamEvent]]) -> None:
        assert call_sequences
        self._sequences = call_sequences
        self.stream_calls: int = 0

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        idx = min(self.stream_calls, len(self._sequences) - 1)
        self.stream_calls += 1
        for event in self._sequences[idx]:
            yield event


# ---------------------------------------------------------------------------
# TestCircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """CB1–CB5: Circuit breaker must be wired around self._llm.stream() in turn()."""

    @pytest.mark.asyncio
    async def test_llm_exception_records_failure_on_circuit_breaker(self):
        """CB1: When the LLM raises, the circuit breaker records a failure.

        After one LLM failure the circuit must NOT yet be open (threshold=3 by
        default or configured), but the internal failure count must have advanced.
        """
        from sr2.orchestrator import SR2

        failing_llm = FailingLLM(RuntimeError("backend down"))
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": failing_llm},
            token_counter=CharacterTokenCounter(),
        )

        # One failure should not open the circuit (threshold > 1).
        # It should propagate the exception while recording the failure.
        with pytest.raises(Exception):
            async for _ in sr2.turn(make_user_input()):
                pass

        # The circuit breaker must have recorded the failure.
        # Access via sr2._circuit_breaker (implementation detail exposed for testing).
        cb = sr2._circuit_breaker
        assert cb._failure_count >= 1, (
            f"Expected failure_count >= 1 after one LLM exception, "
            f"got {cb._failure_count}. "
            f"Is the circuit breaker wired around self._llm.stream()?"
        )

    @pytest.mark.asyncio
    async def test_three_consecutive_failures_open_circuit(self):
        """CB2: After `failure_threshold` (3) consecutive LLM failures the circuit opens.

        Uses a config with failure_threshold=3 (default). After the third
        failure, CircuitBreaker.state must be OPEN.
        """
        from sr2.orchestrator import SR2
        from sr2.degradation.circuit_breaker import CircuitState

        failing_llm = FailingLLM(RuntimeError("backend down"))
        config = make_minimal_config(circuit_breaker_failure_threshold=3)
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": failing_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Three failing turns.
        for _ in range(3):
            with pytest.raises(Exception):
                async for _ in sr2.turn(make_user_input()):
                    pass

        cb = sr2._circuit_breaker
        assert cb.state == CircuitState.OPEN, (
            f"Expected circuit to be OPEN after 3 consecutive LLM failures, "
            f"got state={cb.state!r}."
        )

    @pytest.mark.asyncio
    async def test_open_circuit_raises_without_calling_llm(self):
        """CB3: When the circuit is open, turn() raises CircuitBreakerOpenError
        and the LLM is NOT invoked.

        Verifies both the error type and that stream_calls is not incremented.
        """
        from sr2.orchestrator import SR2
        from sr2.degradation.circuit_breaker import CircuitBreakerOpenError  # noqa: F401

        failing_llm = FailingLLM(RuntimeError("backend down"))
        config = make_minimal_config(circuit_breaker_failure_threshold=3)
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": failing_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Force the circuit open via three failures.
        for _ in range(3):
            with pytest.raises(Exception):
                async for _ in sr2.turn(make_user_input()):
                    pass

        calls_before = failing_llm.stream_calls

        # The fourth call must raise CircuitBreakerOpenError without calling LLM.
        from sr2.degradation.circuit_breaker import CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            async for _ in sr2.turn(make_user_input()):
                pass

        assert failing_llm.stream_calls == calls_before, (
            f"LLM must NOT be called when circuit is open. "
            f"stream_calls went from {calls_before} to {failing_llm.stream_calls}."
        )

    @pytest.mark.asyncio
    async def test_circuit_allows_probe_after_recovery_timeout(self):
        """CB4: After `recovery_timeout` seconds the circuit moves to HALF_OPEN
        and allows exactly one probe call through.

        Uses a very short recovery_timeout (0.05 s) so the test doesn't hang.
        """
        from sr2.orchestrator import SR2
        from sr2.degradation.circuit_breaker import CircuitBreakerOpenError, CircuitState

        failing_llm = FailingLLM(RuntimeError("backend down"))
        # Short recovery timeout: 50 ms.
        config = make_minimal_config(
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout=0.05,
        )
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": failing_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Open the circuit.
        for _ in range(3):
            with pytest.raises(Exception):
                async for _ in sr2.turn(make_user_input()):
                    pass

        # Circuit is open — call must raise immediately.
        with pytest.raises(CircuitBreakerOpenError):
            async for _ in sr2.turn(make_user_input()):
                pass

        # Wait for recovery_timeout to elapse.
        await asyncio.sleep(0.1)

        # Now the circuit should allow one probe through (HALF_OPEN).
        # The probe will fail (LLM still broken), but it must not raise
        # CircuitBreakerOpenError — it raises the underlying LLM error instead.
        with pytest.raises(Exception) as exc_info:
            async for _ in sr2.turn(make_user_input()):
                pass

        assert not isinstance(exc_info.value, CircuitBreakerOpenError), (
            "After recovery_timeout the circuit should allow a probe. "
            "Expected the underlying LLM error, not CircuitBreakerOpenError."
        )

    @pytest.mark.asyncio
    async def test_successful_llm_call_resets_failure_count(self):
        """CB5: A successful LLM call resets the failure count to 0.

        After 2 failures (threshold=3) followed by 1 success, the circuit must
        remain CLOSED and failure_count must be 0.
        """
        from sr2.orchestrator import SR2
        from sr2.degradation.circuit_breaker import CircuitState

        failing_llm = FailingLLM(RuntimeError("backend down"))
        config = make_minimal_config(circuit_breaker_failure_threshold=3)
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": failing_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Two failures (circuit should still be CLOSED, count=2).
        for _ in range(2):
            with pytest.raises(Exception):
                async for _ in sr2.turn(make_user_input()):
                    pass

        cb = sr2._circuit_breaker
        assert cb.state == CircuitState.CLOSED, (
            "Circuit must remain CLOSED after 2 failures when threshold=3."
        )
        assert cb._failure_count == 2

        # Replace the LLM with a successful one.
        good_llm = SimpleMockLLM([
            StreamEvent(type="text", text="All good."),
            StreamEvent(type="end"),
        ])
        sr2._llm = good_llm

        async for _ in sr2.turn(make_user_input()):
            pass

        assert cb.state == CircuitState.CLOSED, (
            "Circuit must remain CLOSED after a successful call."
        )
        assert cb._failure_count == 0, (
            f"Failure count must reset to 0 after a successful call, "
            f"got {cb._failure_count}."
        )


# ---------------------------------------------------------------------------
# TestLLMTimeout
# ---------------------------------------------------------------------------


class TestLLMTimeout:
    """TO1–TO4: llm_timeout_seconds config field bounds each LLM stream call."""

    def test_pipeline_config_has_llm_timeout_seconds_field(self):
        """TO1: PipelineConfig.llm_timeout_seconds must exist and default to None."""
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "hi"})],
                )
            ]
        )
        assert hasattr(config, "llm_timeout_seconds"), (
            "PipelineConfig must have a 'llm_timeout_seconds' field. "
            "Add: llm_timeout_seconds: float | None = None"
        )
        assert config.llm_timeout_seconds is None, (
            "llm_timeout_seconds must default to None (no timeout)."
        )

    def test_pipeline_config_accepts_float_llm_timeout(self):
        """TO1 (corollary): PipelineConfig accepts a positive float for llm_timeout_seconds."""
        config = make_minimal_config(llm_timeout_seconds=30.0)
        assert config.llm_timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_llm_timeout_raises_when_stream_is_slow(self):
        """TO2 + TO3: When llm_timeout_seconds is set and the LLM stream takes longer,
        turn() raises asyncio.TimeoutError (or a subclass / wrapper of it).

        The SlowLLM sleeps for 1 s; we set a 0.05 s timeout → must time out.
        """
        from sr2.orchestrator import SR2

        slow_llm = SlowLLM(slow_seconds=1.0)
        config = make_minimal_config(llm_timeout_seconds=0.05)
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": slow_llm},
            token_counter=CharacterTokenCounter(),
        )

        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            async for _ in sr2.turn(make_user_input()):
                pass

    @pytest.mark.asyncio
    async def test_llm_timeout_raises_asyncio_timeout_error_type(self):
        """TO3 (strict): The raised exception must be asyncio.TimeoutError or a subclass.

        Catches the exception and checks isinstance to allow implementation-defined
        wrapper classes that subclass asyncio.TimeoutError.
        """
        from sr2.orchestrator import SR2

        slow_llm = SlowLLM(slow_seconds=1.0)
        config = make_minimal_config(llm_timeout_seconds=0.05)
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": slow_llm},
            token_counter=CharacterTokenCounter(),
        )

        raised: BaseException | None = None
        try:
            async for _ in sr2.turn(make_user_input()):
                pass
        except BaseException as exc:
            raised = exc

        assert raised is not None, "Expected an exception from slow LLM with short timeout."
        assert isinstance(raised, (asyncio.TimeoutError, TimeoutError)), (
            f"Expected asyncio.TimeoutError or TimeoutError, got {type(raised).__name__!r}. "
            f"The implementation must wrap asyncio.wait_for() or raise asyncio.TimeoutError."
        )

    @pytest.mark.asyncio
    async def test_no_timeout_when_llm_timeout_seconds_is_none(self):
        """TO4: When llm_timeout_seconds=None (default), slow LLMs complete normally.

        The SlowLLM sleeps for 0.1 s — with no timeout configured this must
        complete without error.
        """
        from sr2.orchestrator import SR2

        slow_llm = SlowLLM(
            slow_seconds=0.1,
            events=[
                StreamEvent(type="text", text="eventually done"),
                StreamEvent(type="end"),
            ],
        )
        config = make_minimal_config()  # llm_timeout_seconds=None by default
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": slow_llm},
            token_counter=CharacterTokenCounter(),
        )

        events = [e async for e in sr2.turn(make_user_input())]
        text_events = [e for e in events if e.type == "text"]

        assert len(text_events) >= 1, (
            "With llm_timeout_seconds=None, slow LLM must complete normally. "
            "No timeout should be applied."
        )
        assert text_events[0].text == "eventually done"

    @pytest.mark.asyncio
    async def test_timeout_applies_per_llm_call_not_total_turn(self):
        """TO2 (per-call): The timeout is applied per LLM stream call, not over the whole turn.

        A two-iteration turn where each LLM call takes 0.08 s. With a 0.15 s timeout,
        each individual call completes in time, so the turn succeeds overall.
        Each call: 0.08 s < 0.15 s → no timeout.
        """
        from sr2.orchestrator import SR2
        from sr2.models import ToolResultBlock, ToolUseBlock

        async def stub_executor(block: ToolUseBlock) -> ToolResultBlock:
            return ToolResultBlock(tool_use_id=block.id, content="result")

        # Each LLM call takes 0.08 s, well under the 0.15 s per-call timeout.
        class SlowSequentialLLM:
            """Two-call LLM: first returns tool_use, second returns text. Both slow."""

            def __init__(self, per_call_delay: float) -> None:
                self._delay = per_call_delay
                self.stream_calls = 0

            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                self.stream_calls += 1
                await asyncio.sleep(self._delay)
                if self.stream_calls == 1:
                    yield StreamEvent(
                        type="tool_use",
                        tool_use_id="call_001",
                        tool_name="get_weather",
                        tool_input={"location": "Oslo"},
                    )
                    yield StreamEvent(type="end")
                else:
                    yield StreamEvent(type="text", text="Final answer.")
                    yield StreamEvent(type="end")

        slow_llm = SlowSequentialLLM(per_call_delay=0.08)
        config = make_minimal_config(llm_timeout_seconds=0.15)
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": slow_llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        # Must complete without timeout — each call is within the per-call limit.
        events = [e async for e in sr2.turn(make_user_input())]
        assert any(e.type == "end" for e in events), (
            "Turn must complete successfully when each LLM call is within the per-call timeout. "
            "The timeout is per-call, not over the whole turn."
        )
        assert slow_llm.stream_calls == 2, (
            f"Expected 2 LLM calls (tool loop), got {slow_llm.stream_calls}."
        )
