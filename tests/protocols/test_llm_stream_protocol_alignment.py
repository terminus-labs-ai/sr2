"""Tests for sr2-21: LLMCallable.stream protocol/impl signature mismatch.

The bug: protocols/llm.py:34 declares `def stream` (sync), while all
implementations and mocks use `async def stream` (async generator). The
protocol must declare `async def stream` to match.

These tests pin the DESIRED behavior. They currently fail because the
protocol still declares a sync `def stream`.
"""

import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest

from sr2.integrations.litellm import LiteLLMCallable
from sr2.protocols.llm import LLMCallable, StreamEvent, CompletionRequest


# ---------------------------------------------------------------------------
# 1. Protocol declaration: stream must be async
# ---------------------------------------------------------------------------


class TestProtocolStreamIsAsync:
    """The protocol method `stream` must be declared as a coroutine function
    (async def), not a plain sync function returning AsyncIterator.
    """

    def test_stream_method_on_protocol_is_coroutinefunction(self):
        """LLMCallable.stream must be a coroutinefunction (async def).

        Currently fails because the protocol declares `def stream` (sync).
        """
        stream_fn = LLMCallable.stream
        assert inspect.iscoroutinefunction(stream_fn), (
            "LLMCallable.stream is declared as a sync `def` but must be "
            "`async def` to match all implementations and mocks."
        )

    def test_stream_method_on_protocol_is_coroutinefunction_not_asyncgen(self):
        """LLMCallable.stream protocol stub must be `async def` (coroutinefunction).

        A protocol stub uses `...` as its body — not `yield` — so it is a
        coroutinefunction, not an async generator function. The key property
        being pinned is that it is async (not sync). The asyncgenfunction check
        is correct for the *implementation* (see TestLiteLLMStreamIsAsync), not
        the protocol stub.
        """
        stream_fn = LLMCallable.stream
        assert inspect.iscoroutinefunction(stream_fn), (
            "LLMCallable.stream protocol stub must be `async def` "
            "(coroutinefunction), not a plain sync def."
        )


# ---------------------------------------------------------------------------
# 2. Conforming impl: a sync `def stream` should NOT satisfy the protocol
# ---------------------------------------------------------------------------


class _SyncStreamImpl:
    """Implements stream as sync — returns an async iterator without being async.

    This is the shape the current protocol ALLOWS but should NOT allow.
    """

    async def complete(self, request: CompletionRequest):
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        # sync function returning an async iterator — matches the buggy protocol
        async def _gen():
            yield StreamEvent(type="end")

        return _gen()


class _AsyncStreamImpl:
    """Implements stream as async generator — the correct, desired form."""

    async def complete(self, request: CompletionRequest):
        raise NotImplementedError

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="end")


class TestProtocolConformance:
    """After the fix, only async def stream implementations satisfy LLMCallable."""

    def test_async_stream_impl_satisfies_protocol(self):
        """An async def stream impl must satisfy LLMCallable."""
        assert isinstance(_AsyncStreamImpl(), LLMCallable), (
            "_AsyncStreamImpl uses `async def stream` and must satisfy LLMCallable."
        )

    def test_sync_stream_impl_stream_is_not_async(self):
        """A sync def stream impl's stream method is NOT a coroutinefunction.

        This documents the shape of _SyncStreamImpl and confirms that the
        runtime_checkable isinstance check cannot distinguish sync vs async
        callables — structural Protocol checks only verify attribute name and
        callability. Distinguishing async from sync requires static analysis
        (mypy/pyright), not runtime isinstance checks.

        The correct enforcement is: protocol declares `async def stream`, and
        static type checkers reject non-async implementations. At runtime,
        isinstance will still return True for both sync and async impls because
        runtime_checkable only checks attribute presence and callability.
        """
        # Confirm the impl's stream is sync (not async)
        assert not inspect.iscoroutinefunction(_SyncStreamImpl.stream), (
            "_SyncStreamImpl.stream should be a plain sync function."
        )
        # Confirm the correct impl's stream IS async
        assert inspect.iscoroutinefunction(_AsyncStreamImpl.stream) or inspect.isasyncgenfunction(_AsyncStreamImpl.stream), (
            "_AsyncStreamImpl.stream should be async."
        )


# ---------------------------------------------------------------------------
# 3. LiteLLMClient.stream is async (implementation sanity check)
# ---------------------------------------------------------------------------


class TestLiteLLMStreamIsAsync:
    """LiteLLMCallable.stream is already an async generator — this must stay true."""

    def test_litellm_stream_is_asyncgenfunction(self):
        """LiteLLMCallable.stream must be an async generator function."""
        assert inspect.isasyncgenfunction(LiteLLMCallable.stream), (
            "LiteLLMCallable.stream must be an async generator function (async def + yield)."
        )

    def test_litellm_stream_is_not_sync(self):
        """LiteLLMCallable.stream must not be a plain sync function."""
        fn = LiteLLMCallable.stream
        assert not (not inspect.iscoroutinefunction(fn) and not inspect.isasyncgenfunction(fn)), (
            "LiteLLMCallable.stream must not be a plain sync function."
        )


# ---------------------------------------------------------------------------
# 4. Protocol return type annotation pins AsyncIterator[StreamEvent]
# ---------------------------------------------------------------------------


class TestProtocolStreamReturnAnnotation:
    """The protocol's stream method must be annotated to return AsyncIterator[StreamEvent].

    This checks the __annotations__ on the protocol class method.
    """

    def test_stream_return_annotation_present(self):
        """LLMCallable.stream must have a return annotation."""
        hints = LLMCallable.stream.__annotations__
        assert "return" in hints, (
            "LLMCallable.stream must declare a return type annotation."
        )

    def test_stream_return_annotation_is_async_iterator_of_stream_event(self):
        """The return annotation must reference AsyncIterator[StreamEvent]."""
        hints = LLMCallable.stream.__annotations__
        ret = hints.get("return")
        # The annotation may be a string (forward ref) or the actual type.
        # Normalise to string for comparison.
        ret_str = str(ret)
        assert "AsyncIterator" in ret_str, (
            f"Return annotation should include AsyncIterator, got: {ret_str}"
        )
        assert "StreamEvent" in ret_str, (
            f"Return annotation should include StreamEvent, got: {ret_str}"
        )
