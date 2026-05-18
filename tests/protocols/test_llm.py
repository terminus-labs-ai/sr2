"""Tests for sr2.protocols.llm — StreamEvent model + LLMCallable protocol.

Covers:
  1. StreamEvent model construction — text, usage, and end events with correct defaults.
  2. StreamEvent type validation — only "text", "usage", "end" are valid.
  3. LLMCallable protocol compliance — requires both complete() and stream().
  4. StreamEvent serialization — round-trip through model_dump / model_validate.
"""

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from sr2.models import TokenUsage
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMCallable,
    StreamEvent,
)


# ---------------------------------------------------------------------------
# 1. StreamEvent model construction
# ---------------------------------------------------------------------------


class TestStreamEventConstruction:
    def test_text_event(self):
        event = StreamEvent(type="text", text="hello")
        assert event.type == "text"
        assert event.text == "hello"
        assert event.usage is None

    def test_text_event_default_text(self):
        event = StreamEvent(type="text")
        assert event.text == ""

    def test_usage_event(self):
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        event = StreamEvent(type="usage", usage=usage)
        assert event.type == "usage"
        assert event.usage is not None
        assert event.usage.input_tokens == 10
        assert event.usage.output_tokens == 5
        assert event.text == ""

    def test_end_event(self):
        event = StreamEvent(type="end")
        assert event.type == "end"
        assert event.text == ""
        assert event.usage is None

    def test_end_event_ignores_text(self):
        """End events can carry text (empty by default), but usage should be None."""
        event = StreamEvent(type="end", text="ignored")
        assert event.text == "ignored"

    def test_usage_event_with_cache_fields(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )
        event = StreamEvent(type="usage", usage=usage)
        assert event.usage.cache_creation_input_tokens == 20
        assert event.usage.cache_read_input_tokens == 30


# ---------------------------------------------------------------------------
# 2. StreamEvent type validation
# ---------------------------------------------------------------------------


class TestStreamEventTypeValidation:
    def test_valid_types(self):
        for t in ("text", "usage", "end"):
            event = StreamEvent(type=t)
            assert event.type == t

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            StreamEvent(type="start")

    def test_invalid_type_arbitrary_string(self):
        with pytest.raises(ValidationError):
            StreamEvent(type="content_block_delta")

    def test_empty_type_rejected(self):
        with pytest.raises(ValidationError):
            StreamEvent(type="")

    def test_missing_type_rejected(self):
        with pytest.raises(ValidationError):
            StreamEvent()  # type is required (Literal, no default)


# ---------------------------------------------------------------------------
# 3. LLMCallable protocol compliance
# ---------------------------------------------------------------------------


class _FullImpl:
    """Implements both complete() and stream()."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            id="r1",
            content=[],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="text", text="hi")
        yield StreamEvent(type="end")


class _MissingStream:
    """Implements complete() only — should NOT satisfy LLMCallable."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            id="r1",
            content=[],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )


class _MissingComplete:
    """Implements stream() only — should NOT satisfy LLMCallable."""

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="end")


class TestLLMCallableProtocol:
    def test_full_impl_satisfies_protocol(self):
        assert isinstance(_FullImpl(), LLMCallable)

    def test_missing_stream_does_not_satisfy(self):
        assert not isinstance(_MissingStream(), LLMCallable)

    def test_missing_complete_does_not_satisfy(self):
        assert not isinstance(_MissingComplete(), LLMCallable)

    def test_protocol_is_runtime_checkable(self):
        """If @runtime_checkable is missing, isinstance() raises TypeError."""
        assert isinstance(_FullImpl(), LLMCallable)


# ---------------------------------------------------------------------------
# 4. StreamEvent serialization round-trip
# ---------------------------------------------------------------------------


class TestStreamEventSerialization:
    def test_text_event_round_trip(self):
        original = StreamEvent(type="text", text="hello world")
        data = original.model_dump()
        restored = StreamEvent.model_validate(data)
        assert restored == original

    def test_usage_event_round_trip(self):
        usage = TokenUsage(
            input_tokens=42,
            output_tokens=7,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        )
        original = StreamEvent(type="usage", usage=usage)
        data = original.model_dump()
        restored = StreamEvent.model_validate(data)
        assert restored == original
        assert restored.usage.input_tokens == 42

    def test_end_event_round_trip(self):
        original = StreamEvent(type="end")
        data = original.model_dump()
        restored = StreamEvent.model_validate(data)
        assert restored == original

    def test_serialized_shape_text(self):
        event = StreamEvent(type="text", text="hi")
        data = event.model_dump()
        assert data["type"] == "text"
        assert data["text"] == "hi"
        assert data["usage"] is None

    def test_serialized_shape_usage(self):
        event = StreamEvent(type="usage", usage=TokenUsage(input_tokens=1))
        data = event.model_dump()
        assert data["type"] == "usage"
        assert data["usage"]["input_tokens"] == 1
