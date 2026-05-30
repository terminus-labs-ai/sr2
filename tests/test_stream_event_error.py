"""Tests for StreamEvent error type (sr2-3 / FR10)."""

import pytest
from sr2.protocols.llm import StreamEvent


class TestStreamEventTypeError:
    """Verify 'error' is a valid StreamEvent type with errors field."""

    def test_error_type_is_valid(self):
        """StreamEvent accepts type='error'."""
        event = StreamEvent(type="error", errors=["something went wrong"])
        assert event.type == "error"
        assert event.errors == ["something went wrong"]

    def test_error_with_multiple_errors(self):
        """errors field can hold multiple strings."""
        event = StreamEvent(
            type="error",
            errors=["error one", "error two", "error three"],
        )
        assert event.errors is not None
        assert len(event.errors) == 3

    def test_errors_field_defaults_to_none(self):
        """errors field is None by default for non-error events."""
        event = StreamEvent(type="text", text="hello")
        assert event.errors is None

    def test_errors_field_optional_on_error_type(self):
        """errors field is optional even on type='error' (can be None)."""
        event = StreamEvent(type="error")
        assert event.errors is None

    def test_existing_types_still_work(self):
        """All existing StreamEvent types remain valid."""
        for t in ("text", "usage", "end", "tool_use", "iteration_complete", "tool_use_emitted", "tool_result_received"):
            event = StreamEvent(type=t)
            assert event.type == t
            assert event.errors is None
