"""Tests for shared LoopResult and ToolCallRecord models."""

from __future__ import annotations

from sr2.models.loop_result import LoopResult, ToolCallRecord


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_required_fields(self):
        record = ToolCallRecord(
            tool_name="bash",
            arguments={"command": "ls"},
            result="file.txt",
            duration_ms=42.0,
            success=True,
        )
        assert record.tool_name == "bash"
        assert record.success is True

    def test_optional_defaults(self):
        record = ToolCallRecord(
            tool_name="read",
            arguments={},
            result="",
            duration_ms=0.0,
            success=False,
        )
        assert record.error is None
        assert record.call_id == ""
        assert record.iteration == 0


class TestLoopResult:
    """Tests for LoopResult dataclass."""

    def test_defaults(self):
        result = LoopResult(response_text="hello")
        assert result.tool_calls == []
        assert result.iterations == 0
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.cached_tokens == 0
        assert result.stopped_reason == "complete"

    def test_total_tokens(self):
        result = LoopResult(
            response_text="",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert result.total_tokens == 1500

    def test_cache_hit_rate(self):
        result = LoopResult(
            response_text="",
            total_input_tokens=1000,
            cached_tokens=750,
        )
        assert result.cache_hit_rate == 0.75

    def test_cache_hit_rate_zero_input(self):
        """Zero input tokens should return 0.0, not raise ZeroDivisionError."""
        result = LoopResult(response_text="")
        assert result.cache_hit_rate == 0.0


class TestImportPaths:
    """Verify re-export paths work."""

    def test_import_from_sr2_models(self):
        from sr2.models import LoopResult as LR, ToolCallRecord as TCR

        assert LR is LoopResult
        assert TCR is ToolCallRecord

    def test_import_from_sr2_runtime_llm(self):
        from sr2_runtime.llm import LoopResult as LR, ToolCallRecord as TCR

        assert LR is LoopResult
        assert TCR is ToolCallRecord
