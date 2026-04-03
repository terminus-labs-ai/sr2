"""Tests for compaction rules."""

import pytest

from sr2.compaction.rules import (
    CollapseRule,
    CompactionInput,
    ReferenceRule,
    ResultSummaryRule,
    SchemaAndSampleRule,
    SupersedeRule,
    get_rule,
)


class TestSchemaAndSampleRule:
    """Tests for SchemaAndSampleRule."""

    def test_long_content_compacted(self):
        """Long content is compacted with line count and sample."""
        content = "\n".join(f"line {i}" for i in range(20))
        inp = CompactionInput(content=content, content_type="tool_output", tokens=200)
        rule = SchemaAndSampleRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "20 lines" in output.content
        assert "Sample:" in output.content

    def test_short_content_not_compacted(self):
        """Short content (<=3 lines) is not compacted."""
        content = "line 1\nline 2\nline 3"
        inp = CompactionInput(content=content, content_type="tool_output", tokens=20)
        rule = SchemaAndSampleRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is False
        assert output.content == content


class TestReferenceRule:
    """Tests for ReferenceRule."""

    def test_file_content_replaced(self):
        """File content replaced with path + metadata."""
        inp = CompactionInput(
            content="lots of code...",
            content_type="file_content",
            tokens=500,
            metadata={"file_path": "/src/main.py", "line_count": 200, "language": "python"},
        )
        rule = ReferenceRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "/src/main.py" in output.content
        assert "200 lines" in output.content
        assert "python" in output.content
        assert output.recovery_hint == 'read_file("/src/main.py")'

    def test_missing_metadata(self):
        """Missing metadata handled gracefully."""
        inp = CompactionInput(content="data", content_type="file_content", tokens=100)
        rule = ReferenceRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "unknown" in output.content


class TestResultSummaryRule:
    """Tests for ResultSummaryRule."""

    def test_exit_code_zero(self):
        """Exit code 0 shows checkmark."""
        inp = CompactionInput(
            content="test passed\nall good\nno errors",
            content_type="code_execution",
            tokens=50,
            metadata={"exit_code": 0},
        )
        rule = ResultSummaryRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "\u2713" in output.content
        assert "Exit 0" in output.content

    def test_exit_code_nonzero(self):
        """Exit code 1 shows X mark."""
        inp = CompactionInput(
            content="error occurred\ntraceback...\nfailed",
            content_type="code_execution",
            tokens=50,
            metadata={"exit_code": 1},
        )
        rule = ResultSummaryRule()
        output = rule.compact(inp, {})

        assert "\u2717" in output.content
        assert "Exit 1" in output.content

    def test_output_truncated(self):
        """Output truncated to max_output_lines."""
        content = "\n".join(f"line {i}" for i in range(20))
        inp = CompactionInput(
            content=content, content_type="code_execution", tokens=200,
            metadata={"exit_code": 0},
        )
        rule = ResultSummaryRule()
        output = rule.compact(inp, {"max_output_lines": 2})

        assert "more lines" in output.content


class TestSupersedeRule:
    """Tests for SupersedeRule."""

    def test_superseded_marker(self):
        """Produces superseded marker."""
        inp = CompactionInput(
            content="old data", content_type="tool_output", tokens=100,
            metadata={"superseded_by_turn": "turn 5"},
        )
        rule = SupersedeRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "superseded by turn 5" in output.content


class TestCollapseRule:
    """Tests for CollapseRule."""

    def test_collapse_to_oneliner(self):
        """Produces one-liner with tool name."""
        inp = CompactionInput(
            content="file saved successfully", content_type="confirmation", tokens=20,
            metadata={"tool_name": "write_file", "args_summary": "main.py"},
        )
        rule = CollapseRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "write_file(main.py)" in output.content


class TestGetRule:
    """Tests for get_rule registry function."""

    def test_get_existing_rule(self):
        """get_rule returns correct rule."""
        rule = get_rule("schema_and_sample")
        assert isinstance(rule, SchemaAndSampleRule)

    def test_get_nonexistent_rule(self):
        """get_rule raises KeyError for unknown strategy."""
        with pytest.raises(KeyError, match="Unknown compaction strategy"):
            get_rule("nonexistent")
