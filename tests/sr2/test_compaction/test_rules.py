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

    def test_recovery_hint_with_tool_name(self):
        """recovery_hint=True produces 'Re-fetch with {tool_name}' format."""
        content = "\n".join(f"line {i}" for i in range(20))
        inp = CompactionInput(
            content=content, content_type="tool_output", tokens=200,
            metadata={"tool_name": "search_files"},
        )
        rule = SchemaAndSampleRule()
        output = rule.compact(inp, {"recovery_hint": True})

        assert output.recovery_hint == "Re-fetch with search_files"

    def test_recovery_hint_missing_tool_name_falls_back(self):
        """When metadata has no tool_name, recovery hint uses 'the tool'."""
        content = "\n".join(f"line {i}" for i in range(20))
        inp = CompactionInput(
            content=content, content_type="tool_output", tokens=200,
            metadata={},
        )
        rule = SchemaAndSampleRule()
        output = rule.compact(inp, {"recovery_hint": True})

        assert output.recovery_hint == "Re-fetch with the tool"

    def test_recovery_hint_disabled(self):
        """recovery_hint=False (or absent) produces no hint."""
        content = "\n".join(f"line {i}" for i in range(20))
        inp = CompactionInput(
            content=content, content_type="tool_output", tokens=200,
            metadata={"tool_name": "search_files"},
        )
        rule = SchemaAndSampleRule()
        output = rule.compact(inp, {})

        assert output.recovery_hint is None

    def test_recovery_hint_not_compacted_short_content(self):
        """Short content (not compacted) produces no recovery hint."""
        content = "line 1\nline 2\nline 3"
        inp = CompactionInput(
            content=content, content_type="tool_output", tokens=20,
            metadata={"tool_name": "search_files"},
        )
        rule = SchemaAndSampleRule()
        output = rule.compact(inp, {"recovery_hint": True})

        assert output.was_compacted is False
        assert output.recovery_hint is None


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

    def test_recovery_hint_always_produced(self):
        """ReferenceRule always produces a recovery hint, regardless of config."""
        inp = CompactionInput(
            content="lots of code...",
            content_type="file_content",
            tokens=500,
            metadata={"file_path": "/src/utils.py"},
        )
        rule = ReferenceRule()
        # Even without recovery_hint in config, ReferenceRule always sets it
        output = rule.compact(inp, {})

        assert output.recovery_hint == 'read_file("/src/utils.py")'

    def test_missing_metadata_recovery_hint_uses_unknown(self):
        """Missing metadata produces recovery hint with 'unknown' path."""
        inp = CompactionInput(content="data", content_type="file_content", tokens=100)
        rule = ReferenceRule()
        output = rule.compact(inp, {})

        assert output.was_compacted is True
        assert "unknown" in output.content
        assert output.recovery_hint == 'read_file("unknown")'


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

    def test_recovery_hint_from_result_path(self):
        """recovery_hint comes from metadata.result_path."""
        inp = CompactionInput(
            content="test passed\nall good",
            content_type="code_execution",
            tokens=50,
            metadata={"exit_code": 0, "result_path": "/tmp/result_abc.log"},
        )
        rule = ResultSummaryRule()
        output = rule.compact(inp, {})

        assert output.recovery_hint == "/tmp/result_abc.log"

    def test_recovery_hint_none_when_no_result_path(self):
        """Without result_path in metadata, recovery_hint is None."""
        inp = CompactionInput(
            content="test passed\nall good",
            content_type="code_execution",
            tokens=50,
            metadata={"exit_code": 0},
        )
        rule = ResultSummaryRule()
        output = rule.compact(inp, {})

        assert output.recovery_hint is None


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
