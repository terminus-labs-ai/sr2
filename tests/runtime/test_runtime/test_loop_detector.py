"""Tests for the loop detector."""

import pytest

from sr2_runtime.llm.loop import ToolCallRecord
from sr2_runtime.llm.loop_detector import LoopDetection, detect_loop


def _record(name: str, args: dict | None = None, result: str = "ok") -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=name,
        arguments=args or {},
        result=result,
        duration_ms=10.0,
        success=True,
    )


class TestDetectLoop:
    """Tests for detect_loop()."""

    def test_no_calls_returns_false(self):
        assert not detect_loop([]).detected

    def test_below_threshold_returns_false(self):
        calls = [_record("bash", {"cmd": "ls"}), _record("bash", {"cmd": "ls"})]
        assert not detect_loop(calls, threshold=3).detected

    def test_identical_args_detected(self):
        calls = [
            _record("recall_memory", {"query": "auth"}),
            _record("recall_memory", {"query": "auth"}),
            _record("recall_memory", {"query": "auth"}),
        ]
        result = detect_loop(calls, window=6, threshold=3)
        assert result.detected
        assert result.tool_name == "recall_memory"
        assert result.pattern == "identical_args"
        assert result.count == 3

    def test_identical_args_mixed_tools_no_false_positive(self):
        """Different tools with different args and results should not trigger."""
        calls = [
            _record("bash", {"cmd": "ls"}, result="dir listing"),
            _record("view_file", {"path": "a.py"}, result="file a contents"),
            _record("bash", {"cmd": "cat"}, result="cat output"),
            _record("view_file", {"path": "b.py"}, result="file b contents"),
            _record("bash", {"cmd": "pwd"}, result="/home/user"),
        ]
        assert not detect_loop(calls).detected

    def test_same_tool_different_args_no_trigger(self):
        """Same tool but with different arguments and results each time is fine."""
        calls = [
            _record("bash", {"cmd": "ls"}, result="dir listing"),
            _record("bash", {"cmd": "cat foo.py"}, result="file contents"),
            _record("bash", {"cmd": "grep bar"}, result="match found"),
            _record("bash", {"cmd": "pwd"}, result="/home/user"),
        ]
        assert not detect_loop(calls).detected

    def test_same_tool_dominant_same_results(self):
        calls = [
            _record("bash", {"cmd": "grep sensors a.py"}, result="line 42: sensors"),
            _record("view_file", {"path": "b.py"}, result="contents"),
            _record("bash", {"cmd": "grep -n sensors a.py"}, result="line 42: sensors"),
            _record("bash", {"cmd": "grep sensors a.py"}, result="line 42: sensors"),
        ]
        result = detect_loop(calls, window=6, threshold=3)
        assert result.detected
        assert result.tool_name == "bash"
        assert result.pattern == "same_tool_dominant"

    def test_same_tool_dominant_all_unique_args_no_trigger(self):
        """Same tool, same results, but every call has unique args — not a loop."""
        calls = [
            _record("bash", {"cmd": "grep foo a.py"}, result="not found"),
            _record("bash", {"cmd": "grep bar b.py"}, result="not found"),
            _record("bash", {"cmd": "grep baz c.py"}, result="not found"),
        ]
        assert not detect_loop(calls, window=6, threshold=3).detected

    def test_same_tool_dominant_different_results_no_trigger(self):
        """Same tool dominating but with meaningfully different results is OK."""
        calls = [
            _record("bash", {"cmd": "grep a"}, result="found in file1"),
            _record("bash", {"cmd": "grep b"}, result="found in file2"),
            _record("bash", {"cmd": "grep c"}, result="found in file3"),
        ]
        assert not detect_loop(calls, window=6, threshold=3).detected

    def test_window_limits_scope(self):
        """Old identical calls outside the window don't trigger detection."""
        old_calls = [_record("recall_memory", {"query": "x"})] * 5
        new_calls = [
            _record("bash", {"cmd": "ls"}, result="dir listing"),
            _record("view_file", {"path": "a.py"}, result="file a"),
            _record("bash", {"cmd": "cat"}, result="cat output"),
            _record("view_file", {"path": "b.py"}, result="file b"),
            _record("bash", {"cmd": "pwd"}, result="/home"),
            _record("view_file", {"path": "c.py"}, result="file c"),
        ]
        assert not detect_loop(old_calls + new_calls, window=6, threshold=3).detected

    def test_custom_threshold(self):
        calls = [_record("bash", {"cmd": "ls"})] * 5
        assert not detect_loop(calls, threshold=6).detected
        assert detect_loop(calls, threshold=5).detected

    def test_four_identical_in_large_window(self):
        """Identical calls detected even in larger window."""
        calls = [
            _record("view_file", {"path": "x.py"}, result="x"),
            _record("recall_memory", {"query": "auth"}, result="no results"),
            _record("recall_memory", {"query": "auth"}, result="no results"),
            _record("view_file", {"path": "y.py"}, result="y"),
            _record("recall_memory", {"query": "auth"}, result="no results"),
            _record("recall_memory", {"query": "auth"}, result="no results"),
        ]
        result = detect_loop(calls, window=6, threshold=4)
        assert result.detected
        assert result.pattern == "identical_args"
        assert result.count == 4
