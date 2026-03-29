"""Tests for SR2Runtime result types."""

from sr2.runtime.result import RuntimeMetrics, RuntimeResult


class TestRuntimeMetrics:
    def test_defaults(self):
        m = RuntimeMetrics()
        assert m.total_tokens == 0
        assert m.prompt_tokens == 0
        assert m.completion_tokens == 0
        assert m.llm_calls == 0
        assert m.tool_calls == 0
        assert m.compaction_events == 0
        assert m.cache_hit_rate == 0.0
        assert m.wall_time_ms == 0.0

    def test_to_dict_defaults(self):
        m = RuntimeMetrics()
        d = m.to_dict()
        assert d == {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "compaction_events": 0,
            "cache_hit_rate": 0.0,
            "wall_time_ms": 0.0,
        }

    def test_to_dict_round_trips(self):
        m = RuntimeMetrics(
            total_tokens=150,
            prompt_tokens=100,
            completion_tokens=50,
            llm_calls=3,
            tool_calls=2,
            compaction_events=1,
            cache_hit_rate=0.85,
            wall_time_ms=1234.5,
        )
        d = m.to_dict()
        reconstructed = RuntimeMetrics(**d)
        assert reconstructed == m
        assert reconstructed.to_dict() == d


class TestRuntimeResult:
    def test_defaults(self):
        r = RuntimeResult(output="hello")
        assert r.output == "hello"
        assert r.success is True
        assert r.error is None
        assert r.metrics == RuntimeMetrics()
        assert r.tool_results == []
        assert r.metadata == {}

    def test_error_result(self):
        r = RuntimeResult(output="", success=False, error="something broke")
        assert r.output == ""
        assert r.success is False
        assert r.error == "something broke"

    def test_with_tool_results(self):
        tools = [
            {"tool": "search", "result": "found 3 items"},
            {"tool": "read_file", "result": "contents here"},
        ]
        r = RuntimeResult(output="done", tool_results=tools)
        assert len(r.tool_results) == 2
        assert r.tool_results[0]["tool"] == "search"
        assert r.tool_results[1]["result"] == "contents here"

    def test_with_metadata(self):
        meta = {"agent_id": "edi", "session": "abc-123", "tags": ["test"]}
        r = RuntimeResult(output="ok", metadata=meta)
        assert r.metadata["agent_id"] == "edi"
        assert r.metadata["session"] == "abc-123"
        assert r.metadata["tags"] == ["test"]

    def test_populated_metrics(self):
        m = RuntimeMetrics(total_tokens=200, llm_calls=5, wall_time_ms=500.0)
        r = RuntimeResult(output="result", metrics=m)
        assert r.metrics.total_tokens == 200
        assert r.metrics.llm_calls == 5
        assert r.metrics.to_dict()["wall_time_ms"] == 500.0
