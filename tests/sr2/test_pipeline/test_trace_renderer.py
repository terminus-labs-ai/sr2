"""Tests for CLI trace renderers."""

import time

from sr2.pipeline.trace import TraceEvent, TurnTrace
from sr2.pipeline.trace_renderer import render_brief, render_default, render_full


def _make_trace(**overrides) -> TurnTrace:
    """Build a realistic TurnTrace with all events populated."""
    turn = overrides.get("turn_number", 12)
    t = TurnTrace(
        turn_number=turn,
        session_id="test-session",
        interface_name="task_runner",
        started_at=time.monotonic(),
    )

    t.add(TraceEvent(turn=turn, stage="input", timestamp=time.monotonic(), duration_ms=0.1, data={
        "trigger_input": "deploy the auth service to staging",
        "session_turns": 12,
        "session_id": "test-session",
        "interface_name": "task_runner",
        "tool_count": 5,
    }))

    t.add(TraceEvent(turn=turn, stage="resolve", timestamp=time.monotonic(), duration_ms=23.0, data={
        "layers": [
            {"name": "core", "tokens": 1200, "cache_status": "hit", "items": 2, "circuit_breaker": "closed"},
            {"name": "memory", "tokens": 580, "cache_status": "hit", "items": 3, "circuit_breaker": "closed"},
            {"name": "conversation", "tokens": 2067, "cache_status": "append", "items": 12, "circuit_breaker": "closed"},
        ],
        "total_tokens": 3847,
        "budget": 8192,
        "utilization": 0.47,
        "cache_efficiency": 0.94,
    }))

    t.add(TraceEvent(turn=turn, stage="retrieve", timestamp=time.monotonic(), duration_ms=5.0, data={
        "query": "deploy the auth service to staging",
        "strategy": "hybrid",
        "candidates_scored": 10,
        "results_returned": 3,
        "top_k": 10,
        "threshold": 0.0,
        "results": [
            {"key": "user_preferences", "relevance_score": 0.91, "selected": True},
            {"key": "project_stack", "relevance_score": 0.87, "selected": True},
            {"key": "deploy_process", "relevance_score": 0.72, "selected": True},
            {"key": "code_style", "relevance_score": 0.28, "selected": False},
        ],
        "latency_ms": 5.0,
    }))

    t.add(TraceEvent(turn=turn, stage="zones", timestamp=time.monotonic(), duration_ms=0.1, data={
        "summarized": {"turns": 2, "tokens": 320},
        "compacted": {"turns": 7, "tokens": 1150},
        "raw": {"turns": 3, "tokens": 597},
        "session_notes": {"count": 0, "tokens": 0},
        "total_tokens": 2067,
    }))

    t.add(TraceEvent(turn=turn, stage="tool_state", timestamp=time.monotonic(), duration_ms=0.1, data={
        "current_state": "planning",
        "allowed_tools": ["read_file", "search", "check_status"],
        "denied_tools": ["execute_code", "deploy"],
        "tool_choice": "auto",
        "denied_attempts": 0,
    }))

    t.add(TraceEvent(turn=turn, stage="llm_request", timestamp=time.monotonic(), duration_ms=0.1, data={
        "message_count": 14,
        "tool_count": 5,
        "tool_choice": "auto",
    }))

    t.add(TraceEvent(turn=turn, stage="llm_response", timestamp=time.monotonic(), duration_ms=1240.0, data={
        "role": "assistant",
        "content_preview": "I'll check the current deployment status first...",
        "content_tokens": 340,
        "tool_calls": [{"name": "check_deploy_status", "arguments_preview": 'service="auth"'}],
        "latency_ms": 1240.0,
    }))

    t.add(TraceEvent(turn=turn, stage="post_process", timestamp=time.monotonic(), duration_ms=45.0, data={
        "memory_extraction": {"memories_extracted": 1, "conflicts_detected": 0},
        "compaction": {
            "turns_compacted": 4,
            "original_tokens": 2990,
            "compacted_tokens": 1150,
            "tokens_saved": 1840,
            "strategy": "rule_based",
            "details": [
                {"turn_number": 6, "content_type": "tool_output", "rule": "schema_and_sample",
                 "original_tokens": 890, "compacted_tokens": 120},
            ],
        },
        "summarization": None,
    }))

    t.add(TraceEvent(turn=turn, stage="metrics", timestamp=time.monotonic(), duration_ms=0.1, data={
        "token_savings_this_turn": 1840,
        "cache_efficiency": 0.94,
        "degradation_level": 0,
        "budget_utilization": 0.47,
    }))

    return t


def _make_minimal_trace() -> TurnTrace:
    """Trace with only an input event."""
    t = TurnTrace(turn_number=1, session_id="s", interface_name="test", started_at=time.monotonic())
    t.add(TraceEvent(turn=1, stage="input", timestamp=time.monotonic(), duration_ms=0.1, data={
        "trigger_input": "hello",
        "session_turns": 1,
        "session_id": "s",
        "interface_name": "test",
        "tool_count": 0,
    }))
    return t


def _make_warning_trace() -> TurnTrace:
    """Trace that triggers warnings (high utilization, circuit breaker open, low cache)."""
    t = TurnTrace(turn_number=5, session_id="s", interface_name="test", started_at=time.monotonic())
    t.add(TraceEvent(turn=5, stage="resolve", timestamp=time.monotonic(), duration_ms=10.0, data={
        "layers": [
            {"name": "core", "tokens": 1000, "cache_status": "hit", "items": 1, "circuit_breaker": "closed"},
            {"name": "memory", "tokens": 0, "cache_status": "skip", "items": 0, "circuit_breaker": "open"},
        ],
        "total_tokens": 7500,
        "budget": 8000,
        "utilization": 0.9375,
        "cache_efficiency": 0.3,
    }))
    return t


class TestRenderDefault:
    def test_basic_sections_present(self):
        result = render_default(_make_trace())
        assert "Turn 12" in result
        assert "INPUT" in result
        assert "PIPELINE" in result
        assert "LLM REQUEST" in result
        assert "LLM RESPONSE" in result
        assert "POST-PROCESS" in result
        assert "METRICS" in result

    def test_missing_events_no_crash(self):
        result = render_default(_make_minimal_trace())
        assert "Turn 1" in result
        assert "INPUT" in result
        assert "PIPELINE" not in result
        assert "LLM" not in result

    def test_with_warnings(self):
        result = render_default(_make_warning_trace())
        assert "⚠" in result

    def test_layer_table(self):
        result = render_default(_make_trace())
        assert "core" in result
        assert "memory" in result
        assert "conversation" in result

    def test_memory_retrieval_results(self):
        result = render_default(_make_trace())
        assert "user_preferences" in result
        assert "0.91" in result
        assert "3 results" in result

    def test_zones_display(self):
        result = render_default(_make_trace())
        assert "summarized" in result
        assert "compacted" in result
        assert "raw" in result

    def test_tool_calls_shown(self):
        result = render_default(_make_trace())
        assert "check_deploy_status" in result

    def test_circuit_breaker_open_shown(self):
        result = render_default(_make_warning_trace())
        assert "CIRCUIT OPEN" in result


class TestRenderFull:
    def test_returns_string(self):
        result = render_full(_make_trace())
        assert isinstance(result, str)
        assert "Turn 12" in result

    def test_includes_compaction_detail(self):
        result = render_full(_make_trace())
        assert "Compaction detail" in result
        assert "schema_and_sample" in result

    def test_no_detail_when_no_compaction(self):
        result = render_full(_make_minimal_trace())
        assert "Compaction detail" not in result


class TestRenderBrief:
    def test_starts_with_turn_number(self):
        result = render_brief(_make_trace())
        assert result.startswith("T12")

    def test_single_line(self):
        result = render_brief(_make_trace())
        assert "\n" not in result

    def test_contains_key_metrics(self):
        result = render_brief(_make_trace())
        assert "tk" in result
        assert "│" in result
        assert "mem" in result
        assert "cache" in result

    def test_warning_appended(self):
        result = render_brief(_make_warning_trace())
        assert "⚠" in result

    def test_no_warning_when_healthy(self):
        result = render_brief(_make_trace())
        assert "⚠" not in result
