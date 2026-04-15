from __future__ import annotations

import time

import pytest

from sr2.pipeline.trace import TraceEvent, TraceCollector, TurnTrace


class TestTraceEvent:
    def test_creation(self):
        ev = TraceEvent(turn=1, stage="resolve", timestamp=1.0, duration_ms=5.0, data={"key": "val"})
        assert ev.turn == 1
        assert ev.stage == "resolve"
        assert ev.timestamp == 1.0
        assert ev.duration_ms == 5.0
        assert ev.data == {"key": "val"}

    def test_frozen_immutability(self):
        ev = TraceEvent(turn=1, stage="input", timestamp=1.0, duration_ms=0.0, data={})
        with pytest.raises(AttributeError):
            ev.turn = 2  # type: ignore[misc]
        with pytest.raises(AttributeError):
            ev.stage = "other"  # type: ignore[misc]


class TestTurnTrace:
    def _make_trace(self) -> TurnTrace:
        return TurnTrace(turn_number=1, session_id="s1", interface_name="http", started_at=100.0)

    def test_add_and_get(self):
        trace = self._make_trace()
        ev = TraceEvent(turn=1, stage="resolve", timestamp=100.1, duration_ms=3.0, data={})
        trace.add(ev)
        assert trace.get("resolve") is ev
        assert trace.get("nonexistent") is None

    def test_get_returns_first_match(self):
        trace = self._make_trace()
        ev1 = TraceEvent(turn=1, stage="resolve", timestamp=100.1, duration_ms=1.0, data={"first": True})
        ev2 = TraceEvent(turn=1, stage="resolve", timestamp=100.2, duration_ms=2.0, data={"first": False})
        trace.add(ev1)
        trace.add(ev2)
        assert trace.get("resolve") is ev1

    def test_total_duration_ms(self):
        trace = self._make_trace()
        trace.add(TraceEvent(turn=1, stage="resolve", timestamp=100.1, duration_ms=3.5, data={}))
        trace.add(TraceEvent(turn=1, stage="retrieve", timestamp=100.2, duration_ms=1.5, data={}))
        assert trace.total_duration_ms == 5.0

    def test_total_duration_ms_empty(self):
        trace = self._make_trace()
        assert trace.total_duration_ms == 0.0

    # --- Warning tests ---

    def test_warning_high_utilization(self):
        trace = self._make_trace()
        trace.add(TraceEvent(turn=1, stage="resolve", timestamp=100.1, duration_ms=1.0, data={"utilization": 0.95}))
        assert any("budget utilization 95.0%" in w for w in trace.warnings)

    def test_warning_utilization_at_boundary(self):
        """Exactly 0.9 should NOT trigger."""
        trace = self._make_trace()
        trace.add(TraceEvent(turn=1, stage="resolve", timestamp=100.1, duration_ms=1.0, data={"utilization": 0.9}))
        assert not any("budget utilization" in w for w in trace.warnings)

    def test_warning_circuit_breaker_open(self):
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="resolve", timestamp=100.1, duration_ms=1.0,
            data={"layers": [{"name": "memory", "circuit_breaker": "open"}]},
        ))
        assert "circuit breaker open: memory" in trace.warnings

    def test_warning_low_cache_efficiency(self):
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="resolve", timestamp=100.1, duration_ms=1.0,
            data={"cache_efficiency": 0.3},
        ))
        assert any("cache efficiency 30.0%" in w for w in trace.warnings)

    def test_warning_no_summarization_many_compacted(self):
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="zones", timestamp=100.1, duration_ms=1.0,
            data={"compacted_turns": 25, "summarized_turns": 0},
        ))
        assert "no summarization despite 25 compacted turns" in trace.warnings

    def test_warning_compacted_with_summarization_ok(self):
        """If summarized_turns > 0, no warning even with many compacted."""
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="zones", timestamp=100.1, duration_ms=1.0,
            data={"compacted_turns": 25, "summarized_turns": 3},
        ))
        assert not any("no summarization" in w for w in trace.warnings)

    def test_warning_empty_retrieval(self):
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="retrieve", timestamp=100.1, duration_ms=1.0,
            data={"results_returned": 0},
        ))
        assert "memory retrieval returned 0 results" in trace.warnings

    def test_warning_degradation_active(self):
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="metrics", timestamp=100.1, duration_ms=1.0,
            data={"degradation_level": 2},
        ))
        assert "degradation level 2 active" in trace.warnings

    def test_no_warnings_healthy(self):
        trace = self._make_trace()
        trace.add(TraceEvent(
            turn=1, stage="resolve", timestamp=100.1, duration_ms=1.0,
            data={"utilization": 0.5, "cache_efficiency": 0.8, "layers": []},
        ))
        trace.add(TraceEvent(
            turn=1, stage="zones", timestamp=100.2, duration_ms=1.0,
            data={"compacted_turns": 5, "summarized_turns": 1},
        ))
        trace.add(TraceEvent(
            turn=1, stage="retrieve", timestamp=100.3, duration_ms=1.0,
            data={"results_returned": 3},
        ))
        trace.add(TraceEvent(
            turn=1, stage="metrics", timestamp=100.4, duration_ms=1.0,
            data={"degradation_level": 0},
        ))
        assert trace.warnings == []

    # --- to_dict ---

    def test_to_dict(self):
        trace = self._make_trace()
        ev = TraceEvent(turn=1, stage="input", timestamp=100.1, duration_ms=0.5, data={"text": "hello"})
        trace.add(ev)
        d = trace.to_dict()
        assert d["turn_number"] == 1
        assert d["session_id"] == "s1"
        assert d["interface_name"] == "http"
        assert d["started_at"] == 100.0
        assert len(d["events"]) == 1
        assert d["events"][0]["stage"] == "input"
        assert d["events"][0]["data"] == {"text": "hello"}


class TestTraceCollector:
    def test_begin_emit_end_lifecycle(self):
        collector = TraceCollector()
        collector.begin_turn(1, "s1", "http")
        collector.emit("input", {"text": "hi"}, duration_ms=0.1)
        collector.emit("resolve", {"utilization": 0.5}, duration_ms=2.0)
        trace = collector.end_turn()

        assert trace is not None
        assert trace.turn_number == 1
        assert len(trace.events) == 2
        assert trace.get("input") is not None
        assert trace.get("resolve") is not None

    def test_ring_buffer_eviction(self):
        collector = TraceCollector(max_turns=3)
        for i in range(4):
            collector.begin_turn(i, "s1", "http")
            collector.emit("input", {}, duration_ms=0.0)
            collector.end_turn()

        traces = collector.traces
        assert len(traces) == 3
        # Oldest (turn 0) should be evicted
        assert traces[0].turn_number == 1
        assert traces[2].turn_number == 3

    def test_listeners_called_on_end_turn(self):
        collector = TraceCollector()
        received: list[TurnTrace] = []
        collector.on_turn_complete(lambda t: received.append(t))

        collector.begin_turn(1, "s1", "http")
        collector.emit("input", {})
        trace = collector.end_turn()

        assert len(received) == 1
        assert received[0] is trace

    def test_multiple_listeners(self):
        collector = TraceCollector()
        calls_a: list[int] = []
        calls_b: list[int] = []
        collector.on_turn_complete(lambda t: calls_a.append(t.turn_number))
        collector.on_turn_complete(lambda t: calls_b.append(t.turn_number))

        collector.begin_turn(5, "s1", "http")
        collector.end_turn()

        assert calls_a == [5]
        assert calls_b == [5]

    def test_emit_noop_when_no_turn(self):
        collector = TraceCollector()
        # Should not raise
        collector.emit("input", {"text": "ignored"}, duration_ms=1.0)
        assert collector.last() is None

    def test_end_turn_returns_none_when_no_turn(self):
        collector = TraceCollector()
        assert collector.end_turn() is None

    def test_last_returns_most_recent(self):
        collector = TraceCollector()
        collector.begin_turn(1, "s1", "http")
        collector.end_turn()
        collector.begin_turn(2, "s1", "http")
        collector.end_turn()

        assert collector.last() is not None
        assert collector.last().turn_number == 2

    def test_last_returns_none_when_empty(self):
        collector = TraceCollector()
        assert collector.last() is None

    def test_traces_returns_all(self):
        collector = TraceCollector()
        for i in range(5):
            collector.begin_turn(i, "s1", "http")
            collector.emit("input", {"n": i})
            collector.end_turn()

        traces = collector.traces
        assert len(traces) == 5
        assert [t.turn_number for t in traces] == [0, 1, 2, 3, 4]

    def test_traces_returns_copy(self):
        collector = TraceCollector()
        collector.begin_turn(1, "s1", "http")
        collector.end_turn()
        traces = collector.traces
        traces.clear()
        assert len(collector.traces) == 1
