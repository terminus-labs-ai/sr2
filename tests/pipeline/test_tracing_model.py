"""Tests for sr2.pipeline.tracing — FiringRecord, Tracer, CollectingTracer.

Covers:
  FR1:  FiringRecord is a frozen dataclass with all required fields.
  FR1:  FiringRecord raises on mutation (frozen).
  FR1:  FiringRecord defaults: status="ok", error=None.
  FR1:  FiringRecord accepts status="failed" with an error string.
  FR1:  FiringRecord accepts each valid kind value.
  FR2:  Tracer is a runtime_checkable protocol — isinstance works on duck-typed objects.
  FR3:  CollectingTracer starts empty.
  FR3:  CollectingTracer.on_firing appends records in order.
  FR3:  CollectingTracer.get_trace() returns records in insertion order.
  FR3:  CollectingTracer.clear() empties the buffer.
  FR3:  CollectingTracer satisfies isinstance(ct, Tracer).
  FR3:  get_trace() returns an independent copy — clearing does not affect prior reference.
"""

import dataclasses

import pytest

from sr2.pipeline.tracing import CollectingTracer, FiringRecord, Tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(**overrides) -> FiringRecord:
    """Build a valid FiringRecord with sensible defaults."""
    defaults = dict(
        turn_seq=0,
        firing_seq=0,
        kind="resolver",
        component_name="test_resolver",
        layer="conversation",
        trigger_events=["evt_001"],
        content_before=[],
        content_after=[],
        tokens_before=10,
        tokens_after=12,
        tokens_delta=2,
        duration_ms=5.0,
    )
    defaults.update(overrides)
    return FiringRecord(**defaults)


# ---------------------------------------------------------------------------
# FR1 — FiringRecord frozen dataclass
# ---------------------------------------------------------------------------


def test_firing_record_is_frozen():
    """FiringRecord raises FrozenInstanceError on any field mutation."""
    record = make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.turn_seq = 99  # type: ignore[misc]


def test_firing_record_all_fields_round_trip():
    """FiringRecord constructed with all fields stores and returns them correctly."""
    record = FiringRecord(
        turn_seq=1,
        firing_seq=2,
        kind="transformer",
        component_name="my_transformer",
        layer="summary",
        trigger_events=["e1", "e2"],
        content_before=["block_a"],
        content_after=["block_b"],
        tokens_before=100,
        tokens_after=80,
        tokens_delta=-20,
        duration_ms=12.5,
        status="ok",
        error=None,
    )
    assert record.turn_seq == 1
    assert record.firing_seq == 2
    assert record.kind == "transformer"
    assert record.component_name == "my_transformer"
    assert record.layer == "summary"
    assert record.trigger_events == ["e1", "e2"]
    assert record.content_before == ["block_a"]
    assert record.content_after == ["block_b"]
    assert record.tokens_before == 100
    assert record.tokens_after == 80
    assert record.tokens_delta == -20
    assert record.duration_ms == 12.5
    assert record.status == "ok"
    assert record.error is None


def test_firing_record_defaults_status_and_error():
    """FiringRecord defaults status to 'ok' and error to None when omitted."""
    record = make_record()
    assert record.status == "ok"
    assert record.error is None


def test_firing_record_failed_status_with_error():
    """FiringRecord accepts status='failed' and a non-None error string."""
    record = make_record(status="failed", error="something went wrong")
    assert record.status == "failed"
    assert record.error == "something went wrong"


def test_firing_record_kind_resolver():
    """FiringRecord accepts kind='resolver'."""
    record = make_record(kind="resolver")
    assert record.kind == "resolver"


def test_firing_record_kind_transformer():
    """FiringRecord accepts kind='transformer'."""
    record = make_record(kind="transformer")
    assert record.kind == "transformer"


def test_firing_record_kind_tool_provider():
    """FiringRecord accepts kind='tool_provider'."""
    record = make_record(kind="tool_provider")
    assert record.kind == "tool_provider"


# ---------------------------------------------------------------------------
# FR2 — Tracer protocol runtime_checkable
# ---------------------------------------------------------------------------


def test_tracer_protocol_is_runtime_checkable_positive():
    """isinstance(obj, Tracer) returns True for objects with on_firing and on_compile."""

    class DuckTracer:
        def on_firing(self, record: FiringRecord) -> None:
            pass

        def on_compile(self, request) -> None:
            pass

    obj = DuckTracer()
    assert isinstance(obj, Tracer)


def test_tracer_protocol_is_runtime_checkable_negative():
    """isinstance(obj, Tracer) returns False for objects without on_firing."""

    class NotATracer:
        def some_other_method(self) -> None:
            pass

    obj = NotATracer()
    assert not isinstance(obj, Tracer)


# ---------------------------------------------------------------------------
# FR3 — CollectingTracer
# ---------------------------------------------------------------------------


def test_collecting_tracer_starts_empty():
    """CollectingTracer.get_trace() returns [] before any records are added."""
    ct = CollectingTracer()
    assert ct.get_trace() == []


def test_collecting_tracer_on_firing_appends():
    """CollectingTracer.on_firing adds a record to the buffer."""
    ct = CollectingTracer()
    r = make_record(firing_seq=1)
    ct.on_firing(r)
    assert len(ct.get_trace()) == 1
    assert ct.get_trace()[0] is r


def test_collecting_tracer_insertion_order():
    """CollectingTracer.get_trace() returns records in insertion order."""
    ct = CollectingTracer()
    r0 = make_record(firing_seq=0)
    r1 = make_record(firing_seq=1)
    r2 = make_record(firing_seq=2)
    ct.on_firing(r0)
    ct.on_firing(r1)
    ct.on_firing(r2)
    trace = ct.get_trace()
    assert trace[0] is r0
    assert trace[1] is r1
    assert trace[2] is r2


def test_collecting_tracer_clear_empties_buffer():
    """CollectingTracer.clear() causes get_trace() to return [] afterwards."""
    ct = CollectingTracer()
    ct.on_firing(make_record())
    ct.on_firing(make_record())
    ct.clear()
    assert ct.get_trace() == []


def test_collecting_tracer_satisfies_tracer_protocol():
    """isinstance(CollectingTracer(), Tracer) is True."""
    ct = CollectingTracer()
    assert isinstance(ct, Tracer)


def test_collecting_tracer_get_trace_returns_independent_copy():
    """get_trace() returns an independent copy — clearing the tracer does not affect a previously returned list."""
    ct = CollectingTracer()
    r = make_record(firing_seq=7)
    ct.on_firing(r)
    snapshot = ct.get_trace()
    ct.clear()
    # snapshot must still contain the record
    assert len(snapshot) == 1
    assert snapshot[0] is r
    # and the tracer is now empty
    assert ct.get_trace() == []
