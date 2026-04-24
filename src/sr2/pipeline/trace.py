from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class TraceEvent:
    """A single stage event within a pipeline turn."""

    turn: int
    stage: str
    timestamp: float
    duration_ms: float
    data: dict


@dataclass
class TurnTrace:
    """Aggregates all trace events for a single pipeline turn."""

    turn_number: int
    session_id: str
    interface_name: str
    started_at: float
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)

    def get(self, stage: str) -> TraceEvent | None:
        for event in self.events:
            if event.stage == stage:
                return event
        return None

    @property
    def total_duration_ms(self) -> float:
        return sum(e.duration_ms for e in self.events)

    @property
    def warnings(self) -> list[str]:
        result: list[str] = []

        resolve = self.get("resolve")
        if resolve is not None:
            utilization = resolve.data.get("utilization", 0.0)
            if utilization > 0.9:
                pct = round(utilization * 100, 1)
                result.append(f"budget utilization {pct}% — headroom < 10%")

            for layer in resolve.data.get("layers", []):
                if layer.get("circuit_breaker") == "open":
                    result.append(f"circuit breaker open: {layer.get('name', 'unknown')}")

            cache_efficiency = resolve.data.get("cache_efficiency")
            if cache_efficiency is not None and cache_efficiency < 0.5:
                pct = round(cache_efficiency * 100, 1)
                result.append(f"cache efficiency {pct}% — prefix unstable")

        zones = self.get("zones")
        if zones is not None:
            compacted = zones.data.get("compacted_turns", 0)
            summarized = zones.data.get("summarized_turns", 0)
            if compacted > 20 and summarized == 0:
                result.append(f"no summarization despite {compacted} compacted turns")

        retrieve = self.get("retrieve")
        if retrieve is not None:
            if retrieve.data.get("results_returned") == 0:
                result.append("memory retrieval returned 0 results")

        metrics = self.get("metrics")
        if metrics is not None:
            level = metrics.data.get("degradation_level", 0)
            if level > 0:
                result.append(f"degradation level {level} active")

        return result

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "session_id": self.session_id,
            "interface_name": self.interface_name,
            "started_at": self.started_at,
            "events": [
                {
                    "turn": e.turn,
                    "stage": e.stage,
                    "timestamp": e.timestamp,
                    "duration_ms": e.duration_ms,
                    "data": e.data,
                }
                for e in self.events
            ],
        }


class TraceCollector:
    """Ring-buffer collector for pipeline turn traces.

    Supports concurrent sessions: each session_id gets its own active trace,
    so an async post_process for session A won't be clobbered by a new
    begin_turn for session B.

    Backward-compatible: callers that don't pass session_id to emit/end_turn
    fall back to a global ``_current`` slot (runtime mode, single session).
    """

    def __init__(self, max_turns: int = 50) -> None:
        self._traces: deque[TurnTrace] = deque(maxlen=max_turns)
        # Per-session active traces (proxy mode — concurrent sessions)
        self._active: dict[str, TurnTrace] = {}
        # Legacy single-slot (runtime mode — backward compat)
        self._current: TurnTrace | None = None
        self._listeners: list[Callable[[TurnTrace], None]] = []

    def begin_turn(self, turn_number: int, session_id: str, interface_name: str) -> None:
        trace = TurnTrace(
            turn_number=turn_number,
            session_id=session_id,
            interface_name=interface_name,
            started_at=time.monotonic(),
        )
        self._active[session_id] = trace
        # Keep _current in sync for backward compat (runtime mode uses it)
        self._current = trace

    def emit(
        self,
        stage: str,
        data: dict,
        duration_ms: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        target = self._resolve(session_id)
        if target is None:
            return
        event = TraceEvent(
            turn=target.turn_number,
            stage=stage,
            timestamp=time.monotonic(),
            duration_ms=duration_ms,
            data=data,
        )
        target.add(event)

    def end_turn(
        self,
        session_id: str | None = None,
        duration_ms: float = 0.0,
    ) -> TurnTrace | None:
        target = self._resolve(session_id)
        if target is None:
            return None
        # Remove from active tracking
        self._active.pop(target.session_id, None)
        if self._current is target:
            self._current = None
        self._traces.append(target)
        for listener in self._listeners:
            listener(target)
        return target

    def on_turn_complete(self, callback: Callable[[TurnTrace], None]) -> None:
        self._listeners.append(callback)

    @property
    def traces(self) -> list[TurnTrace]:
        return list(self._traces)

    def last(self) -> TurnTrace | None:
        if self._traces:
            return self._traces[-1]
        return None

    def _resolve(self, session_id: str | None) -> TurnTrace | None:
        """Find the active trace for a session, falling back to _current."""
        if session_id is not None:
            return self._active.get(session_id)
        return self._current
