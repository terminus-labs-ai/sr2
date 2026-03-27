"""Bridge metrics exporter — Prometheus text exposition format."""

from __future__ import annotations

from bridge.engine import BridgeEngine
from bridge.session_tracker import SessionTracker


class BridgeMetricsExporter:
    """Exports bridge metrics in Prometheus text exposition format."""

    def __init__(self, engine: BridgeEngine, session_tracker: SessionTracker):
        self._engine = engine
        self._session_tracker = session_tracker

    def export(self) -> str:
        """Export current bridge metrics as Prometheus text format."""
        sessions = self._session_tracker.all_sessions()
        lines: list[str] = []

        # Active sessions gauge
        lines.append("# HELP sr2_bridge_active_sessions Number of active bridge sessions")
        lines.append("# TYPE sr2_bridge_active_sessions gauge")
        lines.append(f"sr2_bridge_active_sessions {len(sessions)}")
        lines.append("")

        # Per-session request counter
        lines.append("# HELP sr2_bridge_session_requests Total requests per session")
        lines.append("# TYPE sr2_bridge_session_requests counter")
        for sid, session in sessions.items():
            lines.append(f'sr2_bridge_session_requests{{session="{sid}"}} {session.request_count}')
        lines.append("")

        # Per-session zone token counts
        lines.append("# HELP sr2_bridge_session_tokens Estimated tokens per session zone")
        lines.append("# TYPE sr2_bridge_session_tokens gauge")
        for sid, session in sessions.items():
            m = self._engine.get_session_metrics(session)
            for zone in ("summarized", "compacted", "raw"):
                lines.append(
                    f'sr2_bridge_session_tokens{{session="{sid}",zone="{zone}"}} '
                    f"{m[f'{zone}_count']}"
                )
        lines.append("")

        # Post-processing error counter
        lines.append("# HELP sr2_bridge_postprocess_errors_total Total post-processing errors")
        lines.append("# TYPE sr2_bridge_postprocess_errors_total counter")
        lines.append(f"sr2_bridge_postprocess_errors_total {self._engine.postprocess_error_count}")
        lines.append("")

        # Circuit breaker state per feature
        lines.append(
            "# HELP sr2_bridge_circuit_breaker_state Circuit breaker state (0=closed, 1=open)"
        )
        lines.append("# TYPE sr2_bridge_circuit_breaker_state gauge")
        for feature in ("summarization", "memory_extraction", "memory_retrieval"):
            state = 1 if self._engine.is_breaker_open(feature) else 0
            lines.append(f'sr2_bridge_circuit_breaker_state{{feature="{feature}"}} {state}')
        lines.append("")

        # Per-session before/after token counts and compaction ratio
        lines.append(
            "# HELP sr2_bridge_request_tokens_before "
            "Estimated tokens before optimization (last request)"
        )
        lines.append("# TYPE sr2_bridge_request_tokens_before gauge")
        lines.append(
            "# HELP sr2_bridge_request_tokens_after "
            "Estimated tokens after optimization (last request)"
        )
        lines.append("# TYPE sr2_bridge_request_tokens_after gauge")
        lines.append(
            "# HELP sr2_bridge_compaction_ratio "
            "Ratio of tokens after/before optimization (last request)"
        )
        lines.append("# TYPE sr2_bridge_compaction_ratio gauge")
        for sid, (before, after) in self._engine.last_request_tokens.items():
            lines.append(f'sr2_bridge_request_tokens_before{{session="{sid}"}} {before}')
            lines.append(f'sr2_bridge_request_tokens_after{{session="{sid}"}} {after}')
            ratio = after / before if before > 0 else 1.0
            lines.append(f'sr2_bridge_compaction_ratio{{session="{sid}"}} {ratio:.4f}')
        lines.append("")

        # Summarization latency
        lines.append(
            "# HELP sr2_bridge_summarization_duration_seconds Duration of last summarization call"
        )
        lines.append("# TYPE sr2_bridge_summarization_duration_seconds gauge")
        duration = self._engine.last_summarization_duration
        if duration is not None:
            lines.append(f"sr2_bridge_summarization_duration_seconds {duration:.4f}")

        return "\n".join(lines) + "\n"
