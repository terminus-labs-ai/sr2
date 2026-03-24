"""Bridge metrics exporter — Prometheus text exposition format."""

from __future__ import annotations

from bridge.engine import BridgeEngine
from bridge.session_tracker import BridgeSession, SessionTracker


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
            lines.append(
                f'sr2_bridge_session_requests{{session="{sid}"}} {session.request_count}'
            )
        lines.append("")

        # Per-session zone token counts
        lines.append("# HELP sr2_bridge_session_tokens Estimated tokens per session zone")
        lines.append("# TYPE sr2_bridge_session_tokens gauge")
        for sid, session in sessions.items():
            m = self._engine.get_session_metrics(session)
            for zone in ("summarized", "compacted", "raw"):
                lines.append(
                    f'sr2_bridge_session_tokens{{session="{sid}",zone="{zone}"}} '
                    f'{m[f"{zone}_count"]}'
                )

        return "\n".join(lines) + "\n"
