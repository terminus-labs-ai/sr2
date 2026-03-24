"""Session identification and lifecycle tracking for the bridge."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from bridge.config import BridgeSessionConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Tracked state for a bridge session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    request_count: int = 0
    last_message_count: int = 0

    def touch(self) -> None:
        self.last_seen = time.time()
        self.request_count += 1


class SessionTracker:
    """Maps incoming requests to SR2 sessions.

    Strategies:
    - system_hash: Hash the system prompt text. Claude Code's system prompt is
      session-specific (contains CLAUDE.md, project context), so this naturally
      groups requests from the same session.
    - header: Use the X-SR2-Session-ID request header.
    - single: All requests map to a single session.
    """

    def __init__(self, config: BridgeSessionConfig):
        self._strategy = config.strategy
        self._idle_timeout = config.idle_timeout_minutes * 60
        self._sessions: dict[str, SessionInfo] = {}

    def identify(
        self,
        body: dict,
        headers: dict[str, str],
        system_prompt: str | None = None,
    ) -> str:
        """Determine the session ID for a request."""
        if self._strategy == "header":
            session_id = headers.get("x-sr2-session-id", "default")
        elif self._strategy == "single":
            session_id = "default"
        else:
            # system_hash (default)
            text = system_prompt or ""
            if text:
                session_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            else:
                session_id = "no-system"

        # Get or create session info
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionInfo(session_id=session_id)
            logger.info("New session: %s (strategy=%s)", session_id, self._strategy)

        self._sessions[session_id].touch()
        return session_id

    def get(self, session_id: str) -> SessionInfo | None:
        """Get session info by ID."""
        return self._sessions.get(session_id)

    def cleanup_idle(self) -> list[str]:
        """Remove sessions that have been idle past the timeout.

        Returns list of removed session IDs.
        """
        now = time.time()
        expired = [
            sid
            for sid, info in self._sessions.items()
            if (now - info.last_seen) > self._idle_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Expired idle session: %s", sid)
        return expired

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def all_sessions(self) -> dict[str, SessionInfo]:
        return dict(self._sessions)
