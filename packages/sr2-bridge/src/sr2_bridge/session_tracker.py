"""Session identification and lifecycle tracking for the bridge."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sr2.compaction.engine import ConversationTurn

from sr2_bridge.config import BridgeSessionConfig

logger = logging.getLogger(__name__)


@dataclass
class BridgeSession:
    """Tracked state for a bridge session.

    Unifies session identification (tracker) and conversation state (engine)
    into a single object. Previously split across SessionInfo and SessionState.
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    request_count: int = 0
    last_message_count: int = 0
    last_message_hash: str = ""

    # Conversation state (previously in engine.SessionState)
    turn_counter: int = 0
    turns: list[ConversationTurn] = field(default_factory=list)

    # Per-message MD5 hashes for compaction detection
    message_hashes: list[str] = field(default_factory=list)

    # System-reminder dedup state
    system_reminder_hashes: set[str] = field(default_factory=set)
    system_reminder_content: list[str] = field(default_factory=list)

    def touch(self) -> None:
        self.last_seen = time.time()
        self.request_count += 1


class SessionTracker:
    """Maps incoming requests to bridge sessions.

    Session name is defined in config (like runtime SessionConfig). All requests
    use the configured session unless the client sends an X-SR2-Session-ID header
    to override — enabling session sharing across different clients (Claude Code,
    OpenCode, etc.).

    Session resets are detected by message count decrease (e.g. /clear in Claude
    Code sends shorter history).
    """

    def __init__(self, config: BridgeSessionConfig):
        self._default_name = config.name
        self._idle_timeout = config.idle_timeout_minutes * 60
        self._sessions: dict[str, BridgeSession] = {}

    def identify(
        self,
        body: dict,
        headers: dict[str, str],
        system_prompt: str | None = None,
    ) -> str:
        """Determine the session ID for a request.

        Priority: X-SR2-Session-ID header > config name.
        """
        session_id = headers.get("x-sr2-session-id", "") or self._default_name

        # Get or create session
        if session_id not in self._sessions:
            self._sessions[session_id] = BridgeSession(session_id=session_id)
            logger.info("New session: %s", session_id)

        self._sessions[session_id].touch()
        return session_id

    def get(self, session_id: str) -> BridgeSession | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def destroy(self, session_id: str) -> str | None:
        """Remove a session and return its ID, or None if not found."""
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info("Destroyed session: %s", session_id)
            return session_id
        return None

    def cleanup_idle(self) -> list[str]:
        """Remove sessions that have been idle past the timeout.

        Returns list of removed session IDs.
        """
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if (now - session.last_seen) > self._idle_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Expired idle session: %s", sid)
        return expired

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def all_sessions(self) -> dict[str, BridgeSession]:
        return dict(self._sessions)

    def restore_session(self, session: BridgeSession) -> None:
        """Restore a previously persisted session into the tracker."""
        self._sessions[session.session_id] = session
        logger.info(
            "Restored session: %s (turns=%d, requests=%d)",
            session.session_id,
            session.turn_counter,
            session.request_count,
        )
