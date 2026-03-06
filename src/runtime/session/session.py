"""Session state management with lifecycle policies."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from runtime.session.store import SessionStore


@dataclass
class SessionConfig:
    """Config for a named session."""

    name: str
    lifecycle: Literal["persistent", "ephemeral", "rolling"] = "persistent"
    max_turns: int = 200
    idle_timeout_minutes: int = 60


class Session:
    """A conversation session with a lifecycle."""

    def __init__(self, session_id: str, config: SessionConfig | None = None):
        self.id = session_id
        self.config = config or SessionConfig(name=session_id)
        self.turns: list[dict] = []
        self.metadata: dict = {}
        self.created_at: datetime = datetime.now(UTC)
        self.last_activity: datetime = datetime.now(UTC)

    def add_user_message(self, content: str) -> None:
        self.turns.append({"role": "user", "content": content})
        self.last_activity = datetime.now(UTC)
        self._enforce_rolling()

    def add_assistant_message(self, content: str) -> None:
        self.turns.append({"role": "assistant", "content": content})
        self.last_activity = datetime.now(UTC)
        self._enforce_rolling()

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        call_id: str = "",
    ) -> None:
        """Add a tool call + result pair (two turns)."""
        self.turns.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
                "content_type": "tool_call",
                "metadata": {"tool_name": tool_name, "args_summary": str(arguments)[:100]},
            }
        )
        self.turns.append(
            {
                "role": "tool_result",
                "content": result,
                "content_type": "tool_output",
                "tool_call_id": call_id,
                "metadata": {"tool_name": tool_name},
            }
        )

    def add_tool_calls_grouped(
        self,
        calls: list[tuple[str, dict, str, str]],
    ) -> None:
        """Add multiple tool calls from a single LLM response.

        Each call is (tool_name, arguments, result, call_id).
        Creates one assistant message with all tool_calls, followed by
        individual tool_result messages — matching the OpenAI message format.
        """
        if not calls:
            return

        tool_calls_payload = []
        for tool_name, arguments, _result, call_id in calls:
            tool_calls_payload.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )

        tool_names = [c[0] for c in calls]
        self.turns.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_payload,
                "content_type": "tool_call",
                "metadata": {"tool_names": tool_names},
            }
        )

        for tool_name, _arguments, result, call_id in calls:
            self.turns.append(
                {
                    "role": "tool_result",
                    "content": result,
                    "content_type": "tool_output",
                    "tool_call_id": call_id,
                    "metadata": {"tool_name": tool_name},
                }
            )
        self.last_activity = datetime.now(UTC)
        self._enforce_rolling()

    def inject_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Inject a message from another session or system process.

        Used by post_to_session tool and cross-session context sharing.
        """
        entry = {"role": role, "content": content, "injected": True}
        if metadata:
            entry["metadata"] = metadata
        self.turns.append(entry)
        self.last_activity = datetime.now(UTC)

    def to_history(self) -> list[dict]:
        return self.turns

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def user_message_count(self) -> int:
        return sum(1 for t in self.turns if t.get("role") == "user")

    def get_last_user_message(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.get("role") == "user":
                return turn.get("content")
        return None

    def _enforce_rolling(self) -> None:
        """For rolling lifecycle: drop oldest turns when over max."""
        if self.config.lifecycle == "rolling" and len(self.turns) > self.config.max_turns:
            excess = len(self.turns) - self.config.max_turns
            self.turns = self.turns[excess:]


class SessionManager:
    """Manages named sessions with lifecycle enforcement."""

    def __init__(
        self,
        session_configs: dict[str, SessionConfig] | None = None,
        default_config: SessionConfig | None = None,
        store: SessionStore | None = None,
        # Legacy kwargs for backward compat with existing tests
        max_turns: int = 200,
        idle_timeout_minutes: int = 60,
    ):
        self._configs = session_configs or {}
        self._default = default_config or SessionConfig(
            name="_default",
            max_turns=max_turns,
            idle_timeout_minutes=idle_timeout_minutes,
        )
        self._sessions: dict[str, Session] = {}
        self._store = store

    async def get_or_create(self, session_name: str) -> Session:
        """Get from cache, load from DB, or create new."""
        # Check in-memory cache first
        if session_name in self._sessions:
            return self._sessions[session_name]

        # Try loading from persistent store
        if self._store:
            session = await self._store.load(session_name)
            if session:
                self._sessions[session_name] = session
                return session

        # Create new
        config = self._configs.get(
            session_name,
            SessionConfig(
                name=session_name,
                max_turns=self._default.max_turns,
                idle_timeout_minutes=self._default.idle_timeout_minutes,
            ),
        )
        session = Session(session_name, config)
        self._sessions[session_name] = session

        # Persist new session
        if self._store and config.lifecycle != "ephemeral":
            await self._store.save(session)

        return session

    def get(self, session_name: str) -> Session | None:
        return self._sessions.get(session_name)

    async def close(self, session_id: str) -> Session | None:
        """Alias for destroy (backward compat)."""
        return await self.destroy(session_id)

    async def destroy(self, session_name: str) -> Session | None:
        """Destroy a session from cache AND store."""
        session = self._sessions.pop(session_name, None)
        if self._store:
            await self._store.delete(session_name)
        return session

    def create_ephemeral(self, base_name: str) -> Session:
        """Create a uniquely-named ephemeral session.

        Returns a session named "{base_name}_{uuid}" that will be
        destroyed after processing.
        """
        name = f"{base_name}_{uuid.uuid4().hex[:8]}"
        base_cfg = self._configs.get(base_name, self._default)
        config = SessionConfig(
            name=name,
            lifecycle="ephemeral",
            max_turns=base_cfg.max_turns,
            idle_timeout_minutes=base_cfg.idle_timeout_minutes,
        )
        session = Session(name, config)
        self._sessions[name] = session
        return session

    async def save_session(self, session_name: str) -> None:
        """Explicitly save a session to the persistent store."""
        session = self._sessions.get(session_name)
        if session and self._store and session.config.lifecycle != "ephemeral":
            await self._store.save(session)

    async def load_active_sessions(self) -> int:
        """Load all active sessions from store into cache. Called on startup.
        Returns count of sessions loaded."""
        if not self._store:
            return 0
        active_ids = await self._store.list_active()
        loaded = 0
        for sid in active_ids:
            if sid not in self._sessions:
                session = await self._store.load(sid)
                if session:
                    self._sessions[sid] = session
                    loaded += 1
        return loaded

    @property
    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def cleanup_idle(self) -> int:
        """Close persistent sessions that have been idle past timeout. Returns count closed."""
        now = datetime.now(UTC)
        to_close = []
        for sid, session in self._sessions.items():
            if session.config.lifecycle == "ephemeral":
                continue
            idle_min = (now - session.last_activity).total_seconds() / 60
            if idle_min > session.config.idle_timeout_minutes:
                to_close.append(sid)
        for sid in to_close:
            self._sessions.pop(sid)
        return len(to_close)
