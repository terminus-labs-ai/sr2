"""Persistent session storage.

Sessions are stored as rows in PostgreSQL. Each turn is a JSONB array element.
On every turn added, the session row is updated.
On startup, active sessions are loaded from the database.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Protocol

from sr2_runtime.session.session import Session, SessionConfig

logger = logging.getLogger(__name__)


class SessionStore(Protocol):
    """Protocol for session persistence backends."""

    async def save(self, session: Session) -> None:
        """Save or update a session."""
        ...

    async def load(self, session_id: str) -> Session | None:
        """Load a session by ID. Returns None if not found."""
        ...

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        ...

    async def list_active(self) -> list[str]:
        """List IDs of non-expired sessions."""
        ...

    async def update_turns(self, session_id: str, turns: list[dict]) -> None:
        """Update the turns for an existing session (incremental save)."""
        ...


class InMemorySessionStore:
    """In-memory session store for testing. No persistence."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    async def save(self, session: Session) -> None:
        self._store[session.id] = {
            "id": session.id,
            "config_name": session.config.name,
            "lifecycle": session.config.lifecycle,
            "max_turns": session.config.max_turns,
            "idle_timeout_minutes": session.config.idle_timeout_minutes,
            "turns": session.turns,
            "metadata": session.metadata,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
        }

    async def load(self, session_id: str) -> Session | None:
        data = self._store.get(session_id)
        if not data:
            return None
        config = SessionConfig(
            name=data.get("config_name", session_id),
            lifecycle=data.get("lifecycle", "persistent"),
            max_turns=data.get("max_turns", 200),
            idle_timeout_minutes=data.get("idle_timeout_minutes", 60),
        )
        session = Session(session_id, config)
        session.turns = data.get("turns", [])
        session.metadata = data.get("metadata", {})
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        return session

    async def delete(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    async def list_active(self) -> list[str]:
        return list(self._store.keys())

    async def update_turns(self, session_id: str, turns: list[dict]) -> None:
        if session_id in self._store:
            self._store[session_id]["turns"] = turns
            self._store[session_id]["last_activity"] = datetime.now(UTC).isoformat()


class PostgresSessionStore:
    """PostgreSQL session store.

    Table schema:
        sessions(
            id TEXT PRIMARY KEY,
            config_name TEXT NOT NULL,
            lifecycle TEXT NOT NULL DEFAULT 'persistent',
            turns JSONB NOT NULL DEFAULT '[]',
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    def __init__(self, pool):
        self._pool = pool

    async def create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    config_name TEXT NOT NULL,
                    lifecycle TEXT NOT NULL DEFAULT 'persistent',
                    turns JSONB NOT NULL DEFAULT '[]'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_lifecycle
                    ON sessions(lifecycle) WHERE lifecycle != 'ephemeral';
                CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
                    ON sessions(last_activity);
            """)

    async def save(self, session: Session) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, config_name, lifecycle, turns, metadata, created_at, last_activity)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    turns = $4::jsonb,
                    metadata = $5::jsonb,
                    last_activity = $7
                """,
                session.id,
                session.config.name,
                session.config.lifecycle,
                json.dumps(session.turns),
                json.dumps(session.metadata),
                session.created_at,
                session.last_activity,
            )

    async def load(self, session_id: str) -> Session | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        if not row:
            return None
        config = SessionConfig(
            name=row["config_name"],
            lifecycle=row["lifecycle"],
        )
        session = Session(session_id, config)
        session.turns = json.loads(row["turns"]) if isinstance(row["turns"], str) else row["turns"]
        session.metadata = (
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        )
        session.created_at = row["created_at"]
        session.last_activity = row["last_activity"]
        return session

    async def delete(self, session_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)
        return "DELETE 1" in result

    async def list_active(self) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM sessions WHERE lifecycle != 'ephemeral' ORDER BY last_activity DESC"
            )
        return [r["id"] for r in rows]

    async def update_turns(self, session_id: str, turns: list[dict]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions SET turns = $1::jsonb, last_activity = NOW()
                WHERE id = $2
                """,
                json.dumps(turns),
                session_id,
            )
