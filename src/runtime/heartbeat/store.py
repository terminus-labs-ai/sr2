"""Heartbeat persistence — protocol, in-memory, and PostgreSQL implementations."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Protocol

from runtime.heartbeat.model import Heartbeat, HeartbeatStatus

logger = logging.getLogger(__name__)


class HeartbeatStore(Protocol):
    """Protocol for heartbeat persistence backends."""

    async def upsert(self, heartbeat: Heartbeat) -> None:
        """Insert or update a heartbeat. If key conflicts, update existing."""
        ...

    async def get(self, heartbeat_id: str) -> Heartbeat | None:
        """Get a heartbeat by ID."""
        ...

    async def get_by_key(self, key: str) -> Heartbeat | None:
        """Get a heartbeat by its unique key."""
        ...

    async def list_due(self, now: datetime) -> list[Heartbeat]:
        """List pending heartbeats whose fire_at <= now."""
        ...

    async def update_status(self, heartbeat_id: str, status: HeartbeatStatus) -> None:
        """Update the status of a heartbeat."""
        ...

    async def cancel_by_key(self, key: str) -> bool:
        """Cancel a heartbeat by key. Returns True if found and cancelled."""
        ...

    async def list_pending(self, agent_name: str) -> list[Heartbeat]:
        """List all pending heartbeats for an agent."""
        ...


class InMemoryHeartbeatStore:
    """In-memory heartbeat store for testing."""

    def __init__(self) -> None:
        self._heartbeats: dict[str, Heartbeat] = {}
        self._key_index: dict[str, str] = {}  # key -> heartbeat_id

    async def upsert(self, heartbeat: Heartbeat) -> None:
        if heartbeat.key and heartbeat.key in self._key_index:
            # Update existing heartbeat with same key
            existing_id = self._key_index[heartbeat.key]
            existing = self._heartbeats[existing_id]
            existing.prompt = heartbeat.prompt
            existing.fire_at = heartbeat.fire_at
            existing.context_turns = heartbeat.context_turns
            existing.status = HeartbeatStatus.pending
            existing.updated_at = datetime.now(UTC)
            existing.source_session = heartbeat.source_session
            return

        self._heartbeats[heartbeat.id] = heartbeat
        if heartbeat.key:
            self._key_index[heartbeat.key] = heartbeat.id

    async def get(self, heartbeat_id: str) -> Heartbeat | None:
        return self._heartbeats.get(heartbeat_id)

    async def get_by_key(self, key: str) -> Heartbeat | None:
        hb_id = self._key_index.get(key)
        if hb_id:
            return self._heartbeats.get(hb_id)
        return None

    async def list_due(self, now: datetime) -> list[Heartbeat]:
        due = [
            hb
            for hb in self._heartbeats.values()
            if hb.fire_at <= now and hb.status == HeartbeatStatus.pending
        ]
        due.sort(key=lambda h: h.fire_at)
        return due[:50]

    async def update_status(self, heartbeat_id: str, status: HeartbeatStatus) -> None:
        hb = self._heartbeats.get(heartbeat_id)
        if hb:
            hb.status = status
            hb.updated_at = datetime.now(UTC)

    async def cancel_by_key(self, key: str) -> bool:
        hb_id = self._key_index.get(key)
        if not hb_id:
            return False
        hb = self._heartbeats.get(hb_id)
        if not hb or hb.status != HeartbeatStatus.pending:
            return False
        hb.status = HeartbeatStatus.cancelled
        hb.updated_at = datetime.now(UTC)
        return True

    async def list_pending(self, agent_name: str) -> list[Heartbeat]:
        return [
            hb
            for hb in self._heartbeats.values()
            if hb.agent_name == agent_name and hb.status == HeartbeatStatus.pending
        ]


class PostgresHeartbeatStore:
    """PostgreSQL heartbeat store.

    Uses the same asyncpg pool as session storage.
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    async def create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS heartbeats (
                    id TEXT PRIMARY KEY,
                    key TEXT UNIQUE,
                    agent_name TEXT NOT NULL,
                    source_session TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    context_turns JSONB DEFAULT '[]'::jsonb,
                    status TEXT NOT NULL DEFAULT 'pending',
                    fire_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    recurring BOOLEAN DEFAULT FALSE,
                    interval_seconds INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_heartbeats_fire_at
                    ON heartbeats(fire_at) WHERE status = 'pending';
                CREATE INDEX IF NOT EXISTS idx_heartbeats_agent
                    ON heartbeats(agent_name) WHERE status = 'pending';
            """)

    async def upsert(self, heartbeat: Heartbeat) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO heartbeats (
                    id, key, agent_name, source_session, prompt,
                    context_turns, status, fire_at, created_at, updated_at,
                    recurring, interval_seconds
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (key) WHERE key IS NOT NULL DO UPDATE SET
                    prompt = EXCLUDED.prompt,
                    fire_at = EXCLUDED.fire_at,
                    context_turns = EXCLUDED.context_turns,
                    status = 'pending',
                    updated_at = NOW(),
                    source_session = EXCLUDED.source_session
                """,
                heartbeat.id,
                heartbeat.key,
                heartbeat.agent_name,
                heartbeat.source_session,
                heartbeat.prompt,
                json.dumps(heartbeat.context_turns),
                heartbeat.status.value,
                heartbeat.fire_at,
                heartbeat.created_at,
                heartbeat.updated_at,
                heartbeat.recurring,
                heartbeat.interval_seconds,
            )

    async def get(self, heartbeat_id: str) -> Heartbeat | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM heartbeats WHERE id = $1", heartbeat_id)
        return self._row_to_heartbeat(row) if row else None

    async def get_by_key(self, key: str) -> Heartbeat | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM heartbeats WHERE key = $1", key)
        return self._row_to_heartbeat(row) if row else None

    async def list_due(self, now: datetime) -> list[Heartbeat]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM heartbeats
                WHERE fire_at <= $1 AND status = 'pending'
                ORDER BY fire_at
                LIMIT 50
                """,
                now,
            )
        return [self._row_to_heartbeat(r) for r in rows]

    async def update_status(self, heartbeat_id: str, status: HeartbeatStatus) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE heartbeats SET status = $1, updated_at = NOW()
                WHERE id = $2
                """,
                status.value,
                heartbeat_id,
            )

    async def cancel_by_key(self, key: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE heartbeats SET status = 'cancelled', updated_at = NOW()
                WHERE key = $1 AND status = 'pending'
                """,
                key,
            )
        return "UPDATE 1" in result

    async def list_pending(self, agent_name: str) -> list[Heartbeat]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM heartbeats
                WHERE agent_name = $1 AND status = 'pending'
                ORDER BY fire_at
                """,
                agent_name,
            )
        return [self._row_to_heartbeat(r) for r in rows]

    @staticmethod
    def _row_to_heartbeat(row) -> Heartbeat:
        context_turns = row["context_turns"]
        if isinstance(context_turns, str):
            context_turns = json.loads(context_turns)
        return Heartbeat(
            id=row["id"],
            key=row["key"],
            agent_name=row["agent_name"],
            source_session=row["source_session"],
            prompt=row["prompt"],
            fire_at=row["fire_at"],
            context_turns=context_turns,
            status=HeartbeatStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            recurring=row["recurring"],
            interval_seconds=row["interval_seconds"],
        )
