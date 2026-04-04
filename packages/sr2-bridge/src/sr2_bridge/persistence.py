"""SQLite-backed session persistence for the SR2 Bridge."""

from __future__ import annotations

import json
import logging

import aiosqlite

from sr2.compaction.engine import ConversationTurn
from sr2.pipeline.conversation import ConversationZones

from sr2_bridge.session_tracker import BridgeSession

logger = logging.getLogger(__name__)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS bridge_sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    last_message_count INTEGER NOT NULL DEFAULT 0,
    last_message_hash TEXT NOT NULL DEFAULT '',
    turn_counter INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bridge_turns (
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT,
    metadata TEXT,
    compacted INTEGER NOT NULL DEFAULT 0,
    zone TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_number),
    FOREIGN KEY (session_id) REFERENCES bridge_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bridge_summaries (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (session_id, position),
    FOREIGN KEY (session_id) REFERENCES bridge_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bridge_session_notes (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (session_id, position),
    FOREIGN KEY (session_id) REFERENCES bridge_sessions(session_id) ON DELETE CASCADE
);
"""


class BridgeSessionStore:
    """Persists bridge session state to SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open database connection and create tables."""
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.executescript(_CREATE_TABLES)
        await self._conn.commit()
        logger.info("Session persistence connected (db=%s)", self._db_path)

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def save_session(self, session: BridgeSession, zones: ConversationZones) -> None:
        """Persist session metadata and zone state."""
        if not self._conn:
            return

        async with self._conn.execute("BEGIN"):
            # Upsert session metadata
            await self._conn.execute(
                """
                INSERT INTO bridge_sessions
                    (session_id, created_at, last_seen, request_count,
                     last_message_count, last_message_hash, turn_counter)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    request_count = excluded.request_count,
                    last_message_count = excluded.last_message_count,
                    last_message_hash = excluded.last_message_hash,
                    turn_counter = excluded.turn_counter
                """,
                (
                    session.session_id,
                    session.created_at,
                    session.last_seen,
                    session.request_count,
                    session.last_message_count,
                    session.last_message_hash,
                    session.turn_counter,
                ),
            )

            # Replace turns (delete + insert is simpler than diffing)
            await self._conn.execute(
                "DELETE FROM bridge_turns WHERE session_id = ?",
                (session.session_id,),
            )
            for turn in zones.compacted:
                await self._insert_turn(session.session_id, turn, "compacted")
            for turn in zones.raw:
                await self._insert_turn(session.session_id, turn, "raw")

            # Replace summaries
            await self._conn.execute(
                "DELETE FROM bridge_summaries WHERE session_id = ?",
                (session.session_id,),
            )
            for i, summary in enumerate(zones.summarized):
                await self._conn.execute(
                    "INSERT INTO bridge_summaries (session_id, position, content) VALUES (?, ?, ?)",
                    (session.session_id, i, summary),
                )

            # Replace session notes
            await self._conn.execute(
                "DELETE FROM bridge_session_notes WHERE session_id = ?",
                (session.session_id,),
            )
            for i, note in enumerate(zones.session_notes):
                await self._conn.execute(
                    "INSERT INTO bridge_session_notes (session_id, position, content) VALUES (?, ?, ?)",
                    (session.session_id, i, note),
                )

        await self._conn.commit()

    async def _insert_turn(self, session_id: str, turn: ConversationTurn, zone: str) -> None:
        """Insert a single turn row."""
        metadata_json = json.dumps(turn.metadata) if turn.metadata else None
        await self._conn.execute(
            """
            INSERT INTO bridge_turns
                (session_id, turn_number, role, content, content_type, metadata, compacted, zone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn.turn_number,
                turn.role,
                turn.content,
                turn.content_type,
                metadata_json,
                1 if turn.compacted else 0,
                zone,
            ),
        )

    async def load_all_sessions(
        self,
    ) -> list[tuple[BridgeSession, ConversationZones]]:
        """Load all persisted sessions with their zone state."""
        if not self._conn:
            return []

        results: list[tuple[BridgeSession, ConversationZones]] = []

        async with self._conn.execute("SELECT * FROM bridge_sessions") as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            session_id = row[0]
            session = BridgeSession(
                session_id=session_id,
                created_at=row[1],
                last_seen=row[2],
                request_count=row[3],
                last_message_count=row[4],
                last_message_hash=row[5],
                turn_counter=row[6],
            )

            # Load turns grouped by zone
            compacted: list[ConversationTurn] = []
            raw: list[ConversationTurn] = []

            async with self._conn.execute(
                "SELECT turn_number, role, content, content_type, metadata, compacted, zone "
                "FROM bridge_turns WHERE session_id = ? ORDER BY turn_number",
                (session_id,),
            ) as cursor:
                turn_rows = await cursor.fetchall()

            for tr in turn_rows:
                metadata = json.loads(tr[4]) if tr[4] else None
                turn = ConversationTurn(
                    turn_number=tr[0],
                    role=tr[1],
                    content=tr[2],
                    content_type=tr[3],
                    metadata=metadata,
                    compacted=bool(tr[5]),
                )
                if tr[6] == "compacted":
                    compacted.append(turn)
                else:
                    raw.append(turn)

            # Load summaries
            summarized: list[str] = []
            async with self._conn.execute(
                "SELECT content FROM bridge_summaries WHERE session_id = ? ORDER BY position",
                (session_id,),
            ) as cursor:
                summary_rows = await cursor.fetchall()
            summarized = [r[0] for r in summary_rows]

            # Load session notes
            session_notes: list[str] = []
            async with self._conn.execute(
                "SELECT content FROM bridge_session_notes WHERE session_id = ? ORDER BY position",
                (session_id,),
            ) as cursor:
                notes_rows = await cursor.fetchall()
            session_notes = [r[0] for r in notes_rows]

            zones = ConversationZones(
                summarized=summarized,
                compacted=compacted,
                raw=raw,
                session_notes=session_notes,
            )
            results.append((session, zones))

        logger.info("Loaded %d persisted sessions", len(results))
        return results

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and its turns/summaries (CASCADE)."""
        if not self._conn:
            return
        await self._conn.execute(
            "DELETE FROM bridge_sessions WHERE session_id = ?",
            (session_id,),
        )
        await self._conn.commit()
