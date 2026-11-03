"""sr2-161 / AC31 — closed episodes leave the recent working set."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from ulid import ULID

from sr2.config.models import ResolverConfig
from sr2.continuity.models import (
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
)
from sr2.continuity.store import ContinuityStore
from sr2.models import Message, TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.provenance import (
    TURN_ARCHIVE_LAYER,
    ContextAuthority,
    ContextLifecycle,
    ContextScope,
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
)
from sr2.pipeline.resolvers.working_set import WorkingSetResolver


SESSION_ID = "episode-exit-session"
BASE_TIME = datetime(2026, 11, 3, tzinfo=timezone.utc)
ORIGIN = EntryOrigin(kind="resolver", name="episode_exit_fixture")


def _message(text: str, role: str = "assistant") -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


def _entry(
    text: str,
    turn_seq: int,
    *,
    lifecycle: ContextLifecycle = ContextLifecycle.ARCHIVE,
) -> Entry:
    return Entry(
        id=str(ULID()),
        content=_message(text),
        sources=(),
        origin=ORIGIN,
        layer=TURN_ARCHIVE_LAYER,
        session_id=SESSION_ID,
        created_at=BASE_TIME + timedelta(seconds=turn_seq),
        lifecycle=lifecycle,
        meta={"turn_seq": turn_seq},
    )


async def _provenance(entries: list[Entry]) -> InMemoryProvenanceStore:
    store = InMemoryProvenanceStore()
    for entry in entries:
        await store.write(entry)
    return store


async def _closed_episode_store(tmp_path, covered: list[Entry]) -> ContinuityStore:
    store = ContinuityStore(tmp_path / "continuity.sqlite3")
    await store.connect()
    await store.append(
        ContinuityRecord(
            id="episode-close-1",
            subject_id="episode-1",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.EPISODE,
            content="Turns 1 through 4 are complete.",
            scope=ContextScope(owner_id="owner", agent_id="agent"),
            authority=ContextAuthority.DERIVED,
            provenance_ids=tuple(entry.id for entry in covered),
            confidence=1.0,
            salience=1.0,
            created_at=BASE_TIME + timedelta(seconds=5),
            meta={
                "episode_state": "closed",
                "episode_id": "episode-1",
                "title": "Closed topic",
                "turn_range": [1, 4],
            },
        )
    )
    return store


class _ResolverDependencies:
    def __init__(
        self,
        provenance_store: InMemoryProvenanceStore,
        continuity_store: ContinuityStore | None,
    ) -> None:
        self._inner = Dependencies(
            session_id=SESSION_ID,
            continuity_store=continuity_store,
        )
        self.provenance_store = provenance_store

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _resolver(
    provenance_store: InMemoryProvenanceStore,
    continuity_store: ContinuityStore | None,
) -> WorkingSetResolver:
    config = ResolverConfig(
        type="working_set",
        config={"recent_token_budget": 10_000},
    )
    return WorkingSetResolver.build(
        config,
        _ResolverDependencies(provenance_store, continuity_store),  # type: ignore[arg-type]
    )


def _event() -> Event:
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="core",
        data=[TextBlock(text="turn 6 current input")],
    )


def _texts(content: list[object]) -> list[str]:
    texts: list[str] = []
    for item in content:
        if isinstance(item, Message):
            texts.extend(
                block.text for block in item.content if isinstance(block, TextBlock)
            )
        elif isinstance(item, TextBlock):
            texts.append(item.text)
    return texts


@pytest.mark.asyncio
async def test_ac31_r1_closed_episode_turns_are_absent_with_room(tmp_path):
    closed_entries = [_entry(f"closed turn {seq}", seq) for seq in range(1, 5)]
    recent_entry = _entry("turn 5 remains recent", 5)
    provenance = await _provenance([*closed_entries, recent_entry])
    continuity = await _closed_episode_store(tmp_path, closed_entries)

    result = await _resolver(provenance, continuity).resolve([_event()])

    texts = _texts(result.content)
    assert all(f"closed turn {seq}" not in texts for seq in range(1, 5))
    assert "turn 5 remains recent" in texts
    await continuity.close()


@pytest.mark.asyncio
async def test_ac31_r2_current_input_survives_closed_episode_exit(tmp_path):
    closed_entries = [_entry(f"closed turn {seq}", seq) for seq in range(1, 5)]
    provenance = await _provenance(closed_entries)
    continuity = await _closed_episode_store(tmp_path, closed_entries)

    result = await _resolver(provenance, continuity).resolve([_event()])

    assert "turn 6 current input" in _texts(result.content)
    await continuity.close()


@pytest.mark.asyncio
async def test_ac31_r3_pinned_plan_from_closed_turn_survives(tmp_path):
    closed_entries = [_entry(f"closed turn {seq}", seq) for seq in range(1, 5)]
    pinned_plan = _entry(
        "pinned plan step formed in turn 3",
        3,
        lifecycle=ContextLifecycle.PINNED,
    )
    provenance = await _provenance([*closed_entries, pinned_plan])
    continuity = await _closed_episode_store(tmp_path, closed_entries)

    result = await _resolver(provenance, continuity).resolve([_event()])

    assert "pinned plan step formed in turn 3" in _texts(result.content)
    await continuity.close()


@pytest.mark.asyncio
async def test_ac31_r4_no_close_record_preserves_pre_amendment_window(tmp_path):
    entries = [_entry(f"archive turn {seq}", seq) for seq in range(1, 6)]
    provenance = await _provenance(entries)
    empty_continuity = ContinuityStore(tmp_path / "empty-continuity.sqlite3")
    await empty_continuity.connect()

    pre_amendment = await _resolver(provenance, None).resolve([_event()])
    no_close_record = await _resolver(provenance, empty_continuity).resolve([_event()])

    assert no_close_record.content == pre_amendment.content
    await empty_continuity.close()
