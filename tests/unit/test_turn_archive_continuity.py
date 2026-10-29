"""sr2-157: archive continuity through the public SR2 API (AC1-c1–c4)."""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from conftest import MockLLM, make_minimal_config, make_user_input
from sr2.models import TextBlock
from sr2.orchestrator import SR2
from sr2.pipeline.provenance import (
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
    read_turn_archive,
)
from sr2.pipeline.token_counting import CharacterTokenCounter


def make_sr2(store):
    return SR2(
        pipeline_config=make_minimal_config(),
        llm=MockLLM(),
        token_counter=CharacterTokenCounter(),
        session_id="s",
        provenance_store=store,
    )


async def run_turn(sr2):
    events = [event async for event in sr2.turn(make_user_input("hello"))]
    await sr2.aclose()
    assert events[-1].type == "end"
    assert not any(event.type == "error" for event in events)


async def test_existing_archive_continues_highest_turn_and_last_sequence():
    """AC1-c1: ordering uses turn/seq, and only this session's archive."""
    store = InMemoryProvenanceStore()
    # Insertion order, timestamps and ids must not select the last entry.
    for entry_id, turn, seq, session, layer in [
        ("last", 2, 7, "s", "turn_archive"),
        ("earlier-in-turn", 2, 3, "s", "turn_archive"),
        ("older-turn", 1, 99, "s", "turn_archive"),
        ("first-turn", 0, 0, "s", "turn_archive"),
        ("other-session", 20, 100, "other", "turn_archive"),
        ("other-layer", 30, 100, "s", "session"),
    ]:
        await store.write(Entry(
            id=entry_id,
            content=TextBlock(text="archived"),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="turn_archivist"),
            layer=layer,
            session_id=session,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            meta={"turn_seq": turn, "seq": seq},
        ))
    store.get_session = AsyncMock(wraps=store.get_session)
    sr2 = make_sr2(store)

    await sr2.continue_archive()
    await run_turn(sr2)
    await run_turn(sr2)

    store.get_session.assert_awaited_once_with("s", layer="turn_archive")
    entries = await read_turn_archive(store, "s", 3)
    assert len(entries) == 2
    assert [entry.meta["turn_seq"] for entry in entries] == [3, 3]
    assert entries[0].sources == ("last",)
    assert entries[0].origin == EntryOrigin("transformer", "turn_archivist")
    following = await read_turn_archive(store, "s", 4)
    assert len(following) == 2
    assert following[0].sources == (entries[-1].id,)


async def test_empty_archive_keeps_turn_zero_and_genesis():
    """AC1-c2: an empty session retains the original behavior."""
    store = InMemoryProvenanceStore()
    sr2 = make_sr2(store)

    await sr2.continue_archive()
    await run_turn(sr2)

    entries = await read_turn_archive(store, "s", 0)
    assert len(entries) == 2
    assert [entry.meta["turn_seq"] for entry in entries] == [0, 0]
    assert entries[0].sources == ()
    assert entries[0].origin == EntryOrigin("resolver", "turn_archivist")


async def test_successive_instances_continue_without_duplicate_order_keys():
    """AC1-c3: rebuilding a facade within one process also resumes."""
    store = InMemoryProvenanceStore()
    first = make_sr2(store)
    await first.continue_archive()
    await run_turn(first)
    original = await read_turn_archive(store, "s", 0)
    assert len(original) == 2

    second = make_sr2(store)
    await second.continue_archive()
    await run_turn(second)

    resumed = await read_turn_archive(store, "s", 1)
    assert len(resumed) == 2
    assert [entry.meta["turn_seq"] for entry in resumed] == [1, 1]
    assert resumed[0].sources == (original[-1].id,)
    assert resumed[0].origin == EntryOrigin("transformer", "turn_archivist")
    entries = await store.get_session("s", layer="turn_archive")
    assert len(entries) == 4
    assert len({(entry.meta["turn_seq"], entry.meta["seq"]) for entry in entries}) == 4


async def test_archive_read_failure_logs_session_and_keeps_genesis(caplog):
    """AC1-c4: failed restoration does not prevent a fresh turn."""
    store = InMemoryProvenanceStore()
    get_session = store.get_session
    store.get_session = AsyncMock(side_effect=OSError("archive unavailable"))
    sr2 = make_sr2(store)

    with caplog.at_level(logging.ERROR, logger="sr2.orchestrator"):
        await sr2.continue_archive()
    store.get_session.assert_awaited_once_with("s", layer="turn_archive")
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "session=s" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], OSError)
    await run_turn(sr2)

    entries = await get_session("s", layer="turn_archive")
    assert len(entries) == 2
    assert [entry.meta["turn_seq"] for entry in entries] == [0, 0]
    assert entries[0].sources == ()
    assert entries[0].origin == EntryOrigin("resolver", "turn_archivist")
