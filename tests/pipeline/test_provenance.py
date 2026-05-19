"""Tests for sr2.pipeline.provenance — Entry, EntryOrigin, ProvenanceStore, InMemoryProvenanceStore.

Covers:
  FR1:  Entry is a frozen dataclass with id, content, sources, origin, layer,
        session_id, created_at, meta fields.
  FR2:  Entry.__post_init__ enforces transformer-origin entries must have sources;
        resolver-origin entries must have no sources.
  FR3:  EntryOrigin is a frozen dataclass with kind and name.
  FR8:  ProvenanceStore protocol — InMemoryProvenanceStore satisfies isinstance check.
  FR10: InMemoryProvenanceStore.write() stores an entry retrievable via get().
  FR10: InMemoryProvenanceStore.write_batch() stores all entries.
  FR10: InMemoryProvenanceStore.get() returns None for unknown id.
  FR10: InMemoryProvenanceStore.get_lineage() traverses ancestor graph with
        optional depth limit.
  FR10: InMemoryProvenanceStore.get_session() returns all entries for a session,
        optionally filtered by layer or since datetime.
"""

import dataclasses
from datetime import datetime, timezone

import pytest

from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import (
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
    ProvenanceStore,
)


# ---------------------------------------------------------------------------
# Helper — build valid Entry objects with sensible defaults
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_id() -> str:
    """Return a 26-char synthetic ULID-shaped ID for test use."""
    global _COUNTER
    _COUNTER += 1
    return f"{_COUNTER:026d}"


def make_entry(
    *,
    id: str | None = None,
    sources: tuple[str, ...] = (),
    origin_kind: str = "resolver",
    origin_name: str = "test_resolver",
    layer: str = "conversation",
    session_id: str = "session-abc",
    created_at: datetime | None = None,
    meta: dict | None = None,
) -> Entry:
    """Build a valid Entry with sensible defaults.

    By default produces a resolver-origin genesis entry (sources=()).
    Pass origin_kind="transformer" and non-empty sources for transformer entries.
    """
    return Entry(
        id=id if id is not None else _next_id(),
        content=TextBlock(text="test content"),
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name=origin_name),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at if created_at is not None else datetime.now(tz=timezone.utc),
        meta=meta if meta is not None else {},
    )


def make_message_entry(
    *,
    id: str | None = None,
    sources: tuple[str, ...] = (),
    origin_kind: str = "resolver",
    session_id: str = "session-abc",
    layer: str = "conversation",
    created_at: datetime | None = None,
) -> Entry:
    """Build a valid Entry backed by a Message content object."""
    return Entry(
        id=id if id is not None else _next_id(),
        content=Message(role="user", content=[TextBlock(text="hello")]),
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name="test_resolver"),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at if created_at is not None else datetime.now(tz=timezone.utc),
        meta={},
    )


# ---------------------------------------------------------------------------
# 1. EntryOrigin dataclass
# ---------------------------------------------------------------------------


class TestEntryOrigin:
    def test_create_resolver_origin(self):
        origin = EntryOrigin(kind="resolver", name="system_prompt_resolver")
        assert origin.kind == "resolver"
        assert origin.name == "system_prompt_resolver"

    def test_create_transformer_origin(self):
        origin = EntryOrigin(kind="transformer", name="summarization")
        assert origin.kind == "transformer"
        assert origin.name == "summarization"

    def test_is_frozen_kind(self):
        """EntryOrigin is frozen — assigning kind raises an error."""
        origin = EntryOrigin(kind="resolver", name="r")
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            origin.kind = "transformer"  # type: ignore[misc]

    def test_is_frozen_name(self):
        """EntryOrigin is frozen — assigning name raises an error."""
        origin = EntryOrigin(kind="resolver", name="r")
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            origin.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Entry dataclass — fields and immutability
# ---------------------------------------------------------------------------


class TestEntryFields:
    def test_resolver_entry_fields_present(self):
        """All declared fields are accessible on a resolver entry."""
        now = datetime.now(tz=timezone.utc)
        entry = Entry(
            id="00000000000000000000000001",
            content=TextBlock(text="hello"),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="my_resolver"),
            layer="system",
            session_id="sess-1",
            created_at=now,
            meta={"key": "value"},
        )
        assert entry.id == "00000000000000000000000001"
        assert entry.sources == ()
        assert entry.origin.kind == "resolver"
        assert entry.origin.name == "my_resolver"
        assert entry.layer == "system"
        assert entry.session_id == "sess-1"
        assert entry.created_at is now
        assert entry.meta == {"key": "value"}

    def test_meta_defaults_to_empty_dict(self):
        """meta field defaults to empty dict when not provided."""
        entry = Entry(
            id=_next_id(),
            content=TextBlock(text="x"),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="r"),
            layer="core",
            session_id="s",
            created_at=datetime.now(tz=timezone.utc),
        )
        assert entry.meta == {}

    def test_is_frozen_id(self):
        """Entry is frozen — assigning id raises an error."""
        entry = make_entry()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            entry.id = "newid"  # type: ignore[misc]

    def test_is_frozen_layer(self):
        """Entry is frozen — assigning layer raises an error."""
        entry = make_entry()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            entry.layer = "other"  # type: ignore[misc]

    def test_is_frozen_sources(self):
        """Entry is frozen — assigning sources raises an error."""
        entry = make_entry()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            entry.sources = ("x",)  # type: ignore[misc]

    def test_id_is_26_chars(self):
        """ULID-shaped ID is a 26-character string."""
        entry = make_entry(id="A" * 26)
        assert isinstance(entry.id, str)
        assert len(entry.id) == 26

    def test_sources_is_tuple(self):
        """sources field is a tuple of strings."""
        entry = make_entry(sources=())
        assert isinstance(entry.sources, tuple)

    def test_content_can_be_message(self):
        """content field accepts a Message object."""
        entry = make_message_entry()
        assert isinstance(entry.content, Message)


# ---------------------------------------------------------------------------
# 3. Entry.__post_init__ invariants
# ---------------------------------------------------------------------------


class TestEntryPostInitInvariants:
    def test_transformer_origin_with_empty_sources_raises_value_error(self):
        """Transformer-origin entry with sources=() must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)transformer"):
            Entry(
                id=_next_id(),
                content=TextBlock(text="x"),
                sources=(),
                origin=EntryOrigin(kind="transformer", name="summarization"),
                layer="conversation",
                session_id="s",
                created_at=datetime.now(tz=timezone.utc),
            )

    def test_resolver_origin_with_non_empty_sources_raises_value_error(self):
        """Resolver-origin entry with non-empty sources must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)resolver"):
            Entry(
                id=_next_id(),
                content=TextBlock(text="x"),
                sources=(_next_id(),),
                origin=EntryOrigin(kind="resolver", name="my_resolver"),
                layer="conversation",
                session_id="s",
                created_at=datetime.now(tz=timezone.utc),
            )

    def test_valid_transformer_origin_entry_no_error(self):
        """Transformer-origin entry with non-empty sources is valid."""
        src_id = _next_id()
        entry = Entry(
            id=_next_id(),
            content=TextBlock(text="summary"),
            sources=(src_id,),
            origin=EntryOrigin(kind="transformer", name="summarization"),
            layer="conversation",
            session_id="s",
            created_at=datetime.now(tz=timezone.utc),
        )
        assert entry.sources == (src_id,)

    def test_valid_resolver_origin_entry_no_error(self):
        """Resolver-origin entry with empty sources is valid."""
        entry = Entry(
            id=_next_id(),
            content=TextBlock(text="from resolver"),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="my_resolver"),
            layer="system",
            session_id="s",
            created_at=datetime.now(tz=timezone.utc),
        )
        assert entry.sources == ()

    def test_transformer_origin_with_multiple_sources_is_valid(self):
        """Transformer-origin entry may have multiple source IDs."""
        src1 = _next_id()
        src2 = _next_id()
        entry = Entry(
            id=_next_id(),
            content=TextBlock(text="compacted"),
            sources=(src1, src2),
            origin=EntryOrigin(kind="transformer", name="compaction"),
            layer="conversation",
            session_id="s",
            created_at=datetime.now(tz=timezone.utc),
        )
        assert len(entry.sources) == 2


# ---------------------------------------------------------------------------
# 4. ProvenanceStore protocol — isinstance check
# ---------------------------------------------------------------------------


class TestProvenanceStoreProtocol:
    def test_in_memory_store_satisfies_protocol(self):
        """InMemoryProvenanceStore satisfies the ProvenanceStore runtime-checkable protocol."""
        store = InMemoryProvenanceStore()
        assert isinstance(store, ProvenanceStore)


# ---------------------------------------------------------------------------
# 5. InMemoryProvenanceStore — write and get
# ---------------------------------------------------------------------------


class TestInMemoryProvenanceStoreWriteGet:
    @pytest.mark.asyncio
    async def test_write_then_get_returns_same_entry(self):
        """write() stores entry; get() retrieves it by id."""
        store = InMemoryProvenanceStore()
        entry = make_entry(id=_next_id())
        await store.write(entry)
        retrieved = await store.get(entry.id)
        assert retrieved is entry

    @pytest.mark.asyncio
    async def test_get_unknown_id_returns_none(self):
        """get() returns None when entry_id is not in the store."""
        store = InMemoryProvenanceStore()
        result = await store.get("nonexistent-id-000000000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_write_batch_stores_all_entries(self):
        """write_batch() stores all provided entries."""
        store = InMemoryProvenanceStore()
        ids = [_next_id() for _ in range(3)]
        entries = [make_entry(id=eid) for eid in ids]
        await store.write_batch(entries)
        for eid, entry in zip(ids, entries):
            retrieved = await store.get(eid)
            assert retrieved is entry

    @pytest.mark.asyncio
    async def test_write_batch_empty_list_is_noop(self):
        """write_batch() with an empty list does not raise."""
        store = InMemoryProvenanceStore()
        await store.write_batch([])

    @pytest.mark.asyncio
    async def test_second_write_overwrites_first(self):
        """Writing an entry with the same id twice keeps the latest write."""
        store = InMemoryProvenanceStore()
        eid = _next_id()
        first = make_entry(id=eid, layer="system")
        second = make_entry(id=eid, layer="conversation")
        await store.write(first)
        await store.write(second)
        retrieved = await store.get(eid)
        assert retrieved.layer == "conversation"


# ---------------------------------------------------------------------------
# 6. InMemoryProvenanceStore — get_lineage
# ---------------------------------------------------------------------------


class TestInMemoryProvenanceStoreGetLineage:
    @pytest.mark.asyncio
    async def test_genesis_entry_lineage_returns_only_itself(self):
        """get_lineage on an entry with no sources returns just that entry."""
        store = InMemoryProvenanceStore()
        genesis = make_entry(id=_next_id(), sources=())
        await store.write(genesis)
        lineage = await store.get_lineage(genesis.id)
        assert len(lineage) == 1
        assert lineage[0].id == genesis.id

    @pytest.mark.asyncio
    async def test_one_level_lineage_returns_entry_and_sources(self):
        """Lineage of a transformer entry includes the entry and all its sources."""
        store = InMemoryProvenanceStore()
        src1 = make_entry(id=_next_id(), sources=())
        src2 = make_entry(id=_next_id(), sources=())
        summary = make_entry(
            id=_next_id(),
            sources=(src1.id, src2.id),
            origin_kind="transformer",
            origin_name="summarization",
        )
        await store.write_batch([src1, src2, summary])

        lineage = await store.get_lineage(summary.id)
        lineage_ids = {e.id for e in lineage}
        assert summary.id in lineage_ids
        assert src1.id in lineage_ids
        assert src2.id in lineage_ids
        assert len(lineage) == 3

    @pytest.mark.asyncio
    async def test_lineage_depth_limit_stops_at_immediate_parents(self):
        """depth=1 returns only the entry and its direct sources, not grandparents."""
        store = InMemoryProvenanceStore()
        grandparent = make_entry(id=_next_id(), sources=())
        parent = make_entry(
            id=_next_id(),
            sources=(grandparent.id,),
            origin_kind="transformer",
            origin_name="step1",
        )
        child = make_entry(
            id=_next_id(),
            sources=(parent.id,),
            origin_kind="transformer",
            origin_name="step2",
        )
        await store.write_batch([grandparent, parent, child])

        lineage = await store.get_lineage(child.id, depth=1)
        lineage_ids = {e.id for e in lineage}
        assert child.id in lineage_ids
        assert parent.id in lineage_ids
        assert grandparent.id not in lineage_ids

    @pytest.mark.asyncio
    async def test_lineage_depth_minus_one_traverses_full_graph(self):
        """depth=-1 (default) traverses the full ancestor graph."""
        store = InMemoryProvenanceStore()
        grandparent = make_entry(id=_next_id(), sources=())
        parent = make_entry(
            id=_next_id(),
            sources=(grandparent.id,),
            origin_kind="transformer",
            origin_name="step1",
        )
        child = make_entry(
            id=_next_id(),
            sources=(parent.id,),
            origin_kind="transformer",
            origin_name="step2",
        )
        await store.write_batch([grandparent, parent, child])

        lineage = await store.get_lineage(child.id, depth=-1)
        lineage_ids = {e.id for e in lineage}
        assert child.id in lineage_ids
        assert parent.id in lineage_ids
        assert grandparent.id in lineage_ids
        assert len(lineage) == 3

    @pytest.mark.asyncio
    async def test_lineage_of_unknown_id_returns_empty(self):
        """get_lineage for an id not in the store returns an empty list."""
        store = InMemoryProvenanceStore()
        result = await store.get_lineage("00000000000000000000000000")
        assert result == []


# ---------------------------------------------------------------------------
# 7. InMemoryProvenanceStore — get_session
# ---------------------------------------------------------------------------


class TestInMemoryProvenanceStoreGetSession:
    @pytest.mark.asyncio
    async def test_get_session_returns_all_entries_for_session(self):
        """get_session returns every entry with the given session_id."""
        store = InMemoryProvenanceStore()
        sess = "session-xyz"
        entries = [make_entry(id=_next_id(), session_id=sess) for _ in range(3)]
        other = make_entry(id=_next_id(), session_id="other-session")
        await store.write_batch(entries + [other])

        result = await store.get_session(sess)
        result_ids = {e.id for e in result}
        for e in entries:
            assert e.id in result_ids
        assert other.id not in result_ids

    @pytest.mark.asyncio
    async def test_get_session_filtered_by_layer(self):
        """get_session with layer= returns only entries for that layer."""
        store = InMemoryProvenanceStore()
        sess = "session-layer-test"
        system_entry = make_entry(id=_next_id(), session_id=sess, layer="system")
        convo_entry1 = make_entry(id=_next_id(), session_id=sess, layer="conversation")
        convo_entry2 = make_entry(id=_next_id(), session_id=sess, layer="conversation")
        await store.write_batch([system_entry, convo_entry1, convo_entry2])

        result = await store.get_session(sess, layer="conversation")
        result_ids = {e.id for e in result}
        assert convo_entry1.id in result_ids
        assert convo_entry2.id in result_ids
        assert system_entry.id not in result_ids

    @pytest.mark.asyncio
    async def test_get_session_filtered_by_since(self):
        """get_session with since= returns only entries created at or after that time."""
        store = InMemoryProvenanceStore()
        sess = "session-since-test"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        old = make_entry(id=_next_id(), session_id=sess, created_at=t0)
        at_boundary = make_entry(id=_next_id(), session_id=sess, created_at=t1)
        after = make_entry(id=_next_id(), session_id=sess, created_at=t2)
        await store.write_batch([old, at_boundary, after])

        result = await store.get_session(sess, since=t1)
        result_ids = {e.id for e in result}
        assert at_boundary.id in result_ids
        assert after.id in result_ids
        assert old.id not in result_ids

    @pytest.mark.asyncio
    async def test_get_session_unknown_session_returns_empty_list(self):
        """get_session for an unknown session_id returns an empty list."""
        store = InMemoryProvenanceStore()
        result = await store.get_session("no-such-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_session_layer_and_since_combined(self):
        """get_session applies both layer and since filters when both are given."""
        store = InMemoryProvenanceStore()
        sess = "session-combined"
        t0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        old_system = make_entry(id=_next_id(), session_id=sess, layer="system", created_at=t0)
        old_convo = make_entry(id=_next_id(), session_id=sess, layer="conversation", created_at=t0)
        new_system = make_entry(id=_next_id(), session_id=sess, layer="system", created_at=t2)
        new_convo = make_entry(id=_next_id(), session_id=sess, layer="conversation", created_at=t2)
        await store.write_batch([old_system, old_convo, new_system, new_convo])

        result = await store.get_session(sess, layer="conversation", since=t1)
        result_ids = {e.id for e in result}
        assert new_convo.id in result_ids
        assert old_convo.id not in result_ids
        assert new_system.id not in result_ids
        assert old_system.id not in result_ids
