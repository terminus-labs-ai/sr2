"""sr2-92 / AC1 — provenance store cage for the causal-metadata extension
(unit layer).

Rows:
  AC1-r9  — SQLite round-trip of every causal field (reason, authority,
            lifecycle, full ContextScope, meta, sources, created_at) across
            close/reopen, lineage preserved.
  AC1-r10 — writes whose sources reference IDs absent from the store are
            rejected (InMemoryProvenanceStore AND SQLiteProvenanceStore);
            a valid-order batch succeeds as control.
  AC1-r11 — a second write for a stored id with a different payload leaves
            the stored entry unchanged (append-only immutability; both
            stores).
  AC1-r12 — the SQLite store must also hold entries for raw-content
            compiled items (the no-store bypass removed; store side).

Grounding: the spec symbols do not exist yet; they are referenced through
the module object (``prov.ContextReason`` etc.) so collection succeeds and
each test fails individually with a clean AttributeError until C
implements. Entry constructor calls pass only today's fields (new fields
must carry spec defaults), so the tests pin store-visible behavior rather
than constructor mechanics.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

import sr2.pipeline.provenance as prov
from sr2.models import Message, TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Spec-vocabulary accessors (module attribute access; red-run hygiene)
# ---------------------------------------------------------------------------


def ContextReason():
    return prov.ContextReason


def ContextAuthority():
    return prov.ContextAuthority


def ContextLifecycle():
    return prov.ContextLifecycle


def ContextScope(**kwargs):
    return prov.ContextScope(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_id() -> str:
    global _COUNTER
    _COUNTER += 1
    return f"SP{_COUNTER:024d}"


def _make_scope():
    """Full-spectrum ContextScope exercising every subfield."""
    return ContextScope(
        owner_id="diego",
        agent_id="spectre",
        projects=("sr2", "spectre"),
        frame_id="frame-001",
        topics=("provenance", "context"),
        sensitivity=("personal",),
    )


def _causal_fields():
    """One complete set of causal-field values for construction."""
    return dict(
        reason=ContextReason().LIVE_SOURCE,
        authority=ContextAuthority().LIVE_SOURCE,
        lifecycle=ContextLifecycle().PROJECT,
        scope=_make_scope(),
    )


def _new_entry(
    *,
    entry_id: str,
    content,
    sources: tuple = (),
    origin_kind: str = "resolver",
    origin_name: str = "test_resolver",
    layer: str = "conversation",
    session_id: str = "session-spine",
    created_at: datetime | None = None,
    meta: dict | None = None,
):
    """Construct a valid Entry using only the fields that exist TODAY.

    Causal fields must arrive with spec-declared defaults; every causal
    assertion then uses dataclasses.replace(), which raises a clean
    AttributeError naming the missing field on the red run.
    """
    return prov.Entry(
        id=entry_id,
        content=content,
        sources=sources,
        origin=prov.EntryOrigin(kind=origin_kind, name=origin_name),
        layer=layer,
        session_id=session_id,
        created_at=(
            created_at
            if created_at is not None
            else datetime.now(tz=timezone.utc).replace(microsecond=0)
        ),
        meta=meta if meta is not None else {},
    )


def _with_causal(entry):
    """Return entry with a full causal-field set applied."""
    fields = _causal_fields()
    for name in fields:
        if not hasattr(entry, name):
            raise AttributeError(
                f"Entry lacks causal field {name!r} (sr2-92 target interface)"
            )
    return dataclasses.replace(entry, **fields)


async def _dangling_ok(store) -> bool:
    """True when the store rejects an entry whose sources are absent.

    A store may reject at write() time or refuse to persist the row; either
    behavior counts as rejection. Returning False means the dangling-source
    entry was accepted and stored — the behavior sr2-92 removes.
    """
    dangling = "D" * 26
    assert await store.get(dangling) is None
    entry = _new_entry(
        entry_id=_next_id(),
        content=TextBlock(text="references a nonexistent source"),
        sources=(dangling,),
        origin_kind="transformer",
        origin_name="fixture",
    )
    try:
        await store.write(entry)
    except Exception:
        return True
    return await store.get(entry.id) is None


# ---------------------------------------------------------------------------
# AC1-r9 — SQLite round-trip of the full causal payload
# ---------------------------------------------------------------------------


class TestAC1R9SQLiteCausalRoundTrip:
    async def test_full_causal_entry_survives_close_and_reopen(self, tmp_path):
        """AC1-r9: Entry with reason, authority, lifecycle, full ContextScope,
        meta and sources written to SQLiteProvenanceStore; close; reopen ->
        get() returns an Entry with every field equal, lineage preserved."""
        db_path = tmp_path / "provenance-r9.db"

        src_a = _new_entry(
            entry_id=_next_id(),
            content=TextBlock(text="source A: user statement"),
        )
        src_b = _new_entry(
            entry_id=_next_id(),
            content=Message(role="user", content=[TextBlock(text="source B")]),
            layer="dialogue",
        )
        original = _with_causal(
            _new_entry(
                entry_id=_next_id(),
                content=TextBlock(text="derived summary content"),
                sources=(src_a.id, src_b.id),
                origin_kind="transformer",
                origin_name="fixture_summarizer",
                meta={
                    "kind": "summary",
                    "revision": "abc123",
                    "nested": {"count": 2, "tags": ["x", "y"]},
                },
            )
        )

        store = SQLiteProvenanceStore(path=db_path)
        await store.connect()
        try:
            # Valid order: sources first, then the entry referencing them.
            await store.write_batch([src_a, src_b, original])
            mid = await store.get(original.id)
            assert mid is not None
            assert mid == original, "field equality broken before reopen"
        finally:
            await store.close()

        reopened = SQLiteProvenanceStore(path=db_path)
        await reopened.connect()
        try:
            got = await reopened.get(original.id)
            assert got is not None, "entry missing after reopen"
            assert got == original, (
                "SQLite round-trip changed at least one Entry field"
            )

            # Field-by-field, so a failure names the broken field.
            assert got.reason == original.reason
            assert got.authority == original.authority
            assert got.lifecycle == original.lifecycle
            assert isinstance(got.scope, prov.ContextScope)
            assert got.scope.owner_id == original.scope.owner_id
            assert got.scope.agent_id == original.scope.agent_id
            assert tuple(got.scope.projects) == tuple(original.scope.projects)
            assert got.scope.frame_id == original.scope.frame_id
            assert tuple(got.scope.topics) == tuple(original.scope.topics)
            assert tuple(got.scope.sensitivity) == tuple(original.scope.sensitivity)
            assert got.meta == original.meta
            assert set(got.sources) == set(original.sources)
            assert got.created_at == original.created_at
            assert got.content == original.content
            assert got.origin == original.origin
            assert got.layer == original.layer
            assert got.session_id == original.session_id

            # Lineage preserved through reopen: reaches both source entries.
            lineage = await reopened.get_lineage(original.id)
            lineage_ids = {e.id for e in lineage}
            assert original.id in lineage_ids
            assert src_a.id in lineage_ids
            assert src_b.id in lineage_ids
        finally:
            await reopened.close()


# ---------------------------------------------------------------------------
# AC1-r10 — dangling-source writes are rejected; valid order succeeds
# ---------------------------------------------------------------------------


class TestAC1R10DanglingSourcesRejected:
    async def test_in_memory_write_with_dangling_source_rejected(self):
        """AC1-r10: InMemoryProvenanceStore rejects a write whose sources
        reference IDs absent from the store."""
        store = prov.InMemoryProvenanceStore()
        assert await _dangling_ok(store), (
            "InMemoryProvenanceStore accepted an entry with a dangling source"
        )

    async def test_in_memory_valid_order_batch_succeeds(self):
        """AC1-r10 control: the same graph written in valid order succeeds."""
        store = prov.InMemoryProvenanceStore()
        source = _new_entry(
            entry_id=_next_id(), content=TextBlock(text="real source")
        )
        derived = _new_entry(
            entry_id=_next_id(),
            content=TextBlock(text="derived content"),
            sources=(source.id,),
            origin_kind="transformer",
            origin_name="fixture",
        )
        await store.write_batch([source, derived])

        assert await store.get(source.id) is not None
        got = await store.get(derived.id)
        assert got is not None
        assert source.id in got.sources

    async def test_sqlite_write_with_dangling_source_rejected(self, tmp_path):
        """AC1-r10: SQLiteProvenanceStore rejects a write whose sources
        reference IDs absent from the store, persisting nothing dangling.

        Rejection must not be an FK accident whose residue survives: after
        the rejected write, NEITHER the entry row NOR any entry_sources row
        naming it may exist. The connection-level queries below check both
        tables directly, so a store that silently skipped the entry_sources
        row while persisting the entry (or left half-written rows behind on
        a rollback path) fails this test even though the FK constraint
        raised IntegrityError.
        """
        db_path = tmp_path / "provenance-r10.db"
        store = SQLiteProvenanceStore(path=db_path)
        await store.connect()
        try:
            assert await _dangling_ok(store), (
                "SQLiteProvenanceStore accepted an entry with a dangling source"
            )
        finally:
            await store.close()

        # Nothing dangling persisted, in either table. Fresh connection: the
        # rows, if any, are durable file state, not connection state.
        checker = SQLiteProvenanceStore(path=db_path)
        await checker.connect()
        try:
            assert checker._db is not None  # fixture-internal read-back
            # Rows written by this test file always carry the SP id prefix;
            # the file's stores are dedicated to this tmp_path DB, so an
            # empty table means this test persisted nothing.
            async with checker._db.execute(
                "SELECT id FROM entries WHERE id LIKE 'SP%'"
            ) as cursor:
                entry_rows = await cursor.fetchall()
            assert entry_rows == [], (
                "dangling-source write left entry rows behind: "
                f"{[r[0] for r in entry_rows]!r}"
            )
            async with checker._db.execute(
                "SELECT entry_id, source_id FROM entry_sources "
                "WHERE source_id NOT IN (SELECT id FROM entries) "
                "OR entry_id NOT IN (SELECT id FROM entries)"
            ) as cursor:
                dangling_rows = await cursor.fetchall()
            assert dangling_rows == [], (
                "entry_sources rows outlive the FK rejection: "
                f"{dangling_rows!r}"
            )
        finally:
            await checker.close()

    async def test_sqlite_valid_order_batch_succeeds(self, tmp_path):
        """AC1-r10 control: SQLite accepts the same graph written in valid
        order (sources before the referencing entry)."""
        store = SQLiteProvenanceStore(path=tmp_path / "provenance-r10b.db")
        await store.connect()
        try:
            source = _new_entry(
                entry_id=_next_id(), content=TextBlock(text="real source")
            )
            derived = _new_entry(
                entry_id=_next_id(),
                content=TextBlock(text="derived content"),
                sources=(source.id,),
                origin_kind="transformer",
                origin_name="fixture",
            )
            await store.write_batch([source, derived])

            assert await store.get(source.id) is not None
            got = await store.get(derived.id)
            assert got is not None
            assert source.id in got.sources
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC1-r11 — immutability: second write for a stored id cannot change it
# ---------------------------------------------------------------------------


class TestAC1R11StoredEntryImmutable:
    async def test_in_memory_second_write_with_different_payload_is_noop(self):
        """AC1-r11: rewriting a stored id with a different payload leaves the
        stored entry unchanged (InMemoryProvenanceStore).

        The replacement changes only meta, so the payload remains a valid
        Entry under any store-side source-lineage validation: this test
        isolates immutability, not lineage."""
        store = prov.InMemoryProvenanceStore()
        original = _with_causal(
            _new_entry(
                entry_id=_next_id(),
                content=TextBlock(text="original"),
                meta={"version": 1},
            )
        )
        await store.write(original)

        replacement = dataclasses.replace(
            original, meta={"version": 2, "tampered": True}
        )
        await store.write(replacement)

        got = await store.get(original.id)
        assert got is not None
        assert got == original, "stored entry changed after a second write"
        assert got.meta == {"version": 1}

    async def test_sqlite_second_write_with_different_payload_is_noop(self, tmp_path):
        """AC1-r11: rewriting a stored id with a different payload leaves the
        stored entry unchanged (SQLiteProvenanceStore).

        Meta-only payload change (valid under lineage validation) so this
        test isolates immutability."""
        store = SQLiteProvenanceStore(path=tmp_path / "provenance-r11.db")
        await store.connect()
        try:
            original = _with_causal(
                _new_entry(
                    entry_id=_next_id(),
                    content=TextBlock(text="original"),
                    meta={"version": 1},
                )
            )
            await store.write(original)

            replacement = dataclasses.replace(
                original, meta={"version": 2, "tampered": True}
            )
            try:
                await store.write(replacement)
            except Exception:
                # A rejecting store is acceptable; a silent overwrite is not.
                pass

            got = await store.get(original.id)
            assert got is not None
            assert got == original, "stored entry changed after a second write"
            assert got.meta == {"version": 1}
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC1-r12 (store side) — raw-content compiled items reach the SQLite store
# ---------------------------------------------------------------------------


class TestAC1R12RawItemsReachSQLiteStore:
    async def test_raw_content_layer_flushes_stored_entries_to_sqlite(self, tmp_path):
        """AC1-r12: a resolver returning raw ResolvedContent (no entries,
        continuity-disabled shape) has its compiled items written through to
        the SQLiteProvenanceStore after process_pending.

        Uses a capture decorator at the store boundary so the written Entry
        (with its id and session_id) can then be fetched from SQLite
        itself — proving persistence, not just buffering.
        """
        from sr2.pipeline.layer import Layer

        class CapturingStore:
            """Transparent pass-through that records every write."""

            def __init__(self, inner):
                self._inner = inner
                self.written: list = []

            async def write(self, entry):
                self.written.append(entry)
                await self._inner.write(entry)

            async def write_batch(self, entries):
                self.written.extend(entries)
                await self._inner.write_batch(entries)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def matches(item, content) -> bool:
            if item == content:
                return True
            try:
                return item.model_dump() == content.model_dump()
            except (AttributeError, TypeError, ValueError):
                return False

        inner = SQLiteProvenanceStore(path=tmp_path / "provenance-r12.db")
        await inner.connect()
        store = CapturingStore(inner)
        try:
            layer = Layer(
                name="conversation",
                target=CompilationTarget.MESSAGES,
                position=AppendStrategy(),
                token_budget=None,
                resolvers=[],
                transformers=[],
                token_counter=CharacterTokenCounter(),
                event_bus=EventBus(),
                provenance_store=store,
            )

            block = TextBlock(text="raw content must reach SQLite")
            msg = Message(role="user", content=[TextBlock(text="and so must I")])
            layer.add_content(
                ResolvedContent(
                    resolver_name="r",
                    source_layer="test",
                    content=[block, msg],
                )
            )

            # Compiled content unchanged (current behavior): both raw items
            # remain in the layer and compile into the request.
            assert block in layer.get_content()
            assert msg in layer.get_content()
            compiled = layer.compile()
            assert any(
                block in (m.content if isinstance(m.content, list) else [])
                for m in compiled
            )

            await layer.process_pending()

            # Both raw items were written...
            written_block = [e for e in store.written if matches(e.content, block)]
            written_msg = [e for e in store.written if matches(e.content, msg)]
            assert len(written_block) == 1, (
                "raw TextBlock never reached the provenance store write path"
            )
            assert len(written_msg) == 1, (
                "raw Message never reached the provenance store write path"
            )

            # ...and are actually persisted in SQLite.
            for entry in (written_block[0], written_msg[0]):
                got = await inner.get(entry.id)
                assert got is not None, (
                    f"entry {entry.id} was written but is not in the SQLite store"
                )
                assert matches(got.content, entry.content)
        finally:
            await inner.close()
