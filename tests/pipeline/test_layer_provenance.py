"""Tests for provenance-aware pipeline models and Layer migration.

Covers:
  FR4:  ResolvedContent gains `entries` field alongside `content` (backward compat).
  FR5:  TransformationResult gains `entries` field alongside `content` (backward compat).
  FR6:  Layer._content becomes list[Entry] internally; Layer.get_content() still returns
        list[ContentBlock | Message] (backward compat).
  FR7:  Layer.add_content persists entries to ProvenanceStore via process_pending.
  FR12: Layer.compile output is unchanged after entry migration.
  FR13: PipelineEngine accepts optional provenance_store parameter.

Backward-compat guarantee:
  All existing tests in test_layer.py and test_engine.py call:
    - ResolvedContent(content=[TextBlock(...)]) — using content= kwarg
    - layer.add_content(resolved)             — synchronously
    - layer.get_content()                      — returns list[ContentBlock | Message]
  None of these behaviors may regress. Every test in this file was written
  with that invariant in mind.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sr2.models import Message, TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import (
    CompilationTarget,
    ResolvedContent,
    TransformationResult,
)
from sr2.pipeline.provenance import (
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
)
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Helper — local copy of make_entry (never import from test files)
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_id() -> str:
    """Return a 26-char synthetic ULID-shaped ID for test use."""
    global _COUNTER
    _COUNTER += 1
    return f"PROV{_COUNTER:022d}"


def make_entry(
    *,
    id: str | None = None,
    content=None,
    sources: tuple[str, ...] = (),
    origin_kind: str = "resolver",
    origin_name: str = "test_resolver",
    layer: str = "conversation",
    session_id: str = "session-test",
    created_at: datetime | None = None,
    meta: dict | None = None,
) -> Entry:
    """Build a valid resolver-origin Entry with sensible defaults."""
    return Entry(
        id=id if id is not None else _next_id(),
        content=content if content is not None else TextBlock(text="test content"),
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name=origin_name),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at if created_at is not None else datetime.now(tz=timezone.utc),
        meta=meta if meta is not None else {},
    )


# ---------------------------------------------------------------------------
# Helper — build a Layer with sane defaults
# ---------------------------------------------------------------------------


def make_layer(
    name: str = "conversation",
    target: CompilationTarget = CompilationTarget.MESSAGES,
    provenance_store=None,
) -> "Layer":  # noqa: F821 — Layer imported below
    from sr2.pipeline.layer import Layer

    kwargs = dict(
        name=name,
        target=target,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
    )
    if provenance_store is not None:
        kwargs["provenance_store"] = provenance_store
    return Layer(**kwargs)


# ---------------------------------------------------------------------------
# Stub resolver that returns entries (new path)
# ---------------------------------------------------------------------------


class EntryResolver:
    """Resolver that returns ResolvedContent with entries= populated."""

    def __init__(
        self,
        name: str = "entry_resolver",
        entries: list[Entry] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._entries = entries or []
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self.max_executions = max_executions
        self.execution_count = 0

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            entries=self._entries,
        )


# ---------------------------------------------------------------------------
# 1. Model field tests — ResolvedContent
# ---------------------------------------------------------------------------


class TestResolvedContentFields:
    def test_entries_kwarg_accepted(self):
        """FR4: ResolvedContent accepts entries= kwarg without raising."""
        entry = make_entry()
        resolved = ResolvedContent(
            resolver_name="r",
            source_layer="s",
            entries=[entry],
        )
        assert resolved.entries == [entry]

    def test_entries_defaults_to_empty_list(self):
        """entries field defaults to [] when not provided."""
        resolved = ResolvedContent(
            resolver_name="r",
            source_layer="s",
            content=[TextBlock(text="hi")],
        )
        assert resolved.entries == []

    def test_content_kwarg_still_accepted(self):
        """FR4 backward compat: content= kwarg still works — does not raise."""
        block = TextBlock(text="hello")
        resolved = ResolvedContent(
            resolver_name="r",
            source_layer="s",
            content=[block],
        )
        assert resolved.content == [block]

    def test_content_defaults_to_empty_list(self):
        """content field defaults to [] when not provided (new-path callers omit it)."""
        entry = make_entry()
        resolved = ResolvedContent(
            resolver_name="r",
            source_layer="s",
            entries=[entry],
        )
        assert resolved.content == []

    def test_both_fields_coexist(self):
        """entries and content can both be set simultaneously."""
        entry = make_entry()
        block = TextBlock(text="hi")
        resolved = ResolvedContent(
            resolver_name="r",
            source_layer="s",
            content=[block],
            entries=[entry],
        )
        assert resolved.content == [block]
        assert resolved.entries == [entry]


# ---------------------------------------------------------------------------
# 2. Model field tests — TransformationResult
# ---------------------------------------------------------------------------


class TestTransformationResultFields:
    def test_entries_kwarg_accepted(self):
        """FR5: TransformationResult accepts entries= kwarg without raising."""
        entry = make_entry(
            sources=("src-1" + "0" * 21,),
            origin_kind="transformer",
            origin_name="my_transformer",
        )
        result = TransformationResult(
            transformer_name="t",
            source_layer="s",
            entries=[entry],
        )
        assert result.entries == [entry]

    def test_entries_defaults_to_empty_list(self):
        """entries field defaults to [] when not provided."""
        result = TransformationResult(
            transformer_name="t",
            source_layer="s",
            content=[TextBlock(text="x")],
        )
        assert result.entries == []

    def test_content_kwarg_still_accepted(self):
        """FR5 backward compat: content= kwarg still works — does not raise."""
        block = TextBlock(text="transformed")
        result = TransformationResult(
            transformer_name="t",
            source_layer="s",
            content=[block],
        )
        assert result.content == [block]

    def test_content_defaults_to_none(self):
        """content field defaults to None when not provided alongside entries."""
        src_id = _next_id()
        entry = make_entry(
            sources=(src_id,),
            origin_kind="transformer",
            origin_name="t",
        )
        result = TransformationResult(
            transformer_name="t",
            source_layer="s",
            entries=[entry],
        )
        assert result.content is None


# ---------------------------------------------------------------------------
# 3. Layer construction with provenance_store
# ---------------------------------------------------------------------------


class TestLayerConstructionProvenance:
    def test_layer_without_provenance_store_constructs_fine(self):
        """FR12: Layer without explicit provenance_store constructs without error."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer is not None

    def test_layer_without_provenance_store_has_default_store(self):
        """Layer without explicit store has an InMemoryProvenanceStore as default."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert isinstance(layer._provenance_store, InMemoryProvenanceStore)

    def test_layer_with_explicit_provenance_store_uses_it(self):
        """FR12: Layer passed an explicit provenance_store uses that store."""
        from sr2.pipeline.layer import Layer

        store = InMemoryProvenanceStore()
        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
            provenance_store=store,
        )
        assert layer._provenance_store is store


# ---------------------------------------------------------------------------
# 4. Layer.add_content with entries= (new path)
# ---------------------------------------------------------------------------


class TestLayerAddContentWithEntries:
    def test_get_content_returns_entry_content_blocks(self):
        """FR6: add_content(entries=[e1, e2]) → get_content() returns [e1.content, e2.content]."""
        layer = make_layer()
        entry1 = make_entry(content=TextBlock(text="first"))
        entry2 = make_entry(content=TextBlock(text="second"))

        resolved = ResolvedContent(
            resolver_name="r",
            source_layer="conversation",
            entries=[entry1, entry2],
        )
        layer.add_content(resolved)

        content = layer.get_content()
        assert len(content) == 2
        assert content[0] == entry1.content
        assert content[1] == entry2.content

    def test_get_content_returns_content_objects_not_entries(self):
        """FR6: get_content() returns ContentBlock/Message objects, not Entry wrappers."""
        layer = make_layer()
        entry = make_entry(content=TextBlock(text="payload"))

        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                entries=[entry],
            )
        )

        content = layer.get_content()
        assert len(content) == 1
        # The item in content must be the TextBlock, NOT the Entry
        assert isinstance(content[0], TextBlock)
        assert not isinstance(content[0], Entry)

    def test_entries_path_buffers_for_store_write(self):
        """After add_content with entries, _pending_writes contains those entries."""
        from sr2.pipeline.layer import Layer

        store = InMemoryProvenanceStore()
        layer = Layer(
            name="test",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
            provenance_store=store,
        )

        entry1 = make_entry()
        entry2 = make_entry()
        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                entries=[entry1, entry2],
            )
        )

        # _pending_writes should contain the two entries awaiting flush
        assert hasattr(layer, "_pending_writes")
        pending_ids = {e.id for e in layer._pending_writes}
        assert entry1.id in pending_ids
        assert entry2.id in pending_ids

    def test_add_content_is_sync_no_await_needed(self):
        """FR6: add_content remains synchronous — callers don't need to await it."""
        import inspect

        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        entry = make_entry()
        result = layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                entries=[entry],
            )
        )
        # add_content must return None (not a coroutine)
        assert not inspect.isawaitable(result)


# ---------------------------------------------------------------------------
# 5. Layer.add_content with content= (old backward-compat path)
# ---------------------------------------------------------------------------


class TestLayerAddContentBackwardCompat:
    def test_old_content_path_returns_same_blocks(self):
        """FR6 backward compat: add_content(content=[block]) → get_content() returns [block]."""
        layer = make_layer()
        block = TextBlock(text="legacy content")

        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                content=[block],
            )
        )

        content = layer.get_content()
        assert len(content) == 1
        assert content[0].text == "legacy content"

    def test_old_content_path_does_not_write_to_store(self):
        """FR7: Old content= path does NOT write to the provenance store."""
        store = InMemoryProvenanceStore()
        layer = make_layer(provenance_store=store)

        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                content=[TextBlock(text="x")],
            )
        )

        # _pending_writes should be empty — no store writes for old-path content
        pending = getattr(layer, "_pending_writes", [])
        assert pending == []

    def test_multiple_old_path_calls_accumulate_content(self):
        """Backward compat: multiple content= add_content calls accumulate in order."""
        layer = make_layer()

        layer.add_content(
            ResolvedContent(
                resolver_name="r1",
                source_layer="test",
                content=[TextBlock(text="first")],
            )
        )
        layer.add_content(
            ResolvedContent(
                resolver_name="r2",
                source_layer="test",
                content=[TextBlock(text="second")],
            )
        )

        content = layer.get_content()
        assert len(content) == 2
        assert content[0].text == "first"
        assert content[1].text == "second"


# ---------------------------------------------------------------------------
# 6. process_pending flushes entries to store
# ---------------------------------------------------------------------------


class TestLayerProcessPendingFlushesStore:
    @pytest.mark.asyncio
    async def test_process_pending_writes_entries_to_store(self):
        """FR7: After process_pending, entries added via entries= path are in the store."""
        store = InMemoryProvenanceStore()
        layer = make_layer(provenance_store=store)
        entry = make_entry()

        layer.add_content(
            ResolvedContent(resolver_name="r", source_layer="test", entries=[entry])
        )
        await layer.process_pending()

        retrieved = await store.get(entry.id)
        assert retrieved is entry

    @pytest.mark.asyncio
    async def test_process_pending_after_old_path_add_content_is_noop(self):
        """Backward compat: process_pending after old content= path does not write to store."""
        store = InMemoryProvenanceStore()
        layer = make_layer(provenance_store=store)

        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                content=[TextBlock(text="x")],
            )
        )
        await layer.process_pending()

        session_entries = await store.get_session("session-test")
        assert session_entries == []

    @pytest.mark.asyncio
    async def test_process_pending_with_no_pending_writes_is_noop(self):
        """process_pending with nothing buffered does not error."""
        store = InMemoryProvenanceStore()
        layer = make_layer(provenance_store=store)

        # No add_content calls — no pending writes
        await layer.process_pending()
        session_entries = await store.get_session("session-test")
        assert session_entries == []

    @pytest.mark.asyncio
    async def test_full_engine_run_entries_in_store(self):
        """FR7: Full engine run with entry-returning resolver → entries are in the engine's store."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.layer import Layer

        store = InMemoryProvenanceStore()
        entry = make_entry()
        resolver = EntryResolver(entries=[entry])
        bus = EventBus()

        layer = Layer(
            name="conversation",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
            provenance_store=store,
        )

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        await engine.run(user_input=[])

        retrieved = await store.get(entry.id)
        assert retrieved is entry


# ---------------------------------------------------------------------------
# 7. Layer.get_content always returns content objects (not Entry wrappers)
# ---------------------------------------------------------------------------


class TestLayerGetContentReturnType:
    def test_entries_path_get_content_returns_content_objects(self):
        """After entries path, get_content returns unwrapped content, not Entry objects."""
        layer = make_layer()
        msg = Message(role="user", content=[TextBlock(text="hello")])
        entry = make_entry(content=msg)

        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                entries=[entry],
            )
        )

        content = layer.get_content()
        assert len(content) == 1
        assert isinstance(content[0], Message)
        assert content[0] is msg

    def test_mixed_entries_and_no_content_blocks(self):
        """Multiple entries: get_content extracts each entry's content in order."""
        layer = make_layer()
        block1 = TextBlock(text="alpha")
        block2 = TextBlock(text="beta")
        entry1 = make_entry(content=block1)
        entry2 = make_entry(content=block2)

        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                entries=[entry1, entry2],
            )
        )

        content = layer.get_content()
        assert content[0] is block1
        assert content[1] is block2


# ---------------------------------------------------------------------------
# 8. Layer.compile unchanged after migration
# ---------------------------------------------------------------------------


class TestLayerCompileAfterMigration:
    def test_system_layer_compile_with_entries_returns_text_blocks(self):
        """FR12: SYSTEM layer compiled from entries still returns list[TextBlock]."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        block = TextBlock(text="You are a helpful assistant.")
        entry = make_entry(content=block)
        layer.add_content(
            ResolvedContent(
                resolver_name="sys",
                source_layer="system_prompt",
                entries=[entry],
            )
        )

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert len(compiled) == 1
        assert isinstance(compiled[0], TextBlock)
        assert compiled[0].text == "You are a helpful assistant."

    def test_messages_layer_compile_with_message_entries_returns_messages(self):
        """FR12: MESSAGES layer compiled from Message entries still returns list[Message]."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="conversation",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        msg = Message(role="user", content=[TextBlock(text="hello")])
        entry = make_entry(content=msg)
        layer.add_content(
            ResolvedContent(
                resolver_name="session",
                source_layer="conversation",
                entries=[entry],
            )
        )

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert len(compiled) == 1
        assert isinstance(compiled[0], Message)
        assert compiled[0].role == "user"

    def test_system_layer_compile_with_content_path_returns_text_blocks(self):
        """Backward compat: SYSTEM layer using old content= path compiles correctly."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        layer.add_content(
            ResolvedContent(
                resolver_name="sys",
                source_layer="system_prompt",
                content=[TextBlock(text="You are EDI.")],
            )
        )

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert len(compiled) == 1
        assert isinstance(compiled[0], TextBlock)
        assert compiled[0].text == "You are EDI."


# ---------------------------------------------------------------------------
# 9. PipelineEngine provenance_store parameter
# ---------------------------------------------------------------------------


class TestPipelineEngineProvenanceStore:
    def test_engine_without_provenance_store_constructs_fine(self):
        """FR12: PipelineEngine without provenance_store= constructs without error."""
        from sr2.pipeline.engine import PipelineEngine

        engine = PipelineEngine(
            layers=[],
            token_counter=CharacterTokenCounter(),
        )
        assert engine is not None

    def test_engine_without_provenance_store_has_default_store(self):
        """Engine without explicit store has an InMemoryProvenanceStore as default."""
        from sr2.pipeline.engine import PipelineEngine

        engine = PipelineEngine(
            layers=[],
            token_counter=CharacterTokenCounter(),
        )
        assert isinstance(engine._provenance_store, InMemoryProvenanceStore)

    def test_engine_with_explicit_provenance_store_uses_it(self):
        """FR12: Engine passed an explicit provenance_store uses that store."""
        from sr2.pipeline.engine import PipelineEngine

        store = InMemoryProvenanceStore()
        engine = PipelineEngine(
            layers=[],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        assert engine._provenance_store is store

    @pytest.mark.asyncio
    async def test_engine_passes_store_to_layers(self):
        """Engine propagates its provenance_store to each layer on construction."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.layer import Layer

        store = InMemoryProvenanceStore()
        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        # The layer should now use the engine's store
        assert layer._provenance_store is store
