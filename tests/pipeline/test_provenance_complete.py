"""sr2-92 / AC1 — Provenance-complete compiled context (unit layer).

Cage for the removal of the raw-content/no-store path and for the causal
metadata the spec's Interfaces section adds to ``sr2.pipeline.provenance``
(ContextReason, ContextAuthority, ContextLifecycle, ContextScope, and Entry
gaining reason/authority/lifecycle/scope).

One test per confirmed case-table row (AC1-r1..r8, r12). Store-level rows
(r9-r11) live in tests/pipeline/test_sqlite_provenance.py.

Grounding rules honored here:
  - Values are asserted only where the case table / spec fixes them. Where a
    value is not spec-fixed, presence/validity is asserted (isinstance of the
    spec enum, non-None).
  - Mechanism is NOT pinned: whether causal fields arrive via resolver-built
    entries, layer synthesis of raw content, or defaults normalized at the
    store boundary is the implementer's choice. Tests exercise public APIs
    (resolver -> layer -> process_pending / engine run -> provenance store)
    only.
  - The spec symbols do not exist yet, so they are referenced through the
    module object (``prov.ContextReason`` etc.). Collection succeeds; each
    test fails individually with a clean AttributeError until C implements.
"""

from __future__ import annotations

import dataclasses
import inspect
from datetime import datetime, timezone

import pytest

import sr2.pipeline.provenance as prov
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Spec-vocabulary accessors — module attribute access, NOT module-level
# imports, so a missing symbol fails per-test instead of breaking collection.
# ---------------------------------------------------------------------------


def ContextReason():
    return prov.ContextReason


def ContextAuthority():
    return prov.ContextAuthority


def ContextLifecycle():
    return prov.ContextLifecycle


def ContextScope(**kwargs):
    return prov.ContextScope(**kwargs)


def _valid_authority(value) -> None:
    """Assert presence/validity of an authority value the table leaves unfixed."""
    assert value is not None
    assert isinstance(value, ContextAuthority())


def _valid_lifecycle(value) -> None:
    assert value is not None
    assert isinstance(value, ContextLifecycle())


# ---------------------------------------------------------------------------
# ID helper — 26-char ULID-shaped, same convention as existing provenance tests
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_id() -> str:
    global _COUNTER
    _COUNTER += 1
    return f"PC{_COUNTER:024d}"


def _matches_content(item, content) -> bool:
    """Content equality that tolerates pydantic models, dataclasses, and
    unhashable fields alike."""
    if item == content:
        return True
    try:
        return item.model_dump() == content.model_dump()
    except (AttributeError, TypeError, ValueError):
        return False


def _find_stored(store, content):
    """Return the stored Entry whose content equals ``content``.

    Asserts the compiled item maps to exactly one stored entry. Entries are
    compared by field values (content equality), never by object identity.
    """
    matches = [e for e in store._store.values() if _matches_content(e.content, content)]
    assert len(matches) == 1, (
        f"expected exactly one stored Entry for content {content!r}, "
        f"got {len(matches)}"
    )
    return matches[0]


def _entry_for_text(store, text: str):
    """Return the stored Entry wrapping the TextBlock whose text equals ``text``."""
    return _find_stored(store, TextBlock(text=text))


# ---------------------------------------------------------------------------
# Fixture drivers (pipeline-level, per the case-table grounding rules)
# ---------------------------------------------------------------------------


def _new_entry(
    *,
    entry_id: str,
    content,
    sources: tuple = (),
    origin_kind: str,
    origin_name: str,
    layer: str,
    session_id: str,
):
    """Construct a valid Entry for test driving.

    Passes only the constructor arguments that exist TODAY (id, content,
    sources, origin, layer, session_id, created_at). Any new causal fields
    C adds must carry spec-declared defaults; construction succeeds either
    way, so the assertion target stays store-visible behavior, not the
    constructor signature.
    """
    return prov.Entry(
        id=entry_id,
        content=content,
        sources=sources,
        origin=prov.EntryOrigin(kind=origin_kind, name=origin_name),
        layer=layer,
        session_id=session_id,
        created_at=datetime.now(tz=timezone.utc),
    )


class _RetrievalDeclaringStore:
    """Store-boundary declarer for AC1-r5.

    The pipeline synthesizes provenance for raw content without causal
    fields. To exercise RETRIEVAL round-tripping without pinning whether the
    reason comes from resolver-built entries or layer synthesis, this
    boundary store declares the retrieval reason/lifecycle for the one
    fixture item on first write; the assertions then check what the store
    holds for that compiled item.
    """

    def __init__(self, content, reason, lifecycle):
        self._inner = prov.InMemoryProvenanceStore()
        self._content = content
        self._reason = reason
        self._lifecycle = lifecycle
        self.seen_write = False

    def _declare(self, entry):
        if not _matches_content(entry.content, self._content):
            return entry
        try:
            return dataclasses.replace(
                entry, reason=self._reason, lifecycle=self._lifecycle
            )
        except TypeError as exc:  # pragma: no cover - documents the red reason
            raise AttributeError(
                "Entry lacks causal fields reason/lifecycle "
                f"(sr2-92 target interface): {exc}"
            ) from exc

    async def write(self, entry):
        self.seen_write = True
        await self._inner.write(self._declare(entry))

    async def write_batch(self, entries):
        self.seen_write = True
        await self._inner.write_batch([self._declare(e) for e in entries])

    def __getattr__(self, name):
        return getattr(self._inner, name)


class EntryResolver:
    """Resolver returning ResolvedContent with entries= populated."""

    def __init__(
        self,
        name: str = "entry_resolver",
        entries: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        source_layer: str = "test",
    ):
        self.name = name
        self._entries = entries or []
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer=self._source_layer,
            entries=self._entries,
        )


class RawResolver:
    """Resolver returning raw ResolvedContent.content (legacy path)."""

    def __init__(
        self,
        name: str = "raw_resolver",
        content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        source_layer: str = "test",
    ):
        self.name = name
        self._content = content or []
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer=self._source_layer,
            content=self._content,
        )


class EntryTransformer:
    """Transformer emitting content plus entries= (reason TRANSFORMATION)."""

    def __init__(
        self,
        name: str = "entry_transformer",
        entries: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        source_layer: str = "conversation",
    ):
        self.name = name
        self._entries = entries or []
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer

    async def transform(self, content: list, events: list[Event]) -> TransformationResult:
        self.execution_count += 1
        # content=None so set_content is skipped: pre-transform entries stay
        # in the layer while the transformation entries are processed.
        return TransformationResult(
            transformer_name=self.name,
            source_layer=self._source_layer,
            content=None,
            entries=self._entries,
        )


def make_layer(
    name: str = "conversation",
    target: CompilationTarget = CompilationTarget.MESSAGES,
    provenance_store=None,
):
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


def _make_static_resolver(text: str):
    """Production StaticResolver with a minimal in-test config (no conftest edits)."""
    from sr2.pipeline.resolvers.static import StaticResolver

    class _Cfg:
        type = "static"
        name = None
        config = {"text": text}
        subscriptions: list = []
        max_executions = 1

    return StaticResolver(_Cfg())


def _make_input_resolver():
    """Production InputResolver with a minimal in-test config."""
    from sr2.pipeline.resolvers.input import InputResolver

    class _Cfg:
        type = "input"
        name = None
        config: dict = {}
        subscriptions: list = []
        max_executions = 1

    return InputResolver(_Cfg())


def _user_input_event(text: str) -> Event:
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="test",
        data=[TextBlock(text=text)],
    )


# ---------------------------------------------------------------------------
# AC1-r1 — Static context (StaticResolver): every compiled item maps to a
# stored immutable Entry with reason/authority/lifecycle/scope.
# ---------------------------------------------------------------------------


class TestAC1R1StaticContextStoredEntry:
    async def test_static_text_compiled_item_maps_to_stored_entry_with_causal_fields(
        self,
    ):
        """AC1-r1: StaticResolver TextBlock in a compiled turn -> stored Entry
        retrievable by ID, reason SYSTEM_CONFIG, valid authority, valid
        lifecycle, ContextScope with non-empty owner_id; compiled content
        unchanged."""
        from sr2.pipeline.engine import PipelineEngine

        store = prov.InMemoryProvenanceStore()
        static_text = "You are a helpful assistant."
        resolver = _make_static_resolver(static_text)
        layer = make_layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            provenance_store=store,
        )
        layer.resolvers = [resolver]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        result = await engine.end_turn()

        # Compiled content preserved (no-store removal must not change output).
        assert result.request.system is not None
        assert any(b.text == static_text for b in result.request.system)

        # The compiled item maps to a stored immutable Entry...
        stored = _entry_for_text(store, static_text)
        fetched = await store.get(stored.id)
        assert fetched is not None
        assert fetched == stored

        # ...with the causal fields the spec names.
        assert fetched.reason == ContextReason().SYSTEM_CONFIG
        _valid_authority(fetched.authority)
        _valid_lifecycle(fetched.lifecycle)
        assert isinstance(fetched.scope, prov.ContextScope)
        assert isinstance(fetched.scope.owner_id, str)
        assert fetched.scope.owner_id != ""


# ---------------------------------------------------------------------------
# AC1-r2 — Live markdown state (MarkdownFileResolver on a fixture file).
# ---------------------------------------------------------------------------


class TestAC1R2LiveMarkdownStoredEntry:
    async def test_markdown_state_maps_to_stored_live_source_entry(self, tmp_path):
        """AC1-r2: MarkdownFileResolver content -> stored Entry with reason
        LIVE_SOURCE, authority LIVE_SOURCE, and a stable source reference for
        the file recorded in meta."""
        from sr2.pipeline.protocols import Resolver
        from sr2.pipeline.resolvers.markdown_file import MarkdownFileResolver

        md_file = tmp_path / "state.md"
        md_text = "# Current state\nWork stopped at the resolver design."
        md_file.write_text(md_text, encoding="utf-8")

        class _Cfg:
            type = "markdown_file"
            name = None
            config = {"path": str(md_file)}
            subscriptions: list = []
            max_executions = 1

        resolver = MarkdownFileResolver(_Cfg())
        assert isinstance(resolver, Resolver)  # protocol sanity

        store = prov.InMemoryProvenanceStore()
        layer = make_layer(name="live_state", provenance_store=store)
        layer.resolvers = [resolver]
        await layer.process_pending()  # no-op flush; resolve manually below

        resolved = await resolver.resolve(
            [Event(name="turn_start", phase=EventPhase.STARTING, source_layer="test")]
        )
        assert resolved.content and resolved.content[0].text == md_text
        layer.add_content(resolved)
        await layer.process_pending()

        stored = _entry_for_text(store, md_text)
        assert stored.reason == ContextReason().LIVE_SOURCE
        assert stored.authority == ContextAuthority().LIVE_SOURCE

        # meta records a stable source reference for the file: at least one
        # string value naming the fixture file's path.
        refs = [v for v in stored.meta.values() if isinstance(v, str)]
        assert any(str(md_file) in ref for ref in refs), (
            f"no meta value references the source file {str(md_file)!r}; "
            f"meta={stored.meta!r}"
        )


# ---------------------------------------------------------------------------
# AC1-r3 — Prior/recent dialogue: seeded prior Messages replayed into the
# compiled request (production SessionResolver).
# ---------------------------------------------------------------------------


class TestAC1R3SeededDialogueStoredEntries:
    async def test_seeded_prior_messages_map_to_stored_entries_with_causal_fields(
        self,
    ):
        """AC1-r3: seeded prior user + assistant messages -> each replayed
        message maps to a stored Entry; user -> USER_INPUT/USER,
        assistant -> ASSISTANT_OUTPUT/AGENT_TENTATIVE; lifecycle RECENT."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.resolvers.session import SessionResolver

        class _Cfg:
            type = "session"
            name = None
            config: dict = {}
            subscriptions: list = []
            max_executions = 2**62

        session_resolver = SessionResolver(_Cfg())
        prior_user = Message(role="user", content=[TextBlock(text="prior question")])
        prior_assistant = Message(
            role="assistant", content=[TextBlock(text="prior answer")]
        )
        session_resolver.seed([prior_user, prior_assistant])

        store = prov.InMemoryProvenanceStore()
        layer = make_layer(name="conversation", provenance_store=store)
        layer.resolvers = [session_resolver]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        result = await engine.end_turn()

        # Replayed into the compiled request. A standalone Message in layer
        # content compiles into the request messages (current behavior);
        # exact request shape is pinned by the existing engine tests, not
        # here — assert the payload texts made it in.
        assert result.request.messages is not None
        compiled_texts = {
            block.text
            for m in result.request.messages
            for block in (m.content if isinstance(m.content, list) else [])
            if isinstance(block, TextBlock)
        }
        assert "prior question" in compiled_texts
        assert "prior answer" in compiled_texts

        # Each replayed message maps to a stored Entry with the fixed values.
        user_entry = _find_stored(store, prior_user)
        assert user_entry.reason == ContextReason().USER_INPUT
        assert user_entry.authority == ContextAuthority().USER
        assert user_entry.lifecycle == ContextLifecycle().RECENT

        assistant_entry = _find_stored(store, prior_assistant)
        assert assistant_entry.reason == ContextReason().ASSISTANT_OUTPUT
        assert assistant_entry.authority == ContextAuthority().AGENT_TENTATIVE
        assert assistant_entry.lifecycle == ContextLifecycle().RECENT


# ---------------------------------------------------------------------------
# AC1-r4 — Current user input (production InputResolver).
# ---------------------------------------------------------------------------


class TestAC1R4CurrentUserInputStoredEntry:
    async def test_current_user_input_maps_to_stored_turn_entry(self):
        """AC1-r4: InputResolver-wrapped user input -> stored Entry with
        reason USER_INPUT, authority USER, lifecycle TURN."""
        from sr2.pipeline.engine import PipelineEngine

        store = prov.InMemoryProvenanceStore()
        input_resolver = _make_input_resolver()
        layer = make_layer(name="conversation", provenance_store=store)
        layer.resolvers = [input_resolver]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        await engine.continue_turn(
            events=[_user_input_event("what is the weather?")], iteration_seq=0
        )
        result = await engine.end_turn()

        # Compiled into the request as user-message content (current
        # behavior preserved).
        expected = Message(
            role="user", content=[TextBlock(text="what is the weather?")]
        )
        assert result.request.messages is not None
        compiled = list(result.request.messages)
        assert expected in compiled or any(
            expected.content == m.content for m in compiled
        )

        stored = _find_stored(store, expected)
        assert stored.reason == ContextReason().USER_INPUT
        assert stored.authority == ContextAuthority().USER
        assert stored.lifecycle == ContextLifecycle().TURN


# ---------------------------------------------------------------------------
# AC1-r5 — Retrieved continuity item (fixture retrieval-class resolver).
# Production continuity retrieval is sr2-95; the row allows a fixture
# resolver driving the pipeline.
# ---------------------------------------------------------------------------


class TestAC1R5RetrievalStoredEntry:
    async def test_retrieved_item_maps_to_stored_retrieved_entry(self):
        """AC1-r5: retrieval-class resolver injects fetched content -> stored
        Entry with reason RETRIEVAL and lifecycle RETRIEVED."""
        from sr2.pipeline.engine import PipelineEngine

        retrieved_block = TextBlock(text="retrieved: backend decision was SQLite")

        # Mechanism-free driver: emit raw content now (current
        # continuity-disabled shape) and declare the item's causal fields
        # through the store vocabulary. The pipeline must persist the item
        # with the declared/derived retrieval reason and lifecycle.
        raw_resolver = RawResolver(
            name="continuity_retrieval",
            content=[retrieved_block],
            source_layer="retrieval",
        )
        store = _RetrievalDeclaringStore(
            content=retrieved_block,
            reason=ContextReason().RETRIEVAL,
            lifecycle=ContextLifecycle().RETRIEVED,
        )
        layer = make_layer(name="conversation", provenance_store=store)
        layer.resolvers = [resolver]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        result = await engine.end_turn()

        compiled = list(result.request.messages or [])
        assert retrieved_block in compiled or any(
            retrieved_block in (m.content if isinstance(m.content, list) else [])
            for m in compiled
        )

        stored = _find_stored(store, retrieved_block)
        assert stored.reason == ContextReason().RETRIEVAL
        assert stored.lifecycle == ContextLifecycle().RETRIEVED

        # The declaring store must have seen the item through the real
        # pipeline write path (not a pre-seeded shortcut).
        assert store.seen_write, (
            "retrieved item never reached the provenance store write path"
        )


# ---------------------------------------------------------------------------
# AC1-r6 — ToolUseBlock and ToolResultBlock in compiled content.
 # Production tool-loop wiring is sr2-93; row r6 allows direct layer driving.
# ---------------------------------------------------------------------------


class TestAC1R6ToolBlocksStoredEntries:
    async def test_tool_use_and_tool_result_map_to_stored_entries(self):
        """AC1-r6: ToolUseBlock -> reason TOOL_USE; ToolResultBlock -> reason
        TOOL_RESULT; each authority a valid ContextAuthority (value not
        spec-fixed); both stored and compiled.

        Entries are driven through the entries= API (the resolver-origin
        entry path already exists); the test asserts the store sees the
        causal reasons, not who set them.
        """
        store = prov.InMemoryProvenanceStore()
        layer = make_layer(name="conversation", provenance_store=store)

        tool_use = ToolUseBlock(
            id="call_001", name="get_weather", input={"location": "Oslo"}
        )
        tool_result = ToolResultBlock(tool_use_id="call_001", content="sunny, 15C")

        use_entry = _new_entry(
            entry_id=_next_id(),
            content=tool_use,
            origin_kind="resolver",
            origin_name="tool_events",
            layer="conversation",
            session_id="session-ac1r6",
        )
        result_entry = _new_entry(
            entry_id=_next_id(),
            content=tool_result,
            origin_kind="resolver",
            origin_name="tool_events",
            layer="conversation",
            session_id="session-ac1r6",
        )
        if not hasattr(use_entry, "reason"):
            raise AttributeError(
                "Entry lacks the 'reason' field (sr2-92 target interface)"
            )
        use_entry = dataclasses.replace(use_entry, reason=ContextReason().TOOL_USE)
        result_entry = dataclasses.replace(
            result_entry, reason=ContextReason().TOOL_RESULT
        )

        layer.add_content(
            ResolvedContent(
                resolver_name="tool_events",
                source_layer="conversation",
                entries=[use_entry, result_entry],
            )
        )
        await layer.process_pending()

        # Compiled content contains both blocks (current behavior).
        content = layer.get_content()
        assert tool_use in content
        assert tool_result in content

        stored_use = await store.get(use_entry.id)
        assert stored_use is not None
        assert stored_use.reason == ContextReason().TOOL_USE
        _valid_authority(stored_use.authority)

        stored_result = await store.get(result_entry.id)
        assert stored_result is not None
        assert stored_result.reason == ContextReason().TOOL_RESULT
        _valid_authority(stored_result.authority)


# ---------------------------------------------------------------------------
# AC1-r7 — Final assistant output content.
# ---------------------------------------------------------------------------


class TestAC1R7AssistantOutputStoredEntry:
    async def test_assistant_output_maps_to_stored_entry(self):
        """AC1-r7: final assistant output content -> stored Entry with reason
        ASSISTANT_OUTPUT and authority AGENT_TENTATIVE."""
        store = prov.InMemoryProvenanceStore()
        layer = make_layer(name="conversation", provenance_store=store)

        assistant_msg = Message(
            role="assistant", content=[TextBlock(text="the final answer")]
        )
        layer.add_content(
            ResolvedContent(
                resolver_name="assistant_output",
                source_layer="conversation",
                content=[assistant_msg],
            )
        )
        await layer.process_pending()

        # Compiled content preserved (current behavior).
        assert assistant_msg in layer.get_content()

        stored = _find_stored(store, assistant_msg)
        assert stored.reason == ContextReason().ASSISTANT_OUTPUT
        assert stored.authority == ContextAuthority().AGENT_TENTATIVE


# ---------------------------------------------------------------------------
# AC1-r8 — Transformer output: TransformationResult.entries accompanying
# transformed content.
# ---------------------------------------------------------------------------


class TestAC1R8TransformationEntriesLineage:
    async def test_transformer_entries_stored_with_real_source_lineage(self):
        """AC1-r8: transformation entries are stored with reason
        TRANSFORMATION; their sources are entry IDs present in the store;
        get_lineage reaches the original source entries."""
        from sr2.pipeline.engine import PipelineEngine

        store = prov.InMemoryProvenanceStore()

        original = TextBlock(text="original conversation content")
        src_entry = _new_entry(
            entry_id=_next_id(),
            content=original,
            origin_kind="resolver",
            origin_name="fixture_origin",
            layer="conversation",
            session_id="session-ac1r8",
        )

        summary = TextBlock(text="summary: original conversation content")
        t_entry = _new_entry(
            entry_id=_next_id(),
            content=summary,
            sources=(src_entry.id,),
            origin_kind="transformer",
            origin_name="fixture_summarizer",
            layer="conversation",
            session_id="session-ac1r8",
        )
        if not hasattr(t_entry, "reason"):
            # Clean per-test failure with the spec reason instead of the
            # dataclass ValidationError from an unknown enum value.
            raise AttributeError(
                "Entry lacks the 'reason' field (sr2-92 target interface)"
            )
        t_entry = dataclasses.replace(t_entry, reason=ContextReason().TRANSFORMATION)

        layer = make_layer(name="conversation", provenance_store=store)
        layer.resolvers = [
            EntryResolver(name="fixture_origin", entries=[src_entry], source_layer="conversation")
        ]
        layer.transformers = [
            EntryTransformer(name="fixture_summarizer", entries=[t_entry])
        ]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        await engine.end_turn()

        # Both entries stored; transformation reason recorded.
        stored_t = await store.get(t_entry.id)
        assert stored_t is not None
        assert stored_t.reason == ContextReason().TRANSFORMATION
        stored_src = await store.get(src_entry.id)
        assert stored_src is not None

        # Sources reference IDs present in the store...
        for source_id in stored_t.sources:
            assert await store.get(source_id) is not None, (
                f"transformation entry {stored_t.id} references absent source {source_id!r}"
            )

        # ...and lineage traversal reaches the original source entry.
        lineage = await store.get_lineage(t_entry.id)
        lineage_ids = {e.id for e in lineage}
        assert src_entry.id in lineage_ids
        assert t_entry.id in lineage_ids


# ---------------------------------------------------------------------------
# AC1-r12 — Backward compat: raw content, no entries (continuity-disabled
# style). Compiled content identical AND raw items now stored.
# ---------------------------------------------------------------------------


class LegacyRawResolver:
    """Legacy-shape resolver: returns raw content only, exactly as
    continuity-disabled configurations produce today."""

    def __init__(self):
        self.name = "legacy_raw"
        self.max_executions = 1
        self.execution_count = 0
        self.subscriptions = [
            EventSubscription(event_name="user_input", phase=EventPhase.STARTING)
        ]

    async def resolve(self, events: list) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="legacy",
            content=[TextBlock(text="legacy raw content")],
        )


class TestAC1R12RawPathStillStored:
    async def test_engine_run_with_legacy_raw_resolver_compiles_current_behavior(
        self,
    ):
        """AC1-r12 (compiled side): an engine run with a raw-content-only
        resolver compiles the same request contents as current behavior."""
        from sr2.pipeline.engine import PipelineEngine

        store = prov.InMemoryProvenanceStore()
        layer = make_layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            provenance_store=store,
        )
        layer.resolvers = [LegacyRawResolver()]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        await engine.continue_turn(
            events=[_user_input_event("go")], iteration_seq=0
        )
        result = await engine.end_turn()

        assert result.request.system is not None
        assert any(
            b.text == "legacy raw content" for b in result.request.system
        )

    async def test_raw_content_items_are_now_written_to_the_store(self):
        """AC1-r12 (stored side): raw ResolvedContent.content items map to
        stored Entries after process_pending — the no-store bypass is gone."""
        store = prov.InMemoryProvenanceStore()
        layer = make_layer(name="conversation", provenance_store=store)

        block = TextBlock(text="raw content must now be stored")
        layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                content=[block],
            )
        )

        # Compiled content identical to current behavior, immediately.
        assert block in layer.get_content()

        await layer.process_pending()

        stored = _find_stored(store, block)
        _valid_authority(stored.authority)
        _valid_lifecycle(stored.lifecycle)
        assert isinstance(stored.scope, prov.ContextScope)

    async def test_transformer_raw_content_is_now_written_to_the_store(self):
        """AC1-r12 (stored side, transformer path): a transformer that returns
        only raw content has that content stored too — no no-store bypass
        anywhere on the compiled-content path."""
        from sr2.pipeline.engine import PipelineEngine

        store = prov.InMemoryProvenanceStore()
        original = TextBlock(text="pre-transform content")
        transformed = TextBlock(text="post-transform content")

        class _RawTransformer:
            name = "raw_transformer"
            max_executions = 1
            execution_count = 0
            subscriptions = [
                EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
            ]

            async def transform(self, content, events) -> TransformationResult:
                self.execution_count += 1
                return TransformationResult(
                    transformer_name=self.name,
                    source_layer="conversation",
                    content=[original, transformed],
                )

        layer = make_layer(name="conversation", provenance_store=store)
        layer.transformers = [_RawTransformer()]

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )
        next_seq = engine.turn_seq + 1
        await engine.start_turn(turn_seq=next_seq)
        result = await engine.end_turn()

        compiled_texts = [
            block.text
            for m in (result.request.messages or [])
            for block in (m.content if isinstance(m.content, list) else [])
            if isinstance(block, TextBlock)
        ]
        assert "pre-transform content" in compiled_texts
        assert "post-transform content" in compiled_texts

        # Both raw transformer outputs map to stored entries.
        _find_stored(store, original)
        _find_stored(store, transformed)

    async def test_add_content_remains_synchronous(self):
        """AC1-r12 (shape): storing the raw path must not change add_content's
        synchronous signature (public API backward compat)."""
        layer = make_layer()
        result = layer.add_content(
            ResolvedContent(
                resolver_name="r",
                source_layer="test",
                content=[TextBlock(text="x")],
            )
        )
        assert result is None
        assert not inspect.isawaitable(result)
