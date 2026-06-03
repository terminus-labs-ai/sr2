"""Tests for MemoryExtractionTransformer.

Acceptance criteria covered:
  - build() raises ConfigError when memory_store absent from deps
  - build() succeeds when memory_store present via typed field
  - build() uses memory_extractor from typed field if present, else defaults to RuleBasedExtractor
  - subscriptions() returns subscription to assistant_response.COMPLETED
  - transform() extracts memories from response text and saves to store
  - transform() returns TransformationResult with content=None (pass-through)
  - transform() with text that yields no memories still returns pass-through result
"""

from __future__ import annotations

import pytest

from sr2.config.models import ConfigError, TransformerConfig
from sr2.memory import (
    InMemoryMemoryStore,
    Memory,
    MemoryExtractor,
    RuleBasedExtractor,
)
from sr2.memory.schema import ExtractionResult, MemoryScope
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.protocols import Transformer


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def make_config(max_executions: int = 10) -> TransformerConfig:
    """Build a minimal TransformerConfig for MemoryExtractionTransformer."""
    return TransformerConfig(
        type="memory_extraction",
        max_executions=max_executions,
    )


def make_deps(
    *,
    memory_store=None,
    memory_extractor=None,
) -> Dependencies:
    """Build Dependencies using typed fields."""
    return Dependencies(  # type: ignore[call-arg]
        memory_store=memory_store,
        memory_extractor=memory_extractor,
    )


def make_completed_event(text: str = "I prefer dark mode") -> Event:
    """Build a plausible assistant_response.COMPLETED event carrying TextBlocks."""
    return Event(
        name="assistant_response",
        phase=EventPhase.COMPLETED,
        source_layer="engine",
        data=[TextBlock(text=text)],
    )


def build_transformer(deps: Dependencies, config: TransformerConfig | None = None):
    """Convenience: import and build MemoryExtractionTransformer."""
    from sr2.memory.extraction_transformer import MemoryExtractionTransformer

    cfg = config or make_config()
    return MemoryExtractionTransformer.build(cfg, deps)


# ---------------------------------------------------------------------------
# Stub extractor — controllable for testing
# ---------------------------------------------------------------------------


class StubExtractor:
    """Returns a fixed list of memories, regardless of input text."""

    def __init__(self, memories: list[Memory] | None = None) -> None:
        self._memories = memories or []
        self.calls: list[str] = []

    def extract(
        self,
        turn_text: str,
        turn_id: str | None = None,
        scope_override: MemoryScope | None = None,
    ) -> ExtractionResult:
        self.calls.append(turn_text)
        return ExtractionResult(memories=list(self._memories), source_turn_id=turn_id)


class EmptyExtractor(StubExtractor):
    """Always returns zero memories."""

    def __init__(self) -> None:
        super().__init__(memories=[])


# ---------------------------------------------------------------------------
# TestBuildConfigError
# ---------------------------------------------------------------------------


class TestBuildConfigError:
    """build() raises ConfigError when memory_store is absent from deps."""

    def test_raises_config_error_when_store_absent(self):
        """ConfigError raised when deps has no memory_store."""
        deps = make_deps()  # no memory_store
        config = make_config()

        with pytest.raises(ConfigError):
            from sr2.memory.extraction_transformer import MemoryExtractionTransformer
            MemoryExtractionTransformer.build(config, deps)

    def test_raises_config_error_not_key_error(self):
        """Exception type must be ConfigError, not KeyError or AttributeError."""
        deps = make_deps()
        config = make_config()

        exc = None
        try:
            from sr2.memory.extraction_transformer import MemoryExtractionTransformer
            MemoryExtractionTransformer.build(config, deps)
        except Exception as e:
            exc = e

        assert exc is not None, "Expected an exception"
        assert isinstance(exc, ConfigError), (
            f"Expected ConfigError, got {type(exc).__name__}: {exc}"
        )

    def test_error_message_mentions_memory_store(self):
        """ConfigError message should mention memory_store."""
        deps = make_deps()
        config = make_config()

        with pytest.raises(ConfigError) as exc_info:
            from sr2.memory.extraction_transformer import MemoryExtractionTransformer
            MemoryExtractionTransformer.build(config, deps)

        assert "memory_store" in str(exc_info.value).lower()

    def test_empty_deps_also_raises(self):
        """Empty Dependencies raises ConfigError when memory_store is absent."""
        deps = Dependencies()
        config = make_config()

        with pytest.raises(ConfigError):
            from sr2.memory.extraction_transformer import MemoryExtractionTransformer
            MemoryExtractionTransformer.build(config, deps)


# ---------------------------------------------------------------------------
# TestBuildSuccess
# ---------------------------------------------------------------------------


class TestBuildSuccess:
    """build() succeeds when memory_store is present in deps typed field."""

    def test_build_succeeds_with_memory_store(self):
        """build() does not raise when memory_store is in deps."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)
        config = make_config()

        from sr2.memory.extraction_transformer import MemoryExtractionTransformer
        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_build_returns_transformer_instance(self):
        """build() returns an instance of MemoryExtractionTransformer."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)
        config = make_config()

        result = MemoryExtractionTransformer.build(config, deps)
        assert isinstance(result, MemoryExtractionTransformer)

    def test_build_satisfies_transformer_protocol(self):
        """Built instance satisfies the Transformer protocol."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)

        transformer = build_transformer(deps)
        assert isinstance(transformer, Transformer)

    def test_build_is_classmethod(self):
        """build is a classmethod on MemoryExtractionTransformer."""
        import inspect
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        assert isinstance(
            inspect.getattr_static(MemoryExtractionTransformer, "build"),
            classmethod,
        )


# ---------------------------------------------------------------------------
# TestExtractorInjection
# ---------------------------------------------------------------------------


class TestExtractorInjection:
    """build() uses memory_extractor from typed field if present, else defaults to RuleBasedExtractor."""

    def test_uses_provided_extractor(self):
        """When memory_extractor is in deps, it is used (not RuleBasedExtractor)."""
        store = InMemoryMemoryStore()
        stub = StubExtractor()
        deps = make_deps(memory_store=store, memory_extractor=stub)

        transformer = build_transformer(deps)

        # The transformer should hold our stub, not a RuleBasedExtractor
        assert transformer._extractor is stub

    def test_defaults_to_rule_based_extractor_when_absent(self):
        """When memory_extractor absent from deps, defaults to RuleBasedExtractor."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)  # no extractor

        transformer = build_transformer(deps)

        assert isinstance(transformer._extractor, RuleBasedExtractor)

    @pytest.mark.asyncio
    async def test_provided_extractor_is_called_during_transform(self):
        """Stub extractor is actually invoked during transform()."""
        store = InMemoryMemoryStore()
        stub = StubExtractor()
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        event = make_completed_event("I prefer dark mode")
        await transformer.transform([], [event])

        assert len(stub.calls) > 0


# ---------------------------------------------------------------------------
# TestSubscriptions
# ---------------------------------------------------------------------------


class TestSubscriptions:
    """subscriptions returns a subscription to assistant_response.COMPLETED."""

    def test_has_subscriptions_attribute(self):
        """Transformer has a .subscriptions attribute."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)
        transformer = build_transformer(deps)

        assert hasattr(transformer, "subscriptions")

    def test_subscriptions_is_list(self):
        """subscriptions is a list."""
        from sr2.pipeline.events import EventSubscription

        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)
        transformer = build_transformer(deps)

        assert isinstance(transformer.subscriptions, list)

    def test_subscriptions_contains_assistant_response_completed(self):
        """subscriptions includes event_name='assistant_response' with phase=COMPLETED."""
        from sr2.pipeline.events import EventSubscription

        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)
        transformer = build_transformer(deps)

        names = [s.event_name for s in transformer.subscriptions]
        assert "assistant_response" in names

    def test_subscription_phase_is_completed(self):
        """The assistant_response subscription has phase=COMPLETED."""
        from sr2.pipeline.events import EventPhase, EventSubscription

        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)
        transformer = build_transformer(deps)

        sub = next(
            s for s in transformer.subscriptions if s.event_name == "assistant_response"
        )
        assert sub.phase == EventPhase.COMPLETED


# ---------------------------------------------------------------------------
# TestTransformPassThrough
# ---------------------------------------------------------------------------


class TestTransformPassThrough:
    """transform() returns TransformationResult with content=None (pass-through)."""

    @pytest.mark.asyncio
    async def test_returns_transformation_result(self):
        """transform() returns a TransformationResult."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store, memory_extractor=EmptyExtractor())
        transformer = build_transformer(deps)

        result = await transformer.transform([], [make_completed_event()])
        assert isinstance(result, TransformationResult)

    @pytest.mark.asyncio
    async def test_content_is_none_pass_through(self):
        """transform() always returns content=None (does not modify the layer)."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store, memory_extractor=EmptyExtractor())
        transformer = build_transformer(deps)

        result = await transformer.transform([], [make_completed_event()])
        assert result.content is None

    @pytest.mark.asyncio
    async def test_content_is_none_even_when_memories_extracted(self):
        """content=None even when memories are extracted — pure pass-through."""
        store = InMemoryMemoryStore()
        memory = Memory(content="I prefer dark mode", scope=MemoryScope.SHARED)
        stub = StubExtractor(memories=[memory])
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        result = await transformer.transform([], [make_completed_event("I prefer dark mode")])
        assert result.content is None

    @pytest.mark.asyncio
    async def test_transformer_name_is_set(self):
        """TransformationResult.transformer_name is a non-empty string."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store, memory_extractor=EmptyExtractor())
        transformer = build_transformer(deps)

        result = await transformer.transform([], [make_completed_event()])
        assert isinstance(result.transformer_name, str)
        assert result.transformer_name != ""

    @pytest.mark.asyncio
    async def test_source_layer_is_set(self):
        """TransformationResult.source_layer is a non-empty string."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store, memory_extractor=EmptyExtractor())
        transformer = build_transformer(deps)

        result = await transformer.transform([], [make_completed_event()])
        assert isinstance(result.source_layer, str)
        assert result.source_layer != ""

    @pytest.mark.asyncio
    async def test_transform_is_coroutine(self):
        """transform() is async (returns a coroutine)."""
        import asyncio

        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store, memory_extractor=EmptyExtractor())
        transformer = build_transformer(deps)

        coro = transformer.transform([], [])
        assert asyncio.iscoroutine(coro)
        await coro  # clean up


# ---------------------------------------------------------------------------
# TestTransformExtractsAndSaves
# ---------------------------------------------------------------------------


class TestTransformExtractsAndSaves:
    """transform() extracts memories from response text and saves to store."""

    @pytest.mark.asyncio
    async def test_extracted_memories_are_saved(self):
        """Memories returned by extractor are saved to the store."""
        store = InMemoryMemoryStore()
        memory = Memory(content="I prefer dark mode", scope=MemoryScope.SHARED)
        stub = StubExtractor(memories=[memory])
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        await transformer.transform([], [make_completed_event("I prefer dark mode")])

        saved = store.get_all()
        assert len(saved) == 1
        assert saved[0].content == "I prefer dark mode"

    @pytest.mark.asyncio
    async def test_multiple_memories_all_saved(self):
        """All memories from extraction are saved, not just the first."""
        store = InMemoryMemoryStore()
        m1 = Memory(content="I prefer dark mode", scope=MemoryScope.SHARED)
        m2 = Memory(content="decided to use Python", scope=MemoryScope.PROJECT)
        stub = StubExtractor(memories=[m1, m2])
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        await transformer.transform([], [make_completed_event("some text")])

        saved = store.get_all()
        assert len(saved) == 2

    @pytest.mark.asyncio
    async def test_text_extracted_from_event_data(self):
        """Text passed to extractor comes from the event's TextBlock content."""
        store = InMemoryMemoryStore()
        stub = StubExtractor()
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        event_text = "I prefer dark mode and use Python"
        await transformer.transform([], [make_completed_event(event_text)])

        assert len(stub.calls) > 0
        assert event_text in stub.calls[0]

    @pytest.mark.asyncio
    async def test_no_memories_extracted_still_returns_pass_through(self):
        """When extractor returns no memories, result is still pass-through."""
        store = InMemoryMemoryStore()
        stub = EmptyExtractor()
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        result = await transformer.transform([], [make_completed_event("xyz")])

        assert result.content is None
        assert store.get_all() == []

    @pytest.mark.asyncio
    async def test_no_relevant_events_no_extraction(self):
        """If events list contains no assistant_response.COMPLETED, no extraction occurs."""
        store = InMemoryMemoryStore()
        stub = StubExtractor(memories=[Memory(content="fact", scope=MemoryScope.SHARED)])
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        # Different event name — should not trigger extraction
        other_event = Event(
            name="turn_start",
            phase=EventPhase.COMPLETED,
            source_layer="engine",
            data=[TextBlock(text="I prefer dark mode")],
        )

        await transformer.transform([], [other_event])

        # Stub should not have been called, or store should have nothing
        # (transformer only processes assistant_response.COMPLETED events)
        saved = store.get_all()
        assert len(saved) == 0

    @pytest.mark.asyncio
    async def test_empty_events_list_returns_pass_through(self):
        """Empty events list → pass-through, no extraction attempt."""
        store = InMemoryMemoryStore()
        stub = StubExtractor(memories=[Memory(content="fact", scope=MemoryScope.SHARED)])
        deps = make_deps(memory_store=store, memory_extractor=stub)
        transformer = build_transformer(deps)

        result = await transformer.transform([], [])

        assert result.content is None
        assert store.get_all() == []

    @pytest.mark.asyncio
    async def test_integration_with_real_rule_based_extractor(self):
        """Integration: real RuleBasedExtractor runs and saves at least some memories
        for text that contains obvious memory triggers."""
        store = InMemoryMemoryStore()
        deps = make_deps(memory_store=store)  # uses default RuleBasedExtractor
        transformer = build_transformer(deps)

        # "I prefer" is a known pattern in RuleBasedExtractor
        event = make_completed_event("I prefer dark mode for coding sessions")
        result = await transformer.transform([], [event])

        # pass-through
        assert result.content is None
        # at least one memory saved (rule-based extractor should match "I prefer")
        saved = store.get_all()
        assert len(saved) >= 1
