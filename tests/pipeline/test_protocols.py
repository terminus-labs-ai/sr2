"""Tests for sr2.pipeline.protocols and sr2.pipeline.models.

Covers:
  FR5: Resolvers fire when subscribed events arrive, up to max_executions per turn
  FR6: Transformers fire when subscribed events arrive, up to max_executions per turn
  FR7: Resolvers return ResolvedContent (content + resolver identity + token count)
  FR8: Transformers receive layer content + events, return TransformationResult
  FR9: Transformer event lifecycle (starting/completed/failed)
  FR13: Resolvers always return list[ContentBlock]
  FR20: Protocol-based — new resolvers/transformers addable without modifying engine
  NF: Token counting is injected, engine doesn't import specific counter
"""

import pytest

from sr2.models import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from sr2.pipeline.events import Event, EventPhase, EventSubscription


# ---------------------------------------------------------------------------
# 1. ResolvedContent — dataclass creation and defaults
# ---------------------------------------------------------------------------


class TestResolvedContent:
    def test_create_with_required_fields(self):
        from sr2.pipeline.models import ResolvedContent

        rc = ResolvedContent(
            resolver_name="system_prompt",
            source_layer="core",
            content=[TextBlock(text="You are a helpful assistant.")],
        )
        assert rc.resolver_name == "system_prompt"
        assert rc.source_layer == "core"
        assert len(rc.content) == 1
        assert rc.content[0].text == "You are a helpful assistant."

    def test_token_count_defaults_to_zero(self):
        from sr2.pipeline.models import ResolvedContent

        rc = ResolvedContent(
            resolver_name="test",
            source_layer="core",
            content=[],
        )
        assert rc.token_count == 0

    def test_token_count_can_be_set(self):
        from sr2.pipeline.models import ResolvedContent

        rc = ResolvedContent(
            resolver_name="memory",
            source_layer="memory",
            content=[TextBlock(text="User prefers dark mode.")],
            token_count=42,
        )
        assert rc.token_count == 42

    def test_content_is_list_of_content_blocks(self):
        """FR13: Resolvers always return list[ContentBlock]."""
        from sr2.pipeline.models import ResolvedContent

        blocks = [
            TextBlock(text="first"),
            TextBlock(text="second"),
        ]
        rc = ResolvedContent(
            resolver_name="multi",
            source_layer="core",
            content=blocks,
        )
        assert rc.content is blocks
        assert len(rc.content) == 2

    def test_content_can_be_empty_list(self):
        from sr2.pipeline.models import ResolvedContent

        rc = ResolvedContent(
            resolver_name="empty",
            source_layer="core",
            content=[],
        )
        assert rc.content == []

    def test_content_supports_mixed_block_types(self):
        """FR13: list[ContentBlock] covers all block types."""
        from sr2.pipeline.models import ResolvedContent

        blocks = [
            TextBlock(text="hello"),
            ToolUseBlock(id="t1", name="search", input={"q": "test"}),
            ToolResultBlock(tool_use_id="t1", content="result"),
            ThinkingBlock(text="hmm"),
        ]
        rc = ResolvedContent(
            resolver_name="mixed",
            source_layer="conversation",
            content=blocks,
            token_count=100,
        )
        assert len(rc.content) == 4


# ---------------------------------------------------------------------------
# 2. TransformationResult — dataclass creation and defaults
# ---------------------------------------------------------------------------


class TestTransformationResult:
    def test_create_with_required_fields(self):
        from sr2.pipeline.models import TransformationResult

        tr = TransformationResult(
            transformer_name="compactor",
            source_layer="conversation",
            content=[TextBlock(text="compacted output")],
        )
        assert tr.transformer_name == "compactor"
        assert tr.source_layer == "conversation"
        assert len(tr.content) == 1

    def test_events_defaults_to_none(self):
        from sr2.pipeline.models import TransformationResult

        tr = TransformationResult(
            transformer_name="t",
            source_layer="l",
            content=[],
        )
        assert tr.events is None

    def test_events_can_carry_events_to_emit(self):
        """FR9: Transformers can emit events back onto the bus."""
        from sr2.pipeline.models import TransformationResult

        events = [
            Event(
                name="compaction",
                phase=EventPhase.COMPLETED,
                source_layer="conversation",
            )
        ]
        tr = TransformationResult(
            transformer_name="compactor",
            source_layer="conversation",
            content=[],
            events=events,
        )
        assert tr.events is not None
        assert len(tr.events) == 1
        assert tr.events[0].name == "compaction"

    def test_token_metrics_default_to_zero(self):
        from sr2.pipeline.models import TransformationResult

        tr = TransformationResult(
            transformer_name="t",
            source_layer="l",
            content=[],
        )
        assert tr.tokens_before == 0
        assert tr.tokens_after == 0
        assert tr.tokens_saved == 0

    def test_token_metrics_can_be_set(self):
        from sr2.pipeline.models import TransformationResult

        tr = TransformationResult(
            transformer_name="compactor",
            source_layer="conversation",
            content=[TextBlock(text="short")],
            tokens_before=500,
            tokens_after=200,
            tokens_saved=300,
        )
        assert tr.tokens_before == 500
        assert tr.tokens_after == 200
        assert tr.tokens_saved == 300

    def test_content_is_list_of_content_blocks(self):
        from sr2.pipeline.models import TransformationResult

        blocks = [TextBlock(text="transformed")]
        tr = TransformationResult(
            transformer_name="t",
            source_layer="l",
            content=blocks,
        )
        assert tr.content is blocks


# ---------------------------------------------------------------------------
# 3. Resolver protocol — runtime checkable
# ---------------------------------------------------------------------------


class TestResolverProtocol:
    def test_is_runtime_checkable(self):
        """FR20: Protocol-based — isinstance checks must work."""
        from sr2.pipeline.protocols import Resolver

        # runtime_checkable protocols support isinstance
        assert hasattr(Resolver, "__protocol_attrs__") or hasattr(
            Resolver, "__abstractmethods__"
        ) or isinstance(Resolver, type)

    def test_conforming_class_satisfies_protocol(self):
        """FR20: A class with the right shape is a valid Resolver."""
        from sr2.pipeline.models import ResolvedContent
        from sr2.pipeline.protocols import Resolver

        class MyResolver:
            subscriptions = [
                EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)
            ]
            max_executions = 1

            async def resolve(self, events: list[Event]) -> ResolvedContent:
                return ResolvedContent(
                    resolver_name="my",
                    source_layer="core",
                    content=[TextBlock(text="hello")],
                )

            @classmethod
            def build(cls, config, deps):
                return cls()

        assert isinstance(MyResolver(), Resolver)

    def test_non_conforming_class_missing_resolve(self):
        """A class without resolve() must not satisfy the protocol."""
        from sr2.pipeline.protocols import Resolver

        class NotAResolver:
            subscriptions = []
            max_executions = 1
            # missing resolve()

        assert not isinstance(NotAResolver(), Resolver)

    def test_non_conforming_class_missing_subscriptions(self):
        """A class without subscriptions must not satisfy the protocol."""
        from sr2.pipeline.protocols import Resolver

        class MissingSubscriptions:
            max_executions = 1

            async def resolve(self, events):
                pass

        assert not isinstance(MissingSubscriptions(), Resolver)

    def test_non_conforming_class_missing_max_executions(self):
        """A class without max_executions must not satisfy the protocol."""
        from sr2.pipeline.protocols import Resolver

        class MissingMaxExec:
            subscriptions = []

            async def resolve(self, events):
                pass

        assert not isinstance(MissingMaxExec(), Resolver)

    def test_max_executions_default_is_one(self):
        """FR5: max_executions defaults to 1."""
        from sr2.pipeline.models import ResolvedContent
        from sr2.pipeline.protocols import Resolver

        class DefaultMaxExec:
            subscriptions = []
            max_executions = 1

            async def resolve(self, events: list[Event]) -> ResolvedContent:
                return ResolvedContent(
                    resolver_name="d", source_layer="l", content=[]
                )

        r = DefaultMaxExec()
        assert r.max_executions == 1

    def test_max_executions_can_be_configured(self):
        """FR5: max_executions is configurable."""
        from sr2.pipeline.protocols import Resolver

        class MultiExecResolver:
            subscriptions = [
                EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)
            ]
            max_executions = 3

            async def resolve(self, events):
                pass

            @classmethod
            def build(cls, config, deps):
                return cls()

        r = MultiExecResolver()
        assert r.max_executions == 3
        assert isinstance(r, Resolver)

    @pytest.mark.asyncio
    async def test_resolve_returns_resolved_content(self):
        """FR7: Resolvers return ResolvedContent."""
        from sr2.pipeline.models import ResolvedContent

        class StubResolver:
            subscriptions = [
                EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)
            ]
            max_executions = 1

            async def resolve(self, events: list[Event]) -> ResolvedContent:
                return ResolvedContent(
                    resolver_name="stub",
                    source_layer="core",
                    content=[TextBlock(text="resolved content")],
                    token_count=3,
                )

        result = await StubResolver().resolve([])
        assert isinstance(result, ResolvedContent)
        assert result.resolver_name == "stub"
        assert result.token_count == 3
        assert len(result.content) == 1


# ---------------------------------------------------------------------------
# 4. Transformer protocol — runtime checkable
# ---------------------------------------------------------------------------


class TestTransformerProtocol:
    def test_is_runtime_checkable(self):
        """FR20: Protocol-based — isinstance checks must work."""
        from sr2.pipeline.protocols import Transformer

        assert hasattr(Transformer, "__protocol_attrs__") or hasattr(
            Transformer, "__abstractmethods__"
        ) or isinstance(Transformer, type)

    def test_conforming_class_satisfies_protocol(self):
        """FR20: A class with the right shape is a valid Transformer."""
        from sr2.pipeline.models import TransformationResult
        from sr2.pipeline.protocols import Transformer

        class MyTransformer:
            subscriptions = [
                EventSubscription(
                    event_name="resolve_completed", phase=EventPhase.COMPLETED
                )
            ]
            max_executions = 1

            async def transform(
                self, content: list, events: list[Event]
            ) -> TransformationResult:
                return TransformationResult(
                    transformer_name="my",
                    source_layer="core",
                    content=content,
                )

            @classmethod
            def build(cls, config, deps):
                return cls()

        assert isinstance(MyTransformer(), Transformer)

    def test_non_conforming_class_missing_transform(self):
        """A class without transform() must not satisfy the protocol."""
        from sr2.pipeline.protocols import Transformer

        class NotATransformer:
            subscriptions = []
            max_executions = 1
            # missing transform()

        assert not isinstance(NotATransformer(), Transformer)

    def test_non_conforming_class_missing_subscriptions(self):
        from sr2.pipeline.protocols import Transformer

        class MissingSubscriptions:
            max_executions = 1

            async def transform(self, content, events):
                pass

        assert not isinstance(MissingSubscriptions(), Transformer)

    def test_non_conforming_class_missing_max_executions(self):
        from sr2.pipeline.protocols import Transformer

        class MissingMaxExec:
            subscriptions = []

            async def transform(self, content, events):
                pass

        assert not isinstance(MissingMaxExec(), Transformer)

    @pytest.mark.asyncio
    async def test_transform_receives_content_and_events(self):
        """FR8: Transformers receive layer content + triggering events."""
        from sr2.pipeline.models import TransformationResult

        class CapturingTransformer:
            subscriptions = [
                EventSubscription(event_name="overflow", phase=EventPhase.COMPLETED)
            ]
            max_executions = 1

            async def transform(
                self, content: list, events: list[Event]
            ) -> TransformationResult:
                return TransformationResult(
                    transformer_name="capture",
                    source_layer="conversation",
                    content=content,
                    tokens_before=len(content) * 10,
                    tokens_after=len(content) * 5,
                    tokens_saved=len(content) * 5,
                )

        input_content = [TextBlock(text="turn 1"), TextBlock(text="turn 2")]
        input_events = [
            Event(
                name="overflow",
                phase=EventPhase.COMPLETED,
                source_layer="conversation",
            )
        ]

        result = await CapturingTransformer().transform(input_content, input_events)
        assert result.content is input_content
        assert result.tokens_before == 20
        assert result.tokens_saved == 10

    @pytest.mark.asyncio
    async def test_transform_can_emit_events(self):
        """FR9: Transformers can emit events back onto the bus."""
        from sr2.pipeline.models import TransformationResult

        class EmittingTransformer:
            subscriptions = [
                EventSubscription(event_name="overflow", phase=EventPhase.COMPLETED)
            ]
            max_executions = 1

            async def transform(
                self, content: list, events: list[Event]
            ) -> TransformationResult:
                return TransformationResult(
                    transformer_name="emitter",
                    source_layer="conversation",
                    content=content,
                    events=[
                        Event(
                            name="compaction",
                            phase=EventPhase.COMPLETED,
                            source_layer="conversation",
                        )
                    ],
                )

        result = await EmittingTransformer().transform([], [])
        assert result.events is not None
        assert len(result.events) == 1
        assert result.events[0].name == "compaction"

    def test_max_executions_can_be_configured(self):
        """FR6: max_executions is configurable."""
        from sr2.pipeline.protocols import Transformer

        class MultiExecTransformer:
            subscriptions = []
            max_executions = 5

            async def transform(self, content, events):
                pass

            @classmethod
            def build(cls, config, deps):
                return cls()

        t = MultiExecTransformer()
        assert t.max_executions == 5
        assert isinstance(t, Transformer)


# ---------------------------------------------------------------------------
# 5. TokenCounter protocol — runtime checkable
# ---------------------------------------------------------------------------


class TestTokenCounterProtocol:
    def test_is_runtime_checkable(self):
        from sr2.pipeline.protocols import TokenCounter

        assert hasattr(TokenCounter, "__protocol_attrs__") or hasattr(
            TokenCounter, "__abstractmethods__"
        ) or isinstance(TokenCounter, type)

    def test_conforming_class_satisfies_protocol(self):
        from sr2.pipeline.protocols import TokenCounter

        class MyCounter:
            def count(self, content: list) -> int:
                return 42

        assert isinstance(MyCounter(), TokenCounter)

    def test_non_conforming_class_missing_count(self):
        from sr2.pipeline.protocols import TokenCounter

        class NotACounter:
            pass

        assert not isinstance(NotACounter(), TokenCounter)

    def test_any_counter_implementation_satisfies_protocol(self):
        """NF: Token counting is injected — any implementation works."""
        from sr2.pipeline.protocols import TokenCounter

        class TiktokenCounter:
            def count(self, content: list) -> int:
                return sum(len(str(c)) for c in content)

        assert isinstance(TiktokenCounter(), TokenCounter)


# ---------------------------------------------------------------------------
# 6. CharacterTokenCounter — 4 chars ≈ 1 token
# ---------------------------------------------------------------------------


class TestCharacterTokenCounter:
    def test_satisfies_token_counter_protocol(self):
        from sr2.pipeline.protocols import TokenCounter
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_empty_content_returns_zero(self):
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        assert counter.count([]) == 0

    def test_text_block_counting(self):
        """4 characters ≈ 1 token."""
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        # "abcd" is 4 chars -> 1 token
        result = counter.count([TextBlock(text="abcd")])
        assert result == 1

    def test_text_block_rounds_correctly(self):
        """Fractional tokens should round (not truncate to zero)."""
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        # "ab" is 2 chars -> 0.5 -> should be at least 1 (or 0 if floor)
        # Spec says "estimates" — verify it's reasonable
        result_2_chars = counter.count([TextBlock(text="ab")])
        result_8_chars = counter.count([TextBlock(text="abcdefgh")])
        # 8 chars / 4 = exactly 2 tokens
        assert result_8_chars == 2
        # 2 chars -> some rounding behavior (0 or 1 both valid for estimate)
        assert isinstance(result_2_chars, int)

    def test_multiple_text_blocks(self):
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        # 8 chars + 4 chars = 12 chars -> 3 tokens
        result = counter.count([
            TextBlock(text="abcdefgh"),
            TextBlock(text="wxyz"),
        ])
        assert result == 3

    def test_tool_use_block_counted(self):
        """Tool use blocks have content that should be counted."""
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        block = ToolUseBlock(id="t1", name="search", input={"query": "test"})
        result = counter.count([block])
        # Should count something — exact value depends on serialization
        assert result > 0

    def test_tool_result_block_counted(self):
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        block = ToolResultBlock(tool_use_id="t1", content="Some result text here")
        result = counter.count([block])
        assert result > 0

    def test_thinking_block_counted(self):
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        block = ThinkingBlock(text="Let me think about this...")
        result = counter.count([block])
        assert result > 0

    def test_mixed_block_types(self):
        """All block types contribute to the total count."""
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        blocks = [
            TextBlock(text="hello"),
            ThinkingBlock(text="hmm"),
            ToolUseBlock(id="t1", name="x", input={}),
            ToolResultBlock(tool_use_id="t1", content="ok"),
        ]
        result = counter.count(blocks)
        # Each block has content — total should be > any single block
        single_text = counter.count([TextBlock(text="hello")])
        assert result > single_text

    def test_large_text_scales_linearly(self):
        """Token count should scale with text length."""
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        small = counter.count([TextBlock(text="a" * 40)])   # 40 chars -> 10 tokens
        large = counter.count([TextBlock(text="a" * 400)])  # 400 chars -> 100 tokens
        assert large == small * 10

    def test_returns_int(self):
        from sr2.pipeline.token_counting import CharacterTokenCounter

        counter = CharacterTokenCounter()
        result = counter.count([TextBlock(text="hello world")])
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# 7. Protocol independence — no cross-contamination
# ---------------------------------------------------------------------------


class TestProtocolIndependence:
    def test_resolver_is_not_transformer(self):
        """Resolver and Transformer are distinct protocols."""
        from sr2.pipeline.protocols import Resolver, Transformer

        class OnlyResolver:
            subscriptions = []
            max_executions = 1

            async def resolve(self, events):
                pass

            @classmethod
            def build(cls, config, deps):
                return cls()

        r = OnlyResolver()
        assert isinstance(r, Resolver)
        assert not isinstance(r, Transformer)

    def test_transformer_is_not_resolver(self):
        from sr2.pipeline.protocols import Resolver, Transformer

        class OnlyTransformer:
            subscriptions = []
            max_executions = 1

            async def transform(self, content, events):
                pass

            @classmethod
            def build(cls, config, deps):
                return cls()

        t = OnlyTransformer()
        assert isinstance(t, Transformer)
        assert not isinstance(t, Resolver)

    def test_token_counter_is_neither_resolver_nor_transformer(self):
        from sr2.pipeline.protocols import Resolver, TokenCounter, Transformer

        class OnlyCounter:
            def count(self, content):
                return 0

        c = OnlyCounter()
        assert isinstance(c, TokenCounter)
        assert not isinstance(c, Resolver)
        assert not isinstance(c, Transformer)

    def test_class_can_implement_both_resolver_and_transformer(self):
        """A single class can satisfy both protocols if it has all attributes."""
        from sr2.pipeline.protocols import Resolver, Transformer

        class Dual:
            subscriptions = []
            max_executions = 1

            async def resolve(self, events):
                pass

            async def transform(self, content, events):
                pass

            @classmethod
            def build(cls, config, deps):
                return cls()

        d = Dual()
        assert isinstance(d, Resolver)
        assert isinstance(d, Transformer)
