"""Tests for sr2.pipeline.layer — Layer class.

Covers:
  FR10: Layer checks token budget after resolvers report; emits overflow event if exceeded
  FR11: Force truncate (oldest first) when over budget after transformers exhaust
  FR12: Layer formats content for compilation target (system/messages/tools)
  FR13: Resolvers always return list[ContentBlock] — layer handles wrapping
  FR14: Compilation target inferred from name, overridable
  FR15: Position strategy is pluggable (prefix, append built-in)
  FR17: Layer is "done" when all components are exhausted, idle, or completed
"""

import pytest

from sr2.models import (
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sr2.pipeline.compilation import AppendStrategy, PrefixStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Stubs — minimal protocol-conforming fakes for testing
# ---------------------------------------------------------------------------


class StubResolver:
    """A resolver that returns predetermined content."""

    def __init__(
        self,
        name: str = "stub_resolver",
        content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._content = content or []
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            content=self._content,
        )


class StubTransformer:
    """A transformer that returns content unchanged (or with a fixed transform)."""

    def __init__(
        self,
        name: str = "stub_transformer",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        transform_fn=None,
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._transform_fn = transform_fn

    async def transform(
        self, content: list, events: list[Event]
    ) -> TransformationResult:
        self.execution_count += 1
        if self._transform_fn:
            result_content = self._transform_fn(content)
        else:
            result_content = content
        return TransformationResult(
            transformer_name=self.name,
            source_layer="test",
            content=result_content,
        )


# ---------------------------------------------------------------------------
# Helper to make text content of a specific token size (using CharacterTokenCounter)
# CharacterTokenCounter: 4 chars = 1 token
# ---------------------------------------------------------------------------


def make_text_block(text: str) -> TextBlock:
    return TextBlock(text=text)


def make_text_blocks_with_tokens(token_count: int, label: str = "x") -> list[TextBlock]:
    """Create a single TextBlock that counts as approximately `token_count` tokens."""
    # CharacterTokenCounter: 4 chars = 1 token
    return [TextBlock(text=label * (token_count * 4))]


# ---------------------------------------------------------------------------
# 1. Layer construction
# ---------------------------------------------------------------------------


class TestLayerConstruction:
    def test_construct_with_all_params(self):
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        counter = CharacterTokenCounter()
        resolver = StubResolver()
        transformer = StubTransformer()

        layer = Layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1000,
            resolvers=[resolver],
            transformers=[transformer],
            token_counter=counter,
            event_bus=bus,
        )
        assert layer.name == "system_prompt"
        assert layer.target == CompilationTarget.SYSTEM
        assert layer.token_budget == 1000
        assert len(layer.resolvers) == 1
        assert len(layer.transformers) == 1

    def test_construct_with_no_budget(self):
        """Layer with token_budget=None means no budget enforcement."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="unbounded",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.token_budget is None

    def test_construct_with_no_transformers(self):
        """Layer with empty transformers list is valid."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="simple",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=100,
            resolvers=[StubResolver()],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.transformers == []


# ---------------------------------------------------------------------------
# 2. Content management — get_content() and add_content()
# ---------------------------------------------------------------------------


class TestLayerContentManagement:
    def test_get_content_empty_initially(self):
        """get_content() returns empty list before any resolvers run."""
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
        assert layer.get_content() == []

    def test_add_content_from_resolver(self):
        """After adding resolved content, get_content() returns it."""
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

        resolved = ResolvedContent(
            resolver_name="sys",
            source_layer="test",
            content=[TextBlock(text="You are helpful.")],
        )
        layer.add_content(resolved)

        content = layer.get_content()
        assert len(content) == 1
        assert content[0].text == "You are helpful."

    def test_multiple_resolvers_content_accumulated_in_order(self):
        """Content from multiple add_content calls accumulates in order."""
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

        layer.add_content(ResolvedContent(
            resolver_name="first",
            source_layer="test",
            content=[TextBlock(text="first")],
        ))
        layer.add_content(ResolvedContent(
            resolver_name="second",
            source_layer="test",
            content=[TextBlock(text="second")],
        ))

        content = layer.get_content()
        assert len(content) == 2
        assert content[0].text == "first"
        assert content[1].text == "second"


# ---------------------------------------------------------------------------
# 3. is_done()
# ---------------------------------------------------------------------------


class TestLayerIsDone:
    def test_no_resolvers_no_transformers_is_done(self):
        """Layer with no components is done immediately."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="empty",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is True

    def test_resolver_never_fired_is_idle_and_done(self):
        """A resolver that never fired (subscribed event never arrived) is idle = done."""
        from sr2.pipeline.layer import Layer

        resolver = StubResolver(
            subscriptions=[EventSubscription(event_name="some_event", phase=EventPhase.STARTING)],
            max_executions=1,
        )
        # execution_count stays 0 — never triggered

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is True

    def test_resolver_hit_max_executions_is_exhausted_and_done(self):
        """A resolver that has hit max_executions is exhausted = done."""
        from sr2.pipeline.layer import Layer

        resolver = StubResolver(max_executions=1)
        resolver.execution_count = 1  # simulate having fired once

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is True

    def test_resolver_fired_but_not_at_max_executions_is_not_done(self):
        """A resolver that has fired but hasn't hit max_executions is NOT done."""
        from sr2.pipeline.layer import Layer

        resolver = StubResolver(max_executions=3)
        resolver.execution_count = 1  # fired once, but max is 3

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is False

    def test_transformer_hit_max_executions_is_done(self):
        """A transformer that has hit max_executions is exhausted = done."""
        from sr2.pipeline.layer import Layer

        transformer = StubTransformer(max_executions=1)
        transformer.execution_count = 1

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[transformer],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is True

    def test_transformer_not_exhausted_means_layer_not_done(self):
        """If any transformer hasn't hit max_executions and has fired, layer is not done."""
        from sr2.pipeline.layer import Layer

        transformer = StubTransformer(max_executions=2)
        transformer.execution_count = 1  # fired once, max is 2

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[transformer],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is False

    def test_mixed_components_all_done(self):
        """Layer is done only when ALL resolvers and transformers are done."""
        from sr2.pipeline.layer import Layer

        resolver = StubResolver(max_executions=1)
        resolver.execution_count = 1  # exhausted

        transformer = StubTransformer(max_executions=1)
        transformer.execution_count = 1  # exhausted

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[transformer],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is True

    def test_one_component_not_done_means_layer_not_done(self):
        """If even one component isn't done, the layer isn't done."""
        from sr2.pipeline.layer import Layer

        resolver_done = StubResolver(max_executions=1)
        resolver_done.execution_count = 1

        resolver_not_done = StubResolver(max_executions=2)
        resolver_not_done.execution_count = 1

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver_done, resolver_not_done],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        assert layer.is_done() is False


# ---------------------------------------------------------------------------
# 4. Budget checking
# ---------------------------------------------------------------------------


class TestLayerBudgetChecking:
    def test_under_budget_no_overflow_event(self):
        """FR10: Layer under budget does not emit overflow event."""
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        overflow_events = []
        bus.subscribe(
            EventSubscription(event_name="overflow"),
            lambda e: overflow_events.append(e),
        )

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=100,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        # Add content under budget: "abcd" = 4 chars = 1 token, well under 100
        layer.add_content(ResolvedContent(
            resolver_name="sys",
            source_layer="test",
            content=[TextBlock(text="abcd")],
        ))
        layer.check_budget()

        assert overflow_events == []

    def test_over_budget_emits_overflow_event(self):
        """FR10: Layer over budget emits overflow event on the bus."""
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        overflow_events = []
        bus.subscribe(
            EventSubscription(event_name="overflow"),
            lambda e: overflow_events.append(e),
        )

        layer = Layer(
            name="test_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=2,  # 2 tokens max
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        # Add content over budget: 20 chars = 5 tokens, over budget of 2
        layer.add_content(ResolvedContent(
            resolver_name="big",
            source_layer="test_layer",
            content=[TextBlock(text="a" * 20)],
        ))
        layer.check_budget()

        assert len(overflow_events) == 1
        assert overflow_events[0].name == "overflow"
        assert overflow_events[0].source_layer == "test_layer"

    def test_no_budget_never_emits_overflow(self):
        """FR10: Layer with token_budget=None never emits overflow."""
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        overflow_events = []
        bus.subscribe(
            EventSubscription(event_name="overflow"),
            lambda e: overflow_events.append(e),
        )

        layer = Layer(
            name="unbounded",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        # Add a lot of content
        layer.add_content(ResolvedContent(
            resolver_name="huge",
            source_layer="unbounded",
            content=[TextBlock(text="a" * 10000)],
        ))
        layer.check_budget()

        assert overflow_events == []


# ---------------------------------------------------------------------------
# 5. Force truncate
# ---------------------------------------------------------------------------


class TestLayerForceTruncate:
    def test_force_truncate_removes_oldest_content_first(self):
        """FR11: Force truncate removes oldest content first until under budget."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=3,  # 3 tokens max
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        # Add 3 blocks, each 2 tokens (8 chars each). Total = 6 tokens, budget = 3.
        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="test",
            content=[TextBlock(text="aaaaaaaa")],  # 8 chars = 2 tokens
        ))
        layer.add_content(ResolvedContent(
            resolver_name="r2",
            source_layer="test",
            content=[TextBlock(text="bbbbbbbb")],  # 8 chars = 2 tokens
        ))
        layer.add_content(ResolvedContent(
            resolver_name="r3",
            source_layer="test",
            content=[TextBlock(text="cccc")],  # 4 chars = 1 token
        ))

        # Total: 5 tokens, budget: 3 tokens. Need to remove 2 tokens.
        # Oldest first: remove "aaaaaaaa" (2 tokens). Now 3 tokens = exactly budget.
        result = layer.force_truncate()

        content = layer.get_content()
        # Should have removed the oldest block(s) until under budget
        texts = [b.text for b in content]
        assert "aaaaaaaa" not in texts
        assert "cccc" in texts  # newest should survive

    def test_force_truncate_returns_warning(self):
        """FR11: Force truncate emits a warning string."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="test",
            content=[TextBlock(text="a" * 40)],  # 40 chars = 10 tokens, way over budget of 1
        ))

        warning = layer.force_truncate()
        assert isinstance(warning, str)
        assert len(warning) > 0

    def test_force_truncate_with_no_budget_does_nothing(self):
        """Force truncate on a layer with no budget is a no-op."""
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

        content_blocks = [TextBlock(text="a" * 1000)]
        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="test",
            content=content_blocks,
        ))

        result = layer.force_truncate()
        # Content should be unchanged
        assert layer.get_content() == content_blocks
        # No warning needed
        assert result is None or result == ""

    def test_force_truncate_under_budget_does_nothing(self):
        """Force truncate when already under budget should be a no-op."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=100,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="test",
            content=[TextBlock(text="abcd")],  # 1 token, well under 100
        ))

        result = layer.force_truncate()
        assert len(layer.get_content()) == 1
        assert result is None or result == ""


# ---------------------------------------------------------------------------
# 6. Compile — system target
# ---------------------------------------------------------------------------


class TestLayerCompileSystem:
    def test_system_target_produces_text_blocks(self):
        """FR12: system target -> list[TextBlock]."""
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

        layer.add_content(ResolvedContent(
            resolver_name="sys",
            source_layer="system_prompt",
            content=[TextBlock(text="You are a helpful assistant.")],
        ))

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert len(compiled) == 1
        assert isinstance(compiled[0], TextBlock)
        assert compiled[0].text == "You are a helpful assistant."

    def test_system_target_multiple_content_blocks(self):
        """FR12: Multiple ContentBlocks compile to multiple TextBlocks."""
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

        layer.add_content(ResolvedContent(
            resolver_name="persona",
            source_layer="system_prompt",
            content=[TextBlock(text="You are EDI.")],
        ))
        layer.add_content(ResolvedContent(
            resolver_name="rules",
            source_layer="system_prompt",
            content=[TextBlock(text="Follow these rules.")],
        ))

        compiled = layer.compile()
        assert len(compiled) == 2
        assert all(isinstance(b, TextBlock) for b in compiled)
        assert compiled[0].text == "You are EDI."
        assert compiled[1].text == "Follow these rules."


# ---------------------------------------------------------------------------
# 7. Compile — messages target
# ---------------------------------------------------------------------------


class TestLayerCompileMessages:
    def test_messages_target_produces_messages(self):
        """FR12: messages target -> list[Message]."""
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

        layer.add_content(ResolvedContent(
            resolver_name="session",
            source_layer="conversation",
            content=[TextBlock(text="Hello there")],
        ))

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert len(compiled) == 1
        assert isinstance(compiled[0], Message)
        assert len(compiled[0].content) == 1
        assert compiled[0].content[0].text == "Hello there"

    def test_messages_target_preserves_content_types(self):
        """FR12: Message compilation preserves different content block types."""
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

        mixed_content = [
            TextBlock(text="user said hello"),
            ToolUseBlock(id="t1", name="search", input={"q": "test"}),
            ToolResultBlock(tool_use_id="t1", content="result"),
        ]
        layer.add_content(ResolvedContent(
            resolver_name="session",
            source_layer="conversation",
            content=mixed_content,
        ))

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert all(isinstance(m, Message) for m in compiled)
        # Verify the content blocks are preserved inside the Message(s)
        all_blocks = []
        for msg in compiled:
            all_blocks.extend(msg.content)
        block_types = {type(b) for b in all_blocks}
        assert TextBlock in block_types
        assert ToolUseBlock in block_types
        assert ToolResultBlock in block_types


# ---------------------------------------------------------------------------
# 8. Compile — tools target
# ---------------------------------------------------------------------------


class TestLayerCompileTools:
    def test_tools_target_empty_compiles_to_empty_list(self):
        """FR12: Empty tools layer compiles to empty list."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="tools",
            target=CompilationTarget.TOOLS,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        compiled = layer.compile()
        assert compiled == []

    def test_tools_target_with_tool_definitions_added(self):
        """Tools added to a tools layer should compile to ToolDefinition list."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="tools",
            target=CompilationTarget.TOOLS,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        tool_def = ToolDefinition(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        layer.add_tool_definitions([tool_def])

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert len(compiled) == 1
        assert isinstance(compiled[0], ToolDefinition)
        assert compiled[0].name == "search"


# ---------------------------------------------------------------------------
# 9. Position strategy
# ---------------------------------------------------------------------------


class TestLayerPositionStrategy:
    def test_append_strategy_orders_content_in_add_order(self):
        """AppendStrategy: content from second resolver goes after first."""
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

        layer.add_content(ResolvedContent(
            resolver_name="first",
            source_layer="test",
            content=[TextBlock(text="AAA")],
        ))
        layer.add_content(ResolvedContent(
            resolver_name="second",
            source_layer="test",
            content=[TextBlock(text="BBB")],
        ))

        content = layer.get_content()
        assert content[0].text == "AAA"
        assert content[1].text == "BBB"

    def test_prefix_strategy_orders_new_content_before_existing(self):
        """PrefixStrategy: content from second resolver goes before first."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=PrefixStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        layer.add_content(ResolvedContent(
            resolver_name="first",
            source_layer="test",
            content=[TextBlock(text="AAA")],
        ))
        layer.add_content(ResolvedContent(
            resolver_name="second",
            source_layer="test",
            content=[TextBlock(text="BBB")],
        ))

        content = layer.get_content()
        assert content[0].text == "BBB"  # second added, but prefixed
        assert content[1].text == "AAA"


# ---------------------------------------------------------------------------
# 10. Integration-style: full cycle
# ---------------------------------------------------------------------------


class TestLayerFullCycle:
    def test_resolve_budget_check_truncate_compile(self):
        """Full cycle: add content -> over budget -> force truncate -> compile."""
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        overflow_events = []
        bus.subscribe(
            EventSubscription(event_name="overflow"),
            lambda e: overflow_events.append(e),
        )

        layer = Layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=5,  # 5 tokens max
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        # Add content: 12 chars = 3 tokens
        layer.add_content(ResolvedContent(
            resolver_name="old_resolver",
            source_layer="system_prompt",
            content=[TextBlock(text="aaaaaaaaaaaa")],  # 12 chars = 3 tokens
        ))

        # Add more content: 16 chars = 4 tokens. Total now: 7 tokens, over budget of 5.
        layer.add_content(ResolvedContent(
            resolver_name="new_resolver",
            source_layer="system_prompt",
            content=[TextBlock(text="bbbbbbbbbbbbbbbb")],  # 16 chars = 4 tokens
        ))

        # Check budget -> should emit overflow
        layer.check_budget()
        assert len(overflow_events) == 1

        # Force truncate -> should remove oldest until under budget
        warning = layer.force_truncate()
        assert warning is not None and len(warning) > 0

        # Content should now be under budget
        counter = CharacterTokenCounter()
        remaining = layer.get_content()
        assert counter.count(remaining) <= 5

        # Compile should produce TextBlocks for system target
        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert all(isinstance(b, TextBlock) for b in compiled)

    def test_compile_empty_layer_returns_empty_list(self):
        """Compiling a layer with no content returns an empty list."""
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

        compiled = layer.compile()
        assert compiled == []

    def test_set_content_replaces_existing(self):
        """set_content() replaces all content (used after transformer runs)."""
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

        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="test",
            content=[TextBlock(text="original")],
        ))
        assert layer.get_content()[0].text == "original"

        # set_content replaces all content (e.g., after a transformer runs)
        layer.set_content([TextBlock(text="transformed")])
        assert len(layer.get_content()) == 1
        assert layer.get_content()[0].text == "transformed"


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------


class TestLayerEdgeCases:
    def test_budget_exactly_met_no_overflow(self):
        """Content exactly at budget should NOT trigger overflow."""
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        overflow_events = []
        bus.subscribe(
            EventSubscription(event_name="overflow"),
            lambda e: overflow_events.append(e),
        )

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=2,  # 2 tokens
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        # 8 chars = exactly 2 tokens = exactly budget
        layer.add_content(ResolvedContent(
            resolver_name="exact",
            source_layer="test",
            content=[TextBlock(text="abcdefgh")],
        ))
        layer.check_budget()

        assert overflow_events == []

    def test_force_truncate_removes_multiple_blocks_if_needed(self):
        """Force truncate may need to remove multiple blocks to get under budget."""
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1,  # 1 token = 4 chars
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        # Add 4 blocks of 2 tokens each = 8 tokens total
        for label in ["aaaa1111", "bbbb2222", "cccc3333", "dddd"]:
            layer.add_content(ResolvedContent(
                resolver_name=f"r_{label[:4]}",
                source_layer="test",
                content=[TextBlock(text=label)],
            ))

        layer.force_truncate()
        remaining = layer.get_content()

        # Should have removed enough oldest blocks to get under/at 1 token (4 chars)
        counter = CharacterTokenCounter()
        assert counter.count(remaining) <= 1

    def test_overflow_event_has_correct_source_layer(self):
        """The overflow event should reference the correct source_layer."""
        from sr2.pipeline.layer import Layer

        bus = EventBus()
        overflow_events = []
        bus.subscribe(
            EventSubscription(event_name="overflow"),
            lambda e: overflow_events.append(e),
        )

        layer = Layer(
            name="my_special_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="my_special_layer",
            content=[TextBlock(text="a" * 100)],
        ))
        layer.check_budget()

        assert overflow_events[0].source_layer == "my_special_layer"
