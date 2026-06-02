"""Tests for sr2.pipeline.engine — PipelineEngine, PipelineResult, PipelineMetrics, LayerMetrics.

Covers:
  FR1:  Pipeline instantiates layers with resolvers and transformers
  FR2:  Resolvers and transformers subscribe to events on the bus
  FR4:  Pipeline emits turn_start and user_input events at run start
  FR5:  Resolvers fire when subscribed events arrive, up to max_executions
  FR6:  Transformers fire when subscribed events arrive, up to max_executions
  FR7:  Resolvers return ResolvedContent
  FR8:  Transformers receive layer content + triggering events, return TransformationResult
  FR10: Layer checks token budget after resolvers; emits overflow if exceeded
  FR11: Force truncate oldest first + warning if still over budget
  FR16: Pipeline terminates when bus empty + all layers done -> turn_end
  FR18: Compilation groups layers by target, orders by declaration, applies position strategy
  FR19: PipelineResult with CompletionRequest + per-layer metrics
"""

import pytest

from conftest import run_engine
from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
from sr2.pipeline.compilation import AppendStrategy, PrefixStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest


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
        source_layer: str = "test",
    ):
        self.name = name
        self._content = content or []
        self.subscriptions = subscriptions or []
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


class CapturingResolver:
    """A resolver that captures the events it receives."""

    def __init__(
        self,
        name: str = "capturing_resolver",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        source_layer: str = "test",
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer
        self.captured_events: list[list[Event]] = []

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        self.captured_events.append(list(events))
        return ResolvedContent(
            resolver_name=self.name,
            source_layer=self._source_layer,
            content=[TextBlock(text=f"resolved by {self.name}")],
        )


class StubTransformer:
    """A transformer that returns content unchanged (or with a fixed transform)."""

    def __init__(
        self,
        name: str = "stub_transformer",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        transform_fn=None,
        events_to_emit: list[Event] | None = None,
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._transform_fn = transform_fn
        self._events_to_emit = events_to_emit

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
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
            events=self._events_to_emit,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text(text: str) -> TextBlock:
    return TextBlock(text=text)


def make_text_blocks_with_tokens(token_count: int, label: str = "x") -> list[TextBlock]:
    """Create a single TextBlock that counts as approximately `token_count` tokens.

    CharacterTokenCounter: 4 chars = 1 token.
    """
    return [TextBlock(text=label * (token_count * 4))]


def make_system_layer(
    name: str = "system_prompt",
    resolvers: list | None = None,
    transformers: list | None = None,
    token_budget: int | None = None,
    position=None,
    token_counter=None,
    event_bus=None,
):
    """Helper to build a system-target Layer."""
    from sr2.pipeline.layer import Layer

    return Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=position or AppendStrategy(),
        token_budget=token_budget,
        resolvers=resolvers or [],
        transformers=transformers or [],
        token_counter=token_counter or CharacterTokenCounter(),
        event_bus=event_bus or EventBus(),
    )


def make_messages_layer(
    name: str = "conversation",
    resolvers: list | None = None,
    transformers: list | None = None,
    token_budget: int | None = None,
    position=None,
    token_counter=None,
    event_bus=None,
):
    """Helper to build a messages-target Layer."""
    from sr2.pipeline.layer import Layer

    return Layer(
        name=name,
        target=CompilationTarget.MESSAGES,
        position=position or AppendStrategy(),
        token_budget=token_budget,
        resolvers=resolvers or [],
        transformers=transformers or [],
        token_counter=token_counter or CharacterTokenCounter(),
        event_bus=event_bus or EventBus(),
    )


# ---------------------------------------------------------------------------
# 1. PipelineResult & metrics dataclasses
# ---------------------------------------------------------------------------


class TestPipelineResultDataclass:
    def test_pipeline_result_has_request_and_metrics(self):
        """PipelineResult must have request (CompletionRequest) and metrics (PipelineMetrics)."""
        from sr2.pipeline.engine import PipelineMetrics, PipelineResult

        request = CompletionRequest(messages=[])
        metrics = PipelineMetrics(layers={}, total_tokens=0, warnings=[])
        result = PipelineResult(request=request, metrics=metrics)

        assert result.request is request
        assert result.metrics is metrics

    def test_pipeline_metrics_has_layers_total_tokens_warnings(self):
        """PipelineMetrics must have layers dict, total_tokens int, warnings list."""
        from sr2.pipeline.engine import PipelineMetrics

        metrics = PipelineMetrics(
            layers={"system": None},  # type: ignore — testing shape
            total_tokens=42,
            warnings=["something truncated"],
        )

        assert isinstance(metrics.layers, dict)
        assert metrics.total_tokens == 42
        assert metrics.warnings == ["something truncated"]

    def test_pipeline_metrics_empty_defaults(self):
        """PipelineMetrics with empty values."""
        from sr2.pipeline.engine import PipelineMetrics

        metrics = PipelineMetrics(layers={}, total_tokens=0, warnings=[])

        assert metrics.layers == {}
        assert metrics.total_tokens == 0
        assert metrics.warnings == []

    def test_layer_metrics_has_all_fields(self):
        """LayerMetrics has tokens_used, token_budget, budget_remaining, force_truncated,
        resolver_executions, transformer_executions."""
        from sr2.pipeline.engine import LayerMetrics

        lm = LayerMetrics(
            tokens_used=50,
            token_budget=100,
            budget_remaining=50,
            force_truncated=False,
            resolver_executions={"sys_prompt": 1},
            transformer_executions={"compactor": 2},
        )

        assert lm.tokens_used == 50
        assert lm.token_budget == 100
        assert lm.budget_remaining == 50
        assert lm.force_truncated is False
        assert lm.resolver_executions == {"sys_prompt": 1}
        assert lm.transformer_executions == {"compactor": 2}

    def test_layer_metrics_no_budget(self):
        """LayerMetrics with no budget: token_budget and budget_remaining are None."""
        from sr2.pipeline.engine import LayerMetrics

        lm = LayerMetrics(
            tokens_used=50,
            token_budget=None,
            budget_remaining=None,
            force_truncated=False,
            resolver_executions={},
            transformer_executions={},
        )

        assert lm.token_budget is None
        assert lm.budget_remaining is None


# ---------------------------------------------------------------------------
# 2. PipelineEngine construction
# ---------------------------------------------------------------------------


class TestPipelineEngineConstruction:
    def test_construct_with_layers_and_counter(self):
        """PipelineEngine can be constructed with layers and a token_counter."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        layers = [make_system_layer(token_counter=counter)]

        engine = PipelineEngine(layers=layers, token_counter=counter)

        assert engine is not None

    def test_construct_with_empty_layers(self):
        """PipelineEngine with empty layers list is valid."""
        from sr2.pipeline.engine import PipelineEngine

        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())

        assert engine is not None


# ---------------------------------------------------------------------------
# 3. Pipeline.run() — basic execution
# ---------------------------------------------------------------------------


class TestPipelineRunBasicExecution:
    @pytest.mark.asyncio
    async def test_single_resolver_produces_content_in_completion_request(self):
        """FR4/FR5/FR7/FR18/FR19: One layer with one resolver subscribed to turn_start
        produces content that appears in the CompletionRequest."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="You are helpful.")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request is not None
        assert isinstance(result.request, CompletionRequest)
        # System layer should populate system field
        assert result.request.system is not None
        assert len(result.request.system) == 1
        assert result.request.system[0].text == "You are helpful."

    @pytest.mark.asyncio
    async def test_user_input_data_passed_to_resolvers(self):
        """FR4: user_input event carries the user's input blocks as data.
        Resolvers subscribing to user_input can access the data."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        capturing = CapturingResolver(
            name="input_capturer",
            subscriptions=[EventSubscription(event_name="user_input", phase=EventPhase.COMPLETED)],
            source_layer="conversation",
        )

        layer = make_messages_layer(
            name="conversation",
            resolvers=[capturing],
            token_counter=counter,
            event_bus=bus,
        )

        user_input = [TextBlock(text="Hello, world!")]
        engine = PipelineEngine(layers=[layer], token_counter=counter)
        await run_engine(engine, user_input)

        # The resolver should have been called with an event carrying the user input
        assert len(capturing.captured_events) >= 1
        # At least one of the event lists should contain the user_input event
        found_user_input = False
        for event_list in capturing.captured_events:
            for ev in event_list:
                if ev.name == "user_input" and ev.data is not None:
                    assert ev.data == user_input
                    found_user_input = True
        assert found_user_input, "Resolver should receive user_input event with data"

    @pytest.mark.asyncio
    async def test_multiple_layers_compile_in_declaration_order(self):
        """FR18: Multiple system layers compile in declaration order."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver_a = StubResolver(
            name="persona",
            content=[TextBlock(text="You are EDI.")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        resolver_b = StubResolver(
            name="rules",
            content=[TextBlock(text="Follow these rules.")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer_a = make_system_layer(
            name="system_persona",
            resolvers=[resolver_a],
            token_counter=counter,
            event_bus=bus,
        )
        layer_b = make_system_layer(
            name="system_rules",
            resolvers=[resolver_b],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer_a, layer_b], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request.system is not None
        texts = [b.text for b in result.request.system]
        # layer_a declared first -> its content comes first
        assert texts.index("You are EDI.") < texts.index("Follow these rules.")


# ---------------------------------------------------------------------------
# 4. Pipeline.run() — event flow
# ---------------------------------------------------------------------------


class TestPipelineRunEventFlow:
    @pytest.mark.asyncio
    async def test_turn_start_event_is_emitted(self):
        """FR4: turn_start event is emitted at the start of run."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # Use a resolver subscribed to turn_start to prove it fires
        resolver = StubResolver(
            name="start_watcher",
            content=[TextBlock(text="started")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        await run_engine(engine, [])

        # If resolver fired, turn_start was emitted
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_user_input_event_carries_input_blocks(self):
        """FR4: user_input event carries the user's input blocks as data."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        capturing = CapturingResolver(
            name="input_reader",
            subscriptions=[EventSubscription(event_name="user_input", phase=EventPhase.COMPLETED)],
        )

        layer = make_messages_layer(
            name="conversation",
            resolvers=[capturing],
            token_counter=counter,
            event_bus=bus,
        )

        user_blocks = [TextBlock(text="What time is it?")]
        engine = PipelineEngine(layers=[layer], token_counter=counter)
        await run_engine(engine, user_blocks)

        assert len(capturing.captured_events) >= 1
        user_input_events = [
            ev
            for event_list in capturing.captured_events
            for ev in event_list
            if ev.name == "user_input"
        ]
        assert len(user_input_events) >= 1
        assert user_input_events[0].data == user_blocks

    @pytest.mark.asyncio
    async def test_turn_end_event_is_emitted(self):
        """FR16: turn_end event is emitted after all layers are done."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # Transformer subscribed to turn_end proves it fires
        turn_end_transformer = StubTransformer(
            name="turn_end_watcher",
            subscriptions=[EventSubscription(event_name="turn_end")],
        )

        layer = make_system_layer(
            name="system_prompt",
            transformers=[turn_end_transformer],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        await run_engine(engine, [])

        # If transformer fired, turn_end was emitted
        assert turn_end_transformer.execution_count == 1


# ---------------------------------------------------------------------------
# 5. Pipeline.run() — transformers
# ---------------------------------------------------------------------------


class TestPipelineRunTransformers:
    @pytest.mark.asyncio
    async def test_transformer_fires_and_transforms_content(self):
        """FR6/FR8: Transformer subscribed to turn_end fires and transforms layer content."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="original content")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        # Transformer replaces content
        transformer = StubTransformer(
            name="replacer",
            subscriptions=[EventSubscription(event_name="turn_end")],
            transform_fn=lambda content: [TextBlock(text="transformed content")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            transformers=[transformer],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request.system is not None
        texts = [b.text for b in result.request.system]
        assert "transformed content" in texts

    @pytest.mark.asyncio
    async def test_transformer_emits_events_downstream(self):
        """Transformer that emits events -> downstream subscribers see them."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="base")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        # Transformer emits a custom event
        emitting_transformer = StubTransformer(
            name="emitter",
            subscriptions=[EventSubscription(event_name="turn_end")],
            events_to_emit=[
                Event(
                    name="custom_event",
                    phase=EventPhase.COMPLETED,
                    source_layer="system_prompt",
                )
            ],
        )

        # Second resolver subscribes to the custom event
        downstream_resolver = CapturingResolver(
            name="downstream",
            subscriptions=[EventSubscription(event_name="custom_event", phase=EventPhase.COMPLETED)],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver, downstream_resolver],
            transformers=[emitting_transformer],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        await run_engine(engine, [])

        # The downstream resolver should have been triggered by the custom event
        assert downstream_resolver.execution_count >= 1
        custom_events = [
            ev
            for event_list in downstream_resolver.captured_events
            for ev in event_list
            if ev.name == "custom_event"
        ]
        assert len(custom_events) >= 1


# ---------------------------------------------------------------------------
# 6. Pipeline.run() — budget enforcement
# ---------------------------------------------------------------------------


class TestPipelineRunBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_over_budget_force_truncates_and_warns(self):
        """FR10/FR11: Layer over budget gets force-truncated; warning appears in metrics."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # Resolver produces 10 tokens of content (40 chars)
        resolver = StubResolver(
            name="big_resolver",
            content=make_text_blocks_with_tokens(10, "x"),
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_budget=3,  # budget is only 3 tokens — content is 10
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        # Metrics should show force truncation
        layer_metrics = result.metrics.layers["system_prompt"]
        assert layer_metrics.force_truncated is True

        # Warnings should contain a truncation warning
        assert len(result.metrics.warnings) >= 1
        assert any("system_prompt" in w for w in result.metrics.warnings)

    @pytest.mark.asyncio
    async def test_under_budget_no_truncation_no_warning(self):
        """Layer under budget -> no truncation, no warning."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # Resolver produces 2 tokens (8 chars)
        resolver = StubResolver(
            name="small_resolver",
            content=make_text_blocks_with_tokens(2, "y"),
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_budget=100,  # plenty of budget
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        layer_metrics = result.metrics.layers["system_prompt"]
        assert layer_metrics.force_truncated is False

        # No truncation warnings
        truncation_warnings = [w for w in result.metrics.warnings if "system_prompt" in w]
        assert truncation_warnings == []


# ---------------------------------------------------------------------------
# 7. Pipeline.run() — compilation
# ---------------------------------------------------------------------------


class TestPipelineRunCompilation:
    @pytest.mark.asyncio
    async def test_system_layers_compile_to_request_system(self):
        """FR18: System layers compile to CompletionRequest.system."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="System instructions here.")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request.system is not None
        assert len(result.request.system) >= 1
        assert any(b.text == "System instructions here." for b in result.request.system)

    @pytest.mark.asyncio
    async def test_messages_layers_compile_to_request_messages(self):
        """FR18: Messages layers compile to CompletionRequest.messages."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="session",
            content=[TextBlock(text="User said hello")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_messages_layer(
            name="conversation",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request.messages is not None
        assert len(result.request.messages) >= 1
        assert isinstance(result.request.messages[0], Message)

    @pytest.mark.asyncio
    async def test_multiple_system_layers_merge_via_position(self):
        """FR18: Multiple system layers merge in declaration order via their position strategies."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver_a = StubResolver(
            name="persona",
            content=[TextBlock(text="Persona block")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        resolver_b = StubResolver(
            name="rules",
            content=[TextBlock(text="Rules block")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer_a = make_system_layer(
            name="system_persona",
            resolvers=[resolver_a],
            token_counter=counter,
            event_bus=bus,
        )
        layer_b = make_system_layer(
            name="system_rules",
            resolvers=[resolver_b],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer_a, layer_b], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request.system is not None
        texts = [b.text for b in result.request.system]
        assert "Persona block" in texts
        assert "Rules block" in texts
        # Declaration order preserved
        assert texts.index("Persona block") < texts.index("Rules block")

    @pytest.mark.asyncio
    async def test_all_targets_populated(self):
        """FR18/FR19: When system + messages + tools layers exist, all three fields populated."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.layer import Layer

        counter = CharacterTokenCounter()
        bus = EventBus()

        sys_resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="System text")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        msg_resolver = StubResolver(
            name="msg",
            content=[TextBlock(text="User message")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        system_layer = make_system_layer(
            name="system_prompt",
            resolvers=[sys_resolver],
            token_counter=counter,
            event_bus=bus,
        )
        messages_layer = make_messages_layer(
            name="conversation",
            resolvers=[msg_resolver],
            token_counter=counter,
            event_bus=bus,
        )
        tools_layer = Layer(
            name="tools",
            target=CompilationTarget.TOOLS,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=counter,
            event_bus=bus,
        )
        # Add tool definitions directly to the tools layer
        tools_layer.add_tool_definitions([
            ToolDefinition(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ])

        engine = PipelineEngine(
            layers=[system_layer, messages_layer, tools_layer],
            token_counter=counter,
        )
        result = await run_engine(engine, [])

        assert result.request.system is not None
        assert len(result.request.system) >= 1

        assert result.request.messages is not None
        assert len(result.request.messages) >= 1

        assert result.request.tools is not None
        assert len(result.request.tools) >= 1
        assert result.request.tools[0].name == "search"


# ---------------------------------------------------------------------------
# 8. Pipeline.run() — metrics
# ---------------------------------------------------------------------------


class TestPipelineRunMetrics:
    @pytest.mark.asyncio
    async def test_metrics_has_entry_for_each_layer(self):
        """FR19: PipelineResult.metrics.layers has entry for each layer."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        layer_a = make_system_layer(name="system_prompt", token_counter=counter, event_bus=bus)
        layer_b = make_messages_layer(name="conversation", token_counter=counter, event_bus=bus)

        engine = PipelineEngine(layers=[layer_a, layer_b], token_counter=counter)
        result = await run_engine(engine, [])

        assert "system_prompt" in result.metrics.layers
        assert "conversation" in result.metrics.layers

    @pytest.mark.asyncio
    async def test_layer_metrics_tokens_used_reflects_actual_count(self):
        """LayerMetrics.tokens_used reflects the actual token count of the layer's content."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # "abcdefgh" = 8 chars = 2 tokens
        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="abcdefgh")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        lm = result.metrics.layers["system_prompt"]
        assert lm.tokens_used == 2

    @pytest.mark.asyncio
    async def test_resolver_execution_counts_tracked(self):
        """LayerMetrics.resolver_executions tracks per-resolver execution counts."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="data")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        lm = result.metrics.layers["system_prompt"]
        assert "sys_prompt" in lm.resolver_executions
        assert lm.resolver_executions["sys_prompt"] == 1

    @pytest.mark.asyncio
    async def test_transformer_execution_counts_tracked(self):
        """LayerMetrics.transformer_executions tracks per-transformer execution counts."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        transformer = StubTransformer(
            name="compactor",
            subscriptions=[EventSubscription(event_name="turn_end")],
        )

        layer = make_system_layer(
            name="system_prompt",
            transformers=[transformer],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        lm = result.metrics.layers["system_prompt"]
        assert "compactor" in lm.transformer_executions
        assert lm.transformer_executions["compactor"] == 1

    @pytest.mark.asyncio
    async def test_force_truncated_flag_true_when_truncated(self):
        """LayerMetrics.force_truncated is True when force truncation occurred."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="big",
            content=make_text_blocks_with_tokens(20, "z"),  # 20 tokens
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_budget=3,  # only 3 tokens allowed
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.metrics.layers["system_prompt"].force_truncated is True

    @pytest.mark.asyncio
    async def test_total_tokens_sums_all_layers(self):
        """PipelineMetrics.total_tokens sums all layers."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # Layer A: "abcdefgh" = 8 chars = 2 tokens
        resolver_a = StubResolver(
            name="r_a",
            content=[TextBlock(text="abcdefgh")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        # Layer B: "wxyzwxyz" = 8 chars = 2 tokens
        resolver_b = StubResolver(
            name="r_b",
            content=[TextBlock(text="wxyzwxyz")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer_a = make_system_layer(
            name="system_a",
            resolvers=[resolver_a],
            token_counter=counter,
            event_bus=bus,
        )
        layer_b = make_system_layer(
            name="system_b",
            resolvers=[resolver_b],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer_a, layer_b], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.metrics.total_tokens == 4  # 2 + 2

    @pytest.mark.asyncio
    async def test_warnings_include_force_truncation_warnings(self):
        """PipelineMetrics.warnings includes force truncation warnings."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="big",
            content=make_text_blocks_with_tokens(20, "w"),
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="oversized_layer",
            resolvers=[resolver],
            token_budget=2,
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        assert len(result.metrics.warnings) >= 1
        assert any("oversized_layer" in w for w in result.metrics.warnings)


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


class TestPipelineRunEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_user_input(self):
        """Pipeline completes with empty user_input (empty list of ContentBlocks)."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="System.")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])  # empty input

        assert result.request is not None
        assert result.request.system is not None
        assert len(result.request.system) >= 1

    @pytest.mark.asyncio
    async def test_resolver_subscribed_to_event_that_never_fires(self):
        """Resolver subscribed to an event that never fires -> layer is still done, pipeline completes."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # This resolver subscribes to "some_custom_event" which is never emitted
        idle_resolver = StubResolver(
            name="idle",
            content=[TextBlock(text="Should not appear")],
            subscriptions=[
                EventSubscription(event_name="some_custom_event", phase=EventPhase.COMPLETED)
            ],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[idle_resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        # Pipeline should complete successfully
        assert result.request is not None
        # The idle resolver should not have fired
        assert idle_resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_max_executions_enforced(self):
        """FR5: Resolver fires only once even if event arrives multiple times
        (max_executions=1 enforced by engine)."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # Resolver with max_executions=1 subscribed to turn_start
        resolver = StubResolver(
            name="once_only",
            content=[TextBlock(text="once")],
            subscriptions=[EventSubscription(event_name="turn_start")],
            max_executions=1,
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        await run_engine(engine, [])

        # Resolver should have fired exactly once even though max_executions enforcement
        # is the engine's responsibility
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_empty_layers_produces_valid_result(self):
        """Pipeline with no layers produces a valid PipelineResult with empty request."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        engine = PipelineEngine(layers=[], token_counter=counter)
        result = await run_engine(engine, [])

        assert result.request is not None
        assert result.metrics is not None
        assert result.metrics.total_tokens == 0
        assert result.metrics.layers == {}
        assert result.metrics.warnings == []

    @pytest.mark.asyncio
    async def test_layer_metrics_budget_remaining_calculated(self):
        """LayerMetrics.budget_remaining = token_budget - tokens_used (when budget exists)."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        # "abcdefgh" = 8 chars = 2 tokens
        resolver = StubResolver(
            name="small",
            content=[TextBlock(text="abcdefgh")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_budget=10,  # budget of 10, using 2
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        lm = result.metrics.layers["system_prompt"]
        assert lm.token_budget == 10
        assert lm.tokens_used == 2
        assert lm.budget_remaining == 8  # 10 - 2

    @pytest.mark.asyncio
    async def test_layer_metrics_no_budget_remaining_is_none(self):
        """LayerMetrics.budget_remaining is None when layer has no budget."""
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()
        bus = EventBus()

        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="data")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_budget=None,  # no budget
            token_counter=counter,
            event_bus=bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=counter)
        result = await run_engine(engine, [])

        lm = result.metrics.layers["system_prompt"]
        assert lm.token_budget is None
        assert lm.budget_remaining is None
