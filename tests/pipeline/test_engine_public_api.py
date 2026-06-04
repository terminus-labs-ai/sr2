"""Tests for sr2-71: Engine public turn API.

Verifies the orchestrator no longer reaches into engine privates and that
the new public API surface is correct.

Covers:
  - engine.turn_seq property returns current turn sequence number
  - engine.compile() returns a CompletionRequest
  - engine.run_loop() is a public method (drains bus until quiescent)
  - layer.begin_turn() replaces engine reaching into layer._turn_seq etc.
  - orchestrator.turn() uses public engine API (no _turn_seq, _compile_request, _run_loop)
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubResolver:
    """Minimal resolver stub."""

    def __init__(
        self,
        name: str = "stub",
        content: list[TextBlock] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._content = content or [TextBlock(text="stub")]
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(resolver_name=self.name, source_layer="test", content=self._content)


def _make_layer(name: str = "test_layer") -> Layer:
    return Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
    )


# ---------------------------------------------------------------------------
# 1. engine.turn_seq property
# ---------------------------------------------------------------------------


class TestEngineTurnSeqProperty:
    def test_turn_seq_starts_at_negative_one(self):
        """Engine._turn_seq starts at -1; turn_seq property exposes it."""
        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
        assert engine.turn_seq == -1

    def test_turn_seq_updates_after_start_turn(self):
        """After start_turn(n), turn_seq returns n."""

        async def _run():
            engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
            await engine.start_turn(turn_seq=5)
            assert engine.turn_seq == 5

        import asyncio
        asyncio.get_event_loop().run_until_complete(_run())

    @pytest.mark.asyncio
    async def test_turn_seq_increments_across_turns(self):
        """turn_seq reflects sequential turn numbers."""
        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
        await engine.start_turn(turn_seq=1)
        assert engine.turn_seq == 1
        await engine.start_turn(turn_seq=2)
        assert engine.turn_seq == 2

    def test_turn_seq_is_read_only(self):
        """turn_seq is a property without a setter."""
        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
        # Accessing via property is fine; assigning should raise AttributeError
        with pytest.raises(AttributeError):
            engine.turn_seq = 42  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. engine.compile() public method
# ---------------------------------------------------------------------------


class TestEngineCompile:
    def test_compile_returns_completion_request(self):
        """engine.compile() returns a CompletionRequest."""
        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
        result = engine.compile()
        assert isinstance(result, CompletionRequest)

    def test_compile_with_content(self):
        """compile() includes content from layers."""
        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="Hello")],
        )
        layer = Layer(
            name="system",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        # Pre-populate content
        layer.set_content([TextBlock(text="Hello")])

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        request = engine.compile()

        assert request.system is not None
        assert any(b.text == "Hello" for b in request.system)

    def test_compile_equivalent_to_internal_method(self):
        """engine.compile() delegates to _compile_request() — same result."""
        layer = _make_layer()
        layer.set_content([TextBlock(text="test")])
        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())

        public_result = engine.compile()
        private_result = engine._compile_request()

        assert public_result.system == private_result.system
        assert public_result.messages == private_result.messages
        assert public_result.tools == private_result.tools


# ---------------------------------------------------------------------------
# 3. engine.run_loop() public method
# ---------------------------------------------------------------------------


class TestEngineRunLoop:
    @pytest.mark.asyncio
    async def test_run_loop_is_public_method(self):
        """run_loop() is a public method on PipelineEngine (no leading underscore)."""
        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
        assert hasattr(engine, "run_loop")
        assert callable(engine.run_loop)
        # Verify it's not a private method
        assert not engine.run_loop.__name__.startswith("_")

    @pytest.mark.asyncio
    async def test_run_loop_drains_empty_bus(self):
        """run_loop() on a quiescent engine returns immediately."""
        engine = PipelineEngine(layers=[], token_counter=CharacterTokenCounter())
        # Should not hang or raise
        await engine.run_loop()

    @pytest.mark.asyncio
    async def test_run_loop_processes_events(self):
        """run_loop() drains queued events and processes layers."""
        resolver = StubResolver(
            name="user_input_resolver",
            content=[TextBlock(text="captured")],
            subscriptions=[EventSubscription(event_name="user_input", phase=EventPhase.COMPLETED)],
        )
        layer = Layer(
            name="messages",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        layer.begin_turn(turn_seq=1, next_firing_seq_fn=engine._next_firing_seq)

        # Queue event on engine's bus (engine wires layers to its own bus)
        engine.bus.queue(
            Event(
                name="user_input",
                phase=EventPhase.COMPLETED,
                source_layer="engine",
                data=[TextBlock(text="hi")],
            )
        )
        await engine.run_loop()

        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 4. layer.begin_turn()
# ---------------------------------------------------------------------------


class TestLayerBeginTurn:
    def test_begin_turn_sets_turn_seq(self):
        """begin_turn() sets the layer's turn_seq."""
        layer = _make_layer()
        layer.begin_turn(turn_seq=42, next_firing_seq_fn=lambda: 0)
        assert layer._turn_seq == 42

    def test_begin_turn_clears_content(self):
        """begin_turn() resets content to empty list."""
        layer = _make_layer()
        layer.set_content([TextBlock(text="old content")])
        layer.begin_turn(turn_seq=1, next_firing_seq_fn=lambda: 0)
        assert layer.get_content() == []

    def test_begin_turn_clears_pending_events(self):
        """begin_turn() clears pending events."""
        layer = _make_layer()
        layer._pending_events = [
            Event(name="fake", phase=EventPhase.COMPLETED, source_layer="test")
        ]
        layer.begin_turn(turn_seq=1, next_firing_seq_fn=lambda: 0)
        assert layer._pending_events == []

    def test_begin_turn_resets_tools(self):
        """begin_turn() resets tool definitions when tool_providers exist."""
        from sr2.models import ToolDefinition

        layer = _make_layer()
        layer.tool_providers = [object()]  # dummy provider
        layer.add_tool_definitions(
            [ToolDefinition(name="old_tool", description="old", input_schema={})]
        )
        layer.begin_turn(turn_seq=1, next_firing_seq_fn=lambda: 0)
        assert layer.get_tool_definitions() == []

    def test_begin_turn_sets_firing_seq_fn(self):
        """begin_turn() sets the layer's next_firing_seq callable."""
        layer = _make_layer()
        counter = [0]
        def seq_fn():
            counter[0] += 1
            return counter[0]
        layer.begin_turn(turn_seq=1, next_firing_seq_fn=seq_fn)
        assert layer._next_firing_seq() == 1
        assert layer._next_firing_seq() == 2


# ---------------------------------------------------------------------------
# 5. Orchestrator uses public API (static analysis)
# ---------------------------------------------------------------------------


class TestOrchestratorNoUnderscoreReachIns:
    """Verify orchestrator.turn() uses public engine API.

    These tests parse the orchestrator source to verify no underscore
    reach-ins across the orchestrator→engine→layer boundary.
    """

    @pytest.fixture
    def orchestrator_source(self) -> str:
        """Read the orchestrator source file."""
        import sr2.orchestrator as mod

        return inspect.getsource(mod)

    def test_no_engine_turn_seq_access(self, orchestrator_source: str):
        """Orchestrator should not access _engine._turn_seq."""
        assert "_engine._turn_seq" not in orchestrator_source, (
            "Orchestrator must use engine.turn_seq property, not _engine._turn_seq"
        )

    def test_no_engine_compile_request_access(self, orchestrator_source: str):
        """Orchestrator should not access _engine._compile_request()."""
        assert "_engine._compile_request" not in orchestrator_source, (
            "Orchestrator must use engine.compile(), not _engine._compile_request()"
        )

    def test_no_engine_run_loop_access(self, orchestrator_source: str):
        """Orchestrator should not access _engine._run_loop()."""
        assert "_engine._run_loop" not in orchestrator_source, (
            "Orchestrator must use engine.run_loop(), not _engine._run_loop()"
        )

    def test_no_layer_private_access_from_engine(self):
        """Engine should not access layer._turn_seq, layer._next_firing_seq, layer._pending_events directly.

        Instead it should use layer.begin_turn().
        """
        import sr2.pipeline.engine as mod
        source = inspect.getsource(mod)

        # Inside start_turn(), the engine should not mutate layer privates directly
        assert "layer._turn_seq" not in source, (
            "Engine must use layer.begin_turn(), not layer._turn_seq = ..."
        )
        assert "layer._next_firing_seq" not in source, (
            "Engine must use layer.begin_turn(), not layer._next_firing_seq = ..."
        )
        assert "layer._pending_events" not in source, (
            "Engine must use layer.begin_turn(), not layer._pending_events = ..."
        )

    def test_engine_uses_layer_begin_turn(self):
        """Engine.start_turn() should call layer.begin_turn()."""
        import sr2.pipeline.engine as mod
        source = inspect.getsource(mod)

        assert "begin_turn(" in source, (
            "Engine.start_turn() must call layer.begin_turn()"
        )


# ---------------------------------------------------------------------------
# 6. conftest run_engine uses public API
# ---------------------------------------------------------------------------


class TestConftestRunEnginePublicApi:
    """Verify conftest.run_engine() uses public engine API."""

    def test_run_engine_does_not_access_turn_seq_private(self):
        """conftest.run_engine should not access engine._turn_seq."""
        import pathlib

        conftest_path = pathlib.Path(__file__).parent.parent / "conftest.py"
        source = conftest_path.read_text()

        assert "engine._turn_seq" not in source, (
            "run_engine() must use engine.turn_seq property, not engine._turn_seq"
        )
