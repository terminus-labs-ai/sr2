"""Tests for tracer injection plumbing and turn_seq / firing_seq coordination.

Covers:
  FR9:  Tracer optional param on Layer, PipelineEngine, SR2 — default None, forwarded
  FR10: Zero-cost seam — tracer=None accepted without error
  FR8:  Engine _turn_seq and firing_seq counters — monotonic, reset per turn
"""

from __future__ import annotations

import pytest

from conftest import run_engine

from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer, Tracer


# ---------------------------------------------------------------------------
# Minimal construction helpers
# ---------------------------------------------------------------------------


def _make_layer(name: str = "test_layer", tracer=None) -> Layer:
    """Build a minimal system Layer, optionally with a tracer."""
    kwargs = dict(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
    )
    if tracer is not None:
        kwargs["tracer"] = tracer
    return Layer(**kwargs)


def _make_engine(layers=None, tracer=None):
    from sr2.pipeline.engine import PipelineEngine

    kwargs = dict(
        layers=layers if layers is not None else [],
        token_counter=CharacterTokenCounter(),
    )
    if tracer is not None:
        kwargs["tracer"] = tracer
    return PipelineEngine(**kwargs)


def _make_sr2(tracer=None):
    """Build a minimal SR2 instance with a stub LLM, optionally with a tracer."""
    from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig
    from sr2.orchestrator import SR2

    class _StubLLM:
        async def complete(self, request):  # pragma: no cover
            ...

        async def stream(self, request):  # pragma: no cover
            return
            yield

    pipeline_config = PipelineConfig(
        layers=[
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[ResolverConfig(type="static", config={"text": "hi"})],
            )
        ]
    )
    kwargs = dict(
        pipeline_config=pipeline_config,
        llm={"default": _StubLLM()},
        token_counter=CharacterTokenCounter(),
    )
    if tracer is not None:
        kwargs["tracer"] = tracer
    return SR2(**kwargs)


# ---------------------------------------------------------------------------
# FR9 / FR10 — Layer tracer injection
# ---------------------------------------------------------------------------


def test_layer_accepts_tracer_none_default():
    """Layer constructed without tracer kwarg: _tracer is None, no crash."""
    layer = _make_layer()
    assert layer._tracer is None


def test_layer_accepts_tracer_none_explicit():
    """Layer constructed with tracer=None explicitly: _tracer is None, no crash."""
    layer = Layer(
        name="test",
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
        tracer=None,
    )
    assert layer._tracer is None


def test_layer_accepts_collecting_tracer():
    """Layer constructed with a CollectingTracer: _tracer is the passed instance."""
    tracer = CollectingTracer()
    layer = Layer(
        name="test",
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
        tracer=tracer,
    )
    assert layer._tracer is tracer


def test_layer_tracer_satisfies_protocol():
    """CollectingTracer satisfies the Tracer protocol (isinstance check)."""
    tracer = CollectingTracer()
    assert isinstance(tracer, Tracer)


# ---------------------------------------------------------------------------
# FR9 / FR10 — PipelineEngine tracer injection
# ---------------------------------------------------------------------------


def test_engine_accepts_tracer_none_default():
    """PipelineEngine constructed without tracer kwarg: _tracer is None, no crash."""
    engine = _make_engine()
    assert engine._tracer is None


def test_engine_accepts_tracer_none_explicit():
    """PipelineEngine constructed with tracer=None explicitly: _tracer is None, no crash."""
    from sr2.pipeline.engine import PipelineEngine

    engine = PipelineEngine(
        layers=[],
        token_counter=CharacterTokenCounter(),
        tracer=None,
    )
    assert engine._tracer is None


def test_engine_with_tracer_threads_to_layers():
    """PipelineEngine with CollectingTracer: every layer's _tracer is the same instance."""
    tracer = CollectingTracer()
    layers = [_make_layer("layer_a"), _make_layer("layer_b"), _make_layer("layer_c")]

    from sr2.pipeline.engine import PipelineEngine

    engine = PipelineEngine(
        layers=layers,
        token_counter=CharacterTokenCounter(),
        tracer=tracer,
    )

    for layer in engine._layers:
        assert layer._tracer is tracer, (
            f"Layer '{layer.name}' has _tracer={layer._tracer!r}, expected the injected tracer"
        )


def test_engine_with_tracer_none_layers_have_none():
    """PipelineEngine with tracer=None: no tracer propagated to layers."""
    layers = [_make_layer("layer_a"), _make_layer("layer_b")]

    from sr2.pipeline.engine import PipelineEngine

    engine = PipelineEngine(
        layers=layers,
        token_counter=CharacterTokenCounter(),
        tracer=None,
    )

    for layer in engine._layers:
        assert layer._tracer is None


# ---------------------------------------------------------------------------
# FR9 / FR10 — SR2 tracer injection
# ---------------------------------------------------------------------------


def test_sr2_accepts_tracer_none_default():
    """SR2 constructed without tracer kwarg: _tracer is None, no crash."""
    sr2 = _make_sr2()
    assert sr2._tracer is None


def test_sr2_accepts_tracer_none_explicit():
    """SR2 constructed with tracer=None explicitly: _tracer is None, no crash."""
    sr2 = _make_sr2(tracer=None)
    assert sr2._tracer is None


@pytest.mark.asyncio
async def test_sr2_accepts_collecting_tracer():
    """SR2 constructed with a CollectingTracer: tracer receives firing records during a turn."""
    tracer = CollectingTracer()
    sr2 = _make_sr2(tracer=tracer)
    async for _ in sr2.turn([]):
        pass
    assert tracer.get_trace(), (
        "Expected CollectingTracer to have received at least one FiringRecord — "
        "tracer was not wired to the engine layers"
    )


@pytest.mark.asyncio
async def test_sr2_threads_tracer_to_engine_layers():
    """SR2 with CollectingTracer: tracer receives firings from all engine layers during a turn."""
    tracer = CollectingTracer()
    sr2 = _make_sr2(tracer=tracer)
    assert sr2._engine.layers, "Expected SR2 engine to have at least one layer"
    async for _ in sr2.turn([]):
        pass
    # All layers must funnel through the same tracer — verify by observing its records
    assert tracer.get_trace(), (
        "Expected CollectingTracer to have captured FiringRecords after turn() — "
        "tracer was not threaded to the engine layers"
    )


# ---------------------------------------------------------------------------
# FR8 — turn_seq counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_turn_seq_starts_before_zero():
    """Engine _turn_seq starts at a sentinel < 0 before any run() call."""
    engine = _make_engine()
    assert engine._turn_seq < 0


@pytest.mark.asyncio
async def test_engine_turn_seq_increments_to_zero_on_first_run():
    """Engine _turn_seq is 0 after the first run() call."""
    engine = _make_engine()
    await run_engine(engine, [])
    assert engine._turn_seq == 0


@pytest.mark.asyncio
async def test_engine_turn_seq_increments_to_one_on_second_run():
    """Engine _turn_seq is 1 after the second run() call."""
    engine = _make_engine()
    await run_engine(engine, [])
    await run_engine(engine, [])
    assert engine._turn_seq == 1


@pytest.mark.asyncio
async def test_engine_turn_seq_increments_monotonically():
    """Engine _turn_seq increments by exactly 1 on each run() call."""
    engine = _make_engine()
    for expected in range(3):
        await run_engine(engine, [])
        assert engine._turn_seq == expected


# ---------------------------------------------------------------------------
# FR8 — firing_seq counter
# ---------------------------------------------------------------------------


def test_engine_next_firing_seq_is_monotonic():
    """Calling _next_firing_seq() twice yields a strictly increasing sequence."""
    engine = _make_engine()
    first = engine._next_firing_seq()
    second = engine._next_firing_seq()
    assert second > first


@pytest.mark.asyncio
async def test_engine_next_firing_seq_is_callable():
    """Engine exposes a _next_firing_seq() callable (method or function)."""
    engine = _make_engine()
    assert callable(engine._next_firing_seq)
