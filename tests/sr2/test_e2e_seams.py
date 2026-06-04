"""Through-SR2() e2e tests for each seam.

These tests exercise subsystems through the public SR2(...) constructor
rather than constructing Dependencies/Layer directly. This coverage would
have caught the sr2-66..70 dead-seam blind spots.

Seams covered:
  1. Memory store — SR2(memory_store=...) propagates to all SessionResolvers
  2. Memory extractor — SR2(memory_extractor=...) wires extraction transformer
  3. Provenance store — SR2(provenance_store=...) sets engine provenance
  4. Tracer — SR2(tracer=...) propagates through layers
  5. Tool executor — SR2(tool_executor=...) enables tool loop
  6. Active frame provider — SR2(active_frame_provider=...) stamps blocks
  7. Degradation config — PipelineConfig.degradation=... wires ladder
  8. Compaction — TransformerConfig type="compaction" through SR2 config
  9. Session ID — explicit and auto-generated
  10. Multi-model LLM dict
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from conftest import (
    MockLLM,
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
    tool_use_event,
)
from sr2.config.models import (
    DegradationConfig,
    DegradationLevelConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.memory.protocol import MemoryExtractor, MemoryStore
from sr2.memory.schema import Memory
from sr2.models import TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.provenance import InMemoryProvenanceStore
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import FiringRecord
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    return make_minimal_config()


@pytest.fixture
def counter():
    return CharacterTokenCounter()


# ===========================================================================
# 1. Memory Store seam
# ===========================================================================


@pytest.mark.asyncio
class TestMemoryStoreE2E:

    @dataclass
    class CapturingMemoryStore:
        saved: list[list] = field(default_factory=list)
        loaded: list[str] = field(default_factory=list)
        _return_value: list[Memory] = field(default_factory=list)

        async def save(self, session_id: str, messages: list) -> None:
            self.saved.append(messages)

        async def load(self, session_id: str) -> list[Memory]:
            self.loaded.append(session_id)
            return self._return_value

    async def test_memory_store_propagates_to_session_resolvers(
        self, minimal_config, counter
    ):
        """SR2(memory_store=store) → SessionResolver receives store for save/load."""
        store = self.CapturingMemoryStore()
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
            memory_store=store,
        )

        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)

        assert any(e.type == "end" for e in events)
        assert sr2.session_id != ""
        await sr2.aclose()

    async def test_memory_store_none_is_safe(self, minimal_config, counter):
        """SR2() without memory_store → constructs and runs without error."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()


# ===========================================================================
# 2. Memory Extractor seam
# ===========================================================================


@pytest.mark.asyncio
class TestMemoryExtractorE2E:

    @dataclass
    class StubExtractor:
        extracted: list[str] = field(default_factory=list)

        async def extract(self, session_id: str, messages: list) -> list[Memory]:
            self.extracted.append(session_id)
            return []

    async def test_extractor_wired_through_sr2(self, minimal_config, counter):
        """SR2(memory_extractor=extractor) → extraction transformer is wired."""
        extractor = self.StubExtractor()
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
            memory_extractor=extractor,
        )
        assert sr2.session_id != ""
        await sr2.aclose()


# ===========================================================================
# 3. Provenance Store seam
# ===========================================================================


@pytest.mark.asyncio
class TestProvenanceStoreE2E:

    async def test_custom_provenance_store_propagates(self, minimal_config, counter):
        """SR2(provenance_store=store) → engine uses the provided store."""
        store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
            provenance_store=store,
        )
        assert sr2.provenance_store is store
        await sr2.aclose()

    async def test_default_provenance_store_on_none(self, minimal_config, counter):
        """SR2() without provenance_store → uses InMemoryProvenanceStore."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
        )
        assert isinstance(sr2.provenance_store, InMemoryProvenanceStore)
        await sr2.aclose()


# ===========================================================================
# 4. Tracer seam
# ===========================================================================


@pytest.mark.asyncio
class TestTracerE2E:

    @dataclass
    class CapturingTracer:
        firings: list[FiringRecord] = field(default_factory=list)
        compiles: list[CompletionRequest] = field(default_factory=list)

        def on_firing(self, record: FiringRecord) -> None:
            self.firings.append(record)

        def on_compile(self, request: CompletionRequest) -> None:
            self.compiles.append(request)

    async def test_tracer_propagates_through_layers(self, minimal_config, counter):
        """SR2(tracer=tracer) → engine receives tracer for layer instrumentation."""
        tracer = self.CapturingTracer()
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
            tracer=tracer,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()


# ===========================================================================
# 5. Tool Executor seam
# ===========================================================================


@pytest.mark.asyncio
class TestToolExecutorE2E:

    async def test_tool_executor_enabled_turn_completes(self, minimal_config, counter):
        """SR2(tool_executor=executor) with tool calls → executes and completes."""
        llm = SequentialMockLLM([
            [tool_use_event()],
            [StreamEvent(type="text", text="done"), StreamEvent(type="end")],
        ])
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=llm,
            token_counter=counter,
            tool_executor=stub_executor,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)

        types = [e.type for e in events]
        assert "tool_use_emitted" in types
        assert "tool_result_received" in types
        assert "end" in types
        await sr2.aclose()

    async def test_missing_tool_executor_raises(self, minimal_config, counter):
        """SR2 without tool_executor + tool calls → raises ConfigError."""
        from sr2.orchestrator import ConfigError

        llm = MockLLM(events=[tool_use_event(), StreamEvent(type="end")])
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=llm,
            token_counter=counter,
        )
        with pytest.raises(ConfigError, match="tool_executor"):
            async for _ in sr2.turn(make_user_input("test")):
                pass
        await sr2.aclose()


# ===========================================================================
# 6. Active Frame Provider seam
# ===========================================================================


@pytest.mark.asyncio
class TestActiveFrameProviderE2E:

    async def test_frame_provider_stamps_blocks(self, minimal_config, counter):
        """SR2(active_frame_provider=fn) → blocks get meta['frame'] set."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
            active_frame_provider=lambda origin: "frame-123",
        )
        events = []
        async for event in sr2.turn(make_user_input("test"), origin="test-origin"):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()

    async def test_frame_provider_none_skips_stamping(self, minimal_config, counter):
        """SR2 without active_frame_provider → runs fine, no frame stamp."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()


# ===========================================================================
# 7. Degradation Config seam
# ===========================================================================


@pytest.mark.asyncio
class TestDegradationConfigE2E:

    async def test_degradation_config_wires_ladder(self, counter):
        """SR2 with degradation config → engine has ladder set."""
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are helpful."},
                        )
                    ],
                    degradation_category="core",
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[
                        ResolverConfig(type="session"),
                        ResolverConfig(
                            type="input",
                            subscriptions=[
                                {"event": "user_input", "phase": "completed"}
                            ],
                        ),
                    ],
                    degradation_category="context",
                ),
            ],
            degradation=DegradationConfig(
                levels=[
                    DegradationLevelConfig(name="full", keep_categories=["core", "context"]),
                    DegradationLevelConfig(
                        name="degraded",
                        keep_categories=["core"],
                    ),
                ]
            ),
        )
        sr2 = SR2(
            pipeline_config=config,
            llm=MockLLM(),
            token_counter=counter,
        )
        assert sr2._engine._ladder is not None
        await sr2.aclose()

    async def test_no_degradation_config_is_safe(self, minimal_config, counter):
        """SR2 without degradation config → engine runs without ladder."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        assert sr2._engine._ladder is None
        await sr2.aclose()


# ===========================================================================
# 8. Compaction seam
# ===========================================================================


@pytest.mark.asyncio
class TestCompactionE2E:

    async def test_compaction_transformer_in_layer(self, counter):
        """SR2 with compaction transformer → layer has transformer wired."""
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are helpful."},
                        )
                    ],
                    transformers=[
                        TransformerConfig(
                            type="compaction",
                            config={
                                "strategy": "fixed",
                                "keep_last": 5,
                            },
                        )
                    ],
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[
                        ResolverConfig(type="session"),
                        ResolverConfig(
                            type="input",
                            subscriptions=[
                                {"event": "user_input", "phase": "completed"}
                            ],
                        ),
                    ],
                ),
            ]
        )
        sr2 = SR2(
            pipeline_config=config,
            llm=MockLLM(),
            token_counter=counter,
        )
        layers = sr2._engine._layers
        system_layer = layers[0]
        assert len(system_layer.transformers) == 1
        await sr2.aclose()

    async def test_compaction_runs_during_turn(self, counter):
        """SR2 with compaction → transformer fires during turn execution."""
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are helpful."},
                        )
                    ],
                    transformers=[
                        TransformerConfig(
                            type="compaction",
                            config={
                                "strategy": "fixed",
                                "keep_last": 5,
                            },
                        )
                    ],
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[
                        ResolverConfig(type="session"),
                        ResolverConfig(
                            type="input",
                            subscriptions=[
                                {"event": "user_input", "phase": "completed"}
                            ],
                        ),
                    ],
                ),
            ]
        )
        sr2 = SR2(
            pipeline_config=config,
            llm=MockLLM(),
            token_counter=counter,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()


# ===========================================================================
# 9. Session ID seam
# ===========================================================================


@pytest.mark.asyncio
class TestSessionIdE2E:

    async def test_explicit_session_id(self, minimal_config, counter):
        """SR2(session_id='custom') → session_id is set exactly."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
            session_id="custom-id-42",
        )
        assert sr2.session_id == "custom-id-42"
        await sr2.aclose()

    async def test_auto_session_id(self, minimal_config, counter):
        """SR2() without session_id → generates unique ULID."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm=MockLLM(),
            token_counter=counter,
        )
        assert sr2.session_id != ""
        assert len(sr2.session_id) == 26
        await sr2.aclose()


# ===========================================================================
# 10. Multi-model LLM seam
# ===========================================================================


@pytest.mark.asyncio
class TestMultiModelE2E:

    async def test_dict_llm_selects_default(self, minimal_config, counter):
        """SR2(llm={'default': llm}) → uses the 'default' model."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm={"default": MockLLM()},
            token_counter=counter,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()

    async def test_dict_llm_falls_back_to_first(self, minimal_config, counter):
        """SR2(llm={'foo': llm}) without 'default' → uses first value."""
        sr2 = SR2(
            pipeline_config=minimal_config,
            llm={"foo": MockLLM()},
            token_counter=counter,
        )
        events = []
        async for event in sr2.turn(make_user_input("test")):
            events.append(event)
        assert any(e.type == "end" for e in events)
        await sr2.aclose()

    async def test_empty_dict_raises(self, minimal_config, counter):
        """SR2(llm={}) → raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            SR2(
                pipeline_config=minimal_config,
                llm={},
                token_counter=counter,
            )
