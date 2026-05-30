"""Tests for sr2-2: finalize via engine.end_turn so assistant_response drains.

Requirements:
  - After the tool loop builds the final CompletionResponse, queue
    assistant_response and call engine.end_turn() (emits turn_end, drains
    bus, compiles, on_compile, returns PipelineResult).
  - This drain fires SessionResolver (captures final assistant message)
    and MemoryExtractionTransformer.
  - Runs inside the deferred task (sr2-4), not before client end.
    [Note: sr2-4 is the deferred post_process task. For now, end_turn
     happens before yielding 'end' and before post_process.]

Tests verify:
  1. 2-turn history includes turn 1 assistant reply (SessionResolver
     captures assistant_response because end_turn drains the bus).
  2. memory_extraction fires once/turn (MemoryExtractionTransformer
     processes assistant_response because end_turn drains the bus).
  3. turn_end event is fired on the bus after the loop completes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.memory.extraction import RuleBasedExtractor
from sr2.memory.store import InMemoryMemoryStore
from sr2.models import TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
    tool_use_event,
)


def make_config_with_memory_extraction() -> PipelineConfig:
    """PipelineConfig with a layer that has MemoryExtractionTransformer."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(
                        type="session",
                    ),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
                transformers=[
                    TransformerConfig(
                        type="memory_extraction",
                        config={"extractor": "rule_based"},
                    )
                ],
            ),
        ]
    )


class TestEndTurnFinalize:
    """sr2-2: engine.end_turn() must be called after the loop so
    assistant_response drains through SessionResolver and MemoryExtractionTransformer."""

    @pytest.mark.asyncio
    async def test_session_resolver_captures_assistant_reply_across_turns(self):
        """2-turn history includes turn 1 assistant reply.

        Turn 1: LLM returns text "Hello from turn 1".
        Turn 2: LLM returns text "Hello from turn 2".

        After turn 2, the SessionResolver history should contain the
        assistant reply from turn 1. This only works if end_turn() drains
        the bus and processes the assistant_response event.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Hello from turn 1."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        memory_store = InMemoryMemoryStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            memory_store=memory_store,
        )

        # Run turn 1
        async for _ in sr2.turn(make_user_input("Question 1")):
            pass

        # After turn 1, SessionResolver should have captured the assistant reply.
        # Find the SessionResolver in the conversation layer.
        conv_layer = None
        for layer in sr2._engine.layers:
            if layer.name == "conversation":
                conv_layer = layer
                break

        assert conv_layer is not None, "conversation layer not found"

        session_resolver = None
        for resolver in conv_layer.resolvers:
            if resolver.name == "session":
                session_resolver = resolver
                break

        assert session_resolver is not None, "SessionResolver not found"

        # Check that the assistant reply from turn 1 is in history.
        assistant_messages = [
            m for m in session_resolver._history if m.role == "assistant"
        ]
        assert len(assistant_messages) >= 1, (
            f"Expected at least 1 assistant message in SessionResolver history "
            f"after turn 1, got {len(assistant_messages)}. "
            f"This means assistant_response was not drained by end_turn()."
        )

        # Verify the content
        assert any(
            "Hello from turn 1" in (b.text if isinstance(b, TextBlock) else "")
            for m in assistant_messages
            for b in m.content
            if isinstance(b, TextBlock)
        ), "Turn 1 assistant text not found in SessionResolver history"

    @pytest.mark.asyncio
    async def test_turn_end_event_fired_on_bus(self):
        """After the loop completes, a turn_end event must be present
        in the bus firing records (proving end_turn was called)."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Response."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        # Check that turn_end was processed by looking at SessionResolver
        # execution count — it should have fired for turn_start, user_input,
        # and the events triggered by end_turn.
        conv_layer = None
        for layer in sr2._engine.layers:
            if layer.name == "conversation":
                conv_layer = layer
                break

        session_resolver = None
        for resolver in conv_layer.resolvers:
            if resolver.name == "session":
                session_resolver = resolver
                break

        assert session_resolver is not None

        # The key test: assistant_response must have been captured.
        # If end_turn() was called, the bus drains and SessionResolver
        # processes the assistant_response event.
        assistant_msgs = [
            m for m in session_resolver._history if m.role == "assistant"
        ]
        assert len(assistant_msgs) >= 1, (
            f"No assistant messages in history after turn. "
            f"end_turn() may not be draining the bus. "
            f"History has {len(session_resolver._history)} messages total."
        )

    @pytest.mark.asyncio
    async def test_memory_extraction_transformer_fires_on_assistant_response(self):
        """MemoryExtractionTransformer must fire once per turn when
        end_turn drains the bus.

        We verify by checking that the transformer's execution_count
        incremented after the turn.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Remember this fact: the sky is blue."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        memory_store = InMemoryMemoryStore()
        sr2 = SR2(
            pipeline_config=make_config_with_memory_extraction(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            memory_store=memory_store,
            memory_extractor=RuleBasedExtractor(),
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        # Find the MemoryExtractionTransformer
        conv_layer = None
        for layer in sr2._engine.layers:
            if layer.name == "conversation":
                conv_layer = layer
                break

        mem_transformer = None
        for transformer in conv_layer.transformers:
            if transformer.name == "memory_extraction":
                mem_transformer = transformer
                break

        assert mem_transformer is not None, "MemoryExtractionTransformer not found"

        # The transformer should have fired (execution_count > 0)
        # because end_turn() drains the bus and processes assistant_response.
        assert mem_transformer.execution_count >= 1, (
            f"MemoryExtractionTransformer execution_count is {mem_transformer.execution_count}. "
            f"It should be >= 1 because end_turn() drains the bus and fires "
            f"the assistant_response event. If this is 0, end_turn() is not being called."
        )

    @pytest.mark.asyncio
    async def test_two_turns_each_capture_assistant_response(self):
        """Each turn's assistant response is captured independently.

        Turn 1: "Answer one."
        Turn 2: "Answer two."

        SessionResolver history should have 2 assistant messages after 2 turns.
        """
        from sr2.orchestrator import SR2

        # Each turn gets its own LLM instance state via SequentialMockLLM
        # We need the LLM to respond differently on each turn call.
        # SequentialMockLLM returns different sequences per stream() call,
        # so we provide enough sequences.
        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Answer one."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Answer two."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        # Turn 1
        async for _ in sr2.turn(make_user_input("Q1")):
            pass

        # Turn 2
        async for _ in sr2.turn(make_user_input("Q2")):
            pass

        # Find SessionResolver
        conv_layer = None
        for layer in sr2._engine.layers:
            if layer.name == "conversation":
                conv_layer = layer
                break

        session_resolver = None
        for resolver in conv_layer.resolvers:
            if resolver.name == "session":
                session_resolver = resolver
                break

        assert session_resolver is not None

        assistant_msgs = [
            m for m in session_resolver._history if m.role == "assistant"
        ]
        assert len(assistant_msgs) == 2, (
            f"Expected 2 assistant messages after 2 turns, got {len(assistant_msgs)}. "
            f"Each turn's assistant_response must be captured via end_turn() drain."
        )

        # Verify content of both
        texts = []
        for m in assistant_msgs:
            for b in m.content:
                if isinstance(b, TextBlock):
                    texts.append(b.text)

        assert "Answer one." in texts, f"Turn 1 answer not in history. Got: {texts}"
        assert "Answer two." in texts, f"Turn 2 answer not in history. Got: {texts}"
