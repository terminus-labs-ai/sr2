"""Tests for sr2.sr2.SR2 — the public facade class.

The facade is a thin wrapper over sr2.orchestrator.SR2. Its job:
  1. Accept the same constructor signature as the orchestrator.
  2. Delegate seed_session() and turn() to the internal orchestrator.

Behaviors under test (bead obsidian-t9t.1, spec section 1A):
  - seed_session + turn produces correct messages
  - tool providers re-fire every turn (execution_count reset)

All tests import from sr2.sr2 (the facade), NOT sr2.orchestrator.
Tests must FAIL against the current stub (class SR2: def sr2(): pass).
Mock the LLM callable at the boundary.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    ToolProviderConfig,
)
from sr2.models import Message, TextBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import (
    CompletionRequest,
    StreamEvent,
)
from conftest import MockLLM, make_minimal_config, make_user_input


# ---------------------------------------------------------------------------
# Helpers shared across all test classes
# ---------------------------------------------------------------------------


def make_message(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def _all_message_text(request: CompletionRequest) -> str:
    return " ".join(
        block.text
        for msg in request.messages
        for block in msg.content
        if hasattr(block, "text")
    )


# ---------------------------------------------------------------------------
# 1. Construction — facade accepts the spec signature
# ---------------------------------------------------------------------------


class TestFacadeConstruction:
    """SR2 facade can be constructed with the spec signature."""

    def test_constructs_with_required_args(self):
        """SR2(pipeline_config, llm, token_counter) constructs without error."""
        from sr2.sr2 import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert sr2 is not None

    def test_constructs_with_all_args(self):
        """SR2 accepts session_id and provenance_store=None."""
        from sr2.sr2 import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id="test-session-id",
            provenance_store=None,
        )
        assert sr2 is not None

    def test_dict_without_default_key_is_accepted(self):
        """SR2 accepts a dict without a 'default' key (sr2-14: magic string removed)."""
        from sr2.sr2 import SR2

        config = make_minimal_config()
        # A dict without "default" is now valid — first value used as driver.
        instance = SR2(
            pipeline_config=config,
            llm={"other": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert instance is not None

    def test_facade_has_seed_session_method(self):
        """SR2 facade exposes a seed_session method."""
        from sr2.sr2 import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert hasattr(sr2, "seed_session")
        assert callable(sr2.seed_session)

    def test_facade_has_turn_method(self):
        """SR2 facade exposes a turn method."""
        from sr2.sr2 import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert hasattr(sr2, "turn")
        assert callable(sr2.turn)


# ---------------------------------------------------------------------------
# 2. seed_session + turn produces correct messages
# ---------------------------------------------------------------------------


class TestSeedSessionAndTurn:
    """seed_session() followed by turn() produces correct messages in the request."""

    @pytest.mark.asyncio
    async def test_seeded_messages_appear_in_turn_request(self):
        """Messages passed to seed_session() appear in the LLM CompletionRequest
        during the subsequent turn() call."""
        from sr2.sr2 import SR2

        mock_llm = MockLLM()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        sr2.seed_session([make_message("user", "seeded prior message")])

        async for _ in sr2.turn(make_user_input("current input")):
            pass

        assert len(mock_llm.stream_calls) == 1
        request = mock_llm.stream_calls[0]
        all_text = _all_message_text(request)
        assert "seeded prior message" in all_text

    @pytest.mark.asyncio
    async def test_seeded_messages_appear_before_current_input(self):
        """Seeded history precedes the live turn input in the CompletionRequest."""
        from sr2.sr2 import SR2

        mock_llm = MockLLM()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        sr2.seed_session([make_message("user", "history msg")])

        async for _ in sr2.turn(make_user_input("live input")):
            pass

        request = mock_llm.stream_calls[0]
        all_texts = [
            block.text
            for msg in request.messages
            for block in msg.content
            if hasattr(block, "text")
        ]
        history_idx = next((i for i, t in enumerate(all_texts) if "history msg" in t), None)
        live_idx = next((i for i, t in enumerate(all_texts) if "live input" in t), None)
        assert history_idx is not None, "seeded message not in request"
        assert live_idx is not None, "live input not in request"
        assert history_idx < live_idx, "seeded history must precede live input"

    @pytest.mark.asyncio
    async def test_turn_yields_stream_events(self):
        """turn() yields StreamEvent objects from the LLM."""
        from sr2.sr2 import SR2

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="foo"),
            StreamEvent(type="text", text="bar"),
            StreamEvent(type="end"),
        ])
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        events = [e async for e in sr2.turn(make_user_input())]

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 2
        assert text_events[0].text == "foo"
        assert text_events[1].text == "bar"

    @pytest.mark.asyncio
    async def test_turn_is_async_iterable(self):
        """turn() returns an object usable with 'async for'."""
        from sr2.sr2 import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        result = sr2.turn(make_user_input())
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_seed_session_is_synchronous(self):
        """seed_session() must be synchronous — not a coroutine."""
        import inspect
        from sr2.sr2 import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        result = sr2.seed_session([make_message("user", "test")])
        assert not inspect.iscoroutine(result), "seed_session() must be synchronous"

    @pytest.mark.asyncio
    async def test_second_turn_also_produces_output(self):
        """Two consecutive turn() calls both succeed and yield events."""
        from sr2.sr2 import SR2

        mock_llm = MockLLM()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        async for _ in sr2.turn(make_user_input("first")):
            pass
        second_events = [e async for e in sr2.turn(make_user_input("second"))]
        assert len(second_events) > 0


# ---------------------------------------------------------------------------
# 3. Tool providers re-fire every turn (execution_count reset)
# ---------------------------------------------------------------------------


class _SpyToolProvider:
    """Minimal ToolProvider that records fire count across turns."""

    name: str = "spy_tp"

    def __init__(self) -> None:
        from sr2.pipeline.events import EventSubscription

        self.subscriptions = [EventSubscription(event_name="turn_start")]
        self.max_executions: int = 1
        self.execution_count: int = 0
        self.fire_count: int = 0

    async def provide(self, events: list) -> list:
        self.fire_count += 1
        self.execution_count += 1
        return []

    @classmethod
    def build(cls, config: Any, deps: Any) -> "_SpyToolProvider":
        return cls()


def _make_config_with_tool_provider() -> PipelineConfig:
    """Config with a tools layer that hosts a spy tool provider."""
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
                name="tools",
                target="tools",
                resolvers=[],
                tool_providers=[ToolProviderConfig(type="spy_tp")],
            ),
        ]
    )


def _make_spy_ep_side_effect(spy_instance: _SpyToolProvider):
    """Return an entry_points side_effect that exposes the spy tool provider."""

    def _side_effect(group: str) -> list:
        if group == "sr2.resolvers":
            from sr2.pipeline.resolvers.static import StaticResolver

            ep = MagicMock(spec=importlib.metadata.EntryPoint)
            ep.name = "static"
            ep.load.return_value = StaticResolver
            dist = MagicMock()
            dist.name = "sr2"
            ep.dist = dist
            return [ep]

        if group == "sr2.tool_providers":
            ep = MagicMock(spec=importlib.metadata.EntryPoint)
            ep.name = "spy_tp"

            class _Factory:
                """Wraps the spy so build() returns the shared instance."""

                @classmethod
                def build(cls, config: Any, deps: Any) -> _SpyToolProvider:
                    return spy_instance

            ep.load.return_value = _Factory
            dist = MagicMock()
            dist.name = "sr2-test"
            ep.dist = dist
            return [ep]

        return []

    return _side_effect


@pytest.fixture()
def reset_plugin_registries():
    """Reset orchestrator-level PluginRegistry caches between tests."""
    import sr2.orchestrator as orch

    def _reset():
        for registry in (orch._RESOLVERS, orch._TRANSFORMERS, orch._TOOL_PROVIDERS):
            registry._discovered = False
            registry._classes = {}
            registry._collisions = {}

    _reset()
    yield
    _reset()


class TestToolProvidersRefireEachTurn:
    """Tool providers must re-fire on every turn (execution_count reset in turn())."""

    @pytest.mark.asyncio
    async def test_tool_provider_fires_on_first_turn(self, reset_plugin_registries):
        """Tool provider fires during the first turn()."""
        from sr2.sr2 import SR2

        spy = _SpyToolProvider()
        config = _make_config_with_tool_provider()
        side_effect = _make_spy_ep_side_effect(spy)

        with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
            sr2 = SR2(
                pipeline_config=config,
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )
            async for _ in sr2.turn(make_user_input("turn one")):
                pass

        assert spy.fire_count >= 1, "Tool provider must fire on turn 1"

    @pytest.mark.asyncio
    async def test_tool_provider_fires_on_second_turn(self, reset_plugin_registries):
        """Tool provider fires again on turn 2 — execution_count must be reset.

        Without the execution_count reset in SR2.turn(), a max_executions=1
        provider fires only once and is skipped on all subsequent turns.
        """
        from sr2.sr2 import SR2

        spy = _SpyToolProvider()
        config = _make_config_with_tool_provider()
        side_effect = _make_spy_ep_side_effect(spy)

        with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
            sr2 = SR2(
                pipeline_config=config,
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )
            async for _ in sr2.turn(make_user_input("turn one")):
                pass
            fire_count_after_turn1 = spy.fire_count

            async for _ in sr2.turn(make_user_input("turn two")):
                pass
            fire_count_after_turn2 = spy.fire_count

        assert fire_count_after_turn1 >= 1, "Tool provider must fire on turn 1"
        assert fire_count_after_turn2 > fire_count_after_turn1, (
            "Tool provider must re-fire on turn 2 — "
            "execution_count reset is missing in SR2.turn()"
        )

    @pytest.mark.asyncio
    async def test_llm_called_on_both_turns(self, reset_plugin_registries):
        """LLM is called on both turns (basic liveness check for multi-turn)."""
        from sr2.sr2 import SR2

        mock_llm = MockLLM()
        spy = _SpyToolProvider()
        config = _make_config_with_tool_provider()
        side_effect = _make_spy_ep_side_effect(spy)

        with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
            sr2 = SR2(
                pipeline_config=config,
                llm={"default": mock_llm},
                token_counter=CharacterTokenCounter(),
            )
            async for _ in sr2.turn(make_user_input("first")):
                pass
            async for _ in sr2.turn(make_user_input("second")):
                pass

        assert len(mock_llm.stream_calls) == 2


