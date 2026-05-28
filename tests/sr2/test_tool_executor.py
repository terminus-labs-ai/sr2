"""Tests for sr2-24: ToolExecutor type alias, constructor injection, missing-executor error.

Requirements:
  FR1 — Add `ToolExecutor = Callable[[ToolUseBlock], Awaitable[ToolResultBlock]]` type alias.
  FR2 — `SR2.__init__` accepts `tool_executor: ToolExecutor | None = None`.
  FR6 — If the LLM returns a tool_use block and no executor is configured: raise ConfigError.

Tests will FAIL until the feature is implemented (red phase).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Callable

import pytest

from sr2.config.models import ConfigError, LayerConfig, PipelineConfig, ResolverConfig, EventSubscriptionConfig
from sr2.models import TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    """Minimal LLMCallable for testing — returns configurable stream events."""

    def __init__(self, events: list[StreamEvent] | None = None) -> None:
        self._events: list[StreamEvent] = events or [
            StreamEvent(type="text", text="Hello"),
            StreamEvent(type="end"),
        ]
        self.stream_calls: list[CompletionRequest] = []

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        for event in self._events:
            yield event


def make_user_input(text: str = "hello") -> list:
    return [TextBlock(text=text)]


def make_minimal_config() -> PipelineConfig:
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
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ]
    )


def make_tool_use_stream() -> list[StreamEvent]:
    """Stream events that include a tool_use block."""
    return [
        StreamEvent(type="text", text="Let me look that up."),
        StreamEvent(
            type="tool_use",
            tool_use_id="call_abc123",
            tool_name="get_weather",
            tool_input={"location": "Oslo"},
        ),
        StreamEvent(type="end"),
    ]


async def stub_executor(block: ToolUseBlock) -> ToolResultBlock:
    """A minimal valid ToolExecutor callable."""
    return ToolResultBlock(
        tool_use_id=block.id,
        content=f"result for {block.name}",
    )


# ---------------------------------------------------------------------------
# FR1: ToolExecutor type alias is exported
# ---------------------------------------------------------------------------


class TestToolExecutorTypeAlias:
    def test_tool_executor_importable_from_sr2_orchestrator(self):
        """ToolExecutor type alias must be importable from sr2.orchestrator."""
        from sr2.orchestrator import ToolExecutor  # noqa: F401

    def test_tool_executor_alias_is_not_none(self):
        """ToolExecutor alias is accessible and non-None from sr2.orchestrator."""
        from sr2.orchestrator import ToolExecutor

        # The alias itself is not a runtime class; verify it can be used as a type annotation.
        # A stub_executor callable matches its shape: (ToolUseBlock) -> Awaitable[ToolResultBlock].
        assert ToolExecutor is not None


# ---------------------------------------------------------------------------
# FR2: SR2.__init__ accepts tool_executor kwarg
# ---------------------------------------------------------------------------


class TestToolExecutorConstructorInjection:
    def test_sr2_accepts_tool_executor_kwarg_callable(self):
        """SR2(tool_executor=<callable>) constructs without error."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        assert sr2 is not None

    def test_sr2_accepts_tool_executor_none(self):
        """SR2(tool_executor=None) constructs without error — it's the default."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=None,
        )

        assert sr2 is not None

    def test_sr2_omitting_tool_executor_defaults_to_none(self):
        """Omitting tool_executor entirely must not raise — it defaults to None."""
        from sr2.orchestrator import SR2

        # No tool_executor kwarg at all.
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert sr2 is not None

    def test_sync_callable_raises_type_error_on_invocation(self):
        """Passing a sync (non-coroutine) callable raises TypeError or ConfigError when a tool_use is received."""
        from sr2.orchestrator import SR2

        def sync_executor(block: ToolUseBlock) -> ToolResultBlock:  # type: ignore[return]
            return ToolResultBlock(tool_use_id=block.id, content="sync result")

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=make_tool_use_stream())},
            token_counter=CharacterTokenCounter(),
            tool_executor=sync_executor,  # type: ignore[arg-type]
        )

        # Calling a sync callable where an awaitable is expected must surface an error.
        with pytest.raises((TypeError, ConfigError)):
            asyncio.get_event_loop().run_until_complete(
                _collect(sr2.turn(make_user_input()))
            )


async def _collect(ait: AsyncIterator) -> list:
    return [e async for e in ait]


# ---------------------------------------------------------------------------
# FR6: Raise ConfigError when tool_use returned but no executor configured
# ---------------------------------------------------------------------------


class TestMissingExecutorError:
    @pytest.mark.asyncio
    async def test_raises_config_error_when_tool_use_and_no_executor(self):
        """ConfigError raised when LLM returns tool_use block and executor is None."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=make_tool_use_stream())},
            token_counter=CharacterTokenCounter(),
            tool_executor=None,
        )

        with pytest.raises(ConfigError):
            async for _ in sr2.turn(make_user_input()):
                pass

    @pytest.mark.asyncio
    async def test_raises_config_error_message_mentions_tool_executor(self):
        """ConfigError message gives actionable guidance about configuring a tool_executor."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=make_tool_use_stream())},
            token_counter=CharacterTokenCounter(),
            tool_executor=None,
        )

        with pytest.raises(ConfigError, match="tool_executor"):
            async for _ in sr2.turn(make_user_input()):
                pass

    @pytest.mark.asyncio
    async def test_no_error_when_executor_configured_and_tool_use_returned(self):
        """No ConfigError when LLM returns tool_use block and a valid executor is set."""
        from sr2.orchestrator import SR2

        call_log: list[ToolUseBlock] = []

        async def recording_executor(block: ToolUseBlock) -> ToolResultBlock:
            call_log.append(block)
            return ToolResultBlock(
                tool_use_id=block.id,
                content=f"result: {block.name}",
            )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=make_tool_use_stream())},
            token_counter=CharacterTokenCounter(),
            tool_executor=recording_executor,
        )

        # Must not raise ConfigError
        events = [e async for e in sr2.turn(make_user_input())]
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_executor_called_with_tool_use_block(self):
        """The configured executor is invoked with the ToolUseBlock from the LLM response."""
        from sr2.orchestrator import SR2

        call_log: list[ToolUseBlock] = []

        async def recording_executor(block: ToolUseBlock) -> ToolResultBlock:
            call_log.append(block)
            return ToolResultBlock(
                tool_use_id=block.id,
                content="result",
            )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=make_tool_use_stream())},
            token_counter=CharacterTokenCounter(),
            tool_executor=recording_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        # Allow fire-and-forget to settle
        await asyncio.sleep(0)

        assert len(call_log) == 1, (
            f"Expected executor to be called once, got {len(call_log)} calls"
        )
        block = call_log[0]
        assert isinstance(block, ToolUseBlock)
        assert block.name == "get_weather"
        assert block.id == "call_abc123"
        assert block.input == {"location": "Oslo"}

    @pytest.mark.asyncio
    async def test_text_only_response_does_not_raise_without_executor(self):
        """No ConfigError when LLM returns only text (no tool_use), even with executor=None."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=[
                StreamEvent(type="text", text="Hello there"),
                StreamEvent(type="end"),
            ])},
            token_counter=CharacterTokenCounter(),
            tool_executor=None,
        )

        # Must not raise — no tool_use in the response.
        events = [e async for e in sr2.turn(make_user_input())]
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_config_error_is_not_swallowed_by_post_process(self):
        """ConfigError propagates to the caller — it is not caught internally."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM(events=make_tool_use_stream())},
            token_counter=CharacterTokenCounter(),
            tool_executor=None,
        )

        # ConfigError must propagate out of the async for loop to the caller.
        raised = False
        try:
            async for _ in sr2.turn(make_user_input()):
                pass
        except ConfigError:
            raised = True

        assert raised, "ConfigError must propagate to the caller of turn()"
