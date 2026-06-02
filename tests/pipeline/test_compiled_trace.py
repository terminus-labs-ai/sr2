"""Tests for obsidian-40c: Compiled-turn trace (FR1–FR4).

Covers:
  FR1: Tracer protocol gains on_compile(request: CompletionRequest) -> None
  FR2: CollectingTracer.compiled_request attribute — init, update, clear
  FR3: PipelineEngine calls tracer.on_compile() exactly once per run_engine() call, guarded by tracer is not None
  FR4: render_compiled_request(request) renders human-readable output
"""

from __future__ import annotations

import pytest

from conftest import run_engine
from sr2.models import Message, TextBlock, ToolDefinition
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer, Tracer
from sr2.protocols.llm import CompletionRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    *,
    system_text: str | None = None,
    messages: list[Message] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> CompletionRequest:
    """Build a minimal CompletionRequest for testing."""
    system = [TextBlock(text=system_text)] if system_text is not None else None
    return CompletionRequest(
        system=system,
        messages=messages or [],
        tools=tools or None,
    )


def _make_user_message(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def _make_assistant_message(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)])


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


def _make_engine(layers=None, tracer=None):
    from sr2.pipeline.engine import PipelineEngine

    kwargs = dict(
        layers=layers if layers is not None else [],
        token_counter=CharacterTokenCounter(),
    )
    if tracer is not None:
        kwargs["tracer"] = tracer
    return PipelineEngine(**kwargs)


# ---------------------------------------------------------------------------
# FR1 — Tracer protocol has on_compile
# ---------------------------------------------------------------------------


class TestTracerProtocolOnCompile:
    def test_collecting_tracer_satisfies_tracer_protocol(self) -> None:
        """CollectingTracer is still an instance of Tracer after on_compile is added."""
        tracer = CollectingTracer()
        assert isinstance(tracer, Tracer)

    def test_on_compile_does_not_raise(self) -> None:
        """Calling on_compile(request) on CollectingTracer raises no exception."""
        tracer = CollectingTracer()
        request = _make_request(system_text="You are helpful.")
        tracer.on_compile(request)  # must not raise


# ---------------------------------------------------------------------------
# FR2 — CollectingTracer.compiled_request attribute
# ---------------------------------------------------------------------------


class TestCollectingTracerCompiledRequest:
    def test_compiled_request_is_none_initially(self) -> None:
        """compiled_request is None before any on_compile call."""
        tracer = CollectingTracer()
        assert tracer.compiled_request is None

    def test_compiled_request_set_after_on_compile(self) -> None:
        """After on_compile(request), compiled_request is the passed request."""
        tracer = CollectingTracer()
        request = _make_request(system_text="system prompt")
        tracer.on_compile(request)
        assert tracer.compiled_request is request

    def test_compiled_request_updated_on_second_call(self) -> None:
        """After a second on_compile(request2), compiled_request is request2 (most recent wins)."""
        tracer = CollectingTracer()
        request1 = _make_request(system_text="first")
        request2 = _make_request(system_text="second")
        tracer.on_compile(request1)
        tracer.on_compile(request2)
        assert tracer.compiled_request is request2

    def test_clear_resets_compiled_request_to_none(self) -> None:
        """clear() resets compiled_request back to None."""
        tracer = CollectingTracer()
        request = _make_request(system_text="something")
        tracer.on_compile(request)
        assert tracer.compiled_request is not None  # precondition
        tracer.clear()
        assert tracer.compiled_request is None

    def test_clear_also_clears_firing_buffer(self) -> None:
        """clear() resets both the firing buffer and compiled_request."""
        tracer = CollectingTracer()
        request = _make_request(system_text="x")
        tracer.on_compile(request)
        tracer.clear()
        assert tracer.compiled_request is None
        assert tracer.get_trace() == []


# ---------------------------------------------------------------------------
# FR3 — PipelineEngine calls on_compile during run_engine()
# ---------------------------------------------------------------------------


class TestEngineCallsOnCompile:
    @pytest.mark.asyncio
    async def test_compiled_request_populated_after_run(self) -> None:
        """With a CollectingTracer, tracer.compiled_request is a CompletionRequest after run_engine()."""
        tracer = CollectingTracer()
        engine = _make_engine(tracer=tracer)

        await run_engine(engine, [])

        assert tracer.compiled_request is not None
        assert isinstance(tracer.compiled_request, CompletionRequest)

    @pytest.mark.asyncio
    async def test_on_compile_called_exactly_once_per_run(self) -> None:
        """on_compile is called exactly once per run_engine() call."""
        call_log: list[CompletionRequest] = []

        class CountingTracer(CollectingTracer):
            def on_compile(self, request: CompletionRequest) -> None:
                call_log.append(request)
                super().on_compile(request)

        tracer = CountingTracer()
        engine = _make_engine(tracer=tracer)

        await run_engine(engine, [])

        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_on_compile_called_once_per_run_across_multiple_runs(self) -> None:
        """on_compile is called exactly once per run(), not accumulated across turns."""
        call_log: list[CompletionRequest] = []

        class CountingTracer(CollectingTracer):
            def on_compile(self, request: CompletionRequest) -> None:
                call_log.append(request)
                super().on_compile(request)

        tracer = CountingTracer()
        engine = _make_engine(tracer=tracer)

        await run_engine(engine, [])
        await run_engine(engine, [])
        await run_engine(engine, [])

        assert len(call_log) == 3

    @pytest.mark.asyncio
    async def test_engine_with_no_tracer_runs_without_error(self) -> None:
        """With tracer=None, run_engine() completes without error (no on_compile attempted)."""
        engine = _make_engine(tracer=None)
        result = await run_engine(engine, [])
        assert result is not None

    @pytest.mark.asyncio
    async def test_on_compile_not_called_when_tracer_is_none(self) -> None:
        """When tracer=None, no AttributeError or on_compile call is attempted."""
        # If the guard is missing, calling on_compile on None would raise AttributeError.
        # A clean run with tracer=None proves the guard is in place.
        engine = _make_engine(tracer=None)
        # Must not raise
        await run_engine(engine, [])


# ---------------------------------------------------------------------------
# FR4 — render_compiled_request
# ---------------------------------------------------------------------------


class TestRenderCompiledRequest:
    def _render(self, request: CompletionRequest) -> str:
        from sr2.pipeline.tracing import render_compiled_request
        return render_compiled_request(request)

    def test_returns_string(self) -> None:
        """render_compiled_request returns a str."""
        request = _make_request(system_text="You are helpful.")
        result = self._render(request)
        assert isinstance(result, str)

    def test_system_text_appears_in_output(self) -> None:
        """With system blocks present, system text appears in output."""
        request = _make_request(system_text="You are a helpful assistant.")
        output = self._render(request)
        assert "You are a helpful assistant." in output

    def test_message_role_appears_in_output(self) -> None:
        """With messages, the message role appears in output."""
        request = _make_request(
            messages=[_make_user_message("Hello"), _make_assistant_message("Hi")]
        )
        output = self._render(request)
        assert "user" in output.lower() or "assistant" in output.lower()

    def test_message_content_appears_in_output(self) -> None:
        """With messages, message content text appears in output."""
        request = _make_request(messages=[_make_user_message("What is the capital of France?")])
        output = self._render(request)
        assert "What is the capital of France?" in output

    def test_tool_name_appears_in_output(self) -> None:
        """With tools, the tool name appears in output."""
        request = _make_request(tools=[_make_tool("web_search")])
        output = self._render(request)
        assert "web_search" in output

    def test_no_crash_with_system_none_and_tools_none(self) -> None:
        """With system=None and tools=None (messages only), render does not crash."""
        request = _make_request(messages=[_make_user_message("ping")])
        output = self._render(request)
        assert isinstance(output, str)

    def test_no_crash_with_empty_messages(self) -> None:
        """With empty messages list, render does not crash."""
        request = CompletionRequest(
            system=None,
            messages=[],
            tools=None,
        )
        output = self._render(request)
        assert isinstance(output, str)

    def test_multiple_system_blocks_all_appear(self) -> None:
        """With multiple system blocks, all text segments appear in output."""
        request = CompletionRequest(
            system=[TextBlock(text="Block one."), TextBlock(text="Block two.")],
            messages=[],
            tools=None,
        )
        output = self._render(request)
        assert "Block one." in output
        assert "Block two." in output

    def test_multiple_tools_all_appear(self) -> None:
        """With multiple tools, all tool names appear in output."""
        request = _make_request(tools=[_make_tool("search"), _make_tool("calculate")])
        output = self._render(request)
        assert "search" in output
        assert "calculate" in output

    def test_both_user_and_assistant_roles_appear(self) -> None:
        """Multi-message conversation: both user and assistant roles represented in output."""
        request = _make_request(
            messages=[
                _make_user_message("Question?"),
                _make_assistant_message("Answer."),
            ]
        )
        output = self._render(request)
        assert "user" in output.lower()
        assert "assistant" in output.lower()
