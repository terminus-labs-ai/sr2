"""Tests for sr2.runtime.agent — SR2Runtime."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from sr2.runtime.agent import SR2Runtime
from sr2.runtime.config import AgentConfig, ModelConfig, PersonaConfig, ToolConfig
from sr2.runtime.llm import LLMResponse
from sr2.runtime.result import RuntimeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processed_context(
    messages: list[dict] | None = None,
    tool_schemas: list[dict] | None = None,
    tool_choice: str = "auto",
):
    """Create a mock ProcessedContext."""
    ctx = MagicMock()
    ctx.messages = messages or [{"role": "user", "content": "hi"}]
    ctx.tool_schemas = tool_schemas or []
    ctx.tool_choice = tool_choice
    ctx.pipeline_result = MagicMock()
    ctx.pipeline_result.compaction_count = 0
    return ctx


def _make_runtime(config: AgentConfig) -> SR2Runtime:
    """Build an SR2Runtime with mocked internals (no real SR2)."""
    with patch("sr2.runtime.agent.SR2Runtime._build_sr2") as mock_build:
        mock_build.return_value = AsyncMock()
        runtime = SR2Runtime(config)
    return runtime


def _wire_mocks(
    runtime: SR2Runtime,
    *,
    llm_responses: list[LLMResponse] | None = None,
    processed: MagicMock | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Replace _sr2 and _llm with mocks, return (mock_sr2, mock_llm)."""
    mock_sr2 = AsyncMock()
    mock_sr2.process = AsyncMock(
        return_value=processed or _make_processed_context()
    )
    mock_sr2.post_process = AsyncMock()
    runtime._sr2 = mock_sr2

    mock_llm = AsyncMock()
    responses = llm_responses or [
        LLMResponse(content="done", tool_calls=[], prompt_tokens=10,
                    completion_tokens=5, total_tokens=15),
    ]
    mock_llm.complete = AsyncMock(side_effect=responses)
    runtime._llm = mock_llm

    return mock_sr2, mock_llm


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_from_config(self, tmp_path, sample_config_dict):
        """from_config() loads YAML and constructs a runtime."""
        yaml_path = tmp_path / "agent.yaml"
        yaml_path.write_text(yaml.dump(sample_config_dict))

        with patch("sr2.runtime.agent.SR2Runtime._build_sr2") as mock_build:
            mock_build.return_value = AsyncMock()
            runtime = SR2Runtime.from_config(yaml_path)

        assert runtime.config.name == "test-agent"
        assert isinstance(runtime, SR2Runtime)

    def test_from_dict(self, sample_config_dict):
        """from_dict() constructs a runtime from a plain dict."""
        with patch("sr2.runtime.agent.SR2Runtime._build_sr2") as mock_build:
            mock_build.return_value = AsyncMock()
            runtime = SR2Runtime.from_dict(sample_config_dict)

        assert runtime.config.name == "test-agent"

    def test_name_property(self, sample_agent_config):
        """name property returns config name."""
        runtime = _make_runtime(sample_agent_config)
        assert runtime.name == "test-agent"

    def test_session_id_includes_name(self, sample_agent_config):
        """Session ID incorporates the agent name."""
        runtime = _make_runtime(sample_agent_config)
        assert "test-agent" in runtime._session_id

    def test_no_tools_creates_no_executor(self):
        """When no tools configured, _tools is None."""
        config = AgentConfig(
            name="no-tools",
            model=ModelConfig(name="m"),
            persona=PersonaConfig(system_prompt="test"),
            tools=[],
        )
        runtime = _make_runtime(config)
        assert runtime._tools is None


# ---------------------------------------------------------------------------
# execute() — simple task (no tool calls)
# ---------------------------------------------------------------------------


class TestExecuteSimple:

    async def test_simple_task_returns_output(self, sample_agent_config):
        """A simple task with no tool calls returns model output."""
        runtime = _make_runtime(sample_agent_config)
        mock_sr2, mock_llm = _wire_mocks(runtime)

        result = await runtime.execute("Say hello")

        assert result.success is True
        assert result.output == "done"
        assert result.error is None

    async def test_simple_task_calls_process(self, sample_agent_config):
        """execute() calls sr2.process() with the task."""
        runtime = _make_runtime(sample_agent_config)
        mock_sr2, _ = _wire_mocks(runtime)

        await runtime.execute("Say hello")

        mock_sr2.process.assert_called_once()
        call_kwargs = mock_sr2.process.call_args.kwargs
        assert call_kwargs["trigger_input"] == "Say hello"
        assert call_kwargs["system_prompt"] == "You are a helpful test agent."

    async def test_simple_task_calls_post_process(self, sample_agent_config):
        """After final response, post_process() records the assistant turn."""
        runtime = _make_runtime(sample_agent_config)
        mock_sr2, _ = _wire_mocks(runtime)

        await runtime.execute("Say hello")

        mock_sr2.post_process.assert_called_once()
        call_kwargs = mock_sr2.post_process.call_args.kwargs
        assert call_kwargs["role"] == "assistant"
        assert call_kwargs["content"] == "done"
        assert call_kwargs["user_message"] == "Say hello"

    async def test_simple_task_metrics(self, sample_agent_config):
        """Token metrics are populated from LLM response."""
        runtime = _make_runtime(sample_agent_config)
        _wire_mocks(runtime)

        result = await runtime.execute("Say hello")

        assert result.metrics.llm_calls == 1
        assert result.metrics.prompt_tokens == 10
        assert result.metrics.completion_tokens == 5
        assert result.metrics.total_tokens == 15
        assert result.metrics.tool_calls == 0
        assert result.metrics.wall_time_ms > 0


# ---------------------------------------------------------------------------
# execute() — context injection
# ---------------------------------------------------------------------------


class TestExecuteWithContext:

    async def test_context_injected_before_process(self, sample_agent_config):
        """When context is provided, post_process is called first with it."""
        runtime = _make_runtime(sample_agent_config)
        mock_sr2, _ = _wire_mocks(runtime)

        await runtime.execute("Do something", context="Prior step output")

        # post_process called twice: context injection + final assistant turn
        assert mock_sr2.post_process.call_count == 2

        # First call: context injection
        first_call = mock_sr2.post_process.call_args_list[0]
        assert first_call.kwargs["role"] == "user"
        assert "Prior step output" in first_call.kwargs["content"]

        # Second call: assistant response
        second_call = mock_sr2.post_process.call_args_list[1]
        assert second_call.kwargs["role"] == "assistant"

    async def test_context_turn_increments_counter(self, sample_agent_config):
        """Context injection uses turn 1, assistant response uses turn 2."""
        runtime = _make_runtime(sample_agent_config)
        mock_sr2, _ = _wire_mocks(runtime)

        await runtime.execute("Do it", context="ctx")

        turns = [
            c.kwargs["turn_number"]
            for c in mock_sr2.post_process.call_args_list
        ]
        assert turns == [1, 2]


# ---------------------------------------------------------------------------
# execute() — tool calling loop
# ---------------------------------------------------------------------------


class TestExecuteToolCalls:

    async def test_tool_call_then_final(self, sample_agent_config):
        """LLM returns tool call first, then final text on second iteration."""
        runtime = _make_runtime(sample_agent_config)

        # Mock tool executor
        mock_tools = AsyncMock()
        mock_tools.get_schemas.return_value = [
            {"name": "search", "parameters": {}}
        ]
        mock_tools.execute = AsyncMock(return_value="search result")
        runtime._tools = mock_tools

        # Two LLM responses: tool call, then text
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
                prompt_tokens=20, completion_tokens=10, total_tokens=30,
            ),
            LLMResponse(
                content="Here is the answer",
                tool_calls=[],
                prompt_tokens=30, completion_tokens=15, total_tokens=45,
            ),
        ]
        mock_sr2, _ = _wire_mocks(runtime, llm_responses=responses)

        result = await runtime.execute("Find something")

        assert result.success is True
        assert result.output == "Here is the answer"
        assert result.metrics.llm_calls == 2
        assert result.metrics.tool_calls == 1
        assert len(result.tool_results) == 1
        assert result.tool_results[0]["tool"] == "search"
        assert result.tool_results[0]["result"] == "search result"
        assert result.tool_results[0]["error"] is None

    async def test_tool_error_captured(self, sample_agent_config):
        """Tool execution errors are captured in tool_results."""
        runtime = _make_runtime(sample_agent_config)

        mock_tools = AsyncMock()
        mock_tools.get_schemas.return_value = [{"name": "fail", "parameters": {}}]
        mock_tools.execute = AsyncMock(side_effect=RuntimeError("boom"))
        runtime._tools = mock_tools

        responses = [
            LLMResponse(
                content=None,
                tool_calls=[{"name": "fail", "arguments": {}}],
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ),
            LLMResponse(
                content="recovered",
                tool_calls=[],
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ),
        ]
        _wire_mocks(runtime, llm_responses=responses)

        result = await runtime.execute("Try failing tool")

        assert result.success is True
        assert result.output == "recovered"
        assert result.tool_results[0]["error"] == "boom"
        assert result.tool_results[0]["result"] is None

    async def test_tool_result_fed_to_post_process(self, sample_agent_config):
        """Tool results are fed back through sr2.post_process()."""
        runtime = _make_runtime(sample_agent_config)

        mock_tools = AsyncMock()
        mock_tools.get_schemas.return_value = [{"name": "echo", "parameters": {}}]
        mock_tools.execute = AsyncMock(return_value="echoed")
        runtime._tools = mock_tools

        responses = [
            LLMResponse(
                content=None,
                tool_calls=[{"name": "echo", "arguments": {"msg": "hi"}}],
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ),
            LLMResponse(
                content="final",
                tool_calls=[],
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ),
        ]
        mock_sr2, _ = _wire_mocks(runtime, llm_responses=responses)

        await runtime.execute("Echo test")

        # Find the post_process call for tool_result
        tool_calls = [
            c for c in mock_sr2.post_process.call_args_list
            if c.kwargs.get("role") == "tool_result"
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].kwargs["content"] == "echoed"
        assert tool_calls[0].kwargs["tool_results"][0]["tool_name"] == "echo"


# ---------------------------------------------------------------------------
# execute() — max_tool_iterations
# ---------------------------------------------------------------------------


class TestMaxToolIterations:

    async def test_loop_stops_at_max_iterations(self, sample_agent_config):
        """Tool loop stops at max_tool_iterations even if LLM keeps calling tools."""
        sample_agent_config.output.max_tool_iterations = 3
        runtime = _make_runtime(sample_agent_config)

        mock_tools = AsyncMock()
        mock_tools.get_schemas.return_value = [{"name": "loop", "parameters": {}}]
        mock_tools.execute = AsyncMock(return_value="ok")
        runtime._tools = mock_tools

        # LLM always returns tool calls (never produces final text)
        always_tool = LLMResponse(
            content="partial",
            tool_calls=[{"name": "loop", "arguments": {}}],
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        _wire_mocks(
            runtime,
            llm_responses=[always_tool, always_tool, always_tool],
        )

        result = await runtime.execute("Infinite loop")

        assert result.metrics.llm_calls == 3
        assert result.metrics.tool_calls == 3
        # Output is the content from the last response
        assert result.output == "partial"


# ---------------------------------------------------------------------------
# execute() — error handling
# ---------------------------------------------------------------------------


class TestExecuteErrors:

    async def test_exception_returns_error_result(self, sample_agent_config):
        """Exceptions during execute() produce success=False result."""
        runtime = _make_runtime(sample_agent_config)
        mock_sr2 = AsyncMock()
        mock_sr2.process = AsyncMock(side_effect=RuntimeError("sr2 broke"))
        runtime._sr2 = mock_sr2

        result = await runtime.execute("Fail please")

        assert result.success is False
        assert "sr2 broke" in result.error
        assert result.output == ""
        assert result.metrics.wall_time_ms > 0


# ---------------------------------------------------------------------------
# execute() — token accumulation
# ---------------------------------------------------------------------------


class TestTokenAccumulation:

    async def test_tokens_accumulate_across_iterations(self, sample_agent_config):
        """Token counts sum across multiple LLM calls in the tool loop."""
        runtime = _make_runtime(sample_agent_config)

        mock_tools = AsyncMock()
        mock_tools.get_schemas.return_value = [{"name": "t", "parameters": {}}]
        mock_tools.execute = AsyncMock(return_value="ok")
        runtime._tools = mock_tools

        responses = [
            LLMResponse(
                content=None,
                tool_calls=[{"name": "t", "arguments": {}}],
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
            ),
            LLMResponse(
                content="done",
                tool_calls=[],
                prompt_tokens=200, completion_tokens=80, total_tokens=280,
            ),
        ]
        _wire_mocks(runtime, llm_responses=responses)

        result = await runtime.execute("accumulate")

        assert result.metrics.prompt_tokens == 300
        assert result.metrics.completion_tokens == 130
        assert result.metrics.total_tokens == 430
        assert result.metrics.llm_calls == 2


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:

    async def test_reset_clears_turn_counter(self, sample_agent_config):
        """reset() sets turn counter back to zero."""
        runtime = _make_runtime(sample_agent_config)
        runtime._turn_counter = 5
        # Use MagicMock so destroy_session (sync) does not produce warnings
        runtime._sr2._conversation = MagicMock()

        await runtime.reset()

        assert runtime._turn_counter == 0

    async def test_reset_destroys_session(self, sample_agent_config):
        """reset() calls destroy_session on the conversation manager."""
        runtime = _make_runtime(sample_agent_config)
        mock_conv = MagicMock()
        runtime._sr2._conversation = mock_conv

        await runtime.reset()

        mock_conv.destroy_session.assert_called_once_with(runtime._session_id)

    async def test_reset_tolerates_missing_conversation_manager(
        self, sample_agent_config
    ):
        """reset() does not fail if SR2 has no _conversation attribute."""
        runtime = _make_runtime(sample_agent_config)
        # Remove _conversation if present
        if hasattr(runtime._sr2, "_conversation"):
            del runtime._sr2._conversation

        await runtime.reset()  # Should not raise
        assert runtime._turn_counter == 0


# ---------------------------------------------------------------------------
# _build_pipeline_dict()
# ---------------------------------------------------------------------------


class TestBuildPipelineDict:

    def test_maps_context_window(self, sample_agent_config):
        """context_window maps to token_budget."""
        sample_agent_config.context.context_window = 65536
        runtime = _make_runtime(sample_agent_config)

        d = runtime._build_pipeline_dict()
        assert d["token_budget"] == 65536

    def test_maps_active_turns(self, sample_agent_config):
        """conversation.active_turns maps to compaction.raw_window."""
        sample_agent_config.context.conversation = {"active_turns": 20}
        runtime = _make_runtime(sample_agent_config)

        d = runtime._build_pipeline_dict()
        assert d["compaction"]["raw_window"] == 20

    def test_maps_memory_enabled(self, sample_agent_config):
        """memory.enabled maps to memory.extract."""
        sample_agent_config.context.memory = {"enabled": True}
        runtime = _make_runtime(sample_agent_config)

        d = runtime._build_pipeline_dict()
        assert d["memory"]["extract"] is True

    def test_pipeline_override_merged(self, sample_agent_config):
        """pipeline_override fields are merged into the dict."""
        sample_agent_config.context.pipeline_override = {
            "degradation": {"enabled": True},
        }
        runtime = _make_runtime(sample_agent_config)

        d = runtime._build_pipeline_dict()
        assert d["degradation"] == {"enabled": True}
