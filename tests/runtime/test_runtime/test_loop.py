"""Tests for the LLM loop."""

from unittest.mock import AsyncMock

import pytest

from sr2_runtime.llm import LLMResponse, LLMLoop
from sr2.tools.models import (
    ToolDefinition,
    ToolManagementConfig,
    ToolStateConfig,
    ToolTransitionConfig,
)
from sr2.tools.state_machine import ToolStateMachine


def _text_response(content="Hello", input_tokens=100, output_tokens=50, cached=0):
    """LLM response with text only (no tool calls)."""
    return LLMResponse(
        content=content,
        tool_calls=[],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        model="test-model",
    )


def _tool_response(tool_calls, content="", input_tokens=100, output_tokens=50, cached=0):
    """LLM response with tool calls."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        model="test-model",
    )


class TestLLMLoop:
    """Tests for the LLM loop."""

    def _make_loop(self, llm_responses, tool_results=None, max_iterations=25):
        """Create an LLMLoop with mocked dependencies."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=llm_responses)

        mock_executor = AsyncMock()
        if tool_results:
            mock_executor.execute = AsyncMock(side_effect=tool_results)
        else:
            mock_executor.execute = AsyncMock(return_value="tool result")

        loop = LLMLoop(
            llm_client=mock_llm,
            tool_executor=mock_executor,
            max_iterations=max_iterations,
        )
        return loop, mock_llm, mock_executor

    @pytest.mark.asyncio
    async def test_text_response_completes_immediately(self):
        loop, _, _ = self._make_loop([_text_response("Done")])
        result = await loop.run([{"role": "user", "content": "hi"}])

        assert result.stopped_reason == "complete"
        assert result.response_text == "Done"
        assert result.iterations == 1
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_one_tool_call_then_text(self):
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {"q": "test"}}]
        responses = [
            _tool_response(tool_calls, content="Searching..."),
            _text_response("Found it!"),
        ]
        loop, _, executor = self._make_loop(responses, ["search result"])

        result = await loop.run([{"role": "user", "content": "find test"}])

        assert result.stopped_reason == "complete"
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"
        assert result.response_text == "Found it!"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_response(self):
        tool_calls = [
            {"id": "tc_1", "name": "search", "arguments": {"q": "a"}},
            {"id": "tc_2", "name": "read", "arguments": {"file": "b.txt"}},
        ]
        responses = [
            _tool_response(tool_calls),
            _text_response("All done"),
        ]
        loop, _, executor = self._make_loop(responses, ["result_a", "result_b"])

        result = await loop.run([{"role": "user", "content": "do stuff"}])

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "search"
        assert result.tool_calls[1].tool_name == "read"

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        tool_calls = [{"id": "tc_1", "name": "loop_tool", "arguments": {}}]
        # Always return tool calls — never complete
        responses = [_tool_response(tool_calls) for _ in range(3)]
        loop, _, _ = self._make_loop(responses, max_iterations=3)

        result = await loop.run([{"role": "user", "content": "go"}])

        assert result.stopped_reason == "max_iterations"
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_llm_call_error(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
        mock_executor = AsyncMock()

        loop = LLMLoop(llm_client=mock_llm, tool_executor=mock_executor)
        result = await loop.run([{"role": "user", "content": "hi"}])

        assert result.stopped_reason == "error"
        assert "API down" in result.response_text

    @pytest.mark.asyncio
    async def test_tool_execution_failure_captured(self):
        tool_calls = [{"id": "tc_1", "name": "bad_tool", "arguments": {}}]
        responses = [
            _tool_response(tool_calls),
            _text_response("Continuing after error"),
        ]
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=responses)
        mock_executor = AsyncMock()
        mock_executor.execute = AsyncMock(side_effect=RuntimeError("tool broke"))

        loop = LLMLoop(llm_client=mock_llm, tool_executor=mock_executor)
        result = await loop.run([{"role": "user", "content": "test"}])

        assert result.stopped_reason == "complete"
        assert result.tool_calls[0].success is False
        assert result.tool_calls[0].error == "tool broke"

    @pytest.mark.asyncio
    async def test_token_counts_accumulate(self):
        tool_calls = [{"id": "tc_1", "name": "t", "arguments": {}}]
        responses = [
            _tool_response(tool_calls, input_tokens=200, output_tokens=100, cached=50),
            _text_response("done", input_tokens=300, output_tokens=150, cached=100),
        ]
        loop, _, _ = self._make_loop(responses, ["ok"])

        result = await loop.run([{"role": "user", "content": "go"}])

        assert result.total_input_tokens == 500
        assert result.total_output_tokens == 250
        assert result.cached_tokens == 150
        assert result.total_tokens == 750

    @pytest.mark.asyncio
    async def test_cache_hit_rate_across_iterations(self):
        tool_calls = [{"id": "tc_1", "name": "t", "arguments": {}}]
        responses = [
            _tool_response(tool_calls, input_tokens=1000, cached=500),
            _text_response("done", input_tokens=1000, cached=800),
        ]
        loop, _, _ = self._make_loop(responses, ["ok"])

        result = await loop.run([{"role": "user", "content": "go"}])

        assert result.cache_hit_rate == 1300 / 2000

    @pytest.mark.asyncio
    async def test_messages_grow_during_loop(self):
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {"q": "x"}}]
        responses = [
            _tool_response(tool_calls),
            _text_response("done"),
        ]
        loop, _, _ = self._make_loop(responses, ["found"])

        messages = [{"role": "user", "content": "find x"}]
        await loop.run(messages)

        # Should now have: user + assistant(tool_calls) + tool_result = 3 messages
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_passed_correctly(self):
        tool_calls = [{"id": "tc_1", "name": "calc", "arguments": {"x": 5, "y": 10}}]
        responses = [
            _tool_response(tool_calls),
            _text_response("15"),
        ]
        loop, _, executor = self._make_loop(responses, ["15"])

        await loop.run([{"role": "user", "content": "add 5 + 10"}])

        executor.execute.assert_called_once_with("calc", {"x": 5, "y": 10})


class TestLLMLoopWithToolMasking:
    """Tests for tool masking integration in the LLM loop."""

    def _make_loop(self, llm_responses, tool_results=None, max_iterations=25):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=llm_responses)
        mock_executor = AsyncMock()
        if tool_results:
            mock_executor.execute = AsyncMock(side_effect=tool_results)
        else:
            mock_executor.execute = AsyncMock(return_value="tool result")
        loop = LLMLoop(
            llm_client=mock_llm,
            tool_executor=mock_executor,
            max_iterations=max_iterations,
        )
        return loop, mock_llm, mock_executor

    def _make_state_machine(
        self,
        tool_names,
        states=None,
        transitions=None,
        strategy="allowed_list",
        initial_state="default",
    ):
        tools = [ToolDefinition(name=n, description=f"{n} tool") for n in tool_names]
        if states is None:
            states = [ToolStateConfig(name="default")]
        mgmt = ToolManagementConfig(
            tools=tools,
            states=states,
            transitions=transitions or [],
            masking_strategy=strategy,
            initial_state=initial_state,
        )
        return ToolStateMachine(mgmt)

    @pytest.mark.asyncio
    async def test_no_state_machine_uses_original_schemas(self):
        """Without a state machine, behavior is unchanged."""
        loop, mock_llm, _ = self._make_loop([_text_response("Hello")])
        schemas = [{"name": "search", "description": "search"}]

        result = await loop.run(
            [{"role": "user", "content": "hi"}],
            tool_schemas=schemas,
        )

        assert result.response_text == "Hello"
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["tools"] == schemas

    @pytest.mark.asyncio
    async def test_state_machine_filters_tools_on_first_call(self):
        """State machine filters tool schemas before the first LLM call."""
        loop, mock_llm, _ = self._make_loop([_text_response("Done")])

        sm = self._make_state_machine(
            ["search", "delete"],
            states=[
                ToolStateConfig(name="safe", allowed_tools=["search"]),
            ],
            initial_state="safe",
        )

        # Pass all schemas but state machine should filter
        all_schemas = [
            {"name": "search", "description": "search"},
            {"name": "delete", "description": "delete"},
        ]
        await loop.run(
            [{"role": "user", "content": "hi"}],
            tool_schemas=all_schemas,
            tool_state_machine=sm,
        )

        # LLM should only see the filtered schemas from masking output
        call_kwargs = mock_llm.complete.call_args
        passed_tools = call_kwargs.kwargs["tools"]
        tool_names = [t["name"] for t in passed_tools]
        assert "search" in tool_names
        assert "delete" not in tool_names

    @pytest.mark.asyncio
    async def test_state_transition_after_tool_call(self):
        """Tool execution triggers state transition, changing available tools."""
        # Scenario: start in "planning" state (only "plan" tool), calling "plan"
        # transitions to "execution" state (only "execute" tool)
        sm = self._make_state_machine(
            ["plan", "execute"],
            states=[
                ToolStateConfig(name="planning", allowed_tools=["plan"]),
                ToolStateConfig(name="execution", allowed_tools=["execute"]),
            ],
            transitions=[
                ToolTransitionConfig(
                    from_state="planning",
                    to_state="execution",
                    trigger="agent_action",
                    condition="tool_name == 'plan'",
                ),
            ],
            initial_state="planning",
        )

        plan_call = [{"id": "tc_1", "name": "plan", "arguments": {}}]
        exec_call = [{"id": "tc_2", "name": "execute", "arguments": {}}]
        responses = [
            _tool_response(plan_call),   # iteration 1: calls plan
            _tool_response(exec_call),   # iteration 2: calls execute (now in execution state)
            _text_response("All done"),  # iteration 3: text response
        ]
        loop, mock_llm, _ = self._make_loop(responses, ["plan done", "exec done"])

        result = await loop.run(
            [{"role": "user", "content": "go"}],
            tool_schemas=[
                {"name": "plan", "description": "plan"},
                {"name": "execute", "description": "execute"},
            ],
            tool_state_machine=sm,
        )

        assert result.stopped_reason == "complete"
        assert result.iterations == 3
        assert sm.current_state_name == "execution"

        # First call should have only "plan" tool, second call should have only "execute"
        calls = mock_llm.complete.call_args_list
        first_tools = calls[0].kwargs["tools"]
        second_tools = calls[1].kwargs["tools"]
        assert [t["name"] for t in first_tools] == ["plan"]
        assert [t["name"] for t in second_tools] == ["execute"]

    @pytest.mark.asyncio
    async def test_no_transition_keeps_same_tools(self):
        """When no transition matches, tools stay the same."""
        sm = self._make_state_machine(
            ["search", "read"],
            states=[ToolStateConfig(name="default", allowed_tools="all")],
        )

        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {}}]
        responses = [
            _tool_response(tool_calls),
            _text_response("Found it"),
        ]
        loop, mock_llm, _ = self._make_loop(responses, ["result"])

        await loop.run(
            [{"role": "user", "content": "find"}],
            tool_schemas=[
                {"name": "search", "description": "s"},
                {"name": "read", "description": "r"},
            ],
            tool_state_machine=sm,
        )

        assert sm.current_state_name == "default"
        # Both calls should have the same tools
        calls = mock_llm.complete.call_args_list
        assert len(calls[0].kwargs["tools"]) == 2
        assert len(calls[1].kwargs["tools"]) == 2

    @pytest.mark.asyncio
    async def test_none_strategy_passes_all_tools(self):
        """'none' masking strategy passes all tools through."""
        sm = self._make_state_machine(
            ["a", "b"],
            strategy="none",
        )

        loop, mock_llm, _ = self._make_loop([_text_response("ok")])
        await loop.run(
            [{"role": "user", "content": "hi"}],
            tool_schemas=[{"name": "a", "description": "a"}, {"name": "b", "description": "b"}],
            tool_state_machine=sm,
        )

        passed_tools = mock_llm.complete.call_args.kwargs["tools"]
        assert len(passed_tools) == 2


class TestMaxIterationsSynthesis:
    """Tests for forced synthesis when max_iterations reached with pending tool calls."""

    def _make_loop(self, llm_responses, tool_results=None, max_iterations=3):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=llm_responses)

        mock_executor = AsyncMock()
        if tool_results:
            mock_executor.execute = AsyncMock(side_effect=tool_results)
        else:
            mock_executor.execute = AsyncMock(return_value="ok")

        loop = LLMLoop(
            llm_client=mock_llm, tool_executor=mock_executor, max_iterations=max_iterations
        )
        return loop, mock_llm, mock_executor

    @pytest.mark.asyncio
    async def test_structured_tool_calls_at_max_iter_forces_synthesis(self):
        """When max_iterations is hit and the last response was a structured tool call
        (has_tool_calls=True, content='', raw_tool_call_text=''), a final synthesis
        call should be made so the user gets text, not an empty string."""
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {}}]
        # 3 iterations of tool calls, then synthesis response
        responses = [
            _tool_response(tool_calls) for _ in range(3)
        ] + [_text_response("Here is my final answer.")]
        loop, mock_llm, _ = self._make_loop(responses, max_iterations=3)

        result = await loop.run([{"role": "user", "content": "find info"}])

        assert result.stopped_reason == "max_iterations"
        assert result.response_text == "Here is my final answer."
        # 3 loop iterations + 1 synthesis call = 4 total LLM calls
        assert mock_llm.complete.call_count == 4
        # The synthesis call should have tools=None and tool_choice="none"
        synthesis_call = mock_llm.complete.call_args_list[-1]
        assert synthesis_call.kwargs.get("tools") is None
        assert synthesis_call.kwargs.get("tool_choice") == "none"

    @pytest.mark.asyncio
    async def test_raw_tool_call_text_at_max_iter_forces_synthesis(self):
        """Original behavior: raw_tool_call_text (XML-parsed) tool calls also
        trigger synthesis at max_iterations."""
        raw_response = LLMResponse(
            content="",
            tool_calls=[{"id": "tc_1", "name": "search", "arguments": {}}],
            raw_tool_call_text="<tool_call>search</tool_call>",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
        )
        responses = [raw_response] * 3 + [_text_response("Synthesized.")]
        loop, mock_llm, _ = self._make_loop(responses, max_iterations=3)

        result = await loop.run([{"role": "user", "content": "go"}])

        assert result.stopped_reason == "max_iterations"
        assert result.response_text == "Synthesized."
        assert mock_llm.complete.call_count == 4

    @pytest.mark.asyncio
    async def test_no_synthesis_when_last_response_has_content(self):
        """If max_iterations is hit but last response already has content
        (e.g., text + tool calls), no extra synthesis call is needed."""
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {}}]
        responses = [
            _tool_response(tool_calls),
            _tool_response(tool_calls),
            _tool_response(tool_calls, content="I also have text"),
        ]
        loop, mock_llm, _ = self._make_loop(responses, max_iterations=3)

        result = await loop.run([{"role": "user", "content": "go"}])

        assert result.stopped_reason == "max_iterations"
        assert result.response_text == "I also have text"
        # No extra synthesis call — 3 iterations only
        assert mock_llm.complete.call_count == 3
