"""End-to-end tests for A2A round-trip: tool call → HTTP → response → session recording."""

import json
import os
import tempfile
from itertools import groupby
from operator import attrgetter
from unittest.mock import AsyncMock, patch

import pytest

from runtime.llm import LLMResponse, LLMLoop
from runtime.llm.context_bridge import ContextBridge
from runtime.session.session import Session, SessionConfig
from sr2.a2a.client import A2AClientTool, A2AToolConfig
from runtime.tool_executor import ToolExecutor


def _text_response(content="Hello"):
    return LLMResponse(
        content=content, tool_calls=[], input_tokens=100,
        output_tokens=50, cached_tokens=0, model="test",
    )


def _tool_response(tool_calls, content=""):
    return LLMResponse(
        content=content, tool_calls=tool_calls, input_tokens=100,
        output_tokens=50, cached_tokens=0, model="test",
    )


class TestA2ARoundTrip:
    """Tests that verify A2A tool results flow correctly through the LLM loop."""

    @pytest.mark.asyncio
    async def test_a2a_tool_result_in_llm_messages(self):
        """A2A tool result is appended to messages so the LLM sees it."""
        a2a_response = "The remote agent says: task completed successfully."

        async def mock_http(url, payload, timeout):
            return {"status": "completed", "result": a2a_response}

        tool = A2AClientTool(
            config=A2AToolConfig(name="ask_agent", target_url="http://remote:8008"),
            http_callable=mock_http,
        )
        executor = ToolExecutor()
        executor.register("ask_agent", tool)

        tool_calls = [{"id": "tc_1", "name": "ask_agent", "arguments": {"message": "Do task"}}]
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[
            _tool_response(tool_calls),
            _text_response("Based on the remote agent: task completed."),
        ])

        loop = LLMLoop(llm_client=mock_llm, tool_executor=executor)
        messages = [{"role": "user", "content": "Ask the other agent to do the task"}]
        result = await loop.run(messages)

        # The LLM should have been called twice
        assert result.iterations == 2
        assert result.stopped_reason == "complete"

        # The tool result should be in the messages
        tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == a2a_response

        # The tool call record should have the A2A response
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result == a2a_response
        assert result.tool_calls[0].success is True

    @pytest.mark.asyncio
    async def test_a2a_error_visible_in_tool_result(self):
        """A2A HTTP error produces an error string visible to the LLM."""

        async def mock_http(url, payload, timeout):
            raise ConnectionError("Connection refused")

        tool = A2AClientTool(
            config=A2AToolConfig(name="ask_agent", target_url="http://remote:8008"),
            http_callable=mock_http,
        )
        executor = ToolExecutor()
        executor.register("ask_agent", tool)

        tool_calls = [{"id": "tc_1", "name": "ask_agent", "arguments": {"message": "Hello"}}]
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[
            _tool_response(tool_calls),
            _text_response("The remote agent is unreachable."),
        ])

        loop = LLMLoop(llm_client=mock_llm, tool_executor=executor)
        messages = [{"role": "user", "content": "Call remote"}]
        result = await loop.run(messages)

        # Error should be in the tool result
        tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
        assert "A2A call failed" in tool_result_msgs[0]["content"]
        assert "Connection refused" in tool_result_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_a2a_empty_response_visible(self):
        """Empty A2A response is still passed to the LLM (not silently dropped)."""

        async def mock_http(url, payload, timeout):
            return {"status": "completed", "result": ""}

        tool = A2AClientTool(
            config=A2AToolConfig(name="ask_agent", target_url="http://remote:8008"),
            http_callable=mock_http,
        )
        executor = ToolExecutor()
        executor.register("ask_agent", tool)

        tool_calls = [{"id": "tc_1", "name": "ask_agent", "arguments": {"message": "Hello"}}]
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[
            _tool_response(tool_calls),
            _text_response("Remote agent returned empty response."),
        ])

        loop = LLMLoop(llm_client=mock_llm, tool_executor=executor)
        messages = [{"role": "user", "content": "Ask"}]
        result = await loop.run(messages)

        # Empty string should be the tool result (not dropped)
        tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == ""


class TestSessionToolCallGrouping:
    """Tests that tool calls are recorded correctly grouped by LLM iteration."""

    def test_single_tool_call_recorded_correctly(self):
        """Single tool call creates correct assistant + tool_result pair."""
        session = Session("test")
        session.add_tool_calls_grouped([
            ("search", {"q": "test"}, "found it", "tc_1"),
        ])

        assert len(session.turns) == 2
        assert session.turns[0]["role"] == "assistant"
        assert len(session.turns[0]["tool_calls"]) == 1
        assert session.turns[0]["tool_calls"][0]["function"]["name"] == "search"
        assert session.turns[1]["role"] == "tool_result"
        assert session.turns[1]["content"] == "found it"

    def test_parallel_tool_calls_grouped_in_one_assistant_message(self):
        """Multiple tool calls from same iteration share one assistant message."""
        session = Session("test")
        session.add_tool_calls_grouped([
            ("search", {"q": "a"}, "result_a", "tc_1"),
            ("read", {"file": "b.txt"}, "result_b", "tc_2"),
        ])

        # Should have 3 turns: 1 assistant (with 2 tool_calls) + 2 tool_results
        assert len(session.turns) == 3

        assistant_msg = session.turns[0]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["tool_calls"]) == 2
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "search"
        assert assistant_msg["tool_calls"][1]["function"]["name"] == "read"

        assert session.turns[1]["role"] == "tool_result"
        assert session.turns[1]["tool_call_id"] == "tc_1"
        assert session.turns[1]["content"] == "result_a"

        assert session.turns[2]["role"] == "tool_result"
        assert session.turns[2]["tool_call_id"] == "tc_2"
        assert session.turns[2]["content"] == "result_b"

    def test_grouped_tool_calls_reconstruct_valid_llm_messages(self):
        """Session turns from grouped tool calls produce valid LLM messages."""
        session = Session("test")
        session.add_user_message("Do parallel tasks")
        session.add_tool_calls_grouped([
            ("search", {"q": "a"}, "result_a", "tc_1"),
            ("a2a_call", {"message": "hello"}, "remote response", "tc_2"),
        ])
        session.add_assistant_message("Tasks complete.")

        bridge = ContextBridge()
        from sr2.pipeline.engine import CompiledContext

        compiled = CompiledContext(layers={}, content="", tokens=0)
        messages = bridge.build_messages(compiled, session.turns)

        # user, assistant(tool_calls), tool, tool, assistant
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 2
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "tc_1"
        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "tc_2"
        assert messages[3]["content"] == "remote response"
        assert messages[4]["role"] == "assistant"
        assert messages[4]["content"] == "Tasks complete."

    def test_old_add_tool_call_still_works(self):
        """Legacy add_tool_call method still works for backward compatibility."""
        session = Session("test")
        session.add_tool_call("search", {"q": "test"}, "found", "tc_1")

        assert len(session.turns) == 2
        assert session.turns[0]["role"] == "assistant"
        assert session.turns[1]["role"] == "tool_result"

    def test_multi_iteration_tool_calls_produce_separate_groups(self):
        """Tool calls across multiple iterations each get their own assistant message."""
        session = Session("test")

        # Iteration 0: two parallel tool calls
        session.add_tool_calls_grouped([
            ("search", {"q": "a"}, "found_a", "tc_1"),
            ("search", {"q": "b"}, "found_b", "tc_2"),
        ])
        # Iteration 1: one tool call
        session.add_tool_calls_grouped([
            ("read", {"file": "c.txt"}, "file_content", "tc_3"),
        ])
        session.add_assistant_message("All done")

        # 3 + 2 + 2 + 1 = assistant(2tc) + 2tr + assistant(1tc) + 1tr + assistant
        assert len(session.turns) == 6

        # First group
        assert len(session.turns[0]["tool_calls"]) == 2
        # Second group
        assert len(session.turns[3]["tool_calls"]) == 1


class TestAgentA2AIntegration:
    """Agent-level integration: _handle_trigger → A2A tool → session → valid reconstruction."""

    @staticmethod
    def _setup_config_dir() -> str:
        tmpdir = tempfile.mkdtemp()
        iface_dir = os.path.join(tmpdir, "interfaces")
        os.makedirs(iface_dir)

        with open(os.path.join(tmpdir, "agent.yaml"), "w") as f:
            f.write("""
pipeline:
  token_budget: 8000
  compaction:
    enabled: false
    raw_window: 5
    rules: []
  summarization:
    enabled: false
  retrieval:
    enabled: false
  intent_detection:
    enabled: false
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config
runtime:
  llm:
    model:
      name: "test-model"
    fast_model:
      name: "test-fast-model"
""")

        with open(os.path.join(iface_dir, "user_message.yaml"), "w") as f:
            f.write(f"""
extends: {tmpdir}/agent.yaml
pipeline:
  token_budget: 8000
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config
""")

        return tmpdir

    @pytest.mark.asyncio
    async def test_a2a_tool_call_recorded_in_session_correctly(self):
        """Full agent flow: LLM calls A2A tool → response recorded → session valid."""
        from runtime.agent import Agent, AgentConfig
        from runtime.plugins.base import TriggerContext

        config_dir = self._setup_config_dir()
        a2a_response = "Remote agent completed the analysis."

        with patch("runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")

            # First call: LLM calls A2A tool. Second call: LLM returns text.
            mock_llm.complete = AsyncMock(side_effect=[
                LLMResponse(
                    content="Let me ask the remote agent.",
                    tool_calls=[{
                        "id": "call_a2a_1",
                        "name": "ask_remote",
                        "arguments": {"message": "Analyze this data"},
                    }],
                    input_tokens=200, output_tokens=80,
                ),
                LLMResponse(
                    content=f"The remote agent said: {a2a_response}",
                    tool_calls=[],
                    input_tokens=300, output_tokens=100,
                ),
            ])

            agent = Agent(config=AgentConfig(name="test", config_dir=config_dir))
            agent._loop._llm = mock_llm

            # Register a mock A2A tool
            async def mock_http(url, payload, timeout):
                return {"status": "completed", "result": a2a_response}

            a2a_tool = A2AClientTool(
                config=A2AToolConfig(
                    name="ask_remote",
                    target_url="http://remote:8008",
                    description="Ask remote agent",
                ),
                http_callable=mock_http,
            )
            agent._tool_executor.register("ask_remote", a2a_tool)
            agent._a2a_client_tools.append(a2a_tool)

            # Send a message that triggers A2A
            trigger = TriggerContext(
                interface_name="user_message",
                plugin_name="api",
                session_name="test_a2a_session",
                session_lifecycle="persistent",
                input_data="Ask the remote agent to analyze the data",
            )
            response = await agent._handle_trigger(trigger)

        # Response should incorporate the A2A result
        assert a2a_response in response

        # Session should have valid structure
        session = agent._sessions.get("test_a2a_session")
        assert session is not None

        # Expected turns: user + assistant(tool_call) + tool_result + assistant(final)
        assert session.turn_count == 4

        # Verify structure
        assert session.turns[0]["role"] == "user"
        assert session.turns[1]["role"] == "assistant"
        assert session.turns[1]["tool_calls"] is not None
        assert len(session.turns[1]["tool_calls"]) == 1
        assert session.turns[1]["tool_calls"][0]["function"]["name"] == "ask_remote"
        assert session.turns[2]["role"] == "tool_result"
        assert session.turns[2]["content"] == a2a_response
        assert session.turns[2]["tool_call_id"] == "call_a2a_1"
        assert session.turns[3]["role"] == "assistant"
        assert a2a_response in session.turns[3]["content"]

    @pytest.mark.asyncio
    async def test_session_with_a2a_reconstructs_valid_messages_for_next_turn(self):
        """After A2A call, session turns produce valid messages for the next LLM call."""
        from runtime.agent import Agent, AgentConfig
        from runtime.plugins.base import TriggerContext

        config_dir = self._setup_config_dir()
        a2a_response = "Analysis complete: all systems nominal."

        with patch("runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")

            # Turn 1: A2A call
            # Turn 2: Follow-up question (no tools)
            mock_llm.complete = AsyncMock(side_effect=[
                # Turn 1: LLM calls A2A
                LLMResponse(
                    content="",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "ask_remote",
                        "arguments": {"message": "Check systems"},
                    }],
                    input_tokens=200, output_tokens=80,
                ),
                # Turn 1: LLM responds after seeing tool result
                LLMResponse(
                    content="Systems are nominal.",
                    tool_calls=[],
                    input_tokens=300, output_tokens=50,
                ),
                # Turn 2: Follow-up — LLM must see prior A2A result in context
                LLMResponse(
                    content="Yes, all systems checked out fine in the previous analysis.",
                    tool_calls=[],
                    input_tokens=400, output_tokens=60,
                ),
            ])

            agent = Agent(config=AgentConfig(name="test", config_dir=config_dir))
            agent._loop._llm = mock_llm

            async def mock_http(url, payload, timeout):
                return {"status": "completed", "result": a2a_response}

            a2a_tool = A2AClientTool(
                config=A2AToolConfig(name="ask_remote", target_url="http://remote:8008"),
                http_callable=mock_http,
            )
            agent._tool_executor.register("ask_remote", a2a_tool)
            agent._a2a_client_tools.append(a2a_tool)

            # Turn 1: triggers A2A
            trigger1 = TriggerContext(
                interface_name="user_message",
                plugin_name="api",
                session_name="multi_turn",
                session_lifecycle="persistent",
                input_data="Check all systems via the remote agent",
            )
            await agent._handle_trigger(trigger1)

            # Turn 2: follow-up (the LLM should see the A2A result in history)
            trigger2 = TriggerContext(
                interface_name="user_message",
                plugin_name="api",
                session_name="multi_turn",
                session_lifecycle="persistent",
                input_data="Were the systems OK?",
            )
            response2 = await agent._handle_trigger(trigger2)

        # The second LLM call should have received messages with the A2A result
        # Check the messages passed to the third complete() call (turn 2)
        third_call_args = mock_llm.complete.call_args_list[2]
        messages = third_call_args.kwargs["messages"]

        # Find the tool result in the messages
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == a2a_response

        # The assistant message with tool_calls should have exactly 1 tool_call
        assistant_tc_msgs = [
            m for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_tc_msgs) == 1
        assert len(assistant_tc_msgs[0]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_parallel_a2a_and_regular_tool_grouped_correctly(self):
        """When LLM calls A2A + regular tool in parallel, both are grouped in one assistant msg."""
        from runtime.agent import Agent, AgentConfig
        from runtime.plugins.base import TriggerContext
        from runtime.tool_executor import SimpleTool

        config_dir = self._setup_config_dir()

        with patch("runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")

            mock_llm.complete = AsyncMock(side_effect=[
                # LLM calls both tools in parallel
                LLMResponse(
                    content="",
                    tool_calls=[
                        {"id": "call_a2a", "name": "ask_remote",
                         "arguments": {"message": "Get status"}},
                        {"id": "call_search", "name": "local_search",
                         "arguments": {"query": "logs"}},
                    ],
                    input_tokens=200, output_tokens=80,
                ),
                # LLM responds with combined results
                LLMResponse(
                    content="Remote says OK, local logs are clean.",
                    tool_calls=[],
                    input_tokens=400, output_tokens=60,
                ),
            ])

            agent = Agent(config=AgentConfig(name="test", config_dir=config_dir))
            agent._loop._llm = mock_llm

            async def mock_http(url, payload, timeout):
                return {"status": "completed", "result": "All systems go"}

            a2a_tool = A2AClientTool(
                config=A2AToolConfig(name="ask_remote", target_url="http://remote:8008"),
                http_callable=mock_http,
            )
            agent._tool_executor.register("ask_remote", a2a_tool)
            agent._a2a_client_tools.append(a2a_tool)

            async def search_fn(query=""):
                return f"Found: {query} results"

            agent._tool_executor.register("local_search", SimpleTool(search_fn))

            trigger = TriggerContext(
                interface_name="user_message",
                plugin_name="api",
                session_name="parallel_test",
                session_lifecycle="persistent",
                input_data="Check remote status and search local logs",
            )
            await agent._handle_trigger(trigger)

        session = agent._sessions.get("parallel_test")

        # Expected: user + assistant(2 tool_calls) + tool_result + tool_result + assistant
        assert session.turn_count == 5

        # The assistant message should have BOTH tool calls grouped
        assistant_tc = session.turns[1]
        assert assistant_tc["role"] == "assistant"
        assert len(assistant_tc["tool_calls"]) == 2

        tool_names = {tc["function"]["name"] for tc in assistant_tc["tool_calls"]}
        assert tool_names == {"ask_remote", "local_search"}

        # Both tool results should follow
        assert session.turns[2]["role"] == "tool_result"
        assert session.turns[3]["role"] == "tool_result"
