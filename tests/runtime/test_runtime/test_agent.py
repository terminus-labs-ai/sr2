"""Tests for the Agent class."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_runtime.agent import Agent, AgentConfig
from sr2_runtime.plugins.base import TriggerContext
from sr2_runtime.tool_executor import SimpleTool

# Resolve the project root (where configs/defaults.yaml lives)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _setup_config_dir() -> str:
    """Create a temp directory with minimal agent + interface configs."""
    tmpdir = tempfile.mkdtemp()
    iface_dir = os.path.join(tmpdir, "interfaces")
    os.makedirs(iface_dir)

    # agent.yaml — self-contained, no extends to avoid defaults.yaml compaction issues
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

    # user_message.yaml
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


@pytest.fixture
def config_dir():
    return _setup_config_dir()


@pytest.fixture
def agent(config_dir):
    """Create an Agent with mocked LLM."""
    config = AgentConfig(
        name="test_agent",
        config_dir=config_dir,
    )

    # Mock the LLM so we don't need real API keys
    with patch("sr2_runtime.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
        mock_llm.fast_complete = AsyncMock(return_value="extracted")
        mock_llm.complete = AsyncMock(return_value=MagicMock(
            content="Test response",
            tool_calls=[],
            has_tool_calls=False,
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            model="test",
        ))

        agent = Agent(config=config)
        # Replace the loop's LLM with our mock too
        agent._loop._llm = mock_llm
        yield agent


class TestAgent:
    """Tests for the Agent class."""

    def test_agent_initializes(self, agent):
        assert agent._name == "test_agent"
        assert agent._sessions is not None
        assert agent._plugins is not None
        assert agent._plugin_registry is not None

    def test_post_to_session_registered(self, agent):
        assert agent._tool_executor.has("post_to_session")

    @pytest.mark.asyncio
    async def test_handle_user_message_returns_string(self, agent):
        from sr2_runtime.llm import LLMResponse

        agent._loop._llm.complete = AsyncMock(return_value=LLMResponse(
            content="Hello there!",
            input_tokens=100,
            output_tokens=50,
        ))

        response = await agent.handle_user_message("Hi", session_id="test_session")
        assert isinstance(response, str)
        assert response == "Hello there!"

    @pytest.mark.asyncio
    async def test_handle_user_message_adds_turns_to_session(self, agent):
        from sr2_runtime.llm import LLMResponse

        agent._loop._llm.complete = AsyncMock(return_value=LLMResponse(
            content="Reply",
            input_tokens=100,
            output_tokens=50,
        ))

        await agent.handle_user_message("Hello", session_id="s1")
        session = agent._sessions.get("s1")
        assert session is not None
        # Should have user message + assistant response
        assert session.turn_count >= 2

    def test_register_tool(self, agent):
        async def my_tool(**kwargs) -> str:
            return "result"

        agent.register_tool("my_tool", SimpleTool(my_tool))
        assert agent._tool_executor.has("my_tool")

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, agent):
        await agent.start()
        await agent.shutdown()

    def test_discover_interfaces(self, config_dir, agent):
        interfaces = agent._sr2._discover_interfaces(config_dir)
        assert "user_message" in interfaces

    @pytest.mark.asyncio
    async def test_post_processing_runs_async(self, agent):
        from sr2_runtime.llm import LLMResponse

        agent._loop._llm.complete = AsyncMock(return_value=LLMResponse(
            content="Response",
            input_tokens=100,
            output_tokens=50,
        ))

        # Mock post-processor to verify it's called
        agent._sr2._post_processor.process = AsyncMock()

        await agent.handle_user_message("Test", session_id="s2")
        # Give the fire-and-forget task a moment to run
        await asyncio.sleep(0.05)

        agent._sr2._post_processor.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_message_no_llm_override_uses_defaults(self, agent):
        """User message without llm overrides -> loop receives None overrides."""
        from sr2_runtime.llm import LLMResponse

        agent._loop._llm.complete = AsyncMock(return_value=LLMResponse(
            content="OK",
            input_tokens=100,
            output_tokens=50,
        ))
        agent._loop.run = AsyncMock(return_value=MagicMock(
            response_text="OK",
            tool_calls=[],
            iterations=1,
            total_tokens=150,
            cache_hit_rate=0.0,
        ))

        await agent.handle_user_message("Hi", session_id="s_default")

        call_kwargs = agent._loop.run.call_args
        # runtime.llm is inherited from parent and extracted as override
        # The override matches the base model, so it is functionally a no-op
        override = call_kwargs.kwargs.get("model_config_override")
        assert override is not None
        assert override.name == "test-model"

    @pytest.mark.asyncio
    async def test_handle_trigger_with_ephemeral_session(self, agent):
        from sr2_runtime.llm import LoopResult

        agent._loop.run = AsyncMock(return_value=LoopResult(
            response_text="done",
            iterations=1,
        ))

        trigger = TriggerContext(
            interface_name="user_message",
            plugin_name="timer",
            session_name="heartbeat",
            session_lifecycle="ephemeral",
            input_data="",
        )
        result = await agent._handle_trigger(trigger)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_handle_trigger_clear_session(self, agent):
        await agent._sessions.get_or_create("test_sess")
        trigger = TriggerContext(
            interface_name="user_message",
            plugin_name="telegram",
            session_name="test_sess",
            session_lifecycle="persistent",
            input_data="__clear_session__",
        )
        result = await agent._handle_trigger(trigger)
        assert result == "Session cleared."
        assert agent._sessions.get("test_sess") is None


def _setup_config_dir_with_heartbeat() -> str:
    """Create a temp directory with agent + heartbeat interface with LLM overrides."""
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
      name: "agent-default-model"
      api_base: "http://localhost:11434"
""")

    # heartbeat_email.yaml — with LLM overrides
    with open(os.path.join(iface_dir, "heartbeat_email.yaml"), "w") as f:
        f.write(f"""
extends: {tmpdir}/agent.yaml
pipeline:
  token_budget: 3000
  llm:
    model:
      name: "ollama/qwen2.5-coder:7b"
      api_base: "http://localhost:11435"
      max_tokens: 500
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config
""")

    # user_message.yaml — no LLM overrides
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


@pytest.fixture
def agent_with_heartbeat():
    """Agent with a heartbeat interface that has LLM overrides."""
    config_dir = _setup_config_dir_with_heartbeat()
    config = AgentConfig(name="test_override", config_dir=config_dir)

    with patch("sr2_runtime.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
        mock_llm.fast_complete = AsyncMock(return_value="extracted")
        mock_llm.complete = AsyncMock(return_value=MagicMock(
            content="OK",
            tool_calls=[],
            has_tool_calls=False,
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            model="test",
        ))
        agent = Agent(config=config)
        agent._loop._llm = mock_llm
        yield agent


class TestModelOverrides:
    """Tests for per-interface LLM model overrides (HOTFIX-01)."""

    @pytest.mark.asyncio
    async def test_heartbeat_with_llm_overrides(self, agent_with_heartbeat):
        """Heartbeat with llm.model + llm.api_base override -> loop receives both."""
        from sr2_runtime.llm import LoopResult

        agent = agent_with_heartbeat
        agent._loop.run = AsyncMock(return_value=LoopResult(
            response_text='{"action_needed": false}',
            iterations=1,
        ))

        # Trigger via _handle_trigger with heartbeat_email interface
        trigger = TriggerContext(
            interface_name="heartbeat_email",
            plugin_name="timer",
            session_name="hb_email",
            session_lifecycle="ephemeral",
            input_data="",
        )
        await agent._handle_trigger(trigger)

        call_kwargs = agent._loop.run.call_args
        override = call_kwargs.kwargs["model_config_override"]
        assert override is not None
        assert override.name == "ollama/qwen2.5-coder:7b"
        assert override.api_base == "http://localhost:11435"
        assert override.max_tokens == 500

    @pytest.mark.asyncio
    async def test_user_message_falls_back_to_agent_api_base(self, agent_with_heartbeat):
        """User message without llm.api_base -> falls back to agent default."""
        from sr2_runtime.llm import LoopResult

        agent = agent_with_heartbeat
        agent._loop.run = AsyncMock(return_value=LoopResult(
            response_text="Hello!",
            iterations=1,
        ))

        await agent.handle_user_message("Hi", session_id="test")

        call_kwargs = agent._loop.run.call_args
        # runtime.llm is inherited from parent and extracted as override
        # The override matches the base model, so it is functionally a no-op
        override = call_kwargs.kwargs["model_config_override"]
        assert override is not None
        assert override.name == "agent-default-model"


class TestModelParamsFlow:
    """Tests for model_params flowing from config through agent to loop."""

    @pytest.mark.asyncio
    async def test_default_model_params_stored_on_agent(self):
        """Agent stores default model_params from runtime config."""
        config_dir = _setup_config_dir()
        # Write agent.yaml with model_params
        with open(os.path.join(config_dir, "agent.yaml"), "w") as f:
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
      model_params:
        temperature: 0.3
        top_p: 0.95
""")

        config = AgentConfig(name="test_params", config_dir=config_dir)
        with patch("sr2_runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")
            agent = Agent(config=config)

        assert agent._agent_config.runtime.llm.model.model_params.temperature == 0.3
        assert agent._agent_config.runtime.llm.model.model_params.top_p == 0.95
        assert agent._loop._default_model_config.model_params.temperature == 0.3

    @pytest.mark.asyncio
    async def test_model_params_passed_to_loop(self):
        """model_config_override from interface config flows to loop.run()."""
        from sr2_runtime.llm import LoopResult

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
      model_params:
        temperature: 0.7
""")

        # Interface with model override including model_params
        with open(os.path.join(iface_dir, "creative.yaml"), "w") as f:
            f.write(f"""
extends: {tmpdir}/agent.yaml
pipeline:
  token_budget: 8000
  llm:
    model:
      model_params:
        temperature: 1.5
        top_p: 0.8
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config
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

        config = AgentConfig(name="test_params_flow", config_dir=tmpdir)
        with patch("sr2_runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")
            agent = Agent(config=config)
            agent._loop._llm = mock_llm
            agent._loop.run = AsyncMock(return_value=LoopResult(
                response_text="Creative output",
                iterations=1,
            ))

        trigger = TriggerContext(
            interface_name="creative",
            plugin_name="api",
            session_name="creative_session",
            session_lifecycle="ephemeral",
            input_data="Write something creative",
        )
        await agent._handle_trigger(trigger)

        call_kwargs = agent._loop.run.call_args
        override = call_kwargs.kwargs["model_config_override"]
        assert override is not None
        assert override.model_params.temperature == 1.5
        assert override.model_params.top_p == 0.8
