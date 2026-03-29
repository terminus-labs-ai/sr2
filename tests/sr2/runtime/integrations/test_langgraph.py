"""Tests for SR2Node LangGraph integration."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from sr2.runtime.config import AgentConfig, ModelConfig, PersonaConfig
from sr2.runtime.integrations.langgraph.node import SR2Node
from sr2.runtime.integrations.langgraph.state import SR2GraphState
from sr2.runtime.result import RuntimeMetrics, RuntimeResult


@pytest.fixture
def agent_config():
    return AgentConfig(
        name="test-agent",
        model=ModelConfig(name="test-model"),
        persona=PersonaConfig(system_prompt="You are a test agent."),
    )


@pytest.fixture
def mock_runtime():
    rt = AsyncMock()
    rt.name = "test-agent"
    rt.execute = AsyncMock(
        return_value=RuntimeResult(
            output="test output",
            metrics=RuntimeMetrics(total_tokens=100),
        )
    )
    rt.reset = AsyncMock()
    return rt


@pytest.fixture
def node(mock_runtime, agent_config):
    with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
        mock_cls.return_value = mock_runtime
        n = SR2Node(agent_config)
        return n


class TestSR2NodeConstruction:
    def test_from_agent_config(self, mock_runtime, agent_config):
        with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
            mock_cls.return_value = mock_runtime
            n = SR2Node(agent_config)
            mock_cls.assert_called_once_with(agent_config)
            assert n.runtime is mock_runtime

    def test_from_dict(self, mock_runtime):
        config_dict = {
            "name": "dict-agent",
            "model": {"name": "test-model"},
            "persona": {"system_prompt": "test"},
        }
        with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
            mock_cls.from_dict.return_value = mock_runtime
            n = SR2Node(config_dict)
            mock_cls.from_dict.assert_called_once_with(config_dict)
            assert n.runtime is mock_runtime

    def test_from_path_string(self, mock_runtime):
        with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
            mock_cls.from_config.return_value = mock_runtime
            n = SR2Node("/path/to/config.yaml")
            mock_cls.from_config.assert_called_once_with("/path/to/config.yaml")
            assert n.runtime is mock_runtime

    def test_unsupported_config_type(self):
        with pytest.raises(TypeError, match="Unsupported config type"):
            SR2Node(42)


class TestSR2NodeExecute:
    async def test_execute_reads_correct_state_keys(self, node, mock_runtime):
        state = {"current_task": "do research", "prior_output": "prior context"}

        await node._execute(state)

        mock_runtime.execute.assert_called_once_with(
            task="do research", context="prior context"
        )

    async def test_execute_defaults_missing_keys(self, node, mock_runtime):
        state: dict = {}

        await node._execute(state)

        mock_runtime.execute.assert_called_once_with(task="", context=None)

    async def test_execute_writes_output(self, node):
        state = {"current_task": "summarize"}

        result = await node._execute(state)

        assert result["prior_output"] == "test output"
        assert result["outputs"]["test-agent"] == "test output"

    async def test_execute_accumulates_metrics(self, node):
        state = {
            "current_task": "analyze",
            "metrics": {"other-agent": {"total_tokens": 50}},
        }

        result = await node._execute(state)

        assert "other-agent" in result["metrics"]
        assert result["metrics"]["test-agent"]["total_tokens"] == 100

    async def test_execute_accumulates_outputs(self, node):
        state = {
            "current_task": "step2",
            "outputs": {"step1-agent": "step1 result"},
        }

        result = await node._execute(state)

        assert result["outputs"]["step1-agent"] == "step1 result"
        assert result["outputs"]["test-agent"] == "test output"


class TestSR2NodeCustomKeys:
    async def test_custom_task_key(self, mock_runtime, agent_config):
        with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
            mock_cls.return_value = mock_runtime
            n = SR2Node(agent_config, task_key="instruction")

        state = {"instruction": "custom task"}
        await n._execute(state)

        mock_runtime.execute.assert_called_once_with(
            task="custom task", context=None
        )

    async def test_custom_context_key(self, mock_runtime, agent_config):
        with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
            mock_cls.return_value = mock_runtime
            n = SR2Node(agent_config, context_key="background")

        state = {"current_task": "t", "background": "custom context"}
        await n._execute(state)

        mock_runtime.execute.assert_called_once_with(
            task="t", context="custom context"
        )

    async def test_custom_output_key(self, mock_runtime, agent_config):
        with patch("sr2.runtime.integrations.langgraph.node.SR2Runtime") as mock_cls:
            mock_cls.return_value = mock_runtime
            n = SR2Node(agent_config, output_key="research_result")

        state = {"current_task": "t"}
        result = await n._execute(state)

        assert "research_result" in result["outputs"]
        assert "research_result" in result["metrics"]


class TestSR2NodeDelegation:
    async def test_reset_propagates(self, node, mock_runtime):
        await node.reset()
        mock_runtime.reset.assert_awaited_once()

    def test_name_property(self, node):
        assert node.name == "test-agent"

    async def test_acall_delegates_to_execute(self, node, mock_runtime):
        state = {"current_task": "test task"}

        result = await node.__acall__(state)

        mock_runtime.execute.assert_called_once()
        assert result["prior_output"] == "test output"


class TestSR2GraphState:
    def test_state_is_typed_dict(self):
        state: SR2GraphState = {
            "current_task": "research",
            "prior_output": None,
            "outputs": {},
            "metrics": {},
            "iteration": 0,
            "metadata": {},
        }
        assert state["current_task"] == "research"

    def test_state_total_false(self):
        """SR2GraphState uses total=False so all keys are optional."""
        state: SR2GraphState = {}
        assert isinstance(state, dict)
