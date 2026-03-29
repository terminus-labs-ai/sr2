"""Tests for LangGraph integration — SR2Node wrapping the runtime Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from runtime.integrations.langgraph.node import SR2Node
from runtime.integrations.langgraph.state import SR2GraphState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent():
    """Mock Agent instance with handle_user_message."""
    agent = AsyncMock()
    agent.handle_user_message = AsyncMock(return_value="agent response")
    agent.start = AsyncMock()
    agent.shutdown = AsyncMock()
    return agent


@pytest.fixture
def node(mock_agent):
    """SR2Node with a mocked Agent."""
    with patch("runtime.integrations.langgraph.node.Agent") as mock_cls:
        mock_cls.return_value = mock_agent
        n = SR2Node("researcher", "/tmp/fake-config")
    return n


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_name_property(self, node):
        assert node.name == "researcher"

    def test_default_output_key_is_name(self, node):
        assert node._output_key == "researcher"

    def test_custom_output_key(self):
        with patch("runtime.integrations.langgraph.node.Agent"):
            n = SR2Node("x", "/tmp", output_key="custom")
        assert n._output_key == "custom"

    def test_agent_config_constructed(self):
        with patch("runtime.integrations.langgraph.node.Agent") as mock_cls:
            SR2Node("test", "/some/dir", defaults_path="custom/defaults.yaml")
            cfg = mock_cls.call_args[0][0]
            assert cfg.name == "test"
            assert cfg.config_dir == "/some/dir"
            assert cfg.defaults_path == "custom/defaults.yaml"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_simple_task(self, node, mock_agent):
        state = {"current_task": "hello"}
        result = await node._execute(state)

        mock_agent.handle_user_message.assert_called_once()
        call_msg = mock_agent.handle_user_message.call_args[0][0]
        assert call_msg == "hello"
        assert result["prior_output"] == "agent response"
        assert result["outputs"]["researcher"] == "agent response"

    async def test_with_prior_context(self, node, mock_agent):
        state = {"current_task": "code it", "prior_output": "research findings"}
        await node._execute(state)

        call_msg = mock_agent.handle_user_message.call_args[0][0]
        assert "research findings" in call_msg
        assert "code it" in call_msg

    async def test_empty_task(self, node, mock_agent):
        state = {}
        result = await node._execute(state)

        call_msg = mock_agent.handle_user_message.call_args[0][0]
        assert call_msg == ""
        assert result["prior_output"] == "agent response"

    async def test_preserves_existing_outputs(self, node, mock_agent):
        state = {
            "current_task": "task",
            "outputs": {"prior_node": "prior result"},
        }
        result = await node._execute(state)

        assert result["outputs"]["prior_node"] == "prior result"
        assert result["outputs"]["researcher"] == "agent response"

    async def test_auto_starts_agent(self, node, mock_agent):
        assert not node._started
        await node._execute({"current_task": "hi"})
        mock_agent.start.assert_called_once()
        assert node._started

    async def test_does_not_restart_if_already_started(self, node, mock_agent):
        node._started = True
        await node._execute({"current_task": "hi"})
        mock_agent.start.assert_not_called()

    async def test_session_id_uses_node_name(self, node, mock_agent):
        await node._execute({"current_task": "hi"})
        session_id = mock_agent.handle_user_message.call_args[0][1]
        assert session_id == "langgraph-researcher"


# ---------------------------------------------------------------------------
# Custom keys
# ---------------------------------------------------------------------------


class TestCustomKeys:
    async def test_custom_task_key(self, mock_agent):
        with patch("runtime.integrations.langgraph.node.Agent") as mock_cls:
            mock_cls.return_value = mock_agent
            n = SR2Node("x", "/tmp", task_key="objective")

        await n._execute({"objective": "custom task"})
        call_msg = mock_agent.handle_user_message.call_args[0][0]
        assert call_msg == "custom task"

    async def test_custom_context_key(self, mock_agent):
        with patch("runtime.integrations.langgraph.node.Agent") as mock_cls:
            mock_cls.return_value = mock_agent
            n = SR2Node("x", "/tmp", context_key="prev")

        await n._execute({"current_task": "task", "prev": "ctx"})
        call_msg = mock_agent.handle_user_message.call_args[0][0]
        assert "ctx" in call_msg

    async def test_custom_output_key(self, mock_agent):
        with patch("runtime.integrations.langgraph.node.Agent") as mock_cls:
            mock_cls.return_value = mock_agent
            n = SR2Node("x", "/tmp", output_key="result")

        result = await n._execute({"current_task": "task"})
        assert "result" in result["outputs"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start(self, node, mock_agent):
        await node.start()
        mock_agent.start.assert_called_once()
        assert node._started

    async def test_start_idempotent(self, node, mock_agent):
        await node.start()
        await node.start()
        mock_agent.start.assert_called_once()

    async def test_shutdown(self, node, mock_agent):
        node._started = True
        await node.shutdown()
        mock_agent.shutdown.assert_called_once()
        assert not node._started

    async def test_shutdown_when_not_started(self, node, mock_agent):
        await node.shutdown()
        mock_agent.shutdown.assert_not_called()

    async def test_reset(self, node, mock_agent):
        node._started = True
        await node.reset()
        mock_agent.shutdown.assert_called_once()
        assert not node._started


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------


class TestAsyncInterface:
    async def test_acall(self, node, mock_agent):
        result = await node.__acall__({"current_task": "hello"})
        assert result["prior_output"] == "agent response"


# ---------------------------------------------------------------------------
# State TypedDict
# ---------------------------------------------------------------------------


class TestSR2GraphState:
    def test_is_typed_dict(self):
        state: SR2GraphState = {"current_task": "test"}
        assert state["current_task"] == "test"

    def test_all_keys_optional(self):
        state: SR2GraphState = {}
        assert isinstance(state, dict)
