"""Tests for MCP prompt discovery and retrieval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime.config import MCPServerConfig
from runtime.mcp.client import MCPGetPromptHandler, MCPManager


def _make_mock_prompt(name, description="", arguments=None):
    """Create a mock MCP prompt."""
    p = MagicMock()
    p.name = name
    p.description = description
    p.arguments = arguments
    return p


def _make_mock_prompt_message(role, text):
    """Create a mock prompt message."""
    msg = MagicMock()
    msg.role = role
    content = MagicMock()
    content.text = text
    msg.content = content
    return msg


class TestMCPManagerPrompts:
    """Tests for prompt discovery and access on MCPManager."""

    @pytest.mark.asyncio
    async def test_discover_prompts_stores_info(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="prompts", url="test")

        arg = MagicMock()
        arg.name = "language"
        arg.description = "Programming language"
        arg.required = True

        prompts = [_make_mock_prompt("code_review", "Review code", arguments=[arg])]
        session = MagicMock()
        session.list_prompts = AsyncMock(return_value=MagicMock(prompts=prompts))

        await mgr._discover_prompts(config, session)

        assert len(mgr._discovered_prompts["prompts"]) == 1
        assert mgr._discovered_prompts["prompts"][0]["name"] == "code_review"
        assert mgr._prompt_server_map["code_review"] == "prompts"

    @pytest.mark.asyncio
    async def test_discover_prompts_handles_no_support(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="no_prompts", url="test")
        session = MagicMock()
        session.list_prompts = AsyncMock(side_effect=Exception("Not supported"))

        await mgr._discover_prompts(config, session)

        assert mgr._discovered_prompts.get("no_prompts") is None

    @pytest.mark.asyncio
    async def test_list_prompts_all(self):
        mgr = MCPManager()
        mgr._discovered_prompts = {
            "s1": [{"name": "a"}],
            "s2": [{"name": "b"}],
        }
        result = await mgr.list_prompts()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_prompts_filtered(self):
        mgr = MCPManager()
        mgr._discovered_prompts = {
            "s1": [{"name": "a"}],
            "s2": [{"name": "b"}],
        }
        result = await mgr.list_prompts(server_name="s1")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_prompt_returns_content(self):
        mgr = MCPManager()
        mgr._prompt_server_map["code_review"] = "prompts"

        msg = _make_mock_prompt_message("user", "Review this Python code.")
        session = MagicMock()
        session.get_prompt = AsyncMock(return_value=MagicMock(messages=[msg]))
        # Pre-populate session for _get_session
        mgr._sessions["prompts"] = session
        mgr._last_activity["prompts"] = 0

        result = await mgr.get_prompt("code_review", {"language": "python"})
        assert "Review this Python code." in result
        session.get_prompt.assert_called_once_with("code_review", {"language": "python"})

    @pytest.mark.asyncio
    async def test_get_prompt_unknown_name_raises(self):
        mgr = MCPManager()

        with pytest.raises(KeyError, match="No MCP server has prompt"):
            await mgr.get_prompt("nonexistent")

    @pytest.mark.asyncio
    async def test_get_prompt_disconnected_server_raises(self):
        mgr = MCPManager()
        mgr._prompt_server_map["test"] = "gone"
        # No config registered, so _get_session will raise
        with pytest.raises(KeyError, match="No MCP server config"):
            await mgr.get_prompt("test")

    def test_get_prompt_tool_schemas(self):
        mgr = MCPManager()
        schemas = mgr.get_prompt_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "mcp_get_prompt"


class TestMCPGetPromptHandler:
    """Tests for MCPGetPromptHandler."""

    @pytest.mark.asyncio
    async def test_handler_calls_get_prompt(self):
        mgr = MCPManager()
        mgr.get_prompt = AsyncMock(return_value="[user] Review this code.")

        handler = MCPGetPromptHandler(mgr)
        result = await handler.execute(name="code_review", server="prompts", language="python")

        mgr.get_prompt.assert_called_once_with(
            "code_review", {"language": "python"}, server_name="prompts"
        )
        assert result == "[user] Review this code."
