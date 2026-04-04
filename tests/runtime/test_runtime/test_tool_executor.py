"""Tests for tool executor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2_runtime.session import SessionManager
from sr2_runtime.tool_executor import ToolExecutor, SimpleTool, PostToSessionTool


class TestToolExecutor:
    """Tests for the ToolExecutor."""

    @pytest.mark.asyncio
    async def test_register_and_execute(self):
        executor = ToolExecutor()

        async def search(query: str = "") -> str:
            return f"results for: {query}"

        executor.register("search", SimpleTool(search))
        result = await executor.execute("search", {"query": "test"})
        assert result == "results for: test"

    @pytest.mark.asyncio
    async def test_execute_unregistered_raises(self):
        executor = ToolExecutor()
        with pytest.raises(KeyError, match="No handler registered"):
            await executor.execute("unknown", {})

    @pytest.mark.asyncio
    async def test_execute_unregistered_lists_available_tools(self):
        """Error message should list available tools to help the model self-correct."""
        executor = ToolExecutor()
        handler = SimpleTool(AsyncMock(return_value="ok"))
        executor.register("search_repos", handler)
        executor.register("get_file", handler)

        with pytest.raises(KeyError, match="Available tools:.*get_file.*search_repos"):
            await executor.execute("list_repositories", {})

    @pytest.mark.asyncio
    async def test_call_count_tracks_per_tool(self):
        executor = ToolExecutor()

        async def noop(**kwargs) -> str:
            return "ok"

        executor.register("tool_a", SimpleTool(noop))
        executor.register("tool_b", SimpleTool(noop))

        await executor.execute("tool_a", {})
        await executor.execute("tool_a", {})
        await executor.execute("tool_b", {})

        assert executor.get_call_count("tool_a") == 2
        assert executor.get_call_count("tool_b") == 1

    @pytest.mark.asyncio
    async def test_total_calls(self):
        executor = ToolExecutor()

        async def noop(**kwargs) -> str:
            return "ok"

        executor.register("a", SimpleTool(noop))
        executor.register("b", SimpleTool(noop))

        await executor.execute("a", {})
        await executor.execute("b", {})
        await executor.execute("a", {})

        assert executor.total_calls == 3

    @pytest.mark.asyncio
    async def test_simple_tool_wraps_function(self):
        async def greet(name: str = "world") -> str:
            return f"hello {name}"

        tool = SimpleTool(greet)
        result = await tool.execute(name="claude")
        assert result == "hello claude"

    def test_registered_tools_lists_all(self):
        executor = ToolExecutor()

        async def noop(**kwargs) -> str:
            return "ok"

        executor.register("alpha", SimpleTool(noop))
        executor.register("beta", SimpleTool(noop))

        assert set(executor.registered_tools) == {"alpha", "beta"}


class TestPostToSessionTool:
    """Tests for the PostToSessionTool."""

    @pytest.mark.asyncio
    async def test_injects_message_into_session(self):
        mgr = SessionManager()
        tool = PostToSessionTool(mgr, {})
        result = await tool.execute(session="main_chat", message="Hello user!")
        assert "posted" in result

        session = mgr.get("main_chat")
        assert session is not None
        assert session.turns[-1]["content"] == "Hello user!"
        assert session.turns[-1]["injected"] is True
        assert session.turns[-1]["metadata"]["priority"] == "important"

    @pytest.mark.asyncio
    async def test_calls_plugin_send(self):
        mgr = SessionManager()
        mock_plugin = AsyncMock()
        tool = PostToSessionTool(mgr, {"main_chat": mock_plugin})

        result = await tool.execute(session="main_chat", message="Alert!")
        assert "posted" in result
        mock_plugin.send.assert_called_once_with("main_chat", "Alert!", {"priority": "important"})

    @pytest.mark.asyncio
    async def test_handles_plugin_send_failure(self):
        mgr = SessionManager()
        mock_plugin = AsyncMock()
        mock_plugin.send.side_effect = Exception("connection lost")
        tool = PostToSessionTool(mgr, {"main_chat": mock_plugin})

        result = await tool.execute(session="main_chat", message="Alert!")
        assert "delivery failed" in result
        # Message should still be in session
        session = mgr.get("main_chat")
        assert session.turns[-1]["content"] == "Alert!"

    @pytest.mark.asyncio
    async def test_no_plugin_still_injects(self):
        mgr = SessionManager()
        tool = PostToSessionTool(mgr, {})
        result = await tool.execute(session="orphan_session", message="FYI")
        assert "posted" in result
        session = mgr.get("orphan_session")
        assert session.turns[-1]["content"] == "FYI"

    @pytest.mark.asyncio
    async def test_missing_args_returns_error(self):
        mgr = SessionManager()
        tool = PostToSessionTool(mgr, {})
        result = await tool.execute()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_priority_passed_through(self):
        mgr = SessionManager()
        tool = PostToSessionTool(mgr, {})
        await tool.execute(session="s1", message="urgent!", priority="urgent")
        session = mgr.get("s1")
        assert session.turns[-1]["metadata"]["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_integrates_with_tool_executor(self):
        mgr = SessionManager()
        executor = ToolExecutor()
        tool = PostToSessionTool(mgr, {})
        executor.register("post_to_session", tool)

        result = await executor.execute("post_to_session", {"session": "chat", "message": "hi"})
        assert "posted" in result
        assert executor.get_call_count("post_to_session") == 1
