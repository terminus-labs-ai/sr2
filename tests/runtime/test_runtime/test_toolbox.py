"""Tests for the Toolbox meta-tool."""

import json
from unittest.mock import AsyncMock

import pytest

from sr2_runtime.toolbox import Toolbox, ToolboxEntry


class DummyToolHandler:
    def __init__(self, name="dummy", result="ok"):
        self._name = name
        self._result = result

    @property
    def tool_definition(self) -> dict:
        return {
            "name": self._name,
            "description": f"Does {self._name}",
            "parameters": {"type": "object", "properties": {}},
        }

    async def execute(self, **kwargs) -> str:
        return self._result


class TestToolboxEntry:
    """Tests for ToolboxEntry dataclass."""

    def test_tier_defaults_to_toolbox(self):
        """ToolboxEntry without explicit tier defaults to 'toolbox'."""
        entry = ToolboxEntry(
            name="test", one_liner="A test tool", handler=DummyToolHandler()
        )
        assert entry.tier == "toolbox"


class TestToolbox:
    """Tests for the Toolbox meta-tool."""

    def _make_entry(
        self, name="dummy", one_liner="Does stuff", result="ok", tier="toolbox"
    ) -> ToolboxEntry:
        return ToolboxEntry(
            name=name,
            one_liner=one_liner,
            handler=DummyToolHandler(name=name, result=result),
            tier=tier,
        )

    # --- register / has ---

    def test_register_and_has(self):
        """register() adds to registry, has() reflects it."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="alpha"))
        assert toolbox.has("alpha") is True
        assert toolbox.has("nonexistent") is False

    # --- list ---

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Empty toolbox returns 'No tools available' message."""
        toolbox = Toolbox()
        result = await toolbox.execute(action="list")
        assert "No tools available" in result

    @pytest.mark.asyncio
    async def test_list_shows_registered_tools(self):
        """List shows all registered toolbox-tier tools with one-liners."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="search", one_liner="Search things"))
        toolbox.register(self._make_entry(name="calc", one_liner="Do math"))
        result = await toolbox.execute(action="list")
        assert "- search: Search things" in result
        assert "- calc: Do math" in result

    @pytest.mark.asyncio
    async def test_list_excludes_full_tier_tools(self):
        """List excludes full-tier tools (they're already in context)."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="visible", tier="toolbox"))
        toolbox.register(self._make_entry(name="injected", tier="full"))
        result = await toolbox.execute(action="list")
        assert "visible" in result
        assert "injected" not in result

    @pytest.mark.asyncio
    async def test_list_empty_after_excluding_full_tier(self):
        """If only full-tier tools exist, list says no tools available."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="injected", tier="full"))
        result = await toolbox.execute(action="list")
        assert "No tools available" in result

    # --- describe ---

    @pytest.mark.asyncio
    async def test_describe_returns_schema(self):
        """Describe returns JSON of the tool's schema."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="search"))
        result = await toolbox.execute(action="describe", tool="search")
        schema = json.loads(result)
        assert schema["name"] == "search"
        assert "parameters" in schema

    @pytest.mark.asyncio
    async def test_describe_unknown_tool_errors(self):
        """Describe with unknown tool returns error mentioning 'list'."""
        toolbox = Toolbox()
        result = await toolbox.execute(action="describe", tool="nonexistent")
        assert "error" in result.lower() or "not found" in result.lower()
        assert "list" in result.lower()

    @pytest.mark.asyncio
    async def test_describe_missing_tool_name_errors(self):
        """Describe without tool param returns error string."""
        toolbox = Toolbox()
        result = await toolbox.execute(action="describe")
        assert "error" in result.lower() or "required" in result.lower()

    # --- use ---

    @pytest.mark.asyncio
    async def test_use_delegates_to_handler(self):
        """Use delegates to handler.execute with correct arguments."""
        handler = DummyToolHandler(name="search", result="found 42 results")
        handler.execute = AsyncMock(return_value="found 42 results")
        entry = ToolboxEntry(
            name="search", one_liner="Search things", handler=handler
        )
        toolbox = Toolbox()
        toolbox.register(entry)

        result = await toolbox.execute(
            action="use", tool="search", arguments={"query": "test"}
        )
        handler.execute.assert_called_once_with(query="test")
        assert result == "found 42 results"

    @pytest.mark.asyncio
    async def test_use_empty_arguments(self):
        """Use without arguments param passes empty kwargs to handler."""
        handler = DummyToolHandler(name="status", result="all good")
        handler.execute = AsyncMock(return_value="all good")
        entry = ToolboxEntry(
            name="status", one_liner="Check status", handler=handler
        )
        toolbox = Toolbox()
        toolbox.register(entry)

        result = await toolbox.execute(action="use", tool="status")
        handler.execute.assert_called_once_with()
        assert result == "all good"

    @pytest.mark.asyncio
    async def test_use_catches_handler_exception(self):
        """Handler exception is caught and returned as error string."""
        handler = DummyToolHandler(name="explode")
        handler.execute = AsyncMock(side_effect=RuntimeError("boom"))
        entry = ToolboxEntry(
            name="explode", one_liner="Goes boom", handler=handler
        )
        toolbox = Toolbox()
        toolbox.register(entry)

        result = await toolbox.execute(action="use", tool="explode")
        assert "error" in result.lower() or "boom" in result.lower()
        # Must not raise
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_use_unknown_tool_errors(self):
        """Use with unknown tool returns error mentioning 'list'."""
        toolbox = Toolbox()
        result = await toolbox.execute(action="use", tool="ghost")
        assert "list" in result.lower()

    @pytest.mark.asyncio
    async def test_use_missing_tool_name_errors(self):
        """Use without tool param returns error string."""
        toolbox = Toolbox()
        result = await toolbox.execute(action="use")
        assert "error" in result.lower() or "required" in result.lower()

    # --- unknown action ---

    @pytest.mark.asyncio
    async def test_unknown_action_errors(self):
        """Unknown action returns error string."""
        toolbox = Toolbox()
        result = await toolbox.execute(action="invalid")
        assert "error" in result.lower() or "unknown" in result.lower()

    # --- tool_definition ---

    def test_tool_definition_is_compact(self):
        """Schema should be under ~100 tokens."""
        toolbox = Toolbox()
        schema = toolbox.tool_definition
        token_estimate = len(json.dumps(schema)) / 4
        assert token_estimate < 100, (
            f"Schema is ~{token_estimate:.0f} tokens, must be under 100"
        )

    def test_tool_definition_has_name(self):
        """tool_definition includes the tool name."""
        toolbox = Toolbox()
        assert toolbox.tool_definition["name"] == Toolbox.TOOL_NAME

    # --- get_full_tier_schemas ---

    def test_get_full_tier_schemas(self):
        """Returns schemas only for full-tier entries."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="basic", tier="toolbox"))
        toolbox.register(self._make_entry(name="summary_tool", tier="summary"))
        toolbox.register(self._make_entry(name="injected", tier="full"))

        schemas = toolbox.get_full_tier_schemas()
        names = [s["name"] for s in schemas]
        assert "injected" in names
        assert "basic" not in names
        assert "summary_tool" not in names

    def test_get_full_tier_schemas_empty(self):
        """No full-tier tools returns empty list."""
        toolbox = Toolbox()
        toolbox.register(self._make_entry(name="basic", tier="toolbox"))
        assert toolbox.get_full_tier_schemas() == []

    def test_get_full_tier_schemas_no_tools(self):
        """Empty toolbox returns empty list."""
        toolbox = Toolbox()
        assert toolbox.get_full_tier_schemas() == []
