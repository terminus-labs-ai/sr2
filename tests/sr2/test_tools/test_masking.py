"""Tests for tool masking strategies."""

import pytest

from sr2.tools.masking import (
    AllowedListStrategy,
    LogitMaskStrategy,
    NoMaskingStrategy,
    PrefillStrategy,
    get_masking_strategy,
)
from sr2.tools.models import ToolDefinition, ToolStateConfig


def _make_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(name="read_file", description="Read a file"),
        ToolDefinition(name="write_file", description="Write a file"),
        ToolDefinition(name="bash", description="Execute bash"),
        ToolDefinition(name="rm", description="Delete file"),
    ]


class TestAllowedListStrategy:
    def test_filters_by_state(self):
        """State allows 2 of 4 tools -> only those 2 in output."""
        state = ToolStateConfig(name="safe", allowed_tools=["read_file", "bash"])
        strategy = AllowedListStrategy()
        result = strategy.apply(_make_tools(), state)

        assert set(result["allowed_tools"]) == {"read_file", "bash"}

    def test_includes_tool_schemas(self):
        """Includes tool_schemas for allowed tools."""
        state = ToolStateConfig(name="safe", allowed_tools=["read_file"])
        strategy = AllowedListStrategy()
        result = strategy.apply(_make_tools(), state)

        assert len(result["tool_schemas"]) == 1
        assert result["tool_schemas"][0]["name"] == "read_file"


class TestPrefillStrategy:
    def test_produces_response_prefix(self):
        """Produces response prefix with forced tool name."""
        state = ToolStateConfig(name="default", allowed_tools="all")
        strategy = PrefillStrategy(forced_tool="bash")
        result = strategy.apply(_make_tools(), state)

        assert '"tool": "bash"' in result["response_prefix"]
        assert result["forced_tool"] == "bash"

    def test_no_allowed_tools(self):
        """No allowed tools -> empty prefix."""
        state = ToolStateConfig(name="empty", allowed_tools=[])
        strategy = PrefillStrategy()
        result = strategy.apply(_make_tools(), state)

        assert result["response_prefix"] == ""
        assert result["forced_tool"] is None


class TestLogitMaskStrategy:
    def test_correct_split(self):
        """Correct split between allowed and denied tokens."""
        state = ToolStateConfig(name="safe", allowed_tools=["read_file", "bash"])
        strategy = LogitMaskStrategy()
        result = strategy.apply(_make_tools(), state)

        assert set(result["allowed_tool_tokens"]) == {"read_file", "bash"}
        assert set(result["denied_tool_tokens"]) == {"write_file", "rm"}


class TestNoMaskingStrategy:
    def test_all_tools(self):
        """All tools in output regardless of state."""
        state = ToolStateConfig(name="restricted", allowed_tools=["read_file"])
        strategy = NoMaskingStrategy()
        result = strategy.apply(_make_tools(), state)

        assert len(result["tool_schemas"]) == 4
        assert result["tool_choice"] == "auto"


class TestRawParametersPassthrough:
    """Ensure MCP tool schemas (raw JSON Schema) survive the masking pipeline."""

    def test_allowed_list_preserves_raw_parameters(self):
        """ToolDefinition with raw_parameters should output them in to_function_schema."""
        raw_params = {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repo owner"},
                "repo": {"type": "string", "description": "Repo name"},
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["owner", "repo", "path"],
        }
        tools = [
            ToolDefinition(
                name="get_file_contents",
                description="Get file contents",
                raw_parameters=raw_params,
            ),
        ]
        state = ToolStateConfig(name="default", allowed_tools="all")
        strategy = AllowedListStrategy()
        result = strategy.apply(tools, state)

        schema = result["tool_schemas"][0]
        assert schema["name"] == "get_file_contents"
        assert "owner" in schema["parameters"]["properties"]
        assert "repo" in schema["parameters"]["properties"]
        assert "path" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["owner", "repo", "path"]

    def test_no_masking_preserves_raw_parameters(self):
        """NoMaskingStrategy also preserves raw_parameters."""
        raw_params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        tools = [
            ToolDefinition(name="search", description="Search", raw_parameters=raw_params),
            ToolDefinition(name="plain_tool", description="No params"),
        ]
        state = ToolStateConfig(name="default", allowed_tools="all")
        strategy = NoMaskingStrategy()
        result = strategy.apply(tools, state)

        # search should have full params
        search_schema = [s for s in result["tool_schemas"] if s["name"] == "search"][0]
        assert "query" in search_schema["parameters"]["properties"]

        # plain_tool should have empty params (built from empty ToolParameter list)
        plain_schema = [s for s in result["tool_schemas"] if s["name"] == "plain_tool"][0]
        assert plain_schema["parameters"]["properties"] == {}

    def test_raw_parameters_none_uses_parameter_list(self):
        """When raw_parameters is None, to_function_schema builds from parameters list."""
        from sr2.tools.models import ToolParameter

        tools = [
            ToolDefinition(
                name="my_tool",
                description="A tool",
                parameters=[
                    ToolParameter(name="arg1", type="string", description="First arg", required=True),
                ],
            ),
        ]
        schema = tools[0].to_function_schema()
        assert "arg1" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["arg1"]


class TestDeniedToolsPrecedence:
    """denied_tools takes precedence over allowed_tools (doc: guide-tool-masking.md)."""

    def test_allowed_list_excludes_denied_from_all(self):
        """allowed_tools='all' + denied_tools=['rm'] -> rm excluded from masking output."""
        state = ToolStateConfig(name="executing", allowed_tools="all", denied_tools=["rm"])
        strategy = AllowedListStrategy()
        result = strategy.apply(_make_tools(), state)

        assert "rm" not in result["allowed_tools"]
        assert set(result["allowed_tools"]) == {"read_file", "write_file", "bash"}
        schema_names = [s["name"] for s in result["tool_schemas"]]
        assert "rm" not in schema_names

    def test_logit_mask_excludes_denied_from_all(self):
        """LogitMaskStrategy correctly splits denied tools from allowed_tools='all'."""
        state = ToolStateConfig(name="executing", allowed_tools="all", denied_tools=["rm"])
        strategy = LogitMaskStrategy()
        result = strategy.apply(_make_tools(), state)

        assert "rm" in result["denied_tool_tokens"]
        assert "rm" not in result["allowed_tool_tokens"]

    # --- Precedence: tool in BOTH allowed and denied ---

    def test_tool_in_both_lists_is_denied(self):
        """A tool in both allowed_tools and denied_tools must be denied."""
        state = ToolStateConfig(
            name="conflict",
            allowed_tools=["read_file", "bash"],
            denied_tools=["bash"],
        )
        assert not state.is_tool_allowed("bash")

    def test_allowed_list_denies_tool_in_both_lists(self):
        """AllowedListStrategy excludes a tool that appears in both lists."""
        state = ToolStateConfig(
            name="conflict",
            allowed_tools=["read_file", "write_file", "bash"],
            denied_tools=["write_file"],
        )
        strategy = AllowedListStrategy()
        result = strategy.apply(_make_tools(), state)

        assert "write_file" not in result["allowed_tools"]
        assert set(result["allowed_tools"]) == {"read_file", "bash"}
        schema_names = [s["name"] for s in result["tool_schemas"]]
        assert "write_file" not in schema_names

    def test_logit_mask_denies_tool_in_both_lists(self):
        """LogitMaskStrategy treats a tool in both lists as denied."""
        state = ToolStateConfig(
            name="conflict",
            allowed_tools=["read_file", "bash"],
            denied_tools=["bash"],
        )
        strategy = LogitMaskStrategy()
        result = strategy.apply(_make_tools(), state)

        assert "bash" in result["denied_tool_tokens"]
        assert "bash" not in result["allowed_tool_tokens"]

    def test_prefill_skips_denied_tool(self):
        """PrefillStrategy falls back when the only allowed tool is also denied."""
        state = ToolStateConfig(
            name="conflict",
            allowed_tools=["bash"],
            denied_tools=["bash"],
        )
        strategy = PrefillStrategy()
        result = strategy.apply(_make_tools(), state)

        assert result["response_prefix"] == ""
        assert result["forced_tool"] is None

    # --- Only in allowed_tools (not denied) ---

    def test_tool_only_in_allowed_is_permitted(self):
        """A tool in allowed_tools but not denied_tools is allowed."""
        state = ToolStateConfig(
            name="selective",
            allowed_tools=["read_file", "bash"],
            denied_tools=["rm"],
        )
        assert state.is_tool_allowed("read_file")
        assert state.is_tool_allowed("bash")

    # --- Only in denied_tools ---

    def test_tool_only_in_denied_is_blocked(self):
        """A tool in denied_tools (with allowed_tools='all') is blocked."""
        state = ToolStateConfig(
            name="restricted",
            allowed_tools="all",
            denied_tools=["rm", "bash"],
        )
        assert not state.is_tool_allowed("rm")
        assert not state.is_tool_allowed("bash")
        # Others still allowed
        assert state.is_tool_allowed("read_file")

    # --- In neither list ---

    def test_tool_in_neither_list_with_explicit_allowed(self):
        """Tool not in allowed_tools (explicit list) and not in denied_tools -> denied."""
        state = ToolStateConfig(
            name="narrow",
            allowed_tools=["read_file"],
            denied_tools=["rm"],
        )
        # bash is in neither list; with an explicit allowed list it's implicitly denied
        assert not state.is_tool_allowed("bash")

    def test_tool_in_neither_list_with_all_allowed(self):
        """Tool not in denied_tools with allowed_tools='all' -> allowed."""
        state = ToolStateConfig(
            name="open",
            allowed_tools="all",
            denied_tools=["rm"],
        )
        assert state.is_tool_allowed("bash")
        assert state.is_tool_allowed("write_file")


class TestGetMaskingStrategy:
    def test_get_existing(self):
        """get_masking_strategy returns correct instance."""
        strategy = get_masking_strategy("allowed_list")
        assert isinstance(strategy, AllowedListStrategy)

    def test_get_unknown(self):
        """get_masking_strategy raises KeyError for unknown."""
        with pytest.raises(KeyError, match="Unknown masking strategy"):
            get_masking_strategy("unknown")
