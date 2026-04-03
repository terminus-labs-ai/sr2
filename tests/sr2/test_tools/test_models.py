"""Tests for tool definition models."""

from sr2.tools.models import (
    ToolDefinition,
    ToolManagementConfig,
    ToolParameter,
    ToolStateConfig,
)


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_minimal_args(self):
        """ToolDefinition with minimal args produces valid defaults."""
        tool = ToolDefinition(name="read_file")
        assert tool.name == "read_file"
        assert tool.type == "standard"
        assert tool.category == "read"
        assert tool.requires_confirmation is False
        assert tool.parameters == []

    def test_to_function_schema(self):
        """to_function_schema() produces correct OpenAI-style schema."""
        tool = ToolDefinition(
            name="search",
            description="Search for files",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query", required=True),
            ],
        )
        schema = tool.to_function_schema()

        assert schema["name"] == "search"
        assert schema["description"] == "Search for files"
        assert "query" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]

    def test_to_function_schema_with_enum(self):
        """to_function_schema() includes required params and enums."""
        tool = ToolDefinition(
            name="format",
            parameters=[
                ToolParameter(name="style", type="string", enum=["json", "yaml", "toml"]),
            ],
        )
        schema = tool.to_function_schema()

        assert schema["parameters"]["properties"]["style"]["enum"] == ["json", "yaml", "toml"]

    def test_a2a_tool_type(self):
        """ToolDefinition with type='a2a_tool' is valid."""
        tool = ToolDefinition(name="delegate", type="a2a_tool")
        assert tool.type == "a2a_tool"


class TestToolStateConfig:
    """Tests for ToolStateConfig."""

    def test_all_allows_except_denied(self):
        """allowed_tools='all' allows everything except denied."""
        state = ToolStateConfig(name="default", allowed_tools="all", denied_tools=["rm"])
        assert state.is_tool_allowed("read_file") is True
        assert state.is_tool_allowed("rm") is False

    def test_explicit_allowed_list(self):
        """Explicit allowed list only allows listed tools."""
        state = ToolStateConfig(name="restricted", allowed_tools=["read_file", "search"])
        assert state.is_tool_allowed("read_file") is True
        assert state.is_tool_allowed("write_file") is False

    def test_denied_takes_precedence(self):
        """denied_tools takes precedence over allowed_tools."""
        state = ToolStateConfig(
            name="test",
            allowed_tools=["read_file", "write_file"],
            denied_tools=["write_file"],
        )
        assert state.is_tool_allowed("read_file") is True
        assert state.is_tool_allowed("write_file") is False


class TestToolManagementConfig:
    """Tests for ToolManagementConfig."""

    def test_defaults(self):
        """Defaults to single 'default' state with all tools allowed."""
        config = ToolManagementConfig()
        assert len(config.states) == 1
        assert config.states[0].name == "default"
        assert config.states[0].allowed_tools == "all"
        assert config.initial_state == "default"
        assert config.masking_strategy == "allowed_list"
