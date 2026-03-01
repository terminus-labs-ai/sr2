"""Tests for A2A Agent Card generator."""

import json

from sr2.a2a.card import AgentCardGenerator
from sr2.tools.models import ToolDefinition, ToolParameter


class TestAgentCardGenerator:

    def test_minimal_config_produces_valid_card(self):
        """generate() with minimal config produces valid card structure."""
        gen = AgentCardGenerator(agent_name="test-agent")
        card = gen.generate()

        assert "name" in card
        assert "url" in card
        assert "version" in card
        assert "capabilities" in card

    def test_card_includes_all_fields(self):
        """Card includes name, url, version, capabilities."""
        gen = AgentCardGenerator(
            agent_name="my-agent",
            description="A test agent",
            url="http://example.com",
            version="2.0.0",
        )
        card = gen.generate()

        assert card["name"] == "my-agent"
        assert card["description"] == "A test agent"
        assert card["url"] == "http://example.com"
        assert card["version"] == "2.0.0"

    def test_auth_schemes_included_when_provided(self):
        """Auth schemes included when provided."""
        gen = AgentCardGenerator(
            agent_name="secure-agent",
            auth_schemes=["bearer", "api_key"],
        )
        card = gen.generate()

        assert "authentication" in card
        assert card["authentication"]["schemes"] == ["bearer", "api_key"]

    def test_skill_from_tool_converts_correctly(self):
        """skill_from_tool() converts ToolDefinition correctly."""
        gen = AgentCardGenerator(agent_name="agent")
        tool = ToolDefinition(
            name="read_file",
            description="Read a file from disk",
            parameters=[
                ToolParameter(name="path", type="string", required=True),
            ],
        )
        skill = gen.skill_from_tool(tool)

        assert skill["id"] == "read_file"
        assert skill["name"] == "Read File"
        assert skill["description"] == "Read a file from disk"
        assert "inputSchema" in skill

    def test_skill_from_tool_includes_output_schema(self):
        """skill_from_tool() includes output schema when provided."""
        gen = AgentCardGenerator(agent_name="agent")
        tool = ToolDefinition(name="search", description="Search")
        output_schema = {"type": "object", "properties": {"results": {"type": "array"}}}
        skill = gen.skill_from_tool(tool, output_schema=output_schema)

        assert skill["outputSchema"] == output_schema

    def test_to_json_produces_valid_json(self):
        """to_json() produces valid JSON string."""
        gen = AgentCardGenerator(agent_name="json-agent")
        result = gen.to_json()

        parsed = json.loads(result)
        assert parsed["name"] == "json-agent"

    def test_capabilities_reflect_flags(self):
        """Capabilities reflect streaming and long_running flags."""
        gen = AgentCardGenerator(
            agent_name="capable-agent",
            streaming=True,
            long_running=True,
        )
        card = gen.generate()

        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"]["longRunningTasks"] is True

    def test_no_auth_section_when_empty(self):
        """No auth section when auth_schemes is empty."""
        gen = AgentCardGenerator(agent_name="no-auth-agent")
        card = gen.generate()

        assert "authentication" not in card
