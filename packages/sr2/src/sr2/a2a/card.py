"""Auto-generate A2A Agent Cards from agent configuration."""

import json

from sr2.tools.models import ToolDefinition


class AgentCardGenerator:
    """Generates A2A Agent Cards from agent configuration."""

    def __init__(
        self,
        agent_name: str,
        description: str = "",
        url: str = "http://localhost:8008",
        version: str = "1.0.0",
        auth_schemes: list[str] | None = None,
        streaming: bool = False,
        long_running: bool = False,
    ):
        self._name = agent_name
        self._description = description
        self._url = url
        self._version = version
        self._auth = auth_schemes or []
        self._streaming = streaming
        self._long_running = long_running

    def generate(self, skills: list[dict] | None = None) -> dict:
        """Generate a complete Agent Card as a dict."""
        card = {
            "name": self._name,
            "description": self._description,
            "url": self._url,
            "version": self._version,
            "capabilities": {
                "streaming": self._streaming,
                "longRunningTasks": self._long_running,
            },
        }

        if self._auth:
            card["authentication"] = {"schemes": self._auth}

        if skills:
            card["skills"] = skills

        return card

    def skill_from_tool(self, tool: ToolDefinition, output_schema: dict | None = None) -> dict:
        """Convert a ToolDefinition into an A2A skill entry."""
        schema = tool.to_function_schema()
        skill = {
            "id": tool.name,
            "name": tool.name.replace("_", " ").title(),
            "description": tool.description,
            "inputSchema": schema.get("parameters", {}),
        }
        if output_schema:
            skill["outputSchema"] = output_schema
        return skill

    def to_json(self, skills: list[dict] | None = None) -> str:
        """Generate Agent Card as a JSON string."""
        return json.dumps(self.generate(skills), indent=2)
