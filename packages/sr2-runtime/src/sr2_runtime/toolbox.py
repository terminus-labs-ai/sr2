"""Toolbox meta-tool — a single compact tool that gates access to many."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolboxEntry:
    """A tool registered in the toolbox."""

    name: str
    one_liner: str
    handler: Any  # has .tool_definition property + async execute(**kwargs) -> str
    tier: Literal["toolbox", "summary", "full"] = "toolbox"


class Toolbox:
    """Meta-tool that exposes many tools through a single compact schema."""

    TOOL_NAME = "toolbox"

    def __init__(self) -> None:
        self._tools: dict[str, ToolboxEntry] = {}

    def register(self, entry: ToolboxEntry) -> None:
        self._tools[entry.name] = entry

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_definition(self) -> dict:
        return {
            "name": self.TOOL_NAME,
            "description": "Access specialized tools. Use action=list to browse, describe for schema, use to invoke.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "describe", "use"],
                    },
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["action"],
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action")
        tool_name = kwargs.get("tool")
        arguments = kwargs.get("arguments", {})

        if action == "list":
            return self._action_list()
        elif action == "describe":
            return self._action_describe(tool_name)
        elif action == "use":
            return await self._action_use(tool_name, arguments)
        else:
            return f"Error: unknown action '{action}'. Use list, describe, or use."

    def _action_list(self) -> str:
        lines = [
            f"- {e.name}: {e.one_liner}"
            for e in self._tools.values()
            if e.tier != "full"
        ]
        if not lines:
            return "No tools available in toolbox."
        return "\n".join(lines)

    def _action_describe(self, tool_name: str | None) -> str:
        if not tool_name:
            return "Error: 'tool' parameter is required for describe."
        entry = self._tools.get(tool_name)
        if not entry:
            return f"Error: tool '{tool_name}' not found. Use action=list to see available tools."
        return json.dumps(entry.handler.tool_definition)

    async def _action_use(self, tool_name: str | None, arguments: dict) -> str:
        if not tool_name:
            return "Error: 'tool' parameter is required for use."
        entry = self._tools.get(tool_name)
        if not entry:
            return f"Error: tool '{tool_name}' not found. Use action=list to see available tools."
        try:
            return await entry.handler.execute(**arguments)
        except Exception as e:
            return f"Error executing '{tool_name}': {e}"

    def get_full_tier_schemas(self) -> list[dict]:
        return [
            e.handler.tool_definition
            for e in self._tools.values()
            if e.tier == "full"
        ]
