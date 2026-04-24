"""Tool definition models for the tool management system."""

from typing import Literal

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """A single parameter in a tool definition."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: str | int | float | bool | None = None
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """A tool available to the agent."""

    name: str = Field(description="Unique tool name")
    type: Literal["standard", "retrieval"] = "standard"
    description: str = Field(default="", description="Description shown to the LLM")
    parameters: list[ToolParameter] = Field(default_factory=list)
    raw_parameters: dict | None = Field(
        default=None,
        description="Raw JSON Schema for parameters (from MCP inputSchema). "
        "When set, to_function_schema() uses this instead of building from parameters list.",
    )
    category: Literal["read", "write", "execute", "dangerous"] = "read"
    requires_confirmation: bool = False

    def to_function_schema(self) -> dict:
        """Convert to OpenAI-style function calling schema."""
        if self.raw_parameters is not None:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": self.raw_parameters,
            }

        properties = {}
        required = []
        for p in self.parameters:
            prop: dict = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolStateConfig(BaseModel):
    """A named state that determines which tools are available."""

    name: str = Field(description="State name, e.g. 'default', 'planning'")
    description: str = ""
    allowed_tools: list[str] | Literal["all"] = "all"
    denied_tools: list[str] = Field(default_factory=list)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in this state."""
        if tool_name in self.denied_tools:
            return False
        if self.allowed_tools == "all":
            return True
        return tool_name in self.allowed_tools


class ToolTransitionConfig(BaseModel):
    """A transition rule between tool states."""

    from_state: str = Field(description="Source state name, or 'any'")
    to_state: str = Field(description="Target state name")
    trigger: Literal["agent_intent", "agent_action", "pipeline_signal", "user_confirmation"] = (
        "agent_intent"
    )
    condition: str = Field(default="", description="Condition expression as string")


class ToolManagementConfig(BaseModel):
    """Top-level tool management configuration."""

    tools: list[ToolDefinition] = Field(default_factory=list)
    states: list[ToolStateConfig] = Field(
        default_factory=lambda: [
            ToolStateConfig(name="default", allowed_tools="all"),
        ]
    )
    transitions: list[ToolTransitionConfig] = Field(default_factory=list)
    masking_strategy: Literal["prefill", "allowed_list", "logit_mask", "none"] = "allowed_list"
    initial_state: str = "default"
