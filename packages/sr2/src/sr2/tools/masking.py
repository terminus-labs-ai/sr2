"""Tool masking strategies that control which tools the LLM can select."""

from typing import Protocol

from sr2.tools.models import ToolDefinition, ToolStateConfig


class MaskingStrategy(Protocol):
    """Protocol for tool masking strategies."""

    def apply(self, tools: list[ToolDefinition], state: ToolStateConfig) -> dict: ...


class AllowedListStrategy:
    """Filter tools by allowed list. Most common strategy for hosted APIs."""

    def apply(self, tools: list[ToolDefinition], state: ToolStateConfig) -> dict:
        allowed = [t for t in tools if state.is_tool_allowed(t.name)]
        return {
            "tool_choice": "auto",
            "allowed_tools": [t.name for t in allowed],
            "tool_schemas": [t.to_function_schema() for t in allowed],
        }


class PrefillStrategy:
    """Force a specific tool via response prefix. Used for single-tool interfaces."""

    def __init__(self, forced_tool: str | None = None):
        self._forced = forced_tool

    def apply(self, tools: list[ToolDefinition], state: ToolStateConfig) -> dict:
        allowed = [t for t in tools if state.is_tool_allowed(t.name)]
        if not allowed:
            return {"response_prefix": "", "forced_tool": None}

        target = self._forced or allowed[0].name
        return {
            "response_prefix": f'{{"tool": "{target}"',
            "forced_tool": target,
        }


class LogitMaskStrategy:
    """Token-level masking for self-hosted inference (vLLM, TGI)."""

    def apply(self, tools: list[ToolDefinition], state: ToolStateConfig) -> dict:
        allowed_names = [t.name for t in tools if state.is_tool_allowed(t.name)]
        denied_names = [t.name for t in tools if not state.is_tool_allowed(t.name)]
        return {
            "allowed_tool_tokens": allowed_names,
            "denied_tool_tokens": denied_names,
        }


class NoMaskingStrategy:
    """No masking — all tools available."""

    def apply(self, tools: list[ToolDefinition], state: ToolStateConfig) -> dict:
        return {
            "tool_choice": "auto",
            "tool_schemas": [t.to_function_schema() for t in tools],
        }


MASKING_STRATEGIES: dict[str, MaskingStrategy] = {
    "allowed_list": AllowedListStrategy(),
    "prefill": PrefillStrategy(),
    "logit_mask": LogitMaskStrategy(),
    "none": NoMaskingStrategy(),
}


def get_masking_strategy(name: str) -> MaskingStrategy:
    """Get a masking strategy by name."""
    if name not in MASKING_STRATEGIES:
        raise KeyError(f"Unknown masking strategy: {name}")
    return MASKING_STRATEGIES[name]
