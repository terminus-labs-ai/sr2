"""Thin LLM client for SR2Runtime."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sr2.runtime.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalized LLM response."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_litellm(cls, response: Any) -> LLMResponse:
        """Parse LiteLLM response into normalized format."""
        msg = response.choices[0].message
        content = msg.content

        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        usage = response.usage
        return cls(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )


class RuntimeLLMClient:
    """Async LLM client wrapping LiteLLM.

    Follows patterns from src/runtime/llm/client.py but stripped
    to the minimum needed for SR2Runtime.execute().
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model_string = self._build_model_string()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        """Call the LLM with compiled messages.

        Args:
            messages: OpenAI-format message list (from SR2.process()).
            tools: OpenAI-format tool schemas (function defs, NOT wrapped).
            tool_choice: Tool choice directive.

        Returns:
            LLMResponse with content, tool_calls, and usage.
        """
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model_string,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra,
        }

        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        if tools:
            # Wrap in OpenAI function-calling format
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await litellm.acompletion(**kwargs)
        return LLMResponse.from_litellm(response)

    def _build_model_string(self) -> str:
        """Build LiteLLM model string from config.

        Maps provider to LiteLLM prefix following existing patterns
        in src/runtime/llm/client.py:
          ollama -> "ollama_chat/<name>" (for tool calling support)
          litellm -> "<name>" (pass through)
          openai-compat -> "openai/<name>"
          other -> "<provider>/<name>"
        """
        provider = self.config.provider.lower()
        name = self.config.name

        if provider == "litellm":
            return name
        if provider == "ollama":
            return f"ollama_chat/{name}"
        if provider == "openai-compat":
            return f"openai/{name}"
        return f"{provider}/{name}"
