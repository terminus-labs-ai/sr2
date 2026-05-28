from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from sr2 import ContentBlock, Message, TextBlock, TokenUsage, ToolDefinition


class CompletionRequest(BaseModel):
  system: list[TextBlock] | None = None
  messages: list[Message]
  tools: list[ToolDefinition] | None = None


class CompletionResponse(BaseModel):
  id: str
  content: list[ContentBlock]
  stop_reason: str | None = None  # "end_turn", "tool_use", "max_tokens"
  usage: TokenUsage


class StreamEvent(BaseModel):
  type: Literal["text", "usage", "end", "tool_use"]
  text: str = ""
  usage: TokenUsage | None = None
  tool_use_id: str = ""
  tool_name: str = ""
  tool_input: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMCallable(Protocol):
  async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
  async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
