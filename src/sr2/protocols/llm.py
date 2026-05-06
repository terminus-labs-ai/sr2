from typing import Protocol

from pydantic import BaseModel

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


class LLMCallable(Protocol):
  async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
