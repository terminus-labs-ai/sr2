from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from sr2 import ContentBlock, Message, TextBlock, TokenUsage, ToolDefinition
from sr2.models import ToolResultBlock, ToolUseBlock


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
  type: Literal["text", "thinking", "usage", "end", "tool_use", "iteration_complete", "tool_use_emitted", "tool_result_received", "error"]
  text: str = ""
  usage: TokenUsage | None = None
  tool_use_id: str = ""
  tool_name: str = ""
  tool_input: dict[str, Any] = Field(default_factory=dict)
  tool_uses: list[ToolUseBlock] | None = None
  tool_results: list[ToolResultBlock] | None = None
  iteration: int | None = None
  errors: list[str] | None = None


@runtime_checkable
class LLMCallable(Protocol):
  async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
  async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
