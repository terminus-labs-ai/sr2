from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ContentBlockBase(BaseModel):
  token_count: int | None = None
  meta: dict[str, Any] = Field(default_factory=dict)
  """Generic metadata attached to every content block.

  Used by infrastructure layers (e.g. the planning layer's *frame* tag) for
  internal bookkeeping.  Never serialised into the LLM wire payload — wire
  serializers access block fields explicitly and never call ``model_dump()``.
  """


class TextBlock(ContentBlockBase):
  type: Literal["text"] = "text"
  text: str


class ToolUseBlock(ContentBlockBase):
  type: Literal["tool_use"] = "tool_use"
  id: str
  name: str
  input: dict[str, Any]


class ToolResultBlock(ContentBlockBase):
  type: Literal["tool_result"] = "tool_result"
  tool_use_id: str
  content: str | list[TextBlock]
  is_error: bool = False
  compacted: bool = False


class ThinkingBlock(ContentBlockBase):
  type: Literal["thinking"] = "thinking"
  text: str


ContentBlock = Annotated[
  TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock,
  Field(discriminator="type"),
]


class Zone(str, Enum):
  RAW = "raw"
  COMPACTED = "compacted"
  SUMMARIZED = "summarized"


class Message(BaseModel):
  role: Literal["user", "assistant"]
  content: list[ContentBlock]
  turn_index: int | None = None
  timestamp: str | None = None
  zone: Zone | None = None


class ToolDefinition(BaseModel):
  name: str
  description: str = ""
  input_schema: dict[str, Any]


class TokenUsage(BaseModel):
  input_tokens: int = 0
  output_tokens: int = 0
  cache_creation_input_tokens: int = 0
  cache_read_input_tokens: int = 0
