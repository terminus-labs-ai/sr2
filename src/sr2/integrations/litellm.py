from __future__ import annotations

import json
from collections.abc import AsyncIterator

import litellm

from sr2.models import TextBlock, TokenUsage, ToolUseBlock, ToolResultBlock
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


class LiteLLMCallable:
  def __init__(self, model: str, base_url: str | None = None, **kwargs) -> None:
    # When hitting an OpenAI-compatible endpoint with a bare model name,
    # litellm needs a provider prefix to route the call correctly.
    if base_url is not None and "/" not in model:
      model = f"openai/{model}"
    self._model = model
    self._kwargs: dict = kwargs
    if base_url is not None:
      self._kwargs["base_url"] = base_url

  def _build_messages(self, request: CompletionRequest) -> list[dict]:
    result: list[dict] = []
    for msg in request.messages:
      # ToolResultBlock → emit one "tool" role message per block
      if any(isinstance(b, ToolResultBlock) for b in msg.content):
        for block in msg.content:
          if isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
              content = block.content
            else:
              content = "".join(b.text for b in block.content)
            result.append({
              "role": "tool",
              "tool_call_id": block.tool_use_id,
              "content": content,
            })
      # ToolUseBlock → emit one assistant message with tool_calls
      elif any(isinstance(b, ToolUseBlock) for b in msg.content):
        text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
        text_content = "".join(text_parts) if text_parts else None
        tool_calls = [
          {
            "id": block.id,
            "type": "function",
            "function": {
              "name": block.name,
              "arguments": json.dumps(block.input),
            },
          }
          for block in msg.content
          if isinstance(block, ToolUseBlock)
        ]
        result.append({
          "role": "assistant",
          "content": text_content,
          "tool_calls": tool_calls,
        })
      # Default: plain text message
      else:
        result.append({
          "role": msg.role,
          "content": "".join(
            b.text for b in msg.content if hasattr(b, "text")
          ),
        })
    return result

  def _build_extra(self, request: CompletionRequest) -> dict:
    extra: dict = {}
    if request.system is not None:
      extra["system"] = "".join(b.text for b in request.system)
    if request.tools is not None:
      extra["tools"] = [
        {
          "type": "function",
          "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
          },
        }
        for tool in request.tools
      ]
    return extra

  async def complete(self, request: CompletionRequest) -> CompletionResponse:
    messages = self._build_messages(request)
    extra = self._build_extra(request)

    resp = await litellm.acompletion(
      model=self._model,
      messages=messages,
      **self._kwargs,
      **extra,
    )

    choice = resp.choices[0]
    tool_calls = choice.message.tool_calls

    if tool_calls:
      content: list = []
      if choice.message.content:
        content.append(TextBlock(text=choice.message.content))
      for tc in tool_calls:
        content.append(ToolUseBlock(
          id=tc.id,
          name=tc.function.name,
          input=json.loads(tc.function.arguments),
        ))
      stop_reason = "tool_use"
    else:
      content = [TextBlock(text=choice.message.content or "")]
      stop_reason = choice.finish_reason

    return CompletionResponse(
      id=resp.id,
      content=content,
      stop_reason=stop_reason,
      usage=TokenUsage(
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
      ),
    )

  async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
    messages = self._build_messages(request)
    extra = self._build_extra(request)

    response = await litellm.acompletion(
      model=self._model,
      messages=messages,
      stream=True,
      **self._kwargs,
      **extra,
    )

    # index → {"id": str, "name": str, "arguments": str}
    tool_call_acc: dict[int, dict] = {}

    async for chunk in response:
      if chunk.choices:
        delta = chunk.choices[0].delta

        # Text content
        if delta.content:
          yield StreamEvent(type="text", text=delta.content)

        # Tool call deltas
        if delta.tool_calls is not None:
          for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in tool_call_acc:
              tool_call_acc[idx] = {"id": tc_delta.id, "name": tc_delta.function.name, "arguments": ""}
            tool_call_acc[idx]["arguments"] += tc_delta.function.arguments

      # Usage
      if getattr(chunk, "usage", None) is not None:
        yield StreamEvent(
          type="usage",
          usage=TokenUsage(
            input_tokens=chunk.usage.prompt_tokens,
            output_tokens=chunk.usage.completion_tokens,
          ),
        )

    for idx in sorted(tool_call_acc):
      acc = tool_call_acc[idx]
      yield StreamEvent(
        type="tool_use",
        tool_use_id=acc["id"],
        tool_name=acc["name"],
        tool_input=json.loads(acc["arguments"]),
      )

    yield StreamEvent(type="end")
