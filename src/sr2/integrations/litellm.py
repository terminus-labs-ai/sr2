from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import litellm

from sr2.models import TextBlock, TokenUsage, ToolUseBlock, ToolResultBlock
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

logger = logging.getLogger(__name__)

# obsidian-1fmv: cap on raw argument text kept in WARNING logs.
RAW_ARG_LOG_LIMIT = 2000


def _finish_reason_text(value: object) -> str:
  """Normalise a finish_reason to a log-safe string."""
  if value is None:
    return "unknown"
  if isinstance(value, str):
    return value
  return "unknown"


def _safe_parse_tool_arguments(arguments: str) -> tuple[dict | None, str | None]:
  """json.loads for tool-call arguments that survives invalid JSON.

  Returns (parsed, None) on success, (None, error_message) on failure.
  """
  try:
    return json.loads(arguments), None
  except (json.JSONDecodeError, TypeError) as exc:
    return None, str(exc)


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

  @property
  def model(self) -> str:
    return self._model

  def _build_messages(self, request: CompletionRequest) -> list[dict]:
    result: list[dict] = []
    if request.system is not None:
      result.append({"role": "system", "content": "".join(b.text for b in request.system)})
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
      finish_reason = _finish_reason_text(choice.finish_reason)
      for tc in tool_calls:
        name = tc.function.name
        arguments = tc.function.arguments
        parsed, parse_error = _safe_parse_tool_arguments(arguments)
        if parsed is not None:
          content.append(ToolUseBlock(id=tc.id, name=name, input=parsed))
          continue
        truncated = choice.finish_reason == "length"
        if truncated:
          logger.warning(
            "truncated tool call (finish_reason=length): tool=%s error=%s raw=%r",
            name,
            parse_error,
            arguments[:RAW_ARG_LOG_LIMIT],
          )
        else:
          logger.warning(
            "invalid tool-call arguments: tool=%s finish_reason=%s error=%s raw=%r",
            name,
            finish_reason,
            parse_error,
            arguments[:RAW_ARG_LOG_LIMIT],
          )
        content.append(
          ToolUseBlock(
            id=tc.id,
            name=name,
            input={},
            meta={
              "invalid_arguments": True,
              "truncated": truncated,
              "error": parse_error,
              "raw_arguments": arguments[:RAW_ARG_LOG_LIMIT],
            },
          )
        )
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
    finish_reason: str | None = None

    async for chunk in response:
      if chunk.choices:
        choice = chunk.choices[0]
        delta = choice.delta

        # obsidian-1fmv: record finish_reason so malformed/truncated tool
        # calls can be reported accurately.
        if choice.finish_reason is not None:
          finish_reason = choice.finish_reason

        # Text content
        if delta.content:
          yield StreamEvent(type="text", text=delta.content)

        # Thinking / reasoning content (OpenAI reasoning_content, Anthropic thinking_blocks)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning is not None:
          yield StreamEvent(type="thinking", text=reasoning)

        thinking_blocks = getattr(delta, "thinking_blocks", None)
        if thinking_blocks is not None:
          for tb in thinking_blocks:
            if isinstance(tb, dict) and tb.get("type") == "thinking" and tb.get("thinking"):
              yield StreamEvent(type="thinking", text=tb["thinking"])

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

    # obsidian-1fmv: finish_reason "length" with pending tool calls means the
    # model was cut off — a truncation, not an invalid-arguments case.
    length_finish = finish_reason == "length" and bool(tool_call_acc)
    if length_finish:
      logger.warning(
        "stream ended with finish_reason=length while %d tool call(s) were pending — "
        "reporting as truncation",
        len(tool_call_acc),
      )

    # Parse accumulated tool calls without raising on invalid JSON — the
    # orchestrator converts unparseable calls into tool results that tell the
    # model to retry.
    for idx in sorted(tool_call_acc):
      acc = tool_call_acc[idx]
      parsed, parse_error = _safe_parse_tool_arguments(acc["arguments"])
      if parsed is not None:
        meta: dict[str, Any] = {}
        if length_finish:
          # Arguments are complete, but the stop was length-driven — tag the
          # event so callers can see the call arrived at a length boundary.
          meta["truncated"] = True
        yield StreamEvent(
          type="tool_use",
          tool_use_id=acc["id"],
          tool_name=acc["name"],
          tool_input=parsed,
          meta=meta,
        )
        continue

      meta: dict[str, Any] = {
        "invalid_arguments": True,
        "error": parse_error,
        "raw_arguments": acc["arguments"][:RAW_ARG_LOG_LIMIT],
      }
      if length_finish:
        meta["truncated"] = True
      logger.warning(
        "invalid tool-call arguments: tool=%s finish_reason=%s error=%s raw=%r",
        acc["name"],
        _finish_reason_text(finish_reason),
        parse_error,
        acc["arguments"][:RAW_ARG_LOG_LIMIT],
      )
      yield StreamEvent(
        type="tool_use",
        tool_use_id=acc["id"],
        tool_name=acc["name"],
        tool_input={},
        meta=meta,
      )

    yield StreamEvent(type="end")
