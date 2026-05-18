"""LiteLLM-backed implementation of the LLMCallable protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator

import litellm

from sr2.models import TextBlock, TokenUsage
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


class LiteLLMClient:
    """Concrete LLMCallable that delegates to LiteLLM's acompletion."""

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self._kwargs = kwargs

    def _build_messages(self, request: CompletionRequest) -> list[dict[str, str]]:
        return [
            {"role": msg.role, "content": "".join(block.text for block in msg.content)}
            for msg in request.messages
        ]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        messages = self._build_messages(request)
        extra: dict[str, object] = {}
        if request.system is not None:
            extra["system"] = "".join(block.text for block in request.system)
        resp = await litellm.acompletion(
            model=self.model,
            messages=messages,
            **self._kwargs,
            **extra,
        )
        return CompletionResponse(
            id=resp.id,
            content=[TextBlock(text=resp.choices[0].message.content)],
            stop_reason=resp.choices[0].finish_reason,
            usage=TokenUsage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            ),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        messages = self._build_messages(request)
        extra: dict[str, object] = {}
        if request.system is not None:
            extra["system"] = "".join(block.text for block in request.system)
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=True,
            **self._kwargs,
            **extra,
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield StreamEvent(type="text", text=content)
            if getattr(chunk, "usage", None) is not None:
                yield StreamEvent(
                    type="usage",
                    usage=TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    ),
                )
        yield StreamEvent(type="end")
