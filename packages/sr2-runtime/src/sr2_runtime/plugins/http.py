"""HTTP API interface plugin."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Coroutine
from typing import Any, TypeVar

from sr2_runtime.llm.streaming import (
    StreamEndEvent,
    StreamEvent,
    StreamRetractEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from sr2_runtime.plugins.base import TriggerContext

logger = logging.getLogger(__name__)

_DISCONNECT_POLL_INTERVAL_S = 0.5

_T = TypeVar("_T")


class HTTPPlugin:
    """HTTP API interface plugin.

    Provides FastAPI routes. Started by the Agent when creating the HTTP app.
    Unlike other plugins, this one doesn't run its own event loop — it
    registers routes on the shared FastAPI app.

    Config:
        plugin: http
        port: 8008
        session:
          name: "{request.session_id}"
          lifecycle: persistent
        pipeline: interfaces/user_message.yaml
    """

    def __init__(self, interface_name: str, config: dict, agent_callback):
        self._name = interface_name
        self._config = config
        self._callback = agent_callback
        self._session_config = config.get("session", {})
        self._port = config.get("port", 8008)
        self._agent_name = config.get("agent_name", "agent")

    async def start(self) -> None:
        """HTTP plugin doesn't self-start. Routes are registered in create_app()."""
        logger.info(f"HTTP plugin '{self._name}' registered (port {self._port})")

    async def stop(self) -> None:
        pass

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """HTTP can't proactively send. No-op."""
        pass

    def _resolve_session(self, session_id: str) -> tuple[str, str]:
        """Return (session_name, lifecycle) for a given request session_id."""
        session_name = self._session_config.get("name", session_id)
        if "{request.session_id}" in session_name:
            session_name = session_name.replace("{request.session_id}", session_id)
        lifecycle = self._session_config.get("lifecycle", "persistent")
        return session_name, lifecycle

    async def _run_with_disconnect_guard(
        self, request: Any, coro: "Coroutine[Any, Any, _T]"
    ) -> "_T | None":
        """Run a coroutine, cancelling it if the HTTP client disconnects."""
        task = asyncio.create_task(coro)
        while not task.done():
            if await request.is_disconnected():
                logger.info("HTTP client disconnected, cancelling agent task")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return None
            await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)
        return task.result()

    def get_routes(self):
        """Return FastAPI route handlers for the Agent to mount."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        async def chat(request: Request):
            body = await request.json()
            message = body.get("message", "")
            session_id = body.get("session_id", "http_default")
            session_name, lifecycle = self._resolve_session(session_id)

            wants_stream = "text/event-stream" in request.headers.get("accept", "")

            if wants_stream:
                return await self._streaming_chat_response(
                    message,
                    session_name,
                    lifecycle,
                    session_id,
                    wrapper=_wrap_chat_sse,
                )

            trigger = TriggerContext(
                interface_name=self._name,
                plugin_name="http",
                session_name=session_name,
                session_lifecycle=lifecycle,
                input_data=message,
                metadata={"session_id": session_id},
            )
            response = await self._run_with_disconnect_guard(request, self._callback(trigger))
            if response is None:
                from fastapi.responses import Response

                return Response(status_code=499)
            return JSONResponse({"response": response})

        async def openai_chat(request: Request):
            body = await request.json()
            messages = body.get("messages", [])

            # Extract the last user message
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        user_message = " ".join(
                            p.get("text", "") for p in content if p.get("type") == "text"
                        )
                    else:
                        user_message = content
                    break

            session_id = f"openai_{body.get('model', 'default')}"
            session_name, lifecycle = self._resolve_session(session_id)
            model_id = body.get("model", f"sr2-{self._agent_name}")
            wants_stream = body.get("stream", False)

            if wants_stream:
                return await self._streaming_chat_response(
                    user_message,
                    session_name,
                    lifecycle,
                    session_id,
                    wrapper=lambda: _wrap_openai_sse(model_id),
                )

            trigger = TriggerContext(
                interface_name=self._name,
                plugin_name="http",
                session_name=session_name,
                session_lifecycle=lifecycle,
                input_data=user_message,
                metadata={"session_id": session_id},
            )
            response = await self._run_with_disconnect_guard(request, self._callback(trigger))
            if response is None:
                from fastapi.responses import Response

                return Response(status_code=499)
            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            )

        async def openai_models(request: Request):
            model_id = f"sr2-{self._agent_name}"
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "sr2",
                        }
                    ],
                }
            )

        return {
            "chat": chat,
            "openai_chat": openai_chat,
            "openai_models": openai_models,
        }

    async def _streaming_chat_response(
        self,
        message: str,
        session_name: str,
        lifecycle: str,
        session_id: str,
        wrapper,
    ):
        """Run trigger with a streaming callback and return a StreamingResponse."""
        from fastapi.responses import StreamingResponse

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        async def stream_callback(event: StreamEvent) -> None:
            await queue.put(event)

        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="http",
            session_name=session_name,
            session_lifecycle=lifecycle,
            input_data=message,
            metadata={"session_id": session_id},
            respond_callback=stream_callback,
        )

        # Run the trigger in a background task so we can yield events as they arrive
        async def run_trigger():
            try:
                await self._callback(trigger)
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(run_trigger())

        format_event = wrapper() if callable(wrapper) else wrapper

        async def event_generator():
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    sse_data = format_event(event)
                    if sse_data is not None:
                        yield sse_data
                # Yield final done marker
                yield format_event(None)
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def _wrap_chat_sse():
    """Return a formatter that converts StreamEvents to SSE lines for /chat."""

    def format_event(event: StreamEvent | None) -> str | None:
        if event is None:
            return "data: [DONE]\n\n"
        if isinstance(event, TextDeltaEvent):
            return f"data: {json.dumps({'type': 'text', 'content': event.content})}\n\n"
        if isinstance(event, ToolStartEvent):
            return f"data: {json.dumps({'type': 'tool_start', 'tool': event.tool_name})}\n\n"
        if isinstance(event, ToolResultEvent):
            return f"data: {json.dumps({'type': 'tool_result', 'tool': event.tool_name, 'success': event.success})}\n\n"
        if isinstance(event, StreamEndEvent):
            return f"data: {json.dumps({'type': 'end', 'content': event.full_text})}\n\n"
        if isinstance(event, StreamRetractEvent):
            return f"data: {json.dumps({'type': 'retract'})}\n\n"
        return None

    return format_event


def _wrap_openai_sse(model_id: str):
    """Return a formatter that converts StreamEvents to OpenAI-compatible SSE chunks."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def format_event(event: StreamEvent | None) -> str | None:
        if event is None:
            return "data: [DONE]\n\n"
        if isinstance(event, TextDeltaEvent):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event.content},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(chunk)}\n\n"
        if isinstance(event, StreamEndEvent):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            return f"data: {json.dumps(chunk)}\n\n"
        # Tool events and retractions are not part of the OpenAI streaming spec
        return None

    return format_event
