"""FastAPI application for the SR2 Bridge proxy."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from sr2_bridge.adapters import AnthropicAdapter, ExecutionAdapter
from sr2_bridge.config import BridgeConfig
from sr2_bridge.engine import BridgeEngine
from sr2_bridge.forwarder import BridgeForwarder
from sr2_bridge.bridge_metrics import BridgeMetricsExporter
from sr2_bridge.llm import APIKeyCache
from sr2_bridge.session_tracker import BridgeSession, SessionTracker

logger = logging.getLogger(__name__)

_DISCONNECT_POLL_INTERVAL_S = 0.5

# Models considered "fast/small" — if the incoming request uses one of these,
# the bridge rewrites to forwarding.fast_model (or forwarding.model as fallback).
_FAST_MODEL_MARKERS = {"haiku", "flash", "mini", "small", "fast"}


def _is_fast_model(model_name: str) -> bool:
    """Check if a model name looks like a fast/small model."""
    lower = model_name.lower()
    return any(marker in lower for marker in _FAST_MODEL_MARKERS)


def _rewrite_model(body: dict, fwd_config) -> None:
    """Rewrite model field in request body if configured."""
    if not fwd_config.model:
        return
    original_model = body.get("model", "")
    if fwd_config.fast_model and _is_fast_model(original_model):
        body["model"] = fwd_config.fast_model
    else:
        body["model"] = fwd_config.model
    if body["model"] != original_model:
        logger.debug("Model rewrite: %s -> %s", original_model, body["model"])


def _log_token_reduction(
    session_id: str, messages: list[dict], optimized_messages: list[dict]
) -> None:
    """Log token reduction from optimization."""
    original_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
    optimized_tokens = sum(len(str(m.get("content", ""))) // 4 for m in optimized_messages)
    if original_tokens != optimized_tokens:
        logger.info(
            "Session %s: optimized %d -> %d est. tokens (%.0f%% reduction)",
            session_id,
            original_tokens,
            optimized_tokens,
            (1 - optimized_tokens / max(original_tokens, 1)) * 100,
        )


def _build_streaming_response(
    forwarder: BridgeForwarder,
    engine: BridgeEngine,
    adapter: AnthropicAdapter,
    body: dict,
    headers: dict[str, str],
    session: BridgeSession,
    query_params: str | None,
) -> StreamingResponse:
    """Build a streaming response that captures text for post-processing."""
    return StreamingResponse(
        _stream_and_capture(
            forwarder,
            engine,
            adapter,
            body,
            headers,
            session,
            query_params=query_params,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _build_sync_response(
    forwarder: BridgeForwarder,
    session_id: str,
    body: dict,
    headers: dict[str, str],
    query_params: str | None,
) -> Response:
    """Build a non-streaming response."""
    response = await forwarder.forward(
        "/v1/messages",
        body,
        headers,
        query_params=query_params or None,
    )
    if response.status_code >= 400:
        logger.warning(
            "Session %s: upstream returned %d: %s",
            session_id,
            response.status_code,
            response.content[:500].decode("utf-8", errors="replace"),
        )
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )


def create_bridge_app(
    bridge_config: BridgeConfig,
    engine: BridgeEngine,
    forwarder: BridgeForwarder,
    session_tracker: SessionTracker,
    key_cache: APIKeyCache | None = None,
) -> FastAPI:
    """Create the FastAPI bridge proxy application."""

    async def _cleanup_loop():
        """Periodically clean up idle sessions."""
        while True:
            await asyncio.sleep(60)
            expired = session_tracker.cleanup_idle()
            for sid in expired:
                engine.destroy_session(sid)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await forwarder.start()

        # Restore persisted sessions if enabled
        if engine.session_store:
            await engine.session_store.connect()
            for session, zones in await engine.session_store.load_all_sessions():
                session_tracker.restore_session(session)
                engine.conversation_manager.restore_zones(session.session_id, zones)

        logger.info(
            "SR2 Bridge started on %s:%d -> %s",
            bridge_config.host,
            bridge_config.port,
            bridge_config.forwarding.upstream_url,
        )
        cleanup_task = asyncio.create_task(_cleanup_loop())
        try:
            yield
        finally:
            cleanup_task.cancel()
            await engine.shutdown()
            await forwarder.stop()
            logger.info("SR2 Bridge stopped")

    app = FastAPI(title="SR2 Bridge", lifespan=lifespan)
    adapter = AnthropicAdapter(tool_type_overrides=bridge_config.tool_type_overrides or None)
    _key_cache = key_cache or APIKeyCache()

    # Claude Code execution adapter (optional)
    cc_adapter: ExecutionAdapter | None = None
    if bridge_config.claude_code.enabled:
        from sr2_bridge.adapters import get_execution_adapter

        cc_adapter = get_execution_adapter(
            "claude_code",
            bridge_config.claude_code.model_dump(),
        )
        logger.info("Claude Code execution adapter enabled")

    # Track startup time for health endpoint
    start_time = time.time()

    # --- Main proxy route ---

    @app.post("/v1/messages")
    async def proxy_messages(request: Request):
        """Main proxy route: optimize context, forward to upstream, stream back."""
        body = await request.json()
        headers = dict(request.headers)
        is_streaming = body.get("stream", False)

        # Cache API key for bridge-internal LLM calls
        _key_cache.update(headers)

        # Extract messages and identify session
        system, messages = adapter.extract_messages(body)
        session_id = session_tracker.identify(body, headers, system)
        session = session_tracker.get(session_id)

        logger.debug(
            "Session %s: %d messages, stream=%s",
            session_id,
            len(messages),
            is_streaming,
        )

        # Rewrite model if configured
        _rewrite_model(body, bridge_config.forwarding)

        # Optimize context
        try:
            system_injection, optimized_messages = await engine.optimize(
                system=system,
                messages=messages,
                session=session,
                adapter=adapter,
            )
        except Exception:
            logger.exception("Optimization failed for session %s, forwarding original", session_id)
            system_injection = None
            optimized_messages = messages

        # Route to Claude Code adapter if enabled
        if cc_adapter is not None:
            # Combine system injection with original system prompt
            combined_system = system or ""
            if system_injection:
                combined_system = (
                    f"{system_injection}\n\n{combined_system}"
                    if combined_system
                    else system_injection
                )

            try:
                execute_task = asyncio.create_task(
                    cc_adapter.stream_execute(
                        system_prompt=combined_system or None,
                        messages=optimized_messages,
                    )
                )

                # Poll for client disconnect while Claude Code runs
                while not execute_task.done():
                    if await request.is_disconnected():
                        logger.info(
                            "Session %s: client disconnected, cancelling Claude Code",
                            session_id,
                        )
                        execute_task.cancel()
                        try:
                            await execute_task
                        except asyncio.CancelledError:
                            pass
                        return Response(status_code=499)
                    await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)

                loop_result = execute_task.result()
            except Exception:
                logger.exception("Claude Code execution failed for session %s", session_id)
                return JSONResponse(
                    status_code=500,
                    content={"error": "Claude Code execution failed"},
                )

            # Post-process async
            if loop_result.response_text:
                task = asyncio.create_task(engine.post_process(session, loop_result.response_text))
                task.add_done_callback(_log_task_exception)

            # Return as Anthropic Messages API format
            return JSONResponse(
                content={
                    "id": f"msg_cc_{session_id}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": loop_result.response_text}],
                    "model": "claude-code",
                    "stop_reason": "end_turn"
                    if loop_result.stopped_reason == "complete"
                    else "error",
                    "usage": {
                        "input_tokens": loop_result.total_input_tokens,
                        "output_tokens": loop_result.total_output_tokens,
                        "cache_read_input_tokens": loop_result.cached_tokens,
                    },
                }
            )

        # Rebuild body with optimized messages (upstream forwarding path)
        optimized_body = adapter.rebuild_body(body, optimized_messages, system_injection)
        _log_token_reduction(session_id, messages, optimized_messages)

        # Log message structure for debugging 400s
        opt_msg_roles = [m.get("role", "?") for m in optimized_body.get("messages", [])]
        logger.debug(
            "Session %s: forwarding %d messages, roles=%s, model=%s",
            session_id,
            len(opt_msg_roles),
            opt_msg_roles,
            optimized_body.get("model"),
        )

        if is_streaming:
            return _build_streaming_response(
                forwarder,
                engine,
                adapter,
                optimized_body,
                headers,
                session,
                query_params=request.url.query,
            )
        else:
            return await _build_sync_response(
                forwarder,
                session_id,
                optimized_body,
                headers,
                query_params=request.url.query,
            )

    # --- Infrastructure endpoints ---

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - start_time, 1),
            "active_sessions": session_tracker.active_sessions,
            "upstream": bridge_config.forwarding.upstream_url,
        }

    metrics_exporter = BridgeMetricsExporter(engine, session_tracker)

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            content=metrics_exporter.export(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # --- Passthrough routes (config-driven allowlist) ---

    def _make_passthrough_handler(path: str):
        async def passthrough(request: Request):
            body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
            headers = dict(request.headers)
            response = await forwarder.forward_passthrough(request.method, path, body, headers)
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )

        return passthrough

    for passthrough_path in bridge_config.allowed_passthrough_paths:
        app.api_route(
            passthrough_path,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            name=f"passthrough_{passthrough_path.replace('/', '_')}",
        )(_make_passthrough_handler(passthrough_path))

    return app


def _log_task_exception(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Bridge post-processing failed: %s", exc, exc_info=exc)


async def _stream_and_capture(
    forwarder: BridgeForwarder,
    engine: BridgeEngine,
    adapter: AnthropicAdapter,
    body: dict,
    headers: dict[str, str],
    session: BridgeSession,
    query_params: str | None = None,
):
    """Stream SSE from upstream, passthrough each chunk, accumulate response text."""
    accumulated: list[str] = []

    async for chunk in forwarder.forward_streaming(
        "/v1/messages",
        body,
        headers,
        query_params=query_params or None,
    ):
        yield chunk
        text = adapter.parse_sse_text(chunk)
        if text:
            accumulated.append(text)

    # Post-process after stream completes (fire-and-forget with error logging)
    if accumulated:
        full_text = "".join(accumulated)
        task = asyncio.create_task(engine.post_process(session, full_text))
        task.add_done_callback(_log_task_exception)
