"""FastAPI application for the SR2 Bridge proxy."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from bridge.adapters.anthropic import AnthropicAdapter
from bridge.adapters.base import BridgeAdapter
from bridge.adapters.openai import OpenAIAdapter
from bridge.config import BridgeConfig
from bridge.engine import BridgeEngine
from bridge.forwarder import BridgeForwarder
from bridge.bridge_metrics import BridgeMetricsExporter
from bridge.llm import APIKeyCache
from bridge.session_tracker import BridgeSession, SessionTracker

logger = logging.getLogger(__name__)

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
    adapter: BridgeAdapter,
    body: dict,
    headers: dict[str, str],
    session: BridgeSession,
    query_params: str | None,
    upstream_path: str = "/v1/messages",
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
            upstream_path=upstream_path,
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
    upstream_path: str = "/v1/messages",
) -> Response:
    """Build a non-streaming response."""
    response = await forwarder.forward(
        upstream_path,
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
    anthropic_adapter = AnthropicAdapter(
        tool_type_overrides=bridge_config.tool_type_overrides or None,
    )
    openai_adapter = OpenAIAdapter(
        tool_type_overrides=bridge_config.tool_type_overrides or None,
    )
    _key_cache = key_cache or APIKeyCache()

    # Track startup time for health endpoint
    start_time = time.time()

    # --- Shared proxy logic ---

    async def _proxy_request(
        request: Request,
        adapter: BridgeAdapter,
        upstream_path: str,
    ):
        """Shared proxy handler: optimize context, forward to upstream."""
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

        # Rebuild body with optimized messages
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
                upstream_path=upstream_path,
            )
        else:
            return await _build_sync_response(
                forwarder,
                session_id,
                optimized_body,
                headers,
                query_params=request.url.query,
                upstream_path=upstream_path,
            )

    # --- API routes ---

    @app.post("/v1/messages")
    async def proxy_messages(request: Request):
        """Anthropic Messages API proxy."""
        return await _proxy_request(request, anthropic_adapter, "/v1/messages")

    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(request: Request):
        """OpenAI Chat Completions API proxy."""
        return await _proxy_request(request, openai_adapter, "/v1/chat/completions")

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
    adapter: BridgeAdapter,
    body: dict,
    headers: dict[str, str],
    session: BridgeSession,
    query_params: str | None = None,
    upstream_path: str = "/v1/messages",
):
    """Stream SSE from upstream, passthrough each chunk, accumulate response text."""
    accumulated: list[str] = []

    async for chunk in forwarder.forward_streaming(
        upstream_path,
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
