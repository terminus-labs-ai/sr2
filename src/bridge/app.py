"""FastAPI application for the SR2 Bridge proxy."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from bridge.adapters.anthropic import AnthropicAdapter
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
    adapter = AnthropicAdapter()
    _key_cache = key_cache or APIKeyCache()

    # Track startup time for health endpoint
    start_time = time.time()

    # Lifecycle handled via lifespan context manager (see below)

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
            session_id, len(messages), is_streaming,
        )

        # Rewrite model if configured
        fwd = bridge_config.forwarding
        if fwd.model:
            original_model = body.get("model", "")
            if fwd.fast_model and _is_fast_model(original_model):
                body["model"] = fwd.fast_model
            else:
                body["model"] = fwd.model
            if body["model"] != original_model:
                logger.debug("Model rewrite: %s -> %s", original_model, body["model"])

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

        if is_streaming:
            return StreamingResponse(
                _stream_and_capture(
                    forwarder, engine, adapter, optimized_body, headers, session,
                    query_params=request.url.query,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await forwarder.forward(
                "/v1/messages", optimized_body, headers,
                query_params=request.url.query or None,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )

    # --- Infrastructure endpoints (must be before catchall) ---

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
            response = await forwarder.forward_passthrough(
                request.method, path, body, headers
            )
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
        "/v1/messages", body, headers, query_params=query_params or None,
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
