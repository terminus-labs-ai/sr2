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
from bridge.session_tracker import BridgeSession, SessionTracker

logger = logging.getLogger(__name__)


def create_bridge_app(
    bridge_config: BridgeConfig,
    engine: BridgeEngine,
    forwarder: BridgeForwarder,
    session_tracker: SessionTracker,
) -> FastAPI:
    """Create the FastAPI bridge proxy application."""
    app = FastAPI(title="SR2 Bridge")
    adapter = AnthropicAdapter()

    # Track startup time for health endpoint
    start_time = time.time()

    # --- Lifecycle ---

    @app.on_event("startup")
    async def startup():
        await forwarder.start()
        logger.info(
            "SR2 Bridge started on %s:%d -> %s",
            bridge_config.host,
            bridge_config.port,
            bridge_config.forwarding.upstream_url,
        )

    @app.on_event("shutdown")
    async def shutdown():
        await forwarder.stop()
        logger.info("SR2 Bridge stopped")

    # --- Main proxy route ---

    @app.post("/v1/messages")
    async def proxy_messages(request: Request):
        """Main proxy route: optimize context, forward to upstream, stream back."""
        body = await request.json()
        headers = dict(request.headers)
        is_streaming = body.get("stream", False)

        # Extract messages and identify session
        system, messages = adapter.extract_messages(body)
        session_id = session_tracker.identify(body, headers, system)
        session = session_tracker.get(session_id)

        logger.debug(
            "Session %s: %d messages, stream=%s",
            session_id, len(messages), is_streaming,
        )

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
                    forwarder, engine, adapter, optimized_body, headers, session
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await forwarder.forward("/v1/messages", optimized_body, headers)
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


    # --- Passthrough routes ---

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        """Passthrough to upstream count_tokens endpoint."""
        body = await request.body()
        headers = dict(request.headers)
        response = await forwarder.forward_passthrough(
            "POST", "/v1/messages/count_tokens", body, headers
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catchall(request: Request, path: str):
        """Catchall passthrough for any other API endpoints."""
        body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
        headers = dict(request.headers)
        response = await forwarder.forward_passthrough(
            request.method, f"/{path}", body, headers
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    return app


async def _stream_and_capture(
    forwarder: BridgeForwarder,
    engine: BridgeEngine,
    adapter: AnthropicAdapter,
    body: dict,
    headers: dict[str, str],
    session: BridgeSession,
):
    """Stream SSE from upstream, passthrough each chunk, accumulate response text."""
    accumulated: list[str] = []

    async for chunk in forwarder.forward_streaming("/v1/messages", body, headers):
        yield chunk
        text = adapter.parse_sse_text(chunk)
        if text:
            accumulated.append(text)

    # Post-process after stream completes (fire-and-forget)
    if accumulated:
        full_text = "".join(accumulated)
        asyncio.create_task(engine.post_process(session, full_text))
