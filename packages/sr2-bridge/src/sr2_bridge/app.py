"""FastAPI application for the SR2 Bridge proxy."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from sr2_bridge.adapters.anthropic import AnthropicAdapter, transform_system_prompt
from sr2_bridge.adapters.base import BridgeAdapter
from sr2_bridge.adapters.openai import OpenAIAdapter
from sr2_bridge.config import BridgeConfig
from sr2_bridge.engine import BridgeEngine
from sr2_bridge.forwarder import BridgeForwarder
from sr2_bridge.bridge_metrics import BridgeMetricsExporter
from sr2_bridge.llm import APIKeyCache
from sr2_bridge.request_logger import BridgeRequestLogger
from sr2_bridge.session_tracker import BridgeSession, SessionTracker

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


def _extract_agent_name(headers: dict[str, str]) -> str | None:
    """Extract agent name from headers.

    Checks X-SR2-Agent-Name first. Falls back to parsing the agent name
    from X-SR2-Session-ID when it uses the ``name,X-SR2-Agent-Name: value``
    convention that Claude Code emits.
    """
    agent_name = headers.get("x-sr2-agent-name")
    if agent_name:
        return agent_name

    session_id = headers.get("x-sr2-session-id", "")
    marker = "X-SR2-Agent-Name: "
    idx = session_id.find(marker)
    if idx != -1:
        return session_id[idx + len(marker):]

    return None


def _dump_request(
    direction: str,
    session_id: str,
    body: dict,
    headers: dict | None = None,
    note: str = "",
) -> None:
    """Dump full request for debugging. direction = INCOMING or OUTGOING."""
    system = body.get("system", "")
    system_len = len(json.dumps(system)) if system else 0
    msgs = body.get("messages", [])
    has_reminder = "<system-reminder>" in json.dumps(msgs)

    msg_summaries = []
    for i, m in enumerate(msgs):
        content = m.get("content", "")
        if isinstance(content, list):
            blocks = [
                f"{b.get('type', '?')}({len(b.get('text', ''))}ch)"
                if b.get("type") == "text"
                else b.get("type", "?")
                for b in content
                if isinstance(b, dict)
            ]
            content_desc = f"[{', '.join(blocks)}]"
        else:
            content_desc = f"str({len(content)}ch)"
        msg_summaries.append(f"  msg[{i}] role={m.get('role')}, content={content_desc}")

    header_str = ""
    if headers:
        safe_headers = {
            k: (v[:8] + "..." if k.lower() in ("x-api-key", "authorization") else v)
            for k, v in headers.items()
        }
        header_str = f"\n  headers: {json.dumps(safe_headers, indent=4)}"

    logger.info(
        "\n====== %s %s ======\n"
        "  session: %s\n"
        "  model: %s\n"
        "  stream: %s\n"
        "  system_prompt_length: %d chars\n"
        "  message_count: %d\n"
        "  has_system_reminder_in_msgs: %s\n"
        "%s%s\n"
        "  %s\n"
        "====== /%s ======",
        direction,
        f"({note})" if note else "",
        session_id,
        body.get("model", "?"),
        body.get("stream", "not set"),
        system_len,
        len(msgs),
        has_reminder,
        "\n".join(msg_summaries),
        header_str,
        note or "",
        direction,
    )


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
    agent_name: str | None = None,
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
            agent_name=agent_name,
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

    # Log full response
    try:
        resp_body = json.loads(response.content)
        resp_text = ""
        for block in resp_body.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                resp_text += block.get("text", "")
        usage = resp_body.get("usage", {})
        logger.info(
            "\n====== RESPONSE (non-streaming) ======\n"
            "  session: %s\n"
            "  status: %d\n"
            "  model: %s\n"
            "  stop_reason: %s\n"
            "  usage: input=%s, output=%s, cache_creation=%s, cache_read=%s\n"
            "  response_text_length: %d chars\n"
            "  preview: %s\n"
            "====== /RESPONSE ======",
            session_id,
            response.status_code,
            resp_body.get("model", "?"),
            resp_body.get("stop_reason", "?"),
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
            usage.get("cache_creation_input_tokens", "?"),
            usage.get("cache_read_input_tokens", "?"),
            len(resp_text),
            resp_text[:200].replace("\n", " ") + ("..." if len(resp_text) > 200 else ""),
        )
    except Exception:
        logger.info(
            "RESPONSE (non-streaming): session=%s, status=%d, body_length=%d",
            session_id,
            response.status_code,
            len(response.content),
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
    request_logger: BridgeRequestLogger | None = None,
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

    _pg_pool = None  # Track asyncpg pool for cleanup

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _pg_pool
        await forwarder.start()
        if request_logger:
            await request_logger.start()

        # Wire PostgreSQL memory store if configured (requires async pool creation)
        mem_cfg = bridge_config.memory
        if mem_cfg.enabled and mem_cfg.backend == "postgres" and mem_cfg.database_url:
            try:
                import asyncpg

                _pg_pool = await asyncpg.create_pool(mem_cfg.database_url)
                await engine._sr2.set_postgres_store(_pg_pool)
                logger.info("Bridge memory: PostgreSQL store connected (%s)", mem_cfg.database_url)
            except ImportError:
                logger.warning(
                    "PostgreSQL memory requested but asyncpg/sr2-pro not installed. "
                    "Memory will use in-memory store."
                )
            except Exception:
                logger.error("Failed to connect PostgreSQL memory store", exc_info=True)

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
            if _pg_pool:
                await _pg_pool.close()
            await forwarder.stop()
            if request_logger:
                await request_logger.stop()
            logger.info("SR2 Bridge stopped")

    app = FastAPI(title="SR2 Bridge", lifespan=lifespan)
    anthropic_adapter = AnthropicAdapter(tool_type_overrides=bridge_config.tool_type_overrides or None)
    openai_adapter = OpenAIAdapter(tool_type_overrides=bridge_config.tool_type_overrides or None)
    _key_cache = key_cache or APIKeyCache()

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

        # Extract per-request agent identity for memory scoping
        agent_name = _extract_agent_name(headers)

        # Extract messages and identify session
        system, messages = anthropic_adapter.extract_messages(body)
        session_id = session_tracker.identify(body, headers, system)
        session = session_tracker.get(session_id)

        # Log incoming request
        _dump_request("INCOMING", session_id, body, headers)
        if request_logger:
            request_logger.log_incoming(session_id, system, messages, body)

        # Apply system prompt transform (before SR2 injection)
        if bridge_config.system_prompt.resolved_content:
            original_system = system
            system = transform_system_prompt(system, bridge_config.system_prompt)
            body["system"] = system  # write back so rebuild_body() sees transformed version
            if request_logger:
                request_logger.log_transformed_system(session_id, original_system, system)

        # Rewrite model if configured
        original_model = body.get("model", "")
        _rewrite_model(body, bridge_config.forwarding)

        # Fast/small model requests (haiku, flash, mini) are Claude Code's
        # internal pre-processing (summarization, token counting). Bypass
        # SR2 optimization entirely — just forward with system prompt
        # transform and model rewrite applied. This prevents the haiku
        # preflight from polluting per-message hash state.
        if _is_fast_model(original_model):
            logger.info(
                "Session %s: fast model bypass (%s), forwarding without optimization",
                session_id,
                original_model,
            )
            if is_streaming:
                return _build_streaming_response(
                    forwarder, engine, anthropic_adapter, body, headers,
                    session, query_params=request.url.query, agent_name=agent_name,
                )
            else:
                return await _build_sync_response(
                    forwarder, session_id, body, headers,
                    query_params=request.url.query,
                )

        # Optimize context
        try:
            system_injection, optimized_messages = await engine.optimize(
                system=system,
                messages=messages,
                session=session,
                adapter=anthropic_adapter,
                agent_name=agent_name,
            )
        except Exception:
            logger.exception("Optimization failed for session %s, forwarding original", session_id)
            system_injection = None
            optimized_messages = messages

        # Rebuild body with optimized messages
        optimized_body = anthropic_adapter.rebuild_body(body, optimized_messages, system_injection)
        _dump_request("OUTGOING", session_id, optimized_body, note=f"injection={'yes' if system_injection else 'no'}")
        _log_token_reduction(session_id, messages, optimized_messages)

        # Log outgoing body
        if request_logger:
            request_logger.log_outgoing(session_id, optimized_body)

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
                anthropic_adapter,
                optimized_body,
                headers,
                session,
                query_params=request.url.query,
                agent_name=agent_name,
            )
        else:
            return await _build_sync_response(
                forwarder,
                session_id,
                optimized_body,
                headers,
                query_params=request.url.query,
            )

    # --- OpenAI Chat Completions proxy route ---

    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(request: Request):
        """OpenAI-compatible proxy route: optimize context, forward to upstream, stream back."""
        body = await request.json()
        headers = dict(request.headers)
        is_streaming = body.get("stream", False)

        # Cache API key for bridge-internal LLM calls
        _key_cache.update(headers)

        # Extract agent name for memory scoping (optional)
        agent_name = _extract_agent_name(headers)

        # Extract messages and identify session
        system, messages = openai_adapter.extract_messages(body)
        session_id = session_tracker.identify(body, headers, system)
        session = session_tracker.get(session_id)

        # Log incoming request
        if request_logger:
            request_logger.log_incoming(session_id, system, messages, body)

        logger.debug(
            "Session %s: %d messages (openai), stream=%s, agent=%s",
            session_id,
            len(messages),
            is_streaming,
            agent_name or "(default)",
        )

        # Apply system prompt transform (before SR2 injection)
        if bridge_config.system_prompt.resolved_content:
            original_system = system
            system = transform_system_prompt(system, bridge_config.system_prompt)
            # For OpenAI, system prompt is in messages — no top-level field to update.
            # The rebuild_body() will re-inject the transformed system prompt.
            if request_logger:
                request_logger.log_transformed_system(session_id, original_system, system)

        # Rewrite model if configured
        _rewrite_model(body, bridge_config.forwarding)

        # Optimize context
        try:
            system_injection, optimized_messages = await engine.optimize(
                system=system,
                messages=messages,
                session=session,
                adapter=openai_adapter,
                agent_name=agent_name,
            )
        except Exception:
            logger.exception("Optimization failed for session %s, forwarding original", session_id)
            system_injection = None
            optimized_messages = messages

        # Rebuild body with optimized messages (re-inserts system message)
        optimized_body = openai_adapter.rebuild_body(body, optimized_messages, system_injection)
        _log_token_reduction(session_id, messages, optimized_messages)

        # Log outgoing body
        if request_logger:
            request_logger.log_outgoing(session_id, optimized_body)

        # Log message structure for debugging 400s
        opt_msg_roles = [m.get("role", "?") for m in optimized_body.get("messages", [])]
        logger.debug(
            "Session %s: forwarding %d messages (openai), roles=%s, model=%s",
            session_id,
            len(opt_msg_roles),
            opt_msg_roles,
            optimized_body.get("model"),
        )

        if is_streaming:
            return _build_streaming_response(
                forwarder,
                engine,
                openai_adapter,
                optimized_body,
                headers,
                session,
                query_params=request.url.query,
                upstream_path="/v1/chat/completions",
                agent_name=agent_name,
            )
        else:
            return await _build_sync_response(
                forwarder,
                session_id,
                optimized_body,
                headers,
                query_params=request.url.query,
                upstream_path="/v1/chat/completions",
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
    adapter: BridgeAdapter,
    body: dict,
    headers: dict[str, str],
    session: BridgeSession,
    query_params: str | None = None,
    upstream_path: str = "/v1/messages",
    agent_name: str | None = None,
):
    """Stream SSE from upstream, passthrough each chunk, accumulate response text."""
    accumulated: list[str] = []
    chunk_count = 0
    data_chunks = 0

    async for chunk in forwarder.forward_streaming(
        upstream_path,
        body,
        headers,
        query_params=query_params or None,
    ):
        yield chunk
        chunk_count += 1
        decoded = chunk.decode("utf-8", errors="replace").strip()
        if decoded.startswith("data: "):
            data_chunks += 1
            if chunk_count <= 10 or "text_delta" in decoded:
                logger.debug("SSE chunk #%d: %s", chunk_count, decoded[:200])
        text = adapter.parse_sse_text(chunk)
        if text:
            accumulated.append(text)

    logger.info("SSE stream ended: %d total chunks, %d data chunks, %d text extracted",
                chunk_count, data_chunks, len(accumulated))

    # Log streaming response completion
    full_text = "".join(accumulated) if accumulated else ""
    logger.info(
        "\n====== RESPONSE (streaming) ======\n"
        "  session: %s\n"
        "  accumulated_text_length: %d chars\n"
        "  preview: %s\n"
        "====== /RESPONSE ======",
        session.session_id,
        len(full_text),
        full_text[:200].replace("\n", " ") + ("..." if len(full_text) > 200 else ""),
    )

    # Post-process after stream completes (fire-and-forget with error logging)
    if accumulated:
        task = asyncio.create_task(engine.post_process(session, full_text, agent_name=agent_name))
        task.add_done_callback(_log_task_exception)
