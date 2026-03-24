"""Tests for the SR2 Bridge proxy.

Covers:
- Adapter: wire-format conversion roundtrips
- Session: identification strategies, idle cleanup, unified state
- Engine: compaction triggering, session reset, format-agnostic contract
- App: HTTP integration via ASGI transport
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from sr2.compaction.engine import ConversationTurn
from sr2.config.models import CompactionConfig, PipelineConfig

from bridge.adapters.anthropic import AnthropicAdapter
from bridge.app import create_bridge_app, _is_fast_model
from bridge.bridge_metrics import BridgeMetricsExporter
from bridge.config import BridgeConfig, BridgeSessionConfig
from bridge.engine import BridgeEngine
from bridge.forwarder import BridgeForwarder
from bridge.session_tracker import BridgeSession, SessionTracker


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    """Test Anthropic wire-format adapter."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    # --- extract_messages ---

    def test_extract_string_system(self):
        body = {"system": "You are helpful.", "messages": [{"role": "user", "content": "Hi"}]}
        system, messages = self.adapter.extract_messages(body)
        assert system == "You are helpful."
        assert len(messages) == 1

    def test_extract_content_block_system(self):
        body = {
            "system": [
                {"type": "text", "text": "Line one."},
                {"type": "text", "text": "Line two."},
            ],
            "messages": [],
        }
        system, messages = self.adapter.extract_messages(body)
        assert "Line one." in system
        assert "Line two." in system
        assert len(messages) == 0

    def test_extract_no_system(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        system, messages = self.adapter.extract_messages(body)
        assert system is None

    # --- rebuild_body ---

    def test_rebuild_preserves_non_message_fields(self):
        body = {"model": "claude-3", "max_tokens": 4096, "messages": [], "system": "original"}
        rebuilt = self.adapter.rebuild_body(body, [{"role": "user", "content": "hi"}], None)
        assert rebuilt["model"] == "claude-3"
        assert rebuilt["max_tokens"] == 4096
        assert len(rebuilt["messages"]) == 1

    def test_rebuild_with_string_system_injection(self):
        body = {"system": "Base prompt.", "messages": []}
        rebuilt = self.adapter.rebuild_body(body, [], "Injected summary.")
        assert rebuilt["system"].startswith("Injected summary.")
        assert "Base prompt." in rebuilt["system"]

    def test_rebuild_with_content_block_system_injection(self):
        body = {"system": [{"type": "text", "text": "Base."}], "messages": []}
        rebuilt = self.adapter.rebuild_body(body, [], "Summary.")
        assert isinstance(rebuilt["system"], list)
        assert rebuilt["system"][0]["text"] == "Summary."
        assert rebuilt["system"][1]["text"] == "Base."

    def test_rebuild_injection_creates_system_when_absent(self):
        body = {"messages": []}
        rebuilt = self.adapter.rebuild_body(body, [], "Summary.")
        assert rebuilt["system"] == "Summary."

    # --- parse_sse_text ---

    def test_parse_text_delta(self):
        chunk = b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n'
        assert self.adapter.parse_sse_text(chunk) == "Hello"

    def test_parse_non_text_event(self):
        chunk = b'data: {"type":"message_start","message":{"id":"msg_123"}}\n'
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_done_signal(self):
        chunk = b"data: [DONE]\n"
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_event_line(self):
        chunk = b"event: content_block_delta\n"
        assert self.adapter.parse_sse_text(chunk) is None

    # --- messages_to_turns ---

    def test_text_message_to_turn(self):
        messages = [{"role": "user", "content": "Hello world"}]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello world"
        assert turns[0].turn_number == 0

    def test_tool_use_message_to_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "name": "search", "input": {"q": "test"}},
                ],
            }
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=5)
        assert len(turns) == 1
        assert turns[0].turn_number == 5
        assert "[tool_use: search" in turns[0].content
        assert turns[0].content_type == "tool_output"

    def test_tool_result_message_to_turn(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "Found 3 results."},
                ],
            }
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert turns[0].content_type == "tool_output"
        assert "Found 3 results." in turns[0].content

    def test_turn_numbering_sequential(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=10)
        assert [t.turn_number for t in turns] == [10, 11, 12]

    def test_original_message_preserved_in_metadata(self):
        msg = {"role": "user", "content": "Hello"}
        turns = self.adapter.messages_to_turns([msg], turn_counter_start=0)
        assert turns[0].metadata["_original_message"] is msg

    # --- turns_to_messages ---

    def test_compacted_turn_becomes_plain_text(self):
        turn = ConversationTurn(
            turn_number=0, role="user", content="Compacted content", compacted=True
        )
        messages = self.adapter.turns_to_messages([turn], [])
        assert messages == [{"role": "user", "content": "Compacted content"}]

    def test_raw_turn_preserves_original(self):
        original = {"role": "user", "content": [{"type": "text", "text": "Rich format"}]}
        turn = ConversationTurn(
            turn_number=0,
            role="user",
            content="Rich format",
            metadata={"_original_message": original},
        )
        messages = self.adapter.turns_to_messages([turn], [original])
        assert messages[0] is original

    def test_roundtrip_preserves_content(self):
        """messages_to_turns → turns_to_messages roundtrip for raw turns."""
        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns = self.adapter.messages_to_turns(original_messages, 0)
        rebuilt = self.adapter.turns_to_messages(turns, original_messages)
        # Raw turns should preserve originals
        assert rebuilt[0] is original_messages[0]
        assert rebuilt[1] is original_messages[1]


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSessionTracker:
    """Test config-driven session identification."""

    def test_uses_config_name(self):
        """Session ID comes from config name."""
        tracker = SessionTracker(BridgeSessionConfig(name="my-project"))
        sid = tracker.identify({}, {})
        assert sid == "my-project"

    def test_default_name(self):
        """Default config name is 'default'."""
        tracker = SessionTracker(BridgeSessionConfig())
        sid = tracker.identify({}, {})
        assert sid == "default"

    def test_header_overrides_config(self):
        """X-SR2-Session-ID header overrides config name."""
        tracker = SessionTracker(BridgeSessionConfig(name="my-project"))
        sid = tracker.identify({}, {"x-sr2-session-id": "custom-session"})
        assert sid == "custom-session"

    def test_cross_client_sharing(self):
        """Different clients with same header share session."""
        tracker = SessionTracker(BridgeSessionConfig(name="my-project"))
        sid1 = tracker.identify({}, {"x-sr2-session-id": "shared", "x-api-key": "key-1"})
        sid2 = tracker.identify({}, {"x-sr2-session-id": "shared", "x-api-key": "key-2"})
        assert sid1 == sid2 == "shared"

    def test_no_header_ignores_system_prompt(self):
        """Without header, different system prompts still get same session."""
        tracker = SessionTracker(BridgeSessionConfig(name="stable"))
        sid1 = tracker.identify({}, {}, system_prompt="Prompt A")
        sid2 = tracker.identify({}, {}, system_prompt="Prompt B")
        assert sid1 == sid2 == "stable"

    def test_idle_cleanup_removes_expired(self):
        tracker = SessionTracker(BridgeSessionConfig(idle_timeout_minutes=1))
        tracker.identify({}, {})
        for session in tracker.all_sessions().values():
            session.last_seen = time.time() - 120
        expired = tracker.cleanup_idle()
        assert len(expired) == 1
        assert tracker.active_sessions == 0

    def test_idle_cleanup_keeps_active(self):
        tracker = SessionTracker(BridgeSessionConfig(idle_timeout_minutes=60))
        tracker.identify({}, {})
        expired = tracker.cleanup_idle()
        assert len(expired) == 0
        assert tracker.active_sessions == 1

    def test_destroy_returns_id(self):
        tracker = SessionTracker(BridgeSessionConfig())
        sid = tracker.identify({}, {})
        result = tracker.destroy(sid)
        assert result == sid
        assert tracker.active_sessions == 0

    def test_destroy_nonexistent_returns_none(self):
        tracker = SessionTracker(BridgeSessionConfig())
        result = tracker.destroy("nonexistent")
        assert result is None


class TestBridgeSession:
    """Test unified session state."""

    def test_touch_increments_request_count(self):
        session = BridgeSession(session_id="test")
        assert session.request_count == 0
        session.touch()
        assert session.request_count == 1

    def test_session_has_turn_fields(self):
        session = BridgeSession(session_id="test")
        assert session.turn_counter == 0
        assert session.turns == []
        assert session.last_message_count == 0


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestBridgeEngine:
    """Test format-agnostic engine logic."""

    def _make_engine(self, raw_window: int = 5) -> BridgeEngine:
        config = PipelineConfig(compaction=CompactionConfig(raw_window=raw_window))
        return BridgeEngine(config)

    def _make_session(self, session_id: str = "test") -> BridgeSession:
        return BridgeSession(session_id=session_id)

    @pytest.mark.asyncio
    async def test_passthrough_on_empty_messages(self):
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()
        injection, messages = await engine.optimize(
            system="Prompt", messages=[], session=session, adapter=adapter
        )
        assert injection is None
        assert messages == []

    @pytest.mark.asyncio
    async def test_messages_tracked_incrementally(self):
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert session.last_message_count == 2
        assert session.turn_counter == 2

        # Add more messages
        msgs.append({"role": "user", "content": "How are you?"})
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert session.last_message_count == 3
        assert session.turn_counter == 3

    @pytest.mark.asyncio
    async def test_session_reset_on_shorter_history(self):
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "More"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert session.last_message_count == 3

        # Shorter history → reset
        shorter = [{"role": "user", "content": "Fresh start"}]
        await engine.optimize(system=None, messages=shorter, session=session, adapter=adapter)
        assert session.last_message_count == 1
        assert session.turn_counter == 1

    @pytest.mark.asyncio
    async def test_compaction_triggered_beyond_raw_window(self):
        engine = self._make_engine(raw_window=2)
        session = self._make_session()
        adapter = AnthropicAdapter()

        # Build up messages beyond raw_window
        msgs = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Turn 3"},
        ]
        injection, optimized = await engine.optimize(
            system=None, messages=msgs, session=session, adapter=adapter
        )
        # Should have some compacted + raw turns
        assert len(optimized) == 5  # all turns present
        metrics = engine.get_session_metrics(session)
        assert metrics["raw_count"] == 2  # raw_window=2
        assert metrics["compacted_count"] == 3  # rest compacted

    @pytest.mark.asyncio
    async def test_engine_never_touches_wire_format(self):
        """Engine delegates all wire format handling to adapter."""
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        # Content blocks (Anthropic-specific format)
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_result", "content": "Result data"},
                ],
            }
        ]
        injection, optimized = await engine.optimize(
            system=None, messages=msgs, session=session, adapter=adapter
        )
        # Engine should return results without error
        assert optimized is not None

    @pytest.mark.asyncio
    async def test_post_process_increments_turn_counter(self):
        engine = self._make_engine()
        session = self._make_session()
        session.turn_counter = 5
        await engine.post_process(session, "Assistant response")
        assert session.turn_counter == 6


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestBridgeMetrics:
    """Test bridge metrics exporter."""

    def test_export_empty(self):
        config = PipelineConfig()
        engine = BridgeEngine(config)
        tracker = SessionTracker(BridgeSessionConfig())
        exporter = BridgeMetricsExporter(engine, tracker)
        output = exporter.export()
        assert "sr2_bridge_active_sessions 0" in output

    def test_export_with_session(self):
        config = PipelineConfig()
        engine = BridgeEngine(config)
        tracker = SessionTracker(BridgeSessionConfig())
        tracker.identify({}, {}, system_prompt="test")
        exporter = BridgeMetricsExporter(engine, tracker)
        output = exporter.export()
        assert "sr2_bridge_active_sessions 1" in output
        assert "sr2_bridge_session_requests" in output
        assert "sr2_bridge_session_tokens" in output


# ---------------------------------------------------------------------------
# App integration tests (ASGI transport, no real network)
# ---------------------------------------------------------------------------

def _make_mock_forwarder() -> BridgeForwarder:
    """Create a mock forwarder that returns canned responses."""
    forwarder = MagicMock(spec=BridgeForwarder)
    forwarder.start = AsyncMock()
    forwarder.stop = AsyncMock()
    forwarder.last_body = None  # Track last forwarded body
    forwarder.response_json = None  # Override response content

    # Non-streaming response — captures body
    async def _capture_forward(path, body, headers, **kwargs):
        forwarder.last_body = body
        resp = MagicMock()
        if forwarder.response_json:
            resp.content = json.dumps(forwarder.response_json).encode()
        else:
            resp.content = json.dumps({"type": "message", "content": [{"type": "text", "text": "Hello"}]}).encode()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        return resp

    forwarder.forward = AsyncMock(side_effect=_capture_forward)

    # Passthrough response
    passthrough_response = MagicMock()
    passthrough_response.content = b'{"token_count": 42}'
    passthrough_response.status_code = 200
    passthrough_response.headers = {"content-type": "application/json"}
    forwarder.forward_passthrough = AsyncMock(return_value=passthrough_response)

    # Streaming: return an async iterator
    async def mock_streaming(*args, **kwargs) -> AsyncIterator[bytes]:
        chunks = [
            b'event: content_block_delta\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n',
            b'data: {"type":"message_stop"}\n',
        ]
        for chunk in chunks:
            yield chunk

    forwarder.forward_streaming = mock_streaming

    return forwarder


def _make_test_app(forwarding: dict | None = None):
    """Create a test app with mock forwarder."""
    from bridge.config import BridgeForwardingConfig
    from bridge.llm import APIKeyCache

    fwd_config = BridgeForwardingConfig(**(forwarding or {}))
    bridge_config = BridgeConfig(forwarding=fwd_config)
    key_cache = APIKeyCache()
    engine = BridgeEngine(PipelineConfig(), bridge_config=bridge_config, key_cache=key_cache)
    forwarder = _make_mock_forwarder()
    tracker = SessionTracker(BridgeSessionConfig())
    app = create_bridge_app(bridge_config, engine, forwarder, tracker, key_cache)
    return app, forwarder


class TestBridgeApp:
    """Integration tests using httpx ASGITransport."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        app, _ = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "uptime_seconds" in data
            assert "active_sessions" in data

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        app, _ = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "sr2_bridge_active_sessions" in response.text

    @pytest.mark.asyncio
    async def test_non_streaming_proxy(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert response.status_code == 200
            forwarder.forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_proxy(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert response.status_code == 200
            # Should receive SSE chunks
            assert "content_block_delta" in response.text or "message_stop" in response.text

    @pytest.mark.asyncio
    async def test_count_tokens_passthrough(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/messages/count_tokens",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
            assert response.status_code == 200
            forwarder.forward_passthrough.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self):
        app, _ = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/unknown/endpoint")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_batches_passthrough(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/messages/batches",
                content=b'{"requests": []}',
            )
            assert response.status_code == 200
            forwarder.forward_passthrough.assert_called()


# ---------------------------------------------------------------------------
# LLM callable factory tests
# ---------------------------------------------------------------------------

class TestAPIKeyCache:
    """Test API key extraction and caching."""

    def test_extract_x_api_key(self):
        from bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"x-api-key": "sk-ant-test123"})
        assert cache.key == "sk-ant-test123"

    def test_extract_bearer_token(self):
        from bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"authorization": "Bearer sk-bearer-key"})
        assert cache.key == "sk-bearer-key"

    def test_x_api_key_takes_precedence(self):
        from bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"x-api-key": "direct-key", "authorization": "Bearer bearer-key"})
        assert cache.key == "direct-key"

    def test_no_key_returns_none(self):
        from bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"content-type": "application/json"})
        assert cache.key is None

    def test_key_updates_on_new_value(self):
        from bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"x-api-key": "key-1"})
        assert cache.key == "key-1"
        cache.update({"x-api-key": "key-2"})
        assert cache.key == "key-2"

    def test_empty_bearer_not_cached(self):
        from bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"authorization": "Bearer "})
        # "Bearer " with no token should not cache empty string
        # Actually, it extracts "" which is falsy, so no update
        assert cache.key is None


class TestLLMCallableFactory:
    """Test callable creation and key resolution."""

    def test_summarization_callable_uses_dedicated_key(self):
        from bridge.config import BridgeLLMModelConfig
        from bridge.llm import APIKeyCache, make_summarization_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="dedicated-key")
        cache = APIKeyCache()
        callable_fn = make_summarization_callable(config, cache, "https://api.example.com")
        # Should be a callable (async function)
        assert callable(callable_fn)

    def test_extraction_callable_created(self):
        from bridge.config import BridgeLLMModelConfig
        from bridge.llm import APIKeyCache, make_extraction_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="key")
        callable_fn = make_extraction_callable(config, APIKeyCache(), "https://api.example.com")
        assert callable(callable_fn)

    def test_intent_callable_created(self):
        from bridge.config import BridgeLLMModelConfig
        from bridge.llm import APIKeyCache, make_intent_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="key")
        callable_fn = make_intent_callable(config, APIKeyCache(), "https://api.example.com")
        assert callable(callable_fn)

    def test_embedding_callable_created(self):
        from bridge.config import BridgeLLMModelConfig
        from bridge.llm import APIKeyCache, make_embedding_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="key")
        callable_fn = make_embedding_callable(config, APIKeyCache(), "https://api.example.com")
        assert callable(callable_fn)

    @pytest.mark.asyncio
    async def test_summarization_raises_without_key(self):
        from bridge.config import BridgeLLMModelConfig
        from bridge.llm import APIKeyCache, make_summarization_callable

        config = BridgeLLMModelConfig(model="test-model")  # no dedicated key
        cache = APIKeyCache()  # no cached key
        callable_fn = make_summarization_callable(config, cache, "https://api.example.com")
        with pytest.raises(RuntimeError, match="No API key"):
            await callable_fn("system", "prompt")

    @pytest.mark.asyncio
    async def test_extraction_raises_without_key(self):
        from bridge.config import BridgeLLMModelConfig
        from bridge.llm import APIKeyCache, make_extraction_callable

        config = BridgeLLMModelConfig(model="test-model")
        callable_fn = make_extraction_callable(config, APIKeyCache(), "https://api.example.com")
        with pytest.raises(RuntimeError, match="No API key"):
            await callable_fn("prompt")


# ---------------------------------------------------------------------------
# Circuit breaker + degradation integration tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerIntegration:
    """Test circuit breaker and degradation ladder in engine."""

    def _make_engine_with_summarization(self):
        """Create engine with a failing summarization callable."""
        from bridge.config import BridgeLLMConfig, BridgeLLMModelConfig, BridgeDegradationConfig

        config = PipelineConfig(compaction=CompactionConfig(raw_window=2))
        bridge_config = BridgeConfig(
            llm=BridgeLLMConfig(
                summarization=BridgeLLMModelConfig(
                    model="test-model",
                    api_key="test-key",
                ),
            ),
            degradation=BridgeDegradationConfig(
                circuit_breaker_threshold=2,
                circuit_breaker_cooldown_seconds=3600,
            ),
        )
        from bridge.llm import APIKeyCache
        engine = BridgeEngine(config, bridge_config=bridge_config, key_cache=APIKeyCache())
        return engine

    def test_degradation_starts_at_full(self):
        engine = self._make_engine_with_summarization()
        assert engine.degradation_level == "full"

    def test_circuit_breaker_status_exposed(self):
        engine = self._make_engine_with_summarization()
        status = engine.circuit_breaker_status
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# Memory config + engine wiring tests
# ---------------------------------------------------------------------------

class TestMemoryConfig:
    """Test memory configuration models."""

    def test_defaults(self):
        from bridge.config import BridgeMemoryConfig
        cfg = BridgeMemoryConfig()
        assert cfg.enabled is False
        assert cfg.db_path == "sr2_bridge_memory.db"
        assert cfg.retrieval_strategy == "keyword"
        assert cfg.retrieval_top_k == 10

    def test_memory_disabled_by_default_in_engine(self):
        engine = BridgeEngine(PipelineConfig())
        assert engine._memory_store is None
        assert engine._memory_extractor is None

    def test_memory_store_created_when_enabled(self):
        from bridge.config import (
            BridgeLLMConfig,
            BridgeLLMModelConfig,
            BridgeMemoryConfig,
        )
        from bridge.llm import APIKeyCache

        bridge_config = BridgeConfig(
            memory=BridgeMemoryConfig(enabled=True, db_path=":memory:"),
            llm=BridgeLLMConfig(
                extraction=BridgeLLMModelConfig(model="test", api_key="key"),
            ),
        )
        engine = BridgeEngine(
            PipelineConfig(),
            bridge_config=bridge_config,
            key_cache=APIKeyCache(),
        )
        assert engine._memory_store is not None
        assert engine._memory_initialized is False  # lazy init

    def test_memory_not_created_without_extraction_config(self):
        from bridge.config import BridgeMemoryConfig
        from bridge.llm import APIKeyCache

        bridge_config = BridgeConfig(
            memory=BridgeMemoryConfig(enabled=True),
            # No llm.extraction configured
        )
        engine = BridgeEngine(
            PipelineConfig(),
            bridge_config=bridge_config,
            key_cache=APIKeyCache(),
        )
        assert engine._memory_store is None


# ---------------------------------------------------------------------------
# Engine memory extraction + retrieval tests
# ---------------------------------------------------------------------------

class TestEngineMemoryExtraction:
    """Test memory extraction in post_process."""

    @pytest.mark.asyncio
    async def test_post_process_without_memory_still_works(self):
        """Engine without memory config should still increment turn counter."""
        engine = BridgeEngine(PipelineConfig())
        session = BridgeSession(session_id="test")
        session.turn_counter = 3
        await engine.post_process(session, "response text")
        assert session.turn_counter == 4

    @pytest.mark.asyncio
    async def test_memory_lazy_initialization(self):
        """_ensure_memory_initialized should be idempotent."""
        from bridge.config import BridgeLLMConfig, BridgeLLMModelConfig, BridgeMemoryConfig
        from bridge.llm import APIKeyCache

        bridge_config = BridgeConfig(
            memory=BridgeMemoryConfig(enabled=True, db_path=":memory:"),
            llm=BridgeLLMConfig(
                extraction=BridgeLLMModelConfig(model="test", api_key="key"),
            ),
        )
        key_cache = APIKeyCache()
        key_cache.update({"x-api-key": "test-key"})
        engine = BridgeEngine(PipelineConfig(), bridge_config=bridge_config, key_cache=key_cache)

        assert not engine._memory_initialized
        await engine._ensure_memory_initialized()
        assert engine._memory_initialized
        assert engine._memory_extractor is not None
        assert engine._conflict_detector is not None
        assert engine._conflict_resolver is not None
        assert engine._retriever is not None

        # Second call is a no-op
        await engine._ensure_memory_initialized()
        assert engine._memory_initialized

        await engine.shutdown()


class TestEngineRetrievalQuery:
    """Test _extract_retrieval_query static method."""

    def test_extracts_string_content(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What is Python?"},
        ]
        query = BridgeEngine._extract_retrieval_query(messages)
        assert query == "What is Python?"

    def test_extracts_content_block_text(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me about"},
                    {"type": "text", "text": "machine learning"},
                ],
            },
        ]
        query = BridgeEngine._extract_retrieval_query(messages)
        assert "Tell me about" in query
        assert "machine learning" in query

    def test_skips_non_user_messages(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
        ]
        query = BridgeEngine._extract_retrieval_query(messages)
        assert query == "First question"

    def test_returns_none_for_empty(self):
        assert BridgeEngine._extract_retrieval_query([]) is None

    def test_returns_none_for_assistant_only(self):
        messages = [{"role": "assistant", "content": "Only assistant"}]
        assert BridgeEngine._extract_retrieval_query(messages) is None

    def test_caps_at_500_chars(self):
        messages = [{"role": "user", "content": "x" * 1000}]
        query = BridgeEngine._extract_retrieval_query(messages)
        assert len(query) == 500

    def test_tool_result_content_block(self):
        """Tool result messages should not be picked as retrieval query."""
        messages = [
            {"role": "user", "content": "A question"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "Result data"}],
            },
        ]
        # Last user message has no text blocks, so falls through to the earlier one
        query = BridgeEngine._extract_retrieval_query(messages)
        assert query == "A question"


class TestModelRewriting:
    """Test model rewriting in the bridge proxy."""

    def test_is_fast_model_haiku(self):
        assert _is_fast_model("claude-haiku-4-5-20251001")

    def test_is_fast_model_flash(self):
        assert _is_fast_model("openai/glm-4.7-flash-cpu")

    def test_is_fast_model_mini(self):
        assert _is_fast_model("gpt-4o-mini")

    def test_is_not_fast_model(self):
        assert not _is_fast_model("claude-sonnet-4-20250514")
        assert not _is_fast_model("openai/qwen-32b")

    @pytest.mark.asyncio
    async def test_model_rewritten_in_proxy(self):
        """When forwarding.model is set, body model gets rewritten."""
        app, forwarder = _make_test_app(
            forwarding={"model": "openai/my-model", "fast_model": "openai/my-fast"},
        )
        forwarder.response_json = {"id": "1", "content": [{"type": "text", "text": "hi"}]}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "hello"}],
            }, headers={"x-api-key": "test"})
            assert resp.status_code == 200
            # Check that forwarder received the rewritten model
            assert forwarder.last_body["model"] == "openai/my-model"

    @pytest.mark.asyncio
    async def test_fast_model_rewritten_in_proxy(self):
        """Fast models get rewritten to fast_model config."""
        app, forwarder = _make_test_app(
            forwarding={"model": "openai/my-model", "fast_model": "openai/my-fast"},
        )
        forwarder.response_json = {"id": "1", "content": [{"type": "text", "text": "hi"}]}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/messages", json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hello"}],
            }, headers={"x-api-key": "test"})
            assert resp.status_code == 200
            assert forwarder.last_body["model"] == "openai/my-fast"

    @pytest.mark.asyncio
    async def test_no_model_config_passthrough(self):
        """Without model config, the original model passes through."""
        app, forwarder = _make_test_app()
        forwarder.response_json = {"id": "1", "content": [{"type": "text", "text": "hi"}]}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "hello"}],
            }, headers={"x-api-key": "test"})
            assert resp.status_code == 200
            assert forwarder.last_body["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_fast_model_falls_back_to_model(self):
        """When fast_model is None, fast models use the base model."""
        app, forwarder = _make_test_app(
            forwarding={"model": "openai/my-model"},
        )
        forwarder.response_json = {"id": "1", "content": [{"type": "text", "text": "hi"}]}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/messages", json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hello"}],
            }, headers={"x-api-key": "test"})
            assert resp.status_code == 200
            assert forwarder.last_body["model"] == "openai/my-model"


class TestEngineShutdown:
    """Test engine shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_shutdown_without_memory(self):
        """Shutdown is safe when memory is not configured."""
        engine = BridgeEngine(PipelineConfig())
        await engine.shutdown()  # should not raise

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_memory_store(self):
        from bridge.config import BridgeLLMConfig, BridgeLLMModelConfig, BridgeMemoryConfig
        from bridge.llm import APIKeyCache

        bridge_config = BridgeConfig(
            memory=BridgeMemoryConfig(enabled=True, db_path=":memory:"),
            llm=BridgeLLMConfig(
                extraction=BridgeLLMModelConfig(model="test", api_key="key"),
            ),
        )
        key_cache = APIKeyCache()
        key_cache.update({"x-api-key": "test-key"})
        engine = BridgeEngine(PipelineConfig(), bridge_config=bridge_config, key_cache=key_cache)

        # Initialize and then shutdown
        await engine._ensure_memory_initialized()
        assert engine._memory_store._conn is not None
        await engine.shutdown()
        assert engine._memory_store._conn is None
