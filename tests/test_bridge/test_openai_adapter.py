"""Tests for the OpenAI Chat Completions adapter.

Covers:
- extract_messages: system/developer role extraction
- rebuild_body: system injection, field preservation
- parse_sse_text: OpenAI SSE delta parsing
- messages_to_turns: text, tool_calls, tool role conversion
- turns_to_messages: compacted/raw roundtrip, orphan handling
- App integration: /v1/chat/completions route
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from sr2.compaction.engine import ConversationTurn
from sr2.config.models import PipelineConfig

from bridge.adapters.openai import OpenAIAdapter
from bridge.app import create_bridge_app
from bridge.config import BridgeConfig, BridgeForwardingConfig, BridgeSessionConfig
from bridge.engine import BridgeEngine
from bridge.forwarder import BridgeForwarder
from bridge.llm import APIKeyCache
from bridge.session_tracker import SessionTracker


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


class TestOpenAIAdapterExtract:
    """Test extract_messages."""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_extract_system_message(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        }
        system, messages = self.adapter.extract_messages(body)
        assert system == "You are helpful."
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_extract_developer_role(self):
        body = {
            "messages": [
                {"role": "developer", "content": "Internal instructions."},
                {"role": "user", "content": "Hi"},
            ]
        }
        system, messages = self.adapter.extract_messages(body)
        assert system == "Internal instructions."
        assert len(messages) == 1

    def test_extract_multiple_system_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "Part one."},
                {"role": "system", "content": "Part two."},
                {"role": "user", "content": "Hi"},
            ]
        }
        system, messages = self.adapter.extract_messages(body)
        assert "Part one." in system
        assert "Part two." in system
        assert len(messages) == 1

    def test_extract_no_system(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        system, messages = self.adapter.extract_messages(body)
        assert system is None
        assert len(messages) == 1

    def test_extract_content_parts_system(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Line one."},
                        {"type": "text", "text": "Line two."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ]
        }
        system, messages = self.adapter.extract_messages(body)
        assert "Line one." in system
        assert "Line two." in system

    def test_extract_empty_messages(self):
        body = {"messages": []}
        system, messages = self.adapter.extract_messages(body)
        assert system is None
        assert messages == []


class TestOpenAIAdapterRebuild:
    """Test rebuild_body."""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_preserves_non_message_fields(self):
        body = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        rebuilt = self.adapter.rebuild_body(
            body, [{"role": "user", "content": "Hi"}], None
        )
        assert rebuilt["model"] == "gpt-4"
        assert rebuilt["temperature"] == 0.7
        assert rebuilt["max_tokens"] == 100

    def test_rebuild_restores_system_message(self):
        body = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        }
        rebuilt = self.adapter.rebuild_body(
            body, [{"role": "user", "content": "Hi"}], None
        )
        assert rebuilt["messages"][0]["role"] == "system"
        assert rebuilt["messages"][0]["content"] == "Be helpful."
        assert rebuilt["messages"][1]["role"] == "user"

    def test_rebuild_with_system_injection(self):
        body = {
            "messages": [
                {"role": "system", "content": "Base prompt."},
                {"role": "user", "content": "Hi"},
            ]
        }
        rebuilt = self.adapter.rebuild_body(
            body, [{"role": "user", "content": "Hi"}], "Injected summary."
        )
        system_msg = rebuilt["messages"][0]
        assert system_msg["role"] == "system"
        assert "Injected summary." in system_msg["content"]
        assert "Base prompt." in system_msg["content"]

    def test_rebuild_injection_creates_system_when_absent(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        rebuilt = self.adapter.rebuild_body(
            body, [{"role": "user", "content": "Hi"}], "Summary."
        )
        assert rebuilt["messages"][0] == {"role": "system", "content": "Summary."}

    def test_rebuild_no_system_no_injection(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        rebuilt = self.adapter.rebuild_body(
            body, [{"role": "user", "content": "Hi"}], None
        )
        assert len(rebuilt["messages"]) == 1
        assert rebuilt["messages"][0]["role"] == "user"


class TestOpenAIAdapterSSE:
    """Test parse_sse_text."""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_parse_content_delta(self):
        chunk = b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n'
        assert self.adapter.parse_sse_text(chunk) == "Hello"

    def test_parse_no_content_delta(self):
        chunk = b'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n'
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_done_signal(self):
        chunk = b"data: [DONE]\n"
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_event_line(self):
        chunk = b"event: message\n"
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_empty_choices(self):
        chunk = b'data: {"choices":[]}\n'
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_malformed_json(self):
        chunk = b"data: {bad json}\n"
        assert self.adapter.parse_sse_text(chunk) is None


class TestOpenAIAdapterTurns:
    """Test messages_to_turns and turns_to_messages."""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    # --- messages_to_turns ---

    def test_text_message_to_turn(self):
        messages = [{"role": "user", "content": "Hello world"}]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello world"
        assert turns[0].turn_number == 0

    def test_turn_numbering_sequential(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=10)
        assert [t.turn_number for t in turns] == [10, 11, 12]

    def test_tool_calls_message_to_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            }
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert len(turns) == 1
        assert "[tool_call: bash" in turns[0].content
        assert turns[0].content_type == "code_execution"

    def test_tool_result_message_to_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "file contents here",
            },
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert turns[1].content_type == "file_content"
        assert "file contents here" in turns[1].content
        assert turns[1].metadata.get("tool_name") == "read"

    def test_original_message_preserved_in_metadata(self):
        msg = {"role": "user", "content": "Hello"}
        turns = self.adapter.messages_to_turns([msg], turn_counter_start=0)
        assert turns[0].metadata["_original_message"] is msg

    def test_content_parts_array(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert "Look at this" in turns[0].content

    # --- turns_to_messages ---

    def test_compacted_turn_becomes_plain_text(self):
        turn = ConversationTurn(
            turn_number=0, role="user", content="Compacted content", compacted=True
        )
        messages = self.adapter.turns_to_messages([turn], [])
        assert messages == [{"role": "user", "content": "Compacted content"}]

    def test_compacted_tool_turn_becomes_assistant(self):
        """Compacted tool-role turns become assistant (tool role needs tool_call_id)."""
        turn = ConversationTurn(
            turn_number=0, role="tool", content="[tool_result: bash]\noutput", compacted=True
        )
        messages = self.adapter.turns_to_messages([turn], [])
        assert messages[0]["role"] == "assistant"

    def test_raw_turn_preserves_original(self):
        original = {"role": "user", "content": "Hello"}
        turn = ConversationTurn(
            turn_number=0,
            role="user",
            content="Hello",
            metadata={"_original_message": original},
        )
        messages = self.adapter.turns_to_messages([turn], [original])
        assert messages[0] is original

    def test_orphaned_tool_result_flattened(self):
        """Tool result referencing a compacted tool_call gets flattened."""
        # First: compacted assistant turn (tool_calls were compacted away)
        compacted_assistant = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="[tool_call: bash(ls)]",
            compacted=True,
        )
        # Second: raw tool result referencing the now-missing call_1
        tool_msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file1.txt\nfile2.txt",
        }
        tool_turn = ConversationTurn(
            turn_number=1,
            role="tool",
            content="[tool_result: bash]\nfile1.txt\nfile2.txt",
            metadata={"_original_message": tool_msg},
        )
        messages = self.adapter.turns_to_messages(
            [compacted_assistant, tool_turn], []
        )
        # The tool result should be flattened to assistant role
        assert messages[1]["role"] == "assistant"
        assert "tool_call_id" not in messages[1]

    def test_roundtrip_preserves_content(self):
        """messages_to_turns -> turns_to_messages roundtrip for raw turns."""
        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns = self.adapter.messages_to_turns(original_messages, 0)
        rebuilt = self.adapter.turns_to_messages(turns, original_messages)
        assert rebuilt[0] is original_messages[0]
        assert rebuilt[1] is original_messages[1]

    def test_roundtrip_with_tool_calls(self):
        """Roundtrip with tool_calls preserves structure."""
        original_messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
            {"role": "assistant", "content": "I see file1.txt"},
        ]
        turns = self.adapter.messages_to_turns(original_messages, 0)
        rebuilt = self.adapter.turns_to_messages(turns, original_messages)
        assert len(rebuilt) == 4
        assert rebuilt[0] is original_messages[0]
        assert rebuilt[1] is original_messages[1]
        assert rebuilt[2] is original_messages[2]
        assert rebuilt[3] is original_messages[3]


# ---------------------------------------------------------------------------
# App integration tests
# ---------------------------------------------------------------------------


def _make_mock_forwarder() -> BridgeForwarder:
    """Create a mock forwarder returning OpenAI-format responses."""
    forwarder = MagicMock(spec=BridgeForwarder)
    forwarder.start = AsyncMock()
    forwarder.stop = AsyncMock()
    forwarder.last_body = None
    forwarder.last_path = None

    async def _capture_forward(path, body, headers, **kwargs):
        forwarder.last_body = body
        forwarder.last_path = path
        resp = MagicMock()
        resp.content = json.dumps(
            {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello"}, "index": 0}
                ],
            }
        ).encode()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        return resp

    forwarder.forward = AsyncMock(side_effect=_capture_forward)

    async def mock_streaming(*args, **kwargs) -> AsyncIterator[bytes]:
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n',
            b'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n',
            b"data: [DONE]\n",
        ]
        for chunk in chunks:
            yield chunk

    forwarder.forward_streaming = mock_streaming

    passthrough_response = MagicMock()
    passthrough_response.content = b"{}"
    passthrough_response.status_code = 200
    passthrough_response.headers = {"content-type": "application/json"}
    forwarder.forward_passthrough = AsyncMock(return_value=passthrough_response)

    return forwarder


def _make_test_app():
    fwd_config = BridgeForwardingConfig()
    bridge_config = BridgeConfig(forwarding=fwd_config)
    key_cache = APIKeyCache()
    engine = BridgeEngine(PipelineConfig(), bridge_config=bridge_config, key_cache=key_cache)
    forwarder = _make_mock_forwarder()
    tracker = SessionTracker(BridgeSessionConfig())
    app = create_bridge_app(bridge_config, engine, forwarder, tracker, key_cache)
    return app, forwarder


class TestOpenAIChatCompletionsRoute:
    """Integration tests for /v1/chat/completions endpoint."""

    @pytest.mark.asyncio
    async def test_non_streaming(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-coder",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hi"},
                    ],
                },
            )
            assert response.status_code == 200
            forwarder.forward.assert_called_once()
            # Verify forwarded to correct upstream path
            assert forwarder.last_path == "/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_streaming(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-coder",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert response.status_code == 200
            assert "Hi" in response.text or "[DONE]" in response.text

    @pytest.mark.asyncio
    async def test_system_message_extracted_and_restored(self):
        """System message should be extracted for optimization, then restored."""
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-coder",
                    "messages": [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Hi"},
                    ],
                },
            )
            forwarded = forwarder.last_body
            # System should be restored as a system-role message
            roles = [m["role"] for m in forwarded["messages"]]
            assert "system" in roles

    @pytest.mark.asyncio
    async def test_model_preserved(self):
        app, forwarder = _make_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-coder:30b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert forwarder.last_body["model"] == "qwen3-coder:30b"
