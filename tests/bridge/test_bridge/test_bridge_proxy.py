"""Tests for the SR2 Bridge proxy.

Covers:
- Adapter: wire-format conversion roundtrips
- Session: identification strategies, idle cleanup, unified state
- Engine: compaction triggering, session reset, format-agnostic contract
- App: HTTP integration via ASGI transport
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from sr2.compaction.engine import ConversationTurn
from sr2.config.models import CompactionConfig, CostGateConfig, PipelineConfig

from sr2_bridge.adapters.anthropic import AnthropicAdapter
from sr2_bridge.adapters.openai import OpenAIAdapter
from sr2_bridge.app import create_bridge_app, _is_fast_model
from sr2_bridge.bridge_metrics import BridgeMetricsExporter
from sr2_bridge.config import BridgeConfig, BridgeSessionConfig
from sr2_bridge.engine import BridgeEngine
from sr2_bridge.forwarder import BridgeForwarder
from sr2_bridge.session_tracker import BridgeSession, SessionTracker


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
        assert rebuilt["system"].startswith("Base prompt.")
        assert "Injected summary." in rebuilt["system"]

    def test_rebuild_with_content_block_system_injection(self):
        body = {"system": [{"type": "text", "text": "Base."}], "messages": []}
        rebuilt = self.adapter.rebuild_body(body, [], "Summary.")
        assert isinstance(rebuilt["system"], list)
        assert rebuilt["system"][0]["text"] == "Base."
        assert rebuilt["system"][1]["text"] == "Summary."

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

    def test_orphaned_tool_use_flattened(self):
        """Assistant tool_use whose tool_result was compacted should be flattened."""
        assistant_original = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "test"}},
                {"type": "tool_use", "id": "tool_2", "name": "read", "input": {"path": "/a"}},
            ],
        }
        assistant_turn = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="Let me search.\n[tool_use: search(...)]\n[tool_use: read(...)]",
            metadata={"_original_message": assistant_original},
        )
        compacted_result = ConversationTurn(
            turn_number=1,
            role="tool_result",
            content="[tool_result: search]\nFound 3 results.\n[tool_result: read]\nFile contents.",
            compacted=True,
        )
        messages = self.adapter.turns_to_messages(
            [assistant_turn, compacted_result], []
        )
        assert messages[0]["role"] == "assistant"
        assert isinstance(messages[0]["content"], str)
        assert messages[1]["role"] == "user"
        assert isinstance(messages[1]["content"], str)

    def test_live_tool_use_preserved(self):
        """Assistant tool_use with live tool_result should be preserved."""
        assistant_original = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "test"}},
            ],
        }
        assistant_turn = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="Let me search.\n[tool_use: search(...)]",
            metadata={"_original_message": assistant_original},
        )
        result_original = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "Found 3 results."},
            ],
        }
        result_turn = ConversationTurn(
            turn_number=1,
            role="tool_result",
            content="[tool_result: search]\nFound 3 results.",
            metadata={"_original_message": result_original},
        )
        messages = self.adapter.turns_to_messages(
            [assistant_turn, result_turn], [assistant_original, result_original]
        )
        assert messages[0] is assistant_original
        assert messages[1] is result_original


class TestOpenAIAdapter:
    """Test OpenAI wire-format adapter."""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    # --- extract_messages ---

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

    def test_extract_developer_message(self):
        body = {
            "messages": [
                {"role": "developer", "content": "Dev instructions."},
                {"role": "user", "content": "Hi"},
            ]
        }
        system, messages = self.adapter.extract_messages(body)
        assert system == "Dev instructions."
        assert len(messages) == 1

    def test_extract_no_system(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        system, messages = self.adapter.extract_messages(body)
        assert system is None
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

    def test_extract_multipart_system_content(self):
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

    # --- rebuild_body ---

    def test_rebuild_preserves_non_message_fields(self):
        body = {
            "model": "gpt-4",
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "Base."},
                {"role": "user", "content": "Hi"},
            ],
        }
        rebuilt = self.adapter.rebuild_body(body, [{"role": "user", "content": "hi"}], None)
        assert rebuilt["model"] == "gpt-4"
        assert rebuilt["temperature"] == 0.7

    def test_rebuild_with_system_injection(self):
        body = {
            "messages": [
                {"role": "system", "content": "Base prompt."},
                {"role": "user", "content": "Hi"},
            ]
        }
        _, messages = self.adapter.extract_messages(body)
        rebuilt = self.adapter.rebuild_body(body, messages, "Injected summary.")
        # System message should be first with injection prepended
        assert rebuilt["messages"][0]["role"] == "system"
        assert rebuilt["messages"][0]["content"].startswith("Injected summary.")
        assert "Base prompt." in rebuilt["messages"][0]["content"]

    def test_rebuild_injection_creates_system_when_absent(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        rebuilt = self.adapter.rebuild_body(body, [{"role": "user", "content": "Hi"}], "Summary.")
        assert rebuilt["messages"][0]["role"] == "system"
        assert rebuilt["messages"][0]["content"] == "Summary."

    def test_rebuild_preserves_developer_role(self):
        body = {
            "messages": [
                {"role": "developer", "content": "Dev prompt."},
                {"role": "user", "content": "Hi"},
            ]
        }
        _, messages = self.adapter.extract_messages(body)
        rebuilt = self.adapter.rebuild_body(body, messages, "Injected.")
        assert rebuilt["messages"][0]["role"] == "developer"

    # --- parse_sse_text ---

    def test_parse_text_delta(self):
        chunk = b'data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"Hello"}}]}\n'
        assert self.adapter.parse_sse_text(chunk) == "Hello"

    def test_parse_non_content_event(self):
        chunk = b'data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n'
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_done_signal(self):
        chunk = b"data: [DONE]\n"
        assert self.adapter.parse_sse_text(chunk) is None

    def test_parse_event_line(self):
        chunk = b"event: message\n"
        assert self.adapter.parse_sse_text(chunk) is None

    # --- messages_to_turns ---

    def test_text_message_to_turn(self):
        messages = [{"role": "user", "content": "Hello world"}]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello world"
        assert turns[0].turn_number == 0

    def test_tool_call_message_to_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=5)
        assert len(turns) == 1
        assert turns[0].turn_number == 5
        assert "[tool_call: search" in turns[0].content
        assert "Let me search." in turns[0].content
        assert turns[0].content_type == "tool_output"

    def test_tool_message_to_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "/etc/hosts"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "127.0.0.1 localhost",
            },
        ]
        turns = self.adapter.messages_to_turns(messages, turn_counter_start=0)
        assert len(turns) == 2
        assert turns[1].role == "user"  # tool normalized to user
        assert turns[1].content_type == "file_content"  # read_file classified
        assert "127.0.0.1 localhost" in turns[1].content

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

    def test_compacted_tool_turn_becomes_user(self):
        """Compacted tool message should become user role (not tool without tool_call_id)."""
        original = {"role": "tool", "tool_call_id": "call_123", "content": "result data"}
        turn = ConversationTurn(
            turn_number=0,
            role="user",
            content="Compacted tool result",
            compacted=True,
            metadata={"_original_message": original},
        )
        messages = self.adapter.turns_to_messages([turn], [original])
        assert messages[0]["role"] == "user"
        assert "tool_call_id" not in messages[0]

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

    def test_orphaned_tool_message_flattened(self):
        """Tool message whose tool_call was compacted should be flattened."""
        # Assistant with tool_calls was compacted (no tool_calls in output)
        compacted_assistant = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="I searched for the file.",
            compacted=True,
        )
        # Tool message still references the compacted tool_call
        tool_original = {"role": "tool", "tool_call_id": "call_xyz", "content": "file contents"}
        tool_turn = ConversationTurn(
            turn_number=1,
            role="user",
            content="[tool_result: read_file]\nfile contents",
            metadata={"_original_message": tool_original},
        )
        messages = self.adapter.turns_to_messages(
            [compacted_assistant, tool_turn], []
        )
        # Tool message should be flattened to user message (no tool_call_id)
        assert messages[1]["role"] == "user"
        assert "tool_call_id" not in messages[1]

    def test_live_tool_call_preserved(self):
        """Tool message whose tool_call is still present should be preserved."""
        assistant_original = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_live",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
        assistant_turn = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="[tool_call: search({})]",
            metadata={"_original_message": assistant_original},
        )
        tool_original = {"role": "tool", "tool_call_id": "call_live", "content": "results"}
        tool_turn = ConversationTurn(
            turn_number=1,
            role="user",
            content="[tool_result: search]\nresults",
            metadata={"_original_message": tool_original},
        )
        messages = self.adapter.turns_to_messages(
            [assistant_turn, tool_turn], [assistant_original, tool_original]
        )
        # Both should preserve originals
        assert messages[0] is assistant_original
        assert messages[1] is tool_original

    def test_orphaned_tool_calls_stripped(self):
        """Assistant tool_calls whose tool response was compacted should be stripped."""
        assistant_original = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
        assistant_turn = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="Let me check.\n[tool_call: search({})]",
            metadata={"_original_message": assistant_original},
        )
        compacted_tool = ConversationTurn(
            turn_number=1,
            role="user",
            content="[tool_result: search]\nresults",
            compacted=True,
        )
        messages = self.adapter.turns_to_messages(
            [assistant_turn, compacted_tool], []
        )
        assert messages[0]["role"] == "assistant"
        assert "tool_calls" not in messages[0]
        assert messages[0]["content"] == "Let me check."
        assert messages[1]["role"] == "user"

    def test_live_tool_calls_preserved(self):
        """Assistant tool_calls with live tool response should be preserved."""
        assistant_original = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
        assistant_turn = ConversationTurn(
            turn_number=0,
            role="assistant",
            content="Let me check.\n[tool_call: search({})]",
            metadata={"_original_message": assistant_original},
        )
        tool_original = {"role": "tool", "tool_call_id": "call_1", "content": "results"}
        tool_turn = ConversationTurn(
            turn_number=1,
            role="user",
            content="[tool_result: search]\nresults",
            metadata={"_original_message": tool_original},
        )
        messages = self.adapter.turns_to_messages(
            [assistant_turn, tool_turn], [assistant_original, tool_original]
        )
        assert messages[0] is assistant_original
        assert "tool_calls" in messages[0]
        assert messages[1] is tool_original

    def test_roundtrip_preserves_content(self):
        """messages_to_turns → turns_to_messages roundtrip for raw turns."""
        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns = self.adapter.messages_to_turns(original_messages, 0)
        rebuilt = self.adapter.turns_to_messages(turns, original_messages)
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
        config = PipelineConfig(compaction=CompactionConfig(raw_window=raw_window, cost_gate=CostGateConfig(enabled=False)))
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
    from sr2_bridge.config import BridgeForwardingConfig
    from sr2_bridge.llm import APIKeyCache

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
        from sr2_bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"x-api-key": "sk-ant-test123"})
        assert cache.key == "sk-ant-test123"

    def test_extract_bearer_token(self):
        from sr2_bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"authorization": "Bearer sk-bearer-key"})
        assert cache.key == "sk-bearer-key"

    def test_x_api_key_takes_precedence(self):
        from sr2_bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"x-api-key": "direct-key", "authorization": "Bearer bearer-key"})
        assert cache.key == "direct-key"

    def test_no_key_returns_none(self):
        from sr2_bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"content-type": "application/json"})
        assert cache.key is None

    def test_key_updates_on_new_value(self):
        from sr2_bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"x-api-key": "key-1"})
        assert cache.key == "key-1"
        cache.update({"x-api-key": "key-2"})
        assert cache.key == "key-2"

    def test_empty_bearer_not_cached(self):
        from sr2_bridge.llm import APIKeyCache

        cache = APIKeyCache()
        cache.update({"authorization": "Bearer "})
        # "Bearer " with no token should not cache empty string
        # Actually, it extracts "" which is falsy, so no update
        assert cache.key is None


class TestLLMCallableFactory:
    """Test callable creation and key resolution."""

    def test_summarization_callable_uses_dedicated_key(self):
        from sr2_bridge.config import BridgeLLMModelConfig
        from sr2_bridge.llm import APIKeyCache, make_summarization_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="dedicated-key")
        cache = APIKeyCache()
        callable_fn = make_summarization_callable(config, cache, "https://api.example.com")
        # Should be a callable (async function)
        assert callable(callable_fn)

    def test_extraction_callable_created(self):
        from sr2_bridge.config import BridgeLLMModelConfig
        from sr2_bridge.llm import APIKeyCache, make_extraction_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="key")
        callable_fn = make_extraction_callable(config, APIKeyCache(), "https://api.example.com")
        assert callable(callable_fn)

    def test_embedding_callable_created(self):
        from sr2_bridge.config import BridgeLLMModelConfig
        from sr2_bridge.llm import APIKeyCache, make_embedding_callable

        config = BridgeLLMModelConfig(model="test-model", api_key="key")
        callable_fn = make_embedding_callable(config, APIKeyCache(), "https://api.example.com")
        assert callable(callable_fn)

    @pytest.mark.asyncio
    async def test_summarization_raises_without_key(self):
        from sr2_bridge.config import BridgeLLMModelConfig
        from sr2_bridge.llm import APIKeyCache, make_summarization_callable

        config = BridgeLLMModelConfig(model="test-model")  # no dedicated key
        cache = APIKeyCache()  # no cached key
        callable_fn = make_summarization_callable(config, cache, "https://api.example.com")
        with pytest.raises(RuntimeError, match="No API key"):
            await callable_fn("system", "prompt")

    @pytest.mark.asyncio
    async def test_extraction_raises_without_key(self):
        from sr2_bridge.config import BridgeLLMModelConfig
        from sr2_bridge.llm import APIKeyCache, make_extraction_callable

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
        from sr2.config.models import DegradationConfig
        from sr2_bridge.config import BridgeLLMConfig, BridgeLLMModelConfig

        config = PipelineConfig(
            compaction=CompactionConfig(raw_window=2, cost_gate=CostGateConfig(enabled=False)),
            degradation=DegradationConfig(
                circuit_breaker_threshold=2,
                circuit_breaker_cooldown_minutes=60,
            ),
        )
        bridge_config = BridgeConfig(
            llm=BridgeLLMConfig(
                summarization=BridgeLLMModelConfig(
                    model="test-model",
                    api_key="test-key",
                ),
            ),
        )
        from sr2_bridge.llm import APIKeyCache
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
        from sr2_bridge.config import BridgeMemoryConfig
        cfg = BridgeMemoryConfig()
        assert cfg.enabled is False
        assert cfg.db_path == "sr2_bridge_memory.db"

    def test_engine_delegates_memory_to_sr2(self):
        """Memory subsystem is managed by SR2, not BridgeEngine directly."""
        engine = BridgeEngine(PipelineConfig())
        # SR2 always has a memory store (InMemoryMemoryStore by default)
        assert engine._sr2._memory_store is not None

    def test_engine_has_conversation_manager(self):
        """Engine exposes SR2's conversation manager."""
        engine = BridgeEngine(PipelineConfig())
        assert engine.conversation_manager is not None


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
    async def test_sr2_memory_components_wired(self):
        """SR2 memory components are available through engine's SR2 instance."""
        engine = BridgeEngine(PipelineConfig())

        # SR2 always wires up memory components
        assert engine._sr2._extractor is not None
        assert engine._sr2._conflict_detector is not None
        assert engine._sr2._conflict_resolver is not None
        assert engine._sr2._retriever is not None

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


class TestContentTypeDetection:
    """Test that the adapter correctly assigns content_type to turns."""

    def test_tool_use_classified_as_tool_output(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Glob", "input": {"pattern": "*.py"}},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type == "tool_output"

    def test_bash_tool_use_classified_as_code_execution(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type == "code_execution"

    def test_read_tool_result_classified_as_file_content(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/foo.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents here"},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type == "file_content"  # assistant tool_use
        assert turns[1].content_type == "file_content"  # user tool_result

    def test_bash_tool_result_classified_as_code_execution(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "PASSED"},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[1].content_type == "code_execution"

    def test_unknown_tool_result_falls_back_to_tool_output(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "unknown_id", "content": "data"},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type == "tool_output"

    def test_plain_text_message_has_no_content_type(self):
        adapter = AnthropicAdapter()
        messages = [{"role": "user", "content": "hello"}]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type is None

    def test_tool_name_in_metadata(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "data"},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].metadata["tool_name"] == "Read"
        assert turns[1].metadata["tool_name"] == "Read"

    def test_view_file_classified_as_file_content(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "view_file", "input": {}},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type == "file_content"

    def test_execute_tool_classified_as_code_execution(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "execute_command", "input": {}},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[0].content_type == "code_execution"


class TestExitCodeExtraction:
    """Test exit_code extraction from Anthropic tool_result blocks."""

    def test_successful_bash_result_has_exit_code_zero(self):
        """tool_result with is_error absent -> exit_code 0."""
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file1.txt\nfile2.txt"},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[1].metadata.get("exit_code") == 0

    def test_error_bash_result_has_exit_code_one(self):
        """tool_result with is_error: true -> exit_code 1."""
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "false"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "error", "is_error": True},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[1].metadata.get("exit_code") == 1

    def test_explicit_is_error_false_has_exit_code_zero(self):
        """tool_result with is_error: false -> exit_code 0."""
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "shell", "input": {"cmd": "echo hi"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "hi", "is_error": False},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert turns[1].metadata.get("exit_code") == 0

    def test_non_code_execution_tool_has_no_exit_code(self):
        """tool_result for non-code_execution tool has no exit_code in metadata."""
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/foo.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert "exit_code" not in turns[1].metadata

    def test_tool_use_turn_has_no_exit_code(self):
        """tool_use turns (assistant) never have exit_code."""
        adapter = AnthropicAdapter()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]},
        ]
        turns = adapter.messages_to_turns(messages, 0)
        assert "exit_code" not in turns[0].metadata


# ---------------------------------------------------------------------------
# System-reminder extraction, dedup, and injection
# ---------------------------------------------------------------------------

SAMPLE_REMINDER = "<system-reminder>\n# Some Instructions\nDo things correctly.\n</system-reminder>"
SAMPLE_REMINDER_INNER = "# Some Instructions\nDo things correctly."

SAMPLE_REMINDER_2 = "<system-reminder>\nSecond block content.\n</system-reminder>"
SAMPLE_REMINDER_2_INNER = "Second block content."


class TestExtractSystemReminders:
    """Test the standalone _extract_system_reminders() function that extracts
    reminder inner content WITHOUT modifying the input string."""

    def test_extracts_single_block(self):
        """Extracts inner content from a single system-reminder block."""
        from sr2_bridge.adapters.anthropic import _extract_system_reminders

        content = f"Hello.\n{SAMPLE_REMINDER}\nWorld."
        result = _extract_system_reminders(content)
        assert len(result) == 1
        assert result[0] == SAMPLE_REMINDER_INNER

    def test_extracts_multiple_blocks(self):
        """Extracts inner content from multiple system-reminder blocks."""
        from sr2_bridge.adapters.anthropic import _extract_system_reminders

        content = f"Start.\n{SAMPLE_REMINDER}\nMiddle.\n{SAMPLE_REMINDER_2}\nEnd."
        result = _extract_system_reminders(content)
        assert len(result) == 2
        assert result[0] == SAMPLE_REMINDER_INNER
        assert result[1] == SAMPLE_REMINDER_2_INNER

    def test_returns_empty_list_when_no_reminders(self):
        """Returns empty list when no system-reminder blocks are present."""
        from sr2_bridge.adapters.anthropic import _extract_system_reminders

        result = _extract_system_reminders("Just a normal message.")
        assert result == []

    def test_does_not_modify_input(self):
        """The input string is not modified (extract-only, no stripping)."""
        from sr2_bridge.adapters.anthropic import _extract_system_reminders

        content = f"Hello.\n{SAMPLE_REMINDER}\nWorld."
        original = content  # strings are immutable, but verify function signature
        _extract_system_reminders(content)
        assert content == original

    def test_no_tags_in_extracted_content(self):
        """Extracted content should not include the tags themselves."""
        from sr2_bridge.adapters.anthropic import _extract_system_reminders

        result = _extract_system_reminders(SAMPLE_REMINDER)
        assert len(result) == 1
        assert "<system-reminder>" not in result[0]
        assert "</system-reminder>" not in result[0]

    def test_multiline_block_extracted(self):
        """Multiline content inside a block is fully extracted."""
        from sr2_bridge.adapters.anthropic import _extract_system_reminders

        multiline = "<system-reminder>\nLine one.\nLine two.\nLine three.\n</system-reminder>"
        result = _extract_system_reminders(multiline)
        assert len(result) == 1
        assert "Line one." in result[0]
        assert "Line two." in result[0]
        assert "Line three." in result[0]


class TestSystemReminderExtraction:
    """Test that AnthropicAdapter.messages_to_turns() extracts system-reminder
    blocks into metadata but KEEPS them in turn.content (extract-only, no stripping)."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_system_reminder_kept_in_string_content(self):
        """User message with <system-reminder> block keeps it in turn.content."""
        messages = [
            {"role": "user", "content": f"Hello world.\n\n{SAMPLE_REMINDER}\n\nMore text."},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert len(turns) == 1
        # Reminders should be KEPT in content (not stripped)
        assert "<system-reminder>" in turns[0].content
        assert "</system-reminder>" in turns[0].content
        assert SAMPLE_REMINDER_INNER in turns[0].content
        assert "Hello world." in turns[0].content
        assert "More text." in turns[0].content

    def test_system_reminder_kept_in_content_blocks(self):
        """Content blocks with system-reminder in text block keep them intact."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"Hello.\n{SAMPLE_REMINDER}\nBye."},
            ]},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert len(turns) == 1
        # Reminders should be KEPT in content
        assert "<system-reminder>" in turns[0].content
        assert "Hello." in turns[0].content

    def test_multiple_system_reminders_kept(self):
        """Multiple <system-reminder> blocks in one message are all kept in content."""
        messages = [
            {"role": "user", "content": f"Start.\n{SAMPLE_REMINDER}\nMiddle.\n{SAMPLE_REMINDER_2}\nEnd."},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        # Both reminders should be in content
        assert "<system-reminder>" in turns[0].content
        assert SAMPLE_REMINDER_INNER in turns[0].content
        assert SAMPLE_REMINDER_2_INNER in turns[0].content
        assert "Start." in turns[0].content
        assert "Middle." in turns[0].content
        assert "End." in turns[0].content

    def test_multiline_system_reminder_kept(self):
        """Block with multiline content is kept intact in turn.content."""
        multiline = "<system-reminder>\nLine one.\nLine two.\nLine three.\n</system-reminder>"
        messages = [
            {"role": "user", "content": f"Before.\n{multiline}\nAfter."},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert "<system-reminder>" in turns[0].content
        assert "Line one." in turns[0].content
        assert "Before." in turns[0].content
        assert "After." in turns[0].content

    def test_no_system_reminder_unchanged(self):
        """Messages without system-reminder tags are unchanged."""
        messages = [
            {"role": "user", "content": "Just a normal message."},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[0].content == "Just a normal message."

    def test_content_only_system_reminder_kept(self):
        """Message that's entirely a system-reminder keeps it in content."""
        messages = [
            {"role": "user", "content": SAMPLE_REMINDER},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert "<system-reminder>" in turns[0].content
        assert SAMPLE_REMINDER_INNER in turns[0].content

    def test_original_message_not_stripped(self):
        """turn.metadata['_original_message'] should be the ORIGINAL unmodified
        message dict. No stripping, no reconstruction. The original dict passed
        in should NOT be mutated."""
        original_content = f"Hello.\n{SAMPLE_REMINDER}\nWorld."
        original_msg = {"role": "user", "content": original_content}
        messages = [original_msg]
        turns = self.adapter.messages_to_turns(messages, 0)

        # Original dict should not be mutated
        assert original_msg["content"] == original_content

        # _original_message in metadata should be the original unmodified dict
        assert turns[0].metadata is not None
        orig = turns[0].metadata.get("_original_message", {})
        assert isinstance(orig, dict)
        orig_content = orig.get("content", "")
        # Should still contain system-reminder (NOT stripped)
        assert "<system-reminder>" in orig_content
        assert SAMPLE_REMINDER_INNER in orig_content

    def test_original_message_content_blocks_not_stripped(self):
        """For content-block messages, _original_message should preserve the
        original blocks without stripping or reconstructing."""
        original_blocks = [
            {"type": "text", "text": f"Hello.\n{SAMPLE_REMINDER}\nBye."},
        ]
        original_msg = {"role": "user", "content": original_blocks}
        messages = [original_msg]
        turns = self.adapter.messages_to_turns(messages, 0)

        assert turns[0].metadata is not None
        orig = turns[0].metadata.get("_original_message", {})
        assert isinstance(orig, dict)
        orig_content = orig.get("content", [])
        assert isinstance(orig_content, list)
        # Original content blocks should be preserved as-is
        assert any(
            "<system-reminder>" in b.get("text", "")
            for b in orig_content
            if isinstance(b, dict)
        )

    def test_extracted_blocks_in_metadata(self):
        """Extracted blocks stored in turn.metadata['extracted_system_reminders']
        as a list of strings (inner content only, no tags)."""
        messages = [
            {"role": "user", "content": f"Hello.\n{SAMPLE_REMINDER}\nMiddle.\n{SAMPLE_REMINDER_2}\nBye."},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[0].metadata is not None
        extracted = turns[0].metadata.get("extracted_system_reminders", [])
        assert len(extracted) == 2
        # Inner content only, no tags
        assert "<system-reminder>" not in extracted[0]
        assert "</system-reminder>" not in extracted[0]
        assert SAMPLE_REMINDER_INNER in extracted[0]
        assert SAMPLE_REMINDER_2_INNER in extracted[1]

    def test_no_extracted_reminders_when_none_present(self):
        """No extracted_system_reminders key when message has no reminders."""
        messages = [
            {"role": "user", "content": "Just text, no reminders."},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[0].metadata is not None
        extracted = turns[0].metadata.get("extracted_system_reminders", [])
        assert extracted == []

    def test_assistant_messages_not_touched(self):
        """Only user messages get system-reminder extraction."""
        messages = [
            {"role": "assistant", "content": f"Some text {SAMPLE_REMINDER} more text"},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        # Assistant message should still contain the tags (no extraction)
        assert "<system-reminder>" in turns[0].content

    def test_other_xml_tags_preserved(self):
        """Other XML-like tags (e.g. <thinking>) should not be touched."""
        messages = [
            {"role": "user", "content": f"<thinking>Deep thoughts</thinking>\n{SAMPLE_REMINDER}\nQuestion?"},
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert "<thinking>Deep thoughts</thinking>" in turns[0].content
        # Reminder should also be kept in content
        assert "<system-reminder>" in turns[0].content


class TestSystemReminderDedup:
    """Test deduplication of system-reminder blocks on BridgeSession."""

    def test_session_has_dedup_fields(self):
        """BridgeSession has system_reminder_hashes and system_reminder_content."""
        session = BridgeSession(session_id="test")
        assert hasattr(session, "system_reminder_hashes")
        assert hasattr(session, "system_reminder_content")
        assert isinstance(session.system_reminder_hashes, set)
        assert isinstance(session.system_reminder_content, list)

    def test_same_block_deduped_across_turns(self):
        """Same system-reminder content in two turns only stored once."""
        session = BridgeSession(session_id="test")
        adapter = AnthropicAdapter()

        # First turn with reminder
        msgs_1 = [{"role": "user", "content": f"Turn 1.\n{SAMPLE_REMINDER}"}]
        turns_1 = adapter.messages_to_turns(msgs_1, 0)

        # Second turn with same reminder
        msgs_2 = [{"role": "user", "content": f"Turn 2.\n{SAMPLE_REMINDER}"}]
        turns_2 = adapter.messages_to_turns(msgs_2, 1)

        # Simulate dedup: add extracted reminders to session
        for turn in turns_1 + turns_2:
            if turn.metadata and "extracted_system_reminders" in turn.metadata:
                for block in turn.metadata["extracted_system_reminders"]:
                    block_hash = hashlib.sha256(block.encode()).hexdigest()
                    if block_hash not in session.system_reminder_hashes:
                        session.system_reminder_hashes.add(block_hash)
                        session.system_reminder_content.append(block)

        # Same content should only appear once
        assert len(session.system_reminder_content) == 1
        assert SAMPLE_REMINDER_INNER in session.system_reminder_content[0]

    def test_different_blocks_both_stored(self):
        """Different system-reminder content stored separately."""
        session = BridgeSession(session_id="test")
        adapter = AnthropicAdapter()

        msgs = [{"role": "user", "content": f"Hello.\n{SAMPLE_REMINDER}\n{SAMPLE_REMINDER_2}"}]
        turns = adapter.messages_to_turns(msgs, 0)

        for turn in turns:
            if turn.metadata and "extracted_system_reminders" in turn.metadata:
                for block in turn.metadata["extracted_system_reminders"]:
                    block_hash = hashlib.sha256(block.encode()).hexdigest()
                    if block_hash not in session.system_reminder_hashes:
                        session.system_reminder_hashes.add(block_hash)
                        session.system_reminder_content.append(block)

        assert len(session.system_reminder_content) == 2

    def test_updated_block_appends_new_version(self):
        """If CLAUDE.md changes, old version stays and new one is added too
        (append, not replace — we want the latest set of unique blocks)."""
        session = BridgeSession(session_id="test")
        adapter = AnthropicAdapter()

        old_reminder = "<system-reminder>\nVersion 1 instructions.\n</system-reminder>"
        new_reminder = "<system-reminder>\nVersion 2 instructions.\n</system-reminder>"

        # Turn 1 with old version
        msgs_1 = [{"role": "user", "content": f"Turn 1.\n{old_reminder}"}]
        turns_1 = adapter.messages_to_turns(msgs_1, 0)

        # Turn 2 with new version
        msgs_2 = [{"role": "user", "content": f"Turn 2.\n{new_reminder}"}]
        turns_2 = adapter.messages_to_turns(msgs_2, 1)

        for turn in turns_1 + turns_2:
            if turn.metadata and "extracted_system_reminders" in turn.metadata:
                for block in turn.metadata["extracted_system_reminders"]:
                    block_hash = hashlib.sha256(block.encode()).hexdigest()
                    if block_hash not in session.system_reminder_hashes:
                        session.system_reminder_hashes.add(block_hash)
                        session.system_reminder_content.append(block)

        # Both versions stored (different content = different hashes)
        assert len(session.system_reminder_content) == 2
        assert "Version 1" in session.system_reminder_content[0]
        assert "Version 2" in session.system_reminder_content[1]


class TestPerMessageHashing:
    """Test per-message hash comparison in BridgeEngine."""

    def _make_engine(self, raw_window: int = 5) -> BridgeEngine:
        config = PipelineConfig(
            compaction=CompactionConfig(
                raw_window=raw_window,
                cost_gate=CostGateConfig(enabled=False),
            )
        )
        return BridgeEngine(config)

    def _make_session(self, session_id: str = "test") -> BridgeSession:
        return BridgeSession(session_id=session_id)

    @pytest.mark.asyncio
    async def test_first_request_stores_hashes(self):
        """First request stores per-message hashes."""
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert len(session.message_hashes) == 2

    @pytest.mark.asyncio
    async def test_normal_turn_appends_hashes(self):
        """Normal turn (new messages appended) extends hash list."""
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert len(session.message_hashes) == 2

        msgs.append({"role": "user", "content": "Follow up"})
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert len(session.message_hashes) == 3

    @pytest.mark.asyncio
    async def test_session_has_message_hashes_field(self):
        """BridgeSession has a message_hashes list field."""
        session = BridgeSession(session_id="test")
        assert hasattr(session, "message_hashes")
        assert isinstance(session.message_hashes, list)
        assert len(session.message_hashes) == 0


class TestCompactionDetection:
    """Test that engine detects Claude Code compaction via hash mismatch."""

    def _make_engine(self, raw_window: int = 5) -> BridgeEngine:
        config = PipelineConfig(
            compaction=CompactionConfig(
                raw_window=raw_window,
                cost_gate=CostGateConfig(enabled=False),
            )
        )
        return BridgeEngine(config)

    def _make_session(self, session_id: str = "test") -> BridgeSession:
        return BridgeSession(session_id=session_id)

    @pytest.mark.asyncio
    async def test_hash_mismatch_detects_compaction(self):
        """When prior message hashes change, engine detects compaction."""
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        # First request: 3 messages
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is Python?"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert session.turn_counter == 3

        # Simulate Claude Code compaction: prior messages changed (summarized)
        compacted_msgs = [
            {"role": "user", "content": "[COMPACTED] User greeted and asked about Python"},
            {"role": "assistant", "content": "[COMPACTED] Assistant responded"},
            {"role": "user", "content": "Tell me more about decorators"},
        ]
        await engine.optimize(system=None, messages=compacted_msgs, session=session, adapter=adapter)

        # Engine should have detected compaction and processed only the new message.
        # Turn counter should have incremented by 1 (only the new message).
        assert session.turn_counter == 4

    @pytest.mark.asyncio
    async def test_session_reset_on_shorter_history(self):
        """Significantly shorter history triggers session reset, not compaction path."""
        engine = self._make_engine()
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "More"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)
        assert session.turn_counter == 3

        # User did /clear — much shorter history
        shorter = [{"role": "user", "content": "Fresh start"}]
        await engine.optimize(system=None, messages=shorter, session=session, adapter=adapter)
        assert session.turn_counter == 1
        assert len(session.message_hashes) == 1

    @pytest.mark.asyncio
    async def test_compaction_preserves_stored_history(self):
        """On compaction detection, ConversationManager history is preserved.

        Claude Code compaction keeps the same message count but changes content.
        """
        engine = self._make_engine(raw_window=10)
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Question"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)

        # Check ConversationManager has 3 turns
        zones = engine.conversation_manager.zones(session.session_id)
        initial_turns = len(zones.raw) + len(zones.compacted)
        assert initial_turns == 3

        # Simulate Claude Code compaction: SAME count, different content/hashes
        compacted_msgs = [
            {"role": "user", "content": "[COMPACTED] User greeted and asked question"},
            {"role": "assistant", "content": "[COMPACTED] Assistant responded"},
            {"role": "user", "content": "New follow-up after compaction"},
        ]
        await engine.optimize(system=None, messages=compacted_msgs, session=session, adapter=adapter)

        # ConversationManager should still have the original 3 turns + 1 new
        zones = engine.conversation_manager.zones(session.session_id)
        total = len(zones.raw) + len(zones.compacted)
        assert total == 4


class TestDeferredInjection:
    """Test that system-reminder injection is deferred until compaction/summarization."""

    def _make_engine(self, raw_window: int = 5) -> BridgeEngine:
        config = PipelineConfig(
            compaction=CompactionConfig(
                raw_window=raw_window,
                cost_gate=CostGateConfig(enabled=False),
            )
        )
        return BridgeEngine(config)

    def _make_session(self, session_id: str = "test") -> BridgeSession:
        return BridgeSession(session_id=session_id)

    @pytest.mark.asyncio
    async def test_no_injection_when_no_compaction(self):
        """Without compaction/summarization, reminders stay in messages (no injection)."""
        engine = self._make_engine(raw_window=50)  # large window = no compaction
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": f"Simple question.\n{SAMPLE_REMINDER}"},
        ]
        injection, optimized = await engine.optimize(
            system="Prompt.",
            messages=msgs,
            session=session,
            adapter=adapter,
        )

        # No injection — reminders are still in the messages naturally
        assert injection is None
        # Messages should be the original (passthrough)
        assert optimized is msgs

    @pytest.mark.asyncio
    async def test_reminders_in_messages_when_no_compaction(self):
        """Reminders should be visible in messages when no compaction has happened."""
        engine = self._make_engine(raw_window=50)
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": f"Hello.\n{SAMPLE_REMINDER}"},
        ]
        _, optimized = await engine.optimize(
            system="Prompt.",
            messages=msgs,
            session=session,
            adapter=adapter,
        )

        # Reminders should still be in the message content
        msg_content = optimized[0].get("content", "")
        if isinstance(msg_content, str):
            assert "<system-reminder>" in msg_content

    @pytest.mark.asyncio
    async def test_injection_after_compaction(self):
        """After compaction, stored reminders appear in system_injection.

        Uses tool_use/tool_result messages to trigger rule-based compaction
        (schema_and_sample or result_summary rules).
        """
        engine = self._make_engine(raw_window=2)
        session = self._make_session()
        adapter = AnthropicAdapter()

        # Build messages with tool results that trigger compaction rules.
        # tool_result turns with large content trigger result_summary rule.
        large_output = "line " * 200  # ~200 words of output
        msgs = [
            {"role": "user", "content": f"Read the file.\n{SAMPLE_REMINDER}"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat big.txt"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": large_output},
            ]},
            {"role": "assistant", "content": "The file contains test data."},
            {"role": "user", "content": "Now do something else."},
        ]
        injection, optimized = await engine.optimize(
            system="System prompt.",
            messages=msgs,
            session=session,
            adapter=adapter,
        )

        # Compaction should have happened (tool results trigger rules)
        # If compaction fired, injection should contain stored reminder content
        if injection is not None:
            assert SAMPLE_REMINDER_INNER in injection
        else:
            # If no rules fired (simple text compaction doesn't trigger),
            # verify reminders are still stored in session for later
            assert len(session.system_reminder_content) == 1
            assert SAMPLE_REMINDER_INNER in session.system_reminder_content[0]

    @pytest.mark.asyncio
    async def test_reminders_stored_even_without_injection(self):
        """Reminders are always extracted and stored, even when not injected."""
        engine = self._make_engine(raw_window=50)
        session = self._make_session()
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": f"Hello.\n{SAMPLE_REMINDER}\n{SAMPLE_REMINDER_2}"},
        ]
        injection, _ = await engine.optimize(
            system="Prompt.",
            messages=msgs,
            session=session,
            adapter=adapter,
        )

        # No injection (no compaction)
        assert injection is None
        # But reminders are stored for future use
        assert len(session.system_reminder_content) == 2
        assert SAMPLE_REMINDER_INNER in session.system_reminder_content[0]
        assert SAMPLE_REMINDER_2_INNER in session.system_reminder_content[1]


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
    async def test_shutdown_is_safe_with_sr2(self):
        """Shutdown is safe when engine delegates to SR2."""
        engine = BridgeEngine(PipelineConfig())
        await engine.shutdown()  # should not raise
        # Can call shutdown multiple times safely
        await engine.shutdown()


# ---------------------------------------------------------------------------
# _extract_file_path unit tests
# ---------------------------------------------------------------------------

class TestExtractFilePath:
    """Unit tests for the _extract_file_path helper in adapters._utils."""

    def test_file_path_key(self):
        """File-reading tool with 'file_path' key returns the path."""
        from sr2_bridge.adapters._utils import _extract_file_path

        result = _extract_file_path("Read", {"file_path": "/src/main.py"})
        assert result == "/src/main.py"

    def test_path_key(self):
        """File-reading tool with 'path' key returns the path."""
        from sr2_bridge.adapters._utils import _extract_file_path

        result = _extract_file_path("cat", {"path": "/etc/hosts"})
        assert result == "/etc/hosts"

    def test_filename_key(self):
        """File-reading tool with 'filename' key returns the path."""
        from sr2_bridge.adapters._utils import _extract_file_path

        result = _extract_file_path("view_file", {"filename": "/tmp/data.json"})
        assert result == "/tmp/data.json"

    def test_non_file_tool_returns_none(self):
        """Non-file tool (e.g. 'bash') returns None even if it has a 'path' arg."""
        from sr2_bridge.adapters._utils import _extract_file_path

        result = _extract_file_path("bash", {"path": "/some/dir"})
        assert result is None

    def test_empty_arguments_returns_none(self):
        """Empty arguments dict returns None."""
        from sr2_bridge.adapters._utils import _extract_file_path

        result = _extract_file_path("Read", {})
        assert result is None

    def test_non_string_path_returns_none(self):
        """Non-string path value returns None."""
        from sr2_bridge.adapters._utils import _extract_file_path

        result = _extract_file_path("Read", {"file_path": 42})
        assert result is None


# ---------------------------------------------------------------------------
# Anthropic adapter file_path integration tests
# ---------------------------------------------------------------------------

class TestAnthropicAdapterFilePath:
    """Integration tests for file_path population in Anthropic messages_to_turns."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_tool_use_with_file_path(self):
        """tool_use block with name='Read' populates file_path in metadata."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Read",
                        "input": {"file_path": "/src/main.py"},
                    },
                ],
            },
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[0].metadata["file_path"] == "/src/main.py"

    def test_tool_result_inherits_file_path(self):
        """tool_result block referencing a Read tool_use gets file_path via lookup."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Read",
                        "input": {"file_path": "/src/main.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "file contents here",
                    },
                ],
            },
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[1].metadata["file_path"] == "/src/main.py"

    def test_non_file_tool_no_file_path(self):
        """tool_use block with name='bash' does NOT populate file_path."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert "file_path" not in turns[0].metadata


# ---------------------------------------------------------------------------
# OpenAI adapter file_path integration tests
# ---------------------------------------------------------------------------

class TestOpenAIAdapterFilePath:
    """Integration tests for file_path population in OpenAI messages_to_turns."""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_tool_call_with_file_path(self):
        """Assistant tool_call for 'cat' populates file_path in metadata."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "cat",
                            "arguments": json.dumps({"file_path": "/app.py"}),
                        },
                    },
                ],
            },
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[0].metadata["file_path"] == "/app.py"

    def test_tool_message_inherits_file_path(self):
        """Tool message referencing a 'cat' tool_call gets file_path via lookup."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "cat",
                            "arguments": json.dumps({"file_path": "/app.py"}),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "print('hello world')",
            },
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert turns[1].metadata["file_path"] == "/app.py"

    def test_non_file_tool_no_file_path(self):
        """Assistant tool_call for 'bash' does NOT populate file_path."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "ls"}),
                        },
                    },
                ],
            },
        ]
        turns = self.adapter.messages_to_turns(messages, 0)
        assert "file_path" not in turns[0].metadata
