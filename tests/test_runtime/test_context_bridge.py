"""Tests for context-to-messages bridge."""

import json

import pytest

from sr2.pipeline.engine import CompiledContext
from sr2.resolvers.registry import ResolvedContent
from runtime.llm import ContextBridge


def _make_compiled(layers: dict[str, list[tuple[str, str]]]) -> CompiledContext:
    """Build a CompiledContext with named layers.

    Args:
        layers: dict of layer_name -> list of (key, content) tuples
    """
    resolved_layers = {}
    for name, items in layers.items():
        resolved_layers[name] = [
            ResolvedContent(key=k, content=c, tokens=len(c.split()))
            for k, c in items
        ]
    total_tokens = sum(r.tokens for items in resolved_layers.values() for r in items)
    return CompiledContext(
        content="",
        tokens=total_tokens,
        layers=resolved_layers,
    )


class TestContextBridge:
    """Tests for the ContextBridge."""

    def setup_method(self):
        self.bridge = ContextBridge()

    def test_core_layer_in_system_message(self):
        compiled = _make_compiled({"core": [("prompt", "You are a helpful assistant.")]})
        messages = self.bridge.build_messages(compiled, [])
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    def test_memory_in_system_message_after_core(self):
        compiled = _make_compiled({
            "core": [("prompt", "System prompt.")],
            "memory": [("mem1", "User likes coffee.")],
        })
        messages = self.bridge.build_messages(compiled, [])
        assert messages[0]["role"] == "system"
        assert "System prompt." in messages[0]["content"]
        assert "User likes coffee." in messages[0]["content"]
        # Core comes before memory
        assert messages[0]["content"].index("System prompt.") < messages[0]["content"].index(
            "User likes coffee."
        )

    def test_session_turns_mapped_to_correct_roles(self):
        compiled = _make_compiled({"core": [("p", "Prompt")]})
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        messages = self.bridge.build_messages(compiled, turns)
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there"

    def test_tool_result_mapped_to_tool_role(self):
        compiled = _make_compiled({"core": [("p", "Prompt")]})
        turns = [
            {
                "role": "tool_result",
                "content": "search result here",
                "tool_call_id": "call_123",
                "metadata": {"tool_name": "search"},
            },
        ]
        messages = self.bridge.build_messages(compiled, turns)
        tool_msg = messages[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "search result here"
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["name"] == "search"

    def test_current_input_added_if_not_last(self):
        compiled = _make_compiled({"core": [("p", "Prompt")]})
        messages = self.bridge.build_messages(compiled, [], current_input="New message")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "New message"

    def test_current_input_not_duplicated(self):
        compiled = _make_compiled({"core": [("p", "Prompt")]})
        turns = [{"role": "user", "content": "Hello"}]
        messages = self.bridge.build_messages(compiled, turns, current_input="Hello")
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1

    def test_append_tool_result(self):
        messages = [{"role": "user", "content": "test"}]
        result = self.bridge.append_tool_result(
            messages, tool_call_id="tc_1", tool_name="search", result="found it"
        )
        assert result[-1]["role"] == "tool"
        assert result[-1]["tool_call_id"] == "tc_1"
        assert result[-1]["name"] == "search"
        assert result[-1]["content"] == "found it"

    def test_append_assistant_tool_calls_formats_for_api(self):
        messages = [{"role": "user", "content": "test"}]
        tool_calls = [
            {"id": "tc_1", "name": "search", "arguments": {"query": "test"}},
        ]
        result = self.bridge.append_assistant_tool_calls(messages, "Searching...", tool_calls)
        assistant_msg = result[-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Searching..."
        assert assistant_msg["tool_calls"][0]["type"] == "function"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "search"
        assert '"query": "test"' in assistant_msg["tool_calls"][0]["function"]["arguments"]

    def test_empty_compiled_produces_valid_messages(self):
        compiled = _make_compiled({})
        messages = self.bridge.build_messages(compiled, [])
        assert isinstance(messages, list)
        assert len(messages) == 0  # No system message if no layers

    def test_session_tool_calls_with_python_str_arguments(self):
        """Reproduce litellm ollama_pt crash: arguments stored with str() not json.dumps().

        Session.add_tool_call used str(arguments) which produces Python repr
        with single quotes like "{'key': 'val'}". LiteLLM's ollama provider
        calls json.loads() on arguments and crashes with JSONDecodeError.
        """
        compiled = _make_compiled({"core": [("p", "Prompt")]})
        turns = [
            {"role": "user", "content": "Show me the README"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "get_file_contents",
                        # This is what str(dict) produces — single quotes, NOT valid JSON
                        "arguments": "{'path': 'README.md', 'repo': 'Diego/edi'}",
                    },
                }],
            },
            {
                "role": "tool_result",
                "content": "# README\nHello world",
                "tool_call_id": "tc_1",
                "metadata": {"tool_name": "get_file_contents"},
            },
            {"role": "assistant", "content": "Here is the README."},
        ]
        messages = self.bridge.build_messages(compiled, turns)

        # Find the assistant message with tool calls
        tc_msg = [m for m in messages if m.get("tool_calls")][0]
        args_str = tc_msg["tool_calls"][0]["function"]["arguments"]

        # Must be valid JSON (not Python repr)
        parsed = json.loads(args_str)  # This would crash before the fix
        assert parsed == {"path": "README.md", "repo": "Diego/edi"}

    def test_sanitize_tool_calls_with_dict_arguments(self):
        """Arguments stored as raw dict (not string) should be json-serialized."""
        tool_calls = [{
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": {"query": "test"},  # dict, not string
            },
        }]
        sanitized = ContextBridge._sanitize_tool_calls(tool_calls)
        args = sanitized[0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"query": "test"}

    def test_sanitize_tool_calls_valid_json_unchanged(self):
        """Valid JSON string arguments should pass through unchanged."""
        tool_calls = [{
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}',
            },
        }]
        sanitized = ContextBridge._sanitize_tool_calls(tool_calls)
        assert sanitized[0]["function"]["arguments"] == '{"query": "test"}'
