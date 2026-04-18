"""Tests for context-to-messages bridge."""

import json


from sr2.compaction.engine import ConversationTurn
from sr2.pipeline.conversation import ConversationZones
from sr2.pipeline.engine import CompiledContext
from sr2.resolvers.registry import ResolvedContent
from sr2_runtime.llm import ContextBridge


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


class TestBuildMessagesFromZones:
    """Tests for ContextBridge.build_messages_from_zones()."""

    def setup_method(self):
        self.bridge = ContextBridge()

    @staticmethod
    def _make_turn(
        turn_number: int,
        role: str,
        content: str,
        content_type: str | None = None,
        metadata: dict | None = None,
        compacted: bool = False,
    ) -> ConversationTurn:
        return ConversationTurn(
            turn_number=turn_number,
            role=role,
            content=content,
            content_type=content_type,
            metadata=metadata,
            compacted=compacted,
        )

    @staticmethod
    def _make_zones(
        summarized: list[str] | None = None,
        compacted: list[ConversationTurn] | None = None,
        raw: list[ConversationTurn] | None = None,
        session_notes: list[str] | None = None,
    ) -> ConversationZones:
        return ConversationZones(
            summarized=summarized or [],
            compacted=compacted or [],
            raw=raw or [],
            session_notes=session_notes or [],
        )

    # --- Requirement 1: System message from compiled layers ---

    def test_compiled_layers_become_system_message(self):
        compiled = _make_compiled({
            "core": [("prompt", "You are helpful.")],
            "memory": [("mem", "Likes tea.")],
        })
        zones = self._make_zones()
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        assert messages[0]["role"] == "system"
        assert "You are helpful." in messages[0]["content"]
        assert "Likes tea." in messages[0]["content"]

    def test_empty_compiled_no_zones_returns_no_messages(self):
        compiled = _make_compiled({})
        zones = self._make_zones()
        messages = self.bridge.build_messages_from_zones(compiled, zones)
        # No layers, no turns → no messages at all
        assert messages == []

    # --- Requirement 2: Skip session layer ---

    def test_skip_session_layer_excludes_from_system(self):
        compiled = _make_compiled({
            "core": [("prompt", "System prompt.")],
            "session": [("sess", "Session content that should be skipped.")],
        })
        zones = self._make_zones()
        messages = self.bridge.build_messages_from_zones(
            compiled, zones, skip_session_layer="session"
        )

        system_content = messages[0]["content"]
        assert "System prompt." in system_content
        assert "Session content that should be skipped." not in system_content

    def test_skip_session_layer_none_includes_all(self):
        compiled = _make_compiled({
            "core": [("prompt", "Core.")],
            "session": [("sess", "Session.")],
        })
        zones = self._make_zones()
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        system_content = messages[0]["content"]
        assert "Core." in system_content
        assert "Session." in system_content

    # --- Requirement 3: Summaries in system message ---

    def test_summaries_appended_to_system_message(self):
        compiled = _make_compiled({"core": [("prompt", "Base prompt.")]})
        zones = self._make_zones(
            summarized=["User discussed project architecture."],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        system_content = messages[0]["content"]
        assert "[Previous conversation summary]" in system_content
        assert "User discussed project architecture." in system_content

    def test_multiple_summaries_all_included(self):
        compiled = _make_compiled({"core": [("prompt", "Prompt.")]})
        zones = self._make_zones(
            summarized=["Summary one.", "Summary two."],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        system_content = messages[0]["content"]
        assert "Summary one." in system_content
        assert "Summary two." in system_content

    # --- Requirement 4: Session notes in system message ---

    def test_session_notes_appended_to_system_message(self):
        compiled = _make_compiled({"core": [("prompt", "Prompt.")]})
        zones = self._make_zones(
            session_notes=["Important: user prefers verbose output."],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        system_content = messages[0]["content"]
        assert "Important: user prefers verbose output." in system_content

    def test_summaries_and_notes_both_in_system(self):
        compiled = _make_compiled({"core": [("prompt", "Prompt.")]})
        zones = self._make_zones(
            summarized=["Earlier discussion about API design."],
            session_notes=["Note: use REST not GraphQL."],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        system_content = messages[0]["content"]
        assert "Earlier discussion about API design." in system_content
        assert "Note: use REST not GraphQL." in system_content

    # --- Requirement 5: Turns as individual messages ---

    def test_raw_turns_become_messages(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            raw=[
                self._make_turn(1, "user", "Hello"),
                self._make_turn(2, "assistant", "Hi there"),
            ],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there"

    def test_compacted_turns_before_raw(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            compacted=[
                self._make_turn(1, "user", "Old question", compacted=True),
                self._make_turn(2, "assistant", "Old answer", compacted=True),
            ],
            raw=[
                self._make_turn(3, "user", "New question"),
                self._make_turn(4, "assistant", "New answer"),
            ],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        # System + 4 turns
        turn_messages = [m for m in messages if m["role"] != "system"]
        assert len(turn_messages) == 4
        assert turn_messages[0]["content"] == "Old question"
        assert turn_messages[1]["content"] == "Old answer"
        assert turn_messages[2]["content"] == "New question"
        assert turn_messages[3]["content"] == "New answer"

    def test_tool_result_turn_mapped_correctly(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            raw=[
                self._make_turn(
                    1,
                    "tool_result",
                    "search results here",
                    metadata={"tool_call_id": "call_abc", "tool_name": "search"},
                ),
            ],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        tool_msg = messages[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "search results here"
        assert tool_msg["tool_call_id"] == "call_abc"
        assert tool_msg["name"] == "search"

    def test_tool_role_turn_mapped_correctly(self):
        """Turns with role='tool' should also map properly."""
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            raw=[
                self._make_turn(
                    1,
                    "tool",
                    "tool output",
                    metadata={"tool_call_id": "call_xyz", "tool_name": "read_file"},
                ),
            ],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        tool_msg = messages[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "tool output"
        assert tool_msg["tool_call_id"] == "call_xyz"
        assert tool_msg["name"] == "read_file"

    def test_assistant_with_tool_calls_in_metadata(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        tool_calls = [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }
        ]
        zones = self._make_zones(
            raw=[
                self._make_turn(
                    1,
                    "assistant",
                    "Let me search.",
                    metadata={"tool_calls": tool_calls},
                ),
            ],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["tool_calls"] == tool_calls

    def test_assistant_without_tool_calls_is_plain(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            raw=[self._make_turn(1, "assistant", "Just text.")],
        )
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Just text."
        assert "tool_calls" not in messages[1]

    # --- Requirement 6: Current input ---

    def test_current_input_appended_when_not_last(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            raw=[
                self._make_turn(1, "user", "First message"),
                self._make_turn(2, "assistant", "Response"),
            ],
        )
        messages = self.bridge.build_messages_from_zones(
            compiled, zones, current_input="Follow-up question"
        )

        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Follow-up question"

    def test_current_input_not_duplicated_when_already_last(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones(
            raw=[self._make_turn(1, "user", "My question")],
        )
        messages = self.bridge.build_messages_from_zones(
            compiled, zones, current_input="My question"
        )

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1

    def test_current_input_with_no_turns(self):
        compiled = _make_compiled({"core": [("p", "Prompt.")]})
        zones = self._make_zones()
        messages = self.bridge.build_messages_from_zones(
            compiled, zones, current_input="Hello"
        )

        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    # --- Requirement 7: Empty zones ---

    def test_empty_zones_just_system_message(self):
        compiled = _make_compiled({"core": [("prompt", "You are an assistant.")]})
        zones = self._make_zones()
        messages = self.bridge.build_messages_from_zones(compiled, zones)

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "You are an assistant." in messages[0]["content"]

    # --- Combined scenarios ---

    def test_full_scenario_all_zones_populated(self):
        """Integration: summaries + notes + compacted + raw + current input."""
        compiled = _make_compiled({
            "core": [("prompt", "You are helpful.")],
            "memory": [("mem", "User likes Python.")],
        })
        zones = self._make_zones(
            summarized=["Previous discussion about testing."],
            session_notes=["Note: prefer pytest."],
            compacted=[
                self._make_turn(1, "user", "Old question", compacted=True),
                self._make_turn(2, "assistant", "Old answer", compacted=True),
            ],
            raw=[
                self._make_turn(3, "user", "Recent question"),
                self._make_turn(4, "assistant", "Recent answer"),
            ],
        )
        messages = self.bridge.build_messages_from_zones(
            compiled, zones, current_input="New question"
        )

        # System message has layers + summary + notes
        system = messages[0]
        assert system["role"] == "system"
        assert "You are helpful." in system["content"]
        assert "User likes Python." in system["content"]
        assert "Previous discussion about testing." in system["content"]
        assert "Note: prefer pytest." in system["content"]

        # 4 turn messages + 1 current input
        non_system = messages[1:]
        assert len(non_system) == 5
        assert non_system[0]["content"] == "Old question"
        assert non_system[1]["content"] == "Old answer"
        assert non_system[2]["content"] == "Recent question"
        assert non_system[3]["content"] == "Recent answer"
        assert non_system[4]["content"] == "New question"
        assert non_system[4]["role"] == "user"

    def test_skip_session_with_summaries_and_turns(self):
        """Session layer skipped while summaries and turns still work."""
        compiled = _make_compiled({
            "core": [("prompt", "Core.")],
            "session": [("sess", "Skippable session content.")],
        })
        zones = self._make_zones(
            summarized=["Summary of earlier chat."],
            raw=[self._make_turn(1, "user", "Hello")],
        )
        messages = self.bridge.build_messages_from_zones(
            compiled, zones, skip_session_layer="session"
        )

        system = messages[0]
        assert "Core." in system["content"]
        assert "Skippable session content." not in system["content"]
        assert "Summary of earlier chat." in system["content"]
        assert messages[1]["content"] == "Hello"
