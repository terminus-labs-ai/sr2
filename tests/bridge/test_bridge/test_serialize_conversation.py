"""Tests for ClaudeCodeAdapter.serialize_conversation."""

from __future__ import annotations


from sr2_bridge.adapters.claude_code import ClaudeCodeAdapter


class TestSerializeConversation:
    """Verify serialize_conversation extracts the latest user message correctly."""

    def test_simple_text_message(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you?"},
        ]
        system, prompt = ClaudeCodeAdapter.serialize_conversation("sys", messages)
        assert system == "sys"
        assert prompt == "how are you?"

    def test_empty_messages(self):
        system, prompt = ClaudeCodeAdapter.serialize_conversation("sys", [])
        assert system == "sys"
        assert prompt == ""

    def test_no_user_message(self):
        messages = [{"role": "assistant", "content": "hi"}]
        system, prompt = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert system is None
        assert prompt == ""

    def test_content_block_list_text_only(self):
        """Content-block list with text blocks extracts text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "text", "text": "in detail"},
                ],
            }
        ]
        _, prompt = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert prompt == "describe this\nin detail"

    def test_content_block_list_with_image(self):
        """Image blocks get a text placeholder since CLI can't pass binary."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        },
                    },
                ],
            }
        ]
        _, prompt = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert "what is this?" in prompt
        assert "[Attached image: image/png]" in prompt

    def test_content_block_list_with_document(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "review this"},
                    {"type": "document", "source": {"type": "base64", "data": "..."}},
                ],
            }
        ]
        _, prompt = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert "review this" in prompt
        assert "[Attached document]" in prompt

    def test_content_block_list_unknown_type(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "check"},
                    {"type": "video", "data": "..."},
                ],
            }
        ]
        _, prompt = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert "[Attached media: video]" in prompt

    def test_only_latest_user_message(self):
        """Only the last user message should be extracted."""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "third"},
        ]
        _, prompt = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert prompt == "third"

    def test_system_prompt_passthrough(self):
        messages = [{"role": "user", "content": "hi"}]
        system, _ = ClaudeCodeAdapter.serialize_conversation("my system prompt", messages)
        assert system == "my system prompt"

    def test_none_system_prompt(self):
        messages = [{"role": "user", "content": "hi"}]
        system, _ = ClaudeCodeAdapter.serialize_conversation(None, messages)
        assert system is None
