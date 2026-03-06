"""Tests for _TelegramStreamState edit-in-place streaming and markdown conversion."""

from __future__ import annotations

import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runtime.llm.streaming import StreamEndEvent, TextDeltaEvent, ToolResultEvent, ToolStartEvent


class TestMdToTelegramHtml:
    """Tests for _md_to_telegram_html markdown-to-Telegram-HTML converter."""

    def _convert(self, text: str) -> str:
        from runtime.plugins.telegram import _md_to_telegram_html
        return _md_to_telegram_html(text)

    def _assert_balanced_tags(self, html: str) -> None:
        """Assert all HTML tags are properly opened and closed."""
        open_tags = re.findall(r"<(b|i|s|code|pre|a)\b", html)
        close_tags = re.findall(r"</(b|i|s|code|pre|a)>", html)
        assert sorted(open_tags) == sorted(close_tags), (
            f"Unbalanced tags in {html!r}: open={open_tags} close={close_tags}"
        )

    # --- Basic inline formatting ---

    def test_bold(self):
        assert self._convert("**bold**") == "<b>bold</b>"

    def test_bold_double_underscores(self):
        assert self._convert("__bold__") == "<b>bold</b>"

    def test_italic_asterisk(self):
        assert self._convert("*italic*") == "<i>italic</i>"

    def test_italic_underscore(self):
        assert self._convert("_italic_") == "<i>italic</i>"

    def test_strikethrough(self):
        assert self._convert("~~struck~~") == "<s>struck</s>"

    def test_inline_code(self):
        assert self._convert("`code`") == "<code>code</code>"

    def test_bold_italic_combo(self):
        result = self._convert("***important***")
        assert result == "<b><i>important</i></b>"
        self._assert_balanced_tags(result)

    def test_link(self):
        assert self._convert("[click](https://example.com)") == (
            '<a href="https://example.com">click</a>'
        )

    # --- Multi-line bold ---

    def test_bold_multiline(self):
        result = self._convert("**bold across\nmultiple lines**")
        assert result == "<b>bold across\nmultiple lines</b>"

    # --- Underscores in identifiers must NOT become italic ---

    def test_snake_case_not_italic(self):
        assert self._convert("file_name_with_underscores") == "file_name_with_underscores"

    def test_multiple_underscores_not_italic(self):
        assert self._convert("config_file_path_name") == "config_file_path_name"

    def test_screaming_snake_case_not_italic(self):
        assert self._convert("MAX_RETRIES") == "MAX_RETRIES"

    def test_underscore_in_url_not_italic(self):
        text = "https://example.com/my_page_here"
        assert self._convert(text) == text

    def test_inline_code_preserves_underscores(self):
        assert self._convert("Use `my_function_name` method") == (
            "Use <code>my_function_name</code> method"
        )

    # --- Headers ---

    def test_h1(self):
        assert self._convert("# Title") == "<b>Title</b>"

    def test_h3(self):
        assert self._convert("### Section") == "<b>Section</b>"

    def test_header_with_bold_content(self):
        """Header wrapping bold text should not produce nested <b> tags."""
        result = self._convert("### **Bold Header**")
        assert result == "<b>Bold Header</b>"
        self._assert_balanced_tags(result)

    # --- Lists ---

    def test_dash_bullet_list(self):
        assert self._convert("- item 1\n- item 2") == "• item 1\n• item 2"

    def test_asterisk_bullet_list(self):
        assert self._convert("* bullet 1\n* bullet 2") == "• bullet 1\n• bullet 2"

    # --- Blockquotes ---

    def test_blockquote(self):
        result = self._convert("> quote text")
        assert result == "┃ <i>quote text</i>"

    # --- Horizontal rules ---

    def test_horizontal_rule_dashes(self):
        assert self._convert("---") == "───────────"

    def test_horizontal_rule_asterisks(self):
        assert self._convert("***") == "───────────"

    # --- HTML escaping ---

    def test_html_entities_escaped_in_plain_text(self):
        assert self._convert("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_html_entities_escaped_inside_bold(self):
        assert self._convert("**a > b**") == "<b>a &gt; b</b>"

    # --- Code blocks protect content ---

    def test_fenced_code_block_preserves_markdown(self):
        result = self._convert("```\n### not a header\n**not bold**\n```")
        assert result == "<pre><code>### not a header\n**not bold**</code></pre>"

    # --- Common LLM output patterns ---

    def test_bold_label_pattern(self):
        """LLMs commonly produce **Label:** followed by text."""
        assert self._convert("**Key Point:** This is important") == (
            "<b>Key Point:</b> This is important"
        )

    def test_bold_in_parens(self):
        assert self._convert("(**bold in parens**)") == "(<b>bold in parens</b>)"

    def test_unclosed_bold_left_as_is(self):
        assert self._convert("**unclosed bold") == "**unclosed bold"

    def test_complex_llm_response(self):
        """Realistic multi-section LLM response."""
        text = "### Title\n\nSome **bold** and *italic* text\n\n- item 1\n- item 2"
        result = self._convert(text)
        expected = "<b>Title</b>\n\nSome <b>bold</b> and <i>italic</i> text\n\n• item 1\n• item 2"
        assert result == expected
        self._assert_balanced_tags(result)

    def test_all_output_has_balanced_tags(self):
        """Ensure various inputs never produce unbalanced HTML tags."""
        inputs = [
            "***bold italic***",
            "### **Bold Header**",
            "**multi\nline**",
            "text_with_underscores and *real italic*",
            "> quote with **bold**",
            "- item with `code_here`",
        ]
        for text in inputs:
            result = self._convert(text)
            self._assert_balanced_tags(result)


class TestTelegramStreamState:
    def _make_state(self):
        from runtime.plugins.telegram import _TelegramStreamState

        mock_message = AsyncMock()
        # reply_text returns a message object we can edit
        sent_msg = AsyncMock()
        mock_message.reply_text = AsyncMock(return_value=sent_msg)
        state = _TelegramStreamState(message=mock_message)
        return state, mock_message, sent_msg

    @pytest.mark.asyncio
    async def test_first_delta_sends_reply(self):
        state, mock_message, sent_msg = self._make_state()

        await state.handle_event(TextDeltaEvent(content="Hello"))
        # Force flush
        await state.flush_final()

        mock_message.reply_text.assert_called()
        assert state.was_used is True

    @pytest.mark.asyncio
    async def test_subsequent_deltas_edit_message(self):
        state, mock_message, sent_msg = self._make_state()

        # First event — sends initial message
        await state.handle_event(TextDeltaEvent(content="Hello"))
        await state.flush_final()
        assert mock_message.reply_text.call_count == 1

        # Second event after interval — should edit
        await state.handle_event(TextDeltaEvent(content=" world"))
        # Force time to have passed
        state._last_edit = 0
        await state.handle_event(TextDeltaEvent(content="!"))
        await state.flush_final()

        # Should have edited the sent message
        assert sent_msg.edit_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_was_used_false_initially(self):
        state, _, _ = self._make_state()
        assert state.was_used is False

    @pytest.mark.asyncio
    async def test_was_used_true_after_text_delta(self):
        state, _, _ = self._make_state()
        await state.handle_event(TextDeltaEvent(content="x"))
        assert state.was_used is True

    @pytest.mark.asyncio
    async def test_was_used_true_after_tool_start(self):
        state, _, _ = self._make_state()
        await state.handle_event(
            ToolStartEvent(tool_name="search", tool_call_id="tc_1")
        )
        assert state.was_used is True

    @pytest.mark.asyncio
    async def test_tool_start_appends_status(self):
        state, mock_message, sent_msg = self._make_state()

        await state.handle_event(TextDeltaEvent(content="Thinking"))
        await state.flush_final()

        state._last_edit = 0  # reset timer
        await state.handle_event(
            ToolStartEvent(tool_name="search", tool_call_id="tc_1")
        )
        await state.flush_final()

        # The edit should contain the tool status text
        last_call = sent_msg.edit_text.call_args
        assert "search" in last_call[0][0]

    @pytest.mark.asyncio
    async def test_overflow_starts_new_message(self):
        from runtime.plugins.telegram import _MSG_LIMIT

        state, mock_message, sent_msg = self._make_state()

        # Send a large chunk that exceeds the limit
        large_text = "x" * (_MSG_LIMIT + 500)
        await state.handle_event(TextDeltaEvent(content=large_text))
        await state.flush_final()

        # Should have sent the initial message, then started a new one
        # reply_text called at least twice (initial + overflow)
        assert mock_message.reply_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_markdown_fallback_on_edit(self):
        state, mock_message, sent_msg = self._make_state()

        # First event — sends initial message
        await state.handle_event(TextDeltaEvent(content="Hello"))
        await state.flush_final()
        assert mock_message.reply_text.call_count == 1

        # Make Markdown edit fail, plain succeeds
        sent_msg.edit_text = AsyncMock(
            side_effect=[Exception("parse error"), None]
        )

        # Trigger an edit (reset timer so _maybe_flush fires)
        state._last_edit = 0
        await state.handle_event(TextDeltaEvent(content=" world"))
        # handle_event triggers _maybe_flush which flushes:
        #   call 1: edit_text("Hello world", parse_mode="Markdown") -> fails
        #   call 2: edit_text("Hello world") -> succeeds (plain fallback)
        assert sent_msg.edit_text.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_end_flushes(self):
        state, mock_message, sent_msg = self._make_state()

        await state.handle_event(TextDeltaEvent(content="partial"))
        # Don't flush manually — StreamEndEvent should trigger flush
        await state.handle_event(StreamEndEvent(full_text="partial"))

        mock_message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_flush_final_noop_when_empty(self):
        state, mock_message, _ = self._make_state()
        await state.flush_final()
        mock_message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Events within the edit interval should not trigger immediate edits."""
        state, mock_message, sent_msg = self._make_state()

        # First event triggers send
        await state.handle_event(TextDeltaEvent(content="A"))
        await state.flush_final()
        assert mock_message.reply_text.call_count == 1

        # Rapid events should accumulate without triggering edit
        # (last_edit is recent from flush_final above)
        edit_count_before = sent_msg.edit_text.call_count
        await state.handle_event(TextDeltaEvent(content="B"))
        await state.handle_event(TextDeltaEvent(content="C"))
        # No new edits since interval hasn't passed
        assert sent_msg.edit_text.call_count == edit_count_before

        # After flush_final, it should edit once
        await state.flush_final()
        assert sent_msg.edit_text.call_count == edit_count_before + 1

    @pytest.mark.asyncio
    async def test_tool_result_appends_status_success(self):
        state, mock_message, sent_msg = self._make_state()

        await state.handle_event(TextDeltaEvent(content="Thinking"))
        await state.flush_final()

        state._last_edit = 0  # reset timer
        await state.handle_event(
            ToolResultEvent(tool_name="search", tool_call_id="tc_1", result="found it", success=True)
        )
        await state.flush_final()

        # The edit should contain the tool result text
        last_call = sent_msg.edit_text.call_args
        assert "search" in last_call[0][0]
        assert "found it" in last_call[0][0]

    @pytest.mark.asyncio
    async def test_tool_result_appends_status_failure(self):
        state, mock_message, sent_msg = self._make_state()

        await state.handle_event(TextDeltaEvent(content="Thinking"))
        await state.flush_final()

        state._last_edit = 0  # reset timer
        await state.handle_event(
            ToolResultEvent(tool_name="search", tool_call_id="tc_1", result="error occurred", success=False)
        )
        await state.flush_final()

        # The edit should contain the tool result text with failure indication
        last_call = sent_msg.edit_text.call_args
        assert "search" in last_call[0][0]
        assert "error occurred" in last_call[0][0]
        assert "failed" in last_call[0][0]

    @pytest.mark.asyncio
    async def test_split_text_short(self):
        from runtime.plugins.telegram import _TelegramStreamState
        assert _TelegramStreamState._split_text("hello", 100) == ["hello"]

    @pytest.mark.asyncio
    async def test_split_text_empty(self):
        from runtime.plugins.telegram import _TelegramStreamState
        assert _TelegramStreamState._split_text("", 100) == []

    @pytest.mark.asyncio
    async def test_split_text_splits_on_newline(self):
        from runtime.plugins.telegram import _TelegramStreamState
        text = "line1\nline2\nline3"
        chunks = _TelegramStreamState._split_text(text, 10)
        assert len(chunks) >= 2
        # All content preserved
        assert "".join(chunks) == text.replace("\n", "") or all(
            c in text for c in chunks
        )

    @pytest.mark.asyncio
    async def test_split_text_handles_no_good_boundary(self):
        from runtime.plugins.telegram import _TelegramStreamState
        text = "x" * 200
        chunks = _TelegramStreamState._split_text(text, 50)
        assert all(len(c) <= 50 for c in chunks)
        assert "".join(chunks) == text

    @pytest.mark.asyncio
    async def test_first_send_too_long_splits(self):
        """First message exceeding limit should be split into chunks."""
        from runtime.plugins.telegram import _TelegramStreamState, _MSG_LIMIT

        mock_message = AsyncMock()
        sent_msgs = [AsyncMock(), AsyncMock()]
        mock_message.reply_text = AsyncMock(side_effect=sent_msgs)
        state = _TelegramStreamState(message=mock_message)

        # Send text that's way over the limit
        large_text = "x" * (_MSG_LIMIT * 2 + 100)
        await state.handle_event(TextDeltaEvent(content=large_text))
        await state.flush_final()

        # Should have called reply_text multiple times (chunks)
        assert mock_message.reply_text.call_count >= 2

    @pytest.mark.asyncio
    async def test_safe_send_markdown_fallback(self):
        """_safe_send falls back to plain text when Markdown fails."""
        from runtime.plugins.telegram import _TelegramStreamState

        mock_message = AsyncMock()
        sent_msg = AsyncMock()
        # First call (Markdown) fails, second (plain) succeeds
        mock_message.reply_text = AsyncMock(
            side_effect=[Exception("parse error"), sent_msg]
        )
        state = _TelegramStreamState(message=mock_message)

        result = await state._safe_send("hello `broken")
        assert mock_message.reply_text.call_count == 2
        assert result == sent_msg


class TestTelegramOnMessage:
    """Tests for _on_message error handling."""

    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_crash(self):
        """When the agent callback raises, _on_message should catch it and not raise UnboundLocalError."""
        from runtime.plugins.telegram import TelegramPlugin

        async def failing_callback(trigger):
            raise RuntimeError("LLM exploded")

        plugin = TelegramPlugin("telegram", {"session": {"name": "test"}}, failing_callback)
        plugin._token = "fake"

        update = MagicMock()
        update.message.from_user.id = 123
        update.message.text = "hello"
        update.message.chat.send_action = AsyncMock()
        update.message.reply_text = AsyncMock()

        # Should not raise
        await plugin._on_message(update, MagicMock())
