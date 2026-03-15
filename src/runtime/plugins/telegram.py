"""Telegram bot interface plugin."""

from __future__ import annotations

import json
import logging
import os
import re
import time

from runtime.plugins.base import TriggerContext
from runtime.llm.streaming import (
    StreamEndEvent,
    StreamEvent,
    StreamRetractEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)

logger = logging.getLogger(__name__)

# Telegram message limit is 4096 chars; leave margin for formatting
_MSG_LIMIT = 4000
_EDIT_INTERVAL = 0.8  # seconds between edit_text calls


def _esc(text: str) -> str:
    """Escape HTML entities for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_to_telegram_html(text: str) -> str:
    """Convert common Markdown to Telegram-safe HTML.

    Handles fenced code blocks, inline code, bold, italic, strikethrough,
    links, headers, bullet/numbered lists, blockquotes, and escapes HTML
    entities in non-formatted text.
    """

    # Collect protected spans (start, end, replacement) so we don't double-process
    protected: list[tuple[int, int, str]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e, _ in protected)

    # 1. Fenced code blocks: ```lang\ncode\n```  (must be first — content is literal)
    for m in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
        code = m.group(1).rstrip("\n")
        protected.append((m.start(), m.end(), f"<pre><code>{_esc(code)}</code></pre>"))

    # --- Line-level formatting (runs on raw text, skipping code blocks) ---
    def _line_transform(text: str, protected: list) -> str:
        """Apply line-level markdown transforms, skipping protected code spans."""
        lines = text.split("\n")
        result_lines = []
        pos = 0  # track character position in original text
        for i, line in enumerate(lines):
            line_start = pos
            line_end = pos + len(line)
            # Check if this line falls inside a protected code block
            in_code = any(s <= line_start and line_end <= e for s, e, _ in protected)
            if not in_code:
                # Horizontal rules: --- or *** or ___ on their own line
                if re.match(r"^(---|\*\*\*|___)$", line):
                    line = "───────────"
                # Headers: strip # prefix and any wrapping ** from content
                else:
                    hm = re.match(r"^#{1,6}\s+(.+)$", line)
                    if hm:
                        inner = hm.group(1).strip()
                        # Strip wrapping bold markers if present: **text** → text
                        inner = re.sub(r"^\*\*(.+)\*\*$", r"\1", inner)
                        inner = re.sub(r"^__(.+)__$", r"\1", inner)
                        line = f"\x02BOLD\x02{inner}\x02/BOLD\x02"
                    # Bullet lists
                    elif re.match(r"^[-*]\s+", line):
                        line = re.sub(r"^[-*]\s+", "• ", line)
                    # Blockquotes
                    elif re.match(r"^>\s?", line):
                        bq_match = re.match(r"^>\s?(.*)$", line)
                        if bq_match:
                            line = "\x02BQSTART\x02" + bq_match.group(1) + "\x02BQEND\x02"
            result_lines.append(line)
            pos = line_end + 1  # +1 for the \n
        return "\n".join(result_lines)

    text = _line_transform(text, protected)

    # 2. Inline code: `code`
    for m in re.finditer(r"`([^`\n]+)`", text):
        if _overlaps(m.start(), m.end()):
            continue
        protected.append((m.start(), m.end(), f"<code>{_esc(m.group(1))}</code>"))

    # 3. Links: [text](url) — only create <a> tags for http/https URLs;
    #    Telegram rejects non-HTTP schemes and will fail the entire HTML parse.
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        if _overlaps(m.start(), m.end()):
            continue
        url = m.group(2)
        link_text = m.group(1)
        if url.startswith(("http://", "https://")):
            protected.append((m.start(), m.end(), f'<a href="{url}">{_esc(link_text)}</a>'))
        else:
            # Non-HTTP link: render as text only
            protected.append((m.start(), m.end(), _esc(link_text)))

    # 4. Bold+italic: ***text***
    for m in re.finditer(r"\*\*\*(.+?)\*\*\*", text):
        if _overlaps(m.start(), m.end()):
            continue
        protected.append((m.start(), m.end(), f"<b><i>{_esc(m.group(1))}</i></b>"))

    # 5. Bold: **text** or __text__ (may span lines)
    for m in re.finditer(r"\*\*(.+?)\*\*|__(.+?)__", text, re.DOTALL):
        if _overlaps(m.start(), m.end()):
            continue
        content = m.group(1) if m.group(1) is not None else m.group(2)
        protected.append((m.start(), m.end(), f"<b>{_esc(content)}</b>"))

    # 6. Strikethrough: ~~text~~
    for m in re.finditer(r"~~(.+?)~~", text):
        if _overlaps(m.start(), m.end()):
            continue
        protected.append((m.start(), m.end(), f"<s>{_esc(m.group(1))}</s>"))

    # 7. Italic: *text* (not inside words) or _text_ (not inside words/identifiers)
    #    Word-boundary guards prevent matching underscores in snake_case identifiers
    #    and URLs like https://example.com/my_page_here
    for m in re.finditer(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", text):
        if _overlaps(m.start(), m.end()):
            continue
        protected.append((m.start(), m.end(), f"<i>{_esc(m.group(1))}</i>"))
    for m in re.finditer(r"(?<!\w)_([^_\n]+?)_(?!\w)", text):
        if _overlaps(m.start(), m.end()):
            continue
        protected.append((m.start(), m.end(), f"<i>{_esc(m.group(1))}</i>"))

    # Sort by start position and build result
    protected.sort(key=lambda x: x[0])

    parts: list[str] = []
    pos = 0
    for start, end, replacement in protected:
        if start < pos:
            continue  # overlapping — skip
        # Escape plain text between protected spans
        plain = text[pos:start]
        parts.append(_esc(plain))
        parts.append(replacement)
        pos = end

    # Remaining plain text
    parts.append(_esc(text[pos:]))

    result = "".join(parts)

    # Convert sentinel markers to HTML tags (these were placed by line-level transforms)
    result = result.replace("\x02BOLD\x02", "<b>").replace("\x02/BOLD\x02", "</b>")
    result = result.replace("\x02BQSTART\x02", "┃ <i>").replace("\x02BQEND\x02", "</i>")

    return result


def _strip_markdown(text: str) -> str:
    """Strip Markdown formatting to produce clean plain text.

    Used as fallback when Telegram rejects our HTML conversion.
    """
    # Fenced code blocks → just the code
    text = re.sub(r"```(?:\w*)\n(.*?)```", r"\1", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bold+italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    # Strikethrough
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Italic (word-boundary guards)
    text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bullet lists
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^>\s?", "┃ ", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^(---|\*\*\*|___)$", "───────────", text, flags=re.MULTILINE)
    return text


class _TelegramStreamState:
    """Manages edit-in-place streaming for a single Telegram response.

    On first ``TextDeltaEvent``, sends a new message via ``reply_text()`` and
    stores the message object.  Subsequent deltas are accumulated in a buffer
    and flushed to Telegram via ``edit_text()`` at ~0.8 s intervals.

    When the accumulated text exceeds ``_MSG_LIMIT`` chars the current message
    is finalized and a new one is started.
    """

    def __init__(self, message):
        self._reply_to = message  # telegram Message to reply to
        self._current_msg = None  # the live message being edited
        self._buffer = ""
        self._last_edit: float = 0.0
        self.was_used = False

    async def handle_event(self, event: StreamEvent) -> None:
        """Stream callback handed to ``TriggerContext.respond_callback``."""
        logger.debug(f"Handling StreamEvent: {event}.")
        if isinstance(event, TextDeltaEvent):
            self.was_used = True
            self._buffer += event.content
            await self._maybe_flush()

        elif isinstance(event, StreamRetractEvent):
            # The text we streamed was actually a tool call, not user-facing.
            # Delete the live message and reset so tool status starts fresh.
            logger.info("Retracting streamed text (was a tool call, not content)")
            if self._current_msg:
                try:
                    await self._current_msg.delete()
                except Exception as e:
                    logger.debug(f"Could not delete retracted message: {e}")
            self._current_msg = None
            self._buffer = ""

        elif isinstance(event, StreamEndEvent):
            await self._flush()

    async def flush_final(self) -> None:
        """Ensure any remaining buffer text is sent after the loop ends."""
        logger.info(
            f"_TelegramStreamState.flush_final; used: {self.was_used}.\n{self._current_msg}"
        )
        if self._buffer:
            await self._flush()

    # --- internals ---

    async def _maybe_flush(self) -> None:
        now = time.monotonic()
        logger.debug(f"Telegram._maybe_flush - Last edit {now - self._last_edit:.2f}s ago")
        if now - self._last_edit < _EDIT_INTERVAL:
            return
        await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return

        # If we'd exceed the message limit, finalize current and start fresh
        if self._current_msg and len(self._buffer) > _MSG_LIMIT:
            # Finalize existing message with the first _MSG_LIMIT chars
            overflow = self._buffer[_MSG_LIMIT:]
            self._buffer = self._buffer[:_MSG_LIMIT]
            await self._edit_or_send()
            # Start a new message with the overflow
            self._current_msg = None
            self._buffer = overflow
            if self._buffer:
                await self._edit_or_send()
            return

        await self._edit_or_send()

    async def _edit_or_send(self) -> None:
        text = self._buffer
        if not text:
            return

        # If text exceeds Telegram's limit, split and send as multiple messages
        if len(text) > _MSG_LIMIT:
            # Finalize current message if one exists
            if self._current_msg:
                try:
                    await self._current_msg.edit_text(
                        _md_to_telegram_html(text[:_MSG_LIMIT]), parse_mode="HTML"
                    )
                except Exception as html_err:
                    logger.warning(
                        f"Telegram HTML parse failed (overflow), falling back: {html_err}"
                    )
                    try:
                        await self._current_msg.edit_text(_strip_markdown(text[:_MSG_LIMIT]))
                    except Exception as e:
                        logger.debug(f"Telegram edit_text failed: {e}")

            # Send remaining chunks as new messages
            remaining = text if not self._current_msg else text[_MSG_LIMIT:]
            self._current_msg = None
            for chunk in self._split_text(remaining, _MSG_LIMIT):
                self._current_msg = await self._safe_send(chunk)
            self._buffer = ""
            self._last_edit = time.monotonic()
            return

        if self._current_msg is None:
            # First message — send as a reply
            self._current_msg = await self._safe_send(text)
        else:
            # Edit existing message
            try:
                await self._current_msg.edit_text(_md_to_telegram_html(text), parse_mode="HTML")
            except Exception as html_err:
                logger.warning(
                    f"Telegram HTML parse failed, falling back to plain text: {html_err}"
                )
                try:
                    await self._current_msg.edit_text(_strip_markdown(text))
                except Exception as e:
                    logger.debug(f"Telegram edit_text failed: {e}")
        self._last_edit = time.monotonic()

    async def _safe_send(self, text: str):
        """Send a reply, falling back from HTML to stripped plain text."""
        try:
            return await self._reply_to.reply_text(_md_to_telegram_html(text), parse_mode="HTML")
        except Exception as html_err:
            logger.warning(f"Telegram HTML parse failed, falling back to plain text: {html_err}")
            try:
                return await self._reply_to.reply_text(_strip_markdown(text))
            except Exception as e:
                logger.warning(f"Telegram reply_text failed: {e}")
                return None

    @staticmethod
    def _split_text(text: str, max_len: int) -> list[str]:
        """Split text into chunks, preferring newline/space boundaries."""
        if not text:
            return []
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = text.rfind(" ", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        return chunks


class TelegramPlugin:
    """Telegram bot interface plugin.

    Config (from YAML):
        plugin: telegram
        session:
          name: main_chat
          lifecycle: persistent
        pipeline: interfaces/user_message.yaml

    Environment variables:
        TELEGRAM_BOT_TOKEN - Bot API token
        TELEGRAM_USER_IDS - Comma-separated allowed user IDs
    """

    def __init__(self, interface_name: str, config: dict, agent_callback):
        self._name = interface_name
        self._config = config
        self._callback = agent_callback
        self._session_config = config.get("session", {})

        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        user_ids_str = os.environ.get("TELEGRAM_USER_IDS", "")
        self._allowed_users = (
            {int(x) for x in user_ids_str.split(",") if x.strip()} if user_ids_str else None
        )

        self._app = None
        self._bot = None

    async def start(self) -> None:
        """Start the Telegram bot with polling."""
        if not self._token:
            logger.warning(f"Telegram plugin '{self._name}': no TELEGRAM_BOT_TOKEN set, skipping")
            return

        from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters

        self._app = ApplicationBuilder().token(self._token).build()
        self._bot = self._app.bot

        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("forget", self._cmd_forget))
        self._app.add_handler(CommandHandler("newsession", self._cmd_newsession))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        # Restore last active session
        try:
            trigger = TriggerContext(
                interface_name=self._name,
                plugin_name="telegram",
                session_name="_plugin_state_restore",
                session_lifecycle="ephemeral",
                input_data="__get_active_session__",
                metadata={"command": "get_active_session"},
            )
            resp = await self._callback(trigger)
            if resp:
                data = json.loads(resp)
                if data.get("active_session"):
                    self._session_config["name"] = data["active_session"]
                    logger.info(
                        f"Telegram plugin '{self._name}' restored active session: "
                        f"{data['active_session']}"
                    )
        except Exception as e:
            logger.debug(f"Could not restore active session: {e}")

        logger.info(f"Telegram plugin '{self._name}' started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info(f"Telegram plugin '{self._name}' stopped")

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """Proactively send a message to all allowed users."""
        logger.info(f"Telegram.send - {self._bot} - {self._allowed_users}")
        if not self._bot or not self._allowed_users:
            return
        for uid in self._allowed_users:
            for chunk in _TelegramStreamState._split_text(content, _MSG_LIMIT):
                try:
                    await self._bot.send_message(
                        chat_id=uid, text=_md_to_telegram_html(chunk), parse_mode="HTML"
                    )
                except Exception as html_err:
                    logger.warning(f"Telegram HTML parse failed (send): {html_err}")
                    try:
                        await self._bot.send_message(chat_id=uid, text=_strip_markdown(chunk))
                    except Exception as e:
                        logger.warning(f"Telegram send to {uid} failed: {e}")

    async def _on_message(self, update, context) -> None:
        """Handle incoming Telegram message."""
        logger.info(f"Telegram message received from {update.message.from_user.id}")
        if not update.message or not update.message.text:
            return
        user_id = update.message.from_user.id
        if self._allowed_users and user_id not in self._allowed_users:
            await update.message.reply_text("Access denied.")
            return

        await update.message.chat.send_action("typing")
        await update.message.reply_text("Processing...")

        session_name = self._session_config.get("name", f"telegram_{user_id}")
        lifecycle = self._session_config.get("lifecycle", "persistent")

        stream_state = _TelegramStreamState(message=update.message)

        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="telegram",
            session_name=session_name,
            session_lifecycle=lifecycle,
            input_data=update.message.text,
            metadata={"user_id": user_id, "chat_id": update.message.chat.id},
            respond_callback=stream_state.handle_event,
        )

        logger.info("Telegram TriggerContext sent, waiting...")
        response = ""
        try:
            response = await self._callback(trigger)
        except Exception as e:
            logger.exception(f"Telegram trigger error: {e}")
            await self._safe_reply(update.message, f"An error occurred: {e}")
            return

        logger.info(f"Telegram trigger response: {response}")
        await stream_state.flush_final()

        logger.info(f"Telegram flush_final: {response}")
        # Fallback: if streaming wasn't used, send response as chunked messages
        if not stream_state.was_used and response:
            await self._safe_reply(update.message, response)

    async def _cmd_status(self, update, context) -> None:
        """Show agent status with session metrics."""
        user_id = update.message.from_user.id
        session_name = self._session_config.get("name", f"telegram_{user_id}")
        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="telegram",
            session_name=session_name,
            session_lifecycle="persistent",
            input_data="__get_status__",
            metadata={"command": "status"},
        )
        try:
            resp = await self._callback(trigger)
            data = json.loads(resp)

            model = data.get("model", "unknown")
            lines = [
                f"<b>{data.get('agent_name', 'Agent')}</b> — Online",
                f"Model: <code>{model}</code>",
            ]

            # Context & token metrics
            m = data.get("metrics", {})
            budget = data.get("token_budget")
            if m:
                lines.append("")
                lines.append("<b>Context:</b>")
                total = m.get("total_tokens")
                if total is not None and budget:
                    pct = total / budget * 100 if budget else 0
                    lines.append(f"  Tokens: {int(total):,} / {int(budget):,} ({pct:.0f}%)")
                elif total is not None:
                    lines.append(f"  Tokens: {int(total):,}")
                headroom = m.get("budget_headroom_ratio")
                if headroom is not None:
                    lines.append(f"  Headroom: {headroom * 100:.0f}%")
                cache = m.get("cache_hit_rate")
                if cache is not None:
                    lines.append(f"  Cache hit rate: {cache * 100:.0f}%")
                window = m.get("raw_window_utilization")
                if window is not None:
                    lines.append(f"  Window utilization: {window * 100:.0f}%")
                trunc = m.get("truncation_events")
                if trunc is not None and trunc > 0:
                    lines.append(f"  Truncation events: {int(trunc)}")
                savings = m.get("token_savings_cumulative")
                if savings is not None and savings > 0:
                    lines.append(f"  Token savings (cumulative): {int(savings):,}")

            # Session info
            sess = data.get("session")
            if sess:
                lines.append("")
                lines.append(f"<b>Session:</b> <code>{sess['name']}</code>")
                lines.append(f"  Turns: {sess['turn_count']} ({sess['user_message_count']} user)")
                lines.append(f"  Created: {sess['created_at'][:19]}")
                lines.append(f"  Last activity: {sess['last_activity'][:19]}")
                if m:
                    iters = m.get("loop_iterations")
                    tools = m.get("loop_tool_calls")
                    if iters is not None or tools is not None:
                        parts = []
                        if iters is not None:
                            parts.append(f"{int(iters)} loop iters")
                        if tools is not None:
                            parts.append(f"{int(tools)} tool calls")
                        lines.append(f"  Last turn: {', '.join(parts)}")

            # Other sessions
            active = data.get("active_sessions", [])
            if active and len(active) > 1:
                lines.append("")
                lines.append(f"<b>Active sessions ({len(active)}):</b>")
                for s in active:
                    if s.startswith("_plugin_state_"):
                        continue
                    marker = " (current)" if sess and s == sess["name"] else ""
                    lines.append(f"  • <code>{s}</code>{marker}")

            # MCP
            mcp = data.get("mcp_servers", [])
            if mcp:
                lines.append("")
                lines.append(f"<b>MCP servers ({len(mcp)}):</b>")
                for s in mcp:
                    lines.append(f"  • {s}")

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            logger.exception(f"Status command failed: {e}")
            await update.message.reply_text("Online.")

    async def _cmd_forget(self, update, context) -> None:
        """Clear the session for this user."""
        user_id = update.message.from_user.id
        session_name = self._session_config.get("name", f"telegram_{user_id}")
        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="telegram",
            session_name=session_name,
            session_lifecycle="persistent",
            input_data="__clear_session__",
            metadata={"command": "forget"},
        )
        await self._callback(trigger)
        await update.message.reply_text("Session cleared.")

    async def _cmd_newsession(self, update, context) -> None:
        """Start a new session, clearing the current one."""
        user_id = update.message.from_user.id
        session_name = self._session_config.get("name", f"telegram_{user_id}")
        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="telegram",
            session_name=session_name,
            session_lifecycle="persistent",
            input_data="__clear_session__",
            metadata={"command": "newsession"},
        )
        await self._callback(trigger)
        await update.message.reply_text("New session started.")

    async def _cmd_resume(self, update, context) -> None:
        """List or switch to an existing session."""
        args = (context.args or []) if context else []

        if args:
            # Switch to the specified session
            target = args[0]
            self._session_config["name"] = target

            # Persist the choice
            trigger = TriggerContext(
                interface_name=self._name,
                plugin_name="telegram",
                session_name="_plugin_state_resume",
                session_lifecycle="ephemeral",
                input_data=f"__set_active_session__:{target}",
                metadata={"command": "resume"},
            )
            await self._callback(trigger)
            await update.message.reply_text(
                f"Switched to session <code>{target}</code>.", parse_mode="HTML"
            )
            return

        # No args — list available sessions
        user_id = update.message.from_user.id
        session_name = self._session_config.get("name", f"telegram_{user_id}")
        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="telegram",
            session_name=session_name,
            session_lifecycle="persistent",
            input_data="__list_sessions__",
            metadata={"command": "resume"},
        )
        try:
            resp = await self._callback(trigger)
            sessions = json.loads(resp)

            if not sessions:
                await update.message.reply_text("No active sessions.")
                return

            lines = ["<b>Available sessions:</b>"]
            for s in sessions:
                marker = " (current)" if s["id"] == session_name else ""
                lines.append(
                    f"  • <code>{s['id']}</code>{marker}\n"
                    f"    Turns: {s['turn_count']} | "
                    f"Last: {s['last_activity'][:19]}"
                )
            lines.append("")
            lines.append("Use /resume &lt;name&gt; to switch.")

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            logger.exception(f"Resume command failed: {e}")
            await update.message.reply_text("Could not list sessions.")

    async def _safe_reply(self, message, text: str) -> None:
        """Send with HTML, fall back to plain text on parse failure.

        Splits long messages into chunks to stay within Telegram's 4096 char limit.
        """
        for chunk in _TelegramStreamState._split_text(text, _MSG_LIMIT):
            try:
                await message.reply_text(_md_to_telegram_html(chunk), parse_mode="HTML")
            except Exception as html_err:
                logger.warning(f"Telegram HTML parse failed (_safe_reply): {html_err}")
                try:
                    await message.reply_text(_strip_markdown(chunk))
                except Exception as e:
                    logger.warning(f"Telegram reply failed: {e}")
