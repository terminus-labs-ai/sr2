"""Telegram bot interface plugin."""

from __future__ import annotations

import logging
import os
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
        logger.info(f"Handling StreamEvent: {event}.")
        if isinstance(event, TextDeltaEvent):
            self.was_used = True
            self._buffer += event.content
            await self._maybe_flush()

        elif isinstance(event, ToolStartEvent):
            self.was_used = True
            self._buffer += f"\n_Using {event.tool_name}..._\n"
            await self._maybe_flush()

        elif isinstance(event, ToolResultEvent):
            self.was_used = True
            if event.success:
                self._buffer += f"\n_Using {event.tool_name}: {event.result}_\n"
            else:
                self._buffer += f"\n_Using {event.tool_name}: failed ({event.result})_\n"
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
        logger.info(f"Telegram._maybe_flush - Last edit {now - self._last_edit:.2f}s ago")
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
                    await self._current_msg.edit_text(text[:_MSG_LIMIT], parse_mode="Markdown")
                except Exception:
                    try:
                        await self._current_msg.edit_text(text[:_MSG_LIMIT])
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
                await self._current_msg.edit_text(text, parse_mode="Markdown")
            except Exception:
                try:
                    await self._current_msg.edit_text(text)
                except Exception as e:
                    logger.debug(f"Telegram edit_text failed: {e}")
        self._last_edit = time.monotonic()

    async def _safe_send(self, text: str):
        """Send a reply, falling back from Markdown to plain text."""
        try:
            return await self._reply_to.reply_text(text, parse_mode="Markdown")
        except Exception:
            try:
                return await self._reply_to.reply_text(text)
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
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
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
                    await self._bot.send_message(chat_id=uid, text=chunk, parse_mode="Markdown")
                except Exception:
                    try:
                        await self._bot.send_message(chat_id=uid, text=chunk)
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

    async def _safe_reply(self, message, text: str) -> None:
        """Send with Markdown, fall back to plain text on parse failure.

        Splits long messages into chunks to stay within Telegram's 4096 char limit.
        """
        for chunk in _TelegramStreamState._split_text(text, _MSG_LIMIT):
            try:
                await message.reply_text(chunk, parse_mode="Markdown")
            except Exception:
                try:
                    await message.reply_text(chunk)
                except Exception as e:
                    logger.warning(f"Telegram reply failed: {e}")
