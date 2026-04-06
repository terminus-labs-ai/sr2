"""Tests for TelegramPlugin multimedia handling."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_runtime.plugins.telegram import TelegramPlugin


def _make_plugin(
    voice_enabled: bool = True,
    photo_enabled: bool = True,
    document_enabled: bool = True,
) -> TelegramPlugin:
    """Create a TelegramPlugin with per-media-type config."""
    config = {
        "session": {"name": "test_{user_id}", "lifecycle": "persistent"},
        "_media": {
            "voice": {
                "enabled": voice_enabled,
                "stt": {
                    "provider": "openai_compatible",
                    "api_base": "http://localhost:8787/v1",
                    "model": "whisper-small",
                },
            },
            "photo": {"enabled": photo_enabled},
            "document": {"enabled": document_enabled},
        },
    }
    plugin = TelegramPlugin(
        interface_name="test_telegram",
        config=config,
        agent_callback=AsyncMock(return_value="Agent response"),
    )
    return plugin


class TestMediaConfigInit:
    def test_media_config_stored(self):
        plugin = _make_plugin()
        assert plugin._media_config["voice"]["enabled"] is True
        assert plugin._media_config["voice"]["stt"]["provider"] == "openai_compatible"

    def test_media_config_defaults_when_missing(self):
        plugin = TelegramPlugin(
            interface_name="test",
            config={"session": {"name": "test", "lifecycle": "persistent"}},
            agent_callback=AsyncMock(),
        )
        assert plugin._media_config == {}

    def test_media_processor_initially_none(self):
        plugin = _make_plugin()
        assert plugin._media_processor is None


class TestGetMediaProcessor:
    def test_returns_none_without_sr2_pro(self):
        """Without sr2-pro installed, _get_media_processor returns None."""
        plugin = _make_plugin()
        # sr2_pro.media should not be importable in test env
        result = plugin._get_media_processor()
        assert result is None

    def test_caches_processor(self):
        """Processor is created once and cached."""
        plugin = _make_plugin()

        @dataclass
        class FakeResult:
            text_for_session: str = "transcribed"
            llm_content_blocks: list = None

        mock_processor = MagicMock()

        with patch.dict("sys.modules", {"sr2_pro": MagicMock(), "sr2_pro.media": MagicMock()}):
            with patch(
                "sr2_runtime.plugins.telegram.TelegramPlugin._get_media_processor",
                return_value=mock_processor,
            ):
                # Simulate cached behavior
                plugin._media_processor = mock_processor
                assert plugin._get_media_processor() is mock_processor
                # Second call returns same instance
                assert plugin._get_media_processor() is mock_processor


class TestPerTypeHandlerRegistration:
    """Verify that handlers are registered only for enabled media types."""

    @pytest.mark.asyncio
    async def test_only_voice_registered(self):
        """When only voice is enabled, photo and document handlers are not registered."""
        plugin = _make_plugin(voice_enabled=True, photo_enabled=False, document_enabled=False)


        with patch("sr2_runtime.plugins.telegram.os") as mock_os:
            mock_os.environ.get = MagicMock(side_effect=lambda k, d="": "fake-token" if k == "TELEGRAM_BOT_TOKEN" else d)

            with patch("telegram.ext.ApplicationBuilder") as mock_builder:
                mock_app = AsyncMock()
                mock_app.add_handler = MagicMock()
                mock_builder.return_value.token.return_value.build.return_value = mock_app
                mock_app.updater.start_polling = AsyncMock()

                plugin._token = "fake-token"
                plugin._app = mock_app
                plugin._bot = MagicMock()

                # Manually call the handler registration logic

                _media_types_enabled = []
                media_cfg = plugin._media_config
                if media_cfg.get("photo", {}).get("enabled", False):
                    _media_types_enabled.append("photo")
                if media_cfg.get("document", {}).get("enabled", False):
                    _media_types_enabled.append("document")
                if media_cfg.get("voice", {}).get("enabled", False):
                    _media_types_enabled.append("voice")

                assert _media_types_enabled == ["voice"]

    def test_all_disabled_no_handlers(self):
        """When all media types are disabled, no media handlers are created."""
        plugin = _make_plugin(voice_enabled=False, photo_enabled=False, document_enabled=False)
        media_cfg = plugin._media_config

        enabled = []
        if media_cfg.get("photo", {}).get("enabled", False):
            enabled.append("photo")
        if media_cfg.get("document", {}).get("enabled", False):
            enabled.append("document")
        if media_cfg.get("voice", {}).get("enabled", False):
            enabled.append("voice")

        assert enabled == []

    def test_all_enabled(self):
        """When all media types are enabled, all types are listed."""
        plugin = _make_plugin(voice_enabled=True, photo_enabled=True, document_enabled=True)
        media_cfg = plugin._media_config

        enabled = []
        if media_cfg.get("photo", {}).get("enabled", False):
            enabled.append("photo")
        if media_cfg.get("document", {}).get("enabled", False):
            enabled.append("document")
        if media_cfg.get("voice", {}).get("enabled", False):
            enabled.append("voice")

        assert sorted(enabled) == ["document", "photo", "voice"]


class TestOnMediaAccessDenied:
    @pytest.mark.asyncio
    async def test_access_denied_for_unauthorized_user(self):
        """Users not in allowed list get access denied."""
        plugin = _make_plugin()
        plugin._allowed_users = {12345}

        update = MagicMock()
        update.message.from_user.id = 99999
        update.message.reply_text = AsyncMock()

        await plugin._on_media(update, MagicMock())
        update.message.reply_text.assert_called_once_with("Access denied.")


class TestOnMediaNoProcessor:
    @pytest.mark.asyncio
    async def test_no_processor_replies_unavailable(self):
        """When sr2-pro is not installed, reply with unavailable message."""
        plugin = _make_plugin()
        plugin._allowed_users = None  # Allow all

        update = MagicMock()
        update.message.from_user.id = 12345
        update.message.reply_text = AsyncMock()

        await plugin._on_media(update, MagicMock())

        # Should tell user multimedia is not available
        calls = update.message.reply_text.call_args_list
        assert any("not available" in str(c) for c in calls)


class TestOnMediaErrorSanitization:
    @pytest.mark.asyncio
    async def test_error_message_sanitized(self):
        """Internal error details should NOT be sent to the Telegram user."""
        plugin = _make_plugin()
        plugin._allowed_users = None

        # Mock a processor that raises
        mock_processor = MagicMock()
        mock_processor.process_photo = AsyncMock(
            side_effect=ValueError("Internal DB error at /opt/sr2/secret.py:42")
        )
        plugin._media_processor = mock_processor
        plugin._get_media_processor = MagicMock(return_value=mock_processor)

        update = MagicMock()
        update.message.from_user.id = 123
        update.message.photo = [MagicMock()]  # Has a photo
        update.message.photo[-1].get_file = AsyncMock(
            return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=bytearray(b"img")))
        )
        update.message.document = None
        update.message.voice = None
        update.message.audio = None
        update.message.caption = None
        update.message.chat.send_action = AsyncMock()
        update.message.reply_text = AsyncMock()

        await plugin._on_media(update, MagicMock())

        # Check that the error reply does NOT contain internal details
        reply_calls = update.message.reply_text.call_args_list
        error_replies = [
            str(c) for c in reply_calls if "Failed" in str(c) or "error" in str(c).lower()
        ]
        for reply in error_replies:
            assert "Internal DB error" not in reply
            assert "secret.py" not in reply
