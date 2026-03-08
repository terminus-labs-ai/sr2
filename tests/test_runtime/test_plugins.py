"""Tests for interface plugins."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runtime.plugins.base import TriggerContext
from runtime.plugins.registry import PluginRegistry, create_default_registry
from runtime.plugins.timer import TimerPlugin
from runtime.plugins.telegram import TelegramPlugin
from runtime.plugins.http import HTTPPlugin
from runtime.plugins.a2a import A2APlugin
from runtime.plugins.single_shot import SingleShotPlugin


# --- Registry ---


class TestPluginRegistry:
    def test_register_and_get(self):
        reg = PluginRegistry()
        reg.register("test", TimerPlugin)
        assert reg.get("test") is TimerPlugin

    def test_get_unknown_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="Unknown plugin"):
            reg.get("nope")

    def test_available(self):
        reg = PluginRegistry()
        reg.register("a", TimerPlugin)
        reg.register("b", HTTPPlugin)
        assert set(reg.available) == {"a", "b"}

    def test_create_default_registry(self):
        reg = create_default_registry()
        assert "telegram" in reg.available
        assert "timer" in reg.available
        assert "http" in reg.available
        assert "a2a" in reg.available
        assert "single_shot" in reg.available


# --- Timer Plugin ---


class TestTimerPlugin:
    def test_init_defaults(self):
        cb = AsyncMock()
        plugin = TimerPlugin("email_check", {"interval_seconds": 60}, cb)
        assert plugin._interval == 60
        assert plugin._enabled is True

    def test_init_disabled(self):
        cb = AsyncMock()
        plugin = TimerPlugin("email_check", {"enabled": False}, cb)
        assert plugin._enabled is False

    @pytest.mark.asyncio
    async def test_start_disabled_noop(self):
        cb = AsyncMock()
        plugin = TimerPlugin("t", {"enabled": False}, cb)
        await plugin.start()
        assert plugin._task is None

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        cb = AsyncMock(return_value="ok")
        plugin = TimerPlugin("t", {"interval_seconds": 0.05, "session": {"name": "hb", "lifecycle": "ephemeral"}}, cb)
        await plugin.start()
        assert plugin._running is True
        assert plugin._task is not None
        await asyncio.sleep(0.15)
        await plugin.stop()
        assert plugin._running is False
        assert cb.call_count >= 1
        trigger = cb.call_args[0][0]
        assert isinstance(trigger, TriggerContext)
        assert trigger.plugin_name == "timer"
        assert trigger.session_name == "hb"
        assert trigger.session_lifecycle == "ephemeral"

    @pytest.mark.asyncio
    async def test_send_is_noop(self):
        cb = AsyncMock()
        plugin = TimerPlugin("t", {}, cb)
        await plugin.send("session", "msg")  # Should not raise


# --- Telegram Plugin ---


class TestTelegramPlugin:
    def test_init_no_token(self):
        cb = AsyncMock()
        with patch.dict("os.environ", {}, clear=True):
            plugin = TelegramPlugin("tg", {"session": {"name": "main"}}, cb)
            assert plugin._token == ""

    def test_init_with_users(self):
        cb = AsyncMock()
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_IDS": "1,2,3"}):
            plugin = TelegramPlugin("tg", {}, cb)
            assert plugin._allowed_users == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_start_skips_without_token(self):
        cb = AsyncMock()
        with patch.dict("os.environ", {}, clear=True):
            plugin = TelegramPlugin("tg", {}, cb)
            await plugin.start()  # Should not raise
            assert plugin._app is None

    def test_split_short_text(self):
        from runtime.plugins.telegram import _TelegramStreamState
        assert _TelegramStreamState._split_text("hello", 10) == ["hello"]

    def test_split_long_text(self):
        from runtime.plugins.telegram import _TelegramStreamState
        text = "word " * 20  # 100 chars
        chunks = _TelegramStreamState._split_text(text, 30)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 30

    @pytest.mark.asyncio
    async def test_send_no_bot(self):
        cb = AsyncMock()
        plugin = TelegramPlugin("tg", {}, cb)
        await plugin.send("session", "msg")  # No-op, no bot


# --- HTTP Plugin ---


class TestHTTPPlugin:
    def test_init(self):
        cb = AsyncMock()
        plugin = HTTPPlugin("api", {"port": 9000, "session": {"name": "http_sess"}}, cb)
        assert plugin._port == 9000

    @pytest.mark.asyncio
    async def test_start_stop(self):
        cb = AsyncMock()
        plugin = HTTPPlugin("api", {}, cb)
        await plugin.start()  # No-op
        await plugin.stop()   # No-op

    @pytest.mark.asyncio
    async def test_send_is_noop(self):
        cb = AsyncMock()
        plugin = HTTPPlugin("api", {}, cb)
        await plugin.send("session", "msg")

    def test_get_routes_returns_chat(self):
        cb = AsyncMock(return_value="response text")
        plugin = HTTPPlugin("api", {"session": {"name": "test_sess", "lifecycle": "persistent"}}, cb)
        routes = plugin.get_routes()
        assert "chat" in routes

    @pytest.mark.asyncio
    async def test_chat_route_creates_trigger(self):
        cb = AsyncMock(return_value="hi there")
        plugin = HTTPPlugin(
            "api",
            {"session": {"name": "{request.session_id}", "lifecycle": "persistent"}},
            cb,
        )
        routes = plugin.get_routes()

        # Mock FastAPI Request
        mock_request = AsyncMock()
        mock_request.json.return_value = {"message": "hello", "session_id": "sess_42"}
        mock_request.headers = {}

        resp = await routes["chat"](mock_request)
        assert cb.call_count == 1
        trigger = cb.call_args[0][0]
        assert trigger.session_name == "sess_42"
        assert trigger.input_data == "hello"
        assert trigger.plugin_name == "http"


# --- A2A Plugin ---


class TestA2APlugin:
    def test_init(self):
        cb = AsyncMock()
        plugin = A2APlugin("a2a", {"session": {"name": "a2a_{task_id}"}}, cb)
        assert plugin._name == "a2a"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        cb = AsyncMock()
        plugin = A2APlugin("a2a", {}, cb)
        await plugin.start()
        await plugin.stop()

    @pytest.mark.asyncio
    async def test_handle_a2a_request(self):
        cb = AsyncMock(return_value="task result")
        plugin = A2APlugin(
            "agent_calls",
            {"session": {"name": "a2a_{task_id}", "lifecycle": "ephemeral"}},
            cb,
        )
        result = await plugin.handle_a2a_request("t123", "do something", {"extra": "data"})
        assert result == "task result"
        trigger = cb.call_args[0][0]
        assert trigger.session_name == "a2a_t123"
        assert trigger.session_lifecycle == "ephemeral"
        assert trigger.input_data == "do something"
        assert trigger.metadata["task_id"] == "t123"
        assert trigger.metadata["extra"] == "data"

    @pytest.mark.asyncio
    async def test_send_is_noop(self):
        cb = AsyncMock()
        plugin = A2APlugin("a2a", {}, cb)
        await plugin.send("session", "msg")


# --- SingleShot Plugin ---


class TestSingleShotPlugin:
    def test_init(self):
        cb = AsyncMock()
        plugin = SingleShotPlugin("task_runner", {"session": {"name": "runner", "lifecycle": "ephemeral"}}, cb)
        assert plugin._name == "task_runner"

    @pytest.mark.asyncio
    async def test_start_stop_are_noops(self):
        cb = AsyncMock()
        plugin = SingleShotPlugin("task_runner", {}, cb)
        await plugin.start()
        await plugin.stop()

    @pytest.mark.asyncio
    async def test_send_is_noop(self):
        cb = AsyncMock()
        plugin = SingleShotPlugin("task_runner", {}, cb)
        await plugin.send("session", "msg")

    @pytest.mark.asyncio
    async def test_run_fires_trigger(self):
        cb = AsyncMock(return_value="done")
        plugin = SingleShotPlugin(
            "task_runner",
            {"session": {"name": "runner", "lifecycle": "ephemeral"}},
            cb,
        )
        result = await plugin.run("implement auth")
        assert result == "done"
        assert cb.call_count == 1
        trigger = cb.call_args[0][0]
        assert isinstance(trigger, TriggerContext)
        assert trigger.plugin_name == "single_shot"
        assert trigger.session_name == "runner"
        assert trigger.session_lifecycle == "ephemeral"
        assert trigger.input_data == "implement auth"

    @pytest.mark.asyncio
    async def test_run_uses_interface_name_as_default_session(self):
        cb = AsyncMock(return_value="ok")
        plugin = SingleShotPlugin("my_runner", {}, cb)
        await plugin.run("hello")
        trigger = cb.call_args[0][0]
        assert trigger.session_name == "my_runner"
        assert trigger.session_lifecycle == "ephemeral"
