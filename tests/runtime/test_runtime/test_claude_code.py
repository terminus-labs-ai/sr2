"""Tests for the Claude Code CLI provider."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_bridge.adapters.claude_code import ClaudeCodeAdapter
from sr2_bridge.adapters.claude_code_config import ClaudeCodeAdapterConfig
from sr2_runtime.llm.streaming import (
    StreamEndEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)

# Aliases for minimal diff — the refactor moved ClaudeCodeProvider to
# ClaudeCodeAdapter and ClaudeCodeConfig to ClaudeCodeAdapterConfig.
ClaudeCodeConfig = ClaudeCodeAdapterConfig
ClaudeCodeProvider = ClaudeCodeAdapter


def _default_config(**overrides) -> ClaudeCodeConfig:
    """Create a config with defaults, patching path validation."""
    defaults = {
        "path": "/usr/bin/true",  # Exists on all Linux
        "allowed_tools": ["Read", "Glob"],
        "timeout_seconds": 30,
        "max_concurrent": 2,
    }
    defaults.update(overrides)
    # Remove fields not in ClaudeCodeAdapterConfig
    defaults.pop("enabled", None)
    return ClaudeCodeConfig(**defaults)


def _make_stream_lines(*events: dict) -> bytes:
    """Encode events as newline-delimited JSON (stream-json format)."""
    return b"\n".join(json.dumps(e).encode() for e in events) + b"\n"


# --- Fixtures ---


@pytest.fixture
def mock_subprocess():
    """Fixture that patches create_subprocess_exec and returns a controllable process."""
    proc = AsyncMock()
    proc.returncode = 0
    proc.stdout = AsyncMock()
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# --- Config tests ---


def test_config_defaults():
    config = ClaudeCodeConfig()
    assert config.path == "claude"
    assert "Read" in config.allowed_tools
    assert config.max_concurrent == 3
    assert config.timeout_seconds == 300
    assert config.working_directory is None
    assert config.env == {}


def test_config_custom_values():
    config = ClaudeCodeConfig(
        path="/opt/bin/claude",
        allowed_tools=["Bash", "Edit"],
        permission_mode="acceptEdits",
        max_turns=10,
        max_budget_usd=5.0,
        max_concurrent=5,
        timeout_seconds=60,
        working_directory="/home/workspace",
        env={"FOO": "bar"},
    )
    assert config.max_concurrent == 5
    assert config.working_directory == "/home/workspace"
    assert config.env == {"FOO": "bar"}


# --- Provider init tests ---


def test_cli_not_installed():
    config = ClaudeCodeConfig(path="/nonexistent/claude-fake-path")
    with pytest.raises(FileNotFoundError, match="Claude Code CLI not found"):
        ClaudeCodeProvider(config)


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_provider_init_success(mock_which):
    config = _default_config()
    provider = ClaudeCodeProvider(config)
    assert provider._resolved_path == "/usr/local/bin/claude"


# --- Command building tests ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_builds_command(mock_which):
    config = _default_config(
        allowed_tools=["Read", "Glob", "Grep"],
        dangerously_skip_permissions=False,
        permission_mode="acceptEdits",
        max_turns=10,
        max_budget_usd=2.5,
    )
    provider = ClaudeCodeProvider(config)
    cmd = provider._build_command("hello world", system_prompt="Be helpful.")

    assert cmd[0] == "/usr/local/bin/claude"
    assert "-p" in cmd
    assert "hello world" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--system-prompt" in cmd
    assert "Be helpful." in cmd
    assert "--allowedTools" in cmd
    assert "Read,Glob,Grep" in cmd
    assert "--dangerously-skip-permissions" not in cmd
    assert "--permission-mode" in cmd
    assert "acceptEdits" in cmd
    assert "--max-turns" in cmd
    assert "10" in cmd
    assert "--max-budget-usd" in cmd
    assert "2.5" in cmd


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_builds_command_minimal(mock_which):
    config = _default_config(
        allowed_tools=[],
        dangerously_skip_permissions=False,
        permission_mode=None,
        max_turns=None,
        max_budget_usd=None,
    )
    provider = ClaudeCodeProvider(config)
    cmd = provider._build_command("test")

    assert "-p" in cmd
    assert "--system-prompt" not in cmd
    assert "--permission-mode" not in cmd
    assert "--max-turns" not in cmd
    assert "--max-budget-usd" not in cmd
    assert "--dangerously-skip-permissions" not in cmd


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_builds_command_dangerously_skip_permissions(mock_which):
    config = _default_config(
        dangerously_skip_permissions=True,
        permission_mode="acceptEdits",
    )
    provider = ClaudeCodeProvider(config)
    cmd = provider._build_command("test")

    assert "--dangerously-skip-permissions" in cmd
    # permission_mode should be skipped when dangerously_skip_permissions is set
    assert "--permission-mode" not in cmd


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_builds_subprocess_kwargs_with_cwd(mock_which):
    config = _default_config(working_directory="/home/workspace", env={"FOO": "bar"})
    provider = ClaudeCodeProvider(config)
    kwargs = provider._build_subprocess_kwargs()

    assert kwargs["cwd"] == "/home/workspace"
    assert "FOO" in kwargs["env"]
    assert kwargs["env"]["FOO"] == "bar"


# --- Stream parsing tests ---


async def _run_with_stream_lines(provider, lines: bytes, stream_callback=None):
    """Helper: mock subprocess with given stdout lines and run stream_execute()."""

    async def _fake_create_subprocess(*args, **kwargs):
        proc = AsyncMock()
        proc.returncode = 0
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock()
        proc.kill = MagicMock()

        # Make stdout iterable line-by-line
        stdout_lines = lines.split(b"\n")

        async def _aiter():
            for line in stdout_lines:
                if line.strip():
                    yield line + b"\n"

        proc.stdout = _aiter()
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_create_subprocess):
        return await provider.stream_execute(
            system_prompt="test system",
            messages=[{"role": "user", "content": "test"}],
            stream_callback=stream_callback,
        )


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_stream_parsing_text(mock_which):
    provider = ClaudeCodeProvider(_default_config())
    events = []

    lines = _make_stream_lines(
        {"type": "system", "subtype": "init", "session_id": "abc-123"},
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello, world!"}],
            },
        },
        {
            "type": "result",
            "result": "Hello, world!",
            "total_input_tokens": 100,
            "total_output_tokens": 50,
        },
    )

    async def callback(event):
        events.append(event)

    result = await _run_with_stream_lines(provider, lines, callback)

    assert result.response_text == "Hello, world!"
    assert result.total_input_tokens == 100
    assert result.total_output_tokens == 50
    assert result.stopped_reason == "complete"

    text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_events) >= 1
    # StreamEndEvent is emitted by the Agent after stream_execute returns,
    # not by the adapter itself.


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_stream_parsing_tool_events(mock_which):
    provider = ClaudeCodeProvider(_default_config())
    events = []

    lines = _make_stream_lines(
        {"type": "system", "subtype": "init", "session_id": "abc-123"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Read",
                        "input": {"file_path": "/tmp/test.txt"},
                    },
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "file contents here",
            "is_error": False,
        },
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "The file contains: file contents here"}],
            },
        },
        {
            "type": "result",
            "result": "The file contains: file contents here",
            "total_input_tokens": 200,
            "total_output_tokens": 30,
        },
    )

    async def callback(event):
        events.append(event)

    result = await _run_with_stream_lines(provider, lines, callback)

    assert result.response_text == "The file contains: file contents here"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "Read"
    assert result.tool_calls[0].success is True

    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_name == "Read"
    assert len(tool_results) == 1
    assert tool_results[0].success is True


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_malformed_json_skipped(mock_which):
    provider = ClaudeCodeProvider(_default_config())

    lines = b"not valid json\n" + _make_stream_lines(
        {"type": "result", "result": "ok", "total_input_tokens": 10, "total_output_tokens": 5},
    )

    result = await _run_with_stream_lines(provider, lines)
    assert result.response_text == "ok"
    assert result.stopped_reason == "complete"


# --- Error handling tests ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_nonzero_exit(mock_which):
    provider = ClaudeCodeProvider(_default_config())

    async def _fake_create_subprocess(*args, **kwargs):
        proc = AsyncMock()
        proc.returncode = 1
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"authentication failed")
        proc.wait = AsyncMock()
        proc.kill = MagicMock()

        async def _empty():
            return
            yield  # noqa: E275 — make it an async generator

        proc.stdout = _empty()
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_create_subprocess):
        result = await provider.stream_execute(system_prompt=None, messages=[{"role": "user", "content": "test"}])

    assert result.stopped_reason == "error"
    assert "authentication failed" in result.response_text


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_subprocess_timeout(mock_which):
    config = _default_config(timeout_seconds=10)
    provider = ClaudeCodeProvider(config)
    provider._timeout = 1  # Override for fast test

    async def _fake_create_subprocess(*args, **kwargs):
        proc = AsyncMock()
        proc.returncode = None
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock()
        proc.kill = MagicMock()

        async def _hang():
            await asyncio.sleep(100)
            return
            yield  # noqa: E275 — make it an async generator

        proc.stdout = _hang()
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_create_subprocess):
        result = await provider.stream_execute(system_prompt=None, messages=[{"role": "user", "content": "test"}])

    assert result.stopped_reason == "error"
    assert "timed out" in result.response_text


# --- Concurrency tests ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_semaphore_limits_concurrent(mock_which):
    config = _default_config(max_concurrent=1, timeout_seconds=10)
    provider = ClaudeCodeProvider(config)

    call_count = 0
    call_order = []

    async def _fake_create_subprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        current = call_count
        call_order.append(f"start_{current}")

        proc = AsyncMock()
        proc.returncode = 0
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock()
        proc.kill = MagicMock()

        lines = _make_stream_lines(
            {
                "type": "result",
                "result": f"response_{current}",
                "total_input_tokens": 1,
                "total_output_tokens": 1,
            },
        )

        async def _aiter():
            await asyncio.sleep(0.1)  # Small delay to test semaphore
            for line in lines.split(b"\n"):
                if line.strip():
                    yield line + b"\n"
            call_order.append(f"end_{current}")

        proc.stdout = _aiter()
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_create_subprocess):
        results = await asyncio.gather(
            provider.stream_execute(system_prompt=None, messages=[{"role": "user", "content": "req1"}]),
            provider.stream_execute(system_prompt=None, messages=[{"role": "user", "content": "req2"}]),
        )

    assert len(results) == 2
    # With semaphore=1, second request must wait for first
    # Both should complete successfully
    assert all(r.stopped_reason == "complete" for r in results)


# --- Shutdown tests ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_shutdown_kills_active(mock_which):
    provider = ClaudeCodeProvider(_default_config())

    # Simulate an active process
    mock_proc = MagicMock()
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()
    provider._active_processes.append(mock_proc)

    await provider.shutdown()

    mock_proc.kill.assert_called_once()
    assert len(provider._active_processes) == 0


# --- LoopResult integration tests ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
async def test_loop_result_fully_populated(mock_which):
    provider = ClaudeCodeProvider(_default_config())

    lines = _make_stream_lines(
        {"type": "system", "subtype": "init", "session_id": "sess-1"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "file1.txt\nfile2.txt",
            "is_error": False,
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "Read",
                        "input": {"file_path": "file1.txt"},
                    },
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "t2",
            "content": "hello world",
            "is_error": False,
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Found 2 files."}]},
        },
        {
            "type": "result",
            "result": "Found 2 files.",
            "total_input_tokens": 500,
            "total_output_tokens": 100,
            "total_cache_read_tokens": 200,
        },
    )

    result = await _run_with_stream_lines(provider, lines)

    assert result.response_text == "Found 2 files."
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "Bash"
    assert result.tool_calls[1].tool_name == "Read"
    assert result.total_input_tokens == 500
    assert result.total_output_tokens == 100
    assert result.cached_tokens == 200
    assert result.iterations >= 1
    assert result.stopped_reason == "complete"
