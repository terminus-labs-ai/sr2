"""Claude Code CLI bridge adapter — spawns ``claude -p`` with full SR2 context.

The adapter receives SR2's compiled context (system prompt + optimized
conversation history) from the bridge engine and passes it to Claude Code:

- System prompt via ``--system-prompt`` flag
- Full conversation history serialized as the prompt string to ``-p``
- Streaming via ``--output-format stream-json``
- Auth via ``CLAUDE_CODE_OAUTH_TOKEN`` env var (no ``--bare`` by default)

Claude Code runs its own agentic loop internally; SR2 retains ownership of
context engineering (compaction, summarization, memory).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2_bridge.adapters.claude_code_config import ClaudeCodeAdapterConfig

from sr2_runtime.llm.loop import LoopResult, ToolCallRecord

logger = logging.getLogger(__name__)


@dataclass
class _StreamResult:
    """Per-call accumulation state.  No instance variables — fixes concurrency bug."""

    full_text: str = ""
    last_emitted_text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    pending_tools: dict[str, dict] = field(default_factory=dict)
    current_iteration: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class ClaudeCodeAdapter:
    """Bridge adapter that spawns ``claude -p`` with SR2-compiled context.

    Implements the ``ExecutionAdapter`` protocol: receives the full optimized
    context from :meth:`BridgeEngine.optimize` and streams results back as a
    :class:`LoopResult`.
    """

    def __init__(self, config: ClaudeCodeAdapterConfig) -> None:
        self._path = config.path
        self._allowed_tools = config.allowed_tools
        self._bare = config.bare
        self._dangerously_skip_permissions = config.dangerously_skip_permissions
        self._permission_mode = config.permission_mode
        self._max_turns = config.max_turns
        self._max_budget_usd = config.max_budget_usd
        self._timeout = config.timeout_seconds
        self._working_directory = config.working_directory
        self._env = config.env
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._active_processes: list[asyncio.subprocess.Process] = []

        # Verify claude binary exists at init time
        resolved = shutil.which(self._path)
        if not resolved:
            raise FileNotFoundError(
                f"Claude Code CLI not found at '{self._path}'. "
                "Install with: curl -fsSL https://claude.ai/install.sh | bash"
            )
        self._resolved_path = resolved
        logger.info("Claude Code adapter initialized: %s", self._resolved_path)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Build the ``claude`` CLI command from config and arguments."""
        cmd = [
            self._resolved_path,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
        ]

        if self._bare:
            cmd.append("--bare")

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if self._allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._allowed_tools)])
        if self._dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        elif self._permission_mode:
            cmd.extend(["--permission-mode", self._permission_mode])
        if self._max_turns is not None:
            cmd.extend(["--max-turns", str(self._max_turns)])
        if self._max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self._max_budget_usd)])

        return cmd

    def _build_subprocess_kwargs(self) -> dict:
        """Build kwargs for :func:`asyncio.create_subprocess_exec`."""
        kwargs: dict = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if self._working_directory:
            kwargs["cwd"] = self._working_directory
        if self._env:
            kwargs["env"] = {**os.environ, **self._env}
        return kwargs

    # ------------------------------------------------------------------
    # Context serialization
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_conversation(
        system_prompt: str | None,
        messages: list[dict],
    ) -> tuple[str | None, str]:
        """Extract the latest user message as the prompt.

        SR2's pipeline compiles all context (memories, session summaries,
        retrieved knowledge, tool schemas) into the system prompt passed
        via ``--system-prompt``.  The ``-p`` prompt only needs the current
        user message — SR2 is the sole context provider.

        Returns:
            (system_prompt, prompt) tuple ready for ``_build_command``.
        """
        last_user_message = ""
        for msg in reversed(messages):
            role = msg.get("role", "user")
            if role == "user":
                content = msg.get("content", "")
                # Handle Anthropic content-block lists
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    content = "\n".join(text_parts)
                last_user_message = content
                break

        return system_prompt, last_user_message

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def stream_execute(
        self,
        system_prompt: str | None,
        messages: list[dict],
        stream_callback=None,
    ) -> LoopResult:
        """Execute Claude Code with the full SR2-compiled context.

        This is the ``ExecutionAdapter`` entry point.  The bridge engine
        calls this after ``optimize()`` with the optimized system prompt
        and conversation history.

        Args:
            system_prompt: The compiled system prompt from SR2's pipeline.
            messages: The optimized conversation messages.
            stream_callback: Optional async callback for real-time streaming
                events (TextDeltaEvent, ToolStartEvent, ToolResultEvent, etc.).

        Returns:
            A fully populated :class:`LoopResult`.
        """
        sys_prompt, prompt = self.serialize_conversation(system_prompt, messages)
        async with self._semaphore:
            return await self._run_subprocess(
                prompt=prompt, system_prompt=sys_prompt, stream_callback=stream_callback,
            )

    async def _run_subprocess(
        self,
        prompt: str,
        system_prompt: str | None,
        stream_callback=None,
    ) -> LoopResult:
        """Spawn and manage a single ``claude`` subprocess."""
        cmd = self._build_command(prompt, system_prompt)
        kwargs = self._build_subprocess_kwargs()

        logger.info("Spawning Claude Code: %s -p '<prompt>' ...", cmd[0])
        logger.debug("Full command: %s", cmd)

        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        self._active_processes.append(proc)

        # Per-call state — no instance variables (concurrency-safe)
        acc = _StreamResult()

        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            try:
                await asyncio.wait_for(
                    self._consume_stream(proc=proc, acc=acc, stream_callback=stream_callback),
                    timeout=self._timeout,
                )
            except TimeoutError:
                logger.error(
                    "Claude Code subprocess timed out after %ds, killing",
                    self._timeout,
                )
                proc.kill()
                await proc.wait()
                stderr_out = await proc.stderr.read()
                if stderr_out:
                    logger.warning(
                        "Claude Code stderr on timeout: %s",
                        stderr_out.decode()[:500],
                    )
                return LoopResult(
                    response_text=f"Error: Claude Code timed out after {self._timeout} seconds.",
                    stopped_reason="error",
                )
            except asyncio.CancelledError:
                logger.info(
                    "Claude Code subprocess cancelled (client disconnect), killing PID %s",
                    proc.pid,
                )
                proc.kill()
                await proc.wait()
                raise

            await proc.wait()

            if proc.returncode != 0:
                stderr_out = await proc.stderr.read()
                stderr_text = (
                    stderr_out.decode(errors="replace")[:1000] if stderr_out else ""
                )
                logger.error(
                    "Claude Code exited with code %d: %s",
                    proc.returncode,
                    stderr_text,
                )
                return LoopResult(
                    response_text=(
                        f"Error: Claude Code exited with code {proc.returncode}. "
                        f"{stderr_text}".strip()
                    ),
                    stopped_reason="error",
                )

            # Read remaining stderr
            stderr_out = await proc.stderr.read()
            if stderr_out:
                stderr_text = stderr_out.decode(errors="replace").strip()
                if stderr_text:
                    logger.debug("Claude Code stderr: %s", stderr_text[:500])

        finally:
            if proc in self._active_processes:
                self._active_processes.remove(proc)

        result = LoopResult(
            response_text=acc.full_text,
            tool_calls=acc.tool_calls,
            iterations=max(acc.current_iteration, 1),
            total_input_tokens=acc.input_tokens,
            total_output_tokens=acc.output_tokens,
            cached_tokens=acc.cached_tokens,
            stopped_reason="complete",
        )

        logger.info(
            "Claude Code complete: %d chars, %d tool calls, %d iterations, %d total tokens",
            len(acc.full_text),
            len(acc.tool_calls),
            acc.current_iteration,
            acc.input_tokens + acc.output_tokens,
        )

        return result

    # ------------------------------------------------------------------
    # Stream parsing
    # ------------------------------------------------------------------

    async def _consume_stream(
        self,
        proc: asyncio.subprocess.Process,
        acc: _StreamResult,
        stream_callback=None,
    ) -> None:
        """Read and parse ``stream-json`` lines from stdout.

        All accumulation state lives in the ``acc`` dataclass — no instance
        variables are touched, making concurrent calls safe.

        When *stream_callback* is provided, streaming events are emitted
        in real-time so callers (HTTP SSE, Telegram) can display progress
        as Claude Code works through its agentic loop.
        """
        from sr2_runtime.llm.streaming import (
            TextDeltaEvent,
            ToolStartEvent,
            ToolResultEvent,
        )

        assert proc.stdout is not None

        async def _emit(event):
            if stream_callback is not None:
                try:
                    await stream_callback(event)
                except Exception:
                    logger.debug("Stream callback error (non-fatal)", exc_info=True)

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed JSON from Claude Code, skipping: %s", line[:200]
                )
                continue

            event_type = event.get("type", "")

            # System init
            if event_type == "system" and event.get("subtype") == "init":
                logger.info(
                    "Claude Code session: %s", event.get("session_id", "unknown")
                )
                continue

            # Assistant message (text + tool_use blocks)
            if event_type == "assistant":
                acc.current_iteration += 1
                msg = event.get("message", {})
                content_blocks = msg.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            acc.full_text = text
                            # Emit only the NEW text (delta), not the full cumulative text
                            prev = getattr(acc, "last_emitted_text", "")
                            if text != prev:
                                delta = text[len(prev):] if text.startswith(prev) else text
                                if delta:
                                    await _emit(TextDeltaEvent(content=delta))
                                    acc.last_emitted_text = text
                    elif block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        acc.pending_tools[tool_id] = {
                            "name": tool_name,
                            "arguments": tool_input,
                            "start_time": time.perf_counter(),
                            "iteration": acc.current_iteration,
                        }
                        await _emit(ToolStartEvent(
                            tool_name=tool_name,
                            tool_call_id=tool_id,
                            arguments=tool_input,
                        ))
                continue

            # Tool result
            if event_type == "tool_result" or (
                event_type == "assistant"
                and event.get("message", {}).get("role") == "tool"
            ):
                tool_id = event.get("tool_use_id", "")
                content = event.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "")
                        for c in content
                        if c.get("type") == "text"
                    )
                is_error = event.get("is_error", False)

                pending = acc.pending_tools.pop(tool_id, None)
                if pending:
                    duration = (
                        time.perf_counter() - pending["start_time"]
                    ) * 1000
                    acc.tool_calls.append(
                        ToolCallRecord(
                            tool_name=pending["name"],
                            arguments=pending["arguments"],
                            result=str(content)[:5000],
                            duration_ms=duration,
                            success=not is_error,
                            error=str(content)[:1000] if is_error else None,
                            call_id=tool_id,
                            iteration=pending["iteration"],
                        )
                    )
                    await _emit(ToolResultEvent(
                        tool_name=pending["name"],
                        tool_call_id=tool_id,
                        result=str(content)[:500],
                        success=not is_error,
                    ))
                continue

            # Result event — final summary with token counts
            if event_type == "result":
                acc.full_text = event.get("result", acc.full_text)
                acc.input_tokens = event.get("total_input_tokens", 0)
                acc.output_tokens = event.get("total_output_tokens", 0)
                acc.cached_tokens = event.get("total_cache_read_tokens", 0)

                cost = event.get("total_cost_usd", 0)
                if cost:
                    logger.info("Claude Code cost: $%.4f", cost)
                continue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Kill all active Claude Code subprocesses."""
        for proc in list(self._active_processes):
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        self._active_processes.clear()
        logger.info("Claude Code adapter shut down")
