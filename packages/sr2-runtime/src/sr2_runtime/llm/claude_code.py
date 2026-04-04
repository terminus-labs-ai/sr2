"""Claude Code CLI provider — spawns `claude -p` as an LLM backend.

Claude Code handles its own tool execution (Bash, Edit, Read, MCPs, etc.)
internally. SR2's LLMLoop is bypassed; SR2 retains ownership of context
engineering (compaction, summarization, memory) and session management.

The provider runs Claude Code with ``--bare`` so that SR2 is the sole memory
system — no CLAUDE.md files, no auto-memory, no hooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING

from sr2_runtime.llm.loop import LoopResult, ToolCallRecord
from sr2_runtime.llm.streaming import (
    StreamCallback,
    StreamEndEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)

if TYPE_CHECKING:
    from sr2_runtime.config import ClaudeCodeConfig

logger = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Wraps the Claude Code CLI as an LLM provider.

    Instead of calling an API via LiteLLM, spawns ``claude --bare -p``
    with ``--output-format stream-json``. Claude Code runs its own agentic
    loop internally; SR2 provides context via ``--system-prompt``.
    """

    def __init__(self, config: ClaudeCodeConfig) -> None:
        self._path = config.path
        self._allowed_tools = config.allowed_tools
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
        logger.info(f"Claude Code provider initialized: {self._resolved_path}")

    def _build_command(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Build the claude CLI command from config and arguments."""
        cmd = [
            self._resolved_path,
            "--bare",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if self._allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._allowed_tools)])
        if self._permission_mode:
            cmd.extend(["--permission-mode", self._permission_mode])
        if self._max_turns is not None:
            cmd.extend(["--max-turns", str(self._max_turns)])
        if self._max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self._max_budget_usd)])

        return cmd

    def _build_subprocess_kwargs(self) -> dict:
        """Build kwargs for asyncio.create_subprocess_exec."""
        kwargs: dict = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if self._working_directory:
            kwargs["cwd"] = self._working_directory
        if self._env:
            kwargs["env"] = {**os.environ, **self._env}
        return kwargs

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LoopResult:
        """Run claude -p and return a fully populated LoopResult."""
        # Delegate to stream_complete without a callback
        return await self.stream_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            stream_callback=None,
        )

    async def stream_complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        stream_callback: StreamCallback | None = None,
        stream_tool_events: bool = True,
    ) -> LoopResult:
        """Run claude -p with stream-json, emit StreamEvents, return LoopResult.

        Parses stream-json events to populate a real LoopResult with tool call
        records, token counts, and iteration counts. Emits TextDeltaEvent,
        ToolStartEvent, and ToolResultEvent to the stream callback.
        """
        async with self._semaphore:
            return await self._run_subprocess(
                prompt=prompt,
                system_prompt=system_prompt,
                stream_callback=stream_callback,
                stream_tool_events=stream_tool_events,
            )

    async def _run_subprocess(
        self,
        prompt: str,
        system_prompt: str | None,
        stream_callback: StreamCallback | None,
        stream_tool_events: bool,
    ) -> LoopResult:
        """Internal: spawn and manage a single claude subprocess."""
        cmd = self._build_command(prompt, system_prompt)
        kwargs = self._build_subprocess_kwargs()

        logger.info(f"Spawning Claude Code: {cmd[0]} --bare -p '<prompt>' ...")
        logger.debug(f"Full command: {cmd}")

        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        self._active_processes.append(proc)

        result = LoopResult(response_text="")
        full_text = ""
        tool_calls: list[ToolCallRecord] = []
        # Track pending tool starts for pairing with results
        pending_tools: dict[str, dict] = {}
        iterations = 0
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            try:
                await asyncio.wait_for(
                    self._consume_stream(
                        proc=proc,
                        stream_callback=stream_callback,
                        stream_tool_events=stream_tool_events,
                        full_text_acc=[full_text],
                        tool_calls=tool_calls,
                        pending_tools=pending_tools,
                        token_acc={
                            "input": 0,
                            "output": 0,
                            "cached": 0,
                            "iterations": 0,
                        },
                    ),
                    timeout=self._timeout,
                )
                full_text = self._full_text
                input_tokens = self._token_acc["input"]
                output_tokens = self._token_acc["output"]
                cached_tokens = self._token_acc["cached"]
                iterations = self._token_acc["iterations"]
            except TimeoutError:
                logger.error(f"Claude Code subprocess timed out after {self._timeout}s, killing")
                proc.kill()
                await proc.wait()
                stderr_out = await proc.stderr.read()
                result.response_text = (
                    f"Error: Claude Code timed out after {self._timeout} seconds."
                )
                result.stopped_reason = "error"
                if stderr_out:
                    logger.warning(f"Claude Code stderr on timeout: {stderr_out.decode()[:500]}")
                return result

            # Wait for process to finish
            await proc.wait()

            # Check exit code
            if proc.returncode != 0:
                stderr_out = await proc.stderr.read()
                stderr_text = stderr_out.decode(errors="replace")[:1000] if stderr_out else ""
                logger.error(f"Claude Code exited with code {proc.returncode}: {stderr_text}")
                result.response_text = (
                    f"Error: Claude Code exited with code {proc.returncode}. {stderr_text}".strip()
                )
                result.stopped_reason = "error"
                return result

            # Read any remaining stderr
            stderr_out = await proc.stderr.read()
            if stderr_out:
                stderr_text = stderr_out.decode(errors="replace").strip()
                if stderr_text:
                    logger.debug(f"Claude Code stderr: {stderr_text[:500]}")

        finally:
            if proc in self._active_processes:
                self._active_processes.remove(proc)

        # Send final stream end event
        if stream_callback:
            await stream_callback(StreamEndEvent(full_text=full_text))

        result.response_text = full_text
        result.tool_calls = tool_calls
        result.iterations = max(iterations, 1)
        result.total_input_tokens = input_tokens
        result.total_output_tokens = output_tokens
        result.cached_tokens = cached_tokens
        result.stopped_reason = "complete"

        logger.info(
            f"Claude Code complete: {len(full_text)} chars, "
            f"{len(tool_calls)} tool calls, {iterations} iterations, "
            f"{input_tokens + output_tokens} total tokens"
        )

        return result

    async def _consume_stream(
        self,
        proc: asyncio.subprocess.Process,
        stream_callback: StreamCallback | None,
        stream_tool_events: bool,
        full_text_acc: list[str],
        tool_calls: list[ToolCallRecord],
        pending_tools: dict[str, dict],
        token_acc: dict[str, int],
    ) -> None:
        """Read and parse stream-json lines from stdout."""
        assert proc.stdout is not None

        self._full_text = ""
        self._token_acc = token_acc
        current_iteration = 0

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON from Claude Code, skipping: {line[:200]}")
                continue

            event_type = event.get("type", "")

            # System init — log session info
            if event_type == "system" and event.get("subtype") == "init":
                logger.info(f"Claude Code session: {event.get('session_id', 'unknown')}")
                continue

            # Text delta from streaming
            if event_type == "assistant":
                current_iteration += 1
                msg = event.get("message", {})
                content_blocks = msg.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            self._full_text = text
                            if stream_callback:
                                await stream_callback(TextDeltaEvent(content=text))
                    elif block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        pending_tools[tool_id] = {
                            "name": tool_name,
                            "arguments": tool_input,
                            "start_time": time.perf_counter(),
                            "iteration": current_iteration,
                        }
                        if stream_callback and stream_tool_events:
                            await stream_callback(
                                ToolStartEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tool_id,
                                    arguments=tool_input,
                                )
                            )
                continue

            # Tool result
            if event_type == "tool_result" or (
                event_type == "assistant" and event.get("message", {}).get("role") == "tool"
            ):
                tool_id = event.get("tool_use_id", "")
                content = event.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if c.get("type") == "text"
                    )
                is_error = event.get("is_error", False)

                pending = pending_tools.pop(tool_id, None)
                if pending:
                    duration = (time.perf_counter() - pending["start_time"]) * 1000
                    tool_calls.append(
                        ToolCallRecord(
                            tool_name=pending["name"],
                            arguments=pending["arguments"],
                            result=str(content)[:5000],
                            duration_ms=duration,
                            success=not is_error,
                            call_id=tool_id,
                            iteration=pending["iteration"],
                        )
                    )
                    if stream_callback and stream_tool_events:
                        await stream_callback(
                            ToolResultEvent(
                                tool_name=pending["name"],
                                tool_call_id=tool_id,
                                result=str(content)[:500],
                                success=not is_error,
                            )
                        )
                continue

            # Result event — final summary
            if event_type == "result":
                self._full_text = event.get("result", self._full_text)
                token_acc["input"] = event.get("total_input_tokens", 0)
                token_acc["output"] = event.get("total_output_tokens", 0)
                token_acc["cached"] = event.get("total_cache_read_tokens", 0)
                token_acc["iterations"] = current_iteration

                cost = event.get("total_cost_usd", 0)
                if cost:
                    logger.info(f"Claude Code cost: ${cost:.4f}")
                continue

    async def shutdown(self) -> None:
        """Kill all active Claude Code subprocesses."""
        for proc in list(self._active_processes):
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        self._active_processes.clear()
        logger.info("Claude Code provider shut down")
