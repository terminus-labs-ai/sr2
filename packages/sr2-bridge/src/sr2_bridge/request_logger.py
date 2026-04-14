"""JSONL request/response logger for the SR2 Bridge.

Writes structured log entries to a dedicated file via an async queue
so logging never blocks request handling. Each line is a self-contained
JSON object with a ``type`` discriminator field.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import IO

from sr2_bridge.config import BridgeLoggingConfig

logger = logging.getLogger(__name__)


def _message_summary(msg: dict) -> dict:
    """Build a compact summary of a single message (role, content length, tool names)."""
    role = msg.get("role", "?")
    content = msg.get("content", "")

    if isinstance(content, str):
        return {"role": role, "content_length": len(content)}

    if isinstance(content, list):
        total_length = 0
        tool_names: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                total_length += len(block.get("text", ""))
            elif btype == "tool_use":
                tool_names.append(block.get("name", "?"))
                total_length += len(str(block.get("input", "")))
            elif btype == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    total_length += sum(
                        len(b.get("text", "")) for b in rc if isinstance(b, dict)
                    )
                else:
                    total_length += len(str(rc))
        summary: dict = {"role": role, "content_length": total_length}
        if tool_names:
            summary["tool_names"] = tool_names
        return summary

    return {"role": role, "content_length": len(str(content))}


def _truncate(text: str | None, max_len: int | None) -> str | None:
    """Truncate text to max_len if configured."""
    if text is None or max_len is None:
        return text
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [truncated, {len(text)} total]"


class BridgeRequestLogger:
    """Async JSONL logger for bridge request/response payloads."""

    def __init__(self, config: BridgeLoggingConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=1000)
        self._writer_task: asyncio.Task | None = None
        self._file: IO[str] | None = None
        self._drop_count: int = 0

    async def start(self) -> None:
        """Open output file and start background writer task."""
        self._file = open(self._config.output_path, "a")  # noqa: SIM115
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info("Request logger started: %s", self._config.output_path)

    async def stop(self) -> None:
        """Signal writer to drain and close."""
        if self._writer_task:
            # Sentinel to stop the writer
            await self._queue.put(None)
            await self._writer_task
            self._writer_task = None
        if self._file:
            self._file.close()
            self._file = None
        if self._drop_count > 0:
            logger.warning("Request logger dropped %d entries due to backpressure", self._drop_count)

    def log_incoming(
        self,
        session_id: str,
        system: str | None,
        messages: list[dict],
        body: dict,
    ) -> None:
        """Log the incoming request (original system prompt + messages)."""
        max_len = self._config.max_content_length
        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "incoming",
            "session": session_id,
            "model": body.get("model", ""),
            "message_count": len(messages),
            "message_summaries": [_message_summary(m) for m in messages],
        }
        if self._config.log_system_prompt and system is not None:
            entry["system_prompt"] = _truncate(system, max_len)
            entry["system_prompt_length"] = len(system)
        if self._config.log_messages:
            entry["messages"] = [
                {
                    **_message_summary(m),
                    "content": _truncate(
                        m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", "")),
                        max_len,
                    ),
                }
                for m in messages
            ]
        self._enqueue(entry)

    def log_transformed_system(
        self,
        session_id: str,
        original: str | None,
        transformed: str | None,
    ) -> None:
        """Log the system prompt before and after transform."""
        max_len = self._config.max_content_length
        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "system_transform",
            "session": session_id,
        }
        if original is not None:
            entry["original_length"] = len(original)
            if self._config.log_system_prompt:
                entry["original"] = _truncate(original, max_len)
        if transformed is not None:
            entry["transformed_length"] = len(transformed)
            if self._config.log_system_prompt:
                entry["transformed"] = _truncate(transformed, max_len)
        self._enqueue(entry)

    def log_outgoing(self, session_id: str, rebuilt_body: dict) -> None:
        """Log the final body sent upstream."""
        if not self._config.log_rebuilt_body:
            return

        max_len = self._config.max_content_length
        system = rebuilt_body.get("system", "")
        if isinstance(system, list):
            system_text = "\n".join(
                b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            system_text = str(system) if system else ""

        messages = rebuilt_body.get("messages", [])
        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "outgoing",
            "session": session_id,
            "model": rebuilt_body.get("model", ""),
            "system_length": len(system_text),
            "message_count": len(messages),
            "message_summaries": [_message_summary(m) for m in messages],
        }
        if self._config.log_system_prompt:
            entry["system"] = _truncate(system_text, max_len)
        if self._config.log_messages:
            entry["messages"] = [
                {
                    **_message_summary(m),
                    "content": _truncate(
                        m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", "")),
                        max_len,
                    ),
                }
                for m in messages
            ]
        self._enqueue(entry)

    def _enqueue(self, entry: dict) -> None:
        """Put entry on the queue, dropping if full."""
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            self._drop_count += 1
            if self._drop_count % 100 == 1:
                logger.warning(
                    "Request logger queue full, dropping entries (total dropped: %d)",
                    self._drop_count,
                )

    async def _writer_loop(self) -> None:
        """Background task: dequeue entries and write as JSONL lines."""
        while True:
            entry = await self._queue.get()
            if entry is None:
                # Drain remaining entries before stopping
                while not self._queue.empty():
                    remaining = self._queue.get_nowait()
                    if remaining is not None and self._file:
                        self._file.write(json.dumps(remaining, default=str) + "\n")
                if self._file:
                    self._file.flush()
                break
            if self._file:
                try:
                    self._file.write(json.dumps(entry, default=str) + "\n")
                    self._file.flush()
                except Exception:
                    logger.warning("Request logger write failed", exc_info=True)
