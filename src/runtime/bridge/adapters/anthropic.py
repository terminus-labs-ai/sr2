"""Anthropic Messages API adapter for the SR2 Bridge."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class AnthropicAdapter:
    """Implements BridgeAdapter for the Anthropic Messages API (/v1/messages).

    Handles:
    - System prompt as a top-level 'system' field (string or content-block list)
    - Messages with role/content (text, tool_use, tool_result content blocks)
    - SSE event parsing for content_block_delta with text_delta
    """

    def extract_messages(self, body: dict) -> tuple[str | None, list[dict]]:
        """Extract system prompt and messages from Anthropic request body."""
        system = body.get("system")
        if isinstance(system, list):
            # Content-block list format: [{"type": "text", "text": "..."}]
            system = "\n".join(
                block.get("text", "") for block in system if block.get("type") == "text"
            )
        elif system is not None:
            system = str(system)

        messages = body.get("messages", [])
        return system, messages

    def rebuild_body(
        self,
        original_body: dict,
        optimized_messages: list[dict],
        system_injection: str | None,
    ) -> dict:
        """Rebuild Anthropic request body with optimized messages."""
        body = dict(original_body)
        body["messages"] = optimized_messages

        if system_injection:
            existing_system = body.get("system")
            if isinstance(existing_system, list):
                # Prepend injection as a text block
                injection_block = {"type": "text", "text": system_injection}
                body["system"] = [injection_block] + existing_system
            elif existing_system:
                body["system"] = f"{system_injection}\n\n{existing_system}"
            else:
                body["system"] = system_injection

        return body

    def parse_sse_text(self, chunk: bytes) -> str | None:
        """Extract text from an Anthropic SSE chunk.

        Anthropic streams events like:
            event: content_block_delta
            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
        """
        try:
            line = chunk.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

        if not line.startswith("data: "):
            return None

        json_str = line[6:]
        if json_str == "[DONE]":
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if data.get("type") != "content_block_delta":
            return None

        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text")

        return None
