"""Anthropic Messages API adapter for the SR2 Bridge."""

from __future__ import annotations

import json
import logging

from sr2.compaction.engine import ConversationTurn

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _detect_content_type(content_blocks: list) -> str | None:
    """Detect the dominant content type from Anthropic content blocks."""
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("tool_result", "tool_use"):
            return "tool_output"
    return None


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

    def messages_to_turns(
        self,
        messages: list[dict],
        turn_counter_start: int,
    ) -> list[ConversationTurn]:
        """Convert Anthropic wire-format messages to ConversationTurns."""
        turns = []
        counter = turn_counter_start
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle Anthropic content blocks
            if isinstance(content, list):
                content_type = _detect_content_type(content)
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            text_parts.append(
                                f"[tool_use: {block.get('name', '?')}"
                                f"({_truncate(str(block.get('input', '')), 200)})]"
                            )
                        elif block.get("type") == "tool_result":
                            result_content = block.get("content", "")
                            if isinstance(result_content, list):
                                result_content = " ".join(
                                    b.get("text", "")
                                    for b in result_content
                                    if isinstance(b, dict)
                                )
                            text_parts.append(f"[tool_result]\n{result_content}")
                        else:
                            text_parts.append(str(block))
                    else:
                        text_parts.append(str(block))
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content)
                content_type = None

            turn = ConversationTurn(
                turn_number=counter,
                role=role,
                content=content_str,
                content_type=content_type,
                metadata={"_original_message": msg},
            )
            counter += 1
            turns.append(turn)

        return turns

    def turns_to_messages(
        self,
        turns: list[ConversationTurn],
        original_messages: list[dict],
    ) -> list[dict]:
        """Convert ConversationTurns back to Anthropic wire-format messages.

        For compacted turns, emits plain text content. For raw turns that
        still have their original structure, preserves it.
        """
        messages = []
        for turn in turns:
            if turn.compacted:
                messages.append({"role": turn.role, "content": turn.content})
            else:
                original = turn.metadata.get("_original_message") if turn.metadata else None
                if original:
                    messages.append(original)
                else:
                    messages.append({"role": turn.role, "content": turn.content})
        return messages
