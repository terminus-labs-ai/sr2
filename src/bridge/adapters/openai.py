"""OpenAI Chat Completions adapter for the SR2 Bridge."""

from __future__ import annotations

import json
import logging

from sr2.compaction.engine import ConversationTurn

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# Tool name classification (mirrors Anthropic adapter logic).
_TOOL_CONTENT_TYPE_MAP = {
    "read": "file_content",
    "cat": "file_content",
    "view_file": "file_content",
    "bash": "code_execution",
    "shell": "code_execution",
    "execute": "code_execution",
    "run": "code_execution",
    "terminal": "code_execution",
}


def _classify_tool_name(
    tool_name: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Classify a tool name into a content_type for compaction rules."""
    lower = tool_name.lower()
    if overrides:
        for marker, content_type in overrides.items():
            if marker.lower() in lower:
                return content_type
    for marker, content_type in _TOOL_CONTENT_TYPE_MAP.items():
        if marker in lower:
            return content_type
    return "tool_output"


class OpenAIAdapter:
    """Implements BridgeAdapter for the OpenAI Chat Completions API.

    Handles:
    - System prompt as a message with role "system" or "developer"
    - Messages with role/content (text strings or content-part arrays)
    - Tool calls via assistant tool_calls + tool role messages
    - SSE parsing for chat.completion.chunk with delta.content
    """

    def __init__(self, tool_type_overrides: dict[str, str] | None = None) -> None:
        self._tool_type_overrides = tool_type_overrides or {}

    def extract_messages(self, body: dict) -> tuple[str | None, list[dict]]:
        """Extract system prompt and non-system messages from OpenAI request body.

        In OpenAI format, the system prompt is a message with role "system"
        or "developer". We pull those out and concatenate them.
        """
        messages = body.get("messages", [])
        system_parts: list[str] = []
        non_system: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            if role in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Content-part array: [{"type": "text", "text": "..."}]
                    text = "\n".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                else:
                    text = str(content)
                if text:
                    system_parts.append(text)
            else:
                non_system.append(msg)

        system = "\n\n".join(system_parts) if system_parts else None
        return system, non_system

    def rebuild_body(
        self,
        original_body: dict,
        optimized_messages: list[dict],
        system_injection: str | None,
    ) -> dict:
        """Rebuild OpenAI request body with optimized messages.

        Preserves model, temperature, etc. Re-inserts system prompt as a
        system-role message at the front.
        """
        body = dict(original_body)

        # Collect original system messages to reconstruct
        original_system_parts: list[str] = []
        for msg in body.get("messages", []):
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = "\n".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                else:
                    text = str(content)
                if text:
                    original_system_parts.append(text)

        # Build final messages list
        final_messages: list[dict] = []

        # Prepend system message(s)
        if system_injection and original_system_parts:
            combined = system_injection + "\n\n" + "\n\n".join(original_system_parts)
            final_messages.append({"role": "system", "content": combined})
        elif system_injection:
            final_messages.append({"role": "system", "content": system_injection})
        elif original_system_parts:
            for part in original_system_parts:
                final_messages.append({"role": "system", "content": part})

        final_messages.extend(optimized_messages)
        body["messages"] = final_messages
        return body

    def parse_sse_text(self, chunk: bytes) -> str | None:
        """Extract text from an OpenAI SSE chunk.

        OpenAI streams events like:
            data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}
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

        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        return delta.get("content")

    def messages_to_turns(
        self,
        messages: list[dict],
        turn_counter_start: int,
    ) -> list[ConversationTurn]:
        """Convert OpenAI wire-format messages to ConversationTurns."""
        turns = []
        counter = turn_counter_start

        # Build tool_call_id -> function name map
        tool_name_map: dict[str, str] = {}
        for msg in messages:
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id", "")
                fn_name = tc.get("function", {}).get("name", "")
                if tc_id and fn_name:
                    tool_name_map[tc_id] = fn_name

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            # Determine content_type
            content_type: str | None = None

            if role == "assistant" and tool_calls:
                # Assistant message with tool calls
                parts: list[str] = []
                if content:
                    parts.append(str(content))
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "?")
                    fn_args = fn.get("arguments", "")
                    parts.append(f"[tool_call: {fn_name}({_truncate(fn_args, 200)})]")
                content_str = "\n".join(parts)
                # Classify based on first tool call
                first_name = tool_calls[0].get("function", {}).get("name", "")
                content_type = _classify_tool_name(first_name, self._tool_type_overrides)
            elif role == "tool":
                # Tool result message
                fn_name = tool_name_map.get(tool_call_id or "", "tool")
                content_str = f"[tool_result: {fn_name}]\n{content}"
                content_type = _classify_tool_name(fn_name, self._tool_type_overrides)
            elif isinstance(content, list):
                # Content-part array (e.g., vision)
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    else:
                        text_parts.append(str(part))
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content)

            # Extract tool name for metadata
            tool_name = None
            if tool_calls:
                tool_name = tool_calls[0].get("function", {}).get("name")
            elif role == "tool" and tool_call_id:
                tool_name = tool_name_map.get(tool_call_id)

            meta: dict = {"_original_message": msg}
            if tool_name:
                meta["tool_name"] = tool_name

            turn = ConversationTurn(
                turn_number=counter,
                role=role,
                content=content_str,
                content_type=content_type,
                metadata=meta,
            )
            counter += 1
            turns.append(turn)

        return turns

    def turns_to_messages(
        self,
        turns: list[ConversationTurn],
        original_messages: list[dict],
    ) -> list[dict]:
        """Convert ConversationTurns back to OpenAI wire-format messages.

        Compacted turns become plain text. Raw turns preserve original structure.
        Validates tool_call/tool pairing to avoid orphaned tool results.
        """
        messages = []

        # Track live tool_call IDs (present as actual tool_calls, not compacted)
        live_tool_call_ids: set[str] = set()

        for turn in turns:
            if turn.compacted:
                # Compacted content — flatten to plain text
                # Tool role can't exist without a tool_call_id reference,
                # so compacted tool turns become assistant messages
                out_role = "assistant" if turn.role == "tool" else turn.role
                messages.append({"role": out_role, "content": turn.content})
            else:
                original = turn.metadata.get("_original_message") if turn.metadata else None
                if not original:
                    messages.append({"role": turn.role, "content": turn.content})
                    continue

                # Check for orphaned tool results
                if original.get("role") == "tool":
                    ref_id = original.get("tool_call_id", "")
                    if ref_id and ref_id not in live_tool_call_ids:
                        # Orphaned — flatten
                        messages.append({"role": "assistant", "content": turn.content})
                        logger.debug(
                            "Flattened turn %d: orphaned tool result (tool_call was compacted)",
                            turn.turn_number,
                        )
                        continue

                # Register tool_call IDs from assistant messages
                for tc in original.get("tool_calls", []):
                    tc_id = tc.get("id", "")
                    if tc_id:
                        live_tool_call_ids.add(tc_id)

                messages.append(original)

        return messages
