"""OpenAI Chat Completions API adapter for the SR2 Bridge."""

from __future__ import annotations

import json
import logging

from sr2.compaction.engine import ConversationTurn

from sr2_bridge.adapters._utils import _classify_tool_name, _extract_file_path, _truncate

logger = logging.getLogger(__name__)


def _detect_content_type_openai(
    message: dict,
    tool_name_map: dict[str, str] | None = None,
    tool_type_overrides: dict[str, str] | None = None,
) -> str | None:
    """Detect the dominant content type from an OpenAI message.

    For assistant messages with tool_calls: classifies based on function name.
    For tool messages: looks up the tool_call_id in tool_name_map
    to find the originating function name, then classifies.
    """
    tool_name_map = tool_name_map or {}
    role = message.get("role", "")

    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            # Classify based on the first tool call's function name
            first_call = tool_calls[0]
            fn = first_call.get("function", {})
            return _classify_tool_name(fn.get("name", ""), tool_type_overrides)

    if role == "tool":
        tool_call_id = message.get("tool_call_id", "")
        if tool_call_id and tool_call_id in tool_name_map:
            return _classify_tool_name(tool_name_map[tool_call_id], tool_type_overrides)
        return "tool_output"

    return None


class OpenAIAdapter:
    """Implements BridgeAdapter for the OpenAI Chat Completions API (/v1/chat/completions).

    Handles:
    - System prompt as a message with role "system" or "developer"
    - Messages with role/content (text strings or multipart content arrays)
    - Tool calls via tool_calls array on assistant messages
    - Tool results via role="tool" messages with tool_call_id
    - SSE event parsing for choices[0].delta.content
    """

    def __init__(self, tool_type_overrides: dict[str, str] | None = None) -> None:
        self._tool_type_overrides = tool_type_overrides or {}

    def extract_messages(self, body: dict) -> tuple[str | None, list[dict]]:
        """Extract system prompt and messages from OpenAI request body.

        OpenAI puts the system prompt as a message with role "system" or "developer".
        We extract it and return the remaining messages separately.
        """
        messages = body.get("messages", [])
        system_parts: list[str] = []
        non_system: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            if role in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multipart content: [{"type": "text", "text": "..."}]
                    text = "\n".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
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

        Re-inserts the system prompt as a system message at the start of the
        messages list. If system_injection is provided, it is prepended to the
        existing system prompt content.
        """
        body = dict(original_body)

        # Reconstruct messages: system prompt(s) first, then optimized messages
        rebuilt_messages: list[dict] = []

        # Find original system/developer messages to preserve role and format
        original_messages = original_body.get("messages", [])
        system_role = "system"  # default
        original_system_content: str | None = None

        for msg in original_messages:
            role = msg.get("role", "")
            if role in ("system", "developer"):
                system_role = role
                content = msg.get("content", "")
                if isinstance(content, list):
                    original_system_content = "\n".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    original_system_content = str(content)
                break  # Use the first system message's role

        # Build the system message
        if system_injection and original_system_content:
            rebuilt_messages.append({
                "role": system_role,
                "content": f"{system_injection}\n\n{original_system_content}",
            })
        elif system_injection:
            rebuilt_messages.append({
                "role": system_role,
                "content": system_injection,
            })
        elif original_system_content:
            rebuilt_messages.append({
                "role": system_role,
                "content": original_system_content,
            })

        # Append the (already-filtered) optimized messages
        rebuilt_messages.extend(optimized_messages)
        body["messages"] = rebuilt_messages

        return body

    def parse_sse_text(self, chunk: bytes) -> str | None:
        """Extract text from an OpenAI SSE chunk.

        OpenAI streams events like:
            data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"Hello"}}]}
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

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            return None

        delta = choices[0].get("delta", {})
        content = delta.get("content")
        return content if content else None

    def messages_to_turns(
        self,
        messages: list[dict],
        turn_counter_start: int,
    ) -> list[ConversationTurn]:
        """Convert OpenAI wire-format messages to ConversationTurns."""
        turns = []
        counter = turn_counter_start

        # Build tool_call_id -> function_name and tool_call_id -> args maps
        # (assistant tool_calls precede tool messages that reference them)
        tool_name_map: dict[str, str] = {}
        tool_args_map: dict[str, dict] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        tc_id = tc.get("id", "")
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        if tc_id and fn_name:
                            tool_name_map[tc_id] = fn_name
                        fn_args_str = fn.get("arguments", "")
                        if tc_id and fn_args_str:
                            try:
                                tool_args_map[tc_id] = json.loads(fn_args_str)
                            except (json.JSONDecodeError, TypeError):
                                pass

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Determine text representation and content_type
            content_type = _detect_content_type_openai(
                msg, tool_name_map, self._tool_type_overrides
            )

            text_parts: list[str] = []

            # Handle content (can be string, list of parts, or None for tool-call-only messages)
            if isinstance(content, list):
                # Multipart content: [{"type": "text", "text": "..."}, ...]
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            text_parts.append("[image]")
                        else:
                            text_parts.append(str(part))
                    else:
                        text_parts.append(str(part))
            elif content is not None:
                text_parts.append(str(content))

            # Handle tool_calls on assistant messages
            tool_calls = msg.get("tool_calls")
            tool_name = None
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "?")
                    fn_args = fn.get("arguments", "")
                    text_parts.append(
                        f"[tool_call: {fn_name}({_truncate(fn_args, 200)})]"
                    )
                    if tool_name is None:
                        tool_name = fn_name

            # Handle tool role messages
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                resolved_name = tool_name_map.get(tool_call_id, "tool")
                tool_name = resolved_name
                tool_content = str(content) if content else ""
                # Rebuild text_parts to include tool context
                text_parts = [f"[tool_result: {resolved_name}]\n{tool_content}"]

            content_str = "\n".join(text_parts) if text_parts else ""

            meta: dict = {"_original_message": msg}
            if tool_name:
                meta["tool_name"] = tool_name
                # Extract file_path for file-reading tools
                if role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    args = tool_args_map.get(tool_call_id, {})
                elif tool_calls and isinstance(tool_calls, list):
                    fn_args_str = tool_calls[0].get("function", {}).get("arguments", "")
                    try:
                        args = json.loads(fn_args_str) if fn_args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                else:
                    args = {}
                file_path = _extract_file_path(tool_name, args, self._tool_type_overrides)
                if file_path:
                    meta["file_path"] = file_path

            turn = ConversationTurn(
                turn_number=counter,
                role=role if role != "tool" else "user",  # Normalize tool to user role
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

        For compacted turns, emits plain text content. For raw turns that
        still have their original structure, preserves it.

        Validates tool_calls/tool pairing in both directions:
        - Forward: if an assistant tool_calls was compacted, subsequent tool
          messages referencing it are flattened.
        - Reverse: if a tool response was compacted, the preceding assistant
          message's tool_calls field is stripped.
        OpenAI rejects orphaned messages in either direction.
        """
        messages = []

        # Pre-scan: collect tool_call_ids that have live (non-compacted)
        # tool role messages referencing them.
        live_tool_response_ids: set[str] = set()
        for turn in turns:
            if turn.compacted:
                continue
            original = turn.metadata.get("_original_message") if turn.metadata else None
            if not original:
                continue
            if original.get("role") == "tool":
                tc_id = original.get("tool_call_id", "")
                if tc_id:
                    live_tool_response_ids.add(tc_id)

        # Collect tool_call_ids that are present in the output as actual
        # tool_calls (not compacted to plain text).
        live_tool_call_ids: set[str] = set()

        for turn in turns:
            if turn.compacted:
                # Compacted turns always become plain text
                # Use the original role from the original message if available
                original = turn.metadata.get("_original_message") if turn.metadata else None
                orig_role = original.get("role", turn.role) if original else turn.role
                # Don't emit as role="tool" without tool_call_id — use "user" or "assistant"
                if orig_role == "tool":
                    orig_role = "user"
                messages.append({"role": orig_role, "content": turn.content})
            else:
                original = turn.metadata.get("_original_message") if turn.metadata else None
                if not original:
                    messages.append({"role": turn.role, "content": turn.content})
                    continue

                orig_role = original.get("role", "")

                # Check for orphaned tool messages
                if orig_role == "tool":
                    tool_call_id = original.get("tool_call_id", "")
                    if tool_call_id and tool_call_id not in live_tool_call_ids:
                        # Flatten orphaned tool message to plain text
                        messages.append({"role": "user", "content": turn.content})
                        logger.debug(
                            "Flattened turn %d: orphaned tool message (tool_call was compacted/summarized)",
                            turn.turn_number,
                        )
                        continue

                # Register tool_call_ids from assistant messages with tool_calls,
                # but strip tool_calls if their responses were compacted.
                if orig_role == "assistant":
                    tool_calls = original.get("tool_calls")
                    if tool_calls and isinstance(tool_calls, list):
                        has_orphan = any(
                            tc.get("id", "") and tc["id"] not in live_tool_response_ids
                            for tc in tool_calls
                        )
                        if has_orphan:
                            stripped = {k: v for k, v in original.items() if k != "tool_calls"}
                            messages.append(stripped)
                            logger.debug(
                                "Stripped tool_calls from turn %d: orphaned (tool response was compacted/summarized)",
                                turn.turn_number,
                            )
                            continue
                        for tc in tool_calls:
                            tc_id = tc.get("id", "")
                            if tc_id:
                                live_tool_call_ids.add(tc_id)

                messages.append(original)

        return messages
