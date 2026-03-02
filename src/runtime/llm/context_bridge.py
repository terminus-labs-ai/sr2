"""Context-to-messages converter for LLM calls."""

import json

from sr2.pipeline.engine import CompiledContext


class ContextBridge:
    """Converts compiled context into LLM message arrays.

    SR2 compiles context as layered content. LLMs expect
    structured messages (system, user, assistant, tool). This bridge
    translates between the two.
    """

    def build_messages(
        self,
        compiled: CompiledContext,
        session_turns: list[dict],
        current_input: str | None = None,
    ) -> list[dict]:
        """Build the messages array for the LLM.

        Structure:
        1. System message = core layer + memory layer content
        2. Conversation turns from session (preserving roles)
        3. Current user message (if provided and not already in turns)

        Args:
            compiled: The compiled context from SR2
            session_turns: Conversation history from session
            current_input: The current user message (optional, may already be in turns)
        """
        messages = []

        # 1. System message: core + memory layers
        system_parts = []
        core = self._get_layer_content(compiled, "core")
        if core:
            system_parts.append(core)

        memory = self._get_layer_content(compiled, "memory")
        if memory:
            system_parts.append(memory)

        if system_parts:
            messages.append(
                {
                    "role": "system",
                    "content": "\n\n".join(system_parts),
                }
            )

        # 2. Session turns
        for turn in session_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            # Map internal roles to LLM roles
            if role == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": turn.get("tool_call_id", ""),
                        "name": turn.get("metadata", {}).get("tool_name", ""),
                    }
                )
            elif role == "assistant" and turn.get("tool_calls"):
                # Assistant message with tool calls
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": self._sanitize_tool_calls(turn["tool_calls"]),
                    }
                )
            else:
                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        # 3. Current input (if not already the last user message)
        if current_input:
            last_user = None
            for m in reversed(messages):
                if m["role"] == "user":
                    last_user = m["content"]
                    break
            if last_user != current_input:
                messages.append({"role": "user", "content": current_input})

        return messages

    def append_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict]:
        """Append a tool result to the messages array.

        Used during the tool-call loop to avoid full pipeline recompile.
        """
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages

    def append_assistant_tool_calls(
        self,
        messages: list[dict],
        content: str | None,
        tool_calls: list[dict],
        raw_tool_call_text: str = "",
    ) -> list[dict]:
        """Append the assistant's tool call message.

        Required by the API: the assistant message with tool_calls
        must be in the history before the tool results.

        When ``raw_tool_call_text`` is provided (tool call parsed from
        model text rather than structured tool_calls), it is used as the
        assistant message content so the model sees its own output in
        conversation history — important for models like GLM-4 on ollama
        that don't use native tool calling.
        """
        # Convert to API format
        formatted_calls = []
        for tc in tool_calls:
            formatted_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
            )

        # Use raw text as content so models that emit tool calls as text
        # see their own output in the conversation history.
        effective_content = content or raw_tool_call_text or None

        messages.append(
            {
                "role": "assistant",
                "content": effective_content,
                "tool_calls": formatted_calls,
            }
        )
        return messages

    @staticmethod
    def _sanitize_tool_calls(tool_calls: list[dict]) -> list[dict]:
        """Ensure tool call arguments are valid JSON strings.

        Session storage may contain arguments serialized with Python's
        str() (single-quoted dicts) instead of json.dumps(). LiteLLM's
        ollama provider calls json.loads() on arguments, so they must be
        valid JSON strings.
        """
        sanitized = []
        for tc in tool_calls:
            tc = dict(tc)  # shallow copy
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, dict):
                func = dict(func)
                func["arguments"] = json.dumps(args)
                tc["function"] = func
            elif isinstance(args, str):
                # Check if it's valid JSON; if not, try to fix Python repr
                try:
                    json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    # Likely Python str() output like "{'key': 'val'}"
                    # Try ast.literal_eval to parse, then re-serialize
                    import ast

                    try:
                        parsed = ast.literal_eval(args)
                        func = dict(func)
                        func["arguments"] = json.dumps(parsed)
                        tc["function"] = func
                    except (ValueError, SyntaxError):
                        func = dict(func)
                        func["arguments"] = "{}"
                        tc["function"] = func
            sanitized.append(tc)
        return sanitized

    def _get_layer_content(self, compiled: CompiledContext, layer_name: str) -> str:
        """Extract concatenated content from a named layer."""
        items = compiled.layers.get(layer_name, [])
        parts = [item.content for item in items if item.content]
        return "\n".join(parts)
