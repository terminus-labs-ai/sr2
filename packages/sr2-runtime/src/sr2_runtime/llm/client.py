"""LLM client wrapper with model routing and cache metadata extraction."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2_runtime.config import LLMModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""
    raw_tool_call_text: str = ""  # Original text when tool call was parsed from content/reasoning

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens


class LLMClient:
    """Wraps LiteLLM with model routing and cache metadata extraction."""

    def __init__(
        self,
        model: LLMModelConfig,
        fast_model: LLMModelConfig,
        embedding: LLMModelConfig,
    ):
        self._model = model
        self._fast_model = fast_model
        self._embedding = embedding

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        model_config: LLMModelConfig | None = None,
    ) -> LLMResponse:
        """Call the LLM via LiteLLM. Returns structured response with cache metadata."""
        import litellm

        cfg = model_config or self._model
        kwargs: dict = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "timeout": 99999,
        }

        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = tool_choice

            # Ensure Ollama models use native tool calling instead of
            # LiteLLM's fallback (which injects tools into the system prompt
            # and forces format=json, breaking everything).
            provider = self._infer_provider(cfg)
            if provider == "ollama":
                kwargs["model"] = cfg.name.replace("ollama/", "ollama_chat/", 1)

        if cfg.model_params:
            kwargs.update(cfg.model_params.to_api_kwargs())

        logger.debug(f"LLM Call - sendin this package to litellm: {kwargs}")
        response = await litellm.acompletion(**kwargs)
        logger.debug(f"LLM Call - got this response back from litellm: {response}")
        msg = response.choices[0].message

        tool_calls = self._extract_tool_calls(msg)
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        raw_tool_call_text = ""

        # Some models (e.g. GLM-4) put output in reasoning_content
        # instead of content/tool_calls.
        available = {t["name"] for t in tools} if tools else None
        if reasoning and not content and not tool_calls:
            parsed_tc = self._try_parse_tool_call(reasoning, available)
            if parsed_tc:
                tool_calls = [parsed_tc]
                raw_tool_call_text = reasoning
            else:
                content = reasoning

        # Check if content itself is a tool call
        if content and not tool_calls:
            parsed_tc = self._try_parse_tool_call(content, available)
            if parsed_tc:
                tool_calls = [parsed_tc]
                raw_tool_call_text = content
                content = ""
            elif self._looks_like_tool_call(content):
                # Hallucinated tool call — suppress the raw JSON from content
                logger.warning("Content looks like a hallucinated tool call, suppressing")
                raw_tool_call_text = content
                content = ""

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_tool_call_text=raw_tool_call_text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cached_tokens=self._extract_cached_tokens(response.usage),
            model=cfg.name,
        )

    async def stream_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        model_config: LLMModelConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream an LLM completion, yielding text deltas.

        After iteration completes, the final ``LLMResponse`` (including tool
        calls and token counts) is available via ``self.last_stream_response``.
        """
        import litellm

        cfg = model_config or self._model
        kwargs: dict = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "stream": True,
            "timeout": 99999,
        }

        # stream_options is only supported by some providers (OpenAI, Anthropic).
        # Ollama and other local providers reject or mishandle it.
        provider = self._infer_provider(cfg)
        if provider not in ("ollama", "ollama_chat"):
            kwargs["stream_options"] = {"include_usage": True}

        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = tool_choice

            # Ensure Ollama models use native tool calling (see complete()).
            if provider == "ollama":
                kwargs["model"] = cfg.name.replace("ollama/", "ollama_chat/", 1)

        if cfg.model_params:
            kwargs.update(cfg.model_params.to_api_kwargs())

        logger.debug(f"LLM Stream Call - sending this package to litellm: {kwargs}")
        response = await litellm.acompletion(**kwargs)
        logger.debug(f"LLM Stream Call - got this response back from litellm: {response}")

        full_content = ""
        reasoning_content = ""
        # Accumulate tool call chunks keyed by index
        tool_call_accumulators: dict[int, dict] = {}
        usage_prompt = 0
        usage_completion = 0
        usage_cached = 0
        chunk_count = 0

        async for chunk in response:
            chunk_count += 1
            delta = chunk.choices[0].delta if chunk.choices else None

            if chunk_count <= 3 or (chunk_count % 50 == 0):
                logger.debug(
                    f"Stream chunk #{chunk_count}: "
                    f"choices={bool(chunk.choices)}, "
                    f"delta={delta}, "
                    f"content={getattr(delta, 'content', None) if delta else None}, "
                    f"role={getattr(delta, 'role', None) if delta else None}"
                )

            if delta:
                # Text content — stream immediately
                if delta.content:
                    full_content += delta.content
                    yield delta.content

                # Reasoning content — accumulate silently; some models
                # (e.g. GLM-4) put tool call JSON or final answers here
                # instead of in content/tool_calls.
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_content += rc

                # Tool call chunks
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        acc = tool_call_accumulators[idx]
                        if tc_chunk.id:
                            acc["id"] = tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                acc["name"] = tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                acc["arguments"] += tc_chunk.function.arguments

            # Usage from final chunk
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_prompt = chunk.usage.prompt_tokens
                usage_completion = chunk.usage.completion_tokens
                usage_cached = self._extract_cached_tokens(chunk.usage)

        # Build completed tool calls from structured chunks
        completed_tool_calls = []
        for idx in sorted(tool_call_accumulators):
            acc = tool_call_accumulators[idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except (json.JSONDecodeError, AttributeError):
                args = {}
            completed_tool_calls.append({"id": acc["id"], "name": acc["name"], "arguments": args})

        # If the model used reasoning_content instead of content/tool_calls
        # (e.g. GLM-4), try to parse it as a tool call; otherwise treat it
        # as the response text.
        raw_tool_call_text = ""
        available = {t["name"] for t in tools} if tools else None
        if reasoning_content and not full_content and not completed_tool_calls:
            parsed_tc = self._try_parse_tool_call(reasoning_content, available)
            if parsed_tc:
                completed_tool_calls = [parsed_tc]
                raw_tool_call_text = reasoning_content
                logger.debug(f"Parsed tool call from reasoning_content: {parsed_tc['name']}")
            else:
                # Not a tool call — treat as the actual response text and
                # stream it as a single delta so callers see the content.
                full_content = reasoning_content
                yield full_content

        # Check if streamed content is actually a tool call (e.g. model
        # output <tool_call>{...}</tool_call> as plain text)
        if full_content and not completed_tool_calls:
            parsed_tc = self._try_parse_tool_call(full_content, available)
            if parsed_tc:
                completed_tool_calls = [parsed_tc]
                raw_tool_call_text = full_content
                full_content = ""
                logger.debug(f"Parsed tool call from content: {parsed_tc['name']}")
            elif self._looks_like_tool_call(full_content):
                # Model hallucinated a tool call with an invalid tool name.
                # The JSON was already streamed to the user — mark it for
                # retraction so it doesn't show up as raw JSON.
                logger.warning(
                    "Streamed content looks like a hallucinated tool call, "
                    "suppressing as raw_tool_call_text"
                )
                raw_tool_call_text = full_content
                full_content = ""

        logger.debug(
            f"Stream complete: {chunk_count} chunks, "
            f"content_len={len(full_content)}, "
            f"tool_calls={len(completed_tool_calls)}"
        )

        self.last_stream_response = LLMResponse(
            content=full_content,
            tool_calls=completed_tool_calls,
            input_tokens=usage_prompt,
            output_tokens=usage_completion,
            cached_tokens=usage_cached,
            model=cfg.name,
            raw_tool_call_text=raw_tool_call_text,
        )

    async def fast_complete(self, system: str, prompt: str) -> str:
        """Quick completion using the fast model.
        For memory extraction, summarization, intent detection."""
        resp = await self.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model_config=self._fast_model,
        )
        return resp.content

    async def embed(self, text: str) -> list[float]:
        """Generate a text embedding."""
        import litellm

        kwargs: dict = {
            "model": self._embedding.name,
            "input": [text],
        }
        if self._embedding.api_base:
            kwargs["api_base"] = self._embedding.api_base

        provider = self._infer_provider(self._embedding)
        if provider not in ("ollama",):
            kwargs["encoding_format"] = "float"  # llama.cpp rejects null

        response = await litellm.aembedding(**kwargs)
        return response.data[0]["embedding"]

    def _extract_tool_calls(self, msg) -> list[dict]:
        """Extract tool calls from the LLM message object."""
        calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    }
                )
        return calls

    @staticmethod
    def _infer_provider(cfg: LLMModelConfig) -> str:
        """Infer the LLM provider from model name and api_base.

        LiteLLM uses a 'provider/model' naming convention (e.g. 'ollama/llama3').
        When a bare model name is used with an api_base pointing at ollama, we
        detect it from the URL.
        """
        name = cfg.name.lower()
        if "/" in name:
            return name.split("/", 1)[0]
        if cfg.api_base:
            base = cfg.api_base.lower()
            if ":11434" in base or "/ollama" in base:
                return "ollama"
        return "openai"

    @staticmethod
    def _looks_like_tool_call(text: str) -> bool:
        """Check if text looks like a tool-call JSON structure.

        Returns True if the text is a JSON object with a ``"name"`` key and
        optionally an ``"arguments"`` key — the shape models use when they
        emit tool calls as plain text instead of structured API calls.
        Also matches <tool_code>function_name(args)</tool_code> patterns.
        """
        import re

        text = text.strip()

        # Check for <tool_code>...</tool_code> pattern (function-call syntax)
        if re.search(r"<tool_code>\s*\w+\s*\(", text):
            return True

        m = re.search(r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>)?", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        if not (text.startswith("{") and text.endswith("}")):
            return False
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(data, dict) and "name" in data

    @staticmethod
    def _try_parse_tool_code_block(
        text: str, available_tools: set[str] | None = None
    ) -> dict | None:
        """Try to parse <tool_code>function_name(args)</tool_code> patterns.

        Some models (e.g. Llama, Qwen) emit tool calls as function-call
        syntax in <tool_code> blocks rather than structured tool_calls or
        JSON. Handles:
          - <tool_code>func_name("arg")</tool_code>
          - <tool_code>func_name(key="value", key2="value2")</tool_code>
          - <tool_code>func_name({"key": "value"})</tool_code>
        """
        import ast
        import re

        m = re.search(
            r"<tool_code>\s*(\w+)\s*\((.*?)\)\s*(?:</tool_code>)?",
            text,
            re.DOTALL,
        )
        if not m:
            return None

        name = m.group(1)
        raw_args = m.group(2).strip()

        if available_tools is not None and name not in available_tools:
            logger.warning(f"Model hallucinated tool '{name}' not in available tools, ignoring")
            return None

        # Parse arguments
        arguments: dict = {}
        if raw_args:
            # Try JSON first: func({"key": "value"})
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    arguments = parsed
                else:
                    arguments = {"input": parsed}
            except json.JSONDecodeError:
                # Try Python-style kwargs: func(key="value", key2="value2")
                # or positional: func("value")
                try:
                    parsed = ast.literal_eval(f"({raw_args},)")
                    if len(parsed) == 1 and isinstance(parsed[0], dict):
                        arguments = parsed[0]
                    elif len(parsed) == 1:
                        arguments = {"input": parsed[0]}
                    else:
                        arguments = {"args": list(parsed)}
                except (ValueError, SyntaxError):
                    # Try key=value parsing
                    kwarg_pattern = re.findall(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\S+)', raw_args)
                    if kwarg_pattern:
                        for k, v in kwarg_pattern:
                            v = v.strip("\"'")
                            arguments[k] = v
                    else:
                        # Give up parsing, pass raw string
                        arguments = {"input": raw_args}

        return {
            "id": f"tool_code_{name}",
            "name": name,
            "arguments": arguments,
        }

    @staticmethod
    def _try_parse_tool_call(text: str, available_tools: set[str] | None = None) -> dict | None:
        """Try to parse text as a tool call JSON.

        Some models (e.g. GLM-4) emit tool calls as raw JSON in
        reasoning_content or content rather than using structured tool_calls.
        Handles:
          - Raw JSON: {"name": "func", "arguments": {...}}
          - <tool_call>{"name": "func", "arguments": {...}}</tool_call>
          - <tool_code>func_name(args)</tool_code>

        If *available_tools* is provided, rejects tool names that aren't
        in the set (prevents hallucinated tool names from being dispatched).
        """
        import re

        text = text.strip()

        # Try <tool_code>function_name(args)</tool_code> first
        tool_code_result = LLMClient._try_parse_tool_code_block(text, available_tools)
        if tool_code_result:
            return tool_code_result

        # Strip <tool_call>...</tool_call> wrapper if present
        m = re.search(r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>)?", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        if not (text.startswith("{") and text.endswith("}")):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "name" in data:
            name = data["name"]
            if available_tools is not None and name not in available_tools:
                logger.warning(f"Model hallucinated tool '{name}' not in available tools, ignoring")
                return None
            return {
                "id": f"reasoning_{name}",
                "name": name,
                "arguments": data.get("arguments", {}),
            }
        return None

    def _extract_cached_tokens(self, usage) -> int:
        """Extract cached token count from usage metadata.
        Handles differences across providers."""
        # Anthropic
        if hasattr(usage, "cache_read_input_tokens"):
            return getattr(usage, "cache_read_input_tokens", 0)
        # OpenAI
        details = getattr(usage, "prompt_tokens_details", None)
        if details and hasattr(details, "cached_tokens"):
            return details.cached_tokens
        return 0
