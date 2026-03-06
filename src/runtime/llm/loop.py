"""The core agentic loop: LLM call -> tool execution -> repeat."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from runtime.llm.client import LLMClient
from runtime.llm.context_bridge import ContextBridge
from runtime.llm.streaming import (
    StreamCallback,
    StreamEndEvent,
    StreamRetractEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)

if TYPE_CHECKING:
    from runtime.config import LLMModelConfig, StreamContentConfig
    from sr2.tools.state_machine import ToolStateMachine

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool call during the loop."""

    tool_name: str
    arguments: dict
    result: str
    duration_ms: float
    success: bool
    error: str | None = None
    call_id: str = ""


@dataclass
class LoopResult:
    """Result of a full LLM loop execution."""

    response_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_tokens: int = 0
    stopped_reason: str = "complete"  # complete | max_iterations | error

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_input_tokens


class LLMLoop:
    """The core agentic loop: LLM call -> tool execution -> repeat.

    Named after the Quarian oath — this loop keeps the agent alive.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor,  # ToolExecutor instance
        bridge: ContextBridge | None = None,
        max_iterations: int = 25,
        default_model_config: LLMModelConfig | None = None,
    ):
        self._llm = llm_client
        self._tools = tool_executor
        self._bridge = bridge or ContextBridge()
        self._max_iter = max_iterations
        self._default_model_config = default_model_config

    async def run(
        self,
        messages: list[dict],
        tool_schemas: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        model_config_override: LLMModelConfig | None = None,
        tool_state_machine: ToolStateMachine | None = None,
    ) -> LoopResult:
        """Run the LLM loop.

        1. Call LLM with messages + tool schemas
        2. If response has tool calls:
           a. Execute each tool call
           b. Append assistant message (with tool_calls) to messages
           c. Append each tool result to messages
           d. Try state transitions based on tool calls
           e. Call LLM again
        3. Repeat until:
           a. LLM returns text without tool calls -> "complete"
           b. max_iterations reached -> "max_iterations"
           c. Unrecoverable error -> "error"

        If tool_state_machine is provided, tool schemas and tool_choice are
        re-computed from masking output before each LLM call, and state
        transitions are attempted after each tool execution.
        """
        result = LoopResult(response_text="")
        llm_response = None
        resolved_config = model_config_override or self._default_model_config
        consecutive_failures = 0
        last_failed_tool = ""

        # Apply initial masking if state machine is present
        active_tool_schemas = tool_schemas
        active_tool_choice = tool_choice
        if tool_state_machine is not None:
            masking = tool_state_machine.get_masking_output()
            active_tool_schemas = masking.get("tool_schemas", tool_schemas)
            active_tool_choice = masking.get("tool_choice", tool_choice)

        if active_tool_schemas:
            sample = active_tool_schemas[0] if active_tool_schemas else {}
            logger.debug(
                f"LLM loop: {len(active_tool_schemas)} tool schemas, "
                f"sample='{sample.get('name', '?')}' "
                f"params_keys={list(sample.get('parameters', {}).get('properties', {}).keys())[:5]}"
            )

        for i in range(self._max_iter):
            result.iterations = i + 1

            try:
                llm_response = await self._llm.complete(
                    messages=messages,
                    tools=active_tool_schemas,
                    tool_choice=active_tool_choice,
                    model_config=resolved_config,
                )
            except Exception as e:
                logger.error(f"LLM call failed at iteration {i + 1}: {e}")
                result.response_text = f"Error: LLM call failed: {e}"
                result.stopped_reason = "error"
                return result

            # Accumulate token counts
            result.total_input_tokens += llm_response.input_tokens
            result.total_output_tokens += llm_response.output_tokens
            result.cached_tokens += llm_response.cached_tokens

            # No tool calls -> we're done (or hallucinated tool call was suppressed)
            if not llm_response.has_tool_calls:
                if llm_response.raw_tool_call_text:
                    # Model hallucinated a tool call — retry without tools
                    logger.warning("Model emitted hallucinated tool call, retrying without tools")
                    active_tool_schemas = None
                    active_tool_choice = "none"
                    continue
                result.response_text = llm_response.content
                result.stopped_reason = "complete"
                return result

            # Has tool calls -> execute them
            # First, append the assistant message with tool calls
            self._bridge.append_assistant_tool_calls(
                messages,
                llm_response.content,
                llm_response.tool_calls,
                raw_tool_call_text=llm_response.raw_tool_call_text,
            )

            # Execute all tool calls concurrently when there are multiple
            records = await asyncio.gather(
                *[self._execute_tool_call(tc) for tc in llm_response.tool_calls]
            )

            # Process results sequentially: append messages, track failures, state transitions
            for tc, record in zip(llm_response.tool_calls, records):
                result.tool_calls.append(record)

                # Append tool result to messages
                self._bridge.append_tool_result(
                    messages,
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    result=record.result,
                )

                # Track consecutive failures of the same tool
                if not record.success:
                    if tc["name"] == last_failed_tool:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 1
                        last_failed_tool = tc["name"]
                else:
                    consecutive_failures = 0
                    last_failed_tool = ""

                # Try state transition after each tool execution
                if tool_state_machine is not None:
                    transitioned = tool_state_machine.try_transition(
                        "agent_action", {"tool_name": tc["name"]}
                    )
                    if transitioned:
                        masking = tool_state_machine.get_masking_output()
                        active_tool_schemas = masking.get("tool_schemas", tool_schemas)
                        active_tool_choice = masking.get("tool_choice", tool_choice)

            # Circuit breaker: if the same tool has failed repeatedly,
            # force a final call with no tools so the model must respond
            # with text.
            if consecutive_failures >= 3:
                logger.warning(
                    f"Tool '{last_failed_tool}' failed {consecutive_failures} times "
                    "consecutively, disabling tools for final response"
                )
                active_tool_schemas = None
                active_tool_choice = "none"

        # Reached max iterations
        result.response_text = llm_response.content if llm_response else ""
        result.stopped_reason = "max_iterations"
        logger.warning(f"LLM loop reached max iterations ({self._max_iter})")
        return result

    async def run_streaming(
        self,
        messages: list[dict],
        stream_callback: StreamCallback,
        stream_content: StreamContentConfig | None = None,
        tool_schemas: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        model_config_override: LLMModelConfig | None = None,
        tool_state_machine: ToolStateMachine | None = None,
    ) -> LoopResult:
        """Run the LLM loop with streaming.

        Same structure as ``run()`` but uses ``stream_complete()`` and pushes
        ``StreamEvent`` objects to *stream_callback* as tokens arrive.
        Returns the same ``LoopResult`` so callers are unaffected.
        """
        result = LoopResult(response_text="")
        resolved_config = model_config_override or self._default_model_config
        full_text = ""
        consecutive_failures = 0
        last_failed_tool = ""

        # Apply initial masking if state machine is present
        active_tool_schemas = tool_schemas
        active_tool_choice = tool_choice
        if tool_state_machine is not None:
            masking = tool_state_machine.get_masking_output()
            active_tool_schemas = masking.get("tool_schemas", tool_schemas)
            active_tool_choice = masking.get("tool_choice", tool_choice)

        if active_tool_schemas:
            sample = active_tool_schemas[0] if active_tool_schemas else {}
            logger.debug(
                f"LLM streaming loop: {len(active_tool_schemas)} tool schemas, "
                f"sample='{sample.get('name', '?')}' "
                f"params_keys={list(sample.get('parameters', {}).get('properties', {}).keys())[:5]}"
            )

        for i in range(self._max_iter):
            result.iterations = i + 1

            try:
                async for delta in self._llm.stream_complete(
                    messages=messages,
                    tools=active_tool_schemas,
                    tool_choice=active_tool_choice,
                    model_config=resolved_config,
                ):
                    full_text += delta
                    await stream_callback(TextDeltaEvent(content=delta))
            except Exception as e:
                logger.error(f"LLM streaming call failed at iteration {i + 1}: {e}")
                result.response_text = f"Error: LLM call failed: {e}"
                result.stopped_reason = "error"
                await stream_callback(StreamEndEvent(full_text=full_text))
                return result

            llm_response = self._llm.last_stream_response

            # Fallback: if streaming produced no content and no tool calls,
            # retry with non-streaming. Some providers (e.g. ollama with
            # certain models) don't stream correctly.
            if not llm_response.content and not llm_response.has_tool_calls:
                logger.warning("Streaming produced empty response, falling back to non-streaming")
                try:
                    llm_response = await self._llm.complete(
                        messages=messages,
                        tools=active_tool_schemas,
                        tool_choice=active_tool_choice,
                        model_config=resolved_config,
                    )
                except Exception as e:
                    logger.error(f"Non-streaming fallback failed: {e}")
                    result.response_text = f"Error: LLM call failed: {e}"
                    result.stopped_reason = "error"
                    await stream_callback(StreamEndEvent(full_text=""))
                    return result

                # Send the full response as a single delta
                if llm_response.content:
                    full_text = llm_response.content
                    await stream_callback(TextDeltaEvent(content=full_text))

            # Accumulate token counts
            result.total_input_tokens += llm_response.input_tokens
            result.total_output_tokens += llm_response.output_tokens
            result.cached_tokens += llm_response.cached_tokens

            # If tool calls were parsed from streamed text (raw_tool_call_text
            # is set), the JSON was already sent to the user as TextDeltaEvents.
            # Retract it so the interface can clean up.
            if llm_response.raw_tool_call_text and full_text:
                await stream_callback(
                    StreamRetractEvent(retracted_text=llm_response.raw_tool_call_text)
                )
                full_text = ""

            # No tool calls -> we're done (or hallucinated tool call was suppressed)
            if not llm_response.has_tool_calls:
                if llm_response.raw_tool_call_text:
                    # Model hallucinated a tool call — retry without tools so
                    # it produces a real text response instead.
                    logger.warning("Model emitted hallucinated tool call, retrying without tools")
                    active_tool_schemas = None
                    active_tool_choice = "none"
                    continue
                result.response_text = llm_response.content
                result.stopped_reason = "complete"
                await stream_callback(StreamEndEvent(full_text=full_text))
                return result

            # Has tool calls -> execute them
            self._bridge.append_assistant_tool_calls(
                messages,
                llm_response.content,
                llm_response.tool_calls,
                raw_tool_call_text=llm_response.raw_tool_call_text,
            )

            # Notify start for all tools, then execute concurrently
            if stream_content and stream_content.tool_status:
                for tc in llm_response.tool_calls:
                    await stream_callback(
                        ToolStartEvent(
                            tool_name=tc["name"],
                            tool_call_id=tc.get("id", ""),
                            arguments=tc["arguments"],
                        )
                    )

            records = await asyncio.gather(
                *[self._execute_tool_call(tc) for tc in llm_response.tool_calls]
            )

            # Process results sequentially: append messages, notify, track failures, state transitions
            for tc, record in zip(llm_response.tool_calls, records):
                result.tool_calls.append(record)

                # Append tool result to messages
                self._bridge.append_tool_result(
                    messages,
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    result=record.result,
                )

                # Notify about tool result
                if stream_content and stream_content.tool_results:
                    await stream_callback(
                        ToolResultEvent(
                            tool_name=tc["name"],
                            tool_call_id=tc.get("id", ""),
                            result=record.result,
                            success=record.success,
                        )
                    )

                # Track consecutive failures of the same tool
                if not record.success:
                    if tc["name"] == last_failed_tool:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 1
                        last_failed_tool = tc["name"]
                else:
                    consecutive_failures = 0
                    last_failed_tool = ""

                # Try state transition after each tool execution
                if tool_state_machine is not None:
                    transitioned = tool_state_machine.try_transition(
                        "agent_action", {"tool_name": tc["name"]}
                    )
                    if transitioned:
                        masking = tool_state_machine.get_masking_output()
                        active_tool_schemas = masking.get("tool_schemas", tool_schemas)
                        active_tool_choice = masking.get("tool_choice", tool_choice)

            # Circuit breaker: if the same tool has failed repeatedly,
            # force a final call with no tools so the model must respond
            # with text.
            if consecutive_failures >= 3:
                logger.warning(
                    f"Tool '{last_failed_tool}' failed {consecutive_failures} times "
                    "consecutively, disabling tools for final response"
                )
                active_tool_schemas = None
                active_tool_choice = "none"

        # Reached max iterations
        result.response_text = full_text
        result.stopped_reason = "max_iterations"
        logger.warning(f"LLM streaming loop reached max iterations ({self._max_iter})")
        await stream_callback(StreamEndEvent(full_text=full_text))
        return result

    async def _execute_tool_call(self, tc: dict) -> ToolCallRecord:
        """Execute a single tool call and return a record."""
        start = time.perf_counter()

        try:
            logger.info(f"Calling tool '{tc['name']}' with args: {tc['arguments']}")
            tool_result = await self._tools.execute(tc["name"], tc["arguments"])
            duration = (time.perf_counter() - start) * 1000
            logger.info(
                f"Tool '{tc['name']}' completed in {duration:.0f}ms, "
                f"result length: {len(tool_result)} chars"
            )
            logger.debug(f"Tool '{tc['name']}' result: {tool_result[:500]}")
            return ToolCallRecord(
                tool_name=tc["name"],
                arguments=tc["arguments"],
                result=tool_result,
                duration_ms=duration,
                success=True,
                call_id=tc.get("id", ""),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            error_msg = f"Tool execution error: {e}"
            logger.warning(
                f"Tool '{tc['name']}' failed after {duration:.0f}ms: {e}",
                exc_info=True,
            )
            return ToolCallRecord(
                tool_name=tc["name"],
                arguments=tc["arguments"],
                result=error_msg,
                duration_ms=duration,
                success=False,
                error=str(e),
                call_id=tc.get("id", ""),
            )
