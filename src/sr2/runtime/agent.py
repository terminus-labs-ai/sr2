"""SR2Runtime — config-driven single-agent runtime."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from sr2.runtime.config import AgentConfig
from sr2.runtime.llm import RuntimeLLMClient
from sr2.runtime.result import RuntimeMetrics, RuntimeResult
from sr2.runtime.tools import RuntimeToolExecutor

logger = logging.getLogger(__name__)


class SR2Runtime:
    """Config-driven single-agent runtime.

    Wraps the SR2 facade with an LLM client and tool executor into
    a single ``await execute(task)`` interface.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._sr2 = self._build_sr2()
        self._llm = RuntimeLLMClient(config.model)
        self._tools = RuntimeToolExecutor(config.tools) if config.tools else None
        self._session_id = f"runtime-{config.name}"
        self._turn_counter = 0

    @classmethod
    def from_config(cls, path: str | Path) -> SR2Runtime:
        """Create runtime from a YAML config file."""
        return cls(AgentConfig.from_yaml(path))

    @classmethod
    def from_dict(cls, config: dict) -> SR2Runtime:
        """Create runtime from a config dictionary."""
        return cls(AgentConfig.from_dict(config))

    async def execute(
        self,
        task: str,
        context: str | None = None,
    ) -> RuntimeResult:
        """Execute a task with optional orchestrator context.

        Runs the full SR2 pipeline + LLM + tool-calling loop until the
        model produces a final text response or ``max_tool_iterations``
        is reached.
        """
        start = time.perf_counter()
        metrics = RuntimeMetrics()
        tool_results: list[dict[str, Any]] = []
        processed = None

        try:
            # 1. Inject orchestrator context as a preceding user turn
            if context:
                await self._sr2.post_process(
                    turn_number=self._next_turn(),
                    role="user",
                    content=f"Context from prior step:\n{context}",
                    session_id=self._session_id,
                )

            # 2. Get tool schemas
            tool_schemas = self._tools.get_schemas() if self._tools else []

            # 3. Tool-calling loop
            output = ""
            response = None
            for iteration in range(self.config.output.max_tool_iterations):
                # Compile context through SR2
                processed = await self._sr2.process(
                    interface_name="runtime_execute",
                    tool_schemas=tool_schemas,
                    trigger_input=task,
                    session_turns=[],
                    session_id=self._session_id,
                    system_prompt=self.config.persona.system_prompt,
                )

                metrics.llm_calls += 1

                # Call LLM with SR2-compiled messages
                response = await self._llm.complete(
                    messages=processed.messages,
                    tools=processed.tool_schemas or None,
                    tool_choice=processed.tool_choice,
                )

                # Update token metrics
                metrics.prompt_tokens += response.prompt_tokens
                metrics.completion_tokens += response.completion_tokens
                metrics.total_tokens += response.total_tokens

                # No tool calls -> final response
                if not response.tool_calls:
                    output = response.content or ""
                    await self._sr2.post_process(
                        turn_number=self._next_turn(),
                        role="assistant",
                        content=output,
                        session_id=self._session_id,
                        user_message=task,
                    )
                    break

                # Execute tool calls
                for tool_call in response.tool_calls:
                    metrics.tool_calls += 1
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]

                    try:
                        result = await self._tools.execute(tool_name, tool_args)
                        tool_results.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result,
                            "error": None,
                        })
                        result_str = str(result)
                    except Exception as e:
                        tool_results.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": None,
                            "error": str(e),
                        })
                        result_str = f"Tool error: {e}"

                    # Feed tool result back through SR2
                    await self._sr2.post_process(
                        turn_number=self._next_turn(),
                        role="tool_result",
                        content=result_str,
                        session_id=self._session_id,
                        tool_results=[{
                            "tool_name": tool_name,
                            "result": result_str,
                        }],
                    )
            else:
                # Exhausted max iterations
                output = response.content or "" if response else ""

            # Collect metrics from SR2
            if processed is not None:
                self._collect_sr2_metrics(metrics, processed)

            metrics.wall_time_ms = (time.perf_counter() - start) * 1000

            return RuntimeResult(
                output=output,
                success=True,
                metrics=metrics,
                tool_results=tool_results,
            )

        except Exception as e:
            metrics.wall_time_ms = (time.perf_counter() - start) * 1000
            return RuntimeResult(
                output="",
                success=False,
                error=str(e),
                metrics=metrics,
                tool_results=tool_results,
            )

    async def reset(self) -> None:
        """Reset conversation state. Memory persists.

        Note: SR2 does not expose a public reset method. This accesses
        the private ``_conversation`` attribute to destroy the session.
        Flag for future SR2 API improvement.
        """
        self._turn_counter = 0
        conv_mgr = getattr(self._sr2, "_conversation", None)
        if conv_mgr and hasattr(conv_mgr, "destroy_session"):
            conv_mgr.destroy_session(self._session_id)

    @property
    def name(self) -> str:
        """Agent name from config."""
        return self.config.name

    # --- Private methods ---

    def _build_sr2(self):
        """Construct the SR2 facade from AgentConfig."""
        from sr2.sr2 import SR2, SR2Config

        # Build pipeline config dict from AgentConfig.context
        pipeline_dict = self._build_pipeline_dict()

        # Create temp config dir with agent.yaml
        config_dir = tempfile.mkdtemp(prefix="sr2_runtime_")
        agent_yaml_path = Path(config_dir) / "agent.yaml"
        with open(agent_yaml_path, "w") as f:
            yaml.dump({"pipeline": pipeline_dict}, f)

        sr2_config = SR2Config(
            config_dir=config_dir,
            agent_yaml={"pipeline": pipeline_dict},
        )
        return SR2(sr2_config)

    def _build_pipeline_dict(self) -> dict[str, Any]:
        """Map AgentConfig.context to PipelineConfig-compatible dict."""
        ctx = self.config.context
        conv = ctx.conversation

        pipeline: dict[str, Any] = {
            "token_budget": ctx.context_window,
        }

        # Map conversation settings
        if "active_turns" in conv:
            pipeline.setdefault("compaction", {})["raw_window"] = conv["active_turns"]

        # Map memory settings
        mem = ctx.memory
        if "enabled" in mem:
            pipeline.setdefault("memory", {})["extract"] = mem["enabled"]

        # Merge pipeline_override (user can pass raw PipelineConfig fields)
        for key, value in ctx.pipeline_override.items():
            pipeline[key] = value

        return pipeline

    def _next_turn(self) -> int:
        """Increment and return turn counter."""
        self._turn_counter += 1
        return self._turn_counter

    def _collect_sr2_metrics(
        self,
        metrics: RuntimeMetrics,
        processed: Any,
    ) -> None:
        """Pull compaction count, cache hit rate from SR2 pipeline result."""
        try:
            pr = processed.pipeline_result
            if hasattr(pr, "compaction_count"):
                metrics.compaction_events = pr.compaction_count
        except Exception:
            pass  # Metrics are best-effort
