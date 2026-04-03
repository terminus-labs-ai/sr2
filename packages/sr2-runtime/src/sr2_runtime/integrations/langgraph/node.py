"""SR2Node — wraps the SR2 runtime Agent as a LangGraph-compatible node."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from sr2_runtime.agent import Agent, AgentConfig

logger = logging.getLogger(__name__)


class SR2Node:
    """LangGraph node wrapping an SR2 runtime Agent.

    Usage::

        researcher = SR2Node("researcher", "configs/agents/researcher")
        graph.add_node("research", researcher)

    The node delegates to ``Agent.handle_user_message()`` which already
    handles SR2 context compilation, the LLM tool-calling loop, session
    management, and post-processing.

    Supports both sync (``graph.invoke``) and async (``graph.ainvoke``)
    LangGraph execution.
    """

    def __init__(
        self,
        name: str,
        config_dir: str | Path,
        *,
        defaults_path: str = "configs/defaults.yaml",
        task_key: str = "current_task",
        context_key: str = "prior_output",
        output_key: str | None = None,
    ):
        self._name = name
        self._task_key = task_key
        self._context_key = context_key
        self._output_key = output_key or name

        agent_config = AgentConfig(
            name=name,
            config_dir=str(config_dir),
            defaults_path=defaults_path,
        )
        self._agent = Agent(agent_config)
        self._started = False

    async def start(self) -> None:
        """Initialize the agent (plugins, MCP, database, etc.)."""
        if not self._started:
            await self._agent.start()
            self._started = True

    async def shutdown(self) -> None:
        """Shut down the agent gracefully."""
        if self._started:
            await self._agent.shutdown()
            self._started = False

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Sync LangGraph node interface."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._execute(state))
                return future.result()
        except RuntimeError:
            return asyncio.run(self._execute(state))

    async def __acall__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Async LangGraph node interface."""
        return await self._execute(state)

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self._started:
            await self.start()

        task = state.get(self._task_key, "")
        prior_context = state.get(self._context_key)

        # Prepend prior agent output as context if available
        message = task
        if prior_context:
            message = f"Context from prior step:\n{prior_context}\n\nTask:\n{task}"

        session_id = f"langgraph-{self._name}"
        response = await self._agent.handle_user_message(message, session_id)

        # Build state update
        outputs = dict(state.get("outputs", {}))
        outputs[self._output_key] = response

        return {
            "prior_output": response,
            "outputs": outputs,
        }

    async def reset(self) -> None:
        """Reset by shutting down and re-creating the agent."""
        await self.shutdown()
        self._started = False

    @property
    def name(self) -> str:
        return self._name
