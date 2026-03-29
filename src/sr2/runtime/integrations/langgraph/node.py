"""SR2Node — wraps SR2Runtime as a LangGraph-compatible node."""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any

from sr2.runtime.agent import SR2Runtime
from sr2.runtime.config import AgentConfig


class SR2Node:
    """
    LangGraph node wrapping an SR2Runtime agent.

    Usage:
        research = SR2Node("agents/liara.yaml")
        graph.add_node("research", research)

    Supports both sync (graph.invoke) and async (graph.ainvoke) execution.
    """

    def __init__(
        self,
        config: str | Path | dict | AgentConfig,
        task_key: str = "current_task",
        context_key: str = "prior_output",
        output_key: str | None = None,
    ):
        if isinstance(config, (str, Path)):
            self.runtime = SR2Runtime.from_config(config)
        elif isinstance(config, dict):
            self.runtime = SR2Runtime.from_dict(config)
        elif isinstance(config, AgentConfig):
            self.runtime = SR2Runtime(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self._task_key = task_key
        self._context_key = context_key
        self._output_key = output_key or self.runtime.name

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Sync LangGraph node interface."""
        try:
            asyncio.get_running_loop()
            # Already in async context — create a new thread to avoid nesting
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._execute(state))
                return future.result()
        except RuntimeError:
            # No running loop — normal sync context
            return asyncio.run(self._execute(state))

    async def __acall__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Async LangGraph node interface."""
        return await self._execute(state)

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        task = state.get(self._task_key, "")
        prior_context = state.get(self._context_key)

        result = await self.runtime.execute(task=task, context=prior_context)

        # Build state update
        update: dict[str, Any] = {
            "prior_output": result.output,
        }

        outputs = dict(state.get("outputs", {}))
        outputs[self._output_key] = result.output
        update["outputs"] = outputs

        metrics = dict(state.get("metrics", {}))
        metrics[self._output_key] = result.metrics.to_dict()
        update["metrics"] = metrics

        return update

    async def reset(self) -> None:
        await self.runtime.reset()

    @property
    def name(self) -> str:
        return self.runtime.name
