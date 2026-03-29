"""Tool loading and execution for SR2Runtime."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import Any, Callable

from sr2.runtime.config import ToolConfig


class RuntimeToolExecutor:
    """
    Loads tools from module paths and executes them.

    Follows patterns from src/runtime/tool_executor.py.
    """

    def __init__(self, tool_configs: list[ToolConfig]):
        self.tools: dict[str, Callable] = {}
        self.definitions: list[dict[str, Any]] = []
        self._load_tools(tool_configs)

    def _load_tools(self, configs: list[ToolConfig]) -> None:
        for config in configs:
            module = importlib.import_module(config.module)

            # Tool function: prefer function named after tool, then run()
            fn = getattr(module, config.name, None) or getattr(module, "run", None)
            if fn is None:
                raise RuntimeError(
                    f"Tool module '{config.module}' has no "
                    f"'{config.name}' or 'run' function"
                )
            self.tools[config.name] = fn

            # Tool definition: prefer TOOL_DEFINITION attr, else auto-generate
            defn = getattr(module, "TOOL_DEFINITION", None)
            if defn:
                self.definitions.append(defn)
            else:
                self.definitions.append(self._auto_definition(config.name, fn))

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool. Handles both sync and async callables."""
        if tool_name not in self.tools:
            raise KeyError(
                f"Unknown tool: '{tool_name}'. "
                f"Available: {list(self.tools.keys())}"
            )

        fn = self.tools[tool_name]
        if asyncio.iscoroutinefunction(fn):
            return await fn(**arguments)
        return fn(**arguments)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for LLM calls."""
        return self.definitions

    @staticmethod
    def _auto_definition(name: str, fn: Callable) -> dict[str, Any]:
        """Generate minimal tool definition from function signature."""
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        # Resolve stringified annotations (from __future__ annotations)
        try:
            hints = inspect.get_annotations(fn, eval_str=True)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = hints.get(param_name, param.annotation)
            param_type = type_map.get(annotation, "string")
            properties[param_name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": name,
            "description": (fn.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
