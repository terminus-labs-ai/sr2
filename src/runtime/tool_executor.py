"""Tool executor that dispatches tool calls to registered handlers."""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ToolHandler(Protocol):
    """Protocol for tool handlers."""

    async def execute(self, **kwargs) -> str: ...


class ToolExecutor:
    """Dispatches tool calls to registered handlers."""

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}
        self._call_count: dict[str, int] = {}

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        """Register a handler for a tool name."""
        self._handlers[tool_name] = handler
        self._call_count[tool_name] = 0

    def has(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call. Returns result as string.

        Raises KeyError if tool not registered.
        """
        if tool_name not in self._handlers:
            available = ", ".join(sorted(self._handlers.keys()))
            raise KeyError(
                f"No handler registered for tool: {tool_name}. Available tools: {available}"
            )

        self._call_count[tool_name] = self._call_count.get(tool_name, 0) + 1

        handler = self._handlers[tool_name]
        result = await handler.execute(**arguments)
        return str(result)

    @property
    def registered_tools(self) -> list[str]:
        return list(self._handlers.keys())

    def get_call_count(self, tool_name: str) -> int:
        return self._call_count.get(tool_name, 0)

    @property
    def total_calls(self) -> int:
        return sum(self._call_count.values())


class SimpleTool:
    """A simple tool handler wrapping an async function."""

    def __init__(self, func):
        """
        Args:
            func: async function(**kwargs) -> str
        """
        self._func = func

    async def execute(self, **kwargs) -> str:
        return await self._func(**kwargs)


class SaveMemoryTool:
    """Built-in tool that writes a memory directly to the store.

    Bypasses LLM extraction — use this when the user explicitly asks
    the agent to remember something specific.
    """

    def __init__(self, sr2_facade, session_resolver=None):
        self._sr2 = sr2_facade
        self._session_resolver = session_resolver  # callable() -> current session_id

    async def execute(
        self,
        key: str = "",
        value: str = "",
        memory_type: str = "semi_stable",
        **kwargs,
    ) -> str:
        if not key or not value:
            return "Error: 'key' and 'value' are required."
        session_id = self._session_resolver() if self._session_resolver else None
        await self._sr2.save_memory(
            key=key,
            value=value,
            memory_type=memory_type,
            source=f"session:{session_id}" if session_id else None,
        )
        return f"Remembered: [{key}] = {value}"


class PostToSessionTool:
    """Built-in tool that writes a message to a named session.

    When called:
    1. Injects the message into the target session as an assistant message
    2. If the target session is owned by an interface plugin with send() support,
       also pushes the message through that interface (e.g., sends a Telegram message)
    """

    def __init__(self, session_manager, interface_plugins: dict):
        """
        Args:
            session_manager: SessionManager instance
            interface_plugins: mapping of session_name -> plugin instance
        """
        self._sessions = session_manager
        self._plugins = interface_plugins

    async def execute(
        self, session: str = "", message: str = "", priority: str = "important", **kwargs
    ) -> str:
        """Post a message to a session and optionally push via interface."""
        logger.debug(f"Posting message to session '{session}': {message}")
        if not session or not message:
            return "Error: 'session' and 'message' are required."

        target = await self._sessions.get_or_create(session)
        logger.debug(f"Injecting to session {target} the message {message}")
        target.inject_message(
            role="assistant",
            content=message,
            metadata={"injected_by": "post_to_session", "priority": priority},
        )

        plugin = self._plugins.get(session)
        if plugin:
            try:
                logger.debug(
                    f"Calling interface plugin for session '{session}' with message '{message}' and priority '{priority}'"
                )
                await plugin.send(session, message, {"priority": priority})
            except Exception as e:
                logger.error(f"Failed to call interface plugin for session '{session}': {e}")
                return f"Message posted to session '{session}' but delivery failed: {e}"
        else:
            logger.warning(f"PostToSession not a plugin!!! {plugin}")
        return f"Message posted to session '{session}'."
