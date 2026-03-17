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


class RecallMemoryTool:
    """Built-in tool that queries the memory system with filters.

    Provides agents with an explicit "reload" mechanism — when the initial
    context retrieval isn't relevant enough, they can actively search for
    what they need using query text, key prefixes, and scope overrides.

    Read-only: does not modify memories or trigger touch updates.
    """

    def __init__(
        self,
        retriever,
        memory_store,
        scope_config=None,
        key_schema=None,
    ):
        """
        Args:
            retriever: HybridRetriever instance for the agent
            memory_store: MemoryStore instance for key_prefix searches
            scope_config: MemoryScopeConfig for permission enforcement
            key_schema: list[KeySchemaEntry] for reference (not used in logic)
        """
        self._retriever = retriever
        self._store = memory_store
        self._scope_config = scope_config
        self._key_schema = key_schema or []

    @property
    def tool_definition(self) -> dict:
        desc = (
            "Search memory for specific information. Use when the context "
            "provided doesn't contain what you need."
        )
        if self._key_schema:
            prefixes = ", ".join(
                f"'{(e["prefix"] if isinstance(e, dict) else e.prefix).rstrip(".")}'" for e in self._key_schema
            )
            desc += f" Available key prefixes: {prefixes}."

        return {
            "name": "recall_memory",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What you're looking for — a natural language description"
                        ),
                    },
                    "key_prefix": {
                        "type": "string",
                        "description": (
                            "Optional. Filter to a specific key prefix from the "
                            "project's key schema (e.g., 'research', 'decision'). "
                            "Omit to search all."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["private", "project", "all"],
                        "description": (
                            "Optional. Which memory scope to search. "
                            "Defaults to 'all' (both private and project)."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Optional. Number of results to return. Defaults to 5."
                        ),
                    },
                },
                "required": ["query"],
            },
        }

    def _allowed_scopes(self) -> list[str]:
        """Return the scopes the agent is configured to read."""
        if not self._scope_config:
            return ["private", "project"]
        return list(self._scope_config.default_read)

    def _resolve_scope_filter(
        self, requested_scope: str | None
    ) -> tuple[list[str] | None, list[str] | None, str | None]:
        """Resolve the scope filter for a recall query.

        Returns:
            (scope_filter, scope_refs, warning) — warning is set if
            the agent tried to access a scope it doesn't have permission for.
        """
        allowed = self._allowed_scopes()
        warning = None

        if not self._scope_config:
            # No scope config — no filtering
            return None, None, None

        if requested_scope is None or requested_scope == "all":
            effective = allowed
        elif requested_scope in allowed:
            effective = [requested_scope]
        else:
            # Agent tried to access a scope outside their config
            effective = allowed
            warning = (
                f"Scope '{requested_scope}' is not available to this agent. "
                f"Searching within allowed scopes: {allowed}."
            )

        scope_refs: list[str] = []
        if "private" in effective and self._scope_config.agent_name:
            scope_refs.append(f"agent:{self._scope_config.agent_name}")
        if "project" in effective:
            ctx = self._retriever._current_context or {}
            project_id = ctx.get("project_id")
            if project_id:
                scope_refs.append(project_id)

        return effective, scope_refs if scope_refs else None, warning

    async def execute(
        self,
        query: str = "",
        key_prefix: str = "",
        scope: str = "",
        top_k: int = 5,
        **kwargs,
    ) -> str:
        if not query and not key_prefix:
            return "Error: at least 'query' or 'key_prefix' is required."

        # Ensure top_k is reasonable
        top_k = max(1, min(top_k or 5, 20))

        # Resolve scope
        scope_filter, scope_refs, scope_warning = self._resolve_scope_filter(
            scope if scope else None
        )

        results = []

        if key_prefix and not query:
            # Prefix-only search: use store.search_by_key_prefix
            all_memories = await self._store.search_by_key_prefix(key_prefix)
            # Apply scope filtering manually (search_by_key_prefix doesn't support it)
            if scope_filter:
                filtered = []
                for mem in all_memories:
                    if mem.scope not in scope_filter:
                        continue
                    if scope_refs and mem.scope_ref is not None and mem.scope_ref not in scope_refs:
                        continue
                    filtered.append(mem)
                all_memories = filtered
            # Convert to search results
            from sr2.memory.schema import MemorySearchResult
            results = [
                MemorySearchResult(
                    memory=mem, relevance_score=1.0, match_type="key_match"
                )
                for mem in all_memories[:top_k]
            ]
        else:
            # Query-based search via retriever
            # Temporarily override retriever scope for this search
            original_scope = self._retriever._scope_config
            if scope_filter is not None:
                from sr2.config.models import MemoryScopeConfig
                self._retriever._scope_config = MemoryScopeConfig(
                    default_read=scope_filter,
                    default_write=self._scope_config.default_write if self._scope_config else "private",
                    agent_name=self._scope_config.agent_name if self._scope_config else None,
                )
            try:
                results = await self._retriever.retrieve(
                    query=query, top_k=top_k * 2 if key_prefix else top_k,
                    max_tokens=2000,
                )
            finally:
                self._retriever._scope_config = original_scope
                # Clear pending touches — recall_memory is read-only
                self._retriever._pending_touch_ids = []

            # Post-filter by key_prefix if both query and prefix specified
            if key_prefix:
                results = [
                    r for r in results if r.memory.key.startswith(key_prefix)
                ][:top_k]

        # Format output
        return self._format_results(results, scope_warning)

    def _format_results(
        self,
        results: list,
        scope_warning: str | None = None,
    ) -> str:
        if not results and not scope_warning:
            return "No memories found matching your query."

        lines = []
        if scope_warning:
            lines.append(f"Note: {scope_warning}")

        if not results:
            lines.append("No memories found matching your query.")
            return "\n".join(lines)

        lines.append(f"Found {len(results)} memory/memories:\n")
        for r in results:
            m = r.memory
            score = f"{r.relevance_score:.2f}"
            scope_label = f"{m.scope}" + (f":{m.scope_ref}" if m.scope_ref else "")
            lines.append(
                f"[{m.key}] ({m.memory_type}, {scope_label}, relevance={score})\n"
                f"  {m.value}"
            )

        # Hard cap: truncate output to ~2000 tokens (~8000 chars)
        output = "\n\n".join(lines)
        if len(output) > 8000:
            output = output[:7950] + "\n\n... (results truncated)"
        return output
