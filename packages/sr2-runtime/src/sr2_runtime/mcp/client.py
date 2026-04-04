"""MCP client integration for the runtime.

Connects to MCP servers, discovers tools/resources/prompts, handles sampling
requests, and wraps them as ToolHandlers so the ToolExecutor can dispatch
to them transparently.

Uses the official mcp Python SDK.

Connection lifecycle:
1. Agent.start() calls discover_all() — connects to each server, discovers
   tool/resource/prompt schemas, then disconnects immediately.  Schemas are
   cached in memory and never change, keeping the KV-cache prefix stable.
2. When a tool is actually *called*, MCPToolHandler asks MCPManager for a
   live session via _get_session().  The manager connects on demand and keeps
   the connection alive until an idle timeout fires.
3. Agent.shutdown() calls disconnect_all() to tear down any still-open
   connections.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from sr2_runtime.config import MCPServerConfig

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)

# Default idle timeout before tearing down on-demand connections (seconds).
_DEFAULT_IDLE_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


class MCPToolHandler:
    """Wraps a single MCP tool as a ToolHandler for the ToolExecutor.

    Instead of holding a live ClientSession (which can die), it asks the
    MCPManager for a session on demand.  The manager handles connection
    lifecycle transparently.
    """

    def __init__(self, manager: MCPManager, server_name: str, tool_name: str):
        self._manager = manager
        self._server_name = server_name
        self._tool_name = tool_name

    async def execute(self, **kwargs) -> str:
        """Execute the MCP tool call.

        Raises RuntimeError if the MCP server signals an error via isError,
        so the loop's circuit breaker can detect repeated failures.
        """
        logger.debug(
            f"MCP tool '{self._tool_name}' on server '{self._server_name}' called with: {kwargs}"
        )
        session = await self._manager._get_session(self._server_name)
        result = await session.call_tool(self._tool_name, arguments=kwargs)

        parts = []
        for block in result.content:
            if block.type == "text":
                parts.append(block.text)
            elif block.type == "image":
                parts.append(f"[image: {block.mimeType}]")
            else:
                parts.append(str(block))

        text = "\n".join(parts) if parts else "Tool returned no content."

        if getattr(result, "isError", False):
            logger.warning(
                f"MCP tool '{self._tool_name}' on server '{self._server_name}' "
                f"returned isError=True: {text}"
            )
            raise RuntimeError(text)

        return text


# ---------------------------------------------------------------------------
# Resource Handlers (exposed as agent tools when configured)
# ---------------------------------------------------------------------------


class MCPResourceHandler:
    """Wraps MCP resource read as a ToolHandler for on-demand access."""

    def __init__(self, manager: MCPManager):
        self._manager = manager

    async def execute(self, uri: str = "", server: str = "", **kwargs) -> str:
        return await self._manager.read_resource(uri, server_name=server or None)


class MCPListResourcesHandler:
    """Wraps MCP resource listing as a ToolHandler."""

    def __init__(self, manager: MCPManager):
        self._manager = manager

    async def execute(self, server: str = "", **kwargs) -> str:
        resources = await self._manager.list_resources(server_name=server or None)
        return json.dumps(resources, indent=2)


# ---------------------------------------------------------------------------
# Prompt Handler (exposed as agent tool when configured)
# ---------------------------------------------------------------------------


class MCPGetPromptHandler:
    """Wraps MCP prompt retrieval as a ToolHandler."""

    def __init__(self, manager: MCPManager):
        self._manager = manager

    async def execute(self, name: str = "", server: str = "", **kwargs) -> str:
        return await self._manager.get_prompt(name, kwargs or None, server_name=server or None)


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class MCPManager:
    """Manages connections to multiple MCP servers.

    Lifecycle:
    1. Agent.start() calls discover_all() — connects, caches schemas, disconnects.
    2. Tool execution calls _get_session() — connects on demand with idle timeout.
    3. Agent.shutdown() calls disconnect_all() — tears down any open connections.
    """

    def __init__(self):
        self._configs: list[MCPServerConfig] = []
        self._configs_by_name: dict[str, MCPServerConfig] = {}

        # Live connection state (populated on demand)
        self._sessions: dict[str, ClientSession] = {}
        self._server_tasks: dict[str, asyncio.Task] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._ready_events: dict[str, asyncio.Event] = {}
        self._idle_tasks: dict[str, asyncio.Task] = {}
        self._last_activity: dict[str, float] = {}

        # Cached schemas (populated once at discovery, never change)
        self._discovered_tools: dict[str, dict] = {}  # tool_name -> schema
        self._tool_server_map: dict[str, str] = {}  # tool_name -> server_name

        # Resources
        self._discovered_resources: dict[str, list[dict]] = {}  # server_name -> [resource info]
        self._resource_server_map: dict[str, str] = {}  # uri -> server_name

        # Prompts
        self._discovered_prompts: dict[str, list[dict]] = {}  # server_name -> [prompt info]
        self._prompt_server_map: dict[str, str] = {}  # prompt_name -> server_name

        # Sampling
        self._llm_client: Any = None
        self._sampling_timestamps: dict[str, list[float]] = {}  # server -> timestamps

        self._idle_timeout: float = _DEFAULT_IDLE_TIMEOUT

    def add_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server config."""
        self._configs.append(config)
        self._configs_by_name[config.name] = config

    def set_llm_client(self, llm_client: Any) -> None:
        """Wire the agent's LLMClient for sampling callbacks."""
        self._llm_client = llm_client

    # -------------------------------------------------------------------
    # Schema discovery (connect, discover, disconnect)
    # -------------------------------------------------------------------

    async def discover_all(self, tool_executor) -> dict[str, list[str]]:
        """Connect to all servers, discover schemas, register tools, disconnect.

        Schemas are cached permanently so that the tool list presented to the
        LLM never changes (preserving the KV-cache prefix).

        Args:
            tool_executor: ToolExecutor to register discovered tools into.

        Returns:
            Dict of server_name -> list of registered tool names.
        """
        if not self._configs:
            return {}

        if not _MCP_AVAILABLE:
            logger.warning(
                "MCP servers configured but mcp package not installed. "
                "Install with: pip install sr2[mcp]"
            )
            return {c.name: [] for c in self._configs}

        registered = {}

        for config in self._configs:
            try:
                tool_names = await self._discover_server(config, tool_executor)
                registered[config.name] = tool_names
                logger.info(f"MCP server '{config.name}': {len(tool_names)} tools registered")
            except Exception as e:
                logger.error(f"Failed to discover MCP server '{config.name}': {e}")
                registered[config.name] = []

        return registered

    async def _discover_server(self, config: MCPServerConfig, tool_executor) -> list[str]:
        """Connect to a single server, discover everything, disconnect."""
        cm = self._make_transport(config)

        async with cm as streams:
            read, write = streams[0], streams[1]

            session = ClientSession(read, write)
            async with session:
                await session.initialize()

                # Discover tools
                tool_names = await self._discover_tools(config, session, tool_executor)

                # Discover resources
                await self._discover_resources(config, session)

                # Discover prompts
                await self._discover_prompts(config, session)

                return tool_names

    # -------------------------------------------------------------------
    # On-demand connection management
    # -------------------------------------------------------------------

    async def _get_session(self, server_name: str) -> ClientSession:
        """Get a live session for a server, connecting on demand.

        If a connection is already open, reuses it and resets the idle timer.
        Otherwise, starts a new connection in a background task and waits for
        it to be ready.
        """
        # Reset idle timer
        self._last_activity[server_name] = time.monotonic()

        # Already connected?
        if server_name in self._sessions:
            return self._sessions[server_name]

        config = self._configs_by_name.get(server_name)
        if not config:
            raise KeyError(f"No MCP server config for '{server_name}'")

        if not _MCP_AVAILABLE:
            raise RuntimeError("MCP package not installed")

        # Spin up connection in background task
        ready_event = asyncio.Event()
        stop_event = asyncio.Event()
        self._ready_events[server_name] = ready_event
        self._stop_events[server_name] = stop_event

        connect_error: list[Exception] = []

        async def _run():
            try:
                cm = self._make_transport(config)
                async with cm as streams:
                    read, write = streams[0], streams[1]

                    roots_cb = self._build_roots_callback(config)
                    sampling_cb = self._build_sampling_callback(config)

                    session = ClientSession(
                        read,
                        write,
                        list_roots_callback=roots_cb,
                        sampling_callback=sampling_cb,
                    )
                    async with session:
                        await session.initialize()
                        self._sessions[server_name] = session
                        ready_event.set()

                        # Hold open until told to stop
                        await stop_event.wait()
            except Exception as e:
                connect_error.append(e)
                ready_event.set()  # unblock waiter
            finally:
                self._sessions.pop(server_name, None)
                self._ready_events.pop(server_name, None)

        task = asyncio.create_task(_run(), name=f"mcp-ondemand-{server_name}")
        self._server_tasks[server_name] = task

        # Start idle watcher
        self._start_idle_watcher(server_name)

        # Wait for connection to be ready
        await ready_event.wait()

        if connect_error:
            self._server_tasks.pop(server_name, None)
            self._stop_events.pop(server_name, None)
            raise RuntimeError(
                f"Failed to connect MCP server '{server_name}': {connect_error[0]}"
            ) from connect_error[0]

        return self._sessions[server_name]

    def _start_idle_watcher(self, server_name: str) -> None:
        """Start a task that tears down the connection after idle timeout."""
        # Cancel any existing watcher
        old = self._idle_tasks.pop(server_name, None)
        if old and not old.done():
            old.cancel()

        async def _watch():
            while True:
                await asyncio.sleep(self._idle_timeout / 2)
                last = self._last_activity.get(server_name, 0)
                if time.monotonic() - last >= self._idle_timeout:
                    logger.info(
                        f"MCP server '{server_name}' idle for {self._idle_timeout}s, disconnecting"
                    )
                    await self._disconnect_server(server_name)
                    return

        self._idle_tasks[server_name] = asyncio.create_task(
            _watch(), name=f"mcp-idle-{server_name}"
        )

    async def _disconnect_server(self, server_name: str) -> None:
        """Disconnect a single server."""
        event = self._stop_events.pop(server_name, None)
        if event:
            event.set()

        task = self._server_tasks.pop(server_name, None)
        if task:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"MCP server '{server_name}' did not shut down in time, cancelling")
                task.cancel()
            except Exception as e:
                logger.warning(f"Error disconnecting MCP server '{server_name}': {e}")

        idle_task = self._idle_tasks.pop(server_name, None)
        if idle_task and not idle_task.done():
            idle_task.cancel()

        self._sessions.pop(server_name, None)
        self._ready_events.pop(server_name, None)
        self._last_activity.pop(server_name, None)

    # -------------------------------------------------------------------
    # Transport factory
    # -------------------------------------------------------------------

    @staticmethod
    def _make_transport(config: MCPServerConfig):
        """Create the appropriate transport context manager."""
        if config.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.url.split()[0],
                args=config.url.split()[1:] + (config.args or []),
                env=config.env,
            )
            return stdio_client(server_params)
        elif config.transport in ("http", "sse"):
            return streamablehttp_client(config.url, headers=config.headers or {})
        else:
            raise ValueError(f"Unknown MCP transport: {config.transport}")

    # -------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------

    async def _discover_tools(
        self,
        config: MCPServerConfig,
        session: ClientSession,
        tool_executor,
    ) -> list[str]:
        """Discover tools from a connected session and register them."""
        tools_response = await session.list_tools()
        registered_names = []

        for tool in tools_response.tools:
            # If curated list specified, only register those
            if config.tools is not None and tool.name not in config.tools:
                continue

            logger.debug(f"MCP tool '{tool.name}': inputSchema={tool.inputSchema}")

            # Store schema for tool definitions (permanent, never changes)
            self._discovered_tools[tool.name] = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            }
            self._tool_server_map[tool.name] = config.name

            # Register handler that connects on demand
            handler = MCPToolHandler(self, config.name, tool.name)
            tool_executor.register(tool.name, handler)
            registered_names.append(tool.name)

        return registered_names

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI-style function schemas for all discovered MCP tools."""
        return list(self._discovered_tools.values())

    # -------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------

    async def _discover_resources(
        self,
        config: MCPServerConfig,
        session: ClientSession,
    ) -> None:
        """Discover resources from a connected session."""
        try:
            result = await session.list_resources()
        except Exception as e:
            logger.debug(f"MCP server '{config.name}' does not support resources: {e}")
            return

        resources = []
        for r in result.resources:
            uri_str = str(r.uri)
            info = {
                "uri": uri_str,
                "name": r.name if hasattr(r, "name") else "",
                "description": r.description if hasattr(r, "description") else "",
                "mimeType": r.mimeType if hasattr(r, "mimeType") else None,
            }
            resources.append(info)
            self._resource_server_map[uri_str] = config.name

        self._discovered_resources[config.name] = resources
        if resources:
            logger.info(f"MCP server '{config.name}': {len(resources)} resources discovered")

    async def list_resources(self, server_name: str | None = None) -> list[dict]:
        """List all discovered resources, optionally filtered by server."""
        if server_name:
            return list(self._discovered_resources.get(server_name, []))
        all_resources = []
        for resources in self._discovered_resources.values():
            all_resources.extend(resources)
        return all_resources

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        """Read a resource by URI.

        Finds the owning session (connecting on demand) and calls read_resource.
        """
        if not _MCP_AVAILABLE:
            raise RuntimeError("MCP package not installed")

        from pydantic import AnyUrl

        # Determine which server owns this resource
        owner = server_name or self._resource_server_map.get(uri)
        if not owner:
            # Try all servers
            for name in self._configs_by_name:
                try:
                    session = await self._get_session(name)
                    result = await session.read_resource(AnyUrl(uri))
                    return self._extract_resource_content(result)
                except Exception:
                    continue
            raise KeyError(f"No MCP server could read resource: {uri}")

        session = await self._get_session(owner)
        result = await session.read_resource(AnyUrl(uri))
        return self._extract_resource_content(result)

    @staticmethod
    def _extract_resource_content(result) -> str:
        """Extract text content from a ReadResourceResult."""
        parts = []
        for item in result.contents:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "blob"):
                parts.append(f"[binary: {getattr(item, 'mimeType', 'unknown')}]")
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else ""

    def get_resource_tool_schemas(self) -> list[dict]:
        """Get tool schemas for resource access tools."""
        return [
            {
                "name": "mcp_list_resources",
                "description": "List available resources from MCP servers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server": {
                            "type": "string",
                            "description": "Filter by server name. Empty for all servers.",
                        },
                    },
                },
            },
            {
                "name": "mcp_read_resource",
                "description": "Read a resource by URI from an MCP server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "The resource URI to read.",
                        },
                        "server": {
                            "type": "string",
                            "description": "Server name (optional, auto-detected from URI).",
                        },
                    },
                    "required": ["uri"],
                },
            },
        ]

    # -------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------

    async def _discover_prompts(
        self,
        config: MCPServerConfig,
        session: ClientSession,
    ) -> None:
        """Discover prompts from a connected session."""
        try:
            result = await session.list_prompts()
        except Exception as e:
            logger.debug(f"MCP server '{config.name}' does not support prompts: {e}")
            return

        prompts = []
        for p in result.prompts:
            info = {
                "name": p.name,
                "description": p.description if hasattr(p, "description") else "",
                "arguments": [
                    {
                        "name": a.name,
                        "description": getattr(a, "description", ""),
                        "required": getattr(a, "required", False),
                    }
                    for a in (p.arguments or [])
                ]
                if hasattr(p, "arguments") and p.arguments
                else [],
            }
            prompts.append(info)
            self._prompt_server_map[p.name] = config.name

        self._discovered_prompts[config.name] = prompts
        if prompts:
            logger.info(f"MCP server '{config.name}': {len(prompts)} prompts discovered")

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> str:
        """Get a filled prompt from an MCP server.

        Connects on demand if needed.
        """
        if not _MCP_AVAILABLE:
            raise RuntimeError("MCP package not installed")

        owner = server_name or self._prompt_server_map.get(name)
        if not owner:
            raise KeyError(f"No MCP server has prompt: {name}")

        session = await self._get_session(owner)
        result = await session.get_prompt(name, arguments)
        return self._extract_prompt_content(result)

    @staticmethod
    def _extract_prompt_content(result) -> str:
        """Extract text content from a GetPromptResult."""
        parts = []
        for msg in result.messages:
            role = msg.role if hasattr(msg, "role") else "unknown"
            content = msg.content
            if hasattr(content, "text"):
                parts.append(f"[{role}] {content.text}")
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        parts.append(f"[{role}] {block.text}")
                    else:
                        parts.append(f"[{role}] {block}")
            else:
                parts.append(f"[{role}] {content}")
        return "\n".join(parts) if parts else ""

    async def list_prompts(self, server_name: str | None = None) -> list[dict]:
        """List all discovered prompts, optionally filtered by server."""
        if server_name:
            return list(self._discovered_prompts.get(server_name, []))
        all_prompts = []
        for prompts in self._discovered_prompts.values():
            all_prompts.extend(prompts)
        return all_prompts

    def get_prompt_tool_schemas(self) -> list[dict]:
        """Get tool schemas for prompt access tools."""
        return [
            {
                "name": "mcp_get_prompt",
                "description": "Get a filled prompt template from an MCP server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The prompt name.",
                        },
                        "server": {
                            "type": "string",
                            "description": "Server name (optional, auto-detected).",
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": {
                        "type": "string",
                        "description": "Prompt arguments (any extra kwargs are passed as arguments).",
                    },
                },
            },
        ]

    # -------------------------------------------------------------------
    # Roots (callback-based)
    # -------------------------------------------------------------------

    @staticmethod
    def _build_roots_callback(config: MCPServerConfig):
        """Build a list_roots_callback from config roots."""
        if not config.roots:
            return None

        async def _list_roots(context):
            from mcp import types as mcp_types

            roots = [mcp_types.Root(uri=uri, name=uri) for uri in config.roots]
            return mcp_types.ListRootsResult(roots=roots)

        return _list_roots

    # -------------------------------------------------------------------
    # Sampling (callback-based)
    # -------------------------------------------------------------------

    def _build_sampling_callback(self, config: MCPServerConfig):
        """Build a sampling_callback for server-initiated LLM requests."""
        if not config.sampling.enabled or config.sampling.policy == "deny":
            return None

        async def _sampling_handler(context, params):
            from mcp import types as mcp_types

            server_name = config.name

            # Rate limiting
            if not self._check_rate_limit(server_name, config.sampling.rate_limit_per_minute):
                return mcp_types.ErrorData(
                    code=-32600,
                    message=f"Rate limit exceeded for server '{server_name}'",
                )

            if self._llm_client is None:
                return mcp_types.ErrorData(
                    code=-32600,
                    message="No LLM client configured for sampling",
                )

            logger.info(f"MCP sampling request from '{server_name}' ({config.sampling.policy})")

            # Convert MCP messages to LiteLLM/OpenAI format
            messages: list[dict[str, str]] = []
            if params.systemPrompt:
                messages.append({"role": "system", "content": params.systemPrompt})
            for msg in params.messages:
                content_parts = []
                msg_content = msg.content
                if isinstance(msg_content, list):
                    for block in msg_content:
                        if hasattr(block, "text"):
                            content_parts.append(block.text)
                elif hasattr(msg_content, "text"):
                    content_parts.append(msg_content.text)
                else:
                    content_parts.append(str(msg_content))

                role = msg.role if isinstance(msg.role, str) else msg.role.value
                messages.append({"role": role, "content": "\n".join(content_parts)})

            # Cap max_tokens
            max_tokens = min(
                params.maxTokens if params.maxTokens else config.sampling.max_tokens,
                config.sampling.max_tokens,
            )

            try:
                response = await self._llm_client.complete(
                    messages,
                    max_tokens=max_tokens,
                )

                return mcp_types.CreateMessageResult(
                    role="assistant",
                    content=mcp_types.TextContent(type="text", text=response.content),
                    model=response.model or "unknown",
                )
            except Exception as e:
                logger.error(f"Sampling request from '{server_name}' failed: {e}")
                return mcp_types.ErrorData(code=-32600, message=str(e))

        return _sampling_handler

    def _check_rate_limit(self, server_name: str, max_per_minute: int) -> bool:
        """Sliding-window rate limiter for sampling requests."""
        now = time.time()
        timestamps = self._sampling_timestamps.setdefault(server_name, [])
        timestamps[:] = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= max_per_minute:
            return False
        timestamps.append(now)
        return True

    # -------------------------------------------------------------------
    # Disconnect
    # -------------------------------------------------------------------

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        server_names = list(self._server_tasks.keys())
        for name in server_names:
            await self._disconnect_server(name)

        self._server_tasks.clear()
        self._stop_events.clear()
        self._sessions.clear()
        self._ready_events.clear()
        self._idle_tasks.clear()
        self._last_activity.clear()
        logger.info("All MCP servers disconnected")

    # -------------------------------------------------------------------
    # Legacy compat: connect_all forwards to discover_all
    # -------------------------------------------------------------------

    async def connect_all(self, tool_executor) -> dict[str, list[str]]:
        """Alias for discover_all() — kept for backward compatibility."""
        return await self.discover_all(tool_executor)
