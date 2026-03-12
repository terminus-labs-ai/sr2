"""Agent class — the main entry point for the SR2 Runtime."""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass

import yaml

from sr2 import SR2, SR2Config

from runtime.config import AgentYAMLConfig, LLMModelConfig, MCPServerConfig
from runtime.llm import ContextBridge, LLMClient, LLMLoop
from runtime.mcp import MCPManager
from runtime.plugins.base import TriggerContext
from runtime.plugins.registry import create_default_registry
from runtime.session import SessionConfig, SessionManager
from runtime.tool_executor import PostToSessionTool, SaveMemoryTool, ToolExecutor

logger = logging.getLogger(__name__)


def _load_agent_yaml_with_extends(agent_yaml_path: str) -> dict:
    """Load agent YAML with extends directive support.

    Recursively loads parent configs via 'extends' and merges them.
    Relative extends are resolved from the agent.yaml directory.
    """
    visited = set()

    def _load_recursive(path: str) -> dict:
        abs_path = os.path.abspath(path)
        if abs_path in visited:
            raise ValueError(f"Circular extends detected: {abs_path}")
        visited.add(abs_path)

        with open(abs_path) as f:
            config = yaml.safe_load(f) or {}

        extends = config.pop("extends", None)
        if not extends:
            return config

        # Resolve extends path relative to current file's directory
        if extends.startswith("/"):
            parent_path = extends
        else:
            parent_dir = os.path.dirname(abs_path)
            parent_path = os.path.normpath(os.path.join(parent_dir, extends))

        parent_config = _load_recursive(parent_path)

        # Deep merge: parent is base, current file overrides
        return _deep_merge(parent_config, config)

    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge override into base. Lists are replaced, not merged."""
        result = base.copy()
        for key, value in override.items():
            if value is None:
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return _load_recursive(agent_yaml_path)


@dataclass
class AgentConfig:
    """Top-level agent configuration. LLM and runtime settings come from agent.yaml."""

    name: str
    config_dir: str
    defaults_path: str = "configs/defaults.yaml"


class Agent:
    """A running agent instance. The Commander.

    Wires together SR2 (context engineering), the LLM loop,
    tool execution, memory, and interface plugins.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_executor: ToolExecutor | None = None,
    ):
        self._config = config
        self._name = config.name

        # Load raw agent YAML with extends support and validate with Pydantic
        agent_yaml_path = os.path.join(config.config_dir, "agent.yaml")
        self._agent_yaml = _load_agent_yaml_with_extends(agent_yaml_path)
        self._agent_config = AgentYAMLConfig(**self._agent_yaml)
        runtime_conf = self._agent_config.runtime

        # --- Database config ---
        self._db_pool = None
        self._db_url: str | None = None
        self._db_pool_min = runtime_conf.database.pool_min
        self._db_pool_max = runtime_conf.database.pool_max

        if runtime_conf.database.url:
            self._db_url = self._resolve_env_vars(runtime_conf.database.url)

        # LLM — configured from agent.yaml runtime.llm section
        self._llm = LLMClient(
            model=runtime_conf.llm.model,
            fast_model=runtime_conf.llm.fast_model,
            embedding=runtime_conf.llm.embedding,
        )

        # Inject synthetic heartbeat interface before SR2 sees the YAML,
        # so the InterfaceRouter registers it for pipeline routing.
        hb_cfg = self._agent_config.runtime.heartbeat
        if hb_cfg.enabled:
            if hb_cfg.pipeline:
                synthetic_name = f"_heartbeat_{hb_cfg.pipeline}"
                iface_def = {
                    "plugin": "timer",
                    "session": {"name": "heartbeat", "lifecycle": hb_cfg.session_lifecycle},
                    "pipeline": hb_cfg.pipeline,
                }
            else:
                synthetic_name = "heartbeat"
                iface_def = {
                    "plugin": "timer",
                    "session": {"name": "heartbeat", "lifecycle": hb_cfg.session_lifecycle},
                }
            interfaces = self._agent_yaml.setdefault("interfaces", {})
            if synthetic_name not in interfaces:
                interfaces[synthetic_name] = iface_def

        # SR2 facade — owns memory, pipeline, resolvers, compaction, metrics
        # MCP resource/prompt readers are wired after MCPManager is created below
        self._sr2 = SR2(
            SR2Config(
                config_dir=config.config_dir,
                defaults_path=config.defaults_path,
                agent_yaml=self._agent_yaml,
                fast_complete=lambda s, p: self._llm.fast_complete(s, p),
                embed=self._llm.embed,
                mcp_resource_reader=lambda uri, server_name=None: self._mcp_manager.read_resource(
                    uri, server_name=server_name
                ),
                mcp_prompt_reader=lambda name, args=None, server_name=None: (
                    self._mcp_manager.get_prompt(name, args, server_name=server_name)
                ),
            )
        )

        # Tools
        self._tool_executor = tool_executor or ToolExecutor()

        # Bridge + Loop (this bridge is for LLMLoop's append_* methods, separate from facade's)
        self._bridge = ContextBridge()
        self._loop = LLMLoop(
            llm_client=self._llm,
            tool_executor=self._tool_executor,
            bridge=self._bridge,
            max_iterations=runtime_conf.loop.max_iterations,
            default_model_config=runtime_conf.llm.model,
        )

        # --- Sessions (from YAML) ---
        session_cfgs = {}
        for name, cfg in self._agent_config.sessions.items():
            if name == "_default":
                continue
            session_cfgs[name] = SessionConfig(
                name=name,
                max_turns=cfg.get("max_turns", runtime_conf.session.max_turns),
                idle_timeout_minutes=cfg.get(
                    "idle_timeout_minutes", runtime_conf.session.idle_timeout_minutes
                ),
            )
        default_cfg_raw = self._agent_config.sessions.get("_default", {})
        default_cfg = SessionConfig(
            name="_default",
            max_turns=default_cfg_raw.get("max_turns", runtime_conf.session.max_turns),
            idle_timeout_minutes=default_cfg_raw.get(
                "idle_timeout_minutes", runtime_conf.session.idle_timeout_minutes
            ),
        )
        self._sessions = SessionManager(session_configs=session_cfgs, default_config=default_cfg)

        # --- Interface Plugins (from YAML) ---
        self._plugin_registry = create_default_registry()
        self._plugins: dict = {}
        self._session_to_plugin: dict = {}  # session_name -> plugin

        for iface_name, iface_config in self._agent_config.interfaces.items():
            plugin_name = iface_config.plugin
            plugin_cls = self._plugin_registry.get(plugin_name)
            # Pass as dict for plugin compatibility
            plugin = plugin_cls(
                interface_name=iface_name,
                config=iface_config.model_dump(),
                agent_callback=self._handle_trigger,
            )
            self._plugins[iface_name] = plugin

            # Map session name -> plugin for post_to_session delivery
            session_name = iface_config.session.name if iface_config.session else iface_name
            if not session_name.startswith("{"):  # Skip dynamic names
                self._session_to_plugin[session_name] = plugin

        # Register built-in tools
        self._tool_executor.register(
            "post_to_session",
            PostToSessionTool(self._sessions, self._session_to_plugin),
        )
        # _current_session_id and _current_session_turns are set per-trigger
        self._current_session_id: str | None = None
        self._current_session_turns: list[dict] = []

        # Heartbeat system
        self._heartbeat_config = self._agent_config.runtime.heartbeat
        self._heartbeat_scanner: object | None = None  # HeartbeatScanner
        self._heartbeat_store: object | None = None
        self._heartbeat_tools: list = []
        self._tool_executor.register(
            "save_memory",
            SaveMemoryTool(self._sr2, session_resolver=lambda: self._current_session_id),
        )

        # MCP — from agent.yaml
        self._mcp_manager = MCPManager()
        for server in self._agent_config.mcp_servers:
            self._mcp_manager.add_server(
                MCPServerConfig(
                    name=server.name,
                    url=server.url,
                    transport=server.transport,
                    tools=server.tools,
                    headers={k: self._resolve_env_vars(v) for k, v in server.headers.items()}
                    if server.headers
                    else None,
                    env={k: self._resolve_env_vars(v) for k, v in server.env.items()}
                    if server.env
                    else None,
                    args=server.args,
                    roots=[self._resolve_env_vars(r) for r in server.roots]
                    if server.roots
                    else None,
                    resources=server.resources,
                    expose_resources_as_tools=server.expose_resources_as_tools,
                    prompts=server.prompts,
                    expose_prompts_as_tools=server.expose_prompts_as_tools,
                    sampling=server.sampling,
                )
            )

        # Custom tools — auto-imported from YAML declarations
        for ct in self._agent_yaml.get("custom_tools", []):
            import importlib

            mod = importlib.import_module(ct["module"])
            cls = getattr(mod, ct["class"])
            self._tool_executor.register(ct["name"], cls())

        # A2A client tools — register remote agents callable as tools
        self._a2a_client_tools: list = []
        for a2a_conf in self._agent_yaml.get("a2a_clients", []):
            from sr2.a2a.client import A2AClientTool, A2AToolConfig

            tool = A2AClientTool(
                config=A2AToolConfig(
                    name=a2a_conf["name"],
                    target_url=a2a_conf["url"],
                    description=a2a_conf.get("description", ""),
                    timeout_seconds=a2a_conf.get("timeout_seconds", 120.0),
                ),
                http_callable=self._make_http_callable(),
            )
            self._tool_executor.register(a2a_conf["name"], tool)
            self._a2a_client_tools.append(tool)

    # --- Public API ---

    async def handle_user_message(
        self,
        message: str,
        session_id: str = "default",
    ) -> str:
        """Handle a user message. Convenience wrapper around _handle_trigger."""
        trigger = TriggerContext(
            interface_name="user_message",
            plugin_name="api",
            session_name=session_id,
            session_lifecycle="persistent",
            input_data=message,
        )
        return await self._handle_trigger(trigger)

    async def _ensure_ollama_models(self) -> None:
        """Pull any Ollama models that aren't already available locally."""
        runtime_llm = self._agent_config.runtime.llm
        models_to_check: list[tuple[str, str]] = []  # (ollama_name, api_base)

        for cfg in [runtime_llm.model, runtime_llm.fast_model]:
            if cfg and cfg.name.startswith("ollama/") and cfg.api_base:
                ollama_name = cfg.name.removeprefix("ollama/")
                models_to_check.append((ollama_name, cfg.api_base.rstrip("/")))

        if not models_to_check:
            return

        import httpx

        for model_name, base_url in models_to_check:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{base_url}/api/tags")
                    resp.raise_for_status()
                    local_models = [m["name"] for m in resp.json().get("models", [])]

                if model_name in local_models:
                    logger.info(f"Ollama model '{model_name}' already available")
                    continue

                logger.info(
                    f"Pulling Ollama model '{model_name}' from {base_url} (this may take a while)..."
                )
                async with httpx.AsyncClient(timeout=None) as client:
                    resp = await client.post(
                        f"{base_url}/api/pull",
                        json={"name": model_name, "stream": False},
                    )
                    resp.raise_for_status()
                logger.info(f"Ollama model '{model_name}' pulled successfully")

            except Exception as e:
                logger.warning(f"Could not ensure Ollama model '{model_name}': {e}")

    async def start(self) -> None:
        """Start all plugins + MCP connections."""
        # Ensure Ollama models are available before anything else
        await self._ensure_ollama_models()

        # Create database pool if configured
        if self._db_url:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "Database URL is configured but asyncpg is not installed. "
                    "Install with: pip install sr2[postgres]"
                ) from None

            self._db_pool = await asyncpg.create_pool(
                self._db_url,
                min_size=self._db_pool_min,
                max_size=self._db_pool_max,
            )

            # Create persistent stores
            await self._sr2.set_postgres_store(self._db_pool)

            from runtime.session import PostgresSessionStore

            session_store = PostgresSessionStore(self._db_pool)
            await session_store.create_tables()

            # Wire session store into session manager
            self._sessions._store = session_store

            # Load active sessions from database
            loaded = await self._sessions.load_active_sessions()
            logger.info(f"Loaded {loaded} active sessions from database")

            logger.info("Database connected, persistent stores active")

        # Heartbeat system — wire after DB pool is available
        if self._heartbeat_config.enabled:
            from runtime.heartbeat import (
                CancelHeartbeatTool,
                HeartbeatScanner,
                InMemoryHeartbeatStore,
                PostgresHeartbeatStore,
                ScheduleHeartbeatTool,
            )

            if self._db_pool:
                hb_store = PostgresHeartbeatStore(self._db_pool)
                await hb_store.create_tables()
            else:
                hb_store = InMemoryHeartbeatStore()
                logger.warning(
                    "Heartbeat enabled without database — using in-memory store "
                    "(heartbeats will not survive restarts)"
                )

            self._heartbeat_store = hb_store

            schedule_tool = ScheduleHeartbeatTool(
                store=hb_store,
                agent_name=self._name,
                max_context_turns=self._heartbeat_config.max_context_turns,
                session_resolver=lambda: self._current_session_id,
                session_turns_resolver=lambda: self._current_session_turns,
            )
            cancel_tool = CancelHeartbeatTool(store=hb_store)

            self._tool_executor.register("schedule_heartbeat", schedule_tool)
            self._tool_executor.register("cancel_heartbeat", cancel_tool)
            self._heartbeat_tools = [schedule_tool, cancel_tool]

            scanner = HeartbeatScanner(
                store=hb_store,
                agent_callback=self._handle_trigger,
                poll_interval_seconds=self._heartbeat_config.poll_interval_seconds,
                session_lifecycle=self._heartbeat_config.session_lifecycle,
                pipeline=self._heartbeat_config.pipeline,
            )
            self._heartbeat_scanner = scanner
            await scanner.start()
            logger.info("Heartbeat system started")

        # MCP — wire LLM client for sampling, then connect
        self._mcp_manager.set_llm_client(self._llm)
        mcp_result = await self._mcp_manager.connect_all(self._tool_executor)
        for server, tools in mcp_result.items():
            logger.info(f"MCP '{server}': registered tools {tools}")

        # Register MCP resource/prompt tools if any server requests it
        from runtime.mcp.client import (
            MCPResourceHandler,
            MCPListResourcesHandler,
            MCPGetPromptHandler,
        )

        for server in self._agent_config.mcp_servers:
            if server.expose_resources_as_tools and not self._tool_executor.has(
                "mcp_read_resource"
            ):
                self._tool_executor.register(
                    "mcp_read_resource", MCPResourceHandler(self._mcp_manager)
                )
                self._tool_executor.register(
                    "mcp_list_resources", MCPListResourcesHandler(self._mcp_manager)
                )
            if server.expose_prompts_as_tools and not self._tool_executor.has("mcp_get_prompt"):
                self._tool_executor.register(
                    "mcp_get_prompt", MCPGetPromptHandler(self._mcp_manager)
                )

        # Start all interface plugins
        for name, plugin in self._plugins.items():
            try:
                await plugin.start()
            except Exception as e:
                logger.error(f"Plugin '{name}' failed to start: {e}")

        logger.info(f"Agent '{self._name}' started with {len(self._plugins)} interfaces")

    async def shutdown(self) -> None:
        """Stop all plugins + MCP."""
        # Stop heartbeat scanner first
        if self._heartbeat_scanner:
            try:
                await self._heartbeat_scanner.stop()
            except Exception as e:
                logger.warning(f"Heartbeat scanner failed to stop: {e}")

        for name, plugin in self._plugins.items():
            try:
                await plugin.stop()
            except Exception as e:
                logger.warning(f"Plugin '{name}' failed to stop: {e}")
        await self._mcp_manager.disconnect_all()

        # Save all active sessions before closing
        for sid in self._sessions.active_sessions:
            await self._sessions.save_session(sid)

        self._sessions.cleanup_idle()

        # Close database pool
        if self._db_pool:
            await self._db_pool.close()

        logger.info(f"Agent '{self._name}' shut down")

    # --- Registration ---

    def register_tool(self, tool_name: str, handler) -> None:
        """Register a tool handler."""
        self._tool_executor.register(tool_name, handler)

    # --- Core trigger handler ---

    async def _handle_trigger(self, trigger: TriggerContext) -> str:
        """Universal trigger handler. Called by all interface plugins.

        1. Get or create session based on trigger's session config
        2. Handle special commands (__clear_session__)
        3. Add input to session
        4. Route to correct pipeline config via SR2 facade
        5. Run LLM loop
        6. Record results in session
        7. Post-process async
        8. Clean up ephemeral sessions
        9. Return response
        """
        # Session management
        logger.info(f"Agent._handle_trigger called with {trigger.plugin_name}")
        logger.info(
            f"Agent._handle_trigger: lifecycle={trigger.session_lifecycle} session_name={trigger.session_name}"
        )

        if trigger.session_lifecycle == "ephemeral":
            session = self._sessions.create_ephemeral(trigger.session_name)
        else:
            session = await self._sessions.get_or_create(trigger.session_name)
        self._current_session_id = session.id
        self._current_session_turns = session.turns

        # Inject heartbeat context turns into session before processing
        if trigger.metadata and trigger.metadata.get("context_turns"):
            for turn in trigger.metadata["context_turns"]:
                session.turns.append(turn)
            self._current_session_turns = session.turns

        # Special commands
        special = self._handle_special_command(trigger, session)
        if special is not None:
            if asyncio.iscoroutine(special):
                return await special
            return special

        # Add input to session (if not empty — timers have no input)
        # TODO: messages might come from other sources that are not the user. i.e. a2a
        if trigger.input_data:
            session.add_user_message(str(trigger.input_data))

        # Route + compile context via SR2 facade
        ctx = await self._sr2.process(
            interface_name=trigger.interface_name,
            tool_schemas=self._get_tool_schemas(),
            trigger_input=trigger.input_data,
            session_turns=session.turns,
            session_id=session.id,
            system_prompt=self._agent_config.system_prompt,
        )

        # Resolve per-interface model override
        model_config_override = None
        if ctx.model_override:
            model_config_override = self._resolve_model_config(
                self._agent_config.runtime.llm.model, ctx.model_override
            )

        resolved_model = model_config_override or self._agent_config.runtime.llm.model
        use_streaming = resolved_model.stream and trigger.respond_callback is not None

        if use_streaming:
            loop_result = await self._loop.run_streaming(
                ctx.messages,
                stream_callback=trigger.respond_callback,
                stream_content=self._agent_config.runtime.stream_content,
                tool_schemas=ctx.tool_schemas,
                tool_choice=ctx.tool_choice,
                model_config_override=model_config_override,
                tool_state_machine=ctx.state_machine,
            )
        else:
            loop_result = await self._loop.run(
                ctx.messages,
                ctx.tool_schemas,
                tool_choice=ctx.tool_choice,
                model_config_override=model_config_override,
                tool_state_machine=ctx.state_machine,
            )

        # Record in session — group tool calls by loop iteration so the
        # assistant message contains all tool_calls from the same LLM response
        # (required by OpenAI message format).
        from itertools import groupby
        from operator import attrgetter

        for _iteration, group in groupby(loop_result.tool_calls, key=attrgetter("iteration")):
            calls = [(tc.tool_name, tc.arguments, tc.result, tc.call_id) for tc in group]
            session.add_tool_calls_grouped(calls)
        if loop_result.response_text:
            session.add_assistant_message(loop_result.response_text)

        # Persist session (async, don't block response)
        if trigger.session_lifecycle != "ephemeral":
            asyncio.create_task(self._sessions.save_session(session.id))

        # Post-process async
        asyncio.create_task(
            self._sr2.post_process(
                turn_number=session.turn_count,
                role="assistant",
                content=loop_result.response_text or "",
                session_id=session.id,
                user_message=str(trigger.input_data) if trigger.input_data else None,
            )
        )

        # Prefix tracking + metrics
        cache_report = self._sr2.compare_prefix(ctx.compiled_snapshot, loop_result.cached_tokens)
        await self._sr2.collect_metrics(
            pipeline_result=ctx.pipeline_result,
            interface=trigger.interface_name,
            loop_iterations=loop_result.iterations,
            loop_total_tokens=loop_result.total_tokens,
            loop_tool_calls=len(loop_result.tool_calls),
            loop_cache_hit_rate=loop_result.cache_hit_rate,
            cache_report=cache_report,
            session_id=session.id,
            session_messages=session.turns,
            session_turn_count=session.user_message_count,
            session_created_at=session.created_at.timestamp(),
            tool_state_machine=ctx.state_machine,
        )

        # Clean up ephemeral session
        if trigger.session_lifecycle == "ephemeral":

            async def _delayed_destroy():
                await asyncio.sleep(5)
                await self._sessions.destroy(session.id)

            asyncio.create_task(_delayed_destroy())

        return loop_result.response_text or ""

    # --- Special command dispatcher ---

    def _handle_special_command(self, trigger: TriggerContext, session) -> str | None:
        """Handle plugin special commands. Returns response string or None if not a special command."""
        cmd = trigger.input_data or ""

        if cmd == "__clear_session__":

            async def _clear():
                await self._sessions.destroy(trigger.session_name)
                logger.info(
                    f"Agent._handle_trigger Session {trigger.session_name} "
                    f"from {trigger.plugin_name} cleared"
                )
                return "Session cleared."

            return _clear()

        if cmd == "__get_status__":
            status: dict = {
                "agent_name": self._name,
                "model": self._agent_config.runtime.llm.model.name,
                "active_sessions": self._sessions.active_sessions,
                "mcp_servers": list(self._mcp_manager._sessions.keys()),
            }
            # Current session info
            if session:
                status["session"] = {
                    "name": session.id,
                    "turn_count": session.turn_count,
                    "user_message_count": session.user_message_count,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                }
            # Latest metrics from SR2 collector
            try:
                from sr2.metrics.definitions import MetricNames

                snapshots = self._sr2._collector.get_latest(1)
                if snapshots:
                    snap = snapshots[0]
                    metrics: dict = {}
                    for name, key in [
                        ("total_tokens", MetricNames.PIPELINE_TOTAL_TOKENS),
                        ("budget_headroom_tokens", MetricNames.BUDGET_HEADROOM_TOKENS),
                        ("budget_headroom_ratio", MetricNames.BUDGET_HEADROOM_RATIO),
                        ("cache_hit_rate", MetricNames.CACHE_HIT_RATE),
                        ("naive_token_estimate", MetricNames.NAIVE_TOKEN_ESTIMATE),
                        ("token_savings_cumulative", MetricNames.TOKEN_SAVINGS_CUMULATIVE),
                        ("truncation_events", MetricNames.TRUNCATION_EVENTS),
                        ("raw_window_utilization", MetricNames.RAW_WINDOW_UTILIZATION),
                        ("loop_iterations", "sr2_loop_iterations"),
                        ("loop_tool_calls", "sr2_loop_tool_calls"),
                    ]:
                        m = snap.get(key)
                        if m is not None:
                            metrics[name] = m.value
                    status["metrics"] = metrics
                    status["token_budget"] = self._sr2._token_budget
            except Exception as e:
                logger.debug(f"Could not collect metrics for status: {e}")
            return json.dumps(status)

        if cmd == "__list_sessions__":
            sessions_info = []
            for sid in self._sessions.active_sessions:
                s = self._sessions.get(sid)
                if s:
                    sessions_info.append(
                        {
                            "id": s.id,
                            "turn_count": s.turn_count,
                            "user_message_count": s.user_message_count,
                            "created_at": s.created_at.isoformat(),
                            "last_activity": s.last_activity.isoformat(),
                        }
                    )
            return json.dumps(sessions_info)

        if cmd.startswith("__set_active_session__:"):
            session_name = cmd.split(":", 1)[1]

            async def _set():
                state_id = f"_plugin_state_{trigger.interface_name}"
                state_session = await self._sessions.get_or_create(state_id)
                state_session.metadata["active_session"] = session_name
                if self._sessions._store:
                    await self._sessions.save_session(state_id)
                return json.dumps({"active_session": session_name})

            return _set()

        if cmd == "__get_active_session__":

            async def _get():
                state_id = f"_plugin_state_{trigger.interface_name}"
                state_session = self._sessions.get(state_id)
                if not state_session and self._sessions._store:
                    state_session = await self._sessions._store.load(state_id)
                    if state_session:
                        self._sessions._sessions[state_id] = state_session
                active = state_session.metadata.get("active_session") if state_session else None
                return json.dumps({"active_session": active})

            return _get()

        return None

    # --- Internal helpers ---

    @staticmethod
    def _make_http_callable():
        """Create an httpx-based HTTP callable for A2A client tools."""
        import httpx

        async def _http_call(url: str, payload: dict, timeout: float) -> dict:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()

        return _http_call

    @staticmethod
    def _resolve_env_vars(value: str) -> str:
        """Replace ${VAR_NAME} with environment variable values."""

        def _replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{(\w+)\}", _replace, value)

    @staticmethod
    def _resolve_model_config(
        base: LLMModelConfig,
        override,
    ) -> LLMModelConfig:
        """Merge a per-interface LLMModelOverride onto a base LLMModelConfig."""
        merged = base.model_dump()
        if override.name is not None:
            merged["name"] = override.name
        if override.api_base is not None:
            merged["api_base"] = override.api_base
        if override.max_tokens is not None:
            merged["max_tokens"] = override.max_tokens
        if override.model_params is not None:
            base_params = merged.get("model_params", {})
            base_params.update(override.model_params)
            merged["model_params"] = base_params
        return LLMModelConfig(**merged)

    def _generate_agent_card(self) -> dict:
        """Generate an A2A-compatible agent card."""
        return {
            "name": self._name,
            "description": f"{self._name} agent powered by SR2",
            "url": "",
            "provider": {"organization": "SR2"},
            "version": "0.1.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
            },
            "skills": [],
        }

    def _get_tool_schemas(self) -> list[dict]:
        """Get tool schemas for the LLM call."""
        schemas = self._mcp_manager.get_tool_schemas()
        # Add resource/prompt tool schemas if exposed
        for server in self._agent_config.mcp_servers:
            if server.expose_resources_as_tools:
                schemas.extend(self._mcp_manager.get_resource_tool_schemas())
                break
        for server in self._agent_config.mcp_servers:
            if server.expose_prompts_as_tools:
                schemas.extend(self._mcp_manager.get_prompt_tool_schemas())
                break

        # Add A2A client tool schemas
        for tool in self._a2a_client_tools:
            schemas.append(tool.tool_definition)

        # Add heartbeat tool schemas
        for tool in self._heartbeat_tools:
            schemas.append(tool.tool_definition)

        # Add tool_definitions from agent.yaml (e.g. post_to_session)
        for tool_def in self._agent_yaml.get("tool_definitions", []):
            properties = {}
            required = []
            for param in tool_def.get("parameters", []):
                prop: dict = {"type": param.get("type", "string")}
                if "description" in param:
                    prop["description"] = param["description"]
                if "enum" in param:
                    prop["enum"] = param["enum"]
                if "default" in param:
                    prop["default"] = param["default"]
                properties[param["name"]] = prop
                if param.get("required", False):
                    required.append(param["name"])
            schema = {
                "name": tool_def["name"],
                "description": tool_def.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
            schemas.append(schema)

        return schemas
