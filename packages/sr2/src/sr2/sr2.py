"""SR2 facade — single entry point for the runtime into the context-engineering pipeline."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from sr2.cache.policies import create_default_cache_registry
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.loader import ConfigLoader
from sr2.config.models import LLMModelOverride, PipelineConfig
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.dimensions import DimensionalMatcher
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.scope import ScopeDetector
from sr2.memory.registry import get_store
from sr2.memory.store import InMemoryMemoryStore
from sr2.metrics.collector import MetricCollector
from sr2.metrics.definitions import MetricNames
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.post_processor import PostLLMProcessor
from sr2.pipeline.prefix_tracker import PrefixSnapshot
from sr2.pipeline.result import PipelineResult
from sr2.pipeline.router import InterfaceRouter
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import (
    ContentResolverRegistry,
    ResolverContext,
    ResolvedContent,
    estimate_tokens,
)
from sr2.resolvers.runtime_resolver import RuntimeResolver
from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.static_template_resolver import StaticTemplateResolver
from sr2.summarization.engine import SummarizationEngine
from sr2.pipeline.trace import TraceCollector
from sr2.tools.models import ToolDefinition, ToolManagementConfig
from sr2.tools.state_machine import ToolStateMachine

logger = logging.getLogger(__name__)


@dataclass
class SR2Config:
    """Configuration for the SR2 facade."""

    config_dir: str
    agent_yaml: dict
    defaults_path: str | None = None
    config_filename: str = "agent.yaml"
    memory_store: Any = None  # Pre-configured MemoryStore (default: InMemoryMemoryStore)
    extra_resolvers: dict[str, Any] | None = None  # {source_name: resolver_instance}
    fast_complete: Callable | None = None  # async (system, prompt) -> str
    embed: Callable | None = None  # async (text) -> list[float]
    mcp_resource_reader: Callable | None = None  # async (uri, server_name) -> str
    mcp_prompt_reader: Callable | None = None  # async (name, arguments, server_name) -> str
    trace_collector: TraceCollector | None = None  # Pipeline Inspector traces
    preloaded_config: Any = None  # PipelineConfig — skip file-based loading (used by bridge)


@dataclass
class ProcessedContext:
    """Everything the runtime needs to run the LLM loop."""

    messages: list[dict]
    tool_schemas: list[dict]
    tool_choice: str | dict
    state_machine: ToolStateMachine
    model_override: LLMModelOverride | None
    compiled_snapshot: PrefixSnapshot | None
    pipeline_result: PipelineResult


@dataclass
class ProxyResult:
    """Result of proxy_optimize — what the bridge needs to rebuild its response."""

    system_injection: str | None
    zones: Any  # ConversationZones
    compaction_result: Any | None  # CompactionResult | None
    summarization_result: Any | None  # SummarizationResult | None
    current_context: dict[str, str] | None = None


class SR2:
    """Facade over the SR2 context-engineering pipeline.

    Runtime creates one instance and calls process() per trigger.
    All internal wiring (memory, resolvers, pipeline, compaction,
    metrics) is hidden behind this interface.
    """

    def __init__(self, config: SR2Config) -> None:
        self._config = config
        self._trace = config.trace_collector

        # Memory stack
        self._memory_store = (
            config.memory_store if config.memory_store is not None else InMemoryMemoryStore()
        )
        self._retriever = HybridRetriever(
            store=self._memory_store,
            embedding_callable=config.embed,
            trace_collector=self._trace,
        )
        self._matcher = DimensionalMatcher()
        self._conflict_detector = ConflictDetector(store=self._memory_store)
        self._conflict_resolver = ConflictResolver(store=self._memory_store)
        self._extractor = MemoryExtractor(
            llm_callable=lambda prompt: (
                config.fast_complete("You extract structured memories from conversations.", prompt)
                if config.fast_complete
                else None
            ),
            store=self._memory_store,
            embed_callable=config.embed,
        )

        # Config
        self._loader = ConfigLoader(defaults_path=config.defaults_path)

        # Discover pipeline configs from config_dir/interfaces/
        pipeline_interfaces = self._discover_interfaces(config.config_dir)

        # Register agent.yaml interface keys so routing works by interface name
        # (e.g. "test_check" maps to its pipeline file, not just "heartbeat_test")
        self._yaml_interfaces = config.agent_yaml.get("interfaces", {})
        for iface_name, iface_cfg in self._yaml_interfaces.items():
            if iface_name not in pipeline_interfaces and isinstance(iface_cfg, dict):
                pipeline = iface_cfg.get("pipeline")
                if pipeline:
                    full_path = os.path.join(config.config_dir, pipeline)
                    if os.path.exists(full_path):
                        pipeline_interfaces[iface_name] = full_path

        # Register agent-level pipeline config as "_default" fallback so
        # HTTP/API requests work even without an interfaces/ directory
        agent_pipeline = config.agent_yaml.get("pipeline")
        if agent_pipeline and isinstance(agent_pipeline, dict):
            if "_default" not in pipeline_interfaces:
                pipeline_interfaces["_default"] = agent_pipeline

        self._router = InterfaceRouter(interfaces=pipeline_interfaces, loader=self._loader)

        # Resolvers
        self._resolver_reg = self._build_resolver_registry()
        if config.mcp_resource_reader:
            from sr2.resolvers.mcp_resource_resolver import MCPResourceResolver

            self._resolver_reg.register(
                "mcp_resource", MCPResourceResolver(config.mcp_resource_reader)
            )
        if config.mcp_prompt_reader:
            from sr2.resolvers.mcp_prompt_resolver import MCPPromptResolver

            self._resolver_reg.register("mcp_prompt", MCPPromptResolver(config.mcp_prompt_reader))
        if config.extra_resolvers:
            for source_name, resolver in config.extra_resolvers.items():
                self._resolver_reg.register(source_name, resolver)
        self._cache_reg = create_default_cache_registry()

        # Compaction + Summarization
        if config.preloaded_config is not None:
            agent_config = config.preloaded_config
        else:
            agent_yaml_path = os.path.join(config.config_dir, config.config_filename)
            agent_config = self._loader.load(agent_yaml_path)

        # Pipeline — wire CircuitBreaker from config
        deg = agent_config.degradation
        self._engine = PipelineEngine(
            self._resolver_reg,
            self._cache_reg,
            circuit_breaker=CircuitBreaker(
                threshold=deg.circuit_breaker_threshold,
                cooldown_seconds=deg.circuit_breaker_cooldown_minutes * 60,
            ),
            trace_collector=self._trace,
        )

        # Wire retrieval enabled flag now that agent_config is loaded
        try:
            retrieval_resolver = self._resolver_reg.get("retrieval")
            retrieval_resolver.enabled = agent_config.retrieval.enabled
        except KeyError:
            pass  # No retrieval resolver registered

        # Wire key_schema and scope_config now that agent_config is loaded
        key_schema = [s.model_dump() for s in agent_config.memory.key_schema]
        if key_schema:
            self._extractor._key_schema = key_schema

        # Wire scope config to retriever and extractor
        scope_config = agent_config.memory.scope
        self._scope_config_resolved = scope_config  # cached for process() and save_memory()
        if scope_config:
            self._retriever._scope_config = scope_config
            self._extractor._scope_config = scope_config

        # Scope detector (auto-detect scope_ref when not provided via env var)
        self._scope_detector: ScopeDetector | None = None
        if scope_config and config.fast_complete:
            self._scope_detector = ScopeDetector(
                store=self._memory_store,
                llm_callable=lambda prompt: config.fast_complete(
                    "You classify conversations into project scopes.", prompt
                ),
                scope_config=scope_config,
            )

        self._compaction_engine = CompactionEngine(agent_config.compaction)
        self._summarization_engine = SummarizationEngine(
            config=agent_config.summarization,
            llm_callable=lambda s, p: config.fast_complete(s, p) if config.fast_complete else None,
        )

        # Conversation manager
        self._conversation = ConversationManager(
            compaction_engine=self._compaction_engine,
            summarization_engine=self._summarization_engine,
            raw_window=agent_config.compaction.raw_window,
            compacted_max_tokens=agent_config.token_budget // 2,
            trace_collector=self._trace,
        )

        # Wire budget overflow handler into engine
        self._engine._budget_overflow_handler = self._handle_budget_overflow

        # Post-LLM processor
        self._post_processor = PostLLMProcessor(
            conversation_manager=self._conversation,
            memory_extractor=self._extractor if agent_config.memory.extract else None,
            conflict_detector=self._conflict_detector,
            conflict_resolver=self._conflict_resolver,
            retriever=self._retriever,
            trace_collector=self._trace,
        )

        # Metrics
        self._collector = MetricCollector(config.agent_yaml.get("name", "agent"))
        self._alerts = None  # Available via sr2-pro (AlertRuleEngine)

        # Session tracking for new metrics
        self._session_start_times: dict[str, float] = {}
        self._session_turn_counts: dict[str, int] = {}
        self._token_savings_cumulative: dict[str, float] = {}
        self._token_budget = agent_config.token_budget

        # Observability plugins (config-driven)
        obs = agent_config.observability
        self._push_exporters: list = []
        for name in obs.push_exporters:
            try:
                from sr2.metrics.registry import get_push_exporter

                exporter_cls = get_push_exporter(name)
                self._push_exporters.append(exporter_cls(self._collector))
            except ImportError:
                logger.warning("Push exporter '%s' not available", name)

        self._pull_exporter_name = obs.pull_exporter

        if obs.alert_engine:
            try:
                from sr2.plugins.registry import PluginRegistry

                alert_reg: PluginRegistry = PluginRegistry(
                    "sr2.alerts", install_hint="pip install sr2-pro"
                )
                alert_cls = alert_reg.get(obs.alert_engine)
                self._alerts = alert_cls()
            except ImportError:
                logger.warning("Alert engine '%s' not available", obs.alert_engine)

        # Internal bridge for build_messages only
        from sr2.bridge import ContextBridge

        self._bridge = ContextBridge()

    async def process(
        self,
        interface_name: str,
        tool_schemas: list[dict],
        trigger_input: Any,
        session_turns: list[dict],
        session_id: str,
        system_prompt: str,
    ) -> ProcessedContext:
        """Compile context and prepare everything the LLM loop needs.

        Routes to the correct pipeline config, compiles context layers,
        builds messages, and creates the tool state machine with initial masking.
        """
        # Begin trace turn
        if self._trace:
            self._trace.begin_turn(
                turn_number=len(session_turns) + 1,
                session_id=session_id,
                interface_name=interface_name,
            )
            self._trace.emit("input", {
                "trigger_input": str(trigger_input)[:500] if trigger_input else "",
                "session_turns": len(session_turns),
                "session_id": session_id,
                "interface_name": interface_name,
                "tool_count": len(tool_schemas),
            }, session_id=session_id)

        # Route to pipeline config
        try:
            config = self._router.route(interface_name)
        except KeyError:
            logger.warning(
                f"Routing error: interface {interface_name} not found. "
                "Attempting to route based on pipeline path."
            )
            pipeline_path = self._resolve_pipeline(interface_name)
            if pipeline_path and pipeline_path in self._router.registered_interfaces:
                config = self._router.route(pipeline_path)
            elif "user_message" in self._router.registered_interfaces:
                logger.warning(
                    "Interface %r not found, falling back to 'user_message' pipeline config",
                    interface_name,
                )
                config = self._router.route("user_message")
            elif "_default" in self._router.registered_interfaces:
                logger.info("Falling back to agent-level pipeline config")
                config = self._router.route("_default")
            else:
                raise KeyError(
                    f"No pipeline config found for interface '{interface_name}' "
                    "and no fallback available. Create an interfaces/user_message.yaml "
                    "or define a pipeline section in agent.yaml."
                )

        # Model override
        model_override = config.llm.model if config.llm and config.llm.model else None

        # Use pipeline-level system_prompt if defined, otherwise fall back to agent-level
        effective_prompt = config.system_prompt if config.system_prompt else system_prompt

        # Build agent context dict
        agent_context = {
            "system_prompt": effective_prompt,
            "tool_definitions": "",
            "agent_persona": "",
            "session_history": session_turns,
        }

        # Build current_context from env vars (single-shot dispatchers set these)
        current_context: dict[str, str] = {}
        project_id = os.environ.get("SR2_PROJECT_ID")
        if project_id:
            current_context["project_id"] = project_id
        task_source = os.environ.get("SR2_TASK_SOURCE")
        if task_source:
            current_context["source"] = task_source

        # Auto-detect scope_ref if not provided via env var
        if "project_id" not in current_context and self._scope_detector is not None:
            try:
                user_msg = str(trigger_input) if trigger_input else None
                detected = await self._scope_detector.detect(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    session_id=session_id,
                )
                for scope_name, scope_ref in detected.items():
                    if scope_name != "private" and scope_ref is not None:
                        current_context["project_id"] = scope_ref
                        break
            except Exception:
                logger.error("Scope detection failed", exc_info=True)

        # Inject current_context into retriever for this request
        if current_context:
            self._retriever.update_context(current_context)

        # Compile context
        ctx = ResolverContext(
            agent_config=agent_context,
            trigger_input=trigger_input,
            session_id=session_id,
            interface_type=interface_name,
            scope_config=self._scope_config_resolved,
            current_context=current_context or None,
        )
        compiled = await self._engine.compile(config, ctx)
        logger.info(f"SR2.process: context compiled for {interface_name}")

        # Build messages
        messages = self._bridge.build_messages(
            compiled, session_turns, str(trigger_input) if trigger_input else None
        )

        # Build tool state machine from pipeline config
        # Support both flat {"name": ...} and OpenAI {"type": "function", "function": {"name": ...}} formats
        tool_defs = []
        for t in tool_schemas:
            if "function" in t and isinstance(t["function"], dict):
                fn = t["function"]
                tool_defs.append(
                    ToolDefinition(
                        name=fn["name"],
                        description=fn.get("description", ""),
                        raw_parameters=fn.get("parameters"),
                    )
                )
            else:
                tool_defs.append(
                    ToolDefinition(
                        name=t["name"],
                        description=t.get("description", ""),
                        raw_parameters=t.get("parameters"),
                    )
                )
        tool_mgmt = ToolManagementConfig(
            tools=tool_defs,
            states=config.tool_states,
            transitions=config.tool_transitions,
            masking_strategy=config.tool_masking.strategy,
            initial_state=config.tool_masking.initial_state,
        )
        state_machine = ToolStateMachine(tool_mgmt)

        # Get initial masking output
        masking = state_machine.get_masking_output()
        filtered_schemas = masking.get("tool_schemas", tool_schemas)
        tool_choice = masking.get("tool_choice", "auto")

        # Trace: tool state
        if self._trace:
            allowed = masking.get("allowed_tools", tool_defs)
            allowed_names = {t.name for t in allowed}
            self._trace.emit("tool_state", {
                "current_state": state_machine.current_state,
                "allowed_tools": sorted(allowed_names),
                "denied_tools": sorted(t.name for t in tool_defs if t.name not in allowed_names),
                "tool_choice": str(tool_choice),
                "denied_attempts": 0,
                "state_history": [state_machine.current_state],
            }, session_id=session_id)

        # Trace: LLM request summary
        if self._trace:
            self._trace.emit("llm_request", {
                "provider": "caller-managed",
                "message_count": len(messages),
                "system_tokens": len(str(messages[0].get("content", ""))) // 4 if messages else 0,
                "conversation_tokens": sum(
                    len(str(m.get("content", ""))) // 4 for m in messages[1:]
                ) if len(messages) > 1 else 0,
                "tool_count": len(filtered_schemas),
                "tool_choice": str(tool_choice),
            }, session_id=session_id)

        return ProcessedContext(
            messages=messages,
            tool_schemas=filtered_schemas,
            tool_choice=tool_choice,
            state_machine=state_machine,
            model_override=model_override,
            compiled_snapshot=compiled.prefix_snapshot,
            pipeline_result=compiled.pipeline_result,
        )

    async def post_process(
        self,
        turn_number: int,
        role: str,
        content: str,
        session_id: str,
        user_message: str | None = None,
        current_context: dict | None = None,
        extract_only: bool = False,
        tool_results: list[dict] | None = None,
    ) -> None:
        """Fire-and-forget post-LLM processing (memory extraction, compaction).

        user_message: the raw user input for this turn. When provided it is
        prepended to content so the extractor sees the full exchange — this is
        critical for capturing explicit "remember X" commands that only appear
        in the user's message, not in the assistant's reply.

        extract_only: When True, only run memory extraction (skip compaction
            and summarization). Used by ephemeral agents.
        """
        try:
            extract_content = content
            if user_message:
                extract_content = f"User: {user_message}\n\nAssistant: {content}"
            turn = ConversationTurn(
                turn_number=turn_number,
                role=role,
                content=extract_content,
            )
            # Build current_context: prefer explicit param, fall back to env vars
            ctx = current_context
            if ctx is None:
                ctx = {}
                _pid = os.environ.get("SR2_PROJECT_ID")
                if _pid:
                    ctx["project_id"] = _pid
                _src = os.environ.get("SR2_TASK_SOURCE")
                if _src:
                    ctx["source"] = _src
                # Use cached scope detection result if no project_id set
                if ctx and "project_id" not in ctx and self._scope_detector:
                    cached = self._scope_detector._cache.get(session_id)
                    if cached:
                        for scope_name, scope_ref in cached.items():
                            if scope_name != "private" and scope_ref is not None:
                                ctx["project_id"] = scope_ref
                                break
                ctx = ctx or None
            # Add tool result turns before the assistant turn so compaction
            # can apply content_type-based rules to tool outputs
            if tool_results:
                for tr in tool_results:
                    tool_turn = ConversationTurn(
                        turn_number=tr.get("turn_number", turn_number),
                        role="tool_result",
                        content=tr.get("content", ""),
                        content_type=tr.get("content_type", "tool_output"),
                        metadata=tr.get("metadata"),
                    )
                    self._conversation.add_turn(tool_turn, session_id=session_id)

            await self._post_processor.process(
                turn,
                session_id,
                current_context=ctx,
                extract_only=extract_only,
            )

            # Trace: post-process results
            if self._trace:
                cr = self._post_processor.last_compaction_result
                sr = self._post_processor.last_summarization_result
                self._trace.emit("post_process", {
                    "memory_extraction": {
                        "memories_extracted": self._post_processor.last_memories_extracted,
                        "conflicts_detected": self._post_processor.last_conflicts_detected,
                    },
                    "compaction": {
                        "turns_compacted": cr.turns_compacted if cr else 0,
                        "original_tokens": cr.original_tokens if cr else 0,
                        "compacted_tokens": cr.compacted_tokens if cr else 0,
                        "tokens_saved": (cr.original_tokens - cr.compacted_tokens) if cr else 0,
                        "strategy": "rule_based",
                        "details": [
                            {
                                "turn_number": d.turn_number,
                                "role": d.role,
                                "content_type": d.content_type,
                                "rule": d.rule_applied,
                                "original_tokens": d.original_tokens,
                                "compacted_tokens": d.compacted_tokens,
                            }
                            for d in (cr.details if cr else [])
                        ],
                        "cost_gate": {
                            "passed": cr.cost_gate_result.passed,
                            "token_savings_usd": cr.cost_gate_result.token_savings_usd,
                            "cache_invalidation_usd": cr.cost_gate_result.cache_invalidation_usd,
                            "net_savings_usd": cr.cost_gate_result.net_savings_usd,
                        } if cr and cr.cost_gate_result else None,
                    } if cr else None,
                    "summarization": {
                        "triggered": sr is not None,
                        "original_tokens": sr.original_tokens if sr else 0,
                        "summary_tokens": sr.summary_tokens if sr else 0,
                    } if sr else None,
                }, session_id=session_id)
        except Exception:
            logger.error("Post-processing failed", exc_info=True)
        finally:
            # Always close the trace turn — even on error.
            # Without this the trace stays in _current/_active forever and
            # is silently discarded when the next begin_turn() fires.
            if self._trace:
                self._trace.end_turn(session_id=session_id)

    async def save_memory(
        self,
        key: str,
        value: str,
        memory_type: str = "semi_stable",
        source: str | None = None,
        current_context: dict | None = None,
    ) -> None:
        """Directly persist a memory, bypassing LLM extraction.

        Used by the save_memory tool so explicit 'remember X' commands
        are stored with high confidence and no inference step.
        """
        from sr2.memory.schema import STABILITY_DEFAULTS, Memory

        mem = Memory(
            key=key,
            value=value,
            memory_type=memory_type if memory_type in STABILITY_DEFAULTS else "semi_stable",
            stability_score=STABILITY_DEFAULTS.get(memory_type, 0.7),
            confidence_source="explicit_statement",
            source=source,
        )

        # Stamp scope if scope config is wired
        if self._scope_config_resolved:
            if not self._extractor._stamp_scope(mem, current_context):
                raise ValueError(
                    f"Scope '{mem.scope}' not in allowed_write "
                    f"{self._scope_config_resolved.allowed_write}"
                )

        embedding = None
        if self._config.embed:
            try:
                embedding = await self._config.embed(f"{key}: {value}")
            except Exception:
                logger.error("Embedding failed for save_memory key=%s", key, exc_info=True)
        await self._memory_store.save(mem, embedding=embedding)

    async def get_memory_store_size(self) -> int:
        """Get total number of non-archived memories in the store."""
        try:
            return await self._memory_store.count()
        except Exception:
            logger.error("Failed to get memory store size", exc_info=True)
            return 0

    def set_memory_store(self, store) -> None:
        """Rewire all memory components to use a new store."""
        self._memory_store = store
        self._retriever._store = store
        self._conflict_detector._store = store
        self._conflict_resolver._store = store
        self._extractor._store = store
        if self._scope_detector is not None:
            self._scope_detector._store = store

    async def set_postgres_store(self, pool) -> None:
        """Late-bind a PostgreSQL memory store.

        Requires sr2-pro to be installed. Resolves via the store registry.
        """
        store_cls = get_store("postgres")
        store = store_cls(pool)
        await store.create_tables()
        self.set_memory_store(store)

    async def collect_metrics(
        self,
        pipeline_result: PipelineResult,
        interface: str,
        loop_iterations: int,
        loop_total_tokens: int,
        loop_tool_calls: int,
        loop_cache_hit_rate: float,
        cache_report=None,
        session_id: str = "default",
        session_messages: list[dict] | None = None,
        session_turn_count: int | None = None,
        session_created_at: float | None = None,
        tool_state_machine: ToolStateMachine | None = None,
    ) -> None:
        """Collect metrics from a loop execution.

        Args:
            session_turn_count: Actual turn count from the session (user messages).
                If provided, used directly instead of an internal counter.
            session_created_at: Session creation time as a Unix timestamp.
                If provided, used for accurate session duration.
        """
        extra: dict[str, float] = {
            "sr2_loop_iterations": loop_iterations,
            "sr2_loop_total_tokens": loop_total_tokens,
            "sr2_loop_tool_calls": loop_tool_calls,
            "sr2_cache_hit_rate": loop_cache_hit_rate,
        }
        if cache_report is not None:
            extra["sr2_context_prefix_stable"] = 1.0 if cache_report.prefix_stable else 0.0
            extra["sr2_cache_efficiency"] = cache_report.cache_efficiency

        # --- Conversation lifecycle ---
        # Turn count: prefer the real session value, fall back to internal counter
        if session_turn_count is not None:
            extra[MetricNames.CONVERSATION_TURN_COUNT] = float(session_turn_count)
        else:
            count = self._session_turn_counts.get(session_id, 0) + 1
            self._session_turn_counts[session_id] = count
            extra[MetricNames.CONVERSATION_TURN_COUNT] = float(count)

        # Session duration: prefer real session creation time
        if session_created_at is not None:
            extra[MetricNames.SESSION_DURATION_SECONDS] = time.time() - session_created_at
        else:
            if session_id not in self._session_start_times:
                self._session_start_times[session_id] = time.time()
            extra[MetricNames.SESSION_DURATION_SECONDS] = (
                time.time() - self._session_start_times[session_id]
            )

        # Session message count
        if session_messages is not None:
            extra[MetricNames.SESSION_MESSAGE_COUNT] = float(len(session_messages))

        # --- Naive vs managed comparison ---
        # Naive token estimate: sum raw content sizes from session messages
        actual_tokens = pipeline_result.total_tokens
        if session_messages is not None:
            naive_tokens = sum(len(str(m.get("content", ""))) // 4 for m in session_messages)
            extra[MetricNames.NAIVE_TOKEN_ESTIMATE] = float(naive_tokens)

            # Token savings cumulative
            savings_this_turn = max(0.0, naive_tokens - actual_tokens)
            cumulative = self._token_savings_cumulative.get(session_id, 0.0) + savings_this_turn
            self._token_savings_cumulative[session_id] = cumulative
            extra[MetricNames.TOKEN_SAVINGS_CUMULATIVE] = cumulative

            # Cost savings ratio
            if naive_tokens > 0:
                extra[MetricNames.COST_SAVINGS_RATIO] = 1.0 - (actual_tokens / naive_tokens)

        # --- Budget & utilization ---
        headroom = max(0, self._token_budget - actual_tokens)
        extra[MetricNames.BUDGET_HEADROOM_TOKENS] = float(headroom)
        extra[MetricNames.BUDGET_HEADROOM_RATIO] = (
            headroom / self._token_budget if self._token_budget > 0 else 0.0
        )
        extra[MetricNames.BUDGET_UTILIZATION] = (
            actual_tokens / self._token_budget if self._token_budget > 0 else 0.0
        )
        extra[MetricNames.TOKEN_EFFICIENCY] = (
            actual_tokens / self._token_budget if self._token_budget > 0 else 0.0
        )

        # Truncation events
        extra[MetricNames.TRUNCATION_EVENTS] = float(self._engine.truncation_events)

        # --- Zone dynamics ---
        extra[MetricNames.RAW_WINDOW_UTILIZATION] = self._conversation.get_raw_window_utilization(
            session_id
        )

        # Zone token counts
        zones = self._conversation.zones(session_id)
        extra[MetricNames.ZONE_RAW_TOKENS] = float(sum(len(t.content) // 4 for t in zones.raw))
        extra[MetricNames.ZONE_COMPACTED_TOKENS] = float(
            sum(len(t.content) // 4 for t in zones.compacted)
        )
        extra[MetricNames.ZONE_SUMMARIZED_TOKENS] = float(
            sum(len(s) // 4 for s in zones.summarized)
        )

        # Zone transition events (emitted as separate labeled metrics below)
        transitions = self._conversation.get_zone_transitions(session_id)

        # --- Circuit breaker ---
        cb_status = self._engine._circuit_breaker.status()
        cb_activations = sum(1 for s in cb_status.values() if s.get("is_open"))
        extra[MetricNames.CIRCUIT_BREAKER_ACTIVATIONS] = float(cb_activations)

        # --- Tool state machine ---
        extra[MetricNames.STATE_TRANSITION_RATE] = float(
            max(0, len(tool_state_machine.state_history) - 1) if tool_state_machine else 0.0
        )
        extra[MetricNames.DENIED_TOOL_ATTEMPTS] = float(
            tool_state_machine.denied_tool_attempts if tool_state_machine else 0.0
        )

        # --- Memory system ---
        extra[MetricNames.MEMORIES_EXTRACTED] = float(self._post_processor.last_memories_extracted)
        extra[MetricNames.MEMORY_CONFLICTS_DETECTED] = float(
            self._post_processor.last_conflicts_detected
        )
        try:
            store_size = await self._memory_store.count()
            extra[MetricNames.MEMORY_STORE_SIZE] = float(store_size)
        except Exception:
            logger.error("Failed to collect memory store size metric", exc_info=True)

        # --- Retrieval metrics ---
        if self._retriever._total_retrievals > 0:
            extra[MetricNames.RETRIEVAL_LATENCY_MS] = self._retriever.last_latency_ms
            extra[MetricNames.RETRIEVAL_PRECISION] = self._retriever.last_avg_precision
            extra[MetricNames.RETRIEVAL_EMPTY_RATE] = self._retriever.empty_rate

        # --- Compaction metrics (from last post-processing run) ---
        cr = self._post_processor.last_compaction_result
        if cr is not None and cr.original_tokens > 0:
            extra[MetricNames.COMPACTION_RATIO] = cr.compacted_tokens / cr.original_tokens
            total_turns = cr.turns_compacted + len(cr.turns)
            extra[MetricNames.COMPACTION_COVERAGE] = (
                cr.turns_compacted / total_turns if total_turns > 0 else 0.0
            )

        # --- Summarization metrics (from last post-processing run) ---
        sr = self._post_processor.last_summarization_result
        if sr is not None and sr.original_tokens > 0:
            extra[MetricNames.SUMMARIZATION_RATIO] = sr.summary_tokens / sr.original_tokens
            # Fidelity: for structured summaries, measure field completeness
            if hasattr(sr.summary, "key_decisions"):
                filled = sum(
                    1
                    for field in [
                        sr.summary.key_decisions,
                        sr.summary.unresolved,
                        sr.summary.facts,
                        sr.summary.user_preferences,
                        sr.summary.errors_encountered,
                    ]
                    if field
                )
                extra[MetricNames.SUMMARIZATION_FIDELITY] = filled / 5.0
            else:
                extra[MetricNames.SUMMARIZATION_FIDELITY] = 1.0 if sr.summary else 0.0

        snapshot = self._collector.collect(pipeline_result, interface, extra_metrics=extra)

        # Add labeled zone transition events to the snapshot
        for transition_type, count in transitions.items():
            snapshot.add(
                MetricNames.ZONE_TRANSITION_EVENTS,
                float(count),
                "event",
                zone_transition=transition_type,
            )

        # Trace: zones event — conversation zone state snapshot
        if self._trace:
            raw_util = self._conversation.get_raw_window_utilization(session_id)
            self._trace.emit("zones", {
                "summarized_turns": len(zones.summarized),
                "summarized_tokens": int(extra.get(MetricNames.ZONE_SUMMARIZED_TOKENS, 0)),
                "compacted_turns": len(zones.compacted),
                "compacted_tokens": int(extra.get(MetricNames.ZONE_COMPACTED_TOKENS, 0)),
                "raw_turns": len(zones.raw),
                "raw_tokens": int(extra.get(MetricNames.ZONE_RAW_TOKENS, 0)),
                "raw_window_utilization": raw_util,
                "zone_transitions": transitions,
            }, session_id=session_id)

        # Trace: metrics event — key indicators from the snapshot
        if self._trace:
            self._trace.emit("metrics", {
                "invocation_id": snapshot.invocation_id,
                "budget_utilization": extra.get(MetricNames.BUDGET_UTILIZATION, 0.0),
                "budget_headroom_tokens": int(extra.get(MetricNames.BUDGET_HEADROOM_TOKENS, 0)),
                "cache_hit_rate": loop_cache_hit_rate,
                "loop_iterations": loop_iterations,
                "loop_total_tokens": loop_total_tokens,
                "loop_tool_calls": loop_tool_calls,
                "truncation_events": int(extra.get(MetricNames.TRUNCATION_EVENTS, 0)),
                "circuit_breaker_activations": int(extra.get(
                    MetricNames.CIRCUIT_BREAKER_ACTIVATIONS, 0
                )),
                "memories_extracted": int(extra.get(MetricNames.MEMORIES_EXTRACTED, 0)),
                "compaction_ratio": extra.get(MetricNames.COMPACTION_RATIO, 0.0)
                if MetricNames.COMPACTION_RATIO in extra else None,
                "summarization_ratio": extra.get(MetricNames.SUMMARIZATION_RATIO, 0.0)
                if MetricNames.SUMMARIZATION_RATIO in extra else None,
            }, session_id=session_id)

        # Fire alert checks (via plugin)
        if self._alerts is not None:
            await self._alerts.evaluate(snapshot)

    def compare_prefix(
        self,
        compiled_snapshot: PrefixSnapshot | None,
        cached_tokens: int,
    ):
        """Compare prefix snapshot for cache efficiency tracking.

        Returns a CacheReport or None.
        """
        if compiled_snapshot is None:
            return None
        return self._engine._prefix_tracker.compare(compiled_snapshot, cached_tokens)

    def export_metrics(self) -> str:
        """Export collected metrics in Prometheus text exposition format.

        Uses the pull exporter configured via observability.pull_exporter
        (defaults to 'prometheus'). Requires the exporter plugin to be installed.
        """
        if not hasattr(self, "_exporter"):
            from sr2.metrics.registry import get_pull_exporter

            name = self._pull_exporter_name or "prometheus"
            exporter_cls = get_pull_exporter(name)
            self._exporter = exporter_cls(self._collector)
        return self._exporter.export()

    # --- Proxy API (used by sr2-bridge) ---

    async def proxy_optimize(
        self,
        new_turns: list[ConversationTurn],
        session_id: str,
        system_prompt: str | None = None,
        retrieval_query: str | None = None,
    ) -> "ProxyResult":
        """Optimize a pre-built turn list (proxy/bridge mode).

        Bridge receives full message history from Claude Code and converts
        to ConversationTurns itself.  This method runs the same compaction →
        summarization → scope detection → memory retrieval pipeline that
        ``process()`` uses, but without resolver-based content layers.

        Returns a ProxyResult with zones, system injection text, and
        compaction/summarization results so the bridge can rebuild its
        response.
        """
        # Trace: begin turn
        if self._trace:
            self._trace.begin_turn(
                turn_number=new_turns[-1].turn_number if new_turns else 0,
                session_id=session_id,
                interface_name="proxy",
            )
            self._trace.emit("input", {
                "trigger_input": retrieval_query[:500] if retrieval_query else "",
                "session_turns": len(new_turns),
                "session_id": session_id,
                "interface_name": "proxy",
                "tool_count": 0,
            }, session_id=session_id)

        # 1. Add turns to conversation manager
        for turn in new_turns:
            self._conversation.add_turn(turn, session_id)

        # 2. Compaction (local, no circuit breaker needed)
        compaction_result = self._conversation.run_compaction(session_id)
        if compaction_result and compaction_result.turns_compacted > 0:
            logger.info(
                "proxy_optimize: session=%s compacted %d turns (%d -> %d tokens)",
                session_id,
                compaction_result.turns_compacted,
                compaction_result.original_tokens,
                compaction_result.compacted_tokens,
            )

        # 3. Summarization (guarded by circuit breaker)
        summarization_result = None
        if not self._engine._circuit_breaker.is_open("summarization"):
            try:
                summarization_result = await self._conversation.run_summarization(session_id)
                if summarization_result:
                    logger.info(
                        "proxy_optimize: session=%s summarized turns %s (%d -> %d tokens)",
                        session_id,
                        summarization_result.turn_range,
                        summarization_result.original_tokens,
                        summarization_result.summary_tokens,
                    )
                    self._engine._circuit_breaker.record_success("summarization")
            except Exception:
                logger.warning("proxy_optimize: summarization failed", exc_info=True)
                self._engine._circuit_breaker.record_failure("summarization")

        # 4. Scope detection
        current_context: dict[str, str] | None = None
        if self._scope_detector and not self._engine._circuit_breaker.is_open("scope_detection"):
            try:
                detected = await self._scope_detector.detect(
                    system_prompt=system_prompt,
                    user_message=retrieval_query,
                    session_id=session_id,
                )
                if detected:
                    current_context = {}
                    for scope_name, scope_ref in detected.items():
                        if scope_name != "private" and scope_ref is not None:
                            current_context["project_id"] = scope_ref
                            break
                    if not current_context:
                        current_context = None
                if current_context:
                    self._retriever.update_context(current_context)
                self._engine._circuit_breaker.record_success("scope_detection")
            except Exception:
                logger.warning("proxy_optimize: scope detection failed", exc_info=True)
                self._engine._circuit_breaker.record_failure("scope_detection")

        # 5. Memory retrieval
        memory_injection: str | None = None
        if retrieval_query and not self._engine._circuit_breaker.is_open("memory_retrieval"):
            try:
                results = await self._retriever.retrieve(retrieval_query)
                if results:
                    memory_lines = [f"- {r.memory.key}: {r.memory.value}" for r in results]
                    memory_injection = (
                        "[Relevant memories from previous sessions]\n"
                        + "\n".join(memory_lines)
                        + "\n[End of memories]"
                    )
                    logger.info(
                        "proxy_optimize: session=%s retrieved %d memories",
                        session_id,
                        len(results),
                    )
                self._engine._circuit_breaker.record_success("memory_retrieval")
            except Exception:
                logger.warning("proxy_optimize: memory retrieval failed", exc_info=True)
                self._engine._circuit_breaker.record_failure("memory_retrieval")

        # 6. Build system injection
        zones = self._conversation.zones(session_id)
        system_injection: str | None = None
        injection_parts: list[str] = []
        if zones.summarized:
            summary_text = "\n\n".join(zones.summarized)
            injection_parts.append(
                f"[Previous conversation summary]\n{summary_text}\n"
                f"[End of summary — recent conversation follows]"
            )
        if memory_injection:
            injection_parts.append(memory_injection)
        if injection_parts:
            system_injection = "\n\n".join(injection_parts)

        # Trace: zones
        if self._trace:
            self._trace.emit("zones", {
                "summarized_turns": len(zones.summarized),
                "summarized_tokens": sum(len(s) // 4 for s in zones.summarized),
                "compacted_turns": len(zones.compacted),
                "compacted_tokens": sum(len(t.content) // 4 for t in zones.compacted),
                "raw_turns": len(zones.raw),
                "raw_tokens": sum(len(t.content) // 4 for t in zones.raw),
                "raw_window_utilization": self._conversation.get_raw_window_utilization(session_id),
                "zone_transitions": self._conversation.get_zone_transitions(session_id),
            }, session_id=session_id)

        return ProxyResult(
            system_injection=system_injection,
            zones=zones,
            compaction_result=compaction_result,
            summarization_result=summarization_result,
            current_context=current_context,
        )

    async def proxy_post_process(
        self,
        assistant_text: str,
        session_id: str,
        turn_number: int = 0,
        current_context: dict | None = None,
    ) -> None:
        """Post-process after a proxied response completes.

        Delegates to the same PostLLMProcessor used by the runtime, giving
        bridge callers batched extraction, cursor tracking, conflict detection,
        and trace events for free.

        Auto-closes the trace turn when done.  Because the bridge fires this
        via ``create_task`` (fire-and-forget), the trace must be finalized
        here — ``proxy_end_turn()`` is never called externally.
        """
        turn = ConversationTurn(
            turn_number=turn_number,
            role="assistant",
            content=assistant_text,
        )
        try:
            # Trace: llm_response (bridge has the full text but no structured response)
            if self._trace:
                self._trace.emit("llm_response", {
                    "content_length": len(assistant_text),
                    "content_preview": assistant_text[:200],
                    "role": "assistant",
                }, session_id=session_id)

            await self._post_processor.process(
                turn,
                session_id,
                current_context=current_context,
            )

            # Trace: post-process results
            if self._trace:
                cr = self._post_processor.last_compaction_result
                sr = self._post_processor.last_summarization_result
                self._trace.emit("post_process", {
                    "memory_extraction": {
                        "memories_extracted": self._post_processor.last_memories_extracted,
                        "conflicts_detected": self._post_processor.last_conflicts_detected,
                    },
                    "compaction": {
                        "turns_compacted": cr.turns_compacted if cr else 0,
                        "original_tokens": cr.original_tokens if cr else 0,
                        "compacted_tokens": cr.compacted_tokens if cr else 0,
                    } if cr else None,
                    "summarization": {
                        "triggered": sr is not None,
                        "original_tokens": sr.original_tokens if sr else 0,
                        "summary_tokens": sr.summary_tokens if sr else 0,
                    } if sr else None,
                }, session_id=session_id)
        except Exception:
            logger.error("proxy_post_process failed", exc_info=True)
        finally:
            # Always close the trace turn — this is the last action in the
            # proxy pipeline.  Listeners (inspector log, etc.) fire here.
            if self._trace:
                self._trace.end_turn(session_id=session_id)

    def proxy_end_turn(self, duration_ms: float = 0.0, session_id: str | None = None) -> None:
        """End the current trace turn (proxy mode).

        .. deprecated::
            proxy_post_process() now auto-closes the trace.  This method is
            retained for callers that skip post-processing.
        """
        if self._trace:
            self._trace.end_turn(session_id=session_id)

    def reset_session(self, session_id: str) -> None:
        """Reset conversation state for a session (message edit/clear detected)."""
        self._conversation.destroy_session(session_id)

    async def _handle_budget_overflow(
        self,
        layers: dict[str, list[ResolvedContent]],
        budget: int,
        config: PipelineConfig,
        ctx: ResolverContext,
    ) -> dict[str, list[ResolvedContent]] | None:
        """Reduce context via compaction and summarization when over budget.

        Called by PipelineEngine._enforce_budget() before truncation.
        Finds the session layer, seeds the conversation manager if needed,
        runs compaction and summarization, then rebuilds the session content.
        """
        session_id = ctx.session_id or "default"

        # Find the session layer and item from config
        session_layer_name = None
        session_item_key = None
        for layer_cfg in config.layers:
            for item_cfg in layer_cfg.contents:
                if item_cfg.source == "session":
                    session_layer_name = layer_cfg.name
                    session_item_key = item_cfg.key
                    break
            if session_layer_name:
                break

        if not session_layer_name or session_layer_name not in layers:
            logger.debug("Budget overflow handler: no session layer found, skipping")
            return None

        # Find the session item in resolved layers
        session_item_idx = None
        for i, item in enumerate(layers[session_layer_name]):
            if item.key == session_item_key:
                session_item_idx = i
                break

        if session_item_idx is None:
            return None

        # Seed conversation manager from session history if zones are empty
        zones = self._conversation.zones(session_id)
        history = ctx.agent_config.get("session_history", [])
        if not zones.raw and not zones.compacted and history:
            for turn_num, msg in enumerate(history):
                msg_role = msg.get("role", "unknown")
                # Infer content_type from role so compaction rules can match
                content_type = None
                if msg_role == "tool":
                    content_type = "tool_output"
                turn = ConversationTurn(
                    turn_number=turn_num,
                    role=msg_role,
                    content=msg.get("content", ""),
                    content_type=content_type,
                    metadata=msg.get("metadata"),
                )
                self._conversation.add_turn(turn, session_id)

        # Phase 1: Run compaction
        self._conversation.run_compaction(session_id)

        # Phase 2: Check if we need summarization
        zones = self._conversation.zones(session_id)
        non_session_tokens = sum(
            c.tokens
            for name, contents in layers.items()
            for c in contents
            if not (name == session_layer_name and c.key == session_item_key)
        )
        zone_tokens = zones.total_tokens
        if zone_tokens + non_session_tokens > budget:
            await self._conversation.run_summarization(session_id)

        # Phase 3: Rebuild session content from zones
        zones = self._conversation.zones(session_id)
        parts = []
        for summary in zones.summarized:
            parts.append(f"[Previous conversation summary]\n{summary}")
        for turn in zones.compacted + zones.raw:
            parts.append(f"{turn.role}: {turn.content}")

        new_content = "\n".join(parts)
        new_tokens = estimate_tokens(new_content)

        # Replace session item in layers
        layers[session_layer_name][session_item_idx] = ResolvedContent(
            key=session_item_key,
            content=new_content,
            tokens=new_tokens,
            metadata=layers[session_layer_name][session_item_idx].metadata,
        )

        logger.info(
            "Budget overflow handler: reduced session content from %d to %d tokens",
            layers[session_layer_name][session_item_idx].tokens
            if session_item_idx < len(layers[session_layer_name])
            else 0,
            new_tokens,
        )

        return layers

    # --- Private helpers ---

    @staticmethod
    def _discover_interfaces(config_dir: str) -> dict[str, str]:
        """Discover interface pipeline configs from config_dir/interfaces/."""
        interfaces = {}
        iface_dir = os.path.join(config_dir, "interfaces")
        if os.path.isdir(iface_dir):
            for f in os.listdir(iface_dir):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    name = f.rsplit(".", 1)[0]
                    interfaces[name] = os.path.join(iface_dir, f)
        return interfaces

    def _resolve_pipeline(self, interface_name: str) -> str | None:
        """Get the pipeline config path for an interface from raw YAML."""
        iface_config = self._yaml_interfaces.get(interface_name, {})
        pipeline = iface_config.get("pipeline") if isinstance(iface_config, dict) else None
        if pipeline:
            full_path = os.path.join(self._config.config_dir, pipeline)
            if os.path.exists(full_path):
                return full_path
            return pipeline
        return None

    def _build_resolver_registry(self) -> ContentResolverRegistry:
        """Build a registry with standard resolvers."""
        from sr2.resolvers.retrieval_resolver import RetrievalResolver

        reg = ContentResolverRegistry()
        reg.register("config", ConfigResolver())
        reg.register("input", InputResolver())
        reg.register("session", SessionResolver())
        reg.register("runtime", RuntimeResolver())
        reg.register("static_template", StaticTemplateResolver())
        reg.register("retrieval", RetrievalResolver(self._retriever, self._matcher))
        return reg
