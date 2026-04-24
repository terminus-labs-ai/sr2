"""SR2 facade — single entry point for the runtime into the context-engineering pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from sr2.compaction.engine import ConversationTurn
from sr2.config.models import PipelineConfig
from sr2.memory.registry import get_store
from sr2.metrics.manager import MetricSources, MetricsManager
from sr2.pipeline.prefix_tracker import PrefixSnapshot
from sr2.pipeline.result import ActualTokenUsage, PipelineResult
from sr2.resolvers.registry import (
    ResolverContext,
    ResolvedContent,
    estimate_tokens,
)
from sr2.pipeline.trace import TraceCollector
from sr2.tools.models import ToolDefinition, ToolManagementConfig
from sr2.tools.state_machine import ToolStateMachine

logger = logging.getLogger(__name__)


class SR2ConfigurationError(Exception):
    """Raised when SR2 config enables features but required callables are missing."""
    pass


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
    compiled_snapshot: PrefixSnapshot | None
    pipeline_result: PipelineResult


class SR2:
    """Facade over the SR2 context-engineering pipeline.

    Runtime creates one instance and calls process() per trigger.
    All internal wiring (memory, resolvers, pipeline, compaction,
    metrics) is hidden behind this interface.
    """

    def __init__(self, config: SR2Config) -> None:
        from sr2.factory import SR2Factory

        self._config = config

        # Build all components via the factory
        bundle = SR2Factory.build(config)

        # Unpack ComponentBundle onto self
        self._trace = bundle.trace
        self._memory_store = bundle.memory_store
        self._retriever = bundle.retriever
        self._matcher = bundle.matcher
        self._conflict_detector = bundle.conflict_detector
        self._conflict_resolver = bundle.conflict_resolver
        self._extractor = bundle.extractor
        self._router = bundle.router
        self._resolver_reg = bundle.resolver_registry
        self._engine = bundle.engine
        self._retrieval_config = bundle.retrieval_config
        self._scope_config_resolved = bundle.scope_config
        self._scope_detector = bundle.scope_detector
        self._conversation = bundle.conversation
        self._post_processor = bundle.post_processor
        self._collector = bundle.collector
        self._alerts = bundle.alerts
        self._token_budget = bundle.token_budget
        self._push_exporters = bundle.push_exporters
        self._pull_exporter_name = bundle.pull_exporter_name
        self._bridge = bundle.bridge
        self._yaml_interfaces = bundle.yaml_interfaces

        # Wire budget overflow handler into engine (needs self reference)
        self._engine._budget_overflow_handler = self._handle_budget_overflow

        # Build MetricsManager — owns all session tracking state and metrics methods
        metric_sources = MetricSources(
            conversation=self._conversation,
            engine=self._engine,
            post_processor=self._post_processor,
            retriever=self._retriever,
            memory_store=self._memory_store,
            trace=self._trace,
        )
        self._metrics_manager = MetricsManager(
            collector=self._collector,
            sources=metric_sources,
            token_budget=self._token_budget,
            alerts=self._alerts,
            push_exporters=self._push_exporters,
            pull_exporter_name=self._pull_exporter_name,
        )

    def get_raw_window(self, interface_name: str | None = None) -> int:
        """Return the compaction raw_window for the given interface.

        Looks up the interface's PipelineConfig via the router and returns
        its compaction.raw_window.  Falls back to the base pipeline
        raw_window (from ConversationManager) when:
        - interface_name is None
        - the interface is not registered in the router
        """
        if interface_name is not None:
            try:
                config = self._router.route(interface_name)
                return config.compaction.raw_window
            except KeyError:
                pass
        return self._conversation.raw_window

    def reload_interface(self, name: str) -> PipelineConfig:
        """Invalidate cached config for *name* and re-load from disk.

        Delegates to InterfaceRouter.reload_interface().
        """
        return self._router.reload_interface(name)

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

        # Build agent context dict
        agent_context = {
            "system_prompt": system_prompt,
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

        # Seed conversation zones from session history (idempotent)
        seeded = self._conversation.seed_from_history(session_turns, session_id)
        if seeded:
            logger.info(
                "SR2.process: seeded %d turns into zones for session=%s",
                seeded, session_id,
            )

        # Run proactive compaction + summarization before compilation
        self._conversation.run_compaction(
            session_id,
            token_budget=self._token_budget,
            current_tokens=self._conversation.zones(session_id).total_tokens,
        )

        # Run summarization to collapse compacted turns into summaries.
        # Force when compacted zone has many turns — the turn count itself
        # can overwhelm the LLM even if individual turns are small.
        zones = self._conversation.zones(session_id)
        force_summarize = len(zones.compacted) > self._conversation.raw_window * 4
        if zones.compacted:
            await self._conversation.run_summarization(session_id, force=force_summarize)

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

        # Stash compiled token estimate for drift calibration in report_actual_usage()
        self._metrics_manager.record_compiled_tokens(session_id, compiled.tokens)

        # Build messages from zones (compacted/summarized history)
        zones = self._conversation.zones(session_id)
        messages = self._bridge.build_messages_from_zones(
            compiled, zones,
            current_input=str(trigger_input) if trigger_input else None,
            skip_session_layer="session",
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

        # Compute session prefix budget for post-process compaction
        if self._post_processor:
            session_layer = None
            for layer_cfg in config.layers:
                for item_cfg in layer_cfg.contents:
                    if item_cfg.source == "session":
                        session_layer = layer_cfg.name
                        break
                if session_layer:
                    break
            budget = self._engine.session_prefix_tokens(session_layer) if session_layer else 0
            self._post_processor.set_prefix_budget(budget)
            self._post_processor.set_budget_info(
                token_budget=self._token_budget,
                current_tokens=compiled.tokens,
            )

        return ProcessedContext(
            messages=messages,
            tool_schemas=filtered_schemas,
            tool_choice=tool_choice,
            state_machine=state_machine,
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
        model_hint: str | None = None,
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
                model_hint=model_hint,
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
        """Collect metrics from a loop execution. Delegates to MetricsManager."""
        await self._metrics_manager.collect(
            pipeline_result=pipeline_result,
            interface=interface,
            loop_iterations=loop_iterations,
            loop_total_tokens=loop_total_tokens,
            loop_tool_calls=loop_tool_calls,
            loop_cache_hit_rate=loop_cache_hit_rate,
            cache_report=cache_report,
            session_id=session_id,
            session_messages=session_messages,
            session_turn_count=session_turn_count,
            session_created_at=session_created_at,
            tool_state_machine=tool_state_machine,
        )

    def report_actual_usage(
        self,
        usage: ActualTokenUsage,
        session_id: str = "default",
    ) -> None:
        """Report actual token usage from the LLM provider. Delegates to MetricsManager."""
        self._metrics_manager.report_actual_usage(usage, session_id=session_id)

    def estimate_drift(self, session_id: str) -> float | None:
        """Rolling average of (actual - estimated) / actual. Delegates to MetricsManager."""
        return self._metrics_manager.estimate_drift(session_id)

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

    # --- Public API: conversation zones (replaces bridge private attribute access) ---

    def get_zones(self, session_id: str = "default"):
        """Get conversation zones for a session.

        Returns the ConversationZones (summarized/compacted/raw) for the given
        session_id. Used by the bridge for persistence and metrics.
        """
        return self._conversation.zones(session_id)

    def get_zone_transitions(self, session_id: str = "default") -> dict[str, int]:
        """Get cumulative zone transition counts for a session.

        Returns a copy of the transition count dict so callers cannot mutate
        internal state.
        """
        return self._conversation.get_zone_transitions(session_id)

    def restore_zones(self, session_id: str, zones) -> None:
        """Restore previously persisted conversation zones.

        Used by the bridge at startup to reload zones from its persistence
        layer without reaching into SR2 private attributes.
        """
        self._conversation.restore_zones(session_id, zones)

    # --- Public API: circuit breaker / degradation ---

    def is_circuit_breaker_open(self, feature: str) -> bool:
        """Check if the circuit breaker is open for a feature.

        Returns False for unknown feature names (closed by default).
        """
        return self._engine._circuit_breaker.is_open(feature)

    def get_circuit_breaker_status(self) -> dict:
        """Return circuit breaker status for all tracked stages.

        Returns a dict of {stage_name: {"failures": int, "is_open": bool, ...}}.
        Empty when no failures have been recorded.
        """
        return self._engine._circuit_breaker.status()

    def get_degradation_level(self) -> str:
        """Degradation level based on circuit breaker state.

        Returns:
            'full'            — all breakers closed
            'compaction_only' — summarization OR memory_extraction is open
            'passthrough'     — both summarization AND memory_extraction are open
        """
        summarization_open = self._engine._circuit_breaker.is_open("summarization")
        memory_open = self._engine._circuit_breaker.is_open("memory_extraction")
        if summarization_open and memory_open:
            return "passthrough"
        if summarization_open or memory_open:
            return "compaction_only"
        return "full"

    def export_metrics(self) -> str:
        """Export collected metrics in Prometheus text exposition format. Delegates to MetricsManager."""
        return self._metrics_manager.export_metrics()

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
        history = ctx.agent_config.get("session_history", [])
        self._conversation.seed_from_history(history, session_id)

        # Phase 1: Run compaction
        session_prefix = self._engine.session_prefix_tokens(session_layer_name)
        total_tokens = sum(c.tokens for contents in layers.values() for c in contents)
        self._conversation.run_compaction(
            session_id,
            prefix_budget=session_prefix,
            token_budget=budget,
            current_tokens=total_tokens,
        )

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
