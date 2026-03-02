"""SR2 facade — single entry point for the runtime into the context-engineering pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from sr2.cache.policies import create_default_cache_registry
from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.loader import ConfigLoader
from sr2.config.models import LLMModelOverride
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.dimensions import DimensionalMatcher
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.store import InMemoryMemoryStore
from sr2.metrics.alerts import AlertRuleEngine
from sr2.metrics.collector import MetricCollector
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.post_processor import PostLLMProcessor
from sr2.pipeline.prefix_tracker import PrefixSnapshot
from sr2.pipeline.result import PipelineResult
from sr2.pipeline.router import InterfaceRouter
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext
from sr2.resolvers.runtime_resolver import RuntimeResolver
from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.static_template_resolver import StaticTemplateResolver
from sr2.summarization.engine import SummarizationEngine
from sr2.tools.models import ToolDefinition, ToolManagementConfig
from sr2.tools.state_machine import ToolStateMachine

# Lazy import to avoid hard asyncpg dependency
# from sr2.memory.store import PostgresMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class SR2Config:
    """Configuration for the SR2 facade."""

    config_dir: str
    defaults_path: str
    agent_yaml: dict
    fast_complete: Callable | None = None  # async (system, prompt) -> str
    embed: Callable | None = None  # async (text) -> list[float]
    mcp_resource_reader: Callable | None = None  # async (uri, server_name) -> str
    mcp_prompt_reader: Callable | None = None  # async (name, arguments, server_name) -> str


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


class SR2:
    """Facade over the SR2 context-engineering pipeline.

    Runtime creates one instance and calls process() per trigger.
    All internal wiring (memory, resolvers, pipeline, compaction,
    metrics) is hidden behind this interface.
    """

    def __init__(self, config: SR2Config) -> None:
        self._config = config

        # Memory stack
        self._memory_store = InMemoryMemoryStore()
        self._retriever = HybridRetriever(
            store=self._memory_store,
            embedding_callable=config.embed,
        )
        self._matcher = DimensionalMatcher()
        self._conflict_detector = ConflictDetector(store=self._memory_store)
        self._conflict_resolver = ConflictResolver(store=self._memory_store)
        self._extractor = MemoryExtractor(
            llm_callable=lambda prompt: config.fast_complete(
                "You extract structured memories from conversations.", prompt
            ) if config.fast_complete else None,
            store=self._memory_store,
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
            self._resolver_reg.register("mcp_resource", MCPResourceResolver(config.mcp_resource_reader))
        if config.mcp_prompt_reader:
            from sr2.resolvers.mcp_prompt_resolver import MCPPromptResolver
            self._resolver_reg.register("mcp_prompt", MCPPromptResolver(config.mcp_prompt_reader))
        self._cache_reg = create_default_cache_registry()

        # Pipeline
        self._engine = PipelineEngine(self._resolver_reg, self._cache_reg)

        # Compaction + Summarization
        agent_yaml_path = os.path.join(config.config_dir, "agent.yaml")
        agent_config = self._loader.load(agent_yaml_path)
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
        )

        # Post-LLM processor
        self._post_processor = PostLLMProcessor(
            conversation_manager=self._conversation,
            memory_extractor=self._extractor,
            conflict_detector=self._conflict_detector,
            conflict_resolver=self._conflict_resolver,
        )

        # Metrics
        self._collector = MetricCollector(config.agent_yaml.get("name", "agent"))
        self._alerts = AlertRuleEngine()

        # OpenTelemetry (optional — only if otel extras installed)
        try:
            from sr2.metrics.otel_exporter import OTelExporter
            self._otel = OTelExporter(self._collector)
        except ImportError:
            self._otel = None

        # Internal bridge for build_messages only
        from runtime.llm import ContextBridge
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

        # Compile context
        ctx = ResolverContext(
            agent_config=agent_context,
            trigger_input=trigger_input,
            session_id=session_id,
            interface_type=interface_name,
        )
        compiled = await self._engine.compile(config, ctx)
        logger.info(f"SR2.process: context compiled for {interface_name}")

        # Build messages
        messages = self._bridge.build_messages(
            compiled, session_turns, str(trigger_input) if trigger_input else None
        )

        # Build tool state machine from pipeline config
        tool_defs = [
            ToolDefinition(
                name=t["name"],
                description=t.get("description", ""),
                raw_parameters=t.get("parameters"),
            )
            for t in tool_schemas
        ]
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
    ) -> None:
        """Fire-and-forget post-LLM processing (memory extraction, compaction)."""
        try:
            turn = ConversationTurn(
                turn_number=turn_number,
                role=role,
                content=content,
            )
            await self._post_processor.process(turn, session_id)
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")

    def set_memory_store(self, store) -> None:
        """Rewire all memory components to use a new store."""
        self._memory_store = store
        self._retriever._store = store
        self._conflict_detector._store = store
        self._conflict_resolver._store = store
        self._extractor._store = store

    async def set_postgres_store(self, pool) -> None:
        """Late-bind a PostgreSQL memory store.

        Imports PostgresMemoryStore lazily to avoid hard asyncpg dependency.
        """
        from sr2.memory.store import PostgresMemoryStore

        store = PostgresMemoryStore(pool)
        await store.create_tables()
        self.set_memory_store(store)

    def collect_metrics(
        self,
        pipeline_result: PipelineResult,
        interface: str,
        loop_iterations: int,
        loop_total_tokens: int,
        loop_tool_calls: int,
        loop_cache_hit_rate: float,
        cache_report=None,
    ) -> None:
        """Collect metrics from a loop execution."""
        extra = {
            "sr2_loop_iterations": loop_iterations,
            "sr2_loop_total_tokens": loop_total_tokens,
            "sr2_loop_tool_calls": loop_tool_calls,
            "sr2_cache_hit_rate": loop_cache_hit_rate,
        }
        if cache_report is not None:
            extra["sr2_context_prefix_stable"] = (
                1.0 if cache_report.prefix_stable else 0.0
            )
            extra["sr2_cache_efficiency"] = cache_report.cache_efficiency

        self._collector.collect(pipeline_result, interface, extra_metrics=extra)

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
        """Export collected metrics in Prometheus text exposition format."""
        from sr2.metrics.exporter import PrometheusExporter

        if not hasattr(self, "_exporter"):
            self._exporter = PrometheusExporter(self._collector)
        return self._exporter.export()

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
