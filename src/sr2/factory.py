"""SR2Factory — extracted component-wiring logic from SR2.__init__.

SR2Factory.build(config) takes an SR2Config and returns a ComponentBundle
dataclass containing every wired component the SR2 facade needs. This
extraction lets us test wiring logic in isolation and simplifies the
SR2 facade to a thin delegation layer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sr2.sr2 import SR2Config

from sr2.cache.policies import create_default_cache_registry
from sr2.compaction.engine import CompactionEngine
from sr2.config.loader import ConfigLoader
from sr2.config.models import PipelineConfig, RetrievalConfig
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.dimensions import DimensionalMatcher
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.scope import ScopeDetector
from sr2.memory.store import InMemoryMemoryStore
from sr2.metrics.collector import MetricCollector
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.post_processor import PostLLMProcessor
from sr2.pipeline.router import InterfaceRouter
from sr2.pipeline.trace import TraceCollector
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ContentResolverRegistry
from sr2.resolvers.runtime_resolver import RuntimeResolver
from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.static_template_resolver import StaticTemplateResolver
from sr2.summarization.engine import SummarizationEngine

logger = logging.getLogger(__name__)


@dataclass
class ComponentBundle:
    """All wired components the SR2 facade needs."""

    # Pipeline
    engine: PipelineEngine
    conversation: ConversationManager
    post_processor: PostLLMProcessor
    router: InterfaceRouter
    resolver_registry: ContentResolverRegistry
    config: PipelineConfig
    token_budget: int

    # Memory stack
    memory_store: Any
    retriever: HybridRetriever
    matcher: DimensionalMatcher
    extractor: MemoryExtractor
    conflict_detector: ConflictDetector
    conflict_resolver: ConflictResolver

    # Metrics / observability
    collector: MetricCollector
    bridge: Any  # ContextBridge
    retrieval_config: RetrievalConfig
    yaml_interfaces: dict
    push_exporters: list = field(default_factory=list)

    # Optional
    scope_detector: ScopeDetector | None = None
    trace: TraceCollector | None = None
    scope_config: Any = None
    pull_exporter_name: str | None = None
    alerts: Any = None


class SR2Factory:
    """Factory that builds all SR2 components from an SR2Config."""

    @staticmethod
    def build(config: "SR2Config") -> ComponentBundle:
        """Build all components from config and return a ComponentBundle.

        Args:
            config: SR2Config with all settings and callables.

        Returns:
            ComponentBundle with every wired component.

        Raises:
            SR2ConfigurationError: If required callables are missing for enabled features.
        """

        trace = config.trace_collector

        # --- Memory stack ---
        memory_store = (
            config.memory_store if config.memory_store is not None else InMemoryMemoryStore()
        )
        retriever = HybridRetriever(
            store=memory_store,
            embedding_callable=config.embed,
            trace_collector=trace,
        )
        matcher = DimensionalMatcher()
        conflict_detector = ConflictDetector(store=memory_store)
        conflict_resolver = ConflictResolver(store=memory_store)
        extractor = MemoryExtractor(
            llm_callable=lambda prompt: (
                config.fast_complete(
                    "You extract structured memories from conversations.", prompt
                )
                if config.fast_complete
                else None
            ),
            store=memory_store,
            embed_callable=config.embed,
        )

        # --- Config ---
        loader = ConfigLoader(defaults_path=config.defaults_path)

        # Discover pipeline configs from config_dir/interfaces/
        pipeline_interfaces = _discover_interfaces(config.config_dir)

        # Register agent.yaml interface keys so routing works by interface name
        yaml_interfaces = config.agent_yaml.get("interfaces", {})
        for iface_name, iface_cfg in yaml_interfaces.items():
            if iface_name not in pipeline_interfaces and isinstance(iface_cfg, dict):
                pipeline = iface_cfg.get("pipeline")
                if pipeline:
                    full_path = os.path.join(config.config_dir, pipeline)
                    if os.path.exists(full_path):
                        pipeline_interfaces[iface_name] = full_path

        # Register agent-level pipeline config as "_default" fallback
        agent_pipeline = config.agent_yaml.get("pipeline")
        if agent_pipeline and isinstance(agent_pipeline, dict):
            if "_default" not in pipeline_interfaces:
                pipeline_interfaces["_default"] = agent_pipeline

        router = InterfaceRouter(interfaces=pipeline_interfaces, loader=loader)

        # --- Resolvers ---
        resolver_reg = _build_resolver_registry(retriever, matcher)
        if config.mcp_resource_reader:
            from sr2.resolvers.mcp_resource_resolver import MCPResourceResolver

            resolver_reg.register(
                "mcp_resource", MCPResourceResolver(config.mcp_resource_reader)
            )
        if config.mcp_prompt_reader:
            from sr2.resolvers.mcp_prompt_resolver import MCPPromptResolver

            resolver_reg.register("mcp_prompt", MCPPromptResolver(config.mcp_prompt_reader))
        if config.extra_resolvers:
            for source_name, resolver in config.extra_resolvers.items():
                resolver_reg.register(source_name, resolver)

        cache_reg = create_default_cache_registry()

        # --- Load pipeline config ---
        if config.preloaded_config is not None:
            agent_config = config.preloaded_config
        else:
            agent_yaml_path = os.path.join(config.config_dir, config.config_filename)
            agent_config = loader.load(agent_yaml_path)

        # --- Pipeline engine ---
        deg = agent_config.degradation
        engine = PipelineEngine(
            resolver_reg,
            cache_reg,
            circuit_breaker=CircuitBreaker(
                threshold=deg.circuit_breaker_threshold,
                cooldown_seconds=deg.circuit_breaker_cooldown_minutes * 60,
            ),
            trace_collector=trace,
        )

        # Wire retrieval enabled flag
        try:
            retrieval_resolver = resolver_reg.get("retrieval")
            retrieval_resolver.enabled = agent_config.retrieval.enabled
        except KeyError:
            pass

        # Wire key_schema, key_hint_limit
        key_schema = [s.model_dump() for s in agent_config.memory.key_schema]
        if key_schema:
            extractor._key_schema = key_schema
        extractor._key_hint_limit = agent_config.memory.key_hint_limit

        # Wire RetrievalConfig into retriever
        retrieval_config = agent_config.retrieval
        retriever._strategy = agent_config.retrieval.strategy
        retriever._top_k = agent_config.retrieval.top_k

        # Wire scope config
        scope_config = agent_config.memory.scope
        if scope_config:
            retriever._scope_config = scope_config
            extractor._scope_config = scope_config

        # Scope detector
        scope_detector: ScopeDetector | None = None
        if scope_config and config.fast_complete:
            scope_detector = ScopeDetector(
                store=memory_store,
                llm_callable=lambda prompt: config.fast_complete(
                    "You classify conversations into project scopes.", prompt
                ),
                scope_config=scope_config,
            )

        # --- Compaction + Summarization ---
        compaction_engine = CompactionEngine(agent_config.compaction)
        summarization_engine = SummarizationEngine(
            config=agent_config.summarization,
            llm_callable=lambda s, p: config.fast_complete(s, p) if config.fast_complete else None,
        )

        # --- Conversation manager ---
        conversation = ConversationManager(
            compaction_engine=compaction_engine,
            summarization_engine=summarization_engine,
            raw_window=agent_config.compaction.raw_window,
            compacted_max_tokens=agent_config.token_budget // 2,
            trace_collector=trace,
        )

        # --- Post-LLM processor ---
        post_processor = PostLLMProcessor(
            conversation_manager=conversation,
            memory_extractor=extractor if agent_config.memory.extract else None,
            conflict_detector=conflict_detector,
            conflict_resolver=conflict_resolver,
            retriever=retriever,
            trace_collector=trace,
        )

        # --- Metrics ---
        collector = MetricCollector(config.agent_yaml.get("name", "agent"))
        alerts = None

        # Observability plugins (config-driven)
        obs = agent_config.observability
        push_exporters: list = []
        for name in obs.push_exporters:
            try:
                from sr2.metrics.registry import get_push_exporter

                exporter_cls = get_push_exporter(name)
                push_exporters.append(exporter_cls(collector))
            except ImportError:
                logger.warning("Push exporter '%s' not available", name)

        pull_exporter_name = obs.pull_exporter

        if obs.alert_engine:
            try:
                from sr2.plugins.registry import PluginRegistry

                alert_reg: PluginRegistry = PluginRegistry(
                    "sr2.alerts", install_hint="pip install sr2-pro"
                )
                alert_cls = alert_reg.get(obs.alert_engine)
                alerts = alert_cls()
            except ImportError:
                logger.warning("Alert engine '%s' not available", obs.alert_engine)

        # --- Bridge ---
        from sr2.bridge import ContextBridge

        bridge = ContextBridge()

        # --- Validation ---
        _validate_callables(config, agent_config)

        return ComponentBundle(
            engine=engine,
            conversation=conversation,
            post_processor=post_processor,
            router=router,
            resolver_registry=resolver_reg,
            config=agent_config,
            token_budget=agent_config.token_budget,
            memory_store=memory_store,
            retriever=retriever,
            matcher=matcher,
            extractor=extractor,
            conflict_detector=conflict_detector,
            conflict_resolver=conflict_resolver,
            collector=collector,
            bridge=bridge,
            retrieval_config=retrieval_config,
            yaml_interfaces=yaml_interfaces,
            push_exporters=push_exporters,
            scope_detector=scope_detector,
            trace=trace,
            scope_config=scope_config,
            pull_exporter_name=pull_exporter_name,
            alerts=alerts,
        )


# --- Module-level helpers (extracted from SR2 private methods) ---


def _validate_callables(config: "SR2Config", pipeline_config: PipelineConfig) -> None:
    """Check that required callables are present when features are enabled."""
    from sr2.sr2 import SR2ConfigurationError

    errors = []
    if pipeline_config.memory.extract and config.fast_complete is None:
        errors.append("memory.extract is enabled but fast_complete callable is not provided")
    if pipeline_config.summarization.enabled and config.fast_complete is None:
        errors.append(
            "summarization.enabled is True but fast_complete callable is not provided"
        )
    if (
        pipeline_config.compaction.strategy in ("llm", "hybrid")
        and config.fast_complete is None
    ):
        errors.append(
            f"compaction.strategy is '{pipeline_config.compaction.strategy}' "
            "but fast_complete callable is not provided"
        )
    if (
        pipeline_config.retrieval.enabled
        and pipeline_config.retrieval.strategy in ("hybrid", "semantic")
        and config.embed is None
    ):
        errors.append(
            f"retrieval.strategy is '{pipeline_config.retrieval.strategy}' "
            "but embed callable is not provided"
        )
    if errors:
        raise SR2ConfigurationError("\n".join(errors))


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


def _build_resolver_registry(
    retriever: HybridRetriever,
    matcher: DimensionalMatcher,
) -> ContentResolverRegistry:
    """Build a registry with standard resolvers."""
    from sr2.resolvers.retrieval_resolver import RetrievalResolver

    reg = ContentResolverRegistry()
    reg.register("config", ConfigResolver())
    reg.register("input", InputResolver())
    reg.register("session", SessionResolver())
    reg.register("runtime", RuntimeResolver())
    reg.register("static_template", StaticTemplateResolver())
    reg.register("retrieval", RetrievalResolver(retriever, matcher))
    return reg
