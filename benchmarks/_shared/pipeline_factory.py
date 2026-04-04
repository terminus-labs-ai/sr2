"""Factory to wire up real SR2 components for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from sr2.cache.policies import create_default_cache_registry
from sr2.cache.registry import CachePolicyRegistry
from sr2.compaction.engine import CompactionEngine
from sr2.config.models import (
    CompactionConfig,
    CompactionRuleConfig,
    LayerConfig,
    PipelineConfig,
    SummarizationConfig,
)
from sr2.metrics.collector import MetricCollector
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ContentResolverRegistry
from sr2.resolvers.session_resolver import SessionResolver
from sr2.summarization.engine import SummarizationEngine


@dataclass
class BenchmarkPipeline:
    """All the wired-up SR2 components for a benchmark run."""

    engine: PipelineEngine
    conversation_manager: ConversationManager
    compaction_engine: CompactionEngine
    config: PipelineConfig
    metric_collector: MetricCollector
    cache_registry: CachePolicyRegistry
    resolver_registry: ContentResolverRegistry


def _build_resolver_registry() -> ContentResolverRegistry:
    reg = ContentResolverRegistry()
    reg.register("config", ConfigResolver())
    reg.register("input", InputResolver())
    reg.register("session", SessionResolver())
    return reg


def _build_compaction_config(
    compaction_rules: list[CompactionRuleConfig] | None,
    raw_window: int,
) -> CompactionConfig:
    if compaction_rules is None:
        compaction_rules = [
            CompactionRuleConfig(
                type="tool_output",
                strategy="schema_and_sample",
                max_compacted_tokens=80,
                recovery_hint=True,
            )
        ]
    return CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=100,
        rules=compaction_rules,
    )


def _build_pipeline_config(
    token_budget: int,
    compaction_config: CompactionConfig,
) -> PipelineConfig:
    return PipelineConfig(
        token_budget=token_budget,
        compaction=compaction_config,
        layers=[
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[
                    {"key": "system_prompt", "source": "config"},
                    {"key": "tool_definitions", "source": "config"},
                ],
            ),
            LayerConfig(
                name="memory",
                cache_policy="refresh_on_topic_shift",
                contents=[
                    {"key": "retrieved_memories", "source": "config"},
                ],
            ),
            LayerConfig(
                name="conversation",
                cache_policy="append_only",
                contents=[
                    {"key": "session_history", "source": "session"},
                ],
            ),
        ],
    )


def create_benchmark_pipeline(
    compaction_rules: list[CompactionRuleConfig] | None = None,
    raw_window: int = 5,
    compacted_max_tokens: int = 6000,
    token_budget: int = 32000,
    llm_callable=None,
) -> BenchmarkPipeline:
    """Create a fully wired benchmark pipeline using real SR2 components.

    Args:
        compaction_rules: Compaction rule configs. Default: tool_output -> schema_and_sample.
        raw_window: Number of recent turns kept verbatim.
        compacted_max_tokens: Max tokens in the compacted zone before summarization.
        token_budget: Total token budget for the context window.
        llm_callable: async function(system: str, prompt: str) -> str, for summarization.
    """
    resolver_registry = _build_resolver_registry()
    cache_registry = create_default_cache_registry()

    compaction_config = _build_compaction_config(compaction_rules, raw_window)
    compaction_engine = CompactionEngine(compaction_config)

    summarization_engine = None
    if llm_callable is not None:
        summarization_engine = SummarizationEngine(
            config=SummarizationConfig(enabled=True),
            llm_callable=llm_callable,
        )

    conversation_manager = ConversationManager(
        compaction_engine=compaction_engine,
        summarization_engine=summarization_engine,
        raw_window=raw_window,
        compacted_max_tokens=compacted_max_tokens,
    )

    pipeline_engine = PipelineEngine(resolver_registry, cache_registry)
    pipeline_config = _build_pipeline_config(token_budget, compaction_config)
    metric_collector = MetricCollector("benchmark")

    return BenchmarkPipeline(
        engine=pipeline_engine,
        conversation_manager=conversation_manager,
        compaction_engine=compaction_engine,
        config=pipeline_config,
        metric_collector=metric_collector,
        cache_registry=cache_registry,
        resolver_registry=resolver_registry,
    )
