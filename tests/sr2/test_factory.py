"""Tests for SR2Factory — extracted component-wiring logic from SR2.__init__.

SR2Factory.build(config) takes an SR2Config and returns a ComponentBundle
dataclass containing every wired component the SR2 facade needs. This
extraction lets us test wiring logic in isolation and simplifies the
SR2 facade to a thin delegation layer.

Tests verify:
  - Importability of SR2Factory and ComponentBundle
  - ComponentBundle field completeness
  - Default and custom memory store wiring
  - Trace collector propagation
  - Extra resolver registration
  - MCP resolver registration
  - Validation errors for missing callables
  - Backward compatibility (SR2(config) still works)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2.config.models import (
    MemoryConfig,
    PipelineConfig,
    RetrievalConfig,
    SummarizationConfig,
)
from sr2.sr2 import SR2, SR2Config, SR2ConfigurationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_pipeline_config(**overrides) -> PipelineConfig:
    """PipelineConfig with all LLM-dependent features disabled."""
    defaults = {
        "memory": MemoryConfig(extract=False),
        "summarization": SummarizationConfig(enabled=False),
        "retrieval": RetrievalConfig(enabled=False),
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _minimal_sr2_config(
    pipeline_config: PipelineConfig | None = None,
    fast_complete=None,
    embed=None,
    memory_store=None,
    trace_collector=None,
    extra_resolvers=None,
    mcp_resource_reader=None,
    mcp_prompt_reader=None,
) -> SR2Config:
    """Build a minimal SR2Config with a preloaded PipelineConfig."""
    return SR2Config(
        config_dir="/tmp",
        agent_yaml={"name": "test_agent"},
        fast_complete=fast_complete,
        embed=embed,
        memory_store=memory_store,
        trace_collector=trace_collector,
        extra_resolvers=extra_resolvers,
        mcp_resource_reader=mcp_resource_reader,
        mcp_prompt_reader=mcp_prompt_reader,
        preloaded_config=pipeline_config or _minimal_pipeline_config(),
    )


async def _dummy_fast_complete(system: str, prompt: str) -> str:
    return "ok"


async def _dummy_embed(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


class TestImportability:
    """SR2Factory and ComponentBundle must be importable from sr2.factory."""

    def test_sr2_factory_importable(self):
        from sr2.factory import SR2Factory  # noqa: F401

    def test_component_bundle_importable(self):
        from sr2.factory import ComponentBundle  # noqa: F401

    def test_sr2_factory_has_build_method(self):
        from sr2.factory import SR2Factory

        assert hasattr(SR2Factory, "build")
        assert callable(getattr(SR2Factory, "build"))


# ---------------------------------------------------------------------------
# 2. build() returns a ComponentBundle
# ---------------------------------------------------------------------------


class TestBuildReturnsBundle:
    """SR2Factory.build() must return a ComponentBundle instance."""

    def test_build_returns_component_bundle(self):
        from sr2.factory import SR2Factory, ComponentBundle

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)
        assert isinstance(bundle, ComponentBundle)


# ---------------------------------------------------------------------------
# 3. ComponentBundle has all required fields
# ---------------------------------------------------------------------------


class TestComponentBundleFields:
    """ComponentBundle must expose all components the SR2 facade needs."""

    REQUIRED_FIELDS = {
        "engine",
        "conversation",
        "post_processor",
        "memory_store",
        "retriever",
        "matcher",
        "extractor",
        "conflict_detector",
        "conflict_resolver",
        "router",
        "bridge",
        "collector",
        "token_budget",
        "config",
        "retrieval_config",
        "yaml_interfaces",
        "push_exporters",
        "resolver_registry",
    }

    OPTIONAL_FIELDS = {
        "scope_detector",
        "trace",
        "scope_config",
        "pull_exporter_name",
        "alerts",
    }

    def test_all_required_fields_present(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)

        for field_name in self.REQUIRED_FIELDS:
            assert hasattr(bundle, field_name), (
                f"ComponentBundle missing required field: {field_name}"
            )

    def test_all_optional_fields_present(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)

        for field_name in self.OPTIONAL_FIELDS:
            assert hasattr(bundle, field_name), (
                f"ComponentBundle missing optional field: {field_name}"
            )

    def test_component_types(self):
        """Key components have correct types."""
        from sr2.factory import SR2Factory
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.conversation import ConversationManager
        from sr2.pipeline.post_processor import PostLLMProcessor
        from sr2.memory.retrieval import HybridRetriever
        from sr2.memory.dimensions import DimensionalMatcher
        from sr2.memory.extraction import MemoryExtractor
        from sr2.memory.conflicts import ConflictDetector
        from sr2.memory.resolution import ConflictResolver
        from sr2.pipeline.router import InterfaceRouter
        from sr2.bridge import ContextBridge
        from sr2.metrics.collector import MetricCollector

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)

        assert isinstance(bundle.engine, PipelineEngine)
        assert isinstance(bundle.conversation, ConversationManager)
        assert isinstance(bundle.post_processor, PostLLMProcessor)
        assert isinstance(bundle.retriever, HybridRetriever)
        assert isinstance(bundle.matcher, DimensionalMatcher)
        assert isinstance(bundle.extractor, MemoryExtractor)
        assert isinstance(bundle.conflict_detector, ConflictDetector)
        assert isinstance(bundle.conflict_resolver, ConflictResolver)
        assert isinstance(bundle.router, InterfaceRouter)
        assert isinstance(bundle.bridge, ContextBridge)
        assert isinstance(bundle.collector, MetricCollector)

    def test_token_budget_is_int(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)
        assert isinstance(bundle.token_budget, int)
        assert bundle.token_budget > 0

    def test_config_is_pipeline_config(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)
        assert isinstance(bundle.config, PipelineConfig)

    def test_yaml_interfaces_is_dict(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)
        assert isinstance(bundle.yaml_interfaces, dict)

    def test_push_exporters_is_list(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config()
        bundle = SR2Factory.build(config)
        assert isinstance(bundle.push_exporters, list)


# ---------------------------------------------------------------------------
# 4. Default memory store is InMemoryMemoryStore
# ---------------------------------------------------------------------------


class TestDefaultMemoryStore:
    """When no memory_store is provided, InMemoryMemoryStore is used."""

    def test_default_store_is_in_memory(self):
        from sr2.factory import SR2Factory
        from sr2.memory.store import InMemoryMemoryStore

        config = _minimal_sr2_config(memory_store=None)
        bundle = SR2Factory.build(config)
        assert isinstance(bundle.memory_store, InMemoryMemoryStore)


# ---------------------------------------------------------------------------
# 5. Custom memory store is used when provided
# ---------------------------------------------------------------------------


class TestCustomMemoryStore:
    """When a memory_store is provided in SR2Config, it is used directly."""

    def test_custom_store_is_propagated(self):
        from sr2.factory import SR2Factory

        custom_store = MagicMock()
        config = _minimal_sr2_config(memory_store=custom_store)
        bundle = SR2Factory.build(config)
        assert bundle.memory_store is custom_store

    def test_custom_store_wired_to_retriever(self):
        """The custom store must be wired into the retriever."""
        from sr2.factory import SR2Factory

        custom_store = MagicMock()
        config = _minimal_sr2_config(memory_store=custom_store)
        bundle = SR2Factory.build(config)
        assert bundle.retriever._store is custom_store

    def test_custom_store_wired_to_conflict_detector(self):
        """The custom store must be wired into the conflict detector."""
        from sr2.factory import SR2Factory

        custom_store = MagicMock()
        config = _minimal_sr2_config(memory_store=custom_store)
        bundle = SR2Factory.build(config)
        assert bundle.conflict_detector._store is custom_store


# ---------------------------------------------------------------------------
# 6. Trace collector is wired when provided
# ---------------------------------------------------------------------------


class TestTraceCollectorWiring:
    """When trace_collector is provided, it propagates to relevant components."""

    def test_trace_stored_on_bundle(self):
        from sr2.factory import SR2Factory
        from sr2.pipeline.trace import TraceCollector

        trace = TraceCollector()
        config = _minimal_sr2_config(trace_collector=trace)
        bundle = SR2Factory.build(config)
        assert bundle.trace is trace

    def test_trace_none_when_not_provided(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config(trace_collector=None)
        bundle = SR2Factory.build(config)
        assert bundle.trace is None


# ---------------------------------------------------------------------------
# 7. Extra resolvers are registered
# ---------------------------------------------------------------------------


class TestExtraResolvers:
    """Extra resolvers from SR2Config are registered in the resolver registry."""

    def test_extra_resolver_registered(self):
        from sr2.factory import SR2Factory

        mock_resolver = MagicMock()
        config = _minimal_sr2_config(
            extra_resolvers={"custom_source": mock_resolver},
        )
        bundle = SR2Factory.build(config)

        # The resolver registry (inside the engine) should have the custom source
        # We verify by checking the resolver_reg can resolve it
        resolver = bundle.resolver_registry.get("custom_source")
        assert resolver is mock_resolver

    def test_multiple_extra_resolvers_registered(self):
        from sr2.factory import SR2Factory

        resolver_a = MagicMock()
        resolver_b = MagicMock()
        config = _minimal_sr2_config(
            extra_resolvers={"source_a": resolver_a, "source_b": resolver_b},
        )
        bundle = SR2Factory.build(config)

        assert bundle.resolver_registry.get("source_a") is resolver_a
        assert bundle.resolver_registry.get("source_b") is resolver_b

    def test_no_extra_resolvers_is_fine(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config(extra_resolvers=None)
        bundle = SR2Factory.build(config)
        # Standard resolvers still registered
        assert bundle.resolver_registry.get("config") is not None
        assert bundle.resolver_registry.get("input") is not None
        assert bundle.resolver_registry.get("session") is not None


# ---------------------------------------------------------------------------
# 8. SR2ConfigurationError: memory extraction without fast_complete
# ---------------------------------------------------------------------------


class TestValidationMemoryExtraction:
    """Memory extraction enabled without fast_complete must raise."""

    def test_extract_enabled_no_fast_complete_raises(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config(
            pipeline_config=_minimal_pipeline_config(
                memory=MemoryConfig(extract=True),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2Factory.build(config)

        assert "memory.extract" in str(exc_info.value)
        assert "fast_complete" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. SR2ConfigurationError: summarization without fast_complete
# ---------------------------------------------------------------------------


class TestValidationSummarization:
    """Summarization enabled without fast_complete must raise."""

    def test_summarization_enabled_no_fast_complete_raises(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config(
            pipeline_config=_minimal_pipeline_config(
                summarization=SummarizationConfig(enabled=True),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2Factory.build(config)

        assert "summarization.enabled" in str(exc_info.value)
        assert "fast_complete" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10. MCP resolvers registered when readers are provided
# ---------------------------------------------------------------------------


class TestMCPResolverRegistration:
    """MCP resolvers are registered when mcp_resource_reader / mcp_prompt_reader provided."""

    def test_mcp_resource_resolver_registered(self):
        from sr2.factory import SR2Factory

        reader = AsyncMock()
        config = _minimal_sr2_config(mcp_resource_reader=reader)
        bundle = SR2Factory.build(config)

        resolver = bundle.resolver_registry.get("mcp_resource")
        assert resolver is not None

    def test_mcp_prompt_resolver_registered(self):
        from sr2.factory import SR2Factory

        reader = AsyncMock()
        config = _minimal_sr2_config(mcp_prompt_reader=reader)
        bundle = SR2Factory.build(config)

        resolver = bundle.resolver_registry.get("mcp_prompt")
        assert resolver is not None

    def test_no_mcp_resolvers_when_readers_absent(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config(
            mcp_resource_reader=None,
            mcp_prompt_reader=None,
        )
        bundle = SR2Factory.build(config)

        with pytest.raises(KeyError):
            bundle.resolver_registry.get("mcp_resource")

        with pytest.raises(KeyError):
            bundle.resolver_registry.get("mcp_prompt")

    def test_both_mcp_resolvers_registered(self):
        from sr2.factory import SR2Factory

        resource_reader = AsyncMock()
        prompt_reader = AsyncMock()
        config = _minimal_sr2_config(
            mcp_resource_reader=resource_reader,
            mcp_prompt_reader=prompt_reader,
        )
        bundle = SR2Factory.build(config)

        assert bundle.resolver_registry.get("mcp_resource") is not None
        assert bundle.resolver_registry.get("mcp_prompt") is not None


# ---------------------------------------------------------------------------
# 11. Backward compatibility: SR2(config) still works
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Creating SR2 via SR2(config) must still work after the factory extraction."""

    def test_sr2_init_still_works(self):
        """SR2(config) should succeed and produce a functional instance."""
        config = _minimal_sr2_config()
        sr2 = SR2(config)
        assert sr2 is not None

    def test_sr2_has_expected_attributes(self):
        """SR2 instance should still expose the same internal components."""
        config = _minimal_sr2_config()
        sr2 = SR2(config)

        # Key attributes that the runtime relies on
        assert hasattr(sr2, "_engine")
        assert hasattr(sr2, "_conversation")
        assert hasattr(sr2, "_post_processor")
        assert hasattr(sr2, "_memory_store")
        assert hasattr(sr2, "_retriever")
        assert hasattr(sr2, "_collector")
        assert hasattr(sr2, "_bridge")
        assert hasattr(sr2, "_router")
        assert hasattr(sr2, "_token_budget")

    def test_sr2_public_methods_still_exist(self):
        """SR2's public API methods must still be present after refactoring."""
        assert callable(getattr(SR2, "process", None))
        assert callable(getattr(SR2, "post_process", None))
        assert callable(getattr(SR2, "save_memory", None))
        assert callable(getattr(SR2, "collect_metrics", None))
        assert callable(getattr(SR2, "get_zones", None))
        assert callable(getattr(SR2, "get_raw_window", None))
        assert callable(getattr(SR2, "set_memory_store", None))


# ---------------------------------------------------------------------------
# Scope detector wiring
# ---------------------------------------------------------------------------


class TestScopeDetectorWiring:
    """ScopeDetector is created only when scope config + fast_complete both present."""

    def test_scope_detector_none_without_scope_config(self):
        from sr2.factory import SR2Factory

        config = _minimal_sr2_config(
            fast_complete=_dummy_fast_complete,
        )
        bundle = SR2Factory.build(config)
        assert bundle.scope_detector is None

    def test_scope_detector_none_without_fast_complete(self):
        from sr2.factory import SR2Factory
        from sr2.config.models import MemoryScopeConfig

        config = _minimal_sr2_config(
            pipeline_config=_minimal_pipeline_config(
                memory=MemoryConfig(
                    extract=False,
                    scope=MemoryScopeConfig(
                        allowed_read=["private"],
                        allowed_write=["private"],
                        default_write="private",
                        agent_name="test",
                    ),
                ),
            ),
            fast_complete=None,
        )
        bundle = SR2Factory.build(config)
        assert bundle.scope_detector is None

    def test_scope_detector_created_with_both(self):
        from sr2.factory import SR2Factory
        from sr2.config.models import MemoryScopeConfig
        from sr2.memory.scope import ScopeDetector

        config = _minimal_sr2_config(
            pipeline_config=_minimal_pipeline_config(
                memory=MemoryConfig(
                    extract=False,
                    scope=MemoryScopeConfig(
                        allowed_read=["private"],
                        allowed_write=["private"],
                        default_write="private",
                        agent_name="test",
                    ),
                ),
            ),
            fast_complete=_dummy_fast_complete,
        )
        bundle = SR2Factory.build(config)
        assert isinstance(bundle.scope_detector, ScopeDetector)


# ---------------------------------------------------------------------------
# Retrieval config wiring
# ---------------------------------------------------------------------------


class TestRetrievalConfigWiring:
    """Retrieval config from PipelineConfig is propagated to the bundle."""

    def test_retrieval_config_propagated(self):
        from sr2.factory import SR2Factory

        retrieval = RetrievalConfig(enabled=False, strategy="keyword", top_k=5)
        config = _minimal_sr2_config(
            pipeline_config=_minimal_pipeline_config(retrieval=retrieval),
        )
        bundle = SR2Factory.build(config)
        assert bundle.retrieval_config.strategy == "keyword"
        assert bundle.retrieval_config.top_k == 5
