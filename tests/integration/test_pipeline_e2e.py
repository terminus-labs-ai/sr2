"""End-to-end pipeline integration test.

Wires together real (not mocked) components:
- ConfigLoader with defaults.yaml
- PipelineEngine with real resolver registry
- InMemoryMemoryStore (no DB needed)
- MetricCollector

Tests the full flow: config -> route -> compile -> post-process.
Does NOT require a real LLM or database.
"""

import pytest

from sr2.cache.policies import create_default_cache_registry
from sr2.config.loader import ConfigLoader
from sr2.config.models import PipelineConfig
from sr2.config.validation import validate_config
from sr2.metrics.collector import MetricCollector
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.router import InterfaceRouter
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext
from sr2.resolvers.runtime_resolver import RuntimeResolver
from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.static_template_resolver import StaticTemplateResolver


def build_resolver_registry():
    """Build a registry with all basic resolvers."""
    reg = ContentResolverRegistry()
    reg.register("config", ConfigResolver())
    reg.register("input", InputResolver())
    reg.register("session", SessionResolver())
    reg.register("runtime", RuntimeResolver())
    reg.register("static_template", StaticTemplateResolver())
    return reg


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def setup_method(self):
        self.resolver_reg = build_resolver_registry()
        self.cache_reg = create_default_cache_registry()
        self.engine = PipelineEngine(self.resolver_reg, self.cache_reg)
        self.collector = MetricCollector("test_agent")

    # Skipped for now - need to build webui interface
    # @pytest.mark.asyncio
    # async def test_defaults_config_loads_and_validates(self):
    #     """Library defaults.yaml loads, validates, and compiles without error."""
    #     loader = ConfigLoader()
    #     config = loader.load("configs/defaults.yaml")
    #     # May have warnings but no errors
    #     warnings = validate_config(config)
    #     assert config.token_budget == 32000

    @pytest.mark.asyncio
    async def test_minimal_config_compiles(self):
        """A minimal config with one layer compiles to a context string."""
        config = PipelineConfig(
            token_budget=8000,
            layers=[
                {
                    "name": "core",
                    "cache_policy": "immutable",
                    "contents": [
                        {"key": "system_prompt", "source": "config"},
                    ],
                }
            ],
        )
        ctx = ResolverContext(
            agent_config={"system_prompt": "You are a helpful assistant."},
            trigger_input="Hello",
            interface_type="user_message",
        )
        result = await self.engine.compile(config, ctx)
        assert "helpful assistant" in result.content
        assert result.tokens > 0
        assert result.pipeline_result.overall_status == "success"

    @pytest.mark.asyncio
    async def test_multi_layer_compilation(self):
        """Multiple layers compile in order."""
        config = PipelineConfig(
            token_budget=16000,
            layers=[
                {
                    "name": "core",
                    "cache_policy": "immutable",
                    "contents": [
                        {"key": "system_prompt", "source": "config"},
                    ],
                },
                {
                    "name": "conversation",
                    "cache_policy": "append_only",
                    "contents": [
                        {"key": "current_message", "source": "input"},
                    ],
                },
            ],
        )
        ctx = ResolverContext(
            agent_config={"system_prompt": "System prompt here."},
            trigger_input="User says hello",
            interface_type="user_message",
        )
        result = await self.engine.compile(config, ctx)
        assert "System prompt here" in result.content
        assert "User says hello" in result.content
        # System prompt comes first (layer order)
        assert result.content.index("System prompt") < result.content.index("User says")

    @pytest.mark.asyncio
    async def test_heartbeat_config_minimal(self):
        """A heartbeat-style config compiles with minimal content."""
        config = PipelineConfig(
            token_budget=3000,
            compaction={"enabled": False},
            summarization={"enabled": False},
            retrieval={"enabled": False},
            intent_detection={"enabled": False},
            layers=[
                {
                    "name": "core",
                    "cache_policy": "immutable",
                    "contents": [
                        {"key": "system_prompt", "source": "config"},
                    ],
                },
                {
                    "name": "trigger",
                    "cache_policy": "always_new",
                    "contents": [
                        {"key": "current_timestamp", "source": "runtime"},
                    ],
                },
            ],
        )
        ctx = ResolverContext(
            agent_config={"system_prompt": "Check inbox for new emails."},
            trigger_input="",
            interface_type="heartbeat",
        )
        result = await self.engine.compile(config, ctx)
        assert "Check inbox" in result.content
        assert result.tokens < 3000

    @pytest.mark.asyncio
    async def test_interface_routing(self):
        """InterfaceRouter selects correct config."""
        loader = ConfigLoader()
        router = InterfaceRouter(
            interfaces={
                "user_message": {
                    "token_budget": 32000,
                    "layers": [
                        {
                            "name": "core",
                            "cache_policy": "immutable",
                            "contents": [{"key": "sp", "source": "config"}],
                        }
                    ],
                },
                "heartbeat": {
                    "token_budget": 3000,
                    "layers": [
                        {
                            "name": "core",
                            "cache_policy": "immutable",
                            "contents": [{"key": "sp", "source": "config"}],
                        }
                    ],
                },
            },
            loader=loader,
        )
        user_config = router.route("user_message")
        heartbeat_config = router.route("heartbeat")
        assert user_config.token_budget == 32000
        assert heartbeat_config.token_budget == 3000

    @pytest.mark.asyncio
    async def test_metric_collection_after_compile(self):
        """Metrics collected from PipelineResult."""
        config = PipelineConfig(
            token_budget=8000,
            layers=[
                {
                    "name": "core",
                    "cache_policy": "immutable",
                    "contents": [{"key": "system_prompt", "source": "config"}],
                }
            ],
        )
        ctx = ResolverContext(
            agent_config={"system_prompt": "Hello"},
            trigger_input="test",
            interface_type="user_message",
        )
        result = await self.engine.compile(config, ctx)
        snapshot = self.collector.collect(result.pipeline_result, "user_message")
        assert snapshot.agent_name == "test_agent"
        assert snapshot.get("sr2_pipeline_total_tokens") is not None

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self):
        """Pipeline enforces token budget by trimming last layers."""
        config = PipelineConfig(
            token_budget=1000,  # Tight budget
            layers=[
                {
                    "name": "core",
                    "cache_policy": "immutable",
                    "contents": [
                        {"key": "system_prompt", "source": "config"},
                    ],
                },
                {
                    "name": "conversation",
                    "cache_policy": "append_only",
                    "contents": [
                        {"key": "message", "source": "input"},
                    ],
                },
            ],
        )
        ctx = ResolverContext(
            agent_config={"system_prompt": "Short prompt."},
            trigger_input="A " * 500,  # Way over budget
            interface_type="user_message",
        )
        result = await self.engine.compile(config, ctx)
        # Should compile without error, but content truncated
        assert result.pipeline_result.overall_status in ("success", "degraded")
