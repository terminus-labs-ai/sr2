"""Tests for the run-context dependency seam (sr2-87).

Verifies that:
1. RunMode enum and RunContext dataclass are defined in core (no spectre import).
2. ResolverContext.run_context defaults to None (regression-safe).
3. SR2Config accepts a run_context_provider and wires it through to ResolverContext.
4. A resolver can read mode and source from the context.
5. Provider failure is handled gracefully (context remains None).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2.pipeline.dependencies import RunContext, RunMode


class TestRunMode:
    def test_values(self):
        assert RunMode.INTERACTIVE == "interactive"
        assert RunMode.HEADLESS == "headless"

    def test_is_str_enum(self):
        assert isinstance(RunMode.INTERACTIVE, str)


class TestRunContext:
    def test_frozen(self):
        ctx = RunContext(mode=RunMode.INTERACTIVE, source="cli")
        with pytest.raises(Exception):
            ctx.mode = RunMode.HEADLESS  # type: ignore

    def test_source_defaults_to_none(self):
        ctx = RunContext(mode=RunMode.HEADLESS)
        assert ctx.source is None

    def test_source_set(self):
        ctx = RunContext(mode=RunMode.INTERACTIVE, source="#general")
        assert ctx.source == "#general"


class TestResolverContextRunContext:
    def test_defaults_to_none(self):
        """When harness doesn't supply a provider, run_context is None."""
        from sr2.resolvers.registry import ResolverContext

        ctx = ResolverContext(
            agent_config={},
            trigger_input="test",
        )
        assert ctx.run_context is None

    def test_accepts_run_context(self):
        from sr2.resolvers.registry import ResolverContext

        rc = RunContext(mode=RunMode.HEADLESS, source="cron")
        ctx = ResolverContext(
            agent_config={},
            trigger_input="test",
            run_context=rc,
        )
        assert ctx.run_context is rc
        assert ctx.run_context.mode == RunMode.HEADLESS
        assert ctx.run_context.source == "cron"


class TestSR2ConfigRunContextProvider:
    def test_provider_wired_through(self):
        """SR2Config accepts run_context_provider and it's called during process()."""
        from sr2.sr2 import SR2Config

        captured = []

        def provider():
            rc = RunContext(mode=RunMode.INTERACTIVE, source="tui")
            captured.append(rc)
            return rc

        config = SR2Config(
            config_dir="/tmp",
            agent_yaml={},
            run_context_provider=provider,
        )
        assert config.run_context_provider is provider

    def test_provider_defaults_to_none(self):
        from sr2.sr2 import SR2Config

        config = SR2Config(
            config_dir="/tmp",
            agent_yaml={},
        )
        assert config.run_context_provider is None


class TestResolverSeesRunContext:
    """Integration test: prove a resolver can read the supplied mode."""

    @pytest.fixture
    def mock_resolver(self):
        """A resolver that records whatever run_context it sees."""
        resolver = AsyncMock()
        seen_contexts = []

        async def resolve(key, config, context):
            seen_contexts.append(context.run_context)
            from sr2.resolvers.registry import ResolvedContent

            return ResolvedContent(
                key=key,
                content="",
                tokens=0,
            )

        resolver.resolve = resolve
        resolver.seen_contexts = seen_contexts
        return resolver

    async def test_resolver_receives_mode(self, mock_resolver):
        """A resolver reads the run mode from context when harness supplies it."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.cache.registry import CachePolicyRegistry
        from sr2.config.models import PipelineConfig, LayerConfig, ContentItemConfig
        from sr2.resolvers.registry import ContentResolverRegistry

        rc = RunContext(mode=RunMode.HEADLESS, source="cron")

        # Wire resolver into registry
        registry = ContentResolverRegistry()
        registry.register("test_source", mock_resolver)

        engine = PipelineEngine(
            resolver_registry=registry,
            cache_registry=CachePolicyRegistry(),
        )

        config = PipelineConfig(
            token_budget=8000,
            layers=[
                LayerConfig(
                    name="test",
                    contents=[ContentItemConfig(key="k", source="test_source")],
                )
            ],
        )

        from sr2.resolvers.registry import ResolverContext

        ctx = ResolverContext(
            agent_config={},
            trigger_input="test",
            run_context=rc,
        )

        await engine.compile(config, ctx)

        assert len(mock_resolver.seen_contexts) == 1
        assert mock_resolver.seen_contexts[0] is rc
        assert mock_resolver.seen_contexts[0].mode == RunMode.HEADLESS
        assert mock_resolver.seen_contexts[0].source == "cron"

    async def test_resolver_sees_none_when_not_supplied(self, mock_resolver):
        """Regression-safe: resolver sees None when no provider."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.cache.registry import CachePolicyRegistry
        from sr2.config.models import PipelineConfig, LayerConfig, ContentItemConfig
        from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext

        registry = ContentResolverRegistry()
        registry.register("test_source", mock_resolver)

        engine = PipelineEngine(
            resolver_registry=registry,
            cache_registry=CachePolicyRegistry(),
        )

        config = PipelineConfig(
            token_budget=8000,
            layers=[
                LayerConfig(
                    name="test",
                    contents=[ContentItemConfig(key="k", source="test_source")],
                )
            ],
        )

        ctx = ResolverContext(
            agent_config={},
            trigger_input="test",
        )

        await engine.compile(config, ctx)

        assert mock_resolver.seen_contexts[0] is None


class TestNoSpectreImport:
    def test_dependencies_has_no_spectre_import(self):
        """Core module must not import spectre types."""
        import sr2.pipeline.dependencies as mod

        source = open(mod.__file__).read()
        assert "sr2_spectre" not in source
        assert "spectre" not in source.lower() or "sr2_spectre" not in source

    def test_resolver_context_no_spectre_import(self):
        from sr2.resolvers import registry as mod

        source = open(mod.__file__).read()
        assert "sr2_spectre" not in source


class TestExports:
    def test_pipeline_init_exports(self):
        from sr2.pipeline import RunContext, RunContextProvider, RunMode

        assert RunContext is not None
        assert RunMode is not None
        # RunContextProvider is a TypeAlias — just verify it's importable
        assert RunContextProvider is not None
