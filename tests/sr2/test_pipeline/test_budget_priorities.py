"""Tests for priority-aware budget overflow truncation."""

import logging

import pytest

from sr2.config.models import (
    LayerConfig,
    PipelineConfig,
)
from sr2.resolvers.registry import (
    ContentResolverRegistry,
    ResolvedContent,
    ResolverContext,
)
from sr2.cache.policies import create_default_cache_registry
from sr2.pipeline.engine import PipelineEngine


def _ctx() -> ResolverContext:
    return ResolverContext(agent_config={}, trigger_input="", session_id="t")


def _make_engine() -> PipelineEngine:
    resolver_reg = ContentResolverRegistry()
    cache_reg = create_default_cache_registry()
    return PipelineEngine(resolver_reg, cache_reg)


def _content(tokens: int) -> ResolvedContent:
    """Create ResolvedContent with given token count."""
    return ResolvedContent(key="k", content="x" * (tokens * 4), tokens=tokens)


class TestPreservedLayers:
    @pytest.mark.asyncio
    async def test_preserved_layer_never_truncated(self):
        engine = _make_engine()
        layers = {
            "system": [_content(600)],
            "history": [_content(600)],
        }
        config = PipelineConfig(
            token_budget=1000,
            layers=[
                LayerConfig(name="system", contents=[], preserve=True, priority=100),
                LayerConfig(name="history", contents=[], priority=10),
            ],
        )
        result = await engine._enforce_budget(layers, config, _ctx())

        assert sum(c.tokens for c in result["system"]) == 600
        assert sum(c.tokens for c in result["history"]) < 600

    @pytest.mark.asyncio
    async def test_first_layer_always_preserved(self):
        """First layer is preserved even without explicit preserve=True."""
        engine = _make_engine()
        layers = {
            "core": [_content(600)],
            "extra": [_content(600)],
        }
        config = PipelineConfig(
            token_budget=1000,
            layers=[
                LayerConfig(name="core", contents=[]),
                LayerConfig(name="extra", contents=[], priority=0),
            ],
        )
        result = await engine._enforce_budget(layers, config, _ctx())

        assert sum(c.tokens for c in result["core"]) == 600


class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_lower_priority_truncated_first(self):
        engine = _make_engine()
        layers = {
            "high": [_content(500)],
            "medium": [_content(500)],
            "low": [_content(500)],
        }
        config = PipelineConfig(
            token_budget=1200,
            layers=[
                LayerConfig(name="high", contents=[], priority=100, preserve=True),
                LayerConfig(name="medium", contents=[], priority=50),
                LayerConfig(name="low", contents=[], priority=10),
            ],
        )
        result = await engine._enforce_budget(layers, config, _ctx())

        high_tokens = sum(c.tokens for c in result["high"])
        low_tokens = sum(c.tokens for c in result["low"])

        assert high_tokens == 500  # preserved
        assert low_tokens < 500  # truncated first (lower priority)

    @pytest.mark.asyncio
    async def test_within_budget_no_truncation(self):
        engine = _make_engine()
        layers = {
            "a": [_content(100)],
            "b": [_content(100)],
        }
        config = PipelineConfig(
            token_budget=1000,
            layers=[
                LayerConfig(name="a", contents=[], priority=50),
                LayerConfig(name="b", contents=[], priority=10),
            ],
        )
        result = await engine._enforce_budget(layers, config, _ctx())

        assert sum(c.tokens for c in result["a"]) == 100
        assert sum(c.tokens for c in result["b"]) == 100


class TestMinTokens:
    @pytest.mark.asyncio
    async def test_min_tokens_respected(self):
        engine = _make_engine()
        layers = {
            "preserved": [_content(600)],
            "limited": [_content(600)],
        }
        config = PipelineConfig(
            token_budget=1000,
            layers=[
                LayerConfig(name="preserved", contents=[], preserve=True, priority=100),
                LayerConfig(name="limited", contents=[], priority=10, min_tokens=300),
            ],
        )
        result = await engine._enforce_budget(layers, config, _ctx())

        limited_tokens = sum(c.tokens for c in result["limited"])
        assert limited_tokens >= 300


class TestOverflowWarning:
    @pytest.mark.asyncio
    async def test_warning_when_budget_cannot_be_met(self, caplog):
        engine = _make_engine()
        # Both layers preserved, over budget
        layers = {
            "a": [_content(600)],
            "b": [_content(600)],
        }
        config = PipelineConfig(
            token_budget=1000,
            layers=[
                LayerConfig(name="a", contents=[], preserve=True),
                LayerConfig(name="b", contents=[], preserve=True),
            ],
        )
        with caplog.at_level(logging.WARNING):
            await engine._enforce_budget(layers, config, _ctx())

        assert any("still over budget" in r.message for r in caplog.records)


class TestLayerConfigDefaults:
    def test_default_priority(self):
        config = LayerConfig(name="test", contents=[])
        assert config.priority == 0
        assert config.preserve is False
        assert config.min_tokens == 0

    def test_custom_priority(self):
        config = LayerConfig(
            name="system", contents=[], priority=100, preserve=True, min_tokens=50
        )
        assert config.priority == 100
        assert config.preserve is True
        assert config.min_tokens == 50
