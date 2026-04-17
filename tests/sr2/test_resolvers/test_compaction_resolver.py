"""Tests for compaction resolver."""

import pytest

from sr2.compaction.engine import CompactionEngine
from sr2.config.models import CompactionConfig, CompactionRuleConfig, CostGateConfig
from sr2.resolvers.compaction_resolver import CompactionResolver
from sr2.resolvers.registry import ResolverContext


def _make_context(history: list[dict]) -> ResolverContext:
    return ResolverContext(
        agent_config={"session_history": history},
        trigger_input="test",
    )


def _make_engine(raw_window: int = 3) -> CompactionEngine:
    config = CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=10,
        cost_gate=CostGateConfig(enabled=False),
        rules=[
            CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
        ],
    )
    return CompactionEngine(config)


class TestCompactionResolver:
    """Tests for CompactionResolver."""

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """Empty session history returns empty content."""
        engine = _make_engine()
        resolver = CompactionResolver(engine)
        context = _make_context([])
        result = await resolver.resolve("compacted", {}, context)

        assert result.content == ""
        assert result.tokens == 0

    @pytest.mark.asyncio
    async def test_history_shorter_than_raw_window(self):
        """History shorter than raw window returns empty compacted zone."""
        engine = _make_engine(raw_window=5)
        resolver = CompactionResolver(engine)
        context = _make_context([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        result = await resolver.resolve("compacted", {}, context)

        assert result.content == ""

    @pytest.mark.asyncio
    async def test_history_longer_than_raw_window_formatted(self):
        """History longer than raw window has compacted zone formatted."""
        engine = _make_engine(raw_window=2)
        resolver = CompactionResolver(engine)
        context = _make_context([
            {"role": "assistant", "content": "first turn"},
            {"role": "assistant", "content": "second turn"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "fourth"},
        ])
        result = await resolver.resolve("compacted", {}, context)

        assert result.content != ""
        assert "[Turn 0]" in result.content

    @pytest.mark.asyncio
    async def test_metadata_includes_stats(self):
        """Metadata includes compaction statistics."""
        engine = _make_engine(raw_window=2)
        resolver = CompactionResolver(engine)
        context = _make_context([
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "assistant", "content": "third"},
        ])
        result = await resolver.resolve("compacted", {}, context)

        assert "turns_compacted" in result.metadata
        assert "original_tokens" in result.metadata
        assert "compacted_tokens" in result.metadata

    @pytest.mark.asyncio
    async def test_follows_resolver_protocol(self):
        """Resolver follows ContentResolver protocol."""
        engine = _make_engine()
        resolver = CompactionResolver(engine)
        context = _make_context([])
        result = await resolver.resolve("key", {}, context)

        assert result.key == "key"
        assert isinstance(result.content, str)
        assert isinstance(result.tokens, int)
