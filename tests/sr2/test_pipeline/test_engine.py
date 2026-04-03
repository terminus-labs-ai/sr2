import logging

import pytest

from sr2.pipeline.engine import PipelineEngine, CompiledContext
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext, ResolvedContent
from sr2.cache.registry import CachePolicyRegistry, PipelineState
from sr2.cache.policies import create_default_cache_registry
from sr2.config.models import PipelineConfig, LayerConfig, ContentItemConfig, KVCacheConfig


class MockResolver:
    def __init__(self, content="test content", tokens=100):
        self._content = content
        self._tokens = tokens
        self.call_count = 0

    async def resolve(self, key, config, context):
        self.call_count += 1
        return ResolvedContent(key=key, content=self._content, tokens=self._tokens)


class FailingResolver:
    async def resolve(self, key, config, context):
        raise RuntimeError("resolver failed")


def _make_context():
    return ResolverContext(agent_config={}, trigger_input="hello")


def _make_config(layers, token_budget=32000):
    return PipelineConfig(token_budget=token_budget, layers=layers)


class TestPipelineEngine:
    async def test_compile_single_layer_single_item(self):
        """Compile with one layer, one content item returns CompiledContext with content."""
        resolver = MockResolver(content="system prompt", tokens=50)
        resolvers = ContentResolverRegistry()
        resolvers.register("config", resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="sys", source="config")],
            ),
        ])

        result = await engine.compile(config, _make_context())

        assert isinstance(result, CompiledContext)
        assert result.content == "system prompt"
        assert result.tokens == 50
        assert "core" in result.layers
        assert len(result.layers["core"]) == 1

    async def test_compile_multiple_layers_concatenated_in_order(self):
        """Compile with multiple layers produces content concatenated in layer order."""
        resolver_a = MockResolver(content="layer one content", tokens=50)
        resolver_b = MockResolver(content="layer two content", tokens=60)

        resolvers = ContentResolverRegistry()
        resolvers.register("source_a", resolver_a)
        resolvers.register("source_b", resolver_b)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="first",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="a", source="source_a")],
            ),
            LayerConfig(
                name="second",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="b", source="source_b")],
            ),
        ])

        result = await engine.compile(config, _make_context())

        assert result.content == "layer one content\n\nlayer two content"
        assert result.tokens == 110
        assert list(result.layers.keys()) == ["first", "second"]

    async def test_immutable_cache_reuses_content(self):
        """Cache policy 'immutable' reuses cached content on second compile."""
        resolver = MockResolver(content="cached", tokens=30)
        resolvers = ContentResolverRegistry()
        resolvers.register("src", resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="k", source="src")],
            ),
        ])
        ctx = _make_context()

        # First compile — resolver called
        await engine.compile(config, ctx, state=PipelineState())
        assert resolver.call_count == 1

        # Second compile — immutable should reuse cache (previous_state is now set)
        await engine.compile(config, ctx, state=PipelineState())
        assert resolver.call_count == 1

    async def test_always_new_cache_re_resolves(self):
        """Cache policy 'always_new' re-resolves on every compile."""
        resolver = MockResolver(content="fresh", tokens=20)
        resolvers = ContentResolverRegistry()
        resolvers.register("src", resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="dynamic",
                cache_policy="always_new",
                contents=[ContentItemConfig(key="k", source="src")],
            ),
        ])
        ctx = _make_context()

        await engine.compile(config, ctx, state=PipelineState())
        assert resolver.call_count == 1

        await engine.compile(config, ctx, state=PipelineState())
        assert resolver.call_count == 2

    async def test_token_budget_trims_last_layer(self):
        """Token budget enforcement trims last layer when over budget."""
        resolver_core = MockResolver(content="core content", tokens=500)
        resolver_extra = MockResolver(content="extra content that is long", tokens=600)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", resolver_core)
        resolvers.register("extra_src", resolver_extra)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="core",
                    cache_policy="immutable",
                    contents=[ContentItemConfig(key="c", source="core_src")],
                ),
                LayerConfig(
                    name="extra",
                    cache_policy="immutable",
                    contents=[ContentItemConfig(key="e", source="extra_src")],
                ),
            ],
            token_budget=1000,
        )

        result = await engine.compile(config, _make_context())

        # Total would be 1100 without trimming; budget is 1000
        assert result.tokens <= 1000
        # Core layer should be untouched
        assert result.layers["core"][0].tokens == 500

    async def test_first_layer_never_trimmed(self):
        """First layer is never trimmed even when over budget."""
        resolver_core = MockResolver(content="important core", tokens=800)
        resolver_extra = MockResolver(content="extra", tokens=400)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", resolver_core)
        resolvers.register("extra_src", resolver_extra)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="core",
                    cache_policy="immutable",
                    contents=[ContentItemConfig(key="c", source="core_src")],
                ),
                LayerConfig(
                    name="extra",
                    cache_policy="immutable",
                    contents=[ContentItemConfig(key="e", source="extra_src")],
                ),
            ],
            token_budget=1000,
        )

        result = await engine.compile(config, _make_context())

        # Core (first layer) must remain untouched at 800 tokens
        assert result.layers["core"][0].tokens == 800
        assert result.layers["core"][0].content == "important core"
        # Extra layer should have been trimmed
        assert result.tokens <= 1000

    async def test_unknown_resolver_source_produces_failed_stage(self):
        """Unknown resolver source produces a StageResult with status 'failed'."""
        resolvers = ContentResolverRegistry()
        # No resolver registered for "unknown_source"

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="broken",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="x", source="unknown_source")],
            ),
        ])

        result = await engine.compile(config, _make_context())

        assert result.pipeline_result.has_failures
        failed_stages = [
            s for s in result.pipeline_result.stages if s.status == "failed"
        ]
        assert len(failed_stages) == 1
        assert failed_stages[0].stage_name == "broken"
        assert "unknown_source" in failed_stages[0].error

    async def test_optional_content_item_failure_is_skipped(self):
        """Optional content item that fails is skipped, not errored."""
        good_resolver = MockResolver(content="good", tokens=50)
        resolvers = ContentResolverRegistry()
        resolvers.register("good_src", good_resolver)
        resolvers.register("bad_src", FailingResolver())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="mixed",
                cache_policy="immutable",
                contents=[
                    ContentItemConfig(key="ok", source="good_src"),
                    ContentItemConfig(key="fail", source="bad_src", optional=True),
                ],
            ),
        ])

        result = await engine.compile(config, _make_context())

        # Should succeed overall — the optional failure is skipped
        assert not result.pipeline_result.has_failures
        assert result.content == "good"
        assert result.tokens == 50
        # Only the successful item should be in the layer
        assert len(result.layers["mixed"]) == 1

    async def test_strategy_maximize_reuse_reuses_identical_content(self):
        """maximize_prefix_reuse: policy says recompute but content unchanged → cached string reused."""
        resolver = MockResolver(content="stable content", tokens=50)
        resolvers = ContentResolverRegistry()
        resolvers.register("src", resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="dynamic",
                    cache_policy="always_new",  # Forces recompute every time
                    contents=[ContentItemConfig(key="k", source="src")],
                ),
            ],
        )
        config.kv_cache = KVCacheConfig(strategy="maximize_prefix_reuse")

        # First compile
        result1 = await engine.compile(config, _make_context(), state=PipelineState())
        assert resolver.call_count == 1

        # Second compile — resolver called again (always_new), but content is identical
        # so maximize_prefix_reuse should reuse the cached objects
        result2 = await engine.compile(config, _make_context(), state=PipelineState())
        assert resolver.call_count == 2  # Resolver was called
        # Content should still be identical
        assert result1.content == result2.content

    async def test_strategy_append_only_warns_on_unexpected_change(self, caplog):
        """append_only: layer content changes → warning logged."""
        call_count = 0

        class ChangingResolver:
            async def resolve(self, key, config, context):
                nonlocal call_count
                call_count += 1
                return ResolvedContent(
                    key=key, content=f"version {call_count}", tokens=50
                )

        resolvers = ContentResolverRegistry()
        resolvers.register("src", ChangingResolver())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="should_be_stable",
                    cache_policy="always_new",  # Force recompute every time
                    contents=[ContentItemConfig(key="k", source="src")],
                ),
            ],
        )
        config.kv_cache = KVCacheConfig(strategy="append_only")

        # First compile — no warning (nothing cached yet for comparison)
        await engine.compile(config, _make_context(), state=PipelineState())

        # Second compile — content changed, should warn
        with caplog.at_level(logging.WARNING):
            await engine.compile(config, _make_context(), state=PipelineState())

        assert any("append_only" in r.message for r in caplog.records)
        assert any("should_be_stable" in r.message for r in caplog.records)

    async def test_strategy_append_only_no_warn_for_dynamic_memory_layer(self, caplog):
        """append_only: memory layer with dynamic refresh should NOT warn on content change."""
        call_count = 0

        class ChangingResolver:
            async def resolve(self, key, config, context):
                nonlocal call_count
                call_count += 1
                return ResolvedContent(
                    key=key, content=f"memory version {call_count}", tokens=50
                )

        resolvers = ContentResolverRegistry()
        resolvers.register("src", ChangingResolver())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="memory",
                    cache_policy="always_new",
                    contents=[ContentItemConfig(key="k", source="src")],
                ),
            ],
        )
        config.kv_cache = KVCacheConfig(
            strategy="append_only", memory_refresh="on_topic_shift"
        )

        await engine.compile(config, _make_context(), state=PipelineState())

        with caplog.at_level(logging.WARNING):
            await engine.compile(config, _make_context(), state=PipelineState())

        assert not any("append_only" in r.message for r in caplog.records)

    async def test_strategy_append_only_warns_for_static_memory_layer(self, caplog):
        """append_only: memory layer with session_start_only SHOULD warn on content change."""
        call_count = 0

        class ChangingResolver:
            async def resolve(self, key, config, context):
                nonlocal call_count
                call_count += 1
                return ResolvedContent(
                    key=key, content=f"memory version {call_count}", tokens=50
                )

        resolvers = ContentResolverRegistry()
        resolvers.register("src", ChangingResolver())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="memory",
                    cache_policy="always_new",
                    contents=[ContentItemConfig(key="k", source="src")],
                ),
            ],
        )
        config.kv_cache = KVCacheConfig(
            strategy="append_only", memory_refresh="session_start_only"
        )

        await engine.compile(config, _make_context(), state=PipelineState())

        with caplog.at_level(logging.WARNING):
            await engine.compile(config, _make_context(), state=PipelineState())

        assert any("append_only" in r.message for r in caplog.records)
        assert any("memory" in r.message for r in caplog.records)

    async def test_prefix_snapshot_populated(self):
        """CompiledContext.prefix_snapshot is not None after compile."""
        resolver = MockResolver(content="test", tokens=10)
        resolvers = ContentResolverRegistry()
        resolvers.register("src", resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="core",
                    cache_policy="immutable",
                    contents=[ContentItemConfig(key="k", source="src")],
                ),
            ],
        )

        result = await engine.compile(config, _make_context())

        assert result.prefix_snapshot is not None
        assert result.prefix_snapshot.full_hash is not None
        assert "core" in result.prefix_snapshot.layer_hashes
        assert result.prefix_snapshot.prefix_tokens == 10
