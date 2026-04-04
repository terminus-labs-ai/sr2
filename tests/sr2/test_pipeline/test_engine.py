import logging


from sr2.pipeline.engine import PipelineEngine, CompiledContext
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext, ResolvedContent
from sr2.cache.registry import PipelineState
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

    async def test_kv_cache_layer_ordering_l1_l2_l3(self):
        """Compiled context preserves layer order: L1 (core) -> L2 (memory) -> L3 (conversation).

        The pipeline assembles layers in config order (most-stable to least-stable).
        This ensures KV-cache prefix stability: immutable layers come first so that
        the prefix can be reused across turns.
        """
        resolver_core = MockResolver(content="system prompt text", tokens=100)
        resolver_memory = MockResolver(content="retrieved memories", tokens=80)
        resolver_conv = MockResolver(content="user: hello\nassistant: hi", tokens=60)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", resolver_core)
        resolvers.register("memory_src", resolver_memory)
        resolvers.register("conv_src", resolver_conv)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="sys", source="core_src")],
            ),
            LayerConfig(
                name="memory",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="mem", source="memory_src")],
            ),
            LayerConfig(
                name="conversation",
                cache_policy="always_new",
                contents=[ContentItemConfig(key="conv", source="conv_src")],
            ),
        ])

        result = await engine.compile(config, _make_context())

        # Verify layer ordering in the compiled output
        layer_names = list(result.layers.keys())
        assert layer_names == ["core", "memory", "conversation"], (
            f"Layers must be ordered L1->L2->L3, got {layer_names}"
        )

        # Verify content order in the serialized output
        content = result.content
        core_pos = content.index("system prompt text")
        memory_pos = content.index("retrieved memories")
        conv_pos = content.index("user: hello")
        assert core_pos < memory_pos < conv_pos, (
            "Content must be ordered: core < memory < conversation"
        )

    async def test_kv_cache_prefix_stable_across_conversation_changes(self):
        """L1/L2 prefix stays identical when only L3 (conversation) changes between turns.

        This is the core KV-cache invariant: recompiling with new conversation
        content must not alter the serialized L1 (core) or L2 (memory) prefix,
        so the LLM provider can reuse the cached KV-cache prefix.
        """
        resolver_core = MockResolver(content="system prompt text", tokens=100)
        resolver_memory = MockResolver(content="retrieved memories", tokens=80)

        turn_count = 0

        class ConversationResolver:
            """Simulates conversation growing each turn."""

            async def resolve(self, key, config, context):
                nonlocal turn_count
                turn_count += 1
                content = f"user: message {turn_count}\nassistant: reply {turn_count}"
                return ResolvedContent(key=key, content=content, tokens=50 + turn_count * 10)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", resolver_core)
        resolvers.register("memory_src", resolver_memory)
        resolvers.register("conv_src", ConversationResolver())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="sys", source="core_src")],
            ),
            LayerConfig(
                name="memory",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="mem", source="memory_src")],
            ),
            LayerConfig(
                name="conversation",
                cache_policy="always_new",
                contents=[ContentItemConfig(key="conv", source="conv_src")],
            ),
        ])

        # Turn 1
        result1 = await engine.compile(config, _make_context(), state=PipelineState())
        # Turn 2 — conversation changes, core/memory should not
        result2 = await engine.compile(config, _make_context(), state=PipelineState())

        # Conversation content must actually differ between turns
        assert result1.layers["conversation"][0].content != result2.layers["conversation"][0].content

        # L1 and L2 layer hashes must be identical across both compiles
        for layer_name in ("core", "memory"):
            assert result1.prefix_snapshot.layer_hashes[layer_name] == \
                   result2.prefix_snapshot.layer_hashes[layer_name], \
                f"Layer '{layer_name}' hash changed between turns — prefix invalidated"

        # Conversation hash must differ (sanity check)
        assert result1.prefix_snapshot.layer_hashes["conversation"] != \
               result2.prefix_snapshot.layer_hashes["conversation"]

        # The serialized prefix (everything before conversation) must be byte-identical
        prefix1 = result1.content.split("user: message")[0]
        prefix2 = result2.content.split("user: message")[0]
        assert prefix1 == prefix2, "Serialized prefix changed between turns"

    async def test_kv_cache_layer_ordering_multi_item_l1(self):
        """L1 with multiple items (system prompt + tools) all appear before L2 and L3.

        Real pipelines have L1 containing both a system prompt and tool definitions.
        Both must serialize before any memory or conversation content.
        """
        resolver_sys = MockResolver(content="You are an AI assistant.", tokens=40)
        resolver_tools = MockResolver(content="[tool: search, tool: calculate]", tokens=30)
        resolver_mem = MockResolver(content="User prefers concise answers.", tokens=25)
        resolver_conv = MockResolver(content="user: what is 2+2?", tokens=20)

        resolvers = ContentResolverRegistry()
        resolvers.register("sys_src", resolver_sys)
        resolvers.register("tools_src", resolver_tools)
        resolvers.register("mem_src", resolver_mem)
        resolvers.register("conv_src", resolver_conv)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[
                    ContentItemConfig(key="sys", source="sys_src"),
                    ContentItemConfig(key="tools", source="tools_src"),
                ],
            ),
            LayerConfig(
                name="memory",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="mem", source="mem_src")],
            ),
            LayerConfig(
                name="conversation",
                cache_policy="always_new",
                contents=[ContentItemConfig(key="conv", source="conv_src")],
            ),
        ])

        result = await engine.compile(config, _make_context())

        content = result.content
        # Both L1 items must appear before L2
        sys_pos = content.index("You are an AI assistant.")
        tools_pos = content.index("[tool: search, tool: calculate]")
        mem_pos = content.index("User prefers concise answers.")
        conv_pos = content.index("user: what is 2+2?")

        assert sys_pos < tools_pos, "System prompt must come before tools within L1"
        assert tools_pos < mem_pos, "Tools (L1) must come before memory (L2)"
        assert mem_pos < conv_pos, "Memory (L2) must come before conversation (L3)"

    async def test_kv_cache_prefix_stable_over_multiple_turns(self):
        """Prefix remains byte-identical across 5 turns of growing conversation.

        Extends the 2-turn test to verify no drift accumulates over many compiles.
        """
        resolver_core = MockResolver(content="system prompt", tokens=50)
        resolver_memory = MockResolver(content="long-term memories", tokens=40)

        turn_count = 0

        class GrowingConversation:
            async def resolve(self, key, config, context):
                nonlocal turn_count
                turn_count += 1
                lines = [f"turn {i}" for i in range(1, turn_count + 1)]
                return ResolvedContent(
                    key=key, content="\n".join(lines), tokens=10 * turn_count,
                )

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", resolver_core)
        resolvers.register("mem_src", resolver_memory)
        resolvers.register("conv_src", GrowingConversation())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="sys", source="core_src")],
            ),
            LayerConfig(
                name="memory",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="mem", source="mem_src")],
            ),
            LayerConfig(
                name="conversation",
                cache_policy="always_new",
                contents=[ContentItemConfig(key="conv", source="conv_src")],
            ),
        ])

        results = []
        for _ in range(5):
            r = await engine.compile(config, _make_context(), state=PipelineState())
            results.append(r)

        # Extract the prefix (everything before conversation content) from each turn
        first_prefix = results[0].content.split("turn 1")[0]
        for i, r in enumerate(results[1:], start=2):
            prefix = r.content.split("turn 1")[0]
            assert prefix == first_prefix, (
                f"Prefix drifted at turn {i}: expected {first_prefix!r}, got {prefix!r}"
            )

        # Layer hashes for core and memory must be identical across all turns
        ref_hashes = {
            "core": results[0].prefix_snapshot.layer_hashes["core"],
            "memory": results[0].prefix_snapshot.layer_hashes["memory"],
        }
        for i, r in enumerate(results[1:], start=2):
            for layer_name in ("core", "memory"):
                assert r.prefix_snapshot.layer_hashes[layer_name] == ref_hashes[layer_name], (
                    f"Layer '{layer_name}' hash changed at turn {i}"
                )

        # Conversation hashes should all be different (content grows each turn)
        conv_hashes = [r.prefix_snapshot.layer_hashes["conversation"] for r in results]
        assert len(set(conv_hashes)) == 5, "Each turn should produce a unique conversation hash"

    async def test_kv_cache_prefix_stable_with_maximize_prefix_reuse_strategy(self):
        """maximize_prefix_reuse: re-resolved memory with identical content preserves prefix.

        Simulates a realistic scenario where memory uses 'always_new' policy
        (re-resolved every turn) but returns the same content. The
        maximize_prefix_reuse strategy should detect the content is unchanged
        and reuse cached objects, keeping the prefix stable.
        """
        resolver_core = MockResolver(content="system prompt", tokens=50)
        resolver_memory = MockResolver(content="stable memories", tokens=40)

        turn_count = 0

        class ConversationResolver:
            async def resolve(self, key, config, context):
                nonlocal turn_count
                turn_count += 1
                return ResolvedContent(
                    key=key, content=f"conversation turn {turn_count}", tokens=30,
                )

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", resolver_core)
        resolvers.register("mem_src", resolver_memory)
        resolvers.register("conv_src", ConversationResolver())

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config([
            LayerConfig(
                name="core",
                cache_policy="immutable",
                contents=[ContentItemConfig(key="sys", source="core_src")],
            ),
            LayerConfig(
                name="memory",
                cache_policy="always_new",  # Re-resolved every turn
                contents=[ContentItemConfig(key="mem", source="mem_src")],
            ),
            LayerConfig(
                name="conversation",
                cache_policy="always_new",
                contents=[ContentItemConfig(key="conv", source="conv_src")],
            ),
        ])
        config.kv_cache = KVCacheConfig(strategy="maximize_prefix_reuse")

        result1 = await engine.compile(config, _make_context(), state=PipelineState())
        result2 = await engine.compile(config, _make_context(), state=PipelineState())

        # Memory resolver was called twice (always_new forces recompute)
        assert resolver_memory.call_count == 2

        # But content is identical, so maximize_prefix_reuse should keep hashes stable
        assert result1.prefix_snapshot.layer_hashes["memory"] == \
               result2.prefix_snapshot.layer_hashes["memory"], \
            "Memory hash changed despite identical content under maximize_prefix_reuse"

        # Serialized prefix must be byte-identical
        prefix1 = result1.content.split("conversation turn")[0]
        prefix2 = result2.content.split("conversation turn")[0]
        assert prefix1 == prefix2, "Serialized prefix changed despite stable memory content"
