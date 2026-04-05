"""Integration tests for token budget enforcement in PipelineEngine.

Tests that the engine properly enforces token budgets through compaction,
summarization, and truncation as a last resort.
"""

import logging

import pytest

from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext, ResolvedContent, estimate_tokens
from sr2.cache.policies import create_default_cache_registry
from sr2.config.models import PipelineConfig, LayerConfig, ContentItemConfig
from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig
from sr2.pipeline.conversation import ConversationManager
from sr2.summarization.engine import SummarizationEngine
from sr2.config.models import SummarizationConfig


# --- Helpers ---


class MockResolver:
    """Resolver that returns configurable content."""

    def __init__(self, content="test content", tokens=None):
        self._content = content
        self._tokens = tokens if tokens is not None else estimate_tokens(content)

    async def resolve(self, key, config, context):
        return ResolvedContent(key=key, content=self._content, tokens=self._tokens)


class DynamicSessionResolver:
    """Resolver that reads session_history from context (like real SessionResolver)."""

    async def resolve(self, key, config, context):
        history = context.agent_config.get("session_history", [])
        formatted = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in history
        )
        return ResolvedContent(key=key, content=formatted, tokens=estimate_tokens(formatted))


class TrackingOverflowHandler:
    """Budget overflow handler that tracks calls and optionally reduces content."""

    def __init__(self, reduce_to: int | None = None):
        self.call_count = 0
        self.last_excess = 0
        self._reduce_to = reduce_to

    async def __call__(self, layers, budget, config, ctx):
        self.call_count += 1
        total = sum(c.tokens for contents in layers.values() for c in contents)
        self.last_excess = total - budget

        if self._reduce_to is not None:
            # Reduce the last layer's content to fit
            layer_names = list(layers.keys())
            if len(layer_names) > 1:
                last_layer = layer_names[-1]
                first_layer_tokens = sum(c.tokens for c in layers[layer_names[0]])
                target = budget - first_layer_tokens
                if target > 0 and layers[last_layer]:
                    item = layers[last_layer][0]
                    char_limit = target * 4
                    new_content = item.content[:char_limit]
                    layers[last_layer][0] = ResolvedContent(
                        key=item.key,
                        content=new_content,
                        tokens=target,
                        metadata=item.metadata,
                    )
            return layers
        return None  # Signal: could not reduce


def _make_context(session_history=None):
    agent_config = {"system_prompt": "test", "session_history": session_history or []}
    return ResolverContext(agent_config=agent_config, trigger_input="hello", session_id="test-session")


def _make_config(layers, token_budget=32000):
    return PipelineConfig(token_budget=token_budget, layers=layers)


def _make_session_history(num_turns, content_size=200):
    """Create session history with large content per turn."""
    history = []
    for i in range(num_turns):
        content = f"Turn {i}: " + "x" * content_size
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
    return history


def _make_compaction_config(raw_window=3):
    return CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        rules=[
            CompactionRuleConfig(
                type="tool_output",
                strategy="schema_and_sample",
            ),
        ],
    )


# --- Tests ---


class TestBudgetEnforcementBasic:
    """Basic budget enforcement: compile + trim."""

    async def test_context_within_budget_passes_through(self):
        """Context within budget is returned unchanged."""
        resolver = MockResolver(content="short", tokens=100)
        resolvers = ContentResolverRegistry()
        resolvers.register("src", resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [LayerConfig(name="core", cache_policy="immutable",
                         contents=[ContentItemConfig(key="k", source="src")])],
            token_budget=1000,
        )

        result = await engine.compile(config, _make_context())
        assert result.tokens == 100
        assert result.tokens <= 1000

    async def test_context_exceeding_budget_gets_trimmed(self):
        """Context exceeding budget is trimmed to fit."""
        core_resolver = MockResolver(content="core " * 100, tokens=500)
        session_resolver = MockResolver(content="session " * 200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("session_src", session_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="sys", source="core_src")]),
                LayerConfig(name="conversation", cache_policy="always_new",
                            contents=[ContentItemConfig(key="history", source="session_src")]),
            ],
            token_budget=1000,
        )

        result = await engine.compile(config, _make_context())

        # Must not exceed budget
        assert result.tokens <= 1000
        # Core layer untouched
        assert result.layers["core"][0].tokens == 500
        # Truncation event recorded
        assert engine.truncation_events >= 1

    async def test_pipeline_never_returns_over_budget(self):
        """Pipeline NEVER returns a CompiledContext with tokens > budget."""
        core_resolver = MockResolver(tokens=2000)
        extra_resolver = MockResolver(tokens=5000)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("extra_src", extra_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="extra_src")]),
            ],
            token_budget=4000,
        )

        result = await engine.compile(config, _make_context())
        assert result.tokens <= 4000

    async def test_truncation_event_increments_metric(self):
        """Each truncation increments the truncation_events counter."""
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        assert engine.truncation_events == 0
        await engine.compile(config, _make_context())
        assert engine.truncation_events == 1


class TestBudgetOverflowHandler:
    """Tests for the budget_overflow_handler callback."""

    async def test_handler_called_when_over_budget(self):
        """Handler is called when compiled context exceeds budget."""
        handler = TrackingOverflowHandler()
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        await engine.compile(config, _make_context())

        assert handler.call_count == 1
        assert handler.last_excess == 300  # 1300 - 1000

    async def test_handler_not_called_when_under_budget(self):
        """Handler is NOT called when context fits within budget."""
        handler = TrackingOverflowHandler()
        resolver = MockResolver(tokens=100)

        resolvers = ContentResolverRegistry()
        resolvers.register("src", resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=handler,
        )
        config = _make_config(
            [LayerConfig(name="core", cache_policy="immutable",
                         contents=[ContentItemConfig(key="k", source="src")])],
            token_budget=1000,
        )

        await engine.compile(config, _make_context())
        assert handler.call_count == 0

    async def test_successful_handler_prevents_truncation(self):
        """When handler reduces content below budget, no truncation occurs."""
        handler = TrackingOverflowHandler(reduce_to=500)
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(content="x" * 3200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        result = await engine.compile(config, _make_context())

        assert handler.call_count == 1
        assert result.tokens <= 1000
        # No truncation event because handler succeeded
        assert engine.truncation_events == 0

    async def test_failed_handler_falls_back_to_truncation(self):
        """When handler returns None, engine falls back to truncation."""
        handler = TrackingOverflowHandler(reduce_to=None)  # Returns None
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(content="x" * 3200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        result = await engine.compile(config, _make_context())

        assert handler.call_count == 1
        assert result.tokens <= 1000
        # Truncation happened because handler couldn't help
        assert engine.truncation_events == 1

    async def test_handler_exception_falls_back_to_truncation(self, caplog):
        """When handler raises exception, engine logs error and truncates."""
        async def failing_handler(layers, budget, config, ctx):
            raise RuntimeError("handler exploded")

        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(content="x" * 3200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=failing_handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        with caplog.at_level(logging.ERROR):
            result = await engine.compile(config, _make_context())

        assert result.tokens <= 1000
        assert engine.truncation_events == 1
        assert any("handler failed" in r.message.lower() for r in caplog.records)


class TestBudgetWithCompaction:
    """Tests for compaction triggering during budget enforcement."""

    async def test_compaction_triggers_when_budget_pressure_high(self):
        """When over budget, compaction is triggered via the overflow handler."""
        compaction_config = _make_compaction_config(raw_window=2)
        compaction_engine = CompactionEngine(compaction_config)
        conversation = ConversationManager(
            compaction_engine=compaction_engine,
            raw_window=2,
        )

        # Create a handler that uses the conversation manager
        async def compaction_handler(layers, budget, config, ctx):
            session_id = ctx.session_id or "default"
            zones = conversation.zones(session_id)

            # Seed from session history if zones empty
            if not zones.raw and not zones.compacted:
                history = ctx.agent_config.get("session_history", [])
                for i, msg in enumerate(history):
                    turn = ConversationTurn(
                        turn_number=i,
                        role=msg.get("role", "unknown"),
                        content=msg.get("content", ""),
                    )
                    conversation.add_turn(turn, session_id)

            # Run compaction
            conversation.run_compaction(session_id)

            # Rebuild session content from zones
            zones = conversation.zones(session_id)
            parts = []
            for turn in zones.compacted + zones.raw:
                parts.append(f"{turn.role}: {turn.content}")
            new_content = "\n".join(parts)
            new_tokens = estimate_tokens(new_content)

            # Find and replace session item
            for layer_name, contents in layers.items():
                for i, item in enumerate(contents):
                    if item.key == "history":
                        layers[layer_name][i] = ResolvedContent(
                            key=item.key,
                            content=new_content,
                            tokens=new_tokens,
                            metadata=item.metadata,
                        )
                        return layers
            return None

        # Build large session history (20 turns with tool outputs)
        history = _make_session_history(20, content_size=300)

        session_resolver = DynamicSessionResolver()
        core_resolver = MockResolver(content="system prompt", tokens=200)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("session", session_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=compaction_handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="sys", source="core_src")]),
                LayerConfig(name="conversation", cache_policy="always_new",
                            contents=[ContentItemConfig(key="history", source="session")]),
            ],
            token_budget=1000,
        )

        ctx = _make_context(session_history=history)
        result = await engine.compile(config, ctx)

        # Compaction should have been triggered
        zones = conversation.zones("test-session")
        assert len(zones.raw) <= 2  # raw_window = 2
        # Budget should be respected
        assert result.tokens <= 1000


class TestBudgetWithSummarization:
    """Tests for summarization when compaction alone isn't enough."""

    async def test_summarization_triggers_when_compaction_insufficient(self):
        """When compaction alone can't meet budget, summarization is tried."""
        compaction_config = _make_compaction_config(raw_window=2)
        compaction_engine = CompactionEngine(compaction_config)

        summarize_called = False

        async def mock_llm(system, prompt):
            nonlocal summarize_called
            summarize_called = True
            return '{"key_decisions": ["decision1"], "unresolved": [], "facts": ["fact1"], "user_preferences": [], "errors_encountered": []}'

        summarization_engine = SummarizationEngine(
            config=SummarizationConfig(enabled=True, trigger="token_threshold", threshold=0.5),
            llm_callable=mock_llm,
        )

        conversation = ConversationManager(
            compaction_engine=compaction_engine,
            summarization_engine=summarization_engine,
            raw_window=2,
            compacted_max_tokens=500,  # Low threshold to trigger summarization
        )

        async def full_handler(layers, budget, config, ctx):
            session_id = ctx.session_id or "default"
            zones = conversation.zones(session_id)

            # Seed from session history
            if not zones.raw and not zones.compacted:
                history = ctx.agent_config.get("session_history", [])
                for i, msg in enumerate(history):
                    turn = ConversationTurn(
                        turn_number=i,
                        role=msg.get("role", "unknown"),
                        content=msg.get("content", ""),
                    )
                    conversation.add_turn(turn, session_id)

            # Phase 1: Compaction
            conversation.run_compaction(session_id)

            # Phase 2: Check if summarization needed
            zones = conversation.zones(session_id)
            non_session_tokens = sum(
                c.tokens for name, contents in layers.items()
                for c in contents
                if not (c.key == "history")
            )
            zone_tokens = zones.total_tokens
            if zone_tokens + non_session_tokens > budget:
                await conversation.run_summarization(session_id)

            # Rebuild
            zones = conversation.zones(session_id)
            parts = []
            for summary in zones.summarized:
                parts.append(f"[Summary]\n{summary}")
            for turn in zones.compacted + zones.raw:
                parts.append(f"{turn.role}: {turn.content}")
            new_content = "\n".join(parts)
            new_tokens = estimate_tokens(new_content)

            for layer_name, contents in layers.items():
                for i, item in enumerate(contents):
                    if item.key == "history":
                        layers[layer_name][i] = ResolvedContent(
                            key=item.key, content=new_content,
                            tokens=new_tokens, metadata=item.metadata,
                        )
                        return layers
            return None

        # Create a large session history
        history = _make_session_history(30, content_size=400)

        session_resolver = DynamicSessionResolver()
        core_resolver = MockResolver(content="system prompt", tokens=200)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("session", session_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=full_handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="sys", source="core_src")]),
                LayerConfig(name="conversation", cache_policy="always_new",
                            contents=[ContentItemConfig(key="history", source="session")]),
            ],
            token_budget=1500,
        )

        ctx = _make_context(session_history=history)
        result = await engine.compile(config, ctx)

        # Summarization should have been called
        assert summarize_called
        # Budget must be respected (might still need truncation for very large inputs)
        assert result.tokens <= 1500


class TestBudgetChangeBetweenSessions:
    """Test the scenario: budget changes between sessions with existing history."""

    async def test_budget_reduction_with_existing_history(self):
        """Changing budget from 32k to 16k with 20k+ history enforces new budget."""
        # Phase 1: Compile with large budget — fits fine
        large_history = _make_session_history(40, content_size=500)

        session_resolver = DynamicSessionResolver()
        core_resolver = MockResolver(content="system prompt", tokens=500)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("session", session_resolver)

        handler = TrackingOverflowHandler(reduce_to=500)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=handler,
        )

        # First compile with generous budget — should fit
        config_32k = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="sys", source="core_src")]),
                LayerConfig(name="conversation", cache_policy="always_new",
                            contents=[ContentItemConfig(key="history", source="session")]),
            ],
            token_budget=32000,
        )

        ctx = _make_context(session_history=large_history)
        result_32k = await engine.compile(config_32k, ctx)
        # With 40 turns * ~128 tokens each + 500 core, total ~5620 — fits in 32k
        assert result_32k.tokens <= 32000
        assert handler.call_count == 0  # No overflow

        # Phase 2: Same history but smaller budget — must enforce
        config_4k = _make_config(
            [
                LayerConfig(name="core", cache_policy="always_new",
                            contents=[ContentItemConfig(key="sys", source="core_src")]),
                LayerConfig(name="conversation", cache_policy="always_new",
                            contents=[ContentItemConfig(key="history", source="session")]),
            ],
            token_budget=4000,
        )

        result_4k = await engine.compile(config_4k, ctx)

        # Budget MUST be enforced
        assert result_4k.tokens <= 4000
        # Handler should have been called
        assert handler.call_count >= 1

    async def test_budget_reduction_triggers_handler_before_truncation(self):
        """Budget reduction with existing history calls handler first, truncates only if needed."""
        handler_calls = []

        async def tracking_handler(layers, budget, config, ctx):
            total = sum(c.tokens for contents in layers.values() for c in contents)
            handler_calls.append({"total": total, "budget": budget})
            # Don't reduce — let truncation handle it
            return None

        big_content = "x" * 8000  # ~2000 tokens
        session_resolver = MockResolver(content=big_content, tokens=2000)
        core_resolver = MockResolver(content="core", tokens=500)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("session_src", session_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=tracking_handler,
        )

        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="conv", cache_policy="always_new",
                            contents=[ContentItemConfig(key="h", source="session_src")]),
            ],
            token_budget=1500,
        )

        result = await engine.compile(config, _make_context())

        # Handler was called first
        assert len(handler_calls) == 1
        assert handler_calls[0]["total"] == 2500
        assert handler_calls[0]["budget"] == 1500

        # Then truncation kicked in
        assert engine.truncation_events == 1
        assert result.tokens <= 1500

    async def test_multiple_budget_reductions_cumulate_truncation_events(self):
        """Multiple compiles with budget pressure accumulate truncation events."""
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(content="x" * 3200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        ctx = _make_context()

        await engine.compile(config, ctx)
        assert engine.truncation_events == 1

        await engine.compile(config, ctx)
        assert engine.truncation_events == 2

        await engine.compile(config, ctx)
        assert engine.truncation_events == 3


class TestInterfaceSpecificBudgets:
    """Tests for per-interface token budget enforcement."""

    @pytest.mark.parametrize(
        "interface_name,token_budget",
        [
            ("heartbeat", 3000),
            ("a2a", 8000),
            ("user_message", 48000),
        ],
    )
    async def test_interface_budget_enforced(self, interface_name, token_budget):
        """Different interfaces have different token budgets; each is enforced."""
        # Create a resolver that produces content near each budget boundary
        content_tokens = token_budget + 500  # Exceeds budget to trigger enforcement
        big_resolver = MockResolver(content="x" * (content_tokens * 4), tokens=content_tokens)
        core_resolver = MockResolver(content="core", tokens=200)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(
                    name="core",
                    cache_policy="immutable",
                    contents=[ContentItemConfig(key="c", source="core_src")],
                ),
                LayerConfig(
                    name="dynamic",
                    cache_policy="always_new",
                    contents=[ContentItemConfig(key="d", source="big_src")],
                ),
            ],
            token_budget=token_budget,
        )

        result = await engine.compile(config, _make_context())

        assert result.tokens <= token_budget, (
            f"Interface '{interface_name}' budget {token_budget} exceeded: "
            f"got {result.tokens} tokens"
        )

    async def test_small_budget_still_preserves_core_layer(self):
        """Even with a very small budget (heartbeat), the core layer is preserved."""
        core_resolver = MockResolver(content="system prompt", tokens=500)
        extra_resolver = MockResolver(content="x" * 12000, tokens=3000)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("extra_src", extra_resolver)

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
                    cache_policy="always_new",
                    contents=[ContentItemConfig(key="e", source="extra_src")],
                ),
            ],
            token_budget=3000,
        )

        result = await engine.compile(config, _make_context())

        # Core layer must be untouched
        assert result.layers["core"][0].tokens == 500
        assert result.layers["core"][0].content == "system prompt"
        assert result.tokens <= 3000


class TestRouterDrivenBudgetEnforcement:
    """Integration: InterfaceRouter selects different configs per interface,
    each with its own token budget, and the engine enforces them independently.
    """

    def _build_engine_and_router(self, interface_budgets: dict[str, int]):
        """Build a PipelineEngine + InterfaceRouter with per-interface budgets.

        All interfaces share the same two-layer config (core + dynamic),
        differing only in token_budget.
        """
        from sr2.config.loader import ConfigLoader
        from sr2.pipeline.router import InterfaceRouter

        core_resolver = MockResolver(content="system prompt", tokens=500)
        big_resolver = MockResolver(content="x" * 40000, tokens=10000)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())

        interfaces = {}
        for name, budget in interface_budgets.items():
            interfaces[name] = {
                "token_budget": budget,
                "layers": [
                    {
                        "name": "core",
                        "cache_policy": "immutable",
                        "contents": [{"key": "c", "source": "core_src"}],
                    },
                    {
                        "name": "dynamic",
                        "cache_policy": "always_new",
                        "contents": [{"key": "d", "source": "big_src"}],
                    },
                ],
            }

        loader = ConfigLoader()
        router = InterfaceRouter(interfaces, loader)
        return engine, router

    async def test_each_interface_gets_its_own_budget(self):
        """Route to 3 interfaces with different budgets; each enforced independently."""
        budgets = {"heartbeat": 3000, "a2a": 8000, "user_message": 48000}
        engine, router = self._build_engine_and_router(budgets)

        for interface_name, expected_budget in budgets.items():
            config = router.route(interface_name)
            assert config.token_budget == expected_budget

            result = await engine.compile(config, _make_context())
            assert result.tokens <= expected_budget, (
                f"Interface '{interface_name}' exceeded its budget of {expected_budget}: "
                f"got {result.tokens}"
            )

    async def test_same_content_different_budgets_different_truncation(self):
        """Same resolver content compiled under different budgets produces different token counts."""
        budgets = {"small": 3000, "large": 48000}
        engine, router = self._build_engine_and_router(budgets)

        small_config = router.route("small")
        large_config = router.route("large")

        result_small = await engine.compile(small_config, _make_context())
        result_large = await engine.compile(large_config, _make_context())

        # Small budget must truncate more aggressively
        assert result_small.tokens <= 3000
        assert result_large.tokens <= 48000
        assert result_small.tokens < result_large.tokens

    async def test_router_caches_config_per_interface(self):
        """Router caches configs so repeated route() calls return the same object."""
        from sr2.config.loader import ConfigLoader
        from sr2.pipeline.router import InterfaceRouter

        loader = ConfigLoader()
        router = InterfaceRouter(
            {
                "chat": {"token_budget": 24000, "layers": []},
                "heartbeat": {"token_budget": 3000, "layers": []},
            },
            loader,
        )

        chat1 = router.route("chat")
        chat2 = router.route("chat")
        heartbeat = router.route("heartbeat")

        assert chat1 is chat2, "Router should cache configs"
        assert chat1.token_budget != heartbeat.token_budget, (
            "Different interfaces should have different budgets"
        )


class TestPreRotationThreshold:
    """Tests for pre-emptive context rotation threshold."""

    def test_should_rotate_at_75_percent(self):
        """pre_rot_threshold=0.25 means rotation triggers at 75% usage (1 - 0.25)."""
        from sr2.resolvers.preemptive_rotation_resolver import PreemptiveRotationResolver

        budget = 10000
        threshold = 1 - 0.25  # 0.75 — triggers when 75% consumed

        # At exactly 75% — should rotate
        assert PreemptiveRotationResolver.should_rotate(7500, budget, threshold) is True
        # Above 75% — should rotate
        assert PreemptiveRotationResolver.should_rotate(8000, budget, threshold) is True
        # Below 75% — should NOT rotate
        assert PreemptiveRotationResolver.should_rotate(7000, budget, threshold) is False

    def test_rotation_status_details(self):
        """get_rotation_status returns correct ratio and threshold info."""
        from sr2.resolvers.preemptive_rotation_resolver import PreemptiveRotationResolver

        status = PreemptiveRotationResolver.get_rotation_status(
            current_tokens=7500, token_budget=10000, threshold=0.75
        )

        assert status["should_rotate"] is True
        assert status["ratio"] == 0.75
        assert status["threshold"] == 0.75
        assert status["tokens_until_rotation"] == 0

    def test_rotation_not_triggered_below_threshold(self):
        """Rotation not triggered when usage is below threshold."""
        from sr2.resolvers.preemptive_rotation_resolver import PreemptiveRotationResolver

        status = PreemptiveRotationResolver.get_rotation_status(
            current_tokens=5000, token_budget=10000, threshold=0.75
        )

        assert status["should_rotate"] is False
        assert status["ratio"] == 0.5
        assert status["tokens_until_rotation"] == 2500  # 7500 - 5000

    def test_pre_rot_threshold_config_default(self):
        """PipelineConfig.pre_rot_threshold defaults to 0.25."""
        config = PipelineConfig(token_budget=10000, layers=[])
        assert config.pre_rot_threshold == 0.25

    def test_zero_budget_does_not_crash(self):
        """should_rotate with zero budget returns False (no division by zero)."""
        from sr2.resolvers.preemptive_rotation_resolver import PreemptiveRotationResolver

        assert PreemptiveRotationResolver.should_rotate(100, 0, 0.75) is False


class TestBudgetEnforcementLogging:
    """Tests for proper logging during budget enforcement."""

    async def test_warning_logged_on_truncation(self, caplog):
        """Warning is logged when truncation occurs."""
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(content="x" * 3200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(resolvers, create_default_cache_registry())
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        with caplog.at_level(logging.WARNING):
            await engine.compile(config, _make_context())

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("budget exceeded" in m.lower() for m in warning_messages)
        assert any("truncat" in m.lower() for m in warning_messages)

    async def test_info_logged_when_handler_succeeds(self, caplog):
        """Info is logged when handler successfully reduces content."""
        handler = TrackingOverflowHandler(reduce_to=500)
        core_resolver = MockResolver(tokens=500)
        big_resolver = MockResolver(content="x" * 3200, tokens=800)

        resolvers = ContentResolverRegistry()
        resolvers.register("core_src", core_resolver)
        resolvers.register("big_src", big_resolver)

        engine = PipelineEngine(
            resolvers, create_default_cache_registry(),
            budget_overflow_handler=handler,
        )
        config = _make_config(
            [
                LayerConfig(name="core", cache_policy="immutable",
                            contents=[ContentItemConfig(key="c", source="core_src")]),
                LayerConfig(name="extra", cache_policy="always_new",
                            contents=[ContentItemConfig(key="e", source="big_src")]),
            ],
            token_budget=1000,
        )

        with caplog.at_level(logging.INFO):
            await engine.compile(config, _make_context())

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("handler reduced" in m.lower() for m in info_messages)
        assert engine.truncation_events == 0
