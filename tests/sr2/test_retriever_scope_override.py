"""Tests for HybridRetriever.retrieve() scope_override and skip_touch parameters.

These tests verify:
- scope_override is accepted as an optional parameter
- scope_override takes precedence over the instance's scope config
- The instance's scope config is NOT mutated during retrieval with override
- skip_touch is accepted as an optional parameter
- skip_touch=True leaves _pending_touch_ids empty after retrieval
- Concurrent calls with different scope overrides don't interfere

Note: scope_config is accessed via the public property (retriever.scope_config),
not the name-mangled private attribute, per Part E of the design.
"""

import asyncio

import pytest

from sr2.config.models import MemoryScopeConfig
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    return InMemoryMemoryStore()


def _make_retriever(store, scope_config=None) -> HybridRetriever:
    return HybridRetriever(store=store, strategy="keyword", scope_config=scope_config)


def _private_scope(agent_name: str = "agent_alpha") -> MemoryScopeConfig:
    return MemoryScopeConfig(allowed_read=["private"], agent_name=agent_name)


def _shared_scope() -> MemoryScopeConfig:
    return MemoryScopeConfig(allowed_read=["shared"])


async def _save_memory(
    store: InMemoryMemoryStore,
    key: str,
    value: str,
    scope: str = "private",
    scope_ref: str | None = None,
) -> Memory:
    mem = Memory(key=key, value=value, scope=scope, scope_ref=scope_ref)
    await store.save(mem)
    return mem


# ---------------------------------------------------------------------------
# 1. API surface: scope_override parameter is accepted
# ---------------------------------------------------------------------------


class TestScopeOverrideAPIAccepted:
    """retrieve() must accept scope_override without TypeError."""

    @pytest.mark.asyncio
    async def test_retrieve_accepts_scope_override_kwarg(self, store):
        """Calling retrieve() with scope_override=None raises no error."""
        retriever = _make_retriever(store)
        results = await retriever.retrieve("query", scope_override=None)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_accepts_scope_override_config_instance(self, store):
        """Calling retrieve() with a MemoryScopeConfig raises no error."""
        retriever = _make_retriever(store)
        override = _private_scope("agent_beta")
        results = await retriever.retrieve("query", scope_override=override)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_accepts_skip_touch_kwarg(self, store):
        """Calling retrieve() with skip_touch=True raises no error."""
        retriever = _make_retriever(store)
        results = await retriever.retrieve("query", skip_touch=True)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_accepts_all_new_params_together(self, store):
        """Calling retrieve() with both new params raises no error."""
        retriever = _make_retriever(store)
        override = _shared_scope()
        results = await retriever.retrieve("query", scope_override=override, skip_touch=False)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 2. scope_override takes precedence over instance scope config
# ---------------------------------------------------------------------------


class TestScopeOverridePrecedence:
    """When scope_override is provided, it must determine scope filtering."""

    @pytest.mark.asyncio
    async def test_override_allows_reading_otherwise_excluded_scope(self, store):
        """Memory in 'shared' scope is returned when override allows it,
        even if instance config only allows 'private'."""
        # Save a memory in shared scope
        await _save_memory(store, "shared.key", "shared value", scope="shared")

        # Instance config restricts to private only
        instance_cfg = _private_scope("agent_alpha")
        retriever = _make_retriever(store, scope_config=instance_cfg)

        # Override allows shared
        override = _shared_scope()
        results = await retriever.retrieve("shared", scope_override=override)

        values = [r.memory.value for r in results]
        assert "shared value" in values

    @pytest.mark.asyncio
    async def test_override_restricts_to_narrower_scope(self, store):
        """Memory in 'private' scope is excluded when override limits to 'shared'."""
        await _save_memory(
            store, "priv.key", "private value", scope="private", scope_ref="agent:alpha"
        )
        await _save_memory(store, "shared.key", "shared value", scope="shared")

        # Instance config allows private
        instance_cfg = _private_scope("alpha")
        retriever = _make_retriever(store, scope_config=instance_cfg)

        # Override narrows to shared only
        override = _shared_scope()
        results = await retriever.retrieve("value", scope_override=override)

        values = [r.memory.value for r in results]
        assert "private value" not in values

    @pytest.mark.asyncio
    async def test_no_override_falls_back_to_instance_config(self, store):
        """Without scope_override, the instance config is used (baseline check)."""
        await _save_memory(
            store, "priv.key", "private value", scope="private", scope_ref="agent:alpha"
        )
        await _save_memory(store, "shared.key", "shared value", scope="shared")

        instance_cfg = _private_scope("alpha")
        retriever = _make_retriever(store, scope_config=instance_cfg)

        # No override — uses instance config (private only)
        results = await retriever.retrieve("value")
        values = [r.memory.value for r in results]
        # Shared is excluded; private may or may not appear (depends on scope_ref filter)
        assert "shared value" not in values

    @pytest.mark.asyncio
    async def test_none_override_uses_instance_config(self, store):
        """Explicitly passing scope_override=None falls back to instance config."""
        await _save_memory(store, "shared.key", "shared value", scope="shared")

        instance_cfg = _private_scope("alpha")
        retriever = _make_retriever(store, scope_config=instance_cfg)

        results = await retriever.retrieve("shared", scope_override=None)
        values = [r.memory.value for r in results]
        assert "shared value" not in values


# ---------------------------------------------------------------------------
# 3. Instance scope config is NOT mutated
# ---------------------------------------------------------------------------


class TestNoMutation:
    """The instance's scope_config must remain unchanged after a scoped call."""

    @pytest.mark.asyncio
    async def test_instance_scope_config_unchanged_after_override(self, store):
        """After a retrieve() with scope_override, instance.scope_config is unmodified."""
        original_cfg = _private_scope("agent_alpha")
        retriever = _make_retriever(store, scope_config=original_cfg)

        override = _shared_scope()
        await retriever.retrieve("query", scope_override=override)

        # The instance attribute must still be the original object
        assert retriever.scope_config is original_cfg
        assert retriever.scope_config.allowed_read == ["private"]
        assert retriever.scope_config.agent_name == "agent_alpha"

    @pytest.mark.asyncio
    async def test_instance_scope_config_unchanged_after_multiple_overrides(self, store):
        """Multiple override calls don't accumulate state changes."""
        original_cfg = _private_scope("agent_alpha")
        retriever = _make_retriever(store, scope_config=original_cfg)

        for agent_name in ["beta", "gamma", "delta"]:
            override = _private_scope(agent_name)
            await retriever.retrieve("query", scope_override=override)

        assert retriever.scope_config is original_cfg
        assert retriever.scope_config.agent_name == "agent_alpha"

    @pytest.mark.asyncio
    async def test_override_config_object_not_aliased_to_instance(self, store):
        """The override config object is not stored on the retriever after the call."""
        original_cfg = _private_scope("alpha")
        retriever = _make_retriever(store, scope_config=original_cfg)

        override = _shared_scope()
        await retriever.retrieve("query", scope_override=override)

        # Instance's scope config must not have become the override
        assert retriever.scope_config is not override
        assert retriever.scope_config.allowed_read != ["shared"]

    @pytest.mark.asyncio
    async def test_retriever_with_no_scope_config_and_override(self, store):
        """Retriever with scope_config=None; override is used for the call only."""
        await _save_memory(store, "shared.key", "shared value", scope="shared")
        retriever = _make_retriever(store, scope_config=None)

        override = _shared_scope()
        results = await retriever.retrieve("shared", scope_override=override)

        # Override should filter to shared scope
        values = [r.memory.value for r in results]
        assert "shared value" in values

        # After the call, scope_config is still None
        assert retriever.scope_config is None


# ---------------------------------------------------------------------------
# 4. skip_touch parameter
# ---------------------------------------------------------------------------


class TestSkipTouch:
    """skip_touch=True must leave _pending_touch_ids empty after retrieval."""

    @pytest.mark.asyncio
    async def test_skip_touch_false_populates_pending_touch_ids(self, store):
        """Default behavior: _pending_touch_ids is populated after retrieval."""
        await _save_memory(store, "user.fact", "some fact")
        retriever = _make_retriever(store)

        await retriever.retrieve("fact", skip_touch=False)

        assert len(retriever._pending_touch_ids) > 0

    @pytest.mark.asyncio
    async def test_skip_touch_true_leaves_pending_touch_ids_empty(self, store):
        """skip_touch=True: _pending_touch_ids must be empty after retrieval."""
        await _save_memory(store, "user.fact", "some fact")
        retriever = _make_retriever(store)

        await retriever.retrieve("fact", skip_touch=True)

        assert retriever._pending_touch_ids == []

    @pytest.mark.asyncio
    async def test_skip_touch_default_is_false(self, store):
        """Omitting skip_touch behaves the same as skip_touch=False."""
        await _save_memory(store, "user.fact", "another fact")
        retriever = _make_retriever(store)

        # Pre-condition: no pending touches
        assert retriever._pending_touch_ids == []

        await retriever.retrieve("fact")  # no skip_touch arg

        assert len(retriever._pending_touch_ids) > 0

    @pytest.mark.asyncio
    async def test_skip_touch_true_does_not_clear_preexisting_pending_touches(self, store):
        """skip_touch=True must not wipe pending touches from a prior call."""
        await _save_memory(store, "user.fact1", "fact one")
        await _save_memory(store, "user.fact2", "fact two")
        retriever = _make_retriever(store)

        # First call WITHOUT skip_touch populates pending touches
        await retriever.retrieve("fact one")
        ids_after_first = list(retriever._pending_touch_ids)
        assert len(ids_after_first) > 0

        # Second call WITH skip_touch=True — must not clobber ids_after_first
        # (This tests that skip_touch means "don't write to pending_touch_ids for THIS call",
        #  not "clear all pending touches".)
        await retriever.retrieve("fact two", skip_touch=True)

        # The pending IDs from the first call should be preserved
        # OR skip_touch=True means we don't overwrite — either way, nothing wiped
        # The key invariant: the IDs from the first call are still in _pending_touch_ids
        # (since skip_touch=True must not assign a fresh empty list overwriting them)
        assert retriever._pending_touch_ids == ids_after_first

    @pytest.mark.asyncio
    async def test_skip_touch_with_empty_results(self, store):
        """skip_touch=True with no results still leaves _pending_touch_ids empty."""
        retriever = _make_retriever(store)
        await retriever.retrieve("no match here", skip_touch=True)
        assert retriever._pending_touch_ids == []

    @pytest.mark.asyncio
    async def test_skip_touch_combined_with_scope_override(self, store):
        """scope_override + skip_touch=True work together without side effects."""
        await _save_memory(store, "shared.key", "shared value", scope="shared")
        instance_cfg = _private_scope("alpha")
        retriever = _make_retriever(store, scope_config=instance_cfg)

        override = _shared_scope()
        await retriever.retrieve("shared", scope_override=override, skip_touch=True)

        # No mutation on either axis
        assert retriever.scope_config is instance_cfg
        assert retriever._pending_touch_ids == []


# ---------------------------------------------------------------------------
# 5. Thread-safety: concurrent calls with different overrides don't interfere
# ---------------------------------------------------------------------------


class TestConcurrentScopeOverrides:
    """Concurrent retrieve() calls with different scope_overrides must not race."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_different_scope_overrides(self, store):
        """Multiple concurrent retrieve() calls each see only their own override scope."""
        # Save memories in different scopes
        await _save_memory(store, "agent_a.mem", "alpha memory", scope="private",
                           scope_ref="agent:alpha")
        await _save_memory(store, "agent_b.mem", "beta memory", scope="private",
                           scope_ref="agent:beta")

        # Instance scope: private/alpha
        instance_cfg = _private_scope("alpha")
        retriever = _make_retriever(store, scope_config=instance_cfg)

        alpha_override = _private_scope("alpha")
        beta_override = _private_scope("beta")

        # Run both concurrently
        results_a, results_b = await asyncio.gather(
            retriever.retrieve("memory", scope_override=alpha_override),
            retriever.retrieve("memory", scope_override=beta_override),
        )

        # Each call returns results consistent with its override
        values_a = {r.memory.value for r in results_a}
        values_b = {r.memory.value for r in results_b}

        # Cross-contamination: alpha results must not contain beta's memories
        # and vice versa — scope filtering is per-call, not shared state
        assert "beta memory" not in values_a, (
            "alpha call returned beta memory — scope override leaked between concurrent calls"
        )
        assert "alpha memory" not in values_b, (
            "beta call returned alpha memory — scope override leaked between concurrent calls"
        )

    @pytest.mark.asyncio
    async def test_instance_scope_config_unchanged_after_concurrent_calls(self, store):
        """Instance scope config is unchanged after many concurrent overridden calls."""
        original_cfg = _private_scope("original_agent")
        retriever = _make_retriever(store, scope_config=original_cfg)

        overrides = [_private_scope(f"agent_{i}") for i in range(20)]

        await asyncio.gather(
            *[retriever.retrieve("query", scope_override=ov) for ov in overrides]
        )

        assert retriever.scope_config is original_cfg
        assert retriever.scope_config.agent_name == "original_agent"

    @pytest.mark.asyncio
    async def test_concurrent_skip_touch_calls_do_not_accumulate_ids(self, store):
        """Concurrent calls with skip_touch=True don't accumulate pending touch IDs."""
        for i in range(5):
            await _save_memory(store, f"key.{i}", f"value {i}")

        retriever = _make_retriever(store)

        await asyncio.gather(
            *[retriever.retrieve("value", skip_touch=True) for _ in range(10)]
        )

        # After all concurrent skip_touch calls, pending list must be empty
        assert retriever._pending_touch_ids == []
