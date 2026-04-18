"""Tests for MemoryStore.list_memories() query method."""

from datetime import UTC, datetime, timedelta

import pytest

from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


def _mem(
    *,
    key: str = "test.key",
    value: str = "test value",
    memory_type: str = "semi_stable",
    scope: str = "private",
    scope_ref: str | None = None,
    extracted_at: datetime | None = None,
    access_count: int = 0,
    archived: bool = False,
) -> Memory:
    """Build a Memory with explicit test attributes, defaults for the rest."""
    kwargs: dict = dict(
        key=key,
        value=value,
        memory_type=memory_type,
        scope=scope,
        access_count=access_count,
        archived=archived,
    )
    if scope_ref is not None:
        kwargs["scope_ref"] = scope_ref
    if extracted_at is not None:
        kwargs["extracted_at"] = extracted_at
    return Memory(**kwargs)


async def _save_all(store: InMemoryMemoryStore, memories: list[Memory]) -> None:
    """Save multiple memories to the store."""
    for m in memories:
        await store.save(m)


class TestListMemoriesNoFilters:
    """Behavior 1 & 10: basic returns."""

    @pytest.mark.asyncio
    async def test_returns_all_non_archived(self, store):
        """No filters returns every non-archived memory."""
        mems = [_mem(key=f"k{i}") for i in range(3)]
        await _save_all(store, mems)

        result = await store.list_memories()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list(self, store):
        """Empty store returns []."""
        result = await store.list_memories()

        assert result == []


class TestListMemoriesTypeFilter:
    """Behavior 2: memory_types filter."""

    @pytest.mark.asyncio
    async def test_single_type(self, store):
        """memory_types=['ephemeral'] returns only ephemeral memories."""
        await _save_all(store, [
            _mem(key="a", memory_type="ephemeral"),
            _mem(key="b", memory_type="semi_stable"),
            _mem(key="c", memory_type="identity"),
        ])

        result = await store.list_memories(memory_types=["ephemeral"])

        assert len(result) == 1
        assert result[0].key == "a"
        assert result[0].memory_type == "ephemeral"

    @pytest.mark.asyncio
    async def test_multiple_types(self, store):
        """memory_types=['ephemeral', 'dynamic'] returns both types."""
        await _save_all(store, [
            _mem(key="a", memory_type="ephemeral"),
            _mem(key="b", memory_type="dynamic"),
            _mem(key="c", memory_type="identity"),
        ])

        result = await store.list_memories(memory_types=["ephemeral", "dynamic"])

        assert len(result) == 2
        types = {m.memory_type for m in result}
        assert types == {"ephemeral", "dynamic"}

    @pytest.mark.asyncio
    async def test_type_none_returns_all(self, store):
        """memory_types=None doesn't filter by type."""
        await _save_all(store, [
            _mem(key="a", memory_type="ephemeral"),
            _mem(key="b", memory_type="identity"),
        ])

        result = await store.list_memories(memory_types=None)

        assert len(result) == 2


class TestListMemoriesOlderThan:
    """Behavior 3: older_than filter."""

    @pytest.mark.asyncio
    async def test_older_than_filters_by_extracted_at(self, store):
        """Only memories with extracted_at before the cutoff are returned."""
        cutoff = datetime(2025, 6, 1, tzinfo=UTC)
        await _save_all(store, [
            _mem(key="old", extracted_at=datetime(2025, 1, 1, tzinfo=UTC)),
            _mem(key="new", extracted_at=datetime(2025, 12, 1, tzinfo=UTC)),
        ])

        result = await store.list_memories(older_than=cutoff)

        assert len(result) == 1
        assert result[0].key == "old"

    @pytest.mark.asyncio
    async def test_older_than_excludes_exact_match(self, store):
        """Memory with extracted_at == older_than is excluded (strictly before)."""
        cutoff = datetime(2025, 6, 1, tzinfo=UTC)
        await _save_all(store, [
            _mem(key="exact", extracted_at=cutoff),
        ])

        result = await store.list_memories(older_than=cutoff)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_older_than_none_returns_all(self, store):
        """older_than=None doesn't filter by time."""
        await _save_all(store, [
            _mem(key="a", extracted_at=datetime(2020, 1, 1, tzinfo=UTC)),
            _mem(key="b", extracted_at=datetime(2030, 1, 1, tzinfo=UTC)),
        ])

        result = await store.list_memories(older_than=None)

        assert len(result) == 2


class TestListMemoriesScopeFilter:
    """Behavior 4: scope_filter."""

    @pytest.mark.asyncio
    async def test_scope_filter_single(self, store):
        """scope_filter=['project'] returns only project-scoped memories."""
        await _save_all(store, [
            _mem(key="a", scope="private", scope_ref="agent:a"),
            _mem(key="b", scope="project", scope_ref="proj-1"),
        ])

        result = await store.list_memories(scope_filter=["project"])

        assert len(result) == 1
        assert result[0].scope == "project"

    @pytest.mark.asyncio
    async def test_scope_filter_multiple(self, store):
        """scope_filter=['private', 'project'] returns both scopes."""
        await _save_all(store, [
            _mem(key="a", scope="private"),
            _mem(key="b", scope="project"),
        ])

        result = await store.list_memories(scope_filter=["private", "project"])

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_scope_filter_none_returns_all(self, store):
        """scope_filter=None doesn't filter by scope."""
        await _save_all(store, [
            _mem(key="a", scope="private"),
            _mem(key="b", scope="project"),
        ])

        result = await store.list_memories(scope_filter=None)

        assert len(result) == 2


class TestListMemoriesScopeRefs:
    """Behavior 5: scope_refs filter."""

    @pytest.mark.asyncio
    async def test_scope_refs_filters_by_scope_ref(self, store):
        """scope_refs=['agent:a'] returns only memories with that scope_ref."""
        await _save_all(store, [
            _mem(key="a", scope="private", scope_ref="agent:a"),
            _mem(key="b", scope="private", scope_ref="agent:b"),
        ])

        result = await store.list_memories(scope_refs=["agent:a"])

        assert len(result) == 1
        assert result[0].scope_ref == "agent:a"

    @pytest.mark.asyncio
    async def test_scope_refs_multiple(self, store):
        """scope_refs with multiple values returns memories matching any."""
        await _save_all(store, [
            _mem(key="a", scope="project", scope_ref="proj-1"),
            _mem(key="b", scope="project", scope_ref="proj-2"),
            _mem(key="c", scope="project", scope_ref="proj-3"),
        ])

        result = await store.list_memories(scope_refs=["proj-1", "proj-3"])

        assert len(result) == 2
        refs = {m.scope_ref for m in result}
        assert refs == {"proj-1", "proj-3"}

    @pytest.mark.asyncio
    async def test_scope_refs_none_returns_all(self, store):
        """scope_refs=None doesn't filter by scope_ref."""
        await _save_all(store, [
            _mem(key="a", scope_ref="agent:a"),
            _mem(key="b", scope_ref="proj-1"),
        ])

        result = await store.list_memories(scope_refs=None)

        assert len(result) == 2


class TestListMemoriesArchived:
    """Behavior 6: archived filtering."""

    @pytest.mark.asyncio
    async def test_excludes_archived_by_default(self, store):
        """Archived memories are excluded when include_archived=False (default)."""
        await _save_all(store, [
            _mem(key="active"),
            _mem(key="archived", archived=True),
        ])

        result = await store.list_memories()

        assert len(result) == 1
        assert result[0].key == "active"

    @pytest.mark.asyncio
    async def test_includes_archived_when_flag_set(self, store):
        """include_archived=True returns both active and archived."""
        await _save_all(store, [
            _mem(key="active"),
            _mem(key="archived", archived=True),
        ])

        result = await store.list_memories(include_archived=True)

        assert len(result) == 2
        keys = {m.key for m in result}
        assert keys == {"active", "archived"}


class TestListMemoriesAccessCount:
    """Behavior 7: min_access_count / max_access_count."""

    @pytest.mark.asyncio
    async def test_min_access_count(self, store):
        """min_access_count filters out memories below the threshold."""
        await _save_all(store, [
            _mem(key="low", access_count=2),
            _mem(key="high", access_count=10),
        ])

        result = await store.list_memories(min_access_count=5)

        assert len(result) == 1
        assert result[0].key == "high"

    @pytest.mark.asyncio
    async def test_max_access_count(self, store):
        """max_access_count filters out memories above the threshold."""
        await _save_all(store, [
            _mem(key="low", access_count=2),
            _mem(key="high", access_count=10),
        ])

        result = await store.list_memories(max_access_count=5)

        assert len(result) == 1
        assert result[0].key == "low"

    @pytest.mark.asyncio
    async def test_min_and_max_access_count_range(self, store):
        """Both min and max define an inclusive range."""
        await _save_all(store, [
            _mem(key="below", access_count=1),
            _mem(key="in_range", access_count=5),
            _mem(key="above", access_count=20),
        ])

        result = await store.list_memories(min_access_count=3, max_access_count=10)

        assert len(result) == 1
        assert result[0].key == "in_range"

    @pytest.mark.asyncio
    async def test_access_count_boundary_inclusive(self, store):
        """min/max are inclusive — exact boundary values match."""
        await _save_all(store, [
            _mem(key="at_min", access_count=5),
            _mem(key="at_max", access_count=10),
        ])

        result = await store.list_memories(min_access_count=5, max_access_count=10)

        assert len(result) == 2


class TestListMemoriesLimit:
    """Behavior 8: limit."""

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, store):
        """Returns at most `limit` memories."""
        await _save_all(store, [_mem(key=f"k{i}") for i in range(10)])

        result = await store.list_memories(limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_limit_default_is_500(self, store):
        """Default limit allows up to 500 (doesn't truncate small sets)."""
        await _save_all(store, [_mem(key=f"k{i}") for i in range(5)])

        result = await store.list_memories()

        assert len(result) == 5


class TestListMemoriesCombinedFilters:
    """Behavior 9: multiple filters combine with AND logic."""

    @pytest.mark.asyncio
    async def test_type_and_scope_combined(self, store):
        """memory_types + scope_filter applied together (AND)."""
        await _save_all(store, [
            _mem(key="match", memory_type="ephemeral", scope="project", scope_ref="proj-1"),
            _mem(key="wrong_type", memory_type="identity", scope="project", scope_ref="proj-1"),
            _mem(key="wrong_scope", memory_type="ephemeral", scope="private", scope_ref="agent:a"),
        ])

        result = await store.list_memories(
            memory_types=["ephemeral"],
            scope_filter=["project"],
        )

        assert len(result) == 1
        assert result[0].key == "match"

    @pytest.mark.asyncio
    async def test_older_than_and_access_count_combined(self, store):
        """older_than + min_access_count applied together."""
        cutoff = datetime(2025, 6, 1, tzinfo=UTC)
        await _save_all(store, [
            _mem(key="old_high", extracted_at=datetime(2025, 1, 1, tzinfo=UTC), access_count=10),
            _mem(key="old_low", extracted_at=datetime(2025, 1, 1, tzinfo=UTC), access_count=1),
            _mem(key="new_high", extracted_at=datetime(2025, 12, 1, tzinfo=UTC), access_count=10),
        ])

        result = await store.list_memories(older_than=cutoff, min_access_count=5)

        assert len(result) == 1
        assert result[0].key == "old_high"

    @pytest.mark.asyncio
    async def test_all_filters_combined(self, store):
        """All filters applied at once — only the fully matching memory survives."""
        cutoff = datetime(2025, 6, 1, tzinfo=UTC)
        target = _mem(
            key="target",
            memory_type="ephemeral",
            scope="project",
            scope_ref="proj-1",
            extracted_at=datetime(2025, 1, 1, tzinfo=UTC),
            access_count=5,
        )
        decoys = [
            _mem(key="wrong_type", memory_type="identity", scope="project",
                 scope_ref="proj-1", extracted_at=datetime(2025, 1, 1, tzinfo=UTC), access_count=5),
            _mem(key="wrong_scope", memory_type="ephemeral", scope="private",
                 scope_ref="agent:a", extracted_at=datetime(2025, 1, 1, tzinfo=UTC), access_count=5),
            _mem(key="too_new", memory_type="ephemeral", scope="project",
                 scope_ref="proj-1", extracted_at=datetime(2025, 12, 1, tzinfo=UTC), access_count=5),
            _mem(key="too_few_access", memory_type="ephemeral", scope="project",
                 scope_ref="proj-1", extracted_at=datetime(2025, 1, 1, tzinfo=UTC), access_count=1),
            _mem(key="archived_match", memory_type="ephemeral", scope="project",
                 scope_ref="proj-1", extracted_at=datetime(2025, 1, 1, tzinfo=UTC), access_count=5,
                 archived=True),
        ]
        await _save_all(store, [target] + decoys)

        result = await store.list_memories(
            memory_types=["ephemeral"],
            scope_filter=["project"],
            scope_refs=["proj-1"],
            older_than=cutoff,
            min_access_count=3,
            max_access_count=10,
            limit=100,
        )

        assert len(result) == 1
        assert result[0].key == "target"

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty(self, store):
        """Filters that exclude everything return []."""
        await _save_all(store, [
            _mem(key="a", memory_type="identity"),
        ])

        result = await store.list_memories(memory_types=["ephemeral"])

        assert result == []


class TestListMemoriesEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_list_filters_match_nothing(self, store):
        """Empty list filters (vs None) match nothing."""
        await _save_all(store, [_mem(key="a")])

        assert await store.list_memories(memory_types=[]) == []
        assert await store.list_memories(scope_filter=[]) == []
        assert await store.list_memories(scope_refs=[]) == []

    @pytest.mark.asyncio
    async def test_scope_refs_excludes_null_scope_ref(self, store):
        """When scope_refs is set, memories with scope_ref=None are excluded."""
        await _save_all(store, [
            _mem(key="no_ref", scope="project"),  # scope_ref=None
            _mem(key="has_ref", scope="project", scope_ref="proj-1"),
        ])

        result = await store.list_memories(scope_refs=["proj-1"])

        assert len(result) == 1
        assert result[0].key == "has_ref"

    @pytest.mark.asyncio
    async def test_scope_refs_independent_of_scope_filter(self, store):
        """scope_refs works without scope_filter (decoupled)."""
        await _save_all(store, [
            _mem(key="match", scope="private", scope_ref="agent:a"),
            _mem(key="other", scope="private", scope_ref="agent:b"),
        ])

        result = await store.list_memories(scope_refs=["agent:a"])

        assert len(result) == 1
        assert result[0].key == "match"
