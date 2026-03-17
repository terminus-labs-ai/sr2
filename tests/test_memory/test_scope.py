"""Tests for memory scope isolation and sharing."""

import json

import pytest

from sr2.config.models import MemoryScopeConfig
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


# ---------- Extraction: scope stamping ----------


class TestScopeStamping:
    """Tests for scope stamping during extraction."""

    @pytest.mark.asyncio
    async def test_project_write_stamps_scope(self, store):
        """Extract with default_write='project' + project context → project scope."""
        scope_config = MemoryScopeConfig(
            default_write="project",
            agent_name="liara",
        )
        response = json.dumps([{"key": "research.auth", "value": "OAuth2 chosen"}])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(
            llm_callable=mock_llm, store=store, scope_config=scope_config,
        )
        result = await extractor.extract(
            "We decided on OAuth2",
            current_context={"project_id": "galaxy-map", "source": "gm_task:GM-47"},
        )

        assert len(result.memories) == 1
        mem = result.memories[0]
        assert mem.scope == "project"
        assert mem.scope_ref == "galaxy-map"
        assert mem.source == "gm_task:GM-47"

    @pytest.mark.asyncio
    async def test_private_write_stamps_agent(self, store):
        """Extract with default_write='private' + agent_name → private with agent ref."""
        scope_config = MemoryScopeConfig(
            default_write="private",
            agent_name="tali",
        )
        response = json.dumps([{"key": "user.name", "value": "Shepard"}])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(
            llm_callable=mock_llm, store=store, scope_config=scope_config,
        )
        result = await extractor.extract("My name is Shepard")

        assert len(result.memories) == 1
        mem = result.memories[0]
        assert mem.scope == "private"
        assert mem.scope_ref == "agent:tali"

    @pytest.mark.asyncio
    async def test_project_fallback_without_context(self, store):
        """Extract with default_write='project' but no project_id → falls back to private."""
        scope_config = MemoryScopeConfig(
            default_write="project",
            agent_name="garrus",
        )
        response = json.dumps([{"key": "user.pref", "value": "dark mode"}])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(
            llm_callable=mock_llm, store=store, scope_config=scope_config,
        )
        # No current_context at all
        result = await extractor.extract("I prefer dark mode")

        assert len(result.memories) == 1
        mem = result.memories[0]
        assert mem.scope == "private"
        assert mem.scope_ref == "agent:garrus"

    @pytest.mark.asyncio
    async def test_no_scope_config_legacy_behavior(self, store):
        """No scope config → memories get default scope='private', no scope_ref."""
        response = json.dumps([{"key": "user.name", "value": "Alice"}])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("I'm Alice")

        assert len(result.memories) == 1
        mem = result.memories[0]
        assert mem.scope == "private"
        assert mem.scope_ref is None
        assert mem.source is None

    @pytest.mark.asyncio
    async def test_source_field_stored(self, store):
        """Provenance source is stored and retrievable."""
        scope_config = MemoryScopeConfig(default_write="private", agent_name="edi")
        response = json.dumps([{"key": "task.status", "value": "complete"}])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(
            llm_callable=mock_llm, store=store, scope_config=scope_config,
        )
        result = await extractor.extract(
            "Task done",
            current_context={"source": "gm_task:GM-99"},
        )

        mem = result.memories[0]
        assert mem.source == "gm_task:GM-99"
        # Verify it persisted to the store
        stored = await store.get(mem.id)
        assert stored.source == "gm_task:GM-99"


# ---------- Retrieval: scope filtering ----------


class TestScopeRetrieval:
    """Tests for scope-aware retrieval."""

    @pytest.mark.asyncio
    async def test_private_isolation(self, store):
        """Agent A's private memories don't appear in Agent B's retrieval."""
        mem_a = Memory(key="user.name", value="Alice fact", scope="private", scope_ref="agent:a")
        mem_b = Memory(key="user.name", value="Bob fact", scope="private", scope_ref="agent:b")
        await store.save(mem_a)
        await store.save(mem_b)

        scope_b = MemoryScopeConfig(default_read=["private"], agent_name="b")
        retriever = HybridRetriever(
            store=store, strategy="keyword",
            scope_config=scope_b, current_context={},
        )
        results = await retriever.retrieve("fact")

        assert len(results) == 1
        assert results[0].memory.scope_ref == "agent:b"

    @pytest.mark.asyncio
    async def test_project_sharing(self, store):
        """Agent A writes project memory, Agent B on same project retrieves it."""
        shared = Memory(
            key="decision.auth", value="OAuth2",
            scope="project", scope_ref="proj-1",
        )
        await store.save(shared)

        scope_b = MemoryScopeConfig(
            default_read=["private", "project"], agent_name="b",
        )
        retriever = HybridRetriever(
            store=store, strategy="keyword",
            scope_config=scope_b, current_context={"project_id": "proj-1"},
        )
        results = await retriever.retrieve("OAuth2")

        assert len(results) == 1
        assert results[0].memory.value == "OAuth2"

    @pytest.mark.asyncio
    async def test_cross_project_isolation(self, store):
        """Agent on project X doesn't see project Y's memories."""
        mem_x = Memory(key="decision.db", value="PostgreSQL", scope="project", scope_ref="proj-x")
        mem_y = Memory(key="decision.db", value="MySQL", scope="project", scope_ref="proj-y")
        await store.save(mem_x)
        await store.save(mem_y)

        scope = MemoryScopeConfig(
            default_read=["private", "project"], agent_name="agent1",
        )
        retriever = HybridRetriever(
            store=store, strategy="keyword",
            scope_config=scope, current_context={"project_id": "proj-x"},
        )
        results = await retriever.retrieve("decision")

        assert len(results) == 1
        assert results[0].memory.scope_ref == "proj-x"

    @pytest.mark.asyncio
    async def test_legacy_memories_visible(self, store):
        """Memories with scope='private', scope_ref=None surface for any agent."""
        legacy = Memory(key="user.name", value="legacy fact")
        assert legacy.scope == "private"
        assert legacy.scope_ref is None
        await store.save(legacy)

        scope = MemoryScopeConfig(default_read=["private"], agent_name="newagent")
        retriever = HybridRetriever(
            store=store, strategy="keyword",
            scope_config=scope, current_context={},
        )
        results = await retriever.retrieve("legacy")

        assert len(results) == 1
        assert results[0].memory.value == "legacy fact"

    @pytest.mark.asyncio
    async def test_no_scope_config_returns_all(self, store):
        """Agent with no scope config gets all memories (existing behavior)."""
        await store.save(Memory(key="k1", value="private fact", scope="private", scope_ref="agent:a"))
        await store.save(Memory(key="k2", value="project fact", scope="project", scope_ref="proj-1"))
        await store.save(Memory(key="k3", value="legacy fact"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("fact")

        assert len(results) == 3


# ---------- Conflict detection: scope-aware ----------


class TestScopeConflicts:
    """Tests for scope-aware conflict detection."""

    @pytest.mark.asyncio
    async def test_same_key_different_projects_no_conflict(self, store):
        """Same key in different projects doesn't trigger conflict."""
        existing = Memory(
            key="decision.db", value="PostgreSQL",
            scope="project", scope_ref="proj-a",
        )
        await store.save(existing)

        detector = ConflictDetector(store=store)
        new = Memory(
            key="decision.db", value="MySQL",
            scope="project", scope_ref="proj-b",
        )
        conflicts = await detector.detect(new)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_same_key_same_project_triggers_conflict(self, store):
        """Same key in same project triggers conflict."""
        existing = Memory(
            key="decision.db", value="PostgreSQL",
            scope="project", scope_ref="proj-a",
        )
        await store.save(existing)

        detector = ConflictDetector(store=store)
        new = Memory(
            key="decision.db", value="MySQL",
            scope="project", scope_ref="proj-a",
        )
        conflicts = await detector.detect(new)

        assert len(conflicts) == 1
        assert conflicts[0].existing_memory.value == "PostgreSQL"


# ---------- Store-level scope filtering ----------


class TestInMemoryStoreScopeFiltering:
    """Tests for scope filtering in InMemoryMemoryStore."""

    @pytest.mark.asyncio
    async def test_search_keyword_scope_filter(self, store):
        """search_keyword respects scope_filter and scope_refs."""
        await store.save(Memory(key="k1", value="alpha fact", scope="private", scope_ref="agent:a"))
        await store.save(Memory(key="k2", value="beta fact", scope="project", scope_ref="proj-1"))
        await store.save(Memory(key="k3", value="gamma fact", scope="private", scope_ref="agent:b"))

        results = await store.search_keyword(
            "fact", scope_filter=["private"], scope_refs=["agent:a"],
        )
        assert len(results) == 1
        assert results[0].memory.scope_ref == "agent:a"

    @pytest.mark.asyncio
    async def test_search_vector_scope_filter(self, store):
        """search_vector respects scope filtering."""
        await store.save(Memory(key="k1", value="val1", scope="project", scope_ref="proj-1"))
        await store.save(Memory(key="k2", value="val2", scope="project", scope_ref="proj-2"))
        await store.save(Memory(key="k3", value="val3", scope="private", scope_ref="agent:a"))

        results = await store.search_vector(
            [0.1], scope_filter=["project"], scope_refs=["proj-1"],
        )
        assert len(results) == 1
        assert results[0].memory.scope_ref == "proj-1"

    @pytest.mark.asyncio
    async def test_get_by_key_scope_filter(self, store):
        """get_by_key respects scope filtering."""
        await store.save(Memory(key="user.name", value="Alice", scope="private", scope_ref="agent:a"))
        await store.save(Memory(key="user.name", value="Bob", scope="private", scope_ref="agent:b"))

        results = await store.get_by_key(
            "user.name", scope_filter=["private"], scope_refs=["agent:b"],
        )
        assert len(results) == 1
        assert results[0].value == "Bob"

    @pytest.mark.asyncio
    async def test_scope_filter_none_returns_all(self, store):
        """scope_filter=None returns all memories (backward compat)."""
        await store.save(Memory(key="k1", value="fact1", scope="private", scope_ref="agent:a"))
        await store.save(Memory(key="k2", value="fact2", scope="project", scope_ref="proj-1"))

        results = await store.search_keyword("fact")
        assert len(results) == 2
