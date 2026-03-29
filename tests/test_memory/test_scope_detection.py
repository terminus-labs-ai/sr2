"""Tests for automatic scope detection (ScopeDetector)."""

import json

import pytest

from sr2.config.models import MemoryScopeConfig
from sr2.memory.schema import Memory
from sr2.memory.scope import ScopeDetector
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


# ---------- ScopeDetector unit tests ----------


class TestScopeDetector:
    """Tests for ScopeDetector."""

    @pytest.mark.asyncio
    async def test_private_only_no_llm_call(self, store):
        """Private-only config returns deterministic result without LLM call."""
        scope_config = MemoryScopeConfig(
            allowed_read=["private"], agent_name="miranda",
        )

        async def mock_llm(prompt: str) -> str:
            raise AssertionError("LLM should not be called for private-only config")

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect(
            system_prompt="You are Miranda.",
            user_message="Hello",
        )

        assert result == {"private": "agent:miranda"}

    @pytest.mark.asyncio
    async def test_empty_store_no_llm_call(self, store):
        """Empty store returns deterministic result without LLM call."""
        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project"], agent_name="miranda",
        )

        async def mock_llm(prompt: str) -> str:
            raise AssertionError("LLM should not be called when store is empty")

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect(
            system_prompt="You are Miranda.",
            user_message="Hello",
        )

        assert result == {"private": "agent:miranda"}

    @pytest.mark.asyncio
    async def test_detects_project_from_system_prompt(self, store):
        """LLM detects project from system prompt and known scope_refs."""
        # Seed store with project-scoped memories
        await store.save(Memory(
            key="decision.db", value="PostgreSQL",
            scope="project", scope_ref="normandy-sr2",
        ))
        await store.save(Memory(
            key="decision.auth", value="OAuth2",
            scope="project", scope_ref="citadel",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project"], agent_name="edi",
        )

        async def mock_llm(prompt: str) -> str:
            return json.dumps({"project": "normandy-sr2"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect(
            system_prompt="You are the Normandy SR2 assistant.",
            user_message="Check engine status",
        )

        assert result["private"] == "agent:edi"
        assert result["project"] == "normandy-sr2"

    @pytest.mark.asyncio
    async def test_detects_from_user_message(self, store):
        """Detection works from user message alone when system_prompt is None."""
        await store.save(Memory(
            key="status", value="active",
            scope="project", scope_ref="project-alpha",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["project"], agent_name="agent1",
        )

        async def mock_llm(prompt: str) -> str:
            return json.dumps({"project": "project-alpha"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect(
            system_prompt=None,
            user_message="Check project-alpha status",
        )

        assert result["project"] == "project-alpha"

    @pytest.mark.asyncio
    async def test_llm_returns_garbage(self, store):
        """Garbage LLM response returns deterministic-only result."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project"], agent_name="garrus",
        )

        async def mock_llm(prompt: str) -> str:
            return "I don't know what you mean"

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect(
            system_prompt="test", user_message="test",
        )

        # Only deterministic private scope
        assert result == {"private": "agent:garrus"}

    @pytest.mark.asyncio
    async def test_llm_returns_null_scope_ref(self, store):
        """LLM returning null for a scope sets it to None."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project"], agent_name="tali",
        )

        async def mock_llm(prompt: str) -> str:
            return json.dumps({"project": None})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect(
            system_prompt="test", user_message="test",
        )

        assert result["private"] == "agent:tali"
        assert result.get("project") is None

    @pytest.mark.asyncio
    async def test_caching_per_session(self, store):
        """Same session_id returns cached result (LLM called only once)."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["project"], agent_name="liara",
        )

        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"project": "proj-1"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)

        r1 = await detector.detect("sys", "msg", session_id="sess-1")
        r2 = await detector.detect("sys", "msg", session_id="sess-1")

        assert r1 == r2
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_sessions_not_cached(self, store):
        """Different session_ids trigger separate LLM calls."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["project"], agent_name="liara",
        )

        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"project": "proj-1"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)

        await detector.detect("sys", "msg", session_id="a")
        await detector.detect("sys", "msg", session_id="b")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_clears_cache(self, store):
        """invalidate() clears cached result, forcing re-detection."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["project"], agent_name="liara",
        )

        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"project": "proj-1"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)

        await detector.detect("sys", "msg", session_id="s1")
        detector.invalidate("s1")
        await detector.detect("sys", "msg", session_id="s1")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_session_id_not_cached(self, store):
        """session_id=None triggers LLM call every time."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["project"], agent_name="liara",
        )

        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"project": "proj-1"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)

        await detector.detect("sys", "msg", session_id=None)
        await detector.detect("sys", "msg", session_id=None)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_scopes(self, store):
        """Multiple non-private scopes detected independently."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="normandy",
        ))
        await store.save(Memory(
            key="k2", value="v2", scope="team", scope_ref="engineering",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project", "team"],
            allowed_write=["private", "project", "team"],
            agent_name="edi",
        )

        async def mock_llm(prompt: str) -> str:
            return json.dumps({"project": "normandy", "team": "engineering"})

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect("sys", "msg")

        assert result["private"] == "agent:edi"
        assert result["project"] == "normandy"
        assert result["team"] == "engineering"

    @pytest.mark.asyncio
    async def test_llm_callable_exception(self, store):
        """LLM exception returns deterministic-only result."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project"], agent_name="garrus",
        )

        async def mock_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        detector = ScopeDetector(store=store, llm_callable=mock_llm, scope_config=scope_config)
        result = await detector.detect("sys", "msg")

        assert result == {"private": "agent:garrus"}

    @pytest.mark.asyncio
    async def test_llm_callable_none(self, store):
        """No LLM callable returns deterministic-only result."""
        await store.save(Memory(
            key="k1", value="v1", scope="project", scope_ref="proj-1",
        ))

        scope_config = MemoryScopeConfig(
            allowed_read=["private", "project"], agent_name="garrus",
        )

        detector = ScopeDetector(store=store, llm_callable=None, scope_config=scope_config)
        result = await detector.detect("sys", "msg")

        assert result == {"private": "agent:garrus"}


# ---------- _find_last_json_object tests ----------


class TestFindLastJsonObject:
    """Tests for ScopeDetector._find_last_json_object static method."""

    def test_simple_json(self):
        assert ScopeDetector._find_last_json_object('{"project": "sr2"}') == '{"project": "sr2"}'

    def test_commentary_before_json(self):
        text = 'Here is the result:\n{"project": "sr2"}'
        assert ScopeDetector._find_last_json_object(text) == '{"project": "sr2"}'

    def test_nested_objects(self):
        text = '{"outer": {"inner": "value"}}'
        result = ScopeDetector._find_last_json_object(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_multiple_objects_returns_last(self):
        text = '{"first": 1}\nSome text\n{"second": 2}'
        result = ScopeDetector._find_last_json_object(text)
        assert result == '{"second": 2}'

    def test_no_json(self):
        assert ScopeDetector._find_last_json_object("no json here") is None

    def test_empty_string(self):
        assert ScopeDetector._find_last_json_object("") is None

    def test_none_input(self):
        assert ScopeDetector._find_last_json_object(None) is None

    def test_markdown_wrapped_json(self):
        text = '```json\n{"project": "normandy"}\n```'
        result = ScopeDetector._find_last_json_object(text)
        assert result == '{"project": "normandy"}'


# ---------- Store list_scope_refs tests ----------


class TestListScopeRefs:
    """Tests for InMemoryMemoryStore.list_scope_refs."""

    @pytest.mark.asyncio
    async def test_empty_store(self, store):
        result = await store.list_scope_refs()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_distinct_pairs(self, store):
        """Multiple memories with same scope/scope_ref yield one pair."""
        await store.save(Memory(key="k1", value="v1", scope="project", scope_ref="proj-1"))
        await store.save(Memory(key="k2", value="v2", scope="project", scope_ref="proj-1"))
        await store.save(Memory(key="k3", value="v3", scope="private", scope_ref="agent:a"))

        result = await store.list_scope_refs()
        assert len(result) == 2
        assert ("private", "agent:a") in result
        assert ("project", "proj-1") in result

    @pytest.mark.asyncio
    async def test_excludes_archived(self, store):
        """Archived memories excluded by default."""
        await store.save(Memory(key="k1", value="v1", scope="project", scope_ref="proj-1"))
        mem = Memory(key="k2", value="v2", scope="project", scope_ref="proj-2")
        mem.archived = True
        await store.save(mem)

        result = await store.list_scope_refs()
        assert len(result) == 1
        assert result[0] == ("project", "proj-1")

        # With include_archived
        result_all = await store.list_scope_refs(include_archived=True)
        assert len(result_all) == 2

    @pytest.mark.asyncio
    async def test_with_scope_filter(self, store):
        """scope_filter restricts which scopes are returned."""
        await store.save(Memory(key="k1", value="v1", scope="project", scope_ref="proj-1"))
        await store.save(Memory(key="k2", value="v2", scope="private", scope_ref="agent:a"))

        result = await store.list_scope_refs(scope_filter=["project"])
        assert len(result) == 1
        assert result[0] == ("project", "proj-1")

    @pytest.mark.asyncio
    async def test_includes_null_scope_ref(self, store):
        """Memory with scope_ref=None is included."""
        await store.save(Memory(key="k1", value="v1", scope="project"))  # scope_ref=None

        result = await store.list_scope_refs()
        assert len(result) == 1
        assert result[0] == ("project", None)


# ---------- Retriever update_context tests ----------


class TestRetrieverUpdateContext:
    """Tests for HybridRetriever.update_context."""

    def test_update_context(self):
        from sr2.memory.retrieval import HybridRetriever

        store = InMemoryMemoryStore()
        retriever = HybridRetriever(store=store, strategy="keyword")

        retriever.update_context({"project_id": "sr2"})
        assert retriever._current_context == {"project_id": "sr2"}

    def test_update_context_none(self):
        from sr2.memory.retrieval import HybridRetriever

        store = InMemoryMemoryStore()
        retriever = HybridRetriever(
            store=store, strategy="keyword",
            current_context={"project_id": "old"},
        )

        retriever.update_context(None)
        assert retriever._current_context is None
