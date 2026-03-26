"""Tests for the RecallMemoryTool built-in tool."""

import pytest

from runtime.tool_executor import RecallMemoryTool
from sr2.config.models import MemoryScopeConfig
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory, MemorySearchResult
from sr2.memory.store import InMemoryMemoryStore


def _make_memory(key, value, scope="private", scope_ref=None, memory_type="semi_stable"):
    return Memory(key=key, value=value, scope=scope, scope_ref=scope_ref, memory_type=memory_type)


async def _seed_store(store, memories):
    for mem in memories:
        await store.save(mem)


@pytest.fixture
def store():
    return InMemoryMemoryStore()


@pytest.fixture
def retriever(store):
    return HybridRetriever(store=store, strategy="keyword")


@pytest.fixture
def scope_config():
    return MemoryScopeConfig(
        allowed_read=["private"],
        allowed_write=["private"],
        agent_name="edi",
    )


@pytest.fixture
def scope_config_all():
    return MemoryScopeConfig(
        allowed_read=["private", "project"],
        allowed_write=["private"],
        agent_name="edi",
    )


@pytest.fixture
def tool(retriever, store, scope_config):
    return RecallMemoryTool(
        retriever=retriever,
        memory_store=store,
        scope_config=scope_config,
    )


@pytest.fixture
def tool_all_scopes(retriever, store, scope_config_all):
    return RecallMemoryTool(
        retriever=retriever,
        memory_store=store,
        scope_config=scope_config_all,
    )


class TestBasicRecall:
    @pytest.mark.asyncio
    async def test_basic_query(self, tool, store):
        await _seed_store(store, [
            _make_memory("user.name", "Alice", scope_ref="agent:edi"),
            _make_memory("user.lang", "Python is preferred", scope_ref="agent:edi"),
        ])
        result = await tool.execute(query="name")
        assert "Alice" in result
        assert "user.name" in result

    @pytest.mark.asyncio
    async def test_empty_results(self, tool, store):
        result = await tool.execute(query="nonexistent topic")
        assert "No memories found" in result

    @pytest.mark.asyncio
    async def test_missing_query_and_prefix(self, tool):
        result = await tool.execute()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_result_format(self, tool, store):
        await _seed_store(store, [
            _make_memory("api.endpoint", "POST /v1/ingest", scope_ref="agent:edi"),
        ])
        result = await tool.execute(query="endpoint")
        assert "[api.endpoint]" in result
        assert "POST /v1/ingest" in result
        assert "relevance=" in result
        assert "semi_stable" in result

    @pytest.mark.asyncio
    async def test_no_touch_updates(self, tool, store, retriever):
        """recall_memory is read-only — should not leave pending touches."""
        await _seed_store(store, [
            _make_memory("fact.one", "important fact", scope_ref="agent:edi"),
        ])
        await tool.execute(query="important")
        assert retriever._pending_touch_ids == []


class TestKeyPrefixFiltering:
    @pytest.mark.asyncio
    async def test_prefix_only_search(self, tool, store):
        await _seed_store(store, [
            _make_memory("research.llm", "GPT-4 findings", scope_ref="agent:edi"),
            _make_memory("decision.arch", "Use event sourcing", scope_ref="agent:edi"),
            _make_memory("research.eval", "Eval results", scope_ref="agent:edi"),
        ])
        result = await tool.execute(query="", key_prefix="research")
        assert "GPT-4 findings" in result
        assert "Eval results" in result
        assert "event sourcing" not in result

    @pytest.mark.asyncio
    async def test_prefix_with_query(self, tool, store):
        await _seed_store(store, [
            _make_memory("research.llm", "GPT-4 is good at reasoning", scope_ref="agent:edi"),
            _make_memory("decision.llm", "Use GPT-4 for extraction", scope_ref="agent:edi"),
        ])
        result = await tool.execute(query="GPT-4", key_prefix="research")
        assert "reasoning" in result
        assert "extraction" not in result


class TestScopeFiltering:
    @pytest.mark.asyncio
    async def test_default_scope_private_only(self, tool, store):
        """Agent with default_read=["private"] should only see private memories."""
        await _seed_store(store, [
            _make_memory("fact.a", "private fact", scope="private", scope_ref="agent:edi"),
            _make_memory("fact.b", "project fact", scope="project", scope_ref="project_1"),
        ])
        result = await tool.execute(query="fact")
        assert "private fact" in result
        assert "project fact" not in result

    @pytest.mark.asyncio
    async def test_scope_all_with_permission(self, tool_all_scopes, store):
        """Agent with default_read=["private", "project"] can search all."""
        await _seed_store(store, [
            _make_memory("fact.a", "private fact", scope="private", scope_ref="agent:edi"),
            _make_memory("fact.b", "project fact", scope="project", scope_ref=None),
        ])
        result = await tool_all_scopes.execute(query="fact", scope="all")
        assert "private fact" in result
        assert "project fact" in result


class TestScopePermissionEnforcement:
    @pytest.mark.asyncio
    async def test_private_agent_cannot_access_project(self, tool, store):
        """Agent with default_read=["private"] tries scope="project" — should be denied."""
        await _seed_store(store, [
            _make_memory("fact.a", "private fact", scope="private", scope_ref="agent:edi"),
            _make_memory("fact.b", "project secret", scope="project", scope_ref="project_1"),
        ])
        result = await tool.execute(query="fact", scope="project")
        assert "not available" in result
        assert "project secret" not in result
        assert "private fact" in result

    @pytest.mark.asyncio
    async def test_project_scope_override(self, tool_all_scopes, store):
        """Agent with both scopes can narrow to project only."""
        await _seed_store(store, [
            _make_memory("fact.a", "private fact", scope="private", scope_ref="agent:edi"),
            _make_memory("fact.b", "project fact", scope="project", scope_ref=None),
        ])
        result = await tool_all_scopes.execute(query="fact", scope="project")
        assert "project fact" in result
        assert "private fact" not in result


class TestTokenCap:
    @pytest.mark.asyncio
    async def test_output_truncated(self, tool, store):
        """Output should be capped at ~8000 chars."""
        # Create many large memories
        memories = [
            _make_memory(f"data.item_{i}", "x" * 500, scope_ref="agent:edi")
            for i in range(30)
        ]
        await _seed_store(store, memories)
        result = await tool.execute(query="data item", top_k=20)
        assert len(result) <= 8100  # 8000 + some buffer for truncation message


class TestTopK:
    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, tool, store):
        memories = [
            _make_memory(f"item.{i}", f"value {i}", scope_ref="agent:edi")
            for i in range(10)
        ]
        await _seed_store(store, memories)
        result = await tool.execute(query="value", top_k=3)
        # Count how many [item.X] entries appear
        count = result.count("[item.")
        assert count == 3

    @pytest.mark.asyncio
    async def test_top_k_capped_at_20(self, tool, store):
        """top_k should not exceed 20."""
        memories = [
            _make_memory(f"item.{i}", f"value {i}", scope_ref="agent:edi")
            for i in range(25)
        ]
        await _seed_store(store, memories)
        result = await tool.execute(query="value", top_k=50)
        count = result.count("[item.")
        assert count <= 20


class TestToolDefinition:
    def test_schema_structure(self, tool):
        defn = tool.tool_definition
        assert defn["name"] == "recall_memory"
        assert "parameters" in defn
        props = defn["parameters"]["properties"]
        assert "query" in props
        assert "key_prefix" in props
        assert "scope" in props
        assert "top_k" in props
        assert defn["parameters"]["required"] == ["query"]

    def test_schema_includes_key_prefixes(self, retriever, store, scope_config):
        schema_entries = [
            {"prefix": "research.", "description": "Research findings", "examples": []},
            {"prefix": "decision.", "description": "Decisions", "examples": []},
        ]
        tool = RecallMemoryTool(
            retriever=retriever,
            memory_store=store,
            scope_config=scope_config,
            key_schema=schema_entries,
        )
        desc = tool.tool_definition["description"]
        assert "research" in desc
        assert "decision" in desc


class TestNoScopeConfig:
    @pytest.mark.asyncio
    async def test_works_without_scope_config(self, retriever, store):
        """Tool should work when no scope_config is set (legacy/single-agent)."""
        tool = RecallMemoryTool(retriever=retriever, memory_store=store)
        await _seed_store(store, [
            _make_memory("fact.a", "some fact"),
        ])
        result = await tool.execute(query="fact")
        assert "some fact" in result
