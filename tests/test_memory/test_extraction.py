"""Tests for memory extraction."""

import json

import pytest

from sr2.memory.extraction import MemoryExtractor
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


class TestMemoryExtractor:
    """Tests for MemoryExtractor."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self, store):
        """Valid JSON response extracts correct number of memories."""
        response = json.dumps([
            {"key": "user.name", "value": "Alice", "memory_type": "identity", "confidence_source": "explicit_statement"},
            {"key": "user.employer", "value": "Anthropic", "memory_type": "identity", "confidence_source": "direct_answer"},
        ])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("I'm Alice and I work at Anthropic", conversation_id="conv_1", turn_number=1)

        assert len(result.memories) == 2
        assert result.memories[0].key == "user.name"
        assert result.memories[1].value == "Anthropic"
        # source_conversation removed; ExtractionResult now uses source
        assert result.source is None

    @pytest.mark.asyncio
    async def test_json_with_markdown_fences(self, store):
        """JSON wrapped in markdown fences parses correctly."""
        response = '```json\n[{"key": "user.name", "value": "Alice"}]\n```'

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("I'm Alice")

        assert len(result.memories) == 1
        assert result.memories[0].key == "user.name"

    @pytest.mark.asyncio
    async def test_malformed_json(self, store):
        """Malformed JSON returns empty ExtractionResult."""
        async def mock_llm(prompt: str) -> str:
            return "not valid json at all {"

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("some text")

        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_max_memories_truncation(self, store):
        """Response with more than max_memories_per_turn is truncated."""
        items = [{"key": f"k{i}", "value": f"v{i}"} for i in range(10)]

        async def mock_llm(prompt: str) -> str:
            return json.dumps(items)

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store, max_memories_per_turn=3)
        result = await extractor.extract("lots of info")

        assert len(result.memories) == 3

    @pytest.mark.asyncio
    async def test_items_missing_key_or_value_skipped(self, store):
        """Items missing key or value are skipped."""
        response = json.dumps([
            {"key": "user.name", "value": "Alice"},
            {"key": "no_value"},
            {"value": "no_key"},
            {"random": "data"},
        ])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        assert len(result.memories) == 1
        assert result.memories[0].key == "user.name"

    @pytest.mark.asyncio
    async def test_invalid_memory_type_defaults(self, store):
        """Invalid memory_type defaults to semi_stable."""
        response = json.dumps([
            {"key": "k", "value": "v", "memory_type": "invalid_type"}
        ])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        assert len(result.memories) == 1
        assert result.memories[0].memory_type == "semi_stable"
        assert result.memories[0].stability_score == 0.7

    @pytest.mark.asyncio
    async def test_memories_saved_to_store(self, store):
        """Extracted memories are saved to the store."""
        response = json.dumps([{"key": "user.name", "value": "Alice"}])

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("I'm Alice")

        mem_id = result.memories[0].id
        stored = await store.get(mem_id)
        assert stored is not None
        assert stored.value == "Alice"

    @pytest.mark.asyncio
    async def test_build_prompt_includes_key_schema(self, store):
        """_build_prompt() includes key schema when provided."""
        schema = [
            {"prefix": "user.identity", "examples": ["user.identity.name", "user.identity.employer"]},
        ]

        async def mock_llm(prompt: str) -> str:
            return "[]"

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store, key_schema=schema)
        prompt = extractor._build_prompt("test turn")

        assert "user.identity" in prompt
        assert "user.identity.name" in prompt

    @pytest.mark.asyncio
    async def test_json_wrapped_in_prose(self, store):
        """JSON array wrapped in prose text is still extracted."""
        response = 'Here are the extracted memories:\n[{"key": "user.name", "value": "Alice"}]\nHope that helps!'

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("I'm Alice")

        assert len(result.memories) == 1
        assert result.memories[0].key == "user.name"

    @pytest.mark.asyncio
    async def test_empty_conversation(self, store):
        """Empty conversation turn → LLM returns [] → empty result."""
        async def mock_llm(prompt: str) -> str:
            return "[]"

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("")

        assert len(result.memories) == 0
