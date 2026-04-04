"""Tests for memory extraction."""

import json

import pytest

from sr2.config.models import MemoryScopeConfig
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
        response = json.dumps(
            [
                {
                    "key": "user.name",
                    "value": "Alice",
                    "memory_type": "identity",
                    "confidence_source": "explicit_statement",
                },
                {
                    "key": "user.employer",
                    "value": "Anthropic",
                    "memory_type": "identity",
                    "confidence_source": "direct_answer",
                },
            ]
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract(
            "I'm Alice and I work at Anthropic", conversation_id="conv_1", turn_number=1
        )

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
        response = json.dumps(
            [
                {"key": "user.name", "value": "Alice"},
                {"key": "no_value"},
                {"value": "no_key"},
                {"random": "data"},
            ]
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        assert len(result.memories) == 1
        assert result.memories[0].key == "user.name"

    @pytest.mark.asyncio
    async def test_invalid_memory_type_defaults(self, store):
        """Invalid memory_type defaults to semi_stable."""
        response = json.dumps([{"key": "k", "value": "v", "memory_type": "invalid_type"}])

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
            {
                "prefix": "user.identity",
                "examples": ["user.identity.name", "user.identity.employer"],
            },
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


class TestMaxMemoriesBoundary:
    """Tests for max_memories_per_turn boundary behavior."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_items, max_memories, expected",
        [
            (5, 5, 5),   # at cap: N items, max=N → N returned
            (5, 10, 5),  # below cap: N items, max=N+5 → N returned
            (10, 3, 3),  # above cap: N items, max=3 → 3 returned
        ],
        ids=["at_cap", "below_cap", "above_cap"],
    )
    async def test_max_memories_boundary(self, store, num_items, max_memories, expected):
        """max_memories_per_turn correctly caps extraction at various boundaries."""
        items = [{"key": f"fact.{i}", "value": f"value {i}"} for i in range(num_items)]

        async def mock_llm(prompt: str) -> str:
            return json.dumps(items)

        extractor = MemoryExtractor(
            llm_callable=mock_llm, store=store, max_memories_per_turn=max_memories,
        )
        result = await extractor.extract("lots of info")

        assert len(result.memories) == expected


class TestNoiseFilters:
    """Tests for extraction noise filters."""

    @pytest.mark.asyncio
    async def test_files_to_modify_key_filtered(self, store):
        """Keys starting with files_to_modify are dropped."""
        response = json.dumps(
            [
                {
                    "key": "files_to_modify.task_failure",
                    "value": "dispatcher/core/dispatcher.py lines 223-245",
                },
                {"key": "user.name", "value": "Alice"},
            ]
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        assert len(result.memories) == 1
        assert result.memories[0].key == "user.name"

    @pytest.mark.asyncio
    async def test_error_key_with_operational_value_filtered(self, store):
        """Error/failure keys referencing operational artifacts are dropped."""
        response = json.dumps(
            [
                {
                    "key": "research.task_failure_root_cause",
                    "value": "Task failure details exist in Galaxy Map metadata",
                },
                {
                    "key": "decision.error_extraction",
                    "value": "Extract error fields from task metadata",
                },
                {"key": "pattern.error_handling", "value": "Use circuit breakers for resilience"},
            ]
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        # Only the legitimate error_handling pattern survives
        assert len(result.memories) == 1
        assert result.memories[0].key == "pattern.error_handling"

    @pytest.mark.asyncio
    async def test_error_key_with_domain_value_kept(self, store):
        """Error keys with genuine domain knowledge values are kept."""
        response = json.dumps(
            [
                {
                    "key": "pattern.error_handling",
                    "value": "Use retry with exponential backoff for HTTP 429",
                },
            ]
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        assert len(result.memories) == 1


class TestPromptVariants:
    """Tests for extraction prompt selection logic."""

    @pytest.mark.asyncio
    async def test_project_scope_prompt(self, store):
        """Project scope uses project extraction prompt with error rules."""
        scope_config = MemoryScopeConfig(allowed_write=["project"], agent_name="liara")

        async def mock_llm(prompt: str) -> str:
            return "[]"

        extractor = MemoryExtractor(
            llm_callable=mock_llm,
            store=store,
            scope_config=scope_config,
        )
        prompt = extractor._build_prompt("test turn")

        assert "technical findings" in prompt
        assert "YOUR OWN execution" in prompt
        assert "Task metadata or dispatcher" in prompt

    @pytest.mark.asyncio
    async def test_private_interactive_prompt(self, store, monkeypatch):
        """Private scope without SR2_TASK_SOURCE uses personal facts prompt."""
        monkeypatch.delenv("SR2_TASK_SOURCE", raising=False)
        scope_config = MemoryScopeConfig(allowed_write=["private"], agent_name="miranda")

        async def mock_llm(prompt: str) -> str:
            return "[]"

        extractor = MemoryExtractor(
            llm_callable=mock_llm,
            store=store,
            scope_config=scope_config,
        )
        prompt = extractor._build_prompt("test turn")

        assert "personal facts" in prompt.lower()
        assert "YOUR OWN execution" in prompt

    @pytest.mark.asyncio
    async def test_task_runner_prompt(self, store, monkeypatch):
        """Private scope + SR2_TASK_SOURCE uses task runner extraction prompt."""
        monkeypatch.setenv("SR2_TASK_SOURCE", "gm_task:GM-99")
        scope_config = MemoryScopeConfig(allowed_write=["private"], agent_name="tali")

        async def mock_llm(prompt: str) -> str:
            return "[]"

        extractor = MemoryExtractor(
            llm_callable=mock_llm,
            store=store,
            scope_config=scope_config,
        )
        prompt = extractor._build_prompt("test turn")

        assert "reusable implementation patterns" in prompt.lower()
        assert "FUTURE tasks" in prompt
        assert "File paths, class names" in prompt
        assert "YOUR OWN execution" in prompt
        # Should NOT contain personal facts language
        assert "personal facts" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_no_scope_config_uses_personal_prompt(self, store, monkeypatch):
        """No scope config defaults to personal facts prompt."""
        monkeypatch.delenv("SR2_TASK_SOURCE", raising=False)

        async def mock_llm(prompt: str) -> str:
            return "[]"

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        prompt = extractor._build_prompt("test turn")

        assert "personal facts" in prompt.lower()

    @pytest.mark.asyncio
    async def test_all_prompts_include_error_rules(self, store, monkeypatch):
        """All three prompt variants include the execution error rules."""

        async def mock_llm(prompt: str) -> str:
            return "[]"

        error_marker = "YOUR OWN execution"

        # Project scope
        ext_project = MemoryExtractor(
            llm_callable=mock_llm,
            store=store,
            scope_config=MemoryScopeConfig(allowed_write=["project"], agent_name="liara"),
        )
        assert error_marker in ext_project._build_prompt("test")

        # Private interactive
        monkeypatch.delenv("SR2_TASK_SOURCE", raising=False)
        ext_private = MemoryExtractor(
            llm_callable=mock_llm,
            store=store,
            scope_config=MemoryScopeConfig(allowed_write=["private"], agent_name="miranda"),
        )
        assert error_marker in ext_private._build_prompt("test")

        # Task runner
        monkeypatch.setenv("SR2_TASK_SOURCE", "gm_task:GM-100")
        ext_task = MemoryExtractor(
            llm_callable=mock_llm,
            store=store,
            scope_config=MemoryScopeConfig(allowed_write=["private"], agent_name="tali"),
        )
        assert error_marker in ext_task._build_prompt("test")


class TestCommentaryBeforeJson:
    """Tests for LLM responses with commentary text before JSON output."""

    @pytest.mark.asyncio
    async def test_thinking_commentary_with_brackets_in_prose(self, store):
        """Commentary containing '[]' before the actual JSON output is handled."""
        # This is the exact pattern that caused 0 memories in production:
        # qwen3.5:9b outputs commentary mentioning "[]" then the real JSON
        response = (
            'The instructions say "[]", I should return an empty array.\n'
            '\n'
            '[]'
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("tool call with no useful content")

        # Should parse successfully (empty array = 0 memories, not an error)
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_commentary_with_brackets_before_real_memories(self, store):
        """Commentary with brackets before actual JSON memories still extracts."""
        response = (
            'Looking at this conversation, I found some facts.\n'
            'The expected format is [{"key": "..."}] so here it is:\n'
            '\n'
            '[{"key": "user.name", "value": "Alice", "memory_type": "identity", '
            '"confidence_source": "explicit_statement"}]'
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("I am Alice")

        assert len(result.memories) == 1
        assert result.memories[0].key == "user.name"
        assert result.memories[0].value == "Alice"

    @pytest.mark.asyncio
    async def test_thinking_tags_with_brackets_inside(self, store):
        """Thinking tags containing [] are stripped before JSON extraction."""
        response = (
            '<think>\n'
            'Nothing to extract. Return [].\n'
            '</think>\n'
            '[]'
        )

        async def mock_llm(prompt: str) -> str:
            return response

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("empty turn")

        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_empty_llm_response(self, store):
        """None/empty LLM response returns empty result without crashing."""

        async def mock_llm(prompt: str) -> str:
            return ""

        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        result = await extractor.extract("test")

        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_find_last_json_array_nested(self, store):
        """_find_last_json_array handles nested arrays correctly."""
        text = 'prefix [{"key": "a", "tags": ["x", "y"]}] suffix'
        result = MemoryExtractor._find_last_json_array(text)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key"] == "a"

    @pytest.mark.asyncio
    async def test_find_last_json_array_multiple_arrays(self, store):
        """_find_last_json_array returns the last array, not the first."""
        text = '[1, 2, 3] some text [{"key": "real", "value": "data"}]'
        result = MemoryExtractor._find_last_json_array(text)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key"] == "real"
