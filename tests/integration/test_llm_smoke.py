"""Smoke tests against a real LLM.

These tests verify that:
- LiteLLM integration works
- Memory extraction produces valid output
- Summarization produces valid output

Requires: TEST_LLM_API_KEY environment variable.
Uses the cheapest/fastest model available.
"""

import os

import pytest

from tests.integration.conftest import requires_llm

LLM_MODEL = os.environ.get("TEST_LLM_MODEL", "claude-haiku-4-5-20251001")


async def call_llm(prompt: str) -> str:
    """Simple LiteLLM wrapper for testing."""
    import litellm

    response = await litellm.acompletion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


async def call_llm_with_system(system: str, prompt: str) -> str:
    """LiteLLM wrapper with system prompt."""
    import litellm

    response = await litellm.acompletion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


@requires_llm
class TestLLMSmoke:

    @pytest.mark.asyncio
    async def test_memory_extraction(self):
        """Memory extraction with real LLM produces valid memories."""
        from sr2.memory.extraction import MemoryExtractor
        from sr2.memory.store import InMemoryMemoryStore

        store = InMemoryMemoryStore()
        extractor = MemoryExtractor(
            llm_callable=call_llm,
            store=store,
            key_schema=[
                {
                    "prefix": "user.identity",
                    "examples": ["user.identity.name", "user.identity.employer"],
                },
            ],
            max_memories_per_turn=3,
        )

        result = await extractor.extract(
            "Hi, I'm Alice and I just started working at Anthropic as a software engineer.",
            conversation_id="test_conv",
            turn_number=1,
        )

        assert len(result.memories) > 0
        keys = [m.key for m in result.memories]
        # Should extract at least name or employer
        assert any("name" in k or "employer" in k for k in keys)

    @pytest.mark.asyncio
    async def test_summarization(self):
        """Summarization with real LLM produces structured output."""
        from sr2.config.models import SummarizationConfig
        from sr2.summarization.engine import SummarizationEngine

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=call_llm_with_system)

        turns = """
user: I want to build a web app using Python and FastAPI.
assistant: Great choice! FastAPI is excellent for building APIs. What's the main purpose?
user: It's a task management tool for our team of 5. We need real-time updates.
assistant: For real-time updates with FastAPI, I'd recommend WebSockets. Should we use PostgreSQL for storage?
user: Yes, PostgreSQL works. We also need user authentication.
"""
        result = await engine.summarize(turns, "1-5", original_tokens=200)

        assert hasattr(result.summary, "key_decisions") or isinstance(
            result.summary, str
        )
        assert result.summary_tokens > 0
        assert result.summary_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_litellm_basic_call(self):
        """Basic LiteLLM call works."""
        response = await call_llm("Say 'hello' and nothing else.")
        assert "hello" in response.lower()
