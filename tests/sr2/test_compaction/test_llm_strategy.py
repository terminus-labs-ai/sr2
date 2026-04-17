"""Tests for LLM compaction strategy."""

import json

import pytest

from sr2.compaction.engine import CompactionResult, ConversationTurn
from sr2.compaction.llm_strategy import (
    CompactionAnalysis,
    LLMCompactionResult,
    LLMCompactionStrategy,
)
from sr2.config.models import CompactionConfig, CostGateConfig


def _make_turn(num: int, content: str = "turn content here") -> ConversationTurn:
    return ConversationTurn(turn_number=num, role="assistant", content=content)


def _make_llm_response(
    decisions: list[str] | None = None,
    current_state: str = "working on task",
    open_questions: list[str] | None = None,
    key_context: list[str] | None = None,
    summary: str = "The assistant worked on the task.",
) -> str:
    return json.dumps({
        "analysis": {
            "decisions": decisions or ["Use Python for the backend"],
            "current_state": current_state,
            "open_questions": open_questions or [],
            "key_context": key_context or ["Project uses FastAPI"],
        },
        "summary": summary,
    })


class TestLLMCompactionStrategy:
    @pytest.mark.asyncio
    async def test_produces_analysis_and_summary(self):
        async def mock_llm(system: str, prompt: str) -> str:
            return _make_llm_response()

        strategy = LLMCompactionStrategy(llm_callable=mock_llm)
        turns = [_make_turn(i) for i in range(5)]

        result = await strategy.compact(turns)

        assert isinstance(result, LLMCompactionResult)
        assert isinstance(result.analysis, CompactionAnalysis)
        assert result.summary == "The assistant worked on the task."
        assert result.analysis.decisions == ["Use Python for the backend"]
        assert result.analysis.current_state == "working on task"
        assert result.analysis.key_context == ["Project uses FastAPI"]

    @pytest.mark.asyncio
    async def test_analysis_fields_structured(self):
        async def mock_llm(system: str, prompt: str) -> str:
            return _make_llm_response(
                decisions=["Use JWT", "Deploy to K8s"],
                open_questions=["Which DB?"],
                key_context=["Deadline is Friday"],
            )

        strategy = LLMCompactionStrategy(llm_callable=mock_llm)
        result = await strategy.compact([_make_turn(0)])

        assert len(result.analysis.decisions) == 2
        assert result.analysis.open_questions == ["Which DB?"]

    @pytest.mark.asyncio
    async def test_token_counts(self):
        async def mock_llm(system: str, prompt: str) -> str:
            return _make_llm_response(summary="short")

        strategy = LLMCompactionStrategy(llm_callable=mock_llm)
        turns = [_make_turn(i, content="x" * 400) for i in range(3)]

        result = await strategy.compact(turns)

        assert result.original_tokens == 300  # 3 * (400 // 4)
        # LLM returned summary="short" -> compacted_tokens = len("short") // 4 = 1
        expected_compacted = len("short") // 4
        assert result.compacted_tokens == expected_compacted, (
            f"Expected compacted_tokens={expected_compacted}, got {result.compacted_tokens}"
        )

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        async def failing_llm(system: str, prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        strategy = LLMCompactionStrategy(llm_callable=failing_llm)
        turns = [_make_turn(i) for i in range(3)]

        result = await strategy.compact(turns)

        assert "LLM compaction failed" in result.summary
        assert result.analysis.decisions == []
        assert result.analysis.current_state == ""

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self):
        async def bad_json_llm(system: str, prompt: str) -> str:
            return "not json at all"

        strategy = LLMCompactionStrategy(llm_callable=bad_json_llm)
        result = await strategy.compact([_make_turn(0)])

        assert "LLM compaction failed" in result.summary

    @pytest.mark.asyncio
    async def test_empty_analysis_fields(self):
        async def mock_llm(system: str, prompt: str) -> str:
            return json.dumps({"analysis": {}, "summary": "Brief."})

        strategy = LLMCompactionStrategy(llm_callable=mock_llm)
        result = await strategy.compact([_make_turn(0)])

        assert result.analysis.decisions == []
        assert result.analysis.current_state == ""
        assert result.analysis.open_questions == []
        assert result.analysis.key_context == []
        assert result.summary == "Brief."


class TestCompactionResultAnalysis:
    def test_rule_based_result_has_no_analysis(self):
        result = CompactionResult(
            turns=[_make_turn(0)],
            original_tokens=100,
            compacted_tokens=50,
            turns_compacted=1,
        )
        assert result.analysis is None

    def test_result_with_analysis(self):
        analysis = {
            "decisions": ["Use JWT"],
            "current_state": "auth implementation",
            "open_questions": [],
            "key_context": [],
        }
        result = CompactionResult(
            turns=[_make_turn(0)],
            original_tokens=100,
            compacted_tokens=50,
            turns_compacted=1,
            analysis=analysis,
        )
        assert result.analysis is not None
        assert result.analysis["decisions"] == ["Use JWT"]


class TestCompactionConfigStrategy:
    def test_default_strategy_is_rule_based(self):
        config = CompactionConfig(cost_gate=CostGateConfig(enabled=False))
        assert config.strategy == "rule_based"

    def test_llm_strategy_config(self):
        config = CompactionConfig(strategy="llm", llm_compaction_model="gpt-4o-mini", cost_gate=CostGateConfig(enabled=False))
        assert config.strategy == "llm"
        assert config.llm_compaction_model == "gpt-4o-mini"

    def test_hybrid_strategy_config(self):
        config = CompactionConfig(strategy="hybrid", cost_gate=CostGateConfig(enabled=False))
        assert config.strategy == "hybrid"

    def test_invalid_strategy_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompactionConfig(strategy="invalid")

    def test_llm_max_tokens_default(self):
        config = CompactionConfig(cost_gate=CostGateConfig(enabled=False))
        assert config.llm_compaction_max_tokens == 1000
