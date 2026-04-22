"""Tests for SystemPromptTransform dataclass and proxy_optimize() integration.

Audit fix 07: SR2 receives system prompt transform metadata from the bridge
so it can account for the token cost in budget/trace/prefix tracking.

Tests cover:
- SystemPromptTransform dataclass structure and importability
- proxy_optimize() accepts optional system_prompt_transform parameter
- Positive token delta (added tokens) reduces effective compaction budget
- Negative token delta (replaced with shorter prompt) increases effective budget
- Effective budget is clamped to >= 0 when transform exceeds total budget
- Transform metadata appears in trace events
- ProxyResult exposes system_prompt_transform_tokens field
- Backward compatibility: None (no transform) is the default
"""

from __future__ import annotations

import tempfile
from dataclasses import fields as dataclass_fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.config.models import MemoryConfig, PipelineConfig, RetrievalConfig, SummarizationConfig
from sr2.sr2 import SR2, SR2Config, ProxyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_sr2(preloaded_config: PipelineConfig | None = None) -> SR2:
    """Build a minimal SR2 instance with an ephemeral temp dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        return SR2(
            SR2Config(
                config_dir=tmpdir,
                agent_yaml={},
                preloaded_config=preloaded_config or PipelineConfig(
                    memory=MemoryConfig(extract=False),
                    summarization=SummarizationConfig(enabled=False),
                    retrieval=RetrievalConfig(enabled=False),
                ),
            )
        )


# ---------------------------------------------------------------------------
# Part 1: SystemPromptTransform dataclass
# ---------------------------------------------------------------------------

class TestSystemPromptTransformDataclass:
    """SystemPromptTransform dataclass must exist with the required fields."""

    def test_importable_from_sr2_sr2(self):
        """SystemPromptTransform must be importable from sr2.sr2."""
        from sr2.sr2 import SystemPromptTransform  # noqa: F401

    def test_importable_from_sr2_package(self):
        """SystemPromptTransform must be importable from the sr2 package top-level."""
        import sr2
        assert hasattr(sr2, "SystemPromptTransform"), (
            "SystemPromptTransform is not exported from the sr2 package. "
            "Add it to sr2/__init__.py __all__."
        )

    def test_has_transform_type_field(self):
        """Must have transform_type field."""
        from sr2.sr2 import SystemPromptTransform
        field_names = {f.name for f in dataclass_fields(SystemPromptTransform)}
        assert "transform_type" in field_names

    def test_has_original_tokens_field(self):
        """Must have original_tokens field."""
        from sr2.sr2 import SystemPromptTransform
        field_names = {f.name for f in dataclass_fields(SystemPromptTransform)}
        assert "original_tokens" in field_names

    def test_has_transformed_tokens_field(self):
        """Must have transformed_tokens field."""
        from sr2.sr2 import SystemPromptTransform
        field_names = {f.name for f in dataclass_fields(SystemPromptTransform)}
        assert "transformed_tokens" in field_names

    def test_has_content_hash_field(self):
        """Must have content_hash field."""
        from sr2.sr2 import SystemPromptTransform
        field_names = {f.name for f in dataclass_fields(SystemPromptTransform)}
        assert "content_hash" in field_names

    def test_content_hash_defaults_to_none(self):
        """content_hash must be optional (default None)."""
        from sr2.sr2 import SystemPromptTransform
        t = SystemPromptTransform(
            transform_type="prepend",
            original_tokens=100,
            transformed_tokens=150,
        )
        assert t.content_hash is None

    def test_construction_with_all_fields(self):
        """Must construct successfully with all fields explicitly set."""
        from sr2.sr2 import SystemPromptTransform
        t = SystemPromptTransform(
            transform_type="replace",
            original_tokens=200,
            transformed_tokens=800,
            content_hash="abc123",
        )
        assert t.transform_type == "replace"
        assert t.original_tokens == 200
        assert t.transformed_tokens == 800
        assert t.content_hash == "abc123"

    def test_negative_delta_is_representable(self):
        """A transform that shortens the prompt must allow transformed_tokens < original_tokens.

        The token delta is signed: shorter replacement = negative delta.
        The dataclass itself must not prevent this.
        """
        from sr2.sr2 import SystemPromptTransform
        t = SystemPromptTransform(
            transform_type="replace",
            original_tokens=1000,
            transformed_tokens=200,  # shorter than original
            content_hash=None,
        )
        delta = t.transformed_tokens - t.original_tokens
        assert delta == -800  # negative — prompt shrank


# ---------------------------------------------------------------------------
# Part 2: proxy_optimize() signature
# ---------------------------------------------------------------------------

class TestProxyOptimizeSignature:
    """proxy_optimize() must accept an optional system_prompt_transform parameter."""

    @pytest.mark.asyncio
    async def test_accepts_system_prompt_transform_keyword_arg(self):
        """proxy_optimize() must not raise when system_prompt_transform is passed."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        sr2 = _minimal_sr2()
        transform = SystemPromptTransform(
            transform_type="prepend",
            original_tokens=100,
            transformed_tokens=200,
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        # Must not raise TypeError for unknown kwarg
        result = await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="sig-test",
            system_prompt_transform=transform,
        )
        assert isinstance(result, ProxyResult)

    @pytest.mark.asyncio
    async def test_defaults_to_none_when_not_provided(self):
        """system_prompt_transform defaults to None — existing callers unaffected."""
        from sr2.compaction.engine import ConversationTurn

        sr2 = _minimal_sr2()
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        # Must succeed without the parameter
        result = await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="default-none-test",
        )
        assert isinstance(result, ProxyResult)


# ---------------------------------------------------------------------------
# Part 3: Budget adjustment — positive delta (transform adds tokens)
# ---------------------------------------------------------------------------

class TestBudgetAdjustmentPositiveDelta:
    """When transform adds tokens, effective compaction budget must be reduced."""

    @pytest.mark.asyncio
    async def test_positive_delta_reduces_compaction_budget(self):
        """A transform that adds 500 tokens must reduce the compaction budget by 500.

        The test intercepts run_compaction() to capture the token_budget argument
        and verifies it was reduced by the transform's token delta.
        """
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        token_budget = 8000
        config = PipelineConfig(
            token_budget=token_budget,
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        sr2 = _minimal_sr2(config)

        captured_budgets: list[int] = []
        original_run_compaction = sr2._conversation.run_compaction

        def capturing_run_compaction(session_id, token_budget=None, **kwargs):
            if token_budget is not None:
                captured_budgets.append(token_budget)
            return original_run_compaction(session_id, token_budget=token_budget, **kwargs)

        sr2._conversation.run_compaction = capturing_run_compaction

        transform = SystemPromptTransform(
            transform_type="prepend",
            original_tokens=100,
            transformed_tokens=600,  # added 500 tokens
            content_hash="hash-a",
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="budget-positive-test",
            system_prompt_transform=transform,
        )

        assert captured_budgets, "run_compaction was never called — cannot verify budget"

        # The effective budget passed to compaction should be reduced by the delta
        transform_delta = transform.transformed_tokens - transform.original_tokens  # +500
        expected_budget = token_budget - transform_delta  # 7500
        assert captured_budgets[0] == expected_budget, (
            f"Expected compaction budget {expected_budget} (token_budget={token_budget} "
            f"minus transform_delta={transform_delta}), got {captured_budgets[0]}."
        )

    @pytest.mark.asyncio
    async def test_no_transform_uses_full_budget(self):
        """With no transform, compaction budget must equal the full token_budget."""
        from sr2.compaction.engine import ConversationTurn

        token_budget = 8000
        config = PipelineConfig(
            token_budget=token_budget,
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        sr2 = _minimal_sr2(config)

        captured_budgets: list[int] = []
        original_run_compaction = sr2._conversation.run_compaction

        def capturing_run_compaction(session_id, token_budget=None, **kwargs):
            if token_budget is not None:
                captured_budgets.append(token_budget)
            return original_run_compaction(session_id, token_budget=token_budget, **kwargs)

        sr2._conversation.run_compaction = capturing_run_compaction

        turn = ConversationTurn(turn_number=1, role="user", content="hello")
        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="no-transform-budget-test",
        )

        assert captured_budgets, "run_compaction was never called"
        assert captured_budgets[0] == token_budget, (
            f"Expected full budget {token_budget} when no transform, got {captured_budgets[0]}."
        )


# ---------------------------------------------------------------------------
# Part 4: Budget adjustment — negative delta (transform shortens prompt)
# ---------------------------------------------------------------------------

class TestBudgetAdjustmentNegativeDelta:
    """When transform shortens the prompt, effective budget must INCREASE (signed delta)."""

    @pytest.mark.asyncio
    async def test_negative_delta_increases_compaction_budget(self):
        """A replace transform that shortens by 300 tokens must increase budget by 300.

        Signed delta: transformed_tokens - original_tokens = -300
        effective_budget = token_budget - (-300) = token_budget + 300
        """
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        token_budget = 8000
        config = PipelineConfig(
            token_budget=token_budget,
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        sr2 = _minimal_sr2(config)

        captured_budgets: list[int] = []
        original_run_compaction = sr2._conversation.run_compaction

        def capturing_run_compaction(session_id, token_budget=None, **kwargs):
            if token_budget is not None:
                captured_budgets.append(token_budget)
            return original_run_compaction(session_id, token_budget=token_budget, **kwargs)

        sr2._conversation.run_compaction = capturing_run_compaction

        transform = SystemPromptTransform(
            transform_type="replace",
            original_tokens=1000,
            transformed_tokens=700,  # shortened by 300 tokens
            content_hash="hash-b",
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="budget-negative-test",
            system_prompt_transform=transform,
        )

        assert captured_budgets, "run_compaction was never called"

        # delta = 700 - 1000 = -300 (signed)
        transform_delta = transform.transformed_tokens - transform.original_tokens  # -300
        expected_budget = token_budget - transform_delta  # 8000 - (-300) = 8300
        assert captured_budgets[0] == expected_budget, (
            f"Expected budget {expected_budget} for negative delta transform "
            f"(token_budget={token_budget}, delta={transform_delta}), "
            f"got {captured_budgets[0]}. "
            "The implementation must use a SIGNED delta — a replace that shortens "
            "the prompt should FREE up budget, not reduce it."
        )


# ---------------------------------------------------------------------------
# Part 5: Budget clamping — transform must not produce negative budget
# ---------------------------------------------------------------------------

class TestBudgetClamping:
    """When transform delta exceeds total budget, effective budget must clamp to >= 0."""

    @pytest.mark.asyncio
    async def test_budget_clamped_to_zero_when_transform_exceeds_budget(self):
        """A transform that adds more tokens than the total budget must clamp to 0.

        Edge case: transform adds 15,000 tokens but total budget is only 8,000.
        effective_budget without clamping = 8000 - 15000 = -7000 (invalid).
        effective_budget with clamping    = max(0, 8000 - 15000) = 0.
        """
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        token_budget = 8000
        config = PipelineConfig(
            token_budget=token_budget,
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        sr2 = _minimal_sr2(config)

        captured_budgets: list[int] = []
        original_run_compaction = sr2._conversation.run_compaction

        def capturing_run_compaction(session_id, token_budget=None, **kwargs):
            if token_budget is not None:
                captured_budgets.append(token_budget)
            return original_run_compaction(session_id, token_budget=token_budget, **kwargs)

        sr2._conversation.run_compaction = capturing_run_compaction

        # Transform adds way more than the total budget
        transform = SystemPromptTransform(
            transform_type="replace",
            original_tokens=0,
            transformed_tokens=15000,  # adds 15000 tokens — more than 8000 budget
            content_hash="hash-huge",
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="budget-clamp-test",
            system_prompt_transform=transform,
        )

        assert captured_budgets, "run_compaction was never called"
        assert captured_budgets[0] >= 0, (
            f"Effective budget must never be negative. Got {captured_budgets[0]}. "
            "Clamp with max(0, effective_budget) before passing to run_compaction."
        )

    @pytest.mark.asyncio
    async def test_budget_exactly_zero_when_transform_matches_budget(self):
        """Edge: transform delta equal to budget produces budget of exactly 0."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        token_budget = 8000
        config = PipelineConfig(
            token_budget=token_budget,
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        sr2 = _minimal_sr2(config)

        captured_budgets: list[int] = []
        original_run_compaction = sr2._conversation.run_compaction

        def capturing_run_compaction(session_id, token_budget=None, **kwargs):
            if token_budget is not None:
                captured_budgets.append(token_budget)
            return original_run_compaction(session_id, token_budget=token_budget, **kwargs)

        sr2._conversation.run_compaction = capturing_run_compaction

        transform = SystemPromptTransform(
            transform_type="prepend",
            original_tokens=0,
            transformed_tokens=token_budget,  # adds exactly budget tokens
            content_hash="hash-exact",
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="budget-zero-test",
            system_prompt_transform=transform,
        )

        assert captured_budgets, "run_compaction was never called"
        assert captured_budgets[0] == 0, (
            f"Expected budget 0 when transform exactly matches budget, "
            f"got {captured_budgets[0]}."
        )


# ---------------------------------------------------------------------------
# Part 6: Trace events
# ---------------------------------------------------------------------------

class TestTransformTraceEvents:
    """Transform metadata must appear in trace events when a TraceCollector is wired."""

    @pytest.mark.asyncio
    async def test_transform_emits_trace_event(self):
        """proxy_optimize with a non-none transform must emit a 'system_prompt_transform' event."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.pipeline.trace import TraceCollector
        from sr2.sr2 import SystemPromptTransform

        collector = TraceCollector()
        config = PipelineConfig(
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = SR2(SR2Config(
                config_dir=tmpdir,
                agent_yaml={},
                preloaded_config=config,
                trace_collector=collector,
            ))

        transform = SystemPromptTransform(
            transform_type="append",
            original_tokens=100,
            transformed_tokens=300,
            content_hash="trace-hash",
        )
        turn = ConversationTurn(turn_number=1, role="user", content="test")
        session_id = "trace-test"

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id=session_id,
            system_prompt_transform=transform,
        )

        # Collect all emitted event stages from both the ring buffer (ended turns)
        # and the active turn dict (turns not yet ended via end_turn()).
        def _all_stages(collector: TraceCollector) -> list[str]:
            stages = []
            for trace in collector._traces:  # deque[TurnTrace]
                for event in trace.events:
                    stages.append(event.stage)
            for trace in collector._active.values():  # dict[str, TurnTrace]
                for event in trace.events:
                    stages.append(event.stage)
            return stages

        emitted = _all_stages(collector)
        assert "system_prompt_transform" in emitted, (
            f"Expected a 'system_prompt_transform' trace event when transform is provided. "
            f"Emitted stages: {emitted}. "
            "Add trace emission in _proxy_optimize_inner() when transform is active."
        )

    @pytest.mark.asyncio
    async def test_no_trace_event_for_none_transform(self):
        """When transform is None, no 'system_prompt_transform' trace event must be emitted."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.pipeline.trace import TraceCollector

        collector = TraceCollector()
        config = PipelineConfig(
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = SR2(SR2Config(
                config_dir=tmpdir,
                agent_yaml={},
                preloaded_config=config,
                trace_collector=collector,
            ))

        turn = ConversationTurn(turn_number=1, role="user", content="test")
        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="no-trace-test",
            # no system_prompt_transform
        )

        def _all_stages(collector: TraceCollector) -> list[str]:
            stages = []
            for trace in collector._traces:
                for event in trace.events:
                    stages.append(event.stage)
            for trace in collector._active.values():
                for event in trace.events:
                    stages.append(event.stage)
            return stages

        emitted = _all_stages(collector)
        assert "system_prompt_transform" not in emitted, (
            "Unexpected 'system_prompt_transform' trace event when no transform was passed."
        )

    @pytest.mark.asyncio
    async def test_no_trace_event_for_none_transform_type(self):
        """A transform with transform_type='none' must NOT emit a trace event."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.pipeline.trace import TraceCollector
        from sr2.sr2 import SystemPromptTransform

        collector = TraceCollector()
        config = PipelineConfig(
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = SR2(SR2Config(
                config_dir=tmpdir,
                agent_yaml={},
                preloaded_config=config,
                trace_collector=collector,
            ))

        transform = SystemPromptTransform(
            transform_type="none",  # explicit no-op
            original_tokens=100,
            transformed_tokens=100,
        )
        turn = ConversationTurn(turn_number=1, role="user", content="test")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="none-type-trace-test",
            system_prompt_transform=transform,
        )

        def _all_stages(collector: TraceCollector) -> list[str]:
            stages = []
            for trace in collector._traces:
                for event in trace.events:
                    stages.append(event.stage)
            for trace in collector._active.values():
                for event in trace.events:
                    stages.append(event.stage)
            return stages

        emitted = _all_stages(collector)
        assert "system_prompt_transform" not in emitted, (
            "transform_type='none' must not emit a trace event — it is a no-op."
        )


# ---------------------------------------------------------------------------
# Part 7: ProxyResult.system_prompt_transform_tokens
# ---------------------------------------------------------------------------

class TestProxyResultTransformTokens:
    """ProxyResult must expose system_prompt_transform_tokens."""

    def test_proxy_result_has_transform_tokens_field(self):
        """ProxyResult dataclass must have system_prompt_transform_tokens field."""
        field_names = {f.name for f in dataclass_fields(ProxyResult)}
        assert "system_prompt_transform_tokens" in field_names, (
            "ProxyResult is missing 'system_prompt_transform_tokens' field. "
            "Add it with a default of 0."
        )

    def test_proxy_result_transform_tokens_defaults_to_zero(self):
        """system_prompt_transform_tokens must default to 0 for backward compat."""
        result = ProxyResult(
            system_injection=None,
            zones=None,
            compaction_result=None,
            summarization_result=None,
        )
        assert result.system_prompt_transform_tokens == 0, (
            f"Expected default 0, got {result.system_prompt_transform_tokens}."
        )

    @pytest.mark.asyncio
    async def test_proxy_result_includes_transform_tokens_when_set(self):
        """When transform is provided, ProxyResult.system_prompt_transform_tokens must be populated."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        sr2 = _minimal_sr2()
        transform = SystemPromptTransform(
            transform_type="prepend",
            original_tokens=100,
            transformed_tokens=600,  # delta = 500
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        result = await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="result-tokens-test",
            system_prompt_transform=transform,
        )

        # transform_tokens on the result should reflect the delta
        expected = transform.transformed_tokens - transform.original_tokens  # 500
        assert result.system_prompt_transform_tokens == expected, (
            f"ProxyResult.system_prompt_transform_tokens expected {expected} "
            f"(the signed delta), got {result.system_prompt_transform_tokens}."
        )

    @pytest.mark.asyncio
    async def test_proxy_result_transform_tokens_zero_when_no_transform(self):
        """When no transform is passed, ProxyResult.system_prompt_transform_tokens must be 0."""
        from sr2.compaction.engine import ConversationTurn

        sr2 = _minimal_sr2()
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        result = await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="result-no-transform-test",
        )

        assert result.system_prompt_transform_tokens == 0, (
            f"Expected 0 when no transform, got {result.system_prompt_transform_tokens}."
        )


# ---------------------------------------------------------------------------
# Part 8: Prefix stability / content_hash tracking
# ---------------------------------------------------------------------------

class TestPrefixStabilityTracking:
    """SR2 must track per-session transform content hashes for prefix stability detection."""

    @pytest.mark.asyncio
    async def test_transform_hash_stored_per_session(self):
        """After proxy_optimize with a transform, the content_hash must be stored internally."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        sr2 = _minimal_sr2()
        transform = SystemPromptTransform(
            transform_type="prepend",
            original_tokens=100,
            transformed_tokens=200,
            content_hash="unique-hash-xyz",
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="hash-tracking-test",
            system_prompt_transform=transform,
        )

        # The internal dict must exist and store the hash
        assert hasattr(sr2, "_session_transform_hashes"), (
            "SR2 must have a '_session_transform_hashes' dict "
            "to track per-session transform content hashes."
        )
        assert sr2._session_transform_hashes.get("hash-tracking-test") == "unique-hash-xyz", (
            f"Expected content_hash 'unique-hash-xyz' stored for session, "
            f"got {sr2._session_transform_hashes.get('hash-tracking-test')!r}."
        )

    @pytest.mark.asyncio
    async def test_changed_hash_logs_warning(self, caplog):
        """When content_hash changes across calls, a warning must be logged."""
        import logging
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        sr2 = _minimal_sr2()
        session_id = "hash-change-warning-test"

        turn1 = ConversationTurn(turn_number=1, role="user", content="first")
        await sr2.proxy_optimize(
            new_turns=[turn1],
            session_id=session_id,
            system_prompt_transform=SystemPromptTransform(
                transform_type="prepend",
                original_tokens=100,
                transformed_tokens=200,
                content_hash="hash-v1",
            ),
        )

        turn2 = ConversationTurn(turn_number=2, role="user", content="second")
        with caplog.at_level(logging.WARNING, logger="sr2.sr2"):
            await sr2.proxy_optimize(
                new_turns=[turn2],
                session_id=session_id,
                system_prompt_transform=SystemPromptTransform(
                    transform_type="prepend",
                    original_tokens=100,
                    transformed_tokens=200,
                    content_hash="hash-v2",  # changed
                ),
            )

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("transform" in m.lower() or "prefix" in m.lower() for m in warning_messages), (
            "Expected a warning when the transform content_hash changes between calls "
            f"(prefix may be invalidated). Warnings logged: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_null_hash_does_not_trigger_tracking(self):
        """When content_hash is None, no entry must be stored in _session_transform_hashes."""
        from sr2.compaction.engine import ConversationTurn
        from sr2.sr2 import SystemPromptTransform

        sr2 = _minimal_sr2()
        transform = SystemPromptTransform(
            transform_type="append",
            original_tokens=100,
            transformed_tokens=150,
            content_hash=None,
        )
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="null-hash-test",
            system_prompt_transform=transform,
        )

        # content_hash is None — no entry must be stored
        if hasattr(sr2, "_session_transform_hashes"):
            stored = sr2._session_transform_hashes.get("null-hash-test")
            assert stored is None, (
                f"No hash should be stored when content_hash=None, but got {stored!r}."
            )


# ---------------------------------------------------------------------------
# Part 9: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing callers that omit system_prompt_transform must behave identically to before."""

    @pytest.mark.asyncio
    async def test_proxy_result_shape_unchanged_without_transform(self):
        """ProxyResult from a call without transform must have all original fields."""
        from sr2.compaction.engine import ConversationTurn

        sr2 = _minimal_sr2()
        turn = ConversationTurn(turn_number=1, role="user", content="hello")

        result = await sr2.proxy_optimize(
            new_turns=[turn],
            session_id="compat-test",
        )

        # Original fields
        assert hasattr(result, "system_injection")
        assert hasattr(result, "zones")
        assert hasattr(result, "compaction_result")
        assert hasattr(result, "summarization_result")
        assert hasattr(result, "current_context")
        # New field defaults to 0
        assert result.system_prompt_transform_tokens == 0
