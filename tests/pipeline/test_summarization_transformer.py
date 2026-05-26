"""Tests for SummarizationTransformer.

Acceptance Criteria covered:
  AC3:  keep_last_n=3 with 10 turns → [TextBlock(summary), turn8, turn9, turn10]
  AC4:  fewer turns than keep_last_n → no-op (content=None, events=[], entries=[], no LLM call)
  AC5:  emits summarization_complete event with event.data = [TextBlock(summary_text)]
  AC6:  provenance entries have sources = non-empty tuple
  AC9:  model key in config is read by the transformer (routing tested at orchestrator layer)
  AC10: keep_within_tokens keeps only most-recent turns fitting within token budget
  AC11: two summarizations in same session → two distinct provenance entries

FR3:  SummarizationTransformer implements the Transformer protocol.
FR4:  keep_last_n and keep_within_tokens strategies.
FR5:  LLM is injected at construction (mock in tests).

Non-functional:
  - No-op safety: never calls LLM when content <= keep threshold.
  - LLM failure propagates as-is.
  - Transformer origin entries always have non-empty sources.

Note on AC9 (model routing): CompletionRequest has no `model` field — the LLM client
itself handles model selection. The transformer reads `config["model"]` to select which
LLM callable to use (injected by orchestrator). End-to-end model routing is tested at
the orchestrator integration level, not here.
"""

from __future__ import annotations

import pytest

from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.protocols import Transformer
from sr2.protocols.llm import CompletionRequest, CompletionResponse


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    def __init__(self, response_text: str = "This is a summary."):
        self.response_text = response_text
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text=self.response_text)],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest):
        # not used by transformer
        return
        yield  # make it a generator


class FailingLLM:
    """LLM that always raises on complete()."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise RuntimeError("LLM backend unavailable")

    async def stream(self, request: CompletionRequest):
        return
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text_blocks(n: int, prefix: str = "turn") -> list[TextBlock]:
    """Return n TextBlocks with distinguishable text."""
    return [TextBlock(text=f"{prefix}_{i}") for i in range(n)]


def make_config(
    *,
    keep_strategy: str = "keep_last_n",
    keep_last_n: int = 3,
    keep_tokens: int | None = None,
    subscriptions: list[dict] | None = None,
    model: str = "claude-3-haiku",
    max_executions: int = 10,
) -> object:
    """Build a TransformerConfig for SummarizationTransformer."""
    from sr2.config.models import EventSubscriptionConfig, TransformerConfig

    sub_configs = subscriptions or [{"event": "turn_start"}]
    subs = [EventSubscriptionConfig(**s) for s in sub_configs]
    inner = {
        "keep_strategy": keep_strategy,
        "keep_last_n": keep_last_n,
        "model": model,
    }
    if keep_tokens is not None:
        inner["keep_tokens"] = keep_tokens
    return TransformerConfig(
        type="summarization",
        subscriptions=subs,
        config=inner,
        max_executions=max_executions,
    )


def make_events() -> list[Event]:
    return [Event(name="turn_start", phase=EventPhase.COMPLETED, source_layer="engine")]


def build_transformer(config=None, llm=None):
    """Convenience factory that imports the real implementation."""
    from sr2.pipeline.transformers.summarization import SummarizationTransformer

    cfg = config or make_config()
    lm = llm or MockLLM()
    return SummarizationTransformer(config=cfg, llm=lm)


# ---------------------------------------------------------------------------
# FR3: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """FR3: SummarizationTransformer satisfies the Transformer protocol."""

    def test_fr3_implements_transformer_protocol(self):
        """FR3: isinstance check against the runtime-checkable Transformer protocol."""
        transformer = build_transformer()
        assert isinstance(transformer, Transformer)

    def test_fr3_has_subscriptions_attribute(self):
        """FR3: .subscriptions is a list of EventSubscription."""
        transformer = build_transformer()
        assert isinstance(transformer.subscriptions, list)
        assert all(isinstance(s, EventSubscription) for s in transformer.subscriptions)

    def test_fr3_has_max_executions_attribute(self):
        """FR3: .max_executions is an int."""
        transformer = build_transformer()
        assert isinstance(transformer.max_executions, int)

    def test_fr3_subscriptions_configurable_via_yaml(self):
        """FR3: subscriptions come from TransformerConfig.subscriptions."""
        config = make_config(subscriptions=[{"event": "memory_updated"}, {"event": "turn_start"}])
        transformer = build_transformer(config=config)
        names = [s.event_name for s in transformer.subscriptions]
        assert "memory_updated" in names
        assert "turn_start" in names

    def test_fr3_transform_is_coroutine(self):
        """FR3: transform() is a coroutine function (async def)."""
        import asyncio

        transformer = build_transformer()
        coro = transformer.transform([], [])
        assert asyncio.iscoroutine(coro)
        coro.close()  # clean up without running


# ---------------------------------------------------------------------------
# AC4: No-op when content <= keep threshold
# ---------------------------------------------------------------------------


class TestNoOp:
    """AC4: When content count <= keep_last_n, return no-op result without calling LLM."""

    @pytest.mark.asyncio
    async def test_ac4_empty_content_is_noop(self):
        """AC4: Empty content list → no-op result, no LLM call."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)

        result = await transformer.transform([], make_events())

        assert isinstance(result, TransformationResult)
        assert result.content is None
        assert result.entries == []
        assert result.events == [] or result.events is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac4_exactly_keep_last_n_is_noop(self):
        """AC4: Exactly keep_last_n turns → no-op, no LLM call."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(3)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert result.entries == []
        assert result.events == [] or result.events is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac4_fewer_than_keep_last_n_is_noop(self):
        """AC4: Fewer turns than keep_last_n → no-op, no LLM call."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(2)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert result.entries == []
        assert result.events == [] or result.events is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac4_noop_result_has_correct_transformer_name(self):
        """AC4: No-op result still carries transformer_name."""
        transformer = build_transformer()
        result = await transformer.transform([], make_events())

        assert isinstance(result.transformer_name, str)
        assert result.transformer_name != ""

    @pytest.mark.asyncio
    async def test_ac4_keep_last_n_one_turn_is_noop(self):
        """AC4: keep_last_n=1 with exactly 1 turn is no-op."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=1), llm=llm)
        content = make_text_blocks(1)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert llm.calls == []


# ---------------------------------------------------------------------------
# AC3: keep_last_n summarization
# ---------------------------------------------------------------------------


class TestKeepLastN:
    """AC3: keep_last_n=3 with 10 turns → [TextBlock(summary), turn7, turn8, turn9]."""

    @pytest.mark.asyncio
    async def test_ac3_basic_keep_last_n(self):
        """AC3: 10 turns, keep_last_n=3 → summary block + last 3 turns."""
        llm = MockLLM(response_text="Summary of turns 0-6.")
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 4  # 1 summary + 3 kept
        summary_block = result.content[0]
        assert isinstance(summary_block, TextBlock)
        assert summary_block.text == "Summary of turns 0-6."
        # Last 3 turns preserved
        assert result.content[1] is content[7]
        assert result.content[2] is content[8]
        assert result.content[3] is content[9]

    @pytest.mark.asyncio
    async def test_ac3_llm_was_called_once(self):
        """AC3: LLM called exactly once with the turns to summarize."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        await transformer.transform(content, make_events())

        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_ac3_llm_called_with_7_turns_to_summarize(self):
        """AC3: LLM prompt contains content from the 7 turns being summarized."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        await transformer.transform(content, make_events())

        request = llm.calls[0]
        # Extract all text from messages and system prompt to check turns are included
        all_text_parts: list[str] = []
        if request.system:
            all_text_parts.extend(b.text for b in request.system)
        for msg in request.messages:
            all_text_parts.extend(
                block.text for block in msg.content if isinstance(block, TextBlock)
            )
        all_text = " ".join(all_text_parts)
        for i in range(7):
            assert f"turn_{i}" in all_text

    @pytest.mark.asyncio
    async def test_ac3_one_more_than_keep_last_n_triggers_summarization(self):
        """AC3 boundary: keep_last_n+1 turns → summarize 1, keep last keep_last_n."""
        llm = MockLLM(response_text="Brief summary.")
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(4)  # 1 to summarize, 3 to keep

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 4  # 1 summary + 3 kept
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Brief summary."
        assert result.content[1] is content[1]
        assert result.content[2] is content[2]
        assert result.content[3] is content[3]

    @pytest.mark.asyncio
    async def test_ac3_keep_last_n_1_summarizes_all_but_last(self):
        """AC3: keep_last_n=1 with 5 turns → summarize 4, keep last 1."""
        llm = MockLLM(response_text="All-but-last summary.")
        transformer = build_transformer(config=make_config(keep_last_n=1), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 2  # 1 summary + 1 kept
        assert result.content[0].text == "All-but-last summary."
        assert result.content[1] is content[4]


# ---------------------------------------------------------------------------
# AC5: summarization_complete event
# ---------------------------------------------------------------------------


class TestSummarizationCompleteEvent:
    """AC5: Transform emits summarization_complete event with data=[TextBlock(summary)]."""

    @pytest.mark.asyncio
    async def test_ac5_emits_summarization_complete(self):
        """AC5: result.events contains a summarization_complete event."""
        llm = MockLLM(response_text="The summary text.")
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.events is not None
        assert len(result.events) >= 1
        event = next((e for e in result.events if e.name == "summarization_complete"), None)
        assert event is not None, "Expected a 'summarization_complete' event"

    @pytest.mark.asyncio
    async def test_ac5_event_phase_is_completed(self):
        """AC5: summarization_complete event has phase=COMPLETED."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        event = next(e for e in result.events if e.name == "summarization_complete")
        assert event.phase == EventPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_ac5_event_data_is_list_with_text_block(self):
        """AC5: event.data is a list containing one TextBlock with the summary text."""
        summary_text = "Precise summary output."
        llm = MockLLM(response_text=summary_text)
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        event = next(e for e in result.events if e.name == "summarization_complete")
        assert isinstance(event.data, list)
        assert len(event.data) == 1
        assert isinstance(event.data[0], TextBlock)
        assert event.data[0].text == summary_text

    @pytest.mark.asyncio
    async def test_ac5_event_source_layer_is_nonempty(self):
        """AC5: event.source_layer is a non-empty string."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        event = next(e for e in result.events if e.name == "summarization_complete")
        assert isinstance(event.source_layer, str)
        assert len(event.source_layer) > 0

    @pytest.mark.asyncio
    async def test_ac5_no_op_emits_no_summarization_complete_event(self):
        """AC4/AC5: No-op path emits no summarization_complete event."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(3)  # fewer than keep_last_n

        result = await transformer.transform(content, make_events())

        events = result.events or []
        summarization_events = [e for e in events if e.name == "summarization_complete"]
        assert summarization_events == []


# ---------------------------------------------------------------------------
# AC6: Provenance entries — non-empty sources
# ---------------------------------------------------------------------------


class TestProvenanceEntries:
    """AC6: Provenance entries for removed turns have non-empty sources tuples."""

    @pytest.mark.asyncio
    async def test_ac6_entries_returned_for_summarized_turns(self):
        """AC6: Summarizing 7 turns produces provenance entries."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert len(result.entries) > 0

    @pytest.mark.asyncio
    async def test_ac6_all_entries_have_nonempty_sources(self):
        """AC6: Every provenance entry from the transformer has a non-empty sources tuple."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        for entry in result.entries:
            assert entry.sources, f"Entry {entry.id} has empty sources"
            assert len(entry.sources) > 0

    @pytest.mark.asyncio
    async def test_ac6_entry_count_matches_summarized_items(self):
        """AC6: One provenance entry per summarized turn (7 turns summarized → 7 entries)."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert len(result.entries) == 7  # 10 - 3 kept

    @pytest.mark.asyncio
    async def test_ac6_entries_are_transformer_origin(self):
        """AC6: All entries have origin.kind == 'transformer'."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        for entry in result.entries:
            assert entry.origin.kind == "transformer"

    @pytest.mark.asyncio
    async def test_ac6_entry_sources_are_strings(self):
        """AC6: sources are non-empty tuples of strings (ULID or similar)."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        for entry in result.entries:
            assert isinstance(entry.sources, tuple)
            for src in entry.sources:
                assert isinstance(src, str)
                assert len(src) > 0

    @pytest.mark.asyncio
    async def test_ac6_noop_produces_no_entries(self):
        """AC4/AC6: No-op path returns empty entries list."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(3)

        result = await transformer.transform(content, make_events())

        assert result.entries == []


# ---------------------------------------------------------------------------
# AC10: keep_within_tokens strategy
# ---------------------------------------------------------------------------


class TestKeepWithinTokens:
    """AC10: keep_within_tokens keeps the most-recent turns fitting within keep_tokens."""

    @pytest.mark.asyncio
    async def test_ac10_basic_keep_within_tokens(self):
        """AC10: keep_within_tokens=20 with varied turn sizes — keeps most-recent fitting items."""
        # Each TextBlock "turn_N" is 6 chars = 1 token (6//4 = 1).
        # 20 tokens budget: we can fit 20 * 4 = 80 chars.
        # content items: 10 blocks each "turn_N" (6 chars) → each = 1 token.
        # 20 tokens means all 10 fit → no summarization (keep all).
        # Use tighter budget to force summarization.
        llm = MockLLM(response_text="Token-budget summary.")
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=4)
        transformer = build_transformer(config=config, llm=llm)
        # Each "turn_N" is 6 chars → 1 token each (6//4=1).
        # Budget of 4 tokens keeps up to 4 most-recent turns.
        content = make_text_blocks(8)  # 8 turns

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        # First block is the summary
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Token-budget summary."
        # "turn_N" = 6 chars, CharacterTokenCounter counts per-block: 6//4 = 1 token each.
        # Budget=4 tokens, 8 blocks × 1 token each → keep last 4, summarize first 4.
        kept = result.content[1:]
        assert len(kept) == 4
        for i, block in enumerate(kept):
            assert block is content[4 + i]

    @pytest.mark.asyncio
    async def test_ac10_all_fit_within_tokens_is_noop(self):
        """AC10: If all items fit within keep_tokens budget, no-op (no LLM call)."""
        llm = MockLLM()
        # 1000 tokens is more than enough for 5 short turns
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=1000)
        transformer = build_transformer(config=config, llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac10_zero_budget_keeps_nothing(self):
        """AC10: keep_tokens=0 → all items summarized, kept list is empty."""
        llm = MockLLM(response_text="All summarized.")
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=0)
        transformer = build_transformer(config=config, llm=llm)
        content = make_text_blocks(3)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 1  # just the summary, no kept items
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "All summarized."

    @pytest.mark.asyncio
    async def test_ac10_kept_blocks_are_most_recent(self):
        """AC10: The kept blocks are always the most-recent (tail), not random."""
        llm = MockLLM(response_text="Older turns summary.")
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=8)
        transformer = build_transformer(config=config, llm=llm)
        # "turn_N" = 6 chars. CharacterTokenCounter counts per-block: 6//4 = 1 token each.
        # Budget=8 tokens, 10 blocks × 1 token each → keep last 8, summarize first 2.
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        # Should keep turns 2..9 (last 8), summarize turns 0..1
        kept = result.content[1:]
        assert len(kept) == 8
        for i, block in enumerate(kept):
            assert block is content[2 + i]

    @pytest.mark.asyncio
    async def test_ac10_summarized_items_produce_entries(self):
        """AC10: Summarized items produce provenance entries."""
        llm = MockLLM()
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=4)
        transformer = build_transformer(config=config, llm=llm)
        content = make_text_blocks(8)

        result = await transformer.transform(content, make_events())

        assert len(result.entries) > 0
        for entry in result.entries:
            assert len(entry.sources) > 0


# ---------------------------------------------------------------------------
# AC11: Two summarizations in same session
# ---------------------------------------------------------------------------


class TestTwoSummarizations:
    """AC11: Two calls produce distinct provenance entries."""

    @pytest.mark.asyncio
    async def test_ac11_two_summarizations_distinct_entries(self):
        """AC11: Two separate transform() calls each produce their own provenance entries."""
        llm = MockLLM(response_text="Round summary.")
        config = make_config(keep_last_n=2, max_executions=10)
        transformer = build_transformer(config=config, llm=llm)

        content_a = make_text_blocks(5, prefix="alpha")
        content_b = make_text_blocks(5, prefix="beta")

        result_a = await transformer.transform(content_a, make_events())
        result_b = await transformer.transform(content_b, make_events())

        # Both must produce entries
        assert len(result_a.entries) > 0
        assert len(result_b.entries) > 0

        # All entry IDs must be distinct across both results
        ids_a = {e.id for e in result_a.entries}
        ids_b = {e.id for e in result_b.entries}
        assert ids_a.isdisjoint(ids_b), "Entry IDs must not overlap across summarization calls"

    @pytest.mark.asyncio
    async def test_ac11_two_summarizations_two_llm_calls(self):
        """AC11: Two summarizations → LLM called twice."""
        llm = MockLLM()
        config = make_config(keep_last_n=2, max_executions=10)
        transformer = build_transformer(config=config, llm=llm)

        content_a = make_text_blocks(5, prefix="alpha")
        content_b = make_text_blocks(5, prefix="beta")

        await transformer.transform(content_a, make_events())
        await transformer.transform(content_b, make_events())

        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_ac11_two_summarizations_two_events(self):
        """AC11: Each summarization emits its own summarization_complete event."""
        llm = MockLLM(response_text="Repeated summary.")
        config = make_config(keep_last_n=2, max_executions=10)
        transformer = build_transformer(config=config, llm=llm)

        content = make_text_blocks(5)

        result_a = await transformer.transform(content, make_events())
        result_b = await transformer.transform(content, make_events())

        events_a = [e for e in (result_a.events or []) if e.name == "summarization_complete"]
        events_b = [e for e in (result_b.events or []) if e.name == "summarization_complete"]

        assert len(events_a) == 1
        assert len(events_b) == 1


# ---------------------------------------------------------------------------
# LLM failure propagation
# ---------------------------------------------------------------------------


class TestLLMFailurePropagation:
    """Non-functional: LLM failures propagate out of transform()."""

    @pytest.mark.asyncio
    async def test_llm_failure_propagates_as_exception(self):
        """If LLM raises, transform() raises the same exception."""
        failing_llm = FailingLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=failing_llm)
        content = make_text_blocks(5)

        with pytest.raises(RuntimeError, match="LLM backend unavailable"):
            await transformer.transform(content, make_events())

    @pytest.mark.asyncio
    async def test_llm_failure_not_suppressed(self):
        """LLM failure must not be silently caught and replaced with empty result."""
        failing_llm = FailingLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=failing_llm)
        content = make_text_blocks(5)

        raised = False
        try:
            await transformer.transform(content, make_events())
        except Exception:
            raised = True

        assert raised, "Expected an exception to propagate from LLM failure"


# ---------------------------------------------------------------------------
# Constructor and attribute tests
# ---------------------------------------------------------------------------


class TestConstructorAndAttributes:
    """Verify construction, attribute types, and injected LLM."""

    def test_constructor_accepts_config_and_llm(self):
        """SummarizationTransformer(config, llm) constructs without error."""
        transformer = build_transformer()
        assert transformer is not None

    def test_subscriptions_come_from_config(self):
        """Subscriptions are derived from TransformerConfig.subscriptions."""
        config = make_config(subscriptions=[{"event": "context_overflow"}])
        transformer = build_transformer(config=config)
        names = [s.event_name for s in transformer.subscriptions]
        assert "context_overflow" in names

    def test_max_executions_comes_from_config(self):
        """max_executions is taken from TransformerConfig.max_executions."""
        config = make_config(max_executions=7)
        transformer = build_transformer(config=config)
        assert transformer.max_executions == 7

    def test_execution_count_starts_at_zero(self):
        """execution_count is 0 before any transform calls."""
        transformer = build_transformer()
        assert transformer.execution_count == 0

    @pytest.mark.asyncio
    async def test_ac9_injected_llm_is_called_with_model_config(self):
        """AC9: The injected LLM is called when transform fires, regardless of model key.

        Orchestrator maps config["model"] → LLM callable and injects it here.
        At this layer, we verify the injected LLM is what gets called.
        """
        mock_llm = MockLLM(response_text="Model-config summary.")
        config = make_config(model="summarization", keep_last_n=2)
        transformer = build_transformer(config=config, llm=mock_llm)
        content = make_text_blocks(5)

        await transformer.transform(content, make_events())

        assert len(mock_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_actual_summarization(self):
        """execution_count is NOT incremented by the transformer itself on no-op; only on real runs.

        Note: The Layer increments execution_count after each call. However, the
        SummarizationTransformer (like PassiveTransformer) does NOT self-increment.
        This test verifies that execution_count stays 0 before Layer involvement,
        i.e., the transformer doesn't do it internally on no-op.
        """
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(3)  # no-op path

        await transformer.transform(content, make_events())

        # Transformer should not self-increment
        assert transformer.execution_count == 0


# ---------------------------------------------------------------------------
# Result structure validation
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Verify TransformationResult fields are correctly populated."""

    @pytest.mark.asyncio
    async def test_result_has_transformer_name(self):
        """TransformationResult.transformer_name is set."""
        transformer = build_transformer()
        content = make_text_blocks(10)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.transformer_name, str)
        assert result.transformer_name != ""

    @pytest.mark.asyncio
    async def test_result_has_source_layer(self):
        """TransformationResult.source_layer is a string."""
        transformer = build_transformer()
        content = make_text_blocks(10)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.source_layer, str)

    @pytest.mark.asyncio
    async def test_result_content_starts_with_summary_block(self):
        """When summarization fires, result.content[0] is the TextBlock from LLM."""
        summary = "Exactly this text."
        llm = MockLLM(response_text=summary)
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == summary

    @pytest.mark.asyncio
    async def test_result_content_order_summary_then_kept(self):
        """result.content = [summary] + kept_items (summary first, then oldest-to-newest kept)."""
        llm = MockLLM(response_text="S")
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(4)  # summarize [0,1], keep [2,3]

        result = await transformer.transform(content, make_events())

        assert len(result.content) == 3
        assert result.content[0].text == "S"
        assert result.content[1] is content[2]
        assert result.content[2] is content[3]

    @pytest.mark.asyncio
    async def test_result_entries_is_list(self):
        """result.entries is always a list (not None)."""
        transformer = build_transformer()
        content = make_text_blocks(5)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.entries, list)

    @pytest.mark.asyncio
    async def test_result_events_is_list_or_none(self):
        """result.events is a list (not some other type)."""
        transformer = build_transformer()
        content = make_text_blocks(10)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.events, (list, type(None)))


# ---------------------------------------------------------------------------
# TestSummarizationTransformerBuild: build(cls, config, deps) classmethod
# ---------------------------------------------------------------------------


class TestSummarizationTransformerBuild:
    """FR1-FR10: SummarizationTransformer.build(cls, config, deps) classmethod.

    Covers:
    - FR1: build is a classmethod
    - FR2: build accepts (cls, config: TransformerConfig, deps: Dependencies) -> Self
    - FR3: deps.llm is None raises ConfigError mentioning 'summarize' or transformer type
    - FR4: no config["model"] → uses deps.llm["default"]
    - FR5: config["model"] present and key in deps.llm → uses that key
    - FR6: config["model"] present but key absent → falls back to deps.llm["default"]
    - FR7: config.config is None or empty → uses deps.llm["default"]
    - FR8: returns an instance of SummarizationTransformer
    - FR9: returned instance satisfies isinstance(result, Transformer)
    - FR10: existing __init__(config, llm) still works unchanged
    """

    # ------------------------------------------------------------------
    # Imports / fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def transformer_cls(self):
        from sr2.pipeline.transformers.summarization import SummarizationTransformer
        return SummarizationTransformer

    @pytest.fixture
    def default_llm(self):
        return MockLLM(response_text="default llm summary")

    @pytest.fixture
    def named_llm(self):
        return MockLLM(response_text="named llm summary")

    def make_deps(self, default=None, extras: dict | None = None):
        """Build a Dependencies with llm dict, or None."""
        from sr2.pipeline.dependencies import Dependencies

        if default is None and extras is None:
            return Dependencies(llm=None)
        llm_dict: dict = {}
        if default is not None:
            llm_dict["default"] = default
        if extras:
            llm_dict.update(extras)
        return Dependencies(llm=llm_dict)

    # ------------------------------------------------------------------
    # FR1: build is a classmethod
    # ------------------------------------------------------------------

    def test_fr1_build_is_classmethod(self, transformer_cls):
        """FR1: build is accessible as a classmethod on SummarizationTransformer."""
        import inspect
        assert isinstance(
            inspect.getattr_static(transformer_cls, "build"),
            classmethod,
        ), "build must be a classmethod"

    # ------------------------------------------------------------------
    # FR2: build accepts correct signature
    # ------------------------------------------------------------------

    def test_fr2_build_accepts_config_and_deps(self, transformer_cls, default_llm):
        """FR2: build(config, deps) is callable with TransformerConfig and Dependencies."""
        config = make_config()
        deps = self.make_deps(default=default_llm)
        # Should not raise
        result = transformer_cls.build(config, deps)
        assert result is not None

    # ------------------------------------------------------------------
    # FR3: deps.llm is None → ConfigError
    # ------------------------------------------------------------------

    def test_fr3_raises_config_error_when_llm_is_none(self, transformer_cls):
        """FR3: build raises ConfigError (not TypeError) when deps.llm is None."""
        from sr2.config.models import ConfigError
        from sr2.pipeline.dependencies import Dependencies

        config = make_config()
        deps = Dependencies(llm=None)

        with pytest.raises(ConfigError):
            transformer_cls.build(config, deps)

    def test_fr3_error_is_config_error_not_type_error(self, transformer_cls):
        """FR3: The exception type must be ConfigError, not TypeError or AttributeError."""
        from sr2.config.models import ConfigError
        from sr2.pipeline.dependencies import Dependencies

        config = make_config()
        deps = Dependencies(llm=None)

        exc = None
        try:
            transformer_cls.build(config, deps)
        except Exception as e:
            exc = e

        assert exc is not None, "Expected an exception"
        assert isinstance(exc, ConfigError), (
            f"Expected ConfigError, got {type(exc).__name__}: {exc}"
        )

    def test_fr3_error_message_mentions_summarize_or_type(self, transformer_cls):
        """FR3: ConfigError message mentions 'summarize' or the transformer type."""
        from sr2.config.models import ConfigError
        from sr2.pipeline.dependencies import Dependencies

        config = make_config()
        deps = Dependencies(llm=None)

        with pytest.raises(ConfigError) as exc_info:
            transformer_cls.build(config, deps)

        message = str(exc_info.value).lower()
        assert "summariz" in message or "summarization" in message or "llm" in message, (
            f"ConfigError message '{exc_info.value}' should mention the transformer type or llm requirement"
        )

    # ------------------------------------------------------------------
    # FR4: no config["model"] → uses deps.llm["default"]
    # ------------------------------------------------------------------

    def test_fr4_no_model_key_uses_default_llm(self, transformer_cls, default_llm):
        """FR4: When config.config has no 'model' key, uses deps.llm['default']."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        # Build config without a 'model' key in inner dict
        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"keep_strategy": "keep_last_n", "keep_last_n": 3},
            max_executions=10,
        )
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        # The injected LLM should be the default one
        assert result._llm is default_llm

    # ------------------------------------------------------------------
    # FR5: config["model"] present and key in deps.llm → uses that key
    # ------------------------------------------------------------------

    def test_fr5_named_model_key_uses_named_llm(self, transformer_cls, default_llm, named_llm):
        """FR5: config['model'] present and key exists in deps.llm → uses that LLM."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"keep_strategy": "keep_last_n", "keep_last_n": 3, "model": "haiku"},
            max_executions=10,
        )
        deps = self.make_deps(default=default_llm, extras={"haiku": named_llm})

        result = transformer_cls.build(config, deps)

        assert result._llm is named_llm

    def test_fr5_named_model_does_not_use_default(self, transformer_cls, default_llm, named_llm):
        """FR5: When named model key exists, default LLM is not selected."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"model": "opus"},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm, extras={"opus": named_llm})

        result = transformer_cls.build(config, deps)

        assert result._llm is not default_llm
        assert result._llm is named_llm

    # ------------------------------------------------------------------
    # FR6: config["model"] key absent from deps.llm → falls back to default
    # ------------------------------------------------------------------

    def test_fr6_missing_model_key_falls_back_to_default(self, transformer_cls, default_llm):
        """FR6: config['model'] set but key absent from deps.llm → falls back to default."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"model": "nonexistent-model"},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm)  # only "default" key

        result = transformer_cls.build(config, deps)

        assert result._llm is default_llm

    def test_fr6_fallback_does_not_raise(self, transformer_cls, default_llm):
        """FR6: Falling back to default when named key is absent does not raise."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"model": "absent-key"},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm)

        # Must not raise
        result = transformer_cls.build(config, deps)
        assert result is not None

    # ------------------------------------------------------------------
    # FR7: config.config is None or empty → uses deps.llm["default"]
    # ------------------------------------------------------------------

    def test_fr7_empty_config_dict_uses_default_llm(self, transformer_cls, default_llm):
        """FR7: config.config = {} (empty dict) → uses deps.llm['default']."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert result._llm is default_llm

    def test_fr7_default_config_dict_uses_default_llm(self, transformer_cls, default_llm):
        """FR7: TransformerConfig with default config (no explicit config arg) → uses default LLM."""
        from sr2.config.models import TransformerConfig

        # TransformerConfig.config defaults to {}
        config = TransformerConfig(type="summarization")
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert result._llm is default_llm

    # ------------------------------------------------------------------
    # FR8: returns instance of SummarizationTransformer
    # ------------------------------------------------------------------

    def test_fr8_returns_summarization_transformer_instance(self, transformer_cls, default_llm):
        """FR8: build() returns an instance of SummarizationTransformer."""
        config = make_config()
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert isinstance(result, transformer_cls)

    def test_fr8_returned_instance_has_correct_config(self, transformer_cls, default_llm):
        """FR8: The returned instance holds the provided config."""
        config = make_config(keep_last_n=7, max_executions=5)
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert result.max_executions == 5
        assert result._config is config

    # ------------------------------------------------------------------
    # FR9: returned instance satisfies Transformer protocol
    # ------------------------------------------------------------------

    def test_fr9_returned_instance_satisfies_transformer_protocol(
        self, transformer_cls, default_llm
    ):
        """FR9: isinstance(result, Transformer) is True."""
        from sr2.pipeline.protocols import Transformer

        config = make_config()
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert isinstance(result, Transformer)

    # ------------------------------------------------------------------
    # FR10: existing __init__(config, llm) unchanged — existing tests still work
    # ------------------------------------------------------------------

    def test_fr10_direct_init_still_works(self, transformer_cls):
        """FR10: SummarizationTransformer(config, llm) still constructs without error."""
        llm = MockLLM()
        config = make_config()
        # This is the existing construction path — must not be broken
        instance = transformer_cls(config=config, llm=llm)
        assert instance is not None
        assert isinstance(instance, transformer_cls)

    def test_fr10_direct_init_retains_correct_llm(self, transformer_cls):
        """FR10: Directly constructed instance has the injected LLM."""
        llm = MockLLM(response_text="direct init llm")
        config = make_config()
        instance = transformer_cls(config=config, llm=llm)
        assert instance._llm is llm

    @pytest.mark.asyncio
    async def test_fr10_direct_init_transform_still_works(self, transformer_cls):
        """FR10: Directly constructed instance can still call transform() correctly."""
        llm = MockLLM(response_text="still works")
        config = make_config(keep_last_n=2)
        instance = transformer_cls(config=config, llm=llm)
        content = make_text_blocks(5)

        result = await instance.transform(content, make_events())

        assert result.content is not None
        assert result.content[0].text == "still works"

    # ------------------------------------------------------------------
    # Integration: build() → transform() end-to-end
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_build_then_transform_uses_injected_llm(self, transformer_cls, default_llm):
        """Integration: build() then transform() calls the correct injected LLM."""
        config = make_config(keep_last_n=2)
        deps = self.make_deps(default=default_llm)

        instance = transformer_cls.build(config, deps)
        content = make_text_blocks(5)

        result = await instance.transform(content, make_events())

        assert result.content is not None
        assert result.content[0].text == "default llm summary"
        assert len(default_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_build_with_named_model_then_transform_uses_named_llm(
        self, transformer_cls, default_llm, named_llm
    ):
        """Integration: build() with named model key, then transform() uses named LLM."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"keep_strategy": "keep_last_n", "keep_last_n": 2, "model": "sonnet"},
            max_executions=10,
        )
        deps = self.make_deps(default=default_llm, extras={"sonnet": named_llm})

        instance = transformer_cls.build(config, deps)
        content = make_text_blocks(5)

        result = await instance.transform(content, make_events())

        assert result.content is not None
        assert result.content[0].text == "named llm summary"
        assert len(named_llm.calls) == 1
        assert len(default_llm.calls) == 0
