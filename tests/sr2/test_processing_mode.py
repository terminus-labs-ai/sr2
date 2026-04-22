"""Tests for ProcessingMode enum and its integration with proxy_optimize / proxy_post_process.

Covers audit-fix-05: Adding a ProcessingMode enum (FULL/LIGHTWEIGHT/PASSTHROUGH) to SR2's
proxy API so the bridge can express optimization intent without bypassing SR2 entirely.

Verifies:
  - ProcessingMode enum exists with FULL, LIGHTWEIGHT, PASSTHROUGH values
  - ProcessingMode is importable from the sr2 package (public API)
  - proxy_optimize() accepts optional processing_mode parameter (default FULL)
  - proxy_post_process() accepts optional processing_mode parameter (default FULL)
  - FULL mode runs all stages (compaction, summarization, memory retrieval)
  - LIGHTWEIGHT mode skips compaction and summarization but tracks turns and retrieves memory
  - PASSTHROUGH mode skips all SR2 processing, returns no system injection
  - ProxyResult includes a processing_mode field reflecting the mode used
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.compaction.engine import ConversationTurn
from sr2.config.models import MemoryConfig, PipelineConfig, RetrievalConfig, SummarizationConfig
from sr2.sr2 import SR2, SR2Config, ProxyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_sr2(
    config_dir: str,
    agent_yaml: dict | None = None,
    preloaded_config: PipelineConfig | None = None,
) -> SR2:
    """Build a minimal SR2 instance without file I/O."""
    return SR2(
        SR2Config(
            config_dir=config_dir,
            agent_yaml=agent_yaml or {},
            preloaded_config=preloaded_config or PipelineConfig(
                memory=MemoryConfig(extract=False),
                summarization=SummarizationConfig(enabled=False),
                retrieval=RetrievalConfig(enabled=False),
            ),
        )
    )


def _make_turn(n: int = 1, role: str = "user", content: str = "Hello") -> ConversationTurn:
    return ConversationTurn(turn_number=n, role=role, content=content)


# ---------------------------------------------------------------------------
# 1. ProcessingMode enum: existence and values
# ---------------------------------------------------------------------------


class TestProcessingModeEnum:
    """ProcessingMode must be a proper enum with the three required members."""

    def test_processing_mode_importable_from_sr2_sr2(self):
        """ProcessingMode must be importable from sr2.sr2."""
        from sr2.sr2 import ProcessingMode  # noqa: F401

    def test_processing_mode_importable_from_sr2_package(self):
        """ProcessingMode must be importable from the sr2 package (public API)."""
        from sr2 import ProcessingMode  # noqa: F401

    def test_processing_mode_is_enum(self):
        """ProcessingMode must be an Enum subclass."""
        import enum
        from sr2.sr2 import ProcessingMode
        assert issubclass(ProcessingMode, enum.Enum)

    def test_processing_mode_is_str_enum(self):
        """ProcessingMode must also subclass str (for JSON-serialisable values)."""
        from sr2.sr2 import ProcessingMode
        assert issubclass(ProcessingMode, str)

    def test_full_member_exists(self):
        from sr2.sr2 import ProcessingMode
        assert hasattr(ProcessingMode, "FULL")

    def test_lightweight_member_exists(self):
        from sr2.sr2 import ProcessingMode
        assert hasattr(ProcessingMode, "LIGHTWEIGHT")

    def test_passthrough_member_exists(self):
        from sr2.sr2 import ProcessingMode
        assert hasattr(ProcessingMode, "PASSTHROUGH")

    def test_full_value_is_string(self):
        from sr2.sr2 import ProcessingMode
        assert ProcessingMode.FULL.value == "full"

    def test_lightweight_value_is_string(self):
        from sr2.sr2 import ProcessingMode
        assert ProcessingMode.LIGHTWEIGHT.value == "lightweight"

    def test_passthrough_value_is_string(self):
        from sr2.sr2 import ProcessingMode
        assert ProcessingMode.PASSTHROUGH.value == "passthrough"

    def test_exactly_three_members(self):
        """There must be exactly 3 members: FULL, LIGHTWEIGHT, PASSTHROUGH."""
        from sr2.sr2 import ProcessingMode
        assert len(ProcessingMode) == 3

    def test_members_are_distinct(self):
        from sr2.sr2 import ProcessingMode
        values = {ProcessingMode.FULL, ProcessingMode.LIGHTWEIGHT, ProcessingMode.PASSTHROUGH}
        assert len(values) == 3


# ---------------------------------------------------------------------------
# 2. ProcessingMode in __init__.py public API
# ---------------------------------------------------------------------------


class TestProcessingModePublicExport:
    """ProcessingMode must be in sr2.__all__ to be part of the public API."""

    def test_in_sr2_all(self):
        import sr2
        assert "ProcessingMode" in sr2.__all__, (
            "ProcessingMode must appear in sr2.__all__"
        )

    def test_accessible_as_attribute(self):
        import sr2
        assert hasattr(sr2, "ProcessingMode"), (
            "ProcessingMode must be accessible as sr2.ProcessingMode"
        )


# ---------------------------------------------------------------------------
# 3. ProxyResult includes processing_mode field
# ---------------------------------------------------------------------------


class TestProxyResultProcessingModeField:
    """ProxyResult dataclass must carry a processing_mode field."""

    def test_proxy_result_has_processing_mode_field(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ProxyResult)}
        assert "processing_mode" in fields, (
            "ProxyResult must have a 'processing_mode' field"
        )

    def test_proxy_result_processing_mode_defaults_to_full(self):
        from sr2.sr2 import ProcessingMode
        result = ProxyResult(
            system_injection=None,
            zones=None,
            compaction_result=None,
            summarization_result=None,
        )
        assert result.processing_mode == ProcessingMode.FULL

    def test_proxy_result_processing_mode_can_be_set(self):
        from sr2.sr2 import ProcessingMode
        result = ProxyResult(
            system_injection=None,
            zones=None,
            compaction_result=None,
            summarization_result=None,
            processing_mode=ProcessingMode.PASSTHROUGH,
        )
        assert result.processing_mode == ProcessingMode.PASSTHROUGH


# ---------------------------------------------------------------------------
# 4. proxy_optimize() signature accepts processing_mode
# ---------------------------------------------------------------------------


class TestProxyOptimizeSignature:
    """proxy_optimize() must accept an optional processing_mode keyword argument."""

    def test_proxy_optimize_has_processing_mode_param(self):
        sig = inspect.signature(SR2.proxy_optimize)
        assert "processing_mode" in sig.parameters, (
            "proxy_optimize() must have a 'processing_mode' parameter"
        )

    def test_processing_mode_defaults_to_full(self):
        from sr2.sr2 import ProcessingMode
        sig = inspect.signature(SR2.proxy_optimize)
        param = sig.parameters["processing_mode"]
        assert param.default == ProcessingMode.FULL, (
            "proxy_optimize() processing_mode must default to ProcessingMode.FULL"
        )

    def test_processing_mode_is_keyword_or_positional(self):
        """processing_mode must be usable as a keyword argument."""
        sig = inspect.signature(SR2.proxy_optimize)
        param = sig.parameters["processing_mode"]
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )


# ---------------------------------------------------------------------------
# 5. proxy_post_process() signature accepts processing_mode
# ---------------------------------------------------------------------------


class TestProxyPostProcessSignature:
    """proxy_post_process() must accept an optional processing_mode keyword argument."""

    def test_proxy_post_process_has_processing_mode_param(self):
        sig = inspect.signature(SR2.proxy_post_process)
        assert "processing_mode" in sig.parameters, (
            "proxy_post_process() must have a 'processing_mode' parameter"
        )

    def test_processing_mode_defaults_to_full(self):
        from sr2.sr2 import ProcessingMode
        sig = inspect.signature(SR2.proxy_post_process)
        param = sig.parameters["processing_mode"]
        assert param.default == ProcessingMode.FULL, (
            "proxy_post_process() processing_mode must default to ProcessingMode.FULL"
        )

    def test_processing_mode_is_keyword_or_positional(self):
        sig = inspect.signature(SR2.proxy_post_process)
        param = sig.parameters["processing_mode"]
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )


# ---------------------------------------------------------------------------
# 6. proxy_optimize() FULL mode — all stages run
# ---------------------------------------------------------------------------


class TestProxyOptimizeFull:
    """FULL mode (default) must run compaction, summarization, and memory retrieval."""

    @pytest.mark.asyncio
    async def test_full_mode_calls_run_compaction(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_compaction", wraps=sr2._conversation.run_compaction) as mock_compact:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="full_session",
                    processing_mode=ProcessingMode.FULL,
                )
                assert mock_compact.called, "FULL mode must call run_compaction()"

    @pytest.mark.asyncio
    async def test_full_mode_calls_run_summarization(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_summarization", new_callable=AsyncMock) as mock_summ:
                mock_summ.return_value = None
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="full_summ_session",
                    processing_mode=ProcessingMode.FULL,
                )
                assert mock_summ.called, "FULL mode must call run_summarization()"

    @pytest.mark.asyncio
    async def test_full_mode_returns_proxy_result(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="full_result_session",
                processing_mode=ProcessingMode.FULL,
            )
            assert isinstance(result, ProxyResult)

    @pytest.mark.asyncio
    async def test_full_mode_result_carries_correct_mode(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="full_mode_field_session",
                processing_mode=ProcessingMode.FULL,
            )
            assert result.processing_mode == ProcessingMode.FULL

    @pytest.mark.asyncio
    async def test_default_mode_is_full_behavior(self):
        """Calling proxy_optimize() without processing_mode must behave identically to FULL."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_compaction", wraps=sr2._conversation.run_compaction) as mock_compact:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="default_mode_session",
                    # No processing_mode argument — uses default
                )
                assert mock_compact.called, "Default mode must run compaction (same as FULL)"

    @pytest.mark.asyncio
    async def test_full_mode_adds_turns_to_conversation(self):
        """FULL mode must always add turns to the conversation manager."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "full_turns_session"
            await sr2.proxy_optimize(
                new_turns=[_make_turn(1), _make_turn(2, role="assistant", content="OK")],
                session_id=session_id,
                processing_mode=ProcessingMode.FULL,
            )
            zones = sr2._conversation.zones(session_id)
            total_turns = len(zones.raw) + len(zones.compacted)
            assert total_turns >= 2, "FULL mode must add all provided turns to the conversation"


# ---------------------------------------------------------------------------
# 7. proxy_optimize() LIGHTWEIGHT mode — turns tracked, compaction/summarization skipped
# ---------------------------------------------------------------------------


class TestProxyOptimizeLightweight:
    """LIGHTWEIGHT mode must track turns and retrieve memory, but skip compaction/summarization."""

    @pytest.mark.asyncio
    async def test_lightweight_skips_run_compaction(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_compaction") as mock_compact:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="lw_compact_session",
                    processing_mode=ProcessingMode.LIGHTWEIGHT,
                )
                assert not mock_compact.called, "LIGHTWEIGHT mode must not call run_compaction()"

    @pytest.mark.asyncio
    async def test_lightweight_skips_run_summarization(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_summarization", new_callable=AsyncMock) as mock_summ:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="lw_summ_session",
                    processing_mode=ProcessingMode.LIGHTWEIGHT,
                )
                assert not mock_summ.called, "LIGHTWEIGHT mode must not call run_summarization()"

    @pytest.mark.asyncio
    async def test_lightweight_adds_turns_to_conversation(self):
        """LIGHTWEIGHT mode must still add turns (conversation tracking)."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "lw_turns_session"
            await sr2.proxy_optimize(
                new_turns=[_make_turn(1), _make_turn(2, role="assistant", content="Ack")],
                session_id=session_id,
                processing_mode=ProcessingMode.LIGHTWEIGHT,
            )
            zones = sr2._conversation.zones(session_id)
            total_turns = len(zones.raw) + len(zones.compacted)
            assert total_turns >= 2, "LIGHTWEIGHT mode must add turns to the conversation"

    @pytest.mark.asyncio
    async def test_lightweight_returns_proxy_result(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="lw_result_session",
                processing_mode=ProcessingMode.LIGHTWEIGHT,
            )
            assert isinstance(result, ProxyResult)

    @pytest.mark.asyncio
    async def test_lightweight_result_carries_correct_mode(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="lw_mode_field_session",
                processing_mode=ProcessingMode.LIGHTWEIGHT,
            )
            assert result.processing_mode == ProcessingMode.LIGHTWEIGHT

    @pytest.mark.asyncio
    async def test_lightweight_runs_memory_retrieval_when_query_given(self):
        """LIGHTWEIGHT mode must still run memory retrieval (not skipped like PASSTHROUGH)."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._retriever, "retrieve", new_callable=AsyncMock, return_value=[]) as mock_retrieve:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="lw_memory_session",
                    retrieval_query="what did we discuss?",
                    processing_mode=ProcessingMode.LIGHTWEIGHT,
                )
                assert mock_retrieve.called, (
                    "LIGHTWEIGHT mode must call retriever.retrieve() when retrieval_query is given"
                )

    @pytest.mark.asyncio
    async def test_lightweight_compaction_result_is_none(self):
        """ProxyResult.compaction_result must be None in LIGHTWEIGHT mode."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="lw_no_compact_session",
                processing_mode=ProcessingMode.LIGHTWEIGHT,
            )
            assert result.compaction_result is None, (
                "LIGHTWEIGHT mode must not produce a compaction result"
            )

    @pytest.mark.asyncio
    async def test_lightweight_summarization_result_is_none(self):
        """ProxyResult.summarization_result must be None in LIGHTWEIGHT mode."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="lw_no_summ_session",
                processing_mode=ProcessingMode.LIGHTWEIGHT,
            )
            assert result.summarization_result is None, (
                "LIGHTWEIGHT mode must not produce a summarization result"
            )


# ---------------------------------------------------------------------------
# 8. proxy_optimize() PASSTHROUGH mode — all SR2 processing skipped
# ---------------------------------------------------------------------------


class TestProxyOptimizePassthrough:
    """PASSTHROUGH mode must skip all SR2 processing and return an empty ProxyResult."""

    @pytest.mark.asyncio
    async def test_passthrough_skips_run_compaction(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_compaction") as mock_compact:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="pt_compact_session",
                    processing_mode=ProcessingMode.PASSTHROUGH,
                )
                assert not mock_compact.called, "PASSTHROUGH mode must not call run_compaction()"

    @pytest.mark.asyncio
    async def test_passthrough_skips_run_summarization(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._conversation, "run_summarization", new_callable=AsyncMock) as mock_summ:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="pt_summ_session",
                    processing_mode=ProcessingMode.PASSTHROUGH,
                )
                assert not mock_summ.called, "PASSTHROUGH mode must not call run_summarization()"

    @pytest.mark.asyncio
    async def test_passthrough_skips_memory_retrieval(self):
        """PASSTHROUGH mode must not call retriever.retrieve() even with a retrieval_query."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._retriever, "retrieve", new_callable=AsyncMock) as mock_retrieve:
                await sr2.proxy_optimize(
                    new_turns=[_make_turn()],
                    session_id="pt_memory_session",
                    retrieval_query="some query",
                    processing_mode=ProcessingMode.PASSTHROUGH,
                )
                assert not mock_retrieve.called, (
                    "PASSTHROUGH mode must not call retriever.retrieve()"
                )

    @pytest.mark.asyncio
    async def test_passthrough_returns_proxy_result(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="pt_result_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            assert isinstance(result, ProxyResult)

    @pytest.mark.asyncio
    async def test_passthrough_system_injection_is_none(self):
        """PASSTHROUGH mode must return no system injection."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="pt_injection_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            assert result.system_injection is None, (
                "PASSTHROUGH mode must return system_injection=None"
            )

    @pytest.mark.asyncio
    async def test_passthrough_compaction_result_is_none(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="pt_no_compact_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            assert result.compaction_result is None

    @pytest.mark.asyncio
    async def test_passthrough_summarization_result_is_none(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="pt_no_summ_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            assert result.summarization_result is None

    @pytest.mark.asyncio
    async def test_passthrough_result_carries_correct_mode(self):
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="pt_mode_field_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            assert result.processing_mode == ProcessingMode.PASSTHROUGH

    @pytest.mark.asyncio
    async def test_passthrough_skips_scope_detection(self):
        """PASSTHROUGH mode must not run scope detection."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            # Attach a mock scope detector to confirm it's not called
            mock_detector = MagicMock()
            mock_detector.detect = AsyncMock(return_value={})
            sr2._scope_detector = mock_detector
            await sr2.proxy_optimize(
                new_turns=[_make_turn()],
                session_id="pt_scope_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            assert not mock_detector.detect.called, (
                "PASSTHROUGH mode must not call scope_detector.detect()"
            )

    @pytest.mark.asyncio
    async def test_passthrough_adds_turns_for_observability(self):
        """PASSTHROUGH mode must still add turns to the conversation manager.

        The plan specifies: "Add turns -- ALWAYS (even passthrough tracks turns
        for observability)". PASSTHROUGH skips compaction/summarization/retrieval
        but turn tracking must happen so the request is visible in traces and
        zone metrics. The hash-state-pollution concern from the original bridge
        bypass is handled at the bridge-engine level (hash replacement), not by
        suppressing turn tracking in SR2.
        """
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "pt_turns_tracked_session"
            await sr2.proxy_optimize(
                new_turns=[_make_turn(1), _make_turn(2, role="assistant", content="OK")],
                session_id=session_id,
                processing_mode=ProcessingMode.PASSTHROUGH,
            )
            zones = sr2._conversation.zones(session_id)
            total_turns = len(zones.raw) + len(zones.compacted)
            assert total_turns >= 2, (
                "PASSTHROUGH mode must add turns to the conversation manager for observability"
            )


# ---------------------------------------------------------------------------
# 9. proxy_post_process() mode gating
# ---------------------------------------------------------------------------


class TestProxyPostProcessMode:
    """proxy_post_process() must gate post-processing stages on processing_mode."""

    @pytest.mark.asyncio
    async def test_full_mode_calls_post_processor(self):
        """FULL mode must invoke the PostLLMProcessor."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._post_processor, "process", new_callable=AsyncMock) as mock_process:
                await sr2.proxy_post_process(
                    assistant_text="Hello back",
                    session_id="ppfull_session",
                    processing_mode=ProcessingMode.FULL,
                )
                assert mock_process.called, "FULL mode must call post_processor.process()"

    @pytest.mark.asyncio
    async def test_passthrough_skips_post_processor(self):
        """PASSTHROUGH mode must not invoke the PostLLMProcessor."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._post_processor, "process", new_callable=AsyncMock) as mock_process:
                await sr2.proxy_post_process(
                    assistant_text="Hello back",
                    session_id="pppt_session",
                    processing_mode=ProcessingMode.PASSTHROUGH,
                )
                assert not mock_process.called, (
                    "PASSTHROUGH mode must not call post_processor.process()"
                )

    @pytest.mark.asyncio
    async def test_passthrough_does_not_raise(self):
        """PASSTHROUGH proxy_post_process() must return cleanly without raising."""
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            # Should not raise
            await sr2.proxy_post_process(
                assistant_text="Response",
                session_id="pppt_noerr_session",
                processing_mode=ProcessingMode.PASSTHROUGH,
            )

    @pytest.mark.asyncio
    async def test_lightweight_runs_memory_extraction_only(self):
        """LIGHTWEIGHT mode must call post_processor.process() but skip compaction.

        The PostLLMProcessor already has an extract_only parameter — LIGHTWEIGHT
        must pass extract_only=True.
        """
        from sr2.sr2 import ProcessingMode
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._post_processor, "process", new_callable=AsyncMock) as mock_process:
                await sr2.proxy_post_process(
                    assistant_text="Response",
                    session_id="pplw_session",
                    processing_mode=ProcessingMode.LIGHTWEIGHT,
                )
                assert mock_process.called, (
                    "LIGHTWEIGHT mode must call post_processor.process()"
                )
                # Verify it was called with extract_only=True.
                # PostLLMProcessor.process() signature: process(turn, session_id,
                #   current_context=None, extract_only=False, model_hint=None)
                # The implementer must pass extract_only as a keyword argument.
                call_kwargs = mock_process.call_args
                extract_only_value = call_kwargs.kwargs.get("extract_only")
                assert extract_only_value is True, (
                    "LIGHTWEIGHT proxy_post_process must call process() with extract_only=True"
                )

    @pytest.mark.asyncio
    async def test_default_mode_is_full_post_process(self):
        """Calling proxy_post_process() without mode must invoke the PostLLMProcessor (FULL)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            with patch.object(sr2._post_processor, "process", new_callable=AsyncMock) as mock_process:
                await sr2.proxy_post_process(
                    assistant_text="Response",
                    session_id="ppdefault_session",
                    # No processing_mode — defaults to FULL
                )
                assert mock_process.called, "Default mode (FULL) must call post_processor.process()"
