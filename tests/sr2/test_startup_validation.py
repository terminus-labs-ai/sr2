"""Tests for SR2 startup validation of required LLM callables.

SR2.__init__() must validate that when PipelineConfig enables features
requiring LLM callables (fast_complete, embed), those callables are
actually provided. A new SR2ConfigurationError collects all violations
into a single message at init time.

Plan: audit-fix-12
"""

import pytest
from unittest.mock import MagicMock

from sr2.sr2 import SR2, SR2Config
from sr2.config.models import (
    PipelineConfig,
    MemoryConfig,
    SummarizationConfig,
    CompactionConfig,
    RetrievalConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sr2_config(
    pipeline_config: PipelineConfig,
    fast_complete=None,
    embed=None,
) -> SR2Config:
    """Build a minimal SR2Config with a preloaded PipelineConfig.

    Uses preloaded_config so no filesystem access is required. config_dir and
    agent_yaml are set to minimal valid values.
    """
    return SR2Config(
        config_dir="/tmp",
        agent_yaml={"name": "test_agent"},
        fast_complete=fast_complete,
        embed=embed,
        preloaded_config=pipeline_config,
    )


def _make_pipeline_config(**overrides) -> PipelineConfig:
    """Build a PipelineConfig with all LLM-dependent features disabled by default.

    Individual tests enable specific features to test validation.
    """
    defaults = {
        "memory": MemoryConfig(extract=False),
        "summarization": SummarizationConfig(enabled=False),
        "compaction": CompactionConfig(strategy="rule_based"),
        "retrieval": RetrievalConfig(enabled=False),
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


async def _dummy_fast_complete(system: str, prompt: str) -> str:
    return "ok"


async def _dummy_embed(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Import assertion: SR2ConfigurationError must be importable from sr2.sr2
# and from the sr2 package itself.
# ---------------------------------------------------------------------------

class TestSR2ConfigurationErrorExported:
    """SR2ConfigurationError must be importable from sr2.sr2 and from sr2."""

    def test_importable_from_sr2_module(self):
        from sr2.sr2 import SR2ConfigurationError  # noqa: F401

    def test_importable_from_sr2_package(self):
        from sr2 import SR2ConfigurationError  # noqa: F401

    def test_is_exception_subclass(self):
        from sr2.sr2 import SR2ConfigurationError
        assert issubclass(SR2ConfigurationError, Exception)


# ---------------------------------------------------------------------------
# Tests: features that require fast_complete
# ---------------------------------------------------------------------------

class TestMemoryExtractValidation:
    """memory.extract=True requires fast_complete."""

    def test_extract_enabled_no_callable_raises(self):
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=True),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "memory.extract" in error_msg
        assert "fast_complete" in error_msg

    def test_extract_disabled_no_callable_ok(self):
        """memory.extract=False: no callable required."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=False),
            ),
            fast_complete=None,
        )
        # Should not raise
        SR2(config)

    def test_extract_enabled_with_callable_ok(self):
        """memory.extract=True + fast_complete provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=True),
            ),
            fast_complete=_dummy_fast_complete,
        )
        # Should not raise
        SR2(config)


class TestSummarizationValidation:
    """summarization.enabled=True requires fast_complete."""

    def test_summarization_enabled_no_callable_raises(self):
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                summarization=SummarizationConfig(enabled=True),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "summarization.enabled" in error_msg
        assert "fast_complete" in error_msg

    def test_summarization_disabled_no_callable_ok(self):
        """summarization.enabled=False: no callable required."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                summarization=SummarizationConfig(enabled=False),
            ),
            fast_complete=None,
        )
        # Should not raise
        SR2(config)

    def test_summarization_enabled_with_callable_ok(self):
        """summarization.enabled=True + fast_complete provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                summarization=SummarizationConfig(enabled=True),
            ),
            fast_complete=_dummy_fast_complete,
        )
        # Should not raise
        SR2(config)


class TestCompactionStrategyValidation:
    """compaction.strategy='llm' or 'hybrid' requires fast_complete."""

    def test_llm_compaction_no_callable_raises(self):
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                compaction=CompactionConfig(strategy="llm"),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "compaction.strategy" in error_msg
        assert "fast_complete" in error_msg

    def test_hybrid_compaction_no_callable_raises(self):
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                compaction=CompactionConfig(strategy="hybrid"),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "compaction.strategy" in error_msg
        assert "fast_complete" in error_msg

    def test_rule_based_compaction_no_callable_ok(self):
        """compaction.strategy='rule_based': no LLM callable required."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                compaction=CompactionConfig(strategy="rule_based"),
            ),
            fast_complete=None,
        )
        # Should not raise
        SR2(config)

    def test_llm_compaction_with_callable_ok(self):
        """compaction.strategy='llm' + fast_complete provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                compaction=CompactionConfig(strategy="llm"),
            ),
            fast_complete=_dummy_fast_complete,
        )
        # Should not raise
        SR2(config)

    def test_hybrid_compaction_with_callable_ok(self):
        """compaction.strategy='hybrid' + fast_complete provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                compaction=CompactionConfig(strategy="hybrid"),
            ),
            fast_complete=_dummy_fast_complete,
        )
        # Should not raise
        SR2(config)


# ---------------------------------------------------------------------------
# Tests: features that require embed
# ---------------------------------------------------------------------------

class TestRetrievalStrategyValidation:
    """retrieval.strategy='hybrid' or 'semantic' requires embed."""

    def test_hybrid_retrieval_no_embed_raises(self):
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                retrieval=RetrievalConfig(enabled=True, strategy="hybrid"),
            ),
            embed=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "retrieval.strategy" in error_msg
        assert "embed" in error_msg

    def test_semantic_retrieval_no_embed_raises(self):
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                retrieval=RetrievalConfig(enabled=True, strategy="semantic"),
            ),
            embed=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "retrieval.strategy" in error_msg
        assert "embed" in error_msg

    def test_keyword_retrieval_no_embed_ok(self):
        """retrieval.strategy='keyword': no embed callable required."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                retrieval=RetrievalConfig(enabled=True, strategy="keyword"),
            ),
            embed=None,
        )
        # Should not raise
        SR2(config)

    def test_retrieval_disabled_hybrid_strategy_no_embed_ok(self):
        """retrieval.enabled=False: embed not required even if strategy=hybrid."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                retrieval=RetrievalConfig(enabled=False, strategy="hybrid"),
            ),
            embed=None,
        )
        # Should not raise
        SR2(config)

    def test_hybrid_retrieval_with_embed_ok(self):
        """retrieval.strategy='hybrid' + embed provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                retrieval=RetrievalConfig(enabled=True, strategy="hybrid"),
            ),
            embed=_dummy_embed,
        )
        # Should not raise
        SR2(config)

    def test_semantic_retrieval_with_embed_ok(self):
        """retrieval.strategy='semantic' + embed provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                retrieval=RetrievalConfig(enabled=True, strategy="semantic"),
            ),
            embed=_dummy_embed,
        )
        # Should not raise
        SR2(config)


# ---------------------------------------------------------------------------
# Tests: multiple violations collected in one error
# ---------------------------------------------------------------------------

class TestMultipleViolationsReported:
    """When multiple features are misconfigured, all violations appear in one error."""

    def test_multiple_errors_reported(self):
        """extract + summarization + llm compaction all missing fast_complete."""
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=True),
                summarization=SummarizationConfig(enabled=True),
                compaction=CompactionConfig(strategy="llm"),
            ),
            fast_complete=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        # All three violations must be present in the single error message
        assert "memory.extract" in error_msg
        assert "summarization.enabled" in error_msg
        assert "compaction.strategy" in error_msg

    def test_fast_complete_and_embed_both_missing_reports_all(self):
        """Both fast_complete and embed missing: error covers both categories."""
        from sr2.sr2 import SR2ConfigurationError

        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=True),
                retrieval=RetrievalConfig(enabled=True, strategy="semantic"),
            ),
            fast_complete=None,
            embed=None,
        )

        with pytest.raises(SR2ConfigurationError) as exc_info:
            SR2(config)

        error_msg = str(exc_info.value)
        assert "memory.extract" in error_msg
        assert "retrieval.strategy" in error_msg


# ---------------------------------------------------------------------------
# Tests: all features enabled with both callables — no error
# ---------------------------------------------------------------------------

class TestAllFeaturesWithCallablesOk:
    """When all callables are provided, all features can be enabled without error."""

    def test_all_features_with_callables_ok(self):
        """All LLM-dependent features enabled, both callables provided: no error."""
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=True),
                summarization=SummarizationConfig(enabled=True),
                compaction=CompactionConfig(strategy="hybrid"),
                retrieval=RetrievalConfig(enabled=True, strategy="hybrid"),
            ),
            fast_complete=_dummy_fast_complete,
            embed=_dummy_embed,
        )
        # Should not raise
        SR2(config)


# ---------------------------------------------------------------------------
# Tests: completely disabled features — no callables needed
# ---------------------------------------------------------------------------

class TestAllFeaturesDisabledNoCallablesOk:
    """All LLM-dependent features disabled: no callables required."""

    def test_all_disabled_no_callables_ok(self):
        config = _make_sr2_config(
            pipeline_config=_make_pipeline_config(
                memory=MemoryConfig(extract=False),
                summarization=SummarizationConfig(enabled=False),
                compaction=CompactionConfig(strategy="rule_based"),
                retrieval=RetrievalConfig(enabled=False),
            ),
            fast_complete=None,
            embed=None,
        )
        # Should not raise
        SR2(config)
