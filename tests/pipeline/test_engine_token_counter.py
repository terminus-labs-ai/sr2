"""Tests for PipelineEngine token counter injection.

Verifies that PipelineEngine accepts and uses an injected TokenCounter
instead of hardcoded character estimation.
"""

from __future__ import annotations

import pytest

from sr2.config.models import PipelineConfig, LayerConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.tokenization.counting import CharacterCounter, TiktokenCounter, create_token_counter


class _SpyTokenCounter:
    """Token counter that records calls for verification."""

    def __init__(self) -> None:
        self.count_calls: list[str] = []
        self.truncate_calls: list[tuple[str, int]] = []

    def count(self, text: str) -> int:
        self.count_calls.append(text)
        return len(text) // 4

    def truncate(self, text: str, max_tokens: int) -> str:
        self.truncate_calls.append((text, max_tokens))
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]


class TestEngineTokenCounter:
    """Verify PipelineEngine accepts and uses an injected TokenCounter."""

    def test_engine_accepts_token_counter(self):
        """PipelineEngine(config, token_counter=CharacterCounter()) doesn't error."""
        config = PipelineConfig(layers=[])
        counter = CharacterCounter()
        engine = PipelineEngine(config, token_counter=counter)
        assert engine._token_counter is counter

    def test_truncate_uses_token_counter(self):
        """When engine truncates content, it uses the injected counter's truncate()."""
        config = PipelineConfig(layers=[])
        spy = _SpyTokenCounter()
        engine = PipelineEngine(config, token_counter=spy)

        long_content = "a" * 1000
        result = engine._truncate_to_budget(long_content, 10)

        # Spy should have been called
        assert len(spy.truncate_calls) == 1
        assert spy.truncate_calls[0] == (long_content, 10)
        # Result should be truncated
        assert len(result) == 40  # 10 tokens * 4 chars

    def test_default_token_counter_if_none(self):
        """If no token_counter provided, engine creates a default."""
        config = PipelineConfig(layers=[])
        engine = PipelineEngine(config)
        assert engine._token_counter is not None
        assert isinstance(engine._token_counter, (TiktokenCounter, CharacterCounter))

    def test_engine_accepts_registry_deps(self):
        """Engine accepts reducer_deps and provider_deps for PluginRegistry."""
        config = PipelineConfig(layers=[])
        engine = PipelineEngine(
            config,
            reducer_deps={"llm": "mock_llm"},
            provider_deps={"embedding_provider": "mock_embedder"},
        )
        assert engine._reducer_registry._deps == {"llm": "mock_llm"}
        assert engine._provider_registry._deps == {"embedding_provider": "mock_embedder"}
