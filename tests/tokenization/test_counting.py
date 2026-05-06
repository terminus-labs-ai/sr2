"""Tests for TokenCounter protocol and implementations.

Verifies:
- Protocol is runtime_checkable and structurally typed
- CharacterCounter provides dependency-free token estimation
- TiktokenCounter wraps tiktoken for accurate counting
"""

from __future__ import annotations

import pytest

from sr2.protocols.tokenization import TokenCounter
from sr2.tokenization.counting import CharacterCounter, TiktokenCounter


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


class TestTokenCounterProtocol:
    """Verify TokenCounter is runtime_checkable and structurally typed."""

    def test_protocol_is_runtime_checkable(self):
        """TokenCounter should be decorated with @runtime_checkable."""

        class Conforming:
            def count(self, text: str) -> int:
                return 0

            def truncate(self, text: str, max_tokens: int) -> str:
                return text

        assert isinstance(Conforming(), TokenCounter)

    def test_missing_count_does_not_satisfy(self):
        """A class missing `count` must NOT satisfy the protocol."""

        class MissingCount:
            def truncate(self, text: str, max_tokens: int) -> str:
                return text

        assert not isinstance(MissingCount(), TokenCounter)

    def test_missing_truncate_does_not_satisfy(self):
        """A class missing `truncate` must NOT satisfy the protocol."""

        class MissingTruncate:
            def count(self, text: str) -> int:
                return 0

        assert not isinstance(MissingTruncate(), TokenCounter)


# ---------------------------------------------------------------------------
# CharacterCounter
# ---------------------------------------------------------------------------


class TestCharacterCounter:
    """CharacterCounter: 4 chars ~= 1 token, no external deps."""

    def test_count_hello_world(self):
        result = CharacterCounter().count("hello world")
        assert result == len("hello world") // 4

    def test_count_empty_string(self):
        assert CharacterCounter().count("") == 0

    def test_truncate_over_budget(self):
        text = "hello world test text"
        result = CharacterCounter().truncate(text, 2)
        # 2 tokens * 4 chars = 8 chars
        assert result == text[:8]

    def test_truncate_under_budget(self):
        result = CharacterCounter().truncate("hi", 100)
        assert result == "hi"

    def test_satisfies_protocol(self):
        assert isinstance(CharacterCounter(), TokenCounter)


# ---------------------------------------------------------------------------
# TiktokenCounter (guarded — requires tiktoken)
# ---------------------------------------------------------------------------


class TestTiktokenCounter:
    """TiktokenCounter: wraps tiktoken for accurate token counting."""

    def test_satisfies_protocol(self):
        assert isinstance(TiktokenCounter(), TokenCounter)

    def test_count_positive_integer(self):
        result = TiktokenCounter().count("hello")
        assert isinstance(result, int)
        assert result > 0

    def test_truncate_shortens_text(self):
        text = "hello world this is a test"
        result = TiktokenCounter().truncate(text, 3)
        # Result includes "\n... [truncated]" suffix, so check the
        # core content (before suffix) is shorter than the original.
        core = result.replace("\n... [truncated]", "")
        assert len(core) < len(text)

    def test_truncate_stays_under_budget(self):
        text = "hello world this is a test with many tokens"
        counter = TiktokenCounter()
        truncated = counter.truncate(text, 3)
        # The truncated content (minus the suffix marker) should decode
        # to at most 3 tokens. We just verify the counter agrees.
        # The truncated text includes "... [truncated]" suffix, so strip it
        # and check the core content.
        core = truncated.replace("\n... [truncated]", "")
        assert counter.count(core) <= 3

    def test_truncate_under_budget_unchanged(self):
        text = "hi"
        result = TiktokenCounter().truncate(text, 100)
        assert result == text
