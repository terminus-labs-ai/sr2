"""Tests for tokenization/counting.py — tiktoken-based count_tokens and truncate_to_tokens.

Covers:
  - count_tokens(text: str) -> int: non-negative integer for any input
  - count_tokens("") == 0: empty string yields zero
  - truncate_to_tokens(text, budget) -> str: result token count fits within budget
  - truncate_to_tokens with budget larger than content returns content unchanged
  - truncate_to_tokens with budget=0 returns empty string
  - CharacterTokenCounter.count regression: still works against list[ContentBlock]
  - TiktokenTokenCounter implements TokenCounter protocol (if tiktoken available)

Note on tokenizer choice: tests are written against the public interface only.
The specific encoding (cl100k, o200k, etc.) is an implementation detail for
Agent C. Tests use count_tokens round-trip invariants rather than hardcoded
token counts to remain encoding-agnostic.
"""

from __future__ import annotations

import pytest

from sr2.models import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.protocols import TokenCounter
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def require_tiktoken():
    """Skip any test that uses this fixture if tiktoken is not installed."""
    pytest.importorskip("tiktoken", reason="tiktoken not installed")


# ---------------------------------------------------------------------------
# 1. count_tokens — module-level function
#    These tests require tiktoken because count_tokens uses it internally.
# ---------------------------------------------------------------------------


class TestCountTokens:
    """count_tokens(text: str) -> int"""

    @pytest.fixture(autouse=True)
    def _require_tiktoken(self, require_tiktoken):
        pass

    def test_returns_int(self):
        from sr2.tokenization.counting import count_tokens

        result = count_tokens("hello world")
        assert isinstance(result, int)

    def test_non_negative(self):
        from sr2.tokenization.counting import count_tokens

        assert count_tokens("hello world") >= 0

    def test_empty_string_returns_zero(self):
        from sr2.tokenization.counting import count_tokens

        assert count_tokens("") == 0

    def test_longer_text_has_more_tokens_than_shorter(self):
        """More content → at least as many tokens (monotonicity)."""
        from sr2.tokenization.counting import count_tokens

        short = count_tokens("Hi")
        long = count_tokens("Hi, how are you doing today? I hope everything is going well.")
        assert long > short

    def test_whitespace_only_string(self):
        from sr2.tokenization.counting import count_tokens

        result = count_tokens("   ")
        assert isinstance(result, int)
        assert result >= 0

    def test_unicode_text(self):
        from sr2.tokenization.counting import count_tokens

        result = count_tokens("こんにちは世界")
        assert isinstance(result, int)
        assert result > 0

    def test_multiline_text(self):
        from sr2.tokenization.counting import count_tokens

        text = "Line one.\nLine two.\nLine three."
        result = count_tokens(text)
        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# 2. truncate_to_tokens — module-level function
#    These tests require tiktoken because they use count_tokens for assertions.
# ---------------------------------------------------------------------------


class TestTruncateToTokens:
    """truncate_to_tokens(text: str, budget: int) -> str"""

    @pytest.fixture(autouse=True)
    def _require_tiktoken(self, require_tiktoken):
        pass

    def test_returns_string(self):
        from sr2.tokenization.counting import truncate_to_tokens

        result = truncate_to_tokens("hello world", 10)
        assert isinstance(result, str)

    def test_budget_zero_returns_empty_string(self):
        from sr2.tokenization.counting import truncate_to_tokens

        result = truncate_to_tokens("some content here", 0)
        assert result == ""

    def test_result_fits_within_budget(self):
        """Token count of the result must not exceed the budget."""
        from sr2.tokenization.counting import count_tokens, truncate_to_tokens

        text = "The quick brown fox jumps over the lazy dog. " * 20
        budget = 10
        result = truncate_to_tokens(text, budget)
        assert count_tokens(result) <= budget

    def test_budget_larger_than_content_returns_full_content(self):
        """When the text already fits in the budget, return it unchanged."""
        from sr2.tokenization.counting import count_tokens, truncate_to_tokens

        text = "Short text."
        tokens = count_tokens(text)
        # Budget is guaranteed larger than content size.
        result = truncate_to_tokens(text, tokens + 100)
        assert result == text

    def test_exact_budget_returns_full_content(self):
        """When budget exactly equals the token count, return unchanged."""
        from sr2.tokenization.counting import count_tokens, truncate_to_tokens

        text = "Exactly fitting text."
        tokens = count_tokens(text)
        result = truncate_to_tokens(text, tokens)
        assert result == text

    def test_truncation_is_prefix_of_original(self):
        """Truncated result must be a prefix of the original text."""
        from sr2.tokenization.counting import truncate_to_tokens

        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = truncate_to_tokens(text, 5)
        assert text.startswith(result)

    def test_truncation_nonempty_when_budget_positive_and_text_nonempty(self):
        """A positive budget against a non-empty string should yield something."""
        from sr2.tokenization.counting import truncate_to_tokens

        result = truncate_to_tokens("hello", 1)
        assert len(result) > 0

    def test_empty_text_always_returns_empty(self):
        from sr2.tokenization.counting import truncate_to_tokens

        result = truncate_to_tokens("", 50)
        assert result == ""


# ---------------------------------------------------------------------------
# 3. CharacterTokenCounter regression
#    Does NOT depend on tiktoken — tests the existing char-based counter.
# ---------------------------------------------------------------------------


class TestCharacterTokenCounterRegression:
    """CharacterTokenCounter.count(list[ContentBlock]) still works."""

    def test_counts_text_blocks(self):
        counter = CharacterTokenCounter()
        blocks = [TextBlock(text="a" * 8)]
        assert counter.count(blocks) == 2  # 8 chars // 4

    def test_empty_content_list(self):
        counter = CharacterTokenCounter()
        assert counter.count([]) == 0

    def test_counts_thinking_blocks(self):
        counter = CharacterTokenCounter()
        blocks = [ThinkingBlock(text="a" * 12)]
        assert counter.count(blocks) == 3  # 12 // 4

    def test_counts_tool_result_string(self):
        counter = CharacterTokenCounter()
        blocks = [ToolResultBlock(tool_use_id="x", content="a" * 16)]
        assert counter.count(blocks) == 4  # 16 // 4

    def test_counts_tool_result_text_block_list(self):
        counter = CharacterTokenCounter()
        blocks = [
            ToolResultBlock(
                tool_use_id="x",
                content=[TextBlock(text="a" * 8), TextBlock(text="a" * 8)],
            )
        ]
        assert counter.count(blocks) == 4  # 16 // 4

    def test_implements_token_counter_protocol(self):
        counter = CharacterTokenCounter()
        assert isinstance(counter, TokenCounter)


# ---------------------------------------------------------------------------
# 4. TiktokenTokenCounter — protocol compliance and behavior
# ---------------------------------------------------------------------------


class TestTiktokenTokenCounter:
    """TiktokenTokenCounter implements TokenCounter and wraps count_tokens."""

    @pytest.fixture(autouse=True)
    def _require_tiktoken(self, require_tiktoken):
        pass

    def test_class_exists(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        assert TiktokenTokenCounter is not None

    def test_implements_token_counter_protocol(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_count_returns_int(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        blocks = [TextBlock(text="hello world")]
        result = counter.count(blocks)
        assert isinstance(result, int)
        assert result >= 0

    def test_count_empty_blocks_returns_zero(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        assert counter.count([]) == 0

    def test_count_multiple_text_blocks(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        blocks = [TextBlock(text="hello"), TextBlock(text=" world")]
        result = counter.count(blocks)
        assert result > 0

    def test_count_thinking_block(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        blocks = [ThinkingBlock(text="reasoning step")]
        result = counter.count(blocks)
        assert result > 0

    def test_count_tool_result_string(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        blocks = [ToolResultBlock(tool_use_id="x", content="tool output")]
        result = counter.count(blocks)
        assert result > 0

    def test_count_tool_result_text_block_list(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        blocks = [
            ToolResultBlock(
                tool_use_id="x",
                content=[TextBlock(text="part one"), TextBlock(text="part two")],
            )
        ]
        result = counter.count(blocks)
        assert result > 0

    def test_more_content_yields_more_tokens_than_less(self):
        from sr2.tokenization.counting import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        short = counter.count([TextBlock(text="Hi")])
        long = counter.count([TextBlock(text="Hi, how are you doing today? I hope you're well.")])
        assert long > short
