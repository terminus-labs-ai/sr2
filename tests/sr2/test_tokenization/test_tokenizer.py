"""Tests for tokenization implementations."""

import pytest

from sr2.tokenization.tokenizer import CharacterTokenizer, TiktokenTokenizer


class TestCharacterTokenizer:
    """Tests for CharacterTokenizer."""

    def test_count_tokens_basic(self) -> None:
        """Test basic token counting."""
        tokenizer = CharacterTokenizer()
        # 4 characters = 1 token
        assert tokenizer.count_tokens("hello") == 1  # 5 chars / 4 = 1
        assert tokenizer.count_tokens("hello world") == 2  # 11 chars / 4 = 2

    def test_count_tokens_empty(self) -> None:
        """Test empty string."""
        tokenizer = CharacterTokenizer()
        assert tokenizer.count_tokens("") == 1  # Returns minimum of 1

    def test_count_tokens_short(self) -> None:
        """Test short strings."""
        tokenizer = CharacterTokenizer()
        assert tokenizer.count_tokens("hi") == 1  # 2 chars / 4 = 0, min 1

    def test_count_tokens_long(self) -> None:
        """Test long strings."""
        tokenizer = CharacterTokenizer()
        long_text = "a" * 1000
        assert tokenizer.count_tokens(long_text) == 250  # 1000 / 4

    def test_name(self) -> None:
        """Test tokenizer name."""
        tokenizer = CharacterTokenizer()
        assert tokenizer.name() == "character"


class TestTiktokenTokenizer:
    """Tests for TiktokenTokenizer (requires tiktoken)."""

    @pytest.mark.skipif(
        True,  # Skip by default since tiktoken is optional
        reason="tiktoken not required for basic tests",
    )
    def test_tiktoken_available(self) -> None:
        """Test tiktoken initialization if available."""
        try:
            tokenizer = TiktokenTokenizer()
            assert tokenizer.name() == "tiktoken_cl100k_base"
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_tiktoken_import_error(self) -> None:
        """Test error when tiktoken is not installed."""
        # We can't really test this without uninstalling tiktoken
        # Just verify the error message is informative
        try:
            TiktokenTokenizer()
            # If we get here, tiktoken is installed, which is fine
        except ImportError as e:
            assert "tiktoken" in str(e)

    def test_character_fallback(self) -> None:
        """Test that CharacterTokenizer is good fallback."""
        char_tokenizer = CharacterTokenizer()
        # Should give reasonable estimates
        text = "The quick brown fox jumps over the lazy dog"
        tokens = char_tokenizer.count_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4
