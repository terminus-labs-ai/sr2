"""Token counting implementations: character heuristic and tiktoken-based."""

import logging

from typing import Protocol


class Tokenizer(Protocol):
    """Protocol for token counting implementations."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        ...

    def name(self) -> str:
        """Return the name of this tokenizer."""
        ...


class CharacterTokenizer:
    """Simple character heuristic tokenizer: 4 characters ≈ 1 token.

    This is the default fallback. Useful for quick estimates without
    external dependencies.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens using character heuristic."""
        return max(1, len(text) // 4)

    def name(self) -> str:
        """Return tokenizer name."""
        return "character"


logger = logging.getLogger(__name__)


class TiktokenTokenizer:
    """Tiktoken-based tokenizer for accurate token counting.

    Uses OpenAI's tiktoken library for accurate token counting with
    specific encoding schemes (e.g., cl100k_base for GPT-3.5/4).

    Requires: pip install tiktoken
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialize with a specific encoding.

        Args:
            encoding_name: Tiktoken encoding name (default: cl100k_base for GPT models)

        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If encoding_name is not valid
        """
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is not installed. Install it with: pip install tiktoken"
            ) from e

        self._encoding_name = encoding_name
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        try:
            tokens = self._encoder.encode(text, disallowed_special=())
            return len(tokens)
        except Exception:
            logger.error("tiktoken encoding failed, falling back to character heuristic", exc_info=True)
            return max(1, len(text) // 4)

    def name(self) -> str:
        """Return tokenizer name."""
        return f"tiktoken_{self._encoding_name}"
