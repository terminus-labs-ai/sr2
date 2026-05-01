"""Token counting utilities.

Provides a consistent interface for estimating token counts.
Uses tiktoken for accuracy but falls back to character estimation
if the model is unknown.

Design principles:
- DRY: Single source of truth for token counting.
- SRP: Only handles counting — doesn't know about layers or pipeline.
"""

from __future__ import annotations

import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a text string using tiktoken.

    Args:
        text: The text to count.
        model: The model to use for encoding (affects tokenizer).

    Returns:
        Estimated token count.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within a token budget.

    Args:
        text: The text to truncate.
        max_tokens: Maximum token count.
        model: The model to use for encoding.

    Returns:
        Truncated text that fits within the budget.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    encoded = encoding.encode(text)
    if len(encoded) <= max_tokens:
        return text

    truncated = encoding.decode(encoded[:max_tokens])
    return truncated + "\n... [truncated]"
