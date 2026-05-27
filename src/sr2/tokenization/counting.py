"""Tiktoken-based token counting and truncation.

Primary implementation uses tiktoken (cl100k_base encoding).
Import from sr2.tokenization.counting for the public API.
"""

from __future__ import annotations

import json

import tiktoken

from sr2.models import ContentBlock, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock

_ENCODING_NAME = "cl100k_base"
_enc = tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base encoding.

    Returns 0 for an empty string.
    """
    if not text:
        return 0
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, budget: int) -> str:
    """Return the longest prefix of *text* whose token count <= *budget*.

    - budget == 0  → ""
    - text == ""   → ""
    - text already fits → text unchanged
    """
    if budget <= 0 or not text:
        return ""

    tokens = _enc.encode(text)
    if len(tokens) <= budget:
        return text

    # Decode only the first `budget` tokens to get the prefix.
    truncated_tokens = tokens[:budget]
    return _enc.decode(truncated_tokens)


def _extract_text(content: list[ContentBlock]) -> str:
    """Concatenate all text from a list of content blocks."""
    parts: list[str] = []
    for block in content:
        if isinstance(block, (TextBlock, ThinkingBlock)):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(block.name)
            parts.append(json.dumps(block.input))
        elif isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                parts.append(block.content)
            else:
                for tb in block.content:
                    parts.append(tb.text)
    return " ".join(parts)


class TiktokenTokenCounter:
    """TokenCounter implementation backed by tiktoken (cl100k_base).

    Counts tokens across a list of ContentBlock objects by extracting
    all text and counting with the tiktoken encoder.
    """

    def count(self, content: list[ContentBlock]) -> int:
        """Return total token count for all blocks."""
        if not content:
            return 0
        text = _extract_text(content)
        return count_tokens(text)
