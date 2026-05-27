"""Compaction rules for sr2.compaction.

Five rules:
  - schema_and_sample: compact JSON schema+data to schema + 1 sample row
  - ReferenceRule:     replace repeated content blocks with reference pointers
  - result_summary:    truncate long tool results to a token budget
  - supersede:         drop messages whose turn_index is declared superseded
  - collapse:          collapse repeated consecutive turn patterns
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sr2.models import ContentBlock, Message, TextBlock, ToolResultBlock
from sr2.tokenization.counting import count_tokens


# ---------------------------------------------------------------------------
# 1. schema_and_sample
# ---------------------------------------------------------------------------


def schema_and_sample(block: ContentBlock) -> ToolResultBlock | None:
    """Compact a ToolResultBlock containing a schema+data JSON payload.

    Returns a new ToolResultBlock with only the first data row retained,
    or None if the rule does not apply (not JSON, no schema/data, <=1 row).
    """
    if not isinstance(block, ToolResultBlock):
        return None

    content = block.content if isinstance(block.content, str) else _join_text_blocks(block.content)

    try:
        parsed: dict[str, Any] = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    if "schema" not in parsed or "data" not in parsed:
        return None

    data = parsed["data"]
    if not isinstance(data, list) or len(data) <= 1:
        return None

    compacted = {
        "schema": parsed["schema"],
        "data": [data[0]],
    }
    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        content=json.dumps(compacted),
        is_error=block.is_error,
        compacted=True,
    )


def _join_text_blocks(blocks: list[TextBlock]) -> str:
    return " ".join(b.text for b in blocks)


# ---------------------------------------------------------------------------
# 2. ReferenceRule
# ---------------------------------------------------------------------------


class ReferenceRule:
    """Replace repeated TextBlocks with a reference pointer.

    Parameters
    ----------
    seen_registry:
        Mutable dict mapping content_hash -> reference_id. Pass an empty
        dict to start fresh; share the same dict across calls to accumulate
        state across multiple messages.
    """

    def __init__(self, seen_registry: dict[str, str]) -> None:
        self._seen = seen_registry

    def apply(self, block: ContentBlock) -> TextBlock | None:
        """Return a reference TextBlock if content was seen before, else None.

        First occurrence: registers the content hash and returns None.
        Subsequent occurrences: returns a TextBlock containing the reference.
        """
        if not isinstance(block, TextBlock):
            return None

        content_hash = hashlib.sha256(block.text.encode()).hexdigest()[:16]

        if content_hash not in self._seen:
            # First occurrence — register and do not replace
            ref_id = f"ref:{content_hash}"
            self._seen[content_hash] = ref_id
            return None

        # Already seen — replace with reference pointer
        ref_id = self._seen[content_hash]
        return TextBlock(text=f"[{ref_id}]")


# ---------------------------------------------------------------------------
# 3. result_summary
# ---------------------------------------------------------------------------


def result_summary(block: ContentBlock, max_tokens: int = 500) -> ToolResultBlock | None:
    """Truncate a long ToolResultBlock to max_tokens.

    Returns a new (compacted) ToolResultBlock if the content exceeds the
    threshold, or None if the rule does not apply.

    Error results are never summarized.
    """
    if not isinstance(block, ToolResultBlock):
        return None

    if block.is_error:
        return None

    content = block.content if isinstance(block.content, str) else _join_text_blocks(block.content)
    token_count = count_tokens(content)

    if token_count <= max_tokens:
        return None

    # Truncate by character estimate (approximate) — keep proportional prefix
    # Use the ratio of max_tokens to actual tokens to estimate character budget
    ratio = max_tokens / token_count if token_count > 0 else 1.0
    char_budget = max(1, int(len(content) * ratio))
    truncated = content[:char_budget]

    # Append truncation marker
    summary = truncated + f" ... [truncated: {token_count} tokens → {max_tokens}]"

    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        content=summary,
        is_error=False,
        compacted=True,
    )


# ---------------------------------------------------------------------------
# 4. supersede
# ---------------------------------------------------------------------------

_SUPERSEDES_RE = re.compile(r"\[supersedes turn (\d+)\]", re.IGNORECASE)


def supersede(messages: list[Message]) -> list[Message]:
    """Remove messages whose turn_index has been declared superseded.

    A message declares supersession by including the pattern
    ``[supersedes turn N]`` in any of its TextBlock content.

    Returns a new list with superseded messages removed.
    """
    if not messages:
        return []

    # Collect all superseded turn indices
    superseded_indices: set[int] = set()
    for msg in messages:
        for block in msg.content:
            if isinstance(block, TextBlock):
                for match in _SUPERSEDES_RE.finditer(block.text):
                    superseded_indices.add(int(match.group(1)))

    return [m for m in messages if m.turn_index not in superseded_indices]


# ---------------------------------------------------------------------------
# 5. collapse
# ---------------------------------------------------------------------------


def collapse(messages: list[Message], min_occurrences: int = 3) -> list[Message]:
    """Collapse consecutive repeated turn-patterns into a single representative.

    A "pattern" is a repeating sequence of messages (window of 1..N messages).
    When the same window repeats >= min_occurrences times consecutively, only
    the first repetition is kept.

    Strategy: try pattern lengths from 1 up to len(messages)//min_occurrences.
    For each starting position, find the longest repeating pattern and collapse
    it if it meets the threshold.

    Unique messages are always preserved.
    """
    if not messages:
        return []

    def _fingerprint(msg: Message) -> str:
        parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append(f"{msg.role}:{block.text}")
        return "|".join(parts)

    fps = [_fingerprint(m) for m in messages]
    n = len(messages)

    # Try to find repeating windows starting at each position
    # Returns (pattern_len, count) of best repeating pattern at position i
    def _find_repeat(start: int) -> tuple[int, int]:
        best_len = 1
        best_count = 1
        max_pat_len = (n - start) // min_occurrences
        for pat_len in range(1, max_pat_len + 1):
            pattern = fps[start : start + pat_len]
            count = 1
            pos = start + pat_len
            while pos + pat_len <= n and fps[pos : pos + pat_len] == pattern:
                count += 1
                pos += pat_len
            if count >= min_occurrences and pat_len >= best_len:
                best_len = pat_len
                best_count = count
        return best_len, best_count

    result: list[Message] = []
    i = 0

    while i < n:
        pat_len, count = _find_repeat(i)
        if count >= min_occurrences:
            # Keep one copy of the pattern, skip the rest
            result.extend(messages[i : i + pat_len])
            i += pat_len * count
        else:
            result.append(messages[i])
            i += 1

    return result
