"""Token counting implementations."""

from __future__ import annotations

import json

from sr2.models import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

CHARS_PER_TOKEN = 4


class CharacterTokenCounter:
    """Estimates tokens as chars // 4. Zero-dependency fallback."""

    def count(self, content: list[ContentBlock]) -> int:
        total_chars = 0
        for block in content:
            if isinstance(block, (TextBlock, ThinkingBlock)):
                total_chars += len(block.text)
            elif isinstance(block, ToolUseBlock):
                total_chars += len(block.name) + len(json.dumps(block.input))
            elif isinstance(block, ToolResultBlock):
                if isinstance(block.content, str):
                    total_chars += len(block.content)
                else:
                    # list[TextBlock]
                    for tb in block.content:
                        total_chars += len(tb.text)
        return total_chars // CHARS_PER_TOKEN
