"""OpenAI Chat Completions adapter stub for future implementation."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """Stub adapter for OpenAI Chat Completions API.

    Not yet implemented — exists to reserve the interface shape.
    """

    def extract_messages(self, body: dict) -> tuple[str | None, list[dict]]:
        raise NotImplementedError("OpenAI adapter not yet implemented")

    def rebuild_body(
        self,
        original_body: dict,
        optimized_messages: list[dict],
        system_injection: str | None,
    ) -> dict:
        raise NotImplementedError("OpenAI adapter not yet implemented")

    def parse_sse_text(self, chunk: bytes) -> str | None:
        raise NotImplementedError("OpenAI adapter not yet implemented")
