"""Bridge LLM callable factory — creates async callables for SR2 components.

Handles auth resolution: dedicated api_key > piggybacked proxied key.
"""

from __future__ import annotations

import logging

from sr2_bridge.config import BridgeLLMModelConfig

logger = logging.getLogger(__name__)


class APIKeyCache:
    """Caches the API key extracted from proxied request headers.

    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self):
        self._key: str | None = None

    def update(self, headers: dict[str, str]) -> None:
        """Extract and cache API key from request headers."""
        # Anthropic uses x-api-key, OpenAI uses Authorization: Bearer ...
        key = headers.get("x-api-key")
        if not key:
            auth = headers.get("authorization", "")
            if auth.startswith("Bearer "):
                key = auth[7:]
        if key and key != self._key:
            self._key = key
            logger.debug("API key cached from proxied request")

    @property
    def key(self) -> str | None:
        return self._key


def make_summarization_callable(
    config: BridgeLLMModelConfig,
    key_cache: APIKeyCache,
    upstream_url: str,
):
    """Create an async callable matching SummarizationEngine's signature.

    Signature: async (system: str, prompt: str) -> str
    """

    async def summarization_call(system: str, prompt: str) -> str:
        import litellm

        api_key = config.api_key or key_cache.key
        if not api_key:
            raise RuntimeError("No API key available for summarization")

        api_base = config.api_base or upstream_url

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        response = await litellm.acompletion(
            model=config.model,
            messages=messages,
            max_tokens=config.max_tokens,
            api_key=api_key,
            api_base=api_base,
        )
        return response.choices[0].message.content or ""

    return summarization_call


def make_extraction_callable(
    config: BridgeLLMModelConfig,
    key_cache: APIKeyCache,
    upstream_url: str,
):
    """Create an async callable matching MemoryExtractor's signature.

    Signature: async (prompt: str) -> str
    """

    async def extraction_call(prompt: str) -> str:
        import litellm

        api_key = config.api_key or key_cache.key
        if not api_key:
            raise RuntimeError("No API key available for memory extraction")

        api_base = config.api_base or upstream_url

        response = await litellm.acompletion(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            api_key=api_key,
            api_base=api_base,
        )
        return response.choices[0].message.content or ""

    return extraction_call


def make_embedding_callable(
    config: BridgeLLMModelConfig,
    key_cache: APIKeyCache,
    upstream_url: str,
):
    """Create an async callable for embeddings.

    Signature: async (text: str) -> list[float]
    """

    async def embedding_call(text: str) -> list[float]:
        import litellm

        api_key = config.api_key or key_cache.key
        if not api_key:
            raise RuntimeError("No API key available for embeddings")

        api_base = config.api_base or upstream_url

        response = await litellm.aembedding(
            model=config.model,
            input=[text],
            api_key=api_key,
            api_base=api_base,
        )
        return response.data[0]["embedding"]

    return embedding_call
