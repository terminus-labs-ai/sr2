"""Tests for the EmbeddingProvider protocol."""

import asyncio

import pytest

from sr2.protocols import EmbeddingProvider


class TestEmbeddingProviderProtocol:
    """Verify EmbeddingProvider is runtime-checkable and well-formed."""

    def test_is_runtime_checkable(self):
        """EmbeddingProvider should be runtime_checkable."""

        class FakeEmbedder:
            @property
            def dimensions(self) -> int:
                return 384

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 384 for _ in texts]

        embedder = FakeEmbedder()
        assert isinstance(embedder, EmbeddingProvider)

    def test_missing_embed_not_satisfied(self):
        """A class without embed() should NOT satisfy the protocol."""

        class NotAnEmbedder:
            @property
            def dimensions(self) -> int:
                return 384

        obj = NotAnEmbedder()
        assert not isinstance(obj, EmbeddingProvider)

    def test_missing_dimensions_not_satisfied(self):
        """A class without dimensions property should NOT satisfy the protocol."""

        class NotAnEmbedder:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 384 for _ in texts]

        obj = NotAnEmbedder()
        assert not isinstance(obj, EmbeddingProvider)

    def test_functional_call(self):
        """A mock EmbeddingProvider can be called and returns embeddings."""

        class MockEmbedder:
            @property
            def dimensions(self) -> int:
                return 3

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 2.0, 3.0] for _ in texts]

        embedder = MockEmbedder()
        result = asyncio.run(embedder.embed(["hello", "world"]))

        assert len(result) == 2
        assert all(len(vec) == embedder.dimensions for vec in result)
        assert result[0] == [1.0, 2.0, 3.0]
