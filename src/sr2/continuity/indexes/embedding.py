"""Semantic retrieval with a rebuildable SQLite cache of endpoint embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import litellm

from sr2.config.models import EmbeddingIndexConfig
from sr2.continuity.index import (
    ContinuityCandidate,
    ContinuityFailure,
    ContinuityHealth,
)
from sr2.continuity.indexes.structured import StructuredIndex
from sr2.continuity.models import ContinuityKind, ContinuityRecord
from sr2.continuity.policy import RetrievalContext


class EmbeddingIndexError(RuntimeError):
    """An embedding response cannot be used for continuity retrieval."""


class EmbeddingContinuityIndex:
    """Blend endpoint cosine similarity with the existing structured scorer.

    Rebuild from policy-eligible ledger records after reopening. Only vectors
    persist here; record contents and currentness remain owned by the ledger.
    """

    def __init__(
        self,
        path: str | Path,
        model: str,
        base_url: str,
        dimensions: int | None = None,
    ) -> None:
        self._path = path
        self._settings = EmbeddingIndexConfig(
            model=model, base_url=base_url, dimensions=dimensions
        )
        self._settings_hash = hashlib.sha256(
            self._settings.model_dump_json().encode()
        ).hexdigest()
        self._db: aiosqlite.Connection | None = None
        self._structured = StructuredIndex()
        self._vectors: dict[str, list[float]] = {}
        self._failure: ContinuityFailure | None = None

    async def connect(self) -> None:
        """Open the vector cache, retaining entries across ledger replays."""
        if self._db is not None:
            return
        db = await aiosqlite.connect(self._path)
        try:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS continuity_embeddings (
                    record_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    settings_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (record_id, content_hash, settings_hash)
                )"""
            )
            await db.commit()
        except BaseException:
            await db.close()
            raise
        self._db = db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def append(self, records: list[ContinuityRecord]) -> None:
        vectors = await self._cached_vectors(records)
        await self._structured.append(records)
        self._vectors.update(vectors)

    async def rebuild(self, records: AsyncIterator[ContinuityRecord]) -> None:
        replay = [record async for record in records]
        vectors = await self._cached_vectors(replay)
        structured = StructuredIndex()
        await structured.append(replay)
        self._structured = structured
        self._vectors = vectors

    async def search(
        self, context: RetrievalContext, limit: int
    ) -> list[ContinuityCandidate]:
        if not self._vectors or limit <= 0:
            return []
        query = (await self._embed([context.query]))[0]
        candidates = await self._structured.search(context, limit=len(self._vectors))
        results = []
        for candidate in candidates:
            vector = self._vectors[candidate.record.current_record.id]
            query_norm, vector_norm = math.hypot(*query), math.hypot(*vector)
            cosine = (
                math.fsum(
                    (q / query_norm) * (v / vector_norm)
                    for q, v in zip(query, vector, strict=True)
                )
                if query_norm and vector_norm
                else 0.0
            )
            results.append(
                candidate.model_copy(
                    update={
                        "score": 0.6 * cosine + 0.4 * candidate.score,
                        "score_components": {
                            **candidate.score_components,
                            "vector": cosine,
                        },
                    }
                )
            )
        results.sort(key=lambda hit: (-hit.score, hit.record.current_record.id))
        return results[:limit]

    async def health(self) -> ContinuityHealth:
        if self._failure is not None:
            return ContinuityHealth(status="degraded", failures=(self._failure,))
        return ContinuityHealth(status="healthy")

    async def _cached_vectors(
        self, records: list[ContinuityRecord]
    ) -> dict[str, list[float]]:
        await self.connect()
        assert self._db is not None
        vectors = {}
        pending = []
        for record in records:
            text = record.content
            if record.kind == ContinuityKind.EPISODE:
                text = f"{record.meta.get('title', '')}\n{record.content}"
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            async with self._db.execute(
                """SELECT vector FROM continuity_embeddings
                   WHERE record_id = ? AND content_hash = ? AND settings_hash = ?""",
                (record.id, content_hash, self._settings_hash),
            ) as cursor:
                cached = await cursor.fetchone()
            if cached is not None:
                vectors[record.id] = json.loads(cached[0])
            else:
                pending.append((record.id, content_hash, text))
        if pending:
            embedded = await self._embed(
                [text for _, _, text in pending],
                cached_size=len(next(iter(vectors.values()))) if vectors else None,
            )
            try:
                await self._db.executemany(
                    """INSERT OR REPLACE INTO continuity_embeddings
                       (record_id, content_hash, settings_hash, vector)
                       VALUES (?, ?, ?, ?)""",
                    [
                        (record_id, content_hash, self._settings_hash, json.dumps(vector))
                        for (record_id, content_hash, _), vector in zip(pending, embedded)
                    ],
                )
                await self._db.commit()
            except BaseException:
                await self._db.rollback()
                raise
            vectors.update(
                (record_id, vector)
                for (record_id, _, _), vector in zip(pending, embedded)
            )
        return vectors

    async def _embed(
        self, texts: list[str], *, cached_size: int | None = None
    ) -> list[list[float]]:
        try:
            options = {}
            if self._settings.dimensions is not None:
                options["dimensions"] = self._settings.dimensions
            response = await litellm.aembedding(
                model=self._settings.model,
                custom_llm_provider="openai",
                api_base=self._settings.base_url,
                api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
                input=texts,
                allowed_openai_params=["dimensions"],
                max_retries=0,
                **options,
            )
            data = sorted(response.data, key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(len(texts))):
                raise EmbeddingIndexError(
                    "Endpoint returned missing or duplicate embeddings; retry the request."
                )
            vectors = [item["embedding"] for item in data]
            expected_size = self._settings.dimensions
            if expected_size is None and self._vectors:
                expected_size = len(next(iter(self._vectors.values())))
            for vector in vectors:
                if (
                    not isinstance(vector, list)
                    or not vector
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        for value in vector
                    )
                    or (expected_size is not None and len(vector) != expected_size)
                    or (cached_size is not None and len(vector) != cached_size)
                ):
                    raise EmbeddingIndexError(
                        "Endpoint returned invalid vectors; check the model and dimensions."
                    )
                expected_size = len(vector)
        except Exception:
            self._failure = ContinuityFailure(
                capability="index",
                message="Embedding request failed; check the endpoint and index settings, then retry.",
                retryable=True,
            )
            raise
        self._failure = None
        return vectors
