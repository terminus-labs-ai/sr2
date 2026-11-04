"""AC26: public embedding-index contract, cache, configuration and scoring."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import socket
import sqlite3
from contextlib import asynccontextmanager, closing
from dataclasses import replace
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel

from sr2.config.models import ContinuityConfig
from sr2.continuity.index import ContinuityHealth
from sr2.continuity.indexes.structured import StructuredIndex
from sr2.continuity.models import ContinuityKind, ContinuityRecord
from tests.continuity import test_vector_hybrid_index as shared_contract
from tests.continuity._fixtures_index import (
    CONTEXT,
    NOW,
    make_record,
    record_ids,
    records_iter,
)


@pytest.fixture
def endpoint(monkeypatch):
    state = SimpleNamespace(requests=[], texts=[], vectors={}, status=200)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inputs = body["input"]
            texts = [inputs] if isinstance(inputs, str) else inputs
            state.requests.append(
                {"path": self.path, "body": body, "status": state.status}
            )
            state.texts.extend(texts)
            if state.status == 500:
                response = {
                    "error": {
                        "message": "embedding service unavailable",
                        "type": "server_error",
                    }
                }
            else:
                response = {
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": state.vectors.get(text, [1.0, 0.0, 0.0]),
                        }
                        for i, text in enumerate(texts)
                    ],
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
            payload = json.dumps(response).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        state.base_url = f"http://127.0.0.1:{server.server_port}/configured/v1"
        connect = socket.socket.connect

        def fake_endpoint_only(sock, address):
            assert address == server.server_address, (
                f"unexpected network destination: {address}"
            )
            return connect(sock, address)

        monkeypatch.setattr(socket.socket, "connect", fake_endpoint_only)
        monkeypatch.setenv("OPENAI_API_KEY", "unit-test-only")
        thread = Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _embedding_module():
    name = "sr2.continuity.indexes.embedding"
    assert importlib.util.find_spec(name) is not None, (
        "EmbeddingContinuityIndex is not implemented"
    )
    module = importlib.import_module(name)
    assert hasattr(module, "EmbeddingContinuityIndex"), (
        "EmbeddingContinuityIndex is not exported"
    )
    return module


async def _lifecycle(index, *names):
    for name in names:
        method = getattr(index, name, None)
        if method is not None:
            result = method()
            if inspect.isawaitable(result):
                await result
            return


@pytest.fixture
def new_index(tmp_path, endpoint):
    @asynccontextmanager
    async def opened(*, path=None, settings=None):
        if settings is None:
            settings = {"model": "unit-embedding-model", "base_url": endpoint.base_url}
        cls = _embedding_module().EmbeddingContinuityIndex
        index = cls(path=path or tmp_path / "vectors.sqlite", **settings)
        try:
            await _lifecycle(index, "connect")
            yield index
        finally:
            await _lifecycle(index, "aclose", "close")

    return opened


_SHARED_CHECKS = (
    "test_ac17_r2_vector_hybrid_index_satisfies_continuity_index_protocol",
    "test_ac17_r4_append_then_search_returns_indexed_records",
    "test_ac17_r4_support_append_scales_to_many_records",
    "test_ac17_r6_rebuild_from_async_iterator_then_search",
    "test_ac17_r6_support_rebuild_replaces_prior_contents",
    "test_ac17_r8_search_returns_typed_candidates_and_respects_limit",
    "test_ac17_r10_health_reports_status_and_failures",
    "test_ac17_r11_search_publishes_only_backend_neutral_types",
    "test_ac17_r13_search_on_empty_index_returns_empty_list",
    "test_ac17_r15_repeated_searches_publish_identical_scores_and_order",
)


@pytest.mark.parametrize("check", _SHARED_CHECKS)
async def test_ac26_r1_shared_continuity_index_contract(check, new_index, monkeypatch):
    async with new_index() as index:
        monkeypatch.setattr(shared_contract, "VectorHybridIndex", type(index))
        monkeypatch.setattr(shared_contract, "vector_module", _embedding_module())
        monkeypatch.setattr(shared_contract, "construct", lambda *_args: index)
        result = getattr(shared_contract, check)()
        if inspect.isawaitable(result):
            await result


async def test_ac26_r2_reopen_embeds_only_changed_content(
    new_index, endpoint, tmp_path
):
    records = [
        make_record(f"record-{i}", content=f"Archived decision number {i}.")
        for i in range(4)
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
    endpoint.vectors.update(
        {record.content: vector for record, vector in zip(records, vectors)}
    )
    originals = [record.model_dump_json() for record in records]
    cache = tmp_path / "persistent.sqlite"
    async with new_index(path=cache) as index:
        await index.rebuild(records_iter(records))
        assert sorted(endpoint.texts) == sorted(record.content for record in records)

    assert cache.is_file(), "the vector cache must persist in SQLite"
    with closing(sqlite3.connect(f"file:{cache}?mode=ro", uri=True)) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)

    replay = [ContinuityRecord.model_validate_json(raw) for raw in originals]
    replay[-1] = ContinuityRecord.model_validate(
        {**replay[-1].model_dump(), "content": "Revised storage decision."}
    )
    endpoint.vectors[replay[-1].content] = [0.0, 1.0, 1.0]
    queries = dict(zip(("cedar", "cobalt", "amber", "quartz"), replay))
    endpoint.vectors.update(
        {query: endpoint.vectors[record.content] for query, record in queries.items()}
    )
    assert [record.id for record in replay] == [record.id for record in records]
    endpoint.requests.clear()
    endpoint.texts.clear()
    async with new_index(path=cache) as reopened:
        await reopened.rebuild(records_iter(replay))
        assert len(endpoint.requests) == 1, (
            "three unchanged records must reuse their persisted vectors"
        )
        assert endpoint.texts == [replay[-1].content]
        endpoint.texts.clear()
        for query, expected in queries.items():
            context = CONTEXT.model_copy(update={"query": query})
            results = await reopened.search(context, limit=10)
            assert record_ids(results) == record_ids(replay)
            assert {
                c.record.current_record.id: c.record.current_record.content
                for c in results
            } == {record.id: record.content for record in replay}
            assert results[0].record.current_record.id == expected.id
            assert results[0].score > results[1].score
        assert set(endpoint.texts) == set(queries)
    assert [record.model_dump_json() for record in records] == originals


@pytest.mark.parametrize("operation", ["append", "search"])
async def test_ac26_r3_http_500_reports_retryable_index_failure(
    operation, new_index, endpoint
):
    async with new_index() as index:
        if operation == "search":
            await index.append([make_record("existing")])
        endpoint.requests.clear()
        endpoint.status = 500
        try:
            if operation == "append":
                await index.append([make_record("new")])
            else:
                await index.search(CONTEXT, limit=5)
        except Exception:
            # The protocol leaves raising versus returning on failure open.
            pass
        assert endpoint.requests and all(
            request["status"] == 500 for request in endpoint.requests
        )
        health = await index.health()
        assert isinstance(health, ContinuityHealth)
        assert health.status == "degraded"
        assert any(
            f.capability == "index" and f.retryable and f.message
            for f in health.failures
        )


@pytest.mark.parametrize(
    "nearest_vector,farther_vector",
    [
        ([2.0, 0.0, 0.0], [100.0, 100.0, 0.0]),
        ([100.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
    ],
    ids=["dot-product-disagrees", "euclidean-disagrees"],
)
async def test_ac26_r4_nearest_cosine_record_ranks_first(
    nearest_vector, farther_vector, new_index, endpoint
):
    nearest = make_record("z-nearest", content="cedar cobalt")
    farther = make_record("a-farther", content="amber quartz")
    context = CONTEXT.model_copy(update={"query": "recover earlier choice"})
    endpoint.vectors.update(
        {
            nearest.content: nearest_vector,
            farther.content: farther_vector,
            context.query: [1.0, 0.0, 0.0],
        }
    )
    async with new_index() as index:
        await index.append([farther, nearest])
        results = await index.search(context, limit=2)
        assert record_ids(results) == {nearest.id, farther.id}
        assert results[0].record.current_record.id == nearest.id
        assert results[0].score > results[1].score


@pytest.mark.parametrize(
    "dimensions", [3, None], ids=["explicit-dimensions", "omitted-dimensions"]
)
async def test_ac26_r5_yaml_settings_reach_embeddings_endpoint(
    dimensions, new_index, endpoint
):
    dimension_line = "      dimensions: 3\n" if dimensions is not None else ""
    document = yaml.safe_load(
        "continuity:\n"
        "  recent_token_budget: 4096\n"
        "  index:\n"
        "    type: embedding\n"
        "    config:\n"
        "      model: yaml-selected-embedding\n"
        f"      base_url: {endpoint.base_url}\n" + dimension_line
    )
    continuity = ContinuityConfig.model_validate(document["continuity"])
    configured_index = getattr(continuity, "index", None)
    assert configured_index is not None, (
        "public continuity config must retain continuity.index"
    )
    if isinstance(configured_index, BaseModel):
        configured_index = configured_index.model_dump(exclude_none=True)
    assert configured_index["type"] == "embedding"
    settings = configured_index["config"]
    async with new_index(settings=settings) as index:
        await index.append([make_record("yaml-record")])
    assert endpoint.requests, (
        "the parsed configuration must produce an embedding request"
    )
    for request in endpoint.requests:
        assert request["path"] == "/configured/v1/embeddings"
        assert request["body"]["model"] == "yaml-selected-embedding"
        assert request["body"].get("dimensions") == dimensions


async def test_ac26_r6_embeds_record_content_and_episode_title_summary(
    new_index, endpoint
):
    ordinary = make_record("ordinary", content="Prefer smaller deployment batches.")
    episode = ContinuityRecord.model_validate(
        {
            **make_record("episode", kind=ContinuityKind.EPISODE).model_dump(),
            "content": "We chose SQLite for the portable local archive.",
            "meta": {
                "episode_state": "closed",
                "episode_id": "archive-choice",
                "title": "Portable archive design",
                "turn_range": [1, 3],
            },
        }
    )
    async with new_index() as index:
        await index.append([ordinary, episode])
    assert any(ordinary.content in text for text in endpoint.texts)
    assert any(
        episode.meta["title"] in text and episode.content in text
        for text in endpoint.texts
    )


@pytest.mark.parametrize(
    "signal", ["lexical", "salience", "confidence", "recency", "project"]
)
async def test_ac26_r7_structured_signal_changes_score_at_equal_cosine(
    signal, new_index, endpoint
):
    low = make_record("a-low", content="cedar cobalt")
    high = make_record("z-high", content="cedar cobalt")
    low_changes, high_changes = {
        "lexical": ({}, {"content": "amber quartz"}),
        "salience": ({"salience": 0.1}, {"salience": 0.9}),
        "confidence": ({"confidence": 0.1}, {"confidence": 0.9}),
        "recency": ({"created_at": NOW - timedelta(days=7)}, {}),
        "project": ({"scope": replace(low.scope, projects=("other-project",))}, {}),
    }[signal]
    low = ContinuityRecord.model_validate({**low.model_dump(), **low_changes})
    high = ContinuityRecord.model_validate({**high.model_dump(), **high_changes})
    context = CONTEXT.model_copy(update={"query": "amber quartz"})
    reference = StructuredIndex()
    await reference.append([low, high])
    expected = {
        c.record.current_record.id: c for c in await reference.search(context, limit=2)
    }
    async with new_index() as index:
        await index.append([low, high])
        results = await index.search(context, limit=2)
    assert set(endpoint.texts) >= {low.content, high.content, context.query}
    actual = {c.record.current_record.id: c for c in results}
    assert set(actual) == {low.id, high.id}
    assert actual[high.id].score > actual[low.id].score
    assert results[0].record.current_record.id == high.id
    for record in (low, high):
        for component, value in expected[record.id].score_components.items():
            assert actual[record.id].score_components[component] == pytest.approx(value)
