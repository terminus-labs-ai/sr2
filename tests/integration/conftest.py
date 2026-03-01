"""Integration test configuration and fixtures."""

import os

import pytest

POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://sr2:sr2@localhost:54321/sr2_test",
)

LLM_API_KEY = os.environ.get("TEST_LLM_API_KEY")

requires_postgres = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL") and not os.environ.get("RUN_INTEGRATION"),
    reason="Set TEST_POSTGRES_URL or RUN_INTEGRATION=1",
)

requires_llm = pytest.mark.skipif(
    not LLM_API_KEY, reason="TEST_LLM_API_KEY not set"
)


@pytest.fixture
async def pg_pool():
    """Create a connection pool and clean up after tests."""
    import asyncpg

    pool = await asyncpg.create_pool(POSTGRES_URL)
    # Clean test data before each test
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS memories")
    yield pool
    await pool.close()


@pytest.fixture
async def pg_store(pg_pool):
    """Create a PostgresMemoryStore with tables initialized."""
    from sr2.memory.store import PostgresMemoryStore

    store = PostgresMemoryStore(pg_pool)
    await store.create_tables()
    return store
