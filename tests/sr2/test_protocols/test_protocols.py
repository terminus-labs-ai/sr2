"""Tests for SR2 plugin protocols."""

import pytest

from sr2.bridge import ContextBridge
from sr2.protocols.bridge import ContextBridgeProtocol
from sr2.protocols.exporters import PullExporter, PushExporter
from sr2.protocols.alerts import Alert, AlertEngine
from sr2.protocols.stores import EmbeddingStore, LifecycleStore
from sr2.memory.store import InMemoryMemoryStore


class TestContextBridgeProtocol:
    def test_context_bridge_satisfies_protocol(self):
        assert isinstance(ContextBridge(), ContextBridgeProtocol)

    def test_protocol_is_runtime_checkable(self):
        class BadBridge:
            pass

        assert not isinstance(BadBridge(), ContextBridgeProtocol)


class TestPushExporter:
    def test_mock_push_exporter_satisfies_protocol(self):
        class MockOTel:
            def register(self, collector):
                pass

        assert isinstance(MockOTel(), PushExporter)

    def test_non_push_exporter_fails(self):
        class WrongExporter:
            def export(self) -> str:
                return ""

        assert not isinstance(WrongExporter(), PushExporter)


class TestPullExporter:
    def test_mock_pull_exporter_satisfies_protocol(self):
        class MockProm:
            def export(self) -> str:
                return "# HELP sr2_test test\n"

        assert isinstance(MockProm(), PullExporter)

    def test_non_pull_exporter_fails(self):
        class WrongExporter:
            def register(self, collector):
                pass

        assert not isinstance(WrongExporter(), PullExporter)


class TestAlertEngine:
    def test_mock_alert_engine_satisfies_protocol(self):
        class MockAlerts:
            async def evaluate(self, snapshot):
                return []

            def configure(self, rules):
                pass

        assert isinstance(MockAlerts(), AlertEngine)

    def test_alert_dataclass(self):
        alert = Alert(
            metric_name="sr2_cache_hit_rate",
            actual_value=0.3,
            threshold_value=0.5,
            condition="<",
            severity="warning",
            timestamp=1234567890.0,
            message="Cache hit rate below threshold",
        )
        assert alert.metric_name == "sr2_cache_hit_rate"
        assert alert.severity == "warning"
        assert alert.labels == {}

    def test_non_alert_engine_fails(self):
        class NotAnEngine:
            pass

        assert not isinstance(NotAnEngine(), AlertEngine)


class TestLifecycleStore:
    def test_mock_lifecycle_store_satisfies_protocol(self):
        class MockPgStore:
            async def create_tables(self):
                pass

        assert isinstance(MockPgStore(), LifecycleStore)

    def test_in_memory_store_does_not_satisfy(self):
        store = InMemoryMemoryStore()
        assert not isinstance(store, LifecycleStore)


class TestEmbeddingStore:
    def test_mock_embedding_store_satisfies_protocol(self):
        class MockPgStore:
            async def update_embedding(self, memory_id, embedding):
                pass

            async def list_without_embeddings(self, limit=100):
                return []

        assert isinstance(MockPgStore(), EmbeddingStore)

    def test_in_memory_store_does_not_satisfy(self):
        store = InMemoryMemoryStore()
        assert not isinstance(store, EmbeddingStore)


class TestCombinedStoreProtocols:
    def test_full_pg_mock_satisfies_all(self):
        """A Postgres-like store satisfies MemoryStore + LifecycleStore + EmbeddingStore."""

        class FullStore:
            async def save(self, memory, embedding=None):
                pass

            async def get(self, memory_id):
                return None

            async def get_by_key(self, key, include_archived=False, scope_filter=None, scope_refs=None):
                return []

            async def search_by_key_prefix(self, prefix, include_archived=False):
                return []

            async def delete(self, memory_id):
                return False

            async def archive(self, memory_id):
                return False

            async def search_vector(self, embedding, top_k=10, include_archived=False, scope_filter=None, scope_refs=None):
                return []

            async def search_keyword(self, query, top_k=10, include_archived=False, scope_filter=None, scope_refs=None):
                return []

            async def count(self, include_archived=False):
                return 0

            async def list_scope_refs(self, scope_filter=None, include_archived=False):
                return []

            # LifecycleStore
            async def create_tables(self):
                pass

            # EmbeddingStore
            async def update_embedding(self, memory_id, embedding):
                pass

            async def list_without_embeddings(self, limit=100):
                return []

        store = FullStore()
        assert isinstance(store, LifecycleStore)
        assert isinstance(store, EmbeddingStore)
