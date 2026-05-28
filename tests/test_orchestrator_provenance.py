"""Tests for SR2 orchestrator provenance integration — Chunk 4.

Covers:
  FR11: SR2 accepts session_id; mints ULID if None
  FR12: SR2 accepts provenance_store; defaults to InMemoryProvenanceStore
  FR14: _build_layer raises PluginNotFoundError when LayerConfig declares an unknown transformer type
  AC1:  Round-trip test — entries survive SQLite close + reopen
  AC5:  InMemoryProvenanceStore satisfies isinstance(store, ProvenanceStore)
  AC6:  SQLiteProvenanceStore satisfies isinstance(store, ProvenanceStore) (via orchestrator)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.provenance import Entry, EntryOrigin, InMemoryProvenanceStore, ProvenanceStore
from sr2.plugins.errors import PluginNotFoundError
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    StreamEvent,
)

# ---------------------------------------------------------------------------
# Helpers (copied from test_orchestrator.py — not imported from there)
# ---------------------------------------------------------------------------


def make_user_input(text: str = "Hello") -> list:
    """Return a minimal list[ContentBlock] representing user input."""
    return [TextBlock(text=text)]


def make_completion_response(text: str = "I am the assistant.") -> CompletionResponse:
    return CompletionResponse(
        id="test-resp-001",
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


class MockLLM:
    """Minimal LLMCallable implementation for testing."""

    def __init__(self, events: list[StreamEvent] | None = None):
        self._events: list[StreamEvent] = events or [
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="world"),
            StreamEvent(type="end"),
        ]
        self.stream_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return make_completion_response()

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        for event in self._events:
            yield event


def make_minimal_config() -> PipelineConfig:
    """Minimal two-layer PipelineConfig sufficient for testing."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ]
    )


def make_config_with_transformers(layer_name: str = "system") -> PipelineConfig:
    """Config where one layer declares a transformer."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name=layer_name,
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
                transformers=[
                    TransformerConfig(type="some_transformer"),
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ]
    )


_ENTRY_COUNTER = 0


def _next_entry_id() -> str:
    global _ENTRY_COUNTER
    _ENTRY_COUNTER += 1
    return f"ORCH{_ENTRY_COUNTER:022d}"


def make_entry(session_id: str, layer: str = "conversation") -> Entry:
    return Entry(
        id=_next_entry_id(),
        content=TextBlock(text="round-trip content"),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="test_resolver"),
        layer=layer,
        session_id=session_id,
        created_at=datetime.now(tz=timezone.utc),
    )


def make_config_with_empty_transformers() -> PipelineConfig:
    """Config where a layer declares transformers=[] (empty list — should be fine)."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
                transformers=[],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 1. TestSR2SessionId — FR11
# ---------------------------------------------------------------------------


class TestSR2SessionId:
    def test_session_id_auto_minted_when_none(self):
        """SR2() without session_id → sr2.session_id is a non-empty string."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert isinstance(sr2.session_id, str)
        assert len(sr2.session_id) > 0

    def test_session_id_is_ulid_format(self):
        """SR2() without session_id → sr2.session_id is 26 characters (ULID format)."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert len(sr2.session_id) == 26

    def test_explicit_session_id_stored(self):
        """SR2(session_id='my-id') → sr2.session_id == 'my-id'."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id="my-id",
        )

        assert sr2.session_id == "my-id"

    def test_explicit_session_id_is_exact_string(self):
        """SR2(session_id='my-id') → session_id is the exact string passed, not a copy."""
        from sr2.orchestrator import SR2

        custom_id = "explicit-session-42"
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id=custom_id,
        )

        assert sr2.session_id is custom_id

    def test_two_instances_get_different_session_ids(self):
        """Two SR2() without session_id → different session IDs (mint is unique)."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        llm = {"default": MockLLM()}
        counter = CharacterTokenCounter()

        sr2_a = SR2(pipeline_config=config, llm=llm, token_counter=counter)
        sr2_b = SR2(pipeline_config=config, llm=llm, token_counter=counter)

        assert sr2_a.session_id != sr2_b.session_id


# ---------------------------------------------------------------------------
# 2. TestSR2ProvenanceStore — FR12
# ---------------------------------------------------------------------------


class TestSR2ProvenanceStore:
    def test_constructs_without_provenance_store(self):
        """SR2() without provenance_store → constructs without error."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert sr2 is not None

    def test_default_provenance_store_is_in_memory(self):
        """SR2() without provenance_store → engine uses InMemoryProvenanceStore."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert isinstance(sr2._engine._provenance_store, InMemoryProvenanceStore)

    def test_explicit_provenance_store_used_as_is(self):
        """SR2(provenance_store=store) → engine._provenance_store is the exact object."""
        from sr2.orchestrator import SR2

        custom_store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=custom_store,
        )

        assert sr2._engine._provenance_store is custom_store

    def test_provided_store_satisfies_protocol(self):
        """SR2(provenance_store=InMemoryProvenanceStore()) → store satisfies ProvenanceStore."""
        from sr2.orchestrator import SR2

        store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        # AC5: InMemoryProvenanceStore satisfies isinstance check
        assert isinstance(sr2._engine._provenance_store, ProvenanceStore)


# ---------------------------------------------------------------------------
# 3. TestTransformerConfigError — FR14
# ---------------------------------------------------------------------------


class TestTransformerConfigError:
    def test_empty_transformers_list_does_not_raise(self):
        """transformers=[] → no error; empty list is fine."""
        from sr2.orchestrator import SR2

        # Must not raise
        sr2 = SR2(
            pipeline_config=make_config_with_empty_transformers(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert sr2 is not None

    def test_non_empty_transformers_with_unknown_type_raises_plugin_not_found_error(self):
        """transformers=[TransformerConfig(type=unknown)] → raises PluginNotFoundError at SR2 construction."""
        from sr2.orchestrator import SR2

        with pytest.raises(PluginNotFoundError):
            SR2(
                pipeline_config=make_config_with_transformers(),
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )

    def test_plugin_not_found_error_message_contains_unknown_type_name(self):
        """PluginNotFoundError message contains the unknown transformer type name."""
        from sr2.orchestrator import SR2

        config = make_config_with_transformers()

        with pytest.raises(PluginNotFoundError, match="some_transformer"):
            SR2(
                pipeline_config=config,
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )

    def test_plugin_not_found_error_message_mentions_transformer(self):
        """PluginNotFoundError message mentions 'transformer' to guide the user."""
        from sr2.orchestrator import SR2

        with pytest.raises(PluginNotFoundError) as exc_info:
            SR2(
                pipeline_config=make_config_with_transformers(),
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )

        message = str(exc_info.value).lower()
        assert "transformer" in message

    def test_single_layer_with_unknown_transformer_raises(self):
        """Only one layer has transformers → PluginNotFoundError is still raised."""
        from sr2.orchestrator import SR2

        # Second layer has transformers, first does not
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are a helpful assistant."},
                        )
                    ],
                    # No transformers on this layer
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[
                        ResolverConfig(type="session"),
                        ResolverConfig(
                            type="input",
                            subscriptions=[
                                EventSubscriptionConfig(event="user_input", phase="completed")
                            ],
                        ),
                    ],
                    transformers=[
                        TransformerConfig(type="some_transformer"),
                    ],
                ),
            ]
        )

        with pytest.raises(PluginNotFoundError):
            SR2(
                pipeline_config=config,
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )


# ---------------------------------------------------------------------------
# 4. TestProtocolCompliance — AC5 direct
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_in_memory_store_satisfies_protocol(self):
        """AC5: InMemoryProvenanceStore() satisfies isinstance(store, ProvenanceStore)."""
        store = InMemoryProvenanceStore()
        assert isinstance(store, ProvenanceStore)


# ---------------------------------------------------------------------------
# 5. TestSR2RoundTrip — AC1
#
# Built-in resolvers use the old content= path (entries not yet migrated).
# Round-trip tests write entries directly to the store via sr2.session_id,
# verifying that session threading and store persistence work end-to-end.
# ---------------------------------------------------------------------------


class TestSR2RoundTrip:
    @pytest.mark.asyncio
    async def test_in_memory_store_session_id_threaded_to_entries(self):
        """FR11: session_id threads through — entries written with sr2.session_id are queryable."""
        from sr2.orchestrator import SR2

        store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        # Write an entry tagged with SR2's session_id (simulates a migrated resolver)
        entry = make_entry(session_id=sr2.session_id)
        await store.write(entry)

        entries = await store.get_session(sr2.session_id)
        assert len(entries) == 1
        assert entries[0].session_id == sr2.session_id
        assert entries[0].id == entry.id

    @pytest.mark.asyncio
    async def test_sqlite_store_entries_survive_close_and_reopen(self, tmp_path):
        """AC1: Entries written with session_id survive SQLite close → reopen."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "provenance_test.db"
        session_id = "round-trip-session-001"

        # --- Phase 1: write entries to the store ---
        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id=session_id,
            provenance_store=store,
        )

        # Write entries tagged with the session (simulates migrated resolvers)
        entry1 = make_entry(session_id=sr2.session_id, layer="system")
        entry2 = make_entry(session_id=sr2.session_id, layer="conversation")
        await store.write_batch([entry1, entry2])

        entries_before = await store.get_session(session_id)
        assert len(entries_before) == 2
        assert all(e.session_id == session_id for e in entries_before)

        # Close the store (simulates process shutdown)
        await store.close()

        # --- Phase 2: reopen and verify entries still exist ---
        store2 = SQLiteProvenanceStore(db_path=str(db_path))
        await store2.connect()

        entries_after = await store2.get_session(session_id)

        assert len(entries_after) == len(entries_before)
        ids_before = {e.id for e in entries_before}
        ids_after = {e.id for e in entries_after}
        assert ids_before == ids_after
        assert all(e.session_id == session_id for e in entries_after)

        await store2.close()

    @pytest.mark.asyncio
    async def test_sqlite_store_satisfies_protocol_via_orchestrator(self, tmp_path):
        """AC6: SQLiteProvenanceStore plugged into SR2 satisfies isinstance(store, ProvenanceStore)."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "protocol_check.db"
        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        assert isinstance(sr2._engine._provenance_store, ProvenanceStore)

        await store.close()
