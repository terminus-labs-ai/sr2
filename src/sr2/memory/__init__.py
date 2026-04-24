"""Memory subsystem — store backends, extraction, retrieval, and conflict resolution."""

from sr2.memory.registry import register_store
from sr2.memory.store import InMemoryMemoryStore, SQLiteMemoryStore

# Register built-in memory store backends.
register_store("memory", InMemoryMemoryStore)
register_store("sqlite", SQLiteMemoryStore)
