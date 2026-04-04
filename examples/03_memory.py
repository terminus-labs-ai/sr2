"""Memory system — extraction, conflicts, and retrieval.

Demonstrates:
- Extracting structured memories from conversation text
- Conflict detection when facts change
- Conflict resolution strategies (archive old, keep both)
- Hybrid retrieval with keyword search

No LLM needed — uses a mock LLM callable that returns structured JSON.

Run:
    pip install sr2
    python examples/03_memory.py
"""

import asyncio
import json

from sr2.memory.conflicts import ConflictDetector
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


async def mock_llm(prompt: str) -> str:
    """Simulate LLM extraction by returning pre-defined memories.

    In production, this would be an actual LLM call. The extractor sends
    the conversation turn and expects JSON back.
    """
    if "Alice" in prompt and "Anthropic" in prompt:
        return json.dumps([
            {"key": "user.name", "value": "Alice", "memory_type": "identity", "confidence_source": "explicit_statement"},
            {"key": "user.employer", "value": "Anthropic", "memory_type": "semi_stable", "confidence_source": "direct_answer"},
            {"key": "user.language", "value": "Python", "memory_type": "semi_stable", "confidence_source": "contextual_mention"},
        ])
    if "Google" in prompt:
        return json.dumps([
            {"key": "user.employer", "value": "Google", "memory_type": "semi_stable", "confidence_source": "explicit_statement"},
        ])
    return "[]"


async def main():
    store = InMemoryMemoryStore()

    # --- 1. Extract memories from a conversation turn ---
    print("=== Memory Extraction ===\n")

    extractor = MemoryExtractor(llm_callable=mock_llm, store=store)

    result = await extractor.extract(
        "Hi, I'm Alice! I work at Anthropic and mostly write Python.",
        conversation_id="conv_001",
        turn_number=1,
    )

    print(f"Extracted {len(result.memories)} memories:")
    for mem in result.memories:
        print(f"  [{mem.key}] {mem.value} (type={mem.memory_type}, confidence={mem.confidence_source})")
    print()

    # --- 2. Retrieve memories ---
    print("=== Memory Retrieval ===\n")

    retriever = HybridRetriever(store=store, strategy="keyword")

    results = await retriever.retrieve("anthropic", top_k=5)
    print(f"Query 'anthropic' returned {len(results)} results:")
    for r in results:
        print(f"  [{r.memory.key}] {r.memory.value} (score={r.relevance_score:.2f})")
    print()

    results = await retriever.retrieve("python", top_k=5)
    print(f"Query 'python' returned {len(results)} results:")
    for r in results:
        print(f"  [{r.memory.key}] {r.memory.value} (score={r.relevance_score:.2f})")
    print()

    # --- 3. Detect and resolve conflicts ---
    print("=== Conflict Detection & Resolution ===\n")

    # Alice changed jobs
    new_employer = Memory(key="user.employer", value="Google", memory_type="semi_stable")

    detector = ConflictDetector(store=store)
    conflicts = await detector.detect(new_employer)

    print(f"Detected {len(conflicts)} conflict(s):")
    for c in conflicts:
        print(f"  {c.existing_memory.key}: '{c.existing_memory.value}' vs '{c.new_memory.value}' (type={c.conflict_type})")
    print()

    # Resolve with default strategy (latest_wins_archive for semi_stable)
    resolver = ConflictResolver(store=store)
    for conflict in conflicts:
        resolution = await resolver.resolve(conflict)
        print(f"Resolution: {resolution.action}")
        print(f"  Winner: {resolution.winner.value}")
        print(f"  Loser: {resolution.loser.value}")
    print()

    # Verify: old memory is archived, new one is active
    employer_memories = await store.get_by_key("user.employer", include_archived=True)
    print("Employer memories after resolution:")
    for mem in employer_memories:
        status = "ARCHIVED" if mem.archived else "ACTIVE"
        print(f"  [{status}] {mem.value}")
    print()

    # --- 4. Retrieve again — should reflect the update ---
    print("=== Post-Update Retrieval ===\n")

    # Save the new memory
    await store.save(new_employer)

    results = await retriever.retrieve("employer", top_k=5)
    print(f"Query 'employer' returned {len(results)} results:")
    for r in results:
        status = "ARCHIVED" if r.memory.archived else "ACTIVE"
        print(f"  [{r.memory.key}] {r.memory.value} ({status}, score={r.relevance_score:.2f})")

    total = await store.count()
    archived = await store.count(include_archived=True) - total
    print(f"\nStore: {total} active, {archived} archived")


if __name__ == "__main__":
    asyncio.run(main())
