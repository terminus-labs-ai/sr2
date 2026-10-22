"""sr2-93 / AC2 — Turn-archive causal fields and real lineage (unit layer).

Cage for the causal-metadata contract (spec FR2, Interfaces vocabulary:
ContextReason / ContextAuthority / ContextLifecycle) and for FR5 / the
bead's "repair synthetic lineage sources" over the turn archive written by
SR2.turn().

Case-table rows covered here: AC2-r5 (causal fields) and AC2-r6 (lineage).
Rows AC2-r1..r4, r7..r9 live in tests/pipeline/test_turn_archive.py.

Entry-shape facts fixed by the orchestrator amendment (design decision b):

    layer == "turn_archive"; session_id == SR2.session_id;
    meta["turn_seq"] / monotonic per-turn meta["seq"];
    reason  USER_INPUT | ASSISTANT_OUTPUT | TOOL_USE | TOOL_RESULT;
    authority USER for user input, AGENT_TENTATIVE for assistant-produced
              entries, a valid ContextAuthority for tool entries (value not
              spec-fixed — presence/validity asserted only);
    lifecycle ARCHIVE; scope.owner_id non-empty;
    origin EntryOrigin(kind="transformer", name="turn_archivist) for
    entries carrying sources, ("resolver", "turn_archivist") genesis for
    the turn's first entry.

    Causal chain (FR5): each archived entry's sources = (id of the previous
    archived entry for this session,) — real stored ids only, written in
    order so every referenced id exists. Synthetic dangling source ids are
    invalid and lineage traversal from any archived entry must resolve.

    read_turn_archive(store, session_id, turn_seq) -> list[Entry] is fixed
    by the amendment (design decision c).

Both rows drive one tool-loop turn through the public SR2.turn() API —
identical driver shape to AC2-r1 — and read back through the store. The
spec symbols exist from sr2-92, but ``read_turn_archive`` does not: module
attribute access keeps collection green and every test failing with a clean
AttributeError (the expected red) until C implements it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# See test_turn_archive.py: make the shared conftest helpers importable
# regardless of how pytest was invoked.
_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    tool_use_event,
)

import sr2.pipeline.provenance as prov
from sr2.config.models import PipelineConfig
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent


# ---------------------------------------------------------------------------
# Read-surface accessor (amendment-fixed; see module docstring)
# ---------------------------------------------------------------------------


def read_turn_archive(store, session_id: str, turn_seq: int) -> list:
    fn = prov.read_turn_archive
    return fn(store, session_id, turn_seq)


def _valid_authority(value) -> None:
    """Authority values for tool entries are not spec-fixed — assert
    presence/validity only (case-table grounding rule)."""
    assert value is not None
    assert isinstance(value, prov.ContextAuthority)


def _entry_text(entry) -> str:
    """The text an archived entry carries, whichever authorized encoding it
    uses (a Message wrapper holding TextBlocks, or a bare TextBlock). Text
    equality is what AC2 fixes; the wrapper is not asserted.
    """
    content = entry.content
    blocks = content.content if isinstance(content, Message) else [content]
    if not isinstance(blocks, list):
        blocks = []
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


# ---------------------------------------------------------------------------
# Driver — one tool-loop turn, same shape as AC2-r1
# ---------------------------------------------------------------------------


def _config() -> PipelineConfig:
    return make_minimal_config()


async def _drive_one_tool_turn(store):
    """Run one tool-loop turn against the given store; return the SR2
    instance and the ToolResultBlock SR2 received (what TOOL_RESULT must
    equal in the sibling file; here it anchors the sequence shape).

    Sequence: USER_INPUT, ASSISTANT_OUTPUT, TOOL_USE, TOOL_RESULT,
    ASSISTANT_OUTPUT.
    """

    async def scripted_executor(block: ToolUseBlock) -> ToolResultBlock:
        return ToolResultBlock(tool_use_id=block.id, content="sunny, 15C")

    llm = SequentialMockLLM(
        [
            [
                StreamEvent(type="text", text="Let me check"),
                tool_use_event(tool_use_id="tc_1"),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="All sunny."),
                StreamEvent(type="end"),
            ],
        ]
    )

    from sr2.orchestrator import SR2

    sr2 = SR2(
        pipeline_config=_config(),
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        provenance_store=store,
        tool_executor=scripted_executor,
    )
    events = [e async for e in sr2.turn(make_user_input("weather please"), origin="")]
    assert any(e.type == "end" for e in events), "turn must complete"
    await sr2.aclose()
    return sr2


_EXPECTED_REASONS = [
    prov.ContextReason.USER_INPUT,
    prov.ContextReason.ASSISTANT_OUTPUT,
    prov.ContextReason.TOOL_USE,
    prov.ContextReason.TOOL_RESULT,
    prov.ContextReason.ASSISTANT_OUTPUT,
]


# ===========================================================================
# AC2-r5 — Causal fields of the turn's archive entries.
# ===========================================================================


class TestAC2R5CausalFields:
    async def test_archive_entries_carry_spec_causal_metadata(self):
        """AC2-r5: USER_INPUT entry -> authority USER; assistant iteration and
        final entries -> reason ASSISTANT_OUTPUT, authority
        AGENT_TENTATIVE; TOOL_USE/TOOL_RESULT entries -> exact reasons,
        valid ContextAuthority; every archive entry -> lifecycle ARCHIVE,
        session_id == sr2.session_id, scope.owner_id non-empty."""
        store = prov.InMemoryProvenanceStore()
        sr2 = await _drive_one_tool_turn(store)

        entries = await read_turn_archive(store, sr2.session_id, 0)
        assert [e.reason for e in entries] == _EXPECTED_REASONS, (
            "r5 builds on r1's sequence; fix the sequence first"
        )

        # Fixed values by position (the r1 sequence is itself asserted).
        user_entry = entries[0]
        assert user_entry.reason == prov.ContextReason.USER_INPUT
        assert user_entry.authority == prov.ContextAuthority.USER

        for pos in (1, 4):  # intermediate iteration + final assistant output
            assert entries[pos].reason == prov.ContextReason.ASSISTANT_OUTPUT
            assert entries[pos].authority == prov.ContextAuthority.AGENT_TENTATIVE, (
                f"position {pos}: assistant-produced entries are tentative "
                f"(spec FR12/Interfaces), got {entries[pos].authority!r}"
            )

        for pos in (2, 3):  # TOOL_USE, TOOL_RESULT
            assert entries[pos].reason in (
                prov.ContextReason.TOOL_USE,
                prov.ContextReason.TOOL_RESULT,
            )
            # tool-entry authority value is not spec-fixed: validity only
            _valid_authority(entries[pos].authority)

        # Invariants over every archived entry.
        for e in entries:
            assert e.lifecycle == prov.ContextLifecycle.ARCHIVE, (
                f"entry {e.id}: archived payloads have lifecycle ARCHIVE, "
                f"got {e.lifecycle!r}"
            )
            assert e.session_id == sr2.session_id
            assert e.scope is not None
            assert isinstance(e.scope.owner_id, str)
            assert e.scope.owner_id != "", (
                f"entry {e.id}: scope.owner_id must be non-empty"
            )

        # Payload sanity the causal fields attach to (kept minimal; r1
        # owns payload equality). AC2 fixes the ordered payload *content*;
        # how a payload is wrapped in an Entry (Message vs. bare block) is
        # an encoding choice the amendment authorizes but does not fix, so
        # only the user-input text is asserted here — as in the sibling
        # rows.
        user_text = _entry_text(user_entry)
        assert user_text == "weather please", (
            f"USER_INPUT entry must carry the turn's user input text, got {user_text!r}"
        )


# ===========================================================================
# AC2-r6 — Lineage: real stored ids only; traversal reaches user input.
# ===========================================================================


class TestAC2R6LineageRealSources:
    async def test_archive_sources_exist_and_lineage_reaches_user_input(self):
        """AC2-r6: every archive entry's sources are ids present in the store
        (no synthetic/dangling ids), and get_lineage from the final
        ASSISTANT_OUTPUT entry reaches that turn's USER_INPUT entry."""
        store = prov.InMemoryProvenanceStore()
        sr2 = await _drive_one_tool_turn(store)

        entries = await read_turn_archive(store, sr2.session_id, 0)
        assert [e.reason for e in entries] == _EXPECTED_REASONS, (
            "r6 builds on r1's sequence; fix the sequence first"
        )

        # 1. No synthetic/dangling source ids: every referenced id exists.
        #    This includes ids referenced across turns/sessions.
        for e in entries:
            for source_id in e.sources:
                fetched = await store.get(source_id)
                assert fetched is not None, (
                    f"archive entry {e.id} (reason {e.reason!r}) references "
                    f"source {source_id!r} which is not present in the store "
                    "— synthetic dangling source ids are invalid (FR5)"
                )

        # 2. The causal chain is the archive itself: each entry (after the
        #    turn's genesis) references an already-stored archived entry, so
        #    traversal from the final entry walks back to the user input.
        final_entry = entries[-1]
        assert final_entry.reason == prov.ContextReason.ASSISTANT_OUTPUT
        lineage = await store.get_lineage(final_entry.id)
        lineage_ids = {e.id for e in lineage}
        user_entry = entries[0]
        assert user_entry.id in lineage_ids, (
            "get_lineage from the final ASSISTANT_OUTPUT entry must reach "
            "this turn's USER_INPUT entry (FR5: lineage traversal MUST "
            "reach the original user/tool entries); "
            f"lineage ids: {lineage_ids!r}"
        )
        # Every archived entry of the turn is reachable through real links.
        for e in entries:
            assert e.id in lineage_ids, (
                f"entry {e.id} (reason {e.reason!r}) is not reachable from "
                "the final archived entry — the per-entry chain must be "
                "complete and real"
            )
