"""sr2-93 / AC2 — Exact durable turn archive (unit layer).

Cage for FR3 ("SR2 MUST append the complete event sequence it receives for
each turn: user input, assistant outputs, tool calls, and tool results in
order", "Exact means the payload received by SR2") and AC2 ("the ordered
user, assistant, tool-use, and SR2-visible tool-result payloads survive
store close/reopen and reproduce the turn event sequence exactly, including
truncation markers").

Case-table rows covered here: AC2-r1, r2, r3, r4, r7, r8, r9.
Rows AC2-r5 (causal fields) and AC2-r6 (lineage) live in
tests/pipeline/test_provenance_lineage.py.

Read surface FIXED by the orchestrator amendment (design decision c):

    from sr2.pipeline.provenance import read_turn_archive
    read_turn_archive(store, session_id, turn_seq) -> list[Entry]

    Filters layer == "turn_archive" and meta["turn_seq"], returns the
    entries ordered by the durable per-turn meta["seq"]. There is no
    ProvenanceStore protocol change.

Entry-shape facts fixed by the amendment that these tests assert:

    layer == "turn_archive"; session_id == SR2.session_id;
    meta["turn_seq"], monotonic meta["seq"];
    reason USER_INPUT / ASSISTANT_OUTPUT / TOOL_USE / TOOL_RESULT;
    one Entry per payload item — user input -> Message(role="user"),
    each assistant iteration and the final output -> Message(role=
    "assistant"), each ToolUseBlock / ToolResultBlock -> the block itself
    (payload verbatim, upstream truncation markers included). Iterations with no
    SR2-visible text may be omitted (the archive holds what SR2 saw — an
    iteration that produced no text has no text payload to record).

Control events (retry, finish_reason, iteration_complete, repeat_guard,
layer truncation warnings, error) are NOT archived (spec R1); guard-stop
and forced-final-answer endings still archive their executed tool calls/
results and the forced final assistant output.

Grounding rules honored here:
  - Tests drive only public APIs: SR2.turn(), the provenance store, and
    read_turn_archive as fixed above. conftest helpers are reused unedited.
  - Mechanism is NOT pinned: how seq is persisted in SQLite, private class
    names, and the writing component's location are the implementer's
    choice. Only the read-back sequence, payload equality, durability, and
    ordering are asserted.
  - ``read_turn_archive`` does not exist yet: module attribute access keeps
    collection green and makes every test fail with a clean AttributeError
    (the expected red) until C implements it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import the shared helpers the same way sibling test modules do
# (``from conftest import ...``): the tests/ root must be importable even
# when pytest is invoked with only these test paths.
_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from ulid import ULID as _ULID

import sr2.pipeline.provenance as prov
from sr2.config.models import PipelineConfig
from sr2.models import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    tool_use_event,
)


# ---------------------------------------------------------------------------
# Read-surface accessor — module attribute access, NOT a module-level import,
# so the missing helper fails per-test instead of breaking collection.
# ---------------------------------------------------------------------------


def read_turn_archive(store, session_id: str, turn_seq: int) -> list:
    """The amendment-fixed read surface (see module docstring)."""
    fn = prov.read_turn_archive
    return fn(store, session_id, turn_seq)


# ---------------------------------------------------------------------------
# Payload comparison — content equality tolerant of pydantic models and of
# the SQLite JSON round-trip (which may return list-typed content).
# ---------------------------------------------------------------------------


def _content_equal(a, b) -> bool:
    if a == b:
        return True
    try:
        return a.model_dump() == b.model_dump()
    except (AttributeError, TypeError, ValueError):
        return False


def _assert_payload_equal(archived, expected, *, where: str) -> None:
    assert _content_equal(archived, expected), (
        f"{where}: archived content {archived!r} does not equal the "
        f"SR2-visible payload item {expected!r}"
    )


def _assert_message_text(entry, role: str, expected_text: str, *, where: str) -> None:
    """Assert an archived assistant/user entry: a Message with the given role
    carrying the concatenated SR2-visible text for that event.

    Whether the archived Message holds a single TextBlock or (after the
    SQLite round-trip) an equivalent text payload is not pinned — text
    equality is what AC2 fixes.
    """
    content = entry.content
    assert isinstance(content, Message), (
        f"{where}: expected a Message content, got {content!r}"
    )
    assert content.role == role, (
        f"{where}: expected role {role!r}, got {content.role!r}"
    )
    blocks = content.content if isinstance(content.content, list) else []
    text = "".join(b.text for b in blocks if isinstance(b, TextBlock))
    assert text == expected_text, (
        f"{where}: expected archived text {expected_text!r}, got {text!r}"
    )


# ---------------------------------------------------------------------------
# SR2 drivers — turn-driven through the public orchestrator API only
# ---------------------------------------------------------------------------


def _config(**pipeline_kwargs: Any) -> PipelineConfig:
    """Minimal two-layer config, optionally overridden (e.g. repeat_guard=).

    Built via model_copy: PipelineConfig does not accept a round-tripped
    dump (RepeatGuardConfig has no public alias), and copy keeps every
    untouched field exactly as make_minimal_config() defines it.
    """
    base: PipelineConfig = make_minimal_config()
    if pipeline_kwargs:
        base = base.model_copy(update=pipeline_kwargs)
    return base


def _build_sr2(
    *, llm, store, tool_executor=None, session_id: str | None = None, **pipeline_kwargs
):
    """SR2 wired to the minimal config; *llm* may be a factory called with
    the turn's scripted executor (for drivers that build both together)."""
    from sr2.orchestrator import SR2

    resolved = llm(tool_executor) if callable(llm) else llm
    return SR2(
        pipeline_config=_config(**pipeline_kwargs),
        llm={"default": resolved},
        token_counter=CharacterTokenCounter(),
        provenance_store=store,
        session_id=session_id,
        tool_executor=tool_executor,
    )


def _visible_results(events: list[StreamEvent]) -> list[ToolResultBlock]:
    """The SR2-visible tool-result blocks, in occurrence order.

    These are the blocks carried on the ``tool_result_received`` StreamEvents
    — i.e. AFTER ``_stamp_block`` and AFTER ``_RepeatGuard.observe`` may have
    replaced ``result.content`` with the repeat note (sr2-122/123 both mutate
    the executor's block in place). AC2 fixes the archive against the
    *SR2-visible* payload, so this — never the raw executor return object —
    is the comparison target for a TOOL_RESULT entry.
    """
    visible: list[ToolResultBlock] = []
    for event in events:
        if event.type == "tool_result_received" and event.tool_results:
            visible.extend(event.tool_results)
    return visible


def _iter(text: str | None, *calls: StreamEvent) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    if text:
        events.append(StreamEvent(type="text", text=text))
    events.extend(calls)
    events.append(StreamEvent(type="end"))
    return events


def _text_reply(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


def _guard(**overrides: Any):
    """Enabled repeat guard that stops via max_identical_calls on the 3rd
    identical call whatever the results (notes_before_stop=1 keeps the
    same-result note path out of the picture). RepeatGuardConfig instance —
    model_copy(update=...) bypasses validation, so pass the real model, not
    a dict."""
    from sr2.config.models import RepeatGuardConfig

    return RepeatGuardConfig(
        enabled=True, notes_before_stop=1, max_identical_calls=3, **overrides
    )


class RecordingExecutor:
    """Async executor returning scripted contents per tool_use id.

    The archived TOOL_RESULT must equal exactly the tool-result block *as SR2
    saw it*. ``returned`` keeps the objects this executor handed back, but
    they are not the SR2-visible payload: ``_stamp_block`` stamps
    ``meta["frame"]`` on them and ``_RepeatGuard.observe`` may replace
    ``content`` with the repeat note, both in place (sr2-122/123). Drivers
    that assert payload content therefore compare against the block carried
    on the ``tool_result_received`` StreamEvent (see ``_visible_results``);
    the raw returns stay here only to prove the executor actually ran.
    """

    def __init__(self, contents: dict[str, str]) -> None:
        self._contents = contents
        self.returned: list[ToolResultBlock] = []

    async def __call__(self, block: ToolUseBlock) -> ToolResultBlock:
        result = ToolResultBlock(tool_use_id=block.id, content=self._contents[block.id])
        self.returned.append(result)
        return result


async def _drain(sr2, text: str) -> None:
    events = [e async for e in sr2.turn(make_user_input(text), origin="")]
    assert any(e.type == "end" for e in events), (
        "turn did not complete — the archive contract is about completed turns"
    )


def _tool_turn_package(tool_executor):
    """LLM script for one tool-loop turn: iter0 streams text+tool_use, the
    (already-wired) executor returns a result, the second LLM call streams
    the final text."""
    return SequentialMockLLM(
        [_iter("Let me check", tool_use_event(tool_use_id="tc_1")), _text_reply("All sunny.")]
    )


_TURN0 = 0  # SR2's first turn is engine turn_seq 0


# ===========================================================================
# AC2-r1 — One tool-loop turn archives the exact ordered payload sequence.
# ===========================================================================


class TestAC2R1ExactOrderedSequence:
    async def test_tool_loop_turn_archives_ordered_payload_sequence(self):
        """AC2-r1: after one tool-loop turn, read_turn_archive(sid, 0) returns
        exactly [USER_INPUT, ASSISTANT_OUTPUT, TOOL_USE, TOOL_RESULT,
        ASSISTANT_OUTPUT] and each entry's content equals the SR2-visible
        payload item it records."""
        store = prov.InMemoryProvenanceStore()
        executor = RecordingExecutor({"tc_1": "sunny, 15C"})
        sr2 = _build_sr2(
            llm=_tool_turn_package, store=store, tool_executor=executor
        )
        await _drain(sr2, "weather please")

        entries = await read_turn_archive(store, sr2.session_id, _TURN0)
        reasons = [e.reason for e in entries]
        assert reasons == [
            prov.ContextReason.USER_INPUT,
            prov.ContextReason.ASSISTANT_OUTPUT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
            prov.ContextReason.ASSISTANT_OUTPUT,
        ], f"archive reasons must reproduce the turn event sequence exactly: {reasons!r}"

        # user input -> Message(role="user") with the SR2-visible input text
        _assert_message_text(entries[0], "user", "weather please", where="USER_INPUT")

        # intermediate assistant iteration (iter0 streamed text before the call)
        _assert_message_text(entries[1], "assistant", "Let me check", where="iter0 ASSISTANT_OUTPUT")

        # tool call -> the ToolUseBlock itself
        _assert_payload_equal(
            entries[2].content,
            ToolUseBlock(
                id="tc_1", name="get_weather", input={"location": "Oslo"}
            ),
            where="TOOL_USE",
        )

        # tool result -> the ToolResultBlock SR2 received, verbatim
        assert executor.returned, "executor never returned a result"
        _assert_payload_equal(entries[3].content, executor.returned[0], where="TOOL_RESULT")

        # final assistant output
        _assert_message_text(entries[4], "assistant", "All sunny.", where="final ASSISTANT_OUTPUT")


# ===========================================================================
# AC2-r2 — Close/reopen: the sequence and payloads survive durably.
# ===========================================================================


class TestAC2R2CloseReopenDurability:
    async def test_archive_survives_sqlite_close_and_reopen(self, tmp_path):
        """AC2-r2: turn on a connected SQLiteProvenanceStore; after close()
        and reopen on the same file, read_turn_archive returns an identical
        ordered sequence of (reason, content) pairs, block-equal to what was
        received before close."""
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "turn_archive_r2.db"
        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()

        executor = RecordingExecutor({"tc_1": "sunny, 15C"})
        sr2 = _build_sr2(
            llm=_tool_turn_package, store=store, tool_executor=executor
        )
        await _drain(sr2, "weather please")
        await sr2.aclose()

        before = await read_turn_archive(store, sr2.session_id, _TURN0)
        before_pairs = [(e.reason, e.content) for e in before]

        await store.close()

        store2 = SQLiteProvenanceStore(db_path=str(db_path))
        await store2.connect()
        after = await read_turn_archive(store2, sr2.session_id, _TURN0)
        after_pairs = [(e.reason, e.content) for e in after]

        try:
            assert len(after_pairs) == len(before_pairs), (
                f"entry count changed across close/reopen: "
                f"{len(before_pairs)} -> {len(after_pairs)}"
            )
            for (r_before, c_before), (r_after, c_after) in zip(
                before_pairs, after_pairs
            ):
                assert r_before == r_after, f"reason drifted: {r_before!r} -> {r_after!r}"
                assert _content_equal(c_before, c_after), (
                    f"content for {r_before!r} did not survive close/reopen: "
                    f"{c_before!r} vs {c_after!r}"
                )

            # ...and the reopened archive still equals what SR2 received.
            entries = after
            assert [e.reason for e in entries] == [
                prov.ContextReason.USER_INPUT,
                prov.ContextReason.ASSISTANT_OUTPUT,
                prov.ContextReason.TOOL_USE,
                prov.ContextReason.TOOL_RESULT,
                prov.ContextReason.ASSISTANT_OUTPUT,
            ]
            _assert_payload_equal(
                entries[3].content, executor.returned[0], where="TOOL_RESULT after reopen"
            )
        finally:
            await store2.close()


# ===========================================================================
# AC2-r3 — Upstream truncation markers survive verbatim.
# ===========================================================================


class TestAC2R3TruncationMarker:
    async def test_truncation_marker_archived_verbatim(self):
        """AC2-r3: a ToolResultBlock whose content ends with an upstream
        truncation marker is archived exactly, marker included; nothing
        strips the marker and no additional raw payload appears. The
        comparison target is the SR2-visible result block (post-stamping,
        post-guard), per the "SR2-visible ... exactly" contract."""
        truncated = "alpha beta gamma ... [truncated]"
        executor = RecordingExecutor({"tc_1": truncated})
        # iter0 carries text so its assistant iteration is transcribed like
        # every other iteration (amendment 4a); the row under test fixes the
        # tool-result payload, not the iteration's text shape.
        llm = SequentialMockLLM(
            [
                _iter(
                    "Reading the long output",
                    tool_use_event(tool_use_id="tc_1"),
                ),
                _text_reply("Done."),
            ]
        )
        store = prov.InMemoryProvenanceStore()
        sr2 = _build_sr2(llm=llm, store=store, tool_executor=executor)
        events = [e async for e in sr2.turn(make_user_input("go"), origin="")]
        assert any(e.type == "end" for e in events), "turn must complete"
        # The SR2-visible result block, post-stamping and post-guard: this is
        # the payload the TOOL_RESULT entry must equal (see _visible_results).
        visible = _visible_results(events)
        assert len(visible) == 1, (
            f"precondition: one executed tool call must produce exactly one "
            f"SR2-visible tool result, got {visible!r}"
        )

        entries = await read_turn_archive(store, sr2.session_id, _TURN0)
        tool_results = [
            e for e in entries if e.reason == prov.ContextReason.TOOL_RESULT
        ]
        assert len(tool_results) == 1, (
            "the turn made exactly one tool call — exactly one TOOL_RESULT entry "
            f"is expected, got {len(tool_results)}"
        )
        # Exactly the block SR2 received, marker included.
        _assert_payload_equal(tool_results[0].content, visible[0], where="TOOL_RESULT")
        archived_content = tool_results[0].content
        assert isinstance(archived_content, ToolResultBlock)
        assert archived_content.content.endswith("... [truncated]"), (
            f"truncation marker must survive in the archived payload, got "
            f"{archived_content.content!r}"
        )
        # Nothing strips the marker and no additional raw payload appears:
        # the turn's whole archive is exactly the 5-item SR2-visible
        # sequence.
        assert len(entries) == 5, (
            "the archive must contain exactly the SR2-visible sequence "
            "(user, assistant, tool_use, tool_result, assistant); got "
            f"{[e.reason for e in entries]!r}"
        )
        assert [e.reason for e in entries] == [
            prov.ContextReason.USER_INPUT,
            prov.ContextReason.ASSISTANT_OUTPUT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
            prov.ContextReason.ASSISTANT_OUTPUT,
        ]
        assert len(entries) == len({e.id for e in entries})


# ===========================================================================
# AC2-r4 — Multi-iteration interleaving, order durable across close/reopen.
# ===========================================================================


class TestAC2R4MultiIterationOrder:
    async def test_two_tool_iterations_interleave_in_occurrence_order(self, tmp_path):
        """AC2-r4: two tool-loop iterations then a final text iteration ->
        the archive interleaves iter0 assistant, iter0 tool_use, iter0
        tool_result, iter1 assistant, iter1 tool_use, iter1 tool_result,
        final assistant; the order is preserved across SQLite close/reopen."""
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "turn_archive_r4.db"
        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()

        executor = RecordingExecutor({"t0": "r0", "t1": "r1"})
        llm = SequentialMockLLM(
            [
                _iter("i0", tool_use_event(tool_use_id="t0", tool_name="tool_a")),
                _iter("i1", tool_use_event(tool_use_id="t1", tool_name="tool_b")),
                _text_reply("final"),
            ]
        )
        sr2 = _build_sr2(llm=llm, store=store, tool_executor=executor)
        await _drain(sr2, "go")
        await sr2.aclose()

        expected = [
            (prov.ContextReason.USER_INPUT, "go"),
            (prov.ContextReason.ASSISTANT_OUTPUT, "i0"),
            (prov.ContextReason.TOOL_USE, ToolUseBlock(id="t0", name="tool_a", input={"location": "Oslo"})),
            (prov.ContextReason.TOOL_RESULT, executor.returned[0]),
            (prov.ContextReason.ASSISTANT_OUTPUT, "i1"),
            (prov.ContextReason.TOOL_USE, ToolUseBlock(id="t1", name="tool_b", input={"location": "Oslo"})),
            (prov.ContextReason.TOOL_RESULT, executor.returned[1]),
            (prov.ContextReason.ASSISTANT_OUTPUT, "final"),
        ]

        await store.close()
        store2 = SQLiteProvenanceStore(db_path=str(db_path))
        await store2.connect()
        try:
            entries = await read_turn_archive(store2, sr2.session_id, _TURN0)
            assert len(entries) == len(expected), (
                f"expected {len(expected)} archived entries, got {len(entries)}"
            )
            for pos, (entry, (reason, payload)) in enumerate(zip(entries, expected)):
                assert entry.reason == reason, (
                    f"position {pos}: expected reason {reason!r}, got {entry.reason!r}"
                )
                if isinstance(payload, str):
                    role = "user" if reason == prov.ContextReason.USER_INPUT else "assistant"
                    _assert_message_text(entry, role, payload, where=f"position {pos}")
                else:
                    _assert_payload_equal(entry.content, payload, where=f"position {pos}")
        finally:
            await store2.close()


# ===========================================================================
# AC2-r7 — Control events are not archived; guard-stop endings still archive.
# ===========================================================================


class TestAC2R7ControlEventsNotArchived:
    async def test_empty_response_retry_archives_only_payload_events(self):
        """AC2-r7(i): an empty first response triggers the one-shot retry,
        then a text final. The retry StreamEvent proves the control event
        fired; the archive contains only the payload events — the final
        assistant output — and nothing representing retry telemetry."""
        llm = SequentialMockLLM([_text_reply(""), _text_reply("retried answer")])
        store = prov.InMemoryProvenanceStore()
        sr2 = _build_sr2(llm=llm, store=store)
        events = [e async for e in sr2.turn(make_user_input("go"), origin="")]
        assert any(e.type == "retry" for e in events), (
            "precondition: the empty first response must have produced the "
            "one-shot retry control event"
        )
        assert any(e.type == "end" for e in events)

        entries = await read_turn_archive(store, sr2.session_id, _TURN0)
        reasons = [e.reason for e in entries]
        assert reasons == [
            prov.ContextReason.USER_INPUT,
            prov.ContextReason.ASSISTANT_OUTPUT,
        ], f"only payload events may be archived; got {reasons!r}"
        _assert_message_text(entries[0], "user", "go", where="USER_INPUT")
        _assert_message_text(entries[1], "assistant", "retried answer", where="final ASSISTANT_OUTPUT")

    async def test_repeat_guard_stop_archives_executed_calls_and_forced_final(self):
        """AC2-r7(ii): a repeat-guard stop after identical calls. The
        repeat_guard / final_answer StreamEvents prove the control path ran;
        the archive still holds the executed tool calls/results in order and
        the forced final assistant output — and no entry represents
        repeat_guard or finish_reason telemetry. Tool results are compared to
        the SR2-visible blocks (post-stamp, post-guard), so whatever the
        guard left on the stopping call's result is what the archive must
        carry."""
        executor = RecordingExecutor({"c1": "same", "c2": "same", "c3": "same"})
        llm = SequentialMockLLM(
            [
                _iter(None, tool_use_event(tool_use_id="c1")),
                _iter(None, tool_use_event(tool_use_id="c2")),
                _iter(None, tool_use_event(tool_use_id="c3")),
                _text_reply("forced final"),
            ]
        )
        store = prov.InMemoryProvenanceStore()
        sr2 = _build_sr2(
            llm=llm, store=store, repeat_guard=_guard(), tool_executor=executor
        )
        events = [e async for e in sr2.turn(make_user_input("go"), origin="")]
        assert any(e.type == "repeat_guard" for e in events), (
            "precondition: the repeat guard must have stopped the run"
        )
        assert any(
            e.type == "final_answer" and (e.meta or {}).get("reason") == "repeat"
            for e in events
        ), "precondition: the forced final answer (reason=repeat) must have run"
        # The SR2-visible results, captured from the events the guard path
        # emitted: the comparison target for the TOOL_RESULT entries, post
        # _stamp_block and post-guard (sr2-122/123 mutate the executor's
        # block in place, so the raw return object is not what SR2 saw).
        visible = _visible_results(events)

        entries = await read_turn_archive(store, sr2.session_id, _TURN0)
        reasons = [e.reason for e in entries]
        # Payload events only: user input; the executed iterations each
        # contribute tool_use then tool_result (their assistant messages
        # carry no SR2-visible text, so an ASSISTANT_OUTPUT entry for them
        # is permitted, not required — control-shaped entries are what is
        # forbidden); the forced final assistant output.
        assert reasons[0] == prov.ContextReason.USER_INPUT
        assert reasons[-1] == prov.ContextReason.ASSISTANT_OUTPUT
        _assert_message_text(entries[-1], "assistant", "forced final", where="forced final")
        assert [r for r in reasons if r in (prov.ContextReason.TOOL_USE, prov.ContextReason.TOOL_RESULT)] == [
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
        ], f"every executed tool call/result must stay archived in order; got {reasons!r}"
        # Only the four vocabulary reasons — no control-event encoding
        # (repeat_guard / iteration_complete / finish_reason / retry).
        allowed = {
            prov.ContextReason.USER_INPUT,
            prov.ContextReason.ASSISTANT_OUTPUT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
        }
        assert all(r in allowed for r in reasons), f"unexpected archived reason: {reasons!r}"
        # Tool payloads verbatim: each TOOL_USE is the call block, each
        # TOOL_RESULT is the block SR2 received for it (post-stamp,
        # post-guard — the repeat note included where the guard replaced it).
        uses = [e.content for e in entries if e.reason == prov.ContextReason.TOOL_USE]
        assert [b.id for b in uses] == ["c1", "c2", "c3"]
        results = [e.content for e in entries if e.reason == prov.ContextReason.TOOL_RESULT]
        assert len(results) == len(visible) == 3, (
            f"each executed call archives one result; archived {len(results)} "
            f"against {len(visible)} SR2-visible results"
        )
        for call_id, archived, seen in zip(
            ("c1", "c2", "c3"), results, visible, strict=True
        ):
            assert _content_equal(archived, seen), (
                f"TOOL_RESULT for {call_id}: archived {archived!r} does not "
                f"equal the SR2-visible result block {seen!r}"
            )


# ===========================================================================
# AC2-r8 — Two sequential turns: per-turn archives, no bleed or duplication.
# ===========================================================================


class TestAC2R8SequentialTurns:
    async def test_two_turns_archive_separately_without_bleed(self):
        """AC2-r8: two sequential turns on one SR2 instance (turn_seq 0 then
        1); read_turn_archive(sid, 0) and (sid, 1) each return exactly that
        turn's events in order — no duplication, no cross-turn bleed."""
        executor = RecordingExecutor({"tc_1": "sunny, 15C"})
        llm = SequentialMockLLM(
            [
                _iter("Let me check", tool_use_event(tool_use_id="tc_1")),
                _text_reply("All sunny."),
                _iter("Second check", tool_use_event(tool_use_id="tc_1")),
                _text_reply("Still sunny."),
            ]
        )
        store = prov.InMemoryProvenanceStore()
        sr2 = _build_sr2(llm=llm, store=store, tool_executor=executor)
        await _drain(sr2, "first question")
        await _drain(sr2, "second question")
        await sr2.aclose()

        t0 = await read_turn_archive(store, sr2.session_id, 0)
        t1 = await read_turn_archive(store, sr2.session_id, 1)

        assert [e.reason for e in t0] == [
            prov.ContextReason.USER_INPUT,
            prov.ContextReason.ASSISTANT_OUTPUT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
            prov.ContextReason.ASSISTANT_OUTPUT,
        ]
        assert [e.reason for e in t1] == [
            prov.ContextReason.USER_INPUT,
            prov.ContextReason.ASSISTANT_OUTPUT,
            prov.ContextReason.TOOL_USE,
            prov.ContextReason.TOOL_RESULT,
            prov.ContextReason.ASSISTANT_OUTPUT,
        ]

        # Each turn's own user input; no bleed in either direction.
        _assert_message_text(t0[0], "user", "first question", where="turn 0 USER_INPUT")
        _assert_message_text(t1[0], "user", "second question", where="turn 1 USER_INPUT")
        _assert_message_text(t0[-1], "assistant", "All sunny.", where="turn 0 final")
        _assert_message_text(t1[-1], "assistant", "Still sunny.", where="turn 1 final")

        # No duplication anywhere in the session's archive.
        ids = [e.id for e in t0] + [e.id for e in t1]
        assert len(ids) == len(set(ids)), "the same entry appears in both turns"


# ===========================================================================
# AC2-r9 — Ordering durability: durable per-turn seq, not created_at/ids.
# ===========================================================================


# 26-char ULID-shaped ids (the store convention). In r9 they are generated
# in the shuffled write order, so ids are NOT correlated with seq — neither
# an id tie-break nor the identical created_at stamps can reproduce seq
# order; only the durable per-turn meta["seq"] can.


class TestAC2R9DurableOrderingKey:
    async def test_read_order_follows_durable_seq_not_created_at_or_id(self, tmp_path):
        """AC2-r9: archive-shaped entries written through the store with
        identical created_at stamps, distinct meta["seq"], shuffled write
        order and ids uncorrelated with seq; after SQLite reopen,
        read_turn_archive returns append (seq) order — the ordering key is
        the durable per-turn seq."""
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "turn_archive_r9.db"

        # Build 5 archive-shaped entries: seq 0..4. genesis (seq 0) is
        # written first so every referenced source id exists when written
        # (FR5); the remaining entries chain to genesis but are written in
        # shuffled order — ids and timestamps carry no ordering information.
        write_order = [2, 0, 4, 1, 3]
        stamp = datetime.now(tz=timezone.utc)

        genesis = prov.Entry(
            id=str(_ULID()),
            content=TextBlock(text="archive-shaped payload 0"),
            sources=(),
            origin=prov.EntryOrigin(kind="resolver", name="turn_archivist"),
            layer="turn_archive",
            session_id="session-ac2r9",
            created_at=stamp,
            meta={"turn_seq": 7, "seq": 0},
        )
        entries_by_seq: dict[int, prov.Entry] = {0: genesis}
        for seq in (1, 2, 3, 4):
            entries_by_seq[seq] = prov.Entry(
                id=str(_ULID()),
                content=TextBlock(text=f"archive-shaped payload {seq}"),
                sources=(genesis.id,),
                origin=prov.EntryOrigin(kind="transformer", name="turn_archivist"),
                layer="turn_archive",
                session_id="session-ac2r9",
                created_at=stamp,
                meta={"turn_seq": 7, "seq": seq},
            )

        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()
        # Durable writes through the store's public write path. SQLite
        # enforces the source FK per statement (sr2-92), so the genesis
        # entry is persisted first and every source reference already
        # exists when its referent is written — the same all-referenced-
        # ids-exist rule the archive itself must satisfy (FR5). The
        # ordering-relevant properties are untouched: write_order is
        # shuffled against seq, ids are uncorrelated, created_at identical.
        await store.write(genesis)
        for seq in write_order:
            if seq != 0:
                await store.write(entries_by_seq[seq])
        await store.close()

        store2 = SQLiteProvenanceStore(db_path=str(db_path))
        await store2.connect()
        try:
            entries = await read_turn_archive(store2, "session-ac2r9", 7)
            got = [e.meta.get("seq") for e in entries]
            assert got == [0, 1, 2, 3, 4], (
                "read_turn_archive must order by the durable per-turn seq; "
                f"got seqs {got!r} (write order was {write_order!r})"
            )
            # and the content follows the same order
            texts = [
                e.content.text if isinstance(e.content, TextBlock) else None
                for e in entries
            ]
            assert texts == [f"archive-shaped payload {s}" for s in (0, 1, 2, 3, 4)]
        finally:
            await store2.close()
