"""sr2-95 / AC3 — append-only continuity store and rebuildable projections
(unit layer, store half of the cage).

Case-table rows covered here: AC3-r1 .. AC3-r7, AC3-r10.
Rows AC3-r8, r9 (model frozenness) live in tests/continuity/test_models.py.

Contract the rows fix. Landing spot fixed by the bead's store pattern
(``SQLiteProvenanceStore`` lives at ``sr2.pipeline.stores.sqlite``):

    from sr2.continuity.store import ContinuityStore

    store = ContinuityStore(path=tmp_path / "continuity.db")
    await store.connect()                    # idempotent schema creation
    await store.append(record)               # append-only INSERT, never UPDATE
    views = await store.current_views()      # dict[str, ContinuityView]
    await store.rebuild_projections()        # deterministic rebuild from ledger
    events = await store.get_events(subject_id=None)   # full ledger, append order
    await store.close()

The rows assert ONLY through this public surface plus pydantic's public
``model_dump_json`` for serialized bytes. Method naming beyond the bead text
(``append`` / ``rebuild_projections`` are named by the bead; the read
helpers are the minimal surface the rows need to observe outcomes) is the
implementation's; the *behavior* — append-only persistence, byte-identical
earlier records, deterministic rebuild, reopen durability, projection
arithmetic — is fixed.

Projection arithmetic pinned by AC3-r10 (FR4 + FR13):
  - current_record   = latest event for the subject (ledger append order)
  - status           = "suppressed"  if latest event is SUPPRESSED
                       "superseded"  if the latest record is superseded by a
                                     later event naming it in ``supersedes``
                       "dormant"     if latest event is DORMANT
                       "active"      otherwise
  - a record's ``supersedes`` ids are excluded from other subjects'
    current_record selection; SUPPRESSED excludes; UNSUPPRESSED re-includes;
  - history_ids      = the subject's event ids in ledger order.

Red mechanism (deliberate): module-level import of the not-yet-existing
store module; until C implements it, collection fails with ImportError
naming exactly that module.
"""

from __future__ import annotations

import pytest

from continuity._fixtures import (
    ContinuityEventKind,
    ContinuityRecord,
    ContinuityView,
    make_record,
    serialize,
)

from sr2.continuity.store import ContinuityStore


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def open_store(tmp_path, name: str) -> ContinuityStore:
    store = ContinuityStore(path=tmp_path / name)
    await store.connect()
    return store


def ev(ordinal: int, subject: str, kind: "ContinuityEventKind", **kw) -> "ContinuityRecord":
    """Deterministic record factory: id encodes append slot and subject."""
    return make_record(
        f"rec-{ordinal:04d}-{subject}", subject, kind,
        created_at_offset_seconds=ordinal, **kw
    )


async def ledger_bytes(store: ContinuityStore) -> dict[str, str]:
    """{record_id: serialized bytes} for every event in the ledger."""
    events = await store.get_events()
    return {r.id: serialize(r) for r in events}


def expect_status(view: "ContinuityView", status: str) -> None:
    assert isinstance(view, ContinuityView)
    assert view.status == status, (
        f"expected status {status!r}, got {view.status!r}"
    )


# ---------------------------------------------------------------------------
# AC3-r1 — FORMED then SUPERSEDED: both stored, old view flips to
#           "superseded", FORMED bytes unchanged
# ---------------------------------------------------------------------------


class TestSupersedeAppendsAndPreserves:
    async def test_superseded_event_preserves_formed_record_bytes(self, tmp_path):
        """AC3-r1: append FORMED for subject 'backend-choice', then a
        SUPERSEDED event for the same subject. Both events stored; the
        ContinuityView shows the superseding record as current with status
        'superseded' semantics on the old record's projection; the FORMED
        record's serialized bytes are identical before and after the
        second append."""
        store = await open_store(tmp_path, "r1.db")
        try:
            formed = ev(1, "backend-choice", ContinuityEventKind.FORMED)
            await store.append(formed)
            formed_bytes_before = serialize(formed)

            superseding = ev(
                2,
                "backend-choice",
                ContinuityEventKind.SUPERSEDED,
                content="we moved to MySQL",
            )
            await store.append(superseding)

            # Both events stored.
            events = await store.get_events(subject_id="backend-choice")
            stored = {r.id: r for r in events}
            assert formed.id in stored, "FORMED event missing after second append"
            assert superseding.id in stored, "SUPERSEDED event not stored"

            # FORMED record's stored serialization unchanged (byte-identical).
            assert serialize(stored[formed.id]) == formed_bytes_before

            # Projection: the ContinuityView shows the superseding record
            # as current_record and the view's status is "superseded" —
            # the case table's exact wording for this row ("shows the
            # superseding record as current_record with status
            # 'superseded' for the old"). AC3-r10 pins the rest of the
            # status arithmetic.
            views = await store.current_views()
            assert "backend-choice" in views
            view = views["backend-choice"]
            assert view.current_record.id == superseding.id
            expect_status(view, "superseded")
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC3-r2 — FORMED then SUPPRESSED: status "suppressed", bytes unchanged
# ---------------------------------------------------------------------------


class TestSuppressAppendsAndPreserves:
    async def test_suppressed_event_preserves_formed_record_bytes(self, tmp_path):
        """AC3-r2: append FORMED, then SUPPRESSED for the same subject.
        Both events stored; ContinuityView status is 'suppressed'; FORMED
        record's serialization unchanged."""
        store = await open_store(tmp_path, "r2.db")
        try:
            formed = ev(1, "nickname", ContinuityEventKind.FORMED)
            await store.append(formed)
            formed_bytes_before = serialize(formed)

            suppression = ev(2, "nickname", ContinuityEventKind.SUPPRESSED)
            await store.append(suppression)

            events = await store.get_events(subject_id="nickname")
            stored = {r.id: r for r in events}
            assert formed.id in stored and suppression.id in stored
            assert serialize(stored[formed.id]) == formed_bytes_before

            views = await store.current_views()
            expect_status(views["nickname"], "suppressed")
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC3-r3 — full 7-event lifecycle; every earlier event's bytes stay
#           identical after each subsequent append
# ---------------------------------------------------------------------------


class TestFullLifecycleByteIdentity:
    async def test_seven_event_lifecycle_all_bytes_stable(self, tmp_path):
        """AC3-r3: append FORMED, REINFORCED, VERIFIED, SUPERSEDED, DORMANT,
        SUPPRESSED, UNSUPPRESSED for one subject. All 7 events stored; each
        earlier event's serialized bytes are identical before and after
        each subsequent append."""
        kinds = [
            ContinuityEventKind.FORMED,
            ContinuityEventKind.REINFORCED,
            ContinuityEventKind.VERIFIED,
            ContinuityEventKind.SUPERSEDED,
            ContinuityEventKind.DORMANT,
            ContinuityEventKind.SUPPRESSED,
            ContinuityEventKind.UNSUPPRESSED,
        ]
        store = await open_store(tmp_path, "r3.db")
        try:
            for i, kind in enumerate(kinds, start=1):
                record = ev(i, "lifecycle", kind)
                # Snapshot of everything appended so far, taken BEFORE this append.
                snapshot_before = await ledger_bytes(store)
                await store.append(record)
                snapshot_after = await ledger_bytes(store)

                # Every earlier event: byte-identical after this append.
                for rec_id, blob in snapshot_before.items():
                    assert snapshot_after.get(rec_id) == blob, (
                        f"event {rec_id} changed bytes when appending "
                        f"{record.id} ({kind.value}) — store is not append-only"
                    )
                # This event now exists with its own bytes.
                assert snapshot_after.get(record.id) == serialize(record)

            events = await store.get_events(subject_id="lifecycle")
            assert len(events) == 7, (
                f"all 7 lifecycle events must be stored, got {len(events)}"
            )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC3-r4 — durability: close, reopen, read all; bytes identical
# ---------------------------------------------------------------------------


class TestReopenDurability:
    async def test_events_survive_close_and_reopen_byte_identical(self, tmp_path):
        """AC3-r4: append events, close the store, reopen, read all events.
        All events present; each record's serialized bytes identical to the
        values captured before close."""
        path = tmp_path / "r4.db"
        store = ContinuityStore(path=path)
        await store.connect()
        try:
            appended = [
                ev(1, "alpha", ContinuityEventKind.FORMED),
                ev(2, "beta", ContinuityEventKind.FORMED),
                ev(3, "alpha", ContinuityEventKind.REINFORCED, confidence=0.95),
                ev(4, "beta", ContinuityEventKind.SUPPRESSED),
            ]
            for record in appended:
                await store.append(record)
            before_close = {r.id: serialize(r) for r in appended}
        finally:
            await store.close()

        reopened = ContinuityStore(path=path)
        await reopened.connect()
        try:
            events = await reopened.get_events()
            after_reopen = {r.id: serialize(r) for r in events}
            assert set(after_reopen) == set(before_close), (
                "reopen must expose exactly the appended events"
            )
            for rec_id, blob in before_close.items():
                assert after_reopen[rec_id] == blob, (
                    f"event {rec_id} bytes differ after reopen — durable "
                    "storage must round-trip records byte-identically"
                )
            # Round-tripped records reconstruct the same model objects.
            by_id = {r.id: r for r in events}
            for record in appended:
                assert by_id[record.id] == record
        finally:
            await reopened.close()


# ---------------------------------------------------------------------------
# AC3-r5 — deterministic rebuild: two independent rebuilds are identical
# ---------------------------------------------------------------------------


class TestDeterministicRebuild:
    async def test_rebuild_projections_twice_is_byte_identical(self, tmp_path):
        """AC3-r5: rebuild projections from the event log twice
        independently; both projection sets are byte-identical."""
        store = await open_store(tmp_path, "r5.db")
        try:
            await store.append(ev(1, "alpha", ContinuityEventKind.FORMED))
            await store.append(ev(2, "beta", ContinuityEventKind.FORMED))
            await store.append(
                ev(3, "alpha", ContinuityEventKind.SUPERSEDED,
                   supersedes=("rec-0001-alpha",))
            )
            await store.append(ev(4, "beta", ContinuityEventKind.DORMANT))
            await store.append(ev(5, "gamma", ContinuityEventKind.FORMED))

            first = await store.rebuild_projections()
            second = await store.rebuild_projections()

            def dump(views) -> dict[str, str]:
                assert isinstance(views, dict), (
                    "rebuild_projections() must return the projection set "
                    "keyed by subject_id"
                )
                return {
                    subject: view.model_dump_json()
                    for subject, view in views.items()
                }

            d1, d2 = dump(first), dump(second)
            assert d1 == d2, (
                "two independent rebuilds from the same immutable ledger "
                "produced different projections — rebuild is not deterministic"
            )
            # Sanity: the rebuilt set covers every subject in the ledger.
            assert set(d1) == {"alpha", "beta", "gamma"}
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC3-r6 — subject isolation: chains do not interfere
# ---------------------------------------------------------------------------


class TestSubjectIsolation:
    async def test_two_subjects_have_independent_chains_and_views(self, tmp_path):
        """AC3-r6: append FORMED for subject 'A' and FORMED for subject
        'B'. Each subject has its own event chain; view for 'A' unaffected
        by B's events and vice versa."""
        store = await open_store(tmp_path, "r6.db")
        try:
            rec_a = ev(1, "A", ContinuityEventKind.FORMED, content="A says one")
            await store.append(rec_a)
            view_a_after_own = (await store.current_views())["A"]

            rec_b = ev(2, "B", ContinuityEventKind.FORMED, content="B says two")
            await store.append(rec_b)

            views = await store.current_views()
            assert set({"A", "B"}) <= set(views)

            # B's append did not disturb A's view.
            assert views["A"].current_record.id == rec_a.id
            assert views["A"].status == view_a_after_own.status
            assert tuple(views["A"].history_ids) == (rec_a.id,)

            # A's events never appear in B's chain.
            assert views["B"].current_record.id == rec_b.id
            assert tuple(views["B"].history_ids) == (rec_b.id,)

            # Per-subject event reads stay in their own chain.
            events_a = await store.get_events(subject_id="A")
            events_b = await store.get_events(subject_id="B")
            assert [r.id for r in events_a] == [rec_a.id]
            assert [r.id for r in events_b] == [rec_b.id]
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC3-r7 — superseding is purely additive; the old record never mutates
# ---------------------------------------------------------------------------


class TestSupersedeIsPurelyAdditive:
    async def test_supersedes_link_leaves_old_record_bytes_identical(self, tmp_path):
        """AC3-r7: append a record with supersedes=(old_id,), verify the old
        record's bytes remain identical after the new append."""
        store = await open_store(tmp_path, "r7.db")
        try:
            old = ev(1, "db-choice", ContinuityEventKind.FORMED,
                     content="Postgres 16")
            await store.append(old)
            old_bytes_before = serialize(old)

            new = ev(2, "db-choice", ContinuityEventKind.FORMED,
                     content="Postgres 17", supersedes=(old.id,))
            await store.append(new)

            # Old record still retrievable, byte-identical, unmutated.
            stored_old = {r.id: r for r in await store.get_events()}[old.id]
            assert serialize(stored_old) == old_bytes_before
            assert stored_old == old  # full model equality, not just bytes

            # New record present and linking.
            stored_new = {r.id: r for r in await store.get_events()}[new.id]
            assert tuple(stored_new.supersedes) == (old.id,)

            # Additivity: exactly two events in the ledger.
            assert len(await store.get_events()) == 2
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# AC3-r10 — projection arithmetic across subjects
# ---------------------------------------------------------------------------


class TestProjectionArithmetic:
    async def test_current_views_apply_supersede_suppress_unsuppress_dormant(
        self, tmp_path
    ):
        """AC3-r10: append events for multiple subjects, then query current
        projections; each shows the latest non-superseded, non-suppressed
        record as current. SUPERSEDED excludes old, SUPPRESSED excludes,
        UNSUPPRESSED re-includes, DORMANT shows status 'dormant'."""
        store = await open_store(tmp_path, "r10.db")
        try:
            # Subject "lang": FORMED py -> FORMED rust superseding py.
            py = ev(1, "lang", ContinuityEventKind.FORMED, content="python")
            rust = ev(2, "lang", ContinuityEventKind.FORMED,
                      content="rust", supersedes=(py.id,))
            # Subject "secret": FORMED then SUPPRESSED then UNSUPPRESSED.
            secret = ev(3, "secret", ContinuityEventKind.FORMED,
                        content="home town")
            await store.append(py)
            await store.append(rust)
            await store.append(secret)
            await store.append(ev(4, "secret", ContinuityEventKind.SUPPRESSED))
            await store.append(ev(5, "secret", ContinuityEventKind.UNSUPPRESSED))
            # Subject "habit": FORMED then DORMANT.
            habit = ev(6, "habit", ContinuityEventKind.FORMED,
                       content="gym at 6am")
            await store.append(habit)
            await store.append(ev(7, "habit", ContinuityEventKind.DORMANT))
            # Subject "quiet": single FORMED — plain active baseline.
            quiet = ev(8, "quiet", ContinuityEventKind.FORMED,
                       content="prefers email")
            await store.append(quiet)

            views = await store.current_views()
            assert set(views) == {"lang", "secret", "habit", "quiet"}

            # lang: superseded record excluded; the superseding record is current.
            assert views["lang"].current_record.id == rust.id
            assert views["lang"].current_record.id != py.id

            # secret: UNSUPPRESSED re-includes — content is the live record again.
            assert views["secret"].current_record.id == secret.id
            assert views["secret"].status != "suppressed"

            # habit: DORMANT surfaces as status "dormant".
            assert views["habit"].status == "dormant"

            # quiet: untouched baseline is active with its record current.
            assert views["quiet"].current_record.id == quiet.id
            assert views["quiet"].status == "active"

            # FR13 cross-check on lang's chain: the superseded record is
            # excluded from current but the chain (history) still holds it.
            assert py.id in tuple(views["lang"].history_ids)
            assert rust.id in tuple(views["lang"].history_ids)
        finally:
            await store.close()

    async def test_suppressed_beats_earlier_dormant_until_unsuppressed(
        self, tmp_path
    ):
        """AC3-r10 support: status follows the latest event — DORMANT then
        SUPPRESSED reads 'suppressed'; after UNSUPPRESSED the record is no
        longer suppressed. Projection arithmetic must be event-order
        driven, not first-event driven."""
        store = await open_store(tmp_path, "r10b.db")
        try:
            await store.append(ev(1, "mood", ContinuityEventKind.FORMED))
            await store.append(ev(2, "mood", ContinuityEventKind.DORMANT))
            expect_status((await store.current_views())["mood"], "dormant")

            await store.append(ev(3, "mood", ContinuityEventKind.SUPPRESSED))
            expect_status((await store.current_views())["mood"], "suppressed")

            await store.append(ev(4, "mood", ContinuityEventKind.UNSUPPRESSED))
            after = (await store.current_views())["mood"]
            assert after.status != "suppressed"
        finally:
            await store.close()
