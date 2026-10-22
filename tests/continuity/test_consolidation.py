"""sr2-96 / AC5, AC6, AC7 — typed consolidation with authority discipline and
supersession (unit layer of the consolidation cage).

Landing spot fixed by the bead: ``sr2.continuity.consolidation``.

Case table: one section header per row (AC5-r1 .. AC5-r5, AC6-r1 .. AC6-r5,
AC7-r1, AC7-r2); each header cites the AC clause the row holds, and every
test function is named ``test_<ac>_r<N>_*`` for its row.

Red mechanism (deliberate): the module-level import below of the not-yet-existing
module; until C creates it, collection of this file fails with an ImportError
naming exactly ``sr2.continuity.consolidation`` (the sr2-94/sr2-95 pattern).

============================================================================
INTERFACE CONTRACT this cage fixes (everything else in the module is free)
============================================================================

    class ConsolidationRequest(BaseModel, frozen=True):      # spec verbatim
        turn_id: str
        entry_ids: tuple[str, ...]
        owner_id: str
        agent_id: str
        project_hints: tuple[str, ...] = ()

    class RuleBasedContinuityConsolidator:
        def __init__(self, provenance: ProvenanceStore) -> None: ...
        async def consolidate(
            self, request: ConsolidationRequest
        ) -> list[ContinuityRecord]: ...        # implements the Protocol

    # factual-current selection (store.py is frozen and does NOT filter by
    # authority, so the predicate lives in this module):
    def is_factual_current(record_or_view) -> bool: ...
    def factual_current_records(records: Iterable) -> list: ...
    #   includes USER / LIVE_SOURCE / VERIFIED authority,
    #   excludes AGENT_TENTATIVE and DERIVED;
    #   accepts a ContinuityRecord or a ContinuityView (inspects its
    #   current_record); deterministic and stateless.

    # promotion (AC6): a module-level confirmation-marker regex so the marker
    # vocabulary is public, not hidden in a private helper:
    CONFIRMATION_PATTERN  # matches a leading "confirmed" / "that's correct"

============================================================================
DETERMINISTIC RULES the rows pin (all observable through the public surface)
============================================================================

Authority (AC6 core): a record's authority is copied from the source entry's
``authority`` — USER entry -> USER, AGENT_TENTATIVE entry -> AGENT_TENTATIVE
(tentative episode), LIVE_SOURCE entry -> LIVE_SOURCE.

Typing — trigger phrases matched case-insensitively against the entry's text:

    user, LIVE_SOURCE       "i prefer" / "i like"        -> PREFERENCE
    user                    "i always" / "i never"       -> PERSONAL_FACT
    user, LIVE_SOURCE       "i decide" / "we decide"     -> DECISION
                            (rationale text -> meta["rationale"])
    user                    "i want" / "goal"            -> GOAL
    user                    "next, i" / "next i will"/"then i will"
                                                          -> NEXT_ACTION
    user                    "i will" (no other match)    -> COMMITMENT
    user                    "must" / "cannot" / "never deploy"
                                                          -> CONSTRAINT
    user, LIVE_SOURCE       "correction:"                -> CORRECTION
                            (subject id taken from
                             meta["supersedes_subject"] when present)
    user / LIVE_SOURCE,
    no other match                                       -> PROJECT_FACT
    AGENT_TENTATIVE         any claim                    -> EPISODE
                            (subject id taken from
                             meta["about_subject"] when present)

Scope: for every record ``scope.owner_id == request.owner_id`` and
``scope.agent_id == request.agent_id``. Project association follows FR11:
personal-scope kinds (PREFERENCE, PERSONAL_FACT, GOAL, NEXT_ACTION,
COMMITMENT) always carry ZERO projects (valid for personal/global knowledge,
hints or not); project-scope kinds (DECISION, CONSTRAINT, PROJECT_FACT,
CORRECTION, EPISODE) carry ``tuple(sorted(request.project_hints))``.

Provenance (FR5): every returned record's ``provenance_ids`` is exactly the id
of the one real resolved source entry — never synthetic.

Supersession (AC7): a CORRECTION whose source entry meta carries
``supersedes_subject`` gets a ``supersedes`` tuple naming the current record id
of that subject **in the store** at consolidate-time (empty tuple when the
subject is unknown to the store). It appends; it never mutates.

Unresolvable ids in ``request.entry_ids`` contribute no records; a request
whose entries resolve to nothing returns [].
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from continuity._fixtures import (
    ContextAuthority,
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
    make_record,
    serialize,
)
from sr2.continuity.consolidation import (
    CONFIRMATION_PATTERN,
    ConsolidationRequest,
    RuleBasedContinuityConsolidator,
    factual_current_records,
    is_factual_current,
)
from sr2.continuity.store import ContinuityStore
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.provenance import (
    ContextReason,
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers — turn archives built the way the orchestrator's turn archivist
# builds them (layer "turn_archive", meta turn_seq/seq, genesis-first sources,
# authority stamped USER for user input / AGENT_TENTATIVE for assistant
# output), written through InMemoryProvenanceStore.write so FR5 ordering is
# exercised for real.
# ---------------------------------------------------------------------------

_ORIGIN_GENESIS = EntryOrigin(kind="resolver", name="turn_archivist")
_ORIGIN_DERIVED = EntryOrigin(kind="transformer", name="turn_archivist")
_CLOCK = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class TurnArchiveBuilder:
    """Appends one session's turn archive into a provenance store."""

    def __init__(self, store: InMemoryProvenanceStore, session_id: str) -> None:
        self._store = store
        self.session_id = session_id
        self.turn_seq = -1
        self._seq = 0
        self._prev_id: str | None = None

    async def _append(
        self,
        content,
        reason: ContextReason,
        authority: ContextAuthority,
        meta: dict | None = None,
    ) -> str:
        genesis = self._prev_id is None
        entry = Entry(
            id=f"e-{self._seq:03d}",
            content=content,
            sources=() if genesis else (self._prev_id,),
            origin=_ORIGIN_GENESIS if genesis else _ORIGIN_DERIVED,
            layer="turn_archive",
            session_id=self.session_id,
            created_at=_CLOCK,
            reason=reason,
            authority=authority,
            meta={"turn_seq": self.turn_seq, "seq": self._seq, **(meta or {})},
        )
        await self._store.write(entry)
        self._prev_id = entry.id
        self._seq += 1
        return entry.id

    async def next_turn(self) -> None:
        self.turn_seq += 1

    async def user(self, text: str, meta: dict | None = None) -> str:
        return await self._append(
            Message(role="user", content=[TextBlock(text=text)]),
            ContextReason.USER_INPUT,
            ContextAuthority.USER,
            meta,
        )

    async def assistant(self, text: str, meta: dict | None = None) -> str:
        return await self._append(
            Message(role="assistant", content=[TextBlock(text=text)]),
            ContextReason.ASSISTANT_OUTPUT,
            ContextAuthority.AGENT_TENTATIVE,
            meta,
        )

    async def tool_use(self, tool_name: str, tool_input: dict) -> str:
        return await self._append(
            ToolUseBlock(id=f"tu-{self._seq:03d}", name=tool_name, input=tool_input),
            ContextReason.TOOL_USE,
            ContextAuthority.AGENT_TENTATIVE,
        )

    async def live(self, tool_use_id: str, text: str, meta: dict | None = None) -> str:
        return await self._append(
            ToolResultBlock(tool_use_id=tool_use_id, content=text),
            ContextReason.LIVE_SOURCE,
            ContextAuthority.LIVE_SOURCE,
            meta,
        )


async def open_store(tmp_path, name: str) -> ContinuityStore:
    store = ContinuityStore(path=tmp_path / name)
    await store.connect()
    return store


def request_for(
    entry_ids: list[str],
    *,
    project_hints: tuple[str, ...] = (),
    turn_id: str = "turn-0",
) -> "ConsolidationRequest":
    return ConsolidationRequest(
        turn_id=turn_id,
        entry_ids=tuple(entry_ids),
        owner_id="owner-1",
        agent_id="agent-1",
        project_hints=project_hints,
    )


async def consolidate_fixture(
    tmp_path,
    builder: TurnArchiveBuilder,
    entry_ids: list[str],
    *,
    project_hints: tuple[str, ...] = (),
    db_name: str = "consolidation.db",
) -> tuple[list[ContinuityRecord], ContinuityStore, dict[str, str]]:
    """Consolidate *entry_ids* and append every returned record into a fresh
    continuity store. Returns (records, opened store, {entry_id: kind})."""
    consolidator = RuleBasedContinuityConsolidator(provenance=builder_store(builder))
    records = await consolidator.consolidate(
        request_for(entry_ids, project_hints=project_hints)
    )
    store = await open_store(tmp_path, db_name)
    for record in records:
        await store.append(record)
    return records, store, kind_map(records)


def kind_map(records: list[ContinuityRecord]) -> dict[str, ContinuityKind]:
    """{record.id: kind} — record ids are unique because each derives from a
    unique source entry id; raises (via the caller's assertions) otherwise."""
    return {r.id: r.kind for r in records}


def builder_store(builder: TurnArchiveBuilder) -> InMemoryProvenanceStore:
    # Private attribute access on a TEST-ONLY helper object (the builder we
    # constructed), never on production code.
    return builder._store  # noqa: SLF001


def texts_of(records: list[ContinuityRecord]) -> list[str]:
    return [r.content for r in records]


def sole(records: list[ContinuityRecord], kind: ContinuityKind) -> ContinuityRecord:
    hits = [r for r in records if r.kind is kind]
    assert len(hits) == 1, (
        f"expected exactly one {kind.value} record, got {len(hits)}: "
        f"{texts_of(hits)}"
    )
    return hits[0]


# AC5 conversation: NO remember command anywhere — typed inference is automatic.
AC5_TURN = [
    ("user", "I prefer dark roast coffee in the morning.", {}),
    ("user", "I always take meetings before noon.", {}),
    (
        "user",
        "We decided to adopt SQLite for the ledger because aiosqlite fits our async stack.",
        {},
    ),
    ("user", "I want to ship the continuity epic by Friday.", {}),
    ("user", "Next, I will write the consolidation cage.", {}),
    ("user", "I will send the invoice today.", {}),
    ("user", "We must never deploy on Fridays.", {}),
    ("live", "main is at commit d8cc032", {}),
    (
        "user",
        "correction: we adopted SQLite, not Postgres.",
        {"supersedes_subject": "subject-no-such-yet"},
    ),
    ("assistant", "I believe the cache defaults to redis.", {"about_subject": "subject-cache-backend"}),
]


async def build_ac5_turn(tmp_path, session_id: str = "sess-ac5"):
    store = InMemoryProvenanceStore()
    builder = TurnArchiveBuilder(store, session_id)
    await builder.next_turn()
    ids: list[str] = []
    for row in AC5_TURN:
        kind, text, meta = row
        if kind == "user":
            ids.append(await builder.user(text, meta))
        elif kind == "assistant":
            ids.append(await builder.assistant(text, meta))
        else:
            tu = await builder.tool_use("git_head", {"repo": "sr2"})
            ids.append(await builder.live(tu, text, meta))
    return builder, ids


# ---------------------------------------------------------------------------
# AC5-r1 — request contract: the spec's exact five fields, frozen semantics,
#          project_hints defaults to empty and accepts the empty tuple
#          (zero-project personal knowledge is valid — FR11).
# ---------------------------------------------------------------------------


class TestRequestContract:
    def test_ac5_r1_request_fields_defaults_and_frozen(self):
        req = ConsolidationRequest(
            turn_id="t-1",
            entry_ids=("e-000", "e-001"),
            owner_id="owner-1",
            agent_id="agent-1",
        )
        assert req.turn_id == "t-1"
        assert req.entry_ids == ("e-000", "e-001")
        assert req.owner_id == "owner-1"
        assert req.agent_id == "agent-1"
        assert req.project_hints == ()

        hinted = req.model_copy(update={"project_hints": ("sr2",)})
        assert hinted.project_hints == ("sr2",)

        with pytest.raises(ValidationError):
            req.turn_id = "mutated"

    async def test_ac5_r1_empty_entry_ids_yields_no_records(self, tmp_path):
        builder, _ = await build_ac5_turn(tmp_path)
        consolidator = RuleBasedContinuityConsolidator(provenance=builder_store(builder))
        assert await consolidator.consolidate(request_for([])) == []


# ---------------------------------------------------------------------------
# AC5-r2 — automatic typed inference over one ordinary conversation: every
#          typed kind the bead lists is produced without any remember command.
# ---------------------------------------------------------------------------


class TestAutomaticTypedInference:
    async def test_ac5_r2_conversation_yields_every_typed_kind(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        records, _store, kinds = await consolidate_fixture(
            tmp_path, builder, ids, db_name="ac5-r2.db"
        )

        assert len(ids) == len(AC5_TURN)
        expected = {
            ContinuityKind.PREFERENCE,
            ContinuityKind.PERSONAL_FACT,
            ContinuityKind.DECISION,
            ContinuityKind.GOAL,
            ContinuityKind.NEXT_ACTION,
            ContinuityKind.COMMITMENT,
            ContinuityKind.CONSTRAINT,
            ContinuityKind.PROJECT_FACT,
            ContinuityKind.CORRECTION,
            ContinuityKind.EPISODE,
        }
        assert set(kinds.values()) == expected, (
            f"typed inference mismatch:\n"
            + "\n".join(f"  {r.kind.value}: {r.content}" for r in records)
        )


# ---------------------------------------------------------------------------
# AC5-r3 — correct sourcing: every record's provenance_ids names exactly the
#          real source entry id (FR5 — never synthetic), and each source entry
#          yields exactly one record (deterministic, idempotent classification).
# ---------------------------------------------------------------------------


class TestSourcing:
    async def test_ac5_r3_provenance_ids_reference_real_entries(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        records, _store, _kinds = await consolidate_fixture(
            tmp_path, builder, ids, db_name="ac5-r3.db"
        )
        src = builder_store(builder)
        for record in records:
            assert len(record.provenance_ids) == 1, (
                f"record {record.id!r} must name exactly one source entry"
            )
            source_id = record.provenance_ids[0]
            assert source_id in ids, (
                f"provenance id {source_id!r} is not a real archived entry id"
            )
            entry = await src.get(source_id)
            assert entry is not None
            assert entry.session_id == builder.session_id

        # one record per source entry — no duplicates, no synthetic extra rows
        assert len(records) == len(ids)
        provenance = [r.provenance_ids[0] for r in records]
        assert sorted(provenance) == sorted(ids)


# ---------------------------------------------------------------------------
# AC5-r4 — scope classification (FR11): personal/global records carry zero
#          projects even when hints exist; project records carry the hint.
# ---------------------------------------------------------------------------

PERSONAL_KINDS = {
    ContinuityKind.PREFERENCE,
    ContinuityKind.PERSONAL_FACT,
    ContinuityKind.GOAL,
    ContinuityKind.NEXT_ACTION,
    ContinuityKind.COMMITMENT,
}

PROJECT_KINDS = {
    ContinuityKind.DECISION,
    ContinuityKind.CONSTRAINT,
    ContinuityKind.PROJECT_FACT,
    ContinuityKind.CORRECTION,
    ContinuityKind.EPISODE,
}


class TestScopeClassification:
    async def test_ac5_r4_personal_scope_zero_projects(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        # hints ARE supplied: personal records must stay project-free anyway
        records, _store, _kinds = await consolidate_fixture(
            tmp_path, builder, ids, project_hints=("sr2",), db_name="ac5-r4a.db"
        )
        personal = [r for r in records if r.kind in PERSONAL_KINDS]
        assert personal, "fixture must produce personal-scope records"
        for record in personal:
            assert record.scope.owner_id == "owner-1"
            assert record.scope.agent_id == "agent-1"
            assert record.scope.projects == (), (
                f"personal record {record.kind.value} must carry zero projects, "
                f"got {record.scope.projects}"
            )

    async def test_ac5_r4_project_records_carry_hinted_project(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        records, _store, _kinds = await consolidate_fixture(
            tmp_path, builder, ids, project_hints=("sr2",), db_name="ac5-r4b.db"
        )
        project = [r for r in records if r.kind in PROJECT_KINDS]
        assert project, "fixture must produce project-scope records"
        for record in project:
            assert record.scope.owner_id == "owner-1"
            assert record.scope.agent_id == "agent-1"
            assert record.scope.projects == ("sr2",), (
                f"project record {record.kind.value} must carry the hinted "
                f"project, got {record.scope.projects}"
            )
        # zero hints -> project kinds also carry zero projects (hint-driven)
        records2, _store2, _kinds2 = await consolidate_fixture(
            tmp_path, builder, ids, project_hints=(), db_name="ac5-r4c.db"
        )
        for record in records2:
            assert record.scope.projects == ()


# ---------------------------------------------------------------------------
# AC5-r5 — rationale: a decision made with a "because" clause carries that
#          rationale on the record (spec FR11: "decision with rationale").
# ---------------------------------------------------------------------------


class TestDecisionRationale:
    async def test_ac5_r5_decision_carries_rationale(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        decision_entry_id = ids[2]  # "We decided ... because aiosqlite ..."
        records, _store, _kinds = await consolidate_fixture(
            tmp_path, builder, [decision_entry_id], project_hints=("sr2",),
            db_name="ac5-r5.db",
        )
        decision = sole(records, ContinuityKind.DECISION)
        assert decision.provenance_ids == (decision_entry_id,)
        rationale = decision.meta.get("rationale")
        assert isinstance(rationale, str) and rationale, (
            "decision record must carry its rationale text in meta['rationale']"
        )
        assert "aiosqlite" in rationale


# ---------------------------------------------------------------------------
# AC6-r1 — authority discipline at formation: user entry -> USER, assistant
#          entry -> AGENT_TENTATIVE tentative episode, live tool result ->
#          LIVE_SOURCE. Authority is deterministic from the source entry.
# ---------------------------------------------------------------------------


class TestAuthorityAtFormation:
    async def test_ac6_r1_authority_copied_from_source_entry(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        user_entry, assistant_entry, live_entry = ids[0], ids[9], ids[7]
        records, _store, _kinds = await consolidate_fixture(
            tmp_path,
            builder,
            [user_entry, assistant_entry, live_entry],
            project_hints=("sr2",),
            db_name="ac6-r1.db",
        )
        by_src = {r.provenance_ids[0]: r for r in records}

        assert by_src[user_entry].authority is ContextAuthority.USER
        assert by_src[user_entry].kind is ContinuityKind.PREFERENCE

        claim = by_src[assistant_entry]
        assert claim.authority is ContextAuthority.AGENT_TENTATIVE
        assert claim.kind is ContinuityKind.EPISODE

        assert by_src[live_entry].authority is ContextAuthority.LIVE_SOURCE


# ---------------------------------------------------------------------------
# AC6-r2 — factual-current predicate: USER / LIVE_SOURCE / VERIFIED records
#          pass; AGENT_TENTATIVE and DERIVED are excluded. Accepts both
#          records and views.
# ---------------------------------------------------------------------------


def rec(authority: ContextAuthority, id_: str) -> ContinuityRecord:
    return make_record(id_, f"subject-{id_}", ContinuityEventKind.FORMED,
                       authority=authority)


class TestFactualCurrentPredicate:
    def test_ac6_r2_predicate_on_records(self):
        for authority in (
            ContextAuthority.USER,
            ContextAuthority.LIVE_SOURCE,
            ContextAuthority.VERIFIED,
        ):
            assert is_factual_current(rec(authority, f"fc-{authority.value}")) is True
        for authority in (
            ContextAuthority.AGENT_TENTATIVE,
            ContextAuthority.DERIVED,
        ):
            assert is_factual_current(rec(authority, f"nt-{authority.value}")) is False

    def test_ac6_r2_selector_over_iterable(self):
        records = [
            rec(ContextAuthority.USER, "sel-user"),
            rec(ContextAuthority.AGENT_TENTATIVE, "sel-tent"),
            rec(ContextAuthority.DERIVED, "sel-der"),
            rec(ContextAuthority.LIVE_SOURCE, "sel-live"),
            rec(ContextAuthority.VERIFIED, "sel-ver"),
        ]
        kept = factual_current_records(records)
        assert {r.id for r in kept} == {"sel-user", "sel-live", "sel-ver"}

    async def test_ac6_r2_predicate_on_view(self, tmp_path):
        builder, ids = await build_ac5_turn(tmp_path)
        # A tentative assistant claim appended into the ledger: its view must
        # exist yet fail the factual-current predicate.
        claim_id = ids[9]
        _records, store, _kinds = await consolidate_fixture(
            tmp_path, builder, [claim_id], project_hints=("sr2",),
            db_name="ac6-r2.db",
        )
        views = await store.current_views()
        tentative_views = [v for v in views.values() if not is_factual_current(v)]
        assert len(tentative_views) == 1, (
            "exactly the assistant-claim view must fail the factual-current test"
        )
        assert tentative_views[0].current_record.authority is (
            ContextAuthority.AGENT_TENTATIVE
        )


# ---------------------------------------------------------------------------
# AC6-r3 — the headline row: an unsupported assistant claim stays a tentative
#          episode excluded from factual-current retrieval. No user input
#          whatsoever supports it.
# ---------------------------------------------------------------------------


class TestUnsupportedClaimStaysTentative:
    async def test_ac6_r3_excluded_from_factual_current(self, tmp_path):
        store_prov = InMemoryProvenanceStore()
        builder = TurnArchiveBuilder(store_prov, "sess-ac6r3")
        await builder.next_turn()
        await builder.tool_use("git_head", {"repo": "sr2"})
        claim_id = await builder.assistant("I believe the cache defaults to redis.")

        records, store, _kinds = await consolidate_fixture(
            tmp_path, builder, [claim_id], project_hints=("sr2",),
            db_name="ac6-r3.db",
        )
        assert len(records) == 1
        claim = records[0]
        assert claim.authority is ContextAuthority.AGENT_TENTATIVE
        assert claim.kind is ContinuityKind.EPISODE
        assert claim.provenance_ids == (claim_id,)
        assert factual_current_records(records) == []

        views = await store.current_views()
        current = [v.current_record for v in views.values()]
        assert factual_current_records(current) == []


# ---------------------------------------------------------------------------
# AC6-r4 — promotion: a USER-authority confirmation turn for the claim's
#          subject appends a NEW VERIFIED event (authority VERIFIED, its own
#          provenance naming the confirming entry, supersedes == ()), which
#          makes the subject factual-current. The original tentative record
#          survives byte-identical (append-only; promotion never mutates).
# ---------------------------------------------------------------------------


class TestPromotion:
    async def test_ac6_r4_confirmation_appends_verified_event(self, tmp_path):
        subject = "subject-cache-backend"
        builder, ids = await build_ac5_turn(tmp_path, session_id="sess-ac6r4")
        claim_id = ids[9]

        tentative, store, _kinds = await consolidate_fixture(
            tmp_path, builder, [claim_id], project_hints=("sr2",),
            db_name="ac6-r4.db",
        )
        tentative = tentative[0]
        assert tentative.subject_id == subject
        assert tentative.authority is ContextAuthority.AGENT_TENTATIVE

        before = {r.id: serialize(r) for r in await store.get_events()}

        # Confirmation arrives as a later USER-authority turn entry that
        # starts with the public CONFIRMATION_PATTERN and names the subject.
        await builder.next_turn()
        confirm_id = await builder.user(
            "Confirmed: the cache defaults to redis.",
            {"about_subject": subject},
        )
        confirmation_entry = await builder_store(builder).get(confirm_id)
        assert CONFIRMATION_PATTERN.match(
            confirmation_entry.content.content[0].text
        ), "the fixture's confirmation text must match the public pattern"

        promoted = await RuleBasedContinuityConsolidator(
            provenance=builder_store(builder)
        ).consolidate(request_for([confirm_id], project_hints=("sr2",)))
        assert len(promoted) == 1
        verification = promoted[0]
        for record in promoted:
            await store.append(record)

        assert verification.event_kind is ContinuityEventKind.VERIFIED
        assert verification.authority is ContextAuthority.VERIFIED
        assert verification.subject_id == subject
        assert verification.supersedes == ()
        assert verification.provenance_ids == (confirm_id,)

        # factual-current now includes the subject
        views = await store.current_views()
        view = views[subject]
        assert view.current_record.id == verification.id
        assert is_factual_current(verification) is True
        assert is_factual_current(view) is True
        assert [r.id for r in factual_current_records(
            v.current_record for v in views.values()
        )] == [verification.id]

        # append-only: the tentative record still exists, byte-identical
        after = {r.id: serialize(r) for r in await store.get_events()}
        assert before.keys() <= after.keys()
        assert verification.id in after  # exactly one new row was added
        for rid, blob in before.items():
            assert after[rid] == blob, f"record {rid} changed bytes"
        assert tentative.id in after

        chain = await store.get_events(subject_id=subject)
        assert [r.id for r in chain] == [tentative.id, verification.id]
        assert view.history_ids == (tentative.id, verification.id)


# ---------------------------------------------------------------------------
# AC6-r5 — negative promotion control: an unrelated user entry (no
#          confirmation marker, no subject link) promotes nothing — the claim
#          stays excluded from factual-current.
# ---------------------------------------------------------------------------


class TestPromotionControls:
    async def test_ac6_r5_unrelated_user_entry_promotes_nothing(self, tmp_path):
        subject = "subject-cache-backend"
        builder, ids = await build_ac5_turn(tmp_path, session_id="sess-ac6r5")
        claim_id = ids[9]
        tentative, store, _ = await consolidate_fixture(
            tmp_path, builder, [claim_id], project_hints=("sr2",),
            db_name="ac6-r5.db",
        )
        tentative = tentative[0]
        before = {r.id: serialize(r) for r in await store.get_events()}

        await builder.next_turn()
        unrelated_id = await builder.user("What is the weather in Oslo?")
        promoted = await RuleBasedContinuityConsolidator(
            provenance=builder_store(builder)
        ).consolidate(request_for([unrelated_id], project_hints=("sr2",)))

        # nothing may be appended that flips the subject: either no records,
        # or none touching the claim's subject with promotion semantics.
        for record in promoted:
            assert record.subject_id != subject, (
                "an unrelated user entry must not touch the tentative subject"
            )
            await store.append(record)

        views = await store.current_views()
        assert views[subject].current_record.id == tentative.id
        assert is_factual_current(views[subject]) is False

        after = {r.id: serialize(r) for r in await store.get_events()}
        for rid, blob in before.items():
            assert after[rid] == blob

    async def test_ac6_r5b_verified_authority_predicate_edge(self):
        # predicate edge pinned independently of any store round-trip:
        assert is_factual_current(
            rec(ContextAuthority.VERIFIED, "edge-verified")
        ) is True
        assert is_factual_current(
            rec(ContextAuthority.DERIVED, "edge-derived")
        ) is False


# ---------------------------------------------------------------------------
# AC7-r1 — supersession: the correction becomes current, the superseded record
#          is absent from every current view, and the correction names it in
#          ``supersedes``.
# ---------------------------------------------------------------------------


async def correction_setup(tmp_path, db_name: str):
    """Seed the ledger with a FORMED project fact, then consolidate a
    CORRECTION for its subject."""
    store = await open_store(tmp_path, db_name)
    superseded = make_record(
        "rec-pg-0001",
        "subject-db-backend",
        ContinuityEventKind.FORMED,
        kind=ContinuityKind.PROJECT_FACT,
        content="The ledger uses Postgres.",
        authority=ContextAuthority.USER,
    )
    await store.append(superseded)

    builder = TurnArchiveBuilder(InMemoryProvenanceStore(), "sess-ac7")
    await builder.next_turn()
    correction_entry_id = await builder.user(
        "correction: we adopted SQLite, not Postgres.",
        {"supersedes_subject": "subject-db-backend"},
    )
    consolidator = RuleBasedContinuityConsolidator(provenance=builder_store(builder), store=store)
    records = await consolidator.consolidate(
        request_for([correction_entry_id], project_hints=("sr2",))
    )
    assert len(records) == 1
    correction = records[0]
    await store.append(correction)
    return store, superseded, correction, builder, correction_entry_id


class TestSupersession:
    async def test_ac7_r1_correction_is_current_superseded_absent(self, tmp_path):
        store, superseded, correction, _builder, correction_entry_id = (
            await correction_setup(tmp_path, "ac7-r1.db")
        )

        assert correction.kind is ContinuityKind.CORRECTION
        assert correction.subject_id == "subject-db-backend"
        assert correction.supersedes == (superseded.id,)
        assert correction.provenance_ids == (correction_entry_id,)

        views = await store.current_views()
        view = views["subject-db-backend"]
        assert view.current_record.id == correction.id
        assert view.current_record.kind is ContinuityKind.CORRECTION
        assert view.status == "active"
        # the superseded record is absent from normal (current) retrieval
        current_records = [v.current_record for v in views.values()]
        assert all(r.id != superseded.id for r in current_records)


# ---------------------------------------------------------------------------
# AC7-r2 — historical query: the full immutable chain (superseded +
#          correction) with rationale sources survives in get_events and
#          history_ids, and earlier bytes never change.
# ---------------------------------------------------------------------------


class TestSupersessionHistory:
    async def test_ac7_r2_historical_chain_complete_and_immutable(self, tmp_path):
        store, superseded, correction, _builder, correction_entry_id = (
            await correction_setup(tmp_path, "ac7-r2.db")
        )
        superseded_bytes = serialize(superseded)

        views = await store.current_views()
        view = views["subject-db-backend"]
        assert view.history_ids == (superseded.id, correction.id)

        chain = await store.get_events(subject_id="subject-db-backend")
        assert [r.id for r in chain] == [superseded.id, correction.id]

        # the superseded record is fully retrievable from history
        historical = {r.id: r for r in chain}
        assert serialize(historical[superseded.id]) == superseded_bytes

        # rationale sources of the correction are present in the chain
        correction_in_chain = historical[correction.id]
        assert correction_in_chain.provenance_ids == (correction_entry_id,)
        assert set(correction_in_chain.provenance_ids) <= set(
            r.provenance_ids[0] for r in chain if r.provenance_ids
        )

        # rebuild determinism over the same immutable chain
        rebuilt = await store.rebuild_projections()
        assert rebuilt["subject-db-backend"].history_ids == (
            superseded.id,
            correction.id,
        )
        assert rebuilt["subject-db-backend"].current_record.id == correction.id
