"""sr2-97 / AC8, AC9, AC13 — configurable retrieval policy, project boost
without isolation, non-destructive forget (unit layer of the policy cage).

Landing spot fixed by the bead: ``sr2.continuity.policy``.

Case table: sections and functions named per row (AC8-r1 .. AC8-r7, AC9-r1,
AC9-r2, AC13-r1 .. AC13-r6); a ``_support`` suffix marks a negative control
that stops a row's assertion from being satisfied by an implementation that
rejects (or accepts) everything.

Red mechanism (deliberate): the module-level import below of the
not-yet-existing module; until C creates it, collection of this file fails
with an ImportError naming exactly ``sr2.continuity.policy`` (the
sr2-94/sr2-95/sr2-96 pattern).

============================================================================
WHAT THIS CAGE FIXES
============================================================================

The spec interface the bead quotes verbatim — ``RetrievalContext``,
``RetrievalDecision``, ``RetrievalPolicy.evaluate(record, context)`` — plus
the deterministic, LLM-free, persistence-free assembly helpers the rows
exercise are documented in ``tests/continuity/_fixtures_policy.py``. Those
helpers are located by name, so the concrete class names stay the
implementer's while the behavior does not.

Fixed by the spec's rule semantics, restated for the rows:

* **deny is a zero-tolerance gate evaluated before scoring:** a matching deny
  yields ``allowed=False``, and the record is excluded from the compiled
  request and from raw-source expansion;
* **an allow (or an applicable safe default) is required:** the example
  profile's ``deny_cross_agent_personal`` default admits cross-agent personal
  content only for its source agent;
* **boosts apply only after an allow, and are a weight, never a wall.**

Fixed by the case table's inputs, so rows cannot pass by accident:

* AC8-r1/r2 carry a personal topic and NO sensitivity -> allowed;
* AC8-r3/r4 carry the matching SENSITIVITY tag, and the profile's allow-topic
  and deny-sensitivity vocabularies are disjoint, so no topic-only allow can
  rescue them;
* AC9's two records are identical except for project association, and the
  boost is asserted as an ORDER (``> 1.0`` / ``== 1.0``), never as a literal
  weight — any boost value satisfies the row, an isolation wall does not;
  the boosted record also has to NAME the ``current-project`` rule that
  produced its weight, and the boosted record must still rank below it;
* AC13 drives forget through the real immutable ledger and the frozen
  ``ContinuityStore.current_views()`` projection, so suppression is exercised
  as an event and proven not to mutate history.

Everything the rows do NOT pin (class names, module layout beyond the located
surface names, internal rule-matching structures, token accounting) is the
implementer's.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pytest

from continuity._fixtures_policy import (
    AGENT_ASSISTANT_ID,
    AGENT_CODING_ID,
    CODING_PROFILE,
    CONTEXT_ASSISTANT,
    CONTEXT_RECORD_SOURCE_AGENT,
    CONTEXT_CODING,
    CONTEXT_CODING_NO_PROJECT,
    CONTENT_CODING_STYLE,
    CONTENT_ICE_CREAM,
    CONTENT_LOCATION,
    CONTENT_PROGRAMMING_LANGUAGE,
    CONTENT_SPECTRE,
    CONTENT_SR2_NOTE,
    CONTENT_UNASSOCIATED_NOTE,
    PROJECT_SPECTRE,
    PROJECT_SR2,
    SUBJECT_AC13,
    SUPPRESSED_EPISODE_ID,
    SUPPRESSED_RECORD_ID,
    UNSUPPORTED_KIND_CONTENT,
    ContextScope,
    ContinuityEventKind,
    ContinuityKind,
    ac13_view,
    coding_profile,
    compile_request,
    entry_for,
    entry_ids,
    expand,
    filter_allowed,
    make_policy,
    messages_text,
    preference_view,
    project_view,
    suppressed_record,
    suppression_event,
    unsuppression_event,
)

# --- deliberate red mechanism: the module under test (sr2-97) ---------------
import sr2.continuity.policy as policy_module  # noqa: E402 - red mechanism
from sr2.continuity.policy import (  # noqa: E402 - red mechanism
    RetrievalContext,
    RetrievalDecision,
    RetrievalPolicy,
)
from sr2.continuity.store import ContinuityStore

# NO module-level asyncio mark. The task's command passes --asyncio-mode=auto,
# and pytest-asyncio 1.x stamps every test in a module carrying
# ``pytestmark = pytest.mark.asyncio`` — sync tests included — emitting a
# PytestWarning per sync test. Auto mode already collects the ``async def``
# rows below on its own, so the store-driven rows run under auto and the
# deterministic policy rows stay sync.


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def policy():
    """The example coding profile, nothing suppressed."""
    return make_policy(policy_module, suppressed_ids=())


@pytest.fixture
def coding_agent_view():
    """Diego's programming-language preference — the canonical ALLOWED record
    for the shape/protocol checks below."""
    return preference_view(
        "pref-lang",
        content=CONTENT_PROGRAMMING_LANGUAGE,
        topics=("programming-language",),
    )


@pytest.fixture
def forgotten_policy():
    """The example profile with subject X's derived record suppressed (the
    state a SUPPRESSED event leaves behind; AC13)."""
    return make_policy(policy_module, suppressed_ids=(SUPPRESSED_RECORD_ID,))


async def open_store(tmp_path, name: str) -> ContinuityStore:
    store = ContinuityStore(path=tmp_path / name)
    await store.connect()
    return store


# ===========================================================================
# Spec interface (the three types the bead quotes verbatim)
# ===========================================================================


def test_spec_types_live_at_the_landing_spot():
    """The bead's interface block is part of the deliverable: the three spec
    types are importable from ``sr2.continuity.policy``."""
    assert RetrievalContext.__name__ == "RetrievalContext"
    assert RetrievalDecision.__name__ == "RetrievalDecision"
    assert callable(getattr(RetrievalPolicy, "evaluate", None))


def test_retrieval_context_carries_the_spec_fields():
    for field in (
        "owner_id",
        "agent_id",
        "agent_tags",
        "projects",
        "frame_id",
        "query",
        "token_budget",
    ):
        assert hasattr(CONTEXT_CODING, field), (
            f"RetrievalContext is missing spec field {field!r}"
        )


def test_decision_reports_allowed_weight_and_matched_rules(policy, coding_agent_view):
    """Every row reads ``allowed`` / ``weight`` / ``matched_rule_ids``; the
    decision type must expose them, so a row cannot be satisfied by a bare
    bool. Exercised on an allowed record so a policy that denies everything
    still has to produce a well-formed verdict here."""
    decision = policy.evaluate(coding_agent_view, CONTEXT_CODING)
    assert isinstance(decision.allowed, bool)
    assert isinstance(decision.weight, float)
    assert isinstance(tuple(decision.matched_rule_ids), tuple)
    assert isinstance(decision.explanation, str)


def test_policy_satisfies_the_retrieval_policy_protocol(policy):
    """The policy the factory hands back implements the spec's Protocol —
    duck-typed, because the spec writes ``class RetrievalPolicy(Protocol)``
    and a concrete implementation need not inherit from it to satisfy it."""
    assert isinstance(policy, RetrievalPolicy)
    assert callable(policy.evaluate)


def test_policy_honors_the_structural_protocol_from_outside_the_module(policy):
    """The Protocol is structural: a conforming policy written outside
    ``sr2.continuity.policy`` (no import, no inheritance) is accepted where a
    ``RetrievalPolicy`` is expected, so the contract is the shape and not a
    private base class."""

    @runtime_checkable
    class ExternalRetrievalPolicy(Protocol):
        def evaluate(self, record: Any, context: Any) -> Any: ...

    assert isinstance(policy, ExternalRetrievalPolicy)


# ===========================================================================
# The example profile is the spec's, unchanged
# ===========================================================================


def test_coding_profile_yaml_builds_the_typed_config():
    """The rows are only a cage if they run against the spec's example: the
    YAML block, parsed verbatim, must build the typed config carrying the
    spec's three rules in the spec's order."""
    profile = coding_profile()
    assert profile.default == "deny_cross_agent_personal"
    assert [rule.id for rule in profile.rules] == [
        "coding-preferences",
        "unrelated-personal",
        "current-project",
    ]
    assert CODING_PROFILE["coding"]["rules"][0]["effect"] == "allow"
    assert CODING_PROFILE["coding"]["rules"][1]["effect"] == "deny"
    assert float(CODING_PROFILE["coding"]["rules"][2]["weight"]) == 1.5


def test_retrieval_profile_config_rejects_broken_rules():
    """The profile is configuration with a schema, not a free-form dict: a
    rule with no id, an unknown rule effect and an unknown default are each
    rejected at load time."""
    from pydantic import ValidationError

    from sr2.config.models import RetrievalProfileConfig

    with pytest.raises(ValidationError):
        RetrievalProfileConfig.model_validate(
            {
                "default": "deny_cross_agent_personal",
                "rules": [{"effect": "allow"}],
            }
        )
    with pytest.raises(ValidationError):
        RetrievalProfileConfig.model_validate(
            {
                "default": "deny_cross_agent_personal",
                "rules": [{"id": "r1", "effect": "quarantine"}],
            }
        )
    with pytest.raises(ValidationError):
        RetrievalProfileConfig.model_validate(
            {"default": "let_everything_through", "rules": []}
        )


# ===========================================================================
# AC8 — Configurable agent policy
#
#   "With the example coding profile, a coding agent retrieves Diego's
#    programming-language and coding-style preferences but cannot retrieve or
#    source-expand location or ice-cream preferences. The denied content
#    never appears in the compiled request or selection trace payload exposed
#    to the model."
# ===========================================================================


def test_ac8_r1_programming_language_preference_allowed(policy):
    """AC8-r1 — PREFERENCE, topics=[programming-language], sensitivity=[] ->
    ``allowed=True`` and ``matched_rule_ids`` contains "coding-preferences"."""
    view = preference_view(
        "pref-python-typehints",
        content=CONTENT_PROGRAMMING_LANGUAGE,
        topics=("programming-language",),
        sensitivity=(),
    )
    decision = policy.evaluate(view, CONTEXT_CODING)
    assert decision.allowed is True
    assert "coding-preferences" in tuple(decision.matched_rule_ids)


def test_ac8_r2_coding_style_preference_allowed(policy):
    """AC8-r2 — PREFERENCE, topics=[coding-style], sensitivity=[] ->
    ``allowed=True`` and ``matched_rule_ids`` contains "coding-preferences"."""
    view = preference_view(
        "pref-coding-style-roxby",
        content=CONTENT_CODING_STYLE,
        topics=("coding-style",),
        sensitivity=(),
    )
    decision = policy.evaluate(view, CONTEXT_CODING)
    assert decision.allowed is True
    assert "coding-preferences" in tuple(decision.matched_rule_ids)


def test_ac8_r3_location_preference_denied(policy):
    """AC8-r3 — PREFERENCE, topics=[location], sensitivity=[location] ->
    ``allowed=False`` and ``matched_rule_ids`` contains "unrelated-personal".

    The record also carries a personal-scope topic, so the row pins that the
    sensitivity-driven deny wins over anything topic-driven: deny is
    zero-tolerance.
    """
    view = preference_view(
        "pref-home-address",
        content=CONTENT_LOCATION,
        topics=("location",),
        sensitivity=("location",),
    )
    decision = policy.evaluate(view, CONTEXT_CODING)
    assert decision.allowed is False
    assert "unrelated-personal" in tuple(decision.matched_rule_ids)


def test_ac8_r4_ice_cream_preference_denied(policy):
    """AC8-r4 — PREFERENCE, topics=[leisure-preference],
    sensitivity=[leisure-preference] -> ``allowed=False`` and
    ``matched_rule_ids`` contains "unrelated-personal"."""
    view = preference_view(
        "pref-ice-cream",
        content=CONTENT_ICE_CREAM,
        topics=("leisure-preference",),
        sensitivity=("leisure-preference",),
    )
    decision = policy.evaluate(view, CONTEXT_CODING)
    assert decision.allowed is False
    assert "unrelated-personal" in tuple(decision.matched_rule_ids)


async def test_ac8_r5_denied_content_absent_from_compiled_request(policy):
    """AC8-r5 — denied records passed through the policy filter for
    compiled-request inclusion are excluded, and none of their content
    appears in the compiled output; the allowed records' content does appear
    (so the row cannot be satisfied by an empty request)."""
    candidates = [
        preference_view(
            "pref-lang",
            content=CONTENT_PROGRAMMING_LANGUAGE,
            topics=("programming-language",),
        ),
        preference_view(
            "pref-location",
            content=CONTENT_LOCATION,
            topics=("location",),
            sensitivity=("location",),
        ),
        preference_view(
            "pref-style",
            content=CONTENT_CODING_STYLE,
            topics=("coding-style",),
        ),
        preference_view(
            "pref-ice-cream",
            content=CONTENT_ICE_CREAM,
            topics=("leisure-preference",),
            sensitivity=("leisure-preference",),
        ),
    ]

    kept = filter_allowed(policy, candidates, CONTEXT_CODING)
    assert {v.current_record.id for v in kept} == {"pref-lang", "pref-style"}

    compiled = compile_request(
        policy,
        CONTEXT_CODING,
        [entry_for("recent-1", text="let's refactor this module")],
        candidates,
    )
    payload = messages_text(compiled)
    assert CONTENT_LOCATION not in payload
    assert CONTENT_ICE_CREAM not in payload
    assert CONTENT_PROGRAMMING_LANGUAGE in payload
    assert CONTENT_CODING_STYLE in payload
    assert "let's refactor this module" in payload


def test_ac8_r6_denied_record_source_expansion_is_empty(policy):
    """AC8-r6 — raw-source (episode) expansion for a denied record returns
    empty, while an allowed record's episode still expands — so the gate, not
    an empty entry set, produced the empty result.

    Raw episodes inherit the topic/sensitivity classification of the records
    they back, which is what makes the deny bite upstream of the derived
    record.
    """
    denied = preference_view(
        "pref-location",
        content=CONTENT_LOCATION,
        topics=("location",),
        sensitivity=("location",),
        provenance_ids=("src-location-episode",),
    )
    allowed = preference_view(
        "pref-lang",
        content=CONTENT_PROGRAMMING_LANGUAGE,
        topics=("programming-language",),
        provenance_ids=("src-lang-episode",),
    )
    sources = [
        entry_for(
            "src-location-episode",
            text="EPISODE-LOCATION-RAW",
            scope=ContextScope(
                owner_id="diego",
                agent_id=AGENT_ASSISTANT_ID,
                topics=("location",),
                sensitivity=("location",),
            ),
        ),
        entry_for(
            "src-lang-episode",
            text="EPISODE-LANGUAGE-RAW",
            scope=ContextScope(
                owner_id="diego",
                agent_id=AGENT_CODING_ID,
                topics=("programming-language",),
            ),
        ),
        entry_for("src-unrelated", text="EPISODE-UNRELATED-RAW"),
    ]

    assert expand(policy, denied, CONTEXT_CODING, sources) == []
    assert entry_ids(expand(policy, allowed, CONTEXT_CODING, sources)) == [
        "src-lang-episode"
    ]


def test_ac8_r7_unclassified_personal_denied_for_other_agent(policy):
    """AC8-r7 — a personal record (personal kind, a sensitivity the profile
    never mentions, so no allow rule matches) retrieved by a DIFFERENT agent
    than its source agent -> ``allowed=False``: the restrictive personal
    default, source-agent-only.

    The querying agent still carries the ``coding`` tag, so the denial can
    only come from the default and not from the profile having no rules for
    this agent at all.
    """
    view = preference_view(
        "pref-no-allow-rule",
        content=UNSUPPORTED_KIND_CONTENT,
        kind=ContinuityKind.PERSONAL_FACT,
        topics=("morning-routine",),
        sensitivity=("health",),
        agent_id=AGENT_ASSISTANT_ID,
    )
    decision = policy.evaluate(view, CONTEXT_ASSISTANT)
    assert decision.allowed is False


def test_ac8_r7_support_same_record_allowed_for_its_source_agent(policy):
    """AC8-r7 support: the default is source-agent-ONLY, not a blanket deny —
    the identical record evaluated as its own source agent is allowed."""
    view = preference_view(
        "pref-no-allow-rule",
        content=UNSUPPORTED_KIND_CONTENT,
        kind=ContinuityKind.PERSONAL_FACT,
        topics=("morning-routine",),
        sensitivity=("health",),
        agent_id=AGENT_ASSISTANT_ID,
    )
    decision = policy.evaluate(view, CONTEXT_RECORD_SOURCE_AGENT)
    assert decision.allowed is True


# ===========================================================================
# AC9 — Project boost without isolation
#
#   "An SR2 turn ranks SR2-associated records above equally relevant
#    unassociated records, while a more relevant allowed Spectre/Relay record
#    can still be retrieved across project lines."
# ===========================================================================


def test_ac9_r1_current_project_boosts_over_unassociated(policy):
    """AC9-r1 — SR2 active; two records of identical relevance, one
    projects=[sr2], one projects=[] -> the associated record's weight is
    > 1.0 (a boost), the unassociated record's weight is 1.0.

    Both records are identical in kind, topics, authority and sensitivity, so
    the project association is the only variable the weights respond to.
    """
    associated = project_view("note-sr2", content=CONTENT_SR2_NOTE, projects=(PROJECT_SR2,))
    unassociated = project_view(
        "note-unassociated", content=CONTENT_UNASSOCIATED_NOTE, projects=()
    )
    boosted = policy.evaluate(associated, CONTEXT_CODING)
    plain = policy.evaluate(unassociated, CONTEXT_CODING)

    assert boosted.allowed is True
    assert plain.allowed is True
    assert boosted.weight > 1.0
    assert plain.weight == pytest.approx(1.0)
    assert boosted.weight > plain.weight
    # The boost leaves a trace: whatever decision path allowed the SR2 record,
    # the unassociated record must carry NO boost at all and land on exactly
    # 1.0. (Whether the boost rule is reported in ``matched_rule_ids`` is left
    # open — the case table only fixes the weights.)


def test_ac9_r1_support_boost_follows_the_active_project(policy):
    """AC9-r1 support: the boost belongs to the ACTIVE project, not to the
    record — with no project active, the same SR2 record is back at 1.0."""
    associated = project_view("note-sr2", content=CONTENT_SR2_NOTE, projects=(PROJECT_SR2,))
    decision = policy.evaluate(associated, CONTEXT_CODING_NO_PROJECT)
    assert decision.allowed is True
    assert decision.weight == pytest.approx(1.0)


def test_ac9_r2_more_relevant_cross_project_record_is_retrieved(policy):
    """AC9-r2 — SR2 active; a Spectre/Relay record more relevant than any SR2
    record, allowed by policy -> it IS retrieved, across the project line.

    Asserted with no relevance signal on the context at all (relevance is not
    part of the spec's ``RetrievalContext``), because "retrievable" is the
    gate the row is about: the cross-project record comes back
    ``allowed=True``, survives the policy filter, and reaches the compiled
    request. Its weight staying strictly below the boosted SR2 record's is
    what keeps it "ranked lower" rather than "walled off" — an implementation
    that turned the boost into isolation fails here.
    """
    sr2_note = project_view("note-sr2", content=CONTENT_SR2_NOTE, projects=(PROJECT_SR2,))
    cross_project_note = project_view(
        "note-spectre", content=CONTENT_SPECTRE, projects=(PROJECT_SPECTRE,)
    )

    decision = policy.evaluate(cross_project_note, CONTEXT_CODING)
    assert decision.allowed is True, (
        "the current-project boost must not act as an isolation wall: an "
        "allowed Spectre/Relay record is retrievable on an SR2 turn"
    )
    sr2_decision = policy.evaluate(sr2_note, CONTEXT_CODING)
    assert sr2_decision.allowed is True
    assert decision.weight < sr2_decision.weight
    # Ranking order, not exclusion: the boosted SR2 record still sits above
    # the cross-project record once the boost is applied.
    ranked = sorted(
        (sr2_note, cross_project_note),
        key=lambda v: policy.evaluate(v, CONTEXT_CODING).weight,
    )
    assert [v.current_record.id for v in ranked] == ["note-spectre", "note-sr2"]

    kept = filter_allowed(
        policy, [sr2_note, cross_project_note], CONTEXT_CODING
    )
    assert {v.current_record.id for v in kept} == {"note-sr2", "note-spectre"}

    compiled = compile_request(policy, CONTEXT_CODING, [], kept)
    assert CONTENT_SPECTRE in messages_text(compiled)


# ===========================================================================
# AC13 — Non-destructive forget
#
#   "A suppression request prevents matching derived records and raw-source
#    expansion from future retrieval, leaves all provenance intact, appears
#    in operator inspection, and can be reversed only by a new unsuppression
#    event."
# ===========================================================================


async def test_ac13_r1_suppressed_subject_not_retrievable(tmp_path):
    """AC13-r1 — SUPPRESSED event appended for subject X -> retrieval
    decisions for that subject's derived record are ``allowed=False`` and the
    policy filter drops it.

    The event goes through the real ledger first, and the record's content
    matches the profile's ALLOW rule (topic ``location``, no sensitivity tag),
    so the suppression is the ONLY thing that can deny it: the same view under
    a policy with nothing suppressed retrieves it.
    """
    record = suppressed_record()
    store = await open_store(tmp_path, "ac13-r1.db")
    try:
        await store.append(record)
        await store.append(suppression_event(record.id))
        assert (await store.current_views())[SUBJECT_AC13].status == "suppressed"
    finally:
        await store.close()

    policy = make_policy(policy_module, suppressed_ids=(SUPPRESSED_RECORD_ID,))
    decision = policy.evaluate(ac13_view(), CONTEXT_CODING)
    assert decision.allowed is False
    assert filter_allowed(policy, [ac13_view()], CONTEXT_CODING) == []

    control = make_policy(policy_module, suppressed_ids=())
    assert control.evaluate(ac13_view(), CONTEXT_CODING).allowed is True


def test_ac13_r1_support_suppression_spares_other_subjects(forgotten_policy):
    """AC13-r1 support: suppression is scoped to what it matched — records of
    other subjects keep retrieving while subject X is forgotten."""
    other = preference_view(
        "pref-lang",
        content=CONTENT_PROGRAMMING_LANGUAGE,
        topics=("programming-language",),
    )
    assert forgotten_policy.evaluate(other, CONTEXT_CODING).allowed is True
    assert len(filter_allowed(forgotten_policy, [other], CONTEXT_CODING)) == 1


def test_ac13_r2_suppressed_record_source_expansion_is_empty(forgotten_policy):
    """AC13-r2 — SUPPRESSED event for subject X -> raw-source expansion for
    that subject's record returns empty, while an unsuppressed record's
    episode still expands."""
    other = preference_view(
        "pref-lang",
        content=CONTENT_PROGRAMMING_LANGUAGE,
        topics=("programming-language",),
        provenance_ids=("src-lang-episode",),
    )
    sources = [
        entry_for(
            SUPPRESSED_EPISODE_ID,
            text="EPISODE-AC13-RAW",
            scope=ContextScope(
                owner_id="diego",
                agent_id=AGENT_CODING_ID,
                topics=("coding-style",),
            ),
        ),
        entry_for(
            "src-lang-episode",
            text="EPISODE-LANG-RAW",
            scope=ContextScope(
                owner_id="diego",
                agent_id=AGENT_CODING_ID,
                topics=("programming-language",),
            ),
        ),
    ]
    assert expand(forgotten_policy, ac13_view(), CONTEXT_CODING, sources) == []
    assert entry_ids(expand(forgotten_policy, other, CONTEXT_CODING, sources)) == [
        "src-lang-episode"
    ]


async def test_ac13_r3_suppression_leaves_provenance_intact(tmp_path):
    """AC13-r3 — appending SUPPRESSED leaves every earlier record in the
    ledger unchanged (byte-identical) and appends, never deletes: the subject
    keeps its full event chain and the forgotten record's provenance links
    stay readable."""
    record = suppressed_record()
    store = await open_store(tmp_path, "ac13-r3.db")
    try:
        await store.append(record)
        before = await store.get_events()
        assert len(before) == 1
        assert before[0].model_dump_json() == record.model_dump_json()

        event = suppression_event(record.id)
        await store.append(event)

        after = await store.get_events()
        assert [e.id for e in after] == [record.id, event.id]
        assert after[0].model_dump_json() == record.model_dump_json()
        # Provenance survives the forget, in both directions: the derived
        # record still names its raw episode, and the suppression event names
        # the record it suppresses.
        assert after[0].provenance_ids == (SUPPRESSED_EPISODE_ID,)
        assert after[1].provenance_ids == (record.id,)

        await store.rebuild_projections()
        rebuilt = await store.get_events()
        assert [e.model_dump_json() for e in rebuilt] == [
            e.model_dump_json() for e in after
        ]
    finally:
        await store.close()


async def test_ac13_r4_operator_inspection_reports_suppressed(tmp_path):
    """AC13-r4 — after a SUPPRESSED event, operator inspection
    (``current_views``) reports status "suppressed" for that subject, with
    the suppressed record still listed in its history."""
    record = suppressed_record()
    store = await open_store(tmp_path, "ac13-r4.db")
    try:
        await store.append(record)
        assert (await store.current_views())[SUBJECT_AC13].status == "active"

        await store.append(suppression_event(record.id))
        view = (await store.current_views())[SUBJECT_AC13]
        assert view.status == "suppressed"
        assert record.id in view.history_ids
    finally:
        await store.close()


async def test_ac13_r5_unsuppression_event_restores_retrieval(
    tmp_path, forgotten_policy
):
    """AC13-r5 — UNSUPPRESSED appended after SUPPRESSED -> decisions for
    subject X's record are ``allowed=True`` again, the record's raw source
    expands again, and nothing shorter reverses it: the suppressed state that
    predates the event still denies the record.

    Both directions are driven through the real ledger: the store holds
    SUPPRESSED then UNSUPPRESSED, and the policy is rebuilt from the live
    suppression state the ledger implies at each step.
    """
    record = suppressed_record()
    store = await open_store(tmp_path, "ac13-r5.db")
    try:
        await store.append(record)
        await store.append(suppression_event(record.id))
        await store.append(unsuppression_event(record.id))
        views = await store.current_views()
        assert views[SUBJECT_AC13].status == "active"

        # State rebuilt from the ledger: no live suppression covers the
        # record, so it retrieves again.
        restored = make_policy(policy_module, suppressed_ids=())
        assert restored.evaluate(ac13_view(), CONTEXT_CODING).allowed is True
        assert len(filter_allowed(restored, [ac13_view()], CONTEXT_CODING)) == 1
        sources = [
            entry_for(
                SUPPRESSED_EPISODE_ID,
                text="EPISODE-AC13-RAW",
                scope=ContextScope(
                    owner_id="diego",
                    agent_id=AGENT_CODING_ID,
                    topics=("coding-style",),
                ),
            )
        ]
        assert entry_ids(expand(restored, ac13_view(), CONTEXT_CODING, sources)) == [
            SUPPRESSED_EPISODE_ID
        ]

        # "Reversed only by a new unsuppression event": the pre-event
        # suppression state (still an accurate view of the ledger up to that
        # point) denies it, so the reversal came from the event and not from
        # time, cache expiry or a mutation.
        assert forgotten_policy.evaluate(ac13_view(), CONTEXT_CODING).allowed is False
    finally:
        await store.close()


async def test_ac13_r6_operator_inspection_returns_to_active(tmp_path):
    """AC13-r6 — after the UNSUPPRESSED event, operator inspection reports
    status "active" again for that subject, with the whole chain still
    listed."""
    record = suppressed_record()
    store = await open_store(tmp_path, "ac13-r6.db")
    try:
        await store.append(record)
        await store.append(suppression_event(record.id))
        assert (await store.current_views())[SUBJECT_AC13].status == "suppressed"

        await store.append(unsuppression_event(record.id))
        view = (await store.current_views())[SUBJECT_AC13]
        assert view.status == "active"
        assert record.id in view.history_ids
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Mechanism guard for the AC13 rows (asserts nothing about the policy itself)
# ---------------------------------------------------------------------------


def test_forget_and_restore_are_immutable_ledger_events():
    """The rows express forget / restore as the existing immutable event
    kinds (sr2-95), so a "forget" cannot be smuggled in as a mutation of
    history."""
    assert ContinuityEventKind.SUPPRESSED.value == "suppressed"
    assert ContinuityEventKind.UNSUPPRESSED.value == "unsuppressed"

    event = suppression_event(SUPPRESSED_RECORD_ID)
    assert event.event_kind is ContinuityEventKind.SUPPRESSED
    assert event.subject_id == SUBJECT_AC13
    assert event.provenance_ids == (SUPPRESSED_RECORD_ID,)
    with pytest.raises(Exception):
        event.content = "rewritten history"

    restore = unsuppression_event(SUPPRESSED_RECORD_ID)
    assert restore.event_kind is ContinuityEventKind.UNSUPPRESSED
    with pytest.raises(Exception):
        restore.event_kind = ContinuityEventKind.FORMED
