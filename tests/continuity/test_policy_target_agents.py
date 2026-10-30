"""sr2-146 / FR16 — ``target_agents`` matches the querying agent (unit layer).

Landing spot fixed by the bead: ``sr2.continuity.policy._rule_matches``.

FR16 (spec, verbatim): "Each Spectre agent MUST have a configurable profile
containing allow, deny, and boost rules matched on record kind, projects,
topics/tags, sensitivity, source agent, and target agent/profile tags. Deny
is evaluated before scoring and always wins."

The rule's ``agents`` field constrains the record's SOURCE agent
(``record.scope.agent_id``); the rule's ``target_agents`` field constrains
the TARGET — i.e. QUERYING — agent (``context.agent_id``). The two are
distinct match dimensions; an implementation that compares ``target_agents``
against the source agent makes the field a duplicate of ``agents`` and these
rows cannot pass.

Case table: one section per row (FR16-r1 .. FR16-r7); tests are named
``test_<row_id>_<behavior>``. The module-level import below is the deliberate
red mechanism only if the module is missing — it exists on this base commit,
so the red state is the assertion failures: on the unfixed comparison
(``rule.target_agents`` vs ``scope.agent_id``) the r1, r3, r4 and r6 rows
fail; on any "match both" over-fix the r2, r5 and r7 rows fail.
"""

from __future__ import annotations

import sr2.continuity.policy as policy_module  # module under test
from continuity._fixtures_policy import (
    AGENT_ASSISTANT_ID,
    AGENT_CODING_ID,
    AGENT_OTHER_ID,
    FR16_CONTENT,
    ContinuityKind,
    fr16_policy,
    fr16_profile,
    fr16_view,
)
from sr2.continuity.policy import (
    RetrievalContext,
    RetrievalDecision,
)

RULE_ALLOW = "fr16-allow-target"
RULE_DENY = "fr16-deny-target"


def _context(agent_id: str) -> RetrievalContext:
    """The FR16 rows' contexts: only ``agent_id`` varies between rows."""
    return RetrievalContext(
        owner_id="diego",
        agent_id=agent_id,
        agent_tags=(),
        projects=("sr2",),
        frame_id="frame-fr16",
        query="what does the profile target",
        token_budget=2000,
    )


def _decision(row_id: str, **kwargs) -> RetrievalDecision:
    """Every FR16 row names its row id in the explanation, so the red run
    identifies the failing row, not just the failing assertion."""
    return RetrievalDecision(explanation=f"case row {row_id}", **kwargs)


def _policy(profile: object) -> object:
    return fr16_policy(policy_module, profile)


# ---------------------------------------------------------------------------
# FR16-r1 — target_agents constrains the querying agent, not the source
# ---------------------------------------------------------------------------


def test_fr16_r1_target_agents_matches_querying_agent_not_source():
    """Rule ``target_agents: [agent-coding]`` (no ``agents`` field); record's
    source agent is ``agent-assistant``; querying agent is ``agent-coding``.

    The rule matches: the allow rule names the decision and allows the
    record. On the buggy comparison (``target_agents`` vs the record's source
    agent) this rule cannot match, so the row is decided by the default and
    the rule id is absent from ``matched_rule_ids``.
    """
    profile = fr16_profile(
        effect="allow",
        rule_id=RULE_ALLOW,
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_ASSISTANT_ID)
    context = _context(AGENT_CODING_ID)

    decision = _policy(profile).evaluate(record, context)

    assert decision.allowed is True, (
        f"the allow rule must admit the record (got: {decision.explanation})"
    )
    assert RULE_ALLOW in decision.matched_rule_ids, (
        "the rule must name itself in matched_rule_ids; a match on the "
        f"record's source agent cannot see the querying agent ({decision.explanation})"
    )


# ---------------------------------------------------------------------------
# FR16-r2 — a querying agent outside target_agents gets no rule match
# ---------------------------------------------------------------------------


def test_fr16_r2_querying_agent_outside_target_agents_is_not_matched():
    """Same allow rule ``target_agents: [agent-coding]``, same record;
    querying agent is ``agent-assistant``.

    The rule does not match: ``matched_rule_ids`` does not name the rule and
    the record is decided by the profile default. The FR16 record is
    non-personal, so the ``deny_cross_agent_personal`` default allows it —
    the assertion on ``matched_rule_ids`` (not on ``allowed``) is what
    distinguishes "the rule did not match" from "the default decided".
    """
    profile = fr16_profile(
        effect="allow",
        rule_id=RULE_ALLOW,
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_ASSISTANT_ID)
    context = _context(AGENT_ASSISTANT_ID)

    decision = _policy(profile).evaluate(record, context)

    assert RULE_ALLOW not in decision.matched_rule_ids, (
        "the rule must not match when the querying agent is not in "
        f"target_agents ({decision.explanation})"
    )
    assert decision.allowed is True, (
        "with no rule matching, the profile default decides; the FR16 record "
        f"is non-personal, so the default admits it ({decision.explanation})"
    )


# ---------------------------------------------------------------------------
# FR16-r3 — an omitted source-agent dimension never blocks the target match
# ---------------------------------------------------------------------------


def test_fr16_r3_omitted_source_agent_dimension_does_not_block_target_match():
    """Allow rule declaring ONLY ``target_agents: [agent-coding]``; the
    record's source agent is also ``agent-coding``; the querying agent is
    ``agent-coding``.

    The rule matches. The row guards the implementation note's boundary: the
    fix must compare ``target_agents`` to the querying agent without
    dragging the (omitted) ``agents`` dimension into the comparison — an
    omitted field matches anything, so the self-referential source agent
    neither blocks nor enables the match.
    """
    profile = fr16_profile(
        effect="allow",
        rule_id=RULE_ALLOW,
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_CODING_ID)
    context = _context(AGENT_CODING_ID)

    decision = _policy(profile).evaluate(record, context)

    assert RULE_ALLOW in decision.matched_rule_ids, (
        f"an omitted agents field matches anything; the target match stands alone ({decision.explanation})"
    )
    assert decision.allowed is True, decision.explanation


# ---------------------------------------------------------------------------
# FR16-r4 — agents (source) and target_agents (target) match independently
# ---------------------------------------------------------------------------


def test_fr16_r4_agents_and_target_agents_match_independently():
    """Allow rule ``agents: [agent-assistant]`` AND ``target_agents:
    [agent-coding]``; record's source agent is ``agent-assistant``; querying
    agent is ``agent-coding``.

    The rule matches: ``agents`` is satisfied by the record's source agent
    and ``target_agents`` by the querying agent, independently. This is the
    row a "match either operand" or "match only one dimension" implementation
    cannot pass: on the buggy comparison ``agents`` and ``target_agents``
    both compare the same field and the rule cannot be simultaneously
    satisfied by two different agents.
    """
    profile = fr16_profile(
        effect="allow",
        rule_id=RULE_ALLOW,
        agents=(AGENT_ASSISTANT_ID,),
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_ASSISTANT_ID)
    context = _context(AGENT_CODING_ID)

    decision = _policy(profile).evaluate(record, context)

    assert RULE_ALLOW in decision.matched_rule_ids, (
        "agents (source) and target_agents (target) are distinct dimensions; "
        f"each must be checked against its own operand ({decision.explanation})"
    )
    assert decision.allowed is True, decision.explanation


# ---------------------------------------------------------------------------
# FR16-r5 — a target miss vetoes the rule even when the source matches
# ---------------------------------------------------------------------------


def test_fr16_r5_target_mismatch_vetoes_rule_despite_source_match():
    """Allow rule ``agents: [agent-assistant]`` AND ``target_agents:
    [agent-coding]``; record's source agent is ``agent-assistant``; querying
    agent is ``agent-other``.

    The rule does not match: the querying agent fails ``target_agents`` even
    though ``agents`` matches the record's source agent. The rule predicate
    is conjunctive — every declared field must hold.
    """
    profile = fr16_profile(
        effect="allow",
        rule_id=RULE_ALLOW,
        agents=(AGENT_ASSISTANT_ID,),
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_ASSISTANT_ID)
    context = _context(AGENT_OTHER_ID)

    decision = _policy(profile).evaluate(record, context)

    assert RULE_ALLOW not in decision.matched_rule_ids, (
        "the rule is conjunctive: a failed target dimension vetoes the match "
        f"even when agents matches the source ({decision.explanation})"
    )
    assert decision.allowed is True, (
        "no rule matched, so the profile default decides; the non-personal "
        f"FR16 record is admitted by the default ({decision.explanation})"
    )


# ---------------------------------------------------------------------------
# FR16-r6 — a matching deny (on the target dimension) wins before scoring
# ---------------------------------------------------------------------------


def test_fr16_r6_deny_on_target_dimension_wins_before_scoring():
    """Deny rule ``target_agents: [agent-coding]`` over a personal record;
    querying agent is ``agent-coding``.

    The deny matches and wins: ``allowed`` is False, the weight stays
    ``1.0`` (deny is evaluated before scoring, so no boost can alter it) and
    ``matched_rule_ids`` names the deny rule. The record is personal
    (preference, owned by ``agent-coding``) and would be admitted by the
    default for its own source agent — the deny is the only reason it is
    denied, so the row also proves the deny actually matched on the target
    dimension.
    """
    profile = fr16_profile(
        effect="deny",
        rule_id=RULE_DENY,
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(
        source_agent_id=AGENT_CODING_ID, kind=ContinuityKind.PREFERENCE
    )
    context = _context(AGENT_CODING_ID)

    decision = _policy(profile).evaluate(record, context)

    assert decision.allowed is False, (
        "the deny rule must gate the record before scoring "
        f"({decision.explanation})"
    )
    assert decision.weight == 1.0, (
        f"deny is evaluated before scoring; the weight stays 1.0 (got {decision.weight})"
    )
    assert RULE_DENY in decision.matched_rule_ids, (
        f"matched_rule_ids must name the deny rule (got {decision.matched_rule_ids})"
    )


# ---------------------------------------------------------------------------
# FR16-r7 — a deny rule whose target dimension misses leaves the default
# ---------------------------------------------------------------------------


def test_fr16_r7_deny_miss_on_target_dimension_leaves_profile_default():
    """Deny rule ``target_agents: [agent-coding]``; querying agent is
    ``agent-assistant``.

    The deny does not match. With no other matching rule, the profile
    default decides the record: the FR16 record is non-personal, so the
    default allows it — and ``matched_rule_ids`` does not name the deny rule.
    """
    profile = fr16_profile(
        effect="deny",
        rule_id=RULE_DENY,
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_ASSISTANT_ID)
    context = _context(AGENT_ASSISTANT_ID)

    decision = _policy(profile).evaluate(record, context)

    assert RULE_DENY not in decision.matched_rule_ids, (
        "the deny rule must not match when the querying agent is not in "
        f"target_agents ({decision.explanation})"
    )
    assert decision.allowed is True, (
        "no rule matched, so the profile default decides; the non-personal "
        f"FR16 record is admitted by the default ({decision.explanation})"
    )


# ---------------------------------------------------------------------------
# FR16-r8 — source membership cannot substitute for querying-agent membership
# ---------------------------------------------------------------------------


def test_fr16_r8_source_in_target_agents_does_not_match_non_target_querier():
    """A source in target_agents must not enable a non-target querier."""
    profile = fr16_profile(
        effect="allow",
        rule_id=RULE_ALLOW,
        target_agents=(AGENT_CODING_ID,),
    )
    record = fr16_view(source_agent_id=AGENT_CODING_ID)
    context = _context(AGENT_ASSISTANT_ID)

    decision = _policy(profile).evaluate(record, context)

    assert decision.matched_rule_ids == (), (
        "source membership must not substitute for querying-agent membership "
        f"in target_agents ({decision.explanation})"
    )
    assert decision.allowed is True, decision.explanation
