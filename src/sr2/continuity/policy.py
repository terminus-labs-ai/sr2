"""Retrieval policy engine — configurable allow/deny/boost profiles plus
suppression gating (sr2-97 / FR15, AC8, AC9, AC13).

A ``RetrievalPolicy`` answers one question per candidate record: may this
agent, in this context, see it — and at what weight. Evaluation order is
fixed by the spec:

1. **Deny wins before scoring.** Any matching deny rule denies the record;
   no allow or boost is considered, and the decision weight stays ``1.0``.
2. **Suppression.** A record the operator suppressed (an immutable
   SUPPRESSED event in the continuity ledger, sr2-95) is denied. Suppression
   state is an *input* to the policy: the caller reads the ledger and hands
   over the suppressed record ids. Suppression is not a profile rule, so it
   matches no rule id, and it is non-destructive — nothing here deletes or
   rewrites history.
3. **Allow, then boosts.** At least one matching allow rule admits the
   record; every applicable boost rule then multiplies the weight. A
   boost-only match is never an allow.
4. **The profile default** decides everything no allow rule matched. The
   spec's example default ``deny_cross_agent_personal`` reaches only
   *personal* content (any sensitivity tag, or kind ``personal_fact`` /
   ``preference``): unclassified personal information defaults to
   source-agent-only. Non-personal content (project facts, decisions,
   goals, ...) stays retrievable — which is what lets the current-project
   boost rank records without acting as an isolation wall (AC9).

Rule predicates are conjunctive: a rule applies only when the record
satisfies every field the rule declares. Topic and sensitivity
classification are inherited by raw-source entries from the records they
back, so the deny of a derived record and the deny of the raw episode
behind it are the same scope gate, never a text match.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

from sr2.config.models import RetrievalProfileConfig, RetrievalRuleConfig
from sr2.continuity.models import ContinuityKind, ContinuityView
from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import ContextScope, Entry

#: Kinds that are personal matter even without a sensitivity tag. Matching
#: the YAML vocabulary of the spec's example profiles (``kinds:
#: [preference, constraint]``), rule fields and defaults compare against the
#: ``str`` *value* of every enum.
_PERSONAL_KINDS = frozenset(
    {ContinuityKind.PERSONAL_FACT.value, ContinuityKind.PREFERENCE.value}
)


# ---------------------------------------------------------------------------
# Spec interfaces (verbatim shapes from the bead / spec)
# ---------------------------------------------------------------------------


class RetrievalContext(BaseModel, frozen=True):
    """Who is asking, and in what situation."""

    owner_id: str
    agent_id: str
    agent_tags: tuple[str, ...]
    projects: tuple[str, ...]
    frame_id: str | None
    query: str
    active_goal_ids: tuple[str, ...] = ()
    recent_entry_ids: tuple[str, ...] = ()
    token_budget: int


class RetrievalDecision(BaseModel, frozen=True):
    """The policy's verdict for one record in one context."""

    allowed: bool
    weight: float = 1.0
    matched_rule_ids: tuple[str, ...] = ()
    explanation: str


@runtime_checkable
class RetrievalPolicy(Protocol):
    def evaluate(
        self,
        record: ContinuityView,
        context: RetrievalContext,
    ) -> RetrievalDecision: ...


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------


def _enum_value(value: object) -> str:
    return getattr(value, "value", str(value))


def _rule_matches(
    rule: RetrievalRuleConfig, view: ContinuityView, context: RetrievalContext
) -> bool:
    """A rule applies only when the record satisfies *every* declared field."""
    record = view.current_record
    scope = record.scope

    if rule.kinds and _enum_value(record.kind) not in rule.kinds:
        return False
    if rule.topics and not set(rule.topics) & set(scope.topics):
        return False
    if rule.sensitivity and not set(rule.sensitivity) & set(scope.sensitivity):
        return False
    if rule.agents and (scope.agent_id or "") not in rule.agents:
        return False
    if rule.agent_tags and not set(rule.agent_tags) & set(context.agent_tags):
        return False
    if rule.target_agents and (scope.agent_id or "") not in rule.target_agents:
        return False
    if rule.target_agent_tags and not set(rule.target_agent_tags) & set(context.agent_tags):
        return False
    if rule.projects and not set(rule.projects) & set(scope.projects):
        return False
    if rule.project_relation is not None:
        related = bool(set(scope.projects) & set(context.projects))
        if rule.project_relation == "current" and not related:
            return False
        if rule.project_relation == "other" and related:
            return False
    return True


def _is_personal(scope: ContextScope, kind: ContinuityKind) -> bool:
    """Personal content: any sensitivity classification, or a personal kind."""
    return bool(scope.sensitivity) or _enum_value(kind) in _PERSONAL_KINDS


def _default_decides(
    default: str, view: ContinuityView, context: RetrievalContext
) -> tuple[bool, str]:
    """Apply the profile default. Returns ``(allowed, explanation)``."""
    if default == "allow":
        return True, f"profile default {default!r} allows the record"
    if default == "deny":
        return False, f"profile default {default!r} denies the record"
    if default == "deny_cross_agent_personal":
        record = view.current_record
        if _is_personal(record.scope, record.kind) and (
            record.scope.agent_id or ""
        ) != context.agent_id:
            return False, (
                f"profile default {default!r}: personal content owned by "
                f"agent {record.scope.agent_id!r} is source-agent-only and "
                f"{context.agent_id!r} is not that agent"
            )
        return True, (
            f"profile default {default!r} allows the record (not "
            f"cross-agent personal content)"
        )
    # Config models reject unknown defaults at load time; a hand-built
    # profile still fails closed.
    return False, f"unknown profile default {default!r} denies the record"


# ---------------------------------------------------------------------------
# The concrete policy
# ---------------------------------------------------------------------------


class ContinuityRetrievalPolicy:
    """Deterministic, LLM-free policy over a typed profile + suppression set.

    ``suppressed_ids`` names the record ids an operator has suppressed
    (SUPPRESSED events without a later UNSUPPRESSED one). One gate covers
    both a derived record and the raw episode behind it: a view is denied
    when its id is in the set or its projection status is ``"suppressed"``.
    """

    def __init__(
        self,
        profile: RetrievalProfileConfig,
        suppressed_ids: Sequence[str] = (),
    ) -> None:
        self._profile = profile
        self._suppressed = frozenset(suppressed_ids)

    @property
    def profile(self) -> RetrievalProfileConfig:
        return self._profile

    @property
    def suppressed_ids(self) -> frozenset[str]:
        return self._suppressed

    def is_suppressed(self, view: ContinuityView) -> bool:
        record_id = view.current_record.id
        return record_id in self._suppressed or view.status == "suppressed"

    def evaluate(
        self,
        record: ContinuityView,
        context: RetrievalContext,
    ) -> RetrievalDecision:
        matched = tuple(
            rule.id
            for rule in self._profile.rules
            if _rule_matches(rule, record, context)
        )
        deny_ids = tuple(
            rule.id
            for rule in self._profile.rules
            if rule.effect == "deny" and rule.id in matched
        )

        # 1. Deny is a zero-tolerance gate, evaluated before scoring.
        if deny_ids:
            return RetrievalDecision(
                allowed=False,
                weight=1.0,
                matched_rule_ids=deny_ids,
                explanation=(
                    f"deny rule(s) {', '.join(deny_ids)} match before scoring; "
                    f"record {record.current_record.id} is not retrievable"
                ),
            )

        # 2. Suppression (an operator forget, not a profile rule).
        if self.is_suppressed(record):
            return RetrievalDecision(
                allowed=False,
                weight=1.0,
                matched_rule_ids=(),
                explanation=(
                    f"record {record.current_record.id} is suppressed by an "
                    f"operator forget event; only a new unsuppression event "
                    f"restores it"
                ),
            )

        allow_ids = tuple(
            rule.id
            for rule in self._profile.rules
            if rule.effect == "allow" and rule.id in matched
        )
        boost_ids = tuple(
            rule.id
            for rule in self._profile.rules
            if rule.effect == "boost" and rule.id in matched
        )

        # 3. Boosts multiply the weight, but only after an allow.
        if allow_ids:
            weight = 1.0
            for rule in self._profile.rules:
                if rule.effect == "boost" and rule.id in boost_ids:
                    weight *= rule.weight
            return RetrievalDecision(
                allowed=True,
                weight=weight,
                matched_rule_ids=allow_ids + boost_ids,
                explanation=(
                    f"allow rule(s) {', '.join(allow_ids)} admit the record"
                    + (
                        f"; boost(s) {', '.join(boost_ids)} weight it to "
                        f"{weight}"
                        if boost_ids
                        else ""
                    )
                ),
            )

        # 4. No allow matched: the profile default decides. Boosts still
        #    apply to whatever the default admits — the spec's example
        #    profile boosts the current project, and its allow rules cover
        #    only preferences, so a default-admitted project fact is the
        #    AC9 row, not an exception.
        allowed, reason = _default_decides(self._profile.default, record, context)
        weight = 1.0
        if allowed:
            for rule in self._profile.rules:
                if rule.effect == "boost" and rule.id in boost_ids:
                    weight *= rule.weight
        return RetrievalDecision(
            allowed=allowed,
            weight=weight,
            matched_rule_ids=boost_ids,
            explanation=reason,
        )


def make_retrieval_policy(
    profile: RetrievalProfileConfig,
    suppressed_ids: Sequence[str] = (),
) -> ContinuityRetrievalPolicy:
    """Build the policy engine for a typed profile in a suppression state."""
    return ContinuityRetrievalPolicy(profile=profile, suppressed_ids=suppressed_ids)


# ---------------------------------------------------------------------------
# Scope evaluation for raw-source entries
# ---------------------------------------------------------------------------


def evaluate_scope(
    policy: RetrievalPolicy,
    scope: ContextScope,
    context: RetrievalContext,
) -> RetrievalDecision:
    """Run a raw entry's scope through the policy's rule engine.

    Raw episodes inherit the topic/sensitivity classification of the
    records they back, so gating expansion by the entry's own scope is the
    same deny gate as gating the derived record — never a text match.
    """
    profile = getattr(policy, "profile", None)
    if profile is None:
        # A third-party policy object: the Protocol is the contract, so an
        # entry-level decision is not derivable — fail closed.
        return RetrievalDecision(
            allowed=False,
            explanation="policy exposes no profile for raw-source scope gating",
        )
    view = _scope_view(scope)
    deny_ids: list[str] = []
    allow_ids: list[str] = []
    for rule in profile.rules:
        if _rule_matches(rule, view, context):
            if rule.effect == "deny":
                deny_ids.append(rule.id)
            elif rule.effect == "allow":
                allow_ids.append(rule.id)
    if deny_ids:
        return RetrievalDecision(
            allowed=False,
            matched_rule_ids=tuple(deny_ids),
            explanation=f"deny rule(s) {', '.join(deny_ids)} match the entry scope",
        )
    if allow_ids:
        return RetrievalDecision(
            allowed=True,
            matched_rule_ids=tuple(allow_ids),
            explanation=f"allow rule(s) {', '.join(allow_ids)} admit the entry scope",
        )
    allowed, reason = _default_decides(profile.default, view, context)
    return RetrievalDecision(allowed=allowed, explanation=reason)


#: Placeholder timestamp for the synthesized scope view (never evaluated).
_SCOPE_VIEW_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _scope_view(scope: ContextScope, kind: ContinuityKind = ContinuityKind.EPISODE):
    """A minimal active view carrying ``scope`` for default evaluation."""
    from sr2.continuity.models import ContinuityRecord, ContinuityView

    record = ContinuityRecord(
        id="<scope>",
        subject_id="<scope>",
        event_kind="formed",
        kind=kind,
        content="",
        scope=scope,
        authority="derived",
        provenance_ids=(),
        confidence=1.0,
        salience=0.0,
        created_at=_SCOPE_VIEW_EPOCH,
    )
    return ContinuityView(
        subject_id=record.subject_id,
        current_record=record,
        status="active",
        history_ids=(),
    )


# ---------------------------------------------------------------------------
# Assembly surfaces (policy first: the seam AC8-r5/r6 and AC13-r1/r2 test)
# ---------------------------------------------------------------------------


def filter_allowed(
    policy: RetrievalPolicy,
    candidates: Sequence[ContinuityView],
    context: RetrievalContext,
) -> list[ContinuityView]:
    """Keep exactly the candidate views whose decision is allowed."""
    return [
        view
        for view in candidates
        if policy.evaluate(view, context).allowed
    ]


def expand_sources(
    policy: RetrievalPolicy,
    view: ContinuityView,
    context: RetrievalContext,
    sources: Sequence[Entry],
) -> list[Entry]:
    """Raw-source (episode) expansion for one view, gated by the policy.

    ``[]`` when the view itself is denied or suppressed; otherwise only the
    entries the view references (id in ``current_record.provenance_ids``)
    whose own scope is allowed for ``context``.
    """
    if not policy.evaluate(view, context).allowed:
        return []
    referenced = set(view.current_record.provenance_ids)
    return [
        entry
        for entry in sources
        if entry.id in referenced
        and evaluate_scope(policy, entry.scope, context).allowed
    ]


class ContextBundle(BaseModel):
    """The model-facing compiled request: message bodies plus provenance.

    Duck-typed by callers as ``messages`` + ``entry_ids``; denied content
    is absent from it by construction, so it never reaches the model.
    """

    messages: list[Message]
    entry_ids: tuple[str, ...] = ()


def compile_request(
    policy: RetrievalPolicy,
    context: RetrievalContext,
    recent_entries: Sequence[Entry],
    candidates: Sequence[ContinuityView],
) -> ContextBundle:
    """Compile the request the model sees, applying the policy first.

    Recent entries arrive as provenance entries (their ids belong to the
    selection trace); candidates pass through ``filter_allowed``, so denied
    candidate content is absent by construction rather than filtered after
    the fact.
    """
    blocks: list[TextBlock] = []
    entry_ids: list[str] = []

    for entry in recent_entries:
        blocks.append(TextBlock(text=_entry_text(entry)))
        entry_ids.append(entry.id)

    for view in filter_allowed(policy, candidates, context):
        record = view.current_record
        blocks.append(TextBlock(text=record.content))
        entry_ids.extend(record.provenance_ids)

    return ContextBundle(
        messages=[Message(role="user", content=blocks)],
        entry_ids=tuple(entry_ids),
    )


def _entry_text(entry: Entry) -> str:
    content = entry.content
    return getattr(content, "text", None) or str(content)
