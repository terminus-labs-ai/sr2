"""Shared fixtures for the sr2-97 retrieval-policy cage (A-owned).

Landing spot fixed by the bead: ``sr2.continuity.policy``. This module does
NOT import that module at module level — the cage file
(``tests/continuity/test_policy.py``) owns the deliberate red mechanism, so a
red run names ``sr2.continuity.policy`` and not a fixture helper. Everything
here builds against types that already exist on the base commit
(``sr2.continuity.models``, ``sr2.pipeline.provenance``).

============================================================================
INTERFACE CONTRACT THIS CAGE FIXES (everything else is free)
============================================================================

The bead quotes the spec's types verbatim: ``RetrievalContext``,
``RetrievalDecision``, ``RetrievalPolicy.evaluate(record, context)``.
Suppression state is an INPUT to the policy (the caller reads the immutable
ledger and hands over the suppressed record ids);
``ContinuityStore.current_views()`` stays the operator-inspection surface
(sr2-95, frozen).

The rows reach the implementation through four module-level surfaces, located
by name so the concrete class names stay the implementer's while the
BEHAVIOR does not. The first name that exists wins, so a richer earlier name
shadows a fallback:

``make_retrieval_policy``  / ``RetrievalPolicyEngine`` / ``PolicyEngine``
    ``factory(profile=<typed profile>, suppressed_ids=<tuple[str, ...]>)``
    -> policy implementing ``evaluate(record, context) -> RetrievalDecision``.
        ``profile`` is always the typed profile config
        (:func:`coding_profile`); ``suppressed_ids`` names the record ids an
        operator has suppressed, so ONE gate covers a derived record and the
        raw episode behind it.

One consequence worth naming: the policy under test is an INPUT to the three
assembly surfaces, so ``filter_allowed`` / ``expand_sources`` /
``compile_request`` can be tested in a specific suppression state instead of
reaching for a module-level default (which is what AC13 needs).

``filter_allowed``  /  ``filter_allowed_records``
    ``(policy, records: Sequence[ContinuityView], context)
      -> list[ContinuityView]``
        keeps exactly the records whose decision is allowed.

``expand_sources``  /  ``expand_record_sources``
    ``(policy, view, context, sources: Sequence[Entry]) -> list[Entry]``
        ``[]`` when ``view`` is denied or suppressed; otherwise only the
        entries the view references (id in ``current_record.provenance_ids``,
        or a direct entry ``source``) whose OWN ``scope`` is allowed for
        ``context``. Raw-source gating is by entry scope — the deny of a
        derived record and the deny of the raw episode behind it are the same
        gate, never a text match.

``compile_request``  /  ``compile_context_request``
    ``(policy, context, recent_entries: Sequence[Entry],
      candidates: Sequence[ContinuityView])`` -> compiled request with a
    ``messages`` sequence and an ``entry_ids`` sequence (``ContextBundle`` is
    the reference name). Denied candidate content is absent by construction;
    allowed candidate content appears inside some message body, and the
    provenance ids of allowed candidates appear in ``entry_ids``.

    The three assembly surfaces take the policy FIRST and required: they are
    the seams AC8-r5/r6 and AC13-r1/r2 use to observe a policy in a specific
    suppression state, so an implicit module-level default cannot answer for
    them. Which of them is a free function and which is a method of the
    policy class is the implementer's (a method is found via
    :func:`policy_surface`).

============================================================================
RULE SEMANTICS THE ROWS PIN
============================================================================

Order (spec, verbatim): "any matching deny wins (evaluated before scoring);
at least one allow or an applicable safe default is required; boosts apply
only after allow." Concretely:

1. matched deny    -> ``allowed=False``, ``weight == 1.0``, every matching
                      rule id in ``matched_rule_ids``, ``explanation``
                      non-empty.
2. else suppressed -> ``allowed=False``, ``weight == 1.0``,
                      ``matched_rule_ids == ()`` (suppression is not a
                      profile rule). A record is suppressed when its id is in
                      the policy's suppressed-id set or its
                      ``ContinuityView.status == "suppressed"``.
3. else matched allow -> ``allowed=True``, ``weight`` = product of the
                      applicable boost weights (1.0 with no boost). A
                      boost-only match is never an allow.
4. else the profile default decides. The example profile's default is
   ``deny_cross_agent_personal``: personal content (any sensitivity tag, or
   one of the personal kinds PREFERENCE / PERSONAL_FACT) is allowed only when
   the source agent (``record.scope.agent_id``) is the querying agent — the
   spec's "unclassified personal information defaults to source-agent-only";
   anything else is denied.

Rule predicates are conjunctive (a rule applies only when the record
satisfies every field the rule declares). Topic and sensitivity
classification are inherited by raw-source entries from the records they back
— a location preference's source episode carries the ``location`` topic and
sensitivity tags, which is what lets AC8-r6's deny bite upstream — while
PROJECT association does not turn personal content into allowed content.

============================================================================
FOUR FORCED READINGS (the spec leaves these open; the rows cannot run without
a choice, so each is derived from a clause and stated here rather than hidden)
============================================================================

1. **Where the profile config lives.** The spec states the profile in YAML and
   the bead's declared scope includes ``sr2.config.models``, so the typed
   profile is ``sr2.config.models.RetrievalProfileConfig``, reachable from the
   YAML mapping's value (``CODING_PROFILE["coding"]``). Rule fields keep the
   YAML spellings. Its schema must reject an unknown ``effect`` and an unknown
   ``default`` — a profile is configuration, and a typo'd ``effect: alow``
   silently allowing everything is the failure the config owns.

2. **What ``deny_cross_agent_personal`` does with NON-personal content.** The
   example profile's only allow rules are ``kinds: [preference, constraint]``,
   so a PROJECT_FACT never reaches an allow rule — and AC9 requires project
   facts to be retrieved (boosted, even) on an SR2 turn, including across
   project lines. The default's NAME is its scope: it denies PERSONAL content
   crossing between AGENTS. So it decides personal content (any sensitivity
   tag, or kind PREFERENCE / PERSONAL_FACT) by source-agent-only, and does not
   reach non-personal content, which stays retrievable. Denying everything the
   rules do not name would make AC9-r1 and AC9-r2 unsatisfiable with the
   spec's own example profile.

3. **How suppression state reaches the policy.** The bead pairs "immutable
   suppression events" with policy enforcement, and sr2-95 froze the ledger
   (``ContinuityStore.append`` / ``get_events`` / ``current_views``) — this
   bead does not extend it. So the caller reads the ledger and hands the policy
   the suppressed record ids; the AC13 rows pass a single id and assert that
   the id gates BOTH the derived record and its raw-source expansion. Whether
   the policy instead derives the same set by reading a store or a view dict
   is free — the rows only ever assert the enforcement.

4. **The names of the assembly surfaces.** AC8-r5 ("the compiled request or
   selection trace payload exposed to the model") and AC8-r6/AC13-r2
   ("source-expand") name behaviors, not functions. ``filter_allowed``,
   ``expand_sources`` and ``compile_request`` are the minimal surfaces the
   rows need; each takes the policy FIRST and required, and may equally be a
   method of the policy class (found either way). Their outputs are read
   duck-typed: ``messages`` + ``entry_ids`` on the compiled request, an
   iterable of ``Entry`` for expansion, an iterable of views for the filter.

============================================================================
AMBIGUITY CAGES BUILT INTO THE FIXTURE DATA
============================================================================

* **Topic and sensitivity are independent dimensions (FR15).** The example
  profile allows topics ``[programming-language, coding-style, tooling]`` and
  denies sensitivity ``[location, relationship, leisure-preference]`` — the
  lists are disjoint, so AC8-r3/r4 cannot pass through a topic-only allow: the
  deny must be driven by the SENSITIVITY axis. AC8-r3 additionally carries a
  personal topic AND a sensitive tag, so a deny rule matching ANY-ONE-OF its
  declared predicates (not only all-of) still denies it, while AC8-r1/r2
  (personal topic, no sensitivity) stay allowed.
* **The boost is asserted as weight ORDER, never an absolute weight.**
  AC9-r1 reads only ``> 1.0`` / ``== 1.0``, so 1.5, 2.0 or any other value
  satisfies it. AC9-r2 asserts the allowed cross-project record comes out
  ``allowed=True`` with ``weight`` strictly BELOW the boosted SR2 record's: an
  implementation that answered the boost with an isolation wall fails there.
* **The querying agent always carries the ``coding`` tag**, including in
  AC8-r7, so a denial there can only come from the personal default and never
  from the profile having no rules for that agent. AC8-r7's control row flips
  ONLY the source-agent match and expects an allow.
* **The AC13 record matches the profile's ALLOW rule** (topic
  ``coding-style``, no sensitivity), so it is retrievable while unsuppressed
  and denied while suppressed: the suppression is the only variable in those
  rows, and its raw episode is allowed by the same rule.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import yaml

from sr2.continuity.models import (
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
    ContinuityView,
)
from sr2.pipeline.provenance import (
    ContextAuthority,
    ContextReason,
    ContextScope,
    Entry,
    EntryOrigin,
)

__all__ = [
    # Re-exported so a test file imports its types from one place.
    "ContextAuthority",
    "ContextReason",
    "ContextScope",
    "ContinuityEventKind",
    "ContinuityKind",
    "ContinuityRecord",
    "ContinuityView",
    "Entry",
    "AGENT_ASSISTANT_ID",
    "AGENT_ASSISTANT_TAGS",
    "AGENT_CODING_ID",
    "AGENT_CODING_TAGS",
    "CODING_PROFILE",
    "CODING_PROFILE_YAML",
    "CONTENT_CODING_STYLE",
    "CONTENT_ICE_CREAM",
    "CONTENT_LOCATION",
    "CONTENT_PROGRAMMING_LANGUAGE",
    "CONTENT_RELATIONSHIP",
    "CONTENT_SPECTRE",
    "CONTENT_SR2_NOTE",
    "CONTENT_UNASSOCIATED_NOTE",
    "CONTEXT_ASSISTANT",
    "CONTEXT_CODING",
    "CONTEXT_CODING_NO_PROJECT",
    "CONTEXT_RECORD_SOURCE_AGENT",
    "OWNER_ID",
    "PROFILE_NAME",
    "PROJECT_SPECTRE",
    "PROJECT_SR2",
    "SUBJECT_AC13",
    "SUPPRESSED_EPISODE_ID",
    "SUPPRESSED_RECORD_ID",
    "UNSUPPORTED_KIND_CONTENT",
    "ac13_view",
    "coding_profile",
    "compile_request",
    "entry_for",
    "entry_ids",
    "entry_ids_of",
    "entry_text",
    "expand",
    "filter_allowed",
    "located",
    "make_policy",
    "make_view",
    "messages_text",
    "policy_surface",
    "preference_view",
    "project_view",
    "suppressed_record",
    "suppression_event",
    "unsuppression_event",
]

# ---------------------------------------------------------------------------
# Identities (deterministic literals, no randomness)
# ---------------------------------------------------------------------------

OWNER_ID = "diego"

#: The agent the example coding profile governs (``target_agent_tags: [coding]``).
AGENT_CODING_ID = "agent-coding"
AGENT_CODING_TAGS = ("coding",)

#: A different agent, targeted by no coding-profile rule — the querying agent
#: for the AC8-r7 restrictive-default row.
AGENT_ASSISTANT_ID = "agent-assistant"
AGENT_ASSISTANT_TAGS = ("assistant",)

PROJECT_SR2 = "sr2"
PROJECT_SPECTRE = "spectre"

#: Personal content whose kind no rule mentions (AC8-r7: no matching allow,
#: different source agent, so the restrictive personal default decides).
UNSUPPORTED_KIND_CONTENT = "PERSONAL-FACT-NO-ALLOW-RULE-6d21af"

#: Distinct literal contents so "denied content never appears" is an
#: unambiguous substring assertion.
CONTENT_PROGRAMMING_LANGUAGE = "PREF-PROGRAMMING-LANGUAGE-TYPEHINTS-0bda3f"
CONTENT_CODING_STYLE = "PREF-CODING-STYLE-ROXBY-RULES-2f9c41"
CONTENT_LOCATION = "SECRET-PREF-LOCATION-41ac90"
CONTENT_ICE_CREAM = "SECRET-PREF-ICE-CREAM-LEISURE-5b7d33"
CONTENT_RELATIONSHIP = "SECRET-PREF-RELATIONSHIP-8ce1f6"
CONTENT_SPECTRE = "SPECTRELAYER-DEADLETTER-BACKPRESSURE-9a5e02"
CONTENT_SR2_NOTE = "SR2-RESOLVER-CONTRACT-3c7f10"
CONTENT_UNASSOCIATED_NOTE = "GENERIC-PROJECT-NOTE-8e4b21"

#: The subject / record / episode the AC13 rows forget and then remember.
SUBJECT_AC13 = "subject-x"
SUPPRESSED_RECORD_ID = "rec-ac13-subject-x"
SUPPRESSED_EPISODE_ID = "src-ac13-episode"

_BASE_TIME = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# The spec's example coding profile — YAML authoritative, typed shape pinned
# ---------------------------------------------------------------------------

#: Name of the profile inside the ``retrieval_profiles`` mapping.
PROFILE_NAME = "coding"

#: The spec's example coding profile verbatim (the value of a
#: ``retrieval_profiles`` mapping). Authoritative: the typed config must be
#: able to take this shape, which :func:`coding_profile` asserts.
CODING_PROFILE_YAML = """\
coding:
  default: deny_cross_agent_personal
  rules:
    - id: coding-preferences
      effect: allow
      kinds: [preference, constraint]
      topics: [programming-language, coding-style, tooling]
      target_agent_tags: [coding]
    - id: unrelated-personal
      effect: deny
      sensitivity: [location, relationship, leisure-preference]
      target_agent_tags: [coding]
    - id: current-project
      effect: boost
      project_relation: current
      weight: 1.5
"""

#: Parsed, order-preserving form of :data:`CODING_PROFILE_YAML`.
CODING_PROFILE: dict[str, Any] = yaml.safe_load(CODING_PROFILE_YAML)


def coding_profile() -> Any:
    """The example coding profile as the typed config model.

    The spec states the profile in YAML, so the typed config MUST accept that
    shape; this is the one place the cage touches config-model naming
    (``sr2.config.models`` is in the bead's declared scope).
    """
    from sr2.config.models import RetrievalProfileConfig

    profile = RetrievalProfileConfig.model_validate(CODING_PROFILE[PROFILE_NAME])
    assert [rule.id for rule in profile.rules] == [
        "coding-preferences",
        "unrelated-personal",
        "current-project",
    ], "the example profile's rules must stay in the spec's order and identity"
    return profile


# ---------------------------------------------------------------------------
# Retrieval contexts
# ---------------------------------------------------------------------------


def _context(agent_id: str, agent_tags: tuple[str, ...], projects: tuple[str, ...]):
    from sr2.continuity.policy import RetrievalContext

    return RetrievalContext(
        owner_id=OWNER_ID,
        agent_id=agent_id,
        agent_tags=agent_tags,
        projects=projects,
        frame_id="frame-1",
        query="what should we prefer here",
        token_budget=2000,
    )


#: SR2 turn, coding agent: the AC8 allow/deny rows and the AC9 boost rows.
#: The example profile's allow rules are all ``target_agent_tags: [coding]``,
#: so every row that expects an allow or a boost queries as this agent.
CONTEXT_CODING = _context(AGENT_CODING_ID, AGENT_CODING_TAGS, (PROJECT_SR2,))

#: Same agent, no active project: shows the boost follows the ACTIVE project
#: instead of being welded to the coding agent.
CONTEXT_CODING_NO_PROJECT = _context(AGENT_CODING_ID, AGENT_CODING_TAGS, ())

#: AC8-r7's querying agent: the ``coding``-tagged agent on an SR2 turn, but
#: NOT the agent its personal record belongs to (that is
#: :data:`AGENT_ASSISTANT_ID`), which is exactly the cross-agent case the
#: profile's restrictive personal default exists for. The ``coding`` tag is
#: what makes the row mean what it says — an untagged querier would be denied
#: for having no rules at all rather than by the personal default.
CONTEXT_ASSISTANT = _context(AGENT_CODING_ID, AGENT_CODING_TAGS, (PROJECT_SR2,))

#: AC8-r7's negative control: the identical personal record, queried BY ITS
#: OWN SOURCE AGENT, must be ALLOWED — proving r7's denial is "cross-agent
#: personal" and not "this record is undeniable-ly private" or "nobody
#: retrieves anything".
CONTEXT_RECORD_SOURCE_AGENT = _context(AGENT_ASSISTANT_ID, AGENT_CODING_TAGS, ())


# ---------------------------------------------------------------------------
# Views (projections)
# ---------------------------------------------------------------------------


def make_view(
    id: str,
    *,
    subject_id: str | None = None,
    event_kind: ContinuityEventKind = ContinuityEventKind.FORMED,
    kind: ContinuityKind = ContinuityKind.DECISION,
    content: str | None = None,
    scope: ContextScope | None = None,
    authority: ContextAuthority = ContextAuthority.USER,
    provenance_ids: tuple[str, ...] = (),
    status: str = "active",
    history_ids: tuple[str, ...] | None = None,
    offset_seconds: int = 0,
) -> ContinuityView:
    """A deterministic ``ContinuityView`` over one record.

    ``status`` is settable directly: the AC8/AC9 rows need plain active views
    and must not depend on projection arithmetic this bead does not own,
    while the AC13 rows drive suppression through the real ledger.
    """
    record = ContinuityRecord(
        id=id,
        subject_id=subject_id if subject_id is not None else f"subject-{id}",
        event_kind=event_kind,
        kind=kind,
        content=content if content is not None else f"content of {id}",
        scope=scope if scope is not None else ContextScope(owner_id=OWNER_ID),
        authority=authority,
        provenance_ids=provenance_ids,
        confidence=0.9,
        salience=0.5,
        created_at=_BASE_TIME + timedelta(seconds=offset_seconds),
    )
    return ContinuityView(
        subject_id=record.subject_id,
        current_record=record,
        status=status,  # type: ignore[arg-type]
        history_ids=history_ids if history_ids is not None else (id,),
    )


def preference_view(
    id: str,
    *,
    content: str,
    subject_id: str | None = None,
    topics: tuple[str, ...] = (),
    sensitivity: tuple[str, ...] = (),
    kind: ContinuityKind = ContinuityKind.PREFERENCE,
    agent_id: str = AGENT_ASSISTANT_ID,
    projects: tuple[str, ...] = (),
    provenance_ids: tuple[str, ...] = (),
    status: str = "active",
) -> ContinuityView:
    """One of Diego's preference-shaped records, owned by ``agent_id``.

    Topic and sensitivity classification are independent dimensions (FR15),
    exactly as the case table spells its inputs out.
    """
    return make_view(
        id,
        subject_id=subject_id,
        kind=kind,
        content=content,
        scope=ContextScope(
            owner_id=OWNER_ID,
            agent_id=agent_id,
            projects=projects,
            topics=topics,
            sensitivity=sensitivity,
        ),
        provenance_ids=provenance_ids,
        status=status,
    )


def project_view(
    id: str,
    *,
    content: str,
    projects: tuple[str, ...] = (),
    topics: tuple[str, ...] = ("architecture",),
    agent_id: str = AGENT_CODING_ID,
    provenance_ids: tuple[str, ...] | None = None,
) -> ContinuityView:
    """A project-associated record for the AC9 boost rows.

    The pair shares kind, authority, provenance id, topic and an EMPTY
    sensitivity scope, so the project association is the ONLY variable the
    weights may respond to: AC9-r1's "equally relevant" pair is identical on
    every axis the profile reads except the project.

    They are PROJECT_FACTs carrying a topic the profile's allow rules never
    mention, which is deliberate. The example profile's allow rules are
    ``kinds: [preference, constraint]``, so NO project fact matches an allow
    rule and every AC9 row is decided by the profile's
    ``deny_cross_agent_personal`` default. That leaves exactly one reading
    with the spec's example profile in front of it: a project-scoped,
    NON-personal record (no sensitivity tag) is not what a default named
    ``deny_cross_agent_personal`` is about, so it stays retrievable and the
    ``current-project`` boost then applies to it — while the same default
    refuses a coding agent another agent's personal record (AC8-r7). The
    ``explanation`` assertion in the AC9 rows makes that reading explicit
    rather than an accident of the implementation.
    """
    return make_view(
        id,
        kind=ContinuityKind.PROJECT_FACT,
        content=content,
        scope=ContextScope(
            owner_id=OWNER_ID,
            agent_id=agent_id,
            projects=projects,
            topics=topics,
        ),
        provenance_ids=(f"src-{id}",) if provenance_ids is None else provenance_ids,
    )


def ac13_view(status: str = "active") -> ContinuityView:
    """Subject X's derived preference view (AC13).

    Its topic (``coding-style``) matches the coding profile's ALLOW rule and
    its scope carries NO sensitivity tag, so while unsuppressed it is allowed
    and while suppressed it is denied: the suppression is the only variable,
    which is what the AC13 rows assert. ``status`` stays ``"active"`` even in
    the suppressed rows — they pin that the policy gates on the SUPPRESSED
    *event* handed to it, independently of the projection's own status flag
    (the ledger's view status is asserted separately by AC13-r4/r6).
    """
    return preference_view(
        SUPPRESSED_RECORD_ID,
        subject_id=SUBJECT_AC13,
        content=CONTENT_LOCATION,
        topics=("coding-style",),
        agent_id=AGENT_CODING_ID,
        provenance_ids=(SUPPRESSED_EPISODE_ID,),
        status=status,
    )


# ---------------------------------------------------------------------------
# Policy / surface factories (located by name; concrete classes are free)
# ---------------------------------------------------------------------------

_POLICY_FACTORY_NAMES = (
    "make_retrieval_policy",
    "RetrievalPolicyEngine",
    "PolicyEngine",
)
_FILTER_NAMES = ("filter_allowed", "filter_allowed_records")
_EXPAND_NAMES = ("expand_sources", "expand_record_sources")
_COMPILE_NAMES = ("compile_request", "compile_context_request")

#: First-parameter names that identify a method fetched off the policy's
#: CLASS rather than the instance (it is unbound, so it needs binding).
#: Everything else — ``policy``, ``records``, ``view``, … — is the documented
#: first parameter of the module-level surface.
_INSTANCE_PARAMS = frozenset({"self", "cls"})


def located(owner: Any, names: Sequence[str], what: str):
    """Return the first callable named in ``names`` on ``owner``."""
    for name in names:
        candidate = getattr(owner, name, None)
        if candidate is not None and callable(candidate):
            return candidate
    raise AssertionError(
        f"{getattr(owner, '__name__', owner)!r} exposes none of {list(names)} "
        f"for {what}"
    )


def policy_surface(module: Any, policy: Any, names: Sequence[str], what: str):
    """Locate an assembly surface on the module or on the policy class.

    A policy may expose ``expand_sources`` as a method rather than as a module
    function; the rows pin the seam, not which object carries it.
    """
    for owner in (module, policy):
        for name in names:
            candidate = getattr(owner, name, None)
            if candidate is not None and callable(candidate):
                return candidate
    raise AssertionError(
        f"neither sr2.continuity.policy nor {type(policy).__name__} exposes "
        f"{list(names)} for {what}"
    )


def make_policy(
    module: Any,
    *,
    profile: Any | None = None,
    suppressed_ids: Sequence[str] = (),
) -> Any:
    """Build the policy under test from the typed profile + suppressed ids.

    The two inputs the case table needs (which profile, what has been
    forgotten) are named; how the implementation stores them is free.
    """
    factory = located(module, _POLICY_FACTORY_NAMES, "the policy factory")
    return factory(
        profile=coding_profile() if profile is None else profile,
        suppressed_ids=tuple(suppressed_ids),
    )


def expand(policy: Any, view: ContinuityView, context: Any, sources: Sequence[Entry]):
    """Raw-source expansion through the located public surface."""
    fn = policy_surface(_module_of(policy), policy, _EXPAND_NAMES, "source-episode expansion")
    return list(_call(fn, policy, (view, context, list(sources))))


def compile_request(policy: Any, context: Any, recent_entries: Sequence[Entry], candidates):
    """Compile the model-facing request through the located public surface."""
    fn = policy_surface(
        _module_of(policy), policy, _COMPILE_NAMES, "compiled-request assembly"
    )
    return _call(fn, policy, (context, list(recent_entries), list(candidates)))


def filter_allowed(policy: Any, records, context: Any) -> list:
    """Policy filter over candidate views through the located surface."""
    fn = policy_surface(
        _module_of(policy), policy, _FILTER_NAMES, "allowed-record filtering"
    )
    return list(_call(fn, policy, (list(records), context)))


def _module_of(policy: Any) -> Any:
    import sys

    return sys.modules[type(policy).__module__]


def _call(fn: Any, policy: Any, args: tuple):
    """Call the located surface.

    Contract: a module-level function takes the policy as its first argument,
    so it is passed explicitly. A callable fetched off the policy's CLASS is
    unbound (first parameter ``self`` / ``cls``) and needs the instance bound
    first — a policy is a legal home for these surfaces, and finding one there
    must not change the arguments the rows supply.
    """
    import inspect

    if inspect.isfunction(fn):
        params = list(inspect.signature(fn).parameters)
        if params and params[0] in _INSTANCE_PARAMS:
            return fn(policy, *args)
    if inspect.ismethod(fn) or inspect.isbuiltin(fn):
        return fn(*args)
    # Any other callable (functools.partial, a __call__ instance): treat it as
    # the documented module-level surface and pass the policy.
    return fn(policy, *args)


# ---------------------------------------------------------------------------
# Compiled-request readers (public shapes only, duck-typed)
# ---------------------------------------------------------------------------


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    dumped = getattr(block, "model_dump_json", None)
    return dumped() if callable(dumped) else str(block)


def messages_text(compiled: Any) -> str:
    """Every byte of the compiled request the model could see, as text."""
    parts: list[str] = []
    for message in getattr(compiled, "messages", ()) or ():
        content = getattr(message, "content", message)
        if isinstance(content, (list, tuple)):
            parts.extend(_block_text(block) for block in content)
        else:
            parts.append(_block_text(content))
    return "\n".join(parts)


def entry_ids_of(compiled: Any) -> list[str]:
    """The provenance-entry ids the compiled request carries, in order."""
    return list(getattr(compiled, "entry_ids", ()) or ())


# ---------------------------------------------------------------------------
# Provenance entries (raw-source side of AC8-r6 / AC13-r2)
# ---------------------------------------------------------------------------

_ORIGIN_GENESIS = EntryOrigin(kind="resolver", name="policy-fixture")


def entry_for(
    id: str,
    *,
    text: str,
    scope: ContextScope | None = None,
    sources: tuple[str, ...] = (),
    reason: ContextReason = ContextReason.USER_INPUT,
    authority: ContextAuthority = ContextAuthority.USER,
) -> Entry:
    """A genesis provenance entry (raw source episode) for expansion tests."""
    from sr2.models import TextBlock

    return Entry(
        id=id,
        content=TextBlock(text=text),
        sources=sources,
        origin=_ORIGIN_GENESIS,
        layer="turn_archive",
        session_id="policy-session",
        created_at=_BASE_TIME,
        reason=reason,
        authority=authority,
        scope=scope if scope is not None else ContextScope(owner_id=OWNER_ID),
    )


def entry_text(entry: Entry) -> str:
    content = entry.content
    return getattr(content, "text", None) or str(content)


def entry_ids(entries: Sequence[Entry]) -> list[str]:
    """Ids of ``entries``, in order — for compact expansion assertions."""
    return [entry.id for entry in entries]


# ---------------------------------------------------------------------------
# Suppression through the real immutable ledger (AC13)
# ---------------------------------------------------------------------------


def suppressed_record() -> ContinuityRecord:
    """Subject X's FORMED event: the derived record the operator forgets.

    Exactly the record :func:`ac13_view` projects, so a row can append it to
    the real ledger and then evaluate the projection without the two drifting
    apart.
    """
    return ac13_view().current_record


def suppression_event(suppressed_record_id: str, ordinal: int = 0) -> ContinuityRecord:
    """A SUPPRESSED event naming one record id (the bead's "suppression
    request" as an immutable event; AC13-r5 pairs it with
    :func:`unsuppression_event`)."""
    return ContinuityRecord(
        id=f"ev-suppressed-{suppressed_record_id}",
        subject_id=SUBJECT_AC13,
        event_kind=ContinuityEventKind.SUPPRESSED,
        kind=ContinuityKind.PREFERENCE,
        content=f"operator forget request for {suppressed_record_id}",
        scope=ContextScope(owner_id=OWNER_ID, agent_id=AGENT_CODING_ID),
        authority=ContextAuthority.USER,
        provenance_ids=(suppressed_record_id,),
        confidence=1.0,
        salience=0.5,
        created_at=_BASE_TIME + timedelta(seconds=1000 + ordinal),
    )


def unsuppression_event(suppressed_record_id: str, ordinal: int = 0) -> ContinuityRecord:
    """An UNSUPPRESSED event naming one record id (AC13-r5/r6)."""
    return ContinuityRecord(
        id=f"ev-unsuppressed-{suppressed_record_id}",
        subject_id=SUBJECT_AC13,
        event_kind=ContinuityEventKind.UNSUPPRESSED,
        kind=ContinuityKind.PREFERENCE,
        content=f"operator restore for {suppressed_record_id}",
        scope=ContextScope(owner_id=OWNER_ID, agent_id=AGENT_CODING_ID),
        authority=ContextAuthority.USER,
        provenance_ids=(suppressed_record_id,),
        confidence=1.0,
        salience=0.5,
        created_at=_BASE_TIME + timedelta(seconds=2000 + ordinal),
    )


# ---------------------------------------------------------------------------
# sr2-146 (FR16): target_agents fixtures — additive, A-owned.
#
# FR16 makes source agent (``record.scope.agent_id``, matched by the rule's
# ``agents`` field) and target agent (the QUERYING agent,
# ``context.agent_id``, matched by the rule's ``target_agents`` field)
# DISTINCT match dimensions. The fixtures below exist only to run the case
# table: they never reuse or alter the AC8 fixtures above.
# ---------------------------------------------------------------------------

#: A third agent: it is neither :data:`AGENT_ASSISTANT_ID` (a source agent
#: used by the rows) nor :data:`AGENT_CODING_ID` (the querying agent of the
#: allow rows) — the querying agent of the FR16-r2/r5/r7 rows, so a
#: ``target_agents`` list naming only the coding agent cannot match it.
AGENT_OTHER_ID = "agent-other"
AGENT_OTHER_TAGS = ("other",)

#: The record of every FR16 row. A plain project fact: no sensitivity tags
#: and a non-personal kind, so under the profile's
#: ``deny_cross_agent_personal`` default the record is ALLOWED by the default
#: when no rule matches. That is exactly the variable the rows pin —
#: FR16-r2/r5/r7 assert the decision the PROFILE DEFAULT makes, so the
#: default must allow; a row that expected the default to deny could not
#: distinguish "the rule did not match" from "the default denied".
FR16_CONTENT = "PROJECT-FACT-TARGET-AGENTS-CASE-146f0a"
FR16_RECORD_ID = "rec-fr16-project-fact"


def fr16_view(
    *,
    source_agent_id: str = AGENT_ASSISTANT_ID,
    agent_id: str | None = None,
    kind: ContinuityKind = ContinuityKind.PROJECT_FACT,
) -> ContinuityView:
    """One FR16 case-table record: a project fact by default (non-personal,
    so the profile's ``deny_cross_agent_personal`` default admits it when no
    rule matches), optionally re-typed (FR16-r6 uses a preference)."""
    return make_view(
        FR16_RECORD_ID,
        kind=kind,
        content=FR16_CONTENT,
        scope=ContextScope(
            owner_id=OWNER_ID,
            agent_id=agent_id if agent_id is not None else source_agent_id,
            projects=(PROJECT_SR2,),
        ),
    )


def fr16_profile(
    effect: str,
    rule_id: str,
    *,
    agents: tuple[str, ...] = (),
    target_agents: tuple[str, ...] = (),
) -> Any:
    """A one-rule profile isolating exactly one match dimension.

    The default is ``deny_cross_agent_personal`` (the spec's example
    default); with the FR16 record (non-personal, no sensitivity) the
    default admits, so the profile's single rule is the only variable that
    can name (or fail to name) the decision.
    """
    from sr2.config.models import RetrievalProfileConfig

    return RetrievalProfileConfig.model_validate(
        {
            "default": "deny_cross_agent_personal",
            "rules": [
                {
                    "id": rule_id,
                    "effect": effect,
                    "agents": list(agents),
                    "target_agents": list(target_agents),
                }
            ],
        }
    )


def fr16_policy(module: Any, profile: Any) -> Any:
    """Build the policy under test from an :func:`fr16_profile` profile."""
    return make_policy(module, profile=profile)
