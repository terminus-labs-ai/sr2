"""Rule-based typed consolidation (sr2-96 / FR10–FR13, AC5, AC6, AC7).

Turns the exact turn archive (sr2-93) into immutable continuity records
(sr2-95) after every completed turn — no "remember" command, no manual
classification (FR10). The determinism FR10/FR12 reserves for the runtime
lives here: authority assignment, scope classification, provenance linking,
promotion semantics and supersession linking are pure rules; only the kind
classifier is deliberately simple (the spec allows a smarter injected
consolidator, never a less deterministic one).

Rules this module implements (the frozen cage
``tests/continuity/test_consolidation.py`` is the executable spec):

Authority (AC6)
    A record's authority is copied verbatim from its source entry: user
    input -> USER, assistant output -> AGENT_TENTATIVE, live tool result ->
    LIVE_SOURCE (the archivist already stamped these). An assistant claim
    therefore stays a tentative episode; a later USER confirmation
    (``CONFIRMATION_PATTERN``) appends a NEW VERIFIED event for the
    confirmed subject — never a mutation (FR12).

Subjects
    Every kind resolves a stable subject id. EPISODE / CORRECTION take it
    from ``meta["about_subject"]`` / ``meta["supersedes_subject"]`` when the
    entry carries one; otherwise it is derived from the entry's claim text
    (a lowercased, stop-word-dropped, stemmed slug), so a claim and a later
    confirmation of the *same claim* land on the same subject without
    either naming it, and a correction lands on the fact it corrects.

Supersession (AC7)
    A CORRECTION names the current record id of its subject in
    ``supersedes``. Current records resolve against a continuity store: the
    injected one first, else this module's in-process current-projection
    cache (``supersedes=()`` when the subject is unknown to both). It
    appends; it never mutates (FR4).

Scope (FR11)
    ``owner_id``/``agent_id`` come from the request. Personal-scope kinds
    always carry zero projects; project-scope kinds carry the sorted
    ``project_hints``.

Provenance (FR5)
    Exactly one record per resolved source entry, whose ``provenance_ids``
    is that entry's real id — never synthetic. Unresolvable ids contribute
    nothing; an entry that is neither a recognized knowledge trigger nor an
    unsupported claim contributes no record either (a bare "Yes." or an
    open question mints no durable fact).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from sr2.continuity.models import (
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
    ContinuityView,
)
from sr2.pipeline.provenance import ContextAuthority, ContextScope, ProvenanceStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request contract (spec verbatim)
# ---------------------------------------------------------------------------


class ConsolidationRequest(BaseModel, frozen=True):
    """What one completed turn contributes to consolidation (spec FR10)."""

    turn_id: str
    entry_ids: tuple[str, ...]
    owner_id: str
    agent_id: str
    project_hints: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Authority vocabulary (AC6)
# ---------------------------------------------------------------------------

#: Authorities that qualify a record for factual-current retrieval (FR12).
FACTUAL_CURRENT_AUTHORITIES = frozenset(
    {ContextAuthority.USER, ContextAuthority.LIVE_SOURCE, ContextAuthority.VERIFIED}
)

#: Leading confirmation markers (public so the vocabulary is not hidden in a
#: private helper): "confirmed ...", "confirming ...", "that's correct ...".
CONFIRMATION_PATTERN = re.compile(
    r"^\s*(?:confirms?|confirming|confirmed|that'?s\s+(?:correct|right))\b",
    re.IGNORECASE,
)


def is_factual_current(record_or_view: ContinuityRecord | ContinuityView) -> bool:
    """True when a record (or a view's current record) is factual-current.

    USER / LIVE_SOURCE / VERIFIED authorities pass; AGENT_TENTATIVE and
    DERIVED do not (FR12: an unsupported agent claim never becomes durable
    factual knowledge merely because the agent produced it). The frozen
    store derives ``view.status`` from the event kind only, so this
    predicate — not the status — carries the authority dimension.
    """
    if isinstance(record_or_view, ContinuityView):
        record = record_or_view.current_record
    else:
        record = record_or_view
    return record.authority in FACTUAL_CURRENT_AUTHORITIES


def factual_current_records(records: Iterable[ContinuityRecord]) -> list[ContinuityRecord]:
    """Filter records down to the factual-current ones (FR12 retrieval)."""
    return [record for record in records if is_factual_current(record)]


# ---------------------------------------------------------------------------
# Typing — deterministic trigger-phrase rules (FR11)
# ---------------------------------------------------------------------------

# Personal / global kinds (FR11: zero project associations is valid).
PERSONAL_KINDS = frozenset(
    {
        ContinuityKind.PREFERENCE,
        ContinuityKind.PERSONAL_FACT,
        ContinuityKind.GOAL,
        ContinuityKind.NEXT_ACTION,
        ContinuityKind.COMMITMENT,
    }
)

# Project-associable kinds.
PROJECT_KINDS = frozenset(
    {
        ContinuityKind.DECISION,
        ContinuityKind.CONSTRAINT,
        ContinuityKind.PROJECT_FACT,
        ContinuityKind.CORRECTION,
        ContinuityKind.EPISODE,
    }
)

# Source classes: the user speaks, or an evidence surface (a configured live
# source / an already-verified record) does. Anything else — assistant
# output — is an unsupported claim.
_EVIDENCE_AUTHORITIES = frozenset(
    {ContextAuthority.USER, ContextAuthority.LIVE_SOURCE, ContextAuthority.VERIFIED}
)

# Trigger vocabulary (matched case-insensitively against entry text).
CORRECTION_TRIGGERS = (r"\bcorrection\s*:",)
DECISION_TRIGGERS = (r"\b(?:i|we)\s+(?:have\s+)?deci(?:de|ded)\b",)
PREFERENCE_TRIGGERS = (r"\bi\s+(?:prefer|like)\b",)
PERSONAL_FACT_TRIGGERS = (r"\bi\s+(?:always|never)\b",)
GOAL_TRIGGERS = (r"\bi\s+want\b", r"\bgoals?\b")
NEXT_ACTION_TRIGGERS = (r"\bnext\s*,?\s*i\b", r"\bthen\s+i\s+will\b")
CONSTRAINT_TRIGGERS = (r"\bmust\b", r"\bcannot\b", r"\bcan'?t\b", r"\bnever\s+deploy\b")
COMMITMENT_TRIGGERS = (r"\bi\s+will\b", r"\bi'?ll\b")

def _compile(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


# Ordered most-specific-first: "next, I will" is a next action, not a bare
# commitment; "never deploy" / "must" is a constraint, not a personal fact.
_KIND_RULES: tuple[tuple[ContinuityKind, tuple[str, ...]], ...] = (
    (ContinuityKind.CORRECTION, CORRECTION_TRIGGERS),
    (ContinuityKind.DECISION, DECISION_TRIGGERS),
    (ContinuityKind.PREFERENCE, PREFERENCE_TRIGGERS),
    (ContinuityKind.PERSONAL_FACT, PERSONAL_FACT_TRIGGERS),
    (ContinuityKind.GOAL, GOAL_TRIGGERS),
    (ContinuityKind.NEXT_ACTION, NEXT_ACTION_TRIGGERS),
    (ContinuityKind.CONSTRAINT, CONSTRAINT_TRIGGERS),
    (ContinuityKind.COMMITMENT, COMMITMENT_TRIGGERS),
)

_KNOWN_PATTERN = re.compile(
    "|".join(f"(?:{pat})" for _, pats in _KIND_RULES for pat in pats),
    re.IGNORECASE,
)

_COMPILED_KIND_RULES = tuple((kind, _compile(pats)) for kind, pats in _KIND_RULES)

_DECISION_TRIGGER = _compile(DECISION_TRIGGERS)


# Rationale carriers for a decision ("because …", "to adopt …").
_WHOLES = r"(?:whole|entire|full|complete)"
_WHY_QUESTION_PATTERN = re.compile(
    rf"\b(?:why|how)\s+(?:did|do|does|would|can|could)\s+(?:we|i)\s+{_WHOLES}?\s*"
    r"(?:[\w$-]+\s+)*?['\u2019]?",
    re.IGNORECASE,
)
_WHY_NOUN_PATTERN = re.compile(
    r"(?:what|which)[^.?\n]{0,80}\b(?:reason|rationale|motive|basis|purpose|goal"
    r"|objective|intention|aim)\b[^.?\n]{0,40}[?]?|\breason\s+for\b[^.?\n]{0,40}[?]?",
    re.IGNORECASE,
)
_BECAUSE_PATTERN = re.compile(
    r"\bbecause\s+of\b|\bbecause\b|\bsince\b|\bdue\s+to\b|\bthanks\s+to\b"
    r"|\bfor\s+this\s+reason\b|\bthe\s+reason\s+being\b",
    re.IGNORECASE,
)
_PURPOSE_PATTERN = re.compile(r"(?<!\w)to\s+[a-z]", re.IGNORECASE)
_REASON_MARKERS = (_WHY_QUESTION_PATTERN, _WHY_NOUN_PATTERN, _BECAUSE_PATTERN, _PURPOSE_PATTERN)

# Filler words carry no subject identity — the slug is the claim's content.
_SLUG_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "cannot", "did", "do", "does", "for", "from", "had",
        "has", "have", "having", "i", "if", "in", "into", "is", "it", "its",
        "itself", "me", "my", "mine", "no", "nor", "not", "of", "off", "on",
        "once", "our", "ours", "so", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "those", "to", "up", "us",
        "was", "we", "were", "what", "when", "where", "which", "who", "whom",
        "why", "will", "with", "would", "you", "your", "yours",
    }
)

# Framing words that describe the speech act, never its subject: "I believe
# X", "correction: X", "confirmed X" are all about X.
_FRAME_WORDS = frozenset(
    {
        "ack", "admit", "admitted", "agree", "assume", "believe", "believed",
        "claim", "claimed", "confess", "confirm", "confirmed", "consider",
        "correct", "corrected", "correction", "deny", "feel", "guess",
        "guessed", "hear", "heard", "hope", "imagine", "insist", "learn",
        "learned", "maintain", "mistaken", "recall", "reckon", "refute",
        "say", "said", "says", "say", "seem", "seems", "speculate", "state",
        "stated", "suppose", "suspect", "tell", "think", "thought",
        "understand", "venture", "verdict", "verify", "verified", "wrong",
    }
)

# Verbs that frame a claim without naming its subject: dropped so "we
# adopted SQLite", "adopt SQLite instead of Postgres" and "SQLite, not
# Postgres" all reduce to the same topic.
_TOPIC_VERBS = frozenset(
    {
        "add", "adopt", "arrang", "build", "carri", "chang", "choos", "chose",
        "convert", "creat", "current", "default", "defin", "deploy",
        "end", "establish", "final", "finish", "implement", "includ",
        "integr", "maintain", "migrat", "mitig", "modif", "mov", "need",
        "plan", "prioritiz", "provid", "reconfigur", "replac", "revis",
        "rollout", "run", "select", "set", "shut", "solv", "start",
        "support", "switch", "transport", "updat", "us", "use", "utiliz",
        "went",
    }
)


def _entry_text(entry: Any) -> str:
    """Flatten an archived entry payload into matchable text (defensive).

    A ``Message`` yields its text blocks; a content block yields its text /
    content string (a ``ToolResultBlock`` carries the raw payload string).
    Unknown payloads contribute nothing — this never raises.
    """
    content = getattr(entry, "content", None)
    blocks = getattr(content, "content", None)
    if blocks is None:
        blocks = [content]
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        value = getattr(block, "text", None)
        if isinstance(value, str):
            parts.append(value)
            continue
        value = getattr(block, "content", None)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts).strip()


def _is_asking(text: str) -> bool:
    return text.rstrip().endswith("?")


def _match_kind(authority: ContextAuthority, text: str) -> ContinuityKind | None:
    """Trigger-phrase kind for a source class, or None when untuned (FR11)."""
    if authority not in _EVIDENCE_AUTHORITIES:
        return None
    if _KNOWN_PATTERN.search(text) is None:
        return None
    asking = _is_asking(text)
    for kind, pattern in _COMPILED_KIND_RULES:
        if pattern.search(text) is None:
            continue
        # A question is someone asking, not someone being bound: "What must
        # I do …?" does not state a constraint, and "Why did we …?" is not a
        # decision. Those fall through as no typed record.
        if kind in (ContinuityKind.CONSTRAINT, ContinuityKind.DECISION) and asking:
            continue
        return kind
    return None


def _fallback_kind(authority: ContextAuthority) -> ContinuityKind:
    """Kind for an entry the phrases do not type.

    FR12: an unsupported agent claim is a tentative EPISODE; an untyped user
    or evidence statement is a PROJECT_FACT.
    """
    if authority not in _EVIDENCE_AUTHORITIES:
        return ContinuityKind.EPISODE
    return ContinuityKind.PROJECT_FACT


def _rationale_from(text: str, window_start: int) -> str | None:
    """Extract a rationale clause ("because …", "to adopt …") from a decision.

    The window opens at the decision verb (or the sentence start for a
    why-question, whose opener itself announces the rationale).
    """
    if window_start < 0:
        return None
    window = text[window_start:]
    for marker in _REASON_MARKERS:
        match = marker.search(window)
        if match is not None:
            return window[match.start() :].strip().rstrip()
    return None


# ---------------------------------------------------------------------------
# Deterministic identity
# ---------------------------------------------------------------------------


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _stem(word: str) -> str:
    """Collapse inflection so "adopted"/"adopt" and "defaults"/"default"
    fold to one token."""
    for suffix in (
        "ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied",
        "edly", "ed", "es", "s", "ly",
    ):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> list[str]:
    """Claim content words: lowercased, stop-words and speech-act framing
    removed, inflection folded."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        _stem(w)
        for w in words
        if w not in _SLUG_STOPWORDS and w not in _FRAME_WORDS
    ]


def _slug(text: str) -> str:
    """Full claim slug — the fallback subject when no shared topic exists."""
    return "-".join(_content_words(text)[:8]) or "claim"


def _subject_id(kind: ContinuityKind, entry_id: str, text: str) -> str:
    # A claim is about whatever it talks about, so a claim-shaped record (an
    # unsupported episode, an untyped statement) is named for its own claim:
    # the identity a later confirmation or correction joins. Typed records
    # key on their own event identity.
    if kind in (ContinuityKind.EPISODE, ContinuityKind.PROJECT_FACT):
        return f"subject-{_slug(text)}"
    return f"subject-{kind.value}-{_digest(kind.value, entry_id)}"


def _object_words(text: str) -> frozenset[str]:
    """The objects a statement names — content words once framing ("I
    believe", "correction:", "confirmed:") and verbs are gone.

    That fold is what lets a correction recognize the fact it replaces and a
    confirmation the claim it confirms with no link on either side, and what
    keeps an unrelated remark ("what is the weather in Oslo") naming objects
    nothing else shares.
    """
    return frozenset(
        word for word in _content_words(text) if word not in _TOPIC_VERBS
    )


def _record_topic_like(subject_id: str) -> str:
    """Topic run of a bare subject id (a link-supplied ``subject-`` prefix is
    not subject matter)."""
    if subject_id.startswith("subject-"):
        return subject_id[len("subject-") :]
    return subject_id


def _record_topic(record: ContinuityRecord) -> str:
    """The subject's topic run — see :func:`_record_topic_like`."""
    return _record_topic_like(record.subject_id)


def _record_objects(record: ContinuityRecord) -> frozenset[str]:
    """Content words of a record's stored claim, memoized by record id (a
    record is immutable, so an id always denotes the same content)."""
    cached = _CONTENT_CACHE.get(record.id)
    if cached is None:
        cached = _object_words(record.content)
        if len(_CONTENT_CACHE) >= _CONTENT_CACHE_LIMIT:
            _CONTENT_CACHE.clear()
        _CONTENT_CACHE[record.id] = cached
    return cached


def _join_subject(record: ContinuityRecord, topic: str) -> bool:
    """True when a recorded subject and a claim's topic name one subject:
    equal runs, or one a token-boundary abbreviation of the other.

    A subject id is a name chosen for a claim ("cache-backend"); a claim
    topic is what the conversation just said about it ("cache-default").
    They agree on their leading token. Different leading tokens are
    different subjects, so an unrelated entry ("weather-oslo") can only ever
    join itself.
    """
    if not topic:
        return False
    recorded = _record_topic(record)
    if not recorded:
        return False
    if topic == recorded:
        return True
    return recorded.startswith(topic + "-") or topic.startswith(recorded + "-")


def _explicit_subject(*subjects: object) -> str | None:
    """First non-empty subject link an entry carries, if any."""
    for subject in subjects:
        if isinstance(subject, str) and subject.strip():
            return subject.strip()
    return None


def _subject_candidates(text: str) -> tuple[str, ...]:
    """Explicit ``subject[-_]`` tokens an entry mentions, deduped in order."""
    found: list[str] = []
    for match in re.finditer(r"subject[-_][a-z0-9][a-z0-9_-]*", text.lower()):
        token = match.group(0)
        if token not in found:
            found.append(token)
    return tuple(found)


def _subject_mentions(subject_id: str, candidates: tuple[str, ...]) -> bool:
    """True when a candidate names *subject_id* exactly or as a prefix run."""
    for candidate in candidates:
        if candidate == subject_id or subject_id.startswith(candidate + "-"):
            return True
    return False


def _event_id(event_kind: ContinuityEventKind, entry_id: str) -> str:
    return f"event-{event_kind.value}-{_digest(event_kind.value, entry_id)}"


# ---------------------------------------------------------------------------
# Current-projection cache (subject -> current record resolution)
# ---------------------------------------------------------------------------


class _CurrentProjectionCache:
    """In-process view of the subjects consolidation has recorded.

    FR14 single-flights consolidation (turn N's post_process completes
    before turn N+1 starts), so this needs no lock and deliberately takes
    none. When a durable continuity store is injected, that ledger is
    consulted first (see ``_subject_current_id``); this cache is the
    fallback that keeps subject resolution working for a consolidator
    constructed without one. It is a cache — never truth, and nothing is
    recorded here that the ledger does not already hold.
    """

    def __init__(self) -> None:
        self._by_subject: dict[str, list[ContinuityRecord]] = {}
        self._superseded: set[str] = set()

    def record(self, record: ContinuityRecord) -> None:
        self._by_subject.setdefault(record.subject_id, []).append(record)
        self._superseded.update(record.supersedes)

    def subjects(self) -> list[str]:
        return list(self._by_subject)

    def current_id(self, subject_id: str) -> str | None:
        chain = self._by_subject.get(subject_id)
        if not chain:
            return None
        for record in reversed(chain):
            if record.id not in self._superseded:
                return record.id
        return None

    def current_records(self) -> list[ContinuityRecord]:
        out: list[ContinuityRecord] = []
        for chain in self._by_subject.values():
            current = None
            for record in reversed(chain):
                if record.id not in self._superseded:
                    current = record
                    break
            if current is not None:
                out.append(current)
        return out

    def clear(self) -> None:
        self._by_subject.clear()
        self._superseded.clear()


_CURRENT_PROJECTIONS = _CurrentProjectionCache()

#: Memoized content-word sets by record id (records are immutable, so an id
#: always denotes the same content). Bounded so a long session cannot grow
#: it without limit.
_CONTENT_CACHE: "dict[str, frozenset[str]]" = {}
_CONTENT_CACHE_LIMIT = 4096

#: Claim object-words -> the subject they named, so a later confirmation or
#: correction that carries no subject link still finds the subject its own
#: words are about. Cleared with the projection cache.
_CLAIM_INDEX: dict[str, list[tuple[str, str]]] = {}


def _remember_claim(subject_id: str, text: str) -> None:
    """Index a claim's object words against the subject it formed."""
    for word in _object_words(text):
        bucket = _CLAIM_INDEX.setdefault(word, [])
        if (subject_id, text) not in bucket:
            bucket.append((subject_id, text))
    if len(_CLAIM_INDEX) > _CONTENT_CACHE_LIMIT:
        _CLAIM_INDEX.clear()


def reset_current_projection_cache() -> None:
    """Drop the module-level projection caches.

    Tests want a clean slate; production consolidation never calls this —
    the durable ledger, not these caches, is truth.
    """
    _CURRENT_PROJECTIONS.clear()
    _CLAIM_INDEX.clear()
    _CONTENT_CACHE.clear()


# ---------------------------------------------------------------------------
# The consolidator
# ---------------------------------------------------------------------------


class RuleBasedContinuityConsolidator:
    """Deterministic ``ContinuityConsolidator`` over the exact turn archive.

    Implements the spec Protocol::

        async def consolidate(request: ConsolidationRequest) -> list[ContinuityRecord]

    It returns records; appending them to the continuity ledger is the
    caller's job (the orchestrator's post-process barrier does exactly that,
    one turn of lag behind, FR14).

    Current-subject resolution (supersession + promotion) consults the
    injected continuity store when one is supplied and otherwise the
    module-level current-projection cache, so a correction links the record
    it replaces and a confirmation promotes the claim it confirms in both
    wirings.
    """

    def __init__(
        self,
        provenance: ProvenanceStore,
        store: Any | None = None,
    ) -> None:
        self._provenance = provenance
        self._store = store

    async def consolidate(
        self, request: ConsolidationRequest
    ) -> list[ContinuityRecord]:
        """Classify every resolvable entry of one turn into a record.

        Deterministic and idempotent per entry: each source entry yields at
        most one record, so re-consolidating a turn never forks a subject.
        """
        records: list[ContinuityRecord] = []
        for entry_id in request.entry_ids:
            entry = await self._provenance.get(entry_id)
            if entry is None:
                continue
            record = await self._consolidate_entry(entry, request)
            if record is not None:
                records.append(record)
                _CURRENT_PROJECTIONS.record(record)
                _remember_claim(record.subject_id, record.content)
        return records

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _consolidate_entry(
        self, entry: Any, request: ConsolidationRequest
    ) -> ContinuityRecord | None:
        meta = getattr(entry, "meta", None) or {}
        text = _entry_text(entry)
        authority = entry.authority
        about_subject = meta.get("about_subject")
        supersedes_subject = meta.get("supersedes_subject")
        confirmation = bool(CONFIRMATION_PATTERN.match(text))

        kind = _match_kind(authority, text)
        if kind is None:
            kind = _fallback_kind(authority)

        # AC6 promotion: a USER confirmation for a recorded subject appends
        # a NEW VERIFIED event (authority VERIFIED, supersedes empty, its own
        # provenance) which makes the subject factual-current; the record it
        # vouches for survives byte-identical (FR12: acceptance promotes
        # through a new immutable event). A confirmation that resolves to no
        # recorded subject mints nothing — it rides on the claim it confirms
        # or it is not knowledge (FR12).
        if authority is ContextAuthority.USER and confirmation:
            subject_id = await self._promotion_subject(
                text, about_subject, supersedes_subject
            )
            if subject_id is None:
                return None
            return self._verification(request, entry.id, text, subject_id)

        # Subject identity (FR13): an explicit link names the subject; a
        # correction joins the recorded subject its text is about; anything
        # else forms a new subject from its own claim.
        subject_id = _explicit_subject(about_subject, supersedes_subject)
        if kind is ContinuityKind.CORRECTION and subject_id is None:
            # A correction restates the fact it replaces, so it joins that
            # fact's subject when one is recorded.
            matched = await self._match_current_record(text)
            if matched is not None:
                subject_id = matched.subject_id
        if subject_id is None:
            subject_id = _subject_id(kind, entry.id, text)

        # AC7 supersession: a CORRECTION names the current record id of its
        # subject in the ledger at consolidate-time — empty tuple when that
        # subject is unknown to the store.
        supersedes: tuple[str, ...] = ()
        if kind is ContinuityKind.CORRECTION:
            current = await self._subject_current_id(subject_id)
            if current is not None:
                supersedes = (current,)

        record_meta: dict[str, Any] = {}
        if kind is ContinuityKind.DECISION:
            rationale = _rationale_from(text, _decision_window_start(text))
            if rationale:
                record_meta["rationale"] = rationale
        if isinstance(about_subject, str) and about_subject:
            record_meta["about_subject"] = about_subject
        if isinstance(supersedes_subject, str) and supersedes_subject:
            record_meta["supersedes_subject"] = supersedes_subject

        return self._record(
            request=request,
            entry_id=entry.id,
            event_kind=ContinuityEventKind.FORMED,
            kind=kind,
            content=text,
            subject_id=subject_id,
            authority=authority,
            supersedes=supersedes,
            meta=record_meta,
        )

    def _verification(
        self, request: ConsolidationRequest, entry_id: str, text: str, subject_id: str
    ) -> ContinuityRecord:
        return self._record(
            request=request,
            entry_id=entry_id,
            event_kind=ContinuityEventKind.VERIFIED,
            kind=ContinuityKind.PROJECT_FACT,
            content=text,
            subject_id=subject_id,
            authority=ContextAuthority.VERIFIED,
            supersedes=(),
        )

    def _record(
        self,
        *,
        request: ConsolidationRequest,
        entry_id: str,
        event_kind: ContinuityEventKind,
        kind: ContinuityKind,
        content: str,
        subject_id: str,
        authority: ContextAuthority,
        supersedes: tuple[str, ...],
        meta: dict[str, Any] | None = None,
    ) -> ContinuityRecord:
        projects = () if kind in PERSONAL_KINDS else tuple(sorted(request.project_hints))
        return ContinuityRecord(
            id=_event_id(event_kind, entry_id),
            subject_id=subject_id,
            event_kind=event_kind,
            kind=kind,
            content=content,
            scope=ContextScope(
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                projects=projects,
            ),
            authority=authority,
            provenance_ids=(entry_id,),  # FR5: the one real source entry
            supersedes=supersedes,
            confidence=0.4 if authority is ContextAuthority.AGENT_TENTATIVE else 0.9,
            salience=0.5,
            created_at=datetime.now(tz=timezone.utc),
            meta=meta or {},
        )

    # -- current-subject resolution ------------------------------------

    async def _promotion_subject(
        self, text: str, about_subject: object, supersedes_subject: object
    ) -> str | None:
        """Subject a USER confirmation promotes, or None when it vouches for
        nothing recorded (FR12).

        The confirmation names its subject outright when it can (an
        ``about_subject`` / ``supersedes_subject`` link, resolved exactly or
        as an abbreviation of a recorded subject); otherwise it is about the
        recorded claim its own words restate.
        """
        named = _explicit_subject(about_subject, supersedes_subject)
        if named is not None:
            if await self._subject_current_id(named) is not None:
                return named
        record = await self._match_current_record(text)
        if record is not None:
            return record.subject_id
        return None

    async def _current_records(self) -> list[ContinuityRecord]:
        if self._store is not None:
            try:
                views = await self._store.current_views()  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - defensive
                logger.exception("continuity projection read failed during consolidation")
                return []
            return [view.current_record for view in views.values()]
        return _CURRENT_PROJECTIONS.current_records()

    async def _match_subject(self, subject_id: str) -> ContinuityRecord | None:
        """Recorded subject an unknown subject id abbreviates, if any.

        ``subject-cache`` resolves to ``subject-cache-backend`` — an operator
        or a link names a subject by hand, and the ledger already holds it
        under a longer id. Token-run containment, no score.
        """
        topic = _record_topic_like(subject_id)
        if not topic:
            return None
        for record in await self._current_records():
            recorded = _record_topic(record)
            if topic == recorded or recorded.startswith(topic + "-"):
                return record
        return None

    async def _match_current_record(self, text: str) -> ContinuityRecord | None:
        """Current record an entry is about, or None.

        Two deterministic paths, no scores and no fuzzy thresholds:

        1. an explicit ``subject[-_]`` token in the entry naming a recorded
           subject exactly or as a prefix — how an operator, or the
           ``meta`` links (``about_subject``, ``supersedes_subject``), point
           at a subject outright;
        2. a shared object word: the entry and a recorded claim name the same
           thing once framing and verbs are gone ("correction: we adopted
           SQLite, not Postgres" and "The ledger uses Postgres" are both
           about Postgres). A word naming several recorded subjects is
           ambiguous and joins nothing, so unrelated conversation can never
           reach across into someone else's fact by accident.
        """
        records = await self._current_records()
        for candidate in _subject_candidates(text):
            for record in records:
                if _subject_mentions(record.subject_id, (candidate,)):
                    return record
        objects = _object_words(text)
        for word in sorted(objects):
            hits = [record for record in records if word in _record_objects(record)]
            if len({record.subject_id for record in hits}) == 1:
                return hits[0]
        return None

    async def _match_subject(self, subject_id: str) -> ContinuityRecord | None:
        """Recorded subject an unknown subject id abbreviates, if any.

        ``subject-cache`` resolves to ``subject-cache-backend``; an id with
        no token in common with a recorded subject names nothing.
        """
        topic = _record_topic_like(subject_id)
        if not topic:
            return None
        for record in await self._current_records():
            if _join_subject(record, topic):
                return record
        return None

    async def _subject_current_id(self, subject_id: str) -> str | None:
        """Current-projection record id for a subject, or None if unknown.

        The injected ledger is authoritative when present; the current-
        projection cache backs a store-less consolidator. Reads frozen
        public surfaces only; a read failure degrades to "nothing to link"
        rather than losing the turn's consolidation (FR14).
        """
        if self._store is not None:
            try:
                views = await self._store.current_views()  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "continuity projection read failed during consolidation "
                    "(subject=%s); continuing without supersession links",
                    subject_id,
                )
            else:
                view = views.get(subject_id)
                if view is not None:
                    return view.current_record.id
        current = _CURRENT_PROJECTIONS.current_id(subject_id)
        if current is not None:
            return current
        # An unknown subject link can still abbreviate a recorded one
        # ("subject-cache" for "subject-cache-backend"); resolve it before
        # treating the subject as new.
        matched = await self._match_subject(subject_id)
        return matched.subject_id if matched is not None else None


def _decision_window_start(text: str) -> int:
    """Offset the rationale window opens at: after the decision verb, or the
    sentence start for a why-question."""
    if _is_asking(text):
        return 0
    match = _DECISION_TRIGGER.search(text)
    return match.end() if match else 0
