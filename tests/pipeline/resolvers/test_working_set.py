"""sr2-94 / AC4 — Bounded working set (unit layer, resolver half of the cage).

Cage for FR6 (bounded working set: compiled prompt stays within a configured
budget regardless of archive age or turn count) and FR7 (residency rules:
current user input pinned for the turn; lifecycle-pinned content retained;
recent dialogue bounded by its own window), plus AC4's deterministic 500-turn
endurance and the backward-compat row (continuity-disabled configurations
retain the legacy full-replay behavior).

Case-table rows covered here: AC4-r1, r2, r3, r4, r5, r6.
Rows AC4-r7, r8, r9 live in tests/config/test_continuity_config.py.

Contract the rows fix (module path fixed by the bead; internals free):

    from sr2.pipeline.resolvers.working_set import WorkingSetResolver

    WorkingSetResolver.build(ResolverConfig, Dependencies) — Resolver.build
    classmethod; reads "recent_token_budget" (int, required) and
    "hard_token_budget" (int, optional) from ResolverConfig.config. The
    provenance store reaches the selector through its Dependencies (the
    repo's DI container — a typed field, a harness-supplied wrapper, or any
    other route the implementer contracts with; the frozen-ness of
    Dependencies is pinned elsewhere and is not this resolver's to change,
    so the fixture wraps Dependencies and carries the store on the wrapper)
    and the session id through Dependencies.session_id. The store is read
    through the ProvenanceStore protocol (get_session).

    resolve() fires at least once per turn with the turn's user_input event
    (Resolver protocol: build() sets ``max_executions`` from
    ResolverConfig; the fixture passes max_executions=10000 so the
    endurance row's 500 firings are in contract).

    async resolve([Event("user_input", STARTING, data=list[ContentBlock])])
    -> ResolvedContent whose ``content`` is the bounded working set for the
    turn. Whether the selector reports its own token_count is not pinned —
    every budget assertion here recounts with CharacterTokenCounter (the
    repo's deterministic counter, chars // 4).

Entry-shape facts fixed by the existing provenance model (this cage builds
fixtures with them; the resolver reads them through the store):

    layer == "turn_archive" (sr2-93), lifecycle ARCHIVE | PINNED,
    content is a Message (role user/assistant), created_at strictly
    increasing with turn order.

Anti-over-specification: item ordering inside the returned content, the
ResolvedContent.token_count field, resolver/source-layer names, private
attributes, and the eviction algorithm are NOT pinned — only boundedness,
non-droppage of current input and pinned items, and (in the endurance row)
that the current turn's own archived user message may serve as the window
content when the selector does not separately materialize the live event.

Red mechanism (deliberate): WorkingSetResolver is imported at module level
from its declared landing spot — until it exists, this module fails
collection with ModuleNotFoundError naming exactly that module: the
expected red, not scaffolding. AC4-r6 (legacy SessionResolver, unchanged
behavior) lives in its own module WITHOUT that import, so it passes today
and keeps passing — the backward-compat guard, never masked by the red.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ulid import ULID

from sr2.config.models import ResolverConfig
from sr2.models import Message, TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.provenance import (
    TURN_ARCHIVE_LAYER,
    ContextLifecycle,
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
)
from sr2.pipeline.resolvers.working_set import WorkingSetResolver
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SESSION_ID = "working-set-session"

_COUNTER = CharacterTokenCounter()

_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

_RESOLVER_ORIGIN = EntryOrigin(kind="resolver", name="working_set_fixture")


def _padded(prefix: str, target_tokens: int) -> str:
    """Text of exactly *target_tokens* tokens under CharacterTokenCounter.

    Pads to exactly 4 * target_tokens characters (chars // 4 == target).
    """
    total = 4 * target_tokens
    assert len(prefix) <= total, (
        f"prefix {prefix!r} is longer than {target_tokens} tokens; cannot pad"
    )
    text = prefix + "x" * (total - len(prefix))
    assert _COUNTER.count([TextBlock(text=text)]) == target_tokens
    return text


def _message(text: str, role: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


def _msg_tokens(msg: Message) -> int:
    return _COUNTER.count(list(msg.content))


def _make_message_entry(
    text: str,
    role: str,
    lifecycle: ContextLifecycle,
    seq: int,
) -> Entry:
    """Archive-shaped entry: turn_archive layer, ULID id, created_at ordered
    by *seq* (the provenance model itself pins no ordering — the fixture
    supplies one)."""
    return Entry(
        id=str(ULID()),
        content=_message(text, role),
        sources=(),
        origin=_RESOLVER_ORIGIN,
        layer=TURN_ARCHIVE_LAYER,
        session_id=SESSION_ID,
        created_at=_BASE_TIME + timedelta(seconds=seq),
        lifecycle=lifecycle,
    )


def _archived_turn(
    n_turns: int, tokens_per_message: int
) -> list[Entry]:
    """n_turns archived user+assistant pairs, each Message exactly
    *tokens_per_message* tokens under CharacterTokenCounter."""
    entries: list[Entry] = []
    for i in range(n_turns):
        entries.append(
            _make_message_entry(
                _padded(f"U{i:04d}", tokens_per_message),
                "user",
                ContextLifecycle.ARCHIVE,
                i * 2,
            )
        )
        entries.append(
            _make_message_entry(
                _padded(f"A{i:04d}", tokens_per_message),
                "assistant",
                ContextLifecycle.ARCHIVE,
                i * 2 + 1,
            )
        )
    return entries


async def _build_store(entries: list[Entry]) -> InMemoryProvenanceStore:
    store = InMemoryProvenanceStore()
    for entry in entries:
        await store.write(entry)
    return store


def _ws_config(
    recent_token_budget: int, hard_token_budget: int | None = None
) -> ResolverConfig:
    cfg: dict = {"recent_token_budget": recent_token_budget}
    if hard_token_budget is not None:
        cfg["hard_token_budget"] = hard_token_budget
    # max_executions high: the endurance row fires once per turn for 500
    # turns; every resolve() here is an in-contract firing.
    return ResolverConfig(type="working_set", config=cfg, max_executions=10000)


class _WrappedDependencies:
    """Dependencies wrapper carrying the provenance store.

    Dependencies is a frozen dataclass whose frozen-ness and exact field set
    are pinned by tests/pipeline/test_dependencies_and_protocols.py — an
    add_field here would break that suite, which is outside this bead. So
    the fixture injects the store on a duck-typed wrapper: build() reads
    attributes off ``deps``, so both a plain Dependencies and this wrapper
    satisfy it, and the implementer stays free to source the store via a
    new typed field (harness passes the real thing), a wrapper read, or DI
    introspection. What is fixed: the store reaches the resolver through
    the Dependencies argument of build(), and the session id through
    ``deps.session_id``.
    """

    def __init__(self, inner: Dependencies, provenance_store: object) -> None:
        self._inner = inner
        self.provenance_store = provenance_store

    def __getattr__(self, name: str):  # delegate every other attribute
        return getattr(self._inner, name)


def _build_resolver(
    recent_token_budget: int,
    store: InMemoryProvenanceStore,
    hard_token_budget: int | None = None,
) -> WorkingSetResolver:
    """Build the selector through the Resolver.build classmethod (the
    construction path the pipeline itself uses)."""
    deps = _WrappedDependencies(
        Dependencies(session_id=SESSION_ID), provenance_store=store
    )
    return WorkingSetResolver.build(
        _ws_config(recent_token_budget, hard_token_budget), deps
    )


def _user_input_event(text: str) -> Event:
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="core",
        data=[TextBlock(text=text)],
    )


def _all_text(result: ResolvedContent) -> str:
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, Message):
            blocks = item.content if isinstance(item.content, list) else []
            parts.extend(b.text for b in blocks if isinstance(b, TextBlock))
        elif isinstance(item, TextBlock):
            parts.append(item.text)
    return "\n".join(parts)


def _count_output(result: ResolvedContent) -> int:
    """Recount the working set with CharacterTokenCounter.

    Messages and ContentBlocks are the shapes a messages-target resolver
    returns; any other item type is out of contract and fails loudly rather
    than silently counting as zero.
    """
    total = 0
    for item in result.content:
        if isinstance(item, Message):
            blocks = item.content if isinstance(item.content, list) else []
            total += _COUNTER.count(blocks)
        elif isinstance(item, TextBlock):
            total += _COUNTER.count([item])
        else:
            raise AssertionError(
                f"working-set content item {item!r} is neither Message nor "
                "ContentBlock; the selector must return compileable content"
            )
    return total


# ---------------------------------------------------------------------------
# AC4-r1 — recent window is bounded (FR6)
# ---------------------------------------------------------------------------


class TestAC4R1RecentWindowBounded:
    async def test_50_archived_turns_compile_within_recent_budget(self):
        """AC4-r1: 50 archived turns (~50 tokens/Message) with
        recent_token_budget=200 -> resolve() content recounts to <= 200.
        The archive holds ~5000 tokens, so any full replay overshoots."""
        store = await _build_store(_archived_turn(50, 50))
        resolver = _build_resolver(200, store)
        result = await resolver.resolve([_user_input_event("r1 current input")])
        assert isinstance(result, ResolvedContent)
        total = _count_output(result)
        assert total <= 200, (
            f"recent window must bound the compiled working set at 200 "
            f"tokens; got {total}"
        )


# ---------------------------------------------------------------------------
# AC4-r2 — current user input is pinned for the turn (FR7)
# ---------------------------------------------------------------------------


class TestAC4R2CurrentInputPinned:
    async def test_current_user_input_present_in_compiled_working_set(self):
        """AC4-r2: the user_input event's text appears in the returned
        content — the current input is never dropped."""
        store = await _build_store(_archived_turn(50, 50))
        resolver = _build_resolver(200, store)
        result = await resolver.resolve(
            [_user_input_event("CURRENT_INPUT_MARKER")]
        )
        assert "CURRENT_INPUT_MARKER" in _all_text(result), (
            "current user input must be pinned into the working set for the "
            f"turn; output text was {_all_text(result)!r}"
        )


# ---------------------------------------------------------------------------
# AC4-r3 — lifecycle-pinned content survives window pressure (FR7)
# ---------------------------------------------------------------------------


class TestAC4R3PinnedContentRetained:
    async def test_pinned_entry_survives_oversized_recent_window(self):
        """AC4-r3: one PINNED entry ("PINNED_CONTENT") plus 50 archived
        turns with recent_token_budget=200 (far too small for all) -> the
        pinned message is still in the output."""
        pinned = _make_message_entry(
            _padded("PINNED_CONTENT", 50),
            "user",
            ContextLifecycle.PINNED,
            seq=0,
        )
        entries = [pinned] + _archived_turn(50, 50)
        # sanity: fixtures really are 50 tokens per Message
        assert _msg_tokens(pinned.content) == 50
        assert _msg_tokens(entries[1].content) == 50

        store = await _build_store(entries)
        resolver = _build_resolver(200, store)
        result = await resolver.resolve([_user_input_event("r3 current input")])
        assert "PINNED_CONTENT" in _all_text(result), (
            "lifecycle-pinned (system/identity invariant) content must never "
            f"be evicted; output text was {_all_text(result)!r}"
        )


# ---------------------------------------------------------------------------
# AC4-r4 — bound holds regardless of archive size (FR6)
# ---------------------------------------------------------------------------


class TestAC4R4BoundIndependentOfArchiveSize:
    async def test_500_archived_turns_compile_within_150_token_budget(self):
        """AC4-r4: 500 archived turns with recent_token_budget=150 ->
        content recounts to <= 150 regardless of archive size."""
        store = await _build_store(_archived_turn(500, 50))
        resolver = _build_resolver(150, store)
        result = await resolver.resolve([_user_input_event("r4 current input")])
        total = _count_output(result)
        assert total <= 150, (
            f"the recent window must hold at 150 tokens regardless of "
            f"archive age/size; got {total} from a 500-turn archive"
        )


# ---------------------------------------------------------------------------
# AC4-r5 — deterministic 500-turn endurance (AC4)
# ---------------------------------------------------------------------------


class TestAC4R5FiveHundredTurnEndurance:
    async def test_500_turn_simulation_every_turn_bounded_and_current(self):
        """AC4-r5: deterministic 500-turn simulation. Each turn calls
        resolve() with that turn's user_input event (recent_token_budget=
        100) and then appends a user Message and an assistant Message
        (exactly 20 tokens each) to the durable archive, as the turn
        archivist would. For EVERY turn N in 1..500:

          - the output recounts to <= 100 tokens,
          - the current turn's input text is present in the output,
          - no manual reset occurs: this fixture never calls reset()/clear()/
            wipe() — boundedness must come from the selector itself, turn
            after turn, as the archive grows to 1000 entries.

        The current turn's own user Message is archived under the same text
        as the live event, so a selector that surfaces the current input via
        the freshly archived window entry satisfies "present" exactly as one
        that materializes the live event does — either is compliant.
        """
        budget = 100
        store = InMemoryProvenanceStore()
        resolver = _build_resolver(budget, store)

        for n in range(1, 501):
            live_text = _padded(f"WSR5LIVE{n:04d}", 20)
            archived_text = _padded(f"WSR5TURN{n:04d}", 20)
            assert _msg_tokens(_message(live_text, "user")) == 20

            result = await resolver.resolve([_user_input_event(live_text)])

            total = _count_output(result)
            assert total <= budget, (
                f"turn {n}: compiled working set {total} tokens exceeds the "
                f"{budget}-token budget"
            )
            assert live_text in _all_text(result), (
                f"turn {n}: current user input (live event) missing from the "
                "compiled working set"
            )

            # close the turn: append user + assistant (20 tokens each)
            await store.write(
                _make_message_entry(
                    archived_text, "user", ContextLifecycle.ARCHIVE, n * 2
                )
            )
            await store.write(
                _make_message_entry(
                    _padded(f"WSR5A{n:04d}", 20),
                    "assistant",
                    ContextLifecycle.ARCHIVE,
                    n * 2 + 1,
                )
            )
