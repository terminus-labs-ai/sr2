"""Layer — the core unit of the SR2 pipeline.

Holds resolvers and transformers, accumulates content, enforces token
budgets, and compiles content for its compilation target.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
from ulid import ULID
from sr2.pipeline.compilation import PositionStrategy, get_compilation_targets
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.protocols import Resolver, Seedable, TokenCounter, ToolProvider, Transformer
from sr2.pipeline.provenance import (
    ContextAuthority,
    ContextLifecycle,
    ContextReason,
    ContextScope,
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
    ProvenanceStore,
)

from sr2.pipeline.tracing import FiringRecord

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer


# ---------------------------------------------------------------------------
# Component dispatch strategy (sr2-75: DRY + strategy pattern)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ComponentStrategy:
    """Strategy for a single component kind (resolver / transformer / tool_provider).

    Replaces the kind-string if/elif branching in _fire_component and
    _snapshot_after with a data-driven dispatch table. Each strategy defines:

    - invoke: async callable(comp, events, layer) -> result
    - snapshot: callable(layer) -> (content_snapshot, tokens)
    - apply: callable(result, layer) — side effects (add_content, set_content, etc.)
    """
    invoke: Callable[..., Any]
    snapshot: Callable[..., tuple[list, int]]
    apply: Callable[..., None]


def _snapshot_content(layer: "Layer") -> tuple[list, int]:
    content = list(layer._content)
    tokens = layer._token_counter.count(content)
    return content, tokens


def _snapshot_tools(layer: "Layer") -> tuple[list, int]:
    return [td.name for td in layer._tool_definitions], 0


def _apply_resolver(result: Any, layer: "Layer") -> None:
    layer.add_content(result)


def _apply_transformer(result: Any, layer: "Layer") -> None:
    if result.content is not None:
        layer.set_content(result.content)


def _apply_tool_provider(result: Any, layer: "Layer") -> None:
    layer.add_tool_definitions(result)


# Strategy lookup keyed by kind string.
# Keeping the string key preserves backward compatibility with existing
# callers that pass kind="resolver" etc. to _fire_component().
_COMPONENT_DISPATCH: dict[str, _ComponentStrategy] = {
    "resolver": _ComponentStrategy(
        invoke=lambda comp, events, layer: comp.resolve(events),
        snapshot=_snapshot_content,
        apply=_apply_resolver,
    ),
    "transformer": _ComponentStrategy(
        invoke=lambda comp, events, layer: comp.transform(layer.get_content(), events),
        snapshot=_snapshot_content,
        apply=_apply_transformer,
    ),
    "tool_provider": _ComponentStrategy(
        invoke=lambda comp, events, layer: comp.provide(events),
        snapshot=_snapshot_tools,
        apply=_apply_tool_provider,
    ),
}


# ---------------------------------------------------------------------------
# Provenance synthesis for raw content (sr2-92 — no-store bypass removed)
# ---------------------------------------------------------------------------

_CAUSAL_ENUMS: dict[str, type] = {
    "reason": ContextReason,
    "authority": ContextAuthority,
    "lifecycle": ContextLifecycle,
}

# Known production components whose raw output carries a fixed causal
# classification until they emit provenance-aware Entries themselves
# (resolver names and source layers, from sr2.pipeline.resolvers and
# sr2.memory). Unknown names fall through to structural inference.
_PRODUCTION_CAUSAL: dict[str, tuple[ContextReason, ContextAuthority, ContextLifecycle]] = {
    "input": (ContextReason.USER_INPUT, ContextAuthority.USER, ContextLifecycle.TURN),
    "session": (ContextReason.USER_INPUT, ContextAuthority.USER, ContextLifecycle.RECENT),
    "static": (ContextReason.SYSTEM_CONFIG, ContextAuthority.VERIFIED, ContextLifecycle.PINNED),
    "static_text": (
        ContextReason.SYSTEM_CONFIG,
        ContextAuthority.VERIFIED,
        ContextLifecycle.PINNED,
    ),
    "markdown_file": (
        ContextReason.LIVE_SOURCE,
        ContextAuthority.LIVE_SOURCE,
        ContextLifecycle.PROJECT,
    ),
    "memory": (ContextReason.RETRIEVAL, ContextAuthority.DERIVED, ContextLifecycle.RETRIEVED),
    "assistant_output": (
        ContextReason.ASSISTANT_OUTPUT,
        ContextAuthority.AGENT_TENTATIVE,
        ContextLifecycle.RECENT,
    ),
    "tool_events": (ContextReason.TOOL_USE, ContextAuthority.AGENT_TENTATIVE, ContextLifecycle.TURN),
}


def _coerce_causal(declared: dict | None) -> dict:
    """Normalize a provenance declaration into Entry keyword fields.

    Unknown enum values raise ValueError here — at the boundary — rather
    than persisting an invalid Entry.
    """
    out: dict = {}
    for name, enum_cls in _CAUSAL_ENUMS.items():
        value = (declared or {}).get(name)
        if value is None:
            continue
        try:
            out[name] = enum_cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown provenance {name} {value!r}; "
                f"expected one of {[e.value for e in enum_cls]}"
            ) from None
    return out


def _causal_for_item(
    item: ContentBlock | Message,
    source_layer: str,
    declared: dict | None = None,
    default_kind: str = "resolver",
    origin_name: str = "",
) -> dict:
    """Derive the causal fields for one raw compiled item (FR2).

    Priority (highest wins):

    1. ``declared`` — the producer's per-item provenance declaration
       (``ResolvedContent.provenance`` / ``TransformationResult.provenance``,
       parallel to ``content``).
    2. ``item.meta["provenance"]`` — per-block declaration carried on the
       content block itself.
    3. The item's own shape (Message role, block type).
    4. Declared dialogue/live-source component vocabulary (``input``,
       ``session``, ``static``, ``markdown_file``, ...).
    5. Origin kind (transformer output is a transformation).

    The vocabulary is the spec's generic enums only.
    """
    meta: dict[str, Any] = {}

    kind = getattr(item, "type", None)
    role = getattr(item, "role", None) if isinstance(item, Message) else None
    reason: ContextReason
    authority: ContextAuthority
    lifecycle: ContextLifecycle

    # A Message's own dialogue role is the strongest structural signal.
    if role == "user":
        reason, authority, lifecycle = (
            ContextReason.USER_INPUT,
            ContextAuthority.USER,
            ContextLifecycle.RECENT,
        )
    elif role == "assistant":
        reason, authority, lifecycle = (
            ContextReason.ASSISTANT_OUTPUT,
            ContextAuthority.AGENT_TENTATIVE,
            ContextLifecycle.RECENT,
        )
    elif kind == "tool_use":
        reason, authority, lifecycle = (
            ContextReason.TOOL_USE,
            ContextAuthority.AGENT_TENTATIVE,
            ContextLifecycle.TURN,
        )
    elif kind == "tool_result":
        reason, authority, lifecycle = (
            ContextReason.TOOL_RESULT,
            ContextAuthority.DERIVED,
            ContextLifecycle.TURN,
        )
    elif origin_name in _PRODUCTION_CAUSAL:
        reason, authority, lifecycle = _PRODUCTION_CAUSAL[origin_name]
        if reason is ContextReason.LIVE_SOURCE and isinstance(item, TextBlock):
            meta["content_hash"] = hashlib.sha256(
                (item.text or "").encode("utf-8")
            ).hexdigest()
    elif source_layer in _PRODUCTION_CAUSAL:
        reason, authority, lifecycle = _PRODUCTION_CAUSAL[source_layer]
        if reason is ContextReason.LIVE_SOURCE and isinstance(item, TextBlock):
            meta["content_hash"] = hashlib.sha256(
                (item.text or "").encode("utf-8")
            ).hexdigest()
    elif default_kind == "transformer":
        reason, authority, lifecycle = (
            ContextReason.TRANSFORMATION,
            ContextAuthority.DERIVED,
            ContextLifecycle.TURN,
        )
    else:
        reason, authority, lifecycle = (
            ContextReason.SYSTEM_CONFIG,
            ContextAuthority.DERIVED,
            ContextLifecycle.TURN,
        )

    result: dict[str, Any] = {
        "reason": reason,
        "authority": authority,
        "lifecycle": lifecycle,
    }

    # --- 2: per-block declaration on the content item -------------------
    block_declared = getattr(item, "meta", None)
    if isinstance(block_declared, dict):
        block_declared = block_declared.get("provenance")
        if isinstance(block_declared, dict):
            result.update(_coerce_causal(block_declared))
            if block_declared.get("source_ref") is not None:
                meta["source_ref"] = block_declared["source_ref"]

    # --- 1: producer declaration on the resolved content ----------------
    explicit_lifecycle = (declared or {}).get("lifecycle")
    result.update(_coerce_causal(declared))
    for key in ("source_ref", "revision"):
        if declared and declared.get(key) is not None:
            meta[key] = declared[key]

    if explicit_lifecycle is None and origin_name == "input":
        # Fresh current-turn user input (InputResolver wraps the live
        # user_input event; replayed dialogue comes from the session
        # resolver instead).
        result["lifecycle"] = ContextLifecycle.TURN

    if result["reason"] is ContextReason.LIVE_SOURCE:
        # External live/config/file sources carry a stable source reference,
        # revision/hash, and observation time (spec, Interfaces).
        meta.setdefault("source_ref", source_layer)
        meta.setdefault("revision", meta.get("content_hash", ""))
        meta.setdefault("observed_at", datetime.now(tz=timezone.utc).isoformat())

    if meta:
        result["meta"] = meta
    return result


def _content_hash_id(item: object) -> str:
    """Content-addressed entry id: 26 chars from SHA-256 of canonical JSON.

    Repeated compilations of unchanged content produce the same id, so the
    append-only stores deduplicate re-persisted items instead of growing a
    new row per turn. Falls back to a fresh ULID for payloads that cannot
    be serialized deterministically.
    """
    try:
        digest = hashlib.sha256(
            item.model_dump_json().encode("utf-8")  # type: ignore[attr-defined]
        ).hexdigest()
    except (AttributeError, TypeError, ValueError):
        return str(ULID())
    chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 alphabet
    num = int(digest, 16)
    out = ""
    for _ in range(26):
        out = chars[num % 32] + out
        num //= 32
    return out


def _synthesize_entries(
    items: list,
    declarations: list | None,
    origin_name: str,
    origin_kind: str,
    source_layer: str,
    default_sources: tuple[str, ...] = (),
) -> list[Entry]:
    """Build one provenance Entry per raw content item (FR1).

    The real item object is kept as the Entry payload (identity is
    preserved so truncation and compile operate on the same objects); only
    causal fields are derived. ``default_sources`` applies only to origins
    that require sources (transformer); resolver-origin entries stay
    genesis.
    """
    entries: list[Entry] = []
    for index, item in enumerate(items):
        declared = (
            declarations[index]
            if declarations and index < len(declarations)
            and isinstance(declarations[index], dict)
            else None
        )
        entry_id = _content_hash_id(item)
        if declared and declared.get("id"):
            entry_id = declared["id"]
        sources = tuple(declared.get("sources") or ()) if declared else ()
        if not sources and origin_kind == "transformer":
            sources = default_sources
        origin = EntryOrigin(
            kind="transformer" if sources else origin_kind, name=origin_name
        )
        if origin.kind == "transformer" and not sources:
            origin = EntryOrigin(kind="resolver", name=origin_name)
        fields = _causal_for_item(
            item, source_layer, declared, origin_kind, origin_name
        )
        entries.append(
            Entry(
                id=entry_id,
                content=item,
                sources=sources,
                origin=origin,
                layer=source_layer,
                session_id=(declared or {}).get("session_id") or "default",
                created_at=datetime.now(tz=timezone.utc),
                scope=ContextScope(owner_id="default"),
                meta=fields.pop("meta", {}),
                **fields,
            )
        )
    return entries


class Layer:
    """A single pipeline layer that accumulates, budgets, and compiles content."""

    _COMPONENT_DISPATCH: dict[str, _ComponentStrategy] = _COMPONENT_DISPATCH

    def __init__(
        self,
        name: str,
        target: CompilationTarget,
        position: PositionStrategy,
        token_budget: int | None,
        resolvers: list[Resolver],
        transformers: list[Transformer],
        token_counter: TokenCounter,
        event_bus: EventBus | None = None,
        provenance_store: ProvenanceStore | None = None,
        token_threshold_pct: float | None = None,
        tool_providers: list | None = None,
        tracer: "Tracer | None" = None,
        degradation_category: str | None = None,
        priority: int = 0,
        condition: str | None = None,
        run_context_provider: Callable[[], dict[str, str] | None] | None = None,
    ) -> None:
        self.name = name
        self.target = target
        self._position = position
        self.token_budget = token_budget
        self.token_threshold_pct = token_threshold_pct
        self.resolvers = resolvers
        self.transformers = transformers
        self.tool_providers = list(tool_providers) if tool_providers is not None else []
        self._token_counter = token_counter
        self._event_bus: EventBus | None = event_bus
        self._provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )

        self._tracer: "Tracer | None" = tracer

        self._turn_seq: int = 0
        self._next_firing_seq: Callable[[], int] = lambda: 0  # replaced via begin_turn()

        self._content: list[ContentBlock | Message] = []
        self._tool_definitions: list[ToolDefinition] = []
        self._force_truncated: bool = False
        # Collected events from bus callbacks — processed by engine after drain
        self._pending_events: list[Event] = []
        # Entries buffered for store write — flushed in process_pending
        self._pending_writes: list[Entry] = []
        # Entry IDs persisted by this layer's most recent flush — used as
        # causal sources for raw transformer output when nothing else is
        # available (FR5: sources must be real stored IDs).
        self._last_stored_ids: tuple[str, ...] = ()

        # Degradation metadata (FR2 — sr2-82)
        self.degradation_category: str | None = degradation_category
        self.priority: int = priority

        self.condition: str | None = condition
        self._run_context_provider = run_context_provider

    # -- wiring ---------------------------------------------------------------

    def wire(
        self,
        bus: EventBus,
        provenance_store: ProvenanceStore,
        tracer: "Tracer | None",
    ) -> None:
        """Wire this layer to the engine's shared bus, provenance store, and tracer.

        If the layer was previously wired to a different bus, this method
        removes the layer's handle_event callback from that bus before
        switching to the new one. This prevents the old bus from delivering
        events to this layer after rewiring.
        """
        # Unsubscribe handle_event from the old bus if it differs from the new one.
        # Bound methods can't be compared with `is` — compare __func__ and __self__
        # to detect that a stored callback is this layer's handle_event method.
        if self._event_bus is not None and self._event_bus is not bus:
            handle_event_func = self.handle_event.__func__  # type: ignore[attr-defined]
            self._event_bus._subs = [
                (name, cb, is_async)
                for name, cb, is_async in self._event_bus._subs
                if not (
                    hasattr(cb, "__func__")
                    and cb.__func__ is handle_event_func
                    and cb.__self__ is self
                )
            ]

        self._event_bus = bus
        self._provenance_store = provenance_store
        self._tracer = tracer

    # -- turn lifecycle -------------------------------------------------------

    def begin_turn(self, turn_seq: int, next_firing_seq_fn: Callable[[], int]) -> None:
        """Reset this layer's turn state for a new turn.

        Replaces the old pattern where the engine directly mutated
        ``layer._turn_seq``, ``layer._next_firing_seq``, and ``layer._pending_events``.

        Args:
            turn_seq: The sequence number for this turn.
            next_firing_seq_fn: Callable that returns the next firing sequence number.
        """
        self._turn_seq = turn_seq
        self._next_firing_seq = next_firing_seq_fn
        self.set_content([])
        self._pending_events = []
        if self.tool_providers:
            self.reset_tools()

    # -- session seeding ------------------------------------------------------

    def seed(self, messages: list[Message]) -> None:
        """Seed conversation history on any Seedable component in this layer.

        Calls seed() on each resolver that exposes the method. Resolvers that
        don't implement seed() (i.e. non-session resolvers) are silently skipped.
        """
        for resolver in self.resolvers:
            if isinstance(resolver, Seedable):
                resolver.seed(messages)

    # -- subscriptions --------------------------------------------------------

    @property
    def subscriptions(self) -> list[EventSubscription]:
        """All event subscriptions from this layer's resolvers, transformers, and tool providers."""
        subs: list[EventSubscription] = []
        for comp in [*self.resolvers, *self.transformers, *self.tool_providers]:
            subs.extend(comp.subscriptions)
        return subs

    # -- done check -----------------------------------------------------------

    def is_done(self) -> bool:
        """True when all components are idle (never fired) or exhausted (hit max_executions)."""
        for comp in [*self.resolvers, *self.transformers, *self.tool_providers]:
            if comp.execution_count > 0 and comp.execution_count < comp.max_executions:
                return False
        return True

    def is_active(self) -> bool:
        """Return whether this layer participates in the current run context.

        A conditioned layer requires a non-empty string for its condition key.
        The provider is called for each evaluation so later turns can use a
        different run context.
        """
        if self.condition is None:
            return True
        if self._run_context_provider is None:
            return False
        context = self._run_context_provider()
        if context is None:
            return False
        value = context.get(self.condition)
        return isinstance(value, str) and value != ""

    # -- eligibility filter ---------------------------------------------------

    def _eligible(self, components: list, events: list[Event]) -> Any:
        """Generator yielding components that are eligible to fire.

        A component is eligible when:
        - execution_count < max_executions (not exhausted)
        - at least one subscription matches at least one event

        This replaces the repeated guard logic that was duplicated in
        each of the three process_pending loops (resolvers, transformers,
        tool_providers).

        Yields components in the same order they appear in the input list.
        """
        for comp in components:
            if comp.execution_count >= comp.max_executions:
                continue
            if not any(
                s.matches(e)
                for s in comp.subscriptions
                for e in events
            ):
                continue
            yield comp

    # -- event handling (sync callback) ---------------------------------------

    async def _stored_ids(self, candidate_ids: list[str]) -> tuple[str, ...]:
        """Ids from ``candidate_ids`` already present in the provenance store."""
        stored: list[str] = []
        for candidate in candidate_ids:
            try:
                if await self._provenance_store.get(candidate) is not None:
                    stored.append(candidate)
            except Exception:  # pragma: no cover - exotic stores
                continue
        return tuple(stored)

    def handle_event(self, event: Event) -> None:
        """Collect incoming events. Called as a bus callback during drain.

        This is intentionally sync — it just buffers events. The engine
        processes them asynchronously after drain completes.
        """
        self._pending_events.append(event)

    def _snapshot_after(
        self, kind: str, tokens_before: int
    ) -> tuple[list, int, int]:
        """Take the post-firing snapshot using the strategy table.

        Delegates to the strategy's snapshot callable, then computes the
        token delta from the tokens_before value.
        """
        content_after, tokens_after = self._COMPONENT_DISPATCH[kind].snapshot(self)
        return content_after, tokens_after, tokens_after - tokens_before

    def _build_record(
        self,
        *,
        kind: str,
        comp: object,
        events: list[Event],
        content_before: list,
        tokens_before: int,
        content_after: list,
        tokens_after: int,
        tokens_delta: int,
        duration_ms: float,
        iter_seq: int,
        status: str,
        error: str | None = None,
    ) -> FiringRecord:
        return FiringRecord(
            turn_seq=self._turn_seq,
            firing_seq=self._next_firing_seq(),
            kind=kind,
            component_name=comp.name,  # type: ignore[union-attr]
            layer=self.name,
            trigger_events=[e.name for e in events],
            content_before=content_before,
            content_after=content_after,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_delta=tokens_delta,
            duration_ms=duration_ms,
            status=status,
            error=error,
            iteration_seq=iter_seq,
        )

    async def _fire_component(
        self,
        comp: object,
        kind: str,
        events: list[Event],
    ) -> object:
        """Shared try/except/FiringRecord logic for resolvers, transformers, and tool providers.

        Dispatches to the correct component method via the strategy table,
        snapshots content before/after when a tracer is attached, emits a
        FiringRecord on success or failure, and re-raises any exception after
        recording it.

        Returns the raw result from the component call so the caller can apply
        kind-specific post-processing (e.g. transformer execution_count guard,
        result.entries buffering, result.events queuing).

        Signature: _fire_component(comp, kind, events)
          - kind='resolver'      -> calls comp.resolve(events), adds content via add_content()
          - kind='transformer'   -> calls comp.transform(get_content(), events),
                                   applies set_content(result.content) if non-None so the
                                   content_after snapshot reflects the post-transform state;
                                   result.entries / result.events remain the caller's responsibility
          - kind='tool_provider' -> calls comp.provide(events), adds defs via add_tool_definitions()
        """
        strategy = self._COMPONENT_DISPATCH[kind]

        # Snapshot before (tracer guard)
        if self._tracer is not None:
            content_before, tokens_before = strategy.snapshot(self)
            t_start = time.perf_counter()

        try:
            result = await strategy.invoke(comp, events, self)
            strategy.apply(result, self)

            if self._tracer is not None:
                content_after, tokens_after, tokens_delta = self._snapshot_after(kind, tokens_before)
                duration_ms = (time.perf_counter() - t_start) * 1000
                iter_seq = events[0].iteration_seq if events else 1
                self._tracer.on_firing(self._build_record(
                    kind=kind, comp=comp, events=events,
                    content_before=content_before, tokens_before=tokens_before,
                    content_after=content_after, tokens_after=tokens_after,
                    tokens_delta=tokens_delta, duration_ms=duration_ms,
                    iter_seq=iter_seq, status="ok",
                ))

        except Exception as exc:
            if self._tracer is not None:
                content_after, tokens_after, tokens_delta = self._snapshot_after(kind, tokens_before)
                duration_ms = (time.perf_counter() - t_start) * 1000
                iter_seq = events[0].iteration_seq if events else 1
                self._tracer.on_firing(self._build_record(
                    kind=kind, comp=comp, events=events,
                    content_before=content_before, tokens_before=tokens_before,
                    content_after=content_after, tokens_after=tokens_after,
                    tokens_delta=tokens_delta, duration_ms=duration_ms,
                    iter_seq=iter_seq, status="failed", error=str(exc),
                ))
            raise

        return result

    async def process_pending(self) -> bool:
        """Process all pending collected events. Returns True if new events were generated.

        Drives resolvers and transformers against collected events, adds
        content, checks budget, and emits overflow events if needed.
        Also flushes any buffered provenance writes to the store.
        """
        changed = False

        if self._pending_events:
            events = self._pending_events
            self._pending_events = []

            # --- Resolvers ---
            for resolver in self._eligible(self.resolvers, events):
                await self._fire_component(comp=resolver, kind="resolver", events=events)

            # --- Transformers ---
            for transformer in self._eligible(self.transformers, events):
                _count_before = transformer.execution_count
                result = await self._fire_component(comp=transformer, kind="transformer", events=events)
                if transformer.execution_count == _count_before:
                    transformer.execution_count += 1
                # set_content already applied inside _fire_component
                if result.entries:
                    self._pending_writes.extend(result.entries)
                if result.content:
                    # Raw transformer output is compiled content too — it
                    # gets a stored entry like any other item (FR1). Its
                    # sources are declared real entry IDs when given, and
                    # otherwise the entries this run already buffered and
                    # will flush in this same call — never generated
                    # dangling IDs (FR5).
                    raw_sources: tuple[str, ...] = ()
                    for decl in result.provenance or []:
                        if isinstance(decl, dict) and decl.get("sources"):
                            raw_sources = tuple(decl["sources"])
                            break
                    if not raw_sources:
                        raw_sources = await self._stored_ids(
                            [e.id for e in self._pending_writes]
                        )
                    if not raw_sources:
                        raw_sources = self._last_stored_ids or ()
                    self._pending_writes.extend(
                        _synthesize_entries(
                            items=result.content,
                            declarations=result.provenance,
                            origin_name=result.transformer_name,
                            origin_kind="transformer",
                            source_layer=result.source_layer,
                            default_sources=raw_sources,
                        )
                    )
                # Queue events emitted by transformer
                if result.events:
                    for ev in result.events:
                        self._event_bus.queue(ev)

            # --- Tool Providers ---
            for tp in self._eligible(self.tool_providers, events):
                await self._fire_component(comp=tp, kind="tool_provider", events=events)

            # --- Budget check ---
            self.check_budget()

            # Force-truncate if over budget
            warning = self.force_truncate()
            if warning:
                self._event_bus.queue(
                    Event(
                        name="truncation",
                        phase=EventPhase.COMPLETED,
                        source_layer=self.name,
                        data=warning,
                    )
                )

            # True if any events were queued (overflow, truncation, etc.)
            changed = not self._event_bus.is_empty()

        # Flush pending provenance writes (regardless of whether events fired)
        if self._pending_writes:
            await self._provenance_store.write_batch(self._pending_writes)
            self._last_stored_ids = tuple(e.id for e in self._pending_writes)
            self._pending_writes = []

        return changed

    # -- content management --------------------------------------------------

    def get_content(self) -> list[ContentBlock | Message]:
        return list(self._content)

    def add_content(self, resolved: ResolvedContent) -> None:
        if resolved.entries:
            # Entries path — place their content and buffer for the store.
            entry_contents = [e.content for e in resolved.entries]
            self._content = self._position.place(self._content, entry_contents)
            self._pending_writes.extend(resolved.entries)
        if resolved.content:
            # Raw content — placed as-is and synthesized into provenance
            # entries (FR1: there is no raw-content/no-store bypass).
            self._content = self._position.place(self._content, resolved.content)
            self._pending_writes.extend(
                _synthesize_entries(
                    items=resolved.content,
                    declarations=resolved.provenance,
                    origin_name=resolved.resolver_name,
                    origin_kind="resolver",
                    source_layer=resolved.source_layer,
                )
            )
        # Queue any events emitted by the resolver onto the bus
        if resolved.events and self._event_bus is not None:
            for ev in resolved.events:
                self._event_bus.queue(ev)

    def set_content(self, content: list[ContentBlock | Message]) -> None:
        self._content = list(content)
        if not content:
            self._force_truncated = False

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return the current tool definitions accumulated in this layer."""
        return list(self._tool_definitions)

    def reset_tools(self) -> None:
        self._tool_definitions = []

    def add_tool_definitions(self, defs: list[ToolDefinition]) -> None:
        self._tool_definitions.extend(defs)

    # -- budget --------------------------------------------------------------

    def check_budget(self) -> None:
        if self.token_budget is None:
            return
        used = self._token_counter.count(self._content)
        if self.token_threshold_pct is not None and used >= self.token_budget * self.token_threshold_pct:
            self._event_bus.queue(
                Event(
                    name="token_threshold",
                    phase=EventPhase.COMPLETED,
                    source_layer=self.name,
                    data={"used": used, "budget": self.token_budget, "pct": self.token_threshold_pct},
                )
            )
        if used > self.token_budget:
            self._event_bus.queue(
                Event(
                    name="overflow",
                    phase=EventPhase.COMPLETED,
                    source_layer=self.name,
                )
            )

    def force_truncate(self) -> str | None:
        if self.token_budget is None:
            return None

        used = self._token_counter.count(self._content)
        if used <= self.token_budget:
            return None

        self._force_truncated = True
        removed = 0
        while self._content and self._token_counter.count(self._content) > self.token_budget:
            self._content.pop(0)
            removed += 1

        return (
            f"Layer '{self.name}': force-truncated {removed} block(s) "
            f"to fit budget of {self.token_budget} tokens"
        )

    # -- compile --------------------------------------------------------------

    def compile(self) -> list[TextBlock] | list[Message] | list[ToolDefinition]:
        """Compile this layer's content using the TargetCompiler registry.

        Delegates to the registry so Layer.compile() and Engine._compile_request()
        share the same dispatch logic. The registry returns typed output for the
        layer's compilation target.
        """
        compilation_targets = get_compilation_targets()
        compiler = compilation_targets[self.target]

        # Use a dedicated accumulator for the target type; pass throwaway lists
        # for the other two accumulators (compiler only mutates the one matching
        # its CompilationTarget).
        system_out: list[TextBlock] = []
        messages_out: list[Message] = []
        tools_out: list[ToolDefinition] = []

        compiler.collect(
            self._content,
            self._tool_definitions,
            system_out,
            messages_out,
            tools_out,
        )

        if self.target == CompilationTarget.SYSTEM:
            return system_out
        if self.target == CompilationTarget.MESSAGES:
            return messages_out
        return tools_out

    @property
    def blocks(self) -> list[ContentBlock | Message]:
        """Expose accumulated content blocks for engine to read."""
        return list(self._content)

    @property
    def token_count(self) -> int:
        """Current token count of this layer's content.

        Used by the priority shedding subsystem (FR6, sr2-86) to determine
        which layers to retain under budget pressure. Combined with
        ``priority`` to satisfy the ``HasPriorityAndTokens`` protocol.
        """
        return self._token_counter.count(self._content)
