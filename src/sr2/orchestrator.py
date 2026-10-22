"""SR2 — top-level orchestrator.

Wires PipelineConfig → PipelineEngine, drives the turn loop, streams LLM
responses, and emits the assistant_response event on the shared bus.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, TYPE_CHECKING

from ulid import ULID

from sr2.config.models import ConfigError, FinalAnswerConfig, LayerConfig, PipelineConfig, RepeatGuardConfig, ResolverConfig, ToolLoopLimitError, ToolProviderConfig, TransformerConfig
from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from sr2.pipeline.provenance import (
    TURN_ARCHIVE_LAYER,
    ContextAuthority,
    ContextLifecycle,
    ContextReason,
    ContextScope,
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
    ProvenanceStore,
)

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer
    from sr2.memory.protocol import MemoryExtractor, MemoryStore
from sr2.models import Message, TextBlock, ThinkingBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.pipeline.compilation import AppendStrategy, PrefixStrategy
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.protocols import Resolver, TokenCounter, ToolSource, Transformer
from sr2.plugins.registry import PluginRegistry
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMCallable,
    StreamEvent,
)

logger = logging.getLogger(__name__)

# FR1: Type alias for the tool executor callable.
ToolExecutor = Callable[[ToolUseBlock], Awaitable[ToolResultBlock]]


def _malformed_tool_call_message(tool_name: str, meta: Mapping[str, Any]) -> str:
  """Build the tool result a model receives when its tool-call arguments fail to parse.

  obsidian-1fmv: keeps the turn alive. The message distinguishes a hard
  stop (invalid JSON) from a soft one (the call was cut off by the output
  token limit, i.e. finish_reason "length"), so the model knows whether it
  should retry with valid arguments or a shorter payload.
  """
  error = meta.get("error") or "unparseable arguments"
  if meta.get("truncated"):
    return (
      f"Tool call '{tool_name}' was truncated by the output length limit "
      f"before its arguments were complete (arguments ended mid-JSON: {error}). "
      "The call was not executed. Retry with a shorter or split-up argument payload."
    )
  return (
    f"Tool call '{tool_name}' was not executed: its arguments are invalid JSON "
    f"(parse error: {error}). Retry with valid JSON arguments."
  )

# sr2-37: Placeholder text used when the LLM returns an empty response twice
# in a row (original + one retry). Stored in history like normal final text.
EMPTY_RESPONSE_PLACEHOLDER = "[empty model response]"


# sr2-121: How the forced final answer classifies the reason it stopped a run.
# The repeat guard (sr2-122) passes "repeat" with its own reason text.
FinalAnswerReason = Literal["limit", "repeat"]


def _limit_reason_text(max_tool_iterations: int) -> str:
    """sr2-121: the ``{reason}`` text for a run stopped at the tool-call limit.

    Spec spc-959 § Interfaces: ``you reached the tool-call limit of
    {max_tool_iterations} iterations``.
    """
    return f"you reached the tool-call limit of {max_tool_iterations} iterations"


def _repeat_reason_text(first_call: int, identical_calls: int) -> str:
    """sr2-122: the ``{reason}`` text for a run stopped by the repeat guard.

    Spec spc-959 § Interfaces: ``you repeated call {first_call}
    {identical_calls} times``.
    """
    return f"you repeated call {first_call} {identical_calls} times"


@dataclass(frozen=True)
class _GuardStop:
    """sr2-122: the repeat guard ended the run inside one tool iteration.

    Carries the trigger back to ``turn()``: the stop note already replaced
    the triggering call's result, and the forced final answer (reason
    ``repeat``) is built from the iteration's already-built assistant content
    and results (sr2-123), not from its tool_use stream events.
    """

    reason_text: str
    meta: dict


@dataclass(frozen=True)
class ToolIterationResult:
    """Return value of SR2._run_tool_iteration().

    Captures all outputs from one LLM stream + tool-execution cycle so that
    the ``turn()`` async generator can decide what to yield without being
    coupled to the iteration internals.

    When ``tool_use_blocks`` is empty, the iteration is final (no more tools
    to execute) and ``tool_result_blocks`` will also be empty.
    """

    text: str
    """Concatenated text from LLM stream events in this iteration."""

    had_tool_events: bool = False
    """obsidian-1fmv: the LLM emitted at least one tool_use stream event in this
    iteration. Distinct from ``tool_use_blocks`` — malformed tool calls are
    excluded from ``tool_use_blocks`` (they are not executed) yet still make
    the iteration a tool-use iteration that must feed back a tool result."""

    tool_use_blocks: list[ToolUseBlock] = field(default_factory=list)
    """Tool-use blocks extracted from the LLM response that are valid enough to
    execute (empty → nothing to run this iteration)."""

    tool_result_blocks: list[ToolResultBlock] = field(default_factory=list)
    """Results from executing the tool-use blocks (empty if final or no executor)."""

    assistant_content: list = field(default_factory=list)
    """Full assistant content (text + tool_use blocks) for this iteration."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    """TokenUsage from this iteration's LLM call."""

    raw_events: list[StreamEvent] = field(default_factory=list)
    """Raw StreamEvents from the LLM for re-yielding to the caller."""

    forced_final_answer: bool = False
    """sr2-121: True when this iteration is the forced final answer — the tool
    calls of the current iteration were deliberately left unexecuted (each got
    ``final_answer.not_run`` as its result) and one tool-less LLM call has
    already been made to produce ``text``. ``turn()`` finalizes immediately
    instead of rebuilding a tool-carrying request."""

    reason: str = ""
    """sr2-121: why the run was forced — ``"limit"`` or ``"repeat"``. Only
    meaningful when ``forced_final_answer`` is True; carried on the
    ``final_answer`` stream event's meta (spec spc-959 FR 6)."""

    fallback_used: bool = False
    """sr2-121: True when the forced reply was empty, whitespace-only or
    contained a tool call and ``final_answer.fallback`` replaced it."""

    repeat_guard_event: "dict | None" = None
    """sr2-122: the repeat_guard meta (spec spc-959 FR 17) of the guard action
    this iteration produced, if any — ``action`` ``"note"`` or ``"stop"``.
    At most one call per turn triggers an action: the stop ends the run."""

    repeat_guard_stop: "_GuardStop | None" = None
    """sr2-122: set when the repeat guard stopped the run in this iteration.
    ``turn()`` runs the forced final answer with reason ``repeat`` from it
    (spec spc-959 FR 11), regardless of ``final_answer.enabled``."""


# Plugin registries (entry-point based, lazy discovery).
# object is used as the protocol to skip isinstance class-level validation —
# the Resolver/Transformer protocols define instance attributes, so checking
# the class itself yields false negatives. Correctness is enforced at build time.
_RESOLVERS: PluginRegistry = PluginRegistry("sr2.resolvers")
_TRANSFORMERS: PluginRegistry = PluginRegistry("sr2.transformers")
_TOOL_PROVIDERS: PluginRegistry = PluginRegistry("sr2.tool_providers")


def _build_resolver(config: ResolverConfig, deps: Dependencies) -> Any:
    return _RESOLVERS.get(config.type).build(config, deps)


def _build_transformer(config: TransformerConfig, deps: Dependencies) -> Any:
    return _TRANSFORMERS.get(config.type).build(config, deps)


def _build_tool_provider(config: ToolProviderConfig, deps: Dependencies) -> Any:
    return _TOOL_PROVIDERS.get(config.type).build(config, deps)


def reset_discovery() -> None:
    """Reset all plugin registry discovery state.

    Clears the cached entry-point results so the next call to ``get()`` or
    ``names()`` on any registry re-scans entry points.

    Primarily intended as a test hook to isolate tests that rely on
    entry-point discovery from one another or from the live environment.
    """
    _RESOLVERS.reset_discovery()
    _TRANSFORMERS.reset_discovery()
    _TOOL_PROVIDERS.reset_discovery()


def _build_layer(
    layer_config: LayerConfig,
    token_counter: TokenCounter,
    deps: Dependencies,
    bus: EventBus | None = None,
    provenance_store: "ProvenanceStore | None" = None,
) -> Layer:
    target = CompilationTarget(layer_config.target)

    position_str = layer_config.position or "append"
    position = PrefixStrategy() if position_str == "prefix" else AppendStrategy()

    resolvers = [_build_resolver(r, deps) for r in layer_config.resolvers]
    transformers = [_build_transformer(t, deps) for t in (layer_config.transformers or [])]
    tool_providers = [_build_tool_provider(tp, deps) for tp in (layer_config.tool_providers or [])]

    event_bus = bus if bus is not None else EventBus()

    return Layer(
        name=layer_config.name,
        target=target,
        position=position,
        token_budget=layer_config.token_budget,
        token_threshold_pct=layer_config.token_threshold_pct,
        resolvers=resolvers,
        transformers=transformers,
        tool_providers=tool_providers,
        token_counter=token_counter,
        event_bus=event_bus,
        provenance_store=provenance_store,
        degradation_category=layer_config.degradation_category,
        priority=layer_config.priority,
        condition=layer_config.condition,
        run_context_provider=deps.run_context_provider,
    )


class _RepeatGuard:
    """sr2-122: per-turn state of the consecutive repeat guard (spec spc-959 FR 7-17).

    Numbers the executed tool calls of one turn from 1 in assistant order
    (FR 7; malformed calls never reach ``observe``), compares each executed
    call to the previous executed call, and decides note / warning / stop.
    An instance is created per ``turn()`` call (FR 16).

    Comparison is exact: same tool name and same arguments serialized as
    JSON with sorted keys (FR 8). The same-result test is same content and
    same ``is_error`` against the previous call's *real* result, never the
    note that replaced it (FR 10, 15).
    """

    def __init__(self, config: RepeatGuardConfig) -> None:
        self._config = config
        self.call_count = 0
        """Executed calls observed so far this turn (malformed calls excluded)."""

        self._prev_key: tuple[str, str] | None = None
        """(tool name, sorted-key JSON arguments) of the previous executed call."""

        self._prev_result: tuple[Any, bool] | None = None
        """Real (content, is_error) of the previous executed call (FR 15)."""

        self.first_call = 0
        """Number of the first call of the current same-result chain."""

        self.run_start = 0
        """Number of the first call of the current identical-call run (FR 13)."""

        self.identical_calls = 0
        """Consecutive identical calls including the first of the run (FR 13)."""

        self.same_result_repeats = 0
        """Repeats within the run whose result matched the previous call's."""

    def observe(self, block: ToolUseBlock, result: ToolResultBlock) -> dict | None:
        """Number one executed call and classify it.

        Returns the ``repeat_guard`` meta for an emitted action (spec
        spc-959 FR 17) with ``action`` ``"note"`` or ``"stop"``, and mutates
        ``result.content`` to the note (plus warning) in that case. Returns
        ``None`` when the real result passes through untouched.

        Two counts run over the executed calls (case table, confirmed
        2026-10-19): the identical-call run (``run_start`` /
        ``identical_calls``) breaks only on a non-identical call and counts
        whatever the results (FR 13, AC5-r2); the same-result chain
        (``first_call`` / ``same_result_repeats``) breaks when a repeat's
        result differs, so a note always names the first call that returned
        that result (AC5-r1).
        """
        self.call_count += 1
        key = (block.name, json.dumps(block.input, sort_keys=True))
        identical = key == self._prev_key
        same_result = identical and (result.content, result.is_error) == self._prev_result
        # FR 15: keep the REAL result for the next comparison, before the
        # content below is possibly replaced by the note.
        self._prev_key = key
        self._prev_result = (result.content, result.is_error)

        if not identical:
            # FR 14: not identical to the previous executed call — reset both
            # counts; this call starts both the run and the same-result chain.
            self.run_start = self.call_count
            self.identical_calls = 1
            self.first_call = self.call_count
            self.same_result_repeats = 0
            return None

        self.identical_calls += 1
        if same_result:
            self.same_result_repeats += 1
        else:
            # FR 12: identical call, different result — the real result
            # passes; the same-result chain restarts at this call while the
            # identical-call run continues.
            self.same_result_repeats = 0
            self.first_call = self.call_count

        stop = self.same_result_repeats > self._config.notes_before_stop
        stop = stop or self.identical_calls >= self._config.max_identical_calls

        # FR 10: the note names the first call of the same-result chain; the
        # warning belongs to the repeat whose count equals notes_before_stop.
        note = self._config.note.format(
            first_call=self.first_call, repeat=self.same_result_repeats
        )
        if same_result and self.same_result_repeats == self._config.notes_before_stop:
            note += self._config.warning.format(repeat=self.same_result_repeats)

        if not stop:
            if not same_result:
                return None
            result.content = note
            return self._meta("note")

        # FR 11 / 13: the stop. The triggering call's result is the note —
        # except a max_identical_calls stop whose result differs from the
        # previous call's, which passes its real result (FR 13 as amended
        # 2026-10-20, sr2-123).
        if not (stop and not same_result):
            result.content = note
        return self._meta("stop")

    def _meta(self, action: str) -> dict:
        """The repeat_guard event meta (spec spc-959 FR 17)."""
        return {
            "call": self.call_count,
            "first_call": self.first_call,
            "identical_calls": self.identical_calls,
            "same_result_repeats": self.same_result_repeats,
            "action": action,
        }


class _TurnArchivist:
    """sr2-93 (spec FR3/AC2, FR5): appends the SR2-visible turn sequence to
    the provenance store as it happens.

    One Entry per payload item, written in SR2-visible occurrence order:
    user input → ``Message(role="user")``; each assistant iteration with
    SR2-visible text and the final assistant output → ``Message(role=
    "assistant")``; each ToolUseBlock / ToolResultBlock → the block itself,
    verbatim (upstream truncation markers included — SR2 archives the
    payload it received, never claimed raw data). Control events (retry,
    finish_reason, iteration_complete, repeat_guard telemetry) are not
    payloads and are never archived (spec R1).

    Causal shape (amendment-fixed): ``layer == "turn_archive"``, session
    id, ``meta["turn_seq"]`` (engine turn) and monotonic per-turn
    ``meta["seq"]`` (append order, the durable ordering key), lifecycle
    ARCHIVE, non-empty ``scope.owner_id``, origin ``("resolver",
    "turn_archivist")`` genesis for the turn's first entry and
    ``("transformer", "turn_archivist")`` afterwards. Each entry's sources
    are the previously archived entry id for this session — real stored ids
    only, written in order so every referenced id exists when it is
    referenced (FR5; lineage from any archived entry walks back to the
    user input).

    Writes are unconditional (FR3): they never alter compiled context, so
    continuity-disabled behavior is unchanged. A failed store write is
    logged and swallowed — losing the archive must never crash a turn.
    """

    def __init__(self, store: ProvenanceStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id
        self.turn_seq = -1
        self._seq = 0
        self._prev_id: str | None = None
        """Id of the most recently archived entry for this session, or None
        before the first write (the genesis entry must then be
        resolver-origin with empty sources)."""

    async def archive(
        self,
        content,
        reason: ContextReason,
        authority: ContextAuthority,
    ) -> None:
        """Append one SR2-visible payload item, stamped with turn/seq order."""
        genesis = self._prev_id is None
        entry = Entry(
            id=str(ULID()),
            content=content,
            sources=() if genesis else (self._prev_id,),
            origin=EntryOrigin(
                kind="resolver" if genesis else "transformer",
                name="turn_archivist",
            ),
            layer=TURN_ARCHIVE_LAYER,
            session_id=self._session_id,
            created_at=datetime.now(tz=timezone.utc),
            reason=reason,
            authority=authority,
            lifecycle=ContextLifecycle.ARCHIVE,
            # Non-empty owner: the SR2 session is the archive's owner scope.
            scope=ContextScope(owner_id=self._session_id),
            meta={"turn_seq": self.turn_seq, "seq": self._seq},
        )
        try:
            await self._store.write(entry)
        except Exception:
            logger.exception(
                "turn archive write failed (session=%s turn=%s seq=%s)",
                self._session_id,
                self.turn_seq,
                self._seq,
            )
            return
        self._prev_id = entry.id
        self._seq += 1

    async def archive_text(self, text: str) -> None:
        """Archive an assistant iteration/final text as a Message.

        An iteration with no SR2-visible text has no text payload to record
        and contributes nothing (archive = what SR2 saw).
        """
        if not text:
            return
        await self.archive(
            Message(role="assistant", content=[TextBlock(text=text)]),
            ContextReason.ASSISTANT_OUTPUT,
            ContextAuthority.AGENT_TENTATIVE,
        )


class SR2:
    """Top-level orchestrator: turns a PipelineConfig into a streaming turn loop."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        llm: "LLMCallable | dict[str, LLMCallable]",
        token_counter: TokenCounter,
        session_id: str | None = None,
        provenance_store: ProvenanceStore | None = None,
        tracer: "Tracer | None" = None,
        memory_store: "MemoryStore | None" = None,
        memory_extractor: "MemoryExtractor | None" = None,
        tool_executor: "ToolExecutor | None" = None,
        active_frame_provider: Callable[[str], str | None] | None = None,
        run_context_provider: Callable[[], dict[str, str] | None] | None = None,
        tool_source: "ToolSource | None" = None,
    ) -> None:
        # Normalise: bare LLMCallable → single-entry dict under "default".
        # Dict form: no "default" key requirement — callers may use any key names.
        if isinstance(llm, dict):
            llm_dict: dict[str, LLMCallable] = llm
        else:
            llm_dict = {"default": llm}  # type: ignore[assignment]

        if not llm_dict:
            raise ValueError("llm must not be empty")

        # Driver: explicit "default" key takes precedence; otherwise use first value.
        self._llm = llm_dict.get("default") or next(iter(llm_dict.values()))
        self._token_counter = token_counter
        self._tracer = tracer
        self._tool_executor: ToolExecutor | None = tool_executor
        self._max_tool_iterations = pipeline_config.max_tool_iterations
        self._max_parallel_tools = pipeline_config.max_parallel_tools
        self._replay_reasoning = pipeline_config.replay_reasoning
        # sr2-121: forced final answer at the tool-iteration limit.
        self._final_answer: FinalAnswerConfig = pipeline_config.final_answer
        # sr2-122: consecutive repeat guard (state itself is per turn, FR 16).
        self._repeat_guard: RepeatGuardConfig = pipeline_config.repeat_guard
        self.session_id = session_id if session_id is not None else str(ULID())
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=pipeline_config.circuit_breaker_failure_threshold,
            recovery_timeout=pipeline_config.circuit_breaker_recovery_timeout,
        )
        self._llm_timeout_seconds = pipeline_config.llm_timeout_seconds
        self._pp_task: asyncio.Task[None] | None = None

        deps = Dependencies(
            llm=llm_dict,
            memory_store=memory_store,
            memory_extractor=memory_extractor,
            session_id=self.session_id,
            active_frame_provider=active_frame_provider,
            run_context_provider=run_context_provider,
            tool_source=tool_source,
        )
        self._active_frame_provider = deps.active_frame_provider

        # Resolve shared infrastructure before building layers so layers receive
        # the real bus and provenance store — not throwaway placeholders.
        shared_bus = EventBus()
        resolved_provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )

        layers = [
            _build_layer(lc, token_counter, deps, bus=shared_bus, provenance_store=resolved_provenance_store)
            for lc in pipeline_config.layers
        ]

        # sr2-93 (FR3/AC2): append the SR2-visible turn sequence to the same
        # store the layers write to.
        self._turn_archivist = _TurnArchivist(
            resolved_provenance_store, self.session_id
        )

        # FR5: Build the degradation ladder from config (or None if absent)
        ladder = None
        if pipeline_config.degradation is not None:
            from sr2.degradation.ladder import DegradationLadder

            ladder = DegradationLadder.from_config(pipeline_config.degradation)

        self._engine = PipelineEngine(
            layers=layers,
            token_counter=token_counter,
            provenance_store=resolved_provenance_store,
            token_budget=pipeline_config.token_budget,
            tracer=tracer,
            bus=shared_bus,
            ladder=ladder,
        )

    # ------------------------------------------------------------------
    # Block stamping
    # ------------------------------------------------------------------

    def _stamp_block(self, block: Any, origin: str | None = None) -> None:
        """Stamp ``meta["frame"]`` on *block* when an active-frame provider is set.

        The *origin* parameter identifies the transport/source of the current
        turn.  The provider uses it to resolve the correct frame
        (work-frame if open on that origin, else ambient frame).
        """
        if self._active_frame_provider is None:
            return
        frame = self._active_frame_provider(origin or "")
        if frame is not None:
            block.meta["frame"] = frame

    # ------------------------------------------------------------------
    # Public inspection helpers
    # ------------------------------------------------------------------

    @property
    def provenance_store(self) -> "ProvenanceStore":
        """Expose the active provenance store for testing and inspection."""
        return self._engine.provenance_store

    @property
    def bus(self) -> "EventBus":
        """Expose the shared event bus for tool-level event emission."""
        return self._engine.bus

    # ------------------------------------------------------------------
    # Session seeding
    # ------------------------------------------------------------------

    def seed_session(self, messages: list[Message]) -> None:
        """Pre-populate conversation history in all SessionResolver instances.

        Delegates to engine.seed(), which propagates through Layer.seed() to
        each SessionResolver.seed(). Overwrites any existing history.
        No-op if no SessionResolver instances are found.
        """
        self._engine.seed(messages)

    # ------------------------------------------------------------------
    # Tool iteration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_empty_iteration(iteration: ToolIterationResult) -> bool:
        """sr2-37: True when an iteration has no tool calls and no real text.

        "No real text" means empty or whitespace-only — thinking-only output
        produces no text events, so it also counts as empty.
        """
        return (
            not iteration.had_tool_events
            and not iteration.tool_use_blocks
            and not iteration.text.strip()
        )

    async def _execute_tools(
        self,
        tool_use_blocks: list[ToolUseBlock],
        origin: str,
    ) -> list[ToolResultBlock]:
        """Execute all tool-use blocks concurrently and return results.

        Each block is passed to ``self._tool_executor`` and executed via
        ``asyncio.gather`` (preserves input order).  Optional semaphore caps
        concurrency when ``max_parallel_tools`` is set.

        Errors from individual tool calls are wrapped as
        ``ToolResultBlock(is_error=True)``.  ``CancelledError`` propagates
        immediately without wrapping.

        Raises:
            TypeError: If ``tool_executor`` returned a non-awaitable (sync
                       callable passed by mistake).
        """
        # FR4+FR15: execute all tool blocks concurrently via asyncio.gather.
        # Optional semaphore caps concurrency when max_parallel_tools is set.
        sem = (
            asyncio.Semaphore(self._max_parallel_tools)
            if self._max_parallel_tools is not None
            else None
        )

        async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
            coro = self._tool_executor(block)  # type: ignore[misc]
            # If the caller passed a sync callable by mistake, the returned
            # value is not awaitable — that is a configuration error, not a
            # tool error, so we propagate TypeError immediately (not wrapped).
            if not inspect.isawaitable(coro):
                raise TypeError(
                    f"tool_executor must be an async callable (coroutine function); "
                    f"got non-awaitable result {type(coro)!r}"
                )
            try:
                if sem is not None:
                    async with sem:
                        return await coro
                else:
                    return await coro
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return ToolResultBlock(
                    tool_use_id=block.id,
                    content=str(exc),
                    is_error=True,
                )

        # asyncio.gather preserves input order: results[i] maps to tool_use_blocks[i].
        tool_result_blocks: list[ToolResultBlock] = list(
            await asyncio.gather(*(_run_one(b) for b in tool_use_blocks))
        )
        for result_block in tool_result_blocks:
            self._stamp_block(result_block, origin)
        return tool_result_blocks

    def _tool_call_blocks(self, iter_tool_use: list[StreamEvent], origin: str) -> list[ToolUseBlock]:
        """sr2-121: build the ToolUseBlocks of one iteration's tool_use events.

        Carried meta (e.g. the LLM integration's ``invalid_arguments`` flag) is
        preserved, and each block is frame-stamped like every other block the
        orchestrator produces. Split out of ``_run_tool_iteration`` so the
        forced-final-answer path can put the same blocks in the assistant
        message without executing them.
        """
        blocks: list[ToolUseBlock] = []
        for tu_event in iter_tool_use:
            tu_meta = tu_event.meta or {}
            tool_block = ToolUseBlock(
                id=tu_event.tool_use_id,
                name=tu_event.tool_name,
                input=tu_event.tool_input,
            )
            if tu_meta:
                tool_block.meta.update(tu_meta)
            self._stamp_block(tool_block, origin)
            blocks.append(tool_block)
        return blocks

    async def _forced_final_answer(
        self,
        request: CompletionRequest,
        origin: str,
        iteration_seq: int,
        *,
        iter_text: str,
        iter_thinking: list[str],
        iter_tool_use: list[StreamEvent],
        reason: FinalAnswerReason,
        reason_text: str,
        executed: tuple[list, list] | None = None,
    ) -> ToolIterationResult:
        """sr2-121: end a run with one tool-less LLM call (spec spc-959 FR 2-4).

        Builds the current iteration's assistant content (replay thinking and
        text, then the ToolUseBlocks) and its ``final_answer.not_run`` results
        — the synthesized-result precedent is the malformed-call path in
        ``_run_tool_iteration`` — queues the pair on the engine bus so the
        session resolver captures them, appends a user message holding
        ``final_answer.instruction`` with ``{reason}`` filled, and makes
        exactly one LLM call with ``tools=None`` (the litellm integration
        omits the ``tools`` field when it is None).

        ``executed`` (sr2-123, spec spc-959 AC9) is the stop iteration's
        ``(assistant_content, tool_result_blocks)`` when its calls already
        ran — the repeat-guard stop. The forced request then reuses those
        blocks instead of the not_run results and skips the bus pair, which
        ``turn()`` already queued for the executed iteration.

        Non-whitespace text with no tool call in that reply becomes the turn's
        final text; an empty, whitespace-only or tool-calling reply yields
        ``final_answer.fallback`` with ``{reason}`` filled. No retry.

        ``reason``/``reason_text`` are parameters so the repeat guard
        (sr2-122) can reuse this with reason ``"repeat"``.

        The forced call passes ``iteration_seq = -1``: it is not a tool-loop
        iteration, so it can never re-enter the limit path (a tool call in
        the reply is answered with the fallback, not with another forced call).
        """
        # Assistant content for the ending iteration, built exactly like the
        # normal path: replayed reasoning first, then text, then calls.
        if executed is not None:
            # sr2-123 (AC9): the calls already ran — the guard's note or the
            # real result is their content; nothing is rebuilt or not_run.
            assistant_content = list(executed[0])
            tool_blocks = [b for b in assistant_content if isinstance(b, ToolUseBlock)]
        else:
            assistant_content = []
            if self._replay_reasoning and iter_thinking:
                thinking_block = ThinkingBlock(text="".join(iter_thinking))
                self._stamp_block(thinking_block, origin)
                assistant_content.append(thinking_block)
            if iter_text:
                text_block = TextBlock(text=iter_text)
                self._stamp_block(text_block, origin)
                assistant_content.append(text_block)
            tool_blocks = self._tool_call_blocks(iter_tool_use, origin)
            assistant_content.extend(tool_blocks)

        # Every unexecuted call gets final_answer.not_run as its result so the
        # tool_use blocks stay paired on the wire. An executed iteration keeps
        # its own results (sr2-123).
        if executed is not None:
            result_blocks: list[ToolResultBlock] = list(executed[1])
        else:
            result_blocks = []
            for block in tool_blocks:
                result_block = ToolResultBlock(
                    tool_use_id=block.id,
                    content=self._final_answer.not_run,
                )
                self._stamp_block(result_block, origin)
                result_blocks.append(result_block)

        # Session-history capture for this iteration, mirroring the tool-use
        # branch of turn(): the pair goes on the bus so the session resolver
        # records the assistant message and its tool results. An executed
        # iteration's pair is already on the bus (turn() queued it before the
        # guard stop) — queuing it again would duplicate the history (AC9).
        if executed is None:
            self._engine.bus.queue(
                Event(
                    name="tool_use_emitted",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=list(tool_blocks),
                    iteration_seq=iteration_seq,
                )
            )
            self._engine.bus.queue(
                Event(
                    name="tool_result_received",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=list(result_blocks),
                    iteration_seq=iteration_seq,
                )
            )

        instruction_block = TextBlock(
            text=self._final_answer.instruction.format(reason=reason_text)
        )
        self._stamp_block(instruction_block, origin)
        forced_request = CompletionRequest(
            system=request.system,
            messages=list(request.messages)
            + [
                Message(role="assistant", content=assistant_content),
                Message(
                    role="user",
                    content=list(result_blocks),  # type: ignore[arg-type]
                ),
                Message(role="user", content=[instruction_block]),
            ],
            tools=None,
        )

        # forced=True: the reply is judged here (fallback on empty or on a
        # tool call, spec spc-959 FR 4), so the iteration must never execute
        # tools or re-enter the limit path itself.
        forced_iteration = await self._run_tool_iteration(
            forced_request, -1, origin, forced=True
        )

        had_tool_call = bool(forced_iteration.tool_use_blocks)
        text = forced_iteration.text
        fallback_used = had_tool_call or not text.strip()
        final_text = (
            self._final_answer.fallback.format(reason=reason_text)
            if fallback_used
            else text
        )
        final_block = TextBlock(text=final_text)
        self._stamp_block(final_block, origin)

        return ToolIterationResult(
            text=final_text,
            assistant_content=[final_block],
            usage=forced_iteration.usage,
            raw_events=[StreamEvent(type="text", text=final_text)],
            forced_final_answer=True,
            reason=reason,
            fallback_used=fallback_used,
        )

    async def _run_tool_iteration(
        self,
        request: CompletionRequest,
        iteration_seq: int,
        origin: str,
        *,
        forced: bool = False,
        guard: "_RepeatGuard | None" = None,
    ) -> ToolIterationResult:
        """Run one LLM stream + tool-execution cycle.

        Calls the LLM via ``stream()``, classifies events into text /
        tool_use / usage, and if tool-use events are present, builds
        ``ToolUseBlock`` instances, executes them via ``_execute_tools()``,
        and returns the full result.

        Returns a ``ToolIterationResult`` with empty ``tool_use_blocks``
        when the LLM produced no tool calls (i.e. this is the final
        iteration).

        ``forced=True`` (sr2-121) marks the forced-final-answer call: tool
        calls in the reply are reported (``had_tool_events`` /
        ``tool_use_blocks``) but never executed and never re-checked against
        the iteration limit — ``_forced_final_answer`` turns them into the
        fallback text.

        ``guard`` (sr2-122) is the turn's repeat-guard state, or ``None``
        when the guard is disabled: results are then never altered (FR 18).
        The guard is not passed to the forced call — the forced reply's tool
        calls never execute.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open.
            ToolLoopLimitError: If iteration limit exceeded before execution.
            ConfigError: If tool calls present but ``tool_executor`` is None.
        """
        # Accumulate stream events for this iteration.
        iter_text: list[str] = []
        iter_thinking: list[str] = []
        iter_tool_use: list[StreamEvent] = []
        iter_usage: TokenUsage | None = None

        # Circuit breaker check — reject immediately if open.
        if not self._circuit_breaker.allow_request():
            raise CircuitBreakerOpenError("LLM circuit breaker is open")

        # Collect all stream events, applying optional per-call timeout.
        async def _collect_stream() -> list[StreamEvent]:
            collected: list[StreamEvent] = []
            async for event in self._llm.stream(request):
                collected.append(event)
            return collected

        try:
            if self._llm_timeout_seconds is not None:
                raw_events = await asyncio.wait_for(
                    _collect_stream(), timeout=self._llm_timeout_seconds
                )
            else:
                raw_events = await _collect_stream()
            self._circuit_breaker.record_success()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._circuit_breaker.record_failure()
            raise

        # Classify events from this LLM call.
        for event in raw_events:
            if event.type == "text" and event.text:
                iter_text.append(event.text)
            elif event.type == "thinking" and event.text:
                # sr2-113: collect this iteration's reasoning for replay;
                # never forwarded to the caller (that is sr2-110).
                iter_thinking.append(event.text)
            elif event.type == "tool_use":
                iter_tool_use.append(event)
            elif event.type == "usage" and event.usage is not None:
                iter_usage = event.usage

        full_iter_text = "".join(iter_text)
        usage = iter_usage if iter_usage is not None else TokenUsage()

        # No tool calls — return early with empty tool lists.
        if not iter_tool_use:
            return ToolIterationResult(
                text=full_iter_text,
                usage=usage,
                raw_events=raw_events,
            )

        # There are tool_use blocks — check the iteration limit before executing.
        # sr2-121 (spec spc-959 FR 1-2): with final_answer.enabled the limit no
        # longer raises — the iteration's tool calls are left unexecuted and
        # one tool-less LLM call writes the run's final answer. A forced call's
        # own tool calls are never executed either (FR 4): report and return.
        if forced and iter_tool_use:
            forced_blocks = self._tool_call_blocks(iter_tool_use, origin)
            return ToolIterationResult(
                had_tool_events=True,
                text=full_iter_text,
                tool_use_blocks=forced_blocks,
                usage=usage,
                raw_events=raw_events,
            )
        if iteration_seq >= self._max_tool_iterations:
            if not self._final_answer.enabled:
                raise ToolLoopLimitError(
                    f"Tool loop limit exceeded: {iteration_seq} iterations reached "
                    f"(max_tool_iterations={self._max_tool_iterations}). "
                    f"The LLM kept requesting tools without converging."
                )
            return await self._forced_final_answer(
                request,
                origin,
                iteration_seq,
                iter_text=full_iter_text,
                iter_thinking=iter_thinking,
                iter_tool_use=iter_tool_use,
                reason="limit",
                reason_text=_limit_reason_text(self._max_tool_iterations),
            )

        # Build assistant content: text + tool-use blocks.
        assistant_content: list = []
        if self._replay_reasoning and iter_thinking:
            # sr2-113: replay this iteration's reasoning as a ThinkingBlock,
            # placed first so the next request carries it as
            # reasoning_content on this assistant message.
            thinking_block = ThinkingBlock(text="".join(iter_thinking))
            self._stamp_block(thinking_block, origin)
            assistant_content.append(thinking_block)
        if full_iter_text:
            text_block = TextBlock(text=full_iter_text)
            self._stamp_block(text_block, origin)
            assistant_content.append(text_block)

        if self._tool_executor is None:
            raise ConfigError(
                "tool_executor not configured — set tool_executor= on SR2 to handle tool calls"
            )

        # Build all ToolUseBlocks in order (preserves mapping for result ordering).
        # obsidian-1fmv: tool calls whose arguments failed to parse are flagged by
        # the LLM integration via ``meta["invalid_arguments"]``. They are kept in
        # the assistant message (so the wire pairs their tool result) but never
        # sent to the executor — the model receives a tool result explaining the
        # failure and can retry.
        tool_use_blocks: list[ToolUseBlock] = []
        tool_result_blocks: list[ToolResultBlock] = []
        for tool_block in self._tool_call_blocks(iter_tool_use, origin):
            assistant_content.append(tool_block)
            tu_meta = tool_block.meta
            if tu_meta.get("invalid_arguments"):
                malformed_result = ToolResultBlock(
                    tool_use_id=tool_block.id,
                    content=_malformed_tool_call_message(
                        tool_name=tool_block.name,
                        meta=tu_meta,
                    ),
                    is_error=True,
                )
                self._stamp_block(malformed_result, origin)
                tool_result_blocks.append(malformed_result)
            else:
                tool_use_blocks.append(tool_block)

        # Execute (valid) tools concurrently.
        if tool_use_blocks:
            tool_result_blocks.extend(await self._execute_tools(tool_use_blocks, origin))

        # sr2-122 (spec spc-959 FR 7-17): the repeat guard walks the executed
        # calls in assistant order after execution (FR 9 — including calls of
        # the same message). Malformed calls are not in tool_use_blocks, so
        # they are never numbered and never reset the guard (FR 7). Results
        # are mutated in place: the note replaces the real content the model
        # sees, while the guard keeps the real result for the next comparison.
        guard_event: dict | None = None
        if guard is not None:
            # tool_result_blocks holds the malformed results first, then one
            # result per executed block in tool_use_blocks order: the zip
            # pairs each executed block with its own result.
            for block, result_block in zip(tool_use_blocks, tool_result_blocks):
                meta = guard.observe(block, result_block)
                if meta is not None:
                    guard_event = meta

        repeat_guard_event = guard_event
        repeat_guard_stop: _GuardStop | None = None
        if guard_event is not None and guard_event["action"] == "stop" and guard is not None:
            # FR 11: the guard stopped the run. turn() runs the forced final
            # answer with reason "repeat" from this trigger.
            repeat_guard_stop = _GuardStop(
                reason_text=_repeat_reason_text(
                    guard.run_start, guard.identical_calls
                ),
                meta=guard_event,
            )

        return ToolIterationResult(
            had_tool_events=True,
            text=full_iter_text,
            tool_use_blocks=tool_use_blocks,
            tool_result_blocks=tool_result_blocks,
            assistant_content=assistant_content,
            usage=usage,
            raw_events=raw_events,
            repeat_guard_event=repeat_guard_event,
            repeat_guard_stop=repeat_guard_stop,
        )

    # ------------------------------------------------------------------
    # Core turn loop
    # ------------------------------------------------------------------

    async def turn(self, user_input: list, *, origin: str = "") -> AsyncIterator[StreamEvent]:
        """Async generator: runs the pipeline and streams LLM events.

        Implements the multi-iteration tool loop (FR3):
          start_turn → stream LLM → detect tool_use → execute tools →
          continue_turn → repeat until no tool_use → end_turn

        FR11: post_process is awaited once after the full loop (not per iteration).
        FR14: exactly one StreamEvent(type="end") is emitted to the caller.
        FR7:  raises ToolLoopLimitError if iterations exceed max_tool_iterations.

        sr2-80: Finalization (assistant_response + end_turn + post_process) runs
        BEFORE yielding any events in the final iteration, so an early break
        by the consumer does not silently skip session-history capture or
        memory extraction.
        """
        # FR4+FR5+FR6: Await any pending deferred post_process task from the
        # previous turn before starting a new one. This ensures turn N's
        # post_process completes before turn N+1 begins.
        # FR8: Surface deferred-task errors as StreamEvent(type="error") early
        # in the stream. CancelledError propagates unchanged.
        if self._pp_task is not None:
            try:
                await self._pp_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield StreamEvent(type="error", errors=[f"deferred post_process error: {exc}"])
            self._pp_task = None

        # Reset all pipeline components so turn 2+ re-fires them.
        self._engine.reset_execution_counts()

        # FR7: Accumulate bus errors across in-band drains.
        in_band_errors: list[str] = []

        # Start the turn via the new engine API.
        next_seq = self._engine.turn_seq + 1
        await self._engine.start_turn(turn_seq=next_seq)
        # sr2-93 (FR3): per-turn archive ordering follows the engine turn.
        self._turn_archivist.turn_seq = next_seq
        in_band_errors.extend(self._engine.bus.get_errors())

        # Inject user_input as an event.
        if user_input:
            self._engine.bus.queue(
                Event(
                    name="user_input",
                    phase=EventPhase.COMPLETED,
                    source_layer="engine",
                    data=user_input,
                )
            )
            # sr2-93 (FR3/AC2): archive the user input payload — one
            # Message(role="user") per event, the turn's genesis entry —
            # before anything the model produces.
            await self._turn_archivist.archive(
                Message(role="user", content=list(user_input)),
                ContextReason.USER_INPUT,
                ContextAuthority.USER,
            )
            await self._engine.run_loop()
            in_band_errors.extend(self._engine.bus.get_errors())

        # Compile the initial request after user_input is settled.
        compiled = self._engine.compile()

        # ----------------------------------------------------------------
        # Multi-iteration tool loop
        # ----------------------------------------------------------------
        iteration_seq = 0

        # Build the mutable request we'll feed to the LLM each iteration.
        # Between iterations we'll add assistant + tool_result messages directly.
        current_request = compiled

        # sr2-37: Empty-response retry budget — at most ONE retry per turn.
        empty_retry_used = False

        # sr2-122 (spec spc-959 FR 16): the repeat guard's state is per turn;
        # None when the guard is disabled — results are then never altered (FR 18).
        guard = (
            _RepeatGuard(self._repeat_guard)
            if self._repeat_guard.enabled
            else None
        )

        while True:
            iteration = await self._run_tool_iteration(
                current_request, iteration_seq, origin, guard=guard
            )

            # ---------------------------------------------------------------
            # sr2-37: Empty-response guard
            # ---------------------------------------------------------------
            # An "empty iteration" has no tool calls AND no non-whitespace
            # text (thinking-only output counts as empty). Local models
            # occasionally produce these mid-loop; without the guard the turn
            # would end silently with no output.
            if self._is_empty_iteration(iteration):
                if not empty_retry_used:
                    # Retry the SAME request exactly once. The retry does not
                    # consume a tool-loop iteration: iteration_seq is reused,
                    # so a tool call from the retry belongs to this iteration.
                    empty_retry_used = True
                    logger.warning("empty LLM response — retrying the request once")
                    yield StreamEvent(
                        type="retry",
                        iteration=iteration_seq,
                        meta={"reason": "empty_response", "attempt": 1},
                    )
                    iteration = await self._run_tool_iteration(
                        current_request, iteration_seq, origin
                    )

                if self._is_empty_iteration(iteration):
                    # Retry budget spent and still empty — finalize with the
                    # placeholder via the normal final-response path.
                    logger.warning(
                        "empty LLM response after retry — finalizing turn with placeholder"
                    )
                    placeholder_block = TextBlock(text=EMPTY_RESPONSE_PLACEHOLDER)
                    self._stamp_block(placeholder_block, origin)
                    # sr2-93 (AC2): the placeholder is the turn's final
                    # SR2-visible assistant output.
                    await self._turn_archivist.archive_text(EMPTY_RESPONSE_PLACEHOLDER)
                    final_response = CompletionResponse(
                        id="turn-response",
                        content=[placeholder_block],
                        stop_reason="end_turn",
                        usage=iteration.usage,
                    )

                    # sr2-80: Finalize BEFORE yielding any events so an early
                    # consumer break does not skip session-history capture.
                    await self._finalize_turn(final_response)

                    yield StreamEvent(type="text", text=EMPTY_RESPONSE_PLACEHOLDER)

                    # FR7: Surface accumulated in-band bus errors before the final end event.
                    if in_band_errors:
                        yield StreamEvent(type="error", errors=in_band_errors)

                    yield StreamEvent(type="end")
                    return

            if iteration.forced_final_answer:
                # ---------------------------------------------------------------
                # sr2-121: forced final answer (spec spc-959 FR 3, 6). The
                # unexecuted tool calls and their not_run results are already in
                # the forced request's message list; the session resolver
                # captured them via the bus events queued below. Finalize like
                # the normal final iteration so session history and
                # post_process run, and emit exactly one final_answer event
                # before the final text.
                # ---------------------------------------------------------------
                content = list(iteration.assistant_content)
                # sr2-93 (FR11, spec R1): archive the forced final answer like
                # any final assistant output; control telemetry is not
                # archived.
                await self._turn_archivist.archive_text(iteration.text)
                final_response = CompletionResponse(
                    id="turn-response",
                    content=content,
                    stop_reason="end_turn",
                    usage=iteration.usage,
                )

                # sr2-80: Finalize BEFORE yielding any events.
                await self._finalize_turn(final_response)

                yield StreamEvent(
                    type="final_answer",
                    iteration=iteration_seq,
                    meta={
                        "reason": iteration.reason,
                        "fallback_used": iteration.fallback_used,
                    },
                )
                for event in iteration.raw_events:
                    if event.type == "text" and event.text:
                        yield event

                # FR7: Surface accumulated in-band bus errors before the final end event.
                if in_band_errors:
                    yield StreamEvent(type="error", errors=in_band_errors)

                yield StreamEvent(type="end")
                return

            if not iteration.had_tool_events:
                # ---------------------------------------------------------------
                # Final iteration — no more tool calls. Uses had_tool_events (not
                # tool_use_blocks) so an iteration where every tool call was
                # malformed is not mistaken for a final text-only response: it
                # still feeds a tool result back to the model (obsidian-1fmv).
                # ---------------------------------------------------------------
                content: list = []
                if iteration.text:
                    block = TextBlock(text=iteration.text)
                    self._stamp_block(block, origin)
                    content.append(block)
                # sr2-93 (FR3/AC2): archive the final SR2-visible assistant
                # output before finalization.
                await self._turn_archivist.archive_text(iteration.text)
                final_response = CompletionResponse(
                    id="turn-response",
                    content=content,
                    stop_reason="end_turn",
                    usage=iteration.usage,
                )

                # sr2-80: Finalize BEFORE yielding any events so an early
                # consumer break does not skip session-history capture.
                await self._finalize_turn(final_response)

                # Stream text/usage events to the caller (they were already
                # collected; yielding now preserves streaming semantics).
                for event in iteration.raw_events:
                    if event.type == "text" and event.text:
                        yield event
                    elif event.type == "usage" and event.usage is not None:
                        yield event

                # FR7: Surface accumulated in-band bus errors before the final end event.
                if in_band_errors:
                    yield StreamEvent(type="error", errors=in_band_errors)

                yield StreamEvent(type="end")
                return

            # ---------------------------------------------------------------
            # Tool-use iteration — queue bus events, yield, build next request
            # ---------------------------------------------------------------
            tool_use_blocks = iteration.tool_use_blocks
            tool_result_blocks = iteration.tool_result_blocks
            assistant_content = iteration.assistant_content

            # FR9: Queue tool_use_emitted on the engine bus (internal subscribers).
            # obsidian-1fmv: data carries every ToolUseBlock from this iteration,
            # including malformed ones, but never the leading TextBlock —
            # SessionResolver documents tool_use_emitted as "an assistant
            # Message containing tool_use blocks" and builds that message
            # directly from this data. Malformed calls must still be included
            # so their paired tool_result_received stays matched on the wire;
            # any preceding text is not part of that contract and is dropped
            # here exactly as it was before assistant_content existed.
            tool_use_blocks_for_bus = [
                block for block in assistant_content if isinstance(block, ToolUseBlock)
            ]
            self._engine.bus.queue(
                Event(
                    name="tool_use_emitted",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=tool_use_blocks_for_bus,
                    iteration_seq=iteration_seq,
                )
            )

            # FR9: Queue tool_result_received on the engine bus (internal subscribers).
            self._engine.bus.queue(
                Event(
                    name="tool_result_received",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=tool_result_blocks,
                    iteration_seq=iteration_seq,
                )
            )

            # FR10: Queue assistant_iteration_response — the intermediate CompletionResponse
            # from this tool-use iteration (NOT the final turn response).
            intermediate_response = CompletionResponse(
                id="iter-response",
                content=list(assistant_content),
                stop_reason="tool_use",
                usage=iteration.usage,
            )
            # sr2-93 (FR3/AC2): append this iteration's SR2-visible payload
            # items in occurrence order — the iteration's text (when any),
            # then each tool call, then each result exactly as received
            # (post-stamp, post-guard blocks; truncation markers included).
            await self._turn_archivist.archive_text(iteration.text)
            for block in tool_use_blocks_for_bus:
                await self._turn_archivist.archive(
                    block,
                    ContextReason.TOOL_USE,
                    ContextAuthority.AGENT_TENTATIVE,
                )
            for block in tool_result_blocks:
                await self._turn_archivist.archive(
                    block,
                    ContextReason.TOOL_RESULT,
                    ContextAuthority.AGENT_TENTATIVE,
                )
            self._engine.bus.queue(
                Event(
                    name="assistant_iteration_response",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=intermediate_response,
                    iteration_seq=iteration_seq,
                )
            )

            # Stream text events from this iteration to the caller.
            for event in iteration.raw_events:
                if event.type == "text" and event.text:
                    yield event
                elif event.type == "usage" and event.usage is not None:
                    yield event

            # Emit tool_use_emitted after all tool blocks are collected.
            yield StreamEvent(type="tool_use_emitted", tool_uses=tool_use_blocks)

            # Emit tool_result_received after all executor results are collected.
            yield StreamEvent(type="tool_result_received", tool_results=tool_result_blocks)

            # sr2-122 (spec spc-959 FR 11, 17): the guard stopped the run in
            # this iteration. The stop note already replaced the triggering
            # call's result and is in the events above; emit the stop event,
            # then end the run through the forced final answer with reason
            # "repeat" — regardless of final_answer.enabled. The forced call
            # gets no guard: its reply's tool calls never execute.
            if iteration.repeat_guard_stop is not None:
                stop = iteration.repeat_guard_stop
                yield StreamEvent(type="repeat_guard", iteration=iteration_seq, meta=stop.meta)
                # sr2-123 (AC9): this iteration's calls executed — the forced
                # request keeps its assistant content and the results the
                # guard gave each call (the note, or the real result on a
                # different-result max_identical_calls stop, FR 13 amended),
                # never final_answer.not_run. The bus pair queued above is
                # the iteration's only one.
                forced = await self._forced_final_answer(
                    current_request,
                    origin,
                    iteration_seq,
                    iter_text=iteration.text,
                    iter_thinking=[],
                    iter_tool_use=[],
                    reason="repeat",
                    reason_text=stop.reason_text,
                    executed=(
                        iteration.assistant_content,
                        iteration.tool_result_blocks,
                    ),
                )
                # Finalize and stream exactly like the forced_final_answer
                # branch above (sr2-121): one final_answer event, then the
                # final text, then end.
                # sr2-93 (AC2): the forced final output is archived at its
                # own seq, after the executed iteration's calls/results.
                await self._turn_archivist.archive_text(forced.text)
                final_response = CompletionResponse(
                    id="turn-response",
                    content=list(forced.assistant_content),
                    stop_reason="end_turn",
                    usage=forced.usage,
                )

                # sr2-80: Finalize BEFORE yielding any events.
                await self._finalize_turn(final_response)

                yield StreamEvent(
                    type="final_answer",
                    iteration=iteration_seq,
                    meta={
                        "reason": forced.reason,
                        "fallback_used": forced.fallback_used,
                    },
                )
                for event in forced.raw_events:
                    if event.type == "text" and event.text:
                        yield event

                # FR7: Surface accumulated in-band bus errors before the final end event.
                if in_band_errors:
                    yield StreamEvent(type="error", errors=in_band_errors)

                yield StreamEvent(type="end")
                return

            # sr2-122 (spec spc-959 FR 17): emit one repeat_guard event per
            # note this iteration produced (the stop event is emitted at the
            # stop above).
            if iteration.repeat_guard_event is not None:
                yield StreamEvent(
                    type="repeat_guard",
                    iteration=iteration_seq,
                    meta=iteration.repeat_guard_event,
                )

            # Append assistant message (with tool_use blocks) and tool_result
            # user message to the conversation for the next LLM call.
            assistant_msg = Message(role="assistant", content=assistant_content)
            tool_result_msg = Message(
                role="user",
                content=list(tool_result_blocks),  # type: ignore[arg-type]
            )

            # Build the next request by extending messages.
            next_messages = list(current_request.messages) + [assistant_msg, tool_result_msg]
            current_request = CompletionRequest(
                system=current_request.system,
                messages=next_messages,
                tools=current_request.tools,
            )

            # Inform the engine about the tool results so session/state resolvers
            # can capture them for future turns.
            tool_result_events = [
                Event(
                    name="tool_result",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=rb,
                )
                for rb in tool_result_blocks
            ]
            await self._engine.continue_turn(tool_result_events, iteration_seq)
            in_band_errors.extend(self._engine.bus.get_errors())

            # FR13: emit iteration_complete to signal the end of this tool iteration.
            yield StreamEvent(type="iteration_complete", iteration=iteration_seq)

            iteration_seq += 1

    async def _finalize_turn(self, final_response: CompletionResponse) -> None:
        """Queue assistant_response, drain via end_turn, schedule post_process.

        Called BEFORE yielding the final end event (sr2-80) so that an early
        consumer break does not silently skip session-history capture or
        memory extraction.
        """
        # Queue assistant_response on the engine bus so session resolver captures it.
        self._engine.bus.queue(
            Event(
                name="assistant_response",
                phase=EventPhase.COMPLETED,
                source_layer="orchestrator",
                data=final_response,
            )
        )

        # Finalize the turn: fire turn_end, drain the bus (processes
        # assistant_response subscribers: SessionResolver, MemoryExtractionTransformer),
        # compile, and return PipelineResult.
        await self._engine.end_turn()

        # FR4+FR5+FR6: Schedule post_process as a deferred task. The client
        # is freed immediately; the next turn() will await this task before
        # proceeding. Capture final_response by argument.
        self._pp_task = asyncio.create_task(
            self._finalize_and_post_process(final_response)
        )

    # ------------------------------------------------------------------
    # Post-processing (MVP no-op)
    # ------------------------------------------------------------------

    async def _finalize_and_post_process(self, final_response: CompletionResponse) -> None:
        """Deferred post-processing: called by the scheduled task after end_turn.

        Captures final_response by argument to avoid closure issues.
        """
        await self.post_process(final_response)

    async def aclose(self) -> None:
        """Explicit shutdown: await any pending deferred post_process task.

        Safe when no task is pending (no-op). Idempotent — clears _pp_task
        so subsequent calls are also no-ops.

        FR9: Surfaces errors from the deferred task rather than swallowing them.
        """
        if self._pp_task is None:
            return

        try:
            await self._pp_task
        finally:
            self._pp_task = None

    async def post_process(self, response: CompletionResponse) -> None:
        """Post-turn processing hook. No-op in MVP."""
        return
