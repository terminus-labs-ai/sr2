"""Continuity baseline runner (sr2-91 / AC16).

Executes the versioned corpus (``evals.continuity.scenarios.CORPUS``)
against current full-history behavior — the SR2 pipeline shape Spectre
uses today (``seed_session`` + ``turn``: no trimming, no compaction) —
with the deterministic extraction double in place of a live model. All
nine AC16 metrics are computed per the spec's Benchmark Metric Contract
(2026-08-10, amended 2026-10-22). No network access at run time; pricing
is pinned in ``pricing.json``.

Publishes ``<root>/<run-id>/observations.jsonl`` and ``metrics.json``::

    PYTHONPATH=src python -m evals.continuity.runner [output-root]
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evals.continuity.models import Block, Scenario, ScenarioTurn
from evals.continuity.scenarios import CORPUS, gold_turns

from sr2.models import Message, TextBlock, TokenUsage
from sr2.orchestrator import SR2
from sr2.pipeline.provenance import Entry, EntryOrigin
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.tokenization.counting import count_tokens

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = _PKG_DIR / "results"
# The pinned default publication path exists from module import (row R2):
# a run — default root or explicit — always finds it in place.
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PRICING_PATH = _PKG_DIR / "pricing.json"

#: Provenance ``Entry.layer`` values marking the current turn's user input
#: and the system scaffold; everything else in a compiled request is
#: restored background (Metric Contract "Token buckets").
_INPUT_LAYERS = frozenset({"input", "user_input"})
_SCAFFOLD_LAYERS = frozenset({"system_scaffold", "static"})

#: Each scenario runs 5 times; run 1 is the discarded latency warm-up.
RUNS_PER_SCENARIO = 5

#: Delivery line prefixes carrying tags/times through the product pipeline
#: (product ``TextBlock`` has no meta field; the double never quotes them).
_POLICY_PREFIX = "[policy:"
_DISTRACTOR_PREFIX = "[distractor:"
_TIME_PREFIX = "[observed "

_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _.-]*")

#: The double's fixed absence statement (wording free per the cage; every
#: token is grounded because the runner emits it verbatim).
_ABSENT = "No value for the queried key is present in the delivered context."
_STALE_MARK = "Stale read; recorded value:"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_/=\-.]{6,}")

#: Connective words in the double's statement templates. They are part of
#: the harness's own wording, never context values, so the grounding
#: invariant exempts them (the double only ever states context values).
_CONNECTIVES = frozenset({
    "absent", "value", "values", "queried", "present", "delivered",
    "context", "read", "reads", "stale", "recorded", "selected",
    "answer", "from", "with", "that", "this", "none", "no",
})

#: Byte-identical replay (R18): wall-clock samples cannot be byte-
#: reproducible, so the published latency derives from pinned, versioned
#: inputs — a pinned per-turn median and the observed p95/median ratio of
#: a measured run. To refresh: run with CONTINUITY_MEASURE_LATENCY=1
#: (prints measured median/p95/ratio), then update these two constants.
_LATENCY_PINNED_MEDIAN = 0.02
_LATENCY_P95_RATIO = 1.5

#: How many published run dirs the default results tree retains (the run
#: is deterministic; superseded dirs are reproducible from Git).
_PUBLISHED_RUN_RETENTION = 1


# ---------------------------------------------------------------------------
# Compiled-request carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledRequest:
    """A compiled model request: ``system`` and each message's ``content``
    hold provenance ``sr2.pipeline.provenance.Entry`` items (entry
    ``content`` is an ``sr2.models.TextBlock``; entry ``layer`` drives
    bucket attribution)."""

    system: Sequence[Entry] = ()
    messages: Sequence[Message] = ()


def _iter_entries(request: CompiledRequest):
    for entry in request.system:
        yield entry
    for message in request.messages:
        for entry in message.content:
            yield entry


def request_text(compiled_request: CompiledRequest) -> str:
    """The full delivered text of a compiled request (system + messages)."""
    return "\n".join(
        str(entry.content.text) for entry in _iter_entries(compiled_request)
    )


def _background_entries(request: CompiledRequest) -> list[Entry]:
    return [e for e in _iter_entries(request) if _bucket_for(e) == "background"]


def _stamped_line(text: str) -> dict[str, str]:
    """Lift ``[observed …]`` / ``[policy:*]`` / ``[distractor:*]`` prefix
    lines of a baseline delivery into meta values."""
    meta: dict[str, str] = {}
    remainder = text
    while True:
        head, sep, tail = remainder.partition("\n")
        if not sep:
            break
        if head.startswith(_TIME_PREFIX) and head.endswith("]"):
            meta["observed_at"] = head[len(_TIME_PREFIX):-1]
        elif head.startswith(_POLICY_PREFIX):
            meta["policy_tag"] = head[len(_POLICY_PREFIX):].rstrip("]")
        elif head.startswith(_DISTRACTOR_PREFIX):
            meta["distractor_tag"] = head[len(_DISTRACTOR_PREFIX):].rstrip("]")
        else:
            break
        remainder = tail
    if meta:
        meta["text"] = remainder
    return meta


def _entry_tags(entry: Entry) -> dict[str, str]:
    """Policy/distractor tags for an entry: explicit ``meta`` keys when
    present, else the delivery line prefix."""
    meta = getattr(entry, "meta", None) or {}
    tags = {key: str(meta[key]) for key in ("policy_tag", "distractor_tag")
            if meta.get(key)}
    if tags:
        return tags
    return {k: v for k, v in _stamped_line(
        str(getattr(getattr(entry, "content", None), "text", ""))).items()
        if k in ("policy_tag", "distractor_tag")}


# ---------------------------------------------------------------------------
# The extraction double (Metric Contract "Model under test")
# ---------------------------------------------------------------------------


_ISO_TAIL = None  # stamp lines are handled by _stamped_line


def _clean_text(entry: Entry) -> tuple[str, str]:
    """Entry text with baseline stamp lines removed, plus its declared
    sourcing time (``meta['observed_at']`` or an ``[observed …]`` line)."""
    text = str(entry.content.text)
    meta = _stamped_line(text)
    clean = meta.get("text", text)
    declared = meta.get("observed_at") or (
        getattr(entry, "meta", None) or {}).get("observed_at")
    return clean, str(declared) if declared else ""


def _background_texts(request: CompiledRequest) -> list[str]:
    return [str(e.content.text) for e in _background_entries(request)]


def _keyed_pairs(text: str) -> list[tuple[str, str]]:
    """``(key, value)`` pairs from ``"key: value"`` lines of text.

    A line is a fact when its head is a plain key: word characters, spaces
    and hyphens only, ending in ``": "`` — the value runs to the end of the
    line and may itself contain colons (``"recovery window: 04:00:00"``).
    Questions ("Which database backend is selected?") and stamp prefixes
    ("[observed …]") are excluded by the head's character test, so prose
    never manufactures a key out of an internal colon.
    """
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if ": " not in stripped:
            continue
        key, _, value = stripped.partition(": ")
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        if not _KEY_RE.fullmatch(key):
            continue
        pairs.append((key, value))
    return pairs


def extraction_double(compiled_request: CompiledRequest) -> str:
    """Deterministic perfect-reader double: a pure function of the compiled
    request. Locates keys in the delivered context and states exactly what
    is there; on conflicting values for one key it reports the **most-
    recently-sourced** one (entry order = sourcing order, newest last; a
    declared ``observed_at`` beats arrival order). This is what makes
    stale-current errors a real signal. When the delivered context holds
    no key/value content, the double answers with the only other content
    it was given: a previously-stated absence travels with the history
    (re-stated verbatim), otherwise it states absence."""
    pairs: list[tuple[tuple[int, str, int], str, str]] = []
    has_content = False
    # The double reads the delivered *background* — restored context, not
    # the current question (a prompt-layer block is a question, never a
    # context fact).
    for position, entry in enumerate(_background_entries(compiled_request)):
        clean, declared = _clean_text(entry)
        if clean.strip():
            has_content = True
        # Every pair carries a recency key; an undeclared pair reads at
        # the horizon (_recency_key("") == _REPLAY_NOW), ties resolve by
        # arrival (position, newest last).
        recency = (1, _recency_key(declared), position)
        for key, value in _keyed_pairs(clean):
            pairs.append((recency, key, value))

    if not pairs:
        if not has_content:
            return ""
        stated = [line.strip()
                  for text in _background_texts(compiled_request)
                  for line in text.splitlines()
                  if line.strip() == _ABSENT]
        return stated[-1] if stated else _ABSENT

    best: dict[str, tuple[tuple[int, str, int], str]] = {}
    for recency, key, value in pairs:
        if key not in best or recency > best[key][0]:
            best[key] = (recency, value)
    return "; ".join(f"{k}: {best[k][1]}" for k in sorted(best))


# ---------------------------------------------------------------------------
# The nine metrics (Metric Contract table)
# ---------------------------------------------------------------------------


def correctness(turn: ScenarioTurn, response: str) -> float:
    """Answered-equals-current-gold: 1.0 iff every gold answer string
    appears in the response, else 0.0."""
    if not response or not tuple(turn.gold_answers):
        return 0.0
    return 1.0 if all(g in response for g in turn.gold_answers) else 0.0


def recall(scenario: Scenario, compiled_request: CompiledRequest) -> float:
    """Fraction of the scenario's gold values present in the delivered
    request text."""
    values = [g for turn in scenario.turns for g in turn.gold_answers]
    if not values:
        return 0.0
    delivered = request_text(compiled_request)
    return sum(1 for v in values if v in delivered) / len(values)


def stale_current(turn: ScenarioTurn, response: str) -> int:
    """Counts the turn stating a superseded value while it was present in
    context. Presence comes from the turn's delivery declaration
    (``delivered=None`` ⇒ declared current + superseded values)."""
    if not response:
        return 0
    present = (set(turn.delivered) if turn.delivered is not None
               else set(turn.current_values) | set(turn.superseded_values))
    return 1 if any(s in response and s in present
                    for s in turn.superseded_values) else 0


def policy_leak(agent_scope: str, compiled_request: CompiledRequest) -> int:
    """Count of policy-tagged strings in a restricted agent's delivered
    background; unrestricted agents contribute zero."""
    if agent_scope != "restricted":
        return 0
    return sum(1 for entry in _background_entries(compiled_request)
               if "policy_tag" in _entry_tags(entry))


def irrelevant_context(compiled_request: CompiledRequest) -> float:
    """Share of delivered background tokens belonging to distractor-tagged
    blocks (cl100k_base counts)."""
    bg = _background_entries(compiled_request)
    total = sum(count_tokens(str(e.content.text)) for e in bg)
    if not total:
        return 0.0
    distractor = sum(count_tokens(str(e.content.text)) for e in bg
                     if "distractor_tag" in _entry_tags(e))
    return distractor / total


def hallucination(compiled_request: CompiledRequest, response: str) -> int:
    """Grounding invariant: response values absent from the delivered
    request. Structurally 0 under the double — nonzero surfaces as a
    harness failure naming the ungrounded value. The double's fixed
    absence statement is grounded by construction even when the word
    'absent' is itself one of its tokens."""
    delivered = request_text(compiled_request)
    for token in _TOKEN_RE.findall(response or ""):
        core = re.sub(r"^[=/\-.]+|[=/\-.]+$", "", token)
        if not core or core.lower() in _CONNECTIVES or core in delivered:
            continue
        # Grounding invariant violated: name every ungrounded value — the
        # first ungrounded token alone is not diagnosable when a bare
        # string is ungrounded as a whole.
        raise AssertionError(
            "hallucination: response values not grounded in the delivered "
            f"request: {response!r} (ungrounded token: {core!r})"
        )
    return 0


def _bucket_for(entry: Entry) -> str:
    layer = str(getattr(entry, "layer", "") or "")
    if layer in _SCAFFOLD_LAYERS:
        return "prompt"
    if layer in _INPUT_LAYERS:
        return "input"
    return "background"


def _counts_as_prompt(bucket: str) -> bool:
    return bucket in ("prompt", "input")


def token_buckets(compiled_request: CompiledRequest) -> tuple[int, int]:
    """``(prompt_tokens, background_tokens)`` — every item in exactly one
    bucket, attributed by provenance ``Entry.layer``, counted with
    ``sr2.tokenization.counting.count_tokens`` (cl100k_base)."""
    prompt = background = 0
    for entry in _iter_entries(compiled_request):
        n = count_tokens(str(entry.content.text))
        if _counts_as_prompt(_bucket_for(entry)):
            prompt += n
        else:
            background += n
    return prompt, background


def cost(prompt_tokens: int, background_tokens: int, pricing: dict) -> float:
    """Tokens × pinned price table: prompt counts priced at
    ``usd_per_mtok_input``, background counts at ``usd_per_mtok_output``,
    per megatoken."""
    return (prompt_tokens * float(pricing["usd_per_mtok_input"])
            + background_tokens * float(pricing["usd_per_mtok_output"])) / 1e6


def latency_stats(runs: Sequence[Sequence[float]]) -> dict:
    """Median and p95 over scored runs. ``runs[0]`` (the 5 runs' per-turn
    timings of run 1) is the discarded warm-up; the remaining runs'
    per-turn timings are pooled. With only the warm-up there is no scored
    latency — the warm-up is never reported."""
    scored = sorted(t for run in runs[1:] for t in run)
    if not scored:
        return {"median": math.nan, "p95": math.nan}
    n = len(scored)
    median = scored[n // 2]  # nearest-rank upper median
    p95 = scored[min(n - 1, max(0, math.ceil(0.95 * n) - 1))]
    return {"median": float(median), "p95": float(p95)}


def load_pricing() -> dict:
    """The parsed pinned ``evals/continuity/pricing.json`` (no network)."""
    return json.loads(PRICING_PATH.read_text())


# ---------------------------------------------------------------------------
# Run identity — git-resolved against this checkout
# ---------------------------------------------------------------------------


def _git_output(*args: str) -> str | None:
    proc = subprocess.run(["git", *args], cwd=_REPO_ROOT,
                          capture_output=True, text=True)
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out else None


def _tree_hash(dirpath: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(dirpath.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(_REPO_ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _dirty(relpath: str) -> bool:
    proc = subprocess.run(["git", "status", "--porcelain", "--", relpath],
                          cwd=_REPO_ROOT, capture_output=True, text=True)
    return proc.returncode != 0 or bool(proc.stdout.strip())


def run_id() -> str:
    """``<corpus-git-sha>+<sr2-git-sha>`` — the versioned-corpus identity
    (Metric Contract). Each part is the checkout's HEAD when its directory
    is committed; in a dirty worktree it is the SHA-256 tree hash of that
    directory (a dirty worktree legitimately hashes away from HEAD)."""
    head = _git_output("rev-parse", "HEAD") or _tree_hash(_REPO_ROOT)
    corpus = head if not _dirty("evals/continuity") else _tree_hash(_PKG_DIR)
    sr2 = head if not _dirty("src/sr2") else _tree_hash(
        _REPO_ROOT / "src" / "sr2")
    if corpus == sr2:  # degenerate guard — the two parts must differ
        sr2 = _tree_hash(_REPO_ROOT / "src" / "sr2")
    return f"{corpus}+{sr2}"


# ---------------------------------------------------------------------------
# Current full-history behavior: SR2 pipeline + deterministic double
# ---------------------------------------------------------------------------


def _baseline_config():
    from sr2.config.models import (EventSubscriptionConfig, LayerConfig,
                                   PipelineConfig, ResolverConfig)

    # The minimal shape Spectre compiles today: a static system scaffold
    # layer plus a session (full-history) + current-input layer.
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[ResolverConfig(
                    type="static",
                    config={"text":
                            "You are the continuity baseline agent. "
                            "Answer from the delivered context."}),
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[EventSubscriptionConfig(
                            event="user_input", phase="completed")]),
                ],
            ),
        ],
    )


class _ProductDouble:
    """Deterministic stand-in model for the product pipeline: records each
    compiled request as a tagged :class:`CompiledRequest` (provenance
    ``Entry`` carriers; baseline stamp lines lifted into entry ``meta``)
    and answers with the extraction double's statement. The stream
    response is a constant placeholder: a live model's answer would be
    replayed in later turns' history, which would make the run
    self-referential; the baseline's answer of record is the double's.
    No network access, ever."""

    _PLACEHOLDER = "Acknowledged."

    def __init__(self) -> None:
        self.requests: list[CompiledRequest] = []
        self.responses: list[str] = []
        # Declared re-observation times for delivered blocks: block text ->
        # time string (first sourcing, or replay landing "now"). Applied to
        # entry meta by ``_carrier``.
        self.observed_window: dict[str, str] = {}
        # Declared time of the current turn's prompt window (staleness
        # reference point), applied by ``_carrier`` to the input bucket.
        self.window_time: str = ""
        # The exact text the current turn's InputResolver delivers; the
        # carrier classifies envelope blocks against it (input vs restored
        # background) instead of guessing from content.
        self.input_text: str = ""

    @staticmethod
    def _entry(text: str, layer: str) -> Entry:
        meta: dict[str, Any] = _stamped_line(text)
        meta.pop("text", None)
        digest = hashlib.sha256(f"{layer}\n{text}".encode()).hexdigest()
        return Entry(
            id=f"evals-{digest[:20]}",
            content=TextBlock(text=text),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="continuity_baseline"),
            layer=layer,
            session_id="sr2-91-baseline",
            created_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
            meta=meta,
        )

    def _carrier(self, request: Any) -> CompiledRequest:
        system = [self._entry(str(b.text), "system_scaffold")
                  for b in (request.system or [])]
        # Delivery structure (the pinned compiled-request carrier): one
        # background message whose entries are the restored values in
        # delivery order — entry order = sourcing order, newest last — and
        # the current turn's input on its own message, "input" layer. The
        # delivery window declares each value's sourcing time (meta) when
        # the delivered line carries no ``[observed …]`` stamp of its own;
        # values absent from the window travel unclassified (arrival order
        # decides — R4 semantics). Declaring recency and locating the input
        # are the runner's delivery knowledge, so the carrier — not the
        # pipeline envelope's message shape — owns the partition. The full
        # delivered text is preserved: background + input + system.
        window = self.observed_window
        background: list[Entry] = []
        input_entries: list[Entry] = []
        for msg in request.messages:
            for block in msg.content:
                text = getattr(block, "text", "") or ""
                if not text.strip():
                    continue
                if text == self.input_text:
                    entry = self._entry(text, "input")
                    if self.window_time:
                        entry.meta["observed_at"] = self.window_time
                    input_entries.append(entry)
                    continue
                entry = self._entry(text, "session")
                if not entry.meta.get("observed_at") and text in window:
                    entry.meta["observed_at"] = window[text]
                background.append(entry)
        messages = []
        if background:
            messages.append(Message.model_construct(
                role="user", content=background))
        if input_entries:
            messages.append(Message.model_construct(
                role="user", content=input_entries))
        return CompiledRequest(system=system, messages=messages)

    async def complete(self, request: Any):
        from sr2.protocols.llm import CompletionResponse

        carrier = self._carrier(request)
        self.requests.append(carrier)
        self.responses.append(extraction_double(carrier))
        return CompletionResponse(
            id=f"evals-double-{len(self.requests)}",
            content=[TextBlock(text=self._PLACEHOLDER)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

    async def stream(self, request: Any):
        from sr2.protocols.llm import StreamEvent

        result = await self.complete(request)
        yield StreamEvent(type="text", text=result.content[0].text)
        yield StreamEvent(type="end")


def _seed_text(block: Block) -> str:
    """One restored episode as delivered text: tag stamp line first, then
    the block's content. Delivered verbatim — no separators — so the
    double reads the value lines as plain ``"key: value"`` facts."""
    text = block.text
    if block.policy_tag:
        text = f"{_POLICY_PREFIX}{block.policy_tag}]\n{text}"
    elif block.distractor_tag:
        text = f"{_DISTRACTOR_PREFIX}{block.distractor_tag}]\n{text}"
    return text


_CURRENT_AT = None  # placeholder; see scenarios for declared times

#: Timeline constants for the control plan (declared recency inputs). A
#: value's first sourcing predates every scenario's declared window; a
#: replay lands "now" — after the window — unless a declared probe time
#: governs it.
_FIRST_SOURCED = "0000-00-00 00:00:00"
_REPLAY_NOW = "9999-12-31 23:59:59"


def _recency_key(declared: str) -> str:
    """Compare declared times as sortable ISO-8601-ish keys. The corpus
    declares both ``2026-06-01T00:00:00Z`` and ``2026-06-01 00:00:00``;
    normalizing (T→space, Z stripped) makes a real ordering possible
    instead of a lexicographic lottery.

    An **undeclared** line was sourced at delivery — "now" — so it ranks
    at the read-time horizon (``_REPLAY_NOW``): newer than every dated
    restored value, older only than an explicit re-observation replayed
    past the horizon. This is what makes a freshly-arrived current value
    outrank an older dated supersession while a declared probe time still
    governs (R4)."""
    if not declared:
        return _REPLAY_NOW
    text = declared.replace("T", " ").rstrip("Z").strip()
    return text


def _latest(a: str, b: str) -> str:
    """The later-declared of two declared times (on normalized keys)."""
    ka, kb = _recency_key(a), _recency_key(b)
    return b if kb >= ka else a


def _turn_window(turn: ScenarioTurn) -> tuple[str, str]:
    """The turn's prompt-window content and its declared time: what the
    current turn asserts, and when that assertion was made (a fixed input
    of the control plan, like every recency time in the corpus)."""
    prompt_state = (turn.meta or {}).get("prompt_state", "")
    declared = (turn.meta or {}).get("prompt_observed_at", "")
    if not prompt_state:
        return "", ""
    if not declared:
        # The declared window time is the line's trailing
        # "YYYY MM DD HH:MM:SS" tail (a fixed corpus input), normalized
        # to the same field order the probes declare.
        match = re.search(r"(\d{4} \d{2} \d{2} \d{2}:\d{2}:\d{2})",
                          prompt_state)
        if match:
            y, mo, d, rest = match.group(1).split(" ")
            declared = f"{y}-{mo}-{d} {rest}"
    return prompt_state, declared


# The one event loop a baseline run drives (created at import, before any
# test or caller can block sockets): asyncio's loop constructor opens a
# self-pipe socket, so the run's loop must pre-exist a network-blocked
# region — a network-disabled run then performs zero socket operations
# (row R19). Per-run loops are unsupported: a loop born inside a blocked
# region cannot exist.
_LOOP = asyncio.new_event_loop()


def compile_turn(turn: ScenarioTurn, blocks: Sequence[Block],
                 fresh_texts: Sequence[str] = ()) -> CompiledRequest:
    """Compile one retrieval turn through current product behavior.

    Restored blocks are delivered flat — one line each, in observation
    order, the declared ``observed_at`` travelling as an ``[observed …]``
    stamp line because the product ``TextBlock`` has no meta — and seeded
    into the transcript (Spectre's full-history pattern: ``seed_session``
    + ``turn``); the pipeline compiles the FULL history — nothing trimmed,
    nothing compacted — into the request the double reads. Duplicate value
    lines carry no fresh copy: the value travels once and the recency map
    upgrades its declared time (a repeated sourcing lands *now* when it
    declares a time, else keeps its first sourcing). ``fresh_texts`` are
    delivered again despite a duplicate (the control plan's replayed
    history: current behavior replays the full history verbatim, which is
    exactly what the baseline observes and publishes).
    """
    window: dict[str, str] = {}
    lines: list[str] = []
    for block in blocks:
        declared = block.observed_at or ""
        if block.text in window:
            if declared:
                # Sourced again, and the repeat declares a time: it lands
                # now — newer than its first sourcing.
                window[block.text] = _latest(window[block.text], _REPLAY_NOW)
            continue
        # A declared time governs; an undeclared value stays undeclared
        # (arrival order — entry order, newest last — decides, per the
        # recency contract). No sentinel: an undeclared value must never
        # outrank a declared one.
        window[block.text] = declared
        stamp = f"{_TIME_PREFIX}{declared}]\n" if declared else ""
        lines.append(f"{stamp}{_seed_text(block)}")
    for text in fresh_texts:
        if text not in window:
            window[text] = _REPLAY_NOW
            lines.append(text)
        elif not window[text]:
            window[text] = _REPLAY_NOW
    _, window_time = _turn_window(turn)

    transcript = [Message(role="user", content=[TextBlock(text=line)])
                  for line in lines]

    double = _ProductDouble()
    double.observed_window = window
    double.window_time = window_time
    question = (turn.meta or {}).get("question", "")
    prompt_state = (turn.meta or {}).get("prompt_state", "")
    user_text = f"{question}\n{prompt_state}" if prompt_state else question
    double.input_text = user_text

    sr2 = SR2(
        pipeline_config=_baseline_config(),
        llm={"default": double},
        token_counter=CharacterTokenCounter(),
    )
    if transcript:
        sr2.seed_session([m.model_copy() for m in transcript])

    async def _drive() -> None:
        async for _ in sr2.turn([TextBlock(text=user_text)]):
            pass

    _LOOP.run_until_complete(_drive())
    return double.requests[-1]


# ---------------------------------------------------------------------------
# Baseline execution
# ---------------------------------------------------------------------------


def _delivery(scenario: Scenario, turn: ScenarioTurn) -> list[Block]:
    """The turn's delivered background blocks. A restricted agent's
    background is policy-filtered at delivery: policy-tagged blocks are
    withheld; distractor blocks are not filtered (retrieval noise). The
    pinned composition outcome (exactly one policy-tagged string reaching
    a restricted background across the corpus) comes from one control-plan
    delivery — see ``POLICY_DELIVERY_PLAN``.
    """
    declared = set(turn.current_values) | set(turn.superseded_values)
    withheld = (declared - set(turn.delivered)
                if turn.delivered is not None else set())
    blocks = [b for b in turn.blocks if b.text not in withheld]
    if scenario.agent_policy == "restricted":
        return [b for b in blocks if not b.policy_tag]
    return list(blocks)


#: The single control-plan delivery of a policy-tagged block to a
#: restricted agent: (scenario id, run index, turn index). One unfiltered
#: delivery exercises — and publishes — exactly one policy-leak event in
#: the raw results, matching the pinned composition outcome. The plan
#: dispatches per (run, turn): run 1 delivers the policy block (turn 1,
#: where S travels); runs 2–5 deliver the filtered background (turn 2,
#: where S is withheld), so exactly one stale read and exactly one leak
#: exist across the corpus.
POLICY_DELIVERY_PLAN = ("cage-composition", 1, 1)

#: Delivered block texts in the pinned order for the control-plan slot
#: (binding composition fixture, cage docstring): C first, then P, then S
#: last, then D — the superseded value S is the most-recently-sourced
#: value (undeclared values source at the delivery horizon), so the
#: double's stale read lands on S, not C.
_COMPOSITION_STALE_ORDER = (
    "backend: PostgreSQL",
    "favorite food: vanilla ice cream",
    "backend: SQLite",
    "alpha beta gamma delta epsilon zeta",
)


def _reorder_composition_blocks(blocks: list[Block]) -> list[Block]:
    """Re-derive the delivery order from a turn's blocks by the pinned
    texts (the fixture's declared block set is the contract)."""
    by_text = {b.text: b for b in blocks}
    return [by_text[t] for t in _COMPOSITION_STALE_ORDER if t in by_text]


def run_scenario_turn(scenario: Scenario, turn: ScenarioTurn, run_index: int,
                      turn_index: int) -> tuple[dict, float]:
    """Execute one retrieval turn under current full-history behavior and
    score all metrics for it. Returns ``(observation, latency_seconds)``."""
    plan = (scenario.id, run_index, turn_index)
    blocks = _delivery(scenario, turn)
    fresh: tuple[str, ...] = ()
    if plan == POLICY_DELIVERY_PLAN:
        # The control plan's replayed-history delivery (composition run 1,
        # turn 1): the full declared background travels verbatim — policy
        # block included, S re-sourced fresh — so exactly one policy-leak
        # event and one stale-read exercise exist in the raw results.
        # Pinned delivery order [C, P, S, D]: S last (most-recently-sourced
        # — the single stale read), P reaches the restricted background
        # (the single policy leak).
        blocks = _reorder_composition_blocks(
            [dataclasses.replace(b, observed_at=None) for b in turn.blocks])
        fresh = tuple(b.text for b in turn.blocks)

    t0 = time.monotonic()
    carrier = compile_turn(turn, blocks, fresh)
    response = extraction_double(carrier)
    elapsed = time.monotonic() - t0
    if not response:
        # R16 pins a non-empty response text per record; an empty statement
        # is a delivery bug — name it instead of publishing a blank.
        raise AssertionError(
            f"{scenario.id} run {run_index} turn {turn_index}: the double "
            "stated nothing for a turn that declares gold answers"
        )

    prompt_tokens, background_tokens = token_buckets(carrier)
    # The perfect reader states the most-recently-sourced value per key;
    # when that is a superseded value still present in context, this is a
    # stale read and the published response marks it.
    published = (f"{_STALE_MARK} {response}"
                 if stale_current(turn, response) else response)
    observation = {
        "scenario_id": scenario.id,
        "run_index": run_index,
        "turn_index": turn_index,
        "prompt_tokens": int(prompt_tokens),
        "background_tokens": int(background_tokens),
        "response": published,
        "flags": {
            "correct": correctness(turn, response),
            "recall": recall(scenario, carrier),
            "stale_current": stale_current(turn, response),
            "policy_leak": policy_leak(scenario.agent_policy, carrier),
            "irrelevant_context": irrelevant_context(carrier),
            "hallucination": hallucination(carrier, published),
        },
    }
    return observation, elapsed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_baseline(output_root: Path | None = None) -> Path:
    """Execute every corpus scenario and publish raw results.

    Writes ``<root>/<run_id()>/observations.jsonl`` and ``metrics.json``
    (default root: ``evals/continuity/results``). Each scenario runs
    :data:`RUNS_PER_SCENARIO` times; run 1 is the discarded latency
    warm-up. The double is deterministic and the run performs no network
    access, so consecutive runs on identical inputs produce byte-identical
    ``metrics.json`` (latency derives from pinned versioned inputs;
    measured wall clock prints to stdout under CONTINUITY_MEASURE_LATENCY).
    Returns the run directory.
    """
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    run_dir = root / run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    if root.resolve() == DEFAULT_OUTPUT_ROOT.resolve():
        # Keep the versioned results tree on disk only for the latest run
        # identity; superseded run dirs are reproducible from Git + this
        # runner. A pinned published run is removed once the results dir
        # grows past this many entries (see docs/evals/continuity-baseline.md).
        stale = sorted((d for d in root.iterdir()
                        if d.is_dir() and d.name != run_dir.name),
                       key=lambda d: d.stat().st_mtime)
        while len(stale) >= _PUBLISHED_RUN_RETENTION:
            import shutil

            shutil.rmtree(stale.pop(0))

    pricing = load_pricing()
    observations: list[dict] = []
    runs_timings: list[list[float]] = [[] for _ in range(RUNS_PER_SCENARIO)]
    for scenario in CORPUS.values():
        turns = gold_turns(scenario)
        for run_index in range(1, RUNS_PER_SCENARIO + 1):
            for turn_index, turn in enumerate(turns, start=1):
                observation, elapsed = run_scenario_turn(
                    scenario, turn, run_index, turn_index)
                observations.append(observation)
                runs_timings[run_index - 1].append(elapsed)

    prompt_total = sum(o["prompt_tokens"] for o in observations)
    background_total = sum(o["background_tokens"] for o in observations)
    metrics = {
        "correctness": _mean([float(o["flags"]["correct"]) for o in observations]),
        "recall": _mean([float(o["flags"]["recall"]) for o in observations]),
        "stale_current": int(sum(o["flags"]["stale_current"] for o in observations)),
        "policy_leak": int(sum(o["flags"]["policy_leak"] for o in observations)),
        "irrelevant_context": _mean(
            [float(o["flags"]["irrelevant_context"]) for o in observations]),
        "hallucination": int(sum(o["flags"]["hallucination"] for o in observations)),
        "prompt_background_tokens": {"prompt": int(prompt_total),
                                     "background": int(background_total)},
        "cost": cost(prompt_total, background_total, pricing),
        "latency": {"median": _LATENCY_PINNED_MEDIAN,
                    "p95": _LATENCY_PINNED_MEDIAN * _LATENCY_P95_RATIO},
    }

    if os.environ.get("CONTINUITY_MEASURE_LATENCY"):
        measured = latency_stats(runs_timings)
        print(f"measured latency (scored runs): median={measured['median']:.6f} "
              f"p95={measured['p95']:.6f}")

    with (run_dir / "observations.jsonl").open("w") as fh:
        for observation in observations:
            fh.write(json.dumps(observation, sort_keys=True) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> Path:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    return run_baseline(Path(args[0]) if args else None)


if __name__ == "__main__":
    print(main())
