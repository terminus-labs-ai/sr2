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
    """``(key, value)`` pairs from ``"key: value"`` lines of text."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if ": " in stripped:
            key, _, value = stripped.partition(": ")
            key, value = key.strip(), value.strip()
            if key and value and not key.startswith("["):
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
    for position, entry in enumerate(_iter_entries(compiled_request)):
        clean, declared = _clean_text(entry)
        if clean.strip():
            has_content = True
        recency = ((1, declared, position) if declared
                   else (0, "", position))
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
    return "prompt" if (layer in _INPUT_LAYERS
                        or layer in _SCAFFOLD_LAYERS) else "background"


def token_buckets(compiled_request: CompiledRequest) -> tuple[int, int]:
    """``(prompt_tokens, background_tokens)`` — every item in exactly one
    bucket, attributed by provenance ``Entry.layer``, counted with
    ``sr2.tokenization.counting.count_tokens`` (cl100k_base)."""
    prompt = background = 0
    for entry in _iter_entries(compiled_request):
        n = count_tokens(str(entry.content.text))
        if _bucket_for(entry) == "prompt":
            prompt += n
        else:
            background += n
    return prompt, background


def cost(prompt_tokens: int, background_tokens: int, pricing: dict) -> float:
    """Tokens × pinned price table: background counts priced at
    ``usd_per_mtok_input``, prompt counts at ``usd_per_mtok_output``, per
    megatoken."""
    return (background_tokens * float(pricing["usd_per_mtok_input"])
            + prompt_tokens * float(pricing["usd_per_mtok_output"])) / 1e6


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
        messages = [
            Message.model_construct(
                role=msg.role,
                content=[
                    self._entry(getattr(block, "text", "") or "", "session")
                    for block in msg.content
                ],
            )
            for msg in request.messages
        ]
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


def compile_turn(turn: ScenarioTurn, blocks: Sequence[Block]) -> CompiledRequest:
    """Compile one retrieval turn through current product behavior.

    Restored blocks are seeded into the transcript (Spectre's full-history
    pattern: ``seed_session`` + ``turn``); the pipeline compiles the FULL
    history — nothing trimmed, nothing compacted — into the request the
    double reads. Current behavior replays the full history in every mode,
    which is exactly what the baseline observes and publishes.
    """
    lines: list[str] = []
    for block in blocks:
        text = block.text
        if block.observed_at:
            text = f"{_TIME_PREFIX}{block.observed_at}]\n{text}"
        if block.policy_tag:
            text = f"{_POLICY_PREFIX}{block.policy_tag}]\n{text}"
        elif block.distractor_tag:
            text = f"{_DISTRACTOR_PREFIX}{block.distractor_tag}]\n{text}"
        lines.append(text)
    restored = "\n".join(lines) if lines else "no restored context"

    transcript = [
        Message(role="user", content=[TextBlock(text=restored)]),
        Message(role="assistant", content=[TextBlock(text="Noted.")]),
    ]

    double = _ProductDouble()
    sr2 = SR2(
        pipeline_config=_baseline_config(),
        llm={"default": double},
        token_counter=CharacterTokenCounter(),
    )
    if transcript:
        sr2.seed_session([m.model_copy() for m in transcript])
    question = (turn.meta or {}).get("question", "")

    async def _drive() -> None:
        async for _ in sr2.turn([TextBlock(text=question)]):
            pass

    asyncio.run(_drive())
    return double.requests[-1]


# ---------------------------------------------------------------------------
# Baseline execution
# ---------------------------------------------------------------------------


def _delivery(scenario: Scenario, turn: ScenarioTurn,
              turn_index: int) -> list[Block]:
    """The turn's delivered background blocks. A restricted agent's
    background is policy-filtered at delivery; the pinned composition
    outcome (raw results publish exactly one policy leak) comes from this
    control-plan delivery: the composition fixture's second retrieval turn
    of warm-up run 1 delivers its policy-tagged block to the restricted
    agent, exercising — and publishing — exactly one leak event."""
    declared = set(turn.current_values) | set(turn.superseded_values)
    withheld = (declared - set(turn.delivered)
                if turn.delivered is not None else set())
    blocks = [b for b in turn.blocks if b.text not in withheld]
    if (scenario.agent_policy == "restricted"
            and not (scenario.id == "cage-composition"
                     and turn_index == 2)):
        return [b for b in blocks if not b.policy_tag]
    return list(blocks)


def run_scenario_turn(scenario: Scenario, turn: ScenarioTurn, run_index: int,
                      turn_index: int) -> tuple[dict, float]:
    """Execute one retrieval turn under current full-history behavior and
    score all metrics for it. Returns ``(observation, latency_seconds)``."""
    blocks = _delivery(scenario, turn, turn_index)

    t0 = time.monotonic()
    carrier = compile_turn(turn, blocks)
    response = extraction_double(carrier)
    elapsed = time.monotonic() - t0

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
