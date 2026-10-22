"""sr2-91 / AC16 — continuity baseline benchmark cage (unit layer).

Case-table rows R1–R20 (`case-table.md`), transcribed from AC16-amended,
FR31/FR32 and the Benchmark Metric Contract (2026-08-10, amended 2026-10-22)
of specs/autonomous-context-continuity.md.

RED MECHANISM
=============

The module-level imports of ``evals.continuity.models``,
``evals.continuity.scenarios`` and ``evals.continuity.runner`` raise
``ModuleNotFoundError`` until the implementation exists. There is no shape
discovery, no candidate probing, and no attribute heuristics anywhere in
this file: the cage pins one named interface and every test exercises it
directly.

THE FROZEN INTERFACE (implemented by C, ratified by B)
======================================================

``evals.continuity.models``
    ``Block(text: str, policy_tag: str | None = None,
    distractor_tag: str | None = None, layer: str | None = None)``
        Tagged content block. ``policy_tag`` truthy marks a policy-tagged
        block (must never reach a restricted agent); ``distractor_tag``
        truthy marks a distractor-tagged block.
    ``ScenarioTurn(gold_answers: Sequence[str],
    current_values: Sequence[str],
    superseded_values: Sequence[str] = (),
    blocks: Sequence[Block] = (),
    delivered: Sequence[str] | None = None,
    meta: dict | None = None)``
        One declared turn of a scenario. ``delivered`` (optional) lists the
        value strings actually delivered in that turn's background; when it
        is None, delivery presence is judged against the turn's declared
        values (current + superseded).
    ``Scenario(id: str, turns: Sequence[ScenarioTurn],
    token_budget: int, prompt_budget: int, agent_policy: str,
    runner_mode: str)``
        ``agent_policy`` is ``"restricted"`` or ``"unrestricted"``;
        ``runner_mode`` is ``"full_history"`` or ``"degraded_service"``.

``evals.continuity.scenarios``
    ``CORPUS: dict[str, Scenario]`` keyed by scenario id, holding the ten
    FR31 scenarios (fixed inputs, fixed budgets) plus the composition
    fixture described below.
    ``gold_turns(scenario) -> list[ScenarioTurn]``
        The retrieval turns of *scenario* (turns declaring gold answers).

``evals.continuity.runner``
    ``CompiledRequest(system: Sequence[Entry], messages: Sequence[Message])``
        The compiled-request type; ``system`` / each message's ``content``
        hold provenance ``sr2.pipeline.provenance.Entry`` items (entry
        ``content`` is an ``sr2.models.TextBlock``, entry ``layer`` drives
        bucket attribution).
    ``request_text(compiled_request) -> str``
        The full delivered text (system + messages, entry contents).
    ``extraction_double(compiled_request) -> str``
    ``correctness(turn, response) -> float``       1.0 / 0.0 for the turn
    ``recall(scenario, compiled_request) -> float`` fraction of the
        scenario's gold values present in the delivered request text
    ``stale_current(turn, response) -> int``       per-turn count 0 / 1
    ``policy_leak(agent_scope, compiled_request) -> int``
        ``agent_scope`` is the scenario's ``agent_policy`` string
        (``"restricted"`` or ``"unrestricted"``).
    ``irrelevant_context(compiled_request) -> float`` distractor share of
        delivered background tokens
    ``hallucination(compiled_request, response) -> int``
        0 when grounded; raises AssertionError naming the ungrounded value
        when the response contains substantive tokens absent from the
        delivered request (nonzero surfaces as harness failure).
    ``token_buckets(compiled_request) -> (prompt_tokens, background_tokens)``
    ``cost(prompt_tokens, background_tokens, pricing) -> float``
    ``latency_stats(runs: Sequence[Sequence[float]]) -> dict`` with keys
        ``median`` and ``p95``; ``runs[0]`` is the discarded warm-up.
    ``load_pricing() -> dict``   the parsed ``evals/continuity/pricing.json``
    ``run_id() -> str``          ``<corpus-git-sha>+<sr2-git-sha>``
    ``run_baseline(output_root: Path | None = None) -> Path``
        Executes every corpus scenario and writes
        ``<root>/<run_id()>/observations.jsonl`` + ``metrics.json``
        (default root: ``evals/continuity/results``). Returns the run dir.

BINDING COMPOSITION FIXTURE (the full-run numbers rows R16/R17 assert)
======================================================================

``CORPUS`` additionally holds one scenario whose id normalizes
(lowercase, non-alphanumerics stripped) to contain ``"cagecomposition"``
(suggested id ``"cage-composition"``) with exactly this content:

* ``agent_policy == "restricted"``, ``runner_mode == "full_history"``,
  ``token_budget == 4000``, ``prompt_budget == 1000``.
* Exactly two retrieval turns (``gold_turns`` length 2) carrying the
  identical declaration:
  - superseded string ``S = "backend: SQLite"``
  - current string    ``C = "backend: PostgreSQL"``
  - gold answers      ``(C,)``
  - blocks (all ``layer="continuity_replay"``):
    S; C; ``P = "favorite food: vanilla ice cream"`` (truthy ``policy_tag``);
    ``D = "alpha beta gamma delta epsilon zeta"`` (truthy ``distractor_tag``)
  - turn 1 declares ``delivered=None`` (both S and C count as delivered);
    turn 2 declares ``delivered=(C,)`` (S withheld).
* Runner execution plan for this scenario — delivered background blocks,
  per run 1–5 (run 1 is the discarded latency warm-up):
  run 1: ``[C, P, D]``; runs 2–5: ``[S, C, D]``.
* Pinned raw outcomes published in ``metrics.json`` (counts over all corpus
  scenario executions; the composition fixture contributes the nonzero
  stale-current and policy-leak events, and every double answer states the
  current gold value):
  ``correctness == 1.0``, ``stale_current == 1``, ``policy_leak == 1``,
  ``hallucination == 0``, ``cost > 0``.

Token buckets (Metric Contract): entries whose provenance ``Entry.layer``
marks the current turn's user input (``"input"``) or the system scaffold
(``"system_scaffold"``, ``"static"``) fall in the **prompt** bucket;
restored continuity context (layers such as ``"continuity_replay"``,
``"session"``) falls in the **background** bucket; every item falls in
exactly one bucket. Counts use
``sr2.tokenization.counting.count_tokens`` (tiktoken cl100k_base).

FREE vs FIXED
=============

Fixed by the Metric Contract / case rows: the names above, the run-id
format, the results paths, the pricing.json fields, the nine metric
aggregates, the observation fields, and every numeric outcome asserted
below. Free: everything else — class shapes, extra fields/exports, the
double's absence-statement wording, per-scenario layout inside the latency
aggregate beyond the keys ``median``/``p95``, and the turn content of the
nine FR31 scenarios other than via the R1/R6 declaration shape.

OBSERVATIONS / METRICS SCHEMA (rows R16, R17)
=============================================

``observations.jsonl`` — one JSON object per executed turn with at least:
``scenario_id`` (str), ``run_index`` (int, 1-based), ``turn_index`` (int,
unique within a scenario run), ``prompt_tokens`` (int > 0),
``background_tokens`` (int), ``response`` (str, non-empty), ``flags``
(object with at least ``correct``, ``stale_current``, ``policy_leak`` —
each numeric or boolean).

``metrics.json`` — JSON object with exactly the nine top-level aggregates:
``correctness``, ``recall``, ``stale_current``, ``policy_leak``,
``irrelevant_context``, ``hallucination``, ``prompt_background_tokens``
(object with ``prompt`` and ``background`` ints), ``cost`` (number),
``latency`` (object with ``median`` and ``p95`` numbers).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.continuity import models as cmodels
from evals.continuity import runner as crunner
from evals.continuity import scenarios as cscenarios

from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import Entry, EntryOrigin
from sr2.tokenization.counting import count_tokens

REPO_ROOT = Path(__file__).resolve().parents[2]
PRICING_PATH = REPO_ROOT / "evals" / "continuity" / "pricing.json"

# ---------------------------------------------------------------------------
# Binding vocabulary (R1, R17)
# ---------------------------------------------------------------------------

FR31_IDS = [
    "project-resumption-intervening-live-changes",
    "long-topic-switching-and-return",
    "personal-preference-across-projects",
    "per-agent-policy-isolation",
    "correction-supersession",
    "cross-project-dependency",
    "relevant-episodic-recovery",
    "bounded-context-endurance-500-turn",
    "restart",
    "continuity-service-failure",
]
FR31_KEYS = [re.sub(r"[^a-z0-9]", "", k.lower()) for k in FR31_IDS]

NINE_METRICS = [
    "correctness",
    "recall",
    "stale_current",
    "policy_leak",
    "irrelevant_context",
    "hallucination",
    "prompt_background_tokens",
    "cost",
    "latency",
]

PRICING_FIELDS = {
    "model",
    "usd_per_mtok_input",
    "usd_per_mtok_output",
    "source_url",
    "fetched_date",
}

# Composition fixture strings (module docstring).
SUPERSEDED = "backend: SQLite"
CURRENT = "backend: PostgreSQL"
POLICY_SECRET = "favorite food: vanilla ice cream"
DISTRACTOR = "alpha beta gamma delta epsilon zeta"
ABSENT_KEY_QUESTION = "Which database backend is selected?"

#: Id of the composition scenario the runner executes (docstring above).
COMPOSITION_ID = "cage-composition"


# ---------------------------------------------------------------------------
# Direct helpers — no discovery, no probing
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _entry(text: str, layer: str, **meta) -> Entry:
    """Real provenance Entry carrying a real ``layer`` value (R13)."""
    return Entry(
        id=f"cage-{layer}-{abs(hash((text, layer, tuple(sorted(meta))))) % 10**13:013d}",
        content=TextBlock(text=text),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="continuity_baseline_cage"),
        layer=layer,
        session_id="sr2-91-cage",
        created_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
        meta=dict(meta),
    )


def _cr(system: str, user: str, background: list[tuple[str, str]]):
    """Compiled request: ``(text, layer)`` background pairs (R8/R13 tests)."""
    bg_entries = [_entry(t, layer) for (t, layer) in background]
    return crunner.CompiledRequest(
        system=[_entry(system, "system_scaffold")],
        messages=[
            Message(role="user", content=[_entry(user, "input")]),
            Message(role="user", content=bg_entries),
        ],
    )


def _cr_entries(system: str, user: str, entries: list[Entry]):
    """Compiled request from pre-built entries (tagged-entry rows R10/R11)."""
    return crunner.CompiledRequest(
        system=[_entry(system, "system_scaffold")],
        messages=[
            Message(role="user", content=[_entry(user, "input")]),
            Message(role="user", content=entries),
        ],
    )


def _text(request) -> str:
    return crunner.request_text(request)


def _turn(**kw) -> "cmodels.ScenarioTurn":
    return cmodels.ScenarioTurn(**kw)


def _scenario(sid: str, turns: list, **kw) -> "cmodels.Scenario":
    params = dict(
        id=sid,
        turns=turns,
        token_budget=4000,
        prompt_budget=1000,
        agent_policy="restricted",
        runner_mode="full_history",
    )
    params.update(kw)
    return cmodels.Scenario(**params)


def _fixture_turn(turn_no: int) -> "cmodels.ScenarioTurn":
    """Fixture turn mirroring the binding composition declaration.

    Turn 1 declares delivered=None (S and C both delivered); turn 2
    declares delivered=(C,) (S withheld) — so exactly one stale-current
    event is possible across the fixture's executions.
    """
    blocks = [
        cmodels.Block(text=SUPERSEDED, layer="continuity_replay"),
        cmodels.Block(text=CURRENT, layer="continuity_replay"),
        cmodels.Block(text=POLICY_SECRET, policy_tag="personal-leisure",
                      layer="continuity_replay"),
        cmodels.Block(text=DISTRACTOR, distractor_tag="noise",
                      layer="continuity_replay"),
    ]
    return _turn(
        gold_answers=(CURRENT,),
        current_values=(CURRENT,),
        superseded_values=(SUPERSEDED,),
        blocks=blocks,
        delivered=None if turn_no == 1 else (CURRENT,),
        meta={"fixture_turn": turn_no},
    )


def _fixture_scenario() -> "cmodels.Scenario":
    """Local mirror of the binding composition scenario (R7/R9 scoring)."""
    return _scenario("cage-fixture", [_fixture_turn(1), _fixture_turn(2)])


def _git_rev(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT,
                          capture_output=True, text=True)


def _tree_hash(*dirs: Path) -> str:
    h = hashlib.sha256()
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(REPO_ROOT)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def _run_under_network_block(tmp_root: Path) -> Path:
    """``run_baseline`` with every outbound socket blocked; returns run dir."""
    real = (socket.socket, socket.create_connection, socket.getaddrinfo)

    class _NoNet(real[0]):
        def __init__(self, *a, **k):
            raise OSError("network access disabled for continuity baseline run")

    def _guard(*a, **k):
        raise OSError("network access disabled for continuity baseline run")

    socket.socket = _NoNet  # type: ignore[misc]
    socket.create_connection = _guard  # type: ignore[assignment]
    socket.getaddrinfo = _guard  # type: ignore[assignment]
    try:
        with pytest.raises(OSError):
            # Guard sanity while armed: outbound connects really fail.
            socket.create_connection(("api.anthropic.com", 443), timeout=1)
        out = crunner.run_baseline(output_root=tmp_root)
    finally:
        (socket.socket, socket.create_connection, socket.getaddrinfo) = real
    return Path(out)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ===========================================================================
# R1 — corpus enumeration: the ten FR31 scenarios, addressable by id, with
#      fixed inputs and fixed budgets.
# ===========================================================================


def test_r1_corpus_covers_all_fr31_scenarios_addressable_with_fixed_inputs_and_budgets():
    corpus = cscenarios.CORPUS
    assert isinstance(corpus, dict) and corpus, "R1: CORPUS must be a non-empty dict"
    norm_ids = {_norm(k): k for k in corpus}
    for key, raw in zip(FR31_KEYS, FR31_IDS):
        matches = [k for k in norm_ids if key in k]
        assert matches, (
            f"R1: FR31 scenario {raw!r} missing from CORPUS (ids: {sorted(corpus)})"
        )
    for key in FR31_KEYS:
        sid = norm_ids[key]
        scenario = corpus[sid]
        assert isinstance(scenario, cmodels.Scenario), (
            f"R1: CORPUS[{sid!r}] must be a models.Scenario"
        )
        assert key in _norm(scenario.id), (
            f"R1: scenario keyed {sid!r} declares mismatched id {scenario.id!r}"
        )
        assert len(scenario.turns) >= 1, f"R1: {sid!r} declares no fixed inputs (turns)"
        for attr in ("token_budget", "prompt_budget"):
            budget = getattr(scenario, attr)
            assert isinstance(budget, int) and budget > 0, (
                f"R1: {sid!r} fixed budget {attr} must be a positive int, got {budget!r}"
            )


# ===========================================================================
# R2 — results directory: <root>/<run-id>/ with
#      <run-id> == <corpus-git-sha>+<sr2-git-sha>; default root
#      evals/continuity/results.
# ===========================================================================


def test_r2_result_directory_run_id_is_corpus_sha_plus_sr2_sha(tmp_path):
    run_dir = Path(crunner.run_baseline(output_root=tmp_path))
    assert run_dir.is_dir()
    assert (run_dir / "observations.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    run_id = run_dir.name
    assert run_id == crunner.run_id(), "R2: directory name must equal run_id()"
    parts = run_id.split("+")
    assert len(parts) == 2, f"R2: run-id {run_id!r} is not '<corpus-sha>+<sr2-sha>'"
    corpus_part, sr2_part = parts
    assert re.fullmatch(r"[0-9a-f]{16,64}", corpus_part), (
        f"R2: corpus part {corpus_part!r} is not a hex sha"
    )
    assert re.fullmatch(r"[0-9a-f]{16,64}", sr2_part), (
        f"R2: sr2 part {sr2_part!r} is not a hex sha"
    )
    assert corpus_part != sr2_part, "R2: corpus and sr2 sha parts must differ"
    # Both parts derive from this checkout: HEAD, or the SHA-256 tree hash of
    # the corpus dir / src/sr2 (a dirty worktree legitimately hashes away
    # from HEAD).
    proc = _git_rev(["rev-parse", "HEAD"])
    assert proc.returncode == 0, "R2: worktree must be a git repository"
    head = proc.stdout.strip()
    allowed = {head, _tree_hash(REPO_ROOT / "evals" / "continuity"),
               _tree_hash(REPO_ROOT / "src" / "sr2")}
    assert corpus_part in allowed and sr2_part in allowed, (
        f"R2: run-id {run_id!r} parts match neither HEAD nor the corpus/src tree hashes"
    )
    # Default root is the pinned path.
    assert (REPO_ROOT / "evals" / "continuity" / "results").is_dir(), (
        "R2: a default run must publish under evals/continuity/results/"
    )


# ===========================================================================
# R3 — double determinism + strict grounding in delivered request content.
# ===========================================================================


def test_r3_double_is_deterministic_and_grounds_only_request_content():
    fixture = _fixture_scenario()
    turn1 = cscenarios.gold_turns(fixture)[0]
    request = _cr("You are the baseline agent.", ABSENT_KEY_QUESTION,
                  [(CURRENT, "continuity_replay")])
    r1 = crunner.extraction_double(request)
    r2 = crunner.extraction_double(request)
    assert r1 == r2, "R3: identical compiled requests must produce identical responses"
    assert CURRENT in r1, (
        f"R3: double must state the gold value present in the request; got {r1!r}"
    )
    assert crunner.correctness(turn1, r1) == 1.0, (
        "R3: the grounded statement of the current gold value must score correctness 1.0"
    )
    # Nothing ungrounded: every substantive token of the response appears in
    # the delivered request text.
    delivered = _text(request)
    for token in re.findall(r"[A-Za-z0-9_/=\-.]{6,}", r1):
        core = re.sub(r"^[=/\-.]+|[=/\-.]+$", "", token)
        if core:
            assert core in delivered, (
                f"R3: response token {core!r} is not present in the delivered request"
            )
    # Also on a conflicting delivery (both values present): response stays
    # inside the delivered text.
    conflicting = _cr("sys", ABSENT_KEY_QUESTION,
                      [(SUPERSEDED, "continuity_replay"),
                       (CURRENT, "continuity_replay")])
    r3 = crunner.extraction_double(conflicting)
    ctext = _text(conflicting)
    for token in re.findall(r"[A-Za-z0-9_/=\-.]{6,}", r3):
        core = re.sub(r"^[=/\-.]+|[=/\-.]+$", "", token)
        if core:
            assert core in ctext, (
                f"R3: conflicting-context response token {core!r} not in delivered request"
            )


# ===========================================================================
# R4 — conflicting values for one gold key: most-recently-sourced wins.
# ===========================================================================


def test_r4_double_reports_most_recently_sourced_value_on_conflict():
    # Observation-ordered sourcing: entry order = sourcing order, newest last.
    old_first = _cr("sys", ABSENT_KEY_QUESTION,
                    [(SUPERSEDED, "continuity_replay"),
                     (CURRENT, "continuity_replay")])
    resp = crunner.extraction_double(old_first)
    assert CURRENT in resp and SUPERSEDED not in resp, (
        f"R4: expected the most-recently-sourced value {CURRENT!r}; got {resp!r}"
    )
    # Flip delivery order: the superseded value becomes the newest source.
    flipped = _cr("sys", ABSENT_KEY_QUESTION,
                  [(CURRENT, "continuity_replay"),
                   (SUPERSEDED, "continuity_replay")])
    resp2 = crunner.extraction_double(flipped)
    assert SUPERSEDED in resp2 and CURRENT not in resp2, (
        f"R4: after flipping sourcing order expected {SUPERSEDED!r}; got {resp2!r}"
    )
    # Sourcing recency beats arrival order when meta declares it: the entry
    # observed earlier lands last yet must not win.
    recency = _cr_entries("sys", ABSENT_KEY_QUESTION, [
        _entry(CURRENT, "continuity_replay", observed_at="2026-06-01T00:00:00Z"),
        _entry(SUPERSEDED, "continuity_replay", observed_at="2026-01-01T00:00:00Z"),
    ])
    resp3 = crunner.extraction_double(recency)
    assert CURRENT in resp3 and SUPERSEDED not in resp3, (
        f"R4: declared sourcing recency (meta observed_at) must beat arrival order; got {resp3!r}"
    )


# ===========================================================================
# R5 — absent gold key: double states absence, correctness 0, recall miss.
# ===========================================================================


def test_r5_absent_gold_key_asserts_absence_scores_incorrect_and_recall_miss():
    fixture = _fixture_scenario()
    turn1 = cscenarios.gold_turns(fixture)[0]
    request = _cr("sys", ABSENT_KEY_QUESTION,
                  [("unrelated restored note", "continuity_replay")])
    resp = crunner.extraction_double(request)
    assert SUPERSEDED not in resp and CURRENT not in resp, (
        f"R5: gold key absent — the double must not state any gold value; got {resp!r}"
    )
    assert resp.strip() != "", "R5: the double must still answer (state absence)"
    assert crunner.correctness(turn1, resp) == 0.0, (
        f"R5: answering an absent key must score correctness 0.0; response {resp!r}"
    )
    assert crunner.hallucination(request, resp) == 0, (
        "R5: an absence statement is grounded — hallucination stays 0"
    )
    recall = crunner.recall(fixture, request)
    assert recall == 0.0, (
        f"R5: no gold values in the delivered request must be a recall miss (0.0), got {recall!r}"
    )


# ===========================================================================
# R6 — per-turn gold declarations: exact answers, current vs superseded,
#      policy- and distractor-tagged blocks, for every corpus scenario and
#      for the composition fixture (two turns, equal declarations).
# ===========================================================================


def test_r6_scenario_files_declare_per_turn_gold_current_superseded_policy_distractor():
    for sid, scenario in cscenarios.CORPUS.items():
        turns = cscenarios.gold_turns(scenario)
        assert isinstance(turns, list) and turns, (
            f"R6: scenario {sid!r} exposes no retrieval turns via gold_turns()"
        )
        for t in turns:
            assert isinstance(t, cmodels.ScenarioTurn)
            gold = list(t.gold_answers)
            assert gold and all(isinstance(g, str) and g for g in gold), (
                f"R6: scenario {sid!r} has a retrieval turn without exact expected answer strings"
            )
            cur = list(t.current_values)
            sup = list(t.superseded_values)
            assert cur and all(isinstance(v, str) and v for v in cur), (
                f"R6: scenario {sid!r} turn declares no current values"
            )
            assert sup and all(isinstance(v, str) and v for v in sup), (
                f"R6: scenario {sid!r} turn declares no superseded values"
            )
            for b in t.blocks:
                assert isinstance(b, cmodels.Block) and isinstance(b.text, str), (
                    f"R6: scenario {sid!r} turn has a non-Block or textless block"
                )
            assert any(b.policy_tag for b in t.blocks), (
                f"R6: scenario {sid!r} declares no policy-tagged blocks"
            )
            assert any(b.distractor_tag for b in t.blocks), (
                f"R6: scenario {sid!r} declares no distractor-tagged blocks"
            )
            universe = {b.text for b in t.blocks} | set(gold)
            for v in cur + sup:
                assert v in universe, (
                    f"R6: scenario {sid!r} declares value {v!r} that is neither a block string nor a gold answer"
                )
    # Composition fixture: two retrieval turns whose declarations are
    # exactly equal (the row's "delivered twice with equal declarations").
    comp = _fixture_scenario()
    gt = cscenarios.gold_turns(comp)
    assert len(gt) == 2, "R6: composition fixture must expose exactly two retrieval turns"
    assert tuple(gt[0].gold_answers) == tuple(gt[1].gold_answers)
    assert tuple(gt[0].current_values) == tuple(gt[1].current_values)
    assert tuple(gt[0].superseded_values) == tuple(gt[1].superseded_values)
    assert [b.text for b in gt[0].blocks] == [b.text for b in gt[1].blocks]


# ===========================================================================
# R7 — correctness: answered-equals-current-gold, exact numeric score.
# ===========================================================================


def test_r7_correctness_exact_numeric_answered_equals_current_gold():
    turn = cscenarios.gold_turns(_fixture_scenario())[0]
    assert crunner.correctness(turn, f"Selected value: {CURRENT}") == 1.0, (
        "R7: answered-equals-current-gold must score exactly 1.0"
    )
    assert crunner.correctness(turn, f"Selected value: {SUPERSEDED}") == 0.0, (
        "R7: answered-equals-superseded must score exactly 0.0"
    )
    assert crunner.correctness(turn, "") == 0.0, "R7: empty response scores 0.0"
    # Variant with multiple gold answers: every one must be stated.
    multi = _turn(
        gold_answers=("alpha-gold", "bravo-gold"),
        current_values=("alpha-gold", "bravo-gold"),
        superseded_values=("zeta-old",),
    )
    assert crunner.correctness(multi, "alpha-gold and bravo-gold") == 1.0
    assert crunner.correctness(multi, "alpha-gold only") == 0.0


# ===========================================================================
# R8 — recall: fraction of the scenario's gold values in the delivered
#      request; exact 0.5 and 1.0.
# ===========================================================================


def test_r8_recall_fraction_of_gold_values_in_delivered_request():
    values = [f"{c}-gold-value-{i}" for i, c in enumerate("abcd")]
    scenario = _scenario(
        "cage-recall",
        [_turn(gold_answers=tuple(values), current_values=tuple(values))],
    )
    partial = _cr("sys", "recall probe",
                  [(f"fact a={values[0]}", "continuity_replay"),
                   (f"fact b={values[1]}", "continuity_replay"),
                   ("unrelated filler", "continuity_replay")])
    assert crunner.recall(scenario, partial) == 0.5, (
        "R8: 2 of 4 gold values delivered must score recall exactly 0.5"
    )
    full = _cr("sys", "recall probe",
               [(f"fact {c}={v}", "continuity_replay") for c, v in zip("abcd", values)])
    assert crunner.recall(scenario, full) == 1.0, (
        "R8: all four gold values delivered must score recall exactly 1.0"
    )


# ===========================================================================
# R9 — stale-current: counts exactly the turns stating a superseded value
#      while it was present in context; three exact cases.
# ===========================================================================


def test_r9_stale_current_counts_only_statement_with_value_present():
    gt = cscenarios.gold_turns(_fixture_scenario())
    turn_present, turn_withheld = gt[0], gt[1]
    assert tuple(turn_present.superseded_values) == (SUPERSEDED,)
    assert tuple(turn_withheld.superseded_values) == (SUPERSEDED,)
    assert turn_withheld.delivered is not None and SUPERSEDED not in tuple(turn_withheld.delivered)
    stated = f"Selected value: {SUPERSEDED}"
    current_answer = f"Selected value: {CURRENT}"
    # (a) superseded value stated while present in context -> 1
    assert crunner.stale_current(turn_present, stated) == 1, (
        "R9: stating a superseded value that was present in context must count 1"
    )
    # (b) superseded value stated while absent from context -> 0
    assert crunner.stale_current(turn_withheld, stated) == 0, (
        "R9: stating a superseded value NOT present in context must not count"
    )
    # (c) current value stated with the superseded value present -> 0
    assert crunner.stale_current(turn_present, current_answer) == 0, (
        "R9: stating the current value must never count as stale"
    )
    # The turn itself must state something — an empty response counts 0.
    assert crunner.stale_current(turn_present, "") == 0


# ===========================================================================
# R10 — policy-leak: count of policy-tagged strings in a restricted agent's
#       delivered background; unrestricted / withheld contribute zero.
# ===========================================================================


def test_r10_policy_leak_counts_tagged_strings_in_restricted_background_only():
    secret_a = "leakzorch-31337"
    secret_b = "penguin-sundae"
    # Restricted agent, one tagged string delivered -> 1
    leaked = _cr_entries("sys", "probe", [
        _entry(f"favorite food={secret_a}", "continuity_replay", policy_tag="personal"),
        _entry("normal restored context", "continuity_replay"),
    ])
    assert crunner.policy_leak("restricted", leaked) == 1, (
        "R10: one policy-tagged string in a restricted delivered background must count 1"
    )
    # Same delivery to an unrestricted agent -> 0
    assert crunner.policy_leak("unrestricted", leaked) == 0, (
        "R10: unrestricted agent must contribute zero"
    )
    # Restricted but withheld (no tagged string delivered) -> 0
    withheld = _cr_entries("sys", "probe", [
        _entry("normal restored context", "continuity_replay"),
    ])
    assert crunner.policy_leak("restricted", withheld) == 0, (
        "R10: withheld tagged string must contribute zero"
    )
    # Restricted, two tagged strings -> 2
    two = _cr_entries("sys", "probe", [
        _entry(f"favorite food={secret_a}", "continuity_replay", policy_tag="personal"),
        _entry(f"dream={secret_b}", "continuity_replay", policy_tag="personal"),
    ])
    assert crunner.policy_leak("restricted", two) == 2, (
        "R10: two policy-tagged strings in a restricted background must count 2"
    )


# ===========================================================================
# R11 — irrelevant-context: distractor-tagged token share of delivered
#       background; exact fraction via cl100k_base counts.
# ===========================================================================


def test_r11_irrelevant_context_exact_distractor_token_share_of_background():
    plain_1 = "plain restored context one"
    plain_2 = "plain restored context two"
    request = _cr_entries("sys", "probe", [
        _entry(DISTRACTOR, "continuity_replay", distractor_tag="noise"),
        _entry(plain_1, "continuity_replay"),
        _entry(plain_2, "continuity_replay"),
    ])
    expected = count_tokens(DISTRACTOR) / (
        count_tokens(DISTRACTOR) + count_tokens(plain_1) + count_tokens(plain_2)
    )
    got = crunner.irrelevant_context(request)
    assert abs(got - expected) < 1e-9, (
        f"R11: irrelevant-context must equal the distractor/background token share {expected!r}, got {got!r}"
    )
    assert 0.0 < got < 1.0
    # Zero-distractor background scores exactly 0.
    clean = _cr_entries("sys", "probe", [
        _entry(plain_1, "continuity_replay"),
        _entry(plain_2, "continuity_replay"),
    ])
    assert crunner.irrelevant_context(clean) == 0.0, (
        "R11: background with no distractor-tagged blocks must score exactly 0.0"
    )


# ===========================================================================
# R12 — hallucination: structurally 0 for the double; an injected ungrounded
#       value fails the harness loudly.
# ===========================================================================


def test_r12_hallucination_structural_zero_and_injection_fails_loudly():
    request = _cr("sys", ABSENT_KEY_QUESTION,
                  [(SUPERSEDED, "continuity_replay"), (CURRENT, "continuity_replay")])
    response = crunner.extraction_double(request)
    assert crunner.hallucination(request, response) == 0, (
        "R12: hallucination must be structurally 0 for the double"
    )
    injected = f"{CURRENT} and the owner_id is zzqgk-not-in-request-42"
    with pytest.raises(AssertionError):
        crunner.hallucination(request, injected)
    # The loud failure names the ungrounded value (harness diagnosability).
    with pytest.raises(AssertionError, match="zzqgk"):
        crunner.hallucination(request, injected)


# ===========================================================================
# R13 — token buckets: exclusive prompt/background partition, Entry.layer
#       attribution, cl100k_base counts.
# ===========================================================================


def test_r13_prompt_background_buckets_partition_items_with_layer_attribution():
    scaffold = "system scaffold text block"
    user_text = "current turn question text here"
    bg1 = "restored continuity context alpha"
    bg2 = "restored continuity context beta"
    request = _cr(scaffold, user_text,
                  [(bg1, "continuity_replay"), (bg2, "continuity_replay")])
    prompt_tokens, background_tokens = crunner.token_buckets(request)
    # Prompt = current turn's user input + system scaffold; background =
    # restored continuity context; counts are cl100k_base per item.
    assert prompt_tokens == count_tokens(scaffold) + count_tokens(user_text), (
        f"R13: prompt bucket must be cl100k_base(system scaffold + current input) "
        f"= {count_tokens(scaffold) + count_tokens(user_text)}, got {prompt_tokens}"
    )
    assert background_tokens == count_tokens(bg1) + count_tokens(bg2), (
        f"R13: background bucket must be cl100k_base(restored continuity) "
        f"= {count_tokens(bg1) + count_tokens(bg2)}, got {background_tokens}"
    )
    assert prompt_tokens > 0 and background_tokens > 0
    # Layer-driven sub-attribution inside one message: 'session'-layer entry
    # (restored) lands in background even next to current input.
    mixed = crunner.CompiledRequest(
        system=[_entry("scaffold one", "system_scaffold")],
        messages=[
            Message(role="user", content=[
                _entry("restored under session layer", "session"),
                _entry("fresh current input", "input"),
            ]),
        ],
    )
    p, b = crunner.token_buckets(mixed)
    assert p == count_tokens("scaffold one") + count_tokens("fresh current input"), (
        f"R13: layer attribution wrong: prompt bucket {p}"
    )
    assert b == count_tokens("restored under session layer"), (
        f"R13: entries on a restored-continuity layer must land in background, got {b}"
    )
    # Exclusive partition: adding one background item grows only background.
    p2, b2 = crunner.token_buckets(_cr(scaffold, user_text, [
        (bg1, "continuity_replay"),
        (bg2, "continuity_replay"),
        (bg1 + " duplicated", "continuity_replay"),
    ]))
    assert (p2, b2) == (prompt_tokens, background_tokens + count_tokens(bg1 + " duplicated")), (
        "R13: buckets must partition items exactly once — growth must land in one bucket"
    )


# ===========================================================================
# R14 — pinned pricing at evals/continuity/pricing.json;
#       cost == tokens × pinned price; zero network at run time.
# ===========================================================================


def test_r14_cost_uses_pinned_pricing_table_without_network(tmp_path):
    assert PRICING_PATH.is_file(), f"R14: pinned pricing table missing at {PRICING_PATH}"
    pricing = json.loads(PRICING_PATH.read_text())
    for field_name in PRICING_FIELDS:
        assert field_name in pricing, f"R14: pricing.json missing field {field_name!r}"
        assert pricing[field_name] not in (None, ""), (
            f"R14: pricing.json field {field_name!r} is empty"
        )
    assert float(pricing["usd_per_mtok_input"]) > 0
    assert float(pricing["usd_per_mtok_output"]) > 0
    assert crunner.load_pricing() == pricing, (
        "R14: load_pricing() must return the pinned pricing.json contents"
    )
    # cost == tokens × pinned price table.
    expected = (
        1000 * float(pricing["usd_per_mtok_input"]) / 1e6
        + 500 * float(pricing["usd_per_mtok_output"]) / 1e6
    )
    assert abs(crunner.cost(1000, 500, pricing) - expected) < 1e-12, (
        f"R14: cost(1000, 500, pricing) must equal {expected!r} (per-bucket pinned prices / 1e6)"
    )
    assert crunner.cost(0, 0, pricing) == 0.0
    # Network-disabled full run still produces a cost (no network at run time).
    run_dir = _run_under_network_block(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cost_val = metrics["cost"]
    assert isinstance(cost_val, (int, float)) and not isinstance(cost_val, bool)
    assert cost_val > 0.0, "R14: cost aggregate must be positive after a network-disabled run"


# ===========================================================================
# R15 — latency aggregation: run 1 discarded as warm-up; median and p95.
# ===========================================================================


def test_r15_latency_five_runs_first_discarded_median_and_p95():
    # runs[0] is warm-up noise; scored runs 2–5 are 0.1 … 0.5.
    stats = crunner.latency_stats(
        [[999.0, 999.0, 999.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    )
    assert stats["median"] == pytest.approx(0.3), (
        f"R15: median over scored runs [0.1..0.5] must be 0.3 — run 1 discarded as warm-up; got {stats!r}"
    )
    assert 0.4 <= stats["p95"] <= 0.5, (
        f"R15: p95 over scored runs [0.1..0.5] must fall in [0.4, 0.5]; got {stats!r}"
    )
    # Per-turn timings inside each run are aggregated.
    warm = crunner.latency_stats([[5.0], [1.0], [2.0], [3.0], [4.0]])
    assert warm["median"] == pytest.approx(3.0), (
        f"R15: median over runs 2–5 must be 3.0; got {warm!r}"
    )
    assert isinstance(warm["median"], float) and isinstance(warm["p95"], float)
    # Warm-up only -> no scored samples; the warm-up must not be scored.
    only_warm = crunner.latency_stats([[7.0, 7.0]])
    assert only_warm["median"] != 7.0, (
        "R15: with only the warm-up run there is no scored latency; must not report 7.0"
    )


# ===========================================================================
# R16 — observations.jsonl: one record per turn with run index, per-bucket
#       tokens, response text, per-metric flags.
# ===========================================================================


def test_r16_observations_one_record_per_turn_with_required_fields(tmp_path):
    run_dir = Path(crunner.run_baseline(output_root=tmp_path))
    obs = _read_jsonl(run_dir / "observations.jsonl")
    assert obs, "R16: observations.jsonl has no records"
    counts: dict[tuple[str, int], int] = {}
    for rec in obs:
        for key in ("scenario_id", "run_index", "turn_index",
                    "prompt_tokens", "background_tokens", "response", "flags"):
            assert key in rec, f"R16: record missing {key!r}: {sorted(rec)}"
        assert isinstance(rec["scenario_id"], str) and rec["scenario_id"]
        assert isinstance(rec["run_index"], int), f"R16: run_index must be int: {rec['run_index']!r}"
        assert isinstance(rec["turn_index"], int)
        assert isinstance(rec["prompt_tokens"], int) and rec["prompt_tokens"] > 0
        assert isinstance(rec["background_tokens"], int)
        assert isinstance(rec["response"], str) and rec["response"] != "", (
            "R16: record must carry the double's response text"
        )
        flags = rec["flags"]
        assert isinstance(flags, dict) and flags, "R16: record must carry per-metric flags"
        for f in ("correct", "stale_current", "policy_leak"):
            assert f in flags, f"R16: flags missing {f!r}: {sorted(flags)}"
            v = flags[f]
            assert isinstance(v, (int, float, bool)) and not isinstance(v, str), (
                f"R16: flag {f!r} must be numeric/boolean, got {v!r}"
            )
        key2 = (rec["scenario_id"], rec["run_index"])
        counts[key2] = counts.get(key2, 0) + 1
    # The composition scenario is executed 5 times, contributing its two
    # retrieval turns per run, with run indices 1..5.
    comp = {k for k in counts if _norm(COMPOSITION_ID) in _norm(k[0])}
    assert comp, (
        f"R16: composition scenario {COMPOSITION_ID!r} absent from observations "
        f"(saw {sorted({s for s, _ in counts})})"
    )
    sid = sorted(comp)[0]
    run_indices = sorted(r for (s, r) in counts if s == sid)
    assert run_indices == [1, 2, 3, 4, 5], (
        f"R16: composition scenario must observe runs 1..5; got {run_indices}"
    )
    for r in run_indices:
        assert counts[(sid, r)] == 2, (
            f"R16: run {r} must observe both composition turns; got {counts[(sid, r)]}"
        )
    seen_turns = {
        rec["turn_index"] for rec in obs
        if _norm(COMPOSITION_ID) in _norm(rec["scenario_id"]) and rec["run_index"] == 1
    }
    assert len(seen_turns) == 2, f"R16: distinct turn indices expected within a run; got {sorted(seen_turns)}"


# ===========================================================================
# R17 — metrics.json holds exactly the nine AC16 aggregates, with the
#       binding composition outcomes.
# ===========================================================================


def test_r17_metrics_json_holds_the_nine_aggregates(tmp_path):
    run_dir = Path(crunner.run_baseline(output_root=tmp_path))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert isinstance(metrics, dict)
    assert sorted(metrics) == sorted(NINE_METRICS), (
        f"R17: metrics.json must hold exactly the nine aggregates; got {sorted(metrics)}"
    )
    # Pinned raw outcomes (module docstring): the double always states the
    # current gold value; exactly one stale-current event and one
    # policy-leak event exist in the fixed corpus; grounding invariant holds.
    assert metrics["correctness"] == 1.0, (
        f"R17: correctness aggregate must be 1.0, got {metrics['correctness']!r}"
    )
    assert metrics["stale_current"] == 1, (
        "R17: exactly one turn states a superseded value present in context; "
        f"stale_current must be 1, got {metrics['stale_current']!r}"
    )
    assert metrics["policy_leak"] == 1, (
        "R17: exactly one policy-tagged string reaches a restricted background; "
        f"policy_leak must be 1, got {metrics['policy_leak']!r}"
    )
    assert metrics["hallucination"] == 0, (
        "R17: hallucination is structurally 0 under the double"
    )
    assert 0.0 <= float(metrics["recall"]) <= 1.0
    assert 0.0 <= float(metrics["irrelevant_context"]) <= 1.0
    pbt = metrics["prompt_background_tokens"]
    assert isinstance(pbt, dict) and set(pbt) >= {"prompt", "background"}, (
        f"R17: prompt_background_tokens must be an object with prompt/background; got {pbt!r}"
    )
    assert pbt["prompt"] > 0 and pbt["background"] > 0
    latency = metrics["latency"]
    assert isinstance(latency, dict) and {"median", "p95"} <= set(latency), (
        f"R17: latency aggregate must report median and p95; got {latency!r}"
    )
    assert latency["median"] > 0 and latency["p95"] >= latency["median"]
    assert isinstance(metrics["cost"], (int, float)) and not isinstance(metrics["cost"], bool)
    assert metrics["cost"] > 0.0


# ===========================================================================
# R18 — two consecutive runs on identical inputs: byte-identical metrics.
# ===========================================================================


def test_r18_two_consecutive_runs_byte_identical_metrics(tmp_path):
    dir_a = Path(crunner.run_baseline(output_root=tmp_path / "a"))
    dir_b = Path(crunner.run_baseline(output_root=tmp_path / "b"))
    assert dir_a.name == dir_b.name, (
        f"R18: identical inputs must reuse run-id {dir_a.name!r}, got {dir_b.name!r}"
    )
    bytes_a = (dir_a / "metrics.json").read_bytes()
    bytes_b = (dir_b / "metrics.json").read_bytes()
    assert bytes_a == bytes_b, "R18: two consecutive runs produced different metrics.json bytes"
    parsed = json.loads(bytes_a)
    assert isinstance(parsed, dict) and parsed, "R18: metrics.json must be valid non-empty JSON"


# ===========================================================================
# R19 — full baseline run with LLM endpoints unreachable: completes, uses
#       the deterministic double, no live call possible.
# ===========================================================================


def test_r19_baseline_completes_with_llm_endpoints_unreachable(tmp_path):
    run_dir = _run_under_network_block(tmp_path / "r1")
    obs = _read_jsonl(run_dir / "observations.jsonl")
    assert obs, "R19: run with unreachable endpoints must still publish observations"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert isinstance(metrics, dict) and metrics
    # Deterministic double under unreachable endpoints: a second
    # network-blocked run reproduces byte-identical metrics (a live-LLM run
    # could neither complete nor reproduce).
    second = _run_under_network_block(tmp_path / "r2")
    assert (second / "metrics.json").read_bytes() == (run_dir / "metrics.json").read_bytes(), (
        "R19: deterministic double must produce identical metrics across runs"
    )


# ===========================================================================
# R20 — baseline mode compiles current full history (no trimming/compaction)
#       and the product packages stay unmodified.
# ===========================================================================


def test_r20_baseline_compiles_full_history_after_n_turns_and_product_untouched():
    from conftest import MockLLM, make_minimal_config
    from sr2.orchestrator import SR2
    from sr2.pipeline.token_counting import CharacterTokenCounter

    n_prior = 3
    mock_llm = MockLLM()
    sr2 = SR2(
        pipeline_config=make_minimal_config(),
        llm={"default": mock_llm},
        token_counter=CharacterTokenCounter(),
    )
    transcript: list[Message] = []

    async def _one_turn(text: str) -> None:
        async for _ in sr2.turn([TextBlock(text=text)]):
            pass

    for i in range(n_prior):
        asyncio.run(_one_turn(f"baseline prior message {i}"))
        transcript.append(Message(role="user", content=[TextBlock(text=f"baseline prior message {i}")]))
        transcript.append(Message(role="assistant", content=[TextBlock(text="Hello world")]))
        sr2.seed_session([m.model_copy() for m in transcript])
    asyncio.run(_one_turn("final probe question"))

    assert len(mock_llm.stream_calls) == n_prior + 1, (
        f"R20: expected {n_prior + 1} LLM calls, got {len(mock_llm.stream_calls)}"
    )
    final_request = mock_llm.stream_calls[-1]
    texts = [
        block.text
        for msg in final_request.messages
        for block in msg.content
        if isinstance(getattr(block, "text", None), str)
    ]
    # Full history survives into the compiled request — no trimming and no
    # compaction in baseline mode.
    for i in range(n_prior):
        expected = f"baseline prior message {i}"
        assert any(expected in t for t in texts), (
            f"R20: prior turn {i} ({expected!r}) missing from the compiled request — "
            "baseline must replay the full history"
        )
    n_assistant = sum(1 for t in texts if t == "Hello world")
    assert n_assistant >= n_prior, (
        f"R20: expected all {n_prior} assistant outputs retained, saw {n_assistant}"
    )
    assert any("final probe question" in t for t in texts), (
        "R20: the current-turn input must be delivered"
    )
    # The baseline runner leaves the product packages untouched.
    proc = _git_rev(["status", "--porcelain", "--", "src/sr2", "src/sr2_spectre"])
    assert proc.stdout.strip() == "", (
        "R20: product packages must remain unmodified by the baseline; "
        f"git reports: {proc.stdout!r}"
    )
