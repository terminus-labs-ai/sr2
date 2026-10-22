"""Versioned benchmark corpus for the continuity baseline (sr2-91 / AC16).

Fixed inputs and fixed budgets for the ten FR31 scenarios plus the binding
composition fixture pinned by the AC16 cage (``cage-composition``). All
content is literal and deterministic: no clocks, no randomness, no I/O.

Every retrieval turn declares its machine-readable gold per the Benchmark
Metric Contract: exact expected answer strings, current vs superseded
values, policy-tagged blocks (must never reach a restricted agent) and
distractor-tagged blocks. Superseded values carry an older declared
``observed_at`` than their current replacements — the recency inputs the
perfect-reader double resolves conflicts with (stale reads stay measurable
without changing the pinned aggregate outcomes). The degraded-mode
scenarios (``restart``, ``continuity-service-failure``) replay the full
history, matching current behavior.
"""

from __future__ import annotations

from evals.continuity.models import Block, Scenario, ScenarioTurn

_REPLAY = "continuity_replay"

#: Tag vocabularies. Tag values never reach the delivered answer text —
#: tags drive policy-leak / irrelevant-context attribution only.
POLICY_TAG = "restricted-from-agent"
DISTRACTOR_TAG = "noise"

#: Declared sourcing times (recency contract): supersession happened here.
EARLY_AT = "2026-06-01T00:00:00Z"
LATE_AT = "2026-08-01T00:00:00Z"


def _overheard(secret: str) -> Block:
    """A policy-tagged block: a personal fact that must stay withheld."""
    return Block(text=f"overheard: {secret}", policy_tag=POLICY_TAG,
                 layer=_REPLAY, observed_at=LATE_AT)


def _noise(text: str) -> Block:
    """A distractor-tagged block: irrelevant restored context."""
    return Block(text=text, distractor_tag=DISTRACTOR_TAG,
                 layer=_REPLAY, observed_at=LATE_AT)


def _old(text: str) -> Block:
    """Plain restored context, observed before the current values."""
    return Block(text=text, layer=_REPLAY, observed_at=EARLY_AT)


def _scenario(
    sid: str,
    turns: list[ScenarioTurn],
    *,
    token_budget: int = 4000,
    prompt_budget: int = 1000,
    agent_policy: str = "restricted",
    runner_mode: str = "full_history",
) -> Scenario:
    return Scenario(
        id=sid,
        turns=turns,
        token_budget=token_budget,
        prompt_budget=prompt_budget,
        agent_policy=agent_policy,
        runner_mode=runner_mode,
    )


def _turn(
    question: str,
    gold: tuple[str, ...],
    current: tuple[str, ...],
    superseded: tuple[str, ...] = (),
    *,
    extras: tuple[Block, ...] = (),
    delivered: tuple[str, ...] | None = None,
) -> ScenarioTurn:
    """One retrieval turn: its question plus its gold declaration.

    ``extras`` carries tagged restored-context blocks; the current and
    superseded value blocks are appended in observation order
    (superseded-then-current, newest source last).
    """
    blocks: list[Block] = [_old(v) for v in superseded]
    blocks += [Block(text=v, layer=_REPLAY) for v in current]
    blocks += list(extras)
    return ScenarioTurn(
        gold_answers=gold,
        current_values=current,
        superseded_values=superseded,
        blocks=blocks,
        delivered=delivered,
        meta={"question": question},
    )


# ---------------------------------------------------------------------------
# The corpus — ten FR31 scenarios + the pinned composition fixture.
# ---------------------------------------------------------------------------

_SCENARIOS: list[Scenario] = [
    # 1. Project resumption with intervening live changes.
    _scenario("project-resumption-intervening-live-changes", [
        _turn(
            "Which database backend is selected?",
            gold=("database backend: managed PostgreSQL",),
            current=("database backend: managed PostgreSQL",),
            superseded=("database backend: sqlite",),
            extras=(
                _noise("unrelated checkout flow chatter from the sprint room"),
                _overheard("favorite snack: salted licorice"),
            ),
        ),
    ]),
    # 2. Long topic switching and return.
    _scenario("long-topic-switching-and-return", [
        _turn(
            "After the topic switches, which release train are we on?",
            gold=("release train: biweekly Thursday",),
            current=("release train: biweekly Thursday",),
            superseded=("release train: monthly first Monday",),
            extras=(
                _noise("tangents covered: kitchen tiles, kayaking trips"),
                _overheard("dream: flying over a lavender field"),
            ),
        ),
    ]),
    # 3. Personal preference across projects.
    _scenario("personal-preference-across-projects", [
        _turn(
            "Which test runner preference applies to the data tool?",
            gold=("test runner preference: pytest",),
            current=("test runner preference: pytest",),
            superseded=("test runner preference: unittest",),
            extras=(
                _noise("shopping list: oat milk, mustard, flax seeds"),
                _overheard("restored hobby: watercolor painting"),
            ),
        ),
    ]),
    # 4. Per-agent policy isolation.
    _scenario("per-agent-policy-isolation", [
        _turn(
            "Which drafting voice applies to the drafting agent?",
            gold=("drafting voice: formal",),
            current=("drafting voice: formal",),
            superseded=("drafting voice: casual",),
            extras=(
                _overheard("private note for the coding agent: window seat"),
                _noise("background hum about staplers and paperclips"),
            ),
        ),
    ]),
    # 5. Correction / supersession.
    _scenario("correction-supersession", [
        _turn(
            "Which retry interval stands after the correction?",
            gold=("retry interval: 30 seconds",),
            current=("retry interval: 30 seconds",),
            superseded=("retry interval: 5 seconds",),
            extras=(
                _noise("static about warehouse shelving and pallets"),
                _overheard("favorite podcast: true crime midnight"),
            ),
        ),
    ]),
    # 6. Cross-project dependency.
    _scenario("cross-project-dependency", [
        _turn(
            "Which contract version does the relay depend on?",
            gold=("relay contract version: v3",),
            current=("relay contract version: v3",),
            superseded=("relay contract version: v2",),
            extras=(
                _noise("gossip about the office espresso machine"),
                _overheard("birthday gift idea: wool scarf"),
            ),
        ),
    ]),
    # 7. Relevant episodic recovery.
    _scenario("relevant-episodic-recovery", [
        _turn(
            "Which recovery plan did the outage episode end with?",
            gold=("recovery plan: snapshot restore",),
            current=("recovery plan: snapshot restore",),
            superseded=("recovery plan: manual replay",),
            extras=(
                _noise("episode filler about parking permits and umbrellas"),
                _overheard("comfort food after incidents: chicken soup"),
            ),
        ),
    ]),
    # 8. Bounded-context endurance (500-turn scenario, fixed inputs).
    _scenario("bounded-context-endurance-500-turn", [
        _turn(
            "After hundreds of turns, which scaling decision stands?",
            gold=("scaling decision: vertical first",),
            current=("scaling decision: vertical first",),
            superseded=("scaling decision: horizontal first",),
            extras=tuple(
                _noise(f"filler passage {i}: idle chatter segment")
                for i in range(24)
            ) + (_overheard("standing lunch order: lentil soup"),),
        ),
    ], token_budget=8000),
    # 9. Restart — degraded service mode; replay restores the full history.
    _scenario("restart", [
        _turn(
            "After the restart, which session token policy is in force?",
            gold=("session token policy: sliding expiry",),
            current=("session token policy: sliding expiry",),
            superseded=("session token policy: fixed expiry",),
            extras=(
                _noise("startup log noise about daemon flags"),
                _overheard("personal holiday plan: cabin by the lake"),
            ),
        ),
    ], runner_mode="degraded_service"),
    # 10. Continuity-service failure — degraded service mode.
    _scenario("continuity-service-failure", [
        _turn(
            "With the continuity service down, which escalation route holds?",
            gold=("escalation route: pager rotation",),
            current=("escalation route: pager rotation",),
            superseded=("escalation route: email queue",),
            extras=(
                _noise("static about garden gnomes and hose adapters"),
                _overheard("favorite board game: stone harvest"),
            ),
        ),
    ], runner_mode="degraded_service"),
    # Binding composition fixture pinned by the AC16 cage (module docstring
    # of tests/evals/test_continuity_baseline.py). Exact content — the
    # pinned strings, tags, budgets and deliveries; do not edit without a
    # test proposal.
    _scenario("cage-composition", [
        _turn(
            "Which database backend is selected?",
            gold=("backend: PostgreSQL",),
            current=("backend: PostgreSQL",),
            superseded=("backend: SQLite",),
            extras=(
                Block(text="favorite food: vanilla ice cream",
                      policy_tag=POLICY_TAG, layer=_REPLAY),
                Block(text="alpha beta gamma delta epsilon zeta",
                      distractor_tag=DISTRACTOR_TAG, layer=_REPLAY),
            ),
            delivered=None,
        ),
        _turn(
            "Which database backend is selected?",
            gold=("backend: PostgreSQL",),
            current=("backend: PostgreSQL",),
            superseded=("backend: SQLite",),
            extras=(
                Block(text="favorite food: vanilla ice cream",
                      policy_tag=POLICY_TAG, layer=_REPLAY),
                Block(text="alpha beta gamma delta epsilon zeta",
                      distractor_tag=DISTRACTOR_TAG, layer=_REPLAY),
            ),
            delivered=("backend: PostgreSQL",),
        ),
    ]),
]

#: The versioned corpus, keyed by scenario id (FR31 ids + composition).
CORPUS: dict[str, Scenario] = {s.id: s for s in _SCENARIOS}


def gold_turns(scenario: Scenario) -> list[ScenarioTurn]:
    """The retrieval turns of *scenario*: turns declaring gold answers."""
    return [t for t in scenario.turns if tuple(t.gold_answers)]
