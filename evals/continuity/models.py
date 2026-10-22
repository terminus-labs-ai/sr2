"""Tagged scenario data for the continuity baseline benchmark (sr2-91 / AC16).

Pure data containers — no behavior, no runner imports. Shapes are the
frozen interface of ``tests/evals/test_continuity_baseline.py``:
``Block`` / ``ScenarioTurn`` / ``Scenario``. Gold labels are fixture-driven
exact strings (Benchmark Metric Contract, "Gold labels"); scoring lives in
``evals.continuity.runner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Block:
    """One tagged content block declared by a scenario turn.

    ``policy_tag`` truthy marks a policy-tagged block (must never reach a
    restricted agent); ``distractor_tag`` truthy marks a distractor block.
    ``observed_at`` is the declared sourcing time (ISO string): the metric
    contract's recency signal — on conflicting values for one key the
    double reports the most-recently-sourced value. ``layer`` is the
    provenance ``Entry.layer`` the delivery stamps, driving prompt/
    background bucket attribution.
    """

    text: str
    policy_tag: str | None = None
    distractor_tag: str | None = None
    layer: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True)
class ScenarioTurn:
    """One declared turn of a scenario, carrying its machine-readable gold.

    ``delivered`` lists the value strings actually delivered in that turn's
    background; when ``None``, delivery presence is judged against the
    turn's declared values (current + superseded). ``scaffold_observed_at``
    optionally declares the scaffold block's sourcing time (recency
    inputs); it must not collide with any block's ``observed_at``.
    """

    gold_answers: Sequence[str]
    current_values: Sequence[str]
    superseded_values: Sequence[str] = ()
    blocks: Sequence[Block] = ()
    delivered: Sequence[str] | None = None
    meta: dict = field(default_factory=dict)
    scaffold_observed_at: str | None = None


@dataclass(frozen=True)
class Scenario:
    """A versioned benchmark scenario: fixed inputs, fixed budgets.

    ``agent_policy`` is ``"restricted"`` or ``"unrestricted"``;
    ``runner_mode`` is ``"full_history"`` or ``"degraded_service"``.
    """

    id: str
    turns: Sequence[ScenarioTurn]
    token_budget: int
    prompt_budget: int
    agent_policy: str
    runner_mode: str
