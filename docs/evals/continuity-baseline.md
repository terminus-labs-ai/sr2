# Continuity Baseline (sr2-91 / AC16)

Deterministic benchmark of current full-history continuity behavior. The
runner executes every corpus scenario (the ten FR31 scenarios plus the
binding `cage-composition` fixture) five times each through the SR2
pipeline (`seed_session` + `turn`: no trimming, no compaction) with a
deterministic extraction double in place of a live model. No network
access at run time; pricing is pinned in
[`evals/continuity/pricing.json`](../../evals/continuity/pricing.json).

## Results are raw, with no backend selection

Every number below is the raw output of one published run — no model was
selected, weighted, or filtered to improve a metric. The single
deterministic double is the only "model" under test; the corpus, budgets,
delivery plan and pricing are fixed inputs, so a re-run on the same
committed inputs reproduces the `metrics.json` byte for byte.

## Reproduce

```
PYTHONPATH=src python -m evals.continuity.runner
```

Writes `evals/continuity/results/<run-id>/observations.jsonl` and
`metrics.json`. `<run-id>` is `<corpus-tree-sha>+<sr2-tree-sha>` —
git-derived from this checkout (a dirty worktree hashes its tree instead
of HEAD). To refresh the measured latency figures:

```
CONTINUITY_MEASURE_LATENCY=1 PYTHONPATH=src python -m evals.continuity.runner
```

## Published run

Run id:
`3a9dbbe8aa119c05afc64314c16fff96dee484c2+d57260125844fed9e30fba4ab04db5ab32b93a00abe5d382fb1e338660490deb`

Two consecutive publications produced byte-identical `metrics.json`
(sha256 `168f6ee246258d3341048d87d97a1199b8b4cecbde14450321a0b36fb1fff905`);
exactly one run directory is retained (retention constant
`_PUBLISHED_RUN_RETENTION = 1`).

### Raw per-scenario numbers (60 observations = 11 scenarios × 5 runs;
the composition fixture contributes 2 retrieval turns per run = 10)

| scenario | runs | correctness | recall | stale_current | policy_leak | prompt tok | background tok | irrelevant_ctx (mean) | hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded-context-endurance-500-turn | 5 | 1.0000 | 1.0000 | 0 | 0 | 115 | 4100 | 0.965854 | 0 |
| cage-composition | 10 | 0.9000 | 1.0000 | 1 | 1 | 190 | 260 | 1.174831* | 0 |
| continuity-service-failure | 5 | 1.0000 | 1.0000 | 0 | 0 | 120 | 315 | 0.523810 | 0 |
| correction-supersession | 5 | 1.0000 | 1.0000 | 0 | 0 | 105 | 310 | 0.516129 | 0 |
| cross-project-dependency | 5 | 1.0000 | 1.0000 | 0 | 0 | 110 | 305 | 0.508197 | 0 |
| long-topic-switching-and-return | 5 | 1.0000 | 1.0000 | 0 | 0 | 125 | 325 | 0.538462 | 0 |
| per-agent-policy-isolation | 5 | 1.0000 | 1.0000 | 0 | 0 | 110 | 305 | 0.540984 | 0 |
| personal-preference-across-projects | 5 | 1.0000 | 1.0000 | 0 | 0 | 115 | 315 | 0.555556 | 0 |
| project-resumption-intervening-live-changes | 5 | 1.0000 | 1.0000 | 0 | 0 | 95 | 300 | 0.550000 | 0 |
| relevant-episodic-recovery | 5 | 1.0000 | 1.0000 | 0 | 0 | 115 | 310 | 0.516129 | 0 |
| restart | 5 | 1.0000 | 1.0000 | 0 | 0 | 125 | 300 | 0.500000 | 0 |

\* The composition fixture's per-run `irrelevant_context` is
distractor-tokens ÷ background-tokens per turn; run 1 turn 1 delivers the
full unfiltered background (the control-plan policy-leak delivery), so the
mean over its ten observations exceeds 1.0. This is raw output of the
fixed metric contract, not an error.

### Aggregate (`metrics.json`, exact)

| metric | value |
|---|---:|
| correctness | 0.9833333333333333 = (N−1)/N, N = 60 |
| recall | 1.0 |
| stale_current | 1 |
| policy_leak | 1 |
| irrelevant_context | 0.5741624791524388 |
| hallucination | 0 |
| prompt_background_tokens.prompt | 1325 |
| prompt_background_tokens.background | 7145 |
| cost | 0.11115 USD (pinned: 3.0 $/MTok input, 15.0 $/MTok output, claude-sonnet-4-5) |
| latency.median | 0.02 s (pinned) |
| latency.p95 | 0.03 s (pinned) |

The single incorrect observation is the stale read in
`cage-composition` run 1 turn 1 — the control-plan slot where S
(`"backend: SQLite"`) travels as the most-recently-sourced value; the same
delivery carries P (the policy-tagged string), which is the single
policy-leak event. Every other observation states the current gold value.

## Latency disclosure: published figures are pinned constants

The published `latency.median` = 0.02 and `latency.p95` = 0.03 are
**pinned constants** (`_LATENCY_PINNED_MEDIAN`, `_LATENCY_P95_RATIO = 1.5`
in `evals/continuity/runner.py`), kept so that `metrics.json` is
byte-identical across runs (R18). Wall-clock latency is not
byte-reproducible, so it is excluded from the versioned results.

Measured from one `CONTINUITY_MEASURE_LATENCY=1` run of this same
checkout (scored runs 2–5, run 1 discarded as warm-up):

- measured median: **0.000297 s**
- measured p95: **0.001000 s**

The pinned values are ~67× the measured ones; they document the metric
shape, not this checkout's wall clock.
