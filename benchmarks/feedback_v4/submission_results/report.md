# Bracket optimization: previous v4, optimized v4 and fixed-three FIFO

Synthetic experiment, 2026-10-28. MLMaster-style group mode minimizes bracket
makespan, max(required completion timestamps) minus bracket submission time.
Averages below are across two brackets, never across member jobs. MLEvolve-style
individual mode minimizes average submission-to-result JCT. Compare policies
within a submission scenario; those averages count different result events.

## What changed

Previous v4 used a single predicted solo rate for a deferred candidate. If that
rate was optimistic, it postponed a real straggler and lengthened the bracket.
The optimized gate retains the nominal rollout and adds one explicit candidate
recovery scenario: future candidate solo rate = min(prediction, measured packed
rate). This is a hypothesis, **not a measured solo label or a confidence bound**.

For each scenario, regret is an action's predicted bracket duration minus the
better of admit/defer in that same scenario. Choose admit when its maximum regret
is at least one second below defer's maximum regret, in both measurement halves.
This is minimax regret over two discrete scenarios; it is not a guaranteed gain
in actual makespan or a worst-case bound over every possible solo speed.

The change applies only to one outstanding all-required bracket. Individual,
mixed and multi-bracket cohorts retain the nominal objective. FIFO and age
protection are unchanged. Config `feedback_latency.bracket_admission_mode` is
`recovery_regret` by default inside opt-in v4; `nominal` restores previous v4.
The scorer preserves nominal gains and records alternative forecasts and regrets.

## Submission contracts and assumptions

- Batch barrier: A1–E1 arrive together; A2–E2 arrive ten seconds after the last
  first-bracket result. Two group requests, ten jobs total.
- Sequential pipeline: A1–E1 arrive ten seconds apart. Each result enables its
  same-type successor after ten seconds; a single issuer also spaces submissions
  by ten seconds. Ten individual requests; multiple jobs may remain outstanding.

These are illustrative patterns, informed by the
[MLMaster 2 orchestrator](https://github.com/sjtu-sai-agents/EvoMaster/blob/main/playground/ml_master_2/core/playground.py)
and [MLEvolve loop](https://github.com/InternScience/MLEvolve/blob/main/run.py),
not captured framework traffic or claims about default parallelism. Both policies
receive identical jobs and feedback rules; closed-loop arrival timestamps depend
on prior completions, so they are not identical exogenous arrival traces.

Each cycle uses A/B/C/D/E budgets 600/600/200/100/200 updates; memory
4/4/2/2/2 GiB under a 12 GiB budget. Reference solo rates are 12/12/24/6/12
updates/min. Slow-C changes actual C to 2.5; stale predictions still say 24.
The original synthetic interference law is unchanged: all multi-job rates ×5/6,
A/B further multiply .8/.6/.9 with C/D/E, C caps at 2 with A/B, D/E multiply .9
when together. The scheduler never sees this hidden law or future unsubmitted jobs.

Trials last up to one productive minute and are interrupted by arrivals or
completions. All trial updates/time are counted. Rates are noiseless, half-rates
are equal, pre-add rates are observable, and retry/sample-count rules are simplified.
Startup, checkpoint, finalization and delivery cost zero. Fixed-three is the
requested comparator; v4 has no fixed concurrency cap. All runs finish all jobs.
Nearest-rank p95 with two or ten requests equals the maximum, not a reliable
population-tail estimate. These are research illustrations, not production gains.

## reference

### MLMaster 2-style batch barrier

| Policy | Average bracket makespan | p95/max bracket makespan | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 97.22 | 97.22 | 194.61 |
| V4 previous | 78.87 | 78.87 | 157.92 |
| V4 optimized bracket | 86.10 | 86.10 | 172.38 |

### MLEvolve-style sequential pipeline

| Policy | Average job JCT | p95/max job JCT | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 89.36 | 104.65 | 196.14 |
| V4 previous | 55.75 | 91.17 | 159.17 |
| V4 optimized bracket | 55.75 | 91.17 | 159.17 |


## slow C stale prediction

### MLMaster 2-style batch barrier

| Policy | Average bracket makespan | p95/max bracket makespan | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 98.70 | 98.70 | 197.57 |
| V4 previous | 148.39 | 148.39 | 296.95 |
| V4 optimized bracket | 97.75 | 97.75 | 195.67 |

### MLEvolve-style sequential pipeline

| Policy | Average job JCT | p95/max job JCT | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 83.24 | 126.94 | 226.53 |
| V4 previous | 66.37 | 172.20 | 252.70 |
| V4 optimized bracket | 66.37 | 172.20 | 252.70 |


## slow C accurate prediction

### MLMaster 2-style batch barrier

| Policy | Average bracket makespan | p95/max bracket makespan | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 98.70 | 98.70 | 197.57 |
| V4 previous | 97.75 | 97.75 | 195.67 |
| V4 optimized bracket | 97.75 | 97.75 | 195.67 |

### MLEvolve-style sequential pipeline

| Policy | Average job JCT | p95/max job JCT | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 83.24 | 126.94 | 226.53 |
| V4 previous | 67.91 | 101.33 | 195.65 |
| V4 optimized bracket | 67.91 | 101.33 | 195.65 |


All values are minutes. The uncertainty-aware gate improves the severe stale-C
case but regresses the accurate reference case. This tradeoff is explicit;
it does not establish improvement on every bracket. The pipeline schedule is
unchanged between previous and optimized v4 in every case. Strictly one-outstanding
controls also have identical schedules and no packing opportunity.

See [optimization analysis](../optimization_results/report.md) for all development
ablations, the equal-weight three-case average, and twelve prespecified sensitivity
workloads evaluated without further parameter tuning. The same synthetic rate
law limits all these results. Sustained solo calibration and nonzero phase costs
remain necessary for production evaluation.

## Reproduction

```bash
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
python -m localml_scheduler.benchmarks.feedback_v4.bracket_optimization --plot
```

Use the scheduler environment; plotting additionally requires matplotlib.
`results.json` retains every arrival, result, trial, forecast and productive segment.
Simulation units are minutes and updates/min; scorer evidence uses seconds and
updates/s. `summary.csv`, `verification.json`, and `comparison.png` contain metrics,
checks, and the combined Gantt/progress/latency figure. Model artifacts and
prediction records are not changed by this scheduler policy.
