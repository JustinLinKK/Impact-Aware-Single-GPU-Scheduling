# MLMaster bracket makespan and MLEvolve individual job JCT

Synthetic experiment, 2026-10-28. “Average” here is the arithmetic mean: each
feedback request has weight one. Mean job JCT already weights jobs equally;
changing the word would not change the metric. The substantive change is the
unit being averaged: one batch barrier or one individual returned result.

For Q requests, average feedback = sum(C_q - a_q)/Q. For a batch, C_q is the
last required member's result; for an individual request it is that job's result.
There are two batch requests or ten individual requests in the scenarios below.
Compare policies **within** each scenario; the two averages count different events.

**MLMaster's admission objective is bracket makespan, not average member-job
time.** For bracket b submitted at A_b, M_b = max(C_i for i in bracket b) - A_b.
One outstanding group request makes the shared scorer exactly this maximum.
The admission gain is predicted defer-last-completion minus predicted
admit-last-completion; both measurement halves must pass the one-second margin.
“Average bracket makespan” in the tables averages the two bracket durations.
The next bracket is generated only after the preceding bracket returns all results.
Total workflow time is the sum of bracket makespans plus the inter-bracket think
delay. For MLEvolve's pipeline, average individual job JCT is the primary metric.
This naming clarification leaves the already-correct group scoring unchanged.

The main comparison contains only fixed-three FIFO and v4. V3 is removed from
these tables and the chart; earlier five-job results remain a historical appendix.
V4 selects group/individual completion from the request metadata; it is one engine.

## Submission contracts

- **MLMaster 2-style batch barrier:** submit A1–E1 together at time zero. After
  every result in that batch returns, wait 10 seconds, then submit A2–E2 together.
  Each batch is one all-required feedback request.
- **MLEvolve-style sequential pipeline:** submit A1–E1 one at a time, 10 seconds
  apart. Each returned result enables its same-type successor A2–E2 after a
  10-second think delay. A single issuer spaces submissions by at least 10 seconds.
  Multiple jobs can remain outstanding; each is an individual feedback request.

These are illustrative arrival models, not captured framework traffic. The
[MLMaster 2 orchestrator](https://github.com/sjtu-sai-agents/EvoMaster/blob/main/playground/ml_master_2/core/playground.py)
collects parallel improvement results before processing the direction. The
[MLEvolve loop](https://github.com/InternScience/MLEvolve/blob/main/run.py)
staggers draft dispatch and replenishes work on individual completions. “MLMaster”
does not universally imply a batch barrier, and sequential submission does not
imply a single outstanding job. We choose five initial lineages and two cycles
for this mock; these are not claims about framework defaults or a GPU job cap.

Each policy receives the same ten jobs, recipes and feedback/dependency rules.
Future release timestamps differ because completions differ: this is a closed-loop
comparison, **not an identical fixed arrival trace**. Both real end-to-end workflow
time and request latency are reported. Code generation/learning quality are not
simulated; think time is a fixed assumption. No independent Poisson arrival model
is used.

## Work and measurement assumptions

Each cycle uses the original five-job budgets (600/600/200/100/200 updates),
4/4/2/2/2 GiB memory and a 12 GiB GPU. Reference solo rates are 12/12/24/6/12
updates/min. Slow-C changes true C to 2.5; its stale predictor still says 24.
The same synthetic multi-job interference law is retained: solo ×5/6, A/B
multiply .8/.6/.9 with C/D/E, C capped at 2 beside A/B, and D/E multiply .9
when together. Both cycles finish every required update.

V4 uses the production scorer with FIFO/30-minute age protection, a one-second
average-feedback improvement margin in both measured halves, and no concurrency
cap. Trials consume up to one productive minute; arrivals or completions interrupt
them and require fresh evidence. All trial progress and time are counted. Steady
rates are noise-free, so both half-rates are identical. Pre-add rates are assumed
observable from current execution. Retry and sample-count rules are simplified.
No future unsubmitted job or hidden group rate is exposed to the scorer.

Startup, checkpoint, finalization and delivery costs are explicitly zero. These
are policy illustrations, not GPU or production results. The fixed-three cap is
the requested comparator only. Nearest-rank p95 with just two or ten requests
equals the maximum; there is too little data for a reliable tail estimate.

## reference

### MLMaster 2-style batch barrier

| Policy | Average bracket makespan | p95/max bracket makespan | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 97.22 | 97.22 | 194.61 |
| V4 request feedback | 78.87 | 78.87 | 157.92 |

### MLEvolve-style sequential pipeline

| Policy | Average job JCT | p95/max job JCT | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 89.36 | 104.65 | 196.14 |
| V4 request feedback | 55.75 | 91.17 | 159.17 |


## slow C stale prediction

### MLMaster 2-style batch barrier

| Policy | Average bracket makespan | p95/max bracket makespan | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 98.70 | 98.70 | 197.57 |
| V4 request feedback | 148.39 | 148.39 | 296.95 |

### MLEvolve-style sequential pipeline

| Policy | Average job JCT | p95/max job JCT | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 83.24 | 126.94 | 226.53 |
| V4 request feedback | 66.37 | 172.20 | 252.70 |


## slow C accurate prediction

### MLMaster 2-style batch barrier

| Policy | Average bracket makespan | p95/max bracket makespan | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 98.70 | 98.70 | 197.57 |
| V4 request feedback | 97.75 | 97.75 | 195.67 |

### MLEvolve-style sequential pipeline

| Policy | Average job JCT | p95/max job JCT | Workflow makespan |
|---|---:|---:|---:|
| Fixed-three FIFO | 83.24 | 126.94 | 226.53 |
| V4 request feedback | 67.91 | 101.33 | 195.65 |


All table values are minutes. Per-job arrivals, completions and per-request feedback are in `results.json`.

## Reading the comparison

- MLMaster 2-style batch barrier, reference: v4 average bracket makespan is 18.35 minutes lower than fixed-three FIFO.
- MLMaster 2-style batch barrier, slow C stale prediction: v4 average bracket makespan is 49.69 minutes higher than fixed-three FIFO.
- MLMaster 2-style batch barrier, slow C accurate prediction: v4 average bracket makespan is 0.95 minutes lower than fixed-three FIFO.
- MLEvolve-style sequential pipeline, reference: v4 average job JCT is 33.61 minutes lower than fixed-three FIFO.
- MLEvolve-style sequential pipeline, slow C stale prediction: v4 average job JCT is 16.87 minutes lower than fixed-three FIFO.
- MLEvolve-style sequential pipeline, slow C accurate prediction: v4 average job JCT is 15.34 minutes lower than fixed-three FIFO.

The strictly-one-outstanding control produces the same schedule under both
policies: 28.33 minutes
average feedback and 284.83
minutes workflow time, with no packing trials. This control is not the requested
pipelined scenario; it verifies the absence of concurrency opportunity.
Its lower per-request average also illustrates why averages across different
submission contracts are not directly comparable: delaying submissions can lower
observed request latency while increasing total workflow time. Keep both metrics.

A lower job average does not imply a lower batch feedback average. Predictor bias
can still cause poor choices, and a local continuation score does not guarantee
the best future closed-loop schedule. The chart shows all 12 main runs: submission
markers and Gantt segments, optimizer progress, and request/job/workflow metrics.
Require longer, noisy, overhead-inclusive paired workloads before deployment.

## Reproduction

```bash
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
```

Use the scheduler environment; plotting additionally requires matplotlib.
`results.json` retains every arrival, result, trial, forecast and productive
segment. Simulation units are minutes and updates/min; internal scorer evidence
uses seconds and updates/s. `summary.csv`, `verification.json`, and `comparison.png`
provide the compact metrics, checks and combined figure. `single_outstanding_controls`
in the JSON holds the two control traces, also shown in the chart.
