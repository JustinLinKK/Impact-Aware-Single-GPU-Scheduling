# V4 feedback-policy experiment

Synthetic CPU simulation, 2026-10-28. No GPU measurements or predictor training.
The simulator calls `scheduler.feedback_objective.compare_feedback_admission`,
the same scorer used by the opt-in live trial controller.

A/B are initially running; C/D/E wait in FIFO order. All submission times are
zero. Required remaining updates are A=600, B=600, C=200, D=100, E=200; VRAM is
4/4/2/2/2 GiB with a 12 GiB budget. Reference solo rates are 12/12/24/6/12
updates/min. Slow-C cases change actual C to 2.5; the stale prediction stays 24.

Each multi-job rate is solo × 5/6. A/B additionally multiply by .8/.6/.9 when
C/D/E are present. C is capped at 2 updates/min with A or B; co-running D/E
each multiply by .9. These are invented, stable, noiseless rates.

Trials take one productive minute, split into two equal halves. All trial work
and time remain in the trace. Startup, checkpoint, finalization and result
delivery cost **zero in this experiment**; production benefits are unverified.
Pre-trial slopes come from already running jobs, with no new baseline cost.
No incumbent is preempted. One FIFO pass occurs per completion, with fresh
evidence after membership changes. V4 suppresses backfill once the oldest
waiting job reaches 30 minutes. This is an event-driven policy comparison,
not an emulation of the controller's sample-count or retry-clock constraints.
The fixed-three cap is solely the user-requested comparator.

Forecasts drain the current pack and then a frozen serial queue. They use only
current pre/post measurements and predicted solo rates, never hidden future
group rates. Unknown surviving subsets retain parent rates as an assumption.
Both evidence halves must improve weighted mean request latency by >=1 second.
Individual requests count five feedback events; group counts the single A–E
barrier; mixed counts AB, CD, and E once each, with equal request weights.
Both baseline schedules are evaluated under every partition for fair comparison.

With five simultaneous arrivals, nearest-rank p95 equals max JCT and makespan.
This example therefore does not independently establish a tail-latency benefit.
Values below are minutes. Every run completes all five jobs.

## reference

| Policy | Mean job JCT | p95/max JCT | Makespan | Mean mixed feedback | Trial time |
|---|---:|---:|---:|---:|---:|
| Fixed-three FIFO | 84.39 | 97.22 | 97.22 | 89.81 | 0.00 |
| V3: 50% slowdown | 70.16 | 93.17 | 93.17 | 64.32 | 5.00 |
| V4: individual | 53.54 | 78.87 | 78.87 | 57.93 | 5.00 |
| V4: group | 53.54 | 78.87 | 78.87 | 57.93 | 5.00 |
| V4: mixed | 53.54 | 78.87 | 78.87 | 57.93 | 5.00 |

| Policy | A done | B done | C done | D done | E done |
|---|---:|---:|---:|---:|---:|
| Fixed-three FIFO | 75.00 | 75.00 | 77.50 | 97.22 | 97.22 |
| V3: 50% slowdown | 77.80 | 77.80 | 80.02 | 93.17 | 22.00 |
| V4: individual | 70.79 | 70.79 | 78.87 | 23.11 | 24.11 |
| V4: group | 70.79 | 70.79 | 78.87 | 23.11 | 24.11 |
| V4: mixed | 70.79 | 70.79 | 78.87 | 23.11 | 24.11 |

## slow C stale prediction

| Policy | Mean job JCT | p95/max JCT | Makespan | Mean mixed feedback | Trial time |
|---|---:|---:|---:|---:|---:|
| Fixed-three FIFO | 88.63 | 98.70 | 98.70 | 90.31 | 0.00 |
| V3: 50% slowdown | 74.39 | 98.56 | 98.56 | 66.12 | 5.00 |
| V4: individual | 67.44 | 148.39 | 148.39 | 81.10 | 5.00 |
| V4: group | 67.44 | 148.39 | 148.39 | 81.10 | 5.00 |
| V4: mixed | 67.44 | 148.39 | 148.39 | 81.10 | 5.00 |

| Policy | A done | B done | C done | D done | E done |
|---|---:|---:|---:|---:|---:|
| Fixed-three FIFO | 75.00 | 75.00 | 98.70 | 97.22 | 97.22 |
| V3: 50% slowdown | 77.80 | 77.80 | 98.56 | 95.80 | 22.00 |
| V4: individual | 70.79 | 70.79 | 148.39 | 23.11 | 24.11 |
| V4: group | 70.79 | 70.79 | 148.39 | 23.11 | 24.11 |
| V4: mixed | 70.79 | 70.79 | 148.39 | 23.11 | 24.11 |

## slow C accurate prediction

| Policy | Mean job JCT | p95/max JCT | Makespan | Mean mixed feedback | Trial time |
|---|---:|---:|---:|---:|---:|
| Fixed-three FIFO | 88.63 | 98.70 | 98.70 | 90.31 | 0.00 |
| V3: 50% slowdown | 74.39 | 98.56 | 98.56 | 66.12 | 5.00 |
| V4: individual | 65.80 | 97.00 | 97.00 | 74.33 | 3.00 |
| V4: group | 65.80 | 97.00 | 97.00 | 74.33 | 3.00 |
| V4: mixed | 65.80 | 97.00 | 97.00 | 74.33 | 3.00 |

| Policy | A done | B done | C done | D done | E done |
|---|---:|---:|---:|---:|---:|
| Fixed-three FIFO | 75.00 | 75.00 | 98.70 | 97.22 | 97.22 |
| V3: 50% slowdown | 77.80 | 77.80 | 98.56 | 95.80 | 22.00 |
| V4: individual | 85.00 | 85.00 | 97.00 | 21.00 | 41.00 |
| V4: group | 85.00 | 85.00 | 97.00 | 21.00 | 41.00 |
| V4: mixed | 85.00 | 85.00 | 97.00 | 21.00 | 41.00 |

## Interpretation and limits

The request type changes which completions matter; it does not make the forecast
accurate. In particular the slow-C stale prediction can favor quick individual
feedback while greatly extending the final completion. Compare each v4 policy
with both baselines under the same request partition. Do not compare a mixed
request average with a five-job average as if they counted the same events.

All three v4 request partitions choose the **same schedule in these five-job
cases**. The example shows a benefit over the baselines in the reference case,
but does not show that group mode improves over individual mode. With accurate
slow-C slopes, mixed feedback is 74.33 minutes versus v3's 66.12, despite better
job mean JCT and makespan. Restricted forecasts are heuristics and can still
mis-rank future schedules. The separate two-job scorer test verifies distinct
behavior: A=100, B=1 updates, solo rates 1/s, packed rates .5/.4/s yield an
individual gain of 48.625 seconds but a group loss of .25 seconds. Individual
accepts and group rejects with the same completion forecasts.

The previous 148.39 → 97.00 minute counterexample is synthetic and prediction-
dependent. This rerun keeps all progress, tests both completion policies and
uses FIFO/age protection; a better slope still cannot identify co-location
interference without measured trials. No result is an optimality guarantee.

Next evaluation must include staggered and feedback-dependent arrivals, noisy
and drifting rates, large/short jobs, calibrated versus biased predictors,
nonzero startup/checkpoint/validation/delivery overhead and failed/cancelled
requests. Record application-consumed results as well as scheduler receipts.
Compare identical workloads/seeds against v3 and fixed-three FIFO, require all
jobs to finish, report job and request mean/p95/max, starvation, coverage,
trial cost and makespan, and assess multiple traces before deployment.

## Reproduction and artifacts

From the MLEvolve parent directory:

```bash
python -m localml_scheduler.benchmarks.feedback_v4.simulate --plot
```

Use the existing scheduler environment (including PyYAML); plotting additionally
requires matplotlib. The simulation itself uses the standard library.
`results.json` saves every segment, update counter, admission decision and both
forecast alternatives. Forecast timestamps/rates are seconds and updates/s;
simulation timestamps/rates are minutes and updates/min. `summary.csv` contains
metrics, `verification.json` records conservation and memory checks, and
`comparison.png` combines every schedule with update curves and outcome metrics.
