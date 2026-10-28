# Bracket-mode optimization and its tradeoffs

Synthetic CPU experiments, 2026-10-28. Objective: minimize bracket makespan,
then report its arithmetic average across the specified brackets. The individual
MLEvolve policy and FIFO/age candidate ordering are unchanged.

The selected policy reduces the equally weighted three-case development average
from **108.34 to 93.87 minutes**
(13.36% lower). This average describes these fixtures; the scheduler is
not configured with scenario probabilities, and this is not a production estimate.

| Original mock case | Previous v4 | Optimized bracket | Change (min) |
|---|---:|---:|---:|
| reference | 78.87 | 86.10 | +7.23 |
| slow_C_accurate_prediction | 97.75 | 97.75 | +0.00 |
| slow_C_stale_prediction | 148.39 | 97.75 | -50.64 |

The accurate reference case **regresses**. The gain comes from avoiding the much
larger stale-C deferred tail, not from improving every bracket. Accurate slow-C
predictions keep the previous outcome. The optimized variant outperforms the
fixed-three comparator in these three cases; previous v4 loses to fixed-three
in the stale-C case. Full baseline/pipeline results are in the
[submission report](../submission_results/report.md).

## Selected change: two-scenario recovery regret

For each clean measurement half, retain two candidate future-solo hypotheses:

1. Nominal: use the existing predicted solo slope.
2. No recovery: use min(predicted solo slope, candidate's measured packed slope).

All other predictions, current measured pack rates, remaining work, queue order
and phase costs are identical. The packed slope remains a packed observation;
the second hypothesis is not written into model predictions or solo labels.

Let d_h(x) = predicted bracket completion under defer minus that under admit,
for hypothesis x. The two actions' maximum regrets are:

    R_admit = max(0, -d_h(nominal), -d_h(no_recovery))
    R_defer = max(0,  d_h(nominal),  d_h(no_recovery))

Admit only if R_defer - R_admit >= 1 second in **both** independent halves.
Both failures reject, disagreement/invalid evidence is inconclusive. This is
minimax regret over two discrete hypotheses, not a worst-case bound over a
continuous interval and not a guarantee that actual latency improves. A nominal
regression can be accepted to reduce exposure to a more costly deferral error.

Apply this only to one outstanding all-required bracket. Mixed, multi-bracket
and individual cohorts keep the nominal scorer. Preserve memory constraints,
incumbent continuity, productive trial progress, phase costs and retry controls.
No GPU concurrency cap, new network head, predictor update or incumbent pause
is introduced. The opt-in v4 configuration defaults to
`bracket_admission_mode: recovery_regret`; `nominal` restores the previous gate.

## Development alternatives retained for review

These seven variants were explored on the original three mock cases. They are
development data, not held-out evidence. No hidden solo rate or future group
performance was provided to a scheduler. The chosen FIFO/regret variant avoided
adding a ranking rule that did not consistently help. Productive short anchors
with measured singleton rates did not consistently improve the brackets either.

| Variant | Reference | Stale C | Accurate slow C | Three-case average |
|---|---:|---:|---:|---:|
| fifo_regret | 86.10 | 97.75 | 97.75 | 93.87 |
| longest_nominal | 78.76 | 148.99 | 97.20 | 108.32 |
| longest_regret | 85.39 | 114.88 | 97.20 | 99.16 |
| short_anchor | 81.85 | 97.20 | 98.69 | 92.58 |
| short_anchor_measured | 81.85 | 97.20 | 98.69 | 92.58 |
| short_anchor_measured_longest | 81.85 | 97.20 | 97.21 | 92.09 |
| short_all_measured | 80.40 | 99.38 | 101.71 | 93.83 |

## Prespecified sensitivity workloads

After choosing the gate, twelve additional fixtures were saved in
`bracket_validation_cases.json` before evaluation, using seed 20261028. Four have
nearby predicted/actual rates, four an optimistic C prediction, and four a
pessimistic C prediction. Budgets vary from .5–1.5 times the original and trials
last .5, 1 or 2 minutes. The gate/margin was not changed after these results.

Average bracket time: **109.94 → 104.58 minutes**.
Improved fixtures: 4/12; regressed fixtures: 5/12;
unchanged: 3/12.

| Fixture | Previous v4 | Optimized bracket | Change (min) |
|---|---:|---:|---:|
| validation_00_near_prediction | 89.98 | 107.64 | +17.65 |
| validation_01_near_prediction | 111.20 | 116.23 | +5.03 |
| validation_02_near_prediction | 105.50 | 110.36 | +4.86 |
| validation_03_near_prediction | 74.60 | 87.57 | +12.96 |
| validation_04_optimistic_C | 124.03 | 104.09 | -19.94 |
| validation_05_optimistic_C | 139.05 | 112.01 | -27.05 |
| validation_06_optimistic_C | 189.40 | 139.87 | -49.54 |
| validation_07_optimistic_C | 121.61 | 103.87 | -17.74 |
| validation_08_pessimistic_C | 74.03 | 74.03 | +0.00 |
| validation_09_pessimistic_C | 114.78 | 124.18 | +9.40 |
| validation_10_pessimistic_C | 106.06 | 106.06 | +0.00 |
| validation_11_pessimistic_C | 69.02 | 69.02 | +0.00 |

These are sensitivity checks under the **same invented interference law**, not
independent framework traces or broad generalization evidence. Completion-phase
overhead remains zero; trial time and productive interrupted-trial progress are
fully counted. Every required job completes and VRAM remains within 12 GiB.
Longer trials, noisy windows, nonzero checkpoint/delivery costs and changing data
pipelines need separate measured evaluation. Solo-rate calibration is still
valuable: the uncertainty rule manages ambiguity but cannot identify true solo
speed from a packed observation. It can lose when the nominal prediction was right.

## Artifacts and reproduction

`results.json` contains all main, control, development and sensitivity traces,
including both continuation forecasts and regret calculations. `summary.csv`
contains metrics. `all-experiments.png` combines a Gantt chart and optimizer
progress plot for every run, with bracket/JCT and workflow outcomes in each title.
The main comparison also has a more compact chart in `submission_results/`.

```bash
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
python -m localml_scheduler.benchmarks.feedback_v4.bracket_optimization --plot
```

Development ablations use the submission implementation frozen at commit
`04fedcc`; the script loads that source from local Git history. Each uses the
shared scorer plus its explicitly recorded development variation. The live
implementation and main/sensitivity runs use the same current scorer.
