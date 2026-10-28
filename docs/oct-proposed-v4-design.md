# Derived solo-slope admission: oct_proposed_v4 design

Status: design only, 2026-10-27. Base: `oct_proposed_v3`, commit
`4a27bcff00ac6021f4605a8f5f186b2db1a07233`. No scheduler behavior, configuration,
dataset, or model artifact is changed by this document.

The proposed policy replaces v3's 50% incremental incumbent-slowdown criterion
with a measured normalized-progress criterion. It uses the existing predictor's
wall-step output to derive a nominal solo slope, without requiring a new neural
head or a mandatory exclusive profile for every arriving job. The accompanying
[labeling proposal](solo-slope-labeling.md) distinguishes this initial estimate
from future sustained, successful-update measurements.

## Objective and limits

The eventual objective is the time until every submitted job finishes, including
validation, final artifacts, checkpoints, and successful worker exit. Admission
uses a local proxy: more solo-equivalent training work completed before the next
projected update-budget completion. A positive proxy gain is not proof of lower
whole-trace makespan. Future arrivals, remaining subsets, trial costs, finalization,
and job-specific prediction errors can change the outcome.

The initial slope is an **uncalibrated prior**. Correct job ordering does not
establish correct cost ratios. Changing units from latency to its reciprocal
does not improve prediction accuracy or identify co-location interference.
Live trials remain necessary. No claim of improved GPU performance accompanies
this documentation-only branch.

## Inputs, update budgets, and predicted profile

For job i, let K_i be its validated required optimizer updates, u_i its completed
updates, and R_i = K_i - u_i its remaining updates at the decision snapshot.
Counters must retain completed work across trials, checkpoints, and resumes.
Do not reconstruct u_i from completed epochs when exact update progress exists.

Use the runner's explicit total-update budget when provided. Otherwise use
K_i = E_i S_i, where E_i is planned epochs and S_i is optimizer updates per epoch.
When both are declared, require agreement; the current runner contract rejects
inconsistent or truncated recipes rather than silently taking their minimum.
Unknown budgets, negative counters, or u_i > K_i cannot support admission.
R_i = 0 is a completion boundary requiring reconciliation, not a new trial.

For an ordinary single-GPU, fixed-size data loader, with N_i examples per epoch,
physical batch B_i, and accumulation g_i:

\[
L_i=\begin{cases}\lceil N_i/B_i\rceil&\text{retain the partial batch},\\
\lfloor N_i/B_i\rfloor&\text{drop the partial batch},\end{cases}
\qquad S_i=\lceil L_i/g_i\rceil.
\]

The S_i formula assumes an optimizer update flushes any partial accumulation at
each epoch end. Discarding that group, accumulating across epochs, custom
samplers, or an explicit step limit requires the runner's actual update count.
For N=50,000, B=64, g=4, E=10, retaining both partial groups gives L=782,
S=196, and K=1,960. Early stopping can shorten a planned budget; reconcile it
with the job's actual state. One complete optimizer update, not one microbatch
or backward pass, is the counting unit.

Let t_hat_i be the existing **raw nominal wall-step prediction in milliseconds**.
Derive

\[
\widehat s_i^0=1000/\widehat t_i
\quad[\text{nominal optimizer updates/s}].
\]

Do not use GPU-event time, per-sample time, epoch time, or a display-normalized
cost ratio in this conversion. Check both the input and reciprocal for finite,
strictly positive values. Do not clamp invalid outputs into plausible slopes.

The proposed additive field on `predict_profile_options` is shown below for an
illustrative t_hat_i of 28.097898000851274 ms; this is not a recorded model run:

```json
{
  "derived_solo_slope": {
    "value": 35.58985088385271,
    "unit": "updates_per_second",
    "source": "reciprocal_wall_step_prediction",
    "step_semantics": "nominal_optimizer_update"
  }
}
```

The enclosing profile retains `kind: predicted_workload`, `step_cost` in nominal
milliseconds, `artifact_sha256`, `hardware`, `batch_size`, `job_signature`,
`calibration: uncalibrated_prior`, and `requires_online_validation: true`.
The new field is specified here only; it is not implemented on this branch.
The six-output teacher/student tensor interface and existing fields remain intact.
Neither the estimate nor its derived ETA may be written to measured-profile
tables or passed off as a verified drain-time estimate.

All members must have valid profiles from the same artifact and trained-hardware
scale, matching their current workload, resolved batch, precision, optimizer,
accumulation semantics, and supported runtime context. Freeze the full profile
identity and slope for the trial. A changed signature, batch, artifact, hardware,
backend configuration, or membership invalidates the comparison.

If a valid compatible prediction is unavailable for any member, do not authorize
v4 packed admission for that group. Keep incumbents running and leave the
candidate eligible for a future compatible group or solo execution under the
existing memory and execution checks. Do not substitute zero work, mix measured
and uncalibrated cost scales, or silently switch to the 50% policy.

## Admission calculation

Let S be the incumbent group and j the candidate. Measure optimizer updates per
second before and during overlap. For each independent timing half h in {1,2},
use r^-_{i,h} for incumbents before joining and r^+_{i,h} for all members during
overlap. The runner's mean seconds/update can be inverted after validity checks;
equivalently, use completed updates divided by the corresponding elapsed time.

Use fresh, phase-clean windows with matching runtime identity and real temporal
overlap across members. Preserve v3's warm-up, minimum-progress, non-update-phase,
and disjoint-window checks. Pre-add windows describe the **current pack**, which
need not be a solo configuration. Reusing stale solo timings as live evidence is
not allowed.

For each half calculate:

\[
P_h^- = \sum_{i\in S} r^-_{i,h}/\widehat s_i^0,\qquad
P_h^+ = \sum_{i\in S\cup\{j\}}r^+_{i,h}/\widehat s_i^0,\qquad
g_{j,h}=P_h^+/P_h^-.
\]

P is dimensionless solo-equivalent work per wall time. Equivalently, weight each
observed update by c_i = 1/s_hat_i^0. A common rescaling of every c_i cancels from
g, whereas job-specific ratio errors do not. Raw heterogeneous update counts or
busy percentages must not be summed as substitutes for this work measure.

Use the same latest consistent post-trial R_i snapshot for both halves:

\[
H_{j,h}=\min\left(1800\text{ s},
\min_{i\in S}\frac{R_i}{r^-_{i,h}},
\min_{i\in S\cup\{j\}}\frac{R_i}{r^+_{i,h}}\right),
\qquad B_{j,h}=H_{j,h}(g_{j,h}-1).
\]

H stops at the first projected update-budget completion under either
configuration. Rates for a surviving subgroup are not assumed to be known.
B is extra progress expressed in seconds of pre-add-pack-equivalent work;
it is **not seconds of proven makespan saving**. Each valid rate and P^- must be
positive and finite. Missing, zero, stale, or invalid rates make the trial
inconclusive rather than creating an infinite horizon or a fabricated gain.

The provisional research thresholds, applied separately to both halves, are:

\[
g_{j,h}\ge1.05\quad\text{and}\quad B_{j,h}\ge5\text{ s}.
\]

Both halves passing accepts; both failing rejects; disagreement or invalid
evidence is inconclusive. Threshold equality passes. The thresholds are
experimental defaults, not calibrated confidence bounds or validated optima.
Two consistent halves are a repeatability check, not a statistical guarantee.

## Lifecycle, safety, and evidence

- Preserve v3's feasibility checks and heavy-anchor/light-joiner busy-band/FIFO
  ordering, original queue age, scheduling compatibility, memory reservations,
  live VRAM gate, and backend constraints. Busy remains an ordering hint.
- Do not impose a fixed concurrency cap on v4. The fixed-three policy below is
  the explicitly requested evaluation comparator, not a v4 setting.
- Keep incumbents executing through preparation, measurement, and rejection.
  A rejected or inconclusive newcomer checkpoints and requeues alone; all its
  completed updates remain credited. An idle GPU can dispatch a feasible solo
  job without an admission-gain calculation.
- Retain existing trial deadlines, optimizer-update limits, cooldowns, attempt
  limits, teardown checks, and natural-solo opportunity. Membership or budget
  completion during a trial requires fresh evidence for the remaining group.
  Resumption after scheduler restart must validate identity and fresh windows;
  persisted counters do not authorize reuse of stale rates.
- Record the distinct proposed policy identifier `derived_solo_slope_gain_v1`,
  frozen predicted profiles, remaining-update snapshot, before/after rates,
  both P/g/H/B calculations, thresholds, decision, and reason. Incumbent slowdown
  can remain a diagnostic, but is not an additional admission gate.
- Keep trial time already paid separate from the continuation decision: its
  productive updates are in R_i. The initial proxy does not model future
  checkpoint/startup/finalization costs. Record those costs for evaluation and
  a later net-benefit refinement rather than treating B as their compensation.

## Complete five-job synthetic example

This is an illustrative fluid calculation, not measured GPU performance or a
reproduction of v3's full state machine. It conditions on two already-running
jobs and a waiting queue, with no later arrivals. Policy candidates are tried
in FIFO order C,D,E to isolate admission from ranking.

| Job | Initial state | Remaining updates | Memory GiB | Reference solo updates/min | Relative cost c_i |
|---|---|---:|---:|---:|---:|
| A | running | 600 | 4 | 12 | 1 |
| B | running | 600 | 4 | 12 | 1 |
| C | waiting | 200 | 2 | 24 | 0.5 |
| D | waiting | 100 | 2 | 6 | 2 |
| E | waiting | 200 | 2 | 12 | 1 |

Memory budget is 12 GiB. The simulator's hidden rate law for **every** group is:

1. A singleton runs at its solo rate. For a multi-job group, multiply every solo
   rate by 5/6.
2. For each present A or B, additionally multiply by 0.8 if C is present, 0.6 if
   D is present, and 0.9 if E is present. These factors multiply.
3. If C and either A or B are present, cap C's rate at 2 updates/min.
4. When D and E are both present, additionally multiply each of their rates by 0.9.

Every co-runner is slower than its own solo rate. Policies observe rates only in
executed configurations; the complete law is for reproducibility, not an oracle
available to the scheduler. Reference costs are proportional to inverse solo
rates, with an arbitrary common factor of 12.

Initially AB runs at (10,10) updates/min. Trying C gives ABC=(8,8,2):

\[
g_C=\frac{8/12+8/12+2/24}{10/12+10/12}=0.85.
\]

The gain rule rejects C. The 50% rule accepts because incumbent update time
increases by only 10/8 - 1 = 25%. Trying D on AB gives ABD=(6,6,5), hence
g_D=1.10; the gain rule accepts, but the 50% rule rejects the 66.7% incumbent
slowdown. E can subsequently join ABD, producing ABDE=(5.4,5.4,4.5,9) within
the same 12 GiB budget, with gain 28.8/22 = 1.30909 in relative-work units.

For the zero-duration observation limit, fixed-three FIFO dispatches C first;
the 50% policy dispatches C and E; the gain policy dispatches D and E. Their
makespans are respectively 97.22222, 94.05, and 78.55556 minutes. Zero-duration
observation is an explanatory limit, not a deployable trial.

The more useful calculation includes one minute of productive overlap for each
trial, exact stable rates, and the same thresholds as above (convert minutes to
seconds). Rejected updates are retained. Scan the waiting FIFO list once between
committed-member completions; unsuccessful candidates wait for a completion.
A member completing within a trial ends that comparison; retain its progress,
reconcile membership, and scan again. There is no incumbent preemption. The
fixed-three baseline fills up to three slots without trials. Startup,
checkpoint/resume, and finalization costs are zero in this example, and fractional
updates are allowed by the fluid model. Production retains v3's actual progress
minimums and retry limits, so these numbers are not production forecasts.

The reference gain-policy timeline is fully specified below; trial rates follow
the same hidden law, including a trial that ends at E's completion.

| Minutes | Executing jobs | Event |
|---|---|---|
| 0–1 | ABC | reject C after trial |
| 1–2 | ABD | accept D |
| 2–3 | ABDE | accept E |
| 3–23.11111 | ABDE | D completes |
| 23.11111–24.11111 | ABCE | C trial ends at E's completion; fresh comparison needed |
| 24.11111–25.11111 | ABC | reject C |
| 25.11111–70.79111 | AB | A and B complete |
| 70.79111–78.87444 | C | C completes |

### Missing-solo-slope counterexample and correction

Change only C's hidden solo rate to 2.5 updates/min, keeping the same group-rate
law and initial predictor costs. Its observed rate beside A/B remains 2, so the
initial observations are identical. The gain policy still postpones C; after
70.79111 minutes it has 194 updates left, requiring 77.6 more minutes. Makespan
becomes **148.39111 minutes**.

With an accurate solo-slope prediction, C's relative cost becomes 12/2.5=4.8:

\[
g_C=\frac{8/12+8/12+2/2.5}{10/12+10/12}=1.28.
\]

C is admitted. The corrected timeline is ABC at 0–1; ABCD at 1–21 (including
D's trial at 1–2); ABCE at 21–41 (including E's trial at 21–22); ABC at 41–85;
then C alone at 85–97. D completes at 21, E at 41, A/B at 85, and C at 97.

| One-minute-trial scenario | Fixed-three FIFO | 50% slowdown | Gain policy |
|---|---:|---:|---:|
| Reference solo rates and proportional costs | 97.22222 | 93.17 | 78.87444 |
| Slow C, original cost estimate | 98.70370 | 98.56 | 148.39111 |
| Slow C, accurate solo-slope cost | 98.70370 | 98.56 | 97.00 |

All values are minutes. The correction assumes an accurate predictor; it is not
evidence that today's model attains that accuracy. Solo prediction resolves one
ambiguity, not future interference or global optimality.

## Future implementation and evaluation checks

- Confirm reciprocal units, positive finite validation, consistent model identity,
  unchanged six-column model output, and separation of predicted/measured records.
- Check invariance to a common rescaling of all predicted costs. Deliberately
  distort one job's cost to expose the dependence on cross-job ratios.
- Cover exact threshold equality, pass/pass, fail/fail, disagreement, invalid
  evidence, empty GPU, unavailable prediction, unknown/inconsistent update budget,
  R_i=0, and exhausted trial-update budgets.
- Cover non-overlapping windows, validation/checkpoint contamination, stale
  tokens, interrupted trials, restart, changed membership, and updated source,
  batch, or backend. Verify incumbents continue and rejected progress survives.
- Reproduce the synthetic cases using their full rate law and accounting for
  every required update. Treat the example's one-pass retry simplification
  separately from tests of the actual v3 lifecycle.
- Evaluate v4, unchanged v3, and the explicitly requested fixed-three FIFO
  comparator on identical arrival traces, source jobs, scientific update budgets,
  batches, seeds, hardware, and backend. Freeze policy parameters before evaluation.
  Include all trial, preparation, checkpoint, restart, and finalization costs.
- Require every job to finish and its worker to exit successfully before reporting
  completed makespan. A timed-out run is incomplete, not a shorter makespan.
  Report per-job completions, failed/inconclusive trials, slope prediction errors,
  and a combined Gantt/progress/metric figure. Do not tune on held-out evaluation.

## Source references at the v3 base

- [Predicted profile fields and nominal step cost](../prediction/ml_predictor.py),
  `predict_profile_options` and `predict_epoch_cost_options`.
- [Live add-only admission](../scheduler/add_only_trials.py),
  `_evaluate_admission_phase`; [identity/retry lifecycle](../scheduler/cold_start.py).
- [Validated remaining-update calculation](../scheduler/completion_objective.py),
  `remaining_optimizer_updates`; [budget derivation](../prediction/released_student.py),
  `steps_for_job`.
- [Current candidate ordering](../scheduler/placement_planner.py) and
  [current configuration defaults](../config/models.py).

The older [predictor policy note](predictor-indicator-policy.md) describes a
related weighted-progress proposal. It is historical design context, not proof
that the v3 default implements that proposal. The executable v3 decision is the
bounded-interference calculation in `add_only_trials.py`.
