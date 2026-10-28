# Submission-to-completion latency: oct_proposed_v4 design

Status: design only, 2026-10-27. Base: `oct_proposed_v3`, commit
`4a27bcff00ac6021f4605a8f5f186b2db1a07233`. No scheduler behavior, configuration,
dataset, or model artifact is changed by this document.

This revision makes mean job completion time (JCT), measured from submission to
successful completion, the primary objective. It supersedes the initial v4
normalized-progress admission gate as well as v3's 50% slowdown criterion.
It uses the existing predictor's wall-step output to derive a nominal solo slope,
without requiring a new neural head or a mandatory exclusive profile for every
arriving job. The accompanying
[labeling proposal](solo-slope-labeling.md) distinguishes this initial estimate
from future sustained, successful-update measurements.

## Primary metric and completion contract

For logical job i, let a_i be its original durable submission time, recorded when
the service accepts the job, before predictor/source processing, profiling,
preparation, or queueing. Let C_i be the time its required outputs and final
checkpoint are complete and the worker has exited successfully. Define

\[
F_i=C_i-a_i,\qquad
\overline F=\frac1n\sum_{i=1}^n F_i,\qquad
\min_\pi\overline F(\pi).
\]

F_i is JCT, flow time, or turnaround time. It includes queueing, preparation,
execution, trials, paused/requeued periods, data loading, validation, checkpointing,
and finalization. It does not start at first dispatch or end at the last optimizer
update. Original submission identity and time survive retries and resumes;
automatic retries of the same logical request are not new low-latency jobs.
Use the server's durable timestamps consistently; use monotonic clocks for local
interval measurement and flag invalid ordering rather than clamp negative JCT.

For an identical finite cohort, minimizing mean JCT is equivalent to minimizing
total flow time. It is also the area under the number of submitted unfinished
jobs, N_unfinished(t):

\[
\sum_i F_i=\int N_{\mathrm{unfinished}}(t)\,dt.
\]

This rewards earlier job completions, not simply more optimizer work per second.
Every job has weight one by default. Predicted workload size is not a user-priority
weight. Any future service-class weighting must be named as a different objective.

Report mean JCT as primary, plus median, nearest-rank p95, maximum JCT,
cumulative queued time, current/maximum queued age, completion coverage, failures,
and cancellations. Retain makespan max(C_i)-min(a_i), throughput, and normalized
work rate as secondary diagnostics. An empty completed cohort has no mean or p95.
Preserve existing report fields `mean_flow_time_seconds`, `median_flow_time_seconds`,
`p95_flow_time_seconds`, `total_flow_time_seconds`, and `trace_makespan_seconds`;
document the endpoint contract and add explicit coverage/max-JCT/queue-age data
where absent rather than ambiguously renaming existing metrics.
The existing trace wait statistic measures submission to first dispatch;
cumulative queued time additionally includes later paused/requeued intervals.
Queued age here is current time minus original submission for a currently queued
job, not cumulative queue residence. Keep these quantities separately named.

On an incomplete evaluation, report each unfinished job's age as a lower bound
on its JCT and label completed-only aggregates as censored diagnostics. Do not
rank policies by completed-only mean: discarding a long unfinished job can make a
bad scheduler look fast. Failed/cancelled outcomes are reported separately, not
credited as successful completions. Compare the same arrival cohort and drain
it completely for the primary experiment; never use job construction timestamps
in place of actual trace submission times.

## Limits of the objective and initial predictions

Mean JCT can improve while one user's latency becomes much worse. Tail metrics
and an explicit queued-age safeguard are therefore part of the design. No business
deadline or latency SLO has been supplied; this document does not invent one.
Production latency targets must be agreed before deployment, and any mean/tail
tradeoff must be reported against those targets.

The initial slope is an **uncalibrated prior**. Correct job ordering does not
establish correct cost ratios. Changing units from latency to its reciprocal
does not improve prediction accuracy or identify co-location interference.
Live trials remain necessary. Forecasting completion times now depends on absolute
solo-rate calibration as well as cross-job ratios; the old gain ratio's common-scale
invariance no longer establishes the validity of the new decision. The initial
derived-slope path is a research prior, not a production SLA or guaranteed latency
improvement. No claim of improved GPU performance accompanies this branch.

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

## Candidate ordering and queued-age protection

The new latency objective also changes nomination: the old heavy-anchor/light-
joiner busy ordering is no longer the primary rank. Among eligible jobs with a
valid estimate, nominate the smallest predicted remaining solo training time
R_i/s_hat_i^0, then oldest original submission, then stable job ID. Use the same
order at an empty-GPU dispatch boundary and for one-at-a-time trial candidates.
This is a non-preemptive shortest-remaining-work heuristic, not a claim of optimal
SRPT scheduling with interference. Keep hard scheduling compatibility and any
explicit deployment service-class restrictions. Busy remains a diagnostic.

Introduce a proposed `max_queued_age_seconds=1800` research default for queued
jobs, measured from original submission even after partial execution/requeue.
Once a feasible queued job crosses that age, nominate the oldest overdue job
first and stop adding ordinary backfill until it receives service. It may join
through a valid latency-benefit trial; otherwise let incumbents drain and give
it the next feasible empty-GPU solo dispatch. Do not evict incumbents or bypass
memory/compatibility checks. Retry cooldowns and budgets still apply to trials;
they must not eliminate the natural solo opportunity. Unknown-prediction jobs
remain eligible for oldest-first solo dispatch and this protection.
For this safeguard, service means committed packed admission or solo dispatch;
a rejected/inconclusive trial does not clear the overdue job's protection.

The existing `starvation_timeout_seconds` is reporting-only in v3; it is not an
existing admission override. The new queued-age behavior is explicitly proposed,
with a distinct setting. It limits repeated bypass, but cannot guarantee a hard
wait bound while a non-preemptible incumbent runs, under overload, or when no
feasible execution environment exists. Report those reasons and current ages.

## Admission calculation: compare remaining completion times

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

At the decision snapshot t, let O be all arrived, unfinished scheduler-owned
jobs, including the candidate and other waiting jobs. The two actions start from
the same post-trial counters and queue ages:

- `admit`: retain j in S union {j} using the measured overlap rates.
- `defer`: checkpoint/requeue j and continue S using its measured pre-add rates.

Do not compare an after-trial state with a before-trial state: useful trial
updates have already happened under both continuation choices. The candidate's
prior submission age is retained under either action.

For each measurement half h, forecast remaining time to successful completion
D_hat_{i,h}(a) for every i in O under action a. Score the same complete cohort:

\[
\widehat J_h(a)=\sum_{i\in O}\left[(t-a_i)+\widehat D_{i,h}(a)\right],
\qquad
\Delta_h=\frac{\widehat J_h(\mathrm{defer})-
\widehat J_h(\mathrm{admit})}{|O|}.
\]

The common elapsed ages cancel in Delta. Adding age to an otherwise identical
cost does not itself prevent starvation; the queued-age rule above supplies that
separate protection. Waiting jobs cannot be omitted: an admission can delay
their access to the GPU even when it benefits the current pack.

### Explicit initial forecast model

Use a deterministic **drain-current-pack, then serial-tail** approximation for
both actions, with no lookahead to unarrived jobs:

1. Snapshot the full queue, remaining budgets, frozen predicted solo slopes,
   measured rate vector for each half, and available overhead/phase estimates.
   Freeze one queue order using the nomination rule above; the admit action
   removes j from that order, while the defer action retains it there.
2. Advance the initial active group to the next member's completion boundary.
   Use the action's measured rate vector while that membership is unchanged.
   For a surviving multi-job subset, use fresh exact matching subgroup evidence
   if available; otherwise retain its parent rates as an explicitly assumed
   fallback. This fallback is not a measurement or a proven bound on either
   action's latency. When a membership change leaves a singleton, use its frozen
   predicted solo slope. An initial singleton with valid measured pre-add rates
   keeps those rates while its initial membership is unchanged.
3. Account for required validation, finalization, checkpoint and worker-exit
   phases, including their GPU residency. An exhausted optimizer budget is not
   a completed job or a necessarily freed GPU. Record each forecast completion.
4. After the current group has drained, execute remaining queued jobs serially
   in the frozen order using their remaining updates / predicted solo slope,
   plus their required setup and completion phases. Record every completion.
   Do not optimize queue order separately for the two actions or assume an
   unobserved future packed configuration is fast.

This restricted rollout is a feasible scheduling template once its rate/phase
assumptions hold, not a prediction of every future decision. Real scheduling
reconsiders after completions, new arrivals, or material rate changes. Its
restricted serial tail and inherited subset rates can mis-rank actions and must
be tested; the heuristic does not claim globally minimal mean JCT.
The old 1800-second local gain-projection horizon is superseded by this completion
rollout; it is distinct from the new queued-age threshold. Forecasting an entire
long job extends beyond the observed windows. Log that extrapolation and test
rate drift and solo-calibration sensitivity rather than treating it as fresh
measurement evidence.

All forecast-required budgets and rates must be valid. If a waiting job has no
usable estimate, do not silently drop it or assign zero duration: the full-queue
score is inconclusive. Incumbents continue, and unknown jobs retain solo-dispatch
eligibility. A future bounded-uncertainty extension can relax this restriction.

Include future action-dependent pause, checkpoint, reload, startup, and finish
costs using matching measured estimates, without counting them twice. Missing
material costs or unknown GPU-occupancy phases make a production-use forecast
inconclusive. An explicitly flagged zero-overhead assumption is permitted only
for research/synthetic evaluation, with no production latency claim. Trial time
already paid is sunk for this continuation decision but remains in actual JCT.
Consequently a positive continuation gain does not prove that initiating the
trial paid for itself. Keep bounded trial/retry controls and evaluate their total
latency cost against policies that never initiated that trial.

### Decision and diagnostics

The proposed replacement gate is Delta_h >= 1.0 second for each of the two
independent measurement halves: at least one second of predicted mean remaining
JCT saved across O. Name the setting `min_mean_jct_gain_seconds`; this is a
provisional research margin, not a business SLO or a statistical confidence bound.
Both halves passing accepts, both failing rejects, and disagreement or invalid
evidence is inconclusive. Threshold equality passes. Retain memory and trial
validity safeguards. Do not additionally require the old 50% slowdown cap.

Continue recording

\[
P_h^-=\sum_{i\in S}r^-_{i,h}/\widehat s_i^0,\qquad
P_h^+=\sum_{i\in S\cup\{j\}}r^+_{i,h}/\widehat s_i^0,\qquad
g_{j,h}=P_h^+/P_h^-.
\]

These are throughput diagnostics. The former g >= 1.05 and H(g-1) >= 5 seconds
thresholds are **superseded**, not extra gates: requiring them could reject an
overlap that improves JCT. Raw update counts and busy percentages are still not
interchangeable with normalized work. Absolute slope calibration now affects
singleton and queued-tail forecasts; preserve predicted/measured provenance.

### Why the objective changes the decision

Consider a running A with 100 updates left and a queued B with one update left,
both submitted at time zero with solo slope 1 update/s. Deferring B gives
C_A=100, C_B=101, mean JCT=100.5 seconds, and makespan=101 seconds.

Suppose overlap gives r_A=0.5 and r_B=0.4. B finishes at 2.5 seconds; A has 98.75
updates left and finishes alone at 101.25 seconds. Mean JCT is 51.875 seconds,
although makespan grows to 101.25 seconds and normalized throughput during
overlap falls from 1 to 0.9. With exact rates and zero overhead, the revised
criterion saves 48.625 seconds of mean JCT and accepts. This illustrates the
objective, not a universal justification for tail-latency regression.

## Lifecycle, safety, and evidence

- Preserve v3's feasibility checks, original submission time, scheduling
  compatibility, memory reservations, live VRAM gate, and backend constraints.
  Replace busy-first ordering with the latency-oriented nomination and explicit
  queued-age protection above; this is an intentional design change.
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
- Record the revised proposed policy identifier `derived_solo_slope_jct_v1`,
  original submission timestamps, frozen predicted profiles, remaining updates,
  before/after rates, the entire forecast cohort/order, per-action per-job
  completion forecasts, assumed versus measured subgroup/overhead sources,
  both mean-JCT deltas, threshold, decision, and reason. Record P/g and incumbent
  slowdown diagnostically. A forecast is never persisted as observed JCT.
- Keep trial time already paid separate from the continuation decision: its
  productive updates are in R_i. Future action-dependent overhead enters the
  rollout; actual end-to-end reporting includes all costs regardless of whether
  an estimate was available during the decision.

## Historical five-job example and latency reanalysis

This is an illustrative fluid calculation, not measured GPU performance or a
reproduction of v3's full state machine. It conditions on two already-running
jobs and a waiting queue, with no later arrivals. Policy candidates are tried
in FIFO order C,D,E to isolate admission from ranking. These schedules use the
**superseded normalized-progress gate**, not the new JCT rollout. Preserve them
as historical evidence about the solo-slope ambiguity, not a benchmark of the
revised policy.

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
trial, exact stable rates, and the original g >= 1.05, H(g-1) >= 5 seconds,
H <= 1800 seconds settings (convert minutes to seconds). Rejected updates are
retained. Scan the waiting FIFO list once between
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

For a metric-only reanalysis, explicitly set all five submission timestamps to
the snapshot time zero; the original experiment did not specify earlier queue
ages. This assumption must not be applied to real traces. From the same saved
per-job completion times:

| Scenario | Policy | Mean JCT (min) | Nearest-rank p95 JCT (min) | Makespan (min) |
|---|---|---:|---:|---:|
| Reference | Fixed-three FIFO | 84.38889 | 97.22222 | 97.22222 |
| Reference | v3 50% principle | 70.15800 | 93.17000 | 93.17000 |
| Reference | Old gain rule | 53.53578 | 78.87444 | 78.87444 |
| Slow C, original cost | Fixed-three FIFO | 88.62963 | 98.70370 | 98.70370 |
| Slow C, original cost | v3 50% principle | 74.39200 | 98.56000 | 98.56000 |
| Slow C, original cost | Old gain rule | 67.43911 | 148.39111 | 148.39111 |
| Slow C, corrected cost | Old gain rule | 65.80000 | 97.00000 | 97.00000 |

With only five jobs, nearest-rank p95 equals the maximum, so this is not a useful
statistical tail estimate. The formerly failing 148.39-minute schedule has a
lower mean than the 50% schedule, because it finishes several other jobs earlier.
That is precisely why average latency, worst waiting/latency, and coverage must
be reported together. None of these rows demonstrates the new JCT policy's
performance; that requires a new matched evaluation after implementation.

## Future implementation and evaluation checks

- Confirm reciprocal units, positive finite validation, consistent model identity,
  unchanged six-column model output, and separation of predicted/measured records.
- Check consistent milliseconds/seconds conversion. The P/g diagnostic remains
  invariant to common cost scaling, but the JCT forecast need not: test shared
  absolute calibration bias and job-specific slope errors explicitly.
- Cover exact threshold equality, pass/pass, fail/fail, disagreement, invalid
  evidence, empty GPU, unavailable prediction, unknown/inconsistent update budget,
  R_i=0, and exhausted trial-update budgets. Cover missing queued-job forecasts,
  singleton/subgroup transitions, inherited-rate provenance, future overhead,
  and the same post-trial state/order under both actions.
- Cover non-overlapping windows, validation/checkpoint contamination, stale
  tokens, interrupted trials, restart, changed membership, and updated source,
  batch, or backend. Verify incumbents continue and rejected progress survives.
- Reproduce the synthetic cases using their full rate law and accounting for
  every required update. Treat the example's one-pass retry simplification
  separately from tests of the actual v3 lifecycle.
- Test that throughput gain can disagree with JCT gain using the 100-update A /
  one-update B example. Test staggered submissions, retry/resume timestamp
  preservation, finalization after the last update, pending-job inclusion,
  failed/cancelled jobs, and censored-run reporting. Verify the area-under-
  unfinished-jobs identity against summed JCT.
- Test a continuous stream of short jobs against a long queued job and an
  unknown-prediction job. Verify oldest-overdue nomination, suppression of fresh
  backfill, preserved memory safety, and eventual feasible solo dispatch; do not
  claim a hard waiting-time bound while incumbents cannot be preempted.
- Evaluate v4, unchanged v3, and the explicitly requested fixed-three FIFO
  comparator on identical arrival traces, source jobs, scientific update budgets,
  batches, seeds, hardware, and backend. Freeze policy parameters before evaluation.
  Include all trial, preparation, checkpoint, restart, and finalization costs.
  Separate admission-only and ordering/queued-age changes in an ablation so their
  effects are not attributed to the predictor alone.
- Require every job to finish and its worker to exit successfully for the main
  cohort comparison. Rank primarily by mean submit-to-completion JCT, report
  median/p95/max JCT and queued-age/coverage alongside makespan, and disclose
  regressions. Timed-out runs are incomplete, not faster completed-only results.
  Use repeated paired traces and uncertainty estimates, rather than promoting a
  policy from one toy trace. Production promotion also requires agreed service
  latency targets and validation of slope/overhead calibration.
  Include per-job timelines and a combined Gantt/progress/latency figure; never
  tune thresholds on held-out evaluation.

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
- [Existing flow-time report fields](../domain/profiles.py),
  [trace flow-time and nearest-rank p95 calculation](../scheduler/trace_simulator.py),
  and [job timestamps](../domain/jobs.py).

The older [predictor policy note](predictor-indicator-policy.md) describes a
related weighted-progress proposal. It is historical design context, not proof
that the v3 default implements that proposal. The executable v3 decision is the
bounded-interference calculation in `add_only_trials.py`.

Related research supports the distinction in objectives, not this particular
implementation: [Tiresias](https://www.usenix.org/conference/nsdi19/presentation/gu)
targets average JCT with partial/missing job-size knowledge, while
[Themis](https://www.usenix.org/conference/nsdi20/presentation/mahajan) treats
finish-time fairness explicitly. Neither establishes the performance of this
single-GPU, add-only, derived-slope proposal.
