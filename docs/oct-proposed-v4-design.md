# Request feedback latency: oct_proposed_v4

Status: opt-in research implementation, 2026-10-28. Branch `oct_proposed_v4`
began at `oct_proposed_v3` commit `4a27bcf`; this refactor follows the JCT design
at `58e69ea`. The default remains v3. No model training, dataset rewriting,
GPU experiment or deployment is part of this change.

One shared forecast engine supports two **per-request** completion policies:

| Policy | Agent waits for | Request completion |
|---|---|---|
| `individual` | one candidate's usable result | that job's result-ready time |
| `group` | every required candidate in a declared batch | last required result-ready time |

The policies can coexist in the same queue. A workflow identifier alone does
not establish a barrier. A group counts once, and its children are not counted
again as independent requests. This supersedes both the original normalized-
progress gate and the earlier v4 shortest-remaining ordering. Candidate order is
now FIFO with age protection, shared by both completion policies.

The [labeling proposal](solo-slope-labeling.md) still uses derived labels first.
A reciprocal wall-step prediction supplies an uncalibrated solo-slope prior,
not an interference model or an accuracy improvement.

## Average feedback metric and completion contract

Let q denote a logical feedback request, a_q its original accepted submission
and G_q its finalized set of required job IDs. Let C_i include required outputs,
validation, checkpoint/finalization, successful worker exit and result delivery.
For individual requests |G_q|=1; for an all-required group:

\[
C_q=\max_{i\in G_q} C_i,\qquad F_q=C_q-a_q,\qquad
J=\frac{\sum_q w_q F_q}{\sum_q w_q}.
\]

The headline experiment metric is the **average request feedback time**:

\[
\overline F_{\mathrm{request}}=\frac{1}{|Q|}\sum_{q\in Q}(C_q-a_q).
\]

“Average” here means the arithmetic mean. Mean job JCT already gives each job
one equal vote; counting one group once instead changes the unit of observation.
All mock experiments use weight one per request. The general weighted formula
above remains an optional API capability, not an implicit job-size weighting.
The default weight is one **per request**, not per job or per predicted FLOP.
Explicit positive request weights are supported; report weighted and unweighted
results separately when they differ. Group completion is a local barrier, not
necessarily the end of the entire agent workflow. Minimizing one group's latency
is equivalent to minimizing that group's last completion; it is not generally
equivalent to global makespan when several requests arrive at different times.

### MLMaster bracket makespan versus MLEvolve job JCT

For the specified MLMaster workflow, bracket b is submitted together at A_b and
no further generation starts until every required job in B_b returns. Its
scheduling objective is therefore the **bracket makespan**:

\[
M_b=\max_{i\in B_b}C_i-A_b,\qquad \min_\pi M_b(\pi).
\]

The clock includes waiting after submission and all required result-return work.
It does not start at the first GPU dispatch. For one outstanding bracket, v4's
group score is exactly M_b, not the average completion time of its member jobs.
The two-half admission test reduces to

\[
\Delta_{b,h}=\max_{i\in B_b}\widehat C_{i,h}(\mathrm{defer})-
\max_{i\in B_b}\widehat C_{i,h}(\mathrm{admit})\ge\varepsilon.
\]

An early member completion has value only if it helps the last required result
arrive sooner. Reducing average member JCT while delaying that last result does
not improve this objective. The existing `group` implementation already takes
this maximum, and the simulator already releases the next bracket only after
the current bracket finishes. This clarification changes presentation, not the
admission algorithm or its numerical results.

Across a fixed sequence of m brackets, report M_b for each bracket, sum(M_b),
and average bracket makespan sum(M_b)/m. This last average is **across brackets**,
never across their member jobs. From the first bracket's submission to the last
result, total workflow time is sum(M_b) plus inter-bracket generation delays.
With fixed brackets and fixed generation delays, reducing the sum of bracket
makespans reduces workflow time; adaptive changes in future jobs require a
broader evaluation.

For the specified MLEvolve pipeline, each job's result independently enables
further work, so the primary latency objective remains average individual JCT,
sum_i(C_i-a_i)/n. Keep tail JCT and total workflow time alongside it. The report
and chart now name these two primary metrics separately.

The request clock starts before queueing and preparation, and survives retry,
partial execution, checkpoint and resume. Include all trial time in measured
latency. Do not end the clock at the last optimizer update. The existing durable
job `finished_at` is a scheduler-side endpoint; actual agent-consumption latency
requires a caller-provided `feedback_ready_at` receipt and application tracing.
This change does not add a transport-acknowledgment service or rewrite the
existing live metrics API. Synthetic results explicitly assume instant delivery.

For a fixed set of successful requests, weighted total latency is the area under
the weighted number of unfinished requests. Completed requests add a constant to
both actions and are omitted from the admission cohort. A partly completed group
stays outstanding; already completed member timestamps are retained in its max.

Default completion condition is `successful`. Explicit `terminal` means the
caller can proceed with a failure/cancellation result as feedback; it does not
turn failure into a successful training result. A failed required member makes
an outstanding success-only group infeasible, so its comparison is inconclusive.
Fully terminal historical requests are omitted from new decisions, but failed
requests must still appear in evaluation coverage and failure reporting.

Report request mean/median/p95/max, job mean/median/p95/max JCT, queued age,
completion coverage, failure/cancellation and makespan. An incomplete run is
censored: report unfinished ages and do not rank completed-only averages. Preserve
existing `mean_flow_time_seconds`, `p95_flow_time_seconds` and related job fields;
request metrics are separately named in the simulator. With n=5 and simultaneous
submissions, nearest-rank p95 job JCT equals maximum JCT and makespan.

Mean latency can improve while an individual waits longer. The age guard limits
repeated bypass but is not a tail SLO, deadline guarantee or optimality proof.
Admission uses only arrived jobs; endogenous agent arrivals depend on previous
results and should not automatically be modeled as an independent Poisson stream.

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

The additive field on `predict_profile_options` is shown below for an
illustrative t_hat_i of 28.097898000851274 ms; this is not a recorded model run:

```json
{
  "derived_solo_slope": {
    "value": 35.58985088385271,
    "unit": "updates_per_second",
    "source": "reciprocal_wall_step_prediction",
    "step_semantics": "nominal_optimizer_update",
    "artifact_sha256": "<predictor artifact hash>",
    "calibration": "uncalibrated_prior"
  }
}
```

The enclosing profile retains `kind: predicted_workload`, `step_cost` in nominal
milliseconds, `artifact_sha256`, `hardware`, `batch_size`, `job_signature`,
`calibration: uncalibrated_prior`, and `requires_online_validation: true`.
This field is implemented in `predict_profile_options`; measured records are unchanged.
The six-output teacher/student tensor interface and existing fields remain intact.
Neither the estimate nor its derived ETA may be written to measured-profile
tables or passed off as a verified drain-time estimate.

All outstanding jobs in the forecast cohort must have valid profiles from the same artifact and trained-hardware
scale, matching their current workload, resolved batch, precision, optimizer,
accumulation semantics, and supported runtime context. Freeze the full profile
identity and slope for the trial, including queued jobs. A changed signature, batch, artifact, hardware,
backend configuration, or membership invalidates the comparison.

If a valid compatible prediction is unavailable for any member, do not authorize
v4 packed admission for that group. Keep incumbents running and leave the
candidate eligible for a future compatible group or solo execution under the
existing memory and execution checks. Do not substitute zero work, mix measured
and uncalibrated cost scales, or silently switch to the 50% policy.

## Request metadata and configuration

Enable the existing cooperative add-only path explicitly:

```yaml
gpu_scheduler:
  feedback_latency:
    enabled: true
    min_gain_seconds: 1.0
    max_queued_age_seconds: 1800.0
    allow_zero_overhead: false
    phase_model_ttl_seconds: 3600.0
```

These are provisional research settings. Existing backend, cooperative runner,
memory and live-trial prerequisites still apply. Leaving `enabled: false` keeps
v3's candidate ordering and 50% criterion. V4 does not silently fall back to v3
when information is missing, and changing policy/settings during a measurement
invalidates that trial. Natural solo execution remains available.

Without `metadata.feedback_request`, a job is explicitly treated as one
individual successful request, keyed by workflow and job ID. For a group, repeat
the same canonical declaration on **every** required member:

```json
{
  "feedback_request": {
    "workflow_id": "agent-session-1",
    "request_id": "round-7",
    "completion_policy": "group",
    "completion_condition": "successful",
    "required_job_ids": ["candidate-7a", "candidate-7b"],
    "membership_finalized": true,
    "submitted_at": "2026-10-28T12:00:00+00:00",
    "weight": 1.0
  }
}
```

Explicit individual requests use `completion_policy: individual` and one required
ID. Request time must be no later than any member's original submission. Request
keys are `(workflow_id, request_id)`; one job belongs to exactly one request.
Missing members, unsealed/dynamic groups, conflicting declarations, unsupported
policies and overlapping ownership invalidate packed comparison. Submit the
complete group before it can participate in packed admission. Until then its
members remain eligible for feasible solo execution. No group is inferred from
same-workflow or coincident arrival alone.

Reserved future semantics include first acceptable result, k-of-n results,
dependencies/critical paths, speculative cancellation and deadlines. They are
not accepted policy values today. These require explicit utility/completion
contracts, including who may cancel work and which result unblocks the agent.

## Candidate ordering and age protection

Eligible cooperative jobs are nominated by original submission time and stable
job ID, both at idle dispatch and for packed trials. Predicted busy and shortest
remaining time do not determine v4 rank. Compatibility, memory and backend
checks still apply. Retain v3's incumbent continuity, clean measurement windows,
checkpoint preservation, retry limits and per-group cooldowns.

Once a feasible waiting job reaches `max_queued_age_seconds`, nominate the oldest
overdue job and suppress fresh backfill. If it cannot join safely or its trial is
rejected, let residents drain toward its next feasible solo opportunity. A trial
alone does not reset its age/protection. Do not evict incumbents or override
memory checks. Unknown-prediction jobs keep solo eligibility. This is not a hard
waiting-time bound under overload, long non-preemptible jobs or incompatible
resources. Existing v3 `starvation_timeout_seconds` remains reporting-only.

This implementation covers the cooperative add-only path. Noncooperative jobs
still follow existing solo compatibility paths; v4 prevents legacy packed
fallback while workers are active. There is no v4 fixed concurrency cap.

## Shared admission forecast

For candidate j and incumbent set S, use fresh phase-clean pre-add and overlap
windows, split into two independent halves h. Convert measured seconds/update
into rates r^- and r^+ in updates/s. A pre-add slope is the current pack's rate;
it is not necessarily a solo slope. Both alternatives start from the same latest
post-trial remaining updates, including progress since measurement windows ended:

- `admit`: keep j in S union {j} using measured overlap rates.
- `defer`: checkpoint/requeue j and continue S using measured pre-add rates.

For every outstanding request, forecast member completion timestamps
C_hat_i,h(a), then score

\[
\widehat J_h(a)=\sum_{q\in Q_t} w_q
\left[\max_{i\in G_q}\widehat C_{i,h}(a)-a_q\right],
\qquad
\Delta_h=\frac{\widehat J_h(\mathrm{defer})-
\widehat J_h(\mathrm{admit})}{\sum_{q\in Q_t} w_q}.
\]

Completed companions of outstanding groups use their observed timestamp. Every
unfinished arrived job, including waiting jobs, must belong to the scored cohort.
Otherwise the comparison is inconclusive. Request ages cancel between actions;
adding ages to a score does not itself prevent starvation.

Both halves must have Delta_h >= `min_gain_seconds` (initially 1 second) to accept.
Both failures reject; disagreement or invalid evidence is inconclusive. Equality
passes. There is no additional 50% slowdown or normalized-progress gate. The
margin is a research default, not a confidence bound or promised time saving.

The deterministic rollout in `scheduler/feedback_objective.py`:

1. Freeze the full outstanding request cohort, profiles, settings, current active
   membership and one FIFO tail. A new arrival or changed request/profile/cohort
   requires fresh evidence. Current counters supply both alternatives' R_i.
2. Drain each action's initial group to its next training-completion boundary.
   Keep measured rates while membership is unchanged. Use current exact subgroup
   evidence if available; otherwise inherit parent rates with explicit assumed
   provenance. On transition to a singleton use its predicted solo slope. An
   initial singleton keeps its measured rate until membership changes.
3. Reserve required remaining finish phases serially on the GPU. During this
   approximate reserve other jobs make no training progress. Add nonresident
   delivery time to each result timestamp without holding the GPU for delivery.
4. After residents drain, execute the same frozen serial tail using R_i/s_hat_i^0,
   plus startup/resume, finish and delivery estimates. Defer includes the
   candidate's pause cost before the continuation and resume cost in its tail.
   Admit removes the candidate from that same tail.

This is a restricted continuation model, not a prediction of all future
rescheduling or a bound. Only the current trial's measured groups are available;
unknown future interference is not read from the simulator. Real scheduling
reconsiders after completions. The serial tail and inherited subset rates can
mis-rank actions, even with accurate solo slopes.

Absolute prediction scale now matters because measured and predicted seconds
coexist in the forecast. Scaling all reciprocal costs leaves the historical P/g
ratio invariant, but does **not** generally leave this JCT gate invariant.
Multiplying all request weights by a common factor does preserve the decision.
The former H=min(1800 s, next-completion times) and g>=1.05/H(g-1)>=5 s gates are
superseded. The new 1800-second age setting has a different purpose.

Keep normalized progress P^- = sum(r^-_i/s_hat_i^0), P^+ = sum(r^+_i/s_hat_i^0),
g=P^+/P^- and incumbent slowdown ratios as diagnostics. The scorer records them,
the complete cohort/order, per-job and per-request forecasts, both deltas,
phase assumptions, prediction provenance, threshold and reasons. Policy ID is
`derived_solo_slope_feedback_v4`.

### Completion-phase contract

Production-use comparison requires a complete matching estimate on every
outstanding job. The adapter accepts `metadata.feedback_phase_model` with:

```json
{
  "source": "measured_phase_model_v1",
  "covers_required_phases": true,
  "job_signature": "<current packing signature, or job ID>",
  "batch_size": 32,
  "hardware_key": "<store hardware key>",
  "backend": "cuda_process",
  "observed_at": "2026-10-28T12:00:00+00:00",
  "startup_seconds": 2.0,
  "finish_seconds": 5.0,
  "delivery_seconds": 0.1,
  "pause_seconds": 1.0,
  "resume_seconds": 2.0
}
```

Numbers above are illustrative. Values must be finite and nonnegative, identity
must match, and age must be within `phase_model_ttl_seconds`. `finish` covers all
remaining required non-training work, including validation/checkpoint/finalization
and worker release; recurring future non-training phases must be represented,
not dropped. `delivery` is the additional nonresident result-return delay.
Queued jobs with prior progress/checkpoints use resume instead of fresh startup.
This serialized phase proxy is an approximation, not a bound.

**An automatic collector/calibrator for this phase model is not implemented.**
Callers must supply matching observations through the metadata contract. Missing,
stale, malformed or incomplete phase evidence makes packed comparison inconclusive.
Only explicit `allow_zero_overhead: true` permits the labeled research assumption
used in the synthetic example. Trial setup already paid is sunk for continuation
scoring but remains in actual request latency. Positive continuation gain does
not prove that initiating the trial paid for itself.

### The policies really can disagree

With A=100 and B=1 remaining updates, solo rates 1/s, and zero phase costs,
defer predicts A=100 s, B=101 s. Overlap at .5/.4 updates/s predicts B=2.5 s and
A=101.25 s after solo completion. For two individual requests, mean latency falls
from 100.5 to 51.875 s, a 48.625 s gain: accept. For one group AB, feedback grows
from 101 to 101.25 s: reject. Both use identical job forecasts and the same
one-second margin. This behavior is tested in the shared scorer.

## Lifecycle and limitations

The v3 controller still owns warm-up, overlapping clean windows, minimum measured
progress, runtime identity, update limits, teardown, retries and checkpoints.
Only the continuation objective and candidate ordering change when enabled.
Incumbents continue through preparation and rejection; rejected/inconclusive
newcomer progress remains saved. Exhausted budgets, interrupted trials, changed
membership or stale tokens require reconciliation/fresh windows. Missing valid
predictions for any forecast job do not become zero-duration jobs or invoke v3.
An idle GPU can still execute a feasible solo job without a gain calculation.

The request registry is currently validated metadata, not a new server API.
There is no planner for first-result utility, dynamic DAGs, cancellations or
unknown future arrivals. Both policies optimize forecasts, not guaranteed actual
JCT, and neither claims to optimize final scientific quality. A future agent-level
metric can track time-to-acceptable-model or iteration utility separately.

## Main experiment: two submission scenarios

The [main report](../benchmarks/feedback_v4/submission_results/report.md) and
[combined figure](../benchmarks/feedback_v4/submission_results/comparison.png)
compare fixed-three FIFO with request-aware v4. V3 is omitted from the new table
and chart at the user's request; prior traces below remain historical evidence.
The same five job types are run for two cycles, ten jobs per run:

- MLMaster 2-style batch barriers submit A1–E1 together, then A2–E2 ten seconds
  after the last first-batch result. Each batch is one group request.
- MLEvolve-style sequential/pipelined submission starts A1–E1 ten seconds apart.
  Each returned result enables its corresponding successor after ten seconds;
  one issuer spaces submissions by at least ten seconds. Multiple jobs may be
  outstanding; every job is one individual feedback request.

The user confirmed the pipelined interpretation. These are synthetic contracts,
not exact framework reproductions or framework-default parallelism. Source links
and assumptions are recorded in the report. Both policies use identical jobs
and dependency/think-time rules, but their resulting arrival timestamps differ.
Thus this is a closed-loop evaluation, not identical exogenous arrival traces.

| Reference scenario | Policy | Average feedback (min) | p95/max feedback (min) | Workflow makespan (min) |
|---|---|---:|---:|---:|
| Batch barrier | Fixed-three FIFO | 97.22 | 97.22 | 194.61 |
| Batch barrier | V4 group | 78.87 | 78.87 | 157.92 |
| Sequential pipeline | Fixed-three FIFO | 89.36 | 104.65 | 196.14 |
| Sequential pipeline | V4 individual | 55.75 | 91.17 | 159.17 |

All 12 main runs and two strict-single-outstanding controls complete ten jobs,
conserve productive trial updates and respect VRAM. Stale-C prediction worsens
batch feedback to 148.39 minutes and pipeline p95 to 172.20 minutes. Corrected
slopes reduce those to 97.75 and 101.33 minutes, respectively. This remains a
zero-phase-overhead, noiseless synthetic model; prediction and phase calibration
are required before production conclusions.

A strict single-outstanding control has identical outcomes under both policies
and no packing opportunity. It also has lower individual latency but longer
workflow time than the pipeline, demonstrating why the main averages must be
compared within one submission contract. Keep workflow completion time alongside
request feedback: delaying submission can make a request average look better
without making an agent's work finish earlier.

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
be reported together. These historical rows are not evidence for the request-policy refactor. The
new executable rerun is reported separately below.

## Historical five-job rerun

The [historical report](../benchmarks/feedback_v4/results/report.md) and
[combined chart](../benchmarks/feedback_v4/results/comparison.png) cover 15 runs:
fixed-three FIFO, v3 50%, v4 individual, v4 group and v4 mixed, each under reference,
slow-C/stale and slow-C/accurate solo predictions. Every run completes all jobs,
retains trial progress and stays within the example's 12 GiB budget. This is a
synthetic CPU experiment, not a production benchmark. All arrivals are at zero;
startup/checkpoint/finalization/delivery cost zero; trials cost one productive
minute, and the sample/retry lifecycle is simplified explicitly.

| Scenario | Policy | Mean job JCT (min) | p95/max and makespan (min) |
|---|---|---:|---:|
| Reference | Fixed-three FIFO | 84.39 | 97.22 |
| Reference | V3 | 70.16 | 93.17 |
| Reference | V4, each partition | 53.54 | 78.87 |
| Slow C, stale prediction | V3 | 74.39 | 98.56 |
| Slow C, stale prediction | V4, each partition | 67.44 | 148.39 |
| Slow C, accurate prediction | V3 | 74.39 | 98.56 |
| Slow C, accurate prediction | V4, each partition | 65.80 | 97.00 |

All three v4 partitions happen to choose the same schedule here. This supports
neither a claim that group mode dominates individual nor that every request
metric improves. In the accurate slow-C case, mixed AB/CD/E feedback worsens
from v3's 66.12 to 74.33 minutes even though job mean and makespan improve. These
counterexamples are retained in regression tests. The two-job test above
establishes that the implemented policies can differ; broader workloads are
needed to evaluate which is useful for a real agent's dependency pattern.

## Verification and future evaluation

Implemented checks cover positive finite predictions and unchanged tensor shape;
exact threshold, pass/pass, fail/fail and disagreement; partial group completion,
failed required members and duplicate membership; missing/full queue evidence;
post-trial remaining work; phase costs; delivery without GPU occupancy; changed
settings/profiles/cohort; retained newcomer updates and incumbent continuity;
FIFO/age nomination and missing-prediction solo eligibility; common-weight
invariance versus absolute-slope sensitivity; batch barriers, pipelined dependency
releases and no-work-before-submission accounting. Simulation tests
verify all updates, VRAM, finish coverage and historical reference values.
The existing lifecycle suites cover interrupted windows and budget exhaustion.

Future evaluation must test absolute/common calibration bias, job-specific slope
errors, unit consistency, rate drift/noise, nonzero phase overhead, staggered and
feedback-dependent arrivals, starvation under a continuing short-job stream,
restarts, output transport/acknowledgments, failures and censored runs. Compare
unchanged v3 and fixed-three FIFO on identical workloads, recipes, seeds,
backend/hardware and request partitions; require every job to finish. Report
request and job mean/p95/max, queue ages, failures/coverage, all trial/checkpoint/
resume overhead and makespan. Use multiple paired traces and uncertainty, plus
separate admission/ordering ablations. Do not tune margins on held-out evaluation
or promote this feature based on one favorable toy workload.

## Source references

- [Shared request/forecast engine](../scheduler/feedback_objective.py) and
  [controller adapter](../scheduler/feedback_trials.py).
- [Live trial gate](../scheduler/add_only_trials.py),
  [FIFO/age and lifecycle](../scheduler/cold_start.py),
  [placement fallback guard](../scheduler/placement_planner.py).
- [Settings](../config/models.py), [predicted-profile field](../prediction/ml_predictor.py).
- [Validated update budget](../scheduler/completion_objective.py),
  [step derivation](../prediction/released_student.py), [job timestamps](../domain/jobs.py).
- [Submission scenarios](../benchmarks/feedback_v4/submission_scenarios.py),
  [historical five-job simulation](../benchmarks/feedback_v4/simulate.py),
  [scorer tests](../tests/test_feedback_objective.py),
  [live controller tests](../tests/test_feedback_trials.py),
  [simulation regressions](../tests/test_feedback_simulation.py).

The [older predictor note](predictor-indicator-policy.md) is historical context.
The v3 default remains bounded incumbent interference; it is not the gain or
request-latency policy. Dataset measurements and runtime estimates remain separate.
