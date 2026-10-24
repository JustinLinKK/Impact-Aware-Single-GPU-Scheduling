# oct_proposed_v2: fair progress with measured GPU sharing

Research proposal, October 24, 2026. **This branch contains a design, source
notes and checked arithmetic, not an implemented scheduler mode.** It starts
from `paper` commit `607d5e2`. The CPU student integration is currently in the
separate, uncommitted `paper` checkout; implementing v2 must first integrate
that reviewed change. The earlier `oct_proposed` branch remains independent.

The central decision is: **which addition produces useful extra training
progress, repays its measurement cost, and avoids repeatedly neglecting the
same waiting jobs?** A heavy job can be a good starting job, but heaviness alone
does not make it the preferred main job.

## How this differs from the other two designs

| Policy | Starting job | Adding a job | Main limitation |
|---|---|---|---|
| Implemented `busy_anchor` | High predicted busy within priority rules | Low busy, then short remaining time, then submission time | Solo busy does not predict interference |
| Existing `oct_proposed` | Short remaining time with aging | Largest estimated reduction in total completion delays | Sensitive to remaining-time and future co-run estimates |
| New `oct_proposed_v2` | Most underserved job within priority rules; duration breaks ties | Measured extra normalized progress minus trial/setup cost, with individual slowdown checks | Conservative evidence requirements can miss beneficial combinations |

V2 has no permanent heavy-anchor role. It is a different research objective,
not a claim of new scheduling theory or a guaranteed improvement over v1.
Within an equal-priority fresh queue, every service deficit is initially zero,
so its first choice may match shortest-job-first. Its later service accounting,
admission objective, exploration budget and stale-evidence handling distinguish it.

## What other schedulers contribute

These are mechanisms reported by the original papers. Their likely use on our
single GPU is an adaptation, not a reproduction of their cluster results.

| System | Published mechanism | What we borrow here |
|---|---|---|
| [Gandiva, OSDI 2018](https://www.usenix.org/system/files/osdi18-xiao.pdf) | Profiles repetitive training and retains packing when observed throughput improves; harmful packing is undone | Treat busy as a trial hint and validate the actual combination |
| [Tiresias, NSDI 2019](https://www.usenix.org/system/files/nsdi19-gu.pdf) | Uses attained-service scheduling when duration is uncertain, plus waiting-based promotion | Preserve progress history and waiting priority; avoid dependence on an exact finish-time prediction |
| [Themis, NSDI 2020](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf) | Uses finish-time fairness and prioritizes applications far from their fair reference | Measure service shortfall and job-level harm alongside throughput |
| [Gavel, OSDI 2020](https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf) | Represents effective throughput and distinct throughputs for colocated job combinations | Store exact-group rates and compare progress in consistent units |
| [Salus, 2019](https://arxiv.org/abs/1902.04610) | Provides iteration scheduling, fast switching and memory sharing through a modified execution service | Separate a scheduling policy from the runtime mechanisms it would need |
| [Shockwave, NSDI 2023](https://www.usenix.org/system/files/nsdi23-zheng.pdf) | Plans around changing training throughput while accounting for fairness | Invalidate stale measurements when phase or approved batch configuration changes |

For this scenario, a Gandiva-like method would test actual packing, a
Tiresias-like method would account for received service, and a Gavel-like method
would compare measured combinations. Themis motivates measuring who loses from
sharing. Salus-style rapid switching cannot simply be assumed for separate
PyTorch processes; Shockwave motivates revisiting measurements as training changes.
These are qualitative adaptations. Their published speedups do not transfer to
our GPU or workloads without a controlled experiment.

## Decisions in plain language

1. **When idle:** select the feasible job most behind its reference share of
   service, within existing priority and starvation rules. If equally behind,
   prefer shorter trusted remaining time, then earlier submission.
2. **While work runs:** keep existing jobs running. For each feasible newcomer,
   first reuse fresh evidence for that exact combination. Rank profitable
   additions by conservative extra progress, then service deficit and duration.
3. **For an unknown combination:** allow a bounded trial only if the trial
   budget, memory, backend and rollback checks pass. Use CPU student busy/time
   predictions to order exploratory candidates, not to assert a co-run rate.
4. **Keep a newcomer** only when measured progress gain covers its cost, no
   member violates the slowdown limit, and existing admission/trial gates pass.
5. **When a job finishes or changes phase:** invalidate incompatible evidence
   and reconsider additions. Do not automatically replace it with a heavy job.
6. **When a job has waited too long and cannot join:** reserve its turn and stop
   new admissions that would delay it. Drain the current finite group, then run
   the reserved job when feasible. Do not kill incumbents or change training quality.

This first version adds no fairness-driven eviction of accepted incumbents;
existing trial references may temporarily yield jobs and safely roll back a
newcomer. A
reservation does not give an immediate waiting-time bound: a long incumbent
still has to finish. It does not reproduce the fairness guarantees of systems
with checkpoint preemption or fine-grained time slicing. A permanently oversized
job is marked infeasible; it must not reserve the GPU forever.

## Define progress before comparing jobs

Use committed optimizer updates at an unchanged, accuracy-approved training
configuration. For job i and current phase:

```text
t_solo[i]   = measured seconds per optimizer update when running alone
t_group[i] = measured seconds per update in the exact active group
r[i, G]    = t_solo[i] / t_group[i]       # solo-equivalent seconds per wall second
P[G]       = sum of r[i, G] over members
```

This avoids adding incompatible raw steps/second from different models. P may
exceed one when sharing recovers idle intervals. It measures normalized training
progress, not FLOPs, occupancy, accuracy gain, or a partition of GPU capacity.
The solo baseline must match hardware, batch, optimizer, precision, backend and
phase. Changing the baseline settles the old accounting segment first; do not
retroactively reprice previous work.

The student remains CPU TorchScript. Its solo time estimate can initialize
ordering, but measured same-hardware solo references are required for a committed
progress comparison. Unsupported hardware or workload predictions use existing
branch profiles. Busy remains an activity proxy; never add busy percentages to
decide whether a pack fits.

## Account for underserved jobs

Maintain a persistent ledger for each stable job ID. Default share weights w
are one; explicit positive share weights are separate from priority numbers.
Let E be admitted-to-queue jobs that are runnable, feasible alone and neither
user-paused nor dependency-blocked. E includes all such jobs across priority
bands; priority affects dispatch order, not this equal-share accounting reference.
Consequently a lower-priority job can accumulate debt while priority rules still
delay it. Update the virtual reference while the
scheduler is available to execute work:

```text
F[i] += elapsed_seconds * w[i] / sum(w[k] for k in E)
W[i] += newly_committed_updates * frozen_phase_t_solo[i]
debt[i] = max(0, (F[i] - W[i]) / w[i])
```

F is reference service; W is completed solo-equivalent work. A new arrival
starts at zero. Waiting grows its debt; useful progress repays it. Preserve signed
F-W in storage so earned credit survives; clamp only for ranking. Do not accrue
entitlement while WSL is offline or the user pauses the study. Per-job fairness
is the declared unit: it does not prevent a user from submitting many jobs.
Tenant/workflow fairness would require a separate policy and authenticated identity.

Existing effective priority and mandatory starvation reservations take precedence.
Within the selected priority band, use debt bands to avoid ranking churn, then
trusted remaining time and submission order. Missing solo references use waiting
age until a solo baseline is acquired; do not fabricate W from predicted busy.
An illustrative debt-band width is 30 solo-equivalent seconds; fix the width on
validation traces. Retries retain the same ledger. Persist the last accounting
timestamp, entitlement, earned work and reservation atomically. Release a
reservation if its job is cancelled or becomes ineligible. Rollback must restore the last durable progress
cursor and subtract any provisional, lost work, never double-count replayed updates.

This differs from simple least-attained-service: all never-started jobs have
zero attained service, whereas their accumulated waiting debts can differ.
It is an engineering heuristic, not Themis's finish-time fairness metric.

## Require gain that pays for the trial

Maintain conservative empirical rate envelopes for the exact current phase.
They are derived from repeated non-overlapping valid measurement windows with
a held-out calibrated margin. They are **not** statistical coverage guarantees;
even non-overlapping windows can be correlated. Two adjacent windows alone do
not establish a confidence interval. Unknown or
unstable envelopes cannot authorize admission.

For active group A and proposed newcomer j:

```text
delta_P = sum(r_low[i, A+j]) - sum(r_high[i, A])
net_gain = H * delta_P - C
```

H is a conservative planning horizon, capped at the next planned reconfiguration
boundary, earliest plausible member completion and next known phase change.
Use a separately configured planning interval (illustratively 120 seconds), not
the routine telemetry or scheduler polling interval. Monitoring can be frequent
without shrinking the amortization horizon. Remaining duration is
epoch time times epochs left. C is an upper planning estimate of lost
solo-equivalent work from setup, reference collection, switching and rollback.
H is wall seconds; C and net_gain are solo-equivalent seconds. Short remaining
work, early stopping and unknown duration reduce confidence in amortization.
No reliable horizon means defer packing or use a separately budgeted trial;
do not invent a long horizon to make the inequality pass.

Accept only if delta_P is positive, net_gain exceeds a calibrated margin, every
member's worst estimated solo-relative slowdown is within its configured limit,
and all existing drain-gain, memory and backend checks pass. Also bound additional
incumbent slowdown relative to the pre-add group so successive admissions cannot
silently compound harm. Do not trade one job's severe slowdown for another job's
large gain. Passing v2's score never bypasses the current safety/trial gates.

After admission, keep sampling. On degradation stop further admissions and apply
the existing safe newcomer rollback protocol. If rollback cannot be completed
safely, report the condition and drain; do not claim that SIGSTOP frees VRAM.

## Spend less time exploring

Reuse the existing exact-group profile store and rejection TTL. Extend their
records with phase identity, empirical envelopes, measured setup/rollback cost,
decision reason and observation count. A cache key includes all members with multiplicity, source
fingerprints, approved physical batches, optimizer/accumulation/precision,
hardware, backend/MPS configuration and capture schema. A pair result is never
treated as proof that the same pair plus a third job will work.

Deduplicate a running trial, cool down rejected or inconclusive combinations,
and invalidate affected entries after membership, configuration or phase changes.
A single bad observation triggers a temporary cooldown; retain the existing
repeated-bad-trial rule for a durable rejection. Profile lookup must not bypass
live VRAM, current service reservations or mandatory runtime acceptance checks.

Budget exploratory wall time with a persistent token bucket. An illustrative
starting setting is a 30-second initial/capacity credit, replenished at 3% of
unpaused scheduler wall time. Reserve the estimated full attempt cost before
starting and charge actual wall time, including startup and failed references.
This bounds planned trial spending to startup credit plus replenishment. Actual
execution can overrun while waiting for a safe boundary or rollback; an
unresponsive worker provides no hard wall-clock bound. Charge the excess as debt,
report it, and suspend further exploration until budget and runtime health recover.
This budgets exploration time, not the number of concurrent training jobs. Tune these numbers on validation
traces; they are not measurements or recommendations from the cited papers.

If a meaningful trial cannot fit the available budget, run existing work and
defer exploration. Initial solo training remains available without a co-run trial
budget. A starvation reservation can drain the GPU and obtain a solo slot, but
does not bypass trial spending limits. Useful training performed during a trial
still counts in W; charging its entire wall time is a conservative overhead budget.

## Checked examples, not benchmark results

Assume valid exact-group evidence, all memory checks pass, and A alone has
normalized rate 1. All rates below are conservative post-sharing rates, not busy
percentages. The candidate solo rates are normalized to one.

| Candidate | Solo busy hint | A's rate after sharing | Newcomer's rate | Decision |
|---|---:|---:|---:|---|
| B | 10% | 0.55 | 0.15 | Reject: total progress falls to 0.70 |
| C | 40% | 0.90 | 0.75 | Potentially useful: total progress rises to 1.65 |

With H=120 seconds and C=12 solo-equivalent seconds, C's net gain is
120 × (1.65 - 1) - 12 = **66**. With only 10 seconds of overlap it is **-5.5**,
so the same pack should be skipped. Both members satisfy an illustrative 1.5×
solo slowdown ceiling: 1/0.90=1.11 and 1/0.75=1.33.
A different pack giving A=0.50 and newcomer=1.20 has positive aggregate gain,
but fails because A slows by 2×. These examples show why the lightest job is not
necessarily the best partner, and why a good pairing may not repay a short trial.

For the service ledger, two runnable equal-weight jobs over 60 seconds each
receive F=30. If one made 60 seconds of solo-equivalent progress and the other
waited, their ranking debts are 0 and 30. A newly arriving job starts at zero,
so it cannot erase the older job's accrued claim simply by being short.

`docs/research/oct_proposed_v2/check_examples.py` verifies this arithmetic and
the units represented by the example inputs. It is not a scheduler simulator,
GPU experiment, or evidence of a measured speedup.

## Implementation map and evaluation

The existing scheduler already has normalized-throughput checks, drain-time
estimation, profile caching and rejection TTLs. Those are reused, not claimed
as new contributions. Proposed extensions are service-debt ranking/reservation,
cost-aware candidate ordering, empirically conservative evidence, phase-aware
reuse and a persistent exploration budget.

| Area | Existing integration point | Proposed addition |
|---|---|---|
| Candidate order | `scheduler/placement_planner.py` | Separate optional policy; debt bands and no forced heavy replacement |
| Estimates | `scheduler/resource_estimator.py` | Versioned solo references and reliability/horizon fields |
| Measurement | `scheduler/cold_start.py`, `colocation_trials.py`, `colocation_evidence.py` | Envelopes, setup cost and individual slowdown evidence |
| Admission | `scheduler/colocation_decisions.py`, `time_objective.py` | Net-progress filter alongside existing drain-gain checks |
| Persistence | `scheduler/recovery.py`, job store and runtime profile records | Ledger cursor, reservation, token balance and versioned cache |
| Fairness/drift | `scheduler/service.py`, `telemetry.py`, `contention_guard.py` | Debt updates, phase invalidation and drain reservation |
| Progress cursor | `execution/trial_control.py`, `execution/control.py` | Idempotent committed-update and rollback accounting |
| Cache identity | `domain/identity.py` | Extend full colocation keys with explicit phase/configuration identity |
| Decision replay | `scheduler/placement_replay.py` | Recheck debt, reservations and trial budget rather than replaying stale slot order |

Keep CPU TorchScript, branch-profile/live VRAM admission, the current 31 GiB
(31744 MiB) ceiling or any stricter configured limit, approved training batches,
and backend compatibility. Add no fixed maximum-parallel-jobs setting. No model
architecture, optimizer, training accuracy target or active training process
changes are part of this proposal.

Compare baseline, implemented busy_anchor, v1 and v2 using the same arrivals,
hardware, approved batches, progress targets, seeds and profiling costs. Include
short bursts behind long jobs, low-busy bad partners, high-busy good partners,
short overlap, three-way interference, stale phase measurements, heavy survivors,
unknown durations, many new arrivals, and crash/resume with service debt.

First implement pure CPU decision tests and replay recorded branch profiles;
then run controlled GPU trials only as a separate implementation/experiment task.
Do not call a mathematical example a measured comparison. Measure mean/p95 JCT,
makespan, max wait, per-job slowdown and normalized service deficit, useful
training progress, trial seconds and rejection rate, prediction/cache misses,
decision CPU time, VRAM/OOM and training quality. Measure occupancy/FLOPs directly
if claimed. Include cold and warm caches and held-out workloads; fix thresholds
before test traces. Ablate debt, overhead budget, phase invalidation and predictor
hints to establish what actually helps.

Save every trace. After each experiment create the required Gantt-above-metric
plots in one PNG, with all comparison arms together. No schedule experiment has
been run in this branch. Adoption requires useful progress or JCT improvement
without unacceptable tail/fairness regression or quality/memory failures. V2 may
lose to v1 on very short homogeneous queues or when conservative trial budgets
prevent learning; report those outcomes rather than retuning on the test set.
