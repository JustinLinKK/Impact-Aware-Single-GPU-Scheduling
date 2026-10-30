# Iteration 2: resource and duration mechanisms — candidate protocol

This protocol is prepared for review after iteration 1. It has not been launched
or selected from iteration-2 outcomes. The running iteration-1 application is
unchanged. The original predictor weights, profile labels and dataset splits are
unchanged. No neural model is trained by this experiment.

## Comparison

Four arms share asynchronous preparation and dedicated solo template profiles:
v4, the previous conservative revision, the incremental-margin candidate, and
fixed-three FIFO. The candidate removes the historical trial reserve from the
post-trial gain threshold. It retains the pre-trial exploration limit, the
nominal/no-recovery scenarios, both measurement halves, current remaining work,
and future pause/resume costs. This isolates one post-trial decision change.
Report actual trial/skip/accept/reject/inconclusive counts; unchanged exploration
guards may prevent this difference from being exercised.

The provisional plan has four conditions, both request modes and two balanced
repeats: 64 cohorts and 384 evaluation jobs. Each cohort contains three causal
parent/child chains. Individual children arrive two seconds after their parent
receipt; bracket children arrive two seconds after all parent receipts. Every
job must complete its frozen two-epoch successful-update budget.

| Condition | Three parent and child configurations | Solo training duration target |
|---|---|---:|
| complementary-short | tiny, resident, compute | 12 s per job |
| complementary-long | tiny, resident, compute | 40 s per job |
| competing-short | compute, both, both | 12 s per job |
| competing-long | compute, both, both | 40 s per job |

These names describe hypotheses. The actual resource measurements determine
whether they are supported. Tiny/resident share a width-128 residual MLP, two
shared-layer repetitions and batch 32; compute/both use width 2048, eight
repetitions and batch 256. Resident/both additionally allocate an explicit
8192 MiB CUDA buffer, excluded from checkpoints and reconstructed on every
launch. This is a synthetic persistent-memory control. Initialization is charged
to latency. Compute changes also change parameter and activation memory; they
are not perfectly independent axes.

## Calibration and frozen work

Before evaluation, run two dedicated solo replicas of each configuration, each
with 4096 successful AdamW updates over two epochs. Calibration and evaluation
have disjoint initialization and data-pool seeds. This is same-configuration
measurement reuse, not held-out neural prediction accuracy. Four class labels
alone cannot establish generalization to natural training workloads.

Each replica requires a real pause/checkpoint/resume, clean windows from both
launches, at least two warm-up updates per window, verified runtime signatures,
matching phase counters and persisted worker events. A bounded calibration-only
command requests 0.5-second windows with a 1024-update cap. Ordinary scheduler
trials retain their existing 64-update cap. Record actual window duration,
successful counts and sampling overhead; capped windows need not reach 0.5 s.
Reject calibration with no GPU/memory observations inside its training windows.

For each configuration, pooled successful window updates divided by pooled
elapsed window time yields the nominal slope. Allocated/reserved/device peaks
are aggregated across both launches; steady means use observations inside clean
windows. Admission uses a separately labeled reserved/device memory bound.
Sampled GPU busy is not SM occupancy. Record available power and memory-controller
busy separately. Window sampling and synchronization remain part of this
instrumented measurement, so slopes may differ from uninstrumented training.

The driver retains raw GPU process lists, visible process ancestry, namespace
PID aliases and process start ticks. Unknown GPU PIDs fail calibration. Verify
the mapping on the actual rented container; do not reinterpret missing host PID
visibility as proven isolation. Sampling does not exclude unseen activity
between observations.

The CPU-only study planner generates manifests on the actual host because the
runtime version participates in identity. It rounds slope × declared duration
to a two-epoch update budget in multiples of eight, bounded at 65536 updates.
All condition manifests, budgets, seeds, source identities, settings, calibration
table and arm order are frozen before the first evaluation. Report clipped
budgets and realized durations. No evaluation outcome updates templates or
budgets. The campaign must reject a plan whose status is still `candidate`.

## Common runtime amendments

All iteration-2 arms cache one deterministic sample permutation per epoch.
Previously it was regenerated before every optimizer update. CPU verification
shows identical per-update sample hashes and pause/resume state. Caching matters
for the longer update budgets; it changes runtime cost and therefore prevents
attributing cross-iteration absolute differences solely to policy.

Profiles and prepared-cache keys bind the full sealed workload specification,
configuration key, batch, source/runtime and artifact. Mutated specifications or
stale template keys cannot reuse a profile. Allocated memory and timestamped
allocator telemetry are added without changing existing reserved-memory guards.
All four arms use the same instrumentation and preparation path.

## Outcomes and bounded execution

Primary outcomes are average individual submission-to-usable-receipt latency
and average bracket submission-to-all-required-receipts latency. Report request
p95 and maximum, per-job JCT, makespan and every paired repeat. Compare
incremental versus previous revised directly, and both against v4/fixed-three
on the same condition. Preserve failed and unattempted denominators. Do not
pool short/long or different mixtures into a single headline gain.

Retain the Gantt chart above optimizer-update and VRAM curves for every cohort,
with preparation and receipt markers, plus combined comparison figures.
Allocated, reserved and device memory must remain distinguishable. Source/hash,
update/epoch, receipt/exit, profile, causal-arrival and coverage audits must pass
before interpreting rankings. Two repeats are descriptive evidence, not
production validation or a reliable tail-distribution estimate.

Run only after iteration 1 is complete and reviewed. The current candidate plan
ends at 15:30 UTC on October 30, with 600-second job/episode and 14400-second
campaign bounds. Dispatch must also reconcile the active finite lease, collector
and cumulative $20 budget. Locking and signal cleanup apply to owned processes
only. Any required bounded extension is an explicit operational change, never a
silent bypass. Iteration 3 remains a separate frozen validation stage on held-out
configurations/durations/mixtures; this document does not complete the goal.
