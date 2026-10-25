# oct_proposed: completion-aware backfill with aging

This branch is a reviewable research proposal. It contains no alternate running
scheduler, does not switch or overwrite the `paper` working checkout, and does
not deploy a service. The implemented `busy_anchor` policy remains the requested
starting point. These policies need a controlled comparison before either is
claimed to improve performance.

## Motivation and limits of the requested heuristic

The requested policy starts a high-NVML-busy anchor, adds low-busy work, and
replaces the anchor when it finishes. This can help test whether complementary
work fills idle intervals. Its busy signal measures kernel-active time, not
unused FLOPs or additive capacity. NVIDIA defines this metric by the fraction
of a sample interval during which kernels execute; two jobs with low solo busy
can still contend for bandwidth, CPU input or the same compute resources.
[NVML documentation](https://docs.nvidia.com/deploy/archive/R550/nvml-api/structnvmlUtilization__t.html).

Heavy-first also has no mean-completion-time guarantee. Consider four jobs
submitted together that cannot safely overlap: one heavy 60-second job and
three light 1-second jobs. Heavy-first completes them at 60, 61, 62 and 63
seconds (mean 61.5); short-first completes them at 1, 2, 3 and 63 seconds (mean
17.25). Both drain in 63 seconds. This is an analytical counterexample, not an
experiment or a measured speedup, and shows why makespan/busy alone are
insufficient objectives.

[Gandiva](https://www.usenix.org/conference/osdi18/presentation/xiao) motivates
learning from repetitive iterations and measured job/resource fit rather than
assuming resource demand from a single utilization number.
[Tiresias](https://www.usenix.org/conference/nsdi19/presentation/gu) targets mean
JCT even when duration is uncertain, using partial-information and attained-
service scheduling. [Themis](https://www.usenix.org/conference/nsdi20/presentation/mahajan)
shows why finish-time fairness deserves an explicit measure alongside efficiency.
Their settings are GPU clusters; the policy below is our single-GPU proposal,
not a reimplementation of those systems.

## Proposed policy

1. Keep the existing priority aging and starvation override. Apply the same
   approved batch envelope, branch-profile VRAM, live memory gate, backend and
   pair compatibility, 31 GiB ceiling, and live slowdown trials as `busy_anchor`.
   Never add a fixed maximum parallel job count.
2. When idle, choose the smallest valid predicted remaining duration inside the
   current fair priority band. If unavailable, use measured same-hardware
   branch timing; if all durations are missing, use oldest submission first.
3. With active jobs, estimate each feasible addition's piecewise completion
   times using trusted exact colocation rates, or use the existing live trial
   when rates are unknown. Prefer the greatest reduction in the sum of
   projected completion delays across the active members plus newcomer.
   Preserve incumbent slowdown bounds and the current drain-gain safety rule.
   Busy becomes a tie-breaker or a trial-order hint, not a capacity constraint.
4. Re-evaluate after membership changes. Retain active jobs and their training
   progress; do not add a preemption/restart mechanism in this first version.
   A later bounded-preemption variant must account for checkpoint/restore cost.
5. When uncertainty prevents scoring, use duration/submission ordering with
   aging and measured trials, and record the fallback. A100 model predictions
   on V100/5090 cannot supply calibrated timing without a validated transfer;
   the explicit cross-hardware busy experiment is only a hint.

One reviewable scoring candidate is
`sum_i w_i * (C_i_without_addition - C_i_with_addition)` for current jobs and
newcomer, where `C` is projected finish time from the current decision boundary
and `w_i` comes from the existing aging policy. Positive score means a predicted
reduction in aggregate waiting-to-completion. Always report the unweighted
metrics too: a weighting choice must not hide harmed jobs. Confidence/slowdown
terms should be calibrated on validation traces and fixed before test traces.

## Evaluation plan

Use the same arrival traces, hardware, runner, approved physical batches, seeds,
checkpoint settings, profiling warmup, memory budget and concurrency mechanism
for all arms. Compare current `baseline`, requested `busy_anchor`, and the
proposed completion-aware policy. Include FIFO as a transparent reference where
supported. Do not change training quality or admit extra memory for one arm.

Use workload mixes with long/short duration, high/low busy, compute/bandwidth/
input-bound work, staggered arrivals, nearly identical jobs, missing predictions,
biased timing, changed batch size and late-arriving starvation cases. Add an
anchor-completion case with surviving light jobs. Separate cold-profile and
warm-profile runs; hold out architecture families and realistic arrival traces.

Measure mean and median JCT from submission, p95/p99 JCT, makespan, per-job
slowdown relative to equivalent solo training, max wait, starvation incidents,
training progress/quality, rejected trials, trial/checkpoint overhead, fallback
rate, VRAM peak/OOM, and NVML busy. Report FLOPs or occupancy only if independently
measured with a suitable tool, never inferred from NVML busy. Report subgroup
results for duration and hardware; do not pool unlike hardware to mask transfer
failure.

Start with deterministic CPU trace replay using recorded branch profiles and
synthetic edge cases. Test stale predictor receipts, missing step semantics,
accumulation, invalid predictions and completion/restart ordering. Only after
review run a separately authorized live comparison; repeat seeds and report
uncertainty. Save every trace in the repository. For each live experiment,
produce the required combined Gantt-over-metric-node figure; put all comparison
arms in one PNG. No such GPU experiments have been run for this proposal.

Adopt the alternative only if its improvement in mean JCT is robust without
unacceptable p95 slowdown/fairness regressions, memory failures or quality loss.
Keep `busy_anchor` and baseline selectable regardless of the result. A negative
result is useful evidence, not a reason to retune against the test set.

## Publication integration check

This branch also includes the optional released `S_A100` CPU adapter and its
[actual-model integration smoke test](a100_student_integration.md). This adds
compatible model loading and scheduler estimates; the completion-aware policy
above remains a proposal and has not been deployed or benchmarked.
