# V5.1: strict FIFO with adaptive concurrency

Status: experimental policy and comparison design, not a demonstrated latency
improvement. This branch starts from V5 commit
`a44a61a69d8fa92953926c0e3e050e7c1dc5c803`. Predictor weights, labels, splits,
training recipes, batch sizes, optimizer-update budgets and original V5 behavior
stay fixed. The new policy changes how many jobs may be resident, not which
waiting job goes first.

## Objective and exact FIFO contract

Let job i be accepted at a_i, return its usable result at C_i, require K_i
successful optimizer updates, and have u_i committed updates. Remaining work is
R_i = K_i - u_i. A running job with R_i = 0 still occupies its slot until its
process has actually exited and GPU resources have been released. Result delivery
is included in C_i even when it no longer holds the GPU.

Each accepted job receives an immutable acceptance ordinal. Among waiting jobs,
only the oldest may be dispatched. An unprepared or temporarily memory-blocked
head blocks younger jobs. A job that cannot fit alone fails explicitly; it is
not silently skipped. Within a bracket, the submitted list supplies the order.
FIFO concerns controller dispatch, not completion order or the first observed
optimizer-update timestamp: asynchronous model initialization can differ.

The controller maintains an integer target alpha, initially 3 to match the
fixed-three reference's starting point. Starting, training and finishing workers
all count as resident. Increasing alpha permits the next FIFO job to start when
memory permits. Decreasing alpha stops refills until enough workers finish; it
does not pause, evict, restart or reorder existing jobs. Thus actual occupancy can
temporarily exceed a reduced target. There is no additional permanent cap of
three or four: accepted work, head-of-line placement and VRAM bound concurrency.

The two modes use different request objectives:

\[
J_{\mathrm{individual}} = \frac{1}{N}\sum_i(C_i-a_i),
\qquad
J_{\mathrm{bracket}} = \frac{1}{B}\sum_b
  \left(\max_{i\in\mathcal B_b} C_i-a_b\right).
\]

Each bracket has one vote, regardless of its size. Every required member counts,
including queued, preparing and delivering jobs. Unknown required membership
makes a forecast inconclusive. The controller can see only accepted jobs; it
cannot inspect future feedback-dependent children from the experiment manifest.
Average and arithmetic mean name the same metric here. Tail latency, makespan,
failed requests and incomplete denominators remain separate reported outcomes.

## Why concurrency needs a latency objective

Consider two equal jobs with solo duration T, no overhead and identical arrival
times. Serial FIFO has average JCT 1.5T and bracket latency 2T. If their concurrent
per-job rate is q times their solo rate, concurrent average JCT and bracket
latency both equal T/q. Individual mode benefits only for q > 2/3, whereas bracket
mode benefits for q > 1/2. With q = 0.6, concurrency improves bracket latency from
2T to 1.667T but worsens average JCT from 1.5T to 1.667T. These are synthetic
calculations, not A10 results. Aggregate throughput or GPU utilization alone
cannot choose correctly for both modes.

## Research and alternatives considered

- [Gandiva](https://www.usenix.org/conference/osdi18/presentation/xiao) uses the
  repetitive structure of training iterations to inspect performance and adjust
  sharing. Its scheduler also uses framework mechanisms such as time-slicing and
  migration; V5.1 borrows runtime feedback, not those implementation guarantees.
- [Salus](https://arxiv.org/abs/1902.04610) supplies fast switching and memory
  sharing through a consolidated execution service. Our independent CUDA
  processes do not acquire those primitives merely by changing a concurrency
  limit. V5.1 therefore uses natural completion for ordinary decreases.
- [AntMan](https://www.usenix.org/conference/osdi20/presentation/xiao) coordinates
  memory and computation through framework/scheduler co-design. This reinforces
  the need for both a memory guard and interference measurements; spare VRAM
  alone does not establish that another job improves latency.
- [Pollux](https://www.usenix.org/conference/osdi21/presentation/qiao) models
  useful training progress while adapting resources and training configuration.
  Here recipes and required updates are frozen, so successful updates provide a
  work counter; V5.1 does not alter batch size or claim an accuracy/goodput gain.
- [Netflix concurrency limits](https://github.com/Netflix/concurrency-limits)
  illustrates feedback control, bounded adjustments and latency-sensitive
  backoff. Raw request latency is unsuitable as our immediate feedback signal:
  training jobs differ in size and complete too slowly. V5.1 uses update progress
  to forecast request latency instead. Its FIFO queue retains work rather than
  rejecting overload as an RPC service might.

Three candidates were considered. Pure AIMD on utilization or throughput is
cheap but misses the mode-specific objective. A contextual bandit over alpha
needs more independent observations, has delayed rewards, and can confound job
mix with concurrency. A bounded adjacent-cap FIFO forecast is the first
implementation: it directly represents the requested objectives, has a small
action space and avoids V5's branching continuation search. This is a reasoned
choice to test, not a proof of global optimality.

## Observations and the interference approximation

The nominal solo prior remains

\[
\widehat s_i^0=1000/\widehat t_{i,\mathrm{wall\ step\ ms}}.
\]

It is an estimate, not a new measurement or accuracy improvement. All times in
the forecast use compatible physical seconds; unlike the earlier normalized
throughput ratio, this latency forecast is not invariant to arbitrary rescaling
of uncalibrated solo times. Observed solo windows can correct the common scale. The existing
six-output predictor is sufficient for this experimental interface; no new
training head is required. Runtime measured-template controls and the released
S_A10 pipeline remain separate strata.

Collect two nonoverlapping wall-time halves with the same resident membership,
process/recipe identity and non-update phase counters. Each half needs at least
one second and four successful updates per member. Reject stale, nonpositive or
nonfinite rates; an epoch validation/checkpoint or startup crossing invalidates
the pair. A short job may complete before a pair exists: retain its completion
and progress and call its rate evidence inconclusive.

For each half h at observed occupancy k, form a count-level factor

\[
\widehat d_h(k)=\operatorname{median}_{i\in S}
    (r_{i,h}/\widehat s_i^0).
\]

Keep the original measured vectors and identities. Use exact current-group
rates while that membership remains current. For future or different groups,
solo priors multiplied by count factors are predictions. An unobserved count
uses constant aggregate throughput as an explicit extrapolation. For a lower
unobserved count, cap the predicted per-job factor at a fresh observed solo
factor, or at the nominal solo factor one when no solo observation exists. This
avoids inventing super-solo acceleration from a measured packed group; the
nominal anchor is still a prediction, not a physical upper bound. Never relabel
this extrapolation as a measurement of an unobserved pair. Factors inherit solo-prior error and ignore some job-mix
interference. Two halves test consistency; they are not independent experiments
or calibrated confidence bounds. Caches expire and are not shared across
evaluation cohorts or hosts.

The harness retains the existing four-hour phase-model lifetime. Calibration
timestamps are never refreshed merely by reading a profile. Once those phase
priors expire, forecasts can become inconclusive and ordinary FIFO service
continues. Report stale-phase holds and elapsed calibration age separately;
later cohorts may provide timing evidence without demonstrating adaptation.

## Complete control loop

1. Reap actual exits, receive usable results, update successful work and retain
   acceptance order. Release GPU capacity on exit proof, not an R_i = 0 counter.
2. Continue bounded CPU profile preparation in the background. Preparation may
   finish out of order; GPU dispatch may not. Use two preparation workers for
   bracket mode and one for individual mode, with ownership-verified cleanup.
3. Refill in FIFO order up to the current target, subject to the unchanged
   memory guard. Missing forecast information does not stop this ordinary path.
4. On fresh valid evidence and after the dwell interval, score alpha and its
   adjacent targets. Run a deterministic FIFO event simulation from the same
   current state for every alternative. Model startups, successful-update work,
   resource-holding finish, release and result delivery. A reduced target retains
   every current worker until its natural release. Use the head-only memory
   predicate, and exclude unsubmitted future jobs.
5. Compare the appropriate full request objective under each half's rate model.
   Change by one only if both models favor it by at least 5% and one second.
   Ties, disagreement, invalid data or the CPU/event limit mean hold. These
   thresholds are provisional research settings frozen before evaluation.
6. To learn an unseen higher occupancy, allow one productive admission of the
   FIFO head at alpha + 1 after fresh evidence and the dwell interval, when the
   head is ready, memory-safe, and enough estimated running work remains to
   observe it. Permit at most one such exploration per new count in a busy
   period. Record exploration separately from predicted-gain decisions. The
   admitted job continues even if later evidence favors reducing alpha.
7. Observe the new state. Invalidated halves restart measurement, not training.
   Once the queue and workers drain, reset the episode-local learning state for
   a new busy period. Never erase successful updates or silently replay a job.

Initial settings: alpha = 3; one-second/four-update halves; ten-second dwell;
60-second rate lifetime; at most 256 forecast events and a ten-millisecond
elapsed-wall guard around forecasting per decision. This bounds elapsed
forecast computation, not the full observation loop; total observation wall
time and thread CPU time are recorded separately. Exploration needs estimated time to the next
completion exceeding two measurement halves plus head startup plus five seconds.
There is no latency-safety guarantee for an exploratory admission: interference
is unknown and drain-only backoff cannot undo an already running job. Memory
safety checks, job timeouts and ownership cleanup remain independent.

The forecast is an approximate fixed-alpha continuation; execution reconsiders
alpha as evidence arrives. Post-exploration forecasts are not observations of
what the unchosen action would have achieved. All elapsed preparation, control,
exploration and checkpoint work is charged once in observed end-to-end latency.

## Implementation and verification boundary

The opt-in controller is separate from the V5 service. The new three-arm
experiment delegates fixed-three and combined V5 to the original episode path;
V5.1 uses the same jobs, predictor preparation infrastructure, subprocess runner,
receipt parser and memory predicate, with a FIFO dispatch loop. Different
preparation concurrency is reported as a whole-system treatment.

Required checks include FIFO with an unprepared/memory-blocked head and equal
acceptance timestamps; drain without eviction; zero versus unknown remaining
work; actual exit before refill; retained trial/resumed updates; startup/phase
and membership invalidation; short-job success without measurement; missing
predictions; full bracket maxima; forecast timeout with ordinary progress;
bounded exploration; restart invalidating rate evidence without duplicate
dispatch; and numerical simulation/executor agreement on drain/refill traces.
Persistent state must preserve acceptance ordinals, alpha and exploration debt;
a recovered controller cannot infer that an unobserved worker exited.

Initial local verification: all 396 selected CPU tests passed with zero skips,
including the original 347 checks, real released S_A10 source captures and 49
new controller/harness/protocol/cleanup checks. Independent critical review
found no remaining core blocker. These are software checks; exact-host
qualification, fresh calibration, six development episodes and primary GPU
results remain separate gates.

## Revised A10 comparison

The primary arms are `fixed_three`, `combined` (V5) and `v5_1`. Five unchanged
workload conditions, both request modes and two repeated blocks yield 60 cohorts
and 432 jobs per participating host. Run all three arms within each host block;
never allocate one policy to one GPU. Counterbalance arm order and retain
machine/GPU/runtime/calibration identity. The original running V5 ablation stays
immutable as diagnostic evidence. New primary results do not splice its partial
outcomes into the revised comparison.

Before any new GPU evaluation, freeze source/settings/blueprint/controller
defaults, run exact-host CPU checks and collect eight fresh dedicated solo
calibration jobs on that host. Run six separate V5.1 development episodes:
short-mixed, backlog-six and published-predictor workloads in both modes,
covering 48 jobs with disjoint development seeds. Replay their raw evidence and
review it before the primary comparison. CPU regression tests cover head
blocking and drain behavior; GPU activation remains descriptive, not a required
performance result. No thresholds are tuned using evaluation outcomes. Reused seeds
after earlier correctness repairs make this corrected replication, not pristine
untouched validation.

Report average individual JCT or average bracket feedback latency, p95, maximum,
makespan, each within-host paired difference, coverage, alpha versus actual
occupancy, FIFO head blocking, exploration, inconclusive measurements, forecast
CPU time, preparation and checkpoint overhead. Complete receipt/update/epoch,
source/profile and causal-arrival audits remain mandatory. Zero adaptation can
still produce valid end-to-end timing, but cannot prove the adaptive mechanism
helped. Incomplete requests cannot disappear from a latency claim.

A bracket containing only three parents, followed by a barrier and three
children, will generally retain alpha = 3: every parent starts before clean
measurement windows exist, there is no queued head to explore, and drain-only
reductions cannot change already running work. The empty busy period then
resets the cap. Backlogs larger than three and overlapping individual arrivals
offer more opportunity for adaptation. Report this expected null activation
instead of changing the workload or forcing a cap change after seeing results.

The user increased the existing period-2 A10 allowance from $20 to $30. This is
one shared cap with the original October 31 cutover and all accrued liabilities
retained, not a reset or $30 per GPU. Exact finite leases, cleanup reserve,
artifact recovery and provider-confirmed destruction remain mandatory. The
authoritative ledger and operational handoff track dispatch separately from this
design document.
