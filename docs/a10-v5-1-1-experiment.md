# V5.1.1 A10 preparation and evidence reuse

Status: implemented runtime and deterministic guarded screening manifest.
No new GPU campaign has been dispatched at this implementation freeze. The currently
running `053f0a4` fresh-calibration campaign remains authoritative for its own
results. V5.1.1 uses a new branch and separate output namespace.

## Decisions

Keep published S_A10 weights, original labels, five workloads, optimizer/update
budgets, two request modes and strict FIFO. The tested change is persistent,
workload-conditioned experience with bounded alpha control and lower controller
overhead. A10 starts from a hardware prior of 3; memory may reduce initial
dispatch. The A100 prior is specified and CPU-tested, not GPU-validated.

We can reuse immutable workload assets, source bundles and hash-compatible
CPU predictor conversions. Prior timing measurements remain historical controls.
New physical hardware requires qualification and fresh calibration. Prepared
graph caches have exactly the same cold/warm policy for contemporaneous arms.

## Default: economical screening with explicit historical comparisons

Run four independent V5.1.1 traces: two modes x two repeats. Each starts with an
empty experience store, runs all five conditions, then repeats the five in a
second cycle with disjoint seeds. History persists across conditions, calibration
refresh and empty intervals; fresh validity checks continue to gate actions.
No import of previous primary outcomes into online history.

Each cycle has 36 jobs (four six-job conditions plus one twelve-job backlog).
Candidate denominator: **40 cohorts /288 jobs**. There are four independent
traces, not 40 independent learned-policy repetitions. Cycle 2 is a warm-history
segment inside its trace. Repeat 2 reverses the five-condition order to expose
order effects. Episodes are separated by the same deterministic two-second
idle interval after the prior episode has fully drained. This episodic trace
does not establish performance under continuous external arrivals.

Run fixed-three anchors before and after the screening campaign:
short-mixed and backlog-six, both modes = **8 cohorts /72 jobs**. Total screening
denominator **48 cohorts /360 jobs**, excluding qualification/calibration and
development, whose denominators must also be reported. Anchors do not train
candidate history. Record their complete cost, including preparation.

The previous fixed-three, V5 and V5.1 results can be shown alongside the new
candidate with original dates/host/repeat labels. Such a candidate-only policy
comparison is **historical screening**, not proof of a causal improvement or
two fresh paired repeats. Passing anchors does not turn it into a fresh primary.
Do not subtract anchor drift from old latency values to manufacture controls.

Reuse gate: complete SHA-verified baseline export and exact critical code,
recipe, source, seeds, arrival rules, budget, predictor, calibration, runtime,
instrumentation and hardware identity. Missing fields or a mismatch downgrade
to contextual use. The already completed C primary used a different GPU; its
timing baseline currently qualifies only as historical context. The active
refresh study has not completed final export and is pending, not reusable-final.

## Pre-execution implementation amendment

The repeat-two development width variants need their own measured templates.
Preparation therefore measures **16 solo calibration jobs** (8 base and 8
variant), and runs **24 development cohorts /192 jobs**. Subsequent screening
refreshes measure 8 base jobs per generation. Development has a 7200-second
campaign limit with a 7300-second outer limit; preparation and parent review
reserve 21600 seconds. Screening retains 43200 seconds, jobs 1800 seconds,
episodes 3600 seconds, recovery 900 seconds and the original finite host cutoff.
These bounds are frozen before new outcomes, with no runtime lease extension.

The separate controller uses only submitted jobs. Its experience is cold at
each of four independent traces and persists through that trace's idle periods
and real calibration refreshes. Every first dispatch records the fresh device
memory reading, resident reservations, candidate reservation and limit. The
memory reservation is the larger of the measured startup envelope and the
predicted peak plus max(256 MiB, 10%); this is a guard, not a reserved-VRAM label.
Uncertainty, lack of similar observations, and the cost of graph preparation can
prevent an improvement. Report them rather than changing frozen thresholds.

Stability rule, frozen before candidate results: compare same-condition/mode
anchor means to compatible historical means; flag >10% change in either
direction. Also report tails and every individual observation. When exact seed,
calibration or host matching fails, record `not_comparable`, not `stable`.
The small anchor sample does not establish tail equivalence. Failure of any
compatibility/stability gate requires fresh controls for a performance claim.

## Fresh comparison when a claim of improvement is required

Prepared alternative: full fixed-three and V5.1.1 traces with identical recipes,
seed schedule, mode and repeat, **80 cohorts /576 jobs**. Counterbalance which
complete arm trace runs first within mode/repeat. Reset candidate history only
between independent traces. Baseline never receives candidate observations.

Use a common fresh calibration generation for each paired trace *episode index*
where possible, but do not extend a calibration lifetime to force it. Because
whole traces run sequentially, exact generation matching may be impossible.
Prefer paired episode execution with a distinct persistent per-arm history
namespace, identical predefined inter-episode idle policy and shared generation.
That execution interleaves arm episodes; its elapsed idle/aging includes the
other arm's work, is logged and reported. Choose and seal this interleaved
protocol before dispatch; never merge it with uninterrupted screening traces.
The prepared confirmation manifest uses this interleaved protocol.

Per matched episode block: fixed-three and V5.1.1 share the same fresh phase/solo
calibration generation. Reserve both complete episode bounds +300 seconds under
the unchanged 4-hour phase TTL. Refresh only at block boundaries. The existing
three-arm refresh module cannot be reused unchanged for this two-arm protocol;
an audited block-width adapter is a required implementation gate.

V5 can be added as a third fresh arm before sealing if required: **120 cohorts
/864 jobs** across all three arms. Then retain three-arm freshness reservations
and counterbalanced order. This is an alternative, not an extra campaign to run
automatically after screening. Historical V5 results suffice for descriptive
context but cannot substantiate a new paired V5 comparison.

## Measurement and inference

Primary: equal-condition-weight mean request feedback latency per mode over the
entire trace, including its learning cycle; report each of two paired trace
repeats. Secondary: each condition/cycle, p95/max, makespan, total campaign time,
refresh time, controller CPU/wall time, profile preparation and head blocking.
Do not use changing condition proportions to imply a learning curve. Compare
cycle effects only for matched conditions against their controls.

Record initial alpha and its hardware/memory reasons; target vs occupancy;
history matches, exact vs approximate reuse, neighbor provenance/distance;
prediction error; uncertainty, demotion and abstention; exploration successes
and harm; decreases/drain intervals; cache hit rates and controller timeouts.
Zero adaptation is a valid outcome, not a reason to retune. These full-policy
comparisons do not isolate the benefit of history from preparation/overhead
changes; a separately predeclared history-disabled ablation would be needed.

All receipts need exact successful updates, two epochs, source/split/profile
identities, valid clocks, worker exit and FIFO audits. Preserve failed and
unattempted jobs. At time t, retrieved observations must precede t. Replay
history mutations using source-bound events; never run an offline history
builder over the complete trace and feed future data into earlier decisions.

The recipes/seeds in the prepared manifest use new ranges, but recipes and
earlier outcomes informed this design. Label the experiment a prospective
evaluation of a design informed by earlier studies, not pristine validation of
an untouched research hypothesis. Structural-neighbor CPU fixtures and separate
development shape variants check matching behavior, not unseen-model accuracy.

## Implementation and launch gates

1. Implement graph fingerprint extraction/cache, bounded history store and
   deterministic retrieval, rate estimator, hardware prior and FIFO controller;
   wire an opt-in episode adapter and persistent trace state. Baselines remain
   pinned by executable-dependency hashes.
2. Test FIFO head blocking; hardware-path selection; no unobserved counterfactual
   labels; no future history access; repeated-window deduplication; stale-phase
   rejection despite retained history; memory estimate/live gate separation;
   decrease/drain; exit/birth identity; restart/recovery; bounded lookups and
   source capture; calibration block-width semantics; no cross-arm/repeat leaks.
3. Freeze source/settings/protocol/hash-bound graph format. Run all existing CPU
   checks plus new checks, real S_A10 captures and operations tests. Run exact-
   host qualification, eight disjoint dedicated calibration jobs and released
   S_A10 conversion. Fresh preparation never relies on A100 or old-host labels.
4. Separate development: four traces (two modes x two seeds), each short-mixed,
   backlog-six, published-pipeline twice = **24 cohorts /192 jobs**. The second
   development seed includes declared shape variants to exercise approximate
   matching. Independent canonical review must inspect valid FIFO and experience
   replay without requiring favorable latency or forced adaptation. Development
   data never seed the cold-start evaluation store.
5. Verify current owned inventory, current campaign exit/GPU idle and SHA-verified
   recovery before sharing/replacing a rental. Bind actual GPU, runtime, source,
   plan, fresh calibration, review and lease. Dispatch once using exclusive
   intent; verify actual command, birth, ancestry and progress.

The CPU preparation checker deliberately refuses to label this plan launch-ready
until a future runtime/package review provides all required gate evidence.

## Same-period resource limits

Budget authority remains `work/a10-v5-redesign/budget.json`: period 2, original
cutover 1793488910.6600087, cap $30, trigger $29.50, $1 common cleanup/transfer
reserve. Do not copy it into a new spending authority. At preparation, current
full-lease reservation was $13.0013, leaving about $16.9987 unreserved against
the cap. This is a point-in-time envelope, not actual billed expenditure.

One additional 24h A10 lease at a conservative $0.30/hour plus $0.25 incremental
transfer allowance would reserve $7.45. Re-read the live ledger and fresh billing
before any action. Do not infer affordability from cached numbers or reset old
costs. The current guard permits one specific refresh rental; another owned
lease needs a reviewed operational continuation preserving the closed ledger,
all clocks, $30 cap and increasing billing floors. This is an engineering gate,
not a request for renewed human rental permission.

Screening campaign bound: 43200s. Two-arm confirmation: 70200s. Job: 1800s;
episode: 3600s. A bound is a stop condition, not a promise that all denominators
finish. Use actual remaining lease headroom before dispatch. Declared maximum
preparation/review stages total 14400s, recovery 900s, and destructive-cleanup
margin 900s. A fresh 24h lease can cover confirmation's 70200s plus those bounds
exactly; stage/dispatch delays reduce available time and can block launch.
Prefer a shorter separately frozen campaign if the entire declared protocol
still has defensible feasibility; never silently shorten while claiming full
coverage. Do not extend an existing lease implicitly.

A current-host continuation avoids new hardware drift only if all work remains
sequential and finite bounds permit it. Otherwise use a fresh qualified A10 and
collect fresh controls for confirmation. Preparation is not a rental reservation.

Export reviewed Markdown/JSON plus combined Gantt, update, VRAM, alpha, history
and freshness plots under `outputs/a10-v5-redesign/v5.1.1`. Preserve immutable
baseline packages separately. Recover and SHA-verify important artifacts before
exact-owned cleanup; verify provider absence and final period cost. The active
refresh automation continues to own its current campaign's analysis and cleanup.
