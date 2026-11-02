# V5.1.1 — FIFO concurrency with persistent workload experience

Status: finalized design specification and CPU-prepared experiment protocol.
The runtime integration and new GPU evaluation are not implemented by this
document. Base source: `053f0a412e2060b1ba17b25226e4082205e1ae40`.
Branch: `oct_proposed_v5_1_1`. This design supersedes V5.1's short-lived,
count-only evidence mechanism for a future opt-in policy; it does not modify the
active frozen fresh-calibration comparison.

The user's explicit FIFO/alpha experiment overrides the older repository rule
against concurrency caps for this experiment. It does not authorize neural
predictor training, promotion, model deployment, or a new spending period.

## 1. Purpose and invariants

PerfSeer describes a submitted workload. The scheduler learns how similar
workloads progress together on this GPU, and uses that experience to choose
concurrency. A large offline interference dataset is not required. Similarity
is a transfer hypothesis, not proof that jobs coexist efficiently.

- Preserve immutable submission order. Only the earliest pending job can start;
  an unprepared, held or memory-blocked head blocks younger jobs.
- Alpha is the maximum admitted residency, not an observed occupancy or a
  requested CUDA allocation. Decreasing it drains existing jobs without pause,
  reordering or checkpoint/restart. Resource release requires actual exit proof.
- Protect useful optimizer updates, recipe, batch, precision, seeds and result
  requirements. Never infer successful updates from attempted steps.
- At decision time use only already submitted jobs, declared request membership
  and observations available at that time. Unknown future arrivals are not read
  from the benchmark trace. Dependent children arrive from actual feedback.
- Individual objective: mean acceptance-to-usable-feedback time. Group objective:
  mean bracket-arrival-to-last-required-usable-feedback time. Report tails and
  makespan separately. Raw updates/s from different models are not additive
  units of useful work and are not the admission objective.
- Predictor memory is an estimate; allocated VRAM, reserved allocator memory
  and whole-device usage remain distinct. Live bounds and startup margins apply.

## 2. Components and interfaces

| Component | Input | Output and responsibility |
|---|---|---|
| Fingerprint cache | Source-bound captured training graph, recipe, predictor identity | Immutable exact key and bounded structural feature vector |
| Hardware prior | Exact device capabilities and applicable precision/backend | Initial target alpha with reason and uncertainty |
| Experience store | Successful-update windows, actual membership/phases, lifecycle events | Append-only observations indexed by workload and group context |
| Context estimator | Current job, peer mix, phase, compatible historical observations | Per-job rate interval and provenance; abstains when unsupported |
| FIFO simulator | Accepted queue, residents, remaining updates, phase estimates | Completion/feedback clocks for alpha-1, alpha, alpha+1 |
| Admission controller | Predicted objectives, confidence, live memory | Hold/increase/decrease target; never picks another queue member |
| Auditor | Immutable decisions, source keys, events and receipts | FIFO, causal-data-access, history, coverage and objective replay |

Implement these as opt-in modules (`workload_fingerprint`, `fifo_experience`,
`contextual_rates`, `adaptive_fifo_v511`) with a separate harness adapter. Keep
baseline execution closures and original alpha controller unchanged. No neural
weights or original six-output target contract change.

## 3. Workload and context fingerprints

Reuse CPU graph capture and preparation caches, keyed by full source/dependency,
recipe, converter, predictor and hardware identities. Do not repeatedly parse
source on the scheduling path. Raw source text distance is not a performance
metric; retain its hash for provenance, not as a similarity score.

The exact key binds graph nodes/edges/shapes, physical batch, accumulation,
precision/TF32 settings, optimizer, data-loader/compile/backend configuration,
resource ballast and predictor artifact. Include execution details outside the
model graph when they affect timing or memory. Missing features are explicit.

The approximate vector contains normalized operation and training-phase
histograms, graph depth/width sketches, tensor-shape size bins, parameter and
activation bytes, and log solo step/memory predictions. Memory ballast is
separate from the computational graph. Graph topology and counts must affect
similarity; equal six-output profiles alone do not establish equivalence.

Peer context contains actual resident count, number currently training, startup
and finishing counts, and a histogram/moments of peer workload vectors. Preserve
the individual target separately from its peers. This is a compact conditional
description, not a claim that the group can be represented losslessly.

Hard compatibility: same physical GPU/runtime sharing backend and versioned
schema, matching precision/optimizer/accumulation/data path. Cross-device history
is a weak static prior only and cannot authorize a rate-based change. A100
history never becomes an A10 measurement.

Proposed frozen matching rule: exact match first; otherwise equal-weight mean
of four bounded distances (graph structure, shape/configuration, solo profile,
peer mix), each in [0,1]. Admit a neighbor only if every component <=0.25 and
the mean <=0.15. Log all distances and rejected neighbors. Missing required
features cause abstention. These thresholds are design choices, not fitted
claims; changing them after evaluation requires a new version and report.

## 4. Initial alpha is hardware-aware

Use throughput for the actual eligible arithmetic path, not Tensor Core count
alone. Dense and sparse peak rates must not be mixed. Strict `fp32_ieee` with
TF32 disabled uses FP32 capability even if the GPU has Tensor Cores.

For known compatible capabilities, define the deliberately conservative prior:

```
scale = min(device_dense_TFLOPS / A10_dense_TFLOPS_for_same_path,
            device_bandwidth_GBs / 600)
alpha_hardware = clamp(ceil(3 * sqrt(scale)), 1, 6)
```

This is a predeclared heuristic, not a performance law or an optimal cap. Missing
or incomparable capabilities use a prior of 3. Mixed workloads use the smallest
applicable capacity ratio among the accepted initial prefix. Do not infer a
favorable arithmetic path solely from advertised precision or model name.

At an idle start, bound the initial dispatch target by the largest *prepared
FIFO prefix* that fits the live device budget and conservative per-job memory
reservations, then by the experimental ceiling 6. Dispatch each head through a
fresh live-memory check. An unprepared head waits; a head proven too large alone
fails explicitly. A temporary lack of ready jobs is not evidence of saturation.
Reconsider the hardware target as preparation completes; do not permanently
learn alpha=1 merely because only one profile was initially ready.

Reservations use predicted allocated peak plus a measured same-host reserved
memory/startup margin. Unknown reservation error retains conservative fallback
and is not fabricated as zero. The initial prior cannot guarantee absence of OOM.

| Example arithmetic path | A10 starting prior | A100 80GB PCIe illustrative prior |
|---|---:|---:|
| Dense FP16/BF16 Tensor Core eligible | 3 | 5 |
| Strict FP32 without TF32 | 3 | 3 |

These examples precede memory and readiness limits. A100 values use dense
312 TFLOPS FP16, 19.5 TFLOPS FP32 and 1935 GB/s; A10 uses 125, 31.2 and
600 respectively. NVIDIA specifications: [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)
and [A100](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf).
The current A10 workloads are strict FP32. This A10 study cannot establish that
4–5 is better on A100. Hardware-scaling behavior gets CPU checks only here.

## 5. Persistent experience, with separate freshness rules

Record stable training windows with exact successful-update deltas and
monotonic duration. End a window on membership, phase, restart, precision or
recipe changes. Startup, validation, checkpoint and delivery times get separate
records. Every row binds job/trace identity, hardware, source, profile generation,
observed timestamps, acceptance sequence, peer context and validity reasons.

Store raw rate and the original solo reference plus its provenance. The residual
`log(observed_rate / reference_rate)` is conditional on that reference; a
predicted reference is not measured solo truth. Do not transfer a residual across
calibration generations without an explicit, audited re-expression from raw
measurements. Never rewrite old observation timestamps.

Three scopes are distinct:

1. Active-group evidence: exact membership/phase, expires after 60 seconds or a
   context change. Two clean windows of >=1 second and >=4 successful updates
   per training member are required for a current-group estimate.
2. Persistent history: survives empty queues and process restart; bounded to
   4096 episode-balanced records per namespace, retained at most 24 hours, with
   weight `similarity_weight / (1 + age_seconds/3600)`. It supplies a prior.
   An episode cannot contribute thousands of independent votes: at most one
   aggregate per workload/context/phase/occupancy per episode enters retrieval.
3. Completion-phase calibration: retains the existing 4-hour TTL and real
   boundary-refresh protocol. History retention does not extend this TTL.

An approximate estimate needs at least three independent episode aggregates;
use at most 16 nearest compatible aggregates. Weighted median log residual is
the point estimate. Weighted 10th/90th percentiles expanded by a minimum 20%
rate margin form an *empirical uncertainty envelope*, not a calibrated confidence
interval. Missing, inconsistent or older-than-4h evidence cannot alone authorize
an increase. Exact fresh current-group evidence takes precedence. Large errors
on two consecutive valid windows demote that context to exploratory status.

Append-only event records and atomic index snapshots bind parent-state hashes.
Recovery preserves FIFO order, pending exploration and workers' exact births;
it never infers worker exit from absence. Empty busy periods retain compatible
history and the learned context prior instead of resetting all experience.

## 6. Decision and exploration contract

Compare current alpha and adjacent targets in [1,6]. Simulate FIFO drain/refill
over already accepted jobs using per-job rates and phase costs. Existing
residents remain after a target decrease. Model validation/finish/delivery even
when successful optimizer updates are complete. Respect actual bracket
membership and resource release. Unknown necessary evidence produces `hold`.

Evaluate the proposed change under both evidence halves and pessimistic rate
envelopes. Require at least max(1 second, 5%) predicted objective improvement
after estimated decision overhead; retain 10-second dwell. These are inherited
conservative thresholds, not tuned from previous latency winners. Do not select
an increase because one small model has a larger absolute update rate.

When a higher occupancy lacks evidence, permit an explicitly exploratory +1
admission only for a prepared FIFO head, available memory, stable productive
incumbents and remaining useful work longer than startup + observation +5s.
At most one unresolved exploration, one per new context/target and six per
independent trace. Further increases require a valid observation at the new
occupancy. Record adverse outcomes and debt even after empty queues. Raising
alpha can hurt latency; lowering it only limits subsequent admissions.

No claim of counterfactual correctness follows from a before/after rate change:
phases and membership can confound it. Online data are evidence for the model;
fresh complete baseline traces evaluate actual policy benefit.

## 7. Bounded observation and preparation cost

Cache fingerprints during charged asynchronous preparation. Update cheap
counters at most once per second and score on meaningful events (ready head,
membership change, phase boundary or fresh evidence), with minimum 2-second
spacing between non-safety decisions. Emergency memory checks remain independent.
Use the existing shared GPU telemetry snapshot; never run a new NVML/subprocess
query inside every forecast. Treat unavailable/old telemetry as unknown.

Bound retrieval to 16 neighbors, 64 active/queued simulated jobs and 256 events;
larger queues use a conservative hold and retain progress. A 10ms elapsed deadline
covers retrieval plus scoring after cached inputs are available, with checks
between bounded operations. Source capture and telemetry costs are separately
charged and reported. A deadline is not a hard real-time guarantee under OS
descheduling. Measure the entire controller path, not just inner simulation.

GPU kernel-busy percentage does not establish compute or bandwidth saturation.
Use stable update rates and, when supported, appropriate compute/DRAM counters
as context. Early initialization and the first layer are not a sustained sample.

## 8. Scope of empirical claims

Start each independent trace with empty online history; carry history through
its successive episodes. Different policy arms and repeats never share learned
records. This is distinct from immutable predictor/calibration inputs shared
across arms. Historical evaluation logs may be used as declared development
diagnostics, not silently imported into the cold-start primary history.

The prepared pilot repeats five established workload recipes in two learning
cycles with fresh, disjoint seeds. It tests within-template reuse and context
changes. It does not establish generalization to unseen architectures. Cycle 1
includes learning; cycle 2 measures reuse. Count learning and exploration in the
whole-trace objective and total cost. A later warm cycle is not an independent
repeat. Historical controls do not become contemporary paired samples.

See `a10-v5-1-1-experiment.md` and the CPU-only preparation package. Required
runtime work, tests, protocol limitations and launch gates are explicit there.

Research precedent: [Gavel](https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf)
uses reference-job fingerprints and sparse profiling for throughput estimation.
That motivates reuse of observations; it does not validate this matching rule,
hardware heuristic, FIFO policy or its latency benefit.
