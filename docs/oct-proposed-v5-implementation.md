# V5 latency redesign: implementation and A10 ablation

This experimental branch implements the continuation, deferral and bracket plan
from the completed A10 diagnostic study. Its entry point is
`scheduler.experimental_v5.V5Service`, configured with `V5Options`.
The production entry point and original predictor artifacts are unchanged.

## Independent switches

| Switch | Change | Main correctness boundary |
|---|---|---|
| `continuation` | Event-driven request forecasts and the same reference continuation selector during execution | Full known cohort; exact group evidence; resource release before refill; no fabricated future requests |
| `deferral` | Persistent next-service reservation after 30 runnable-deferred seconds or two younger launches | Original FIFO and existing 1,800-second guard; legal solo fallback; no incumbent preemption |
| `bracket` | Ready, sealed group bootstrap on an idle GPU; finishing lifecycle; bounded parallel neural preparation | At most three jobs and configured cap; measured startup envelope; two clean full-group halves; retained trial progress |
| `shadow` | Record the new forecast while v4 remains authoritative | Forecast errors describe only the executed path; counterfactual outcomes are unavailable |

A missing required prediction prevents packed admission. Valid zero remaining
updates means finishing, not missing evidence. A successful worker exit without
a usable receipt means delivering, with no GPU reservation. Receipt publication
is the request-completion event.

## Continuation and evidence

The simulator advances through startup, training, pause, finish, GPU release and
delivery events. It scores every already submitted request, allows at most six
candidate inspections, and caps each comparison at 128 events per rollout and
50 ms foreground forecast CPU time. Exceeding a bound is inconclusive.

The execution reference uses the same FIFO/reservation selector and placement
predicate, including memory, compatibility, configured concurrency, failed groups
and retry restrictions. New arrivals or material identity changes invalidate the
frozen continuation and return to online planning. This is an explicit reference
forecast, not a guarantee that all future arrivals or online decisions are known.

Only exact group measurements with compatible frozen profile, hardware, backend,
process generation and original measurement timestamps can support packed
refills. A surviving pair cannot inherit a triple's rates. An unequal three-job
bootstrap can therefore become inconclusive when its first completion leaves an
unmeasured pair. Such a result follows bounded cleanup/solo fallback and is
reported as unknown, never a measured acceptance.

Both halves use the same current remaining updates. Already completed trial
updates are credited. Future pause/resume costs enter the alternatives; the
elapsed trial is counted once in observed end-to-end latency. Steady training
rates do not independently identify startup/finishing interference.

## Deferral and completion

Deferral records survive checkpoints and restarts. Small amounts of trial
progress do not erase debt. Protection filters the planner's actual candidate
view, avoiding an idle-GPU veto loop. Held, unprepared or solo-infeasible work
cannot block unrelated eligible jobs. Once protected, service may still wait for
a running resident to finish; this is not a completion-time SLA.

Finishing and bootstrap cleanup barriers retain their original deadlines across
restart. Missing a worker in a new supervisor is not proof that its resources
were released. Unknown ownership or ignored pauses end in an explicit bounded
failure, with owned-process cleanup by the experiment controller. Completed jobs
are never extended or paused to manufacture another timing half.

## Bracket preparation

Measured-template controls use the same inexpensive asynchronous lookup in all
six arms. The C/combined neural bracket path uses two independent CPU predictor
processes. Worker loading and source capture occur after acceptance and are
charged to latency. Capture timeouts remain frozen; the wrapper explicitly adds
bounded model-initialization allowance. Shutdown verifies owned process births
and descendants, rather than treating a returned PID or thread exit as cleanup.

A learned peak prediction alone does not establish simultaneous startup memory.
The synthetic measured condition supplies a same-host peak-device envelope with
headroom. The neural condition leaves an unsupported startup envelope unknown
and uses the ordinary safe admission path. No neural weights are retrained.

## Frozen evaluation design

`benchmarks/a10_v5_ablation/blueprint.json` declares six arms:
`v4`, `fixed_three`, `forecast_only`, `deferral_only`, `bracket_only`, `combined`.
Every GPU-instance block runs all arms in each condition and mode. Instance,
physical machine, software, calibration and source identities are retained.
Instances are not treated as independent physical hosts if they share a machine.

Five predeclared conditions cover short mixtures, longer mixtures, memory-limited
pairs, a six-chain backlog and the published-predictor pipeline. Synthetic
resident ballast measures memory pressure, not SM occupancy. Work budgets are
fixed globally, independently of each host's calibration rate. Each job runs its
specified successful optimizer-update count and two epochs. Original predictor
labels and dataset splits are untouched.

Per instance: two repeated blocks, 120 held-out cohorts, 864 job receipts.
Two instance blocks are planned, subject to fresh availability and the cumulative
budget. Development is separate: fixed concurrency 1–4, shadow forecasts and
combined smoke checks with distinct seeds; its receipts do not enter held-out
latency estimates. There is no performance-driven retuning of held-out inputs.

Individual mode measures each job's acceptance-to-usable-result time. Bracket
mode measures request acceptance to the last required usable result. Report
average request latency, p95, maximum, makespan, per-repeat signs, activation,
trial/preparation/checkpoint costs and all failed/unattempted denominators.
Measured and neural-profile strata are reported separately. The planned summary
weights each condition equally within a stratum and each completed instance/run
block equally; it does not silently give larger brackets more weight.

The provisional target is at least 5% average improvement over v4 in each mode,
with no more than 5% p95 worsening or 10% mean regression in an individual
condition. Fixed-three and development-selected fixed concurrency remain visible
comparators. Four repeated blocks are descriptive evidence, not production or
independent generalization proof. Missing requests cannot pass a latency gate
using survivor-only averages.

## Resource limits and deliverables

One global $20 cap includes prior charges conservatively, storage, transfers and
cleanup reserve. Concurrent rentals reserve their combined finite maximum cost;
the cap is never applied separately to each GPU. Cost continues accruing until
provider-confirmed destruction, including after a lease deadline. All ownership
and cleanup actions target exact campaign instance IDs.

CPU tests cover arithmetic, delayed receipts, invalid predictions, empty/exhausted
budgets, actual planner selection, interruption/restart, stale membership, retained
updates and real spawned CPU-process cleanup. Each owned A10 must then pass the
same exact-host CPU checks, visibility qualification, fresh solo calibration and
development review before the held-out campaign. Raw receipts, source hashes,
Gantt/update/VRAM plots and comparison JSON/Markdown are retained and exported.
A negative result is a valid experiment outcome; no checkpoint is promoted.

## Pre-evaluation contract amendment

The first dedicated A10 calibration completed all eight jobs, but the subsequent
released-predictor capture rejected the historical synthetic builder's partial
workload contract. V5 now supplies the complete public `candidate_adapter` v1
contract through one shared builder for every evaluation arm and CPU calibration.
It explicitly describes the existing adapter, batch tensors, loss, optimizer and
training recipe. It does not change the executable workload, update counts,
predictor weights or baseline scheduler.

CPU preflight includes real S_A10 capture and inference for the frozen light,
medium and compute configurations when `A10_V5_TEST_RELEASE` is supplied. The
owned A10 preflight supplies this path explicitly. The earlier successful
calibration and failed capture remain archived as diagnostics. Fresh source pins,
GPU qualification and calibration are required before the repaired evaluation;
no development or held-out outcomes existed when this amendment was made.

Canonical JSON serialization also exposed an order check in the experiment
protocol: it compared configuration-map insertion order with the frozen seed
order list. The list now remains authoritative across serialization; it must
contain each configuration exactly once. Round-trip checks preserve the exact
calibration seeds, update budgets, development inputs and 120-cell comparison
order. This changes validation only, without changing the frozen experiment.
