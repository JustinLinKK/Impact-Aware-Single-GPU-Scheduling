# Variance delta for agent-generated workloads

Research contract, October 24, 2026. This revises v2 in place; it does not deploy
an admission rule or modify training jobs. The CPU reference demonstrates feature
distances and missing-data behavior, not a validated performance predictor.

## Why lineage alone is insufficient

The local MLEvolve code is the strongest evidence about this installation:

| Finding | Local evidence under `/home/justin/MLEvolve` | Consequence |
|---|---|---|
| Improve permits backbone/pipeline replacement and ensembles; debug may retry unchanged code | `agents/improve_agent.py:126`, `agents/debug_agent.py:157` | Stage is descriptive, never a drift class |
| Fusion has multiple sources; aggregation has a virtual parent and real source IDs | `agents/fusion_agent.py:203`, `:384`; `agents/aggregation_agent.py:216` | Preserve all lineage sources, not only parent or branch |
| Scheduler payload already carries precision, batch/accumulation, folds, workers and other hints, but loses parent hash/source IDs and extraction confidence | `engine/executor.py:1096`, `:1493`; `engine/preflight.py:1293` | Extend existing metadata, do not invent a parallel identity system |
| Architecture-switch means family strings differ; family can be inherited when detection fails | `engine/executor.py:1505`, `agents/triggers.py:92` | Same family is not proof of unchanged architecture |
| Script signature ignores batch edits; component counts are scalar heuristics | `engine/script_introspection.py:719`, `:764`, `:787` | Include resolved configuration and explicit execution plan |
| Lesson evidence preserves lineage/diffs, but its structural fingerprint counts call names rather than arguments/connectivity | `lesson_profile_database/evidence.py:49`, `:73`, `:139`, `:292` | Width/kernel/control-flow changes may be missed; `controlled` is not a scheduler guarantee |
| Branch timing can fall back to same branch/family/batch/hardware/backend | `localml_scheduler/scheduler/resource_estimator.py:467` | A compatible timing prior is not a current-child measurement |
| Replay can fall back to family/name and compare total duration plus average VRAM | `localml_scheduler/scheduler/placement_replay.py:49`, `:157`, `:633` | Gate inherited decisions, not just predictor output; busy-anchor already bypasses fixed-slot replay |
| Group identity preserves multiplicity/backend config, but some member matching averages duplicates and evidence merge omits backend config | `localml_scheduler/domain/identity.py:123`, `scheduler/time_objective.py:352`, `scheduler/colocation_evidence.py:176` | Preserve individual evidence and configuration through every lookup/merge |

These are audit locations in the live checkout, which includes pre-existing local
changes; they are not assertions that v2 contains or fixes those runtime paths.
The existing lesson database distinguishes exact retrieval scope from compatible
and similar advice (`docs/lesson_profile_database.md:10`). Even its exact bucket
is family/hardware/workload scoped, not proof of exact executable or group
performance identity. Preserve the advice/evidence separation and strengthen the
execution-evidence contract.

## What the research adds

[MLEvolve](https://arxiv.org/html/2606.06473v1) describes cross-branch evolution;
its Appendix G includes backbone fusion and changes to losses or preprocessing.
[AIDE](https://arxiv.org/html/2502.13138v1) searches programs through draft, debug
and improvement operators. [MLE-STAR](https://arxiv.org/html/2506.15692v3) combines
model retrieval, component refinement and ensembles. These support treating a
job as a complete executable pipeline. They do not establish this installation's
mutation frequencies; estimate those from its own event log.

Unseen applications are not new to all schedulers.
[Quasar](https://csl.stanford.edu/~christos/publications/2014.quasar.asplos.pdf)
combines brief measurements of a new application/dataset with historical evidence,
and revisits classification when behavior changes. Borrow the prior-plus-current-
measurement pattern, not its cluster results or profiling mechanisms.
[NeuSight](https://arxiv.org/abs/2407.13853) composes tile-level estimates for unseen
models/hardware. This motivates operator/tensor descriptors rather than family
names, but solo latency prediction does not establish co-run interference.

[Weisfeiler–Lehman graph features](https://www.jmlr.org/papers/v12/shervashidze11a.html)
provide a tractable way to summarize labeled neighborhoods. A directed, typed
adaptation can expose changes hidden by call-name counts. It remains lossy:
matching features do not prove graph equivalence, and graph equivalence does not
prove performance equivalence. [Adaptive conformal inference](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)
studies coverage under distribution shift; its long-run coverage objective is
not a per-job or OOM guarantee. V2 does not implement or claim those guarantees.

## Keep identity, drift and uncertainty separate

Freeze a versioned workload manifest **after final code review, execution rewrites
and physical configuration resolution**. Include candidate/run/node IDs; parent
and all source IDs/hashes; final source plus imported code, dependencies, config,
data and preprocessing revisions; device/slice; framework/library/compiler/backend
settings; batch, accumulation, precision and trainable state; execution phase and
component plan. Cache extraction results by this manifest. An incomplete dependency
or dynamic-control-flow capture stays explicitly incomplete. Do not execute
arbitrary candidate code merely to obtain a cheap CPU descriptor.

Stage and family help retrieve potential reference records. They do not enter
an authorization threshold. Separate:

* **Identity:** can the old record even describe this exact execution and phase?
* **Variance delta:** what differs between available descriptors?
* **Expected resource change:** signed predictions for peak VRAM, time per update,
  total remaining work, host/I/O demand and contention; these have different units.
* **Uncertainty:** missing fields, unsupported operators, residual calibration
  support, within-phase variability and observation age. Distance from one parent
  is not distance from the student's entire training distribution.

## Define variance delta

The requested name denotes an engineered drift vector, **not statistical variance,
a probability, or a percentage prediction error**:

| Component | Compare | Failure that a family label misses |
|---|---|---|
| Structure | Typed directed operator neighborhoods, gradients, trainable/frozen paths and optimizer work | Same backbone name with new connections or head |
| Scale | Tensor shapes/distributions, parameters, activation estimates, optimizer state, FLOPs/bytes and launch counts | Same operators with wider tensors or higher resolution |
| Recipe | Physical batch, accumulation, optimizer, dtype, checkpointing, compile mode, freeze state | SGD to Adam adds state without changing the forward graph |
| Pipeline/plan | Data/augmentation/I/O path, workers, folds, ensembles, repetition and residency schedule | CPU augmentation starves the GPU; concurrent ensemble raises residency |
| Observed drift | Median/p95 optimizer-update time, robust dispersion, peak allocated/reserved memory, input/transfer waits | Similar solo busy but different burstiness and interference |

For a nonnegative scalar x and reference y use
`d(x,y) = abs(x-y) / max(x,y,unit_floor)`, with a positive unit floor and
bounded in [0,1]. Keep the signed
reference-relative change `(x-y)/max(y,unit_floor)` separately when meaningful.
Categorical mismatch is 1, match 0. Compare fixed-schema histogram counts using
`sum(abs(h_new-h_ref)) / sum(h_new+h_ref)`; label absence is a known zero only
when both extractions certify complete coverage of that histogram. Two certified
empty histograms have distance zero; incomplete extraction remains unknown.

Use the **maximum** scalar drift within each component; report the vector and
its maximum as a conservative retrieval diagnostic. This prevents decreases in
one dimension from cancelling increases in another. It is not a learned weighting
or universal criticality measure. Missing required fields give `unknown`, not
zero; report known partial differences separately. Schema/environment/phase
mismatches are explicit reasons to invalidate exact evidence, not terms that can
be averaged away. There is no proposed universal “delta < 10% means reuse” rule.

Example: a topology-preserving width change may give structure=0 but scale>0.
Changing augmentation may give structure=scale=0 but pipeline>0. Even if all
available components equal zero, unseen control flow or a changed executable
manifest prevents exact reuse. A data-only or seed change may alter shapes,
convergence or total work; zero captured drift is still not proof of equal time.

Within-job variance (for example MAD/median over stable windows) is measured
separately from between-job drift. Compare equal units and phases, excluding
warmup/compilation from steady-state rates but charging their elapsed cost. Count
completed optimizer updates consistently with accumulation. Preserve per-member
rates in a group, and never silently match heterogeneous duplicates by average.

## How the scheduler should use it

1. **Retrieve cheaply.** Search direct ancestors, all fusion sources and nearby
   complete descriptors. Record why a reference matched and where it is missing
   information. Do not average incompatible parents into a synthetic measured job.
2. **Order, do not certify.** CPU TorchScript predictions remain hints. Use
   descriptor novelty and calibration support to prioritize promising, measurable
   candidates within fairness and exploration budgets. Unknown candidates may get
   a solo turn; they must not be rejected forever for lacking a known family.
3. **Measure the candidate.** On its first safe solo slot, collect a bounded set
   of useful training updates with existing checkpoint/cursor semantics. Do not
   clone the job or block all training for a profiling campaign. If the measurement
   is incomplete or nonstationary, keep training conservatively without packing.
4. **Measure the actual group.** Trial one addition under existing memory/backend,
   rollback and exploration controls. Apply v2's conservative net-progress and
   individual-slowdown checks. Similar solo behavior, low delta, a pair profile,
   or a family match cannot certify a new full group. Do not spend on every pair.
5. **Refresh on change.** Reconcile the manifest with executed code/configuration.
   Phase transitions, new data-shape ranges, unfreezing, optimizer/fold/ensemble
   switches, or residual drift retire affected evidence and stop new admissions
   until measurements are valid. Drain/rollback uses existing safe controls.

### Obtain the first measurement without circular admission

Distinguish **unknown timing** from **unsupported memory admission**. A missing
solo rate does not itself make a job ineligible. When no job occupies the GPU,
use the existing cooperative cold-start path with the selected backend running
alone, approved configuration, CPU preflight, live headroom and a conservative
branch-profile memory estimate. For novel memory behavior, this is a guarded
calibration attempt, not an assertion that inherited memory is exact.

Require a supported runner with durable progress, safe boundaries, bounded startup
lease and verified allocator limiting **before model construction**, plus live
memory monitoring and the existing failure/cleanup/recovery path. Reserve allowance
below the 31 GiB ceiling (or stricter configured limit), subtract existing usage
and runtime overhead, and keep the GPU unshared until calibration succeeds.
`execution/trial_control.py:configure_trial_memory` already calls the PyTorch
allocator limit; `scheduler/cold_start.py` supplies an allowance and cooperative
measurement controls. Actual integration must test that the generated runner
honors these contracts. A declared metadata flag alone is not verification.

[PyTorch's allocator limit](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
limits allocator-managed memory, not all
[external CUDA allocations](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html). Live sampling
also cannot prevent every transient overrun. This proposal claims no strict
whole-process VRAM guarantee for arbitrary generated code. If unknown external
allocations or a noncooperative runner prevent supported containment, mark the job
`needs_memory_evidence` with the specific reason and return it to the agent's
existing repair/admission workflow. Do not silently admit it, reserve the GPU
indefinitely, shrink its approved batch, or repeatedly retry a known failure.
An unprofiled program is not universally executable under the stated memory bound.

For eligible cold-start jobs, age determines their first solo turn until measured
service can be used. Persist useful pre-reference updates as unpriced work; settle
once against the first valid same-phase solo reference. If no stable reference is
obtained before the phase ends, tag committed useful solo wall time as an accounting
fallback, excluding setup and lost work. It cannot support a co-run comparison.
The main v2 ledger specifies retry/rollback and unresolved shared-work handling.

For an exact unchanged retry, complete compatible identity and fresh observations
make evidence eligible for normal validation; they do not by themselves permit
sharing. Reuse failed-attempt cooldowns only for unchanged identities. New work
with unknown memory still obeys conservative branch/live-VRAM admission. Do not
turn an unverified family estimate or a good time estimate into a memory bound.

Model composite jobs as components with sequential/concurrent phases. Repeated
sequential folds mostly add work; simultaneously resident models can add peak
memory. Remaining time is the sum of known future phase work, not necessarily
one epoch estimate times a scalar count. Bound or mark unknown dynamic/early-stop
plans and shrink the gain horizon accordingly.

A later optional learning step could predict **resource-specific parent→child
residuals**, with lineage-disjoint validation and uncertainty calibration. Do not
fit one scalar “variance” and use it as a universal correction. No learned delta
model or calibrated reuse threshold is delivered by this revision.

## Evaluation and implementation boundary

The CPU reference accepts already extracted fixed-schema descriptors. It includes
a small directed neighborhood histogram and scalar/category comparisons, plus
explicit unknown-data output. It does not parse arbitrary training programs,
trace models, create manifests, estimate VRAM, calibrate uncertainty, compare
sharing groups, or authorize scheduling. Hand-written fixtures verify the
contract only; the extractor and live evidence gate remain future work.

The next implementation should first carry existing immutable lineage evidence
through the scheduler bridge, then add extraction coverage and apply identity
checks to **every** inheritance path: lesson, solo, branch, group and replay.
Measure extraction time and reject unbounded introspection. Subsequent GPU studies
must retain the original v2 gain, fairness, budget and memory controls.

Use lineage/task/mutation-disjoint validation/test splits, cold caches and unseen
operators. Compare family reuse, no reuse, exact-identity reuse and delta-guided
measurement. Report false low-drift matches, error by target, uncertainty coverage,
trial and CPU overhead, job completion/tail wait and memory failures. Include
same-graph/different-pipeline and same-family/different-shape cases.

Because completion drives MLEvolve's next proposal, also compare end-to-end search
under a fixed agent/time budget and multiple seeds, reporting valid candidates,
branch coverage and best **validation** score versus wall time. Fixed-arrival
replays answer a different question. Do not use held-out test scores to schedule
or tune. Every eventual experiment still needs the repository's combined
Gantt-above-metric plot. No GPU experiment was performed here.
