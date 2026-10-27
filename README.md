# Impact-Aware Single-GPU Scheduling

Agent-owned NVIDIA CUDA documentation enrichment is documented in
[`docs/cuda_docs_integration.md`](../docs/cuda_docs_integration.md). It is
disabled by default and is not part of scheduling, admission, or execution.

`localml_scheduler` is a reusable local-first ML job manager for single-machine agent workflows. V1 focuses on two practical capabilities:

- a single-GPU scheduler with priority queueing, safe-point pause/resume, persistence, and restart recovery
- a RAM-backed baseline-model cache with optional LRU entry-capacity and RAM-percent limits that keeps immutable CPU-side baselines warm and serves isolated copies to worker subprocesses
- time-aware GPU packing that chooses ready jobs and batch sizes inside one preset process backend from predicted completion time and compatibility evidence, with average VRAM used only as a safety gate
- Linux overlap on a configured non-exclusive backend, admitted incrementally by the time-aware policy
- reserved exclusive five-option calibration that persists timing and VRAM measurements for time-aware planning
- one-epoch cooperative addition trials that acquire missing profiles while preserving training progress
- optional hardware-selected PerfSeer student prediction, with CPU-only
  TorchScript inference and per-job branch-profile fallback

This repository is checked out as MLEvolve's pinned `localml_scheduler/` Git
submodule. The repository root is the Python package, preserving existing
`localml_scheduler.*` imports and runner targets.

Run the commands below from the MLEvolve root with its Python environment.
The MLEvolve adapter and hardware-knowledge integration use the host's
`engine/`, `utils/`, and `schema/` files; this extraction preserves those
integration dependencies. Links to `../docs/` refer to MLEvolve documentation.

For development, create a branch in this repository, commit and publish scheduler
changes here, then record the tested revision in MLEvolve with
`git add localml_scheduler`. A parent checkout restores that revision with
`git submodule update --init --recursive`.

## Architecture

- `schemas.py`: serializable job, checkpoint policy, progress, and cache schemas
- `scheduler/`: policy, queue, service loop, recovery, and worker supervision
- `scheduler/placement_planner.py`: workload-ranked anchor, one-newcomer-at-a-time placement planning
- `scheduler/time_objective.py`: verified sequential-versus-packed drain-time scoring
- `scheduler/telemetry.py`: lightweight `nvidia-smi` device telemetry for solo and packed runs
- `execution/`: subprocess launcher, file-based control plane, worker entrypoint, and runner context
- `execution/backends.py`: exclusive, independent CUDA-process, and MPS-client-process launch backends
- `checkpointing/`: atomic local checkpoint save/load
- `model_cache/`: in-memory LRU baseline cache plus a local socket server for worker access
- `storage/`: SQLite-backed jobs, commands, checkpoints, cache metadata, and event history
- `observability/`: JSONL events, log files, and aggregate reports
- `profiling/`: exclusive five-option measurement plus runtime profile helpers
- `prediction/`: scheduler integration for isolated source conversion and the
  PerfSeer submodule's CPU runtime
- `examples/`: toy PyTorch training runner and a demo script
- `adapters/`: thin helpers for wiring job submission from MLEvolve or other systems

## Release integration (`oct_proposed_v2`)

| Component | Deployment | Integration interface | Overhead controls |
| --- | --- | --- | --- |
| Scheduler | One persistent service on the GPU host; optional embedded mode for a single agent process | `SchedulerHTTPClient` over loopback HTTP; `SchedulerClient` and `SchedulerService` for direct calls | Retain queue state, predictor, and telemetry sampler; reuse HTTP connections; submit at job boundaries and wait for completion events |
| Resource predictor | CPU library inside the scheduler process | Direct calls through the existing `MLVramPredictor`, using `TrainingJob.metadata.perfseer_model` as the candidate description | Load the TorchScript deployment student once; bound the feature cache; limit CPU threads |

Run from the MLEvolve root using its Python environment and initialized
`PerfSeer-predictor` submodule. This release branch retains the host integration
dependencies described above; it is not a standalone installation package.

Create a private API token once, then start the service:

```bash
python -c 'import os, secrets; from pathlib import Path; p = Path("localml_scheduler/runtime/paper/api.token"); p.parent.mkdir(parents=True, exist_ok=True); fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, secrets.token_hex(32).encode()); os.close(fd)'
python -m localml_scheduler.cli scheduler serve \
  --settings localml_scheduler/examples/scheduler.paper.yaml \
  --token-file localml_scheduler/runtime/paper/api.token
```

The endpoint binds to `127.0.0.1:8766` and requires the token on every request.
It permits local job execution, so share the token only with trusted agents on
the host. Use one service per runtime root/GPU. Keep that runtime root across
restarts to retain SQLite queue state, checkpoints, and scheduling evidence.
The example uses incremental CUDA-process admission and a training budget of 90% of detected total VRAM.

An agent submits work once, then waits at the job boundary:

```python
from pathlib import Path
from localml_scheduler import SchedulerHTTPClient, TrainingJob

token = Path("localml_scheduler/runtime/paper/api.token").read_text().strip()
with SchedulerHTTPClient(token=token) as scheduler:
    job = scheduler.submit(TrainingJob.create(
        "my_pkg.training:run_training_job",
        "candidate-1",
        "/absolute/path/to/baseline.pt",
        runner_kwargs={"batch_size": 32},
        metadata={"perfseer_model": {
            "source_path": "/absolute/path/to/model.py",
            "entry": "build_model",
            "input_shapes": [["$batch", 3, 224, 224]],
            "input_dtypes": ["float32"],
            "precision": "fp32_ieee",
            "loss": "cross_entropy",
            "optimizer": "adamw",
            "dataset_size": 10000,  # Actual training split; use your sampler count.
        }},
    ))
    result = scheduler.wait_for_completion(job.job_id, timeout=3600)
    if result is not None:
        print(result.status.value)  # COMPLETED, FAILED, or CANCELLED
```

The SDK imports without loading PyTorch or opening the scheduler database.
Use one client per thread; keep it open across jobs. `inspect`, `list_jobs`,
`pause`, `resume`, and `cancel` expose job boundaries without per-step RPCs.
Paths and runner modules must be accessible to the service process. Keep the
existing runner safe-point/checkpoint contract for pause/resume support.

Completion uses a bounded HTTP long poll, awakened by worker-exit or queued
cancellation notifications. The response contains a terminal event and job
snapshot; an expired wait returns `None`. Persisted terminal state also serves
late subscribers after a reconnect/restart. Inspect a stable job ID after an
ambiguous submission error: submissions are not automatically retried, and
duplicate IDs return HTTP 409. An HTTP client does not start another scheduler.

For a single embedded agent, retain the service and use direct calls:

```python
from localml_scheduler import SchedulerClient, SchedulerConfig

client = SchedulerClient(SchedulerConfig.from_file(
    "localml_scheduler/examples/scheduler.paper.yaml"
))
service = client.create_service().start(background=True)
try:
    job = client.submit_from_file("/absolute/path/to/job.yaml")
    result = service.wait_for_completion(job.job_id, timeout=3600)
finally:
    service.stop()
```

The active predictor remains the CPU-only TorchScript deployment student;
inference is a direct call inside the service. Source import/tracing stays in
timed CPU-only subprocesses. Cached features are keyed by source content,
entry, shapes/batch size, dtypes, precision, constructor arguments, and artifact
hash. `prediction.ml.cpu_threads` defaults to 1 and caps PyTorch intra-op threads
in the scheduler process and conversion workers. In embedded mode this also
sets the host process's PyTorch intra-op limit. The student is reloaded only
when a new service is constructed. Unsupported hardware, incompatible artifacts,
or individual conversion failures retain the existing branch-profile fallback.
Timing labels retain their existing meaning; step time is not treated as a
measured full-epoch label.

Telemetry reuses an NVML connection and device handle for the service lifetime
when available, with the existing `nvidia-smi` fallback otherwise. The scheduler
continues its internal control/telemetry loop while agents wait for job outcomes.

## How To Run

Start the scheduler:

```bash
python -m localml_scheduler.cli scheduler start --settings localml_scheduler/configs/scheduler.example.yaml
```

Submit a job:

```bash
python -m localml_scheduler.cli submit localml_scheduler/configs/job.example.yaml
```

Inspect state:

```bash
python -m localml_scheduler.cli list
python -m localml_scheduler.cli status <job_id>
python -m localml_scheduler.cli cache-stats
python -m localml_scheduler.cli report
python -m localml_scheduler.cli scheduler migrate-backend-modes --dry-run
```

Run the demo:

```bash
python -m localml_scheduler.examples.demos submit
python -m localml_scheduler.examples.demos bridge
```

## Cold-start addition trials

An eligible cooperative runner starts on the configured packing backend even
without profiles. The scheduler admits one provisional newcomer at a time,
keeps its requested physical batch, and persists preparation, pre-add timing,
overlap measurement and the decision. With valid released predictions for every
member, CPU-generated virtual profiles replace per-job exclusive profiling:
rounded peak VRAM controls initial fit, and step cost supplies fixed work weights.
The trial skips solo-reference phases entirely. Predictions remain labelled as
predicted evidence. Partial/unsupported predictions use the existing measured
reference fallback; explicitly requested probes and unsupported runners keep
their existing exclusive path.

`gpu_scheduler.colocation.trial_epochs` defaults to **1**. A new worker first
acknowledges its current preparation token, so an old readiness flag cannot
start a trial. Timing windows collect two warmup updates and two complete halves
of at least eight measured updates in total, targeting at least one second or
64 updates. Slow steps do not discard an incomplete window after one second.
Timeouts use observed step durations, with a separate 300-second absolute trial
limit; the predictor's relative time indicator never controls a wall-clock timer.
The candidate's optimizer-step allowance also lets a trial cross an epoch boundary.

In the prediction-first path, the decision compares
`sum(predicted_step_cost / observed_step_seconds)` after versus before joining,
using one frozen model/hardware scale. The active group supplies the short
pre-add window; the newcomer then joins directly. This measures predicted-work
throughput, not exact completion-time gain. No predicted cost is saved as a measured
time profile. Actual observations can later refine memory and timing estimates.

In the measurement fallback, complete timing evidence uses the drain objective.
If epoch totals or subgroup timings are missing, it uses normalized throughput:
`sum(solo_step_seconds / observed_step_seconds)` after versus before admission.
Both halves must improve by `colocation.cold_start_gain_margin` (default 0.03).
Both deteriorating halves reject; incomplete or borderline evidence is
inconclusive. Inherited subgroup rates are not a verified completion forecast.
Only a rejected/inconclusive newcomer checkpoints and requeues; another
candidate or changed membership may still be tried. Solo references and packed
windows are stored separately; inconclusive windows do not create bad profiles.

Retries have a 30-second cooldown and finite limits: two failed attempts for one
group and three for one candidate. A candidate that exhausts these attempts gets
a solo turn, preventing repeated checkpoint/restart overhead and starvation.
Workers retain their process handles until they have actually exited; a draining
worker blocks a new admission. These limits are configurable under
`gpu_scheduler.colocation.cold_start_*`.

The [October 27 verification receipt](verification/admission-work-ranking-20261027.json)
records the lifecycle regressions, CPU student inference, real-image cancer
training and three-job pause/resume/overlap check. These integration checks do
not establish a scheduling speedup; that requires the matched A10 replay.

Generated PyTorch scripts must call `script_scheduler_context()` inside their
entrypoint before allocating CUDA memory, restore the returned context's
checkpoint, and use `TrainingControlHook.safe_point` after complete optimizer
updates. The helper is inactive outside scheduled execution. Preserve model,
optimizer, scaler, LR scheduler, all RNG state, and the exact data position.
[`examples/cold_start_runner.py`](examples/cold_start_runner.py) demonstrates
this contract, including resumable partial epochs. Generation/review checks
require these hooks; printed epoch logs provide telemetry only.

Temporary yields retain Python, CUDA, optimizer, and iterator state and carry
renewable owner/expiry leases. The controller polls at 50 ms while a trial is
active. SQLite updates merge control and progress atomically. Events distinguish
useful measurement time, yields, controller/synchronization time, startup, and
checkpoint cost. Trial wall time includes useful training and must not be
charged again as pure overhead.

Admission uses 90% of total VRAM by default, accounting for live device usage
and pending reservations. There is no additional 31 GiB ceiling or fixed 5%
cold-start deduction. The newcomer receives a PyTorch caching-allocator
limit before model allocation; this is not isolation for arbitrary CUDA
extensions. Runners using unsupported allocators must declare
`trial_allocator_supported=false` and an explicit `estimated_vram_mb` bound.
MPS ceilings are recorded before launch (100% when unknown) and active clients
are never resized. There is no fixed parallel-job cap.

Run the bounded synthetic comparison with the project's PyTorch environment:

```bash
python -m scheduler_benchmark_test.cold_start_comparison --output runs/cold-start-5090
```

It uses four distinct regression models, three short epochs, BF16 compute,
FP32 AdamW state, empty profile stores, and a labelled 2 ms simulated input gap.
Two paired comparisons run in reversed order within a five-minute budget.
The run directory contains settings, source evidence, SQLite/events, progress,
checkpoints, results, and one combined Gantt/metric PNG refreshed after each arm.
`--cpu-smoke` performs a small controller shakedown; `--previous-run PATH`
includes diagnostic comparisons and subtracts their duration from that budget.
These experiments test scheduling behavior, not Petfinder model quality or
FLOP saturation. MPS hardware validation requires an environment with MPS.

## MCP Graph/Vector Surface

`localml_scheduler.mcp_server` exposes a stdio MCP server for hardware-aware
agent context. SQLite remains the scheduler control-plane source of truth;
Neo4j stores measured evidence, and Qdrant stores code knowledge.

Preferred read-only tools for agent integration:

- `get_optimization_context(candidate, limit=8)` combines hardware context,
  graph evidence, derived symptoms, vector evidence, recommendations, risks,
  refs, and confidence.
- `get_profile_evidence(candidate, limit=8)` returns graph-only evidence from
  `SingleJob`, `PackedJob`, and `PackedJobMember`-style profiles.
- `search_code_knowledge(query, filters, record_types, limit=8)` searches
  Qdrant code docs, optimization recipes, and API-symbol chunks.
- `get_code_optimization_context(candidate, graph_context=None, limit=8)`
  bridges graph-derived symptoms into vector retrieval.
- `get_backend_design_guidance(effective_backend, runner_contract, pipeline_stage)`
  returns exact-backend hard rules from the local seed corpus before optional
  semantic ranking, so it remains available when Qdrant is disabled.

`get_job_design_context(...)` remains the graph-only job-design evidence query;
new optimization flows should normally use the combined context APIs above.

For local development, start Qdrant and ingest the repo-curated seed corpus as
code knowledge:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
python -m localml_scheduler.cli code-knowledge ingest --settings localml_scheduler/configs/scheduler.example.yaml
```

Use `--dry-run` to validate and summarize the seed records without writing to
Qdrant. MCP search/context calls return empty results instead of failing the
scheduler when the feature database is disabled or unavailable.

## Custom PyTorch Integration

Point a job at a runner target in `module:function` form, for example:

```yaml
config:
  runner_target: "my_pkg.training:run_training_job"
```

The target receives a `RunnerContext` object with:

- `job`: the fully materialized `TrainingJob`
- `control_hook`: call `safe_point(...)` before training, every N steps, after epochs, or at explicit save points
- `checkpoint_manager`: load/save resume state when needed
- `load_baseline_object()`: fetch a fresh baseline object from the RAM cache, or fall back to disk if needed
- `load_resume_checkpoint()`: access the latest successful checkpoint

Structured runners can expose a batch probe hook in `module:function` form. A
job reserved with `scheduling_class: exclusive_probe` measures the five batch
options configured by `batch_options`, persists timing/VRAM evidence in SQLite,
and then releases the idle boundary. Normal jobs never run the removed
VRAM-saturation search controller.

Structured runners can also opt into runtime probing with `runtime_probe.enabled: true`. The default `epoch_1` strategy treats the first exclusive epoch as calibration, persists a runtime profile keyed by workload signature, hardware, backend, and resolved batch size, and then uses that estimate to reject badly skewed packed groups. Jobs without reliable epoch semantics can use `runtime_probe.strategy: "step_window"` instead.

## PerfSeer ML Prediction

Set `prediction.mode: ml_predictor` to select a registered student model using
the detected GPU name, compute capability, and VRAM. The current registry
contains the NVIDIA A10 model. If the current hardware has no matching artifact,
the scheduler remains operational and uses branch-profile prediction for every
job. The `test_override_enabled` and `test_model_path` fields provide an
explicit test-only cross-hardware bypass.

Predictable jobs may supply:

```yaml
metadata:
  perfseer_model:
    source_path: "/absolute/path/to/model.py"
    entry: "build_model"
    input_shapes: [["$batch", 3, 224, 224]]
    input_dtypes: ["float32"]
    precision: "fp32_ieee"
    constructor_args: []
    constructor_kwargs: {}
```

When this block is absent, the scheduler tries the job source with
`build_model` and `metadata.input_shape`. Import, FX tracing, and shape
propagation run in a timed subprocess with GPU visibility disabled. An
unsupported source, timeout, or invalid result falls back only that job.

There is one production placement mode: `parallel_time_aware`. Its objective is
verified drain-time gain, never GPU-memory utilization. Predicted average used
VRAM in MiB (`avg_vram_mb`) is only a hard admission constraint. Device-level
memory is averaged over a time window: sustained use at the stop threshold
closes only new packed admission, and active work continues normally. Admission
reopens after a complete window below the resume threshold. GPU/SM utilization
is retained only as telemetry and profile evidence.

The deployment also selects exactly one `gpu_scheduler.packing_backend`:
`cuda_process` or `mps_process`. Both launch one independent OS process and
CUDA context per job. `mps_process` additionally connects those client
processes to the scheduler-owned NVIDIA MPS service. `exclusive` is a solo
probe/safety fallback and is not an alternative draft-time optimization target.
If MPS is unavailable, `mps_unavailable_policy` either permits the explicit
exclusive fallback or fails; it never silently changes to packed
`cuda_process` execution.

## Time-aware scheduling

With a released student, `parallel_time_aware` uses predicted busy **ordering**
to choose a heavier idle-GPU anchor and lighter joining candidates. Within equal
busy hints, it prefers lower `epoch_cost × remaining_epochs`. Missing hints sort
after available hints; measured remaining time, explicit priority and job ID
provide fallbacks. Without predictor hints it uses measured remaining runtime.
Submission time and waiting duration do not override that ordering. Each addition
must pass scheduling class, backend and pair compatibility, live-admission, batch-option, and
predicted average-VRAM checks before any concurrent work begins.

`gpu_scheduler.scheduler_decision_mode` selects the trial ordering layer:

- `baseline` (the default) uses the indicator ordering described above.
- `backend_awared` jointly enumerates pairs at an empty-GPU boundary and ranks
  empty-GPU pairs or active-group newcomers with deterministic Pareto fronts.
  The risk vectors are derived from a CPU-only AST fingerprint and differ for
  MPS-client processes and ordinary CUDA processes. Static analysis only orders
  candidates; unknown placements still require the same measured live trial.
  Student workload hints break ties within this measured-benefit/risk ordering.

Backend-aware mode applies backend compatibility, batch/accuracy, conservative
VRAM overhead, known-bad exact-profile, and optimistic trial-amortization gates
before ranking. Exact profile identity includes hardware/runtime identity,
source/graph signature, dtype, batch vector, canonical backend, runner contract,
and backend configuration; epoch count and submission time are excluded. MPS
pair templates are limited to 50/50, 60/40, and 40/60. Active client
allocations are immutable after CUDA-context creation; unsafe incremental
admission is declined and reconsidered after the active configuration drains.

The analyzer never imports submitted modules or initializes CUDA. It recognizes
common linear/GEMM, convolution, attention, normalization, activation,
reduction, embedding, pooling, layout/transfer, recurrent, and optimizer calls,
plus synchronization, blocking/nonblocking transfers, DataLoader workers,
checkpoint/evaluation calls, CPU augmentation, dynamic control flow, and
custom/fused-operation markers. Dynamic dispatch, runtime-generated modules,
opaque extensions, missing shapes, and unsupported custom operators degrade to
low confidence and baseline tie-breaking; they do not block every candidate.

The final admission check is a live trial requiring two fresh epochs from every
member. These are ordinary training epochs: progress and checkpoints are retained
whether the candidate is accepted or paused. If the newcomer reaches epoch two
before an existing member has two complete post-join intervals, its target moves
forward one epoch and it keeps training. Once every member has enough evidence,
the scheduler compares sequential drain time with measured packed drain time:

```text
D_active = piecewise drain of the current active stack
T_seq    = D_active + candidate remaining_epochs * candidate solo epoch_seconds
T_pack   = piecewise drain of the proposed packed stack
gain     = T_seq / T_pack
```

The piecewise drain advances to the next member completion, removes every job
that completed at that boundary, and recalculates rates for the remaining
membership. Exact hardware/backend/batch colocation timing is used for a
remaining multi-job subset and same-backend runtime timing is used for a final
singleton. When complete subset evidence is unavailable, surviving jobs inherit
their parent phase rates so missing evidence never invents an optimistic
speedup.

Admission requires `gain >= gpu_scheduler.colocation.min_gain` (1.0 by
default). A slowdown rejection pauses only the newcomer with `hold: false` and
stalls every further packing attempt until one member of the pre-trial pack
leaves execution. Memory or compatibility failures do not create that stall.
Trial and stall state survive scheduler restart, and stale state is discarded
when active membership no longer matches.

Exact colocation timing profiles are isolated by hardware and the multiset of
packing signature, resolved batch, and backend. Existing members contribute only
complete intervals that started after the newcomer joined; the newcomer may use
its fresh first-epoch runner timing. Raw scalar step timing and stored profiles
never fill missing live-trial evidence. The two newest valid samples per member
are averaged. The trial window has an adaptive five-minute minimum and thirty-
minute maximum; an unverified timeout pauses only the newcomer and writes no
profile or admission stall.

A stored profile must have at least two recent observations before its rates can
be used. Immediate rejection additionally requires two consecutive bad live
trials within 24 hours. An accepted trial clears bad confirmations, and the next
trial replaces rather than averages with an expired profile. Profiles without
confirmation metadata are never trusted for immediate rejection.

### Task-scoped placement replay

Jobs may provide a typed `workload_identity` with `task_key`, `dataset_key`,
`architecture_key`, and `architecture_family`. The MLEvolve adapter also
accepts these as direct arguments. `workflow_id` is execution identity and is
not reused as task identity. Script introspection emits normalized exact and broad architecture
keys when they can be inferred. A reliable task/dataset identity and an
architecture identity are both required for replay.

After three distinct membership episodes produce the same verified width and
backend on the same hardware and scheduler mode, the scheduler persists a
placement template. Width one replays as exclusive; larger widths start an
anchor immediately and fill the first vacant slot as matching jobs arrive.
Replayed jobs reuse the stored per-slot batch size and trusted runtime/VRAM
profiles, bypassing batch probing, live colocation trials, gain scoring, and
rejection stalls. Passive epoch timing and GPU telemetry continue.

Replay is invalidated before dispatch when task, dataset, exact architecture,
or architecture family changes, or when either predicted total training time
or average VRAM changes by at least the configured symmetric ratio. Backend,
parallel-cap, compatibility, cached-batch, live-memory, and aggregate-VRAM
safety checks remain active. A closed live-memory gate waits without deleting
the template; missing or low-confidence evidence and hard safety failures send
the same job back through normal evaluation. Configure this under
`gpu_scheduler.colocation.decision_replay`; the default observation count is
three and both change fractions default to `0.25`.

Released-student epoch costs never replace measured epoch timing in drain-time,
trial-amortization or replay calculations. A job with no measured duration uses
a cooperative measurement trial, or solo execution if its runner cannot cooperate.
Legacy artifacts still require the explicit `train_epoch_ms` target; the ambiguous
`train_time` output is not treated as a full-epoch estimate.

The former throughput controls `makespan_weight`, `flow_time_weight`, and
`min_aggregate_gain` are no longer accepted. `time_v3_flow_only`,
`time_v4_colocation_gain`, and `time_v5_piecewise_drain` are no longer accepted;
configure `time_v6_verified_piecewise_drain`. The `colocation` defaults are
`min_gain: 1.0`, `trial_epochs: 2`, `trial_decision_timeout_seconds: 30`,
`trial_evidence_timeout_min_seconds: 300`,
`trial_evidence_timeout_max_seconds: 1800`,
`profile_rejection_min_bad_trials: 2`,
`profile_rejection_ttl_seconds: 86400`, and `live_trial_enabled: true`.
Decision replay defaults to enabled with `min_stable_observations: 3`,
`training_time_change_fraction: 0.25`, and `vram_change_fraction: 0.25`.

Jobs with `scheduling_class: exclusive_probe` reserve the next idle boundary.
Existing packs drain without preemption and no normal work is admitted during
the reservation. Early stopping is evaluated synchronously at epoch safe
points and persists its patience state in job metadata across pause/restart.
`save_best_checkpoint` protects the tagged best checkpoint from normal pruning.
Reports include saved epochs and estimated wall time saved when the runner
provides a remaining-runtime estimate (or epoch step timing).
Generic runners still own model-state restoration; `restore_best_checkpoint`
is reserved for a runner-level restore hook and is not applied implicitly.

### Recommended early-stopping baseline

The repository's scheduler-aware example runners report `loss` at epoch safe
points. The following is a conservative starting point for those runners. In
the unified repository configuration, place this block under
`scheduler.settings`:

```yaml
early_stopping:
  enabled: true
  metric_name: loss
  mode: min
  patience_epochs: 5
  min_delta: 0.001
  min_epochs: 10
  save_best_checkpoint: true
  restore_best_checkpoint: false
  missing_metric_policy: ignore
```

This requires loss to decrease by more than `0.001` to reset patience, tolerates
five consecutive non-improving epochs, and never stops before epoch 10. The
absolute `min_delta` is metric-scale dependent and should be tuned when typical
changes are much smaller or larger. Prefer an epoch-averaged validation metric
such as `val_loss` when the runner exposes one; in that case, change
`metric_name` to `val_loss` and keep `mode: min`.

`min_epochs` is a stop gate rather than a warm-up reset: non-improving epochs
before epoch 10 still count toward patience. With missing metrics configured as
`ignore`, a missing or non-finite metric emits a warning but neither fails the
job nor consumes patience. `save_best_checkpoint: true` requires the runner to
provide a checkpoint `state_factory`. The raw MLEvolve subprocess adapter does
not currently emit epoch safe points, so early stopping applies only to
scheduler-aware runners that call the control hook.

The deterministic validation fixture can be run with:

```bash
python -m localml_scheduler.scheduler.trace_simulator
```

It compares serial FIFO, a VRAM-fill reference policy, the recursive time-aware
policy, and an exhaustive small-trace oracle, reporting makespan, flow time,
waiting, starvation, jobs/hour, predicted/actual VRAM, realized slowdown, trial
epochs, preserved rejected progress, admission stalls, and saved early-stop
work. The simulator accepts live backend changes, rolling-memory samples,
compatibility matrices, harmful or beneficial slowdown matrices, and validation
sequences. The exhaustive
oracle remains intentionally limited to small, non-preemptive drain-boundary
fixtures.

The real-GPU benchmark performs a hardware-specific five-option calibration,
then runs a one-job-cap time-aware control and normal time-aware placement at
least twice from isolated runtime directories. Historical fill-policy
comparisons exist only in the deterministic simulator. The replay records means, sample
variance, standard deviation, raw runs, hardware identity, makespan, and flow
metrics. A matched exclusive solo control replays each time-aware selected
batch so measured slowdown is not confounded by batch-size changes:

```bash
python scheduler_benchmark_test/repeat_time_aware_benchmark.py \
  --results-dir results/scheduler_benchmark_test/a10_time_aware \
  --data-root /path/to/cassava/prepared/public \
  --repetitions 3
```

The command requires an NVIDIA A10 by default and refuses to mislabel results
from other GPUs. `--allow-hardware-mismatch` exists only for clearly labelled
local harness validation.

Jobs may optionally include a `preload_source` with `model_id`, `model_path`, and `loader_target`. When present, the scheduler warms that shared source in RAM instead of the job's normal baseline target. This is useful for raw MLEvolve runs where many sibling jobs share one immutable starting checkpoint but still execute different generated scripts.

The normal pause flow is:

1. scheduler requests pause
2. worker reaches next safe point
3. checkpoint is saved atomically
4. worker exits cleanly
5. scheduler later redispatches the paused job from checkpoint

## Packed Execution Notes

- `parallel_time_aware` is the only accepted mode; removed fixed-width and VRAM-fill mode names fail configuration validation
- concurrency is derived incrementally from predicted/measured profiles and live GPU telemetry; no fixed job-count cap is configured
- the memory ceiling is detected total VRAM times `predicted_budget_fraction` (default `0.90`); `gpu_vram_gib` is an optional explicit capacity override, and the example leaves it unset. Memory can reject an addition but cannot improve its score
- the packed path is opt-in per job via `packing.eligible: true` and a stable `packing.signature`
- backend compatibility is tracked per canonical backend, so an MPS-process failure cannot poison a CUDA-process profile
- concurrent additions must remain on the active non-exclusive backend
- raw MLEvolve snippet execution remains conservative by default; without an explicit runtime-probe hook they stay exclusive-only for runtime-aware packing

### Admission ranking and memory boundaries

Submission timestamps, elapsed waiting time, and queue sequence do not influence
admission ranking. Static explicit priority selects the bounded candidate window;
workload indicators nominate feasible work. With complete matching student
profiles, cooperative cold-start admission groups feasible candidates whose busy
indicators are less than five points from the group's highest value (idle GPU)
or lowest value (joining). Within each group, higher relative step cost chooses
the idle anchor and lower relative step cost chooses a joining candidate.
Remaining epoch cost, measured time, explicit priority and stable ID break ties.
This five-point grouping is a ranking heuristic, not a calibrated confidence
interval; step cost is relative wall-step work, not FLOPs or GPU occupancy.
It avoids letting tiny near-ceiling busy differences outweigh large work-cost
differences. Missing or incompatible profiles retain the prior busy/remaining-cost
ordering. Known indicators precede unknown; invalid predictions never mean zero
work. The backend-aware path retains its profile-quality, Pareto-risk,
gain, confidence, uncertainty and headroom ordering before this final priority/
workload/identity tie-break. Busy is an ordering hint, not additive GPU capacity.
Final acceptance uses live predicted-work throughput (or measured drain gain
when using the measurement fallback), not the busy score itself.

The historical `PriorityFifoPolicy` name and aging/oldest-window/starvation-time
configuration fields remain accepted for compatibility, but cannot promote or
reserve an old job. Operational timestamps remain for arrival availability,
leases, timeouts, progress, and reporting. The original v2 research proposal
contains historical waiting/fairness ideas; those ideas are not the runtime
admission rules described here.

The default predicted budget and live stop threshold are both 90% of total VRAM.
A live sample at or above 90% immediately stops further admission. The 85%
resume threshold applies only after admission has closed and requires sustained
low usage; it does not cap an open scheduler at 85%. Cooperative allocation
rechecks live free memory before model construction and retains the same 10%
device reserve, instead of subtracting an extra fixed reserve. Existing work is
not paused by this admission gate. External allocations and unmaterialized
reservations may reduce the currently available portion of the budget.

## Limitations In V1

- no distributed scheduling or automatic interception of arbitrary generated Python snippets
- queued command intent is durable in SQLite, but CLI actions rely on the scheduler loop to consume them
- cache payloads assume `torch.save` / `torch.load` compatibility unless a custom loader target is provided

## Tests

```bash
python -m pytest -q localml_scheduler/tests
python -m pytest -q localml_scheduler/tests/test_ml_stress.py
```

## Integrate a code agent with the released predictor

The CPU runtime supports the released **S_A100, S_A10G and S_A10** students and
an explicitly selected compatible transfer student. Select the artifact for the
**deployment GPU**, even when prediction runs on a different CPU host. A10 and
A10G are distinct targets. No predictor retraining is required for this adapter.

1. Obtain a trusted [PerfSeer `paper-release` checkout](https://github.com/JustinLinKK/PerfSeer-predictor/tree/paper-release),
   fetch its checkpoint archive with Git LFS, and extract it. Install the release's
   `portable/requirements-tested.txt` in the scheduler environment.
2. Configure the scheduler's public release code path and extracted model path:

   ```yaml
   prediction:
     mode: ml_predictor
     ml:
       student_release_path: /absolute/PerfSeer-paper-release
       student_model_path: /absolute/extracted/PerfSeer-release-v1/checkpoints/S_A10G
       source_conversion_timeout_seconds: 90
       conversion_max_rss_mb: 4096
       cpu_threads: 1
   gpu_scheduler:
     memory:
       predicted_budget_fraction: 0.90
       live_admission_stop_fraction: 0.90
       live_admission_resume_fraction: 0.85
   ```

   The model directory contains `student.ts`, `normalization.pt`, and
   `architecture.pt`. Their hashes are checked against `assets/manifest.json`,
   which is bound by `RELEASE.json`; embedded hardware and the actual 17-tensor
   TorchScript interface are checked too. The code checkout must be trusted.
3. Have your code agent emit an importable model factory/class or a
   `CandidateAdapter`, together with the **actual training recipe**. Attach it as
   `TrainingJob.metadata["perfseer_model"]`, or write a source-bound JSON sidecar
   with `write_prediction_sidecar`. The latter works with MLEvolve's existing
   `baseline_model_path` submission without changing its executor. Regenerate
   the sidecar after every source mutation.
4. Submit the normal job through `SchedulerHTTPClient` (example above). Keep your
   existing runner's progress reporting, checkpoint and pause/resume contract.
   For a new framework, implement a `run_training_job(context: RunnerContext)`
   callable following [Custom PyTorch Integration](#custom-pytorch-integration)
   and the [working runner example](examples/toy_pytorch_runner.py).
   The predictor describes a training step; it does not execute the full training
   program or replace the runner. Measured branch memory and live VRAM remain
   the admission authority.

For example, this explicit module contract describes a factory `build_model()`
whose forward pass takes `[batch, 4]` floats and returns `[batch, 1]` predictions,
trained with MSE and AdamW. Replace these declarations with your candidate's
real input domains, loss, optimizer and dataset size:

```python
from localml_scheduler.prediction.job_contract import write_prediction_sidecar

recipe = {
    "dataset_size": 8000,
    "drop_last": False,
    "workload_contract": {
        "version": "mlevolve.perfseer_workload.v1",
        "mode": "module",
        "entry": "build_model",
        "inputs": {
            "positional": [{"shape": ["$batch", 4], "dtype": "float32", "distribution": "normal"}],
            "keyword": {},
        },
        "loss": {"kind": "mse", "output_path": [],
                 "target": {"shape": ["$batch", 1], "dtype": "float32", "fill": 0.0}},
        "training": {
            "precision": "fp32_ieee", "gradient_accumulation_steps": 1,
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.01,
                          "foreach": None, "fused": None},
            "scheduler": {"name": "none"}, "backend": "cuda_process",
        },
    },
}
write_prediction_sidecar("/absolute/candidate.py", recipe)
```

Structured inputs, keyword-only masks, bounded token IDs, tuple/dictionary
outputs and the candidate's own scalar training loss are covered by the
[full source contract](docs/source-contract.md). `CandidateAdapter` mode calls
`build_model`, `build_train_batch`, and `training_step`; it checks a declared batch
schema and optimizer. It avoids substituting a generic loss for an agent's real
loss. Explicit `import_roots` support project-local helper modules. The source
and local dependency hashes participate in cache invalidation.

Epoch time equals predicted wall-step time multiplied by optimizer steps per
epoch. Supply the actual training row count, or `steps_per_epoch_by_batch` for a
custom sampler or capped epoch. An explicit per-batch map must include every
requested batch; missing entries trigger fallback rather than silently reverting
to the full dataset. A bare `steps_per_epoch` applies only to the resolved physical
batch. Outputs are wall/GPU step **ms**, epoch **ms**, busy **%**, and average/peak
VRAM **MiB**. These are nominal model output units. The scheduler treats released
epoch cost and busy as **ordering indicators**: it multiplies epoch cost by
remaining epochs for a relative work estimate, and never sums busy or treats it
as free GPU capacity. Whole-job ETA and co-location gains require measured timing.
The [training audit and scheduling contract](docs/predictor-indicator-policy.md)
explain the regression/ranking losses and the single-step label limitation.

For another agent framework, `predict_profile_options(job, batches)` returns one
explicitly predicted profile with step/epoch cost, busy indicator, rounded memory,
batch and model/hardware identity. Keep it separate from measured profile tables.
Use `predict_epoch_cost_options(job, batches)` and
`predict_busy_indicator_options(job, batches)` for hints. `requires_measured_runtime`
is true for released students. The older `predict_seconds_per_epoch_options` and
`predict_busy_percent_options` remain diagnostic compatibility APIs; their names
do not imply calibrated seconds or capacity. Keep measured runtime evidence separate.

Average and peak VRAM predictions are rounded **up to decimal 100 MB blocks**,
with a minimum of 100 MB, matching the predictor release: 470 MB becomes 500 MB.
The API still transports MiB, so the conversion is
`max(1, ceil(raw_MiB * 1048576 / 100000000)) * 100000000 / 1048576`.
Raw model outputs remain available in `last_capture_audits[batch]["raw_prediction"]`
and through `predict_tensors`. Predicted peak VRAM supplies the initial reservation;
an exact matching observation supersedes it. Live VRAM and allocator limits remain
active, and a predicted peak rejection cannot be bypassed by solo fallback.
Rounding is conservative postprocessing, not evidence of improved model accuracy.

**Time quality uses relative ordering**, as in the
[predictor release](https://github.com/JustinLinKK/PerfSeer-predictor/tree/paper-release#model-overview).
For measured durations `a` and `b`, material pairs have
`max(a, b) / min(a, b) >= 1.10`; close material pairs have ratios from 1.10
through less than 1.25. Report wall-step and GPU-step ordering accuracy separately;
correct order receives one credit, a predicted tie half credit, and reversed order
zero. An empty eligible-pair set has no accuracy value. These are relative
comparison criteria, not a fixed 100 ms error allowance. Duration-error diagnostics
can additionally report median and p90 `abs(predicted - measured) / measured`.

For an **unmeasured prediction comparison**, display relative time cost as each
predicted duration divided by the fastest prediction in the named comparison set
(1.00× is fastest). `prediction.released_outputs.relative_time_costs` computes
this display ratio; compare like-for-like step, epoch or total-job durations.
This ratio is not measured ordering accuracy. Raw nominal durations remain
accessible for diagnostics. The scheduler uses measured durations for remaining
runtime and admission gains.

The worker uses the scheduler Python environment; install the candidate's import
dependencies there. It does not automatically install packages.

Source capture currently supports FP32 and accumulation 1. AMP, unsupported
control flow, non-importable programs, unknown recipes and larger accumulation
require a matching strict precomputed training graph or measured-profile
fallback. The worker is CPU-only with a timeout and Linux RSS watchdog; its
memory budget is separate from the GPU's 90% training budget. Failure details
identify conversion/recipe problems without marking the training job failed.
An isolated subprocess is not a security sandbox: use trusted local code or an
external sandbox for untrusted candidates. Network access is disabled during
capture; pretrained downloads are not part of prediction.

Run actual artifact integration tests from the parent of `localml_scheduler`:

```bash
CUDA_VISIBLE_DEVICES='' \
PERFSEER_RELEASE_ROOT=/absolute/PerfSeer-paper-release \
PERFSEER_STUDENT_MODEL_DIR=/absolute/extracted/PerfSeer-release-v1/checkpoints/S_A100 \
PERFSEER_STUDENT_MODELS_ROOT=/absolute/extracted/PerfSeer-release-v1/checkpoints \
python -m pytest localml_scheduler/tests/test_released_a100_student.py \
  localml_scheduler/tests/test_released_hardware_students.py \
  localml_scheduler/tests/test_source_contract.py \
  localml_scheduler/tests/test_prediction_sidecar.py -q
```

Extract all four student artifacts for this full matrix; select `-k A10G` in
`test_released_hardware_students.py` when checking only S_A10G.

These test graph/recipe fidelity, actual CPU student inference, hardware and
artifact identity, and scheduler consumption. They do not measure prediction
accuracy or throughput on a newly generated candidate. The PetFinder check below exercises newly generated MLEvolve source.


### Local MLEvolve PetFinder integration result

Two fresh raw drafts were generated by the local MLEvolve code-agent path using
Codex CLI, then passed through the scheduler's source-bound sidecar interface
and the actual S_A10G student on CPU. Both native forward/loss/backward replay
and the scheduler time, memory and busy APIs agreed. The recipe is batch 4,
FP32 AdamW/MSE, with an explicit eight-step epoch cap.

| Draft | Design | Relative predicted epoch cost | Rounded avg / peak VRAM (decimal MB) | Busy ordering |
| --- | --- | ---: | ---: | ---: |
| candidate_00 | Three-stage CNN + tabular MLP | 1.00× | 900 / 900 | Lighter |
| candidate_01 | Four-stage CNN + tabular MLP | 1.13× | 1000 / 1000 | Heavier |

The reference is candidate_00, the lower epoch-cost prediction in this two-draft
comparison. Both use the same eight-step, three-epoch budget, so the full-job
predicted cost ratio is also 1.13×. Raw time/VRAM outputs and unrounded ratios are
retained in the receipt. No measured A10G durations are available for these drafts,
so their time-ordering accuracy is unmeasured.

These are **predictions**, not measured GPU accuracy or completed training runs.
The drafts are not presented as accepted MLEvolve training candidates: the first
encountered existing provider/preflight infrastructure problems, and the second
was rejected by deterministic training-review checks. Those statuses, exact
source hashes, and the successful graph/inference checks are retained in the
[provenance](verification/mlevolve_petfinder_20261027/provenance.json) and
[original raw prediction receipt](verification/mlevolve_petfinder_20261027/predictions.json).
The [updated output receipt](verification/mlevolve_petfinder_20261027/predictions-rounded-100mb.json)
records the actual repeated CPU predictions, decimal rounding, and relative time costs.
The [indicator-policy receipt](verification/mlevolve_petfinder_20261027/predictions-indicator-policy.json)
checks the new ranking APIs against both actual workload predictions from S_A10G:
candidate_01 is preferred as the heavier anchor, candidate_00 as the lighter joiner.

Reproduce the prediction check from the scheduler's parent directory, with the
candidate dependencies installed in that Python environment:

```bash
CUDA_VISIBLE_DEVICES='' python \
  localml_scheduler/verification/mlevolve_petfinder_20261027/verify_predictions.py \
  --release-root /absolute/PerfSeer-paper-release \
  --model-dir /absolute/extracted/PerfSeer-release-v1/checkpoints/S_A10G \
  --sources localml_scheduler/verification/mlevolve_petfinder_20261027/raw \
  --output /tmp/petfinder-predictions.json
```

The original integration CPU suite recorded 382 passed tests, 11 passed subtests and three
intentional skips (two live-GPU tests and the optional input-only TRAIN probe).
One optional CUDA-document search test fails identically at the untouched base
revision; it is recorded in the [full verification report](verification/scheduler-integration-20261027.json).

The subsequent [prediction-first workflow review](docs/predictor-indicator-policy.md#full-workflow-replace-exclusive-profiling-retain-online-feedback)
corrects runtime use of these outputs. The original drafts lack cooperative
training hooks, so they demonstrate source prediction, not readiness for overlap.
Use the runner contract above for actual training. CPU state-machine tests verify
admission with empty profile stores and zero per-job solo-reference phases.
Latest CPU regression: **428 passed**, 11 subtests passed, three optional/live-GPU
checks skipped; one previously reproduced unrelated CUDA-document-search test
was explicitly deselected. See the [workflow verification receipt](verification/prediction-first-workflow-20261027.json).
For agent-generated training scripts, use the [shared training wrapper](docs/shared-training-wrapper.md).
It owns checkpoint recovery, data position and optimizer-step safe points while
the agent supplies its model, loss and dataset. The two PetFinder drafts now have
adapted runnable copies under `verification/mlevolve_petfinder_20261027/adapted/`;
the original drafts remain preserved. MLEvolve detects the canonical wrapper
entry, and runtime checks reject a recipe that differs from the predictor sidecar.
