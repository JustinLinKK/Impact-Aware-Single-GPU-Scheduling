# localml_scheduler

`localml_scheduler` is a reusable local-first ML job manager for single-machine agent workflows. V1 focuses on two practical capabilities:

- a single-GPU scheduler with priority queueing, safe-point pause/resume, persistence, and restart recovery
- a RAM-backed baseline-model cache with optional LRU entry-capacity and RAM-percent limits that keeps immutable CPU-side baselines warm and serves isolated copies to worker subprocesses
- time-aware GPU packing that chooses ready jobs, batch sizes, and concurrent backends from predicted completion time and compatibility evidence, with average VRAM used only as a safety gate
- Linux overlap on a configured non-exclusive backend, admitted incrementally by the time-aware policy
- reserved exclusive five-option calibration that persists timing and VRAM measurements for time-aware planning
- one-epoch runtime profiling that makes new job families pack-eligible after the first exclusive calibration run
- optional hardware-selected PerfSeer student prediction, with CPU-only
  TorchScript inference and per-job branch-profile fallback

It is intentionally packaged as a root-level module so it can be used by MLEvolve or detached and integrated into other agent pipelines.

## Architecture

- `schemas.py`: serializable job, checkpoint policy, progress, and cache schemas
- `scheduler/`: policy, queue, service loop, recovery, and worker supervision
- `scheduler/placement_planner.py`: shortest-anchor, one-newcomer-at-a-time placement planning
- `scheduler/time_objective.py`: verified sequential-versus-packed drain-time scoring
- `scheduler/telemetry.py`: lightweight `nvidia-smi` device telemetry for solo and packed runs
- `execution/`: subprocess launcher, file-based control plane, worker entrypoint, and runner context
- `execution/backends.py`: exclusive and MPS-backed launch backends
- `checkpointing/`: atomic local checkpoint save/load
- `model_cache/`: in-memory LRU baseline cache plus a local socket server for worker access
- `storage/`: SQLite-backed jobs, commands, checkpoints, cache metadata, and event history
- `observability/`: JSONL events, log files, and aggregate reports
- `profiling/`: exclusive five-option measurement plus runtime profile helpers
- `prediction/`: scheduler integration for isolated source conversion and the
  PerfSeer submodule's CPU runtime
- `examples/`: toy PyTorch training runner and a demo script
- `adapters/`: thin helpers for wiring job submission from MLEvolve or other systems

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
```

Run the demo:

```bash
python -m localml_scheduler.examples.demos submit
python -m localml_scheduler.examples.demos bridge
```

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

Compatibility wrappers such as `get_job_design_context(...)`,
`search_hardware_features(...)`, `get_hardware_feature_context(...)`, and
`get_hardware_optimization_context(...)` remain available for older callers.

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

## Time-aware scheduling

`parallel_time_aware` starts the shortest predicted solo job as an anchor and
then admits one candidate at a time, shortest remaining runtime first. A
starving job overrides that ordering. Each addition must pass scheduling class,
parallel cap, backend and pair compatibility, live-admission, batch-option, and
predicted average-VRAM checks before any concurrent work begins.

`gpu_scheduler.scheduler_decision_mode` selects the trial ordering layer:

- `baseline` (the default) preserves the shortest-anchor and shortest-newcomer
  behavior described above.
- `backend_awared` jointly enumerates pairs at an empty-GPU boundary and ranks
  empty-GPU pairs or active-group newcomers with deterministic Pareto fronts.
  The risk vectors are derived from a CPU-only AST fingerprint and differ for
  MPS, CUDA streams, and ordinary CUDA processes. Static analysis only orders
  candidates; unknown placements still require the same measured live trial.

Backend-aware mode applies backend compatibility, batch/accuracy, conservative
VRAM overhead, known-bad exact-profile, and optimistic trial-amortization gates
before ranking. Exact profile identity includes hardware/runtime identity,
source/graph signature, dtype, batch vector, backend, and backend configuration;
epoch count and submission time are excluded. MPS pair templates are limited to
50/50, 60/40, and 40/60, and stream offsets to 0, quarter-step, and half-step.
Nonzero stream offsets are rejected when step time is unavailable.

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
trial replaces rather than averages with an expired profile. Legacy profiles
without confirmation metadata are never trusted for immediate rejection.

### Task-scoped placement replay

Jobs may provide a typed `workload_identity` with `task_key`, `dataset_key`,
`architecture_key`, and `architecture_family`. The MLEvolve adapter also
accepts these as direct arguments and uses `workflow_id` only as a legacy task
fallback. Script introspection emits normalized exact and broad architecture
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

The local ML adapter consumes an epoch-time prediction only when the selected
artifact explicitly declares the canonical `train_epoch_ms` target. Legacy
PerfSeer `train_time` output is not treated as a full-epoch estimate.

The former throughput controls `makespan_weight`, `flow_time_weight`, and
`min_aggregate_gain` are no longer accepted. `time_v3_flow_only`,
`time_v4_colocation_gain`, and `time_v5_piecewise_drain` are migrated to
`time_v6_verified_piecewise_drain` with a warning. The `colocation` defaults are
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

It compares serial FIFO, legacy VRAM-fill packing, the recursive time-aware
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
- `parallel_job_cap` is optional (`null` means incremental admission has no fixed-width cap)
- the memory ceiling is detected/configured total VRAM times `predicted_budget_fraction`; it can reject an addition but cannot improve its score
- the packed path is opt-in per job via `packing.eligible: true` and a stable `packing.signature`
- backend compatibility is tracked per backend, so an MPS failure does not automatically poison a stream pairing
- concurrent additions must remain on the active non-exclusive backend
- raw MLEvolve snippet execution remains conservative by default; without an explicit runtime-probe hook they stay exclusive-only for runtime-aware packing

## Limitations In V1

- no distributed scheduling or automatic interception of arbitrary generated Python snippets
- queued command intent is durable in SQLite, but CLI actions rely on the scheduler loop to consume them
- cache payloads assume `torch.save` / `torch.load` compatibility unless a custom loader target is provided

## Tests

```bash
python -m pytest -q localml_scheduler/tests
python -m pytest -q localml_scheduler/tests/test_ml_stress.py
```
