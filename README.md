# localml_scheduler

`localml_scheduler` is a reusable local-first ML job manager for single-machine agent workflows. V1 focuses on two practical capabilities:

- a single-GPU scheduler with priority queueing, safe-point pause/resume, persistence, and restart recovery
- a RAM-backed baseline-model cache with optional LRU entry-capacity and RAM-percent limits that keeps immutable CPU-side baselines warm and serves isolated copies to worker subprocesses
- time-aware GPU packing that chooses ready jobs, batch sizes, and concurrent backends from predicted completion time, average VRAM, and compatibility evidence
- optional Linux hybrid overlap across `mps` and `stream` backend groups on one GPU when concurrent groups are enabled
- optional exclusive-path batch-size probing with SQLite-backed reuse for repeated model/device/shape combinations
- one-epoch runtime profiling that makes new job families pack-eligible after the first exclusive calibration run
- optional hardware-selected PerfSeer student prediction, with CPU-only
  TorchScript inference and per-job branch-profile fallback

It is intentionally packaged as a root-level module so it can be used by MLEvolve or detached and integrated into other agent pipelines.

## Architecture

- `schemas.py`: serializable job, checkpoint policy, progress, and cache schemas
- `scheduler/`: policy, queue, service loop, recovery, and worker supervision
- `scheduler/gpu_scheduler.py`: GPU placement planning based on VRAM headroom, compatibility history, runtime skew, and optional auto-pack targets
- `scheduler/telemetry.py`: lightweight `nvidia-smi` device telemetry for solo and packed runs
- `execution/`: subprocess launcher, file-based control plane, worker entrypoint, and runner context
- `execution/backends.py`: exclusive and MPS-backed launch backends
- `checkpointing/`: atomic local checkpoint save/load
- `model_cache/`: in-memory LRU baseline cache plus a local socket server for worker access
- `storage/`: SQLite-backed jobs, commands, checkpoints, cache metadata, and event history
- `observability/`: JSONL events, log files, and aggregate reports
- `profiling/`: exclusive-path batch probe controller plus runtime profile helpers
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
python -m localml_scheduler.examples.demo_submit_jobs
python -m localml_scheduler.examples.demo_mlevolve_bridge
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

Structured runners can also expose an optional batch probe hook in `module:function` form. When `batch_probe.enabled: true` is set on a GPU job running through the exclusive backend, the worker can probe candidate batch sizes before training, persist the selected result in SQLite, and reuse it for later matching jobs.

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

All placement modes use average used GPU VRAM in MiB (`avg_vram_mb`) for
packing. In `parallel_time_aware`, device-level memory is averaged over a time
window: sustained use at the stop threshold closes only new packed admission,
and active work continues normally. Admission reopens after a complete window
below the resume threshold. GPU/SM utilization is not a placement input for
this mode.

## Time-aware scheduling

`parallel_time_aware` requests the five exponent offsets `[-2,-1,0,1,2]`
around each immutable originally requested power-of-two batch size. Exact
batch observations are preferred, and a missing runtime estimate never becomes
zero. Dominated memory/runtime choices are removed before bounded exact or beam
search.

For candidate pack `g`, member `j` has remaining time `p_j`, completion offset
`d_j=p_j`, and drain time `D=max_j(d_j)`. The implemented score is minimized:

```text
L_F = sum(selected weight_j * d_j) + D * sum(unselected weight_j)
score = L_F / max(epsilon, L_F(exclusive anchor))
```

Average VRAM, the parallel cap, backend availability, compatibility history,
live admission state, and exclusive-probe drain state are hard constraints.
There is no VRAM-fill reward. Slowdown prediction and throughput/makespan
controls are intentionally deferred: stored slowdown evidence never affects
placement, and throughput is reported only after execution. Jobs crossing the
starvation timeout become mandatory anchors; if no pack is feasible, the oldest
anchor runs exclusively at the next legal drain boundary.

The former throughput controls `makespan_weight`, `flow_time_weight`, and
`min_aggregate_gain` are no longer accepted. Remove them from existing
configuration before starting the scheduler; `objective.priority_weight` and
`objective.objective_version` remain supported.

Jobs with `scheduling_class: exclusive_probe` reserve the next idle boundary.
Existing packs drain without preemption and no normal work is admitted during
the reservation. Early stopping is evaluated synchronously at epoch safe
points and persists its patience state in job metadata across pause/restart.
`save_best_checkpoint` protects the tagged best checkpoint from normal pruning.
Reports include saved epochs and estimated wall time saved when the runner
provides a remaining-runtime estimate (or epoch step timing).
Generic runners still own model-state restoration; `restore_best_checkpoint`
is reserved for a runner-level restore hook and is not applied implicitly.

The deterministic validation fixture can be run with:

```bash
python -m localml_scheduler.scheduler.trace_simulator
```

It compares serial FIFO, legacy VRAM-fill packing, the time-aware policy, and
an exhaustive small-trace oracle, reporting makespan, flow time, waiting,
starvation, jobs/hour, predicted/actual VRAM, realized slowdown, and saved
early-stop work. The simulator accepts live backend changes, rolling-memory
samples, compatibility matrices, realized-slowdown matrices, and validation
sequences. Slowdown changes simulated actual completion but is not visible to
the placement policy. The exhaustive
oracle remains intentionally limited to small, non-preemptive drain-boundary
fixtures.

The real-GPU benchmark performs a hardware-specific five-option calibration,
then runs serial FIFO, the previous fill-based policy, and time-aware packing
at least twice from isolated runtime directories. It records means, sample
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

- `parallel_default` and `parallel_batch_optimized` keep the legacy fixed-width packed-group behavior and still fall back to exclusive execution when compatibility or memory evidence is missing
- `parallel_auto_pack` ignores `max_packed_jobs_per_gpu` and keeps admitting work until the configured `auto_pack.target_metric` (`vram` or `sm`) is close to its target threshold
- `parallel_time_aware` uses `parallel_job_cap` (`null` means unlimited); if absent, an explicitly supplied legacy `max_packed_jobs_per_gpu` is mapped to it
- `safe_vram_budget_gib` remains a compatibility input for older modes; time-aware mode uses detected/configured total VRAM times `predicted_budget_fraction`
- the packed path is opt-in per job via `packing.eligible: true` and a stable `packing.signature`
- backend compatibility is tracked per backend, so an MPS failure does not automatically poison a stream pairing
- Linux deployments can enable `concurrent_groups_enabled: true` with `concurrent_backend_allowlist: ["mps", "stream"]` to overlap an MPS group and a stream group on the same GPU
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
