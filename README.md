# localml_scheduler

`localml_scheduler` is a reusable local-first ML job manager for single-machine agent workflows. V1 focuses on two practical capabilities:

- a single-GPU scheduler with priority queueing, safe-point pause/resume, persistence, and restart recovery
- a RAM-backed baseline-model cache with optional LRU entry-capacity and RAM-percent limits that keeps immutable CPU-side baselines warm and serves isolated copies to worker subprocesses
- packed GPU scheduling for structured runner jobs, including fixed-width packed groups and `parallel_auto_pack` admission that targets VRAM or SM utilization
- optional Linux hybrid overlap across `mps` and `stream` backend groups on one GPU when concurrent groups are enabled
- optional exclusive-path batch-size probing with SQLite-backed reuse for repeated model/device/shape combinations
- one-epoch runtime profiling that makes new job families pack-eligible after the first exclusive calibration run

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
- the packed path is opt-in per job via `packing.eligible: true` and a stable `packing.signature`
- backend compatibility is tracked per backend, so an MPS failure does not automatically poison a stream pairing
- Linux deployments can enable `concurrent_groups_enabled: true` with `concurrent_backend_allowlist: ["mps", "stream"]` to overlap an MPS group and a stream group on the same GPU
- raw MLEvolve snippet execution remains conservative by default; without an explicit runtime-probe hook they stay exclusive-only for runtime-aware packing

## Limitations In V1

- no three-way packing, distributed scheduling, or automatic interception of arbitrary generated Python snippets
- queued command intent is durable in SQLite, but CLI actions rely on the scheduler loop to consume them
- cache payloads assume `torch.save` / `torch.load` compatibility unless a custom loader target is provided

## Path To V2

The current design keeps resource requirements, placement decisions, scheduling policy, and worker supervision separate so future work can add:

- co-run compatibility prediction
- two-job GPU packing and richer placement logic
- interference-aware dispatch and profiling feedback

Those changes should fit without replacing the job schema, storage model, or training runner contract.

## Tests

```bash
python -m unittest discover localml_scheduler/tests
```
