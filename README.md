# localml_scheduler

`localml_scheduler` is a reusable local-first ML job manager for single-machine agent workflows. The current version focuses on:

- a single-GPU scheduler with priority queueing, safe-point pause/resume, persistence, and restart recovery

- a RAM-backed baseline-model cache with optional LRU entry-capacity and RAM-percent limits that keeps immutable CPU-side baselines warm and serves isolated copies to worker subprocesses

- packed GPU scheduling including fixed-width packed groups and a `parallel_auto_pack` admission path that targets VRAM or SM utilization

- four execution backends (exclusive, MPS, stream, cuda_process) selectable per pack

- optional Linux hybrid overlap across `mps` and `stream` backend groups on one GPU when concurrent groups are enabled

- optional batch-size probing with SQLite-backed reuse for repeated model/device/shape combinations

- one-epoch runtime profiling that makes new job families pack-eligible after the first exclusive calibration run

- a SchedulerKnowledgeBase + FastMCP server that exposes read-only graph queries and Qdrant-backed hardware feature search to external agents

It is packaged as a root-level module so it can be used by MLEvolve or detached and integrated into other agent pipelines.

## Package Layout (verified against HEAD)

- `__init__.py` (top): re-exports `SchedulerClient` from `client.py:1`, `SchedulerConfig` from `config/__init__.py`, `SchedulerEngine` from `engine.py`, plus domain types and DTOs (`__init__.py:1`-30)

- `client.py`: thin RPC-style client surface for submitting jobs and querying state (`client.py:1`-180)

- `engine.py`: factory that wires `SchedulerConfig` into `SchedulerService` plus stores (`engine.py:1`-50)

- `dto.py`: serializable request/response models (`JobCommandRequest`, `JobQuery`, `PreloadRequest`, `ReportQuery`, `SubmitJobRequest`)

- `cli.py`: argparse-driven CLI used for `scheduler start`, `scheduler mcp`, `hardware-features ingest`, `submit`, `list`, `status`, `pause`, `resume`, `cancel`, `preload`

- `mcp_server.py`: `build_mcp_server()` registers tools on FastMCP (see MCP Graph Surface section)

- `graph_knowledge.py`: `SchedulerKnowledgeBase` (`graph_knowledge.py:26`) is the read-only query layer used by `mcp_server.py`

- `hardware.py`: `HardwareProfile` data class plus host-side detection helpers

- `scheduler/`: planner, queue, service loop, supervisor (see Scheduler Internals)

- `domain/`: domain models (`TrainingJob`, profiles, identity helpers, progress snapshots)

- `execution/`: subprocess launcher, backends, file-based control plane, worker entry, stream host

- `storage/`: `StateStore` facade plus SQLite, Neo4j mirror, and Postgres log-store backends

- `profiling/`: `batch_probe.py` and `runtime_probe.py`

- `model_cache/`: in-memory LRU baseline cache plus a local socket server for worker access

- `observability/`: event logger, metrics collector, scheduler-logger setup

- `checkpointing/`: atomic local checkpoint save/load

- `adapters/`: `mlevolve.py` builds `TrainingJob` from MLEvolve runner specs

- `hardware_features/`: Qdrant-backed semantic search store + seed records

- `configs/`: example YAML settings (single-machine, nautilus, full-stack variants)

- `runtime/`: per-run runtime artifacts created at start-up

- `examples/`: toy PyTorch runner plus demo submission scripts

- `tests/`: 15 test modules including new `test_graph_db_validation.py` and `test_hardware_features.py`

## Scheduler Internals (`scheduler/`)

- `service.py` `class SchedulerService` (`service.py:49`) owns the main loop. `run_forever()` (`service.py:153`) repeats: `_poll_active_workers`, `_process_commands`, `_warm_cache`, `_poll_telemetry`, `_enforce_packed_safety`, `_maybe_preempt`, `_dispatch_pending_work` per `scheduler_poll_interval_seconds`

- `placement_planner.py` `class PlacementPlanner` (`placement_planner.py:26`) composes `ResourceEstimator`, `CompatibilityEvaluator`, `RuntimeGuardrail`, `CandidateGenerator`, `ObjectiveScorer` (`placement_planner.py:33`-44). `choose_plan()` (`placement_planner.py:61`) returns a `DispatchPlan` or `None`

- `candidate_generator.py` `class CandidateGenerator` (`candidate_generator.py:13`) yields candidate job groups via `candidate_groups()` (`candidate_generator.py:87`) and per-job batch-size grids via `candidate_batch_sizes()` (`candidate_generator.py:50`) — power-of-two grid path at `candidate_generator.py:56`-66, range path at `candidate_generator.py:67`-73

- `compatibility.py` `class CompatibilityEvaluator` (`compatibility.py:50`). `compatible_group()` (`compatibility.py:63`) rejects packs when any pair has `SoloProfile.avg_gpu_utilization` above `pack_reject_sm_active_ge` or when a `PairProfile` is on cooldown / over `pack_reject_max_slowdown`. Pair scoring lives in `compatibility_score()` (`compatibility.py:13`-39)

- `objective.py` `class ObjectiveScorer` (`objective.py:17`). Three scorers: `evaluate_fixed_group()` (`objective.py:34`) for `PARALLEL_DEFAULT`, `evaluate_optimized_group()` (`objective.py:56`) for `PARALLEL_BATCH_OPTIMIZED` (caches via `CombinationProfile`), `evaluate_auto_pack_group()` (`objective.py:126`) for `PARALLEL_AUTO_PACK`

- `resource_estimator.py` `class ResourceEstimator` (`resource_estimator.py:11`). VRAM fallback chain in `estimate_peak_vram_mb()` (`resource_estimator.py:52`-91): exact `BatchSizeObservation` → nearest-batch interpolation → `BatchProbeProfile` → `SoloProfile` → `job.resource_requirements.estimated_vram_mb` → `0.0`. SM estimate uses a shorter chain (`resource_estimator.py:93`-117)

- `runtime_guardrail.py` `class RuntimeGuardrail` (`runtime_guardrail.py:11`). `runtime_penalty()` (`runtime_guardrail.py:16`) returns `(penalty: float, hard_reject: bool)`. Hard-reject fires (`runtime_guardrail.py:35`) only when every job in the group has a `source="probe"` profile and the ratio `max / min` exceeds `auto_pack.runtime_skew_guardrail_ratio`

- `planning_repository.py` `class PlanningRepository` (Protocol) wraps the store reads the planner needs: `hardware_profile`, `get_solo_profile`, `get_pair_profile`, `get_batch_probe_profile`, `get_batch_size_observation`, `best_combination_profile`

- `policies.py`: scheduling policies (priority/age ordering) consumed by `RunnableJobQueue`

- `queue.py` `class RunnableJobQueue`: orders queued jobs per policy

- `supervisor.py` `class WorkerSupervisor` (`supervisor.py:50`). Dispatches a `DispatchPlan` to the right backend via `BackendRegistry` (`supervisor.py:64`). Returns a `PlacementGroupHandle` (`supervisor.py:35`) that tracks worker processes

- `recovery.py`: scans persisted job state on start and reconciles recoverable jobs

- `telemetry.py`: lightweight `nvidia-smi` sampler used by `service.py:_poll_telemetry`

- `planner_types.py`: `DispatchPlan`, `EvaluatedGroup` dataclasses returned by the planner

## Scheduler Modes (`config/models.py:13`-17)

- `SCHEDULER_MODE_SERIAL_BASIC = "serial_basic"` — single job at a time, exclusive backend

- `SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED = "serial_batch_optimized"` — single job plus exclusive-path batch probe

- `SCHEDULER_MODE_PARALLEL_DEFAULT = "parallel_default"` — fixed-width packed groups, scored by `evaluate_fixed_group`

- `SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED = "parallel_batch_optimized"` — packed groups with per-job batch search, scored by `evaluate_optimized_group`, cached in `CombinationProfile`

- `SCHEDULER_MODE_PARALLEL_AUTO_PACK = "parallel_auto_pack"` — admission targets `auto_pack.target_metric` (`vram` or `sm`), guarded by `RuntimeGuardrail`

## Execution Backends (`execution/backends.py`)

- `ExclusiveBackend` (`backends.py:54`): single-job launch, sets `CUDA_VISIBLE_DEVICES`, one subprocess per job, one CUDA context

- `CudaProcessBackend` (`backends.py:73`): N-job pack, each job in its own subprocess and CUDA context, sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` per the `CudaProcessSettings`. Driver time-slices the contexts

- `MPSBackend` (`backends.py:93`): N-job pack, starts `nvidia-cuda-mps-control -d` (`backends.py:142`-148), allocates `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` per job (`backends.py:121`-140), one MPS-merged CUDA context shared across subprocesses

- `StreamBackend` (`backends.py:161`): N-job pack spawned as one host process running `python -m localml_scheduler.execution.stream_host` (`backends.py:179`-187). Inside the host, `_run_job_in_thread()` (`stream_host.py:23`) runs each job in its own `threading.Thread` (`stream_host.py:64`) with its own `torch.cuda.Stream()` (`stream_host.py:32`-35), all sharing one Python interpreter and one CUDA context

- `BackendRegistry` (`execution/backend_registry.py`) selects the active backend per `DispatchPlan` and reports availability to `WorkerSupervisor`

## Domain Models (`domain/`)

- `domain/jobs.py`: `JobStatus` enum (`jobs.py:13`), `SafePointType` (`jobs.py:29`), `CommandType` (`jobs.py:37`), `ResourceRequirements` (`jobs.py:70`), `PackingSpec` (`jobs.py:86`), `BatchProbeSpec` (`jobs.py:111`), and the main `TrainingJob` dataclass

- `domain/profiles.py`: `SoloProfile` (`profiles.py:39`), `BatchProbeTrialResult` (`profiles.py:72`), `BatchProbeProfile` (`profiles.py:88`), `BatchSizeObservation` (`profiles.py:127`), `PairProfile` (`profiles.py:172`), `CombinationProfile` (`profiles.py:236`), `RuntimeProfile` (`profiles.py:308`), `RunProfile` (`profiles.py:382`), `JobCommand` (`profiles.py:468`)

- `domain/identity.py`: signature builders such as `build_group_signature`, `build_backend_scoped_pair_key`, `build_combination_key`, `build_runtime_profile_key`

- `domain/batching.py`: `BatchResolution.resolved_batch_size()` consumed by both planner and runtime probe

- `domain/common.py`: timestamp parsing and primitive conversion helpers

- `domain/progress.py`: `JobProgress`, `PlacementAssignment`, `ProgressSnapshot`

## Storage (`storage/`)

- `state_store.py`: `class StateStore` (`state_store.py:123`) is the canonical writable facade and `_MirrorStateStore` (`state_store.py:15`) layers SQLite primary + best-effort Neo4j mirror

- `sqlite_store.py`: `class SQLiteStateStore` (`sqlite_store.py:32`) — persistent SQLite source of truth for jobs, profiles, commands, cache metadata

- `neo4j_store.py`: `class Neo4jStateStore` (`neo4j_store.py:73`) — optional Neo4j mirror that keeps a property-graph view of jobs and profiles for the MCP knowledge surface

- `log_store.py`: `class SchedulerLogStore` (`log_store.py:21`) — append-only Postgres analytics for sessions, events, metrics

- `repositories.py`: concrete `PlanningRepository` implementations backed by the stores above

- `models.py`: shared row schemas used by the SQL backends

- `sqlite_bundle.py`: bundles the SQLite schema migration entry points

## Execution Detail (`execution/`)

- `executor.py` `class SubprocessExecutor` (`executor.py:24`): `start(job)` (`executor.py:31`) calls `subprocess.Popen(...)` (`executor.py:47`) with `python -m localml_scheduler.execution.worker_entry --runtime-root <root> --job-id <id>` and returns a `WorkerProcessHandle` (`executor.py:16`)

- `worker_entry.py` `_run_job(runtime_root, job_id)` (`worker_entry.py:17`): loads settings + state store, runs `run_batch_probe_preflight()` (`worker_entry.py:28`), resolves the runner target, executes it, and handles `PauseRequested` / `CancelRequested`

- `worker_runtime.py`: shared helpers, including `create_runner_context()` that wires the `RunnerContext` exposed to user runners

- `runner_protocol.py`: `RunnerContext` dataclass (`runner_protocol.py:45`-56) carrying `job`, `settings`, `store`, `event_logger`, `control_hook`, `checkpoint_manager`, `cache_client`

- `control.py`: file-based pause/cancel signalling. The worker reads a JSON command file at `job_command_path(job_id)` and raises `PauseRequested` / `CancelRequested` (`control.py:17`-21)

- `stream_host.py`: one Python process for all jobs in a stream-backend pack. Each job runs in its own thread with a private `torch.cuda.Stream()` inside one shared CUDA context (`stream_host.py:32`-35, `stream_host.py:64`)

- `backend_registry.py`: maps backend names to backend instances and reports availability for the supervisor and planner

## Profiling (`profiling/`)

- `batch_probe.py`: `run_batch_probe_preflight(context)` (`batch_probe.py:513`) decides between cache hit / cache miss for the job's `BatchProbeProfile`. On miss it calls `_run_probe_controller()` (`batch_probe.py:285`) which drives a binary or power-of-two search via the job's runner-side probe hook. Events emitted include `batch_probe_started`, `batch_probe_cache_hit`, `batch_probe_cache_miss`, `batch_probe_trial`, `batch_probe_failed`, `batch_probe_selected`, `batch_probe_warning`

- `runtime_probe.py`: `runtime_profile_for_job(store, job, backend_name)` (`runtime_probe.py:21`) looks up the persisted `RuntimeProfile` for `(packing.signature, resolved_batch_size, backend_name)`. The `RuntimeProfile` (`profiles.py:308`) stores `startup_seconds`, `epoch_1_seconds`, `steps_per_epoch`, `avg_step_time_ms`, `estimated_total_runtime_seconds`, `confidence`, `observations`, `source`

## Observability (`observability/`)

- `events.py` `class EventLogger`: `emit(event_type, job_id, payload)` writes to JSONL plus `StateStore` plus `SchedulerLogStore`

- `metrics.py` `class MetricsCollector`: `build_report()` aggregates scheduler state into a `SchedulerReport`

- `logging_utils.py` `setup_scheduler_logger()`: configures dual stream + file handlers for the scheduler log path

## Model Cache (`model_cache/`)

- `cache_server.py`: small Unix-domain (or TCP) socket server, length-prefixed pickle framing, threading server class with `CacheRequestHandler` for worker access

- `baseline_cache.py`: LRU RAM cache keyed by model id. Eviction triggers when `memory_budget_bytes` or `max_ram_percent` is exceeded, skipping pinned entries

- `warming.py`: `select_models_to_warm(jobs, top_k=2, selection_policy="top_k")` chooses which baselines to preload before dispatch

## MCP Graph Surface (`mcp_server.py`)

`build_mcp_server(settings)` (`mcp_server.py:13`) constructs a FastMCP server fed by `SchedulerKnowledgeBase`. The registered tools are:

- `get_job_graph_context(job_id)`

- `search_hardware(query=None, limit=10)`

- `get_hardware_context(hardware_key="current", include_scheduler_limits=True)`

- `get_job_design_context(candidate, limit=5)`

- `search_profiles(...)`

- `get_runtime_estimate(...)`

- `recommend_batch_size(...)`

- `recommend_epochs(...)`

- `get_packet_compatibility(...)`

- `search_profile_summaries(query, limit=20)`

- `search_hardware_features(...)`

- `get_hardware_feature_context(...)`

All tools are read-only. They summarize graph evidence and config-derived scheduler limits but do not mutate jobs, profiles, or event history.

The hardware feature tools use Qdrant as an external vector service. For local development, start Qdrant and ingest the repo-curated seed corpus:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
python -m localml_scheduler.cli knowledge ingest-schema --config config.yaml --schema-root schema
```

Use `--dry-run` to validate and summarize seed records without writing to Qdrant. MCP search/context calls return empty results instead of failing the scheduler when the feature database is disabled or unavailable.

## How To Run

Start the scheduler:

```bash
python -m localml_scheduler.cli scheduler start --config config.yaml
```

Run the MCP stdio server:

```bash
python -m localml_scheduler.cli scheduler mcp
```

Submit a job:

```bash
python -m localml_scheduler.cli submit localml_scheduler/examples/job.example.yaml --config config.yaml
```

Inspect state:

```bash
python -m localml_scheduler.cli list
python -m localml_scheduler.cli status <job_id>
```

Control commands:

```bash
python -m localml_scheduler.cli pause <job_id>
python -m localml_scheduler.cli resume <job_id>
python -m localml_scheduler.cli cancel <job_id>
python -m localml_scheduler.cli preload <spec.yaml>
```

Run the demos:

```bash
python -m localml_scheduler.examples.demo_submit_jobs
python -m localml_scheduler.examples.demo_mlevolve_bridge
```

## Custom PyTorch Integration

Point a job at a runner target in `module:function` form:

```yaml
config:
  runner_target: "my_pkg.training:run_training_job"
```

The target receives a `RunnerContext` with `job`, `settings`, `store`, `event_logger`, `control_hook`, `checkpoint_manager`, and `cache_client` (see `execution/runner_protocol.py:45`).

Structured runners can expose:

- a batch-probe hook used when `batch_probe.enabled: true` is set on a GPU job running through the exclusive backend; probed results are persisted in SQLite and reused for matching jobs

- a runtime-probe contract via `runtime_probe.enabled: true`. The default `epoch_1` strategy treats the first exclusive epoch as calibration, persists a `RuntimeProfile` keyed by workload signature, hardware, backend, and resolved batch size, and then uses that estimate to reject badly skewed packed groups. Jobs without reliable epoch semantics can use `runtime_probe.strategy: "step_window"` instead

- a `preload_source` with `model_id`, `model_path`, and `loader_target`. When present, the scheduler warms that shared source in RAM instead of the job's normal baseline target

The pause flow is:

- scheduler requests pause

- worker reaches the next safe point

- checkpoint is saved atomically

- worker exits cleanly

- scheduler later redispatches the paused job from checkpoint

## Packed Execution Notes

- `parallel_default` and `parallel_batch_optimized` use fixed-width packed groups and fall back to exclusive execution when compatibility or memory evidence is missing

- `parallel_auto_pack` ignores `max_packed_jobs_per_gpu` and keeps admitting work until the configured `auto_pack.target_metric` (`vram` or `sm`) is close to its target threshold

- packing is opt-in per job via `packing.eligible: true` and a stable `packing.signature`

- backend compatibility is tracked per backend, so an MPS failure does not poison a stream pairing

- Linux deployments can enable `concurrent_groups_enabled: true` with `concurrent_backend_allowlist: ["mps", "stream"]` to overlap an MPS group and a stream group on the same GPU

- raw MLEvolve snippet execution is conservative; without an explicit runtime-probe hook, jobs stay exclusive-only for runtime-aware packing

## Tests

```bash
python -m unittest discover localml_scheduler/tests
```

New test modules (since the 2026-05-15 refactor):

- `tests/test_graph_db_validation.py` covers the Neo4j-mirrored knowledge graph

- `tests/test_hardware_features.py` covers the Qdrant-backed hardware feature store
