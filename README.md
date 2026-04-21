# localml_scheduler

`localml_scheduler` is a reusable local-first ML job manager for single-machine agent workflows. V1 focuses on two practical capabilities:

- a single-GPU scheduler with priority queueing, safe-point pause/resume, persistence, and restart recovery
- a RAM-backed baseline-model cache that keeps immutable CPU-side baselines warm and serves isolated copies to worker subprocesses

It is intentionally packaged as a root-level module so it can be used by MLEvolve or detached and integrated into other agent pipelines.

## Architecture

- `schemas.py`: serializable job, checkpoint policy, progress, and cache schemas
- `scheduler/`: policy, queue, service loop, recovery, and worker supervision
- `execution/`: subprocess launcher, file-based control plane, worker entrypoint, and runner context
- `checkpointing/`: atomic local checkpoint save/load
- `model_cache/`: in-memory LRU baseline cache plus a local socket server for worker access
- `storage/`: SQLite-backed jobs, commands, checkpoints, cache metadata, and event history
- `observability/`: JSONL events, log files, and aggregate reports
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
```

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

The normal pause flow is:

1. scheduler requests pause
2. worker reaches next safe point
3. checkpoint is saved atomically
4. worker exits cleanly
5. scheduler later redispatches the paused job from checkpoint

## Limitations In V1

- exactly one active training worker owns the GPU slot at a time
- no co-running, interference prediction, MPS packing, or distributed scheduling yet
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
