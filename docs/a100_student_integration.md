# Use the released A100 student on CPU

The optional `S_A100` adapter loads the published D3 TorchScript student and its
17-input graph schema. It predicts wall/GPU step milliseconds, epoch milliseconds,
NVML busy percent, and average/peak VRAM MiB. Epoch time is wall-step time multiplied
by explicit optimizer steps per epoch. NVML busy means kernel-active time, not
FLOP utilization or SM occupancy.

Use the public PerfSeer release code and extract its `S_A100` checkpoint archive.
Install that release's `portable/requirements-tested.txt` in your environment;
`pytest` is also needed for the smoke test. Configure the scheduler:

```yaml
prediction:
  mode: ml_predictor
  ml:
    student_release_path: /absolute/PerfSeer-release
    student_model_path: /absolute/extracted/PerfSeer-release-v1/checkpoints/S_A100
    source_conversion_timeout_seconds: 90
    cpu_threads: 1
```

`student_release_path` is the **code checkout** containing `portable/runtime/`.
`student_model_path` contains `student.ts`, `normalization.pt`, and `architecture.pt`.
If the second path is omitted, it defaults to
`<student_release_path>/extracted/PerfSeer-release-v1/checkpoints/S_A100`.
No private study checkout or neighboring `PerfSeer-predictor` submodule is needed.
The adapter verifies published member hashes against the release manifest, embedded input/hardware
metadata, normalization SHA-256, and architecture SHA-256. This selected artifact requires an A100
hardware identity; CPU inference does not require an installed A100.

For a Python model factory, supply this job metadata alongside the actual batch:

```python
metadata = {"perfseer_model": {
    "source_path": "/absolute/model.py",
    "entry": "build_model",
    "input_shapes": [["$batch", 4]],
    "precision": "fp32",
    "loss": "mean",
    "optimizer": "adamw",
    "dataset_size": 80,
}}
```

Source capture runs a tiny CPU forward/backward trace in a bounded child process;
it does not launch GPU training. The legacy metadata example above accepts a single tensor
output, FP32, accumulation 1, declared `mean`/`mse`/`cross_entropy` loss and
`adam`/`adamw`/`sgd` optimizer. Include constructor arguments, optimizer settings,
and the actual recipe when applicable. Use `steps_per_epoch_by_batch` instead of
`dataset_size` for a custom sampler. A bare `steps_per_epoch` applies only to the
job's resolved physical batch. Dataset-derived steps account for `drop_last` and
gradient accumulation; accumulation greater than one needs a precomputed graph.

The versioned [source contract](source-contract.md) additionally supports structured
inputs/outputs, keyword arguments, bounded integer domains and the actual
`CandidateAdapter.training_step` loss. The [README integration guide](../README.md#integrate-a-code-agent-with-the-released-predictor)
covers A10G/A10 artifacts and source-bound JSON sidecars for other frameworks.

Unsupported source recipes require a strict backward `training_graph_path` (or
`training_graph_paths` keyed by batch), `graph_batch_size`, `graph_hardware_id`,
and `graph_step_semantics: complete_optimizer_step`. The graph must match the
requested batch, accumulation and target hardware. Unsupported conversion,
missing step semantics or invalid artifacts fall back to branch profiles.

The scheduler uses student epoch cost and busy as ordering hints; measured timing
is required for ETA, batch-duration and co-location gain calculations. Busy chooses
heavier idle anchors and lighter joining jobs; remaining epoch cost breaks ties.
Rounded predicted peak memory supports admission with an empty profile store;
exact matching observations and live VRAM refine it. Complete predicted profiles
also replace per-job solo references in cooperative trials: live progress is
weighted by fixed predicted step cost. See the [training/indicator audit](predictor-indicator-policy.md).
The release does not implement
the proposed completion-aware/fair-progress policy or validate its performance.

Run the repeatable integration test from the parent of `localml_scheduler`:

```sh
CUDA_VISIBLE_DEVICES='' \
PERFSEER_RELEASE_ROOT=/absolute/PerfSeer-release \
PERFSEER_STUDENT_MODEL_DIR=/absolute/extracted/PerfSeer-release-v1/checkpoints/S_A100 \
python -m pytest localml_scheduler/tests/test_released_a100_student.py -q
```

This exercises real model loading, source conversion, units, batch-dependent
steps, caching, SQLite-backed memory observations, scheduling estimates, hardware
rejection and failure fallback. It launches no training job or GPU experiment.
An optional `PERFSEER_A100_PROBE` path enables bit-exact direct-runtime parity on
the pinned two-row, input-only A100 TRAIN probe; it is skipped when unavailable.
The tests verify compatibility, not prediction accuracy for arbitrary new models,
colocation safety, throughput, or a live deployment.
